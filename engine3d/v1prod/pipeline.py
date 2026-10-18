"""Stage orchestration.

  slam   : DA3 + OpenCV SLAM + dense geometry          -> 09_debug/work/state*
  model  : layout, objects (+ names), textured shell   -> 09_debug/work/model.pkl
  export : GLBs, 2D maps, geometry, trajectory, depth, previews, evaluation
"""

import json
import os
import pickle
import shutil
import time

import cv2
import numpy as np
import open3d as o3d

from . import state_io

STAGES = ["slam", "model", "export"]

LIMITATIONS = [
    "Scale is approximate (monocular): the scene is scaled so the camera is --camera-height above the fitted floor.",
    "Walls are modelled as Manhattan planes (x / z aligned). Curved or angled walls are not represented as walls "
    "(they stay in the measured mesh / object geometry).",
    "Wall / floor / ceiling appearance comes from the video; surfaces the video never saw are filled "
    "(floor: synthesised tile pattern, walls: median wall colour, ceiling: smoothed) and reported as inferred.",
    "Objects are the measured geometry: sides the camera never saw (backs, undersides) are open or missing.",
    "Closed doors are found by colour/shape in the wall texture (door-sized, floor-standing, different from the wall); "
    "a door with the same colour as the wall is not detected. Swing direction in the plan is a convention.",
    "Windows are only detected when the video sees through them (curtained / reflective windows stay wall texture).",
    "Object names come from Grounding DINO when its environment is available; otherwise objects are numbered only.",
]


def log(msg):
    print(msg, flush=True)


def _frame_weight(st):
    return np.clip(np.array([q.get("blur_rel", 1.0) for q in st["quality"]]), 0.3, 1.5) ** 2


def _bake_frames(st, cap=160):
    u = list(st["usable"])
    if len(u) <= cap:
        return u
    return [u[i] for i in np.linspace(0, len(u) - 1, cap).round().astype(int)]


def exposure_compensate(st, depth_model, n=60000):
    """Per-frame RGB gain so every frame matches the multi-view consensus colour (TSDF
    vertex colours) -> no light / dark bands where frame contributions meet on walls."""
    V = np.asarray(st["model"].vertices)
    C = np.asarray(st["model"].vertex_colors) * 255
    sel = np.random.default_rng(0).choice(len(V), min(n, len(V)), replace=False)
    V, C = V[sel], C[sel]
    frames, K, w2c = st["frames"], st["K_full"], st["w2c_model"]
    H, W = frames[0].shape[:2]
    obs = []                                        # per frame: (radius^2, frame rgb, consensus rgb)
    for k in range(len(frames)):
        Xc = V @ w2c[k][:3, :3].T + w2c[k][:3, 3]
        z = Xc[:, 2]
        ok = z > 0.2
        u = np.round(K[k][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + K[k][0, 2]).astype(int)
        v = np.round(K[k][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + K[k][1, 2]).astype(int)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.flatnonzero(ok)
        vis = idx[np.abs(depth_model[k][v[idx], u[idx]] - z[idx]) < 0.03]
        f = frames[k][v[vis], u[vis]].astype(float)
        good = (f.mean(1) > 25) & (f.mean(1) < 245)
        vis, f = vis[good], f[good]
        r2 = ((u[vis] - W / 2) ** 2 + (v[vis] - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
        obs.append((r2, f, C[vis]))
    # alternate: per-frame RGB gain <-> global radial vignetting 1 + a r^2 + b r^4
    gains = np.ones((len(frames), 3))
    vig = np.zeros(2)
    for _ in range(3):
        for k, (r2, f, c) in enumerate(obs):
            if len(f) > 300:
                fv = f * (1 + vig[0] * r2 + vig[1] * r2 ** 2)[:, None]
                gains[k] = np.clip(np.median(c, 0) / np.maximum(np.median(fv, 0), 1), 0.75, 1.33)
        R2, Y = [], []
        for k, (r2, f, c) in enumerate(obs):
            if len(f) > 300:
                R2.append(r2)
                Y.append(c.mean(1) / np.maximum((f * gains[k]).mean(1), 1) - 1)
        if not R2:
            break
        R2, Y = np.concatenate(R2), np.concatenate(Y)
        bins = np.clip((R2 * 20).astype(int), 0, 19)
        xb = np.array([R2[bins == b].mean() for b in range(20) if (bins == b).sum() > 200])
        yb = np.array([np.median(Y[bins == b]) for b in range(20) if (bins == b).sum() > 200])
        if len(xb) >= 4:
            A = np.c_[xb, xb ** 2]
            vig = np.clip(np.linalg.lstsq(A, yb - yb[0] * 0, rcond=None)[0], -0.5, 0.8)
    vs, us = np.mgrid[0:H, 0:W]
    r2img = ((us - W / 2) ** 2 + (vs - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    vmap = (1 + vig[0] * r2img + vig[1] * r2img ** 2).astype(np.float32)[..., None]
    out = [np.clip(frames[k].astype(np.float32) * vmap * gains[k], 0, 255).astype(np.uint8)
           for k in range(len(frames))]
    log(f"  vignetting model 1 {vig[0]:+.3f} r^2 {vig[1]:+.3f} r^4 (corner gain {1 + vig.sum():.2f})")
    return out, gains


def _wall_colour(model, lay):
    """Median colour of the measured wall surface above furniture height."""
    model.compute_vertex_normals()
    V, C, N = np.asarray(model.vertices), np.asarray(model.vertex_colors), np.asarray(model.vertex_normals)
    m = np.zeros(len(V), bool)
    for w in lay["walls"]:
        a = 0 if w["axis"] == "x" else 2
        m |= (np.abs(V[:, a] - w["offset"]) < 0.03) & (np.abs(N[:, a]) > 0.9) & (V[:, 1] > 1.2)
    if m.sum() < 100:
        return np.array([200.0, 200.0, 200.0])
    return np.median(C[m], 0) * 255


def stage_model(cfg, st, work):
    from . import layout, objects, render, shell
    t0 = time.time()
    log("\n[7] Structural layout (walls, openings, floor region, ceiling)")
    w2c, K, frames = st["w2c_model"], st["K_full"], st["frames"]
    cams = np.linalg.inv(w2c)[:, :3, 3]
    lay = layout.extract(st["model"], st["planes"], cams, cfg)
    log("\n[8] Objects (measured geometry separated from the structure)")
    objs = objects.extract(st["model"], lay, st["depth_up"], st["valid_up"], K, w2c, st["usable"])
    for o in objs:
        lo_, hi_ = np.array(o["bbox_min"]), np.array(o["bbox_max"])
        o["size_m"] = (hi_ - lo_).round(3).tolist()
    seg = inst = None
    if cfg.labels != "off":
        from . import instances, labels
        res = labels.detect(cfg, st, work)
        if res is not None and os.path.exists(f"{work}/masks.npz"):
            # semantic 3D instances from Grounding DINO + SAM masks (splits touching furniture)
            inst, seg = instances.segment(objs, st, f"{work}/detections.json", f"{work}/masks.npz")
            objs = instances.split_objects(objs, inst, seg)
            for o in objs:
                if "instance" in o:
                    objects.trim_to_body(o)
            others = [o for o in objs if "instance" not in o]
            if others:
                labels.name_objects(cfg, st, others, work)
        elif res is not None:
            labels.name_objects(cfg, st, objs, work)
    objs, junk = objects.remove_junk(objs)
    if junk:
        from collections import Counter
        log("  removed reconstruction artefacts: " +
            ", ".join(f"{n} {k}" for k, n in Counter(o["removed_reason"] for o in junk).items()))
    log("\n[9] Textured structural shell (walls / floor / ceiling from the video)")
    H, W = frames[0].shape[:2]
    depth_model = render.mesh_depths(st["model"], K, w2c, W, H)
    frames_c, gains = exposure_compensate(st, depth_model)
    log(f"  exposure compensation: per-frame gain {gains.min():.2f}..{gains.max():.2f}")
    objs, geo_doors, removed = objects.classify_structural_objects(objs, lay, _wall_colour(st["model"], lay))
    for o in removed:
        log(f"  removed {o['size_m']}: {o['removed_reason']}")
    for o in objs:
        if o.get("merged_parts"):
            log(f"  {o['id']} ({o['label']}): merged back detached flat part(s) {o['merged_parts']}")
    for d in geo_doors:
        log(f"  closed door (leaf geometry) on {d['wall']}: {d['width_m']:.2f} m wide")
    blocked = objects.footprint_mask([o for o in objs if not o.get("is_door")], lay["grid"], min_h=0.0, max_h=1.2)
    fdepth = np.where(st["valid_up"], st["depth_up"], 0).astype(np.float32)
    sh = shell.build(lay, blocked, frames_c, K, w2c, depth_model, _bake_frames(st), res=cfg.texture_res,
                     frame_weight=_frame_weight(st), fdepth=fdepth, vdepth=st["depth_vis"])
    objs, n_dup = objects.remove_wall_duplicates(objs, lay, sh["textures"], cfg.texture_res, shell.wall_frame)
    if n_dup:
        log(f"  removed {n_dup} duplicated wall / curtain sheet(s) from objects")
    # doors from leaf geometry, unless the texture already found a door at the same place
    for d in geo_doors:
        dup = False
        for f in sh["features"]:
            if f["wall"] != d["wall"]:
                continue
            ax = 1 if abs(f["xz0"][0] - f["xz1"][0]) < 1e-6 else 0
            a0, a1 = sorted([f["xz0"][ax], f["xz1"][ax]])
            b0, b1 = sorted([d["xz0"][ax], d["xz1"][ax]])
            if min(a1, b1) - max(a0, b0) > 0.3:
                dup = True
        if not dup:
            sh["features"].append(d)
    if getattr(cfg, "furniture", "off") not in ("off", "auto") and seg is not None:   # Hunyuan3D not shipped
        log("\n[9b] Furniture completion (Hunyuan3D-2mini, fitted to the measured objects, video texture)")
        from . import furniture
        objs, decisions = furniture.complete(objs, st, seg, inst, work, cfg)
        sh["furniture_decisions"] = decisions
    if seg is not None:
        n0 = len(objs)
        objs, notes = objects.finalize_objects(objs, lay)
        log(f"  object clean-up: {n0} -> {len(objs)} objects ({', '.join(f'{v} {k}' for k, v in notes.items())})")
    log(f"  model stage {time.time() - t0:.0f}s")
    return {"lay": lay, "objects": objs, "shell": sh}


def save_model(m, work):
    objs = []
    for o in m["objects"]:
        o3d.io.write_triangle_mesh(f"{work}/obj_{o['id']}.ply", o["mesh"])
        objs.append({k: v for k, v in o.items() if k not in ("mesh", "tm")})
    with open(f"{work}/model.pkl", "wb") as f:
        pickle.dump({"lay": m["lay"], "objects": objs, "shell": m["shell"]}, f, protocol=4)


def load_model(work):
    with open(f"{work}/model.pkl", "rb") as f:
        m = pickle.load(f)
    for o in m["objects"]:
        o["mesh"] = o3d.io.read_triangle_mesh(f"{work}/obj_{o['id']}.ply")
        o["mesh"].compute_vertex_normals()
    return m


def stage_export(cfg, st, m, out, t_start):
    from . import evaluate, exports, maps2d
    lay, objs, sh = m["lay"], m["objects"], m["shell"]
    # loops = revisits >= 5 s apart (also applied to states saved before this rule)
    gap = max(15, int(round(5.0 * st["video_fps"] / st["interval"])))
    st["loops"] = [lp for lp in st["loops"] if lp["j"] - lp["i"] >= gap]
    st["loops_full"] = [lp for lp in st["loops_full"] if lp["j"] - lp["i"] >= gap]
    st["stats"]["loops"] = len(st["loops"])
    dirs = {k: f"{out}/{k}" for k in ("01_clean_model", "02_slam_model", "03_floor_plan", "04_geometry",
                                      "05_trajectory", "06_depth", "07_previews", "08_evaluation", "09_debug")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    c2w = np.linalg.inv(st["w2c_model"])
    cams = c2w[:, :3, 3]
    raw_cams = np.linalg.inv(st["w2c_raw_model"])[:, :3, 3]

    log("\n[10] 2D maps: floor plan, occupancy, navigation (A*)")
    rooms = maps2d.segment_rooms(lay, sh["features"])
    maps2d.draw_floor_plan(f"{dirs['03_floor_plan']}/floor_plan", lay, objs, sh["features"], rooms,
                           cam_xz=cams[:, [0, 2]], title=f"Floor plan - {cfg.name}")
    og = maps2d.nav_grid(lay, objs, cams, cfg.grid_resolution, obstacle_max_h=min(2.0, lay["ceiling_h"] - 0.25))
    pts = np.asarray(st["pcd"].points)
    raw_og = maps2d.raw_occupancy(pts, cams, cfg.grid_resolution, obstacle_max_h=min(2.0, lay["ceiling_h"] - 0.25))
    low_conf = [k for k, s in enumerate(st["pose_state"]) if s == "LOW-CONF"]
    planner, path, plen, cl, path_world = maps2d.navigation_images(
        dirs["03_floor_plan"], og, raw_og, cams, raw_cams, cfg.robot_radius, st["loops"], low_conf)
    os.replace(f"{dirs['03_floor_plan']}/raw_occupancy_pointcloud.png", f"{dirs['09_debug']}/raw_occupancy_pointcloud.png")
    layout_json = {
        "units": "approx. metres, model frame (x right, y up, z back, floor y = 0); plan up = -z",
        "ceiling_height_m": round(lay["ceiling_h"], 3),
        "walls": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()} for w in lay["walls"]],
        "walked_passages": lay.get("passages", []),
        "doors_and_openings": sh["features"],
        "rooms": [{k: v for k, v in r.items() if k != "mask"} for r in rooms],
        "objects": [{k: v for k, v in o.items() if k not in ("mesh", "tm")} for o in objs],
    }
    with open(f"{dirs['03_floor_plan']}/layout.json", "w") as f:
        json.dump(layout_json, f, indent=2, default=float)
    log(f"  rooms {[(r['name'], r['size_m'], r['area_m2']) for r in rooms]}")
    log(f"  doors/openings {[(f['type'], f['wall'], f['width_m']) for f in sh['features']]}")
    log(f"  A* {plen:.2f} m, min clearance {cl[0]:.2f} m")

    log("\n[11] 3D exports: clean GLB, SLAM GLB, geometry, trajectory, depth")
    exports.prepare_object_meshes(objs)
    # context fill: measured surfaces wherever the clean model would render empty (black)
    from . import context
    import trimesh as _tm
    base = [g for g in exports.clean_scene(sh, objs, True, context=False).dump() if isinstance(g, _tm.Trimesh)]
    ctx = context.fill_gaps(base, st["model"], st)
    sh["context"] = ctx
    # the context fill is for views from INSIDE (full model, walk-through): in the open-top
    # dollhouse view those grazing-angle measured surfaces are only clutter -> left out
    sh["context_open"] = None
    exports.export_glbs(out, sh, objs, st, path_world)
    exports.export_geometry(out, st)
    exports.export_trajectory(out, st, path_world)
    exports.export_depth(out, st, make_video=not cfg.no_video)
    for n, img in sh["textures"].items():
        os.makedirs(f"{dirs['09_debug']}/textures", exist_ok=True)
        cv2.imwrite(f"{dirs['09_debug']}/textures/{n}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    log("\n[12] Previews")
    import trimesh
    clean_full = [g for g in exports.clean_scene(sh, objs, True).dump() if isinstance(g, trimesh.Trimesh)]
    clean_open = [g for g in exports.clean_scene(sh, objs, False).dump() if isinstance(g, trimesh.Trimesh)]
    shots, top = exports.dollhouse_views(clean_open, lay, labels=["clean model (open top)"] * 4)
    cv2.imwrite(f"{dirs['07_previews']}/clean_model_overview.jpg",
                cv2.cvtColor(cv2.resize(np.vstack([np.hstack(shots[:2]), np.hstack(shots[2:])]), None, fx=0.75,
                                        fy=0.75, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"{dirs['07_previews']}/clean_model_top.jpg", cv2.cvtColor(top, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    L = exports.slam_layers(st, path_world, st["keyframes"])
    shots, top = exports.dollhouse_views(clean_open + list(L.values()), lay, labels=["SLAM model (open top)"] * 4)
    cv2.imwrite(f"{dirs['07_previews']}/slam_model_overview.jpg",
                cv2.cvtColor(cv2.resize(np.vstack([np.hstack(shots[:2]), np.hstack(shots[2:])]), None, fx=0.75,
                                        fy=0.75, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(f"{dirs['07_previews']}/slam_model_top.jpg", cv2.cvtColor(top, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    exports.walkthrough(dirs["07_previews"], st, clean_full, make_video=not cfg.no_video)
    exports.object_gallery(f"{dirs['07_previews']}/objects_gallery.jpg", objs)
    exports.structure_visualisation(f"{dirs['07_previews']}/structure_segmentation.jpg", sh, objs, lay)
    if not cfg.no_video:
        exports.orbit_video(f"{dirs['07_previews']}/orbit_clean_model.mp4", clean_open, lay)
        diag_video(st, og, planner, path, plen, cl, f"{dirs['07_previews']}/slam_diagnostic.mp4", cfg)

    log("\n[13] Evaluation")
    raw_tm = exports.o3d_to_tm(st["model"])
    raw_unreg = st["raw_model"]
    wp = evaluate.wall_planarity(raw_unreg, lay["walls"])
    walls_ev = [{"id": w["id"], "length_m": w["t1"] - w["t0"], **wp[w["id"]],
                 "seen_pct": sh["stats"][w["id"]]["seen_pct"]} for w in lay["walls"]]
    phot = evaluate.photometric(st, clean_full, raw_tm)
    fid = evaluate.fidelity(st["model"], clean_full)
    rc, rs = evaluate.components([raw_tm])
    cc, cs = evaluate.components(clean_full)
    gn = lambda nm: nm["isolated_cells"] + nm["tiny_blobs"]
    ev = {"video": os.path.abspath(cfg.video), "model": cfg.model, "process_res": cfg.process_res,
          "slam": st["stats"],
          "layout": {"ceiling_h_m": lay["ceiling_h"], "walls": len(lay["walls"]),
                     "doors": sum(f["type"] == "door" for f in sh["features"]),
                     "openings": sum(f["type"] != "door" for f in sh["features"]),
                     "passages": len(lay.get("passages", [])),
                     "rooms": [{k: v for k, v in r.items() if k != "mask"} for r in rooms],
                     "floor_m2": float(lay["floor"].sum() * lay["grid"].cell ** 2),
                     "surfaces": sh["stats"]},
          "walls": walls_ev, "photometric": phot, "fidelity": fid,
          "cleanliness": {"raw_components": rc, "raw_small_fragments": rs, "raw_triangles": len(raw_tm.faces),
                          "clean_components": cc, "clean_small_fragments": cs,
                          "clean_triangles": int(sum(len(g.faces) for g in clean_full))},
          "objects": {"count": len(objs), "labels": [o["label"] for o in objs]},
          "maps": {"raw_noise": gn(raw_og.noise_raw), "clean_noise": gn(og.noise), "astar_length_m": plen,
                   "astar_min_clearance_m": cl[0] if path else float("nan")},
          "warnings": quality_warnings(lay, sh, objs, st, rooms),
          "limitations": LIMITATIONS,
          "timing_min": (time.time() - t_start) / 60}
    txt = evaluate.write_report(dirs["08_evaluation"], ev)
    log(txt)
    write_index(out, cfg)


def quality_warnings(lay, sh, objs, st, rooms):
    """Plain-language flags so a weak result is never presented as a good one."""
    w = []
    if lay.get("degenerate"):
        w.append(f"Not a room walkthrough: the camera moved {lay.get('walked_path_m', 0):.1f} m and enclosed no "
                 "floor area; walls / rooms could not be established (partial model of the visible scene only).")
    elif lay.get("walked_path_m", 99) < 3:
        w.append(f"Short walk ({lay['walked_path_m']:.1f} m): coverage of the rooms is limited.")
    rendered = [x for x in lay["walls"] if x.get("rendered", True)]
    if len(rendered) < 3:
        w.append(f"Only {len(rendered)} walls with visual evidence: the room outline is incomplete "
                 "(parts of the room were never filmed).")
    seen = [sh["stats"][x["id"]]["seen_pct"] for x in rendered if x["id"] in sh["stats"]]
    if seen and np.mean(seen) < 60:
        w.append(f"Walls are on average only {np.mean(seen):.0f}% observed: much of the wall texture is inferred.")
    fl = sh["stats"].get("floor", {})
    if fl.get("seen_pct_of_region", 100) < 45:
        w.append(f"Floor only {fl.get('seen_pct_of_region', 0):.0f}% observed: most of it is synthesised.")
    bad = sum(s_ == "LOW-CONF" for s_ in st["pose_state"])
    if bad > 0.25 * len(st["pose_state"]):
        w.append(f"{bad} of {len(st['pose_state'])} camera poses are low-confidence (fast motion / blank walls): "
                 "expect local misalignment.")
    if not rooms:
        w.append("No room could be segmented from the floor plan.")
    return w


def diag_video(st, og, planner, path, plen, cl, path_out, cfg):
    from types import SimpleNamespace
    from .maps2d import model_ground
    from .slam.obstacles import ObstacleDetector
    from .slam.viz import write_diagnostic_video
    gm = model_ground()
    c2w = np.linalg.inv(st["w2c_model"])
    ctx = SimpleNamespace(
        frames_full=st["frames"], depth=st["depth_proc"], valid=st["valid_proc"], K_proc=st["K_proc"],
        w2c_ref=st["w2c_model"], raw_centers=np.linalg.inv(st["w2c_raw_model"])[:, :3, 3],
        ref_centers=c2w[:, :3, 3], tracks=st["tracks"], lm_by_id={lm["id"]: lm for lm in st["lms"]},
        ver=st["ver"], pose_state=st["pose_state"], quality=st["quality"], keyframes=set(st["keyframes"]),
        kf_reasons=st["kf_reasons"], loops=st["loops_full"], grid=og, planner=planner, path=path,
        path_len=plen, clearance=cl, ground=gm, detector=ObstacleDetector(gm, max_h=2.0),
        frame_rmse=st["frame_rmse"], n_feats=st["n_feats"])
    write_diagnostic_video(ctx, path_out, max(1, int(round(cfg.fps))))


INDEX = """# Output index - {name}

Produced by Version 1 Production ({version}) from `{video}`.
All 3D data: approx. metres, y up, floor at y = 0, walls aligned to x / z.

| Folder / file | What it is |
|---|---|
| **01_clean_model/room_clean.glb** | **Clean reconstruction**: straight textured walls, floor, ceiling + measured furniture / objects. No SLAM elements |
| 01_clean_model/room_clean_open_top.glb | Same without ceiling (dollhouse view; walls facing away are culled in most viewers) |
| 01_clean_model/structure_only.glb, objects_only.glb | Structure (walls / floor / ceiling) and objects separately |
| 01_clean_model/objects/*.ply | Every object as its own mesh |
| **02_slam_model/room_slam.glb** | **Clean reconstruction + SLAM**: refined (cyan) + raw DA3 (grey) trajectory, camera poses coloured by pose state (green valid, blue refined, red low-confidence, grey unchecked), verified landmarks (green), loop links (magenta), A* path (yellow), start (green) / end (red) |
| 02_slam_model/room_slam_open_top.glb, slam_layers_only.glb | Dollhouse version; SLAM layers alone |
| **03_floor_plan/floor_plan.png / .svg** | **2D floor plan**: walls, doors, openings / doorways / passages, windows, rooms with sizes, numbered objects |
| 03_floor_plan/occupancy_map.png | Clean occupancy grid + raw / refined trajectory, low-confidence poses, loops |
| 03_floor_plan/navigation_map.png | Clearance field + safe A* path (robot radius) |
| 03_floor_plan/layout.json | Machine-readable layout: walls, doors, openings, rooms, objects |
| 04_geometry/mesh_fused_regularised.ply | Full fused TSDF mesh (plane-regularised, vertex colours) |
| 04_geometry/point_cloud_dense.ply | Filtered dense point cloud (keyframes) |
| 04_geometry/landmarks_verified.ply | Persistent verified landmarks |
| 05_trajectory/trajectory_*_tum.txt, *.ply | Refined and raw camera trajectories (TUM format + line sets) |
| 05_trajectory/cameras.json, poses_per_frame.csv | Intrinsics + camera-to-model pose per frame; per-frame SLAM log |
| 06_depth/depth_maps.npz, depth_preview.mp4, depth_examples.jpg | DA3 depth (raw + filtered) and previews |
| 07_previews/ | Overviews, top views, model vs video, walkthrough / orbit / SLAM diagnostic videos, object gallery, structural segmentation |
| 08_evaluation/evaluation.md / .json | Quantitative validation (structure, fidelity, photometric, SLAM, maps) |
| 09_debug/ | Textures, raw occupancy, intermediate state (`work/`, lets `--from-stage` resume) |
| run_config.json, run_log.txt | Exact settings and full log of this run |
"""


def write_index(out, cfg):
    with open(f"{out}/README_OUTPUTS.md", "w", encoding="utf-8") as f:
        f.write(INDEX.format(name=cfg.name, version=cfg.version, video=os.path.abspath(cfg.video)))


def run(cfg, out, work, start=0):
    from . import stage_slam
    t = time.time()
    if start == 0:
        st = stage_slam.run(cfg, work)
        state_io.save(st, work)
        log(f"  stage 'slam' saved ({time.time() - t:.0f}s)")
    else:
        st = state_io.load(work)
        log(f"  stage 'slam' state loaded")
    if start <= 1:
        m = stage_model(cfg, st, work)
        save_model(m, work)
    else:
        m = load_model(work)
        log("  stage 'model' loaded")
    stage_export(cfg, st, m, out, t)
