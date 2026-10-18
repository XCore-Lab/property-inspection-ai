"""3D exports (clean / SLAM GLBs, geometry, trajectory, cameras, depth) and previews."""

import json
import os

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation

from . import render
from .slam.refine import LOW_CONF, REFINED, UNCHECKED

STATE_RGBA = {"VALID": (0, 200, 0, 255), REFINED: (0, 120, 255, 255),
              LOW_CONF: (255, 0, 0, 255), UNCHECKED: (160, 160, 160, 255)}


def log(msg):
    print(msg, flush=True)


def o3d_to_tm(m):
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                           vertex_colors=(np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8),
                           process=False)


def _colored(m, rgba):
    m.visual.vertex_colors = np.tile(np.array(rgba, np.uint8), (len(m.vertices), 1))
    return m


def tube(P, r, rgba, spheres=True):
    parts = [trimesh.creation.cylinder(radius=r, segment=[a, b], sections=10)
             for a, b in zip(P[:-1], P[1:]) if np.linalg.norm(b - a) > 1e-6]
    if spheres:
        parts += [trimesh.creation.icosphere(1, radius=r * 1.2).apply_translation(p) for p in P]
    return _colored(trimesh.util.concatenate(parts), rgba)


def frustum(c2w, s, rgba):
    R, t = c2w[:3, :3], c2w[:3, 3]
    c = np.array([[-s, -.6 * s, 1.2 * s], [s, -.6 * s, 1.2 * s], [s, .6 * s, 1.2 * s], [-s, .6 * s, 1.2 * s]])
    pts = [t] + [R @ q + t for q in c]
    segs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return _colored(trimesh.util.concatenate(
        [trimesh.creation.cylinder(radius=s * .06, segment=[pts[a], pts[b]], sections=5) for a, b in segs]), rgba)


# ----------------------------------------------------------------------
# scenes
# ----------------------------------------------------------------------

def object_node_name(o):
    return f"{o['id']}_{o['label'].replace(' ', '_')}"


def prepare_object_meshes(objects):
    """o['tm'] = the object as exported: the completed (generated + fitted + video-textured)
    mesh where completion was accepted, else the measured mesh; o['tm_measured'] always."""
    for o in objects:
        o["tm_measured"] = o3d_to_tm(o["mesh"])
        o["tm"] = o["tm_measured"]
        p = o.get("generated_glb")
        if p and os.path.exists(p):
            try:
                o["tm"] = trimesh.load(p, force="mesh")
            except Exception:
                pass


def clean_scene(shell, objects, ceiling=True, measured=False, context=True):
    sc = trimesh.Scene()
    for n, m in shell["meshes"].items():
        if n == "ceiling" and not ceiling:
            continue
        sc.add_geometry(m, node_name=f"structure_{n}", geom_name=f"structure_{n}")
    ctx = shell.get("context" if ceiling else "context_open")
    if context and ctx is not None and len(ctx.faces):
        sc.add_geometry(ctx, node_name="context_measured_surfaces", geom_name="context_measured_surfaces")
    for o in objects:
        if not ceiling and o.get("ceiling_mounted"):
            continue
        g = o["tm_measured"] if measured else o["tm"]
        sc.add_geometry(g, node_name=object_node_name(o), geom_name=object_node_name(o))
    return sc


def slam_layers(st, path_world, keyframes):
    """SLAM visualisation parts (model frame) as {name: trimesh}."""
    c2w = np.linalg.inv(st["w2c_model"])
    c2w_raw = np.linalg.inv(st["w2c_raw_model"])
    ref, raw = c2w[:, :3, 3], c2w_raw[:, :3, 3]
    L = {"slam_trajectory_da3_raw": tube(raw, 0.010, (150, 150, 150, 255), spheres=False),
         "slam_trajectory_refined": tube(ref, 0.018, (0, 180, 255, 255)),
         "slam_start": _colored(trimesh.creation.icosphere(2, radius=0.07).apply_translation(ref[0]), (0, 220, 0, 255)),
         "slam_end": _colored(trimesh.creation.icosphere(2, radius=0.07).apply_translation(ref[-1]), (230, 0, 0, 255))}
    fr = []
    for k in sorted(set(keyframes) | {k for k, s in enumerate(st["pose_state"]) if s == LOW_CONF}):
        fr.append(frustum(c2w[k], 0.09, STATE_RGBA.get(st["pose_state"][k], (160, 160, 160, 255))))
    if fr:
        L["slam_camera_poses"] = trimesh.util.concatenate(fr)
    X = st["landmarks_X"]
    if len(X):
        base = trimesh.creation.icosphere(0, radius=0.012)
        V = (base.vertices[None] + X[:, None]).reshape(-1, 3)
        F = (base.faces[None] + np.arange(len(X))[:, None, None] * len(base.vertices)).reshape(-1, 3)
        L["slam_landmarks"] = _colored(trimesh.Trimesh(V, F, process=False), (30, 255, 120, 255))
    loops = [tube(np.stack([ref[lp["i"]], ref[lp["j"]]]), 0.008, (255, 0, 255, 255), spheres=False)
             for lp in st["loops"]]
    if loops:
        L["slam_loop_links"] = trimesh.util.concatenate(loops)
    if path_world is not None and len(path_world) > 1:
        L["slam_astar_path"] = tube(path_world, 0.02, (255, 210, 0, 255), spheres=False)
    return L


def export_glbs(out, shell, objects, st, path_world):
    d1, d2 = f"{out}/01_clean_model", f"{out}/02_slam_model"
    os.makedirs(f"{d1}/objects", exist_ok=True)
    for f in os.listdir(f"{d1}/objects"):                  # no stale objects from earlier runs
        os.remove(f"{d1}/objects/{f}")
    os.makedirs(d2, exist_ok=True)
    if "tm" not in (objects[0] if objects else {}):
        prepare_object_meshes(objects)
    clean_scene(shell, objects, True).export(f"{d1}/room_clean.glb")
    clean_scene(shell, objects, False).export(f"{d1}/room_clean_open_top.glb")
    if any(o.get("generated_glb") for o in objects):
        # the same model with the purely measured furniture (for comparison / validation)
        clean_scene(shell, objects, True, measured=True).export(f"{d1}/room_clean_measured_furniture.glb")
    s = trimesh.Scene()
    for n, m in shell["meshes"].items():
        s.add_geometry(m, node_name=f"structure_{n}", geom_name=f"structure_{n}")
    s.export(f"{d1}/structure_only.glb")
    s = trimesh.Scene()
    for o in objects:
        s.add_geometry(o["tm"], node_name=object_node_name(o), geom_name=object_node_name(o))
        o3d.io.write_triangle_mesh(f"{d1}/objects/{object_node_name(o)}_measured.ply", o["mesh"])
        if o.get("generated_glb"):
            o["tm"].export(f"{d1}/objects/{object_node_name(o)}_completed.glb")
    if objects:
        s.export(f"{d1}/objects_only.glb")
    L = slam_layers(st, path_world, st["keyframes"])
    for ceiling, name in ((True, "room_slam.glb"), (False, "room_slam_open_top.glb")):
        sc = clean_scene(shell, objects, ceiling)
        for n, g in L.items():
            sc.add_geometry(g, node_name=n, geom_name=n)
        sc.export(f"{d2}/{name}")
    s = trimesh.Scene()
    for n, g in L.items():
        s.add_geometry(g, node_name=n, geom_name=n)
    s.export(f"{d2}/slam_layers_only.glb")
    return L


# ----------------------------------------------------------------------
# geometry, trajectory, cameras, depth
# ----------------------------------------------------------------------

def export_geometry(out, st):
    d = f"{out}/04_geometry"
    os.makedirs(d, exist_ok=True)
    o3d.io.write_triangle_mesh(f"{d}/mesh_fused_regularised.ply", st["model"])
    o3d.io.write_point_cloud(f"{d}/point_cloud_dense.ply", st["pcd"])
    if len(st["landmarks_X"]):
        lp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(st["landmarks_X"]))
        lp.paint_uniform_color([0.1, 0.9, 0.4])
        o3d.io.write_point_cloud(f"{d}/landmarks_verified.ply", lp)


def export_trajectory(out, st, path_world):
    d = f"{out}/05_trajectory"
    os.makedirs(d, exist_ok=True)
    c2w = np.linalg.inv(st["w2c_model"])
    c2w_raw = np.linalg.inv(st["w2c_raw_model"])
    vf = st["video_fps"]
    for name, T in (("trajectory_refined", c2w), ("trajectory_da3_raw", c2w_raw)):
        q = Rotation.from_matrix(T[:, :3, :3]).as_quat()           # x y z w
        with open(f"{d}/{name}_tum.txt", "w") as f:
            f.write("# TUM format: timestamp tx ty tz qx qy qz qw  (camera-to-model, OpenCV camera axes; "
                    "model: x right, y up, z back, floor y=0, approx. metres)\n")
            for k in range(len(T)):
                ts = st["sample_idx"][k] / vf
                f.write(f"{ts:.4f} {T[k,0,3]:.5f} {T[k,1,3]:.5f} {T[k,2,3]:.5f} "
                        f"{q[k,0]:.6f} {q[k,1]:.6f} {q[k,2]:.6f} {q[k,3]:.6f}\n")
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(T[:, :3, 3]),
                                  o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(T) - 1)]))
        ls.paint_uniform_color([0, 0.7, 1] if "refined" in name else [0.6, 0.6, 0.6])
        o3d.io.write_line_set(f"{d}/{name}.ply", ls)
    with open(f"{d}/poses_per_frame.csv", "w") as f:
        f.write("frame,video_frame,time_s,x,y,z,pose_state,pair_check,inlier_ratio,frame_quality,keyframe,"
                "pos_correction_m,rot_correction_deg,reprojection_rmse_px\n")
        kfs = set(st["keyframes"])
        for k in range(len(c2w)):
            v = st["ver"][k]
            f.write(f"{k},{st['sample_idx'][k]},{st['sample_idx'][k] / vf:.3f},{c2w[k,0,3]:.4f},{c2w[k,1,3]:.4f},"
                    f"{c2w[k,2,3]:.4f},{st['pose_state'][k]},{v['status'] if v else '-'},"
                    f"{(v['inlier_ratio'] if v else float('nan')):.3f},{st['quality'][k]['quality']},"
                    f"{int(k in kfs)},{st['pos_correction'][k]:.4f},{st['rot_correction'][k]:.3f},"
                    f"{st['frame_rmse'][k]:.3f}\n")
    cams = {"convention": "c2w = camera-to-model 4x4; OpenCV camera (x right, y down, z forward); "
                          "model frame x right, y up, z back, floor y = 0, approx. metres",
            "image_size": [int(st["frames"][0].shape[1]), int(st["frames"][0].shape[0])],
            "frames": [{"frame": k, "video_frame": int(st["sample_idx"][k]), "K": st["K_full"][k].round(4).tolist(),
                        "c2w": c2w[k].round(6).tolist(), "pose_state": st["pose_state"][k]}
                       for k in range(len(c2w))]}
    with open(f"{d}/cameras.json", "w") as f:
        json.dump(cams, f)
    if path_world is not None:
        np.savetxt(f"{d}/astar_path_3d.txt", path_world, fmt="%.4f", header="x y z (model frame, 5 cm above floor)")


def depth_color(d, lo=None, hi=None):
    v = d > 0
    if not v.any():
        return np.zeros(d.shape + (3,), np.uint8)
    inv = np.zeros_like(d)
    inv[v] = 1 / d[v]
    lo = np.percentile(inv[v], 2) if lo is None else lo
    hi = np.percentile(inv[v], 98) if hi is None else hi
    n = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~v] = 0
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def export_depth(out, st, make_video=True):
    d = f"{out}/06_depth"
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/depth_maps.npz", depth_raw_m=st["depth_vis"].astype(np.float16),
                        depth_filtered_m=np.where(st["valid_proc"], st["depth_proc"], 0).astype(np.float16),
                        K_processed=st["K_proc"], frame_index=np.array(st["sample_idx"]),
                        note="depth in approx. metres at DA3 processed resolution; filtered = DA3 confidence "
                             "cut + OpenCV flying-pixel / bilateral / island filtering")
    if not make_video:
        return
    H, W = st["frames"][0].shape[:2]
    vw = cv2.VideoWriter(f"{d}/depth_preview.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (W * 3 // 2, H // 2 * 1))
    dv = st["depth_vis"]
    inv_all = 1 / dv[dv > 0]
    lo, hi = np.percentile(inv_all, 2), np.percentile(inv_all, 98)
    stills = []
    for k in range(len(st["frames"])):
        a = cv2.resize(st["frames"][k], (W // 2, H // 2))
        b = cv2.resize(depth_color(st["depth_vis"][k], lo, hi), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(depth_color(np.where(st["valid_up"][k], st["depth_up"][k], 0), lo, hi), (W // 2, H // 2),
                       interpolation=cv2.INTER_NEAREST)
        f = np.hstack([a, b, c])
        for x, s in ((6, "video"), (W // 2 + 6, "DA3 depth (raw)"), (W + 6, "filtered + aligned (used for 3D)")):
            cv2.putText(f, s, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if k in (len(st["frames"]) // 4, len(st["frames"]) // 2, 3 * len(st["frames"]) // 4):
            stills.append(f)
    vw.release()
    cv2.imwrite(f"{d}/depth_examples.jpg", cv2.cvtColor(np.vstack(stills), cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------
# previews
# ----------------------------------------------------------------------

def _label(img, s, org=(14, 30), scale=0.7):
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def room_view_params(lay):
    g = lay["grid"]
    rr, cc = np.nonzero(lay["enclosed"] if lay["enclosed"].any() else lay["floor"])
    x0, x1 = g.lo[0] + cc.min() * g.cell, g.lo[0] + (cc.max() + 1) * g.cell
    z0, z1 = g.lo[1] + rr.min() * g.cell, g.lo[1] + (rr.max() + 1) * g.cell
    ctr = np.array([(x0 + x1) / 2, 0.35, (z0 + z1) / 2])
    span = max(x1 - x0, z1 - z0)
    return ctr, span


def dollhouse_views(geoms, lay, W=1280, H=800, labels=None, cull=True):
    R = render.RayRenderer(geoms, cull_backfaces=cull)
    ctr, span = room_view_params(lay)
    K = render.intrinsics(W, H, W * 0.8)
    shots = []
    for i, ang in enumerate((45, 135, 225, 315)):
        a = np.radians(ang)
        eye = ctr + np.array([np.cos(a) * span * 0.75, span * 0.95, np.sin(a) * span * 0.75])
        img, _, _ = R.render(K, render.look_at(eye, ctr), W, H)
        shots.append(_label(img, labels[i] if labels else f"view {i + 1}"))
    top, _, _ = R.render(K, render.look_at(ctr + [0, span * 1.7, 0.001], ctr, up=(0, 0, -1)), W, H)
    return shots, _label(top, "top view (up = -z)")


def orbit_video(path, geoms, lay, W=1280, H=720, n=120, fps=24):
    R = render.RayRenderer(geoms, cull_backfaces=True)
    ctr, span = room_view_params(lay)
    K = render.intrinsics(W, H, W * 0.75)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        a = 2 * np.pi * i / n + np.pi / 4
        eye = ctr + np.array([np.cos(a) * span * 0.8, span * 0.9, np.sin(a) * span * 0.8])
        img, _, _ = R.render(K, render.look_at(eye, ctr), W, H)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()


def walkthrough(out_dir, st, clean_geoms, make_video=True):
    """Video frame | clean model rendered from the SAME camera (exact intrinsics)."""
    frames, K, w2c = st["frames"], st["K_full"], st["w2c_model"]
    H, W = frames[0].shape[:2]
    R = render.RayRenderer(clean_geoms)
    N = len(frames)
    pick = sorted({int(round(x)) for x in np.linspace(N * 0.08, N * 0.92, 5)})
    rows = []
    vw = cv2.VideoWriter(f"{out_dir}/walkthrough_video_vs_model.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5,
                         (W * 2, H)) if make_video else None
    for k in range(N):
        if vw is None and k not in pick:
            continue
        img, _, _ = R.render(K[k], w2c[k], W, H, bg=(0, 0, 0))
        f = np.hstack([frames[k], img])
        _label(f, f"video frame {st['sample_idx'][k]}", (12, 28), 0.6)
        _label(f, "clean 3D model, same camera", (W + 12, 28), 0.6)
        if vw is not None:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        if k in pick:
            rows.append(f)
    if vw is not None:
        vw.release()
    cv2.imwrite(f"{out_dir}/model_vs_video.jpg",
                cv2.cvtColor(cv2.resize(np.vstack(rows), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def object_gallery(path, objects, S=320, max_items=24):
    rows = []
    order = sorted(objects, key=lambda o: (not o.get("generated_glb"), "instance" not in o, -o["triangles"]))
    for o in order[:max_items]:
        g = o["tm"]
        V = g.vertices
        ctr = (V.min(0) + V.max(0)) / 2
        r = max(np.linalg.norm(np.ptp(V, 0)) / 2, 0.1)
        R = render.RayRenderer([g])
        row = []
        for ang in (30, 120, 210, 300):
            a = np.radians(ang)
            eye = ctr + np.array([np.cos(a) * r * 2.3, r * 1.2, np.sin(a) * r * 2.3])
            img, _, _ = R.render(render.intrinsics(S, S, S * 1.05), render.look_at(eye, ctr), S, S, bg=(40, 40, 44))
            row.append(img)
        row = np.hstack(row)
        s = o["size_m"]
        kind = "COMPLETED" if o.get("generated_glb") else "measured"
        _label(row, f"{o['id']}  {o['label']}  [{kind}]  {s[0]:.2f} x {s[2]:.2f} x {s[1]:.2f} m (w x d x h)", (10, 24), 0.55)
        rows.append(row)
    if rows:
        cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def structure_visualisation(path, shell, objects, lay):
    """Walls / floor / ceiling in flat colours, objects coloured by instance: shows the
    structural segmentation of the model."""
    rng = np.random.default_rng(3)
    geoms = []
    pal = {"floor": (205, 200, 190), "ceiling": (230, 230, 230)}
    for n, m in shell["meshes"].items():
        if n == "ceiling":
            continue
        g = trimesh.Trimesh(m.vertices, m.faces, process=False)
        col = pal.get(n, (120 + 40 * (hash(n) % 3), 140, 170))
        g.visual.vertex_colors = np.tile(np.array(col + (255,), np.uint8), (len(g.vertices), 1))
        geoms.append(g)
    for o in objects:
        if o.get("ceiling_mounted"):
            continue
        g = trimesh.Trimesh(o["tm"].vertices, o["tm"].faces, process=False)
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        g.visual.vertex_colors = np.tile(np.array(col + (255,), np.uint8), (len(g.vertices), 1))
        geoms.append(g)
    shots, top = dollhouse_views(geoms, lay, 960, 600)
    R = render.RayRenderer(geoms, cull_backfaces=True)
    sheet = np.vstack([np.hstack([shots[0], shots[2]]), np.hstack([top, shots[1]])])
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
