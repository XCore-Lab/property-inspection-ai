"""Stage 1 - SLAM front end + dense geometry (DA3 + OpenCV 5).

frames -> DA3 (depth, pose, intrinsics) -> RANSAC floor + metric scale
       -> OpenCV LK tracking / SIFT+MAGSAC pose verification / trajectory refinement
       -> persistent landmarks + reprojection check, loop candidates
       -> depth filtering + joint-bilateral upsampling to full res, grazing-angle cut
       -> frame-to-model alignment (per-frame depth scale + point-to-plane ICP)
       -> TSDF (1.5 cm) + free-space carving + fragment removal
       -> model frame: gravity (floor y = 0) + Manhattan yaw (walls on x / z)
       -> Manhattan plane regularisation (walls / floor / ceiling snapped flat)

Everything returned is in the MODEL frame (x right, y up, z back, approx. metres).
"""

import time

import cv2
import numpy as np
import open3d as o3d

from .da3_infer import extract_frames, run_da3
from .slam import geometry as geo
from .slam import room
from .slam.depth_filter import filter_depth
from .slam.keyframes import reconstruction_weights, select_keyframes
from .slam.landmarks import build_landmarks, per_frame_rmse, reprojection_stats
from .slam.loops import detect_loops
from .slam.quality import base_metrics, score_frames
from .slam.refine import refine_trajectory, smoothness_metrics
from .slam.tracking import track_video
from .slam.verify import detect_all, pose_agreement, verify_sequence
from .smooth import smooth_corrections


def log(msg):
    print(msg, flush=True)


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def run(cfg, work):
    T = {}
    t0 = time.time()
    # ---------------------------------------------------------------- frames + DA3
    log("\n[1] Frames + Depth Anything 3")
    frames, sidx, vfps, interval = extract_frames(cfg.video, cfg.fps, cfg.max_frames, cfg.max_image_side)
    N = len(frames)
    Hf, Wf = frames[0].shape[:2]
    log(f"  {N} frames (1 of every {interval} frames of the {vfps:.2f} fps video), {Wf}x{Hf}")
    da3 = run_da3(frames, cfg.model, cfg.process_res, cfg.chunk_size, cfg.chunk_overlap,
                  cache=cfg.da3_cache or f"{work}/da3_prediction.npz")
    depth, conf, w2c, K_proc, images = da3["depth"], da3["conf"], da3["w2c"], da3["K"], da3["img"]
    Hp, Wp = depth.shape[1:]
    sx, sy = Wf / Wp, Hf / Hp
    K_full = np.stack([geo.scale_intrinsics(K, sx, sy) for K in K_proc])
    log(f"  {cfg.model} depth {Wp}x{Hp}, chunks {da3['info'].get('chunks', 1)}")
    T["da3"] = time.time() - t0

    # ---------------------------------------------------------------- floor + scale
    t1 = time.time()
    log("\n[2] Floor plane (RANSAC) + metric scale")
    conf_mask = np.isfinite(depth) & (depth > 0) & (conf >= np.percentile(conf, 30))
    pts0, _, _ = geo.backproject_all(depth, K_proc, w2c, images, conf_mask)
    n, d, inl = geo.estimate_floor_plane(pts0, [geo.c2w_of(Tm)[:3, :3] for Tm in w2c])
    scale = cfg.camera_height / float(np.median(geo.centers(w2c) @ n + d))
    depth = depth * scale
    w2c = w2c.copy()
    w2c[:, :3, 3] *= scale
    ground = geo.GroundFrame(n, d * scale, geo.c2w_of(w2c[0])[:3, 2])
    log(f"  floor inliers {inl*100:.0f}% | scale x{scale:.3f} (camera height {cfg.camera_height} m)")

    # ---------------------------------------------------------------- OpenCV SLAM checks
    log("\n[3] OpenCV 5: LK tracking, SIFT/MAGSAC pose verification, trajectory refinement")
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    qm = [base_metrics(g) for g in grays]
    tracks = track_video(cfg.video, sidx, K_full, size=(Wf, Hf))
    feats, norm = detect_all(grays, "sift")
    ver = verify_sequence(feats, K_full, w2c, norm, img_shape=grays[0].shape)
    quality = score_frames(qm, flow_survival=tracks.survival,
                           inlier_ratio=[None] + [v["inlier_ratio"] for v in ver[1:]])
    w2c_raw = w2c
    w2c_ref, rinfo = refine_trajectory(w2c_raw, ver)
    checks = _count(v["status"] for v in ver[1:])
    states = _count(rinfo["state"])
    log(f"  pair checks {checks} | pose states {states}")
    log(f"  corrections: pos max {rinfo['pos_correction'].max()*100:.1f} cm, "
        f"rot max {rinfo['rot_correction'].max():.2f} deg")
    weights = reconstruction_weights(quality, rinfo["state"], rinfo["jump"], mode="complete")
    usable = [k for k in range(N) if weights[k] > 0]
    kfs, kf_reasons = select_keyframes(w2c_ref, tracks, weights)
    log(f"  frames for the model {len(usable)}/{N} (blurred excluded) | keyframes {len(kfs)}")

    lms, lm_rej = build_landmarks(tracks, depth, conf_mask, K_full, w2c_ref, sx, sy)
    rs = reprojection_stats(lms)
    log(f"  verified landmarks {len(lms):,} | reprojection RMSE {rs['rmse']:.2f} px, "
        f"median {rs['median']:.2f} px")
    orb, _ = detect_all(grays, "orb", 2000)
    # a loop = revisiting a place >= 5 s later (not neighbouring frames at high fps)
    loops, _ = detect_loops(orb, K_full, w2c_ref, grays[0].shape,
                            min_gap=max(15, int(round(5.0 * vfps / interval))))
    log(f"  loop candidates verified: {len(loops)}")
    T["slam"] = time.time() - t1

    # ---------------------------------------------------------------- depth filtering
    t1 = time.time()
    log("\n[4] Depth filtering (flying pixels, joint bilateral) + full-res upsampling + grazing cut")
    finite = np.isfinite(depth) & (depth > 0)
    depth_up = np.zeros((N, Hf, Wf), np.float32)
    valid_up = np.zeros((N, Hf, Wf), bool)
    depth_f = np.zeros_like(depth)
    valid_f = np.zeros_like(conf_mask)
    for k in range(N):
        dk, vk, _ = filter_depth(depth[k], images[k], finite[k])
        depth_up[k], valid_up[k] = room.upsample_depth(dk, vk, frames[k])
        valid_up[k] &= room.grazing_mask(depth_up[k], K_full[k])
        depth_f[k], valid_f[k], _ = filter_depth(depth[k], images[k], conf_mask[k])
    log(f"  {valid_up.mean()*100:.1f}% of full-res pixels kept")

    # ---------------------------------------------------------------- TSDF
    log(f"\n[5] Frame-to-model alignment + TSDF @ {cfg.voxel*100:.1f} cm + free-space carving")

    def fuse(sc, poses, voxel=0.025):
        return geo.tsdf_fusion(depth_up * sc[:, None, None], K_full, poses, frames, usable,
                               masks=valid_up, voxel=voxel, trunc=voxel * 4, max_depth=cfg.max_depth)

    align_stats = []
    w2c_fin = w2c_ref.copy()
    scales = np.ones(N)
    if cfg.align_iters > 0:
        scales, w2c_fin, align_stats = room.align_frames_to_model(
            depth_up, valid_up, K_full, w2c_ref, usable, fuse, iters=cfg.align_iters, log=log)
        depth_up *= scales[:, None, None]
        # keep the low-frequency part of the per-frame ICP corrections (real drift fixes),
        # drop the per-frame jitter: sigma = 0.15 s of video
        sigma = 0.15 * vfps / interval
        w2c_fin = smooth_corrections(w2c_ref, w2c_fin, sigma=sigma)
        log(f"  ICP corrections temporally smoothed (sigma {sigma:.1f} frames)")
    mesh = fuse(np.ones(N), w2c_fin, voxel=cfg.voxel)
    n_tri0 = len(mesh.triangles)
    mesh, n_carved = room.carve_free_space(mesh, depth_up, valid_up, K_full, w2c_fin, usable)
    mesh, n_float = geo.remove_small_components(mesh, min_triangles=400)
    log(f"  {n_tri0:,} tris -> carved {n_carved:,} ghost verts -> {len(mesh.triangles):,} tris "
        f"({n_float:,} in floating fragments removed)")
    T["dense"] = time.time() - t1

    # ---------------------------------------------------------------- model frame + planes
    log("\n[6] Model frame (gravity + Manhattan) + plane regularisation")
    V = ground.align(np.asarray(mesh.vertices))
    F = np.asarray(mesh.triangles)
    C = np.asarray(mesh.vertex_colors)
    m_al = room.make_mesh(V, F, C).filter_smooth_taubin(number_of_iterations=8)
    m_al.compute_vertex_normals()
    V, Nn = np.asarray(m_al.vertices), np.asarray(m_al.vertex_normals)
    yaw = room.manhattan_yaw(V, Nn)
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    A = np.eye(4)                                   # world -> model
    A[:3, :3] = R_yaw @ ground.R_align
    A[:3, 3] = -A[:3, :3] @ ground.origin
    V_unreg = V @ R_yaw.T
    Nn = Nn @ R_yaw.T
    V, planes = room.regularise_planes(V_unreg, Nn, room.vertex_areas(V_unreg, F))
    raw_model = room.make_mesh(V_unreg, F, C)        # before plane snapping (for evaluation)
    model = room.make_mesh(V, F, C)
    log(f"  Manhattan yaw {np.degrees(yaw):.1f} deg | {len(planes)} planes")

    to_model = lambda P: np.asarray(P) @ A[:3, :3].T + A[:3, 3]
    w2c_model = np.stack([np.linalg.inv(A @ geo.c2w_of(Tm)) for Tm in w2c_fin])
    w2c_raw_model = np.stack([np.linalg.inv(A @ geo.c2w_of(Tm)) for Tm in w2c_raw])

    # dense point cloud (DA3 confidence cut + OpenCV filters, keyframes)
    pts, cols, _ = geo.backproject_all(depth_f, K_proc, w2c_fin, images, valid_f, frames=kfs)
    pcd, pc_out = geo.clean_pointcloud(pts, cols, voxel=0.01)
    pcd.points = o3d.utility.Vector3dVector(to_model(np.asarray(pcd.points)))
    lm_X = to_model(np.array([lm["X"] for lm in lms])) if lms else np.zeros((0, 3))

    sm_raw, sm_ref = smoothness_metrics(w2c_raw), smoothness_metrics(w2c_fin)
    ag_raw, ag_ref = pose_agreement(ver, K_full, w2c_raw), pose_agreement(ver, K_full, w2c_fin)
    T["total_stage1"] = time.time() - t0

    state = {
        "frames": frames, "sample_idx": sidx, "video_fps": vfps, "interval": interval,
        "K_full": K_full, "K_proc": K_proc, "images": images, "sx": sx, "sy": sy,
        "depth_proc": depth_f, "valid_proc": valid_f, "depth_up": depth_up, "valid_up": valid_up,
        # raw DA3 depth of EVERY pixel (processed res, final scale): occlusion tests for
        # texturing (thin chair legs / slats are removed from the filtered depth)
        "depth_vis": (np.where(finite, depth, 0) * scales[:, None, None]).astype(np.float32),
        "w2c_model": w2c_model, "w2c_raw_model": w2c_raw_model, "world_to_model": A,
        "pose_state": list(rinfo["state"]), "jump": list(rinfo["jump"]),
        "pos_correction": rinfo["pos_correction"], "rot_correction": rinfo["rot_correction"],
        "quality": quality, "ver": ver, "keyframes": kfs, "kf_reasons": kf_reasons, "usable": usable,
        "landmarks_X": lm_X, "landmark_stats": rs, "landmark_rejected": lm_rej, "n_feats": [len(f[0]) for f in feats],
        "frame_rmse": per_frame_rmse(lms, N), "lms": lms, "tracks": tracks,
        "loops": [{k: v for k, v in lp.items() if k in ("i", "j", "inliers", "inlier_ratio", "rot_err")} for lp in loops],
        "loops_full": loops,
        "model": model, "raw_model": raw_model, "planes": planes, "pcd": pcd, "pc_outlier_pct": pc_out,
        "stats": {"frames": N, "frames_used": len(usable), "keyframes": len(kfs), "scale": scale,
                  "floor_inliers": inl, "pair_checks": checks, "pose_states": states,
                  "alignment": align_stats, "manhattan_yaw_deg": float(np.degrees(yaw)),
                  "da3": {k: v for k, v in da3["info"].items()},
                  "smoothness_raw": sm_raw, "smoothness_refined": sm_ref,
                  "opencv_agreement_raw": ag_raw, "opencv_agreement_refined": ag_ref,
                  "landmarks": len(lms), "reprojection": rs, "loops": len(loops),
                  "pointcloud_outlier_pct": pc_out, "timing_s": T},
    }
    return state
