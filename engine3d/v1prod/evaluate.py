"""Quantitative validation of the production result.

* Structure  - per wall: RMS distance of the raw (un-snapped) reconstruction to the
               fitted wall plane (how flat the real surface measured), support area,
               texture coverage; corner angles are 90 deg by construction (Manhattan)
* Fidelity   - clean model vs raw fused mesh: completeness (% of the raw surface within
               5 cm of the clean model) and accuracy (% of the clean model within 5 cm of
               the raw surface; the rest is inferred: floor under furniture, ceiling)
* Photometric- clean model and raw mesh rendered from the real video cameras (exact
               intrinsics) vs the video frames: PSNR / SSIM on pixels both cover
* Cleanliness- connected components / floating fragments, raw vs clean
* SLAM       - pose checks, refinement, landmarks + reprojection, loops, smoothness
* Maps       - occupancy noise raw point-cloud grid vs clean layout grid
"""

import json

import cv2
import numpy as np
import open3d as o3d
import trimesh

from . import render


def _ssim(a, b, mask):
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    m = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    return float(s[m].mean()) if m.any() else float("nan")


def photometric(st, geoms_clean, raw_tm, step=5):
    frames, K, w2c = st["frames"], st["K_full"], st["w2c_model"]
    H, W = frames[0].shape[:2]
    Rc = render.RayRenderer(geoms_clean)
    Rr = render.RayRenderer([raw_tm])
    keys = ("psnr", "psnr_exposure_matched", "ssim", "coverage")
    res = {"clean": {k: [] for k in keys}, "raw": {k: [] for k in keys}}
    for k in range(0, len(frames), step):
        gt = cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY)
        ic, _, mc = Rc.render(K[k], w2c[k], W, H)
        ir, _, mr = Rr.render(K[k], w2c[k], W, H)
        both = mc & mr
        for name, img, m in (("clean", ic, mc), ("raw", ir, mr)):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if both.any():
                a, b = g[both].astype(float), gt[both].astype(float)
                mse = np.mean((a - b) ** 2)
                s = (a * b).sum() / max((a * a).sum(), 1e-6)       # per-image exposure gain
                mse_e = np.mean((np.clip(a * s, 0, 255) - b) ** 2)
            else:
                mse = mse_e = np.nan
            res[name]["psnr"].append(10 * np.log10(255 ** 2 / max(mse, 1e-6)))
            res[name]["psnr_exposure_matched"].append(10 * np.log10(255 ** 2 / max(mse_e, 1e-6)))
            res[name]["ssim"].append(_ssim(g, gt, both))
            res[name]["coverage"].append(float(m.mean()))
    return {n: {k: float(np.nanmean(v)) for k, v in d.items()} for n, d in res.items()}


def fidelity(raw_o3d, clean_tms, n=300_000, tol=0.05):
    cm = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in clean_tms])
    c = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cm.vertices), o3d.utility.Vector3iVector(cm.faces))
    a = raw_o3d.sample_points_uniformly(n)
    b = c.sample_points_uniformly(n)
    d_ab = np.asarray(a.compute_point_cloud_distance(b))
    d_ba = np.asarray(b.compute_point_cloud_distance(a))
    return {"completeness_pct_raw_within_5cm_of_clean": float(np.mean(d_ab < tol) * 100),
            "accuracy_pct_clean_within_5cm_of_raw": float(np.mean(d_ba < tol) * 100),
            "median_raw_to_clean_cm": float(np.median(d_ab) * 100),
            "median_clean_to_raw_cm": float(np.median(d_ba) * 100)}


def components(tm_list):
    out = 0
    small = 0
    for g in tm_list:
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(g.vertices), o3d.utility.Vector3iVector(g.faces))
        ids, counts, _ = m.cluster_connected_triangles()
        counts = np.asarray(counts)
        out += len(counts)
        small += int((counts < 200).sum())
    return out, small


def wall_planarity(raw_model, walls, band=0.10):
    raw_model.compute_vertex_normals()
    V, N = np.asarray(raw_model.vertices), np.asarray(raw_model.vertex_normals)
    out = {}
    for w in walls:
        a = 0 if w["axis"] == "x" else 2
        o = 2 - a
        d = V[:, a] - w["offset"]
        m = (np.abs(d) < band) & (N[:, a] * w["sign"] > 0.8) & (V[:, o] > w["t0"]) & (V[:, o] < w["t1"]) \
            & (V[:, 1] > 0.1)
        out[w["id"]] = {"rms_cm": float(np.sqrt(np.mean(d[m] ** 2)) * 100) if m.sum() > 50 else float("nan"),
                        "p90_abs_cm": float(np.percentile(np.abs(d[m]), 90) * 100) if m.sum() > 50 else float("nan"),
                        "support_vertices": int(m.sum())}
    return out


def write_report(out_dir, ev):
    with open(f"{out_dir}/evaluation.json", "w") as f:
        json.dump(ev, f, indent=2, default=float)
    s, L = ev["slam"], []
    L.append("# Version 1 Production - evaluation\n")
    L.append(f"Video: `{ev['video']}` | frames {s['frames']} (used {s['frames_used']}) | "
             f"model {ev['model']} @ {ev['process_res']} px\n")
    if ev.get("warnings"):
        L.append("## WARNINGS\n")
        L.extend(f"- **{x}**" for x in ev["warnings"])
        L.append("")
    else:
        L.append("No quality warnings (walk length, wall evidence, floor coverage and pose confidence within the "
                 "expected range).\n")
    L.append("## Structure\n")
    lay = ev["layout"]
    L.append(f"- Ceiling height **{lay['ceiling_h_m']:.2f} m**, {lay['walls']} wall segments, "
             f"{lay['doors']} doors, {lay['openings']} openings/doorways, {lay['passages']} walked-through passages")
    for r in lay["rooms"]:
        L.append(f"- {r['name']}: {r['size_m'][0]:.2f} x {r['size_m'][1]:.2f} m, {r['area_m2']:.1f} m2")
    L.append("\n| wall | length (m) | raw surface RMS to plane (cm) | 90th pct (cm) | texture from video |")
    L.append("|---|---:|---:|---:|---:|")
    for w in ev["walls"]:
        L.append(f"| {w['id']} | {w['length_m']:.2f} | {w['rms_cm']:.1f} | {w['p90_abs_cm']:.1f} | {w['seen_pct']:.0f}% |")
    L.append("\nCorners are exact 90 deg (Manhattan snapping); the raw RMS shows how flat the measured wall "
             "was before it was replaced by the plane.\n")
    L.append("## Clean model vs raw reconstruction\n")
    f = ev["fidelity"]
    p = ev["photometric"]
    L.append("| metric | raw fused mesh | clean model |")
    L.append("|---|---:|---:|")
    L.append(f"| PSNR vs video frames (dB, same cameras) | {p['raw']['psnr']:.2f} | {p['clean']['psnr']:.2f} |")
    L.append(f"| PSNR, exposure-matched (dB) | {p['raw']['psnr_exposure_matched']:.2f} | "
             f"{p['clean']['psnr_exposure_matched']:.2f} |")
    L.append(f"| SSIM vs video frames | {p['raw']['ssim']:.3f} | {p['clean']['ssim']:.3f} |")
    L.append(f"| image coverage | {p['raw']['coverage'] * 100:.0f}% | {p['clean']['coverage'] * 100:.0f}% |")
    c = ev["cleanliness"]
    L.append(f"| connected components | {c['raw_components']:,} | {c['clean_components']:,} |")
    L.append(f"| fragments < 200 triangles | {c['raw_small_fragments']:,} | {c['clean_small_fragments']:,} |")
    L.append(f"| triangles | {c['raw_triangles']:,} | {c['clean_triangles']:,} |")
    L.append(f"\n- Completeness: {f['completeness_pct_raw_within_5cm_of_clean']:.1f}% of the raw surface lies within "
             f"5 cm of the clean model (median {f['median_raw_to_clean_cm']:.1f} cm)")
    L.append(f"- Accuracy: {f['accuracy_pct_clean_within_5cm_of_raw']:.1f}% of the clean model lies within 5 cm of "
             f"measured geometry (median {f['median_clean_to_raw_cm']:.1f} cm); the remainder is inferred surface "
             "(floor under furniture, rarely filmed ceiling, wall areas never seen)")
    L.append(f"- Objects: {ev['objects']['count']} separated ({', '.join(ev['objects']['labels'])})\n")
    L.append("## SLAM\n")
    L.append(f"- Pose checks (OpenCV SIFT + MAGSAC essential matrix vs DA3): {s['pair_checks']}")
    L.append(f"- Pose states: {s['pose_states']}")
    ag0, ag1 = s["opencv_agreement_raw"], s["opencv_agreement_refined"]
    L.append(f"- OpenCV agreement rotation error: raw {ag0['rot_err_mean']:.2f} deg -> refined {ag1['rot_err_mean']:.2f} deg; "
             f"translation direction: raw {ag0['trans_err_median']:.1f} deg -> refined {ag1['trans_err_median']:.1f} deg")
    sm0, sm1 = s["smoothness_raw"], s["smoothness_refined"]
    L.append(f"- Trajectory smoothness (mean accel): raw {sm0['pos_accel_mean']:.3f} m -> refined {sm1['pos_accel_mean']:.3f} m")
    rp = s["reprojection"]
    L.append(f"- Verified landmarks {s['landmarks']:,} | reprojection mean {rp['mean']:.2f} px, median {rp['median']:.2f} px, "
             f"RMSE {rp['rmse']:.2f} px")
    L.append(f"- Loop candidates verified: {s['loops']}")
    al = s.get("alignment") or []
    if al:
        L.append(f"- Frame-to-model alignment: median depth disagreement {al[0]['median_depth_residual_m'] * 100:.1f} cm "
                 f"-> {al[-1]['median_depth_residual_m'] * 100:.1f} cm")
    m = ev["maps"]
    L.append("\n## Maps\n")
    L.append(f"- Raw point-cloud occupancy noise: {m['raw_noise']} | clean layout map noise: {m['clean_noise']}")
    L.append(f"- A* start -> end: {m['astar_length_m']:.2f} m, min clearance {m['astar_min_clearance_m']:.2f} m")
    L.append("\n## Honest limitations\n")
    L.extend(f"- {x}" for x in ev["limitations"])
    L.append(f"\nTiming: {ev['timing_min']:.1f} min total.\n")
    with open(f"{out_dir}/evaluation.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return "\n".join(L)
