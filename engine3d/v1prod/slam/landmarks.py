"""Sections 2 + 4 - Persistent 3D landmarks and reprojection quality.

Every LK track becomes a landmark candidate:
  * each observation is back-projected with DA3 depth -> per-view 3D points
  * landmark = median of those; depth consistency = spread across views
  * multi-view triangulation (cv2.triangulatePoints) from the widest-baseline
    pair of observations, compared with the DA3 position
  * reprojection error per observation via cv2.projectPoints
Rejection: short tracks, inconsistent depth, poor triangulation agreement,
high reprojection error.
"""

import cv2
import numpy as np

from .geometry import c2w_of


def _bilinear(img, u, v):
    H, W = img.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0, v0 = int(u), int(v)
    a, b = u - u0, v - v0
    return ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
            + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])


def reprojection_errors(X, obs, K_full, w2c):
    """Pixel error of world point X against each (k, u, v) observation."""
    errs = []
    for k, u, v in obs:
        T = w2c[k]
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        proj, _ = cv2.projectPoints(X.reshape(1, 1, 3), rvec, T[:3, 3], K_full[k], None)
        errs.append(np.linalg.norm(proj.ravel() - (u, v)))
    return np.array(errs)


def build_landmarks(tracks, depth, valid, K_full, w2c, sx, sy, min_len=3,
                    max_depth_spread=0.10, max_tri_disagree=0.15,
                    min_tri_angle=2.0, max_reproj=4.0, filter_=True):
    """Returns list of landmark dicts and summary stats.

    With filter_=False every track with >= 2 observations is kept (baseline).
    """
    lms = []
    rejected = {"short": 0, "depth": 0, "triangulation": 0, "reprojection": 0, "no_depth": 0}
    for tid, obs in tracks.obs.items():
        if len(obs) < (min_len if filter_ else 2):
            rejected["short"] += 1
            continue
        pts3, dvals = [], []
        for k, u, v in obs:
            up, vp = u / sx, v / sy
            iu, iv = int(round(min(up, depth.shape[2] - 1))), int(round(min(vp, depth.shape[1] - 1)))
            if not valid[k][iv, iu]:
                continue
            d = _bilinear(depth[k], up, vp)
            if not np.isfinite(d) or d <= 0:
                continue
            Xc = np.linalg.inv(K_full[k]) @ np.array([u, v, 1.0]) * d
            c2w = c2w_of(w2c[k])
            pts3.append(c2w[:3, :3] @ Xc + c2w[:3, 3])
            dvals.append(d)
        if len(pts3) < (2 if filter_ else 1):
            rejected["no_depth"] += 1
            continue
        pts3 = np.array(pts3)
        X = np.median(pts3, axis=0)
        mean_d = float(np.mean(dvals))
        spread = float(np.median(np.linalg.norm(pts3 - X, axis=1)) / mean_d)
        if filter_ and spread > max_depth_spread:
            rejected["depth"] += 1
            continue

        # Triangulate from the widest-baseline pair of observations
        tri_err, tri_angle = np.nan, 0.0
        a, b = obs[0], obs[-1]
        Ca, Cb = c2w_of(w2c[a[0]])[:3, 3], c2w_of(w2c[b[0]])[:3, 3]
        ra, rb = X - Ca, X - Cb
        tri_angle = float(np.degrees(np.arccos(np.clip(
            ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12), -1, 1))))
        if tri_angle >= min_tri_angle:
            Pa = K_full[a[0]] @ w2c[a[0]][:3]
            Pb = K_full[b[0]] @ w2c[b[0]][:3]
            Xh = cv2.triangulatePoints(Pa, Pb, np.array([[a[1]], [a[2]]]),
                                       np.array([[b[1]], [b[2]]]))
            Xt = (Xh[:3] / Xh[3]).ravel()
            in_front = (w2c[a[0]][:3, :3] @ Xt + w2c[a[0]][:3, 3])[2] > 0
            tri_err = (float(np.linalg.norm(Xt - X) / mean_d) if in_front else np.inf)
            if filter_ and tri_err > max_tri_disagree:
                rejected["triangulation"] += 1
                continue

        errs = reprojection_errors(X, obs, K_full, w2c)
        if filter_ and np.mean(errs) > max_reproj:
            rejected["reprojection"] += 1
            continue
        lms.append({"id": tid, "X": X, "obs": obs, "errs": errs, "spread": spread,
                    "tri_err": tri_err, "tri_angle": tri_angle})
    return lms, rejected


def reprojection_stats(lms, thresholds=(1.0, 2.0, 5.0)):
    if not lms:
        return {"mean": np.nan, "median": np.nan, "rmse": np.nan, "n_obs": 0}
    e = np.concatenate([lm["errs"] for lm in lms])
    out = {"mean": float(e.mean()), "median": float(np.median(e)),
           "rmse": float(np.sqrt(np.mean(e ** 2))), "n_obs": int(len(e))}
    for t in thresholds:
        out[f"pct_below_{t:g}px"] = float(np.mean(e < t) * 100)
    return out


def per_frame_rmse(lms, n_frames):
    sq = [[] for _ in range(n_frames)]
    for lm in lms:
        for (k, _, _), e in zip(lm["obs"], lm["errs"]):
            sq[k].append(e * e)
    return [float(np.sqrt(np.mean(s))) if s else np.nan for s in sq]
