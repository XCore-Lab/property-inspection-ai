"""Section 8 - Loop candidate detection (experimental, detection only).

ORB descriptors per frame. For every pair of frames at least `min_gap`
samples apart, mutual ratio-test matches are counted; the strongest pairs are
verified with the same geometric check as pose verification (essential
matrix RANSAC + recoverPose) and must also have well-spread, one-to-one
inliers (bunched matches are a classic false-loop pattern).

Accepted loops are only reported / visualised (no pose-graph optimisation).
For each one the DA3 relative pose between the two (temporally distant)
frames is compared with OpenCV's: a small rotation error means DA3's
trajectory is self-consistent over that long gap (low drift).
"""

import cv2
import numpy as np

from .geometry import c2w_of
from .verify import ratio_match, verify_pair


def detect_loops(orb_feats, K_full, w2c, img_shape, min_gap=15, min_matches=30,
                 min_inliers=20, min_inlier_ratio=0.4, min_coverage=0.3,
                 min_unique=0.8, top_per_frame=2, ratio=0.8):
    N = len(orb_feats)
    cands = []
    for j in range(N):
        scores = []
        for i in range(0, j - min_gap + 1):
            m = ratio_match(orb_feats[i][1], orb_feats[j][1], cv2.NORM_HAMMING, ratio)
            if len(m) >= min_matches:
                scores.append((len(m), i))
        scores.sort(reverse=True)
        for _, i in scores[:top_per_frame]:
            v = verify_pair(orb_feats[i], orb_feats[j], K_full[i], K_full[j], w2c[i], w2c[j],
                            cv2.NORM_HAMMING, img_shape=img_shape, ratio=ratio)
            inl = v["inlier_mask"]
            n_in = int(inl.sum())
            if n_in:
                uniq = min(len(np.unique(np.round(v["pts_a"][inl]), axis=0)),
                           len(np.unique(np.round(v["pts_b"][inl]), axis=0))) / n_in
            else:
                uniq = 0.0
            ok = (n_in >= min_inliers and v["inlier_ratio"] >= min_inlier_ratio
                  and v["coverage"] >= min_coverage and uniq >= min_unique)
            dist = float(np.linalg.norm(c2w_of(w2c[i])[:3, 3] - c2w_of(w2c[j])[:3, 3]))
            cands.append({"i": i, "j": j, "matches": v["matches"], "inliers": n_in,
                          "inlier_ratio": v["inlier_ratio"], "accepted": ok,
                          "da3_distance": dist, "pts_i": v["pts_a"], "pts_j": v["pts_b"],
                          "mask": inl, "coverage": v["coverage"], "unique": uniq,
                          "rot_err": v["rot_err"], "trans_err": v["trans_err"],
                          "R_cv": v["R_cv"]})
    # keep the strongest accepted loop per neighbourhood of frames
    acc = sorted([c for c in cands if c["accepted"]], key=lambda c: -c["inliers"])
    chosen = []
    for c in acc:
        if all(abs(c["i"] - o["i"]) > 3 or abs(c["j"] - o["j"]) > 3 for o in chosen):
            chosen.append(c)
    return chosen, cands


def draw_loop(img_i, img_j, c, max_lines=80):
    h = min(img_i.shape[0], img_j.shape[0])
    a = cv2.resize(img_i, (int(img_i.shape[1] * h / img_i.shape[0]), h))
    b = cv2.resize(img_j, (int(img_j.shape[1] * h / img_j.shape[0]), h))
    sa, sb = h / img_i.shape[0], h / img_j.shape[0]
    canvas = np.hstack([a, b])
    for t in np.flatnonzero(c["mask"])[:max_lines]:
        p = tuple(int(x) for x in c["pts_i"][t] * sa)
        q = (int(c["pts_j"][t][0] * sb + a.shape[1]), int(c["pts_j"][t][1] * sb))
        cv2.line(canvas, p, q, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, (0, 255, 255), -1)
        cv2.circle(canvas, q, 3, (0, 255, 255), -1)
    lines = [f"LOOP CANDIDATE frame {c['i']+1} <-> {c['j']+1} | inliers {c['inliers']}/{c['matches']} "
             f"({c['inlier_ratio']*100:.0f}%) | spread {c['coverage']*100:.0f}%",
             f"Drift check: DA3 vs chained OpenCV rotation {c.get('chain_err', float('nan')):.2f} deg",
             f"(direct wide-baseline loop pose vs DA3: {c['rot_err']:.2f} deg - low precision)"]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 16 + 28 * len(lines)), (0, 0, 0), -1)
    for n, t in enumerate(lines):
        cv2.putText(canvas, t, (10, 28 + 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
