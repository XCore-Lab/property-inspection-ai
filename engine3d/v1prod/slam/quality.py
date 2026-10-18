"""Section 7 - Frame quality assessment (OpenCV).

Per frame: edge sharpness (blur), brightness / clipped exposure, feature
count, feature distribution over a 4x4 grid, plus optical-flow survival and
RANSAC inlier ratio once tracking/verification have run.

Blur: plain Laplacian variance mistakes texture-less (white-wall) frames for
blurred ones. Instead, at the strongest 1% of edges, the ratio of 2nd to 1st
derivative is measured separately for x and y (min of both, so one-directional
motion blur is caught). Blur widens edges and lowers this ratio independent of
how much texture the frame has. Judged relative to the video's median.
"""

import cv2
import numpy as np

GOOD, ACCEPTABLE, POOR = "GOOD", "ACCEPTABLE", "POOR"


def edge_sharpness(gray):
    x = gray.astype(np.float32)
    out = []
    for dx, dy in ((1, 0), (0, 1)):
        d1 = np.abs(cv2.Sobel(x, cv2.CV_32F, dx, dy, ksize=3))
        d2 = np.abs(cv2.Sobel(x, cv2.CV_32F, 2 * dx, 2 * dy, ksize=3))
        m = d1 > np.percentile(d1, 99)
        out.append(float(np.median(d2[m] / (d1[m] + 1e-6))) if m.any() else 0.0)
    return min(out)


def base_metrics(gray, n_grid=4):
    blur = edge_sharpness(gray)
    brightness = float(gray.mean())
    clipped = float(np.mean((gray < 10) | (gray > 245)))
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=0.01,
                                      minDistance=7)
    n_feat = 0 if corners is None else len(corners)
    coverage = 0.0
    if n_feat:
        H, W = gray.shape
        c = corners.reshape(-1, 2)
        gx = np.clip((c[:, 0] / W * n_grid).astype(int), 0, n_grid - 1)
        gy = np.clip((c[:, 1] / H * n_grid).astype(int), 0, n_grid - 1)
        counts = np.bincount(gy * n_grid + gx, minlength=n_grid * n_grid)
        coverage = float(np.mean(counts >= 5))
    return {"blur": blur, "brightness": brightness, "clipped": clipped,
            "features": n_feat, "coverage": coverage}


def _grade(value, good, acceptable, higher_is_better=True):
    if not higher_is_better:
        value, good, acceptable = -value, -good, -acceptable
    if value >= good:
        return 2
    if value >= acceptable:
        return 1
    return 0


def score_frames(metrics, flow_survival=None, inlier_ratio=None):
    """Adds 'grades' and 'quality' to each metrics dict (in place)."""
    med_blur = np.median([m["blur"] for m in metrics]) or 1.0
    for i, m in enumerate(metrics):
        g = {
            "blur": _grade(m["blur"] / med_blur, 0.6, 0.4),
            "features": _grade(m["features"], 500, 150),
            "coverage": _grade(m["coverage"], 0.5, 0.3),
            "exposure": _grade(m["clipped"], 0.10, 0.30, higher_is_better=False),
        }
        m["blur_rel"] = m["blur"] / med_blur
        if flow_survival is not None and flow_survival[i] is not None:
            m["flow_survival"] = flow_survival[i]
            g["flow"] = _grade(flow_survival[i], 0.7, 0.4)
        if inlier_ratio is not None and inlier_ratio[i] is not None:
            m["inlier_ratio"] = inlier_ratio[i]
            g["inliers"] = _grade(inlier_ratio[i], 0.5, 0.3)
        m["grades"] = g
        poor = sum(v == 0 for v in g.values())
        good = sum(v == 2 for v in g.values())
        if poor >= 2 or g["blur"] == 0 or g["features"] == 0:
            m["quality"] = POOR
        elif poor == 0 and good >= len(g) - 1:
            m["quality"] = GOOD
        else:
            m["quality"] = ACCEPTABLE
    return metrics
