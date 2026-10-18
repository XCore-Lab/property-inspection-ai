"""Section 5 - OpenCV-assisted depth quality filtering (before 3D fusion).

Order per frame (DA3 processed resolution):
 1. DA3 confidence mask
 2. Flying-pixel removal: pixels on strong depth discontinuities
    (local max/min depth ratio from dilate/erode)
 3. (optional, off by default) unsupported depth edges: depth edges with no
    nearby Canny image edge. Ablation showed it deletes real white-on-white
    wall corners (no image edge there), so it is disabled by default.
 4. Edge-aware smoothing: joint bilateral filter guided by the RGB image
    (cv2.ximgproc, contrib); changes > 3 % of depth are rejected so object
    boundaries are never smeared
 5. Morphological opening of the valid mask + removal of small connected
    components (isolated depth islands)
"""

import cv2
import numpy as np


def filter_depth(depth, rgb, conf_mask, jump_ratio=0.08, edge_ratio=0.04,
                 remove_unsupported_edges=False, morph_cleanup=True,
                 min_component_frac=0.002, max_smooth_change=0.03):
    d = depth.astype(np.float32)
    valid = conf_mask & np.isfinite(d) & (d > 0)
    stats = {"conf_removed": 1 - valid.mean()}

    k3 = np.ones((3, 3), np.uint8)
    d_safe = np.where(valid, d, np.nan)
    fill = np.nan_to_num(d_safe, nan=float(np.nanmedian(d_safe)))
    dil, ero = cv2.dilate(fill, k3), cv2.erode(fill, k3)
    rel_jump = (dil - ero) / np.maximum(fill, 1e-6)

    flying = valid & (rel_jump > jump_ratio)
    valid &= ~flying
    stats["flying_removed"] = flying.mean()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.dilate(cv2.Canny(gray, 50, 150), k3, iterations=2) > 0
    depth_edges = rel_jump > edge_ratio
    unsupported = cv2.dilate((depth_edges & ~img_edges).astype(np.uint8), k3) > 0
    unsupported &= valid
    if not remove_unsupported_edges:
        unsupported[:] = False
    valid &= ~unsupported
    stats["edge_removed"] = unsupported.mean()

    out = d.copy()
    if hasattr(cv2, "ximgproc"):
        sm = cv2.ximgproc.jointBilateralFilter(rgb.astype(np.float32), fill, d=7,
                                               sigmaColor=20, sigmaSpace=3)
    else:
        sm = cv2.bilateralFilter(fill, 7, float(np.median(fill) * 0.05), 3)
    ok = valid & (np.abs(sm - d) / np.maximum(d, 1e-6) < max_smooth_change)
    out[ok] = sm[ok]
    stats["smoothed"] = ok.mean()

    if not morph_cleanup:
        stats["island_removed"] = 0.0
        stats["kept"] = valid.mean()
        return out, valid, stats
    opened = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, k3)
    n, labels, cc, _ = cv2.connectedComponentsWithStats(opened, connectivity=4)
    small = cc[:, cv2.CC_STAT_AREA] < min_component_frac * valid.size
    small[0] = False
    keep = (opened > 0) & ~small[labels]
    stats["island_removed"] = (valid & ~keep).mean()
    valid = keep
    stats["kept"] = valid.mean()
    return out, valid, stats
