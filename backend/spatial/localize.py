"""Place 2D inspection findings in the 3D room: box -> depth -> SLAM pose -> model point,
then snap to the nearest sampled surface (wall / floor / ceiling) for labelling."""
from __future__ import annotations

import numpy as np

from backend.spatial.coverage import Samples
from backend.spatial.slam_data import SlamRun


def box_to_world(run: SlamRun, frame: int, box: tuple[float, float, float, float],
                 image_size: tuple[int, int]) -> tuple[np.ndarray | None, float | None]:
    """box in pixels of an image of `image_size` (= the video frame). Uses the median of
    valid depth over the central 40 % of the box (robust to box edges / background)."""
    W, H = image_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(2.0, (x2 - x1) * 0.2), max(2.0, (y2 - y1) * 0.2)
    dh, dw = run.depth.shape[1:]
    sx, sy = dw / W, dh / H
    u0, u1 = int(max(0, (cx - hw) * sx)), int(min(dw, (cx + hw) * sx + 1))
    v0, v1 = int(max(0, (cy - hh) * sy)), int(min(dh, (cy + hh) * sy + 1))
    patch = run.depth[frame][v0:v1, u0:u1]
    vals = patch[patch > 0]
    if vals.size == 0:   # no depth at the box centre: whole box, then the whole frame, then 2 m
        bu0, bu1 = int(max(0, x1 * sx)), int(min(dw, x2 * sx + 1))
        bv0, bv1 = int(max(0, y1 * sy)), int(min(dh, y2 * sy + 1))
        box_patch = run.depth[frame][bv0:bv1, bu0:bu1]
        vals = box_patch[box_patch > 0]
    if vals.size == 0:
        frame_d = run.depth[frame]
        vals = frame_d[frame_d > 0]
    d = float(np.median(vals)) if vals.size else 2.0
    Kd = run.K_depth[frame]
    u, v = cx * sx, cy * sy
    pc = np.array([(u - Kd[0, 2]) / Kd[0, 0] * d, (v - Kd[1, 2]) / Kd[1, 1] * d, d, 1.0])
    return (run.c2w[frame] @ pc)[:3], d


def snap_to_surface(p: np.ndarray, samples: Samples, max_dist: float = 0.6) -> tuple[str, np.ndarray, float]:
    d = np.linalg.norm(samples.xyz - p, axis=1)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return "", p, float(d[i])
    return samples.group[i], samples.xyz[i], float(d[i])
