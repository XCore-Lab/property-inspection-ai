"""Geometric obstacle detection on DA3 depth (carried over from Version 1).

Each pixel is lifted to 3D and classified by height above the fitted floor:
floor = walkable, 0.15 m .. max_h = obstacle. Distance is horizontal ground
distance; the danger level is taken inside the walking corridor ahead.
"""

import cv2
import numpy as np



def visualize_depth(depth, percentile=2):
    """Inverse-depth colour map (OpenCV TURBO; replaces DA3's matplotlib-based helper). RGB uint8."""
    d = np.asarray(depth, np.float32).copy()
    valid = d > 0
    inv = np.zeros_like(d)
    inv[valid] = 1.0 / d[valid]
    if valid.sum() > 10:
        lo, hi = np.percentile(inv[valid], percentile), np.percentile(inv[valid], 100 - percentile)
    else:
        lo, hi = 0.0, 1.0
    n = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = cv2.cvtColor(cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb

from .geometry import c2w_of, pixel_grid

DANGER_RGB = {"NEAR": (255, 40, 40), "MEDIUM": (255, 150, 0),
              "FAR": (255, 230, 0), "CLEAR": (60, 220, 60)}


class ObstacleDetector:
    NONE, FAR, MEDIUM, NEAR = 0, 1, 2, 3

    def __init__(self, ground, near=1.0, medium=2.0, far=4.0, min_h=0.15,
                 max_h=2.0, floor_h=0.08, corridor=0.4):
        self.g = ground
        self.near, self.medium, self.far = near, medium, far
        self.min_h, self.max_h, self.floor_h, self.corridor = min_h, max_h, floor_h, corridor

    def detect(self, depth, K, w2c, valid):
        H, W = depth.shape
        c2w = c2w_of(w2c)
        Xc = (np.linalg.inv(K) @ pixel_grid(H, W)) * depth.reshape(1, -1)
        Xw = (c2w[:3, :3] @ Xc + c2w[:3, 3:4]).T
        h = self.g.height(Xw).reshape(H, W)
        rel = self.g.to_2d(Xw) - self.g.to_2d(c2w[:3, 3][None])[0]
        fwd = self.g.dir_2d(c2w[:3, 2])
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        right = np.array([fwd[1], -fwd[0]])
        ahead, lateral = (rel @ fwd).reshape(H, W), (rel @ right).reshape(H, W)
        dist = np.linalg.norm(rel, axis=1).reshape(H, W)

        floor = valid & (np.abs(h) < self.floor_h)
        obst = valid & (h > self.min_h) & (h < self.max_h)
        zones = np.zeros((H, W), np.uint8)
        zones[obst & (dist < self.far)] = self.FAR
        zones[obst & (dist < self.medium)] = self.MEDIUM
        zones[obst & (dist < self.near)] = self.NEAR
        corr = obst & (ahead > 0) & (np.abs(lateral) < self.corridor)
        dmin = float(np.percentile(dist[corr], 2)) if corr.any() else float("inf")
        level = ("NEAR" if dmin < self.near else "MEDIUM" if dmin < self.medium
                 else "FAR" if dmin < self.far else "CLEAR")
        return zones, floor, {"min_distance": dmin, "level": level}

    def render(self, depth, K, w2c, valid, alpha=0.55):
        zones, floor, info = self.detect(depth, K, w2c, valid)
        base = visualize_depth(depth)
        gray = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        vis = cv2.addWeighted(base, 0.35, gray, 0.65, 0)
        ov = vis.copy()
        ov[floor] = (40, 200, 90)
        ov[zones == self.FAR] = (255, 230, 0)
        ov[zones == self.MEDIUM] = (255, 130, 0)
        ov[zones == self.NEAR] = (255, 20, 20)
        vis = cv2.addWeighted(vis, 1 - alpha, ov, alpha, 0)
        return vis, info
