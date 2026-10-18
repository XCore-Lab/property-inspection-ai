"""Section 9 - Occupancy grid with OpenCV refinement + distance-field A*.

3D points -> ground projection -> evidence grid (obstacle hits vs floor hits)
-> OpenCV morphology (opening removes isolated cells, closing joins broken
walls) -> distance transform -> continuous safety cost -> A* -> safe path.
"""

import heapq

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
COLORS = {UNKNOWN: (50, 50, 50), FREE: (240, 235, 230), OCCUPIED: (60, 60, 180)}  # BGR


def noise_metrics(occ):
    """Isolated occupied cells and tiny obstacle blobs (< 4 cells)."""
    occ8 = occ.astype(np.uint8)
    nb = cv2.filter2D(occ8, -1, np.ones((3, 3), np.float32)) - occ8
    isolated = int(np.sum(occ & (nb == 0)))
    n, _, st, _ = cv2.connectedComponentsWithStats(occ8, connectivity=8)
    tiny = int(np.sum(st[1:, cv2.CC_STAT_AREA] < 4))
    return {"isolated_cells": isolated, "tiny_blobs": tiny, "components": n - 1,
            "occupied_cells": int(occ.sum())}


class OccupancyGrid:
    def __init__(self, ground, resolution=0.05, obstacle_min_h=0.15,
                 obstacle_max_h=2.0, floor_max_h=0.08, margin=0.6):
        self.g = ground
        self.res = resolution
        self.obs_min, self.obs_max, self.floor_h = obstacle_min_h, obstacle_max_h, floor_max_h
        self.margin = margin

    def fit_extent(self, pts_xy, cam_xy):
        lo = np.minimum(np.percentile(pts_xy, 1, axis=0), cam_xy.min(axis=0)) - self.margin
        hi = np.maximum(np.percentile(pts_xy, 99, axis=0), cam_xy.max(axis=0)) + self.margin
        self.lo = lo
        self.W = int(np.ceil((hi[0] - lo[0]) / self.res))
        self.H = int(np.ceil((hi[1] - lo[1]) / self.res))
        self.hits = np.zeros((self.H, self.W), np.float32)
        self.floor = np.zeros((self.H, self.W), np.float32)
        self.grid = np.zeros((self.H, self.W), np.uint8)

    def xy_to_cell(self, xy):
        xy = np.atleast_2d(xy)
        col = np.floor((xy[:, 0] - self.lo[0]) / self.res).astype(int)
        row = self.H - 1 - np.floor((xy[:, 1] - self.lo[1]) / self.res).astype(int)
        return col, row

    def cell_to_xy(self, col, row):
        x = self.lo[0] + (np.asarray(col) + 0.5) * self.res
        y = self.lo[1] + (self.H - 1 - np.asarray(row) + 0.5) * self.res
        return np.stack([x, y], axis=-1)

    def world_to_cell(self, P):
        c, r = self.xy_to_cell(self.g.to_2d(P))
        return int(np.clip(c[0], 0, self.W - 1)), int(np.clip(r[0], 0, self.H - 1))

    def _acc(self, target, xy):
        c, r = self.xy_to_cell(xy)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        np.add.at(target, (r[ok], c[ok]), 1)

    def integrate(self, points):
        h = self.g.height(points)
        xy = self.g.to_2d(points)
        self._acc(self.hits, xy[(h > self.obs_min) & (h < self.obs_max)])
        self._acc(self.floor, xy[np.abs(h) < self.floor_h])

    def finalize(self, cam_xy, min_hits=8, occ_ratio=0.3, clear_radius=0.2, morph=True):
        occ = (self.hits >= min_hits) & (self.hits >= occ_ratio * self.floor)
        self.noise_raw = noise_metrics(occ)
        if morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            occ8 = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_OPEN, k)
            occ8 = cv2.morphologyEx(occ8, cv2.MORPH_CLOSE, k)
            occ = occ8.astype(bool)
        free = (self.floor > 0) & ~occ

        walked = np.zeros((self.H, self.W), np.uint8)
        c, r = self.xy_to_cell(cam_xy)
        rp = max(1, int(round(clear_radius / self.res)))
        for i in range(len(c)):
            cv2.circle(walked, (int(c[i]), int(r[i])), rp, 1, -1)
            if i:
                cv2.line(walked, (int(c[i-1]), int(r[i-1])), (int(c[i]), int(r[i])), 1, 2 * rp)
        walked = walked.astype(bool)
        occ &= ~walked
        free |= walked

        self.grid[:] = UNKNOWN
        self.grid[free] = FREE
        self.grid[occ] = OCCUPIED
        self.noise = noise_metrics(occ)
        self.dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * self.res

    def stats(self):
        f = int(np.sum(self.grid == FREE))
        o = int(np.sum(self.grid == OCCUPIED))
        return {"size": f"{self.W}x{self.H}", "resolution": self.res, "free": f,
                "occupied": o, "unknown": self.grid.size - f - o,
                "coverage": (f + o) / self.grid.size * 100,
                "free_area": f * self.res ** 2, "occupied_area": o * self.res ** 2}


class SafePlanner:
    """A* whose step cost grows smoothly as the distance to obstacles shrinks."""

    def __init__(self, grid, robot_radius=0.2, safety=0.6, unknown_cost=4.0, weight=4.0):
        self.grid = grid
        self.robot_radius, self.safety = robot_radius, safety
        self.blocked = grid.dist < robot_radius
        self.cost = (1.0 + weight * np.clip((safety - grid.dist) / safety, 0, 1)
                     + unknown_cost * (grid.grid == UNKNOWN))

    def plan(self, start_world, goal_world):
        s, g = self.grid.world_to_cell(start_world), self.grid.world_to_cell(goal_world)
        blocked = self.blocked.copy()
        for c, r in (s, g):
            blocked[max(0, r-2):r+3, max(0, c-2):c+3] = False
        H, W = blocked.shape
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        h = lambda a: ((a[0]-g[0])**2 + (a[1]-g[1])**2) ** 0.5
        open_, came, gs, closed = [(h(s), s)], {}, {s: 0.0}, set()
        while open_:
            _, cur = heapq.heappop(open_)
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if cur in closed:
                continue
            closed.add(cur)
            for dc, dr, st in nbrs:
                nc, nr = cur[0] + dc, cur[1] + dr
                if 0 <= nc < W and 0 <= nr < H and not blocked[nr, nc]:
                    t = gs[cur] + st * self.cost[nr, nc]
                    if t < gs.get((nc, nr), 1e18):
                        came[(nc, nr)] = cur
                        gs[(nc, nr)] = t
                        heapq.heappush(open_, (t + h((nc, nr)), (nc, nr)))
        return None

    def path_length(self, path):
        if not path or len(path) < 2:
            return 0.0
        p = np.array(path, float)
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * self.grid.res)

    def clearance(self, path):
        if not path:
            return np.nan, np.nan
        d = np.array([self.grid.dist[r, c] for c, r in path])
        return float(d.min()), float(d.mean())


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render_grid(grid, px=6, safety=False, planner=None):
    """Base map (BGR). safety=True shades free space by obstacle distance."""
    img = np.zeros((grid.H, grid.W, 3), np.uint8)
    for k, c in COLORS.items():
        img[grid.grid == k] = c
    if safety:
        d = np.clip(grid.dist / (planner.safety if planner else 0.6), 0, 1)
        heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        free = grid.grid == FREE
        img[free] = (0.55 * heat[free] + 0.45 * img[free]).astype(np.uint8)
    img = cv2.resize(img, (grid.W * px, grid.H * px), interpolation=cv2.INTER_NEAREST)
    step = int(round(1.0 / grid.res)) * px
    for x in range(0, img.shape[1], step):
        cv2.line(img, (x, 0), (x, img.shape[0]), (95, 95, 95), 1)
    for y in range(0, img.shape[0], step):
        cv2.line(img, (0, y), (img.shape[1], y), (95, 95, 95), 1)
    occ = cv2.resize((grid.grid == OCCUPIED).astype(np.uint8), (grid.W * px, grid.H * px),
                     interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (30, 30, 120), 1)
    return img


def cells_px(grid, xy, px):
    c, r = grid.xy_to_cell(xy)
    return [(int(ci * px + px // 2), int(ri * px + px // 2)) for ci, ri in zip(c, r)]


def draw_polyline(img, pts, color, width, upto=None):
    n = len(pts) if upto is None else min(upto + 1, len(pts))
    for i in range(n - 1):
        cv2.line(img, pts[i], pts[i + 1], color, width, cv2.LINE_AA)


def draw_path(img, grid, path, px, planner=None, band=True):
    if not path or len(path) < 2:
        return
    pts = [(int(c * px + px // 2), int(r * px + px // 2)) for c, r in path]
    if band and planner is not None:
        overlay = img.copy()
        rr = int(round(planner.robot_radius / grid.res * px))
        for i in range(len(pts) - 1):
            cv2.line(overlay, pts[i], pts[i + 1], (0, 200, 255), 2 * rr)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (0, 90, 120), 5, cv2.LINE_AA)
        cv2.line(img, pts[i], pts[i + 1], (0, 230, 255), 3, cv2.LINE_AA)


def scale_bar(img, grid, px):
    bar = int(round(1.0 / grid.res)) * px
    y = img.shape[0] - 12
    cv2.line(img, (10, y), (10 + bar, y), (255, 255, 255), 3)
    cv2.putText(img, "1 m", (10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
