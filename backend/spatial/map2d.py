"""2D inspection map (plan view: x right, z down = looking straight down; plan up = -z,
the same orientation as the 3D engine's floor_plan.png).

Layers: rooms, floor coverage, walls coloured by coverage per 10 cm column, doors,
furniture footprints, trajectory, current camera + field of view, inspection keyframes,
findings. `render(upto=f)` draws the state at SLAM frame f (live replay);
`render()` draws the final state.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.spatial.coverage import SURF_CEIL, SURF_FLOOR, SURF_WALL, Coverage
from backend.spatial.slam_data import SlamRun

# BGR
C = {"bg": (250, 250, 250), "room": (236, 236, 236), "unseen": (180, 180, 250), "seen": (120, 200, 255),
     "good": (140, 215, 140), "inspected": (70, 170, 60), "wall_unseen": (40, 40, 220), "wall_seen": (0, 165, 255),
     "wall_good": (60, 180, 60), "wall_inspected": (20, 120, 20), "furn": (205, 205, 205), "furn_edge": (120, 120, 120),
     "traj": (200, 110, 20), "cam": (180, 60, 0), "fov": (255, 200, 120), "kf": (180, 0, 180), "text": (30, 30, 30),
     "door": (40, 90, 160), "DETECTED": (38, 38, 220), "NEEDS_REVIEW": (12, 150, 234), "UNKNOWN": (235, 99, 37)}
STATE_FILL = {0: C["unseen"], 1: C["seen"], 2: C["good"], 3: C["inspected"]}
STATE_WALL = {0: C["wall_unseen"], 1: C["wall_seen"], 2: C["wall_good"], 3: C["wall_inspected"]}


def _put(img, text, org, scale=0.5, color=(30, 30, 30), thick=1):
    # outline by offsetting (OpenCV 5 thick strokes are wider)
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class PlanMap:
    def __init__(self, run: SlamRun, cov: Coverage, px_per_m: int = 90, margin: float = 0.7, max_px: int = 900):
        self.run, self.cov, self.px = run, cov, px_per_m
        L = run.layout
        xs, zs = [], []
        for r in L.get("rooms", []):
            x0, z0, x1, z1 = r["bbox_m"]
            xs += [x0, x1]; zs += [z0, z1]  # noqa: E702
        for w in L["walls"]:
            a = [w["offset"], w["t0"]] if w["axis"] == "x" else [w["t0"], w["offset"]]
            b = [w["offset"], w["t1"]] if w["axis"] == "x" else [w["t1"], w["offset"]]
            xs += [a[0], b[0]]; zs += [a[1], b[1]]  # noqa: E702
        cc = run.cam_centres()
        xs += list(cc[:, 0]); zs += list(cc[:, 2])  # noqa: E702
        self.x0, self.z0 = min(xs) - margin, min(zs) - margin
        ext = max(max(xs) + margin - self.x0, max(zs) + margin - self.z0)
        self.px = px_per_m = int(min(px_per_m, max_px / ext))   # big houses: keep the map <= max_px
        self.W = int((max(xs) + margin - self.x0) * px_per_m)
        self.H = int((max(zs) + margin - self.z0) * px_per_m)
        self.base = self._base()

    def p(self, x, z):
        return int(round((x - self.x0) * self.px)), int(round((z - self.z0) * self.px))

    def _base(self):
        img = np.full((self.H, self.W, 3), C["bg"], np.uint8)
        fm = self.run.layout.get("_floor_mask")
        if fm is not None:                    # true floor region
            fl, lo, cell = fm["floor"], fm["lo"], fm["cell"]
            ys, xs = np.mgrid[0:self.H, 0:self.W]
            ci = ((self.x0 + (xs + 0.5) / self.px - lo[0]) / cell).astype(int)
            ri = ((self.z0 + (ys + 0.5) / self.px - lo[1]) / cell).astype(int)
            ok = (ci >= 0) & (ci < fl.shape[1]) & (ri >= 0) & (ri < fl.shape[0])
            m = np.zeros((self.H, self.W), bool)
            m[ok] = fl[ri[ok], ci[ok]]
            img[m] = C["room"]
        else:
            for r in self.run.layout.get("rooms", []):
                x0, z0, x1, z1 = r["bbox_m"]
                cv2.rectangle(img, self.p(x0, z0), self.p(x1, z1), C["room"], -1)
        return img

    def _floor_layer(self, img, st):
        S = self.cov.samples
        h = max(1, int(S.step * self.px / 2))
        m = S.kind == SURF_FLOOR
        for (x, _, z), s in zip(S.xyz[m], st[m]):
            u, v = self.p(x, z)
            cv2.rectangle(img, (u - h, v - h), (u + h, v + h), STATE_FILL[int(s)], -1)

    def _furniture(self, img):
        for o in self.run.layout.get("objects", []):
            fp = o.get("footprint")
            if not fp or o.get("ceiling_mounted"):
                continue
            (cx, cz), (sx, sz), a = fp["center"], fp["size"], fp.get("angle_deg", 0.0)
            box = cv2.boxPoints(((cx, cz), (sx, sz), a))
            pts = np.array([self.p(x, z) for x, z in box], np.int32)
            cv2.fillPoly(img, [pts], C["furn"])
            cv2.polylines(img, [pts], True, C["furn_edge"], 1, cv2.LINE_AA)

    def _walls(self, img, st, thick=7):
        S = self.cov.samples
        G = np.array(S.group)
        for w in self.run.layout["walls"]:
            m = np.where((G == w["id"]) & (S.kind == SURF_WALL))[0]
            if not len(m):
                continue
            tk = np.round(S.t[m] / S.step).astype(int)
            for k in np.unique(tk):
                col = st[m[tk == k]]
                # a column's colour = the state reached by the majority of its height
                s = int(np.sort(col)[len(col) // 2])
                t0, t1 = k * S.step - S.step / 2, k * S.step + S.step / 2
                if w["axis"] == "x":
                    a, b = self.p(w["offset"], t0), self.p(w["offset"], t1)
                else:
                    a, b = self.p(t0, w["offset"]), self.p(t1, w["offset"])
                cv2.line(img, a, b, STATE_WALL[s], thick, cv2.LINE_AA)
        for d in self.run.layout.get("doors_and_openings", []):
            if "xz0" in d and "xz1" in d:
                a, b = self.p(*d["xz0"]), self.p(*d["xz1"])
                cv2.line(img, a, b, C["door"], 3, cv2.LINE_AA)

    def _trajectory(self, img, upto):
        cc = self.run.cam_centres()[: upto + 1]
        pts = np.array([self.p(x, z) for x, _, z in cc], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, C["traj"], 2, cv2.LINE_AA)
        if len(pts):
            cv2.circle(img, tuple(pts[0]), 6, (0, 170, 0), -1, cv2.LINE_AA)

    def _camera(self, img, f, fov_len=2.2):
        c2w = self.run.c2w[f]
        c = c2w[:3, 3]
        fwd = c2w[:3, 2]
        fx = self.run.K[f][0, 0]
        half = np.arctan(self.run.image_size[0] / 2 / fx)
        yaw = np.arctan2(fwd[0], fwd[2])
        pts = [self.p(c[0], c[2])]
        for a in np.linspace(-half, half, 9):
            pts.append(self.p(c[0] + fov_len * np.sin(yaw + a), c[2] + fov_len * np.cos(yaw + a)))
        over = img.copy()
        cv2.fillPoly(over, [np.array(pts, np.int32)], C["fov"])
        cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
        cv2.circle(img, pts[0], 8, C["cam"], -1, cv2.LINE_AA)
        tip = self.p(c[0] + 0.4 * np.sin(yaw), c[2] + 0.4 * np.cos(yaw))
        cv2.arrowedLine(img, pts[0], tip, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.4)

    def _keyframes(self, img, kfs, upto):
        for n, f in enumerate(kfs, 1):
            if upto is not None and f > upto:
                continue
            c = self.run.c2w[f][:3, 3]
            u, v = self.p(c[0], c[2])
            cv2.rectangle(img, (u - 5, v - 5), (u + 5, v + 5), C["kf"], 2, cv2.LINE_AA)

    def _findings(self, img, findings, upto):
        for fd in findings:
            if fd.get("xyz") is None or (upto is not None and fd["frame"] > upto):
                continue
            x, _, z = fd["xyz"]
            u, v = self.p(x, z)
            col = C.get(fd["state"], C["UNKNOWN"])
            cv2.circle(img, (u, v), 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), 11, col, 3, cv2.LINE_AA)
            _put(img, str(fd["n"]), (u - 5 if fd["n"] < 10 else u - 9, v + 5), 0.45, col, 1)

    def _scale_bar(self, img):
        x, y = 15, self.H - 15
        cv2.line(img, (x, y), (x + self.px, y), (0, 0, 0), 2)
        _put(img, "1 m", (x + self.px // 2 - 12, y - 6), 0.45)
        _put(img, "N (plan up)", (self.W - 110, 20), 0.45)

    def render(self, upto: int | None = None, kfs: list[int] | None = None, findings: list[dict] | None = None,
               show_camera: bool = True) -> np.ndarray:
        st = self.cov.state(upto)
        img = self.base.copy()
        self._floor_layer(img, st)
        self._furniture(img)
        self._walls(img, st)
        f_last = self.run.n - 1 if upto is None else upto
        self._trajectory(img, f_last)
        self._keyframes(img, kfs or [], upto)
        if show_camera and upto is not None:
            self._camera(img, upto)
        elif upto is None:
            c = self.run.c2w[-1][:3, 3]
            cv2.circle(img, self.p(c[0], c[2]), 6, (0, 0, 200), -1, cv2.LINE_AA)
        self._findings(img, findings or [], upto)
        self._scale_bar(img)
        return img


def legend(width: int) -> np.ndarray:
    items = [("Wall/floor not seen", C["wall_unseen"]), ("Seen (far / oblique)", C["wall_seen"]),
             ("Inspectable (<=3 m)", C["wall_good"]), ("Inspected by AI", C["wall_inspected"]),
             ("Walk path", C["traj"]), ("AI keyframe", C["kf"]), ("Defect: detected", C["DETECTED"]),
             ("Defect: needs review", C["NEEDS_REVIEW"])]
    rows = (len(items) + 1) // 2
    img = np.full((rows * 22 + 10, width, 3), 255, np.uint8)
    cw = width // 2
    for i, (t, col) in enumerate(items):
        x, y = 10 + (i % 2) * cw, 18 + (i // 2) * 22
        cv2.rectangle(img, (x, y - 11), (x + 16, y + 3), col, -1)
        _put(img, t, (x + 22, y), 0.45)
    return img


def stats_panel(width: int, height: int, stats: dict, t: float, extra: list[str]) -> np.ndarray:
    img = np.full((height, width, 3), 255, np.uint8)
    y = 26
    _put(img, f"Walkthrough t = {t:5.1f} s", (12, y), 0.62, (0, 0, 0), 2)
    y += 30
    for name in ("walls", "floor", "ceiling"):
        k = stats["by_kind"][name]
        _put(img, f"{name.capitalize():8s} seen {k['seen_pct']:5.1f}%   inspectable {k['inspectable_pct']:5.1f}%", (12, y), 0.5)
        bar_w = width - 24
        cv2.rectangle(img, (12, y + 6), (12 + bar_w, y + 14), (225, 225, 225), -1)
        cv2.rectangle(img, (12, y + 6), (12 + int(bar_w * k["seen_pct"] / 100), y + 14), C["wall_seen"], -1)
        cv2.rectangle(img, (12, y + 6), (12 + int(bar_w * k["inspectable_pct"] / 100), y + 14), C["wall_good"], -1)
        y += 34
    for line in extra:
        _put(img, line[:70], (12, y), 0.48)
        y += 21
    return img
