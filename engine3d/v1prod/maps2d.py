"""2D outputs: architectural floor plan (PNG + SVG), clean occupancy / navigation map
with a safe A* path, and the raw point-cloud occupancy (for comparison / debugging).

All maps are top-down in the model frame: image right = +x, image up = -z.
"""

import cv2
import numpy as np

from .layout import _disk, fill_holes, wall_endpoints
from .slam.geometry import GroundFrame
from .slam.occupancy import (FREE, OCCUPIED, UNKNOWN, OccupancyGrid, SafePlanner, cells_px,
                             draw_path, draw_polyline, noise_metrics, render_grid, scale_bar)

WALL_T = 0.12          # drawn wall thickness (m), on the back side of the measured inner face

C_BG = (250, 250, 248)
C_FLOOR = (236, 241, 244)
C_FLOOR_OUT = (242, 244, 245)
C_WALL = (40, 40, 44)
C_OBJ = (196, 203, 208)
C_OBJ_EDGE = (120, 128, 135)
C_DOOR = (30, 110, 210)
C_OPEN = (40, 150, 60)
C_WIN = (200, 140, 30)
C_TEXT = (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def model_ground():
    """Ground frame in MODEL coordinates (floor y = 0): 2D = (x, -z)."""
    return GroundFrame(np.array([0.0, 1.0, 0.0]), 0.0, np.array([0.0, 0.0, -1.0]))


# ----------------------------------------------------------------------
# rooms
# ----------------------------------------------------------------------

def segment_rooms(lay, features):
    """Rooms = parts of the enclosed floor separated by walls and by the openings /
    doors / walked passages in them (openings are closed with a line for the split)."""
    g = lay["grid"]
    m = lay["enclosed"].astype(np.uint8).copy()
    cut = np.zeros_like(m)
    spans = [(f["xz0"], f["xz1"]) for f in features] + \
            [(([p["offset"], p["t0"]] if p["wall_axis"] == "x" else [p["t0"], p["offset"]]),
              ([p["offset"], p["t1"]] if p["wall_axis"] == "x" else [p["t1"], p["offset"]]))
             for p in lay.get("passages", [])]
    for a, b in spans:
        pa, pb = g.pt(np.array(a))[0], g.pt(np.array(b))[0]
        cv2.line(cut, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), 1, 2)
    from .layout import draw_walls
    cut = draw_walls(g, lay["walls"], thickness=2, img=cut)
    m[cut > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    rooms = []
    for c in range(1, n):
        area = st[c, cv2.CC_STAT_AREA] * g.cell ** 2
        if area < 1.5:
            continue
        mask = lab == c
        rr, cc = np.nonzero(mask)
        x0, x1 = g.lo[0] + cc.min() * g.cell, g.lo[0] + (cc.max() + 1) * g.cell
        z0, z1 = g.lo[1] + rr.min() * g.cell, g.lo[1] + (rr.max() + 1) * g.cell
        rooms.append({"mask": mask, "area_m2": round(float(area), 2),
                      "bbox_m": [round(float(x0), 2), round(float(z0), 2), round(float(x1), 2), round(float(z1), 2)],
                      "size_m": [round(float(x1 - x0), 2), round(float(z1 - z0), 2)],
                      "centroid": [float(g.lo[0] + (cc.mean() + 0.5) * g.cell), float(g.lo[1] + (rr.mean() + 0.5) * g.cell)]})
    rooms.sort(key=lambda r: -r["area_m2"])
    for i, r in enumerate(rooms):
        r["id"] = f"room_{i + 1}"
        a, b = sorted(r["size_m"])
        if i > 0 and a < 1.6 and b / max(a, 1e-6) > 2.2:
            r["name"] = "Corridor" if sum(x["name"].startswith("Corridor") for x in rooms[:i]) == 0 else f"Corridor {i + 1}"
        else:
            r["name"] = "Main room" if i == 0 else f"Room {i + 1}"
    return rooms


# ----------------------------------------------------------------------
# floor plan
# ----------------------------------------------------------------------

class Canvas:
    def __init__(self, lo, hi, ppm=110, margin=1.2):
        self.lo = np.asarray(lo, float) - margin
        self.hi = np.asarray(hi, float) + margin
        self.ppm = ppm
        self.W = int((self.hi[0] - self.lo[0]) * ppm)
        self.H = int((self.hi[1] - self.lo[1]) * ppm)
        self.img = np.full((self.H, self.W, 3), C_BG, np.uint8)
        self.svg = []

    def p(self, xz):
        xz = np.atleast_2d(np.asarray(xz, float))
        return np.c_[(xz[:, 0] - self.lo[0]) * self.ppm, (xz[:, 1] - self.lo[1]) * self.ppm]

    def ip(self, xz):
        return tuple(np.round(self.p(xz)[0]).astype(int))

    # drawing primitives mirrored to SVG
    def poly(self, pts_xz, fill, edge=None, width=1):
        P = self.p(pts_xz)
        cv2.fillPoly(self.img, [np.round(P * 4).astype(np.int32)], fill, cv2.LINE_AA, shift=2)
        if edge is not None:
            cv2.polylines(self.img, [np.round(P * 4).astype(np.int32)], True, edge, width, cv2.LINE_AA, shift=2)
        self.svg.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in P)}" fill="{_hex(fill)}" '
                        f'stroke="{_hex(edge) if edge else "none"}" stroke-width="{width}"/>')

    def line(self, a, b, color, width, dash=False):
        pa, pb = self.p(a)[0], self.p(b)[0]
        if dash:
            L = np.linalg.norm(pb - pa)
            n = max(1, int(L / 10))
            for i in range(0, n, 2):
                q0, q1 = pa + (pb - pa) * i / n, pa + (pb - pa) * min(i + 1, n) / n
                cv2.line(self.img, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), color, width, cv2.LINE_AA)
        else:
            cv2.line(self.img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), color, width, cv2.LINE_AA)
        self.svg.append(f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" '
                        f'stroke="{_hex(color)}" stroke-width="{width}"' + (' stroke-dasharray="6,6"' if dash else "") + "/>")

    def arc(self, c, r, a0, a1, color, width):
        pc = self.p(c)[0]
        rp = r * self.ppm
        cv2.ellipse(self.img, tuple(np.round(pc).astype(int)), (int(rp), int(rp)), 0, a0, a1, color, width, cv2.LINE_AA)
        t0, t1 = np.radians(a0), np.radians(a1)
        s, e = pc + rp * np.array([np.cos(t0), np.sin(t0)]), pc + rp * np.array([np.cos(t1), np.sin(t1)])
        large = 1 if abs(a1 - a0) > 180 else 0
        sweep = 1 if a1 > a0 else 0
        self.svg.append(f'<path d="M {s[0]:.1f} {s[1]:.1f} A {rp:.1f} {rp:.1f} 0 {large} {sweep} {e[0]:.1f} {e[1]:.1f}" '
                        f'fill="none" stroke="{_hex(color)}" stroke-width="{width}"/>')

    def text(self, s, xz_or_px, scale=0.5, color=C_TEXT, center=True, bold=1, px=False):
        org = np.asarray(xz_or_px, float) if px else self.p(xz_or_px)[0]
        (tw, th), _ = cv2.getTextSize(s, FONT, scale, bold)
        o = (int(org[0] - tw / 2), int(org[1] + th / 2)) if center else (int(org[0]), int(org[1]))
        cv2.putText(self.img, s, o, FONT, scale, color, bold, cv2.LINE_AA)
        self.svg.append(f'<text x="{o[0]}" y="{o[1]}" font-family="Arial" font-size="{scale * 30:.0f}" '
                        f'fill="{_hex(color)}">{_esc(s)}</text>')

    def save_svg(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
                    f'viewBox="0 0 {self.W} {self.H}">\n<rect width="100%" height="100%" fill="{_hex(C_BG)}"/>\n')
            f.write("\n".join(self.svg))
            f.write("\n</svg>\n")


def _hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mask_polys(mask, g, eps_m=0.04):
    """Raster mask -> simplified polygons (x, z)."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, eps_m / g.cell, True).reshape(-1, 2).astype(float)
        out.append(np.c_[g.lo[0] + (c[:, 0] + 0.5) * g.cell, g.lo[1] + (c[:, 1] + 0.5) * g.cell])
    return out


def object_footprint(o, cell=0.02):
    """Clean top-down footprint: raster of the object between 10 cm and 2 m, opened
    (drops thin spill), main component(s); drawn as its minimum-area rectangle when the
    shape is essentially rectangular (furniture), else as a simplified outline."""
    P = np.asarray(o["mesh"].vertices)
    P = P[(P[:, 1] > 0.10) & (P[:, 1] < 2.0)][:, [0, 2]]
    if len(P) < 20:
        return None
    lo = P.min(0) - 0.1
    ij = ((P - lo) / cell).astype(int)
    m = np.zeros(tuple(ij.max(0)[::-1] + 3), np.uint8)
    m[ij[:, 1], ij[:, 0]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = st[1:, cv2.CC_STAT_AREA].max()
    m = np.isin(lab, [c for c in range(1, n) if st[c, cv2.CC_STAT_AREA] >= 0.15 * big]).astype(np.uint8)
    ys, xs = np.nonzero(m)
    pts = np.c_[xs, ys].astype(np.float32)
    rect = cv2.minAreaRect(pts)
    area_rect = max(rect[1][0] * rect[1][1], 1)
    if m.sum() / area_rect > 0.62:
        ang = rect[2]
        if abs(ang - 90 * round(ang / 90.0)) < 12:
            # nearly wall-parallel: axis-aligned box over the robust extent (plan convention)
            lo_q, hi_q = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0) + 1
            poly = np.array([[lo_q[0], lo_q[1]], [hi_q[0], lo_q[1]], [hi_q[0], hi_q[1]], [lo_q[0], hi_q[1]]],
                            np.float32)
        else:
            poly = cv2.boxPoints(rect)
    else:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 2, True).reshape(-1, 2).astype(np.float32)
    return poly * cell + lo + cell / 2


def _wall_quad(w, t0=None, t1=None, thick=WALL_T):
    t0 = w["t0"] if t0 is None else t0
    t1 = w["t1"] if t1 is None else t1
    o, s = w["offset"], w["sign"]
    b = o - s * thick                          # back face (wall body behind the inner face)
    if w["axis"] == "x":
        return np.array([[o, t0], [o, t1], [b, t1], [b, t0]])
    return np.array([[t0, o], [t1, o], [t1, b], [t0, b]])


def _along(w, xz):
    return xz[1] if w["axis"] == "x" else xz[0]


def _pt(w, t, off=0.0):
    o = w["offset"] + w["sign"] * off
    return np.array([o, t]) if w["axis"] == "x" else np.array([t, o])


def draw_floor_plan(out_base, lay, objects, features, rooms, cam_xz=None, title="Floor plan"):
    g = lay["grid"]
    ends = np.array([p for w in lay["walls"] for p in wall_endpoints(w)])
    rr, cc = np.nonzero(lay["floor"])
    fl = np.c_[g.lo[0] + cc * g.cell, g.lo[1] + rr * g.cell]
    allp = np.vstack([fl, ends]) if len(ends) else fl
    cv_ = Canvas(allp.min(0), allp.max(0))
    # floor: seen-through (outside) lighter, rooms
    for poly in _mask_polys(lay["floor"] & ~lay["enclosed"], g):
        cv_.poly(poly, C_FLOOR_OUT)
    for poly in _mask_polys(lay["enclosed"], g):
        cv_.poly(poly, C_FLOOR)
    # light 1 m grid inside the floor
    # objects (footprints = convex hull of the object between 5 cm and 2 m)
    for i, o in enumerate(objects):
        if o.get("ceiling_mounted") or o.get("is_door"):
            continue
        fp = object_footprint(o)
        if fp is None:
            continue
        cv_.poly(fp, C_OBJ, C_OBJ_EDGE, 1)
        cv_.text(str(i + 1), fp.mean(0), 0.45, (60, 60, 60))
    # walls with gaps for openings / doors / passages
    wall_gaps = {w["id"]: [] for w in lay["walls"]}
    for f in features:
        w = next(w for w in lay["walls"] if w["id"] == f["wall"])
        a, b = sorted([_along(w, f["xz0"]), _along(w, f["xz1"])])
        wall_gaps[w["id"]].append((a, b, f))
    for w in lay["walls"]:
        t = w["t0"]
        for a, b, f in sorted(wall_gaps[w["id"]], key=lambda x: x[0]):
            if a > t:
                cv_.poly(_wall_quad(w, t, a), C_WALL)
            t = max(t, b)
        if w["t1"] > t:
            cv_.poly(_wall_quad(w, t, w["t1"]), C_WALL)
        for a, b, f in wall_gaps[w["id"]]:
            jam0, jam1 = _pt(w, a), _pt(w, b)
            if f["type"] == "door":
                # closed door: leaf drawn open 90 deg into the room + swing arc (hinge side unknown -> left end)
                width = b - a
                leaf_end = _pt(w, a, off=width)
                cv_.line(jam0, leaf_end, C_DOOR, 2)
                p0 = cv_.p(jam0)[0]
                v_leaf = cv_.p(leaf_end)[0] - p0
                v_wall = cv_.p(jam1)[0] - p0
                a0, a1 = np.degrees(np.arctan2(v_leaf[1], v_leaf[0])), np.degrees(np.arctan2(v_wall[1], v_wall[0]))
                if a1 - a0 > 180:
                    a1 -= 360
                if a0 - a1 > 180:
                    a0 -= 360
                cv_.arc(jam0, width, min(a0, a1), max(a0, a1), C_DOOR, 1)
                cv_.line(jam0, jam1, C_DOOR, 1, dash=True)
            elif f["type"].startswith("window") or f["type"] == "glass door":
                cv_.poly(_wall_quad(w, a, b), (255, 255, 255), C_WIN, 1)
                cv_.line(_pt(w, a, -WALL_T / 2), _pt(w, b, -WALL_T / 2), C_WIN, 1)
            else:
                cv_.line(jam0, jam1, C_OPEN, 1, dash=True)
            # jambs
            for tt in (a, b):
                cv_.line(_pt(w, tt), _pt(w, tt, -WALL_T), C_WALL, 3)
    # walked-through passages
    for p in lay.get("passages", []):
        w = {"axis": p["wall_axis"], "offset": p["offset"], "sign": 1}
        a0 = _pt(w, p["t0"])
        a1 = _pt(w, p["t1"])
        cv_.line(a0, a1, C_OPEN, 2, dash=True)
    # wall dimension labels (inner face length), placed inside the room
    for w in lay["walls"]:
        L = w["t1"] - w["t0"]
        if L < 0.9:
            continue
        mid = _pt(w, (w["t0"] + w["t1"]) / 2,
                  off=-(WALL_T + (0.42 if w["axis"] == "x" else 0.16)))     # outside the wall
        cv_.text(f"{L:.2f} m", mid, 0.42, (90, 90, 90))
    # rooms
    for r in rooms:
        c = np.array(r["centroid"])
        cv_.text(r["name"], c, 0.62, C_TEXT, bold=2)
        cv_.text(f"{r['size_m'][0]:.1f} x {r['size_m'][1]:.1f} m  |  {r['area_m2']:.1f} m2", c + [0, 0.28], 0.45)
    # entrance marker: where the walk started
    if cam_xz is not None and len(cam_xz):
        s = cv_.ip(cam_xz[0])
        cv2.circle(cv_.img, s, 7, (40, 170, 40), -1, cv2.LINE_AA)
        cv_.text("start", (s[0] + 10, s[1] + 5), 0.42, (40, 120, 40), center=False, px=True)
    # scale bar, north arrow (= -z, 'up' of the map), legend, title
    H, W = cv_.H, cv_.W
    x0, y0 = 30, H - 30
    cv2.rectangle(cv_.img, (x0, y0 - 6), (x0 + cv_.ppm, y0), C_WALL, -1)
    cv2.rectangle(cv_.img, (x0 + cv_.ppm, y0 - 6), (x0 + 2 * cv_.ppm, y0), (255, 255, 255), -1)
    cv2.rectangle(cv_.img, (x0, y0 - 6), (x0 + 2 * cv_.ppm, y0), C_WALL, 1)
    cv_.text("0", (x0, y0 - 12), 0.4, center=False, px=True)
    cv_.text("1", (x0 + cv_.ppm - 4, y0 - 12), 0.4, center=False, px=True)
    cv_.text("2 m", (x0 + 2 * cv_.ppm - 8, y0 - 12), 0.4, center=False, px=True)
    cv_.text(title, (30, 40), 0.8, C_TEXT, center=False, bold=2, px=True)
    img = cv_.img
    # legend panel
    legend = [(C_WALL, "wall (measured inner face)"), (C_DOOR, "door (closed, swing assumed)"),
              (C_OPEN, "opening / doorway / walked passage"), (C_WIN, "window / pass-through"),
              (C_OBJ, "furniture / object (numbered)"), (C_FLOOR, "floor (room)"),
              (C_FLOOR_OUT, "floor seen through openings")]
    lw = 330
    panel = np.full((H, lw, 3), C_BG, np.uint8)
    y = 70
    cv2.putText(panel, "Legend", (15, 45), FONT, 0.6, C_TEXT, 2, cv2.LINE_AA)
    for col, s in legend:
        cv2.rectangle(panel, (15, y - 12), (40, y + 4), col, -1)
        cv2.rectangle(panel, (15, y - 12), (40, y + 4), (150, 150, 150), 1)
        cv2.putText(panel, s, (50, y), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)
        y += 26
    y += 10
    cv2.putText(panel, "Objects", (15, y), FONT, 0.6, C_TEXT, 2, cv2.LINE_AA)
    y += 24
    for i, o in enumerate(objects):
        if o.get("ceiling_mounted") or o.get("is_door") or y > H - 60:
            continue
        s = o["size_m"] if "size_m" in o else [0, 0, 0]
        cv2.putText(panel, f"{i + 1:2d} {o['label'][:16]:16s} {s[0]:.1f}x{s[2]:.1f}x{s[1]:.1f} m", (15, y),
                    FONT, 0.38, C_TEXT, 1, cv2.LINE_AA)
        y += 19
    cv2.putText(panel, "approx. metres (monocular scale)", (15, H - 40), FONT, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(panel, "up = -z of the model", (15, H - 20), FONT, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
    full = np.hstack([img, panel])
    cv2.imwrite(out_base + ".png", full)
    cv_.save_svg(out_base + ".svg")
    return full


# ----------------------------------------------------------------------
# occupancy / navigation
# ----------------------------------------------------------------------

def nav_grid(lay, objects, cam_centers, res=0.05, obstacle_max_h=2.0):
    """Clean occupancy from the layout: walls + object footprints = occupied,
    floor region = free, rest unknown."""
    gm = model_ground()
    og = OccupancyGrid(gm, res, obstacle_max_h=obstacle_max_h)
    g = lay["grid"]
    rr, cc = np.nonzero(lay["floor"])
    fl3 = np.c_[g.lo[0] + (cc + 0.5) * g.cell, np.zeros(len(rr)), g.lo[1] + (rr + 0.5) * g.cell]
    og.fit_extent(gm.to_2d(fl3), gm.to_2d(cam_centers))
    occ = np.zeros((og.H, og.W), np.float32)
    free = np.zeros((og.H, og.W), np.float32)
    og._acc(free, gm.to_2d(fl3))
    for w in lay["walls"]:
        n = max(2, int((w["t1"] - w["t0"]) / 0.01))
        t = np.linspace(w["t0"], w["t1"], n)
        for off in np.arange(0, WALL_T + 1e-6, 0.02):
            o = w["offset"] - w["sign"] * off
            P = np.c_[np.full(n, o), np.zeros(n), t] if w["axis"] == "x" else np.c_[t, np.zeros(n), np.full(n, o)]
            og._acc(occ, gm.to_2d(P))
    # openings / doors / passages are not obstacles (doors: assumed openable)
    for o in objects:
        if o.get("is_door") or o.get("ceiling_mounted"):
            continue
        P = np.asarray(o["mesh"].vertices)
        P = P[(P[:, 1] > 0.08) & (P[:, 1] < obstacle_max_h)]
        if len(P):
            og._acc(occ, gm.to_2d(P))
    occm = occ > 0
    occm = cv2.morphologyEx(occm.astype(np.uint8), cv2.MORPH_CLOSE, _disk(1)) > 0
    # the camera carrier walked here: no obstacle along the walked path
    walked = np.zeros_like(occm, np.uint8)
    cpx = [og.xy_to_cell(gm.to_2d(c[None]))for c in cam_centers]
    rp = max(1, int(round(0.25 / res)))
    for i, (cc_, rr_) in enumerate(cpx):
        cv2.circle(walked, (int(cc_[0]), int(rr_[0])), rp, 1, -1)
        if i:
            cv2.line(walked, (int(cpx[i - 1][0][0]), int(cpx[i - 1][1][0])), (int(cc_[0]), int(rr_[0])), 1, 2 * rp)
    occm &= walked == 0
    free[walked > 0] = 1
    freem = cv2.dilate((free > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    freem &= ~occm
    og.noise_raw = noise_metrics(occm)
    og.grid[:] = UNKNOWN
    og.grid[freem] = FREE
    og.grid[occm] = OCCUPIED
    og.noise = noise_metrics(occm)
    og.dist = np.minimum(cv2.distanceTransform((~occm).astype(np.uint8), cv2.DIST_L2, 5) * res, 10.0)
    return og


def raw_occupancy(pcd_points, cam_centers, res=0.05, obstacle_max_h=2.0):
    """Version-2 style evidence grid from the dense point cloud (for comparison)."""
    gm = model_ground()
    og = OccupancyGrid(gm, res, obstacle_max_h=obstacle_max_h)
    h = gm.height(pcd_points)
    ceil = np.percentile(h, 99.5)
    og.fit_extent(gm.to_2d(pcd_points[h < ceil - 0.1]), gm.to_2d(cam_centers))
    og.integrate(pcd_points)
    og.finalize(gm.to_2d(cam_centers))
    return og


def _legend(img, title_, rows):
    panel = np.full((max(img.shape[0], 440), 320, 3), 35, np.uint8)
    cv2.putText(panel, title_, (15, 32), FONT, 0.6, (255, 255, 255), 2)
    y = 70
    for col, lab in rows:
        if col is not None:
            cv2.rectangle(panel, (18, y - 12), (38, y + 4), col, -1)
        cv2.putText(panel, lab, (48 if col is not None else 18, y), FONT, 0.43, (220, 220, 220), 1)
        y += 26
    if img.shape[0] < panel.shape[0]:
        img = np.vstack([img, np.full((panel.shape[0] - img.shape[0], img.shape[1], 3), 35, np.uint8)])
    return np.hstack([img, panel[:img.shape[0]]])


def navigation_images(out_dir, og, raw_og, cam_centers, raw_centers, robot_radius, loops, low_conf):
    gm = model_ground()
    px = 6
    planner = SafePlanner(og, robot_radius=robot_radius)
    path = planner.plan(cam_centers[0], cam_centers[-1])
    plen = planner.path_length(path)
    cl = planner.clearance(path)
    ref_px = cells_px(og, gm.to_2d(cam_centers), px)
    raw_px = cells_px(og, gm.to_2d(raw_centers), px)
    st = og.stats()
    base = [((240, 235, 230), "Free (floor)"), ((60, 60, 180), "Occupied (wall / object)"),
            ((50, 50, 50), "Unknown")]
    img = render_grid(og, px)
    draw_polyline(img, raw_px, (150, 150, 150), 2)
    draw_polyline(img, ref_px, (220, 170, 40), 3)
    for lp in loops:
        cv2.line(img, ref_px[lp["i"]], ref_px[lp["j"]], (255, 0, 255), 1, cv2.LINE_AA)
    for k in low_conf:
        cv2.drawMarker(img, ref_px[k], (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.circle(img, ref_px[0], 8, (0, 200, 0), -1)
    cv2.circle(img, ref_px[-1], 8, (0, 0, 220), -1)
    scale_bar(img, og, px)
    occ_img = _legend(img, "OCCUPANCY MAP (clean)", base + [
        ((150, 150, 150), "DA3 raw trajectory"), ((220, 170, 40), "Refined trajectory"),
        ((0, 0, 255), "Low-confidence pose"), ((255, 0, 255), "Loop candidate"),
        ((0, 200, 0), "Start"), ((0, 0, 220), "End"),
        (None, f"{st['size']} cells @ {og.res:.2f} m"),
        (None, f"Free {st['free_area']:.1f} m2 | occupied {st['occupied_area']:.1f} m2"),
        (None, f"Noise: {og.noise['isolated_cells']} isolated, {og.noise['tiny_blobs']} blobs"),
        (None, "top = -z (model frame)")])
    cv2.imwrite(f"{out_dir}/occupancy_map.png", occ_img)

    img = render_grid(og, px, safety=True, planner=planner)
    draw_polyline(img, ref_px, (220, 170, 40), 2)
    draw_path(img, og, path, px, planner)
    cv2.circle(img, ref_px[0], 8, (0, 200, 0), -1)
    cv2.circle(img, ref_px[-1], 8, (0, 0, 220), -1)
    scale_bar(img, og, px)
    nav_img = _legend(img, "NAVIGATION MAP + SAFE A*", base + [
        (None, "Free space shaded by clearance"), (None, "  dark = near obstacle, yellow = safe"),
        ((220, 170, 40), "Walked trajectory"), ((0, 230, 255), "A* path start -> end"),
        ((0, 200, 255), f"Robot footprint r = {robot_radius:.2f} m"),
        (None, f"Length {plen:.2f} m" if path else "No path"),
        (None, f"Min clearance {cl[0]:.2f} m, mean {cl[1]:.2f} m" if path else "")])
    cv2.imwrite(f"{out_dir}/navigation_map.png", nav_img)

    img = render_grid(raw_og, px)
    rp = cells_px(raw_og, gm.to_2d(cam_centers), px)
    draw_polyline(img, rp, (220, 170, 40), 2)
    scale_bar(img, raw_og, px)
    cv2.imwrite(f"{out_dir}/raw_occupancy_pointcloud.png", _legend(img, "RAW OCCUPANCY (point cloud)", base + [
        (None, f"noise before morphology: {raw_og.noise_raw['isolated_cells']} isolated, "
               f"{raw_og.noise_raw['tiny_blobs']} blobs"),
        (None, "debug / comparison only")]))
    path_world = None
    if path:
        cells = np.array(path)
        path_world = gm.to_3d(og.cell_to_xy(cells[:, 0], cells[:, 1]), h=0.05)
    return planner, path, plen, cl, path_world
