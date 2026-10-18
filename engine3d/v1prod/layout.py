"""Structural layout from the regularised model: wall segments, floor region, ceiling.

Model frame: x right, y up, z back, floor y = 0, walls aligned to x / z (Manhattan).

Walls
  * candidates = the Manhattan wall planes of the plane regularisation (x = c or z = c)
  * planes of the same orientation closer than `merge_dist` are one wall (duplicate
    surfaces from residual drift) - the offset of the best-supported one is kept
  * the extent of a wall along its plane comes from its support raster (5 cm cells in
    (position, height)): a column counts as wall when it has support ABOVE furniture
    height (walls reach the ceiling, sofas / tables / chairs do not) and at least
    `min_support_h` of support in total; gaps up to `max_gap` are closed (unobserved
    strips, door lintels); runs shorter than `min_wall_length` are dropped
  * corners: a wall end within `corner_snap` of a perpendicular wall is moved exactly
    onto it, so walls meet in sharp, closed corners
Floor
  * flood fill from the walked camera path, bounded by the wall lines, intersected with
    the observed floor support (closed 1 m, holes filled: floor under furniture) plus
    observed floor seen through openings
Ceiling height = the best-supported downward-facing horizontal plane above 1.8 m.
"""

import cv2
import numpy as np

AX = {"x": 0, "y": 1, "z": 2}


def log(msg):
    print(msg, flush=True)


def ceiling_height(planes, V):
    ceil = [p for p in planes if p["axis"] == "y" and p["normal_sign"] < 0 and p["offset"] > 1.8]
    if ceil:
        return float(max(ceil, key=lambda p: p["area_m2"])["offset"])
    return float(np.percentile(V[:, 1], 99.5))


def _runs(mask):
    """[start, end) index runs of True."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close_1d(mask, gap):
    out = mask.copy()
    runs = _runs(mask)
    for (s0, e0), (s1, e1) in zip(runs[:-1], runs[1:]):
        if s1 - e0 <= gap:
            out[e0:s1] = True
    return out


def extract_walls(V, N, area_v, planes, h, cam_xz, cell=0.05, merge_dist=0.2, band=0.04,
                  furniture_h=1.45, min_support_h=0.5, max_gap=2.5, min_wall_length=0.5,
                  min_plane_area=0.3):
    """Returns a list of wall dicts:
       axis ('x' | 'z'), offset, sign (normal direction along axis, facing the room),
       t0, t1 (extent along the other horizontal axis), support_m2, residual_rms (m)."""
    cands = [p for p in planes if p["axis"] in ("x", "z") and p["area_m2"] >= min_plane_area]
    # group same-orientation planes that are closer than merge_dist
    groups = []
    for p in sorted(cands, key=lambda p: -p["area_m2"]):
        for g in groups:
            q = g[0]
            if q["axis"] == p["axis"] and q["normal_sign"] == p["normal_sign"] and \
                    abs(q["offset"] - p["offset"]) < merge_dist:
                g.append(p)
                break
        else:
            groups.append([p])
    top = min(furniture_h, h - 0.35)
    walls = []
    for g in groups:
        a = AX[g[0]["axis"]]
        o = 2 if a == 0 else 0
        sign = g[0]["normal_sign"]
        off = g[0]["offset"]
        sup = np.zeros(len(V), bool)
        for p in g:
            sup |= np.abs(V[:, a] - p["offset"]) < band
        sup &= (N[:, a] * sign > 0.7) & (V[:, 1] > 0.03) & (V[:, 1] < h - 0.03)
        if sup.sum() < 50:
            continue
        t, y, w = V[sup, o], V[sup, 1], area_v[sup]
        lo = np.floor(t.min() / cell) * cell
        nt = int(np.ceil((t.max() - lo) / cell)) + 1
        ny = int(np.ceil(h / cell)) + 1
        ti = np.clip(((t - lo) / cell).astype(int), 0, nt - 1)
        yi = np.clip((y / cell).astype(int), 0, ny - 1)
        R = np.zeros((nt, ny))
        np.add.at(R, (ti, yi), w)
        occ = R > 0.25 * cell * cell
        high = occ[:, int(top / cell):].any(1)
        tall = occ.sum(1) * cell >= min_support_h
        col = high & tall
        col = _close_1d(col, int(round(max_gap / cell)))
        for s, e in _runs(col):
            L = (e - s) * cell
            if L < min_wall_length:
                continue
            t0, t1 = lo + s * cell, lo + e * cell
            m = sup & (V[:, o] >= t0) & (V[:, o] <= t1)
            walls.append({"axis": "xz"[a // 2], "offset": float(off), "sign": int(sign),
                          "t0": float(t0), "t1": float(t1),
                          "support_m2": float(area_v[m].sum()),
                          "plane_support_m2": float(area_v[sup].sum()),
                          "merged_planes": len(g)})
    # a wall must face the walked space (the room side): the camera path has to be
    # on its normal side somewhere along its extent
    keep = []
    for wll in walls:
        a = AX[wll["axis"]]
        o = 2 if a == 0 else 0
        ci, co = (0, 1) if a == 0 else (1, 0)          # cam_xz columns: x, z
        along = (cam_xz[:, co] > wll["t0"] - 1.0) & (cam_xz[:, co] < wll["t1"] + 1.0)
        side = (cam_xz[:, ci] - wll["offset"]) * wll["sign"]
        if (along & (side > 0.2)).any() or not along.any():
            keep.append(wll)
    return keep


def snap_corners(walls, snap=0.4, extend=1.2):
    """Move wall ends onto nearby perpendicular walls -> closed, sharp corners.
    Trimming / extending by up to `snap`; extending OUTWARD (the wall is too short, e.g.
    it stops at a recessed door next to the corner) by up to `extend`. Openings in the
    extended part are cut later by the see-through test / walked passages."""
    for w in walls:
        a = w["axis"]
        perp = [q for q in walls if q["axis"] != a]
        for end in ("t0", "t1"):
            best, bd = None, extend
            for q in perp:
                # q runs along our axis direction at position q.offset (in our t coordinate);
                # it must reach our plane (our offset lies within q's extent, +-snap)
                if not (q["t0"] - snap <= w["offset"] <= q["t1"] + snap):
                    continue
                dd = abs(q["offset"] - w[end])
                outward = (q["offset"] < w[end]) if end == "t0" else (q["offset"] > w[end])
                if dd < (extend if outward else snap) and dd < bd:
                    best, bd = q, dd
            if best is not None:
                w[end] = float(best["offset"])
                w.setdefault("corners", []).append(end)
    for w in walls:                                   # keep t0 < t1
        if w["t1"] < w["t0"]:
            w["t0"], w["t1"] = w["t1"], w["t0"]
    return [w for w in walls if w["t1"] - w["t0"] > 0.2]


def dedupe_walls(walls, dist=0.3, overlap=0.6):
    """Drop a wall that duplicates a better-supported parallel wall facing the same way
    (offset within `dist`, extents overlapping by >= `overlap` of the shorter)."""
    walls = sorted(walls, key=lambda w: -w["support_m2"])
    out = []
    for w in walls:
        dup = False
        for q in out:
            if q["axis"] != w["axis"] or q["sign"] != w["sign"] or abs(q["offset"] - w["offset"]) > dist:
                continue
            ov = min(q["t1"], w["t1"]) - max(q["t0"], w["t0"])
            if ov > overlap * min(q["t1"] - q["t0"], w["t1"] - w["t0"]):
                dup = True
                break
        if not dup:
            out.append(w)
    return out


def wall_endpoints(w):
    """2D (x, z) endpoints of a wall."""
    if w["axis"] == "x":
        return np.array([w["offset"], w["t0"]]), np.array([w["offset"], w["t1"]])
    return np.array([w["t0"], w["offset"]]), np.array([w["t1"], w["offset"]])


class Grid2D:
    """Top-down raster in the model frame: column = x, row = z."""

    def __init__(self, lo, hi, cell):
        self.lo, self.cell = np.asarray(lo, float), cell
        self.W = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
        self.H = int(np.ceil((hi[1] - lo[1]) / cell)) + 1

    def ij(self, xz):
        xz = np.atleast_2d(xz)
        c = np.floor((xz[:, 0] - self.lo[0]) / self.cell).astype(int)
        r = np.floor((xz[:, 1] - self.lo[1]) / self.cell).astype(int)
        return r, c

    def raster(self, xz, weights=None):
        r, c = self.ij(xz)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        m = np.zeros((self.H, self.W), np.float32)
        np.add.at(m, (r[ok], c[ok]), 1 if weights is None else weights[ok])
        return m

    def centers(self):
        xs = self.lo[0] + (np.arange(self.W) + 0.5) * self.cell
        zs = self.lo[1] + (np.arange(self.H) + 0.5) * self.cell
        return np.meshgrid(xs, zs)

    def pt(self, xz):
        """float pixel coords (col, row) for drawing."""
        xz = np.atleast_2d(xz)
        return np.c_[(xz[:, 0] - self.lo[0]) / self.cell, (xz[:, 1] - self.lo[1]) / self.cell]


def _disk(r_cells):
    k = 2 * int(r_cells) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask):
    m = mask.astype(np.uint8)
    H, W = m.shape
    pad = np.zeros((H + 2, W + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((H + 4, W + 4), np.uint8), (0, 0), 1)
    holes = ff[1:-1, 1:-1] == 0
    return mask | holes


def draw_walls(grid, walls, thickness=1, value=1, img=None):
    img = np.zeros((grid.H, grid.W), np.uint8) if img is None else img
    for w in walls:
        p, q = wall_endpoints(w)
        a, b = grid.pt(p)[0], grid.pt(q)[0]
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), value, thickness)
    return img


def floor_region(V, N, walls, cam_xz, cell=0.05, support_close=1.0, walk_r=0.3, passages=()):
    """Returns (grid, floor_mask, enclosed_mask, observed_floor_mask)."""
    fl = (V[:, 1] < 0.05) & (N[:, 1] > 0.7)
    Pf = V[fl][:, [0, 2]]
    ends = np.array([p for w in walls for p in wall_endpoints(w)]) if walls else np.zeros((0, 2))
    allp = np.vstack([Pf, cam_xz] + ([ends] if len(ends) else []))
    lo, hi = allp.min(0) - 0.6, allp.max(0) + 0.6
    g = Grid2D(lo, hi, cell)
    obs = g.raster(Pf) > 0
    obs = cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_CLOSE, _disk(1)) > 0
    walked = np.zeros((g.H, g.W), np.uint8)
    pc = g.pt(cam_xz)
    rp = max(1, int(round(walk_r / cell)))
    for i in range(len(pc)):
        cv2.circle(walked, tuple(np.round(pc[i]).astype(int)), rp, 1, -1)
        if i:
            cv2.line(walked, tuple(np.round(pc[i - 1]).astype(int)), tuple(np.round(pc[i]).astype(int)), 1, 2 * rp)
    walked = walked > 0
    barrier = draw_walls(g, walls, thickness=2) > 0
    # walked-through doorways closed for the room flood fill (each room judged on its own)
    closed = barrier.copy()
    for p in passages:
        a0 = [p["offset"], p["t0"]] if p["wall_axis"] == "x" else [p["t0"], p["offset"]]
        a1 = [p["offset"], p["t1"]] if p["wall_axis"] == "x" else [p["t1"], p["offset"]]
        pa, pb = g.pt(np.array(a0))[0], g.pt(np.array(a1))[0]
        tmp = np.zeros_like(closed, np.uint8)
        cv2.line(tmp, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), 1, 2)
        closed |= tmp > 0
    support = cv2.morphologyEx((obs | walked).astype(np.uint8), cv2.MORPH_CLOSE,
                               _disk(support_close / cell / 2)) > 0
    support = fill_holes(support)
    free = (~closed).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    seeds = np.unique(lab[walked & ~closed])
    seeds = seeds[seeds > 0]
    # a room fully bounded by walls (its flood component does not reach the grid border)
    # is floor everywhere inside, including under furniture standing against the walls;
    # an open / leaking component is limited to the observed floor support
    enclosed = np.zeros_like(support)
    for sd in seeds:
        comp = lab == sd
        leaks = comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
        enclosed |= (comp & support) if leaks else comp
    # remove thin spurs, fill holes (floor under furniture)
    enclosed = cv2.morphologyEx(enclosed.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)) > 0
    enclosed = fill_holes(enclosed) & ~barrier
    # give the doorway cells (closed only for the flood) back to the floor
    enclosed |= closed & ~barrier & (cv2.dilate(enclosed.astype(np.uint8), _disk(1)) > 0)
    # keep the enclosed components that contain the walked path
    n, lab, st, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[walked & enclosed])] = True
    keep[0] = False
    enclosed = keep[lab]
    # observed floor beyond the enclosed area (seen through openings), >= 0.5 m2
    outside = obs & ~cv2.dilate(enclosed.astype(np.uint8), _disk(1)).astype(bool) & ~barrier
    n, lab, st, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8), connectivity=4)
    extra = np.zeros_like(outside)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] * cell * cell >= 0.5:
            extra |= lab == c
    extra = cv2.morphologyEx(extra.astype(np.uint8), cv2.MORPH_OPEN, _disk(2)) > 0
    floor = enclosed | extra
    return g, floor, enclosed, obs


def fill_to_walls(grid, enclosed, walls, reach=1.2):
    """Grow the floor region across the band between it and each wall (room side of the
    wall, within its extent, never behind any wall), keeping only what connects to the
    existing floor: the room then reaches its walls."""
    X, Z = grid.centers()
    xz = np.c_[X.ravel(), Z.ravel()]
    band = np.zeros(len(xz), bool)
    behind = np.zeros(len(xz), bool)
    for w in walls:
        a, o = (0, 1) if w["axis"] == "x" else (1, 0)
        d = (xz[:, a] - w["offset"]) * w["sign"]
        inside = (xz[:, o] > w["t0"]) & (xz[:, o] < w["t1"])
        band |= inside & (d > 0) & (d < reach)
        behind |= inside & (d < 0) & (d > -0.5)
    cand = (band & ~behind).reshape(enclosed.shape) | enclosed
    cand &= ~(draw_walls(grid, walls, thickness=1) > 0)
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    keep = np.unique(lab[enclosed & cand])
    keep = keep[keep > 0]
    return np.isin(lab, keep) | enclosed


def trim_walls_to_floor(walls, grid, enclosed, reach=0.25):
    """Clip each wall to the part that actually borders the enclosed floor (removes wall
    stubs that stick out of the room past a corner)."""
    out = []
    dil = cv2.dilate(enclosed.astype(np.uint8), _disk(max(1, int(round(reach / grid.cell))))) > 0
    for w in walls:
        L = w["t1"] - w["t0"]
        n = max(2, int(np.ceil(L / grid.cell)))
        t = np.linspace(w["t0"], w["t1"], n)
        xz = np.c_[np.full(n, w["offset"]), t] if w["axis"] == "x" else np.c_[t, np.full(n, w["offset"])]
        r, c = grid.ij(xz)
        ok = (r >= 0) & (r < grid.H) & (c >= 0) & (c < grid.W)
        near = np.zeros(n, bool)
        near[ok] = dil[r[ok], c[ok]]
        near = _close_1d(near, int(round(2.5 / grid.cell)))      # remove stubs, keep interior gaps
        runs = [(s, e) for s, e in _runs(near) if (e - s) * grid.cell >= 0.3]
        for s, e in runs:
            q = dict(w)
            q["t0"], q["t1"] = float(t[s]), float(t[min(e, n - 1)])
            # keep exact corner positions when the run reaches the original end
            if s == 0:
                q["t0"] = w["t0"]
            if e >= n - 1:
                q["t1"] = w["t1"]
            out.append(q)
    return out


def drop_weak_walls(walls, min_support=0.8, short=1.0, isolated_support=1.5):
    """Weak evidence: plane support below min_support, or a short wall standing alone
    (no corner with another wall) without strong support - a furniture face, an open door
    leaf or a cupboard side, not a wall."""
    out = []
    for w in walls:
        if w.get("plane_support_m2", 1e9) < min_support:
            continue
        L = w["t1"] - w["t0"]
        if L < short and not w.get("corners") and w.get("plane_support_m2", 1e9) < isolated_support:
            continue
        out.append(w)
    return out


def split_at_crossings(walls, cam_xz, half_width=0.4, weak_support=2.0):
    """The camera cannot walk through a wall: where the trajectory crosses a wall line,
    cut a doorway (+-half_width around the crossing, merged with nearby crossings)."""
    out, passages = [], []
    for w in walls:
        a = 0 if w["axis"] == "x" else 1          # column of cam_xz across the wall
        o = 1 - a
        s = cam_xz[:, a] - w["offset"]
        cross = []
        for i in range(len(cam_xz) - 1):
            if s[i] == 0 or s[i] * s[i + 1] < 0:
                f = s[i] / (s[i] - s[i + 1]) if s[i] != s[i + 1] else 0.0
                tc = cam_xz[i, o] + f * (cam_xz[i + 1, o] - cam_xz[i, o])
                if w["t0"] - 0.05 < tc < w["t1"] + 0.05:
                    cross.append(tc)
        if not cross:
            out.append(w)
            continue
        if w.get("plane_support_m2", 1e9) < weak_support:
            # a weakly supported "wall" the camera walks straight through is a furniture
            # face or a ghost, not a wall
            continue
        cross = sorted(cross)
        cuts = []
        for tc in cross:
            if cuts and tc - half_width <= cuts[-1][1]:
                cuts[-1][1] = tc + half_width
            else:
                cuts.append([tc - half_width, tc + half_width])
        t = w["t0"]
        for c0, c1 in cuts:
            passages.append({"wall_axis": w["axis"], "offset": w["offset"], "t0": c0, "t1": c1})
            if c0 - t >= 0.3:
                q = dict(w)
                q["t0"], q["t1"] = t, c0
                out.append(q)
            t = max(t, c1)
        if w["t1"] - t >= 0.3:
            q = dict(w)
            q["t0"] = t
            out.append(q)
    return out, passages


def extract(model, planes, cam_centers, cfg):
    """Full layout: walls (snapped), ceiling height, floor raster."""
    import open3d as o3d  # noqa
    from .slam.room import vertex_areas
    model.compute_vertex_normals()
    V, N = np.asarray(model.vertices), np.asarray(model.vertex_normals)
    area_v = vertex_areas(V, np.asarray(model.triangles))
    h = ceiling_height(planes, V)
    cam_xz = cam_centers[:, [0, 2]]
    walls = extract_walls(V, N, area_v, planes, h, cam_xz, min_wall_length=cfg.min_wall_length)
    walls = dedupe_walls(walls)
    walls = snap_corners(walls)
    walls, passages = split_at_crossings(walls, cam_xz)
    grid, floor, enclosed, obs = floor_region(V, N, walls, cam_xz, cell=cfg.grid_resolution, passages=passages)
    path_len = float(np.linalg.norm(np.diff(cam_xz, axis=0), axis=1).sum()) if len(cam_xz) > 1 else 0.0
    degenerate = enclosed.sum() * grid.cell ** 2 < 3.0 and floor.sum() > enclosed.sum()
    if degenerate:
        # not a walkthrough (camera hardly moved / no enclosed walked space): use the
        # observed floor as the room region so the visible part is still modelled
        log(f"  WARNING: walked path {path_len:.1f} m, enclosed area "
            f"{enclosed.sum() * grid.cell ** 2:.1f} m2 - not a room walkthrough; using the observed floor")
        enclosed = floor.copy()
    # a wall borders the room if it is within 1.2 m of the floor region (the observed floor
    # often stops short of the walls: glossy floors, furniture standing along the walls)
    walls = trim_walls_to_floor(walls, grid, enclosed, reach=1.2)
    walls = [w for w in walls if w["t1"] - w["t0"] >= cfg.min_wall_length * 0.6]
    walls = drop_weak_walls(walls)
    grown = fill_to_walls(grid, enclosed, walls)
    if grown.sum() > enclosed.sum():
        log(f"  floor extended to the walls: +{(grown.sum() - enclosed.sum()) * grid.cell ** 2:.1f} m2")
    floor = floor | grown
    enclosed = grown
    for i, w in enumerate(walls):
        w["id"] = f"wall_{i:02d}"
        w["length_m"] = round(w["t1"] - w["t0"], 3)
    log(f"  ceiling {h:.2f} m | {len(walls)} wall segments | {len(passages)} walked-through passages | floor {floor.sum() * grid.cell ** 2:.1f} m2 "
        f"(enclosed {enclosed.sum() * grid.cell ** 2:.1f} m2)")
    for w in walls:
        log(f"    {w['id']}: {w['axis']} = {w['offset']:+.2f} m (normal {'+' if w['sign'] > 0 else '-'}{w['axis']}), "
            f"{w['t0']:+.2f}..{w['t1']:+.2f} ({w['length_m']:.2f} m), support {w['support_m2']:.1f} m2")
    return {"ceiling_h": h, "walls": walls, "passages": passages, "grid": grid, "floor": floor, "enclosed": enclosed,
            "walked_path_m": path_len, "degenerate": bool(degenerate),
            "floor_observed": obs}
