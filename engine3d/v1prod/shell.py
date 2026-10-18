"""Clean structural shell: flat walls / floor / ceiling textured with the real video.

Every texel (cfg.texture_res px per metre) of a planar surface is coloured from the
video frames that see exactly that surface point:
  * the measured model depth (furniture included) must agree with the texel depth,
    so furniture standing in front of a wall is never painted onto it
  * grazing views rejected; weights = viewing angle^2 / distance^2, feathered towards
    the image border (no seams from frame edges)
Openings: texels the video consistently sees THROUGH (model depth well behind the
wall) are cut out as real holes (door-sized components only) and classified as
doorway / window-or-pass-through by their position. Closed doors are found in the
wall texture as door-sized, floor-standing regions whose colour differs from the wall.
Floor: unseen floor (under furniture) and floor texels contaminated by furniture feet
/ shadows next to objects are refilled from clean floor one tile period away (OpenCV
DFT autocorrelation), the remainder with OpenCV inpainting. Ceiling (rarely filmed):
smoothed.
"""

import cv2
import numpy as np
import trimesh
from PIL import Image

from .layout import _disk


def log(msg):
    print(msg, flush=True)


def _bilinear(img, u, v):
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    I = img
    return ((1 - fx) * (1 - fy) * I[y0, x0] + fx * (1 - fy) * I[y0, x0 + 1]
            + (1 - fx) * fy * I[y0 + 1, x0] + fx * fy * I[y0 + 1, x0 + 1])


def bake(P, nrm, frames, K, w2c, depth, frame_ids, min_cos=0.25, surf_tol=(0.08, 0.04), beyond=0.35,
         frame_weight=None, power=2.0, fdepth=None, vdepth=None, extra=False):
    """Colour for world points P (M,3) on a surface with normal nrm.

    Visibility uses BOTH the frame's own (aligned, filtered) depth map `fdepth` and the
    fused model depth `depth`: the frame depth sees thin objects the mesh has holes in
    (plastic chair slats, legs), the model depth covers pixels the depth filter dropped.
    A texel is taken from a frame only if the frame depth agrees with it and the model
    has nothing in front of it. `vdepth` (raw DA3 depth of every pixel, processed
    resolution, aligned scale) is the occlusion test for thin structures the depth
    filter removed: anything clearly in front of the texel blocks it.
    Returns (rgb float (M,3), weight (M,), n_surface_hits, n_beyond_hits); with extra=True
    also (view-through colour, its weight, n_beyond_hits WITH reconstructed geometry behind)."""
    acc = np.zeros((len(P), 3))
    wsum = np.zeros(len(P))
    n_surf = np.zeros(len(P), np.float32)
    n_bey = np.zeros(len(P), np.float32)
    n_geo = np.zeros(len(P), np.float32)
    acc_t = np.zeros((len(P), 3))
    wsum_t = np.zeros(len(P))
    H, W = frames[0].shape[:2]
    for k in frame_ids:
        R, t = w2c[k][:3, :3], w2c[k][:3, 3]
        Xc = P @ R.T + t
        z = Xc[:, 2]
        front = z > 0.3
        if not front.any():
            continue
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        u[front] = K[k][0, 0] * Xc[front, 0] / z[front] + K[k][0, 2]
        v[front] = K[k][1, 1] * Xc[front, 1] / z[front] + K[k][1, 2]
        ok = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        cam = -R.T @ t
        ray = cam - P[idx]
        dist = np.linalg.norm(ray, axis=1)
        cosw = (ray @ nrm) / dist
        g = cosw > min_cos
        idx, cosw = idx[g], cosw[g]
        zi = z[idx]
        ui, vi = np.round(u[idx]).astype(int), np.round(v[idx]).astype(int)
        dm = depth[k][vi, ui]
        tol = np.maximum(surf_tol[0], surf_tol[1] * zi)
        if fdepth is not None:
            df = fdepth[k][vi, ui]
            zr = np.where(df > 0, df, dm)
            occluded = (dm > 0) & (dm < zi - tol)            # model surface in front
        else:
            zr = dm
            occluded = np.zeros(len(idx), bool)
        if vdepth is not None:
            hv, wv = vdepth[k].shape
            dvis = vdepth[k][np.clip((vi * hv) // H, 0, hv - 1), np.clip((ui * wv) // W, 0, wv - 1)]
            occluded |= (dvis > 0) & (dvis < zi - tol)
        surf = (zr > 0) & (np.abs(zr - zi) < tol) & ~occluded
        n_bey[idx[(zr > zi + beyond) & ((dm == 0) | (dm > zi + beyond * 0.5))]] += 1
        n_bey[idx[zr == 0]] += 0.5                      # nothing measured there: weak evidence
        if extra:
            n_geo[idx[(dm > zi + beyond)]] += 1          # reconstructed surface behind the wall
            tv = idx[~occluded]                          # what the camera sees at this spot
            wt = cosw[~occluded] ** 2 / np.maximum(zi[~occluded], 0.5) ** 2
            acc_t[tv] += _bilinear(frames[k], u[tv], v[tv]) * wt[:, None]
            wsum_t[tv] += wt
        s = idx[surf]
        n_surf[s] += 1
        edge = np.minimum.reduce([u[s], W - 1 - u[s], v[s], H - 1 - v[s]])
        # frontal, close and sharp views dominate (power > 1): less multi-view blur
        w = (cosw[surf] ** 2 / np.maximum(zi[surf], 0.5) ** 2) ** power * np.clip(edge / 80.0, 0.02, 1.0)
        if frame_weight is not None:
            w = w * frame_weight[k]
        acc[s] += _bilinear(frames[k], u[s], v[s]) * w[:, None]
        wsum[s] += w
    rgb = np.zeros((len(P), 3))
    seen = wsum > 0
    rgb[seen] = acc[seen] / wsum[seen, None]
    if extra:
        rgb_t = np.zeros((len(P), 3))
        st_ = wsum_t > 0
        rgb_t[st_] = acc_t[st_] / wsum_t[st_, None]
        return rgb, wsum, n_surf, n_bey, rgb_t, wsum_t, n_geo
    return rgb, wsum, n_surf, n_bey


def fill_unseen(img, unseen, seen, res, small_m2=0.3, down=8):
    """Small unseen patches: OpenCV inpainting at full resolution. Large unseen areas
    (occluded by furniture / never filmed): inpainting at 1/`down` resolution, upscaled -
    a smooth continuation of the SURROUNDING colours (a door stays brown, a curtain stays
    cream) without the streaks full-resolution inpainting draws across large holes."""
    if not unseen.any() or not seen.any():
        return img
    out = img.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats(unseen.astype(np.uint8), 8)
    big = np.zeros_like(unseen)
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] / res ** 2 >= small_m2:
            big |= lab == c
    if big.any():
        H, W = unseen.shape
        h2, w2 = max(2, H // down), max(2, W // down)
        f = np.where(unseen[..., None], 0, img).astype(np.float32)
        wv = cv2.resize((~unseen).astype(np.float32), (w2, h2), interpolation=cv2.INTER_AREA)
        sm = cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA) / np.maximum(wv[..., None], 1e-3)
        lo = cv2.inpaint(np.clip(sm, 0, 255).astype(np.uint8), (wv < 0.5).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        up = cv2.GaussianBlur(cv2.resize(lo, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), down / 2)
        core = cv2.erode(big.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        out[core] = up[core]
        unseen = unseen & ~core
    return cv2.inpaint(out, unseen.astype(np.uint8), 5, cv2.INPAINT_TELEA)


# ----------------------------------------------------------------------
# floor tile-period fill (from the V3 clean room)
# ----------------------------------------------------------------------

def tile_period(gray, valid, lo_px, hi_px):
    """Repeat distance (px) of a periodic floor along columns and rows from the
    autocorrelation (OpenCV DFT) + the peak strength (0..1, 1 = perfectly periodic)."""
    g = np.where(valid, gray - gray[valid].mean(), 0).astype(np.float32)
    F = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
    ac = cv2.idft(cv2.mulSpectrums(F, F, 0, conjB=True), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    # normalise by the overlap of the valid mask with itself
    Fm = cv2.dft(valid.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    n = cv2.idft(cv2.mulSpectrums(Fm, Fm, 0, conjB=True), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    ac = ac / np.maximum(n, 1)
    ac = ac / max(ac[0, 0], 1e-6)
    out = []
    for prof in (ac[0, :], ac[:, 0]):
        hi = min(hi_px, len(prof) // 2)
        # a real repeat is a LOCAL maximum of the autocorrelation (not the edge of the
        # search window on the decaying slope)
        seg = prof[max(lo_px - 1, 1):hi + 1]
        loc = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
        if not loc:
            out.append((lo_px, 0.0))
            continue
        i = max(loc, key=lambda j: seg[j]) + max(lo_px - 1, 1)
        out.append((i, float(prof[i])))
    return out


def fill_floor(img, hole, res, region):
    """Fill unseen / contaminated floor with a synthesised clean tile.

    The floor texture is averaged modulo the tile period (phase-aligned mean tile) over
    all clean texels -> a clean tile at the right phase everywhere. Its brightness is
    modulated by the local low-frequency shading of the real floor (inpainted into the
    hole). Non-periodic floors (weak autocorrelation peak) get the local colour only."""
    valid = ~hole & region
    if valid.sum() < 400 or not hole.any():
        return img, {}
    f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (px, sx), (py, sy) = tile_period(gray, valid, int(0.2 * res), int(1.2 * res))
    info = {"tile_period_m": [round(px / res, 3), round(py / res, 3)], "periodicity": round(min(sx, sy), 3)}
    # low-frequency shading (local mean colour), inpainted into the hole
    k = int(res * 0.6) | 1
    wv = cv2.GaussianBlur(valid.astype(np.float32), (k, k), 0)
    low = cv2.GaussianBlur(f * valid[..., None], (k, k), 0) / np.maximum(wv[..., None], 1e-3)
    low_u8 = np.clip(low, 0, 255).astype(np.uint8)
    # only confidently observed floor is a colour source (the outside of the region is black)
    low = cv2.inpaint(low_u8, (wv < 0.2).astype(np.uint8), 9, cv2.INPAINT_TELEA).astype(np.float32)
    out = f.copy()
    H, W = hole.shape
    if min(sx, sy) > 0.15:
        rr, cc = np.nonzero(valid)
        detail = f - cv2.GaussianBlur(f * valid[..., None], (k, k), 0) / np.maximum(wv[..., None], 1e-3)
        tile = np.zeros((py, px, 3))
        cnt = np.zeros((py, px))
        np.add.at(tile, (rr % py, cc % px), detail[rr, cc])
        np.add.at(cnt, (rr % py, cc % px), 1)
        tile = tile / np.maximum(cnt, 1)[..., None]
        hr, hc = np.nonzero(hole)
        out[hr, hc] = low[hr, hc] + tile[hr % py, hc % px]
    else:
        out[hole] = low[hole]
    # soften the seam between real and synthesised floor
    edge = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge &= ~cv2.erode(hole.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bl = cv2.GaussianBlur(out, (5, 5), 0)
    out[edge] = bl[edge]
    return np.clip(out, 0, 255).astype(np.uint8), info


# ----------------------------------------------------------------------
# surfaces
# ----------------------------------------------------------------------

def wall_frame(w, h):
    """origin (top-left seen from the room), du (along, left->right seen from the room),
    dv (down), length, height, inward normal."""
    if w["axis"] == "x":
        n = np.array([w["sign"], 0.0, 0.0])
        # looking along -n from inside: right-hand direction = cross(up, n) ... choose so the
        # texture is not mirrored when viewed from the room
        du = np.cross(n, [0, 1, 0])
        a, b = np.array([w["offset"], h, w["t0"]]), np.array([w["offset"], h, w["t1"]])
    else:
        n = np.array([0.0, 0.0, w["sign"]])
        du = np.cross(n, [0, 1, 0])
        a, b = np.array([w["t0"], h, w["offset"]]), np.array([w["t1"], h, w["offset"]])
    du = du / np.linalg.norm(du)
    o = a if (b - a) @ du > 0 else b
    L = float(np.linalg.norm(b - a))
    return o, du, np.array([0.0, -1.0, 0.0]), L, h, n


def grid_mesh(o, du, dv, L, Hh, img, keep_cell=None, cell=0.05, face_normal=None):
    """Planar grid (cells dropped where keep_cell is False) with UVs into img."""
    nu, nv = max(1, int(round(L / cell))), max(1, int(round(Hh / cell)))
    a, b = np.meshgrid(np.linspace(0, L, nu + 1), np.linspace(0, Hh, nv + 1))
    V = o + a.reshape(-1, 1) * du + b.reshape(-1, 1) * dv
    UV = np.c_[a.reshape(-1) / L, 1 - b.reshape(-1) / Hh]
    rr, cc = np.meshgrid(np.arange(nv), np.arange(nu), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    if keep_cell is not None:
        th, tw = keep_cell.shape
        ti = np.minimum(th - 1, ((rr + 0.5) / nv * th).astype(int))
        tj = np.minimum(tw - 1, ((cc + 0.5) / nu * tw).astype(int))
        sel = keep_cell[ti, tj]
        rr, cc = rr[sel], cc[sel]
    i0 = rr * (nu + 1) + cc
    q = np.stack([i0, i0 + 1, i0 + nu + 2, i0 + nu + 1], 1)
    flip = face_normal is not None and np.dot(np.cross(du, dv), face_normal) < 0
    if flip:
        Fc = np.concatenate([q[:, [0, 2, 1]], q[:, [0, 3, 2]]])
    else:
        Fc = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(img), metallicFactor=0.0,
                                              roughnessFactor=0.95, doubleSided=False)
    m = trimesh.Trimesh(V, Fc, visual=trimesh.visual.TextureVisuals(uv=UV, material=mat), process=False)
    m.remove_unreferenced_vertices()
    return m


def _rect_components(mask, res, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] / res ** 2 >= min_area:
            x, y, w, h = st[c, :4]
            out.append((lab == c, x / res, y / res, w / res, h / res, st[c, cv2.CC_STAT_AREA] / (w * h)))
    return out


def bake_wall(w, h, frames, K, w2c, depth, frame_ids, res, frame_weight=None, fdepth=None, vdepth=None):
    o, du, dv, L, Hh, n = wall_frame(w, h)
    tw, th = max(2, int(L * res)), max(2, int(Hh * res))
    jj, ii = np.meshgrid((np.arange(tw) + 0.5) / res, (np.arange(th) + 0.5) / res)
    P = o + jj.reshape(-1, 1) * du + ii.reshape(-1, 1) * dv
    # wider surface tolerance: flush doors / TVs / pictures up to ~10 cm in front of the
    # wall are part of the wall texture
    rgb, wsum, n_s, n_b, rgb_t, wsum_t, n_geo = bake(P, n, frames, K, w2c, depth, frame_ids,
                                                     surf_tol=(0.10, 0.04), frame_weight=frame_weight,
                                                     fdepth=fdepth, vdepth=vdepth, extra=True)
    seen = (wsum > 0).reshape(th, tw)
    img = np.clip(rgb, 0, 255).astype(np.uint8).reshape(th, tw, 3)
    # openings: consistently seen through
    op = ((n_b >= 3) & (n_b > 2 * n_s)).reshape(th, tw).astype(np.uint8)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    op = cv2.morphologyEx(cv2.morphologyEx(op, cv2.MORPH_OPEN, k9), cv2.MORPH_CLOSE, k9)
    opening = np.zeros((th, tw), bool)
    openings = []
    for m, x, y, ww, hh, fill in _rect_components(op, res, 0.25):
        top = Hh - y                                    # height of the top edge above floor
        bottom = Hh - (y + hh)
        if min(ww, hh) < 0.35:
            continue
        # openings are rectangular: fill the bounding box (door frames are straight)
        r0, r1 = int(y * res), int((y + hh) * res)
        c0, c1 = int(x * res), int((x + ww) * res)
        box = np.zeros_like(opening)
        box[r0:r1, c0:c1] = True
        if fill < 0.55:
            box = m
        # something reconstructed behind it? (another room) -> real hole. Nothing (glass
        # door / window to the dark outside, sky) -> keep the wall closed and paint it with
        # what the video shows there: no black holes into nothing
        bm = box.ravel()
        geo = float(n_geo[bm].sum() / max(n_b[bm].sum(), 1e-6))
        if geo < 0.35:
            paint = box & (wsum_t.reshape(th, tw) > 0)
            img[paint] = np.clip(rgb_t.reshape(th, tw, 3)[paint], 0, 255).astype(np.uint8)
            seen = seen | paint
            kind = "glass door" if bottom < 0.2 and hh > 1.5 else "window"
            openings.append({"type": kind, "closed": True, "along_m": [round(x, 3), round(x + ww, 3)],
                             "bottom_m": round(bottom, 3), "top_m": round(top, 3),
                             "width_m": round(ww, 3), "height_m": round(hh, 3), "geometry_behind": round(geo, 2)})
            continue
        opening |= box
        kind = "doorway" if bottom < 0.2 and hh > 1.5 else ("window/pass-through" if bottom > 0.3 else "opening")
        openings.append({"type": kind, "along_m": [round(x, 3), round(x + ww, 3)],
                         "bottom_m": round(bottom, 3), "top_m": round(top, 3),
                         "width_m": round(ww, 3), "height_m": round(hh, 3), "geometry_behind": round(geo, 2)})
    unseen = ~seen & ~opening
    img = fill_unseen(img, unseen, seen, res)
    doors = detect_doors(img, seen & ~opening, res, Hh)
    mesh = grid_mesh(o, du, dv, L, Hh, img, keep_cell=~opening, face_normal=n)
    stats = {"seen_pct": round(float(seen.mean()) * 100, 1), "opening_pct": round(float(opening.mean()) * 100, 1)}
    return mesh, img, opening, openings, doors, stats, (o, du, dv, L, Hh, n)


def detect_doors(img, seen, res, Hh, min_dE=22):
    """Closed doors: door-sized, floor-standing regions whose colour differs from the wall."""
    if seen.mean() < 0.2:
        return []
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 2), cv2.COLOR_RGB2LAB).astype(np.float32)
    med = np.median(lab[seen], axis=0)
    dE = np.linalg.norm((lab - med) * [100 / 255, 1, 1], axis=2)
    # candidates on the whole texture (a door partly hidden by furniture is filled with its
    # own colour); each door must still be mostly OBSERVED (checked below)
    cand = (dE > min_dE).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cand = cv2.morphologyEx(cv2.morphologyEx(cand, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # remove thin horizontal / vertical structures (skirting boards, cables, rails, frames)
    # that would connect a door to everything else along the wall
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.15 * res)), max(3, int(0.30 * res))))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kd)
    doors = []
    for m, x, y, ww, hh, fill in _rect_components(cand, res, 0.8):
        bottom = Hh - (y + hh)
        if bottom >= 0.25 or not 1.6 < hh < max(2.6, Hh + 0.05) or seen[m].mean() < 0.5:
            continue
        # the door = a run of TALL columns (colour over >= 80 % of the region height);
        # shorter furniture / shadows attached at the bottom of the region are excluded
        col_h = m.sum(0) / res
        tall = col_h >= 0.8 * hh
        c0 = int(round(x * res))
        for s_, e_ in _runs_1d(tall[c0:c0 + int(round(ww * res)) + 1]):
            w_run = (e_ - s_) / res
            if not 0.55 < w_run < 1.6:
                continue
            sub = m[:, c0 + s_:c0 + e_]
            if sub.sum() / max(sub.shape[1] * hh * res, 1) < 0.6:
                continue
            xa = x + s_ / res
            doors.append({"type": "door", "along_m": [round(xa, 3), round(xa + w_run, 3)],
                          "bottom_m": round(bottom, 3), "top_m": round(bottom + hh, 3),
                          "width_m": round(w_run, 3), "height_m": round(hh, 3)})
    return doors


def _runs_1d(mask):
    m = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def behind_walls(xz, walls, margin=0.0):
    """True for points on the back side of any wall (within the wall's extent)."""
    b = np.zeros(len(xz), bool)
    for w in walls:
        a, o = (0, 1) if w["axis"] == "x" else (1, 0)
        d = (xz[:, a] - w["offset"]) * w["sign"]
        inside = (xz[:, o] > w["t0"] + 0.02) & (xz[:, o] < w["t1"] - 0.02)
        b |= inside & (d < -margin) & (d > -0.5)
    return b


def snap_to_walls(V, walls, dist=0.09):
    """Move floor / ceiling boundary vertices exactly onto the nearby wall planes, so
    floor, ceiling and walls meet without a seam."""
    V = V.copy()
    for w in walls:
        a, o = (0, 2) if w["axis"] == "x" else (2, 0)
        d = V[:, a] - w["offset"]
        m = (np.abs(d) < dist) & (V[:, o] > w["t0"] - dist) & (V[:, o] < w["t1"] + dist)
        V[m, a] = w["offset"]
    return V


def bake_horizontal(lay, which, frames, K, w2c, depth, frame_ids, res, blocked=None, frame_weight=None,
                    fdepth=None, vdepth=None):
    """Floor (y = 0, facing up) or ceiling (y = h, facing down) over the floor region."""
    g = lay["grid"]
    # clean model: floor / ceiling of the enclosed (walked) rooms only; floor seen only
    # through openings is low-evidence (grazing, often dark) and stays in the 2D plan
    mask = lay["enclosed"]
    # extend up to the walls (the region stops a cell short of the wall lines), never past them
    X, Z = g.centers()
    grow = cv2.dilate(mask.astype(np.uint8), _disk(3)) > 0
    beh = behind_walls(np.c_[X.ravel(), Z.ravel()], lay["walls"]).reshape(mask.shape)
    mask = mask | (grow & ~beh)
    rows, cols = np.nonzero(mask)
    x0, x1 = g.lo[0] + cols.min() * g.cell, g.lo[0] + (cols.max() + 1) * g.cell
    z0, z1 = g.lo[1] + rows.min() * g.cell, g.lo[1] + (rows.max() + 1) * g.cell
    y = 0.0 if which == "floor" else lay["ceiling_h"]
    n = np.array([0, 1.0, 0]) if which == "floor" else np.array([0, -1.0, 0])
    L, D = x1 - x0, z1 - z0
    tw, th = max(2, int(L * res)), max(2, int(D * res))
    jj, ii = np.meshgrid((np.arange(tw) + 0.5) / res, (np.arange(th) + 0.5) / res)
    P = np.c_[x0 + jj.ravel(), np.full(jj.size, y), z0 + ii.ravel()]
    # texel inside the region?
    r, c = g.ij(P[:, [0, 2]])
    inside = mask[np.clip(r, 0, g.H - 1), np.clip(c, 0, g.W - 1)].reshape(th, tw)
    rgb, wsum, _, _ = bake(P, n, frames, K, w2c, depth, frame_ids, min_cos=0.45 if which == "floor" else 0.25,
                           frame_weight=frame_weight, fdepth=fdepth, vdepth=vdepth)
    if blocked is not None:
        b = blocked[np.clip(r, 0, g.H - 1), np.clip(c, 0, g.W - 1)]
        wsum = np.where(b, 0, wsum)
    seen = (wsum > 0).reshape(th, tw)
    img = np.clip(rgb, 0, 255).astype(np.uint8).reshape(th, tw, 3)
    stats = {"seen_pct_of_region": round(float((seen & inside).sum() / max(inside.sum(), 1)) * 100, 1)}
    hole = inside & ~seen
    if which == "floor" and (seen & inside).sum() > 1000:
        # weak evidence (only far / grazing views, e.g. a dark room seen through a door)
        # that also disagrees with the floor colour -> refilled; well-seen rugs stay
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        med = np.median(lab[seen & inside], axis=0)
        dE = np.linalg.norm((lab - med) * [100 / 255, 1, 1], axis=2)
        weak = (wsum.reshape(th, tw) < 2e-3) & seen & inside & (dE > 20)
        weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        hole |= weak & inside
        stats["weak_evidence_refilled_pct"] = round(float(weak.sum() / max(inside.sum(), 1)) * 100, 1)
    if which == "floor":
        # contamination next to objects (feet, shadows, smeared chair legs): far from the floor colour
        if blocked is not None and seen.any():
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            med = np.median(lab[seen & inside], axis=0)
            dE = np.linalg.norm((lab - med) * [100 / 255, 1, 1], axis=2)
            near = cv2.dilate(blocked.astype(np.uint8), _disk(max(1, int(round(0.3 / g.cell))))) > 0
            near_t = near[np.clip(r, 0, g.H - 1), np.clip(c, 0, g.W - 1)].reshape(th, tw)
            bad = seen & inside & near_t & (dE > 20)
            bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            hole |= bad & inside
            stats["refilled_near_objects_pct"] = round(float(bad.sum() / max(inside.sum(), 1)) * 100, 1)
        img, finfo = fill_floor(img, hole, res, inside)
        stats.update(finfo)
        hole = np.zeros_like(hole)
    if hole.any() and (inside & ~hole).any():
        img = cv2.inpaint(img, (hole | ~inside).astype(np.uint8), 7, cv2.INPAINT_TELEA) if which == "ceiling" \
            else cv2.inpaint(img, hole.astype(np.uint8), 7, cv2.INPAINT_TELEA)
    if which == "ceiling":
        img = cv2.GaussianBlur(img, (0, 0), 10)
    o = np.array([x0, y, z0])
    du, dv = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    mesh = grid_mesh(o, du, dv, L, D, img, keep_cell=inside, face_normal=n)
    mesh.vertices = snap_to_walls(np.asarray(mesh.vertices), lay["walls"])
    return mesh, img, stats


def build(lay, objects_blocked, frames, K, w2c, depth, frame_ids, res=160, frame_weight=None, fdepth=None,
          vdepth=None):
    """Returns dict(meshes {name: trimesh}, textures {name: img}, openings, doors, stats)."""
    h = lay["ceiling_h"]
    meshes, tex, stats, feats = {}, {}, {}, []
    for w in lay["walls"]:
        m, img, opening, ops, doors, st, frame = bake_wall(w, h, frames, K, w2c, depth, frame_ids, res, frame_weight, fdepth, vdepth)
        w["seen_pct"] = st["seen_pct"]
        if st["seen_pct"] < 15:
            # almost never filmed: no visual evidence -> kept in the plan / layout, not
            # rendered in 3D (would be an arbitrary colour slab)
            stats[w["id"]] = dict(st, rendered=False)
            w["rendered"] = False
            log(f"    {w['id']}: texture {st['seen_pct']:.0f}% from video -> not rendered (no evidence)")
            continue
        w["rendered"] = True
        meshes[w["id"]] = m
        tex[w["id"]] = img
        stats[w["id"]] = st
        o, du = frame[0], frame[1]
        for f in ops + doors:
            a, b = f["along_m"]
            p0, p1 = o + a * du, o + b * du
            f["wall"] = w["id"]
            f["xz0"], f["xz1"] = [round(float(p0[0]), 3), round(float(p0[2]), 3)], [round(float(p1[0]), 3), round(float(p1[2]), 3)]
            feats.append(f)
        w["openings"] = ops
        w["doors"] = doors
        log(f"    {w['id']}: texture {st['seen_pct']:.0f}% from video | openings {len(ops)} | doors {len(doors)}")
    m, img, st = bake_horizontal(lay, "floor", frames, K, w2c, depth, frame_ids, res, blocked=objects_blocked,
                                frame_weight=frame_weight, fdepth=fdepth, vdepth=vdepth)
    meshes["floor"], tex["floor"], stats["floor"] = m, img, st
    log(f"    floor: {st}")
    m, img, st = bake_horizontal(lay, "ceiling", frames, K, w2c, depth, frame_ids, max(32, res // 2),
                                frame_weight=frame_weight, fdepth=fdepth, vdepth=vdepth)
    meshes["ceiling"], tex["ceiling"], stats["ceiling"] = m, img, st
    log(f"    ceiling: {st}")
    return {"meshes": meshes, "textures": tex, "features": feats, "stats": stats}
