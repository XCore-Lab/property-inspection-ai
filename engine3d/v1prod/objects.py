"""Objects (furniture, doors standing open, curtains, fans, ...) = the reconstructed
geometry that is not structure.

From the regularised TSDF model:
  * remove the floor (y < floor_cut), the ceiling band and wall surfaces (within
    `wall_band` of a wall plane, normal parallel to the wall normal, inside its extent)
  * remove geometry behind walls / outside the floor region (other side of a wall,
    reconstruction spill through openings)
  * split into connected components; drop fragments (few triangles / tiny extent)
  * group nearby components into object instances (DBSCAN on the components' points)
  * per object: small holes filled (open3d tensor fill_holes), non-manifold edges and
    degenerate triangles removed, statistics (centre, size, oriented footprint, height)
Object meshes keep the measured geometry and the video colours (no invented shape).
"""

import cv2
import numpy as np
import open3d as o3d

from .layout import _disk


def log(msg):
    print(msg, flush=True)


def _submesh(mesh, keep_tri):
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_triangles_by_mask(~keep_tri)
    m.remove_unreferenced_vertices()
    return m


def structural_vertex_mask(V, N, lay, floor_cut=0.035, floor_band=0.10, ceil_band=0.08,
                           wall_band=0.06, flat_band=0.12):
    """Floor / ceiling / wall surface. Flat, wall-parallel surfaces up to `flat_band` in
    front of a wall (closed doors in their frames, TVs, pictures, curtains) belong to the
    wall: they are kept as wall texture instead of thin slab objects."""
    h = lay["ceiling_h"]
    s = (V[:, 1] < floor_cut) | (V[:, 1] > h - ceil_band)
    s |= (V[:, 1] < floor_band) & (N[:, 1] > 0.6)          # floor bumps / TSDF floor noise
    for w in lay["walls"]:
        a = 0 if w["axis"] == "x" else 2
        o = 2 - a
        d = (V[:, a] - w["offset"]) * w["sign"]          # > 0 on the room side
        inside = (V[:, o] > w["t0"] - 0.1) & (V[:, o] < w["t1"] + 0.1)
        near = d > -0.1                                   # bounded: not everything behind a wall
        s |= inside & near & (d < wall_band) & (np.abs(N[:, a]) > 0.5)
        s |= inside & near & (d < flat_band) & (np.abs(N[:, a]) > 0.93)
        s |= inside & (d < -0.02) & (d > -0.4)            # inside / just behind the wall body
    return s


def multiview_support(V, depth, valid, K, w2c, frames, tol=(0.04, 0.02)):
    """Number of frames whose (aligned, filtered) depth confirms each vertex."""
    sup = np.zeros(len(V), np.int32)
    H, W = depth.shape[1:]
    for k in frames:
        R, t = w2c[k][:3, :3], w2c[k][:3, 3]
        Xc = V @ R.T + t
        z = Xc[:, 2]
        f = z > 0.1
        u = np.full(len(V), -1)
        v = np.full(len(V), -1)
        u[f] = np.round(K[k][0, 0] * Xc[f, 0] / z[f] + K[k][0, 2]).astype(int)
        v[f] = np.round(K[k][1, 1] * Xc[f, 1] / z[f] + K[k][1, 2]).astype(int)
        ok = f & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.flatnonzero(ok)
        d = depth[k][v[idx], u[idx]]
        good = valid[k][v[idx], u[idx]] & (np.abs(d - z[idx]) < np.maximum(tol[0], tol[1] * z[idx]))
        sup[idx[good]] += 1
    return sup


def over_floor_mask(V, lay, margin=0.1, region="floor"):
    g = lay["grid"]
    fl = cv2.dilate(lay[region].astype(np.uint8), _disk(max(1, int(round(margin / g.cell))))) > 0
    r, c = g.ij(V[:, [0, 2]])
    ok = (r >= 0) & (r < g.H) & (c >= 0) & (c < g.W)
    m = np.zeros(len(V), bool)
    m[ok] = fl[r[ok], c[ok]]
    return m


def fill_small_holes(mesh, hole_size=0.08):
    try:
        t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        f = t.fill_holes(hole_size=hole_size).to_legacy()
        if len(f.triangles) >= len(mesh.triangles):
            # fill_holes drops vertex colours of the new vertices only (there are none: it
            # reuses the boundary vertices) - keep colours of the original mesh
            if len(f.vertices) == len(mesh.vertices):
                f.vertex_colors = mesh.vertex_colors
                return f
    except Exception:
        pass
    return mesh


def extract(model, lay, depth=None, valid=None, K=None, w2c=None, frames=None, min_support=3,
            min_triangles=600, min_extent=0.15, group_eps=0.06, min_height=0.12):
    model.compute_vertex_normals()
    V, N = np.asarray(model.vertices), np.asarray(model.vertex_normals)
    F = np.asarray(model.triangles)
    struct = structural_vertex_mask(V, N, lay)
    keepv = ~struct & over_floor_mask(V, lay, region="enclosed")
    n_unsup = 0
    if depth is not None:
        cand = np.flatnonzero(keepv)
        sup = multiview_support(V[cand], depth, valid, K, w2c, frames)
        weak = cand[sup < min_support]
        keepv[weak] = False
        n_unsup = len(weak)
    rest = _submesh(model, keepv[F].all(1))
    rest.remove_degenerate_triangles()
    rest.remove_non_manifold_edges()
    # split at 30 cm below the ceiling: ceiling-attached parts (beams, rods, lights) become
    # their own objects and can never bridge furniture into one instance
    cut = lay["ceiling_h"] - 0.30
    Vr, Fr = np.asarray(rest.vertices), np.asarray(rest.triangles)
    high = Vr[Fr][:, :, 1].max(1) > cut
    objects = []
    for part, is_high in ((_submesh(rest, ~high), False), (_submesh(rest, high), True)):
        for o in _instances(part, min_triangles, min_extent, group_eps, min_height):
            o["ceiling_mounted"] = is_high
            objects.append(o)
    objects.sort(key=lambda o: -o["triangles"])
    for i, o in enumerate(objects):
        o["id"] = f"object_{i:02d}"
        o["label"] = "object"
    log(f"  {len(objects)} objects separated from the structure "
        f"({sum(o['triangles'] for o in objects):,} triangles; {n_unsup:,} vertices without "
        f"multi-view support removed)")
    return objects


def _instances(rest, min_triangles, min_extent, group_eps, min_height):
    """Connected components -> filtered -> grouped (DBSCAN) into object instances."""
    if len(rest.triangles) == 0:
        return []
    ids, counts, _ = rest.cluster_connected_triangles()
    ids, counts = np.asarray(ids), np.asarray(counts)
    Vr, Fr = np.asarray(rest.vertices), np.asarray(rest.triangles)
    # component filter
    good = []
    for c in np.flatnonzero(counts >= min_triangles // 4):
        tri = ids == c
        P = Vr[np.unique(Fr[tri])]
        ext = np.ptp(P, axis=0)
        if counts[c] >= min_triangles or ext.max() >= 0.35:
            if ext.max() >= min_extent:
                good.append(c)
    if not good:
        return []
    keep_tri = np.isin(ids, good)
    rest = _submesh(rest, keep_tri)
    # group components into object instances
    ids, counts, _ = rest.cluster_connected_triangles()
    ids = np.asarray(ids)
    Vr, Fr = np.asarray(rest.vertices), np.asarray(rest.triangles)
    comp_pts = []
    for c in range(ids.max() + 1):
        vv = np.unique(Fr[ids == c])
        P = Vr[vv]
        if len(P) > 400:
            P = P[np.random.default_rng(c).choice(len(P), 400, replace=False)]
        comp_pts.append(P)
    allP = np.vstack(comp_pts)
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(comp_pts)])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(allP))
    lab = np.asarray(pc.cluster_dbscan(eps=group_eps, min_points=1))
    comp_group = np.zeros(len(comp_pts), int)
    for i in range(len(comp_pts)):
        comp_group[i] = np.bincount(lab[owner == i]).argmax()
    objects = []
    for gi in np.unique(comp_group):
        comps = np.flatnonzero(comp_group == gi)
        m = _submesh(rest, np.isin(ids, comps))
        P = np.asarray(m.vertices)
        if len(m.triangles) < min_triangles:
            continue
        ext = np.ptp(P, axis=0)
        if ext[1] < min_height and (P[:, 1].min() > 0.3 or P[:, 1].max() < 0.2):
            continue                                   # thin shelf-less slivers / floor noise
        if max(ext[0], ext[2]) < 0.2 and ext[1] > 1.2:
            continue                                   # door-frame / wall-edge slivers
        m = fill_small_holes(m)
        m.compute_vertex_normals()
        P = np.asarray(m.vertices)
        (cx, cz), (w, d), ang = cv2.minAreaRect(P[:, [0, 2]].astype(np.float32))
        objects.append({"mesh": m, "center": [float(cx), float((P[:, 1].min() + P[:, 1].max()) / 2), float(cz)],
                        "footprint": {"center": [float(cx), float(cz)], "size": [float(w), float(d)],
                                      "angle_deg": float(ang)},
                        "bbox_min": P.min(0).tolist(), "bbox_max": P.max(0).tolist(),
                        "height_m": float(P[:, 1].max()), "bottom_m": float(P[:, 1].min()),
                        "triangles": len(m.triangles)})
    return objects


def footprint_mask(objects, grid, min_h=0.08, max_h=2.0):
    """Top-down raster of the object geometry between min_h and max_h (obstacles)."""
    m = np.zeros((grid.H, grid.W), np.uint8)
    for o in objects:
        P = np.asarray(o["mesh"].vertices)
        P = P[(P[:, 1] > min_h) & (P[:, 1] < max_h)]
        if not len(P):
            continue
        r, c = grid.ij(P[:, [0, 2]])
        ok = (r >= 0) & (r < grid.H) & (c >= 0) & (c < grid.W)
        m[r[ok], c[ok]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(1))
    return m > 0


def _pca_thickness(P):
    c = P - P.mean(0)
    sub = c[np.random.default_rng(0).choice(len(c), min(len(c), 4000), replace=False)]
    n = np.linalg.svd(sub, full_matrices=False)[2][2]
    t = c @ n
    return float(np.percentile(t, 95) - np.percentile(t, 5)), n


def classify_structural_objects(objects, lay, wall_rgb):
    """Split the object list into
      * real objects,
      * closed doors standing in their frames: flat, door-sized, parallel to and within
        25 cm of a wall -> kept in 3D, reported as a door on that wall (not an obstacle),
      * wall ghosts: a single thin vertical sheet in wall colour (no top surface, which
        any real object below eye height would show) = misregistered wall -> removed.
    Returns (objects, doors, removed)."""
    keep, doors, removed = [], [], []
    wl = cv2.cvtColor(np.uint8([[np.clip(wall_rgb, 0, 255)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    for o in objects:
        P = np.asarray(o["mesh"].vertices)
        C = np.asarray(o["mesh"].vertex_colors)
        thick, n = _pca_thickness(P)
        ext = np.ptp(P, axis=0)
        vertical = abs(n[1]) < 0.2
        if vertical and thick < 0.2 and ext[1] > 1.7 and o["bottom_m"] < 0.3:
            width = float(np.hypot(ext[0], ext[2]))
            for w in lay["walls"]:
                a = 0 if w["axis"] == "x" else 2
                if abs(n[a]) < 0.9 or not 0.45 < width < 1.6:
                    continue
                d = (P[:, a] - w["offset"]) * w["sign"]
                t = P[:, 2 - a]
                if -0.1 < np.median(d) < 0.25 and t.min() > w["t0"] - 0.2 and t.max() < w["t1"] + 0.2:
                    t0, t1 = float(np.percentile(t, 2)), float(np.percentile(t, 98))
                    xz0 = [w["offset"], t0] if w["axis"] == "x" else [t0, w["offset"]]
                    xz1 = [w["offset"], t1] if w["axis"] == "x" else [t1, w["offset"]]
                    doors.append({"type": "door", "wall": w["id"], "source": "door leaf geometry",
                                  "width_m": round(t1 - t0, 3), "height_m": round(float(P[:, 1].max()), 3),
                                  "bottom_m": 0.0, "top_m": round(float(P[:, 1].max()), 3),
                                  "xz0": [round(float(v), 3) for v in xz0], "xz1": [round(float(v), 3) for v in xz1],
                                  "object": o["id"]})
                    o["label"] = "door"
                    o["is_door"] = True
                    break
            if o.get("is_door"):
                keep.append(o)
                continue
        lab = cv2.cvtColor((np.clip(C, 0, 1) * 255).astype(np.uint8)[None], cv2.COLOR_RGB2LAB)[0].astype(float)
        dE = float(np.linalg.norm((np.median(lab, 0) - wl) * [100 / 255, 1, 1]))
        o["mesh"].compute_vertex_normals()
        top = float(np.mean(np.asarray(o["mesh"].vertex_normals)[:, 1] > 0.7))
        if vertical and thick < 0.10 and dE < 18 and top < 0.05 and o["label"] == "object":
            host = _nearby_host(o, objects, P)
            if host is not None:
                # a detached flat part of a neighbouring object (e.g. a sofa back cushion
                # separated by a small reconstruction gap): merge it back, never delete it
                host.setdefault("merged_parts", []).append(o["id"])
                o["merged_into"] = host["id"]
                continue
            o["removed_reason"] = f"wall ghost (planar {thick * 100:.0f} cm, wall colour dE {dE:.0f})"
            removed.append(o)
            continue
        keep.append(o)
    # apply merges
    for o in objects:
        if o.get("merged_into"):
            host = next(h for h in objects if h["id"] == o["merged_into"])
            host["mesh"] = host["mesh"] + o["mesh"]
            host["mesh"].compute_vertex_normals()
            P = np.asarray(host["mesh"].vertices)
            host["bbox_min"], host["bbox_max"] = P.min(0).tolist(), P.max(0).tolist()
            host["size_m"] = np.ptp(P, axis=0).round(3).tolist()
            host["triangles"] = len(host["mesh"].triangles)
    for i, o in enumerate(keep):
        o["id"] = f"object_{i:02d}"
    return keep, doors, removed


def _nearby_host(o, objects, P, dist=0.2):
    """Largest other object whose geometry comes within `dist` of P."""
    from scipy.spatial import cKDTree
    t = cKDTree(P[:: max(1, len(P) // 3000)])
    best = None
    for h in objects:
        if h is o or h.get("merged_into") or h["triangles"] < o["triangles"]:
            continue
        Q = np.asarray(h["mesh"].vertices)
        Q = Q[:: max(1, len(Q) // 20000)]
        d, _ = t.query(Q, distance_upper_bound=dist)
        if np.isfinite(d).any() and (best is None or h["triangles"] > best["triangles"]):
            best = h
    return best


def remove_wall_duplicates(objects, lay, wall_textures, res, wall_frame, max_dist=1.2, max_dE=16,
                           min_tris=150):
    """Duplicated surfaces: a thin vertical sheet inside an object, parallel to a nearby
    wall, whose colour matches the wall texture right behind it, is a displaced copy of
    that wall surface (e.g. a sheer curtain whose depth DA3 got wrong) -> removed.
    Returns the number of removed sheets."""
    removed = 0
    keep_objs = []
    for o in objects:
        m = o["mesh"]
        ids, counts, _ = m.cluster_connected_triangles()
        ids, counts = np.asarray(ids), np.asarray(counts)
        V, F, C = np.asarray(m.vertices), np.asarray(m.triangles), np.asarray(m.vertex_colors)
        drop = np.zeros(len(counts), bool)
        for c in np.flatnonzero(counts >= min_tris):
            vv = np.unique(F[ids == c])
            P = V[vv]
            thick, n = _pca_thickness(P)
            if thick > 0.06 or abs(n[1]) > 0.2:
                continue
            for w in lay["walls"]:
                if w["id"] not in wall_textures:
                    continue
                a = 0 if w["axis"] == "x" else 2
                if abs(n[a]) < 0.9:
                    continue
                d = (P[:, a] - w["offset"]) * w["sign"]
                if not (0 < np.median(d) < max_dist):
                    continue
                o_, du, dv, L, Hh, _ = wall_frame(w, lay["ceiling_h"])
                img = wall_textures[w["id"]]
                u = ((P - o_) @ du * res).astype(int)
                v = ((P - o_) @ dv * res).astype(int)
                ok = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
                if ok.mean() < 0.8:
                    continue
                wl = cv2.cvtColor(img[v[ok], u[ok]][None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(float)
                cl = cv2.cvtColor((np.clip(C[vv][ok], 0, 1) * 255).astype(np.uint8)[None],
                                  cv2.COLOR_RGB2LAB)[0].astype(float)
                dE = np.linalg.norm((np.median(cl, 0) - np.median(wl, 0)) * [100 / 255, 1, 1])
                if dE < max_dE:
                    drop[c] = True
                    removed += 1
                    break
        if drop.any():
            m = _submesh(m, ~drop[ids])
            m.compute_vertex_normals()
            o["mesh"] = m
            if len(m.triangles) < 300:
                continue
            P = np.asarray(m.vertices)
            o["bbox_min"], o["bbox_max"] = P.min(0).tolist(), P.max(0).tolist()
            o["size_m"] = np.ptp(P, axis=0).round(3).tolist()
            o["triangles"] = len(m.triangles)
        keep_objs.append(o)
    return keep_objs, removed


def _extents(P):
    """PCA extents (largest first) of a point set."""
    c = P - P.mean(0)
    sub = c[np.random.default_rng(0).choice(len(c), min(len(c), 5000), replace=False)]
    ax = np.linalg.svd(sub, full_matrices=False)[2]
    q = c @ ax.T
    return np.sort(np.percentile(q, 97, axis=0) - np.percentile(q, 3, axis=0))[::-1]


def remove_junk(objects):
    """Drop reconstruction artefacts that are not objects (applies to objects WITHOUT a
    semantic instance):
      * floor sheets: everything below 35 cm and mostly horizontal - reflections of a
        glossy floor that DA3 puts slightly above it
      * edge strips: long (> 1.2 m) and narrow (< 0.3 m) - wall / ceiling edges, frames
      * loose sheets: a single thin (< 6 cm) surface of > 0.5 m2 - misplaced wall pieces
    Returns (kept, removed)."""
    keep, removed = [], []
    for o in objects:
        if "instance" in o or o.get("is_door"):
            keep.append(o)
            continue
        m = o["mesh"]
        m.compute_vertex_normals()
        P, N = np.asarray(m.vertices), np.asarray(m.vertex_normals)
        e = _extents(P)
        why = None
        if P[:, 1].max() < 0.35 and np.mean(np.abs(N[:, 1]) > 0.7) > 0.5:
            why = "floor reflection sheet"
        elif e[0] > 1.2 and e[1] < 0.3:
            why = "edge strip"
        elif e[2] < 0.06 and e[0] * e[1] > 0.5:
            why = "loose sheet"
        if why:
            o["removed_reason"] = why
            removed.append(o)
        else:
            keep.append(o)
    return keep, removed


def trim_to_body(o, eps=0.06, keep_frac=0.15, dist=0.05):
    """Keep only the spatially coherent main body of an instance mesh (strips / sheets that
    were attached through the reconstruction are cut off)."""
    from scipy.spatial import cKDTree
    m = o["mesh"]
    P = np.asarray(m.vertices)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.015)
    X = np.asarray(pc.points)
    lab = np.asarray(pc.cluster_dbscan(eps=eps, min_points=3))
    if lab.max() < 0:
        return o
    cnt = np.bincount(lab[lab >= 0])
    body = X[np.isin(lab, np.flatnonzero(cnt >= keep_frac * cnt.max()))]
    d, _ = cKDTree(body).query(P)
    keepv = d < dist
    F = np.asarray(m.triangles)
    if keepv[F].all(1).sum() < 300:
        return o
    o["mesh"] = _submesh(m, keepv[F].all(1))
    o["mesh"].compute_vertex_normals()
    P = np.asarray(o["mesh"].vertices)
    o["bbox_min"], o["bbox_max"] = P.min(0).tolist(), P.max(0).tolist()
    o["size_m"] = np.ptp(P, axis=0).round(3).tolist()
    o["triangles"] = len(o["mesh"].triangles)
    o["height_m"], o["bottom_m"] = float(P[:, 1].max()), float(P[:, 1].min())
    return o


def finalize_objects(objects, lay, touch=0.06, share=0.3, min_tris=3000):
    """Final clean-up after instance segmentation and furniture completion.
      * a remainder (no semantic instance) touching an instance: part of that piece of
        furniture -> dropped if the instance was completed (the complete model replaces
        it), otherwise merged back into the instance
      * unrecognised pieces in the ceiling band, thin unrecognised slabs at wall height,
        small or sheet-like unrecognised fragments -> dropped (they stay in the full
        fused mesh output, 04_geometry)
    Returns (kept, log lines)."""
    from scipy.spatial import cKDTree
    h = lay["ceiling_h"]
    inst = [o for o in objects if "instance" in o]
    trees = [(o, cKDTree(np.asarray(o["mesh"].vertices)[::3])) for o in inst]
    keep, notes = list(inst), {"merged into furniture": 0, "absorbed by completed furniture": 0,
                               "ceiling band": 0, "wall-height slab": 0, "small / sheet fragment": 0}
    for o in objects:
        if "instance" in o:
            continue
        if o.get("is_door"):
            keep.append(o)
            continue
        P = np.asarray(o["mesh"].vertices)
        best, bshare = None, 0.0
        for q, t in trees:
            d, _ = t.query(P[:: max(1, len(P) // 4000)], distance_upper_bound=touch)
            s_ = float(np.isfinite(d).mean())
            if s_ > bshare:
                best, bshare = q, s_
        if best is not None and bshare >= share:
            if best.get("generated_glb"):
                notes["absorbed by completed furniture"] += 1
            else:
                best["mesh"] = best["mesh"] + o["mesh"]
                best["mesh"].compute_vertex_normals()
                best["triangles"] = len(best["mesh"].triangles)
                notes["merged into furniture"] += 1
            continue
        e = _extents(P)
        if P[:, 1].min() > h - 0.6:
            notes["ceiling band"] += 1
            continue
        if P[:, 1].min() > 1.0 and e[2] < 0.2:
            notes["wall-height slab"] += 1
            continue
        if len(o["mesh"].triangles) < min_tris or e[1] < 0.2 or e[2] < 0.05:
            notes["small / sheet fragment"] += 1
            continue
        keep.append(o)
    keep.sort(key=lambda o: -o["triangles"])
    return keep, {k: v for k, v in notes.items() if v}
