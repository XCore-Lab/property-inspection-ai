"""3D furniture instances from multi-view 2D instance masks (Grounding DINO + SAM).

The TSDF surfaces of touching furniture are connected (a sofa and its coffee table, two
armchairs side by side), so geometry alone cannot separate them. Instead:
  1. candidate points = the measured object geometry, voxelised (2 cm)
  2. every SAM mask of every frame -> the set of VISIBLE voxels (model-depth test) that
     project into it
  3. masks of the same category are clustered across frames by 3D overlap (containment
     of their voxel sets); two masks of the same frame never join (they are different
     instances in that image) -> one cluster = one physical object
  4. every voxel goes to the instance whose masks contain it most consistently (share of
     the instance's frames that see the voxel and put it inside the mask)
  5. the measured object meshes are split along these instance labels
Each instance also gets its best view (large, fully inside the image, least occluded) and
the SAM cut-out from that view: the input image for shape completion.
"""

import json
import os

import cv2
import numpy as np
import open3d as o3d
from scipy import sparse

GROUPS = {
    "sofa": ["sofa"],
    "seat": ["armchair", "chair", "stool", "bench"],
    "table": ["coffee table", "dining table", "table", "desk"],
    "storage": ["cupboard", "wardrobe", "bookshelf", "cabinet", "drawers", "shelf", "kitchen counter"],
    "bed": ["bed"],
    "fan": ["fan"],
    "plant": ["plant"],
    "lamp": ["lamp"],
    "tv": ["television"],
    "staircase": ["staircase"],
    "appliance": ["refrigerator", "washing machine", "stove", "air conditioner", "sink"],
    "bucket": ["bucket", "box", "bag"],
}
# categories that are part of / flat on the structure: never 3D instances
STRUCTURAL = {"door", "curtain", "picture frame", "mirror", "clock"}


def log(msg):
    print(msg, flush=True)


def canon(label):
    """'armchair chair' -> ('seat', 'armchair'): group + the most specific vocab word."""
    for g, words in GROUPS.items():
        for w in words:                              # words are ordered most specific first
            if w in label:
                return g, w
    return None, None


def load_masks(det_path, mask_path):
    d = json.load(open(det_path))["detections"]
    M = np.load(mask_path)
    shp = tuple(M["shape"])
    out = []
    for k, ds in d.items():
        for i, det in enumerate(ds):
            key = f"f{k}_{i}"
            if key not in M.files:
                continue
            g, w = canon(det["label"])
            if g is None:
                continue
            m = np.unpackbits(M[key])[:shp[0] * shp[1]].reshape(shp).astype(bool)
            out.append({"frame": int(k), "group": g, "word": w, "score": det["score"], "box": det["box"],
                        "mask": m})
    # same frame, same group, strongly overlapping masks -> keep the better one
    keep = []
    by_frame = {}
    for r in sorted(out, key=lambda r: -r["score"]):
        dup = False
        for q in by_frame.get((r["frame"], r["group"]), []):
            inter = (q["mask"] & r["mask"]).sum()
            if inter / max(min(q["mask"].sum(), r["mask"].sum()), 1) > 0.6:
                dup = True
                break
        if not dup:
            by_frame.setdefault((r["frame"], r["group"]), []).append(r)
            keep.append(r)
    return keep, shp


def segment(objects, st, det_path, mask_path, voxel=0.02, min_masks=3, contain=0.5, assign=0.35):
    from .render import mesh_depths
    masks, shp = load_masks(det_path, mask_path)
    if not masks:
        return [], None
    Hm, Wm = shp
    H, W = st["frames"][0].shape[:2]
    sx, sy = Wm / W, Hm / H
    # candidate voxels from all object meshes
    allV = np.vstack([np.asarray(o["mesh"].vertices) for o in objects])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(allV)).voxel_down_sample(voxel)
    X = np.asarray(pc.points)
    frames = sorted({m["frame"] for m in masks})
    K = st["K_full"][frames].copy()
    K[:, 0] *= sx
    K[:, 1] *= sy
    K[:, 0, 2] = (st["K_full"][frames][:, 0, 2] + 0.5) * sx - 0.5
    K[:, 1, 2] = (st["K_full"][frames][:, 1, 2] + 0.5) * sy - 0.5
    w2c = st["w2c_model"][frames]
    depth = mesh_depths(st["model"], K, w2c, Wm, Hm)
    fidx = {k: i for i, k in enumerate(frames)}
    vis = np.zeros((len(frames), len(X)), bool)
    uv = [None] * len(frames)
    for i in range(len(frames)):
        Xc = X @ w2c[i][:3, :3].T + w2c[i][:3, 3]
        z = Xc[:, 2]
        ok = z > 0.1
        u = np.full(len(X), -1)
        v = np.full(len(X), -1)
        u[ok] = np.round(K[i][0, 0] * Xc[ok, 0] / z[ok] + K[i][0, 2]).astype(int)
        v[ok] = np.round(K[i][1, 1] * Xc[ok, 1] / z[ok] + K[i][1, 2]).astype(int)
        ok &= (u >= 0) & (u < Wm) & (v >= 0) & (v < Hm)
        idx = np.flatnonzero(ok)
        dm = depth[i][v[idx], u[idx]]
        ok[idx] = (dm == 0) | (z[idx] < dm + 0.05)
        vis[i] = ok
        uv[i] = (u, v)
    rows, cols = [], []
    for j, m in enumerate(masks):
        i = fidx[m["frame"]]
        u, v = uv[i]
        idx = np.flatnonzero(vis[i])
        inside = idx[m["mask"][v[idx], u[idx]]]
        m["n"] = len(inside)
        rows += [j] * len(inside)
        cols += inside.tolist()
    Msp = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(masks), len(X)))
    inter = (Msp @ Msp.T).tocoo()
    size = np.array([m["n"] for m in masks])
    # union-find over same-group masks, never two masks of one frame in one instance
    parent = list(range(len(masks)))
    frames_of = [{m["frame"]} for m in masks]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for a, b, c in zip(inter.row, inter.col, inter.data):
        if a >= b or masks[a]["group"] != masks[b]["group"] or masks[a]["frame"] == masks[b]["frame"]:
            continue
        cont = c / max(min(size[a], size[b]), 1)
        if c >= 15 and cont > contain:
            pairs.append((cont, a, b))
    for _, a, b in sorted(pairs, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb or frames_of[ra] & frames_of[rb]:
            continue
        parent[rb] = ra
        frames_of[ra] |= frames_of[rb]
    clusters = {}
    for j in range(len(masks)):
        if size[j] >= 15:
            clusters.setdefault(find(j), []).append(j)
    inst = [c for c in clusters.values() if len(c) >= min_masks]
    # voxel -> instance
    score = np.zeros((len(inst), len(X)), np.float32)
    for n, c in enumerate(inst):
        fr = sorted({masks[j]["frame"] for j in c})
        seen = vis[[fidx[f] for f in fr]].sum(0)
        hit = np.asarray(Msp[c].sum(0)).ravel()
        score[n] = np.where(seen >= 2, hit / np.maximum(seen, 1), 0)
    best = score.argmax(0) if len(inst) else np.zeros(len(X), int)
    lab = np.where(score.max(0) >= assign if len(inst) else np.zeros(len(X), bool), best, -1)
    instances = []
    for n, c in enumerate(inst):
        words = [masks[j]["word"] for j in c]
        word = max(set(words), key=words.count)
        # best view: large, not cut by the image border, least occluded
        best_v, best_s = None, -1
        pts = np.flatnonzero(lab == n)
        for j in c:
            m = masks[j]
            x0, y0, x1, y1 = m["box"]
            if x0 < 4 or y0 < 4 or x1 > W - 4 or y1 > H - 4:
                continue
            i = fidx[m["frame"]]
            u, v = uv[i]
            pv = pts[vis[i][pts]]
            if len(pv) < 20:
                continue
            cont = m["mask"][v[pv], u[pv]].mean()
            s = m["mask"].mean() * cont ** 2 * m["score"]
            if s > best_s:
                best_v, best_s = j, s
        instances.append({"group": masks[c[0]]["group"], "label": word, "n_masks": len(c),
                          "frames": sorted({masks[j]["frame"] for j in c}), "voxels": int((lab == n).sum()),
                          "best_mask": best_v})
    log(f"  instance segmentation: {len(masks)} masks -> {len(instances)} 3D instances "
        f"({', '.join(i['label'] for i in instances)})")
    mask_instance = {j: n for n, c in enumerate(inst) for j in c}
    return instances, {"X": X, "lab": lab, "masks": masks, "shape": shp, "mask_instance": mask_instance}


def split_objects(objects, instances, seg, min_vertices=250, snap=0.04):
    """Split the measured object meshes along the instance labels."""
    from scipy.spatial import cKDTree
    from .objects import _submesh
    if not instances:
        return objects
    tree = cKDTree(seg["X"])
    out = []
    pieces = {}
    for o in objects:
        V = np.asarray(o["mesh"].vertices)
        d, j = tree.query(V, distance_upper_bound=snap)
        vl = np.full(len(V), -1)
        ok = np.isfinite(d)
        vl[ok] = seg["lab"][j[ok]]
        F = np.asarray(o["mesh"].triangles)
        tl = vl[F]
        # triangle label = majority of its vertices
        maj = np.where((tl[:, 0] == tl[:, 1]) | (tl[:, 0] == tl[:, 2]), tl[:, 0], tl[:, 1])
        labs, cnt = np.unique(maj[maj >= 0], return_counts=True)
        big = [l_ for l_, c in zip(labs, cnt) if c >= min_vertices]
        if not big:
            out.append(o)
            continue
        for l_ in big:
            pieces.setdefault(l_, []).append(_submesh(o["mesh"], maj == l_))
        rest = ~np.isin(maj, big)
        if rest.sum() >= 600:
            r = dict(o)
            r["mesh"] = _submesh(o["mesh"], rest)
            r["label"] = "object"
            out.append(r)
    for l_, parts in pieces.items():
        m = parts[0]
        for p in parts[1:]:
            m += p
        m.compute_vertex_normals()
        P = np.asarray(m.vertices)
        ins = instances[l_]
        out.append({"mesh": m, "label": ins["label"], "group": ins["group"], "instance": int(l_),
                    "center": ((P.min(0) + P.max(0)) / 2).tolist(),
                    "bbox_min": P.min(0).tolist(), "bbox_max": P.max(0).tolist(),
                    "height_m": float(P[:, 1].max()), "bottom_m": float(P[:, 1].min()),
                    "triangles": len(m.triangles), "ceiling_mounted": False,
                    "size_m": np.ptp(P, axis=0).round(3).tolist()})
    out.sort(key=lambda o: -o["triangles"])
    for i, o in enumerate(out):
        o["id"] = f"object_{i:02d}"
        o.setdefault("size_m", np.ptp(np.asarray(o["mesh"].vertices), axis=0).round(3).tolist())
    log(f"  objects after instance split: {len(out)} "
        f"({sum('instance' in o for o in out)} semantic instances)")
    return out


def input_cutout(frame_rgb, mask_half, size=512, margin=0.12):
    """SAM mask of the best view -> square RGBA cut-out on white (shape-generation input)."""
    H, W = frame_rgb.shape[:2]
    m = cv2.resize(mask_half.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + margin)
    rgba = np.dstack([frame_rgb, m * 255]).astype(np.uint8)
    pad = int(half) + 2
    rgba = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
    cx, cy = cx + pad, cy + pad
    crop = rgba[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop[crop[..., 3] < 128, :3] = 255
    return crop
