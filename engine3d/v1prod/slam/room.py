"""Room-model helpers for build_room_model.py.

* full-resolution depth (OpenCV joint-bilateral upsampling of DA3 depth)
* Manhattan plane regularisation (floor / ceiling / walls snapped flat)
* multi-view consistency (grazing-angle mask, frame-to-model alignment,
  free-space carving) and floor-hole filling
* legacy open3d hidden-window rendering (Filament headless is not on Windows)

All geometry here works in the ALIGNED frame (x right, y up, z back, floor
y = 0, approx. metres) unless stated otherwise.
"""

import cv2
import numpy as np
import open3d as o3d


# ----------------------------------------------------------------------
# Depth upsampling
# ----------------------------------------------------------------------

def upsample_depth(d_lo, v_lo, rgb_full, max_change=0.03):
    """DA3 depth (processed res) -> full frame res, edge-aware.

    Normalised bilinear interpolation (invalid pixels never bleed in), then a
    joint bilateral filter guided by the full-res RGB; changes > max_change
    of the depth are rejected so depth edges are not smeared."""
    H, W = rgb_full.shape[:2]
    v = v_lo.astype(np.float32)
    num = cv2.resize(np.where(v_lo, d_lo, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    den = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)
    d = num / np.maximum(den, 1e-6)
    valid = cv2.resize(v_lo.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    valid &= den > 0.99                     # drop the interpolated rim around holes
    fill = np.where(valid, d, float(np.median(d[valid])) if valid.any() else 1.0).astype(np.float32)
    sm = cv2.ximgproc.jointBilateralFilter(rgb_full.astype(np.float32), fill, d=9,
                                           sigmaColor=18, sigmaSpace=4)
    ok = valid & (np.abs(sm - fill) / np.maximum(fill, 1e-6) < max_change)
    out = np.where(ok, sm, fill)
    out[~valid] = 0
    return out.astype(np.float32), valid


# ----------------------------------------------------------------------
# Manhattan plane regularisation
# ----------------------------------------------------------------------

def _peaks(x, w, lo, hi, bin_=0.01, min_w=0.0, sep=0.12):
    """1-D weighted histogram peaks (plane offsets)."""
    if len(x) == 0:
        return []
    bins = np.arange(lo, hi + bin_, bin_)
    h, e = np.histogram(x, bins, weights=w)
    h = np.convolve(h, np.ones(5) / 5, mode="same")
    c = (e[:-1] + e[1:]) / 2
    order = np.argsort(-h)
    out = []
    for i in order:
        if h[i] * 5 < min_w:   # smoothed -> total mass in +-2 bins
            break
        if all(abs(c[i] - p) > sep for p, _ in out):
            out.append((float(c[i]), float(h[i] * 5)))
    return out


def manhattan_yaw(V, N):
    """Dominant wall direction (radians, mod 90 deg) from vertical-ish normals."""
    wall = np.abs(N[:, 1]) < 0.3
    a = np.arctan2(N[wall, 2], N[wall, 0])
    h, e = np.histogram(np.mod(a, np.pi / 2), 180, range=(0, np.pi / 2))
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(7) / 7, mode="same")[3:-3]
    return float((e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2)


def _footprint_mask(uv, cell=0.1):
    """Occupancy raster of a plane's inlier support (for snapping only inside it)."""
    lo = uv.min(0) - cell
    ij = np.floor((uv - lo) / cell).astype(int)
    shp = ij.max(0) + 2
    m = np.zeros(shp, np.uint8)
    m[ij[:, 0], ij[:, 1]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 6      # >= 0.06 m2 patches
    return lo, cell, keep[lab]


def _in_footprint(fp, uv):
    lo, cell, m = fp
    ij = np.floor((uv - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < m.shape[0]) & (ij[:, 1] < m.shape[1])
    r = np.zeros(len(uv), bool)
    r[ok] = m[ij[ok, 0], ij[ok, 1]]
    return r


def regularise_planes(V, N, tri_area_v, snap_full=0.025, snap_zero=0.06, min_area=0.6):
    """Snap near-planar vertices onto dominant Manhattan planes.

    V, N: aligned vertices / normals (already rotated so walls are x/z aligned).
    tri_area_v: per-vertex area (m2) for plane support.
    Returns (V_new, planes) where planes = list of dict(axis, offset, sign, area).
    Snapping is soft: full within snap_full, fading to zero at snap_zero, and only
    for vertices whose normal agrees with the plane and that lie inside the
    plane's observed footprint (so furniture near a wall is left alone)."""
    V = V.copy()
    planes = []
    lo, hi = V.min(0) - 0.1, V.max(0) + 0.1
    for axis in (1, 0, 2):   # floor/ceiling first, then walls
        for sign in (1, -1):
            nmask = N[:, axis] * sign > 0.8
            for off, area in _peaks(V[nmask, axis], tri_area_v[nmask], lo[axis], hi[axis],
                                    min_w=min_area):
                d = V[:, axis] - off
                sup = nmask & (np.abs(d) < snap_full)
                if tri_area_v[sup].sum() < min_area:
                    continue
                other = [a for a in (0, 1, 2) if a != axis]
                fp = _footprint_mask(V[sup][:, other])
                cand = (N[:, axis] * sign > 0.6) & (np.abs(d) < snap_zero)
                cand &= _in_footprint(fp, V[:, other])
                w = np.clip((snap_zero - np.abs(d[cand])) / (snap_zero - snap_full), 0, 1)
                V[cand, axis] -= w * d[cand]
                planes.append({"axis": "xyz"[axis], "offset": off, "normal_sign": sign,
                               "area_m2": float(tri_area_v[sup].sum()),
                               "snapped_vertices": int(cand.sum())})
    return V, planes


def vertex_areas(V, F):
    a = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    out = np.zeros(len(V))
    for k in range(3):
        np.add.at(out, F[:, k], a / 3)
    return out


def make_mesh(V, F, C):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.vertex_colors = o3d.utility.Vector3dVector(C)
    m.compute_vertex_normals()
    return m


def crop_mesh(mesh, keep_v):
    m = o3d.geometry.TriangleMesh(mesh)
    F = np.asarray(m.triangles)
    m.remove_triangles_by_mask(~keep_v[F].all(1))
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


# ----------------------------------------------------------------------
# Rendering (legacy Visualizer, hidden window)
# ----------------------------------------------------------------------

class Renderer:
    def __init__(self, W, H, bg=(0.12, 0.12, 0.13), light=False, back_faces=False):
        self.W, self.H = W, H
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=W, height=H, visible=False)
        ro = self.vis.get_render_option()
        ro.background_color = np.asarray(bg)
        ro.light_on = light
        ro.mesh_show_back_face = back_faces   # False -> walls facing away vanish (dollhouse)
        self._geoms = []

    def set_geometries(self, geoms):
        for g in self._geoms:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self._geoms = list(geoms)
        for g in self._geoms:
            self.vis.add_geometry(g, reset_bounding_box=True)

    def render(self, K, w2c):
        """K: 3x3 (cx, cy must be W/2-0.5, H/2-0.5), w2c: 4x4 OpenCV camera."""
        vc = self.vis.get_view_control()
        p = o3d.camera.PinholeCameraParameters()
        p.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.W, self.H, K[0, 0], K[1, 1],
                                                        self.W / 2 - 0.5, self.H / 2 - 0.5)
        p.extrinsic = w2c
        vc.convert_from_pinhole_camera_parameters(p, allow_arbitrary=True)
        vc.set_constant_z_near(0.05)
        vc.set_constant_z_far(50.0)
        self.vis.poll_events()
        self.vis.update_renderer()
        img = np.asarray(self.vis.capture_screen_float_buffer(True))
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def close(self):
        self.vis.destroy_window()


def look_at(eye, target, up=(0, 1, 0)):
    """w2c (OpenCV: x right, y down, z forward) looking from eye to target."""
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 0, -1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


# ----------------------------------------------------------------------
# Multi-view consistency: grazing-angle mask, frame-to-model alignment,
# free-space carving
# ----------------------------------------------------------------------

def grazing_mask(depth, K, min_cos=0.2):
    """False where the camera ray hits the surface at a grazing angle
    (> ~78 deg): those pixels produce the streaky 'comb' artefacts."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ray = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
    X = ray * depth[..., None]
    du = np.zeros_like(X)
    dv = np.zeros_like(X)
    du[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dv[1:-1] = X[2:] - X[:-2]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1)
    cos = np.abs((n * ray).sum(-1)) / np.maximum(nn * np.linalg.norm(ray, axis=-1), 1e-9)
    return (cos >= min_cos) | (nn == 0)


def _model_depth(scene, K, w2c, W, H):
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        o3d.core.Tensor(K), o3d.core.Tensor(w2c), W, H)
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy()
    d = rays.numpy()[..., 3:]
    z = t * (d @ w2c[:3, :3].T)[..., 2]      # distance along ray -> camera z
    z[~np.isfinite(z)] = 0
    return z


def _world_points(depth, valid, K, w2c, step=4):
    H, W = depth.shape
    vs, us = np.mgrid[0:H:step, 0:W:step]
    d = depth[vs, us]
    ok = valid[vs, us] & (d > 0)
    ray = np.stack([(us[ok] - K[0, 2]) / K[0, 0], (vs[ok] - K[1, 2]) / K[1, 1], np.ones(ok.sum())], 1)
    Xc = ray * d[ok][:, None]
    c2w = np.linalg.inv(w2c)
    return Xc @ c2w[:3, :3].T + c2w[:3, 3]


def align_frames_to_model(depth, valid, K_all, w2c_all, frames, fuse, iters=2, sub=4,
                          max_t=0.15, max_r=4.0, log=print):
    """Per-frame depth scale + rigid pose correction against the fused model.

    Each iteration: fuse -> raycast the model into every frame -> robust depth
    scale (median model/frame ratio on consistent pixels) -> point-to-plane ICP
    of the frame's points to the model. Corrections outside the guards are
    rejected. Returns (scales, w2c_new, stats)."""
    N = len(depth)
    H, W = depth.shape[1:]
    scales = np.ones(N)
    w2c = w2c_all.copy()
    stats = []
    for it in range(iters):
        mesh = fuse(scales, w2c)
        tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tm)
        tgt = mesh.sample_points_uniformly(600_000)
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
        res_before, res_after, n_acc = [], [], 0
        for k in frames:
            Ks = K_all[k].copy()
            Ks[:2] /= sub
            Ks[0, 2] = (K_all[k][0, 2] + 0.5) / sub - 0.5
            Ks[1, 2] = (K_all[k][1, 2] + 0.5) / sub - 0.5
            Ws, Hs = W // sub, H // sub
            zm = _model_depth(scene, Ks, w2c[k], Ws, Hs)
            df = cv2.resize(depth[k] * scales[k], (Ws, Hs), interpolation=cv2.INTER_NEAREST)
            vf = cv2.resize(valid[k].astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST) > 0
            ok = vf & (zm > 0) & (df > 0)
            r = zm[ok] / df[ok]
            cons = np.abs(r - 1) < 0.15
            if cons.sum() < 200:
                continue
            res_before.append(np.median(np.abs(zm[ok][cons] - df[ok][cons])))
            s = float(np.median(r[cons]))
            if 0.85 < s < 1.15:
                scales[k] *= s
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                _world_points(depth[k] * scales[k], valid[k], K_all[k], w2c[k], step=sub)))
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, 0.06, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(k=0.03)),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T = reg.transformation
            dt = np.linalg.norm(T[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
            if reg.fitness > 0.35 and dt < max_t and dr < max_r:
                w2c[k] = w2c[k] @ np.linalg.inv(T)       # world point p -> T p
                n_acc += 1
                res_after.append(reg.inlier_rmse)
        st = {"iter": it + 1, "median_depth_residual_m": float(np.median(res_before)) if res_before else float("nan"),
              "icp_accepted": n_acc, "icp_rmse_m": float(np.median(res_after)) if res_after else float("nan"),
              "scale_range": [float(scales.min()), float(scales.max())]}
        log(f"    align iter {it+1}: median depth disagreement {st['median_depth_residual_m']*100:.1f} cm | "
            f"ICP accepted {n_acc}/{len(frames)} (rmse {st['icp_rmse_m']*100:.1f} cm) | "
            f"depth scale {scales.min():.3f}..{scales.max():.3f}")
        stats.append(st)
    return scales, w2c, stats


def carve_free_space(mesh, depth, valid, K_all, w2c_all, frames, sub=2, min_violate=2):
    """Remove surface that the cameras see through (ghost layers).

    For every vertex and frame: 'support' if the frame's depth agrees with the
    vertex depth, 'violation' if the frame sees clearly beyond it (so the vertex
    lies in observed free space). Vertices with violations >= min_violate and
    more violations than supports are removed."""
    V = np.asarray(mesh.vertices)
    sup = np.zeros(len(V), np.int32)
    vio = np.zeros(len(V), np.int32)
    H, W = depth.shape[1:]
    for k in frames:
        R, t = w2c_all[k][:3, :3], w2c_all[k][:3, 3]
        Xc = V @ R.T + t
        z = Xc[:, 2]
        front = z > 0.1
        u = np.full(len(V), -1)
        v = np.full(len(V), -1)
        u[front] = np.round(K_all[k][0, 0] * Xc[front, 0] / z[front] + K_all[k][0, 2]).astype(int)
        v[front] = np.round(K_all[k][1, 1] * Xc[front, 1] / z[front] + K_all[k][1, 2]).astype(int)
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.flatnonzero(inb)
        d = depth[k][v[idx], u[idx]]
        ok = valid[k][v[idx], u[idx]] & (d > 0)
        idx, d = idx[ok], d[ok]
        zz = z[idx]
        tol = np.maximum(0.04, 0.02 * zz)
        sup[idx[np.abs(d - zz) < tol]] += 1
        vio[idx[d > zz + np.maximum(0.12, 0.05 * zz)]] += 1
    bad = (vio >= min_violate) & (vio > sup)
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_triangles_by_mask(bad[np.asarray(m.triangles)].any(1))
    m.remove_unreferenced_vertices()
    return m, int(bad.sum())


def fill_floor_holes(mesh, cell=0.02, close_m=0.7, y=-0.003):
    """Fill enclosed, never-observed floor (behind / under furniture) on the
    fitted floor plane. Colours come from cv2.inpaint of the observed floor.
    Returns (mesh_with_fill, fill_mesh, inferred_area_m2)."""
    V, N, C = (np.asarray(a) for a in (mesh.vertices, mesh.vertex_normals, mesh.vertex_colors))
    floor = (np.abs(V[:, 1]) < 0.03) & (N[:, 1] > 0.7)
    low = ~floor & (V[:, 1] < 1.2)
    lo = V[floor][:, [0, 2]].min(0) - 0.2
    hi = V[floor][:, [0, 2]].max(0) + 0.2
    shp = tuple(np.ceil((hi - lo) / cell).astype(int) + 1)

    def raster(P):
        ij = np.floor((P[:, [0, 2]] - lo) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < shp[0]) & (ij[:, 1] < shp[1])
        return ij[ok], ok

    ij, ok = raster(V[floor])
    obs = np.zeros(shp, np.uint8)
    obs[ij[:, 0], ij[:, 1]] = 1
    col = np.zeros(shp + (3,), np.float64)
    cnt = np.zeros(shp)
    np.add.at(col, (ij[:, 0], ij[:, 1]), C[floor][ok])
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    col = (col / np.maximum(cnt, 1)[..., None] * 255).astype(np.uint8)
    ij2, _ = raster(V[low])
    fur = np.zeros(shp, np.uint8)
    fur[ij2[:, 0], ij2[:, 1]] = 1
    obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k = int(close_m / cell) | 1
    region = cv2.morphologyEx(obs | fur, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    fill = (region > 0) & (obs == 0)
    if not fill.any():
        return mesh, None, 0.0
    col = cv2.inpaint(col, (obs == 0).astype(np.uint8), 7, cv2.INPAINT_TELEA)
    ii, jj = np.nonzero(fill)
    x0, z0 = lo[0] + ii * cell, lo[1] + jj * cell
    x1, z1 = x0 + cell, z0 + cell
    quad = np.stack([np.c_[x0, np.full_like(x0, y), z0], np.c_[x1, np.full_like(x0, y), z0],
                     np.c_[x1, np.full_like(x0, y), z1], np.c_[x0, np.full_like(x0, y), z1]], 1)
    Vf = quad.reshape(-1, 3)
    b = np.arange(len(ii))[:, None] * 4
    Ff = np.concatenate([b + [0, 3, 2], b + [0, 2, 1]])      # +y facing
    Cf = np.repeat(col[ii, jj].astype(np.float64) / 255, 4, axis=0)
    fm = make_mesh(Vf, Ff, Cf).merge_close_vertices(cell * 0.01)
    fm.compute_vertex_normals()
    return mesh + fm, fm, float(fill.sum() * cell * cell)
