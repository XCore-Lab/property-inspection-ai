"""Pose utilities, ground plane / metric scale, back-projection and TSDF.

Conventions
-----------
* DA3 extrinsics are world-to-camera (w2c), OpenCV camera axes
  (x right, y down, z forward).
* All poses in this package are stored as 4x4 w2c matrices.
* Relative motion i -> j (OpenCV convention, x_j = R x_i + t) is
  T_ji = w2c_j @ inv(w2c_i).
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------
# Pose helpers
# ----------------------------------------------------------------------

def to44(ext):
    e = np.eye(4, dtype=np.float64)
    e[:ext.shape[0], :4] = ext
    return e


def to44_batch(exts):
    return np.stack([to44(e) for e in exts])


def c2w_of(w2c):
    return np.linalg.inv(w2c)


def centers(w2c_all):
    return np.stack([c2w_of(T)[:3, 3] for T in w2c_all])


def relative(w2c_i, w2c_j):
    """T_ji: maps camera-i coordinates to camera-j coordinates."""
    return w2c_j @ np.linalg.inv(w2c_i)


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def vec_angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def average_rotations(Rs, weights):
    """Weighted chordal mean of rotation matrices."""
    return Rotation.from_matrix(np.asarray(Rs)).mean(weights=np.asarray(weights)).as_matrix()


def scale_intrinsics(K, sx, sy):
    S = np.diag([sx, sy, 1.0])
    return S @ K


# ----------------------------------------------------------------------
# Back-projection
# ----------------------------------------------------------------------

def pixel_grid(H, W):
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3).T.astype(np.float64)


def backproject_frame(depth, K, w2c, mask=None):
    """Return world points (M,3) and flat pixel indices for valid pixels."""
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    idx = np.flatnonzero(valid.reshape(-1))
    pix = pixel_grid(H, W)[:, idx]
    Xc = (np.linalg.inv(K) @ pix) * depth.reshape(-1)[idx][None]
    c2w = c2w_of(w2c)
    Xw = (c2w[:3, :3] @ Xc + c2w[:3, 3:4]).T
    return Xw, idx


def backproject_all(depth, K_all, w2c_all, images, masks, frames=None):
    frames = range(len(depth)) if frames is None else frames
    pts, cols, fid = [], [], []
    for i in frames:
        Xw, idx = backproject_frame(depth[i], K_all[i], w2c_all[i], masks[i])
        pts.append(Xw.astype(np.float32))
        cols.append(images[i].reshape(-1, 3)[idx].astype(np.float64) / 255.0)
        fid.append(np.full(len(idx), i, np.int32))
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3)), np.zeros(0, np.int32)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(fid)


def clean_pointcloud(points, colors, voxel=0.01, nb=20, std=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if voxel > 0 and len(pcd.points) > 10000:
        pcd = pcd.voxel_down_sample(voxel)
    n_before = len(pcd.points)
    if n_before > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    outlier_pct = 100.0 * (1 - len(pcd.points) / max(n_before, 1))
    return pcd, outlier_pct


def tsdf_fusion(depth, K_all, w2c_all, images, frames, masks=None,
                voxel=0.02, trunc=0.08, max_depth=8.0, weights=None):
    """TSDF over selected frames. Needs open3d 0.19 (0.20 ScalableTSDF is broken).

    weights[i] = integer integration weight (a frame integrated twice counts
    double in the running TSDF average)."""
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    H, W = depth.shape[1:]
    for i in frames:
        d = depth[i].astype(np.float32).copy()
        bad = ~np.isfinite(d) | (d <= 0) | (d > max_depth)
        if masks is not None:
            bad |= ~masks[i]
        d[bad] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(images[i].astype(np.uint8))),
            o3d.geometry.Image(np.ascontiguousarray(d)),
            depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
        K = K_all[i]
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, float(K[0, 0]), float(K[1, 1]),
                                                 float(K[0, 2]), float(K[1, 2]))
        for _ in range(1 if weights is None else max(1, int(weights[i]))):
            vol.integrate(rgbd, intr, w2c_all[i])
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ----------------------------------------------------------------------
# Ground plane + metric scale
# ----------------------------------------------------------------------

def estimate_floor_plane(points, c2w_rots, iters=500, seed=0):
    """RANSAC floor fit among the lowest points.

    Returns (n, d, inlier_ratio) with height(p) = p @ n + d, n pointing up.
    """
    up_guess = -np.mean([R[:, 1] for R in c2w_rots], axis=0)
    up_guess /= np.linalg.norm(up_guess)
    h = points @ up_guess
    low = points[h < np.percentile(h, 30)].astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(low) > 200_000:
        low = low[rng.choice(len(low), 200_000, replace=False)]
    thresh = 0.01 * (np.percentile(h, 99) - np.percentile(h, 1))

    best_count, best = 0, None
    for _ in range(iters):
        s = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ up_guess < 0:
            n = -n
        if n @ up_guess < 0.9:
            continue
        d = -n @ s[0]
        count = np.count_nonzero(np.abs(low @ n + d) < thresh)
        if count > best_count:
            best_count, best = count, (n, d)

    if best is None:
        return up_guess, -np.percentile(h, 2), 0.0
    n, d = best
    inl = low[np.abs(low @ n + d) < thresh]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][2]
    if n @ up_guess < 0:
        n = -n
    return n, -n @ c, best_count / len(low)


class GroundFrame:
    """Gravity-aligned map frame: x = right, y = forward (first heading)."""

    def __init__(self, normal, offset, forward_hint):
        self.up = normal / np.linalg.norm(normal)
        self.d = float(offset)
        f = forward_hint - (forward_hint @ self.up) * self.up
        self.fwd = f / np.linalg.norm(f)
        self.right = np.cross(self.fwd, self.up)
        self.origin = -self.d * self.up

    def height(self, P):
        return np.asarray(P) @ self.up + self.d

    def to_2d(self, P):
        Q = np.asarray(P) - self.origin
        return np.stack([Q @ self.right, Q @ self.fwd], axis=-1)

    def dir_2d(self, v):
        return np.array([v @ self.right, v @ self.fwd])

    def to_3d(self, xy, h=0.0):
        xy = np.atleast_2d(xy)
        return self.origin + xy[:, :1] * self.right + xy[:, 1:2] * self.fwd + h * self.up

    @property
    def R_align(self):
        return np.stack([self.right, self.up, -self.fwd])

    def align(self, P):
        """World -> export frame (x right, y up, z back), floor at y = 0."""
        Q = np.asarray(P, dtype=np.float64) - self.origin
        return Q @ self.R_align.T


def remove_small_components(mesh, min_triangles=300):
    """Drop floating mesh fragments (connected components with few triangles)."""
    if len(mesh.triangles) == 0:
        return mesh, 0
    ids, counts, _ = mesh.cluster_connected_triangles()
    ids, counts = np.asarray(ids), np.asarray(counts)
    remove = counts[ids] < min_triangles
    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_triangles_by_mask(remove)
    m.remove_unreferenced_vertices()
    return m, int(remove.sum())


def surface_coverage(ref_mesh, mesh, tol=0.05, n=200_000):
    """% of ref_mesh surface lying within tol of mesh (uniformly sampled)."""
    if len(mesh.triangles) == 0 or len(ref_mesh.triangles) == 0:
        return 0.0
    a = ref_mesh.sample_points_uniformly(n)
    b = mesh.sample_points_uniformly(n)
    d = np.asarray(a.compute_point_cloud_distance(b))
    return float(np.mean(d < tol) * 100)
