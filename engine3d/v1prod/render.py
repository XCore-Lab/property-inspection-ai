"""Headless ray-cast renderer (open3d RaycastingScene) for textured / vertex-coloured
trimesh geometry. Exact pinhole intrinsics (any principal point), unlit (the
lighting is in the photo textures), optional back-face culling (dollhouse view:
walls facing away from the viewer disappear) and optional simple shading."""

import numpy as np
import open3d as o3d
import trimesh


class Part:
    """One mesh with a colour source: texture (uv + image) or vertex colours."""

    def __init__(self, tm):
        self.V = np.asarray(tm.vertices, np.float64)
        self.F = np.asarray(tm.faces, np.int64)
        self.tex = self.uv = self.vc = None
        vis = tm.visual
        if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None and vis.material is not None:
            img = getattr(vis.material, "baseColorTexture", None) or getattr(vis.material, "image", None)
            if img is not None:
                self.tex = np.asarray(img.convert("RGB"), np.float32)
                self.uv = np.asarray(vis.uv, np.float64)
        if self.tex is None:
            try:
                self.vc = np.asarray(tm.visual.vertex_colors, np.float32)[:, :3]
            except Exception:
                self.vc = np.full((len(self.V), 3), 180, np.float32)
        e1, e2 = self.V[self.F[:, 1]] - self.V[self.F[:, 0]], self.V[self.F[:, 2]] - self.V[self.F[:, 0]]
        n = np.cross(e1, e2)
        self.fn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _sample_tex(tex, uv):
    H, W = tex.shape[:2]
    x = np.clip(uv[:, 0] * (W - 1), 0, W - 1)
    y = np.clip((1 - uv[:, 1]) * (H - 1), 0, H - 1)       # trimesh uv: v up
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1 - fx) * (1 - fy) * tex[y0, x0] + fx * (1 - fy) * tex[y0, x1]
            + (1 - fx) * fy * tex[y1, x0] + fx * fy * tex[y1, x1])


class RayRenderer:
    def __init__(self, geoms, cull_backfaces=False):
        """geoms: list of trimesh.Trimesh (or a trimesh.Scene)."""
        if isinstance(geoms, trimesh.Scene):
            geoms = [g for g in geoms.dump() if isinstance(g, trimesh.Trimesh)]
        self.parts = [Part(g) for g in geoms if len(g.faces)]
        self.scene = o3d.t.geometry.RaycastingScene()
        self.ids = {}
        for i, p in enumerate(self.parts):
            gid = self.scene.add_triangles(o3d.core.Tensor(p.V.astype(np.float32)),
                                           o3d.core.Tensor(p.F.astype(np.uint32)))
            self.ids[gid] = i
        self.cull = cull_backfaces

    def _cast(self, rays):
        ans = self.scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        return (ans["t_hit"].numpy(), ans["geometry_ids"].numpy(), ans["primitive_ids"].numpy(),
                ans["primitive_uvs"].numpy())

    def render(self, K, w2c, W, H, bg=(30, 30, 33), shade=0.0, max_layers=4):
        """Returns (rgb uint8 HxWx3, depth z (0 = miss), hit mask)."""
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(np.asarray(K, np.float64)), o3d.core.Tensor(np.asarray(w2c, np.float64)),
            int(W), int(H)).numpy().reshape(-1, 6)
        n = len(rays)
        dirs = rays[:, 3:] / np.linalg.norm(rays[:, 3:], axis=1, keepdims=True)
        orig = rays[:, :3].copy()
        rays = np.c_[orig, dirs]
        t_acc = np.zeros(n)
        final = np.full(n, -1)
        gid_f = np.zeros(n, np.int64)
        pid_f = np.zeros(n, np.int64)
        uv_f = np.zeros((n, 2), np.float32)
        todo = np.arange(n)
        for layer in range(max_layers if self.cull else 1):
            t, g, p, uv = self._cast(rays[todo])
            hit = np.isfinite(t)
            todo_hit = todo[hit]
            if not len(todo_hit):
                break
            gh, ph = g[hit].astype(np.int64), p[hit].astype(np.int64)
            keep = np.ones(len(todo_hit), bool)
            if self.cull:
                fn = self._fn_batch(gh, ph)
                keep = (fn * rays[todo_hit, 3:]).sum(1) < 0
            acc = todo_hit[keep]
            t_acc[acc] += t[hit][keep]
            final[acc] = 1
            gid_f[acc], pid_f[acc], uv_f[acc] = gh[keep], ph[keep], uv[hit][keep]
            rej = todo_hit[~keep]
            if not len(rej):
                break
            step = t[hit][~keep] + 1e-4
            rays[rej, :3] += rays[rej, 3:] * step[:, None]
            t_acc[rej] += step
            todo = rej
        rgb = np.tile(np.array(bg, np.float32), (n, 1))
        hitm = final > 0
        idx = np.flatnonzero(hitm)
        for gid, pi in self.ids.items():
            sel = idx[gid_f[idx] == gid]
            if not len(sel):
                continue
            P = self.parts[pi]
            f = pid_f[sel]
            b1, b2 = uv_f[sel, 0:1], uv_f[sel, 1:2]
            b0 = 1 - b1 - b2
            tri = P.F[f]
            if P.tex is not None:
                uvp = b0 * P.uv[tri[:, 0]] + b1 * P.uv[tri[:, 1]] + b2 * P.uv[tri[:, 2]]
                col = _sample_tex(P.tex, uvp)
            else:
                col = b0 * P.vc[tri[:, 0]] + b1 * P.vc[tri[:, 1]] + b2 * P.vc[tri[:, 2]]
            if shade > 0:
                cosv = np.abs((P.fn[f] * dirs[sel]).sum(1))[:, None]
                col = col * ((1 - shade) + shade * cosv)
            rgb[sel] = col
        R = np.asarray(w2c)[:3, :3]
        z = np.zeros(n)
        z[hitm] = t_acc[hitm] * (dirs[hitm] @ R.T)[:, 2]
        return (np.clip(rgb, 0, 255).astype(np.uint8).reshape(H, W, 3), z.reshape(H, W),
                hitm.reshape(H, W))

    def _fn_batch(self, gh, ph):
        out = np.zeros((len(gh), 3))
        for gid, pi in self.ids.items():
            m = gh == gid
            if m.any():
                out[m] = self.parts[pi].fn[ph[m]]
        return out


def look_at(eye, target, up=(0, 1, 0)):
    """w2c (OpenCV camera: x right, y down, z forward)."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 0, -1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z])
    T[:3, 3] = -T[:3, :3] @ eye
    return T


def intrinsics(W, H, f):
    return np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])


def mesh_depths(mesh_o3d, K, w2c, W, H, frames=None):
    """Depth (z) of an open3d mesh rendered into each camera."""
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    out = []
    for k in (range(len(K)) if frames is None else frames):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K[k]), o3d.core.Tensor(w2c[k]), W, H)
        t = sc.cast_rays(rays)["t_hit"].numpy()
        d = rays.numpy()[..., 3:]
        z = t * (d @ w2c[k][:3, :3].T)[..., 2]
        z[~np.isfinite(z)] = 0
        out.append(z.astype(np.float32))
    return np.stack(out)
