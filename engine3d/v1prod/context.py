"""Context fill: no black holes where the video saw something.

The clean model (planar walls, floor, ceiling, objects) deliberately leaves out geometry
that is not room structure or a piece of furniture: the double-height stairwell above
the ceiling line, rooms seen only through an opening, spaces outside the walked floor.
Seen from the video cameras such places render as empty (black). Here every video camera
renders the clean model and the full measured reconstruction; wherever the clean model
shows nothing but the reconstruction has a surface, that measured surface is added
(triangles hit in >= `min_hits` views, grown by one ring, small fragments dropped,
reflections below the floor excluded).
"""

import numpy as np
import open3d as o3d
import trimesh

from . import render


def log(msg):
    print(msg, flush=True)


def fill_gaps(clean_geoms, raw_o3d, st, max_frames=70, scale=0.25, min_hits=1, min_tris=150, rings=3):
    frames = list(st["usable"])
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    H, W = st["frames"][0].shape[:2]
    Ws, Hs = int(W * scale), int(H * scale)
    Rc = render.RayRenderer(clean_geoms)
    raw = o3d.t.geometry.RaycastingScene()
    raw.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(raw_o3d))
    F = np.asarray(raw_o3d.triangles)
    hits = np.zeros(len(F), np.int32)
    for k in frames:
        K = st["K_full"][k].copy()
        K[:2] *= scale
        K[0, 2] = (st["K_full"][k][0, 2] + 0.5) * scale - 0.5
        K[1, 2] = (st["K_full"][k][1, 2] + 0.5) * scale - 0.5
        _, _, hit = Rc.render(K, st["w2c_model"][k], Ws, Hs)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(K), o3d.core.Tensor(st["w2c_model"][k]), Ws, Hs)
        ans = raw.cast_rays(rays)
        pid = ans["primitive_ids"].numpy()
        ok = np.isfinite(ans["t_hit"].numpy()) & ~hit
        np.add.at(hits, pid[ok].astype(np.int64), 1)
    keep = hits >= min_hits
    if not keep.any():
        return None
    # grow by `rings` rings (fills pin-holes between hit triangles)
    for _ in range(rings):
        vk = np.zeros(len(np.asarray(raw_o3d.vertices)), bool)
        vk[F[keep].ravel()] = True
        keep |= vk[F].sum(1) >= 1
    V = np.asarray(raw_o3d.vertices)
    keep &= V[F].min(1)[:, 1] > -0.05                 # glossy-floor reflections below the floor
    m = o3d.geometry.TriangleMesh(raw_o3d)
    m.remove_triangles_by_mask(~keep)
    m.remove_unreferenced_vertices()
    ids, cnt, _ = m.cluster_connected_triangles()
    ids, cnt = np.asarray(ids), np.asarray(cnt)
    m.remove_triangles_by_mask(cnt[ids] < min_tris)
    m.remove_unreferenced_vertices()
    if len(m.triangles) == 0:
        return None
    # (open3d's tensor fill_holes crashes on large context meshes - ring growth is used instead)
    tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles),
                         vertex_colors=(np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8),
                         process=False)
    log(f"  context fill: {len(tm.faces):,} measured triangles added where the clean model was empty "
        f"(seen in >= {min_hits} of {len(frames)} views)")
    return tm
