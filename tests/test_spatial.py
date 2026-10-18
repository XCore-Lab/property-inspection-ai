"""Coverage / localization on a synthetic 4 x 4 m room with analytic depth maps."""
from pathlib import Path

import numpy as np

from backend.spatial.coverage import SURF_WALL, compute_coverage, make_samples, mark_inspected, select_keyframes
from backend.spatial.localize import box_to_world, snap_to_surface
from backend.spatial.slam_data import SlamRun

LAYOUT = {
    "ceiling_height_m": 2.5,
    "walls": [  # room x in [0,4], z in [0,4]; sign = normal into the room
        {"id": "w_west", "axis": "x", "offset": 0.0, "sign": 1, "t0": 0.0, "t1": 4.0},
        {"id": "w_east", "axis": "x", "offset": 4.0, "sign": -1, "t0": 0.0, "t1": 4.0},
        {"id": "w_north", "axis": "z", "offset": 0.0, "sign": 1, "t0": 0.0, "t1": 4.0},
        {"id": "w_south", "axis": "z", "offset": 4.0, "sign": -1, "t0": 0.0, "t1": 4.0},
    ],
    "rooms": [{"id": "r1", "name": "Room", "bbox_m": [0, 0, 4, 4], "area_m2": 16, "size_m": [4, 4]}],
    "objects": [],
}
W, H = 160, 120
K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])


def cam_facing_north(x: float, z: float) -> np.ndarray:
    """Camera at (x, 1.5, z) looking along -z (towards the north wall at z = 0)."""
    c2w = np.eye(4)
    c2w[:3, 0] = [-1, 0, 0]     # camera x (right) -> world -x when looking along -z
    c2w[:3, 1] = [0, -1, 0]     # camera y (down)  -> world -y
    c2w[:3, 2] = [0, 0, -1]     # camera z (fwd)   -> world -z
    c2w[:3, 3] = [x, 1.5, z]
    return c2w


def run_for(c2ws: list[np.ndarray], block: float | None = None) -> SlamRun:
    """Depth = distance along the optical axis to the north wall (z = 0); optional
    obstacle plane at z = block covering the left half of the image."""
    n = len(c2ws)
    depth = np.zeros((n, H, W), np.float32)
    for i, c in enumerate(c2ws):
        depth[i] = c[2, 3]                     # camera z = distance to the z = 0 wall (fronto-parallel)
        if block is not None:
            depth[i][:, : W // 2] = c[2, 3] - block
    return SlamRun(root=Path("."), image_size=(W, H), video_frames=np.arange(n), times=np.arange(n) / 10.0,
                   K=np.repeat(K[None], n, 0), c2w=np.array(c2ws), pose_state=["VALID"] * n,
                   frame_quality=["GOOD"] * n, depth=depth, K_depth=np.repeat(K[None], n, 0), layout=LAYOUT)


def test_samples_cover_layout():
    S = make_samples(LAYOUT, 0.1)
    assert (S.kind == SURF_WALL).sum() == 4 * 40 * 25
    assert S.groups["w_north"]["label"].startswith("north")
    assert S.groups["r1"]["area_m2"] == 16.0


def test_seen_vs_inspectable_and_unseen_walls():
    run = run_for([cam_facing_north(2.0, 2.5)])         # 2.5 m from the north wall
    S = make_samples(LAYOUT, 0.1)
    cov = compute_coverage(run, S, good_range=3.0)
    st = cov.stats()["by_group"]
    assert st["w_north"]["seen_pct"] > 30 and st["w_north"]["inspectable_pct"] > 30
    assert st["w_south"]["seen_pct"] == 0              # behind the camera
    far = compute_coverage(run_for([cam_facing_north(2.0, 3.9)]), S, good_range=3.0).stats()["by_group"]
    assert far["w_north"]["seen_pct"] > 0 and far["w_north"]["inspectable_pct"] == 0   # 3.9 m > 3 m


def test_occlusion_hides_wall():
    S = make_samples(LAYOUT, 0.1)
    free = compute_coverage(run_for([cam_facing_north(2.0, 2.0)]), S)
    blocked = compute_coverage(run_for([cam_facing_north(2.0, 2.0)], block=1.0), S)
    assert blocked.stats()["by_group"]["w_north"]["seen_pct"] < free.stats()["by_group"]["w_north"]["seen_pct"]


def test_live_state_and_keyframes():
    run = run_for([cam_facing_north(1.0, 2.0), cam_facing_north(3.0, 2.0), cam_facing_north(3.0, 2.0)])
    S = make_samples(LAYOUT, 0.1)
    cov = compute_coverage(run, S)
    assert (cov.state(0) >= 2).sum() < (cov.state(1) >= 2).sum()      # coverage grows over time
    kf = select_keyframes(cov, np.ones(run.n), min_gap=1, min_frames=0)
    assert 1 <= len(kf) <= 2 and not {1, 2} <= set(kf)                 # frames 1 and 2 are the same view
    mark_inspected(cov, kf)
    assert cov.stats()["overall"]["inspected_pct"] > 0
    assert any(g["surface"] == "w_south" for g in cov.gaps())


def test_box_to_world_lands_on_wall():
    run = run_for([cam_facing_north(2.0, 2.0)])
    S = make_samples(LAYOUT, 0.1)
    p, d = box_to_world(run, 0, (70, 50, 90, 70), (W, H))            # image centre
    assert abs(d - 2.0) < 1e-3 and abs(p[2]) < 0.02 and abs(p[0] - 2.0) < 0.05 and abs(p[1] - 1.5) < 0.05
    surf, _, dist = snap_to_surface(p, S)
    assert surf == "w_north" and dist < 0.1
