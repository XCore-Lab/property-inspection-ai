"""Read a 3D engine SLAM run (data/slam/<name>/) into one object.

Everything is in the Production model frame: x right, y up, z back, floor y = 0,
approx. metres. Cameras are OpenCV (x right, y down, z forward), c2w = camera-to-model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SlamRun:
    root: Path
    image_size: tuple[int, int]                # (W, H) of the video frames
    video_frames: np.ndarray                   # (F,) source video frame index per SLAM frame
    times: np.ndarray                          # (F,) seconds
    K: np.ndarray                              # (F, 3, 3) at image_size
    c2w: np.ndarray                            # (F, 4, 4)
    pose_state: list[str]
    frame_quality: list[str]
    depth: np.ndarray                          # (F, h, w) float32 metres (filtered), 0 = invalid
    K_depth: np.ndarray                        # (F, 3, 3) at depth resolution
    layout: dict
    video_fps: float = 30.0
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.video_frames)

    @property
    def ceiling(self) -> float:
        return float(self.layout.get("ceiling_height_m", 2.6))

    def cam_centres(self) -> np.ndarray:
        return self.c2w[:, :3, 3]

    def glb(self, open_top: bool = True) -> Path:
        name = "room_slam_open_top.glb" if open_top else "room_slam.glb"
        return self.root / "02_slam_model" / name

    def frame_index_for_video_frame(self, vf: int) -> int:
        return int(np.argmin(np.abs(self.video_frames - vf)))


def load_slam_run(root: str | Path) -> SlamRun:
    root = Path(root)
    cams = json.loads((root / "05_trajectory" / "cameras.json").read_text())
    frames = cams["frames"]
    W, H = cams["image_size"]
    vf = np.array([f["video_frame"] for f in frames])
    K = np.array([f["K"] for f in frames], float)
    c2w = np.array([f["c2w"] for f in frames], float)
    states = [f.get("pose_state", "") for f in frames]

    quality = {}
    times = {}
    pcsv = root / "05_trajectory" / "poses_per_frame.csv"
    if pcsv.exists():
        with open(pcsv, newline="") as fh:
            for r in csv.DictReader(fh):
                quality[int(r["video_frame"])] = r.get("frame_quality", "")
                times[int(r["video_frame"])] = float(r["time_s"])
    cfg = {}
    if (root / "run_config.json").exists():
        cfg = json.loads((root / "run_config.json").read_text())
    video_fps = 30.0
    if times and len(times) > 1:
        v = sorted(times)
        if times[v[-1]] > 0:
            video_fps = v[-1] / times[v[-1]]

    d = np.load(root / "06_depth" / "depth_maps.npz")
    dfi = d["frame_index"]
    order = [int(np.where(dfi == x)[0][0]) for x in vf]      # align depth to camera frames
    depth = d["depth_filtered_m"][order].astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    Kd = d["K_processed"][order].astype(float)
    layout = json.loads((root / "03_floor_plan" / "layout.json").read_text())
    fm = load_floor_mask(root)
    if fm is not None:
        layout["_floor_mask"] = fm
    return SlamRun(root=root, image_size=(int(W), int(H)), video_frames=vf,
                   times=np.array([times.get(int(x), x / video_fps) for x in vf]),
                   K=K, c2w=c2w, pose_state=states, frame_quality=[quality.get(int(x), "") for x in vf],
                   depth=depth, K_depth=Kd, layout=layout, video_fps=video_fps, extras={"config": cfg})


APP = Path(__file__).resolve().parents[2]
PROD_PY = Path(sys.executable)      # same environment as the 3D engine
CACHE = APP / "data" / "slam_cache"


def load_floor_mask(root: Path) -> dict | None:
    """The 3D engine's floor raster (true floor region, not the room rectangles). model.pkl
    needs the engine's own classes, so it is exported once in a subprocess and cached under
    data/slam_cache (nothing is written into the engine output)."""
    pkl = root / "09_debug" / "work" / "model.pkl"
    if not pkl.exists():
        return None
    key = hashlib.md5(f"{root.resolve()}|{pkl.stat().st_mtime}".encode()).hexdigest()[:12]
    out = CACHE / f"{root.name}_{key}_floor.npz"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tool = APP / "engine3d" / "tools" / "export_floor_mask.py"
        if not PROD_PY.exists():
            return None
        r = subprocess.run([str(PROD_PY), str(tool), str(root), str(out)], capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            return None
    d = np.load(out)
    return {"floor": d["floor"], "lo": d["lo"], "cell": float(d["cell"])}
