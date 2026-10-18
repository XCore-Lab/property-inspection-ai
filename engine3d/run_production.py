"""3D engine: room / house reconstruction from a walkthrough video (Depth Anything 3 + OpenCV SLAM).

One command, one video in, the complete organised output set out:

    python engine3d/run_production.py --video samples/room_walkthrough.mp4 --output-dir data/slam/room

Called by scripts/inspect_walkthrough.py. Object names come from Gemini (v1prod/labels.py).
"""

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))   # project root: the bundled depth_anything_3 package

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from v1prod import state_io  # noqa: E402

VERSION = "1.0.0-production"
STAGES = ["slam", "model", "export"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Version 1 Production reconstruction pipeline")
    ap.add_argument("--video", required=True)
    ap.add_argument("--output-dir", default=None, help="default: output/<video name>")
    ap.add_argument("--name", default=None, help="run name (default: video file name)")
    # capture / DA3
    ap.add_argument("--fps", type=float, default=10.0, help="frame sampling rate (10: more views per object = more complete furniture)")
    ap.add_argument("--max-frames", type=int, default=600, help="cap; sampling interval grows beyond it")
    ap.add_argument("--max-image-side", type=int, default=960,
                    help="downscale frames whose longer side is larger (memory with 1080p / 4K videos)")
    ap.add_argument("--model", default="DA3-BASE")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=100, help="DA3 frames per chunk (6 GB GPU: <= 100 @ 504)")
    ap.add_argument("--chunk-overlap", type=int, default=20)
    ap.add_argument("--da3-cache", default=None, help="reuse a DA3 prediction .npz (same video / fps / model)")
    # geometry
    ap.add_argument("--camera-height", type=float, default=1.5, help="sets the metric scale")
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--max-depth", type=float, default=6.5)
    ap.add_argument("--align-iters", type=int, default=2)
    # layout / objects / maps
    ap.add_argument("--texture-res", type=int, default=160, help="wall / floor texels per metre")
    ap.add_argument("--min-wall-length", type=float, default=0.5)
    ap.add_argument("--grid-resolution", type=float, default=0.05)
    ap.add_argument("--robot-radius", type=float, default=0.2)
    ap.add_argument("--labels", choices=["auto", "on", "off"], default="auto",
                    help="object names with Gemini (needs PI_GEMINI_API_KEY or PI_GEMINI_VERTEX)")
    ap.add_argument("--furniture", choices=["auto", "off"], default="off",
                    help="furniture completion (Hunyuan3D) is not part of this build; measured objects only")
    ap.add_argument("--no-video", action="store_true", help="skip the preview / diagnostic videos")
    ap.add_argument("--from-stage", choices=STAGES, default=None,
                    help="resume from a stage using the saved state in <output>/09_debug/work")
    args = ap.parse_args(argv)
    args.name = args.name or os.path.splitext(os.path.basename(args.video))[0]
    args.output_dir = args.output_dir or os.path.join(HERE, "output", args.name)
    args.version = VERSION
    return args


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = parse_args(argv)
    out = cfg.output_dir
    work = os.path.join(out, "09_debug", "work")
    os.makedirs(work, exist_ok=True)
    sys.stdout = Tee(os.path.join(out, "run_log.txt"))
    t_all = time.time()
    print(f"=== Version 1 Production {VERSION} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"video {os.path.abspath(cfg.video)} -> {os.path.abspath(out)}")
    print(f"OpenCV {cv2.__version__} | numpy {np.__version__} | open3d {o3d.__version__}")
    with open(os.path.join(out, "run_config.json"), "w") as f:
        json.dump({"version": VERSION, **vars(cfg)}, f, indent=2)

    start = STAGES.index(cfg.from_stage) if cfg.from_stage else 0
    if start > 0 and not state_io.exists(work):
        sys.exit("no saved state - run without --from-stage first")

    from v1prod import pipeline
    pipeline.run(cfg, out, work, start)
    print(f"\n=== finished in {(time.time() - t_all) / 60:.1f} min -> {os.path.abspath(out)} ===")


if __name__ == "__main__":
    main()
