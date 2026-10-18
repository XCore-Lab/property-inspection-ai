"""Run the FULL pipeline on a folder (or files) - stage-major batching, so a 6 GB GPU
swaps models only twice per batch. Writes a summary + a combined property report.

    python scripts/inspect_folder.py data/rooms --property P001 [--room living_room] [--tiled on]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import report_service as reports  # noqa: E402
from backend.services.pipeline import InspectionPipeline  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--property", default="")
    ap.add_argument("--room", default=None)
    ap.add_argument("--component", default=None)
    ap.add_argument("--tiled", default=None, choices=["on", "off", "auto"])
    ap.add_argument("--batch", type=int, default=8, help="images per stage-major batch")
    ap.add_argument("--out", default=None, help="write summary + property report here")
    a = ap.parse_args()

    files: list[Path] = []
    for p in map(Path, a.paths):
        files += sorted(x for x in p.iterdir() if x.suffix.lower() in EXTS) if p.is_dir() else [p]
    pipe = InspectionPipeline()
    ids = []
    for f in files:
        ins = pipe.create(f.read_bytes(), f.name, property_id=a.property, room_type=a.room, component=a.component,
                          tiled=a.tiled)
        ids.append(ins.inspection_id)
    t0 = time.perf_counter()
    results = []
    for i in range(0, len(ids), a.batch):
        chunk = ids[i:i + a.batch]
        results += pipe.run_many(chunk, progress=lambda iid, st: print(f"  {iid}: {st}", flush=True) if st in ("scene", "done") else None)
    dt = time.perf_counter() - t0

    rows = []
    for ins in results:
        rows.append({"inspection_id": ins.inspection_id, "image": ins.source_image, "status": ins.status,
                     "room": ins.room, "error": ins.error,
                     "detected": [f"{d.component}:{d.defect.type}" for d in ins.detections if d.state.value == "DETECTED"],
                     "needs_review": [f"{d.component}:{d.defect.type}" for d in ins.detections if d.state.value != "DETECTED"],
                     "rejected": len(ins.rejected), "seconds": ins.timings.total_s,
                     "stage_seconds": {"scene": ins.timings.scene_s, "locate": ins.timings.localization_s,
                                       "verify": ins.timings.verification_s},
                     "peak_vram_mb": ins.timings.peak_vram_mb})
        print(f"{ins.inspection_id} {ins.source_image:36s} {ins.status:6s} room={ins.room:12s} "
              f"det={len(rows[-1]['detected'])} rev={len(rows[-1]['needs_review'])} rej={rows[-1]['rejected']} "
              f"{ins.error}")
    out = Path(a.out) if a.out else pipe.store.root / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"wall_seconds": round(dt, 1), "per_image_seconds": round(dt / max(1, len(ids)), 1),
                                                  "items": rows}, indent=2))
    done = [r for r in results if r.status == "done"]
    if done:
        html = reports.render_html(done, [pipe.store.dir(r.inspection_id) for r in done], property_id=a.property)
        (out / "property_report.html").write_text(html, encoding="utf-8")
    print(f"\n{len(ids)} images in {dt:.1f}s ({dt / max(1, len(ids)):.1f}s/image) -> {out}")


if __name__ == "__main__":
    main()
