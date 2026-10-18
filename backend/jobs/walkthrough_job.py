"""Cloud Run Job entry point: process ONE walkthrough whose upload and <name>.job.json the web service wrote to the
shared data dir (the GCS bucket mounted at PI_DATA_DIR), with the same _run() the in-process queue uses.

    python -m backend.jobs.walkthrough_job --name <walkthrough name>

With PI_GPU_JOB set, the web service starts one execution per upload (backend/api/walkthroughs.py: _dispatch), so the
GPU is billed only while a walkthrough is being processed. Status, log and results land in the shared data dir, where
the web service reads them."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from backend.api import walkthroughs as wt

log = logging.getLogger("walkthrough_job")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    name = ap.parse_args().name
    rec = wt._read(name)
    if not rec or not rec.get("video"):
        log.error("no job record for %s in %s", name, wt.UPLOADS)
        return 2
    job = wt._new_job(rec["name"], rec.get("mode") or "advanced", rec["video"], rec.get("property_id") or "",
                      rec.get("slam_args") or "")
    job["created"] = rec.get("created") or job["created"]
    job["counts"].update(rec.get("counts") or {})     # frames_est from the upload: early ETA
    try:
        wt._run(job)
    except Exception:  # noqa: BLE001
        log.exception("walkthrough %s crashed", name)
        job.update(status="failed", finished=time.time())
        wt._save(job)
        return 1
    log.info("walkthrough %s: %s (rc=%s)", name, job["status"], job["returncode"])
    return 0 if job["status"] == "done" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
