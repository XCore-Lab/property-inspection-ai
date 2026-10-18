"""Walkthrough (video + 3D map + coverage) endpoints. The heavy work runs in
scripts/inspect_walkthrough.py as a subprocess (it runs the 3D engine, then the Gemini inspection).

One 3D reconstruction runs per GPU, so jobs wait in a FIFO queue. Each job's status
(queued / running / done / failed / interrupted + times) is kept in <data>/uploads/<name>.job.json
next to its log (on Cloud Run the data dir is the GCS bucket): the UI can say a job is waiting, and
after a restart the jobs the old process was running show up as "interrupted" instead of vanishing."""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.config.settings import get_settings
from backend.services import walk_progress
from backend.services.gpu_service import get_arbiter, gpu_info, slam_lock

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["walkthroughs"])
ROOT = Path(__file__).resolve().parents[2]
WALK = Path(get_settings().data_dir) / "walkthrough"
UPLOADS = Path(get_settings().data_dir) / "uploads"
NAME_RX = re.compile(r"[A-Za-z0-9_-]+")
ACTIVE = ("queued", "running")
QUEUED_LINE = "[queued] waiting for the GPU (one 3D reconstruction at a time)\n"
# fields written to <name>.job.json (the rest of a job dict is live progress state of this process)
PERSIST = ("name", "mode", "status", "created", "started", "finished", "returncode", "video", "property_id",
           "slam_args")
_jobs: dict[str, dict] = {}          # jobs submitted to THIS process
_lock = threading.Lock()             # guards _jobs and the queued/running checks
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None


# ---- persisted job status ---------------------------------------------------------------------------
def _job_file(name: str) -> Path:
    return UPLOADS / f"{name}.job.json"


def _save(job: dict) -> None:
    p = _job_file(job["name"])
    tmp = p.with_name(p.name + ".tmp")
    data = json.dumps({k: job.get(k) for k in PERSIST}, indent=1)
    try:
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        try:                         # a mount that can't rename over a file: write it in place
            p.write_text(data, encoding="utf-8")
        except OSError as e:         # the status file is for the UI; never fail the job over it
            log.warning("could not write %s: %s", p, e)


def _read(name: str) -> dict | None:
    try:
        return json.loads(_job_file(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _job(name: str) -> dict | None:
    """This process's job, else the persisted record (e.g. a job from before a restart)."""
    return _jobs.get(name) or _read(name)


def _records() -> dict[str, dict]:
    recs = {}
    for p in UPLOADS.glob("*.job.json"):
        name = p.name[: -len(".job.json")]
        if rec := _read(name):
            recs[name] = rec
    recs.update(_jobs)
    return recs


def recover_jobs() -> list[str]:
    """Run once at startup: a job that was queued or running in the previous process died with it
    (its subprocess and the in-memory queue are gone), so mark it interrupted."""
    out = []
    for p in sorted(UPLOADS.glob("*.job.json")):
        name = p.name[: -len(".job.json")]
        rec = _read(name)
        if rec and rec.get("status") in ACTIVE and name not in _jobs:
            rec.update(name=name, status="interrupted", finished=time.time())
            _save(rec)
            out.append(name)
    if out:
        log.warning("walkthrough jobs interrupted by a restart: %s", ", ".join(out))
    return out


# ---- the GPU queue -----------------------------------------------------------------------------------
def _command(job: dict) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "inspect_walkthrough.py"), "--video", str(UPLOADS / job["video"]),
           "--run-slam", "--name", job["name"], "--mode", job["mode"]]
    if job.get("property_id"):
        cmd += ["--property", job["property_id"]]
    if job.get("slam_args"):
        cmd += ["--slam-args", job["slam_args"]]
    return cmd


def _scan(job: dict, log_path: Path) -> None:
    if log_path.exists():
        walk_progress.scan(log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                           job["seen"], job["counts"])


def _run(job: dict) -> None:
    log_path = UPLOADS / f"{job['name']}.log"
    arb = get_arbiter()
    with slam_lock, arb.lock:            # one 3D job per GPU; API inspections wait meanwhile
        arb.release_all()                # free the server's LocateAnything / Gemma VRAM for Depth Anything 3
        job.update(status="running", started=time.time())
        _save(job)
        with open(log_path, "a", encoding="utf-8") as fh:
            proc = subprocess.Popen(_command(job), stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT,
                                    env=os.environ.copy())
            while True:                  # timestamps each step marker even when no browser is polling
                try:
                    proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    _scan(job, log_path)
    _scan(job, log_path)
    job.update(status="done" if proc.returncode == 0 else "failed", finished=time.time(),
               returncode=proc.returncode)
    _save(job)


def _work_loop() -> None:
    while True:
        name = _queue.get()
        job = _jobs.get(name)
        if not job or job["status"] != "queued":
            continue
        try:
            _run(job)
        except Exception:  # noqa: BLE001
            log.exception("walkthrough %s crashed", job["name"])
            job.update(status="failed", finished=time.time())
            _save(job)


def _enqueue(job: dict) -> None:
    global _worker
    (UPLOADS / f"{job['name']}.log").write_text(QUEUED_LINE, encoding="utf-8")
    _save(job)
    _queue.put(job["name"])
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work_loop, name="walkthrough-gpu", daemon=True)
            _worker.start()


def _new_job(name: str, mode: str, video: str, property_id: str = "", slam_args: str = "") -> dict:
    return {"name": name, "mode": mode, "status": "queued", "created": time.time(), "started": None,
            "finished": None, "returncode": None, "video": video, "property_id": property_id,
            "slam_args": slam_args, "seen": {}, "counts": {}}


def _queue_info(job: dict) -> dict:
    """Where a queued job stands: how many walkthroughs are ahead of it and how far the running one is."""
    now = time.time()
    with _lock:
        jobs = list(_jobs.values())
    before = [j for j in jobs if j["status"] == "queued" and (j["created"], j["name"]) < (job["created"], job["name"])]
    run = next((j for j in jobs if j["status"] == "running"), None)
    info = {"ahead": len(before) + (run is not None), "waiting_s": round(now - job["created"])}
    if run:
        prog = walk_progress.estimate(run["seen"], run["counts"], run["started"], now)
        info.update(running=run["name"], running_started=run["started"],
                    running_elapsed_s=round(now - run["started"]), running_eta_s=prog.get("eta_s"))
    return info


def _store_upload(src, path: Path) -> int:
    """Copy the uploaded video into the data dir (streamed; runs in a worker thread because on Cloud Run
    the data dir is a GCS mount) and return its frame count (0 = unknown)."""
    src.seek(0)
    with open(path, "wb") as fh:
        shutil.copyfileobj(src, fh, 1 << 20)
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n
    except Exception:  # noqa: BLE001
        return 0


def _accepted(job: dict) -> dict:
    name = job["name"]
    out = {"name": name, "mode": job["mode"], "status": job["status"], "log": f"/api/v1/walkthrough/{name}/log",
           "view": f"/walkthroughs/{name}/"}
    if job["status"] == "queued":
        out["queue"] = _queue_info(job)
    return out


# ---- endpoints ---------------------------------------------------------------------------------------
@router.get("/walkthroughs")
def list_walkthroughs():
    out = []
    for p in sorted(WALK.glob("*/walkthrough.json")):
        try:
            w = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        r = w.get("risk") or {}
        out.append({"name": w["name"], "url": f"/walkthroughs/{w['name']}/", "duration_s": w["duration_s"],
                    "findings": len(w["findings"]), "coverage": w["coverage"]["overall"],
                    "rooms": [r["name"] for r in w["rooms"]], "grade": r.get("grade"),
                    "risk_level": r.get("risk_level"), "decision": r.get("decision"),
                    "mode": w.get("mode", "advanced")})
    for name, j in sorted(_records().items(), key=lambda kv: kv[1].get("created") or 0):
        if j.get("status") != "done":
            out.append({"name": name, "status": j.get("status", "unknown"), "mode": j.get("mode"),
                        "created": j.get("created"), "log": f"/api/v1/walkthrough/{name}/log"})
    return out


NO_GPU = ("Video processing needs the GPU, and this demo server is running without one. "
          "The finished walkthroughs on this page can still be opened.")


def _has_gpu() -> bool:
    """Depth Anything 3 needs CUDA: on a CPU-only deploy (GPU=0) it would run out of memory or take hours,
    so that server only shows walkthroughs that were processed elsewhere."""
    return bool(gpu_info()["cuda"])


@router.post("/walkthrough")
async def create_walkthrough(request: Request, video: UploadFile = File(...), property_id: str = Form(""),
                             slam_args: str = Form(""), mode: str = Form("advanced")):
    """Upload a walkthrough video. Runs the 3D engine (GPU) + AI inspection in the background;
    poll GET /api/v1/walkthrough/{name}/log. One walkthrough is processed at a time, in upload order:
    while another is processing the new one is "queued" (the log response then has a "queue" block).
    mode=lite: faster, lighter 3D build (3 cm, 1 alignment pass, no preview videos).
    mode=advanced: full-detail 3D + preview videos. Both use the same AI models."""
    if not _has_gpu():
        raise HTTPException(503, NO_GPU)
    mode = "lite" if mode.strip().lower() == "lite" else "advanced"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video.filename or "walkthrough.mp4").stem)[:55] or "walkthrough"
    if mode == "lite":
        name += "_lite"          # keeps a Lite and an Advanced run of the same video side by side
    file_name = f"{name}{Path(video.filename or '.mp4').suffix or '.mp4'}"
    with _lock:                  # reserve the name before the (slow) copy so a second upload gets 409
        if _jobs.get(name, {}).get("status") in ACTIVE:
            raise HTTPException(409, "a walkthrough with this name is already queued or running")
        job = _jobs[name] = _new_job(name, mode, file_name, property_id, slam_args)
    try:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        n = await run_in_threadpool(_store_upload, video.file, UPLOADS / file_name)
    except Exception:
        with _lock:
            _jobs.pop(name, None)
        raise
    if n > 0:                    # SLAM samples 1 of every 3 frames: early ETA before the log reports the frame count
        job["counts"]["frames_est"] = max(1, n // 3)
    await run_in_threadpool(_enqueue, job)
    return _accepted(job)


@router.post("/walkthrough/{name}/retry")
def retry_walkthrough(name: str):
    """Process an interrupted (server restarted) or failed walkthrough again from its saved upload:
    no re-upload needed, and 3D steps that already finished are reused."""
    if not NAME_RX.fullmatch(name):
        raise HTTPException(404)
    if not _has_gpu():
        raise HTTPException(503, NO_GPU)
    with _lock:
        old = _job(name)
        if not old:
            raise HTTPException(404, "unknown walkthrough")
        if name in _jobs and old.get("status") in ACTIVE:
            raise HTTPException(409, "this walkthrough is already queued or running")
        if old.get("status") == "done":
            raise HTTPException(409, "this walkthrough already finished")
        if not old.get("video") or not (UPLOADS / old["video"]).exists():
            raise HTTPException(410, "the uploaded video is no longer stored - upload it again")
        job = _jobs[name] = _new_job(name, old.get("mode") or "advanced", old["video"], old.get("property_id") or "",
                                     old.get("slam_args") or "")
    _enqueue(job)
    return _accepted(job)


@router.get("/walkthrough/{name}/log")
def walkthrough_log(name: str):
    if not NAME_RX.fullmatch(name):
        raise HTTPException(404)
    p = UPLOADS / f"{name}.log"
    if not p.exists():
        raise HTTPException(404, "no log")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    j = _job(name) or {}
    out = {"status": j.get("status", "unknown"), "tail": lines[-40:], "progress": None, "mode": j.get("mode"),
           "created": j.get("created"), "started": j.get("started"), "finished": j.get("finished")}
    if "seen" in j and j.get("started"):
        out["progress"] = walk_progress.estimate(j["seen"], j["counts"], j["started"])
    if out["status"] == "queued" and name in _jobs:
        out["queue"] = _queue_info(j)
    return out


@router.get("/walkthrough/{name}/video")
def walkthrough_video(name: str):
    """The stored upload, so the progress view can play its scanning preview when the browser has no
    local copy of the video (page reloaded, second tab, job started from the command line)."""
    if not NAME_RX.fullmatch(name):
        raise HTTPException(404)
    j = _job(name) or {}
    p = UPLOADS / j["video"] if j.get("video") else None
    if p is None or not p.is_file():
        raise HTTPException(404, "no stored video")
    return FileResponse(p, media_type="video/mp4")
