"""Walkthrough job queue: persisted status files, queued jobs, restart recovery, retry.
The 3D subprocess is replaced by a tiny script that waits for a per-job gate file."""
import json
import sys
import time

import pytest
from fastapi.testclient import TestClient

FAKE_JOB = """
import pathlib, sys, time
gate, code = pathlib.Path(sys.argv[1]), int(sys.argv[2])
print("[1] Depth Anything 3", flush=True)
while not (gate.exists() or (gate.parent / "ALL").exists()):
    time.sleep(0.05)
print("[12:00:00] done in 1s", flush=True)
sys.exit(code)
"""


@pytest.fixture()
def walk(settings, tmp_path, monkeypatch):
    from backend.api import walkthroughs as wt
    from backend.main import app
    uploads, gates = tmp_path / "uploads", tmp_path / "gates"
    uploads.mkdir()
    gates.mkdir()
    script = tmp_path / "fake_job.py"
    script.write_text(FAKE_JOB)
    exit_codes: dict[str, int] = {}
    monkeypatch.setattr(wt, "UPLOADS", uploads)
    monkeypatch.setattr(wt, "WALK", tmp_path / "walkthrough")
    monkeypatch.setattr(wt, "_jobs", {})
    monkeypatch.setattr(wt, "_has_gpu", lambda: True)      # these tests run a fake job, no CUDA needed
    monkeypatch.setattr(wt, "_command", lambda job: [sys.executable, str(script), str(gates / job["name"]),
                                                     str(exit_codes.get(job["name"], 0))])

    class W:
        mod, dir = wt, uploads
        client = TestClient(app)
        codes = exit_codes

        @staticmethod
        def release(name):
            (gates / name).touch()

        @staticmethod
        def upload(filename, **data):
            return W.client.post("/api/v1/walkthrough", files={"video": (filename, b"\0" * 64, "video/mp4")}, data=data)

        @staticmethod
        def log(name):
            return W.client.get(f"/api/v1/walkthrough/{name}/log").json()

        @staticmethod
        def record(name):
            return json.loads((uploads / f"{name}.job.json").read_text())

        @staticmethod
        def wait(name, status, timeout=20):
            end = time.time() + timeout
            while time.time() < end:
                if W.log(name)["status"] == status:
                    return
                time.sleep(0.05)
            raise AssertionError(f"{name} never reached {status}: {W.log(name)}")

    yield W
    (gates / "ALL").touch()                       # never leave the shared worker blocked
    end = time.time() + 20
    while any(j["status"] in wt.ACTIVE for j in wt._jobs.values()) and time.time() < end:
        time.sleep(0.05)


def test_job_status_persisted_through_lifecycle(walk):
    r = walk.upload("first.mp4", mode="lite", property_id="P7")
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "first_lite"
    assert (walk.dir / "first_lite.mp4").read_bytes() == b"\0" * 64
    walk.wait("first_lite", "running")
    rec = walk.record("first_lite")
    assert rec["status"] == "running" and rec["mode"] == "lite" and rec["property_id"] == "P7"
    assert rec["created"] <= rec["started"] and rec["finished"] is None
    walk.release("first_lite")
    walk.wait("first_lite", "done")
    rec = walk.record("first_lite")
    assert rec["status"] == "done" and rec["returncode"] == 0 and rec["finished"] >= rec["started"]
    lg = walk.log("first_lite")
    assert lg["progress"]["percent"] == 100.0
    assert any("[queued]" in line for line in lg["tail"]) and any("done in" in line for line in lg["tail"])
    # a finished job without a walkthrough.json is not listed as pending
    assert not [w for w in walk.client.get("/api/v1/walkthroughs").json() if w["name"] == "first_lite"]


def test_second_upload_waits_in_queue(walk):
    walk.upload("first.mp4", mode="lite")
    walk.wait("first_lite", "running")
    r = walk.upload("second.mp4", mode="lite")
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert r.json()["queue"]["running"] == "first_lite"
    q = walk.log("second_lite")
    assert q["status"] == "queued" and q["started"] is None and q["progress"] is None
    assert q["queue"]["ahead"] == 1 and q["queue"]["running"] == "first_lite"
    assert q["queue"]["running_elapsed_s"] >= 0 and q["queue"]["running_started"] <= time.time()
    assert "running_eta_s" in q["queue"]
    assert walk.record("second_lite")["status"] == "queued"
    # a third one waits behind both
    walk.upload("third.mp4")
    assert walk.log("third")["queue"]["ahead"] == 2
    # the same video again while it is waiting: refused, nothing re-queued
    assert walk.upload("second.mp4", mode="lite").status_code == 409
    pending = {w["name"]: w["status"] for w in walk.client.get("/api/v1/walkthroughs").json() if "status" in w}
    assert pending == {"first_lite": "running", "second_lite": "queued", "third": "queued"}
    # the queue is FIFO: second starts when first finishes, third still waits
    walk.release("first_lite")
    walk.wait("second_lite", "running")
    q3 = walk.log("third")["queue"]
    assert q3["ahead"] == 1 and q3["running"] == "second_lite"
    assert walk.record("first_lite")["status"] == "done"
    walk.release("second_lite")
    walk.release("third")
    walk.wait("third", "done")


def test_failed_job_is_recorded(walk):
    walk.codes["broken"] = 3
    walk.upload("broken.mp4")
    walk.release("broken")
    walk.wait("broken", "failed")
    rec = walk.record("broken")
    assert rec["status"] == "failed" and rec["returncode"] == 3
    assert [w["status"] for w in walk.client.get("/api/v1/walkthroughs").json() if w["name"] == "broken"] == ["failed"]


def test_restart_marks_interrupted(walk):
    t = time.time() - 600
    for name, status in [("was_running", "running"), ("was_queued", "queued"), ("old_done", "done"),
                         ("old_failed", "failed")]:
        (walk.dir / f"{name}.job.json").write_text(json.dumps(
            {"name": name, "mode": "lite", "status": status, "created": t, "video": f"{name}.mp4",
             "started": t + 5 if status != "queued" else None, "finished": None}))
        (walk.dir / f"{name}.log").write_text("[queued] waiting\n[1] Depth Anything 3\n")
    assert walk.mod.recover_jobs() == ["was_queued", "was_running"]
    for name in ("was_running", "was_queued"):
        rec = walk.record(name)
        assert rec["status"] == "interrupted" and rec["finished"] >= t and rec["created"] == t
        assert walk.log(name)["status"] == "interrupted"
    assert walk.record("old_done")["status"] == "done"
    assert walk.record("old_failed")["status"] == "failed"
    pending = {w["name"]: w["status"] for w in walk.client.get("/api/v1/walkthroughs").json() if "status" in w}
    assert pending == {"was_running": "interrupted", "was_queued": "interrupted", "old_failed": "failed"}
    assert walk.mod.recover_jobs() == []          # idempotent


def test_app_startup_runs_recovery(walk):
    (walk.dir / "x.job.json").write_text(json.dumps({"name": "x", "status": "running", "created": 1.0}))
    with TestClient(walk.client.app):
        pass
    assert walk.record("x")["status"] == "interrupted"


def test_retry_interrupted_job_without_reupload(walk):
    (walk.dir / "again.mp4").write_bytes(b"\0" * 64)
    (walk.dir / "again.job.json").write_text(json.dumps(
        {"name": "again", "mode": "lite", "status": "interrupted", "created": 1.0, "video": "again.mp4",
         "property_id": "P3"}))
    (walk.dir / "again.log").write_text("[1] Depth Anything 3\n[3] old run\n")
    r = walk.client.post("/api/v1/walkthrough/again/retry")
    assert r.status_code == 200, r.text
    assert walk.client.post("/api/v1/walkthrough/again/retry").status_code == 409     # already queued/running
    walk.wait("again", "running")
    assert not any("old run" in line for line in walk.log("again")["tail"])         # fresh log
    rec = walk.record("again")
    assert rec["mode"] == "lite" and rec["property_id"] == "P3" and rec["created"] > 1.0
    walk.release("again")
    walk.wait("again", "done")
    assert walk.client.post("/api/v1/walkthrough/again/retry").status_code == 409     # finished
    assert walk.client.post("/api/v1/walkthrough/nope/retry").status_code == 404
    (walk.dir / "gone.job.json").write_text(json.dumps({"name": "gone", "status": "interrupted", "video": "gone.mp4"}))
    assert walk.client.post("/api/v1/walkthrough/gone/retry").status_code == 410


def test_log_rejects_bad_names(walk):
    assert walk.client.get("/api/v1/walkthrough/..%2Fx/log").status_code == 404
    assert walk.client.post("/api/v1/walkthrough/a.b/retry").status_code == 404


def test_no_gpu_refuses_video(walk, monkeypatch):
    """A CPU-only deploy shows finished walkthroughs but must not try to run Depth Anything 3."""
    monkeypatch.setattr(walk.mod, "_has_gpu", lambda: False)
    r = walk.upload("house.mp4")
    assert r.status_code == 503
    assert "needs the GPU" in r.json()["detail"]
    assert not list(walk.dir.iterdir())                   # nothing stored, nothing queued
    assert walk.client.post("/api/v1/walkthrough/house/retry").status_code == 503


def test_stored_video_served(walk):
    """The progress view can replay the stored upload after a reload (no local copy in the browser)."""
    assert walk.upload("clip.mp4").status_code == 200
    r = walk.client.get("/api/v1/walkthrough/clip/video")
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.content == b"\0" * 64
    assert walk.client.get("/api/v1/walkthrough/nope/video").status_code == 404
    assert walk.client.get("/api/v1/walkthrough/a.b/video").status_code == 404
    walk.release("clip")
