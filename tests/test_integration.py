"""Integration tests (spec phase 22) with mock models: image -> ... -> report, and the HTTP API."""
import json
import time

from fastapi.testclient import TestClient

from backend.models.schemas import Inspection, ResultState
from backend.services.pipeline import InspectionPipeline


def test_pipeline_end_to_end(settings, image_bytes):
    p = InspectionPipeline(settings)
    ins = p.create(image_bytes, "kitchen.jpg", property_id="P001")
    ins = p.run(ins.inspection_id)
    assert ins.status == "done", ins.error
    assert ins.room == "kitchen"
    assert ins.prompts                                            # router produced prompts
    assert ins.detections or ins.rejected                         # localizer -> crop -> verifier ran
    d = p.store.dir(ins.inspection_id)
    for f in ("original.jpg", "annotated.jpg", "annotated_debug.jpg", "inspection.json", "result.json",
              "report.html", "report.md", "report.txt", "raw/scene_raw.txt", "raw/localizer_raw.json",
              "raw/verification_raw.json"):
        assert (d / f).exists(), f
    for det in ins.detections + ins.rejected:
        assert (d / det.crop_path).exists()
        assert det.state in (ResultState.DETECTED, ResultState.NEEDS_REVIEW, ResultState.UNKNOWN, ResultState.NOT_DETECTED)
    # exhaust hood present -> functional checks are manual, never judged
    assert any(m["component"] == "exhaust_hood" and m["check"] == "Motor function" for m in ins.manual_checks)
    # compact JSON follows the target schema
    pub = json.loads((d / "result.json").read_text())
    assert set(pub) >= {"inspection_id", "property_id", "room", "source_image", "detections"}
    for det in pub["detections"]:
        assert set(det["localization"]) == {"x1", "y1", "x2", "y2"}
        assert det["models"]["localization"] and det["models"]["verification"]
    # reload round-trip validates against the schema
    assert Inspection.model_validate_json((d / "inspection.json").read_text()).inspection_id == ins.inspection_id
    assert "PROPERTY INSPECTION REPORT" in (d / "report.txt").read_text()


def test_user_hints_skip_scene(settings, image_bytes):
    p = InspectionPipeline(settings)
    ins = p.create(image_bytes, "x.jpg", room_type="bathroom", component="grout")
    ins = p.run(ins.inspection_id)
    assert ins.status == "done" and ins.room == "bathroom"
    assert {c.component for c in ins.components} == {"grout"}
    assert all("grout" in ph for ph in ins.prompts) or ins.prompts


def test_tiled_mode(settings):
    from tests.conftest import make_image
    p = InspectionPipeline(settings)
    ins = p.create(make_image(2400, 1600), "big.jpg", tiled="on")
    ins = p.run(ins.inspection_id)
    assert ins.status == "done"
    raw = json.loads((p.store.dir(ins.inspection_id) / "raw" / "localizer_raw.json").read_text())
    assert any(r["view"] == "tile" for r in raw)


def test_bad_image_rejected(settings):
    p = InspectionPipeline(settings)
    try:
        p.create(b"not an image", "x.jpg")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_api_flow(settings, image_bytes):
    from backend.main import app
    with TestClient(app) as c:
        assert c.get("/api/v1/health").json()["status"] == "ok"
        tax = c.get("/api/v1/taxonomy").json()
        assert any(r["key"] == "kitchen" for r in tax["rooms"])
        r = c.post("/api/v1/inspect", files={"image": ("k.jpg", image_bytes, "image/jpeg")},
                   data={"property_id": "P9"})
        assert r.status_code == 200, r.text
        iid = r.json()["inspection_id"]
        for _ in range(100):
            ins = c.get(f"/api/v1/inspection/{iid}").json()
            if ins["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert ins["status"] == "done", ins.get("error")
        assert c.get(f"/api/v1/inspection/{iid}/report").status_code == 200
        assert "PROPERTY INSPECTION REPORT" in c.get(f"/api/v1/inspection/{iid}/report?format=md").text
        assert c.get(f"/api/v1/inspection/{iid}/result").json()["property_id"] == "P9"
        assert c.get(f"/api/v1/inspection/{iid}/image/annotated").status_code == 200
        assert c.get("/api/v1/property/P9/report").status_code == 200
        assert c.get("/api/v1/inspection/INS-9999").status_code == 404
        assert c.get("/api/v1/inspection/../../etc").status_code == 404
        assert c.post("/api/v1/inspect", files={"image": ("x.jpg", b"junk", "image/jpeg")}).status_code == 400
        assert c.get("/").status_code == 200


def test_video_only_scope(settings, image_bytes):
    """Default product scope: single-photo uploads are refused and the portfolio lists walkthroughs only
    (the keyframe inspections a walkthrough stores are not counted as separate records)."""
    from backend.main import app
    settings.photo_upload = False
    InspectionPipeline(settings).create(image_bytes, "keyframe.jpg", property_id="P1")
    with TestClient(app) as c:
        r = c.post("/api/v1/inspect", files={"image": ("k.jpg", image_bytes, "image/jpeg")})
        assert r.status_code == 404
        assert "walkthrough" in r.json()["detail"]
        pf = c.get("/api/v1/portfolio").json()
        assert all(row["kind"] == "walkthrough" for row in pf["rows"])
