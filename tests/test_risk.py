"""Risk engine, evidence-integrity checks and the assistant/portfolio API (mock Gemini)."""
from backend.models.schemas import BBox, DefectInfo, Detection, ResultState
from backend.services import integrity_service, risk_service
from tests.conftest import make_image


def _det(defect, sev, state=ResultState.DETECTED, comp="wall", n=1):
    return Detection(issue_id=f"INS-0001-{n:02d}", room="kitchen", component=comp, component_label=comp,
                     defect=DefectInfo(type=defect, label=defect, visible=True, severity=sev),
                     localization=BBox(x1=1, y1=1, x2=50, y2=50), state=state)


def test_clean_property_proceeds(settings):
    r = risk_service.assess_findings([], settings)
    assert r["grade"] == "A" and r["decision"] == "PROCEED" and r["repair_cost"]["high"] == 0


def test_severe_water_damage_is_referred(settings):
    r = risk_service.assess_findings([_det("water_damage", "severe"), _det("mold", "moderate", n=2)], settings,
                                     property_value=300000)
    assert r["decision"].startswith("REFER") and r["risk_level"] in ("High", "Critical")
    assert r["insurance"]["perils"][0]["category"] == "water"
    assert r["repair_cost"]["low"] > 0 and r["valuation"]["adjusted_value"][1] < 300000


def test_review_findings_count_less(settings):
    hi = risk_service.assess_findings([_det("crack", "moderate")], settings)
    lo = risk_service.assess_findings([_det("crack", "moderate", ResultState.NEEDS_REVIEW)], settings)
    assert lo["condition_score"] > hi["condition_score"]


def test_low_coverage_needs_more_evidence(settings):
    r = risk_service.assess_findings([], settings, coverage_pct=30)
    assert r["decision"] == "REFER - additional evidence required"


def test_duplicate_photo_other_property_is_flagged(settings, tmp_path):
    from PIL import Image
    import io

    class I:  # minimal inspection stand-in
        def __init__(self, iid, pid):
            self.inspection_id, self.property_id = iid, pid
    data = make_image()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    a = integrity_service.check(data, img, I("INS-0001", "P1"), settings)
    b = integrity_service.check(data, img, I("INS-0002", "P2"), settings)
    assert not any(f["code"].startswith("duplicate") for f in a["flags"])
    assert any(f["code"] == "duplicate_other_property" for f in b["flags"]) and b["level"] == "high"
    r = risk_service.assess_findings([], settings, integrity_flags=b["flags"])
    assert r["decision"] == "REFER - evidence integrity"


def test_api_risk_assistant_portfolio(settings):
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        r = c.post("/api/v1/inspect?wait=true", files={"image": ("a.jpg", make_image(), "image/jpeg")},
                   data={"property_id": "P9", "property_value": "250000"})
        ins = r.json()
        assert ins["status"] == "done" and ins["risk"]["grade"] in "ABCDE" and ins["integrity"]["dhash"]
        iid = ins["inspection_id"]
        s = c.post(f"/api/v1/inspection/{iid}/summary").json()
        assert s["summary"]
        a = c.post(f"/api/v1/inspection/{iid}/ask", json={"question": "What should the underwriter do?"}).json()
        assert a["answer"]
        p = c.get("/api/v1/portfolio").json()
        assert p["n"] >= 1 and any(row["id"] == iid for row in p["rows"])
        h = c.get("/api/v1/health").json()
        assert h["vlm"]["backend"] == "mock" and "cloud" in h
