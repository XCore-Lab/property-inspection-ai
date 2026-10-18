"""Gemini underwriting assistant + portfolio risk view.

  POST /api/v1/inspection/{id}/summary     Gemini writes an underwriter / claims-handler summary (cached)
  POST /api/v1/inspection/{id}/ask         grounded Q&A over one inspection (record + annotated photo)
  POST /api/v1/walkthrough/{name}/ask      grounded Q&A over a whole video walkthrough
  GET  /api/v1/portfolio                   risk overview across all inspections and walkthroughs

The assistant only sees the inspection record produced by the rule + risk engines and is told to
cite issue ids and never invent findings, values or decisions (compliance-friendly, explainable).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["assistant"])

SYSTEM = """You are an AI assistant for property underwriters, mortgage valuers and insurance claims handlers.
Answer ONLY from the inspection record below (JSON produced by a vision pipeline, a deterministic rule engine
and a risk engine). Rules:
- Cite issue ids (e.g. INS-0003-02) for every finding you mention.
- Never invent defects, measurements, values, costs or decisions; if the record does not contain it, say so.
- Findings in state NEEDS_REVIEW / MANUAL_CHECK_REQUIRED are unconfirmed - say they need a person to check.
- Do not claim structural safety or unsafety; recommend a qualified surveyor where the record does.
- Costs and valuation effects are indicative ranges from configurable rules, not quotes or appraisals.
- Be concise and practical (bullets welcome). Currency: {currency}."""

SUMMARY_TASK = """Write a short underwriting summary (max 170 words) with these headings:
**Overall condition** (grade, score, risk level), **Key findings** (most important first, with issue ids),
**Insurance view** (perils, premium loading), **Lending / valuation view** (collateral risk, valuation impact,
repair cost range), **Evidence integrity** (any flags), **Recommended next step** (the decision and why)."""


class Ask(BaseModel):
    question: str


def _record(ins) -> dict:
    keep = lambda d: {"issue_id": d.issue_id, "room": d.room, "component": d.component_label,  # noqa: E731
                      "defect": d.defect.label, "severity": d.defect.severity, "state": d.state.value,
                      "description": d.description, "manual_checks": d.manual_checks,
                      "professional_review": d.professional_review, "review_reasons": d.review_reasons,
                      "gemini_second_opinion": d.validation}
    return {"inspection_id": ins.inspection_id, "property_id": ins.property_id, "room": ins.room_label or ins.room,
            "image": ins.source_image, "image_quality": ins.image_quality.model_dump() if ins.image_quality else None,
            "components": [c.model_dump() for c in ins.components],
            "findings": [keep(d) for d in ins.detections], "manual_checks": ins.manual_checks,
            "unknown_items": ins.unknown_items, "risk": ins.risk, "integrity": ins.integrity}


def _load(request: Request, iid: str):
    p = request.app.state.pipeline
    try:
        ins = p.store.load(iid)
    except KeyError:
        raise HTTPException(404, "inspection not found") from None
    if ins.status != "done":
        raise HTTPException(409, f"inspection is {ins.status}")
    return p, ins


def _annotated(p, iid: str) -> list[Image.Image]:
    f = p.store.dir(iid) / "annotated.jpg"
    if not f.exists():
        return []
    im = Image.open(f).convert("RGB")
    im.thumbnail((1280, 1280))
    return [im]


@router.post("/inspection/{inspection_id}/summary")
def summary(request: Request, inspection_id: str, refresh: bool = False):
    p, ins = _load(request, inspection_id)
    if ins.ai_summary and not refresh:
        return {"summary": ins.ai_summary, "cached": True}
    prompt = SYSTEM.format(currency=p.s.currency) + "\n\nINSPECTION RECORD:\n" + \
        json.dumps(_record(ins), default=str) + "\n\nTASK:\n" + SUMMARY_TASK
    try:
        text = p.vision.text(prompt, _annotated(p, inspection_id), max_tokens=900)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini error: {e}") from e
    ins.ai_summary = text.strip()
    p.store.save(ins)
    return {"summary": ins.ai_summary, "cached": False}


@router.post("/inspection/{inspection_id}/ask")
def ask(request: Request, inspection_id: str, body: Ask):
    p, ins = _load(request, inspection_id)
    q = body.question.strip()[:1000]
    if not q:
        raise HTTPException(422, "empty question")
    prompt = SYSTEM.format(currency=p.s.currency) + "\n\nINSPECTION RECORD:\n" + \
        json.dumps(_record(ins), default=str) + f"\n\nQUESTION:\n{q}"
    try:
        return {"answer": p.vision.text(prompt, _annotated(p, inspection_id), max_tokens=700).strip()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini error: {e}") from e


@router.post("/walkthrough/{name}/ask")
def ask_walkthrough(request: Request, name: str, body: Ask):
    p = request.app.state.pipeline
    f = Path(p.s.data_dir) / "walkthrough" / name / "walkthrough.json"
    if not f.exists():
        raise HTTPException(404, "walkthrough not found")
    w = json.loads(f.read_text(encoding="utf-8"))
    rec = {k: w.get(k) for k in ("name", "property_id", "duration_s", "coverage", "rooms", "risk", "gaps")}
    rec["findings"] = [{k: x.get(k) for k in ("issue_id", "room", "surface", "component", "defect", "severity",
                                                "state", "description", "position") if k in x}
                       for x in w.get("findings", [])]
    q = body.question.strip()[:1000]
    prompt = SYSTEM.format(currency=p.s.currency) + "\n\nVIDEO WALKTHROUGH RECORD (3D map + coverage + findings):\n" + \
        json.dumps(rec, default=str)[:60000] + f"\n\nQUESTION:\n{q}"
    try:
        return {"answer": p.vision.text(prompt, max_tokens=700).strip()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Gemini error: {e}") from e


@router.get("/portfolio")
def portfolio(request: Request):
    """Portfolio risk overview (local aggregation; the same rows stream to BigQuery when configured)."""
    p = request.app.state.pipeline
    rows = []
    # Photo records only count when single-photo uploads are on; otherwise the store holds only the
    # keyframe inspections of walkthroughs, which are already rolled up into each walkthrough row.
    for i in (p.store.list() if p.s.photo_upload else []):
        if i.status != "done" or not i.risk:
            continue
        rows.append({"id": i.inspection_id, "kind": "photo", "property_id": i.property_id,
                     "label": i.source_image, "created_at": i.created_at, **_risk_cols(i.risk),
                     "integrity_level": (i.integrity or {}).get("level", "low")})
    for f in sorted((Path(p.s.data_dir) / "walkthrough").glob("*/walkthrough.json")):
        try:
            w = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if w.get("risk"):
            rows.append({"id": w["name"], "kind": "walkthrough", "property_id": w.get("property_id", ""),
                         "label": w["name"], "created_at": w.get("created_at"), **_risk_cols(w["risk"]),
                         "integrity_level": "low", "coverage_pct": (w.get("coverage") or {}).get("overall")})
    by = lambda k: {v: sum(1 for r in rows if r.get(k) == v) for v in sorted({r.get(k) for r in rows if r.get(k)})}  # noqa: E731
    return {"n": len(rows), "by_grade": by("grade"), "by_risk_level": by("risk_level"), "by_decision": by("decision"),
            "repair_exposure": {"low": sum(r["repair_low"] or 0 for r in rows),
                                "high": sum(r["repair_high"] or 0 for r in rows), "currency": p.s.currency},
            "integrity_flagged": sum(1 for r in rows if r["integrity_level"] in ("medium", "high")),
            "rows": sorted(rows, key=lambda r: str(r.get("created_at") or ""), reverse=True)}


def _risk_cols(r: dict) -> dict:
    return {"grade": r.get("grade"), "condition_score": r.get("condition_score"), "risk_level": r.get("risk_level"),
            "decision": r.get("decision"), "repair_low": (r.get("repair_cost") or {}).get("low"),
            "repair_high": (r.get("repair_cost") or {}).get("high"),
            "valuation_pct": (r.get("valuation") or {}).get("impact_pct")}
