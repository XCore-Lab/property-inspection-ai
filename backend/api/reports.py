from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from backend.services import report_service as reports

router = APIRouter(prefix="/api/v1", tags=["reports"])


@router.get("/inspection/{inspection_id}/report")
def report(request: Request, inspection_id: str, format: str = Query("html", pattern="^(html|md|txt|json)$")):
    pipeline = request.app.state.pipeline
    try:
        ins = pipeline.store.load(inspection_id)
    except KeyError:
        raise HTTPException(404, "inspection not found") from None
    if ins.status != "done":
        raise HTTPException(409, f"inspection is {ins.status}")
    d = pipeline.store.dir(inspection_id)
    if format == "json":
        return pipeline.public_json(ins)
    if format == "md":
        return PlainTextResponse(reports.render_markdown(ins), media_type="text/markdown")
    if format == "txt":
        return PlainTextResponse(reports.render_text(ins))
    return HTMLResponse(reports.render_html([ins], [d]))


@router.get("/property/{property_id}/report", response_class=HTMLResponse)
def property_report(request: Request, property_id: str):
    """Multi-image property report: every finished inspection with this property_id."""
    pipeline = request.app.state.pipeline
    items = [i for i in pipeline.store.list(property_id) if i.status == "done"]
    if not items:
        raise HTTPException(404, "no finished inspections for this property")
    from backend.services import risk_service
    flags = [f for i in items for f in (i.integrity or {}).get("flags", [])]
    prisk = risk_service.assess_findings([d for i in items for d in i.detections], pipeline.s,
                                         integrity_flags=flags, n_images=len(items))
    return HTMLResponse(reports.render_html(items, [pipeline.store.dir(i.inspection_id) for i in items],
                                            property_id=property_id, property_risk=prisk))
