from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from backend.models.schemas import Inspection

router = APIRouter(prefix="/api/v1", tags=["inspections"])
MAX_UPLOAD = 40 * 2**20


def _ctx(request: Request):
    return request.app.state.pipeline, request.app.state.runner


@router.post("/inspect", response_model=Inspection)
async def inspect(request: Request, image: UploadFile = File(...), room_type: Optional[str] = Form(None),
                  component: Optional[str] = Form(None), property_id: Optional[str] = Form(None),
                  tiled: Optional[str] = Form(None), property_value: Optional[float] = Form(None),
                  wait: bool = Query(False, description="block until done")):
    """Upload one image. Returns immediately with status=queued unless wait=true;
    poll GET /inspection/{id} for progress."""
    pipeline, runner = _ctx(request)
    if not pipeline.s.photo_upload:
        raise HTTPException(404, "Photo uploads are turned off: upload a walkthrough video with POST /api/v1/walkthrough "
                                 "(set PI_PHOTO_UPLOAD=1 to turn them back on).")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image larger than 40 MB")
    if tiled and tiled not in ("on", "off", "auto"):
        raise HTTPException(422, "tiled must be on|off|auto")
    room_type = None if not room_type or room_type in ("auto", "unknown") else room_type
    try:
        ins = pipeline.create(data, image.filename or "upload.jpg", property_id=property_id or "",
                              room_type=room_type, component=component or None, tiled=tiled,
                              property_value=property_value)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    runner.submit(ins.inspection_id)
    if wait:
        while True:
            ins = pipeline.store.load(ins.inspection_id)
            if ins.status in ("done", "failed"):
                break
            await asyncio.sleep(0.5)
    return ins


@router.get("/inspection/{inspection_id}", response_model=Inspection)
def get_inspection(request: Request, inspection_id: str):
    pipeline, _ = _ctx(request)
    try:
        return pipeline.store.load(inspection_id)
    except KeyError:
        raise HTTPException(404, "inspection not found") from None


@router.get("/inspection/{inspection_id}/result")
def get_result(request: Request, inspection_id: str):
    """Compact unified inspection JSON (spec phase 7 schema)."""
    ins = get_inspection(request, inspection_id)
    return request.app.state.pipeline.public_json(ins)


@router.get("/inspections")
def list_inspections(request: Request, property_id: Optional[str] = None):
    pipeline, _ = _ctx(request)
    return [{"inspection_id": i.inspection_id, "property_id": i.property_id, "room": i.room_label or i.room,
             "status": i.status, "source_image": i.source_image, "created_at": i.created_at,
             "n_detected": sum(d.state.value == "DETECTED" for d in i.detections),
             "n_review": sum(d.state.value in ("NEEDS_REVIEW", "UNKNOWN") for d in i.detections),
             "grade": (i.risk or {}).get("grade"), "condition_score": (i.risk or {}).get("condition_score"),
             "risk_level": (i.risk or {}).get("risk_level"), "decision": (i.risk or {}).get("decision"),
             "integrity_level": (i.integrity or {}).get("level")}
            for i in reversed(pipeline.store.list(property_id))]


_FILES = {"annotated": "annotated.jpg", "annotated_debug": "annotated_debug.jpg", "original": "original.jpg"}


@router.get("/inspection/{inspection_id}/image/{kind}")
def get_image(request: Request, inspection_id: str, kind: str):
    pipeline, _ = _ctx(request)
    try:
        d = pipeline.store.dir(inspection_id)
    except KeyError:
        raise HTTPException(404, "inspection not found") from None
    name = _FILES.get(kind)
    if name is None:
        if kind.startswith("crop-") and kind[5:].replace("-", "").replace("INS", "").isdigit():
            name = f"crops/{kind[5:]}.jpg"
        else:
            raise HTTPException(404, "unknown image kind")
    p = d / name
    if not p.exists():
        raise HTTPException(404, "image not available yet")
    return FileResponse(p, media_type="image/jpeg")
