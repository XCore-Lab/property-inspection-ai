"""FastAPI app.  Run:  python -m hypercorn backend.main:app --bind 127.0.0.1:8080   (or run_local.bat)
Then open http://127.0.0.1:8080/ (UI) or /docs (OpenAPI). On Cloud Run the port comes from $PORT and
hypercorn serves HTTP/2 cleartext, so uploads are not limited to Cloud Run's 32 MiB HTTP/1 request size."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import assistant, inspections, reports, walkthroughs
from backend.config.settings import get_settings
from backend.services import cloud_service, gpu_service
from backend.services.jobs import JobRunner
from backend.services.pipeline import InspectionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.pipeline = InspectionPipeline(s)
    app.state.runner = JobRunner(app.state.pipeline)
    # Re-queue anything interrupted by a restart.
    for ins in app.state.pipeline.store.list():
        if ins.status in ("queued", "running"):
            app.state.runner.submit(ins.inspection_id)
    # Walkthrough jobs can't resume (their 3D subprocess died with the old process): mark them interrupted.
    walkthroughs.recover_jobs()
    yield


app = FastAPI(title="Property Inspection AI - risk & valuation", version="2.0.0", lifespan=lifespan,
              description="Gemini-powered property inspection for insurers, mortgage lenders and valuers: "
                          "photo + video walkthrough -> defects, 3D model, coverage map, risk score, "
                          "repair cost, valuation impact, fraud signals and an underwriting assistant.")


@app.middleware("http")
async def guard_and_cache(request: Request, call_next):
    s = get_settings()
    p = request.url.path
    if s.api_key and p.startswith("/api/") and p != "/api/v1/health" and request.method != "OPTIONS":
        key = request.headers.get("x-api-key") or request.query_params.get("key")
        if key != s.api_key:
            return JSONResponse({"detail": "missing or invalid X-API-Key"}, status_code=401)
    resp = await call_next(request)
    # Browsers otherwise reuse an old index.html / app.js after the UI is updated.
    if p == "/" or p.startswith("/static/") or p.endswith((".html", "/")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp

app.include_router(inspections.router)
app.include_router(reports.router)
app.include_router(walkthroughs.router)
app.include_router(assistant.router)
WALK = Path(get_settings().data_dir) / "walkthrough"
WALK.mkdir(parents=True, exist_ok=True)
app.mount("/walkthroughs", StaticFiles(directory=WALK, html=True), name="walkthroughs")


@app.get("/api/v1/health")
def health(request: Request):
    p = request.app.state.pipeline
    return {"status": "ok", **gpu_service.gpu_info(),
            "localizer": p.localizer.info(), "vlm": p.vision.vlm.info(), "scene": p.vision.scene_vlm.info(),
            "validator": p.vision.validator.info() if p.vision.validator else None,
            "residency": p.s.gpu_residency,
            "cloud": cloud_service.status(p.s), "auth": bool(p.s.api_key),
            "queue": request.app.state.runner.q.qsize(), "running": request.app.state.runner.current}


@app.get("/api/v1/taxonomy")
def taxonomy(request: Request):
    t = request.app.state.pipeline.tax
    return {"rooms": [{"key": k, "label": v["label"]} for k, v in t.rooms.items() if k != "unknown"],
            "components": [{"key": k, "label": c.label, "tier": c.tier, "defects": list(c.prompts),
                            "manual_checks": c.manual_checks} for k, c in t.components.items()],
            "defects": {k: v["label"] for k, v in t.defects.items()}}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND / "index.html")
