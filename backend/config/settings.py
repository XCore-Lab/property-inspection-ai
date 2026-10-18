"""Runtime settings. Every field can be overridden with an env var PI_<FIELD>
(e.g. PI_GEMINI_MODEL=gemini-3.8-flash, PI_GCS_BUCKET=my-bucket, PI_MOCK_MODELS=1)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_WIN = sys.platform == "win32"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) for local runs; real env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class Settings:
    # --- paths
    data_dir: str = str(ROOT / "data")
    taxonomy_path: str = str(ROOT / "backend" / "config" / "defect_classes.yaml")
    risk_rules_path: str = str(ROOT / "backend" / "config" / "risk_rules.yaml")

    # --- Model roles (hybrid default):
    #   scene      Gemini           room + components -> which defects to look for (24/24 components in the pilot)
    #   localizer  LocateAnything-3B (local GPU, locate-anything.cpp) -> candidate boxes (high recall)
    #   verifier   Gemma 4 E4B      (local GPU, llama.cpp llama-server) -> crop verification + rule engine
    #   validator  off by default   optional Gemini second opinion (PI_VALIDATE_BACKEND=gemini): agree -> DETECTED,
    #                               disagree -> NEEDS_REVIEW
    #   text       Gemini           documentation, summaries, assistant
    scene_backend: str = "gemini"         # gemini | gemma | mock
    locate_backend: str = "capi"          # capi (LocateAnything in-process) | cli | gemini | mock
    verify_backend: str = "gemma"         # gemma | gemini | mock
    validate_backend: str = "off"         # off | gemini | mock
    vlm_backend: str = "gemini"           # backend for documentation / assistant text (gemini | mock)

    # --- Gemini (Vertex AI on Cloud Run; AI Studio key locally)
    # Auth: PI_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY, or PI_GEMINI_VERTEX=1 + PI_GCP_PROJECT.
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""
    gemini_vertex: bool = False
    gemini_max_boxes: int = 20            # only when locate_backend=gemini
    gemini_parallel: int = 6              # concurrent Gemini calls per image
    verify_strict: bool = False           # strict prompt for the verifier (on automatically when verifier=gemini)
    gcp_project: str = ""
    gcp_location: str = "global"

    # --- local models (downloaded by scripts/setup_models.py; baked into the Docker image)
    models_dir: str = str(ROOT / "models")
    bin_dir: str = str(ROOT / "bin")
    locate_gguf: str = ""                 # "" = <models_dir>/locate_anything/locate-anything-q4_k-vitf16.gguf
    locate_lib: str = ""                  # "" = <bin_dir>/locate-anything/{locate_anything.dll | liblocate_anything.so}
    locate_cli: str = ""                  # "" = <bin_dir>/locate-anything/locate-anything-cli[.exe]
    locate_mode: str = "hybrid"           # hybrid | slow | fast
    locate_device: str = ""               # "" = auto GPU, "cpu" = force CPU (sets LA_DEVICE)
    locate_threads: int = 0
    # The LocateAnything ViT uses dense attention (memory ~ tokens^2): 1024 px ~ 4.1 GB peak.
    locate_max_side: int = 1024
    gemma_gguf: str = ""                  # "" = <models_dir>/gemma4/gemma-4-E4B-it-Q4_0.gguf
    gemma_mmproj: str = ""                # "" = <models_dir>/gemma4/mmproj-gemma-4-E4B-it-Q8_0.gguf
    gemma_gpu_layers: int = 99
    llama_server: str = ""                # "" = <bin_dir>/llama.cpp/llama-server[.exe]
    llama_port: int = 8089
    llama_ctx: int = 8192
    llama_image_max_tokens: int = 1024
    # "swap" = one local model on the GPU at a time (6-8 GB cards); "coexist" = both stay loaded (L4 24 GB)
    gpu_residency: str = "swap"
    vlm_scene_max_pixels: int = 1024 * 32 * 32
    vlm_crop_max_pixels: int = 512 * 32 * 32
    vlm_context_max_pixels: int = 256 * 32 * 32
    vlm_max_new_tokens: int = 384

    # --- pipeline
    crop_padding: float = 0.3             # fraction of box size added on each side (box is outlined in the crop)
    crop_min_side: int = 96               # px, crops are grown to at least this
    max_candidates: int = 16              # verified per image (cost bound)
    max_box_area_frac: float = 0.85       # larger boxes = "whole component", flagged
    min_box_side: int = 6
    nms_iou: float = 0.5
    tiled: str = "off"                    # off | on | auto (auto: long side > tile_auto_min_side)
    tile_size: int = 1024
    tile_overlap: float = 0.2
    tile_auto_min_side: int = 2600
    specialist: bool = True               # classical-CV line evidence (advisory only)
    verify_context: bool = False          # also send the whole scene with the box drawn

    # --- 3D engine (video walkthroughs): Depth Anything 3 + SLAM, runs in this same environment
    da3_model: str = "DA3-BASE"           # DA3-SMALL (fastest) | DA3-BASE (default, Apache-2.0)
    slam_fps: float = 10.0
    slam_max_frames: int = 600
    slam_extra_args: str = ""             # extra engine3d/run_production.py flags
    slam_work_dir: str = ""               # heavy 3D engine work files ("" = <data_dir>/slam; Cloud Run: /tmp/slam)

    # --- risk / valuation (insurance, mortgage, real estate)
    currency: str = "USD"
    cost_multiplier: float = 1.0          # regional labour/material index applied to the repair-cost table

    # --- Google Cloud integrations (all optional; empty = off, local files only)
    gcs_bucket: str = ""                  # archive every inspection / walkthrough to gs://<bucket>/
    bq_dataset: str = ""                  # stream risk rows to BigQuery <project>.<dataset>.inspections
    pubsub_topic: str = ""                # publish "inspection.completed" events
    api_key: str = ""                     # if set, /api/* requires header X-API-Key (simple demo auth)

    # --- product scope: the app inspects properties from video walkthroughs. Single-photo uploads
    # (POST /api/v1/inspect) are off; walkthroughs still run the photo pipeline on their keyframes internally.
    photo_upload: bool = False

    mock_models: bool = False

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        s = cls()
        for f in fields(cls):
            v = os.environ.get("PI_" + f.name.upper())
            if v is None:
                continue
            cur = getattr(s, f.name)
            if isinstance(cur, bool):
                setattr(s, f.name, v.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(cur, int):
                setattr(s, f.name, int(v))
            elif isinstance(cur, float):
                setattr(s, f.name, float(v))
            else:
                setattr(s, f.name, v)
        if not s.gcp_project:
            s.gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        m, bdir = Path(s.models_dir), Path(s.bin_dir)
        s.locate_gguf = s.locate_gguf or str(m / "locate_anything" / "locate-anything-q4_k-vitf16.gguf")
        s.locate_lib = s.locate_lib or str(bdir / "locate-anything" / ("locate_anything.dll" if _WIN else "liblocate_anything.so"))
        s.locate_cli = s.locate_cli or str(bdir / "locate-anything" / ("locate-anything-cli.exe" if _WIN else "locate-anything-cli"))
        s.gemma_gguf = s.gemma_gguf or str(m / "gemma4" / "gemma-4-E4B-it-Q4_0.gguf")
        s.gemma_mmproj = s.gemma_mmproj or str(m / "gemma4" / "mmproj-gemma-4-E4B-it-Q8_0.gguf")
        s.llama_server = s.llama_server or str(bdir / "llama.cpp" / ("llama-server.exe" if _WIN else "llama-server"))
        if s.verify_backend == "gemini":
            s.verify_strict = True
        if s.mock_models:
            s.locate_backend = s.vlm_backend = s.scene_backend = s.verify_backend = "mock"
            s.validate_backend = "mock"
        return s

    @property
    def outputs_dir(self) -> Path:
        return Path(self.data_dir) / "outputs"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s
