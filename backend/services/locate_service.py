"""Open-vocabulary defect localization behind an interchangeable interface.

Backends
  * LocateAnythingCAPI - LocateAnything-3B (nvidia) via the locate-anything.cpp shared library
                         (locate_anything.dll / liblocate_anything.so, CUDA), loaded in-process (default).
  * LocateAnythingCLI  - same engine through locate-anything-cli (one process per call; isolates crashes).
  * GeminiLocalizer    - Gemini returns boxes for the routed defect phrases (box_2d, 0-1000 scale).
  * MockLocalizer      - deterministic fake for tests / UI work without models.

Any other detector (a Vertex AI endpoint, YOLO, ...) can be plugged in by implementing
`Localizer.locate(image_bytes, phrases) -> list[RawBox]`.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from backend.config.settings import Settings
from backend.services.gpu_service import get_arbiter
from backend.services.prompt_service import build_locate_query

log = logging.getLogger(__name__)
MODES = {"hybrid": 0, "slow": 1, "fast": 2}


@dataclass
class RawBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


class Localizer:
    name = "base"
    model_name = "base"

    def locate(self, image_bytes: bytes, phrases: list[str]) -> list[RawBox]:  # pragma: no cover
        raise NotImplementedError

    def release(self) -> None:
        pass

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name}


def _parse_json(s: str) -> list[RawBox]:
    d = json.loads(s)
    out = []
    for det in d.get("detections", []):
        b = det.get("box") or []
        if len(b) == 4:
            out.append(RawBox(det.get("label", ""), *map(float, b)))
    return out


def _model_name(gguf: str) -> str:
    return f"LocateAnything-3B {Path(gguf).stem.replace('locate-anything-', '').upper()}"


def _cuda_dirs(s: Settings, lib: Path) -> list[Path]:
    """Windows: folders with the CUDA runtime DLLs (redistributable next to the build, or bin/cuda)."""
    c = [lib.parent, lib.parents[1] / "cuda-12.6" / "bin", Path(s.bin_dir) / "cuda" / "bin"]
    return [d for x in c for d in (x, x / "x64") if d.exists()]


class LocateAnythingCAPI(Localizer):
    name = "locate-anything.cpp (C API)"

    def __init__(self, s: Settings):
        self.s = s
        self.lib_path = Path(s.locate_lib)
        self.gguf = Path(s.locate_gguf)
        self.model_name = _model_name(s.locate_gguf)
        self._lib = None
        self._ctx = None
        self.load_s = 0.0
        get_arbiter().register("locate", self.release)

    def _lib_handle(self):
        if self._lib is None:
            if not self.lib_path.exists():
                raise RuntimeError(f"LocateAnything library not found: {self.lib_path} "
                                   "(run scripts/setup_models.py, or set PI_LOCATE_LIB)")
            if sys.platform == "win32":
                for d in _cuda_dirs(self.s, self.lib_path):
                    os.add_dll_directory(str(d))
            lib = ctypes.CDLL(str(self.lib_path))
            lib.la_capi_load.restype = ctypes.c_void_p
            lib.la_capi_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
            lib.la_capi_free.argtypes = [ctypes.c_void_p]
            lib.la_capi_locate_buffer.restype = ctypes.c_void_p
            lib.la_capi_locate_buffer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int]
            lib.la_capi_free_string.argtypes = [ctypes.c_void_p]
            lib.la_capi_last_error.restype = ctypes.c_char_p
            lib.la_capi_last_error.argtypes = [ctypes.c_void_p]
            self._lib = lib
        return self._lib

    def _ensure_loaded(self) -> None:
        if self._ctx:
            return
        if not self.gguf.exists():
            raise RuntimeError(f"LocateAnything weights not found: {self.gguf} (run scripts/setup_models.py)")
        if self.s.locate_device:
            os.environ["LA_DEVICE"] = self.s.locate_device
        lib = self._lib_handle()
        t = time.perf_counter()
        ctx = lib.la_capi_load(str(self.gguf).encode(), int(self.s.locate_threads))
        if not ctx:
            raise RuntimeError("la_capi_load failed (see stderr; possibly out of VRAM - try PI_LOCATE_DEVICE=cpu)")
        self._ctx = ctx
        self.load_s = time.perf_counter() - t
        log.info("LocateAnything loaded in %.1fs", self.load_s)

    def locate(self, image_bytes: bytes, phrases: list[str]) -> list[RawBox]:
        with get_arbiter().use("locate"):
            self._ensure_loaded()
            q = build_locate_query(phrases).encode()
            p = self._lib.la_capi_locate_buffer(self._ctx, image_bytes, len(image_bytes), q, MODES.get(self.s.locate_mode, 0))
            if not p:
                err = self._lib.la_capi_last_error(self._ctx) or b""
                raise RuntimeError(f"locate failed: {err.decode(errors='replace') or 'unknown (bad image?)'}")
            try:
                s = ctypes.string_at(p).decode("utf-8", errors="replace")
            finally:
                self._lib.la_capi_free_string(p)
        return _parse_json(s)

    def release(self) -> None:
        if self._ctx and self._lib is not None:
            self._lib.la_capi_free(self._ctx)
            self._ctx = None
            get_arbiter().mark_released("locate")

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "gguf": str(self.gguf),
                "loaded": bool(self._ctx), "available": self.lib_path.exists() and self.gguf.exists()}


class LocateAnythingCLI(Localizer):
    name = "locate-anything.cpp (CLI)"

    def __init__(self, s: Settings):
        self.s = s
        self.model_name = _model_name(s.locate_gguf)

    def locate(self, image_bytes: bytes, phrases: list[str]) -> list[RawBox]:
        with get_arbiter().use("locate"), tempfile.TemporaryDirectory() as td:
            img = Path(td) / "in.jpg"
            out = Path(td) / "out.json"
            img.write_bytes(image_bytes)
            env = dict(os.environ)
            if sys.platform == "win32":
                env["PATH"] = os.pathsep.join(str(d) for d in _cuda_dirs(self.s, Path(self.s.locate_cli))) \
                    + os.pathsep + env.get("PATH", "")
            if self.s.locate_device:
                env["LA_DEVICE"] = self.s.locate_device
            cmd = [self.s.locate_cli, "detect", "--model", self.s.locate_gguf, "--input", str(img),
                   "--prompt", build_locate_query(phrases), "--output", str(out), "--mode", self.s.locate_mode,
                   "--threads", str(self.s.locate_threads)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
            if r.returncode != 0 or not out.exists():
                raise RuntimeError(f"locate-anything-cli failed ({r.returncode}): {r.stderr[-800:]}")
            return _parse_json(out.read_text(encoding="utf-8"))


class MockLocalizer(Localizer):
    """Returns one plausible box per phrase for ~1/3 of phrases (hash-deterministic)."""
    name = "mock"
    model_name = "MockLocalizer"

    def locate(self, image_bytes: bytes, phrases: list[str]) -> list[RawBox]:
        from PIL import Image
        import io
        w, h = Image.open(io.BytesIO(image_bytes)).size
        out = []
        seed = hashlib.md5(image_bytes[:4096]).digest()
        for i, p in enumerate(phrases):
            hv = hashlib.md5(seed + p.encode()).digest()
            if hv[0] % 3:
                continue
            cx, cy = w * (0.2 + 0.6 * hv[1] / 255), h * (0.2 + 0.6 * hv[2] / 255)
            bw, bh = w * (0.08 + 0.2 * hv[3] / 255), h * (0.08 + 0.2 * hv[4] / 255)
            out.append(RawBox(p, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        return out


GEMINI_BOX_SCHEMA = {
    "type": "object", "required": ["detections"],
    "properties": {"detections": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string"},
        "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
        "required": ["label", "box_2d"]}}},
}


class GeminiLocalizer(Localizer):
    """Gemini (cloud) open-vocabulary boxes: box_2d = [ymin, xmin, ymax, xmax] normalised to 0-1000.
    Same contract as LocateAnything: returns candidates per phrase, the VLM verifier filters them."""
    name = "gemini"

    def __init__(self, s: Settings):
        from backend.services.vlm_service import GeminiVLM
        self.s = s
        self.model_name = f"{s.gemini_model} (boxes)"
        self._vlm = GeminiVLM(s)

    def locate(self, image_bytes: bytes, phrases: list[str]) -> list[RawBox]:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        listing = "\n".join(f"- {p}" for p in phrases)
        prompt = (
            "You are the localization step of a property inspection system. Find every visible region in this "
            "photo that may show one of these defects:\n" + listing + "\n\n"
            "Return {\"detections\": [...]}; each item = {\"label\": <one phrase from the list, copied exactly>, "
            "\"box_2d\": [ymin, xmin, ymax, xmax] normalised to 0-1000}. One box per defect instance, tight around "
            "the damaged area (not the whole wall/floor). Box only damage you can actually see - most well-kept "
            "rooms have none. Do not box intact surfaces, paint texture, grout or panel joints, shadows, "
            "reflections, patterns, furniture or decor. "
            f"At most {self.s.gemini_max_boxes} boxes. Empty list if nothing matches.")
        out = self._vlm.generate([img], prompt, 2048, schema=GEMINI_BOX_SCHEMA)
        items = (out.parsed or {}).get("detections") or []
        lower = {p.lower(): p for p in phrases}
        boxes = []
        for it in items:
            b = it.get("box_2d") if isinstance(it, dict) else None
            if not b or len(b) != 4:
                continue
            y1, x1, y2, x2 = (float(v) for v in b)
            label = lower.get(str(it.get("label", "")).strip().lower(), str(it.get("label", "")))
            boxes.append(RawBox(label, min(x1, x2) * w / 1000, min(y1, y2) * h / 1000,
                                max(x1, x2) * w / 1000, max(y1, y2) * h / 1000))
        return boxes


def create_localizer(s: Settings) -> Localizer:
    if s.locate_backend == "mock":
        return MockLocalizer()
    if s.locate_backend == "gemini":
        return GeminiLocalizer(s)
    if s.locate_backend == "cli":
        return LocateAnythingCLI(s)
    return LocateAnythingCAPI(s)
