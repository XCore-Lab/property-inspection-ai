"""Vision-language models: scene/component recognition, crop verification, second-opinion validation.

Roles (see settings): scene = Gemini, verifier = Gemma 4 E4B (llama.cpp, local GPU), validator = Gemini
(strict second opinion), text = Gemini (documentation, summaries, assistant).
JSON is enforced with Gemini's `response_json_schema` / llama.cpp's schema grammar, parsed with a tolerant
repairer and normalised with Pydantic. The raw text is always kept alongside the parsed result for audit.
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import io
import json
import logging
import math
import re
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from backend.config.settings import Settings
from backend.models.schemas import SceneComponent, SceneResult, VerificationResult
from backend.services.gpu_service import get_arbiter
from backend.services.json_utils import extract_json, to_bool
from backend.services.taxonomy import Taxonomy

log = logging.getLogger(__name__)


SCENE_PROMPT = """You are assisting with a visual property-condition inspection.
Identify the room and all visible property components that are relevant for property inspection.

Allowed room values: {rooms}
Preferred component names: {components}

Return JSON only, exactly in this form:
{{"room": "<one allowed room value>", "image_type": "overview|component|close_up", "components": [{{"name": "<component>", "note": "<short visible-condition note or empty>"}}], "authenticity": "real_photo|screen_photo|rendering|ai_generated|unsure"}}

Rules: list at most 8 components, only those clearly visible and relevant to the property's condition; do not guess hidden or partly visible components; furniture and appliances are not components; each note at most 6 words.
authenticity (evidence check for insurers / lenders): "real_photo" for an ordinary camera photo of a real room; "screen_photo" if it is a photo of a monitor, phone screen or printout (moire, bezels, pixel grid); "rendering" for a 3D render / catalogue image; "ai_generated" only with clear generation artefacts; "unsure" otherwise."""


VERIFY_PROMPT = """You are assisting with a visual property-condition inspection.
{views}
Analyze ONLY the visible region inside the red box / in the close-up.

Expected component:
{component}

Candidate defect:
{candidate_defect}

Things that are often mistaken for this defect (NOT defects): {hard_negatives}.

Determine whether visible evidence supports the proposed defect.

Return JSON only:
{{"component": "", "defect_present": true, "defect_type": "", "severity": "minor|moderate|severe|unknown", "description": "<one short sentence, max 25 words>", "alternative_explanation": "<only if defect_present is false: 1-4 words naming what the region shows, else empty>", "manual_check_required": false}}

Rules:
1. Do not invent defects. Set defect_present to true ONLY if the candidate defect itself ({candidate_defect}) is visible. If the region shows a normal surface, joint, pattern, shadow or reflection, or a different kind of defect, set defect_present to false and name what it shows in alternative_explanation (e.g. "crack, not stain").
2. Only describe visually observable evidence, in one short sentence.
3. Do not claim structural damage from appearance alone.
4. If uncertain, use "unknown" for severity.
5. If functionality cannot be determined visually (locks, fans, motors, hinges, opening/closing), set manual_check_required to true.
6. defect_type is the defect you actually see (use the candidate name when it matches).
"""

# Appended when verify_strict is on (used with cloud boxes, which over-propose on clean rooms).
VERIFY_STRICT = """7. STRICT MODE: the candidate came from an automatic detector that often flags intact surfaces. Assume the
   region is NORMAL unless the damage is unmistakable at this zoom: a real break/gap/hole, missing or detached
   material, flaking paint, a distinct stain with clear edges, rust, or broken glass.
   Set defect_present to false for: paint texture or brush marks, uneven lighting, shadows, reflections, grout or
   panel joints, wood grain, fabric/carpet pattern or folds, dirt or dust, furniture/décor edges, cables, and
   anything you would describe as "possible", "slight" or "may be".
"""


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "room": {"type": "string"},
        "image_type": {"type": "string", "enum": ["overview", "component", "close_up"]},
        "components": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["name", "note"]}},
        "authenticity": {"type": "string",
                         "enum": ["real_photo", "screen_photo", "rendering", "ai_generated", "unsure"]},
    },
    "required": ["room", "image_type", "components", "authenticity"],
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "component": {"type": "string"},
        "defect_present": {"type": "boolean"},
        "defect_type": {"type": "string"},
        "severity": {"type": "string", "enum": ["minor", "moderate", "severe", "unknown"]},
        "description": {"type": "string"},
        "alternative_explanation": {"type": "string"},
        "manual_check_required": {"type": "boolean"},
    },
    "required": ["component", "defect_present", "defect_type", "severity", "description",
                 "alternative_explanation", "manual_check_required"],
}


def _limit_pixels(img: Image.Image, max_pixels: int, min_side: int = 64) -> Image.Image:
    w, h = img.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(min_side, int(w * s)), max(min_side, int(h * s))), Image.LANCZOS)
    w, h = img.size
    if min(w, h) < min_side:
        s = min_side / min(w, h)
        img = img.resize((max(min_side, round(w * s)), max(min_side, round(h * s))), Image.LANCZOS)
    return img


@dataclass
class VLMOutput:
    raw: str
    parsed: dict | None
    seconds: float


class VLM:
    name = "base"
    model_name = "base"

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int,
                 schema: dict | None = None) -> VLMOutput:  # pragma: no cover
        raise NotImplementedError

    def text(self, prompt: str, images: list[Image.Image] | None = None, max_tokens: int = 2048,
             schema: dict | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def release(self) -> None:
        pass

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name}


def _die_with_parent() -> None:  # pragma: no cover (Linux only)
    """Linux: SIGKILL the llama-server child when this process dies (no orphan holding VRAM)."""
    try:
        import ctypes as _c
        import signal
        _c.CDLL("libc.so.6").prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


_JOB = None


def _bind_to_parent(proc: subprocess.Popen) -> None:
    """Windows: put the child in a Job Object with KILL_ON_JOB_CLOSE so it dies with
    this Python process, however that process ends (crash, kill, Ctrl+C)."""
    global _JOB
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    if _JOB is None:
        job = k32.CreateJobObjectW(None, None)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)):
            log.warning("SetInformationJobObject failed: %s", ctypes.get_last_error())
            return
        _JOB = job
    h = k32.OpenProcess(0x0001 | 0x0100, False, proc.pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
    if not h or not k32.AssignProcessToJobObject(wintypes.HANDLE(_JOB), wintypes.HANDLE(h)):
        log.warning("AssignProcessToJobObject failed: %s", ctypes.get_last_error())
    if h:
        k32.CloseHandle(wintypes.HANDLE(h))


class LlamaCppVLM(VLM):
    """A vision GGUF (Gemma 4 E4B) served by llama.cpp's llama-server (CUDA build).

    JSON is enforced by a grammar compiled from the JSON schema (`response_format`), so output is
    always syntactically valid. `release()` stops the server process, which returns all its VRAM
    before the localizer runs (residency "swap")."""
    name = "llama.cpp"

    def __init__(self, s: Settings, gguf: str, mmproj: str, model_name: str, gpu_layers: int = 99,
                 extra_args: list[str] | None = None, extra_tokens: int = 0, slot: str = "gemma",
                 port: int | None = None):
        self.s = s
        self.slot = slot                  # GPU-arbiter name; a second VLM (hybrid scene model) gets its own
        self.gguf, self.mmproj, self.gpu_layers = gguf, mmproj, gpu_layers
        self.extra_args, self.extra_tokens = extra_args or [], extra_tokens
        self.model_name = model_name
        self.proc: subprocess.Popen | None = None
        self.port = port or s.llama_port
        self.url = f"http://127.0.0.1:{self.port}"
        self.load_s = 0.0
        self._atexit = False
        get_arbiter().register(slot, self.release)

    def _start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        for p in (self.s.llama_server, self.gguf, self.mmproj):
            if not Path(p).exists():
                raise RuntimeError(f"missing {p} (run scripts/setup_models.py, or set PI_LLAMA_SERVER / PI_GEMMA_GGUF)")
        cmd = [self.s.llama_server, "-m", self.gguf, "--mmproj", self.mmproj,
               "-ngl", str(self.gpu_layers), "-c", str(self.s.llama_ctx), "-np", "1", "--host", "127.0.0.1",
               "--port", str(self.port), "--image-max-tokens", str(self.s.llama_image_max_tokens),
               "--no-webui", "-fa", "on"] + self.extra_args
        log_path = Path(self.s.data_dir) / "llama_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        self._log = open(log_path, "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags,
                                     preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
        _bind_to_parent(self.proc)   # an orphaned server would silently hold ~2.5 GB of VRAM
        if not self._atexit:
            atexit.register(self.release)
            self._atexit = True
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see {log_path}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200 and b"ok" in r.read():
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)
        else:
            self.release()
            raise RuntimeError("llama-server did not become ready in 180 s")
        self.load_s = time.perf_counter() - t
        log.info("llama-server ready in %.1fs", self.load_s)

    def release(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        get_arbiter().mark_released(self.slot)

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int,
                 schema: dict | None = None) -> VLMOutput:
        content = []
        for im in images:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=92)
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}})
        content.append({"type": "text", "text": prompt})
        body: dict = {"messages": [{"role": "user", "content": content}], "temperature": 0.0, "top_k": 1,
                      "max_tokens": max_new_tokens + self.extra_tokens, "repeat_penalty": 1.05, "cache_prompt": False}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        with get_arbiter().use(self.slot):
            self._start()
            t = time.perf_counter()
            req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            dt = time.perf_counter() - t
        raw = resp["choices"][0]["message"].get("content") or ""
        return VLMOutput(raw=raw, parsed=extract_json(raw), seconds=dt)

    def text(self, prompt: str, images: list[Image.Image] | None = None, max_tokens: int = 2048,
             schema: dict | None = None) -> str:
        return self.generate(images or [], prompt, max_tokens, schema).raw

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name, "gguf": self.gguf,
                "running": bool(self.proc and self.proc.poll() is None),
                "available": Path(self.s.llama_server).exists() and Path(self.gguf).exists()}


class MockVLM(VLM):
    """Deterministic stand-in so the whole pipeline/API/UI can run without models."""
    name = "mock"
    model_name = "MockVLM"

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int,
                 schema: dict | None = None) -> VLMOutput:
        h = hashlib.md5(images[-1].tobytes()[:4096] + prompt.encode()).digest()
        if "Allowed room values" in prompt:
            d = {"room": "kitchen", "image_type": "overview",
                 "components": [{"name": "countertop"}, {"name": "wall tiles"}, {"name": "cabinets"},
                                {"name": "range hood"}, {"name": "floor tiles"}]}
        else:
            comp = prompt.split("Expected component:\n", 1)[-1].split("\n", 1)[0]
            cand = prompt.split("Candidate defect:\n", 1)[-1].split("\n", 1)[0]
            present = h[0] % 4 != 0
            d = {"component": comp, "defect_present": present, "defect_type": cand if present else "",
                 "severity": ["minor", "moderate", "unknown"][h[1] % 3] if present else "unknown",
                 "description": f"Mock: visible {cand} on the {comp}." if present else "Mock: normal surface.",
                 "alternative_explanation": "" if present else "shadow",
                 "manual_check_required": False}
        raw = json.dumps(d)
        return VLMOutput(raw=raw, parsed=d, seconds=0.0)

    def text(self, prompt: str, images: list[Image.Image] | None = None, max_tokens: int = 2048,
             schema: dict | None = None) -> str:
        if schema:
            return json.dumps({})
        return "Mock Gemini answer: based on the inspection record, see the findings table and risk summary."


class GeminiVLM(VLM):
    """Google Gemini (cloud) via the google-genai SDK: Gemini API key or Vertex AI.
    No local GPU use. JSON is enforced with `response_json_schema`."""
    name = "gemini"

    _client_lock = threading.Lock()      # one shared client: parallel threads must not race to create it
    _shared: dict = {}

    def __init__(self, s: Settings):
        self.s = s
        self.model_name = s.gemini_model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        key_id = ("vertex", self.s.gcp_project, self.s.gcp_location) if self.s.gemini_vertex else ("api",)
        with GeminiVLM._client_lock:
            if key_id in GeminiVLM._shared:
                self._client = GeminiVLM._shared[key_id]
                return self._client
            import os
            from google import genai
            if self.s.gemini_vertex:
                self._client = genai.Client(vertexai=True, project=self.s.gcp_project or None,
                                            location=self.s.gcp_location,
                                            http_options=genai.types.HttpOptions(timeout=120_000))
            else:
                key = self.s.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not key:
                    raise RuntimeError("Gemini: set PI_GEMINI_API_KEY / GEMINI_API_KEY, or PI_GEMINI_VERTEX=1 + PI_GCP_PROJECT")
                self._client = genai.Client(api_key=key, http_options=genai.types.HttpOptions(timeout=120_000))
            GeminiVLM._shared[key_id] = self._client
        return self._client

    def generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int,
                 schema: dict | None = None) -> VLMOutput:
        from google.genai import types
        parts = []
        for im in images:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=92)
            parts.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg"))
        parts.append(prompt)
        cfg = dict(temperature=0.0, max_output_tokens=max(max_new_tokens, 512), response_mime_type="application/json")
        if schema:
            cfg["response_json_schema"] = schema
        if "flash" in self.s.gemini_model:   # no hidden "thinking" tokens: faster, cheaper, bounded output
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        t = time.perf_counter()
        for attempt in range(8):
            try:
                resp = self._get_client().models.generate_content(
                    model=self.s.gemini_model, contents=parts, config=types.GenerateContentConfig(**cfg))
                break
            except Exception as e:  # noqa: BLE001  (429 / 503: back off and retry, ~5 min total)
                if attempt == 7 or "PerDay" in str(e) or \
                        not any(c in str(e) for c in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "timed out", "Timeout")):
                    raise
                time.sleep(min(60, 2 ** attempt * 5))
        raw = resp.text or ""
        return VLMOutput(raw=raw, parsed=extract_json(raw), seconds=time.perf_counter() - t)

    def text(self, prompt: str, images: list[Image.Image] | None = None, max_tokens: int = 2048,
             schema: dict | None = None) -> str:
        """Free-text / JSON generation for reports and the underwriting assistant."""
        from google.genai import types
        parts = []
        for im in images or []:
            b = io.BytesIO()
            im.save(b, "JPEG", quality=90)
            parts.append(types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg"))
        parts.append(prompt)
        cfg = dict(temperature=0.2, max_output_tokens=max_tokens)
        if schema:
            cfg.update(response_mime_type="application/json", response_json_schema=schema)
        if "flash" in self.s.gemini_model:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        for attempt in range(6):
            try:
                resp = self._get_client().models.generate_content(
                    model=self.s.gemini_model, contents=parts, config=types.GenerateContentConfig(**cfg))
                return resp.text or ""
            except Exception as e:  # noqa: BLE001
                if attempt == 5 or not any(c in str(e) for c in ("429", "500", "503", "UNAVAILABLE", "INTERNAL", "timed out")):
                    raise
                time.sleep(min(30, 2 ** attempt * 3))
        return ""

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name,
                "mode": "vertex" if self.s.gemini_vertex else "api_key"}


_GEMMA: dict[str, VLM] = {}


def create_vlm(s: Settings, backend: str | None = None) -> VLM:
    """backend = gemini | gemma | mock (default: s.vlm_backend). One Gemma server is shared by all roles."""
    b = backend or s.vlm_backend
    if b == "mock" or s.mock_models:
        return MockVLM()
    if b == "gemma":
        key = s.gemma_gguf
        if key not in _GEMMA:
            stem = Path(s.gemma_gguf).stem                    # gemma-4-E4B-it-Q4_0 -> "Gemma-4-E4B Q4_0"
            name = stem.replace("-it-", " ").replace("gemma", "Gemma", 1)
            _GEMMA[key] = LlamaCppVLM(s, s.gemma_gguf, s.gemma_mmproj, name, s.gemma_gpu_layers,
                                      ["--reasoning", "off"])
        return _GEMMA[key]
    return GeminiVLM(s)


# ---------------------------------------------------------------- task wrappers


class VisionService:
    """Scene + verification + validation prompts on top of per-role VLM backends."""

    def __init__(self, vlm: VLM, tax: Taxonomy, s: Settings, scene_vlm: VLM | None = None,
                 validator: VLM | None = None, text_vlm: VLM | None = None):
        self.vlm, self.tax, self.s = vlm, tax, s          # vlm = crop verifier
        self.scene_vlm = scene_vlm or vlm
        self.validator = validator                        # None = no second opinion
        self.text_vlm = text_vlm or (validator if validator is not None else vlm)

    @classmethod
    def from_settings(cls, tax: Taxonomy, s: Settings) -> "VisionService":
        validator = None if s.validate_backend in ("off", "") else create_vlm(s, s.validate_backend)
        return cls(create_vlm(s, s.verify_backend), tax, s, scene_vlm=create_vlm(s, s.scene_backend),
                   validator=validator, text_vlm=create_vlm(s, s.vlm_backend))

    def release(self) -> None:
        for m in {id(x): x for x in (self.vlm, self.scene_vlm, self.validator, self.text_vlm) if x}.values():
            m.release()

    def text(self, prompt: str, images: list[Image.Image] | None = None, max_tokens: int = 2048) -> str:
        return self.text_vlm.text(prompt, images, max_tokens=max_tokens)

    def identify_scene(self, img: Image.Image) -> tuple[SceneResult, str]:
        rooms = ", ".join(k for k in self.tax.rooms)
        comps = ", ".join(sorted({c.label.lower() for c in self.tax.components.values()}))
        prompt = SCENE_PROMPT.format(rooms=rooms, components=comps)
        out = self.scene_vlm.generate([_limit_pixels(img, self.s.vlm_scene_max_pixels)], prompt,
                                      self.s.vlm_max_new_tokens,
                                schema=SCENE_SCHEMA)
        return self.parse_scene(out.parsed, out.raw), out.raw

    def parse_scene(self, d: dict | None, raw: str = "") -> SceneResult:
        if not d:
            return SceneResult(room="unknown", source="fallback", valid_json=False, notes="unparseable scene output")
        raw_room = str(d.get("room", ""))
        comps: list[SceneComponent] = []
        seen = set()
        items = d.get("components") or []
        if isinstance(items, dict):
            items = [{"name": k} for k in items]
        for c in items:
            name = c.get("name", "") if isinstance(c, dict) else str(c)
            note = (c.get("note") or c.get("condition") or "") if isinstance(c, dict) else ""
            key = self.tax.normalize_component(name)
            if re.search(r"not (clearly )?visible|none|absent", str(note).lower()):
                continue
            if key and key not in seen:
                seen.add(key)
                comps.append(SceneComponent(name=key, raw_name=name, visible_condition_note=str(note)[:200]))
        it = str(d.get("image_type", "unknown")).lower().replace("-", "_").replace(" ", "_")
        return SceneResult(room=self.tax.normalize_room(raw_room), raw_room=raw_room, components=comps,
                           image_type=it if it in ("overview", "component", "close_up") else "unknown",
                           authenticity=str(d.get("authenticity") or ""))

    def verify_defect(self, crop_img: Image.Image, context_img: Image.Image | None,
                      component: str, defect: str, strict: bool | None = None,
                      vlm: VLM | None = None) -> tuple[VerificationResult, str]:
        images, views = [], ""
        if context_img is not None:
            images.append(_limit_pixels(context_img, self.s.vlm_context_max_pixels))
            views = "Image 1 is the whole scene with the candidate region outlined in red. Image 2 is a close-up of that region."
        else:
            views = ("The image is a full-resolution close-up; the candidate region is outlined in red. "
                     "Judge only what is inside the red outline, not the surroundings.")
        images.append(_limit_pixels(crop_img, self.s.vlm_crop_max_pixels))
        cand = self.tax.defect_label(defect).lower()
        prompt = VERIFY_PROMPT.format(views=views, component=self.tax.component_label(component).lower(),
                                      candidate_defect=cand,
                                      hard_negatives=", ".join(self.tax.hard_negatives_for(defect)))
        if self.s.verify_strict if strict is None else strict:
            prompt += VERIFY_STRICT
        out = (vlm or self.vlm).generate(images, prompt, 160, schema=VERIFY_SCHEMA)
        return self.parse_verification(out.parsed, component, defect), out.raw

    def validate_defect(self, crop_img: Image.Image, component: str, defect: str) -> tuple[VerificationResult, str]:
        """Independent second opinion (strict prompt) by the validator model on the same outlined crop."""
        return self.verify_defect(crop_img, None, component, defect, strict=True, vlm=self.validator)

    def parse_verification(self, d: dict | None, component: str, defect: str) -> VerificationResult:
        if not d:
            return VerificationResult(component=component, defect_present=None, defect_type=defect, valid_json=False)
        present = to_bool(d.get("defect_present"))
        dtype = str(d.get("defect_type") or "")
        return VerificationResult(
            component=str(d.get("component") or component),
            defect_present=present,
            defect_type=self.tax.normalize_defect(dtype, component) if dtype else defect,
            severity=d.get("severity"),
            description=str(d.get("description") or "")[:600],
            alternative_explanation=str(d.get("alternative_explanation") or "")[:200],
            manual_check_required=bool(to_bool(d.get("manual_check_required"))),
        )
