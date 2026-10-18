"""Evidence-integrity (fraud) signals for submitted property photos.

  * duplicate / near-duplicate photo already used in an earlier inspection (perceptual dHash),
    high risk when it was submitted for a DIFFERENT property (classic claim / valuation fraud)
  * EXIF: missing camera metadata, editing software, capture date long before submission
  * Gemini authenticity check (photo of a screen, rendering, AI-generated) - added after the scene step

Signals are advisory: they route the case to a human (REFER), they never reject automatically.
"""
from __future__ import annotations

import io
import json
import threading
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

from backend.config.settings import Settings

_lock = threading.Lock()
EDITORS = ("photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "facetune", "affinity", "pixelmator")


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.tobytes())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def _exif(image_bytes: bytes) -> dict:
    try:
        raw = Image.open(io.BytesIO(image_bytes)).getexif()
    except Exception:  # noqa: BLE001
        return {}
    out = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}
    try:
        sub = raw.get_ifd(0x8769)   # Exif sub-IFD: DateTimeOriginal lives here
        out.update({ExifTags.TAGS.get(k, str(k)): v for k, v in sub.items()})
    except Exception:  # noqa: BLE001
        pass
    return {k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in out.items()
            if k in ("Make", "Model", "Software", "DateTime", "DateTimeOriginal", "GPSInfo", "LensModel")}


def check(image_bytes: bytes, img: Image.Image, ins, s: Settings) -> dict:
    flags: list[dict] = []
    exif = _exif(image_bytes)
    if not exif.get("Make") and not exif.get("Model"):
        flags.append({"code": "no_camera_metadata", "level": "low",
                      "message": "No camera metadata (screenshot, messaging-app copy or edited export)"})
    sw = str(exif.get("Software", "")).lower()
    if any(e in sw for e in EDITORS):
        flags.append({"code": "edited", "level": "medium", "message": f"Edited with {exif.get('Software')}"})
    taken = exif.get("DateTimeOriginal") or exif.get("DateTime")
    if taken:
        try:
            dt = datetime.strptime(str(taken)[:19], "%Y:%m:%d %H:%M:%S")
            age = (datetime.now() - dt).days
            if age > 365:
                flags.append({"code": "old_photo", "level": "medium",
                              "message": f"Photo taken {age // 30} months before submission ({dt:%Y-%m-%d})"})
        except ValueError:
            pass

    h = dhash(img)
    index_path = Path(s.data_dir) / "photo_index.json"
    with _lock:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        except Exception:  # noqa: BLE001
            index = []
        best = None
        for e in index:
            dist = bin(int(e["hash"], 16) ^ h).count("1")
            if dist <= 6 and (best is None or dist < best[0]):
                best = (dist, e)
        if best:
            dist, e = best
            other_prop = e.get("property_id") or ""
            if ins.property_id and other_prop and other_prop != ins.property_id:
                flags.append({"code": "duplicate_other_property", "level": "high",
                              "message": f"Same photo already submitted for property {other_prop} "
                                         f"({e['inspection_id']})", "match": e["inspection_id"], "distance": dist})
            else:
                flags.append({"code": "duplicate_photo", "level": "medium",
                              "message": f"Photo previously submitted ({e['inspection_id']})",
                              "match": e["inspection_id"], "distance": dist})
        index.append({"hash": f"{h:016x}", "inspection_id": ins.inspection_id, "property_id": ins.property_id,
                      "at": datetime.now().isoformat(timespec="seconds")})
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index[-5000:]), encoding="utf-8")

    level = "high" if any(f["level"] == "high" for f in flags) else \
            "medium" if any(f["level"] == "medium" for f in flags) else "low"
    return {"dhash": f"{h:016x}", "exif": exif, "flags": flags, "level": level}


def add_authenticity(integrity: dict | None, authenticity: str) -> dict:
    """Merge Gemini's authenticity verdict from the scene step."""
    integrity = integrity or {"flags": [], "level": "low"}
    a = (authenticity or "").lower()
    integrity["authenticity"] = a or "unknown"
    msg = {"screen_photo": "Looks like a photo of a screen / printout, not the property itself",
           "rendering": "Looks like a 3D rendering or catalogue image",
           "ai_generated": "May be AI-generated or digitally manipulated"}.get(a)
    if msg:
        integrity["flags"].append({"code": f"authenticity_{a}", "level": "high", "message": msg})
        integrity["level"] = "high"
    return integrity
