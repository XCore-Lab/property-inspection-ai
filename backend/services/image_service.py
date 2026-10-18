"""Image I/O, quality checks, crops, tiling and box geometry."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from backend.models.schemas import BBox, ImageQuality

# ---------------------------------------------------------------- loading / preprocessing


def load_image(data: bytes | str | Path) -> Image.Image:
    """Decode, apply EXIF orientation, force RGB. Raises ValueError on bad input."""
    try:
        img = Image.open(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
        img = ImageOps.exif_transpose(img)
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot decode image: {e}") from e
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    if min(img.size) < 32:
        raise ValueError(f"image too small: {img.size}")
    return img


def resize_max_side(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Returns (resized, scale) where original_coord = resized_coord / scale."""
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s >= 1.0:
        return img, 1.0
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS), s


def assess_quality(img: Image.Image) -> ImageQuality:
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, (0, 0), fx=min(1.0, 1024 / max(g.shape)), fy=min(1.0, 1024 / max(g.shape)),
                       interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    bright = float(small.mean())
    contrast = float(small.std())
    warn = []
    w, h = img.size
    if max(w, h) < 1000:   # e.g. 480p video frames (~0.4 MP): too coarse to confirm holes/cracks/scratches
        warn.append(f"Low resolution ({w}x{h}); small defects cannot be confirmed and fine cracks/scratches may be missed.")
    if blur < 40:
        warn.append("Image appears blurry; fine defects may be missed or misread.")
    if bright < 45:
        warn.append("Image is dark; stains and cracks may be hidden.")
    elif bright > 225:
        warn.append("Image is over-exposed; surface detail may be lost.")
    if contrast < 18:
        warn.append("Very low contrast image.")
    return ImageQuality(width=w, height=h, blur_score=round(blur, 1), brightness=round(bright, 1),
                        contrast=round(contrast, 1), ok=not warn, warnings=warn)


def to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=quality)
    return b.getvalue()


# ---------------------------------------------------------------- boxes


def clip_box(b: BBox, w: int, h: int) -> BBox:
    x1, x2 = sorted((b.x1, b.x2))
    y1, y2 = sorted((b.y1, b.y2))
    return BBox(x1=min(max(0, x1), w), y1=min(max(0, y1), h), x2=min(max(0, x2), w), y2=min(max(0, y2), h))


def valid_box(b: BBox, w: int, h: int, min_side: int = 6) -> bool:
    vals = b.as_list()
    if any(not np.isfinite(v) for v in vals):
        return False
    return b.w >= min_side and b.h >= min_side and b.x2 <= w + 1 and b.y2 <= h + 1


def containment(a: BBox, b: BBox) -> float:
    """Share of the smaller box that lies inside the other one (1.0 = fully inside)."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    m = min(a.area, b.area)
    return ix * iy / m if m > 0 else 0.0


def iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    u = a.area + b.area - inter
    return inter / u if u > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner` covered by `outer`."""
    ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    return (ix * iy) / inner.area if inner.area > 0 else 0.0


def padded_crop_box(b: BBox, w: int, h: int, pad: float = 0.15, min_side: int = 96) -> BBox:
    bw, bh = b.w, b.h
    px, py = bw * pad, bh * pad
    x1, y1, x2, y2 = b.x1 - px, b.y1 - py, b.x2 + px, b.y2 + py
    # grow tiny crops to min_side around their centre so the VLM gets context
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if x2 - x1 < min_side:
        x1, x2 = cx - min_side / 2, cx + min_side / 2
    if y2 - y1 < min_side:
        y1, y2 = cy - min_side / 2, cy + min_side / 2
    # shift back inside the image instead of truncating when possible
    if x1 < 0:
        x2, x1 = min(w, x2 - x1), 0
    if y1 < 0:
        y2, y1 = min(h, y2 - y1), 0
    if x2 > w:
        x1, x2 = max(0, x1 - (x2 - w)), w
    if y2 > h:
        y1, y2 = max(0, y1 - (y2 - h)), h
    return BBox(x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2))


def crop(img: Image.Image, b: BBox) -> Image.Image:
    return img.crop((int(b.x1), int(b.y1), int(b.x2), int(b.y2)))


def boxed_crop(img: Image.Image, b: BBox, crop_box: BBox) -> Image.Image:
    """Crop `crop_box` and outline the candidate box `b` inside it in red. This is
    what the verifier sees: local context, explicit focus region, and no view of
    the rest of the scene (a whole-scene view made the 2B VLM confirm defects that
    were visible elsewhere in the image)."""
    c = crop(img, crop_box).copy()
    arr = np.asarray(c).copy()
    t = max(2, round(max(c.size) / 200))
    cv2.rectangle(arr, (int(b.x1 - crop_box.x1), int(b.y1 - crop_box.y1)),
                  (int(b.x2 - crop_box.x1), int(b.y2 - crop_box.y1)), (255, 0, 0), t)
    return Image.fromarray(arr)


def context_view(img: Image.Image, b: BBox, max_side: int = 768) -> Image.Image:
    """Whole image (downscaled) with the candidate region outlined, for the verifier."""
    small, s = resize_max_side(img, max_side)
    arr = np.asarray(small).copy()
    t = max(2, round(max(small.size) / 250))
    cv2.rectangle(arr, (int(b.x1 * s), int(b.y1 * s)), (int(b.x2 * s), int(b.y2 * s)), (255, 0, 0), t)
    return Image.fromarray(arr)


# ---------------------------------------------------------------- tiling (phase 12)


@dataclass
class Tile:
    x0: int
    y0: int
    image: Image.Image


def make_tiles(img: Image.Image, tile: int = 1024, overlap: float = 0.2) -> list[Tile]:
    w, h = img.size
    step = max(1, int(tile * (1 - overlap)))

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        if s[-1] != n - tile:
            s.append(n - tile)
        return s

    return [Tile(x, y, img.crop((x, y, min(w, x + tile), min(h, y + tile)))) for y in starts(h) for x in starts(w)]


def nms(boxes: list[BBox], iou_thr: float = 0.5, order: list[float] | None = None) -> list[int]:
    """Greedy NMS. The localizer gives no scores, so by default smaller (tighter) boxes
    are preferred. A box that is >=90% contained in / containing a kept box is also
    suppressed. Returns kept indices."""
    idx = list(range(len(boxes)))
    key = order if order is not None else [-b.area for b in boxes]
    idx.sort(key=lambda i: key[i], reverse=True)
    keep: list[int] = []
    for i in idx:
        if all(iou(boxes[i], boxes[k]) < iou_thr and containment(boxes[i], boxes[k]) < 0.9
               and containment(boxes[k], boxes[i]) < 0.9 for k in keep):
            keep.append(i)
    return keep
