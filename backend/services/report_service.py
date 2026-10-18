"""Annotated images (phase 19) and inspection reports (phase 20): HTML, Markdown, text."""
from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw, ImageFont

from backend.models.schemas import Detection, Inspection, ResultState

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
STATE_COLORS = {
    ResultState.DETECTED: (220, 38, 38),
    ResultState.NEEDS_REVIEW: (234, 150, 12),
    ResultState.UNKNOWN: (37, 99, 235),
    ResultState.MANUAL_CHECK_REQUIRED: (37, 99, 235),
    ResultState.NOT_DETECTED: (120, 120, 120),
}
STATE_LABELS = {
    ResultState.DETECTED: "Detected",
    ResultState.NEEDS_REVIEW: "Needs Review",
    ResultState.UNKNOWN: "Unknown",
    ResultState.MANUAL_CHECK_REQUIRED: "Manual Check",
    ResultState.NOT_DETECTED: "Not Detected",
}


def _font(size: int) -> ImageFont.ImageFont:
    for f in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(img: Image.Image, detections: list[Detection], include_rejected: list[Detection] | None = None) -> Image.Image:
    """Draw boxes coloured by state with '#ID Component · Defect' labels and a legend."""
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")
    W, H = out.size
    t = max(2, round(max(W, H) / 400))
    fs = max(14, round(max(W, H) / 60))
    font = _font(fs)
    items = list(detections) + list(include_rejected or [])
    # draw larger boxes first so small ones stay visible on top
    for det in sorted(items, key=lambda x: -x.localization.area):
        b = det.localization
        col = STATE_COLORS.get(det.state, (120, 120, 120))
        dash = det.state == ResultState.NOT_DETECTED
        if dash:
            _dashed_rect(d, (b.x1, b.y1, b.x2, b.y2), col, max(1, t // 2))
        else:
            d.rectangle((b.x1, b.y1, b.x2, b.y2), outline=col + (255,), width=t)
            d.rectangle((b.x1, b.y1, b.x2, b.y2), fill=col + (28,))
        label = f"#{det.issue_id.split('-')[-1]} {det.component_label} · {det.defect.label}"
        if det.state != ResultState.DETECTED:
            label += f" ({STATE_LABELS[det.state]})"
        tw, th = _text_size(d, label, font)
        lx = min(max(0, b.x1), max(0, W - tw - 8))
        ly = b.y1 - th - 8 if b.y1 - th - 8 >= 0 else min(H - th - 6, b.y2 + 2)
        d.rectangle((lx, ly, lx + tw + 8, ly + th + 6), fill=col + (235,))
        d.text((lx + 4, ly + 2), label, fill=(255, 255, 255), font=font)
    # legend
    lf = _font(max(12, fs * 3 // 4))
    x, y = 8, H - 8
    entries = [(STATE_LABELS[s], STATE_COLORS[s]) for s in (ResultState.DETECTED, ResultState.NEEDS_REVIEW, ResultState.UNKNOWN)]
    lh = _text_size(d, "Ag", lf)[1] + 8
    total_w = sum(_text_size(d, e[0], lf)[0] + lh + 14 for e in entries)
    d.rectangle((x - 4, y - lh - 4, x + total_w, y + 2), fill=(0, 0, 0, 150))
    for name, col in entries:
        d.rectangle((x, y - lh + 2, x + lh - 6, y - 4), fill=col + (255,))
        d.text((x + lh, y - lh + 1), name, fill=(255, 255, 255), font=lf)
        x += _text_size(d, name, lf)[0] + lh + 14
    return out


def _text_size(d: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    return r - l, b - t


def _dashed_rect(d: ImageDraw.ImageDraw, box, col, width: int, dash: int = 10) -> None:
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        n = max(1, int(length // (2 * dash)))
        for i in range(n):
            s0, s1 = (2 * i) / (2 * n), (2 * i + 1) / (2 * n)
            d.line((ax + (bx - ax) * s0, ay + (by - ay) * s0, ax + (bx - ax) * s1, ay + (by - ay) * s1),
                   fill=col + (255,), width=width)


def _b64_img(path: Path, max_side: int = 1400, quality: int = 85) -> str:
    if not path.exists():
        return ""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html", "j2"]))


def render_html(inspections: list[Inspection], out_dirs: list[Path], title: str = "Property Inspection Report",
                property_id: str = "", property_risk: dict | None = None) -> str:
    blocks = []
    for ins, od in zip(inspections, out_dirs):
        crops = {d.issue_id: _b64_img(od / d.crop_path, 360, 80) if d.crop_path else "" for d in ins.detections}
        blocks.append({"ins": ins, "annotated": _b64_img(od / "annotated.jpg"), "crops": crops})
    return _env.get_template("report.html.j2").render(
        title=title, blocks=blocks, property_id=property_id or (inspections[0].property_id if inspections else ""),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"), S=ResultState, state_labels=STATE_LABELS,
        property_risk=property_risk if len(inspections) > 1 or property_risk else None)


def render_markdown(ins: Inspection) -> str:
    L = ["# PROPERTY INSPECTION REPORT", "",
         f"- **Property ID:** {ins.property_id or '-'}",
         f"- **Inspection ID:** {ins.inspection_id}",
         f"- **Date:** {ins.created_at:%Y-%m-%d %H:%M}",
         f"- **Room:** {ins.room_label}",
         f"- **Source image:** {ins.source_image}", ""]
    if ins.image_quality and ins.image_quality.warnings:
        L += ["> **Image quality:** " + " ".join(ins.image_quality.warnings), ""]
    L += ["![Annotated image](annotated.jpg)", "", "## Detected issues", ""]
    active = [d for d in ins.detections if d.state != ResultState.NOT_DETECTED]
    if not active:
        L += ["No visible defects were confirmed in this image.", ""]
    for d in active:
        L += [f"### #{d.issue_id} — {d.component_label}: {d.defect.label}", "",
              f"- **Status:** {STATE_LABELS[d.state]}",
              f"- **Severity (visual):** {d.defect.severity}",
              f"- **Observation:** {d.description or '-'}",
              f"- **Location (px):** ({d.localization.x1:.0f}, {d.localization.y1:.0f}) – ({d.localization.x2:.0f}, {d.localization.y2:.0f})"]
        if d.review_reasons:
            L.append("- **Why review:** " + " ".join(d.review_reasons))
        if d.professional_review:
            L.append("- **Professional assessment needed:** " + ", ".join(d.professional_review))
        L.append("")
    L += ["## Component summary", "", "| Component | Visible condition | Status | Manual checks |", "|---|---|---|---|"]
    for c in ins.components:
        L.append(f"| {c.label} | {c.visible_summary} | {STATE_LABELS[c.visible_state]} | "
                 f"{', '.join(c.manual_checks) or '-'} |")
    L.append("")
    if ins.manual_checks:
        L += ["## Manual / functional checks required", ""]
        for m in ins.manual_checks:
            L.append(f"- [ ] **{m['component_label']}** — {m['check']} _(Manual check required: {m['reason']})_")
        L.append("")
    if ins.unknown_items:
        L += ["## Unknown / unverified items", ""] + [f"- {u}" for u in ins.unknown_items] + [""]
    L += ["---", f"_{ins.disclaimer}_", "",
          f"_Models: localization {ins.options.get('models', {}).get('localization', '')}, "
          f"verification {ins.options.get('models', {}).get('verification', '')}. "
          f"Runtime {ins.timings.total_s:.1f}s._"]
    return "\n".join(L)


def render_text(ins: Inspection) -> str:
    """Plain-text report in the spec's example layout."""
    L = ["PROPERTY INSPECTION REPORT", "", f"Property ID: {ins.property_id or '-'}",
         f"Inspection ID: {ins.inspection_id}", f"Date: {ins.created_at:%Y-%m-%d}", "", ins.room_label, ""]
    by_comp: dict[str, list[Detection]] = {}
    for d in ins.detections:
        if d.state != ResultState.NOT_DETECTED:
            by_comp.setdefault(d.component, []).append(d)
    for c in ins.components:
        L += ["Component:", c.label, ""]
        ds = by_comp.get(c.component, [])
        if ds:
            for d in ds:
                L += ["Issue:", f"Visible {d.defect.label.lower()}", "", "Observation:", d.description or "-", "",
                      "Status:", STATE_LABELS[d.state], ""]
        else:
            L += ["Visible Damage:", "No clear external damage identified.", ""]
        if c.manual_checks:
            L += ["Functional Test:", "Manual check required (" + ", ".join(c.manual_checks) + ").", ""]
        L.append("---------------------------------")
        L.append("")
    return "\n".join(L)
