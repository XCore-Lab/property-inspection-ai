"""Optional specialist evidence (spec phase 13 hook).

A lightweight, training-free OpenCV analyser that measures, inside a crop:
  * linear_score - thin, elongated dark structures (crack / scratch-like)
  * regularity   - how grid-like those structures are (grout joints, tile edges,
                   patterns - the classic crack false positive)
  * blotch_score - compact tonal anomaly area (stain / water-mark-like)

It is ADVISORY: the rule engine only uses it to downgrade a verifier "yes" to
NEEDS_REVIEW when the evidence contradicts it. A trained crack/scratch model can
replace `analyze()` behind the same interface once benchmarking justifies it.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from backend.models.schemas import SpecialistEvidence


def analyze(crop: Image.Image) -> SpecialistEvidence:
    g = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2GRAY)
    s = 512 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = g.shape
    if min(h, w) < 16:
        return SpecialistEvidence(note="crop too small")
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)

    # --- thin dark structures (black-hat) -> elongated connected components
    k = max(7, (min(h, w) // 24) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thr = max(12.0, float(bh.mean() + 2.5 * bh.std()))
    mask = (bh > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    diag = float(np.hypot(h, w))
    lin_len = 0.0
    for i in range(1, n):
        x, y, bw, bhh, area = stats[i]
        if area < 12:
            continue
        pts = np.column_stack(np.where(lab[y:y + bhh, x:x + bw] == i))
        (_, _), (a, b), _ = cv2.minAreaRect(pts.astype(np.float32))
        long_, short_ = max(a, b), max(1.0, min(a, b))
        if long_ / short_ >= 4 and area / max(1.0, long_ * short_) < 0.8 or long_ > 0.25 * diag:
            lin_len += long_
    linear = float(min(1.0, lin_len / (1.5 * diag)))

    # --- grid regularity from Hough line orientations
    edges = cv2.Canny(g, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=int(0.2 * min(h, w)), maxLineGap=4)
    regular = 0.0
    if lines is not None and len(lines) >= 3:
        L = np.asarray(lines).reshape(-1, 4).astype(np.float32)
        ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
        ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
        hist = np.zeros(180)
        np.add.at(hist, ang.astype(int) % 180, ln)
        dom = int(np.argmax(np.convolve(np.r_[hist[-4:], hist, hist[:4]], np.ones(9), "valid")))
        near = lambda a0: np.minimum(np.abs(ang - a0) % 180, 180 - np.abs(ang - a0) % 180) <= 4  # noqa: E731
        on_grid = ln[near(dom) | near((dom + 90) % 180)].sum()
        regular = float(on_grid / max(1e-6, ln.sum())) * min(1.0, len(lines) / 6)

    # --- blotches: compact tonal anomaly vs. local median
    med = cv2.medianBlur(g, max(3, (min(h, w) // 6) | 1))
    dev = cv2.absdiff(g, med)
    bm = (dev > max(18.0, float(dev.mean() + 2 * dev.std()))).astype(np.uint8)
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blotch = float(min(1.0, bm.mean() * 5))

    note = []
    if regular > 0.7:
        note.append("grid-like straight lines (joints/pattern)")
    if linear > 0.25:
        note.append("thin elongated dark structures")
    if blotch > 0.3:
        note.append("tonal blotches")
    return SpecialistEvidence(linear_score=round(linear, 3), regularity=round(regular, 3),
                              blotch_score=round(blotch, 3), note="; ".join(note))
