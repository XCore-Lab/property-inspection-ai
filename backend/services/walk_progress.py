"""Progress % + ETA for a walkthrough job, estimated from its log.

scripts/inspect_walkthrough.py (and the 3D engine, whose output it forwards) print a marker
line at the start of every step. Each step has an expected duration measured on a real run
(jaan_house: 197 SLAM frames, 14 AI keyframes, 42 min on the RTX 3050), scaled by the
actual frame / keyframe / furniture counts once the log reports them. Progress inside the
running step is interpolated from its elapsed time (capped, so the bar never claims a step
is finished early), and the ETA is corrected by how fast the finished steps actually ran.
Steps that are skipped (e.g. SLAM output reused) simply count as done.
"""
from __future__ import annotations

import re
import time

REF_FRAMES = 197
# key, label, marker regex, expected seconds on the reference run, scales with
PHASES = [
    ("p1", "Depth Anything 3 (depth per frame)", r"^\[1\] ", 200, "frames"),
    ("p2", "Floor plane + metric scale", r"^\[2\] ", 5, ""),
    ("p3", "Camera tracking (OpenCV SLAM)", r"^\[3\] ", 60, "frames"),
    ("p4", "Depth filtering", r"^\[4\] ", 40, "frames"),
    ("p5", "3D fusion (TSDF)", r"^\[5\] ", 120, "frames"),
    ("p6", "Gravity + wall alignment", r"^\[6\] ", 20, ""),
    ("p7", "Walls, doors, floor, ceiling", r"^\[7\] ", 10, ""),
    ("p8", "Objects + names (Gemini)", r"^\[8\] ", 40, ""),
    ("p9", "Textured walls / floor / ceiling", r"^\[9\] ", 105, ""),
    ("p10", "2D floor plan + maps", r"^\[10\] ", 20, ""),
    ("p11", "3D exports (GLB)", r"^\[11\] ", 90, ""),
    ("p12", "Previews", r"^\[12\] ", 60, ""),
    ("p13", "Evaluation", r"^\[13\] ", 46, ""),
    ("cov", "Coverage map + keyframe selection", r"coverage over \d+ SLAM frames|SLAM run: ", 6, ""),
    ("ai", "AI defect inspection (Gemini)", r"\d+ AI keyframes", 120, "keyframes"),
    ("render", "Live map video + 3D models", r"findings placed in 3D|rendering live map", 29, ""),
]
_RX = [(k, re.compile(rx)) for k, _, rx, _, _ in PHASES]
LABEL = {k: lab for k, lab, *_ in PHASES}


def scan(lines: list[str], seen: dict[str, float], counts: dict[str, int], now: float | None = None) -> None:
    """Record the first time each phase marker appears. Call repeatedly with the WHOLE log
    while the job runs; counts are recomputed from scratch each call."""
    now = now or time.time()
    counts["kf_done"] = 0
    for line in lines:
        s = line.strip()
        for k, rx in _RX:
            if k not in seen and rx.search(s):
                seen[k] = now
        if m := re.search(r"(\d+) frames \(1 of every", s):
            counts["frames"] = int(m.group(1))
        if m := re.search(r"furniture completion: (\d+) instances", s):
            counts["furniture"] = int(m.group(1))
        if m := re.search(r"(\d+) AI keyframes", s):
            counts["keyframes"] = int(m.group(1))
        if re.search(r"INS-\d+ .*: (done|failed)", s):
            counts["kf_done"] += 1
        if re.search(r"\] done in \d+s", s):
            seen.setdefault("done", now)


def _expected(counts: dict) -> dict[str, float]:
    f = counts.get("frames", counts.get("frames_est", REF_FRAMES)) / REF_FRAMES
    out = {}
    for k, _, _, sec, by in PHASES:
        if by == "frames":
            sec *= f
        elif by == "furniture" and "furniture" in counts:
            sec = 140 * counts["furniture"] + 20
        elif by == "keyframes":
            sec = 38 * counts.get("keyframes", 14)
        out[k] = max(1.0, sec)
    return out


def estimate(seen: dict[str, float], counts: dict, started: float, now: float | None = None) -> dict:
    now = now or time.time()
    elapsed = now - started
    if "done" in seen:
        return {"percent": 100.0, "phase": "Done", "elapsed_s": round(seen["done"] - started), "eta_s": 0}
    exp = _expected(counts)
    keys = [k for k, *_ in PHASES]
    cur_i = max((keys.index(k) for k in seen if k in keys), default=-1)
    if cur_i < 0:   # log not started yet: still loading models
        total = sum(exp.values())
        return {"percent": 0.0, "phase": "Starting", "elapsed_s": round(elapsed), "eta_s": round(total)}
    cur = keys[cur_i]
    done_exp = sum(exp[k] for k in keys[:cur_i])           # skipped phases count as done
    in_cur = now - seen[cur]
    frac = min(0.95, in_cur / exp[cur])
    if cur == "ai" and counts.get("keyframes"):
        frac = max(frac * 0.9, min(0.97, counts.get("kf_done", 0) / counts["keyframes"]))
    total = sum(exp.values())
    percent = 100.0 * (done_exp + frac * exp[cur]) / total
    # speed correction from the phases that actually ran (measured from their markers)
    ran = [k for k in keys[:cur_i] if k in seen]
    actual = (seen[cur] - seen[ran[0]]) if ran else 0.0
    planned = sum(exp[k] for k in keys[keys.index(ran[0]):cur_i]) if ran else 0.0
    speed = min(3.0, max(0.33, actual / planned)) if planned > 120 else 1.0
    remaining = (exp[cur] * (1 - frac) + sum(exp[k] for k in keys[cur_i + 1:])) * speed
    if in_cur > exp[cur] * speed:               # current step overrunning: keep the ETA honest
        remaining += min(in_cur - exp[cur] * speed, exp[cur])
    return {"percent": round(min(99.0, percent), 1), "phase": LABEL[cur],
            "step": cur_i + 1, "steps": len(keys),
            "elapsed_s": round(elapsed), "eta_s": round(remaining)}
