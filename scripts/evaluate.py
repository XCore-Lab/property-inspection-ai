"""Phase 11 evaluation: per-defect precision / recall / F1, IoU, FP/FN rates, runtime.

Ground truth: a JSONL file of DatasetRecord rows (spec phase 9 schema):
    {"image": "IMG_1.jpg", "property": "P001", "room": "bathroom", "component": "wall_tile",
     "defects": [{"type": "crack", "component": "wall_tile", "bbox": [x1, y1, x2, y2]}], ...}
Images with "defects": [] are negatives. A GT defect without a bbox ([0,0,0,0]) is
scored at image level only.

Predictions: a directory of inspections (data/outputs) matched by source_image.
Three evaluations (spec: evaluate localizer / verifier / full pipeline separately):
  * localizer  - every localizer candidate (verified or rejected) counts as a prediction
  * verifier   - accuracy of the verifier's yes/no on candidates that overlap GT vs. not
  * pipeline   - DETECTED + NEEDS_REVIEW detections (and DETECTED-only, strict)

Matching: same defect *category* (see CATEGORIES) and IoU >= --iou (default 0.3;
defect extents are fuzzy) or, for image-level GT, same category anywhere.

    python scripts/evaluate.py --gt data/pilot/labels.jsonl --pred data/outputs --out data/eval
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.models.schemas import BBox, DatasetRecord, Inspection  # noqa: E402
from backend.services.image_service import iou  # noqa: E402

# Spec evaluation-table rows: (component set, defect set)
CATEGORIES = {
    "Wall crack": ({"wall"}, {"crack"}),
    "Ceiling crack": ({"ceiling"}, {"crack"}),
    "Wall/ceiling stain & water damage": ({"wall", "ceiling"}, {"stain", "water_damage", "discoloration"}),
    "Peeling paint": ({"wall", "ceiling"}, {"peeling_paint"}),
    "Hole / surface damage (wall, ceiling)": ({"wall", "ceiling"}, {"hole", "surface_damage"}),
    "Mold": (None, {"mold"}),
    "Tile crack": ({"floor_tile", "wall_tile"}, {"crack", "chip", "missing"}),
    "Grout damage": ({"grout"}, {"crack", "missing_grout", "discoloration", "mold"}),
    "Countertop scratch": ({"countertop"}, {"scratch"}),
    "Countertop stain": ({"countertop"}, {"stain", "discoloration", "burn_mark"}),
    "Rug stain": ({"rug"}, {"stain", "discoloration"}),
    "Rug damage": ({"rug"}, {"tear", "worn", "burn_mark"}),
    "Door damage": ({"door"}, None),
    "Window damage": ({"window"}, None),
    "Lock / hardware": ({"door_lock", "cupboard_lock"}, None),
    "Exhaust": ({"exhaust_hood"}, None),
    "Molding": ({"molding"}, None),
}


def category(component: str | None, defect: str) -> str | None:
    for name, (comps, defs) in CATEGORIES.items():
        if (comps is None or component in comps) and (defs is None or defect in defs):
            return name
    return None


def has_box(b: list[float]) -> bool:
    return len(b) == 4 and (b[2] - b[0]) > 0 and (b[3] - b[1]) > 0


def match(preds: list[tuple[str, BBox]], gts: list[tuple[str, BBox | None]], thr: float):
    """Greedy one-to-one matching within the same category. Returns (tp, fp, fn, ious, fp_items, fn_items)."""
    used = set()
    tp, ious, fp_items = 0, [], []
    for cat, pb in preds:
        best, bi = -1.0, None
        for j, (gc, gb) in enumerate(gts):
            if j in used or gc != cat:
                continue
            v = iou(pb, gb) if gb is not None else 1.0
            if v > best:
                best, bi = v, j
        if bi is not None and (gts[bi][1] is None or best >= thr):
            used.add(bi)
            tp += 1
            if gts[bi][1] is not None:
                ious.append(best)
        else:
            fp_items.append(cat)
    fn_items = [gts[j][0] for j in range(len(gts)) if j not in used]
    return tp, len(fp_items), len(fn_items), ious, fp_items, fn_items


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", default=str(ROOT / "data" / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--iou", type=float, default=0.3)
    a = ap.parse_args()

    gt = [DatasetRecord.model_validate_json(l) for l in Path(a.gt).read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict[str, Inspection] = {}
    for p in sorted(Path(a.pred).glob("INS-*/inspection.json")):
        ins = Inspection.model_validate_json(p.read_text(encoding="utf-8"))
        if ins.status == "done":
            latest[ins.source_image] = ins  # later inspections of the same image win

    out = Path(a.out)
    for sub in ("false_positives", "false_negatives"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True, exist_ok=True)

    modes = {"localizer": defaultdict(lambda: [0, 0, 0, []]), "pipeline": defaultdict(lambda: [0, 0, 0, []]),
             "pipeline_strict": defaultdict(lambda: [0, 0, 0, []])}
    verifier = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    images_per_cat: dict[str, set] = defaultdict(set)
    runtimes, missing, failures = [], [], []

    for rec in gt:
        ins = latest.get(rec.image)
        if ins is None:
            missing.append(rec.image)
            continue
        runtimes.append(ins.timings.total_s)
        gts = []
        for d in rec.defects:
            c = category(d.component or rec.component, d.type)
            if c:
                gts.append((c, BBox(x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3]) if has_box(d.bbox) else None))
                images_per_cat[c].add(rec.image)
        allc = ins.detections + ins.rejected
        sets = {
            "localizer": [(category(d.component, d.defect.type), d.localization) for d in allc],
            "pipeline": [(category(d.component, d.defect.type), d.localization) for d in ins.detections],
            "pipeline_strict": [(category(d.component, d.defect.type), d.localization) for d in ins.detections
                                if d.state.value == "DETECTED"],
        }
        img_level = {c for c, b in gts if b is None}
        for mode, preds in sets.items():
            preds = [(c, b) for c, b in preds if c]
            # image-level GT: several boxes of that category in one image count once
            seen_cat: set = set()
            preds = [(c, b) for c, b in preds if not (c in img_level and (c in seen_cat or seen_cat.add(c)))]
            for c, _ in preds:
                images_per_cat[c].add(rec.image)
            tp, fp, fn, ious, fpi, fni = match(preds, gts, a.iou)
            for cat in {c for c, _ in preds} | {c for c, _ in gts}:
                ctp, cfp, cfn, cious, *_ = match([p for p in preds if p[0] == cat], [g for g in gts if g[0] == cat], a.iou)
                m = modes[mode][cat]
                m[0] += ctp; m[1] += cfp; m[2] += cfn; m[3] += cious  # noqa: E702
            if mode == "pipeline" and (fpi or fni):
                failures.append({"image": rec.image, "inspection_id": ins.inspection_id, "false_positive": fpi, "missed": fni})
                d = Path(a.pred) / ins.inspection_id / "annotated_debug.jpg"
                for kind, items in (("false_positives", fpi), ("false_negatives", fni)):
                    if items and d.exists():
                        shutil.copy(d, out / kind / f"{Path(rec.image).stem}__{ins.inspection_id}.jpg")
        # verifier: did it keep candidates that match GT and reject those that don't?
        for det in allc:
            c = category(det.component, det.defect.type)
            hit = any(gc == c and (gb is None or iou(det.localization, gb) >= a.iou) for gc, gb in gts)
            kept = det.state.value != "NOT_DETECTED"
            verifier[("tp" if kept else "fn") if hit else ("fp" if kept else "tn")] += 1

    def table(mode: str) -> list[str]:
        L = [f"### {mode}", "", "| Defect | Images | TP | FP | FN | Precision | Recall | F1 | mean IoU |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        tot = [0, 0, 0, []]
        for cat in list(CATEGORIES):
            if cat not in modes[mode]:
                continue
            tp, fp, fn, ious = modes[mode][cat]
            tot[0] += tp; tot[1] += fp; tot[2] += fn; tot[3] += ious  # noqa: E702
            p, r, f = prf(tp, fp, fn)
            L.append(f"| {cat} | {len(images_per_cat[cat])} | {tp} | {fp} | {fn} | {p:.2f} | {r:.2f} | {f:.2f} | "
                     f"{(sum(ious) / len(ious)) if ious else float('nan'):.2f} |")
        p, r, f = prf(*tot[:3])
        L.append(f"| **All (micro)** | {len(gt)} | {tot[0]} | {tot[1]} | {tot[2]} | {p:.2f} | {r:.2f} | {f:.2f} | "
                 f"{(sum(tot[3]) / len(tot[3])) if tot[3] else float('nan'):.2f} |")
        return L + [""]

    v = verifier
    vp, vr, vf = prf(v["tp"], v["fp"], v["fn"])
    neg = [r for r in gt if not r.defects]
    neg_fp = sum(1 for r in neg if r.image in latest and latest[r.image].detections)
    L = ["# Evaluation report", "",
         f"- Ground-truth images: {len(gt)} ({len(neg)} negatives); evaluated: {len(gt) - len(missing)}; missing predictions: {len(missing)}",
         f"- IoU threshold: {a.iou} (image-level match when GT has no box)",
         f"- Runtime per image: mean {sum(runtimes) / max(1, len(runtimes)):.1f}s, max {max(runtimes, default=0):.1f}s",
         f"- Negative images with any reported issue (false-positive image rate): {neg_fp}/{len(neg)}", "",
         "Precision/recall are NOT calibrated confidences; categories with few images are anecdotal.", ""]
    for mode in ("localizer", "pipeline", "pipeline_strict"):
        L += table(mode)
    L += ["### verifier (candidate keep/reject)", "",
          f"kept-correct {v['tp']}, kept-wrong {v['fp']}, rejected-correct {v['tn']}, rejected-wrong {v['fn']} "
          f"-> precision {vp:.2f}, recall {vr:.2f}, F1 {vf:.2f}; "
          f"false-positive reduction {v['tn'] / max(1, v['tn'] + v['fp']):.0%} of wrong candidates removed", ""]
    if failures:
        L += ["### Failure cases (pipeline)", "", "| Image | Inspection | False positives | Missed |", "|---|---|---|---|"]
        L += [f"| {f['image']} | {f['inspection_id']} | {', '.join(f['false_positive']) or '-'} | {', '.join(f['missed']) or '-'} |"
              for f in failures]
    (out / "evaluation.md").write_text("\n".join(L), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps({"failures": failures, "missing": missing, "verifier": v}, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
