"""Result fusion / rule engine (spec phases 17 & 18).

States are decided by explicit rules over (localizer candidate, verifier JSON,
image quality, specialist evidence) - never by model-reported probabilities.
"""
from __future__ import annotations

from backend.models.schemas import (ComponentStatus, Detection, ImageQuality, LocalizerDetection,
                                    ResultState, SpecialistEvidence, VerificationResult)
from backend.services.image_service import iou, nms
from backend.services.taxonomy import Taxonomy

LINEAR_DEFECTS = {"crack", "scratch"}
BLOTCH_DEFECTS = {"stain", "water_damage", "discoloration", "mold"}
LOOKALIKE_WORDS = ("shadow", "reflect", "glare", "grout", "joint", "vein", "grain", "pattern", "edge", "corner",
                   "texture", "wire", "cable", "outlet", "seam", "gap line", "decorat", "lighting", "dirt", "debris")
FUNCTIONAL_WORDS = ("work", "function", "operat", "lock", "unlock", "motor", "fan", "airflow", "noise", "electric")


def merge_candidates(cands: list[LocalizerDetection], iou_thr: float = 0.5) -> list[LocalizerDetection]:
    """Per (component, defect) NMS, then collapse cross-phrase duplicates of the same
    component whose boxes nearly coincide (e.g. 'wall stain' + 'water stain on wall')."""
    out: list[LocalizerDetection] = []
    groups: dict[tuple[str, str], list[LocalizerDetection]] = {}
    for c in cands:
        groups.setdefault((c.component, c.defect), []).append(c)
    for g in groups.values():
        keep = nms([c.box for c in g], iou_thr)
        out += [g[i] for i in keep]
    final: list[LocalizerDetection] = []
    for c in sorted(out, key=lambda c: c.box.area):
        dup = next((f for f in final if f.component == c.component and iou(f.box, c.box) > 0.75), None)
        if dup:
            if c.defect != dup.defect and c.label not in dup.merged_labels:
                dup.merged_labels.append(c.label)
            continue
        final.append(c)
    return final


def flag_repetitions(cands: list[LocalizerDetection], min_count: int = 5, max_cv: float = 1.0) -> None:
    """Grounding models often box every instance of the *component* (each tile, each
    panel) for a defect prompt. Many same-label boxes of comparable size are flagged
    (max_cv=1.0 tolerates perspective: floor tiles near vs. far differ ~5x in area)."""
    import statistics
    groups: dict[str, list[LocalizerDetection]] = {}
    for c in cands:
        groups.setdefault(c.label, []).append(c)
    for g in groups.values():
        if len(g) < min_count:
            continue
        areas = [c.box.area for c in g]
        mu = statistics.mean(areas)
        if mu > 0 and statistics.pstdev(areas) / mu <= max_cv:
            for c in g:
                c.repeated = True


# Review reasons that only express the verifier's own uncertainty; an independent second opinion
# (Gemini) that confirms the finding resolves them. Image-quality / box / pattern reasons stay.
SOFT_REASONS = ("Severity could not be assessed", "Possible look-alike noted by verifier", "Verifier describes")


def decide(v: VerificationResult, cand: LocalizerDetection, img_w: int, img_h: int,
           quality: ImageQuality | None, spec: SpecialistEvidence | None, max_area_frac: float,
           tax: Taxonomy) -> tuple[ResultState, list[str]]:
    reasons: list[str] = []
    if v.defect_present is None:
        return ResultState.UNKNOWN, ["Verifier output could not be interpreted."]
    if v.defect_present is False:
        return ResultState.NOT_DETECTED, [f"Verifier: {v.alternative_explanation}" if v.alternative_explanation
                                          else "Verifier found no visible evidence."]

    area_frac = cand.box.area / max(1.0, img_w * img_h)
    if area_frac > max_area_frac:
        reasons.append("Region covers most of the image; defect location is not specific.")
    if v.severity == "unknown":
        reasons.append("Severity could not be assessed visually.")
    if is_lookalike(v.alternative_explanation):
        reasons.append(f"Possible look-alike noted by verifier: {v.alternative_explanation}.")
    if v.defect_type and v.defect_type != cand.defect and not (
            {v.defect_type, cand.defect} <= BLOTCH_DEFECTS):
        reasons.append(f"Verifier describes '{tax.defect_label(v.defect_type)}' rather than the localized "
                       f"'{tax.defect_label(cand.defect)}'.")
    if quality is not None and not quality.ok:
        reasons.append("Image quality limits confidence: " + " ".join(quality.warnings))
    if spec is not None:
        if cand.defect in LINEAR_DEFECTS and spec.regularity > 0.75 and spec.linear_score < 0.15:
            reasons.append("Line evidence is grid-like (could be joints, grout or pattern).")
        if cand.defect in LINEAR_DEFECTS and spec.linear_score < 0.02 and spec.blotch_score < 0.05:
            reasons.append("No thin linear structure measured in the region.")
    if cand.repeated:
        reasons.append("Localizer matched many similar regions (a repeated pattern such as individual tiles); "
                       "confirm this specific one.")
    if min(cand.box.w, cand.box.h) < 12 or area_frac < 0.001:
        reasons.append("Very small region; confirm on site or with a close-up photo.")
    return (ResultState.NEEDS_REVIEW if reasons else ResultState.DETECTED), reasons


def is_lookalike(text: str) -> bool:
    """A short verifier note naming a known look-alike (shadow, grout line, ...)."""
    t = (text or "").strip().lower()
    return bool(t) and len(t.split()) <= 8 and any(w in t for w in LOOKALIKE_WORDS)


def sanitize_description(text: str) -> str:
    """Strip claims the system must never make from a photo (structural safety,
    working/not-working functionality)."""
    import re
    t = (text or "").strip()
    t = re.sub(r"\b(structurally (unsound|sound|safe|unsafe)|structural (failure|integrity is compromised))\b",
               "visible surface condition", t, flags=re.I)
    sents = re.split(r"(?<=[.!?])\s+", t)
    keep = [s for s in sents if not re.search(r"\b(is|are|appears?|seems?) (to be )?(fully )?(working|functional|"
                                             r"operational|not working|non[- ]functional|broken down)\b", s, re.I)]
    return " ".join(keep).strip()


def component_statuses(components: list[str], detections: list[Detection], tax: Taxonomy) -> list[ComponentStatus]:
    out = []
    for c in components:
        cdef = tax.components.get(c)
        mine = [d for d in detections if d.component == c]
        if any(d.state == ResultState.DETECTED for d in mine):
            st = ResultState.DETECTED
            summ = "; ".join(sorted({d.defect.label for d in mine if d.state == ResultState.DETECTED}))
        elif any(d.state in (ResultState.NEEDS_REVIEW, ResultState.UNKNOWN) for d in mine):
            st = ResultState.NEEDS_REVIEW
            summ = "Possible: " + "; ".join(sorted({d.defect.label for d in mine}))
        else:
            st = ResultState.NOT_DETECTED
            summ = "No clear visible damage identified."
        out.append(ComponentStatus(component=c, label=tax.component_label(c), visible_state=st, visible_summary=summ,
                                   manual_checks=list(cdef.manual_checks) if cdef else [],
                                   issue_ids=[d.issue_id for d in mine]))
    return out


def manual_check_items(statuses: list[ComponentStatus], detections: list[Detection]) -> list[dict]:
    items = []
    for s in statuses:
        for chk in s.manual_checks:
            items.append({"component": s.component, "component_label": s.label, "check": chk,
                          "state": ResultState.MANUAL_CHECK_REQUIRED.value,
                          "reason": "Function cannot be verified from a photograph."})
    for d in detections:
        for chk in d.professional_review:
            items.append({"component": d.component, "component_label": d.component_label, "check": chk,
                          "state": ResultState.MANUAL_CHECK_REQUIRED.value, "issue_id": d.issue_id,
                          "reason": f"Visible {d.defect.label.lower()} - significance requires professional assessment."})
    return items


def functional_claim(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in FUNCTIONAL_WORDS)
