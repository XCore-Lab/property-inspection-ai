"""Risk & valuation engine for insurers, mortgage lenders and real-estate valuation.

Turns rule-engine findings into: condition score + grade, risk level, per-category exposure
(insurance perils), indicative repair-cost range, valuation / collateral impact, premium-loading
suggestion and an underwriting recommendation with explicit reasons.

Deterministic and auditable: every number comes from `config/risk_rules.yaml`, never from model
probabilities, and every recommendation lists the rules that fired (compliance / explainability).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

import yaml

from backend.config.settings import Settings


@lru_cache(maxsize=4)
def load_rules(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _state(d: Any) -> str:
    st = d.get("state") if isinstance(d, dict) else getattr(d, "state", "")
    return getattr(st, "value", st) or "UNKNOWN"


def _get(d: Any, *path, default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None)
    return default if cur is None else cur


def assess_findings(findings: Iterable[Any], s: Settings, *, integrity_flags: list[dict] | None = None,
                    coverage_pct: float | None = None, property_value: float | None = None,
                    n_images: int = 1) -> dict:
    """findings: Detection objects or dicts with state, component, defect{type,severity}, issue_id."""
    R = load_rules(s.risk_rules_path)
    sw, sf = R["state_weight"], R["severity_factor"]
    cats = {k: {"label": v["label"], "peril": v["peril"], "points": 0.0, "findings": 0,
                "cost_low": 0.0, "cost_high": 0.0} for k, v in R["categories"].items()}
    items, deduction, cost_lo, cost_hi = [], 0.0, 0.0, 0.0
    for d in findings:
        state = _state(d)
        w = float(sw.get(state, 0.0))
        if w <= 0:
            continue
        dtype = _get(d, "defect", "type", default="default")
        sev = (_get(d, "defect", "severity", default="unknown") or "unknown").lower()
        comp = _get(d, "component", default="")
        rule = R["defects"].get(dtype, R["defects"]["default"])
        cat = R.get("component_category", {}).get(comp, rule["category"])
        f = float(sf.get(sev, sf["unknown"]))
        pts = rule["points"] * f * w
        lo, hi = (c * f * s.cost_multiplier for c in rule["cost"])
        lo, hi = lo * w, hi * w
        deduction += pts
        cost_lo += lo
        cost_hi += hi
        c = cats[cat]
        c["points"] += pts
        c["findings"] += 1
        c["cost_low"] += lo
        c["cost_high"] += hi
        items.append({"issue_id": _get(d, "issue_id", default=""), "component": comp, "defect": dtype,
                      "severity": sev, "state": state, "category": cat, "risk_points": round(pts, 1),
                      "repair_cost": [round(lo), round(hi)]})

    # Multi-image / walkthrough: normalise so a whole house is not penalised for having more photos.
    norm = max(1.0, n_images ** 0.5)
    score = max(0.0, min(100.0, 100.0 - deduction / norm))
    grade = next(g for g in R["grades"] if score >= g["min"])

    severe_collateral = any(i["category"] in ("structural", "water", "fire") and i["state"] == "DETECTED"
                            and i["severity"] == "severe" for i in items)
    level = ("Low" if score >= 85 else "Moderate" if score >= 70 else "High" if score >= 50 else "Critical")
    if severe_collateral and level in ("Low", "Moderate"):
        level = "High"

    reasons: list[str] = []
    for k, c in cats.items():
        if c["findings"]:
            reasons.append(f"{c['label']}: {c['findings']} finding(s), {c['points']:.1f} risk points")
    if severe_collateral:
        reasons.append("Severe structural / water / fire finding confirmed (collateral risk)")

    flags = list(integrity_flags or [])
    high_flags = [f for f in flags if f.get("level") == "high"]
    evidence_ok = True
    if coverage_pct is not None and coverage_pct < R.get("min_coverage_pct", 60):
        evidence_ok = False
        reasons.append(f"Evidence incomplete: only {coverage_pct:.0f}% of surfaces AI-inspected "
                       f"(min {R.get('min_coverage_pct', 60)}%)")
    if high_flags:
        reasons.append("Evidence integrity flag: " + "; ".join(f["message"] for f in high_flags))

    if high_flags:
        decision = "REFER - evidence integrity"
    elif level == "Critical" or severe_collateral:
        decision = "REFER - professional survey required"
    elif not evidence_ok:
        decision = "REFER - additional evidence required"
    elif level == "High":
        decision = "PROCEED WITH CONDITIONS - repairs / retention"
    elif level == "Moderate":
        decision = "PROCEED WITH CONDITIONS - note defects"
    else:
        decision = "PROCEED"

    vpct = R["valuation_impact_pct"][grade["grade"]]
    valuation = {"impact_pct": vpct, "basis": f"Grade {grade['grade']} ({grade['label']})"}
    if property_value:
        valuation["impact_amount"] = [round(property_value * vpct[0] / 100), round(property_value * vpct[1] / 100)]
        valuation["adjusted_value"] = [round(property_value + property_value * vpct[1] / 100 - cost_hi),
                                       round(property_value + property_value * vpct[0] / 100 - cost_lo)]

    perils = [{"category": k, "label": c["label"], "peril": c["peril"], "findings": c["findings"],
               "risk_points": round(c["points"], 1), "repair_cost": [round(c["cost_low"]), round(c["cost_high"])]}
              for k, c in cats.items() if c["findings"]]
    perils.sort(key=lambda p: -p["risk_points"])
    return {
        "condition_score": round(score, 1),
        "grade": grade["grade"], "grade_label": grade["label"],
        "risk_level": level,
        "decision": decision,
        "reasons": reasons,
        "repair_cost": {"low": round(cost_lo), "high": round(cost_hi), "currency": s.currency},
        "valuation": valuation,
        "insurance": {"perils": perils, "premium_loading_pct": R["premium_loading_pct"][level]},
        "mortgage": {"collateral_risk": "High" if severe_collateral or level == "Critical"
                     else ("Medium" if level == "High" else "Low"),
                     "repair_retention": round(cost_hi * 1.2) if level in ("High", "Critical") else 0},
        "evidence": {"integrity_flags": flags, "coverage_pct": coverage_pct, "complete": evidence_ok},
        "findings": items,
        "rules_version": R.get("version", 1),
        "disclaimer": "Decision support only - indicative figures from configurable rules; a qualified "
                      "surveyor / underwriter makes the final decision.",
    }


def assess_inspection(ins, s: Settings) -> dict:
    flags = (ins.integrity or {}).get("flags", [])
    pv = None
    try:
        pv = float(ins.options.get("property_value") or 0) or None
    except (TypeError, ValueError):
        pv = None
    return assess_findings(ins.detections, s, integrity_flags=flags, property_value=pv)
