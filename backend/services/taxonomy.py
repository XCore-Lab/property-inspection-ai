"""Loads defect_classes.yaml and normalizes free-text room/component/defect names."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _singular(s: str) -> str:
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in s.split())


@dataclass
class ComponentDef:
    key: str
    label: str
    aliases: list[str]
    tier: str
    prompts: dict[str, list[str]]
    manual_checks: list[str] = field(default_factory=list)


@dataclass
class Taxonomy:
    rooms: dict[str, dict]
    room_components: dict[str, list[str]]
    defects: dict[str, dict]
    components: dict[str, ComponentDef]
    professional_review: dict[str, dict]
    hard_negatives: dict[str, list[str]]
    router: dict

    # ------------------------------------------------------------ lookups
    def room_label(self, room: str) -> str:
        return self.rooms.get(room, {}).get("label", room.replace("_", " ").title())

    def component_label(self, comp: str) -> str:
        c = self.components.get(comp)
        return c.label if c else comp.replace("_", " ").title()

    def defect_label(self, defect: str) -> str:
        return self.defects.get(defect, {}).get("label", defect.replace("_", " ").capitalize())

    def is_structural(self, defect: str) -> bool:
        return bool(self.defects.get(defect, {}).get("structural"))

    def normalize_room(self, text: str) -> str:
        t = _norm(text)
        if not t:
            return "unknown"
        if t.replace(" ", "_") in self.rooms:
            return t.replace(" ", "_")
        best, best_len = "unknown", 0
        for key, r in self.rooms.items():
            for a in r.get("aliases", []):
                a = _norm(a)
                if a and (t == a or re.search(rf"\b{re.escape(a)}\b", t)) and len(a) > best_len:
                    best, best_len = key, len(a)
        return best

    def normalize_component(self, text: str) -> str | None:
        t = _norm(text)
        if not t:
            return None
        if t.replace(" ", "_") in self.components:
            return t.replace(" ", "_")
        ts = _singular(t)
        best, best_len = None, 0
        for key, c in self.components.items():
            for a in [key.replace("_", " ")] + c.aliases:
                a = _norm(a)
                for cand in (t, ts):
                    if a and (cand == a or re.search(rf"\b{re.escape(a)}\b", cand)) and len(a) > best_len:
                        best, best_len = key, len(a)
        return best

    def normalize_defect(self, text: str, component: str | None = None) -> str:
        t = _norm(text)
        if not t:
            return ""
        if t.replace(" ", "_") in self.defects:
            return t.replace(" ", "_")
        # Phrase lookup from the prompts table first (exact localizer labels).
        comps = [self.components[component]] if component in self.components else list(self.components.values())
        for c in comps:
            for d, phrases in c.prompts.items():
                if any(_norm(p) == t for p in phrases):
                    return d
        rules = [
            (r"grease", "grease_buildup"), (r"rust|corro", "rust"), (r"mou?ld|mildew|fung", "mold"),
            (r"peel|flak|blister", "peeling_paint"), (r"water|leak|damp|moist", "water_damage"),
            (r"glass|pane|shatter", "broken_glass"), (r"burn|scorch", "burn_mark"),
            (r"missing grout|grout missing|gap", "missing_grout"), (r"crack|fracture|fissure", "crack"),
            (r"chip", "chip"), (r"scratch|scuff|abrasion", "scratch"), (r"hole|puncture|dent", "hole"),
            (r"tear|torn|rip|fray", "tear"), (r"worn|wear|thread ?bare", "worn"),
            (r"stain|spot|mark|dirt|soil", "stain"), (r"discolo|fade|yellow", "discoloration"),
            (r"misalign|sag|gap|crooked", "misalignment"), (r"detach|loose|separat|lifting", "detached"),
            (r"missing|absent", "missing"), (r"broken|damag|break", "broken"),
            (r"deform|bulg|bow|warp", "deformation"),
        ]
        for pat, d in rules:
            if re.search(pat, t):
                if d == "broken" and component in ("wall", "ceiling", "floor", "floor_tile", "rug", "countertop", "door", "molding"):
                    return "surface_damage"
                return d
        return "surface_damage"

    def components_for_room(self, room: str) -> list[str]:
        return list(self.room_components.get(room, self.room_components.get("unknown", [])))

    def hard_negatives_for(self, defect: str) -> list[str]:
        return self.hard_negatives.get(defect, self.hard_negatives.get("default", []))

    def professional_checks(self, component: str, defect: str) -> list[str]:
        out: list[str] = []
        for rule in self.professional_review.values():
            if defect in rule.get("triggers", []) and (not rule.get("components") or component in rule["components"]):
                out += [c for c in rule.get("checks", []) if c not in out]
        return out


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> Taxonomy:  # mtime busts the cache when the yaml changes
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    comps = {
        k: ComponentDef(key=k, label=v.get("label", k), aliases=v.get("aliases", []), tier=v.get("tier", "mvp"),
                        prompts={d: list(p) for d, p in (v.get("prompts") or {}).items()},
                        manual_checks=list(v.get("manual_checks", [])))
        for k, v in raw["components"].items()
    }
    tax = Taxonomy(rooms=raw["rooms"], room_components=raw["room_components"], defects=raw["defects"],
                   components=comps, professional_review=raw.get("professional_review", {}),
                   hard_negatives=raw.get("hard_negatives", {}), router=raw.get("router", {}))
    _validate(tax)
    return tax


def _validate(t: Taxonomy) -> None:
    for room, comps in t.room_components.items():
        assert room in t.rooms, f"room_components: unknown room {room}"
        for c in comps:
            assert c in t.components, f"room_components[{room}]: unknown component {c}"
    for c in t.components.values():
        for d in c.prompts:
            assert d in t.defects, f"component {c.key}: unknown defect {d}"


def get_taxonomy(path: str | None = None) -> Taxonomy:
    from backend.config.settings import get_settings
    p = path or get_settings().taxonomy_path
    return _load(p, Path(p).stat().st_mtime)
