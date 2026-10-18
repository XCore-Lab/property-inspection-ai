"""Dynamic prompt router (spec phase 5): scene -> only the relevant localizer phrases."""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from backend.models.schemas import SceneResult
from backend.services.taxonomy import Taxonomy

LOCATE_TEMPLATE = "Locate all the instances that matches the following description: {classes}."


@dataclass
class RoutedPrompt:
    phrase: str
    component: str
    defect: str


@dataclass
class PromptPlan:
    room: str
    components: list[str]
    phrases: list[RoutedPrompt] = field(default_factory=list)
    batches: list[list[RoutedPrompt]] = field(default_factory=list)

    def lookup(self, label: str) -> RoutedPrompt | None:
        """Map a label returned by the localizer back to (component, defect)."""
        key = label.strip().lower().rstrip(".")
        for p in self.phrases:
            if p.phrase.lower() == key:
                return p
        for p in self.phrases:  # tolerate minor echo differences
            if key and (key in p.phrase.lower() or p.phrase.lower() in key):
                return p
        # the localizer occasionally garbles a label ("countop"): fuzzy match
        close = difflib.get_close_matches(key, [p.phrase.lower() for p in self.phrases], n=1, cutoff=0.6)
        if close:
            return next(p for p in self.phrases if p.phrase.lower() == close[0])
        return None

    def query_texts(self) -> list[str]:
        return [build_locate_query([p.phrase for p in b]) for b in self.batches]


def build_locate_query(phrases: list[str]) -> str:
    return LOCATE_TEMPLATE.format(classes="</c>".join(phrases))


def select_components(scene: SceneResult, tax: Taxonomy, component_hint: str | None = None) -> list[str]:
    tiers = set(tax.router.get("enabled_tiers", ["mvp", "v2"]))
    if component_hint:
        c = tax.normalize_component(component_hint)
        if c:
            return [c]
    comps: list[str] = []
    for sc in scene.components:
        c = tax.normalize_component(sc.name) or tax.normalize_component(sc.raw_name)
        if c and c not in comps:
            comps.append(c)
    if not comps:  # VLM gave nothing usable: fall back to the room's typical components
        comps = tax.components_for_room(scene.room)[:6]
    for c in tax.router.get("always_include", []):
        if c not in comps:
            comps.append(c)
    # Lock/handle components imply their parent (and vice versa is NOT assumed).
    comps = [c for c in comps if c in tax.components and tax.components[c].tier in tiers]
    # Room-typical components first (stable), so a VLM over-listing unusual parts
    # (e.g. "door lock" for a fridge handle in a kitchen) is what gets truncated.
    typical = tax.components_for_room(scene.room)
    comps.sort(key=lambda c: typical.index(c) if c in typical else len(typical))
    return comps[: int(tax.router.get("max_components", 8))]


def build_prompt_plan(scene: SceneResult, tax: Taxonomy, component_hint: str | None = None,
                      max_per_call: int | None = None) -> PromptPlan:
    comps = select_components(scene, tax, component_hint)
    max_per_call = max_per_call or int(tax.router.get("max_classes_per_call", 8))
    seen: set[str] = set()
    phrases: list[RoutedPrompt] = []
    # Round-robin over components so the first batches cover every component's
    # most important defect before secondary phrasings.
    per_comp = []
    for c in comps:
        items = []
        for d, ps in tax.components[c].prompts.items():
            for p in ps:
                items.append(RoutedPrompt(phrase=p.strip(), component=c, defect=d))
        per_comp.append(items)
    i = 0
    while any(i < len(x) for x in per_comp):
        for items in per_comp:
            if i < len(items):
                p = items[i]
                k = p.phrase.lower()
                if k not in seen:          # prompt deduplication
                    seen.add(k)
                    phrases.append(p)
        i += 1
    batches = [phrases[j:j + max_per_call] for j in range(0, len(phrases), max_per_call)]
    return PromptPlan(room=scene.room, components=comps, phrases=phrases, batches=batches)
