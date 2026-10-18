"""Unit tests (spec phase 22): prompts, boxes, crops, JSON, normalization, manual routing."""
from PIL import Image

from backend.models.schemas import (BBox, ImageQuality, LocalizerDetection, ResultState, SceneComponent, SceneResult,
                                    SpecialistEvidence, VerificationResult)
from backend.services import fusion_service as fusion
from backend.services import image_service as imgs
from backend.services.json_utils import extract_json
from backend.services.prompt_service import build_locate_query, build_prompt_plan
from backend.services.vlm_service import VisionService, MockVLM


# ---------------------------------------------------------------- taxonomy / router

def test_taxonomy_normalization(tax):
    assert tax.normalize_room("Washroom") == "bathroom"
    assert tax.normalize_room("a small modern kitchen") == "kitchen"
    assert tax.normalize_room("spaceship") == "unknown"
    assert tax.normalize_component("Kitchen Cabinets") == "cupboard"
    assert tax.normalize_component("range hood") == "exhaust_hood"
    assert tax.normalize_component("wall tiles") == "wall_tile"
    assert tax.normalize_component("skirting board") == "molding"
    assert tax.normalize_defect("hairline crack") == "crack"
    assert tax.normalize_defect("countertop scratch", "countertop") == "scratch"
    assert tax.normalize_defect("mildew patch") == "mold"


def test_no_room_specific_defect_classes(tax):
    for d in tax.defects:
        for r in tax.rooms:
            assert not d.startswith(r), f"room-specific defect class {d}"


def test_prompt_router_only_relevant(tax):
    scene = SceneResult(room="kitchen", components=[SceneComponent(name="countertop"), SceneComponent(name="exhaust_hood")])
    plan = build_prompt_plan(scene, tax)
    assert plan.components == ["countertop", "exhaust_hood"]
    comps = {p.component for p in plan.phrases}
    assert comps == {"countertop", "exhaust_hood"}
    assert not any("rug" in p.phrase for p in plan.phrases)
    # dedup + batching
    phrases = [p.phrase.lower() for p in plan.phrases]
    assert len(phrases) == len(set(phrases))
    assert all(len(b) <= tax.router["max_classes_per_call"] for b in plan.batches)
    # round robin: first batch covers both components
    assert {p.component for p in plan.batches[0]} == {"countertop", "exhaust_hood"}


def test_prompt_router_fallback_and_hint(tax):
    plan = build_prompt_plan(SceneResult(room="bathroom", components=[]), tax)
    assert plan.components and set(plan.components) <= set(tax.room_components["bathroom"])
    plan = build_prompt_plan(SceneResult(room="bathroom", components=[SceneComponent(name="wall")]), tax, "grout")
    assert plan.components == ["grout"]


def test_all_rooms_route(tax):
    for room in tax.rooms:
        plan = build_prompt_plan(SceneResult(room=room), tax)
        assert plan.phrases, room


def test_locate_query_format():
    q = build_locate_query(["wall crack", "water stain"])
    assert q == "Locate all the instances that matches the following description: wall crack</c>water stain."


def test_plan_lookup(tax):
    plan = build_prompt_plan(SceneResult(room="kitchen", components=[SceneComponent(name="countertop")]), tax)
    rp = plan.lookup("Countertop Scratch")
    assert rp and rp.component == "countertop" and rp.defect == "scratch"


# ---------------------------------------------------------------- boxes / crops / tiles

def test_box_clip_and_validate():
    b = imgs.clip_box(BBox(x1=-10, y1=50, x2=900, y2=20), 800, 600)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 20, 800, 50)
    assert imgs.valid_box(b, 800, 600)
    assert not imgs.valid_box(BBox(x1=10, y1=10, x2=12, y2=40), 800, 600)
    assert not imgs.valid_box(BBox(x1=float("nan"), y1=10, x2=12, y2=40), 800, 600)


def test_padded_crop_stays_inside_and_min_size():
    cb = imgs.padded_crop_box(BBox(x1=0, y1=0, x2=10, y2=10), 800, 600, pad=0.15, min_side=96)
    assert cb.x1 >= 0 and cb.y1 >= 0 and cb.x2 <= 800 and cb.y2 <= 600
    assert cb.w >= 96 and cb.h >= 96
    cb = imgs.padded_crop_box(BBox(x1=100, y1=100, x2=300, y2=200), 800, 600, pad=0.1)
    assert (cb.x1, cb.y1, cb.x2, cb.y2) == (80, 90, 320, 210)


def test_crop_from_original():
    im = Image.new("RGB", (400, 300))
    c = imgs.crop(im, BBox(x1=10, y1=20, x2=110, y2=70))
    assert c.size == (100, 50)


def test_tiles_cover_image():
    im = Image.new("RGB", (2500, 1500))
    tiles = imgs.make_tiles(im, 1024, 0.2)
    assert tiles[0].x0 == 0 and tiles[0].y0 == 0
    assert max(t.x0 + t.image.size[0] for t in tiles) == 2500
    assert max(t.y0 + t.image.size[1] for t in tiles) == 1500


def test_nms_and_merge():
    a, b, c = BBox(x1=0, y1=0, x2=100, y2=100), BBox(x1=5, y1=5, x2=100, y2=100), BBox(x1=300, y1=300, x2=350, y2=350)
    keep = imgs.nms([a, b, c], 0.5)
    assert len(keep) == 2
    cands = [LocalizerDetection(label="wall stain", box=a, component="wall", defect="stain"),
             LocalizerDetection(label="water stain on wall", box=b, component="wall", defect="water_damage"),
             LocalizerDetection(label="wall crack", box=c, component="wall", defect="crack")]
    merged = fusion.merge_candidates(cands)
    assert len(merged) == 2
    assert any(m.merged_labels for m in merged)


# ---------------------------------------------------------------- JSON handling

def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1, "b": true}\n```') == {"a": 1, "b": True}
    assert extract_json("Sure! {'defect_present': True, 'severity': 'minor',}") == {"defect_present": True, "severity": "minor"}
    assert extract_json('{"a": {"b": [1, 2]}, "c": "x"} trailing') == {"a": {"b": [1, 2]}, "c": "x"}
    assert extract_json('{"component": "wall", "description": "a long trunc') == {"component": "wall", "description": "a long trunc"}
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_verification_normalization(tax, settings):
    q = VisionService(MockVLM(), tax, settings)
    v = q.parse_verification({"defect_present": "yes", "defect_type": "hairline crack", "severity": "Big"}, "wall", "crack")
    assert v.defect_present is True and v.defect_type == "crack" and v.severity == "unknown"
    v = q.parse_verification(None, "wall", "crack")
    assert v.defect_present is None and not v.valid_json


def test_scene_parse(tax, settings):
    q = VisionService(MockVLM(), tax, settings)
    s = q.parse_scene({"room": "Washroom", "components": ["Wall Tiles", {"name": "grout lines"}, "toilet", "wall tiles"]})
    assert s.room == "bathroom"
    assert [c.name for c in s.components] == ["wall_tile", "grout"]
    assert q.parse_scene(None).source == "fallback"


# ---------------------------------------------------------------- rule engine / manual routing

def _cand(defect="crack", comp="wall", box=None):
    return LocalizerDetection(label="x", box=box or BBox(x1=100, y1=100, x2=300, y2=250), component=comp, defect=defect)


GOOD_Q = ImageQuality(width=800, height=600, blur_score=200, brightness=120, contrast=40, ok=True)


def test_decide_states(tax):
    v = VerificationResult(defect_present=True, defect_type="crack", severity="minor", description="thin line")
    st, r = fusion.decide(v, _cand(), 800, 600, GOOD_Q, None, 0.85, tax)
    assert st == ResultState.DETECTED and not r
    st, _ = fusion.decide(v.model_copy(update={"severity": "unknown"}), _cand(), 800, 600, GOOD_Q, None, 0.85, tax)
    assert st == ResultState.NEEDS_REVIEW
    st, _ = fusion.decide(v.model_copy(update={"defect_present": False}), _cand(), 800, 600, GOOD_Q, None, 0.85, tax)
    assert st == ResultState.NOT_DETECTED
    st, _ = fusion.decide(v.model_copy(update={"defect_present": None}), _cand(), 800, 600, GOOD_Q, None, 0.85, tax)
    assert st == ResultState.UNKNOWN
    big = _cand(box=BBox(x1=0, y1=0, x2=800, y2=600))
    assert fusion.decide(v, big, 800, 600, GOOD_Q, None, 0.85, tax)[0] == ResultState.NEEDS_REVIEW
    grid = SpecialistEvidence(linear_score=0.05, regularity=0.9)
    assert fusion.decide(v, _cand(), 800, 600, GOOD_Q, grid, 0.85, tax)[0] == ResultState.NEEDS_REVIEW
    bad_q = GOOD_Q.model_copy(update={"ok": False, "warnings": ["blurry"]})
    assert fusion.decide(v, _cand(), 800, 600, bad_q, None, 0.85, tax)[0] == ResultState.NEEDS_REVIEW


def test_manual_checks_for_functional_components(tax):
    st = fusion.component_statuses(["exhaust_hood", "door_lock", "wall"], [], tax)
    items = fusion.manual_check_items(st, [])
    checks = {(i["component"], i["check"]) for i in items}
    assert ("exhaust_hood", "Motor function") in checks
    assert ("door_lock", "Deadbolt operation") in checks
    assert not any(c == "wall" for c, _ in checks)
    assert all(i["state"] == "MANUAL_CHECK_REQUIRED" for i in items)
    assert st[0].visible_state == ResultState.NOT_DETECTED


def test_structural_professional_review(tax):
    assert "Structural significance" in tax.professional_checks("wall", "crack")
    assert tax.professional_checks("countertop", "scratch") == []


def test_sanitize_description():
    s = fusion.sanitize_description("Visible crack on wall. The wall is structurally unsound. The fan appears to be working.")
    assert "structurally unsound" not in s and "working" not in s and "Visible crack" in s


def test_quality_flags():
    q = imgs.assess_quality(Image.new("RGB", (400, 300), (10, 10, 10)))
    assert not q.ok and any("dark" in w for w in q.warnings)


def test_repetition_flag(tax):
    cands = [_cand(box=BBox(x1=i * 50, y1=0, x2=i * 50 + 40, y2=40)) for i in range(6)]
    for c in cands:
        c.label = "cracked floor tile"
    fusion.flag_repetitions(cands)
    assert all(c.repeated for c in cands)
    v = VerificationResult(defect_present=True, defect_type="crack", severity="minor")
    assert fusion.decide(v, cands[0], 800, 600, GOOD_Q, None, 0.85, tax)[0] == ResultState.NEEDS_REVIEW
    few = cands[:3]
    for c in few:
        c.repeated = False
    fusion.flag_repetitions(few)
    assert not any(c.repeated for c in few)


def test_carpet_and_furniture_not_inspected(tax):
    scene = SceneResult(room="bedroom", components=[SceneComponent(name="rug"), SceneComponent(name="wardrobe"),
                                                    SceneComponent(name="wall")])
    plan = build_prompt_plan(scene, tax)
    assert plan.components == ["wall"]
    for room in tax.rooms:
        assert not {"rug", "cupboard", "cupboard_lock"} & set(build_prompt_plan(SceneResult(room=room), tax).components)
