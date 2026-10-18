"""End-to-end inspection pipeline (spec phase 15 / section 30).

    image -> quality/preprocess + integrity (EXIF / duplicate photo) -> Gemini scene -> prompt router
          -> LocateAnything-3B boxes (local GPU) -> high-res crops -> Gemma 4 E4B verification (+ CV evidence)
          -> rule engine -> Gemini second opinion (agree -> DETECTED, disagree -> NEEDS_REVIEW)
          -> risk engine (condition score, repair cost, valuation impact, underwriting)
          -> unified JSON -> annotated image -> reports -> optional GCS / BigQuery / Pub/Sub

Stages run stage-major over a batch (all scenes, all localization, all verification, ...): on one GPU the
local models are swapped only twice per batch, while Gemini stages run the images in parallel.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from backend.config.settings import Settings, get_settings
from backend.models.schemas import (BBox, DefectInfo, Detection, Inspection, LocalizerDetection, ModelsUsed,
                                    ResultState, SceneComponent, SceneResult)
from backend.services import fusion_service as fusion
from backend.services import image_service as imgs
from backend.services import cloud_service
from backend.services import integrity_service
from backend.services import report_service as reports
from backend.services import risk_service
from backend.services import specialist_service
from backend.services.gpu_service import VramMonitor
from backend.services.locate_service import Localizer, create_localizer
from backend.services.prompt_service import PromptPlan, build_prompt_plan
from backend.services.vlm_service import VisionService, create_vlm
from backend.services.storage import Store
from backend.services.taxonomy import Taxonomy, get_taxonomy

log = logging.getLogger(__name__)
Progress = Callable[[str, str], None]  # (inspection_id, stage)


@dataclass
class Job:
    ins: Inspection
    original: Image.Image
    locate_img: Image.Image
    locate_scale: float
    room_hint: str | None = None
    component_hint: str | None = None
    tiled: bool = False
    plan: PromptPlan | None = None
    candidates: list[LocalizerDetection] = field(default_factory=list)
    raw_locate: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)


class InspectionPipeline:
    def __init__(self, s: Settings | None = None, localizer: Localizer | None = None, vision: VisionService | None = None):
        self.s = s or get_settings()
        self.tax: Taxonomy = get_taxonomy(self.s.taxonomy_path)
        self.localizer = localizer or create_localizer(self.s)
        self.vision = vision or VisionService.from_settings(self.tax, self.s)
        self.store = Store(self.s.outputs_dir)

    # ------------------------------------------------------------ public API

    def create(self, image_bytes: bytes, filename: str, property_id: str = "", room_type: str | None = None,
               component: str | None = None, tiled: str | None = None,
               property_value: float | None = None) -> Inspection:
        """Validate + persist the upload and return a queued inspection (fast, no models)."""
        original = imgs.load_image(image_bytes)          # raises ValueError on bad input
        iid = self.store.new_id()
        d = self.store.dir(iid)
        ext = Path(filename).suffix.lower() if Path(filename).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp") else ".jpg"
        (d / f"source{ext}").write_bytes(image_bytes)    # untouched original upload
        original.save(d / "original.jpg", quality=95)    # EXIF-oriented RGB copy used for crops
        ins = Inspection(inspection_id=iid, property_id=property_id or "", created_at=datetime.now(),
                         source_image=Path(filename).name, status="queued", stage="queued",
                         options={"room_type": room_type, "component": component, "tiled": tiled or self.s.tiled,
                                  "property_value": property_value})
        ins.integrity = integrity_service.check(image_bytes, original, ins, self.s)
        self.store.save(ins)
        return ins

    def run(self, inspection_id: str, progress: Progress | None = None) -> Inspection:
        return self.run_many([inspection_id], progress)[0]

    def run_many(self, ids: list[str], progress: Progress | None = None) -> list[Inspection]:
        jobs: list[Job] = []
        results: dict[str, Inspection] = {}
        for iid in ids:
            try:
                jobs.append(self._prepare(iid, progress))
            except Exception as e:  # noqa: BLE001
                results[iid] = self._fail(iid, e)
        for stage in (self._stage_scene, self._stage_locate, self._stage_verify, self._stage_validate,
                      self._stage_finalize):
            def one(j: Job, stage=stage) -> Job | None:
                try:
                    stage(j, progress)
                    return j
                except Exception as e:  # noqa: BLE001
                    log.exception("inspection %s failed in %s", j.ins.inspection_id, stage.__name__)
                    results[j.ins.inspection_id] = self._fail(j.ins.inspection_id, e, j.ins)
                    return None
            if not jobs:
                break
            # Images in parallel; local GPU models are serialised inside by the GPU arbiter.
            with ThreadPoolExecutor(max(1, min(4, len(jobs)))) as ex:
                jobs = [j for j in ex.map(one, jobs) if j is not None]
        for j in jobs:
            results[j.ins.inspection_id] = j.ins
        return [results[i] for i in ids]

    def inspect_file(self, path: str | Path, **kw) -> Inspection:
        p = Path(path)
        ins = self.create(p.read_bytes(), p.name, **kw)
        return self.run(ins.inspection_id)

    # ------------------------------------------------------------ stages

    def _progress(self, j: Job, stage: str, progress: Progress | None) -> None:
        j.ins.stage = stage
        j.ins.status = "running"
        self.store.save(j.ins)
        if progress:
            progress(j.ins.inspection_id, stage)

    def _prepare(self, iid: str, progress: Progress | None) -> Job:
        ins = self.store.load(iid)
        t = time.perf_counter()
        original = Image.open(self.store.dir(iid) / "original.jpg").convert("RGB")
        opts = ins.options
        tiled_opt = (opts.get("tiled") or "off").lower()
        tiled = tiled_opt == "on" or (tiled_opt == "auto" and max(original.size) > self.s.tile_auto_min_side)
        locate_img, scale = imgs.resize_max_side(original, self.s.locate_max_side)
        ins.image_quality = imgs.assess_quality(original)
        ins.timings.preprocess_s = round(time.perf_counter() - t, 3)
        ins.options["models"] = ModelsUsed(localization=self.localizer.model_name,
                                           verification=self.vision.vlm.model_name,
                                           scene=self.vision.scene_vlm.model_name,
                                           validation=self.vision.validator.model_name
                                           if self.vision.validator else "").model_dump()
        j = Job(ins=ins, original=original, locate_img=locate_img, locate_scale=scale,
                room_hint=opts.get("room_type") or None, component_hint=opts.get("component") or None, tiled=tiled)
        j.t0 = time.perf_counter() - ins.timings.preprocess_s
        self._progress(j, "preprocessed", progress)
        return j

    def _stage_scene(self, j: Job, progress: Progress | None) -> None:
        self._progress(j, "scene", progress)
        t = time.perf_counter()
        with VramMonitor() as vm:
            if j.room_hint and j.component_hint:
                comp = self.tax.normalize_component(j.component_hint) or j.component_hint
                scene = SceneResult(room=self.tax.normalize_room(j.room_hint), raw_room=j.room_hint, source="user",
                                    components=[SceneComponent(name=comp, raw_name=j.component_hint)])
                raw = "(skipped: room and component supplied by user)"
            else:
                scene, raw = self.vision.identify_scene(j.original)
                if j.room_hint:
                    scene.raw_room = f"{scene.raw_room} (user: {j.room_hint})"
                    scene.room = self.tax.normalize_room(j.room_hint)
                    scene.source = "user"
        self.store.write_raw(j.ins.inspection_id, "scene_raw.txt", raw)
        j.ins.scene = scene
        if scene.authenticity:
            j.ins.integrity = integrity_service.add_authenticity(j.ins.integrity, scene.authenticity)
        j.ins.room = scene.room
        j.ins.room_label = self.tax.room_label(scene.room)
        j.plan = build_prompt_plan(scene, self.tax, j.component_hint)
        j.ins.prompts = [p.phrase for p in j.plan.phrases]
        j.ins.timings.scene_s = round(time.perf_counter() - t, 3)
        j.ins.timings.peak_vram_mb["scene"] = round(vm.peak_mb, 0)
        if not scene.valid_json:
            j.ins.unknown_items.append("Scene analysis output was not valid JSON; routed by room defaults.")

    def _stage_locate(self, j: Job, progress: Progress | None) -> None:
        self._progress(j, "localization", progress)
        t = time.perf_counter()
        W, H = j.original.size
        views: list[tuple[Image.Image, int, int, float, str]] = [(j.locate_img, 0, 0, j.locate_scale, "full")]
        if j.tiled:
            for tile in imgs.make_tiles(j.original, self.s.tile_size, self.s.tile_overlap):
                timg, ts = imgs.resize_max_side(tile.image, self.s.locate_max_side)
                views.append((timg, tile.x0, tile.y0, ts, "tile"))
        cands: list[LocalizerDetection] = []
        with VramMonitor() as vm:
            for vimg, x0, y0, sc, src in views:
                data = imgs.to_jpeg_bytes(vimg)
                # Gemini takes the whole phrase list in one call; small local detectors need batches.
                batches = [j.plan.phrases] if self.localizer.name == "gemini" and j.plan.phrases else j.plan.batches
                for batch in batches:
                    phrases = [p.phrase for p in batch]
                    tb = time.perf_counter()
                    boxes = self.localizer.locate(data, phrases)
                    j.raw_locate.append({"view": src, "offset": [x0, y0], "scale": sc, "phrases": phrases,
                                         "seconds": round(time.perf_counter() - tb, 2),
                                         "boxes": [b.__dict__ for b in boxes]})
                    for b in boxes:
                        box = BBox(x1=x0 + b.x1 / sc, y1=y0 + b.y1 / sc, x2=x0 + b.x2 / sc, y2=y0 + b.y2 / sc)
                        box = imgs.clip_box(box, W, H)
                        if not imgs.valid_box(box, W, H, self.s.min_box_side):
                            continue
                        rp = j.plan.lookup(b.label)
                        if rp:
                            comp, defect = rp.component, rp.defect
                        else:
                            comp = self.tax.normalize_component(b.label) or (j.plan.components[0] if j.plan.components else "wall")
                            defect = self.tax.normalize_defect(b.label, comp)
                        cands.append(LocalizerDetection(label=b.label, box=box, component=comp, defect=defect, source=src))
        self.store.write_raw(j.ins.inspection_id, "localizer_raw.json", j.raw_locate)
        merged = fusion.merge_candidates(cands, self.s.nms_iou)
        fusion.flag_repetitions(merged)
        # Bound verification cost: specific (not whole-image) boxes first, keep plan order otherwise.
        order = {p.phrase: i for i, p in enumerate(j.plan.phrases)}
        merged.sort(key=lambda c: (c.box.area / (W * H) > self.s.max_box_area_frac, order.get(c.label, 999)))
        if len(merged) > self.s.max_candidates:
            skipped = merged[self.s.max_candidates:]
            j.ins.unknown_items.append(
                f"{len(skipped)} further localized candidate(s) were not verified (limit {self.s.max_candidates}): "
                + ", ".join(sorted({c.label for c in skipped})))
            merged = merged[: self.s.max_candidates]
        j.candidates = merged
        j.ins.timings.localization_s = round(time.perf_counter() - t, 3)
        j.ins.timings.peak_vram_mb["localization"] = round(vm.peak_mb, 0)

    def _stage_verify(self, j: Job, progress: Progress | None) -> None:
        self._progress(j, "verification", progress)
        t = time.perf_counter()
        W, H = j.original.size
        d = self.store.dir(j.ins.inspection_id)
        (d / "crops").mkdir(exist_ok=True)
        raws = []
        dets: list[Detection] = []
        with VramMonitor() as vm:
            prepared = []
            for n, c in enumerate(j.candidates, 1):
                issue_id = f"{j.ins.inspection_id}-{n:02d}"
                cb = imgs.padded_crop_box(c.box, W, H, self.s.crop_padding, self.s.crop_min_side)
                crop_img = imgs.boxed_crop(j.original, c.box, cb)   # full-res, candidate outlined
                crop_rel = f"crops/{issue_id}.jpg"
                crop_img.save(d / crop_rel, quality=92)
                ctx = imgs.context_view(j.original, c.box) if self.s.verify_context else None
                prepared.append((issue_id, c, cb, crop_img, crop_rel, ctx))
            # Cloud verifier: concurrent calls; local verifier (Gemma): one GPU, sequential.
            par = self.s.gemini_parallel if self.vision.vlm.name in ("gemini", "mock") else 1
            with ThreadPoolExecutor(max(1, par)) as ex:
                verdicts = list(ex.map(lambda p: self.vision.verify_defect(p[3], p[5], p[1].component, p[1].defect),
                                       prepared))
            for (issue_id, c, cb, crop_img, crop_rel, ctx), (v, raw) in zip(prepared, verdicts):
                spec = specialist_service.analyze(imgs.crop(j.original, c.box)) if self.s.specialist else None
                state, reasons = fusion.decide(v, c, W, H, j.ins.image_quality, spec, self.s.max_box_area_frac, self.tax)
                defect = v.defect_type if (v.defect_present and v.defect_type in self.tax.defects) else c.defect
                cdef = self.tax.components.get(c.component)
                manual = list(cdef.manual_checks) if cdef else []
                prof = self.tax.professional_checks(c.component, defect) if state != ResultState.NOT_DETECTED else []
                desc = fusion.sanitize_description(v.description)
                det = Detection(
                    issue_id=issue_id, room=j.ins.room, component=c.component,
                    component_label=self.tax.component_label(c.component),
                    defect=DefectInfo(type=defect, label=self.tax.defect_label(defect),
                                      visible=bool(v.defect_present), severity=v.severity),
                    localization=c.box.rounded(), state=state,
                    visible_evidence=desc, description=desc,
                    assessment={ResultState.DETECTED: "detected", ResultState.NEEDS_REVIEW: "needs_review",
                                ResultState.NOT_DETECTED: "not_detected"}.get(state, "unknown"),
                    manual_check_required=bool(manual or prof or v.manual_check_required
                                               or fusion.functional_claim(v.description)),
                    manual_checks=manual, professional_review=prof, review_reasons=reasons,
                    localizer_label=" / ".join([c.label] + c.merged_labels), specialist=spec, crop_path=crop_rel,
                    source_image=j.ins.source_image, models=ModelsUsed(**j.ins.options["models"]))
                dets.append(det)
                raws.append({"issue_id": issue_id, "candidate": c.model_dump(), "crop_box": cb.model_dump(),
                             "raw": raw, "parsed": v.model_dump()})
                if state == ResultState.UNKNOWN:
                    j.ins.unknown_items.append(f"#{issue_id}: {det.component_label} {det.defect.label.lower()} "
                                               "candidate could not be verified (invalid model output).")
        self.store.write_raw(j.ins.inspection_id, "verification_raw.json", raws)
        # One place = one finding: boxes that overlap (IoU > 0.5) or where one lies inside the other
        # (>= 80 %) are the same spot, even if the localizer gave them different names. Keep the best one:
        # DETECTED first, then a specific defect over generic "surface damage", then the tighter box.
        kept: list[Detection] = []
        rank = {ResultState.DETECTED: 0, ResultState.NEEDS_REVIEW: 1}
        order = lambda x: (rank.get(x.state, 2), x.defect.type == "surface_damage", x.localization.area)
        for x in sorted((x for x in dets if x.state != ResultState.NOT_DETECTED), key=order):
            dup = next((k for k in kept if imgs.iou(k.localization, x.localization) > 0.5
                        or imgs.containment(k.localization, x.localization) >= 0.8), None)
            if dup is None:
                kept.append(x)
            else:
                x.state = ResultState.NOT_DETECTED
                x.review_reasons = [f"Duplicate of {dup.issue_id} (same place)."]
        j.ins.detections = sorted(kept, key=lambda x: x.issue_id)
        j.ins.rejected = [x for x in dets if x.state == ResultState.NOT_DETECTED]
        j.ins.timings.verification_s = round(time.perf_counter() - t, 3)
        j.ins.timings.peak_vram_mb["verification"] = round(vm.peak_mb, 0)

    def _stage_validate(self, j: Job, progress: Progress | None) -> None:
        """Gemini second opinion on every finding the verifier kept (strict prompt, same outlined crop).
        Both agree -> DETECTED (unless a hard rule-engine reason remains); disagree -> NEEDS_REVIEW;
        verifier output unusable -> Gemini decides between NEEDS_REVIEW and NOT_DETECTED."""
        if self.vision.validator is None or not j.ins.detections:
            return
        self._progress(j, "validation", progress)
        t = time.perf_counter()
        d = self.store.dir(j.ins.inspection_id)
        dets = list(j.ins.detections)

        def check(det: Detection):
            crop = Image.open(d / det.crop_path).convert("RGB")
            return self.vision.validate_defect(crop, det.component, det.defect.type)

        with ThreadPoolExecutor(max(1, self.s.gemini_parallel)) as ex:
            verdicts = list(ex.map(check, dets))
        raws, keep, dropped = [], [], []
        for det, (v, raw) in zip(dets, verdicts):
            agrees = v.defect_present
            before = det.state
            reasons = list(det.review_reasons)
            if agrees is None:
                effect = "validator output unusable - unchanged"
            elif agrees:
                soft = [r for r in reasons if r.startswith(fusion.SOFT_REASONS)]
                reasons = [r for r in reasons if r not in soft]
                if det.defect.severity in (None, "unknown") and v.severity not in (None, "unknown"):
                    det.defect.severity = v.severity
                if before == ResultState.UNKNOWN:
                    det.state = ResultState.NEEDS_REVIEW
                    reasons.append("Verifier output was unusable; only the Gemini second opinion confirms it.")
                else:
                    det.state = ResultState.NEEDS_REVIEW if reasons else ResultState.DETECTED
                effect = "confirmed"
            else:
                if before == ResultState.UNKNOWN:
                    det.state = ResultState.NOT_DETECTED
                else:
                    det.state = ResultState.NEEDS_REVIEW
                alt = f" ({v.alternative_explanation})" if v.alternative_explanation else ""
                reasons.append(f"Gemini second opinion did not confirm this{alt}.")
                effect = "disputed"
            det.review_reasons = reasons
            det.assessment = {ResultState.DETECTED: "detected", ResultState.NEEDS_REVIEW: "needs_review",
                              ResultState.NOT_DETECTED: "not_detected"}.get(det.state, "unknown")
            det.validation = {"model": self.vision.validator.model_name, "agrees": agrees, "severity": v.severity,
                              "description": fusion.sanitize_description(v.description),
                              "alternative": v.alternative_explanation, "effect": effect,
                              "state_before": before.value, "state_after": det.state.value}
            det.models.validation = self.vision.validator.model_name
            (dropped if det.state == ResultState.NOT_DETECTED else keep).append(det)
            raws.append({"issue_id": det.issue_id, "raw": raw, "parsed": v.model_dump(), "effect": effect})
        j.ins.detections = keep
        j.ins.rejected = j.ins.rejected + dropped
        self.store.write_raw(j.ins.inspection_id, "validation_raw.json", raws)
        j.ins.timings.validation_s = round(time.perf_counter() - t, 3)

    def _stage_finalize(self, j: Job, progress: Progress | None) -> None:
        self._progress(j, "reporting", progress)
        t_fin = time.perf_counter()
        ins = j.ins
        comps = list(j.plan.components)
        for dd in ins.detections:
            if dd.component not in comps:
                comps.append(dd.component)
        ins.components = fusion.component_statuses(comps, ins.detections, self.tax)
        ins.manual_checks = fusion.manual_check_items(ins.components, ins.detections)
        ins.risk = risk_service.assess_inspection(ins, self.s)
        d = self.store.dir(ins.inspection_id)
        reports.annotate(j.original, ins.detections).save(d / "annotated.jpg", quality=92)
        reports.annotate(j.original, ins.detections, ins.rejected).save(d / "annotated_debug.jpg", quality=88)
        tm = ins.timings   # own compute time (in stage-major batches wall time includes other images)
        tm.total_s = round(tm.preprocess_s + tm.scene_s + tm.localization_s + tm.verification_s + tm.validation_s
                           + (time.perf_counter() - t_fin), 3)
        ins.status, ins.stage = "done", "done"
        self.store.save(ins)
        self.write_reports(ins)
        cloud_service.publish_inspection(ins, d, self.s)   # no-op unless GCS / BigQuery / Pub/Sub configured
        if progress:
            progress(ins.inspection_id, "done")

    def write_reports(self, ins: Inspection) -> None:
        d = self.store.dir(ins.inspection_id)
        (d / "report.html").write_text(reports.render_html([ins], [d]), encoding="utf-8")
        (d / "report.md").write_text(reports.render_markdown(ins), encoding="utf-8")
        (d / "report.txt").write_text(reports.render_text(ins), encoding="utf-8")
        (d / "result.json").write_text(json.dumps(self.public_json(ins), indent=2, default=str), encoding="utf-8")

    @staticmethod
    def public_json(ins: Inspection) -> dict:
        """The spec's compact 'unified inspection JSON' (phase 7 target schema)."""
        return {
            "inspection_id": ins.inspection_id, "property_id": ins.property_id, "room": ins.room,
            "source_image": ins.source_image,
            "detections": [{
                "issue_id": d.issue_id, "component": d.component,
                "defect": {"type": d.defect.type, "visible": d.defect.visible, "severity": d.defect.severity},
                "localization": {k: int(v) for k, v in d.localization.model_dump().items()},
                "state": d.state.value, "description": d.description,
                "manual_check_required": d.manual_check_required,
                "models": d.models.model_dump(),
            } for d in ins.detections],
            "manual_checks": ins.manual_checks,
            "unknown_items": ins.unknown_items,
            "risk": ins.risk,
            "integrity": ins.integrity,
        }

    def _fail(self, iid: str, e: Exception, ins: Inspection | None = None) -> Inspection:
        try:
            ins = ins or self.store.load(iid)
        except Exception:  # noqa: BLE001
            ins = Inspection(inspection_id=iid, created_at=datetime.now())
        ins.status, ins.error = "failed", f"{type(e).__name__}: {e}"
        try:
            self.store.save(ins)
        except Exception:  # noqa: BLE001
            pass
        return ins
