"""Pydantic schemas: model I/O contracts, unified inspection JSON, dataset metadata."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ResultState(str, Enum):
    """Spec phase 18. Never derived from model 'probabilities'."""
    DETECTED = "DETECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    MANUAL_CHECK_REQUIRED = "MANUAL_CHECK_REQUIRED"
    NOT_DETECTED = "NOT_DETECTED"
    UNKNOWN = "UNKNOWN"


Severity = Literal["minor", "moderate", "severe", "unknown"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def rounded(self) -> "BBox":
        return BBox(x1=round(self.x1), y1=round(self.y1), x2=round(self.x2), y2=round(self.y2))


# ---------------------------------------------------------------- model contracts

class SceneComponent(BaseModel):
    name: str
    raw_name: str = ""
    visible_condition_note: str = ""


class SceneResult(BaseModel):
    room: str = "unknown"
    raw_room: str = ""
    components: list[SceneComponent] = Field(default_factory=list)
    image_type: Literal["overview", "component", "close_up", "unknown"] = "unknown"
    notes: str = ""
    source: Literal["model", "user", "fallback"] = "model"
    valid_json: bool = True
    authenticity: str = ""   # Gemini: real_photo | screen_photo | rendering | ai_generated | unsure


class VerificationResult(BaseModel):
    """Gemini crop verification, normalized."""
    component: str = ""
    defect_present: Optional[bool] = None       # None = could not be parsed / undecided
    defect_type: str = ""
    severity: Severity = "unknown"
    description: str = ""
    manual_check_required: bool = False
    alternative_explanation: str = ""            # e.g. "grout line", "shadow"
    valid_json: bool = True

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str:
        v = str(v or "unknown").strip().lower()
        return v if v in ("minor", "moderate", "severe", "unknown") else "unknown"


class LocalizerDetection(BaseModel):
    label: str                     # phrase the localizer returned
    box: BBox                      # ORIGINAL image pixel coordinates
    component: str = ""
    defect: str = ""
    source: str = "full"           # full | tile
    merged_labels: list[str] = Field(default_factory=list)
    repeated: bool = False         # one of many similar same-label boxes (pattern / component instances)


class SpecialistEvidence(BaseModel):
    name: str = "line_evidence_cv"
    linear_score: float = 0.0      # 0..1, thin elongated dark structures
    regularity: float = 0.0        # 0..1, high = grid-like (grout / pattern)
    blotch_score: float = 0.0      # 0..1, compact dark/colour anomaly
    note: str = ""


# ---------------------------------------------------------------- unified output (phase 7)

class DefectInfo(BaseModel):
    type: str
    label: str = ""
    visible: bool = True
    severity: Severity = "unknown"


class ModelsUsed(BaseModel):
    localization: str = "LocateAnything-3B"
    verification: str = "Gemma-4-E4B"
    scene: str = ""                   # "" = same model as verification
    validation: str = ""              # second-opinion model ("" = none)


class Detection(BaseModel):
    issue_id: str
    room: str
    component: str
    component_label: str = ""
    defect: DefectInfo
    localization: BBox
    state: ResultState
    visible_evidence: str = ""
    assessment: str = ""                    # detected | needs_review | not_detected | unknown
    description: str = ""
    manual_check_required: bool = False
    manual_checks: list[str] = Field(default_factory=list)
    professional_review: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    localizer_label: str = ""
    specialist: Optional[SpecialistEvidence] = None
    crop_path: str = ""
    source_image: str = ""
    models: ModelsUsed = Field(default_factory=ModelsUsed)
    # Gemini second opinion: {model, agrees, severity, description, alternative, effect}
    validation: Optional[dict[str, Any]] = None


class ComponentStatus(BaseModel):
    component: str
    label: str
    visible_state: ResultState          # DETECTED / NEEDS_REVIEW / NOT_DETECTED for visible checks
    visible_summary: str
    manual_checks: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)


class ImageQuality(BaseModel):
    width: int
    height: int
    blur_score: float                   # variance of Laplacian (higher = sharper)
    brightness: float                   # 0..255 mean luminance
    contrast: float                     # luminance std
    ok: bool
    warnings: list[str] = Field(default_factory=list)


class Timings(BaseModel):
    preprocess_s: float = 0.0
    scene_s: float = 0.0
    localization_s: float = 0.0
    verification_s: float = 0.0
    validation_s: float = 0.0
    total_s: float = 0.0
    peak_vram_mb: dict[str, float] = Field(default_factory=dict)


class Inspection(BaseModel):
    inspection_id: str
    property_id: str = ""
    created_at: datetime
    status: Literal["queued", "running", "done", "failed"] = "queued"
    stage: str = ""
    error: str = ""
    room: str = "unknown"
    room_label: str = ""
    source_image: str = ""
    image_quality: Optional[ImageQuality] = None
    scene: Optional[SceneResult] = None
    prompts: list[str] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    rejected: list[Detection] = Field(default_factory=list)   # verifier said "not present" (debug / eval)
    components: list[ComponentStatus] = Field(default_factory=list)
    manual_checks: list[dict[str, Any]] = Field(default_factory=list)
    unknown_items: list[str] = Field(default_factory=list)
    timings: Timings = Field(default_factory=Timings)
    options: dict[str, Any] = Field(default_factory=dict)
    risk: Optional[dict[str, Any]] = None        # risk_service: condition score, repair cost, valuation impact
    integrity: Optional[dict[str, Any]] = None   # integrity_service: EXIF / duplicate-photo fraud signals
    ai_summary: str = ""                         # Gemini underwriter summary
    disclaimer: str = (
        "AI-assisted visual screening only. Findings describe visible appearance; they do not "
        "establish structural safety, root cause, or whether any mechanism, lock, motor or fan works. "
        "Items marked Needs Review / Manual Check must be confirmed by a person."
    )


# ---------------------------------------------------------------- dataset metadata (phase 9)

class DatasetDefect(BaseModel):
    type: str
    component: Optional[str] = None
    bbox: list[float] = Field(min_length=4, max_length=4)


class DatasetRecord(BaseModel):
    image: str
    property: str
    room: str
    component: Optional[str] = None
    defects: list[DatasetDefect] = Field(default_factory=list)
    lighting: Literal["normal", "low", "bright", "mixed", "unknown"] = "unknown"
    distance: Literal["close", "component", "overview", "unknown"] = "unknown"
    verified_by: Literal["human", "model", "unverified"] = "unverified"
    hard_negative: list[str] = Field(default_factory=list)
    notes: str = ""
