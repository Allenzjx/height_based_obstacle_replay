"""Pure state container for the Vision Task workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VISION_SOURCE_GENERATED = "generated_test_obstacle"
VISION_SOURCE_EXTERNAL = "external_unknown_obstacle"

PHASE_INACTIVE = "INACTIVE"
PHASE_READY = "READY"
PHASE_GENERATING_OBSTACLE = "GENERATING_OBSTACLE"
PHASE_WAITING_FOR_SCENE = "WAITING_FOR_SCENE"
PHASE_DETECTING = "DETECTING"
PHASE_STABLE_DETECTED = "STABLE_DETECTED"
PHASE_VALIDATING = "VALIDATING"
PHASE_VALIDATION_FAILED = "VALIDATION_FAILED"
PHASE_VALIDATED = "VALIDATED"
PHASE_STEPS_LOADING = "STEPS_LOADING"
PHASE_STEPS_READY = "STEPS_READY"
PHASE_PLAYING = "PLAYING"
PHASE_BLOCKED = "BLOCKED"


@dataclass
class VisionTaskState:
    active: bool = False
    source_mode: str = VISION_SOURCE_GENERATED
    phase: str = PHASE_INACTIVE
    requested_height_cm: int | None = None
    generated_height_cm: int | None = None
    scene_height_cm: int | None = None
    obstacle_revision: int = 0
    generation_request_id: str = ""
    generation_frame_baseline: int = 0
    generation_detection_baseline: int = 0
    detected_height_cm: int | None = None
    detected_revision: int = 0
    validation_checked: bool = False
    validation_passed: bool = False
    validated_height_cm: int | None = None
    validated_detection_revision: int = 0
    steps_ready: bool = False
    steps_height_cm: int | None = None
    steps_path: str = ""
    steps_count: int = 0
    last_action: str = ""
    block_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inactive_state(source_mode: str = VISION_SOURCE_GENERATED) -> VisionTaskState:
    return VisionTaskState(active=False, source_mode=source_mode, phase=PHASE_INACTIVE)


def ready_state(source_mode: str) -> VisionTaskState:
    return VisionTaskState(active=True, source_mode=source_mode, phase=PHASE_READY, last_action="started")


def normalize_source_mode(value: str) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("/", "_")
    text = "_".join(part for part in text.split("_") if part)
    if text in {"external", "unknown", "external_unknown", "external_unknown_obstacle"}:
        return VISION_SOURCE_EXTERNAL
    return VISION_SOURCE_GENERATED
