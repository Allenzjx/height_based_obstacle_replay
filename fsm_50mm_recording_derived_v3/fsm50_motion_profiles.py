"""Recording-derived, phase-local motion profiles for the 50 mm Macro FSM.

The authoritative inputs are the SHA-bound accepted steps and production Fast
Replay plan identity.  The audit export remains the command/ownership view,
while this module also rebuilds and binds every full PlaybackSegment and event
field needed by measured endpoint completion.  It does not reinterpret source
commands or synthesize gait endpoints; complete adapter target snapshots remain
the source-action transport representation.

There is deliberately no Isaac dependency here.  The worker rebuilds this
library locally and binds its SHA before a run.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from command_model import (
    DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    clamp_servo_command,
)
from completion_aware_segment import SegmentCompletionSpec
from motion_speed import load_motion_reference
from playback import plan_from_steps, playback_plan_to_payload
from sequence_model import load_steps_jsonl

from .fsm50_macro_state_model import (
    MacroStateId,
    MacroSubphase,
    PHYSICAL_PHASE_TO_MACRO_STATE,
)


SUCCESS_RESULTS = {
    "REPLAY_TASK_SUCCESS",
    "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE",
}
DEFAULT_PRIMARY_VERSION = "v003_20260805_224517_157723_manual"
DEFAULT_DISPATCH_INTERVAL_S = 1.0 / 120.0


# Gate-B phase windows are functional evidence windows, so several of them
# intentionally overlap.  A command still needs exactly one runtime owner.
# These ranges are the result of a source-by-source causal cursor audit against
# the four sealed Gate-A successes: a traverse owns every command needed before
# its post-cross TOP event, including a concurrent pair-advance/recovery command
# when that physical response occurs inside it.  Feedback-only acknowledgement
# states are omitted instead of being given synthetic empty profiles.
CANONICAL_SEGMENT_OWNERSHIP_RANGES: Mapping[
    str, tuple[tuple[MacroStateId, int, int], ...]
] = {
    "v003_20260805_224517_157723_manual": (
        (MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT, 0, 6),
        (MacroStateId.S2_FR_TRAVERSE, 7, 23),
        # FL first reaches post-cross TOP transiently inside segment 40.  The
        # state-local TOP latch retains that causal event even though FL is AIR
        # again at the segment tail; later RR commands are not borrowed.
        (MacroStateId.S3_FL_TRAVERSE, 24, 40),
        (MacroStateId.S6_RR_TRAVERSE, 41, 56),
        (MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE, 57, 101),
        (MacroStateId.S9_FINAL_ADVANCE, 102, 103),
        (MacroStateId.S10_POSTURE_RECOVERY, 104, 111),
    ),
    "v008_20260806_211408_578700_manual": (
        (MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT, 0, 4),
        (MacroStateId.S2_FR_TRAVERSE, 5, 23),
        (MacroStateId.S3_FL_TRAVERSE, 24, 38),
        (MacroStateId.S4_FRONT_PAIR_ADVANCE, 39, 40),
        (MacroStateId.S5_PRE_RR_COM_SHIFT, 41, 60),
        (MacroStateId.S6_RR_TRAVERSE, 61, 74),
        (MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE, 75, 117),
        (MacroStateId.S10_POSTURE_RECOVERY, 118, 118),
    ),
    "v009_20260806_215232_433234_manual": (
        (MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT, 0, 6),
        (MacroStateId.S2_FR_TRAVERSE, 7, 28),
        (MacroStateId.S3_FL_TRAVERSE, 29, 51),
        (MacroStateId.S4_FRONT_PAIR_ADVANCE, 52, 53),
        (MacroStateId.S5_PRE_RR_COM_SHIFT, 54, 73),
        (MacroStateId.S6_RR_TRAVERSE, 74, 90),
        # FL is AIR at the S6 tail and only returns TOP after the segment-92
        # PRE_RL_COM_SHIFT command.  S7 therefore owns this causal support
        # prefix; a passive acknowledgement would have no evidenced progress.
        (MacroStateId.S7_PRE_RL_SUPPORT_SETUP, 91, 92),
        (MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE, 93, 130),
        (MacroStateId.S10_POSTURE_RECOVERY, 131, 131),
    ),
    "v010_20260806_220745_363972_manual": (
        (MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT, 0, 4),
        (MacroStateId.S2_FR_TRAVERSE, 5, 20),
        (MacroStateId.S3_FL_TRAVERSE, 21, 41),
        (MacroStateId.S4_FRONT_PAIR_ADVANCE, 42, 43),
        (MacroStateId.S5_PRE_RR_COM_SHIFT, 44, 60),
        (MacroStateId.S6_RR_TRAVERSE, 61, 91),
        (MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE, 92, 137),
        (MacroStateId.S9_FINAL_ADVANCE, 138, 139),
        (MacroStateId.S10_POSTURE_RECOVERY, 140, 141),
    ),
}


# Normally the alignment span and runtime owner are the same macro state.
# v009's S7 exception deliberately keeps the source action's Gate-B
# PRE_RL_COM_SHIFT label while leasing that causal prefix to the support guard;
# the Recording step/command is not renamed.
CANONICAL_PHASE_SOURCE_STATE: Mapping[
    tuple[str, MacroStateId], MacroStateId
] = {
    (
        "v009_20260806_215232_433234_manual",
        MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
    ): MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_targets(
    values: Mapping[str, Any], names: Sequence[str], *, label: str
) -> dict[str, float]:
    missing = sorted(set(names) - set(values))
    unknown = sorted(set(values) - set(names))
    if missing or unknown:
        raise ValueError(
            f"{label} must contain the exact canonical actuator set; "
            f"missing={missing} unknown={unknown}"
        )
    result = {name: float(values[name]) for name in names}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    return not normalized.startswith(("0", "false", "no", "reject", "unsuitable"))


def _parse_segment_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NONE", "MISSING"}:
        return None
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if not numbers:
        return None
    first, last = (numbers[0], numbers[0]) if len(numbers) == 1 else (numbers[0], numbers[-1])
    if first < 0 or last < first:
        raise ValueError(f"invalid fast_segment_range: {value!r}")
    return first, last


def _subphase_for_phase(phase: str) -> MacroSubphase:
    upper = str(phase).upper()
    if "RECOVERY" in upper:
        return MacroSubphase.RECOVERY
    if "UNLOAD" in upper:
        return MacroSubphase.UNLOAD
    if "LIFT" in upper:
        return MacroSubphase.LIFT
    if "FACE_CROSS" in upper:
        return MacroSubphase.FACE_CLEAR
    if "TOP_PLACE" in upper:
        return MacroSubphase.TOP_PLACE
    if "ADVANCE" in upper or "APPROACH" in upper:
        return MacroSubphase.ADVANCE
    if "SHIFT" in upper or "SUPPORT_SETUP" in upper:
        return MacroSubphase.PRELOAD
    return MacroSubphase.PRELOAD


@dataclass(frozen=True)
class RecordingProfileSource:
    source_version: str
    task_result: str
    plan_path: Path
    gate_a_run_dir: Path
    plan_sha256: str
    plan_file_sha256: str
    video_sha256: str = ""
    worker_request_path: Path | None = None
    worker_request_sha256: str = ""
    accepted_steps_path: Path | None = None
    accepted_steps_sha256: str = ""
    full_plan_payload_sha256: str = ""
    full_plan_payload: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.task_result not in SUCCESS_RESULTS:
            raise ValueError("motion profile sources must be Gate-A task successes")
        if not self.plan_path.is_file():
            raise FileNotFoundError(self.plan_path)
        if not self.gate_a_run_dir.is_dir():
            raise FileNotFoundError(self.gate_a_run_dir)
        if _sha256_file(self.plan_path) != self.plan_file_sha256:
            raise ValueError("Fast plan file SHA mismatch")
        if self.worker_request_path is not None and not self.worker_request_path.is_file():
            raise FileNotFoundError(self.worker_request_path)
        if self.accepted_steps_path is not None and not self.accepted_steps_path.is_file():
            raise FileNotFoundError(self.accepted_steps_path)
        if (
            self.worker_request_path is not None
            and _sha256_file(self.worker_request_path) != self.worker_request_sha256
        ):
            raise ValueError("Gate-A worker request SHA mismatch")
        if (
            self.accepted_steps_path is not None
            and _sha256_file(self.accepted_steps_path) != self.accepted_steps_sha256
        ):
            raise ValueError("Gate-A accepted_steps SHA mismatch")
        payload = copy.deepcopy(dict(self.full_plan_payload or {}))
        if payload:
            payload_sha = _stable_sha256(payload)
            if payload_sha != self.full_plan_payload_sha256:
                raise ValueError("full PlaybackPlan payload SHA mismatch")
            if str(payload.get("plan_sha256", "")) != self.plan_sha256:
                raise ValueError("full PlaybackPlan identity differs from Fast plan")
        object.__setattr__(self, "full_plan_payload", payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "task_result": self.task_result,
            "plan_path": str(self.plan_path.resolve()),
            "gate_a_run_dir": str(self.gate_a_run_dir.resolve()),
            "plan_sha256": self.plan_sha256,
            "plan_file_sha256": self.plan_file_sha256,
            "video_sha256": self.video_sha256,
            "worker_request_path": (
                ""
                if self.worker_request_path is None
                else str(self.worker_request_path.resolve())
            ),
            "worker_request_sha256": self.worker_request_sha256,
            "accepted_steps_path": (
                ""
                if self.accepted_steps_path is None
                else str(self.accepted_steps_path.resolve())
            ),
            "accepted_steps_sha256": self.accepted_steps_sha256,
            "full_plan_payload_sha256": self.full_plan_payload_sha256,
        }


@dataclass(frozen=True)
class PhaseWindow:
    physical_phase: str
    first_segment: int
    last_segment: int

    def __post_init__(self) -> None:
        if self.physical_phase not in PHYSICAL_PHASE_TO_MACRO_STATE:
            raise ValueError(f"unknown canonical phase {self.physical_phase!r}")
        if self.first_segment < 0 or self.last_segment < self.first_segment:
            raise ValueError("phase segment bounds are invalid")


@dataclass(frozen=True)
class PhaseSpan:
    source_version: str
    state_id: MacroStateId
    strategy: str
    windows: tuple[PhaseWindow, ...]
    suitable_for_first_fsm: bool = True

    def __post_init__(self) -> None:
        if not self.source_version or not self.strategy:
            raise ValueError("source_version and strategy are required")
        if not self.windows:
            raise ValueError("a phase span requires at least one window")
        for window in self.windows:
            mapped = PHYSICAL_PHASE_TO_MACRO_STATE[window.physical_phase]
            if mapped != self.state_id:
                raise ValueError(
                    f"phase {window.physical_phase} belongs to {mapped.value}, "
                    f"not {self.state_id.value}"
                )

    @property
    def first_segment(self) -> int:
        return min(window.first_segment for window in self.windows)

    @property
    def last_segment(self) -> int:
        return max(window.last_segment for window in self.windows)

    @property
    def physical_phases(self) -> tuple[str, ...]:
        return tuple(window.physical_phase for window in self.windows)


@dataclass(frozen=True)
class MotionKeyframe:
    time_s: float
    source_time_s: float
    sequence_index: int
    source_segment_index: int
    source_step_index: int
    physical_phase: str
    subphase: MacroSubphase
    servo_targets_deg: Mapping[str, float]
    wheel_targets_rad_s: Mapping[str, float]
    commands: tuple[str, ...] = ()
    source_event_indices: tuple[int, ...] = ()
    dispatch_kind: str = "segment_start"
    atomic_concurrent: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_s)) or self.time_s < 0.0:
            raise ValueError("keyframe time_s must be finite and non-negative")
        if not math.isfinite(float(self.source_time_s)) or self.source_time_s < 0.0:
            raise ValueError("keyframe source_time_s must be finite and non-negative")
        servos = _finite_targets(self.servo_targets_deg, SERVO_JOINT_NAMES, label="servo targets")
        wheels = _finite_targets(self.wheel_targets_rad_s, WHEEL_JOINT_NAMES, label="wheel targets")
        if set(servos) != set(SERVO_JOINT_NAMES) or set(wheels) != set(WHEEL_JOINT_NAMES):
            raise ValueError("keyframes must carry complete canonical actuator maps")
        for name, target in servos.items():
            if abs(clamp_servo_command(name, target) - target) > 1.0e-9:
                raise ValueError(f"unsafe recording-derived servo target {name}={target}")
        if any(abs(value) > DEFAULT_MAX_WHEEL_SPEED_RAD_S + 1.0e-9 for value in wheels.values()):
            raise ValueError("recording-derived wheel target exceeds the production speed limit")
        if self.atomic_concurrent:
            changed_servo = bool(self.commands) and any("servo " in command for command in self.commands)
            moving_wheel = any(abs(value) > 1.0e-12 for value in wheels.values())
            if not (changed_servo and moving_wheel):
                raise ValueError("atomic_concurrent needs a servo command and non-zero wheel target")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subphase"] = self.subphase.value
        payload["servo_targets_deg"] = dict(self.servo_targets_deg)
        payload["wheel_targets_rad_s"] = dict(self.wheel_targets_rad_s)
        return payload


@dataclass(frozen=True)
class ProfileSample:
    keyframe_index: int
    keyframe: MotionKeyframe
    profile_fraction: float
    timeline_complete: bool


@dataclass(frozen=True)
class RecordingSegmentOwnership:
    """One immutable, provenance-bound segment range owned by one macro state."""

    source_version: str
    state_id: MacroStateId
    phase_source_state_id: MacroStateId
    first_segment: int
    last_segment: int
    source_plan_sha256: str
    evidence_basis: str

    def __post_init__(self) -> None:
        if not self.source_version or not self.source_plan_sha256:
            raise ValueError("segment ownership requires source and plan identity")
        if self.first_segment < 0 or self.last_segment < self.first_segment:
            raise ValueError("segment ownership bounds are invalid")
        if not self.evidence_basis.strip():
            raise ValueError("segment ownership requires a causal evidence basis")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "state_id": self.state_id.value,
            "phase_source_state_id": self.phase_source_state_id.value,
            "first_segment": self.first_segment,
            "last_segment": self.last_segment,
            "source_plan_sha256": self.source_plan_sha256,
            "evidence_basis": self.evidence_basis,
        }


@dataclass(frozen=True)
class PlaybackSegmentBinding:
    """One owned production segment plus its exact, SHA-bound event slice."""

    source_version: str
    source_plan_sha256: str
    source_plan_payload_sha256: str
    accepted_steps_sha256: str
    segment_payload: Mapping[str, Any]
    event_payloads: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.source_version:
            raise ValueError("segment binding requires source_version")
        for label, value in (
            ("source_plan_sha256", self.source_plan_sha256),
            ("source_plan_payload_sha256", self.source_plan_payload_sha256),
            ("accepted_steps_sha256", self.accepted_steps_sha256),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        segment = copy.deepcopy(dict(self.segment_payload or {}))
        events = tuple(copy.deepcopy(dict(row or {})) for row in self.event_payloads)
        required = {
            "segment_index",
            "source_step",
            "source_step_id",
            "event_start_index",
            "event_count",
            "servo_duration_s",
            "servo_targets",
            "wheel_active_duration_s",
            "explicit_hold_s",
            "servo_tolerance_deg",
            "recorded_servo_residual_deg",
            "legacy_missing_endpoint",
        }
        missing = sorted(required - set(segment))
        if missing:
            raise ValueError(f"segment binding payload is missing fields: {missing}")
        segment_index = int(segment["segment_index"])
        event_start = int(segment["event_start_index"])
        event_count = int(segment["event_count"])
        if segment_index < 0 or event_start < 0 or event_count < 0:
            raise ValueError("segment binding indices must be non-negative")
        if event_count != len(events):
            raise ValueError("segment binding event_count differs from event slice")
        for offset, event in enumerate(events):
            if int(event.get("segment_index", -1)) != segment_index:
                raise ValueError("segment binding contains a cross-segment event")
            global_index = int(event.get("global_command_index", 0) or 0)
            if global_index <= 0:
                raise ValueError("segment binding event lacks canonical command index")
            if int(event.get("source_step", -1)) != int(segment["source_step"]):
                raise ValueError("segment binding event source step differs")
            if event_start + offset < 0:
                raise ValueError("segment binding event offset is invalid")
        # The shared helper is the authoritative strictness boundary.  This
        # intentionally accepts legacy_missing_endpoint with its strict 1 deg
        # live tolerance rather than rejecting older successful recordings.
        SegmentCompletionSpec.from_mapping(
            {
                "segment_index": segment_index,
                "source_step": int(segment["source_step"]),
                "source_step_id": str(segment.get("source_step_id", "")),
                "servo_targets_deg": dict(segment.get("servo_targets", {}) or {}),
                "servo_duration_s": segment.get("servo_duration_s", 0.0),
                "servo_tolerance_deg": segment.get("servo_tolerance_deg", 1.0),
                "recorded_servo_residual_deg": dict(
                    segment.get("recorded_servo_residual_deg", {}) or {}
                ),
                "legacy_missing_endpoint": segment.get(
                    "legacy_missing_endpoint", False
                ),
                "wheel_active_duration_s": segment.get(
                    "wheel_active_duration_s", 0.0
                ),
                "explicit_hold_s": segment.get("explicit_hold_s", 0.0),
            }
        )
        object.__setattr__(self, "segment_payload", segment)
        object.__setattr__(self, "event_payloads", events)

    @property
    def segment_index(self) -> int:
        return int(self.segment_payload["segment_index"])

    @property
    def source_step(self) -> int:
        return int(self.segment_payload["source_step"])

    @property
    def source_step_id(self) -> str:
        return str(self.segment_payload.get("source_step_id", ""))

    @property
    def completion_spec(self) -> SegmentCompletionSpec:
        return SegmentCompletionSpec.from_mapping(
            {
                "segment_index": self.segment_index,
                "source_step": self.source_step,
                "source_step_id": self.source_step_id,
                "servo_targets_deg": dict(
                    self.segment_payload.get("servo_targets", {}) or {}
                ),
                "servo_duration_s": self.segment_payload.get(
                    "servo_duration_s", 0.0
                ),
                "servo_tolerance_deg": self.segment_payload.get(
                    "servo_tolerance_deg", 1.0
                ),
                "recorded_servo_residual_deg": dict(
                    self.segment_payload.get("recorded_servo_residual_deg", {})
                    or {}
                ),
                "legacy_missing_endpoint": self.segment_payload.get(
                    "legacy_missing_endpoint", False
                ),
                "wheel_active_duration_s": self.segment_payload.get(
                    "wheel_active_duration_s", 0.0
                ),
                "explicit_hold_s": self.segment_payload.get(
                    "explicit_hold_s", 0.0
                ),
            }
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.playback_segment_binding.v1",
            "source_version": self.source_version,
            "source_plan_sha256": self.source_plan_sha256,
            "source_plan_payload_sha256": self.source_plan_payload_sha256,
            "accepted_steps_sha256": self.accepted_steps_sha256,
            "segment": copy.deepcopy(dict(self.segment_payload)),
            "events": [copy.deepcopy(dict(row)) for row in self.event_payloads],
            "completion_spec": self.completion_spec.to_mapping(),
        }


@dataclass(frozen=True)
class PhaseMotionProfile:
    profile_id: str
    source_version: str
    state_id: MacroStateId
    strategy: str
    physical_phases: tuple[str, ...]
    keyframes: tuple[MotionKeyframe, ...]
    nominal_duration_s: float
    source_plan_sha256: str
    source_plan_file_sha256: str
    gate_a_run_dir: str
    source_segment_range: tuple[int, int]
    source_step_indices: tuple[int, ...]
    source_commands: tuple[str, ...]
    dispatch_interval_s: float = DEFAULT_DISPATCH_INTERVAL_S
    worker_request_path: str = ""
    worker_request_sha256: str = ""
    accepted_steps_path: str = ""
    accepted_steps_sha256: str = ""
    source_plan_payload_sha256: str = ""
    segment_bindings: tuple[PlaybackSegmentBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id or not self.source_version or not self.strategy:
            raise ValueError("profile identity is incomplete")
        if not self.keyframes:
            raise ValueError("a motion profile needs keyframes")
        times = [float(frame.time_s) for frame in self.keyframes]
        if any(after <= before for before, after in zip(times, times[1:])):
            raise ValueError("keyframe dispatch times must be strictly increasing")
        if self.nominal_duration_s + 1.0e-9 < times[-1]:
            raise ValueError("nominal_duration_s precedes the last keyframe")
        if not math.isfinite(self.nominal_duration_s) or self.nominal_duration_s < 0.0:
            raise ValueError("nominal_duration_s must be finite and non-negative")
        if self.segment_bindings:
            expected = list(
                range(self.source_segment_range[0], self.source_segment_range[1] + 1)
            )
            start_frames = [
                frame
                for frame in self.keyframes
                if frame.dispatch_kind == "segment_start"
            ]
            start_indices = [frame.source_segment_index for frame in start_frames]
            if len(start_indices) != len(set(start_indices)):
                raise ValueError("duplicate source action segment_start coordinate")
            actual = [binding.segment_index for binding in self.segment_bindings]
            if actual != expected:
                raise ValueError(
                    "profile segment bindings must exactly cover the owned range"
                )
            starts = {frame.source_segment_index: frame for frame in start_frames}
            if sorted(starts) != expected:
                raise ValueError("profile segment starts differ from segment bindings")
            for binding in self.segment_bindings:
                frame = starts[binding.segment_index]
                if (
                    binding.source_version != self.source_version
                    or binding.source_plan_sha256 != self.source_plan_sha256
                    or binding.source_step != frame.source_step_index
                    or binding.accepted_steps_sha256 != self.accepted_steps_sha256
                    or binding.source_plan_payload_sha256
                    != self.source_plan_payload_sha256
                ):
                    raise ValueError("profile segment binding identity mismatch")

    def segment_binding(self, segment_index: int) -> PlaybackSegmentBinding:
        matches = [
            binding
            for binding in self.segment_bindings
            if binding.segment_index == int(segment_index)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"profile has no exact completion binding for segment {segment_index}"
            )
        return matches[0]

    @property
    def final_servo_targets_deg(self) -> dict[str, float]:
        return dict(self.keyframes[-1].servo_targets_deg)

    @property
    def safe_hold_wheel_targets_rad_s(self) -> dict[str, float]:
        return {name: 0.0 for name in WHEEL_JOINT_NAMES}

    def sample(self, elapsed_s: float) -> ProfileSample:
        elapsed = max(0.0, float(elapsed_s))
        index = 0
        for candidate, frame in enumerate(self.keyframes):
            if frame.time_s <= elapsed + 1.0e-12:
                index = candidate
            else:
                break
        fraction = 1.0 if self.nominal_duration_s <= 1.0e-12 else min(
            1.0, elapsed / self.nominal_duration_s
        )
        return ProfileSample(
            keyframe_index=index,
            keyframe=self.keyframes[index],
            profile_fraction=fraction,
            timeline_complete=elapsed + 1.0e-9 >= self.nominal_duration_s,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.phase_motion_profile.v1",
            "profile_id": self.profile_id,
            "source_version": self.source_version,
            "state_id": self.state_id.value,
            "strategy": self.strategy,
            "physical_phases": list(self.physical_phases),
            "nominal_duration_s": self.nominal_duration_s,
            "source_plan_sha256": self.source_plan_sha256,
            "source_plan_file_sha256": self.source_plan_file_sha256,
            "gate_a_run_dir": self.gate_a_run_dir,
            "source_segment_range": list(self.source_segment_range),
            "source_step_indices": list(self.source_step_indices),
            "source_commands": list(self.source_commands),
            "dispatch_interval_s": self.dispatch_interval_s,
            "worker_request_path": self.worker_request_path,
            "worker_request_sha256": self.worker_request_sha256,
            "accepted_steps_path": self.accepted_steps_path,
            "accepted_steps_sha256": self.accepted_steps_sha256,
            "source_plan_payload_sha256": self.source_plan_payload_sha256,
            "segment_bindings": [
                binding.to_mapping() for binding in self.segment_bindings
            ],
            "keyframes": [frame.to_mapping() for frame in self.keyframes],
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_mapping())


@dataclass(frozen=True)
class MotionProfileLibrary:
    profiles: tuple[PhaseMotionProfile, ...]
    successful_sources: tuple[RecordingProfileSource, ...]
    alignment_path: str
    segment_ownership: tuple[RecordingSegmentOwnership, ...] = ()
    library_id: str = "fsm50-gate-c-successful-recording-profiles-v1"

    def __post_init__(self) -> None:
        ids = [profile.profile_id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("motion profile ids must be unique")
        keys = [
            (profile.source_version, profile.state_id, profile.strategy)
            for profile in self.profiles
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate source/state/strategy profile")
        if self.segment_ownership:
            self._validate_segment_ownership()

    def _validate_segment_ownership(self) -> None:
        source_by_version = {
            source.source_version: source for source in self.successful_sources
        }
        ownership_sources = {
            item.source_version for item in self.segment_ownership
        }
        if ownership_sources != set(source_by_version):
            raise ValueError(
                "canonical ownership sources must exactly match successful sources"
            )
        profile_by_source_state = {
            (profile.source_version, profile.state_id): profile
            for profile in self.profiles
        }
        if len(profile_by_source_state) != len(self.profiles):
            raise ValueError("canonical ownership permits one profile per source/state")
        ownership_keys: set[tuple[str, MacroStateId]] = set()
        for version, source in source_by_version.items():
            owned = sorted(
                (
                    item
                    for item in self.segment_ownership
                    if item.source_version == version
                ),
                key=lambda item: item.first_segment,
            )
            if not owned:
                raise ValueError(f"canonical ownership is missing source {version}")
            plan = json.loads(source.plan_path.read_text(encoding="utf-8-sig"))
            segment_count = len(list(plan.get("segments", ()) or ()))
            flattened = [
                segment
                for item in owned
                for segment in range(item.first_segment, item.last_segment + 1)
            ]
            if flattened != list(range(segment_count)):
                raise ValueError(
                    f"canonical ownership for {version} must cover 0..{segment_count - 1} "
                    "exactly once and in order"
                )
            for item in owned:
                if item.source_plan_sha256 != source.plan_sha256:
                    raise ValueError(
                        f"canonical ownership plan identity mismatch for {version}"
                    )
                key = (version, item.state_id)
                if key in ownership_keys:
                    raise ValueError("duplicate canonical source/state ownership")
                ownership_keys.add(key)
                profile = profile_by_source_state.get(key)
                if profile is None:
                    raise ValueError(
                        f"canonical ownership has no profile for {version}/{item.state_id.value}"
                    )
                if profile.source_segment_range != (
                    item.first_segment,
                    item.last_segment,
                ):
                    raise ValueError("profile range differs from canonical ownership")
                starts = [
                    frame.source_segment_index
                    for frame in profile.keyframes
                    if frame.dispatch_kind == "segment_start"
                ]
                if starts != list(range(item.first_segment, item.last_segment + 1)):
                    raise ValueError(
                        "profile segment-start actions do not exactly match ownership"
                    )
        if set(profile_by_source_state) != ownership_keys:
            raise ValueError("profiles and canonical segment ownership differ")

    def profiles_for_state(
        self, source_version: str, state_id: MacroStateId | str
    ) -> tuple[PhaseMotionProfile, ...]:
        wanted = state_id if isinstance(state_id, MacroStateId) else MacroStateId(state_id)
        return tuple(
            profile
            for profile in self.profiles
            if profile.source_version == source_version and profile.state_id == wanted
        )

    def get(
        self,
        source_version: str,
        state_id: MacroStateId | str,
        *,
        strategy: str = "PRIMARY_PROFILE",
    ) -> PhaseMotionProfile:
        candidates = self.profiles_for_state(source_version, state_id)
        exact = [profile for profile in candidates if profile.strategy == strategy]
        if exact:
            return exact[0]
        prefix = [
            profile
            for profile in candidates
            if profile.strategy.startswith(strategy + "_")
            or strategy.startswith(profile.strategy + "_")
        ]
        if prefix:
            return sorted(prefix, key=lambda item: item.profile_id)[0]
        if strategy == "PRIMARY_PROFILE" and len(candidates) == 1:
            return candidates[0]
        raise KeyError(f"no {source_version}/{MacroStateId(state_id).value}/{strategy} profile")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.motion_profile_library.v1",
            "library_id": self.library_id,
            "alignment_path": self.alignment_path,
            "successful_sources": [source.to_mapping() for source in self.successful_sources],
            "segment_ownership": [
                item.to_mapping()
                for item in sorted(
                    self.segment_ownership,
                    key=lambda value: (
                        value.source_version,
                        value.first_segment,
                        value.state_id.value,
                    ),
                )
            ],
            "profiles": [
                profile.to_mapping()
                for profile in sorted(self.profiles, key=lambda item: item.profile_id)
            ],
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_mapping())


def _canonical_full_plan_payload(plan: Any) -> dict[str, Any]:
    payload = playback_plan_to_payload(plan)
    timing = copy.deepcopy(dict(payload.get("timing", {}) or {}))
    # These two profiling timestamps are deliberately nondeterministic and
    # carry no motion semantics.  Every segment/event/completion field remains
    # present and is covered by source_plan_payload_sha256.
    timing.pop("plan_build_start", None)
    timing.pop("plan_build_end", None)
    payload["timing"] = timing
    if payload.get("path", "") != "":
        raise ValueError("rebuilt completion plan unexpectedly carries a path")
    return payload


def _legacy_v1_plan_fingerprint(plan: Any) -> str:
    """Rebuild the exact pre-execution-semantics Gate-A v1 plan identity."""

    payload = {
        "final_time_s": round(float(plan.final_time_s), 9),
        "profile": plan.profile,
        "total_steps": int(plan.total_steps),
        "events": [
            {
                "time_s": round(float(event.time_s), 9),
                "command": str(event.command),
                "source_step": event.source_step,
                "source_step_id": event.source_step_id,
                "command_index_in_step": event.command_index_in_step,
                "commands_in_step": event.commands_in_step,
                "global_command_index": event.global_command_index,
                "base_command": event.base_command,
                "base_duration_s": round(float(event.base_duration_s), 9),
                "planned_duration_s": round(float(event.planned_duration_s), 9),
                "segment_index": int(event.segment_index),
                "channel": event.channel,
                "dispatch_command": bool(event.dispatch_command),
            }
            for event in plan.events
        ],
        "segments": [
            {
                "segment_index": int(segment.segment_index),
                "source_step": int(segment.source_step),
                "event_start_index": int(segment.event_start_index),
                "event_count": int(segment.event_count),
                "servo_tolerance_deg": round(
                    float(segment.servo_tolerance_deg), 9
                ),
                "recorded_servo_residual_deg": {
                    name: round(float(value), 9)
                    for name, value in sorted(
                        segment.recorded_servo_residual_deg.items()
                    )
                },
                "legacy_missing_endpoint": bool(
                    segment.legacy_missing_endpoint
                ),
            }
            for segment in plan.segments
        ],
    }
    return _stable_sha256(payload)


def _legacy_v1_full_plan_payload(plan: Any, *, plan_sha256: str) -> dict[str, Any]:
    payload = _canonical_full_plan_payload(plan)
    if payload.get("execution_semantics") != "recorded_timeline_open_loop_v1":
        raise ValueError("legacy Gate-A v1 compatibility requires Fast open-loop content")
    payload.pop("execution_semantics")
    payload["plan_sha256"] = plan_sha256
    return payload


def _validate_fast_export_against_full_plan(
    document: Mapping[str, Any], full_payload: Mapping[str, Any]
) -> None:
    exported = list(document.get("segments", ()) or ())
    segments = list(full_payload.get("segments", ()) or ())
    events = list(full_payload.get("events", ()) or ())
    if len(exported) != len(segments):
        raise ValueError("Fast plan export segment count differs from rebuilt plan")
    for index, (row, segment) in enumerate(zip(exported, segments)):
        start = int(segment["event_start_index"])
        stop = start + int(segment["event_count"])
        segment_events = events[start:stop]
        comparisons = {
            "decoded_segment_index": int(segment["segment_index"]),
            "source_step_index": int(segment["source_step"]),
            "commands": [str(event["base_command"]) for event in segment_events],
            "servo_target_deg": dict(segment.get("servo_targets", {}) or {}),
            "wheel_target_rad_s": dict(
                segment.get("wheel_applied_target_rad_s", {}) or {}
            ),
            "servo_duration_s": float(segment["servo_duration_s"]),
            "wheel_duration_s": float(segment["wheel_active_duration_s"]),
            "explicit_hold_s": float(segment["explicit_hold_s"]),
            "final_segment_duration_s": float(segment["planned_end_s"])
            - float(segment["planned_start_s"]),
        }
        for key, expected in comparisons.items():
            actual = row.get(key)
            if isinstance(expected, float):
                try:
                    matches = abs(float(actual) - expected) <= 1.0e-9
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"Fast plan export differs from rebuilt plan at segment {index}.{key}"
                )


def _load_gate_a_full_plan_binding(
    *,
    source_version: str,
    run_dir: Path,
    exported_document: Mapping[str, Any],
) -> tuple[Path, str, Path, str, dict[str, Any], str]:
    request_path = (run_dir / "worker_task_replay_request.json").resolve()
    if not request_path.is_file():
        raise FileNotFoundError(request_path)
    request_sha = _sha256_file(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if str(request.get("schema_version", "")) != "fsm50.worker_task_replay_request.v1":
        raise ValueError("Gate-A worker request schema is invalid")
    if request.get("enabled") is not True:
        raise ValueError("Gate-A worker request is not enabled")
    if str(request.get("source_version", "")) != source_version:
        raise ValueError("Gate-A worker request source_version mismatch")
    if Path(str(request.get("run_dir", ""))).resolve() != run_dir.resolve():
        raise ValueError("Gate-A worker request run_dir mismatch")

    accepted_path = Path(str(request.get("accepted_steps_path", ""))).resolve()
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)
    accepted_sha = _sha256_file(accepted_path)
    if accepted_sha != str(request.get("accepted_steps_sha256", "")).lower():
        raise ValueError("Gate-A accepted_steps SHA mismatch")
    steps = load_steps_jsonl(accepted_path)
    if len(steps) != int(request.get("step_count", -1)):
        raise ValueError("Gate-A accepted step count differs from request")
    reference = load_motion_reference()
    rebuilt = plan_from_steps(
        steps,
        profile="fast",
        max_wheel_speed=reference.wheel_velocity_limit_rad_s,
        label=f"50 mm {source_version} full completion binding",
        sequence_total_steps=len(steps),
    )
    expected_plan_sha = str(request.get("plan_sha256", "")).lower()
    request_has_semantics = "execution_semantics" in request
    export_has_semantics = "execution_semantics" in exported_document
    if request_has_semantics != export_has_semantics:
        raise ValueError("Gate-A request/export execution semantics presence differs")
    if request_has_semantics:
        expected_semantics = str(request.get("execution_semantics", "") or "")
        if (
            expected_semantics != rebuilt.execution_semantics
            or str(exported_document.get("execution_semantics", "") or "")
            != expected_semantics
            or rebuilt.plan_sha256 != expected_plan_sha
        ):
            raise ValueError("Gate-A rebuilt plan semantics/SHA differs from request")
        full_payload = _canonical_full_plan_payload(rebuilt)
    else:
        legacy_plan_sha = _legacy_v1_plan_fingerprint(rebuilt)
        if legacy_plan_sha != expected_plan_sha:
            raise ValueError("Gate-A rebuilt legacy-v1 plan SHA differs from request")
        full_payload = _legacy_v1_full_plan_payload(
            rebuilt, plan_sha256=legacy_plan_sha
        )
    if expected_plan_sha != str(exported_document.get("plan_sha256", "")).lower():
        raise ValueError("Gate-A request plan SHA differs from Fast export")
    if len(rebuilt.events) != int(request.get("plan_event_count", -1)):
        raise ValueError("Gate-A rebuilt event count differs from request")
    if len(rebuilt.segments) != int(request.get("plan_segment_count", -1)):
        raise ValueError("Gate-A rebuilt segment count differs from request")
    _validate_fast_export_against_full_plan(exported_document, full_payload)
    return (
        request_path,
        request_sha,
        accepted_path,
        accepted_sha,
        full_payload,
        _stable_sha256(full_payload),
    )


def discover_successful_gate_a_sources(project_root: str | Path) -> tuple[RecordingProfileSource, ...]:
    """Discover only SHA-bound Gate-A task successes, never failed recordings."""

    root = Path(project_root).resolve()
    module_root = root / "fsm_50mm_recording_derived_v3"
    table_path = module_root / "reports" / "50MM_REPLAY_TASK_SUCCESS_TABLE.csv"
    plan_root = module_root / "reports" / "recording_fast_plans"
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    sources: list[RecordingProfileSource] = []
    with table_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            task_result = str(row.get("task_result", "") or "")
            if str(row.get("evaluation_status", "") or "") != "EVALUATED":
                continue
            if task_result not in SUCCESS_RESULTS:
                continue
            version = str(row.get("version", "") or "")
            plan_path = plan_root / f"{version}_fast_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
            video_path = Path(str(row.get("video_path", "") or ""))
            run_dir = video_path.parent
            video_sha = ""
            manifest_path = run_dir / "viewport_buffer_video_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                video_sha = str(manifest.get("video_sha256", "") or "")
            (
                worker_request_path,
                worker_request_sha,
                accepted_steps_path,
                accepted_steps_sha,
                full_plan_payload,
                full_plan_payload_sha,
            ) = _load_gate_a_full_plan_binding(
                source_version=version,
                run_dir=run_dir,
                exported_document=plan,
            )
            sources.append(
                RecordingProfileSource(
                    source_version=version,
                    task_result=task_result,
                    plan_path=plan_path,
                    gate_a_run_dir=run_dir,
                    plan_sha256=str(plan.get("plan_sha256", "") or ""),
                    plan_file_sha256=_sha256_file(plan_path),
                    video_sha256=video_sha,
                    worker_request_path=worker_request_path,
                    worker_request_sha256=worker_request_sha,
                    accepted_steps_path=accepted_steps_path,
                    accepted_steps_sha256=accepted_steps_sha,
                    full_plan_payload_sha256=full_plan_payload_sha,
                    full_plan_payload=full_plan_payload,
                )
            )
    if not sources:
        raise ValueError("Gate A contains no successful 50 mm recording")
    return tuple(sorted(sources, key=lambda item: item.source_version))


def load_phase_spans(
    alignment_path: str | Path,
    *,
    successful_versions: Iterable[str],
    expected_plan_sha256: Mapping[str, str] | None = None,
) -> tuple[PhaseSpan, ...]:
    """Load Gate-B rows and coalesce their 20 phases into the macro graph."""

    path = Path(alignment_path).resolve()
    allowed = set(successful_versions)
    grouped: dict[tuple[str, MacroStateId, str], list[PhaseWindow]] = {}
    suitable: dict[tuple[str, MacroStateId, str], bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            version = str(row.get("version", row.get("source_version", "")) or "")
            if version not in allowed:
                continue
            expected_sha = str(
                dict(expected_plan_sha256 or {}).get(version, "") or ""
            ).lower()
            row_sha = str(row.get("plan_sha256", "") or "").lower()
            if expected_sha and row_sha != expected_sha:
                raise ValueError(
                    f"Gate-B alignment plan_sha256 mismatch for {version}: "
                    f"expected {expected_sha}, got {row_sha or '<missing>'}"
                )
            phase = str(row.get("phase", row.get("physical_phase", "")) or "").strip().upper()
            if phase not in PHYSICAL_PHASE_TO_MACRO_STATE or phase == "INITIALIZE":
                continue
            status = str(row.get("phase_status", "") or "").upper()
            if status in {"ABSENT", "NOT_PRESENT", "UNMAPPED"}:
                continue
            bounds = _parse_segment_range(
                row.get("fast_segment_range")
                or row.get("segment_range")
                or (
                    f"{row.get('fast_segment_first_index')}:{row.get('fast_segment_last_index')}"
                    if row.get("fast_segment_first_index") not in {None, ""}
                    else ""
                )
            )
            if bounds is None:
                continue
            strategy = str(
                row.get("strategy_profile", row.get("strategy", "PRIMARY_PROFILE"))
                or "PRIMARY_PROFILE"
            ).strip().upper()
            key = (version, PHYSICAL_PHASE_TO_MACRO_STATE[phase], strategy)
            grouped.setdefault(key, []).append(PhaseWindow(phase, *bounds))
            suitable[key] = suitable.get(key, True) and _parse_bool(
                row.get("suitable_for_first_fsm", True)
            )
    spans = [
        PhaseSpan(
            source_version=version,
            state_id=state_id,
            strategy=strategy,
            windows=tuple(
                sorted(windows, key=lambda item: (item.first_segment, item.last_segment))
            ),
            suitable_for_first_fsm=suitable[(version, state_id, strategy)],
        )
        for (version, state_id, strategy), windows in grouped.items()
    ]
    return tuple(sorted(spans, key=lambda item: (item.source_version, item.state_id.value, item.strategy)))


def _phase_for_segment(span: PhaseSpan, segment_index: int) -> str:
    matching = [
        window.physical_phase
        for window in span.windows
        if window.first_segment <= segment_index <= window.last_segment
    ]
    if matching:
        return matching[-1]
    # Functional windows may overlap or contain a short connective command.
    # Label such a command by the closest aligned physical phase without
    # altering its recorded target or order.
    nearest = min(
        span.windows,
        key=lambda window: min(
            abs(segment_index - window.first_segment),
            abs(segment_index - window.last_segment),
        ),
    )
    return nearest.physical_phase


def load_recording_phase_profile(
    source: RecordingProfileSource,
    span: PhaseSpan,
    *,
    owned_segment_range: tuple[int, int] | None = None,
    owner_state_id: MacroStateId | None = None,
    minimum_dispatch_interval_s: float = DEFAULT_DISPATCH_INTERVAL_S,
) -> PhaseMotionProfile:
    """Compile one canonically owned state window without changing commands."""

    if source.source_version != span.source_version:
        raise ValueError("profile source and phase span versions differ")
    if minimum_dispatch_interval_s <= 0.0 or not math.isfinite(minimum_dispatch_interval_s):
        raise ValueError("minimum_dispatch_interval_s must be finite and positive")
    document = json.loads(source.plan_path.read_text(encoding="utf-8-sig"))
    if str(document.get("source_version", "")) != source.source_version:
        raise ValueError("Fast plan source_version mismatch")
    if str(document.get("plan_sha256", "")) != source.plan_sha256:
        raise ValueError("Fast plan identity changed after Gate A")
    segments = list(document.get("segments", []) or [])
    first_segment, last_segment = (
        owned_segment_range
        if owned_segment_range is not None
        else (span.first_segment, span.last_segment)
    )
    if first_segment < 0 or last_segment < first_segment:
        raise ValueError("owned segment range is invalid")
    if last_segment >= len(segments):
        raise ValueError("aligned segment range exceeds the Fast plan")

    servo_state = {name: 0.0 for name in SERVO_JOINT_NAMES}
    wheel_state = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    first_source_time = float(segments[first_segment]["command_start_s"])
    raw_actions: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        segment_index = int(raw.get("decoded_segment_index", index))
        servo_updates = dict(raw.get("servo_target_deg", {}) or {})
        wheel_updates = dict(raw.get("wheel_target_rad_s", {}) or {})
        if index < first_segment:
            for name, value in servo_updates.items():
                servo_state[name] = float(value)
            wheel_state.update({name: float(value) for name, value in wheel_updates.items()})
            continue
        if index > last_segment:
            break
        raw_actions.append(
            {
                "source_time_s": max(0.0, float(raw.get("command_start_s", 0.0)) - first_source_time),
                "order": segment_index * 2,
                "segment_index": segment_index,
                "step_index": int(raw.get("source_step_index", 0) or 0),
                "servo_updates": servo_updates,
                "wheel_updates": wheel_updates,
                "commands": tuple(str(command) for command in raw.get("commands", ()) or ()),
                "event_indices": tuple(int(value) for value in raw.get("source_event_indices", ()) or ()),
                "dispatch_kind": "segment_start",
                "atomic_concurrent": bool(raw.get("concurrent", False)),
            }
        )
        duration = max(0.0, float(raw.get("final_segment_duration_s", 0.0) or 0.0))
        wheel_duration = max(0.0, float(raw.get("wheel_duration_s", 0.0) or 0.0))
        moving_wheel = any(abs(float(value)) > 1.0e-12 for value in wheel_updates.values())
        if moving_wheel and wheel_duration > 0.0 and wheel_duration + 1.0e-9 < duration:
            raw_actions.append(
                {
                    "source_time_s": max(
                        0.0,
                        float(raw.get("command_start_s", 0.0))
                        - first_source_time
                        + wheel_duration,
                    ),
                    "order": segment_index * 2 + 1,
                    "segment_index": segment_index,
                    "step_index": int(raw.get("source_step_index", 0) or 0),
                    "servo_updates": {},
                    "wheel_updates": {name: 0.0 for name in WHEEL_JOINT_NAMES},
                    "commands": (),
                    "event_indices": (),
                    "dispatch_kind": "wheel_channel_completion_stop",
                    "atomic_concurrent": False,
                }
            )

    raw_actions.sort(key=lambda action: (action["source_time_s"], action["order"]))
    frames: list[MotionKeyframe] = []
    last_dispatch = -minimum_dispatch_interval_s
    all_commands: list[str] = []
    step_indices: set[int] = set()
    for sequence_index, action in enumerate(raw_actions):
        servo_state.update(
            {name: float(value) for name, value in action["servo_updates"].items()}
        )
        wheel_state.update(
            {name: float(value) for name, value in action["wheel_updates"].items()}
        )
        servos = _finite_targets(servo_state, SERVO_JOINT_NAMES, label="compiled servo targets")
        wheels = _finite_targets(wheel_state, WHEEL_JOINT_NAMES, label="compiled wheel targets")
        source_time = float(action["source_time_s"])
        dispatch_time = max(source_time, last_dispatch + minimum_dispatch_interval_s)
        phase = _phase_for_segment(span, int(action["segment_index"]))
        frames.append(
            MotionKeyframe(
                time_s=dispatch_time,
                source_time_s=source_time,
                sequence_index=sequence_index,
                source_segment_index=int(action["segment_index"]),
                source_step_index=int(action["step_index"]),
                physical_phase=phase,
                subphase=_subphase_for_phase(phase),
                servo_targets_deg=servos,
                wheel_targets_rad_s=wheels,
                commands=tuple(action["commands"]),
                source_event_indices=tuple(action["event_indices"]),
                dispatch_kind=str(action["dispatch_kind"]),
                atomic_concurrent=bool(action["atomic_concurrent"]),
            )
        )
        last_dispatch = dispatch_time
        all_commands.extend(action["commands"])
        step_indices.add(int(action["step_index"]))

    final_segment = segments[last_segment]
    source_duration = max(
        0.0, float(final_segment.get("command_end_s", first_source_time)) - first_source_time
    )
    nominal_duration = max(source_duration, frames[-1].time_s)
    owned_state = owner_state_id or span.state_id
    profile_id = (
        f"{source.source_version}:{owned_state.value}:{span.strategy}:"
        f"segments-{first_segment}-{last_segment}"
    )
    segment_bindings: tuple[PlaybackSegmentBinding, ...] = ()
    if source.full_plan_payload:
        full_segments = list(source.full_plan_payload.get("segments", ()) or ())
        full_events = list(source.full_plan_payload.get("events", ()) or ())
        bindings: list[PlaybackSegmentBinding] = []
        for segment_index in range(first_segment, last_segment + 1):
            segment_payload = copy.deepcopy(dict(full_segments[segment_index]))
            if int(segment_payload.get("segment_index", -1)) != segment_index:
                raise ValueError("full PlaybackPlan segment order is not canonical")
            event_start = int(segment_payload["event_start_index"])
            event_count = int(segment_payload["event_count"])
            event_payloads = tuple(
                copy.deepcopy(dict(row))
                for row in full_events[event_start : event_start + event_count]
            )
            bindings.append(
                PlaybackSegmentBinding(
                    source_version=source.source_version,
                    source_plan_sha256=source.plan_sha256,
                    source_plan_payload_sha256=source.full_plan_payload_sha256,
                    accepted_steps_sha256=source.accepted_steps_sha256,
                    segment_payload=segment_payload,
                    event_payloads=event_payloads,
                )
            )
        segment_bindings = tuple(bindings)
    return PhaseMotionProfile(
        profile_id=profile_id,
        source_version=source.source_version,
        state_id=owned_state,
        strategy=span.strategy,
        physical_phases=span.physical_phases,
        keyframes=tuple(frames),
        nominal_duration_s=nominal_duration,
        source_plan_sha256=source.plan_sha256,
        source_plan_file_sha256=source.plan_file_sha256,
        gate_a_run_dir=str(source.gate_a_run_dir.resolve()),
        source_segment_range=(first_segment, last_segment),
        source_step_indices=tuple(sorted(step_indices)),
        source_commands=tuple(all_commands),
        dispatch_interval_s=minimum_dispatch_interval_s,
        worker_request_path=(
            ""
            if source.worker_request_path is None
            else str(source.worker_request_path.resolve())
        ),
        worker_request_sha256=source.worker_request_sha256,
        accepted_steps_path=(
            ""
            if source.accepted_steps_path is None
            else str(source.accepted_steps_path.resolve())
        ),
        accepted_steps_sha256=source.accepted_steps_sha256,
        source_plan_payload_sha256=source.full_plan_payload_sha256,
        segment_bindings=segment_bindings,
    )


def build_profile_library(
    project_root: str | Path,
    *,
    alignment_path: str | Path | None = None,
    minimum_dispatch_interval_s: float = DEFAULT_DISPATCH_INTERVAL_S,
) -> MotionProfileLibrary:
    root = Path(project_root).resolve()
    sources = discover_successful_gate_a_sources(root)
    resolved_alignment = (
        Path(alignment_path).resolve()
        if alignment_path is not None
        else root
        / "fsm_50mm_recording_derived_v3"
        / "reports"
        / "50MM_COMMON_PHASE_ALIGNMENT.csv"
    )
    spans = load_phase_spans(
        resolved_alignment,
        successful_versions=(source.source_version for source in sources),
        expected_plan_sha256={
            source.source_version: source.plan_sha256 for source in sources
        },
    )
    source_by_version = {source.source_version: source for source in sources}
    if set(source_by_version) != set(CANONICAL_SEGMENT_OWNERSHIP_RANGES):
        raise ValueError(
            "sealed Gate-A source set differs from the audited canonical "
            "segment-ownership table"
        )
    span_by_source_state: dict[tuple[str, MacroStateId], PhaseSpan] = {}
    for span in spans:
        key = (span.source_version, span.state_id)
        if key in span_by_source_state:
            raise ValueError("Gate-B has multiple strategies for one source/state")
        span_by_source_state[key] = span

    ownership: list[RecordingSegmentOwnership] = []
    profiles_list: list[PhaseMotionProfile] = []
    traversal_states = {
        MacroStateId.S2_FR_TRAVERSE,
        MacroStateId.S3_FL_TRAVERSE,
        MacroStateId.S6_RR_TRAVERSE,
        MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
    }
    for source in sources:
        for owner_state, first_segment, last_segment in (
            CANONICAL_SEGMENT_OWNERSHIP_RANGES[source.source_version]
        ):
            phase_source_state = CANONICAL_PHASE_SOURCE_STATE.get(
                (source.source_version, owner_state), owner_state
            )
            span = span_by_source_state.get(
                (source.source_version, phase_source_state)
            )
            if span is None or not span.suitable_for_first_fsm:
                raise ValueError(
                    f"canonical owner {source.source_version}/{owner_state.value} "
                    "has no suitable Gate-B phase evidence"
                )
            basis = (
                "Gate-A current-batch cursor audit retains every action through "
                "phase-local lift/cross/post-cross-TOP completion"
                if owner_state in traversal_states
                else "Gate-B command order plus Gate-A causal response audit; "
                "connective actions assigned once without replay"
            )
            ownership.append(
                RecordingSegmentOwnership(
                    source_version=source.source_version,
                    state_id=owner_state,
                    phase_source_state_id=phase_source_state,
                    first_segment=first_segment,
                    last_segment=last_segment,
                    source_plan_sha256=source.plan_sha256,
                    evidence_basis=basis,
                )
            )
            profiles_list.append(
                load_recording_phase_profile(
                    source,
                    span,
                    owned_segment_range=(first_segment, last_segment),
                    owner_state_id=owner_state,
                    minimum_dispatch_interval_s=minimum_dispatch_interval_s,
                )
            )
    profiles = tuple(profiles_list)
    if not profiles:
        raise ValueError("Gate-B alignment produced no suitable Gate-C profiles")
    return MotionProfileLibrary(
        profiles=profiles,
        successful_sources=sources,
        alignment_path=str(resolved_alignment),
        segment_ownership=tuple(ownership),
    )
