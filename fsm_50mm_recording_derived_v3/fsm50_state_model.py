"""Typed, serializable state-table model for the recording-derived 50 mm FSM.

This module defines data and timing semantics only.  It does not claim that a
candidate recording fragment has been physically validated, and it has no
Isaac or controller dependency.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .com_transfer_primitives import COMTransferMethod, GuardDecision, Leg


class ProvenanceStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    PENDING_REPLAY = "PENDING_REPLAY"
    PHYSICALLY_VERIFIED = "PHYSICALLY_VERIFIED"
    REJECTED = "REJECTED"


class PrimaryDiagonal(str, Enum):
    NONE = "NONE"
    FL_RR = "FL_RR"
    FR_RL = "FR_RL"
    BALANCED = "BALANCED"
    DYNAMIC = "DYNAMIC"


class ServoTrajectoryType(str, Enum):
    HOLD = "HOLD"
    LINEAR = "LINEAR"
    RECORDING_RAMP = "RECORDING_RAMP"
    CUBIC = "CUBIC"


class WheelTrajectoryType(str, Enum):
    HOLD = "HOLD"
    STEP = "STEP"
    LINEAR_RAMP = "LINEAR_RAMP"
    RECORDING_PROFILE = "RECORDING_PROFILE"


@dataclass(frozen=True)
class GuardSpec:
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_value(cls, value: GuardSpec | str | Mapping[str, Any]) -> GuardSpec:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            decoded = _decode_json_cell(value)
            if isinstance(decoded, Mapping):
                value = decoded
            else:
                return cls(str(decoded or ""))
        if not isinstance(value, Mapping):
            raise TypeError("guard must be a string, mapping, or GuardSpec")
        return cls(
            kind=str(value.get("kind", value.get("name", ""))),
            parameters=dict(value.get("parameters", value.get("params", {})) or {}),
            description=str(value.get("description", "")),
        )


@dataclass(frozen=True)
class RetryPolicy:
    maximum_retries: int = 0
    compensation_scale: float = 0.0
    abort_on_exhaustion: bool = True
    retry_state_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.maximum_retries < 0:
            raise ValueError("maximum_retries must be non-negative")
        if not math.isfinite(float(self.compensation_scale)) or self.compensation_scale < 0.0:
            raise ValueError("compensation_scale must be finite and non-negative")

    @classmethod
    def from_value(
        cls, value: RetryPolicy | Mapping[str, Any] | int | str
    ) -> RetryPolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(maximum_retries=value)
        if isinstance(value, str):
            decoded = _decode_json_cell(value)
            if isinstance(decoded, int):
                return cls(maximum_retries=decoded)
            value = decoded
        if not isinstance(value, Mapping):
            raise TypeError("retry_policy must be an integer, mapping, or RetryPolicy")
        return cls(
            maximum_retries=int(value.get("maximum_retries", value.get("max_retries", 0))),
            compensation_scale=float(value.get("compensation_scale", 0.0)),
            abort_on_exhaustion=_as_bool(value.get("abort_on_exhaustion", True)),
            retry_state_id=str(value.get("retry_state_id", "")),
            notes=str(value.get("notes", "")),
        )


REQUIRED_STATE_FIELDS: tuple[str, ...] = (
    "state_id",
    "state_name",
    "description",
    "source_recording_version",
    "source_step_indices",
    "source_event_indices",
    "source_telemetry_time_range",
    "active_leg",
    "swing_leg",
    "impulse_leg",
    "support_legs",
    "primary_diagonal",
    "secondary_support",
    "target_com_direction",
    "com_transfer_method",
    "servo_start_target",
    "servo_end_target",
    "servo_trajectory_type",
    "wheel_start_target",
    "wheel_end_target",
    "wheel_trajectory_type",
    "atomic_concurrent",
    "min_duration",
    "settle_duration",
    "max_duration",
    "entry_guard",
    "progress_guard",
    "exit_guard",
    "abort_guard",
    "hysteresis",
    "required_consecutive_samples",
    "retry_policy",
    "expected_com_displacement",
    "expected_com_velocity_direction",
    "expected_load_transfer",
    "expected_contact_change",
    "expected_clearance",
    "expected_body_response",
    "allowed_contact_drift",
    "allowed_roll_pitch",
    "allowed_angular_velocity",
    "joint_limit_margin",
    "notes",
)

# These fields are required by the controller/provenance audit, but remain
# optional when loading the older standalone CSV/JSON state-table fixtures.
# The controller YAML compiler below always materializes them.
EXTENDED_STATE_FIELDS: tuple[str, ...] = (
    "phase",
    "guard",
    "command_profile",
    "target_com_leg",
    "source_fast_segment_indices",
    "source_run_directory",
    "source_sha256",
    "selection_reason",
    "compatibility_result",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "on"}:
        return True
    if normalized in {"false", "no", "0", "off", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _optional_leg(value: Any) -> Leg | None:
    if value is None or str(value).strip().upper() in {"", "NONE", "NULL"}:
        return None
    return value if isinstance(value, Leg) else Leg(str(value).strip().upper())


def _legs(value: Any) -> tuple[Leg, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        decoded = _decode_json_cell(value)
        if isinstance(decoded, str):
            raw: Iterable[Any] = [item.strip() for item in decoded.split(",") if item.strip()]
        else:
            raw = decoded
    else:
        raw = value
    result: list[Leg] = []
    for item in raw:
        parsed = _optional_leg(item)
        if parsed is not None:
            result.append(parsed)
    return tuple(result)


def _float_pair(value: Any, *, optional: bool, label: str) -> tuple[float, float] | None:
    if value is None or value == "":
        if optional:
            return None
        raise ValueError(f"{label} is required")
    if isinstance(value, str):
        value = _decode_json_cell(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    pair = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in pair):
        raise ValueError(f"{label} must contain finite values")
    return pair


def _indices(value: Any) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = _decode_json_cell(value)
    return tuple(int(item) for item in value)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = _decode_json_cell(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _float_mapping(value: Any, *, label: str) -> dict[str, float]:
    return {key: float(item) for key, item in _mapping(value, label=label).items()}


def _decode_json_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        if "," in stripped:
            return [item.strip() for item in stripped.split(",")]
        return stripped


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(_plain(key)): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FSM50State:
    state_id: str
    state_name: str
    description: str
    source_recording_version: str
    source_step_indices: tuple[int, ...]
    source_event_indices: tuple[int, ...]
    source_telemetry_time_range: tuple[float, float] | None
    active_leg: Leg | None
    swing_leg: Leg | None
    impulse_leg: Leg | None
    support_legs: tuple[Leg, ...]
    primary_diagonal: PrimaryDiagonal
    secondary_support: tuple[Leg, ...]
    target_com_direction: tuple[float, float] | None
    com_transfer_method: COMTransferMethod
    servo_start_target: Mapping[str, float]
    servo_end_target: Mapping[str, float]
    servo_trajectory_type: ServoTrajectoryType
    wheel_start_target: Mapping[str, float]
    wheel_end_target: Mapping[str, float]
    wheel_trajectory_type: WheelTrajectoryType
    atomic_concurrent: bool
    min_duration: float
    settle_duration: float
    max_duration: float
    entry_guard: GuardSpec
    progress_guard: GuardSpec
    exit_guard: GuardSpec
    abort_guard: GuardSpec
    hysteresis: Mapping[str, float]
    required_consecutive_samples: int
    retry_policy: RetryPolicy
    expected_com_displacement: tuple[float, float] | None
    expected_com_velocity_direction: tuple[float, float] | None
    expected_load_transfer: Mapping[str, float]
    expected_contact_change: Mapping[str, str]
    expected_clearance: Mapping[str, float]
    expected_body_response: Mapping[str, Any]
    allowed_contact_drift: float
    allowed_roll_pitch: tuple[float, float]
    allowed_angular_velocity: float
    joint_limit_margin: float
    notes: str
    provenance_status: ProvenanceStatus = ProvenanceStatus.PENDING_REPLAY
    provenance_note: str = "recording-derived candidate; physical replay pending"
    phase: str = ""
    guard: str = ""
    command_profile: str = ""
    target_com_leg: Leg | None = None
    source_fast_segment_indices: tuple[int, ...] = ()
    source_run_directory: str = ""
    source_sha256: str = ""
    selection_reason: str = ""
    compatibility_result: str = "PENDING_REPLAY"

    def __post_init__(self) -> None:
        if not self.state_id.strip() or not self.state_name.strip():
            raise ValueError("state_id and state_name are required")
        for name, value in (
            ("min_duration", self.min_duration),
            ("settle_duration", self.settle_duration),
            ("max_duration", self.max_duration),
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_duration > self.max_duration:
            raise ValueError("min_duration cannot exceed max_duration")
        if self.settle_duration > self.max_duration:
            raise ValueError("settle_duration cannot exceed max_duration")
        if self.required_consecutive_samples < 1:
            raise ValueError("required_consecutive_samples must be positive")
        if self.source_telemetry_time_range is not None:
            start_s, end_s = self.source_telemetry_time_range
            if start_s < 0.0 or end_s < start_s:
                raise ValueError("source_telemetry_time_range must be ordered and non-negative")
        if len(set(self.support_legs)) != len(self.support_legs):
            raise ValueError("support_legs contains duplicates")
        if self.swing_leg is not None and self.swing_leg in self.support_legs:
            raise ValueError("swing_leg cannot also be a support leg")
        if self.com_transfer_method != COMTransferMethod.NONE:
            if self.target_com_direction is None or math.hypot(*self.target_com_direction) <= 1.0e-12:
                raise ValueError("COM transfer states require a non-zero target_com_direction")
        if self.atomic_concurrent:
            has_servo = bool(self.servo_end_target)
            has_wheel = any(abs(float(value)) > 1.0e-12 for value in self.wheel_end_target.values())
            if not has_servo or not has_wheel:
                raise ValueError("atomic_concurrent requires both servo and non-zero wheel targets")
        for name, value in (
            ("allowed_contact_drift", self.allowed_contact_drift),
            ("allowed_angular_velocity", self.allowed_angular_velocity),
            ("joint_limit_margin", self.joint_limit_margin),
        ):
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(not math.isfinite(float(value)) or value < 0.0 for value in self.allowed_roll_pitch):
            raise ValueError("allowed_roll_pitch values must be finite and non-negative")
        for mapping_name, values in (
            ("servo_start_target", self.servo_start_target),
            ("servo_end_target", self.servo_end_target),
            ("wheel_start_target", self.wheel_start_target),
            ("wheel_end_target", self.wheel_end_target),
        ):
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"{mapping_name} contains a non-finite target")
        for mapping_name, values in (
            ("hysteresis", self.hysteresis),
            ("expected_load_transfer", self.expected_load_transfer),
            ("expected_clearance", self.expected_clearance),
        ):
            if any(not math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"{mapping_name} contains a non-finite value")
        if any(float(value) < 0.0 for value in self.hysteresis.values()):
            raise ValueError("hysteresis values must be non-negative")
        if self.provenance_status == ProvenanceStatus.PHYSICALLY_VERIFIED:
            missing = []
            if not self.source_event_indices:
                missing.append("source_event_indices")
            if self.source_telemetry_time_range is None:
                missing.append("source_telemetry_time_range")
            if not self.source_fast_segment_indices:
                missing.append("source_fast_segment_indices")
            if not self.source_run_directory:
                missing.append("source_run_directory")
            if not self.source_sha256:
                missing.append("source_sha256")
            if not self.selection_reason:
                missing.append("selection_reason")
            if str(self.compatibility_result).upper() not in {
                "COMPATIBLE",
                "PHYSICALLY_VERIFIED",
                "PASS",
            }:
                missing.append("compatibility_result")
            if missing:
                raise ValueError(
                    "PHYSICALLY_VERIFIED provenance requires " + ", ".join(missing)
                )

    @property
    def pending_physical_replay(self) -> bool:
        return self.provenance_status in {
            ProvenanceStatus.CANDIDATE,
            ProvenanceStatus.PENDING_REPLAY,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {key: _plain(getattr(self, key)) for key in REQUIRED_STATE_FIELDS} | {
            "provenance_status": self.provenance_status.value,
            "provenance_note": self.provenance_note,
            **{key: _plain(getattr(self, key)) for key in EXTENDED_STATE_FIELDS},
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, strict: bool = True) -> FSM50State:
        missing = [key for key in REQUIRED_STATE_FIELDS if key not in values]
        if strict and missing:
            raise ValueError(f"state row is missing required fields: {', '.join(missing)}")

        def required(name: str, default: Any = None) -> Any:
            return values[name] if name in values else default

        source_range = _float_pair(
            required("source_telemetry_time_range"), optional=True, label="source_telemetry_time_range"
        )
        allowed_roll_pitch = _float_pair(
            required("allowed_roll_pitch", (0.0, 0.0)), optional=False, label="allowed_roll_pitch"
        )
        assert allowed_roll_pitch is not None
        primary_raw = required("primary_diagonal", "NONE")
        transfer_raw = required("com_transfer_method", "NONE")
        servo_type_raw = required("servo_trajectory_type", "HOLD")
        wheel_type_raw = required("wheel_trajectory_type", "HOLD")
        provenance_raw = required("provenance_status", "PENDING_REPLAY")
        return cls(
            state_id=str(required("state_id", "")),
            state_name=str(required("state_name", "")),
            description=str(required("description", "")),
            source_recording_version=str(required("source_recording_version", "")),
            source_step_indices=_indices(required("source_step_indices", ())),
            source_event_indices=_indices(required("source_event_indices", ())),
            source_telemetry_time_range=source_range,
            active_leg=_optional_leg(required("active_leg")),
            swing_leg=_optional_leg(required("swing_leg")),
            impulse_leg=_optional_leg(required("impulse_leg")),
            support_legs=_legs(required("support_legs", ())),
            primary_diagonal=primary_raw if isinstance(primary_raw, PrimaryDiagonal) else PrimaryDiagonal(str(primary_raw).upper()),
            secondary_support=_legs(required("secondary_support", ())),
            target_com_direction=_float_pair(required("target_com_direction"), optional=True, label="target_com_direction"),
            com_transfer_method=transfer_raw if isinstance(transfer_raw, COMTransferMethod) else COMTransferMethod(str(transfer_raw).upper()),
            servo_start_target=_float_mapping(required("servo_start_target", {}), label="servo_start_target"),
            servo_end_target=_float_mapping(required("servo_end_target", {}), label="servo_end_target"),
            servo_trajectory_type=servo_type_raw if isinstance(servo_type_raw, ServoTrajectoryType) else ServoTrajectoryType(str(servo_type_raw).upper()),
            wheel_start_target=_float_mapping(required("wheel_start_target", {}), label="wheel_start_target"),
            wheel_end_target=_float_mapping(required("wheel_end_target", {}), label="wheel_end_target"),
            wheel_trajectory_type=wheel_type_raw if isinstance(wheel_type_raw, WheelTrajectoryType) else WheelTrajectoryType(str(wheel_type_raw).upper()),
            atomic_concurrent=_as_bool(required("atomic_concurrent", False)),
            min_duration=float(required("min_duration", 0.0)),
            settle_duration=float(required("settle_duration", 0.0)),
            max_duration=float(required("max_duration", 0.0)),
            entry_guard=GuardSpec.from_value(required("entry_guard", "UNSPECIFIED")),
            progress_guard=GuardSpec.from_value(required("progress_guard", "UNSPECIFIED")),
            exit_guard=GuardSpec.from_value(required("exit_guard", "UNSPECIFIED")),
            abort_guard=GuardSpec.from_value(required("abort_guard", "UNSPECIFIED")),
            hysteresis=_float_mapping(required("hysteresis", {}), label="hysteresis"),
            required_consecutive_samples=int(required("required_consecutive_samples", 1)),
            retry_policy=RetryPolicy.from_value(required("retry_policy", 0)),
            expected_com_displacement=_float_pair(required("expected_com_displacement"), optional=True, label="expected_com_displacement"),
            expected_com_velocity_direction=_float_pair(required("expected_com_velocity_direction"), optional=True, label="expected_com_velocity_direction"),
            expected_load_transfer=_float_mapping(required("expected_load_transfer", {}), label="expected_load_transfer"),
            expected_contact_change={key: str(value) for key, value in _mapping(required("expected_contact_change", {}), label="expected_contact_change").items()},
            expected_clearance=_float_mapping(required("expected_clearance", {}), label="expected_clearance"),
            expected_body_response=_mapping(required("expected_body_response", {}), label="expected_body_response"),
            allowed_contact_drift=float(required("allowed_contact_drift", 0.0)),
            allowed_roll_pitch=allowed_roll_pitch,
            allowed_angular_velocity=float(required("allowed_angular_velocity", 0.0)),
            joint_limit_margin=float(required("joint_limit_margin", 0.0)),
            notes=str(required("notes", "")),
            provenance_status=provenance_raw if isinstance(provenance_raw, ProvenanceStatus) else ProvenanceStatus(str(provenance_raw).upper()),
            provenance_note=str(required("provenance_note", "recording-derived candidate; physical replay pending")),
            phase=str(required("phase", "")),
            guard=str(required("guard", "")),
            command_profile=str(required("command_profile", "")),
            target_com_leg=_optional_leg(required("target_com_leg")),
            source_fast_segment_indices=_indices(
                required("source_fast_segment_indices", ())
            ),
            source_run_directory=str(required("source_run_directory", "")),
            source_sha256=str(required("source_sha256", "")),
            selection_reason=str(required("selection_reason", "")),
            compatibility_result=str(
                required("compatibility_result", "PENDING_REPLAY")
            ),
        )


def _expand_controller_rows(
    document: Mapping[str, Any],
    rows: Sequence[Any],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Compile controller YAML rows into executable state-table rows.

    Older exported state tables already contain explicit start/end targets and
    have no ``command_profiles`` section.  Controller YAML instead names one
    endpoint profile per state.  Compilation is deliberately deterministic:
    each state's start target is the preceding happy-path endpoint and its end
    target is the named profile.  The runtime executor still re-latches the
    actually applied command on entry, so a restore/retry never trusts this
    static continuity hint over live state.
    """

    defaults_raw = document.get("state_defaults", {})
    defaults = dict(defaults_raw or {}) if isinstance(defaults_raw, Mapping) else {}
    profiles_raw = document.get("command_profiles", {})
    if profiles_raw is None:
        profiles_raw = {}
    if not isinstance(profiles_raw, Mapping):
        raise TypeError("command_profiles must be a mapping")
    profiles = {str(name): value for name, value in profiles_raw.items()}
    metadata = dict(document.get("metadata", {}) or {})
    global_source_version = str(metadata.get("source_recording_version", ""))
    global_source_sha256 = str(metadata.get("source_recording_sha256", ""))

    previous_servos: dict[str, float] = {}
    previous_wheels: dict[str, float] = {}
    expanded: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"state row {index} must be a mapping")
        row = {**defaults, **dict(raw)}
        profile_name = str(row.get("command_profile", "") or "")
        if profile_name:
            if profile_name not in profiles:
                raise ValueError(
                    f"state {row.get('state_id', index)} references unknown "
                    f"command profile {profile_name!r}"
                )
            profile_raw = profiles[profile_name]
            if not isinstance(profile_raw, Mapping):
                raise TypeError(f"command profile {profile_name!r} must be a mapping")
            profile = dict(profile_raw)
            servos = _float_mapping(profile.get("servos", {}), label=f"{profile_name}.servos")
            wheels = _float_mapping(profile.get("wheels", {}), label=f"{profile_name}.wheels")
            if not dict(row.get("servo_start_target", {}) or {}):
                row["servo_start_target"] = dict(previous_servos or servos)
            if not dict(row.get("servo_end_target", {}) or {}):
                row["servo_end_target"] = dict(servos)
            if not dict(row.get("wheel_start_target", {}) or {}):
                row["wheel_start_target"] = dict(previous_wheels or wheels)
            if not dict(row.get("wheel_end_target", {}) or {}):
                row["wheel_end_target"] = dict(wheels)
            if not tuple(row.get("source_step_indices", ()) or ()):
                row["source_step_indices"] = list(profile.get("source_steps", ()) or ())
            previous_servos = dict(row["servo_end_target"])
            previous_wheels = dict(row["wheel_end_target"])
        elif profiles and strict:
            raise ValueError(
                f"state {row.get('state_id', index)} has no command_profile"
            )

        source_version = str(row.get("source_recording_version", "") or "")
        row.setdefault("phase", "")
        row.setdefault("guard", "")
        row.setdefault("command_profile", profile_name)
        row.setdefault("target_com_leg", "NONE")
        row.setdefault("source_fast_segment_indices", [])
        row.setdefault("source_run_directory", "")
        row.setdefault(
            "source_sha256",
            global_source_sha256
            if source_version and source_version == global_source_version
            else "",
        )
        row.setdefault(
            "selection_reason",
            str(row.get("provenance_note", "") or "selection pending live replay"),
        )
        row.setdefault("compatibility_result", "PENDING_REPLAY")
        expanded.append(row)
    return expanded


@dataclass(frozen=True)
class FSM50StateTable:
    states: tuple[FSM50State, ...]
    schema_version: str = "fsm50-state-table.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = [state.state_id for state in self.states]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("state_id values must be unique")

    def get(self, state_id: str) -> FSM50State:
        for state in self.states:
            if state.state_id == state_id:
                return state
        raise KeyError(state_id)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": _plain(self.metadata),
            "states": [state.to_mapping() for state in self.states],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], *, strict: bool = True) -> FSM50StateTable:
        rows = values.get("states")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("state table must contain a states sequence")
        expanded_rows = _expand_controller_rows(values, rows, strict=strict)
        return cls(
            states=tuple(
                FSM50State.from_mapping(row, strict=strict) for row in expanded_rows
            ),
            schema_version=str(values.get("schema_version", "fsm50-state-table.v1")),
            metadata={
                **dict(values.get("metadata", {}) or {}),
                "actuators": _plain(dict(values.get("actuators", {}) or {})),
                "thresholds": _plain(dict(values.get("thresholds", {}) or {})),
                "command_profiles": _plain(
                    dict(values.get("command_profiles", {}) or {})
                ),
            },
        )

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> FSM50StateTable:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".csv":
            with source.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            return cls.from_mapping({"states": rows}, strict=strict)
        text = source.read_text(encoding="utf-8")
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                values = json.loads(text)
            else:
                values = yaml.safe_load(text)
        else:
            values = json.loads(text)
        if not isinstance(values, Mapping):
            raise ValueError("state-table document must be a mapping")
        return cls.from_mapping(values, strict=strict)

    def export(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix == ".csv":
            columns = REQUIRED_STATE_FIELDS + (
                "provenance_status",
                "provenance_note",
            ) + EXTENDED_STATE_FIELDS
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for state in self.states:
                    row = state.to_mapping()
                    writer.writerow(
                        {
                            key: value
                            if isinstance(value, (str, int, float)) or value is None
                            else json.dumps(value, ensure_ascii=False, sort_keys=True)
                            for key, value in row.items()
                        }
                    )
        elif suffix in {".yaml", ".yml"}:
            payload = self.to_mapping()
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        else:
            destination.write_text(json.dumps(self.to_mapping(), indent=2, ensure_ascii=False), encoding="utf-8")
        return destination


@dataclass
class TransitionDwellTracker:
    """Apply minimum dwell, stable dwell, consecutive samples, and timeout."""

    state: FSM50State
    entered_at_s: float
    stable_started_at_s: float | None = None
    consecutive_samples: int = 0

    def reset(self, entered_at_s: float) -> None:
        self.entered_at_s = float(entered_at_s)
        self.stable_started_at_s = None
        self.consecutive_samples = 0

    def update(self, *, time_s: float, exit_condition: bool) -> GuardDecision:
        elapsed = max(0.0, float(time_s) - self.entered_at_s)
        if elapsed > self.state.max_duration:
            return GuardDecision(False, True, "state maximum timeout exceeded", {"elapsed_s": elapsed})
        if exit_condition and elapsed >= self.state.min_duration:
            if self.stable_started_at_s is None:
                self.stable_started_at_s = float(time_s)
            self.consecutive_samples += 1
        else:
            self.stable_started_at_s = None
            self.consecutive_samples = 0
        stable_dwell = (
            0.0
            if self.stable_started_at_s is None
            else max(0.0, float(time_s) - self.stable_started_at_s)
        )
        satisfied = bool(
            self.consecutive_samples >= self.state.required_consecutive_samples
            and stable_dwell >= self.state.settle_duration
        )
        unmet: list[str] = []
        if elapsed < self.state.min_duration:
            unmet.append("minimum dwell not reached")
        if self.consecutive_samples < self.state.required_consecutive_samples:
            unmet.append("consecutive-sample requirement not reached")
        if stable_dwell < self.state.settle_duration:
            unmet.append("stable dwell not reached")
        return GuardDecision(
            satisfied,
            reason="" if satisfied else "; ".join(unmet),
            metrics={
                "elapsed_s": elapsed,
                "stable_dwell_s": stable_dwell,
                "consecutive_samples": self.consecutive_samples,
            },
            unmet=tuple(unmet),
        )


def make_pending_candidate_state(
    *,
    state_id: str,
    state_name: str,
    source_recording_version: str,
    source_step_indices: Sequence[int],
    description: str = "",
    active_leg: Leg | None = None,
    support_legs: Sequence[Leg] = (),
    com_transfer_method: COMTransferMethod = COMTransferMethod.NONE,
    target_com_direction: tuple[float, float] | None = None,
) -> FSM50State:
    """Create an explicitly non-verified scaffold without invented thresholds."""

    return FSM50State(
        state_id=state_id,
        state_name=state_name,
        description=description,
        source_recording_version=source_recording_version,
        source_step_indices=tuple(int(value) for value in source_step_indices),
        source_event_indices=(),
        source_telemetry_time_range=None,
        active_leg=active_leg,
        swing_leg=active_leg,
        impulse_leg=None,
        support_legs=tuple(support_legs),
        primary_diagonal=PrimaryDiagonal.DYNAMIC,
        secondary_support=(),
        target_com_direction=target_com_direction,
        com_transfer_method=com_transfer_method,
        servo_start_target={},
        servo_end_target={},
        servo_trajectory_type=ServoTrajectoryType.RECORDING_RAMP,
        wheel_start_target={},
        wheel_end_target={},
        wheel_trajectory_type=WheelTrajectoryType.RECORDING_PROFILE,
        atomic_concurrent=False,
        min_duration=0.0,
        settle_duration=0.0,
        max_duration=1.0,
        entry_guard=GuardSpec("PENDING_REPLAY_THRESHOLDS"),
        progress_guard=GuardSpec("PENDING_REPLAY_THRESHOLDS"),
        exit_guard=GuardSpec("PENDING_REPLAY_THRESHOLDS"),
        abort_guard=GuardSpec("FAIL_SAFE_STOP"),
        hysteresis={},
        required_consecutive_samples=1,
        retry_policy=RetryPolicy(),
        expected_com_displacement=None,
        expected_com_velocity_direction=None,
        expected_load_transfer={},
        expected_contact_change={},
        expected_clearance={},
        expected_body_response={},
        allowed_contact_drift=0.0,
        allowed_roll_pitch=(0.0, 0.0),
        allowed_angular_velocity=0.0,
        joint_limit_margin=0.0,
        notes="No physical thresholds are inferred from recording endpoints.",
        provenance_status=ProvenanceStatus.PENDING_REPLAY,
        provenance_note="candidate provenance only; clean physical replay required",
    )


__all__ = [
    "FSM50State",
    "FSM50StateTable",
    "GuardSpec",
    "PrimaryDiagonal",
    "ProvenanceStatus",
    "REQUIRED_STATE_FIELDS",
    "RetryPolicy",
    "ServoTrajectoryType",
    "TransitionDwellTracker",
    "WheelTrajectoryType",
    "make_pending_candidate_state",
]
