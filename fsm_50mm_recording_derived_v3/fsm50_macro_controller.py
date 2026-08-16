"""Feedback controller for the small recording-derived 50 mm Macro FSM.

An outer-cycle permit and a SHA-bound measured-completion token select the next
recorded command primitive.  Nominal profile time never advances the source
cursor or declares a physical phase complete.  Graph completion still comes
from live geometry/contact/body events, with bounded hold/retry and fail-closed
safety behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from completion_aware_segment import SegmentCompletionSpec, SegmentDecision

from .fsm50_macro_state_model import (
    MacroFSMGraph,
    MacroGuardKind,
    MacroStateId,
    MacroStateSpec,
    MacroSubphase,
    build_default_macro_graph,
)
from .fsm50_motion_profiles import (
    DEFAULT_PRIMARY_VERSION,
    MotionKeyframe,
    MotionProfileLibrary,
    PlaybackSegmentBinding,
    PhaseMotionProfile,
    build_profile_library,
)


LEGS = ("FL", "FR", "RL", "RR")


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_exact_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be an explicit bool")
    return value


def _required_exact_optional_bool(values: Mapping[str, Any], key: str) -> bool | None:
    value = values[key]
    if value is not None and type(value) is not bool:
        raise ValueError(f"{key} must be bool or None")
    return value


def _strict_vector(value: Any, length: int, *, label: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        order = ("x", "y", "z", "w")[:length]
        missing = [name for name in order if name not in value]
        if missing:
            raise ValueError(f"{label} is missing components: {missing}")
        raw = tuple(value[name] for name in order)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != length:
            raise ValueError(f"{label} must have exactly {length} elements")
        raw = tuple(value)
    else:
        raise ValueError(f"{label} must be a {length}-element vector")
    if any(isinstance(item, bool) for item in raw):
        raise ValueError(f"{label} contains a boolean instead of a number")
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains a non-numeric value") from exc
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} contains a non-finite value")
    return values


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _target_map(
    values: Any, names: Sequence[str], *, label: str, require_complete: bool = False
) -> dict[str, float]:
    raw = dict(values or {}) if isinstance(values, Mapping) else {}
    missing = sorted(set(names) - set(raw))
    unknown = sorted(set(raw) - set(names))
    if require_complete and (missing or unknown):
        raise ValueError(
            f"{label} must contain the exact canonical actuator set; "
            f"missing={missing} unknown={unknown}"
        )
    if unknown:
        raise ValueError(f"{label} contains unknown actuators: {unknown}")
    try:
        result = {
            name: _finite_scalar(raw[name], label=f"{label}.{name}")
            for name in names
        }
    except KeyError as exc:
        raise ValueError(f"{label} is missing {exc.args[0]}") from exc
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError(f"{label} contains non-finite values")
    return result


def _leg_scalar_map(
    values: Any,
    *,
    label: str,
    require_all: bool = False,
    allow_none: bool = True,
) -> dict[str, float | None]:
    raw = dict(values or {}) if isinstance(values, Mapping) else {}
    missing = sorted(set(LEGS) - set(raw))
    unknown = sorted(set(raw) - set(LEGS))
    if require_all and (missing or unknown):
        raise ValueError(
            f"{label} must contain the exact four-leg set; missing={missing} unknown={unknown}"
        )
    result: dict[str, float | None] = {}
    for leg in LEGS:
        value = raw.get(leg, raw.get(leg.lower()))
        if value is None:
            if not allow_none:
                raise ValueError(f"{label}.{leg} must be finite")
            result[leg] = None
            continue
        result[leg] = _finite_scalar(value, label=f"{label}.{leg}")
    return result


def _leg_vector_map(values: Any) -> dict[str, tuple[float, float, float]]:
    raw = dict(values or {}) if isinstance(values, Mapping) else {}
    missing = sorted(set(LEGS) - set(raw))
    unknown = sorted(set(raw) - set(LEGS))
    if missing or unknown:
        raise ValueError(
            "wheel_center_w_m must contain the exact four-leg set; "
            f"missing={missing} unknown={unknown}"
        )
    return {
        leg: _strict_vector(raw[leg], 3, label=f"wheel_center_w_m.{leg}")
        for leg in LEGS
    }


def _leg_class_map(values: Any) -> dict[str, str]:
    raw = dict(values or {}) if isinstance(values, Mapping) else {}
    missing = sorted(set(LEGS) - set(raw))
    unknown = sorted(set(raw) - set(LEGS))
    if missing or unknown:
        raise ValueError(
            "wheel_contact_classes must contain the exact four-leg set; "
            f"missing={missing} unknown={unknown}"
        )
    result: dict[str, str] = {}
    for leg in LEGS:
        value = raw[leg]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"wheel_contact_classes.{leg} must be a non-empty string")
        result[leg] = value.strip().upper()
    return result


@dataclass(frozen=True)
class MacroObservation:
    """Deployment-available subset consumed by the Macro FSM.

    Exact COM is optional.  When absent, the controller labels body-position
    displacement as a proxy; it never reports that proxy as measured COM.
    """

    robot_state_finite: bool
    base_position_m: tuple[float, float, float]
    base_roll_rad: float
    base_pitch_rad: float
    base_angular_velocity_rad_s: tuple[float, float, float]
    com_position_m: tuple[float, float, float] | None
    servo_targets_deg: Mapping[str, float]
    wheel_targets_rad_s: Mapping[str, float]
    wheel_center_w_m: Mapping[str, tuple[float, float, float]]
    wheel_contact_classes: Mapping[str, str]
    wheel_contact_load_n: Mapping[str, float | None]
    wheel_front_face_clearance_m: Mapping[str, float | None]
    wheel_top_clearance_m: Mapping[str, float | None]
    obstacle_front_face_x_m: float
    obstacle_top_z_m: float
    actuator_targets_applied: bool
    dispatch_error: str
    robot_fell: bool
    body_stuck: bool | None
    dangerous_collision: bool | None
    severe_penetration: bool | None
    joint_limit_violation: bool
    unsafe_joint_target: bool
    active_leg_trapped: bool | None
    wheel_drive_up_without_required_lift: bool
    body_crossed_front_face: bool
    final_recoverable: bool
    posture_complete: bool

    def __post_init__(self) -> None:
        for label, value in (
            ("robot_state_finite", self.robot_state_finite),
            ("actuator_targets_applied", self.actuator_targets_applied),
            ("robot_fell", self.robot_fell),
            ("joint_limit_violation", self.joint_limit_violation),
            ("unsafe_joint_target", self.unsafe_joint_target),
            (
                "wheel_drive_up_without_required_lift",
                self.wheel_drive_up_without_required_lift,
            ),
            ("body_crossed_front_face", self.body_crossed_front_face),
            ("final_recoverable", self.final_recoverable),
            ("posture_complete", self.posture_complete),
        ):
            if type(value) is not bool:
                raise ValueError(f"{label} must be a bool")
        for label, value in (
            ("body_stuck", self.body_stuck),
            ("dangerous_collision", self.dangerous_collision),
            ("severe_penetration", self.severe_penetration),
            ("active_leg_trapped", self.active_leg_trapped),
        ):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{label} must be bool or None")
        if not isinstance(self.dispatch_error, str):
            raise ValueError("dispatch_error must be a string")

        base_position = _strict_vector(
            self.base_position_m, 3, label="base_position_m"
        )
        angular_velocity = _strict_vector(
            self.base_angular_velocity_rad_s,
            3,
            label="base_angular_velocity_rad_s",
        )
        com_position = (
            None
            if self.com_position_m is None
            else _strict_vector(self.com_position_m, 3, label="com_position_m")
        )
        servo_targets = _target_map(
            self.servo_targets_deg,
            SERVO_JOINT_NAMES,
            label="servo_targets_deg",
            require_complete=True,
        )
        wheel_targets = _target_map(
            self.wheel_targets_rad_s,
            WHEEL_JOINT_NAMES,
            label="wheel_targets_rad_s",
            require_complete=True,
        )
        wheel_centers = _leg_vector_map(self.wheel_center_w_m)
        wheel_classes = _leg_class_map(self.wheel_contact_classes)
        wheel_loads = _leg_scalar_map(
            self.wheel_contact_load_n,
            label="wheel_contact_load_n",
            require_all=True,
            allow_none=True,
        )
        front_clearances = _leg_scalar_map(
            self.wheel_front_face_clearance_m,
            label="wheel_front_face_clearance_m",
            require_all=True,
            allow_none=False,
        )
        top_clearances = _leg_scalar_map(
            self.wheel_top_clearance_m,
            label="wheel_top_clearance_m",
            require_all=True,
            allow_none=False,
        )
        base_roll = _finite_scalar(self.base_roll_rad, label="base_roll_rad")
        base_pitch = _finite_scalar(self.base_pitch_rad, label="base_pitch_rad")
        obstacle_front = _finite_scalar(
            self.obstacle_front_face_x_m,
            label="obstacle_front_face_x_m",
        )
        obstacle_top = _finite_scalar(
            self.obstacle_top_z_m,
            label="obstacle_top_z_m",
        )

        # The dataclass is frozen for controller callers.  Canonical copies
        # also prevent later mutation of caller-owned dictionaries from
        # changing the observation after validation.
        object.__setattr__(self, "base_position_m", base_position)
        object.__setattr__(self, "base_roll_rad", base_roll)
        object.__setattr__(self, "base_pitch_rad", base_pitch)
        object.__setattr__(self, "base_angular_velocity_rad_s", angular_velocity)
        object.__setattr__(self, "com_position_m", com_position)
        object.__setattr__(self, "servo_targets_deg", servo_targets)
        object.__setattr__(self, "wheel_targets_rad_s", wheel_targets)
        object.__setattr__(self, "wheel_center_w_m", wheel_centers)
        object.__setattr__(self, "wheel_contact_classes", wheel_classes)
        object.__setattr__(self, "wheel_contact_load_n", wheel_loads)
        object.__setattr__(self, "wheel_front_face_clearance_m", front_clearances)
        object.__setattr__(self, "wheel_top_clearance_m", top_clearances)
        object.__setattr__(self, "obstacle_front_face_x_m", obstacle_front)
        object.__setattr__(self, "obstacle_top_z_m", obstacle_top)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MacroObservation":
        for required in (
            "robot_state_finite",
            "actuator_targets_applied",
            "base_position_m",
            "base_roll_rad",
            "base_pitch_rad",
            "servo_targets_deg",
            "wheel_targets_rad_s",
            "wheel_center_w_m",
            "wheel_contact_classes",
            "wheel_contact_load_n",
            "wheel_front_face_clearance_m",
            "wheel_top_clearance_m",
            "obstacle_front_face_x_m",
            "obstacle_top_z_m",
            "dispatch_error",
            "robot_fell",
            "body_stuck",
            "dangerous_collision",
            "severe_penetration",
            "joint_limit_violation",
            "unsafe_joint_target",
            "active_leg_trapped",
            "wheel_drive_up_without_required_lift",
            "body_crossed_front_face",
            "final_recoverable",
            "posture_complete",
        ):
            if required not in values:
                raise ValueError(f"macro observation is missing required field {required}")
        if not any(
            key in values
            for key in ("base_angular_velocity_rad_s", "root_angular_velocity_w")
        ):
            raise ValueError(
                "macro observation is missing required angular-velocity field"
            )
        dispatch_error = values["dispatch_error"]
        if not isinstance(dispatch_error, str):
            raise ValueError("dispatch_error must be a string")
        root_pose = values.get("root_pose_w", ())
        base_position = values.get("base_position_m", values.get("root_position_w"))
        if base_position is None and isinstance(root_pose, Sequence):
            base_position = root_pose[:3]
        com_raw = values.get("com_position_m", values.get("center_of_mass_w_m"))
        com_position = None if com_raw is None else _strict_vector(
            com_raw, 3, label="com_position_m"
        )
        classes = _leg_class_map(
            values.get(
                "wheel_contact_classes",
                values.get("final_wheel_contact_classes", {}),
            )
        )
        obstacle_front = values.get("obstacle_front_face_x_m")
        obstacle_top = values.get("obstacle_top_z_m")
        try:
            obstacle_front_value = float(obstacle_front)
        except (TypeError, ValueError):
            raise ValueError("obstacle_front_face_x_m must be finite")
        try:
            obstacle_top_value = float(obstacle_top)
        except (TypeError, ValueError):
            raise ValueError("obstacle_top_z_m must be finite")
        if not math.isfinite(obstacle_front_value) or not math.isfinite(obstacle_top_value):
            raise ValueError("obstacle geometry must be finite")
        return cls(
            robot_state_finite=_required_exact_bool(values, "robot_state_finite"),
            base_position_m=_strict_vector(base_position, 3, label="base_position_m"),
            base_roll_rad=_finite_scalar(values.get("base_roll_rad"), label="base_roll_rad"),
            base_pitch_rad=_finite_scalar(values.get("base_pitch_rad"), label="base_pitch_rad"),
            base_angular_velocity_rad_s=_strict_vector(
                values.get(
                    "base_angular_velocity_rad_s",
                    values.get("root_angular_velocity_w"),
                ),
                3,
                label="base_angular_velocity_rad_s",
            ),
            com_position_m=com_position,
            servo_targets_deg=_target_map(
                values.get(
                    "servo_targets_deg",
                    values.get("command_space_servo_target_deg", {}),
                ),
                SERVO_JOINT_NAMES,
                label="servo_targets_deg",
                require_complete=True,
            ),
            wheel_targets_rad_s=_target_map(
                values.get(
                    "wheel_targets_rad_s",
                    values.get("wheel_command_velocity_rad_s", {}),
                ),
                WHEEL_JOINT_NAMES,
                label="wheel_targets_rad_s",
                require_complete=True,
            ),
            wheel_center_w_m=_leg_vector_map(values.get("wheel_center_w_m", {})),
            wheel_contact_classes=classes,
            wheel_contact_load_n=_leg_scalar_map(
                values.get("wheel_contact_load_n", {}),
                label="wheel_contact_load_n",
                require_all=True,
                allow_none=True,
            ),
            wheel_front_face_clearance_m=_leg_scalar_map(
                values.get("wheel_front_face_clearance_m", {}),
                label="wheel_front_face_clearance_m",
                require_all=True,
                allow_none=False,
            ),
            wheel_top_clearance_m=_leg_scalar_map(
                values.get("wheel_top_clearance_m", {}),
                label="wheel_top_clearance_m",
                require_all=True,
                allow_none=False,
            ),
            obstacle_front_face_x_m=obstacle_front_value,
            obstacle_top_z_m=obstacle_top_value,
            actuator_targets_applied=_required_exact_bool(
                values, "actuator_targets_applied"
            ),
            dispatch_error=dispatch_error,
            robot_fell=_required_exact_bool(values, "robot_fell"),
            body_stuck=_required_exact_optional_bool(values, "body_stuck"),
            dangerous_collision=_required_exact_optional_bool(
                values, "dangerous_collision"
            ),
            severe_penetration=_required_exact_optional_bool(
                values, "severe_penetration"
            ),
            joint_limit_violation=_required_exact_bool(
                values, "joint_limit_violation"
            ),
            unsafe_joint_target=_required_exact_bool(values, "unsafe_joint_target"),
            active_leg_trapped=_required_exact_optional_bool(
                values, "active_leg_trapped"
            ),
            wheel_drive_up_without_required_lift=_required_exact_bool(
                values, "wheel_drive_up_without_required_lift"
            ),
            body_crossed_front_face=_required_exact_bool(
                values, "body_crossed_front_face"
            ),
            final_recoverable=_required_exact_bool(values, "final_recoverable"),
            posture_complete=_required_exact_bool(values, "posture_complete"),
        )

    def body_or_com_position(self) -> tuple[tuple[float, float, float], str]:
        if self.com_position_m is not None:
            return self.com_position_m, "MEASURED_COM"
        return self.base_position_m, "BASE_POSITION_PROXY"

    def leg_crossed(self, leg: str) -> bool:
        clearance = self.wheel_front_face_clearance_m.get(leg)
        if clearance is not None:
            return float(clearance) > 0.0
        center = self.wheel_center_w_m.get(leg)
        return bool(
            center is not None
            and self.obstacle_front_face_x_m is not None
            and float(center[0]) > float(self.obstacle_front_face_x_m)
        )

    def leg_airborne(self, leg: str) -> bool:
        top_clearance = self.wheel_top_clearance_m.get(leg)
        return bool(
            self.wheel_contact_classes.get(leg) == "AIR"
            or (top_clearance is not None and float(top_clearance) >= 0.003)
        )

    def leg_top(self, leg: str) -> bool:
        # Geometry classification alone can label a wheel TOP while it is
        # still behind the front plane.  Crossing is an independent event.
        return self.wheel_contact_classes.get(leg) == "TOP" and self.leg_crossed(leg)

    def leg_support_candidate(self, leg: str) -> bool:
        contact_class = self.wheel_contact_classes.get(leg, "UNKNOWN")
        load = self.wheel_contact_load_n.get(leg)
        if load is not None:
            return float(load) > 0.0
        return contact_class in {"GROUND", "TOP"}


class MacroTerminalOutcome(str, Enum):
    RUNNING = "RUNNING"
    TASK_SUCCESS = "TASK_SUCCESS"
    TASK_SUCCESS_POSTURE_INCOMPLETE = "TASK_SUCCESS_POSTURE_INCOMPLETE"
    SAFE_STOP = "SAFE_STOP"


SOURCE_ACTION_IDENTITY_SCHEMA_VERSION = "fsm50.source_action_identity.v1"

SOURCE_ACTION_DISPATCH_KINDS = frozenset(
    {"segment_start", "wheel_channel_completion_stop"}
)

COMMAND_PROVENANCE_KINDS = frozenset(
    {
        "NONE",
        "SOURCE_ACTION",
        "BOUNDARY_ZERO_WHEELS",
        "COMPLETION_WHEEL_STOP",
        "HOLD_ZERO_WHEELS",
        "SAFE_STOP_ZERO_WHEELS",
        "SUCCESS_ZERO_WHEELS",
    }
)

SEGMENT_COMPLETION_TOKEN_SCHEMA_VERSION = "fsm50.macro_segment_completion_token.v1"
SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION = "fsm50.macro_segment_completion_control.v1"
SEGMENT_COMPLETION_KINDS = frozenset(
    {"WAIT", "WHEEL_STOP_DUE", "COMPLETE", "FAIL"}
)
SEGMENT_COMPLETION_CONTROL_KINDS = frozenset({"NONE", "START", "WHEEL_STOP"})
SEGMENT_COMPLETION_DECISION_KEYS = frozenset(
    SegmentDecision.__dataclass_fields__
)
SEGMENT_COMPLETION_TOKEN_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "profile_source_version",
        "owner_state",
        "source_plan_sha256",
        "source_plan_payload_sha256",
        "accepted_steps_sha256",
        "source_segment_index",
        "source_step_index",
        "source_step_id",
        "start_command_epoch",
        "start_sim_step",
        "start_readback_sha256",
        "decision",
        "decision_sha256",
    }
)

_COMMAND_PROVENANCE_KEYS = frozenset(
    {
        "kind",
        "source_action_identity",
        "source_version",
        "source_segment_index",
        "source_step_index",
        "source_time_s",
        "source_event_indices",
        "commands",
        "dispatch_kind",
        "sequence_index",
    }
)


def _exact_nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact non-negative int")
    return value


def _source_action_identity_payload(
    *,
    source_version: str,
    source_segment_index: int,
    source_step_index: int,
    source_time_s: float,
    source_event_indices: Sequence[int],
    commands: Sequence[str],
    dispatch_kind: str,
    sequence_index: int,
) -> dict[str, Any]:
    """Canonical, versioned payload hashed into a source-action identity."""

    return {
        "schema_version": SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
        "source_version": source_version,
        "source_segment_index": source_segment_index,
        "source_step_index": source_step_index,
        "source_time_s": source_time_s,
        "source_event_indices": list(source_event_indices),
        "commands": list(commands),
        "dispatch_kind": dispatch_kind,
        "sequence_index": sequence_index,
    }


def _canonical_command_provenance(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("decision.command_provenance must be a mapping")
    missing = sorted(_COMMAND_PROVENANCE_KEYS - set(values))
    unknown = sorted(set(values) - _COMMAND_PROVENANCE_KEYS)
    if missing or unknown:
        raise ValueError(
            "decision.command_provenance must use the exact schema; "
            f"missing={missing} unknown={unknown}"
        )
    kind = values["kind"]
    if type(kind) is not str or kind not in COMMAND_PROVENANCE_KINDS:
        raise ValueError("decision.command_provenance.kind is invalid")

    if kind != "SOURCE_ACTION":
        expected_empty = {
            "source_action_identity": "",
            "source_version": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_time_s": None,
            "source_event_indices": (),
            "commands": (),
            "dispatch_kind": "",
            "sequence_index": None,
        }
        for key, expected in expected_empty.items():
            actual = values[key]
            if key in {"source_event_indices", "commands"}:
                if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
                    raise ValueError(
                        f"decision.command_provenance.{key} must be an empty sequence"
                    )
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(
                    f"decision.command_provenance.{key} must be empty for {kind}"
                )
        return {
            "kind": kind,
            "source_action_identity": "",
            "source_version": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_time_s": None,
            "source_event_indices": (),
            "commands": (),
            "dispatch_kind": "",
            "sequence_index": None,
        }

    source_version = values["source_version"]
    dispatch_kind = values["dispatch_kind"]
    if type(source_version) is not str or not source_version.strip():
        raise ValueError("SOURCE_ACTION source_version must be a non-empty string")
    if type(dispatch_kind) is not str or not dispatch_kind.strip():
        raise ValueError("SOURCE_ACTION dispatch_kind must be a non-empty string")
    if dispatch_kind not in SOURCE_ACTION_DISPATCH_KINDS:
        raise ValueError("SOURCE_ACTION dispatch_kind is not an allowed source primitive")
    segment = _exact_nonnegative_int(
        values["source_segment_index"], label="source_segment_index"
    )
    step = _exact_nonnegative_int(
        values["source_step_index"], label="source_step_index"
    )
    sequence_index = _exact_nonnegative_int(
        values["sequence_index"], label="sequence_index"
    )
    source_time_s = _finite_scalar(values["source_time_s"], label="source_time_s")
    if source_time_s < 0.0:
        raise ValueError("source_time_s must be non-negative")

    raw_events = values["source_event_indices"]
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError("source_event_indices must be a sequence")
    events = tuple(
        _exact_nonnegative_int(value, label="source_event_indices item")
        for value in raw_events
    )
    raw_commands = values["commands"]
    if not isinstance(raw_commands, Sequence) or isinstance(raw_commands, (str, bytes)):
        raise ValueError("commands must be a sequence")
    commands = tuple(raw_commands)
    if any(type(command) is not str or not command.strip() for command in commands):
        raise ValueError("commands must contain only non-empty strings")
    if dispatch_kind == "segment_start" and (not events or not commands):
        raise ValueError(
            "segment_start provenance requires non-empty events and commands"
        )
    if dispatch_kind == "wheel_channel_completion_stop" and (events or commands):
        raise ValueError(
            "wheel_channel_completion_stop provenance requires empty events and commands"
        )

    identity = values["source_action_identity"]
    if (
        type(identity) is not str
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("source_action_identity must be a lowercase SHA-256 digest")
    expected_identity = _stable_sha256(
        _source_action_identity_payload(
            source_version=source_version,
            source_segment_index=segment,
            source_step_index=step,
            source_time_s=source_time_s,
            source_event_indices=events,
            commands=commands,
            dispatch_kind=dispatch_kind,
            sequence_index=sequence_index,
        )
    )
    if identity != expected_identity:
        raise ValueError("source_action_identity does not match its canonical payload")
    return {
        "kind": kind,
        "source_action_identity": identity,
        "source_version": source_version,
        "source_segment_index": segment,
        "source_step_index": step,
        "source_time_s": source_time_s,
        "source_event_indices": events,
        "commands": commands,
        "dispatch_kind": dispatch_kind,
        "sequence_index": sequence_index,
    }


def _empty_command_provenance(kind: str = "NONE") -> dict[str, Any]:
    return _canonical_command_provenance(
        {
            "kind": kind,
            "source_action_identity": "",
            "source_version": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_time_s": None,
            "source_event_indices": (),
            "commands": (),
            "dispatch_kind": "",
            "sequence_index": None,
        }
    )


@dataclass(frozen=True)
class MacroSegmentCompletionToken:
    """Worker-produced, measured-completion evidence for the active segment."""

    profile_id: str
    profile_source_version: str
    owner_state: str
    source_plan_sha256: str
    source_plan_payload_sha256: str
    accepted_steps_sha256: str
    source_segment_index: int
    source_step_index: int
    source_step_id: str
    start_command_epoch: int
    start_sim_step: int
    start_readback_sha256: str
    decision: Mapping[str, Any]
    decision_sha256: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.profile_source_version:
            raise ValueError("completion token profile identity is incomplete")
        try:
            MacroStateId(self.owner_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("completion token owner_state is invalid") from exc
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
                raise ValueError(f"completion token {label} is invalid")
        segment = _exact_nonnegative_int(
            self.source_segment_index, label="completion token source_segment_index"
        )
        step = _exact_nonnegative_int(
            self.source_step_index, label="completion token source_step_index"
        )
        start_epoch = _exact_nonnegative_int(
            self.start_command_epoch, label="completion token start_command_epoch"
        )
        start_step = _exact_nonnegative_int(
            self.start_sim_step, label="completion token start_sim_step"
        )
        if (
            type(self.start_readback_sha256) is not str
            or len(self.start_readback_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.start_readback_sha256
            )
        ):
            raise ValueError("completion token start_readback_sha256 is invalid")
        if not isinstance(self.source_step_id, str):
            raise ValueError("completion token source_step_id must be a string")
        decision = json.loads(
            json.dumps(dict(self.decision), ensure_ascii=False, allow_nan=False)
        )
        missing = sorted(SEGMENT_COMPLETION_DECISION_KEYS - set(decision))
        unknown = sorted(set(decision) - SEGMENT_COMPLETION_DECISION_KEYS)
        if missing or unknown:
            raise ValueError(
                "completion token decision must use the exact helper schema; "
                f"missing={missing} unknown={unknown}"
            )
        kind = decision["kind"]
        if type(kind) is not str or kind not in SEGMENT_COMPLETION_KINDS:
            raise ValueError("completion token decision kind is invalid")
        decision_segment = _exact_nonnegative_int(
            decision["segment_index"], label="completion decision segment_index"
        )
        decision_step = _exact_nonnegative_int(
            decision["source_step"], label="completion decision source_step"
        )
        if type(decision["source_step_id"]) is not str:
            raise ValueError("completion decision source_step_id must be a string")
        if (
            decision_segment != segment
            or decision_step != step
            or decision["source_step_id"] != self.source_step_id
        ):
            raise ValueError("completion token decision identity mismatch")
        _finite_scalar(decision["sim_time_s"], label="completion token sim_time_s")
        _exact_nonnegative_int(
            decision["sim_step"], label="completion token sim_step"
        )
        if int(decision["sim_step"]) <= start_step:
            raise ValueError("completion token must follow the start physics step")
        for key in ("segment_done", "wheel_stop_due", "wheel_stop_acknowledged"):
            if type(decision[key]) is not bool:
                raise ValueError(f"completion token decision.{key} must be bool")
        if type(decision["failure_reason"]) is not str or type(
            decision["failure_code"]
        ) is not str:
            raise ValueError("completion token failure fields must be strings")
        semantic_flags = {
            "WAIT": (False, False, False),
            "WHEEL_STOP_DUE": (False, True, False),
            "COMPLETE": (True, False, False),
            "FAIL": (False, False, True),
        }
        expected_done, expected_due, expects_failure = semantic_flags[kind]
        if (
            decision["segment_done"] is not expected_done
            or decision["wheel_stop_due"] is not expected_due
        ):
            raise ValueError(f"{kind} token carries contradictory completion flags")
        if decision["wheel_stop_acknowledged"] and kind == "WHEEL_STOP_DUE":
            raise ValueError("WHEEL_STOP_DUE cannot already acknowledge its stop")
        if bool(decision["failure_reason"]) is not expects_failure:
            raise ValueError(f"{kind} token failure_reason semantics are invalid")
        if not expects_failure and decision["failure_code"]:
            raise ValueError(f"{kind} token cannot carry a failure_code")
        expected_sha = _stable_sha256(decision)
        if self.decision_sha256 != expected_sha:
            raise ValueError("completion token decision SHA mismatch")
        object.__setattr__(self, "source_segment_index", segment)
        object.__setattr__(self, "source_step_index", step)
        object.__setattr__(self, "start_command_epoch", start_epoch)
        object.__setattr__(self, "start_sim_step", start_step)
        object.__setattr__(self, "decision", decision)

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "MacroSegmentCompletionToken":
        if not isinstance(values, Mapping):
            raise ValueError("completion token must be a mapping")
        missing = sorted(SEGMENT_COMPLETION_TOKEN_KEYS - set(values))
        unknown = sorted(set(values) - SEGMENT_COMPLETION_TOKEN_KEYS)
        if missing or unknown:
            raise ValueError(
                "completion token must use the exact schema; "
                f"missing={missing} unknown={unknown}"
            )
        if str(values.get("schema_version", "")) != SEGMENT_COMPLETION_TOKEN_SCHEMA_VERSION:
            raise ValueError("completion token schema_version is invalid")
        return cls(
            profile_id=str(values.get("profile_id", "")),
            profile_source_version=str(values.get("profile_source_version", "")),
            owner_state=str(values.get("owner_state", "")),
            source_plan_sha256=str(values.get("source_plan_sha256", "")),
            source_plan_payload_sha256=str(
                values.get("source_plan_payload_sha256", "")
            ),
            accepted_steps_sha256=str(values.get("accepted_steps_sha256", "")),
            source_segment_index=values.get("source_segment_index"),
            source_step_index=values.get("source_step_index"),
            source_step_id=values.get("source_step_id", ""),
            start_command_epoch=values.get("start_command_epoch"),
            start_sim_step=values.get("start_sim_step"),
            start_readback_sha256=str(values.get("start_readback_sha256", "")),
            decision=dict(values.get("decision", {}) or {}),
            decision_sha256=str(values.get("decision_sha256", "")),
        )

    @classmethod
    def from_control_decision(
        cls,
        control: Mapping[str, Any],
        decision: Mapping[str, Any],
        *,
        start_sim_step: int,
        start_readback_sha256: str,
    ) -> "MacroSegmentCompletionToken":
        parsed = _canonical_segment_completion_control(control)
        if parsed["kind"] not in {"START", "WHEEL_STOP"}:
            raise ValueError("completion token needs an active segment control")
        decision_copy = dict(decision)
        return cls(
            profile_id=parsed["profile_id"],
            profile_source_version=parsed["profile_source_version"],
            owner_state=parsed["owner_state"],
            source_plan_sha256=parsed["source_plan_sha256"],
            source_plan_payload_sha256=parsed["source_plan_payload_sha256"],
            accepted_steps_sha256=parsed["accepted_steps_sha256"],
            source_segment_index=parsed["source_segment_index"],
            source_step_index=parsed["source_step_index"],
            source_step_id=parsed["source_step_id"],
            start_command_epoch=parsed["start_command_epoch"],
            start_sim_step=start_sim_step,
            start_readback_sha256=start_readback_sha256,
            decision=decision_copy,
            decision_sha256=_stable_sha256(decision_copy),
        )

    @property
    def kind(self) -> str:
        return str(self.decision["kind"])

    @property
    def sim_step(self) -> int:
        return int(self.decision["sim_step"])

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SEGMENT_COMPLETION_TOKEN_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_source_version": self.profile_source_version,
            "owner_state": self.owner_state,
            "source_plan_sha256": self.source_plan_sha256,
            "source_plan_payload_sha256": self.source_plan_payload_sha256,
            "accepted_steps_sha256": self.accepted_steps_sha256,
            "source_segment_index": self.source_segment_index,
            "source_step_index": self.source_step_index,
            "source_step_id": self.source_step_id,
            "start_command_epoch": self.start_command_epoch,
            "start_sim_step": self.start_sim_step,
            "start_readback_sha256": self.start_readback_sha256,
            "decision": copy.deepcopy(dict(self.decision)),
            "decision_sha256": self.decision_sha256,
        }


_SEGMENT_COMPLETION_CONTROL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "profile_source_version",
        "owner_state",
        "source_plan_sha256",
        "source_plan_payload_sha256",
        "accepted_steps_sha256",
        "source_segment_index",
        "source_step_index",
        "source_step_id",
        "start_command_epoch",
        "completion_spec",
        "source_action_identity",
        "source_action",
        "completion_token_sha256",
    }
)


def _canonical_segment_completion_control(values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("segment_completion_control must be a mapping")
    missing = sorted(_SEGMENT_COMPLETION_CONTROL_KEYS - set(values))
    unknown = sorted(set(values) - _SEGMENT_COMPLETION_CONTROL_KEYS)
    if missing or unknown:
        raise ValueError(
            "segment_completion_control must use the exact schema; "
            f"missing={missing} unknown={unknown}"
        )
    if values["schema_version"] != SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION:
        raise ValueError("segment completion control schema is invalid")
    kind = values["kind"]
    if type(kind) is not str or kind not in SEGMENT_COMPLETION_CONTROL_KINDS:
        raise ValueError("segment completion control kind is invalid")
    if type(values["source_action"]) is not bool:
        raise ValueError("segment completion control source_action must be bool")
    if kind == "NONE":
        expected = {
            "profile_id": "",
            "profile_source_version": "",
            "owner_state": "",
            "source_plan_sha256": "",
            "source_plan_payload_sha256": "",
            "accepted_steps_sha256": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_step_id": "",
            "start_command_epoch": None,
            "completion_spec": {},
            "source_action_identity": "",
            "source_action": False,
            "completion_token_sha256": "",
        }
        for key, expected_value in expected.items():
            if values[key] != expected_value:
                raise ValueError(f"NONE segment completion control must empty {key}")
        return copy.deepcopy(dict(values))

    for key in (
        "profile_id",
        "profile_source_version",
        "source_plan_sha256",
        "source_plan_payload_sha256",
        "accepted_steps_sha256",
    ):
        value = values[key]
        if type(value) is not str or not value:
            raise ValueError(f"segment completion control {key} is required")
    try:
        owner_state = MacroStateId(values["owner_state"]).value
    except (TypeError, ValueError) as exc:
        raise ValueError("segment completion control owner_state is invalid") from exc
    segment = _exact_nonnegative_int(
        values["source_segment_index"], label="completion control segment"
    )
    step = _exact_nonnegative_int(
        values["source_step_index"], label="completion control step"
    )
    start_epoch = _exact_nonnegative_int(
        values["start_command_epoch"], label="completion control start epoch"
    )
    if not isinstance(values["source_step_id"], str):
        raise ValueError("completion control source_step_id must be a string")
    spec = SegmentCompletionSpec.from_mapping(
        dict(values.get("completion_spec", {}) or {})
    )
    if (
        spec.segment_index != segment
        or spec.source_step != step
        or spec.source_step_id != values["source_step_id"]
    ):
        raise ValueError("completion control spec identity mismatch")
    source_identity = values["source_action_identity"]
    token_sha = values["completion_token_sha256"]
    if kind == "START":
        if values["source_action"] is not True or not source_identity or token_sha:
            raise ValueError("START completion control provenance is invalid")
    else:
        if type(token_sha) is not str or len(token_sha) != 64:
            raise ValueError("WHEEL_STOP control requires its due token SHA")
        if values["source_action"] is True and not source_identity:
            raise ValueError("source wheel stop requires source action identity")
        if values["source_action"] is False and source_identity:
            raise ValueError("dynamic wheel stop cannot claim a source action")
    result = copy.deepcopy(dict(values))
    result["completion_spec"] = spec.to_mapping()
    result["source_segment_index"] = segment
    result["source_step_index"] = step
    result["owner_state"] = owner_state
    result["start_command_epoch"] = start_epoch
    return result


def _empty_segment_completion_control() -> dict[str, Any]:
    return _canonical_segment_completion_control(
        {
            "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
            "kind": "NONE",
            "profile_id": "",
            "profile_source_version": "",
            "owner_state": "",
            "source_plan_sha256": "",
            "source_plan_payload_sha256": "",
            "accepted_steps_sha256": "",
            "source_segment_index": None,
            "source_step_index": None,
            "source_step_id": "",
            "start_command_epoch": None,
            "completion_spec": {},
            "source_action_identity": "",
            "source_action": False,
            "completion_token_sha256": "",
        }
    )


def _segment_completion_control(
    kind: str,
    *,
    profile: PhaseMotionProfile,
    binding: PlaybackSegmentBinding,
    source_action_identity: str,
    source_action: bool,
    owner_state: MacroStateId,
    start_command_epoch: int,
    completion_token_sha256: str = "",
) -> dict[str, Any]:
    return _canonical_segment_completion_control(
        {
            "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
            "kind": kind,
            "profile_id": profile.profile_id,
            "profile_source_version": profile.source_version,
            "owner_state": owner_state.value,
            "source_plan_sha256": profile.source_plan_sha256,
            "source_plan_payload_sha256": binding.source_plan_payload_sha256,
            "accepted_steps_sha256": binding.accepted_steps_sha256,
            "source_segment_index": binding.segment_index,
            "source_step_index": binding.source_step,
            "source_step_id": binding.source_step_id,
            "start_command_epoch": start_command_epoch,
            "completion_spec": binding.completion_spec.to_mapping(),
            "source_action_identity": source_action_identity,
            "source_action": source_action,
            "completion_token_sha256": completion_token_sha256,
        }
    )


@dataclass(frozen=True)
class MacroDecision:
    macro_state: MacroStateId
    subphase: MacroSubphase
    profile_id: str
    profile_source_version: str
    profile_strategy: str
    phase_elapsed_s: float
    profile_fraction: float
    servo_targets_deg: Mapping[str, float]
    wheel_targets_rad_s: Mapping[str, float]
    command_epoch: int
    command_changed: bool
    source_action_consumed: bool
    target_changed: bool
    command_provenance: Mapping[str, Any]
    segment_completion_control: Mapping[str, Any]
    transition_events: tuple[str, ...]
    reason: str
    guard_evidence: Mapping[str, Any]
    retry_count: int
    terminal: bool
    terminal_outcome: MacroTerminalOutcome

    def __post_init__(self) -> None:
        servo_targets = _target_map(
            self.servo_targets_deg,
            SERVO_JOINT_NAMES,
            label="decision.servo_targets_deg",
            require_complete=True,
        )
        wheel_targets = _target_map(
            self.wheel_targets_rad_s,
            WHEEL_JOINT_NAMES,
            label="decision.wheel_targets_rad_s",
            require_complete=True,
        )
        if type(self.command_epoch) is not int or self.command_epoch < 0:
            raise ValueError("decision.command_epoch must be a non-negative int")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("decision.retry_count must be a non-negative int")
        if (
            type(self.command_changed) is not bool
            or type(self.source_action_consumed) is not bool
            or type(self.target_changed) is not bool
            or type(self.terminal) is not bool
        ):
            raise ValueError("decision status flags must be bool")
        provenance = _canonical_command_provenance(self.command_provenance)
        completion_control = _canonical_segment_completion_control(
            self.segment_completion_control
        )
        if self.command_changed != self.target_changed:
            raise ValueError("command_changed must exactly equal target_changed")
        if self.command_changed and self.command_epoch == 0:
            raise ValueError("a changed target requires a positive command_epoch")
        if self.source_action_consumed != (provenance["kind"] == "SOURCE_ACTION"):
            raise ValueError(
                "source_action_consumed must exactly match SOURCE_ACTION provenance"
            )
        if self.target_changed and provenance["kind"] == "NONE":
            raise ValueError("a target-changing decision requires dispatch provenance")
        if not self.target_changed and provenance["kind"] not in {
            "NONE",
            "SOURCE_ACTION",
        }:
            raise ValueError(
                "an unchanged target map cannot claim non-source dispatch provenance"
            )
        control_kind = completion_control["kind"]
        if (
            control_kind != "NONE"
            and completion_control["owner_state"] != self.macro_state.value
        ):
            raise ValueError("segment completion control owner differs from decision state")
        if control_kind == "START":
            if (
                provenance["kind"] != "SOURCE_ACTION"
                or provenance["dispatch_kind"] != "segment_start"
                or provenance["source_action_identity"]
                != completion_control["source_action_identity"]
                or completion_control["start_command_epoch"] != self.command_epoch
            ):
                raise ValueError("START completion control lacks its source action")
        elif control_kind == "WHEEL_STOP":
            if completion_control["start_command_epoch"] > self.command_epoch:
                raise ValueError("WHEEL_STOP precedes its segment start epoch")
            if completion_control["source_action"]:
                if (
                    provenance["kind"] != "SOURCE_ACTION"
                    or provenance["dispatch_kind"]
                    != "wheel_channel_completion_stop"
                    or provenance["source_action_identity"]
                    != completion_control["source_action_identity"]
                ):
                    raise ValueError("source WHEEL_STOP control provenance mismatch")
            elif provenance["kind"] != "COMPLETION_WHEEL_STOP":
                raise ValueError("dynamic WHEEL_STOP control provenance mismatch")
        elif provenance["kind"] == "COMPLETION_WHEEL_STOP":
            raise ValueError("completion wheel stop requires WHEEL_STOP control")
        phase_elapsed = _finite_scalar(
            self.phase_elapsed_s, label="decision.phase_elapsed_s"
        )
        fraction = _finite_scalar(
            self.profile_fraction, label="decision.profile_fraction"
        )
        if phase_elapsed < 0.0 or not 0.0 <= fraction <= 1.0:
            raise ValueError("decision progress is outside its valid range")
        object.__setattr__(self, "servo_targets_deg", servo_targets)
        object.__setattr__(self, "wheel_targets_rad_s", wheel_targets)
        object.__setattr__(self, "command_provenance", provenance)
        object.__setattr__(self, "segment_completion_control", completion_control)
        object.__setattr__(self, "phase_elapsed_s", phase_elapsed)
        object.__setattr__(self, "profile_fraction", fraction)

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["macro_state"] = self.macro_state.value
        payload["subphase"] = self.subphase.value
        payload["terminal_outcome"] = self.terminal_outcome.value
        payload["servo_targets_deg"] = dict(self.servo_targets_deg)
        payload["wheel_targets_rad_s"] = dict(self.wheel_targets_rad_s)
        payload["transition_events"] = list(self.transition_events)
        payload["guard_evidence"] = dict(self.guard_evidence)
        payload["command_provenance"] = dict(self.command_provenance)
        payload["command_provenance"]["source_event_indices"] = list(
            self.command_provenance["source_event_indices"]
        )
        payload["command_provenance"]["commands"] = list(
            self.command_provenance["commands"]
        )
        payload["segment_completion_control"] = copy.deepcopy(
            dict(self.segment_completion_control)
        )
        return payload


@dataclass(frozen=True)
class MacroControllerBundle:
    graph: MacroFSMGraph
    profiles: MotionProfileLibrary
    primary_source_version: str = DEFAULT_PRIMARY_VERSION
    bundle_id: str = "fsm50-gate-c-control-bundle-v1"

    def __post_init__(self) -> None:
        versions = {source.source_version for source in self.profiles.successful_sources}
        if self.primary_source_version not in versions:
            raise ValueError("primary_source_version is not a Gate-A success")
        for state in self.graph.states:
            if state.profile_required and not self.profiles.profiles_for_state(
                self.primary_source_version, state.state_id
            ):
                raise ValueError(
                    f"primary profile is missing {state.state_id.value}"
                )

    @property
    def graph_sha256(self) -> str:
        return self.graph.sha256

    @property
    def profile_library_sha256(self) -> str:
        return self.profiles.sha256

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.macro_controller_bundle.v1",
            "bundle_id": self.bundle_id,
            "graph_id": self.graph.graph_id,
            "graph_sha256": self.graph_sha256,
            "profile_library_id": self.profiles.library_id,
            "profile_library_sha256": self.profile_library_sha256,
            "primary_source_version": self.primary_source_version,
            "successful_source_versions": [
                source.source_version for source in self.profiles.successful_sources
            ],
        }

    @property
    def bundle_sha256(self) -> str:
        return _stable_sha256(self.to_mapping())


def build_gate_c_bundle(
    project_root: str | Path,
    *,
    alignment_path: str | Path | None = None,
    primary_source_version: str = DEFAULT_PRIMARY_VERSION,
) -> MacroControllerBundle:
    return MacroControllerBundle(
        graph=build_default_macro_graph(),
        profiles=build_profile_library(project_root, alignment_path=alignment_path),
        primary_source_version=primary_source_version,
    )


@dataclass
class _GuardResult:
    satisfied: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class _CommandEvent:
    source_action_consumed: bool
    target_changed: bool
    command_changed: bool
    command_provenance: Mapping[str, Any]
    segment_completion_control: Mapping[str, Any] = field(
        default_factory=_empty_segment_completion_control
    )

    def __post_init__(self) -> None:
        if (
            type(self.source_action_consumed) is not bool
            or type(self.target_changed) is not bool
            or type(self.command_changed) is not bool
        ):
            raise ValueError("command-event flags must be exact bools")
        provenance = _canonical_command_provenance(self.command_provenance)
        completion_control = _canonical_segment_completion_control(
            self.segment_completion_control
        )
        if self.command_changed != self.target_changed:
            raise ValueError("command event changed flags disagree")
        if self.source_action_consumed != (provenance["kind"] == "SOURCE_ACTION"):
            raise ValueError("command event source provenance disagrees with consumption")
        if self.target_changed and provenance["kind"] == "NONE":
            raise ValueError("changed command event lacks dispatch provenance")
        if not self.target_changed and provenance["kind"] not in {
            "NONE",
            "SOURCE_ACTION",
        }:
            raise ValueError("unchanged command event claims a non-source dispatch")
        object.__setattr__(self, "command_provenance", provenance)
        object.__setattr__(self, "segment_completion_control", completion_control)


def _no_command_event() -> _CommandEvent:
    return _CommandEvent(False, False, False, _empty_command_provenance())


def _non_source_command_event(kind: str, *, target_changed: bool) -> _CommandEvent:
    # Provenance describes an actual physical target dispatch.  Transition
    # events retain boundary/safe-stop semantics when the already-held 8+4 map
    # is unchanged, while dispatch provenance remains NONE.
    if not target_changed:
        return _no_command_event()
    return _CommandEvent(
        False,
        True,
        True,
        _empty_command_provenance(kind),
    )


class MacroFSMController:
    """One graph, selectable recording profiles, and live-event transitions."""

    def __init__(
        self,
        graph: MacroFSMGraph,
        profiles: MotionProfileLibrary,
        *,
        primary_source_version: str = DEFAULT_PRIMARY_VERSION,
    ) -> None:
        self.graph = graph
        self.profiles = profiles
        self.primary_source_version = primary_source_version
        self._reset_internal()

    @classmethod
    def from_bundle(cls, bundle: MacroControllerBundle) -> "MacroFSMController":
        return cls(
            bundle.graph,
            bundle.profiles,
            primary_source_version=bundle.primary_source_version,
        )

    def _reset_internal(self) -> None:
        self._started = False
        self._state_id = self.graph.initial_state
        self._source_version = self.primary_source_version
        self._strategy = ""
        self._state_started_at_s = 0.0
        self._profile_started_at_s = 0.0
        self._last_time_s = 0.0
        self._active_profile: PhaseMotionProfile | None = None
        self._visible_keyframe_indices: list[int] = []
        self._visible_cursor = 0
        self._active_segment_binding: PlaybackSegmentBinding | None = None
        self._active_segment_start_identity = ""
        self._active_segment_start_epoch: int | None = None
        self._active_segment_start_sim_step: int | None = None
        self._active_segment_start_readback_sha256 = ""
        self._completed_segment_indices: set[int] = set()
        self._wheel_stop_requested_segment: int | None = None
        self._last_completion_token_sim_step: int | None = None
        self._last_completion_token_sha256 = ""
        self._last_completion_decision: dict[str, Any] = {}
        self._profile_completed_at_s: float | None = None
        self._consumed_source_actions: set[str] = set()
        self._consumed_source_coordinates: set[tuple[str, int, str]] = set()
        self._command_epoch = 0
        self._current_servo_targets = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self._entry_body_position = (0.0, 0.0, 0.0)
        self._entry_position_source = "BASE_POSITION_PROXY"
        self._target_com_unit_xy: tuple[float, float] | None = None
        self._episode_airborne_before_crossing = {leg: False for leg in LEGS}
        self._episode_crossed_seen = {leg: False for leg in LEGS}
        self._episode_top_seen = {leg: False for leg in LEGS}
        self._state_airborne_before_crossing = {leg: False for leg in LEGS}
        self._state_crossed_seen = {leg: False for leg in LEGS}
        self._state_top_seen = {leg: False for leg in LEGS}
        self._boundary_from_state: MacroStateId | None = None
        self._boundary_to_state: MacroStateId | None = None
        self._boundary_episode_crossed_seen = {leg: False for leg in LEGS}
        self._boundary_episode_top_seen = {leg: False for leg in LEGS}
        self._boundary_predecessor_entry_position = (0.0, 0.0, 0.0)
        self._boundary_transition_position = (0.0, 0.0, 0.0)
        self._boundary_inherited_forward_progress_m = 0.0
        self._boundary_inherited_projected_displacement_m = 0.0
        self._boundary_body_crossed_front_face = False
        self._retry_count = 0
        self._state_attempted_profile_ids: set[str] = set()
        self._hold_started_at_s: float | None = None
        self._terminal_outcome = MacroTerminalOutcome.RUNNING
        self._last_reason = "not started"
        self._last_guard_evidence: dict[str, Any] = {}
        self._last_decision_command_epoch: int | None = None
        self._last_decision_servo_targets: dict[str, float] | None = None
        self._last_decision_wheel_targets: dict[str, float] | None = None
        self._last_decision_consumed_source_action_count: int | None = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.macro_controller_status.v1",
            "started": self._started,
            "macro_state": self._state_id.value,
            "source_version": self._source_version,
            "strategy": self._strategy,
            "profile_id": "" if self._active_profile is None else self._active_profile.profile_id,
            "profile_source_version": (
                "" if self._active_profile is None else self._active_profile.source_version
            ),
            "profile_strategy": (
                "" if self._active_profile is None else self._active_profile.strategy
            ),
            "command_epoch": self._command_epoch,
            "retry_count": self._retry_count,
            "terminal": self._terminal_outcome != MacroTerminalOutcome.RUNNING,
            "terminal_outcome": self._terminal_outcome.value,
            "last_reason": self._last_reason,
            "guard_evidence": dict(self._last_guard_evidence),
            "consumed_source_action_count": len(self._consumed_source_actions),
            "active_source_segment_index": (
                None
                if self._active_segment_binding is None
                else self._active_segment_binding.segment_index
            ),
            "completed_profile_segment_count": len(self._completed_segment_indices),
            "last_completion_token_sha256": self._last_completion_token_sha256,
        }

    def reset(
        self,
        initial_observation: MacroObservation | Mapping[str, Any],
        *,
        sim_time_s: float,
        profile_id: str | None = None,
        source_version: str | None = None,
        strategy: str | None = None,
    ) -> MacroDecision:
        self._reset_internal()
        if profile_id is not None and profile_id != self.profiles.library_id:
            raise ValueError("profile_id does not match the locally built profile library")
        observation = (
            initial_observation
            if isinstance(initial_observation, MacroObservation)
            else MacroObservation.from_mapping(initial_observation)
        )
        requested_source = source_version or self.primary_source_version
        successful = {source.source_version for source in self.profiles.successful_sources}
        if requested_source not in successful:
            raise ValueError("source_version is not a Gate-A success")
        if not math.isfinite(float(sim_time_s)):
            raise ValueError("sim_time_s must be finite")
        self._started = True
        self._source_version = requested_source
        self._strategy = str(strategy or "").upper()
        self._last_time_s = float(sim_time_s)
        self._current_servo_targets = dict(observation.servo_targets_deg)
        self._current_wheel_targets = dict(observation.wheel_targets_rad_s)
        self._enter_state(self.graph.initial_state, observation, float(sim_time_s))
        self._last_guard_evidence = self._nullable_safety_evidence(observation)
        self._last_reason = "controller reset; waiting for live initialization guard"
        return self._decision(
            observation,
            sim_time_s=float(sim_time_s),
            command_event=_no_command_event(),
            transition_events=("RESET:S0_INITIALIZE",),
            reason=self._last_reason,
        )

    def tick(
        self,
        observation: MacroObservation | Mapping[str, Any],
        *,
        sim_time_s: float,
        segment_completion_token: MacroSegmentCompletionToken
        | Mapping[str, Any]
        | None = None,
        source_cursor_permit: bool = False,
    ) -> MacroDecision:
        if not self._started:
            raise RuntimeError("reset() must be called before tick()")
        observed = (
            observation
            if isinstance(observation, MacroObservation)
            else MacroObservation.from_mapping(observation)
        )
        # Retain the tri-state producer truth even on an immediate hard-stop
        # path.  None means unavailable/unknown, never a manufactured False.
        self._last_guard_evidence.update(self._nullable_safety_evidence(observed))
        now = float(sim_time_s)
        if not math.isfinite(now):
            return self._safe_stop(observed, self._last_time_s, "non-finite simulation time")
        if now + 1.0e-12 < self._last_time_s:
            return self._safe_stop(observed, now, "simulation time moved backwards")
        if type(source_cursor_permit) is not bool:
            return self._safe_stop(
                observed, now, "source_cursor_permit must be an exact bool"
            )
        self._last_time_s = now
        if self._terminal_outcome != MacroTerminalOutcome.RUNNING:
            return self._decision(
                observed,
                sim_time_s=now,
                command_event=_no_command_event(),
                transition_events=(),
                reason=self._last_reason,
            )
        hard_failure = self._hard_failure_reason(observed)
        if hard_failure:
            return self._safe_stop(observed, now, hard_failure)

        self._update_episode_events(observed)
        state = self.graph.get(self._state_id)
        try:
            token = (
                None
                if segment_completion_token is None
                else segment_completion_token
                if isinstance(
                    segment_completion_token, MacroSegmentCompletionToken
                )
                else MacroSegmentCompletionToken.from_mapping(
                    segment_completion_token
                )
            )
            if token is not None and not source_cursor_permit:
                raise ValueError(
                    "completion token is forbidden outside an outer-cycle cursor permit"
                )
            command_event = self._advance_profile(
                now,
                segment_completion_token=token,
                source_cursor_permit=source_cursor_permit,
            )
            timeline_complete = self._profile_complete(now)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return self._safe_stop(
                observed,
                now,
                f"segment completion contract failure: {exc}",
            )
        guard = self._evaluate_guard(state, observed, timeline_complete=timeline_complete)
        self._last_guard_evidence = guard.evidence

        # Once a source/control/map-changing command occupies this callback,
        # return it and defer any transition.  The one empty-slot exception is
        # handled below: a final COMPLETE with an already-held boundary map may
        # transition and consume exactly the next state's first source action.
        if (
            command_event.source_action_consumed
            or command_event.target_changed
            or command_event.segment_completion_control["kind"] != "NONE"
        ):
            self._last_reason = (
                "segment/source command occupied this physics slot; transition deferred"
            )
            return self._decision(
                observed,
                sim_time_s=now,
                command_event=command_event,
                transition_events=(),
                reason=self._last_reason,
            )

        if guard.satisfied:
            old = state.state_id
            next_state = state.next_state
            if next_state == self.graph.success_state:
                return self._finish_success(observed, now, old)
            boundary_changed = self._zero_wheels_for_boundary()
            self._record_boundary_carry(old, next_state, observed)
            self._enter_state(next_state, observed, now)
            transition_command_event = _non_source_command_event(
                "BOUNDARY_ZERO_WHEELS", target_changed=boundary_changed
            )
            # A final COMPLETE token can leave an otherwise empty permitted
            # outer slot.  If zeroing the boundary is already the held map,
            # use that same logical decision for exactly the next state's first
            # canonical segment_start.  This is deliberately non-recursive:
            # profile-free feedback states, later guards, and every subsequent
            # source action remain deferred to later callbacks.
            if (
                source_cursor_permit
                and token is not None
                and token.kind == "COMPLETE"
                and timeline_complete
                and not boundary_changed
            ):
                try:
                    transition_command_event = self._start_next_segment()
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    return self._safe_stop(
                        observed,
                        now,
                        f"next-state segment start contract failure: {exc}",
                    )
            self._last_reason = guard.reason or f"physical completion guard passed for {old.value}"
            return self._decision(
                observed,
                sim_time_s=now,
                command_event=transition_command_event,
                transition_events=(f"EXIT:{old.value}", f"ENTER:{next_state.value}"),
                reason=self._last_reason,
            )

        transition_events: tuple[str, ...] = ()
        command_event = _no_command_event()
        # A recording profile has no wall-clock eligibility deadline here.
        # Its shared segment executor owns finite measured-completion liveness.
        # The graph's bounded guard hold begins only after the final segment's
        # COMPLETE token, so nominal recording time cannot consume that budget.
        if timeline_complete:
            if self._hold_started_at_s is None:
                self._hold_started_at_s = now
                hold_changed = False
                if any(abs(value) > 1.0e-12 for value in self._current_wheel_targets.values()):
                    self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
                    self._command_epoch += 1
                    hold_changed = True
                command_event = _non_source_command_event(
                    "HOLD_ZERO_WHEELS", target_changed=hold_changed
                )
                transition_events = (f"HOLD:{state.state_id.value}",)
            hold_elapsed = max(0.0, now - self._hold_started_at_s)
            if hold_elapsed > state.retry_policy.maximum_hold_s:
                if self._retry_count < state.retry_policy.maximum_retries and self._retry_safe(state, observed):
                    self._retry_count += 1
                    # This is a feedback retry, not a motion-profile replay.
                    # Keep the current servo hold and zero wheels, and grant
                    # one more bounded observation window.  Rewinding even a
                    # same-source full snapshot can create a large target jump.
                    self._hold_started_at_s = now
                    self._last_reason = (
                        f"bounded hold-target guard recheck {self._retry_count}/"
                        f"{state.retry_policy.maximum_retries}; no recorded action replayed: "
                        f"{guard.reason}"
                    )
                    return self._decision(
                        observed,
                        sim_time_s=now,
                        command_event=command_event,
                        transition_events=(f"RETRY:{state.state_id.value}:{self._retry_count}",),
                        reason=self._last_reason,
                    )
                return self._safe_stop(
                    observed,
                    now,
                    f"{state.state_id.value} hold/retry exhausted: {guard.reason}",
                )
            self._last_reason = f"bounded hold awaiting physical completion: {guard.reason}"
        else:
            self._last_reason = "advancing recording-derived phase profile; transition guard not yet eligible"

        return self._decision(
            observed,
            sim_time_s=now,
            command_event=command_event,
            transition_events=transition_events,
            reason=self._last_reason,
        )

    def _hard_failure_reason(self, observation: MacroObservation) -> str:
        failures = []
        if not observation.robot_state_finite:
            failures.append("non-finite robot state")
        if not observation.actuator_targets_applied:
            failures.append("actuator target could not be applied")
        if observation.dispatch_error:
            failures.append(f"command dispatch error: {observation.dispatch_error}")
        if observation.robot_fell:
            failures.append("robot fall")
        if observation.body_stuck is True:
            failures.append("body stuck")
        if observation.dangerous_collision is True:
            failures.append("dangerous body collision")
        if observation.severe_penetration is True:
            failures.append("severe penetration")
        if observation.joint_limit_violation or observation.unsafe_joint_target:
            failures.append("joint target/limit safety violation")
        if observation.active_leg_trapped is True:
            failures.append("active leg trapped by obstacle")
        if observation.wheel_drive_up_without_required_lift:
            failures.append("wheel-only drive-up without required lift")
        return "; ".join(failures)

    @staticmethod
    def _nullable_safety_evidence(
        observation: MacroObservation,
    ) -> dict[str, bool | None]:
        return {
            "body_stuck": observation.body_stuck,
            "body_stuck_available": observation.body_stuck is not None,
            "active_leg_trapped": observation.active_leg_trapped,
            "active_leg_trapped_available": (
                observation.active_leg_trapped is not None
            ),
        }

    @staticmethod
    def _source_action_provenance(
        profile: PhaseMotionProfile, frame: MotionKeyframe
    ) -> dict[str, Any]:
        identity = _stable_sha256(
            _source_action_identity_payload(
                source_version=profile.source_version,
                source_segment_index=frame.source_segment_index,
                source_step_index=frame.source_step_index,
                source_time_s=frame.source_time_s,
                source_event_indices=frame.source_event_indices,
                commands=frame.commands,
                dispatch_kind=frame.dispatch_kind,
                sequence_index=frame.sequence_index,
            )
        )
        return _canonical_command_provenance(
            {
                "kind": "SOURCE_ACTION",
                "source_action_identity": identity,
                "source_version": profile.source_version,
                "source_segment_index": frame.source_segment_index,
                "source_step_index": frame.source_step_index,
                "source_time_s": frame.source_time_s,
                "source_event_indices": frame.source_event_indices,
                "commands": frame.commands,
                "dispatch_kind": frame.dispatch_kind,
                "sequence_index": frame.sequence_index,
            }
        )

    @classmethod
    def _action_identity(cls, profile: PhaseMotionProfile, frame: MotionKeyframe) -> str:
        return str(
            cls._source_action_provenance(profile, frame)["source_action_identity"]
        )

    @classmethod
    def _action_coordinate(
        cls, profile: PhaseMotionProfile, frame: MotionKeyframe
    ) -> tuple[str, int, str]:
        provenance = cls._source_action_provenance(profile, frame)
        return (
            str(provenance["source_version"]),
            int(provenance["source_segment_index"]),
            str(provenance["dispatch_kind"]),
        )

    def _select_profile(self, state: MacroStateSpec) -> PhaseMotionProfile | None:
        if not state.profile_required and not self.profiles.profiles_for_state(
            self._source_version, state.state_id
        ):
            return None
        candidates = self.profiles.profiles_for_state(self._source_version, state.state_id)
        if not candidates:
            if state.profile_required:
                raise KeyError(f"missing profile for {self._source_version}/{state.state_id.value}")
            return None
        preferences: list[str] = []
        if state.state_id == MacroStateId.S10_POSTURE_RECOVERY:
            preferences.append("RECOVERY_PROFILE")
        if self._strategy:
            preferences.append(self._strategy)
        preferences.append("PRIMARY_PROFILE")
        for preference in preferences:
            matching = [
                candidate
                for candidate in candidates
                if candidate.strategy == preference
                or candidate.strategy.startswith(preference + "_")
            ]
            if matching:
                return sorted(matching, key=lambda item: item.profile_id)[0]
        if len(candidates) == 1:
            return candidates[0]
        return sorted(candidates, key=lambda item: item.profile_id)[0]

    def _prepare_profile(
        self,
        state: MacroStateSpec,
        *,
        profile_override: PhaseMotionProfile | None = None,
    ) -> None:
        profile = profile_override if profile_override is not None else self._select_profile(state)
        if profile is not None and profile.source_version != self._source_version:
            raise ValueError(
                "runtime cross-source profile switching is forbidden; "
                "choose the recording source once at reset"
            )
        self._active_profile = profile
        self._visible_cursor = 0
        self._active_segment_binding = None
        self._active_segment_start_identity = ""
        self._active_segment_start_epoch = None
        self._active_segment_start_sim_step = None
        self._active_segment_start_readback_sha256 = ""
        self._completed_segment_indices = set()
        self._wheel_stop_requested_segment = None
        self._last_completion_token_sim_step = None
        self._last_completion_token_sha256 = ""
        self._last_completion_decision = {}
        self._profile_completed_at_s = None
        if profile is None:
            self._visible_keyframe_indices = []
            return
        if not profile.segment_bindings:
            raise ValueError(
                f"profile {profile.profile_id} lacks SHA-bound completion segments"
            )
        self._state_attempted_profile_ids.add(profile.profile_id)
        visible: list[int] = []
        profile_coordinates: set[tuple[str, int, str]] = set()
        profile_identities: set[str] = set()
        for index, frame in enumerate(profile.keyframes):
            identity = self._action_identity(profile, frame)
            coordinate = self._action_coordinate(profile, frame)
            if identity in profile_identities or coordinate in profile_coordinates:
                raise ValueError(
                    f"duplicate source action in profile {profile.profile_id}: {coordinate}"
                )
            if (
                identity in self._consumed_source_actions
                or coordinate in self._consumed_source_coordinates
            ):
                raise ValueError(
                    f"source action would be consumed twice: {coordinate}"
                )
            profile_identities.add(identity)
            profile_coordinates.add(coordinate)
            visible.append(index)
        self._visible_keyframe_indices = visible

    def _enter_state(
        self, state_id: MacroStateId, observation: MacroObservation, sim_time_s: float
    ) -> None:
        self._state_id = state_id
        self._state_started_at_s = sim_time_s
        self._profile_started_at_s = sim_time_s
        self._retry_count = 0
        self._state_attempted_profile_ids = set()
        self._hold_started_at_s = None
        self._state_airborne_before_crossing = {leg: False for leg in LEGS}
        self._state_crossed_seen = {leg: False for leg in LEGS}
        self._state_top_seen = {leg: False for leg in LEGS}
        self._seed_state_entry_event(self.graph.get(state_id), observation)
        position, source = observation.body_or_com_position()
        self._entry_body_position = position
        self._entry_position_source = source
        self._target_com_unit_xy = self._target_direction_unit(
            self.graph.get(state_id), observation, position
        )
        if self._boundary_to_state == state_id:
            dx = (
                self._boundary_transition_position[0]
                - self._boundary_predecessor_entry_position[0]
            )
            dy = (
                self._boundary_transition_position[1]
                - self._boundary_predecessor_entry_position[1]
            )
            self._boundary_inherited_forward_progress_m = dx
            self._boundary_inherited_projected_displacement_m = (
                0.0
                if self._target_com_unit_xy is None
                else dx * self._target_com_unit_xy[0]
                + dy * self._target_com_unit_xy[1]
            )
        self._prepare_profile(self.graph.get(state_id))

    def _record_boundary_carry(
        self,
        from_state: MacroStateId,
        to_state: MacroStateId,
        observation: MacroObservation,
    ) -> None:
        """Snapshot only the immediate predecessor's causal boundary evidence."""

        position, _ = observation.body_or_com_position()
        self._boundary_from_state = from_state
        self._boundary_to_state = to_state
        self._boundary_episode_crossed_seen = dict(self._episode_crossed_seen)
        self._boundary_episode_top_seen = dict(self._episode_top_seen)
        self._boundary_predecessor_entry_position = self._entry_body_position
        self._boundary_transition_position = position
        self._boundary_inherited_forward_progress_m = (
            position[0] - self._entry_body_position[0]
        )
        self._boundary_inherited_projected_displacement_m = 0.0
        self._boundary_body_crossed_front_face = observation.body_crossed_front_face

    def _boundary_carry_is_fresh(
        self, state_id: MacroStateId, from_state: MacroStateId
    ) -> bool:
        return bool(
            self._boundary_to_state == state_id
            and self._boundary_from_state == from_state
        )

    def _seed_state_entry_event(
        self, state: MacroStateSpec, observation: MacroObservation
    ) -> None:
        """Preserve only a still-live active-leg event at an adjacent boundary.

        S1->S2 and S5->S6 may sample the unload event on the transition physics
        step.  Seeding that same observation avoids losing it.  Historical AIR
        from an earlier state is never copied: the active leg must still be
        airborne and not crossed in this exact entry observation.
        """

        leg = state.active_leg
        if not leg:
            return
        crossed_now = observation.leg_crossed(leg)
        top_now = observation.leg_top(leg)
        self._state_airborne_before_crossing[leg] = bool(
            observation.leg_airborne(leg) and not crossed_now
        )
        self._state_crossed_seen[leg] = crossed_now
        self._state_top_seen[leg] = top_now

    def _zero_wheels_for_boundary(self) -> bool:
        moving = any(abs(value) > 1.0e-12 for value in self._current_wheel_targets.values())
        if moving:
            self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            self._command_epoch += 1
        return moving

    @staticmethod
    def _target_direction_unit(
        state: MacroStateSpec,
        observation: MacroObservation,
        position: tuple[float, float, float],
    ) -> tuple[float, float] | None:
        leg = state.completion_guard.target_com_leg
        if not leg:
            return None
        target = observation.wheel_center_w_m.get(leg)
        if target is None:
            return None
        dx, dy = float(target[0]) - position[0], float(target[1]) - position[1]
        norm = math.hypot(dx, dy)
        if norm <= 1.0e-9:
            return None
        return dx / norm, dy / norm

    def _consume_source_frame(
        self,
        frame: MotionKeyframe,
        *,
        completion_control: Mapping[str, Any] | None = None,
    ) -> _CommandEvent:
        assert self._active_profile is not None
        provenance = self._source_action_provenance(self._active_profile, frame)
        identity = str(provenance["source_action_identity"])
        coordinate = (
            str(provenance["source_version"]),
            int(provenance["source_segment_index"]),
            str(provenance["dispatch_kind"]),
        )
        if (
            identity in self._consumed_source_actions
            or coordinate in self._consumed_source_coordinates
        ):
            raise ValueError(f"duplicate source action consumption: {coordinate}")
        servo_targets = dict(frame.servo_targets_deg)
        wheel_targets = dict(frame.wheel_targets_rad_s)
        target_changed = bool(
            servo_targets != self._current_servo_targets
            or wheel_targets != self._current_wheel_targets
        )
        if target_changed:
            self._current_servo_targets = servo_targets
            self._current_wheel_targets = wheel_targets
            self._command_epoch += 1
        self._visible_cursor += 1
        self._consumed_source_actions.add(identity)
        self._consumed_source_coordinates.add(coordinate)
        return _CommandEvent(
            True,
            target_changed,
            target_changed,
            provenance,
            completion_control or _empty_segment_completion_control(),
        )

    def _validate_active_completion_token(
        self, token: MacroSegmentCompletionToken
    ) -> PlaybackSegmentBinding:
        profile = self._active_profile
        binding = self._active_segment_binding
        if profile is None or binding is None:
            raise ValueError("completion token arrived without an active source segment")
        if (
            token.profile_id != profile.profile_id
            or token.profile_source_version != profile.source_version
            or token.owner_state != self._state_id.value
            or token.source_plan_sha256 != profile.source_plan_sha256
            or token.source_plan_payload_sha256
            != binding.source_plan_payload_sha256
            or token.accepted_steps_sha256 != binding.accepted_steps_sha256
            or token.source_segment_index != binding.segment_index
            or token.source_step_index != binding.source_step
            or token.source_step_id != binding.source_step_id
        ):
            raise ValueError("completion token is stale or cross-profile/source/state")
        if token.start_command_epoch != self._active_segment_start_epoch:
            raise ValueError("completion token start command epoch mismatch")
        if self._active_segment_start_sim_step is None:
            self._active_segment_start_sim_step = token.start_sim_step
            self._active_segment_start_readback_sha256 = (
                token.start_readback_sha256
            )
        elif (
            token.start_sim_step != self._active_segment_start_sim_step
            or token.start_readback_sha256
            != self._active_segment_start_readback_sha256
        ):
            raise ValueError("completion token start step/readback binding changed")
        if (
            self._last_completion_token_sim_step is not None
            and token.sim_step <= self._last_completion_token_sim_step
        ):
            raise ValueError("completion token sim_step is stale or duplicated")
        self._last_completion_token_sim_step = token.sim_step
        self._last_completion_token_sha256 = token.sha256
        self._last_completion_decision = copy.deepcopy(dict(token.decision))
        return binding

    def _start_next_segment(self) -> _CommandEvent:
        profile = self._active_profile
        if profile is None or self._active_segment_binding is not None:
            return _no_command_event()
        if self._visible_cursor >= len(self._visible_keyframe_indices):
            return _no_command_event()
        index = self._visible_keyframe_indices[self._visible_cursor]
        frame = profile.keyframes[index]
        if frame.dispatch_kind != "segment_start":
            raise ValueError(
                "completion-aware cursor reached a wheel stop without an active segment"
            )
        binding = profile.segment_binding(frame.source_segment_index)
        if binding.source_step != frame.source_step_index:
            raise ValueError("source action step differs from completion segment")
        # Compute the exact post-action epoch first; START control binds the
        # epoch even when this legal source action is a same-target no-op.
        prospective_changed = bool(
            dict(frame.servo_targets_deg) != self._current_servo_targets
            or dict(frame.wheel_targets_rad_s) != self._current_wheel_targets
        )
        start_epoch = self._command_epoch + (1 if prospective_changed else 0)
        provenance = self._source_action_provenance(profile, frame)
        control = _segment_completion_control(
            "START",
            profile=profile,
            binding=binding,
            source_action_identity=str(provenance["source_action_identity"]),
            source_action=True,
            owner_state=self._state_id,
            start_command_epoch=start_epoch,
        )
        event = self._consume_source_frame(frame, completion_control=control)
        if self._command_epoch != start_epoch:
            raise RuntimeError("segment START epoch differs from source action")
        self._active_segment_binding = binding
        self._active_segment_start_identity = str(
            provenance["source_action_identity"]
        )
        self._active_segment_start_epoch = start_epoch
        self._active_segment_start_sim_step = None
        self._active_segment_start_readback_sha256 = ""
        self._wheel_stop_requested_segment = None
        self._last_completion_token_sim_step = None
        self._last_completion_token_sha256 = ""
        self._last_completion_decision = {}
        return event

    def _request_completion_wheel_stop(
        self,
        token: MacroSegmentCompletionToken,
        binding: PlaybackSegmentBinding,
    ) -> _CommandEvent:
        profile = self._active_profile
        assert profile is not None
        if self._wheel_stop_requested_segment is not None:
            raise ValueError("completion wheel stop was already requested")
        if not any(
            abs(value) > 1.0e-12 for value in self._current_wheel_targets.values()
        ):
            raise ValueError("completion helper requested a stop for zero wheels")
        next_frame: MotionKeyframe | None = None
        if self._visible_cursor < len(self._visible_keyframe_indices):
            next_frame = profile.keyframes[
                self._visible_keyframe_indices[self._visible_cursor]
            ]
        if (
            next_frame is not None
            and next_frame.source_segment_index == binding.segment_index
            and next_frame.dispatch_kind == "wheel_channel_completion_stop"
        ):
            provenance = self._source_action_provenance(profile, next_frame)
            control = _segment_completion_control(
                "WHEEL_STOP",
                profile=profile,
                binding=binding,
                source_action_identity=str(provenance["source_action_identity"]),
                source_action=True,
                owner_state=self._state_id,
                start_command_epoch=int(self._active_segment_start_epoch),
                completion_token_sha256=token.sha256,
            )
            event = self._consume_source_frame(
                next_frame, completion_control=control
            )
            if not event.target_changed:
                raise ValueError("source wheel completion stop changed no target")
        else:
            self._current_wheel_targets = {
                name: 0.0 for name in WHEEL_JOINT_NAMES
            }
            self._command_epoch += 1
            control = _segment_completion_control(
                "WHEEL_STOP",
                profile=profile,
                binding=binding,
                source_action_identity="",
                source_action=False,
                owner_state=self._state_id,
                start_command_epoch=int(self._active_segment_start_epoch),
                completion_token_sha256=token.sha256,
            )
            event = _CommandEvent(
                False,
                True,
                True,
                _empty_command_provenance("COMPLETION_WHEEL_STOP"),
                control,
            )
        self._wheel_stop_requested_segment = binding.segment_index
        return event

    def _complete_active_segment(
        self, token: MacroSegmentCompletionToken, sim_time_s: float
    ) -> None:
        binding = self._active_segment_binding
        if binding is None:
            raise ValueError("no active segment can complete")
        if self._wheel_stop_requested_segment is not None and token.decision[
            "wheel_stop_acknowledged"
        ] is not True:
            raise ValueError("segment completed without wheel-stop acknowledgement")
        if binding.segment_index in self._completed_segment_indices:
            raise ValueError("source segment completed twice")
        self._completed_segment_indices.add(binding.segment_index)
        self._active_segment_binding = None
        self._active_segment_start_identity = ""
        self._active_segment_start_epoch = None
        self._active_segment_start_sim_step = None
        self._active_segment_start_readback_sha256 = ""
        self._wheel_stop_requested_segment = None
        if self._active_profile is not None and len(
            self._completed_segment_indices
        ) == len(self._active_profile.segment_bindings):
            self._profile_completed_at_s = sim_time_s

    def _advance_profile(
        self,
        sim_time_s: float,
        *,
        segment_completion_token: MacroSegmentCompletionToken | None,
        source_cursor_permit: bool,
    ) -> _CommandEvent:
        if self._active_profile is None:
            if segment_completion_token is not None:
                raise ValueError("completion token arrived in a profile-free state")
            return _no_command_event()
        if segment_completion_token is not None:
            binding = self._validate_active_completion_token(
                segment_completion_token
            )
            if segment_completion_token.kind == "FAIL":
                raise RuntimeError(
                    "completion helper failed "
                    f"{segment_completion_token.decision['failure_reason']}:"
                    f"{segment_completion_token.decision['failure_code']}"
                )
            if segment_completion_token.kind == "WHEEL_STOP_DUE":
                return self._request_completion_wheel_stop(
                    segment_completion_token, binding
                )
            if segment_completion_token.kind == "COMPLETE":
                self._complete_active_segment(
                    segment_completion_token, sim_time_s
                )
            elif segment_completion_token.kind != "WAIT":
                raise ValueError("unsupported completion token kind")
        if not source_cursor_permit:
            return _no_command_event()
        if self._active_segment_binding is not None:
            return _no_command_event()
        return self._start_next_segment()

    def _profile_complete(self, sim_time_s: float) -> bool:
        del sim_time_s
        if self._active_profile is None:
            return True
        expected_segments = {
            binding.segment_index for binding in self._active_profile.segment_bindings
        }
        return bool(
            self._active_segment_binding is None
            and self._visible_cursor >= len(self._visible_keyframe_indices)
            and self._completed_segment_indices == expected_segments
            and self._profile_completed_at_s is not None
        )

    def _update_episode_events(self, observation: MacroObservation) -> None:
        for leg in LEGS:
            crossed_now = observation.leg_crossed(leg)
            airborne_now = observation.leg_airborne(leg)
            top_now = observation.leg_top(leg)
            if (
                airborne_now
                and not self._episode_crossed_seen[leg]
                and not crossed_now
            ):
                self._episode_airborne_before_crossing[leg] = True
            if airborne_now and not self._state_crossed_seen[leg] and not crossed_now:
                self._state_airborne_before_crossing[leg] = True
            self._episode_crossed_seen[leg] = (
                self._episode_crossed_seen[leg] or crossed_now
            )
            self._episode_top_seen[leg] = self._episode_top_seen[leg] or top_now
            self._state_crossed_seen[leg] = self._state_crossed_seen[leg] or crossed_now
            self._state_top_seen[leg] = self._state_top_seen[leg] or top_now

    def _position_progress(self, observation: MacroObservation) -> tuple[float, float, str]:
        current, source = observation.body_or_com_position()
        forward = current[0] - self._entry_body_position[0]
        projected = 0.0
        if self._target_com_unit_xy is not None:
            dx = current[0] - self._entry_body_position[0]
            dy = current[1] - self._entry_body_position[1]
            projected = dx * self._target_com_unit_xy[0] + dy * self._target_com_unit_xy[1]
        return forward, projected, source

    def _evaluate_guard(
        self,
        state: MacroStateSpec,
        observation: MacroObservation,
        *,
        timeline_complete: bool,
    ) -> _GuardResult:
        guard = state.completion_guard
        attitude_safe = bool(
            abs(observation.base_roll_rad) <= guard.maximum_abs_roll_rad
            and abs(observation.base_pitch_rad) <= guard.maximum_abs_pitch_rad
        )
        forward, projected, position_source = self._position_progress(observation)
        evidence: dict[str, Any] = {
            "profile_timeline_complete": timeline_complete,
            "position_evidence_source": position_source,
            "state_entry_position_source": self._entry_position_source,
            "forward_progress_m": forward,
            "target_direction_projected_displacement_m": projected,
            "attitude_safe": attitude_safe,
            "state_airborne_before_crossing": dict(self._state_airborne_before_crossing),
            "state_crossed_seen": dict(self._state_crossed_seen),
            "state_top_seen": dict(self._state_top_seen),
            "episode_crossed_seen": dict(self._episode_crossed_seen),
            "episode_top_seen": dict(self._episode_top_seen),
            "boundary_from_state": (
                "" if self._boundary_from_state is None else self._boundary_from_state.value
            ),
            "boundary_to_state": (
                "" if self._boundary_to_state is None else self._boundary_to_state.value
            ),
            "boundary_inherited_forward_progress_m": (
                self._boundary_inherited_forward_progress_m
            ),
            "boundary_inherited_projected_displacement_m": (
                self._boundary_inherited_projected_displacement_m
            ),
            **self._nullable_safety_evidence(observation),
        }
        if guard.profile_must_complete and not timeline_complete:
            return _GuardResult(False, evidence, "phase-local profile timeline is still active")
        if not attitude_safe:
            return _GuardResult(False, evidence, "body attitude is outside the recoverable guard")

        kind = guard.kind
        physical = False
        reason = "physical completion event not observed"
        if kind == MacroGuardKind.INITIALIZED:
            physical = observation.robot_state_finite and observation.actuator_targets_applied
            reason = "finite articulation and writable targets observed" if physical else reason
        elif kind == MacroGuardKind.COM_SHIFT_OR_UNLOAD:
            active_unloaded = bool(
                guard.active_leg
                and self._state_airborne_before_crossing[guard.active_leg]
            )
            inherited_fresh = bool(
                state.state_id == MacroStateId.S5_PRE_RR_COM_SHIFT
                and self._boundary_carry_is_fresh(
                    state.state_id, MacroStateId.S4_FRONT_PAIR_ADVANCE
                )
            )
            inherited_projected = (
                self._boundary_inherited_projected_displacement_m
                if inherited_fresh
                else 0.0
            )
            displacement_ready = max(projected, inherited_projected) >= (
                guard.minimum_com_displacement_m
            )
            evidence.update(
                active_leg_unloaded=active_unloaded,
                minimum_com_displacement_m=guard.minimum_com_displacement_m,
                displacement_ready=displacement_ready,
                inherited_target_direction_displacement_m=inherited_projected,
                inherited_displacement_fresh=inherited_fresh,
            )
            physical = active_unloaded or displacement_ready
            reason = (
                "active leg unloaded or body/COM moved toward target support region"
                if physical
                else "neither unload nor target-direction displacement observed"
            )
        elif kind == MacroGuardKind.LEG_TRAVERSED:
            leg = guard.active_leg
            airborne = bool(leg and self._state_airborne_before_crossing[leg])
            crossed = bool(leg and self._state_crossed_seen[leg])
            # The active wheel's post-cross TOP event is phase-local and
            # latched.  It may become AIR again at a recorded segment tail.
            # Earlier support wheels use episode-latched post-cross TOP events;
            # simultaneous all-TOP is only a secondary diagnostic.
            top = bool(leg and self._state_top_seen[leg])
            required_top_evidence = {
                item: bool(
                    self._state_top_seen[item]
                    if item == leg
                    else self._episode_top_seen[item]
                )
                for item in guard.required_top_legs
            }
            required_top = all(required_top_evidence.values())
            physical = bool(
                crossed
                and top
                and required_top
                and (airborne or not guard.require_airborne_before_crossing)
            )
            evidence.update(
                active_leg=leg,
                active_leg_airborne_before_crossing=airborne,
                active_leg_crossed=crossed,
                active_leg_top=top,
                required_top_legs=list(guard.required_top_legs),
                required_top_evidence=required_top_evidence,
                required_top_ready=required_top,
            )
            reason = "active leg lifted, crossed, and reached top" if physical else reason
        elif kind == MacroGuardKind.FRONT_PAIR_ADVANCED:
            carry_fresh = self._boundary_carry_is_fresh(
                state.state_id, MacroStateId.S3_FL_TRAVERSE
            )
            top_evidence = {
                item: bool(
                    self._state_top_seen[item]
                    or (
                        carry_fresh
                        and self._boundary_episode_top_seen[item]
                    )
                )
                for item in guard.required_top_legs
            }
            tops = all(top_evidence.values())
            rear_clearances = [
                observation.wheel_front_face_clearance_m.get(leg) for leg in ("RL", "RR")
            ]
            rear_approached = any(
                value is not None and float(value) >= -0.08 for value in rear_clearances
            )
            progress_ready = forward >= guard.minimum_body_progress_m
            inherited_progress = (
                self._boundary_inherited_forward_progress_m if carry_fresh else 0.0
            )
            inherited_progress_ready = (
                inherited_progress >= guard.minimum_body_progress_m
            )
            physical = tops and (
                progress_ready or inherited_progress_ready or rear_approached
            )
            evidence.update(
                front_pair_top=tops,
                front_pair_top_evidence=top_evidence,
                rear_face_approach_seen=rear_approached,
                minimum_body_progress_m=guard.minimum_body_progress_m,
                predecessor_carry_fresh=carry_fresh,
                inherited_forward_progress_m=inherited_progress,
                inherited_progress_ready=inherited_progress_ready,
            )
            reason = "front pair top and rear/body approach event observed" if physical else reason
        elif kind == MacroGuardKind.SUPPORT_SETUP:
            support = {
                leg: observation.leg_support_candidate(leg)
                for leg in guard.required_support_legs
            }
            physical = all(support.values())
            evidence.update(
                candidate_support_legs=list(guard.required_support_legs),
                candidate_support_evidence=support,
                support_claim="candidate geometry/load only; not an anchored-support claim",
            )
            reason = "candidate support geometry observed" if physical else reason
        elif kind == MacroGuardKind.FINAL_ADVANCED:
            progress_ready = forward >= guard.minimum_body_progress_m
            carry_fresh = self._boundary_carry_is_fresh(
                state.state_id, MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE
            )
            inherited_progress = (
                self._boundary_inherited_forward_progress_m if carry_fresh else 0.0
            )
            inherited_progress_ready = (
                inherited_progress >= guard.minimum_body_progress_m
            )
            current_major_crossed = sum(self._state_crossed_seen.values()) >= 3
            inherited_major_crossed = bool(
                carry_fresh
                and sum(self._boundary_episode_crossed_seen.values()) >= 3
            )
            major_crossed = current_major_crossed or inherited_major_crossed
            body_crossed = observation.body_crossed_front_face
            physical = bool(
                (body_crossed if guard.require_body_crossed else True)
                and (progress_ready or inherited_progress_ready or major_crossed)
            )
            evidence.update(
                body_crossed_front_face=body_crossed,
                current_state_crossed_wheel_count=sum(
                    self._state_crossed_seen.values()
                ),
                inherited_crossed_wheel_count=(
                    sum(self._boundary_episode_crossed_seen.values())
                    if carry_fresh
                    else 0
                ),
                minimum_body_progress_m=guard.minimum_body_progress_m,
                predecessor_carry_fresh=carry_fresh,
                inherited_forward_progress_m=inherited_progress,
                inherited_progress_ready=inherited_progress_ready,
            )
            reason = "body and major wheel group crossed with recovery workspace" if physical else reason
        elif kind == MacroGuardKind.POSTURE_RECOVERED:
            body_crossed = observation.body_crossed_front_face
            physical = bool(
                observation.final_recoverable
                and (body_crossed if guard.require_body_crossed else True)
            )
            evidence.update(
                body_crossed_front_face=body_crossed,
                final_recoverable=observation.final_recoverable,
                posture_complete=observation.posture_complete,
            )
            reason = "task crossed and final state is recoverable" if physical else reason
        return _GuardResult(physical, evidence, reason)

    def _retry_safe(self, state: MacroStateSpec, observation: MacroObservation) -> bool:
        if not state.retry_policy.retry_requires_safe_attitude:
            return True
        guard = state.completion_guard
        return bool(
            abs(observation.base_roll_rad) <= guard.maximum_abs_roll_rad
            and abs(observation.base_pitch_rad) <= guard.maximum_abs_pitch_rad
            and not observation.robot_fell
            and observation.active_leg_trapped is not True
        )

    def _safe_stop(
        self, observation: MacroObservation, sim_time_s: float, reason: str
    ) -> MacroDecision:
        changed = any(abs(value) > 1.0e-12 for value in self._current_wheel_targets.values())
        self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        if changed:
            self._command_epoch += 1
        previous = self._state_id
        self._state_id = self.graph.safe_stop_state
        self._active_profile = None
        self._terminal_outcome = MacroTerminalOutcome.SAFE_STOP
        self._last_reason = reason
        return self._decision(
            observation,
            sim_time_s=sim_time_s,
            command_event=_non_source_command_event(
                "SAFE_STOP_ZERO_WHEELS", target_changed=changed
            ),
            transition_events=(f"SAFE_STOP:{previous.value}",),
            reason=reason,
        )

    def _finish_success(
        self,
        observation: MacroObservation,
        sim_time_s: float,
        previous: MacroStateId,
    ) -> MacroDecision:
        changed = any(abs(value) > 1.0e-12 for value in self._current_wheel_targets.values())
        self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        if changed:
            self._command_epoch += 1
        self._state_id = self.graph.success_state
        self._active_profile = None
        self._terminal_outcome = (
            MacroTerminalOutcome.TASK_SUCCESS
            if observation.posture_complete
            else MacroTerminalOutcome.TASK_SUCCESS_POSTURE_INCOMPLETE
        )
        self._last_reason = "50 mm traversal complete; posture classified independently"
        return self._decision(
            observation,
            sim_time_s=sim_time_s,
            command_event=_non_source_command_event(
                "SUCCESS_ZERO_WHEELS", target_changed=changed
            ),
            transition_events=(f"EXIT:{previous.value}", "ENTER:SUCCESS"),
            reason=self._last_reason,
        )

    def _decision(
        self,
        observation: MacroObservation,
        *,
        sim_time_s: float,
        command_event: _CommandEvent,
        transition_events: tuple[str, ...],
        reason: str,
    ) -> MacroDecision:
        provenance_kind = str(command_event.command_provenance["kind"])
        if command_event.target_changed:
            if provenance_kind == "NONE":
                raise RuntimeError("target map changed without dispatch provenance")
        elif provenance_kind not in {"NONE", "SOURCE_ACTION"}:
            raise RuntimeError(
                "non-source dispatch provenance is forbidden when the target map is unchanged"
            )

        if self._last_decision_command_epoch is None:
            if (
                self._command_epoch != 0
                or command_event.source_action_consumed
                or command_event.target_changed
                or provenance_kind != "NONE"
                or self._consumed_source_actions
            ):
                raise RuntimeError("controller reset decision has invalid command provenance")
        else:
            assert self._last_decision_servo_targets is not None
            assert self._last_decision_wheel_targets is not None
            assert self._last_decision_consumed_source_action_count is not None
            map_changed = bool(
                self._current_servo_targets != self._last_decision_servo_targets
                or self._current_wheel_targets != self._last_decision_wheel_targets
            )
            if map_changed != command_event.target_changed:
                raise RuntimeError(
                    "target_changed does not match the exact 8+4 target-map delta"
                )
            epoch_delta = self._command_epoch - self._last_decision_command_epoch
            expected_epoch_delta = 1 if command_event.target_changed else 0
            if epoch_delta != expected_epoch_delta:
                raise RuntimeError(
                    "command_epoch drift does not match the target-map delta"
                )
            consumed_delta = (
                len(self._consumed_source_actions)
                - self._last_decision_consumed_source_action_count
            )
            expected_consumed_delta = 1 if command_event.source_action_consumed else 0
            if consumed_delta != expected_consumed_delta:
                raise RuntimeError(
                    "source action consumption drift does not match decision provenance"
                )
            if command_event.source_action_consumed:
                identity = str(
                    command_event.command_provenance["source_action_identity"]
                )
                coordinate = (
                    str(command_event.command_provenance["source_version"]),
                    int(command_event.command_provenance["source_segment_index"]),
                    str(command_event.command_provenance["dispatch_kind"]),
                )
                if (
                    identity not in self._consumed_source_actions
                    or coordinate not in self._consumed_source_coordinates
                ):
                    raise RuntimeError(
                        "SOURCE_ACTION provenance is not bound to the newly consumed action"
                    )

        elapsed = max(0.0, sim_time_s - self._profile_started_at_s)
        if self._terminal_outcome != MacroTerminalOutcome.RUNNING:
            subphase = (
                MacroSubphase.COMPLETE
                if self._state_id == self.graph.success_state
                else MacroSubphase.SAFE_STOP
            )
            fraction = 1.0
            profile_id = ""
        elif self._hold_started_at_s is not None:
            subphase = MacroSubphase.HOLD
            fraction = 1.0
            profile_id = "" if self._active_profile is None else self._active_profile.profile_id
        elif self._active_profile is None:
            subphase = MacroSubphase.PRELOAD
            fraction = 1.0
            profile_id = ""
        else:
            profile_id = self._active_profile.profile_id
            if self._visible_cursor:
                frame_index = self._visible_keyframe_indices[
                    min(self._visible_cursor - 1, len(self._visible_keyframe_indices) - 1)
                ]
                subphase = self._active_profile.keyframes[frame_index].subphase
            else:
                subphase = self._active_profile.keyframes[0].subphase
            segment_count = len(self._active_profile.segment_bindings)
            fraction = (
                1.0
                if segment_count == 0
                else min(
                    1.0,
                    len(self._completed_segment_indices) / segment_count,
                )
            )
        decision = MacroDecision(
            macro_state=self._state_id,
            subphase=subphase,
            profile_id=profile_id,
            profile_source_version=(
                "" if self._active_profile is None else self._active_profile.source_version
            ),
            profile_strategy=(
                "" if self._active_profile is None else self._active_profile.strategy
            ),
            phase_elapsed_s=elapsed,
            profile_fraction=fraction,
            servo_targets_deg=dict(self._current_servo_targets),
            wheel_targets_rad_s=dict(self._current_wheel_targets),
            command_epoch=self._command_epoch,
            command_changed=command_event.command_changed,
            source_action_consumed=command_event.source_action_consumed,
            target_changed=command_event.target_changed,
            command_provenance=dict(command_event.command_provenance),
            segment_completion_control=dict(
                command_event.segment_completion_control
            ),
            transition_events=transition_events,
            reason=reason,
            guard_evidence=dict(self._last_guard_evidence),
            retry_count=self._retry_count,
            terminal=self._terminal_outcome != MacroTerminalOutcome.RUNNING,
            terminal_outcome=self._terminal_outcome,
        )
        self._last_decision_command_epoch = decision.command_epoch
        self._last_decision_servo_targets = dict(decision.servo_targets_deg)
        self._last_decision_wheel_targets = dict(decision.wheel_targets_rad_s)
        self._last_decision_consumed_source_action_count = len(
            self._consumed_source_actions
        )
        return decision
