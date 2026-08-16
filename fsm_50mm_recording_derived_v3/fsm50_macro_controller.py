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
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from command_model import (
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    command_limits_for_servo,
)
from completion_aware_segment import SegmentCompletionSpec, SegmentDecision

from .fsm50_macro_state_model import (
    FINAL_RECOVERY_FEEDBACK_LIMITS,
    MacroFSMGraph,
    MacroGuardKind,
    MacroStateId,
    MacroStateSpec,
    MacroSubphase,
    build_default_macro_graph,
)
from .fsm50_centroidal_support import (
    CentroidalSupportEvidence,
    EvidenceStatus,
    SupportModel,
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
LEG_SERVO_JOINTS: Mapping[str, tuple[str, str]] = {
    "FL": ("front_left_hip", "front_left_knee"),
    "FR": ("front_right_hip", "front_right_knee"),
    "RL": ("rear_left_hip", "rear_left_knee"),
    "RR": ("rear_right_hip", "rear_right_knee"),
}


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


FEEDBACK_RECOVERY_OBSERVATION_SCHEMA = "fsm50.feedback_recovery_observation.v1"
FEEDBACK_RECOVERY_OBSERVATION_PAYLOAD_KEYS = frozenset(
    {
        "sim_step",
        "physics_time_s",
        "observed_command_epoch",
        "n_plus_one_verified",
        "verified_command_epoch",
        "readback_servo_targets_deg",
        "readback_wheel_targets_rad_s",
        "measured_servo_positions_deg",
        "measured_servo_velocities_deg_s",
        "joint_limit_margin_deg",
        "base_position_m",
        "base_roll_rad",
        "base_pitch_rad",
        "base_angular_velocity_rad_s",
        "wheel_center_w_m",
        "wheel_front_face_clearance_m",
        "wheel_top_clearance_m",
        "obstacle_front_face_x_m",
        "obstacle_top_z_m",
        "body_crossed_front_face",
        "final_recoverable",
        "posture_complete",
    }
)


@dataclass(frozen=True)
class FeedbackRecoveryObservation:
    """Strict worker-injected readback for one current physics observation.

    Centroidal/support truth remains in :class:`CentroidalSupportEvidence`.
    This companion binds command N+1/readback and articulation values needed
    by the S10 diagnostic feedback loop; it must identify the same tick.
    """

    schema_version: str
    sim_step: int
    physics_time_s: float
    observed_command_epoch: int
    n_plus_one_verified: bool
    verified_command_epoch: int | None
    readback_servo_targets_deg: Mapping[str, float]
    readback_wheel_targets_rad_s: Mapping[str, float]
    measured_servo_positions_deg: Mapping[str, float]
    measured_servo_velocities_deg_s: Mapping[str, float]
    joint_limit_margin_deg: Mapping[str, float]
    base_position_m: tuple[float, float, float]
    base_roll_rad: float
    base_pitch_rad: float
    base_angular_velocity_rad_s: tuple[float, float, float]
    wheel_center_w_m: Mapping[str, tuple[float, float, float]]
    wheel_front_face_clearance_m: Mapping[str, float]
    wheel_top_clearance_m: Mapping[str, float]
    obstacle_front_face_x_m: float
    obstacle_top_z_m: float
    body_crossed_front_face: bool
    final_recoverable: bool
    posture_complete: bool
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != FEEDBACK_RECOVERY_OBSERVATION_SCHEMA:
            raise ValueError("feedback recovery observation schema mismatch")
        if type(self.sim_step) is not int or self.sim_step < 0:
            raise ValueError("feedback recovery sim_step must be an exact non-negative int")
        physics_time = _finite_scalar(
            self.physics_time_s, label="feedback_recovery.physics_time_s"
        )
        if physics_time < 0.0:
            raise ValueError("feedback recovery physics_time_s must be non-negative")
        if type(self.observed_command_epoch) is not int or self.observed_command_epoch < 0:
            raise ValueError("observed_command_epoch must be an exact non-negative int")
        if type(self.n_plus_one_verified) is not bool:
            raise ValueError("n_plus_one_verified must be an exact bool")
        if self.verified_command_epoch is not None and (
            type(self.verified_command_epoch) is not int
            or self.verified_command_epoch < 0
        ):
            raise ValueError("verified_command_epoch must be None or an exact non-negative int")
        if self.n_plus_one_verified != (self.verified_command_epoch is not None):
            raise ValueError("N+1 verification flag/epoch are inconsistent")
        for label, value in (
            ("body_crossed_front_face", self.body_crossed_front_face),
            ("final_recoverable", self.final_recoverable),
            ("posture_complete", self.posture_complete),
        ):
            if type(value) is not bool:
                raise ValueError(f"feedback_recovery.{label} must be an exact bool")
        readback = _target_map(
            self.readback_servo_targets_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.readback_servo_targets_deg",
            require_complete=True,
        )
        readback_wheels = _target_map(
            self.readback_wheel_targets_rad_s,
            WHEEL_JOINT_NAMES,
            label="feedback_recovery.readback_wheel_targets_rad_s",
            require_complete=True,
        )
        measured = _target_map(
            self.measured_servo_positions_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.measured_servo_positions_deg",
            require_complete=True,
        )
        velocities = _target_map(
            self.measured_servo_velocities_deg_s,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.measured_servo_velocities_deg_s",
            require_complete=True,
        )
        margins = _target_map(
            self.joint_limit_margin_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.joint_limit_margin_deg",
            require_complete=True,
        )
        if any(value < 0.0 for value in margins.values()):
            raise ValueError("joint_limit_margin_deg values must be non-negative")
        base_position = _strict_vector(
            self.base_position_m, 3, label="feedback_recovery.base_position_m"
        )
        base_roll = _finite_scalar(
            self.base_roll_rad, label="feedback_recovery.base_roll_rad"
        )
        base_pitch = _finite_scalar(
            self.base_pitch_rad, label="feedback_recovery.base_pitch_rad"
        )
        angular_velocity = _strict_vector(
            self.base_angular_velocity_rad_s,
            3,
            label="feedback_recovery.base_angular_velocity_rad_s",
        )
        centers = _leg_vector_map(self.wheel_center_w_m)
        front_clearance = _leg_scalar_map(
            self.wheel_front_face_clearance_m,
            label="feedback_recovery.wheel_front_face_clearance_m",
            require_all=True,
            allow_none=False,
        )
        top_clearance = _leg_scalar_map(
            self.wheel_top_clearance_m,
            label="feedback_recovery.wheel_top_clearance_m",
            require_all=True,
            allow_none=False,
        )
        obstacle_front = _finite_scalar(
            self.obstacle_front_face_x_m,
            label="feedback_recovery.obstacle_front_face_x_m",
        )
        obstacle_top = _finite_scalar(
            self.obstacle_top_z_m,
            label="feedback_recovery.obstacle_top_z_m",
        )
        object.__setattr__(self, "physics_time_s", physics_time)
        object.__setattr__(
            self, "readback_servo_targets_deg", MappingProxyType(readback)
        )
        object.__setattr__(
            self, "readback_wheel_targets_rad_s", MappingProxyType(readback_wheels)
        )
        object.__setattr__(
            self, "measured_servo_positions_deg", MappingProxyType(measured)
        )
        object.__setattr__(
            self, "measured_servo_velocities_deg_s", MappingProxyType(velocities)
        )
        object.__setattr__(
            self, "joint_limit_margin_deg", MappingProxyType(margins)
        )
        object.__setattr__(self, "base_position_m", base_position)
        object.__setattr__(self, "base_roll_rad", base_roll)
        object.__setattr__(self, "base_pitch_rad", base_pitch)
        object.__setattr__(self, "base_angular_velocity_rad_s", angular_velocity)
        object.__setattr__(self, "wheel_center_w_m", MappingProxyType(centers))
        object.__setattr__(
            self,
            "wheel_front_face_clearance_m",
            MappingProxyType({leg: float(front_clearance[leg]) for leg in LEGS}),
        )
        object.__setattr__(
            self,
            "wheel_top_clearance_m",
            MappingProxyType({leg: float(top_clearance[leg]) for leg in LEGS}),
        )
        object.__setattr__(self, "obstacle_front_face_x_m", obstacle_front)
        object.__setattr__(self, "obstacle_top_z_m", obstacle_top)
        if (
            type(self.payload_sha256) is not str
            or len(self.payload_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.payload_sha256)
            or self.payload_sha256 != _stable_sha256(self._payload_mapping())
        ):
            raise ValueError("feedback recovery payload SHA mismatch")

    def _payload_mapping(self) -> dict[str, Any]:
        return {
            "sim_step": self.sim_step,
            "physics_time_s": self.physics_time_s,
            "observed_command_epoch": self.observed_command_epoch,
            "n_plus_one_verified": self.n_plus_one_verified,
            "verified_command_epoch": self.verified_command_epoch,
            "readback_servo_targets_deg": dict(self.readback_servo_targets_deg),
            "readback_wheel_targets_rad_s": dict(
                self.readback_wheel_targets_rad_s
            ),
            "measured_servo_positions_deg": dict(self.measured_servo_positions_deg),
            "measured_servo_velocities_deg_s": dict(self.measured_servo_velocities_deg_s),
            "joint_limit_margin_deg": dict(self.joint_limit_margin_deg),
            "base_position_m": list(self.base_position_m),
            "base_roll_rad": self.base_roll_rad,
            "base_pitch_rad": self.base_pitch_rad,
            "base_angular_velocity_rad_s": list(self.base_angular_velocity_rad_s),
            "wheel_center_w_m": {
                leg: list(self.wheel_center_w_m[leg]) for leg in LEGS
            },
            "wheel_front_face_clearance_m": dict(
                self.wheel_front_face_clearance_m
            ),
            "wheel_top_clearance_m": dict(self.wheel_top_clearance_m),
            "obstacle_front_face_x_m": self.obstacle_front_face_x_m,
            "obstacle_top_z_m": self.obstacle_top_z_m,
            "body_crossed_front_face": self.body_crossed_front_face,
            "final_recoverable": self.final_recoverable,
            "posture_complete": self.posture_complete,
        }

    @classmethod
    def create(
        cls,
        *,
        sim_step: int,
        physics_time_s: float,
        observed_command_epoch: int,
        n_plus_one_verified: bool,
        verified_command_epoch: int | None,
        readback_servo_targets_deg: Mapping[str, float],
        readback_wheel_targets_rad_s: Mapping[str, float],
        measured_servo_positions_deg: Mapping[str, float],
        measured_servo_velocities_deg_s: Mapping[str, float],
        joint_limit_margin_deg: Mapping[str, float],
        base_position_m: Sequence[float],
        base_roll_rad: float,
        base_pitch_rad: float,
        base_angular_velocity_rad_s: Sequence[float],
        wheel_center_w_m: Mapping[str, Sequence[float]],
        wheel_front_face_clearance_m: Mapping[str, float],
        wheel_top_clearance_m: Mapping[str, float],
        obstacle_front_face_x_m: float,
        obstacle_top_z_m: float,
        body_crossed_front_face: bool,
        final_recoverable: bool,
        posture_complete: bool,
    ) -> "FeedbackRecoveryObservation":
        readback = _target_map(
            readback_servo_targets_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.readback_servo_targets_deg",
            require_complete=True,
        )
        readback_wheels = _target_map(
            readback_wheel_targets_rad_s,
            WHEEL_JOINT_NAMES,
            label="feedback_recovery.readback_wheel_targets_rad_s",
            require_complete=True,
        )
        measured = _target_map(
            measured_servo_positions_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.measured_servo_positions_deg",
            require_complete=True,
        )
        velocities = _target_map(
            measured_servo_velocities_deg_s,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.measured_servo_velocities_deg_s",
            require_complete=True,
        )
        margins = _target_map(
            joint_limit_margin_deg,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.joint_limit_margin_deg",
            require_complete=True,
        )
        centers = _leg_vector_map(wheel_center_w_m)
        front = _leg_scalar_map(
            wheel_front_face_clearance_m,
            label="feedback_recovery.wheel_front_face_clearance_m",
            require_all=True,
            allow_none=False,
        )
        top = _leg_scalar_map(
            wheel_top_clearance_m,
            label="feedback_recovery.wheel_top_clearance_m",
            require_all=True,
            allow_none=False,
        )
        payload = {
            "sim_step": sim_step,
            "physics_time_s": _finite_scalar(
                physics_time_s, label="feedback_recovery.physics_time_s"
            ),
            "observed_command_epoch": observed_command_epoch,
            "n_plus_one_verified": n_plus_one_verified,
            "verified_command_epoch": verified_command_epoch,
            "readback_servo_targets_deg": readback,
            "readback_wheel_targets_rad_s": readback_wheels,
            "measured_servo_positions_deg": measured,
            "measured_servo_velocities_deg_s": velocities,
            "joint_limit_margin_deg": margins,
            "base_position_m": list(
                _strict_vector(
                    base_position_m, 3, label="feedback_recovery.base_position_m"
                )
            ),
            "base_roll_rad": _finite_scalar(
                base_roll_rad, label="feedback_recovery.base_roll_rad"
            ),
            "base_pitch_rad": _finite_scalar(
                base_pitch_rad, label="feedback_recovery.base_pitch_rad"
            ),
            "base_angular_velocity_rad_s": list(
                _strict_vector(
                    base_angular_velocity_rad_s,
                    3,
                    label="feedback_recovery.base_angular_velocity_rad_s",
                )
            ),
            "wheel_center_w_m": {
                leg: list(centers[leg]) for leg in LEGS
            },
            "wheel_front_face_clearance_m": {
                leg: float(front[leg]) for leg in LEGS
            },
            "wheel_top_clearance_m": {leg: float(top[leg]) for leg in LEGS},
            "obstacle_front_face_x_m": _finite_scalar(
                obstacle_front_face_x_m,
                label="feedback_recovery.obstacle_front_face_x_m",
            ),
            "obstacle_top_z_m": _finite_scalar(
                obstacle_top_z_m, label="feedback_recovery.obstacle_top_z_m"
            ),
            "body_crossed_front_face": body_crossed_front_face,
            "final_recoverable": final_recoverable,
            "posture_complete": posture_complete,
        }
        return cls(
            schema_version=FEEDBACK_RECOVERY_OBSERVATION_SCHEMA,
            payload_sha256=_stable_sha256(payload),
            **payload,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload_sha256": self.payload_sha256,
            "payload": self._payload_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeedbackRecoveryObservation":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "payload_sha256", "payload"
        }:
            raise ValueError("feedback recovery observation root keys are not exact")
        if value["schema_version"] != FEEDBACK_RECOVERY_OBSERVATION_SCHEMA:
            raise ValueError("feedback recovery observation schema mismatch")
        payload = value["payload"]
        if (
            not isinstance(payload, Mapping)
            or set(payload) != FEEDBACK_RECOVERY_OBSERVATION_PAYLOAD_KEYS
        ):
            raise ValueError("feedback recovery observation payload keys are not exact")
        if value["payload_sha256"] != _stable_sha256(payload):
            raise ValueError("feedback recovery payload SHA mismatch")
        return cls(
            schema_version=FEEDBACK_RECOVERY_OBSERVATION_SCHEMA,
            payload_sha256=str(value["payload_sha256"]),
            **payload,
        )


@dataclass(frozen=True)
class MacroObservation:
    """Deployment-available subset consumed by the Macro FSM.

    COM/support guards consume exactly one SHA-bound current-tick centroidal
    envelope.  ``base_position_m`` and the legacy optional ``com_position_m``
    remain telemetry/body-progress fields only; neither can satisfy a COM or
    physical-support guard.
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
    centroidal_support_evidence: CentroidalSupportEvidence | Mapping[str, Any]
    feedback_recovery_observation: FeedbackRecoveryObservation | Mapping[str, Any]

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
        centroidal = (
            self.centroidal_support_evidence
            if isinstance(self.centroidal_support_evidence, CentroidalSupportEvidence)
            else CentroidalSupportEvidence.from_mapping(
                self.centroidal_support_evidence
            )
        )
        feedback = (
            self.feedback_recovery_observation
            if isinstance(
                self.feedback_recovery_observation,
                FeedbackRecoveryObservation,
            )
            else FeedbackRecoveryObservation.from_mapping(
                self.feedback_recovery_observation
            )
        )
        tolerance = max(1.0e-9, centroidal.physics_dt_s * 1.0e-6)
        if (
            feedback.sim_step != centroidal.sim_step
            or not math.isclose(
                feedback.physics_time_s,
                centroidal.physics_time_s,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ):
            raise ValueError(
                "feedback recovery observation and centroidal evidence are not the same tick"
            )
        feedback_cross_checks = {
            "servo target readback": (
                servo_targets == dict(feedback.readback_servo_targets_deg)
            ),
            "wheel target readback": (
                wheel_targets == dict(feedback.readback_wheel_targets_rad_s)
            ),
            "base position": base_position == feedback.base_position_m,
            "base roll": base_roll == feedback.base_roll_rad,
            "base pitch": base_pitch == feedback.base_pitch_rad,
            "base angular velocity": (
                angular_velocity == feedback.base_angular_velocity_rad_s
            ),
            "wheel centers": wheel_centers == dict(feedback.wheel_center_w_m),
            "front-face clearances": (
                front_clearances
                == dict(feedback.wheel_front_face_clearance_m)
            ),
            "top clearances": (
                top_clearances == dict(feedback.wheel_top_clearance_m)
            ),
            "obstacle front face": (
                obstacle_front == feedback.obstacle_front_face_x_m
            ),
            "obstacle top": obstacle_top == feedback.obstacle_top_z_m,
            "body crossed": (
                self.body_crossed_front_face == feedback.body_crossed_front_face
            ),
            "final recoverable": (
                self.final_recoverable == feedback.final_recoverable
            ),
            "posture complete": self.posture_complete == feedback.posture_complete,
        }
        failed_cross_checks = sorted(
            label for label, matches in feedback_cross_checks.items() if not matches
        )
        if failed_cross_checks:
            raise ValueError(
                "macro observation disagrees with SHA-bound feedback fields: "
                + ", ".join(failed_cross_checks)
            )

        # The dataclass is frozen for controller callers.  Canonical copies
        # also prevent later mutation of caller-owned dictionaries from
        # changing the observation after validation.
        object.__setattr__(self, "base_position_m", base_position)
        object.__setattr__(self, "base_roll_rad", base_roll)
        object.__setattr__(self, "base_pitch_rad", base_pitch)
        object.__setattr__(self, "base_angular_velocity_rad_s", angular_velocity)
        object.__setattr__(self, "com_position_m", com_position)
        object.__setattr__(self, "servo_targets_deg", MappingProxyType(servo_targets))
        object.__setattr__(self, "wheel_targets_rad_s", MappingProxyType(wheel_targets))
        object.__setattr__(self, "wheel_center_w_m", MappingProxyType(wheel_centers))
        object.__setattr__(self, "wheel_contact_classes", MappingProxyType(wheel_classes))
        object.__setattr__(self, "wheel_contact_load_n", MappingProxyType(wheel_loads))
        object.__setattr__(
            self, "wheel_front_face_clearance_m", MappingProxyType(front_clearances)
        )
        object.__setattr__(
            self, "wheel_top_clearance_m", MappingProxyType(top_clearances)
        )
        object.__setattr__(self, "obstacle_front_face_x_m", obstacle_front)
        object.__setattr__(self, "obstacle_top_z_m", obstacle_top)
        object.__setattr__(self, "centroidal_support_evidence", centroidal)
        object.__setattr__(self, "feedback_recovery_observation", feedback)

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
            "centroidal_support_evidence",
            "feedback_recovery_observation",
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
            centroidal_support_evidence=CentroidalSupportEvidence.from_mapping(
                values["centroidal_support_evidence"]
            )
            if not isinstance(
                values["centroidal_support_evidence"],
                CentroidalSupportEvidence,
            )
            else values["centroidal_support_evidence"],
            feedback_recovery_observation=FeedbackRecoveryObservation.from_mapping(
                values["feedback_recovery_observation"]
            )
            if not isinstance(
                values["feedback_recovery_observation"],
                FeedbackRecoveryObservation,
            )
            else values["feedback_recovery_observation"],
        )

    def true_com_position(self) -> tuple[float, float, float] | None:
        measurement = self.centroidal_support_evidence.whole_body_com
        return measurement.position_w_m if measurement.available else None

    def leg_crossed(self, leg: str) -> bool:
        return bool(
            self.feedback_recovery_observation.wheel_front_face_clearance_m[leg]
            > 0.0
        )

    def leg_airborne(self, leg: str) -> bool:
        row = self.centroidal_support_evidence.wheel_contacts.by_leg()[leg]
        top_clearance = (
            self.feedback_recovery_observation.wheel_top_clearance_m[leg]
        )
        return bool(
            (
                row.evidence_available
                and not row.measurement.active
                and row.measurement.surface_kind == "AIR"
            )
            or top_clearance >= 0.003
        )

    def leg_top(self, leg: str) -> bool:
        row = self.centroidal_support_evidence.wheel_contacts.by_leg()[leg]
        return bool(
            row.evidence_available
            and row.measurement.active
            and row.measurement.surface_kind == "OBSTACLE_TOP"
            and self.leg_crossed(leg)
        )

    def leg_support_candidate(self, leg: str) -> bool:
        return bool(
            self.centroidal_support_evidence.wheel_contacts.by_leg()[
                leg
            ].support_qualified
        )


class MacroTerminalOutcome(str, Enum):
    RUNNING = "RUNNING"
    TASK_SUCCESS = "TASK_SUCCESS"
    TASK_SUCCESS_POSTURE_INCOMPLETE = "TASK_SUCCESS_POSTURE_INCOMPLETE"
    SAFE_STOP = "SAFE_STOP"


class FeedbackRecoveryStage(str, Enum):
    REFERENCE_PROFILE = "REFERENCE_PROFILE"
    OBSERVE_AIR = "OBSERVE_AIR"
    SAFE_PROBE = "SAFE_PROBE"
    RETURN_TO_REFERENCE = "RETURN_TO_REFERENCE"
    SELECT_DESCENT = "SELECT_DESCENT"
    INCREMENT = "INCREMENT"
    CONTACT_DWELL = "CONTACT_DWELL"
    SETTLE = "SETTLE"
    COMPLETE = "COMPLETE"
    POSTURE_INCOMPLETE = "POSTURE_INCOMPLETE"


class FeedbackRecoveryAction(str, Enum):
    CONSERVATIVE_DIAGNOSTIC_PROBE = "CONSERVATIVE_DIAGNOSTIC_PROBE"
    RETURN_TO_IMMUTABLE_REFERENCE = "RETURN_TO_IMMUTABLE_REFERENCE"
    BOUNDED_DESCENT_INCREMENT = "BOUNDED_DESCENT_INCREMENT"


FEEDBACK_RECOVERY_EVIDENCE_BINDING_SCHEMA = (
    "fsm50.feedback_recovery_evidence_binding.v1"
)
FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA = "fsm50.feedback_recovery_configuration.v1"
FEEDBACK_RECOVERY_TARGET_MAP_SCHEMA = "fsm50.feedback_recovery_target_map.v1"


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
        "FEEDBACK_RECOVERY",
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
        "recovery_stage",
        "recovery_action",
        "recovery_evidence_sha256",
        "recovery_centroidal_evidence_sha256",
        "recovery_feedback_observation_sha256",
        "recovery_target_map_sha256",
        "recovery_direction_sign",
        "recovery_attempt",
        "recovery_leg",
        "recovery_joint",
        "recovery_configuration_sha256",
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

    recovery_keys = {
        "recovery_stage": "",
        "recovery_action": "",
        "recovery_evidence_sha256": "",
        "recovery_centroidal_evidence_sha256": "",
        "recovery_feedback_observation_sha256": "",
        "recovery_target_map_sha256": "",
        "recovery_direction_sign": None,
        "recovery_attempt": None,
        "recovery_leg": "",
        "recovery_joint": "",
        "recovery_configuration_sha256": "",
    }
    if kind == "FEEDBACK_RECOVERY":
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
                    raise ValueError(f"decision.command_provenance.{key} must be empty")
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"{key} must be empty for FEEDBACK_RECOVERY")
        stage, action = values["recovery_stage"], values["recovery_action"]
        leg, joint = values["recovery_leg"], values["recovery_joint"]
        evidence = values["recovery_evidence_sha256"]
        centroidal_evidence = values["recovery_centroidal_evidence_sha256"]
        feedback_evidence = values["recovery_feedback_observation_sha256"]
        target_map = values["recovery_target_map_sha256"]
        direction_sign = values["recovery_direction_sign"]
        configuration = values["recovery_configuration_sha256"]
        attempt = values["recovery_attempt"]
        stage_action = {
            FeedbackRecoveryStage.SAFE_PROBE.value:
                FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE.value,
            FeedbackRecoveryStage.RETURN_TO_REFERENCE.value:
                FeedbackRecoveryAction.RETURN_TO_IMMUTABLE_REFERENCE.value,
            FeedbackRecoveryStage.INCREMENT.value:
                FeedbackRecoveryAction.BOUNDED_DESCENT_INCREMENT.value,
        }
        if stage not in stage_action or action != stage_action[stage]:
            raise ValueError("FEEDBACK_RECOVERY stage/action pair is invalid")
        if (
            leg not in LEGS
            or joint not in SERVO_JOINT_NAMES
            or joint not in LEG_SERVO_JOINTS[leg]
        ):
            raise ValueError("FEEDBACK_RECOVERY leg/joint identity is invalid")
        for label, digest in (
            ("recovery_evidence_sha256", evidence),
            ("recovery_centroidal_evidence_sha256", centroidal_evidence),
            ("recovery_feedback_observation_sha256", feedback_evidence),
            ("recovery_target_map_sha256", target_map),
            ("recovery_configuration_sha256", configuration),
        ):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        expected_evidence = _stable_sha256(
            {
                "schema_version": FEEDBACK_RECOVERY_EVIDENCE_BINDING_SCHEMA,
                "centroidal_support_evidence_sha256": centroidal_evidence,
                "feedback_recovery_observation_sha256": feedback_evidence,
            }
        )
        if evidence != expected_evidence:
            raise ValueError("FEEDBACK_RECOVERY composite evidence SHA mismatch")
        if direction_sign not in {-1, 1} or type(direction_sign) is not int:
            raise ValueError("FEEDBACK_RECOVERY direction sign must be exact -1/+1")
        if type(attempt) is not int or attempt <= 0:
            raise ValueError("FEEDBACK_RECOVERY attempt must be positive")
        return {
            "kind": kind,
            **expected_empty,
            "recovery_stage": stage,
            "recovery_action": action,
            "recovery_evidence_sha256": evidence,
            "recovery_centroidal_evidence_sha256": centroidal_evidence,
            "recovery_feedback_observation_sha256": feedback_evidence,
            "recovery_target_map_sha256": target_map,
            "recovery_direction_sign": direction_sign,
            "recovery_attempt": attempt,
            "recovery_leg": leg,
            "recovery_joint": joint,
            "recovery_configuration_sha256": configuration,
        }

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
            **recovery_keys,
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
            **recovery_keys,
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
    for key, expected in recovery_keys.items():
        if values[key] != expected:
            raise ValueError(f"decision.command_provenance.{key} must be empty for SOURCE_ACTION")
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
        **recovery_keys,
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
            "recovery_stage": "",
            "recovery_action": "",
            "recovery_evidence_sha256": "",
            "recovery_centroidal_evidence_sha256": "",
            "recovery_feedback_observation_sha256": "",
            "recovery_target_map_sha256": "",
            "recovery_direction_sign": None,
            "recovery_attempt": None,
            "recovery_leg": "",
            "recovery_joint": "",
            "recovery_configuration_sha256": "",
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
        if provenance["kind"] == "FEEDBACK_RECOVERY":
            expected_target_map_sha256 = _stable_sha256(
                {
                    "schema_version": FEEDBACK_RECOVERY_TARGET_MAP_SCHEMA,
                    "servo_targets_deg": servo_targets,
                    "wheel_targets_rad_s": wheel_targets,
                }
            )
            if (
                provenance["recovery_target_map_sha256"]
                != expected_target_map_sha256
            ):
                raise ValueError(
                    "FEEDBACK_RECOVERY target-map SHA does not match decision 8+4 map"
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


def _feedback_recovery_command_event(
    *,
    stage: FeedbackRecoveryStage,
    action: FeedbackRecoveryAction,
    centroidal_evidence_sha256: str,
    feedback_observation_sha256: str,
    target_map_sha256: str,
    direction_sign: int,
    attempt: int,
    leg: str,
    joint: str,
    configuration_sha256: str,
) -> _CommandEvent:
    evidence_sha256 = _stable_sha256(
        {
            "schema_version": FEEDBACK_RECOVERY_EVIDENCE_BINDING_SCHEMA,
            "centroidal_support_evidence_sha256": centroidal_evidence_sha256,
            "feedback_recovery_observation_sha256": feedback_observation_sha256,
        }
    )
    provenance = dict(_empty_command_provenance())
    provenance.update(
        kind="FEEDBACK_RECOVERY",
        recovery_stage=stage.value,
        recovery_action=action.value,
        recovery_evidence_sha256=evidence_sha256,
        recovery_centroidal_evidence_sha256=centroidal_evidence_sha256,
        recovery_feedback_observation_sha256=feedback_observation_sha256,
        recovery_target_map_sha256=target_map_sha256,
        recovery_direction_sign=direction_sign,
        recovery_attempt=attempt,
        recovery_leg=leg,
        recovery_joint=joint,
        recovery_configuration_sha256=configuration_sha256,
    )
    return _CommandEvent(
        False,
        True,
        True,
        _canonical_command_provenance(provenance),
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
        self._entry_position_source = "BASE_POSITION_BODY_PROGRESS_ONLY"
        self._entry_true_com_position: tuple[float, float, float] | None = None
        self._target_com_unit_xy: tuple[float, float] | None = None
        self._last_centroidal_sim_step: int | None = None
        self._centroidal_body_identity: tuple[
            tuple[str, ...], tuple[float, ...], float
        ] | None = None
        self._s8_release_open = False
        self._s8_release_baseline_com_position: tuple[float, float, float] | None = None
        self._s8_release_baseline_evidence_sha256 = ""
        self._feedback_stage = FeedbackRecoveryStage.REFERENCE_PROFILE
        self._feedback_pending_epoch: int | None = None
        self._feedback_pending_sim_step: int | None = None
        self._feedback_pending_action = ""
        self._feedback_pending_leg = ""
        self._feedback_pending_joint = ""
        self._feedback_pending_sign = 0
        self._feedback_reference_targets: dict[str, float] = {}
        self._feedback_configuration_sha256 = ""
        self._feedback_active_leg = ""
        self._feedback_joint_index = 0
        self._feedback_probe_sign = 1
        self._feedback_probe_results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        self._feedback_selected_signs: dict[tuple[str, str, str], int] = {}
        self._feedback_selected_joint = ""
        self._feedback_selected_sign = 0
        self._feedback_baseline: dict[str, Any] = {}
        self._feedback_action_count = 0
        self._feedback_increment_count = 0
        self._feedback_increment_count_by_leg = {leg: 0 for leg in LEGS}
        self._feedback_contact_started_at_s: float | None = None
        self._feedback_settle_started_at_s: float | None = None
        self._feedback_settle_window_started_at_s: float | None = None
        self._feedback_settle_posture_incomplete = False
        self._feedback_exhaustion_reason = ""
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
            "feedback_recovery_stage": self._feedback_stage.value,
            "feedback_recovery_pending_epoch": self._feedback_pending_epoch,
            "feedback_recovery_action_count": self._feedback_action_count,
            "feedback_recovery_exhaustion_reason": self._feedback_exhaustion_reason,
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
        self._require_current_evidence(observation, float(sim_time_s), reset=True)
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
        try:
            self._require_current_evidence(observed, now, reset=False)
        except ValueError as exc:
            return self._safe_stop(observed, now, f"centroidal/feedback evidence failure: {exc}")
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
                observation=observed,
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

        if state.state_id == MacroStateId.S10_POSTURE_RECOVERY and timeline_complete:
            try:
                feedback_event, feedback_reason = self._advance_feedback_recovery(
                    observed,
                    sim_time_s=now,
                    source_cursor_permit=source_cursor_permit,
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                return self._safe_stop(
                    observed,
                    now,
                    f"feedback recovery contract/safety failure: {exc}",
                )
            self._last_reason = feedback_reason
            guard = self._evaluate_guard(
                state, observed, timeline_complete=timeline_complete
            )
            self._last_guard_evidence = guard.evidence
            if feedback_event.target_changed:
                return self._decision(
                    observed,
                    sim_time_s=now,
                    command_event=feedback_event,
                    transition_events=(),
                    reason=feedback_reason,
                )
            if self._feedback_stage == FeedbackRecoveryStage.POSTURE_INCOMPLETE:
                support = self._physical_support_snapshot(observed)
                feedback = observed.feedback_recovery_observation
                velocities_settled = bool(
                    max(
                        abs(value)
                        for value in feedback.measured_servo_velocities_deg_s.values()
                    )
                    <= float(
                        FINAL_RECOVERY_FEEDBACK_LIMITS[
                            "maximum_abs_joint_velocity_deg_s"
                        ]
                    )
                    and max(
                        abs(value)
                        for value in feedback.base_angular_velocity_rad_s
                    )
                    <= float(
                        FINAL_RECOVERY_FEEDBACK_LIMITS[
                            "maximum_abs_body_angular_velocity_rad_s"
                        ]
                    )
                )
                if (
                    feedback.final_recoverable
                    and feedback.body_crossed_front_face
                    and support["support_viable"]
                    and support["support_wrench_proven"]
                    and velocities_settled
                ):
                    return self._finish_success(
                        observed,
                        now,
                        state.state_id,
                        force_posture_incomplete=True,
                    )
                return self._safe_stop(
                    observed,
                    now,
                    "feedback recovery exhausted without a recoverable supported posture",
                )
            if self._feedback_stage != FeedbackRecoveryStage.COMPLETE:
                return self._decision(
                    observed,
                    sim_time_s=now,
                    command_event=_no_command_event(),
                    transition_events=(),
                    reason=feedback_reason,
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
                    transition_command_event = self._start_next_segment(observed)
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

    def _require_current_evidence(
        self,
        observation: MacroObservation,
        sim_time_s: float,
        *,
        reset: bool,
    ) -> None:
        evidence = observation.centroidal_support_evidence
        feedback = observation.feedback_recovery_observation
        tolerance = max(1.0e-9, evidence.physics_dt_s * 1.0e-6)
        if not math.isclose(
            evidence.physics_time_s,
            sim_time_s,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("centroidal evidence sim_time does not match controller callback")
        if feedback.sim_step != evidence.sim_step or not math.isclose(
            feedback.physics_time_s,
            sim_time_s,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("feedback evidence does not match the current centroidal tick")
        com = evidence.whole_body_com
        identity = (
            tuple(com.body_names),
            tuple(float(value) for value in com.body_masses_kg),
            float(com.total_mass_kg) if com.total_mass_kg is not None else 0.0,
        )
        if reset:
            self._centroidal_body_identity = identity
        elif self._centroidal_body_identity != identity:
            raise ValueError("whole-body COM identity/masses changed across controller ticks")
        if not reset and self._last_centroidal_sim_step is not None and (
            evidence.sim_step <= self._last_centroidal_sim_step
        ):
            raise ValueError("centroidal evidence sim_step is stale or duplicated")
        self._last_centroidal_sim_step = evidence.sim_step

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
                "recovery_stage": "",
                "recovery_action": "",
                "recovery_evidence_sha256": "",
                "recovery_centroidal_evidence_sha256": "",
                "recovery_feedback_observation_sha256": "",
                "recovery_target_map_sha256": "",
                "recovery_direction_sign": None,
                "recovery_attempt": None,
                "recovery_leg": "",
                "recovery_joint": "",
                "recovery_configuration_sha256": "",
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
        self._entry_body_position = observation.base_position_m
        self._entry_position_source = "BASE_POSITION_BODY_PROGRESS_ONLY"
        self._entry_true_com_position = observation.true_com_position()
        self._target_com_unit_xy = self._target_direction_unit(
            self.graph.get(state_id),
            observation,
            self._entry_true_com_position,
        )
        if state_id == MacroStateId.S7_PRE_RL_SUPPORT_SETUP:
            self._s8_release_baseline_com_position = observation.true_com_position()
            self._s8_release_baseline_evidence_sha256 = (
                observation.centroidal_support_evidence.payload_sha256
            )
        if state_id == MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE:
            self._s8_release_open = False
        if state_id == MacroStateId.S10_POSTURE_RECOVERY:
            self._feedback_stage = FeedbackRecoveryStage.REFERENCE_PROFILE
            self._feedback_settle_started_at_s = None
            self._feedback_settle_window_started_at_s = None
            self._feedback_settle_posture_incomplete = False
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

        position = observation.base_position_m
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
        position: tuple[float, float, float] | None,
    ) -> tuple[float, float] | None:
        leg = state.completion_guard.target_com_leg
        if not leg or position is None:
            return None
        target = observation.feedback_recovery_observation.wheel_center_w_m.get(leg)
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

    def _start_next_segment(self, observation: MacroObservation) -> _CommandEvent:
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
        if (
            self._state_id == MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE
            and frame.physical_phase
            == self.graph.get(self._state_id).completion_guard.release_physical_phase
            and not self._s8_release_open
        ):
            projected, source = self._s8_release_progress(observation)
            support = self._physical_support_snapshot(observation)
            release = bool(
                source == "WHOLE_BODY_COM_FROM_S7_ENTRY_CURRENT_TICK"
                and projected
                >= self.graph.get(self._state_id).completion_guard.minimum_com_displacement_m
                and self._measured_leg_unloaded(observation, "RL")
                and support["support_viable"]
                and support["support_wrench_proven"]
            )
            self._last_guard_evidence.update(
                s8_release_phase=frame.physical_phase,
                s8_release_true_com_projected_displacement_m=projected,
                s8_release_com_baseline_evidence_sha256=(
                    self._s8_release_baseline_evidence_sha256
                ),
                s8_release_rl_measured_unloaded=self._measured_leg_unloaded(
                    observation, "RL"
                ),
                s8_release_support_viable=support["support_viable"],
                s8_release_wrench_proven=support["support_wrench_proven"],
                s8_release_open=release,
                s8_release_body_root_proxy_eligible=False,
            )
            if not release:
                changed = self._zero_wheels_for_boundary()
                return _non_source_command_event(
                    "HOLD_ZERO_WHEELS", target_changed=changed
                )
            self._s8_release_open = True
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
        observation: MacroObservation,
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
        return self._start_next_segment(observation)

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
        current = observation.base_position_m
        source = "BASE_POSITION_BODY_PROGRESS_ONLY"
        forward = current[0] - self._entry_body_position[0]
        projected = 0.0
        true_com = observation.true_com_position()
        if (
            self._target_com_unit_xy is not None
            and self._entry_true_com_position is not None
            and true_com is not None
        ):
            dx = true_com[0] - self._entry_true_com_position[0]
            dy = true_com[1] - self._entry_true_com_position[1]
            projected = dx * self._target_com_unit_xy[0] + dy * self._target_com_unit_xy[1]
            source = "WHOLE_BODY_COM_CURRENT_TICK"
        return forward, projected, source

    def _s8_release_progress(
        self, observation: MacroObservation
    ) -> tuple[float, str]:
        baseline = self._s8_release_baseline_com_position
        current = observation.true_com_position()
        target = (
            observation.feedback_recovery_observation.wheel_center_w_m.get("FR")
        )
        if baseline is None or current is None or target is None:
            return 0.0, "UNAVAILABLE"
        tx, ty = target[0] - baseline[0], target[1] - baseline[1]
        norm = math.hypot(tx, ty)
        if norm <= 1.0e-9:
            return 0.0, "UNAVAILABLE"
        dx, dy = current[0] - baseline[0], current[1] - baseline[1]
        return (
            dx * tx / norm + dy * ty / norm,
            "WHOLE_BODY_COM_FROM_S7_ENTRY_CURRENT_TICK",
        )

    @staticmethod
    def _physical_support_snapshot(
        observation: MacroObservation,
    ) -> dict[str, Any]:
        evidence = observation.centroidal_support_evidence
        by_leg = evidence.wheel_contacts.by_leg()
        qualified = tuple(
            leg for leg in LEGS if by_leg[leg].support_qualified
        )
        region = evidence.support_region
        validated = tuple(region.support_legs)
        support_set_bound = bool(validated == qualified)
        wrench = evidence.contact_wrench_feasibility
        wrench_proven = bool(
            wrench.status == EvidenceStatus.PROVEN
            and wrench.proven_feasible
            and wrench.physics_tick == evidence.sim_step
        )
        hull_viable = bool(
            support_set_bound
            and len(validated) in {3, 4}
            and region.status == EvidenceStatus.PROVEN
            and region.model == SupportModel.STRICT_COPLANAR_CONVEX_HULL
            and region.signed_margin_m is not None
            and region.signed_margin_m >= 0.0
        )
        diagonal_viable = bool(
            support_set_bound
            and len(validated) == 2
            and frozenset(validated)
            in {frozenset(("FL", "RR")), frozenset(("FR", "RL"))}
            and region.status == EvidenceStatus.PROVEN
            and region.model == SupportModel.DIAGONAL_LINE_SEGMENT
            and region.diagonal
            == ("FL_RR" if frozenset(validated) == frozenset(("FL", "RR")) else "FR_RL")
            and region.between_contacts is True
            and region.finite_patch_approximation
            and region.corridor_signed_margin_m is not None
            and region.corridor_signed_margin_m >= 0.0
        )
        multi_height_wrench_viable = bool(
            support_set_bound
            and len(validated) in {3, 4}
            and region.status == EvidenceStatus.NOT_PROVEN
            and region.model == SupportModel.MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED
            and wrench_proven
        )
        support_margin = (
            region.signed_margin_m
            if hull_viable
            else region.corridor_signed_margin_m
            if diagonal_viable
            else None
        )
        return {
            "centroidal_evidence_sha256": evidence.payload_sha256,
            "centroidal_sim_step": evidence.sim_step,
            "whole_body_com_available": evidence.whole_body_com.available,
            "support_region_status": evidence.support_region.status.value,
            "support_model": evidence.support_region.model.value,
            "validated_support_legs": list(validated),
            "support_qualified_legs": list(qualified),
            "support_set_bound_to_current_qualified_contacts": support_set_bound,
            "support_viable": bool(
                evidence.whole_body_com.available
                and len(qualified) >= 2
                and (hull_viable or diagonal_viable or multi_height_wrench_viable)
            ),
            "support_hull_viable": hull_viable,
            "support_diagonal_corridor_viable": diagonal_viable,
            "support_multi_height_wrench_viable": multi_height_wrench_viable,
            "support_stability_margin_m": support_margin,
            "support_wrench_proven": wrench_proven,
            "wrench_force_residual_norm_n": (
                evidence.contact_wrench_feasibility.force_residual_norm_n
            ),
            "wrench_moment_residual_norm_nm": (
                evidence.contact_wrench_feasibility.moment_residual_norm_nm
            ),
            "wrench_maximum_friction_utilization": (
                evidence.contact_wrench_feasibility.maximum_friction_utilization
            ),
            "primary_diagonal": evidence.diagonal_support.primary_diagonal,
            "primary_diagonal_status": evidence.diagonal_support.status.value,
            "per_leg_normal_load_n": {
                leg: by_leg[leg].normal_load_n for leg in LEGS
            },
            "per_leg_support_qualified": {
                leg: by_leg[leg].support_qualified for leg in LEGS
            },
        }

    @staticmethod
    def _measured_leg_unloaded(
        observation: MacroObservation, leg: str
    ) -> bool:
        evidence = observation.centroidal_support_evidence
        row = evidence.wheel_contacts.by_leg()[leg]
        maximum = evidence.wheel_contacts.thresholds.maximum_active_leg_load_n
        return bool(
            row.evidence_available
            and row.normal_load_n is not None
            and row.normal_load_n <= maximum
        )

    @staticmethod
    def _four_top_contacts_ready(observation: MacroObservation) -> bool:
        evidence = observation.centroidal_support_evidence
        dwell_required = float(FINAL_RECOVERY_FEEDBACK_LIMITS["contact_dwell_s"])
        for leg, row in evidence.wheel_contacts.by_leg().items():
            measurement = row.measurement
            if not (
                row.evidence_available
                and row.support_qualified
                and measurement.active
                and measurement.surface_kind == "OBSTACLE_TOP"
                and measurement.surface_dwell_verified
                and measurement.dwell_s is not None
                and measurement.dwell_s >= dwell_required
                and row.normal_load_n is not None
                and row.normal_load_n
                >= evidence.wheel_contacts.thresholds.minimum_normal_force_n
            ):
                return False
        return True

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
            **self._physical_support_snapshot(observation),
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
                and self._measured_leg_unloaded(observation, guard.active_leg)
            )
            com_available = observation.true_com_position() is not None
            displacement_ready = bool(
                com_available
                and position_source == "WHOLE_BODY_COM_CURRENT_TICK"
                and projected >= guard.minimum_com_displacement_m
            )
            viable_support = bool(evidence["support_viable"])
            evidence.update(
                active_leg_measured_unloaded=active_unloaded,
                minimum_com_displacement_m=guard.minimum_com_displacement_m,
                true_com_displacement_ready=displacement_ready,
                viable_current_support=viable_support,
                body_root_proxy_guard_eligible=False,
            )
            physical = bool((active_unloaded or displacement_ready) and viable_support)
            reason = (
                "true COM shifted or active leg measured unloaded with viable current support"
                if physical
                else "strict COM/unload and viable-support guard is not proven"
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
            diagonal = observation.centroidal_support_evidence.diagonal_support
            primary = bool(
                diagonal.status == EvidenceStatus.PROVEN
                and diagonal.primary_diagonal
                == "_".join(guard.required_primary_diagonal)
                and diagonal.active_swing_leg == guard.active_leg
                and evidence["support_viable"]
            )
            qualified = set(evidence["support_qualified_legs"])
            validated = set(evidence["validated_support_legs"])
            required = set(guard.required_support_legs)
            alternate = bool(
                evidence["support_viable"]
                and len(validated) >= 3
                and required.issubset(validated)
                and validated.issubset(qualified)
            )
            wrench = bool(evidence["support_wrench_proven"])
            physical = bool((primary or alternate) and wrench)
            evidence.update(
                required_primary_diagonal=list(guard.required_primary_diagonal),
                required_support_legs=list(guard.required_support_legs),
                required_primary_diagonal_proven=primary,
                declaratively_validated_alternate_support=alternate,
                geometry_only_support_eligible=False,
            )
            reason = "validated support set/primary diagonal and wrench are proven" if physical else "strict support/diagonal/wrench proof unavailable"
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
            feedback = observation.feedback_recovery_observation
            body_crossed = feedback.body_crossed_front_face
            contacts_ready = self._four_top_contacts_ready(observation)
            qd_settled = max(
                abs(value) for value in feedback.measured_servo_velocities_deg_s.values()
            ) <= float(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_abs_joint_velocity_deg_s"])
            angular_settled = max(
                abs(value) for value in feedback.base_angular_velocity_rad_s
            ) <= float(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_abs_body_angular_velocity_rad_s"])
            physical = bool(
                feedback.final_recoverable
                and (body_crossed if guard.require_body_crossed else True)
                and contacts_ready
                and evidence["support_viable"]
                and evidence["support_wrench_proven"]
                and qd_settled
                and angular_settled
                and self._feedback_stage == FeedbackRecoveryStage.COMPLETE
            )
            evidence.update(
                body_crossed_front_face=body_crossed,
                final_recoverable=feedback.final_recoverable,
                posture_complete=feedback.posture_complete,
                four_top_load_dwell_contacts=contacts_ready,
                joint_velocity_settled=qd_settled,
                body_angular_velocity_settled=angular_settled,
                feedback_recovery_stage=self._feedback_stage.value,
            )
            reason = "task crossed and final state is recoverable" if physical else reason
        return _GuardResult(physical, evidence, reason)

    def _recovery_configuration_identity(
        self, observation: MacroObservation, leg: str
    ) -> str:
        feedback = observation.feedback_recovery_observation
        profile = self._active_profile
        return _stable_sha256(
            {
                "schema_version": FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA,
                "leg": leg,
                "macro_state": self._state_id.value,
                "selected_source_version": self._source_version,
                "reference_profile_id": "" if profile is None else profile.profile_id,
                "reference_profile_source_version": (
                    "" if profile is None else profile.source_version
                ),
                "reference_profile_source_plan_sha256": (
                    "" if profile is None else profile.source_plan_sha256
                ),
                "centroidal_evidence_sha256": (
                    observation.centroidal_support_evidence.payload_sha256
                ),
                "feedback_observation_sha256": feedback.payload_sha256,
                "servo_reference_targets_deg": dict(
                    self._feedback_reference_targets
                ),
                "measured_servo_positions_deg": dict(
                    feedback.measured_servo_positions_deg
                ),
                "wheel_center_w_m": {
                    item: list(feedback.wheel_center_w_m[item]) for item in LEGS
                },
                "body_crossed_front_face": feedback.body_crossed_front_face,
                "final_recoverable": feedback.final_recoverable,
                "posture_complete": feedback.posture_complete,
            }
        )

    def _recovery_safety_failure(
        self, observation: MacroObservation
    ) -> str:
        state = self.graph.get(MacroStateId.S10_POSTURE_RECOVERY)
        support = self._physical_support_snapshot(observation)
        feedback = observation.feedback_recovery_observation
        failures: list[str] = []
        if (
            abs(feedback.base_roll_rad)
            > state.completion_guard.maximum_abs_roll_rad
            or abs(feedback.base_pitch_rad)
            > state.completion_guard.maximum_abs_pitch_rad
        ):
            failures.append("feedback recovery attitude safety bound exceeded")
        if not support["support_viable"] or not support["support_wrench_proven"]:
            failures.append("feedback recovery lost proven support/wrench feasibility")
        minimum_margin = min(feedback.joint_limit_margin_deg.values())
        required_margin = float(
            FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"]
        )
        if minimum_margin < required_margin:
            failures.append("feedback recovery joint-limit margin is unsafe")
        return "; ".join(failures)

    def _probe_preserves_baseline(
        self, observation: MacroObservation, joint: str
    ) -> tuple[bool, tuple[str, ...]]:
        if not self._feedback_baseline:
            return False, ("immutable feedback baseline is unavailable",)
        support = self._physical_support_snapshot(observation)
        feedback = observation.feedback_recovery_observation
        reasons: list[str] = []
        baseline_support = set(self._feedback_baseline["support_qualified_legs"])
        current_support = set(support["support_qualified_legs"])
        if not (baseline_support - {self._feedback_active_leg}).issubset(
            current_support
        ):
            reasons.append("qualified support set worsened")
        if (
            abs(feedback.base_roll_rad)
            > self._feedback_baseline["abs_roll_rad"] + 1.0e-9
            or abs(feedback.base_pitch_rad)
            > self._feedback_baseline["abs_pitch_rad"] + 1.0e-9
        ):
            reasons.append("body attitude worsened")
        if (
            feedback.joint_limit_margin_deg[joint] + 1.0e-9
            < self._feedback_baseline["joint_limit_margin_deg"][joint]
        ):
            reasons.append("probed-joint limit margin worsened")
        baseline_margin = self._feedback_baseline["support_stability_margin_m"]
        current_margin = support["support_stability_margin_m"]
        if (
            baseline_margin is not None
            and current_margin is not None
            and current_margin + 1.0e-9 < baseline_margin
        ):
            reasons.append("support-region stability margin worsened")
        for key, label in (
            ("wrench_force_residual_norm_n", "force residual"),
            ("wrench_moment_residual_norm_nm", "moment residual"),
            ("wrench_maximum_friction_utilization", "friction utilization"),
        ):
            baseline_value = self._feedback_baseline[key]
            current_value = support[key]
            if (
                baseline_value is not None
                and current_value is not None
                and current_value > baseline_value + 1.0e-9
            ):
                reasons.append(f"contact-wrench {label} worsened")
        return not reasons, tuple(reasons)

    def _recovery_candidate_legs(
        self, observation: MacroObservation
    ) -> tuple[str, ...]:
        by_leg = observation.centroidal_support_evidence.wheel_contacts.by_leg()
        centers = observation.feedback_recovery_observation.wheel_center_w_m
        candidates: list[str] = []
        for leg in ("FR", "FL", "RR", "RL"):
            row = by_leg[leg]
            landed = bool(
                row.support_qualified
                and row.measurement.active
                and row.measurement.surface_kind == "OBSTACLE_TOP"
            )
            if landed:
                continue
            feedback = observation.feedback_recovery_observation
            crossed = centers[leg][0] > feedback.obstacle_front_face_x_m
            near_top = centers[leg][2] >= feedback.obstacle_top_z_m - 0.02
            if (
                row.evidence_available
                and not row.measurement.active
                and row.measurement.surface_kind == "AIR"
                and crossed
                and near_top
            ):
                candidates.append(leg)
        return tuple(candidates)

    def _begin_recovery_leg(
        self, observation: MacroObservation, leg: str
    ) -> None:
        self._feedback_active_leg = leg
        self._feedback_joint_index = 0
        self._feedback_probe_sign = 1
        self._feedback_increment_count = 0
        self._feedback_reference_targets = dict(self._current_servo_targets)
        # Candidate discovery and permit cadence can precede the first physical
        # probe by several observations.  Do not claim a configuration identity
        # until a probe is actually dispatched from a worker-visible current
        # observation.  The reference target map remains immutable meanwhile.
        self._feedback_configuration_sha256 = ""
        self._feedback_baseline = {}
        self._feedback_stage = FeedbackRecoveryStage.SAFE_PROBE

    def _freeze_recovery_configuration(
        self, observation: MacroObservation
    ) -> None:
        if self._feedback_configuration_sha256 or self._feedback_baseline:
            raise RuntimeError("feedback recovery configuration was already frozen")
        if self._feedback_active_leg not in LEGS:
            raise RuntimeError("feedback recovery active-leg identity is unavailable")
        _target_map(
            self._feedback_reference_targets,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.reference_targets",
            require_complete=True,
        )
        feedback = observation.feedback_recovery_observation
        support = self._physical_support_snapshot(observation)
        configuration_sha256 = self._recovery_configuration_identity(
            observation, self._feedback_active_leg
        )
        self._feedback_configuration_sha256 = configuration_sha256
        self._feedback_baseline = {
            "configuration_sha256": configuration_sha256,
            "wheel_z_m": feedback.wheel_center_w_m[self._feedback_active_leg][2],
            "measured_servo_positions_deg": dict(
                feedback.measured_servo_positions_deg
            ),
            "support_qualified_legs": tuple(support["support_qualified_legs"]),
            "support_stability_margin_m": support["support_stability_margin_m"],
            "wrench_force_residual_norm_n": support[
                "wrench_force_residual_norm_n"
            ],
            "wrench_moment_residual_norm_nm": support[
                "wrench_moment_residual_norm_nm"
            ],
            "wrench_maximum_friction_utilization": support[
                "wrench_maximum_friction_utilization"
            ],
            "abs_roll_rad": abs(feedback.base_roll_rad),
            "abs_pitch_rad": abs(feedback.base_pitch_rad),
            "joint_limit_margin_deg": dict(feedback.joint_limit_margin_deg),
        }

    def _clear_feedback_pending(self) -> None:
        self._feedback_pending_epoch = None
        self._feedback_pending_sim_step = None
        self._feedback_pending_action = ""
        self._feedback_pending_leg = ""
        self._feedback_pending_joint = ""
        self._feedback_pending_sign = 0

    def _advance_probe_cursor(self, sign: int) -> None:
        if sign > 0:
            self._feedback_probe_sign = -1
            self._feedback_stage = FeedbackRecoveryStage.SAFE_PROBE
        elif self._feedback_joint_index + 1 < len(
            LEG_SERVO_JOINTS[self._feedback_active_leg]
        ):
            self._feedback_joint_index += 1
            self._feedback_probe_sign = 1
            self._feedback_stage = FeedbackRecoveryStage.SAFE_PROBE
        else:
            self._feedback_stage = FeedbackRecoveryStage.SELECT_DESCENT

    def _begin_feedback_settle(
        self,
        *,
        sim_time_s: float,
        posture_incomplete: bool,
        reason: str = "",
    ) -> None:
        self._feedback_stage = FeedbackRecoveryStage.SETTLE
        self._feedback_settle_started_at_s = None
        self._feedback_settle_window_started_at_s = sim_time_s
        self._feedback_settle_posture_incomplete = posture_incomplete
        if posture_incomplete and reason:
            self._feedback_exhaustion_reason = reason

    @staticmethod
    def _top_contact_load_seen(observation: MacroObservation, leg: str) -> bool:
        row = observation.centroidal_support_evidence.wheel_contacts.by_leg()[leg]
        return bool(
            row.evidence_available
            and row.measurement.active
            and row.measurement.surface_kind == "OBSTACLE_TOP"
            and row.normal_load_n is not None
            and row.normal_load_n
            >= observation.centroidal_support_evidence.wheel_contacts.thresholds.minimum_normal_force_n
        )

    @staticmethod
    def _feedback_targets_within_limits(
        targets: Mapping[str, float],
    ) -> tuple[bool, str]:
        margin = float(FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"])
        for name, target in targets.items():
            lower, upper = command_limits_for_servo(name)
            if not lower + margin <= target <= upper - margin:
                return False, f"{name} target violates command-limit margin"
        return True, ""

    def _feedback_acknowledge_pending(
        self, observation: MacroObservation
    ) -> None:
        if self._feedback_pending_epoch is None:
            return
        feedback = observation.feedback_recovery_observation
        expected_step = (
            None
            if self._feedback_pending_sim_step is None
            else self._feedback_pending_sim_step + 1
        )
        if (
            not feedback.n_plus_one_verified
            or feedback.verified_command_epoch != self._feedback_pending_epoch
            or feedback.observed_command_epoch != self._feedback_pending_epoch
            or expected_step is None
            or feedback.sim_step != expected_step
            or dict(feedback.readback_servo_targets_deg)
            != self._current_servo_targets
            or dict(feedback.readback_wheel_targets_rad_s)
            != self._current_wheel_targets
        ):
            raise ValueError(
                "feedback recovery requires exact issued-step+1 epoch/full-map readback"
            )
        safety = self._recovery_safety_failure(observation)
        if safety:
            raise RuntimeError(safety)
        action = self._feedback_pending_action
        joint = self._feedback_pending_joint
        sign = self._feedback_pending_sign
        if action == FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE.value:
            baseline_q = self._feedback_baseline[
                "measured_servo_positions_deg"
            ][joint]
            current_q = feedback.measured_servo_positions_deg[joint]
            dz = (
                feedback.wheel_center_w_m[self._feedback_active_leg][2]
                - self._feedback_baseline["wheel_z_m"]
            )
            dq = current_q - baseline_q
            sign_response_valid = bool(
                abs(dq) >= 0.02 and (dq > 0.0) == (sign > 0)
            )
            baseline_preserved, unsafe_reasons = self._probe_preserves_baseline(
                observation, joint
            )
            key = (
                self._feedback_active_leg,
                joint,
                sign,
                self._feedback_configuration_sha256,
            )
            self._feedback_probe_results[key] = {
                "dz_m": dz,
                "dq_deg": dq,
                "sign_response_valid": sign_response_valid,
                "baseline_preserved": baseline_preserved,
                "unsafe_reasons": unsafe_reasons,
                "centroidal_evidence_sha256": (
                    observation.centroidal_support_evidence.payload_sha256
                ),
                "feedback_observation_sha256": feedback.payload_sha256,
            }
            self._feedback_stage = FeedbackRecoveryStage.RETURN_TO_REFERENCE
        elif action == FeedbackRecoveryAction.RETURN_TO_IMMUTABLE_REFERENCE.value:
            baseline_q = self._feedback_baseline[
                "measured_servo_positions_deg"
            ][joint]
            if abs(feedback.measured_servo_positions_deg[joint] - baseline_q) > 0.2:
                raise RuntimeError(
                    "feedback probe failed to return to immutable reference"
                )
            self._advance_probe_cursor(sign)
        elif action == FeedbackRecoveryAction.BOUNDED_DESCENT_INCREMENT.value:
            baseline_preserved, reasons = self._probe_preserves_baseline(
                observation, joint
            )
            if not baseline_preserved:
                raise RuntimeError(
                    "selected descent degraded its proven baseline: "
                    + "; ".join(reasons)
                )
            if self._top_contact_load_seen(observation, self._feedback_active_leg):
                self._feedback_contact_started_at_s = feedback.physics_time_s
                self._feedback_stage = FeedbackRecoveryStage.CONTACT_DWELL
            elif self._feedback_increment_count_by_leg[
                self._feedback_active_leg
            ] >= int(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_increments_per_leg"]):
                self._begin_feedback_settle(
                    sim_time_s=feedback.physics_time_s,
                    posture_incomplete=True,
                    reason=(
                        f"{self._feedback_active_leg} bounded descent increments exhausted"
                    ),
                )
            else:
                self._feedback_stage = FeedbackRecoveryStage.INCREMENT
        else:
            raise ValueError("pending feedback action is invalid")
        self._clear_feedback_pending()

    def _issue_feedback_target(
        self,
        observation: MacroObservation,
        *,
        targets: Mapping[str, float],
        stage: FeedbackRecoveryStage,
        action: FeedbackRecoveryAction,
        joint: str,
        sign: int,
    ) -> _CommandEvent:
        canonical = _target_map(
            targets,
            SERVO_JOINT_NAMES,
            label="feedback_recovery.targets",
            require_complete=True,
        )
        within_limits, limit_reason = self._feedback_targets_within_limits(canonical)
        if not within_limits:
            raise RuntimeError(limit_reason)
        wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        if canonical == self._current_servo_targets and wheels == self._current_wheel_targets:
            raise RuntimeError("feedback recovery target map did not change")
        maximum_actions = int(
            FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_feedback_actions"]
        )
        if self._feedback_action_count >= maximum_actions:
            raise RuntimeError("feedback action bound was not preflighted")
        target_map_sha256 = _stable_sha256(
            {
                "schema_version": FEEDBACK_RECOVERY_TARGET_MAP_SCHEMA,
                "servo_targets_deg": canonical,
                "wheel_targets_rad_s": wheels,
            }
        )
        self._current_servo_targets = canonical
        self._current_wheel_targets = wheels
        self._command_epoch += 1
        self._feedback_action_count += 1
        self._feedback_pending_epoch = self._command_epoch
        self._feedback_pending_sim_step = (
            observation.centroidal_support_evidence.sim_step
        )
        self._feedback_pending_action = action.value
        self._feedback_pending_leg = self._feedback_active_leg
        self._feedback_pending_joint = joint
        self._feedback_pending_sign = sign
        return _feedback_recovery_command_event(
            stage=stage,
            action=action,
            centroidal_evidence_sha256=(
                observation.centroidal_support_evidence.payload_sha256
            ),
            feedback_observation_sha256=(
                observation.feedback_recovery_observation.payload_sha256
            ),
            target_map_sha256=target_map_sha256,
            direction_sign=sign,
            attempt=self._feedback_action_count,
            leg=self._feedback_active_leg,
            joint=joint,
            configuration_sha256=self._feedback_configuration_sha256,
        )

    def _advance_feedback_recovery(
        self,
        observation: MacroObservation,
        *,
        sim_time_s: float,
        source_cursor_permit: bool,
    ) -> tuple[_CommandEvent, str]:
        if self._feedback_stage == FeedbackRecoveryStage.REFERENCE_PROFILE:
            self._feedback_stage = FeedbackRecoveryStage.OBSERVE_AIR
        if self._feedback_pending_epoch is not None:
            self._feedback_acknowledge_pending(observation)
        feedback = observation.feedback_recovery_observation
        if (
            feedback.observed_command_epoch != self._command_epoch
            or dict(feedback.readback_servo_targets_deg)
            != self._current_servo_targets
            or dict(feedback.readback_wheel_targets_rad_s)
            != self._current_wheel_targets
        ):
            raise ValueError(
                "feedback recovery observation is not bound to the current "
                "command epoch/full 8+4 readback"
            )
        safety = self._recovery_safety_failure(observation)
        if safety:
            raise RuntimeError(safety)
        # A newly measured load-bearing top contact is itself the feedback
        # event we were seeking.  Latch it before a permitted INCREMENT can
        # deepen the target, then hold the exact current 8+4 map while the
        # independent contact dwell/qualification evidence accrues.
        if (
            self._feedback_stage == FeedbackRecoveryStage.INCREMENT
            and self._feedback_active_leg
            and self._top_contact_load_seen(observation, self._feedback_active_leg)
        ):
            self._feedback_contact_started_at_s = (
                observation.feedback_recovery_observation.physics_time_s
            )
            self._feedback_stage = FeedbackRecoveryStage.CONTACT_DWELL
        if self._feedback_stage == FeedbackRecoveryStage.OBSERVE_AIR:
            candidates = self._recovery_candidate_legs(observation)
            if not candidates:
                if self._four_top_contacts_ready(observation):
                    self._begin_feedback_settle(
                        sim_time_s=sim_time_s,
                        posture_incomplete=False,
                    )
                else:
                    self._begin_feedback_settle(
                        sim_time_s=sim_time_s,
                        posture_incomplete=True,
                        reason=(
                            "no safe crossed/top-height AIR leg is eligible for feedback"
                        ),
                    )
            else:
                self._begin_recovery_leg(observation, candidates[0])
        if self._feedback_stage == FeedbackRecoveryStage.SELECT_DESCENT:
            choices: list[tuple[float, str, int]] = []
            for joint in LEG_SERVO_JOINTS[self._feedback_active_leg]:
                for sign in (1, -1):
                    key = (
                        self._feedback_active_leg,
                        joint,
                        sign,
                        self._feedback_configuration_sha256,
                    )
                    result = self._feedback_probe_results.get(key)
                    if (
                        result
                        and result["sign_response_valid"]
                        and result["baseline_preserved"]
                        and result["dz_m"]
                        <= -float(
                            FINAL_RECOVERY_FEEDBACK_LIMITS["minimum_descent_m"]
                        )
                    ):
                        choices.append((float(result["dz_m"]), joint, sign))
            if not choices:
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason=(
                        f"{self._feedback_active_leg} has no safe measured descent sign"
                    ),
                )
            else:
                _dz, joint, sign = min(choices)
                cache_key = (
                    self._feedback_active_leg,
                    joint,
                    self._feedback_configuration_sha256,
                )
                self._feedback_selected_signs[cache_key] = sign
                self._feedback_selected_joint = joint
                self._feedback_selected_sign = sign
                self._feedback_increment_count = 0
                self._feedback_stage = FeedbackRecoveryStage.INCREMENT
        if self._feedback_stage == FeedbackRecoveryStage.CONTACT_DWELL:
            row = observation.centroidal_support_evidence.wheel_contacts.by_leg()[
                self._feedback_active_leg
            ]
            dwell = row.measurement.dwell_s
            contact_present = self._top_contact_load_seen(
                observation, self._feedback_active_leg
            )
            contact_qualified = bool(
                contact_present
                and row.support_qualified
                and row.measurement.surface_kind == "OBSTACLE_TOP"
                and row.measurement.surface_dwell_verified
            )
            if contact_qualified and dwell is not None and dwell >= float(
                FINAL_RECOVERY_FEEDBACK_LIMITS["contact_dwell_s"]
            ):
                self._feedback_reference_targets = dict(self._current_servo_targets)
                self._feedback_baseline = {}
                self._feedback_active_leg = ""
                self._feedback_contact_started_at_s = None
                self._feedback_stage = FeedbackRecoveryStage.OBSERVE_AIR
            elif contact_present:
                if self._feedback_contact_started_at_s is None:
                    self._feedback_contact_started_at_s = sim_time_s
                if sim_time_s - self._feedback_contact_started_at_s > float(
                    FINAL_RECOVERY_FEEDBACK_LIMITS[
                        "maximum_contact_dwell_wait_s"
                    ]
                ):
                    self._begin_feedback_settle(
                        sim_time_s=sim_time_s,
                        posture_incomplete=True,
                        reason=(
                            f"{self._feedback_active_leg} contact dwell did not verify"
                        ),
                    )
            elif (
                row.evidence_available
                and not row.measurement.active
                and row.measurement.surface_kind == "AIR"
            ):
                self._feedback_contact_started_at_s = None
                if self._feedback_increment_count_by_leg[
                    self._feedback_active_leg
                ] >= int(
                    FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_increments_per_leg"]
                ):
                    self._begin_feedback_settle(
                        sim_time_s=sim_time_s,
                        posture_incomplete=True,
                        reason=(
                            f"{self._feedback_active_leg} bounded descent increments exhausted"
                        ),
                    )
                else:
                    self._feedback_stage = FeedbackRecoveryStage.INCREMENT
            else:
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason=(
                        f"{self._feedback_active_leg} contact evidence became unavailable"
                    ),
                )
        if self._feedback_stage == FeedbackRecoveryStage.SETTLE:
            feedback = observation.feedback_recovery_observation
            support = self._physical_support_snapshot(observation)
            base_settled = bool(
                feedback.final_recoverable
                and feedback.body_crossed_front_face
                and support["support_viable"]
                and support["support_wrench_proven"]
                and max(
                    abs(value)
                    for value in feedback.measured_servo_velocities_deg_s.values()
                )
                <= float(
                    FINAL_RECOVERY_FEEDBACK_LIMITS[
                        "maximum_abs_joint_velocity_deg_s"
                    ]
                )
                and max(abs(value) for value in feedback.base_angular_velocity_rad_s)
                <= float(
                    FINAL_RECOVERY_FEEDBACK_LIMITS[
                        "maximum_abs_body_angular_velocity_rad_s"
                    ]
                )
            )
            settled = bool(
                base_settled
                and (
                    self._feedback_settle_posture_incomplete
                    or self._four_top_contacts_ready(observation)
                )
            )
            if not settled:
                self._feedback_settle_started_at_s = None
            elif self._feedback_settle_started_at_s is None:
                self._feedback_settle_started_at_s = sim_time_s
            elif sim_time_s - self._feedback_settle_started_at_s >= float(
                FINAL_RECOVERY_FEEDBACK_LIMITS["settle_dwell_s"]
            ):
                self._feedback_stage = (
                    FeedbackRecoveryStage.POSTURE_INCOMPLETE
                    if self._feedback_settle_posture_incomplete
                    else FeedbackRecoveryStage.COMPLETE
                )
            if (
                self._feedback_stage == FeedbackRecoveryStage.SETTLE
                and self._feedback_settle_window_started_at_s is not None
                and sim_time_s - self._feedback_settle_window_started_at_s
                > float(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_settle_wait_s"])
            ):
                raise RuntimeError("feedback recovery settle bound exhausted")
        if self._feedback_stage in {
            FeedbackRecoveryStage.COMPLETE,
            FeedbackRecoveryStage.POSTURE_INCOMPLETE,
            FeedbackRecoveryStage.CONTACT_DWELL,
            FeedbackRecoveryStage.SETTLE,
        }:
            return _no_command_event(), f"feedback recovery {self._feedback_stage.value.lower()}"
        if not source_cursor_permit:
            return _no_command_event(), "feedback recovery holds unchanged off 15 Hz permit boundary"
        maximum_actions = int(
            FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_feedback_actions"]
        )
        delta = float(FINAL_RECOVERY_FEEDBACK_LIMITS["probe_delta_deg"])
        if self._feedback_stage == FeedbackRecoveryStage.SAFE_PROBE:
            joint = LEG_SERVO_JOINTS[self._feedback_active_leg][
                self._feedback_joint_index
            ]
            sign = self._feedback_probe_sign
            targets = dict(self._feedback_reference_targets)
            targets[joint] += sign * delta
            within_limits, limit_reason = self._feedback_targets_within_limits(
                targets
            )
            remaining = maximum_actions - self._feedback_action_count
            if not within_limits:
                # A pre-dispatch skip has no configuration SHA: there is no
                # physical probe baseline for the worker to reconstruct yet.
                # Once a prior probe has frozen the leg configuration, retain
                # later unsafe-sign diagnostics under that exact identity.
                if self._feedback_configuration_sha256:
                    key = (
                        self._feedback_active_leg,
                        joint,
                        sign,
                        self._feedback_configuration_sha256,
                    )
                    self._feedback_probe_results[key] = {
                        "dz_m": 0.0,
                        "dq_deg": 0.0,
                        "sign_response_valid": False,
                        "baseline_preserved": False,
                        "unsafe_reasons": (limit_reason,),
                        "centroidal_evidence_sha256": (
                            observation.centroidal_support_evidence.payload_sha256
                        ),
                        "feedback_observation_sha256": (
                            observation.feedback_recovery_observation.payload_sha256
                        ),
                    }
                self._advance_probe_cursor(sign)
                return _no_command_event(), "unsafe probe sign skipped without dispatch"
            if remaining < 2:
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason="feedback action bound cannot reserve probe return",
                )
                return _no_command_event(), "bounded feedback actions exhausted"
            if not self._feedback_configuration_sha256:
                self._freeze_recovery_configuration(observation)
            return self._issue_feedback_target(
                observation,
                targets=targets,
                stage=FeedbackRecoveryStage.SAFE_PROBE,
                action=FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE,
                joint=joint,
                sign=sign,
            ), "issued one conservative diagnostic probe map"
        if self._feedback_stage == FeedbackRecoveryStage.RETURN_TO_REFERENCE:
            joint = self._feedback_pending_joint or LEG_SERVO_JOINTS[
                self._feedback_active_leg
            ][self._feedback_joint_index]
            sign = self._feedback_probe_sign
            return self._issue_feedback_target(
                observation,
                targets=self._feedback_reference_targets,
                stage=FeedbackRecoveryStage.RETURN_TO_REFERENCE,
                action=FeedbackRecoveryAction.RETURN_TO_IMMUTABLE_REFERENCE,
                joint=joint,
                sign=sign,
            ), "returning to immutable reference before next probe"
        if self._feedback_stage == FeedbackRecoveryStage.INCREMENT:
            joint = self._feedback_selected_joint
            sign = self._feedback_selected_sign
            cache_key = (
                self._feedback_active_leg,
                joint,
                self._feedback_configuration_sha256,
            )
            if self._feedback_selected_signs.get(cache_key) != sign:
                raise RuntimeError("selected feedback sign cache identity mismatch")
            if self._feedback_action_count >= maximum_actions:
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason="feedback action bound exhausted before descent increment",
                )
                return _no_command_event(), "bounded feedback actions exhausted"
            if self._feedback_increment_count_by_leg[
                self._feedback_active_leg
            ] >= int(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_increments_per_leg"]):
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason=(
                        f"{self._feedback_active_leg} cumulative increment bound exhausted"
                    ),
                )
                return _no_command_event(), "bounded leg increments exhausted"
            self._feedback_increment_count += 1
            targets = dict(self._feedback_reference_targets)
            targets[joint] += sign * float(
                FINAL_RECOVERY_FEEDBACK_LIMITS["increment_delta_deg"]
            ) * self._feedback_increment_count
            within_limits, limit_reason = self._feedback_targets_within_limits(
                targets
            )
            if not within_limits:
                self._feedback_increment_count -= 1
                self._begin_feedback_settle(
                    sim_time_s=sim_time_s,
                    posture_incomplete=True,
                    reason=limit_reason,
                )
                return _no_command_event(), "descent target limit bound reached"
            self._feedback_increment_count_by_leg[self._feedback_active_leg] += 1
            return self._issue_feedback_target(
                observation,
                targets=targets,
                stage=FeedbackRecoveryStage.INCREMENT,
                action=FeedbackRecoveryAction.BOUNDED_DESCENT_INCREMENT,
                joint=joint,
                sign=sign,
            ), "issued one bounded selected-sign descent increment"
        return _no_command_event(), "feedback recovery observation-only stage"

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
        *,
        force_posture_incomplete: bool = False,
    ) -> MacroDecision:
        changed = any(abs(value) > 1.0e-12 for value in self._current_wheel_targets.values())
        self._current_wheel_targets = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        if changed:
            self._command_epoch += 1
        self._state_id = self.graph.success_state
        self._active_profile = None
        self._terminal_outcome = (
            MacroTerminalOutcome.TASK_SUCCESS
            if (
                observation.feedback_recovery_observation.posture_complete
                and not force_posture_incomplete
            )
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
        elif (
            self._state_id == MacroStateId.S10_POSTURE_RECOVERY
            and self._feedback_stage != FeedbackRecoveryStage.REFERENCE_PROFILE
        ):
            subphase = MacroSubphase.FEEDBACK_RECOVERY
            fraction = 1.0
            profile_id = (
                "" if self._active_profile is None else self._active_profile.profile_id
            )
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
