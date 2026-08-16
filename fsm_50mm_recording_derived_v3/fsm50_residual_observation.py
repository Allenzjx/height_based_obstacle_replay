"""Deployable actor observation for the 50 mm residual controller.

The schema is deliberately Isaac-free.  It consumes the same finite telemetry
fields that the production macro worker already exposes, so a policy trained
with this vector does not depend on simulator-only force, load, or exact-COM
signals.  ``com_proxy_relative_support_xy`` is a geometric body/support proxy;
it is not presented as a force-derived stability margin.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES


ACTOR_OBSERVATION_SCHEMA_VERSION = "fsm50.residual_actor_observation.v1"

RUNNING_MACRO_STATES = (
    "S0_INITIALIZE",
    "S1_APPROACH_AND_PRE_FR_SHIFT",
    "S2_FR_TRAVERSE",
    "S3_FL_TRAVERSE",
    "S4_FRONT_PAIR_ADVANCE",
    "S5_PRE_RR_COM_SHIFT",
    "S6_RR_TRAVERSE",
    "S7_PRE_RL_SUPPORT_SETUP",
    "S8_RL_COM_SHIFT_AND_TRAVERSE",
    "S9_FINAL_ADVANCE",
    "S10_POSTURE_RECOVERY",
)

AUTHORIZED_SOURCE_VERSIONS = (
    "v003_20260805_224517_157723_manual",
    "v008_20260806_211408_578700_manual",
    "v009_20260806_215232_433234_manual",
)

LEG_NAMES = ("FL", "FR", "RL", "RR")
CONTACT_CLASS_NAMES = ("AIR", "GROUND", "FRONT_FACE", "TOP")
RESIDUAL_ACTION_DIM = len(SERVO_JOINT_NAMES) + len(WHEEL_JOINT_NAMES)


_FIELD_WIDTHS = (
    ("macro_state_one_hot", len(RUNNING_MACRO_STATES)),
    ("source_version_one_hot", len(AUTHORIZED_SOURCE_VERSIONS)),
    ("profile_fraction", 1),
    ("active_leg_one_hot", len(LEG_NAMES)),
    ("support_leg_mask", len(LEG_NAMES)),
    ("base_roll_pitch_yaw_rad", 3),
    ("root_linear_velocity_w_m_s", 3),
    ("root_angular_velocity_w_rad_s", 3),
    ("com_proxy_relative_support_xy_m", 2),
    ("servo_joint_position_rad", len(SERVO_JOINT_NAMES)),
    ("servo_joint_velocity_rad_s", len(SERVO_JOINT_NAMES)),
    ("nominal_servo_target_deg", len(SERVO_JOINT_NAMES)),
    ("servo_actual_error_deg", len(SERVO_JOINT_NAMES)),
    ("wheel_velocity_rad_s", len(WHEEL_JOINT_NAMES)),
    ("nominal_wheel_target_rad_s", len(WHEEL_JOINT_NAMES)),
    ("wheel_contact_class_one_hot", len(LEG_NAMES) * len(CONTACT_CLASS_NAMES)),
    ("wheel_front_face_clearance_m", len(LEG_NAMES)),
    ("wheel_top_clearance_m", len(LEG_NAMES)),
    ("obstacle_front_top_relative_to_base_m", 2),
    ("previous_residual", RESIDUAL_ACTION_DIM),
    ("final_recovery_max_servo_error_deg", 1),
    ("geometry_support_candidate_count", 1),
    ("body_crossed_front_face", 1),
)


def _build_field_slices() -> dict[str, tuple[int, int]]:
    cursor = 0
    result: dict[str, tuple[int, int]] = {}
    for name, width in _FIELD_WIDTHS:
        result[name] = (cursor, cursor + width)
        cursor += width
    return result


ACTOR_OBSERVATION_FIELD_SLICES = _build_field_slices()
ACTOR_OBSERVATION_DIM = sum(width for _, width in _FIELD_WIDTHS)
ACTOR_OBSERVATION_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "schema_version": ACTOR_OBSERVATION_SCHEMA_VERSION,
            "field_widths": list(_FIELD_WIDTHS),
            "macro_states": list(RUNNING_MACRO_STATES),
            "source_versions": list(AUTHORIZED_SOURCE_VERSIONS),
            "legs": list(LEG_NAMES),
            "contact_classes": list(CONTACT_CLASS_NAMES),
            "servo_order": list(SERVO_JOINT_NAMES),
            "wheel_order": list(WHEEL_JOINT_NAMES),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class ResidualObservationError(ValueError):
    """Raised when deployable actor evidence is missing or malformed."""


@dataclass(frozen=True)
class ResidualActorObservation:
    values: tuple[float, ...]
    schema_version: str = ACTOR_OBSERVATION_SCHEMA_VERSION
    schema_sha256: str = ACTOR_OBSERVATION_SCHEMA_SHA256

    def __post_init__(self) -> None:
        if len(self.values) != ACTOR_OBSERVATION_DIM:
            raise ResidualObservationError(
                f"actor observation length {len(self.values)} != {ACTOR_OBSERVATION_DIM}"
            )
        if not all(math.isfinite(value) for value in self.values):
            raise ResidualObservationError("actor observation contains a non-finite value")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema_sha256": self.schema_sha256,
            "values": list(self.values),
        }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ResidualObservationError(f"{label} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResidualObservationError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ResidualObservationError(f"{label} must be finite")
    return result


def _finite_vector(value: Any, width: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ResidualObservationError(f"{label} must be a sequence of length {width}")
    if len(value) != width:
        raise ResidualObservationError(f"{label} length {len(value)} != {width}")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _exact_finite_map(value: Any, names: Sequence[str], label: str) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise ResidualObservationError(f"{label} must be a mapping")
    expected = set(names)
    actual = set(value)
    if actual != expected:
        raise ResidualObservationError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return tuple(_finite(value[name], f"{label}.{name}") for name in names)


def _one_hot(value: str, names: Sequence[str], label: str, *, allow_empty: bool = False) -> tuple[float, ...]:
    if value == "" and allow_empty:
        return tuple(0.0 for _ in names)
    if value not in names:
        raise ResidualObservationError(f"unsupported {label}: {value!r}")
    return tuple(1.0 if item == value else 0.0 for item in names)


def _support_proxy_xy(telemetry: Mapping[str, Any], support_legs: tuple[str, ...]) -> tuple[float, float]:
    base_position = telemetry.get("base_position_m")
    if not isinstance(base_position, Mapping) or set(base_position) != {"x", "y", "z"}:
        raise ResidualObservationError("base_position_m must contain exactly x/y/z")
    base_x = _finite(base_position["x"], "base_position_m.x")
    base_y = _finite(base_position["y"], "base_position_m.y")
    wheel_centers = telemetry.get("wheel_center_w_m")
    if not isinstance(wheel_centers, Mapping) or set(wheel_centers) != set(LEG_NAMES):
        raise ResidualObservationError("wheel_center_w_m must contain exactly FL/FR/RL/RR")
    if not support_legs:
        raise ResidualObservationError("support_legs must not be empty")
    support_xy = {
        leg: _finite_vector(wheel_centers[leg], 3, f"wheel_center_w_m.{leg}")[:2]
        for leg in support_legs
    }
    centroid_x = sum(value[0] for value in support_xy.values()) / len(support_xy)
    centroid_y = sum(value[1] for value in support_xy.values()) / len(support_xy)
    return (base_x - centroid_x, base_y - centroid_y)


def build_residual_actor_observation(
    telemetry: Mapping[str, Any],
    previous_residual: Sequence[float],
) -> ResidualActorObservation:
    """Build the fixed deployable actor vector from one worker telemetry row."""

    if not isinstance(telemetry, Mapping):
        raise ResidualObservationError("telemetry must be a mapping")

    macro_state = str(telemetry.get("macro_state", ""))
    source_version = str(telemetry.get("source_version", ""))
    active_leg = str(telemetry.get("active_leg", ""))

    support_raw = telemetry.get("support_legs")
    if not isinstance(support_raw, Sequence) or isinstance(support_raw, (str, bytes, bytearray)):
        raise ResidualObservationError("support_legs must be a sequence")
    support_legs = tuple(str(item) for item in support_raw)
    if len(set(support_legs)) != len(support_legs) or any(leg not in LEG_NAMES for leg in support_legs):
        raise ResidualObservationError("support_legs contains duplicates or unsupported legs")

    profile_fraction_raw = telemetry.get("profile_fraction")
    if profile_fraction_raw is None and macro_state == "S0_INITIALIZE":
        profile_fraction = 0.0
    else:
        profile_fraction = _finite(profile_fraction_raw, "profile_fraction")
        if not 0.0 <= profile_fraction <= 1.0:
            raise ResidualObservationError("profile_fraction must be within [0, 1]")

    joint_q = telemetry.get("joint_q_rad")
    joint_qd = telemetry.get("joint_qd_rad_s")
    all_joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    all_q = _exact_finite_map(joint_q, all_joint_names, "joint_q_rad")
    all_qd = _exact_finite_map(joint_qd, all_joint_names, "joint_qd_rad_s")
    servo_q = all_q[: len(SERVO_JOINT_NAMES)]
    servo_qd = all_qd[: len(SERVO_JOINT_NAMES)]
    wheel_qd = all_qd[len(SERVO_JOINT_NAMES) :]

    nominal_servo = _exact_finite_map(
        telemetry.get("servo_targets_deg"), SERVO_JOINT_NAMES, "servo_targets_deg"
    )
    servo_error = _exact_finite_map(
        telemetry.get("canonical_servo_actual_error_deg"),
        SERVO_JOINT_NAMES,
        "canonical_servo_actual_error_deg",
    )
    nominal_wheel = _exact_finite_map(
        telemetry.get("wheel_targets_rad_s"), WHEEL_JOINT_NAMES, "wheel_targets_rad_s"
    )

    contact_classes = telemetry.get("wheel_contact_classes")
    if not isinstance(contact_classes, Mapping) or set(contact_classes) != set(LEG_NAMES):
        raise ResidualObservationError("wheel_contact_classes must contain exactly FL/FR/RL/RR")
    contact_one_hot: list[float] = []
    for leg in LEG_NAMES:
        contact_one_hot.extend(_one_hot(str(contact_classes[leg]), CONTACT_CLASS_NAMES, f"{leg} contact"))

    front_clearance = _exact_finite_map(
        telemetry.get("wheel_front_face_clearance_m"), LEG_NAMES, "wheel_front_face_clearance_m"
    )
    top_clearance = _exact_finite_map(
        telemetry.get("wheel_top_clearance_m"), LEG_NAMES, "wheel_top_clearance_m"
    )
    previous = _finite_vector(previous_residual, RESIDUAL_ACTION_DIM, "previous_residual")

    base_position = telemetry["base_position_m"]
    obstacle_relative = (
        _finite(telemetry.get("obstacle_front_face_x_m"), "obstacle_front_face_x_m")
        - _finite(base_position["x"], "base_position_m.x"),
        _finite(telemetry.get("obstacle_top_z_m"), "obstacle_top_z_m")
        - _finite(base_position["z"], "base_position_m.z"),
    )

    support_count = telemetry.get("geometry_support_candidate_count")
    if isinstance(support_count, bool) or not isinstance(support_count, int) or not 0 <= support_count <= 4:
        raise ResidualObservationError("geometry_support_candidate_count must be an integer within [0, 4]")
    body_crossed = telemetry.get("body_crossed_front_face")
    if not isinstance(body_crossed, bool):
        raise ResidualObservationError("body_crossed_front_face must be bool")

    values = (
        *_one_hot(macro_state, RUNNING_MACRO_STATES, "macro_state"),
        *_one_hot(source_version, AUTHORIZED_SOURCE_VERSIONS, "source_version"),
        profile_fraction,
        *_one_hot(active_leg, LEG_NAMES, "active_leg", allow_empty=True),
        *(1.0 if leg in support_legs else 0.0 for leg in LEG_NAMES),
        _finite(telemetry.get("base_roll_rad"), "base_roll_rad"),
        _finite(telemetry.get("base_pitch_rad"), "base_pitch_rad"),
        _finite(telemetry.get("base_yaw_rad"), "base_yaw_rad"),
        *_finite_vector(telemetry.get("root_linear_velocity_w"), 3, "root_linear_velocity_w"),
        *_finite_vector(telemetry.get("root_angular_velocity_w"), 3, "root_angular_velocity_w"),
        *_support_proxy_xy(telemetry, support_legs),
        *servo_q,
        *servo_qd,
        *nominal_servo,
        *servo_error,
        *wheel_qd,
        *nominal_wheel,
        *contact_one_hot,
        *front_clearance,
        *top_clearance,
        *obstacle_relative,
        *previous,
        max(abs(value) for value in servo_error),
        float(support_count),
        1.0 if body_crossed else 0.0,
    )
    return ResidualActorObservation(values=tuple(values))


__all__ = [
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_FIELD_SLICES",
    "ACTOR_OBSERVATION_SCHEMA_SHA256",
    "ACTOR_OBSERVATION_SCHEMA_VERSION",
    "AUTHORIZED_SOURCE_VERSIONS",
    "CONTACT_CLASS_NAMES",
    "LEG_NAMES",
    "RESIDUAL_ACTION_DIM",
    "RUNNING_MACRO_STATES",
    "ResidualActorObservation",
    "ResidualObservationError",
    "build_residual_actor_observation",
]
