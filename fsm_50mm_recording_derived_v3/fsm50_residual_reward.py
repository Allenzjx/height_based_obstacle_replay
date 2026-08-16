"""Auditable phase-local reward for bounded residual PPO.

The reward scores stability and completion of the phase already selected by
the Macro FSM.  It deliberately contains no leg-order selector and no penalty
for forward/passive motion by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


REWARD_SCHEMA_VERSION = "fsm50.residual_reward.v1"
ALLOWED_TRAINING_STATES = {
    "S5_PRE_RR_COM_SHIFT",
    "S7_PRE_RL_SUPPORT_SETUP",
    "S8_RL_COM_SHIFT_AND_TRAVERSE",
    "S10_POSTURE_RECOVERY",
}


class ResidualRewardError(ValueError):
    pass


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ResidualRewardError(f"{label} must not be bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResidualRewardError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ResidualRewardError(f"{label} must be finite")
    return result


def _vector(value: Sequence[float], width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != width:
        raise ResidualRewardError(f"{label} must have length {width}")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class ResidualRewardInput:
    macro_state: str
    base_roll_rad: float
    base_pitch_rad: float
    root_linear_velocity_w: tuple[float, float, float]
    root_angular_velocity_w: tuple[float, float, float]
    normalized_action: tuple[float, ...]
    previous_normalized_action: tuple[float, ...]
    geometry_support_candidate_count: int
    max_servo_endpoint_error_deg: float
    active_leg_airborne_event: bool = False
    active_leg_front_face_cross_event: bool = False
    active_leg_top_event: bool = False
    body_cross_event: bool = False
    next_phase_completed: bool = False
    all_wheels_top: bool = False
    posture_stable: bool = False
    fsm_success: bool = False
    hard_failure: bool = False

    def __post_init__(self) -> None:
        if self.macro_state not in ALLOWED_TRAINING_STATES:
            raise ResidualRewardError(f"state is not enabled for R1 reward: {self.macro_state!r}")
        for name in (
            "active_leg_airborne_event",
            "active_leg_front_face_cross_event",
            "active_leg_top_event",
            "body_cross_event",
            "next_phase_completed",
            "all_wheels_top",
            "posture_stable",
            "fsm_success",
            "hard_failure",
        ):
            if type(getattr(self, name)) is not bool:
                raise ResidualRewardError(f"{name} must be bool")
        if (
            type(self.geometry_support_candidate_count) is not int
            or not 0 <= self.geometry_support_candidate_count <= 4
        ):
            raise ResidualRewardError("geometry_support_candidate_count must be an integer in [0, 4]")
        object.__setattr__(self, "base_roll_rad", _finite(self.base_roll_rad, "base_roll_rad"))
        object.__setattr__(self, "base_pitch_rad", _finite(self.base_pitch_rad, "base_pitch_rad"))
        object.__setattr__(
            self, "root_linear_velocity_w", _vector(self.root_linear_velocity_w, 3, "root_linear_velocity_w")
        )
        object.__setattr__(
            self, "root_angular_velocity_w", _vector(self.root_angular_velocity_w, 3, "root_angular_velocity_w")
        )
        object.__setattr__(self, "normalized_action", _vector(self.normalized_action, 12, "normalized_action"))
        object.__setattr__(
            self,
            "previous_normalized_action",
            _vector(self.previous_normalized_action, 12, "previous_normalized_action"),
        )
        error = _finite(self.max_servo_endpoint_error_deg, "max_servo_endpoint_error_deg")
        if error < 0.0:
            raise ResidualRewardError("max_servo_endpoint_error_deg must be non-negative")
        object.__setattr__(self, "max_servo_endpoint_error_deg", error)


@dataclass(frozen=True)
class ResidualRewardResult:
    total: float
    components: dict[str, float]
    schema_version: str = REWARD_SCHEMA_VERSION


def compute_residual_reward(value: ResidualRewardInput) -> ResidualRewardResult:
    if not isinstance(value, ResidualRewardInput):
        raise ResidualRewardError("reward input must be ResidualRewardInput")

    roll_scale = math.radians(20.0)
    pitch_scale = math.radians(15.0)
    angular_scale = math.radians(25.0)
    angular_norm = math.sqrt(sum(component * component for component in value.root_angular_velocity_w))
    action_norm_sq = sum(component * component for component in value.normalized_action)
    delta_norm_sq = sum(
        (current - previous) ** 2
        for current, previous in zip(value.normalized_action, value.previous_normalized_action)
    )

    components = {
        "tilt_stability": -0.10
        * ((value.base_roll_rad / roll_scale) ** 2 + (value.base_pitch_rad / pitch_scale) ** 2),
        "angular_stability": -0.05 * min((angular_norm / angular_scale) ** 2, 4.0),
        # Forward x is intentionally absent: natural forward/passive motion is
        # not itself a failure under the task definition.
        "lateral_vertical_velocity": -0.02
        * (
            (value.root_linear_velocity_w[1] / 0.10) ** 2
            + (value.root_linear_velocity_w[2] / 0.10) ** 2
        ),
        "residual_magnitude": -0.01 * action_norm_sq,
        "residual_rate": -0.02 * delta_norm_sq,
        "support_geometry": 0.05 * (value.geometry_support_candidate_count / 4.0),
        "phase_completion": 1.0 if value.next_phase_completed else 0.0,
        "fsm_success": 10.0 if value.fsm_success else 0.0,
        "hard_failure": -100.0 if value.hard_failure else 0.0,
    }

    if value.macro_state == "S8_RL_COM_SHIFT_AND_TRAVERSE":
        components.update(
            {
                "active_leg_airborne": 1.0 if value.active_leg_airborne_event else 0.0,
                "active_leg_front_face_cross": 2.0 if value.active_leg_front_face_cross_event else 0.0,
                "active_leg_top": 4.0 if value.active_leg_top_event else 0.0,
                "body_cross": 5.0 if value.body_cross_event else 0.0,
            }
        )
    if value.macro_state == "S10_POSTURE_RECOVERY":
        components.update(
            {
                "final_servo_error": -0.05 * (value.max_servo_endpoint_error_deg / 3.0) ** 2,
                "all_wheels_top": 0.5 if value.all_wheels_top else 0.0,
                "posture_stable": 5.0 if value.posture_stable else 0.0,
            }
        )
    total = sum(components.values())
    if not math.isfinite(total):  # defensive; all inputs/components are finite above
        raise ResidualRewardError("computed reward is non-finite")
    return ResidualRewardResult(total=total, components=components)


__all__ = [
    "ALLOWED_TRAINING_STATES",
    "REWARD_SCHEMA_VERSION",
    "ResidualRewardError",
    "ResidualRewardInput",
    "ResidualRewardResult",
    "compute_residual_reward",
]
