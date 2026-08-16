from __future__ import annotations

import math

import pytest

from fsm_50mm_recording_derived_v3.fsm50_residual_reward import (
    ResidualRewardError,
    ResidualRewardInput,
    compute_residual_reward,
)


def _input(**overrides) -> ResidualRewardInput:
    values = {
        "macro_state": "S8_RL_COM_SHIFT_AND_TRAVERSE",
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "root_linear_velocity_w": (0.2, 0.0, 0.0),
        "root_angular_velocity_w": (0.0, 0.0, 0.0),
        "normalized_action": (0.0,) * 12,
        "previous_normalized_action": (0.0,) * 12,
        "geometry_support_candidate_count": 4,
        "max_servo_endpoint_error_deg": 0.0,
    }
    values.update(overrides)
    return ResidualRewardInput(**values)


def test_forward_velocity_is_not_penalized_by_itself() -> None:
    slow = compute_residual_reward(_input(root_linear_velocity_w=(0.0, 0.0, 0.0)))
    forward = compute_residual_reward(_input(root_linear_velocity_w=(1.0, 0.0, 0.0)))
    assert forward.total == slow.total


def test_s8_ordered_events_receive_explicit_bonuses() -> None:
    base = compute_residual_reward(_input())
    events = compute_residual_reward(
        _input(
            active_leg_airborne_event=True,
            active_leg_front_face_cross_event=True,
            active_leg_top_event=True,
            body_cross_event=True,
        )
    )
    assert events.total - base.total == pytest.approx(12.0)


def test_hard_failure_dominates_and_success_is_distinct() -> None:
    failure = compute_residual_reward(_input(hard_failure=True))
    success = compute_residual_reward(_input(fsm_success=True))
    assert failure.components["hard_failure"] == -100.0
    assert success.components["fsm_success"] == 10.0
    assert failure.total < success.total


def test_s10_scores_recovery_without_wheel_drive_reward() -> None:
    result = compute_residual_reward(
        _input(
            macro_state="S10_POSTURE_RECOVERY",
            max_servo_endpoint_error_deg=1.0,
            all_wheels_top=True,
            posture_stable=True,
        )
    )
    assert result.components["all_wheels_top"] == 0.5
    assert result.components["posture_stable"] == 5.0
    assert "active_leg_top" not in result.components


@pytest.mark.parametrize(
    "kwargs",
    [
        {"macro_state": "S2_FR_TRAVERSE"},
        {"base_roll_rad": float("nan")},
        {"geometry_support_candidate_count": 5},
        {"normalized_action": (0.0,) * 11},
        {"hard_failure": 1},
    ],
)
def test_reward_inputs_fail_closed(kwargs) -> None:
    with pytest.raises(ResidualRewardError):
        _input(**kwargs)


def test_stability_penalties_match_documented_scales() -> None:
    result = compute_residual_reward(
        _input(
            base_roll_rad=math.radians(20.0),
            base_pitch_rad=math.radians(15.0),
            root_angular_velocity_w=(math.radians(25.0), 0.0, 0.0),
        )
    )
    assert result.components["tilt_stability"] == pytest.approx(-0.2)
    assert result.components["angular_stability"] == pytest.approx(-0.05)
