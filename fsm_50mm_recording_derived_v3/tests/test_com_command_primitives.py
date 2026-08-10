from __future__ import annotations

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import (
    AnchoredCommandConfig,
    AnchoredStage,
    AnchoredSupportAngleCommandGenerator,
    GuardDecision,
    ImpulseCommandConfig,
    ImpulseReactionCommandGenerator,
    ImpulseStage,
    Leg,
)


PROVENANCE = {
    "source_recording_version": "v012",
    "source_step_indices": [14, 15],
    "status": "PENDING_REPLAY",
}


def _impulse(maximum_retries: int = 1) -> ImpulseReactionCommandGenerator:
    config = ImpulseCommandConfig(
        impulse_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        preload_servo_targets_deg={"rear_left_knee": 20.0},
        push_servo_targets_deg={"rear_left_knee": 35.0},
        release_servo_targets_deg={"rear_left_knee": 21.5},
        push_wheel_targets_rad_s={"front_left_ankle": -0.3},
        release_wheel_targets_rad_s={"front_left_ankle": 0.0},
        optional_wheel_assist=True,
        retry_pulse_scale=1.1,
        maximum_retries=maximum_retries,
        source_provenance=PROVENANCE,
    )
    generator = ImpulseReactionCommandGenerator(config)
    generator.reset(
        servo_targets_deg={"rear_left_knee": 0.0},
        wheel_targets_rad_s={"front_left_ankle": 0.0},
    )
    return generator


def test_impulse_command_sequence_is_guard_gated_and_has_release_coast() -> None:
    generator = _impulse()
    assert generator.stage == ImpulseStage.PRELOAD
    assert not generator.advance(GuardDecision(False, reason="load low"))
    assert generator.stage == ImpulseStage.PRELOAD

    stages = [generator.stage]
    while generator.advance(GuardDecision(True)):
        stages.append(generator.stage)
    assert stages == list(ImpulseStage)

    generator = _impulse()
    generator.advance(GuardDecision(True))  # PUSH
    push = generator.command(progress=1.0)
    assert push.atomic_concurrent
    assert push.servo_targets_deg["rear_left_knee"] == 35.0
    generator.advance(GuardDecision(True))  # RELEASE
    release = generator.command(progress=1.0)
    generator.advance(GuardDecision(True))  # COAST
    coast = generator.command(progress=0.0)
    assert coast.hold
    assert coast.servo_targets_deg == release.servo_targets_deg
    assert coast.wheel_targets_rad_s == release.wheel_targets_rad_s


def test_impulse_retry_only_restarts_from_failed_verify() -> None:
    generator = _impulse(maximum_retries=1)
    for _ in range(5):
        assert generator.advance(GuardDecision(True))
    assert generator.stage == ImpulseStage.VERIFY
    assert generator.request_retry(GuardDecision(False, reason="insufficient transfer"))
    assert generator.stage == ImpulseStage.PRELOAD
    assert generator.retry_index == 1
    for _ in range(5):
        generator.advance(GuardDecision(True))
    assert not generator.request_retry(GuardDecision(False, reason="still insufficient"))


def test_nonzero_wheel_assist_must_be_explicit_and_provenanced() -> None:
    with pytest.raises(ValueError, match="provenance"):
        ImpulseCommandConfig(
            impulse_leg=Leg.FR,
            support_legs=(Leg.FL, Leg.RL, Leg.RR),
            preload_servo_targets_deg={},
            push_servo_targets_deg={},
            release_servo_targets_deg={},
        )
    with pytest.raises(ValueError, match="wheel assist"):
        ImpulseCommandConfig(
            impulse_leg=Leg.FR,
            support_legs=(Leg.FL, Leg.RL, Leg.RR),
            preload_servo_targets_deg={},
            push_servo_targets_deg={},
            release_servo_targets_deg={},
            push_wheel_targets_rad_s={"front_left_ankle": 0.2},
            source_provenance=PROVENANCE,
        )


def _anchored() -> AnchoredSupportAngleCommandGenerator:
    return AnchoredSupportAngleCommandGenerator(
        AnchoredCommandConfig(
            support_legs=(Leg.FL, Leg.RR),
            active_leg=Leg.RL,
            support_servo_start_deg={"front_left_hip": 0.0, "rear_right_hip": 0.0},
            geometry_preload_servo_deg={"front_left_hip": 5.0, "rear_right_hip": -5.0},
            support_angle_end_deg={"front_left_hip": 10.0, "rear_right_hip": -10.0},
            wheel_targets_rad_s={
                "front_left_ankle": 0.0,
                "rear_right_ankle": 0.0,
            },
            anti_drift_gain_rad_s_per_m=10.0,
            maximum_anti_drift_wheel_rad_s=0.1,
            source_provenance=PROVENANCE,
        )
    )


def test_anchored_sequence_requires_real_contact_anchor_and_bounds_feedback() -> None:
    generator = _anchored()
    assert not generator.advance(GuardDecision(True))
    generator.capture_anchors({Leg.FL: (0.0, 0.2), Leg.RR: (-0.3, -0.2)})
    assert generator.advance(GuardDecision(True))
    assert generator.stage == AnchoredStage.GEOMETRY_PRELOAD
    midpoint = generator.command(
        progress=0.5,
        signed_contact_drift_m={Leg.FL: 0.5, Leg.RR: -0.5},
    )
    assert midpoint.servo_targets_deg["front_left_hip"] == 2.5
    assert midpoint.wheel_targets_rad_s["front_left_ankle"] == -0.1
    assert midpoint.wheel_targets_rad_s["rear_right_ankle"] == 0.1
    assert midpoint.active_leg == Leg.RL
    assert midpoint.correction_excluded_legs == (Leg.RL,)

    seen = [AnchoredStage.SUPPORT_ANCHOR, generator.stage]
    while generator.advance(GuardDecision(True)):
        seen.append(generator.stage)
    assert seen == list(AnchoredStage)
