from __future__ import annotations

import math

import pytest

from fsm_50mm_recording_derived_v3.com_transfer_primitives import (
    ActiveLegMode,
    AnchoredSupportAngleGuard,
    AnchoredTransferConfig,
    COMDirectionalConfig,
    COMDirectionalDetector,
    ContinuousOffsetManager,
    FinalSuccessCriteria,
    FinalSuccessObservation,
    ImpulseReactionGuard,
    ImpulseStage,
    ImpulseTransferConfig,
    Leg,
    LegIKCandidate,
    PlaceLoadDwellGuard,
    SingleRelatchTarget,
    StateScopedWheelRamp,
    TransferObservation,
    accept_per_leg_ik,
    evaluate_final_success,
    evaluate_lift_to_swing_clearance,
    isolate_active_leg_corrections,
)


def test_per_leg_ik_preserves_exact_boundary_reference_and_is_not_all_or_nothing() -> None:
    result = accept_per_leg_ik(
        {
            Leg.FL: LegIKCandidate(
                leg=Leg.FL,
                requested_targets_deg={"front_left_hip": 135.0000005},
                reference_targets_deg={"front_left_hip": 135.0},
                joint_limits_deg={"front_left_hip": (-135.0, 135.0)},
            ),
            Leg.RR: LegIKCandidate(
                leg=Leg.RR,
                requested_targets_deg={"rear_right_knee": 215.0},
                reference_targets_deg={"rear_right_knee": 0.0},
                joint_limits_deg={"rear_right_knee": (-60.0, 210.0)},
            ),
        }
    )

    assert result.decisions[Leg.FL].accepted
    assert result.decisions[Leg.FL].targets_deg["front_left_hip"] == 135.0
    assert result.decisions[Leg.FL].preserved_reference_joints == ("front_left_hip",)
    assert not result.decisions[Leg.RR].accepted
    assert result.rejected_legs == (Leg.RR,)
    assert result.accepted_targets_deg[Leg.FL]["front_left_hip"] == 135.0


def test_active_leg_correction_isolation_and_place_blend() -> None:
    base = {
        leg: {f"{leg.value}_hip": 10.0, f"{leg.value}_knee": 20.0}
        for leg in Leg
    }
    correction = {
        leg: {f"{leg.value}_hip": 4.0, f"{leg.value}_knee": -2.0}
        for leg in Leg
    }

    protected = isolate_active_leg_corrections(
        base,
        correction,
        active_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        mode=ActiveLegMode.SWING_CLEAR,
    )
    assert protected.scales[Leg.RL] == 0.0
    assert protected.targets_deg[Leg.RL]["RL_hip"] == 10.0
    assert protected.scales[Leg.FR] == 0.0
    assert protected.targets_deg[Leg.FL]["FL_hip"] == 14.0
    assert protected.targets_deg[Leg.RR]["RR_knee"] == 18.0

    placement = isolate_active_leg_corrections(
        base,
        correction,
        active_leg=Leg.RL,
        support_legs=(Leg.FL, Leg.RR),
        mode=ActiveLegMode.PLACE,
        place_confirm_blend=0.25,
    )
    assert placement.targets_deg[Leg.RL]["RL_hip"] == pytest.approx(11.0)
    assert placement.targets_deg[Leg.RL]["RL_knee"] == pytest.approx(19.5)


def test_com_directional_detector_uses_direction_and_consecutive_samples() -> None:
    detector = COMDirectionalDetector(
        COMDirectionalConfig(
            target_direction_xy=(1.0, 1.0),
            minimum_displacement_m=0.02,
            minimum_projected_velocity_m_s=0.001,
            maximum_lateral_displacement_m=0.01,
            required_consecutive_samples=2,
        )
    )
    assert not detector.update((0.0, 0.0), (0.0, 0.0)).satisfied
    first = detector.update((0.015, 0.015), (0.01, 0.01))
    second = detector.update((0.016, 0.016), (0.01, 0.01))
    assert first.projected_displacement_m > 0.02
    assert not first.satisfied
    assert second.satisfied

    detector.reset((0.0, 0.0))
    wrong_way = detector.update((-0.03, -0.03), (-0.01, -0.01))
    assert not wrong_way.satisfied
    assert wrong_way.projected_displacement_m < 0.0


def _observation(
    *,
    time_s: float,
    com_x: float = 0.0,
    com_vx: float = 0.0,
    fl_point: tuple[float, float] = (0.0, 0.0),
    rr_point: tuple[float, float] = (1.0, 0.0),
    rl_load: float = 8.0,
    fr_load: float = 3.0,
    angular_speed: float = 0.5,
) -> TransferObservation:
    return TransferObservation(
        time_s=time_s,
        com_xy=(com_x, 0.0),
        com_velocity_xy=(com_vx, 0.0),
        contact_points_xy={
            Leg.FL: fl_point,
            Leg.FR: (0.0, 1.0),
            Leg.RL: (1.0, 1.0),
            Leg.RR: rr_point,
        },
        contact_active={leg: True for leg in Leg},
        contact_load_n={Leg.FL: 10.0, Leg.FR: fr_load, Leg.RL: rl_load, Leg.RR: 10.0},
        roll_deg=1.0,
        pitch_deg=1.0,
        angular_speed_deg_s=angular_speed,
    )


def test_anchored_transfer_requires_direction_and_rejects_contact_drift() -> None:
    config = AnchoredTransferConfig(
        support_legs=(Leg.FL, Leg.RR),
        target_direction_xy=(1.0, 0.0),
        minimum_com_displacement_m=0.02,
        maximum_contact_drift_m=0.01,
        minimum_support_load_n=5.0,
        maximum_roll_deg=10.0,
        maximum_pitch_deg=10.0,
        maximum_angular_speed_deg_s=5.0,
        required_consecutive_samples=2,
    )
    guard = AnchoredSupportAngleGuard(config)
    assert not guard.update(_observation(time_s=0.0)).satisfied
    assert not guard.update(_observation(time_s=0.1, com_x=0.021, com_vx=0.01)).satisfied
    assert guard.update(_observation(time_s=0.2, com_x=0.022, com_vx=0.01)).satisfied

    guard.reset()
    guard.update(_observation(time_s=0.0))
    drifted = guard.update(
        _observation(time_s=0.1, com_x=0.03, fl_point=(0.02, 0.0))
    )
    assert drifted.abort
    assert "drift" in drifted.reason


def test_impulse_guard_has_distinct_preload_push_release_coast_settle_and_verify_evidence() -> None:
    guard = ImpulseReactionGuard(
        ImpulseTransferConfig(
            support_legs=(Leg.FL, Leg.RR),
            impulse_leg=Leg.RL,
            target_leg=Leg.FR,
            swing_leg=Leg.RL,
            target_direction_xy=(1.0, 0.0),
            minimum_support_load_n=5.0,
            minimum_preload_force_n=7.0,
            minimum_push_velocity_m_s=0.02,
            minimum_push_displacement_m=0.002,
            minimum_transfer_displacement_m=0.02,
            maximum_release_load_ratio=0.5,
            maximum_settle_velocity_m_s=0.005,
            maximum_settle_angular_speed_deg_s=2.0,
            minimum_target_load_increase_n=4.0,
            maximum_swing_load_share=0.1,
            maximum_roll_deg=15.0,
            maximum_pitch_deg=15.0,
            required_consecutive_samples=2,
        )
    )

    assert not guard.update(ImpulseStage.PRELOAD, _observation(time_s=0.0)).satisfied
    assert guard.update(ImpulseStage.PRELOAD, _observation(time_s=0.1)).satisfied
    assert not guard.update(
        ImpulseStage.PUSH,
        _observation(time_s=0.2, com_x=0.004, com_vx=0.03),
    ).satisfied
    assert guard.update(
        ImpulseStage.PUSH,
        _observation(time_s=0.3, com_x=0.006, com_vx=0.03),
    ).satisfied
    release_not_ready = guard.update(
        ImpulseStage.RELEASE,
        _observation(time_s=0.31, com_x=0.007, com_vx=0.025, rl_load=6.0),
    )
    assert not release_not_ready.satisfied
    assert "has not released" in release_not_ready.reason
    assert not guard.update(
        ImpulseStage.RELEASE,
        _observation(time_s=0.32, com_x=0.008, com_vx=0.02, rl_load=3.0),
    ).satisfied
    assert guard.update(
        ImpulseStage.RELEASE,
        _observation(time_s=0.33, com_x=0.009, com_vx=0.02, rl_load=3.0),
    ).satisfied
    coast_not_ready = guard.update(
        ImpulseStage.COAST,
        _observation(time_s=0.34, com_x=0.015, com_vx=0.015, rl_load=3.0),
    )
    assert not coast_not_ready.satisfied
    assert "below target" in coast_not_ready.reason
    assert not guard.update(
        ImpulseStage.COAST,
        _observation(time_s=0.35, com_x=0.021, com_vx=0.01, rl_load=3.0),
    ).satisfied
    assert guard.update(
        ImpulseStage.COAST,
        _observation(time_s=0.36, com_x=0.022, com_vx=0.008, rl_load=3.0),
    ).satisfied
    assert not guard.update(
        ImpulseStage.SETTLE,
        _observation(time_s=0.4, com_x=0.025, com_vx=0.003, angular_speed=1.0),
    ).satisfied
    assert guard.update(
        ImpulseStage.SETTLE,
        _observation(time_s=0.5, com_x=0.026, com_vx=0.002, angular_speed=1.0),
    ).satisfied
    assert not guard.update(
        ImpulseStage.VERIFY,
        _observation(time_s=0.6, com_x=0.026, fr_load=8.0, rl_load=1.0),
    ).satisfied
    assert guard.update(
        ImpulseStage.VERIFY,
        _observation(time_s=0.7, com_x=0.026, fr_load=8.0, rl_load=1.0),
    ).satisfied


def test_stale_target_can_relatch_once_before_convergence_only() -> None:
    latch = SingleRelatchTarget(geometry_change_threshold_m=0.01)
    initial = latch.update((0.1, 0.1), (Leg.FL, Leg.RR), support_geometry_change_m=0.0)
    assert initial.target_xy == (0.1, 0.1)
    second = latch.update(
        (0.2, 0.1),
        (Leg.FR, Leg.RL),
        support_geometry_change_m=0.02,
        reason="diagonal changed",
    )
    assert second.relatched
    assert second.relatch_count == 1
    exhausted = latch.update(
        (0.3, 0.1),
        (Leg.FL, Leg.RR),
        support_geometry_change_m=0.02,
    )
    assert not exhausted.relatched
    assert exhausted.target_xy == (0.2, 0.1)

    latch.reset()
    latch.update((0.1, 0.1), (Leg.FL, Leg.RR), support_geometry_change_m=0.0)
    frozen = latch.update(
        (0.2, 0.1),
        (Leg.FR, Leg.RL),
        support_geometry_change_m=0.02,
        convergence_started=True,
    )
    assert not frozen.relatched
    assert frozen.frozen


def test_offsets_are_carried_across_state_and_cleared_by_rate_limit() -> None:
    manager = ContinuousOffsetManager()
    manager.initialize({"rear_left_knee": 12.0})
    manager.enter_state("SHIFT", {"rear_left_knee": 12.0})
    assert manager.current_offsets["rear_left_knee"] == 12.0
    carried = manager.enter_state("UNLOAD")
    assert carried["rear_left_knee"] == 12.0
    manager.request_clear()
    updated = manager.update(0.1, maximum_rate_deg_s=20.0)
    assert updated["rear_left_knee"] == 10.0

    fresh = ContinuousOffsetManager()
    entered = fresh.enter_state("SHIFT", {"rear_left_knee": 12.0})
    assert entered["rear_left_knee"] == 0.0
    assert fresh.update(0.1, maximum_rate_deg_s=20.0)["rear_left_knee"] == 2.0


def test_wheel_ramp_is_scoped_to_named_state() -> None:
    ramp = StateScopedWheelRamp(frozenset({"RL_SWING"}), 1.0)
    unchanged_prefix = ramp.update("FR_PREFIX", {"fl": 0.8}, dt_s=0.1)
    assert unchanged_prefix["fl"] == 0.8
    first = ramp.update(
        "RL_SWING", {"fl": 1.2}, dt_s=0.1, entry_velocity_rad_s={"fl": 0.0}
    )
    second = ramp.update("RL_SWING", {"fl": 1.2}, dt_s=0.1)
    assert first["fl"] == pytest.approx(0.1)
    assert second["fl"] == pytest.approx(0.2)
    not_global = ramp.update("FR_PLACE", {"fl": 1.2}, dt_s=0.1)
    assert not_global["fl"] == 1.2


def test_lift_to_swing_rejects_clearance_loss() -> None:
    rejected = evaluate_lift_to_swing_clearance(
        lift_peak_clearance_m=0.018,
        current_clearance_m=0.017,
        predicted_swing_clearance_m=0.006,
        minimum_clearance_m=0.01,
        maximum_clearance_drop_m=0.008,
    )
    assert not rejected.allowed
    accepted = evaluate_lift_to_swing_clearance(
        lift_peak_clearance_m=0.018,
        current_clearance_m=0.017,
        predicted_swing_clearance_m=0.014,
        minimum_clearance_m=0.01,
        maximum_clearance_drop_m=0.008,
    )
    assert accepted.allowed


def test_place_requires_continuous_top_load_dwell() -> None:
    guard = PlaceLoadDwellGuard(5.0, 0.2, required_consecutive_samples=3)
    assert not guard.update(time_s=0.0, contact_class="TOP", load_n=6.0).satisfied
    assert not guard.update(time_s=0.1, contact_class="TOP", load_n=6.0).satisfied
    assert guard.update(time_s=0.2, contact_class="TOP", load_n=6.0).satisfied
    assert not guard.update(time_s=0.3, contact_class="AIR", load_n=0.0).satisfied
    assert guard.load_started_at_s is None


def _successful_final_observation() -> FinalSuccessObservation:
    return FinalSuccessObservation(
        leg_traversal_valid={leg: True for leg in Leg},
        leg_airborne_seen={leg: True for leg in Leg},
        illegal_drive_up={leg: False for leg in Leg},
        final_contact_class={leg: "TOP" for leg in Leg},
        final_load_n={leg: 10.0 for leg in Leg},
        final_top_load_dwell_s={leg: 0.5 for leg in Leg},
        home_error_deg={"joint_1": 0.5, "joint_2": -0.4},
        roll_deg=1.0,
        pitch_deg=-1.0,
        root_linear_speed_m_s=0.005,
        root_angular_speed_deg_s=0.5,
        forward_clearance_m=0.2,
        concurrent_home_recovery_verified=True,
    )


def test_final_success_requires_airborne_top_load_settle_and_concurrent_recovery() -> None:
    criteria = FinalSuccessCriteria(
        minimum_top_load_n=5.0,
        minimum_top_load_dwell_s=0.2,
        maximum_home_error_deg=2.0,
        maximum_roll_deg=5.0,
        maximum_pitch_deg=5.0,
        maximum_root_linear_speed_m_s=0.01,
        maximum_root_angular_speed_deg_s=1.0,
        minimum_forward_clearance_m=0.1,
    )
    assert evaluate_final_success(_successful_final_observation(), criteria).satisfied

    base = _successful_final_observation()
    failed = FinalSuccessObservation(
        **{
            **base.__dict__,
            "leg_airborne_seen": {**base.leg_airborne_seen, Leg.RL: False},
            "concurrent_home_recovery_verified": False,
        }
    )
    decision = evaluate_final_success(failed, criteria)
    assert not decision.satisfied
    assert any("RL has no AIRBORNE" in item for item in decision.unmet)
    assert any("concurrent" in item for item in decision.unmet)

    non_finite = FinalSuccessObservation(
        **{**base.__dict__, "root_linear_speed_m_s": math.nan}
    )
    assert not evaluate_final_success(non_finite, criteria).satisfied


def test_guard_inputs_reject_non_finite_direction() -> None:
    with pytest.raises(ValueError):
        COMDirectionalDetector(
            COMDirectionalConfig((math.nan, 0.0), minimum_displacement_m=0.1)
        )
