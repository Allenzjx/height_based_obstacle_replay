from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

from command_model import WHEEL_FORWARD_SIGN, WHEEL_JOINT_NAMES
from playback import PlaybackPlan, PlaybackSegment

from fsm_50mm_recording_derived_v3.wheel_integral_evidence import (
    AVAILABLE,
    FAIL,
    NOT_EVALUABLE,
    PASS,
    PHYSICS_DT_S,
    evaluate_wheel_integral_evidence,
)


def _wheel_map(value: float) -> dict[str, float]:
    return {name: float(value) for name in WHEEL_JOINT_NAMES}


def _segment(
    index: int,
    *,
    target: float | dict[str, float],
    duration_ticks: int,
) -> PlaybackSegment:
    targets = _wheel_map(target) if isinstance(target, (int, float)) else dict(target)
    start = float(Fraction(index * duration_ticks, 120))
    duration = float(Fraction(duration_ticks, 120))
    return PlaybackSegment(
        segment_index=index,
        source_step=index,
        source_step_id=f"step-{index}",
        event_start_index=index,
        event_count=1,
        planned_start_s=start,
        planned_end_s=start + duration,
        base_duration_s=duration,
        servo_base_duration_s=0.0,
        servo_duration_s=0.0,
        servo_targets={},
        wheel_active_duration_s=duration,
        wheel_base_velocity=dict(targets),
        wheel_requested_velocity_rad_s=dict(targets),
        wheel_applied_target_rad_s=dict(targets),
    )


def _plan(*segments: PlaybackSegment) -> PlaybackPlan:
    return PlaybackPlan(
        path=Path("synthetic.jsonl"),
        events=[],
        final_time_s=max((segment.planned_end_s for segment in segments), default=0.0),
        label="wheel-integral-test",
        plan_sha256="a" * 64,
        segments=list(segments),
    )


def _batch(
    *,
    batch_id: str,
    kind: str,
    first_step: int,
    targets: dict[str, float],
    segment_index: int | None,
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "dispatch_kind": kind,
        "segment_index": segment_index,
        "wheel_targets_rad_s": dict(targets),
        "ack_present": True,
        "ack_valid": True,
        "ack_error": "",
        "adapter_called": True,
        "applied_sim_step": first_step - 1,
        "first_physics_step": first_step,
        "motion_start_skew_s": 0.0,
    }


def _active_target(
    batches: list[dict[str, object]], step: int
) -> dict[str, float]:
    active = _wheel_map(0.0)
    for batch in batches:
        if int(batch["first_physics_step"]) <= step:
            active = dict(batch["wheel_targets_rad_s"])
        else:
            break
    return active


def _rows(
    batches: list[dict[str, object]],
    *,
    first_step: int,
    last_step: int,
    wheel_direction: float = 1.0,
    positions: dict[int, dict[str, float]] | None = None,
    velocities: dict[int, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    positions = positions or {}
    velocities = velocities or {}
    direction = -1.0 if wheel_direction < 0.0 else 1.0
    for step in range(first_step, last_step + 1):
        canonical_target = _active_target(batches, step)
        raw_target = {
            name: canonical_target[name]
            / (direction * float(WHEEL_FORWARD_SIGN[name]))
            for name in WHEEL_JOINT_NAMES
        }
        rows.append(
            {
                "sim_step": step,
                "physics_dt_s": PHYSICS_DT_S,
                "wheel_direction": wheel_direction,
                "wheel_forward_sign": {
                    name: float(WHEEL_FORWARD_SIGN[name])
                    for name in WHEEL_JOINT_NAMES
                },
                "physx_drive_target_evidence_valid": True,
                "physx_joint_velocity_target_rad_s": raw_target,
                "measured_joint_position_rad": dict(
                    positions.get(step, _wheel_map(0.0))
                ),
                "measured_joint_velocity_rad_s": dict(
                    velocities.get(step, _wheel_map(0.0))
                ),
            }
        )
    return rows


def _one_segment_case(
    *,
    target: float,
    duration_ticks: int,
    final_stop_step: int,
) -> tuple[PlaybackPlan, dict[str, object], list[dict[str, object]]]:
    plan = _plan(_segment(0, target=target, duration_ticks=duration_ticks))
    batches = [
        _batch(
            batch_id="segment-0",
            kind="source_segment_start",
            first_step=10,
            targets=_wheel_map(target),
            segment_index=0,
        ),
        _batch(
            batch_id="final-stop",
            kind="final_safety_stop",
            first_step=final_stop_step,
            targets=_wheel_map(0.0),
            segment_index=0,
        ),
    ]
    rows = _rows(batches, first_step=9, last_step=final_stop_step)
    return plan, {"motion_batches": batches}, rows


def test_ack_boundaries_are_inclusive_then_exclusive_per_segment() -> None:
    plan = _plan(
        _segment(0, target=1.0, duration_ticks=2),
        _segment(1, target=2.0, duration_ticks=2),
    )
    batches = [
        _batch(
            batch_id="segment-0",
            kind="source_segment_start",
            first_step=10,
            targets=_wheel_map(1.0),
            segment_index=0,
        ),
        _batch(
            batch_id="segment-1",
            kind="source_segment_start",
            first_step=12,
            targets=_wheel_map(2.0),
            segment_index=1,
        ),
        _batch(
            batch_id="final-stop",
            kind="final_safety_stop",
            first_step=14,
            targets=_wheel_map(0.0),
            segment_index=1,
        ),
    ]
    result = evaluate_wheel_integral_evidence(
        plan=plan,
        timing_trace={"motion_batches": batches},
        telemetry_rows=_rows(batches, first_step=9, last_step=14),
    )

    assert result["target_integral_verdict"] == PASS
    assert result["measured_unwrap_status"] == AVAILABLE
    assert result["overall_verdict"] == NOT_EVALUABLE
    assert result["segments"][0]["physics_tick_count"] == 2
    assert result["segments"][1]["physics_tick_count"] == 2
    assert math.isclose(
        result["segments"][0]["target_integrals"]["front_right_ankle"][
            "physx_target_integral_rad"
        ],
        2.0 / 120.0,
    )
    assert math.isclose(
        result["segments"][1]["target_integrals"]["front_right_ankle"][
            "physx_target_integral_rad"
        ],
        4.0 / 120.0,
    )
    assert [row["segment_index"] for row in result["ticks"]] == [
        0,
        0,
        1,
        1,
        None,
    ]


def test_existing_left_right_wheel_signs_canonicalize_targets_and_q() -> None:
    plan, timing, _ = _one_segment_case(
        target=1.0, duration_ticks=2, final_stop_step=12
    )
    batches = list(timing["motion_batches"])
    positions: dict[int, dict[str, float]] = {9: _wheel_map(0.0)}
    velocities: dict[int, dict[str, float]] = {9: _wheel_map(0.0)}
    for step, amount in ((10, 0.005), (11, 0.010)):
        positions[step] = {
            name: amount / float(WHEEL_FORWARD_SIGN[name])
            for name in WHEEL_JOINT_NAMES
        }
        velocities[step] = {
            name: 1.0 / float(WHEEL_FORWARD_SIGN[name])
            for name in WHEEL_JOINT_NAMES
        }
    rows = _rows(
        batches,
        first_step=9,
        last_step=12,
        positions=positions,
        velocities=velocities,
    )
    result = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=rows
    )

    assert result["target_integral_verdict"] == PASS
    for name in WHEEL_JOINT_NAMES:
        assert result["ticks"][0]["physx_target_rad_s"][name] == 1.0
        assert math.isclose(
            result["measured_rotation"][name]["net_rotation_rad"], 0.010
        )


def test_one_tick_integral_error_passes_but_two_ticks_fail() -> None:
    one_plan, one_timing, one_rows = _one_segment_case(
        target=1.0, duration_ticks=2, final_stop_step=13
    )
    one = evaluate_wheel_integral_evidence(
        plan=one_plan, timing_trace=one_timing, telemetry_rows=one_rows
    )
    assert one["target_integral_verdict"] == PASS
    assert one["segments"][0]["physics_tick_count"] == 3
    assert math.isclose(
        abs(one["full_target_integrals"]["front_right_ankle"]["error_rad"]),
        1.0 / 120.0,
    )

    two_plan, two_timing, two_rows = _one_segment_case(
        target=1.0, duration_ticks=2, final_stop_step=14
    )
    two = evaluate_wheel_integral_evidence(
        plan=two_plan, timing_trace=two_timing, telemetry_rows=two_rows
    )
    assert two["target_integral_verdict"] == FAIL
    assert two["overall_verdict"] == FAIL
    assert two["segments"][0]["physics_tick_count"] == 4


def test_zero_plan_target_requires_exact_zero_not_nextafter() -> None:
    plan, timing, rows = _one_segment_case(
        target=0.0, duration_ticks=2, final_stop_step=12
    )
    tiny = math.nextafter(0.0, math.inf)
    rows[1]["physx_joint_velocity_target_rad_s"]["front_right_ankle"] = tiny
    result = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=rows
    )

    comparison = result["segments"][0]["target_integrals"][
        "front_right_ankle"
    ]
    assert comparison["zero_target_requires_exact_zero"] is True
    assert comparison["verdict"] == FAIL
    assert result["target_integral_verdict"] == FAIL


def test_two_pi_and_four_pi_storage_wraps_use_principal_delta_and_report_tv() -> None:
    plan, timing, _ = _one_segment_case(
        target=0.0, duration_ticks=3, final_stop_step=13
    )
    batches = list(timing["motion_batches"])
    positions = {step: _wheel_map(0.0) for step in range(9, 14)}
    velocities = {step: _wheel_map(0.0) for step in range(9, 14)}
    positions[9]["front_right_ankle"] = 2.0 * math.pi - 0.005
    positions[9]["rear_right_ankle"] = 4.0 * math.pi - 0.005
    for name in ("front_right_ankle", "rear_right_ankle"):
        positions[10][name] = 0.005
        positions[11][name] = 0.015
        positions[12][name] = 0.005
        velocities[9][name] = 1.2
        velocities[10][name] = 1.2
        velocities[11][name] = 1.2
        velocities[12][name] = -1.2
    result = evaluate_wheel_integral_evidence(
        plan=plan,
        timing_trace=timing,
        telemetry_rows=_rows(
            batches,
            first_step=9,
            last_step=13,
            positions=positions,
            velocities=velocities,
        ),
    )

    assert result["measured_unwrap_status"] == AVAILABLE
    for name in ("front_right_ankle", "rear_right_ankle"):
        metrics = result["measured_rotation"][name]
        assert math.isclose(metrics["net_rotation_rad"], 0.01, abs_tol=1.0e-12)
        assert math.isclose(metrics["total_variation_rad"], 0.03, abs_tol=1.0e-12)
        assert math.isclose(metrics["forward_rotation_rad"], 0.02, abs_tol=1.0e-12)
        assert math.isclose(metrics["reverse_rotation_rad"], 0.01, abs_tol=1.0e-12)
        assert math.isclose(
            metrics["zero_target_total_variation_rad"], 0.03, abs_tol=1.0e-12
        )
    assert result["measured_tracking_verdict"] == NOT_EVALUABLE
    assert "no registered" in result["measured_tracking_reason"]


def test_invalid_ack_missing_tick_and_nan_target_are_not_evaluable() -> None:
    plan, timing, rows = _one_segment_case(
        target=1.0, duration_ticks=2, final_stop_step=12
    )
    invalid_timing = {"motion_batches": [dict(row) for row in timing["motion_batches"]]}
    invalid_timing["motion_batches"][0]["ack_valid"] = False
    invalid_ack = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=invalid_timing, telemetry_rows=rows
    )
    assert invalid_ack["target_integral_verdict"] == NOT_EVALUABLE

    missing_tick = evaluate_wheel_integral_evidence(
        plan=plan,
        timing_trace=timing,
        telemetry_rows=[row for row in rows if row["sim_step"] != 11],
    )
    assert missing_tick["target_integral_verdict"] == NOT_EVALUABLE
    assert any("missing" in reason for reason in missing_tick["target_not_evaluable_reasons"])

    nan_rows = [dict(row) for row in rows]
    nan_rows[1] = dict(nan_rows[1])
    nan_rows[1]["physx_joint_velocity_target_rad_s"] = dict(
        nan_rows[1]["physx_joint_velocity_target_rad_s"]
    )
    nan_rows[1]["physx_joint_velocity_target_rad_s"]["rear_right_ankle"] = float("nan")
    nan_target = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=nan_rows
    )
    assert nan_target["target_integral_verdict"] == NOT_EVALUABLE


def test_batch_identity_mismatch_and_nonexact_dt_are_not_evaluable() -> None:
    plan, timing, rows = _one_segment_case(
        target=1.0, duration_ticks=2, final_stop_step=12
    )
    identity_timing = {
        "motion_batches": [dict(row) for row in timing["motion_batches"]]
    }
    identity_timing["motion_batches"][0]["wheel_targets_rad_s"] = _wheel_map(1.0)
    identity_timing["motion_batches"][0]["wheel_targets_rad_s"][
        "front_left_ankle"
    ] = 0.5
    identity = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=identity_timing, telemetry_rows=rows
    )
    assert identity["target_integral_verdict"] == NOT_EVALUABLE
    assert any("identity mismatch" in reason for reason in identity["structural_errors"])

    wrong_dt_rows = [dict(row) for row in rows]
    wrong_dt_rows[1] = dict(wrong_dt_rows[1])
    wrong_dt_rows[1]["physics_dt_s"] = math.nextafter(PHYSICS_DT_S, math.inf)
    wrong_dt = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=wrong_dt_rows
    )
    assert wrong_dt["target_integral_verdict"] == NOT_EVALUABLE
    assert any("exactly 1/120" in reason for reason in wrong_dt["target_not_evaluable_reasons"])


def test_nan_q_or_insufficient_speed_bound_only_invalidates_measured_unwrap() -> None:
    plan, timing, rows = _one_segment_case(
        target=0.0, duration_ticks=2, final_stop_step=12
    )
    nan_q_rows = [dict(row) for row in rows]
    nan_q_rows[1] = dict(nan_q_rows[1])
    nan_q_rows[1]["measured_joint_position_rad"] = dict(
        nan_q_rows[1]["measured_joint_position_rad"]
    )
    nan_q_rows[1]["measured_joint_position_rad"]["front_right_ankle"] = float("nan")
    nan_q = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=nan_q_rows
    )
    assert nan_q["target_integral_verdict"] == PASS
    assert nan_q["measured_unwrap_status"] == NOT_EVALUABLE

    ambiguous_rows = [dict(row) for row in rows]
    ambiguous_rows[1] = dict(ambiguous_rows[1])
    ambiguous_rows[1]["measured_joint_position_rad"] = dict(
        ambiguous_rows[1]["measured_joint_position_rad"]
    )
    ambiguous_rows[1]["measured_joint_position_rad"]["front_right_ankle"] = 0.10
    ambiguous = evaluate_wheel_integral_evidence(
        plan=plan, timing_trace=timing, telemetry_rows=ambiguous_rows
    )
    assert ambiguous["target_integral_verdict"] == PASS
    assert ambiguous["measured_unwrap_status"] == NOT_EVALUABLE
    assert any(
        "speed bound" in reason
        for reason in ambiguous["measured_not_evaluable_reasons"]
    )
