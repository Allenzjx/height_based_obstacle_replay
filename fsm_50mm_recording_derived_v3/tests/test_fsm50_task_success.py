from __future__ import annotations

import math

import pytest

from fsm_50mm_recording_derived_v3.fsm50_task_success import (
    EVALUATED,
    NOT_EVALUATED,
    POSTURE_COMPLETE,
    POSTURE_INCOMPLETE,
    POSTURE_NOT_APPLICABLE,
    REPLAY_TASK_FAIL,
    REPLAY_TASK_NOT_EVALUATED,
    REPLAY_TASK_SUCCESS,
    REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE,
    TABLE_COLUMNS,
    ManualVideoVerdict,
    classify_replay_task,
)


LEGS = ("FL", "FR", "RL", "RR")
SERVO_JOINTS = (
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
)
WHEEL_JOINTS = (
    "front_left_ankle",
    "front_right_ankle",
    "rear_left_ankle",
    "rear_right_ankle",
)
ALL_JOINTS = (*SERVO_JOINTS, *WHEEL_JOINTS)


def _completed_result() -> dict[str, object]:
    return {
        "schema_version": "fsm50.recording_replay_result.v1",
        "source_version": "v003_20260805_224517_157723_manual",
        "dispatch_complete": True,
        "scheduler_complete": True,
        "scheduler_stop_reason": "complete",
        "timed_out": False,
        "simulation_app_stopped": False,
        "motion_start_ready": True,
        "lifecycle": {"failed": False, "finalized": True},
        # These old strict outcomes are deliberately not Gate-A task outcomes.
        "classification": "PARTIAL_SUCCESS",
        "physical_success": False,
        "strict_full_success": False,
        "plan_event_count": 160,
        "plan_segment_count": 112,
        "scheduler_status": {
            "count": 160,
            "event_count": 160,
            "events_sent": 160,
            "index": 160,
            "segment_count": 112,
            "segment_index": 112,
            "progress_detail": {"total_steps": 24},
        },
        "maximum_abs_roll_rad": 0.18,
        "maximum_abs_pitch_rad": 0.15,
        "actual_viewport_video": True,
        "video_path": "C:/runs/v003/actual_viewport_video.mp4",
        "video": {
            "actual_viewport_video": True,
            "artifact_valid": True,
            "full_decode": {"valid": True},
            "full_decode_all_frames": True,
            "valid": True,
        },
    }


def _traversal() -> dict[str, object]:
    return {
        # This intentionally stays false: the legacy strict linkage episode
        # criterion is not the task-level definition of whether lifting occurred.
        "all_legs_valid": False,
        "any_illegal_drive_up": False,
        "legs": {
            leg: {
                "airborne_seen_before_top": True,
                "front_face_crossing_s": 10.0 + index,
                "linkage_lift_valid": index in (1, 3),
            }
            for index, leg in enumerate(LEGS)
        },
    }


def _physical_evidence() -> dict[str, object]:
    return {
        "schema_version": "fsm50.physical_evidence.v2",
        "source_version": "v003_20260805_224517_157723_manual",
        "evidence_complete": True,
        "physical_success": False,
        "strict_criteria": {
            "all_legs_linkage_lift_valid": False,
            "contact_drift_safe": False,
            "final_all_loaded": False,
            "final_all_top": False,
            "final_velocity_stable": True,
        },
        "criteria": {
            "all_legs_linkage_lift_valid": {
                "availability": "AVAILABLE",
                "passed": False,
            },
            "attitude_safe": {"availability": "AVAILABLE", "passed": True},
            "collision_safe": {"availability": "AVAILABLE", "passed": True},
            "contact_drift_safe": {
                "availability": "AVAILABLE",
                "passed": False,
            },
            "final_all_loaded": {"availability": "AVAILABLE", "passed": False},
            "final_all_top": {"availability": "AVAILABLE", "passed": False},
            "final_velocity_stable": {
                "availability": "AVAILABLE",
                "passed": True,
            },
            "joint_limits_safe": {"availability": "AVAILABLE", "passed": True},
            "no_illegal_drive_up": {"availability": "AVAILABLE", "passed": True},
            "penetration_safe": {"availability": "AVAILABLE", "passed": True},
        },
        "traversal": _traversal(),
        "final_wheel_contact_classes": {
            "FL": "AIR",
            "FR": "TOP",
            "RL": "TOP",
            "RR": "TOP",
        },
        "final_all_top": False,
        "final_all_loaded": False,
        "final_velocity_stable": True,
        "maximum_contact_drift_m": 0.0345,
        "maximum_abs_roll_rad": 0.18,
        "maximum_abs_pitch_rad": 0.15,
        "dangerous_collision_count": 0,
        "joint_limit_violation_count": 0,
        "maximum_collision_penetration_m": 0.0002,
        "maximum_allowed_penetration_m": 0.003,
    }


def _final_telemetry() -> dict[str, object]:
    return {
        "source_version": "v003_20260805_224517_157723_manual",
        # Exact normal-development worker state/target shapes.  These coexist
        # with the legacy flat fields until all completed artifacts migrate.
        "robot_state_finite": True,
        "base_position_m": {"x": 0.84, "y": 0.0, "z": 0.21},
        "base_quaternion_wxyz": [0.997, 0.065, -0.018, 0.001],
        "base_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "joint_q_rad": {name: 0.0 for name in ALL_JOINTS},
        "joint_qd_rad_s": {name: 0.0 for name in ALL_JOINTS},
        "servo_targets_deg": {name: 0.0 for name in SERVO_JOINTS},
        "wheel_targets_rad_s": {name: 0.0 for name in WHEEL_JOINTS},
        "base_roll_rad": 0.13,
        "base_pitch_rad": -0.04,
        "base_x_m": 0.84,
        "stability_state": "safe",
        "active_contact_count": 3,
        "dangerous_collision": False,
        "joint_limit_violation": False,
        "maximum_collision_penetration_m": 0.00001,
        "wheel_contact_classes": {
            "FL": "AIR",
            "FR": "TOP",
            "RL": "TOP",
            "RR": "TOP",
        },
        "wheel_front_face_clearance_m": {
            "FL": 0.54,
            "FR": 0.66,
            "RL": 0.14,
            "RR": 0.19,
        },
    }


def _classify(
    result: dict[str, object] | None = None,
    physical: dict[str, object] | None = None,
    telemetry: dict[str, object] | None = None,
    video: ManualVideoVerdict | dict[str, object] | None = None,
):
    return classify_replay_task(
        completed_result=result if result is not None else _completed_result(),
        physical_evidence=physical if physical is not None else _physical_evidence(),
        final_telemetry_row=telemetry if telemetry is not None else _final_telemetry(),
        video_verdict=video,
    )


def test_strict_physical_failures_do_not_redefine_completed_traversal() -> None:
    assessment = _classify()

    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert assessment.posture_result == POSTURE_INCOMPLETE
    assert assessment.body_crossed_front_face is True
    assert assessment.required_leg_lift_completed is True
    assert assessment.final_recoverable is True
    assert assessment.hard_failure_reasons == ()
    assert "CONTACT_DRIFT_DIAGNOSTIC_FAILED" in assessment.secondary_diagnostics
    assert "FINAL_SINGLE_WHEEL_AIR" in assessment.secondary_diagnostics
    assert "LEGACY_STRICT_LIFT_CRITERION_FAILED" in assessment.secondary_diagnostics


def test_contact_drift_alone_is_secondary_and_posture_can_be_complete() -> None:
    physical = _physical_evidence()
    physical["final_wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}
    physical["final_all_top"] = True
    physical["final_all_loaded"] = True
    physical["final_velocity_stable"] = True
    physical["final_posture_complete"] = True
    telemetry = _final_telemetry()
    telemetry["active_contact_count"] = 4
    telemetry["wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.task_result == REPLAY_TASK_SUCCESS
    assert assessment.posture_result == POSTURE_COMPLETE
    assert assessment.hard_failure_reasons == ()
    assert "CONTACT_DRIFT_DIAGNOSTIC_FAILED" in assessment.secondary_diagnostics


def test_strict_rest_failure_is_posture_incomplete_not_task_failure() -> None:
    physical = _physical_evidence()
    physical["final_wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}
    physical["final_all_top"] = True
    physical["final_all_loaded"] = True
    physical["final_velocity_stable"] = False
    telemetry = _final_telemetry()
    telemetry["active_contact_count"] = 4
    telemetry["wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert "STRICT_REST_INCOMPLETE" in assessment.secondary_diagnostics
    assert assessment.hard_failure_reasons == ()


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda r, p, t: r.__setitem__("dispatch_complete", False),
            "COMMAND_DISPATCH_INCOMPLETE",
        ),
        (
            lambda r, p, t: r.__setitem__("scheduler_complete", False),
            "SCHEDULER_INCOMPLETE",
        ),
        (lambda r, p, t: p.__setitem__("robot_fell", True), "ROBOT_FALL"),
        (lambda r, p, t: p.__setitem__("body_stuck", True), "BODY_STUCK"),
        (
            lambda r, p, t: p.__setitem__(
                "body_crossed_front_face", False
            ),
            "BODY_NOT_CROSSED_FRONT_FACE",
        ),
        (
            lambda r, p, t: p.__setitem__(
                "required_leg_lift_completed", False
            ),
            "REQUIRED_LEG_LIFT_MISSING",
        ),
        (
            lambda r, p, t: p.__setitem__("wheel_drive_up_without_required_lift", True),
            "WHEEL_DRIVE_UP_WITHOUT_REQUIRED_LIFT",
        ),
        (
            lambda r, p, t: t.__setitem__("dangerous_collision", True),
            "DANGEROUS_BODY_COLLISION",
        ),
        (
            lambda r, p, t: t.__setitem__(
                "joint_limit_violation", True
            ),
            "JOINT_LIMIT_VIOLATION",
        ),
        (lambda r, p, t: p.__setitem__("severe_penetration", True), "SEVERE_PENETRATION"),
        (
            lambda r, p, t: p.__setitem__("active_leg_trapped", True),
            "ACTIVE_LEG_TRAPPED",
        ),
        (
            lambda r, p, t: t.__setitem__("unsafe_joint_target", True),
            "UNSAFE_JOINT_TARGET",
        ),
        (
            lambda r, p, t: p.__setitem__("final_recoverable", False),
            "FINAL_STATE_IRRECOVERABLE",
        ),
        (
            lambda r, p, t: r.__setitem__(
                "actuator_targets_applied", False
            ),
            "ACTUATOR_TARGET_NOT_APPLIED",
        ),
    ],
)
def test_every_hard_failure_forces_task_fail(mutator, reason: str) -> None:
    result = _completed_result()
    physical = _physical_evidence()
    telemetry = _final_telemetry()
    mutator(result, physical, telemetry)

    assessment = _classify(result=result, physical=physical, telemetry=telemetry)

    assert assessment.task_result == REPLAY_TASK_FAIL
    assert assessment.evaluation_status == EVALUATED
    assert assessment.posture_result == POSTURE_NOT_APPLICABLE
    assert reason in assessment.hard_failure_reasons


def test_observed_dispatch_failure_wins_over_missing_video_evidence() -> None:
    result = _completed_result()
    result["dispatch_complete"] = False
    result.pop("video")

    assessment = _classify(result=result)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert "COMMAND_DISPATCH_INCOMPLETE" in assessment.hard_failure_reasons
    assert "VIDEO_EVIDENCE_UNAVAILABLE" in assessment.not_evaluated_reasons


def test_observed_fall_wins_over_missing_penetration_evidence() -> None:
    physical = _physical_evidence()
    telemetry = _final_telemetry()
    physical["robot_fell"] = True
    physical["criteria"].pop("penetration_safe")
    physical.pop("maximum_collision_penetration_m")
    physical.pop("maximum_allowed_penetration_m")
    telemetry.pop("maximum_collision_penetration_m")

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert "ROBOT_FALL" in assessment.hard_failure_reasons
    assert "EVIDENCE_MISSING:severe_penetration" in assessment.not_evaluated_reasons


def test_infrastructure_failure_still_wins_over_observed_hard_failure() -> None:
    result = _completed_result()
    physical = _physical_evidence()
    result["timed_out"] = True
    physical["robot_fell"] = True

    assessment = _classify(result=result, physical=physical)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert "INFRASTRUCTURE_PROCESS_FAILURE" in assessment.not_evaluated_reasons
    assert "ROBOT_FALL" in assessment.hard_failure_reasons


def test_type_conflict_still_wins_over_observed_hard_failure() -> None:
    physical = _physical_evidence()
    physical["robot_fell"] = True
    physical["final_wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}
    telemetry = _final_telemetry()
    telemetry["wheel_contact_classes"]["FL"] = "AIR"

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert "EVIDENCE_CONFLICT:final_wheel_classes" in assessment.not_evaluated_reasons
    assert "ROBOT_FALL" in assessment.hard_failure_reasons


def test_nonfinite_nested_evidence_fails_closed() -> None:
    telemetry = _final_telemetry()
    telemetry["base_roll_rad"] = math.nan

    assessment = _classify(telemetry=telemetry)

    assert assessment.task_result == REPLAY_TASK_FAIL
    assert any(
        reason.startswith("NONFINITE_CORE_STATE:")
        for reason in assessment.hard_failure_reasons
    )


def test_nonfinite_task_geometry_is_not_mislabeled_as_task_failure() -> None:
    telemetry = _final_telemetry()
    telemetry["wheel_front_face_clearance_m"]["RL"] = math.inf

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert not any(
        reason.startswith("NONFINITE_CORE_STATE:")
        for reason in assessment.hard_failure_reasons
    )


def test_nonfinite_joint_state_is_a_hard_failure() -> None:
    telemetry = _final_telemetry()
    telemetry["measured_joint_position_rad"] = {
        "front_left_hip": math.nan,
    }

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert any(
        reason.startswith("NONFINITE_CORE_STATE:")
        for reason in assessment.hard_failure_reasons
    )


def test_normal_session_robot_state_finite_false_is_a_hard_failure() -> None:
    telemetry = _final_telemetry()
    telemetry["robot_state_finite"] = False

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert (
        "NONFINITE_CORE_STATE:final_telemetry_row.robot_state_finite"
        in assessment.hard_failure_reasons
    )


def test_explicit_nonfinite_core_state_marker_is_a_hard_failure() -> None:
    result = _completed_result()
    result["nonfinite_core_state_detected"] = True

    assessment = _classify(result=result)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert (
        "NONFINITE_CORE_STATE:completed_result.nonfinite_core_state_detected"
        in assessment.hard_failure_reasons
    )


@pytest.mark.parametrize(
    "field",
    [
        "base_position_m",
        "base_quaternion_wxyz",
        "base_linear_velocity_m_s",
        "base_angular_velocity_rad_s",
        "joint_q_rad",
        "joint_qd_rad_s",
        "servo_targets_deg",
        "wheel_targets_rad_s",
    ],
)
def test_present_null_normal_session_core_field_is_a_hard_failure(
    field: str,
) -> None:
    telemetry = _final_telemetry()
    telemetry[field] = None

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert (
        f"NONFINITE_CORE_STATE:final_telemetry_row.{field}"
        in assessment.hard_failure_reasons
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("base_position_m", {"x": 0.84, "y": 0.0}),
        ("base_quaternion_wxyz", [1.0, 0.0, 0.0]),
        ("base_linear_velocity_m_s", [0.0, 0.0]),
        ("base_angular_velocity_rad_s", [0.0, 0.0, 0.0, 0.0]),
        ("joint_q_rad", {name: 0.0 for name in ALL_JOINTS[:-1]}),
        ("joint_qd_rad_s", {name: 0.0 for name in ALL_JOINTS[:-1]}),
        ("servo_targets_deg", {name: 0.0 for name in SERVO_JOINTS[:-1]}),
        ("wheel_targets_rad_s", {name: 0.0 for name in WHEEL_JOINTS[:-1]}),
    ],
)
def test_malformed_normal_session_core_shape_is_not_evaluated(
    field: str, invalid_value: object
) -> None:
    telemetry = _final_telemetry()
    telemetry[field] = invalid_value

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert (
        f"EVIDENCE_SHAPE_ERROR:final_telemetry_row.{field}"
        in assessment.not_evaluated_reasons
    )


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("base_position_m", lambda value: value.__setitem__("z", math.nan)),
        ("base_quaternion_wxyz", lambda value: value.__setitem__(2, math.inf)),
        ("joint_q_rad", lambda value: value.__setitem__(ALL_JOINTS[3], math.nan)),
        (
            "joint_qd_rad_s",
            lambda value: value.__setitem__(ALL_JOINTS[7], -math.inf),
        ),
        (
            "servo_targets_deg",
            lambda value: value.__setitem__(SERVO_JOINTS[1], math.nan),
        ),
        (
            "wheel_targets_rad_s",
            lambda value: value.__setitem__(WHEEL_JOINTS[2], math.inf),
        ),
    ],
)
def test_nested_nonfinite_normal_session_core_field_is_a_hard_failure(
    field: str, mutate
) -> None:
    telemetry = _final_telemetry()
    mutate(telemetry[field])

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert any(
        reason.startswith(f"NONFINITE_CORE_STATE:final_telemetry_row.{field}")
        for reason in assessment.hard_failure_reasons
    )


def test_null_optional_contact_loads_do_not_become_core_failure() -> None:
    telemetry = _final_telemetry()
    telemetry["wheel_contact_load_n"] = {leg: None for leg in LEGS}
    telemetry["wheel_contact_load_available"] = False

    assessment = _classify(telemetry=telemetry)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert not any(
        reason.startswith("NONFINITE_CORE_STATE:")
        for reason in assessment.hard_failure_reasons
    )


def test_nonfinite_strict_contact_diagnostic_is_secondary_only() -> None:
    physical = _physical_evidence()
    physical["maximum_contact_drift_m"] = math.nan

    assessment = _classify(physical=physical)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert any(
        reason.startswith("SECONDARY_DIAGNOSTIC_UNAVAILABLE:")
        for reason in assessment.secondary_diagnostics
    )


def test_malformed_strict_only_contact_record_is_secondary_only() -> None:
    physical = _physical_evidence()
    physical["criteria"]["contact_drift_safe"]["passed"] = "false"

    assessment = _classify(physical=physical)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert (
        "SECONDARY_DIAGNOSTIC_UNAVAILABLE:"
        "physical_evidence.criteria.contact_drift_safe.passed"
        in assessment.secondary_diagnostics
    )


def test_bool_fields_do_not_accept_integer_truthiness() -> None:
    result = _completed_result()
    result["dispatch_complete"] = 1

    assessment = _classify(result=result)

    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert assessment.evaluation_status == NOT_EVALUATED
    assert (
        "EVIDENCE_TYPE_ERROR:completed_result.dispatch_complete"
        in assessment.not_evaluated_reasons
    )


@pytest.mark.parametrize(
    ("counter", "bad_value", "reason"),
    [
        ("count", 159, "SCHEDULER_EVENT_COUNT_MISMATCH"),
        ("event_count", 159, "SCHEDULER_EVENT_COUNT_MISMATCH"),
        ("events_sent", 159, "SCHEDULER_EVENT_COUNT_MISMATCH"),
        ("index", 159, "SCHEDULER_EVENT_COUNT_MISMATCH"),
        ("segment_count", 111, "SCHEDULER_SEGMENT_COUNT_MISMATCH"),
        ("segment_index", 111, "SCHEDULER_SEGMENT_COUNT_MISMATCH"),
    ],
)
def test_scheduler_counts_must_exactly_match_the_plan(
    counter: str,
    bad_value: int,
    reason: str,
) -> None:
    result = _completed_result()
    result["scheduler_status"][counter] = bad_value

    assessment = _classify(result=result)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert reason in assessment.hard_failure_reasons


def test_dispatch_ledger_plan_count_conflict_is_task_fail() -> None:
    result = _completed_result()
    result["dispatch_ledger"] = {
        "complete": True,
        "plan_event_count": 159,
        "plan_segment_count": 112,
    }

    assessment = _classify(result=result)

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_FAIL
    assert "DISPATCH_LEDGER_PLAN_COUNT_MISMATCH" in assessment.hard_failure_reasons


def test_conflicting_final_wheel_classes_are_not_evaluated() -> None:
    physical = _physical_evidence()
    physical["final_wheel_contact_classes"] = {leg: "TOP" for leg in LEGS}
    telemetry = _final_telemetry()
    telemetry["wheel_contact_classes"]["FL"] = "AIR"

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert (
        "EVIDENCE_CONFLICT:final_wheel_classes"
        in assessment.not_evaluated_reasons
    )


@pytest.mark.parametrize(
    "field",
    [
        "second_isaac_instance",
        "second_simulator_process",
        "second_simulator_process_detected",
    ],
)
def test_second_simulator_process_is_infrastructure_not_evaluated(
    field: str,
) -> None:
    result = _completed_result()
    result[field] = True

    assessment = _classify(result=result)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert "SECOND_SIMULATOR_PROCESS" in assessment.not_evaluated_reasons
    assert assessment.hard_failure_reasons == ()


def test_motion_start_not_ready_is_not_a_robot_task_failure() -> None:
    result = _completed_result()
    result["motion_start_ready"] = False

    assessment = _classify(result=result)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert "MOTION_START_NOT_READY" in assessment.not_evaluated_reasons
    assert "ACTUATOR_TARGET_NOT_APPLIED" not in assessment.hard_failure_reasons


def test_null_optional_sensor_evidence_can_be_supplemented() -> None:
    physical = _physical_evidence()
    telemetry = _final_telemetry()
    optional_fields = (
        "robot_fell",
        "fall_detected",
        "body_stuck",
        "permanently_stuck",
        "illegal_drive_up",
        "dangerous_collision",
        "joint_limit_violation",
        "severe_penetration",
        "active_leg_trapped",
        "leg_trapped_by_obstacle",
        "unsafe_joint_target",
        "joint_target_outside_safe_limit",
    )
    for field in optional_fields:
        physical[field] = None
        telemetry[field] = None
    physical["final_posture_complete"] = None
    physical["final_home_pose_complete"] = None
    physical["contact_drift_safe"] = None
    physical["traversal"]["any_illegal_drive_up"] = None
    for leg in LEGS:
        physical["traversal"]["legs"][leg][
            "airborne_seen_before_top"
        ] = None
        physical["traversal"]["legs"][leg]["front_face_crossing_s"] = None
    physical["criteria"]["no_illegal_drive_up"]["passed"] = None

    assessment = _classify(
        physical=physical,
        telemetry=telemetry,
        video=ManualVideoVerdict(
            task_completed=True,
            body_crossed_front_face=True,
            required_leg_lift_completed=True,
            final_recoverable=True,
            posture_incomplete=True,
            robot_fell=False,
            body_stuck=False,
            wheel_drive_up_without_required_lift=False,
            dangerous_body_collision=False,
            joint_limit_violation=False,
            severe_penetration=False,
            irrecoverable=False,
        ),
    )

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert not any(
        reason.startswith("EVIDENCE_TYPE_ERROR:")
        for reason in assessment.not_evaluated_reasons
    )


@pytest.mark.parametrize(
    "video_value",
    [None, {"valid": False, "actual_viewport_video": True}],
)
def test_no_valid_video_and_no_manual_verdict_is_not_evaluated(
    video_value: object,
) -> None:
    result = _completed_result()
    if video_value is None:
        result.pop("video")
    else:
        result["video"] = video_value

    assessment = _classify(result=result)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert "VIDEO_EVIDENCE_UNAVAILABLE" in assessment.not_evaluated_reasons


def test_manual_video_verdict_can_replace_missing_video_artifact() -> None:
    result = _completed_result()
    result.pop("video")

    assessment = _classify(
        result=result,
        video=ManualVideoVerdict(
            task_completed=True,
            body_crossed_front_face=True,
            required_leg_lift_completed=True,
            final_recoverable=True,
            posture_incomplete=True,
        ),
    )

    assert assessment.evaluation_status == EVALUATED
    assert assessment.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE


def test_missing_core_evidence_fails_closed() -> None:
    physical = _physical_evidence()
    physical.pop("traversal")
    telemetry = _final_telemetry()
    telemetry.pop("stability_state")
    telemetry.pop("active_contact_count")

    assessment = _classify(physical=physical, telemetry=telemetry)

    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert assessment.evaluation_status == NOT_EVALUATED
    assert "EVIDENCE_MISSING:required_leg_lift_completed" in assessment.not_evaluated_reasons
    assert "EVIDENCE_MISSING:final_recoverable" in assessment.not_evaluated_reasons


@pytest.mark.parametrize(
    "mutator",
    [
        lambda r: r["lifecycle"].__setitem__("failed", True),
        lambda r: r.__setitem__("timed_out", True),
        lambda r: r.__setitem__("simulation_app_stopped", True),
        lambda r: r.__setitem__("artifact_valid", False),
    ],
)
def test_infrastructure_failures_are_not_mislabeled_as_robot_task_fail(mutator) -> None:
    result = _completed_result()
    mutator(result)

    assessment = _classify(result=result)

    assert assessment.evaluation_status == NOT_EVALUATED
    assert assessment.task_result == REPLAY_TASK_NOT_EVALUATED
    assert assessment.posture_result == POSTURE_NOT_APPLICABLE
    assert assessment.hard_failure_reasons == ()
    assert assessment.not_evaluated_reasons


def test_manual_video_verdict_can_supply_missing_task_facts_but_not_override_hard_failure() -> None:
    physical = _physical_evidence()
    physical.pop("traversal")
    physical.pop("final_wheel_contact_classes")
    telemetry = _final_telemetry()
    telemetry.pop("wheel_front_face_clearance_m")
    telemetry.pop("stability_state")
    telemetry.pop("active_contact_count")

    video = ManualVideoVerdict(
        task_completed=True,
        body_crossed_front_face=True,
        required_leg_lift_completed=True,
        final_recoverable=True,
        posture_incomplete=True,
        notes=("reviewer saw all required legs clear the front face",),
    )
    supplied = _classify(physical=physical, telemetry=telemetry, video=video)
    assert supplied.task_result == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert supplied.final_recoverable is True
    assert "reviewer saw all required legs clear the front face" in supplied.notes

    telemetry["dangerous_collision"] = True
    hard_failure = _classify(physical=physical, telemetry=telemetry, video=video)
    assert hard_failure.task_result == REPLAY_TASK_FAIL
    assert "DANGEROUS_BODY_COLLISION" in hard_failure.hard_failure_reasons


def test_manual_video_confirmed_failure_is_fail_closed() -> None:
    assessment = _classify(
        video={
            "task_completed": False,
            "body_crossed_front_face": False,
            "required_leg_lift_completed": False,
            "final_recoverable": False,
            "first_actual_failure_phase": "RL_FACE_CROSS",
            "notes": ["RL remained behind the face"],
        }
    )

    assert assessment.task_result == REPLAY_TASK_FAIL
    assert "MANUAL_VIDEO_CONFIRMED_FAILURE" in assessment.hard_failure_reasons
    assert assessment.first_actual_failure_phase == "RL_FACE_CROSS"


def test_output_contains_stable_table_columns_and_explanations() -> None:
    assessment = _classify()
    row = assessment.to_table_row()

    assert tuple(row) == TABLE_COLUMNS
    assert row["version"] == "v003_20260805_224517_157723_manual"
    assert row["step_count"] == 24
    assert row["fast_segment_count"] == 112
    assert row["task_result"] == REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE
    assert row["posture_result"] == POSTURE_INCOMPLETE
    assert row["final_wheel_classes"] == "FL=AIR;FR=TOP;RL=TOP;RR=TOP"
    assert row["peak_roll"] == pytest.approx(0.18)
    assert row["peak_pitch"] == pytest.approx(0.15)
    assert row["video_path"].endswith("actual_viewport_video.mp4")
    assert row["first_actual_failure_phase"] == ""
    assert "FINAL_SINGLE_WHEEL_AIR" in row["notes"]
    assert "task core complete" in row["classification_reasons"]


def test_top_level_inputs_must_be_mappings() -> None:
    with pytest.raises(TypeError, match="completed_result must be a mapping"):
        classify_replay_task(  # type: ignore[arg-type]
            completed_result=[],
            physical_evidence=_physical_evidence(),
            final_telemetry_row=_final_telemetry(),
        )
