"""Fail-closed startup evidence for production recording replay.

``REST_QUALIFICATION`` remains the strict passive-settle diagnostic and keeps
the existing 0.02 rad/s servo threshold.  ``MOTION_START_READY`` is a separate
read-only decision matching the production Fast Replay lifecycle: the runtime
may be ready to execute a verified plan while passive servo motion prevents a
strict-rest PASS.  No threshold in this module changes physics or writes robot
state.
"""

from __future__ import annotations

import math
import hashlib
import json
from typing import Any, Mapping

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from robot_ground_diagnostics import (
    GROUND_OK,
    VISUAL_ONLY_INTERSECTION,
    motion_status_from_worker_status,
)


LEGS = ("FL", "FR", "RL", "RR")
SAFE_GROUND_CLASSIFICATIONS = {GROUND_OK, VISUAL_ONLY_INTERSECTION}
LEG_TO_WHEEL_BODY = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}


def _nested_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value.detach().cpu().tolist())
    except Exception:
        try:
            return list(value)
        except Exception:
            return []


def _obstacle_relative_pose(
    adapter: Any,
    obstacle_geometry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    geometry = dict(obstacle_geometry or {})
    robot = getattr(adapter, "robot", None)
    data = getattr(robot, "data", None)
    names = [str(name) for name in list(getattr(robot, "body_names", []) or [])]
    states = _nested_list(getattr(data, "body_link_state_w", None))
    if not states:
        states = _nested_list(getattr(data, "body_state_w", None))
    while len(states) == 1 and isinstance(states[0], list):
        states = states[0]
    positions: dict[str, list[float]] = {}
    errors: list[str] = []
    if not names or len(states) != len(names):
        errors.append(
            "robot body names/state rows are unavailable or have mismatched length"
        )
    else:
        for index, name in enumerate(names):
            row = list(states[index] or []) if isinstance(states[index], list) else []
            parsed = [_finite(value) for value in row[:3]]
            if len(parsed) != 3 or not all(math.isfinite(value) for value in parsed):
                errors.append(f"{name}: body position is unavailable/non-finite")
                continue
            positions[name] = parsed
    try:
        front_x = float(geometry["front_face_x_m"])
        top_z = float(geometry.get("top_z_m", geometry["height_m"]))
        center_y = float(geometry.get("center_y_m", 0.0))
    except (KeyError, TypeError, ValueError):
        front_x = top_z = center_y = float("nan")
        errors.append("obstacle front/top/center geometry is unavailable")
    relative: dict[str, dict[str, Any]] = {}
    centers: dict[str, list[float]] = {}
    for leg, body in LEG_TO_WHEEL_BODY.items():
        center = positions.get(body)
        if center is None:
            errors.append(f"{leg}: required wheel body {body!r} is unavailable")
            continue
        centers[leg] = center
        relative[leg] = {
            "center_w": center,
            "x_from_obstacle_front_m": center[0] - front_x,
            "y_from_obstacle_center_m": center[1] - center_y,
            "z_from_obstacle_top_m": center[2] - top_z,
        }
    return {
        "valid": bool(
            not errors
            and len(centers) == len(LEG_TO_WHEEL_BODY)
            and all(math.isfinite(value) for value in (front_x, top_z, center_y))
        ),
        "error": "; ".join(dict.fromkeys(errors)),
        "source": "live articulation body_link_state_w + measured obstacle geometry",
        "obstacle_geometry": geometry,
        "wheel_centers_w": centers,
        "wheel_obstacle_relative_pose": relative,
    }


def capture_live_motion_start_snapshot(
    adapter: Any,
    scene_handle: Any,
    obstacle_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture and enrich one current pre-dispatch frame without state writes."""

    from .grounding_diagnostics import enrich_grounding_tick

    capture = getattr(adapter, "capture_motion_start_base_evidence", None)
    if not callable(capture):
        return {
            "sim_step": getattr(adapter, "sim_steps", None),
            "sim_time_s": getattr(adapter, "sim_time", None),
            "diagnostic_evidence_valid": False,
            "diagnostic_evidence_error": (
                "adapter.capture_motion_start_base_evidence is unavailable"
            ),
        }
    try:
        base = dict(capture() or {})
    except Exception as exc:
        return {
            "sim_step": getattr(adapter, "sim_steps", None),
            "sim_time_s": getattr(adapter, "sim_time", None),
            "diagnostic_evidence_valid": False,
            "diagnostic_evidence_error": (
                "motion-start base capture failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    try:
        enriched = enrich_grounding_tick(adapter, scene_handle, base)
        enriched["obstacle_relative_pose"] = _obstacle_relative_pose(
            adapter, obstacle_geometry
        )
        return enriched
    except Exception as exc:
        base["diagnostic_evidence_valid"] = False
        base["diagnostic_evidence_error"] = (
            "motion-start enrichment failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return base


def _finite(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def _finite_mapping(
    value: Any,
    expected: tuple[str, ...],
) -> tuple[dict[str, float], str]:
    if not isinstance(value, Mapping):
        return {}, "value is not a mapping"
    raw = dict(value)
    if set(raw) != set(expected):
        return {}, (
            f"keys do not exactly match required set; missing={sorted(set(expected) - set(raw))} "
            f"unexpected={sorted(set(raw) - set(expected))}"
        )
    parsed = {name: _finite(raw[name]) for name in expected}
    nonfinite = [name for name, number in parsed.items() if not math.isfinite(number)]
    if nonfinite:
        return parsed, "non-finite values: " + ", ".join(nonfinite)
    return parsed, ""


def rest_qualification_summary(ground_reference: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize, but never reinterpret, the strict grounding result."""

    ground = dict(ground_reference or {})
    diagnostics = dict(ground.get("grounded_reference_diagnostics", {}) or {})
    passed = bool(
        ground.get("grounded_reference_valid") is True
        and ground.get("grounded_reference_physics_valid") is True
        and ground.get("grounded_reference_stable") is True
        and diagnostics.get("checked") is True
        and diagnostics.get("physical_ground_safe") is True
        and str(diagnostics.get("classification", ""))
        in SAFE_GROUND_CLASSIFICATIONS
    )
    return {
        "gate": "REST_QUALIFICATION",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "servo_speed_threshold_rad_s": _finite(
            ground.get("servo_speed_threshold_rad_s")
        ),
        "observed_servo_speed_rad_s": _finite(
            ground.get("acceptance_max_servo_joint_velocity_rad_s")
            if ground.get("acceptance_max_servo_joint_velocity_rad_s") is not None
            else ground.get("final_servo_joint_velocity_rad_s")
        ),
        "stable_frames": int(ground.get("stable_frames", 0) or 0),
        "stable_frames_required": int(
            ground.get("stable_frames_required", 0) or 0
        ),
        "block_reason": str(
            diagnostics.get("ground_reference_block_reason", "")
            or ground.get("ground_reference_block_reason", "")
        ),
    }


def evaluate_motion_start_ready(
    *,
    ground_reference: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    production_runtime_ready: bool,
    expected_sim_step: int | None = None,
    expected_adapter_runtime_instance_id: str = "",
    plan_identity: Mapping[str, Any] | None = None,
    command_dispatch_idle: bool = True,
    root_seed_applied: bool = False,
    vertical_speed_limit_m_s: float = 0.01,
    wheel_speed_limit_rad_s: float = 0.20,
    penetration_limit_m: float = 0.003,
    maximum_roll_rad: float = math.radians(45.0),
    maximum_pitch_rad: float = math.radians(45.0),
    maximum_angular_velocity_rad_s: float = 3.0,
    target_readback_tolerance: float = 1.0e-6,
    support_force_min_n: float = 1.0e-6,
) -> dict[str, Any]:
    """Evaluate a fresh pre-dispatch snapshot without requiring strict rest.

    Only the existing production worker motion policy, existing ground
    penetration limit, zero-wheel start boundary, and evidence-integrity
    checks are authoritative at this stage.  Attitude, root angular velocity,
    servo velocity, and measured loads are recorded for the future success
    envelope; they are deliberately not promoted into guessed startup limits
    before three successful v003 runs exist.
    """

    frame = dict(snapshot or {})
    ground = dict(ground_reference or {})
    diagnostics = dict(ground.get("grounded_reference_diagnostics", {}) or {})
    penetration = dict(frame.get("penetration", {}) or {})
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, **evidence: Any) -> None:
        checks[name] = {"passed": bool(passed), **evidence}

    check(
        "production_runtime_ready",
        production_runtime_ready is True,
        observed=bool(production_runtime_ready),
        provenance="production worker publishes ready independent of REST_QUALIFICATION",
    )
    check(
        "no_historical_root_seed",
        root_seed_applied is False
        and int(frame.get("root_state_write_count", 0) or 0) == 0
        and not list(frame.get("root_state_write_events", []) or []),
        observed=bool(root_seed_applied),
        root_state_write_count=int(frame.get("root_state_write_count", 0) or 0),
        root_state_write_events=list(frame.get("root_state_write_events", []) or []),
    )
    observed_instance_id = str(frame.get("adapter_runtime_instance_id", "") or "")
    check(
        "fresh_adapter_instance",
        bool(observed_instance_id)
        and (
            not expected_adapter_runtime_instance_id
            or observed_instance_id == str(expected_adapter_runtime_instance_id)
        ),
        expected_adapter_runtime_instance_id=str(
            expected_adapter_runtime_instance_id or ""
        ),
        observed_adapter_runtime_instance_id=observed_instance_id,
    )
    check(
        "fresh_snapshot",
        expected_sim_step is None
        or int(frame.get("sim_step", -1)) == int(expected_sim_step),
        expected_sim_step=expected_sim_step,
        observed_sim_step=frame.get("sim_step"),
    )
    identity = dict(plan_identity or {})
    identity_errors: list[str] = []
    for key in (
        "source_version",
        "source_sha256",
        "plan_sha256",
        "plan_id",
        "request_id",
        "worker_session_id",
    ):
        if not str(identity.get(key, "") or ""):
            identity_errors.append(f"{key} is missing")
    for key in ("event_count", "segment_count"):
        try:
            if int(identity.get(key, 0) or 0) <= 0:
                identity_errors.append(f"{key} is not positive")
        except (TypeError, ValueError):
            identity_errors.append(f"{key} is invalid")
    check(
        "plan_identity_bound_and_no_prior_dispatch",
        command_dispatch_idle is True and not identity_errors,
        command_dispatch_idle=bool(command_dispatch_idle),
        plan_identity=identity,
        error="; ".join(identity_errors),
    )
    expected_command_state = dict(
        identity.get("source_initial_command_state", {}) or {}
    )
    live_command_state = dict(frame.get("command_state", {}) or {})
    expected_command_text = json.dumps(
        expected_command_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    live_command_text = json.dumps(
        live_command_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_command_sha = hashlib.sha256(
        expected_command_text.encode("utf-8")
    ).hexdigest()
    live_command_sha = hashlib.sha256(
        live_command_text.encode("utf-8")
    ).hexdigest()
    declared_command_sha = str(
        identity.get("source_initial_command_state_sha256", "") or ""
    ).lower()
    check(
        "live_command_state_matches_source_initial_state",
        bool(expected_command_state)
        and declared_command_sha == expected_command_sha
        and live_command_sha == expected_command_sha,
        declared_source_initial_command_state_sha256=declared_command_sha,
        computed_source_initial_command_state_sha256=expected_command_sha,
        live_command_state_sha256=live_command_sha,
        expected_command_state=expected_command_state,
        live_command_state=live_command_state,
    )
    observer_error = str(frame.get("diagnostic_observer_error", "") or "")
    check(
        "diagnostic_evidence_complete",
        frame.get("diagnostic_evidence_valid") is True and not observer_error,
        error=str(frame.get("diagnostic_evidence_error", "") or observer_error),
    )
    ground_classification = str(
        penetration.get("classification", "")
        or diagnostics.get("classification", "")
        or ""
    )
    live_ground_checked = bool(
        penetration.get("valid") is True
        or diagnostics.get("checked") is True
    )
    live_ground_safe = bool(
        penetration.get("physical_ground_safe") is True
        if "physical_ground_safe" in penetration
        else diagnostics.get("physical_ground_safe") is True
    )
    worker_motion_ready, worker_motion_reason, worker_ground_state = (
        motion_status_from_worker_status(
            {
                "runtime_ready": production_runtime_ready is True,
                "ready": production_runtime_ready is True,
                "robot_ground": {
                    "classification": ground_classification,
                    "checked": live_ground_checked,
                    "physical_ground_safe": live_ground_safe,
                },
            }
        )
    )
    check(
        "production_worker_motion_policy",
        worker_motion_ready,
        ground_state=worker_ground_state,
        reason=worker_motion_reason,
        provenance=(
            "robot_ground_diagnostics.motion_status_from_worker_status; "
            "does not require REST_QUALIFICATION"
        ),
    )
    check(
        "ground_geometry_safe",
        live_ground_checked
        and live_ground_safe
        and ground_classification in SAFE_GROUND_CLASSIFICATIONS,
        checked=live_ground_checked,
        physical_ground_safe=live_ground_safe,
        classification=ground_classification,
        evidence_source=(
            "fresh penetration snapshot"
            if penetration.get("valid") is True
            else "ground initialization diagnostics fallback"
        ),
    )
    support_clearance = dict(
        ground.get("ground_support_clearance_evidence", {}) or {}
    )
    check(
        "ground_contact_resolved",
        ground.get("ground_contact_resolved") is True
        and support_clearance.get("valid") is True,
        ground_contact_resolved=ground.get("ground_contact_resolved"),
        support_clearance_evidence=support_clearance,
        provenance="shared SimRobotAdapter grounding contact/clearance evidence",
    )

    root_velocity = list(frame.get("root_velocity", []) or [])
    root_velocity_values = [_finite(value) for value in root_velocity]
    root_velocity_valid = bool(
        frame.get("root_velocity_evidence_valid") is True
        and len(root_velocity_values) == 6
        and all(math.isfinite(value) for value in root_velocity_values)
    )
    vertical_speed = (
        abs(root_velocity_values[2]) if root_velocity_valid else float("nan")
    )
    angular_speed = (
        math.sqrt(sum(value * value for value in root_velocity_values[3:6]))
        if root_velocity_valid
        else float("nan")
    )
    check(
        "root_motion_evidence_complete",
        root_velocity_valid,
        vertical_speed_m_s=vertical_speed,
        vertical_speed_limit_m_s=float(vertical_speed_limit_m_s),
        angular_speed_rad_s=angular_speed,
        angular_speed_limit_rad_s=float(maximum_angular_velocity_rad_s),
        limits_are_envelope_gate=False,
        error=str(frame.get("root_velocity_evidence_error", "") or ""),
    )
    roll = _finite(frame.get("roll_rad"))
    pitch = _finite(frame.get("pitch_rad"))
    check(
        "attitude_evidence_complete",
        math.isfinite(roll)
        and math.isfinite(pitch),
        roll_rad=roll,
        pitch_rad=pitch,
        maximum_roll_rad=float(maximum_roll_rad),
        maximum_pitch_rad=float(maximum_pitch_rad),
        limits_are_envelope_gate=False,
    )
    obstacle_relative = dict(frame.get("obstacle_relative_pose", {}) or {})
    check(
        "obstacle_relative_pose_complete",
        obstacle_relative.get("valid") is True
        and set(
            dict(obstacle_relative.get("wheel_obstacle_relative_pose", {}) or {})
        )
        == set(LEGS),
        evidence=obstacle_relative,
    )

    joint_pos, joint_pos_error = _finite_mapping(
        frame.get("joint_position_by_name"),
        tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES),
    )
    joint_vel, joint_vel_error = _finite_mapping(
        frame.get("joint_velocity_by_name"),
        tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES),
    )
    physx_pos, physx_pos_error = _finite_mapping(
        frame.get("joint_position_target_by_name"),
        tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES),
    )
    servo_command, servo_command_error = _finite_mapping(
        frame.get("servo_command_target_by_name"), tuple(SERVO_JOINT_NAMES)
    )
    servo_readback_error, servo_error_error = _finite_mapping(
        frame.get("servo_command_to_readback_error_by_name"),
        tuple(SERVO_JOINT_NAMES),
    )
    joint_errors = [
        error
        for error in (
            joint_pos_error,
            joint_vel_error,
            physx_pos_error,
            servo_command_error,
            servo_error_error,
            str(frame.get("joint_state_evidence_error", "") or ""),
        )
        if error
    ]
    max_servo_command_error = max(
        (abs(value) for value in servo_readback_error.values()), default=float("nan")
    )
    check(
        "joint_and_physx_position_targets_valid",
        frame.get("joint_state_evidence_valid") is True
        and not joint_errors
        and max_servo_command_error <= float(target_readback_tolerance),
        maximum_servo_command_to_physx_error_rad=max_servo_command_error,
        tolerance_rad=float(target_readback_tolerance),
        error="; ".join(joint_errors),
        joint_count=len(joint_pos),
        physx_target_count=len(physx_pos),
        servo_command_count=len(servo_command),
    )
    wheel_measured_speed = max(
        (abs(joint_vel.get(name, float("nan"))) for name in WHEEL_JOINT_NAMES),
        default=float("nan"),
    )
    check(
        "wheel_measured_motion_safe",
        math.isfinite(wheel_measured_speed)
        and wheel_measured_speed <= float(wheel_speed_limit_rad_s),
        maximum_wheel_speed_rad_s=wheel_measured_speed,
        limit_rad_s=float(wheel_speed_limit_rad_s),
    )

    logical_wheels, logical_error = _finite_mapping(
        frame.get("wheel_target_velocity_by_name"), tuple(WHEEL_JOINT_NAMES)
    )
    physx_wheels, physx_wheel_error = _finite_mapping(
        frame.get("wheel_target_readback_velocity_by_name"), tuple(WHEEL_JOINT_NAMES)
    )
    wheel_errors, wheel_error_error = _finite_mapping(
        frame.get("wheel_target_command_to_readback_error_by_name"),
        tuple(WHEEL_JOINT_NAMES),
    )
    wheel_target_errors = [
        error
        for error in (
            logical_error,
            physx_wheel_error,
            wheel_error_error,
            str(frame.get("wheel_target_evidence_error", "") or ""),
        )
        if error
    ]
    check(
        "wheel_targets_zero_and_physx_verified",
        frame.get("wheel_target_evidence_valid") is True
        and not wheel_target_errors
        and all(abs(value) <= float(target_readback_tolerance) for value in logical_wheels.values())
        and all(abs(value) <= float(target_readback_tolerance) for value in physx_wheels.values())
        and all(abs(value) <= float(target_readback_tolerance) for value in wheel_errors.values()),
        logical_targets=logical_wheels,
        physx_targets=physx_wheels,
        command_to_physx_error=wheel_errors,
        tolerance=float(target_readback_tolerance),
        error="; ".join(wheel_target_errors),
    )

    wheel_forces = dict(frame.get("wheel_net_forces_w", {}) or {})
    support_errors: list[str] = []
    upward_force: dict[str, float] = {}
    if set(wheel_forces) != set(LEGS):
        support_errors.append("wheel force keys do not exactly match FL/FR/RL/RR")
    for leg in LEGS:
        raw_vector = list(wheel_forces.get(leg, []) or [])
        vector = [_finite(value) for value in raw_vector]
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            support_errors.append(f"{leg} wheel force vector is incomplete/non-finite")
            continue
        upward_force[leg] = max(0.0, vector[2])
    check(
        "four_wheel_force_evidence_complete",
        frame.get("wheel_force_evidence_valid") is True
        and not support_errors
        and len(upward_force) == len(LEGS),
        upward_force_n=upward_force,
        minimum_force_n=None,
        support_force_min_n_argument_recorded_only=float(support_force_min_n),
        learned_load_envelope_gate=False,
        error="; ".join(support_errors)
        or str(frame.get("wheel_force_evidence_error", "") or ""),
    )
    nonwheel_contacts = list(frame.get("nonwheel_contacts", []) or [])
    check(
        "no_nonwheel_contact",
        not nonwheel_contacts,
        active_contacts=nonwheel_contacts,
    )

    penetration_value = _finite(penetration.get("maximum_collision_penetration_m"))
    check(
        "penetration_safe",
        penetration.get("valid") is True
        and penetration.get("physical_ground_safe") is True
        and math.isfinite(penetration_value)
        and penetration_value <= float(penetration_limit_m),
        maximum_collision_penetration_m=penetration_value,
        limit_m=float(penetration_limit_m),
        error=str(penetration.get("error", "") or ""),
    )

    max_servo_speed = max(
        (abs(joint_vel.get(name, float("nan"))) for name in SERVO_JOINT_NAMES),
        default=float("nan"),
    )
    ready = bool(checks and all(row["passed"] for row in checks.values()))
    failed_checks = [name for name, row in checks.items() if not row["passed"]]
    return {
        "schema_version": "fsm50.motion_start_readiness.v1",
        "gate": "MOTION_START_READY",
        "ready": ready,
        "status": "PASS" if ready else "FAIL",
        "classification": "MOTION_START_READY" if ready else "MOTION_START_BLOCKED",
        "rest_qualification": rest_qualification_summary(ground),
        "strict_rest_is_required": False,
        "observed_servo_speed_rad_s": max_servo_speed,
        "servo_speed_note": (
            "recorded for the v003 success envelope; the existing 0.02 rad/s "
            "limit remains exclusive to REST_QUALIFICATION"
        ),
        "checks": checks,
        "failed_checks": failed_checks,
        "snapshot_sim_step": frame.get("sim_step"),
        "snapshot_sim_time_s": frame.get("sim_time_s"),
        "adapter_runtime_instance_id": observed_instance_id,
        "plan_identity": identity,
        "envelope_status": "PENDING_THREE_SUCCESSFUL_V003_FAST_REPLAYS",
        "production_worker_motion_ready": bool(worker_motion_ready),
        "production_worker_motion_reason": str(worker_motion_reason or ""),
        "production_worker_ground_state": str(worker_ground_state),
        "writes_robot_state": False,
        "historical_root_pose_seeded": bool(root_seed_applied),
    }
