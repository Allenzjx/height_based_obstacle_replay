from __future__ import annotations

import hashlib
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from command_model import (
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    command_limits_for_servo,
)
from completion_aware_segment import (
    CompletionAwareSegmentExecutor,
    SegmentCompletionSpec,
    SegmentFeedback,
)
from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
    FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA,
    FEEDBACK_RECOVERY_EVIDENCE_BINDING_SCHEMA,
    FEEDBACK_RECOVERY_OBSERVATION_PAYLOAD_KEYS,
    FEEDBACK_RECOVERY_TARGET_MAP_SCHEMA,
    FeedbackRecoveryAction,
    FeedbackRecoveryObservation,
    FeedbackRecoveryStage,
    MacroFSMController as ProductionMacroFSMController,
    MacroObservation,
    MacroSegmentCompletionToken,
    MacroTerminalOutcome,
    SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
    _canonical_command_provenance,
    _empty_command_provenance,
    build_gate_c_bundle,
)
from fsm_50mm_recording_derived_v3.fsm50_centroidal_support import (
    CentroidalAngularMomentumRateMeasurement,
    CentroidalSupportEvidence,
    ContactWrenchFeasibility,
    EvidenceStatus,
    WholeBodyCOMMeasurement,
    WheelContactMeasurement,
    assess_contact_wrench_feasibility,
    assess_primary_diagonal_support,
    assess_support_region,
    unavailable_whole_body_com,
    validate_wheel_contact_frame,
)
from fsm_50mm_recording_derived_v3.fsm50_macro_state_model import (
    FINAL_RECOVERY_FEEDBACK_LIMITS,
    MacroGuardKind,
    MacroStateId,
    MacroSubphase,
    build_default_macro_graph,
)
from fsm_50mm_recording_derived_v3.fsm50_motion_profiles import (
    DEFAULT_PRIMARY_VERSION,
    MotionKeyframe,
    MotionProfileLibrary,
    PlaybackSegmentBinding,
    PhaseMotionProfile,
    build_profile_library,
    discover_successful_gate_a_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GATE_A_WHEEL_RADIUS_M = 0.04998999834060672


def _complete_servos(value: float = 0.0) -> dict[str, float]:
    return {name: value for name in SERVO_JOINT_NAMES}


def _complete_wheels(value: float = 0.0) -> dict[str, float]:
    return {name: value for name in WHEEL_JOINT_NAMES}


def _attach_strict_evidence(
    payload: dict[str, object],
    *,
    physics_time_s: float = 0.0,
    sim_step: int | None = None,
    observed_command_epoch: int = 0,
    verified_command_epoch: int | None = None,
    readback_targets: dict[str, float] | None = None,
    readback_wheel_targets: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build synthetic physical evidence for pure controller unit tests."""

    result = dict(payload)
    time_s = float(physics_time_s)
    step = (
        max(0, int(round(time_s * 1_000_000.0)))
        if sim_step is None
        else sim_step
    )
    dt = 1.0e-6 if step == 0 else time_s / step
    base_raw = result.get("base_position_m", (0.0, 0.0, 0.1))
    if isinstance(base_raw, dict):
        base = tuple(float(base_raw[key]) for key in ("x", "y", "z"))
    else:
        base = tuple(float(value) for value in base_raw)  # type: ignore[arg-type]
    evidence_base = base if len(base) == 3 else (0.0, 0.0, 0.1)
    classes = dict(result["wheel_contact_classes"])  # type: ignore[arg-type]
    centers = dict(result["wheel_center_w_m"])  # type: ignore[arg-type]
    active_centers = [
        tuple(float(value) for value in centers[leg])
        for leg in ("FL", "FR", "RL", "RR")
        if str(classes[leg]).upper() in {"GROUND", "TOP"}
    ]
    if active_centers:
        support_center_xy = (
            sum(value[0] for value in active_centers) / len(active_centers),
            sum(value[1] for value in active_centers) / len(active_centers),
        )
    else:
        support_center_xy = (0.0, 0.0)
    requested_com = result.get(
        "test_whole_body_com_position_m",
        (
            support_center_xy[0],
            support_center_xy[1],
            evidence_base[2],
        ),
    )
    com_position = tuple(float(value) for value in requested_com)  # type: ignore[arg-type]
    com_available = result.get("test_whole_body_com_available", True) is True
    com = (
        WholeBodyCOMMeasurement(
            com_measurement_available=True,
            acceleration_available=True,
            physics_tick=step,
            sim_time_s=time_s,
            physics_dt_s=dt,
            body_names=("synthetic_body",),
            body_masses_kg=(10.0,),
            total_mass_kg=10.0,
            position_w_m=com_position,
            velocity_w_m_s=(0.0, 0.0, 0.0),
            acceleration_w_m_s2=(0.0, 0.0, 0.0),
            source="unit_test.synthetic_whole_body_com",
        )
        if com_available
        else unavailable_whole_body_com(
            "unit test requested unavailable COM",
            source="unit_test.unavailable",
        )
    )
    angular = CentroidalAngularMomentumRateMeasurement(
        available=com_available,
        physics_tick=step if com_available else None,
        sim_time_s=time_s if com_available else None,
        body_names=("synthetic_body",) if com_available else (),
        angular_momentum_rate_w_nm=(0.0, 0.0, 0.0) if com_available else None,
        source="unit_test.synthetic_angular_rate",
        errors=() if com_available else ("unavailable",),
    )
    obstacle_top = float(result["obstacle_top_z_m"])
    requested_loads = dict(result["wheel_contact_load_n"])  # type: ignore[arg-type]
    active_count = sum(
        str(classes[leg]).upper() in {"GROUND", "TOP"} for leg in ("FL", "FR", "RL", "RR")
    )
    default_load = 98.1 / active_count if active_count else 0.0
    dwell_by_leg = dict(
        result.get("test_contact_dwell_s", {leg: 0.5 for leg in ("FL", "FR", "RL", "RR")})  # type: ignore[arg-type]
    )
    dwell_verified_by_leg = dict(
        result.get(
            "test_contact_dwell_verified",
            {leg: True for leg in ("FL", "FR", "RL", "RR")},
        )  # type: ignore[arg-type]
    )
    wrench_requested = result.get("test_wrench_proven", True) is True
    measurements = []
    for leg in ("FL", "FR", "RL", "RR"):
        contact_class = str(classes[leg]).upper()
        active = contact_class in {"GROUND", "TOP"}
        surface = "OBSTACLE_TOP" if contact_class == "TOP" else "GROUND" if active else "AIR"
        surface_height = obstacle_top if surface == "OBSTACLE_TOP" else 0.0 if surface == "GROUND" else None
        requested_load = requested_loads.get(leg)
        load = float(
            requested_load
            if requested_load is not None
            else default_load
            if active
            else 0.0
        )
        center = tuple(float(value) for value in centers[leg])
        point = (center[0], center[1], float(surface_height)) if active else None
        measurements.append(
            WheelContactMeasurement(
                leg=leg,
                wheel_body_name={"FL": "front_left_wheel", "FR": "front_right_wheel", "RL": "rear_left_wheel", "RR": "rear_right_wheel"}[leg],
                physics_tick=step,
                sim_time_s=time_s,
                surface_kind=surface,
                surface_height_m=surface_height,
                surface_normal_w=(0.0, 0.0, 1.0) if active else None,
                active=active,
                contact_point_w_m=point,
                normal_force_w_n=(0.0, 0.0, load),
                friction_force_w_n=(0.0, 0.0, 0.0),
                contact_moment_w_nm=(0.0, 0.0, 0.0) if active else None,
                contact_moment_model=(
                    "MEASURED"
                    if active and wrench_requested
                    else "POINT_CONTACT_ZERO_CONSERVATIVE"
                    if active
                    else ""
                ),
                dwell_s=float(dwell_by_leg[leg]) if active else None,
                surface_dwell_verified=(
                    bool(dwell_verified_by_leg[leg]) if active else False
                ),
                slip_speed_m_s=0.0 if active else None,
                contact_drift_speed_m_s=0.0 if active else None,
                friction_coefficient=0.8 if active else None,
                finite_patch_radius_m=0.03 if active else None,
                source="unit_test.synthetic_contact",
            )
        )
    contacts = validate_wheel_contact_frame(
        measurements,
        physics_tick=step,
        sim_time_s=time_s,
        physics_dt_s=dt,
    )
    support = assess_support_region(com, contacts)
    wrench = assess_contact_wrench_feasibility(
        com,
        contacts,
        angular_momentum_rate=angular,
    )
    diagonal = assess_primary_diagonal_support(
        com,
        contacts,
        active_swing_leg=str(result.get("test_active_swing_leg", "RL")),
        wrench_feasibility=wrench,
    )
    centroidal = CentroidalSupportEvidence.create(
        sim_step=step,
        physics_time_s=time_s,
        physics_dt_s=dt,
        whole_body_com=com,
        centroidal_angular_momentum_rate=angular,
        wheel_contacts=contacts,
        support_region=support,
        contact_wrench_feasibility=wrench,
        diagonal_support=diagonal,
    )
    readback = dict(readback_targets or result["servo_targets_deg"])  # type: ignore[arg-type]
    readback_wheels = dict(
        readback_wheel_targets or result["wheel_targets_rad_s"]  # type: ignore[arg-type]
    )
    result["servo_targets_deg"] = readback
    result["wheel_targets_rad_s"] = readback_wheels
    measured = dict(result.get("test_measured_servo_positions_deg", readback))  # type: ignore[arg-type]
    velocities = dict(result.get("test_measured_servo_velocities_deg_s", _complete_servos()))  # type: ignore[arg-type]
    margins = dict(result.get("test_joint_limit_margin_deg", _complete_servos(20.0)))  # type: ignore[arg-type]
    feedback = FeedbackRecoveryObservation.create(
        sim_step=step,
        physics_time_s=time_s,
        observed_command_epoch=observed_command_epoch,
        n_plus_one_verified=verified_command_epoch is not None,
        verified_command_epoch=verified_command_epoch,
        readback_servo_targets_deg=readback,
        readback_wheel_targets_rad_s=readback_wheels,
        measured_servo_positions_deg=measured,
        measured_servo_velocities_deg_s=velocities,
        joint_limit_margin_deg=margins,
        base_position_m=evidence_base,
        base_roll_rad=(
            float(result["base_roll_rad"])
            if math.isfinite(float(result["base_roll_rad"]))
            else 0.0
        ),
        base_pitch_rad=(
            float(result["base_pitch_rad"])
            if math.isfinite(float(result["base_pitch_rad"]))
            else 0.0
        ),
        base_angular_velocity_rad_s=(
            result["base_angular_velocity_rad_s"]
            if isinstance(result["base_angular_velocity_rad_s"], (tuple, list))
            and len(result["base_angular_velocity_rad_s"]) == 3  # type: ignore[arg-type]
            and all(
                math.isfinite(float(value))
                for value in result["base_angular_velocity_rad_s"]  # type: ignore[union-attr]
            )
            else (0.0, 0.0, 0.0)
        ),
        wheel_center_w_m=centers,
        wheel_front_face_clearance_m=result["wheel_front_face_clearance_m"],  # type: ignore[arg-type]
        wheel_top_clearance_m=result["wheel_top_clearance_m"],  # type: ignore[arg-type]
        obstacle_front_face_x_m=float(result["obstacle_front_face_x_m"]),
        obstacle_top_z_m=float(result["obstacle_top_z_m"]),
        body_crossed_front_face=bool(result["body_crossed_front_face"]),
        final_recoverable=bool(result["final_recoverable"]),
        posture_complete=bool(result["posture_complete"]),
    )
    result["centroidal_support_evidence"] = centroidal.to_mapping()
    result["feedback_recovery_observation"] = feedback.to_mapping()
    return result


def _synthetic_binding(
    *,
    source_version: str,
    source_plan_sha256: str,
    source_plan_payload_sha256: str,
    accepted_steps_sha256: str,
    frame: MotionKeyframe,
    wheel_active_duration_s: float = 0.0,
    explicit_hold_s: float = 0.0,
) -> PlaybackSegmentBinding:
    sparse_targets = {
        name: float(value)
        for name, value in frame.servo_targets_deg.items()
        if abs(float(value)) > 1.0e-12
    }
    event = {
        "segment_index": frame.source_segment_index,
        "source_step": frame.source_step_index,
        "source_step_id": f"synthetic-step-{frame.source_step_index}",
        "global_command_index": frame.sequence_index + 1,
        "base_command": frame.commands[0] if frame.commands else "synthetic",
    }
    return PlaybackSegmentBinding(
        source_version=source_version,
        source_plan_sha256=source_plan_sha256,
        source_plan_payload_sha256=source_plan_payload_sha256,
        accepted_steps_sha256=accepted_steps_sha256,
        segment_payload={
            "segment_index": frame.source_segment_index,
            "source_step": frame.source_step_index,
            "source_step_id": event["source_step_id"],
            "event_start_index": 0,
            "event_count": 1,
            "servo_duration_s": 0.0,
            "servo_targets": sparse_targets,
            "wheel_active_duration_s": wheel_active_duration_s,
            "explicit_hold_s": explicit_hold_s,
            "servo_tolerance_deg": 1.0,
            "recorded_servo_residual_deg": {},
            "legacy_missing_endpoint": bool(sparse_targets),
        },
        event_payloads=(event,),
    )


def _completion_decision_mapping(
    control: dict[str, object],
    *,
    kind: str,
    sim_time_s: float,
    sim_step: int,
    wheel_stop_acknowledged: bool = False,
) -> dict[str, object]:
    executor = CompletionAwareSegmentExecutor()
    spec = SegmentCompletionSpec(
        segment_index=int(control["source_segment_index"]),
        source_step=int(control["source_step_index"]),
        source_step_id=str(control["source_step_id"]),
        servo_targets_deg={},
        servo_duration_s=0.0,
        servo_tolerance_deg=1.0,
        recorded_servo_residual_deg={},
        legacy_missing_endpoint=False,
        wheel_active_duration_s=0.0,
        explicit_hold_s=0.0,
    )
    executor.start(
        spec,
        start_elapsed_s=0.0,
        start_sim_time_s=0.0,
        start_sim_step=max(0, sim_step - 1),
    )
    payload = executor.observe(
        SegmentFeedback(
            elapsed_s=max(0.0, sim_time_s),
            sim_time_s=max(0.0, sim_time_s),
            sim_step=sim_step,
            servo_errors_deg={},
            servo_velocity_deg_s={},
        )
    ).to_mapping()
    payload.update(
        kind=kind,
        segment_done=kind == "COMPLETE",
        wheel_stop_due=kind == "WHEEL_STOP_DUE",
        wheel_stop_acknowledged=wheel_stop_acknowledged,
        failure_reason="invalid_joint_state" if kind == "FAIL" else "",
        failure_code="synthetic_failure" if kind == "FAIL" else "",
    )
    return payload


def _observation_mapping(
    *,
    classes: dict[str, str] | None = None,
    base: tuple[float, float, float] = (0.0, 0.0, 0.1),
    face: dict[str, float] | None = None,
    top_clearance: dict[str, float] | None = None,
    **updates: object,
) -> dict[str, object]:
    classes = dict(classes or {leg: "GROUND" for leg in ("FL", "FR", "RL", "RR")})
    face = dict(face or {leg: -0.3 for leg in ("FL", "FR", "RL", "RR")})
    top_clearance = dict(
        top_clearance or {leg: -0.05 for leg in ("FL", "FR", "RL", "RR")}
    )
    centers = {
        "FL": (0.2, 0.2, 0.05),
        "FR": (0.2, -0.2, 0.05),
        "RL": (-0.2, 0.2, 0.05),
        "RR": (-0.2, -0.2, 0.05),
    }
    for leg, clearance in face.items():
        if clearance > 0.0:
            x, y, z = centers[leg]
            centers[leg] = (0.5 + clearance, y, z)
    payload: dict[str, object] = {
        "robot_state_finite": True,
        "actuator_targets_applied": True,
        "base_position_m": {"x": base[0], "y": base[1], "z": base[2]},
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "base_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "servo_targets_deg": _complete_servos(),
        "wheel_targets_rad_s": _complete_wheels(),
        "wheel_center_w_m": centers,
        "wheel_contact_classes": classes,
        "wheel_contact_load_n": {leg: None for leg in ("FL", "FR", "RL", "RR")},
        "wheel_front_face_clearance_m": face,
        "wheel_top_clearance_m": top_clearance,
        "obstacle_front_face_x_m": 0.5,
        "obstacle_top_z_m": 0.05,
        "dispatch_error": "",
        "robot_fell": False,
        "body_stuck": None,
        "dangerous_collision": None,
        "severe_penetration": None,
        "joint_limit_violation": False,
        "unsafe_joint_target": False,
        "active_leg_trapped": None,
        "wheel_drive_up_without_required_lift": False,
        "body_crossed_front_face": False,
        "final_recoverable": False,
        "posture_complete": False,
    }
    payload.update(updates)
    return _attach_strict_evidence(payload)


def _with_leg(
    payload: dict[str, object], leg: str, contact_class: str, *, crossed: bool = False
) -> dict[str, object]:
    result = dict(payload)
    classes = dict(result["wheel_contact_classes"])  # type: ignore[arg-type]
    classes[leg] = contact_class
    result["wheel_contact_classes"] = classes
    face = dict(result["wheel_front_face_clearance_m"])  # type: ignore[arg-type]
    face[leg] = 0.1 if crossed else -0.1
    result["wheel_front_face_clearance_m"] = face
    top = dict(result["wheel_top_clearance_m"])  # type: ignore[arg-type]
    top[leg] = 0.01 if contact_class == "AIR" else 0.0 if contact_class == "TOP" else -0.05
    result["wheel_top_clearance_m"] = top
    centers = dict(result["wheel_center_w_m"])  # type: ignore[arg-type]
    _x, y, z = centers[leg]
    if crossed:
        x = 0.7 if leg in {"FL", "FR"} else 0.5
    else:
        x = 0.2 if leg in {"FL", "FR"} else -0.2
    centers[leg] = (x, y, z)
    result["wheel_center_w_m"] = centers
    return _attach_strict_evidence(result)


def _top_legs(payload: dict[str, object], *legs: str) -> dict[str, object]:
    result = dict(payload)
    for leg in legs:
        result = _with_leg(result, leg, "TOP", crossed=True)
    return result


def _clean_gate_a_observation(row: dict[str, object]) -> dict[str, object]:
    """Rebuild the deployment observation from one sealed minimal row."""

    centers = {
        leg: tuple(float(value) for value in values)
        for leg, values in dict(row["wheel_center_w_m"]).items()  # type: ignore[arg-type]
    }
    face = {
        leg: float(value)
        for leg, value in dict(row["wheel_front_face_clearance_m"]).items()  # type: ignore[arg-type]
    }
    top = {
        leg: float(value)
        for leg, value in dict(row["wheel_top_clearance_m"]).items()  # type: ignore[arg-type]
    }
    classes = {
        leg: str(value).upper()
        for leg, value in dict(row["wheel_contact_classes"]).items()  # type: ignore[arg-type]
    }
    front_plane = sum(
        centers[leg][0] - GATE_A_WHEEL_RADIUS_M - face[leg]
        for leg in ("FL", "FR", "RL", "RR")
    ) / 4.0
    top_plane = sum(
        centers[leg][2] - GATE_A_WHEEL_RADIUS_M - top[leg]
        for leg in ("FL", "FR", "RL", "RR")
    ) / 4.0
    base_raw = dict(row["base_position_m"])  # type: ignore[arg-type]
    base = (
        float(base_raw["x"]),
        float(base_raw["y"]),
        float(base_raw["z"]),
    )
    roll = float(row["base_roll_rad"])
    pitch = float(row["base_pitch_rad"])
    support_count = sum(value not in {"AIR", "UNKNOWN"} for value in classes.values())
    recoverable = bool(
        abs(roll) < math.radians(70.0)
        and abs(pitch) < math.radians(70.0)
        and support_count >= 2
    )
    velocity_stable = bool(row.get("final_velocity_stable", False))
    return _attach_strict_evidence({
        "robot_state_finite": bool(row["robot_state_finite"]),
        "actuator_targets_applied": True,
        "base_position_m": base,
        "base_roll_rad": roll,
        "base_pitch_rad": pitch,
        "base_angular_velocity_rad_s": row["base_angular_velocity_rad_s"],
        "servo_targets_deg": row["servo_targets_deg"],
        "wheel_targets_rad_s": row["wheel_targets_rad_s"],
        "wheel_center_w_m": centers,
        "wheel_contact_classes": classes,
        "wheel_contact_load_n": row["wheel_contact_load_n"],
        "wheel_front_face_clearance_m": face,
        "wheel_top_clearance_m": top,
        "obstacle_front_face_x_m": front_plane,
        "obstacle_top_z_m": top_plane,
        "dispatch_error": "",
        "robot_fell": bool(
            abs(roll) >= math.radians(85.0)
            or abs(pitch) >= math.radians(85.0)
        ),
        "body_stuck": None,
        "dangerous_collision": None,
        "severe_penetration": None,
        "joint_limit_violation": row.get("joint_limit_violation") is True,
        "unsafe_joint_target": False,
        "active_leg_trapped": None,
        "wheel_drive_up_without_required_lift": False,
        "body_crossed_front_face": base[0] > front_plane,
        "final_recoverable": recoverable,
        "posture_complete": bool(
            recoverable
            and velocity_stable
            and all(value == "TOP" for value in classes.values())
        ),
    })


def _observation_payload(value: MacroObservation | dict[str, object]) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {
        "robot_state_finite": value.robot_state_finite,
        "actuator_targets_applied": value.actuator_targets_applied,
        "base_position_m": value.base_position_m,
        "base_roll_rad": value.base_roll_rad,
        "base_pitch_rad": value.base_pitch_rad,
        "base_angular_velocity_rad_s": value.base_angular_velocity_rad_s,
        "com_position_m": value.com_position_m,
        "servo_targets_deg": dict(value.servo_targets_deg),
        "wheel_targets_rad_s": dict(value.wheel_targets_rad_s),
        "wheel_center_w_m": dict(value.wheel_center_w_m),
        "wheel_contact_classes": dict(value.wheel_contact_classes),
        "wheel_contact_load_n": dict(value.wheel_contact_load_n),
        "wheel_front_face_clearance_m": dict(value.wheel_front_face_clearance_m),
        "wheel_top_clearance_m": dict(value.wheel_top_clearance_m),
        "obstacle_front_face_x_m": value.obstacle_front_face_x_m,
        "obstacle_top_z_m": value.obstacle_top_z_m,
        "dispatch_error": value.dispatch_error,
        "robot_fell": value.robot_fell,
        "body_stuck": value.body_stuck,
        "dangerous_collision": value.dangerous_collision,
        "severe_penetration": value.severe_penetration,
        "joint_limit_violation": value.joint_limit_violation,
        "unsafe_joint_target": value.unsafe_joint_target,
        "active_leg_trapped": value.active_leg_trapped,
        "wheel_drive_up_without_required_lift": value.wheel_drive_up_without_required_lift,
        "body_crossed_front_face": value.body_crossed_front_face,
        "final_recoverable": value.final_recoverable,
        "posture_complete": value.posture_complete,
    }


def _current_test_observation(
    value: MacroObservation | dict[str, object],
    *,
    sim_time_s: float,
    sim_step: int,
    command_epoch: int,
    verified_command_epoch: int | None,
    readback_targets: dict[str, float],
    readback_wheel_targets: dict[str, float],
) -> MacroObservation:
    payload = _observation_payload(value)
    payload = _attach_strict_evidence(
        payload,
        physics_time_s=sim_time_s,
        sim_step=sim_step,
        observed_command_epoch=command_epoch,
        verified_command_epoch=verified_command_epoch,
        readback_targets=readback_targets,
        readback_wheel_targets=readback_wheel_targets,
    )
    return MacroObservation.from_mapping(payload)


class MacroFSMController(ProductionMacroFSMController):
    """Legacy test adapter that refreshes synthetic evidence every callback."""

    def reset(self, initial_observation, *, sim_time_s, **kwargs):  # type: ignore[no-untyped-def]
        self._test_evidence_step = 0
        initial_payload = _observation_payload(initial_observation)
        observed = _current_test_observation(
            initial_observation,
            sim_time_s=sim_time_s,
            sim_step=self._test_evidence_step,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=dict(initial_payload["servo_targets_deg"]),  # type: ignore[arg-type]
            readback_wheel_targets=dict(initial_payload["wheel_targets_rad_s"]),  # type: ignore[arg-type]
        )
        return super().reset(observed, sim_time_s=sim_time_s, **kwargs)

    def tick(self, observation, *, sim_time_s, **kwargs):  # type: ignore[no-untyped-def]
        self._test_evidence_step += 1
        pending = self._feedback_pending_epoch
        epoch = pending if pending is not None else self._command_epoch
        observed = _current_test_observation(
            observation,
            sim_time_s=sim_time_s,
            sim_step=self._test_evidence_step,
            command_epoch=epoch,
            verified_command_epoch=pending,
            readback_targets=dict(self._current_servo_targets),
            readback_wheel_targets=dict(self._current_wheel_targets),
        )
        return super().tick(observed, sim_time_s=sim_time_s, **kwargs)


def _physical_evidence_ready(
    state: object,
    evidence: dict[str, object],
    observation: MacroObservation,
) -> bool:
    guard = state.completion_guard  # type: ignore[attr-defined]
    if not bool(evidence.get("attitude_safe", False)):
        return False
    kind = guard.kind
    if kind == MacroGuardKind.INITIALIZED:
        return observation.robot_state_finite and observation.actuator_targets_applied
    if kind == MacroGuardKind.COM_SHIFT_OR_UNLOAD:
        state_air = dict(evidence["state_airborne_before_crossing"])
        unloaded = bool(guard.active_leg and state_air[guard.active_leg])
        projected = float(evidence["target_direction_projected_displacement_m"])
        inherited = 0.0
        if (
            state.state_id == MacroStateId.S5_PRE_RR_COM_SHIFT  # type: ignore[attr-defined]
            and evidence.get("boundary_from_state")
            == MacroStateId.S4_FRONT_PAIR_ADVANCE.value
            and evidence.get("boundary_to_state") == state.state_id.value  # type: ignore[attr-defined]
        ):
            inherited = float(
                evidence["boundary_inherited_projected_displacement_m"]
            )
        return unloaded or max(projected, inherited) >= guard.minimum_com_displacement_m
    if kind == MacroGuardKind.LEG_TRAVERSED:
        leg = guard.active_leg
        state_air = dict(evidence["state_airborne_before_crossing"])
        state_crossed = dict(evidence["state_crossed_seen"])
        state_top = dict(evidence["state_top_seen"])
        episode_top = dict(evidence["episode_top_seen"])
        required = all(
            bool(state_top[item] if item == leg else episode_top[item])
            for item in guard.required_top_legs
        )
        return bool(
            leg
            and state_crossed[leg]
            and state_top[leg]
            and required
            and (state_air[leg] or not guard.require_airborne_before_crossing)
        )
    if kind == MacroGuardKind.FRONT_PAIR_ADVANCED:
        fresh = bool(
            evidence.get("boundary_from_state") == MacroStateId.S3_FL_TRAVERSE.value
            and evidence.get("boundary_to_state") == state.state_id.value  # type: ignore[attr-defined]
        )
        episode_top = dict(evidence["episode_top_seen"])
        tops = fresh and all(episode_top[item] for item in guard.required_top_legs)
        inherited = (
            float(evidence["boundary_inherited_forward_progress_m"])
            if fresh
            else 0.0
        )
        rear = any(
            float(observation.wheel_front_face_clearance_m[leg]) >= -0.08
            for leg in ("RL", "RR")
        )
        return bool(
            tops
            and (
                float(evidence["forward_progress_m"]) >= guard.minimum_body_progress_m
                or inherited >= guard.minimum_body_progress_m
                or rear
            )
        )
    if kind == MacroGuardKind.SUPPORT_SETUP:
        return all(
            observation.leg_support_candidate(leg)
            for leg in guard.required_support_legs
        )
    if kind == MacroGuardKind.FINAL_ADVANCED:
        fresh = bool(
            evidence.get("boundary_from_state")
            == MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE.value
            and evidence.get("boundary_to_state") == state.state_id.value  # type: ignore[attr-defined]
        )
        inherited = (
            float(evidence["boundary_inherited_forward_progress_m"])
            if fresh
            else 0.0
        )
        episode_crossed = dict(evidence["episode_crossed_seen"])
        return bool(
            observation.body_crossed_front_face
            and (
                float(evidence["forward_progress_m"]) >= guard.minimum_body_progress_m
                or inherited >= guard.minimum_body_progress_m
                or (fresh and sum(bool(value) for value in episode_crossed.values()) >= 3)
            )
        )
    if kind == MacroGuardKind.POSTURE_RECOVERED:
        return bool(
            observation.final_recoverable
            and (observation.body_crossed_front_face if guard.require_body_crossed else True)
        )
    return False


def _test_library() -> MotionProfileLibrary:
    graph = build_default_macro_graph()
    sources = discover_successful_gate_a_sources(PROJECT_ROOT)
    profiles: list[PhaseMotionProfile] = []
    source_by_prefix = {
        source.source_version.split("_", 1)[0]: source for source in sources
    }
    for source_prefix, strategy in (
        ("v003", "PRIMARY_PROFILE"),
        ("v008", "ALTERNATE_PROFILE_2"),
        ("v009", "ALTERNATE_PROFILE_1"),
        ("v010", "ALTERNATE_PROFILE_3"),
    ):
        source = source_by_prefix[source_prefix]
        for index, state in enumerate(graph.states):
            if state.state_id in {
                MacroStateId.S0_INITIALIZE,
                MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
                MacroStateId.SUCCESS,
                MacroStateId.SAFE_STOP,
            }:
                continue
            # Alternate sources exercise reset-time selection and hot-switch rejection.
            if source_prefix != "v003" and state.state_id not in {
                MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
                MacroStateId.S5_PRE_RR_COM_SHIFT,
                MacroStateId.S10_POSTURE_RECOVERY,
            }:
                continue
            state_strategy = (
                "RECOVERY_PROFILE_2"
                if source_prefix == "v003" and state.state_id == MacroStateId.S10_POSTURE_RECOVERY
                else "RECOVERY_PROFILE_1"
                if source_prefix != "v003" and state.state_id == MacroStateId.S10_POSTURE_RECOVERY
                else strategy
            )
            wheel_value = 0.3 if state.state_id == MacroStateId.S4_FRONT_PAIR_ADVANCE else 0.0
            frame = MotionKeyframe(
                time_s=0.0,
                source_time_s=0.0,
                sequence_index=0,
                source_segment_index=index,
                source_step_index=index + 1,
                physical_phase=state.physical_phases[0],
                subphase=MacroSubphase.PRELOAD,
                servo_targets_deg=_complete_servos(float(index)),
                wheel_targets_rad_s=_complete_wheels(wheel_value),
                commands=(f"servo front_left_hip {float(index)}",),
                source_event_indices=(index,),
            )
            profiles.append(
                PhaseMotionProfile(
                    profile_id=f"{source.source_version}:{state.state_id.value}:{state_strategy}",
                    source_version=source.source_version,
                    state_id=state.state_id,
                    strategy=state_strategy,
                    physical_phases=state.physical_phases,
                    keyframes=(frame,),
                    nominal_duration_s=0.0,
                    source_plan_sha256=source.plan_sha256,
                    source_plan_file_sha256=source.plan_file_sha256,
                    gate_a_run_dir=str(source.gate_a_run_dir),
                    source_segment_range=(index, index),
                    source_step_indices=(index + 1,),
                    source_commands=frame.commands,
                    worker_request_path=str(source.worker_request_path),
                    worker_request_sha256=source.worker_request_sha256,
                    accepted_steps_path=str(source.accepted_steps_path),
                    accepted_steps_sha256=source.accepted_steps_sha256,
                    source_plan_payload_sha256=source.full_plan_payload_sha256,
                    segment_bindings=(
                        _synthetic_binding(
                            source_version=source.source_version,
                            source_plan_sha256=source.plan_sha256,
                            source_plan_payload_sha256=source.full_plan_payload_sha256,
                            accepted_steps_sha256=source.accepted_steps_sha256,
                            frame=frame,
                        ),
                    ),
                )
            )
    return MotionProfileLibrary(
        profiles=tuple(profiles),
        successful_sources=sources,
        alignment_path="synthetic-short-profiles-derived-from-real-success-identities",
    )


def _hermetic_feedback_library() -> MotionProfileLibrary:
    """Minimal source/profile identity for feedback-only contract tests.

    This intentionally does not stand in for Gate-A admission.  Real recording
    tests continue to fail closed when their plan fingerprints are stale.
    """

    frame = MotionKeyframe(
        time_s=0.0,
        source_time_s=0.0,
        sequence_index=0,
        source_segment_index=0,
        source_step_index=1,
        physical_phase="FINAL_POSTURE_RECOVERY",
        subphase=MacroSubphase.RECOVERY,
        servo_targets_deg=_complete_servos(),
        wheel_targets_rad_s=_complete_wheels(),
        commands=("servo front_right_hip 0.0",),
        source_event_indices=(0,),
    )
    source_plan_sha256 = "1" * 64
    source_plan_payload_sha256 = "2" * 64
    accepted_steps_sha256 = "3" * 64
    binding = _synthetic_binding(
        source_version=DEFAULT_PRIMARY_VERSION,
        source_plan_sha256=source_plan_sha256,
        source_plan_payload_sha256=source_plan_payload_sha256,
        accepted_steps_sha256=accepted_steps_sha256,
        frame=frame,
    )
    profile = PhaseMotionProfile(
        profile_id="hermetic-feedback-configuration:S10",
        source_version=DEFAULT_PRIMARY_VERSION,
        state_id=MacroStateId.S10_POSTURE_RECOVERY,
        strategy="RECOVERY_PROFILE",
        physical_phases=("FINAL_POSTURE_RECOVERY",),
        keyframes=(frame,),
        nominal_duration_s=0.0,
        source_plan_sha256=source_plan_sha256,
        source_plan_file_sha256="4" * 64,
        gate_a_run_dir="HERMETIC_FEEDBACK_CONTRACT_ONLY",
        source_segment_range=(0, 0),
        source_step_indices=(1,),
        source_commands=frame.commands,
        accepted_steps_sha256=accepted_steps_sha256,
        source_plan_payload_sha256=source_plan_payload_sha256,
        segment_bindings=(binding,),
    )
    return MotionProfileLibrary(
        profiles=(profile,),
        successful_sources=(
            SimpleNamespace(source_version=DEFAULT_PRIMARY_VERSION),
        ),
        alignment_path="hermetic-feedback-configuration-contract-only",
    )


class MacroObservationContractTests(unittest.TestCase):
    def test_boolean_string_is_rejected_instead_of_becoming_truthy(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit bool"):
            MacroObservation.from_mapping(
                _observation_mapping(robot_state_finite="false")
            )

    def test_rejects_missing_wrong_shape_and_nonfinite_core_state(self) -> None:
        cases = [
            {},
            _observation_mapping(base_position_m=[0.0, 0.0]),
            _observation_mapping(base_angular_velocity_rad_s=[0.0, math.nan, 0.0]),
            _observation_mapping(robot_state_finite="not-a-boolean"),
        ]
        for payload in cases:
            with self.subTest(payload=list(payload)):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(payload)

    def test_every_hard_safety_and_outcome_field_is_required_and_type_strict(self) -> None:
        required_bool_fields = (
            "robot_state_finite",
            "actuator_targets_applied",
            "robot_fell",
            "joint_limit_violation",
            "unsafe_joint_target",
            "wheel_drive_up_without_required_lift",
            "body_crossed_front_face",
            "final_recoverable",
            "posture_complete",
        )
        for field_name in required_bool_fields:
            missing = _observation_mapping()
            del missing[field_name]
            with self.subTest(field=field_name, failure="missing"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(missing)
            wrong_type = _observation_mapping()
            wrong_type[field_name] = 0
            with self.subTest(field=field_name, failure="wrong_type"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(wrong_type)

        for field_name in (
            "body_stuck",
            "dangerous_collision",
            "severe_penetration",
            "active_leg_trapped",
        ):
            missing = _observation_mapping()
            del missing[field_name]
            with self.subTest(field=field_name, failure="missing"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(missing)
            invalid = _observation_mapping()
            invalid[field_name] = "unknown"
            with self.subTest(field=field_name, failure="wrong_type"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(invalid)

        missing_dispatch = _observation_mapping()
        del missing_dispatch["dispatch_error"]
        with self.assertRaises(ValueError):
            MacroObservation.from_mapping(missing_dispatch)
        invalid_dispatch = _observation_mapping(dispatch_error=None)
        with self.assertRaises(ValueError):
            MacroObservation.from_mapping(invalid_dispatch)

    def test_unproduced_safety_fields_accept_and_preserve_exact_tristate(self) -> None:
        for field_name in ("body_stuck", "active_leg_trapped"):
            for value in (None, False, True):
                with self.subTest(field=field_name, value=value, path="mapping"):
                    observation = MacroObservation.from_mapping(
                        _observation_mapping(**{field_name: value})
                    )
                    self.assertIs(getattr(observation, field_name), value)
                    direct = MacroObservation(
                        **{**observation.__dict__, field_name: value}
                    )
                    self.assertIs(getattr(direct, field_name), value)

    def test_rejects_incomplete_actuator_and_wheel_geometry_maps(self) -> None:
        map_fields = (
            "servo_targets_deg",
            "wheel_targets_rad_s",
            "wheel_center_w_m",
            "wheel_contact_classes",
            "wheel_contact_load_n",
            "wheel_front_face_clearance_m",
            "wheel_top_clearance_m",
        )
        for field_name in map_fields:
            payload = _observation_mapping()
            incomplete = dict(payload[field_name])  # type: ignore[arg-type]
            del incomplete[next(iter(incomplete))]
            payload[field_name] = incomplete
            with self.subTest(field=field_name, failure="missing"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(payload)

            payload = _observation_mapping()
            with_unknown = dict(payload[field_name])  # type: ignore[arg-type]
            with_unknown["UNKNOWN_KEY"] = 0.0
            payload[field_name] = with_unknown
            with self.subTest(field=field_name, failure="unknown"):
                with self.assertRaises(ValueError):
                    MacroObservation.from_mapping(payload)

    def test_direct_constructor_rejects_noncanonical_maps_vectors_and_booleans(self) -> None:
        valid = MacroObservation.from_mapping(_observation_mapping())
        base = dict(valid.__dict__)
        cases: list[dict[str, object]] = []
        for field_name in (
            "servo_targets_deg",
            "wheel_targets_rad_s",
            "wheel_center_w_m",
            "wheel_contact_classes",
            "wheel_contact_load_n",
            "wheel_front_face_clearance_m",
            "wheel_top_clearance_m",
        ):
            incomplete = dict(base[field_name])  # type: ignore[arg-type]
            del incomplete[next(iter(incomplete))]
            cases.append({field_name: incomplete})
            with_unknown = dict(base[field_name])  # type: ignore[arg-type]
            with_unknown["UNKNOWN_KEY"] = 0.0
            cases.append({field_name: with_unknown})
        cases.extend(
            (
                {"base_position_m": (0.0, 0.0)},
                {"base_angular_velocity_rad_s": (0.0, math.nan, 0.0)},
                {
                    "servo_targets_deg": {
                        **dict(base["servo_targets_deg"]),  # type: ignore[arg-type]
                        SERVO_JOINT_NAMES[0]: math.nan,
                    }
                },
                {"robot_state_finite": "false"},
                {"actuator_targets_applied": 1},
            )
        )
        for update in cases:
            with self.subTest(update=tuple(update)):
                with self.assertRaises(ValueError):
                    MacroObservation(**{**base, **update})

    def test_direct_constructor_requires_every_safety_and_terminal_field(self) -> None:
        valid = MacroObservation.from_mapping(_observation_mapping())
        base = dict(valid.__dict__)
        required_bool_fields = (
            "robot_state_finite",
            "actuator_targets_applied",
            "robot_fell",
            "joint_limit_violation",
            "unsafe_joint_target",
            "wheel_drive_up_without_required_lift",
            "body_crossed_front_face",
            "final_recoverable",
            "posture_complete",
        )
        for field_name in required_bool_fields:
            missing = dict(base)
            del missing[field_name]
            with self.subTest(field=field_name, failure="missing"):
                with self.assertRaises(TypeError):
                    MacroObservation(**missing)
            with self.subTest(field=field_name, failure="wrong_type"):
                with self.assertRaises(ValueError):
                    MacroObservation(**{**base, field_name: 0})
        for field_name in (
            "body_stuck",
            "dangerous_collision",
            "severe_penetration",
            "active_leg_trapped",
        ):
            missing = dict(base)
            del missing[field_name]
            with self.subTest(field=field_name, failure="missing"):
                with self.assertRaises(TypeError):
                    MacroObservation(**missing)
            with self.subTest(field=field_name, failure="wrong_type"):
                with self.assertRaises(ValueError):
                    MacroObservation(**{**base, field_name: "unknown"})
        missing_dispatch = dict(base)
        del missing_dispatch["dispatch_error"]
        with self.assertRaises(TypeError):
            MacroObservation(**missing_dispatch)
        with self.assertRaises(ValueError):
            MacroObservation(**{**base, "dispatch_error": None})

    def test_nonfinite_target_or_geometry_value_fails_closed(self) -> None:
        cases = []
        for field_name, key in (
            ("servo_targets_deg", SERVO_JOINT_NAMES[0]),
            ("wheel_targets_rad_s", WHEEL_JOINT_NAMES[0]),
            ("wheel_front_face_clearance_m", "FR"),
            ("wheel_top_clearance_m", "FR"),
        ):
            payload = _observation_mapping()
            values = dict(payload[field_name])  # type: ignore[arg-type]
            values[key] = math.nan
            payload[field_name] = values
            cases.append(payload)
        for payload in cases:
            with self.assertRaises(ValueError):
                MacroObservation.from_mapping(payload)

    def test_top_behind_front_plane_is_not_crossing_or_top_completion(self) -> None:
        payload = _with_leg(_observation_mapping(), "FR", "TOP", crossed=False)
        observation = MacroObservation.from_mapping(payload)
        self.assertFalse(observation.leg_crossed("FR"))
        self.assertFalse(observation.leg_top("FR"))

    def test_strict_feedback_schema_is_exact_sha_bound_and_deeply_immutable(self) -> None:
        observation = MacroObservation.from_mapping(_observation_mapping())
        feedback = observation.feedback_recovery_observation
        mapping = feedback.to_mapping()
        self.assertEqual(
            set(mapping["payload"]),
            FEEDBACK_RECOVERY_OBSERVATION_PAYLOAD_KEYS,
        )
        with self.assertRaises(TypeError):
            feedback.readback_servo_targets_deg[SERVO_JOINT_NAMES[0]] = 99.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            feedback.wheel_center_w_m["FR"] = (9.0, 9.0, 9.0)  # type: ignore[index]

        tampered = feedback.to_mapping()
        tampered["payload"]["measured_servo_positions_deg"][  # type: ignore[index]
            SERVO_JOINT_NAMES[0]
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "payload SHA"):
            FeedbackRecoveryObservation.from_mapping(tampered)

        extra = feedback.to_mapping()
        extra["payload"]["unknown"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "payload keys"):
            FeedbackRecoveryObservation.from_mapping(extra)

    def test_tampered_centroid_and_cross_tick_feedback_are_rejected(self) -> None:
        payload = _observation_mapping()
        centroidal = json.loads(
            json.dumps(payload["centroidal_support_evidence"])
        )
        centroidal["payload"]["whole_body_com"]["position_w_m"][0] = 0.1
        payload["centroidal_support_evidence"] = centroidal
        with self.assertRaisesRegex(ValueError, "payload SHA"):
            MacroObservation.from_mapping(payload)

        payload = _observation_mapping()
        feedback = json.loads(
            json.dumps(payload["feedback_recovery_observation"])
        )
        feedback["payload"]["sim_step"] = 1
        feedback["payload_sha256"] = hashlib.sha256(
            json.dumps(
                feedback["payload"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload["feedback_recovery_observation"] = feedback
        with self.assertRaisesRegex(ValueError, "same tick"):
            MacroObservation.from_mapping(payload)

    def test_sha_bound_feedback_duplicates_must_match_macro_observation(self) -> None:
        payload = _observation_mapping()
        centers = dict(payload["wheel_center_w_m"])  # type: ignore[arg-type]
        centers["FR"] = (9.0, -0.2, 0.05)
        payload["wheel_center_w_m"] = centers
        with self.assertRaisesRegex(ValueError, "SHA-bound feedback fields"):
            MacroObservation.from_mapping(payload)


class _CompletionDrivingController(MacroFSMController):
    """Test harness that supplies exact measured-completion tokens on demand."""

    def reset(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self._test_step = 0
        self._test_start_step = 0
        self._test_start_control: dict[str, object] | None = None
        self._test_wheel_stop_acked = False
        return super().reset(*args, **kwargs)

    def tick(self, observation: object, *, sim_time_s: float, **kwargs: object):  # type: ignore[no-untyped-def]
        self._test_step += 1
        kwargs.setdefault("source_cursor_permit", True)
        supplied = kwargs.get("segment_completion_token")
        generated_kind = ""
        if supplied is None and self._test_start_control is not None:
            next_is_source_stop = False
            if (
                self._active_profile is not None
                and self._visible_cursor < len(self._visible_keyframe_indices)
                and not self._test_wheel_stop_acked
            ):
                frame = self._active_profile.keyframes[
                    self._visible_keyframe_indices[self._visible_cursor]
                ]
                next_is_source_stop = bool(
                    frame.source_segment_index
                    == self._test_start_control["source_segment_index"]
                    and frame.dispatch_kind == "wheel_channel_completion_stop"
                )
            generated_kind = (
                "WHEEL_STOP_DUE" if next_is_source_stop else "COMPLETE"
            )
            decision = _completion_decision_mapping(
                self._test_start_control,
                kind=generated_kind,
                sim_time_s=sim_time_s,
                sim_step=max(self._test_step, self._test_start_step + 1),
                wheel_stop_acknowledged=self._test_wheel_stop_acked,
            )
            supplied = MacroSegmentCompletionToken.from_control_decision(
                self._test_start_control,
                decision,
                start_sim_step=self._test_start_step,
                start_readback_sha256="a" * 64,
            )
            kwargs["segment_completion_token"] = supplied
        result = super().tick(observation, sim_time_s=sim_time_s, **kwargs)
        control = dict(result.segment_completion_control)
        if control["kind"] == "START":
            self._test_start_control = control
            self._test_start_step = self._test_step
            self._test_wheel_stop_acked = False
        elif control["kind"] == "WHEEL_STOP":
            self._test_wheel_stop_acked = True
        elif generated_kind == "COMPLETE":
            self._test_start_control = None
            self._test_wheel_stop_acked = False
        return result


class MacroControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_default_macro_graph()
        cls.library = _test_library()

    def _controller(self) -> MacroFSMController:
        return _CompletionDrivingController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )

    def _production_s10(
        self, payload: dict[str, object]
    ) -> tuple[ProductionMacroFSMController, MacroObservation]:
        initial = _current_test_observation(
            payload,
            sim_time_s=0.0,
            sim_step=0,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=dict(payload["servo_targets_deg"]),  # type: ignore[arg-type]
            readback_wheel_targets=dict(payload["wheel_targets_rad_s"]),  # type: ignore[arg-type]
        )
        controller = ProductionMacroFSMController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        controller.reset(initial, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S10_POSTURE_RECOVERY,
            initial,
            0.0,
        )
        controller._active_profile = None
        return controller, initial

    @staticmethod
    def _s10_air_candidate() -> dict[str, object]:
        payload = _top_legs(_observation_mapping(), "FL", "RR", "RL")
        payload = _with_leg(payload, "FR", "AIR", crossed=True)
        payload.update(
            body_crossed_front_face=True,
            final_recoverable=True,
            posture_complete=False,
        )
        return _attach_strict_evidence(payload)

    @staticmethod
    def _worker_reconstructed_configuration_sha256(
        controller: ProductionMacroFSMController,
        observation: MacroObservation,
        *,
        leg: str,
        reference_targets: dict[str, float],
    ) -> str:
        feedback = observation.feedback_recovery_observation
        profile = controller._active_profile
        payload = {
            "schema_version": FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA,
            "leg": leg,
            "macro_state": controller._state_id.value,
            "selected_source_version": controller._source_version,
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
            "servo_reference_targets_deg": reference_targets,
            "measured_servo_positions_deg": dict(
                feedback.measured_servo_positions_deg
            ),
            "wheel_center_w_m": {
                item: list(feedback.wheel_center_w_m[item])
                for item in ("FL", "FR", "RL", "RR")
            },
            "body_crossed_front_face": feedback.body_crossed_front_face,
            "final_recoverable": feedback.final_recoverable,
            "posture_complete": feedback.posture_complete,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _start_raw_s1(
        self,
    ) -> tuple[MacroFSMController, dict[str, object], dict[str, object]]:
        controller = MacroFSMController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        entered = controller.tick(
            ground, sim_time_s=0.01, source_cursor_permit=False
        )
        self.assertEqual(
            entered.macro_state, MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT
        )
        started = controller.tick(
            ground, sim_time_s=0.02, source_cursor_permit=True
        )
        self.assertEqual(started.segment_completion_control["kind"], "START")
        return controller, ground, dict(started.segment_completion_control)

    def _completion_token(
        self,
        control: dict[str, object],
        *,
        kind: str,
        sim_time_s: float,
        sim_step: int,
        start_sim_step: int = 10,
        start_readback_sha256: str = "c" * 64,
        wheel_stop_acknowledged: bool = False,
    ) -> MacroSegmentCompletionToken:
        return MacroSegmentCompletionToken.from_control_decision(
            control,
            _completion_decision_mapping(
                control,
                kind=kind,
                sim_time_s=sim_time_s,
                sim_step=sim_step,
                wheel_stop_acknowledged=wheel_stop_acknowledged,
            ),
            start_sim_step=start_sim_step,
            start_readback_sha256=start_readback_sha256,
        )

    def _v010_controller(self, observation: dict[str, object]) -> MacroFSMController:
        library = build_profile_library(PROJECT_ROOT)
        source_version = next(
            source.source_version
            for source in library.successful_sources
            if source.source_version.startswith("v010_")
        )
        controller = MacroFSMController(
            self.graph,
            library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        controller.reset(
            observation,
            sim_time_s=0.0,
            source_version=source_version,
        )
        return controller

    def _prime_real_final_segment(
        self,
        controller: MacroFSMController,
        state_id: MacroStateId,
        observation: dict[str, object],
        *,
        sim_time_s: float,
    ) -> object:
        controller._enter_state(
            state_id,
            MacroObservation.from_mapping(observation),
            sim_time_s,
        )
        profile = controller._active_profile
        self.assertIsNotNone(profile)
        assert profile is not None
        last_segment = max(binding.segment_index for binding in profile.segment_bindings)
        positions = [
            position
            for position, frame_index in enumerate(controller._visible_keyframe_indices)
            if profile.keyframes[frame_index].source_segment_index == last_segment
            and profile.keyframes[frame_index].dispatch_kind == "segment_start"
        ]
        self.assertEqual(len(positions), 1)
        position = positions[0]
        self.assertGreater(position, 0)
        previous = profile.keyframes[controller._visible_keyframe_indices[position - 1]]
        controller._visible_cursor = position
        controller._completed_segment_indices = {
            binding.segment_index
            for binding in profile.segment_bindings
            if binding.segment_index != last_segment
        }
        controller._current_servo_targets = dict(previous.servo_targets_deg)
        controller._current_wheel_targets = dict(previous.wheel_targets_rad_s)
        controller._last_decision_servo_targets = dict(previous.servo_targets_deg)
        controller._last_decision_wheel_targets = dict(previous.wheel_targets_rad_s)
        controller._last_decision_command_epoch = controller._command_epoch
        controller._last_decision_consumed_source_action_count = len(
            controller._consumed_source_actions
        )
        started = controller.tick(
            observation,
            sim_time_s=sim_time_s + 0.01,
            source_cursor_permit=True,
        )
        self.assertTrue(started.source_action_consumed)
        self.assertEqual(started.command_provenance["source_segment_index"], last_segment)
        self.assertEqual(started.segment_completion_control["kind"], "START")
        self.assertEqual(
            controller._visible_cursor,
            len(controller._visible_keyframe_indices),
        )
        return started

    def test_bundle_sha_is_stable_across_independent_full_plan_rebuilds(self) -> None:
        first = build_gate_c_bundle(PROJECT_ROOT)
        second = build_gate_c_bundle(PROJECT_ROOT)
        self.assertEqual(first.profile_library_sha256, second.profile_library_sha256)
        self.assertEqual(first.bundle_sha256, second.bundle_sha256)

    def _two_action_profile(
        self,
        *,
        duplicate_coordinate: bool = False,
        final_wheel_value: float = 0.0,
    ) -> PhaseMotionProfile:
        base = self.library.get(
            DEFAULT_PRIMARY_VERSION,
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            strategy="PRIMARY_PROFILE",
        )
        template = base.keyframes[0]
        first = replace(
            template,
            time_s=0.0,
            source_time_s=100.0,
            sequence_index=100,
            source_segment_index=1000,
            source_step_index=1001,
            servo_targets_deg=_complete_servos(12.5),
            wheel_targets_rad_s=_complete_wheels(),
            commands=("servo front_left_hip 12.5",),
            source_event_indices=(2000,),
            dispatch_kind="segment_start",
        )
        second = replace(
            first,
            time_s=0.01,
            source_time_s=100.01,
            sequence_index=101,
            source_segment_index=(1000 if duplicate_coordinate else 1001),
            source_step_index=1002,
            commands=(
                "servo front_left_hip 12.5 duplicate-payload"
                if duplicate_coordinate
                else "servo front_left_hip 12.5"
            ,),
            source_event_indices=(2001,),
            wheel_targets_rad_s=_complete_wheels(final_wheel_value),
        )
        template_binding = base.segment_bindings[0]
        bindings = tuple(
            _synthetic_binding(
                source_version=base.source_version,
                source_plan_sha256=base.source_plan_sha256,
                source_plan_payload_sha256=template_binding.source_plan_payload_sha256,
                accepted_steps_sha256=template_binding.accepted_steps_sha256,
                frame=frame,
            )
            for frame in (first, second)
        )
        return replace(
            base,
            profile_id=(
                "synthetic:duplicate-coordinate"
                if duplicate_coordinate
                else (
                    "synthetic:two-identical-target-maps"
                    if final_wheel_value == 0.0
                    else "synthetic:moving-final-boundary"
                )
            ),
            keyframes=(first, second),
            nominal_duration_s=0.01,
            source_segment_range=(1000, 1001),
            source_step_indices=(1001, 1002),
            source_commands=first.commands + second.commands,
            segment_bindings=bindings,
        )

    def test_same_target_source_action_consumes_without_dispatch_and_clears_provenance(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        reset = controller.reset(ground, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            MacroObservation.from_mapping(ground),
            0.0,
        )
        profile = self._two_action_profile()
        controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=profile,
        )
        controller._profile_started_at_s = 0.0

        first = controller.tick(ground, sim_time_s=0.001)
        self.assertTrue(first.source_action_consumed)
        self.assertTrue(first.target_changed)
        self.assertTrue(first.command_changed)
        self.assertEqual(first.command_epoch, reset.command_epoch + 1)
        self.assertEqual(first.command_provenance["kind"], "SOURCE_ACTION")

        second = controller.tick(ground, sim_time_s=0.02)
        self.assertTrue(second.source_action_consumed)
        self.assertFalse(second.target_changed)
        self.assertFalse(second.command_changed)
        self.assertEqual(second.command_epoch, first.command_epoch)
        self.assertEqual(second.command_provenance["kind"], "SOURCE_ACTION")
        self.assertEqual(second.segment_completion_control["kind"], "START")
        self.assertTrue(second.segment_completion_control["source_action"])
        self.assertEqual(
            second.segment_completion_control["start_command_epoch"],
            second.command_epoch,
        )
        self.assertNotEqual(
            second.command_provenance["source_action_identity"],
            first.command_provenance["source_action_identity"],
        )
        self.assertEqual(second.transition_events, ())
        self.assertEqual(controller.status["consumed_source_action_count"], 2)

        held = controller.tick(ground, sim_time_s=0.03)
        self.assertFalse(held.source_action_consumed)
        self.assertFalse(held.target_changed)
        self.assertFalse(held.command_changed)
        self.assertEqual(held.command_epoch, second.command_epoch)
        self.assertEqual(held.command_provenance["kind"], "NONE")
        self.assertEqual(held.command_provenance["source_action_identity"], "")
        self.assertIn("HOLD:S1_APPROACH_AND_PRE_FR_SHIFT", held.transition_events)

        first_mapping = second.to_mapping()
        second_mapping = second.to_mapping()
        self.assertEqual(first_mapping, second_mapping)
        self.assertIsInstance(
            first_mapping["command_provenance"]["source_event_indices"], list
        )
        self.assertIsInstance(first_mapping["command_provenance"]["commands"], list)
        first_mapping["command_provenance"]["commands"].append("tamper")
        self.assertNotEqual(first_mapping, second.to_mapping())

    def test_v010_final_complete_coalesces_s3_s4_and_s4_s5_first_actions(self) -> None:
        base = _observation_mapping(base=(-0.01, 0.01, 0.1))
        front = _top_legs(base, "FR", "FL")

        s3_controller = self._v010_controller(front)
        s3_started = self._prime_real_final_segment(
            s3_controller,
            MacroStateId.S3_FL_TRAVERSE,
            front,
            sim_time_s=1.0,
        )
        s3_controller._state_airborne_before_crossing["FL"] = True
        s3_count = int(s3_controller.status["consumed_source_action_count"])
        s3_epoch = s3_started.command_epoch
        s3_complete = self._completion_token(
            dict(s3_started.segment_completion_control),
            kind="COMPLETE",
            sim_time_s=1.02,
            sim_step=102,
            start_sim_step=101,
        )
        entered_s4 = s3_controller.tick(
            front,
            sim_time_s=1.02,
            segment_completion_token=s3_complete,
            source_cursor_permit=True,
        )
        self.assertEqual(
            entered_s4.transition_events,
            ("EXIT:S3_FL_TRAVERSE", "ENTER:S4_FRONT_PAIR_ADVANCE"),
        )
        self.assertTrue(entered_s4.source_action_consumed)
        self.assertTrue(entered_s4.target_changed)
        self.assertEqual(entered_s4.command_epoch, s3_epoch + 1)
        self.assertEqual(entered_s4.command_provenance["kind"], "SOURCE_ACTION")
        self.assertEqual(entered_s4.command_provenance["source_segment_index"], 42)
        self.assertEqual(entered_s4.segment_completion_control["kind"], "START")
        self.assertEqual(
            entered_s4.segment_completion_control["owner_state"],
            MacroStateId.S4_FRONT_PAIR_ADVANCE.value,
        )
        self.assertEqual(
            s3_controller.status["consumed_source_action_count"], s3_count + 1
        )

        approach = dict(front)
        face = dict(approach["wheel_front_face_clearance_m"])  # type: ignore[arg-type]
        face["RR"] = -0.05
        approach["wheel_front_face_clearance_m"] = face
        approach = _attach_strict_evidence(approach)
        s4_controller = self._v010_controller(approach)
        s4_controller._episode_top_seen["FR"] = True
        s4_controller._episode_top_seen["FL"] = True
        s4_controller._record_boundary_carry(
            MacroStateId.S3_FL_TRAVERSE,
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            MacroObservation.from_mapping(approach),
        )
        s4_started = self._prime_real_final_segment(
            s4_controller,
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            approach,
            sim_time_s=2.0,
        )
        s4_count = int(s4_controller.status["consumed_source_action_count"])
        s4_epoch = s4_started.command_epoch
        s4_complete = self._completion_token(
            dict(s4_started.segment_completion_control),
            kind="COMPLETE",
            sim_time_s=2.02,
            sim_step=202,
            start_sim_step=201,
        )
        entered_s5 = s4_controller.tick(
            approach,
            sim_time_s=2.02,
            segment_completion_token=s4_complete,
            source_cursor_permit=True,
        )
        self.assertEqual(
            entered_s5.transition_events,
            ("EXIT:S4_FRONT_PAIR_ADVANCE", "ENTER:S5_PRE_RR_COM_SHIFT"),
        )
        self.assertTrue(entered_s5.source_action_consumed)
        self.assertTrue(entered_s5.target_changed)
        self.assertEqual(entered_s5.command_epoch, s4_epoch + 1)
        self.assertEqual(entered_s5.command_provenance["kind"], "SOURCE_ACTION")
        self.assertEqual(entered_s5.command_provenance["source_segment_index"], 44)
        self.assertEqual(entered_s5.segment_completion_control["kind"], "START")
        self.assertEqual(
            entered_s5.segment_completion_control["owner_state"],
            MacroStateId.S5_PRE_RR_COM_SHIFT.value,
        )
        self.assertEqual(
            s4_controller.status["consumed_source_action_count"], s4_count + 1
        )

    def test_coalesced_next_source_same_target_consumes_without_batch(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        reset = controller.reset(ground, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            MacroObservation.from_mapping(ground),
            0.0,
        )
        profile = self._two_action_profile()
        next_profile = self.library.get(
            DEFAULT_PRIMARY_VERSION,
            MacroStateId.S2_FR_TRAVERSE,
            strategy="PRIMARY_PROFILE",
        )
        next_frame = next_profile.keyframes[0]
        final_frame = replace(
            profile.keyframes[1],
            servo_targets_deg=dict(next_frame.servo_targets_deg),
            wheel_targets_rad_s=dict(next_frame.wheel_targets_rad_s),
        )
        binding_template = profile.segment_bindings[1]
        profile = replace(
            profile,
            profile_id="synthetic:next-state-same-target",
            keyframes=(profile.keyframes[0], final_frame),
            segment_bindings=(
                profile.segment_bindings[0],
                _synthetic_binding(
                    source_version=profile.source_version,
                    source_plan_sha256=profile.source_plan_sha256,
                    source_plan_payload_sha256=binding_template.source_plan_payload_sha256,
                    accepted_steps_sha256=binding_template.accepted_steps_sha256,
                    frame=final_frame,
                ),
            ),
        )
        controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=profile,
        )
        controller._profile_started_at_s = 0.0
        controller.tick(ground, sim_time_s=0.01)
        fr_air = _with_leg(ground, "FR", "AIR", crossed=False)
        final_started = controller.tick(fr_air, sim_time_s=0.02)
        self.assertTrue(final_started.target_changed)
        coalesced = controller.tick(fr_air, sim_time_s=0.03)
        self.assertEqual(
            coalesced.transition_events,
            ("EXIT:S1_APPROACH_AND_PRE_FR_SHIFT", "ENTER:S2_FR_TRAVERSE"),
        )
        self.assertTrue(coalesced.source_action_consumed)
        self.assertFalse(coalesced.target_changed)
        self.assertFalse(coalesced.command_changed)
        self.assertEqual(coalesced.command_epoch, final_started.command_epoch)
        self.assertEqual(coalesced.command_provenance["kind"], "SOURCE_ACTION")
        self.assertEqual(
            coalesced.command_provenance["source_segment_index"],
            next_frame.source_segment_index,
        )
        self.assertEqual(coalesced.segment_completion_control["kind"], "START")
        self.assertEqual(
            coalesced.segment_completion_control["owner_state"],
            MacroStateId.S2_FR_TRAVERSE.value,
        )
        self.assertEqual(reset.command_epoch + 2, coalesced.command_epoch)

    def test_changed_boundary_defers_next_state_source_action(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            MacroObservation.from_mapping(ground),
            0.0,
        )
        controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=self._two_action_profile(final_wheel_value=0.3),
        )
        controller._profile_started_at_s = 0.0
        controller.tick(ground, sim_time_s=0.01)
        fr_air = _with_leg(ground, "FR", "AIR", crossed=False)
        final_started = controller.tick(fr_air, sim_time_s=0.02)
        self.assertTrue(any(final_started.wheel_targets_rad_s.values()))
        boundary = controller.tick(fr_air, sim_time_s=0.03)
        self.assertEqual(
            boundary.transition_events,
            ("EXIT:S1_APPROACH_AND_PRE_FR_SHIFT", "ENTER:S2_FR_TRAVERSE"),
        )
        self.assertFalse(boundary.source_action_consumed)
        self.assertTrue(boundary.target_changed)
        self.assertEqual(boundary.command_provenance["kind"], "BOUNDARY_ZERO_WHEELS")
        self.assertEqual(boundary.segment_completion_control["kind"], "NONE")
        self.assertTrue(all(value == 0.0 for value in boundary.wheel_targets_rad_s.values()))
        next_started = controller.tick(fr_air, sim_time_s=0.04)
        self.assertTrue(next_started.source_action_consumed)
        self.assertEqual(next_started.command_provenance["kind"], "SOURCE_ACTION")

    def test_prior_complete_and_feedback_only_transitions_do_not_coalesce(self) -> None:
        base = _observation_mapping(base=(-0.01, 0.01, 0.1))
        fr_top = _top_legs(base, "FR")
        fl_air = _with_leg(fr_top, "FL", "AIR", crossed=False)
        controller = self._v010_controller(fl_air)
        started = self._prime_real_final_segment(
            controller,
            MacroStateId.S3_FL_TRAVERSE,
            fl_air,
            sim_time_s=3.0,
        )
        complete = self._completion_token(
            dict(started.segment_completion_control),
            kind="COMPLETE",
            sim_time_s=3.02,
            sim_step=302,
            start_sim_step=301,
        )
        held = controller.tick(
            fl_air,
            sim_time_s=3.02,
            segment_completion_token=complete,
            source_cursor_permit=True,
        )
        self.assertIn("HOLD:S3_FL_TRAVERSE", held.transition_events)
        self.assertFalse(held.source_action_consumed)
        front_top = _top_legs(fr_top, "FR", "FL")
        transitioned = controller.tick(
            front_top,
            sim_time_s=3.03,
            source_cursor_permit=True,
        )
        self.assertEqual(
            transitioned.transition_events,
            ("EXIT:S3_FL_TRAVERSE", "ENTER:S4_FRONT_PAIR_ADVANCE"),
        )
        self.assertFalse(transitioned.source_action_consumed)
        self.assertEqual(transitioned.segment_completion_control["kind"], "NONE")
        next_started = controller.tick(
            front_top,
            sim_time_s=3.04,
            source_cursor_permit=True,
        )
        self.assertTrue(next_started.source_action_consumed)

        support = _top_legs(base, "FR", "FL", "RR")
        feedback = self._controller()
        feedback.reset(support, sim_time_s=0.0)
        feedback._enter_state(
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP,
            MacroObservation.from_mapping(support),
            0.0,
        )
        feedback_transition = feedback.tick(support, sim_time_s=0.01)
        self.assertEqual(
            feedback_transition.transition_events,
            ("EXIT:S7_PRE_RL_SUPPORT_SETUP", "ENTER:S8_RL_COM_SHIFT_AND_TRAVERSE"),
        )
        self.assertFalse(feedback_transition.source_action_consumed)
        self.assertEqual(feedback_transition.segment_completion_control["kind"], "NONE")
        feedback_start = feedback.tick(support, sim_time_s=0.02)
        self.assertTrue(feedback_start.source_action_consumed)
        self.assertEqual(
            feedback_start.segment_completion_control["owner_state"],
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE.value,
        )

    def test_cursor_requires_explicit_outer_cycle_permit_and_hard_safety_stays_live(self) -> None:
        controller = MacroFSMController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller.tick(ground, sim_time_s=0.001, source_cursor_permit=False)
        for substep in range(1, 8):
            decision = controller.tick(
                ground,
                sim_time_s=0.001 + substep / 120.0,
            )
            self.assertFalse(decision.source_action_consumed)
            self.assertEqual(controller.status["consumed_source_action_count"], 0)
        boundary = controller.tick(
            ground,
            sim_time_s=0.001 + 8.0 / 120.0,
            source_cursor_permit=True,
        )
        self.assertTrue(boundary.source_action_consumed)
        self.assertEqual(boundary.segment_completion_control["kind"], "START")

        safety = MacroFSMController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        safety.reset(ground, sim_time_s=0.0)
        stopped = safety.tick(
            _observation_mapping(robot_fell=True),
            sim_time_s=1.0 / 120.0,
            source_cursor_permit=False,
        )
        self.assertEqual(stopped.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertIn("robot fall", stopped.reason)

    def test_completion_token_schema_semantics_and_start_step_are_fail_closed(self) -> None:
        _, _, control = self._start_raw_s1()
        token = self._completion_token(
            control, kind="WAIT", sim_time_s=0.03, sim_step=18
        )
        mapping = token.to_mapping()

        extra = dict(mapping)
        extra["unexpected"] = "tamper"
        with self.assertRaisesRegex(ValueError, "exact schema"):
            MacroSegmentCompletionToken.from_mapping(extra)
        missing = dict(mapping)
        del missing["owner_state"]
        with self.assertRaisesRegex(ValueError, "exact schema"):
            MacroSegmentCompletionToken.from_mapping(missing)

        decision_extra = dict(token.decision)
        decision_extra["unexpected"] = True
        decision_extra_mapping = dict(mapping)
        decision_extra_mapping["decision"] = decision_extra
        decision_extra_mapping["decision_sha256"] = hashlib.sha256(
            json.dumps(
                decision_extra,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "exact helper schema"):
            MacroSegmentCompletionToken.from_mapping(decision_extra_mapping)

        contradictory = dict(token.decision)
        contradictory["segment_done"] = True
        contradictory_mapping = dict(mapping)
        contradictory_mapping["decision"] = contradictory
        contradictory_mapping["decision_sha256"] = hashlib.sha256(
            json.dumps(
                contradictory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "contradictory"):
            MacroSegmentCompletionToken.from_mapping(contradictory_mapping)

        semantic_tampers = []
        due_acked = _completion_decision_mapping(
            control, kind="WHEEL_STOP_DUE", sim_time_s=0.03, sim_step=18
        )
        due_acked["wheel_stop_acknowledged"] = True
        semantic_tampers.append(("due_ack", due_acked, "cannot already acknowledge"))
        complete_due = _completion_decision_mapping(
            control, kind="COMPLETE", sim_time_s=0.03, sim_step=18
        )
        complete_due["wheel_stop_due"] = True
        semantic_tampers.append(("complete_due", complete_due, "contradictory"))
        fail_done = _completion_decision_mapping(
            control, kind="FAIL", sim_time_s=0.03, sim_step=18
        )
        fail_done["segment_done"] = True
        semantic_tampers.append(("fail_done", fail_done, "contradictory"))
        for label, tampered_decision, message in semantic_tampers:
            tampered_mapping = dict(mapping)
            tampered_mapping["decision"] = tampered_decision
            tampered_mapping["decision_sha256"] = hashlib.sha256(
                json.dumps(
                    tampered_decision,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    MacroSegmentCompletionToken.from_mapping(tampered_mapping)

        with self.assertRaisesRegex(ValueError, "follow the start physics step"):
            self._completion_token(
                control,
                kind="WAIT",
                sim_time_s=0.03,
                sim_step=18,
                start_sim_step=18,
            )

    def test_completion_token_rejects_cross_binding_stale_and_changed_start_readback(self) -> None:
        for label, mutation in (
            ("state", {"owner_state": MacroStateId.S2_FR_TRAVERSE.value}),
            ("profile", {"profile_id": "wrong-profile"}),
            ("source_version", {"profile_source_version": "wrong-source"}),
            ("source_plan", {"source_plan_sha256": "0" * 64}),
            (
                "full_plan",
                {"source_plan_payload_sha256": "1" * 64},
            ),
            ("accepted", {"accepted_steps_sha256": "2" * 64}),
            ("epoch", {"start_command_epoch": 999}),
        ):
            controller, ground, current_control = self._start_raw_s1()
            token = self._completion_token(
                current_control, kind="WAIT", sim_time_s=0.03, sim_step=18
            )
            tampered = replace(token, **mutation)
            rejected = controller.tick(
                ground,
                sim_time_s=0.03,
                segment_completion_token=tampered,
                source_cursor_permit=True,
            )
            with self.subTest(label=label):
                self.assertEqual(
                    rejected.terminal_outcome, MacroTerminalOutcome.SAFE_STOP
                )
                self.assertIn("completion token", rejected.reason)

        controller, ground, current_control = self._start_raw_s1()
        first = self._completion_token(
            current_control, kind="WAIT", sim_time_s=0.03, sim_step=18
        )
        accepted = controller.tick(
            ground,
            sim_time_s=0.03,
            segment_completion_token=first,
            source_cursor_permit=True,
        )
        self.assertEqual(accepted.terminal_outcome, MacroTerminalOutcome.RUNNING)
        duplicate = controller.tick(
            ground,
            sim_time_s=0.04,
            segment_completion_token=first,
            source_cursor_permit=True,
        )
        self.assertEqual(duplicate.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertIn("stale or duplicated", duplicate.reason)

        for label, changed_step, changed_readback in (
            ("start_step", 11, "c" * 64),
            ("start_readback", 10, "d" * 64),
        ):
            controller, ground, current_control = self._start_raw_s1()
            first = self._completion_token(
                current_control, kind="WAIT", sim_time_s=0.03, sim_step=18
            )
            controller.tick(
                ground,
                sim_time_s=0.03,
                segment_completion_token=first,
                source_cursor_permit=True,
            )
            changed_start = self._completion_token(
                current_control,
                kind="WAIT",
                sim_time_s=0.04,
                sim_step=26,
                start_sim_step=changed_step,
                start_readback_sha256=changed_readback,
            )
            rejected = controller.tick(
                ground,
                sim_time_s=0.04,
                segment_completion_token=changed_start,
                source_cursor_permit=True,
            )
            with self.subTest(label=label):
                self.assertEqual(
                    rejected.terminal_outcome, MacroTerminalOutcome.SAFE_STOP
                )
                self.assertIn(
                    "start step/readback binding changed", rejected.reason
                )

    def test_measured_completion_extension_is_not_preempted_by_nominal_state_timeout(self) -> None:
        real_library = build_profile_library(PROJECT_ROOT)
        v009 = next(
            source.source_version
            for source in real_library.successful_sources
            if source.source_version.startswith("v009_")
        )
        real_controller = MacroFSMController(
            self.graph,
            real_library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        ground = _observation_mapping()
        real_controller.reset(
            ground, sim_time_s=0.0, source_version=v009
        )
        real_controller._enter_state(
            MacroStateId.S5_PRE_RR_COM_SHIFT,
            MacroObservation.from_mapping(ground),
            0.0,
        )
        first_s5 = real_controller.tick(
            ground, sim_time_s=0.01, source_cursor_permit=True
        )
        self.assertEqual(
            first_s5.command_provenance["source_segment_index"], 54
        )
        v009_extension = real_controller.tick(
            ground, sim_time_s=2.60, source_cursor_permit=False
        )
        self.assertEqual(
            v009_extension.terminal_outcome, MacroTerminalOutcome.RUNNING
        )
        self.assertIsNone(real_controller._hold_started_at_s)

        controller, ground, control = self._start_raw_s1()
        still_waiting = controller.tick(
            ground,
            sim_time_s=2.60,
            source_cursor_permit=False,
        )
        self.assertEqual(still_waiting.terminal_outcome, MacroTerminalOutcome.RUNNING)
        self.assertIsNone(controller._hold_started_at_s)

        complete = self._completion_token(
            control,
            kind="COMPLETE",
            sim_time_s=2.61,
            sim_step=322,
        )
        held = controller.tick(
            ground,
            sim_time_s=2.61,
            segment_completion_token=complete,
            source_cursor_permit=True,
        )
        self.assertEqual(held.terminal_outcome, MacroTerminalOutcome.RUNNING)
        self.assertEqual(held.transition_events, ("HOLD:S1_APPROACH_AND_PRE_FR_SHIFT",))
        self.assertAlmostEqual(controller._hold_started_at_s or 0.0, 2.61)

    def test_dynamic_completion_wheel_stop_does_not_consume_a_source_action(self) -> None:
        base = self.library.get(
            DEFAULT_PRIMARY_VERSION,
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            strategy="PRIMARY_PROFILE",
        )
        frame = replace(
            base.keyframes[0], wheel_targets_rad_s=_complete_wheels(0.3)
        )
        template_binding = base.segment_bindings[0]
        profile = replace(
            base,
            profile_id="synthetic:dynamic-completion-wheel-stop",
            keyframes=(frame,),
            nominal_duration_s=0.0,
            segment_bindings=(
                _synthetic_binding(
                    source_version=base.source_version,
                    source_plan_sha256=base.source_plan_sha256,
                    source_plan_payload_sha256=template_binding.source_plan_payload_sha256,
                    accepted_steps_sha256=template_binding.accepted_steps_sha256,
                    frame=frame,
                    wheel_active_duration_s=0.01,
                    explicit_hold_s=0.01,
                ),
            ),
        )
        controller = MacroFSMController(
            self.graph,
            self.library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller._state_id = MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT
        controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=profile,
        )
        started = controller.tick(
            ground, sim_time_s=0.01, source_cursor_permit=True
        )
        control = dict(started.segment_completion_control)
        due = self._completion_token(
            control, kind="WHEEL_STOP_DUE", sim_time_s=0.02, sim_step=18
        )
        stopped = controller.tick(
            ground,
            sim_time_s=0.02,
            segment_completion_token=due,
            source_cursor_permit=True,
        )
        self.assertFalse(stopped.source_action_consumed)
        self.assertEqual(
            stopped.command_provenance["kind"], "COMPLETION_WHEEL_STOP"
        )
        self.assertEqual(stopped.segment_completion_control["kind"], "WHEEL_STOP")
        self.assertFalse(stopped.segment_completion_control["source_action"])
        self.assertEqual(controller.status["consumed_source_action_count"], 1)

    def test_duplicate_or_malformed_source_action_fails_closed(self) -> None:
        controller = self._controller()
        controller.reset(_observation_mapping(), sim_time_s=0.0)
        with self.assertRaisesRegex(ValueError, "duplicate source action"):
            controller._prepare_profile(
                self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
                profile_override=self._two_action_profile(duplicate_coordinate=True),
            )

        profile = self._two_action_profile()
        malformed_frame = replace(
            profile.keyframes[0], source_segment_index=True
        )
        with self.assertRaisesRegex(ValueError, "exact non-negative int"):
            controller._source_action_provenance(profile, malformed_frame)

        unknown_kind_frame = replace(
            profile.keyframes[0], dispatch_kind="self_consistent_but_unknown"
        )
        with self.assertRaisesRegex(ValueError, "not an allowed source primitive"):
            controller._source_action_provenance(profile, unknown_kind_frame)

        empty_segment_frame = replace(
            profile.keyframes[0], commands=(), source_event_indices=()
        )
        with self.assertRaisesRegex(ValueError, "segment_start provenance requires"):
            controller._source_action_provenance(profile, empty_segment_frame)

        nonempty_stop_frame = replace(
            profile.keyframes[0], dispatch_kind="wheel_channel_completion_stop"
        )
        with self.assertRaisesRegex(ValueError, "completion_stop provenance requires"):
            controller._source_action_provenance(profile, nonempty_stop_frame)

        valid_stop_frame = replace(
            nonempty_stop_frame, commands=(), source_event_indices=()
        )
        valid_stop = controller._source_action_provenance(
            profile, valid_stop_frame
        )
        self.assertEqual(valid_stop["dispatch_kind"], "wheel_channel_completion_stop")
        self.assertEqual(valid_stop["commands"], ())
        self.assertEqual(valid_stop["source_event_indices"], ())

        paired_controller = self._controller()
        paired_controller.reset(_observation_mapping(), sim_time_s=0.0)
        start_frame = replace(
            profile.keyframes[0], wheel_targets_rad_s=_complete_wheels(0.3)
        )
        stop_frame = replace(
            start_frame,
            time_s=0.01,
            source_time_s=start_frame.source_time_s + 0.01,
            sequence_index=start_frame.sequence_index + 1,
            dispatch_kind="wheel_channel_completion_stop",
            commands=(),
            source_event_indices=(),
            wheel_targets_rad_s=_complete_wheels(),
        )
        template_binding = profile.segment_bindings[0]
        paired_profile = replace(
            profile,
            profile_id="synthetic:start-and-completion-stop-same-segment",
            keyframes=(start_frame, stop_frame),
            nominal_duration_s=0.01,
            source_segment_range=(1000, 1000),
            source_step_indices=(1001,),
            source_commands=start_frame.commands,
            segment_bindings=(
                _synthetic_binding(
                    source_version=profile.source_version,
                    source_plan_sha256=profile.source_plan_sha256,
                    source_plan_payload_sha256=template_binding.source_plan_payload_sha256,
                    accepted_steps_sha256=template_binding.accepted_steps_sha256,
                    frame=start_frame,
                    wheel_active_duration_s=0.01,
                ),
            ),
        )
        paired_controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=paired_profile,
        )
        paired_controller._profile_started_at_s = 0.0
        start_decision = paired_controller.tick(
            _observation_mapping(), sim_time_s=0.001
        )
        stop_decision = paired_controller.tick(
            _observation_mapping(), sim_time_s=0.02
        )
        self.assertTrue(start_decision.source_action_consumed)
        self.assertTrue(stop_decision.source_action_consumed)
        self.assertEqual(
            {
                coordinate[2]
                for coordinate in paired_controller._consumed_source_coordinates
            },
            {"segment_start", "wheel_channel_completion_stop"},
        )

    def test_decision_rejects_missing_tampered_or_inconsistent_source_provenance(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        reset = controller.reset(ground, sim_time_s=0.0)
        profile = self._two_action_profile()
        controller._prepare_profile(
            self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT),
            profile_override=profile,
        )
        source = controller.tick(ground, sim_time_s=0.001)
        valid = dict(source.command_provenance)

        missing = dict(valid)
        del missing["commands"]
        with self.assertRaisesRegex(ValueError, "exact schema"):
            replace(source, command_provenance=missing)

        extra = dict(valid)
        extra["unexpected"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "exact schema"):
            replace(source, command_provenance=extra)

        tampered = dict(valid)
        tampered["commands"] = ("servo front_left_hip 99",)
        with self.assertRaisesRegex(ValueError, "does not match"):
            replace(source, command_provenance=tampered)

        bool_index = dict(valid)
        bool_index["source_segment_index"] = True
        with self.assertRaisesRegex(ValueError, "exact non-negative int"):
            replace(source, command_provenance=bool_index)

        with self.assertRaisesRegex(ValueError, "must exactly match"):
            replace(source, source_action_consumed=False)
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            replace(source, command_changed=False)

        none_for_changed = dict(reset.command_provenance)
        with self.assertRaisesRegex(ValueError, "requires dispatch provenance"):
            replace(
                source,
                source_action_consumed=False,
                command_provenance=none_for_changed,
            )

        false_non_source_dispatch = dict(reset.command_provenance)
        false_non_source_dispatch["kind"] = "SAFE_STOP_ZERO_WHEELS"
        with self.assertRaisesRegex(ValueError, "unchanged target map"):
            replace(reset, command_provenance=false_non_source_dispatch)

    def test_controller_and_worker_recompute_the_same_real_v003_identity(self) -> None:
        from fsm_50mm_recording_derived_v3.worker_macro_fsm_session import (
            _source_action_identity as worker_source_action_identity,
            _source_action_identity_payload as worker_identity_payload,
        )

        library = build_profile_library(PROJECT_ROOT)
        profile = next(
            profile
            for profile in library.profiles
            if profile.source_version == DEFAULT_PRIMARY_VERSION
            and any(frame.source_segment_index == 7 for frame in profile.keyframes)
        )
        frame = next(
            frame for frame in profile.keyframes if frame.source_segment_index == 7
        )
        provenance = MacroFSMController._source_action_provenance(profile, frame)
        self.assertEqual(
            worker_identity_payload(provenance)["schema_version"],
            SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
        )
        self.assertEqual(
            worker_source_action_identity(provenance),
            provenance["source_action_identity"],
        )

    def test_internal_map_epoch_or_consumption_drift_fails_closed(self) -> None:
        ground = _observation_mapping()

        map_controller = self._controller()
        map_controller.reset(ground, sim_time_s=0.0)
        map_controller._current_servo_targets[SERVO_JOINT_NAMES[0]] = 1.0
        with self.assertRaisesRegex(RuntimeError, "target_changed"):
            map_controller.tick(ground, sim_time_s=0.01)

        epoch_controller = self._controller()
        epoch_controller.reset(ground, sim_time_s=0.0)
        epoch_controller._command_epoch += 1
        with self.assertRaisesRegex(RuntimeError, "command_epoch drift"):
            epoch_controller.tick(ground, sim_time_s=0.01)

        consumed_controller = self._controller()
        consumed_controller.reset(ground, sim_time_s=0.0)
        consumed_controller._consumed_source_actions.add("0" * 64)
        consumed_controller._consumed_source_coordinates.add(
            (DEFAULT_PRIMARY_VERSION, 9999, "segment_start")
        )
        with self.assertRaisesRegex(RuntimeError, "consumption drift"):
            consumed_controller.tick(ground, sim_time_s=0.01)

    def test_safe_stop_provenance_only_claims_an_actual_zero_wheel_dispatch(self) -> None:
        stopped = self._controller()
        reset = stopped.reset(_observation_mapping(), sim_time_s=0.0)
        decision = stopped.tick(
            _observation_mapping(robot_fell=True), sim_time_s=0.01
        )
        self.assertFalse(decision.target_changed)
        self.assertFalse(decision.command_changed)
        self.assertEqual(decision.command_epoch, reset.command_epoch)
        self.assertEqual(decision.command_provenance["kind"], "NONE")
        self.assertIn("SAFE_STOP:S0_INITIALIZE", decision.transition_events)

        moving_payload = _observation_mapping(
            wheel_targets_rad_s=_complete_wheels(0.25)
        )
        moving = self._controller()
        moving_reset = moving.reset(moving_payload, sim_time_s=0.0)
        stopped_from_motion = moving.tick(
            _observation_mapping(robot_fell=True), sim_time_s=0.01
        )
        self.assertTrue(stopped_from_motion.target_changed)
        self.assertTrue(stopped_from_motion.command_changed)
        self.assertEqual(
            stopped_from_motion.command_epoch, moving_reset.command_epoch + 1
        )
        self.assertEqual(
            stopped_from_motion.command_provenance["kind"],
            "SAFE_STOP_ZERO_WHEELS",
        )

    def test_hard_failure_safe_stops_with_exact_full_targets(self) -> None:
        controller = self._controller()
        base = _observation_mapping()
        controller.reset(base, sim_time_s=0.0)
        decision = controller.tick(
            _observation_mapping(robot_fell=True), sim_time_s=0.01
        )
        self.assertEqual(decision.macro_state, MacroStateId.SAFE_STOP)
        self.assertEqual(decision.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertEqual(set(decision.servo_targets_deg), set(SERVO_JOINT_NAMES))
        self.assertEqual(set(decision.wheel_targets_rad_s), set(WHEEL_JOINT_NAMES))
        self.assertTrue(all(value == 0.0 for value in decision.wheel_targets_rad_s.values()))

    def test_unknown_unproduced_safety_does_not_hard_fail_and_is_retained(self) -> None:
        controller = self._controller()
        unknown = _observation_mapping(
            body_stuck=None,
            active_leg_trapped=None,
        )
        reset = controller.reset(unknown, sim_time_s=0.0)
        self.assertIsNone(reset.guard_evidence["body_stuck"])
        self.assertIsNone(reset.guard_evidence["active_leg_trapped"])
        self.assertFalse(reset.guard_evidence["body_stuck_available"])
        self.assertFalse(reset.guard_evidence["active_leg_trapped_available"])

        initialized = controller.tick(unknown, sim_time_s=0.01)
        self.assertNotEqual(initialized.macro_state, MacroStateId.SAFE_STOP)
        self.assertEqual(initialized.terminal_outcome, MacroTerminalOutcome.RUNNING)
        self.assertIsNone(initialized.guard_evidence["body_stuck"])
        self.assertIsNone(initialized.guard_evidence["active_leg_trapped"])

        controller.tick(unknown, sim_time_s=0.02)
        controller.tick(unknown, sim_time_s=0.03)
        retry = controller.tick(unknown, sim_time_s=0.84)
        self.assertNotEqual(retry.macro_state, MacroStateId.SAFE_STOP)
        self.assertEqual(retry.retry_count, 1)
        self.assertIsNone(retry.guard_evidence["body_stuck"])
        self.assertIsNone(retry.guard_evidence["active_leg_trapped"])

    def test_explicit_true_unproduced_safety_signals_hard_stop(self) -> None:
        for field_name, reason in (
            ("body_stuck", "body stuck"),
            ("active_leg_trapped", "active leg trapped"),
        ):
            with self.subTest(field=field_name):
                controller = self._controller()
                controller.reset(_observation_mapping(), sim_time_s=0.0)
                decision = controller.tick(
                    _observation_mapping(**{field_name: True}), sim_time_s=0.01
                )
                self.assertEqual(decision.macro_state, MacroStateId.SAFE_STOP)
                self.assertEqual(
                    decision.terminal_outcome, MacroTerminalOutcome.SAFE_STOP
                )
                self.assertIn(reason, decision.reason)
                self.assertIs(decision.guard_evidence[field_name], True)
                self.assertTrue(
                    decision.guard_evidence[f"{field_name}_available"]
                )

    def test_unmet_primary_guard_rechecks_hold_target_without_action_replay(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller.tick(ground, sim_time_s=0.01)  # S0 -> S1
        primary = controller.tick(ground, sim_time_s=0.02)  # primary command epoch
        hold = controller.tick(ground, sim_time_s=0.03)  # bounded hold
        retry = controller.tick(ground, sim_time_s=0.84)
        self.assertEqual(retry.macro_state, MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT)
        self.assertEqual(retry.retry_count, 1)
        self.assertEqual(retry.profile_strategy, "PRIMARY_PROFILE")
        self.assertEqual(retry.profile_source_version, DEFAULT_PRIMARY_VERSION)
        self.assertEqual(retry.servo_targets_deg, hold.servo_targets_deg)
        self.assertEqual(retry.wheel_targets_rad_s, hold.wheel_targets_rad_s)
        self.assertEqual(retry.command_epoch, hold.command_epoch)
        self.assertFalse(retry.command_changed)
        self.assertEqual(controller.status["consumed_source_action_count"], 1)
        self.assertIn("RETRY:S1_APPROACH_AND_PRE_FR_SHIFT:1", retry.transition_events)
        self.assertEqual(self.graph.sha256, controller.graph.sha256)
        self.assertGreaterEqual(primary.command_epoch, 1)

        exhausted = controller.tick(ground, sim_time_s=1.65)
        self.assertEqual(exhausted.macro_state, MacroStateId.SAFE_STOP)
        self.assertEqual(exhausted.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)

    def test_recording_source_is_selectable_only_at_reset(self) -> None:
        real_library = build_profile_library(PROJECT_ROOT)
        source_v009 = next(
            source.source_version
            for source in real_library.successful_sources
            if source.source_version.startswith("v009_")
        )
        controller = MacroFSMController(
            self.graph,
            real_library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        controller.reset(
            _observation_mapping(), sim_time_s=0.0, source_version=source_v009
        )
        entered = controller.tick(_observation_mapping(), sim_time_s=0.01)
        self.assertEqual(entered.profile_source_version, source_v009)
        self.assertEqual(entered.profile_strategy, "ALTERNATE_PROFILE_1")

    def test_runtime_cross_source_snapshot_jump_is_rejected(self) -> None:
        real_library = build_profile_library(PROJECT_ROOT)
        source_v009 = next(
            source.source_version
            for source in real_library.successful_sources
            if source.source_version.startswith("v009_")
        )
        controller = MacroFSMController(
            self.graph,
            real_library,
            primary_source_version=DEFAULT_PRIMARY_VERSION,
        )
        ground = MacroObservation.from_mapping(_observation_mapping())
        controller.reset(ground, sim_time_s=0.0)
        controller.tick(ground, sim_time_s=0.01)  # enter v003 S1
        alternate = real_library.get(
            source_v009,
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            strategy="ALTERNATE_PROFILE_1",
        )
        state = self.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT)
        with self.assertRaisesRegex(ValueError, "cross-source"):
            controller._prepare_profile(state, profile_override=alternate)
        self.assertEqual(controller.status["profile_source_version"], DEFAULT_PRIMARY_VERSION)

    def _reach_s8_with_early_rl_air(self) -> tuple[MacroFSMController, float, dict[str, object]]:
        c = self._controller()
        p = _observation_mapping()
        c.reset(p, sim_time_s=0.0)
        early_rl_air = _with_leg(p, "RL", "AIR", crossed=False)
        # Preserve RL's rear-wheel x position; this is an early lift event,
        # not an invented approach/crossing displacement.
        early_rl_air["wheel_center_w_m"] = dict(p["wheel_center_w_m"])  # type: ignore[arg-type]
        c.tick(early_rl_air, sim_time_s=0.01)  # enter S1; episode sees early RL air
        c.tick(p, sim_time_s=0.02)  # S1 command
        shifted = _observation_mapping(base=(-0.01, 0.01, 0.1))
        c.tick(shifted, sim_time_s=0.03)  # S2
        fr_air = _with_leg(shifted, "FR", "AIR", crossed=False)
        c.tick(fr_air, sim_time_s=0.04)
        front = _top_legs(shifted, "FR")
        c.tick(front, sim_time_s=0.05)  # S3
        fl_air = _with_leg(front, "FL", "AIR", crossed=False)
        c.tick(fl_air, sim_time_s=0.06)
        front = _top_legs(front, "FR", "FL")
        c.tick(front, sim_time_s=0.07)  # S4
        c.tick(front, sim_time_s=0.08)  # moving-wheel profile command
        approach = dict(front)
        face = dict(approach["wheel_front_face_clearance_m"])  # type: ignore[arg-type]
        face["RR"] = -0.05
        approach["wheel_front_face_clearance_m"] = face
        c.tick(approach, sim_time_s=0.09)  # S5, zero boundary
        rr_air = _with_leg(approach, "RR", "AIR", crossed=False)
        c.tick(rr_air, sim_time_s=0.10)
        c.tick(rr_air, sim_time_s=0.11)  # S6
        c.tick(rr_air, sim_time_s=0.12)
        three_top = _top_legs(front, "FR", "FL", "RR")
        c.tick(three_top, sim_time_s=0.13)  # S7
        enter_s8 = c.tick(three_top, sim_time_s=0.14)
        self.assertEqual(enter_s8.macro_state, MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE)
        return c, 0.14, three_top

    def test_early_rl_air_does_not_satisfy_s8_phase_local_lift_guard(self) -> None:
        controller, time_s, three_top = self._reach_s8_with_early_rl_air()
        controller.tick(three_top, sim_time_s=time_s + 0.01)  # S8 command
        rl_top_without_new_lift = _top_legs(three_top, "RL")
        decision = controller.tick(rl_top_without_new_lift, sim_time_s=time_s + 0.02)
        self.assertEqual(decision.macro_state, MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE)
        self.assertFalse(
            decision.guard_evidence["state_airborne_before_crossing"]["RL"]
        )
        self.assertNotEqual(decision.macro_state, MacroStateId.S9_FINAL_ADVANCE)

    def test_s1_to_s2_preserves_only_live_boundary_fr_airborne_event(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller.tick(ground, sim_time_s=0.01)  # S1
        controller.tick(ground, sim_time_s=0.02)  # S1 command
        boundary_air = _with_leg(ground, "FR", "AIR", crossed=False)
        entered_s2 = controller.tick(boundary_air, sim_time_s=0.03)
        self.assertEqual(entered_s2.macro_state, MacroStateId.S2_FR_TRAVERSE)
        fr_top = _top_legs(ground, "FR")
        command = controller.tick(fr_top, sim_time_s=0.04)
        self.assertTrue(command.guard_evidence["state_airborne_before_crossing"]["FR"])
        traversed = controller.tick(fr_top, sim_time_s=0.05)
        self.assertEqual(traversed.macro_state, MacroStateId.S3_FL_TRAVERSE)

    def test_s5_to_s6_preserves_live_boundary_rr_airborne_event(self) -> None:
        controller = self._controller()
        p = _observation_mapping()
        controller.reset(p, sim_time_s=0.0)
        controller.tick(p, sim_time_s=0.01)
        controller.tick(p, sim_time_s=0.02)
        shifted = _observation_mapping(base=(-0.01, 0.01, 0.1))
        controller.tick(shifted, sim_time_s=0.03)
        controller.tick(_with_leg(shifted, "FR", "AIR"), sim_time_s=0.04)
        front = _top_legs(shifted, "FR")
        controller.tick(front, sim_time_s=0.05)
        controller.tick(_with_leg(front, "FL", "AIR"), sim_time_s=0.06)
        front = _top_legs(front, "FR", "FL")
        moving = controller.tick(front, sim_time_s=0.07)
        self.assertTrue(any(abs(value) > 0.0 for value in moving.wheel_targets_rad_s.values()))
        settled = controller.tick(front, sim_time_s=0.08)
        self.assertTrue(all(value == 0.0 for value in settled.wheel_targets_rad_s.values()))
        approach = dict(front)
        face = dict(approach["wheel_front_face_clearance_m"])  # type: ignore[arg-type]
        face["RR"] = -0.05
        approach["wheel_front_face_clearance_m"] = face
        boundary = controller.tick(approach, sim_time_s=0.09)
        self.assertFalse(boundary.command_changed)
        self.assertEqual(boundary.macro_state, MacroStateId.S5_PRE_RR_COM_SHIFT)
        self.assertTrue(all(value == 0.0 for value in boundary.wheel_targets_rad_s.values()))
        rr_air = _with_leg(approach, "RR", "AIR", crossed=False)
        controller.tick(rr_air, sim_time_s=0.10)  # S5 command
        entered_s6 = controller.tick(rr_air, sim_time_s=0.11)
        self.assertEqual(entered_s6.macro_state, MacroStateId.S6_RR_TRAVERSE)
        three_top = _top_legs(front, "FR", "FL", "RR")
        traversed = controller.tick(three_top, sim_time_s=0.12)
        self.assertTrue(traversed.guard_evidence["state_airborne_before_crossing"]["RR"])
        self.assertEqual(traversed.macro_state, MacroStateId.S7_PRE_RL_SUPPORT_SETUP)

    def test_traverse_latches_active_top_and_prior_support_top_at_air_tail(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        fr_top = _top_legs(ground, "FR")
        fl_air = _with_leg(fr_top, "FL", "AIR", crossed=False)
        controller._episode_top_seen["FR"] = True
        controller._enter_state(
            MacroStateId.S3_FL_TRAVERSE,
            MacroObservation.from_mapping(fl_air),
            1.0,
        )

        fl_top = _top_legs(fl_air, "FL")
        fl_top_fr_air = _with_leg(fl_top, "FR", "AIR", crossed=True)
        emitted = controller.tick(fl_top_fr_air, sim_time_s=1.01)
        self.assertEqual(emitted.macro_state, MacroStateId.S3_FL_TRAVERSE)
        self.assertTrue(emitted.command_changed)

        air_tail = _with_leg(fl_top_fr_air, "FL", "AIR", crossed=True)
        transitioned = controller.tick(air_tail, sim_time_s=1.02)
        self.assertEqual(transitioned.macro_state, MacroStateId.S4_FRONT_PAIR_ADVANCE)
        self.assertTrue(transitioned.guard_evidence["active_leg_top"])
        self.assertEqual(
            transitioned.guard_evidence["required_top_evidence"],
            {"FR": True, "FL": True},
        )

    def test_immediate_boundary_carry_is_fresh_and_not_episode_global(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        controller._episode_top_seen.update({"FR": True, "FL": True})
        controller._state_id = MacroStateId.S3_FL_TRAVERSE
        controller._entry_body_position = (0.0, 0.0, 0.1)
        front_air = _with_leg(
            _with_leg(
                _observation_mapping(base=(0.05, 0.0, 0.1)),
                "FR",
                "AIR",
                crossed=True,
            ),
            "FL",
            "AIR",
            crossed=True,
        )
        controller._record_boundary_carry(
            MacroStateId.S3_FL_TRAVERSE,
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            MacroObservation.from_mapping(front_air),
        )
        controller._enter_state(
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            MacroObservation.from_mapping(front_air),
            1.0,
        )
        fresh = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S4_FRONT_PAIR_ADVANCE),
            MacroObservation.from_mapping(front_air),
            timeline_complete=True,
        )
        self.assertTrue(fresh.satisfied)
        self.assertTrue(fresh.evidence["predecessor_carry_fresh"])
        self.assertTrue(fresh.evidence["inherited_progress_ready"])

        controller._boundary_from_state = MacroStateId.S2_FR_TRAVERSE
        stale = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S4_FRONT_PAIR_ADVANCE),
            MacroObservation.from_mapping(front_air),
            timeline_complete=True,
        )
        self.assertFalse(stale.satisfied)
        self.assertFalse(stale.evidence["predecessor_carry_fresh"])

    def test_s5_rejects_body_proxy_and_accepts_current_com_or_rr_unload(self) -> None:
        controller = self._controller()
        start = _observation_mapping(
            base=(0.0, 0.0, 0.1),
            test_whole_body_com_position_m=(0.0, 0.0, 0.1),
        )
        controller.reset(start, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S5_PRE_RR_COM_SHIFT,
            MacroObservation.from_mapping(start),
            0.0,
        )

        body_only = _observation_mapping(
            base=(0.06, 0.03, 0.1),
            test_whole_body_com_position_m=(0.0, 0.0, 0.1),
        )
        proxy_rejected = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S5_PRE_RR_COM_SHIFT),
            MacroObservation.from_mapping(body_only),
            timeline_complete=True,
        )
        self.assertFalse(proxy_rejected.satisfied)
        self.assertFalse(proxy_rejected.evidence["body_root_proxy_guard_eligible"])

        com_shift = _observation_mapping(
            test_whole_body_com_position_m=(0.01, 0.01, 0.1)
        )
        current_com = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S5_PRE_RR_COM_SHIFT),
            MacroObservation.from_mapping(com_shift),
            timeline_complete=True,
        )
        self.assertTrue(current_com.satisfied)
        self.assertTrue(current_com.evidence["true_com_displacement_ready"])

        rr_air = _with_leg(start, "RR", "AIR", crossed=False)
        measured_unload = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S5_PRE_RR_COM_SHIFT),
            MacroObservation.from_mapping(rr_air),
            timeline_complete=True,
        )
        self.assertTrue(measured_unload.satisfied)
        self.assertTrue(measured_unload.evidence["active_leg_measured_unloaded"])

    def test_s1_rejects_root_proxy_and_accepts_true_com_or_fr_unload(self) -> None:
        controller = self._controller()
        start = _observation_mapping(
            base=(0.0, 0.0, 0.1),
            test_whole_body_com_position_m=(0.0, 0.0, 0.1),
        )
        controller.reset(start, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT,
            MacroObservation.from_mapping(start),
            0.0,
        )
        state = controller.graph.get(MacroStateId.S1_APPROACH_AND_PRE_FR_SHIFT)

        root_only = _observation_mapping(
            base=(-0.06, 0.04, 0.1),
            test_whole_body_com_position_m=(0.0, 0.0, 0.1),
        )
        rejected = controller._evaluate_guard(
            state, MacroObservation.from_mapping(root_only), timeline_complete=True
        )
        self.assertFalse(rejected.satisfied)
        self.assertFalse(rejected.evidence["body_root_proxy_guard_eligible"])

        true_com = _observation_mapping(
            test_whole_body_com_position_m=(-0.01, 0.01, 0.1)
        )
        shifted = controller._evaluate_guard(
            state, MacroObservation.from_mapping(true_com), timeline_complete=True
        )
        self.assertTrue(shifted.satisfied)
        self.assertTrue(shifted.evidence["true_com_displacement_ready"])

        fr_air = _with_leg(start, "FR", "AIR", crossed=False)
        unloaded = controller._evaluate_guard(
            state, MacroObservation.from_mapping(fr_air), timeline_complete=True
        )
        self.assertTrue(unloaded.satisfied)
        self.assertTrue(unloaded.evidence["active_leg_measured_unloaded"])

    def test_recording_telemetry_identity_is_not_new_physical_evidence(self) -> None:
        library = build_profile_library(PROJECT_ROOT)
        expected_action_counts = {"v003": 112, "v008": 119, "v009": 132, "v010": 142}
        for source in library.successful_sources:
            with self.subTest(source=source.source_version):
                rows = [
                    json.loads(line)
                    for line in source.gate_a_run_dir.joinpath(
                        "minimal_telemetry.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(rows)
                self.assertTrue(
                    all(
                        "centroidal_support_evidence" not in row
                        and "feedback_recovery_observation" not in row
                        for row in rows
                    )
                )

                profiles = tuple(
                    profile
                    for profile in library.profiles
                    if profile.source_version == source.source_version
                )
                frames = tuple(
                    frame for profile in profiles for frame in profile.keyframes
                )
                prefix = source.source_version.split("_", 1)[0]
                self.assertEqual(len(frames), expected_action_counts[prefix])
                coordinates = {
                    (source.source_version, frame.source_segment_index, frame.dispatch_kind)
                    for frame in frames
                }
                self.assertEqual(len(coordinates), len(frames))
                for profile in profiles:
                    self.assertEqual(profile.source_version, source.source_version)
                    for frame in profile.keyframes:
                        provenance = ProductionMacroFSMController._source_action_provenance(
                            profile, frame
                        )
                        self.assertEqual(
                            provenance["source_version"], source.source_version
                        )
                        self.assertEqual(provenance["kind"], "SOURCE_ACTION")
                s10_profiles = tuple(
                    profile
                    for profile in profiles
                    if profile.state_id == MacroStateId.S10_POSTURE_RECOVERY
                )
                self.assertTrue(s10_profiles)
                self.assertTrue(
                    all(
                        profile.source_version == source.source_version
                        for profile in s10_profiles
                    )
                )

    def test_feedback_provenance_exact_pair_leg_joint_and_composite_binding(self) -> None:
        centroid_sha = "a" * 64
        feedback_sha = "b" * 64
        evidence_sha = hashlib.sha256(
            json.dumps(
                {
                    "schema_version": FEEDBACK_RECOVERY_EVIDENCE_BINDING_SCHEMA,
                    "centroidal_support_evidence_sha256": centroid_sha,
                    "feedback_recovery_observation_sha256": feedback_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        provenance = dict(_empty_command_provenance())
        provenance.update(
            kind="FEEDBACK_RECOVERY",
            recovery_stage=FeedbackRecoveryStage.SAFE_PROBE.value,
            recovery_action=(
                FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE.value
            ),
            recovery_evidence_sha256=evidence_sha,
            recovery_centroidal_evidence_sha256=centroid_sha,
            recovery_feedback_observation_sha256=feedback_sha,
            recovery_target_map_sha256="c" * 64,
            recovery_direction_sign=1,
            recovery_attempt=1,
            recovery_leg="FR",
            recovery_joint="front_right_hip",
            recovery_configuration_sha256="d" * 64,
        )
        canonical = _canonical_command_provenance(provenance)
        self.assertEqual(canonical["recovery_evidence_sha256"], evidence_sha)
        self.assertEqual(canonical["recovery_direction_sign"], 1)

        invalid_pair = dict(provenance)
        invalid_pair["recovery_stage"] = FeedbackRecoveryStage.INCREMENT.value
        with self.assertRaisesRegex(ValueError, "stage/action pair"):
            _canonical_command_provenance(invalid_pair)
        cross_leg = dict(provenance)
        cross_leg["recovery_joint"] = "front_left_hip"
        with self.assertRaisesRegex(ValueError, "leg/joint"):
            _canonical_command_provenance(cross_leg)
        rebound = dict(provenance)
        rebound["recovery_feedback_observation_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "composite evidence"):
            _canonical_command_provenance(rebound)

    def test_s7_requires_current_model_bound_support_and_proven_wrench(self) -> None:
        controller = self._controller()
        ground = _observation_mapping()
        controller.reset(ground, sim_time_s=0.0)
        state = controller.graph.get(MacroStateId.S7_PRE_RL_SUPPORT_SETUP)

        primary = _observation_mapping(
            classes={"FL": "TOP", "FR": "AIR", "RL": "AIR", "RR": "TOP"},
            face={"FL": 0.1, "FR": -0.1, "RL": -0.1, "RR": 0.1},
            top_clearance={"FL": 0.0, "FR": 0.01, "RL": 0.01, "RR": 0.0},
            test_active_swing_leg="RL",
        )
        primary_result = controller._evaluate_guard(
            state, MacroObservation.from_mapping(primary), timeline_complete=True
        )
        self.assertTrue(primary_result.satisfied)
        self.assertTrue(primary_result.evidence["required_primary_diagonal_proven"])

        alternate = _with_leg(
            _top_legs(_observation_mapping(), "FR", "FL", "RR"),
            "RL",
            "AIR",
            crossed=False,
        )
        alternate_result = controller._evaluate_guard(
            state, MacroObservation.from_mapping(alternate), timeline_complete=True
        )
        self.assertTrue(alternate_result.satisfied)
        self.assertTrue(
            alternate_result.evidence["declaratively_validated_alternate_support"]
        )
        self.assertTrue(
            alternate_result.evidence[
                "support_set_bound_to_current_qualified_contacts"
            ]
        )

        missing_required = _with_leg(alternate, "RR", "AIR", crossed=False)
        missing_result = controller._evaluate_guard(
            state,
            MacroObservation.from_mapping(missing_required),
            timeline_complete=True,
        )
        self.assertFalse(missing_result.satisfied)
        self.assertFalse(
            missing_result.evidence["declaratively_validated_alternate_support"]
        )

        geometry_only = dict(alternate)
        geometry_only["test_wrench_proven"] = False
        geometry_only = _attach_strict_evidence(geometry_only)
        geometry_result = controller._evaluate_guard(
            state,
            MacroObservation.from_mapping(geometry_only),
            timeline_complete=True,
        )
        self.assertFalse(geometry_result.satisfied)
        self.assertFalse(geometry_result.evidence["support_wrench_proven"])

    def test_s8_holds_rl_lift_cursor_then_releases_once_from_s7_com_baseline(self) -> None:
        baseline = _with_leg(
            _top_legs(_observation_mapping(), "FR", "FL", "RR"),
            "RL",
            "AIR",
            crossed=False,
        )
        controller = self._controller()
        controller.reset(baseline, sim_time_s=0.0)
        baseline_observation = MacroObservation.from_mapping(baseline)
        controller._enter_state(
            MacroStateId.S7_PRE_RL_SUPPORT_SETUP, baseline_observation, 0.0
        )
        baseline_sha = controller._s8_release_baseline_evidence_sha256
        controller._enter_state(
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
            baseline_observation,
            0.0,
        )
        profile = controller._active_profile
        self.assertIsNotNone(profile)
        assert profile is not None
        frame = replace(
            profile.keyframes[0], physical_phase="RL_UNLOAD_AND_LIFT"
        )
        controller._active_profile = replace(profile, keyframes=(frame,))
        controller._visible_keyframe_indices = [0]
        controller._visible_cursor = 0
        before_count = len(controller._consumed_source_actions)

        held = controller._start_next_segment(baseline_observation)
        self.assertFalse(held.source_action_consumed)
        self.assertEqual(controller._visible_cursor, 0)
        self.assertEqual(len(controller._consumed_source_actions), before_count)
        self.assertTrue(all(value == 0.0 for value in controller._current_wheel_targets.values()))

        centers = dict(baseline["wheel_center_w_m"])  # type: ignore[arg-type]
        loads = {"FR": 40.0, "FL": 29.05, "RR": 29.05, "RL": 0.0}
        total = sum(loads.values())
        shifted_com = (
            sum(loads[leg] * centers[leg][0] for leg in ("FR", "FL", "RR")) / total,
            sum(loads[leg] * centers[leg][1] for leg in ("FR", "FL", "RR")) / total,
            0.1,
        )
        shifted = dict(baseline)
        shifted["wheel_contact_load_n"] = loads
        shifted["test_whole_body_com_position_m"] = shifted_com
        shifted = _attach_strict_evidence(shifted)
        shifted_observation = MacroObservation.from_mapping(shifted)
        released = controller._start_next_segment(shifted_observation)
        self.assertTrue(released.source_action_consumed)
        self.assertTrue(controller._s8_release_open)
        self.assertEqual(controller._visible_cursor, 1)
        self.assertEqual(controller._s8_release_baseline_evidence_sha256, baseline_sha)
        second = controller._start_next_segment(shifted_observation)
        self.assertFalse(second.source_action_consumed)
        self.assertEqual(len(controller._consumed_source_actions), before_count + 1)

    def test_feedback_probe_return_negative_probe_is_permit_gated_and_n_plus_one_bound(self) -> None:
        payload = self._s10_air_candidate()
        controller, _initial = self._production_s10(payload)
        source_count = len(controller._consumed_source_actions)

        step1 = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        positive = controller.tick(
            step1, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertEqual(positive.command_provenance["kind"], "FEEDBACK_RECOVERY")
        self.assertEqual(positive.command_provenance["recovery_direction_sign"], 1)
        self.assertEqual(
            positive.command_provenance["recovery_action"],
            FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE.value,
        )
        self.assertTrue(all(value == 0.0 for value in positive.wheel_targets_rad_s.values()))

        measured_positive = dict(controller._current_servo_targets)
        centers_positive = dict(payload["wheel_center_w_m"])  # type: ignore[arg-type]
        x, y, z = centers_positive["FR"]
        centers_positive["FR"] = (x, y, z - 0.001)
        positive_ack_payload = dict(payload)
        positive_ack_payload["wheel_center_w_m"] = centers_positive
        positive_ack_payload["test_measured_servo_positions_deg"] = measured_positive
        step2 = _current_test_observation(
            positive_ack_payload,
            sim_time_s=2.0e-6,
            sim_step=2,
            command_epoch=controller._command_epoch,
            verified_command_epoch=controller._command_epoch,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        acknowledged = controller.tick(
            step2, sim_time_s=2.0e-6, source_cursor_permit=False
        )
        self.assertFalse(acknowledged.command_changed)
        self.assertEqual(
            controller.status["feedback_recovery_stage"],
            FeedbackRecoveryStage.RETURN_TO_REFERENCE.value,
        )

        step3 = _current_test_observation(
            positive_ack_payload,
            sim_time_s=3.0e-6,
            sim_step=3,
            command_epoch=controller._command_epoch,
            verified_command_epoch=controller._command_epoch,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        returned = controller.tick(
            step3, sim_time_s=3.0e-6, source_cursor_permit=True
        )
        self.assertEqual(
            returned.command_provenance["recovery_action"],
            FeedbackRecoveryAction.RETURN_TO_IMMUTABLE_REFERENCE.value,
        )
        self.assertEqual(returned.command_provenance["recovery_direction_sign"], 1)

        return_ack_payload = dict(payload)
        return_ack_payload["test_measured_servo_positions_deg"] = _complete_servos()
        step4 = _current_test_observation(
            return_ack_payload,
            sim_time_s=4.0e-6,
            sim_step=4,
            command_epoch=controller._command_epoch,
            verified_command_epoch=controller._command_epoch,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        return_ack = controller.tick(
            step4, sim_time_s=4.0e-6, source_cursor_permit=False
        )
        self.assertFalse(return_ack.command_changed)

        step5 = _current_test_observation(
            return_ack_payload,
            sim_time_s=5.0e-6,
            sim_step=5,
            command_epoch=controller._command_epoch,
            verified_command_epoch=controller._command_epoch,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        negative = controller.tick(
            step5, sim_time_s=5.0e-6, source_cursor_permit=True
        )
        self.assertEqual(
            negative.command_provenance["recovery_action"],
            FeedbackRecoveryAction.CONSERVATIVE_DIAGNOSTIC_PROBE.value,
        )
        self.assertEqual(negative.command_provenance["recovery_direction_sign"], -1)
        self.assertEqual(len(controller._consumed_source_actions), source_count)

    def _assert_safe_first_probe_freezes_worker_reconstructible_dispatch_configuration(self) -> None:
        payload = self._s10_air_candidate()
        controller, _initial = self._production_s10(payload)
        reference_targets = dict(controller._current_servo_targets)
        current = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=reference_targets,
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        expected = self._worker_reconstructed_configuration_sha256(
            controller,
            current,
            leg="FR",
            reference_targets=reference_targets,
        )
        issued = controller.tick(
            current, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertTrue(issued.command_changed)
        self.assertEqual(
            issued.command_provenance["recovery_configuration_sha256"], expected
        )
        self.assertEqual(controller._feedback_configuration_sha256, expected)
        self.assertEqual(controller._feedback_baseline["configuration_sha256"], expected)
        self.assertEqual(controller._feedback_reference_targets, reference_targets)

    def _assert_unsafe_first_sign_skip_freezes_configuration_at_first_actual_dispatch(self) -> None:
        payload = self._s10_air_candidate()
        reference_targets = dict(payload["servo_targets_deg"])  # type: ignore[arg-type]
        _lower, upper = command_limits_for_servo("front_right_hip")
        reference_targets["front_right_hip"] = upper - float(
            FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"]
        )
        payload["servo_targets_deg"] = reference_targets
        payload = _attach_strict_evidence(payload)
        controller, initial = self._production_s10(payload)

        first = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=reference_targets,
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        skipped = controller.tick(
            first, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertFalse(skipped.command_changed)
        self.assertEqual(controller._feedback_action_count, 0)
        self.assertEqual(controller._feedback_configuration_sha256, "")
        self.assertEqual(controller._feedback_baseline, {})
        self.assertEqual(controller._feedback_reference_targets, reference_targets)
        stale_candidate = controller._recovery_configuration_identity(initial, "FR")

        later_payload = dict(payload)
        centers = dict(later_payload["wheel_center_w_m"])  # type: ignore[arg-type]
        x, y, z = centers["FR"]
        centers["FR"] = (x, y, z - 0.001)
        later_payload["wheel_center_w_m"] = centers
        later = _current_test_observation(
            later_payload,
            sim_time_s=2.0e-6,
            sim_step=2,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=reference_targets,
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        expected = self._worker_reconstructed_configuration_sha256(
            controller,
            later,
            leg="FR",
            reference_targets=reference_targets,
        )
        self.assertNotEqual(expected, stale_candidate)
        issued = controller.tick(
            later, sim_time_s=2.0e-6, source_cursor_permit=True
        )
        self.assertTrue(issued.command_changed)
        self.assertEqual(
            issued.command_provenance["recovery_direction_sign"], -1
        )
        self.assertEqual(
            issued.command_provenance["recovery_configuration_sha256"], expected
        )
        self.assertEqual(controller._feedback_configuration_sha256, expected)
        self.assertEqual(
            controller._feedback_baseline["wheel_z_m"],
            later.feedback_recovery_observation.wheel_center_w_m["FR"][2],
        )
        self.assertEqual(controller._feedback_reference_targets, reference_targets)

    def test_feedback_skipped_n_plus_one_fails_closed(self) -> None:
        payload = self._s10_air_candidate()
        controller, _initial = self._production_s10(payload)
        issued_observation = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        issued = controller.tick(
            issued_observation, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertTrue(issued.command_changed)
        skipped = _current_test_observation(
            payload,
            sim_time_s=3.0e-6,
            sim_step=3,
            command_epoch=controller._command_epoch,
            verified_command_epoch=controller._command_epoch,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        stopped = controller.tick(
            skipped, sim_time_s=3.0e-6, source_cursor_permit=False
        )
        self.assertEqual(stopped.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertIn("exact issued-step+1", stopped.reason)

    def test_feedback_baseline_rejects_stale_command_epoch_or_readback(self) -> None:
        payload = self._s10_air_candidate()
        controller, _initial = self._production_s10(payload)
        controller._command_epoch = 1
        controller._last_decision_command_epoch = 1
        stale = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=0,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        stopped = controller.tick(
            stale, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertEqual(stopped.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertIn("current command epoch/full 8+4", stopped.reason)

    def test_first_top_load_holds_for_dwell_before_another_increment(self) -> None:
        first_contact = _top_legs(_observation_mapping(), "FL", "FR", "RR", "RL")
        first_contact.update(
            body_crossed_front_face=True,
            final_recoverable=True,
            posture_complete=False,
            test_contact_dwell_s={
                leg: 0.05 if leg == "FR" else 0.5
                for leg in ("FL", "FR", "RL", "RR")
            },
            test_contact_dwell_verified={
                leg: leg != "FR" for leg in ("FL", "FR", "RL", "RR")
            },
        )
        first_contact = _attach_strict_evidence(first_contact)
        controller, _initial = self._production_s10(first_contact)
        controller._feedback_active_leg = "FR"
        controller._feedback_reference_targets = dict(controller._current_servo_targets)
        controller._feedback_configuration_sha256 = "f" * 64
        controller._feedback_selected_joint = "front_right_hip"
        controller._feedback_selected_sign = 1
        controller._feedback_selected_signs[
            ("FR", "front_right_hip", controller._feedback_configuration_sha256)
        ] = 1
        controller._feedback_stage = FeedbackRecoveryStage.INCREMENT
        before_epoch = controller._command_epoch
        before_targets = dict(controller._current_servo_targets)
        current = _current_test_observation(
            first_contact,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=before_epoch,
            verified_command_epoch=None,
            readback_targets=before_targets,
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        held = controller.tick(current, sim_time_s=1.0e-6, source_cursor_permit=True)
        self.assertFalse(held.command_changed)
        self.assertEqual(controller._command_epoch, before_epoch)
        self.assertEqual(dict(held.servo_targets_deg), before_targets)
        self.assertEqual(
            controller.status["feedback_recovery_stage"],
            FeedbackRecoveryStage.CONTACT_DWELL.value,
        )

    def test_feedback_action_bound_is_preflighted_without_target_mutation(self) -> None:
        payload = self._s10_air_candidate()
        controller, initial = self._production_s10(payload)
        controller._begin_recovery_leg(initial, "FR")
        controller._feedback_action_count = int(
            FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_feedback_actions"]
        )
        before_epoch = controller._command_epoch
        before_targets = dict(controller._current_servo_targets)
        current = _current_test_observation(
            payload,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=before_epoch,
            verified_command_epoch=None,
            readback_targets=before_targets,
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        decision = controller.tick(
            current, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertFalse(decision.command_changed)
        self.assertEqual(controller._command_epoch, before_epoch)
        self.assertEqual(dict(decision.servo_targets_deg), before_targets)
        self.assertEqual(
            controller.status["feedback_recovery_stage"],
            FeedbackRecoveryStage.SETTLE.value,
        )

    def test_feedback_recovery_safety_breach_safe_stops(self) -> None:
        payload = self._s10_air_candidate()
        controller, _initial = self._production_s10(payload)
        unsafe = dict(payload)
        unsafe["base_roll_rad"] = math.radians(31.0)
        unsafe = _attach_strict_evidence(unsafe)
        current = _current_test_observation(
            unsafe,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=controller._command_epoch,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        stopped = controller.tick(
            current, sim_time_s=1.0e-6, source_cursor_permit=True
        )
        self.assertEqual(stopped.terminal_outcome, MacroTerminalOutcome.SAFE_STOP)
        self.assertIn("attitude safety bound", stopped.reason)

    def test_four_top_contact_dwell_and_settle_reaches_full_success(self) -> None:
        complete = _top_legs(_observation_mapping(), "FL", "FR", "RR", "RL")
        complete.update(
            body_crossed_front_face=True,
            final_recoverable=True,
            posture_complete=True,
        )
        complete = _attach_strict_evidence(complete)
        controller, _initial = self._production_s10(complete)
        settling = _current_test_observation(
            complete,
            sim_time_s=1.0e-6,
            sim_step=1,
            command_epoch=controller._command_epoch,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        first = controller.tick(
            settling, sim_time_s=1.0e-6, source_cursor_permit=False
        )
        self.assertFalse(first.terminal)
        settled = _current_test_observation(
            complete,
            sim_time_s=0.251001,
            sim_step=2,
            command_epoch=controller._command_epoch,
            verified_command_epoch=None,
            readback_targets=dict(controller._current_servo_targets),
            readback_wheel_targets=dict(controller._current_wheel_targets),
        )
        terminal = controller.tick(
            settled, sim_time_s=0.251001, source_cursor_permit=False
        )
        self.assertEqual(terminal.macro_state, MacroStateId.SUCCESS)
        self.assertEqual(terminal.terminal_outcome, MacroTerminalOutcome.TASK_SUCCESS)

    def test_feedback_path_reaches_posture_incomplete_success(self) -> None:
        controller = self._controller()
        supported = _top_legs(_observation_mapping(), "FL", "RR", "RL")
        supported.update(
            body_crossed_front_face=True,
            final_recoverable=True,
            posture_complete=False,
        )
        supported = _attach_strict_evidence(supported)
        observation = MacroObservation.from_mapping(supported)
        controller.reset(observation, sim_time_s=0.0)
        controller._enter_state(
            MacroStateId.S10_POSTURE_RECOVERY,
            observation,
            0.0,
        )
        controller._active_profile = None
        source_count = int(controller.status["consumed_source_action_count"])

        settling = controller.tick(supported, sim_time_s=0.01)
        self.assertEqual(settling.macro_state, MacroStateId.S10_POSTURE_RECOVERY)
        self.assertEqual(
            controller.status["feedback_recovery_stage"],
            FeedbackRecoveryStage.SETTLE.value,
        )
        terminal = controller.tick(supported, sim_time_s=0.27)
        self.assertEqual(terminal.macro_state, MacroStateId.SUCCESS)
        self.assertEqual(
            terminal.terminal_outcome,
            MacroTerminalOutcome.TASK_SUCCESS_POSTURE_INCOMPLETE,
        )
        self.assertTrue(terminal.terminal)
        self.assertEqual(
            controller.status["consumed_source_action_count"], source_count
        )


class FeedbackConfigurationFreezeTests(unittest.TestCase):
    """Hermetic regression for the worker-reconstructible S10 configuration."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_default_macro_graph()
        cls.library = _hermetic_feedback_library()

    _controller = MacroControllerTests._controller
    _production_s10 = MacroControllerTests._production_s10
    _s10_air_candidate = staticmethod(MacroControllerTests._s10_air_candidate)
    _worker_reconstructed_configuration_sha256 = staticmethod(
        MacroControllerTests._worker_reconstructed_configuration_sha256
    )
    test_safe_first_probe_freezes_worker_reconstructible_dispatch_configuration = (
        MacroControllerTests._assert_safe_first_probe_freezes_worker_reconstructible_dispatch_configuration
    )
    test_unsafe_first_sign_skip_freezes_configuration_at_first_actual_dispatch = (
        MacroControllerTests._assert_unsafe_first_sign_skip_freezes_configuration_at_first_actual_dispatch
    )
    test_feedback_provenance_exact_pair_leg_joint_and_composite_binding = (
        MacroControllerTests.test_feedback_provenance_exact_pair_leg_joint_and_composite_binding
    )
    test_feedback_probe_return_negative_probe_is_permit_gated_and_n_plus_one_bound = (
        MacroControllerTests.test_feedback_probe_return_negative_probe_is_permit_gated_and_n_plus_one_bound
    )
    test_feedback_skipped_n_plus_one_fails_closed = (
        MacroControllerTests.test_feedback_skipped_n_plus_one_fails_closed
    )
    test_feedback_baseline_rejects_stale_command_epoch_or_readback = (
        MacroControllerTests.test_feedback_baseline_rejects_stale_command_epoch_or_readback
    )
    test_first_top_load_holds_for_dwell_before_another_increment = (
        MacroControllerTests.test_first_top_load_holds_for_dwell_before_another_increment
    )
    test_feedback_action_bound_is_preflighted_without_target_mutation = (
        MacroControllerTests.test_feedback_action_bound_is_preflighted_without_target_mutation
    )
    test_feedback_recovery_safety_breach_safe_stops = (
        MacroControllerTests.test_feedback_recovery_safety_breach_safe_stops
    )
    test_four_top_contact_dwell_and_settle_reaches_full_success = (
        MacroControllerTests.test_four_top_contact_dwell_and_settle_reaches_full_success
    )
    test_feedback_path_reaches_posture_incomplete_success = (
        MacroControllerTests.test_feedback_path_reaches_posture_incomplete_success
    )


if __name__ == "__main__":
    unittest.main()
