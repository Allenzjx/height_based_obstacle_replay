from __future__ import annotations

import hashlib
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from completion_aware_segment import (
    CompletionAwareSegmentExecutor,
    SegmentCompletionSpec,
    SegmentFeedback,
)
from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
    MacroFSMController,
    MacroObservation,
    MacroSegmentCompletionToken,
    MacroTerminalOutcome,
    SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
    build_gate_c_bundle,
)
from fsm_50mm_recording_derived_v3.fsm50_macro_state_model import (
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
    return payload


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
    x, y, z = centers[leg]
    centers[leg] = (0.6 if crossed else 0.2, y, z)
    result["wheel_center_w_m"] = centers
    return result


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
    return {
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
    }


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

    def test_s5_accepts_only_fresh_predecessor_projected_displacement(self) -> None:
        controller = self._controller()
        start = _observation_mapping(base=(0.0, 0.0, 0.1))
        controller.reset(start, sim_time_s=0.0)
        controller._state_id = MacroStateId.S4_FRONT_PAIR_ADVANCE
        controller._entry_body_position = (0.0, 0.0, 0.1)
        moved = _observation_mapping(base=(0.06, 0.03, 0.1))
        moved_observation = MacroObservation.from_mapping(moved)
        controller._record_boundary_carry(
            MacroStateId.S4_FRONT_PAIR_ADVANCE,
            MacroStateId.S5_PRE_RR_COM_SHIFT,
            moved_observation,
        )
        controller._enter_state(
            MacroStateId.S5_PRE_RR_COM_SHIFT, moved_observation, 1.0
        )
        fresh = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S5_PRE_RR_COM_SHIFT),
            moved_observation,
            timeline_complete=True,
        )
        self.assertTrue(fresh.satisfied)
        self.assertTrue(fresh.evidence["inherited_displacement_fresh"])
        self.assertGreater(
            fresh.evidence["inherited_target_direction_displacement_m"], 0.003
        )

        controller._boundary_from_state = MacroStateId.S3_FL_TRAVERSE
        stale = controller._evaluate_guard(
            controller.graph.get(MacroStateId.S5_PRE_RR_COM_SHIFT),
            moved_observation,
            timeline_complete=True,
        )
        self.assertFalse(stale.satisfied)

    def test_four_success_held_telemetry_shadow_is_causal_and_nonterminal_safe(self) -> None:
        graph = build_default_macro_graph()
        library = build_profile_library(PROJECT_ROOT)
        expected_action_counts = {"v003": 112, "v008": 119, "v009": 132, "v010": 142}
        expected_same_target_counts = {"v003": 3, "v008": 3, "v009": 4, "v010": 2}
        traversal_states = {
            MacroStateId.S2_FR_TRAVERSE,
            MacroStateId.S3_FL_TRAVERSE,
            MacroStateId.S6_RR_TRAVERSE,
            MacroStateId.S8_RL_COM_SHIFT_AND_TRAVERSE,
        }
        owner_by_key = {
            (item.source_version, item.state_id): item
            for item in library.segment_ownership
        }
        for source in library.successful_sources:
            with self.subTest(source=source.source_version):
                telemetry_path = source.gate_a_run_dir / "minimal_telemetry.jsonl"
                rows = [
                    json.loads(line)
                    for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                first_time = float(rows[0]["sim_time_s"])
                relative_times = [float(row["sim_time_s"]) - first_time for row in rows]
                controller = MacroFSMController(
                    graph,
                    library,
                    primary_source_version=DEFAULT_PRIMARY_VERSION,
                )
                reset = controller.reset(
                    _clean_gate_a_observation(rows[0]),
                    sim_time_s=0.0,
                    source_version=source.source_version,
                )
                previous_decision = reset
                source_action_ledger: list[dict[str, object]] = []
                source_action_steps: set[int] = set()
                physical_batch_steps: set[int] = set()
                coalesced_transition_starts: list[
                    tuple[MacroStateId, MacroStateId, int]
                ] = []
                same_target_source_segments: list[int] = []
                row_index = 0
                transitions: dict[MacroStateId, tuple[float, int, str]] = {}
                state_entry_s = {MacroStateId.S0_INITIALIZE: 0.0}
                physical_evidence_s: dict[MacroStateId, float] = {}
                hold_enter_s: dict[MacroStateId, float] = {}
                hold_snapshots: dict[
                    MacroStateId,
                    tuple[set[tuple[object, ...]], dict[str, float]],
                ] = {}
                active_start_control: dict[str, object] | None = None
                active_start_step = 0
                active_wheel_stop_acked = False
                profile_completion_s: dict[MacroStateId, float] = {}
                decision = None
                end_step = math.ceil((relative_times[-1] + 10.0) * 120.0)
                for step in range(1, end_step + 1):
                    now = step / 120.0
                    while (
                        row_index + 1 < len(rows)
                        and relative_times[row_index + 1] <= now + 1.0e-12
                    ):
                        row_index += 1
                    active_before = MacroStateId(controller.status["macro_state"])
                    observation = MacroObservation.from_mapping(
                        _clean_gate_a_observation(rows[row_index])
                    )
                    cursor_permit = step % 8 == 0
                    completion_token = None
                    generated_kind = ""
                    if cursor_permit and active_start_control is not None:
                        next_is_source_stop = False
                        if (
                            controller._active_profile is not None
                            and controller._visible_cursor
                            < len(controller._visible_keyframe_indices)
                            and not active_wheel_stop_acked
                        ):
                            next_frame = controller._active_profile.keyframes[
                                controller._visible_keyframe_indices[
                                    controller._visible_cursor
                                ]
                            ]
                            next_is_source_stop = bool(
                                next_frame.source_segment_index
                                == active_start_control["source_segment_index"]
                                and next_frame.dispatch_kind
                                == "wheel_channel_completion_stop"
                            )
                        production_cursor = int(rows[row_index]["segment_cursor"])
                        if next_is_source_stop:
                            generated_kind = "WHEEL_STOP_DUE"
                        elif production_cursor > int(
                            active_start_control["source_segment_index"]
                        ):
                            generated_kind = "COMPLETE"
                        else:
                            generated_kind = "WAIT"
                        helper_decision = _completion_decision_mapping(
                            active_start_control,
                            kind=generated_kind,
                            sim_time_s=now,
                            sim_step=step,
                            wheel_stop_acknowledged=active_wheel_stop_acked,
                        )
                        completion_token = (
                            MacroSegmentCompletionToken.from_control_decision(
                                active_start_control,
                                helper_decision,
                                start_sim_step=active_start_step,
                                start_readback_sha256="b" * 64,
                            )
                        )
                        if generated_kind == "COMPLETE":
                            owner = owner_by_key.get(
                                (
                                    source.source_version,
                                    MacroStateId(
                                        active_start_control["owner_state"]
                                    ),
                                )
                            )
                            if (
                                owner is not None
                                and int(
                                    active_start_control[
                                        "source_segment_index"
                                    ]
                                )
                                == owner.last_segment
                            ):
                                profile_completion_s[owner.state_id] = now
                    decision = controller.tick(
                        observation,
                        sim_time_s=now,
                        segment_completion_token=completion_token,
                        source_cursor_permit=cursor_permit,
                    )
                    completion_control = dict(
                        decision.segment_completion_control
                    )
                    if completion_control["kind"] == "START":
                        active_start_control = completion_control
                        active_start_step = step
                        active_wheel_stop_acked = False
                    elif completion_control["kind"] == "WHEEL_STOP":
                        active_wheel_stop_acked = True
                    elif generated_kind == "COMPLETE":
                        active_start_control = None
                        active_wheel_stop_acked = False
                    prior_decision = previous_decision
                    exact_map_changed = bool(
                        dict(decision.servo_targets_deg)
                        != dict(previous_decision.servo_targets_deg)
                        or dict(decision.wheel_targets_rad_s)
                        != dict(previous_decision.wheel_targets_rad_s)
                    )
                    self.assertEqual(decision.target_changed, exact_map_changed)
                    self.assertEqual(decision.command_changed, exact_map_changed)
                    self.assertEqual(
                        decision.command_epoch - previous_decision.command_epoch,
                        1 if exact_map_changed else 0,
                    )
                    if exact_map_changed:
                        self.assertNotIn(step, physical_batch_steps)
                        physical_batch_steps.add(step)
                    provenance = dict(decision.command_provenance)
                    if decision.source_action_consumed:
                        self.assertTrue(cursor_permit)
                        self.assertEqual(step % 8, 0)
                        self.assertNotIn(step, source_action_steps)
                        source_action_steps.add(step)
                        self.assertEqual(provenance["kind"], "SOURCE_ACTION")
                        source_action_ledger.append(provenance)
                        if not decision.target_changed:
                            same_target_source_segments.append(
                                int(provenance["source_segment_index"])
                            )
                    elif exact_map_changed:
                        self.assertIn(
                            provenance["kind"],
                            {
                                "BOUNDARY_ZERO_WHEELS",
                                "HOLD_ZERO_WHEELS",
                                "SAFE_STOP_ZERO_WHEELS",
                                "SUCCESS_ZERO_WHEELS",
                            },
                        )
                    else:
                        self.assertEqual(provenance["kind"], "NONE")
                        self.assertEqual(provenance["source_action_identity"], "")
                    previous_decision = decision
                    active_spec = graph.get(active_before)
                    if (
                        active_before not in physical_evidence_s
                        and _physical_evidence_ready(
                            active_spec,
                            dict(decision.guard_evidence),
                            observation,
                        )
                    ):
                        physical_evidence_s[active_before] = now
                    for event in decision.transition_events:
                        if event.startswith("HOLD:"):
                            held_state = MacroStateId(event.split(":", 1)[1])
                            hold_enter_s[held_state] = now
                            hold_snapshots[held_state] = (
                                set(controller._consumed_source_actions),
                                dict(decision.servo_targets_deg),
                            )
                    if active_before in hold_snapshots:
                        held_actions, held_servos = hold_snapshots[active_before]
                        self.assertEqual(
                            controller._consumed_source_actions, held_actions
                        )
                        self.assertEqual(decision.servo_targets_deg, held_servos)
                        self.assertTrue(
                            all(
                                abs(value) <= 1.0e-12
                                for value in decision.wheel_targets_rad_s.values()
                            )
                        )
                    exits = [
                        event.split(":", 1)[1]
                        for event in decision.transition_events
                        if event.startswith("EXIT:")
                    ]
                    enters = [
                        event.split(":", 1)[1]
                        for event in decision.transition_events
                        if event.startswith("ENTER:")
                    ]
                    if exits and decision.source_action_consumed:
                        self.assertEqual(generated_kind, "COMPLETE")
                        self.assertEqual(len(exits), 1)
                        self.assertEqual(len(enters), 1)
                        self.assertTrue(
                            all(
                                abs(value) <= 1.0e-12
                                for value in prior_decision.wheel_targets_rad_s.values()
                            )
                        )
                        self.assertEqual(
                            decision.segment_completion_control["kind"], "START"
                        )
                        self.assertEqual(
                            decision.segment_completion_control["owner_state"],
                            enters[0],
                        )
                        coalesced_transition_starts.append(
                            (
                                MacroStateId(exits[0]),
                                MacroStateId(enters[0]),
                                int(provenance["source_segment_index"]),
                            )
                        )
                    elif exits:
                        self.assertTrue(
                            all(
                                abs(value) <= 1.0e-12
                                for value in decision.wheel_targets_rad_s.values()
                            )
                        )
                    for exited_value in exits:
                        exited = MacroStateId(exited_value)
                        cursor = int(rows[row_index]["segment_cursor"])
                        transitions[exited] = (
                            now,
                            cursor,
                            str(rows[row_index]["wheel_contact_classes"].get(
                                graph.get(exited).active_leg, ""
                            )),
                        )
                        owner = owner_by_key.get((source.source_version, exited))
                        if owner is not None:
                            self.assertIn(
                                (
                                    source.source_version,
                                    owner.last_segment,
                                    "segment_start",
                                ),
                                controller._consumed_source_coordinates,
                            )
                            if exited in traversal_states:
                                self.assertGreaterEqual(
                                    cursor, owner.last_segment + 1
                                )
                    for event in decision.transition_events:
                        if event.startswith("ENTER:"):
                            entered = MacroStateId(event.split(":", 1)[1])
                            state_entry_s[entered] = now
                    if decision.terminal:
                        break
                self.assertIsNotNone(decision)
                self.assertNotEqual(
                    decision.terminal_outcome,
                    MacroTerminalOutcome.SAFE_STOP,
                    msg=(
                        f"{source.source_version}: {decision.reason}; "
                        f"phase_elapsed={decision.phase_elapsed_s}; "
                        f"guard={decision.guard_evidence}; transitions={transitions}"
                    ),
                )
                self.assertIn(
                    decision.terminal_outcome,
                    {
                        MacroTerminalOutcome.TASK_SUCCESS,
                        MacroTerminalOutcome.TASK_SUCCESS_POSTURE_INCOMPLETE,
                    },
                )
                prefix = source.source_version.split("_", 1)[0]
                expected_frames = sorted(
                    (
                        frame
                        for profile in library.profiles
                        if profile.source_version == source.source_version
                        for frame in profile.keyframes
                    ),
                    key=lambda frame: (
                        frame.source_segment_index,
                        frame.dispatch_kind,
                    ),
                )
                self.assertEqual(len(expected_frames), expected_action_counts[prefix])
                self.assertEqual(len(source_action_ledger), len(expected_frames))
                self.assertEqual(len(source_action_steps), len(source_action_ledger))
                self.assertEqual(
                    len(physical_batch_steps),
                    decision.command_epoch - reset.command_epoch,
                )
                for old_state, new_state, _ in coalesced_transition_starts:
                    self.assertEqual(graph.get(old_state).next_state, new_state)
                actual_coordinates = [
                    (
                        item["source_version"],
                        item["source_segment_index"],
                        item["dispatch_kind"],
                    )
                    for item in source_action_ledger
                ]
                expected_coordinates = [
                    (
                        source.source_version,
                        frame.source_segment_index,
                        frame.dispatch_kind,
                    )
                    for frame in expected_frames
                ]
                self.assertEqual(actual_coordinates, expected_coordinates)
                self.assertEqual(len(set(actual_coordinates)), len(actual_coordinates))
                for item, frame in zip(source_action_ledger, expected_frames):
                    self.assertEqual(item["source_step_index"], frame.source_step_index)
                    self.assertEqual(item["source_time_s"], frame.source_time_s)
                    self.assertEqual(
                        tuple(item["source_event_indices"]), frame.source_event_indices
                    )
                    self.assertEqual(tuple(item["commands"]), frame.commands)
                    self.assertEqual(item["sequence_index"], frame.sequence_index)
                    identity_payload = {
                        "schema_version": SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
                        "source_version": item["source_version"],
                        "source_segment_index": item["source_segment_index"],
                        "source_step_index": item["source_step_index"],
                        "source_time_s": item["source_time_s"],
                        "source_event_indices": list(item["source_event_indices"]),
                        "commands": list(item["commands"]),
                        "dispatch_kind": item["dispatch_kind"],
                        "sequence_index": item["sequence_index"],
                    }
                    expected_identity = hashlib.sha256(
                        json.dumps(
                            identity_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    self.assertEqual(
                        item["source_action_identity"], expected_identity
                    )
                self.assertEqual(
                    len(same_target_source_segments), expected_same_target_counts[prefix]
                )
                if prefix == "v003":
                    self.assertEqual(same_target_source_segments, [7, 41, 57])
                    self.assertEqual(
                        len(source_action_ledger) - len(same_target_source_segments),
                        109,
                    )
                self.assertTrue(traversal_states.issubset(transitions))
                for state_id, (transition_s, _, _) in transitions.items():
                    if state_id in {
                        MacroStateId.SUCCESS,
                        MacroStateId.SAFE_STOP,
                    }:
                        continue
                    evidence_s = physical_evidence_s[state_id]
                    completion_s = profile_completion_s.get(
                        state_id, state_entry_s[state_id]
                    )
                    if evidence_s > completion_s + 2.0 / 120.0 + 1.0e-9:
                        self.assertIn(state_id, hold_enter_s)
                        self.assertGreaterEqual(
                            hold_enter_s[state_id], completion_s
                        )
                    self.assertGreaterEqual(
                        transition_s + 1.0 / 120.0,
                        max(completion_s, evidence_s),
                    )
                if source.source_version.startswith("v003_"):
                    self.assertGreaterEqual(
                        transitions[MacroStateId.S3_FL_TRAVERSE][1], 41
                    )

    def test_feedback_path_reaches_posture_incomplete_success(self) -> None:
        controller = self._controller()
        p = _observation_mapping()
        controller.reset(p, sim_time_s=0.0)
        controller.tick(p, sim_time_s=0.01)
        controller.tick(p, sim_time_s=0.02)
        shifted = _observation_mapping(base=(-0.01, 0.01, 0.1))
        controller.tick(shifted, sim_time_s=0.03)
        fr_air = _with_leg(shifted, "FR", "AIR")
        controller.tick(fr_air, sim_time_s=0.04)
        front = _top_legs(shifted, "FR")
        controller.tick(front, sim_time_s=0.05)
        controller.tick(_with_leg(front, "FL", "AIR"), sim_time_s=0.06)
        front = _top_legs(front, "FR", "FL")
        controller.tick(front, sim_time_s=0.07)
        controller.tick(front, sim_time_s=0.08)
        approach = dict(front)
        face = dict(approach["wheel_front_face_clearance_m"])  # type: ignore[arg-type]
        face["RR"] = -0.05
        approach["wheel_front_face_clearance_m"] = face
        controller.tick(approach, sim_time_s=0.09)
        rr_air = _with_leg(approach, "RR", "AIR")
        controller.tick(rr_air, sim_time_s=0.10)
        controller.tick(rr_air, sim_time_s=0.11)
        controller.tick(rr_air, sim_time_s=0.12)
        three_top = _top_legs(front, "FR", "FL", "RR")
        controller.tick(three_top, sim_time_s=0.13)
        controller.tick(three_top, sim_time_s=0.14)
        controller.tick(_with_leg(three_top, "RL", "AIR"), sim_time_s=0.15)
        all_top = _top_legs(three_top, "RL")
        controller.tick(all_top, sim_time_s=0.16)
        crossed = dict(all_top)
        crossed["body_crossed_front_face"] = True
        controller.tick(crossed, sim_time_s=0.17)
        controller.tick(crossed, sim_time_s=0.18)
        recoverable = dict(crossed)
        recoverable["final_recoverable"] = True
        controller.tick(recoverable, sim_time_s=0.19)
        terminal = controller.tick(recoverable, sim_time_s=0.20)
        self.assertEqual(terminal.macro_state, MacroStateId.SUCCESS)
        self.assertEqual(
            terminal.terminal_outcome,
            MacroTerminalOutcome.TASK_SUCCESS_POSTURE_INCOMPLETE,
        )
        self.assertTrue(terminal.terminal)


if __name__ == "__main__":
    unittest.main()
