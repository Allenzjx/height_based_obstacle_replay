from __future__ import annotations

import argparse
import copy
import json
import math
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from command_model import JOINT_COMMAND_SIGN, SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from completion_aware_segment import SegmentDecision, SegmentDecisionKind
from fsm_50mm_recording_derived_v3 import fsm50_macro_runner as runner
from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
    FeedbackRecoveryObservation,
    build_gate_c_bundle,
)
from fsm_50mm_recording_derived_v3.fsm50_centroidal_support import (
    CentroidalAngularMomentumRateMeasurement,
    CentroidalSupportEvidence,
    WholeBodyCOMMeasurement,
    WheelContactMeasurement,
    assess_contact_wrench_feasibility,
    assess_primary_diagonal_support,
    assess_support_region,
    validate_wheel_contact_frame,
)
from fsm_50mm_recording_derived_v3.fsm50_motion_profiles import DEFAULT_PRIMARY_VERSION
from fsm_50mm_recording_derived_v3.fsm50_macro_state_model import (
    build_default_macro_graph,
)
from fsm_50mm_recording_derived_v3.worker_macro_fsm_session import (
    WorkerMacroFSMSession,
    _source_action_identity,
    load_worker_macro_fsm_request,
)


_FAKE_PLAN_SHA256 = "2" * 64
_FAKE_PLAN_PAYLOAD_SHA256 = "3" * 64
_FAKE_ACCEPTED_STEPS_SHA256 = "4" * 64
_FAKE_PROFILE_ID = "fake-segment-profile"
_FAKE_OWNER_STATE = "S1_APPROACH_AND_PRE_FR_SHIFT"
_FAKE_STEP_ID = "fake-step-000"


def _synthetic_terminal_telemetry_row(
    step: int, *, terminal: bool, source_version: str
) -> dict:
    """Build a physically closed all-four-TOP telemetry sample."""

    dt = 1.0 / 120.0
    sim_time = step * dt
    legs = ("FL", "FR", "RL", "RR")
    centers = {
        "FL": (0.70, 0.18, 0.10),
        "FR": (0.70, -0.18, 0.10),
        "RL": (0.50, 0.18, 0.10),
        "RR": (0.50, -0.18, 0.10),
    }
    com = WholeBodyCOMMeasurement(
        com_measurement_available=True,
        acceleration_available=True,
        physics_tick=step,
        sim_time_s=sim_time,
        physics_dt_s=dt,
        body_names=("synthetic_body",),
        body_masses_kg=(10.0,),
        total_mass_kg=10.0,
        position_w_m=(0.60, 0.0, 0.20),
        velocity_w_m_s=(0.0, 0.0, 0.0),
        acceleration_w_m_s2=(0.0, 0.0, 0.0),
        source="runner_test.synthetic_whole_body_com",
    )
    angular = CentroidalAngularMomentumRateMeasurement(
        available=True,
        physics_tick=step,
        sim_time_s=sim_time,
        body_names=("synthetic_body",),
        angular_momentum_rate_w_nm=(0.0, 0.0, 0.0),
        source="runner_test.synthetic_angular_rate",
        errors=(),
    )
    measurements = []
    for leg in legs:
        point = centers[leg]
        measurements.append(
            WheelContactMeasurement(
                leg=leg,
                wheel_body_name={
                    "FL": "front_left_wheel",
                    "FR": "front_right_wheel",
                    "RL": "rear_left_wheel",
                    "RR": "rear_right_wheel",
                }[leg],
                physics_tick=step,
                sim_time_s=sim_time,
                surface_kind="OBSTACLE_TOP",
                surface_height_m=0.10,
                surface_normal_w=(0.0, 0.0, 1.0),
                active=True,
                contact_point_w_m=point,
                normal_force_w_n=(0.0, 0.0, 24.525),
                friction_force_w_n=(0.0, 0.0, 0.0),
                contact_moment_w_nm=(0.0, 0.0, 0.0),
                contact_moment_model="MEASURED",
                dwell_s=0.50,
                surface_dwell_verified=True,
                slip_speed_m_s=0.0,
                contact_drift_speed_m_s=0.0,
                friction_coefficient=0.8,
                finite_patch_radius_m=0.03,
                source="runner_test.synthetic_contact",
            )
        )
    contacts = validate_wheel_contact_frame(
        measurements,
        physics_tick=step,
        sim_time_s=sim_time,
        physics_dt_s=dt,
    )
    support = assess_support_region(com, contacts)
    wrench = assess_contact_wrench_feasibility(
        com, contacts, angular_momentum_rate=angular
    )
    diagonal = assess_primary_diagonal_support(
        com,
        contacts,
        active_swing_leg="",
        wrench_feasibility=wrench,
    )
    centroidal = CentroidalSupportEvidence.create(
        sim_step=step,
        physics_time_s=sim_time,
        physics_dt_s=dt,
        whole_body_com=com,
        centroidal_angular_momentum_rate=angular,
        wheel_contacts=contacts,
        support_region=support,
        contact_wrench_feasibility=wrench,
        diagonal_support=diagonal,
    )
    servo_zero = {name: 0.0 for name in SERVO_JOINT_NAMES}
    wheel_zero = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    feedback = FeedbackRecoveryObservation.create(
        sim_step=step,
        physics_time_s=sim_time,
        observed_command_epoch=0,
        n_plus_one_verified=False,
        verified_command_epoch=None,
        readback_servo_targets_deg=servo_zero,
        readback_wheel_targets_rad_s=wheel_zero,
        measured_servo_positions_deg=servo_zero,
        measured_servo_velocities_deg_s=servo_zero,
        joint_limit_margin_deg={name: 20.0 for name in SERVO_JOINT_NAMES},
        base_position_m=(0.60, 0.0, 0.20),
        base_roll_rad=0.0,
        base_pitch_rad=0.0,
        base_angular_velocity_rad_s=(0.0, 0.0, 0.0),
        wheel_center_w_m=centers,
        wheel_front_face_clearance_m={leg: centers[leg][0] - 0.50 for leg in legs},
        wheel_top_clearance_m={leg: 0.0 for leg in legs},
        obstacle_front_face_x_m=0.50,
        obstacle_top_z_m=0.10,
        body_crossed_front_face=True,
        final_recoverable=True,
        posture_complete=True,
    )
    all_joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
    return {
        "schema_version": runner.TELEMETRY_SCHEMA,
        "source_version": source_version,
        "sim_step": step,
        "sim_time_s": sim_time,
        "centroidal_support_evidence": centroidal.to_mapping(),
        "feedback_recovery_observation": feedback.to_mapping(),
        "base_position_m": {"x": 0.60, "y": 0.0, "z": 0.20},
        "base_roll_rad": 0.0,
        "base_pitch_rad": 0.0,
        "root_angular_velocity_w": [0.0, 0.0, 0.0],
        "joint_position_rad": {name: 0.0 for name in all_joint_names},
        "joint_velocity_rad_s": {name: 0.0 for name in all_joint_names},
        "wheel_center_w_m": {leg: list(centers[leg]) for leg in legs},
        "obstacle_front_face_x_m": 0.50,
        "obstacle_top_z_m": 0.10,
        "body_crossed_front_face": True,
        "final_recoverable": True,
        "posture_complete": True,
        "macro_state": "SUCCESS" if terminal else "S10_POSTURE_RECOVERY",
        "subphase": "COMPLETE" if terminal else "SETTLE",
        "profile_id": "",
        "profile_source_version": "",
        "profile_strategy": "",
        "phase_elapsed_s": 0.0,
        "profile_fraction": 1.0,
        "command_epoch": 0,
        "transition_events": [],
        "controller_terminal": terminal,
        "controller_terminal_outcome": "TASK_SUCCESS" if terminal else "RUNNING",
        "feedback_recovery_stage": "COMPLETE" if terminal else "SETTLE",
        "feedback_recovery_action_count": 0,
        "feedback_recovery_exhaustion_reason": "",
    }


def _fake_completion_spec() -> dict:
    return {
        "segment_index": 0,
        "source_step": 0,
        "source_step_id": _FAKE_STEP_ID,
        "servo_targets_deg": {"front_left_hip": 0.0},
        "servo_duration_s": 0.0,
        "servo_tolerance_deg": 1.0,
        "recorded_servo_residual_deg": {"front_left_hip": 0.0},
        "legacy_missing_endpoint": False,
        "wheel_active_duration_s": 0.1,
        "explicit_hold_s": 0.0,
    }


def _fake_source_provenance(source_version: str) -> dict:
    row = {
        "kind": "SOURCE_ACTION",
        "source_action_identity": "",
        "source_version": source_version,
        "source_segment_index": 0,
        "source_step_index": 0,
        "source_time_s": 0.0,
        "source_event_indices": [0],
        "commands": ["servo front_left_hip 0"],
        "dispatch_kind": "segment_start",
        "sequence_index": 0,
    }
    row["source_action_identity"] = _source_action_identity(row)
    return row


def _fake_segment_binding(source_version: str) -> dict:
    return {
        "schema_version": "fsm50.playback_segment_binding.v1",
        "source_version": source_version,
        "source_plan_sha256": _FAKE_PLAN_SHA256,
        "source_plan_payload_sha256": _FAKE_PLAN_PAYLOAD_SHA256,
        "accepted_steps_sha256": _FAKE_ACCEPTED_STEPS_SHA256,
        "segment": {
            "segment_index": 0,
            "source_step": 0,
            "source_step_id": _FAKE_STEP_ID,
        },
        "events": [],
        "completion_spec": _fake_completion_spec(),
    }


def _synthetic_segment_completion_row(
    expected_action: dict,
    row_index: int,
    *,
    start_step: int | None = None,
    physical_start: bool = True,
) -> dict:
    provenance = dict(expected_action["command_provenance"])
    binding = dict(expected_action["segment_completion_binding"])
    spec = copy.deepcopy(dict(binding["completion_spec"]))
    start_step = 100 + row_index * 100000 if start_step is None else start_step
    minimum_duration_s = max(
        float(spec["servo_duration_s"]),
        float(spec["wheel_active_duration_s"]),
        float(spec["explicit_hold_s"]),
    )
    duration_steps = max(8, int(math.ceil(minimum_duration_s * 120.0 / 8.0)) * 8)
    terminal_step = start_step + duration_steps
    start_time = start_step / 120.0
    terminal_time = terminal_step / 120.0
    sparse_targets = dict(spec.get("servo_targets_deg", {}) or {})
    final = SegmentDecision(
        kind=SegmentDecisionKind.COMPLETE,
        segment_index=spec["segment_index"],
        source_step=spec["source_step"],
        source_step_id=spec["source_step_id"],
        sim_time_s=terminal_time,
        sim_step=terminal_step,
        segment_elapsed_s=terminal_time - start_time,
        servo_planned_done=True,
        reference_position_done=True,
        servo_done=True,
        servo_errors_deg={name: 0.0 for name in sparse_targets},
        servo_velocity_deg_s={name: 0.0 for name in sparse_targets},
        max_servo_error_deg=0.0,
        servo_tolerance_deg=spec["servo_tolerance_deg"],
        recorded_servo_residual_deg=dict(
            spec.get("recorded_servo_residual_deg", {}) or {}
        ),
        legacy_missing_endpoint=spec["legacy_missing_endpoint"],
        contact_candidate=False,
        contact_extension_s=0.0,
        contact_grace_done=True,
        contact_stable=True,
        contact_window_ready=False,
        contact_window_duration_s=0.0,
        contact_window_cap_deg=0.0,
        contact_window_min_deg=0.0,
        contact_window_max_deg=0.0,
        contact_window_slope_deg_s=0.0,
        divergence_window_duration_s=0.0,
        divergence_window_min_deg=0.0,
        divergence_window_max_deg=0.0,
        divergence_window_slope_deg_s=0.0,
        velocities_near_zero=True,
        wheel_elapsed_s=max(
            terminal_time - start_time,
            float(spec["wheel_active_duration_s"]),
        ),
        wheel_duration_s=spec["wheel_active_duration_s"],
        wheel_done=True,
        hold_elapsed_s=max(
            terminal_time - start_time,
            float(spec["explicit_hold_s"]),
        ),
        hold_duration_s=spec["explicit_hold_s"],
        hold_done=True,
        segment_done=True,
        wheel_stop_due=False,
        wheel_stop_acknowledged=False,
        stalled_for_s=0.0,
        worst_joint="",
        tracking_evidence={"supported": True, "converged": True},
        phase="COMPLETE",
        failure_reason="",
        failure_code="",
    ).to_mapping()
    final_sha = runner._sha256_mapping(final)
    tracking_calls = 1 if sparse_targets else 0
    readback_sha = runner._sha256_mapping(
        {
            "segment": row_index,
            "sim_step": start_step + 1,
            "kind": "synthetic-test-readback",
        }
    )
    return {
        "schema_version": runner.SEGMENT_COMPLETION_SCHEMA,
        "segment_completion_index": row_index,
        "source_version": provenance["source_version"],
        "profile_id": expected_action["profile_id"],
        "profile_source_version": expected_action["profile_source_version"],
        "owner_state": expected_action["owner_state"],
        "source_plan_sha256": expected_action["source_plan_sha256"],
        "source_plan_payload_sha256": binding["source_plan_payload_sha256"],
        "accepted_steps_sha256": binding["accepted_steps_sha256"],
        "source_segment_index": provenance["source_segment_index"],
        "source_step_index": provenance["source_step_index"],
        "source_step_id": spec["source_step_id"],
        "completion_spec": spec,
        "effective_completion_spec": spec,
        "dynamic_servo_duration_s": spec["servo_duration_s"],
        "effective_servo_reference_velocity_deg_s": 150.0,
        "pre_action_canonical_servo_targets_deg": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "start_source_action_identity": provenance["source_action_identity"],
        "source_action_consumption_index": expected_action["source_action_index"],
        "start_command_epoch": row_index + 1,
        "start_sim_step": start_step,
        "start_sim_time_s": start_time,
        "start_physical_dispatch": physical_start,
        "start_batch_id": (
            f"fake-batch-{row_index + 1}" if physical_start else ""
        ),
        "start_first_physics_step": (
            start_step + 1 if physical_start else None
        ),
        "start_readback_verified": True,
        "start_readback_verified_sim_step": (
            start_step + 1 if physical_start else start_step
        ),
        "start_readback_sha256": readback_sha,
        "retained_epoch_same_target": not physical_start,
        "tracking_begin_count": tracking_calls,
        "tracking_begin_sim_step": start_step,
        "tracking_begin_evidence": {
            "called": bool(sparse_targets),
            "sparse_joint_names": sorted(sparse_targets),
        },
        "tracking_end_count": tracking_calls,
        "tracking_end_attempt_count": tracking_calls,
        "tracking_lifecycle_closed": True,
        "tracking_end_sim_step": terminal_step,
        "tracking_end_sim_time_s": terminal_time,
        "tracking_end_reason": "COMPLETE",
        "tracking_end_evidence": (
            {"ended": True, "tracking_completion_deferred": False}
            if sparse_targets
            else {
                "ended": True,
                "required": False,
                "reason": "segment has no sparse servo endpoints",
            }
        ),
        "observation_decisions": [final],
        "last_decision": final,
        "last_decision_sha256": final_sha,
        "wheel_stop": None,
        "terminal_kind": "COMPLETE",
        "terminal_sim_step": terminal_step,
        "terminal_sim_time_s": terminal_time,
        "terminal_decision_sha256": final_sha,
    }


def _synthetic_retained_source_consumption_row(
    expected_action: dict,
    *,
    expected_action_count: int,
    source_action_index: int,
    sim_step: int,
    runtime_instance_id: str,
    boundary_batch_id: str,
) -> dict:
    """Build one canonical same-target source-consumption row for runner tests."""

    command_transform = {
        "schema_version": "fsm50.servo_command_transform.v1",
        "standing_pose_deg_by_servo": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "command_sign_by_servo": {
            name: float(JOINT_COMMAND_SIGN[name])
            for name in SERVO_JOINT_NAMES
        },
    }
    pre_action_readback = {
        "sim_step": sim_step,
        "command_epoch": 0,
        "batch_id": boundary_batch_id,
        "canonical_servo_targets_deg": dict(
            expected_action["servo_targets_deg"]
        ),
        "canonical_wheel_targets_rad_s": dict(
            expected_action["wheel_targets_rad_s"]
        ),
        "servo_command_transform": command_transform,
        "servo_command_transform_sha256": runner._sha256_mapping(
            command_transform
        ),
        "expected_servo_drive_targets_rad": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "actual_servo_drive_targets_rad": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "actual_wheel_drive_targets_rad_s": {
            name: 0.0 for name in WHEEL_JOINT_NAMES
        },
        "adapter_runtime_instance_id": runtime_instance_id,
        "root_state_write_count": 0,
        "physics_dt_s": 1.0 / 120.0,
    }
    readback_sha = runner._sha256_mapping(pre_action_readback)
    return {
        "schema_version": "fsm50.source_action_consumption.v1",
        "source_action_index": source_action_index,
        "expected_source_action_count": expected_action_count,
        "sim_time_s": sim_step / 120.0,
        "sim_step": sim_step,
        "macro_state": expected_action["owner_state"],
        "subphase": "",
        "profile_id": expected_action["profile_id"],
        "profile_source_version": expected_action["profile_source_version"],
        "profile_strategy": expected_action["profile_strategy"],
        "source_plan_sha256": expected_action["source_plan_sha256"],
        "profile_library_sha256": _FakeBundle.profile_library_sha256,
        "bundle_sha256": _FakeBundle.bundle_sha256,
        "command_provenance": dict(expected_action["command_provenance"]),
        "servo_targets_deg": dict(expected_action["servo_targets_deg"]),
        "wheel_targets_rad_s": dict(expected_action["wheel_targets_rad_s"]),
        "target_changed": False,
        # A retained same-target source action consumes source identity without
        # inventing a new physical/controller epoch.
        "dispatch_epoch": 0,
        "physical_dispatch_required": False,
        "physical_dispatch_applied": False,
        "physical_dispatch_index": None,
        "batch_id": "",
        "n_plus_one_verified": False,
        "n_plus_one_verified_sim_step": None,
        "n_plus_one_readback_sha256": "",
        "pre_action_verified_command_epoch": 0,
        "pre_action_verified_readback": pre_action_readback,
        "pre_action_verified_readback_sha256": readback_sha,
    }


def _valid_environment_identity() -> dict:
    return {
        "environment_lock_path": str(runner.DEFAULT_ENVIRONMENT_LOCK_PATH),
        "environment_lock_sha256": "8" * 64,
        "locked_source_file_count": 1,
        "required_source_file_count": 1,
        "source_closure_complete": True,
        "source_verification_sha256": "9" * 64,
    }


class _FakeGraph:
    graph_id = "fake-graph"
    _graph = build_default_macro_graph()
    initial_state = _graph.initial_state
    success_state = _graph.success_state

    def get(self, state):
        return self._graph.get(state)


class _FakeProfiles:
    library_id = "fake-profiles"

    def __init__(self, source_version: str):
        self.source_version = source_version

    def to_mapping(self):
        provenance = _fake_source_provenance(self.source_version)
        return {
            "profiles": [
                {
                    "profile_id": _FAKE_PROFILE_ID,
                    "source_version": self.source_version,
                    "state_id": _FAKE_OWNER_STATE,
                    "strategy": "PRIMARY_PROFILE",
                    "source_plan_sha256": _FAKE_PLAN_SHA256,
                    "source_segment_range": [0, 0],
                    "keyframes": [
                        {
                            **provenance,
                            "servo_targets_deg": {
                                name: 0.0 for name in SERVO_JOINT_NAMES
                            },
                            "wheel_targets_rad_s": {
                                name: 0.0 for name in WHEEL_JOINT_NAMES
                            },
                        }
                    ],
                    "segment_bindings": [
                        _fake_segment_binding(self.source_version)
                    ],
                }
            ],
            "segment_ownership": [
                {
                    "source_version": self.source_version,
                    "state_id": _FAKE_OWNER_STATE,
                    "first_segment": 0,
                    "last_segment": 0,
                    "source_plan_sha256": _FAKE_PLAN_SHA256,
                }
            ],
        }


class _FakeBundle:
    graph = _FakeGraph()
    graph_sha256 = "a" * 64
    profile_library_sha256 = "b" * 64
    bundle_sha256 = "c" * 64

    def __init__(self, primary_source_version=DEFAULT_PRIMARY_VERSION):
        self.primary_source_version = primary_source_version
        self.profiles = _FakeProfiles(primary_source_version)

    def to_mapping(self):
        return {
            "graph_id": self.graph.graph_id,
            "graph_sha256": self.graph_sha256,
            "profile_library_sha256": self.profile_library_sha256,
            "bundle_sha256": self.bundle_sha256,
        }


class _DifferentGraphBundle(_FakeBundle):
    graph = SimpleNamespace(graph_id="different-graph")
    graph_sha256 = "9" * 64


class _Lock:
    active = 0

    def __init__(self):
        self.acquired = False

    def acquire(self):
        assert _Lock.active == 0
        _Lock.active += 1
        self.acquired = True

    def release(self):
        if self.acquired:
            _Lock.active -= 1
            self.acquired = False


class _Process:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class _FakeClient:
    next_pid = 400
    instances = []

    def __init__(self, worker_args):
        if self.instances:
            assert self.instances[-1].process.poll() == 0
        type(self).next_pid += 1
        self.process = _Process(type(self).next_pid)
        self.pid = self.process.pid
        self.worker_args = worker_args
        self.request = runner.strict_json_load(worker_args.fsm50_macro_request_path)
        self.latest_macro_fsm_terminal = {}
        self._history = []
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        return None

    def poll(self):
        return None

    def status(self):
        request = self.request
        preflight = {
            key: request[key]
            for key in (
                "schema_version",
                "request_id",
                "source_version",
                "profile_id",
                "graph_id",
                "graph_sha256",
                "profile_library_sha256",
                "bundle_sha256",
                "height_mm",
                "run_dir",
                "trial_kind",
                "trial_index",
                "telemetry_hz",
                "video_fps",
                "filtered_contact_bank_enabled",
            )
        }
        preflight.update(
            enabled=True,
            execution_mode="normal_development",
            preflight_ok=True,
        )
        status = {
            "ready": True,
            "worker_pid": self.pid,
            "worker_session_id": f"worker-{self.pid}",
            "adapter_runtime_instance_id": f"adapter-{self.pid}",
            "root_state_write_count": 0,
            "runtime_version": "test-isaac-runtime-1",
            "macro_fsm_preflight_ready": True,
            "macro_fsm_request_id": request["request_id"],
            "worker_macro_fsm_preflight": preflight,
            "worker_macro_fsm_session": {
                "enabled": True,
                "execution_mode": "normal_development",
                "request_id": request["request_id"],
                "bundle_sha256": request["bundle_sha256"],
                "state": "ready_for_start",
                "filtered_contact_bank_enabled": True,
                "deployment_safety_evidence": {
                    "available": True,
                    "dangerous_body_collision": False,
                    "severe_penetration": False,
                    "source_version": request["source_version"],
                    "task_success_table_sha256": request[
                        "task_success_table_sha256"
                    ],
                    "alignment_sha256": request["alignment_sha256"],
                    "gate_a_row_sha256": "d" * 64,
                    "live_geometry_sha256": "e" * 64,
                    "initial_state_sha256": "f" * 64,
                    "deployment_binding_sha256": "1" * 64,
                },
            },
            "worker_artifact_session": {"enabled": False},
            "worker_artifact_preflight": {"enabled": False},
            "operation_ack_history": list(self._history),
        }
        if self._history:
            status["last_operation_ack"] = self._history[-1]
        if self.latest_macro_fsm_terminal:
            status["last_macro_fsm_terminal"] = dict(self.latest_macro_fsm_terminal)
        return status

    def start_macro_fsm(self, **identity):
        request = self.request
        assert identity["bundle_sha256"] == request["bundle_sha256"]
        run_dir = Path(request["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        worker_result = run_dir / "worker_macro_fsm_result.json"
        task_inputs = run_dir / "macro_task_inputs.json"
        source_consumption = (
            run_dir / "macro_source_action_consumption.jsonl"
        )
        segment_completion = run_dir / "macro_segment_completion_ledger.jsonl"
        feedback_recovery = (
            run_dir / "macro_feedback_recovery_action_ledger.jsonl"
        )
        dispatch_ledger = run_dir / "macro_dispatch_ledger.jsonl"
        transition_ledger = run_dir / "macro_transition_evidence.jsonl"
        telemetry_ledger = run_dir / "minimal_macro_telemetry.jsonl"
        video = run_dir / "actual_viewport_video.mp4"
        if request["bundle_sha256"] == _FakeBundle.bundle_sha256:
            ledger_bundle = _FakeBundle(request["source_version"])
        else:
            ledger_bundle = build_gate_c_bundle(
                runner.PROJECT_ROOT,
                alignment_path=request["alignment_path"],
                primary_source_version=request["source_version"],
            )
        expected_starts, _expected_stops = runner._expected_segment_start_actions(
            bundle=ledger_bundle,
            request=request,
            run_dir=run_dir,
        )
        segment_rows = [
            _synthetic_segment_completion_row(
                expected,
                index,
                start_step=109 + index * 100000,
                physical_start=False,
            )
            for index, expected in enumerate(expected_starts)
        ]
        source_rows = [
            _synthetic_retained_source_consumption_row(
                expected,
                expected_action_count=len(expected_starts),
                source_action_index=index,
                sim_step=109 + index * 100000,
                runtime_instance_id=f"adapter-{self.pid}",
                boundary_batch_id=f"{request['request_id']}:start-boundary",
            )
            for index, expected in enumerate(expected_starts)
        ]
        for segment, source in zip(segment_rows, source_rows):
            pre_action = source["pre_action_verified_readback"]
            segment.update(
                start_command_epoch=source["dispatch_epoch"],
                start_sim_step=source["sim_step"],
                start_sim_time_s=source["sim_time_s"],
                pre_action_canonical_servo_targets_deg=dict(
                    pre_action["canonical_servo_targets_deg"]
                ),
                start_readback_verified_sim_step=pre_action["sim_step"],
                start_readback_sha256=source[
                    "pre_action_verified_readback_sha256"
                ],
            )
        command_transform = dict(
            source_rows[0]["pre_action_verified_readback"][
                "servo_command_transform"
            ]
        )
        command_transform_sha256 = runner._sha256_mapping(command_transform)
        boundary_ack = {
            "batch_id": f"{request['request_id']}:start-boundary",
            "source": "fsm50_macro_start_boundary",
            "error": "",
            "applied_sim_step": 100,
            "first_physics_step": 101,
            "motion_start_skew_s": 0.0,
            "physics_dt_s": 1.0 / 120.0,
            "servo_applied": True,
            "wheel_applied": True,
            "servo_targets_applied": {
                name: 0.0 for name in SERVO_JOINT_NAMES
            },
            "wheel_targets_applied": {
                name: 0.0 for name in WHEEL_JOINT_NAMES
            },
            "recording_metadata": {},
        }
        boundary_readback = {
            "sim_step": 101,
            "command_epoch": 0,
            "batch_id": boundary_ack["batch_id"],
            "canonical_servo_targets_deg": dict(
                boundary_ack["servo_targets_applied"]
            ),
            "canonical_wheel_targets_rad_s": dict(
                boundary_ack["wheel_targets_applied"]
            ),
            "servo_command_transform": command_transform,
            "servo_command_transform_sha256": command_transform_sha256,
            "expected_servo_drive_targets_rad": {
                name: 0.0 for name in SERVO_JOINT_NAMES
            },
            "actual_servo_drive_targets_rad": {
                name: 0.0 for name in SERVO_JOINT_NAMES
            },
            "actual_wheel_drive_targets_rad_s": {
                name: 0.0 for name in WHEEL_JOINT_NAMES
            },
            "adapter_runtime_instance_id": f"adapter-{self.pid}",
            "root_state_write_count": 0,
            "physics_dt_s": 1.0 / 120.0,
        }
        boundary_claim = {
            "start_boundary_ack": boundary_ack,
            "start_boundary_readback": boundary_readback,
            "start_boundary_readback_sha256": runner._sha256_mapping(
                boundary_readback
            ),
        }
        source_consumption.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in source_rows
            ),
            encoding="utf-8",
        )
        source_sha256 = runner.sha256_file(source_consumption)
        segment_completion.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in segment_rows
            ),
            encoding="utf-8",
        )
        segment_sha256 = runner.sha256_file(segment_completion)
        feedback_recovery.write_text("", encoding="utf-8")
        feedback_recovery_sha256 = runner.sha256_file(feedback_recovery)
        dispatch_ledger.write_text("", encoding="utf-8")
        dispatch_ledger_sha256 = runner.sha256_file(dispatch_ledger)
        graph = _FakeGraph._graph
        ordered_states = []
        state = graph.initial_state
        while True:
            ordered_states.append(state.value)
            if state == graph.success_state:
                break
            state = graph.get(state).next_state
        owned_steps_by_state: dict[str, list[int]] = {
            state_id: [] for state_id in ordered_states
        }
        for row in source_rows:
            owned_steps_by_state.setdefault(str(row["macro_state"]), []).append(
                int(row["sim_step"])
            )
        for row in segment_rows:
            owned_steps_by_state.setdefault(str(row["owner_state"]), []).extend(
                (int(row["start_sim_step"]), int(row["terminal_sim_step"]))
            )
        profiles_by_state = {
            str(profile["state_id"]): profile
            for profile in ledger_bundle.profiles.to_mapping()["profiles"]
            if str(profile["source_version"]) == request["source_version"]
        }
        transition_steps = [int(boundary_ack["first_physics_step"])]
        for index in range(1, len(ordered_states)):
            prior_state = ordered_states[index - 1]
            target_state = ordered_states[index]
            prior_owned = owned_steps_by_state.get(prior_state, [])
            target_owned = owned_steps_by_state.get(target_state, [])
            minimum_step = transition_steps[-1] + 1
            if prior_owned:
                # Leave deterministic room for completion-control actions that
                # a ledger-tamper test injects after the original fake capture.
                minimum_step = max(minimum_step, max(prior_owned) + 32)
            if target_owned:
                minimum_step = min(minimum_step, min(target_owned))
            if minimum_step <= transition_steps[-1] or (
                prior_owned and minimum_step <= max(prior_owned)
            ):
                raise AssertionError(
                    "synthetic transition schedule cannot separate state owners"
                )
            transition_steps.append(minimum_step)
        transition_rows = []
        for index, to_state in enumerate(ordered_states):
            step = transition_steps[index]
            from_state = "" if index == 0 else ordered_states[index - 1]
            profile = profiles_by_state.get(to_state)
            profiled = profile is not None
            transition_rows.append(
                {
                    "schema_version": runner.TRANSITION_SCHEMA,
                    "transition_index": index,
                    "sim_time_s": step / 120.0,
                    "sim_step": step,
                    "from_state": from_state,
                    "to_state": to_state,
                    "subphase": (
                        "COMPLETE" if to_state == "SUCCESS" else "PRELOAD"
                    ),
                    "profile_id": str(profile["profile_id"]) if profiled else "",
                    "profile_source_version": (
                        str(profile["source_version"]) if profiled else ""
                    ),
                    "profile_strategy": (
                        str(profile["strategy"]) if profiled else ""
                    ),
                    "command_epoch": 0,
                    "phase_elapsed_s": 0.0,
                    "profile_fraction": (
                        0.0 if profiled else 1.0
                    ),
                    "events": (
                        [f"RESET:{to_state}"]
                        if index == 0
                        else [
                            f"EXIT:{from_state}",
                            f"ENTER:{to_state}",
                        ]
                    ),
                    "reason": "synthetic runner transition",
                    "guard_evidence": {},
                    "retry_count": 0,
                    "observation_sha256": f"{(index + 7) % 10}" * 64,
                }
            )
        transition_ledger.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in transition_rows
            ),
            encoding="utf-8",
        )
        transition_sha256 = runner.sha256_file(transition_ledger)
        telemetry_rows = [
            _synthetic_terminal_telemetry_row(
                step,
                terminal=True,
                source_version=request["source_version"],
            )
            for step in (
                transition_steps[-1] + 8,
                transition_steps[-1] + 16,
                transition_steps[-1] + 24,
                transition_steps[-1] + 32,
                transition_steps[-1] + 40,
            )
        ]
        telemetry_ledger.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for row in telemetry_rows
            ),
            encoding="utf-8",
        )
        telemetry_sha256 = runner.sha256_file(telemetry_ledger)
        final_raw_sha256 = runner._sha256_mapping(telemetry_rows[-1])
        terminal_auditor = WorkerMacroFSMSession(
            load_worker_macro_fsm_request(
                Path(self.worker_args.fsm50_macro_request_path)
            ),
            worker_session_id="synthetic-terminal-auditor",
        )
        terminal_auditor.rows = copy.deepcopy(telemetry_rows)
        terminal_auditor.durable_servo_command_transform = copy.deepcopy(
            command_transform
        )
        terminal_auditor.controller_terminal_outcome = "TASK_SUCCESS"
        settle_assessment = (
            terminal_auditor._feedback_recovery_terminal_settle_assessment(
                outcome="TASK_SUCCESS"
            )
        )
        assert settle_assessment["complete"] is True
        recovery_closure_core = {
            "schema_version": runner.TERMINAL_RECOVERY_CLOSURE_SCHEMA,
            "outcome": "TASK_SUCCESS",
            "telemetry_jsonl_path": str(telemetry_ledger),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "telemetry_sample_count": len(telemetry_rows),
            "final_raw_telemetry_row_sha256": final_raw_sha256,
            "settle_assessment": settle_assessment,
        }
        recovery_closure = {
            **recovery_closure_core,
            "closure_sha256": runner._sha256_mapping(recovery_closure_core),
        }
        transition_claim = {
            "transition_evidence_path": str(transition_ledger),
            "transition_evidence_sha256": transition_sha256,
            "transition_count": len(transition_rows),
        }
        telemetry_claim = {
            "telemetry_jsonl_path": str(telemetry_ledger),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "telemetry_sample_count": len(telemetry_rows),
            "final_raw_telemetry_row_sha256": final_raw_sha256,
        }
        source_claim = {
            "source_action_consumption_path": str(source_consumption),
            "source_action_consumption_sha256": source_sha256,
            "source_action_consumption_count": len(source_rows),
            "expected_source_action_count": len(source_rows),
            "source_action_coverage_complete": True,
            "source_action_coverage_errors": [],
            "command_dispatch_count": 0,
            **boundary_claim,
        }
        ledger_claim = {
            "segment_completion_ledger_path": str(segment_completion),
            "segment_completion_ledger_sha256": segment_sha256,
            "segment_completion_count": len(segment_rows),
            "expected_segment_completion_count": len(expected_starts),
            "segment_completion_coverage_complete": True,
            "segment_completion_coverage_errors": [],
        }
        feedback_claim = {
            "feedback_recovery_action_ledger_schema_version": (
                runner.FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA
            ),
            "feedback_recovery_action_count": 0,
            "feedback_recovery_verified_action_count": 0,
            "feedback_recovery_physical_response_verified_action_count": 0,
            "feedback_recovery_action_coverage_complete": True,
            "feedback_recovery_action_coverage_errors": [],
            "feedback_recovery_action_ledger_path": str(feedback_recovery),
            "feedback_recovery_action_ledger_sha256": (
                feedback_recovery_sha256
            ),
            "feedback_recovery_dispatch_ledger_path": str(dispatch_ledger),
            "feedback_recovery_dispatch_ledger_sha256": (
                dispatch_ledger_sha256
            ),
            "feedback_recovery_dispatch_count": 0,
        }
        task_payload = {
            "schema_version": runner.TASK_INPUTS_SCHEMA,
            "completed_result": {
                **source_claim,
                **ledger_claim,
                **feedback_claim,
                **transition_claim,
                **telemetry_claim,
                "controller_terminal_outcome": "TASK_SUCCESS",
                "terminal_recovery_closure": recovery_closure,
                "dispatch_complete": True,
                "scheduler_complete": True,
                "macro_controller": {
                    **source_claim,
                    **ledger_claim,
                    **feedback_claim,
                    **transition_claim,
                    **telemetry_claim,
                    "terminal_outcome": "TASK_SUCCESS",
                    "terminal_recovery_closure": recovery_closure,
                },
            },
            "physical_evidence": {
                "body_stuck": None,
                "body_stuck_available": False,
                "body_stuck_source": (
                    "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
                ),
                "active_leg_trapped": None,
                "active_leg_trapped_available": False,
                "active_leg_trapped_source": (
                    "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
                ),
            },
            "final_telemetry_row": copy.deepcopy(telemetry_rows[-1]),
        }
        runner.atomic_write_json(task_inputs, task_payload)
        runner.atomic_write_json(
            worker_result,
            {
                "schema_version": runner.SESSION_SCHEMA,
                "execution_mode": "normal_development",
                "request_id": request["request_id"],
                "source_version": request["source_version"],
                "profile_id": request["profile_id"],
                "graph_id": request["graph_id"],
                "graph_sha256": request["graph_sha256"],
                "profile_library_sha256": request["profile_library_sha256"],
                "bundle_sha256": request["bundle_sha256"],
                "run_dir": request["run_dir"],
                "macro_fsm_complete": True,
                "video_writer_quiesced": True,
                "filtered_contact_bank_enabled": True,
                "physics_dt_s": 1.0 / 120.0,
                "worker_pid": self.pid,
                "worker_session_id": f"worker-{self.pid}",
                "adapter_runtime_instance_id": f"adapter-{self.pid}",
                "artifact_request_id": "",
                "root_state_write_count": 0,
                "servo_command_transform": command_transform,
                "servo_command_transform_sha256": command_transform_sha256,
                "safe_stop_status": "VERIFIED",
                "safe_stop_verified": True,
                "safe_stop_error": "",
                "error": "",
                "task_inputs_path": str(task_inputs),
                "segment_completion_path": str(segment_completion),
                "segment_completion_sha256": segment_sha256,
                **source_claim,
                **ledger_claim,
                **feedback_claim,
                **transition_claim,
                **telemetry_claim,
                "controller_terminal_outcome": "TASK_SUCCESS",
                "terminal_recovery_closure": recovery_closure,
            },
        )
        video.write_bytes(b"video")
        self._history.append(
            {
                "type": "operation_ack",
                "operation": "start_macro_fsm",
                "request_id": request["request_id"],
                "accepted": True,
                "error": "",
                "worker_pid": self.pid,
                "worker_session_id": f"worker-{self.pid}",
                "adapter_runtime_instance_id": f"adapter-{self.pid}",
                "artifact_request_id": "",
                "root_state_write_count": 0,
                "physics_dt_s": 1.0 / 120.0,
                "graph_sha256": request["graph_sha256"],
                "profile_library_sha256": request["profile_library_sha256"],
                "bundle_sha256": request["bundle_sha256"],
                "start_boundary_ack": boundary_ack,
                "first_controller_tick_physics_step": 101,
                "earliest_profile_dispatch_physics_step": 108,
                "earliest_profile_actuation_physics_step": 109,
            }
        )
        terminal = {
            "type": "macro_fsm_complete",
            "operation": "macro_fsm",
            "phase": "MACRO_FSM_COMPLETE",
            "accepted": True,
            "macro_fsm_complete": True,
            "request_id": request["request_id"],
            "source_version": request["source_version"],
            "profile_id": request["profile_id"],
            "graph_id": request["graph_id"],
            "graph_sha256": request["graph_sha256"],
            "profile_library_sha256": request["profile_library_sha256"],
            "bundle_sha256": request["bundle_sha256"],
            "run_dir": request["run_dir"],
            "worker_result_path": str(worker_result),
            "task_inputs_path": str(task_inputs),
            "video_path": str(video),
            "video_writer_quiesced": True,
            "filtered_contact_bank_enabled": True,
            "physics_dt_s": 1.0 / 120.0,
            "worker_pid": self.pid,
            "worker_session_id": f"worker-{self.pid}",
            "adapter_runtime_instance_id": f"adapter-{self.pid}",
            "artifact_request_id": "",
            "root_state_write_count": 0,
            "servo_command_transform": command_transform,
            "servo_command_transform_sha256": command_transform_sha256,
            "safe_stop_status": "VERIFIED",
            "safe_stop_verified": True,
            "safe_stop_error": "",
            "error": "",
            "task_inputs": task_payload,
            "segment_completion_path": str(segment_completion),
            "segment_completion_sha256": segment_sha256,
            **source_claim,
            **ledger_claim,
            **feedback_claim,
            **transition_claim,
            **telemetry_claim,
            "controller_terminal_outcome": "TASK_SUCCESS",
            "terminal_recovery_closure": recovery_closure,
        }
        self.latest_macro_fsm_terminal = terminal
        self._history.append(
            {
                **terminal,
                "type": "operation_ack",
                "operation": "macro_fsm",
                "worker_pid": self.pid,
                "worker_session_id": f"worker-{self.pid}",
                "adapter_runtime_instance_id": f"adapter-{self.pid}",
                "artifact_request_id": "",
                "root_state_write_count": 0,
            }
        )

    def shutdown(self, *, mode, timeout_s, force_on_timeout, request_id):
        assert mode == "fast"
        self.process.returncode = 0
        identity = {
            "worker_pid": self.pid,
            "worker_session_id": f"worker-{self.pid}",
            "adapter_runtime_instance_id": f"adapter-{self.pid}",
            "artifact_request_id": "",
            "macro_fsm_request_id": self.request["request_id"],
            "root_state_write_count": 0,
            "runtime_version": "test-isaac-runtime-1",
            "schema_version": runner.MACRO_FAST_CLOSE_SCHEMA,
        }
        close_kwargs = {"wait_for_replicator": False, "skip_cleanup": True}
        return {
            "pid": self.pid,
            "returncode": 0,
            "forced_termination": False,
            "normal_exit": True,
            "timed_out": False,
            "requested_mode": "fast",
            "request_id": request_id,
            "shutdown_ack": {
                "type": "operation_ack",
                "operation": "shutdown",
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **identity,
            },
            "close_requested_ack": {
                "type": "close_requested",
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **identity,
            },
            "close_requested_receipt": {
                "type": "close_receipt",
                "close_event_type": "close_requested",
                "received": True,
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **identity,
            },
            "close_returned_ack": {
                "type": "close_returned",
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **identity,
            },
            "close_returned_receipt": {
                "type": "close_receipt",
                "close_event_type": "close_returned",
                "received": True,
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **identity,
            },
        }

    def close(self):
        self.closed = True


def _args(root: Path) -> argparse.Namespace:
    alignment = root / "alignment.csv"
    alignment.write_text("phase,range\nA,0:1\n", encoding="utf-8")
    return argparse.Namespace(
        source_version=DEFAULT_PRIMARY_VERSION,
        output_root=root / "runs",
        alignment_path=alignment,
        task_success_table_path=runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
        telemetry_hz=15.0,
        post_run_settle_s=0.5,
        task_timeout_s=240.0,
        terminal_timeout_s=2.0,
        operation_timeout_s=2.0,
        sim_startup_timeout_s=2.0,
        sim_worker_status_timeout_s=10.0,
        sim_worker_log_lines=200,
        worker_launch_mode="auto",
        worker_python_exe="",
        isaaclab_bat="C:/robotics_sim/IsaacLab/isaaclab.bat",
        device="cuda:0",
        livestream=0,
        experience="",
        accept_isaac_eula=False,
    )


def _fake_baseline_attempt(
    root: Path,
) -> tuple[_FakeBundle, runner.MacroAttempt]:
    root.mkdir(parents=True, exist_ok=True)
    args = _args(root)
    bundle = _FakeBundle(args.source_version)
    attempt = runner.run_one_macro(
        cli_args=args,
        bundle=bundle,
        source_version=args.source_version,
        trial_kind="baseline",
        trial_index=0,
        client_factory=_FakeClient,
    )
    return bundle, attempt


def _validate_attempt_terminal(
    attempt: runner.MacroAttempt,
    bundle: _FakeBundle,
) -> dict:
    manifest = runner.strict_json_load(attempt.manifest_path)
    request = runner.strict_json_load(manifest["request_path"])
    return runner.validate_macro_terminal(
        dict(manifest["terminal"]),
        request=request,
        run_dir=attempt.run_dir,
        worker_binding=dict(manifest["worker_binding"]),
        bundle=bundle,
    )


def _write_segment_completion_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _rewrite_self_consistent_ledger_chain(
    attempt: runner.MacroAttempt,
    mutate_rows,
) -> dict:
    manifest = runner.strict_json_load(attempt.manifest_path)
    terminal = copy.deepcopy(manifest["terminal"])
    terminal_ack = copy.deepcopy(manifest["terminal_ack"])
    worker_result_path = Path(terminal["worker_result_path"])
    task_inputs_path = Path(terminal["task_inputs_path"])
    ledger_path = Path(terminal["segment_completion_ledger_path"])
    worker_result = runner.strict_json_load(worker_result_path)
    task_inputs = runner.strict_json_load(task_inputs_path)
    rows = runner._strict_jsonl_load(ledger_path)
    mutate_rows(rows)
    _write_segment_completion_rows(ledger_path, rows)
    ledger_sha256 = runner.sha256_file(ledger_path)
    claim = {
        "segment_completion_ledger_path": str(ledger_path),
        "segment_completion_ledger_sha256": ledger_sha256,
        "segment_completion_count": len(rows),
        "expected_segment_completion_count": terminal[
            "expected_segment_completion_count"
        ],
        "segment_completion_coverage_complete": True,
        "segment_completion_coverage_errors": [],
    }
    completed = task_inputs["completed_result"]
    controller = completed["macro_controller"]
    for mapping in (completed, controller):
        mapping.update(copy.deepcopy(claim))
    runner.atomic_write_json(task_inputs_path, task_inputs)
    for mapping in (terminal, terminal_ack, worker_result):
        mapping.update(copy.deepcopy(claim))
        mapping["segment_completion_path"] = str(ledger_path)
        mapping["segment_completion_sha256"] = ledger_sha256
    terminal["task_inputs"] = copy.deepcopy(task_inputs)
    terminal_ack["task_inputs"] = copy.deepcopy(task_inputs)
    runner.atomic_write_json(worker_result_path, worker_result)
    manifest.update(copy.deepcopy(claim))
    manifest["terminal"] = terminal
    manifest["terminal_ack"] = terminal_ack
    runner.atomic_write_json(attempt.manifest_path, manifest)
    return manifest


def _add_valid_dynamic_wheel_stop(rows: list[dict]) -> None:
    row = rows[0]
    start_step = row["start_sim_step"]
    start_time = row["start_sim_time_s"]
    row["dynamic_servo_duration_s"] = 0.1
    row["effective_completion_spec"]["servo_duration_s"] = 0.1
    wait = copy.deepcopy(row["observation_decisions"][-1])
    wait_step = start_step + 8
    wait_time = start_time + 8.0 / 120.0
    wait.update(
        kind=SegmentDecisionKind.WAIT.value,
        sim_step=wait_step,
        sim_time_s=wait_time,
        segment_elapsed_s=8.0 / 120.0,
        servo_planned_done=False,
        reference_position_done=False,
        servo_done=False,
        servo_errors_deg={},
        servo_velocity_deg_s={},
        max_servo_error_deg=0.0,
        wheel_elapsed_s=8.0 / 120.0,
        wheel_done=False,
        hold_elapsed_s=8.0 / 120.0,
        segment_done=False,
        wheel_stop_due=False,
        wheel_stop_acknowledged=False,
        worst_joint="unknown",
        phase="waiting",
    )
    due = copy.deepcopy(row["observation_decisions"][-1])
    due_step = start_step + 16
    due_time = start_time + 16.0 / 120.0
    due.update(
        kind=SegmentDecisionKind.WHEEL_STOP_DUE.value,
        sim_step=due_step,
        sim_time_s=due_time,
        segment_elapsed_s=16.0 / 120.0,
        reference_position_done=False,
        servo_done=False,
        servo_errors_deg={"front_left_hip": 2.0},
        servo_velocity_deg_s={"front_left_hip": 0.0},
        max_servo_error_deg=2.0,
        wheel_elapsed_s=16.0 / 120.0,
        hold_elapsed_s=16.0 / 120.0,
        segment_done=False,
        wheel_stop_due=True,
        wheel_stop_acknowledged=False,
        worst_joint="front_left_hip",
        phase="servo_completion_extension",
    )
    final = copy.deepcopy(row["observation_decisions"][-1])
    final_step = start_step + 24
    final_time = start_time + 24.0 / 120.0
    final.update(
        sim_step=final_step,
        sim_time_s=final_time,
        segment_elapsed_s=24.0 / 120.0,
        wheel_elapsed_s=24.0 / 120.0,
        hold_elapsed_s=24.0 / 120.0,
        wheel_stop_acknowledged=True,
        worst_joint="front_left_hip",
        phase="waiting",
    )
    due_sha256 = runner._sha256_mapping(due)
    completion_token = runner.MacroSegmentCompletionToken(
        profile_id=row["profile_id"],
        profile_source_version=row["profile_source_version"],
        owner_state=row["owner_state"],
        source_plan_sha256=row["source_plan_sha256"],
        source_plan_payload_sha256=row["source_plan_payload_sha256"],
        accepted_steps_sha256=row["accepted_steps_sha256"],
        source_segment_index=row["source_segment_index"],
        source_step_index=row["source_step_index"],
        source_step_id=row["source_step_id"],
        start_command_epoch=row["start_command_epoch"],
        start_sim_step=start_step,
        start_readback_sha256=row["start_readback_sha256"],
        decision=due,
        decision_sha256=due_sha256,
    )
    final_sha256 = runner._sha256_mapping(final)
    row.update(
        observation_decisions=[wait, due, final],
        last_decision=final,
        last_decision_sha256=final_sha256,
        wheel_stop={
            "generated": True,
            "source_action": False,
            "source_action_consumption_index": None,
            "completion_token_sha256": completion_token.sha256,
            "applied_sim_step": due_step,
            "first_physics_step": due_step + 1,
            "batch_id": "fake-dynamic-wheel-stop",
            "n_plus_one_verified": True,
            "n_plus_one_verified_sim_step": due_step + 1,
            "n_plus_one_readback_sha256": "7" * 64,
        },
        tracking_end_sim_step=final_step,
        tracking_end_sim_time_s=final_time,
        terminal_kind=SegmentDecisionKind.COMPLETE.value,
        terminal_sim_step=final_step,
        terminal_sim_time_s=final_time,
        terminal_decision_sha256=final_sha256,
    )


def _successful_review_document(attempt: runner.MacroAttempt) -> dict:
    document = runner.strict_json_load(
        attempt.run_dir / runner.MANUAL_VERDICT_TEMPLATE_NAME
    )
    document["review_complete"] = True
    document["reviewed_utc"] = "2026-08-15T04:00:00+00:00"
    document["verdict"].update(
        task_completed=True,
        body_crossed_front_face=True,
        required_leg_lift_completed=True,
        final_recoverable=True,
        posture_incomplete=False,
        robot_fell=False,
        body_stuck=False,
        wheel_drive_up_without_required_lift=False,
        dangerous_body_collision=False,
        joint_limit_violation=False,
        severe_penetration=False,
    )
    return document


def _review_success(attempt: runner.MacroAttempt) -> None:
    document = _successful_review_document(attempt)
    verdict_path = attempt.run_dir / "review-input.json"
    runner.atomic_write_json(verdict_path, document)
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=_FakeBundle(attempt.source_version),
    ):
        result = runner.review_run(
            run_dir=attempt.run_dir,
            verdict_path=verdict_path,
        )
    assert result["manual_review_status"] == "MACRO_FSM_TASK_SUCCESS"


def _gate_d_review_attempt(
    tmp_path: Path,
) -> tuple[_FakeBundle, dict, runner.MacroAttempt, Path]:
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 900
    source_version = runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS[0]
    args = _args(tmp_path)
    args.source_version = source_version
    args.output_root = tmp_path / "gate_d"
    bundle = _FakeBundle(source_version)
    closure_payload = {
        "schema_version": runner.GATE_C_CLOSURE_EVIDENCE_SCHEMA,
        "baseline": {"run_dir": str(tmp_path / "gate_c" / "baseline")},
        "repeats": [],
        "bundle_sha256": bundle.bundle_sha256,
        "graph_sha256": bundle.graph_sha256,
        "profile_library_sha256": bundle.profile_library_sha256,
    }
    closure = {
        **closure_payload,
        "closure_sha256": runner._sha256_mapping(closure_payload),
    }
    attempt = runner.run_one_macro(
        cli_args=args,
        bundle=bundle,
        source_version=source_version,
        trial_kind=runner.GATE_D_TRIAL_KIND,
        trial_index=0,
        gate_c_closure=closure,
        client_factory=_FakeClient,
    )
    verdict_path = tmp_path / "gate-d-review-input.json"
    runner.atomic_write_json(verdict_path, _successful_review_document(attempt))
    return bundle, closure, attempt, verdict_path


def _assert_gate_d_review_rejected(
    *,
    attempt: runner.MacroAttempt,
    verdict_path: Path,
    bundle: _FakeBundle,
    closure: dict,
) -> None:
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=bundle,
    ), mock.patch.object(
        runner,
        "_validate_persisted_gate_c_closure",
        return_value=closure,
    ), pytest_raises(runner.MacroRunnerContractError):
        runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict_path)
    assert not (attempt.run_dir / runner.MANUAL_VERDICT_NAME).exists()


def test_environment_lock_fails_closed_before_worker_launch(tmp_path: Path):
    canonical_source = str(runner.__file__)
    valid_verification = {
        "ok": True,
        "files": [
            {
                "path": canonical_source,
                "expected_sha256": "a" * 64,
                "actual_sha256": "a" * 64,
                "ok": True,
            }
        ],
        "required_source_file_count": 1,
        "locked_source_file_count": 1,
        "missing_from_lock": [],
        "source_closure_complete": True,
    }
    identity = runner._validate_canonical_environment_lock(
        load_lock=lambda _path: {
            "schema_version": "fsm50.environment_lock.v1"
        },
        verify_sources=lambda _lock: copy.deepcopy(valid_verification),
    )
    assert identity["environment_lock_path"] == str(
        runner.DEFAULT_ENVIRONMENT_LOCK_PATH.resolve()
    )
    assert identity["environment_lock_sha256"] == runner.sha256_file(
        runner.DEFAULT_ENVIRONMENT_LOCK_PATH
    )
    assert identity["source_closure_complete"] is True

    stale_cases = []
    stale = copy.deepcopy(valid_verification)
    stale["ok"] = False
    stale_cases.append(stale)
    incomplete = copy.deepcopy(valid_verification)
    incomplete["source_closure_complete"] = False
    incomplete["missing_from_lock"] = ["missing.py"]
    stale_cases.append(incomplete)
    mismatched_count = copy.deepcopy(valid_verification)
    mismatched_count["required_source_file_count"] = 2
    stale_cases.append(mismatched_count)
    mismatched_file = copy.deepcopy(valid_verification)
    mismatched_file["ok"] = False
    mismatched_file["files"][0]["ok"] = False
    stale_cases.append(mismatched_file)
    malformed_file = copy.deepcopy(valid_verification)
    malformed_file["ok"] = False
    malformed_file["files"] = ["not-a-verification-row"]
    stale_cases.append(malformed_file)
    for verification in stale_cases:
        with pytest_raises(runner.MacroRunnerContractError):
            runner._validate_canonical_environment_lock(
                load_lock=lambda _path: {},
                verify_sources=lambda _lock, row=verification: copy.deepcopy(row),
            )

    escaped_lock = tmp_path / "environment_lock_50mm.json"
    escaped_lock.write_text("{}\n", encoding="utf-8")
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_canonical_environment_lock(
            lock_path=escaped_lock,
            load_lock=lambda _path: {},
            verify_sources=lambda _lock: copy.deepcopy(valid_verification),
        )

    prelock_root = tmp_path / "prelock"
    prelock_root.mkdir()
    args = _args(prelock_root)
    calls: list[str] = []

    def reject_stale_lock():
        calls.append("environment")
        raise runner.MacroRunnerContractError("stale environment lock")

    def forbidden_bundle(*_args, **_kwargs):
        calls.append("bundle")
        raise AssertionError("bundle rebuild must not precede lock admission")

    with pytest_raises(runner.MacroRunnerContractError):
        runner.run_trials(
            args,
            trial_kind="baseline",
            count=1,
            client_factory=_FakeClient,
            bundle_builder=forbidden_bundle,
            lock_factory=_Lock,
            process_snapshot_fn=lambda: [],
            conflict_detector=lambda _rows: [],
            environment_lock_validator=reject_stale_lock,
        )
    assert calls == ["environment"]

    under_lock_root = tmp_path / "under-lock"
    under_lock_root.mkdir()
    args = _args(under_lock_root)
    validation_calls = 0
    client_count = len(_FakeClient.instances)

    def changing_lock_identity():
        nonlocal validation_calls
        validation_calls += 1
        result = _valid_environment_identity()
        if validation_calls == 2:
            result["source_verification_sha256"] = "0" * 64
        return result

    with pytest_raises(runner.MacroRunnerContractError):
        runner.run_trials(
            args,
            trial_kind="baseline",
            count=1,
            client_factory=_FakeClient,
            bundle_builder=lambda *_args, **_kwargs: _FakeBundle(),
            lock_factory=_Lock,
            process_snapshot_fn=lambda: [],
            conflict_detector=lambda _rows: [],
            environment_lock_validator=changing_lock_identity,
        )
    assert validation_calls == 2
    assert len(_FakeClient.instances) == client_count
    assert _Lock.active == 0


def test_segment_completion_artifact_chain_rejects_every_layer_tamper(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 1000

    bundle, attempt = _fake_baseline_attempt(tmp_path / "terminal-alias")
    manifest = runner.strict_json_load(attempt.manifest_path)
    request = runner.strict_json_load(manifest["request_path"])
    terminal = copy.deepcopy(manifest["terminal"])
    terminal["segment_completion_sha256"] = "0" * 64
    with pytest_raises(runner.MacroRunnerContractError):
        runner.validate_macro_terminal(
            terminal,
            request=request,
            run_dir=attempt.run_dir,
            worker_binding=manifest["worker_binding"],
            bundle=bundle,
        )

    bundle, attempt = _fake_baseline_attempt(tmp_path / "result-claim")
    manifest = runner.strict_json_load(attempt.manifest_path)
    result_path = Path(manifest["terminal"]["worker_result_path"])
    durable = runner.strict_json_load(result_path)
    durable["segment_completion_ledger_sha256"] = "0" * 64
    durable["segment_completion_sha256"] = "0" * 64
    runner.atomic_write_json(result_path, durable)
    with pytest_raises(runner.MacroRunnerContractError):
        _validate_attempt_terminal(attempt, bundle)

    bundle, attempt = _fake_baseline_attempt(tmp_path / "task-claim")
    manifest = runner.strict_json_load(attempt.manifest_path)
    task_path = Path(manifest["terminal"]["task_inputs_path"])
    task_inputs = runner.strict_json_load(task_path)
    task_inputs["completed_result"]["segment_completion_count"] = 0
    runner.atomic_write_json(task_path, task_inputs)
    with pytest_raises(runner.MacroRunnerContractError):
        _validate_attempt_terminal(attempt, bundle)

    bundle, attempt = _fake_baseline_attempt(tmp_path / "controller-claim")
    manifest = runner.strict_json_load(attempt.manifest_path)
    task_path = Path(manifest["terminal"]["task_inputs_path"])
    task_inputs = runner.strict_json_load(task_path)
    task_inputs["completed_result"]["macro_controller"][
        "segment_completion_coverage_errors"
    ] = ["tampered"]
    runner.atomic_write_json(task_path, task_inputs)
    with pytest_raises(runner.MacroRunnerContractError):
        _validate_attempt_terminal(attempt, bundle)

    bundle, attempt = _fake_baseline_attempt(tmp_path / "ledger-bytes")
    manifest = runner.strict_json_load(attempt.manifest_path)
    ledger_path = Path(manifest["segment_completion_ledger_path"])
    ledger_path.write_bytes(ledger_path.read_bytes() + b"{}\n")
    with pytest_raises(runner.MacroRunnerContractError):
        _validate_attempt_terminal(attempt, bundle)

    bundle, attempt = _fake_baseline_attempt(tmp_path / "terminal-ack")
    manifest = runner.strict_json_load(attempt.manifest_path)
    request = runner.strict_json_load(manifest["request_path"])
    terminal = _validate_attempt_terminal(attempt, bundle)
    acknowledgement = copy.deepcopy(manifest["terminal_ack"])
    acknowledgement["task_inputs"]["completed_result"][
        "segment_completion_count"
    ] = 0
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_terminal_ack(
            acknowledgement,
            terminal=terminal,
            request=request,
            run_dir=attempt.run_dir,
            worker_binding=manifest["worker_binding"],
        )

    bundle, attempt = _fake_baseline_attempt(tmp_path / "manifest-claim")
    manifest = runner.strict_json_load(attempt.manifest_path)
    manifest["segment_completion_count"] = 0
    runner.atomic_write_json(attempt.manifest_path, manifest)
    verdict_path = tmp_path / "manifest-claim-verdict.json"
    runner.atomic_write_json(verdict_path, _successful_review_document(attempt))
    with mock.patch.object(
        runner, "build_gate_c_bundle", return_value=bundle
    ), pytest_raises(runner.MacroRunnerContractError):
        runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict_path)
    assert not (attempt.run_dir / runner.MANUAL_VERDICT_NAME).exists()


def test_segment_completion_ledger_rejects_self_consistent_row_tampering(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 1100

    def extra_key(rows):
        rows[0]["unrecognized"] = True

    def plan_identity(rows):
        rows[0]["source_plan_payload_sha256"] = "0" * 64

    def boolean_index(rows):
        rows[0]["segment_completion_index"] = False

    def completion_spec(rows):
        rows[0]["completion_spec"]["source_step_id"] = "tampered"

    def effective_duration(rows):
        rows[0]["dynamic_servo_duration_s"] = 1.0
        rows[0]["effective_completion_spec"]["servo_duration_s"] = 1.0

    def retained_old_readback(rows):
        row = rows[0]
        row["start_physical_dispatch"] = False
        row["start_batch_id"] = ""
        row["start_first_physics_step"] = None
        row["start_readback_verified_sim_step"] = row["start_sim_step"] - 1
        row["retained_epoch_same_target"] = True

    def source_start_epoch(rows):
        rows[0]["start_command_epoch"] += 1

    def source_start_tick(rows):
        rows[0]["start_sim_step"] += 1
        rows[0]["start_sim_time_s"] = rows[0]["start_sim_step"] / 120.0

    def tracking_lifecycle(rows):
        rows[0]["tracking_end_count"] = 0

    def decision_schema(rows):
        row = rows[0]
        final = copy.deepcopy(row["observation_decisions"][-1])
        final["unrecognized"] = True
        final_sha256 = runner._sha256_mapping(final)
        row["observation_decisions"][-1] = final
        row["last_decision"] = final
        row["last_decision_sha256"] = final_sha256
        row["terminal_decision_sha256"] = final_sha256

    def planned_decision_missing_feedback(rows):
        row = rows[0]
        final = copy.deepcopy(row["observation_decisions"][-1])
        final["servo_errors_deg"] = {}
        final["servo_velocity_deg_s"] = {}
        final_sha256 = runner._sha256_mapping(final)
        row["observation_decisions"][-1] = final
        row["last_decision"] = final
        row["last_decision_sha256"] = final_sha256
        row["terminal_decision_sha256"] = final_sha256

    def terminal_kind(rows):
        rows[0]["terminal_kind"] = SegmentDecisionKind.FAIL.value

    def untriggered_wheel_stop(rows):
        rows[0]["wheel_stop"] = {
            "generated": True,
            "source_action": False,
            "source_action_consumption_index": None,
            "completion_token_sha256": "5" * 64,
            "applied_sim_step": rows[0]["terminal_sim_step"],
            "first_physics_step": rows[0]["terminal_sim_step"] + 1,
            "batch_id": "untriggered",
            "n_plus_one_verified": True,
            "n_plus_one_verified_sim_step": rows[0]["terminal_sim_step"] + 1,
            "n_plus_one_readback_sha256": "6" * 64,
        }

    for index, mutate in enumerate(
        (
            extra_key,
            plan_identity,
            boolean_index,
            completion_spec,
            effective_duration,
            retained_old_readback,
            source_start_epoch,
            source_start_tick,
            tracking_lifecycle,
            decision_schema,
            planned_decision_missing_feedback,
            terminal_kind,
            untriggered_wheel_stop,
        )
    ):
        bundle, attempt = _fake_baseline_attempt(tmp_path / f"row-{index}")
        _rewrite_self_consistent_ledger_chain(attempt, mutate)
        with pytest_raises(runner.MacroRunnerContractError):
            _validate_attempt_terminal(attempt, bundle)

    bundle, attempt = _fake_baseline_attempt(tmp_path / "dynamic-stop")
    _rewrite_self_consistent_ledger_chain(attempt, _add_valid_dynamic_wheel_stop)
    _validate_attempt_terminal(attempt, bundle)

    def invalid_completion_token(rows):
        rows[0]["wheel_stop"]["completion_token_sha256"] = "0" * 64

    _rewrite_self_consistent_ledger_chain(attempt, invalid_completion_token)
    with pytest_raises(runner.MacroRunnerContractError):
        _validate_attempt_terminal(attempt, bundle)


def test_segment_completion_allows_same_tick_next_dispatch_but_not_earlier(
    tmp_path: Path,
):
    bundle = build_gate_c_bundle(runner.PROJECT_ROOT)
    paths = runner.allocate_run_paths(
        tmp_path / "runs",
        source_version=bundle.primary_source_version,
        trial_kind="baseline",
        trial_index=0,
        bundle_sha256=bundle.bundle_sha256,
    )
    request = runner.build_worker_macro_request(
        bundle,
        paths,
        request_id="same-tick-completion-test",
        source_version=bundle.primary_source_version,
        alignment_path=runner.DEFAULT_ALIGNMENT_PATH,
        task_success_table_path=runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
        trial_kind="baseline",
        trial_index=0,
        telemetry_hz=15.0,
        post_run_settle_s=0.5,
        timeout_s=120.0,
    )
    runner.atomic_write_json(paths.request_path, request)
    starts, stops = runner._expected_segment_start_actions(
        bundle=bundle,
        request=request,
        run_dir=paths.run_dir,
    )
    previous = _synthetic_segment_completion_row(starts[0], 0)
    current = _synthetic_segment_completion_row(
        starts[1],
        1,
        start_step=previous["terminal_sim_step"],
    )

    # A COMPLETE observation applies no batch and has no same-step wheel stop.
    # The next physical segment owns the single batch at that step and proves
    # its actuation with the exact N+1 readback.
    assert previous["terminal_kind"] == SegmentDecisionKind.COMPLETE.value
    assert previous["wheel_stop"] is None
    assert current["start_physical_dispatch"] is True
    assert current["start_sim_step"] == previous["terminal_sim_step"]
    assert current["start_first_physics_step"] == current["start_sim_step"] + 1
    assert (
        current["start_readback_verified_sim_step"]
        == current["start_first_physics_step"]
    )
    batch_steps = [previous["start_sim_step"], current["start_sim_step"]]
    assert len(batch_steps) == len(set(batch_steps))

    runner._validate_segment_completion_row(
        previous,
        row_index=0,
        expected_action=starts[0],
        expected_stop_action=stops.get(0),
        previous_terminal_sim_step=None,
    )
    runner._validate_segment_completion_row(
        current,
        row_index=1,
        expected_action=starts[1],
        expected_stop_action=stops.get(1),
        previous_terminal_sim_step=previous["terminal_sim_step"],
    )

    earlier = _synthetic_segment_completion_row(
        starts[1],
        1,
        start_step=previous["terminal_sim_step"] - 1,
    )
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_segment_completion_row(
            earlier,
            row_index=1,
            expected_action=starts[1],
            expected_stop_action=stops.get(1),
            previous_terminal_sim_step=previous["terminal_sim_step"],
        )


def test_macro_start_ack_requires_render8_outer_cadence(tmp_path: Path):
    class StaleRender2Client(_FakeClient):
        def start_macro_fsm(self, **identity):
            super().start_macro_fsm(**identity)
            self._history[0]["earliest_profile_dispatch_physics_step"] = 103
            self._history[0]["earliest_profile_actuation_physics_step"] = 104

    StaleRender2Client.instances.clear()
    args = _args(tmp_path)
    with pytest_raises(runner.MacroRunnerContractError):
        runner.run_one_macro(
            cli_args=args,
            bundle=_FakeBundle(),
            source_version=args.source_version,
            trial_kind="baseline",
            trial_index=0,
            client_factory=StaleRender2Client,
        )
    assert len(StaleRender2Client.instances) == 1
    assert StaleRender2Client.instances[0].process.poll() == 0
    assert StaleRender2Client.instances[0].closed is True


def test_start_boundary_ack_is_independently_validated_and_manifest_bound():
    request = {
        "request_id": "request",
        "graph_sha256": "a" * 64,
        "profile_library_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
    }
    worker_binding = {
        "worker_pid": 401,
        "worker_session_id": "worker-401",
        "adapter_runtime_instance_id": "adapter-401",
    }
    boundary = {
        "batch_id": "request:start-boundary",
        "source": "fsm50_macro_start_boundary",
        "error": "",
        "applied_sim_step": 100,
        "first_physics_step": 101,
        "motion_start_skew_s": 0.0,
        "physics_dt_s": 1.0 / 120.0,
        "servo_applied": True,
        "wheel_applied": True,
        "servo_targets_applied": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "wheel_targets_applied": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        "recording_metadata": {},
    }
    acknowledgement = {
        "type": "operation_ack",
        "operation": "start_macro_fsm",
        "request_id": "request",
        "accepted": True,
        "error": "",
        "worker_pid": 401,
        "worker_session_id": "worker-401",
        "adapter_runtime_instance_id": "adapter-401",
        "artifact_request_id": "",
        "root_state_write_count": 0,
        "physics_dt_s": 1.0 / 120.0,
        "graph_sha256": "a" * 64,
        "profile_library_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
        "start_boundary_ack": boundary,
        "first_controller_tick_physics_step": 101,
        "earliest_profile_dispatch_physics_step": 108,
        "earliest_profile_actuation_physics_step": 109,
    }
    validated = runner._validate_macro_start_ack(
        acknowledgement,
        request=request,
        worker_binding=worker_binding,
        expected_boundary_ack=boundary,
    )
    assert validated["start_boundary_ack"] == boundary

    for key, forged in (
        ("source", "forged-source"),
        ("motion_start_skew_s", 0.5),
        ("physics_dt_s", 0.5),
    ):
        tampered = copy.deepcopy(acknowledgement)
        tampered["start_boundary_ack"][key] = forged
        with pytest_raises(runner.MacroRunnerContractError):
            runner._validate_macro_start_ack(
                tampered,
                request=request,
                worker_binding=worker_binding,
                expected_boundary_ack=tampered["start_boundary_ack"],
            )

    different_persisted = copy.deepcopy(acknowledgement)
    different_persisted["start_boundary_ack"]["batch_id"] = "other-boundary"
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_macro_start_ack(
            different_persisted,
            request=request,
            worker_binding=worker_binding,
            expected_boundary_ack=boundary,
        )

    coherently_forged = copy.deepcopy(acknowledgement)
    coherently_forged["start_boundary_ack"]["batch_id"] = "forged-boundary"
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_macro_start_ack(
            coherently_forged,
            request=request,
            worker_binding=worker_binding,
            expected_boundary_ack=coherently_forged["start_boundary_ack"],
        )


def test_real_gate_a_gate_b_bundle_rebuild_and_request_are_sha_bound(tmp_path: Path):
    bundle = build_gate_c_bundle(runner.PROJECT_ROOT)
    assert bundle.primary_source_version == DEFAULT_PRIMARY_VERSION
    assert len(bundle.profiles.profiles) == 33
    assert len(bundle.profiles.segment_ownership) == 33
    paths = runner.allocate_run_paths(
        tmp_path / "runs",
        source_version=bundle.primary_source_version,
        trial_kind="baseline",
        trial_index=0,
        bundle_sha256=bundle.bundle_sha256,
    )
    request = runner.build_worker_macro_request(
        bundle,
        paths,
        request_id="request",
        source_version=bundle.primary_source_version,
        alignment_path=runner.DEFAULT_ALIGNMENT_PATH,
        task_success_table_path=runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
        trial_kind="baseline",
        trial_index=0,
    )
    runner.atomic_write_json(paths.request_path, request)
    loaded = load_worker_macro_fsm_request(paths.request_path)
    assert loaded is not None
    assert loaded.bundle_sha256 == bundle.bundle_sha256
    assert loaded.graph_sha256 == bundle.graph_sha256
    assert loaded.profile_library_sha256 == bundle.profile_library_sha256
    assert loaded.alignment_sha256 == runner.sha256_file(runner.DEFAULT_ALIGNMENT_PATH)
    assert loaded.task_success_table_sha256 == runner.sha256_file(
        runner.DEFAULT_TASK_SUCCESS_TABLE_PATH
    )
    alternate = tmp_path / "alternate_success.csv"
    alternate.write_text(
        runner.DEFAULT_TASK_SUCCESS_TABLE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest_raises(runner.MacroRunnerContractError):
        runner.build_worker_macro_request(
            bundle,
            paths,
            request_id="alternate-request",
            source_version=bundle.primary_source_version,
            alignment_path=runner.DEFAULT_ALIGNMENT_PATH,
            task_success_table_path=alternate,
            trial_kind="baseline",
            trial_index=0,
        )


def test_gate_d_sources_rebuild_same_graph_and_load_as_cross_version(tmp_path: Path):
    canonical = build_gate_c_bundle(runner.PROJECT_ROOT)
    gate_d_bundle_shas = set()
    expected_action_counts = {
        "v008_20260806_211408_578700_manual": 119,
        "v009_20260806_215232_433234_manual": 132,
        "v010_20260806_220745_363972_manual": 142,
    }
    for source_version in runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS:
        bundle = build_gate_c_bundle(
            runner.PROJECT_ROOT,
            primary_source_version=source_version,
        )
        assert bundle.graph_sha256 == canonical.graph_sha256
        assert bundle.profile_library_sha256 == canonical.profile_library_sha256
        assert bundle.primary_source_version == source_version
        gate_d_bundle_shas.add(bundle.bundle_sha256)
        paths = runner.allocate_run_paths(
            tmp_path / "cross_version_macro_fsm",
            source_version=source_version,
            trial_kind=runner.GATE_D_TRIAL_KIND,
            trial_index=0,
            bundle_sha256=bundle.bundle_sha256,
        )
        request = runner.build_worker_macro_request(
            bundle,
            paths,
            request_id=f"request-{source_version}",
            source_version=source_version,
            alignment_path=runner.DEFAULT_ALIGNMENT_PATH,
            task_success_table_path=runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
            trial_kind=runner.GATE_D_TRIAL_KIND,
            trial_index=0,
        )
        runner.atomic_write_json(paths.request_path, request)
        loaded = load_worker_macro_fsm_request(paths.request_path)
        assert loaded is not None
        assert loaded.source_version == source_version
        assert loaded.trial_kind == runner.GATE_D_TRIAL_KIND
        assert loaded.bundle_sha256 == bundle.bundle_sha256
        session = WorkerMacroFSMSession(loaded, worker_session_id="offline-audit")
        actions = session._build_expected_source_actions(bundle)
        assert len(actions) == expected_action_counts[source_version]
        assert [
            row["command_provenance"]["source_segment_index"] for row in actions
        ] == list(range(expected_action_counts[source_version]))
        assert paths.run_dir.parent.name == "trials"
        assert paths.run_dir.parent.parent.name == source_version
    assert len(gate_d_bundle_shas) == len(runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS)


def test_baseline_then_three_repeats_use_four_fresh_serial_workers(tmp_path: Path):
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 400
    args = _args(tmp_path)
    common = {
        "client_factory": _FakeClient,
        "bundle_builder": lambda *_args, **_kwargs: _FakeBundle(),
        "lock_factory": _Lock,
        "process_snapshot_fn": lambda: [],
        "conflict_detector": lambda _rows: [],
        "environment_lock_validator": _valid_environment_identity,
    }
    baseline = runner.run_trials(
        args, trial_kind="baseline", count=1, **common
    )
    repeats = runner.run_trials(
        args, trial_kind="repeat", count=3, **common
    )
    assert len(baseline) == 1
    assert len(repeats) == 3
    assert [item.trial_index for item in repeats] == [0, 1, 2]
    assert len(_FakeClient.instances) == 4
    assert len({client.pid for client in _FakeClient.instances}) == 4
    assert all(client.process.poll() == 0 for client in _FakeClient.instances)
    assert all(client.closed for client in _FakeClient.instances)
    assert _Lock.active == 0
    for attempt in [*baseline, *repeats]:
        manifest = runner.strict_json_load(attempt.manifest_path)
        assert manifest["controller_complete"] is True
        assert manifest["shutdown_verified"] is True
        assert manifest["bundle_sha256"] == _FakeBundle.bundle_sha256
        assert manifest["manual_review_status"] == (
            "NOT_EVALUATED_PENDING_SHA_BOUND_VIDEO_REVIEW"
        )
        task_inputs = runner.strict_json_load(
            manifest["terminal"]["task_inputs_path"]
        )
        physical = task_inputs["physical_evidence"]
        for field in ("body_stuck", "active_leg_trapped"):
            assert physical[field] is None
            assert physical[f"{field}_available"] is False
            assert physical[f"{field}_source"] == (
                "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
            )
        assert manifest["shutdown_outcome"]["close_returned_ack"][
            "type"
        ] == "close_returned"
        assert manifest["shutdown_outcome"]["close_returned_receipt"][
            "close_event_type"
        ] == "close_returned"


def test_transition_telemetry_and_machine_outcome_are_not_self_asserted(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    args = _args(tmp_path)
    attempt = runner.run_trials(
        args,
        trial_kind="baseline",
        count=1,
        client_factory=_FakeClient,
        bundle_builder=lambda *_args, **_kwargs: _FakeBundle(),
        lock_factory=_Lock,
        process_snapshot_fn=lambda: [],
        conflict_detector=lambda _rows: [],
        environment_lock_validator=_valid_environment_identity,
    )[0]
    run = attempt.run_dir
    transition_rows = runner._strict_jsonl_load(
        run / "macro_transition_evidence.jsonl"
    )
    source_rows = runner._strict_jsonl_load(
        run / "macro_source_action_consumption.jsonl"
    )
    segment_rows = runner._strict_jsonl_load(
        run / "macro_segment_completion_ledger.jsonl"
    )
    dispatch_rows = []
    runner._validate_transition_rows(
        transition_rows,
        bundle=_FakeBundle(),
        source_rows=source_rows,
        dispatch_rows=dispatch_rows,
        segment_rows=segment_rows,
        boundary_first_physics_step=101,
        source_version=args.source_version,
    )
    for mutate in (
        lambda rows: rows[0].__setitem__("subphase", {}),
        lambda rows: rows[0].__setitem__("phase_elapsed_s", -9.0),
        lambda rows: rows[1].__setitem__("profile_id", "forged"),
    ):
        changed = copy.deepcopy(transition_rows)
        mutate(changed)
        with pytest.raises(runner.MacroRunnerContractError):
            runner._validate_transition_rows(
                changed,
                bundle=_FakeBundle(),
                source_rows=source_rows,
                dispatch_rows=dispatch_rows,
                segment_rows=segment_rows,
                boundary_first_physics_step=101,
                source_version=args.source_version,
            )
    appended = copy.deepcopy(transition_rows)
    extra = copy.deepcopy(appended[-1])
    extra.update(
        transition_index=len(appended),
        sim_step=extra["sim_step"] + 1,
        sim_time_s=(extra["sim_step"] + 1) / 120.0,
        from_state="SUCCESS",
        to_state="SUCCESS",
        events=["RETRY:SUCCESS:0"],
    )
    appended.append(extra)
    with pytest.raises(runner.MacroRunnerContractError):
        runner._validate_transition_rows(
            appended,
            bundle=_FakeBundle(),
            source_rows=source_rows,
            dispatch_rows=dispatch_rows,
            segment_rows=segment_rows,
            boundary_first_physics_step=101,
            source_version=args.source_version,
        )

    telemetry_rows = runner._strict_jsonl_load(
        run / "minimal_macro_telemetry.jsonl"
    )
    runner._validate_telemetry_chronology(
        telemetry_rows, source_version=args.source_version
    )
    missing_source = copy.deepcopy(telemetry_rows)
    missing_source[0].pop("source_version")
    with pytest.raises(runner.MacroRunnerContractError):
        runner._validate_telemetry_chronology(
            missing_source, source_version=args.source_version
        )
    post_success_regression = copy.deepcopy(telemetry_rows)
    regressed = copy.deepcopy(post_success_regression[0])
    regressed_step = transition_rows[-1]["sim_step"] + 1
    regressed.update(
        sim_step=regressed_step,
        sim_time_s=regressed_step / 120.0,
        macro_state="S10_POSTURE_RECOVERY",
        controller_terminal=False,
        controller_terminal_outcome="RUNNING",
    )
    post_success_regression.insert(0, regressed)
    with pytest.raises(runner.MacroRunnerContractError):
        runner._validate_telemetry_chronology(
            post_success_regression,
            source_version=args.source_version,
            terminal_transition=transition_rows[-1],
            success_state="SUCCESS",
            terminal_outcome="TASK_SUCCESS",
        )

    request = load_worker_macro_fsm_request(
        run / "worker_macro_fsm_request.json"
    )
    auditor = WorkerMacroFSMSession(
        request, worker_session_id="terminal-row-test"
    )
    durable = runner.strict_json_load(run / "worker_macro_fsm_result.json")
    auditor.durable_servo_command_transform = durable[
        "servo_command_transform"
    ]
    good = auditor._feedback_recovery_terminal_row_assessment(
        row=telemetry_rows[-1], outcome="TASK_SUCCESS"
    )
    assert good["valid"] is True
    missing_wheel = copy.deepcopy(telemetry_rows[-1])
    missing_wheel["joint_position_rad"].pop(WHEEL_JOINT_NAMES[0])
    bad = auditor._feedback_recovery_terminal_row_assessment(
        row=missing_wheel, outcome="TASK_SUCCESS"
    )
    assert bad["valid"] is False
    assert "kinematic row" in bad["error"]

    document = _successful_review_document(attempt)
    document["verdict"]["posture_incomplete"] = False
    machine_incomplete = copy.deepcopy(
        runner.strict_json_load(run / runner.RUNNER_MANIFEST_NAME)["terminal"]
    )
    machine_incomplete["controller_terminal_outcome"] = (
        "TASK_SUCCESS_POSTURE_INCOMPLETE"
    )
    assert runner._validate_manual_verdict_document(
        document,
        request=runner.strict_json_load(
            run / "worker_macro_fsm_request.json"
        ),
        terminal=machine_incomplete,
        run=run,
    ) == "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"


def test_gate_d_sources_each_use_explicit_cross_version_contract(tmp_path: Path):
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 500
    common = {
        "client_factory": _FakeClient,
        "bundle_builder": lambda *_args, **kwargs: _FakeBundle(
            kwargs["primary_source_version"]
        ),
        "lock_factory": _Lock,
        "process_snapshot_fn": lambda: [],
        "conflict_detector": lambda _rows: [],
        "environment_lock_validator": _valid_environment_identity,
    }
    gate_c_args = _args(tmp_path)
    gate_c_args.output_root = tmp_path / "gate_c"
    baseline = runner.run_trials(
        gate_c_args,
        trial_kind="baseline",
        count=1,
        **common,
    )[0]
    repeats = runner.run_trials(
        gate_c_args,
        trial_kind="repeat",
        count=3,
        **common,
    )
    for attempt in [baseline, *repeats]:
        _review_success(attempt)
    validated_closure = runner._validate_gate_c_closure(
        SimpleNamespace(
            gate_c_baseline_run_dir=baseline.run_dir,
            alignment_path=gate_c_args.alignment_path,
            task_success_table_path=gate_c_args.task_success_table_path,
        ),
        bundle_builder=common["bundle_builder"],
    )

    attempts = []
    for source_version in runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS:
        args = _args(tmp_path)
        args.source_version = source_version
        args.output_root = tmp_path / "cross_version_macro_fsm"
        args.gate_c_baseline_run_dir = baseline.run_dir
        attempts.extend(
            runner.run_trials(
                args,
                trial_kind=runner.GATE_D_TRIAL_KIND,
                count=1,
                **common,
            )
        )
    assert len(attempts) == 3
    assert len(_FakeClient.instances) == 7
    assert all(item.controller_complete and item.shutdown_verified for item in attempts)
    assert [item.source_version for item in attempts] == list(
        runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS
    )
    for attempt in attempts:
        manifest = runner.strict_json_load(attempt.manifest_path)
        request = runner.strict_json_load(manifest["request_path"])
        assert manifest["trial_kind"] == runner.GATE_D_TRIAL_KIND
        assert request["trial_kind"] == runner.GATE_D_TRIAL_KIND
        assert request["source_version"] == attempt.source_version
        assert manifest["manual_review_status"] == (
            "NOT_EVALUATED_PENDING_SHA_BOUND_VIDEO_REVIEW"
        )
        assert manifest["gate_c_closure"] == validated_closure
        assert manifest["gate_c_closure_sha256"] == validated_closure[
            "closure_sha256"
        ]
        assert runner._validate_persisted_gate_c_closure(
            manifest,
            request=request,
            bundle_builder=common["bundle_builder"],
        ) == validated_closure
        assert len(manifest["gate_c_closure"]["repeats"]) == 3
        for row in [
            manifest["gate_c_closure"]["baseline"],
            *manifest["gate_c_closure"]["repeats"],
        ]:
            for key in (
                "runner_manifest_sha256",
                "request_sha256",
                "manual_verdict_sha256",
                "worker_result_sha256",
                "task_inputs_sha256",
                "video_sha256",
            ):
                assert len(row[key]) == 64
            assert row["segment_completion_schema_version"] == (
                runner.SEGMENT_COMPLETION_SCHEMA
            )
            assert Path(row["segment_completion_ledger_path"]).is_file()
            assert len(row["segment_completion_ledger_sha256"]) == 64
            assert row["segment_completion_count"] == (
                row["expected_segment_completion_count"]
            )
            assert row["segment_completion_coverage_complete"] is True
            assert row["segment_completion_coverage_errors"] == []

        tampered_manifest = copy.deepcopy(manifest)
        tampered_manifest["gate_c_closure"]["baseline"]["request_id"] = "tampered"
        with pytest_raises(runner.MacroRunnerContractError):
            runner._validate_persisted_gate_c_closure(
                tampered_manifest,
                request=request,
                bundle_builder=common["bundle_builder"],
            )

    stale_verdict = repeats[0].run_dir / runner.MANUAL_VERDICT_NAME
    original_verdict = stale_verdict.read_bytes()
    stale_verdict.write_bytes(original_verdict + b"\n")
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_gate_c_closure(args, bundle_builder=common["bundle_builder"])
    stale_verdict.write_bytes(original_verdict)

    stale_ledger = Path(
        validated_closure["repeats"][0]["segment_completion_ledger_path"]
    )
    original_ledger = stale_ledger.read_bytes()
    stale_ledger.write_bytes(original_ledger + b"\n")
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_gate_c_closure(args, bundle_builder=common["bundle_builder"])
    stale_ledger.write_bytes(original_ledger)

    def mismatched_builder(*_args, **kwargs):
        source = kwargs["primary_source_version"]
        if source == DEFAULT_PRIMARY_VERSION:
            return _FakeBundle(source)
        return _DifferentGraphBundle(source)

    with pytest_raises(runner.MacroRunnerContractError):
        runner.run_trials(
            args,
            trial_kind=runner.GATE_D_TRIAL_KIND,
            count=1,
            client_factory=_FakeClient,
            bundle_builder=mismatched_builder,
            lock_factory=_Lock,
            process_snapshot_fn=lambda: [],
            conflict_detector=lambda _rows: [],
            environment_lock_validator=_valid_environment_identity,
        )

    with pytest_raises(runner.MacroRunnerContractError):
        runner.run_trials(
            args,
            trial_kind=runner.GATE_D_TRIAL_KIND,
            count=2,
            **common,
        )

    extra = baseline.run_dir.parent.parent / "repeats" / "extra-accepted-repeat"
    extra.mkdir(parents=True)
    runner.atomic_write_json(
        extra / runner.RUNNER_MANIFEST_NAME,
        {
            "schema_version": runner.RUNNER_MANIFEST_SCHEMA,
            "source_version": DEFAULT_PRIMARY_VERSION,
            "trial_kind": "repeat",
            "trial_index": 3,
            "controller_complete": True,
            "shutdown_verified": True,
                "manual_review_status": "MACRO_FSM_TASK_SUCCESS",
        },
    )
    with pytest_raises(runner.MacroRunnerContractError):
        runner._validate_gate_c_closure(args, bundle_builder=common["bundle_builder"])


def test_gate_d_review_revalidates_full_attempt_before_binding(tmp_path: Path):
    bundle, closure, attempt, verdict_path = _gate_d_review_attempt(tmp_path)
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=bundle,
    ), mock.patch.object(
        runner,
        "_validate_persisted_gate_c_closure",
        return_value=closure,
    ) as closure_validator:
        result = runner.review_run(
            run_dir=attempt.run_dir,
            verdict_path=verdict_path,
        )
    assert result["manual_review_status"] == "MACRO_FSM_TASK_SUCCESS"
    closure_validator.assert_called_once()
    manifest = runner.strict_json_load(attempt.manifest_path)
    bound = attempt.run_dir / runner.MANUAL_VERDICT_NAME
    assert manifest["manual_verdict_path"] == str(bound)
    assert manifest["manual_verdict_sha256"] == runner.sha256_file(bound)


def test_gate_c_review_remains_compatible_with_current_rebuilt_bundle(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    _FakeClient.next_pid = 950
    args = _args(tmp_path)
    args.output_root = tmp_path / "gate_c"
    args.alignment_path = runner.DEFAULT_ALIGNMENT_PATH
    bundle = build_gate_c_bundle(
        runner.PROJECT_ROOT,
        alignment_path=args.alignment_path,
        primary_source_version=args.source_version,
    )
    attempt = runner.run_one_macro(
        cli_args=args,
        bundle=bundle,
        source_version=args.source_version,
        trial_kind="baseline",
        trial_index=0,
        client_factory=_FakeClient,
    )
    verdict_path = tmp_path / "gate-c-review-input.json"
    runner.atomic_write_json(verdict_path, _successful_review_document(attempt))
    result = runner.review_run(
        run_dir=attempt.run_dir,
        verdict_path=verdict_path,
    )
    assert result["manual_review_status"] == "MACRO_FSM_TASK_SUCCESS"


def test_gate_d_review_rejects_manifest_runtime_chain_tampering(tmp_path: Path):
    bundle, closure, attempt, verdict_path = _gate_d_review_attempt(tmp_path)
    manifest_path = attempt.manifest_path
    original = manifest_path.read_bytes()
    pristine = runner.strict_json_load(manifest_path)

    mutations = (
        lambda value: value.__setitem__("run_dir", str(tmp_path / "wrong-run")),
        lambda value: value.__setitem__(
            "source_version", runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS[1]
        ),
        lambda value: value.__setitem__("trial_kind", "baseline"),
        lambda value: value.__setitem__("trial_index", 1),
        lambda value: value.__setitem__("bundle_sha256", "0" * 64),
        lambda value: value.__setitem__(
            "manual_review_status", "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
        ),
        lambda value: value["worker_binding"][
            "worker_macro_fsm_preflight"
        ].__setitem__("request_id", "tampered"),
        lambda value: value["terminal"].__setitem__("request_id", "tampered"),
        lambda value: value["terminal_ack"].__setitem__(
            "request_id", "tampered"
        ),
        lambda value: value["shutdown_outcome"].__setitem__(
            "forced_termination", True
        ),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(pristine)
        mutate(tampered)
        runner.atomic_write_json(manifest_path, tampered)
        _assert_gate_d_review_rejected(
            attempt=attempt,
            verdict_path=verdict_path,
            bundle=bundle,
            closure=closure,
        )
        manifest_path.write_bytes(original)


def test_gate_d_review_rejects_request_and_rebuilt_bundle_tampering(tmp_path: Path):
    bundle, closure, attempt, verdict_path = _gate_d_review_attempt(tmp_path)
    manifest_path = attempt.manifest_path
    request_path = attempt.run_dir / "worker_macro_fsm_request.json"
    original_manifest_bytes = manifest_path.read_bytes()
    original_request_bytes = request_path.read_bytes()
    pristine_manifest = runner.strict_json_load(manifest_path)
    pristine_request = runner.strict_json_load(request_path)

    escaped_request = tmp_path / "escaped-worker-request.json"
    escaped_request.write_bytes(original_request_bytes)
    escaped_manifest = copy.deepcopy(pristine_manifest)
    escaped_manifest["request_path"] = str(escaped_request)
    escaped_manifest["request_sha256"] = runner.sha256_file(escaped_request)
    runner.atomic_write_json(manifest_path, escaped_manifest)
    _assert_gate_d_review_rejected(
        attempt=attempt,
        verdict_path=verdict_path,
        bundle=bundle,
        closure=closure,
    )
    manifest_path.write_bytes(original_manifest_bytes)

    alternate_request = attempt.run_dir / "alternate-worker-request.json"
    alternate_request.write_bytes(original_request_bytes)
    alternate_manifest = copy.deepcopy(pristine_manifest)
    alternate_manifest["request_path"] = str(alternate_request)
    alternate_manifest["request_sha256"] = runner.sha256_file(alternate_request)
    runner.atomic_write_json(manifest_path, alternate_manifest)
    _assert_gate_d_review_rejected(
        attempt=attempt,
        verdict_path=verdict_path,
        bundle=bundle,
        closure=closure,
    )
    manifest_path.write_bytes(original_manifest_bytes)

    stale_manifest = copy.deepcopy(pristine_manifest)
    stale_manifest["request_sha256"] = "0" * 64
    runner.atomic_write_json(manifest_path, stale_manifest)
    _assert_gate_d_review_rejected(
        attempt=attempt,
        verdict_path=verdict_path,
        bundle=bundle,
        closure=closure,
    )
    manifest_path.write_bytes(original_manifest_bytes)

    request_mutations = (
        lambda value: value.__setitem__("schema_version", "wrong"),
        lambda value: value.__setitem__(
            "source_version", runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS[1]
        ),
        lambda value: value.__setitem__("trial_kind", "repeat"),
        lambda value: value.__setitem__("trial_index", 1),
        lambda value: value.__setitem__("run_dir", str(tmp_path / "wrong-run")),
        lambda value: value.__setitem__("bundle_sha256", "0" * 64),
    )
    for mutate in request_mutations:
        tampered_request = copy.deepcopy(pristine_request)
        mutate(tampered_request)
        runner.atomic_write_json(request_path, tampered_request)
        tampered_manifest = copy.deepcopy(pristine_manifest)
        tampered_manifest["request_sha256"] = runner.sha256_file(request_path)
        runner.atomic_write_json(manifest_path, tampered_manifest)
        _assert_gate_d_review_rejected(
            attempt=attempt,
            verdict_path=verdict_path,
            bundle=bundle,
            closure=closure,
        )
        request_path.write_bytes(original_request_bytes)
        manifest_path.write_bytes(original_manifest_bytes)

    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=_DifferentGraphBundle(attempt.source_version),
    ), mock.patch.object(
        runner,
        "_validate_persisted_gate_c_closure",
        return_value=closure,
    ), pytest_raises(runner.MacroRunnerContractError):
        runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict_path)
    assert not (attempt.run_dir / runner.MANUAL_VERDICT_NAME).exists()


def test_cli_requires_baseline_repeat_and_exact_three_count():
    parser = runner.build_parser()
    baseline = parser.parse_args(["run"])
    assert baseline.command == "run"
    assert baseline.source_version == DEFAULT_PRIMARY_VERSION
    repeat = parser.parse_args(
        ["repeat", "--baseline-run-dir", "C:/run"]
    )
    assert repeat.command == "repeat"
    assert repeat.count == 3
    review = parser.parse_args(
        ["review", "--run-dir", "C:/run", "--verdict", "C:/verdict.json"]
    )
    assert review.command == "review"
    gate_d_source = runner.AUTHORIZED_GATE_D_SOURCE_VERSIONS[0]
    cross_version = parser.parse_args(
        [
            "cross-version",
            "--source-version",
            gate_d_source,
            "--gate-c-baseline-run-dir",
            "C:/gate-c-baseline",
        ]
    )
    assert cross_version.command == "cross-version"
    assert cross_version.source_version == gate_d_source
    assert cross_version.output_root == runner.DEFAULT_GATE_D_OUTPUT_ROOT
    assert cross_version.gate_c_baseline_run_dir == Path("C:/gate-c-baseline")
    with pytest_raises(SystemExit):
        parser.parse_args(["cross-version"])
    with pytest_raises(SystemExit):
        parser.parse_args(
            ["cross-version", "--source-version", gate_d_source]
        )
    with pytest_raises(SystemExit):
        parser.parse_args(
            [
                "cross-version",
                "--source-version",
                DEFAULT_PRIMARY_VERSION,
                "--gate-c-baseline-run-dir",
                "C:/gate-c-baseline",
            ]
        )
    with pytest_raises(SystemExit):
        parser.parse_args(["run", "--source-version", "v009_manual"])
    with pytest_raises(ValueError):
        runner.allocate_run_paths(
            Path("C:/invalid"),
            source_version="v009_manual",
            trial_kind="baseline",
            trial_index=0,
            bundle_sha256="c" * 64,
        )


def test_fast_close_requires_every_bound_row_and_validates_optional_returned_pair(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    args = _args(tmp_path)
    attempt = runner.run_trials(
        args,
        trial_kind="baseline",
        count=1,
        client_factory=_FakeClient,
        bundle_builder=lambda *_args, **_kwargs: _FakeBundle(),
        lock_factory=_Lock,
        process_snapshot_fn=lambda: [],
        conflict_detector=lambda _rows: [],
        environment_lock_validator=_valid_environment_identity,
    )[0]
    manifest = runner.strict_json_load(attempt.manifest_path)
    outcome = manifest["shutdown_outcome"]
    binding = manifest["worker_binding"]
    shutdown_request_id = outcome["request_id"]
    macro_request_id = manifest["request_id"]
    runner.validate_fast_shutdown(
        outcome,
        shutdown_request_id=shutdown_request_id,
        macro_request_id=macro_request_id,
        worker_binding=binding,
    )
    mutations = []
    missing_receipt = copy.deepcopy(outcome)
    missing_receipt["close_requested_receipt"] = {}
    mutations.append(missing_receipt)
    bad_schema = copy.deepcopy(outcome)
    bad_schema["shutdown_ack"]["schema_version"] = "wrong"
    mutations.append(bad_schema)
    bad_runtime = copy.deepcopy(outcome)
    bad_runtime["close_requested_ack"]["runtime_version"] = "wrong"
    mutations.append(bad_runtime)
    bad_root = copy.deepcopy(outcome)
    bad_root["close_requested_receipt"]["root_state_write_count"] = 1
    mutations.append(bad_root)
    half_returned = copy.deepcopy(outcome)
    half_returned["close_returned_receipt"] = {}
    mutations.append(half_returned)
    for malformed in mutations:
        with pytest_raises(runner.MacroRunnerContractError):
            runner.validate_fast_shutdown(
                malformed,
                shutdown_request_id=shutdown_request_id,
                macro_request_id=macro_request_id,
                worker_binding=binding,
            )


def test_posture_incomplete_success_is_not_repeat_eligible_but_review_exit_zero(
    tmp_path: Path,
):
    _FakeClient.instances.clear()
    args = _args(tmp_path)
    attempt = runner.run_trials(
        args,
        trial_kind="baseline",
        count=1,
        client_factory=_FakeClient,
        bundle_builder=lambda *_args, **_kwargs: _FakeBundle(),
        lock_factory=_Lock,
        process_snapshot_fn=lambda: [],
        conflict_detector=lambda _rows: [],
        environment_lock_validator=_valid_environment_identity,
    )[0]
    document = runner.strict_json_load(
        attempt.run_dir / runner.MANUAL_VERDICT_TEMPLATE_NAME
    )
    document["review_complete"] = True
    document["reviewed_utc"] = "2026-08-14T20:00:00+00:00"
    document["verdict"].update(
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
    )
    verdict_path = tmp_path / "repeat-verdict.json"
    runner.atomic_write_json(verdict_path, document)
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=_FakeBundle(attempt.source_version),
    ):
        reviewed = runner.review_run(
            run_dir=attempt.run_dir,
            verdict_path=verdict_path,
        )
    assert reviewed["manual_review_status"] == (
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
    )
    args.baseline_run_dir = attempt.run_dir
    with pytest_raises(runner.MacroRunnerContractError):
        with mock.patch.object(
            runner, "build_gate_c_bundle", return_value=_FakeBundle()
        ):
            runner._validate_repeat_baseline(args)
    bound_verdict = attempt.run_dir / runner.MANUAL_VERDICT_NAME
    bound_verdict.write_text("{}\n", encoding="utf-8")
    with pytest_raises(runner.MacroRunnerContractError):
        with mock.patch.object(
            runner, "build_gate_c_bundle", return_value=_FakeBundle()
        ):
            runner._validate_repeat_baseline(args)
    with mock.patch.object(
        runner,
        "review_run",
        return_value={
            "manual_review_status": "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
        },
    ):
        assert runner.main(
            ["review", "--run-dir", str(attempt.run_dir), "--verdict", str(tmp_path / "v.json")]
        ) == 0


def test_manual_review_binds_task_inputs_and_preserves_posture_incomplete(tmp_path: Path):
    _FakeClient.instances.clear()
    args = _args(tmp_path)
    attempt = runner.run_trials(
        args,
        trial_kind="baseline",
        count=1,
        client_factory=_FakeClient,
        bundle_builder=lambda *_args, **_kwargs: _FakeBundle(),
        lock_factory=_Lock,
        process_snapshot_fn=lambda: [],
        conflict_detector=lambda _rows: [],
        environment_lock_validator=_valid_environment_identity,
    )[0]
    template_path = attempt.run_dir / runner.MANUAL_VERDICT_TEMPLATE_NAME
    document = runner.strict_json_load(template_path)
    document["review_complete"] = True
    document["reviewed_utc"] = "2026-08-14T20:00:00+00:00"
    document["verdict"].update(
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
    )
    verdict = tmp_path / "verdict.json"
    runner.atomic_write_json(verdict, document)
    task_inputs = Path(document["task_inputs_path"])
    original_task_inputs = task_inputs.read_bytes()
    task_inputs.write_text("{}\n", encoding="utf-8")
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=_FakeBundle(attempt.source_version),
    ), pytest_raises(runner.MacroRunnerContractError):
        runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict)
    assert not (attempt.run_dir / runner.MANUAL_VERDICT_NAME).exists()
    task_inputs.write_bytes(original_task_inputs)
    with mock.patch.object(
        runner,
        "build_gate_c_bundle",
        return_value=_FakeBundle(attempt.source_version),
    ):
        result = runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict)
    assert result["manual_review_status"] == (
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
    )
    with pytest_raises(runner.MacroRunnerContractError):
        runner.review_run(run_dir=attempt.run_dir, verdict_path=verdict)


class pytest_raises:
    def __init__(self, error):
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, kind, _value, _tb):
        if kind is None:
            raise AssertionError(f"expected {self.error.__name__}")
        return bool(issubclass(kind, self.error))
