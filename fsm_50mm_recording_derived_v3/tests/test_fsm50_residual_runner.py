from __future__ import annotations

import argparse
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES

from fsm_50mm_recording_derived_v3 import fsm50_residual_runner as runner
from fsm_50mm_recording_derived_v3.fsm50_macro_controller import (
    build_gate_c_bundle,
)
from fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session import (
    load_worker_residual_macro_fsm_request,
)


SOURCE = "v003_20260805_224517_157723_manual"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    runner._atomic_write_json(path, value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _targets(value: float = 0.0) -> tuple[dict[str, float], dict[str, float]]:
    servos = {name: 0.0 for name in SERVO_JOINT_NAMES}
    servos[SERVO_JOINT_NAMES[0]] = float(value)
    wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    return servos, wheels


def _provenance() -> dict[str, Any]:
    return {
        "kind": "SOURCE_ACTION",
        "dispatch_kind": "segment_start",
        "sequence_index": 0,
        "source_action_identity": "2" * 64,
        "source_version": SOURCE,
        "source_segment_index": 0,
        "source_step_index": 1,
        "source_time_s": 0.0,
        "source_event_indices": [0],
        "commands": ["servo front_left_hip 1.0"],
    }


def _completion_spec() -> dict[str, Any]:
    return {
        "segment_index": 0,
        "source_step": 1,
        "source_step_id": "step_001",
        "servo_targets_deg": {SERVO_JOINT_NAMES[0]: 1.0},
        "servo_duration_s": 0.1,
        "wheel_active_duration_s": 0.0,
        "explicit_hold_s": 0.0,
        "servo_tolerance_deg": 1.0,
        "recorded_servo_residual_deg": {},
        "legacy_missing_endpoint": False,
    }


def _zero_transform(request: Any, servos: Mapping[str, float], wheels: Mapping[str, float]) -> dict[str, Any]:
    transform = {
        "schema_version": "fsm50.direct_command_residual_transform.v1",
        "action_schema": "fsm50.direct_command_residual_action.v1",
        "action_names": list(runner.RESIDUAL_ACTION_NAMES),
        "source_version": request.source_version,
        "profile_strategy": "PRIMARY_PROFILE",
        "macro_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
        "subphase": "PRELOAD",
        "contract_sha256": "3" * 64,
        "raw_normalized_action": [0.0] * 12,
        "clipped_normalized_action": [0.0] * 12,
        "phase_mask": [True] * 12,
        "residual_min_command_units": [-1.0] * 12,
        "residual_max_command_units": [1.0] * 12,
        "previous_applied_residual": [0.0] * 12,
        "requested_residual": [0.0] * 12,
        "rate_limited_residual": [0.0] * 12,
        "applied_residual": [0.0] * 12,
        "nominal_servo_targets_deg": dict(servos),
        "nominal_wheel_targets_rad_s": dict(wheels),
        "applied_servo_targets_deg": dict(servos),
        "applied_wheel_targets_rad_s": dict(wheels),
        "clip_reasons_by_action": {
            name: [] for name in runner.RESIDUAL_ACTION_NAMES
        },
        "zero_identity": True,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
        "core_transform_sha256": "4" * 64,
        "nominal_command_epoch": 1,
        "physical_command_epoch_before": 0,
        "force_zero_residual": False,
        "force_zero_wheels": False,
    }
    transform["evidence_sha256"] = runner._canonical_json_sha256(transform)
    return transform


def _base_source_row() -> dict[str, Any]:
    servos, wheels = _targets(1.0)
    row = {key: None for key in runner._SOURCE_ROW_KEYS}
    row.update(
        {
            "schema_version": "fsm50.source_action_consumption.v1",
            "source_action_index": 0,
            "expected_source_action_count": 1,
            "sim_time_s": 1.0,
            "sim_step": 10,
            "macro_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
            "subphase": "PRELOAD",
            "profile_id": "profile",
            "profile_source_version": SOURCE,
            "profile_strategy": "PRIMARY_PROFILE",
            "source_plan_sha256": "5" * 64,
            "profile_library_sha256": "library",
            "bundle_sha256": "bundle",
            "command_provenance": _provenance(),
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
            "target_changed": True,
            "dispatch_epoch": 1,
            "physical_dispatch_required": True,
            "physical_dispatch_applied": True,
            "physical_dispatch_index": 0,
            "batch_id": "base-request:macro:000001",
            "n_plus_one_verified": True,
            "n_plus_one_verified_sim_step": 11,
            "n_plus_one_readback_sha256": "6" * 64,
            "pre_action_verified_command_epoch": 0,
            "pre_action_verified_readback_sha256": "7" * 64,
            "nominal_target_changed": True,
            "applied_target_changed": True,
            "physical_command_epoch": 1,
            "applied_servo_targets_deg": servos,
            "applied_wheel_targets_rad_s": wheels,
            "residual_transform_sha256": "4" * 64,
        }
    )
    return row


def _base_completion_row() -> dict[str, Any]:
    spec = _completion_spec()
    before, _ = _targets(0.0)
    row = {key: None for key in runner._COMPLETION_ROW_KEYS}
    row.update(
        {
            "schema_version": runner.SEGMENT_COMPLETION_SCHEMA,
            "segment_completion_index": 0,
            "source_version": SOURCE,
            "profile_id": "profile",
            "profile_source_version": SOURCE,
            "owner_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
            "source_plan_sha256": "5" * 64,
            "source_plan_payload_sha256": "8" * 64,
            "accepted_steps_sha256": "9" * 64,
            "source_segment_index": 0,
            "source_step_index": 1,
            "source_step_id": "step_001",
            "completion_spec": spec,
            "effective_completion_spec": copy.deepcopy(spec),
            "dynamic_servo_duration_s": 0.1,
            "effective_servo_reference_velocity_deg_s": 150.0,
            "pre_action_canonical_servo_targets_deg": before,
            "start_source_action_identity": "2" * 64,
            "source_action_consumption_index": 0,
            "start_command_epoch": 1,
            "start_sim_step": 10,
            "start_sim_time_s": 1.0,
            "start_physical_dispatch": True,
            "start_batch_id": "base-request:macro:000001",
            "start_first_physics_step": 11,
            "start_readback_verified": True,
            "start_readback_verified_sim_step": 11,
            "start_readback_sha256": "6" * 64,
            "retained_epoch_same_target": False,
            "tracking_begin_count": 1,
            "tracking_begin_sim_step": 10,
            "tracking_begin_evidence": {
                "called": True,
                "sparse_joint_names": [SERVO_JOINT_NAMES[0]],
            },
            "tracking_end_count": 1,
            "tracking_end_attempt_count": 1,
            "tracking_lifecycle_closed": True,
            "tracking_end_sim_step": 18,
            "tracking_end_sim_time_s": 1.1,
            "tracking_end_reason": "COMPLETE",
            "tracking_end_evidence": {
                "ended": True,
                "tracking_completion_deferred": False,
            },
            "observation_decisions": [],
            "last_decision": {"kind": "COMPLETE"},
            "last_decision_sha256": "a" * 64,
            "wheel_stop": None,
            "terminal_kind": "COMPLETE",
            "terminal_sim_step": 18,
            "terminal_sim_time_s": 1.1,
            "terminal_decision_sha256": "a" * 64,
            "effective_completion_servo_targets_deg": dict(
                spec["servo_targets_deg"]
            ),
            "effective_completion_targets_sha256": runner._canonical_json_sha256(
                spec["servo_targets_deg"]
            ),
            "nominal_completion_spec_sha256": runner._canonical_json_sha256(spec),
            "pre_action_applied_servo_targets_deg": before,
            "start_physical_command_epoch": 1,
            "latched_servo_residual_deg": {SERVO_JOINT_NAMES[0]: 0.0},
            "nominal_target_changed": True,
            "applied_target_changed": True,
        }
    )
    return row


def _empty_sparse_completion_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    spec = copy.deepcopy(result["completion_spec"])
    spec["servo_targets_deg"] = {}
    spec["servo_duration_s"] = 0.0
    result.update(
        {
            "completion_spec": spec,
            "effective_completion_spec": copy.deepcopy(spec),
            "dynamic_servo_duration_s": 0.0,
            "effective_completion_servo_targets_deg": {},
            "effective_completion_targets_sha256": (
                runner._canonical_json_sha256({})
            ),
            "nominal_completion_spec_sha256": (
                runner._canonical_json_sha256(spec)
            ),
            "latched_servo_residual_deg": {},
            "tracking_begin_count": 0,
            "tracking_begin_evidence": {
                "called": False,
                "sparse_joint_names": [],
            },
            "tracking_end_count": 0,
            "tracking_end_attempt_count": 0,
            "tracking_end_evidence": {
                "ended": True,
                "required": False,
                "reason": "segment has no sparse servo endpoints",
            },
        }
    )
    return result


def _transition_row() -> dict[str, Any]:
    row = {key: None for key in runner._TRANSITION_ROW_KEYS}
    row.update(
        {
            "schema_version": "fsm50.macro_transition_evidence.v1",
            "transition_index": 0,
            "sim_time_s": 0.9,
            "sim_step": 9,
            "from_state": "",
            "to_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
            "subphase": "PRELOAD",
            "profile_id": "profile",
            "profile_source_version": SOURCE,
            "profile_strategy": "PRIMARY_PROFILE",
            "command_epoch": 0,
            "phase_elapsed_s": 0.0,
            "profile_fraction": 0.0,
            "events": ["RESET:S0_INITIALIZE"],
            "reason": "test deterministic transition",
            "guard_evidence": {},
            "retry_count": 0,
            "observation_sha256": "b" * 64,
        }
    )
    return row


def _completed_result() -> dict[str, Any]:
    return {
        "source_version": SOURCE,
        "expected_step_count": 1,
        "expected_segment_count": 1,
        "completed_segment_count": 1,
        "expected_segment_completion_count": 1,
        "segment_completion_count": 1,
        "source_action_coverage_complete": True,
        "segment_completion_coverage_complete": True,
        "consumed_segment_start_count": 1,
        "dispatch_complete": True,
        "scheduler_complete": True,
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
        "body_crossed_front_face": True,
        "required_leg_lift_completed": {},
        "final_recoverable": True,
        "wheel_drive_up_without_required_lift": False,
        "nonfinite_core_state_detected": False,
    }


def _write_reference_projection_fixture(root: Path) -> dict[str, Any]:
    root.mkdir()
    source = _base_source_row()
    for key in set(source) - (runner._SOURCE_ROW_KEYS - {
        "nominal_target_changed",
        "applied_target_changed",
        "physical_command_epoch",
        "applied_servo_targets_deg",
        "applied_wheel_targets_rad_s",
        "residual_transform_sha256",
    }):
        source.pop(key, None)
    dispatch = {key: None for key in runner._DISPATCH_ROW_KEYS}
    servos, wheels = _targets(1.0)
    dispatch.update(
        {
            "dispatch_index": 0,
            "macro_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
            "subphase": "PRELOAD",
            "profile_id": "profile",
            "profile_source_version": SOURCE,
            "profile_strategy": "PRIMARY_PROFILE",
            "command_epoch": 1,
            "command_provenance": _provenance(),
            "source_action_consumption_index": 0,
            "changed_servo_names": [SERVO_JOINT_NAMES[0]],
            "changed_wheel_names": [],
            "concurrent": False,
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
        }
    )
    for key in (
        "nominal_command_epoch",
        "physical_command_epoch",
        "nominal_target_changed",
        "applied_target_changed",
        "dispatch_cause",
        "nominal_servo_targets_deg",
        "nominal_wheel_targets_rad_s",
        "residual_policy_id",
        "residual_policy_sha256",
        "residual_transform",
    ):
        dispatch.pop(key, None)
    completion = _base_completion_row()
    for key in set(completion) - set(runner.macro_runner._SEGMENT_COMPLETION_ROW_KEYS):
        completion.pop(key, None)
    transition = _transition_row()
    _write_jsonl(root / "source.jsonl", [source])
    _write_jsonl(root / "dispatch.jsonl", [dispatch])
    _write_jsonl(root / "completion.jsonl", [completion])
    _write_jsonl(root / "transition.jsonl", [transition])
    inputs = {
        "schema_version": "fsm50.macro_task_inputs.v1",
        "completed_result": _completed_result(),
    }
    _write_json(root / "macro_task_inputs.json", inputs)
    result = {
        "schema_version": "fsm50.worker_macro_fsm_session.v1",
        "source_version": SOURCE,
        "macro_fsm_complete": True,
        "controller_terminal_outcome": "TASK_SUCCESS_POSTURE_INCOMPLETE",
        "controller_terminal_reason": "test success",
        "expected_source_action_count": 1,
        "source_action_consumption_count": 1,
        "command_dispatch_count": 1,
        "source_action_coverage_complete": True,
        "segment_completion_count": 1,
        "expected_segment_completion_count": 1,
        "segment_completion_coverage_complete": True,
        "transition_count": 1,
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
        "source_action_consumption_path": str(root / "source.jsonl"),
        "dispatch_ledger_path": str(root / "dispatch.jsonl"),
        "segment_completion_ledger_path": str(root / "completion.jsonl"),
        "transition_evidence_path": str(root / "transition.jsonl"),
    }
    _write_json(root / "worker_macro_fsm_result.json", result)
    return runner._semantic_projection(root)


def _write_live_artifacts(request: Any, worker: Mapping[str, Any]) -> dict[str, Any]:
    run = request.run_dir
    source = _base_source_row()
    source["profile_library_sha256"] = request.profile_library_sha256
    source["bundle_sha256"] = request.bundle_sha256
    source["batch_id"] = f"{request.base_request.request_id}:macro:000001"
    servos, wheels = _targets(1.0)
    transform = _zero_transform(request, servos, wheels)
    source["residual_transform_sha256"] = transform["core_transform_sha256"]
    metadata = {
        "nominal_command_epoch": 1,
        "physical_command_epoch": 1,
        "dispatch_cause": "NOMINAL_AND_RESIDUAL",
        "nominal_servo_targets_deg": servos,
        "nominal_wheel_targets_rad_s": wheels,
        "residual_transform_sha256": transform["core_transform_sha256"],
        "residual_evidence_sha256": transform["evidence_sha256"],
    }
    dispatch = {key: None for key in runner._DISPATCH_ROW_KEYS}
    dispatch.update(
        {
            "schema_version": "fsm50.macro_dispatch_ledger.v1",
            "dispatch_index": 0,
            "sim_time_s": 1.0,
            "sim_step": 10,
            "macro_state": "S1_APPROACH_AND_PRE_FR_SHIFT",
            "subphase": "PRELOAD",
            "profile_id": "profile",
            "profile_source_version": SOURCE,
            "profile_strategy": "PRIMARY_PROFILE",
            "command_epoch": 1,
            "command_provenance": _provenance(),
            "source_action_consumption_index": 0,
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
            "changed_servo_names": [SERVO_JOINT_NAMES[0]],
            "changed_wheel_names": [],
            "concurrent": False,
            "batch_id": source["batch_id"],
            "ack": {
                "batch_id": source["batch_id"],
                "source": "fsm50_macro_controller",
                "applied_sim_step": 10,
                "first_physics_step": 11,
                "servo_targets_applied": servos,
                "wheel_targets_applied": wheels,
                "recording_metadata": metadata,
            },
            "n_plus_one_verified": True,
            "n_plus_one_verified_sim_step": 11,
            "n_plus_one_readback_sha256": "6" * 64,
            "nominal_command_epoch": 1,
            "physical_command_epoch": 1,
            "nominal_target_changed": True,
            "applied_target_changed": True,
            "dispatch_cause": "NOMINAL_AND_RESIDUAL",
            "nominal_servo_targets_deg": servos,
            "nominal_wheel_targets_rad_s": wheels,
            "residual_policy_id": request.policy_id,
            "residual_policy_sha256": request.policy_sha256,
            "residual_transform": transform,
        }
    )
    completion = _base_completion_row()
    completion["profile_source_version"] = request.source_version
    completion["source_version"] = request.source_version
    completion["start_batch_id"] = source["batch_id"]
    transition = _transition_row()
    transition["profile_source_version"] = request.source_version
    _write_jsonl(run / "source.jsonl", [source])
    _write_jsonl(run / "macro_dispatch_ledger.jsonl", [dispatch])
    _write_jsonl(run / "completion.jsonl", [completion])
    _write_jsonl(run / "transition.jsonl", [transition])

    safe_servos = servos
    zero_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    safe_batch = f"{request.base_request.request_id}:safe-stop:0001"
    safe_ack = {
        "batch_id": safe_batch,
        "source": "fsm50_macro_safe_stop",
        "applied_sim_step": 20,
        "first_physics_step": 21,
        "servo_targets_applied": safe_servos,
        "wheel_targets_applied": zero_wheels,
        "recording_metadata": {"command_epoch": 1},
    }
    safe_readback = {
        "batch_id": safe_batch,
        "sim_step": 21,
        "command_epoch": 1,
        "canonical_servo_targets_deg": safe_servos,
        "canonical_wheel_targets_rad_s": zero_wheels,
        "actual_servo_drive_targets_rad": {
            name: 0.0 for name in SERVO_JOINT_NAMES
        },
        "actual_wheel_drive_targets_rad_s": zero_wheels,
        "adapter_runtime_instance_id": worker["adapter_runtime_instance_id"],
        "root_state_write_count": 0,
        "physics_dt_s": 1.0 / 120.0,
    }
    safe_readback_sha256 = runner._canonical_json_sha256(safe_readback)
    last_readback = copy.deepcopy(safe_readback)
    last_readback.update({"batch_id": "", "command_epoch": None, "sim_step": 22})
    completed = _completed_result()
    completed["direct_command_residual"] = {
        "enabled": True,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
        "transform_count": 2,
        "physical_command_epoch": 2,
        "last_verified_physical_command_epoch": 2,
        "last_transform": transform,
    }
    base_inputs = {
        "schema_version": "fsm50.macro_task_inputs.v1",
        "completed_result": completed,
    }
    base_inputs_path = run / "macro_task_inputs.json"
    _write_json(base_inputs_path, base_inputs)
    base_result = {
        "schema_version": "fsm50.worker_macro_fsm_session.v1",
        "execution_mode": "normal_development",
        "request_id": request.base_request.request_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "run_dir": str(run),
        "macro_fsm_complete": True,
        "controller_terminal_outcome": "TASK_SUCCESS_POSTURE_INCOMPLETE",
        "controller_terminal_reason": "test success",
        "expected_source_action_count": 1,
        "source_action_consumption_count": 1,
        "command_dispatch_count": 1,
        "source_action_coverage_complete": True,
        "segment_completion_count": 1,
        "expected_segment_completion_count": 1,
        "segment_completion_coverage_complete": True,
        "segment_completion_coverage_errors": [],
        "transition_count": 1,
        "source_action_consumption_path": str(run / "source.jsonl"),
        "dispatch_ledger_path": str(run / "macro_dispatch_ledger.jsonl"),
        "segment_completion_ledger_path": str(run / "completion.jsonl"),
        "transition_evidence_path": str(run / "transition.jsonl"),
        "task_inputs_path": str(base_inputs_path),
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
        "safe_stop_error": "",
        "safe_stop_ack": safe_ack,
        "safe_stop_readback": safe_readback,
        "safe_stop_readback_sha256": safe_readback_sha256,
        "last_target_readback": last_readback,
        "video_writer_quiesced": True,
        "worker_pid": worker["worker_pid"],
        "worker_session_id": worker["worker_session_id"],
        "adapter_runtime_instance_id": worker["adapter_runtime_instance_id"],
        "root_state_write_count": 0,
        "physics_dt_s": 1.0 / 120.0,
        "direct_command_residual": {
            "enabled": True,
            "policy_id": request.policy_id,
            "policy_sha256": request.policy_sha256,
            "transform_count": 2,
            "physical_command_epoch": 2,
            "last_verified_physical_command_epoch": 2,
            "last_transform": transform,
        },
        "error": "",
    }
    base_result_path = run / "worker_macro_fsm_result.json"
    _write_json(base_result_path, base_result)
    video_path = run / "viewport.mp4"
    video_path.write_bytes(b"fake-live-viewport-video")
    base_terminal = {
        "type": "macro_fsm_complete",
        "operation": "macro_fsm",
        "phase": "MACRO_FSM_COMPLETE",
        "accepted": True,
        "macro_fsm_complete": True,
        "request_id": request.base_request.request_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "run_dir": str(run),
        "worker_pid": worker["worker_pid"],
        "worker_session_id": worker["worker_session_id"],
        "adapter_runtime_instance_id": worker["adapter_runtime_instance_id"],
        "root_state_write_count": 0,
        "video_writer_quiesced": True,
        "video_path": str(video_path),
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
        "safe_stop_error": "",
        "safe_stop_ack": safe_ack,
        "safe_stop_readback": safe_readback,
        "safe_stop_readback_sha256": safe_readback_sha256,
        "last_target_readback": last_readback,
        "physics_dt_s": 1.0 / 120.0,
        "task_inputs_path": str(base_inputs_path),
        "worker_result_path": str(base_result_path),
        "task_inputs": base_inputs,
        "error": "",
    }
    evidence = {
        "schema_version": runner.GATE_E_COMMAND_EVIDENCE_SCHEMA,
        "policy_kind": runner.POLICY_KIND,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
        "transform_count": 2,
        "physical_command_epoch": 2,
        "last_verified_physical_command_epoch": 2,
        "last_transform_available": True,
        "last_transform_zero_identity_verified": True,
        "last_transform_sha256": "",
        "last_transform": transform,
        "dispatch_ledger_path": str(run / "macro_dispatch_ledger.jsonl"),
        "dispatch_ledger_sha256": runner.sha256_file(
            run / "macro_dispatch_ledger.jsonl"
        ),
        "dispatch_row_count": 1,
        "residual_dispatch_zero_identity_count": 1,
        "all_durable_residual_dispatches_zero_identity": True,
        "checkpoint_loaded": False,
        "ppo_training_performed": False,
    }
    outer_inputs = {
        "schema_version": runner.GATE_E_TASK_INPUTS_SCHEMA,
        "execution_mode": runner.EXECUTION_MODE,
        "operation": runner.OPERATION,
        "gate_e_zero_residual": request.gate_e_identity(payload_role="task_inputs"),
        "base_task_inputs_path": str(base_inputs_path),
        "base_task_inputs_sha256": runner.sha256_file(base_inputs_path),
        "base_task_inputs": base_inputs,
        "residual_command_evidence": evidence,
    }
    outer_inputs_path = run / runner.GATE_E_TASK_INPUTS_NAME
    _write_json(outer_inputs_path, outer_inputs)
    outer_result = {
        "schema_version": runner.GATE_E_WORKER_RESULT_SCHEMA,
        "execution_mode": runner.EXECUTION_MODE,
        "operation": runner.OPERATION,
        "gate_e_zero_residual": request.gate_e_identity(payload_role="worker_result"),
        "base_worker_result_path": str(base_result_path),
        "base_worker_result_sha256": runner.sha256_file(base_result_path),
        "base_worker_result": base_result,
        "task_inputs_path": str(outer_inputs_path),
        "task_inputs_sha256": runner.sha256_file(outer_inputs_path),
        "task_inputs": outer_inputs,
        "residual_command_evidence": evidence,
        "macro_fsm_complete": True,
        "error": "",
    }
    outer_result_path = run / runner.GATE_E_WORKER_RESULT_NAME
    _write_json(outer_result_path, outer_result)
    terminal = copy.deepcopy(base_terminal)
    terminal.update(
        {
            "operation": runner.OPERATION,
            "execution_mode": runner.EXECUTION_MODE,
            "request_id": request.request_id,
            "request_identity_sha256": request.request_identity_sha256,
            "residual_macro_fsm_request_id": request.request_id,
            "base_macro_fsm_request_id": request.base_request.request_id,
            "base_macro_fsm_request_path": str(request.base_request_path),
            "base_macro_fsm_request_sha256": request.base_request_sha256,
            "base_macro_fsm_terminal": base_terminal,
            "base_macro_fsm_terminal_sha256": runner._canonical_json_sha256(
                base_terminal
            ),
            "base_task_inputs_path": str(base_inputs_path),
            "base_task_inputs_sha256": runner.sha256_file(base_inputs_path),
            "base_worker_result_path": str(base_result_path),
            "base_worker_result_sha256": runner.sha256_file(base_result_path),
            "task_inputs_path": str(outer_inputs_path),
            "task_inputs_sha256": runner.sha256_file(outer_inputs_path),
            "worker_result_path": str(outer_result_path),
            "worker_result_sha256": runner.sha256_file(outer_result_path),
            "task_inputs": outer_inputs,
            "residual_command_evidence": evidence,
            "gate_e_zero_residual": request.gate_e_identity(
                payload_role="terminal"
            ),
        }
    )
    return terminal


def _write_generated_dispatch_then_retained_source_artifacts(
    request: Any, worker: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    outer_terminal = _write_live_artifacts(request, worker)
    run = request.run_dir
    sources = runner._strict_jsonl(run / "source.jsonl", label="source fixture")
    dispatches = runner._strict_jsonl(
        run / "macro_dispatch_ledger.jsonl", label="dispatch fixture"
    )
    completions = runner._strict_jsonl(
        run / "completion.jsonl", label="completion fixture"
    )

    sources[0]["expected_source_action_count"] = 2
    servos, wheels = _targets(2.0)
    transform = _zero_transform(request, servos, wheels)
    transform.update(
        {
            "macro_state": "S2_FR_TRAVERSE",
            "subphase": "FACE_CLEAR",
            "nominal_command_epoch": 2,
            "physical_command_epoch_before": 1,
            "core_transform_sha256": "c" * 64,
        }
    )
    transform.pop("evidence_sha256")
    transform["evidence_sha256"] = runner._canonical_json_sha256(transform)
    generated_provenance = copy.deepcopy(_provenance())
    generated_provenance.update(
        {
            "kind": "BOUNDARY_ZERO_WHEELS",
            "dispatch_kind": "",
            "sequence_index": 1,
            "source_action_identity": "d" * 64,
            "source_segment_index": 1,
            "source_step_index": 2,
            "commands": ["generated zero-wheel boundary"],
        }
    )
    generated_batch = f"{request.base_request.request_id}:macro:000002"
    metadata = {
        "nominal_command_epoch": 2,
        "physical_command_epoch": 2,
        "dispatch_cause": "NOMINAL_AND_RESIDUAL",
        "nominal_servo_targets_deg": servos,
        "nominal_wheel_targets_rad_s": wheels,
        "residual_transform_sha256": transform["core_transform_sha256"],
        "residual_evidence_sha256": transform["evidence_sha256"],
    }
    generated_dispatch = copy.deepcopy(dispatches[0])
    generated_dispatch.update(
        {
            "dispatch_index": 1,
            "sim_time_s": 2.0,
            "sim_step": 20,
            "macro_state": "S2_FR_TRAVERSE",
            "subphase": "FACE_CLEAR",
            "command_epoch": 2,
            "nominal_command_epoch": 2,
            "physical_command_epoch": 2,
            "command_provenance": generated_provenance,
            "source_action_consumption_index": None,
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
            "nominal_servo_targets_deg": servos,
            "nominal_wheel_targets_rad_s": wheels,
            "changed_servo_names": [SERVO_JOINT_NAMES[0]],
            "changed_wheel_names": [],
            "concurrent": False,
            "batch_id": generated_batch,
            "n_plus_one_verified": True,
            "n_plus_one_verified_sim_step": 21,
            "n_plus_one_readback_sha256": "e" * 64,
            "residual_transform": transform,
        }
    )
    generated_dispatch["ack"] = {
        "batch_id": generated_batch,
        "source": "fsm50_macro_controller",
        "applied_sim_step": 20,
        "first_physics_step": 21,
        "servo_targets_applied": servos,
        "wheel_targets_applied": wheels,
        "recording_metadata": metadata,
    }
    dispatches.append(generated_dispatch)

    retained_source = copy.deepcopy(sources[0])
    retained_provenance = copy.deepcopy(_provenance())
    retained_provenance.update(
        {
            "sequence_index": 1,
            "source_action_identity": "f" * 64,
            "source_segment_index": 1,
            "source_step_index": 2,
            "commands": ["retained source target"],
        }
    )
    retained_readback_sha256 = "a" * 64
    retained_source.update(
        {
            "source_action_index": 1,
            "expected_source_action_count": 2,
            "sim_time_s": 2.2,
            "sim_step": 22,
            "macro_state": "S2_FR_TRAVERSE",
            "subphase": "FACE_CLEAR",
            "command_provenance": retained_provenance,
            "target_changed": False,
            "dispatch_epoch": 2,
            "servo_targets_deg": servos,
            "wheel_targets_rad_s": wheels,
            "pre_action_verified_command_epoch": 2,
            "pre_action_verified_readback_sha256": retained_readback_sha256,
            "physical_dispatch_required": False,
            "physical_dispatch_applied": False,
            "physical_dispatch_index": None,
            "batch_id": "",
            "n_plus_one_verified": False,
            "n_plus_one_verified_sim_step": None,
            "n_plus_one_readback_sha256": "",
            "nominal_target_changed": False,
            "applied_target_changed": False,
            "physical_command_epoch": 2,
            "applied_servo_targets_deg": servos,
            "applied_wheel_targets_rad_s": wheels,
            "residual_transform_sha256": transform["core_transform_sha256"],
        }
    )
    sources.append(retained_source)

    retained_completion = copy.deepcopy(completions[0])
    retained_completion.update(
        {
            "segment_completion_index": 1,
            "source_segment_index": 1,
            "source_step_index": 2,
            "source_step_id": "step_002",
            "start_source_action_identity": "f" * 64,
            "source_action_consumption_index": 1,
            "start_command_epoch": 2,
            "start_sim_step": 22,
            "start_sim_time_s": 2.2,
            "start_physical_dispatch": False,
            "start_batch_id": "",
            "start_first_physics_step": None,
            "start_readback_verified": True,
            "start_readback_verified_sim_step": 22,
            "start_readback_sha256": retained_readback_sha256,
            "retained_epoch_same_target": True,
            "tracking_begin_sim_step": 22,
            "tracking_end_sim_step": 28,
            "tracking_end_sim_time_s": 2.8,
            "terminal_sim_step": 28,
            "terminal_sim_time_s": 2.8,
            "start_physical_command_epoch": 2,
            "nominal_target_changed": False,
            "applied_target_changed": False,
        }
    )
    completions.append(retained_completion)
    _write_jsonl(run / "source.jsonl", sources)
    _write_jsonl(run / "macro_dispatch_ledger.jsonl", dispatches)
    _write_jsonl(run / "completion.jsonl", completions)

    base_result = runner._strict_json_load(
        run / "worker_macro_fsm_result.json", label="result fixture"
    )
    base_result.update(
        {
            "expected_source_action_count": 2,
            "source_action_consumption_count": 2,
            "command_dispatch_count": 2,
            "segment_completion_count": 2,
            "expected_segment_completion_count": 2,
        }
    )
    base_result["direct_command_residual"].update(
        {
            "transform_count": 3,
            "physical_command_epoch": 3,
            "last_verified_physical_command_epoch": 3,
        }
    )
    base_terminal = outer_terminal["base_macro_fsm_terminal"]
    base_terminal["safe_stop_ack"].update(
        {"applied_sim_step": 30, "first_physics_step": 31}
    )
    base_terminal["safe_stop_readback"].update(
        {
            "batch_id": base_terminal["safe_stop_ack"]["batch_id"],
            "sim_step": 31,
        }
    )
    base_terminal["safe_stop_readback_sha256"] = (
        runner._canonical_json_sha256(base_terminal["safe_stop_readback"])
    )
    base_terminal["last_target_readback"].update(
        {
            "batch_id": "",
            "sim_step": 32,
        }
    )
    base_result["safe_stop_ack"] = copy.deepcopy(base_terminal["safe_stop_ack"])
    base_result["safe_stop_readback"] = copy.deepcopy(
        base_terminal["safe_stop_readback"]
    )
    base_result["safe_stop_readback_sha256"] = base_terminal[
        "safe_stop_readback_sha256"
    ]
    base_result["last_target_readback"] = copy.deepcopy(
        base_terminal["last_target_readback"]
    )
    return base_result, base_terminal


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


class _FakeLock:
    def __init__(self):
        self.acquired = False

    def acquire(self):
        self.acquired = True

    def release(self):
        self.acquired = False


class _FakeClient:
    def __init__(self, request: Any, reference_projection: Mapping[str, Any]):
        self.request = request
        self.reference_projection = reference_projection
        self.pid = 900001
        self.process = _FakeProcess(self.pid)
        self.latest_macro_fsm_terminal = {}
        self._history: list[dict[str, Any]] = []
        self.closed = False
        self._worker = {
            "worker_pid": self.pid,
            "worker_session_id": "fresh-worker-session",
            "adapter_runtime_instance_id": "fresh-adapter-runtime",
        }

    def start(self):
        return None

    def poll(self):
        return None

    def status(self):
        session = {
            "execution_mode": runner.EXECUTION_MODE,
            "operation": runner.OPERATION,
            "residual_macro_fsm_request_id": self.request.request_id,
            "base_macro_fsm_request_id": self.request.base_request.request_id,
            "state": "ready_for_start",
            "gate_e_zero_residual": self.request.gate_e_identity(
                payload_role="status"
            ),
            "direct_command_residual": {
                "enabled": True,
                "policy_id": self.request.policy_id,
                "policy_sha256": self.request.policy_sha256,
                "transform_count": 0,
                "physical_command_epoch": 0,
            },
        }
        return {
            "ready": True,
            "residual_macro_fsm_preflight_ready": True,
            "worker_residual_macro_fsm_preflight": self.request.preflight_payload(),
            "worker_residual_macro_fsm_session": session,
            **self._worker,
            "runtime_version": "fake-isaac-runtime-1",
            "root_state_write_count": 0,
            "worker_artifact_session": {"enabled": False},
            "worker_artifact_preflight": {"enabled": False},
            "operation_ack_history": list(self._history),
            "last_macro_fsm_terminal": self.latest_macro_fsm_terminal,
        }

    def start_residual_macro_fsm(self, **kwargs):
        self.start_kwargs = dict(kwargs)
        boundary_servos, boundary_wheels = _targets(0.0)
        start_ack = {
            "type": "operation_ack",
            "operation": "start_macro_fsm",
            "request_id": self.request.request_id,
            "accepted": True,
            "rejection_reason": "",
            "error": "",
            **self._worker,
            "artifact_request_id": "",
            "root_state_write_count": 0,
            "physics_dt_s": 1.0 / 120.0,
            "execution_mode": runner.EXECUTION_MODE,
            "request_identity_sha256": self.request.request_identity_sha256,
            "residual_macro_fsm_request_id": self.request.request_id,
            "base_macro_fsm_request_id": self.request.base_request.request_id,
            "base_macro_fsm_request_sha256": self.request.base_request_sha256,
            "base_start_payload_sha256": "c" * 64,
            "graph_sha256": self.request.graph_sha256,
            "profile_library_sha256": self.request.profile_library_sha256,
            "bundle_sha256": self.request.bundle_sha256,
            "gate_e_zero_residual": self.request.gate_e_identity(
                payload_role="start_ack"
            ),
            "start_boundary_ack": {
                "applied_sim_step": 1,
                "first_physics_step": 2,
                "motion_start_skew_s": 0.0,
                "physics_dt_s": 1.0 / 120.0,
                "servo_targets_applied": boundary_servos,
                "wheel_targets_applied": boundary_wheels,
            },
            "first_controller_tick_physics_step": 2,
            "earliest_profile_dispatch_physics_step": 9,
            "earliest_profile_actuation_physics_step": 10,
        }
        terminal = _write_live_artifacts(self.request, self._worker)
        terminal_ack = copy.deepcopy(terminal)
        terminal_ack["type"] = "operation_ack"
        terminal_ack["operation"] = "macro_fsm"
        self._history = [start_ack, terminal_ack]
        self.latest_macro_fsm_terminal = terminal

    def shutdown(self, *, mode, timeout_s, force_on_timeout, request_id):
        self.process.returncode = 0
        close_kwargs = {"wait_for_replicator": False, "skip_cleanup": True}
        common = {
            **self._worker,
            "artifact_request_id": "",
            "macro_fsm_request_id": self.request.request_id,
            "residual_macro_fsm_request_id": self.request.request_id,
            "base_macro_fsm_request_id": self.request.base_request.request_id,
            "root_state_write_count": 0,
            "runtime_version": "fake-isaac-runtime-1",
            "schema_version": runner.macro_runner.MACRO_FAST_CLOSE_SCHEMA,
        }

        def row(kind: str, role: str, **extra):
            return {
                "type": kind,
                "request_id": request_id,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": close_kwargs,
                **common,
                "gate_e_zero_residual": self.request.gate_e_identity(
                    payload_role=role
                ),
                **extra,
            }

        return {
            "pid": self.pid,
            "returncode": 0,
            "forced_termination": False,
            "normal_exit": True,
            "timed_out": False,
            "requested_mode": "fast",
            "request_id": request_id,
            "shutdown_ack": row(
                "operation_ack", "shutdown_ack", operation="shutdown"
            ),
            "close_requested_ack": row("close_requested", "close_requested"),
            "close_requested_receipt": row(
                "close_receipt",
                "close_requested",
                close_event_type="close_requested",
                received=True,
            ),
            "close_returned_ack": row("close_returned", "close_returned"),
            "close_returned_receipt": row(
                "close_receipt",
                "close_returned",
                close_event_type="close_returned",
                received=True,
            ),
        }

    def close(self):
        self.closed = True


def _args(output_root: Path, reference: Path) -> argparse.Namespace:
    return argparse.Namespace(
        command="run",
        source_version=SOURCE,
        reviewed_reference_run_dir=reference,
        gate_c_baseline_run_dir=reference,
        output_root=output_root,
        alignment_path=runner.macro_runner.DEFAULT_ALIGNMENT_PATH,
        task_success_table_path=runner.macro_runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
        telemetry_hz=runner.DEFAULT_TELEMETRY_HZ,
        post_run_settle_s=runner.DEFAULT_POST_SETTLE_S,
        task_timeout_s=240.0,
        terminal_timeout_s=2.0,
        operation_timeout_s=2.0,
        sim_startup_timeout_s=2.0,
        sim_worker_status_timeout_s=1.0,
        sim_worker_log_lines=20,
        worker_launch_mode="current-python",
        worker_python_exe="",
        isaaclab_bat="",
        device="cpu",
        livestream=0,
        experience="",
        accept_isaac_eula=False,
    )


def _simple_request(root: Path) -> Any:
    return SimpleNamespace(
        run_dir=root,
        request_id="outer-request",
        request_identity_sha256="c" * 64,
        source_version=SOURCE,
        profile_id="profile",
        graph_id="graph",
        graph_sha256="a" * 64,
        profile_library_sha256="b" * 64,
        bundle_sha256="c" * 64,
        policy_kind="ZERO",
        policy_id="zero",
        policy_sha256="d" * 64,
        residual_core_sha256="e" * 64,
        envelope_canonical_sha256="f" * 64,
        base_request_path=root / "base-request.json",
        base_request_sha256="1" * 64,
        base_request=SimpleNamespace(request_id="base-request"),
        gate_e_identity=lambda *, payload_role: {
            "role": payload_role,
            "request_id": "outer-request",
        },
    )


def _run_fake_pending(root: Path) -> dict[str, Any]:
    reference = root / "reference"
    projection = _write_reference_projection_fixture(reference)
    output = root / "fresh-output"
    args = _args(output, reference)
    environment = {
        "environment_lock_path": str(
            runner.macro_runner.DEFAULT_ENVIRONMENT_LOCK_PATH
        ),
        "environment_lock_sha256": "e" * 64,
        "locked_source_file_count": 1,
        "required_source_file_count": 1,
        "source_closure_complete": True,
        "source_verification_sha256": "f" * 64,
    }
    closure = {
        "schema_version": runner.macro_runner.GATE_C_CLOSURE_EVIDENCE_SCHEMA,
        "baseline": {
            "request_id": "old-base",
            "worker_pid": 10,
            "worker_session_id": "old-worker",
            "adapter_runtime_instance_id": "old-adapter",
        },
        "repeats": [
            {
                "request_id": f"old-repeat-{index}",
                "worker_pid": 20 + index,
                "worker_session_id": f"old-worker-{index}",
                "adapter_runtime_instance_id": f"old-adapter-{index}",
            }
            for index in range(3)
        ],
        "closure_sha256": "a" * 64,
    }
    reference_evidence = {
        "identity": {
            "run_dir": str(reference),
            "source_version": SOURCE,
            "trial_kind": "baseline",
            "trial_index": 0,
            "manual_review_status": (
                "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
            ),
            "manifest_path": str(reference / runner.REFERENCE_MANIFEST_NAME),
            "manifest_sha256": "1" * 64,
            "request_id": "old-reference-request",
            "worker_pid": 99,
            "worker_session_id": "old-reference-worker",
            "adapter_runtime_instance_id": "old-reference-adapter",
            "semantic_projection_sha256": runner._canonical_json_sha256(
                projection
            ),
        },
        "projection": projection,
    }
    holder: dict[str, Any] = {}

    def environment_validator():
        return copy.deepcopy(environment)

    def closure_validator(*_args, **_kwargs):
        return copy.deepcopy(closure)

    def reference_validator(**_kwargs):
        return copy.deepcopy(reference_evidence)

    def load_request(path, *, environment_validator=None):
        value = load_worker_residual_macro_fsm_request(
            path, environment_validator=environment_validator
        )
        holder["request"] = value
        return value

    def client_factory(_worker_args):
        return _FakeClient(holder["request"], projection)

    attempt = runner.run_one_zero_r0(
        args,
        client_factory=client_factory,
        environment_lock_validator=environment_validator,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        residual_request_loader=load_request,
        lock_factory=_FakeLock,
        process_snapshot_fn=lambda: [],
        conflict_detector=lambda rows: [],
    )
    return {
        "attempt": attempt,
        "output": output,
        "projection": projection,
        "environment_validator": environment_validator,
        "closure_validator": closure_validator,
        "reference_validator": reference_validator,
        "residual_request_loader": load_request,
    }


def _review_dependencies(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "environment_lock_validator": fixture["environment_validator"],
        "closure_validator": fixture["closure_validator"],
        "reference_validator": fixture["reference_validator"],
        "residual_request_loader": fixture["residual_request_loader"],
    }


def _completed_review_document(context: Mapping[str, Any]) -> dict[str, Any]:
    document = runner._manual_verdict_template(context)
    document["review_complete"] = True
    document["reviewed_utc"] = "2026-08-15T12:34:56+00:00"
    document["verdict"] = {
        "task_completed": True,
        "body_crossed_front_face": True,
        "required_leg_lift_completed": True,
        "final_recoverable": True,
        "posture_incomplete": True,
        "robot_fell": False,
        "body_stuck": False,
        "wheel_drive_up_without_required_lift": False,
        "dangerous_body_collision": False,
        "joint_limit_violation": False,
        "severe_penetration": False,
        "notes": "manual ZERO-R0 video review complete",
    }
    return document


class ResidualRunnerTests(unittest.TestCase):
    def test_parser_has_no_checkpoint_action_policy_or_count_surface(self):
        parser = runner.build_parser()
        run_parser = next(
            action.choices["run"]
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        destinations = {action.dest for action in run_parser._actions}
        for forbidden in (
            "checkpoint",
            "checkpoint_path",
            "actions",
            "action",
            "count",
            "policy",
            "policy_kind",
        ):
            self.assertNotIn(forbidden, destinations)

    def test_public_start_requires_exact_residual_client_seam(self):
        class OldClient:
            def start_macro_fsm(self, **kwargs):
                pass

        with self.assertRaisesRegex(
            runner.ResidualRunnerContractError, "public Gate-E residual start API"
        ):
            runner._send_residual_start(
                OldClient(),
                request=SimpleNamespace(),
                worker_session_id="worker",
            )

    def test_zero_transform_rejects_nonzero_action_even_if_resealed(self):
        request = SimpleNamespace(
            source_version=SOURCE,
            policy_id="zero",
            policy_sha256="d" * 64,
        )
        servos, wheels = _targets(1.0)
        transform = _zero_transform(request, servos, wheels)
        transform["raw_normalized_action"][0] = 0.001
        unhashed = dict(transform)
        unhashed.pop("evidence_sha256")
        transform["evidence_sha256"] = runner._canonical_json_sha256(unhashed)
        with self.assertRaisesRegex(
            runner.ResidualRunnerContractError, "raw_normalized_action"
        ):
            runner._validate_zero_transform(
                transform, request=request, label="tampered"
            )

    def test_exact_reviewed_reference_constants_are_source_closed(self):
        self.assertEqual(set(runner.EXACT_REVIEWED_REFERENCE_RUNS), set(runner.ALLOWED_SOURCE_VERSIONS))
        for source, run in runner.EXACT_REVIEWED_REFERENCE_RUNS.items():
            self.assertTrue(run.is_dir(), source)
            self.assertEqual(
                runner.sha256_file(run / runner.REFERENCE_MANIFEST_NAME),
                runner.EXACT_REVIEWED_REFERENCE_MANIFEST_SHA256[source],
            )

    def test_valid_fake_live_end_to_end_writes_one_r0_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            projection = _write_reference_projection_fixture(reference)
            output = root / "fresh-output"
            args = _args(output, reference)
            environment = {
                "environment_lock_path": str(runner.macro_runner.DEFAULT_ENVIRONMENT_LOCK_PATH),
                "environment_lock_sha256": "e" * 64,
                "locked_source_file_count": 1,
                "required_source_file_count": 1,
                "source_closure_complete": True,
                "source_verification_sha256": "f" * 64,
            }
            closure = {
                "schema_version": runner.macro_runner.GATE_C_CLOSURE_EVIDENCE_SCHEMA,
                "baseline": {
                    "request_id": "old-base",
                    "worker_pid": 10,
                    "worker_session_id": "old-worker",
                    "adapter_runtime_instance_id": "old-adapter",
                },
                "repeats": [
                    {
                        "request_id": f"old-repeat-{index}",
                        "worker_pid": 20 + index,
                        "worker_session_id": f"old-worker-{index}",
                        "adapter_runtime_instance_id": f"old-adapter-{index}",
                    }
                    for index in range(3)
                ],
                "closure_sha256": "a" * 64,
            }
            reference_evidence = {
                "identity": {
                    "run_dir": str(reference),
                    "source_version": SOURCE,
                    "trial_kind": "baseline",
                    "trial_index": 0,
                    "request_id": "old-reference-request",
                    "worker_pid": 99,
                    "worker_session_id": "old-reference-worker",
                    "adapter_runtime_instance_id": "old-reference-adapter",
                    "semantic_projection_sha256": runner._canonical_json_sha256(
                        projection
                    ),
                },
                "projection": projection,
            }
            holder: dict[str, Any] = {}

            def load_request(path, *, environment_validator=None):
                value = load_worker_residual_macro_fsm_request(
                    path, environment_validator=environment_validator
                )
                holder["request"] = value
                return value

            def client_factory(_worker_args):
                return _FakeClient(holder["request"], projection)

            attempt = runner.run_one_zero_r0(
                args,
                client_factory=client_factory,
                environment_lock_validator=lambda: copy.deepcopy(environment),
                closure_validator=lambda *_args, **_kwargs: copy.deepcopy(closure),
                reference_validator=lambda **_kwargs: copy.deepcopy(
                    reference_evidence
                ),
                residual_request_loader=load_request,
                lock_factory=_FakeLock,
                process_snapshot_fn=lambda: [],
                conflict_detector=lambda rows: [],
            )
            self.assertTrue(attempt.macro_fsm_complete)
            manifest = runner._strict_json_load(
                attempt.manifest_path, label="test R0 manifest"
            )
            self.assertTrue(manifest["live_isaac_execution"])
            self.assertEqual(manifest["policy_kind"], "ZERO")
            self.assertFalse(manifest["checkpoint_loaded"])
            self.assertFalse(manifest["ppo_training_performed"])
            self.assertTrue(manifest["semantic_projection_equal"])
            self.assertTrue(manifest["shutdown_verified"])
            self.assertEqual(
                manifest["video_sha256"],
                runner.sha256_file(Path(manifest["video_path"])),
            )
            self.assertEqual(
                manifest["manual_video_status"],
                runner.MANUAL_VIDEO_STATUS_PENDING,
            )
            self.assertEqual(
                [path.name for path in output.glob("*manifest*.json")],
                [runner.RUNNER_MANIFEST_NAME],
            )

    def test_valid_review_is_first_write_only_and_returns_r1_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _run_fake_pending(Path(tmp))
            manifest_path = fixture["attempt"].manifest_path
            dependencies = _review_dependencies(fixture)
            context = runner._machine_review_context(
                manifest_path,
                expect_reviewed=False,
                **dependencies,
            )
            document = _completed_review_document(context)
            operator_verdict = Path(tmp) / "operator-verdict.json"
            _write_json(operator_verdict, document)
            result = runner.review_r0_run(
                run_dir=fixture["output"],
                verdict_path=operator_verdict,
                **dependencies,
            )
            self.assertEqual(
                result["schema_version"], runner.REVIEW_VALIDATION_SCHEMA
            )
            self.assertEqual(set(result), runner.REVIEW_VALIDATION_KEYS)
            self.assertEqual(
                result["manual_review_status"],
                runner.MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS_POSTURE_INCOMPLETE,
            )
            self.assertEqual(
                result["manual_review_classification"],
                "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
            )
            self.assertTrue(result["reviewed_success"])
            self.assertTrue(result["shutdown_verified"])
            self.assertTrue(result["semantic_projection_equal"])
            self.assertTrue(result["macro_fsm_complete"])
            self.assertEqual(
                result,
                runner._validate_reviewed_r0_manifest_impl(
                    manifest_path,
                    **dependencies,
                ),
            )
            reviewed = runner._strict_json_load(
                manifest_path, label="reviewed test manifest"
            )
            pending = {
                key: copy.deepcopy(reviewed[key])
                for key in runner._PENDING_MANIFEST_KEYS
            }
            pending["manual_video_status"] = runner.MANUAL_VIDEO_STATUS_PENDING
            self.assertEqual(
                reviewed["pre_review_manifest_sha256"],
                runner._serialized_json_sha256(pending),
            )
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError,
                "keys are not exact|first-write only",
            ):
                runner.review_r0_run(
                    run_dir=fixture["output"],
                    verdict_path=operator_verdict,
                    **dependencies,
                )

    def test_review_rejects_machine_and_verdict_tamper_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _run_fake_pending(root)
            manifest_path = fixture["attempt"].manifest_path
            dependencies = _review_dependencies(fixture)
            original_manifest = runner._strict_json_load(
                manifest_path, label="original pending manifest"
            )
            context = runner._machine_review_context(
                manifest_path,
                expect_reviewed=False,
                **dependencies,
            )
            document = _completed_review_document(context)
            operator_verdict = root / "operator-verdict.json"
            _write_json(operator_verdict, document)

            def reject_manifest(mutator, pattern):
                tampered = copy.deepcopy(original_manifest)
                mutator(tampered)
                _write_json(manifest_path, tampered)
                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError, pattern
                ):
                    runner.review_r0_run(
                        run_dir=fixture["output"],
                        verdict_path=operator_verdict,
                        **dependencies,
                    )
                self.assertFalse(
                    (fixture["output"] / runner.MANUAL_VERDICT_NAME).exists()
                )
                _write_json(manifest_path, original_manifest)

            cases = (
                (
                    "path",
                    lambda value: value.__setitem__(
                        "residual_request_path", str(root / "escape.json")
                    ),
                    "escapes|missing",
                ),
                (
                    "hash",
                    lambda value: value.__setitem__(
                        "residual_request_sha256", "0" * 64
                    ),
                    "path/SHA binding",
                ),
                (
                    "identity",
                    lambda value: value.__setitem__(
                        "request_identity_sha256", "0" * 64
                    ),
                    "request_identity_sha256 mismatch",
                ),
                (
                    "environment",
                    lambda value: value["canonical_environment_lock"].__setitem__(
                        "environment_lock_sha256", "0" * 64
                    ),
                    "environment lock is stale",
                ),
                (
                    "code",
                    lambda value: value["gate_e_code_identity"].__setitem__(
                        "identity_sha256", "0" * 64
                    ),
                    "code identity is stale",
                ),
                (
                    "reference",
                    lambda value: value["reviewed_reference"].__setitem__(
                        "manifest_sha256", "0" * 64
                    ),
                    "reviewed reference is stale",
                ),
                (
                    "closure",
                    lambda value: value.__setitem__(
                        "gate_c_closure_sha256", "0" * 64
                    ),
                    "gate_c_closure_sha256 mismatch",
                ),
                (
                    "fast-close",
                    lambda value: value["shutdown_outcome"][
                        "close_requested_receipt"
                    ].pop("gate_e_zero_residual"),
                    "fast-close identity mismatch",
                ),
            )
            for name, mutate, pattern in cases:
                with self.subTest(kind=name):
                    reject_manifest(mutate, pattern)

            video_path = Path(original_manifest["video_path"])
            original_video = video_path.read_bytes()
            video_path.write_bytes(original_video + b"tamper")
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError, "video validation is stale"
            ):
                runner.review_r0_run(
                    run_dir=fixture["output"],
                    verdict_path=operator_verdict,
                    **dependencies,
                )
            video_path.write_bytes(original_video)

            dispatch_path = fixture["output"] / "macro_dispatch_ledger.jsonl"
            original_dispatch = dispatch_path.read_bytes()
            row = runner._strict_jsonl(dispatch_path, label="dispatch tamper")[0]
            row["residual_transform"]["raw_normalized_action"][0] = 0.25
            _write_jsonl(dispatch_path, [row])
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError,
                "command evidence differs|SHA mismatch|raw_normalized_action",
            ):
                runner.review_r0_run(
                    run_dir=fixture["output"],
                    verdict_path=operator_verdict,
                    **dependencies,
                )
            dispatch_path.write_bytes(original_dispatch)

            failed_verdict = copy.deepcopy(document)
            failed_verdict["verdict"]["robot_fell"] = True
            _write_json(operator_verdict, failed_verdict)
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError,
                "classification differs",
            ):
                runner.review_r0_run(
                    run_dir=fixture["output"],
                    verdict_path=operator_verdict,
                    **dependencies,
                )
            self.assertFalse(
                (fixture["output"] / runner.MANUAL_VERDICT_NAME).exists()
            )

    def test_review_callable_signature_and_cli_surface_are_exact(self):
        signature = inspect.signature(runner.validate_reviewed_r0_manifest)
        self.assertEqual(
            list(signature.parameters),
            ["manifest_path", "environment_lock_validator"],
        )
        self.assertEqual(
            signature.parameters["environment_lock_validator"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        parser = runner.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(subparsers.choices), {"run", "review", "emit-review-template"}
        )
        self.assertEqual(
            {
                action.dest
                for action in subparsers.choices["review"]._actions
                if action.dest != "help"
            },
            {"run_dir", "verdict"},
        )

    def test_dispatch_tamper_rejects_applied_map_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _simple_request(root)
            worker = {
                "worker_pid": 1,
                "worker_session_id": "worker",
                "adapter_runtime_instance_id": "adapter",
            }
            terminal = _write_live_artifacts(request, worker)
            path = root / "macro_dispatch_ledger.jsonl"
            row = runner._strict_jsonl(path, label="dispatch")[0]
            row["servo_targets_deg"][SERVO_JOINT_NAMES[0]] = 2.0
            _write_jsonl(path, [row])
            result = runner._strict_json_load(
                root / "worker_macro_fsm_result.json", label="result"
            )
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError, "nominal ZERO identity"
            ):
                runner._validate_zero_ledgers(
                    request=request,
                    base_result=result,
                    base_terminal=terminal["base_macro_fsm_terminal"],
                )

    def test_dispatch_tamper_rejects_wrong_physical_epoch_and_n_plus_one(self):
        for field, value, pattern in (
            ("physical_command_epoch", 2, "physical epoch"),
            ("n_plus_one_verified", False, "ACK/N\\+1"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                path = root / "macro_dispatch_ledger.jsonl"
                row = runner._strict_jsonl(path, label="dispatch")[0]
                row[field] = value
                _write_jsonl(path, [row])
                result = runner._strict_json_load(
                    root / "worker_macro_fsm_result.json", label="result"
                )
                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError, pattern
                ):
                    runner._validate_zero_ledgers(
                        request=request,
                        base_result=result,
                        base_terminal=terminal["base_macro_fsm_terminal"],
                    )

    def test_generated_dispatch_then_retained_source_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _simple_request(root)
            worker = {
                "worker_pid": 1,
                "worker_session_id": "worker",
                "adapter_runtime_instance_id": "adapter",
            }
            result, base_terminal = (
                _write_generated_dispatch_then_retained_source_artifacts(
                    request, worker
                )
            )

            ledger = runner._validate_zero_ledgers(
                request=request,
                base_result=result,
                base_terminal=base_terminal,
            )

            self.assertEqual(ledger["source_action_count"], 2)
            self.assertEqual(ledger["nominal_dispatch_count"], 2)
            self.assertEqual(ledger["controller_physical_command_epoch"], 2)

    def test_retained_source_and_completion_tamper_fail_closed(self):
        cases = (
            (
                "physical-epoch",
                "source.jsonl",
                lambda rows: rows[1].__setitem__("physical_command_epoch", 1),
                "retained-target evidence",
            ),
            (
                "pre-action-epoch",
                "source.jsonl",
                lambda rows: rows[1].__setitem__(
                    "pre_action_verified_command_epoch", 1
                ),
                "retained-target evidence",
            ),
            (
                "completion-readback",
                "completion.jsonl",
                lambda rows: rows[1].__setitem__(
                    "start_readback_sha256", "b" * 64
                ),
                "retained readback binding",
            ),
        )
        for name, relative_path, mutate, pattern in cases:
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                result, base_terminal = (
                    _write_generated_dispatch_then_retained_source_artifacts(
                        request, worker
                    )
                )
                path = root / relative_path
                rows = runner._strict_jsonl(path, label=f"{name} fixture")
                mutate(rows)
                _write_jsonl(path, rows)

                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError, pattern
                ):
                    runner._validate_zero_ledgers(
                        request=request,
                        base_result=result,
                        base_terminal=base_terminal,
                    )

    def test_safe_stop_readback_and_final_snapshot_tamper_fail_closed(self):
        def self_consistent_result_ack_drift(result, terminal):
            result["safe_stop_ack"]["applied_sim_step"] += 8
            result["safe_stop_ack"]["first_physics_step"] += 8

        def result_copy_drift(result, terminal):
            result["safe_stop_readback"]["sim_step"] += 1

        def digest_drift(result, terminal):
            result["safe_stop_readback_sha256"] = "0" * 64
            terminal["safe_stop_readback_sha256"] = "0" * 64

        def exact_n_plus_one_drift(result, terminal):
            terminal["safe_stop_readback"]["sim_step"] += 1
            digest = runner._canonical_json_sha256(
                terminal["safe_stop_readback"]
            )
            terminal["safe_stop_readback_sha256"] = digest
            result["safe_stop_readback"] = copy.deepcopy(
                terminal["safe_stop_readback"]
            )
            result["safe_stop_readback_sha256"] = digest

        def final_wheel_drift(result, terminal):
            name = WHEEL_JOINT_NAMES[0]
            terminal["last_target_readback"][
                "actual_wheel_drive_targets_rad_s"
            ][name] = 0.1
            result["last_target_readback"] = copy.deepcopy(
                terminal["last_target_readback"]
            )

        for name, mutate in (
            ("self-consistent-result-ack", self_consistent_result_ack_drift),
            ("result-copy", result_copy_drift),
            ("digest", digest_drift),
            ("exact-n-plus-one", exact_n_plus_one_drift),
            ("final-wheel", final_wheel_drift),
        ):
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                base_terminal = terminal["base_macro_fsm_terminal"]
                result = runner._strict_json_load(
                    root / "worker_macro_fsm_result.json", label="result"
                )
                mutate(result, base_terminal)

                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError,
                    "safe-stop one-batch/N\\+1 evidence",
                ):
                    runner._validate_zero_ledgers(
                        request=request,
                        base_result=result,
                        base_terminal=base_terminal,
                    )

    def test_safe_stop_and_final_readback_schema_tamper_fail_closed(self):
        def synchronize_safe(result, terminal):
            digest = runner._canonical_json_sha256(
                terminal["safe_stop_readback"]
            )
            terminal["safe_stop_readback_sha256"] = digest
            result["safe_stop_readback"] = copy.deepcopy(
                terminal["safe_stop_readback"]
            )
            result["safe_stop_readback_sha256"] = digest

        def safe_extra(result, terminal):
            terminal["safe_stop_readback"]["unexpected"] = True
            synchronize_safe(result, terminal)

        def safe_missing(result, terminal):
            terminal["safe_stop_readback"].pop("root_state_write_count")
            synchronize_safe(result, terminal)

        def final_extra(result, terminal):
            terminal["last_target_readback"]["unexpected"] = True
            result["last_target_readback"] = copy.deepcopy(
                terminal["last_target_readback"]
            )

        def final_missing(result, terminal):
            terminal["last_target_readback"].pop("command_epoch")
            result["last_target_readback"] = copy.deepcopy(
                terminal["last_target_readback"]
            )

        for name, mutate in (
            ("safe-extra", safe_extra),
            ("safe-missing", safe_missing),
            ("final-extra", final_extra),
            ("final-missing", final_missing),
        ):
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                base_terminal = terminal["base_macro_fsm_terminal"]
                result = runner._strict_json_load(
                    root / "worker_macro_fsm_result.json", label="result"
                )
                mutate(result, base_terminal)

                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError,
                    "readback schema keys are not exact",
                ):
                    runner._validate_zero_ledgers(
                        request=request,
                        base_result=result,
                        base_terminal=base_terminal,
                    )

    def test_empty_sparse_completion_and_deferred_sparse_end_are_valid(self):
        for kind in ("empty", "sparse-ended-false"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                path = root / "completion.jsonl"
                row = runner._strict_jsonl(path, label="completion")[0]
                if kind == "empty":
                    row = _empty_sparse_completion_row(row)
                else:
                    row["tracking_end_evidence"]["ended"] = False
                    row["tracking_end_evidence"][
                        "tracking_completion_deferred"
                    ] = True
                _write_jsonl(path, [row])
                result = runner._strict_json_load(
                    root / "worker_macro_fsm_result.json", label="result"
                )

                ledger = runner._validate_zero_ledgers(
                    request=request,
                    base_result=result,
                    base_terminal=terminal["base_macro_fsm_terminal"],
                )
                self.assertEqual(ledger["segment_completion_count"], 1)

    def test_completion_tracking_contract_tamper_fails_closed(self):
        def count_drift(row):
            row["tracking_begin_count"] = 1

        def latched_key_drift(row):
            row["latched_servo_residual_deg"] = {}

        def latched_value_drift(row):
            row["latched_servo_residual_deg"][SERVO_JOINT_NAMES[0]] = 0.1

        def begin_called_drift(row):
            row["tracking_begin_evidence"]["called"] = False

        def begin_names_drift(row):
            row["tracking_begin_evidence"]["sparse_joint_names"] = []

        def sparse_ended_nonbool(row):
            row["tracking_end_evidence"]["ended"] = 1

        def sparse_deferred_drift(row):
            row["tracking_end_evidence"]["ended"] = False
            row["tracking_end_evidence"][
                "tracking_completion_deferred"
            ] = False

        def empty_required_drift(row):
            row["tracking_end_evidence"]["required"] = True

        def empty_ended_drift(row):
            row["tracking_end_evidence"]["ended"] = False

        def empty_reason_drift(row):
            row["tracking_end_evidence"]["reason"] = "tampered"

        def begin_step_drift(row):
            row["tracking_begin_sim_step"] += 1

        def end_step_drift(row):
            row["tracking_end_sim_step"] += 1

        def lifecycle_reason_drift(row):
            row["tracking_end_reason"] = "tampered"

        for name, empty, mutate in (
            ("empty-count", True, count_drift),
            ("latched-key", False, latched_key_drift),
            ("latched-value", False, latched_value_drift),
            ("begin-called", False, begin_called_drift),
            ("begin-names", False, begin_names_drift),
            ("sparse-ended-nonbool", False, sparse_ended_nonbool),
            ("sparse-deferred", False, sparse_deferred_drift),
            ("empty-required", True, empty_required_drift),
            ("empty-ended", True, empty_ended_drift),
            ("empty-reason", True, empty_reason_drift),
            ("begin-step", False, begin_step_drift),
            ("end-step", False, end_step_drift),
            ("lifecycle-reason", False, lifecycle_reason_drift),
        ):
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = _simple_request(root)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                path = root / "completion.jsonl"
                row = runner._strict_jsonl(path, label="completion")[0]
                if empty:
                    row = _empty_sparse_completion_row(row)
                mutate(row)
                _write_jsonl(path, [row])
                result = runner._strict_json_load(
                    root / "worker_macro_fsm_result.json", label="result"
                )

                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError,
                    "effective target/lifecycle mismatch",
                ):
                    runner._validate_zero_ledgers(
                        request=request,
                        base_result=result,
                        base_terminal=terminal["base_macro_fsm_terminal"],
                    )

    def test_completion_tamper_rejects_effective_target_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _simple_request(root)
            worker = {
                "worker_pid": 1,
                "worker_session_id": "worker",
                "adapter_runtime_instance_id": "adapter",
            }
            terminal = _write_live_artifacts(request, worker)
            path = root / "completion.jsonl"
            row = runner._strict_jsonl(path, label="completion")[0]
            row["effective_completion_servo_targets_deg"][
                SERVO_JOINT_NAMES[0]
            ] = 1.5
            row["effective_completion_targets_sha256"] = (
                runner._canonical_json_sha256(
                    row["effective_completion_servo_targets_deg"]
                )
            )
            _write_jsonl(path, [row])
            result = runner._strict_json_load(
                root / "worker_macro_fsm_result.json", label="result"
            )
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError,
                "effective target/lifecycle",
            ):
                runner._validate_zero_ledgers(
                    request=request,
                    base_result=result,
                    base_terminal=terminal["base_macro_fsm_terminal"],
                )

    def test_outer_terminal_rejects_safe_stop_copy_drift(self):
        def ack_drift(terminal):
            terminal["safe_stop_ack"]["applied_sim_step"] += 8
            terminal["safe_stop_ack"]["first_physics_step"] += 8

        def readback_drift(terminal):
            terminal["safe_stop_readback"]["sim_step"] += 8
            terminal["safe_stop_readback_sha256"] = (
                runner._canonical_json_sha256(
                    terminal["safe_stop_readback"]
                )
            )

        def digest_drift(terminal):
            terminal["safe_stop_readback_sha256"] = "0" * 64

        def final_drift(terminal):
            terminal["last_target_readback"]["sim_step"] += 8

        for name, mutate in (
            ("ack", ack_drift),
            ("readback", readback_drift),
            ("digest", digest_drift),
            ("final", final_drift),
        ):
            with self.subTest(kind=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = root / "reference"
                projection = _write_reference_projection_fixture(reference)
                run = root / "run"
                run.mkdir()
                request = _simple_request(run)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                mutate(terminal)

                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError,
                    "outer safe-stop evidence differs",
                ):
                    runner._validate_outer_terminal(
                        terminal,
                        request=request,
                        worker_binding=worker,
                        reference_projection=projection,
                    )

    def test_outer_terminal_rejects_hash_and_semantic_projection_tamper(self):
        for kind in ("hash", "semantic"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                reference = root / "reference"
                projection = _write_reference_projection_fixture(reference)
                run = root / "run"
                run.mkdir()
                request = _simple_request(run)
                worker = {
                    "worker_pid": 1,
                    "worker_session_id": "worker",
                    "adapter_runtime_instance_id": "adapter",
                }
                terminal = _write_live_artifacts(request, worker)
                if kind == "hash":
                    terminal["base_worker_result_sha256"] = "0" * 64
                    pattern = "SHA mismatch"
                else:
                    projection = copy.deepcopy(projection)
                    projection["transitions"][0]["to_state"] = "TAMPERED"
                    pattern = "semantic projection"
                with self.assertRaisesRegex(
                    runner.ResidualRunnerContractError, pattern
                ):
                    runner._validate_outer_terminal(
                        terminal,
                        request=request,
                        worker_binding=worker,
                        reference_projection=projection,
                    )

    def test_real_shape_failed_terminal_preserves_outer_and_base_failure(self):
        outer_request_id = "7c230bed36ed4af3b7d75deb93e4e5eb"
        base_request_id = "4cee99dc23924d1482339c19698ae9a5"
        failure = "ResidualContractError: profile_strategy must be non-empty text"
        terminal = {
            "type": "macro_fsm_failed",
            "operation": runner.OPERATION,
            "request_id": outer_request_id,
            "residual_macro_fsm_request_id": outer_request_id,
            "base_macro_fsm_request_id": base_request_id,
            "accepted": False,
            "macro_fsm_complete": False,
            "error": failure,
            "controller_terminal_outcome": "RUNTIME_FAILURE",
            "base_macro_fsm_terminal": {
                "type": "macro_fsm_failed",
                "operation": "macro_fsm",
                "request_id": base_request_id,
                "accepted": False,
                "macro_fsm_complete": False,
                "error": failure,
                "controller_terminal_outcome": "RUNTIME_FAILURE",
            },
        }
        request = _simple_request(Path(".").resolve())

        with self.assertRaises(runner.ResidualRunnerContractError) as caught:
            runner._validate_outer_terminal(
                terminal,
                request=request,
                worker_binding={},
                reference_projection={},
            )

        prefix = "Gate-E worker reported macro_fsm_failed: "
        message = str(caught.exception)
        self.assertTrue(message.startswith(prefix))
        detail = json.loads(message[len(prefix) :])
        self.assertEqual(
            detail["outer"],
            {
                "accepted": False,
                "base_macro_fsm_request_id": base_request_id,
                "controller_terminal_outcome": "RUNTIME_FAILURE",
                "controller_terminal_reason": None,
                "error": failure,
                "macro_fsm_complete": False,
                "operation": runner.OPERATION,
                "request_id": outer_request_id,
                "residual_macro_fsm_request_id": outer_request_id,
                "type": "macro_fsm_failed",
            },
        )
        self.assertEqual(
            detail["base"],
            {
                "accepted": False,
                "base_macro_fsm_request_id": None,
                "controller_terminal_outcome": "RUNTIME_FAILURE",
                "controller_terminal_reason": None,
                "error": failure,
                "macro_fsm_complete": False,
                "operation": "macro_fsm",
                "request_id": base_request_id,
                "residual_macro_fsm_request_id": None,
                "type": "macro_fsm_failed",
            },
        )
        self.assertNotIn("terminal type mismatch", message)

    def test_terminal_ack_requires_macro_transport_operation(self):
        request = _simple_request(Path("." ).resolve())
        terminal = {
            "type": "macro_fsm_complete",
            "operation": runner.OPERATION,
            "request_id": request.request_id,
            "gate_e_zero_residual": request.gate_e_identity(
                payload_role="terminal"
            ),
        }
        ack = copy.deepcopy(terminal)
        ack["type"] = "operation_ack"
        ack["operation"] = "macro_fsm"
        self.assertEqual(
            runner._validate_terminal_ack(
                ack, terminal=terminal, request=request
            )["operation"],
            "macro_fsm",
        )
        ack["operation"] = runner.OPERATION
        with self.assertRaisesRegex(
            runner.ResidualRunnerContractError, "identity mismatch"
        ):
            runner._validate_terminal_ack(
                ack, terminal=terminal, request=request
            )

    def test_fast_close_rejects_missing_gate_e_receipt_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = _simple_request(Path(tmp))
            client = _FakeClient(request, {})
            outcome = client.shutdown(
                mode="fast",
                timeout_s=1.0,
                force_on_timeout=True,
                request_id="shutdown-id",
            )
            del outcome["close_requested_receipt"]["gate_e_zero_residual"]
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError,
                "fast-close identity mismatch",
            ):
                runner._validate_residual_fast_shutdown(
                    outcome,
                    shutdown_request_id="shutdown-id",
                    request=request,
                    worker_binding={
                        **client._worker,
                        "runtime_version": "fake-isaac-runtime-1",
                    },
                )

    def test_ready_binding_rejects_reused_worker_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = _simple_request(Path(tmp))
            request.preflight_payload = lambda: {"schema_version": "preflight"}
            request.base_request = SimpleNamespace(request_id="base-request")
            client = _FakeClient(request, {})
            status = client.status()
            status["worker_residual_macro_fsm_preflight"] = {
                "schema_version": "preflight"
            }
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError, "not fresh"
            ):
                runner._validate_worker_binding(
                    status,
                    request=request,
                    stale_identities={"worker_pid": {client.pid}},
                )

    def test_ledger_tamper_rejects_residual_only_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            projection = _write_reference_projection_fixture(reference)
            request = _simple_request(root)
            worker = {
                "worker_pid": 1,
                "worker_session_id": "worker",
                "adapter_runtime_instance_id": "adapter",
            }
            terminal = _write_live_artifacts(request, worker)
            dispatch_path = root / "macro_dispatch_ledger.jsonl"
            row = runner._strict_jsonl(dispatch_path, label="dispatch")[0]
            row["dispatch_cause"] = "RESIDUAL_ONLY"
            _write_jsonl(dispatch_path, [row])
            base_result = runner._strict_json_load(
                root / "worker_macro_fsm_result.json", label="result"
            )
            with self.assertRaisesRegex(
                runner.ResidualRunnerContractError, "nominal ZERO identity"
            ):
                runner._validate_zero_ledgers(
                    request=request,
                    base_result=base_result,
                    base_terminal=terminal["base_macro_fsm_terminal"],
                )
            self.assertTrue(projection)


if __name__ == "__main__":
    unittest.main()
