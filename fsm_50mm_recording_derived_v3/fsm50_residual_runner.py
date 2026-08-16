"""Strict single-run Gate-E live R0 ZERO-residual launcher.

This runner is intentionally narrower than :mod:`fsm50_macro_runner`.  It
admits one of the three reviewed Gate-C/Gate-D sources, one exact reviewed r4
reference run, and the exact reviewed Gate-C four-run closure.  It creates one
fresh output directory, launches one fresh worker through the dedicated
``fsm50_residual_macro_request_path`` route, and accepts only the immutable
ZERO policy supplied by :mod:`worker_residual_macro_fsm_session`.

No checkpoint, action vector, policy selector, training switch, or repeat
count is part of the public API.  The durable manifest is written only by this
module and records the complete preflight, live worker, ledger, semantic
projection, and fast-close closure.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES

from . import fsm50_macro_runner as macro_runner
from .fsm50_direct_command_residual import RESIDUAL_ACTION_NAMES, ZeroResidualPolicy
from .fsm50_macro_controller import MacroControllerBundle, build_gate_c_bundle
from .fsm50_residual_envelope import (
    ALLOWED_SOURCE_VERSIONS,
    DEFAULT_CONFIG_PATH,
    load_residual_envelope,
)
from .worker_macro_fsm_session import (
    DEFAULT_POST_SETTLE_S,
    DEFAULT_TELEMETRY_HZ,
    DEFAULT_VIDEO_FPS,
    EXPECTED_RENDER_SUBSTEPS,
    SEGMENT_COMPLETION_SCHEMA,
)
from .worker_residual_macro_fsm_session import (
    EXECUTION_MODE,
    GATE_E_COMMAND_EVIDENCE_SCHEMA,
    GATE_E_TASK_INPUTS_NAME,
    GATE_E_TASK_INPUTS_SCHEMA,
    GATE_E_WORKER_RESULT_NAME,
    GATE_E_WORKER_RESULT_SCHEMA,
    OPERATION,
    POLICY_KIND,
    WorkerResidualMacroFSMRequest,
    build_worker_residual_macro_fsm_request,
    current_code_sha256,
    load_worker_residual_macro_fsm_request,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
SIM_PROCESS_CLIENT_PATH = PROJECT_ROOT / "sim_process_client.py"
SIM_WORKER_PROCESS_PATH = PROJECT_ROOT / "sim_worker_process.py"
RESIDUAL_WRAPPER_PATH = MODULE_ROOT / "worker_residual_macro_fsm_session.py"
RESIDUAL_WORKER_PATH = MODULE_ROOT / "worker_macro_fsm_session.py"
RESIDUAL_CORE_PATH = MODULE_ROOT / "fsm50_direct_command_residual.py"
RESIDUAL_ENVELOPE_PATH = MODULE_ROOT / "fsm50_residual_envelope.py"

RUNNER_MANIFEST_NAME = "fsm50_residual_r0_manifest.json"
RUNNER_MANIFEST_SCHEMA = "fsm50.residual_r0_runner_manifest.v1"
SEMANTIC_PROJECTION_SCHEMA = "fsm50.residual_r0_semantic_projection.v1"
ZERO_LEDGER_VALIDATION_SCHEMA = "fsm50.residual_r0_ledger_validation.v1"
MANUAL_VIDEO_STATUS_PENDING = "PENDING_SHA_BOUND_MANUAL_VIDEO_REVIEW"
MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS = "ZERO_R0_REVIEWED_TASK_SUCCESS"
MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS_POSTURE_INCOMPLETE = (
    "ZERO_R0_REVIEWED_TASK_SUCCESS_POSTURE_INCOMPLETE"
)
MANUAL_VERDICT_NAME = "manual_zero_r0_video_verdict.json"
MANUAL_VERDICT_TEMPLATE_NAME = "manual_zero_r0_video_verdict.template.json"
MANUAL_VERDICT_SCHEMA = "fsm50.zero_residual_manual_video_verdict.v1"
REVIEW_VALIDATION_SCHEMA = "fsm50.reviewed_zero_r0_manifest_validation.v1"
BASE_REQUEST_NAME = "worker_macro_fsm_request.json"
RESIDUAL_REQUEST_NAME = "worker_residual_macro_fsm_request.json"
REFERENCE_MANIFEST_NAME = macro_runner.RUNNER_MANIFEST_NAME

_LOWER_SHA256 = frozenset("0123456789abcdef")
_ZERO_ACTION = tuple(0.0 for _ in RESIDUAL_ACTION_NAMES)
_ACCEPTED_REVIEW_STATUSES = frozenset(
    {
        "MACRO_FSM_TASK_SUCCESS",
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
    }
)
_REVIEWED_STATUS_BY_CLASSIFICATION = MappingProxyType(
    {
        "MACRO_FSM_TASK_SUCCESS": MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS,
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE": (
            MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS_POSTURE_INCOMPLETE
        ),
    }
)
REVIEWED_ZERO_SUCCESS_STATUSES = frozenset(
    _REVIEWED_STATUS_BY_CLASSIFICATION.values()
)

_VERDICT_KEYS = frozenset(
    {
        "task_completed",
        "body_crossed_front_face",
        "required_leg_lift_completed",
        "final_recoverable",
        "posture_incomplete",
        "robot_fell",
        "body_stuck",
        "wheel_drive_up_without_required_lift",
        "dangerous_body_collision",
        "joint_limit_violation",
        "severe_penetration",
        "notes",
    }
)
_MANUAL_VERDICT_KEYS = frozenset(
    {
        "schema_version",
        "review_complete",
        "reviewed_utc",
        "source_version",
        "residual_macro_fsm_request_id",
        "request_identity_sha256",
        "base_macro_fsm_request_id",
        "bundle_sha256",
        "graph_sha256",
        "profile_library_sha256",
        "run_dir",
        "residual_request_path",
        "residual_request_sha256",
        "pending_manifest_path",
        "pending_manifest_sha256",
        "reviewed_reference_run_dir",
        "reviewed_reference_manifest_path",
        "reviewed_reference_manifest_sha256",
        "reviewed_reference_manual_review_status",
        "reference_semantic_projection_sha256",
        "outer_worker_result_path",
        "outer_worker_result_sha256",
        "outer_task_inputs_path",
        "outer_task_inputs_sha256",
        "base_worker_result_path",
        "base_worker_result_sha256",
        "base_task_inputs_path",
        "base_task_inputs_sha256",
        "video_path",
        "video_sha256",
        "start_ack_sha256",
        "terminal_sha256",
        "terminal_ack_sha256",
        "shutdown_outcome_sha256",
        "zero_ledger_validation_sha256",
        "gate_c_closure_sha256",
        "canonical_environment_lock_sha256",
        "gate_e_code_identity_sha256",
        "source_binding_sha256",
        "verdict",
    }
)

_PENDING_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "live_isaac_execution",
        "execution_mode",
        "operation",
        "policy_kind",
        "policy_id",
        "policy_sha256",
        "checkpoint_loaded",
        "ppo_training_performed",
        "source_version",
        "run_dir",
        "request_id",
        "request_identity_sha256",
        "residual_request_path",
        "residual_request_sha256",
        "base_request_id",
        "base_request_path",
        "base_request_sha256",
        "bundle",
        "bundle_sha256",
        "graph_sha256",
        "profile_library_sha256",
        "canonical_environment_lock",
        "gate_e_code_identity",
        "source_binding",
        "gate_c_closure",
        "gate_c_closure_sha256",
        "reviewed_reference",
        "reference_semantic_projection_sha256",
        "singleton_revalidation_equal",
        "worker_binding",
        "start_ack",
        "terminal",
        "terminal_ack",
        "terminal_validation",
        "semantic_projection_equal",
        "video_path",
        "video_sha256",
        "shutdown_outcome",
        "shutdown_verified",
        "macro_fsm_complete",
        "manual_video_status",
        "error",
    }
)
_REVIEW_MANIFEST_KEYS = frozenset(
    {
        "manual_review_classification",
        "manual_verdict_schema",
        "manual_verdict_path",
        "manual_verdict_sha256",
        "pre_review_manifest_sha256",
        "reviewed_utc",
    }
)
_REVIEWED_MANIFEST_KEYS = _PENDING_MANIFEST_KEYS.union(_REVIEW_MANIFEST_KEYS)
REVIEW_VALIDATION_KEYS = frozenset(
    {
        "schema_version",
        "manifest_path",
        "manifest_sha256",
        "source_version",
        "request_id",
        "request_identity_sha256",
        "base_request_id",
        "bundle_sha256",
        "graph_sha256",
        "profile_library_sha256",
        "canonical_environment_lock_sha256",
        "manual_review_status",
        "manual_review_classification",
        "reviewed_utc",
        "manual_verdict_path",
        "manual_verdict_sha256",
        "video_path",
        "video_sha256",
        "worker_result_path",
        "worker_result_sha256",
        "task_inputs_path",
        "task_inputs_sha256",
        "base_worker_result_path",
        "base_worker_result_sha256",
        "base_task_inputs_path",
        "base_task_inputs_sha256",
        "gate_e_code_identity_sha256",
        "source_binding_sha256",
        "gate_c_closure_sha256",
        "reference_semantic_projection_sha256",
        "zero_ledger_validation_sha256",
        "shutdown_verified",
        "semantic_projection_equal",
        "macro_fsm_complete",
        "reviewed_success",
    }
)

_V003 = "v003_20260805_224517_157723_manual"
_V008 = "v008_20260806_211408_578700_manual"
_V009 = "v009_20260806_215232_433234_manual"

EXACT_REVIEWED_REFERENCE_RUNS = MappingProxyType(
    {
        _V003: (
            MODULE_ROOT
            / "runs"
            / "v003_macro_fsm_completion_aware_coalesced_r4"
            / _V003
            / "baseline"
            / "20260815T105857_601132Z_baseline_00_5742cda6e438"
        ).resolve(),
        _V008: (
            MODULE_ROOT
            / "runs"
            / "cross_version_macro_fsm_completion_aware_coalesced_r4"
            / _V008
            / "trials"
            / "20260815T113449_254770Z_cross_version_00_0a47cee05cbb"
        ).resolve(),
        _V009: (
            MODULE_ROOT
            / "runs"
            / "cross_version_macro_fsm_completion_aware_coalesced_r4"
            / _V009
            / "trials"
            / "20260815T114309_706871Z_cross_version_00_0e5686354eed"
        ).resolve(),
    }
)
EXACT_REVIEWED_REFERENCE_MANIFEST_SHA256 = MappingProxyType(
    {
        _V003: "e4f74586cda839724ae070473e1dfd34704e1750d44e9129751ec0e6d12fb499",
        _V008: "7108c69d9f03a843a19405114a61c696616d068e5e05e8f7dc4c4d1d68ef619e",
        _V009: "e46d6d06fb02ce967c911b93f14a8fdfed4b7bef13a41a4b0f66bfd3df000691",
    }
)
EXACT_GATE_C_BASELINE_RUN = EXACT_REVIEWED_REFERENCE_RUNS[_V003]
EXACT_GATE_C_BASELINE_MANIFEST_SHA256 = (
    EXACT_REVIEWED_REFERENCE_MANIFEST_SHA256[_V003]
)
EXACT_GATE_C_CLOSURE_SHA256 = (
    "800edbe4aa2feb47d690cd545222d77d5d032533553ad945c72794f727088a17"
)

_OUTER_TASK_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "execution_mode",
        "operation",
        "gate_e_zero_residual",
        "base_task_inputs_path",
        "base_task_inputs_sha256",
        "base_task_inputs",
        "residual_command_evidence",
    }
)
_OUTER_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "execution_mode",
        "operation",
        "gate_e_zero_residual",
        "base_worker_result_path",
        "base_worker_result_sha256",
        "base_worker_result",
        "task_inputs_path",
        "task_inputs_sha256",
        "task_inputs",
        "residual_command_evidence",
        "macro_fsm_complete",
        "error",
    }
)
_DISPATCH_ROW_KEYS = frozenset(
    {
        "schema_version",
        "dispatch_index",
        "sim_time_s",
        "sim_step",
        "macro_state",
        "subphase",
        "profile_id",
        "profile_source_version",
        "profile_strategy",
        "command_epoch",
        "command_provenance",
        "source_action_consumption_index",
        "servo_targets_deg",
        "wheel_targets_rad_s",
        "changed_servo_names",
        "changed_wheel_names",
        "concurrent",
        "batch_id",
        "ack",
        "n_plus_one_verified",
        "n_plus_one_verified_sim_step",
        "n_plus_one_readback_sha256",
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
    }
)
_SOURCE_ROW_KEYS = frozenset(
    {
        "schema_version",
        "source_action_index",
        "expected_source_action_count",
        "sim_time_s",
        "sim_step",
        "profile_id",
        "profile_library_sha256",
        "profile_source_version",
        "profile_strategy",
        "macro_state",
        "subphase",
        "source_plan_sha256",
        "bundle_sha256",
        "command_provenance",
        "target_changed",
        "dispatch_epoch",
        "servo_targets_deg",
        "wheel_targets_rad_s",
        "pre_action_verified_command_epoch",
        "pre_action_verified_readback_sha256",
        "physical_dispatch_required",
        "physical_dispatch_applied",
        "physical_dispatch_index",
        "batch_id",
        "n_plus_one_verified",
        "n_plus_one_verified_sim_step",
        "n_plus_one_readback_sha256",
        "nominal_target_changed",
        "applied_target_changed",
        "physical_command_epoch",
        "applied_servo_targets_deg",
        "applied_wheel_targets_rad_s",
        "residual_transform_sha256",
    }
)
_COMPLETION_ROW_KEYS = frozenset(macro_runner._SEGMENT_COMPLETION_ROW_KEYS).union(
    {
        "effective_completion_servo_targets_deg",
        "effective_completion_targets_sha256",
        "nominal_completion_spec_sha256",
        "pre_action_applied_servo_targets_deg",
        "start_physical_command_epoch",
        "latched_servo_residual_deg",
        "nominal_target_changed",
        "applied_target_changed",
    }
)
_TRANSITION_ROW_KEYS = frozenset(
    {
        "schema_version",
        "transition_index",
        "sim_time_s",
        "sim_step",
        "from_state",
        "to_state",
        "subphase",
        "profile_id",
        "profile_source_version",
        "profile_strategy",
        "command_epoch",
        "phase_elapsed_s",
        "profile_fraction",
        "events",
        "reason",
        "guard_evidence",
        "retry_count",
        "observation_sha256",
    }
)
_TARGET_READBACK_KEYS = frozenset(
    {
        "sim_step",
        "command_epoch",
        "batch_id",
        "canonical_servo_targets_deg",
        "canonical_wheel_targets_rad_s",
        "actual_servo_drive_targets_rad",
        "actual_wheel_drive_targets_rad_s",
        "adapter_runtime_instance_id",
        "root_state_write_count",
        "physics_dt_s",
    }
)


class ResidualRunnerContractError(RuntimeError):
    """A fail-closed R0 admission, execution, or evidence check failed."""


@dataclass(frozen=True)
class ResidualRunPaths:
    run_dir: Path
    base_request_path: Path
    residual_request_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ResidualAttempt:
    run_dir: Path
    source_version: str
    request_id: str
    macro_fsm_complete: bool
    shutdown_verified: bool
    semantic_projection_equal: bool
    manifest_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResidualRunnerContractError("non-finite value in durable evidence")
        return value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise ResidualRunnerContractError(
        f"unsupported durable evidence type: {type(value).__name__}"
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _serialized_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact bytes emitted by :func:`_atomic_write_json`."""

    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_utc(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ResidualRunnerContractError(f"{label} is not exact UTC text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResidualRunnerContractError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResidualRunnerContractError(f"{label} is not UTC")
    return value


def _strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_LOWER_SHA256)
    )


def _require_sha(value: Any, *, label: str) -> str:
    if not _is_lower_sha256(value):
        raise ResidualRunnerContractError(f"{label} is not a lowercase SHA-256")
    return str(value)


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    raw = value.get(key)
    if type(raw) is not str or not raw or raw != raw.strip():
        raise ResidualRunnerContractError(f"{label} {key} is not exact text")
    return raw


def _strict_json_load(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ResidualRunnerContractError(f"{label} is missing: {source}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ResidualRunnerContractError(
                    f"duplicate JSON key in {label}: {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ResidualRunnerContractError(
                    f"non-finite JSON token in {label}: {token}"
                )
            ),
        )
    except ResidualRunnerContractError:
        raise
    except Exception as exc:
        raise ResidualRunnerContractError(f"cannot load {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ResidualRunnerContractError(f"{label} is not a JSON object")
    return value


def _strict_jsonl(path: str | Path, *, label: str) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise ResidualRunnerContractError(f"{label} is missing: {source}")
    text = source.read_text(encoding="utf-8")
    if not text or not text.endswith("\n"):
        raise ResidualRunnerContractError(
            f"{label} is empty or lacks its final newline"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ResidualRunnerContractError(
                f"{label} contains a blank row at line {line_number}"
            )
        try:
            row = json.loads(
                line,
                object_pairs_hook=lambda pairs: _pairs_without_duplicates(
                    pairs, label=f"{label}:{line_number}"
                ),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ResidualRunnerContractError(
                        f"non-finite JSON token in {label}:{line_number}: {token}"
                    )
                ),
            )
        except ResidualRunnerContractError:
            raise
        except Exception as exc:
            raise ResidualRunnerContractError(
                f"cannot load {label}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ResidualRunnerContractError(
                f"{label}:{line_number} is not a JSON object"
            )
        rows.append(row)
    return rows


def _pairs_without_duplicates(
    pairs: list[tuple[str, Any]], *, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResidualRunnerContractError(
                f"duplicate JSON key in {label}: {key!r}"
            )
        result[key] = value
    return result


def _path_in_run(path: str | Path, run_dir: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ResidualRunnerContractError(
            f"{label} escapes the exact run directory"
        ) from exc
    if not resolved.is_file():
        raise ResidualRunnerContractError(f"{label} is missing: {resolved}")
    return resolved


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    macro_runner.atomic_write_json(path, _jsonable(value))


def _select(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: copy.deepcopy(value.get(key)) for key in keys}


def _semantic_projection(run_dir: str | Path) -> dict[str, Any]:
    """Project only controller-owned semantics; physics timing is diagnostic."""

    run = Path(run_dir).resolve()
    result = _strict_json_load(
        run / "worker_macro_fsm_result.json", label="base Macro worker result"
    )
    inputs = _strict_json_load(
        run / "macro_task_inputs.json", label="base Macro task inputs"
    )
    source_rows = _strict_jsonl(
        _path_in_run(
            _required_text(result, "source_action_consumption_path", label="result"),
            run,
            label="source-action ledger",
        ),
        label="source-action ledger",
    )
    dispatch_rows = _strict_jsonl(
        _path_in_run(
            _required_text(result, "dispatch_ledger_path", label="result"),
            run,
            label="dispatch ledger",
        ),
        label="dispatch ledger",
    )
    completion_rows = _strict_jsonl(
        _path_in_run(
            _required_text(result, "segment_completion_ledger_path", label="result"),
            run,
            label="completion ledger",
        ),
        label="completion ledger",
    )
    transition_rows = _strict_jsonl(
        _path_in_run(
            _required_text(result, "transition_evidence_path", label="result"),
            run,
            label="transition ledger",
        ),
        label="transition ledger",
    )

    source_projection = [
        _select(
            row,
            (
                "source_action_index",
                "profile_id",
                "profile_source_version",
                "profile_strategy",
                "macro_state",
                "subphase",
                "command_provenance",
                "source_plan_sha256",
                "target_changed",
                "dispatch_epoch",
                "servo_targets_deg",
                "wheel_targets_rad_s",
            ),
        )
        for row in source_rows
    ]
    dispatch_projection = []
    for row in dispatch_rows:
        item = _select(
            row,
            (
                "dispatch_index",
                "macro_state",
                "subphase",
                "profile_id",
                "profile_source_version",
                "profile_strategy",
                "command_epoch",
                "command_provenance",
                "source_action_consumption_index",
                "changed_servo_names",
                "changed_wheel_names",
                "concurrent",
            ),
        )
        item["servo_targets_deg"] = copy.deepcopy(
            row.get("nominal_servo_targets_deg", row.get("servo_targets_deg"))
        )
        item["wheel_targets_rad_s"] = copy.deepcopy(
            row.get("nominal_wheel_targets_rad_s", row.get("wheel_targets_rad_s"))
        )
        dispatch_projection.append(item)

    completion_projection = []
    for row in completion_rows:
        wheel_stop = row.get("wheel_stop")
        semantic_stop = None
        if isinstance(wheel_stop, Mapping):
            semantic_stop = _select(
                wheel_stop,
                (
                    "generated",
                    "source_action",
                    "source_action_consumption_index",
                    "n_plus_one_verified",
                ),
            )
        item = _select(
            row,
            (
                "segment_completion_index",
                "source_version",
                "profile_id",
                "profile_source_version",
                "owner_state",
                "source_plan_sha256",
                "source_plan_payload_sha256",
                "accepted_steps_sha256",
                "source_segment_index",
                "source_step_index",
                "source_step_id",
                "completion_spec",
                "effective_completion_spec",
                "dynamic_servo_duration_s",
                "effective_servo_reference_velocity_deg_s",
                "pre_action_canonical_servo_targets_deg",
                "start_source_action_identity",
                "source_action_consumption_index",
                "start_command_epoch",
                "start_physical_dispatch",
                "retained_epoch_same_target",
                "tracking_begin_count",
                "tracking_end_count",
                "tracking_end_attempt_count",
                "tracking_lifecycle_closed",
                "tracking_end_reason",
                "terminal_kind",
            ),
        )
        item["wheel_stop"] = semantic_stop
        completion_projection.append(item)

    transition_projection = [
        _select(
            row,
            (
                "transition_index",
                "from_state",
                "to_state",
                "subphase",
                "profile_id",
                "profile_source_version",
                "profile_strategy",
                "command_epoch",
                "events",
                "reason",
                "retry_count",
            ),
        )
        for row in transition_rows
    ]
    completed = inputs.get("completed_result")
    if not isinstance(completed, Mapping):
        raise ResidualRunnerContractError(
            "base Macro task inputs lack completed_result"
        )
    outcome = {
        **_select(
            result,
            (
                "source_version",
                "macro_fsm_complete",
                "controller_terminal_outcome",
                "controller_terminal_reason",
                "expected_source_action_count",
                "source_action_consumption_count",
                "source_action_coverage_complete",
                "segment_completion_count",
                "expected_segment_completion_count",
                "segment_completion_coverage_complete",
                "transition_count",
                "safe_stop_status",
                "safe_stop_verified",
            ),
        ),
        "completed_result": _select(
            completed,
            (
                "source_version",
                "expected_step_count",
                "expected_segment_count",
                "completed_segment_count",
                "expected_segment_completion_count",
                "segment_completion_count",
                "source_action_coverage_complete",
                "segment_completion_coverage_complete",
                "consumed_segment_start_count",
                "dispatch_complete",
                "scheduler_complete",
                "safe_stop_status",
                "safe_stop_verified",
                "body_crossed_front_face",
                "required_leg_lift_completed",
                "final_recoverable",
                "wheel_drive_up_without_required_lift",
                "nonfinite_core_state_detected",
            ),
        ),
    }
    projection = {
        "schema_version": SEMANTIC_PROJECTION_SCHEMA,
        "diagnostic_only_exclusions": [
            "wall_clock_and_sim_timing",
            "physics_feedback_and_completion_diagnostics",
            "request_worker_session_adapter_and_batch_ids",
            "video_and_telemetry_measurements",
        ],
        "source_actions": source_projection,
        "nominal_dispatches": dispatch_projection,
        "segment_completions": completion_projection,
        "transitions": transition_projection,
        "outcome": outcome,
    }
    return projection


def _artifact_identity(run_dir: Path) -> dict[str, Any]:
    result_path = run_dir / "worker_macro_fsm_result.json"
    result = _strict_json_load(result_path, label="reference worker result")
    paths = {
        "worker_result": result_path,
        "task_inputs": run_dir / "macro_task_inputs.json",
        "source_action_ledger": Path(
            _required_text(result, "source_action_consumption_path", label="result")
        ).resolve(),
        "dispatch_ledger": Path(
            _required_text(result, "dispatch_ledger_path", label="result")
        ).resolve(),
        "completion_ledger": Path(
            _required_text(result, "segment_completion_ledger_path", label="result")
        ).resolve(),
        "transition_ledger": Path(
            _required_text(result, "transition_evidence_path", label="result")
        ).resolve(),
    }
    identity: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = _path_in_run(path, run_dir, label=f"reference {name}")
        identity[f"{name}_path"] = str(resolved)
        identity[f"{name}_sha256"] = sha256_file(resolved)
    return identity


def _validate_reviewed_reference(
    *,
    source_version: str,
    run_dir: str | Path,
    bundle: MacroControllerBundle,
    gate_c_closure: Mapping[str, Any],
    alignment_path: str | Path,
    task_success_table_path: str | Path,
    expected_paths: Mapping[str, Path] = EXACT_REVIEWED_REFERENCE_RUNS,
    expected_manifest_sha256: Mapping[str, str] = (
        EXACT_REVIEWED_REFERENCE_MANIFEST_SHA256
    ),
) -> dict[str, Any]:
    if source_version not in ALLOWED_SOURCE_VERSIONS:
        raise ResidualRunnerContractError("R0 source is not reviewed/authorized")
    run = Path(run_dir).resolve()
    if run != Path(expected_paths[source_version]).resolve():
        raise ResidualRunnerContractError(
            "reviewed reference run is not the exact source-matched r4 run"
        )
    manifest_path = run / REFERENCE_MANIFEST_NAME
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != expected_manifest_sha256[source_version]
    ):
        raise ResidualRunnerContractError(
            "reviewed reference manifest bytes differ from the frozen r4 review"
        )
    manifest = _strict_json_load(manifest_path, label="reviewed reference manifest")
    expected_kind = "baseline" if source_version == _V003 else "cross_version"
    if (
        manifest.get("schema_version") != macro_runner.RUNNER_MANIFEST_SCHEMA
        or manifest.get("run_dir") != str(run)
        or manifest.get("source_version") != source_version
        or manifest.get("trial_kind") != expected_kind
        or manifest.get("trial_index") != 0
        or manifest.get("controller_complete") is not True
        or manifest.get("shutdown_verified") is not True
        or manifest.get("manual_review_status") not in _ACCEPTED_REVIEW_STATUSES
        or str(manifest.get("error", "") or "")
    ):
        raise ResidualRunnerContractError(
            "reviewed reference manifest is not a reviewed successful attempt"
        )
    if (
        bundle.primary_source_version != source_version
        or manifest.get("bundle") != _jsonable(bundle.to_mapping())
        or manifest.get("bundle_sha256") != bundle.bundle_sha256
    ):
        raise ResidualRunnerContractError(
            "reviewed reference bundle is not the current source bundle"
        )

    request_path = _path_in_run(
        _required_text(manifest, "request_path", label="reference manifest"),
        run,
        label="reference request",
    )
    if (
        request_path != (run / BASE_REQUEST_NAME).resolve()
        or manifest.get("request_sha256") != sha256_file(request_path)
    ):
        raise ResidualRunnerContractError("reviewed reference request is stale")
    request = _strict_json_load(request_path, label="reviewed reference request")
    loaded = macro_runner.load_worker_macro_fsm_request(request_path)
    if loaded is None:
        raise ResidualRunnerContractError("reviewed reference request did not load")
    alignment = Path(alignment_path).resolve()
    success_table = Path(task_success_table_path).resolve()
    for key, expected in {
        "source_version": source_version,
        "profile_id": bundle.profiles.library_id,
        "graph_id": bundle.graph.graph_id,
        "graph_sha256": bundle.graph_sha256,
        "profile_library_sha256": bundle.profile_library_sha256,
        "bundle_sha256": bundle.bundle_sha256,
        "run_dir": str(run),
        "alignment_path": str(alignment),
        "alignment_sha256": sha256_file(alignment),
        "task_success_table_path": str(success_table),
        "task_success_table_sha256": sha256_file(success_table),
        "trial_kind": expected_kind,
        "trial_index": 0,
        "capture_video": True,
        "video_fps": DEFAULT_VIDEO_FPS,
        "filtered_contact_bank_enabled": False,
    }.items():
        if request.get(key) != expected:
            raise ResidualRunnerContractError(
                f"reviewed reference request {key} mismatch"
            )

    if source_version == _V003:
        baseline = gate_c_closure.get("baseline")
        if (
            not isinstance(baseline, Mapping)
            or baseline.get("run_dir") != str(run)
            or baseline.get("runner_manifest_sha256") != sha256_file(manifest_path)
        ):
            raise ResidualRunnerContractError(
                "v003 reference is not the exact Gate-C closure baseline"
            )
        if "gate_c_closure" in manifest or "gate_c_closure_sha256" in manifest:
            raise ResidualRunnerContractError(
                "v003 reference unexpectedly contains Gate-D closure fields"
            )
    else:
        stored_closure = manifest.get("gate_c_closure")
        if (
            not isinstance(stored_closure, Mapping)
            or not _strict_json_equal(dict(stored_closure), dict(gate_c_closure))
            or manifest.get("gate_c_closure_sha256")
            != gate_c_closure.get("closure_sha256")
        ):
            raise ResidualRunnerContractError(
                "Gate-D reference does not bind the exact current Gate-C closure"
            )

    worker_binding = macro_runner.validate_macro_worker_binding(
        dict(manifest.get("worker_binding", {}) or {}), request=request
    )
    terminal = macro_runner.validate_macro_terminal(
        dict(manifest.get("terminal", {}) or {}),
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
        bundle=bundle,
    )
    if terminal.get("type") != "macro_fsm_complete":
        raise ResidualRunnerContractError("reviewed reference terminal is not complete")
    macro_runner._validate_terminal_ack(
        dict(manifest.get("terminal_ack", {}) or {}),
        terminal=terminal,
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
    )
    shutdown = dict(manifest.get("shutdown_outcome", {}) or {})
    shutdown_id = _required_text(shutdown, "request_id", label="reference shutdown")
    macro_runner.validate_fast_shutdown(
        shutdown,
        shutdown_request_id=shutdown_id,
        macro_request_id=request["request_id"],
        worker_binding=worker_binding,
    )
    verdict_path = _path_in_run(
        _required_text(manifest, "manual_verdict_path", label="reference manifest"),
        run,
        label="reference manual verdict",
    )
    if (
        verdict_path != (run / macro_runner.MANUAL_VERDICT_NAME).resolve()
        or manifest.get("manual_verdict_sha256") != sha256_file(verdict_path)
    ):
        raise ResidualRunnerContractError("reviewed reference verdict is stale")
    classification = macro_runner._validate_manual_verdict_document(
        _strict_json_load(verdict_path, label="reference manual verdict"),
        request=request,
        terminal=terminal,
        run=run,
    )
    if classification != manifest.get("manual_review_status"):
        raise ResidualRunnerContractError(
            "reviewed reference verdict classification mismatch"
        )

    projection = _semantic_projection(run)
    identity = {
        "run_dir": str(run),
        "source_version": source_version,
        "trial_kind": expected_kind,
        "trial_index": 0,
        "manual_review_status": classification,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "manual_verdict_path": str(verdict_path),
        "manual_verdict_sha256": sha256_file(verdict_path),
        "request_id": request["request_id"],
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "bundle_sha256": bundle.bundle_sha256,
        "graph_sha256": bundle.graph_sha256,
        "profile_library_sha256": bundle.profile_library_sha256,
        **_artifact_identity(run),
        "semantic_projection_sha256": _canonical_json_sha256(projection),
    }
    return {"identity": identity, "projection": projection}


def _validate_gate_c_admission(
    args: argparse.Namespace,
    *,
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
) -> dict[str, Any]:
    baseline = Path(args.gate_c_baseline_run_dir).resolve()
    if baseline != EXACT_GATE_C_BASELINE_RUN:
        raise ResidualRunnerContractError(
            "Gate-C input is not the exact reviewed r4 baseline"
        )
    manifest_path = baseline / REFERENCE_MANIFEST_NAME
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != EXACT_GATE_C_BASELINE_MANIFEST_SHA256
    ):
        raise ResidualRunnerContractError(
            "Gate-C baseline manifest differs from the frozen reviewed bytes"
        )
    try:
        closure = dict(
            macro_runner._validate_gate_c_closure(
                argparse.Namespace(
                    gate_c_baseline_run_dir=baseline,
                    alignment_path=args.alignment_path,
                    task_success_table_path=args.task_success_table_path,
                ),
                bundle_builder=bundle_builder,
            )
        )
    except Exception as exc:
        raise ResidualRunnerContractError(
            f"Gate-C reviewed closure is not current: {exc}"
        ) from exc
    if closure.get("closure_sha256") != EXACT_GATE_C_CLOSURE_SHA256:
        raise ResidualRunnerContractError(
            "Gate-C closure digest differs from the exact reviewed r4 closure"
        )
    return closure


def _current_gate_e_code_identity() -> dict[str, Any]:
    """Return every code/config digest on the live R0 route."""

    wrapper_code = dict(current_code_sha256())
    required_paths = {
        "fsm50_residual_runner.py": Path(__file__).resolve(),
        "worker_residual_macro_fsm_session.py": RESIDUAL_WRAPPER_PATH,
        "worker_macro_fsm_session.py": RESIDUAL_WORKER_PATH,
        "fsm50_direct_command_residual.py": RESIDUAL_CORE_PATH,
        "fsm50_residual_envelope.py": RESIDUAL_ENVELOPE_PATH,
        "sim_worker_process.py": SIM_WORKER_PROCESS_PATH,
        "sim_process_client.py": SIM_PROCESS_CLIENT_PATH,
    }
    code_sha256 = dict(wrapper_code)
    for name, path in required_paths.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ResidualRunnerContractError(
                f"Gate-E live route code is missing: {resolved}"
            )
        actual = sha256_file(resolved)
        if name in code_sha256 and code_sha256[name] != actual:
            raise ResidualRunnerContractError(
                f"Gate-E wrapper code binding disagrees for {name}"
            )
        code_sha256[name] = actual
    for name, digest in code_sha256.items():
        _require_sha(digest, label=f"Gate-E code {name}")

    envelope = load_residual_envelope(DEFAULT_CONFIG_PATH, verify_evidence=True)
    if not envelope.evidence_verified:
        raise ResidualRunnerContractError(
            "Gate-E residual envelope evidence is not reviewed/current"
        )
    config_sha = sha256_file(DEFAULT_CONFIG_PATH)
    core_sha = sha256_file(RESIDUAL_CORE_PATH)
    if (
        wrapper_code.get(RESIDUAL_CORE_PATH.name) != core_sha
        or wrapper_code.get(RESIDUAL_ENVELOPE_PATH.name)
        != sha256_file(RESIDUAL_ENVELOPE_PATH)
    ):
        raise ResidualRunnerContractError(
            "Gate-E wrapper/core/envelope code bindings disagree"
        )
    payload = {
        "code_sha256": dict(sorted(code_sha256.items())),
        "envelope_config_path": str(DEFAULT_CONFIG_PATH.resolve()),
        "envelope_config_file_sha256": config_sha,
        "envelope_canonical_sha256": envelope.canonical_sha256,
        "residual_core_path": str(RESIDUAL_CORE_PATH.resolve()),
        "residual_core_sha256": core_sha,
    }
    return {**payload, "identity_sha256": _canonical_json_sha256(payload)}


def _build_production_worker_args(
    cli_args: argparse.Namespace, residual_request_path: Path
) -> argparse.Namespace:
    from height_replay_ui import build_parser as build_ui_parser
    from height_replay_ui import normalize_motion_args

    worker_args = build_ui_parser().parse_args([])
    worker_args.height_mm = 50
    worker_args.height_cm = None
    worker_args.profile = "fast"
    worker_args.render_interval = EXPECTED_RENDER_SUBSTEPS
    worker_args.headless = False
    worker_args.no_sim = False
    worker_args.sim_launch_mode = "subprocess"
    worker_args.fsm50_gate_request_path = ""
    worker_args.fsm50_task_request_path = ""
    worker_args.fsm50_macro_request_path = ""
    worker_args.fsm50_residual_macro_request_path = str(
        residual_request_path.resolve()
    )
    for name in (
        "device",
        "worker_launch_mode",
        "worker_python_exe",
        "isaaclab_bat",
        "sim_startup_timeout_s",
        "sim_worker_status_timeout_s",
        "sim_worker_log_lines",
        "accept_isaac_eula",
        "livestream",
        "experience",
    ):
        if hasattr(cli_args, name):
            setattr(worker_args, name, getattr(cli_args, name))
    normalize_motion_args(worker_args)
    if (
        str(getattr(worker_args, "fsm50_residual_macro_request_path", "") or "")
        != str(residual_request_path.resolve())
        or any(
            str(getattr(worker_args, name, "") or "")
            for name in (
                "fsm50_gate_request_path",
                "fsm50_task_request_path",
                "fsm50_macro_request_path",
            )
        )
    ):
        raise ResidualRunnerContractError(
            "worker args did not preserve the exclusive Gate-E request path"
        )
    return worker_args


def _default_client_factory(args: argparse.Namespace) -> Any:
    from sim_process_client import SimProcessClient

    return SimProcessClient(args)


def _wait_for_residual_terminal(
    client: Any,
    *,
    request_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() <= deadline:
        client.poll()
        status = dict(client.status() or {})
        for raw in (
            getattr(client, "latest_macro_fsm_terminal", None),
            status.get("last_macro_fsm_terminal"),
        ):
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            if (
                row.get("type") in {"macro_fsm_complete", "macro_fsm_failed"}
                and row.get("operation") == OPERATION
                and row.get("request_id") == request_id
            ):
                return row
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise ResidualRunnerContractError(
                "Gate-E worker exited before residual terminal"
            )
        time.sleep(max(0.001, float(poll_interval_s)))
    raise ResidualRunnerContractError("timed out waiting for Gate-E terminal")


def _fresh_identity_values(reference: Mapping[str, Any], closure: Mapping[str, Any]) -> dict[str, set[Any]]:
    values = {
        "request_id": {reference.get("request_id")},
        "worker_pid": {reference.get("worker_pid")},
        "worker_session_id": {reference.get("worker_session_id")},
        "adapter_runtime_instance_id": {
            reference.get("adapter_runtime_instance_id")
        },
    }
    attempts: list[Any] = [closure.get("baseline")]
    attempts.extend(list(closure.get("repeats", []) or []))
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        for key in values:
            values[key].add(attempt.get(key))
    return values


def _validate_worker_binding(
    status: Mapping[str, Any],
    *,
    request: WorkerResidualMacroFSMRequest,
    stale_identities: Mapping[str, set[Any]],
) -> dict[str, Any]:
    result = dict(status)
    preflight = result.get("worker_residual_macro_fsm_preflight")
    session = result.get("worker_residual_macro_fsm_session")
    if (
        result.get("ready") is not True
        or result.get("residual_macro_fsm_preflight_ready") is not True
        or not isinstance(preflight, Mapping)
        or not isinstance(session, Mapping)
    ):
        raise ResidualRunnerContractError(
            "Gate-E worker residual preflight/session is not ready"
        )
    expected_preflight = request.preflight_payload()
    if not _strict_json_equal(dict(preflight), expected_preflight):
        raise ResidualRunnerContractError(
            "Gate-E worker preflight differs from the loaded request"
        )
    for key, expected in {
        "execution_mode": EXECUTION_MODE,
        "operation": OPERATION,
        "residual_macro_fsm_request_id": request.request_id,
        "base_macro_fsm_request_id": request.base_request.request_id,
        "state": "ready_for_start",
    }.items():
        if session.get(key) != expected:
            raise ResidualRunnerContractError(
                f"Gate-E worker session {key} mismatch"
            )
    if not _strict_json_equal(
        session.get("gate_e_zero_residual"),
        request.gate_e_identity(payload_role="status"),
    ):
        raise ResidualRunnerContractError("Gate-E status identity mismatch")
    direct = session.get("direct_command_residual")
    if (
        not isinstance(direct, Mapping)
        or direct.get("enabled") is not True
        or direct.get("policy_id") != request.policy_id
        or direct.get("policy_sha256") != request.policy_sha256
        or direct.get("transform_count") != 0
        or direct.get("physical_command_epoch") != 0
    ):
        raise ResidualRunnerContractError(
            "Gate-E ready session is not pristine ZERO-R0"
        )
    for key in (
        "worker_pid",
        "worker_session_id",
        "adapter_runtime_instance_id",
    ):
        value = result.get(key)
        valid = type(value) is int and value > 0 if key == "worker_pid" else (
            type(value) is str and bool(value)
        )
        if (
            not valid
            or value in stale_identities.get(key, set())
            or (key == "worker_pid" and value == os.getpid())
        ):
            raise ResidualRunnerContractError(
                f"Gate-E worker {key} is invalid or not fresh"
            )
    runtime_version = result.get("runtime_version")
    if (
        type(runtime_version) is not str
        or not runtime_version
        or runtime_version.lower() == "unavailable"
        or result.get("root_state_write_count") != 0
    ):
        raise ResidualRunnerContractError(
            "Gate-E worker runtime/root-state identity is invalid"
        )
    for name in ("worker_artifact_session", "worker_artifact_preflight"):
        disabled = result.get(name)
        if not isinstance(disabled, Mapping) or disabled.get("enabled") is not False:
            raise ResidualRunnerContractError(
                f"strict artifact pipeline unexpectedly enabled: {name}"
            )
    for forbidden in (
        "worker_task_replay_session",
        "worker_task_replay_preflight",
        "worker_macro_fsm_session",
        "worker_macro_fsm_preflight",
    ):
        if forbidden in result:
            raise ResidualRunnerContractError(
                f"non-Gate-E worker route is present: {forbidden}"
            )
    return result


def _send_residual_start(
    client: Any,
    *,
    request: WorkerResidualMacroFSMRequest,
    worker_session_id: str,
) -> None:
    method = getattr(client, "start_residual_macro_fsm", None)
    if not callable(method):
        raise ResidualRunnerContractError(
            "SimProcessClient has no public Gate-E residual start API"
        )
    kwargs = {
        "request_id": request.request_id,
        "worker_session_id": worker_session_id,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "request_identity_sha256": request.request_identity_sha256,
        "policy_kind": POLICY_KIND,
        "policy_sha256": request.policy_sha256,
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
    }
    parameters = inspect.signature(method).parameters
    if not any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    ):
        if set(parameters) != set(kwargs):
            raise ResidualRunnerContractError(
                "SimProcessClient Gate-E start ABI is not exact"
            )
    method(**kwargs)


def _validate_start_ack(
    acknowledgement: Mapping[str, Any],
    *,
    request: WorkerResidualMacroFSMRequest,
    worker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    ack = dict(acknowledgement)
    for key, expected in {
        "type": "operation_ack",
        "operation": "start_macro_fsm",
        "request_id": request.request_id,
        "accepted": True,
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "artifact_request_id": "",
        "root_state_write_count": 0,
        "physics_dt_s": 1.0 / 120.0,
        "execution_mode": EXECUTION_MODE,
        "request_identity_sha256": request.request_identity_sha256,
        "residual_macro_fsm_request_id": request.request_id,
        "base_macro_fsm_request_id": request.base_request.request_id,
        "base_macro_fsm_request_sha256": request.base_request_sha256,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
    }.items():
        if ack.get(key) != expected:
            raise ResidualRunnerContractError(f"Gate-E start ACK {key} mismatch")
    if str(ack.get("error", "") or "") or str(
        ack.get("rejection_reason", "") or ""
    ):
        raise ResidualRunnerContractError("Gate-E start ACK reports rejection/error")
    if not _strict_json_equal(
        ack.get("gate_e_zero_residual"),
        request.gate_e_identity(payload_role="start_ack"),
    ):
        raise ResidualRunnerContractError("Gate-E start ACK identity mismatch")
    boundary = ack.get("start_boundary_ack")
    if not isinstance(boundary, Mapping):
        raise ResidualRunnerContractError("Gate-E start boundary ACK is missing")
    applied_step = boundary.get("applied_sim_step")
    first_step = boundary.get("first_physics_step")
    if (
        type(applied_step) is not int
        or type(first_step) is not int
        or first_step != applied_step + 1
        or boundary.get("motion_start_skew_s") != 0.0
        or boundary.get("physics_dt_s") != 1.0 / 120.0
        or set(dict(boundary.get("servo_targets_applied", {}) or {}))
        != set(SERVO_JOINT_NAMES)
        or dict(boundary.get("wheel_targets_applied", {}) or {})
        != {name: 0.0 for name in WHEEL_JOINT_NAMES}
        or ack.get("first_controller_tick_physics_step") != first_step
        or ack.get("earliest_profile_dispatch_physics_step")
        != first_step + EXPECTED_RENDER_SUBSTEPS - 1
        or ack.get("earliest_profile_actuation_physics_step")
        != first_step + EXPECTED_RENDER_SUBSTEPS
    ):
        raise ResidualRunnerContractError(
            "Gate-E zero-wheel start boundary ACK is invalid"
        )
    return ack


def _is_exact_zero_vector(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == len(RESIDUAL_ACTION_NAMES)
        and all(type(item) in (int, float) and float(item) == 0.0 for item in value)
    )


def _validate_zero_transform(
    value: Any,
    *,
    request: WorkerResidualMacroFSMRequest,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualRunnerContractError(f"{label} is not an object")
    transform = copy.deepcopy(dict(value))
    for key in (
        "raw_normalized_action",
        "clipped_normalized_action",
        "previous_applied_residual",
        "requested_residual",
        "rate_limited_residual",
        "applied_residual",
    ):
        if not _is_exact_zero_vector(transform.get(key)):
            raise ResidualRunnerContractError(
                f"{label} {key} is not exact 12-D ZERO"
            )
    if (
        transform.get("zero_identity") is not True
        or transform.get("action_names") != list(RESIDUAL_ACTION_NAMES)
        or transform.get("policy_id") != request.policy_id
        or transform.get("policy_sha256") != request.policy_sha256
    ):
        raise ResidualRunnerContractError(
            f"{label} does not bind the immutable ZERO policy/identity branch"
        )
    for prefix in ("servo", "wheel"):
        nominal = transform.get(f"nominal_{prefix}_targets_deg" if prefix == "servo" else f"nominal_{prefix}_targets_rad_s")
        applied = transform.get(f"applied_{prefix}_targets_deg" if prefix == "servo" else f"applied_{prefix}_targets_rad_s")
        if not isinstance(nominal, Mapping) or not _strict_json_equal(nominal, applied):
            raise ResidualRunnerContractError(
                f"{label} {prefix} applied targets differ from nominal"
            )
    _require_sha(transform.get("contract_sha256"), label=f"{label} contract")
    _require_sha(
        transform.get("core_transform_sha256"),
        label=f"{label} core transform",
    )
    evidence_sha = _require_sha(
        transform.get("evidence_sha256"), label=f"{label} evidence"
    )
    unhashed = dict(transform)
    unhashed.pop("evidence_sha256", None)
    if _canonical_json_sha256(unhashed) != evidence_sha:
        raise ResidualRunnerContractError(f"{label} evidence digest mismatch")
    return transform


def _validated_target_map(
    value: Any, *, names: Sequence[str], label: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ResidualRunnerContractError(f"{label} target keys mismatch")
    result: dict[str, float] = {}
    for name in names:
        item = value.get(name)
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ResidualRunnerContractError(f"{label} {name} is not finite")
        result[name] = float(item)
    return result


def _validate_zero_ledgers(
    *,
    request: WorkerResidualMacroFSMRequest,
    base_result: Mapping[str, Any],
    base_terminal: Mapping[str, Any],
) -> dict[str, Any]:
    run = request.run_dir.resolve()
    source_path = _path_in_run(
        _required_text(base_result, "source_action_consumption_path", label="result"),
        run,
        label="Gate-E source-action ledger",
    )
    dispatch_path = _path_in_run(
        _required_text(base_result, "dispatch_ledger_path", label="result"),
        run,
        label="Gate-E dispatch ledger",
    )
    completion_path = _path_in_run(
        _required_text(base_result, "segment_completion_ledger_path", label="result"),
        run,
        label="Gate-E completion ledger",
    )
    transition_path = _path_in_run(
        _required_text(base_result, "transition_evidence_path", label="result"),
        run,
        label="Gate-E transition ledger",
    )
    source_rows = _strict_jsonl(source_path, label="Gate-E source-action ledger")
    dispatch_rows = _strict_jsonl(dispatch_path, label="Gate-E dispatch ledger")
    completion_rows = _strict_jsonl(
        completion_path, label="Gate-E completion ledger"
    )
    transition_rows = _strict_jsonl(
        transition_path, label="Gate-E transition ledger"
    )
    expected_source_count = len(source_rows)
    if (
        expected_source_count <= 0
        or len(completion_rows) != expected_source_count
        or base_result.get("expected_source_action_count") != expected_source_count
        or base_result.get("source_action_consumption_count") != expected_source_count
        or base_result.get("expected_segment_completion_count") != expected_source_count
        or base_result.get("segment_completion_count") != expected_source_count
        or base_result.get("command_dispatch_count") != len(dispatch_rows)
    ):
        raise ResidualRunnerContractError(
            "Gate-E source/completion counts are not exact and complete"
        )
    if (
        base_result.get("source_action_coverage_complete") is not True
        or base_result.get("segment_completion_coverage_complete") is not True
        or list(base_result.get("segment_completion_coverage_errors", []) or [])
    ):
        raise ResidualRunnerContractError(
            "Gate-E source/completion coverage is incomplete"
        )

    dispatch_by_index: dict[int, dict[str, Any]] = {}
    used_steps: set[int] = set()
    used_batches: set[str] = set()
    last_dispatch_step: int | None = None
    for index, row in enumerate(dispatch_rows):
        if set(row) != _DISPATCH_ROW_KEYS:
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} schema keys are not exact"
            )
        if row.get("dispatch_index") != index:
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} index mismatch"
            )
        nominal_servos = _validated_target_map(
            row.get("nominal_servo_targets_deg"),
            names=SERVO_JOINT_NAMES,
            label=f"dispatch {index} nominal servos",
        )
        nominal_wheels = _validated_target_map(
            row.get("nominal_wheel_targets_rad_s"),
            names=WHEEL_JOINT_NAMES,
            label=f"dispatch {index} nominal wheels",
        )
        if (
            not _strict_json_equal(row.get("servo_targets_deg"), nominal_servos)
            or not _strict_json_equal(row.get("wheel_targets_rad_s"), nominal_wheels)
            or row.get("nominal_command_epoch") != row.get("command_epoch")
            or row.get("nominal_target_changed") is not True
            or row.get("applied_target_changed") is not True
            or row.get("dispatch_cause") != "NOMINAL_AND_RESIDUAL"
            or row.get("residual_policy_id") != request.policy_id
            or row.get("residual_policy_sha256") != request.policy_sha256
        ):
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} is not nominal ZERO identity"
            )
        physical_epoch = row.get("physical_command_epoch")
        if type(physical_epoch) is not int or physical_epoch != index + 1:
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} physical epoch is not contiguous"
            )
        sim_step = row.get("sim_step")
        batch_id = row.get("batch_id")
        if (
            type(sim_step) is not int
            or sim_step in used_steps
            or (last_dispatch_step is not None and sim_step <= last_dispatch_step)
            or type(batch_id) is not str
            or not batch_id
            or batch_id in used_batches
            or ":residual:" in batch_id
            or not batch_id.endswith(f":macro:{int(row['command_epoch']):06d}")
        ):
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} violates one-batch/no-residual-update"
            )
        used_steps.add(sim_step)
        used_batches.add(batch_id)
        last_dispatch_step = sim_step
        ack = row.get("ack")
        if not isinstance(ack, Mapping):
            raise ResidualRunnerContractError(f"Gate-E dispatch row {index} lacks ACK")
        first_step = ack.get("first_physics_step")
        if (
            ack.get("batch_id") != batch_id
            or ack.get("source") != "fsm50_macro_controller"
            or ack.get("applied_sim_step") != sim_step
            or type(first_step) is not int
            or first_step != sim_step + 1
            or row.get("n_plus_one_verified") is not True
            or row.get("n_plus_one_verified_sim_step") != first_step
            or not _is_lower_sha256(row.get("n_plus_one_readback_sha256"))
            or not _strict_json_equal(ack.get("servo_targets_applied"), nominal_servos)
            or not _strict_json_equal(ack.get("wheel_targets_applied"), nominal_wheels)
        ):
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} ACK/N+1 binding is invalid"
            )
        metadata = ack.get("recording_metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("physical_command_epoch") != physical_epoch
            or metadata.get("nominal_command_epoch") != row.get("command_epoch")
            or metadata.get("dispatch_cause") != "NOMINAL_AND_RESIDUAL"
            or not _strict_json_equal(
                metadata.get("nominal_servo_targets_deg"), nominal_servos
            )
            or not _strict_json_equal(
                metadata.get("nominal_wheel_targets_rad_s"), nominal_wheels
            )
        ):
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} metadata lost nominal/physical binding"
            )
        transform = _validate_zero_transform(
            row.get("residual_transform"), request=request, label=f"dispatch {index}"
        )
        if (
            not _strict_json_equal(
                transform.get("nominal_servo_targets_deg"), nominal_servos
            )
            or not _strict_json_equal(
                transform.get("nominal_wheel_targets_rad_s"), nominal_wheels
            )
            or transform.get("physical_command_epoch_before") != index
            or metadata.get("residual_evidence_sha256")
            != transform.get("evidence_sha256")
            or metadata.get("residual_transform_sha256")
            != transform.get("core_transform_sha256")
        ):
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} transform binding is invalid"
            )
        dispatch_by_index[index] = row

    linked_dispatch_indices: set[int] = set()
    last_source_step: int | None = None
    for index, row in enumerate(source_rows):
        if set(row) != _SOURCE_ROW_KEYS:
            raise ResidualRunnerContractError(
                f"Gate-E source row {index} schema keys are not exact"
            )
        nominal_changed = row.get("target_changed")
        source_step = row.get("sim_step")
        physical_epoch = row.get("physical_command_epoch")
        pre_action_epoch = row.get("pre_action_verified_command_epoch")
        if (
            row.get("source_action_index") != index
            or row.get("expected_source_action_count") != expected_source_count
            or type(nominal_changed) is not bool
            or type(source_step) is not int
            or (last_source_step is not None and source_step <= last_source_step)
            or type(physical_epoch) is not int
            or not 0 <= physical_epoch <= len(dispatch_rows)
            or type(pre_action_epoch) is not int
            or pre_action_epoch < 0
            or not _is_lower_sha256(
                row.get("pre_action_verified_readback_sha256")
            )
            or not _is_lower_sha256(row.get("residual_transform_sha256"))
            or row.get("nominal_target_changed") is not nominal_changed
            or row.get("applied_target_changed") is not nominal_changed
            or row.get("physical_dispatch_required") is not nominal_changed
            or row.get("physical_dispatch_applied") is not nominal_changed
            or not _strict_json_equal(
                row.get("applied_servo_targets_deg"), row.get("servo_targets_deg")
            )
            or not _strict_json_equal(
                row.get("applied_wheel_targets_rad_s"),
                row.get("wheel_targets_rad_s"),
            )
        ):
            raise ResidualRunnerContractError(
                f"Gate-E source row {index} nominal/applied identity is invalid"
            )
        last_source_step = source_step
        verified_dispatch_count = sum(
            1
            for dispatch_row in dispatch_rows
            if int(dispatch_row["n_plus_one_verified_sim_step"]) <= source_step
        )
        if nominal_changed:
            dispatch_index = row.get("physical_dispatch_index")
            dispatch = (
                dispatch_by_index.get(dispatch_index)
                if type(dispatch_index) is int
                else None
            )
            if (
                dispatch is None
                or dispatch.get("source_action_consumption_index") != index
                or physical_epoch != dispatch.get("physical_command_epoch")
                or dispatch.get("sim_step") != source_step
                or verified_dispatch_count != physical_epoch - 1
                or row.get("batch_id") != dispatch.get("batch_id")
                or row.get("n_plus_one_verified") is not True
                or row.get("n_plus_one_verified_sim_step")
                != dispatch.get("n_plus_one_verified_sim_step")
                or row.get("n_plus_one_readback_sha256")
                != dispatch.get("n_plus_one_readback_sha256")
                or row.get("residual_transform_sha256")
                != dispatch["residual_transform"].get("core_transform_sha256")
            ):
                raise ResidualRunnerContractError(
                    f"Gate-E source row {index} dispatch/N+1 link is invalid"
                )
            linked_dispatch_indices.add(dispatch_index)
        else:
            retained_dispatch = (
                None
                if physical_epoch == 0
                else dispatch_by_index.get(physical_epoch - 1)
            )
            if (
                row.get("physical_dispatch_index") is not None
                or row.get("batch_id") != ""
                or row.get("n_plus_one_verified") is not False
                or row.get("n_plus_one_verified_sim_step") is not None
                or row.get("n_plus_one_readback_sha256") != ""
                or pre_action_epoch != row.get("dispatch_epoch")
                or verified_dispatch_count != physical_epoch
                or (
                    retained_dispatch is not None
                    and (
                        retained_dispatch.get("physical_command_epoch")
                        != physical_epoch
                        or retained_dispatch.get("command_epoch")
                        != row.get("dispatch_epoch")
                        or int(
                            retained_dispatch["n_plus_one_verified_sim_step"]
                        )
                        > source_step
                        or not _strict_json_equal(
                            row.get("applied_servo_targets_deg"),
                            retained_dispatch.get("servo_targets_deg"),
                        )
                        or not _strict_json_equal(
                            row.get("applied_wheel_targets_rad_s"),
                            retained_dispatch.get("wheel_targets_rad_s"),
                        )
                    )
                )
            ):
                raise ResidualRunnerContractError(
                    f"Gate-E source row {index} retained-target evidence is invalid"
                )
    source_linked_dispatch_indices: set[int] = set()
    for index, dispatch in dispatch_by_index.items():
        source_index = dispatch.get("source_action_consumption_index")
        if source_index is None:
            continue
        if type(source_index) is not int:
            raise ResidualRunnerContractError(
                f"Gate-E dispatch row {index} source link is invalid"
            )
        source_linked_dispatch_indices.add(index)
    if source_linked_dispatch_indices != linked_dispatch_indices:
        raise ResidualRunnerContractError(
            "Gate-E source/dispatch link closure is invalid"
        )

    for index, row in enumerate(completion_rows):
        if set(row) != _COMPLETION_ROW_KEYS:
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} schema keys are not exact"
            )
        spec = row.get("completion_spec")
        effective = row.get("effective_completion_spec")
        effective_targets = row.get("effective_completion_servo_targets_deg")
        latched_residual = row.get("latched_servo_residual_deg")
        begin_evidence = row.get("tracking_begin_evidence")
        end_evidence = row.get("tracking_end_evidence")
        if (
            not isinstance(spec, Mapping)
            or not isinstance(spec.get("servo_targets_deg"), Mapping)
            or not isinstance(effective_targets, Mapping)
            or not isinstance(latched_residual, Mapping)
            or not isinstance(begin_evidence, Mapping)
            or not isinstance(end_evidence, Mapping)
        ):
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} effective target/lifecycle mismatch"
            )
        sparse_targets = dict(spec["servo_targets_deg"])
        sparse_names = set(sparse_targets)
        tracking_expected = int(bool(sparse_names))
        begin_evidence_valid = bool(
            set(begin_evidence) == {"called", "sparse_joint_names"}
            and begin_evidence.get("called") is bool(sparse_names)
            and begin_evidence.get("sparse_joint_names") == sorted(sparse_names)
        )
        if sparse_names:
            ended = end_evidence.get("ended")
            end_evidence_valid = bool(
                type(ended) is bool
                and end_evidence.get("tracking_completion_deferred")
                is (ended is False)
            )
        else:
            end_evidence_valid = _strict_json_equal(
                end_evidence,
                {
                    "ended": True,
                    "required": False,
                    "reason": "segment has no sparse servo endpoints",
                },
            )
        if (
            row.get("segment_completion_index") != index
            or not _strict_json_equal(spec, effective)
            or not _strict_json_equal(effective_targets, sparse_targets)
            or row.get("nominal_completion_spec_sha256")
            != _canonical_json_sha256(dict(spec))
            or row.get("effective_completion_targets_sha256")
            != _canonical_json_sha256(dict(effective_targets))
            or set(latched_residual) != sparse_names
            or any(
                type(value) not in (int, float) or float(value) != 0.0
                for value in latched_residual.values()
            )
            or not begin_evidence_valid
            or not end_evidence_valid
            or not _strict_json_equal(
                row.get("pre_action_applied_servo_targets_deg"),
                row.get("pre_action_canonical_servo_targets_deg"),
            )
            or row.get("nominal_target_changed")
            is not row.get("applied_target_changed")
            or row.get("terminal_kind") != "COMPLETE"
            or row.get("tracking_lifecycle_closed") is not True
            or row.get("tracking_begin_sim_step") != row.get("start_sim_step")
            or row.get("tracking_end_sim_step") != row.get("terminal_sim_step")
            or row.get("tracking_end_sim_time_s")
            != row.get("terminal_sim_time_s")
            or row.get("tracking_end_reason") != row.get("terminal_kind")
            or type(row.get("tracking_begin_count")) is not int
            or row.get("tracking_begin_count") != tracking_expected
            or type(row.get("tracking_end_count")) is not int
            or row.get("tracking_end_count") != tracking_expected
            or type(row.get("tracking_end_attempt_count")) is not int
            or row.get("tracking_end_attempt_count") != tracking_expected
        ):
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} effective target/lifecycle mismatch"
            )
        source_index = row.get("source_action_consumption_index")
        if type(source_index) is not int or not 0 <= source_index < len(source_rows):
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} source link is invalid"
            )
        source = source_rows[source_index]
        source_changed = source.get("target_changed") is True
        if (
            row.get("start_command_epoch") != source.get("dispatch_epoch")
            or row.get("start_physical_command_epoch")
            != source.get("physical_command_epoch")
            or row.get("start_physical_dispatch")
            is not source.get("target_changed")
            or row.get("retained_epoch_same_target")
            is source.get("target_changed")
        ):
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} start epoch/dispatch mismatch"
            )
        if source_changed:
            if (
                row.get("start_batch_id") != source.get("batch_id")
                or row.get("start_first_physics_step")
                != source.get("n_plus_one_verified_sim_step")
                or row.get("start_readback_verified") is not True
                or row.get("start_readback_verified_sim_step")
                != source.get("n_plus_one_verified_sim_step")
                or row.get("start_readback_sha256")
                != source.get("n_plus_one_readback_sha256")
            ):
                raise ResidualRunnerContractError(
                    f"Gate-E completion row {index} dispatched readback binding is invalid"
                )
        elif (
            row.get("start_batch_id") != ""
            or row.get("start_first_physics_step") is not None
            or row.get("start_readback_verified") is not True
            or row.get("start_readback_verified_sim_step")
            != source.get("sim_step")
            or row.get("start_readback_sha256")
            != source.get("pre_action_verified_readback_sha256")
        ):
            raise ResidualRunnerContractError(
                f"Gate-E completion row {index} retained readback binding is invalid"
            )

    for index, row in enumerate(transition_rows):
        if set(row) != _TRANSITION_ROW_KEYS or row.get("transition_index") != index:
            raise ResidualRunnerContractError(
                f"Gate-E transition row {index} schema/index mismatch"
            )
        if index and row.get("from_state") != transition_rows[index - 1].get("to_state"):
            raise ResidualRunnerContractError(
                f"Gate-E transition row {index} breaks the state chain"
            )

    safe_ack = base_terminal.get("safe_stop_ack")
    safe_readback = base_terminal.get("safe_stop_readback")
    safe_readback_sha256 = base_terminal.get("safe_stop_readback_sha256")
    last_readback = base_terminal.get("last_target_readback")
    if (
        not isinstance(safe_ack, Mapping)
        or not isinstance(safe_readback, Mapping)
        or not isinstance(last_readback, Mapping)
    ):
        raise ResidualRunnerContractError(
            "Gate-E terminal lacks safe-stop ACK/readback"
        )
    if (
        set(safe_readback) != _TARGET_READBACK_KEYS
        or set(last_readback) != _TARGET_READBACK_KEYS
    ):
        raise ResidualRunnerContractError(
            "Gate-E safe-stop/final readback schema keys are not exact"
        )
    safe_servos = _validated_target_map(
        safe_ack.get("servo_targets_applied"),
        names=SERVO_JOINT_NAMES,
        label="safe-stop ACK servos",
    )
    safe_wheels = _validated_target_map(
        safe_ack.get("wheel_targets_applied"),
        names=WHEEL_JOINT_NAMES,
        label="safe-stop ACK wheels",
    )
    readback_servos = _validated_target_map(
        safe_readback.get("canonical_servo_targets_deg"),
        names=SERVO_JOINT_NAMES,
        label="safe-stop readback canonical servos",
    )
    readback_wheels = _validated_target_map(
        safe_readback.get("canonical_wheel_targets_rad_s"),
        names=WHEEL_JOINT_NAMES,
        label="safe-stop readback canonical wheels",
    )
    _validated_target_map(
        safe_readback.get("actual_servo_drive_targets_rad"),
        names=SERVO_JOINT_NAMES,
        label="safe-stop readback actual servos",
    )
    actual_readback_wheels = _validated_target_map(
        safe_readback.get("actual_wheel_drive_targets_rad_s"),
        names=WHEEL_JOINT_NAMES,
        label="safe-stop readback actual wheels",
    )
    final_servos = _validated_target_map(
        last_readback.get("canonical_servo_targets_deg"),
        names=SERVO_JOINT_NAMES,
        label="final readback canonical servos",
    )
    final_wheels = _validated_target_map(
        last_readback.get("canonical_wheel_targets_rad_s"),
        names=WHEEL_JOINT_NAMES,
        label="final readback canonical wheels",
    )
    _validated_target_map(
        last_readback.get("actual_servo_drive_targets_rad"),
        names=SERVO_JOINT_NAMES,
        label="final readback actual servos",
    )
    actual_final_wheels = _validated_target_map(
        last_readback.get("actual_wheel_drive_targets_rad_s"),
        names=WHEEL_JOINT_NAMES,
        label="final readback actual wheels",
    )
    safe_applied_step = safe_ack.get("applied_sim_step")
    safe_first_step = safe_ack.get("first_physics_step")
    zero_wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
    safe_batch_id = safe_ack.get("batch_id")
    safe_metadata = safe_ack.get("recording_metadata")
    if (
        safe_ack.get("source") != "fsm50_macro_safe_stop"
        or type(safe_batch_id) is not str
        or not safe_batch_id
        or safe_batch_id in used_batches
        or safe_wheels != zero_wheels
        or type(safe_applied_step) is not int
        or safe_applied_step in used_steps
        or type(safe_first_step) is not int
        or safe_first_step != safe_applied_step + 1
        or safe_readback.get("batch_id") != safe_batch_id
        or safe_readback.get("sim_step") != safe_first_step
        or not isinstance(safe_metadata, Mapping)
        or safe_readback.get("command_epoch")
        != safe_metadata.get("command_epoch")
        or not _strict_json_equal(readback_servos, safe_servos)
        or readback_wheels != zero_wheels
        or actual_readback_wheels != zero_wheels
        or safe_readback.get("adapter_runtime_instance_id")
        != base_terminal.get("adapter_runtime_instance_id")
        or safe_readback.get("root_state_write_count") != 0
        or safe_readback.get("physics_dt_s") != base_terminal.get("physics_dt_s")
        or safe_readback_sha256
        != _canonical_json_sha256(dict(safe_readback))
        or not _strict_json_equal(base_result.get("safe_stop_ack"), safe_ack)
        or not _strict_json_equal(
            base_result.get("safe_stop_readback"), safe_readback
        )
        or base_result.get("safe_stop_readback_sha256")
        != safe_readback_sha256
        or not _strict_json_equal(
            base_result.get("last_target_readback"), last_readback
        )
        or type(last_readback.get("sim_step")) is not int
        or int(last_readback["sim_step"]) < safe_first_step
        or not _strict_json_equal(final_servos, safe_servos)
        or final_wheels != zero_wheels
        or actual_final_wheels != zero_wheels
        or last_readback.get("adapter_runtime_instance_id")
        != base_terminal.get("adapter_runtime_instance_id")
        or last_readback.get("root_state_write_count") != 0
        or last_readback.get("physics_dt_s") != base_terminal.get("physics_dt_s")
        or base_terminal.get("safe_stop_status") != "VERIFIED"
        or base_terminal.get("safe_stop_verified") is not True
        or str(base_terminal.get("safe_stop_error", "") or "")
        or base_result.get("safe_stop_status") != "VERIFIED"
        or base_result.get("safe_stop_verified") is not True
        or str(base_result.get("safe_stop_error", "") or "")
    ):
        raise ResidualRunnerContractError(
            "Gate-E safe-stop one-batch/N+1 evidence is invalid"
        )

    direct = base_result.get("direct_command_residual")
    if not isinstance(direct, Mapping):
        raise ResidualRunnerContractError(
            "Gate-E base result lacks direct residual evidence"
        )
    transform_count = direct.get("transform_count")
    if (
        direct.get("enabled") is not True
        or direct.get("policy_id") != request.policy_id
        or direct.get("policy_sha256") != request.policy_sha256
        or type(transform_count) is not int
        or transform_count < len(dispatch_rows)
        or direct.get("physical_command_epoch") != len(dispatch_rows) + 1
        or direct.get("last_verified_physical_command_epoch")
        != len(dispatch_rows) + 1
    ):
        raise ResidualRunnerContractError(
            "Gate-E physical epoch/transform count closure is invalid"
        )
    last_transform = _validate_zero_transform(
        direct.get("last_transform"), request=request, label="last residual transform"
    )
    if any(float(value) != 0.0 for value in last_transform["applied_residual"]):
        raise ResidualRunnerContractError("Gate-E last residual is not ZERO")

    return {
        "schema_version": ZERO_LEDGER_VALIDATION_SCHEMA,
        "source_action_count": len(source_rows),
        "segment_completion_count": len(completion_rows),
        "nominal_dispatch_count": len(dispatch_rows),
        "transition_count": len(transition_rows),
        "transform_count": transform_count,
        "controller_physical_command_epoch": len(dispatch_rows),
        "safe_stop_physical_command_epoch": len(dispatch_rows) + 1,
        "all_actions_exact_zero": True,
        "all_applied_maps_bit_exact_nominal": True,
        "all_effective_completion_targets_bit_exact_nominal": True,
        "residual_only_dispatch_count": 0,
        "additional_dispatch_count": 0,
        "one_batch_per_sim_step": True,
        "all_dispatch_n_plus_one_verified": True,
        "safe_stop_n_plus_one_verified": True,
        "source_action_ledger_path": str(source_path),
        "source_action_ledger_sha256": sha256_file(source_path),
        "dispatch_ledger_path": str(dispatch_path),
        "dispatch_ledger_sha256": sha256_file(dispatch_path),
        "segment_completion_ledger_path": str(completion_path),
        "segment_completion_ledger_sha256": sha256_file(completion_path),
        "transition_ledger_path": str(transition_path),
        "transition_ledger_sha256": sha256_file(transition_path),
    }


def _raise_failed_terminal_diagnostic(terminal: Mapping[str, Any]) -> None:
    """Reject a failed Gate-E terminal without obscuring its base failure.

    The live transport intentionally delivers both successful and failed Macro
    terminals.  Success validation remains strict below, but a genuine worker
    failure must not be collapsed into a generic success-schema mismatch.  Keep
    the raw protocol identities and failure reasons in one deterministic JSON
    object so the runner manifest remains useful for post-mortem review.
    """

    outer = dict(terminal)
    raw_base = outer.get("base_macro_fsm_terminal")
    base = dict(raw_base) if isinstance(raw_base, Mapping) else {}
    if (
        outer.get("type") != "macro_fsm_failed"
        and base.get("type") != "macro_fsm_failed"
    ):
        return

    fields = (
        "type",
        "operation",
        "request_id",
        "residual_macro_fsm_request_id",
        "base_macro_fsm_request_id",
        "accepted",
        "macro_fsm_complete",
        "error",
        "controller_terminal_outcome",
        "controller_terminal_reason",
    )
    detail = {
        "outer": {name: outer.get(name) for name in fields},
        "base": {name: base.get(name) for name in fields},
    }
    raise ResidualRunnerContractError(
        "Gate-E worker reported macro_fsm_failed: "
        + json.dumps(
            detail,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _validate_outer_terminal(
    terminal: Mapping[str, Any],
    *,
    request: WorkerResidualMacroFSMRequest,
    worker_binding: Mapping[str, Any],
    reference_projection: Mapping[str, Any],
) -> dict[str, Any]:
    outer = copy.deepcopy(dict(terminal))
    _raise_failed_terminal_diagnostic(outer)
    run = request.run_dir.resolve()
    for key, expected in {
        "type": "macro_fsm_complete",
        "operation": OPERATION,
        "phase": "MACRO_FSM_COMPLETE",
        "accepted": True,
        "macro_fsm_complete": True,
        "execution_mode": EXECUTION_MODE,
        "request_id": request.request_id,
        "request_identity_sha256": request.request_identity_sha256,
        "residual_macro_fsm_request_id": request.request_id,
        "base_macro_fsm_request_id": request.base_request.request_id,
        "base_macro_fsm_request_path": str(request.base_request_path),
        "base_macro_fsm_request_sha256": request.base_request_sha256,
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
        "run_dir": str(run),
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "root_state_write_count": 0,
        "video_writer_quiesced": True,
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
    }.items():
        if outer.get(key) != expected:
            raise ResidualRunnerContractError(f"Gate-E terminal {key} mismatch")
    if str(outer.get("error", "") or ""):
        raise ResidualRunnerContractError("Gate-E terminal reports an error")
    if not _strict_json_equal(
        outer.get("gate_e_zero_residual"),
        request.gate_e_identity(payload_role="terminal"),
    ):
        raise ResidualRunnerContractError("Gate-E terminal identity mismatch")

    base_terminal = outer.get("base_macro_fsm_terminal")
    if not isinstance(base_terminal, Mapping):
        raise ResidualRunnerContractError("Gate-E embedded base terminal is missing")
    base_terminal = copy.deepcopy(dict(base_terminal))
    if _canonical_json_sha256(base_terminal) != outer.get(
        "base_macro_fsm_terminal_sha256"
    ):
        raise ResidualRunnerContractError("Gate-E embedded base terminal SHA mismatch")
    for key, expected in {
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
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "root_state_write_count": 0,
        "video_writer_quiesced": True,
        "safe_stop_status": "VERIFIED",
        "safe_stop_verified": True,
    }.items():
        if base_terminal.get(key) != expected:
            raise ResidualRunnerContractError(
                f"Gate-E embedded base terminal {key} mismatch"
            )
    if str(base_terminal.get("error", "") or ""):
        raise ResidualRunnerContractError("Gate-E embedded base terminal is in error")
    for key in (
        "safe_stop_ack",
        "safe_stop_readback",
        "safe_stop_readback_sha256",
        "last_target_readback",
    ):
        if not _strict_json_equal(outer.get(key), base_terminal.get(key)):
            raise ResidualRunnerContractError(
                "Gate-E outer safe-stop evidence differs from embedded base "
                f"terminal at {key}"
            )

    base_inputs_path = _path_in_run(
        _required_text(outer, "base_task_inputs_path", label="outer terminal"),
        run,
        label="base task inputs",
    )
    base_result_path = _path_in_run(
        _required_text(outer, "base_worker_result_path", label="outer terminal"),
        run,
        label="base worker result",
    )
    outer_inputs_path = _path_in_run(
        _required_text(outer, "task_inputs_path", label="outer terminal"),
        run,
        label="outer task inputs",
    )
    outer_result_path = _path_in_run(
        _required_text(outer, "worker_result_path", label="outer terminal"),
        run,
        label="outer worker result",
    )
    video_path = _path_in_run(
        _required_text(base_terminal, "video_path", label="base terminal"),
        run,
        label="Gate-E viewport video",
    )
    if outer.get("video_path") != str(video_path) or video_path.stat().st_size <= 0:
        raise ResidualRunnerContractError(
            "Gate-E viewport video path/bytes are not durably bound"
        )
    if (
        base_inputs_path != (run / "macro_task_inputs.json").resolve()
        or base_result_path != (run / "worker_macro_fsm_result.json").resolve()
        or outer_inputs_path != (run / GATE_E_TASK_INPUTS_NAME).resolve()
        or outer_result_path != (run / GATE_E_WORKER_RESULT_NAME).resolve()
    ):
        raise ResidualRunnerContractError("Gate-E terminal artifact names are not canonical")
    for path, claimed, label in (
        (base_inputs_path, outer.get("base_task_inputs_sha256"), "base task inputs"),
        (base_result_path, outer.get("base_worker_result_sha256"), "base worker result"),
        (outer_inputs_path, outer.get("task_inputs_sha256"), "outer task inputs"),
        (outer_result_path, outer.get("worker_result_sha256"), "outer worker result"),
    ):
        if sha256_file(path) != claimed:
            raise ResidualRunnerContractError(f"Gate-E {label} SHA mismatch")
    base_inputs = _strict_json_load(base_inputs_path, label="base task inputs")
    base_result = _strict_json_load(base_result_path, label="base worker result")
    outer_inputs = _strict_json_load(outer_inputs_path, label="outer task inputs")
    outer_result = _strict_json_load(outer_result_path, label="outer worker result")
    if set(outer_inputs) != _OUTER_TASK_INPUT_KEYS or set(outer_result) != _OUTER_RESULT_KEYS:
        raise ResidualRunnerContractError("Gate-E outer artifact schema keys are not exact")
    if (
        outer_inputs.get("schema_version") != GATE_E_TASK_INPUTS_SCHEMA
        or outer_result.get("schema_version") != GATE_E_WORKER_RESULT_SCHEMA
        or outer_inputs.get("execution_mode") != EXECUTION_MODE
        or outer_result.get("execution_mode") != EXECUTION_MODE
        or outer_inputs.get("operation") != OPERATION
        or outer_result.get("operation") != OPERATION
        or outer_result.get("macro_fsm_complete") is not True
        or str(outer_result.get("error", "") or "")
    ):
        raise ResidualRunnerContractError("Gate-E outer result/task inputs are invalid")
    for payload, role in (
        (outer_inputs, "task_inputs"),
        (outer_result, "worker_result"),
    ):
        if not _strict_json_equal(
            payload.get("gate_e_zero_residual"),
            request.gate_e_identity(payload_role=role),
        ):
            raise ResidualRunnerContractError(f"Gate-E {role} identity mismatch")
    if (
        outer_inputs.get("base_task_inputs_path") != str(base_inputs_path)
        or outer_inputs.get("base_task_inputs_sha256") != sha256_file(base_inputs_path)
        or not _strict_json_equal(outer_inputs.get("base_task_inputs"), base_inputs)
        or outer_result.get("base_worker_result_path") != str(base_result_path)
        or outer_result.get("base_worker_result_sha256") != sha256_file(base_result_path)
        or not _strict_json_equal(outer_result.get("base_worker_result"), base_result)
        or outer_result.get("task_inputs_path") != str(outer_inputs_path)
        or outer_result.get("task_inputs_sha256") != sha256_file(outer_inputs_path)
        or not _strict_json_equal(outer_result.get("task_inputs"), outer_inputs)
        or not _strict_json_equal(outer.get("task_inputs"), outer_inputs)
    ):
        raise ResidualRunnerContractError("Gate-E outer/embedded artifact binding mismatch")
    if (
        base_result.get("schema_version") != "fsm50.worker_macro_fsm_session.v1"
        or base_result.get("request_id") != request.base_request.request_id
        or base_result.get("macro_fsm_complete") is not True
        or base_result.get("source_version") != request.source_version
        or base_result.get("bundle_sha256") != request.bundle_sha256
        or base_result.get("task_inputs_path") != str(base_inputs_path)
        or str(base_result.get("error", "") or "")
        or base_inputs.get("schema_version") != "fsm50.macro_task_inputs.v1"
        or not _strict_json_equal(base_terminal.get("task_inputs"), base_inputs)
    ):
        raise ResidualRunnerContractError("Gate-E embedded base artifacts are invalid")

    evidence = outer.get("residual_command_evidence")
    if not isinstance(evidence, Mapping):
        raise ResidualRunnerContractError("Gate-E residual command evidence is missing")
    if (
        not _strict_json_equal(evidence, outer_inputs.get("residual_command_evidence"))
        or not _strict_json_equal(evidence, outer_result.get("residual_command_evidence"))
        or evidence.get("schema_version") != GATE_E_COMMAND_EVIDENCE_SCHEMA
        or evidence.get("policy_kind") != POLICY_KIND
        or evidence.get("policy_id") != request.policy_id
        or evidence.get("policy_sha256") != request.policy_sha256
        or evidence.get("residual_core_sha256") != request.residual_core_sha256
        or evidence.get("envelope_canonical_sha256")
        != request.envelope_canonical_sha256
        or evidence.get("all_durable_residual_dispatches_zero_identity") is not True
        or evidence.get("checkpoint_loaded") is not False
        or evidence.get("ppo_training_performed") is not False
    ):
        raise ResidualRunnerContractError("Gate-E residual command evidence mismatch")

    ledger = _validate_zero_ledgers(
        request=request, base_result=base_result, base_terminal=base_terminal
    )
    if (
        evidence.get("dispatch_ledger_path") != ledger["dispatch_ledger_path"]
        or evidence.get("dispatch_ledger_sha256") != ledger["dispatch_ledger_sha256"]
        or evidence.get("dispatch_row_count") != ledger["nominal_dispatch_count"]
        or evidence.get("residual_dispatch_zero_identity_count")
        != ledger["nominal_dispatch_count"]
        or evidence.get("transform_count") != ledger["transform_count"]
        or evidence.get("physical_command_epoch")
        != ledger["safe_stop_physical_command_epoch"]
        or evidence.get("last_verified_physical_command_epoch")
        != ledger["safe_stop_physical_command_epoch"]
    ):
        raise ResidualRunnerContractError(
            "Gate-E wrapper command evidence differs from ledger closure"
        )
    _validate_zero_transform(
        evidence.get("last_transform"),
        request=request,
        label="outer last residual transform",
    )

    projection = _semantic_projection(run)
    if not _strict_json_equal(projection, dict(reference_projection)):
        raise ResidualRunnerContractError(
            "Gate-E ZERO run semantic projection differs from reviewed r4 reference"
        )
    return {
        "outer_terminal": outer,
        "base_terminal": base_terminal,
        "outer_task_inputs_path": str(outer_inputs_path),
        "outer_task_inputs_sha256": sha256_file(outer_inputs_path),
        "outer_worker_result_path": str(outer_result_path),
        "outer_worker_result_sha256": sha256_file(outer_result_path),
        "base_task_inputs_path": str(base_inputs_path),
        "base_task_inputs_sha256": sha256_file(base_inputs_path),
        "base_worker_result_path": str(base_result_path),
        "base_worker_result_sha256": sha256_file(base_result_path),
        "video_path": str(video_path),
        "video_sha256": sha256_file(video_path),
        "zero_ledger_validation": ledger,
        "semantic_projection": projection,
        "semantic_projection_sha256": _canonical_json_sha256(projection),
        "semantic_projection_equal": True,
    }


def _validate_terminal_ack(
    acknowledgement: Mapping[str, Any],
    *,
    terminal: Mapping[str, Any],
    request: WorkerResidualMacroFSMRequest,
) -> dict[str, Any]:
    ack = dict(acknowledgement)
    if set(ack) != set(terminal):
        raise ResidualRunnerContractError(
            "Gate-E terminal ACK keys differ from raw terminal"
        )
    if ack.get("type") != "operation_ack":
        raise ResidualRunnerContractError("Gate-E terminal ACK type mismatch")
    for key, expected in terminal.items():
        if key not in {"type", "operation"} and not _strict_json_equal(
            ack.get(key), expected
        ):
            raise ResidualRunnerContractError(
                f"Gate-E terminal ACK {key} differs from raw terminal"
            )
    if (
        ack.get("operation") != "macro_fsm"
        or ack.get("request_id") != request.request_id
        or not _strict_json_equal(
            ack.get("gate_e_zero_residual"),
            request.gate_e_identity(payload_role="terminal"),
        )
    ):
        raise ResidualRunnerContractError("Gate-E terminal ACK identity mismatch")
    return ack


def _validate_residual_fast_shutdown(
    outcome: Mapping[str, Any],
    *,
    shutdown_request_id: str,
    request: WorkerResidualMacroFSMRequest,
    worker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = macro_runner.validate_fast_shutdown(
        outcome,
        shutdown_request_id=shutdown_request_id,
        macro_request_id=request.request_id,
        worker_binding=worker_binding,
    )
    for name, role in (
        ("shutdown_ack", "shutdown_ack"),
        ("close_requested_ack", "close_requested"),
        ("close_requested_receipt", "close_requested"),
        ("close_returned_ack", "close_returned"),
        ("close_returned_receipt", "close_returned"),
    ):
        row = result.get(name)
        if row in (None, {}):
            if name.startswith("close_returned"):
                continue
            raise ResidualRunnerContractError(f"Gate-E fast close lacks {name}")
        if (
            not isinstance(row, Mapping)
            or row.get("residual_macro_fsm_request_id") != request.request_id
            or row.get("base_macro_fsm_request_id")
            != request.base_request.request_id
            or not _strict_json_equal(
                row.get("gate_e_zero_residual"),
                request.gate_e_identity(payload_role=role),
            )
        ):
            raise ResidualRunnerContractError(
                f"Gate-E fast-close identity mismatch: {name}"
            )
    return dict(result)


def _allocate_run_paths(output_root: str | Path) -> ResidualRunPaths:
    run = Path(output_root).resolve()
    if run.exists():
        raise ResidualRunnerContractError(
            "Gate-E output root must be one fresh, nonexistent directory"
        )
    run.mkdir(parents=True, exist_ok=False)
    return ResidualRunPaths(
        run_dir=run,
        base_request_path=run / BASE_REQUEST_NAME,
        residual_request_path=run / RESIDUAL_REQUEST_NAME,
        manifest_path=run / RUNNER_MANIFEST_NAME,
    )


def _build_base_request(
    *,
    args: argparse.Namespace,
    paths: ResidualRunPaths,
    bundle: MacroControllerBundle,
    base_request_id: str,
    reference_trial_kind: str,
) -> dict[str, Any]:
    macro_paths = macro_runner.MacroRunPaths(
        run_dir=paths.run_dir,
        request_path=paths.base_request_path,
        manifest_path=paths.run_dir / "unused_base_runner_manifest.json",
    )
    return macro_runner.build_worker_macro_request(
        bundle,
        macro_paths,
        request_id=base_request_id,
        source_version=args.source_version,
        alignment_path=args.alignment_path,
        task_success_table_path=args.task_success_table_path,
        trial_kind=reference_trial_kind,
        trial_index=0,
        telemetry_hz=args.telemetry_hz,
        post_run_settle_s=args.post_run_settle_s,
        timeout_s=args.task_timeout_s,
    )


def _validate_request_admission(
    request: WorkerResidualMacroFSMRequest,
    *,
    paths: ResidualRunPaths,
    source_version: str,
    bundle: MacroControllerBundle,
    code_identity: Mapping[str, Any],
    environment: Mapping[str, Any],
    stale_identities: Mapping[str, set[Any]],
) -> None:
    if (
        request.request_path != paths.residual_request_path.resolve()
        or request.base_request_path != paths.base_request_path.resolve()
        or request.run_dir != paths.run_dir.resolve()
        or request.source_version != source_version
        or request.bundle_sha256 != bundle.bundle_sha256
        or request.graph_sha256 != bundle.graph_sha256
        or request.profile_library_sha256 != bundle.profile_library_sha256
        or request.environment_lock_sha256
        != environment.get("environment_lock_sha256")
        or request.request_id in stale_identities.get("request_id", set())
        or request.base_request.request_id in stale_identities.get("request_id", set())
        or request.request_id == request.base_request.request_id
    ):
        raise ResidualRunnerContractError(
            "Gate-E wrapper/base request admission identity mismatch"
        )
    current_code = code_identity.get("code_sha256")
    if not isinstance(current_code, Mapping):
        raise ResidualRunnerContractError("Gate-E code identity is malformed")
    for name, digest in request.code_sha256.items():
        if current_code.get(name) != digest:
            raise ResidualRunnerContractError(
                f"Gate-E request code binding differs from preflight: {name}"
            )
    for key, expected in {
        "residual_core_sha256": request.residual_core_sha256,
        "envelope_config_file_sha256": request.envelope_config_file_sha256,
        "envelope_canonical_sha256": request.envelope_canonical_sha256,
    }.items():
        if code_identity.get(key) != expected:
            raise ResidualRunnerContractError(
                f"Gate-E request/preflight {key} mismatch"
            )
    policy = ZeroResidualPolicy()
    if (
        request.policy_kind != POLICY_KIND
        or request.policy_id != policy.policy_id
        or request.policy_sha256 != policy.policy_sha256
    ):
        raise ResidualRunnerContractError("Gate-E request is not immutable ZERO-R0")


def _preflight_admission(
    args: argparse.Namespace,
    *,
    bundle_builder: Callable[..., MacroControllerBundle],
    environment_validator: Callable[[], Mapping[str, Any]],
    closure_validator: Callable[..., Mapping[str, Any]],
    reference_validator: Callable[..., Mapping[str, Any]],
    code_identity_builder: Callable[[], Mapping[str, Any]],
    require_fresh_output: bool = True,
) -> dict[str, Any]:
    if args.source_version not in ALLOWED_SOURCE_VERSIONS:
        raise ResidualRunnerContractError("Gate-E source is not v003/v008/v009")
    output = Path(args.output_root).resolve()
    if any(_is_within(output, protected) for protected in EXACT_REVIEWED_REFERENCE_RUNS.values()):
        raise ResidualRunnerContractError(
            "Gate-E output root cannot be inside a reviewed reference run"
        )
    if require_fresh_output and output.exists():
        raise ResidualRunnerContractError(
            "Gate-E output root must not exist before preflight"
        )
    environment = dict(environment_validator())
    if not environment or environment.get("source_closure_complete") is not True:
        raise ResidualRunnerContractError(
            "canonical environment-lock preflight returned no exact closure"
        )
    closure = dict(closure_validator(args, bundle_builder=bundle_builder))
    bundle = bundle_builder(
        PROJECT_ROOT,
        alignment_path=args.alignment_path,
        primary_source_version=args.source_version,
    )
    if bundle.primary_source_version != args.source_version:
        raise ResidualRunnerContractError(
            "current Gate-E source bundle primary source mismatch"
        )
    reference = dict(
        reference_validator(
            source_version=args.source_version,
            run_dir=args.reviewed_reference_run_dir,
            bundle=bundle,
            gate_c_closure=closure,
            alignment_path=args.alignment_path,
            task_success_table_path=args.task_success_table_path,
        )
    )
    if not isinstance(reference.get("identity"), Mapping) or not isinstance(
        reference.get("projection"), Mapping
    ):
        raise ResidualRunnerContractError(
            "reviewed reference validator returned malformed evidence"
        )
    code_identity = dict(code_identity_builder())
    if not code_identity or not _is_lower_sha256(
        code_identity.get("identity_sha256")
    ):
        raise ResidualRunnerContractError(
            "Gate-E live code/config identity is incomplete"
        )
    source_binding = {
        "bundle": _jsonable(bundle.to_mapping()),
        "bundle_sha256": bundle.bundle_sha256,
        "graph_sha256": bundle.graph_sha256,
        "profile_library_sha256": bundle.profile_library_sha256,
        "alignment_path": str(Path(args.alignment_path).resolve()),
        "alignment_sha256": sha256_file(args.alignment_path),
        "task_success_table_path": str(Path(args.task_success_table_path).resolve()),
        "task_success_table_sha256": sha256_file(args.task_success_table_path),
        "reviewed_reference_artifacts": dict(reference["identity"]),
    }
    source_binding["identity_sha256"] = _canonical_json_sha256(source_binding)
    return {
        "environment": environment,
        "gate_c_closure": closure,
        "bundle": bundle,
        "reviewed_reference": reference,
        "code_identity": code_identity,
        "source_binding": source_binding,
    }


def _admission_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in (
        "environment",
        "gate_c_closure",
        "reviewed_reference",
        "code_identity",
        "source_binding",
    ):
        if not _strict_json_equal(left.get(key), right.get(key)):
            return False
    left_bundle = left.get("bundle")
    right_bundle = right.get("bundle")
    return bool(
        isinstance(left_bundle, MacroControllerBundle)
        and isinstance(right_bundle, MacroControllerBundle)
        and _strict_json_equal(left_bundle.to_mapping(), right_bundle.to_mapping())
    )


def run_one_zero_r0(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
    environment_lock_validator: Callable[[], Mapping[str, Any]] = (
        macro_runner._validate_canonical_environment_lock
    ),
    closure_validator: Callable[..., Mapping[str, Any]] = _validate_gate_c_admission,
    reference_validator: Callable[..., Mapping[str, Any]] = (
        _validate_reviewed_reference
    ),
    code_identity_builder: Callable[[], Mapping[str, Any]] = (
        _current_gate_e_code_identity
    ),
    residual_request_builder: Callable[..., Mapping[str, Any]] = (
        build_worker_residual_macro_fsm_request
    ),
    residual_request_loader: Callable[..., WorkerResidualMacroFSMRequest | None] = (
        load_worker_residual_macro_fsm_request
    ),
    lock_factory: Callable[[], Any] | None = None,
    process_snapshot_fn: Callable[[], list[dict[str, Any]]] | None = None,
    conflict_detector: Callable[
        [Iterable[dict[str, Any]]], list[dict[str, Any]]
    ]
    | None = None,
) -> ResidualAttempt:
    preflight = _preflight_admission(
        args,
        bundle_builder=bundle_builder,
        environment_validator=environment_lock_validator,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        code_identity_builder=code_identity_builder,
    )
    paths = _allocate_run_paths(args.output_root)
    base_request_id = uuid.uuid4().hex
    outer_request_id = uuid.uuid4().hex
    reference_identity = dict(preflight["reviewed_reference"]["identity"])
    base_request = _build_base_request(
        args=args,
        paths=paths,
        bundle=preflight["bundle"],
        base_request_id=base_request_id,
        reference_trial_kind=str(reference_identity["trial_kind"]),
    )
    _atomic_write_json(paths.base_request_path, base_request)
    residual_payload = dict(
        residual_request_builder(
            base_request_path=paths.base_request_path,
            request_id=outer_request_id,
            environment_validator=environment_lock_validator,
        )
    )
    _atomic_write_json(paths.residual_request_path, residual_payload)
    loaded_request = residual_request_loader(
        paths.residual_request_path,
        environment_validator=environment_lock_validator,
    )
    if loaded_request is None:
        raise ResidualRunnerContractError("Gate-E residual request did not load")
    stale_identities = _fresh_identity_values(
        reference_identity, preflight["gate_c_closure"]
    )
    _validate_request_admission(
        loaded_request,
        paths=paths,
        source_version=args.source_version,
        bundle=preflight["bundle"],
        code_identity=preflight["code_identity"],
        environment=preflight["environment"],
        stale_identities=stale_identities,
    )

    if lock_factory is None or process_snapshot_fn is None or conflict_detector is None:
        default_lock, default_snapshot, default_conflicts = (
            macro_runner._process_guard_dependencies()
        )
        lock_factory = lock_factory or default_lock
        process_snapshot_fn = process_snapshot_fn or default_snapshot
        conflict_detector = conflict_detector or default_conflicts

    worker_args = _build_production_worker_args(args, paths.residual_request_path)
    factory = client_factory or _default_client_factory
    worker_binding: dict[str, Any] = {}
    start_ack: dict[str, Any] = {}
    terminal: dict[str, Any] = {}
    terminal_validation: dict[str, Any] = {}
    terminal_ack: dict[str, Any] = {}
    shutdown_outcome: dict[str, Any] = {}
    shutdown_verified = False
    under_lock_admission: dict[str, Any] = {}
    error = ""
    client: Any | None = None
    lock = lock_factory()
    try:
        lock.acquire()
        macro_runner._assert_no_simulator_process(
            process_snapshot_fn, conflict_detector
        )
        under_lock_admission = _preflight_admission(
            args,
            bundle_builder=bundle_builder,
            environment_validator=environment_lock_validator,
            closure_validator=closure_validator,
            reference_validator=reference_validator,
            code_identity_builder=code_identity_builder,
            require_fresh_output=False,
        )
        if not _admission_equal(preflight, under_lock_admission):
            raise ResidualRunnerContractError(
                "Gate-E admission changed before singleton launch"
            )
        if (
            sha256_file(paths.base_request_path)
            != loaded_request.base_request_sha256
            or sha256_file(paths.residual_request_path)
            != loaded_request.request_file_sha256
        ):
            raise ResidualRunnerContractError(
                "Gate-E request artifacts changed before singleton launch"
            )
        under_lock_request = residual_request_loader(
            paths.residual_request_path,
            environment_validator=environment_lock_validator,
        )
        if under_lock_request is None or not _strict_json_equal(
            under_lock_request.gate_e_identity(payload_role="status"),
            loaded_request.gate_e_identity(payload_role="status"),
        ):
            raise ResidualRunnerContractError(
                "Gate-E request identity changed under singleton"
            )
        loaded_request = under_lock_request
        client = factory(worker_args)
        client.start()
        ready = macro_runner.wait_for_worker_ready(
            client, timeout_s=args.sim_startup_timeout_s
        )
        worker_binding = _validate_worker_binding(
            ready,
            request=loaded_request,
            stale_identities=stale_identities,
        )
        process = getattr(client, "process", None)
        process_pid = getattr(process, "pid", None)
        if process_pid is not None and process_pid != worker_binding["worker_pid"]:
            raise ResidualRunnerContractError(
                "SimProcessClient owned PID differs from ready worker PID"
            )
        _send_residual_start(
            client,
            request=loaded_request,
            worker_session_id=worker_binding["worker_session_id"],
        )
        start_ack = _validate_start_ack(
            macro_runner.wait_for_operation_ack(
                client,
                operation="start_macro_fsm",
                request_id=loaded_request.request_id,
                timeout_s=args.operation_timeout_s,
            ),
            request=loaded_request,
            worker_binding=worker_binding,
        )
        terminal = _wait_for_residual_terminal(
            client,
            request_id=loaded_request.request_id,
            timeout_s=args.terminal_timeout_s,
        )
        terminal_validation = _validate_outer_terminal(
            terminal,
            request=loaded_request,
            worker_binding=worker_binding,
            reference_projection=preflight["reviewed_reference"]["projection"],
        )
        terminal_ack = _validate_terminal_ack(
            macro_runner.wait_for_operation_ack(
                client,
                operation="macro_fsm",
                request_id=loaded_request.request_id,
                timeout_s=args.operation_timeout_s,
            ),
            terminal=terminal,
            request=loaded_request,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None and getattr(client, "process", None) is not None:
            shutdown_request_id = uuid.uuid4().hex
            try:
                shutdown_outcome = dict(
                    client.shutdown(
                        mode="fast",
                        timeout_s=60.0,
                        force_on_timeout=True,
                        request_id=shutdown_request_id,
                    )
                    or {}
                )
                if terminal:
                    _validate_residual_fast_shutdown(
                        shutdown_outcome,
                        shutdown_request_id=shutdown_request_id,
                        request=loaded_request,
                        worker_binding=worker_binding,
                    )
                    shutdown_verified = True
            except Exception as close_exc:
                close_error = f"{type(close_exc).__name__}: {close_exc}"
                error = f"{error}; {close_error}" if error else close_error
            try:
                client.close()
            except Exception:
                pass
        try:
            lock.release()
        except Exception as lock_exc:
            lock_error = f"{type(lock_exc).__name__}: {lock_exc}"
            error = f"{error}; {lock_error}" if error else lock_error

    if bool(getattr(lock, "acquired", False)):
        error = f"{error}; singleton lock did not release" if error else (
            "singleton lock did not release"
        )
    try:
        macro_runner._assert_no_simulator_process(
            process_snapshot_fn, conflict_detector
        )
    except Exception as process_exc:
        process_error = f"{type(process_exc).__name__}: {process_exc}"
        error = f"{error}; {process_error}" if error else process_error
    complete = bool(
        not error
        and terminal.get("type") == "macro_fsm_complete"
        and terminal.get("accepted") is True
        and terminal_validation.get("semantic_projection_equal") is True
        and shutdown_verified
    )
    manifest = {
        "schema_version": RUNNER_MANIFEST_SCHEMA,
        "created_utc": _utc_now(),
        "live_isaac_execution": True,
        "execution_mode": EXECUTION_MODE,
        "operation": OPERATION,
        "policy_kind": POLICY_KIND,
        "policy_id": loaded_request.policy_id,
        "policy_sha256": loaded_request.policy_sha256,
        "checkpoint_loaded": False,
        "ppo_training_performed": False,
        "source_version": args.source_version,
        "run_dir": str(paths.run_dir),
        "request_id": loaded_request.request_id,
        "request_identity_sha256": loaded_request.request_identity_sha256,
        "residual_request_path": str(paths.residual_request_path),
        "residual_request_sha256": sha256_file(paths.residual_request_path),
        "base_request_id": loaded_request.base_request.request_id,
        "base_request_path": str(paths.base_request_path),
        "base_request_sha256": sha256_file(paths.base_request_path),
        "bundle": _jsonable(preflight["bundle"].to_mapping()),
        "bundle_sha256": preflight["bundle"].bundle_sha256,
        "graph_sha256": preflight["bundle"].graph_sha256,
        "profile_library_sha256": preflight["bundle"].profile_library_sha256,
        "canonical_environment_lock": _jsonable(preflight["environment"]),
        "gate_e_code_identity": _jsonable(preflight["code_identity"]),
        "source_binding": _jsonable(preflight["source_binding"]),
        "gate_c_closure": _jsonable(preflight["gate_c_closure"]),
        "gate_c_closure_sha256": preflight["gate_c_closure"].get(
            "closure_sha256"
        ),
        "reviewed_reference": _jsonable(reference_identity),
        "reference_semantic_projection_sha256": reference_identity.get(
            "semantic_projection_sha256"
        ),
        "singleton_revalidation_equal": bool(
            under_lock_admission and _admission_equal(preflight, under_lock_admission)
        ),
        "worker_binding": _jsonable(worker_binding),
        "start_ack": _jsonable(start_ack),
        "terminal": _jsonable(terminal),
        "terminal_ack": _jsonable(terminal_ack),
        "terminal_validation": _jsonable(terminal_validation),
        "semantic_projection_equal": terminal_validation.get(
            "semantic_projection_equal"
        )
        is True,
        "video_path": terminal_validation.get("video_path", ""),
        "video_sha256": terminal_validation.get("video_sha256", ""),
        "shutdown_outcome": _jsonable(shutdown_outcome),
        "shutdown_verified": shutdown_verified,
        "macro_fsm_complete": complete,
        "manual_video_status": (
            MANUAL_VIDEO_STATUS_PENDING if complete else "NOT_ELIGIBLE"
        ),
        "error": error,
    }
    _atomic_write_json(paths.manifest_path, manifest)
    if not complete:
        raise ResidualRunnerContractError(
            error or "Gate-E R0 did not close with exact successful evidence"
        )
    return ResidualAttempt(
        run_dir=paths.run_dir,
        source_version=args.source_version,
        request_id=loaded_request.request_id,
        macro_fsm_complete=True,
        shutdown_verified=True,
        semantic_projection_equal=True,
        manifest_path=paths.manifest_path,
    )


def _machine_review_context(
    manifest_path: str | Path,
    *,
    expect_reviewed: bool,
    environment_lock_validator: Callable[[], Mapping[str, Any]],
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
    closure_validator: Callable[..., Mapping[str, Any]] = _validate_gate_c_admission,
    reference_validator: Callable[..., Mapping[str, Any]] = _validate_reviewed_reference,
    code_identity_builder: Callable[[], Mapping[str, Any]] = _current_gate_e_code_identity,
    residual_request_loader: Callable[..., WorkerResidualMacroFSMRequest | None] = (
        load_worker_residual_macro_fsm_request
    ),
) -> dict[str, Any]:
    """Rebuild every machine-owned R0 claim without trusting review flags."""

    manifest_file = Path(manifest_path).resolve()
    run = manifest_file.parent
    if manifest_file != (run / RUNNER_MANIFEST_NAME).resolve():
        raise ResidualRunnerContractError("R0 manifest path/name is not canonical")
    if any(_is_within(run, item) for item in EXACT_REVIEWED_REFERENCE_RUNS.values()):
        raise ResidualRunnerContractError("R0 run overlaps a reviewed source run")
    manifest = _strict_json_load(manifest_file, label="Gate-E R0 manifest")
    expected_keys = _REVIEWED_MANIFEST_KEYS if expect_reviewed else _PENDING_MANIFEST_KEYS
    if set(manifest) != expected_keys:
        raise ResidualRunnerContractError("Gate-E R0 manifest keys are not exact")
    _require_utc(manifest.get("created_utc"), label="R0 manifest created_utc")
    source_version = manifest.get("source_version")
    expected_statuses = (
        REVIEWED_ZERO_SUCCESS_STATUSES
        if expect_reviewed
        else frozenset({MANUAL_VIDEO_STATUS_PENDING})
    )
    if (
        manifest.get("schema_version") != RUNNER_MANIFEST_SCHEMA
        or manifest.get("run_dir") != str(run)
        or manifest.get("live_isaac_execution") is not True
        or manifest.get("execution_mode") != EXECUTION_MODE
        or manifest.get("operation") != OPERATION
        or manifest.get("policy_kind") != POLICY_KIND
        or manifest.get("checkpoint_loaded") is not False
        or manifest.get("ppo_training_performed") is not False
        or source_version not in ALLOWED_SOURCE_VERSIONS
        or manifest.get("singleton_revalidation_equal") is not True
        or manifest.get("shutdown_verified") is not True
        or manifest.get("macro_fsm_complete") is not True
        or manifest.get("semantic_projection_equal") is not True
        or manifest.get("manual_video_status") not in expected_statuses
        or str(manifest.get("error", "") or "")
    ):
        raise ResidualRunnerContractError(
            "Gate-E R0 manifest is not an exact successful review candidate"
        )
    _require_sha(
        manifest.get("request_identity_sha256"),
        label="R0 request identity",
    )
    source_binding = manifest.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ResidualRunnerContractError("R0 source binding is missing")
    alignment_path = Path(
        _required_text(source_binding, "alignment_path", label="source binding")
    ).resolve()
    task_success_path = Path(
        _required_text(
            source_binding,
            "task_success_table_path",
            label="source binding",
        )
    ).resolve()
    preflight_args = argparse.Namespace(
        source_version=source_version,
        output_root=run,
        reviewed_reference_run_dir=EXACT_REVIEWED_REFERENCE_RUNS[source_version],
        gate_c_baseline_run_dir=EXACT_GATE_C_BASELINE_RUN,
        alignment_path=alignment_path,
        task_success_table_path=task_success_path,
    )
    preflight = _preflight_admission(
        preflight_args,
        bundle_builder=bundle_builder,
        environment_validator=environment_lock_validator,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        code_identity_builder=code_identity_builder,
        require_fresh_output=False,
    )
    reference_identity = dict(preflight["reviewed_reference"]["identity"])
    for actual, expected, label in (
        (manifest.get("bundle"), _jsonable(preflight["bundle"].to_mapping()), "bundle"),
        (manifest.get("canonical_environment_lock"), preflight["environment"], "environment lock"),
        (manifest.get("gate_e_code_identity"), preflight["code_identity"], "code identity"),
        (manifest.get("source_binding"), preflight["source_binding"], "source binding"),
        (manifest.get("gate_c_closure"), preflight["gate_c_closure"], "Gate-C closure"),
        (manifest.get("reviewed_reference"), reference_identity, "reviewed reference"),
    ):
        if not _strict_json_equal(actual, expected):
            raise ResidualRunnerContractError(f"R0 manifest {label} is stale")
    for key, expected in {
        "bundle_sha256": preflight["bundle"].bundle_sha256,
        "graph_sha256": preflight["bundle"].graph_sha256,
        "profile_library_sha256": preflight["bundle"].profile_library_sha256,
        "gate_c_closure_sha256": preflight["gate_c_closure"].get("closure_sha256"),
        "reference_semantic_projection_sha256": reference_identity.get(
            "semantic_projection_sha256"
        ),
    }.items():
        if manifest.get(key) != expected:
            raise ResidualRunnerContractError(f"R0 manifest {key} mismatch")

    paths = ResidualRunPaths(
        run_dir=run,
        base_request_path=_path_in_run(
            _required_text(manifest, "base_request_path", label="R0 manifest"),
            run,
            label="base request",
        ),
        residual_request_path=_path_in_run(
            _required_text(
                manifest, "residual_request_path", label="R0 manifest"
            ),
            run,
            label="residual request",
        ),
        manifest_path=manifest_file,
    )
    if (
        paths.base_request_path != (run / BASE_REQUEST_NAME).resolve()
        or paths.residual_request_path != (run / RESIDUAL_REQUEST_NAME).resolve()
        or manifest.get("base_request_sha256")
        != sha256_file(paths.base_request_path)
        or manifest.get("residual_request_sha256")
        != sha256_file(paths.residual_request_path)
    ):
        raise ResidualRunnerContractError("R0 request path/SHA binding is stale")
    request = residual_request_loader(
        paths.residual_request_path,
        environment_validator=environment_lock_validator,
    )
    if request is None:
        raise ResidualRunnerContractError("R0 residual request did not load")
    stale_identities = _fresh_identity_values(
        reference_identity, preflight["gate_c_closure"]
    )
    _validate_request_admission(
        request,
        paths=paths,
        source_version=source_version,
        bundle=preflight["bundle"],
        code_identity=preflight["code_identity"],
        environment=preflight["environment"],
        stale_identities=stale_identities,
    )
    for key, expected in {
        "request_id": request.request_id,
        "request_identity_sha256": request.request_identity_sha256,
        "base_request_id": request.base_request.request_id,
        "policy_id": request.policy_id,
        "policy_sha256": request.policy_sha256,
    }.items():
        if manifest.get(key) != expected:
            raise ResidualRunnerContractError(f"R0 manifest {key} mismatch")

    worker_binding = _validate_worker_binding(
        dict(manifest.get("worker_binding", {}) or {}),
        request=request,
        stale_identities=stale_identities,
    )
    start_ack = _validate_start_ack(
        dict(manifest.get("start_ack", {}) or {}),
        request=request,
        worker_binding=worker_binding,
    )
    terminal = dict(manifest.get("terminal", {}) or {})
    terminal_validation = _validate_outer_terminal(
        terminal,
        request=request,
        worker_binding=worker_binding,
        reference_projection=preflight["reviewed_reference"]["projection"],
    )
    terminal_ack = _validate_terminal_ack(
        dict(manifest.get("terminal_ack", {}) or {}),
        terminal=terminal,
        request=request,
    )
    shutdown = dict(manifest.get("shutdown_outcome", {}) or {})
    shutdown_request_id = _required_text(
        shutdown, "request_id", label="R0 shutdown outcome"
    )
    shutdown_validation = _validate_residual_fast_shutdown(
        shutdown,
        shutdown_request_id=shutdown_request_id,
        request=request,
        worker_binding=worker_binding,
    )
    if (
        not _strict_json_equal(
            manifest.get("terminal_validation"), terminal_validation
        )
        or not _strict_json_equal(shutdown, shutdown_validation)
        or manifest.get("video_path") != terminal_validation["video_path"]
        or manifest.get("video_sha256") != terminal_validation["video_sha256"]
    ):
        raise ResidualRunnerContractError(
            "R0 stored terminal/shutdown/video validation is stale"
        )

    if expect_reviewed:
        pending = {
            key: copy.deepcopy(manifest[key]) for key in _PENDING_MANIFEST_KEYS
        }
        pending["manual_video_status"] = MANUAL_VIDEO_STATUS_PENDING
        pre_review_sha = _require_sha(
            manifest.get("pre_review_manifest_sha256"),
            label="pre-review R0 manifest",
        )
        if _serialized_json_sha256(pending) != pre_review_sha:
            raise ResidualRunnerContractError(
                "review changed machine evidence or pre-review manifest binding"
            )
    else:
        pre_review_sha = sha256_file(manifest_file)
    return {
        "manifest": manifest,
        "manifest_path": manifest_file,
        "run_dir": run,
        "paths": paths,
        "request": request,
        "preflight": preflight,
        "reference_identity": reference_identity,
        "worker_binding": worker_binding,
        "start_ack": start_ack,
        "terminal": terminal,
        "terminal_ack": terminal_ack,
        "terminal_validation": terminal_validation,
        "shutdown_outcome": shutdown_validation,
        "pre_review_manifest_sha256": pre_review_sha,
    }


def _manual_verdict_template(context: Mapping[str, Any]) -> dict[str, Any]:
    manifest = context["manifest"]
    request = context["request"]
    reference = context["reference_identity"]
    terminal_validation = context["terminal_validation"]
    ledger = terminal_validation["zero_ledger_validation"]
    environment = context["preflight"]["environment"]
    code_identity = context["preflight"]["code_identity"]
    source_binding = context["preflight"]["source_binding"]
    return {
        "schema_version": MANUAL_VERDICT_SCHEMA,
        "review_complete": False,
        "reviewed_utc": "",
        "source_version": request.source_version,
        "residual_macro_fsm_request_id": request.request_id,
        "request_identity_sha256": request.request_identity_sha256,
        "base_macro_fsm_request_id": request.base_request.request_id,
        "bundle_sha256": request.bundle_sha256,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "run_dir": str(context["run_dir"]),
        "residual_request_path": str(context["paths"].residual_request_path),
        "residual_request_sha256": sha256_file(
            context["paths"].residual_request_path
        ),
        "pending_manifest_path": str(context["manifest_path"]),
        "pending_manifest_sha256": context["pre_review_manifest_sha256"],
        "reviewed_reference_run_dir": reference["run_dir"],
        "reviewed_reference_manifest_path": reference["manifest_path"],
        "reviewed_reference_manifest_sha256": reference["manifest_sha256"],
        "reviewed_reference_manual_review_status": reference[
            "manual_review_status"
        ],
        "reference_semantic_projection_sha256": reference[
            "semantic_projection_sha256"
        ],
        "outer_worker_result_path": terminal_validation[
            "outer_worker_result_path"
        ],
        "outer_worker_result_sha256": terminal_validation[
            "outer_worker_result_sha256"
        ],
        "outer_task_inputs_path": terminal_validation["outer_task_inputs_path"],
        "outer_task_inputs_sha256": terminal_validation[
            "outer_task_inputs_sha256"
        ],
        "base_worker_result_path": terminal_validation["base_worker_result_path"],
        "base_worker_result_sha256": terminal_validation[
            "base_worker_result_sha256"
        ],
        "base_task_inputs_path": terminal_validation["base_task_inputs_path"],
        "base_task_inputs_sha256": terminal_validation[
            "base_task_inputs_sha256"
        ],
        "video_path": terminal_validation["video_path"],
        "video_sha256": terminal_validation["video_sha256"],
        "start_ack_sha256": _canonical_json_sha256(context["start_ack"]),
        "terminal_sha256": _canonical_json_sha256(context["terminal"]),
        "terminal_ack_sha256": _canonical_json_sha256(context["terminal_ack"]),
        "shutdown_outcome_sha256": _canonical_json_sha256(
            context["shutdown_outcome"]
        ),
        "zero_ledger_validation_sha256": _canonical_json_sha256(ledger),
        "gate_c_closure_sha256": manifest["gate_c_closure_sha256"],
        "canonical_environment_lock_sha256": environment[
            "environment_lock_sha256"
        ],
        "gate_e_code_identity_sha256": code_identity["identity_sha256"],
        "source_binding_sha256": source_binding["identity_sha256"],
        "verdict": {
            "task_completed": None,
            "body_crossed_front_face": None,
            "required_leg_lift_completed": None,
            "final_recoverable": None,
            "posture_incomplete": None,
            "robot_fell": None,
            "body_stuck": None,
            "wheel_drive_up_without_required_lift": None,
            "dangerous_body_collision": None,
            "joint_limit_violation": None,
            "severe_penetration": None,
            "notes": "",
        },
    }


def _validate_manual_verdict_document(
    document: Mapping[str, Any], *, context: Mapping[str, Any]
) -> str:
    if set(document) != _MANUAL_VERDICT_KEYS:
        raise ResidualRunnerContractError("ZERO-R0 manual verdict keys are not exact")
    verdict = document.get("verdict")
    if not isinstance(verdict, Mapping) or set(verdict) != _VERDICT_KEYS:
        raise ResidualRunnerContractError("ZERO-R0 verdict booleans are not exact")
    expected = _manual_verdict_template(context)
    for key in _MANUAL_VERDICT_KEYS - {
        "review_complete",
        "reviewed_utc",
        "verdict",
    }:
        if not _strict_json_equal(document.get(key), expected.get(key)):
            raise ResidualRunnerContractError(
                f"ZERO-R0 manual verdict {key} binding mismatch"
            )
    if document.get("review_complete") is not True:
        raise ResidualRunnerContractError("ZERO-R0 manual verdict is incomplete")
    _require_utc(document.get("reviewed_utc"), label="manual verdict reviewed_utc")
    for key in _VERDICT_KEYS - {"notes"}:
        if type(verdict.get(key)) is not bool:
            raise ResidualRunnerContractError(f"manual verdict {key} must be bool")
    if type(verdict.get("notes")) is not str:
        raise ResidualRunnerContractError("manual verdict notes must be text")
    hard_failures = (
        "robot_fell",
        "body_stuck",
        "wheel_drive_up_without_required_lift",
        "dangerous_body_collision",
        "joint_limit_violation",
        "severe_penetration",
    )
    success = bool(
        verdict["task_completed"]
        and verdict["body_crossed_front_face"]
        and verdict["required_leg_lift_completed"]
        and verdict["final_recoverable"]
        and not any(verdict[key] for key in hard_failures)
    )
    if not success:
        classification = "MACRO_FSM_TASK_FAIL"
    elif verdict["posture_incomplete"]:
        classification = "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
    else:
        classification = "MACRO_FSM_TASK_SUCCESS"
    expected_classification = context["reference_identity"][
        "manual_review_status"
    ]
    if (
        classification not in _REVIEWED_STATUS_BY_CLASSIFICATION
        or classification != expected_classification
    ):
        raise ResidualRunnerContractError(
            "ZERO-R0 verdict classification differs from reviewed source reference"
        )
    return classification


def _validate_reviewed_r0_manifest_impl(
    manifest_path: str | Path,
    *,
    environment_lock_validator: Callable[[], Mapping[str, Any]],
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
    closure_validator: Callable[..., Mapping[str, Any]] = _validate_gate_c_admission,
    reference_validator: Callable[..., Mapping[str, Any]] = _validate_reviewed_reference,
    code_identity_builder: Callable[[], Mapping[str, Any]] = _current_gate_e_code_identity,
    residual_request_loader: Callable[..., WorkerResidualMacroFSMRequest | None] = (
        load_worker_residual_macro_fsm_request
    ),
) -> dict[str, Any]:
    context = _machine_review_context(
        manifest_path,
        expect_reviewed=True,
        environment_lock_validator=environment_lock_validator,
        bundle_builder=bundle_builder,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        code_identity_builder=code_identity_builder,
        residual_request_loader=residual_request_loader,
    )
    manifest = context["manifest"]
    bound_verdict = _path_in_run(
        _required_text(manifest, "manual_verdict_path", label="reviewed manifest"),
        context["run_dir"],
        label="bound ZERO-R0 manual verdict",
    )
    if (
        bound_verdict != (context["run_dir"] / MANUAL_VERDICT_NAME).resolve()
        or manifest.get("manual_verdict_schema") != MANUAL_VERDICT_SCHEMA
        or manifest.get("manual_verdict_sha256") != sha256_file(bound_verdict)
    ):
        raise ResidualRunnerContractError("reviewed ZERO-R0 verdict binding is stale")
    document = _strict_json_load(bound_verdict, label="ZERO-R0 manual verdict")
    classification = _validate_manual_verdict_document(document, context=context)
    reviewed_status = _REVIEWED_STATUS_BY_CLASSIFICATION[classification]
    if (
        manifest.get("manual_review_classification") != classification
        or manifest.get("manual_video_status") != reviewed_status
        or manifest.get("reviewed_utc") != document.get("reviewed_utc")
    ):
        raise ResidualRunnerContractError("reviewed ZERO-R0 classification mismatch")
    terminal_validation = context["terminal_validation"]
    result = {
        "schema_version": REVIEW_VALIDATION_SCHEMA,
        "manifest_path": str(context["manifest_path"]),
        "manifest_sha256": sha256_file(context["manifest_path"]),
        "source_version": context["request"].source_version,
        "request_id": context["request"].request_id,
        "request_identity_sha256": context["request"].request_identity_sha256,
        "base_request_id": context["request"].base_request.request_id,
        "bundle_sha256": context["request"].bundle_sha256,
        "graph_sha256": context["request"].graph_sha256,
        "profile_library_sha256": context["request"].profile_library_sha256,
        "canonical_environment_lock_sha256": context["preflight"][
            "environment"
        ]["environment_lock_sha256"],
        "manual_review_status": reviewed_status,
        "manual_review_classification": classification,
        "reviewed_utc": document["reviewed_utc"],
        "manual_verdict_path": str(bound_verdict),
        "manual_verdict_sha256": sha256_file(bound_verdict),
        "video_path": terminal_validation["video_path"],
        "video_sha256": terminal_validation["video_sha256"],
        "worker_result_path": terminal_validation["outer_worker_result_path"],
        "worker_result_sha256": terminal_validation[
            "outer_worker_result_sha256"
        ],
        "task_inputs_path": terminal_validation["outer_task_inputs_path"],
        "task_inputs_sha256": terminal_validation["outer_task_inputs_sha256"],
        "base_worker_result_path": terminal_validation["base_worker_result_path"],
        "base_worker_result_sha256": terminal_validation[
            "base_worker_result_sha256"
        ],
        "base_task_inputs_path": terminal_validation["base_task_inputs_path"],
        "base_task_inputs_sha256": terminal_validation[
            "base_task_inputs_sha256"
        ],
        "gate_e_code_identity_sha256": context["preflight"]["code_identity"][
            "identity_sha256"
        ],
        "source_binding_sha256": context["preflight"]["source_binding"][
            "identity_sha256"
        ],
        "gate_c_closure_sha256": manifest["gate_c_closure_sha256"],
        "reference_semantic_projection_sha256": manifest[
            "reference_semantic_projection_sha256"
        ],
        "zero_ledger_validation_sha256": _canonical_json_sha256(
            terminal_validation["zero_ledger_validation"]
        ),
        "shutdown_verified": True,
        "semantic_projection_equal": True,
        "macro_fsm_complete": True,
        "reviewed_success": True,
    }
    for key, value in result.items():
        if key.endswith("sha256"):
            _require_sha(value, label=f"review validation {key}")
    if set(result) != REVIEW_VALIDATION_KEYS:
        raise ResidualRunnerContractError(
            "review validation normalized keys are not exact"
        )
    return result


def validate_reviewed_r0_manifest(
    manifest_path: str | Path,
    *,
    environment_lock_validator: Callable[[], Mapping[str, Any]] = (
        macro_runner._validate_canonical_environment_lock
    ),
) -> dict[str, Any]:
    """Return normalized R1 admission only after full ZERO-R0 revalidation."""

    return _validate_reviewed_r0_manifest_impl(
        manifest_path,
        environment_lock_validator=environment_lock_validator,
    )


def review_r0_run(
    *,
    run_dir: str | Path,
    verdict_path: str | Path,
    environment_lock_validator: Callable[[], Mapping[str, Any]] = (
        macro_runner._validate_canonical_environment_lock
    ),
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
    closure_validator: Callable[..., Mapping[str, Any]] = _validate_gate_c_admission,
    reference_validator: Callable[..., Mapping[str, Any]] = _validate_reviewed_reference,
    code_identity_builder: Callable[[], Mapping[str, Any]] = _current_gate_e_code_identity,
    residual_request_loader: Callable[..., WorkerResidualMacroFSMRequest | None] = (
        load_worker_residual_macro_fsm_request
    ),
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    manifest_path = run / RUNNER_MANIFEST_NAME
    context = _machine_review_context(
        manifest_path,
        expect_reviewed=False,
        environment_lock_validator=environment_lock_validator,
        bundle_builder=bundle_builder,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        code_identity_builder=code_identity_builder,
        residual_request_loader=residual_request_loader,
    )
    bound_verdict = (run / MANUAL_VERDICT_NAME).resolve()
    if bound_verdict.exists() or any(
        key in context["manifest"] for key in _REVIEW_MANIFEST_KEYS
    ):
        raise ResidualRunnerContractError(
            "ZERO-R0 review is first-write only; verdict/review already exists"
        )
    document = _strict_json_load(verdict_path, label="operator ZERO-R0 verdict")
    classification = _validate_manual_verdict_document(document, context=context)
    _atomic_write_json(bound_verdict, document)
    manifest = copy.deepcopy(context["manifest"])
    manifest.update(
        {
            "manual_video_status": _REVIEWED_STATUS_BY_CLASSIFICATION[
                classification
            ],
            "manual_review_classification": classification,
            "manual_verdict_schema": MANUAL_VERDICT_SCHEMA,
            "manual_verdict_path": str(bound_verdict),
            "manual_verdict_sha256": sha256_file(bound_verdict),
            "pre_review_manifest_sha256": context[
                "pre_review_manifest_sha256"
            ],
            "reviewed_utc": document["reviewed_utc"],
        }
    )
    _atomic_write_json(manifest_path, manifest)
    return _validate_reviewed_r0_manifest_impl(
        manifest_path,
        environment_lock_validator=environment_lock_validator,
        bundle_builder=bundle_builder,
        closure_validator=closure_validator,
        reference_validator=reference_validator,
        code_identity_builder=code_identity_builder,
        residual_request_loader=residual_request_loader,
    )


def emit_review_template(
    *,
    run_dir: str | Path,
    output_path: str | Path | None = None,
    environment_lock_validator: Callable[[], Mapping[str, Any]] = (
        macro_runner._validate_canonical_environment_lock
    ),
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    context = _machine_review_context(
        run / RUNNER_MANIFEST_NAME,
        expect_reviewed=False,
        environment_lock_validator=environment_lock_validator,
    )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else (run / MANUAL_VERDICT_TEMPLATE_NAME).resolve()
    )
    if destination.exists() or destination in {
        (run / RUNNER_MANIFEST_NAME).resolve(),
        (run / MANUAL_VERDICT_NAME).resolve(),
    }:
        raise ResidualRunnerContractError(
            "review template output must be a fresh non-authoritative path"
        )
    _atomic_write_json(destination, _manual_verdict_template(context))
    return {
        "template_path": str(destination),
        "template_sha256": sha256_file(destination),
        "schema_version": MANUAL_VERDICT_SCHEMA,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one strict live Gate-E ZERO-R0 residual identity trial."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Launch exactly one fresh ZERO-R0 worker.")
    run.add_argument(
        "--source-version",
        choices=tuple(ALLOWED_SOURCE_VERSIONS),
        required=True,
    )
    run.add_argument("--reviewed-reference-run-dir", type=Path, required=True)
    run.add_argument("--gate-c-baseline-run-dir", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--alignment-path",
        type=Path,
        default=macro_runner.DEFAULT_ALIGNMENT_PATH,
    )
    run.add_argument(
        "--task-success-table-path",
        type=Path,
        default=macro_runner.DEFAULT_TASK_SUCCESS_TABLE_PATH,
    )
    run.add_argument("--telemetry-hz", type=float, default=DEFAULT_TELEMETRY_HZ)
    run.add_argument(
        "--post-run-settle-s", type=float, default=DEFAULT_POST_SETTLE_S
    )
    run.add_argument("--task-timeout-s", type=float, default=240.0)
    run.add_argument("--terminal-timeout-s", type=float, default=7200.0)
    run.add_argument("--operation-timeout-s", type=float, default=30.0)
    run.add_argument("--sim-startup-timeout-s", type=float, default=600.0)
    run.add_argument("--sim-worker-status-timeout-s", type=float, default=10.0)
    run.add_argument("--sim-worker-log-lines", type=int, default=200)
    run.add_argument(
        "--worker-launch-mode",
        choices=("auto", "current-python", "isaaclab-bat", "explicit-python"),
        default="auto",
    )
    run.add_argument("--worker-python-exe", default="")
    run.add_argument("--isaaclab-bat", default="C:/robotics_sim/IsaacLab/isaaclab.bat")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--livestream", type=int, default=0)
    run.add_argument("--experience", default="")
    run.add_argument("--accept-isaac-eula", action="store_true", default=False)
    review = commands.add_parser(
        "review", help="Bind one SHA-closed manual ZERO-R0 video verdict."
    )
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--verdict", type=Path, required=True)
    template = commands.add_parser(
        "emit-review-template",
        help="Revalidate a pending R0 and emit its exact verdict template.",
    )
    template.add_argument("--run-dir", type=Path, required=True)
    template.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        result = review_r0_run(run_dir=args.run_dir, verdict_path=args.verdict)
    elif args.command == "emit-review-template":
        result = emit_review_template(
            run_dir=args.run_dir, output_path=args.output
        )
    elif args.command == "run":
        attempt = run_one_zero_r0(args)
        result = {
            "run_dir": str(attempt.run_dir),
            "manifest_path": str(attempt.manifest_path),
            "macro_fsm_complete": attempt.macro_fsm_complete,
            "shutdown_verified": attempt.shutdown_verified,
            "semantic_projection_equal": attempt.semantic_projection_equal,
        }
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXACT_GATE_C_BASELINE_RUN",
    "EXACT_REVIEWED_REFERENCE_RUNS",
    "MANUAL_VERDICT_NAME",
    "MANUAL_VERDICT_SCHEMA",
    "MANUAL_VIDEO_STATUS_PENDING",
    "MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS",
    "MANUAL_VIDEO_STATUS_REVIEWED_SUCCESS_POSTURE_INCOMPLETE",
    "REVIEWED_ZERO_SUCCESS_STATUSES",
    "REVIEW_VALIDATION_KEYS",
    "REVIEW_VALIDATION_SCHEMA",
    "RUNNER_MANIFEST_NAME",
    "RUNNER_MANIFEST_SCHEMA",
    "ResidualAttempt",
    "ResidualRunnerContractError",
    "ResidualRunPaths",
    "build_parser",
    "emit_review_template",
    "main",
    "review_r0_run",
    "run_one_zero_r0",
    "sha256_file",
    "validate_reviewed_r0_manifest",
]
