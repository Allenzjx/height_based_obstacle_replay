"""Serial normal-development runner for the recording-derived 50 mm Macro FSM.

This module is orchestration only.  It sends no actuator command or playback
plan: the production worker rebuilds the SHA-bound controller bundle locally,
and :class:`WorkerMacroFSMSession` applies controller decisions through the one
existing :class:`SimRobotAdapter` motion-batch path at the physics rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from completion_aware_segment import SegmentDecisionKind
from sim_ipc_protocol import MACRO_FAST_CLOSE_SCHEMA

from .fsm50_macro_controller import (
    MacroControllerBundle,
    MacroSegmentCompletionToken,
    build_gate_c_bundle,
)
from .worker_macro_fsm_session import (
    AUTHORIZED_GATE_D_SOURCE_VERSIONS,
    CANONICAL_GATE_C_SOURCE_VERSION,
    CANONICAL_TASK_SUCCESS_TABLE_SHA256,
    DEFAULT_POST_SETTLE_S,
    DEFAULT_TELEMETRY_HZ,
    DEFAULT_VIDEO_FPS,
    GATE_D_TRIAL_KIND,
    EXPECTED_RENDER_SUBSTEPS,
    REQUEST_SCHEMA,
    SEGMENT_COMPLETION_SCHEMA,
    SESSION_SCHEMA,
    TASK_INPUTS_SCHEMA,
    WorkerMacroFSMSession,
    load_worker_macro_fsm_request,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "runs" / "v003_macro_fsm"
DEFAULT_GATE_D_OUTPUT_ROOT = MODULE_ROOT / "runs" / "cross_version_macro_fsm"
DEFAULT_ALIGNMENT_PATH = MODULE_ROOT / "reports" / "50MM_COMMON_PHASE_ALIGNMENT.csv"
DEFAULT_TASK_SUCCESS_TABLE_PATH = (
    MODULE_ROOT / "reports" / "50MM_REPLAY_TASK_SUCCESS_TABLE.csv"
)
DEFAULT_ENVIRONMENT_LOCK_PATH = MODULE_ROOT / "reports" / "environment_lock_50mm.json"
RUNNER_MANIFEST_NAME = "macro_fsm_runner_manifest.json"
RUNNER_MANIFEST_SCHEMA = "fsm50.macro_fsm_runner_manifest.v1"
MANUAL_VERDICT_NAME = "manual_macro_video_verdict.json"
MANUAL_VERDICT_TEMPLATE_NAME = "manual_macro_video_verdict.template.json"
MANUAL_VERDICT_SCHEMA = "fsm50.manual_macro_video_verdict.v1"
GATE_C_CLOSURE_EVIDENCE_SCHEMA = "fsm50.gate_c_closure_evidence.v1"

_LOWER_SHA256 = frozenset("0123456789abcdef")
_SEGMENT_LEDGER_BINDING_KEYS = (
    "segment_completion_ledger_path",
    "segment_completion_ledger_sha256",
    "segment_completion_count",
    "expected_segment_completion_count",
    "segment_completion_coverage_complete",
    "segment_completion_coverage_errors",
)
_SEGMENT_COMPLETION_ROW_KEYS = frozenset(
    {
        "schema_version",
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
        "start_sim_step",
        "start_sim_time_s",
        "start_physical_dispatch",
        "start_batch_id",
        "start_first_physics_step",
        "start_readback_verified",
        "start_readback_verified_sim_step",
        "start_readback_sha256",
        "retained_epoch_same_target",
        "tracking_begin_count",
        "tracking_begin_sim_step",
        "tracking_begin_evidence",
        "tracking_end_count",
        "tracking_end_attempt_count",
        "tracking_lifecycle_closed",
        "tracking_end_sim_step",
        "tracking_end_sim_time_s",
        "tracking_end_reason",
        "tracking_end_evidence",
        "observation_decisions",
        "last_decision",
        "last_decision_sha256",
        "wheel_stop",
        "terminal_kind",
        "terminal_sim_step",
        "terminal_sim_time_s",
        "terminal_decision_sha256",
    }
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
_VERDICT_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "review_complete",
        "source_version",
        "request_id",
        "bundle_sha256",
        "run_dir",
        "worker_result_path",
        "worker_result_sha256",
        "task_inputs_path",
        "task_inputs_sha256",
        "video_path",
        "video_sha256",
        "reviewed_utc",
        "verdict",
    }
)


class MacroRunnerContractError(RuntimeError):
    """A fail-closed orchestration or identity contract was not satisfied."""


def _validate_source_trial_contract(
    source_version: str,
    trial_kind: str,
    *,
    error_type: type[Exception] = MacroRunnerContractError,
) -> str:
    """Validate the closed Gate-C/Gate-D source matrix and return its branch."""

    if source_version == CANONICAL_GATE_C_SOURCE_VERSION:
        if trial_kind == "baseline":
            return "baseline"
        if trial_kind == "repeat":
            return "repeats"
        message = "canonical Gate-C v003 permits only baseline or repeat trials"
    elif source_version in AUTHORIZED_GATE_D_SOURCE_VERSIONS:
        if trial_kind == GATE_D_TRIAL_KIND:
            return "trials"
        message = "authorized Gate-D sources permit only cross_version trials"
    else:
        message = "Macro source is not authorized for Gate-C or Gate-D execution"
    raise error_type(message)


@dataclass(frozen=True)
class MacroRunPaths:
    run_dir: Path
    request_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class MacroAttempt:
    run_dir: Path
    source_version: str
    trial_kind: str
    trial_index: int
    controller_complete: bool
    shutdown_verified: bool
    manifest_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return json.dumps(
        _jsonable(left),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == json.dumps(
        _jsonable(right),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_canonical_task_success_table(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    canonical = DEFAULT_TASK_SUCCESS_TABLE_PATH.resolve()
    if resolved != canonical:
        raise MacroRunnerContractError(
            "Macro execution requires the canonical sealed Gate-A task-success table"
        )
    if not canonical.is_file():
        raise MacroRunnerContractError("canonical Gate-A task-success table is missing")
    if sha256_file(canonical) != CANONICAL_TASK_SUCCESS_TABLE_SHA256:
        raise MacroRunnerContractError(
            "canonical Gate-A task-success table SHA does not match the sealed identity"
        )
    return canonical


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("runner artifacts cannot contain NaN or Infinity")
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return str(value.value)
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _jsonable(value.to_mapping())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _jsonable(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def strict_json_load(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}: {source}")
            result[key] = value
        return result

    value = json.loads(
        source.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def _strict_jsonl_load(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    text = source.read_text(encoding="utf-8")
    if not text or not text.endswith("\n"):
        raise MacroRunnerContractError(
            f"segment completion ledger is empty or lacks its final newline: {source}"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise MacroRunnerContractError(
                f"segment completion ledger contains a blank row at line {line_number}"
            )

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(
                        f"duplicate JSON key {key!r}: {source}:{line_number}"
                    )
                result[key] = value
            return result

        try:
            row = json.loads(
                line,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
        except Exception as exc:
            raise MacroRunnerContractError(
                f"segment completion ledger row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise MacroRunnerContractError(
                f"segment completion ledger row {line_number} is not an object"
            )
        rows.append(row)
    return rows


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and set(value).issubset(_LOWER_SHA256)
    )


def _environment_lock_dependencies() -> tuple[
    Callable[[Path], dict[str, Any]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    from .run_fsm50 import _load_environment_lock, _verify_locked_source_hashes

    return _load_environment_lock, _verify_locked_source_hashes


def _validate_canonical_environment_lock(
    *,
    lock_path: str | Path = DEFAULT_ENVIRONMENT_LOCK_PATH,
    load_lock: Callable[[Path], Mapping[str, Any]] | None = None,
    verify_sources: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove the canonical source closure is current before any Macro rebuild."""

    canonical = DEFAULT_ENVIRONMENT_LOCK_PATH.resolve()
    resolved = Path(lock_path).resolve()
    if resolved != canonical:
        raise MacroRunnerContractError(
            "Macro execution requires the canonical environment_lock_50mm.json"
        )
    if load_lock is None or verify_sources is None:
        default_load, default_verify = _environment_lock_dependencies()
        load_lock = load_lock or default_load
        verify_sources = verify_sources or default_verify
    try:
        lock_sha256_before = sha256_file(resolved)
        lock = dict(load_lock(resolved))
        verification = dict(verify_sources(lock))
        lock_sha256_after = sha256_file(resolved)
    except Exception as exc:
        raise MacroRunnerContractError(
            f"canonical environment lock is unavailable or invalid: {resolved}"
        ) from exc
    files = verification.get("files")
    locked_count = verification.get("locked_source_file_count")
    required_count = verification.get("required_source_file_count")
    invalid_files: list[str] = []
    if isinstance(files, list):
        for row in files:
            if not isinstance(row, Mapping):
                invalid_files.append("<malformed verification row>")
            elif row.get("ok") is not True:
                invalid_files.append(str(row.get("path", "") or "<unknown path>"))
    problems: list[str] = []
    if verification.get("ok") is not True:
        problems.append("source hash verification is not current")
    if lock_sha256_after != lock_sha256_before:
        problems.append("environment lock changed during source verification")
    if verification.get("source_closure_complete") is not True:
        problems.append("source closure is incomplete")
    if (
        type(locked_count) is not int
        or type(required_count) is not int
        or locked_count <= 0
        or required_count <= 0
        or locked_count != required_count
    ):
        problems.append("locked/required source counts are not exact")
    if not isinstance(files, list) or len(files) != locked_count:
        problems.append("verified source row count is not exact")
    if list(verification.get("missing_from_lock", []) or []):
        problems.append("required sources are missing from the lock")
    if invalid_files:
        problems.append(
            "locked source mismatches: " + ", ".join(invalid_files[:8])
        )
    if problems:
        raise MacroRunnerContractError(
            "canonical environment lock is stale: " + "; ".join(problems)
        )
    verification_sha256 = _sha256_mapping(verification)
    return {
        "environment_lock_path": str(resolved),
        "environment_lock_sha256": lock_sha256_after,
        "locked_source_file_count": locked_count,
        "required_source_file_count": required_count,
        "source_closure_complete": True,
        "source_verification_sha256": verification_sha256,
    }


def allocate_run_paths(
    output_root: str | Path,
    *,
    source_version: str,
    trial_kind: str,
    trial_index: int,
    bundle_sha256: str,
) -> MacroRunPaths:
    branch = _validate_source_trial_contract(
        source_version,
        trial_kind,
        error_type=ValueError,
    )
    label = f"{_utc_stamp()}_{trial_kind}_{trial_index:02d}_{bundle_sha256[:12]}"
    run_dir = (Path(output_root) / source_version / branch / label).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return MacroRunPaths(
        run_dir=run_dir,
        request_path=run_dir / "worker_macro_fsm_request.json",
        manifest_path=run_dir / RUNNER_MANIFEST_NAME,
    )


def build_worker_macro_request(
    bundle: MacroControllerBundle,
    paths: MacroRunPaths,
    *,
    request_id: str,
    source_version: str,
    alignment_path: str | Path,
    task_success_table_path: str | Path,
    trial_kind: str,
    trial_index: int,
    telemetry_hz: float = DEFAULT_TELEMETRY_HZ,
    post_run_settle_s: float = DEFAULT_POST_SETTLE_S,
    timeout_s: float = 240.0,
) -> dict[str, Any]:
    _validate_source_trial_contract(source_version, trial_kind)
    if bundle.primary_source_version != source_version:
        raise MacroRunnerContractError(
            "Macro bundle primary source does not match the requested source"
        )
    alignment = Path(alignment_path).resolve()
    success_table = _validate_canonical_task_success_table(task_success_table_path)
    if not alignment.is_file() or not success_table.is_file():
        raise FileNotFoundError("Gate-B alignment and Gate-A success table are required")
    request = {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": str(request_id),
        "source_version": str(source_version),
        "profile_id": bundle.profiles.library_id,
        "graph_id": bundle.graph.graph_id,
        "graph_sha256": bundle.graph_sha256,
        "profile_library_sha256": bundle.profile_library_sha256,
        "bundle_sha256": bundle.bundle_sha256,
        "height_mm": 50,
        "run_dir": str(paths.run_dir),
        "alignment_path": str(alignment),
        "alignment_sha256": sha256_file(alignment),
        "task_success_table_path": str(success_table),
        "task_success_table_sha256": sha256_file(success_table),
        "trial_kind": trial_kind,
        "trial_index": int(trial_index),
        "telemetry_hz": float(telemetry_hz),
        "video_fps": DEFAULT_VIDEO_FPS,
        "capture_video": True,
        "post_run_settle_s": float(post_run_settle_s),
        "timeout_s": float(timeout_s),
        "filtered_contact_bank_enabled": False,
    }
    return request


def _build_production_worker_args(
    cli_args: argparse.Namespace, request_path: Path
) -> argparse.Namespace:
    """Use official UI physics defaults and opt into only the Macro hook."""

    from height_replay_ui import build_parser as build_ui_parser
    from height_replay_ui import normalize_motion_args

    worker_args = build_ui_parser().parse_args([])
    worker_args.height_mm = 50
    worker_args.height_cm = None
    worker_args.profile = "fast"
    worker_args.render_interval = 8
    worker_args.headless = False
    worker_args.no_sim = False
    worker_args.sim_launch_mode = "subprocess"
    worker_args.fsm50_gate_request_path = ""
    worker_args.fsm50_task_request_path = ""
    worker_args.fsm50_macro_request_path = str(request_path.resolve())
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
    return worker_args


def wait_for_worker_ready(
    client: Any, *, timeout_s: float, poll_interval_s: float = 0.02
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        client.poll()
        last = dict(client.status() or {})
        if last.get("ready") is True:
            return last
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise MacroRunnerContractError(
                f"worker exited before ready: returncode={process.poll()}"
            )
        if str(last.get("error", "") or ""):
            raise MacroRunnerContractError(f"worker startup failed: {last['error']}")
        time.sleep(max(0.001, float(poll_interval_s)))
    raise MacroRunnerContractError(f"worker readiness timed out: {last!r}")


def validate_macro_worker_binding(
    status: Mapping[str, Any], *, request: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(status)
    preflight = result.get("worker_macro_fsm_preflight")
    session = result.get("worker_macro_fsm_session")
    if result.get("macro_fsm_preflight_ready") is not True:
        raise MacroRunnerContractError("worker Macro FSM preflight is not ready")
    if not isinstance(preflight, Mapping) or not isinstance(session, Mapping):
        raise MacroRunnerContractError("worker has no Macro preflight/session")
    for key, expected in {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": request["request_id"],
        "source_version": request["source_version"],
        "profile_id": request["profile_id"],
        "graph_id": request["graph_id"],
        "graph_sha256": request["graph_sha256"],
        "profile_library_sha256": request["profile_library_sha256"],
        "bundle_sha256": request["bundle_sha256"],
        "height_mm": 50,
        "run_dir": request["run_dir"],
        "trial_kind": request["trial_kind"],
        "trial_index": request["trial_index"],
        "telemetry_hz": request["telemetry_hz"],
        "video_fps": DEFAULT_VIDEO_FPS,
        "filtered_contact_bank_enabled": False,
        "preflight_ok": True,
    }.items():
        if preflight.get(key) != expected:
            raise MacroRunnerContractError(f"Macro preflight {key} mismatch")
    for key in (
        "worker_pid",
        "worker_session_id",
        "adapter_runtime_instance_id",
        "runtime_version",
    ):
        value = result.get(key)
        if key == "worker_pid":
            valid = type(value) is int and value > 0
        else:
            valid = type(value) is str and bool(value)
            if key == "runtime_version":
                valid = bool(valid and value.lower() != "unavailable")
        if not valid:
            raise MacroRunnerContractError(f"ready Macro worker has invalid {key}")
    if type(result.get("root_state_write_count")) is not int or result.get("root_state_write_count") != 0:
        raise MacroRunnerContractError("normal Macro worker unexpectedly wrote root state")
    if (
        session.get("enabled") is not True
        or session.get("execution_mode") != "normal_development"
        or session.get("request_id") != request["request_id"]
        or session.get("bundle_sha256") != request["bundle_sha256"]
        or session.get("state") != "ready_for_start"
        or session.get("filtered_contact_bank_enabled") is not False
    ):
        raise MacroRunnerContractError("Macro session identity/state is invalid")
    deployed = session.get("deployment_safety_evidence")
    if not isinstance(deployed, Mapping):
        raise MacroRunnerContractError(
            "Macro session has no deployed initial safety evidence"
        )
    for key, expected in {
        "available": True,
        "dangerous_body_collision": False,
        "severe_penetration": False,
        "source_version": request["source_version"],
        "task_success_table_sha256": request["task_success_table_sha256"],
        "alignment_sha256": request["alignment_sha256"],
    }.items():
        if deployed.get(key) != expected:
            raise MacroRunnerContractError(
                f"Macro deployed safety evidence {key} mismatch"
            )
    for key in (
        "gate_a_row_sha256",
        "live_geometry_sha256",
        "initial_state_sha256",
        "deployment_binding_sha256",
    ):
        value = deployed.get(key)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise MacroRunnerContractError(
                f"Macro deployed safety evidence {key} is invalid"
            )
    for name in ("worker_artifact_session", "worker_artifact_preflight"):
        disabled = result.get(name)
        if not isinstance(disabled, Mapping) or disabled.get("enabled") is not False:
            raise MacroRunnerContractError(f"strict artifact pipeline is enabled: {name}")
    if "worker_task_replay_session" in result or "worker_task_replay_preflight" in result:
        raise MacroRunnerContractError("task replay hook must be absent on a Macro worker")
    return result


def wait_for_operation_ack(
    client: Any,
    *,
    operation: str,
    request_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() <= deadline:
        client.poll()
        status = dict(client.status() or {})
        rows = list(status.get("operation_ack_history", []) or [])
        if isinstance(status.get("last_operation_ack"), Mapping):
            rows.append(dict(status["last_operation_ack"]))
        for raw in reversed(rows):
            row = dict(raw or {})
            if row.get("operation") == operation and row.get("request_id") == request_id:
                return row
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise MacroRunnerContractError(f"worker exited waiting for {operation}")
        time.sleep(max(0.001, float(poll_interval_s)))
    raise MacroRunnerContractError(f"timed out waiting for {operation} ACK")


def wait_for_macro_terminal(
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
                and row.get("operation") == "macro_fsm"
                and row.get("request_id") == request_id
            ):
                return row
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise MacroRunnerContractError("worker exited before Macro terminal")
        time.sleep(max(0.001, float(poll_interval_s)))
    raise MacroRunnerContractError("timed out waiting for Macro terminal")


def _path_in_run(path: str | Path, run_dir: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise MacroRunnerContractError(f"artifact escapes run_dir: {resolved}") from exc
    if not resolved.is_file():
        raise MacroRunnerContractError(f"required artifact is missing: {resolved}")
    return resolved


def _expected_segment_start_actions(
    *,
    bundle: MacroControllerBundle,
    request: Mapping[str, Any],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Rebuild the same sealed source sequence used by the worker."""

    request_path = (run_dir / "worker_macro_fsm_request.json").resolve()
    if not _strict_json_equal(strict_json_load(request_path), dict(request)):
        raise MacroRunnerContractError(
            "segment ledger request differs from the exact durable request"
        )
    try:
        loaded = load_worker_macro_fsm_request(request_path)
        if loaded is None:
            raise RuntimeError("Macro request is disabled")
        auditor = WorkerMacroFSMSession(
            loaded,
            worker_session_id="runner-static-segment-ledger-audit",
        )
        actions = auditor._build_expected_source_actions(bundle)
    except Exception as exc:
        raise MacroRunnerContractError(
            "could not rebuild canonical segment bindings for ledger validation"
        ) from exc
    starts = [
        dict(row)
        for row in actions
        if row["command_provenance"]["dispatch_kind"] == "segment_start"
    ]
    if [
        row["command_provenance"]["source_segment_index"] for row in starts
    ] != list(range(len(starts))):
        raise MacroRunnerContractError(
            "rebuilt segment-start bindings do not cover canonical 0..N-1"
        )
    completion_stops: dict[int, dict[str, Any]] = {}
    for row in actions:
        provenance = row["command_provenance"]
        if provenance["dispatch_kind"] != "wheel_channel_completion_stop":
            continue
        segment = int(provenance["source_segment_index"])
        if segment in completion_stops:
            raise MacroRunnerContractError(
                "rebuilt source contains duplicate wheel completion stops"
            )
        completion_stops[segment] = dict(row)
    return starts, completion_stops


def _successful_ledger_claim(
    mapping: Mapping[str, Any],
    *,
    label: str,
    require_aliases: bool,
) -> dict[str, Any]:
    allowed_completion_keys = set(_SEGMENT_LEDGER_BINDING_KEYS)
    if require_aliases:
        allowed_completion_keys.update(
            {"segment_completion_path", "segment_completion_sha256"}
        )
    unexpected = sorted(
        key
        for key in mapping
        if str(key).startswith("segment_completion_")
        and key not in allowed_completion_keys
    )
    if unexpected:
        raise MacroRunnerContractError(
            f"{label} has unexpected segment-completion fields: {unexpected}"
        )
    claim = {key: _jsonable(mapping.get(key)) for key in _SEGMENT_LEDGER_BINDING_KEYS}
    if require_aliases:
        for alias, canonical in (
            ("segment_completion_path", "segment_completion_ledger_path"),
            ("segment_completion_sha256", "segment_completion_ledger_sha256"),
        ):
            if mapping.get(alias) != mapping.get(canonical):
                raise MacroRunnerContractError(
                    f"{label} {alias}/{canonical} aliases differ"
                )
    completed = claim["segment_completion_count"]
    expected = claim["expected_segment_completion_count"]
    if (
        type(completed) is not int
        or type(expected) is not int
        or completed <= 0
        or expected <= 0
        or completed != expected
        or claim["segment_completion_coverage_complete"] is not True
        or claim["segment_completion_coverage_errors"] != []
    ):
        raise MacroRunnerContractError(
            f"{label} does not claim exact successful segment coverage"
        )
    if not _is_lower_sha256(claim["segment_completion_ledger_sha256"]):
        raise MacroRunnerContractError(f"{label} ledger SHA-256 is invalid")
    return claim


def _validate_segment_completion_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    expected_action: Mapping[str, Any],
    expected_stop_action: Mapping[str, Any] | None,
    previous_terminal_sim_step: int | None,
) -> None:
    label = f"segment completion ledger row {row_index}"
    if set(row) != _SEGMENT_COMPLETION_ROW_KEYS:
        missing = sorted(_SEGMENT_COMPLETION_ROW_KEYS - set(row))
        extra = sorted(set(row) - _SEGMENT_COMPLETION_ROW_KEYS)
        raise MacroRunnerContractError(
            f"{label} keys are not exact; missing={missing} extra={extra}"
        )
    provenance = dict(expected_action["command_provenance"])
    binding = expected_action.get("segment_completion_binding")
    if not isinstance(binding, Mapping):
        raise MacroRunnerContractError(f"{label} has no rebuilt completion binding")
    spec = dict(binding["completion_spec"])
    exact = {
        "schema_version": SEGMENT_COMPLETION_SCHEMA,
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
        "start_source_action_identity": provenance["source_action_identity"],
        "source_action_consumption_index": expected_action["source_action_index"],
    }
    for key, expected in exact.items():
        if not _strict_json_equal(row.get(key), expected):
            raise MacroRunnerContractError(f"{label} {key} differs from bundle")

    effective = row.get("effective_completion_spec")
    dynamic_duration = row.get("dynamic_servo_duration_s")
    if (
        not isinstance(effective, Mapping)
        or set(effective) != set(spec)
        or type(dynamic_duration) not in (int, float)
        or not math.isfinite(float(dynamic_duration))
        or float(dynamic_duration) < 0.0
        or effective.get("servo_duration_s") != dynamic_duration
        or any(
            not _strict_json_equal(effective.get(key), value)
            for key, value in spec.items()
            if key != "servo_duration_s"
        )
    ):
        raise MacroRunnerContractError(f"{label} effective completion spec is invalid")
    if row.get("effective_servo_reference_velocity_deg_s") != 150.0:
        raise MacroRunnerContractError(f"{label} changed the 150 deg/s reference")
    previous_targets = row.get("pre_action_canonical_servo_targets_deg")
    if (
        not isinstance(previous_targets, Mapping)
        or set(previous_targets) != set(SERVO_JOINT_NAMES)
        or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in previous_targets.values()
        )
    ):
        raise MacroRunnerContractError(f"{label} pre-action target map is invalid")

    start_epoch = row.get("start_command_epoch")
    start_step = row.get("start_sim_step")
    start_time = row.get("start_sim_time_s")
    if (
        type(start_epoch) is not int
        or start_epoch < 0
        or type(start_step) is not int
        or start_step < 0
        or (
            previous_terminal_sim_step is not None
            # COMPLETE is an observation, not a command batch.  The next
            # segment may therefore apply its sole batch in that same
            # scheduler evaluation; its physical actuation/readback remains
            # N+1 and is checked below.  An earlier start is still invalid.
            and start_step < previous_terminal_sim_step
        )
        or type(start_time) not in (int, float)
        or not math.isfinite(float(start_time))
        or float(start_time) < 0.0
        or row.get("start_readback_verified") is not True
        or not _is_lower_sha256(row.get("start_readback_sha256"))
    ):
        raise MacroRunnerContractError(f"{label} start/readback identity is invalid")
    physical = row.get("start_physical_dispatch")
    verified_step = row.get("start_readback_verified_sim_step")
    if type(physical) is not bool:
        raise MacroRunnerContractError(f"{label} start dispatch flag is invalid")
    if physical:
        if (
            type(row.get("start_batch_id")) is not str
            or not row.get("start_batch_id")
            or row.get("start_first_physics_step") != start_step + 1
            or verified_step != start_step + 1
            or row.get("retained_epoch_same_target") is not False
        ):
            raise MacroRunnerContractError(
                f"{label} physical start lacks exact N+1 readback"
            )
    elif (
        row.get("start_batch_id") != ""
        or row.get("start_first_physics_step") is not None
        or verified_step != start_step
        or row.get("retained_epoch_same_target") is not True
    ):
        raise MacroRunnerContractError(
            f"{label} retained same-target start identity is invalid"
        )

    sparse_targets = dict(spec.get("servo_targets_deg", {}) or {})
    expected_tracking_calls = 1 if sparse_targets else 0
    begin_evidence = row.get("tracking_begin_evidence")
    end_evidence = row.get("tracking_end_evidence")
    if (
        row.get("tracking_begin_count") != expected_tracking_calls
        or row.get("tracking_end_count") != expected_tracking_calls
        or row.get("tracking_end_attempt_count") != expected_tracking_calls
        or row.get("tracking_lifecycle_closed") is not True
        or row.get("tracking_begin_sim_step") != start_step
        or row.get("tracking_end_reason") != SegmentDecisionKind.COMPLETE.value
        or not isinstance(begin_evidence, Mapping)
        or begin_evidence.get("called") is not bool(sparse_targets)
        or sorted(begin_evidence.get("sparse_joint_names", []) or [])
        != sorted(sparse_targets)
        or not isinstance(end_evidence, Mapping)
        or type(end_evidence.get("ended")) is not bool
    ):
        raise MacroRunnerContractError(f"{label} tracking lifecycle is invalid")
    if sparse_targets and end_evidence.get("tracking_completion_deferred") is not (
        end_evidence.get("ended") is False
    ):
        raise MacroRunnerContractError(
            f"{label} ended=false diagnostic identity is invalid"
        )

    decisions = row.get("observation_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise MacroRunnerContractError(f"{label} has no completion observations")
    prior_decision_step = start_step
    due_rows: list[Mapping[str, Any]] = []
    due_token_sha256s: list[str] = []
    for decision_index, raw_decision in enumerate(decisions):
        if not isinstance(raw_decision, Mapping):
            raise MacroRunnerContractError(
                f"{label} decision {decision_index} is not an object"
            )
        decision = dict(raw_decision)
        kind = decision.get("kind")
        decision_step = decision.get("sim_step")
        try:
            decision_token = MacroSegmentCompletionToken(
                profile_id=str(row["profile_id"]),
                profile_source_version=str(row["profile_source_version"]),
                owner_state=str(row["owner_state"]),
                source_plan_sha256=str(row["source_plan_sha256"]),
                source_plan_payload_sha256=str(
                    row["source_plan_payload_sha256"]
                ),
                accepted_steps_sha256=str(row["accepted_steps_sha256"]),
                source_segment_index=int(row["source_segment_index"]),
                source_step_index=int(row["source_step_index"]),
                source_step_id=str(row["source_step_id"]),
                start_command_epoch=int(row["start_command_epoch"]),
                start_sim_step=int(row["start_sim_step"]),
                start_readback_sha256=str(row["start_readback_sha256"]),
                decision=decision,
                decision_sha256=_sha256_mapping(decision),
            )
        except Exception as exc:
            raise MacroRunnerContractError(
                f"{label} decision {decision_index} violates the exact helper schema"
            ) from exc
        if not _strict_json_equal(
            decision_token.to_mapping()["decision"], decision
        ):
            raise MacroRunnerContractError(
                f"{label} decision {decision_index} does not round-trip exactly"
            )
        decision_time = decision.get("sim_time_s")
        segment_elapsed = decision.get("segment_elapsed_s")
        servo_errors = decision.get("servo_errors_deg")
        servo_velocities = decision.get("servo_velocity_deg_s")
        requires_live_servo_maps = bool(
            decision.get("servo_planned_done") is True
            or kind
            in {
                SegmentDecisionKind.WHEEL_STOP_DUE.value,
                SegmentDecisionKind.COMPLETE.value,
                SegmentDecisionKind.FAIL.value,
            }
        )
        expected_servo_map_keys = (
            set(sparse_targets) if requires_live_servo_maps else set()
        )
        if (
            type(decision_time) not in (int, float)
            or type(segment_elapsed) not in (int, float)
            or not math.isfinite(float(decision_time))
            or not math.isfinite(float(segment_elapsed))
            or float(segment_elapsed) < 0.0
            or not math.isclose(
                float(segment_elapsed),
                float(decision_time) - float(start_time),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not isinstance(servo_errors, Mapping)
            or not isinstance(servo_velocities, Mapping)
            or set(servo_errors) != expected_servo_map_keys
            or set(servo_velocities) != expected_servo_map_keys
            or any(
                type(value) not in (int, float) or not math.isfinite(float(value))
                for value in [*servo_errors.values(), *servo_velocities.values()]
            )
            or decision.get("servo_tolerance_deg")
            != effective["servo_tolerance_deg"]
            or not _strict_json_equal(
                decision.get("recorded_servo_residual_deg"),
                effective["recorded_servo_residual_deg"],
            )
            or decision.get("legacy_missing_endpoint")
            is not effective["legacy_missing_endpoint"]
            or decision.get("wheel_duration_s")
            != effective["wheel_active_duration_s"]
            or decision.get("hold_duration_s") != effective["explicit_hold_s"]
            or type(decision.get("wheel_elapsed_s")) not in (int, float)
            or not math.isclose(
                float(decision["wheel_elapsed_s"]),
                float(segment_elapsed),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or type(decision.get("hold_elapsed_s")) not in (int, float)
            or not math.isclose(
                float(decision["hold_elapsed_s"]),
                float(segment_elapsed),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or decision.get("servo_planned_done")
            is not bool(float(segment_elapsed) + 1.0e-9 >= float(dynamic_duration))
            or decision.get("wheel_done")
            is not bool(
                float(segment_elapsed) + 1.0e-9
                >= float(effective["wheel_active_duration_s"])
            )
            or decision.get("hold_done")
            is not bool(
                float(segment_elapsed) + 1.0e-9
                >= float(effective["explicit_hold_s"])
            )
            or decision.get("wheel_stop_acknowledged") is not bool(due_rows)
        ):
            raise MacroRunnerContractError(
                f"{label} decision {decision_index} completion inputs are invalid"
            )
        if (
            kind not in {item.value for item in SegmentDecisionKind}
            or decision.get("segment_index") != spec["segment_index"]
            or decision.get("source_step") != spec["source_step"]
            or decision.get("source_step_id") != spec["source_step_id"]
            or type(decision_step) is not int
            or decision_step <= prior_decision_step
            or (decision_step - start_step) % 8 != 0
            or not math.isclose(
                float(decision_time) - float(start_time),
                (decision_step - start_step) / 120.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise MacroRunnerContractError(
                f"{label} decision {decision_index} identity is invalid"
            )
        prior_decision_step = decision_step
        if kind == SegmentDecisionKind.WHEEL_STOP_DUE.value:
            if float(effective["wheel_active_duration_s"]) <= 0.0:
                raise MacroRunnerContractError(
                    f"{label} WHEEL_STOP_DUE has no positive wheel duration"
                )
            due_rows.append(decision)
            due_token_sha256s.append(decision_token.sha256)
        if decision_index < len(decisions) - 1 and kind in {
            SegmentDecisionKind.COMPLETE.value,
            SegmentDecisionKind.FAIL.value,
        }:
            raise MacroRunnerContractError(f"{label} observed after a terminal decision")
    final = dict(decisions[-1])
    if (
        final.get("kind") != SegmentDecisionKind.COMPLETE.value
        or final.get("servo_done") is not True
        or final.get("wheel_done") is not True
        or final.get("hold_done") is not True
        or final.get("segment_done") is not True
        or final.get("wheel_stop_due") is not False
        or str(final.get("failure_reason", "") or "")
        or str(final.get("failure_code", "") or "")
        or row.get("terminal_kind") != SegmentDecisionKind.COMPLETE.value
        or not _strict_json_equal(row.get("last_decision"), final)
        or row.get("last_decision_sha256") != _sha256_mapping(final)
        or row.get("terminal_decision_sha256") != _sha256_mapping(final)
        or row.get("terminal_sim_step") != final.get("sim_step")
        or row.get("terminal_sim_time_s") != final.get("sim_time_s")
        or row.get("tracking_end_sim_step") != final.get("sim_step")
        or row.get("tracking_end_sim_time_s") != final.get("sim_time_s")
    ):
        raise MacroRunnerContractError(f"{label} terminal COMPLETE evidence is invalid")

    wheel_stop = row.get("wheel_stop")
    if len(due_rows) > 1:
        raise MacroRunnerContractError(f"{label} repeats WHEEL_STOP_DUE")
    if not due_rows:
        if wheel_stop is not None:
            raise MacroRunnerContractError(
                f"{label} has a wheel stop without WHEEL_STOP_DUE"
            )
        if final.get("wheel_stop_acknowledged") is not False:
            raise MacroRunnerContractError(
                f"{label} direct COMPLETE unexpectedly acknowledges a wheel stop"
            )
    else:
        due = due_rows[0]
        if (
            due.get("wheel_stop_due") is not True
            or due.get("segment_done") is not False
            or due.get("wheel_stop_acknowledged") is not False
            or not isinstance(wheel_stop, Mapping)
        ):
            raise MacroRunnerContractError(f"{label} wheel-stop trigger is invalid")
        stop_keys = {
            "generated",
            "source_action",
            "source_action_consumption_index",
            "completion_token_sha256",
            "applied_sim_step",
            "first_physics_step",
            "batch_id",
            "n_plus_one_verified",
            "n_plus_one_verified_sim_step",
            "n_plus_one_readback_sha256",
        }
        source_stop = expected_stop_action is not None
        expected_stop_index = (
            expected_stop_action["source_action_index"] if source_stop else None
        )
        if (
            set(wheel_stop) != stop_keys
            or wheel_stop.get("generated") is not (not source_stop)
            or wheel_stop.get("source_action") is not source_stop
            or wheel_stop.get("source_action_consumption_index")
            != expected_stop_index
            or wheel_stop.get("completion_token_sha256")
            != due_token_sha256s[0]
            or wheel_stop.get("applied_sim_step") != due.get("sim_step")
            or type(wheel_stop.get("applied_sim_step")) is not int
            or wheel_stop.get("first_physics_step")
            != wheel_stop.get("applied_sim_step") + 1
            or type(wheel_stop.get("batch_id")) is not str
            or not wheel_stop.get("batch_id")
            or wheel_stop.get("n_plus_one_verified") is not True
            or wheel_stop.get("n_plus_one_verified_sim_step")
            != wheel_stop.get("first_physics_step")
            or not _is_lower_sha256(
                wheel_stop.get("n_plus_one_readback_sha256")
            )
            or final.get("wheel_stop_acknowledged") is not True
        ):
            raise MacroRunnerContractError(f"{label} wheel-stop N+1 evidence is invalid")


def _validate_successful_segment_completion_ledger(
    *,
    terminal: Mapping[str, Any],
    durable_result: Mapping[str, Any],
    task_inputs: Mapping[str, Any],
    request: Mapping[str, Any],
    run_dir: Path,
    bundle: MacroControllerBundle,
) -> dict[str, Any]:
    if durable_result.get("schema_version") != SESSION_SCHEMA:
        raise MacroRunnerContractError("durable Macro result schema is incompatible")
    if task_inputs.get("schema_version") != TASK_INPUTS_SCHEMA:
        raise MacroRunnerContractError("Macro task_inputs schema is incompatible")
    if not _strict_json_equal(terminal.get("task_inputs"), dict(task_inputs)):
        raise MacroRunnerContractError(
            "raw terminal embedded task_inputs differs from its durable artifact"
        )
    completed = task_inputs.get("completed_result")
    if not isinstance(completed, Mapping):
        raise MacroRunnerContractError("Macro task_inputs completed_result is missing")
    if (
        completed.get("dispatch_complete") is not True
        or completed.get("scheduler_complete") is not True
    ):
        raise MacroRunnerContractError(
            "Macro task_inputs does not prove completed dispatch/scheduling"
        )
    controller = completed.get("macro_controller")
    if not isinstance(controller, Mapping):
        raise MacroRunnerContractError(
            "Macro task_inputs macro_controller completion binding is missing"
        )
    claims = [
        _successful_ledger_claim(terminal, label="raw terminal", require_aliases=True),
        _successful_ledger_claim(
            durable_result, label="durable worker result", require_aliases=True
        ),
        _successful_ledger_claim(
            completed, label="task_inputs.completed_result", require_aliases=False
        ),
        _successful_ledger_claim(
            controller,
            label="task_inputs.completed_result.macro_controller",
            require_aliases=False,
        ),
    ]
    if any(not _strict_json_equal(claim, claims[0]) for claim in claims[1:]):
        raise MacroRunnerContractError(
            "segment completion ledger claims differ across terminal/result/task_inputs"
        )
    claim = claims[0]
    ledger_path = _path_in_run(
        str(claim["segment_completion_ledger_path"] or ""), run_dir
    )
    if ledger_path != (run_dir / "macro_segment_completion_ledger.jsonl").resolve():
        raise MacroRunnerContractError(
            "segment completion ledger path is not the canonical in-run filename"
        )
    if sha256_file(ledger_path) != claim["segment_completion_ledger_sha256"]:
        raise MacroRunnerContractError("segment completion ledger SHA is stale")
    rows = _strict_jsonl_load(ledger_path)
    starts, stops = _expected_segment_start_actions(
        bundle=bundle,
        request=request,
        run_dir=run_dir,
    )
    if (
        len(rows) != len(starts)
        or len(rows) != claim["segment_completion_count"]
    ):
        raise MacroRunnerContractError(
            "segment completion ledger count differs from rebuilt source bindings"
        )
    previous_terminal: int | None = None
    for index, (row, expected_action) in enumerate(zip(rows, starts)):
        _validate_segment_completion_row(
            row,
            row_index=index,
            expected_action=expected_action,
            expected_stop_action=stops.get(index),
            previous_terminal_sim_step=previous_terminal,
        )
        previous_terminal = int(row["terminal_sim_step"])
    return {
        "segment_completion_schema_version": SEGMENT_COMPLETION_SCHEMA,
        **claim,
    }


def _validate_manifest_segment_completion_binding(
    manifest: Mapping[str, Any], binding: Mapping[str, Any]
) -> None:
    unexpected = sorted(
        key
        for key in manifest
        if str(key).startswith("segment_completion_") and key not in binding
    )
    if unexpected:
        raise MacroRunnerContractError(
            "runner manifest has unexpected segment-completion fields: "
            f"{unexpected}"
        )
    for key, expected in binding.items():
        if not _strict_json_equal(manifest.get(key), expected):
            raise MacroRunnerContractError(
                f"runner manifest segment completion {key} mismatch"
            )


def validate_macro_terminal(
    terminal: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    run_dir: Path,
    worker_binding: Mapping[str, Any],
    bundle: MacroControllerBundle,
) -> dict[str, Any]:
    result = dict(terminal)
    complete = result.get("type") == "macro_fsm_complete"
    expected_phase = "MACRO_FSM_COMPLETE" if complete else "MACRO_FSM_FAILED"
    for key, expected in {
        "operation": "macro_fsm",
        "phase": expected_phase,
        "accepted": complete,
        "macro_fsm_complete": complete,
        "request_id": request["request_id"],
        "source_version": request["source_version"],
        "profile_id": request["profile_id"],
        "graph_id": request["graph_id"],
        "graph_sha256": request["graph_sha256"],
        "profile_library_sha256": request["profile_library_sha256"],
        "bundle_sha256": request["bundle_sha256"],
        "run_dir": str(run_dir.resolve()),
        "video_writer_quiesced": True,
        "filtered_contact_bank_enabled": False,
        "physics_dt_s": 1.0 / 120.0,
        "worker_pid": worker_binding.get("worker_pid"),
        "worker_session_id": worker_binding.get("worker_session_id"),
        "adapter_runtime_instance_id": worker_binding.get(
            "adapter_runtime_instance_id"
        ),
        "artifact_request_id": "",
        "root_state_write_count": 0,
    }.items():
        if result.get(key) != expected:
            raise MacroRunnerContractError(f"Macro terminal {key} mismatch")
    if complete and (
        result.get("safe_stop_status") != "VERIFIED"
        or result.get("safe_stop_verified") is not True
        or str(result.get("error", "") or "")
    ):
        raise MacroRunnerContractError(
            "successful Macro terminal lacks verified atomic safe stop or is in error"
        )
    worker_result_path = _path_in_run(
        str(result.get("worker_result_path", "") or ""), run_dir
    )
    task_inputs_path = _path_in_run(
        str(result.get("task_inputs_path", "") or ""), run_dir
    )
    _path_in_run(str(result.get("video_path", "") or ""), run_dir)
    if worker_result_path != (run_dir / "worker_macro_fsm_result.json").resolve():
        raise MacroRunnerContractError("durable Macro result filename is not canonical")
    if task_inputs_path != (run_dir / "macro_task_inputs.json").resolve():
        raise MacroRunnerContractError("Macro task_inputs filename is not canonical")
    durable_result = strict_json_load(worker_result_path)
    for key, expected in {
        "execution_mode": "normal_development",
        "request_id": request["request_id"],
        "source_version": request["source_version"],
        "profile_id": request["profile_id"],
        "graph_id": request["graph_id"],
        "graph_sha256": request["graph_sha256"],
        "profile_library_sha256": request["profile_library_sha256"],
        "bundle_sha256": request["bundle_sha256"],
        "run_dir": str(run_dir.resolve()),
        "macro_fsm_complete": complete,
        "video_writer_quiesced": True,
        "filtered_contact_bank_enabled": False,
        "physics_dt_s": 1.0 / 120.0,
        "worker_pid": worker_binding.get("worker_pid"),
        "worker_session_id": worker_binding.get("worker_session_id"),
        "adapter_runtime_instance_id": worker_binding.get(
            "adapter_runtime_instance_id"
        ),
        "artifact_request_id": "",
        "root_state_write_count": 0,
    }.items():
        if durable_result.get(key) != expected:
            raise MacroRunnerContractError(f"durable Macro result {key} mismatch")
    if (
        durable_result.get("task_inputs_path") != str(task_inputs_path)
        or (complete and str(durable_result.get("error", "") or ""))
    ):
        raise MacroRunnerContractError(
            "durable Macro result task_inputs/error binding is invalid"
        )
    for key in ("safe_stop_status", "safe_stop_verified", "safe_stop_error"):
        if durable_result.get(key) != result.get(key):
            raise MacroRunnerContractError(
                f"durable Macro result {key} does not match raw terminal"
            )
    if complete:
        task_inputs = strict_json_load(task_inputs_path)
        _validate_successful_segment_completion_ledger(
            terminal=result,
            durable_result=durable_result,
            task_inputs=task_inputs,
            request=request,
            run_dir=run_dir,
            bundle=bundle,
        )
    return result


def _validate_terminal_ack(
    acknowledgement: Mapping[str, Any],
    *,
    terminal: Mapping[str, Any],
    request: Mapping[str, Any],
    run_dir: Path,
    worker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the durable Macro terminal operation acknowledgement."""

    result = dict(acknowledgement)
    if set(result) != set(terminal):
        raise MacroRunnerContractError(
            "Macro terminal operation ACK keys differ from the raw terminal"
        )
    for key, expected in {
        "type": "operation_ack",
        "operation": "macro_fsm",
        "phase": terminal["phase"],
        "request_id": request["request_id"],
        "accepted": terminal["accepted"],
        "macro_fsm_complete": terminal["macro_fsm_complete"],
        "source_version": request["source_version"],
        "profile_id": request["profile_id"],
        "graph_id": request["graph_id"],
        "graph_sha256": request["graph_sha256"],
        "profile_library_sha256": request["profile_library_sha256"],
        "bundle_sha256": request["bundle_sha256"],
        "run_dir": str(run_dir.resolve()),
        "video_writer_quiesced": True,
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "artifact_request_id": "",
        "root_state_write_count": 0,
    }.items():
        if result.get(key) != expected:
            raise MacroRunnerContractError(
                f"Macro terminal operation ACK {key} mismatch"
            )
    for key, expected in terminal.items():
        if key != "type" and not _strict_json_equal(result.get(key), expected):
            raise MacroRunnerContractError(
                f"Macro terminal operation ACK {key} differs from terminal"
            )
    if terminal.get("macro_fsm_complete") is True:
        for key in (
            *_SEGMENT_LEDGER_BINDING_KEYS,
            "segment_completion_path",
            "segment_completion_sha256",
        ):
            if not _strict_json_equal(result.get(key), terminal.get(key)):
                raise MacroRunnerContractError(
                    f"Macro terminal operation ACK {key} differs from terminal"
                )
    return result


def validate_fast_shutdown(
    outcome: Mapping[str, Any],
    *,
    shutdown_request_id: str,
    macro_request_id: str,
    worker_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(outcome)
    problems: list[str] = []
    identity = {
        "worker_pid": worker_binding.get("worker_pid"),
        "worker_session_id": worker_binding.get("worker_session_id"),
        "adapter_runtime_instance_id": worker_binding.get("adapter_runtime_instance_id"),
        "artifact_request_id": "",
        "macro_fsm_request_id": macro_request_id,
        "root_state_write_count": 0,
        "runtime_version": worker_binding.get("runtime_version"),
        "schema_version": MACRO_FAST_CLOSE_SCHEMA,
    }
    if (
        result.get("returncode") != 0
        or result.get("forced_termination") is not False
        or result.get("normal_exit") is not True
    ):
        problems.append("worker did not exit naturally with returncode 0")
    if result.get("requested_mode") != "fast" or result.get("timed_out") is not False:
        problems.append("fast shutdown did not complete")
    if result.get("pid") != identity["worker_pid"]:
        problems.append("owned worker PID mismatch")
    if result.get("request_id") != shutdown_request_id:
        problems.append("shutdown outcome request_id mismatch")
    close_kwargs = {"wait_for_replicator": False, "skip_cleanup": True}
    required_rows = (
        ("shutdown_ack", result.get("shutdown_ack"), "operation_ack"),
        ("close_requested_ack", result.get("close_requested_ack"), "close_requested"),
        ("close_requested_receipt", result.get("close_requested_receipt"), "close_receipt"),
    )
    returned_ack = result.get("close_returned_ack")
    returned_receipt = result.get("close_returned_receipt")
    returned_present = (
        isinstance(returned_ack, Mapping)
        and bool(returned_ack)
        or isinstance(returned_receipt, Mapping)
        and bool(returned_receipt)
    )
    rows = list(required_rows)
    if returned_present:
        if not (
            isinstance(returned_ack, Mapping)
            and bool(returned_ack)
            and isinstance(returned_receipt, Mapping)
            and bool(returned_receipt)
        ):
            problems.append("close_returned ACK/receipt must be present as one pair")
        rows.extend(
            (
                ("close_returned_ack", returned_ack, "close_returned"),
                ("close_returned_receipt", returned_receipt, "close_receipt"),
            )
        )
    for name, row, expected_type in rows:
        if not isinstance(row, Mapping):
            problems.append(f"{name} missing")
            continue
        if (
            row.get("type") != expected_type
            or row.get("request_id") != shutdown_request_id
            or row.get("mode") != "fast"
            or row.get("accepted") is not True
            or row.get("close_kwargs") != close_kwargs
            or str(row.get("error", "") or "")
        ):
            problems.append(f"{name} envelope invalid")
            continue
        if expected_type == "operation_ack" and row.get("operation") != "shutdown":
            problems.append("shutdown_ack operation invalid")
        if expected_type == "close_receipt" and (
            row.get("received") is not True
            or row.get("close_event_type")
            != (
                "close_returned"
                if name == "close_returned_receipt"
                else "close_requested"
            )
        ):
            problems.append(f"{name} receipt invalid")
        for key, expected in identity.items():
            if row.get(key) != expected:
                problems.append(f"{name} {key} mismatch")
    if problems:
        raise MacroRunnerContractError("fast close not verified: " + "; ".join(problems))
    return result


def _manual_verdict_template(
    *, request: Mapping[str, Any], terminal: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    worker_result = _path_in_run(terminal["worker_result_path"], run_dir)
    task_inputs = _path_in_run(terminal["task_inputs_path"], run_dir)
    video = _path_in_run(terminal["video_path"], run_dir)
    return {
        "schema_version": MANUAL_VERDICT_SCHEMA,
        "review_complete": False,
        "source_version": request["source_version"],
        "request_id": request["request_id"],
        "bundle_sha256": request["bundle_sha256"],
        "run_dir": str(run_dir.resolve()),
        "worker_result_path": str(worker_result),
        "worker_result_sha256": sha256_file(worker_result),
        "task_inputs_path": str(task_inputs),
        "task_inputs_sha256": sha256_file(task_inputs),
        "video_path": str(video),
        "video_sha256": sha256_file(video),
        "reviewed_utc": "",
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


def _default_client_factory(args: argparse.Namespace) -> Any:
    from sim_process_client import SimProcessClient

    return SimProcessClient(args)


def run_one_macro(
    *,
    cli_args: argparse.Namespace,
    bundle: MacroControllerBundle,
    source_version: str,
    trial_kind: str,
    trial_index: int,
    gate_c_closure: Mapping[str, Any] | None = None,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
) -> MacroAttempt:
    paths = allocate_run_paths(
        cli_args.output_root,
        source_version=source_version,
        trial_kind=trial_kind,
        trial_index=trial_index,
        bundle_sha256=bundle.bundle_sha256,
    )
    request_id = uuid.uuid4().hex
    request = build_worker_macro_request(
        bundle,
        paths,
        request_id=request_id,
        source_version=source_version,
        alignment_path=cli_args.alignment_path,
        task_success_table_path=cli_args.task_success_table_path,
        trial_kind=trial_kind,
        trial_index=trial_index,
        telemetry_hz=cli_args.telemetry_hz,
        post_run_settle_s=cli_args.post_run_settle_s,
        timeout_s=cli_args.task_timeout_s,
    )
    atomic_write_json(paths.request_path, request)
    worker_args = _build_production_worker_args(cli_args, paths.request_path)
    factory = client_factory or _default_client_factory
    client = factory(worker_args)
    worker_binding: dict[str, Any] = {}
    start_ack: dict[str, Any] = {}
    terminal: dict[str, Any] = {}
    terminal_ack: dict[str, Any] = {}
    segment_completion_binding: dict[str, Any] = {}
    shutdown_outcome: dict[str, Any] = {}
    shutdown_verified = False
    error = ""
    try:
        client.start()
        ready = wait_for_worker_ready(client, timeout_s=cli_args.sim_startup_timeout_s)
        worker_binding = validate_macro_worker_binding(ready, request=request)
        client.start_macro_fsm(
            request_id=request_id,
            worker_session_id=worker_binding["worker_session_id"],
            source_version=source_version,
            profile_id=request["profile_id"],
            graph_id=request["graph_id"],
            graph_sha256=request["graph_sha256"],
            profile_library_sha256=request["profile_library_sha256"],
            bundle_sha256=request["bundle_sha256"],
        )
        start_ack = wait_for_operation_ack(
            client,
            operation="start_macro_fsm",
            request_id=request_id,
            timeout_s=cli_args.operation_timeout_s,
        )
        for key, expected in {
            "type": "operation_ack",
            "operation": "start_macro_fsm",
            "request_id": request_id,
            "accepted": True,
            "worker_pid": worker_binding["worker_pid"],
            "worker_session_id": worker_binding["worker_session_id"],
            "adapter_runtime_instance_id": worker_binding["adapter_runtime_instance_id"],
            "artifact_request_id": "",
            "root_state_write_count": 0,
            "physics_dt_s": 1.0 / 120.0,
            "graph_sha256": request["graph_sha256"],
            "profile_library_sha256": request["profile_library_sha256"],
            "bundle_sha256": request["bundle_sha256"],
        }.items():
            if start_ack.get(key) != expected:
                raise MacroRunnerContractError(f"Macro start ACK {key} mismatch")
        boundary = start_ack.get("start_boundary_ack")
        if not isinstance(boundary, Mapping):
            raise MacroRunnerContractError("Macro start boundary ACK is missing")
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
            or start_ack.get("first_controller_tick_physics_step") != first_step
            or start_ack.get("earliest_profile_dispatch_physics_step")
            != first_step + EXPECTED_RENDER_SUBSTEPS - 1
            or start_ack.get("earliest_profile_actuation_physics_step")
            != first_step + EXPECTED_RENDER_SUBSTEPS
        ):
            raise MacroRunnerContractError("Macro zero-wheel start boundary ACK is invalid")
        terminal = validate_macro_terminal(
            wait_for_macro_terminal(
                client,
                request_id=request_id,
                timeout_s=cli_args.terminal_timeout_s,
            ),
            request=request,
            run_dir=paths.run_dir,
            worker_binding=worker_binding,
            bundle=bundle,
        )
        segment_completion_binding = {
            "segment_completion_schema_version": SEGMENT_COMPLETION_SCHEMA,
            **_successful_ledger_claim(
                terminal,
                label="validated raw terminal",
                require_aliases=True,
            ),
        }
        terminal_ack = wait_for_operation_ack(
            client,
            operation="macro_fsm",
            request_id=request_id,
            timeout_s=cli_args.operation_timeout_s,
        )
        terminal_ack = _validate_terminal_ack(
            terminal_ack,
            terminal=terminal,
            request=request,
            run_dir=paths.run_dir,
            worker_binding=worker_binding,
        )
        atomic_write_json(
            paths.run_dir / MANUAL_VERDICT_TEMPLATE_NAME,
            _manual_verdict_template(
                request=request,
                terminal=terminal,
                run_dir=paths.run_dir,
            ),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        process = getattr(client, "process", None)
        if process is not None:
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
                    validate_fast_shutdown(
                        shutdown_outcome,
                        shutdown_request_id=shutdown_request_id,
                        macro_request_id=request_id,
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
    controller_complete = bool(
        terminal.get("type") == "macro_fsm_complete"
        and terminal.get("accepted") is True
    )
    manifest = {
        "schema_version": RUNNER_MANIFEST_SCHEMA,
        "created_utc": _utc_now(),
        "source_version": source_version,
        "trial_kind": trial_kind,
        "trial_index": trial_index,
        "request_id": request_id,
        "run_dir": str(paths.run_dir),
        "request_path": str(paths.request_path),
        "request_sha256": sha256_file(paths.request_path),
        "bundle": bundle.to_mapping(),
        "bundle_sha256": bundle.bundle_sha256,
        "worker_binding": _jsonable(worker_binding),
        "start_ack": _jsonable(start_ack),
        "terminal": _jsonable(terminal),
        "terminal_ack": _jsonable(terminal_ack),
        **_jsonable(segment_completion_binding),
        "controller_complete": controller_complete,
        "manual_review_status": (
            "NOT_EVALUATED_PENDING_SHA_BOUND_VIDEO_REVIEW"
            if controller_complete
            else "NOT_ELIGIBLE"
        ),
        "shutdown_outcome": _jsonable(shutdown_outcome),
        "shutdown_verified": shutdown_verified,
        "error": error,
    }
    if gate_c_closure is not None:
        closure = _jsonable(dict(gate_c_closure))
        claimed_closure_sha = closure.get("closure_sha256")
        unhashed_closure = {
            key: value for key, value in closure.items() if key != "closure_sha256"
        }
        if claimed_closure_sha != _sha256_mapping(unhashed_closure):
            raise MacroRunnerContractError(
                "Gate-C closure evidence changed before Gate-D manifest binding"
            )
        manifest["gate_c_closure"] = closure
        manifest["gate_c_closure_sha256"] = claimed_closure_sha
    atomic_write_json(paths.manifest_path, manifest)
    if error or not controller_complete or not shutdown_verified:
        raise MacroRunnerContractError(
            error
            or "Macro run did not produce a successful terminal and verified close"
        )
    return MacroAttempt(
        run_dir=paths.run_dir,
        source_version=source_version,
        trial_kind=trial_kind,
        trial_index=trial_index,
        controller_complete=True,
        shutdown_verified=True,
        manifest_path=paths.manifest_path,
    )


def _process_guard_dependencies() -> tuple[Any, Callable[[], list[dict[str, Any]]], Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]]]:
    from .run_fsm50 import (
        ReplaySingletonLock,
        _existing_simulator_processes,
        _os_process_snapshot,
    )

    return ReplaySingletonLock, _os_process_snapshot, _existing_simulator_processes


def _assert_no_simulator_process(
    snapshot_fn: Callable[[], list[dict[str, Any]]],
    conflict_fn: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    conflicts = list(conflict_fn(snapshot_fn()))
    if conflicts:
        compact = [
            {key: row.get(key) for key in ("pid", "name", "command_line")}
            for row in conflicts
        ]
        raise MacroRunnerContractError(
            "SECOND_SIMULATOR_PROCESS: " + json.dumps(compact, ensure_ascii=False)
        )


def run_trials(
    args: argparse.Namespace,
    *,
    trial_kind: str,
    count: int,
    client_factory: Callable[[argparse.Namespace], Any] | None = None,
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
    lock_factory: Callable[[], Any] | None = None,
    process_snapshot_fn: Callable[[], list[dict[str, Any]]] | None = None,
    conflict_detector: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    environment_lock_validator: Callable[[], Mapping[str, Any]] | None = None,
) -> list[MacroAttempt]:
    _validate_source_trial_contract(args.source_version, trial_kind)
    if type(count) is not int or count < 1:
        raise MacroRunnerContractError("Macro trial count must be a positive integer")
    if trial_kind == GATE_D_TRIAL_KIND and count != 1:
        raise MacroRunnerContractError(
            "each Gate-D cross-version invocation requires exactly one fresh worker"
        )
    validate_environment = (
        environment_lock_validator or _validate_canonical_environment_lock
    )
    prelock_environment = dict(validate_environment())
    if not prelock_environment:
        raise MacroRunnerContractError(
            "canonical environment-lock admission returned no identity"
        )
    gate_c_closure = None
    if trial_kind == GATE_D_TRIAL_KIND:
        gate_c_closure = _validate_gate_c_closure(
            args,
            bundle_builder=bundle_builder,
        )
    _validate_canonical_task_success_table(args.task_success_table_path)
    bundle = bundle_builder(
        PROJECT_ROOT,
        alignment_path=args.alignment_path,
        primary_source_version=args.source_version,
    )
    if bundle.primary_source_version != args.source_version:
        raise MacroRunnerContractError(
            "rebuilt Macro bundle primary source does not match the requested source"
        )
    if gate_c_closure is not None and (
        bundle.graph_sha256 != gate_c_closure["graph_sha256"]
        or bundle.profile_library_sha256
        != gate_c_closure["profile_library_sha256"]
    ):
        raise MacroRunnerContractError(
            "Gate-D bundle does not preserve the validated Gate-C graph/profile library"
        )
    if lock_factory is None or process_snapshot_fn is None or conflict_detector is None:
        default_lock, default_snapshot, default_conflicts = _process_guard_dependencies()
        lock_factory = lock_factory or default_lock
        process_snapshot_fn = process_snapshot_fn or default_snapshot
        conflict_detector = conflict_detector or default_conflicts
    attempts: list[MacroAttempt] = []
    for index in range(int(count)):
        lock = lock_factory()
        try:
            lock.acquire()
            _assert_no_simulator_process(process_snapshot_fn, conflict_detector)
            under_lock_environment = dict(validate_environment())
            if not _strict_json_equal(under_lock_environment, prelock_environment):
                raise MacroRunnerContractError(
                    "canonical environment lock/source closure changed before worker launch"
                )
            if gate_c_closure is not None:
                under_lock_closure = _validate_gate_c_closure(
                    args,
                    bundle_builder=bundle_builder,
                )
                if not _strict_json_equal(under_lock_closure, gate_c_closure):
                    raise MacroRunnerContractError(
                        "Gate-C closure changed before Gate-D worker launch"
                    )
            attempt = run_one_macro(
                cli_args=args,
                bundle=bundle,
                source_version=args.source_version,
                trial_kind=trial_kind,
                trial_index=index,
                gate_c_closure=gate_c_closure,
                client_factory=client_factory,
            )
            if gate_c_closure is not None:
                post_run_closure = _validate_gate_c_closure(
                    args,
                    bundle_builder=bundle_builder,
                )
                if not _strict_json_equal(post_run_closure, gate_c_closure):
                    raise MacroRunnerContractError(
                        "Gate-C closure changed during Gate-D execution"
                    )
        finally:
            lock.release()
        if bool(getattr(lock, "acquired", False)):
            raise MacroRunnerContractError("singleton lock did not release")
        attempts.append(attempt)
        if not attempt.shutdown_verified:
            break
        _assert_no_simulator_process(process_snapshot_fn, conflict_detector)
    return attempts


def _validate_manual_verdict_document(
    document: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    terminal: Mapping[str, Any],
    run: Path,
) -> str:
    if set(document) != _VERDICT_DOCUMENT_KEYS:
        raise MacroRunnerContractError("manual verdict document keys are not exact")
    verdict = document.get("verdict")
    if not isinstance(verdict, Mapping) or set(verdict) != _VERDICT_KEYS:
        raise MacroRunnerContractError("manual verdict keys are not exact")
    expected = _manual_verdict_template(
        request=request,
        terminal=terminal,
        run_dir=run,
    )
    for key in _VERDICT_DOCUMENT_KEYS - {"review_complete", "reviewed_utc", "verdict"}:
        if document.get(key) != expected.get(key):
            raise MacroRunnerContractError(f"manual verdict {key} binding mismatch")
    if document.get("review_complete") is not True or not str(document.get("reviewed_utc", "") or ""):
        raise MacroRunnerContractError("manual verdict is incomplete")
    for key in _VERDICT_KEYS - {"notes"}:
        if type(verdict.get(key)) is not bool:
            raise MacroRunnerContractError(f"manual verdict {key} must be bool")
    if type(verdict.get("notes")) is not str:
        raise MacroRunnerContractError("manual verdict notes must be text")
    success = bool(
        verdict["task_completed"]
        and verdict["body_crossed_front_face"]
        and verdict["required_leg_lift_completed"]
        and verdict["final_recoverable"]
        and not any(
            verdict[key]
            for key in (
                "robot_fell",
                "body_stuck",
                "wheel_drive_up_without_required_lift",
                "dangerous_body_collision",
                "joint_limit_violation",
                "severe_penetration",
            )
        )
    )
    if not success:
        return "MACRO_FSM_TASK_FAIL"
    return (
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE"
        if verdict["posture_incomplete"]
        else "MACRO_FSM_TASK_SUCCESS"
    )


def review_run(*, run_dir: str | Path, verdict_path: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    manifest_path = run / RUNNER_MANIFEST_NAME
    manifest = strict_json_load(manifest_path)
    source_version = str(manifest.get("source_version", "") or "")
    trial_kind = str(manifest.get("trial_kind", "") or "")
    branch = _validate_source_trial_contract(source_version, trial_kind)
    trial_index = manifest.get("trial_index")
    valid_trial_indices = {0, 1, 2} if trial_kind == "repeat" else {0}
    if (
        manifest.get("schema_version") != RUNNER_MANIFEST_SCHEMA
        or manifest.get("run_dir") != str(run)
        or run.parent.name != branch
        or run.parent.parent.name != source_version
        or type(trial_index) is not int
        or trial_index not in valid_trial_indices
        or type(manifest.get("request_id")) is not str
        or not manifest.get("request_id")
        or manifest.get("controller_complete") is not True
        or manifest.get("shutdown_verified") is not True
        or manifest.get("manual_review_status")
        != "NOT_EVALUATED_PENDING_SHA_BOUND_VIDEO_REVIEW"
        or str(manifest.get("error", "") or "")
    ):
        raise MacroRunnerContractError("run is not eligible for manual promotion")

    request_path = _path_in_run(
        str(manifest.get("request_path", "") or ""),
        run,
    )
    if (
        request_path != run / "worker_macro_fsm_request.json"
        or manifest.get("request_sha256") != sha256_file(request_path)
    ):
        raise MacroRunnerContractError("run request artifact is stale")
    request = strict_json_load(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise MacroRunnerContractError("run request schema is incompatible")
    try:
        loaded_request = load_worker_macro_fsm_request(request_path)
    except Exception as exc:
        raise MacroRunnerContractError("run request did not load") from exc
    if loaded_request is None:
        raise MacroRunnerContractError("run request did not load")

    alignment = Path(str(request.get("alignment_path", "") or "")).resolve()
    if (
        not alignment.is_file()
        or request.get("alignment_sha256") != sha256_file(alignment)
    ):
        raise MacroRunnerContractError("run Gate-B alignment artifact is stale")
    success_table = _validate_canonical_task_success_table(
        str(request.get("task_success_table_path", "") or "")
    )
    rebuilt_bundle = build_gate_c_bundle(
        PROJECT_ROOT,
        alignment_path=alignment,
        primary_source_version=source_version,
    )
    if (
        rebuilt_bundle.primary_source_version != source_version
        or manifest.get("bundle") != _jsonable(rebuilt_bundle.to_mapping())
        or manifest.get("bundle_sha256") != rebuilt_bundle.bundle_sha256
    ):
        raise MacroRunnerContractError("run bundle identity is stale")

    request_expected = {
        "schema_version": REQUEST_SCHEMA,
        "enabled": True,
        "execution_mode": "normal_development",
        "request_id": manifest["request_id"],
        "source_version": source_version,
        "profile_id": rebuilt_bundle.profiles.library_id,
        "graph_id": rebuilt_bundle.graph.graph_id,
        "graph_sha256": rebuilt_bundle.graph_sha256,
        "profile_library_sha256": rebuilt_bundle.profile_library_sha256,
        "bundle_sha256": rebuilt_bundle.bundle_sha256,
        "height_mm": 50,
        "run_dir": str(run),
        "alignment_path": str(alignment),
        "alignment_sha256": sha256_file(alignment),
        "task_success_table_path": str(success_table),
        "task_success_table_sha256": sha256_file(success_table),
        "trial_kind": trial_kind,
        "trial_index": trial_index,
        "capture_video": True,
        "video_fps": DEFAULT_VIDEO_FPS,
        "filtered_contact_bank_enabled": False,
    }
    for key, expected in request_expected.items():
        if request.get(key) != expected:
            raise MacroRunnerContractError(f"run request {key} mismatch")

    worker_binding = validate_macro_worker_binding(
        dict(manifest.get("worker_binding", {}) or {}),
        request=request,
    )
    terminal = validate_macro_terminal(
        dict(manifest.get("terminal", {}) or {}),
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
        bundle=rebuilt_bundle,
    )
    if terminal.get("type") != "macro_fsm_complete":
        raise MacroRunnerContractError("run terminal is not successful")
    _validate_manifest_segment_completion_binding(
        manifest,
        {
            "segment_completion_schema_version": SEGMENT_COMPLETION_SCHEMA,
            **_successful_ledger_claim(
                terminal,
                label="reviewed raw terminal",
                require_aliases=True,
            ),
        },
    )
    _validate_terminal_ack(
        dict(manifest.get("terminal_ack", {}) or {}),
        terminal=terminal,
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
    )
    shutdown = dict(manifest.get("shutdown_outcome", {}) or {})
    shutdown_request_id = shutdown.get("request_id")
    if type(shutdown_request_id) is not str or not shutdown_request_id:
        raise MacroRunnerContractError("run shutdown request id is invalid")
    validate_fast_shutdown(
        shutdown,
        shutdown_request_id=shutdown_request_id,
        macro_request_id=request["request_id"],
        worker_binding=worker_binding,
    )

    if trial_kind == GATE_D_TRIAL_KIND:
        _validate_persisted_gate_c_closure(
            manifest,
            request=request,
            bundle_builder=build_gate_c_bundle,
        )
    elif "gate_c_closure" in manifest or "gate_c_closure_sha256" in manifest:
        raise MacroRunnerContractError(
            "Gate-C run unexpectedly contains Gate-D closure evidence"
        )
    document = strict_json_load(verdict_path)
    classification = _validate_manual_verdict_document(
        document,
        request=request,
        terminal=terminal,
        run=run,
    )
    bound_path = run / MANUAL_VERDICT_NAME
    atomic_write_json(bound_path, document)
    manifest["manual_review_status"] = classification
    manifest["manual_verdict_path"] = str(bound_path)
    manifest["manual_verdict_sha256"] = sha256_file(bound_path)
    manifest["reviewed_utc"] = document["reviewed_utc"]
    atomic_write_json(manifest_path, manifest)
    return {
        "run_dir": str(run),
        "manual_review_status": manifest["manual_review_status"],
        "manual_verdict_path": str(bound_path),
    }


def _common_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    source_choices: tuple[str, ...] = (CANONICAL_GATE_C_SOURCE_VERSION,),
    source_default: str | None = CANONICAL_GATE_C_SOURCE_VERSION,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> None:
    parser.add_argument(
        "--source-version",
        choices=source_choices,
        default=source_default,
        required=source_default is None,
    )
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--alignment-path", type=Path, default=DEFAULT_ALIGNMENT_PATH)
    parser.add_argument(
        "--task-success-table-path",
        type=Path,
        default=DEFAULT_TASK_SUCCESS_TABLE_PATH,
    )
    parser.add_argument("--telemetry-hz", type=float, default=DEFAULT_TELEMETRY_HZ)
    parser.add_argument("--post-run-settle-s", type=float, default=DEFAULT_POST_SETTLE_S)
    parser.add_argument("--task-timeout-s", type=float, default=240.0)
    parser.add_argument("--terminal-timeout-s", type=float, default=7200.0)
    parser.add_argument("--operation-timeout-s", type=float, default=30.0)
    parser.add_argument("--sim-startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--sim-worker-status-timeout-s", type=float, default=10.0)
    parser.add_argument("--sim-worker-log-lines", type=int, default=200)
    parser.add_argument(
        "--worker-launch-mode",
        choices=("auto", "current-python", "isaaclab-bat", "explicit-python"),
        default="auto",
    )
    parser.add_argument("--worker-python-exe", default="")
    parser.add_argument("--isaaclab-bat", default="C:/robotics_sim/IsaacLab/isaaclab.bat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--livestream", type=int, default=0)
    parser.add_argument("--experience", default="")
    parser.add_argument("--accept-isaac-eula", action="store_true", default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 50 mm recording-derived Macro FSM.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one v003 baseline in one fresh worker.")
    _common_run_arguments(run)
    repeat = commands.add_parser("repeat", help="Run exactly three fresh-worker repeats.")
    _common_run_arguments(repeat)
    repeat.add_argument("--baseline-run-dir", type=Path, required=True)
    repeat.add_argument("--count", type=int, default=3)
    cross_version = commands.add_parser(
        "cross-version",
        help="Run one authorized Gate-D source in one fresh worker.",
    )
    _common_run_arguments(
        cross_version,
        source_choices=AUTHORIZED_GATE_D_SOURCE_VERSIONS,
        source_default=None,
        output_root=DEFAULT_GATE_D_OUTPUT_ROOT,
    )
    cross_version.add_argument(
        "--gate-c-baseline-run-dir",
        type=Path,
        required=True,
        help=(
            "Reviewed-success v003 baseline whose sibling repeats directory "
            "contains exactly the three reviewed-success repeats 0, 1, and 2."
        ),
    )
    review = commands.add_parser("review", help="Bind a manual viewport-video verdict.")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--verdict", type=Path, required=True)
    return parser


def _validate_repeat_baseline(args: argparse.Namespace) -> None:
    _validate_canonical_task_success_table(args.task_success_table_path)
    run = Path(args.baseline_run_dir).resolve()
    baseline = strict_json_load(run / RUNNER_MANIFEST_NAME)
    if (
        baseline.get("schema_version") != RUNNER_MANIFEST_SCHEMA
        or baseline.get("trial_kind") != "baseline"
        or baseline.get("source_version") != args.source_version
        or baseline.get("shutdown_verified") is not True
        or baseline.get("manual_review_status")
        not in {
            "MACRO_FSM_TASK_SUCCESS",
            "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
        }
    ):
        raise MacroRunnerContractError(
            "repeat requires a reviewed-success baseline with verified shutdown"
        )
    request_path = Path(str(baseline.get("request_path", "") or "")).resolve()
    if (
        not request_path.is_file()
        or baseline.get("request_sha256") != sha256_file(request_path)
    ):
        raise MacroRunnerContractError("baseline request artifact is missing or stale")
    request = strict_json_load(request_path)
    if (
        request.get("source_version") != args.source_version
        or request.get("bundle_sha256") != baseline.get("bundle_sha256")
        or request.get("alignment_path") != str(Path(args.alignment_path).resolve())
        or request.get("alignment_sha256") != sha256_file(args.alignment_path)
        or request.get("task_success_table_path")
        != str(Path(args.task_success_table_path).resolve())
        or request.get("task_success_table_sha256")
        != sha256_file(args.task_success_table_path)
    ):
        raise MacroRunnerContractError(
            "baseline Gate-A/Gate-B source/report binding is stale"
        )
    bound_verdict = run / MANUAL_VERDICT_NAME
    claimed_verdict = Path(
        str(baseline.get("manual_verdict_path", "") or "")
    ).resolve()
    if (
        claimed_verdict != bound_verdict
        or not bound_verdict.is_file()
        or baseline.get("manual_verdict_sha256") != sha256_file(bound_verdict)
    ):
        raise MacroRunnerContractError(
            "baseline manual verdict artifact is missing or stale"
        )
    classification = _validate_manual_verdict_document(
        strict_json_load(bound_verdict),
        request=request,
        terminal=dict(baseline.get("terminal", {}) or {}),
        run=run,
    )
    if (
        classification != baseline.get("manual_review_status")
        or classification
        not in {
            "MACRO_FSM_TASK_SUCCESS",
            "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
        }
    ):
        raise MacroRunnerContractError(
            "baseline manual verdict does not prove task success"
        )
    rebuilt = build_gate_c_bundle(
        PROJECT_ROOT,
        alignment_path=args.alignment_path,
        primary_source_version=args.source_version,
    )
    if baseline.get("bundle_sha256") != rebuilt.bundle_sha256:
        raise MacroRunnerContractError("baseline bundle identity is stale")
    _validate_reviewed_gate_c_attempt(
        run,
        expected_trial_kind="baseline",
        expected_trial_index=0,
        alignment_path=args.alignment_path,
        task_success_table_path=args.task_success_table_path,
        rebuilt_bundle=rebuilt,
    )


def _validate_reviewed_gate_c_attempt(
    run_dir: str | Path,
    *,
    expected_trial_kind: str,
    expected_trial_index: int,
    alignment_path: str | Path,
    task_success_table_path: str | Path,
    rebuilt_bundle: MacroControllerBundle,
) -> dict[str, Any]:
    """Revalidate one reviewed Gate-C attempt from its durable artifacts."""

    run = Path(run_dir).resolve()
    expected_branch = "baseline" if expected_trial_kind == "baseline" else "repeats"
    if (
        not run.is_dir()
        or run.parent.name != expected_branch
        or run.parent.parent.name != CANONICAL_GATE_C_SOURCE_VERSION
    ):
        raise MacroRunnerContractError(
            f"Gate-C {expected_trial_kind} run is not in its canonical sibling branch"
        )
    manifest_path = run / RUNNER_MANIFEST_NAME
    manifest = strict_json_load(manifest_path)
    accepted_statuses = {
        "MACRO_FSM_TASK_SUCCESS",
        "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
    }
    if (
        manifest.get("schema_version") != RUNNER_MANIFEST_SCHEMA
        or manifest.get("run_dir") != str(run)
        or manifest.get("source_version") != CANONICAL_GATE_C_SOURCE_VERSION
        or manifest.get("trial_kind") != expected_trial_kind
        or manifest.get("trial_index") != expected_trial_index
        or manifest.get("controller_complete") is not True
        or manifest.get("shutdown_verified") is not True
        or manifest.get("manual_review_status") not in accepted_statuses
        or str(manifest.get("error", "") or "")
    ):
        raise MacroRunnerContractError(
            f"Gate-C {expected_trial_kind}{expected_trial_index} is not a reviewed success"
        )

    bundle_mapping = _jsonable(rebuilt_bundle.to_mapping())
    if (
        rebuilt_bundle.primary_source_version != CANONICAL_GATE_C_SOURCE_VERSION
        or manifest.get("bundle") != bundle_mapping
        or manifest.get("bundle_sha256") != rebuilt_bundle.bundle_sha256
    ):
        raise MacroRunnerContractError("Gate-C closure bundle identity is stale")

    request_path = _path_in_run(
        str(manifest.get("request_path", "") or ""),
        run,
    )
    if (
        request_path != run / "worker_macro_fsm_request.json"
        or manifest.get("request_sha256") != sha256_file(request_path)
    ):
        raise MacroRunnerContractError("Gate-C closure request artifact is stale")
    request = strict_json_load(request_path)
    loaded_request = load_worker_macro_fsm_request(request_path)
    if loaded_request is None:
        raise MacroRunnerContractError("Gate-C closure request did not load")
    success_table = _validate_canonical_task_success_table(task_success_table_path)
    alignment = Path(alignment_path).resolve()
    request_expected = {
        "request_id": manifest.get("request_id"),
        "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
        "profile_id": rebuilt_bundle.profiles.library_id,
        "graph_id": rebuilt_bundle.graph.graph_id,
        "graph_sha256": rebuilt_bundle.graph_sha256,
        "profile_library_sha256": rebuilt_bundle.profile_library_sha256,
        "bundle_sha256": rebuilt_bundle.bundle_sha256,
        "height_mm": 50,
        "run_dir": str(run),
        "alignment_path": str(alignment),
        "alignment_sha256": sha256_file(alignment),
        "task_success_table_path": str(success_table),
        "task_success_table_sha256": sha256_file(success_table),
        "trial_kind": expected_trial_kind,
        "trial_index": expected_trial_index,
        "capture_video": True,
        "video_fps": DEFAULT_VIDEO_FPS,
        "filtered_contact_bank_enabled": False,
    }
    for key, expected in request_expected.items():
        if request.get(key) != expected:
            raise MacroRunnerContractError(f"Gate-C closure request {key} mismatch")

    worker_binding = validate_macro_worker_binding(
        dict(manifest.get("worker_binding", {}) or {}),
        request=request,
    )
    terminal = validate_macro_terminal(
        dict(manifest.get("terminal", {}) or {}),
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
        bundle=rebuilt_bundle,
    )
    if terminal.get("type") != "macro_fsm_complete":
        raise MacroRunnerContractError("Gate-C closure terminal is not successful")
    ledger_binding = {
        "segment_completion_schema_version": SEGMENT_COMPLETION_SCHEMA,
        **_successful_ledger_claim(
            terminal,
            label="Gate-C closure raw terminal",
            require_aliases=True,
        ),
    }
    _validate_manifest_segment_completion_binding(manifest, ledger_binding)
    _validate_terminal_ack(
        dict(manifest.get("terminal_ack", {}) or {}),
        terminal=terminal,
        request=request,
        run_dir=run,
        worker_binding=worker_binding,
    )

    shutdown = dict(manifest.get("shutdown_outcome", {}) or {})
    shutdown_request_id = shutdown.get("request_id")
    if type(shutdown_request_id) is not str or not shutdown_request_id:
        raise MacroRunnerContractError("Gate-C closure shutdown request id is invalid")
    validate_fast_shutdown(
        shutdown,
        shutdown_request_id=shutdown_request_id,
        macro_request_id=request["request_id"],
        worker_binding=worker_binding,
    )

    verdict_path = _path_in_run(
        str(manifest.get("manual_verdict_path", "") or ""),
        run,
    )
    if (
        verdict_path != run / MANUAL_VERDICT_NAME
        or manifest.get("manual_verdict_sha256") != sha256_file(verdict_path)
    ):
        raise MacroRunnerContractError("Gate-C closure manual verdict is stale")
    classification = _validate_manual_verdict_document(
        strict_json_load(verdict_path),
        request=request,
        terminal=terminal,
        run=run,
    )
    if classification != manifest.get("manual_review_status"):
        raise MacroRunnerContractError(
            "Gate-C closure manual verdict classification mismatch"
        )
    verdict_document = strict_json_load(verdict_path)
    return {
        "run_dir": str(run),
        "source_version": CANONICAL_GATE_C_SOURCE_VERSION,
        "trial_kind": expected_trial_kind,
        "trial_index": expected_trial_index,
        "manual_review_status": classification,
        "runner_manifest_path": str(manifest_path),
        "runner_manifest_sha256": sha256_file(manifest_path),
        "request_path": str(request_path),
        "request_sha256": sha256_file(request_path),
        "manual_verdict_path": str(verdict_path),
        "manual_verdict_sha256": sha256_file(verdict_path),
        "worker_result_path": verdict_document["worker_result_path"],
        "worker_result_sha256": verdict_document["worker_result_sha256"],
        "task_inputs_path": verdict_document["task_inputs_path"],
        "task_inputs_sha256": verdict_document["task_inputs_sha256"],
        "video_path": verdict_document["video_path"],
        "video_sha256": verdict_document["video_sha256"],
        "request_id": request["request_id"],
        "worker_pid": worker_binding["worker_pid"],
        "worker_session_id": worker_binding["worker_session_id"],
        "adapter_runtime_instance_id": worker_binding[
            "adapter_runtime_instance_id"
        ],
        "bundle_sha256": rebuilt_bundle.bundle_sha256,
        "graph_sha256": rebuilt_bundle.graph_sha256,
        "profile_library_sha256": rebuilt_bundle.profile_library_sha256,
        **ledger_binding,
    }


def _validate_gate_c_closure(
    args: argparse.Namespace,
    *,
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
) -> dict[str, Any]:
    """Require one reviewed baseline and exactly three reviewed sibling repeats."""

    baseline_run = Path(args.gate_c_baseline_run_dir).resolve()
    source_root = baseline_run.parent.parent
    if (
        baseline_run.parent.name != "baseline"
        or source_root.name != CANONICAL_GATE_C_SOURCE_VERSION
    ):
        raise MacroRunnerContractError(
            "Gate-D prerequisite must point to a canonical v003 baseline run"
        )
    rebuilt = bundle_builder(
        PROJECT_ROOT,
        alignment_path=args.alignment_path,
        primary_source_version=CANONICAL_GATE_C_SOURCE_VERSION,
    )
    baseline = _validate_reviewed_gate_c_attempt(
        baseline_run,
        expected_trial_kind="baseline",
        expected_trial_index=0,
        alignment_path=args.alignment_path,
        task_success_table_path=args.task_success_table_path,
        rebuilt_bundle=rebuilt,
    )

    repeat_root = source_root / "repeats"
    if not repeat_root.is_dir():
        raise MacroRunnerContractError("Gate-C sibling repeats directory is missing")
    accepted_repeat_dirs: list[Path] = []
    for manifest_path in sorted(repeat_root.glob(f"*/{RUNNER_MANIFEST_NAME}")):
        candidate = strict_json_load(manifest_path)
        if (
            candidate.get("schema_version") == RUNNER_MANIFEST_SCHEMA
            and candidate.get("source_version") == CANONICAL_GATE_C_SOURCE_VERSION
            and candidate.get("trial_kind") == "repeat"
            and candidate.get("controller_complete") is True
            and candidate.get("shutdown_verified") is True
            and candidate.get("manual_review_status")
            in {
                "MACRO_FSM_TASK_SUCCESS",
                "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
            }
        ):
            accepted_repeat_dirs.append(manifest_path.parent.resolve())
    if len(accepted_repeat_dirs) != 3:
        raise MacroRunnerContractError(
            "Gate-D requires exactly three reviewed-success sibling Gate-C repeats; "
            f"found {len(accepted_repeat_dirs)}"
        )

    indexed: dict[int, Path] = {}
    for run in accepted_repeat_dirs:
        candidate = strict_json_load(run / RUNNER_MANIFEST_NAME)
        index = candidate.get("trial_index")
        if type(index) is not int or index in indexed:
            raise MacroRunnerContractError(
                "Gate-C reviewed repeats do not have distinct integer indices"
            )
        indexed[index] = run
    if set(indexed) != {0, 1, 2}:
        raise MacroRunnerContractError(
            "Gate-C reviewed repeat indices must be exactly 0, 1, and 2"
        )
    repeats = [
        _validate_reviewed_gate_c_attempt(
            indexed[index],
            expected_trial_kind="repeat",
            expected_trial_index=index,
            alignment_path=args.alignment_path,
            task_success_table_path=args.task_success_table_path,
            rebuilt_bundle=rebuilt,
        )
        for index in range(3)
    ]
    attempts = [baseline, *repeats]
    for identity_key in (
        "run_dir",
        "request_id",
        "worker_pid",
        "worker_session_id",
        "adapter_runtime_instance_id",
    ):
        values = [row[identity_key] for row in attempts]
        if len(set(values)) != 4:
            raise MacroRunnerContractError(
                f"Gate-C closure does not prove four fresh {identity_key} identities"
            )
    closure = {
        "schema_version": GATE_C_CLOSURE_EVIDENCE_SCHEMA,
        "baseline": baseline,
        "repeats": repeats,
        "bundle_sha256": rebuilt.bundle_sha256,
        "graph_sha256": rebuilt.graph_sha256,
        "profile_library_sha256": rebuilt.profile_library_sha256,
    }
    return {
        **closure,
        "closure_sha256": _sha256_mapping(closure),
    }


def _validate_persisted_gate_c_closure(
    manifest: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    bundle_builder: Callable[..., MacroControllerBundle] = build_gate_c_bundle,
) -> dict[str, Any]:
    """Revalidate a Gate-D manifest's immutable Gate-C admission snapshot."""

    stored = manifest.get("gate_c_closure")
    if not isinstance(stored, Mapping):
        raise MacroRunnerContractError("Gate-D manifest has no Gate-C closure evidence")
    closure = dict(stored)
    claimed = closure.get("closure_sha256")
    unhashed = {key: value for key, value in closure.items() if key != "closure_sha256"}
    if (
        closure.get("schema_version") != GATE_C_CLOSURE_EVIDENCE_SCHEMA
        or type(claimed) is not str
        or manifest.get("gate_c_closure_sha256") != claimed
        or _sha256_mapping(unhashed) != claimed
    ):
        raise MacroRunnerContractError("Gate-D Gate-C closure digest is invalid")
    baseline = closure.get("baseline")
    if not isinstance(baseline, Mapping):
        raise MacroRunnerContractError("Gate-D Gate-C baseline evidence is unavailable")
    current = _validate_gate_c_closure(
        argparse.Namespace(
            gate_c_baseline_run_dir=baseline.get("run_dir"),
            alignment_path=request.get("alignment_path"),
            task_success_table_path=request.get("task_success_table_path"),
        ),
        bundle_builder=bundle_builder,
    )
    if not _strict_json_equal(current, closure):
        raise MacroRunnerContractError(
            "Gate-D Gate-C closure artifacts changed after admission"
        )
    return current


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        result = review_run(run_dir=args.run_dir, verdict_path=args.verdict)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["manual_review_status"] in {
            "MACRO_FSM_TASK_SUCCESS",
            "MACRO_FSM_TASK_SUCCESS_POSTURE_INCOMPLETE",
        } else 1
    if args.command == "run":
        attempts = run_trials(args, trial_kind="baseline", count=1)
    elif args.command == "repeat":
        if args.count != 3:
            raise MacroRunnerContractError("the Gate-C repeat contract requires exactly 3 repeats")
        _validate_canonical_environment_lock()
        _validate_repeat_baseline(args)
        attempts = run_trials(args, trial_kind="repeat", count=3)
    elif args.command == "cross-version":
        attempts = run_trials(args, trial_kind=GATE_D_TRIAL_KIND, count=1)
    else:
        raise AssertionError(args.command)
    print(
        json.dumps(
            {
                "run_dirs": [str(item.run_dir) for item in attempts],
                "shutdown_verified": [item.shutdown_verified for item in attempts],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if attempts and all(item.controller_complete and item.shutdown_verified for item in attempts) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_GATE_D_SOURCE_VERSIONS",
    "DEFAULT_ALIGNMENT_PATH",
    "DEFAULT_GATE_D_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TASK_SUCCESS_TABLE_PATH",
    "GATE_C_CLOSURE_EVIDENCE_SCHEMA",
    "GATE_D_TRIAL_KIND",
    "MANUAL_VERDICT_SCHEMA",
    "MacroAttempt",
    "MacroRunPaths",
    "MacroRunnerContractError",
    "RUNNER_MANIFEST_NAME",
    "RUNNER_MANIFEST_SCHEMA",
    "allocate_run_paths",
    "atomic_write_json",
    "build_parser",
    "build_worker_macro_request",
    "main",
    "review_run",
    "run_one_macro",
    "run_trials",
    "sha256_file",
    "strict_json_load",
    "validate_fast_shutdown",
    "validate_macro_terminal",
    "validate_macro_worker_binding",
]
