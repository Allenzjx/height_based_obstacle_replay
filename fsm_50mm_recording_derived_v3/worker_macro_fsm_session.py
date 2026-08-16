"""Normal-development Macro FSM session owned by the production Isaac worker.

The session deliberately does not own a simulation loop.  It attaches to the
existing :class:`SimRobotAdapter` telemetry hook, which is invoked after every
120 Hz physics substep.  Feedback therefore selects the command for the next
physics tick while telemetry and the active viewport remain sampled at 15 Hz.

No playback plan or legacy 57-state controller participates in this path.
Recording-derived profiles are rebuilt locally from the Gate-B artifacts and
bound to the request by deterministic SHA-256 identities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES, clamp_servo_command
from completion_aware_segment import (
    CompletionAwareSegmentExecutor,
    SegmentCompletionSpec,
    SegmentDecisionKind,
    SegmentFeedback,
)
from telemetry.com_metrics import quat_wxyz_to_rpy

from .fsm50_direct_command_residual import (
    RESIDUAL_ACTION_NAMES,
    ZERO_RESIDUAL_ACTION,
    ResidualPhaseContract,
    ResidualTransformInput,
    compose_direct_command_residual,
)
from .support_classifier import (
    LEGS,
    ObstacleGeometry,
    WheelObservation,
    classify_wheel_contact,
)
from .viewport_buffer_video import ActiveViewportBufferVideoRecorder


REQUEST_SCHEMA = "fsm50.worker_macro_fsm_request.v1"
SESSION_SCHEMA = "fsm50.worker_macro_fsm_session.v1"
TELEMETRY_SCHEMA = "fsm50.minimal_macro_telemetry.v1"
TASK_INPUTS_SCHEMA = "fsm50.macro_task_inputs.v1"
TRANSITION_SCHEMA = "fsm50.macro_transition_evidence.v1"
DISPATCH_SCHEMA = "fsm50.macro_dispatch_ledger.v1"
SOURCE_ACTION_CONSUMPTION_SCHEMA = "fsm50.source_action_consumption.v1"
SEGMENT_COMPLETION_SCHEMA = "fsm50.macro_segment_completion.v1"
RESIDUAL_POLICY_OBSERVATION_SCHEMA = "fsm50.worker_residual_observation.v1"
SOURCE_ACTION_IDENTITY_SCHEMA_VERSION = "fsm50.source_action_identity.v1"
DEFAULT_TELEMETRY_HZ = 15.0
DEFAULT_VIDEO_FPS = 15.0
DEFAULT_POST_SETTLE_S = 0.5
EXPECTED_PHYSICS_DT_S = 1.0 / 120.0
EXPECTED_RENDER_SUBSTEPS = 8
EXPECTED_RENDER_DT_S = EXPECTED_PHYSICS_DT_S * EXPECTED_RENDER_SUBSTEPS
EXPECTED_SERVO_REFERENCE_VELOCITY_DEG_S = 150.0
CANONICAL_GATE_C_SOURCE_VERSION = "v003_20260805_224517_157723_manual"
AUTHORIZED_GATE_D_SOURCE_VERSIONS = (
    "v008_20260806_211408_578700_manual",
    "v009_20260806_215232_433234_manual",
    "v010_20260806_220745_363972_manual",
)
GATE_D_TRIAL_KIND = "cross_version"
CANONICAL_TASK_SUCCESS_TABLE_SHA256 = (
    "5549ee54b8e1aa17954c8d7dc0c9c88feee445ad32dbf6f58f6ccf900f45419c"
)
WHEEL_RADIUS_M = 0.04998999834060672
TARGET_COMMAND_TOLERANCE = 1.0e-6
DRIVE_READBACK_TOLERANCE = 2.0e-6
UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE = (
    "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
)

_COMMAND_PROVENANCE_KEYS = frozenset(
    {
        "kind",
        "source_action_identity",
        "source_version",
        "source_segment_index",
        "source_step_index",
        "source_time_s",
        "source_event_indices",
        "commands",
        "dispatch_kind",
        "sequence_index",
    }
)
_COMMAND_PROVENANCE_KINDS = frozenset(
    {
        "NONE",
        "SOURCE_ACTION",
        "BOUNDARY_ZERO_WHEELS",
        "HOLD_ZERO_WHEELS",
        "SAFE_STOP_ZERO_WHEELS",
        "SUCCESS_ZERO_WHEELS",
        "COMPLETION_WHEEL_STOP",
    }
)
_EMPTY_COMMAND_PROVENANCE = {
    "kind": "NONE",
    "source_action_identity": "",
    "source_version": "",
    "source_segment_index": None,
    "source_step_index": None,
    "source_time_s": None,
    "source_event_indices": [],
    "commands": [],
    "dispatch_kind": "",
    "sequence_index": None,
}

SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION = (
    "fsm50.macro_segment_completion_control.v1"
)
SEGMENT_COMPLETION_TOKEN_SCHEMA_VERSION = "fsm50.macro_segment_completion_token.v1"
_SEGMENT_COMPLETION_CONTROL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "profile_id",
        "profile_source_version",
        "owner_state",
        "source_plan_sha256",
        "source_plan_payload_sha256",
        "accepted_steps_sha256",
        "source_segment_index",
        "source_step_index",
        "source_step_id",
        "start_command_epoch",
        "completion_spec",
        "source_action_identity",
        "source_action",
        "completion_token_sha256",
    }
)
_EMPTY_SEGMENT_COMPLETION_CONTROL = {
    "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
    "kind": "NONE",
    "profile_id": "",
    "profile_source_version": "",
    "owner_state": "",
    "source_plan_sha256": "",
    "source_plan_payload_sha256": "",
    "accepted_steps_sha256": "",
    "source_segment_index": None,
    "source_step_index": None,
    "source_step_id": "",
    "start_command_epoch": None,
    "completion_spec": {},
    "source_action_identity": "",
    "source_action": False,
    "completion_token_sha256": "",
}

LEG_TO_WHEEL_BODY = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "enabled",
        "execution_mode",
        "request_id",
        "source_version",
        "profile_id",
        "graph_id",
        "graph_sha256",
        "profile_library_sha256",
        "bundle_sha256",
        "height_mm",
        "run_dir",
        "alignment_path",
        "alignment_sha256",
        "task_success_table_path",
        "task_success_table_sha256",
        "trial_kind",
        "trial_index",
        "telemetry_hz",
        "video_fps",
        "capture_video",
        "post_run_settle_s",
        "timeout_s",
        "filtered_contact_bank_enabled",
    }
)

_CSV_COLUMNS = (
    "sim_time_s",
    "sim_step",
    "macro_state",
    "subphase",
    "profile_id",
    "phase_elapsed_s",
    "profile_fraction",
    "command_epoch",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "base_roll_rad",
    "base_pitch_rad",
    "base_yaw_rad",
    "active_leg",
    "geometry_support_candidate_count",
    "robot_state_finite",
    "joint_limit_violation",
    "stability_state",
    "retry_count",
    "controller_terminal",
    "controller_terminal_outcome",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_action_identity_payload(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return the controller's eight-field, SHA-bound source-action identity."""

    return {
        "schema_version": SOURCE_ACTION_IDENTITY_SCHEMA_VERSION,
        "source_version": provenance.get("source_version"),
        "source_segment_index": provenance.get("source_segment_index"),
        "source_step_index": provenance.get("source_step_index"),
        "source_time_s": provenance.get("source_time_s"),
        "source_event_indices": provenance.get("source_event_indices"),
        "commands": provenance.get("commands"),
        "dispatch_kind": provenance.get("dispatch_kind"),
        "sequence_index": provenance.get("sequence_index"),
    }


def _source_action_identity(provenance: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(_source_action_identity_payload(provenance))


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return str(value.value)
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        return _jsonable(value.to_mapping())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped identity data without bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_json_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(
        _jsonable(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    _jsonable(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            base = dict(row.get("base_position_m", {}) or {})
            writer.writerow(
                {
                    "sim_time_s": row.get("sim_time_s"),
                    "sim_step": row.get("sim_step"),
                    "macro_state": row.get("macro_state", ""),
                    "subphase": row.get("subphase", ""),
                    "profile_id": row.get("profile_id", ""),
                    "phase_elapsed_s": row.get("phase_elapsed_s"),
                    "profile_fraction": row.get("profile_fraction"),
                    "command_epoch": row.get("command_epoch"),
                    "base_x_m": base.get("x"),
                    "base_y_m": base.get("y"),
                    "base_z_m": base.get("z"),
                    "base_roll_rad": row.get("base_roll_rad"),
                    "base_pitch_rad": row.get("base_pitch_rad"),
                    "base_yaw_rad": row.get("base_yaw_rad"),
                    "active_leg": row.get("active_leg", ""),
                    "geometry_support_candidate_count": row.get(
                        "geometry_support_candidate_count"
                    ),
                    "robot_state_finite": row.get("robot_state_finite"),
                    "joint_limit_violation": row.get("joint_limit_violation"),
                    "stability_state": row.get("stability_state", ""),
                    "retry_count": row.get("retry_count"),
                    "controller_terminal": row.get("controller_terminal"),
                    "controller_terminal_outcome": row.get(
                        "controller_terminal_outcome", ""
                    ),
                }
            )
    os.replace(temporary, path)


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_int(raw: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = raw.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _required_float(raw: Mapping[str, Any], key: str) -> float:
    value = raw.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _required_sha(raw: Mapping[str, Any], key: str) -> str:
    value = _required_text(raw, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class WorkerMacroFSMRequest:
    request_path: Path
    request_id: str
    source_version: str
    profile_id: str
    graph_id: str
    graph_sha256: str
    profile_library_sha256: str
    bundle_sha256: str
    height_mm: int
    run_dir: Path
    alignment_path: Path
    alignment_sha256: str
    task_success_table_path: Path
    task_success_table_sha256: str
    trial_kind: str
    trial_index: int
    telemetry_hz: float
    video_fps: float
    post_run_settle_s: float
    timeout_s: float
    capture_video: bool = True

    def preflight_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA,
            "enabled": True,
            "execution_mode": "normal_development",
            "request_id": self.request_id,
            "source_version": self.source_version,
            "profile_id": self.profile_id,
            "graph_id": self.graph_id,
            "graph_sha256": self.graph_sha256,
            "profile_library_sha256": self.profile_library_sha256,
            "bundle_sha256": self.bundle_sha256,
            "height_mm": self.height_mm,
            "run_dir": str(self.run_dir),
            "trial_kind": self.trial_kind,
            "trial_index": self.trial_index,
            "telemetry_hz": self.telemetry_hz,
            "video_fps": self.video_fps,
            "filtered_contact_bank_enabled": False,
            "preflight_ok": True,
        }


def load_worker_macro_fsm_request(
    request_path: str | Path | None,
) -> WorkerMacroFSMRequest | None:
    text = str(request_path or "").strip()
    if not text:
        return None
    path = Path(text).resolve()
    raw = _strict_json(path)
    missing = sorted(_REQUIRED_KEYS - set(raw))
    unexpected = sorted(set(raw) - _REQUIRED_KEYS)
    if missing:
        raise ValueError("macro FSM request is missing keys: " + ", ".join(missing))
    if unexpected:
        raise ValueError("macro FSM request has unexpected keys: " + ", ".join(unexpected))
    if raw.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unsupported macro FSM request schema")
    if raw.get("enabled") is not True:
        raise ValueError("macro FSM request must set enabled=true")
    if raw.get("execution_mode") != "normal_development":
        raise ValueError("macro FSM request must use normal_development mode")
    if raw.get("capture_video") is not True:
        raise ValueError("normal Macro FSM execution requires viewport video")
    if raw.get("filtered_contact_bank_enabled") is not False:
        raise ValueError("normal Macro FSM execution forbids the filtered contact bank")

    alignment_path = Path(_required_text(raw, "alignment_path")).resolve()
    table_path = Path(_required_text(raw, "task_success_table_path")).resolve()
    for label, source, sha_key in (
        ("alignment", alignment_path, "alignment_sha256"),
        ("task success table", table_path, "task_success_table_sha256"),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        claimed = _required_sha(raw, sha_key)
        if _sha256_file(source).lower() != claimed:
            raise ValueError(f"{label} SHA-256 does not match the request")

    telemetry_hz = _required_float(raw, "telemetry_hz")
    video_fps = _required_float(raw, "video_fps")
    settle_s = _required_float(raw, "post_run_settle_s")
    timeout_s = _required_float(raw, "timeout_s")
    if not 10.0 <= telemetry_hz <= 30.0:
        raise ValueError("macro telemetry_hz must remain in the minimal 10..30 Hz range")
    if not math.isclose(video_fps, DEFAULT_VIDEO_FPS, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("macro viewport video must use 15 fps")
    if settle_s <= 0.0 or timeout_s <= 0.0:
        raise ValueError("macro settle and timeout values must be positive")
    trial_kind = _required_text(raw, "trial_kind")
    height_mm = _required_int(raw, "height_mm", minimum=1)
    if height_mm != 50:
        raise ValueError("recording-derived Macro FSM request height_mm must equal 50")

    source_version = _required_text(raw, "source_version")
    if source_version == CANONICAL_GATE_C_SOURCE_VERSION:
        if trial_kind not in {"baseline", "repeat"}:
            raise ValueError(
                "canonical Gate-C v003 requires trial_kind baseline or repeat"
            )
    elif source_version in AUTHORIZED_GATE_D_SOURCE_VERSIONS:
        if trial_kind != GATE_D_TRIAL_KIND:
            raise ValueError(
                "authorized Gate-D sources require trial_kind cross_version"
            )
    else:
        raise ValueError(
            "Macro FSM source is not authorized for Gate-C or Gate-D execution"
        )

    return WorkerMacroFSMRequest(
        request_path=path,
        request_id=_required_text(raw, "request_id"),
        source_version=source_version,
        profile_id=_required_text(raw, "profile_id"),
        graph_id=_required_text(raw, "graph_id"),
        graph_sha256=_required_sha(raw, "graph_sha256"),
        profile_library_sha256=_required_sha(raw, "profile_library_sha256"),
        bundle_sha256=_required_sha(raw, "bundle_sha256"),
        height_mm=height_mm,
        run_dir=Path(_required_text(raw, "run_dir")).resolve(),
        alignment_path=alignment_path,
        alignment_sha256=_required_sha(raw, "alignment_sha256"),
        task_success_table_path=table_path,
        task_success_table_sha256=_required_sha(raw, "task_success_table_sha256"),
        trial_kind=trial_kind,
        trial_index=_required_int(raw, "trial_index", minimum=0),
        telemetry_hz=telemetry_hz,
        video_fps=video_fps,
        post_run_settle_s=settle_s,
        timeout_s=timeout_s,
        capture_video=True,
    )


def validate_worker_macro_start_binding(
    request: WorkerMacroFSMRequest,
    message: Mapping[str, Any],
    *,
    expected_worker_session_id: str,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "request_id": request.request_id,
        "worker_session_id": str(expected_worker_session_id),
        "source_version": request.source_version,
        "profile_id": request.profile_id,
        "graph_id": request.graph_id,
        "graph_sha256": request.graph_sha256,
        "profile_library_sha256": request.profile_library_sha256,
        "bundle_sha256": request.bundle_sha256,
    }
    for key, value in expected.items():
        if message.get(key) != value:
            errors.append(f"macro start {key} does not match the worker request")
    return errors


def _default_bundle_builder(project_root: Path, request: WorkerMacroFSMRequest) -> Any:
    from .fsm50_macro_controller import build_gate_c_bundle

    canonical_success_table = (
        Path(project_root).resolve()
        / "fsm_50mm_recording_derived_v3"
        / "reports"
        / "50MM_REPLAY_TASK_SUCCESS_TABLE.csv"
    ).resolve()
    if request.task_success_table_path != canonical_success_table:
        raise RuntimeError(
            "Macro request task-success table is not the canonical Gate-A report"
        )
    if request.task_success_table_sha256 != CANONICAL_TASK_SUCCESS_TABLE_SHA256:
        raise RuntimeError(
            "Macro request task-success table SHA is not the sealed Gate-A report SHA"
        )
    for label, path, expected_sha in (
        ("Gate-B alignment", request.alignment_path, request.alignment_sha256),
        (
            "Gate-A task-success table",
            request.task_success_table_path,
            request.task_success_table_sha256,
        ),
    ):
        if not path.is_file() or _sha256_file(path).lower() != expected_sha:
            raise RuntimeError(f"{label} changed after Macro request admission")
    return build_gate_c_bundle(
        project_root,
        alignment_path=request.alignment_path,
        primary_source_version=request.source_version,
    )


def _default_controller_factory(bundle: Any) -> Any:
    from .fsm50_macro_controller import MacroFSMController

    return MacroFSMController.from_bundle(bundle)


def _default_observation_factory(payload: Mapping[str, Any]) -> Any:
    from .fsm50_macro_controller import MacroObservation

    return MacroObservation.from_mapping(payload)


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return "" if raw is None else str(raw)


def _is_task_success_outcome(value: Any) -> bool:
    return _enum_text(value).upper() in {
        "SUCCESS",
        "SUCCEEDED",
        "TASK_SUCCESS",
        "TASK_SUCCESS_POSTURE_INCOMPLETE",
    }


def _exact_single_vector(value: Any, length: int) -> tuple[np.ndarray, bool]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float), False
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    array = array.reshape(-1)
    return array, bool(array.size == length)


def _new_traversal_record() -> dict[str, Any]:
    return {
        "airborne_seen_before_crossing": False,
        "airborne_first_s": None,
        "front_face_crossing_s": None,
        "top_seen": False,
        "top_first_s": None,
        "illegal_drive_up": False,
    }


def _default_safety_evidence_factory(
    adapter: Any, scene_handle: Any
) -> Mapping[str, Any]:
    """Prove only the clean initial boundary from existing live geometry."""

    capture = getattr(adapter, "capture_macro_safety_evidence", None)
    if callable(capture):
        try:
            raw = capture(scene_handle=scene_handle)
        except Exception as exc:
            return {
                "available": False,
                "dangerous_body_collision": None,
                "severe_penetration": None,
                "source": "capture_macro_safety_evidence",
                "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return dict(raw or {}) if isinstance(raw, Mapping) else {
            "available": False,
            "dangerous_body_collision": None,
            "severe_penetration": None,
            "source": "capture_macro_safety_evidence",
            "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
            "error": "capture_macro_safety_evidence did not return a mapping",
        }
    try:
        from sim_obstacle_scene import measure_scene_baseline

        geometry = dict(measure_scene_baseline(scene_handle, adapter) or {})
        ground = dict(
            adapter.validate_robot_ground_contact(apply_correction=False) or {}
        )
    except Exception as exc:
        return {
            "available": False,
            "dangerous_body_collision": None,
            "severe_penetration": None,
            "source": "LIVE_INITIAL_SCENE_GEOMETRY",
            "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
            "error": f"{type(exc).__name__}: {exc}",
        }
    errors: list[str] = []

    def finite_vector(value: Any, length: int, label: str) -> list[float]:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            errors.append(f"{label} must contain {length} values")
            return []
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError):
            errors.append(f"{label} is not numeric")
            return []
        if not all(math.isfinite(item) for item in result):
            errors.append(f"{label} contains non-finite values")
            return []
        return result

    obstacle_min = finite_vector(geometry.get("obstacle_bounds_min_m"), 3, "obstacle bounds min")
    obstacle_max = finite_vector(geometry.get("obstacle_bounds_max_m"), 3, "obstacle bounds max")
    robot_min = finite_vector(geometry.get("robot_collision_bounds_min_m"), 3, "robot bounds min")
    robot_max = finite_vector(geometry.get("robot_collision_bounds_max_m"), 3, "robot bounds max")
    root_pose = finite_vector(geometry.get("robot_root_pose"), 7, "robot root pose")
    wheels = list(geometry.get("wheel_collision_centers", []) or [])
    if geometry.get("available") is not True:
        errors.append("live scene baseline geometry is unavailable")
    if len(wheels) != len(LEGS):
        errors.append("live scene geometry must resolve exactly four wheel colliders")
    for index, row in enumerate(wheels):
        if not isinstance(row, Mapping):
            errors.append(f"wheel collider {index} is not a mapping")
            continue
        finite_vector(row.get("center_m"), 3, f"wheel collider {index} center")
        finite_vector(row.get("bounds_min_m"), 3, f"wheel collider {index} min")
        finite_vector(row.get("bounds_max_m"), 3, f"wheel collider {index} max")
        radius = row.get("radius_m")
        if type(radius) not in (int, float) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
            errors.append(f"wheel collider {index} radius is invalid")
    clearance = geometry.get("robot_collision_front_to_obstacle_front_m")
    if type(clearance) not in (int, float) or not math.isfinite(float(clearance)):
        errors.append("robot/obstacle front-face clearance is unavailable")
    elif float(clearance) <= 0.0:
        errors.append("robot collision bounds overlap the obstacle at initial boundary")
    maximum_penetration = ground.get("maximum_collision_penetration_m")
    raw_tolerance = getattr(
        getattr(adapter, "config", None),
        "ground_penetration_tolerance_m",
        None,
    )
    if (
        type(raw_tolerance) not in (int, float)
        or not math.isfinite(float(raw_tolerance))
        or float(raw_tolerance) <= 0.0
    ):
        errors.append("deployed ground penetration tolerance is invalid")
        tolerance = float("nan")
    else:
        tolerance = float(raw_tolerance)
    if type(maximum_penetration) not in (int, float) or not math.isfinite(
        float(maximum_penetration)
    ):
        errors.append("initial ground penetration evidence is unavailable")
    elif not math.isfinite(tolerance) or float(maximum_penetration) > tolerance:
        errors.append("initial collision penetration exceeds the deployed tolerance")
    return {
        "available": not errors,
        "dangerous_body_collision": False if not errors else None,
        "severe_penetration": False if not errors else None,
        "source": "LIVE_INITIAL_SCENE_GEOMETRY_AND_GROUND_DIAGNOSTICS",
        "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
        "geometry": geometry,
        "ground_diagnostics": ground,
        "robot_root_pose": root_pose,
        "obstacle_bounds_min_m": obstacle_min,
        "obstacle_bounds_max_m": obstacle_max,
        "robot_collision_bounds_min_m": robot_min,
        "robot_collision_bounds_max_m": robot_max,
        "initial_robot_to_obstacle_clearance_m": clearance,
        "initial_maximum_ground_penetration_m": maximum_penetration,
        "ground_penetration_tolerance_m": tolerance,
        "ground_penetration_tolerance_source": (
            "adapter.config.ground_penetration_tolerance_m"
        ),
        "error": "; ".join(errors),
    }


class WorkerMacroFSMSession:
    """Execute one feedback Macro FSM inside the existing production worker."""

    def __init__(
        self,
        request: WorkerMacroFSMRequest,
        *,
        worker_session_id: str,
        bundle_builder: Callable[[Path, WorkerMacroFSMRequest], Any] = _default_bundle_builder,
        controller_factory: Callable[[Any], Any] = _default_controller_factory,
        observation_factory: Callable[[Mapping[str, Any]], Any] = _default_observation_factory,
        recorder_factory: Callable[..., Any] = ActiveViewportBufferVideoRecorder,
        safety_evidence_factory: Callable[[Any, Any], Mapping[str, Any]] = (
            _default_safety_evidence_factory
        ),
        residual_policy: Any | None = None,
        residual_contract_provider: (
            Callable[[Mapping[str, Any]], ResidualPhaseContract] | None
        ) = None,
    ) -> None:
        self.request = request
        self.worker_session_id = str(worker_session_id)
        self.bundle_builder = bundle_builder
        self.controller_factory = controller_factory
        self.observation_factory = observation_factory
        self.recorder_factory = recorder_factory
        self.safety_evidence_factory = safety_evidence_factory
        if (residual_policy is None) != (residual_contract_provider is None):
            raise ValueError(
                "residual_policy and residual_contract_provider must be supplied together"
            )
        if residual_policy is not None and not callable(
            getattr(residual_policy, "act", None)
        ):
            raise ValueError("residual_policy.act must be callable")
        if residual_contract_provider is not None and not callable(
            residual_contract_provider
        ):
            raise ValueError("residual_contract_provider must be callable")
        self.residual_policy = residual_policy
        self.residual_contract_provider = residual_contract_provider
        self.residual_enabled = residual_policy is not None
        self.residual_policy_id = ""
        self.residual_policy_sha256 = ""
        if self.residual_enabled:
            policy_id = getattr(residual_policy, "policy_id", None)
            policy_sha256 = getattr(residual_policy, "policy_sha256", None)
            if type(policy_id) is not str or not policy_id:
                raise ValueError("residual_policy.policy_id must be non-empty text")
            if (
                type(policy_sha256) is not str
                or len(policy_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in policy_sha256
                )
            ):
                raise ValueError(
                    "residual_policy.policy_sha256 must be a lowercase SHA-256"
                )
            self.residual_policy_id = policy_id
            self.residual_policy_sha256 = policy_sha256
        self.state = "created"
        self.error = ""
        self.infrastructure_failure = False
        self.simulation_app_stopped = False
        self.adapter: Any | None = None
        self.scene_handle: Any | None = None
        self.bundle: Any | None = None
        self.controller: Any | None = None
        self.recorder: Any | None = None
        self.observer_attached = False
        self.telemetry_attached = False
        self.video: dict[str, Any] = {}
        self.video_writer_quiesced = False
        self.started_sim_time_s: float | None = None
        self.completed_sim_time_s: float | None = None
        self.boundary_first_physics_step: int | None = None
        self.physics_dt_s: float | None = None
        self.boundary_ack: dict[str, Any] = {}
        self.bundle_identity: dict[str, Any] = {}
        self.target_audit: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.transition_rows: list[dict[str, Any]] = []
        self.dispatch_rows: list[dict[str, Any]] = []
        self.source_action_consumption_rows: list[dict[str, Any]] = []
        self.segment_completion_rows: list[dict[str, Any]] = []
        self.expected_source_actions: list[dict[str, Any]] = []
        self.next_source_action_index = 0
        self.next_sample_sim_time_s = 0.0
        self.last_decision: Any | None = None
        self.last_decision_mapping: dict[str, Any] = {}
        self.last_epoch: int | None = None
        self.last_servo_targets: dict[str, float] | None = None
        self.last_wheel_targets: dict[str, float] | None = None
        self.last_applied_servo_targets: dict[str, float] | None = None
        self.last_applied_wheel_targets: dict[str, float] | None = None
        self.last_applied_residual = ZERO_RESIDUAL_ACTION
        self.physical_command_epoch = 0
        self.last_verified_physical_command_epoch: int | None = None
        self.last_residual_transform: dict[str, Any] = {}
        self.residual_transform_count = 0
        self.active_completion_latched_servo_residual_deg: dict[str, float] = {}
        self.last_macro_state = ""
        self.pending_readback: dict[str, Any] | None = None
        self.last_verified_servo_targets: dict[str, float] | None = None
        self.last_verified_wheel_targets: dict[str, float] | None = None
        self.last_verified_command_epoch: int | None = None
        self.last_target_readback: dict[str, Any] = {}
        self.boundary_readback_verified = False
        self.segment_completion_executor = CompletionAwareSegmentExecutor()
        self.active_segment_completion_row_index: int | None = None
        self.last_segment_completion_token: Any | None = None
        self.outer_render_cycle_index = -1
        self.outer_render_substeps_remaining = 0
        self.outer_render_cycle_start_sim_step: int | None = None
        self.outer_render_boundary_permit = False
        self.last_completion_observation_sim_step: int | None = None
        self.terminal_stop_request: dict[str, Any] | None = None
        self.safe_stop_status = "NOT_REQUESTED"
        self.safe_stop_verified = False
        self.safe_stop_error = ""
        self.safe_stop_ack: dict[str, Any] = {}
        self.safe_stop_readback: dict[str, Any] = {}
        self.safe_stop_readback_sha256 = ""
        self.safe_stop_count = 0
        self.deployment_safety_evidence: dict[str, Any] = {}
        self.dangerous_collision_detected = False
        self.severe_penetration_detected = False
        self.body_stuck_detected = False
        self.active_leg_trapped_detected = False
        self.optional_runtime_evidence = {
            name: {
                "sample_count": 0,
                "available_sample_count": 0,
                "source": "",
            }
            for name in ("body_stuck", "active_leg_trapped")
        }
        self.controller_terminal_outcome = ""
        self.controller_terminal_reason = ""
        self.controller_tick_count = 0
        self.command_dispatch_count = 0
        self.last_batch_attempt_sim_step: int | None = None
        self.peak_roll_rad = 0.0
        self.peak_pitch_rad = 0.0
        self.nonfinite_state_detected = False
        self.joint_limit_violation_detected = False
        self.traversal = {leg: _new_traversal_record() for leg in LEGS}
        # Phase-local lift history is authoritative for illegal drive-up.  A
        # startup AIR transient must never authorize a much later traverse.
        # Each macro-state entry resets the record, including consecutive
        # states with the same leg.  A lift is seeded only if it remains live
        # in the actual state-entry observation; AIR that returned to support
        # before the traverse cannot authorize a later crossing.
        self.phase_traversal = {leg: _new_traversal_record() for leg in LEGS}
        self.active_traversal_leg = ""
        self.active_traversal_state = ""
        self.terminal_payload: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"complete", "failed"}

    @property
    def fast_close_ready(self) -> bool:
        return bool(
            self.terminal
            and self.video_writer_quiesced
            and self.terminal_payload
            and Path(str(self.terminal_payload.get("worker_result_path", "") or "")).is_file()
        )

    def status_dict(self) -> dict[str, Any]:
        status = {
            "schema_version": SESSION_SCHEMA,
            "enabled": True,
            "execution_mode": "normal_development",
            "request_id": self.request.request_id,
            "source_version": self.request.source_version,
            "profile_id": self.request.profile_id,
            "bundle_sha256": self.request.bundle_sha256,
            "state": self.state,
            "terminal": self.terminal,
            "fast_close_ready": self.fast_close_ready,
            "video_writer_quiesced": self.video_writer_quiesced,
            "controller_tick_count": self.controller_tick_count,
            "command_dispatch_count": self.command_dispatch_count,
            "expected_source_action_count": len(self.expected_source_actions),
            "source_action_consumption_count": len(
                self.source_action_consumption_rows
            ),
            "segment_completion_count": sum(
                row.get("terminal_kind") == SegmentDecisionKind.COMPLETE.value
                for row in self.segment_completion_rows
            ),
            "active_segment_completion_row_index": (
                self.active_segment_completion_row_index
            ),
            "outer_render_cycle_index": self.outer_render_cycle_index,
            "outer_render_substeps_remaining": (
                self.outer_render_substeps_remaining
            ),
            "pending_readback": _jsonable(self.pending_readback),
            "last_verified_command_epoch": self.last_verified_command_epoch,
            "safe_stop_status": self.safe_stop_status,
            "safe_stop_verified": self.safe_stop_verified,
            "safe_stop_error": self.safe_stop_error,
            "deployment_safety_evidence": _jsonable(
                self.deployment_safety_evidence
            ),
            "telemetry_sample_count": len(self.rows),
            "physics_dt_s": self.physics_dt_s,
            "filtered_contact_bank_enabled": False,
            "error": self.error,
        }
        if self.residual_enabled:
            status["direct_command_residual"] = {
                "enabled": True,
                "policy_id": self.residual_policy_id,
                "policy_sha256": self.residual_policy_sha256,
                "transform_count": self.residual_transform_count,
                "physical_command_epoch": self.physical_command_epoch,
                "last_verified_physical_command_epoch": (
                    self.last_verified_physical_command_epoch
                ),
                "last_applied_servo_targets_deg": dict(
                    self.last_applied_servo_targets or {}
                ),
                "last_applied_wheel_targets_rad_s": dict(
                    self.last_applied_wheel_targets or {}
                ),
                "last_transform_sha256": str(
                    self.last_residual_transform.get("transform_sha256", "")
                ),
            }
        return status

    def prepare_after_adapter(
        self,
        *,
        adapter: Any,
        scene_handle: Any,
        project_root: Path,
    ) -> None:
        if self.state != "created":
            raise RuntimeError(f"macro session prepare in state={self.state}")
        if getattr(adapter, "telemetry_collector", None) is not None:
            raise RuntimeError("Macro FSM found another telemetry collector")
        if getattr(adapter, "artifact_render_observer", None) is not None:
            raise RuntimeError("Macro FSM found another viewport observer")
        self.adapter = adapter
        self.scene_handle = scene_handle
        try:
            physics_dt_s = float(adapter.sim.get_physics_dt())
        except Exception as exc:
            raise RuntimeError("production adapter physics dt is unavailable") from exc
        if not math.isfinite(physics_dt_s) or not math.isclose(
            physics_dt_s,
            EXPECTED_PHYSICS_DT_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "Macro FSM requires physics dt=1/120 s; "
                f"actual={physics_dt_s!r}"
            )
        self.physics_dt_s = physics_dt_s
        self._validate_render_cadence(adapter)
        self.deployment_safety_evidence = self._capture_deployment_safety_evidence(
            adapter
        )
        self._validate_deployment_safety_evidence(
            self.deployment_safety_evidence,
            expected_sim_step=int(getattr(adapter, "sim_steps", 0) or 0),
            require_clear=True,
        )
        self.bundle = self.bundle_builder(Path(project_root).resolve(), self.request)
        self.bundle_identity = self._validate_bundle_identity(self.bundle)
        self.target_audit = self._audit_bundle_targets(self.bundle, adapter)
        if self.target_audit.get("available") is not True:
            raise RuntimeError(
                "Macro profile target audit unavailable: "
                + "; ".join(self.target_audit.get("errors", []) or [])
            )
        if self.target_audit.get("unsafe") is True:
            raise RuntimeError(
                "Macro profile target audit found unsafe targets: "
                + json.dumps(
                    self.target_audit.get("violations", []),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        self.expected_source_actions = self._build_expected_source_actions(
            self.bundle
        )
        self.next_source_action_index = 0
        self.controller = self.controller_factory(self.bundle)
        self.state = "ready_for_start"

    def start(self) -> dict[str, Any]:
        if self.state != "ready_for_start":
            raise RuntimeError(f"Macro FSM is not ready for start: state={self.state}")
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Macro FSM adapter is unavailable")
        if getattr(adapter, "telemetry_collector", None) is not None:
            raise RuntimeError("Macro FSM found another telemetry collector at start")
        if getattr(adapter, "artifact_render_observer", None) is not None:
            raise RuntimeError("Macro FSM found another viewport observer at start")

        initial_payload = self._observation_payload(adapter)
        initial_servos = self._validated_target_map(
            getattr(adapter, "joint_command_deg", None),
            names=SERVO_JOINT_NAMES,
            label="pre-start adapter servo targets",
        )
        initial_wheels = self._validated_target_map(
            getattr(adapter, "wheel_speeds", None),
            names=WHEEL_JOINT_NAMES,
            label="pre-start adapter wheel targets",
        )
        self.last_target_readback = self._capture_and_validate_target_readback(
            adapter,
            servo_targets=initial_servos,
            wheel_targets=initial_wheels,
            expected_sim_step=int(getattr(adapter, "sim_steps", 0) or 0),
        )
        self.last_verified_servo_targets = dict(initial_servos)
        self.last_verified_wheel_targets = dict(initial_wheels)
        initial_payload["actuator_targets_applied"] = True
        initial_payload["actuator_target_source"] = "PHYSX_DRIVE_TARGET_READBACK"
        initial_payload["actuator_target_readback"] = self.last_target_readback
        self._attach_drive_readback_layers(initial_payload)
        self._validate_hard_safety(initial_payload, startup=True)
        self.request.run_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = self.recorder_factory(
            self.request.run_dir,
            enabled=True,
            fps=self.request.video_fps,
        )
        if self.recorder.start() is not True or str(getattr(self.recorder, "error", "") or ""):
            raise RuntimeError(
                "viewport recorder did not start: "
                + str(getattr(self.recorder, "error", "") or "unknown error")
            )
        adapter.attach_artifact_render_observer(self.recorder)
        self.observer_attached = True
        adapter.attach_telemetry(self)
        self.telemetry_attached = True

        servos = dict(initial_servos)
        wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        batch_id = f"{self.request.request_id}:start-boundary"
        self.last_batch_attempt_sim_step = int(
            getattr(adapter, "sim_steps", 0) or 0
        )
        ack = dict(
            adapter.apply_motion_batch(
                {
                    "batch_id": batch_id,
                    "source": "fsm50_macro_start_boundary",
                    "servo_targets_deg": servos,
                    "wheel_targets_rad_s": wheels,
                    "wheel_generation": int(getattr(adapter, "wheel_generation", 0) or 0),
                }
            )
            or {}
        )
        self._validate_batch_ack(
            ack,
            batch_id=batch_id,
            servo_targets=servos,
            wheel_targets=wheels,
            expected_sim_step=int(getattr(adapter, "sim_steps", 0) or 0),
            expected_physics_dt_s=float(self.physics_dt_s),
            expected_source="fsm50_macro_start_boundary",
            expected_recording_metadata={},
        )
        self.last_applied_servo_targets = dict(servos)
        self.last_applied_wheel_targets = dict(wheels)
        self.last_applied_residual = ZERO_RESIDUAL_ACTION
        self.physical_command_epoch = 0
        self.last_verified_physical_command_epoch = None
        self._establish_pending_readback(
            kind="boundary",
            batch_id=batch_id,
            ack=ack,
            servo_targets=servos,
            wheel_targets=wheels,
            command_epoch=0,
            physical_command_epoch=(0 if self.residual_enabled else None),
        )
        self.boundary_ack = _jsonable(ack)
        self.boundary_first_physics_step = int(ack["first_physics_step"])
        self.next_sample_sim_time_s = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        self.state = "boundary_pending"
        self._sample(initial_payload, decision=None, force=True)
        return {
            "accepted": True,
            "request_id": self.request.request_id,
            "source_version": self.request.source_version,
            "profile_id": self.request.profile_id,
            **self.bundle_identity,
            "physics_dt_s": self.physics_dt_s,
            "expected_source_action_count": len(self.expected_source_actions),
            "start_boundary_ack": self.boundary_ack,
            "controller_reset_pending": True,
            "first_controller_tick_physics_step": self.boundary_first_physics_step,
            # reset() observes the completed zero-wheel boundary at N+1.  Hard
            # safety and physical guards remain 120 Hz, while a source action
            # is permitted only on the eighth/final physics substep of this
            # production render cycle.
            "earliest_profile_dispatch_physics_step": (
                self.boundary_first_physics_step
                + EXPECTED_RENDER_SUBSTEPS
                - 1
            ),
            "earliest_profile_actuation_physics_step": (
                self.boundary_first_physics_step + EXPECTED_RENDER_SUBSTEPS
            ),
            "error": "",
        }

    @staticmethod
    def _validate_render_cadence(adapter: Any) -> None:
        timing = getattr(adapter, "_render_step_timing", None)
        if not callable(timing):
            raise RuntimeError(
                "Macro FSM requires the production adapter render-step timing"
            )
        try:
            elapsed_s, substeps = timing()
            elapsed_s = float(elapsed_s)
        except Exception as exc:
            raise RuntimeError(
                "Macro FSM render-step timing is unavailable"
            ) from exc
        if (
            type(substeps) is not int
            or substeps != EXPECTED_RENDER_SUBSTEPS
            or not math.isfinite(elapsed_s)
            or not math.isclose(
                elapsed_s,
                EXPECTED_RENDER_DT_S,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise RuntimeError(
                "Macro FSM requires production render_interval=8 at 120 Hz; "
                f"actual_substeps={substeps!r} actual_elapsed_s={elapsed_s!r}"
            )

    def _consume_outer_render_substep(self, *, sim_step: int) -> bool:
        if self.outer_render_substeps_remaining <= 0:
            raise RuntimeError(
                "Macro physics callback arrived outside before_adapter_step() cadence"
            )
        start_step = self.outer_render_cycle_start_sim_step
        if type(start_step) is not int:
            raise RuntimeError("Macro render-cycle start identity is unavailable")
        completed = EXPECTED_RENDER_SUBSTEPS - self.outer_render_substeps_remaining + 1
        expected_step = start_step + completed
        if sim_step != expected_step:
            raise RuntimeError(
                "Macro render cycle lost or duplicated a physics callback: "
                f"expected={expected_step} actual={sim_step}"
            )
        boundary = self.outer_render_substeps_remaining == 1
        self.outer_render_substeps_remaining -= 1
        return boundary

    def on_step(self, adapter: Any, _physics_dt: float) -> None:
        """Run feedback at the existing 120 Hz post-physics collector hook."""

        if self.state not in {
            "boundary_pending",
            "running",
            "terminal_command_pending_readback",
            "safe_stop_pending_readback",
            "settling",
        }:
            return
        try:
            actual_dt = float(_physics_dt)
            if self.physics_dt_s is None or not math.isclose(
                actual_dt,
                self.physics_dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    "physics dt drift during Macro FSM callback: "
                    f"expected={self.physics_dt_s!r} actual={actual_dt!r}"
                )
            root_writes = getattr(adapter, "root_state_write_count", None)
            if type(root_writes) is not int or root_writes != 0:
                raise RuntimeError("normal Macro FSM observed root_state_write_count drift")
            sim_step = int(getattr(adapter, "sim_steps", -1))
            source_cursor_permit = self._consume_outer_render_substep(
                sim_step=sim_step
            )
            self.outer_render_boundary_permit = source_cursor_permit
            completed_readback_kind = ""
            if self.pending_readback is not None:
                try:
                    completed_readback_kind = self._verify_pending_readback(
                        adapter, sim_step=sim_step
                    )
                except Exception as readback_exc:
                    if str(self.pending_readback.get("kind", "")) == "safe_stop":
                        self._fail_safe_stop_readback(readback_exc)
                    else:
                        self._request_failure(
                            f"{type(readback_exc).__name__}: {readback_exc}",
                            infrastructure_failure=True,
                        )
                    return
            elif (
                self.last_verified_servo_targets is not None
                and self.last_verified_wheel_targets is not None
            ):
                self.last_target_readback = self._capture_and_validate_target_readback(
                    adapter,
                    servo_targets=self.last_verified_servo_targets,
                    wheel_targets=self.last_verified_wheel_targets,
                    expected_sim_step=sim_step,
                )
            payload = self._observation_payload(adapter)
            payload["actuator_targets_applied"] = True
            payload["actuator_target_source"] = "PHYSX_DRIVE_TARGET_READBACK"
            payload["actuator_target_readback"] = self.last_target_readback
            self._attach_drive_readback_layers(payload)
            self._validate_hard_safety(payload)
            self._sample(payload, decision=self.last_decision)
            if completed_readback_kind == "safe_stop":
                stop_request = dict(self.terminal_stop_request or {})
                self.terminal_stop_request = None
                if stop_request.get("success_after_stop") is True:
                    self.completed_sim_time_s = float(payload["sim_time_s"])
                    self.state = "settling"
                else:
                    self.state = "failed_pending_finalize"
                return
            if self.terminal_stop_request is not None:
                self._begin_safe_stop(adapter, payload)
                return
            if self.state == "settling":
                return
            sim_time = float(payload["sim_time_s"])
            observation = self.observation_factory(payload)
            if self.state == "boundary_pending":
                if (
                    not self.boundary_readback_verified
                    or self.boundary_first_physics_step is None
                    or sim_step != self.boundary_first_physics_step
                ):
                    raise RuntimeError(
                        "controller reset requires exact verified N+1 zero boundary"
                    )
                self.started_sim_time_s = sim_time
                decision = self.controller.reset(
                    observation,
                    sim_time_s=sim_time,
                    profile_id=self.request.profile_id,
                    source_version=self.request.source_version,
                )
                self.state = "running"
            else:
                if (
                    self.started_sim_time_s is not None
                    and sim_time - self.started_sim_time_s > self.request.timeout_s
                ):
                    raise RuntimeError("Macro FSM session timed out")
                completion_token = (
                    self._observe_active_segment_completion(
                        adapter=adapter,
                        payload=payload,
                    )
                    if source_cursor_permit
                    else None
                )
                self.last_segment_completion_token = completion_token
                decision = self.controller.tick(
                    observation,
                    sim_time_s=sim_time,
                    segment_completion_token=completion_token,
                    source_cursor_permit=source_cursor_permit,
                )
            self.controller_tick_count += 1
            self._process_decision(adapter, payload, decision)
        except Exception as exc:
            if self.pending_readback is not None and str(
                self.pending_readback.get("kind", "")
            ) == "safe_stop":
                self._fail_safe_stop_readback(exc)
            else:
                self._request_failure(
                    f"{type(exc).__name__}: {exc}",
                    infrastructure_failure=(
                        isinstance(exc, (AttributeError, TypeError))
                        or "physics dt" in str(exc).lower()
                        or "readback" in str(exc).lower()
                        or "root_state_write_count" in str(exc)
                    ),
                )

    def before_adapter_step(self) -> None:
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Macro render-cycle scheduling lacks its adapter")
        self._validate_render_cadence(adapter)
        if self.state == "ready_for_start":
            # The production worker calls this hook around outer render steps
            # before the Macro start request is accepted.  No active callback
            # accounting exists yet: on_step() intentionally ignores those
            # pre-start physics callbacks, so they must not arm an eight-step
            # cycle that can never be consumed.
            self.outer_render_boundary_permit = False
            return
        if self.outer_render_substeps_remaining != 0:
            raise RuntimeError(
                "prior Macro outer render cycle did not deliver exactly eight "
                "physics callbacks"
            )
        start_step = int(getattr(adapter, "sim_steps", -1))
        if start_step < 0:
            raise RuntimeError("Macro render-cycle start sim step is invalid")
        self.outer_render_cycle_index += 1
        self.outer_render_cycle_start_sim_step = start_step
        self.outer_render_substeps_remaining = EXPECTED_RENDER_SUBSTEPS
        self.outer_render_boundary_permit = False

    def after_adapter_step(self) -> dict[str, Any] | None:
        if self.terminal_payload is not None:
            return None
        if self.state == "failed_pending_finalize":
            return self._finalize(success=False, error=self.error)
        if self.state == "settling" and self.completed_sim_time_s is not None:
            current = float(getattr(self.adapter, "sim_time", 0.0) or 0.0)
            if current - self.completed_sim_time_s + 1.0e-9 >= self.request.post_run_settle_s:
                return self._finalize(success=True, error="")
        return None

    def fail(
        self,
        error: str,
        *,
        infrastructure_failure: bool = False,
        simulation_app_stopped: bool = False,
    ) -> dict[str, Any] | None:
        if self.terminal_payload is not None:
            return dict(self.terminal_payload)
        self.infrastructure_failure = bool(self.infrastructure_failure or infrastructure_failure)
        self.simulation_app_stopped = bool(
            self.simulation_app_stopped or simulation_app_stopped
        )
        self._request_failure(str(error or "Macro FSM failed"), infrastructure_failure=infrastructure_failure)
        if simulation_app_stopped:
            if self.state == "safe_stop_pending_readback":
                self.safe_stop_status = "SAFE_STOP_READBACK_UNAVAILABLE"
                self.safe_stop_verified = False
                self.safe_stop_error = (
                    "simulation stopped before the required N+1 safe-stop readback"
                )
            elif (
                self.safe_stop_verified is not True
                and self.safe_stop_status
                not in {
                    "SAFE_STOP_APPLICATION_FAILED",
                    "SAFE_STOP_READBACK_FAILED",
                }
            ):
                self.safe_stop_status = (
                    "SAFE_STOP_NOT_APPLIED_SIMULATION_STOPPED"
                )
                self.safe_stop_verified = False
                self.safe_stop_error = (
                    "simulation stopped before the queued safe stop could be "
                    "applied or verified"
                )
            if self.safe_stop_verified is not True and self.safe_stop_error:
                suffix = "SAFE_STOP_UNVERIFIED: " + self.safe_stop_error
                if suffix not in self.error:
                    self.error = "; ".join(
                        part for part in (self.error, suffix) if part
                    )
            self.pending_readback = None
            self.terminal_stop_request = None
            self.state = "failed_pending_finalize"
        if self.state == "failed_pending_finalize":
            return self._finalize(success=False, error=self.error)
        return None

    def _validate_bundle_identity(self, bundle: Any) -> dict[str, Any]:
        graph = _attribute(bundle, "graph")
        profiles = _attribute(bundle, "profiles")
        graph_id = _enum_text(_attribute(graph, "graph_id"))
        graph_sha = _enum_text(
            _attribute(bundle, "graph_sha256", _attribute(graph, "sha256"))
        ).lower()
        profiles_sha = _enum_text(
            _attribute(bundle, "profile_library_sha256", _attribute(profiles, "sha256"))
        ).lower()
        bundle_sha = _enum_text(_attribute(bundle, "bundle_sha256")).lower()
        if not bundle_sha and hasattr(bundle, "to_mapping"):
            bundle_sha = _canonical_json_sha256(bundle.to_mapping())
        actual = {
            "graph_id": graph_id,
            "graph_sha256": graph_sha,
            "profile_library_sha256": profiles_sha,
            "bundle_sha256": bundle_sha,
        }
        expected = {
            "graph_id": self.request.graph_id,
            "graph_sha256": self.request.graph_sha256,
            "profile_library_sha256": self.request.profile_library_sha256,
            "bundle_sha256": self.request.bundle_sha256,
        }
        mismatches = [key for key in expected if actual.get(key) != expected[key]]
        if mismatches:
            raise RuntimeError("Macro bundle identity mismatch: " + ", ".join(mismatches))
        return actual

    def _audit_bundle_targets(self, bundle: Any, adapter: Any) -> dict[str, Any]:
        # The bundle mapping intentionally contains identities only.  Audit the
        # locally rebuilt profile library itself so every recording-derived
        # keyframe is checked before the controller can start.
        profiles = _attribute(bundle, "profiles")
        mapping = _jsonable(
            profiles.to_mapping() if hasattr(profiles, "to_mapping") else profiles
        )
        target_sets: list[
            tuple[str, Mapping[str, Any], Mapping[str, Any], bool]
        ] = []

        def walk(value: Any, label: str) -> None:
            if isinstance(value, Mapping):
                servos = value.get("servo_targets_deg")
                wheels = value.get("wheel_targets_rad_s")
                if isinstance(servos, Mapping) or isinstance(wheels, Mapping):
                    # Completion specifications intentionally carry only the
                    # sparse servo endpoints for their source segment.  Every
                    # other profile target row remains a canonical full 8+4
                    # command and is audited as such below.
                    sparse_completion_spec = label.endswith(".completion_spec")
                    target_sets.append(
                        (
                            label,
                            dict(servos or {}),
                            dict(wheels or {}),
                            sparse_completion_spec,
                        )
                    )
                for key, item in value.items():
                    walk(item, f"{label}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{label}[{index}]")

        walk(mapping, "bundle")
        errors: list[str] = []
        violations: list[dict[str, Any]] = []
        command_to_actual = getattr(adapter, "command_to_actual_target_deg", None)
        final_limits = getattr(adapter, "get_final_target_limits_deg", None)
        max_wheel_speed = float(getattr(adapter, "max_wheel_speed", float("nan")))
        if not callable(command_to_actual) or not callable(final_limits):
            errors.append("production servo target transforms are unavailable")
        if not math.isfinite(max_wheel_speed) or max_wheel_speed <= 0.0:
            errors.append("production wheel target limit is unavailable")
        for label, servos, wheels, sparse_completion_spec in target_sets:
            unknown_servos = sorted(set(servos) - set(SERVO_JOINT_NAMES))
            unknown_wheels = sorted(set(wheels) - set(WHEEL_JOINT_NAMES))
            missing_servos = (
                []
                if sparse_completion_spec
                else sorted(set(SERVO_JOINT_NAMES) - set(servos))
            )
            missing_wheels = (
                []
                if sparse_completion_spec
                else sorted(set(WHEEL_JOINT_NAMES) - set(wheels))
            )
            if sparse_completion_spec and wheels:
                unknown_wheels = sorted(set(wheels))
            if unknown_servos or unknown_wheels or missing_servos or missing_wheels:
                violations.append(
                    {
                        "location": label,
                        "kind": "noncanonical_target_names",
                        "servo_names": unknown_servos,
                        "wheel_names": unknown_wheels,
                        "missing_servo_names": missing_servos,
                        "missing_wheel_names": missing_wheels,
                    }
                )
            for name, raw in servos.items():
                try:
                    target = float(raw)
                except (TypeError, ValueError):
                    target = float("nan")
                if not math.isfinite(target):
                    violations.append({"location": label, "joint": name, "kind": "nonfinite_servo"})
                    continue
                if name not in SERVO_JOINT_NAMES or not callable(command_to_actual) or not callable(final_limits):
                    continue
                clamped = float(clamp_servo_command(name, target))
                actual = float(command_to_actual(name, target))
                minimum, maximum = (float(item) for item in final_limits(name))
                if not math.isclose(clamped, target, rel_tol=0.0, abs_tol=1.0e-9) or not minimum <= actual <= maximum:
                    violations.append(
                        {
                            "location": label,
                            "joint": name,
                            "kind": "servo_target_would_clamp_or_exceed_safe_limit",
                            "target_deg": target,
                            "clamped_deg": clamped,
                            "actual_target_deg": actual,
                            "safe_actual_min_deg": minimum,
                            "safe_actual_max_deg": maximum,
                        }
                    )
            for name, raw in wheels.items():
                try:
                    target = float(raw)
                except (TypeError, ValueError):
                    target = float("nan")
                if not math.isfinite(target) or (
                    math.isfinite(max_wheel_speed)
                    and abs(target) > max_wheel_speed + 1.0e-9
                ):
                    violations.append(
                        {
                            "location": label,
                            "joint": name,
                            "kind": "unsafe_wheel_target",
                            "target_rad_s": None if not math.isfinite(target) else target,
                            "safe_abs_max_rad_s": max_wheel_speed,
                        }
                    )
        if not target_sets:
            errors.append("Macro bundle exposes no auditable target timelines")
        return {
            "available": not errors,
            "unsafe": True if violations else False if not errors else None,
            "source": "recording_derived_macro_bundle_target_audit",
            "target_set_count": len(target_sets),
            "errors": errors,
            "violations": violations,
        }

    def _build_expected_source_actions(self, bundle: Any) -> list[dict[str, Any]]:
        """Rebuild the canonical source-action sequence from the sealed bundle.

        Controller provenance is never accepted as self-authenticating.  The
        worker independently rebuilds every expected identity, owner, profile,
        plan binding, and full 8-servo/4-wheel target map before motion starts.
        """

        profiles = _attribute(bundle, "profiles")
        raw = profiles.to_mapping() if hasattr(profiles, "to_mapping") else profiles
        mapping = _jsonable(raw)
        if not isinstance(mapping, Mapping):
            raise RuntimeError("Macro profile library mapping is unavailable")
        profile_rows = mapping.get("profiles")
        ownership_rows = mapping.get("segment_ownership")
        if not isinstance(profile_rows, list) or not isinstance(ownership_rows, list):
            raise RuntimeError(
                "Macro profile library lacks canonical profiles/segment ownership"
            )

        source = self.request.source_version
        owned = [
            dict(row)
            for row in ownership_rows
            if isinstance(row, Mapping) and row.get("source_version") == source
        ]
        if not owned:
            raise RuntimeError("Macro source has no canonical segment ownership")
        for row in owned:
            for name in ("first_segment", "last_segment"):
                if type(row.get(name)) is not int or int(row[name]) < 0:
                    raise RuntimeError(f"canonical ownership {name} is invalid")
            if int(row["last_segment"]) < int(row["first_segment"]):
                raise RuntimeError("canonical ownership segment range is reversed")
            if not isinstance(row.get("state_id"), str) or not row["state_id"]:
                raise RuntimeError("canonical ownership state_id is invalid")
            plan_sha = row.get("source_plan_sha256")
            if (
                type(plan_sha) is not str
                or len(plan_sha) != 64
                or any(character not in "0123456789abcdef" for character in plan_sha)
            ):
                raise RuntimeError("canonical ownership plan identity is invalid")
        owned.sort(key=lambda row: int(row["first_segment"]))
        flattened = [
            segment
            for row in owned
            for segment in range(
                int(row["first_segment"]), int(row["last_segment"]) + 1
            )
        ]
        if flattened != list(range(len(flattened))):
            raise RuntimeError(
                "canonical ownership must cover source segments 0..N-1 exactly once"
            )
        owner_by_state: dict[str, dict[str, Any]] = {}
        for row in owned:
            state_id = str(row["state_id"])
            if state_id in owner_by_state:
                raise RuntimeError("canonical ownership repeats a source/state owner")
            owner_by_state[state_id] = row

        actions_by_state: dict[str, list[dict[str, Any]]] = {}
        seen_profile_ids: set[str] = set()
        seen_owner_states: set[str] = set()
        for raw_profile in profile_rows:
            if not isinstance(raw_profile, Mapping):
                raise RuntimeError("Macro profile row is malformed")
            profile = dict(raw_profile)
            if profile.get("source_version") != source:
                continue
            profile_id = profile.get("profile_id")
            state_id = profile.get("state_id")
            strategy = profile.get("strategy")
            plan_sha = profile.get("source_plan_sha256")
            if type(profile_id) is not str or not profile_id:
                raise RuntimeError("canonical source profile_id is invalid")
            if profile_id in seen_profile_ids:
                raise RuntimeError("canonical source profile_id is duplicated")
            seen_profile_ids.add(profile_id)
            if type(state_id) is not str or state_id not in owner_by_state:
                raise RuntimeError("canonical source profile has no ownership state")
            if state_id in seen_owner_states:
                raise RuntimeError("canonical source ownership has multiple profiles")
            seen_owner_states.add(state_id)
            if type(strategy) is not str or not strategy:
                raise RuntimeError("canonical source profile strategy is invalid")
            owner = owner_by_state[state_id]
            if plan_sha != owner["source_plan_sha256"]:
                raise RuntimeError("canonical source profile/ownership plan mismatch")
            source_range = profile.get("source_segment_range")
            expected_range = [owner["first_segment"], owner["last_segment"]]
            if not _strict_json_equal(source_range, expected_range):
                raise RuntimeError("canonical source profile range differs from ownership")
            keyframes = profile.get("keyframes")
            if not isinstance(keyframes, list):
                raise RuntimeError("canonical source profile keyframes are unavailable")
            expected_segments = list(
                range(int(owner["first_segment"]), int(owner["last_segment"]) + 1)
            )
            raw_bindings = profile.get("segment_bindings")
            if not isinstance(raw_bindings, list):
                raise RuntimeError(
                    "canonical source profile completion bindings are unavailable"
                )
            binding_by_segment: dict[int, dict[str, Any]] = {}
            binding_keys = {
                "schema_version",
                "source_version",
                "source_plan_sha256",
                "source_plan_payload_sha256",
                "accepted_steps_sha256",
                "segment",
                "events",
                "completion_spec",
            }
            for raw_binding in raw_bindings:
                if not isinstance(raw_binding, Mapping) or set(raw_binding) != binding_keys:
                    raise RuntimeError(
                        "canonical completion binding schema is not exact"
                    )
                binding = dict(raw_binding)
                if (
                    binding.get("schema_version")
                    != "fsm50.playback_segment_binding.v1"
                    or binding.get("source_version") != source
                    or binding.get("source_plan_sha256") != plan_sha
                ):
                    raise RuntimeError(
                        "canonical completion binding source/plan identity mismatch"
                    )
                for digest_name in (
                    "source_plan_payload_sha256",
                    "accepted_steps_sha256",
                ):
                    digest = binding.get(digest_name)
                    if (
                        type(digest) is not str
                        or len(digest) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in digest
                        )
                    ):
                        raise RuntimeError(
                            f"canonical completion binding {digest_name} is invalid"
                        )
                if not isinstance(binding.get("segment"), Mapping) or not isinstance(
                    binding.get("events"), list
                ):
                    raise RuntimeError(
                        "canonical completion binding payload/events are malformed"
                    )
                try:
                    completion_spec = SegmentCompletionSpec.from_mapping(
                        binding.get("completion_spec", {})
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "canonical completion binding spec is invalid"
                    ) from exc
                segment_payload = dict(binding["segment"])
                if (
                    int(segment_payload.get("segment_index", -1))
                    != completion_spec.segment_index
                    or int(segment_payload.get("source_step", -1))
                    != completion_spec.source_step
                    or str(segment_payload.get("source_step_id", ""))
                    != completion_spec.source_step_id
                ):
                    raise RuntimeError(
                        "canonical completion binding segment/spec identity mismatch"
                    )
                if completion_spec.segment_index in binding_by_segment:
                    raise RuntimeError(
                        "canonical source profile repeats a completion binding"
                    )
                binding_by_segment[completion_spec.segment_index] = {
                    **binding,
                    "completion_spec": completion_spec.to_mapping(),
                }
            if sorted(binding_by_segment) != expected_segments:
                raise RuntimeError(
                    "canonical completion bindings do not exactly cover ownership"
                )
            actual_start_segments: list[int] = []
            profile_actions: list[dict[str, Any]] = []
            seen_coordinates: set[tuple[int, str]] = set()
            seen_starts: set[int] = set()
            prior_source_time = -1.0
            prior_sequence_index = -1
            for raw_frame in keyframes:
                if not isinstance(raw_frame, Mapping):
                    raise RuntimeError("canonical source keyframe is malformed")
                frame = dict(raw_frame)
                provenance = {
                    "kind": "SOURCE_ACTION",
                    "source_action_identity": "",
                    "source_version": frame.get("source_version", source),
                    "source_segment_index": frame.get("source_segment_index"),
                    "source_step_index": frame.get("source_step_index"),
                    "source_time_s": frame.get("source_time_s"),
                    "source_event_indices": frame.get("source_event_indices"),
                    "commands": frame.get("commands"),
                    "dispatch_kind": frame.get("dispatch_kind"),
                    "sequence_index": frame.get("sequence_index"),
                }
                self._validate_source_action_provenance_shape(provenance)
                provenance["source_action_identity"] = _source_action_identity(
                    provenance
                )
                segment = int(provenance["source_segment_index"])
                dispatch_kind = str(provenance["dispatch_kind"])
                coordinate = (segment, dispatch_kind)
                if coordinate in seen_coordinates:
                    raise RuntimeError(
                        "canonical source profile repeats an action coordinate"
                    )
                seen_coordinates.add(coordinate)
                source_time = float(provenance["source_time_s"])
                sequence_index = int(provenance["sequence_index"])
                if (
                    source_time + 1.0e-12 < prior_source_time
                    or sequence_index != prior_sequence_index + 1
                ):
                    raise RuntimeError(
                        "canonical source profile action order is not exact"
                    )
                prior_source_time = source_time
                prior_sequence_index = sequence_index
                if dispatch_kind == "segment_start":
                    actual_start_segments.append(segment)
                    seen_starts.add(segment)
                elif segment not in seen_starts:
                    raise RuntimeError(
                        "wheel completion source action precedes its segment start"
                    )
                profile_actions.append(
                    {
                        "source_action_index": -1,
                        "owner_state": state_id,
                        "profile_id": profile_id,
                        "profile_source_version": source,
                        "profile_strategy": strategy,
                        "source_plan_sha256": plan_sha,
                        "command_provenance": provenance,
                        "servo_targets_deg": self._validated_target_map(
                            frame.get("servo_targets_deg"),
                            names=SERVO_JOINT_NAMES,
                            label="canonical source servo_targets_deg",
                        ),
                        "wheel_targets_rad_s": self._validated_target_map(
                            frame.get("wheel_targets_rad_s"),
                            names=WHEEL_JOINT_NAMES,
                            label="canonical source wheel_targets_rad_s",
                        ),
                        "segment_completion_binding": (
                            binding_by_segment[segment]
                            if dispatch_kind == "segment_start"
                            else None
                        ),
                    }
                )
                if dispatch_kind == "segment_start":
                    spec_targets = dict(
                        binding_by_segment[segment]["completion_spec"].get(
                            "servo_targets_deg", {}
                        )
                    )
                    full_targets = profile_actions[-1]["servo_targets_deg"]
                    if (
                        not set(spec_targets).issubset(SERVO_JOINT_NAMES)
                        or any(
                            not math.isclose(
                                float(full_targets[name]),
                                float(target),
                                rel_tol=0.0,
                                abs_tol=1.0e-12,
                            )
                            for name, target in spec_targets.items()
                        )
                    ):
                        raise RuntimeError(
                            "completion spec sparse servo endpoints differ from "
                            "the canonical source action"
                        )
            if actual_start_segments != expected_segments:
                raise RuntimeError(
                    "canonical source segment-start actions do not exactly match ownership"
                )
            actions_by_state[state_id] = profile_actions
        if seen_owner_states != set(owner_by_state):
            raise RuntimeError("canonical source ownership/profile coverage is incomplete")
        actions = [
            action
            for owner in owned
            for action in actions_by_state[str(owner["state_id"])]
        ]
        segments = [
            int(row["command_provenance"]["source_segment_index"])
            for row in actions
            if row["command_provenance"]["dispatch_kind"] == "segment_start"
        ]
        if segments != flattened:
            raise RuntimeError(
                "canonical source actions must cover segments 0..N-1 exactly once"
            )
        for index, row in enumerate(actions):
            row["source_action_index"] = index
        return actions

    @staticmethod
    def _validate_source_action_provenance_shape(
        provenance: Mapping[str, Any],
    ) -> None:
        if set(provenance) != set(_COMMAND_PROVENANCE_KEYS):
            raise RuntimeError("SOURCE_ACTION provenance keys are not exact")
        if provenance.get("kind") != "SOURCE_ACTION":
            raise RuntimeError("source-action provenance kind is invalid")
        identity = provenance.get("source_action_identity")
        if identity != "" and (
            type(identity) is not str
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise RuntimeError("source-action provenance identity is invalid")
        if type(provenance.get("source_version")) is not str or not provenance.get(
            "source_version"
        ):
            raise RuntimeError("source-action provenance source_version is invalid")
        for name in (
            "source_segment_index",
            "source_step_index",
            "sequence_index",
        ):
            value = provenance.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"source-action provenance {name} is invalid")
        source_time = provenance.get("source_time_s")
        if (
            type(source_time) is not float
            or not math.isfinite(source_time)
            or source_time < 0.0
        ):
            raise RuntimeError("source-action provenance source_time_s is invalid")
        event_indices = provenance.get("source_event_indices")
        if (
            not isinstance(event_indices, list)
            or any(type(value) is not int or value < 0 for value in event_indices)
        ):
            raise RuntimeError(
                "source-action provenance source_event_indices is invalid"
            )
        commands = provenance.get("commands")
        if (
            not isinstance(commands, list)
            or any(type(value) is not str or not value for value in commands)
        ):
            raise RuntimeError("source-action provenance commands are invalid")
        if provenance.get("dispatch_kind") not in {
            "segment_start",
            "wheel_channel_completion_stop",
        }:
            raise RuntimeError("source-action provenance dispatch_kind is invalid")
        if provenance["dispatch_kind"] == "segment_start":
            if not event_indices or not commands:
                raise RuntimeError(
                    "segment_start provenance requires commands and event indices"
                )
        elif event_indices or commands:
            raise RuntimeError(
                "wheel_channel_completion_stop provenance requires empty commands/events"
            )

    def _source_action_coverage_errors(self) -> list[str]:
        errors: list[str] = []
        expected_action_count = len(self.expected_source_actions)
        completed_action_count = len(self.source_action_consumption_rows)
        if completed_action_count != self.next_source_action_index:
            errors.append("source-action ledger/cursor count mismatch")
        if completed_action_count != expected_action_count:
            errors.append(
                "source-action coverage incomplete: "
                f"{completed_action_count}/{expected_action_count}"
            )
        actual_identities = [
            row.get("command_provenance", {}).get("source_action_identity")
            for row in self.source_action_consumption_rows
        ]
        expected_identities = [
            row["command_provenance"]["source_action_identity"]
            for row in self.expected_source_actions
        ]
        if actual_identities != expected_identities:
            errors.append("source-action identity coverage differs from canonical bundle")
        actual_coordinates = [
            (
                row.get("command_provenance", {}).get("source_version"),
                row.get("command_provenance", {}).get("source_segment_index"),
                row.get("command_provenance", {}).get("dispatch_kind"),
            )
            for row in self.source_action_consumption_rows
        ]
        expected_coordinates = [
            (
                row["command_provenance"]["source_version"],
                row["command_provenance"]["source_segment_index"],
                row["command_provenance"]["dispatch_kind"],
            )
            for row in self.expected_source_actions
        ]
        if actual_coordinates != expected_coordinates:
            errors.append("source-action coordinates differ from canonical bundle")
        expected_segments = [
            row["command_provenance"]["source_segment_index"]
            for row in self.expected_source_actions
            if row["command_provenance"]["dispatch_kind"] == "segment_start"
        ]
        actual_segments = [
            row.get("command_provenance", {}).get("source_segment_index")
            for row in self.source_action_consumption_rows
            if row.get("command_provenance", {}).get("dispatch_kind")
            == "segment_start"
        ]
        if expected_segments != list(range(len(expected_segments))):
            errors.append("canonical segment-start coverage is not ordered 0..N-1")
        if actual_segments != expected_segments:
            errors.append(
                "consumed segment-start coverage differs from canonical 0..N-1"
            )
        return errors

    def _segment_completion_coverage_errors(self) -> list[str]:
        errors: list[str] = []
        expected_segments = [
            int(row["command_provenance"]["source_segment_index"])
            for row in self.expected_source_actions
            if row["command_provenance"]["dispatch_kind"] == "segment_start"
        ]
        actual_segments = [
            row.get("source_segment_index") for row in self.segment_completion_rows
        ]
        if self.active_segment_completion_row_index is not None:
            errors.append("a segment completion executor remains active")
        if actual_segments != expected_segments:
            errors.append(
                "segment completion ledger coverage differs from canonical 0..N-1"
            )
        for index, row in enumerate(self.segment_completion_rows):
            label = f"segment completion row {index}"
            targets = dict(row.get("completion_spec", {}).get("servo_targets_deg", {}) or {})
            if row.get("segment_completion_index") != index:
                errors.append(f"{label} index mismatch")
            if row.get("terminal_kind") != SegmentDecisionKind.COMPLETE.value:
                errors.append(f"{label} is not COMPLETE")
            if row.get("start_readback_verified") is not True or not row.get(
                "start_readback_sha256"
            ):
                errors.append(f"{label} lacks exact start target readback")
            if row.get("tracking_lifecycle_closed") is not True:
                errors.append(f"{label} tracking lifecycle is not closed")
            expected_tracking_calls = 1 if targets else 0
            if (
                row.get("tracking_begin_count") != expected_tracking_calls
                or row.get("tracking_end_count") != expected_tracking_calls
            ):
                errors.append(f"{label} tracking begin/end call count is not exact")
            wheel_stop = row.get("wheel_stop")
            if wheel_stop is not None and (
                not isinstance(wheel_stop, Mapping)
                or wheel_stop.get("n_plus_one_verified") is not True
                or not wheel_stop.get("n_plus_one_readback_sha256")
            ):
                errors.append(f"{label} wheel stop lacks exact N+1 readback")
            decisions = row.get("observation_decisions")
            if not isinstance(decisions, list) or not decisions:
                errors.append(f"{label} has no boundary completion observations")
            elif decisions[-1].get("kind") != SegmentDecisionKind.COMPLETE.value:
                errors.append(f"{label} final decision is not COMPLETE")
        return errors

    def _successful_coverage_errors(self) -> list[str]:
        return [
            *self._source_action_coverage_errors(),
            *self._segment_completion_coverage_errors(),
        ]

    def _root_and_joint_state(
        self, adapter: Any
    ) -> tuple[list[float], list[float], dict[str, float], dict[str, float], bool]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        pose, pose_ok = _exact_single_vector(getattr(data, "root_pose_w", None), 7)
        velocity, velocity_ok = _exact_single_vector(getattr(data, "root_vel_w", None), 6)
        names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
        expected = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
        q, q_ok = _exact_single_vector(getattr(data, "joint_pos", None), len(expected))
        qd, qd_ok = _exact_single_vector(getattr(data, "joint_vel", None), len(expected))
        joint_q = {name: float(q[index]) for index, name in enumerate(names[: q.size])}
        joint_qd = {name: float(qd[index]) for index, name in enumerate(names[: qd.size])}
        exact = bool(
            pose_ok
            and velocity_ok
            and q_ok
            and qd_ok
            and len(names) == len(expected)
            and set(names) == set(expected)
        )
        return pose.tolist(), velocity.tolist(), joint_q, joint_qd, exact

    def _wheel_centers(self, adapter: Any) -> dict[str, tuple[float, float, float]]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
        raw = getattr(data, "body_link_state_w", None)
        if raw is None:
            raw = getattr(data, "body_state_w", None)
        try:
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().numpy()
            states = np.asarray(raw, dtype=float)
            if states.ndim == 3 and states.shape[0] == 1:
                states = states[0]
        except Exception:
            return {}
        if states.ndim != 2 or states.shape[1] < 3:
            return {}
        by_name = {
            name: tuple(float(value) for value in states[index, :3])
            for index, name in enumerate(names[: states.shape[0]])
        }
        return {
            leg: by_name[body]
            for leg, body in LEG_TO_WHEEL_BODY.items()
            if body in by_name
        }

    def _obstacle(self) -> ObstacleGeometry:
        config = getattr(self.scene_handle, "config", None)
        front = float(getattr(config, "obstacle_front_x", 0.0) or 0.0)
        bottom = float(getattr(config, "ground_z_m", 0.0) or 0.0)
        height = float(getattr(config, "obstacle_height_m", 0.05) or 0.05)
        length = float(getattr(config, "obstacle_length", 0.0) or 0.0)
        width = float(getattr(config, "obstacle_width", 2.0) or 2.0)
        return ObstacleGeometry(
            front_face_x_m=front,
            top_z_m=bottom + height,
            bottom_z_m=bottom,
            rear_face_x_m=front + length if length > 0.0 else None,
            width_m=width,
        )

    def _joint_limit_violation(
        self, adapter: Any, joint_q: Mapping[str, float]
    ) -> tuple[bool | None, bool]:
        records = dict(getattr(adapter, "safe_joint_limit_records", {}) or {})
        for name in SERVO_JOINT_NAMES:
            record = records.get(name)
            if not isinstance(record, Mapping) or name not in joint_q:
                return None, False
            try:
                value = float(joint_q[name])
                minimum = float(record["min_rad"])
                maximum = float(record["max_rad"])
            except (KeyError, TypeError, ValueError):
                return None, False
            if not all(math.isfinite(item) for item in (value, minimum, maximum)):
                return None, False
            if value < minimum - 1.0e-3 or value > maximum + 1.0e-3:
                return True, True
        return False, True

    def _capture_deployment_safety_evidence(self, adapter: Any) -> dict[str, Any]:
        try:
            raw = self.safety_evidence_factory(adapter, self.scene_handle)
        except Exception as exc:
            raw = {
                "available": False,
                "dangerous_body_collision": None,
                "severe_penetration": None,
                "source": "safety_evidence_factory",
                "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
                "error": f"{type(exc).__name__}: {exc}",
            }
        evidence = dict(_jsonable(dict(raw or {})) or {}) if isinstance(raw, Mapping) else {}
        errors: list[str] = []
        try:
            with self.request.task_success_table_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                rows = [dict(row) for row in csv.DictReader(stream)]
        except Exception as exc:
            rows = []
            errors.append(f"Gate-A table read failed: {type(exc).__name__}: {exc}")
        matches = [
            row
            for row in rows
            if str(row.get("version", "") or "") == self.request.source_version
        ]
        gate_a_row = matches[0] if len(matches) == 1 else {}
        if len(matches) != 1:
            errors.append(
                "authorized Macro source must resolve exactly once in Gate-A table"
            )
        if gate_a_row:
            if gate_a_row.get("evaluation_status") != "EVALUATED" or gate_a_row.get(
                "task_result"
            ) not in {
                "REPLAY_TASK_SUCCESS",
                "REPLAY_TASK_SUCCESS_POSTURE_INCOMPLETE",
            }:
                errors.append(
                    "authorized Macro source Gate-A row is not an evaluated task success"
                )
            notes = str(gate_a_row.get("notes", "") or "").lower()
            if (
                "no fall" not in notes
                or "dangerous body collision" not in notes
                or "severe penetration" not in notes
                or "was visible" not in notes
            ):
                errors.append(
                    "authorized Macro source Gate-A row lacks clean "
                    "collision/penetration review evidence"
                )
        for name in ("dangerous_body_collision", "severe_penetration"):
            if type(evidence.get(name)) is not bool:
                errors.append(f"live initial geometry evidence {name} is not exact bool")
            elif evidence[name] is not False:
                errors.append(f"live initial geometry evidence reports {name}=true")
        pose, velocity, joint_q, joint_qd, exact = self._root_and_joint_state(adapter)
        centers = self._wheel_centers(adapter)
        if not exact or len(pose) != 7 or len(velocity) != 6:
            errors.append("initial root/joint state is not exact and finite")
        if len(joint_q) != len(SERVO_JOINT_NAMES) + len(WHEEL_JOINT_NAMES):
            errors.append("initial joint-position map is incomplete")
        if len(joint_qd) != len(SERVO_JOINT_NAMES) + len(WHEEL_JOINT_NAMES):
            errors.append("initial joint-velocity map is incomplete")
        if len(centers) != len(LEGS):
            errors.append("initial live wheel-center geometry is incomplete")
        initial_joint_limit_violation, joint_limit_available = (
            self._joint_limit_violation(adapter, joint_q)
        )
        if not joint_limit_available:
            errors.append("initial measured joint-limit evidence is unavailable")
        elif initial_joint_limit_violation is True:
            errors.append("initial measured joint-limit violation")
        initial_roll_rad = initial_pitch_rad = float("nan")
        if len(pose) == 7 and all(math.isfinite(float(value)) for value in pose[3:7]):
            initial_roll_rad, initial_pitch_rad, _initial_yaw = quat_wxyz_to_rpy(
                pose[3:7]
            )
        if (
            not math.isfinite(float(initial_roll_rad))
            or not math.isfinite(float(initial_pitch_rad))
            or abs(float(initial_roll_rad)) >= math.radians(85.0)
            or abs(float(initial_pitch_rad)) >= math.radians(85.0)
        ):
            errors.append("initial root attitude is fallen or unavailable")
        root_write_count = getattr(adapter, "root_state_write_count", None)
        if type(root_write_count) is not int or root_write_count != 0:
            errors.append("initial boundary has a root-state write")
        errors.extend(
            [str(evidence.get("error", "") or "")]
            if evidence.get("available") is not True
            else []
        )
        errors = [item for item in errors if item]
        gate_a_row_sha256 = _canonical_json_sha256(gate_a_row) if gate_a_row else ""
        live_geometry_snapshot = {
            "source": evidence.get("source"),
            "geometry": evidence.get("geometry"),
            "ground_diagnostics": evidence.get("ground_diagnostics"),
            "robot_root_pose": evidence.get("robot_root_pose"),
            "obstacle_bounds_min_m": evidence.get("obstacle_bounds_min_m"),
            "obstacle_bounds_max_m": evidence.get("obstacle_bounds_max_m"),
            "robot_collision_bounds_min_m": evidence.get(
                "robot_collision_bounds_min_m"
            ),
            "robot_collision_bounds_max_m": evidence.get(
                "robot_collision_bounds_max_m"
            ),
            "initial_robot_to_obstacle_clearance_m": evidence.get(
                "initial_robot_to_obstacle_clearance_m"
            ),
            "initial_maximum_ground_penetration_m": evidence.get(
                "initial_maximum_ground_penetration_m"
            ),
            "ground_penetration_tolerance_m": evidence.get(
                "ground_penetration_tolerance_m"
            ),
            "ground_penetration_tolerance_source": evidence.get(
                "ground_penetration_tolerance_source"
            ),
        }
        initial_state_snapshot = {
            "root_pose_w": pose,
            "root_velocity_w": velocity,
            "joint_position_rad": joint_q,
            "joint_velocity_rad_s": joint_qd,
            "wheel_centers_w_m": centers,
            "joint_limit_violation": initial_joint_limit_violation,
            "joint_limit_evidence_available": joint_limit_available,
            "base_roll_rad": initial_roll_rad,
            "base_pitch_rad": initial_pitch_rad,
            "sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
            "root_state_write_count": (
                root_write_count if type(root_write_count) is int else -1
            ),
        }
        live_geometry_sha256 = _canonical_json_sha256(live_geometry_snapshot)
        initial_state_sha256 = _canonical_json_sha256(initial_state_snapshot)
        deployment_binding = {
            "source_version": self.request.source_version,
            "task_success_table_path": str(self.request.task_success_table_path),
            "task_success_table_sha256": self.request.task_success_table_sha256,
            "gate_a_row_sha256": gate_a_row_sha256,
            "alignment_path": str(self.request.alignment_path),
            "alignment_sha256": self.request.alignment_sha256,
            "live_geometry_sha256": live_geometry_sha256,
            "initial_state_sha256": initial_state_sha256,
        }
        evidence.update(
            available=not errors,
            dangerous_body_collision=False if not errors else None,
            severe_penetration=False if not errors else None,
            sample_sim_step=int(getattr(adapter, "sim_steps", 0) or 0),
            task_success_table_path=str(self.request.task_success_table_path),
            task_success_table_sha256=self.request.task_success_table_sha256,
            alignment_path=str(self.request.alignment_path),
            alignment_sha256=self.request.alignment_sha256,
            source_version=self.request.source_version,
            gate_a_row=gate_a_row,
            initial_root_pose_w=pose,
            initial_root_velocity_w=velocity,
            initial_joint_position_rad=joint_q,
            initial_joint_velocity_rad_s=joint_qd,
            initial_wheel_centers_w_m=centers,
            initial_joint_limit_violation=initial_joint_limit_violation,
            initial_joint_limit_evidence_available=joint_limit_available,
            initial_base_roll_rad=initial_roll_rad,
            initial_base_pitch_rad=initial_pitch_rad,
            gate_a_row_sha256=gate_a_row_sha256,
            live_geometry_sha256=live_geometry_sha256,
            initial_state_sha256=initial_state_sha256,
            deployment_binding_sha256=_canonical_json_sha256(deployment_binding),
            error="; ".join(dict.fromkeys(errors)),
        )
        return dict(_jsonable(evidence))

    def _capture_runtime_safety_evidence(self, adapter: Any) -> dict[str, Any]:
        capture = getattr(adapter, "capture_macro_runtime_safety_evidence", None)
        if not callable(capture):
            return {
                "available": False,
                "dangerous_body_collision": None,
                "severe_penetration": None,
                "source": "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW",
                "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
                "error": "runtime non-wheel collision/penetration sensor is unavailable",
            }
        try:
            raw = capture(scene_handle=self.scene_handle)
        except Exception as exc:
            return {
                "available": False,
                "dangerous_body_collision": None,
                "severe_penetration": None,
                "source": "capture_macro_runtime_safety_evidence",
                "sample_sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
                "error": f"{type(exc).__name__}: {exc}",
            }
        return dict(_jsonable(dict(raw or {})) or {}) if isinstance(raw, Mapping) else {}

    def _normalize_optional_runtime_bool(
        self,
        evidence: Mapping[str, Any],
        *,
        field: str,
        expected_sim_step: int,
    ) -> dict[str, Any]:
        """Normalize one optional live sensor without manufacturing a clear result."""

        available_key = f"{field}_available"
        source_key = f"{field}_source"
        step_key = f"{field}_sample_sim_step"
        keys = (field, available_key, source_key, step_key)
        present = [key in evidence for key in keys]
        if not any(present):
            value = None
            available = False
            source = UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE
            sample_sim_step = int(expected_sim_step)
        else:
            if not all(present):
                raise RuntimeError(
                    f"runtime {field} producer contract is incomplete"
                )
            value = evidence[field]
            available = evidence[available_key]
            source = evidence[source_key]
            sample_sim_step = evidence[step_key]
            if type(available) is not bool:
                raise RuntimeError(
                    f"runtime {field} availability must be exact bool"
                )
            if type(source) is not str or not source.strip():
                raise RuntimeError(f"runtime {field} source is missing")
            source = source.strip()
            if type(sample_sim_step) is not int or sample_sim_step != int(
                expected_sim_step
            ):
                raise RuntimeError(f"runtime {field} sample step mismatch")
            if available:
                if type(value) is not bool:
                    raise RuntimeError(
                        f"runtime {field} must be exact bool when available"
                    )
                if source == UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE:
                    raise RuntimeError(
                        f"runtime {field} available evidence has unavailable source"
                    )
            elif value is not None:
                raise RuntimeError(
                    f"runtime {field} must remain None when unavailable"
                )

        tracker = self.optional_runtime_evidence[field]
        tracker["sample_count"] = int(tracker["sample_count"]) + 1
        if available:
            prior_source = str(tracker.get("source", "") or "")
            if prior_source and prior_source != source:
                raise RuntimeError(f"runtime {field} producer source drift")
            tracker["source"] = source
            tracker["available_sample_count"] = int(
                tracker["available_sample_count"]
            ) + 1
        return {
            field: value,
            available_key: available,
            source_key: source,
            step_key: sample_sim_step,
        }

    def _terminal_optional_runtime_bool(
        self, field: str, *, detected: bool
    ) -> dict[str, Any]:
        tracker = self.optional_runtime_evidence[field]
        sample_count = int(tracker.get("sample_count", 0) or 0)
        available_count = int(
            tracker.get("available_sample_count", 0) or 0
        )
        complete_live_coverage = bool(
            sample_count > 0 and available_count == sample_count
        )
        available = bool(detected or complete_live_coverage)
        value = True if detected else False if complete_live_coverage else None
        source = (
            str(tracker.get("source", "") or "")
            if available
            else UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE
        )
        return {
            field: value,
            f"{field}_available": available,
            f"{field}_source": source,
            f"{field}_validation_source": source,
            f"{field}_sample_count": sample_count,
            f"{field}_available_sample_count": available_count,
            f"{field}_complete_live_coverage": complete_live_coverage,
        }

    @staticmethod
    def _validate_deployment_safety_evidence(
        evidence: Mapping[str, Any],
        *,
        expected_sim_step: int,
        require_clear: bool,
    ) -> None:
        if evidence.get("available") is not True:
            raise RuntimeError(
                "deployed collision/penetration evidence is unavailable: "
                + str(evidence.get("error", "missing evidence") or "missing evidence")
            )
        if type(evidence.get("sample_sim_step")) is not int or evidence.get(
            "sample_sim_step"
        ) != int(expected_sim_step):
            raise RuntimeError("deployed safety evidence sim-step mismatch")
        if not isinstance(evidence.get("source"), str) or not str(
            evidence.get("source", "")
        ):
            raise RuntimeError("deployed safety evidence source is missing")
        if str(evidence.get("error", "") or ""):
            raise RuntimeError(
                "deployed safety evidence reports an error: "
                + str(evidence.get("error"))
            )
        for name in ("dangerous_body_collision", "severe_penetration"):
            if type(evidence.get(name)) is not bool:
                raise RuntimeError(f"deployed safety evidence {name} must be exact bool")
            if require_clear and evidence[name] is not False:
                raise RuntimeError(f"deployed safety evidence reports {name}=true")
        if require_clear:
            for name in (
                "task_success_table_sha256",
                "alignment_sha256",
                "gate_a_row_sha256",
                "live_geometry_sha256",
                "initial_state_sha256",
                "deployment_binding_sha256",
            ):
                value = evidence.get(name)
                if (
                    type(value) is not str
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise RuntimeError(
                        f"deployed safety evidence {name} is not a SHA-256 identity"
                    )

    @staticmethod
    def _strict_finite_map(
        raw: Any,
        *,
        names: list[str],
        label: str,
        allow_extra: bool = False,
    ) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{label} is not a mapping")
        keys = set(raw)
        expected = set(names)
        if (not allow_extra and keys != expected) or (allow_extra and not expected <= keys):
            raise RuntimeError(f"{label} does not contain the required exact keys")
        result: dict[str, float] = {}
        for name in names:
            value = raw[name]
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise RuntimeError(f"{label}.{name} must be finite numeric data")
            result[name] = float(value)
        return result

    def _capture_and_validate_target_readback(
        self,
        adapter: Any,
        *,
        servo_targets: Mapping[str, float],
        wheel_targets: Mapping[str, float],
        expected_sim_step: int,
        pending: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        capture = getattr(adapter, "capture_motion_start_base_evidence", None)
        if not callable(capture):
            raise RuntimeError("production adapter target readback API is unavailable")
        raw = capture()
        if not isinstance(raw, Mapping):
            raise RuntimeError("production adapter target readback is not a mapping")
        evidence = dict(raw)
        if type(evidence.get("sim_step")) is not int or evidence.get("sim_step") != int(
            expected_sim_step
        ):
            raise RuntimeError("target readback sim_step mismatch or N+1 readback missing")
        if evidence.get("adapter_runtime_instance_id") != str(
            getattr(adapter, "runtime_instance_id", "") or ""
        ):
            raise RuntimeError("target readback adapter runtime identity mismatch")
        if type(evidence.get("root_state_write_count")) is not int or evidence.get(
            "root_state_write_count"
        ) != 0:
            raise RuntimeError("target readback observed a root-state write")
        if list(evidence.get("root_state_write_events", []) or []):
            raise RuntimeError("target readback root-state write ledger is non-empty")
        dt = evidence.get("physics_dt_s")
        if type(dt) not in (int, float) or not math.isfinite(float(dt)) or not math.isclose(
            float(dt), float(self.physics_dt_s), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError("target readback physics dt mismatch")
        if evidence.get("joint_state_evidence_valid") is not True or str(
            evidence.get("joint_state_evidence_error", "") or ""
        ):
            raise RuntimeError(
                "servo PhysX target readback is unavailable: "
                + str(evidence.get("joint_state_evidence_error", "") or "invalid evidence")
            )
        if evidence.get("wheel_target_evidence_valid") is not True or str(
            evidence.get("wheel_target_evidence_error", "") or ""
        ):
            raise RuntimeError(
                "wheel PhysX target readback is unavailable: "
                + str(evidence.get("wheel_target_evidence_error", "") or "invalid evidence")
            )
        expected_servos = self._strict_finite_map(
            servo_targets,
            names=SERVO_JOINT_NAMES,
            label="expected servo targets",
        )
        expected_wheels = self._strict_finite_map(
            wheel_targets,
            names=WHEEL_JOINT_NAMES,
            label="expected wheel targets",
        )
        command_state = evidence.get("command_state")
        if not isinstance(command_state, Mapping) or set(command_state) != {
            "servos",
            "wheels",
        }:
            raise RuntimeError("target readback command_state is not exact")
        command_servos = self._strict_finite_map(
            command_state["servos"],
            names=SERVO_JOINT_NAMES,
            label="target readback canonical servos",
        )
        command_wheels = self._strict_finite_map(
            command_state["wheels"],
            names=WHEEL_JOINT_NAMES,
            label="target readback canonical wheels",
        )
        for name, expected in expected_servos.items():
            if not math.isclose(
                command_servos[name], expected, rel_tol=0.0, abs_tol=TARGET_COMMAND_TOLERANCE
            ):
                raise RuntimeError(f"canonical servo target drift at {name}")
        for name, expected in expected_wheels.items():
            if not math.isclose(
                command_wheels[name], expected, rel_tol=0.0, abs_tol=TARGET_COMMAND_TOLERANCE
            ):
                raise RuntimeError(f"canonical wheel target drift at {name}")

        servo_readback = self._strict_finite_map(
            evidence.get("joint_position_target_by_name"),
            names=SERVO_JOINT_NAMES,
            label="PhysX servo position target readback",
            allow_extra=True,
        )
        servo_buffer = self._strict_finite_map(
            evidence.get("joint_position_target_buffer_by_name"),
            names=SERVO_JOINT_NAMES,
            label="IsaacLab servo target buffer",
            allow_extra=True,
        )
        servo_command = self._strict_finite_map(
            evidence.get("servo_command_target_by_name"),
            names=SERVO_JOINT_NAMES,
            label="adapter servo command target",
        )
        for name in SERVO_JOINT_NAMES:
            if not math.isclose(
                servo_readback[name],
                servo_buffer[name],
                rel_tol=0.0,
                abs_tol=DRIVE_READBACK_TOLERANCE,
            ) or not math.isclose(
                servo_readback[name],
                servo_command[name],
                rel_tol=0.0,
                abs_tol=DRIVE_READBACK_TOLERANCE,
            ):
                raise RuntimeError(f"actual servo drive-target readback drift at {name}")
        wheel_command = self._strict_finite_map(
            evidence.get("wheel_target_velocity_by_name"),
            names=WHEEL_JOINT_NAMES,
            label="adapter wheel drive target",
        )
        wheel_readback = self._strict_finite_map(
            evidence.get("wheel_target_readback_velocity_by_name"),
            names=WHEEL_JOINT_NAMES,
            label="PhysX wheel drive-target readback",
        )
        for name in WHEEL_JOINT_NAMES:
            if not math.isclose(
                wheel_readback[name],
                wheel_command[name],
                rel_tol=0.0,
                abs_tol=DRIVE_READBACK_TOLERANCE,
            ):
                raise RuntimeError(f"actual wheel drive-target readback drift at {name}")

        if pending is not None:
            status = getattr(adapter, "motion_batch_status", None)
            if not isinstance(status, Mapping) or status.get("batch_id") != pending.get(
                "batch_id"
            ):
                raise RuntimeError("pending command batch identity was not retained to N+1")
            if (
                status.get("source") != pending.get("source")
                or status.get("servo_applied") is not True
                or status.get("wheel_applied") is not True
            ):
                raise RuntimeError("pending command atomic/source identity drift")
            if (
                type(status.get("applied_sim_step")) is not int
                or status.get("applied_sim_step")
                != pending.get("applied_sim_step")
            ):
                raise RuntimeError("pending command applied_sim_step drift")
            if (
                type(status.get("first_physics_step")) is not int
                or status.get("first_physics_step")
                != pending.get("expected_sim_step")
            ):
                raise RuntimeError("pending command first_physics_step drift")
            for label, raw_status, expected_status in (
                (
                    "servo",
                    status.get("servo_targets_applied"),
                    pending.get("servo_targets_deg"),
                ),
                (
                    "wheel",
                    status.get("wheel_targets_applied"),
                    pending.get("wheel_targets_rad_s"),
                ),
            ):
                if not isinstance(raw_status, Mapping) or not isinstance(
                    expected_status, Mapping
                ):
                    raise RuntimeError(
                        f"pending command {label} target identity is unavailable"
                    )
                if set(raw_status) != set(expected_status) or any(
                    type(raw_status[name]) not in (int, float)
                    or not math.isfinite(float(raw_status[name]))
                    or not math.isclose(
                        float(raw_status[name]),
                        float(expected_status[name]),
                        rel_tol=0.0,
                        abs_tol=TARGET_COMMAND_TOLERANCE,
                    )
                    for name in expected_status
                ):
                    raise RuntimeError(
                        f"pending command {label} target identity drift"
                    )
            metadata_raw = status.get("recording_metadata")
            if not isinstance(metadata_raw, Mapping) or not _strict_json_equal(
                dict(metadata_raw), dict(pending.get("recording_metadata", {}) or {})
            ):
                raise RuntimeError("pending command recording metadata drift")
            metadata = dict(metadata_raw)
            if pending.get("kind") in {"controller", "safe_stop"}:
                if type(metadata.get("command_epoch")) is not int or metadata.get(
                    "command_epoch"
                ) != pending.get("command_epoch"):
                    raise RuntimeError("pending command epoch identity mismatch at N+1")
        return {
            "sim_step": int(expected_sim_step),
            "command_epoch": None if pending is None else pending.get("command_epoch"),
            "batch_id": "" if pending is None else str(pending.get("batch_id", "")),
            "canonical_servo_targets_deg": command_servos,
            "canonical_wheel_targets_rad_s": command_wheels,
            "actual_servo_drive_targets_rad": servo_readback,
            "actual_wheel_drive_targets_rad_s": wheel_readback,
            "adapter_runtime_instance_id": evidence.get("adapter_runtime_instance_id"),
            "root_state_write_count": evidence.get("root_state_write_count"),
            "physics_dt_s": float(dt),
        }

    def _establish_pending_readback(
        self,
        *,
        kind: str,
        batch_id: str,
        ack: Mapping[str, Any],
        servo_targets: Mapping[str, float],
        wheel_targets: Mapping[str, float],
        command_epoch: int,
        physical_command_epoch: int | None = None,
        physical_dispatch_index: int | None = None,
        source_action_consumption_index: int | None = None,
    ) -> None:
        pending = {
            "kind": str(kind),
            "batch_id": str(batch_id),
            "applied_sim_step": int(ack["applied_sim_step"]),
            "expected_sim_step": int(ack["first_physics_step"]),
            "command_epoch": int(command_epoch),
            "source": str(ack["source"]),
            "recording_metadata": dict(ack.get("recording_metadata", {}) or {}),
            "servo_targets_deg": dict(servo_targets),
            "wheel_targets_rad_s": dict(wheel_targets),
            "physical_dispatch_index": physical_dispatch_index,
            "source_action_consumption_index": source_action_consumption_index,
        }
        if physical_command_epoch is not None:
            if type(physical_command_epoch) is not int or physical_command_epoch < 0:
                raise RuntimeError("physical command epoch must be a non-negative int")
            pending["physical_command_epoch"] = physical_command_epoch
        self.pending_readback = pending

    def _verify_pending_readback(self, adapter: Any, *, sim_step: int) -> str:
        if self.pending_readback is None:
            return ""
        pending = dict(self.pending_readback)
        expected_step = int(pending["expected_sim_step"])
        if int(sim_step) != expected_step:
            raise RuntimeError(
                "pending actuator target readback was not observed at exact N+1: "
                f"expected={expected_step} actual={sim_step}"
            )
        evidence = self._capture_and_validate_target_readback(
            adapter,
            servo_targets=pending["servo_targets_deg"],
            wheel_targets=pending["wheel_targets_rad_s"],
            expected_sim_step=expected_step,
            pending=pending,
        )
        kind = str(pending["kind"])
        if kind == "controller":
            consumption_row: dict[str, Any] | None = None
            dispatch_index = pending.get("physical_dispatch_index")
            if (
                type(dispatch_index) is not int
                or dispatch_index < 0
                or dispatch_index >= len(self.dispatch_rows)
            ):
                raise RuntimeError(
                    "pending controller readback lacks its physical dispatch row"
                )
            dispatch_row = self.dispatch_rows[dispatch_index]
            if (
                dispatch_row.get("batch_id") != pending["batch_id"]
                or dispatch_row.get("command_epoch") != pending["command_epoch"]
            ):
                raise RuntimeError(
                    "pending controller readback/physical dispatch identity mismatch"
                )
            if "physical_command_epoch" in pending and (
                dispatch_row.get("physical_command_epoch")
                != pending["physical_command_epoch"]
            ):
                raise RuntimeError(
                    "pending controller readback/physical epoch identity mismatch"
                )
            consumption_index = pending.get("source_action_consumption_index")
            if consumption_index is not None:
                if (
                    type(consumption_index) is not int
                    or consumption_index < 0
                    or consumption_index >= len(self.source_action_consumption_rows)
                ):
                    raise RuntimeError(
                        "pending source readback lacks its consumption row"
                    )
                consumption_row = self.source_action_consumption_rows[
                    consumption_index
                ]
                if (
                    consumption_row.get("physical_dispatch_index")
                    != dispatch_index
                    or consumption_row.get("batch_id") != pending["batch_id"]
                    or consumption_row.get("dispatch_epoch")
                    != pending["command_epoch"]
                ):
                    raise RuntimeError(
                        "pending source readback/consumption identity mismatch"
                    )
            readback_sha256 = _canonical_json_sha256(evidence)
            dispatch_row["n_plus_one_verified"] = True
            dispatch_row["n_plus_one_verified_sim_step"] = expected_step
            dispatch_row["n_plus_one_readback_sha256"] = readback_sha256
            if consumption_row is not None:
                consumption_row["n_plus_one_verified"] = True
                consumption_row["n_plus_one_verified_sim_step"] = expected_step
                consumption_row[
                    "n_plus_one_readback_sha256"
                ] = readback_sha256
            completion_index = pending.get("segment_completion_row_index")
            if completion_index is not None:
                if (
                    type(completion_index) is not int
                    or completion_index < 0
                    or completion_index >= len(self.segment_completion_rows)
                ):
                    raise RuntimeError(
                        "pending segment start readback lacks its completion row"
                    )
                completion_row = self.segment_completion_rows[completion_index]
                if (
                    completion_row.get("start_batch_id") != pending["batch_id"]
                    or completion_row.get("start_command_epoch")
                    != pending["command_epoch"]
                    or completion_row.get("start_readback_verified") is not False
                ):
                    raise RuntimeError(
                        "pending segment start readback identity mismatch"
                    )
                if "physical_command_epoch" in pending and (
                    completion_row.get("start_physical_command_epoch")
                    != pending["physical_command_epoch"]
                ):
                    raise RuntimeError(
                        "pending segment start physical epoch identity mismatch"
                    )
                completion_row["start_readback_verified"] = True
                completion_row["start_readback_verified_sim_step"] = expected_step
                completion_row["start_readback_sha256"] = readback_sha256
            wheel_stop_index = pending.get(
                "segment_completion_wheel_stop_row_index"
            )
            if wheel_stop_index is not None:
                if (
                    type(wheel_stop_index) is not int
                    or wheel_stop_index < 0
                    or wheel_stop_index >= len(self.segment_completion_rows)
                ):
                    raise RuntimeError(
                        "pending wheel-stop readback lacks its completion row"
                    )
                completion_row = self.segment_completion_rows[wheel_stop_index]
                wheel_stop = completion_row.get("wheel_stop")
                if (
                    not isinstance(wheel_stop, dict)
                    or wheel_stop.get("batch_id") != pending["batch_id"]
                    or wheel_stop.get("n_plus_one_verified") is not False
                ):
                    raise RuntimeError(
                        "pending completion wheel-stop readback identity mismatch"
                    )
                wheel_stop["n_plus_one_verified"] = True
                wheel_stop["n_plus_one_verified_sim_step"] = expected_step
                wheel_stop["n_plus_one_readback_sha256"] = readback_sha256
        self.last_target_readback = evidence
        self.last_verified_servo_targets = dict(pending["servo_targets_deg"])
        self.last_verified_wheel_targets = dict(pending["wheel_targets_rad_s"])
        self.last_verified_command_epoch = int(pending["command_epoch"])
        if "physical_command_epoch" in pending:
            self.last_verified_physical_command_epoch = int(
                pending["physical_command_epoch"]
            )
        self.pending_readback = None
        if kind == "boundary":
            self.boundary_readback_verified = True
        elif kind == "safe_stop":
            self.safe_stop_readback = dict(_jsonable(evidence))
            self.safe_stop_readback_sha256 = _canonical_json_sha256(
                self.safe_stop_readback
            )
            self.safe_stop_status = "VERIFIED"
            self.safe_stop_verified = True
        return kind

    def _attach_drive_readback_layers(self, payload: dict[str, Any]) -> None:
        readback = self.last_target_readback
        servo = dict(readback.get("actual_servo_drive_targets_rad", {}) or {})
        wheels = dict(
            readback.get("actual_wheel_drive_targets_rad_s", {}) or {}
        )
        if set(servo) != set(SERVO_JOINT_NAMES) or set(wheels) != set(
            WHEEL_JOINT_NAMES
        ):
            raise RuntimeError("target readback layers are not exact 8+4 maps")
        payload["physx_servo_drive_target_readback_rad"] = servo
        payload["physx_wheel_drive_target_readback_rad_s"] = wheels

    def _observation_payload(self, adapter: Any) -> dict[str, Any]:
        safety_evidence = self._capture_runtime_safety_evidence(adapter)
        sim_step = int(getattr(adapter, "sim_steps", 0) or 0)
        body_stuck_evidence = self._normalize_optional_runtime_bool(
            safety_evidence,
            field="body_stuck",
            expected_sim_step=sim_step,
        )
        active_leg_trapped_evidence = self._normalize_optional_runtime_bool(
            safety_evidence,
            field="active_leg_trapped",
            expected_sim_step=sim_step,
        )
        safety_evidence.update(body_stuck_evidence)
        safety_evidence.update(active_leg_trapped_evidence)
        pose, root_velocity, joint_q, joint_qd, exact = self._root_and_joint_state(adapter)
        centers = self._wheel_centers(adapter)
        obstacle = self._obstacle()
        expected = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
        servo_targets = dict(getattr(adapter, "joint_command_deg", {}) or {})
        applied_servo_targets = dict(
            getattr(adapter, "servo_applied_command_deg", {}) or {}
        )
        wheel_targets = dict(getattr(adapter, "wheel_speeds", {}) or {})
        targets_exact = bool(
            set(servo_targets) == set(SERVO_JOINT_NAMES)
            and set(applied_servo_targets) == set(SERVO_JOINT_NAMES)
            and set(wheel_targets) == set(WHEEL_JOINT_NAMES)
            and all(
                type(value) in (int, float) and math.isfinite(float(value))
                for value in (
                    *servo_targets.values(),
                    *applied_servo_targets.values(),
                    *wheel_targets.values(),
                )
            )
        )
        measured_servo_actual_deg = {
            name: math.degrees(float(joint_q.get(name, float("nan"))))
            for name in SERVO_JOINT_NAMES
        }
        canonical_servo_actual_error_deg: dict[str, float] = {}
        for name in SERVO_JOINT_NAMES:
            try:
                expected_actual = float(
                    adapter.command_to_actual_target_deg(name, servo_targets[name])
                )
                canonical_servo_actual_error_deg[name] = (
                    measured_servo_actual_deg[name] - expected_actual
                )
            except Exception:
                canonical_servo_actual_error_deg[name] = float("nan")
        values = [
            *pose,
            *root_velocity,
            *(joint_q.get(name, float("nan")) for name in expected),
            *(joint_qd.get(name, float("nan")) for name in expected),
        ]
        finite = bool(
            exact
            and targets_exact
            and len(centers) == len(LEGS)
            and all(math.isfinite(float(v)) for v in values)
        )
        roll = pitch = yaw = float("nan")
        if len(pose) == 7 and all(math.isfinite(float(value)) for value in pose[3:7]):
            roll, pitch, yaw = (float(value) for value in quat_wxyz_to_rpy(pose[3:7]))
            self.peak_roll_rad = max(self.peak_roll_rad, abs(roll))
            self.peak_pitch_rad = max(self.peak_pitch_rad, abs(pitch))
        wheel_classes: dict[str, str] = {}
        face_clearance: dict[str, float] = {}
        top_clearance: dict[str, float] = {}
        relative_s = (
            0.0
            if self.started_sim_time_s is None
            else max(0.0, float(getattr(adapter, "sim_time", 0.0)) - self.started_sim_time_s)
        )
        active_leg = ""
        try:
            active_leg = _enum_text(
                self.bundle.graph.get(self.last_macro_state).active_leg
            )
        except Exception:
            active_leg = ""
        if self.last_macro_state != self.active_traversal_state:
            self.active_traversal_state = self.last_macro_state
            self.active_traversal_leg = active_leg
            if active_leg in self.phase_traversal:
                self.phase_traversal[active_leg] = _new_traversal_record()
                self.phase_traversal[active_leg]["phase_entry_s"] = relative_s
                self.phase_traversal[active_leg]["phase_entry_state"] = (
                    self.last_macro_state
                )
        for leg in LEGS:
            center = centers.get(leg)
            if center is None:
                wheel_classes[leg] = "UNKNOWN"
                continue
            contact = classify_wheel_contact(
                WheelObservation(leg=leg, center_w=center),
                obstacle,
                wheel_radius_m=WHEEL_RADIUS_M,
            )
            wheel_classes[leg] = contact.contact_class.value
            face_clearance[leg] = float(contact.front_face_clearance_m)
            top_clearance[leg] = float(contact.clearance_over_top_m)
            record = self.traversal[leg]
            before_cross = record["front_face_crossing_s"] is None
            crossing_now = bool(
                contact.front_face_clearance_m > 0.0 and before_cross
            )
            lifted = bool(
                contact.contact_class.value == "AIR"
                or center[2] - WHEEL_RADIUS_M > obstacle.bottom_z_m + 0.012
            )
            if before_cross and lifted and not crossing_now:
                record["airborne_seen_before_crossing"] = True
                if record["airborne_first_s"] is None:
                    record["airborne_first_s"] = relative_s
            if crossing_now:
                record["front_face_crossing_s"] = relative_s
            if contact.contact_class.value == "TOP" and not record["top_seen"]:
                record["top_seen"] = True
                record["top_first_s"] = relative_s
            if leg == active_leg:
                phase_record = self.phase_traversal[leg]
                phase_before_cross = phase_record["front_face_crossing_s"] is None
                phase_crossing_now = bool(
                    contact.front_face_clearance_m > 0.0 and phase_before_cross
                )
                if phase_before_cross and lifted and not phase_crossing_now:
                    phase_record["airborne_seen_before_crossing"] = True
                    if phase_record["airborne_first_s"] is None:
                        phase_record["airborne_first_s"] = relative_s
                if phase_crossing_now:
                    phase_record["front_face_crossing_s"] = relative_s
                    phase_record["illegal_drive_up"] = not bool(
                        phase_record["airborne_seen_before_crossing"]
                    )
                    if phase_record["illegal_drive_up"]:
                        record["illegal_drive_up"] = True
                if (
                    contact.contact_class.value == "TOP"
                    and not phase_record["top_seen"]
                ):
                    phase_record["top_seen"] = True
                    phase_record["top_first_s"] = relative_s
        violation, limit_available = self._joint_limit_violation(adapter, joint_q)
        support_legs = tuple(
            leg for leg, value in wheel_classes.items() if value not in {"AIR", "UNKNOWN"}
        )
        velocity_stable = bool(
            finite
            and max((abs(float(v)) for v in root_velocity[:3]), default=0.0) <= 0.02
            and max((abs(float(v)) for v in root_velocity[3:]), default=0.0) <= 0.05
            and max((abs(float(v)) for v in joint_qd.values()), default=0.0) <= 0.10
        )
        recoverable = bool(
            math.isfinite(roll)
            and math.isfinite(pitch)
            and abs(roll) < math.radians(70.0)
            and abs(pitch) < math.radians(70.0)
            and len(support_legs) >= 2
        )
        stability = (
            "stable"
            if recoverable and velocity_stable
            else "recoverable"
            if recoverable
            else "fallen"
            if math.isfinite(roll)
            and math.isfinite(pitch)
            and (abs(roll) >= math.radians(85.0) or abs(pitch) >= math.radians(85.0))
            else "unknown"
        )
        position = pose[:3] if len(pose) >= 3 else [float("nan")] * 3
        body_crossed = bool(position[0] > obstacle.front_face_x_m)
        payload = {
            "schema_version": TELEMETRY_SCHEMA,
            "source_version": self.request.source_version,
            "sim_time_s": float(getattr(adapter, "sim_time", 0.0) or 0.0),
            "sim_step": sim_step,
            "root_pose_w": pose,
            "root_position_w": pose[:3],
            "root_orientation_wxyz": pose[3:7],
            "root_linear_velocity_w": root_velocity[:3],
            "root_angular_velocity_w": root_velocity[3:6],
            "base_position_m": {"x": position[0], "y": position[1], "z": position[2]},
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "base_yaw_rad": yaw,
            "joint_q_rad": joint_q,
            "joint_qd_rad_s": joint_qd,
            "joint_position_rad": joint_q,
            "joint_velocity_rad_s": joint_qd,
            # These are deliberately separate layers.  The canonical endpoint
            # is the source command, the applied command is the adapter's
            # 150-deg/s slew output, and the measured/error maps come from the
            # completed physics step rather than either target buffer.
            "servo_targets_deg": servo_targets,
            "canonical_servo_endpoint_targets_deg": servo_targets,
            "applied_servo_drive_command_deg": applied_servo_targets,
            "measured_servo_actual_deg": measured_servo_actual_deg,
            "canonical_servo_actual_error_deg": (
                canonical_servo_actual_error_deg
            ),
            "wheel_targets_rad_s": wheel_targets,
            # Set true only after this same physics-step payload is paired
            # with independent PhysX target readback in on_step()/start().
            "actuator_targets_applied": False,
            "actuator_target_source": "PENDING_PHYSX_DRIVE_READBACK",
            "dispatch_error": "",
            "wheel_center_w_m": centers,
            "wheel_center_w": centers,
            "wheel_contact_classes": wheel_classes,
            "wheel_contact_class": wheel_classes,
            "wheel_front_face_clearance_m": face_clearance,
            "wheel_top_clearance_m": top_clearance,
            "wheel_contact_load_n": {leg: None for leg in LEGS},
            "support_legs": support_legs,
            "geometry_support_candidate_count": len(support_legs),
            "obstacle_front_face_x_m": obstacle.front_face_x_m,
            "obstacle_top_z_m": obstacle.top_z_m,
            "obstacle_rear_face_x_m": obstacle.rear_face_x_m,
            "body_crossed_front_face": body_crossed,
            "robot_state_finite": finite,
            "robot_fell": stability == "fallen",
            **body_stuck_evidence,
            "dangerous_collision": safety_evidence.get(
                "dangerous_body_collision"
            ),
            "dangerous_body_collision": safety_evidence.get(
                "dangerous_body_collision"
            ),
            "severe_penetration": safety_evidence.get("severe_penetration"),
            "runtime_safety_evidence": safety_evidence,
            "joint_limit_violation": violation,
            "joint_limit_evidence_available": limit_available,
            "unsafe_joint_target": self.target_audit.get("unsafe") is True,
            **active_leg_trapped_evidence,
            "wheel_drive_up_without_required_lift": any(
                record["illegal_drive_up"]
                for record in self.phase_traversal.values()
            ),
            "active_traversal_leg": active_leg,
            "phase_traversal": self.phase_traversal,
            "final_recoverable": recoverable,
            "posture_complete": bool(
                stability == "stable"
                and len(wheel_classes) == len(LEGS)
                and all(value == "TOP" for value in wheel_classes.values())
            ),
            "stability_state": stability,
            "final_velocity_stable": velocity_stable,
            "filtered_contact_bank_enabled": False,
            "wheel_contact_load_available": False,
            "com_measurement_available": False,
            "com_proxy_source": "ROOT_POSITION_AND_WHEEL_GEOMETRY",
        }
        if self.residual_enabled:
            payload["nominal_servo_targets_deg"] = dict(
                self.last_servo_targets
                if self.last_servo_targets is not None
                else servo_targets
            )
            payload["nominal_wheel_targets_rad_s"] = dict(
                self.last_wheel_targets
                if self.last_wheel_targets is not None
                else wheel_targets
            )
            payload["applied_command_servo_targets_deg"] = dict(servo_targets)
            payload["applied_command_wheel_targets_rad_s"] = dict(wheel_targets)
            payload["nominal_command_epoch"] = self.last_epoch
            payload["physical_command_epoch"] = self.physical_command_epoch
            payload["last_verified_physical_command_epoch"] = (
                self.last_verified_physical_command_epoch
            )
            payload["direct_command_residual"] = dict(
                _jsonable(self.last_residual_transform)
            )
        active_spec = (
            self.segment_completion_executor.spec
            if self.active_segment_completion_row_index is not None
            else None
        )
        active_targets = (
            {}
            if active_spec is None
            else dict(active_spec.servo_targets_deg)
        )
        if self.residual_enabled and active_spec is not None:
            active_row = self._active_completion_row()
            raw_effective_targets = active_row.get(
                "effective_completion_servo_targets_deg"
            )
            if not isinstance(raw_effective_targets, Mapping):
                raise RuntimeError(
                    "active residual completion lacks effective target evidence"
                )
            active_targets = dict(raw_effective_targets)
        active_errors: dict[str, float] = {}
        if active_spec is not None:
            for name, target in active_targets.items():
                try:
                    active_errors[name] = measured_servo_actual_deg[name] - float(
                        adapter.command_to_actual_target_deg(name, target)
                    )
                except Exception:
                    active_errors[name] = float("nan")
        payload["active_completion_source_segment_index"] = (
            None if active_spec is None else active_spec.segment_index
        )
        payload["active_completion_sparse_servo_targets_deg"] = (
            active_targets
        )
        if self.residual_enabled:
            payload["active_completion_nominal_sparse_servo_targets_deg"] = (
                {}
                if active_spec is None
                else dict(active_spec.servo_targets_deg)
            )
        payload["active_completion_sparse_servo_actual_error_deg"] = active_errors
        self.nonfinite_state_detected = self.nonfinite_state_detected or not finite
        self.joint_limit_violation_detected = bool(
            self.joint_limit_violation_detected or violation is True
        )
        return payload

    def _validate_hard_safety(self, payload: Mapping[str, Any], *, startup: bool = False) -> None:
        root_writes = getattr(self.adapter, "root_state_write_count", None)
        if type(root_writes) is not int or root_writes != 0:
            raise RuntimeError("normal Macro FSM observed root_state_write_count drift")
        runtime_safety = dict(payload.get("runtime_safety_evidence", {}) or {})
        if runtime_safety.get("available") is True:
            self._validate_deployment_safety_evidence(
                runtime_safety,
                expected_sim_step=int(payload.get("sim_step", -1)),
                require_clear=False,
            )
        else:
            for name in ("dangerous_body_collision", "severe_penetration"):
                value = runtime_safety.get(name)
                if value is not None and type(value) is not bool:
                    raise RuntimeError(f"runtime safety evidence {name} is malformed")
        self.dangerous_collision_detected = bool(
            self.dangerous_collision_detected
            or payload.get("dangerous_body_collision") is True
        )
        self.severe_penetration_detected = bool(
            self.severe_penetration_detected
            or payload.get("severe_penetration") is True
        )
        for field in ("body_stuck", "active_leg_trapped"):
            value = payload.get(field)
            available = payload.get(f"{field}_available")
            source = payload.get(f"{field}_source")
            sample_step = payload.get(f"{field}_sample_sim_step")
            if type(available) is not bool:
                raise RuntimeError(
                    f"runtime {field} availability must be exact bool"
                )
            if type(source) is not str or not source:
                raise RuntimeError(f"runtime {field} source is missing")
            if type(sample_step) is not int or sample_step != payload.get(
                "sim_step"
            ):
                raise RuntimeError(f"runtime {field} sample step mismatch")
            if available:
                if type(value) is not bool:
                    raise RuntimeError(
                        f"runtime {field} must be exact bool when available"
                    )
            elif value is not None:
                raise RuntimeError(
                    f"runtime {field} must remain None when unavailable"
                )
        self.body_stuck_detected = bool(
            self.body_stuck_detected or payload.get("body_stuck") is True
        )
        self.active_leg_trapped_detected = bool(
            self.active_leg_trapped_detected
            or payload.get("active_leg_trapped") is True
        )
        if payload.get("robot_state_finite") is not True:
            raise RuntimeError("non-finite or shape-invalid robot state")
        if payload.get("actuator_targets_applied") is not True:
            raise RuntimeError("production adapter target snapshot is not writable")
        if payload.get("joint_limit_evidence_available") is not True:
            raise RuntimeError("joint-limit evidence is unavailable")
        if payload.get("joint_limit_violation") is True:
            raise RuntimeError("measured joint-limit violation")
        if payload.get("wheel_drive_up_without_required_lift") is True:
            raise RuntimeError("active leg crossed without phase-local lift evidence")
        if str(payload.get("stability_state", "")) == "fallen":
            raise RuntimeError("robot fall detected")
        if payload.get("body_stuck") is True:
            raise RuntimeError("body stuck detected by live runtime producer")
        if payload.get("active_leg_trapped") is True:
            raise RuntimeError(
                "active leg trapped detected by live runtime producer"
            )
        if payload.get("dangerous_body_collision") is True:
            raise RuntimeError("dangerous body collision detected")
        if payload.get("severe_penetration") is True:
            raise RuntimeError("severe penetration detected")

    def _seed_phase_from_transition(
        self, state: str, payload: Mapping[str, Any]
    ) -> None:
        """Bind a new state's lift evidence to its exact transition tick."""

        active_leg = ""
        try:
            active_leg = _enum_text(self.bundle.graph.get(state).active_leg)
        except Exception:
            active_leg = ""
        self.active_traversal_state = state
        self.active_traversal_leg = active_leg
        if active_leg not in self.phase_traversal:
            return
        relative_s = (
            0.0
            if self.started_sim_time_s is None
            else max(
                0.0,
                float(payload.get("sim_time_s", 0.0)) - self.started_sim_time_s,
            )
        )
        record = _new_traversal_record()
        record["phase_entry_s"] = relative_s
        record["phase_entry_state"] = state
        classes = dict(payload.get("wheel_contact_classes", {}) or {})
        face = dict(payload.get("wheel_front_face_clearance_m", {}) or {})
        top = dict(payload.get("wheel_top_clearance_m", {}) or {})
        try:
            crossed_now = float(face.get(active_leg)) > 0.0
        except (TypeError, ValueError):
            crossed_now = False
        try:
            top_clearance = float(top.get(active_leg))
        except (TypeError, ValueError):
            top_clearance = float("-inf")
        airborne_now = bool(
            str(classes.get(active_leg, "") or "").upper() == "AIR"
            or top_clearance >= 0.003
        )
        if airborne_now and not crossed_now:
            record["airborne_seen_before_crossing"] = True
            record["airborne_first_s"] = relative_s
        if crossed_now:
            record["front_face_crossing_s"] = relative_s
            record["illegal_drive_up"] = not bool(
                record["airborne_seen_before_crossing"]
            )
            if record["illegal_drive_up"]:
                self.traversal[active_leg]["illegal_drive_up"] = True
        if str(classes.get(active_leg, "") or "").upper() == "TOP" and crossed_now:
            record["top_seen"] = True
            record["top_first_s"] = relative_s
        self.phase_traversal[active_leg] = record

    def _active_completion_row(self) -> dict[str, Any]:
        index = self.active_segment_completion_row_index
        if (
            type(index) is not int
            or index < 0
            or index >= len(self.segment_completion_rows)
        ):
            raise RuntimeError("active segment-completion ledger identity is invalid")
        row = self.segment_completion_rows[index]
        if row.get("terminal_kind"):
            raise RuntimeError("terminal segment remained active in the worker")
        return row

    @staticmethod
    def _servo_tracking_completion_evidence(
        adapter: Any, targets: Mapping[str, float]
    ) -> dict[str, Any]:
        method = getattr(adapter, "servo_tracking_completion_evidence", None)
        if not callable(method):
            raise RuntimeError(
                "production servo tracking completion evidence is unavailable"
            )
        evidence = method(targets)
        if not isinstance(evidence, Mapping):
            raise RuntimeError(
                "production servo tracking completion evidence is malformed"
            )
        return _jsonable(dict(evidence))

    def _segment_feedback_maps(
        self,
        *,
        adapter: Any,
        payload: Mapping[str, Any],
        spec: SegmentCompletionSpec,
        effective_targets: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        targets = dict(
            spec.servo_targets_deg
            if effective_targets is None
            else effective_targets
        )
        if set(targets) != set(spec.servo_targets_deg):
            raise RuntimeError(
                "effective completion target names differ from the nominal sparse spec"
            )
        if not targets:
            return {}, {}, {"supported": True, "converged": True, "joints": {}}
        actual = payload.get("measured_servo_actual_deg")
        joint_qd = payload.get("joint_qd_rad_s")
        if not isinstance(actual, Mapping) or not isinstance(joint_qd, Mapping):
            raise RuntimeError("measured servo state is unavailable")
        errors: dict[str, float] = {}
        velocities: dict[str, float] = {}
        for name, command_target in targets.items():
            if name not in actual or name not in joint_qd:
                raise RuntimeError(
                    f"measured completion state lacks sparse target joint {name}"
                )
            measured = float(actual[name])
            velocity = math.degrees(float(joint_qd[name]))
            expected = float(
                adapter.command_to_actual_target_deg(name, command_target)
            )
            error = measured - expected
            if not all(math.isfinite(value) for value in (measured, velocity, expected, error)):
                raise RuntimeError(
                    f"measured completion state is non-finite at {name}"
                )
            errors[name] = error
            velocities[name] = velocity
        if set(errors) != set(targets) or set(velocities) != set(targets):
            raise RuntimeError("completion measurement keys are not exact")
        tracking = self._servo_tracking_completion_evidence(adapter, targets)
        return errors, velocities, tracking

    @staticmethod
    def _completion_tracking_targets(row: Mapping[str, Any]) -> dict[str, float]:
        raw = row.get(
            "effective_completion_servo_targets_deg",
            dict(row.get("completion_spec", {}) or {}).get(
                "servo_targets_deg", {}
            ),
        )
        if not isinstance(raw, Mapping):
            raise RuntimeError("segment completion tracking targets are malformed")
        return {str(name): float(value) for name, value in raw.items()}

    def _end_segment_tracking(
        self,
        *,
        adapter: Any,
        row: dict[str, Any],
        sim_step: int,
        sim_time_s: float,
        reason: str,
    ) -> None:
        targets = self._completion_tracking_targets(row)
        if row.get("tracking_lifecycle_closed") is True or (
            targets and row.get("tracking_end_attempt_count") != 0
        ):
            raise RuntimeError("segment servo tracking ended more than once")
        if targets:
            method = getattr(adapter, "end_servo_tracking", None)
            if not callable(method):
                raise RuntimeError("production end_servo_tracking is unavailable")
            row["tracking_end_attempt_count"] = 1
            try:
                result = method(targets)
            except Exception as exc:
                row["tracking_end_evidence"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ended": None,
                }
                raise
            if (
                not isinstance(result, Mapping)
                or type(result.get("ended")) is not bool
            ):
                raise RuntimeError(
                    "end_servo_tracking must return an explicit boolean ended result"
                )
            evidence = _jsonable(dict(result))
            evidence["tracking_completion_deferred"] = result.get("ended") is False
            end_count = 1
        else:
            evidence = {
                "ended": True,
                "required": False,
                "reason": "segment has no sparse servo endpoints",
            }
            end_count = 0
        row["tracking_end_count"] = end_count
        row["tracking_lifecycle_closed"] = True
        row["tracking_end_sim_step"] = int(sim_step)
        row["tracking_end_sim_time_s"] = float(sim_time_s)
        row["tracking_end_reason"] = str(reason)
        row["tracking_end_evidence"] = evidence

    def _completion_token(
        self,
        *,
        row: Mapping[str, Any],
        decision_mapping: Mapping[str, Any],
    ) -> Any:
        from .fsm50_macro_controller import MacroSegmentCompletionToken

        decision = dict(_jsonable(decision_mapping))
        token = MacroSegmentCompletionToken(
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
            decision_sha256=_canonical_json_sha256(decision),
        )
        # Reparse the serialized boundary so controller and worker do not share
        # an in-memory object as their only identity check.
        return MacroSegmentCompletionToken.from_mapping(token.to_mapping())

    def _observe_active_segment_completion(
        self,
        *,
        adapter: Any,
        payload: Mapping[str, Any],
    ) -> Any | None:
        if self.active_segment_completion_row_index is None:
            return None
        if self.pending_readback is not None:
            raise RuntimeError(
                "segment completion cannot precede exact N+1 target readback"
            )
        row = self._active_completion_row()
        if row.get("start_readback_verified") is not True:
            raise RuntimeError(
                "active segment lacks exact new or retained target readback"
            )
        spec = self.segment_completion_executor.spec
        if spec is None:
            raise RuntimeError("active completion row lacks its shared executor")
        sim_step = int(payload.get("sim_step", -1))
        sim_time = float(payload.get("sim_time_s", float("nan")))
        if (
            self.last_completion_observation_sim_step is not None
            and sim_step
            <= self.last_completion_observation_sim_step
        ):
            raise RuntimeError("segment completion feedback did not advance")
        if not self.outer_render_boundary_permit:
            raise RuntimeError("segment completion observed off the render boundary")
        errors, velocities, tracking = self._segment_feedback_maps(
            adapter=adapter,
            payload=payload,
            spec=spec,
            effective_targets=self._completion_tracking_targets(row),
        )
        elapsed = max(
            0.0,
            sim_time
            - float(
                self.started_sim_time_s
                if self.started_sim_time_s is not None
                else sim_time
            ),
        )
        decision = self.segment_completion_executor.observe(
            SegmentFeedback(
                elapsed_s=elapsed,
                sim_time_s=sim_time,
                sim_step=sim_step,
                servo_errors_deg=errors,
                servo_velocity_deg_s=velocities,
                tracking_evidence=tracking,
            )
        )
        decision_mapping = decision.to_mapping()
        self.last_completion_observation_sim_step = sim_step
        row["observation_decisions"].append(decision_mapping)
        row["last_decision"] = decision_mapping
        row["last_decision_sha256"] = _canonical_json_sha256(decision_mapping)
        if decision.kind in {
            SegmentDecisionKind.COMPLETE,
            SegmentDecisionKind.FAIL,
        }:
            self._end_segment_tracking(
                adapter=adapter,
                row=row,
                sim_step=sim_step,
                sim_time_s=sim_time,
                reason=decision.kind.value,
            )
            row["terminal_kind"] = decision.kind.value
            row["terminal_sim_step"] = sim_step
            row["terminal_sim_time_s"] = sim_time
            row["terminal_decision_sha256"] = row["last_decision_sha256"]
            self.active_segment_completion_row_index = None
            self.active_completion_latched_servo_residual_deg = {}
        return self._completion_token(row=row, decision_mapping=decision_mapping)

    @staticmethod
    def _validated_completion_control(
        mapping: Mapping[str, Any],
    ) -> tuple[dict[str, Any], SegmentCompletionSpec | None]:
        raw = mapping.get("segment_completion_control")
        if not isinstance(raw, Mapping) or set(raw) != set(
            _SEGMENT_COMPLETION_CONTROL_KEYS
        ):
            raise RuntimeError(
                "Macro decision segment_completion_control schema is not exact"
            )
        control = dict(_jsonable(raw))
        if control.get("schema_version") != SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION:
            raise RuntimeError("Macro segment completion control schema is invalid")
        kind = control.get("kind")
        if kind not in {"NONE", "START", "WHEEL_STOP"}:
            raise RuntimeError("Macro segment completion control kind is invalid")
        if kind == "NONE":
            if not _strict_json_equal(control, _EMPTY_SEGMENT_COMPLETION_CONTROL):
                raise RuntimeError("NONE segment completion control is not empty")
            return control, None
        if type(control.get("source_action")) is not bool:
            raise RuntimeError("segment completion source_action must be exact bool")
        for field in (
            "profile_id",
            "profile_source_version",
            "owner_state",
            "source_plan_sha256",
            "source_plan_payload_sha256",
            "accepted_steps_sha256",
        ):
            value = control.get(field)
            if type(value) is not str or not value:
                raise RuntimeError(f"segment completion control {field} is missing")
        for field in (
            "source_plan_sha256",
            "source_plan_payload_sha256",
            "accepted_steps_sha256",
        ):
            digest = control[field]
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise RuntimeError(f"segment completion control {field} is invalid")
        for field in (
            "source_segment_index",
            "source_step_index",
            "start_command_epoch",
        ):
            if type(control.get(field)) is not int or int(control[field]) < 0:
                raise RuntimeError(f"segment completion control {field} is invalid")
        if not isinstance(control.get("source_step_id"), str):
            raise RuntimeError("segment completion control source_step_id is invalid")
        try:
            spec = SegmentCompletionSpec.from_mapping(
                control.get("completion_spec", {})
            )
        except Exception as exc:
            raise RuntimeError("segment completion control spec is invalid") from exc
        if (
            spec.segment_index != control["source_segment_index"]
            or spec.source_step != control["source_step_index"]
            or spec.source_step_id != control["source_step_id"]
        ):
            raise RuntimeError("segment completion control/spec identity mismatch")
        source_identity = control.get("source_action_identity")
        token_sha = control.get("completion_token_sha256")
        if kind == "START":
            if (
                control["source_action"] is not True
                or type(source_identity) is not str
                or len(source_identity) != 64
                or token_sha != ""
            ):
                raise RuntimeError("START segment completion provenance is invalid")
        elif (
            type(token_sha) is not str
            or len(token_sha) != 64
            or any(character not in "0123456789abcdef" for character in token_sha)
            or (
                control["source_action"] is True
                and (
                    type(source_identity) is not str
                    or len(source_identity) != 64
                )
            )
            or (control["source_action"] is False and source_identity != "")
        ):
            raise RuntimeError("WHEEL_STOP segment completion provenance is invalid")
        control["completion_spec"] = spec.to_mapping()
        return control, spec

    def _validate_segment_completion_control(
        self,
        *,
        mapping: Mapping[str, Any],
        state: str,
        epoch: int,
        provenance: Mapping[str, Any],
        consumption_row: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], SegmentCompletionSpec | None]:
        control, spec = self._validated_completion_control(mapping)
        kind = control["kind"]
        dispatch_kind = str(provenance.get("dispatch_kind", "") or "")
        provenance_kind = str(provenance.get("kind", "") or "")
        if kind == "NONE":
            if provenance_kind == "SOURCE_ACTION" and dispatch_kind in {
                "segment_start",
                "wheel_channel_completion_stop",
            }:
                raise RuntimeError(
                    "recorded segment action lacks its completion control"
                )
            if provenance_kind == "COMPLETION_WHEEL_STOP":
                raise RuntimeError("dynamic wheel stop lacks its completion control")
            return control, spec
        if not self.outer_render_boundary_permit:
            raise RuntimeError(
                "segment completion control is forbidden off the render boundary"
            )
        if control["owner_state"] != state:
            raise RuntimeError("segment completion owner state differs from decision")
        if control["profile_source_version"] != self.request.source_version:
            raise RuntimeError("segment completion source version drifted")
        if kind == "START":
            if self.active_segment_completion_row_index is not None:
                raise RuntimeError("a new source segment started before completion")
            if (
                consumption_row is None
                or provenance_kind != "SOURCE_ACTION"
                or dispatch_kind != "segment_start"
            ):
                raise RuntimeError("START completion control lacks its source action")
            action_index = consumption_row.get("source_action_index")
            if (
                type(action_index) is not int
                or action_index < 0
                or action_index >= len(self.expected_source_actions)
            ):
                raise RuntimeError("START completion action index is invalid")
            expected = self.expected_source_actions[action_index]
            binding = expected.get("segment_completion_binding")
            if not isinstance(binding, Mapping):
                raise RuntimeError("START source action lacks a sealed segment binding")
            expected_fields = {
                "profile_id": expected["profile_id"],
                "profile_source_version": expected["profile_source_version"],
                "owner_state": expected["owner_state"],
                "source_plan_sha256": expected["source_plan_sha256"],
                "source_plan_payload_sha256": binding[
                    "source_plan_payload_sha256"
                ],
                "accepted_steps_sha256": binding["accepted_steps_sha256"],
                "source_segment_index": provenance["source_segment_index"],
                "source_step_index": provenance["source_step_index"],
                "source_step_id": binding["completion_spec"]["source_step_id"],
                "start_command_epoch": epoch,
                "completion_spec": binding["completion_spec"],
                "source_action_identity": provenance["source_action_identity"],
                "source_action": True,
                "completion_token_sha256": "",
            }
            for field, expected_value in expected_fields.items():
                if not _strict_json_equal(control[field], expected_value):
                    raise RuntimeError(
                        f"START completion control {field} differs from sealed binding"
                    )
            return control, spec

        row = self._active_completion_row()
        token = self.last_segment_completion_token
        if token is None or str(getattr(token, "kind", "")) != "WHEEL_STOP_DUE":
            raise RuntimeError("WHEEL_STOP control lacks the due completion token")
        token_mapping = token.to_mapping()
        token_sha = _canonical_json_sha256(token_mapping)
        for field, expected_value in (
            ("profile_id", row["profile_id"]),
            ("profile_source_version", row["profile_source_version"]),
            ("owner_state", row["owner_state"]),
            ("source_plan_sha256", row["source_plan_sha256"]),
            ("source_plan_payload_sha256", row["source_plan_payload_sha256"]),
            ("accepted_steps_sha256", row["accepted_steps_sha256"]),
            ("source_segment_index", row["source_segment_index"]),
            ("source_step_index", row["source_step_index"]),
            ("source_step_id", row["source_step_id"]),
            ("start_command_epoch", row["start_command_epoch"]),
            ("completion_spec", row["completion_spec"]),
            ("completion_token_sha256", token_sha),
        ):
            if not _strict_json_equal(control[field], expected_value):
                raise RuntimeError(
                    f"WHEEL_STOP completion control {field} identity mismatch"
                )
        if control["source_action"] is True:
            if (
                consumption_row is None
                or provenance_kind != "SOURCE_ACTION"
                or dispatch_kind != "wheel_channel_completion_stop"
                or control["source_action_identity"]
                != provenance.get("source_action_identity")
            ):
                raise RuntimeError(
                    "source WHEEL_STOP control lacks its exact source action"
                )
        elif (
            consumption_row is not None
            or provenance_kind != "COMPLETION_WHEEL_STOP"
            or control["source_action_identity"] != ""
        ):
            raise RuntimeError(
                "dynamic WHEEL_STOP must use independent generated provenance"
            )
        return control, spec

    @staticmethod
    def _effective_servo_reference_velocity_deg_s(adapter: Any) -> float:
        reference = getattr(adapter, "motion_reference", None)
        try:
            requested_raw = reference.servo_reference_velocity_deg_s
            limit_raw = reference.servo_velocity_limit_deg_s
        except Exception as exc:
            raise RuntimeError(
                "adapter effective servo reference velocity is unavailable"
            ) from exc
        if (
            type(requested_raw) not in (int, float)
            or not math.isfinite(float(requested_raw))
            or float(requested_raw) <= 0.0
        ):
            raise RuntimeError(
                "adapter servo reference velocity must be finite and positive"
            )
        requested = float(requested_raw)
        if limit_raw is None:
            rate = requested
        else:
            if (
                type(limit_raw) not in (int, float)
                or not math.isfinite(float(limit_raw))
                or float(limit_raw) <= 0.0
            ):
                raise RuntimeError(
                    "adapter servo velocity limit must be finite and positive or None"
                )
            rate = min(requested, float(limit_raw))
        if not math.isclose(
            rate,
            EXPECTED_SERVO_REFERENCE_VELOCITY_DEG_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "Macro completion requires the unchanged effective 150 deg/s "
                f"servo reference; actual={rate!r}"
            )
        return rate

    def _start_segment_completion(
        self,
        *,
        adapter: Any,
        payload: Mapping[str, Any],
        mapping: Mapping[str, Any],
        control: Mapping[str, Any],
        spec: SegmentCompletionSpec,
        consumption_row: dict[str, Any],
        pre_action_servos: Mapping[str, float],
        applied_servos: Mapping[str, float],
        pre_action_applied_servos: Mapping[str, float],
        physical_command_epoch: int,
        dispatch_ack: Mapping[str, Any] | None,
    ) -> None:
        sparse_targets = dict(spec.servo_targets_deg)
        full_targets = self._validated_target_map(
            mapping.get("servo_targets_deg"),
            names=SERVO_JOINT_NAMES,
            label="segment-start canonical servo endpoints",
        )
        if not set(sparse_targets).issubset(SERVO_JOINT_NAMES) or any(
            not math.isclose(
                float(full_targets[name]),
                float(target),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for name, target in sparse_targets.items()
        ):
            raise RuntimeError(
                "segment completion sparse endpoint differs from source command"
            )
        previous = self._validated_target_map(
            pre_action_servos,
            names=SERVO_JOINT_NAMES,
            label="pre-segment canonical servo endpoints",
        )
        applied_full_targets = self._validated_target_map(
            applied_servos,
            names=SERVO_JOINT_NAMES,
            label="segment-start applied servo targets",
        )
        previous_applied = self._validated_target_map(
            pre_action_applied_servos,
            names=SERVO_JOINT_NAMES,
            label="pre-segment applied servo targets",
        )
        effective_sparse_targets = {
            name: applied_full_targets[name] for name in sparse_targets
        }
        rate = self._effective_servo_reference_velocity_deg_s(adapter)
        maximum_delta = max(
            [
                abs(float(target) - float(previous_applied[name]))
                for name, target in effective_sparse_targets.items()
            ]
            or [0.0]
        )
        dynamic_duration = maximum_delta / rate
        sim_step = int(payload["sim_step"])
        sim_time = float(payload["sim_time_s"])
        elapsed = max(
            0.0,
            sim_time
            - float(
                self.started_sim_time_s
                if self.started_sim_time_s is not None
                else sim_time
            ),
        )
        begin_evidence: dict[str, Any]
        if effective_sparse_targets:
            begin = getattr(adapter, "begin_servo_tracking", None)
            if not callable(begin):
                raise RuntimeError("production begin_servo_tracking is unavailable")
            begin(effective_sparse_targets)
            begin_count = 1
            begin_evidence = {
                "called": True,
                "sparse_joint_names": sorted(effective_sparse_targets),
            }
        else:
            begin_count = 0
            begin_evidence = {
                "called": False,
                "sparse_joint_names": [],
            }
        try:
            self.segment_completion_executor.start(
                spec,
                start_elapsed_s=elapsed,
                start_sim_time_s=sim_time,
                start_sim_step=sim_step,
                servo_duration_s_override=dynamic_duration,
            )
        except Exception:
            if effective_sparse_targets:
                end = getattr(adapter, "end_servo_tracking", None)
                if callable(end):
                    end(effective_sparse_targets)
            raise
        physical_dispatch = dispatch_ack is not None
        retained_sha = (
            ""
            if physical_dispatch
            else _canonical_json_sha256(self.last_target_readback)
        )
        effective_spec = self.segment_completion_executor.spec.to_mapping()
        if self.residual_enabled:
            effective_spec["servo_targets_deg"] = dict(
                effective_sparse_targets
            )
        row = {
            "schema_version": SEGMENT_COMPLETION_SCHEMA,
            "segment_completion_index": len(self.segment_completion_rows),
            "source_version": self.request.source_version,
            "profile_id": control["profile_id"],
            "profile_source_version": control["profile_source_version"],
            "owner_state": control["owner_state"],
            "source_plan_sha256": control["source_plan_sha256"],
            "source_plan_payload_sha256": control[
                "source_plan_payload_sha256"
            ],
            "accepted_steps_sha256": control["accepted_steps_sha256"],
            "source_segment_index": spec.segment_index,
            "source_step_index": spec.source_step,
            "source_step_id": spec.source_step_id,
            "completion_spec": spec.to_mapping(),
            "effective_completion_spec": effective_spec,
            "dynamic_servo_duration_s": dynamic_duration,
            "effective_servo_reference_velocity_deg_s": rate,
            "pre_action_canonical_servo_targets_deg": dict(previous),
            "start_source_action_identity": control["source_action_identity"],
            "source_action_consumption_index": consumption_row[
                "source_action_index"
            ],
            "start_command_epoch": control["start_command_epoch"],
            "start_sim_step": sim_step,
            "start_sim_time_s": sim_time,
            "start_physical_dispatch": physical_dispatch,
            "start_batch_id": (
                "" if dispatch_ack is None else str(dispatch_ack.get("batch_id", ""))
            ),
            "start_first_physics_step": (
                None
                if dispatch_ack is None
                else int(dispatch_ack["first_physics_step"])
            ),
            "start_readback_verified": not physical_dispatch,
            "start_readback_verified_sim_step": (
                int(self.last_target_readback.get("sim_step", -1))
                if not physical_dispatch
                else None
            ),
            "start_readback_sha256": retained_sha,
            "retained_epoch_same_target": not physical_dispatch,
            "tracking_begin_count": begin_count,
            "tracking_begin_sim_step": sim_step,
            "tracking_begin_evidence": begin_evidence,
            "tracking_end_count": 0,
            "tracking_end_attempt_count": 0,
            "tracking_lifecycle_closed": False,
            "observation_decisions": [],
            "last_decision": {},
            "last_decision_sha256": "",
            "wheel_stop": None,
            "terminal_kind": "",
            "terminal_sim_step": None,
            "terminal_sim_time_s": None,
            "terminal_decision_sha256": "",
        }
        if self.residual_enabled:
            latched_residual = {
                name: float(effective_sparse_targets[name])
                - float(sparse_targets[name])
                for name in sparse_targets
            }
            row.update(
                {
                    "effective_completion_servo_targets_deg": dict(
                        effective_sparse_targets
                    ),
                    "effective_completion_targets_sha256": (
                        _canonical_json_sha256(effective_sparse_targets)
                    ),
                    "nominal_completion_spec_sha256": (
                        _canonical_json_sha256(spec.to_mapping())
                    ),
                    "pre_action_applied_servo_targets_deg": dict(
                        previous_applied
                    ),
                    "start_physical_command_epoch": int(
                        physical_command_epoch
                    ),
                    "latched_servo_residual_deg": dict(latched_residual),
                    "nominal_target_changed": bool(
                        consumption_row.get("target_changed")
                    ),
                    "applied_target_changed": bool(
                        consumption_row.get("applied_target_changed")
                    ),
                }
            )
        self.segment_completion_rows.append(row)
        self.active_segment_completion_row_index = int(
            row["segment_completion_index"]
        )
        self.last_completion_observation_sim_step = None
        if self.residual_enabled:
            self.active_completion_latched_servo_residual_deg = dict(
                row["latched_servo_residual_deg"]
            )
        if physical_dispatch:
            if self.pending_readback is None:
                raise RuntimeError(
                    "physical segment start lacks pending N+1 readback"
                )
            self.pending_readback["segment_completion_row_index"] = (
                self.active_segment_completion_row_index
            )

    def _acknowledge_segment_wheel_stop(
        self,
        *,
        payload: Mapping[str, Any],
        control: Mapping[str, Any],
        consumption_row: Mapping[str, Any] | None,
        dispatch_ack: Mapping[str, Any] | None,
    ) -> None:
        row = self._active_completion_row()
        if row.get("wheel_stop") is not None:
            raise RuntimeError("segment wheel stop was dispatched more than once")
        if dispatch_ack is None:
            raise RuntimeError("segment wheel stop lacks a physical 8+4 dispatch")
        applied_step = int(dispatch_ack["applied_sim_step"])
        first_step = int(dispatch_ack["first_physics_step"])
        batch_id = str(dispatch_ack["batch_id"])
        self.segment_completion_executor.acknowledge_wheel_stop(
            applied_sim_step=applied_step,
            first_physics_step=first_step,
            batch_id=batch_id,
        )
        row["wheel_stop"] = {
            "generated": control["source_action"] is False,
            "source_action": control["source_action"],
            "source_action_consumption_index": (
                None
                if consumption_row is None
                else consumption_row["source_action_index"]
            ),
            "completion_token_sha256": control["completion_token_sha256"],
            "applied_sim_step": applied_step,
            "first_physics_step": first_step,
            "batch_id": batch_id,
            "n_plus_one_verified": False,
            "n_plus_one_verified_sim_step": None,
            "n_plus_one_readback_sha256": "",
        }
        if self.pending_readback is None:
            raise RuntimeError("segment wheel stop lacks pending N+1 readback")
        self.pending_readback["segment_completion_wheel_stop_row_index"] = int(
            row["segment_completion_index"]
        )

    def _decision_mapping(self, decision: Any) -> dict[str, Any]:
        if hasattr(decision, "to_mapping") and callable(decision.to_mapping):
            raw = decision.to_mapping()
        elif is_dataclass(decision):
            raw = asdict(decision)
        elif isinstance(decision, Mapping):
            raw = dict(decision)
        else:
            raw = {
                name: getattr(decision, name)
                for name in (
                    "macro_state",
                    "subphase",
                    "profile_id",
                    "phase_elapsed_s",
                    "profile_fraction",
                    "servo_targets_deg",
                    "wheel_targets_rad_s",
                    "command_epoch",
                    "command_changed",
                    "source_action_consumed",
                    "target_changed",
                    "command_provenance",
                    "segment_completion_control",
                    "transition_events",
                    "reason",
                    "terminal",
                    "terminal_outcome",
                )
                if hasattr(decision, name)
            }
        result = dict(_jsonable(raw) or {})
        for name in ("macro_state", "subphase", "profile_id", "terminal_outcome"):
            result[name] = _enum_text(_attribute(decision, name, result.get(name, "")))
        return result

    @staticmethod
    def _validated_policy_action(value: Any) -> tuple[float, ...]:
        if isinstance(value, (str, bytes)):
            raise RuntimeError("residual policy action must be a 12-D numeric sequence")
        try:
            action = tuple(value)
        except TypeError as exc:
            raise RuntimeError(
                "residual policy action must be a 12-D numeric sequence"
            ) from exc
        if len(action) != len(RESIDUAL_ACTION_NAMES):
            raise RuntimeError(
                "residual policy action must follow the exact canonical 12-D order"
            )
        result: list[float] = []
        for index, raw in enumerate(action):
            if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                raise RuntimeError(
                    f"residual policy action[{index}] must be finite numeric data"
                )
            result.append(float(raw))
        return tuple(result)

    def _compose_applied_targets(
        self,
        *,
        adapter: Any,
        payload: Mapping[str, Any],
        mapping: Mapping[str, Any],
        provenance: Mapping[str, Any],
        completion_control: Mapping[str, Any],
        nominal_servos: Mapping[str, float],
        nominal_wheels: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any] | None]:
        """Return the sole physical 8+4 target map for one nominal decision.

        The disabled path is deliberately a direct copy of the already
        validated controller maps.  An enabled policy is sampled only at an
        outer render boundary; unchanged off-boundary controller decisions
        retain the previously applied physical command without invoking the
        provider, policy, or residual composer.
        """

        nominal_servo_map = dict(nominal_servos)
        nominal_wheel_map = dict(nominal_wheels)
        if not self.residual_enabled:
            return nominal_servo_map, nominal_wheel_map, None

        previous_servos = self.last_applied_servo_targets
        previous_wheels = self.last_applied_wheel_targets
        if previous_servos is None or previous_wheels is None:
            raise RuntimeError("residual physical target state is unavailable")
        if not self.outer_render_boundary_permit:
            if self.last_epoch is not None and (
                not _strict_json_equal(
                    nominal_servo_map, self.last_servo_targets
                )
                or not _strict_json_equal(
                    nominal_wheel_map, self.last_wheel_targets
                )
            ):
                raise RuntimeError(
                    "residual target changes require the current outer render boundary"
                )
            return dict(previous_servos), dict(previous_wheels), None
        if self.pending_readback is not None:
            raise RuntimeError(
                "residual transform cannot replace a pending N+1 readback"
            )

        profile_strategy = str(mapping.get("profile_strategy", "") or "")
        state = str(mapping.get("macro_state", "") or "")
        subphase = str(mapping.get("subphase", "") or "")
        context = {
            "schema_version": RESIDUAL_POLICY_OBSERVATION_SCHEMA,
            "source_version": self.request.source_version,
            "profile_id": str(mapping.get("profile_id", "") or ""),
            "profile_source_version": str(
                mapping.get("profile_source_version", "") or ""
            ),
            "profile_strategy": profile_strategy,
            "macro_state": state,
            "subphase": subphase,
            "sim_step": int(payload.get("sim_step", -1)),
            "sim_time_s": float(payload.get("sim_time_s", float("nan"))),
            "outer_render_cycle_index": self.outer_render_cycle_index,
            "outer_render_boundary_permit": True,
            "nominal_command_epoch": int(mapping["command_epoch"]),
            "physical_command_epoch": self.physical_command_epoch,
            "nominal_servo_targets_deg": nominal_servo_map,
            "nominal_wheel_targets_rad_s": nominal_wheel_map,
            "previous_applied_servo_targets_deg": dict(previous_servos),
            "previous_applied_wheel_targets_rad_s": dict(previous_wheels),
            "previous_applied_residual": list(self.last_applied_residual),
            "active_completion_latched_servo_residual_deg": dict(
                self.active_completion_latched_servo_residual_deg
            ),
            "command_provenance": dict(provenance),
            "segment_completion_control": dict(completion_control),
        }
        provider = self.residual_contract_provider
        if provider is None:
            raise RuntimeError("residual contract provider is unavailable")
        contract = provider(dict(_jsonable(context)))
        if not isinstance(contract, ResidualPhaseContract):
            raise RuntimeError(
                "residual contract provider must return ResidualPhaseContract"
            )
        provenance_kind = str(provenance.get("kind", "") or "")
        if provenance_kind == "SOURCE_ACTION" and not profile_strategy:
            raise RuntimeError(
                "SOURCE_ACTION residual transform requires its non-empty "
                "bound profile strategy"
            )
        contract_profile_strategy = (
            profile_strategy or contract.profile_strategy
        )
        if contract.profile_strategy != contract_profile_strategy:
            raise RuntimeError(
                "residual phase contract profile strategy differs from the "
                "verified Macro decision context"
            )
        for label, actual, expected in (
            (
                "source version",
                contract.source_version,
                self.request.source_version,
            ),
            ("Macro state", contract.macro_state, state),
            ("subphase", contract.subphase, subphase),
        ):
            if actual != expected:
                raise RuntimeError(
                    f"residual phase contract {label} differs from the "
                    "verified Macro decision context"
                )
        policy_observation = dict(_jsonable(context))
        policy_observation["contract"] = contract.to_mapping()
        policy_observation["contract_sha256"] = contract.sha256
        policy_observation["worker_observation"] = dict(_jsonable(payload))
        action = self._validated_policy_action(
            self.residual_policy.act(policy_observation)
        )
        terminal = mapping.get("terminal") is True
        stop_kinds = {
            "BOUNDARY_ZERO_WHEELS",
            "HOLD_ZERO_WHEELS",
            "SAFE_STOP_ZERO_WHEELS",
            "SUCCESS_ZERO_WHEELS",
            "COMPLETION_WHEEL_STOP",
        }
        force_zero_residual = terminal
        force_zero_wheels = bool(
            not terminal
            and (
                provenance_kind in stop_kinds
                or completion_control.get("kind") == "WHEEL_STOP"
            )
            and all(value == 0.0 for value in nominal_wheel_map.values())
        )
        transform = compose_direct_command_residual(
            ResidualTransformInput(
                source_version=self.request.source_version,
                profile_strategy=contract_profile_strategy,
                macro_state=state,
                subphase=subphase,
                nominal_servo_targets_deg=nominal_servo_map,
                nominal_wheel_targets_rad_s=nominal_wheel_map,
                normalized_action=action,
                previous_applied_residual=self.last_applied_residual,
                decision_dt_s=(
                    float(self.physics_dt_s) * EXPECTED_RENDER_SUBSTEPS
                ),
                maximum_wheel_speed_rad_s=float(adapter.max_wheel_speed),
                latched_servo_residual_deg=(
                    self.active_completion_latched_servo_residual_deg
                ),
                force_zero_residual=force_zero_residual,
                force_zero_wheels=force_zero_wheels,
            ),
            contract,
        )
        applied_servos = dict(transform.applied_servo_targets_deg)
        applied_wheels = dict(transform.applied_wheel_targets_rad_s)
        self._validate_runtime_targets(adapter, applied_servos, applied_wheels)
        evidence = transform.to_mapping()
        evidence.update(
            {
                "policy_id": self.residual_policy_id,
                "policy_sha256": self.residual_policy_sha256,
                "core_transform_sha256": transform.sha256,
                "nominal_command_epoch": int(mapping["command_epoch"]),
                "physical_command_epoch_before": self.physical_command_epoch,
                "force_zero_residual": force_zero_residual,
                "force_zero_wheels": force_zero_wheels,
            }
        )
        evidence["evidence_sha256"] = _canonical_json_sha256(evidence)
        return applied_servos, applied_wheels, evidence

    def _validate_coalesced_transition_source_action(
        self,
        *,
        mapping: Mapping[str, Any],
        state: str,
        provenance: Mapping[str, Any],
        expected: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        """Allow only a final-COMPLETE, zero-boundary, next-state START.

        The controller may combine the logical EXIT/ENTER with the next
        state's first canonical segment start only when the old state's final
        measured completion left an otherwise empty outer-cycle slot.  The
        worker proves that no zero-wheel boundary batch was elided and that the
        sole possible batch belongs to the new source action.
        """

        previous_state = self.last_macro_state
        sim_step = payload.get("sim_step")
        raw_events = mapping.get("transition_events", ())
        events = list(raw_events) if isinstance(raw_events, (list, tuple)) else []
        if (
            not self.outer_render_boundary_permit
            or type(sim_step) is not int
            or sim_step < 0
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION requires the current outer boundary"
            )
        if (
            not previous_state
            or state == previous_state
            or mapping.get("terminal") is not False
            or str(mapping.get("terminal_outcome", "") or "") != "RUNNING"
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION must be a nonterminal state entry"
            )
        try:
            expected_next_state = _enum_text(
                self.bundle.graph.get(previous_state).next_state
            )
        except Exception as exc:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks its sealed graph edge"
            ) from exc
        if state != expected_next_state or events != [
            f"EXIT:{previous_state}",
            f"ENTER:{state}",
        ]:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION differs from its adjacent graph edge"
            )

        if provenance.get("dispatch_kind") != "segment_start":
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION must be a segment_start"
            )
        state_action_indices = [
            index
            for index, action in enumerate(self.expected_source_actions)
            if action.get("owner_state") == state
        ]
        if (
            not state_action_indices
            or state_action_indices[0] != self.next_source_action_index
            or expected.get("owner_state") != state
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION is not the next state's first action"
            )
        previous_action_indices = [
            index
            for index, action in enumerate(self.expected_source_actions)
            if action.get("owner_state") == previous_state
        ]
        previous_segment_starts = [
            action
            for action in self.expected_source_actions
            if action.get("owner_state") == previous_state
            and action.get("command_provenance", {}).get("dispatch_kind")
            == "segment_start"
        ]
        if (
            not previous_action_indices
            or previous_action_indices[-1] != self.next_source_action_index - 1
            or not previous_segment_starts
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks a final old-state profile action"
            )

        raw_control = mapping.get("segment_completion_control")
        if not isinstance(raw_control, Mapping) or any(
            not _strict_json_equal(raw_control.get(key), value)
            for key, value in {
                "kind": "START",
                "owner_state": state,
                "source_action": True,
                "source_action_identity": provenance.get("source_action_identity"),
                "source_segment_index": provenance.get("source_segment_index"),
                "source_step_index": provenance.get("source_step_index"),
            }.items()
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks its next-state START control"
            )

        if self.pending_readback is not None:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION cannot replace a pending readback"
            )
        if self.last_batch_attempt_sim_step == sim_step:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION cannot share an occupied batch slot"
            )
        if self.active_segment_completion_row_index is not None:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION requires the old segment to be closed"
            )

        expected_verified_servos = (
            self.last_applied_servo_targets
            if self.residual_enabled
            else self.last_servo_targets
        )
        expected_verified_wheels = (
            self.last_applied_wheel_targets
            if self.residual_enabled
            else self.last_wheel_targets
        )
        if (
            self.last_epoch is None
            or self.last_verified_command_epoch != self.last_epoch
            or self.last_servo_targets is None
            or self.last_wheel_targets is None
            or expected_verified_servos is None
            or expected_verified_wheels is None
            or self.last_verified_servo_targets is None
            or self.last_verified_wheel_targets is None
            or not _strict_json_equal(
                self.last_verified_servo_targets, expected_verified_servos
            )
            or not _strict_json_equal(
                self.last_verified_wheel_targets, expected_verified_wheels
            )
            or (
                self.residual_enabled
                and self.last_verified_physical_command_epoch
                != self.physical_command_epoch
            )
            or any(
                abs(float(value)) > 1.0e-12
                for value in self.last_wheel_targets.values()
            )
            or any(
                abs(float(value)) > 1.0e-12
                for value in expected_verified_wheels.values()
            )
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks a verified zero-wheel boundary"
            )
        readback = self.last_target_readback
        if (
            not isinstance(readback, Mapping)
            or readback.get("sim_step") != sim_step
            or not _strict_json_equal(
                readback.get("canonical_servo_targets_deg"),
                expected_verified_servos,
            )
            or not _strict_json_equal(
                readback.get("canonical_wheel_targets_rad_s"),
                expected_verified_wheels,
            )
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks current retained-target readback"
            )

        token = self.last_segment_completion_token
        if token is None or not callable(getattr(token, "to_mapping", None)):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks its current COMPLETE token"
            )
        try:
            token_mapping = dict(token.to_mapping())
            token_kind = str(getattr(token, "kind"))
            token_step = int(getattr(token, "sim_step"))
        except Exception as exc:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION COMPLETE token is malformed"
            ) from exc
        final_old_segment = previous_segment_starts[-1]["command_provenance"][
            "source_segment_index"
        ]
        if (
            token_kind != SegmentDecisionKind.COMPLETE.value
            or token_step != sim_step
            or token_mapping.get("owner_state") != previous_state
            or token_mapping.get("source_segment_index") != final_old_segment
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks the old final COMPLETE"
            )
        if not self.segment_completion_rows:
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION lacks its completed segment row"
            )
        completion_row = self.segment_completion_rows[-1]
        token_row_fields = (
            "profile_id",
            "profile_source_version",
            "owner_state",
            "source_plan_sha256",
            "source_plan_payload_sha256",
            "accepted_steps_sha256",
            "source_segment_index",
            "source_step_index",
            "source_step_id",
            "start_command_epoch",
            "start_sim_step",
            "start_readback_sha256",
        )
        if (
            any(
                not _strict_json_equal(
                    completion_row.get(field), token_mapping.get(field)
                )
                for field in token_row_fields
            )
            or completion_row.get("terminal_kind")
            != SegmentDecisionKind.COMPLETE.value
            or completion_row.get("terminal_sim_step") != sim_step
            or completion_row.get("tracking_lifecycle_closed") is not True
            or completion_row.get("terminal_decision_sha256")
            != token_mapping.get("decision_sha256")
            or completion_row.get("last_decision_sha256")
            != token_mapping.get("decision_sha256")
            or not _strict_json_equal(
                completion_row.get("last_decision"), token_mapping.get("decision")
            )
        ):
            raise RuntimeError(
                "coalesced transition SOURCE_ACTION completion row is not the token's closed COMPLETE"
            )

    def _validate_decision_provenance(
        self,
        *,
        mapping: Mapping[str, Any],
        state: str,
        epoch: int,
        changed: bool,
        targets_changed: bool,
        servos: Mapping[str, float],
        wheels: Mapping[str, float],
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        consumed = mapping.get("source_action_consumed")
        target_changed = mapping.get("target_changed")
        if type(consumed) is not bool:
            raise RuntimeError("Macro decision source_action_consumed must be bool")
        if type(target_changed) is not bool:
            raise RuntimeError("Macro decision target_changed must be bool")
        if target_changed != targets_changed or changed != target_changed:
            raise RuntimeError(
                "Macro decision target_changed/command_changed differs from full target maps"
            )
        raw_provenance = mapping.get("command_provenance")
        if not isinstance(raw_provenance, Mapping):
            raise RuntimeError("Macro decision command_provenance must be a mapping")
        provenance = dict(raw_provenance)
        if set(provenance) != set(_COMMAND_PROVENANCE_KEYS):
            raise RuntimeError("Macro decision command_provenance keys are not exact")
        kind = provenance.get("kind")
        if type(kind) is not str or kind not in _COMMAND_PROVENANCE_KINDS:
            raise RuntimeError("Macro decision command_provenance kind is invalid")

        if kind != "SOURCE_ACTION":
            expected_empty = dict(_EMPTY_COMMAND_PROVENANCE)
            expected_empty["kind"] = kind
            if consumed is not False:
                raise RuntimeError(
                    "non-source command provenance cannot consume a source action"
                )
            if not _strict_json_equal(provenance, expected_empty):
                raise RuntimeError("non-source command provenance fields must be empty")
            if (kind == "NONE") != (target_changed is False):
                raise RuntimeError(
                    "target-changing non-source decisions require exact command provenance"
                )
            if target_changed:
                if self.last_servo_targets is None or not _strict_json_equal(
                    dict(servos), dict(self.last_servo_targets)
                ):
                    raise RuntimeError(
                        "non-source zero-wheel dispatch must retain all servo targets"
                    )
                if any(float(value) != 0.0 for value in wheels.values()):
                    raise RuntimeError(
                        "non-source zero-wheel dispatch must zero all wheel targets"
                    )
                raw_events = mapping.get("transition_events")
                if not isinstance(raw_events, (list, tuple)) or any(
                    type(event) is not str or not event for event in raw_events
                ):
                    raise RuntimeError(
                        "non-source zero-wheel dispatch lacks exact transition events"
                    )
                previous_state = self.last_macro_state
                events = list(raw_events)
                terminal = mapping.get("terminal") is True
                outcome = str(mapping.get("terminal_outcome", "") or "")
                if kind == "COMPLETION_WHEEL_STOP":
                    valid_context = bool(
                        previous_state
                        and state == previous_state
                        and terminal is False
                        and not events
                        and self.active_segment_completion_row_index is not None
                        and self.last_segment_completion_token is not None
                        and str(
                            getattr(
                                self.last_segment_completion_token, "kind", ""
                            )
                        )
                        == "WHEEL_STOP_DUE"
                    )
                elif kind == "BOUNDARY_ZERO_WHEELS":
                    valid_context = bool(
                        previous_state
                        and state != previous_state
                        and terminal is False
                        and events
                        == [f"EXIT:{previous_state}", f"ENTER:{state}"]
                    )
                elif kind == "HOLD_ZERO_WHEELS":
                    valid_context = bool(
                        previous_state
                        and state == previous_state
                        and terminal is False
                        and events == [f"HOLD:{state}"]
                    )
                elif kind == "SAFE_STOP_ZERO_WHEELS":
                    valid_context = bool(
                        previous_state
                        and state == "SAFE_STOP"
                        and terminal
                        and outcome == "SAFE_STOP"
                        and events == [f"SAFE_STOP:{previous_state}"]
                    )
                elif kind == "SUCCESS_ZERO_WHEELS":
                    valid_context = bool(
                        previous_state
                        and state == "SUCCESS"
                        and terminal
                        and _is_task_success_outcome(outcome)
                        and events
                        == [f"EXIT:{previous_state}", "ENTER:SUCCESS"]
                    )
                else:  # kind NONE was rejected above for a changed target map.
                    valid_context = False
                if not valid_context:
                    raise RuntimeError(
                        f"{kind} provenance differs from its exact state/event context"
                    )
            return provenance, None

        if consumed is not True:
            raise RuntimeError("SOURCE_ACTION provenance must consume exactly one action")
        self._validate_source_action_provenance_shape(provenance)
        claimed_identity = provenance["source_action_identity"]
        rebuilt_identity = _source_action_identity(provenance)
        if claimed_identity != rebuilt_identity:
            raise RuntimeError("source-action provenance identity hash mismatch")
        if self.next_source_action_index >= len(self.expected_source_actions):
            raise RuntimeError("unexpected extra/duplicate source action")
        expected = self.expected_source_actions[self.next_source_action_index]
        if not _strict_json_equal(
            provenance, expected["command_provenance"]
        ):
            expected_segment = expected["command_provenance"][
                "source_segment_index"
            ]
            raise RuntimeError(
                "source action is duplicate, missing, malformed, or out of order; "
                f"expected canonical segment {expected_segment}"
            )
        if state != expected["owner_state"]:
            raise RuntimeError("source action executed outside its canonical owner state")
        transition_events = list(mapping.get("transition_events", ()) or ())
        if state != self.last_macro_state or transition_events:
            self._validate_coalesced_transition_source_action(
                mapping=mapping,
                state=state,
                provenance=provenance,
                expected=expected,
                payload=payload,
            )
        for field, expected_field in (
            ("profile_id", expected["profile_id"]),
            ("profile_source_version", expected["profile_source_version"]),
            ("profile_strategy", expected["profile_strategy"]),
        ):
            if mapping.get(field) != expected_field:
                raise RuntimeError(
                    f"source action {field} differs from the canonical bundle"
                )
        if not _strict_json_equal(dict(servos), expected["servo_targets_deg"]):
            raise RuntimeError(
                "source action servo targets differ from the canonical bundle"
            )
        if not _strict_json_equal(dict(wheels), expected["wheel_targets_rad_s"]):
            raise RuntimeError(
                "source action wheel targets differ from the canonical bundle"
            )
        if not target_changed:
            if self.pending_readback is not None:
                raise RuntimeError(
                    "same-target source action cannot inherit a pending readback"
                )
            if self.last_verified_command_epoch != epoch:
                raise RuntimeError(
                    "same-target source action lacks verified retained command epoch"
                )
            expected_verified_servos = (
                self.last_applied_servo_targets
                if self.residual_enabled
                else servos
            )
            expected_verified_wheels = (
                self.last_applied_wheel_targets
                if self.residual_enabled
                else wheels
            )
            if (
                expected_verified_servos is None
                or expected_verified_wheels is None
                or self.last_verified_servo_targets is None
                or self.last_verified_wheel_targets is None
                or not _strict_json_equal(
                    dict(self.last_verified_servo_targets),
                    dict(expected_verified_servos),
                )
                or not _strict_json_equal(
                    dict(self.last_verified_wheel_targets),
                    dict(expected_verified_wheels),
                )
                or (
                    self.residual_enabled
                    and self.last_verified_physical_command_epoch
                    != self.physical_command_epoch
                )
            ):
                raise RuntimeError(
                    "same-target source action differs from verified retained targets"
                )
            if not self.last_target_readback:
                raise RuntimeError(
                    "same-target source action lacks exact N+1 target readback"
                )

        row = {
            "schema_version": SOURCE_ACTION_CONSUMPTION_SCHEMA,
            "source_action_index": self.next_source_action_index,
            "expected_source_action_count": len(self.expected_source_actions),
            "sim_time_s": payload.get("sim_time_s"),
            "sim_step": payload.get("sim_step"),
            "macro_state": state,
            "subphase": mapping.get("subphase", ""),
            "profile_id": expected["profile_id"],
            "profile_source_version": expected["profile_source_version"],
            "profile_strategy": expected["profile_strategy"],
            "source_plan_sha256": expected["source_plan_sha256"],
            "profile_library_sha256": self.request.profile_library_sha256,
            "bundle_sha256": self.request.bundle_sha256,
            "command_provenance": dict(provenance),
            "servo_targets_deg": dict(servos),
            "wheel_targets_rad_s": dict(wheels),
            "target_changed": target_changed,
            "dispatch_epoch": epoch,
            "physical_dispatch_required": target_changed,
            "physical_dispatch_applied": False,
            "physical_dispatch_index": None,
            "batch_id": "",
            "n_plus_one_verified": False,
            "n_plus_one_verified_sim_step": None,
            "n_plus_one_readback_sha256": "",
            "pre_action_verified_command_epoch": self.last_verified_command_epoch,
            "pre_action_verified_readback_sha256": (
                _canonical_json_sha256(self.last_target_readback)
                if self.last_target_readback
                else ""
            ),
        }
        return provenance, row

    def _process_decision(
        self,
        adapter: Any,
        payload: Mapping[str, Any],
        decision: Any,
    ) -> None:
        mapping = self._decision_mapping(decision)
        state = str(mapping.get("macro_state", "") or "")
        if not state or not str(mapping.get("subphase", "") or ""):
            raise RuntimeError("Macro decision state/subphase identity is incomplete")
        for name in ("phase_elapsed_s", "profile_fraction"):
            value = mapping.get(name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise RuntimeError(f"Macro decision {name} must be finite")
        fraction = float(mapping["profile_fraction"])
        if fraction < -1.0e-9 or fraction > 1.0 + 1.0e-9:
            raise RuntimeError("Macro decision profile_fraction is outside 0..1")
        if type(mapping.get("terminal")) is not bool:
            raise RuntimeError("Macro decision terminal must be bool")
        if not str(mapping.get("terminal_outcome", "") or ""):
            raise RuntimeError("Macro decision terminal_outcome is missing")
        if not isinstance(mapping.get("reason", ""), str):
            raise RuntimeError("Macro decision reason must be a string")
        retry_count = mapping.get("retry_count", 0)
        if type(retry_count) is not int or retry_count < 0:
            raise RuntimeError("Macro decision retry_count must be a non-negative int")
        epoch_raw = mapping.get("command_epoch")
        if type(epoch_raw) is not int or epoch_raw < 0:
            raise RuntimeError("Macro decision has an invalid command_epoch")
        epoch = int(epoch_raw)
        changed = mapping.get("command_changed")
        if type(changed) is not bool:
            raise RuntimeError("Macro decision command_changed must be bool")
        servos = self._validated_target_map(
            mapping.get("servo_targets_deg"),
            names=SERVO_JOINT_NAMES,
            label="servo_targets_deg",
        )
        wheels = self._validated_target_map(
            mapping.get("wheel_targets_rad_s"),
            names=WHEEL_JOINT_NAMES,
            label="wheel_targets_rad_s",
        )
        pre_action_servos = dict(
            self.last_servo_targets
            if self.last_servo_targets is not None
            else self.boundary_ack.get("servo_targets_applied", {}) or {}
        )
        pre_action_applied_servos = dict(
            self.last_applied_servo_targets
            if self.last_applied_servo_targets is not None
            else self.boundary_ack.get("servo_targets_applied", {}) or {}
        )
        if self.last_epoch is None:
            boundary_servos = dict(
                self.boundary_ack.get("servo_targets_applied", {}) or {}
            )
            boundary_wheels = dict(
                self.boundary_ack.get("wheel_targets_applied", {}) or {}
            )
            if (
                not self.boundary_readback_verified
                or epoch != 0
                or changed is not False
                or servos != boundary_servos
                or wheels != boundary_wheels
            ):
                raise RuntimeError(
                    "controller reset decision must be epoch=0, unchanged, and "
                    "equal the independently verified zero-boundary snapshot"
                )
            targets_changed = servos != boundary_servos or wheels != boundary_wheels
        else:
            if changed and epoch != self.last_epoch + 1:
                raise RuntimeError("changed Macro command epoch must increase by exactly one")
            if not changed and epoch != self.last_epoch:
                raise RuntimeError("unchanged Macro command unexpectedly changed epoch")
            targets_changed = servos != self.last_servo_targets or wheels != self.last_wheel_targets
        provenance, consumption_row = self._validate_decision_provenance(
            mapping=mapping,
            state=state,
            epoch=epoch,
            changed=changed,
            targets_changed=targets_changed,
            servos=servos,
            wheels=wheels,
            payload=payload,
        )
        completion_control, completion_spec = (
            self._validate_segment_completion_control(
                mapping=mapping,
                state=state,
                epoch=epoch,
                provenance=provenance,
                consumption_row=consumption_row,
            )
        )
        # reset() may intentionally publish the verified adapter snapshot at
        # epoch zero without dispatching it again.  The nominal epoch/target
        # relationship remains controller-owned; the optional transform below
        # separately decides whether the final physical map needs one batch.

        previous_state = self.last_macro_state
        raw_events = mapping.get("transition_events", ())
        if not isinstance(raw_events, (list, tuple)) or any(
            not isinstance(event, str) or not event
            for event in raw_events
        ):
            raise RuntimeError("Macro transition_events must be a list of non-empty strings")
        events = list(raw_events)
        guard_evidence = mapping.get("guard_evidence", {})
        if not isinstance(guard_evidence, Mapping):
            raise RuntimeError("Macro guard_evidence must be a mapping")
        if previous_state and state != previous_state and not events:
            raise RuntimeError("Macro state changed without transition evidence")
        applied_servos, applied_wheels, residual_transform = (
            self._compose_applied_targets(
                adapter=adapter,
                payload=payload,
                mapping=mapping,
                provenance=provenance,
                completion_control=completion_control,
                nominal_servos=servos,
                nominal_wheels=wheels,
            )
        )
        previous_applied_servos = dict(self.last_applied_servo_targets or {})
        previous_applied_wheels = dict(self.last_applied_wheel_targets or {})
        physical_changed = (
            bool(changed)
            if not self.residual_enabled
            else bool(
                not _strict_json_equal(
                    applied_servos, previous_applied_servos
                )
                or not _strict_json_equal(
                    applied_wheels, previous_applied_wheels
                )
            )
        )
        current_step = int(payload["sim_step"])
        if physical_changed and (
            self.pending_readback is not None
            or self.last_batch_attempt_sim_step == current_step
        ):
            raise RuntimeError(
                "physical command dispatch requires one empty callback batch slot"
            )
        next_physical_epoch = self.physical_command_epoch + int(
            physical_changed
        )
        if consumption_row is not None and self.residual_enabled:
            consumption_row.update(
                {
                    "nominal_target_changed": changed,
                    "applied_target_changed": physical_changed,
                    "physical_dispatch_required": physical_changed,
                    "physical_command_epoch": next_physical_epoch,
                    "applied_servo_targets_deg": dict(applied_servos),
                    "applied_wheel_targets_rad_s": dict(applied_wheels),
                    "residual_transform_sha256": (
                        ""
                        if residual_transform is None
                        else str(
                            residual_transform["core_transform_sha256"]
                        )
                    ),
                }
            )
        if state != previous_state:
            self._seed_phase_from_transition(state, payload)
            active = self.active_traversal_leg
            if (
                active in self.phase_traversal
                and self.phase_traversal[active]["illegal_drive_up"]
            ):
                raise RuntimeError(
                    "active leg crossed at state entry without live phase-local lift evidence"
                )
        if events or (previous_state and state != previous_state):
            self.transition_rows.append(
                {
                    "schema_version": TRANSITION_SCHEMA,
                    "transition_index": len(self.transition_rows),
                    "sim_time_s": payload.get("sim_time_s"),
                    "sim_step": payload.get("sim_step"),
                    "from_state": previous_state,
                    "to_state": state,
                    "subphase": mapping.get("subphase", ""),
                    "profile_id": mapping.get("profile_id", ""),
                    "profile_source_version": mapping.get(
                        "profile_source_version", self.request.source_version
                    ),
                    "profile_strategy": mapping.get("profile_strategy", ""),
                    "command_epoch": epoch,
                    "phase_elapsed_s": mapping.get("phase_elapsed_s"),
                    "profile_fraction": mapping.get("profile_fraction"),
                    "events": events,
                    "reason": mapping.get("reason", ""),
                    "guard_evidence": dict(guard_evidence),
                    "retry_count": mapping.get("retry_count", 0),
                    "observation_sha256": _canonical_json_sha256(payload),
                }
            )
        if consumption_row is not None:
            self.source_action_consumption_rows.append(consumption_row)
            self.next_source_action_index += 1
        dispatch_ack: dict[str, Any] | None = None
        if physical_changed:
            self._validate_runtime_targets(
                adapter, applied_servos, applied_wheels
            )
            batch_id = (
                f"{self.request.request_id}:macro:{epoch:06d}"
                if changed
                else (
                    f"{self.request.request_id}:residual:"
                    f"{next_physical_epoch:06d}"
                )
            )
            recording_metadata = {
                "source_version": self.request.source_version,
                "profile_id": mapping.get("profile_id", self.request.profile_id),
                "profile_source_version": mapping.get(
                    "profile_source_version", self.request.source_version
                ),
                "profile_strategy": mapping.get("profile_strategy", ""),
                "macro_state": state,
                "subphase": mapping.get("subphase", ""),
                "command_epoch": epoch,
                "bundle_sha256": self.request.bundle_sha256,
                "command_provenance": dict(provenance),
                "segment_completion_control": dict(completion_control),
                "source_plan_sha256": (
                    ""
                    if consumption_row is None
                    else str(consumption_row["source_plan_sha256"])
                ),
                "source_action_consumption_index": (
                    None
                    if consumption_row is None
                    else int(consumption_row["source_action_index"])
                ),
            }
            if self.residual_enabled:
                recording_metadata.update(
                    {
                        "nominal_command_epoch": epoch,
                        "physical_command_epoch": next_physical_epoch,
                        "nominal_target_changed": changed,
                        "applied_target_changed": True,
                        "dispatch_cause": (
                            "NOMINAL_AND_RESIDUAL"
                            if changed
                            else "RESIDUAL_ONLY"
                        ),
                        "nominal_servo_targets_deg": dict(servos),
                        "nominal_wheel_targets_rad_s": dict(wheels),
                        "residual_policy_id": self.residual_policy_id,
                        "residual_policy_sha256": self.residual_policy_sha256,
                        "residual_transform_sha256": (
                            ""
                            if residual_transform is None
                            else str(
                                residual_transform["core_transform_sha256"]
                            )
                        ),
                        "residual_evidence_sha256": (
                            ""
                            if residual_transform is None
                            else str(residual_transform["evidence_sha256"])
                        ),
                    }
                )
            self.last_batch_attempt_sim_step = current_step
            ack = dict(
                adapter.apply_motion_batch(
                    {
                        "batch_id": batch_id,
                        "source": "fsm50_macro_controller",
                        "servo_targets_deg": applied_servos,
                        "wheel_targets_rad_s": applied_wheels,
                        "wheel_generation": int(getattr(adapter, "wheel_generation", 0) or 0),
                        "recording_metadata": recording_metadata,
                    }
                )
                or {}
            )
            dispatch_ack = ack
            self._validate_batch_ack(
                ack,
                batch_id=batch_id,
                servo_targets=applied_servos,
                wheel_targets=applied_wheels,
                expected_sim_step=current_step,
                expected_physics_dt_s=float(self.physics_dt_s),
                expected_source="fsm50_macro_controller",
                expected_recording_metadata=recording_metadata,
            )
            dispatch_index = len(self.dispatch_rows)
            self._establish_pending_readback(
                kind="controller",
                batch_id=batch_id,
                ack=ack,
                servo_targets=applied_servos,
                wheel_targets=applied_wheels,
                command_epoch=epoch,
                physical_command_epoch=(
                    next_physical_epoch if self.residual_enabled else None
                ),
                physical_dispatch_index=dispatch_index,
                source_action_consumption_index=(
                    None
                    if consumption_row is None
                    else int(consumption_row["source_action_index"])
                ),
            )
            changed_servos = [
                name
                for name in SERVO_JOINT_NAMES
                if previous_applied_servos.get(name) != applied_servos[name]
            ]
            changed_wheels = [
                name
                for name in WHEEL_JOINT_NAMES
                if previous_applied_wheels.get(name) != applied_wheels[name]
            ]
            dispatch_row = {
                "schema_version": DISPATCH_SCHEMA,
                "dispatch_index": dispatch_index,
                "sim_time_s": payload.get("sim_time_s"),
                "sim_step": current_step,
                "macro_state": state,
                "subphase": mapping.get("subphase", ""),
                "profile_id": mapping.get("profile_id", ""),
                "profile_source_version": mapping.get(
                    "profile_source_version", self.request.source_version
                ),
                "profile_strategy": mapping.get("profile_strategy", ""),
                "command_epoch": epoch,
                "command_provenance": dict(provenance),
                "source_action_consumption_index": (
                    None
                    if consumption_row is None
                    else int(consumption_row["source_action_index"])
                ),
                "servo_targets_deg": applied_servos,
                "wheel_targets_rad_s": applied_wheels,
                "changed_servo_names": changed_servos,
                "changed_wheel_names": changed_wheels,
                "concurrent": bool(changed_servos and changed_wheels),
                "batch_id": batch_id,
                "ack": ack,
                "n_plus_one_verified": False,
                "n_plus_one_verified_sim_step": None,
                "n_plus_one_readback_sha256": "",
            }
            if self.residual_enabled:
                dispatch_row.update(
                    {
                        "nominal_command_epoch": epoch,
                        "physical_command_epoch": next_physical_epoch,
                        "nominal_target_changed": changed,
                        "applied_target_changed": True,
                        "dispatch_cause": (
                            "NOMINAL_AND_RESIDUAL"
                            if changed
                            else "RESIDUAL_ONLY"
                        ),
                        "nominal_servo_targets_deg": dict(servos),
                        "nominal_wheel_targets_rad_s": dict(wheels),
                        "residual_policy_id": self.residual_policy_id,
                        "residual_policy_sha256": self.residual_policy_sha256,
                        "residual_transform": dict(
                            _jsonable(residual_transform or {})
                        ),
                    }
                )
            self.dispatch_rows.append(dispatch_row)
            if consumption_row is not None:
                consumption_row["physical_dispatch_applied"] = True
                consumption_row["physical_dispatch_index"] = dispatch_index
                consumption_row["batch_id"] = batch_id
            self.command_dispatch_count += 1
        self.last_applied_servo_targets = dict(applied_servos)
        self.last_applied_wheel_targets = dict(applied_wheels)
        self.physical_command_epoch = next_physical_epoch
        if residual_transform is not None:
            self.last_applied_residual = tuple(
                float(value)
                for value in residual_transform["applied_residual"]
            )
            self.last_residual_transform = dict(
                _jsonable(residual_transform)
            )
            self.residual_transform_count += 1
        if self.residual_enabled and not physical_changed:
            # The current physical readback also proves a newly named nominal
            # epoch when the final applied map is exactly retained.
            self.last_verified_command_epoch = epoch
        if completion_control["kind"] == "START":
            if completion_spec is None or consumption_row is None:
                raise RuntimeError("START completion control lost its validated inputs")
            self._start_segment_completion(
                adapter=adapter,
                payload=payload,
                mapping=mapping,
                control=completion_control,
                spec=completion_spec,
                consumption_row=consumption_row,
                pre_action_servos=pre_action_servos,
                applied_servos=applied_servos,
                pre_action_applied_servos=pre_action_applied_servos,
                physical_command_epoch=next_physical_epoch,
                dispatch_ack=dispatch_ack,
            )
        elif completion_control["kind"] == "WHEEL_STOP":
            self._acknowledge_segment_wheel_stop(
                payload=payload,
                control=completion_control,
                consumption_row=consumption_row,
                dispatch_ack=dispatch_ack,
            )
        self.last_decision = decision
        self.last_decision_mapping = mapping
        self.last_epoch = epoch
        self.last_servo_targets = servos
        self.last_wheel_targets = wheels
        self.last_macro_state = state

        terminal = mapping.get("terminal") is True
        if terminal:
            outcome = str(mapping.get("terminal_outcome", "") or "")
            reason = str(mapping.get("reason", "") or outcome or "Macro controller terminal")
            if _is_task_success_outcome(outcome):
                coverage_errors = self._successful_coverage_errors()
                if coverage_errors:
                    raise RuntimeError(
                        "Macro task-success terminal lacks exact source-action coverage: "
                        + "; ".join(coverage_errors)
                    )
            self.controller_terminal_outcome = outcome
            self.controller_terminal_reason = reason
            self.terminal_stop_request = {
                "success_after_stop": _is_task_success_outcome(outcome),
                "outcome": outcome,
                "reason": reason,
                "terminal_decision_sim_step": int(payload["sim_step"]),
            }
            if self.pending_readback is not None:
                self.state = "terminal_command_pending_readback"
            else:
                self._begin_safe_stop(adapter, payload)

    @staticmethod
    def _validated_target_map(value: Any, *, names: list[str], label: str) -> dict[str, float]:
        if not isinstance(value, Mapping) or set(value) != set(names):
            raise RuntimeError(f"{label} must contain exactly the canonical names")
        result: dict[str, float] = {}
        for name in names:
            raw = value[name]
            if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                raise RuntimeError(f"{label}.{name} must be finite")
            result[name] = float(raw)
        return result

    def _validate_runtime_targets(
        self,
        adapter: Any,
        servos: Mapping[str, float],
        wheels: Mapping[str, float],
    ) -> None:
        for name, target in servos.items():
            clamped = float(clamp_servo_command(name, target))
            actual = float(adapter.command_to_actual_target_deg(name, target))
            minimum, maximum = (float(value) for value in adapter.get_final_target_limits_deg(name))
            if not math.isclose(clamped, target, rel_tol=0.0, abs_tol=1.0e-9):
                raise RuntimeError(f"Macro servo target would clamp: {name}")
            if not minimum <= actual <= maximum:
                raise RuntimeError(f"Macro servo target exceeds safe limit: {name}")
        maximum_wheel = float(getattr(adapter, "max_wheel_speed"))
        for name, target in wheels.items():
            if abs(target) > maximum_wheel + 1.0e-9:
                raise RuntimeError(f"Macro wheel target exceeds safe limit: {name}")

    @staticmethod
    def _validate_batch_ack(
        ack: Mapping[str, Any],
        *,
        batch_id: str,
        servo_targets: Mapping[str, float],
        wheel_targets: Mapping[str, float],
        expected_sim_step: int,
        expected_physics_dt_s: float,
        expected_source: str,
        expected_recording_metadata: Mapping[str, Any],
    ) -> None:
        if type(ack.get("error")) is not str:
            raise RuntimeError("motion batch ACK error field is not text")
        if ack.get("error"):
            raise RuntimeError("motion batch rejected: " + str(ack.get("error")))
        if ack.get("batch_id") != batch_id:
            raise RuntimeError("motion batch ACK id mismatch")
        if ack.get("source") != expected_source:
            raise RuntimeError("motion batch ACK source mismatch")
        if ack.get("servo_applied") is not True or ack.get("wheel_applied") is not True:
            raise RuntimeError("motion batch ACK does not prove one full atomic 8+4 apply")
        if (
            type(ack.get("applied_sim_step")) is not int
            or ack.get("applied_sim_step") != expected_sim_step
        ):
            raise RuntimeError("motion batch ACK applied_sim_step mismatch")
        if (
            type(ack.get("first_physics_step")) is not int
            or ack.get("first_physics_step") != expected_sim_step + 1
        ):
            raise RuntimeError("motion batch ACK first_physics_step mismatch")
        skew = ack.get("motion_start_skew_s")
        if (
            type(skew) not in (int, float)
            or not math.isfinite(float(skew))
            or float(skew) != 0.0
        ):
            raise RuntimeError("motion batch ACK reports servo/wheel skew")
        raw_ack_dt = ack.get("physics_dt_s")
        if type(raw_ack_dt) not in (int, float) or not math.isfinite(
            float(raw_ack_dt)
        ):
            raise RuntimeError("motion batch ACK physics dt is not finite numeric data")
        ack_dt = float(raw_ack_dt)
        if not math.isclose(
            ack_dt,
            expected_physics_dt_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("motion batch ACK physics dt mismatch")
        for label, raw, expected in (
            ("servo", ack.get("servo_targets_applied"), servo_targets),
            ("wheel", ack.get("wheel_targets_applied"), wheel_targets),
        ):
            if not isinstance(raw, Mapping) or set(raw) != set(expected):
                raise RuntimeError(
                    f"motion batch ACK {label} targets are not a complete exact map"
                )
            for name, expected_value in expected.items():
                value = raw[name]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or not math.isclose(
                        float(value),
                        float(expected_value),
                        rel_tol=0.0,
                        abs_tol=TARGET_COMMAND_TOLERANCE,
                    )
                ):
                    raise RuntimeError(
                        f"motion batch ACK {label} target mismatch at {name}"
                    )
        raw_metadata = ack.get("recording_metadata")
        if not isinstance(raw_metadata, Mapping) or not _strict_json_equal(
            dict(raw_metadata), dict(expected_recording_metadata)
        ):
            raise RuntimeError("motion batch ACK recording metadata mismatch")

    def _sample(
        self,
        payload: Mapping[str, Any],
        *,
        decision: Any | None,
        force: bool = False,
    ) -> None:
        sim_time = float(payload.get("sim_time_s", 0.0) or 0.0)
        if not force and sim_time + 1.0e-9 < self.next_sample_sim_time_s:
            return
        period = 1.0 / self.request.telemetry_hz
        self.next_sample_sim_time_s = max(self.next_sample_sim_time_s + period, sim_time + period)
        mapping = self._decision_mapping(decision) if decision is not None else self.last_decision_mapping
        state = str(mapping.get("macro_state", self.last_macro_state) or self.last_macro_state)
        active_leg = ""
        try:
            active_leg = _enum_text(self.bundle.graph.get(state).active_leg) if state else ""
        except Exception:
            active_leg = ""
        row = {
            **dict(payload),
            "macro_state": state,
            "subphase": mapping.get("subphase", ""),
            "profile_id": mapping.get("profile_id", self.request.profile_id),
            "profile_source_version": mapping.get(
                "profile_source_version", self.request.source_version
            ),
            "profile_strategy": mapping.get("profile_strategy", ""),
            "phase_elapsed_s": mapping.get("phase_elapsed_s"),
            "profile_fraction": mapping.get("profile_fraction"),
            "command_epoch": mapping.get("command_epoch", self.last_epoch),
            "transition_events": list(mapping.get("transition_events", []) or []),
            "active_leg": active_leg,
            "retry_count": mapping.get("retry_count", 0),
            "controller_terminal": mapping.get("terminal", False),
            "controller_terminal_outcome": mapping.get("terminal_outcome", ""),
        }
        self.rows.append(dict(_jsonable(row)))

    def _trusted_servo_hold(self, adapter: Any) -> dict[str, float]:
        for candidate, label in (
            (self.last_verified_servo_targets, "last verified servo targets"),
            (
                dict(self.boundary_ack.get("servo_targets_applied", {}) or {}),
                "zero-boundary servo targets",
            ),
            (getattr(adapter, "joint_command_deg", None), "adapter servo targets"),
        ):
            try:
                return self._validated_target_map(
                    candidate,
                    names=SERVO_JOINT_NAMES,
                    label=label,
                )
            except Exception:
                continue
        raise RuntimeError("no complete finite trusted 8-servo hold snapshot is available")

    def _begin_safe_stop(self, adapter: Any, payload: Mapping[str, Any]) -> None:
        if self.pending_readback is not None and str(
            self.pending_readback.get("kind", "")
        ) == "safe_stop":
            return
        self.safe_stop_count += 1
        self.safe_stop_verified = False
        self.safe_stop_readback = {}
        self.safe_stop_readback_sha256 = ""
        self.safe_stop_status = "APPLYING"
        self.safe_stop_error = ""
        try:
            servos = self._trusted_servo_hold(adapter)
            wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            self._validate_runtime_targets(adapter, servos, wheels)
            current_step = int(payload.get("sim_step", getattr(adapter, "sim_steps", -1)))
            epoch = int(
                self.last_verified_command_epoch
                if self.last_verified_command_epoch is not None
                else self.last_epoch
                if self.last_epoch is not None
                else 0
            )
            batch_id = (
                f"{self.request.request_id}:safe-stop:{self.safe_stop_count:04d}"
            )
            metadata = {
                "source_version": self.request.source_version,
                "profile_id": self.request.profile_id,
                "macro_state": self.last_macro_state,
                "command_epoch": epoch,
                "bundle_sha256": self.request.bundle_sha256,
                "safe_stop": True,
                "safe_stop_reason": str(
                    dict(self.terminal_stop_request or {}).get("reason", self.error)
                    or "Macro safe stop"
                ),
            }
            self.last_batch_attempt_sim_step = current_step
            ack = dict(
                adapter.apply_motion_batch(
                    {
                        "batch_id": batch_id,
                        "source": "fsm50_macro_safe_stop",
                        "servo_targets_deg": servos,
                        "wheel_targets_rad_s": wheels,
                        "wheel_generation": int(
                            getattr(adapter, "wheel_generation", 0) or 0
                        )
                        + 1,
                        "recording_metadata": metadata,
                    }
                )
                or {}
            )
            self._validate_batch_ack(
                ack,
                batch_id=batch_id,
                servo_targets=servos,
                wheel_targets=wheels,
                expected_sim_step=current_step,
                expected_physics_dt_s=float(self.physics_dt_s),
                expected_source="fsm50_macro_safe_stop",
                expected_recording_metadata=metadata,
            )
            self.safe_stop_ack = dict(_jsonable(ack))
            safe_stop_physical_epoch: int | None = None
            if self.residual_enabled:
                safe_stop_physical_epoch = self.physical_command_epoch + 1
                self.physical_command_epoch = safe_stop_physical_epoch
                self.last_applied_servo_targets = dict(servos)
                self.last_applied_wheel_targets = dict(wheels)
                self.last_applied_residual = ZERO_RESIDUAL_ACTION
                self.active_completion_latched_servo_residual_deg = {}
            self._establish_pending_readback(
                kind="safe_stop",
                batch_id=batch_id,
                ack=ack,
                servo_targets=servos,
                wheel_targets=wheels,
                command_epoch=epoch,
                physical_command_epoch=safe_stop_physical_epoch,
            )
            self.safe_stop_status = "PENDING_N_PLUS_1_READBACK"
            self.state = "safe_stop_pending_readback"
        except Exception as exc:
            self.safe_stop_status = "SAFE_STOP_APPLICATION_FAILED"
            self.safe_stop_verified = False
            self.safe_stop_error = f"{type(exc).__name__}: {exc}"
            suffix = "SAFE_STOP_APPLICATION_FAILED: " + self.safe_stop_error
            self.error = "; ".join(part for part in (self.error, suffix) if part)
            self.infrastructure_failure = True
            self.pending_readback = None
            self.terminal_stop_request = None
            self.state = "failed_pending_finalize"

    def _fail_safe_stop_readback(self, exc: Exception) -> None:
        self.safe_stop_status = "SAFE_STOP_READBACK_FAILED"
        self.safe_stop_verified = False
        self.safe_stop_error = f"{type(exc).__name__}: {exc}"
        suffix = "SAFE_STOP_READBACK_FAILED: " + self.safe_stop_error
        self.error = "; ".join(part for part in (self.error, suffix) if part)
        self.infrastructure_failure = True
        self.pending_readback = None
        self.terminal_stop_request = None
        self.state = "failed_pending_finalize"

    def _abort_active_segment_completion(self, *, reason: str) -> str:
        if self.active_segment_completion_row_index is None:
            return ""
        row = self._active_completion_row()
        sim_step = int(getattr(self.adapter, "sim_steps", 0) or 0)
        sim_time = float(getattr(self.adapter, "sim_time", 0.0) or 0.0)
        cleanup_error = ""
        targets = self._completion_tracking_targets(row)
        may_call_end = bool(
            (targets and row.get("tracking_end_attempt_count") == 0)
            or (not targets and row.get("tracking_lifecycle_closed") is not True)
        )
        if may_call_end:
            try:
                self._end_segment_tracking(
                    adapter=self.adapter,
                    row=row,
                    sim_step=sim_step,
                    sim_time_s=sim_time,
                    reason="ABORTED",
                )
            except Exception as exc:
                cleanup_error = (
                    "segment tracking abort failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        row["terminal_kind"] = "ABORTED"
        row["terminal_sim_step"] = sim_step
        row["terminal_sim_time_s"] = sim_time
        row["abort_reason"] = str(reason)
        row["abort_cleanup_error"] = cleanup_error
        self.active_segment_completion_row_index = None
        self.active_completion_latched_servo_residual_deg = {}
        return cleanup_error

    def _request_failure(self, error: str, *, infrastructure_failure: bool) -> None:
        if self.terminal_payload is not None:
            return
        resolved = str(error or "Macro FSM failed")
        cleanup_error = self._abort_active_segment_completion(reason=resolved)
        if cleanup_error:
            resolved = "; ".join((resolved, cleanup_error))
        if not self.error:
            self.error = resolved
        elif resolved not in self.error:
            self.error += "; " + resolved
        self.infrastructure_failure = bool(self.infrastructure_failure or infrastructure_failure)
        self.controller_terminal_outcome = (
            self.controller_terminal_outcome or "RUNTIME_FAILURE"
        )
        self.controller_terminal_reason = self.controller_terminal_reason or resolved
        self.terminal_stop_request = {
            "success_after_stop": False,
            "outcome": self.controller_terminal_outcome,
            "reason": resolved,
        }
        current_step = int(getattr(self.adapter, "sim_steps", 0) or 0)
        if (
            self.simulation_app_stopped
            or self.last_batch_attempt_sim_step == current_step
        ):
            # A controller batch may already have reached PhysX before a
            # post-ACK completion/tracking check failed.  Preserve one atomic
            # batch per callback and issue the stop from the next callback.
            # If the app has stopped, fail() instead records that no stop or
            # N+1 verification was possible and durably finalizes the failure.
            return
        payload = {
            "sim_step": current_step,
            "sim_time_s": float(getattr(self.adapter, "sim_time", 0.0) or 0.0),
        }
        self._begin_safe_stop(self.adapter, payload)

    def _detach_and_finalize_video(self) -> dict[str, Any]:
        detach_error = ""
        if self.observer_attached:
            try:
                self.adapter.detach_artifact_render_observer(self.recorder)
            except Exception as exc:
                detach_error = f"viewport observer detach failed: {type(exc).__name__}: {exc}"
            finally:
                self.observer_attached = False
        try:
            video = dict(self.recorder.finalize() or {}) if self.recorder is not None else {}
        except Exception as exc:
            video = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
        self.video_writer_quiesced = True
        if detach_error:
            video["valid"] = False
            video["error"] = "; ".join(
                part for part in (str(video.get("error", "") or ""), detach_error) if part
            )
        self.video = dict(_jsonable(video))
        return self.video

    def _task_inputs(self, *, success: bool, error: str) -> dict[str, Any]:
        final = dict(self.rows[-1] if self.rows else {})
        segment_completion_path = (
            self.request.run_dir / "macro_segment_completion_ledger.jsonl"
        )
        segment_completion_sha256 = (
            _sha256_file(segment_completion_path)
            if segment_completion_path.is_file()
            else ""
        )
        obstacle_front = final.get("obstacle_front_face_x_m")
        base_x = dict(final.get("base_position_m", {}) or {}).get("x")
        clearances = dict(final.get("wheel_front_face_clearance_m", {}) or {})
        crossed_count = sum(
            type(value) in (int, float) and float(value) > 0.0
            for value in clearances.values()
        )
        body_crossed = bool(
            type(base_x) in (int, float)
            and type(obstacle_front) in (int, float)
            and float(base_x) > float(obstacle_front)
            and crossed_count >= 3
        )
        lifts = all(
            record["airborne_seen_before_crossing"] is True
            and record["front_face_crossing_s"] is not None
            for record in self.phase_traversal.values()
        )
        stability = str(final.get("stability_state", "unknown") or "unknown")
        support_count = int(final.get("geometry_support_candidate_count", 0) or 0)
        recoverable = bool(
            body_crossed and support_count >= 2 and stability in {"stable", "recoverable"}
        )
        coverage_errors = self._source_action_coverage_errors()
        source_coverage_complete = not coverage_errors
        segment_completion_errors = self._segment_completion_coverage_errors()
        segment_completion_complete = not segment_completion_errors
        expected_source_action_count = len(self.expected_source_actions)
        completed_source_action_count = len(self.source_action_consumption_rows)
        expected_segment_count = sum(
            row["command_provenance"]["dispatch_kind"] == "segment_start"
            for row in self.expected_source_actions
        )
        consumed_segment_start_count = sum(
            row["command_provenance"]["dispatch_kind"] == "segment_start"
            for row in self.source_action_consumption_rows
        )
        completed_segment_count = sum(
            row.get("terminal_kind") == SegmentDecisionKind.COMPLETE.value
            for row in self.segment_completion_rows
        )
        expected_event_count = sum(
            len(row["command_provenance"]["commands"])
            for row in self.expected_source_actions
        )
        completed_event_count = sum(
            len(row["command_provenance"]["commands"])
            for row in self.source_action_consumption_rows
        )
        expected_step_count = len(
            {
                row["command_provenance"]["source_step_index"]
                for row in self.expected_source_actions
            }
        )
        completed_step_count = len(
            {
                row["command_provenance"]["source_step_index"]
                for row in self.source_action_consumption_rows
            }
        )
        dispatch_complete = bool(
            success
            and _is_task_success_outcome(self.controller_terminal_outcome)
            and source_coverage_complete
            and segment_completion_complete
            and expected_source_action_count > 0
            and expected_segment_count > 0
            and len(self.dispatch_rows) == self.command_dispatch_count
        )
        any_illegal = any(
            record["illegal_drive_up"] for record in self.phase_traversal.values()
        )
        body_stuck_evidence = self._terminal_optional_runtime_bool(
            "body_stuck", detected=self.body_stuck_detected
        )
        active_leg_trapped_evidence = self._terminal_optional_runtime_bool(
            "active_leg_trapped", detected=self.active_leg_trapped_detected
        )
        completed_result = {
            "source_version": self.request.source_version,
            "step_count": completed_step_count,
            "expected_step_count": expected_step_count,
            "fast_segment_count": expected_segment_count,
            "expected_event_count": expected_event_count,
            "sent_event_count": completed_event_count,
            "expected_segment_count": expected_segment_count,
            "completed_segment_count": completed_segment_count,
            "expected_segment_completion_count": expected_segment_count,
            "segment_completion_count": completed_segment_count,
            "source_action_coverage_complete": source_coverage_complete,
            "source_action_coverage_errors": coverage_errors,
            "segment_completion_coverage_complete": (
                segment_completion_complete
            ),
            "segment_completion_coverage_errors": segment_completion_errors,
            "segment_completion_ledger_path": str(segment_completion_path),
            "segment_completion_ledger_sha256": segment_completion_sha256,
            "consumed_segment_start_count": consumed_segment_start_count,
            "physical_command_dispatch_count": self.command_dispatch_count,
            "dispatch_complete": dispatch_complete,
            "scheduler_complete": dispatch_complete,
            "scheduler_stop_reason": self.controller_terminal_outcome or error,
            "timed_out": "timed out" in error.lower(),
            "simulation_app_stopped": self.simulation_app_stopped,
            "lifecycle": {
                "failed": self.infrastructure_failure,
                "failure_kind": "INFRASTRUCTURE" if self.infrastructure_failure else "",
            },
            "motion_start_ready": self.started_sim_time_s is not None,
            "actuator_targets_applied": bool(
                self.last_target_readback and self.safe_stop_verified
            ),
            "safe_stop_status": self.safe_stop_status,
            "safe_stop_verified": self.safe_stop_verified,
            "safe_stop_error": self.safe_stop_error,
            "nonfinite_core_state_detected": self.nonfinite_state_detected,
            "body_crossed_front_face": body_crossed,
            "final_recoverable": recoverable,
            "maximum_abs_roll_rad": self.peak_roll_rad,
            "maximum_abs_pitch_rad": self.peak_pitch_rad,
            "video_path": str(
                self.video.get("video_path", "")
                or getattr(self.recorder, "video_path", "")
                or ""
            ),
            "video": self.video,
            "video_valid": self.video.get("valid") is True,
            "video_full_decode_valid": bool(
                dict(self.video.get("full_decode", {}) or {}).get("valid") is True
                and self.video.get("full_decode_all_frames") is True
            ),
            "manual_video_reviewed": False,
            "second_simulator_process_detected": None,
            "single_simulator_preflight_available": False,
            "macro_controller": {
                "graph_id": self.request.graph_id,
                "graph_sha256": self.request.graph_sha256,
                "profile_library_sha256": self.request.profile_library_sha256,
                "bundle_sha256": self.request.bundle_sha256,
                "profile_id": self.request.profile_id,
                "terminal_outcome": self.controller_terminal_outcome,
                "terminal_reason": self.controller_terminal_reason,
                "controller_tick_count": self.controller_tick_count,
                "command_dispatch_count": self.command_dispatch_count,
                "expected_source_action_count": expected_source_action_count,
                "source_action_consumption_count": completed_source_action_count,
                "source_action_coverage_complete": source_coverage_complete,
                "segment_completion_count": completed_segment_count,
                "expected_segment_completion_count": expected_segment_count,
                "segment_completion_coverage_complete": (
                    segment_completion_complete
                ),
                "segment_completion_coverage_errors": (
                    segment_completion_errors
                ),
                "segment_completion_ledger_path": str(
                    segment_completion_path
                ),
                "segment_completion_ledger_sha256": (
                    segment_completion_sha256
                ),
                "transition_count": len(self.transition_rows),
            },
            "deployment_safety_evidence": self.deployment_safety_evidence,
        }
        if self.residual_enabled:
            completed_result["direct_command_residual"] = {
                **self.status_dict()["direct_command_residual"],
                "last_transform": dict(
                    _jsonable(self.last_residual_transform)
                ),
            }
        physical_evidence = {
            "source_version": self.request.source_version,
            "body_crossed_front_face": body_crossed,
            "required_leg_lift_completed": lifts,
            "final_recoverable": recoverable,
            "robot_fell": True if stability == "fallen" else False if stability in {"stable", "recoverable"} else None,
            **body_stuck_evidence,
            "wheel_drive_up_without_required_lift": any_illegal,
            "dangerous_collision": (
                True if self.dangerous_collision_detected else None
            ),
            "dangerous_collision_available": False,
            "joint_limit_violation": self.joint_limit_violation_detected,
            "joint_limit_evidence_available": True,
            "nonfinite_core_state_detected": self.nonfinite_state_detected,
            "unsafe_joint_target": self.target_audit.get("unsafe"),
            "unsafe_joint_target_evidence_available": self.target_audit.get("available") is True,
            "unsafe_joint_target_validation_source": self.target_audit.get("source", ""),
            **active_leg_trapped_evidence,
            "severe_penetration": (
                True if self.severe_penetration_detected else None
            ),
            "penetration_evidence_available": False,
            "runtime_collision_penetration_classification": (
                "DETECTED_HARD_FAILURE"
                if self.dangerous_collision_detected
                or self.severe_penetration_detected
                else "NOT_EVALUATED_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
            ),
            "initial_deployment_collision_penetration_clear": bool(
                self.deployment_safety_evidence.get("available") is True
                and self.deployment_safety_evidence.get(
                    "dangerous_body_collision"
                )
                is False
                and self.deployment_safety_evidence.get("severe_penetration")
                is False
            ),
            "initial_deployment_evidence_sha256": self.deployment_safety_evidence.get(
                "deployment_binding_sha256", ""
            ),
            "final_wheel_contact_classes": dict(final.get("wheel_contact_classes", {}) or {}),
            "final_all_top": bool(
                len(dict(final.get("wheel_contact_classes", {}) or {})) == len(LEGS)
                and all(value == "TOP" for value in dict(final.get("wheel_contact_classes", {}) or {}).values())
            ),
            "final_all_loaded": None,
            "final_load_available": False,
            "final_velocity_stable": final.get("final_velocity_stable"),
            "final_posture_complete": bool(
                stability == "stable"
                and all(value == "TOP" for value in dict(final.get("wheel_contact_classes", {}) or {}).values())
            ),
            "maximum_abs_roll_rad": self.peak_roll_rad,
            "maximum_abs_pitch_rad": self.peak_pitch_rad,
            "traversal": {
                "legs": self.traversal,
                "phase_local_legs": self.phase_traversal,
                "any_illegal_drive_up": any_illegal,
                "lift_authority": "ACTIVE_LEG_PHASE_LOCAL",
            },
            "observation_scope": {
                "contact_load_available": False,
                "wheel_classification": "GEOMETRY_ONLY",
                "filtered_contact_bank_enabled": False,
                "strict_rest_blocking": False,
                "contact_drift_blocking": False,
                "final_all_top_blocking": False,
                "com_measurement_available": False,
                "com_proxy_source": "ROOT_POSITION_AND_WHEEL_GEOMETRY",
            },
        }
        final.update(physical_evidence)
        return {
            "schema_version": TASK_INPUTS_SCHEMA,
            "completed_result": completed_result,
            "physical_evidence": physical_evidence,
            "final_telemetry_row": final,
        }

    def _finalize(self, *, success: bool, error: str) -> dict[str, Any]:
        if self.terminal_payload is not None:
            return dict(self.terminal_payload)
        if self.adapter is not None:
            try:
                final_payload = self._observation_payload(self.adapter)
                if (
                    self.last_verified_servo_targets is not None
                    and self.last_verified_wheel_targets is not None
                ):
                    self.last_target_readback = (
                        self._capture_and_validate_target_readback(
                            self.adapter,
                            servo_targets=self.last_verified_servo_targets,
                            wheel_targets=self.last_verified_wheel_targets,
                            expected_sim_step=int(
                                getattr(self.adapter, "sim_steps", -1)
                            ),
                        )
                    )
                    final_payload["actuator_targets_applied"] = True
                    final_payload[
                        "actuator_target_source"
                    ] = "PHYSX_DRIVE_TARGET_READBACK"
                    final_payload[
                        "actuator_target_readback"
                    ] = self.last_target_readback
                    self._validate_hard_safety(final_payload)
                self._sample(
                    final_payload, decision=self.last_decision, force=True
                )
            except Exception as exc:
                success = False
                error = "; ".join(
                    part
                    for part in (
                        str(error or self.error or ""),
                        f"final observation/readback failed: {type(exc).__name__}: {exc}",
                    )
                    if part
                )
                self.infrastructure_failure = True
        if self.telemetry_attached:
            if getattr(self.adapter, "telemetry_collector", None) is self:
                self.adapter.attach_telemetry(None)
            self.telemetry_attached = False
        video = self._detach_and_finalize_video()
        if success:
            coverage_errors = self._successful_coverage_errors()
            if coverage_errors:
                success = False
                error = (
                    "successful Macro terminal lacks exact source-action coverage: "
                    + "; ".join(coverage_errors)
                )
        if success and self.nonfinite_state_detected:
            success = False
            error = "non-finite robot state observed during Macro FSM execution"
        if success and (
            self.safe_stop_status != "VERIFIED" or self.safe_stop_verified is not True
        ):
            success = False
            self.infrastructure_failure = True
            error = "successful Macro terminal lacks verified N+1 atomic safe-stop readback"
        if success and video.get("valid") is not True:
            success = False
            self.infrastructure_failure = True
            error = "viewport video finalization failed: " + str(
                video.get("error", "invalid viewport video") or "invalid viewport video"
            )
        self.error = str(error or self.error or "")
        self.state = "complete" if success else "failed"

        telemetry_jsonl = self.request.run_dir / "minimal_macro_telemetry.jsonl"
        telemetry_csv = self.request.run_dir / "minimal_macro_telemetry.csv"
        transitions_path = self.request.run_dir / "macro_transition_evidence.jsonl"
        dispatch_path = self.request.run_dir / "macro_dispatch_ledger.jsonl"
        source_consumption_path = (
            self.request.run_dir / "macro_source_action_consumption.jsonl"
        )
        segment_completion_path = (
            self.request.run_dir / "macro_segment_completion_ledger.jsonl"
        )
        timeline_path = self.request.run_dir / "macro_controller_timeline.json"
        _write_jsonl(telemetry_jsonl, self.rows)
        _write_csv(telemetry_csv, self.rows)
        _write_jsonl(transitions_path, self.transition_rows)
        _write_jsonl(dispatch_path, self.dispatch_rows)
        _write_jsonl(
            source_consumption_path, self.source_action_consumption_rows
        )
        _write_jsonl(segment_completion_path, self.segment_completion_rows)
        source_consumption_sha256 = _sha256_file(source_consumption_path)
        segment_completion_sha256 = _sha256_file(segment_completion_path)
        timeline = list(_attribute(self.controller, "timeline", []) or [])
        _atomic_write_json(timeline_path, {"timeline": timeline})
        inputs = self._task_inputs(success=success, error=self.error)
        inputs_path = self.request.run_dir / "macro_task_inputs.json"
        _atomic_write_json(inputs_path, inputs)
        root_write_count = getattr(self.adapter, "root_state_write_count", None)
        durable_root_write_count = (
            root_write_count if type(root_write_count) is int else -1
        )
        result = {
            "schema_version": SESSION_SCHEMA,
            "execution_mode": "normal_development",
            "source_version": self.request.source_version,
            "profile_id": self.request.profile_id,
            "request_id": self.request.request_id,
            "worker_pid": os.getpid(),
            "worker_session_id": self.worker_session_id,
            "adapter_runtime_instance_id": str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            "artifact_request_id": "",
            "root_state_write_count": durable_root_write_count,
            **self.bundle_identity,
            "run_dir": str(self.request.run_dir),
            "macro_fsm_complete": success,
            "controller_terminal_outcome": self.controller_terminal_outcome,
            "controller_terminal_reason": self.controller_terminal_reason,
            "controller_tick_count": self.controller_tick_count,
            "command_dispatch_count": self.command_dispatch_count,
            "expected_source_action_count": len(self.expected_source_actions),
            "source_action_consumption_count": len(
                self.source_action_consumption_rows
            ),
            "source_action_coverage_complete": not self._source_action_coverage_errors(),
            "source_action_consumption_path": str(source_consumption_path),
            "source_action_consumption_sha256": source_consumption_sha256,
            "segment_completion_count": sum(
                row.get("terminal_kind") == SegmentDecisionKind.COMPLETE.value
                for row in self.segment_completion_rows
            ),
            "expected_segment_completion_count": sum(
                row["command_provenance"]["dispatch_kind"] == "segment_start"
                for row in self.expected_source_actions
            ),
            "segment_completion_coverage_complete": not self._segment_completion_coverage_errors(),
            "segment_completion_coverage_errors": self._segment_completion_coverage_errors(),
            "segment_completion_path": str(segment_completion_path),
            "segment_completion_sha256": segment_completion_sha256,
            "segment_completion_ledger_path": str(segment_completion_path),
            "segment_completion_ledger_sha256": segment_completion_sha256,
            "transition_count": len(self.transition_rows),
            "telemetry_sample_count": len(self.rows),
            "telemetry_jsonl_path": str(telemetry_jsonl),
            "telemetry_csv_path": str(telemetry_csv),
            "transition_evidence_path": str(transitions_path),
            "dispatch_ledger_path": str(dispatch_path),
            "controller_timeline_path": str(timeline_path),
            "task_inputs_path": str(inputs_path),
            "target_audit": self.target_audit,
            "deployment_safety_evidence": self.deployment_safety_evidence,
            "last_target_readback": self.last_target_readback,
            "safe_stop_status": self.safe_stop_status,
            "safe_stop_verified": self.safe_stop_verified,
            "safe_stop_error": self.safe_stop_error,
            "safe_stop_ack": self.safe_stop_ack,
            "safe_stop_readback": self.safe_stop_readback,
            "safe_stop_readback_sha256": self.safe_stop_readback_sha256,
            "video": video,
            "video_writer_quiesced": self.video_writer_quiesced,
            "filtered_contact_bank_enabled": False,
            "physics_dt_s": self.physics_dt_s,
            "error": self.error,
        }
        if self.residual_enabled:
            result["direct_command_residual"] = {
                **self.status_dict()["direct_command_residual"],
                "last_transform": dict(
                    _jsonable(self.last_residual_transform)
                ),
            }
        result_path = self.request.run_dir / "worker_macro_fsm_result.json"
        _atomic_write_json(result_path, result)
        self.terminal_payload = {
            "type": "macro_fsm_complete" if success else "macro_fsm_failed",
            "operation": "macro_fsm",
            "phase": "MACRO_FSM_COMPLETE" if success else "MACRO_FSM_FAILED",
            "accepted": success,
            "macro_fsm_complete": success,
            "request_id": self.request.request_id,
            "worker_pid": os.getpid(),
            "worker_session_id": self.worker_session_id,
            "adapter_runtime_instance_id": str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            "artifact_request_id": "",
            "root_state_write_count": durable_root_write_count,
            "source_version": self.request.source_version,
            "profile_id": self.request.profile_id,
            **self.bundle_identity,
            "run_dir": str(self.request.run_dir),
            "task_inputs_path": str(inputs_path),
            "worker_result_path": str(result_path),
            "segment_completion_path": str(segment_completion_path),
            "segment_completion_sha256": segment_completion_sha256,
            "segment_completion_ledger_path": str(segment_completion_path),
            "segment_completion_ledger_sha256": segment_completion_sha256,
            "segment_completion_count": sum(
                row.get("terminal_kind") == SegmentDecisionKind.COMPLETE.value
                for row in self.segment_completion_rows
            ),
            "expected_segment_completion_count": sum(
                row["command_provenance"]["dispatch_kind"] == "segment_start"
                for row in self.expected_source_actions
            ),
            "segment_completion_coverage_complete": not self._segment_completion_coverage_errors(),
            "segment_completion_coverage_errors": self._segment_completion_coverage_errors(),
            "video_path": str(video.get("video_path", "") or getattr(self.recorder, "video_path", "") or ""),
            "video_writer_quiesced": self.video_writer_quiesced,
            "controller_terminal_outcome": self.controller_terminal_outcome,
            "safe_stop_status": self.safe_stop_status,
            "safe_stop_verified": self.safe_stop_verified,
            "safe_stop_error": self.safe_stop_error,
            "safe_stop_ack": self.safe_stop_ack,
            "safe_stop_readback": self.safe_stop_readback,
            "safe_stop_readback_sha256": self.safe_stop_readback_sha256,
            "deployment_safety_evidence": self.deployment_safety_evidence,
            "last_target_readback": self.last_target_readback,
            "filtered_contact_bank_enabled": False,
            "physics_dt_s": self.physics_dt_s,
            "error": self.error,
            "task_inputs": inputs,
        }
        return dict(self.terminal_payload)


__all__ = [
    "AUTHORIZED_GATE_D_SOURCE_VERSIONS",
    "CANONICAL_TASK_SUCCESS_TABLE_SHA256",
    "DEFAULT_POST_SETTLE_S",
    "DEFAULT_TELEMETRY_HZ",
    "DEFAULT_VIDEO_FPS",
    "GATE_D_TRIAL_KIND",
    "REQUEST_SCHEMA",
    "RESIDUAL_POLICY_OBSERVATION_SCHEMA",
    "WorkerMacroFSMRequest",
    "WorkerMacroFSMSession",
    "load_worker_macro_fsm_request",
    "validate_worker_macro_start_binding",
]
