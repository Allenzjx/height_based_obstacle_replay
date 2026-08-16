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

from command_model import (
    JOINT_COMMAND_SIGN,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    clamp_servo_command,
    command_limits_for_servo,
)
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
from .fsm50_centroidal_support import (
    CentroidalSupportEvidence,
    SupportThresholds,
    WholeBodyCOMMeasurement,
    WheelContactFrame,
    assess_contact_wrench_feasibility,
    assess_primary_diagonal_support,
    assess_support_region,
    measure_isaac_centroidal_angular_momentum_rate,
    measure_isaac_wheel_contacts,
    measure_isaac_whole_body_com,
)
from .fsm50_macro_state_model import (
    FINAL_RECOVERY_FEEDBACK_LIMITS,
    MacroStateId,
    build_default_macro_graph,
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
FEEDBACK_RECOVERY_ACTION_SCHEMA = "fsm50.feedback_recovery_action.v2"
FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA = (
    "fsm50.feedback_recovery_action_ledger.v2"
)
TERMINAL_RECOVERY_CLOSURE_SCHEMA = "fsm50.terminal_recovery_closure.v1"
TRANSITION_ROW_KEYS = frozenset(
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
DEFAULT_TELEMETRY_HZ = 15.0
DEFAULT_VIDEO_FPS = 15.0
DEFAULT_POST_SETTLE_S = 0.5
EXPECTED_PHYSICS_DT_S = 1.0 / 120.0
EXPECTED_RENDER_SUBSTEPS = 8
EXPECTED_RENDER_DT_S = EXPECTED_PHYSICS_DT_S * EXPECTED_RENDER_SUBSTEPS
EXPECTED_SERVO_REFERENCE_VELOCITY_DEG_S = 150.0
FEEDBACK_RECOVERY_PROBE_DELTA_DEG = 0.25
FEEDBACK_RECOVERY_INCREMENT_DELTA_DEG = 0.25
FEEDBACK_RECOVERY_MAXIMUM_ACTIONS = 64
FEEDBACK_RECOVERY_MAXIMUM_INCREMENTS_PER_LEG = 8
FEEDBACK_RECOVERY_CONTACT_DWELL_S = 0.25
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
# Conservative lower bound shared by the unchanged production ground
# (dynamic friction 1.05) and obstacle (dynamic friction 1.00).  The scene
# uses max-combine, so 1.00 never overstates either declared surface.
CONSERVATIVE_SCENE_DYNAMIC_FRICTION = 1.0
FINITE_WHEEL_CONTACT_PATCH_RADIUS_M = 0.01
FILTERED_CONTACT_FORCE_THRESHOLD_N = 1.0
CONTACT_FORCE_AXIS_TOLERANCE_N = 2.0e-4
CONTACT_SURFACE_HEIGHT_TOLERANCE_M = 0.005
TARGET_COMMAND_TOLERANCE = 1.0e-6
DRIVE_READBACK_TOLERANCE = 2.0e-6
UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE = (
    "UNAVAILABLE_REQUIRES_SHA_BOUND_FULL_VIDEO_REVIEW"
)
_FILTERED_CONTACT_SCENE_ROUTE_MARKER = (
    "_fsm50_macro_filtered_contact_scene_request_id"
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
        "recovery_stage",
        "recovery_action",
        "recovery_evidence_sha256",
        "recovery_centroidal_evidence_sha256",
        "recovery_feedback_observation_sha256",
        "recovery_target_map_sha256",
        "recovery_direction_sign",
        "recovery_attempt",
        "recovery_leg",
        "recovery_joint",
        "recovery_configuration_sha256",
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
        "FEEDBACK_RECOVERY",
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
    "recovery_stage": "",
    "recovery_action": "",
    "recovery_evidence_sha256": "",
    "recovery_centroidal_evidence_sha256": "",
    "recovery_feedback_observation_sha256": "",
    "recovery_target_map_sha256": "",
    "recovery_direction_sign": None,
    "recovery_attempt": None,
    "recovery_leg": "",
    "recovery_joint": "",
    "recovery_configuration_sha256": "",
}

_FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA = (
    "fsm50.feedback_recovery_configuration.v1"
)
_FEEDBACK_RECOVERY_CONFIGURATION_KEYS = frozenset(
    {
        "schema_version",
        "leg",
        "macro_state",
        "selected_source_version",
        "reference_profile_id",
        "reference_profile_source_version",
        "reference_profile_source_plan_sha256",
        "centroidal_evidence_sha256",
        "feedback_observation_sha256",
        "servo_reference_targets_deg",
        "measured_servo_positions_deg",
        "wheel_center_w_m",
        "body_crossed_front_face",
        "final_recoverable",
        "posture_complete",
    }
)
_FEEDBACK_RECOVERY_ACTION_ROW_KEYS = frozenset(
    {
        "schema_version",
        "action_index",
        "attempt",
        "sim_step",
        "expected_n_plus_one_sim_step",
        "stage",
        "action",
        "leg",
        "joint",
        "direction_sign",
        "configuration_sha256",
        "configuration_payload",
        "centroidal_evidence_sha256",
        "feedback_observation_sha256",
        "command_provenance",
        "batch_id",
        "dispatch_index",
        "dispatch_ack",
        "ack_sha256",
        "n_plus_one_verified",
        "n_plus_one_verified_sim_step",
        "n_plus_one_readback_sha256",
        "n_plus_one_readback",
        "dispatch_centroidal_support_evidence",
        "dispatch_feedback_recovery_observation",
        "physical_response_verified",
        "physical_response_sim_step",
        "physical_response_centroidal_evidence_sha256",
        "physical_response_feedback_observation_sha256",
        "physical_response_centroidal_support_evidence",
        "physical_response_feedback_recovery_observation",
        "physical_response",
    }
)
_FEEDBACK_RECOVERY_READBACK_KEYS = frozenset(
    {
        "sim_step",
        "command_epoch",
        "batch_id",
        "canonical_servo_targets_deg",
        "canonical_wheel_targets_rad_s",
        "servo_command_transform",
        "servo_command_transform_sha256",
        "expected_servo_drive_targets_rad",
        "actual_servo_drive_targets_rad",
        "actual_wheel_drive_targets_rad_s",
        "adapter_runtime_instance_id",
        "root_state_write_count",
        "physics_dt_s",
    }
)
_SERVO_COMMAND_TRANSFORM_SCHEMA = "fsm50.servo_command_transform.v1"
_SERVO_COMMAND_TRANSFORM_KEYS = frozenset(
    {
        "schema_version",
        "standing_pose_deg_by_servo",
        "command_sign_by_servo",
    }
)
_DURABLE_MOTION_BATCH_ACK_KEYS = frozenset(
    {
        "batch_id",
        "source",
        "error",
        "applied_sim_step",
        "first_physics_step",
        "motion_start_skew_s",
        "physics_dt_s",
        "servo_applied",
        "wheel_applied",
        "servo_targets_applied",
        "wheel_targets_applied",
        "recording_metadata",
    }
)
_SOURCE_ACTION_CONSUMPTION_ROW_KEYS = frozenset(
    {
        "schema_version",
        "source_action_index",
        "expected_source_action_count",
        "sim_time_s",
        "sim_step",
        "macro_state",
        "subphase",
        "profile_id",
        "profile_source_version",
        "profile_strategy",
        "source_plan_sha256",
        "profile_library_sha256",
        "bundle_sha256",
        "command_provenance",
        "servo_targets_deg",
        "wheel_targets_rad_s",
        "target_changed",
        "dispatch_epoch",
        "physical_dispatch_required",
        "physical_dispatch_applied",
        "physical_dispatch_index",
        "batch_id",
        "n_plus_one_verified",
        "n_plus_one_verified_sim_step",
        "n_plus_one_readback_sha256",
        "pre_action_verified_command_epoch",
        "pre_action_verified_readback",
        "pre_action_verified_readback_sha256",
    }
)

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


def _safety_scalar_normalizes_true(value: Any) -> bool:
    """Recognize a hard-safety True before normalizing sibling fields.

    Providers may expose numpy/torch-like scalar wrappers which `_jsonable`
    legitimately normalizes to the exact JSON boolean ``True``.  Inspect each
    hard-safety scalar in isolation so an unrelated later normalization error
    cannot erase a physical True observation.
    """

    try:
        return _jsonable(value) is True
    except BaseException:
        return value is True


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


def _is_lower_sha256(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    filtered_contact_bank_enabled: bool = True

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
            "filtered_contact_bank_enabled": self.filtered_contact_bank_enabled,
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
    if raw.get("filtered_contact_bank_enabled") is not True:
        raise ValueError("normal Macro FSM execution requires the filtered contact bank")

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
        filtered_contact_bank_enabled=True,
    )


def configure_scene_for_macro_fsm(
    scene_config: Any,
    request: WorkerMacroFSMRequest | None,
) -> Any:
    """Install the strict combined contact bank before ``create_scene``.

    The Macro/Residual route is opt-in and may only start from the production
    scene defaults.  The delegated helper changes contact instrumentation only;
    it does not touch pose, obstacle, material, solver, or actuator settings.
    """

    if request is None:
        return scene_config
    if request.filtered_contact_bank_enabled is not True:
        raise ValueError("Macro FSM request does not require the filtered contact bank")
    if getattr(scene_config, "telemetry_contact_sensors_enabled", None) is not False:
        raise ValueError(
            "Macro FSM contact route requires incoming production default "
            "telemetry_contact_sensors_enabled=false"
        )
    if getattr(scene_config, "contact_sensor_factory", None) is not None:
        raise ValueError(
            "Macro FSM contact route requires incoming production default "
            "contact_sensor_factory=None"
        )

    from .filtered_wheel_contact import make_filtered_wheel_contact_sensor_factory
    from .nonwheel_obstacle_contact import (
        configure_scene_for_wheel_and_nonwheel_contacts,
    )

    configure_scene_for_wheel_and_nonwheel_contacts(
        scene_config,
        wheel_factory=make_filtered_wheel_contact_sensor_factory(
            force_threshold_n=1.0
        ),
        force_threshold_n=1.0,
    )
    if (
        getattr(scene_config, "telemetry_contact_sensors_enabled", None) is not True
        or not callable(getattr(scene_config, "contact_sensor_factory", None))
    ):
        raise RuntimeError("Macro FSM combined contact-bank configuration failed")
    setattr(
        scene_config,
        _FILTERED_CONTACT_SCENE_ROUTE_MARKER,
        request.request_id,
    )
    return scene_config


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
        # Offline runner replay sets this from the SHA-bound durable worker
        # result; live sessions bind it directly to the attached adapter.
        self.durable_adapter_runtime_instance_id = ""
        self.scene_handle: Any | None = None
        self.bundle: Any | None = None
        self.controller: Any | None = None
        self.recorder: Any | None = None
        self.observer_attached = False
        self.telemetry_attached = False
        self.video: dict[str, Any] = {}
        self.video_writer_quiesced = False
        self.filtered_contact_bank_enabled = False
        self.filtered_contact_scene_handle: Any | None = None
        self.filtered_contact_bank: Any | None = None
        self.filtered_contact_wheel_bank: Any | None = None
        self.filtered_contact_nonwheel_bank: Any | None = None
        self.filtered_contact_wheel_sensors: dict[str, Any] = {}
        self.filtered_contact_nonwheel_sensors: dict[str, Any] = {}
        self.filtered_contact_nonwheel_specs: tuple[Any, ...] = ()
        self.filtered_contact_nonwheel_spec_identity: tuple[
            tuple[str, str], ...
        ] = ()
        self.filtered_contact_material_evidence: dict[str, Any] = {}
        self.filtered_contact_sample_epoch = 0
        self.filtered_contact_sample: dict[str, Any] = {}
        self.filtered_contact_wheel_rows: list[dict[str, Any]] = []
        self.filtered_contact_nonwheel_rows: list[dict[str, Any]] = []
        self.filtered_contact_surface_kind_by_leg = {
            leg: "AIR" for leg in LEGS
        }
        self.filtered_contact_surface_height_m_by_leg = {
            leg: 0.0 for leg in LEGS
        }
        self.filtered_contact_dwell_s_by_leg = {
            leg: 0.0 for leg in LEGS
        }
        self.filtered_contact_dwell_kind_by_leg = {
            leg: "AIR" for leg in LEGS
        }
        self.filtered_contact_last_sim_step: int | None = None
        self.filtered_contact_last_sim_time_s: float | None = None
        self.filtered_contact_previous_frame: WheelContactFrame | None = None
        self.filtered_contact_frame: WheelContactFrame | None = None
        self.filtered_contact_com: WholeBodyCOMMeasurement | None = None
        self.last_centroidal_support_evidence: CentroidalSupportEvidence | None = None
        self.started_sim_time_s: float | None = None
        self.completed_sim_time_s: float | None = None
        self.boundary_first_physics_step: int | None = None
        self.physics_dt_s: float | None = None
        self.boundary_ack: dict[str, Any] = {}
        self.boundary_readback: dict[str, Any] = {}
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
        self.durable_servo_command_transform: dict[str, Any] = {}
        self.feedback_recovery_action_rows: list[dict[str, Any]] = []
        self.feedback_recovery_sequence_by_configuration: dict[
            str, dict[str, Any]
        ] = {}
        self.feedback_recovery_configuration_by_leg: dict[str, str] = {}
        self.feedback_recovery_active_leg = ""
        self.feedback_recovery_verified_action_count = 0
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
        self.dangerous_collision_detection_evidence: dict[str, Any] = {}
        self.unverified_provider_collision_claim: dict[str, Any] = {}
        self.unverified_provider_penetration_claim: dict[str, Any] = {}
        self.severe_penetration_detected = False
        self.severe_penetration_detection_evidence: dict[str, Any] = {}
        self.runtime_contact_collision_evidence: dict[str, Any] = {
            "sample_count": 0,
            "clear_sample_count": 0,
            "detected_sample_count": 0,
            "last_sample_epoch": None,
            "first_sample_sim_step": None,
            "last_sample_sim_step": None,
            "first_sample_sha256": "",
            "last_sample_sha256": "",
            "first_detected_sim_step": None,
            "last_detected_sim_step": None,
            "first_detected_sample_sha256": "",
            "last_detected_sample_sha256": "",
            "last_collision": None,
            "source": "COMBINED_FILTERED_NONWHEEL_OBSTACLE_CONTACT_CURRENT_TICK",
        }
        self.body_stuck_detected = False
        self.active_leg_trapped_detected = False
        self.optional_runtime_evidence = {
            name: {
                "sample_count": 0,
                "available_sample_count": 0,
                "detected_sample_count": 0,
                "source": "",
                "first_sample_sim_step": None,
                "last_sample_sim_step": None,
                "last_value": None,
                "last_available": None,
                "first_detection_evidence": {},
            }
            for name in ("body_stuck", "active_leg_trapped")
        }
        self.unverified_optional_runtime_true_claims = {
            name: {} for name in ("body_stuck", "active_leg_trapped")
        }
        self._runtime_safety_capture_serial = 0
        self._optional_true_claim_capture_serial = {
            name: None for name in ("body_stuck", "active_leg_trapped")
        }
        self.controller_terminal_outcome = ""
        self.controller_terminal_reason = ""
        self.terminal_recovery_closure: dict[str, Any] = {}
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
            "filtered_contact_bank_enabled": self.filtered_contact_bank_enabled,
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

    def bind_filtered_contact_bank_scene(self, scene_handle: Any) -> None:
        """Bind the session only to a helper-selected, live combined bank."""

        if self.state != "created":
            raise RuntimeError(
                f"Macro FSM contact scene bind in state={self.state}"
            )
        config = getattr(scene_handle, "config", None)
        if (
            config is None
            or getattr(config, _FILTERED_CONTACT_SCENE_ROUTE_MARKER, None)
            != self.request.request_id
            or getattr(config, "telemetry_contact_sensors_enabled", None) is not True
            or not callable(getattr(config, "contact_sensor_factory", None))
        ):
            raise RuntimeError(
                "Macro FSM scene was not selected by the strict combined "
                "contact-bank helper"
            )
        sensor_error = str(getattr(scene_handle, "contact_sensor_error", "") or "")
        sensor = getattr(scene_handle, "contact_sensor", None)
        if sensor_error or sensor is None:
            raise RuntimeError(
                "Macro FSM combined contact bank is unavailable after scene creation: "
                + (sensor_error or "contact sensor is None")
            )
        wheel_bank = getattr(sensor, "wheel_bank", None)
        nonwheel_bank = getattr(sensor, "nonwheel_bank", None)
        if (
            getattr(sensor, "is_filtered_wheel_contact_bank", False) is not True
            or getattr(sensor, "is_nonwheel_obstacle_contact_bank", False) is not True
            or getattr(wheel_bank, "is_filtered_wheel_contact_bank", False) is not True
            or getattr(nonwheel_bank, "is_nonwheel_obstacle_contact_bank", False)
            is not True
        ):
            raise RuntimeError(
                "Macro FSM scene sensor is not the combined filtered wheel/non-wheel bank"
            )
        for label, bank in (("wheel", wheel_bank), ("non-wheel", nonwheel_bank)):
            threshold = getattr(bank, "force_threshold_n", None)
            if (
                type(threshold) not in (int, float)
                or not math.isfinite(float(threshold))
                or not math.isclose(
                    float(threshold), 1.0, rel_tol=0.0, abs_tol=1.0e-12
                )
            ):
                raise RuntimeError(
                    f"Macro FSM {label} contact threshold is not exactly 1.0 N"
                )
        wheel_sensors = getattr(wheel_bank, "sensors", None)
        nonwheel_sensors = getattr(nonwheel_bank, "sensors", None)
        nonwheel_specs = tuple(getattr(nonwheel_bank, "specs", ()) or ())
        nonwheel_spec_identity = tuple(
            (
                str(getattr(spec, "body_name", "") or ""),
                str(getattr(spec, "prim_path", "") or ""),
            )
            for spec in nonwheel_specs
        )
        expected_nonwheel_keys = tuple(
            prim_path for _body_name, prim_path in nonwheel_spec_identity
        )
        if (
            not isinstance(wheel_sensors, Mapping)
            or not isinstance(nonwheel_sensors, Mapping)
            or tuple(wheel_sensors.keys()) != tuple(LEGS)
            or not expected_nonwheel_keys
            or any(not key for key in expected_nonwheel_keys)
            or tuple(nonwheel_sensors.keys()) != expected_nonwheel_keys
        ):
            raise RuntimeError(
                "Macro FSM combined contact child mappings are not exact"
            )
        child_objects = tuple(wheel_sensors.values()) + tuple(
            nonwheel_sensors.values()
        )
        if len({id(value) for value in child_objects}) != len(child_objects):
            raise RuntimeError(
                "Macro FSM combined contact child sensors must be unique and disjoint"
            )
        from .nonwheel_obstacle_contact import OBSTACLE_PRIM_PATH

        if (
            getattr(nonwheel_bank, "obstacle_prim_path", None)
            != OBSTACLE_PRIM_PATH
            or any(not body_name for body_name, _prim_path in nonwheel_spec_identity)
        ):
            raise RuntimeError(
                "Macro FSM non-wheel obstacle contact specification is not exact"
            )
        for (_body_name, prim_path), child in zip(
            nonwheel_spec_identity,
            nonwheel_sensors.values(),
        ):
            cfg = getattr(child, "cfg", None)
            if (
                cfg is None
                or str(getattr(cfg, "prim_path", "") or "") != prim_path
                or tuple(getattr(cfg, "filter_prim_paths_expr", ()) or ())
                != (OBSTACLE_PRIM_PATH,)
            ):
                raise RuntimeError(
                    "Macro FSM non-wheel child prim/filter identity is not exact"
                )
        from .fsm50_residual_scene import GROUND_MATERIAL, OBSTACLE_MATERIAL

        material_values = {
            "ground_static_friction": GROUND_MATERIAL.static_friction,
            "ground_dynamic_friction": GROUND_MATERIAL.dynamic_friction,
            "obstacle_static_friction": OBSTACLE_MATERIAL.static_friction,
            "obstacle_dynamic_friction": OBSTACLE_MATERIAL.dynamic_friction,
        }
        for field, expected in material_values.items():
            value = getattr(config, field, None)
            if type(value) not in (int, float) or not math.isclose(
                float(value), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise RuntimeError(
                    f"Macro FSM scene material field {field} is not canonical"
                )
        if (
            GROUND_MATERIAL.friction_combine_mode != "max"
            or OBSTACLE_MATERIAL.friction_combine_mode != "max"
        ):
            raise RuntimeError("Macro FSM canonical surface combine mode is not max")
        material_evidence = {
            "schema_version": "fsm50.contact_friction_bound.v1",
            "ground": {
                "static_friction": float(GROUND_MATERIAL.static_friction),
                "dynamic_friction": float(GROUND_MATERIAL.dynamic_friction),
                "friction_combine_mode": GROUND_MATERIAL.friction_combine_mode,
            },
            "obstacle": {
                "static_friction": float(OBSTACLE_MATERIAL.static_friction),
                "dynamic_friction": float(OBSTACLE_MATERIAL.dynamic_friction),
                "friction_combine_mode": OBSTACLE_MATERIAL.friction_combine_mode,
            },
            "conservative_pair_dynamic_friction_lower_bound": (
                CONSERVATIVE_SCENE_DYNAMIC_FRICTION
            ),
            "source": (
                "CANONICAL_SCENE_CONFIG_AND_ENVIRONMENT_LOCK_BOUND;"
                "NOT_A_MEASURED_PAIR_COEFFICIENT"
            ),
        }
        material_evidence["payload_sha256"] = _canonical_json_sha256(
            material_evidence
        )
        self.filtered_contact_scene_handle = scene_handle
        self.filtered_contact_bank = sensor
        self.filtered_contact_wheel_bank = wheel_bank
        self.filtered_contact_nonwheel_bank = nonwheel_bank
        self.filtered_contact_wheel_sensors = dict(wheel_sensors)
        self.filtered_contact_nonwheel_sensors = dict(nonwheel_sensors)
        self.filtered_contact_nonwheel_specs = nonwheel_specs
        self.filtered_contact_nonwheel_spec_identity = nonwheel_spec_identity
        self.filtered_contact_material_evidence = material_evidence
        self.filtered_contact_bank_enabled = True

    @staticmethod
    def _same_sensor_mapping(
        current: Any,
        pinned: Mapping[str, Any],
    ) -> bool:
        return bool(
            isinstance(current, Mapping)
            and tuple(current.keys()) == tuple(pinned.keys())
            and all(current[key] is value for key, value in pinned.items())
        )

    def _same_nonwheel_sensor_contract(self, bank: Any) -> bool:
        from .nonwheel_obstacle_contact import OBSTACLE_PRIM_PATH

        specs = tuple(getattr(bank, "specs", ()) or ())
        sensors = getattr(bank, "sensors", None)
        if (
            len(specs) != len(self.filtered_contact_nonwheel_specs)
            or any(
                current is not pinned
                for current, pinned in zip(
                    specs, self.filtered_contact_nonwheel_specs
                )
            )
            or getattr(bank, "obstacle_prim_path", None) != OBSTACLE_PRIM_PATH
            or not self._same_sensor_mapping(
                sensors, self.filtered_contact_nonwheel_sensors
            )
        ):
            return False
        identity = tuple(
            (
                str(getattr(spec, "body_name", "") or ""),
                str(getattr(spec, "prim_path", "") or ""),
            )
            for spec in specs
        )
        if identity != self.filtered_contact_nonwheel_spec_identity:
            return False
        for (_body_name, prim_path), child in zip(identity, sensors.values()):
            cfg = getattr(child, "cfg", None)
            if (
                cfg is None
                or str(getattr(cfg, "prim_path", "") or "") != prim_path
                or tuple(getattr(cfg, "filter_prim_paths_expr", ()) or ())
                != (OBSTACLE_PRIM_PATH,)
            ):
                return False
        return True

    @staticmethod
    def _contact_sensor_clock_values(bank: Any) -> tuple[float, ...]:
        sensors: list[Any] = []
        for child in (
            getattr(bank, "wheel_bank", None),
            getattr(bank, "nonwheel_bank", None),
        ):
            mapping = getattr(child, "sensors", None)
            if not isinstance(mapping, Mapping) or not mapping:
                raise RuntimeError(
                    "combined contact bank lacks its exact child-sensor mapping"
                )
            sensors.extend(mapping.values())
        clocks: list[float] = []
        for sensor in sensors:
            values: list[float] = []
            for attribute in ("_timestamp", "_timestamp_last_update"):
                value = getattr(sensor, attribute, None)
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numpy"):
                    value = value.numpy()
                array = np.asarray(value, dtype=float)
                if (
                    array.ndim != 1
                    or array.size != 1
                    or not np.isfinite(array).all()
                ):
                    raise RuntimeError(
                        f"contact sensor {attribute} is not exact finite [1]"
                    )
                values.append(float(array[0]))
            if not math.isclose(
                values[0], values[1], rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise RuntimeError(
                    "contact sensor clock and last-update timestamp differ"
                )
            clocks.append(values[0])
        if not clocks:
            raise RuntimeError("combined contact bank contains no live sensors")
        return tuple(clocks)

    @staticmethod
    def _finite_contact_vec3(value: Any, *, label: str) -> tuple[float, float, float]:
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise RuntimeError(f"{label} is not an exact finite vec3")
        return tuple(float(item) for item in array)

    def _validated_combined_contact_rows(
        self,
        *,
        bank: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from .filtered_wheel_contact import (
            FILTERED_SURFACES,
            LEG_TO_WHEEL_BODY,
            ROBOT_PRIM_PATH,
        )
        from .nonwheel_obstacle_contact import OBSTACLE_PRIM_PATH

        wheel_bank = bank.wheel_bank
        nonwheel_bank = bank.nonwheel_bank
        wheel_objects = list(
            wheel_bank.filtered_observations(
                env_id=0,
                force_threshold_n=FILTERED_CONTACT_FORCE_THRESHOLD_N,
            )
        )
        expected_pairs = [
            (
                leg,
                LEG_TO_WHEEL_BODY[leg],
                f"{ROBOT_PRIM_PATH}/{LEG_TO_WHEEL_BODY[leg]}",
                filter_index,
                surface,
                other_path,
            )
            for leg in LEGS
            for filter_index, (surface, other_path) in enumerate(
                FILTERED_SURFACES
            )
        ]
        if len(wheel_objects) != len(expected_pairs):
            raise RuntimeError("filtered wheel bank does not expose exact 4x2 rows")
        wheel_rows: list[dict[str, Any]] = []
        for row, expected in zip(wheel_objects, expected_pairs, strict=True):
            identity = (
                getattr(row, "leg", None),
                getattr(row, "wheel_body_name", None),
                getattr(row, "wheel_prim_path", None),
                getattr(row, "filter_index", None),
                getattr(row, "surface", None),
                getattr(row, "other_prim_path", None),
            )
            if identity != expected:
                raise RuntimeError(
                    "filtered wheel contact pair identity/order is not canonical"
                )
            if getattr(row, "force_valid", None) is not True:
                raise RuntimeError("filtered wheel contact force evidence is unavailable")
            normal = self._finite_contact_vec3(
                getattr(row, "normal_force_w", None),
                label=f"{expected[0]}/{expected[4]} normal force",
            )
            friction = self._finite_contact_vec3(
                getattr(row, "friction_force_w", None),
                label=f"{expected[0]}/{expected[4]} friction force",
            )
            total = self._finite_contact_vec3(
                getattr(row, "total_force_w", None),
                label=f"{expected[0]}/{expected[4]} total force",
            )
            if not np.allclose(
                np.asarray(normal, dtype=float) + np.asarray(friction, dtype=float),
                np.asarray(total, dtype=float),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise RuntimeError("filtered wheel contact force decomposition drifted")
            normal_norm = float(np.linalg.norm(normal))
            friction_norm = float(np.linalg.norm(friction))
            total_norm = float(np.linalg.norm(total))
            strict_active = normal_norm > FILTERED_CONTACT_FORCE_THRESHOLD_N
            point_valid = getattr(row, "contact_point_valid", None)
            if type(point_valid) is not bool:
                raise RuntimeError("filtered wheel contact point validity is malformed")
            raw_point = np.asarray(getattr(row, "contact_point_w", None), dtype=float)
            if raw_point.shape != (3,):
                raise RuntimeError("filtered wheel contact point shape is invalid")
            if strict_active and (not point_valid or not np.isfinite(raw_point).all()):
                raise RuntimeError("active filtered wheel contact lacks a finite point")
            if not point_valid and not bool(np.isnan(raw_point).all()):
                raise RuntimeError("inactive wheel contact point is partially non-finite")
            reported_normal = float(getattr(row, "normal_force_n", float("nan")))
            reported_friction = float(
                getattr(row, "friction_force_n", float("nan"))
            )
            reported_total = float(getattr(row, "total_force_n", float("nan")))
            reported_upward = float(
                getattr(row, "upward_force_n", float("nan"))
            )
            if (
                not math.isclose(
                    reported_normal,
                    normal_norm,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    reported_friction,
                    friction_norm,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    reported_total,
                    total_norm,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    reported_upward,
                    max(0.0, normal[2]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise RuntimeError("filtered wheel contact force norm is inconsistent")
            serialized = dict(row.as_dict())
            serialized["contact_point_w"] = (
                [float(item) for item in raw_point]
                if point_valid
                else None
            )
            wheel_rows.append(
                {
                    **serialized,
                    # The shared gate uses the strict Isaac contact-time
                    # predicate.  In particular an exact 1.0 N row is not
                    # active even though the legacy facade reports >=.
                    "active": strict_active,
                    "bank_active": bool(getattr(row, "active", False)),
                }
            )

        nonwheel_objects = list(
            nonwheel_bank.observations(
                env_id=0,
                force_threshold_n=FILTERED_CONTACT_FORCE_THRESHOLD_N,
            )
        )
        specs = tuple(getattr(nonwheel_bank, "specs", ()))
        if not specs or len(nonwheel_objects) != len(specs):
            raise RuntimeError("non-wheel contact bank identity/count is unavailable")
        nonwheel_rows: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        wheel_bodies = set(LEG_TO_WHEEL_BODY.values())
        for row, spec in zip(nonwheel_objects, specs, strict=True):
            body_path = str(getattr(row, "body_prim_path", "") or "")
            if (
                body_path != str(getattr(spec, "prim_path", "") or "")
                or str(getattr(row, "body_name", "") or "")
                != str(getattr(spec, "body_name", "") or "")
                or str(getattr(row, "body_name", "") or "") in wheel_bodies
                or body_path in seen_paths
                or getattr(row, "filter_index", None) != 0
                or getattr(row, "other_prim_path", None) != OBSTACLE_PRIM_PATH
                or getattr(row, "force_valid", None) is not True
            ):
                raise RuntimeError("non-wheel obstacle contact identity is not exact")
            seen_paths.add(body_path)
            normal = self._finite_contact_vec3(
                getattr(row, "normal_force_w", None),
                label=f"{body_path} normal force",
            )
            friction = self._finite_contact_vec3(
                getattr(row, "friction_force_w", None),
                label=f"{body_path} friction force",
            )
            total = self._finite_contact_vec3(
                getattr(row, "total_force_w", None),
                label=f"{body_path} total force",
            )
            if not np.allclose(
                np.asarray(normal, dtype=float) + np.asarray(friction, dtype=float),
                np.asarray(total, dtype=float),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise RuntimeError("non-wheel obstacle force decomposition drifted")
            for reported, vector, label in (
                (getattr(row, "normal_force_n", None), normal, "normal"),
                (getattr(row, "friction_force_n", None), friction, "friction"),
                (getattr(row, "total_force_n", None), total, "total"),
            ):
                if type(reported) not in (int, float) or not math.isclose(
                    float(reported),
                    float(np.linalg.norm(vector)),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise RuntimeError(
                        f"non-wheel obstacle {label} force norm is inconsistent"
                    )
            strict_active = (
                float(np.linalg.norm(normal))
                > FILTERED_CONTACT_FORCE_THRESHOLD_N
            )
            if strict_active != bool(getattr(row, "active", False)):
                raise RuntimeError("non-wheel obstacle threshold predicate drifted")
            point_valid = getattr(row, "contact_point_valid", None)
            if type(point_valid) is not bool:
                raise RuntimeError("non-wheel obstacle point validity is malformed")
            raw_point = np.asarray(getattr(row, "contact_point_w", None), dtype=float)
            if raw_point.shape != (3,):
                raise RuntimeError("non-wheel obstacle contact point shape is invalid")
            if strict_active and (not point_valid or not np.isfinite(raw_point).all()):
                raise RuntimeError("active non-wheel contact lacks a finite point")
            if not point_valid and not bool(np.isnan(raw_point).all()):
                raise RuntimeError("inactive non-wheel contact point is partially non-finite")
            serialized = dict(row.as_dict())
            if not point_valid:
                serialized["contact_point_w"] = None
            nonwheel_rows.append(serialized)
        return wheel_rows, nonwheel_rows

    def _refresh_filtered_contact_evidence(self, adapter: Any) -> None:
        """Refresh and atomically publish one combined same-tick sample."""

        if (
            self.filtered_contact_bank_enabled is not True
            or self.filtered_contact_scene_handle is not self.scene_handle
        ):
            raise RuntimeError("combined filtered contact bank is not bound")
        bank = getattr(self.scene_handle, "contact_sensor", None)
        if (
            bank is None
            or bank is not self.filtered_contact_bank
            or getattr(bank, "wheel_bank", None) is not self.filtered_contact_wheel_bank
            or getattr(bank, "nonwheel_bank", None)
            is not self.filtered_contact_nonwheel_bank
            or not self._same_sensor_mapping(
                getattr(self.filtered_contact_wheel_bank, "sensors", None),
                self.filtered_contact_wheel_sensors,
            )
            or not self._same_nonwheel_sensor_contract(
                self.filtered_contact_nonwheel_bank
            )
            or not callable(getattr(bank, "update", None))
        ):
            raise RuntimeError("combined filtered contact bank cannot be updated")
        sim_step = int(getattr(adapter, "sim_steps", -1))
        sim_time = float(getattr(adapter, "sim_time", float("nan")))
        dt = float(self.physics_dt_s or float("nan"))
        if (
            sim_step < 0
            or not math.isfinite(sim_time)
            or not math.isfinite(dt)
            or dt <= 0.0
            or not math.isclose(
                sim_time,
                sim_step * dt,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
        ):
            raise RuntimeError("contact refresh lacks an exact adapter tick/time")
        if self.filtered_contact_last_sim_step == sim_step:
            if (
                self.filtered_contact_last_sim_time_s is None
                or not math.isclose(
                    self.filtered_contact_last_sim_time_s,
                    sim_time,
                    rel_tol=0.0,
                    abs_tol=max(1.0e-9, dt * 1.0e-6),
                )
                or not self.filtered_contact_sample
            ):
                raise RuntimeError("same-step contact sample identity drifted")
            return
        prior_step = self.filtered_contact_last_sim_step
        consecutive = prior_step is not None and sim_step == prior_step + 1
        if prior_step is not None and sim_step <= prior_step:
            raise RuntimeError("contact sample tick did not advance")
        if not consecutive:
            self.filtered_contact_previous_frame = None
            self.filtered_contact_dwell_s_by_leg = {
                leg: 0.0 for leg in LEGS
            }
            self.filtered_contact_dwell_kind_by_leg = {
                leg: "AIR" for leg in LEGS
            }

        clocks = self._contact_sensor_clock_values(bank)
        clock0 = clocks[0]
        tolerance = max(1.0e-9, dt * 1.0e-6)
        if any(
            not math.isclose(value, clock0, rel_tol=0.0, abs_tol=tolerance)
            for value in clocks[1:]
        ):
            raise RuntimeError("combined child contact banks are not at one clock")
        if consecutive:
            prior_time = self.filtered_contact_last_sim_time_s
            if prior_time is None or not (
                math.isclose(clock0, prior_time, rel_tol=0.0, abs_tol=tolerance)
                or math.isclose(clock0, sim_time, rel_tol=0.0, abs_tol=tolerance)
            ):
                raise RuntimeError(
                    "contact sensor clock regressed across consecutive samples"
                )
        update_dt = sim_time - clock0
        if update_dt < -tolerance:
            raise RuntimeError("contact sensor clock is ahead of the adapter")
        update_dt = max(0.0, update_dt)
        bank.update(update_dt, force_recompute=True)
        if (
            int(getattr(adapter, "sim_steps", -1)) != sim_step
            or not math.isclose(
                float(getattr(adapter, "sim_time", float("nan"))),
                sim_time,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ):
            raise RuntimeError("physics advanced while refreshing combined contacts")
        updated_clocks = self._contact_sensor_clock_values(bank)
        if any(
            not math.isclose(value, sim_time, rel_tol=0.0, abs_tol=tolerance)
            for value in updated_clocks
        ):
            raise RuntimeError("combined contact bank did not reach the adapter tick")

        wheel_rows, nonwheel_rows = self._validated_combined_contact_rows(
            bank=bank
        )
        obstacle = self._obstacle()
        surface_by_leg: dict[str, str] = {}
        surface_height_by_leg: dict[str, float] = {}
        dwell_by_leg: dict[str, float] = {}
        dwell_kind_by_leg: dict[str, str] = {}
        for leg in LEGS:
            active = [
                row for row in wheel_rows
                if row["leg"] == leg and row["active"] is True
            ]
            if len(active) == 0:
                surface = "AIR"
                height = 0.0
                dwell_identity = "AIR"
            elif len(active) > 1:
                # A wheel may touch a sharp ground/obstacle edge.  Keep both
                # pair rows, but do not guess a unique support plane.
                surface = "UNKNOWN"
                height = 0.0
                dwell_identity = "UNKNOWN"
            else:
                row = active[0]
                if row["surface"] == "ground":
                    surface = "GROUND"
                    height = float(obstacle.bottom_z_m)
                elif row["surface"] == "obstacle":
                    point = self._finite_contact_vec3(
                        row["contact_point_w"],
                        label=f"{leg} obstacle contact point",
                    )
                    normal = self._finite_contact_vec3(
                        row["normal_force_w"],
                        label=f"{leg} obstacle normal force",
                    )
                    horizontal = math.hypot(normal[0], normal[1])
                    if (
                        abs(point[2] - float(obstacle.top_z_m))
                        <= CONTACT_SURFACE_HEIGHT_TOLERANCE_M
                        and normal[2] > FILTERED_CONTACT_FORCE_THRESHOLD_N
                        and horizontal <= CONTACT_FORCE_AXIS_TOLERANCE_N
                    ):
                        surface = "OBSTACLE_TOP"
                        height = float(obstacle.top_z_m)
                    else:
                        surface = "FRONT_FACE"
                        height = 0.0
                else:
                    raise RuntimeError("filtered wheel surface label is unknown")
                dwell_identity = (
                    f"{surface}|filter={row['filter_index']}|"
                    f"other={row['other_prim_path']}"
                )
            prior_identity = self.filtered_contact_dwell_kind_by_leg[leg]
            if consecutive and dwell_identity == prior_identity and surface not in {
                "AIR",
                "UNKNOWN",
            }:
                dwell = self.filtered_contact_dwell_s_by_leg[leg] + dt
            else:
                dwell = 0.0
            surface_by_leg[leg] = surface
            surface_height_by_leg[leg] = height
            dwell_by_leg[leg] = dwell
            dwell_kind_by_leg[leg] = dwell_identity

        thresholds = SupportThresholds(
            normal_alignment_tolerance_n=CONTACT_FORCE_AXIS_TOLERANCE_N,
            friction_normal_tolerance_n=CONTACT_FORCE_AXIS_TOLERANCE_N,
        )
        current_com = measure_isaac_whole_body_com(
            adapter,
            env_id=0,
            expected_body_names=tuple(getattr(adapter.robot, "body_names", ())),
        )
        frame = measure_isaac_wheel_contacts(
            adapter,
            bank.wheel_bank,
            env_id=0,
            surface_kind_by_leg=surface_by_leg,
            surface_height_m_by_leg=surface_height_by_leg,
            friction_coefficient_by_leg={
                leg: CONSERVATIVE_SCENE_DYNAMIC_FRICTION for leg in LEGS
            },
            surface_dwell_s_by_leg=dwell_by_leg,
            surface_dwell_kind_by_leg=surface_by_leg,
            previous_frame=self.filtered_contact_previous_frame,
            finite_patch_radius_m_by_leg={
                leg: FINITE_WHEEL_CONTACT_PATCH_RADIUS_M for leg in LEGS
            },
            whole_body_com=current_com,
            contact_moment_model="MEASURED",
            thresholds=thresholds,
        )
        sample_epoch = self.filtered_contact_sample_epoch + 1
        sample_payload = {
            "schema_version": "fsm50.combined_contact_sample.v1",
            "sample_epoch": sample_epoch,
            "sample_sim_step": sim_step,
            "sample_sim_time_s": sim_time,
            "physics_dt_s": dt,
            "wheel_rows": wheel_rows,
            "nonwheel_rows": nonwheel_rows,
            "surface_kind_by_leg": surface_by_leg,
            "surface_height_m_by_leg": surface_height_by_leg,
            "surface_dwell_lower_bound_s_by_leg": dwell_by_leg,
            "surface_dwell_pair_identity_by_leg": dwell_kind_by_leg,
            "friction_coefficient_by_leg": {
                leg: CONSERVATIVE_SCENE_DYNAMIC_FRICTION for leg in LEGS
            },
            "friction_coefficient_source": (
                "CANONICAL_SCENE_CONFIG_AND_ENVIRONMENT_LOCK_BOUND;"
                "NOT_A_MEASURED_PAIR_COEFFICIENT"
            ),
            "friction_material_evidence": dict(
                _jsonable(self.filtered_contact_material_evidence)
            ),
            "wheel_contact_frame_available": frame.available,
            "wheel_contact_frame_errors": list(frame.errors),
        }
        sample_payload["sample_sha256"] = _canonical_json_sha256(
            sample_payload
        )
        # Publish all related rows and identities only after both child banks
        # and the strict wheel-contact frame have completed successfully.
        self.filtered_contact_sample_epoch = sample_epoch
        self.filtered_contact_sample = sample_payload
        self.filtered_contact_wheel_rows = wheel_rows
        self.filtered_contact_nonwheel_rows = nonwheel_rows
        self.filtered_contact_surface_kind_by_leg = surface_by_leg
        self.filtered_contact_surface_height_m_by_leg = surface_height_by_leg
        self.filtered_contact_dwell_s_by_leg = dwell_by_leg
        self.filtered_contact_dwell_kind_by_leg = dwell_kind_by_leg
        self.filtered_contact_last_sim_step = sim_step
        self.filtered_contact_last_sim_time_s = sim_time
        self.filtered_contact_frame = frame
        self.filtered_contact_com = current_com
        self.filtered_contact_previous_frame = frame

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
        self.durable_adapter_runtime_instance_id = str(
            getattr(adapter, "runtime_instance_id", "") or ""
        )
        if not self.durable_adapter_runtime_instance_id:
            raise RuntimeError("Macro FSM adapter runtime identity is unavailable")
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
        if (
            self.filtered_contact_bank_enabled is not True
            or self.filtered_contact_scene_handle is not self.scene_handle
        ):
            raise RuntimeError(
                "Macro FSM cannot start without the helper-selected combined "
                "filtered contact bank"
            )
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Macro FSM adapter is unavailable")
        if getattr(adapter, "telemetry_collector", None) is not None:
            raise RuntimeError("Macro FSM found another telemetry collector at start")
        if getattr(adapter, "artifact_render_observer", None) is not None:
            raise RuntimeError("Macro FSM found another viewport observer at start")

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
        self._refresh_filtered_contact_evidence(adapter)
        initial_payload = self._observation_payload(adapter)
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
        self.boundary_ack = self._durable_motion_batch_ack(
            ack, expected_recording_metadata={}
        )
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
            self._refresh_filtered_contact_evidence(adapter)
            payload = self._observation_payload(adapter)
            payload["actuator_targets_applied"] = True
            payload["actuator_target_source"] = "PHYSX_DRIVE_TARGET_READBACK"
            payload["actuator_target_readback"] = self.last_target_readback
            self._attach_drive_readback_layers(payload)
            if completed_readback_kind == "controller":
                self._capture_feedback_recovery_n_plus_one_response(
                    payload=payload,
                    sim_step=sim_step,
                )
            if completed_readback_kind == "safe_stop":
                # The physical safe-stop has already been independently
                # verified at exact N+1.  Persist this final observation, but
                # do not re-raise the same sticky hard-safety predicate and
                # recursively dispatch another safe-stop batch.
                stop_request = dict(self.terminal_stop_request or {})
                self.terminal_stop_request = None
                if stop_request.get("success_after_stop") is True:
                    # A newly observed hard failure still overrides a planned
                    # success stop; only failure-driven safe stops suppress
                    # the already-latched predicate.
                    self._validate_hard_safety(payload)
                self._sample(payload, decision=self.last_decision)
                if stop_request.get("success_after_stop") is True:
                    self.completed_sim_time_s = float(payload["sim_time_s"])
                    self.state = "settling"
                else:
                    self.state = "failed_pending_finalize"
                return
            self._validate_hard_safety(payload)
            self._sample(payload, decision=self.last_decision)
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
                    "recovery_stage": "",
                    "recovery_action": "",
                    "recovery_evidence_sha256": "",
                    "recovery_centroidal_evidence_sha256": "",
                    "recovery_feedback_observation_sha256": "",
                    "recovery_target_map_sha256": "",
                    "recovery_direction_sign": None,
                    "recovery_attempt": None,
                    "recovery_leg": "",
                    "recovery_joint": "",
                    "recovery_configuration_sha256": "",
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
        for name, expected in {
            "recovery_stage": "",
            "recovery_action": "",
            "recovery_evidence_sha256": "",
            "recovery_centroidal_evidence_sha256": "",
            "recovery_feedback_observation_sha256": "",
            "recovery_target_map_sha256": "",
            "recovery_direction_sign": None,
            "recovery_attempt": None,
            "recovery_leg": "",
            "recovery_joint": "",
            "recovery_configuration_sha256": "",
        }.items():
            if provenance.get(name) != expected:
                raise RuntimeError(
                    "SOURCE_ACTION provenance cannot claim feedback recovery fields"
                )

    def _durable_target_readback_errors(
        self,
        *,
        readback: Mapping[str, Any],
        readback_sha256: str,
        expected_servo_targets: Mapping[str, Any],
        expected_wheel_targets: Mapping[str, Any],
        expected_sim_step: int,
        expected_command_epoch: int | None,
        expected_batch_id: str,
        label: str,
    ) -> list[str]:
        """Validate a durable 8+4 PhysX target readback against an external anchor."""

        errors: list[str] = []
        try:
            transform = readback.get("servo_command_transform")
            standing_pose = (
                transform.get("standing_pose_deg_by_servo", {})
                if isinstance(transform, Mapping)
                else {}
            )
            command_sign = (
                transform.get("command_sign_by_servo", {})
                if isinstance(transform, Mapping)
                else {}
            )
            canonical_servos = readback.get("canonical_servo_targets_deg")
            canonical_wheels = readback.get("canonical_wheel_targets_rad_s")
            expected_drive = readback.get("expected_servo_drive_targets_rad")
            actual_drive = readback.get("actual_servo_drive_targets_rad")
            actual_wheels = readback.get("actual_wheel_drive_targets_rad_s")
            expected_runtime = str(self.durable_adapter_runtime_instance_id or "")
            valid = bool(
                isinstance(readback, Mapping)
                and set(readback) == _FEEDBACK_RECOVERY_READBACK_KEYS
                and _is_lower_sha256(readback_sha256)
                and _canonical_json_sha256(readback) == readback_sha256
                and type(readback.get("sim_step")) is int
                and readback.get("sim_step") == expected_sim_step
                and readback.get("command_epoch") == expected_command_epoch
                and type(readback.get("batch_id")) is str
                and readback.get("batch_id") == expected_batch_id
                and isinstance(canonical_servos, Mapping)
                and set(canonical_servos) == set(SERVO_JOINT_NAMES)
                and _strict_json_equal(
                    dict(canonical_servos), dict(expected_servo_targets)
                )
                and isinstance(canonical_wheels, Mapping)
                and set(canonical_wheels) == set(WHEEL_JOINT_NAMES)
                and _strict_json_equal(
                    dict(canonical_wheels), dict(expected_wheel_targets)
                )
                and isinstance(transform, Mapping)
                and set(transform) == _SERVO_COMMAND_TRANSFORM_KEYS
                and transform.get("schema_version")
                == _SERVO_COMMAND_TRANSFORM_SCHEMA
                and _canonical_json_sha256(transform)
                == readback.get("servo_command_transform_sha256")
                and isinstance(self.durable_servo_command_transform, Mapping)
                and bool(self.durable_servo_command_transform)
                and _strict_json_equal(
                    transform, self.durable_servo_command_transform
                )
                and isinstance(standing_pose, Mapping)
                and set(standing_pose) == set(SERVO_JOINT_NAMES)
                and isinstance(command_sign, Mapping)
                and _strict_json_equal(
                    dict(command_sign),
                    {
                        name: float(JOINT_COMMAND_SIGN[name])
                        for name in SERVO_JOINT_NAMES
                    },
                )
                and isinstance(expected_drive, Mapping)
                and set(expected_drive) == set(SERVO_JOINT_NAMES)
                and isinstance(actual_drive, Mapping)
                and set(actual_drive) == set(SERVO_JOINT_NAMES)
                and isinstance(actual_wheels, Mapping)
                and set(actual_wheels) == set(WHEEL_JOINT_NAMES)
                and all(
                    type(value) in (int, float)
                    and type(value) is not bool
                    and math.isfinite(float(value))
                    for value in [
                        *standing_pose.values(),
                        *canonical_servos.values(),
                        *canonical_wheels.values(),
                        *expected_drive.values(),
                        *actual_drive.values(),
                        *actual_wheels.values(),
                    ]
                )
                and all(
                    math.isclose(
                        float(expected_drive[name]),
                        math.radians(
                            float(standing_pose[name])
                            + float(JOINT_COMMAND_SIGN[name])
                            * float(canonical_servos[name])
                        ),
                        rel_tol=0.0,
                        abs_tol=DRIVE_READBACK_TOLERANCE,
                    )
                    and math.isclose(
                        float(actual_drive[name]),
                        float(expected_drive[name]),
                        rel_tol=0.0,
                        abs_tol=DRIVE_READBACK_TOLERANCE,
                    )
                    for name in SERVO_JOINT_NAMES
                )
                and all(
                    math.isclose(
                        float(actual_wheels[name]),
                        float(canonical_wheels[name]),
                        rel_tol=0.0,
                        abs_tol=DRIVE_READBACK_TOLERANCE,
                    )
                    for name in WHEEL_JOINT_NAMES
                )
                and type(readback.get("adapter_runtime_instance_id")) is str
                and bool(expected_runtime)
                and readback.get("adapter_runtime_instance_id")
                == expected_runtime
                and type(readback.get("root_state_write_count")) is int
                and readback.get("root_state_write_count") == 0
                and type(readback.get("physics_dt_s")) in (int, float)
                and type(readback.get("physics_dt_s")) is not bool
                and math.isclose(
                    float(readback.get("physics_dt_s")),
                    EXPECTED_PHYSICS_DT_S,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
        except Exception:
            valid = False
        if not valid:
            errors.append(f"{label} readback preimage/full identity drifted")
        return errors

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
        prior_source_sim_step: int | None = None
        prior_source_sim_time_s: float | None = None
        for index, row in enumerate(self.source_action_consumption_rows):
            label = f"source-action consumption row {index}"
            if index >= expected_action_count:
                errors.append(f"{label} has no canonical expected action")
                continue
            expected = self.expected_source_actions[index]
            if (
                not isinstance(row, Mapping)
                or (
                    not self.residual_enabled
                    and set(row) != _SOURCE_ACTION_CONSUMPTION_ROW_KEYS
                )
                or not _SOURCE_ACTION_CONSUMPTION_ROW_KEYS.issubset(set(row))
            ):
                errors.append(f"{label} keys are not exact")
                continue
            sim_step = row.get("sim_step")
            sim_time = row.get("sim_time_s")
            dispatch_epoch = row.get("dispatch_epoch")
            target_changed = row.get("target_changed")
            physical_required = row.get("physical_dispatch_required")
            physical_applied = row.get("physical_dispatch_applied")
            pre_action_readback = row.get("pre_action_verified_readback")
            exact_identity = {
                "schema_version": SOURCE_ACTION_CONSUMPTION_SCHEMA,
                "source_action_index": index,
                "expected_source_action_count": expected_action_count,
                "macro_state": expected["owner_state"],
                "profile_id": expected["profile_id"],
                "profile_source_version": expected["profile_source_version"],
                "profile_strategy": expected["profile_strategy"],
                "source_plan_sha256": expected["source_plan_sha256"],
                "profile_library_sha256": self.request.profile_library_sha256,
                "bundle_sha256": self.request.bundle_sha256,
                "command_provenance": expected["command_provenance"],
                "servo_targets_deg": expected["servo_targets_deg"],
                "wheel_targets_rad_s": expected["wheel_targets_rad_s"],
            }
            if any(
                not _strict_json_equal(row.get(key), value)
                for key, value in exact_identity.items()
            ):
                errors.append(f"{label} differs from its canonical source action")
            if (
                type(sim_step) is not int
                or sim_step < 0
                or type(sim_time) not in (int, float)
                or isinstance(sim_time, bool)
                or not math.isfinite(float(sim_time))
                or float(sim_time) < 0.0
                or type(dispatch_epoch) is not int
                or dispatch_epoch < 0
                or not (
                    row.get("pre_action_verified_command_epoch") is None
                    or (
                        type(row.get("pre_action_verified_command_epoch")) is int
                        and row.get("pre_action_verified_command_epoch") >= 0
                    )
                )
                or not isinstance(pre_action_readback, Mapping)
                or set(pre_action_readback) != _FEEDBACK_RECOVERY_READBACK_KEYS
                or _canonical_json_sha256(pre_action_readback)
                != row.get("pre_action_verified_readback_sha256")
                or pre_action_readback.get("command_epoch")
                != row.get("pre_action_verified_command_epoch")
                or type(pre_action_readback.get("sim_step")) is not int
                or pre_action_readback.get("sim_step") > sim_step
                or (
                    self.durable_adapter_runtime_instance_id
                    and pre_action_readback.get("adapter_runtime_instance_id")
                    != self.durable_adapter_runtime_instance_id
                )
                or (
                    self.durable_servo_command_transform
                    and not _strict_json_equal(
                        pre_action_readback.get("servo_command_transform"),
                        self.durable_servo_command_transform,
                    )
                )
                or not _is_lower_sha256(
                    row.get("pre_action_verified_readback_sha256")
                )
                or type(target_changed) is not bool
                or type(physical_required) is not bool
                or type(physical_applied) is not bool
            ):
                errors.append(f"{label} scalar/readback identity is invalid")
                continue
            if not math.isclose(
                float(sim_time),
                float(sim_step) * EXPECTED_PHYSICS_DT_S,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, EXPECTED_PHYSICS_DT_S * 1.0e-6),
            ):
                errors.append(f"{label} tick/time identity is invalid")
            if prior_source_sim_step is not None and (
                sim_step <= prior_source_sim_step
                or float(sim_time) < float(prior_source_sim_time_s)
            ):
                errors.append(
                    f"{label} source-action chronology is not strictly increasing"
                )
            prior_source_sim_step = sim_step
            prior_source_sim_time_s = float(sim_time)

            # The readback immediately preceding a source action is not its own
            # authority.  Anchor it to the most recent physical batch whose N+1
            # step has completed, or to the zero-wheel start boundary before the
            # first controller dispatch.  This prevents a coherently rehashed
            # pre-action map from turning a required physical action into a
            # fabricated retained-target consumption.
            prior_dispatches = [
                candidate
                for candidate in self.dispatch_rows
                if type(candidate.get("n_plus_one_verified_sim_step")) is int
                and candidate.get("n_plus_one_verified_sim_step") <= sim_step
            ]
            if index == 0:
                if prior_dispatches:
                    errors.append(
                        "first source action is preceded by an unauthorized "
                        "controller physical dispatch"
                    )
                anchor = self.boundary_ack
                anchor_servos = anchor.get("servo_targets_applied", {})
                anchor_wheels = anchor.get("wheel_targets_applied", {})
                anchor_epoch = 0
                anchor_batch_id = anchor.get("batch_id")
                boundary_step = self.boundary_readback.get("sim_step")
                if type(boundary_step) is not int or boundary_step >= sim_step:
                    errors.append(
                        f"{label} does not follow the verified start-boundary N+1 readback"
                    )
            elif prior_dispatches:
                anchor = max(
                    prior_dispatches,
                    key=lambda candidate: (
                        int(candidate["n_plus_one_verified_sim_step"]),
                        int(candidate.get("dispatch_index", -1)),
                    ),
                )
                anchor_servos = anchor.get("servo_targets_deg", {})
                anchor_wheels = anchor.get("wheel_targets_rad_s", {})
                anchor_epoch = anchor.get("command_epoch")
                anchor_batch_id = anchor.get("batch_id")
            else:
                anchor = self.boundary_ack
                anchor_servos = anchor.get("servo_targets_applied", {})
                anchor_wheels = anchor.get("wheel_targets_applied", {})
                anchor_epoch = 0
                anchor_batch_id = anchor.get("batch_id")
                boundary_step = self.boundary_readback.get("sim_step")
                if type(boundary_step) is not int or boundary_step >= sim_step:
                    errors.append(
                        f"{label} does not follow the verified start-boundary N+1 readback"
                    )
            readback_errors = self._durable_target_readback_errors(
                readback=pre_action_readback,
                readback_sha256=str(
                    row.get("pre_action_verified_readback_sha256", "") or ""
                ),
                expected_servo_targets=anchor_servos,
                expected_wheel_targets=anchor_wheels,
                expected_sim_step=sim_step,
                expected_command_epoch=anchor_epoch,
                expected_batch_id=anchor_batch_id,
                label=f"{label} pre-action",
            )
            errors.extend(readback_errors)
            pre_action_epoch = pre_action_readback.get("command_epoch")
            expected_dispatch_epoch = (
                pre_action_epoch + 1
                if target_changed is True and type(pre_action_epoch) is int
                else pre_action_epoch
            )
            if (
                type(pre_action_epoch) is not int
                or dispatch_epoch != expected_dispatch_epoch
            ):
                errors.append(
                    f"{label} dispatch epoch does not follow its verified pre-action epoch"
                )
            pre_action_servos = pre_action_readback.get(
                "canonical_servo_targets_deg"
            )
            pre_action_wheels = pre_action_readback.get(
                "canonical_wheel_targets_rad_s"
            )
            if (
                not isinstance(pre_action_servos, Mapping)
                or set(pre_action_servos) != set(SERVO_JOINT_NAMES)
                or not isinstance(pre_action_wheels, Mapping)
                or set(pre_action_wheels) != set(WHEEL_JOINT_NAMES)
                or any(
                    type(value) not in (int, float)
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in [
                        *pre_action_servos.values(),
                        *pre_action_wheels.values(),
                    ]
                )
            ):
                errors.append(f"{label} pre-action canonical 8+4 map is invalid")
                continue
            rebuilt_target_changed = bool(
                not _strict_json_equal(
                    dict(pre_action_servos), expected["servo_targets_deg"]
                )
                or not _strict_json_equal(
                    dict(pre_action_wheels), expected["wheel_targets_rad_s"]
                )
            )
            if not self.residual_enabled and target_changed is not rebuilt_target_changed:
                errors.append(
                    f"{label} target-change flag differs from its verified pre-action 8+4 map"
                )
            if not self.residual_enabled and physical_required is not target_changed:
                errors.append(f"{label} physical requirement differs from target change")
            if physical_applied is not physical_required:
                errors.append(f"{label} physical dispatch completion flag is invalid")
            if physical_applied:
                if (
                    type(row.get("physical_dispatch_index")) is not int
                    or row.get("physical_dispatch_index") < 0
                    or type(row.get("batch_id")) is not str
                    or not row.get("batch_id")
                    or row.get("n_plus_one_verified") is not True
                    or row.get("n_plus_one_verified_sim_step") != sim_step + 1
                    or not _is_lower_sha256(
                        row.get("n_plus_one_readback_sha256")
                    )
                ):
                    errors.append(f"{label} physical N+1 closure is invalid")
            elif (
                row.get("physical_dispatch_index") is not None
                or row.get("batch_id") != ""
                or row.get("n_plus_one_verified") is not False
                or row.get("n_plus_one_verified_sim_step") is not None
                or row.get("n_plus_one_readback_sha256") != ""
            ):
                errors.append(f"{label} retained-target closure is invalid")
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

    def _feedback_recovery_action_closure_errors(
        self,
        *,
        index: int,
        row: Mapping[str, Any],
    ) -> list[str]:
        from .fsm50_macro_controller import FeedbackRecoveryObservation

        label = f"feedback recovery action row {index}"
        errors: list[str] = []
        allowed_stage_actions = {
            "SAFE_PROBE": "CONSERVATIVE_DIAGNOSTIC_PROBE",
            "RETURN_TO_REFERENCE": "RETURN_TO_IMMUTABLE_REFERENCE",
            "INCREMENT": "BOUNDED_DESCENT_INCREMENT",
        }
        leg_joints = {
            "FL": {"front_left_hip", "front_left_knee"},
            "FR": {"front_right_hip", "front_right_knee"},
            "RL": {"rear_left_hip", "rear_left_knee"},
            "RR": {"rear_right_hip", "rear_right_knee"},
        }
        if set(row) != _FEEDBACK_RECOVERY_ACTION_ROW_KEYS:
            return [f"{label} keys are not exact"]
        stage = row.get("stage")
        action = row.get("action")
        leg = row.get("leg")
        joint = row.get("joint")
        sign = row.get("direction_sign")
        action_index = row.get("action_index")
        attempt = row.get("attempt")
        sim_step = row.get("sim_step")
        batch_id = row.get("batch_id")
        if (
            row.get("schema_version") != FEEDBACK_RECOVERY_ACTION_SCHEMA
            or type(action_index) is not int
            or action_index != index
            or type(attempt) is not int
            or attempt != index + 1
            or not 1 <= attempt <= FEEDBACK_RECOVERY_MAXIMUM_ACTIONS
            or type(sim_step) is not int
            or sim_step < 0
            or type(batch_id) is not str
            or not batch_id
            or allowed_stage_actions.get(stage) != action
            or leg not in leg_joints
            or joint not in leg_joints.get(leg, set())
            or type(sign) is not int
            or sign not in {-1, 1}
        ):
            errors.append(f"{label} core action identity drifted")
        try:
            dispatch_centroidal = CentroidalSupportEvidence.from_mapping(
                row.get("dispatch_centroidal_support_evidence", {})
            )
            dispatch_feedback = FeedbackRecoveryObservation.from_mapping(
                row.get("dispatch_feedback_recovery_observation", {})
            )
            response_centroidal = CentroidalSupportEvidence.from_mapping(
                row.get("physical_response_centroidal_support_evidence", {})
            )
            response_feedback = FeedbackRecoveryObservation.from_mapping(
                row.get("physical_response_feedback_recovery_observation", {})
            )
        except Exception as exc:
            return [
                *errors,
                f"{label} strict evidence envelope is invalid: "
                f"{type(exc).__name__}: {exc}",
            ]
        if (
            dispatch_centroidal.payload_sha256
            != row.get("centroidal_evidence_sha256")
            or dispatch_feedback.payload_sha256
            != row.get("feedback_observation_sha256")
            or response_centroidal.payload_sha256
            != row.get("physical_response_centroidal_evidence_sha256")
            or response_feedback.payload_sha256
            != row.get("physical_response_feedback_observation_sha256")
        ):
            errors.append(f"{label} evidence envelope SHA binding drifted")
        dispatch_index = row.get("dispatch_index")
        if (
            type(dispatch_index) is not int
            or dispatch_index < 0
            or dispatch_index >= len(self.dispatch_rows)
        ):
            return [*errors, f"{label} dispatch identity is invalid"]
        dispatch = self.dispatch_rows[dispatch_index]
        provenance = row.get("command_provenance")
        dispatch_provenance = dispatch.get("command_provenance")
        expected_evidence_sha = _canonical_json_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
                "centroidal_support_evidence_sha256": (
                    dispatch_centroidal.payload_sha256
                ),
                "feedback_recovery_observation_sha256": (
                    dispatch_feedback.payload_sha256
                ),
            }
        )
        expected_target_sha = _canonical_json_sha256(
            {
                "schema_version": "fsm50.feedback_recovery_target_map.v1",
                "servo_targets_deg": dict(dispatch.get("servo_targets_deg", {})),
                "wheel_targets_rad_s": dict(
                    dispatch.get("wheel_targets_rad_s", {})
                ),
            }
        )
        empty_source = {
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
        if (
            not isinstance(provenance, Mapping)
            or set(provenance) != _COMMAND_PROVENANCE_KEYS
            or not _strict_json_equal(provenance, dispatch_provenance)
            or provenance.get("kind") != "FEEDBACK_RECOVERY"
            or any(
                not _strict_json_equal(provenance.get(key), value)
                for key, value in empty_source.items()
            )
            or provenance.get("recovery_stage") != stage
            or provenance.get("recovery_action") != action
            or type(provenance.get("recovery_attempt")) is not int
            or provenance.get("recovery_attempt") != row.get("attempt")
            or provenance.get("recovery_leg") != leg
            or provenance.get("recovery_joint") != joint
            or provenance.get("recovery_direction_sign") != sign
            or provenance.get("recovery_configuration_sha256")
            != row.get("configuration_sha256")
            or provenance.get("recovery_centroidal_evidence_sha256")
            != dispatch_centroidal.payload_sha256
            or provenance.get("recovery_feedback_observation_sha256")
            != dispatch_feedback.payload_sha256
            or provenance.get("recovery_evidence_sha256")
            != expected_evidence_sha
            or provenance.get("recovery_target_map_sha256")
            != expected_target_sha
        ):
            errors.append(f"{label} durable provenance/dispatch binding drifted")
        expected_step = row.get("expected_n_plus_one_sim_step")
        dispatch_epoch = dispatch.get("command_epoch")
        expected_batch_id = (
            f"{self.request.request_id}:macro:{dispatch_epoch:06d}"
            if type(dispatch_epoch) is int
            else ""
        )
        if (
            type(dispatch.get("dispatch_index")) is not int
            or dispatch.get("dispatch_index") != dispatch_index
            or type(dispatch.get("sim_step")) is not int
            or dispatch.get("batch_id") != row.get("batch_id")
            or type(dispatch.get("batch_id")) is not str
            or dispatch.get("batch_id") != expected_batch_id
            or dispatch.get("sim_step") != row.get("sim_step")
            or type(dispatch_epoch) is not int
            or type(expected_step) is not int
            or expected_step != sim_step + 1
            or row.get("n_plus_one_verified") is not True
            or type(row.get("n_plus_one_verified_sim_step")) is not int
            or row.get("n_plus_one_verified_sim_step") != expected_step
            or row.get("physical_response_verified") is not True
            or type(row.get("physical_response_sim_step")) is not int
            or row.get("physical_response_sim_step") != expected_step
            or dispatch.get("n_plus_one_verified") is not True
            or type(dispatch.get("n_plus_one_verified_sim_step")) is not int
            or dispatch.get("n_plus_one_verified_sim_step") != expected_step
            or dispatch.get("n_plus_one_readback_sha256")
            != row.get("n_plus_one_readback_sha256")
            or response_centroidal.sim_step != expected_step
            or response_feedback.sim_step != expected_step
            or dispatch_centroidal.sim_step != sim_step
            or dispatch_feedback.sim_step != sim_step
            or not math.isclose(
                dispatch_centroidal.physics_time_s,
                dispatch_feedback.physics_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dispatch_centroidal.physics_dt_s * 1.0e-6),
            )
            or not math.isclose(
                response_centroidal.physics_time_s,
                response_feedback.physics_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, response_centroidal.physics_dt_s * 1.0e-6),
            )
            or not math.isclose(
                response_centroidal.physics_time_s,
                dispatch_centroidal.physics_time_s
                + dispatch_centroidal.physics_dt_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dispatch_centroidal.physics_dt_s * 1.0e-6),
            )
            or not math.isclose(
                dispatch_centroidal.physics_dt_s,
                EXPECTED_PHYSICS_DT_S,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                response_centroidal.physics_dt_s,
                dispatch_centroidal.physics_dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                dispatch_centroidal.physics_time_s,
                sim_step * dispatch_centroidal.physics_dt_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dispatch_centroidal.physics_dt_s * 1.0e-6),
            )
            or not math.isclose(
                float(dispatch.get("sim_time_s", float("nan"))),
                dispatch_centroidal.physics_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dispatch_centroidal.physics_dt_s * 1.0e-6),
            )
        ):
            errors.append(f"{label} exact dispatch/N+1 tick-time identity drifted")
        try:
            configuration_payload, reference, baseline = (
                self._feedback_durable_configuration_context(row=row)
            )
            if configuration_payload.get("leg") != leg:
                raise RuntimeError("configuration leg differs from action")
        except Exception as exc:
            errors.append(
                f"{label} durable configuration baseline is invalid: {exc}"
            )
            configuration_payload = {}
            reference = {}
            baseline = {}
        ack = row.get("dispatch_ack")
        ack_metadata = ack.get("recording_metadata") if isinstance(ack, Mapping) else None
        expected_ack_metadata: dict[str, Any] | None = None
        if configuration_payload:
            profile_matches = [
                candidate
                for candidate in self.expected_source_actions
                if candidate.get("owner_state") == "S10_POSTURE_RECOVERY"
                and candidate.get("profile_id")
                == configuration_payload.get("reference_profile_id")
                and candidate.get("profile_source_version")
                == configuration_payload.get(
                    "reference_profile_source_version"
                )
                and candidate.get("source_plan_sha256")
                == configuration_payload.get(
                    "reference_profile_source_plan_sha256"
                )
            ]
            if len(profile_matches) != 1:
                errors.append(
                    f"{label} durable ACK profile identity is not unique"
                )
            else:
                profile = profile_matches[0]
                expected_ack_metadata = {
                    "source_version": self.request.source_version,
                    "profile_id": profile["profile_id"],
                    "profile_source_version": profile[
                        "profile_source_version"
                    ],
                    "profile_strategy": profile["profile_strategy"],
                    "macro_state": "S10_POSTURE_RECOVERY",
                    "subphase": stage,
                    "command_epoch": dispatch_epoch,
                    "bundle_sha256": self.request.bundle_sha256,
                    "command_provenance": dict(provenance),
                    "segment_completion_control": dict(
                        _EMPTY_SEGMENT_COMPLETION_CONTROL
                    ),
                    "source_plan_sha256": "",
                    "source_action_consumption_index": None,
                }
                if self.residual_enabled:
                    residual_transform = dict(
                        dispatch.get("residual_transform", {}) or {}
                    )
                    expected_ack_metadata.update(
                        {
                            "nominal_command_epoch": dispatch.get(
                                "nominal_command_epoch"
                            ),
                            "physical_command_epoch": dispatch.get(
                                "physical_command_epoch"
                            ),
                            "nominal_target_changed": dispatch.get(
                                "nominal_target_changed"
                            ),
                            "applied_target_changed": dispatch.get(
                                "applied_target_changed"
                            ),
                            "dispatch_cause": dispatch.get("dispatch_cause"),
                            "nominal_servo_targets_deg": dict(
                                dispatch.get("nominal_servo_targets_deg", {})
                            ),
                            "nominal_wheel_targets_rad_s": dict(
                                dispatch.get("nominal_wheel_targets_rad_s", {})
                            ),
                            "residual_policy_id": dispatch.get(
                                "residual_policy_id"
                            ),
                            "residual_policy_sha256": dispatch.get(
                                "residual_policy_sha256"
                            ),
                            "residual_transform_sha256": str(
                                residual_transform.get(
                                    "core_transform_sha256", ""
                                )
                            ),
                            "residual_evidence_sha256": str(
                                residual_transform.get("evidence_sha256", "")
                            ),
                        }
                    )
        if (
            not isinstance(ack, Mapping)
            or set(ack) != _DURABLE_MOTION_BATCH_ACK_KEYS
            or _canonical_json_sha256(ack) != row.get("ack_sha256")
            or not _strict_json_equal(ack, dispatch.get("ack"))
            or type(ack.get("batch_id")) is not str
            or ack.get("batch_id") != row.get("batch_id")
            or ack.get("source") != "fsm50_macro_controller"
            or ack.get("error") != ""
            or type(ack.get("applied_sim_step")) is not int
            or ack.get("applied_sim_step") != sim_step
            or type(ack.get("first_physics_step")) is not int
            or ack.get("first_physics_step") != expected_step
            or ack.get("servo_applied") is not True
            or ack.get("wheel_applied") is not True
            or not self._feedback_target_maps_equal(
                ack.get("servo_targets_applied", {}),
                dispatch.get("servo_targets_deg", {}),
            )
            or not self._feedback_target_maps_equal(
                ack.get("wheel_targets_applied", {}),
                dispatch.get("wheel_targets_rad_s", {}),
            )
            or not isinstance(ack_metadata, Mapping)
            or expected_ack_metadata is None
            or not _strict_json_equal(ack_metadata, expected_ack_metadata)
            or type(ack_metadata.get("command_epoch")) is not int
            or ack_metadata.get("command_epoch") != dispatch.get("command_epoch")
            or dispatch.get("macro_state") != "S10_POSTURE_RECOVERY"
            or dispatch.get("subphase") != stage
            or dispatch.get("profile_id")
            != configuration_payload.get("reference_profile_id")
            or dispatch.get("profile_source_version")
            != configuration_payload.get("reference_profile_source_version")
            or (
                expected_ack_metadata is not None
                and dispatch.get("profile_strategy")
                != expected_ack_metadata.get("profile_strategy")
            )
            or dispatch.get("source_action_consumption_index") is not None
        ):
            errors.append(f"{label} dispatch ACK preimage/identity drifted")
        elif isinstance(ack_metadata, Mapping) and expected_ack_metadata is not None:
            try:
                # Reapply the same atomic timing/target validator used at the
                # live dispatch boundary.  The durable row must not be able to
                # rehash a non-zero skew or a noncanonical physics cadence into
                # an apparently valid ACK.
                self._validate_batch_ack(
                    ack,
                    batch_id=batch_id,
                    servo_targets=dispatch.get("servo_targets_deg", {}),
                    wheel_targets=dispatch.get("wheel_targets_rad_s", {}),
                    expected_sim_step=sim_step,
                    expected_physics_dt_s=EXPECTED_PHYSICS_DT_S,
                    expected_source="fsm50_macro_controller",
                    expected_recording_metadata=expected_ack_metadata,
                )
            except Exception as exc:
                errors.append(f"{label} durable atomic ACK is invalid: {exc}")
        readback = row.get("n_plus_one_readback")
        expected_runtime_instance_id = str(
            self.durable_adapter_runtime_instance_id or ""
        )
        if self.adapter is not None:
            live_runtime_instance_id = str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            )
            if (
                expected_runtime_instance_id
                and live_runtime_instance_id != expected_runtime_instance_id
            ):
                errors.append(f"{label} live adapter runtime identity drifted")
            expected_runtime_instance_id = live_runtime_instance_id
        command_transform = (
            readback.get("servo_command_transform", {})
            if isinstance(readback, Mapping)
            else {}
        )
        standing_pose = (
            command_transform.get("standing_pose_deg_by_servo", {})
            if isinstance(command_transform, Mapping)
            else {}
        )
        command_sign = (
            command_transform.get("command_sign_by_servo", {})
            if isinstance(command_transform, Mapping)
            else {}
        )
        durable_transform = self.durable_servo_command_transform
        durable_transform_valid = bool(
            isinstance(durable_transform, Mapping)
            and set(durable_transform) == _SERVO_COMMAND_TRANSFORM_KEYS
            and durable_transform.get("schema_version")
            == _SERVO_COMMAND_TRANSFORM_SCHEMA
            and _strict_json_equal(durable_transform, command_transform)
        )
        live_transform_valid = True
        if self.adapter is not None and isinstance(standing_pose, Mapping):
            transform = getattr(
                self.adapter, "command_to_actual_target_deg", None
            )
            live_transform_valid = bool(
                callable(transform)
                and set(standing_pose) == set(SERVO_JOINT_NAMES)
                and all(
                    math.isclose(
                        float(standing_pose[name]),
                        float(transform(name, 0.0)),
                        rel_tol=0.0,
                        abs_tol=DRIVE_READBACK_TOLERANCE,
                    )
                    for name in SERVO_JOINT_NAMES
                )
            )
        try:
            readback_valid = bool(
                isinstance(readback, Mapping)
                and set(readback) == _FEEDBACK_RECOVERY_READBACK_KEYS
                and _canonical_json_sha256(readback)
                == row.get("n_plus_one_readback_sha256")
                and type(readback.get("sim_step")) is int
                and readback.get("sim_step") == expected_step
                and type(readback.get("command_epoch")) is int
                and readback.get("command_epoch")
                == dispatch.get("command_epoch")
                and type(readback.get("batch_id")) is str
                and readback.get("batch_id") == row.get("batch_id")
                and self._feedback_target_maps_equal(
                    readback.get("canonical_servo_targets_deg", {}),
                    dispatch.get("servo_targets_deg", {}),
                )
                and self._feedback_target_maps_equal(
                    readback.get("canonical_wheel_targets_rad_s", {}),
                    dispatch.get("wheel_targets_rad_s", {}),
                )
                and isinstance(command_transform, Mapping)
                and set(command_transform) == _SERVO_COMMAND_TRANSFORM_KEYS
                and command_transform.get("schema_version")
                == _SERVO_COMMAND_TRANSFORM_SCHEMA
                and _canonical_json_sha256(command_transform)
                == readback.get("servo_command_transform_sha256")
                and durable_transform_valid
                and set(standing_pose) == set(SERVO_JOINT_NAMES)
                and set(command_sign) == set(SERVO_JOINT_NAMES)
                and _strict_json_equal(
                    dict(command_sign),
                    {
                        name: float(JOINT_COMMAND_SIGN[name])
                        for name in SERVO_JOINT_NAMES
                    },
                )
                and all(
                    type(value) in (int, float)
                    and type(value) is not bool
                    and math.isfinite(float(value))
                    for value in standing_pose.values()
                )
                and live_transform_valid
                and set(readback.get("actual_servo_drive_targets_rad", {}))
                == set(SERVO_JOINT_NAMES)
                and set(readback.get("expected_servo_drive_targets_rad", {}))
                == set(SERVO_JOINT_NAMES)
                and set(readback.get("actual_wheel_drive_targets_rad_s", {}))
                == set(WHEEL_JOINT_NAMES)
                and all(
                    type(value) in (int, float) and math.isfinite(float(value))
                    for value in readback["actual_servo_drive_targets_rad"].values()
                )
                and all(
                    type(value) in (int, float) and math.isfinite(float(value))
                    for value in readback[
                        "expected_servo_drive_targets_rad"
                    ].values()
                )
                and all(
                    type(value) in (int, float) and math.isfinite(float(value))
                    for value in readback[
                        "actual_wheel_drive_targets_rad_s"
                    ].values()
                )
                and self._feedback_target_maps_equal(
                    readback["actual_servo_drive_targets_rad"],
                    readback["expected_servo_drive_targets_rad"],
                )
                and all(
                    math.isclose(
                        float(
                            readback["expected_servo_drive_targets_rad"][name]
                        ),
                        math.radians(
                            float(standing_pose[name])
                            + float(JOINT_COMMAND_SIGN[name])
                            * float(
                                readback["canonical_servo_targets_deg"][name]
                            )
                        ),
                        rel_tol=0.0,
                        abs_tol=DRIVE_READBACK_TOLERANCE,
                    )
                    for name in SERVO_JOINT_NAMES
                )
                and self._feedback_target_maps_equal(
                    readback["actual_wheel_drive_targets_rad_s"],
                    readback["canonical_wheel_targets_rad_s"],
                )
                and type(readback.get("adapter_runtime_instance_id")) is str
                and bool(expected_runtime_instance_id)
                and readback.get("adapter_runtime_instance_id")
                == expected_runtime_instance_id
                and type(readback.get("root_state_write_count")) is int
                and readback.get("root_state_write_count") == 0
                and type(readback.get("physics_dt_s")) in (int, float)
                and type(readback.get("physics_dt_s")) is not bool
                and math.isclose(
                    float(readback.get("physics_dt_s")),
                    dispatch_centroidal.physics_dt_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
        except Exception:
            readback_valid = False
        if not readback_valid:
            errors.append(f"{label} N+1 readback preimage/full identity drifted")
        if (
            dispatch_feedback.observed_command_epoch
            != dispatch_feedback.verified_command_epoch
            or dispatch_feedback.observed_command_epoch + 1
            != dispatch.get("command_epoch")
            or response_feedback.observed_command_epoch
            != dispatch.get("command_epoch")
            or response_feedback.verified_command_epoch
            != dispatch.get("command_epoch")
            or response_feedback.n_plus_one_verified is not True
            or not self._feedback_target_maps_equal(
                response_feedback.readback_servo_targets_deg,
                dispatch.get("servo_targets_deg", {}),
            )
            or not self._feedback_target_maps_equal(
                response_feedback.readback_wheel_targets_rad_s,
                dispatch.get("wheel_targets_rad_s", {}),
            )
        ):
            errors.append(f"{label} N+1 epoch/full-map evidence drifted")
        try:
            self._validate_feedback_recovery_safety(
                centroidal=dispatch_centroidal,
                feedback=dispatch_feedback,
            )
            self._validate_feedback_recovery_safety(
                centroidal=response_centroidal,
                feedback=response_feedback,
            )
        except Exception as exc:
            errors.append(f"{label} persisted safety evidence is invalid: {exc}")
        if reference:
            expected_targets = dict(reference)
            grouped_prior = [
                candidate
                for candidate in self.feedback_recovery_action_rows[:index]
                if candidate.get("configuration_sha256")
                == row.get("configuration_sha256")
            ]
            if action == "CONSERVATIVE_DIAGNOSTIC_PROBE":
                expected_targets[joint] += sign * FEEDBACK_RECOVERY_PROBE_DELTA_DEG
            elif action == "BOUNDED_DESCENT_INCREMENT":
                prior_increments = [
                    candidate
                    for candidate in grouped_prior
                    if candidate.get("action") == "BOUNDED_DESCENT_INCREMENT"
                ]
                if any(
                    candidate.get("joint") != joint
                    or candidate.get("direction_sign") != sign
                    for candidate in prior_increments
                ):
                    errors.append(f"{label} descent selection drifted")
                increment_count = len(prior_increments) + 1
                if increment_count > FEEDBACK_RECOVERY_MAXIMUM_INCREMENTS_PER_LEG:
                    errors.append(f"{label} descent increment bound exceeded")
                expected_targets[joint] += (
                    sign
                    * FEEDBACK_RECOVERY_INCREMENT_DELTA_DEG
                    * increment_count
                )
            if (
                not self._feedback_target_maps_equal(
                    dispatch.get("servo_targets_deg", {}), expected_targets
                )
                or any(
                    float(value) != 0.0
                    for value in dispatch.get("wheel_targets_rad_s", {}).values()
                )
            ):
                errors.append(f"{label} durable bounded 8+4 target drifted")
            margin = float(
                FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"]
            )
            if any(
                not (
                    command_limits_for_servo(name)[0] + margin
                    <= float(target)
                    <= command_limits_for_servo(name)[1] - margin
                )
                for name, target in expected_targets.items()
            ):
                errors.append(f"{label} durable target violates command margin")
            if action == "BOUNDED_DESCENT_INCREMENT":
                try:
                    contact = dispatch_centroidal.wheel_contacts.by_leg()[leg]
                    if (
                        contact.evidence_available
                        and contact.measurement.active
                        and contact.measurement.surface_kind == "OBSTACLE_TOP"
                        and contact.normal_load_n is not None
                        and contact.normal_load_n
                        >= dispatch_centroidal.wheel_contacts.thresholds.minimum_normal_force_n
                    ):
                        errors.append(
                            f"{label} increment was issued after load-bearing TOP contact"
                        )
                except Exception as exc:
                    errors.append(
                        f"{label} TOP-contact increment gate cannot be rebuilt: {exc}"
                    )
        if index == 0:
            try:
                if dispatch_index <= 0:
                    raise RuntimeError(
                        "first feedback action has no preceding physical reference dispatch"
                    )
                reference_dispatch = self.dispatch_rows[dispatch_index - 1]
                reference_verified_step = reference_dispatch.get(
                    "n_plus_one_verified_sim_step"
                )
                reference_epoch = reference_dispatch.get("command_epoch")
                reference_batch_id = (
                    f"{self.request.request_id}:macro:{reference_epoch:06d}"
                    if type(reference_epoch) is int
                    else ""
                )
                reference_provenance = reference_dispatch.get(
                    "command_provenance"
                )
                reference_ack = reference_dispatch.get("ack")
                reference_ack_metadata = (
                    reference_ack.get("recording_metadata")
                    if isinstance(reference_ack, Mapping)
                    else None
                )
                if (
                    type(reference_dispatch.get("dispatch_index")) is not int
                    or reference_dispatch.get("dispatch_index")
                    != dispatch_index - 1
                    or type(reference_epoch) is not int
                    or reference_dispatch.get("batch_id")
                    != reference_batch_id
                    or dispatch_feedback.observed_command_epoch
                    != reference_epoch
                    or dispatch_feedback.verified_command_epoch
                    != reference_epoch
                    or reference_dispatch.get("n_plus_one_verified") is not True
                    or type(reference_verified_step) is not int
                    or not isinstance(reference_ack, Mapping)
                    or set(reference_ack) != _DURABLE_MOTION_BATCH_ACK_KEYS
                    or reference_ack.get("batch_id") != reference_batch_id
                    or reference_ack.get("source")
                    != "fsm50_macro_controller"
                    or reference_ack.get("error") != ""
                    or reference_ack.get("applied_sim_step")
                    != reference_dispatch.get("sim_step")
                    or reference_ack.get("first_physics_step")
                    != reference_verified_step
                    or reference_ack.get("servo_applied") is not True
                    or reference_ack.get("wheel_applied") is not True
                    or not self._feedback_target_maps_equal(
                        reference_ack.get("servo_targets_applied", {}),
                        reference_dispatch.get("servo_targets_deg", {}),
                    )
                    or not self._feedback_target_maps_equal(
                        reference_ack.get("wheel_targets_applied", {}),
                        reference_dispatch.get("wheel_targets_rad_s", {}),
                    )
                    or not isinstance(reference_provenance, Mapping)
                    or reference_provenance.get("kind") != "SOURCE_ACTION"
                    or reference_dispatch.get("macro_state")
                    != "S10_POSTURE_RECOVERY"
                    or reference_dispatch.get("profile_id")
                    != configuration_payload.get("reference_profile_id")
                    or reference_dispatch.get("profile_source_version")
                    != configuration_payload.get(
                        "reference_profile_source_version"
                    )
                    or not isinstance(reference_ack_metadata, Mapping)
                    or not _strict_json_equal(
                        reference_ack_metadata.get("command_provenance"),
                        reference_provenance,
                    )
                    or reference_ack_metadata.get("command_epoch")
                    != reference_epoch
                    or reference_ack_metadata.get("source_version")
                    != self.request.source_version
                    or reference_ack_metadata.get("bundle_sha256")
                    != self.request.bundle_sha256
                    or reference_ack_metadata.get("macro_state")
                    != "S10_POSTURE_RECOVERY"
                    or reference_ack_metadata.get("profile_id")
                    != configuration_payload.get("reference_profile_id")
                    or reference_ack_metadata.get("profile_source_version")
                    != configuration_payload.get(
                        "reference_profile_source_version"
                    )
                    or reference_ack_metadata.get("source_plan_sha256")
                    != configuration_payload.get(
                        "reference_profile_source_plan_sha256"
                    )
                    or reference_verified_step > sim_step
                    or not reference_dispatch.get("n_plus_one_readback_sha256")
                    or not self._feedback_target_maps_equal(
                        dispatch_feedback.readback_servo_targets_deg,
                        reference_dispatch.get("servo_targets_deg", {}),
                    )
                    or not self._feedback_target_maps_equal(
                        dispatch_feedback.readback_wheel_targets_rad_s,
                        reference_dispatch.get("wheel_targets_rad_s", {}),
                    )
                    or any(
                        float(value) != 0.0
                        for value in dispatch_feedback.readback_wheel_targets_rad_s.values()
                    )
                ):
                    raise RuntimeError(
                        "first feedback action does not continue the verified "
                        "zero-wheel S10 reference dispatch"
                    )
                self._validate_batch_ack(
                    reference_ack,
                    batch_id=reference_batch_id,
                    servo_targets=reference_dispatch.get(
                        "servo_targets_deg", {}
                    ),
                    wheel_targets=reference_dispatch.get(
                        "wheel_targets_rad_s", {}
                    ),
                    expected_sim_step=reference_dispatch.get("sim_step"),
                    expected_physics_dt_s=EXPECTED_PHYSICS_DT_S,
                    expected_source="fsm50_macro_controller",
                    expected_recording_metadata=reference_ack_metadata,
                )
            except Exception as exc:
                errors.append(
                    f"{label} first-action physical reference binding is invalid: {exc}"
                )
        else:
            prior = self.feedback_recovery_action_rows[index - 1]
            try:
                prior_dispatch_index = prior.get("dispatch_index")
                prior_response_centroidal = CentroidalSupportEvidence.from_mapping(
                    prior.get(
                        "physical_response_centroidal_support_evidence", {}
                    )
                )
                prior_response = FeedbackRecoveryObservation.from_mapping(
                    prior.get(
                        "physical_response_feedback_recovery_observation", {}
                    )
                )
                chronology_tolerance_s = max(
                    1.0e-9,
                    dispatch_centroidal.physics_dt_s * 1.0e-6,
                )
                if (
                    type(prior_dispatch_index) is not int
                    or dispatch_index <= prior_dispatch_index
                    or dispatch_centroidal.sim_step
                    < prior_response_centroidal.sim_step
                    or dispatch_feedback.sim_step < prior_response.sim_step
                    or dispatch_centroidal.physics_time_s
                    + chronology_tolerance_s
                    < prior_response_centroidal.physics_time_s
                    or dispatch_feedback.physics_time_s
                    + chronology_tolerance_s
                    < prior_response.physics_time_s
                ):
                    errors.append(
                        f"{label} physical dispatch chronology does not follow "
                        "the prior action N+1 response"
                    )
                if (
                    dispatch_feedback.observed_command_epoch
                    != prior_response.observed_command_epoch
                    or not self._feedback_target_maps_equal(
                        dispatch_feedback.readback_servo_targets_deg,
                        prior_response.readback_servo_targets_deg,
                    )
                    or not self._feedback_target_maps_equal(
                        dispatch_feedback.readback_wheel_targets_rad_s,
                        prior_response.readback_wheel_targets_rad_s,
                    )
                ):
                    errors.append(
                        f"{label} pre-action epoch/map does not continue the prior N+1 action"
                    )
            except Exception as exc:
                errors.append(
                    f"{label} prior N+1 action cannot be cross-bound: {exc}"
                )
        actual_response = row.get("physical_response")
        try:
            if action == "CONSERVATIVE_DIAGNOSTIC_PROBE":
                dq = float(response_feedback.measured_servo_positions_deg[joint]) - float(
                    baseline["measured_servo_positions_deg"][joint]
                )
                dx = float(response_feedback.wheel_center_w_m[leg][0]) - float(
                    baseline["wheel_x_m"]
                )
                dz = float(response_feedback.wheel_center_w_m[leg][2]) - float(
                    baseline["wheel_z_m"]
                )
                preserved, reasons = self._feedback_baseline_preserved_values(
                    leg=leg,
                    baseline=baseline,
                    centroidal=response_centroidal,
                    feedback=response_feedback,
                    joint=joint,
                )
                expected_response = {
                    "joint": joint,
                    "direction_sign": sign,
                    "dq_deg": dq,
                    "dx_m": dx,
                    "dz_m": dz,
                    "sign_response_valid": bool(
                        abs(dq) >= 0.02 and (dq > 0.0) == (sign > 0)
                    ),
                    "baseline_preserved": preserved,
                    "unsafe_reasons": list(reasons),
                    "n_plus_one_response_verified": True,
                }
            elif action == "RETURN_TO_IMMUTABLE_REFERENCE":
                return_error = abs(
                    float(response_feedback.measured_servo_positions_deg[joint])
                    - float(baseline["measured_servo_positions_deg"][joint])
                )
                expected_response = {
                    "joint": joint,
                    "direction_sign": sign,
                    "return_error_deg": return_error,
                    "n_plus_one_response_verified": True,
                }
                matching_probes = [
                    candidate
                    for candidate in self.feedback_recovery_action_rows[:index]
                    if candidate.get("configuration_sha256")
                    == row.get("configuration_sha256")
                    and candidate.get("action")
                    == "CONSERVATIVE_DIAGNOSTIC_PROBE"
                    and candidate.get("joint") == joint
                    and candidate.get("direction_sign") == sign
                    and candidate.get("physical_response_verified") is True
                ]
                if return_error > 0.2 or len(matching_probes) != 1:
                    errors.append(
                        f"{label} return did not close one verified probe at its immutable reference"
                    )
            elif action == "BOUNDED_DESCENT_INCREMENT":
                preserved, reasons = self._feedback_baseline_preserved_values(
                    leg=leg,
                    baseline=baseline,
                    centroidal=response_centroidal,
                    feedback=response_feedback,
                    joint=joint,
                )
                expected_response = {
                    "joint": joint,
                    "direction_sign": sign,
                    "baseline_preserved": preserved,
                    "unsafe_reasons": list(reasons),
                    "n_plus_one_response_verified": True,
                }
                if not preserved or reasons:
                    errors.append(
                        f"{label} descent increment degraded its immutable physical baseline"
                    )
            else:
                raise ValueError("unsupported action")
        except Exception as exc:
            errors.append(f"{label} physical response cannot be recomputed: {exc}")
        else:
            if not _strict_json_equal(actual_response, expected_response):
                errors.append(
                    f"{label} physical response differs from its durable evidence"
                )
        return errors

    def _feedback_recovery_coverage_errors(self) -> list[str]:
        errors: list[str] = []
        owned_dispatch_indices = [
            row.get("dispatch_index")
            for row in self.feedback_recovery_action_rows
        ]
        durable_feedback_dispatch_indices = [
            index
            for index, dispatch in enumerate(self.dispatch_rows)
            if isinstance(dispatch.get("command_provenance"), Mapping)
            and dispatch["command_provenance"].get("kind")
            == "FEEDBACK_RECOVERY"
        ]
        if (
            any(type(index) is not int for index in owned_dispatch_indices)
            or len(owned_dispatch_indices) != len(set(owned_dispatch_indices))
            or sorted(owned_dispatch_indices)
            != durable_feedback_dispatch_indices
            or (
                owned_dispatch_indices
                and sorted(owned_dispatch_indices)
                != list(
                    range(
                        min(owned_dispatch_indices),
                        min(owned_dispatch_indices)
                        + len(owned_dispatch_indices),
                    )
                )
            )
        ):
            errors.append(
                "feedback recovery dispatch ownership is not an exact one-action partition"
            )
        if self.feedback_recovery_verified_action_count != len(
            self.feedback_recovery_action_rows
        ):
            errors.append("feedback recovery action/readback count mismatch")
        for index, row in enumerate(self.feedback_recovery_action_rows):
            if (
                row.get("action_index") != index
                or row.get("n_plus_one_verified") is not True
                or not row.get("n_plus_one_readback_sha256")
                or row.get("physical_response_verified") is not True
                or not row.get(
                    "physical_response_centroidal_evidence_sha256"
                )
                or not row.get(
                    "physical_response_feedback_observation_sha256"
                )
            ):
                errors.append(
                    f"feedback recovery action row {index} lacks exact N+1 physical response closure"
                )
                continue
            errors.extend(
                self._feedback_recovery_action_closure_errors(
                    index=index,
                    row=row,
                )
            )
        errors.extend(self._feedback_recovery_durable_sequence_errors())
        return errors

    def _feedback_recovery_top_dwell_closure(
        self,
        *,
        leg: str,
        centroidal_mapping: Mapping[str, Any],
        feedback_mapping: Mapping[str, Any],
    ) -> tuple[bool, str]:
        """Rebuild one current-tick TOP/dwell closure from strict envelopes."""

        try:
            centroidal = CentroidalSupportEvidence.from_mapping(
                centroidal_mapping
            )
            from .fsm50_macro_controller import FeedbackRecoveryObservation

            feedback = FeedbackRecoveryObservation.from_mapping(feedback_mapping)
            if (
                centroidal.sim_step != feedback.sim_step
                or not math.isclose(
                    centroidal.physics_time_s,
                    feedback.physics_time_s,
                    rel_tol=0.0,
                    abs_tol=max(1.0e-9, float(centroidal.physics_dt_s) * 1.0e-6),
                )
            ):
                raise RuntimeError("centroidal/feedback tick identity differs")
            self._validate_feedback_recovery_safety(
                centroidal=centroidal,
                feedback=feedback,
            )
            contact = centroidal.wheel_contacts.by_leg()[leg]
            measurement = contact.measurement
            if not (
                contact.support_qualified
                and contact.evidence_available
                and measurement.active
                and measurement.surface_kind == "OBSTACLE_TOP"
                and contact.normal_load_n is not None
                and contact.normal_load_n
                >= centroidal.wheel_contacts.thresholds.minimum_normal_force_n
                and measurement.surface_dwell_verified
                and measurement.dwell_s is not None
                and measurement.dwell_s
                >= float(FINAL_RECOVERY_FEEDBACK_LIMITS["contact_dwell_s"])
            ):
                return False, "target leg lacks current TOP/load/dwell proof"
            return True, ""
        except Exception as exc:
            return False, f"strict TOP/dwell evidence is invalid: {exc}"

    def _feedback_recovery_terminal_top_dwell_closure(
        self, *, leg: str
    ) -> tuple[bool, str]:
        if not self.rows:
            return False, "final telemetry row is absent"
        final = self.rows[-1]
        centroidal_mapping = final.get("centroidal_support_evidence")
        feedback_mapping = final.get("feedback_recovery_observation")
        if not isinstance(centroidal_mapping, Mapping) or not isinstance(
            feedback_mapping, Mapping
        ):
            return False, "final strict centroidal/feedback envelopes are absent"
        return self._feedback_recovery_top_dwell_closure(
            leg=leg,
            centroidal_mapping=centroidal_mapping,
            feedback_mapping=feedback_mapping,
        )

    def _feedback_recovery_terminal_row_assessment(
        self,
        *,
        row: Mapping[str, Any],
        outcome: str,
    ) -> dict[str, Any]:
        """Recompute one strict final-recovery sample from its two envelopes."""

        from .fsm50_macro_controller import FeedbackRecoveryObservation

        result = {
            "valid": False,
            "error": "",
            "sim_step": None,
            "sim_time_s": None,
            "centroidal_evidence_sha256": "",
            "feedback_observation_sha256": "",
            "all_four_top_load_dwell": False,
            "body_crossed_front_face": False,
            "final_recoverable": False,
            "posture_complete": False,
            "support_viable": False,
            "wrench_proven": False,
            "attitude_safe": False,
            "joint_velocity_settled": False,
            "body_angular_velocity_settled": False,
            "outcome_predicate_satisfied": False,
            "controller_terminal": False,
            "controller_terminal_outcome": "",
            "macro_state": "",
            "feedback_recovery_stage": "",
            "feedback_recovery_exhaustion_reason": "",
        }
        try:
            centroidal = CentroidalSupportEvidence.from_mapping(
                row.get("centroidal_support_evidence", {})
            )
            feedback = FeedbackRecoveryObservation.from_mapping(
                row.get("feedback_recovery_observation", {})
            )
            sim_step = row.get("sim_step")
            sim_time = row.get("sim_time_s")
            tolerance = max(
                1.0e-9, float(centroidal.physics_dt_s) * 1.0e-6
            )
            if (
                type(sim_step) is not int
                or type(sim_time) not in (int, float)
                or isinstance(sim_time, bool)
                or not math.isfinite(float(sim_time))
                or centroidal.sim_step != sim_step
                or feedback.sim_step != sim_step
                or not math.isclose(
                    centroidal.physics_time_s,
                    float(sim_time),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or not math.isclose(
                    feedback.physics_time_s,
                    float(sim_time),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
            ):
                raise RuntimeError("terminal telemetry/evidence tick identity differs")
            row_base = row.get("base_position_m")
            row_joint_q = row.get("joint_position_rad")
            row_joint_qd = row.get("joint_velocity_rad_s")
            row_angular = row.get("root_angular_velocity_w")
            row_centers = row.get("wheel_center_w_m")
            if (
                not isinstance(row_base, Mapping)
                or set(row_base) != {"x", "y", "z"}
                or not isinstance(row_joint_q, Mapping)
                or set(row_joint_q)
                != set(SERVO_JOINT_NAMES) | set(WHEEL_JOINT_NAMES)
                or not isinstance(row_joint_qd, Mapping)
                or set(row_joint_qd)
                != set(SERVO_JOINT_NAMES) | set(WHEEL_JOINT_NAMES)
                or not isinstance(row_angular, (list, tuple))
                or len(row_angular) != 3
                or not isinstance(row_centers, Mapping)
                or set(row_centers) != set(LEGS)
            ):
                raise RuntimeError("terminal raw kinematic row is incomplete")
            transform = self.durable_servo_command_transform
            standing_pose = (
                transform.get("standing_pose_deg_by_servo", {})
                if isinstance(transform, Mapping)
                else {}
            )
            if (
                not isinstance(standing_pose, Mapping)
                or set(standing_pose) != set(SERVO_JOINT_NAMES)
            ):
                raise RuntimeError("terminal servo command transform is unavailable")
            rebuilt_positions = {
                name: float(JOINT_COMMAND_SIGN[name])
                * (
                    math.degrees(float(row_joint_q[name]))
                    - float(standing_pose[name])
                )
                for name in SERVO_JOINT_NAMES
            }
            rebuilt_velocities = {
                name: float(JOINT_COMMAND_SIGN[name])
                * math.degrees(float(row_joint_qd[name]))
                for name in SERVO_JOINT_NAMES
            }
            body_crossed_rebuilt = bool(
                float(row_base["x"]) > float(feedback.obstacle_front_face_x_m)
            )
            if (
                not all(
                    math.isclose(
                        float(row_base[axis]),
                        float(feedback.base_position_m[index]),
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    for index, axis in enumerate(("x", "y", "z"))
                )
                or not math.isclose(
                    float(row.get("base_roll_rad", float("nan"))),
                    float(feedback.base_roll_rad),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(row.get("base_pitch_rad", float("nan"))),
                    float(feedback.base_pitch_rad),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or any(
                    not math.isclose(
                        float(row_angular[index]),
                        float(feedback.base_angular_velocity_rad_s[index]),
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    for index in range(3)
                )
                or any(
                    not math.isclose(
                        rebuilt_positions[name],
                        float(feedback.measured_servo_positions_deg[name]),
                        rel_tol=0.0,
                        abs_tol=1.0e-7,
                    )
                    or not math.isclose(
                        rebuilt_velocities[name],
                        float(feedback.measured_servo_velocities_deg_s[name]),
                        rel_tol=0.0,
                        abs_tol=1.0e-7,
                    )
                    for name in SERVO_JOINT_NAMES
                )
                or not _strict_json_equal(
                    {leg: list(row_centers[leg]) for leg in LEGS},
                    {
                        leg: list(feedback.wheel_center_w_m[leg])
                        for leg in LEGS
                    },
                )
                or row.get("obstacle_front_face_x_m")
                != feedback.obstacle_front_face_x_m
                or row.get("obstacle_top_z_m") != feedback.obstacle_top_z_m
                or row.get("body_crossed_front_face")
                is not body_crossed_rebuilt
                or feedback.body_crossed_front_face is not body_crossed_rebuilt
                or row.get("final_recoverable") is not feedback.final_recoverable
                or row.get("posture_complete") is not feedback.posture_complete
            ):
                raise RuntimeError(
                    "terminal strict envelope differs from raw same-tick kinematics"
                )
            support = self._validate_feedback_recovery_safety(
                centroidal=centroidal,
                feedback=feedback,
            )
            contacts = centroidal.wheel_contacts.by_leg()
            all_four = all(
                contacts[leg].evidence_available
                and contacts[leg].support_qualified
                and contacts[leg].measurement.active
                and contacts[leg].measurement.surface_kind == "OBSTACLE_TOP"
                and contacts[leg].normal_load_n is not None
                and contacts[leg].normal_load_n
                >= centroidal.wheel_contacts.thresholds.minimum_normal_force_n
                and contacts[leg].measurement.surface_dwell_verified
                and contacts[leg].measurement.dwell_s is not None
                and contacts[leg].measurement.dwell_s
                >= float(FINAL_RECOVERY_FEEDBACK_LIMITS["contact_dwell_s"])
                for leg in LEGS
            )
            joint_settled = max(
                abs(float(value))
                for value in feedback.measured_servo_velocities_deg_s.values()
            ) <= float(
                FINAL_RECOVERY_FEEDBACK_LIMITS[
                    "maximum_abs_joint_velocity_deg_s"
                ]
            )
            angular_settled = max(
                abs(float(value))
                for value in feedback.base_angular_velocity_rad_s
            ) <= float(
                FINAL_RECOVERY_FEEDBACK_LIMITS[
                    "maximum_abs_body_angular_velocity_rad_s"
                ]
            )
            base_closed = bool(
                body_crossed_rebuilt
                and feedback.final_recoverable
                and support["support_viable"]
                and support["wrench_proven"]
                and joint_settled
                and angular_settled
            )
            if outcome == "TASK_SUCCESS":
                outcome_closed = bool(
                    base_closed and all_four and feedback.posture_complete
                )
            elif outcome == "TASK_SUCCESS_POSTURE_INCOMPLETE":
                outcome_closed = bool(
                    base_closed
                    and not (
                        all_four and feedback.posture_complete
                    )
                )
            else:
                raise RuntimeError("terminal recovery outcome is not a task-success class")
            state = build_default_macro_graph().get(
                MacroStateId.S10_POSTURE_RECOVERY
            )
            result.update(
                valid=True,
                sim_step=sim_step,
                sim_time_s=float(sim_time),
                centroidal_evidence_sha256=centroidal.payload_sha256,
                feedback_observation_sha256=feedback.payload_sha256,
                all_four_top_load_dwell=all_four,
                body_crossed_front_face=feedback.body_crossed_front_face,
                final_recoverable=feedback.final_recoverable,
                posture_complete=feedback.posture_complete,
                support_viable=support["support_viable"],
                wrench_proven=support["wrench_proven"],
                attitude_safe=bool(
                    abs(feedback.base_roll_rad)
                    <= state.completion_guard.maximum_abs_roll_rad
                    and abs(feedback.base_pitch_rad)
                    <= state.completion_guard.maximum_abs_pitch_rad
                ),
                joint_velocity_settled=joint_settled,
                body_angular_velocity_settled=angular_settled,
                outcome_predicate_satisfied=outcome_closed,
                controller_terminal=row.get("controller_terminal") is True,
                controller_terminal_outcome=str(
                    row.get("controller_terminal_outcome", "") or ""
                ),
                macro_state=str(row.get("macro_state", "") or ""),
                feedback_recovery_stage=str(
                    row.get("feedback_recovery_stage", "") or ""
                ),
                feedback_recovery_exhaustion_reason=str(
                    row.get("feedback_recovery_exhaustion_reason", "") or ""
                ),
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _feedback_recovery_terminal_settle_assessment(
        self, *, outcome: str
    ) -> dict[str, Any]:
        """Prove the terminal physical class over a contiguous telemetry tail."""

        required_dwell = float(
            FINAL_RECOVERY_FEEDBACK_LIMITS["settle_dwell_s"]
        )
        result: dict[str, Any] = {
            "complete": False,
            "error": "",
            "outcome": outcome,
            "required_settle_dwell_s": required_dwell,
            "settle_start_row_index": None,
            "settle_start_sim_step": None,
            "settle_start_sim_time_s": None,
            "settle_end_row_index": None,
            "settle_end_sim_step": None,
            "settle_end_sim_time_s": None,
            "settle_duration_s": 0.0,
            "final_assessment": {},
        }
        if outcome not in {
            "TASK_SUCCESS",
            "TASK_SUCCESS_POSTURE_INCOMPLETE",
        }:
            result["error"] = "terminal outcome is not a task-success class"
            return result
        if not self.rows:
            result["error"] = "terminal telemetry is absent"
            return result
        maximum_gap = max(
            1.5 / float(self.request.telemetry_hz),
            EXPECTED_RENDER_DT_S + EXPECTED_PHYSICS_DT_S,
        )
        qualifying: list[tuple[int, dict[str, Any]]] = []
        next_step: int | None = None
        next_time: float | None = None
        for index in range(len(self.rows) - 1, -1, -1):
            assessment = self._feedback_recovery_terminal_row_assessment(
                row=self.rows[index], outcome=outcome
            )
            if assessment.get("outcome_predicate_satisfied") is not True:
                break
            step = int(assessment["sim_step"])
            sample_time = float(assessment["sim_time_s"])
            if next_step is not None and (
                step >= next_step
                or sample_time >= float(next_time)
                or float(next_time) - sample_time > maximum_gap + 1.0e-9
            ):
                break
            qualifying.append((index, assessment))
            next_step = step
            next_time = sample_time
        if not qualifying:
            result["error"] = "final telemetry does not satisfy its recovery outcome"
            return result
        qualifying.reverse()
        start_index, start = qualifying[0]
        end_index, end = qualifying[-1]
        duration = float(end["sim_time_s"]) - float(start["sim_time_s"])
        result.update(
            settle_start_row_index=start_index,
            settle_start_sim_step=start["sim_step"],
            settle_start_sim_time_s=start["sim_time_s"],
            settle_end_row_index=end_index,
            settle_end_sim_step=end["sim_step"],
            settle_end_sim_time_s=end["sim_time_s"],
            settle_duration_s=duration,
            final_assessment=dict(end),
        )
        expected_stage = (
            "COMPLETE"
            if outcome == "TASK_SUCCESS"
            else "POSTURE_INCOMPLETE"
        )
        if (
            end.get("controller_terminal") is not True
            or end.get("controller_terminal_outcome") != outcome
            or end.get("macro_state") != MacroStateId.SUCCESS.value
            or end.get("feedback_recovery_stage") != expected_stage
            or (
                outcome == "TASK_SUCCESS_POSTURE_INCOMPLETE"
                and not end.get("feedback_recovery_exhaustion_reason")
            )
        ):
            result["error"] = (
                "terminal telemetry controller outcome/stage is inconsistent"
            )
            return result
        if duration + 1.0e-9 < required_dwell:
            result["error"] = "terminal recovery settling dwell is incomplete"
            return result
        result["complete"] = True
        return result

    def _feedback_recovery_terminal_errors(self) -> list[str]:
        # Narrow synthetic worker fixtures without an S10-owned source profile
        # exercise scheduler/ACK mechanics only.  Every production Gate-C/D
        # bundle contains an explicit S10 owner and must pass the physical
        # terminal closure below.
        if not any(
            row.get("owner_state")
            == MacroStateId.S10_POSTURE_RECOVERY.value
            for row in self.expected_source_actions
        ):
            return []
        outcome = str(self.controller_terminal_outcome or "")
        if not outcome:
            return ["terminal recovery outcome is absent"]
        assessment = self._feedback_recovery_terminal_settle_assessment(
            outcome=outcome
        )
        if assessment.get("complete") is True:
            return []
        return [
            "terminal recovery physical/settling closure failed: "
            + str(assessment.get("error", "invalid terminal recovery") or "invalid terminal recovery")
        ]

    def _terminal_recovery_closure_mapping(
        self,
        *,
        telemetry_jsonl_path: Path,
        telemetry_jsonl_sha256: str,
        final_raw_telemetry_row_sha256: str,
    ) -> dict[str, Any]:
        """Bind the existing telemetry ledger to its recomputed terminal class."""

        assessment = self._feedback_recovery_terminal_settle_assessment(
            outcome=str(self.controller_terminal_outcome or "")
        )
        core = {
            "schema_version": TERMINAL_RECOVERY_CLOSURE_SCHEMA,
            "outcome": str(self.controller_terminal_outcome or ""),
            "telemetry_jsonl_path": str(telemetry_jsonl_path),
            "telemetry_jsonl_sha256": str(telemetry_jsonl_sha256),
            "telemetry_sample_count": len(self.rows),
            "final_raw_telemetry_row_sha256": str(
                final_raw_telemetry_row_sha256
            ),
            "settle_assessment": dict(_jsonable(assessment)),
        }
        return {
            **core,
            "closure_sha256": _canonical_json_sha256(core),
        }

    def _feedback_recovery_durable_sequence_errors(self) -> list[str]:
        """Replay the bounded probe/return/increment protocol from ledger rows."""

        errors: list[str] = []
        if len(self.feedback_recovery_action_rows) > FEEDBACK_RECOVERY_MAXIMUM_ACTIONS:
            errors.append("feedback recovery durable action bound exceeded")
        seen_closed: set[str] = set()
        active_configuration = ""
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in self.feedback_recovery_action_rows:
            configuration = str(row.get("configuration_sha256", "") or "")
            if active_configuration and configuration != active_configuration:
                seen_closed.add(active_configuration)
            if configuration in seen_closed:
                errors.append(
                    "feedback recovery configuration action rows are not contiguous"
                )
            active_configuration = configuration
            grouped.setdefault(configuration, []).append(row)
        leg_joints = {
            "FR": ("front_right_hip", "front_right_knee"),
            "FL": ("front_left_hip", "front_left_knee"),
            "RR": ("rear_right_hip", "rear_right_knee"),
            "RL": ("rear_left_hip", "rear_left_knee"),
        }
        margin = float(FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"])
        seen_legs: set[str] = set()
        prior_leg = ""
        grouped_items = list(grouped.items())
        for configuration_index, (configuration, rows) in enumerate(grouped_items):
            label = f"feedback recovery configuration {configuration}"
            try:
                config, reference, _baseline = (
                    self._feedback_durable_configuration_context(row=rows[0])
                )
                leg = str(config["leg"])
                joints = leg_joints[leg]
                first_centroidal = CentroidalSupportEvidence.from_mapping(
                    rows[0]["dispatch_centroidal_support_evidence"]
                )
                from .fsm50_macro_controller import FeedbackRecoveryObservation

                first_feedback = FeedbackRecoveryObservation.from_mapping(
                    rows[0]["dispatch_feedback_recovery_observation"]
                )
            except Exception as exc:
                errors.append(f"{label} cannot be rebuilt: {exc}")
                continue
            if leg in seen_legs:
                errors.append(f"{label} reuses a completed recovery leg")
            seen_legs.add(leg)
            contacts = first_centroidal.wheel_contacts.by_leg()
            eligible = tuple(
                candidate_leg
                for candidate_leg in ("FR", "FL", "RR", "RL")
                if (
                    contacts[candidate_leg].evidence_available
                    and not contacts[candidate_leg].measurement.active
                    and contacts[candidate_leg].measurement.surface_kind == "AIR"
                    and first_feedback.wheel_center_w_m[candidate_leg][0]
                    > first_feedback.obstacle_front_face_x_m
                    and first_feedback.wheel_center_w_m[candidate_leg][2]
                    >= first_feedback.obstacle_top_z_m - 0.02
                    and not contacts[candidate_leg].support_qualified
                )
            )
            if not eligible or eligible[0] != leg:
                errors.append(
                    f"{label} first leg is not the deterministic current recovery candidate"
                )
            if prior_leg:
                prior_contact = contacts[prior_leg]
                measurement = prior_contact.measurement
                if not (
                    prior_contact.support_qualified
                    and measurement.active
                    and measurement.surface_kind == "OBSTACLE_TOP"
                    and measurement.surface_dwell_verified
                    and measurement.dwell_s is not None
                    and measurement.dwell_s
                    >= FEEDBACK_RECOVERY_CONTACT_DWELL_S
                ):
                    errors.append(
                        f"{label} starts before the prior leg has TOP contact dwell"
                    )
            probe_order = (
                (joints[0], 1),
                (joints[0], -1),
                (joints[1], 1),
                (joints[1], -1),
            )

            def safe(pair: tuple[str, int]) -> bool:
                candidate = dict(reference)
                candidate[pair[0]] += (
                    pair[1] * FEEDBACK_RECOVERY_PROBE_DELTA_DEG
                )
                return all(
                    command_limits_for_servo(name)[0] + margin
                    <= float(target)
                    <= command_limits_for_servo(name)[1] - margin
                    for name, target in candidate.items()
                )

            safe_pairs = tuple(pair for pair in probe_order if safe(pair))
            cursor = 0
            awaiting: tuple[str, int] | None = None
            completed: list[tuple[str, int]] = []
            results: dict[tuple[str, int], Mapping[str, Any]] = {}
            increment_pair: tuple[str, int] | None = None
            increment_count = 0
            for row in rows:
                pair = (str(row.get("joint", "")), int(row.get("direction_sign", 0) or 0))
                action = row.get("action")
                if action == "CONSERVATIVE_DIAGNOSTIC_PROBE":
                    while cursor < len(probe_order) and probe_order[cursor] != pair:
                        if safe(probe_order[cursor]):
                            errors.append(f"{label} skipped a command-limit-safe probe")
                            break
                        cursor += 1
                    if (
                        awaiting is not None
                        or increment_pair is not None
                        or cursor >= len(probe_order)
                        or probe_order[cursor] != pair
                        or pair not in safe_pairs
                        or pair in results
                    ):
                        errors.append(f"{label} probe ordering is invalid")
                    awaiting = pair
                    response = row.get("physical_response")
                    if isinstance(response, Mapping):
                        results[pair] = response
                elif action == "RETURN_TO_IMMUTABLE_REFERENCE":
                    if (
                        awaiting != pair
                        or pair not in results
                        or pair in completed
                    ):
                        errors.append(f"{label} return ordering is invalid")
                    response = row.get("physical_response")
                    if (
                        not isinstance(response, Mapping)
                        or float(response.get("return_error_deg", float("inf")))
                        > 0.2
                    ):
                        errors.append(f"{label} return response is not closed")
                    completed.append(pair)
                    awaiting = None
                    cursor += 1
                elif action == "BOUNDED_DESCENT_INCREMENT":
                    while cursor < len(probe_order) and not safe(probe_order[cursor]):
                        cursor += 1
                    if (
                        awaiting is not None
                        or cursor != len(probe_order)
                        or set(completed) != set(safe_pairs)
                        or set(results) != set(safe_pairs)
                        or len(completed) != len(set(completed))
                    ):
                        errors.append(
                            f"{label} increment precedes the complete safe probe matrix"
                        )
                    choices = sorted(
                        (
                            float(result.get("dz_m", float("inf"))),
                            candidate_pair[0],
                            candidate_pair[1],
                        )
                        for candidate_pair, result in results.items()
                        if result.get("sign_response_valid") is True
                        and result.get("baseline_preserved") is True
                        and float(result.get("dz_m", float("inf")))
                        <= -float(
                            FINAL_RECOVERY_FEEDBACK_LIMITS["minimum_descent_m"]
                        )
                        and candidate_pair in set(completed)
                    )
                    selected = None if not choices else (choices[0][1], choices[0][2])
                    if increment_pair is None:
                        increment_pair = pair
                        if selected is None or pair != selected:
                            errors.append(
                                f"{label} first increment is not the deterministic best measured response"
                            )
                    elif pair != increment_pair:
                        errors.append(f"{label} increment selection drifted")
                    increment_count += 1
                    if increment_count > FEEDBACK_RECOVERY_MAXIMUM_INCREMENTS_PER_LEG:
                        errors.append(f"{label} increment count exceeded its bound")
                else:
                    errors.append(f"{label} action kind is invalid")
            while cursor < len(probe_order) and not safe(probe_order[cursor]):
                cursor += 1
            matrix_complete = bool(
                cursor == len(probe_order)
                and set(completed) == set(safe_pairs)
                and set(results) == set(safe_pairs)
                and len(completed) == len(set(completed))
            )
            top_dwell_closed = False
            top_dwell_reason = ""
            if configuration_index + 1 < len(grouped_items):
                next_rows = grouped_items[configuration_index + 1][1]
                next_first = next_rows[0]
                next_centroidal = next_first.get(
                    "dispatch_centroidal_support_evidence"
                )
                next_feedback = next_first.get(
                    "dispatch_feedback_recovery_observation"
                )
                if isinstance(next_centroidal, Mapping) and isinstance(
                    next_feedback, Mapping
                ):
                    top_dwell_closed, top_dwell_reason = (
                        self._feedback_recovery_top_dwell_closure(
                            leg=leg,
                            centroidal_mapping=next_centroidal,
                            feedback_mapping=next_feedback,
                        )
                    )
                else:
                    top_dwell_reason = (
                        "next configuration lacks strict dispatch evidence"
                    )
            else:
                top_dwell_closed, top_dwell_reason = (
                    self._feedback_recovery_terminal_top_dwell_closure(leg=leg)
                )
            if awaiting is not None:
                errors.append(f"{label} ends with an unreturned probe")
            if not matrix_complete:
                errors.append(
                    f"{label} ended before the complete safe probe/return matrix"
                )
            choices = sorted(
                (
                    float(result.get("dz_m", float("inf"))),
                    candidate_pair[0],
                    candidate_pair[1],
                )
                for candidate_pair, result in results.items()
                if result.get("sign_response_valid") is True
                and result.get("baseline_preserved") is True
                and float(result.get("dz_m", float("inf")))
                <= -float(FINAL_RECOVERY_FEEDBACK_LIMITS["minimum_descent_m"])
                and candidate_pair in set(completed)
            )
            if (
                matrix_complete
                and choices
                and increment_pair is None
                and not top_dwell_closed
            ):
                errors.append(
                    f"{label} omitted the deterministic measured descent choice"
                )
            if (
                increment_pair is not None
                and increment_count
                < int(FINAL_RECOVERY_FEEDBACK_LIMITS["maximum_increments_per_leg"])
                and not top_dwell_closed
            ):
                errors.append(
                    f"{label} truncated bounded descent before TOP/dwell or its increment bound"
                )
            prior_leg = leg
        return errors

    def _dispatch_ownership_errors(self) -> list[str]:
        """Require every physical dispatch to have one durable semantic cause."""

        errors: list[str] = []
        if self.command_dispatch_count != len(self.dispatch_rows):
            errors.append("physical dispatch count differs from its durable ledger")
        source_rows = {
            row.get("source_action_index"): row
            for row in self.source_action_consumption_rows
            if type(row.get("source_action_index")) is int
        }
        feedback_rows_by_dispatch: dict[int, list[Mapping[str, Any]]] = {}
        for action in self.feedback_recovery_action_rows:
            dispatch_index = action.get("dispatch_index")
            if type(dispatch_index) is int:
                feedback_rows_by_dispatch.setdefault(dispatch_index, []).append(action)
        generated_wheel_stops_by_batch: dict[str, list[Mapping[str, Any]]] = {}
        for completion in self.segment_completion_rows:
            wheel_stop = completion.get("wheel_stop")
            if (
                isinstance(wheel_stop, Mapping)
                and wheel_stop.get("generated") is True
                and type(wheel_stop.get("batch_id")) is str
                and wheel_stop.get("batch_id")
            ):
                generated_wheel_stops_by_batch.setdefault(
                    str(wheel_stop["batch_id"]), []
                ).append(wheel_stop)
        seen_batches: set[str] = set()
        previous_n_plus_one_step: int | None = None
        prior_verified_servos = dict(
            self.boundary_readback.get("canonical_servo_targets_deg", {})
        )
        prior_verified_wheels = dict(
            self.boundary_readback.get("canonical_wheel_targets_rad_s", {})
        )
        prior_verified_command_epoch = 0
        for index, dispatch in enumerate(self.dispatch_rows):
            label = f"physical dispatch row {index}"
            batch_id = dispatch.get("batch_id")
            provenance = dispatch.get("command_provenance")
            ack = dispatch.get("ack")
            sim_step = dispatch.get("sim_step")
            sim_time_s = dispatch.get("sim_time_s")
            command_epoch = dispatch.get("command_epoch")
            expected_batch_id = (
                f"{self.request.request_id}:residual:"
                f"{int(dispatch.get('physical_command_epoch', -1)):06d}"
                if self.residual_enabled
                and dispatch.get("dispatch_cause") == "RESIDUAL_ONLY"
                and type(dispatch.get("physical_command_epoch")) is int
                else (
                    f"{self.request.request_id}:macro:{command_epoch:06d}"
                    if type(command_epoch) is int
                    else ""
                )
            )
            if (
                dispatch.get("schema_version") != DISPATCH_SCHEMA
                or type(dispatch.get("dispatch_index")) is not int
                or dispatch.get("dispatch_index") != index
                or type(batch_id) is not str
                or not batch_id
                or batch_id != expected_batch_id
                or batch_id in seen_batches
                or type(sim_step) is not int
                or sim_step < 0
                or type(sim_time_s) not in (int, float)
                or type(sim_time_s) is bool
                or not math.isfinite(float(sim_time_s))
                or not math.isclose(
                    float(sim_time_s),
                    float(sim_step) * EXPECTED_PHYSICS_DT_S,
                    rel_tol=0.0,
                    abs_tol=max(1.0e-9, EXPECTED_PHYSICS_DT_S * 1.0e-6),
                )
                or type(command_epoch) is not int
                or command_epoch < 0
                or dispatch.get("n_plus_one_verified") is not True
                or dispatch.get("n_plus_one_verified_sim_step")
                != sim_step + 1
                or not isinstance(ack, Mapping)
                or ack.get("batch_id") != batch_id
                or ack.get("applied_sim_step") != sim_step
                or ack.get("first_physics_step")
                != dispatch.get("n_plus_one_verified_sim_step")
                or not isinstance(provenance, Mapping)
            ):
                errors.append(f"{label} core physical/N+1 identity is invalid")
                continue
            if previous_n_plus_one_step is not None and sim_step < previous_n_plus_one_step:
                errors.append(
                    f"{label} was applied before the prior dispatch N+1 readback"
                )
            previous_n_plus_one_step = sim_step + 1
            ack_metadata = ack.get("recording_metadata")
            base_metadata_keys = {
                "source_version",
                "profile_id",
                "profile_source_version",
                "profile_strategy",
                "macro_state",
                "subphase",
                "command_epoch",
                "bundle_sha256",
                "command_provenance",
                "segment_completion_control",
                "source_plan_sha256",
                "source_action_consumption_index",
            }
            residual_metadata_keys = {
                "nominal_command_epoch",
                "physical_command_epoch",
                "nominal_target_changed",
                "applied_target_changed",
                "dispatch_cause",
                "nominal_servo_targets_deg",
                "nominal_wheel_targets_rad_s",
                "residual_policy_id",
                "residual_policy_sha256",
                "residual_transform_sha256",
                "residual_evidence_sha256",
            }
            expected_metadata_keys = (
                base_metadata_keys | residual_metadata_keys
                if self.residual_enabled
                else base_metadata_keys
            )
            expected_metadata_fields = {
                "source_version": self.request.source_version,
                "profile_id": dispatch.get("profile_id"),
                "profile_source_version": dispatch.get("profile_source_version"),
                "profile_strategy": dispatch.get("profile_strategy"),
                "macro_state": dispatch.get("macro_state"),
                "subphase": dispatch.get("subphase"),
                "command_epoch": command_epoch,
                "bundle_sha256": self.request.bundle_sha256,
                "command_provenance": provenance,
                "source_action_consumption_index": dispatch.get(
                    "source_action_consumption_index"
                ),
            }
            if self.residual_enabled:
                residual_transform = dict(
                    dispatch.get("residual_transform", {}) or {}
                )
                expected_metadata_fields.update(
                    {
                        "nominal_command_epoch": dispatch.get(
                            "nominal_command_epoch"
                        ),
                        "physical_command_epoch": dispatch.get(
                            "physical_command_epoch"
                        ),
                        "nominal_target_changed": dispatch.get(
                            "nominal_target_changed"
                        ),
                        "applied_target_changed": dispatch.get(
                            "applied_target_changed"
                        ),
                        "dispatch_cause": dispatch.get("dispatch_cause"),
                        "nominal_servo_targets_deg": dict(
                            dispatch.get("nominal_servo_targets_deg", {})
                        ),
                        "nominal_wheel_targets_rad_s": dict(
                            dispatch.get("nominal_wheel_targets_rad_s", {})
                        ),
                        "residual_policy_id": self.residual_policy_id,
                        "residual_policy_sha256": self.residual_policy_sha256,
                        "residual_transform_sha256": str(
                            residual_transform.get("core_transform_sha256", "")
                        ),
                        "residual_evidence_sha256": str(
                            residual_transform.get("evidence_sha256", "")
                        ),
                    }
                )
            if (
                set(ack) != _DURABLE_MOTION_BATCH_ACK_KEYS
                or not isinstance(ack_metadata, Mapping)
                or set(ack_metadata) != expected_metadata_keys
                or any(
                    not _strict_json_equal(ack_metadata.get(key), value)
                    for key, value in expected_metadata_fields.items()
                )
            ):
                errors.append(f"{label} durable ACK metadata identity is invalid")
            else:
                try:
                    completion_control, _ = self._validated_completion_control(
                        {
                            "segment_completion_control": ack_metadata[
                                "segment_completion_control"
                            ]
                        }
                    )
                    self._validate_batch_ack(
                        ack,
                        batch_id=batch_id,
                        servo_targets=dispatch.get("servo_targets_deg", {}),
                        wheel_targets=dispatch.get("wheel_targets_rad_s", {}),
                        expected_sim_step=sim_step,
                        expected_physics_dt_s=EXPECTED_PHYSICS_DT_S,
                        expected_source="fsm50_macro_controller",
                        expected_recording_metadata=ack_metadata,
                    )
                    if not _strict_json_equal(
                        completion_control,
                        ack_metadata["segment_completion_control"],
                    ):
                        raise RuntimeError(
                            "segment completion control normalization drifted"
                        )
                except Exception as exc:
                    errors.append(f"{label} durable atomic ACK is invalid: {exc}")
            errors.extend(
                self._durable_target_readback_errors(
                    readback=dispatch.get("n_plus_one_readback", {}),
                    readback_sha256=str(
                        dispatch.get("n_plus_one_readback_sha256", "") or ""
                    ),
                    expected_servo_targets=dispatch.get("servo_targets_deg", {}),
                    expected_wheel_targets=dispatch.get("wheel_targets_rad_s", {}),
                    expected_sim_step=sim_step + 1,
                    expected_command_epoch=command_epoch,
                    expected_batch_id=batch_id,
                    label=f"{label} N+1",
                )
            )
            seen_batches.add(batch_id)
            kind = provenance.get("kind")
            source_index = dispatch.get("source_action_consumption_index")
            feedback_owners = feedback_rows_by_dispatch.get(index, [])
            if (
                not self.residual_enabled
                and command_epoch != prior_verified_command_epoch + 1
            ):
                errors.append(
                    f"{label} command epoch does not advance its prior verified batch"
                )
            if (
                not self.residual_enabled
                and kind
                in {
                    "BOUNDARY_ZERO_WHEELS",
                    "HOLD_ZERO_WHEELS",
                    "SAFE_STOP_ZERO_WHEELS",
                    "SUCCESS_ZERO_WHEELS",
                    "COMPLETION_WHEEL_STOP",
                }
                and (
                    not _strict_json_equal(
                        dispatch.get("servo_targets_deg"),
                        prior_verified_servos,
                    )
                    or not _strict_json_equal(
                        dispatch.get("wheel_targets_rad_s"),
                        {name: 0.0 for name in WHEEL_JOINT_NAMES},
                    )
                )
            ):
                errors.append(
                    f"{label} non-source stop/hold map does not preserve the prior 8-servo authority"
                )
            if kind == "SOURCE_ACTION":
                source_row = source_rows.get(source_index)
                if (
                    type(source_index) is not int
                    or source_row is None
                    or source_row.get("physical_dispatch_applied") is not True
                    or source_row.get("physical_dispatch_index") != index
                    or source_row.get("batch_id") != batch_id
                    or source_row.get("dispatch_epoch")
                    != dispatch.get("command_epoch")
                    or source_row.get("sim_step") != dispatch.get("sim_step")
                    or type(source_row.get("sim_time_s")) not in (int, float)
                    or type(source_row.get("sim_time_s")) is bool
                    or type(dispatch.get("sim_time_s")) not in (int, float)
                    or type(dispatch.get("sim_time_s")) is bool
                    or not math.isclose(
                        float(source_row.get("sim_time_s")),
                        float(dispatch.get("sim_time_s")),
                        rel_tol=0.0,
                        abs_tol=max(
                            1.0e-9, EXPECTED_PHYSICS_DT_S * 1.0e-6
                        ),
                    )
                    or source_row.get("n_plus_one_verified_sim_step")
                    != dispatch.get("n_plus_one_verified_sim_step")
                    or source_row.get("n_plus_one_readback_sha256")
                    != dispatch.get("n_plus_one_readback_sha256")
                    or not _strict_json_equal(
                        source_row.get("command_provenance"), provenance
                    )
                    or feedback_owners
                ):
                    errors.append(
                        f"{label} is not owned by its exact source-action consumption"
                    )
                elif isinstance(ack_metadata, Mapping):
                    expected_action = self.expected_source_actions[source_index]
                    dispatch_kind = provenance.get("dispatch_kind")
                    control = ack_metadata.get("segment_completion_control")
                    metadata_source_ok = bool(
                        ack_metadata.get("source_plan_sha256")
                        == expected_action["source_plan_sha256"]
                        and ack_metadata.get("source_action_consumption_index")
                        == source_index
                    )
                    control_ok = False
                    if isinstance(control, Mapping) and dispatch_kind == "segment_start":
                        binding = expected_action.get("segment_completion_binding")
                        expected_control = {
                            "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
                            "kind": "START",
                            "profile_id": expected_action["profile_id"],
                            "profile_source_version": expected_action[
                                "profile_source_version"
                            ],
                            "owner_state": expected_action["owner_state"],
                            "source_plan_sha256": expected_action[
                                "source_plan_sha256"
                            ],
                            "source_plan_payload_sha256": (
                                binding.get("source_plan_payload_sha256")
                                if isinstance(binding, Mapping)
                                else None
                            ),
                            "accepted_steps_sha256": (
                                binding.get("accepted_steps_sha256")
                                if isinstance(binding, Mapping)
                                else None
                            ),
                            "source_segment_index": provenance.get(
                                "source_segment_index"
                            ),
                            "source_step_index": provenance.get(
                                "source_step_index"
                            ),
                            "source_step_id": (
                                binding.get("completion_spec", {}).get(
                                    "source_step_id"
                                )
                                if isinstance(binding, Mapping)
                                else None
                            ),
                            "start_command_epoch": command_epoch,
                            "completion_spec": (
                                binding.get("completion_spec")
                                if isinstance(binding, Mapping)
                                else None
                            ),
                            "source_action_identity": provenance.get(
                                "source_action_identity"
                            ),
                            "source_action": True,
                            "completion_token_sha256": "",
                        }
                        control_ok = _strict_json_equal(control, expected_control)
                    elif (
                        isinstance(control, Mapping)
                        and dispatch_kind == "wheel_channel_completion_stop"
                    ):
                        matching_segments = [
                            completion
                            for completion in self.segment_completion_rows
                            if completion.get("source_segment_index")
                            == provenance.get("source_segment_index")
                        ]
                        completion = (
                            matching_segments[0]
                            if len(matching_segments) == 1
                            else None
                        )
                        wheel_stop = (
                            completion.get("wheel_stop")
                            if isinstance(completion, Mapping)
                            else None
                        )
                        expected_control = {
                            "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
                            "kind": "WHEEL_STOP",
                            "profile_id": completion.get("profile_id")
                            if isinstance(completion, Mapping)
                            else None,
                            "profile_source_version": completion.get(
                                "profile_source_version"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "owner_state": completion.get("owner_state")
                            if isinstance(completion, Mapping)
                            else None,
                            "source_plan_sha256": completion.get(
                                "source_plan_sha256"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "source_plan_payload_sha256": completion.get(
                                "source_plan_payload_sha256"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "accepted_steps_sha256": completion.get(
                                "accepted_steps_sha256"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "source_segment_index": completion.get(
                                "source_segment_index"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "source_step_index": completion.get("source_step_index")
                            if isinstance(completion, Mapping)
                            else None,
                            "source_step_id": completion.get("source_step_id")
                            if isinstance(completion, Mapping)
                            else None,
                            "start_command_epoch": completion.get(
                                "start_command_epoch"
                            )
                            if isinstance(completion, Mapping)
                            else None,
                            "completion_spec": completion.get("completion_spec")
                            if isinstance(completion, Mapping)
                            else None,
                            "source_action_identity": provenance.get(
                                "source_action_identity"
                            ),
                            "source_action": True,
                            "completion_token_sha256": wheel_stop.get(
                                "completion_token_sha256"
                            )
                            if isinstance(wheel_stop, Mapping)
                            else None,
                        }
                        control_ok = bool(
                            isinstance(wheel_stop, Mapping)
                            and wheel_stop.get("batch_id") == batch_id
                            and _strict_json_equal(control, expected_control)
                        )
                    if not metadata_source_ok or not control_ok:
                        errors.append(
                            f"{label} source ACK metadata/completion authority drifted"
                        )
            elif kind == "FEEDBACK_RECOVERY":
                if source_index is not None or len(feedback_owners) != 1:
                    errors.append(
                        f"{label} is not owned by exactly one feedback action"
                    )
            elif kind == "COMPLETION_WHEEL_STOP":
                completion_control = (
                    ack_metadata.get("segment_completion_control")
                    if isinstance(ack_metadata, Mapping)
                    else None
                )
                owners = generated_wheel_stops_by_batch.get(batch_id, [])
                owner = owners[0] if len(owners) == 1 else None
                matching_segments = [
                    completion
                    for completion in self.segment_completion_rows
                    if isinstance(owner, Mapping)
                    and completion.get("wheel_stop") is owner
                ]
                completion = (
                    matching_segments[0] if len(matching_segments) == 1 else None
                )
                expected_control = {
                    "schema_version": SEGMENT_COMPLETION_CONTROL_SCHEMA_VERSION,
                    "kind": "WHEEL_STOP",
                    "profile_id": completion.get("profile_id")
                    if isinstance(completion, Mapping)
                    else None,
                    "profile_source_version": completion.get(
                        "profile_source_version"
                    )
                    if isinstance(completion, Mapping)
                    else None,
                    "owner_state": completion.get("owner_state")
                    if isinstance(completion, Mapping)
                    else None,
                    "source_plan_sha256": completion.get("source_plan_sha256")
                    if isinstance(completion, Mapping)
                    else None,
                    "source_plan_payload_sha256": completion.get(
                        "source_plan_payload_sha256"
                    )
                    if isinstance(completion, Mapping)
                    else None,
                    "accepted_steps_sha256": completion.get(
                        "accepted_steps_sha256"
                    )
                    if isinstance(completion, Mapping)
                    else None,
                    "source_segment_index": completion.get("source_segment_index")
                    if isinstance(completion, Mapping)
                    else None,
                    "source_step_index": completion.get("source_step_index")
                    if isinstance(completion, Mapping)
                    else None,
                    "source_step_id": completion.get("source_step_id")
                    if isinstance(completion, Mapping)
                    else None,
                    "start_command_epoch": completion.get("start_command_epoch")
                    if isinstance(completion, Mapping)
                    else None,
                    "completion_spec": completion.get("completion_spec")
                    if isinstance(completion, Mapping)
                    else None,
                    "source_action_identity": "",
                    "source_action": False,
                    "completion_token_sha256": owner.get(
                        "completion_token_sha256"
                    )
                    if isinstance(owner, Mapping)
                    else None,
                }
                if (
                    source_index is not None
                    or feedback_owners
                    or len(owners) != 1
                    or not isinstance(completion_control, Mapping)
                    or not _strict_json_equal(completion_control, expected_control)
                    or ack_metadata.get("source_plan_sha256") != ""
                    or ack_metadata.get("source_action_consumption_index") is not None
                ):
                    errors.append(
                        f"{label} lacks its exact generated completion-wheel-stop owner"
                    )
            elif kind in {
                "BOUNDARY_ZERO_WHEELS",
                "HOLD_ZERO_WHEELS",
                "SAFE_STOP_ZERO_WHEELS",
                "SUCCESS_ZERO_WHEELS",
            }:
                candidates = [
                    row
                    for row in self.transition_rows
                    if row.get("sim_step") == sim_step
                    and row.get("to_state") == dispatch.get("macro_state")
                    and row.get("command_epoch") == command_epoch
                ]
                transition = candidates[0] if len(candidates) == 1 else None
                events = (
                    list(transition.get("events", []) or [])
                    if isinstance(transition, Mapping)
                    else []
                )
                from_state = (
                    str(transition.get("from_state", "") or "")
                    if isinstance(transition, Mapping)
                    else ""
                )
                to_state = str(dispatch.get("macro_state", "") or "")
                expected_events = {
                    "BOUNDARY_ZERO_WHEELS": [
                        f"EXIT:{from_state}",
                        f"ENTER:{to_state}",
                    ],
                    "HOLD_ZERO_WHEELS": [f"HOLD:{to_state}"],
                    "SAFE_STOP_ZERO_WHEELS": [f"SAFE_STOP:{from_state}"],
                    "SUCCESS_ZERO_WHEELS": [
                        f"EXIT:{from_state}",
                        "ENTER:SUCCESS",
                    ],
                }[kind]
                if (
                    source_index is not None
                    or feedback_owners
                    or transition is None
                    or events != expected_events
                    or not _strict_json_equal(
                        ack_metadata.get("segment_completion_control"),
                        _EMPTY_SEGMENT_COMPLETION_CONTROL,
                    )
                    or ack_metadata.get("source_plan_sha256") != ""
                    or ack_metadata.get("source_action_consumption_index") is not None
                ):
                    errors.append(
                        f"{label} lacks its exact transition-ledger owner"
                    )
            elif kind == "NONE":
                residual_only_owner = bool(
                    self.residual_enabled
                    and dispatch.get("dispatch_cause") == "RESIDUAL_ONLY"
                    and _strict_json_equal(
                        ack_metadata.get("segment_completion_control"),
                        _EMPTY_SEGMENT_COMPLETION_CONTROL,
                    )
                    and ack_metadata.get("source_plan_sha256") == ""
                    and ack_metadata.get("source_action_consumption_index") is None
                )
                if source_index is not None or feedback_owners or not residual_only_owner:
                    errors.append(
                        f"{label} lacks its exact residual-only non-source owner"
                    )
            else:
                errors.append(f"{label} has an unsupported provenance kind")
            if not self.residual_enabled:
                prior_verified_servos = dict(
                    dispatch.get("servo_targets_deg", {})
                )
                prior_verified_wheels = dict(
                    dispatch.get("wheel_targets_rad_s", {})
                )
                prior_verified_command_epoch = int(command_epoch)
        for source_index, source_row in source_rows.items():
            applied = source_row.get("physical_dispatch_applied")
            dispatch_index = source_row.get("physical_dispatch_index")
            batch_id = source_row.get("batch_id")
            if applied is True:
                if (
                    type(dispatch_index) is not int
                    or dispatch_index < 0
                    or dispatch_index >= len(self.dispatch_rows)
                ):
                    errors.append(
                        f"source action {source_index} lacks its physical dispatch"
                    )
            elif applied is False:
                if dispatch_index is not None or batch_id not in (None, ""):
                    errors.append(
                        f"source action {source_index} falsely claims a nonphysical dispatch"
                    )
            else:
                errors.append(
                    f"source action {source_index} physical dispatch flag is invalid"
                )
        return errors

    def _successful_coverage_errors(self) -> list[str]:
        return [
            *self._source_action_coverage_errors(),
            *self._segment_completion_coverage_errors(),
            *self._feedback_recovery_coverage_errors(),
            *self._dispatch_ownership_errors(),
            *self._feedback_recovery_terminal_errors(),
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

    def _record_runtime_contact_collision_sample(
        self,
        *,
        sample_epoch: int,
        sim_step: int,
        sample_sha256: str,
        collision: bool,
    ) -> None:
        tracker = self.runtime_contact_collision_evidence
        last_epoch = tracker.get("last_sample_epoch")
        if last_epoch is None:
            if sample_epoch != 1 or int(tracker.get("sample_count", 0)) != 0:
                raise RuntimeError(
                    "runtime contact collision ledger did not start at epoch one"
                )
            tracker["first_sample_sim_step"] = sim_step
            tracker["first_sample_sha256"] = sample_sha256
        elif sample_epoch == last_epoch:
            if (
                tracker.get("last_sample_sim_step") != sim_step
                or tracker.get("last_sample_sha256") != sample_sha256
                or tracker.get("last_collision") is not collision
            ):
                raise RuntimeError(
                    "runtime contact collision same-epoch evidence drifted"
                )
            return
        elif sample_epoch != int(last_epoch) + 1:
            raise RuntimeError("runtime contact collision sample epoch is not contiguous")
        tracker["sample_count"] = int(tracker.get("sample_count", 0)) + 1
        tracker["clear_sample_count"] = int(
            tracker.get("clear_sample_count", 0)
        ) + int(not collision)
        tracker["detected_sample_count"] = int(
            tracker.get("detected_sample_count", 0)
        ) + int(collision)
        if collision:
            if tracker.get("first_detected_sim_step") is None:
                tracker["first_detected_sim_step"] = sim_step
                tracker["first_detected_sample_sha256"] = sample_sha256
            tracker["last_detected_sim_step"] = sim_step
            tracker["last_detected_sample_sha256"] = sample_sha256
        tracker["last_sample_epoch"] = sample_epoch
        tracker["last_sample_sim_step"] = sim_step
        tracker["last_sample_sha256"] = sample_sha256
        tracker["last_collision"] = collision

    def _capture_runtime_safety_evidence(self, adapter: Any) -> dict[str, Any]:
        sim_step = int(getattr(adapter, "sim_steps", 0) or 0)
        sample = dict(self.filtered_contact_sample or {})
        rows = list(self.filtered_contact_nonwheel_rows or [])
        sample_sha256 = sample.get("sample_sha256")
        sample_core = dict(sample)
        sample_core.pop("sample_sha256", None)
        if (
            sample.get("sample_sim_step") != sim_step
            or sample.get("sample_sim_step") != self.filtered_contact_last_sim_step
            or sample.get("sample_epoch") != self.filtered_contact_sample_epoch
            or type(sample_sha256) is not str
            or len(sample_sha256) != 64
            or _canonical_json_sha256(sample_core) != sample_sha256
            or not rows
            or sample.get("nonwheel_rows") != rows
            or any(type(row.get("active")) is not bool for row in rows)
        ):
            raise RuntimeError(
                "runtime safety lacks the current exact non-wheel contact sample"
            )
        contact_collision = any(row["active"] is True for row in rows)
        self._record_runtime_contact_collision_sample(
            sample_epoch=int(sample["sample_epoch"]),
            sim_step=sim_step,
            sample_sha256=sample_sha256,
            collision=contact_collision,
        )
        contact_source = (
            "COMBINED_FILTERED_NONWHEEL_OBSTACLE_CONTACT_CURRENT_TICK:"
            + sample_sha256
        )
        capture = getattr(adapter, "capture_macro_runtime_safety_evidence", None)
        if not callable(capture):
            return {
                "available": False,
                # The combined bank covers the configured non-wheel rigid-body
                # obstacle filters.  Its strict >1 N predicate can prove a
                # dangerous body contact (or its absence) independently of a
                # penetration-depth provider.
                "dangerous_body_collision": contact_collision,
                "severe_penetration": None,
                "source": contact_source,
                "sample_sim_step": sim_step,
                "combined_contact_sample_sha256": sample_sha256,
                "nonwheel_contact_row_count": len(rows),
                "filtered_nonwheel_collision": contact_collision,
                "provider_dangerous_body_collision": None,
                "provider_collision_available": False,
                "provider_collision_source": "",
                "provider_collision_sample_sim_step": None,
                "error": "runtime severe-penetration depth evidence is unavailable",
            }
        try:
            raw = capture(scene_handle=self.scene_handle)
        except Exception as exc:
            return {
                "available": False,
                "dangerous_body_collision": contact_collision,
                "severe_penetration": None,
                "source": contact_source,
                "sample_sim_step": sim_step,
                "combined_contact_sample_sha256": sample_sha256,
                "nonwheel_contact_row_count": len(rows),
                "filtered_nonwheel_collision": contact_collision,
                "provider_dangerous_body_collision": None,
                "provider_collision_available": False,
                "provider_collision_source": "",
                "provider_collision_sample_sim_step": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not isinstance(raw, Mapping):
            raise RuntimeError("runtime safety provider did not return a mapping")
        self._runtime_safety_capture_serial += 1
        capture_serial = self._runtime_safety_capture_serial
        collision_claim_preexisting = bool(
            self.unverified_provider_collision_claim
        )
        penetration_claim_preexisting = bool(
            self.unverified_provider_penetration_claim
        )
        optional_claim_preexisting = {
            field: bool(self.unverified_optional_runtime_true_claims[field])
            for field in ("body_stuck", "active_leg_trapped")
        }
        raw_provider_collision: Any
        try:
            raw_provider_collision = raw.get("dangerous_body_collision")
        except Exception:
            raw_provider_collision = None
        try:
            raw_provider_penetration = raw.get("severe_penetration")
        except Exception:
            raw_provider_penetration = None
        raw_provider_collision_true = _safety_scalar_normalizes_true(
            raw_provider_collision
        )
        raw_provider_penetration_true = _safety_scalar_normalizes_true(
            raw_provider_penetration
        )
        raw_optional_true: dict[str, bool] = {}
        raw_optional_claim_created: dict[str, bool] = {}
        for optional_field in ("body_stuck", "active_leg_trapped"):
            try:
                optional_value = raw.get(optional_field)
            except Exception:
                optional_value = None
            raw_optional_true[optional_field] = (
                _safety_scalar_normalizes_true(optional_value)
            )
            raw_optional_claim_created[optional_field] = False
            if raw_optional_true[optional_field]:
                if optional_field == "body_stuck":
                    self.body_stuck_detected = True
                else:
                    self.active_leg_trapped_detected = True
                if not self.unverified_optional_runtime_true_claims[optional_field]:
                    self.unverified_optional_runtime_true_claims[optional_field] = {
                        "classification": (
                            "UNVERIFIED_PROVIDER_"
                            + optional_field.upper()
                            + "_TRUE_CLAIM"
                        ),
                        "field": optional_field,
                        "reported_sample_sim_step": None,
                        "reported_source": "",
                        "provider_payload_sha256": "",
                        "provider_payload_sha256_error": "SNAPSHOT_PENDING",
                        "combined_contact_sample_sha256": sample_sha256,
                    }
                    raw_optional_claim_created[optional_field] = True
                    self._optional_true_claim_capture_serial[optional_field] = (
                        capture_serial
                    )

        # A provider-reported collision is a sticky safety claim as soon as it
        # crosses this capture boundary.  Do not wait for the rest of the
        # observation or hard-safety payload to be built: an unrelated
        # malformed field (or a later COM/readback exception) must never turn
        # a reported collision into terminal ``False`` evidence.
        raw_true_claim_created = False
        if raw_provider_collision_true and not self.unverified_provider_collision_claim:
            # Commit the first exact ``True`` read before taking a mapping
            # snapshot.  A custom/concurrently-mutated Mapping must not be
            # able to report True through ``get`` and then silently turn it
            # into False while ``dict(raw)`` is being constructed.
            self.unverified_provider_collision_claim = {
                "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                "reported_sample_sim_step": None,
                "reported_source": "",
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "SNAPSHOT_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            raw_true_claim_created = True
        raw_penetration_true_claim_created = False
        if (
            raw_provider_penetration_true
            and not self.unverified_provider_penetration_claim
        ):
            self.unverified_provider_penetration_claim = {
                "classification": "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM",
                "reported_sample_sim_step": None,
                "reported_source": "",
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "SNAPSHOT_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            raw_penetration_true_claim_created = True
        try:
            raw_mapping = dict(raw)
        except BaseException as exc:
            if raw_provider_collision_true:
                claim = {
                    "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                    "reported_sample_sim_step": None,
                    "reported_source": "",
                    "provider_payload_sha256": "",
                    "provider_payload_sha256_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "combined_contact_sample_sha256": sample_sha256,
                }
                existing_claim = dict(self.unverified_provider_collision_claim)
                if existing_claim and collision_claim_preexisting:
                    raise RuntimeError(
                        "runtime safety provider unverified collision claim drifted"
                    ) from exc
                self.unverified_provider_collision_claim = claim
            if raw_provider_penetration_true:
                claim = dict(self.unverified_provider_penetration_claim)
                if penetration_claim_preexisting:
                    raise RuntimeError(
                        "runtime safety provider unverified penetration claim drifted"
                    ) from exc
                claim["provider_payload_sha256"] = ""
                claim["provider_payload_sha256_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.unverified_provider_penetration_claim = claim
            for optional_field in ("body_stuck", "active_leg_trapped"):
                if raw_optional_true[optional_field]:
                    if optional_claim_preexisting[optional_field]:
                        raise RuntimeError(
                            "runtime safety provider unverified optional claim drifted"
                        ) from exc
                    claim = dict(
                        self.unverified_optional_runtime_true_claims[optional_field]
                    )
                    claim["provider_payload_sha256_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.unverified_optional_runtime_true_claims[optional_field] = (
                        claim
                    )
            raise
        snapshot_provider_collision = raw_mapping.get(
            "dangerous_body_collision"
        )
        snapshot_provider_penetration = raw_mapping.get("severe_penetration")
        snapshot_provider_collision_true = _safety_scalar_normalizes_true(
            snapshot_provider_collision
        )
        snapshot_provider_penetration_true = _safety_scalar_normalizes_true(
            snapshot_provider_penetration
        )
        snapshot_optional_true = {
            field: _safety_scalar_normalizes_true(raw_mapping.get(field))
            for field in ("body_stuck", "active_leg_trapped")
        }
        for optional_field, optional_true in snapshot_optional_true.items():
            if optional_true:
                if optional_field == "body_stuck":
                    self.body_stuck_detected = True
                else:
                    self.active_leg_trapped_detected = True
                if (
                    raw_optional_claim_created[optional_field]
                    or not self.unverified_optional_runtime_true_claims[
                        optional_field
                    ]
                ):
                    self.unverified_optional_runtime_true_claims[optional_field] = {
                        "classification": (
                            "UNVERIFIED_PROVIDER_"
                            + optional_field.upper()
                            + "_TRUE_CLAIM"
                        ),
                        "field": optional_field,
                        "reported_sample_sim_step": (
                            raw_mapping.get(f"{optional_field}_sample_sim_step")
                        ),
                        "reported_source": (
                            raw_mapping.get(f"{optional_field}_source")
                            if type(raw_mapping.get(f"{optional_field}_source"))
                            is str
                            else ""
                        ),
                        "provider_payload_sha256": "",
                        "provider_payload_sha256_error": "NORMALIZATION_PENDING",
                        "combined_contact_sample_sha256": sample_sha256,
                    }
                    self._optional_true_claim_capture_serial[optional_field] = (
                        capture_serial
                    )
            if raw_optional_true[optional_field] and not optional_true:
                claim = dict(
                    self.unverified_optional_runtime_true_claims[optional_field]
                )
                claim["provider_payload_sha256_error"] = ""
                claim["provider_contract_error"] = (
                    f"provider {optional_field} changed while the mapping "
                    "snapshot was captured"
                )
                self.unverified_optional_runtime_true_claims[optional_field] = claim
                raise RuntimeError(
                    f"runtime safety provider true {optional_field} claim "
                    "drifted during snapshot"
                )
        snapshot_true_claim_created = False
        if (
            snapshot_provider_collision_true
            and (
                raw_true_claim_created
                or not self.unverified_provider_collision_claim
            )
        ):
            # ``Mapping.get`` is not an authoritative snapshot: a custom or
            # concurrently-mutated mapping may return False there and expose
            # True while ``dict(raw)`` is materialized.  Commit the exact True
            # snapshot before recursively normalizing any unrelated field.
            self.unverified_provider_collision_claim = {
                "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                "reported_sample_sim_step": (
                    raw_mapping.get("sample_sim_step")
                    if type(raw_mapping.get("sample_sim_step")) is int
                    else None
                ),
                "reported_source": (
                    raw_mapping.get("source")
                    if type(raw_mapping.get("source")) is str
                    else ""
                ),
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "NORMALIZATION_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            snapshot_true_claim_created = True
        snapshot_penetration_true_claim_created = False
        if (
            snapshot_provider_penetration_true
            and (
                raw_penetration_true_claim_created
                or not self.unverified_provider_penetration_claim
            )
        ):
            self.unverified_provider_penetration_claim = {
                "classification": "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM",
                "reported_sample_sim_step": (
                    raw_mapping.get("sample_sim_step")
                    if type(raw_mapping.get("sample_sim_step")) is int
                    else None
                ),
                "reported_source": (
                    raw_mapping.get("source")
                    if type(raw_mapping.get("source")) is str
                    else ""
                ),
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "NORMALIZATION_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            snapshot_penetration_true_claim_created = True
        if raw_provider_collision_true and not snapshot_provider_collision_true:
            if not collision_claim_preexisting:
                self.unverified_provider_collision_claim = {
                    "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                    "reported_sample_sim_step": raw_mapping.get("sample_sim_step"),
                    "reported_source": str(raw_mapping.get("source", "") or ""),
                    "provider_payload_sha256": "",
                    "provider_payload_sha256_error": "",
                    "provider_contract_error": (
                        "provider dangerous_body_collision changed while the "
                        "mapping snapshot was captured"
                    ),
                    "combined_contact_sample_sha256": sample_sha256,
                }
            raise RuntimeError(
                "runtime safety provider true collision claim drifted during snapshot"
            )
        if (
            raw_provider_penetration_true
            and not snapshot_provider_penetration_true
        ):
            if not penetration_claim_preexisting:
                self.unverified_provider_penetration_claim = {
                    "classification": "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM",
                    "reported_sample_sim_step": raw_mapping.get("sample_sim_step"),
                    "reported_source": str(raw_mapping.get("source", "") or ""),
                    "provider_payload_sha256": "",
                    "provider_payload_sha256_error": "",
                    "provider_contract_error": (
                        "provider severe_penetration changed while the mapping snapshot was captured"
                    ),
                    "combined_contact_sample_sha256": sample_sha256,
                }
            raise RuntimeError(
                "runtime safety provider true penetration claim drifted during snapshot"
            )
        try:
            result = dict(_jsonable(raw_mapping) or {})
        except BaseException as exc:
            if (
                raw_provider_collision_true
                or snapshot_provider_collision_true
            ) and not collision_claim_preexisting:
                claim = dict(self.unverified_provider_collision_claim)
                claim["provider_payload_sha256"] = ""
                claim["provider_payload_sha256_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.unverified_provider_collision_claim = claim
            if (
                raw_provider_penetration_true
                or snapshot_provider_penetration_true
            ) and not penetration_claim_preexisting:
                claim = dict(self.unverified_provider_penetration_claim)
                claim["provider_payload_sha256"] = ""
                claim["provider_payload_sha256_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                self.unverified_provider_penetration_claim = claim
            for optional_field in ("body_stuck", "active_leg_trapped"):
                if (
                    raw_optional_true[optional_field]
                    or snapshot_optional_true[optional_field]
                ) and not optional_claim_preexisting[optional_field]:
                    claim = dict(
                        self.unverified_optional_runtime_true_claims[optional_field]
                    )
                    claim["provider_payload_sha256_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.unverified_optional_runtime_true_claims[optional_field] = (
                        claim
                    )
            raise
        provider_available = result.get("available")
        provider_step = result.get("sample_sim_step")
        provider_collision = result.get("dangerous_body_collision")
        provider_penetration = result.get("severe_penetration")
        provider_source = str(result.get("source", "") or "").strip()
        if (
            raw_provider_collision_true
            or snapshot_provider_collision_true
        ) and provider_collision is not True:
            if not collision_claim_preexisting:
                self.unverified_provider_collision_claim = {
                    "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                    "reported_sample_sim_step": provider_step,
                    "reported_source": provider_source,
                    "provider_payload_sha256": "",
                    "provider_payload_sha256_error": "",
                    "provider_contract_error": (
                        "provider dangerous_body_collision changed while the "
                        "mapping snapshot was captured"
                    ),
                    "combined_contact_sample_sha256": sample_sha256,
                }
            raise RuntimeError(
                "runtime safety provider true collision claim drifted during snapshot"
            )
        if (
            raw_provider_penetration_true
            or snapshot_provider_penetration_true
        ) and provider_penetration is not True:
            if not penetration_claim_preexisting:
                self.unverified_provider_penetration_claim = {
                    "classification": "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM",
                    "reported_sample_sim_step": provider_step,
                    "reported_source": provider_source,
                    "provider_payload_sha256": "",
                    "provider_payload_sha256_error": "",
                    "provider_contract_error": (
                        "provider severe_penetration changed while the mapping snapshot was captured"
                    ),
                    "combined_contact_sample_sha256": sample_sha256,
                }
            raise RuntimeError(
                "runtime safety provider true penetration claim drifted during snapshot"
            )
        penetration_provisional_claim_created = bool(
            raw_penetration_true_claim_created
            or snapshot_penetration_true_claim_created
        )
        if provider_penetration is True and (
            penetration_provisional_claim_created
            or not self.unverified_provider_penetration_claim
        ):
            self.unverified_provider_penetration_claim = {
                "classification": "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM",
                "reported_sample_sim_step": provider_step,
                "reported_source": provider_source,
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "VALIDATION_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            penetration_provisional_claim_created = True
        provisional_true_claim_created = bool(
            raw_true_claim_created or snapshot_true_claim_created
        )
        if provider_collision is True and provisional_true_claim_created:
            # The snapshot confirmed the preliminary exact-True read.  Enrich
            # the already-durable claim with snapshot provenance before any
            # remaining contract validation or canonical hashing can fail.
            self.unverified_provider_collision_claim = {
                "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                "reported_sample_sim_step": provider_step,
                "reported_source": provider_source,
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "VALIDATION_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
        elif provider_collision is True and not self.unverified_provider_collision_claim:
            # Persist the raw safety assertion before validating or hashing
            # any other provider field.  Even a hostile/non-canonical extra
            # value must not make a reported collision disappear from the
            # durable terminal classification.
            self.unverified_provider_collision_claim = {
                "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                "reported_sample_sim_step": provider_step,
                "reported_source": provider_source,
                "provider_payload_sha256": "",
                "provider_payload_sha256_error": "VALIDATION_PENDING",
                "combined_contact_sample_sha256": sample_sha256,
            }
            provisional_true_claim_created = True
        provider_true_contract_error = ""
        if provider_collision is True or provider_penetration is True:
            try:
                self._validate_deployment_safety_evidence(
                    result,
                    expected_sim_step=sim_step,
                    require_clear=False,
                )
            except BaseException as exc:
                provider_true_contract_error = f"{type(exc).__name__}: {exc}"
        if (
            provider_collision is True or provider_penetration is True
        ) and provider_true_contract_error:
            try:
                provider_payload_sha256 = _canonical_json_sha256(result)
                provider_payload_sha256_error = ""
            except BaseException as exc:
                provider_payload_sha256 = ""
                provider_payload_sha256_error = f"{type(exc).__name__}: {exc}"
            claim = {
                "classification": "UNVERIFIED_PROVIDER_TRUE_CLAIM",
                "reported_sample_sim_step": provider_step,
                "reported_source": provider_source,
                "provider_payload_sha256": provider_payload_sha256,
                "provider_payload_sha256_error": provider_payload_sha256_error,
                "provider_contract_error": provider_true_contract_error,
                "combined_contact_sample_sha256": sample_sha256,
            }
            existing_claim = dict(self.unverified_provider_collision_claim)
            if (
                existing_claim
                and not provisional_true_claim_created
                and existing_claim != claim
            ):
                raise RuntimeError(
                    "runtime safety provider unverified collision claim drifted"
                )
            if provider_collision is True:
                if provisional_true_claim_created:
                    self.unverified_provider_collision_claim = claim
            if provider_penetration is True:
                if penetration_provisional_claim_created:
                    self.unverified_provider_penetration_claim = {
                        **claim,
                        "classification": (
                            "UNVERIFIED_PROVIDER_SEVERE_PENETRATION_TRUE_CLAIM"
                        ),
                    }
            raise RuntimeError(
                "runtime safety provider true collision claim is not a valid current-tick contract"
                if provider_collision is True
                else "runtime safety provider true penetration claim is not a valid current-tick contract"
            )
        if provider_available not in (None, False, True) or type(
            provider_available
        ) not in (bool, type(None)):
            raise RuntimeError("runtime safety provider availability is malformed")
        if provider_collision not in (None, False, True) or type(
            provider_collision
        ) not in (bool, type(None)):
            raise RuntimeError("runtime dangerous-body collision value is malformed")
        if provider_available is True and (
            type(provider_step) is not int or provider_step != sim_step
        ):
            raise RuntimeError("runtime safety provider sample is stale")
        if provider_step is not None and (
            type(provider_step) is not int or provider_step != sim_step
        ):
            raise RuntimeError("runtime safety provider sample step mismatch")
        result["dangerous_body_collision"] = bool(
            contact_collision or provider_collision is True
        )
        result.setdefault("sample_sim_step", sim_step)
        result["source"] = ";".join(
            part
            for part in (provider_source, contact_source)
            if part
        )
        result["combined_contact_sample_sha256"] = sample_sha256
        result["nonwheel_contact_row_count"] = len(rows)
        result["filtered_nonwheel_collision"] = contact_collision
        result["provider_dangerous_body_collision"] = provider_collision
        result["provider_collision_available"] = provider_available is True
        result["provider_collision_source"] = provider_source
        result["provider_collision_sample_sim_step"] = provider_step
        if provider_collision is True:
            detection = {
                "sample_sim_step": sim_step,
                "source": result["source"],
                "combined_contact_sample_sha256": sample_sha256,
                "runtime_safety_evidence_sha256": _canonical_json_sha256(result),
            }
            if not self.dangerous_collision_detection_evidence:
                self.dangerous_collision_detection_evidence = detection
            self.dangerous_collision_detected = True
            if provisional_true_claim_created:
                self.unverified_provider_collision_claim = {}
        if provider_penetration is True:
            penetration_detection = {
                "sample_sim_step": sim_step,
                "source": result["source"],
                "combined_contact_sample_sha256": sample_sha256,
                "runtime_safety_evidence_sha256": _canonical_json_sha256(
                    result
                ),
            }
            if not self.severe_penetration_detection_evidence:
                self.severe_penetration_detection_evidence = (
                    penetration_detection
                )
            self.severe_penetration_detected = True
            if penetration_provisional_claim_created:
                self.unverified_provider_penetration_claim = {}
        return result

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
        last_step = tracker.get("last_sample_sim_step")
        if last_step is not None:
            if sample_sim_step < int(last_step):
                raise RuntimeError(f"runtime {field} sample step regressed")
            if sample_sim_step == int(last_step):
                if (
                    tracker.get("last_value") is not value
                    or tracker.get("last_available") is not available
                    or (available and tracker.get("source") != source)
                ):
                    raise RuntimeError(
                        f"runtime {field} same-step evidence drifted"
                    )
                return {
                    field: value,
                    available_key: available,
                    source_key: source,
                    step_key: sample_sim_step,
                }
        tracker["sample_count"] = int(tracker["sample_count"]) + 1
        if tracker.get("first_sample_sim_step") is None:
            tracker["first_sample_sim_step"] = sample_sim_step
        tracker["last_sample_sim_step"] = sample_sim_step
        tracker["last_value"] = value
        tracker["last_available"] = available
        if available:
            prior_source = str(tracker.get("source", "") or "")
            if prior_source and prior_source != source:
                raise RuntimeError(f"runtime {field} producer source drift")
            tracker["source"] = source
            tracker["available_sample_count"] = int(
                tracker["available_sample_count"]
            ) + 1
            tracker["detected_sample_count"] = int(
                tracker["detected_sample_count"]
            ) + int(value is True)
            if value is True:
                if not tracker.get("first_detection_evidence"):
                    tracker["first_detection_evidence"] = {
                        "sample_sim_step": sample_sim_step,
                        "source": source,
                        "combined_contact_sample_sha256": str(
                            evidence.get(
                                "combined_contact_sample_sha256", ""
                            )
                            or ""
                        ),
                        "runtime_safety_evidence_sha256": (
                            _canonical_json_sha256(evidence)
                        ),
                    }
                if field == "body_stuck":
                    self.body_stuck_detected = True
                elif field == "active_leg_trapped":
                    self.active_leg_trapped_detected = True
                # The complete current-tick contract has now been validated
                # and recorded in the optional evidence ledger.  It safely
                # supersedes the provisional pre-normalization True claim.
                if (
                    self._optional_true_claim_capture_serial.get(field)
                    == self._runtime_safety_capture_serial
                ):
                    self.unverified_optional_runtime_true_claims[field] = {}
                    self._optional_true_claim_capture_serial[field] = None
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
        detected_count = int(
            tracker.get("detected_sample_count", 0) or 0
        )
        terminal_step = int(getattr(self.adapter, "sim_steps", -1))
        terminal_time = float(
            getattr(self.adapter, "sim_time", float("nan"))
        )
        first_step = tracker.get("first_sample_sim_step")
        last_step = tracker.get("last_sample_sim_step")
        unverified_true_claim = dict(
            self.unverified_optional_runtime_true_claims[field]
        )
        terminal_time_valid = bool(
            terminal_step >= 0
            and math.isfinite(terminal_time)
            and self.physics_dt_s is not None
            and math.isclose(
                terminal_time,
                terminal_step * float(self.physics_dt_s),
                rel_tol=0.0,
                abs_tol=max(1.0e-9, float(self.physics_dt_s) * 1.0e-6),
            )
        )
        complete_live_coverage = bool(
            sample_count > 0
            and available_count == sample_count
            and detected_count == 0
            and type(first_step) is int
            and type(last_step) is int
            and int(last_step) - int(first_step) + 1 == sample_count
            and int(last_step) == terminal_step
            and terminal_time_valid
            and not unverified_true_claim
        )
        effective_detected = bool(
            detected or detected_count > 0 or unverified_true_claim
        )
        available = bool(effective_detected or complete_live_coverage)
        value = True if effective_detected else False if complete_live_coverage else None
        source = (
            str(
                unverified_true_claim.get("reported_source", "")
                or unverified_true_claim.get("classification", "")
                or tracker.get("source", "")
                or ""
            )
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
            f"{field}_detected_sample_count": detected_count,
            f"{field}_complete_live_coverage": complete_live_coverage,
            f"{field}_first_sample_sim_step": first_step,
            f"{field}_last_sample_sim_step": last_step,
            f"{field}_terminal_adapter_sim_step": terminal_step,
            f"{field}_terminal_adapter_sim_time_s": terminal_time,
            f"{field}_terminal_time_identity_valid": terminal_time_valid,
            f"{field}_unverified_true_claim": unverified_true_claim,
            f"{field}_detection_evidence": dict(
                tracker.get("first_detection_evidence", {}) or {}
            ),
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
        standing_pose_deg_by_servo = {
            name: math.degrees(float(servo_command[name]))
            - float(JOINT_COMMAND_SIGN[name]) * float(command_servos[name])
            for name in SERVO_JOINT_NAMES
        }
        servo_command_transform = {
            "schema_version": _SERVO_COMMAND_TRANSFORM_SCHEMA,
            "standing_pose_deg_by_servo": standing_pose_deg_by_servo,
            "command_sign_by_servo": {
                name: float(JOINT_COMMAND_SIGN[name])
                for name in SERVO_JOINT_NAMES
            },
        }
        if not self.durable_servo_command_transform:
            self.durable_servo_command_transform = dict(
                _jsonable(servo_command_transform)
            )
        elif not _strict_json_equal(
            self.durable_servo_command_transform,
            servo_command_transform,
        ):
            raise RuntimeError(
                "servo command transform changed within one worker session"
            )
        if pending is not None and type(
            pending.get("feedback_recovery_action_index")
        ) is int:
            transform = getattr(adapter, "command_to_actual_target_deg", None)
            if not callable(transform) or any(
                not math.isclose(
                    standing_pose_deg_by_servo[name],
                    float(transform(name, 0.0)),
                    rel_tol=0.0,
                    abs_tol=DRIVE_READBACK_TOLERANCE,
                )
                for name in SERVO_JOINT_NAMES
            ):
                raise RuntimeError(
                    "feedback servo command transform differs from the live adapter calibration"
                )
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
            "command_epoch": (
                self.last_verified_command_epoch
                if pending is None
                else pending.get("command_epoch")
            ),
            "batch_id": (
                str(self.last_target_readback.get("batch_id", "") or "")
                if pending is None
                else str(pending.get("batch_id", ""))
            ),
            "canonical_servo_targets_deg": command_servos,
            "canonical_wheel_targets_rad_s": command_wheels,
            "servo_command_transform": servo_command_transform,
            "servo_command_transform_sha256": _canonical_json_sha256(
                servo_command_transform
            ),
            # This preimage is copied from the independently validated adapter
            # command-target layer above.  Persisting it lets the durable
            # feedback ledger bind the exact PhysX servo readback rather than
            # accepting an arbitrary finite 8-map after the live check.
            "expected_servo_drive_targets_rad": servo_command,
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
            dispatch_row["n_plus_one_readback"] = dict(_jsonable(evidence))
            feedback_action_index = pending.get(
                "feedback_recovery_action_index"
            )
            dispatch_is_feedback = (
                dict(dispatch_row.get("command_provenance", {}) or {}).get(
                    "kind"
                )
                == "FEEDBACK_RECOVERY"
            )
            if (feedback_action_index is not None) != dispatch_is_feedback:
                raise RuntimeError(
                    "pending feedback recovery/readback identity is incomplete"
                )
            if feedback_action_index is not None:
                if (
                    type(feedback_action_index) is not int
                    or feedback_action_index < 0
                    or feedback_action_index
                    >= len(self.feedback_recovery_action_rows)
                ):
                    raise RuntimeError(
                        "pending feedback recovery action index is invalid"
                    )
                action_row = self.feedback_recovery_action_rows[
                    feedback_action_index
                ]
                if (
                    action_row.get("batch_id") != pending["batch_id"]
                    or action_row.get("dispatch_index") != dispatch_index
                    or action_row.get("n_plus_one_verified") is not False
                    or action_row.get("expected_n_plus_one_sim_step")
                    != expected_step
                ):
                    raise RuntimeError(
                        "pending feedback recovery action/readback identity drifted"
                    )
                action_row["n_plus_one_verified"] = True
                action_row["n_plus_one_verified_sim_step"] = expected_step
                action_row["n_plus_one_readback_sha256"] = readback_sha256
                action_row["n_plus_one_readback"] = dict(_jsonable(evidence))
                self.feedback_recovery_verified_action_count += 1
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
            self.boundary_readback = dict(_jsonable(evidence))
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

    def _active_centroidal_leg(self) -> str:
        try:
            leg = _enum_text(
                self.bundle.graph.get(self.last_macro_state).active_leg
            )
        except Exception:
            return ""
        return leg if leg in LEGS else ""

    def _build_centroidal_support_evidence(
        self,
        adapter: Any,
        *,
        active_leg: str,
    ) -> CentroidalSupportEvidence:
        sim_step = int(getattr(adapter, "sim_steps", -1))
        sim_time = float(getattr(adapter, "sim_time", float("nan")))
        frame = self.filtered_contact_frame
        sample = self.filtered_contact_sample
        if (
            frame is None
            or frame.physics_tick != sim_step
            or self.filtered_contact_last_sim_step != sim_step
            or sample.get("sample_sim_step") != sim_step
            or not math.isclose(
                frame.sim_time_s,
                sim_time,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, frame.physics_dt_s * 1.0e-6),
            )
        ):
            raise RuntimeError("centroidal evidence lacks the current combined contact sample")
        expected_body_names = tuple(getattr(adapter.robot, "body_names", ()))
        com = self.filtered_contact_com
        if (
            com is None
            or not com.available
            or com.physics_tick != sim_step
            or com.body_names != expected_body_names
        ):
            raise RuntimeError(
                "centroidal evidence lacks a current whole-body COM measurement"
            )
        angular_rate = measure_isaac_centroidal_angular_momentum_rate(
            adapter,
            com,
            env_id=0,
            expected_body_names=expected_body_names,
        )
        thresholds = frame.thresholds
        support = assess_support_region(com, frame, thresholds=thresholds)
        wrench = assess_contact_wrench_feasibility(
            com,
            frame,
            angular_momentum_rate=angular_rate,
        )
        diagonal = assess_primary_diagonal_support(
            com,
            frame,
            active_swing_leg=active_leg,
            wrench_feasibility=wrench,
            thresholds=thresholds,
            require_wrench_feasibility=True,
        )
        evidence = CentroidalSupportEvidence.create(
            sim_step=sim_step,
            physics_time_s=sim_time,
            physics_dt_s=float(self.physics_dt_s),
            whole_body_com=com,
            centroidal_angular_momentum_rate=angular_rate,
            wheel_contacts=frame,
            support_region=support,
            contact_wrench_feasibility=wrench,
            diagonal_support=diagonal,
        )
        # Reparse the JSON boundary so the controller never receives an
        # envelope that only passed in-memory dataclass construction.
        reparsed = CentroidalSupportEvidence.from_mapping(evidence.to_mapping())
        self.last_centroidal_support_evidence = reparsed
        return reparsed

    @staticmethod
    def _command_space_servo_state(
        adapter: Any,
        joint_q: Mapping[str, float],
        joint_qd: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        positions: dict[str, float] = {}
        velocities: dict[str, float] = {}
        margins: dict[str, float] = {}
        records = dict(getattr(adapter, "safe_joint_limit_records", {}) or {})
        for name in SERVO_JOINT_NAMES:
            actual_position_deg = math.degrees(float(joint_q[name]))
            actual_velocity_deg_s = math.degrees(float(joint_qd[name]))
            zero_actual_deg = float(adapter.command_to_actual_target_deg(name, 0.0))
            sign = float(JOINT_COMMAND_SIGN[name])
            positions[name] = sign * (actual_position_deg - zero_actual_deg)
            velocities[name] = sign * actual_velocity_deg_s
            record = records.get(name)
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    f"feedback recovery lacks safe joint-limit record for {name}"
                )
            lower = float(record.get("min_rad", float("nan")))
            upper = float(record.get("max_rad", float("nan")))
            actual = float(joint_q[name])
            margin = min(actual - lower, upper - actual)
            if not all(math.isfinite(value) for value in (lower, upper, actual, margin)):
                raise RuntimeError(f"feedback recovery joint margin is non-finite at {name}")
            margins[name] = max(0.0, math.degrees(margin))
        return positions, velocities, margins

    def _feedback_recovery_observation_mapping(
        self,
        *,
        adapter: Any,
        joint_q: Mapping[str, float],
        joint_qd: Mapping[str, float],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Importing the controller remains lazy for pre-Isaac CLI imports.
        from .fsm50_macro_controller import FeedbackRecoveryObservation

        readback = dict(self.last_target_readback or {})
        current_sim_step = int(getattr(adapter, "sim_steps", -1))
        if (
            type(readback.get("sim_step")) is not int
            or readback.get("sim_step") != current_sim_step
        ):
            raise RuntimeError(
                "feedback recovery PhysX readback is not from the current physics tick"
            )
        readback_servos = self._validated_target_map(
            readback.get("canonical_servo_targets_deg"),
            names=SERVO_JOINT_NAMES,
            label="feedback recovery PhysX readback servo targets",
        )
        readback_wheels = self._validated_target_map(
            readback.get("canonical_wheel_targets_rad_s"),
            names=WHEEL_JOINT_NAMES,
            label="feedback recovery PhysX readback wheel targets",
        )
        measured_positions, measured_velocities, margins = (
            self._command_space_servo_state(adapter, joint_q, joint_qd)
        )
        # The observed epoch is the epoch of the current, independently
        # verified physical target map.  A post-dispatch bookkeeping failure
        # can leave ``last_epoch`` uncommitted even though PhysX accepted and
        # N+1 verified the batch; using that stale logical value would prevent
        # the mandatory safe-stop from ever being observed or closed.
        observed_epoch = int(
            self.last_verified_command_epoch
            if self.last_verified_command_epoch is not None
            else 0
        )
        verified_epoch = (
            observed_epoch
            if self.pending_readback is None
            and self.last_verified_command_epoch == observed_epoch
            else None
        )
        readback_epoch = readback.get("command_epoch")
        if readback_epoch is not None and (
            type(readback_epoch) is not int or readback_epoch != observed_epoch
        ):
            raise RuntimeError(
                "feedback recovery PhysX readback command epoch is not current"
            )
        base_position = payload.get("base_position_m")
        if not isinstance(base_position, Mapping) or set(base_position) != {
            "x",
            "y",
            "z",
        }:
            raise RuntimeError("feedback recovery base position is not exact")
        value = FeedbackRecoveryObservation.create(
            sim_step=int(getattr(adapter, "sim_steps", -1)),
            physics_time_s=float(getattr(adapter, "sim_time", float("nan"))),
            observed_command_epoch=observed_epoch,
            n_plus_one_verified=verified_epoch is not None,
            verified_command_epoch=verified_epoch,
            readback_servo_targets_deg=readback_servos,
            readback_wheel_targets_rad_s=readback_wheels,
            measured_servo_positions_deg=measured_positions,
            measured_servo_velocities_deg_s=measured_velocities,
            joint_limit_margin_deg=margins,
            base_position_m=[
                base_position["x"],
                base_position["y"],
                base_position["z"],
            ],
            base_roll_rad=payload.get("base_roll_rad"),
            base_pitch_rad=payload.get("base_pitch_rad"),
            base_angular_velocity_rad_s=payload.get("root_angular_velocity_w"),
            wheel_center_w_m=payload.get("wheel_center_w_m"),
            wheel_front_face_clearance_m=payload.get(
                "wheel_front_face_clearance_m"
            ),
            wheel_top_clearance_m=payload.get("wheel_top_clearance_m"),
            obstacle_front_face_x_m=payload.get("obstacle_front_face_x_m"),
            obstacle_top_z_m=payload.get("obstacle_top_z_m"),
            body_crossed_front_face=payload.get("body_crossed_front_face"),
            final_recoverable=payload.get("final_recoverable"),
            posture_complete=payload.get("posture_complete"),
        )
        return FeedbackRecoveryObservation.from_mapping(value.to_mapping()).to_mapping()

    def _observation_payload(self, adapter: Any) -> dict[str, Any]:
        sim_step = int(getattr(adapter, "sim_steps", 0) or 0)
        if self.filtered_contact_last_sim_step != sim_step:
            self._refresh_filtered_contact_evidence(adapter)
        # Runtime safety may consume the same sensor/cache surface.  Capture it
        # only after the combined wheel/non-wheel bank has atomically published
        # the current completed physics tick, never before a stale refresh.
        safety_evidence = self._capture_runtime_safety_evidence(adapter)
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
        active_leg = self._active_centroidal_leg()
        centroidal = self._build_centroidal_support_evidence(
            adapter,
            active_leg=active_leg,
        )
        contact_by_leg = centroidal.wheel_contacts.by_leg()
        measured_contact_loads = {
            leg: (
                float(contact_by_leg[leg].normal_load_n or 0.0)
                if contact_by_leg[leg].evidence_available
                else None
            )
            for leg in LEGS
        }
        physical_support_legs = tuple(
            leg for leg in LEGS if contact_by_leg[leg].support_qualified
        )
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
        geometry_support_legs = tuple(
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
            and centroidal.whole_body_com.available
            and len(physical_support_legs) >= 2
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
            "wheel_contact_load_n": measured_contact_loads,
            "support_legs": physical_support_legs,
            "geometry_support_legs": geometry_support_legs,
            "geometry_support_candidate_count": len(geometry_support_legs),
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
                and all(
                    contact_by_leg[leg].measurement.surface_kind
                    == "OBSTACLE_TOP"
                    and contact_by_leg[leg].support_qualified
                    for leg in LEGS
                )
            ),
            "stability_state": stability,
            "final_velocity_stable": velocity_stable,
            "filtered_contact_bank_enabled": True,
            "filtered_contact_sample": dict(
                _jsonable(self.filtered_contact_sample)
            ),
            "wheel_contact_load_available": centroidal.wheel_contacts.available,
            "com_measurement_available": centroidal.whole_body_com.available,
            "com_position_m": (
                None
                if not centroidal.whole_body_com.available
                else list(centroidal.whole_body_com.position_w_m)
            ),
            "com_proxy_source": "",
            "centroidal_support_evidence": centroidal.to_mapping(),
        }
        payload["feedback_recovery_observation"] = (
            self._feedback_recovery_observation_mapping(
                adapter=adapter,
                joint_q=joint_q,
                joint_qd=joint_qd,
                payload=payload,
            )
        )
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
        collision_now = payload.get("dangerous_body_collision") is True
        collision_detection: dict[str, Any] | None = None
        if collision_now:
            collision_source = runtime_safety.get("source")
            collision_step = runtime_safety.get("sample_sim_step")
            if (
                type(collision_source) is not str
                or not collision_source.strip()
                or type(collision_step) is not int
                or collision_step != payload.get("sim_step")
            ):
                raise RuntimeError(
                    "dangerous collision detection lacks exact current provenance"
                )
            collision_detection = {
                "sample_sim_step": collision_step,
                "source": collision_source.strip(),
                "combined_contact_sample_sha256": str(
                    runtime_safety.get("combined_contact_sample_sha256", "") or ""
                ),
                "runtime_safety_evidence_sha256": _canonical_json_sha256(
                    runtime_safety
                ),
            }
        if collision_detection is not None and not self.dangerous_collision_detection_evidence:
            self.dangerous_collision_detection_evidence = collision_detection
        self.dangerous_collision_detected = bool(
            self.dangerous_collision_detected or collision_now
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
        # Feedback recovery owns an independently bounded diagnostic target.
        # A residual policy may observe it but must never perturb that probe or
        # increment map.
        force_zero_residual = terminal or provenance_kind == "FEEDBACK_RECOVERY"
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

        if kind == "FEEDBACK_RECOVERY":
            if consumed is not False or target_changed is not True:
                raise RuntimeError(
                    "FEEDBACK_RECOVERY must change one target without consuming source"
                )
            for name, expected in {
                "source_action_identity": "",
                "source_version": "",
                "source_segment_index": None,
                "source_step_index": None,
                "source_time_s": None,
                "source_event_indices": [],
                "commands": [],
                "dispatch_kind": "",
                "sequence_index": None,
            }.items():
                if not _strict_json_equal(provenance.get(name), expected):
                    raise RuntimeError(
                        "FEEDBACK_RECOVERY cannot claim Recording source identity"
                    )
            for name in (
                "recovery_stage",
                "recovery_action",
                "recovery_leg",
                "recovery_joint",
            ):
                if type(provenance.get(name)) is not str or not provenance[name]:
                    raise RuntimeError(f"FEEDBACK_RECOVERY {name} is missing")
            for name in (
                "recovery_evidence_sha256",
                "recovery_centroidal_evidence_sha256",
                "recovery_feedback_observation_sha256",
                "recovery_target_map_sha256",
                "recovery_configuration_sha256",
            ):
                digest = provenance.get(name)
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in digest)
                ):
                    raise RuntimeError(f"FEEDBACK_RECOVERY {name} is invalid")
            attempt = provenance.get("recovery_attempt")
            leg = provenance.get("recovery_leg")
            joint = provenance.get("recovery_joint")
            stage = provenance.get("recovery_stage")
            action = provenance.get("recovery_action")
            direction_sign = provenance.get("recovery_direction_sign")
            allowed_stage_actions = {
                "SAFE_PROBE": "CONSERVATIVE_DIAGNOSTIC_PROBE",
                "RETURN_TO_REFERENCE": "RETURN_TO_IMMUTABLE_REFERENCE",
                "INCREMENT": "BOUNDED_DESCENT_INCREMENT",
            }
            leg_joints = {
                "FL": {"front_left_hip", "front_left_knee"},
                "FR": {"front_right_hip", "front_right_knee"},
                "RL": {"rear_left_hip", "rear_left_knee"},
                "RR": {"rear_right_hip", "rear_right_knee"},
            }
            if (
                type(attempt) is not int
                or attempt < 1
                or leg not in leg_joints
                or joint not in leg_joints[leg]
                or type(direction_sign) is not int
                or direction_sign not in {-1, 1}
                or allowed_stage_actions.get(stage) != action
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY stage/action/sign/attempt/leg/joint is invalid"
                )
            centroidal_evidence = payload.get("centroidal_support_evidence")
            feedback_observation = payload.get("feedback_recovery_observation")
            if (
                not isinstance(centroidal_evidence, Mapping)
                or not isinstance(feedback_observation, Mapping)
                or provenance["recovery_centroidal_evidence_sha256"]
                != centroidal_evidence.get("payload_sha256")
                or provenance["recovery_feedback_observation_sha256"]
                != feedback_observation.get("payload_sha256")
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY does not bind the current physical evidence"
                )
            expected_evidence_sha256 = _canonical_json_sha256(
                {
                    "schema_version": "fsm50.feedback_recovery_evidence_binding.v1",
                    "centroidal_support_evidence_sha256": provenance[
                        "recovery_centroidal_evidence_sha256"
                    ],
                    "feedback_recovery_observation_sha256": provenance[
                        "recovery_feedback_observation_sha256"
                    ],
                }
            )
            if provenance["recovery_evidence_sha256"] != expected_evidence_sha256:
                raise RuntimeError(
                    "FEEDBACK_RECOVERY composite evidence SHA is invalid"
                )
            expected_target_map_sha256 = _canonical_json_sha256(
                {
                    "schema_version": "fsm50.feedback_recovery_target_map.v1",
                    "servo_targets_deg": dict(servos),
                    "wheel_targets_rad_s": dict(wheels),
                }
            )
            if (
                provenance["recovery_target_map_sha256"]
                != expected_target_map_sha256
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY target-map SHA differs from the 8+4 decision"
                )
            previous = self.last_servo_targets
            changed_names = (
                []
                if previous is None
                else [
                    name
                    for name in SERVO_JOINT_NAMES
                    if float(servos[name]) != float(previous[name])
                ]
            )
            if (
                not self.outer_render_boundary_permit
                or self.pending_readback is not None
                or self.last_macro_state != "S10_POSTURE_RECOVERY"
                or state != self.last_macro_state
                or mapping.get("terminal") is not False
                or list(mapping.get("transition_events", ()) or ())
                or self.active_segment_completion_row_index is not None
                or self.next_source_action_index != len(self.expected_source_actions)
                or changed_names != [joint]
                or any(float(value) != 0.0 for value in wheels.values())
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY differs from its source-free S10 boundary contract"
                )
            return provenance, None

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
            "pre_action_verified_command_epoch": self.last_target_readback.get(
                "command_epoch"
            ),
            "pre_action_verified_readback": dict(
                _jsonable(self.last_target_readback)
            ),
            "pre_action_verified_readback_sha256": (
                _canonical_json_sha256(self.last_target_readback)
                if self.last_target_readback
                else ""
            ),
        }
        return provenance, row

    @staticmethod
    def _feedback_target_maps_equal(
        actual: Mapping[str, float], expected: Mapping[str, float]
    ) -> bool:
        return bool(
            isinstance(actual, Mapping)
            and isinstance(expected, Mapping)
            and _strict_json_equal(dict(actual), dict(expected))
        )

    @staticmethod
    def _feedback_probe_matrix_choice(
        *,
        sequence: Mapping[str, Any],
        safe_pairs: tuple[tuple[str, int], ...],
    ) -> tuple[str, int]:
        completed_rows = [
            (str(item[0]), int(item[1]))
            for item in sequence.get("completed_probe_pairs", [])
        ]
        result_rows = [
            dict(item) for item in sequence.get("probe_results", [])
        ]
        result_pairs = [
            (str(item["joint"]), int(item["direction_sign"]))
            for item in result_rows
            if item.get("n_plus_one_response_verified") is True
        ]
        if (
            len(completed_rows) != len(set(completed_rows))
            or len(result_pairs) != len(set(result_pairs))
            or set(completed_rows) != set(safe_pairs)
            or set(result_pairs) != set(safe_pairs)
        ):
            raise RuntimeError(
                "FEEDBACK_RECOVERY lacks the complete independently verified safe probe/return matrix"
            )
        choices = sorted(
            (
                float(result["dz_m"]),
                str(result["joint"]),
                int(result["direction_sign"]),
            )
            for result in result_rows
            if (
                result.get("n_plus_one_response_verified") is True
                and result.get("sign_response_valid") is True
                and result.get("baseline_preserved") is True
                and float(result["dz_m"])
                <= -float(
                    FINAL_RECOVERY_FEEDBACK_LIMITS["minimum_descent_m"]
                )
                and (
                    str(result["joint"]),
                    int(result["direction_sign"]),
                )
                in set(completed_rows)
            )
        )
        if not choices:
            raise RuntimeError(
                "FEEDBACK_RECOVERY has no independently measured safe descent response"
            )
        return choices[0][1], choices[0][2]

    def _feedback_durable_configuration_context(
        self,
        *,
        row: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        """Rebuild one recovery configuration only from its durable preimages.

        Runtime sequence state is deliberately not consulted here.  The first
        physical SAFE_PROBE dispatch is the authority for the immutable
        configuration and physical baseline used to re-evaluate every later
        N+1 response in the same configuration.
        """

        configuration = row.get("configuration_payload")
        if (
            not isinstance(configuration, Mapping)
            or set(configuration) != _FEEDBACK_RECOVERY_CONFIGURATION_KEYS
            or configuration.get("schema_version")
            != _FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA
            or _canonical_json_sha256(configuration)
            != row.get("configuration_sha256")
        ):
            raise RuntimeError(
                "durable feedback configuration preimage/SHA is invalid"
            )
        configuration = dict(_jsonable(configuration))
        configuration_sha = str(row["configuration_sha256"])
        grouped = [
            candidate
            for candidate in self.feedback_recovery_action_rows
            if candidate.get("configuration_sha256") == configuration_sha
        ]
        if not grouped:
            raise RuntimeError("durable feedback configuration has no actions")
        grouped.sort(key=lambda candidate: candidate.get("action_index", -1))
        first = grouped[0]
        if (
            first.get("stage") != "SAFE_PROBE"
            or first.get("action") != "CONSERVATIVE_DIAGNOSTIC_PROBE"
            or not _strict_json_equal(
                first.get("configuration_payload"), configuration
            )
            or any(
                not _strict_json_equal(
                    candidate.get("configuration_payload"), configuration
                )
                for candidate in grouped
            )
        ):
            raise RuntimeError(
                "durable feedback configuration does not begin with one bound SAFE_PROBE"
            )
        try:
            first_centroidal = CentroidalSupportEvidence.from_mapping(
                first.get("dispatch_centroidal_support_evidence", {})
            )
            from .fsm50_macro_controller import FeedbackRecoveryObservation

            first_feedback = FeedbackRecoveryObservation.from_mapping(
                first.get("dispatch_feedback_recovery_observation", {})
            )
        except Exception as exc:
            raise RuntimeError(
                "durable feedback configuration first-dispatch evidence is invalid"
            ) from exc
        leg = configuration.get("leg")
        reference = configuration.get("servo_reference_targets_deg")
        measured = configuration.get("measured_servo_positions_deg")
        centers = configuration.get("wheel_center_w_m")
        if (
            leg not in LEGS
            or configuration.get("macro_state") != "S10_POSTURE_RECOVERY"
            or configuration.get("selected_source_version")
            != self.request.source_version
            or configuration.get("centroidal_evidence_sha256")
            != first_centroidal.payload_sha256
            or configuration.get("feedback_observation_sha256")
            != first_feedback.payload_sha256
            or not isinstance(reference, Mapping)
            or set(reference) != set(SERVO_JOINT_NAMES)
            or not isinstance(measured, Mapping)
            or set(measured) != set(SERVO_JOINT_NAMES)
            or not isinstance(centers, Mapping)
            or set(centers) != set(LEGS)
            or not self._feedback_target_maps_equal(
                reference, first_feedback.readback_servo_targets_deg
            )
            or not self._feedback_target_maps_equal(
                measured, first_feedback.measured_servo_positions_deg
            )
            or not _strict_json_equal(
                centers,
                {
                    item: list(first_feedback.wheel_center_w_m[item])
                    for item in LEGS
                },
            )
            or configuration.get("body_crossed_front_face")
            is not first_feedback.body_crossed_front_face
            or configuration.get("final_recoverable")
            is not first_feedback.final_recoverable
            or configuration.get("posture_complete")
            is not first_feedback.posture_complete
        ):
            raise RuntimeError(
                "durable feedback configuration differs from first dispatch evidence"
            )
        profile_actions = [
            candidate
            for candidate in self.expected_source_actions
            if candidate.get("owner_state") == "S10_POSTURE_RECOVERY"
            and candidate.get("profile_id")
            == configuration.get("reference_profile_id")
            and candidate.get("profile_source_version")
            == configuration.get("reference_profile_source_version")
            and candidate.get("source_plan_sha256")
            == configuration.get("reference_profile_source_plan_sha256")
        ]
        source_indices = [
            candidate.get("source_action_index") for candidate in profile_actions
        ]
        if (
            not profile_actions
            or any(type(value) is not int for value in source_indices)
            or len(source_indices) != len(set(source_indices))
        ):
            raise RuntimeError(
                "durable feedback configuration profile identity is not canonical"
            )
        terminal_reference_action = max(
            profile_actions,
            key=lambda candidate: int(candidate["source_action_index"]),
        )
        if not self._feedback_target_maps_equal(
            reference,
            terminal_reference_action.get("servo_targets_deg", {}),
        ):
            raise RuntimeError(
                "durable feedback reference differs from the canonical terminal S10 source action"
            )
        support = self._validate_feedback_recovery_safety(
            centroidal=first_centroidal,
            feedback=first_feedback,
        )
        physical_baseline = {
            "wheel_x_m": float(first_feedback.wheel_center_w_m[leg][0]),
            "wheel_z_m": float(first_feedback.wheel_center_w_m[leg][2]),
            "measured_servo_positions_deg": dict(
                first_feedback.measured_servo_positions_deg
            ),
            "qualified_legs": tuple(support["qualified_legs"]),
            "support_stability_margin_m": support[
                "support_stability_margin_m"
            ],
            "wrench_force_residual_norm_n": support[
                "wrench_force_residual_norm_n"
            ],
            "wrench_moment_residual_norm_nm": support[
                "wrench_moment_residual_norm_nm"
            ],
            "wrench_maximum_friction_utilization": support[
                "wrench_maximum_friction_utilization"
            ],
            "abs_roll_rad": abs(first_feedback.base_roll_rad),
            "abs_pitch_rad": abs(first_feedback.base_pitch_rad),
            "joint_limit_margin_deg": dict(
                first_feedback.joint_limit_margin_deg
            ),
        }
        return configuration, {
            name: float(reference[name]) for name in SERVO_JOINT_NAMES
        }, physical_baseline

    @staticmethod
    def _feedback_physical_support_snapshot(
        centroidal: CentroidalSupportEvidence,
    ) -> dict[str, Any]:
        by_leg = centroidal.wheel_contacts.by_leg()
        qualified = tuple(leg for leg in LEGS if by_leg[leg].support_qualified)
        region = centroidal.support_region
        validated = tuple(region.support_legs)
        support_set_bound = validated == qualified
        wrench = centroidal.contact_wrench_feasibility
        wrench_proven = bool(
            wrench.status.value == "PROVEN"
            and wrench.proven_feasible
            and wrench.physics_tick == centroidal.sim_step
        )
        hull_viable = bool(
            support_set_bound
            and len(validated) in {3, 4}
            and region.status.value == "PROVEN"
            and region.model.value == "STRICT_COPLANAR_CONVEX_HULL"
            and region.signed_margin_m is not None
            and region.signed_margin_m >= 0.0
        )
        diagonal_name = (
            "FL_RR"
            if frozenset(validated) == frozenset(("FL", "RR"))
            else "FR_RL"
        )
        diagonal_viable = bool(
            support_set_bound
            and len(validated) == 2
            and frozenset(validated)
            in {frozenset(("FL", "RR")), frozenset(("FR", "RL"))}
            and region.status.value == "PROVEN"
            and region.model.value == "DIAGONAL_LINE_SEGMENT"
            and region.diagonal == diagonal_name
            and region.between_contacts is True
            and region.finite_patch_approximation
            and region.corridor_signed_margin_m is not None
            and region.corridor_signed_margin_m >= 0.0
        )
        multi_height_viable = bool(
            support_set_bound
            and len(validated) in {3, 4}
            and region.status.value == "NOT_PROVEN"
            and region.model.value
            == "MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED"
            and wrench_proven
        )
        support_margin = (
            region.signed_margin_m
            if hull_viable
            else region.corridor_signed_margin_m
            if diagonal_viable
            else None
        )
        return {
            "qualified_legs": qualified,
            "support_viable": bool(
                centroidal.whole_body_com.available
                and len(qualified) >= 2
                and (hull_viable or diagonal_viable or multi_height_viable)
            ),
            "wrench_proven": wrench_proven,
            "support_stability_margin_m": support_margin,
            "wrench_force_residual_norm_n": wrench.force_residual_norm_n,
            "wrench_moment_residual_norm_nm": wrench.moment_residual_norm_nm,
            "wrench_maximum_friction_utilization": (
                wrench.maximum_friction_utilization
            ),
        }

    @staticmethod
    def _feedback_baseline_preserved(
        *,
        sequence: Mapping[str, Any],
        centroidal: CentroidalSupportEvidence,
        feedback: Any,
        joint: str,
    ) -> tuple[bool, tuple[str, ...]]:
        baseline = dict(sequence.get("physical_baseline", {}) or {})
        return WorkerMacroFSMSession._feedback_baseline_preserved_values(
            leg=str(sequence.get("leg", "") or ""),
            baseline=baseline,
            centroidal=centroidal,
            feedback=feedback,
            joint=joint,
        )

    @staticmethod
    def _feedback_baseline_preserved_values(
        *,
        leg: str,
        baseline: Mapping[str, Any],
        centroidal: CentroidalSupportEvidence,
        feedback: Any,
        joint: str,
    ) -> tuple[bool, tuple[str, ...]]:
        baseline = dict(baseline or {})
        if not baseline:
            return False, ("immutable feedback physical baseline is unavailable",)
        support = WorkerMacroFSMSession._feedback_physical_support_snapshot(
            centroidal
        )
        reasons: list[str] = []
        baseline_support = set(baseline["qualified_legs"])
        current_support = set(support["qualified_legs"])
        if not (baseline_support - {leg}).issubset(current_support):
            reasons.append("qualified support set worsened")
        if (
            abs(feedback.base_roll_rad) > baseline["abs_roll_rad"] + 1.0e-9
            or abs(feedback.base_pitch_rad)
            > baseline["abs_pitch_rad"] + 1.0e-9
        ):
            reasons.append("body attitude worsened")
        if (
            feedback.joint_limit_margin_deg[joint] + 1.0e-9
            < baseline["joint_limit_margin_deg"][joint]
        ):
            reasons.append("probed-joint limit margin worsened")
        baseline_margin = baseline["support_stability_margin_m"]
        current_margin = support["support_stability_margin_m"]
        if (
            baseline_margin is not None
            and current_margin is not None
            and current_margin + 1.0e-9 < baseline_margin
        ):
            reasons.append("support-region stability margin worsened")
        for key, label in (
            ("wrench_force_residual_norm_n", "force residual"),
            ("wrench_moment_residual_norm_nm", "moment residual"),
            (
                "wrench_maximum_friction_utilization",
                "friction utilization",
            ),
        ):
            baseline_value = baseline[key]
            current_value = support[key]
            if (
                baseline_value is not None
                and current_value is not None
                and current_value > baseline_value + 1.0e-9
            ):
                reasons.append(f"contact-wrench {label} worsened")
        return not reasons, tuple(reasons)

    @staticmethod
    def _validate_feedback_recovery_safety(
        *,
        centroidal: CentroidalSupportEvidence,
        feedback: Any,
    ) -> dict[str, Any]:
        support = WorkerMacroFSMSession._feedback_physical_support_snapshot(
            centroidal
        )
        state = build_default_macro_graph().get(
            MacroStateId.S10_POSTURE_RECOVERY
        )
        failures: list[str] = []
        if (
            abs(feedback.base_roll_rad)
            > state.completion_guard.maximum_abs_roll_rad
            or abs(feedback.base_pitch_rad)
            > state.completion_guard.maximum_abs_pitch_rad
        ):
            failures.append("feedback recovery attitude safety bound exceeded")
        if not support["support_viable"] or not support["wrench_proven"]:
            failures.append(
                "feedback recovery lost proven support/wrench feasibility"
            )
        if min(feedback.joint_limit_margin_deg.values()) < float(
            FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"]
        ):
            failures.append("feedback recovery joint-limit margin is unsafe")
        if failures:
            raise RuntimeError("; ".join(failures))
        return support

    def _feedback_recovery_profile_identity(
        self, mapping: Mapping[str, Any]
    ) -> tuple[str, str, str]:
        profile_id = str(mapping.get("profile_id", "") or "")
        profile_source = str(
            mapping.get("profile_source_version", "") or ""
        )
        matches = {
            (
                str(row.get("profile_id", "") or ""),
                str(row.get("profile_source_version", "") or ""),
                str(row.get("source_plan_sha256", "") or ""),
            )
            for row in self.expected_source_actions
            if row.get("owner_state") == "S10_POSTURE_RECOVERY"
            and row.get("profile_id") == profile_id
            and row.get("profile_source_version") == profile_source
        }
        if len(matches) != 1:
            raise RuntimeError(
                "FEEDBACK_RECOVERY lacks one exact S10 reference profile identity"
            )
        return next(iter(matches))

    def _prepare_feedback_recovery_action(
        self,
        *,
        provenance: Mapping[str, Any],
        mapping: Mapping[str, Any],
        payload: Mapping[str, Any],
        servos: Mapping[str, float],
        wheels: Mapping[str, float],
    ) -> dict[str, Any]:
        from .fsm50_macro_controller import FeedbackRecoveryObservation

        try:
            centroidal = CentroidalSupportEvidence.from_mapping(
                payload.get("centroidal_support_evidence", {})
            )
            feedback = FeedbackRecoveryObservation.from_mapping(
                payload.get("feedback_recovery_observation", {})
            )
        except Exception as exc:
            raise RuntimeError(
                "FEEDBACK_RECOVERY physical evidence failed strict revalidation"
            ) from exc
        if (
            feedback.sim_step != centroidal.sim_step
            or not math.isclose(
                feedback.physics_time_s,
                centroidal.physics_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, centroidal.physics_dt_s * 1.0e-6),
            )
            or feedback.observed_command_epoch
            != self.last_verified_command_epoch
            or feedback.verified_command_epoch
            != self.last_verified_command_epoch
            or feedback.n_plus_one_verified is not True
            or self.last_verified_servo_targets is None
            or self.last_verified_wheel_targets is None
            or not self._feedback_target_maps_equal(
                feedback.readback_servo_targets_deg,
                self.last_verified_servo_targets,
            )
            or not self._feedback_target_maps_equal(
                feedback.readback_wheel_targets_rad_s,
                self.last_verified_wheel_targets,
            )
        ):
            raise RuntimeError(
                "FEEDBACK_RECOVERY is not based on the current verified 8+4 readback"
            )
        support_snapshot = self._validate_feedback_recovery_safety(
            centroidal=centroidal,
            feedback=feedback,
        )
        if (
            self.last_servo_targets is None
            or self.last_wheel_targets is None
            or self.last_applied_servo_targets is None
            or self.last_applied_wheel_targets is None
            or not self._feedback_target_maps_equal(
                self.last_applied_servo_targets, self.last_servo_targets
            )
            or not self._feedback_target_maps_equal(
                self.last_applied_wheel_targets, self.last_wheel_targets
            )
            or not self._feedback_target_maps_equal(
                self.last_servo_targets, self.last_verified_servo_targets
            )
            or not self._feedback_target_maps_equal(
                self.last_wheel_targets, self.last_verified_wheel_targets
            )
            or any(float(value) != 0.0 for value in self.last_wheel_targets.values())
            or any(
                float(value) != 0.0
                for value in self.last_applied_wheel_targets.values()
            )
            or (
                self.residual_enabled
                and any(
                    float(value) != 0.0
                    for value in self.last_applied_residual
                )
            )
        ):
            raise RuntimeError(
                "FEEDBACK_RECOVERY requires a zero-residual, zero-wheel verified baseline"
            )

        attempt = int(provenance["recovery_attempt"])
        if (
            attempt != len(self.feedback_recovery_action_rows) + 1
            or attempt > FEEDBACK_RECOVERY_MAXIMUM_ACTIONS
        ):
            raise RuntimeError(
                "FEEDBACK_RECOVERY attempts must start at one, remain contiguous, and stay bounded"
            )
        leg = str(provenance["recovery_leg"])
        joint = str(provenance["recovery_joint"])
        sign = int(provenance["recovery_direction_sign"])
        stage = str(provenance["recovery_stage"])
        action = str(provenance["recovery_action"])
        configuration = str(provenance["recovery_configuration_sha256"])
        leg_joints = {
            "FR": ("front_right_hip", "front_right_knee"),
            "FL": ("front_left_hip", "front_left_knee"),
            "RR": ("rear_right_hip", "rear_right_knee"),
            "RL": ("rear_left_hip", "rear_left_knee"),
        }
        if stage == "INCREMENT":
            contact = centroidal.wheel_contacts.by_leg()[leg]
            if (
                contact.evidence_available
                and contact.measurement.active
                and contact.measurement.surface_kind == "OBSTACLE_TOP"
                and contact.normal_load_n is not None
                and contact.normal_load_n
                >= centroidal.wheel_contacts.thresholds.minimum_normal_force_n
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY must hold a newly load-bearing TOP contact for dwell"
                )
        sequence = self.feedback_recovery_sequence_by_configuration.get(
            configuration
        )
        new_configuration = sequence is None
        if new_configuration:
            if stage != "SAFE_PROBE" or action != "CONSERVATIVE_DIAGNOSTIC_PROBE":
                raise RuntimeError(
                    "a new FEEDBACK_RECOVERY configuration must begin with SAFE_PROBE"
                )
            if leg in self.feedback_recovery_configuration_by_leg:
                raise RuntimeError(
                    "FEEDBACK_RECOVERY cannot reuse a completed leg configuration"
                )
            if configuration in self.feedback_recovery_configuration_by_leg.values():
                raise RuntimeError(
                    "FEEDBACK_RECOVERY configuration SHA cannot be shared across legs"
                )
            contact_by_leg = centroidal.wheel_contacts.by_leg()
            eligible_legs = tuple(
                candidate_leg
                for candidate_leg in ("FR", "FL", "RR", "RL")
                if (
                    contact_by_leg[candidate_leg].evidence_available
                    and not contact_by_leg[candidate_leg].measurement.active
                    and contact_by_leg[candidate_leg].measurement.surface_kind
                    == "AIR"
                    and feedback.wheel_center_w_m[candidate_leg][0]
                    > feedback.obstacle_front_face_x_m
                    and feedback.wheel_center_w_m[candidate_leg][2]
                    >= feedback.obstacle_top_z_m - 0.02
                    and not contact_by_leg[candidate_leg].support_qualified
                )
            )
            if not eligible_legs or eligible_legs[0] != leg:
                raise RuntimeError(
                    "FEEDBACK_RECOVERY first probe leg is not the deterministic current AIR/crossed/top-height candidate"
                )
            if self.feedback_recovery_active_leg:
                prior_leg = self.feedback_recovery_active_leg
                prior = centroidal.wheel_contacts.by_leg()[prior_leg]
                measurement = prior.measurement
                if not (
                    prior.support_qualified
                    and measurement.active
                    and measurement.surface_kind == "OBSTACLE_TOP"
                    and measurement.surface_dwell_verified
                    and measurement.dwell_s is not None
                    and measurement.dwell_s
                    >= FEEDBACK_RECOVERY_CONTACT_DWELL_S
                ):
                    raise RuntimeError(
                        "FEEDBACK_RECOVERY cannot change legs before current TOP load dwell"
                    )
            profile_id, profile_source, source_plan_sha = (
                self._feedback_recovery_profile_identity(mapping)
            )
            configuration_payload = {
                "schema_version": _FEEDBACK_RECOVERY_CONFIGURATION_SCHEMA,
                "leg": leg,
                "macro_state": "S10_POSTURE_RECOVERY",
                "selected_source_version": self.request.source_version,
                "reference_profile_id": profile_id,
                "reference_profile_source_version": profile_source,
                "reference_profile_source_plan_sha256": source_plan_sha,
                "centroidal_evidence_sha256": centroidal.payload_sha256,
                "feedback_observation_sha256": feedback.payload_sha256,
                "servo_reference_targets_deg": dict(self.last_servo_targets),
                "measured_servo_positions_deg": dict(
                    feedback.measured_servo_positions_deg
                ),
                "wheel_center_w_m": {
                    item: list(feedback.wheel_center_w_m[item])
                    for item in LEGS
                },
                "body_crossed_front_face": feedback.body_crossed_front_face,
                "final_recoverable": feedback.final_recoverable,
                "posture_complete": feedback.posture_complete,
            }
            expected_configuration = _canonical_json_sha256(
                configuration_payload
            )
            if configuration != expected_configuration:
                raise RuntimeError(
                    "FEEDBACK_RECOVERY configuration SHA is not independently reproducible"
                )
            sequence = {
                "leg": leg,
                "configuration_payload": dict(_jsonable(configuration_payload)),
                "reference_targets_deg": dict(self.last_servo_targets),
                "physical_baseline": {
                    "wheel_x_m": feedback.wheel_center_w_m[leg][0],
                    "wheel_z_m": feedback.wheel_center_w_m[leg][2],
                    "measured_servo_positions_deg": dict(
                        feedback.measured_servo_positions_deg
                    ),
                    "qualified_legs": tuple(
                        support_snapshot["qualified_legs"]
                    ),
                    "support_stability_margin_m": support_snapshot[
                        "support_stability_margin_m"
                    ],
                    "wrench_force_residual_norm_n": support_snapshot[
                        "wrench_force_residual_norm_n"
                    ],
                    "wrench_moment_residual_norm_nm": support_snapshot[
                        "wrench_moment_residual_norm_nm"
                    ],
                    "wrench_maximum_friction_utilization": support_snapshot[
                        "wrench_maximum_friction_utilization"
                    ],
                    "abs_roll_rad": abs(feedback.base_roll_rad),
                    "abs_pitch_rad": abs(feedback.base_pitch_rad),
                    "joint_limit_margin_deg": dict(
                        feedback.joint_limit_margin_deg
                    ),
                },
                "probe_index": 0,
                "awaiting_return": False,
                "last_probe_joint": "",
                "last_probe_sign": 0,
                "completed_probe_pairs": [],
                "probe_results": [],
                "increment_joint": "",
                "increment_sign": 0,
                "increment_count": 0,
            }
        else:
            sequence = dict(sequence)
            if (
                sequence.get("leg") != leg
                or not isinstance(
                    sequence.get("configuration_payload"), Mapping
                )
                or _canonical_json_sha256(
                    sequence["configuration_payload"]
                )
                != configuration
                or self.feedback_recovery_configuration_by_leg.get(leg)
                != configuration
                or self.feedback_recovery_active_leg != leg
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY configuration/leg sequence identity drifted"
                )

        reference = dict(sequence["reference_targets_deg"])
        probe_order = (
            (leg_joints[leg][0], 1),
            (leg_joints[leg][0], -1),
            (leg_joints[leg][1], 1),
            (leg_joints[leg][1], -1),
        )

        def target_map_within_feedback_limits(
            candidate_targets: Mapping[str, float],
        ) -> bool:
            margin = float(
                FINAL_RECOVERY_FEEDBACK_LIMITS["joint_limit_margin_deg"]
            )
            return all(
                command_limits_for_servo(name)[0] + margin
                <= float(target)
                <= command_limits_for_servo(name)[1] - margin
                for name, target in candidate_targets.items()
            )

        def probe_target_within_limits(candidate_joint: str, candidate_sign: int) -> bool:
            candidate_targets = dict(reference)
            candidate_targets[candidate_joint] += (
                candidate_sign * FEEDBACK_RECOVERY_PROBE_DELTA_DEG
            )
            return target_map_within_feedback_limits(candidate_targets)

        expected_targets = dict(reference)
        if stage == "SAFE_PROBE":
            probe_index = int(sequence["probe_index"])
            while (
                probe_index < len(probe_order)
                and probe_order[probe_index] != (joint, sign)
            ):
                skipped_joint, skipped_sign = probe_order[probe_index]
                if probe_target_within_limits(skipped_joint, skipped_sign):
                    raise RuntimeError(
                        "FEEDBACK_RECOVERY cannot skip a command-limit-safe probe"
                    )
                probe_index += 1
            if (
                sequence["awaiting_return"] is not False
                or sequence["increment_joint"]
                or probe_index >= len(probe_order)
                or (joint, sign) != probe_order[probe_index]
                or not probe_target_within_limits(joint, sign)
            ):
                raise RuntimeError("FEEDBACK_RECOVERY probe sequence is not exact")
            sequence["probe_index"] = probe_index
            expected_targets[joint] += sign * FEEDBACK_RECOVERY_PROBE_DELTA_DEG
            sequence["awaiting_return"] = True
            sequence["last_probe_joint"] = joint
            sequence["last_probe_sign"] = sign
        elif stage == "RETURN_TO_REFERENCE":
            if (
                sequence["awaiting_return"] is not True
                or joint != sequence["last_probe_joint"]
                or sign != sequence["last_probe_sign"]
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY return does not match its preceding probe"
                )
            sequence["awaiting_return"] = False
            sequence["probe_index"] = int(sequence["probe_index"]) + 1
        elif stage == "INCREMENT":
            probe_index = int(sequence["probe_index"])
            while probe_index < len(probe_order):
                skipped_joint, skipped_sign = probe_order[probe_index]
                if probe_target_within_limits(skipped_joint, skipped_sign):
                    break
                probe_index += 1
            sequence["probe_index"] = probe_index
            if (
                sequence["awaiting_return"] is not False
                or probe_index != len(probe_order)
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY increment precedes the complete probe matrix"
                )
            if not sequence["increment_joint"]:
                safe_pairs = tuple(
                    pair
                    for pair in probe_order
                    if probe_target_within_limits(pair[0], pair[1])
                )
                selected_pair = self._feedback_probe_matrix_choice(
                    sequence=sequence,
                    safe_pairs=safe_pairs,
                )
                if (joint, sign) not in set(safe_pairs):
                    raise RuntimeError(
                        "FEEDBACK_RECOVERY selected descent was not physically probed and returned"
                    )
                if (joint, sign) != selected_pair:
                    raise RuntimeError(
                        "FEEDBACK_RECOVERY selected descent is not the deterministic best independently measured response"
                    )
                sequence["increment_joint"] = joint
                sequence["increment_sign"] = sign
            elif (
                sequence["increment_joint"] != joint
                or sequence["increment_sign"] != sign
            ):
                raise RuntimeError(
                    "FEEDBACK_RECOVERY selected descent joint/sign drifted"
                )
            increment_count = int(sequence["increment_count"]) + 1
            if increment_count > FEEDBACK_RECOVERY_MAXIMUM_INCREMENTS_PER_LEG:
                raise RuntimeError("FEEDBACK_RECOVERY per-leg increment bound exceeded")
            sequence["increment_count"] = increment_count
            expected_targets[joint] += (
                sign * FEEDBACK_RECOVERY_INCREMENT_DELTA_DEG * increment_count
            )
        else:
            raise RuntimeError("FEEDBACK_RECOVERY dispatch stage is invalid")
        if not self._feedback_target_maps_equal(servos, expected_targets):
            raise RuntimeError(
                "FEEDBACK_RECOVERY target map differs from its bounded sequence"
            )
        if not target_map_within_feedback_limits(expected_targets):
            raise RuntimeError(
                "FEEDBACK_RECOVERY target map violates the command-limit margin"
            )
        if any(float(value) != 0.0 for value in wheels.values()):
            raise RuntimeError("FEEDBACK_RECOVERY wheel targets must remain zero")
        return {
            "configuration_sha256": configuration,
            "configuration_payload": dict(
                _jsonable(sequence["configuration_payload"])
            ),
            "new_configuration": new_configuration,
            "leg": leg,
            "sequence": sequence,
            "attempt": attempt,
            "stage": stage,
            "action": action,
            "joint": joint,
            "direction_sign": sign,
            "centroidal_evidence_sha256": centroidal.payload_sha256,
            "feedback_observation_sha256": feedback.payload_sha256,
            "dispatch_centroidal_support_evidence": centroidal.to_mapping(),
            "dispatch_feedback_recovery_observation": feedback.to_mapping(),
        }

    def _commit_feedback_recovery_action(
        self,
        *,
        update: Mapping[str, Any],
        provenance: Mapping[str, Any],
        batch_id: str,
        dispatch_index: int,
        sim_step: int,
        expected_sim_step: int,
        ack: Mapping[str, Any],
    ) -> int:
        configuration = str(update["configuration_sha256"])
        leg = str(update["leg"])
        if update["new_configuration"] is True:
            self.feedback_recovery_configuration_by_leg[leg] = configuration
            self.feedback_recovery_active_leg = leg
        self.feedback_recovery_sequence_by_configuration[configuration] = dict(
            update["sequence"]
        )
        row = {
            "schema_version": FEEDBACK_RECOVERY_ACTION_SCHEMA,
            "action_index": len(self.feedback_recovery_action_rows),
            "attempt": int(update["attempt"]),
            "sim_step": int(sim_step),
            "expected_n_plus_one_sim_step": int(expected_sim_step),
            "stage": str(update["stage"]),
            "action": str(update["action"]),
            "leg": leg,
            "joint": str(update["joint"]),
            "direction_sign": int(update["direction_sign"]),
            "configuration_sha256": configuration,
            "configuration_payload": dict(
                _jsonable(update["configuration_payload"])
            ),
            "centroidal_evidence_sha256": str(
                update["centroidal_evidence_sha256"]
            ),
            "feedback_observation_sha256": str(
                update["feedback_observation_sha256"]
            ),
            "command_provenance": dict(provenance),
            "batch_id": str(batch_id),
            "dispatch_index": int(dispatch_index),
            "dispatch_ack": dict(_jsonable(ack)),
            "ack_sha256": _canonical_json_sha256(ack),
            "n_plus_one_verified": False,
            "n_plus_one_verified_sim_step": None,
            "n_plus_one_readback_sha256": "",
            "n_plus_one_readback": {},
            "dispatch_centroidal_support_evidence": dict(
                update["dispatch_centroidal_support_evidence"]
            ),
            "dispatch_feedback_recovery_observation": dict(
                update["dispatch_feedback_recovery_observation"]
            ),
            "physical_response_verified": False,
            "physical_response_sim_step": None,
            "physical_response_centroidal_evidence_sha256": "",
            "physical_response_feedback_observation_sha256": "",
            "physical_response_centroidal_support_evidence": {},
            "physical_response_feedback_recovery_observation": {},
            "physical_response": {},
        }
        self.feedback_recovery_action_rows.append(row)
        return int(row["action_index"])

    def _capture_feedback_recovery_n_plus_one_response(
        self,
        *,
        payload: Mapping[str, Any],
        sim_step: int,
    ) -> None:
        from .fsm50_macro_controller import FeedbackRecoveryObservation

        pending_rows = [
            row
            for row in self.feedback_recovery_action_rows
            if row.get("n_plus_one_verified") is True
            and row.get("n_plus_one_verified_sim_step") == sim_step
            and row.get("physical_response_verified") is False
        ]
        if not pending_rows:
            return
        if len(pending_rows) != 1:
            raise RuntimeError(
                "feedback recovery has multiple N+1 responses on one physics tick"
            )
        row = pending_rows[0]
        try:
            centroidal = CentroidalSupportEvidence.from_mapping(
                payload.get("centroidal_support_evidence", {})
            )
            feedback = FeedbackRecoveryObservation.from_mapping(
                payload.get("feedback_recovery_observation", {})
            )
        except Exception as exc:
            raise RuntimeError(
                "feedback recovery N+1 physical response failed strict evidence validation"
            ) from exc
        dispatch_index = row.get("dispatch_index")
        if (
            type(dispatch_index) is not int
            or dispatch_index < 0
            or dispatch_index >= len(self.dispatch_rows)
        ):
            raise RuntimeError(
                "feedback recovery N+1 response lacks its dispatch identity"
            )
        dispatch = self.dispatch_rows[dispatch_index]
        if (
            feedback.sim_step != sim_step
            or centroidal.sim_step != sim_step
            or row.get("expected_n_plus_one_sim_step") != sim_step
            or not math.isclose(
                feedback.physics_time_s,
                centroidal.physics_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, centroidal.physics_dt_s * 1.0e-6),
            )
            or feedback.observed_command_epoch != dispatch.get("command_epoch")
            or feedback.verified_command_epoch != dispatch.get("command_epoch")
            or feedback.n_plus_one_verified is not True
            or not self._feedback_target_maps_equal(
                feedback.readback_servo_targets_deg,
                dispatch.get("servo_targets_deg", {}),
            )
            or not self._feedback_target_maps_equal(
                feedback.readback_wheel_targets_rad_s,
                dispatch.get("wheel_targets_rad_s", {}),
            )
        ):
            raise RuntimeError(
                "feedback recovery N+1 response is not bound to its exact epoch/full map"
            )
        self._validate_feedback_recovery_safety(
            centroidal=centroidal,
            feedback=feedback,
        )
        configuration = str(row["configuration_sha256"])
        sequence = self.feedback_recovery_sequence_by_configuration.get(
            configuration
        )
        if not isinstance(sequence, Mapping):
            raise RuntimeError(
                "feedback recovery N+1 response lacks its configuration sequence"
            )
        sequence = dict(sequence)
        joint = str(row["joint"])
        sign = int(row["direction_sign"])
        action = str(row["action"])
        baseline = dict(sequence.get("physical_baseline", {}) or {})
        if not baseline:
            raise RuntimeError(
                "feedback recovery N+1 response lacks its immutable physical baseline"
            )
        response: dict[str, Any]
        if action == "CONSERVATIVE_DIAGNOSTIC_PROBE":
            baseline_q = float(
                baseline["measured_servo_positions_deg"][joint]
            )
            current_q = float(feedback.measured_servo_positions_deg[joint])
            dq = current_q - baseline_q
            dx = float(feedback.wheel_center_w_m[row["leg"]][0]) - float(
                baseline["wheel_x_m"]
            )
            dz = float(feedback.wheel_center_w_m[row["leg"]][2]) - float(
                baseline["wheel_z_m"]
            )
            sign_response_valid = bool(
                abs(dq) >= 0.02 and (dq > 0.0) == (sign > 0)
            )
            baseline_preserved, unsafe_reasons = (
                self._feedback_baseline_preserved(
                    sequence=sequence,
                    centroidal=centroidal,
                    feedback=feedback,
                    joint=joint,
                )
            )
            response = {
                "joint": joint,
                "direction_sign": sign,
                "dq_deg": dq,
                "dx_m": dx,
                "dz_m": dz,
                "sign_response_valid": sign_response_valid,
                "baseline_preserved": baseline_preserved,
                "unsafe_reasons": list(unsafe_reasons),
                "n_plus_one_response_verified": True,
            }
            results = [
                dict(item) for item in sequence.get("probe_results", [])
            ]
            if any(
                item.get("joint") == joint
                and item.get("direction_sign") == sign
                for item in results
            ):
                raise RuntimeError(
                    "feedback recovery probe physical response was duplicated"
                )
            results.append(dict(response))
            sequence["probe_results"] = results
        elif action == "RETURN_TO_IMMUTABLE_REFERENCE":
            baseline_q = float(
                baseline["measured_servo_positions_deg"][joint]
            )
            return_error = abs(
                float(feedback.measured_servo_positions_deg[joint]) - baseline_q
            )
            if return_error > 0.2:
                raise RuntimeError(
                    "feedback probe failed to return to immutable reference"
                )
            matching = [
                item
                for item in sequence.get("probe_results", [])
                if item.get("joint") == joint
                and item.get("direction_sign") == sign
                and item.get("n_plus_one_response_verified") is True
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    "feedback recovery return lacks one verified probe response"
                )
            completed = [
                (str(item[0]), int(item[1]))
                for item in sequence.get("completed_probe_pairs", [])
            ]
            if (joint, sign) in completed:
                raise RuntimeError(
                    "feedback recovery return physical response was duplicated"
                )
            completed.append((joint, sign))
            sequence["completed_probe_pairs"] = completed
            response = {
                "joint": joint,
                "direction_sign": sign,
                "return_error_deg": return_error,
                "n_plus_one_response_verified": True,
            }
        elif action == "BOUNDED_DESCENT_INCREMENT":
            baseline_preserved, unsafe_reasons = (
                self._feedback_baseline_preserved(
                    sequence=sequence,
                    centroidal=centroidal,
                    feedback=feedback,
                    joint=joint,
                )
            )
            if not baseline_preserved:
                raise RuntimeError(
                    "selected descent degraded its independently verified baseline: "
                    + "; ".join(unsafe_reasons)
                )
            response = {
                "joint": joint,
                "direction_sign": sign,
                "baseline_preserved": True,
                "unsafe_reasons": [],
                "n_plus_one_response_verified": True,
            }
        else:
            raise RuntimeError(
                "feedback recovery N+1 response action is invalid"
            )
        self.feedback_recovery_sequence_by_configuration[configuration] = sequence
        row["physical_response_verified"] = True
        row["physical_response_sim_step"] = sim_step
        row["physical_response_centroidal_evidence_sha256"] = (
            centroidal.payload_sha256
        )
        row["physical_response_feedback_observation_sha256"] = (
            feedback.payload_sha256
        )
        row["physical_response_centroidal_support_evidence"] = (
            centroidal.to_mapping()
        )
        row["physical_response_feedback_recovery_observation"] = (
            feedback.to_mapping()
        )
        row["physical_response"] = response
        dispatch["feedback_recovery_physical_response_verified"] = True
        dispatch["feedback_recovery_physical_response_sim_step"] = sim_step
        dispatch["feedback_recovery_physical_response_centroidal_sha256"] = (
            centroidal.payload_sha256
        )
        dispatch["feedback_recovery_physical_response_observation_sha256"] = (
            feedback.payload_sha256
        )
        dispatch["feedback_recovery_physical_response"] = dict(response)

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
        feedback_recovery_update = (
            self._prepare_feedback_recovery_action(
                provenance=provenance,
                mapping=mapping,
                payload=payload,
                servos=servos,
                wheels=wheels,
            )
            if provenance.get("kind") == "FEEDBACK_RECOVERY"
            else None
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
        if (
            provenance.get("kind") == "FEEDBACK_RECOVERY"
            and completion_control.get("kind") != "NONE"
        ):
            raise RuntimeError(
                "FEEDBACK_RECOVERY cannot start or complete a Recording segment"
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
            # Freeze only the validated atomic ACK surface.  Adapter-specific
            # diagnostics are intentionally excluded so a malformed extra
            # field cannot leave an already-applied physical command without
            # its pending N+1/action/dispatch ledger records.
            ack = self._durable_motion_batch_ack(
                ack,
                expected_recording_metadata=recording_metadata,
            )
            dispatch_index = len(self.dispatch_rows)
            feedback_recovery_action_index = None
            if feedback_recovery_update is not None:
                feedback_recovery_action_index = (
                    self._commit_feedback_recovery_action(
                        update=feedback_recovery_update,
                        provenance=provenance,
                        batch_id=batch_id,
                        dispatch_index=dispatch_index,
                        sim_step=current_step,
                        expected_sim_step=int(ack["first_physics_step"]),
                        ack=ack,
                    )
                )
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
            if feedback_recovery_action_index is not None:
                self.pending_readback[
                    "feedback_recovery_action_index"
                ] = feedback_recovery_action_index
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
                "n_plus_one_readback": {},
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
            self.controller_terminal_outcome = outcome
            self.controller_terminal_reason = reason
            if _is_task_success_outcome(outcome):
                coverage_errors = self._successful_coverage_errors()
                if coverage_errors:
                    raise RuntimeError(
                        "Macro task-success terminal lacks exact source-action coverage: "
                        + "; ".join(coverage_errors)
                    )
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

    @staticmethod
    def _durable_motion_batch_ack(
        ack: Mapping[str, Any],
        *,
        expected_recording_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Select the exact, validated ACK contract before artifact commit.

        Production adapters may add diagnostic timing fields.  Those fields
        are deliberately outside the durable atomic 8+4 contract and must not
        be allowed to inject a post-apply normalization failure.  Every value
        selected here has already passed ``_validate_batch_ack``; metadata is
        reconstructed from the trusted pre-dispatch payload rather than an
        adapter-owned extra-field mapping.
        """

        durable = {
            "batch_id": ack.get("batch_id"),
            "source": ack.get("source"),
            "error": ack.get("error"),
            "applied_sim_step": ack.get("applied_sim_step"),
            "first_physics_step": ack.get("first_physics_step"),
            "motion_start_skew_s": ack.get("motion_start_skew_s"),
            "physics_dt_s": ack.get("physics_dt_s"),
            "servo_applied": ack.get("servo_applied"),
            "wheel_applied": ack.get("wheel_applied"),
            "servo_targets_applied": dict(
                ack.get("servo_targets_applied", {})
            ),
            "wheel_targets_applied": dict(
                ack.get("wheel_targets_applied", {})
            ),
            "recording_metadata": dict(
                _jsonable(expected_recording_metadata)
            ),
        }
        if set(durable) != _DURABLE_MOTION_BATCH_ACK_KEYS:
            raise RuntimeError("durable motion batch ACK keys are not exact")
        # Canonicalization now cannot observe arbitrary adapter diagnostics.
        _canonical_json_sha256(durable)
        return durable

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
        controller_status = dict(
            _attribute(self.controller, "status", {}) or {}
        )
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
            "controller_reason": mapping.get("reason", ""),
            "guard_evidence": dict(mapping.get("guard_evidence", {}) or {}),
            "feedback_recovery_stage": controller_status.get(
                "feedback_recovery_stage", ""
            ),
            "feedback_recovery_action_count": controller_status.get(
                "feedback_recovery_action_count", 0
            ),
            "feedback_recovery_exhaustion_reason": controller_status.get(
                "feedback_recovery_exhaustion_reason", ""
            ),
        }
        durable_row = dict(_jsonable(row))
        if (
            force
            and self.rows
            and type(durable_row.get("sim_step")) is int
            and self.rows[-1].get("sim_step") == durable_row["sim_step"]
            and type(durable_row.get("sim_time_s")) in (int, float)
            and not isinstance(durable_row.get("sim_time_s"), bool)
            and type(self.rows[-1].get("sim_time_s")) in (int, float)
            and not isinstance(self.rows[-1].get("sim_time_s"), bool)
            and math.isclose(
                float(self.rows[-1]["sim_time_s"]),
                float(durable_row["sim_time_s"]),
                rel_tol=0.0,
                abs_tol=max(1.0e-9, EXPECTED_PHYSICS_DT_S * 1.0e-6),
            )
        ):
            self.rows[-1] = durable_row
        else:
            self.rows.append(durable_row)

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
        source_consumption_path = (
            self.request.run_dir / "macro_source_action_consumption.jsonl"
        )
        segment_completion_path = (
            self.request.run_dir / "macro_segment_completion_ledger.jsonl"
        )
        feedback_recovery_path = (
            self.request.run_dir / "macro_feedback_recovery_action_ledger.jsonl"
        )
        dispatch_path = self.request.run_dir / "macro_dispatch_ledger.jsonl"
        transition_path = self.request.run_dir / "macro_transition_evidence.jsonl"
        telemetry_path = self.request.run_dir / "minimal_macro_telemetry.jsonl"
        source_consumption_sha256 = (
            _sha256_file(source_consumption_path)
            if source_consumption_path.is_file()
            else ""
        )
        segment_completion_sha256 = (
            _sha256_file(segment_completion_path)
            if segment_completion_path.is_file()
            else ""
        )
        feedback_recovery_sha256 = (
            _sha256_file(feedback_recovery_path)
            if feedback_recovery_path.is_file()
            else ""
        )
        dispatch_sha256 = (
            _sha256_file(dispatch_path) if dispatch_path.is_file() else ""
        )
        transition_sha256 = (
            _sha256_file(transition_path) if transition_path.is_file() else ""
        )
        telemetry_sha256 = (
            _sha256_file(telemetry_path) if telemetry_path.is_file() else ""
        )
        final_raw_row_sha256 = (
            _canonical_json_sha256(final) if final else ""
        )
        feedback_recovery_errors = self._feedback_recovery_coverage_errors()
        feedback_recovery_complete = not feedback_recovery_errors
        feedback_recovery_response_count = sum(
            row.get("physical_response_verified") is True
            for row in self.feedback_recovery_action_rows
        )
        boundary_readback = dict(_jsonable(self.boundary_readback))
        boundary_readback_sha256 = (
            _canonical_json_sha256(boundary_readback)
            if boundary_readback
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
        support_count = len(tuple(final.get("support_legs", ()) or ()))
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
            "controller_terminal_outcome": self.controller_terminal_outcome,
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
            "source_action_consumption_count": completed_source_action_count,
            "source_action_consumption_path": str(source_consumption_path),
            "source_action_consumption_sha256": source_consumption_sha256,
            "start_boundary_ack": dict(_jsonable(self.boundary_ack)),
            "start_boundary_readback": boundary_readback,
            "start_boundary_readback_sha256": boundary_readback_sha256,
            "segment_completion_coverage_complete": (
                segment_completion_complete
            ),
            "segment_completion_coverage_errors": segment_completion_errors,
            "segment_completion_ledger_path": str(segment_completion_path),
            "segment_completion_ledger_sha256": segment_completion_sha256,
            "feedback_recovery_action_ledger_schema_version": (
                FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA
            ),
            "feedback_recovery_action_count": len(
                self.feedback_recovery_action_rows
            ),
            "feedback_recovery_verified_action_count": (
                self.feedback_recovery_verified_action_count
            ),
            "feedback_recovery_physical_response_verified_action_count": (
                feedback_recovery_response_count
            ),
            "feedback_recovery_action_coverage_complete": (
                feedback_recovery_complete
            ),
            "feedback_recovery_action_coverage_errors": (
                feedback_recovery_errors
            ),
            "feedback_recovery_action_ledger_path": str(
                feedback_recovery_path
            ),
            "feedback_recovery_action_ledger_sha256": (
                feedback_recovery_sha256
            ),
            "feedback_recovery_dispatch_ledger_path": str(dispatch_path),
            "feedback_recovery_dispatch_ledger_sha256": dispatch_sha256,
            "feedback_recovery_dispatch_count": len(self.dispatch_rows),
            "transition_evidence_path": str(transition_path),
            "transition_evidence_sha256": transition_sha256,
            "transition_count": len(self.transition_rows),
            "telemetry_jsonl_path": str(telemetry_path),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "telemetry_sample_count": len(self.rows),
            "final_raw_telemetry_row_sha256": final_raw_row_sha256,
            "terminal_recovery_closure": dict(
                _jsonable(self.terminal_recovery_closure)
            ),
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
                "source_action_coverage_errors": coverage_errors,
                "source_action_consumption_path": str(source_consumption_path),
                "source_action_consumption_sha256": source_consumption_sha256,
                "start_boundary_ack": dict(_jsonable(self.boundary_ack)),
                "start_boundary_readback": boundary_readback,
                "start_boundary_readback_sha256": boundary_readback_sha256,
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
                "feedback_recovery_action_ledger_schema_version": (
                    FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA
                ),
                "feedback_recovery_action_count": len(
                    self.feedback_recovery_action_rows
                ),
                "feedback_recovery_verified_action_count": (
                    self.feedback_recovery_verified_action_count
                ),
                "feedback_recovery_physical_response_verified_action_count": (
                    feedback_recovery_response_count
                ),
                "feedback_recovery_action_coverage_complete": (
                    feedback_recovery_complete
                ),
                "feedback_recovery_action_coverage_errors": (
                    feedback_recovery_errors
                ),
                "feedback_recovery_action_ledger_path": str(
                    feedback_recovery_path
                ),
                "feedback_recovery_action_ledger_sha256": (
                    feedback_recovery_sha256
                ),
                "feedback_recovery_dispatch_ledger_path": str(dispatch_path),
                "feedback_recovery_dispatch_ledger_sha256": dispatch_sha256,
                "feedback_recovery_dispatch_count": len(self.dispatch_rows),
                "transition_count": len(self.transition_rows),
                "transition_evidence_path": str(transition_path),
                "transition_evidence_sha256": transition_sha256,
                "telemetry_jsonl_path": str(telemetry_path),
                "telemetry_jsonl_sha256": telemetry_sha256,
                "telemetry_sample_count": len(self.rows),
                "final_raw_telemetry_row_sha256": final_raw_row_sha256,
                "terminal_recovery_closure": dict(
                    _jsonable(self.terminal_recovery_closure)
                ),
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
        final_centroidal: CentroidalSupportEvidence | None = None
        try:
            final_centroidal = CentroidalSupportEvidence.from_mapping(
                final.get("centroidal_support_evidence", {})
            )
        except Exception:
            final_centroidal = None
        final_contact_by_leg = (
            {}
            if final_centroidal is None
            else final_centroidal.wheel_contacts.by_leg()
        )
        final_load_available = bool(
            final_centroidal is not None
            and final_centroidal.wheel_contacts.available
            and set(final_contact_by_leg) == set(LEGS)
        )
        final_all_loaded = bool(
            final_load_available
            and all(final_contact_by_leg[leg].support_qualified for leg in LEGS)
        )
        final_all_top = bool(
            final_load_available
            and all(
                final_contact_by_leg[leg].measurement.surface_kind
                == "OBSTACLE_TOP"
                for leg in LEGS
            )
        )
        collision_tracker = dict(self.runtime_contact_collision_evidence)
        collision_sample_count = int(
            collision_tracker.get("sample_count", 0) or 0
        )
        collision_clear_count = int(
            collision_tracker.get("clear_sample_count", 0) or 0
        )
        collision_detected_count = int(
            collision_tracker.get("detected_sample_count", 0) or 0
        )
        collision_counts_consistent = bool(
            collision_clear_count >= 0
            and collision_detected_count >= 0
            and collision_clear_count + collision_detected_count
            == collision_sample_count
        )
        terminal_adapter_sim_step = int(
            getattr(self.adapter, "sim_steps", -1)
        )
        terminal_adapter_sim_time_s = float(
            getattr(self.adapter, "sim_time", float("nan"))
        )
        terminal_time_identity_valid = bool(
            terminal_adapter_sim_step >= 0
            and math.isfinite(terminal_adapter_sim_time_s)
            and self.physics_dt_s is not None
            and self.filtered_contact_last_sim_time_s is not None
            and math.isclose(
                terminal_adapter_sim_time_s,
                terminal_adapter_sim_step * float(self.physics_dt_s),
                rel_tol=0.0,
                abs_tol=max(1.0e-9, float(self.physics_dt_s) * 1.0e-6),
            )
            and math.isclose(
                terminal_adapter_sim_time_s,
                float(self.filtered_contact_last_sim_time_s),
                rel_tol=0.0,
                abs_tol=max(1.0e-9, float(self.physics_dt_s) * 1.0e-6),
            )
        )
        collision_coverage_complete = bool(
            collision_sample_count > 0
            and collision_counts_consistent
            and collision_detected_count == 0
            and collision_clear_count == collision_sample_count
            and collision_tracker.get("last_collision") is False
            and collision_sample_count == self.filtered_contact_sample_epoch
            and type(collision_tracker.get("first_sample_sim_step")) is int
            and type(collision_tracker.get("last_sample_sim_step")) is int
            and int(collision_tracker["last_sample_sim_step"])
            - int(collision_tracker["first_sample_sim_step"])
            + 1
            == collision_sample_count
            and collision_tracker.get("last_sample_epoch")
            == self.filtered_contact_sample_epoch
            and collision_tracker.get("last_sample_sim_step")
            == self.filtered_contact_last_sim_step
            and collision_tracker.get("last_sample_sim_step")
            == terminal_adapter_sim_step
            and terminal_time_identity_valid
        )
        effective_collision_detected = bool(
            self.dangerous_collision_detected or collision_detected_count > 0
        )
        unverified_provider_collision = bool(
            self.unverified_provider_collision_claim
        )
        unverified_provider_penetration = bool(
            self.unverified_provider_penetration_claim
        )
        effective_penetration_detected = bool(
            self.severe_penetration_detected
            or unverified_provider_penetration
        )
        collision_clear_proven = bool(
            collision_coverage_complete and not unverified_provider_collision
        )
        dangerous_collision_available = bool(
            effective_collision_detected or collision_clear_proven
        )
        dangerous_collision_value = (
            True
            if effective_collision_detected
            else False
            if collision_clear_proven
            else None
        )
        contact_detection_evidence = (
            {}
            if collision_detected_count <= 0
            else {
                "sample_sim_step": collision_tracker.get(
                    "first_detected_sim_step"
                ),
                "source": (
                    str(collision_tracker.get("source", "") or "")
                    + ":"
                    + str(
                        collision_tracker.get(
                            "first_detected_sample_sha256", ""
                        )
                        or ""
                    )
                ),
                "combined_contact_sample_sha256": collision_tracker.get(
                    "first_detected_sample_sha256", ""
                ),
                "runtime_safety_evidence_sha256": "",
            }
        )
        effective_detection_evidence = dict(
            self.dangerous_collision_detection_evidence
            or contact_detection_evidence
        )
        runtime_collision_evidence = {
            **collision_tracker,
            "counts_consistent": collision_counts_consistent,
            "coverage_complete": collision_coverage_complete,
            "published_contact_sample_epoch": self.filtered_contact_sample_epoch,
            "last_published_contact_sim_step": self.filtered_contact_last_sim_step,
            "last_published_contact_sim_time_s": (
                self.filtered_contact_last_sim_time_s
            ),
            "terminal_adapter_sim_step": terminal_adapter_sim_step,
            "terminal_adapter_sim_time_s": terminal_adapter_sim_time_s,
            "terminal_time_identity_valid": terminal_time_identity_valid,
        }
        physical_evidence = {
            "source_version": self.request.source_version,
            "body_crossed_front_face": body_crossed,
            "required_leg_lift_completed": lifts,
            "final_recoverable": recoverable,
            "robot_fell": True if stability == "fallen" else False if stability in {"stable", "recoverable"} else None,
            **body_stuck_evidence,
            "wheel_drive_up_without_required_lift": any_illegal,
            "dangerous_collision": dangerous_collision_value,
            "dangerous_collision_available": dangerous_collision_available,
            "dangerous_collision_validation_source": (
                str(
                    self.unverified_provider_collision_claim.get(
                        "reported_source", ""
                    )
                    or self.unverified_provider_collision_claim.get(
                        "classification", ""
                    )
                )
                if unverified_provider_collision
                else effective_detection_evidence.get("source", "")
                if effective_collision_detected
                else collision_tracker.get("source", "")
                if collision_clear_proven
                else UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE
            ),
            "runtime_nonwheel_collision_evidence": runtime_collision_evidence,
            "feedback_recovery_action_ledger": {
                "schema_version": FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA,
                "path": str(feedback_recovery_path),
                "sha256": feedback_recovery_sha256,
                "action_count": len(self.feedback_recovery_action_rows),
                "n_plus_one_verified_action_count": (
                    self.feedback_recovery_verified_action_count
                ),
                "physical_response_verified_action_count": sum(
                    row.get("physical_response_verified") is True
                    for row in self.feedback_recovery_action_rows
                ),
            },
            "dangerous_collision_detection_evidence": effective_detection_evidence,
            "unverified_provider_collision_claim": dict(
                self.unverified_provider_collision_claim
            ),
            "unverified_provider_penetration_claim": dict(
                self.unverified_provider_penetration_claim
            ),
            "severe_penetration_detection_evidence": dict(
                self.severe_penetration_detection_evidence
            ),
            "joint_limit_violation": self.joint_limit_violation_detected,
            "joint_limit_evidence_available": True,
            "nonfinite_core_state_detected": self.nonfinite_state_detected,
            "unsafe_joint_target": self.target_audit.get("unsafe"),
            "unsafe_joint_target_evidence_available": self.target_audit.get("available") is True,
            "unsafe_joint_target_validation_source": self.target_audit.get("source", ""),
            **active_leg_trapped_evidence,
            "severe_penetration": (
                True if effective_penetration_detected else None
            ),
            "penetration_evidence_available": effective_penetration_detected,
            "penetration_validation_source": (
                self.unverified_provider_penetration_claim.get(
                    "reported_source", ""
                )
                if unverified_provider_penetration
                else self.severe_penetration_detection_evidence.get("source", "")
                if self.severe_penetration_detected
                else UNAVAILABLE_RUNTIME_EVIDENCE_SOURCE
            ),
            "runtime_collision_penetration_classification": (
                "DETECTED_HARD_FAILURE"
                if effective_collision_detected
                or effective_penetration_detected
                else "UNVERIFIED_PROVIDER_TRUE_CLAIM_HARD_STOP"
                if unverified_provider_collision
                else (
                    "DANGEROUS_COLLISION_CLEAR_BY_FULL_CURRENT_TICK_FILTERED_"
                    "CONTACT_COVERAGE;PENETRATION_NOT_EVALUATED_REQUIRES_"
                    "SHA_BOUND_FULL_VIDEO_REVIEW"
                )
                if collision_coverage_complete
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
            "final_all_top": final_all_top,
            "final_all_loaded": final_all_loaded,
            "final_load_available": final_load_available,
            "final_velocity_stable": final.get("final_velocity_stable"),
            "final_posture_complete": bool(final.get("posture_complete") is True),
            "maximum_abs_roll_rad": self.peak_roll_rad,
            "maximum_abs_pitch_rad": self.peak_pitch_rad,
            "traversal": {
                "legs": self.traversal,
                "phase_local_legs": self.phase_traversal,
                "any_illegal_drive_up": any_illegal,
                "lift_authority": "ACTIVE_LEG_PHASE_LOCAL",
            },
            "observation_scope": {
                "contact_load_available": final_load_available,
                "wheel_classification": (
                    "FILTERED_CONTACT_PLUS_GEOMETRY"
                    if final_load_available
                    else "FILTERED_CONTACT_UNAVAILABLE"
                ),
                "filtered_contact_bank_enabled": self.filtered_contact_bank_enabled,
                "strict_rest_blocking": True,
                "contact_drift_blocking": True,
                "final_all_top_blocking": True,
                "com_measurement_available": bool(
                    final_centroidal is not None
                    and final_centroidal.whole_body_com.available
                ),
                "com_proxy_source": "",
                "centroidal_support_evidence_sha256": (
                    ""
                    if final_centroidal is None
                    else final_centroidal.payload_sha256
                ),
                "combined_contact_sample_sha256": str(
                    dict(final.get("filtered_contact_sample", {}) or {}).get(
                        "sample_sha256", ""
                    )
                ),
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
                final_payload = self._observation_payload(self.adapter)
                if (
                    self.last_verified_servo_targets is not None
                    and self.last_verified_wheel_targets is not None
                ):
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
        feedback_recovery_path = (
            self.request.run_dir / "macro_feedback_recovery_action_ledger.jsonl"
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
        _write_jsonl(
            feedback_recovery_path,
            self.feedback_recovery_action_rows,
        )
        source_consumption_sha256 = _sha256_file(source_consumption_path)
        segment_completion_sha256 = _sha256_file(segment_completion_path)
        feedback_recovery_sha256 = _sha256_file(feedback_recovery_path)
        dispatch_sha256 = _sha256_file(dispatch_path)
        transition_sha256 = _sha256_file(transitions_path)
        telemetry_sha256 = _sha256_file(telemetry_jsonl)
        final_raw_telemetry_row_sha256 = (
            _canonical_json_sha256(self.rows[-1]) if self.rows else ""
        )
        self.terminal_recovery_closure = (
            self._terminal_recovery_closure_mapping(
                telemetry_jsonl_path=telemetry_jsonl,
                telemetry_jsonl_sha256=telemetry_sha256,
                final_raw_telemetry_row_sha256=(
                    final_raw_telemetry_row_sha256
                ),
            )
        )
        timeline = list(_attribute(self.controller, "timeline", []) or [])
        _atomic_write_json(timeline_path, {"timeline": timeline})
        inputs = self._task_inputs(success=success, error=self.error)
        inputs_path = self.request.run_dir / "macro_task_inputs.json"
        _atomic_write_json(inputs_path, inputs)
        root_write_count = getattr(self.adapter, "root_state_write_count", None)
        durable_root_write_count = (
            root_write_count if type(root_write_count) is int else -1
        )
        durable_servo_command_transform = dict(
            _jsonable(self.durable_servo_command_transform)
        )
        durable_servo_command_transform_sha256 = (
            _canonical_json_sha256(durable_servo_command_transform)
            if durable_servo_command_transform
            else ""
        )
        durable_boundary_readback = dict(_jsonable(self.boundary_readback))
        durable_boundary_readback_sha256 = (
            _canonical_json_sha256(durable_boundary_readback)
            if durable_boundary_readback
            else ""
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
            "servo_command_transform": durable_servo_command_transform,
            "servo_command_transform_sha256": (
                durable_servo_command_transform_sha256
            ),
            "artifact_request_id": "",
            "root_state_write_count": durable_root_write_count,
            "start_boundary_ack": dict(_jsonable(self.boundary_ack)),
            "start_boundary_readback": durable_boundary_readback,
            "start_boundary_readback_sha256": (
                durable_boundary_readback_sha256
            ),
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
            "source_action_coverage_errors": self._source_action_coverage_errors(),
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
            "feedback_recovery_action_count": len(
                self.feedback_recovery_action_rows
            ),
            "feedback_recovery_action_ledger_schema_version": (
                FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA
            ),
            "feedback_recovery_verified_action_count": (
                self.feedback_recovery_verified_action_count
            ),
            "feedback_recovery_physical_response_verified_action_count": sum(
                row.get("physical_response_verified") is True
                for row in self.feedback_recovery_action_rows
            ),
            "feedback_recovery_action_coverage_complete": not (
                self._feedback_recovery_coverage_errors()
            ),
            "feedback_recovery_action_coverage_errors": (
                self._feedback_recovery_coverage_errors()
            ),
            "feedback_recovery_action_ledger_path": str(
                feedback_recovery_path
            ),
            "feedback_recovery_action_ledger_sha256": (
                feedback_recovery_sha256
            ),
            "feedback_recovery_dispatch_ledger_path": str(dispatch_path),
            "feedback_recovery_dispatch_ledger_sha256": dispatch_sha256,
            "feedback_recovery_dispatch_count": len(self.dispatch_rows),
            "transition_count": len(self.transition_rows),
            "telemetry_sample_count": len(self.rows),
            "telemetry_jsonl_path": str(telemetry_jsonl),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "final_raw_telemetry_row_sha256": final_raw_telemetry_row_sha256,
            "terminal_recovery_closure": dict(
                _jsonable(self.terminal_recovery_closure)
            ),
            "telemetry_csv_path": str(telemetry_csv),
            "transition_evidence_path": str(transitions_path),
            "transition_evidence_sha256": transition_sha256,
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
            "filtered_contact_bank_enabled": self.filtered_contact_bank_enabled,
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
            "servo_command_transform": durable_servo_command_transform,
            "servo_command_transform_sha256": (
                durable_servo_command_transform_sha256
            ),
            "artifact_request_id": "",
            "root_state_write_count": durable_root_write_count,
            "start_boundary_ack": dict(_jsonable(self.boundary_ack)),
            "start_boundary_readback": durable_boundary_readback,
            "start_boundary_readback_sha256": (
                durable_boundary_readback_sha256
            ),
            "source_action_consumption_count": len(
                self.source_action_consumption_rows
            ),
            "source_action_coverage_complete": not self._source_action_coverage_errors(),
            "source_action_coverage_errors": self._source_action_coverage_errors(),
            "source_action_consumption_path": str(source_consumption_path),
            "source_action_consumption_sha256": source_consumption_sha256,
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
            "feedback_recovery_action_count": len(
                self.feedback_recovery_action_rows
            ),
            "feedback_recovery_action_ledger_schema_version": (
                FEEDBACK_RECOVERY_ACTION_LEDGER_SCHEMA
            ),
            "feedback_recovery_verified_action_count": (
                self.feedback_recovery_verified_action_count
            ),
            "feedback_recovery_physical_response_verified_action_count": sum(
                row.get("physical_response_verified") is True
                for row in self.feedback_recovery_action_rows
            ),
            "feedback_recovery_action_coverage_complete": not (
                self._feedback_recovery_coverage_errors()
            ),
            "feedback_recovery_action_coverage_errors": (
                self._feedback_recovery_coverage_errors()
            ),
            "feedback_recovery_action_ledger_path": str(
                feedback_recovery_path
            ),
            "feedback_recovery_action_ledger_sha256": (
                feedback_recovery_sha256
            ),
            "feedback_recovery_dispatch_ledger_path": str(dispatch_path),
            "feedback_recovery_dispatch_ledger_sha256": dispatch_sha256,
            "feedback_recovery_dispatch_count": len(self.dispatch_rows),
            "transition_count": len(self.transition_rows),
            "transition_evidence_path": str(transitions_path),
            "transition_evidence_sha256": transition_sha256,
            "telemetry_sample_count": len(self.rows),
            "telemetry_jsonl_path": str(telemetry_jsonl),
            "telemetry_jsonl_sha256": telemetry_sha256,
            "final_raw_telemetry_row_sha256": final_raw_telemetry_row_sha256,
            "terminal_recovery_closure": dict(
                _jsonable(self.terminal_recovery_closure)
            ),
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
            "filtered_contact_bank_enabled": self.filtered_contact_bank_enabled,
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
    "configure_scene_for_macro_fsm",
    "load_worker_macro_fsm_request",
    "validate_worker_macro_start_binding",
]
