"""Opt-in Gate-1 artifact ownership inside the production Isaac worker.

The ordinary UI worker never imports or enables the recording machinery unless
``--fsm50-gate-request-path`` names a validated request.  The session observes
the same ``SimTimePlaybackService.update -> SimRobotAdapter.step`` loop as UI
Play All Fast; it does not compile or execute a second playback path.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping


REQUEST_SCHEMA = "fsm50.worker_recording_gate_request.v1"
SESSION_SCHEMA = "fsm50.worker_recording_session.v1"
POST_SETTLE_S = 0.5
ENVIRONMENT_EQUIVALENCE_ROLES = frozenset({"A1", "A2", "B"})
ORDINARY_UI_DIAGNOSTIC_ROLES = frozenset({"U"})
ORDINARY_UI_QUALIFICATION_SCOPE = "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
REQUEST_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "enabled",
        "artifact_owner",
        "request_id",
        "plan_id",
        "plan_sha256",
        "expected_plan_sha256",
        "plan_event_count",
        "plan_segment_count",
        "source_version",
        "trial_id",
        "contact_mode",
        "environment_equivalence_role",
        "diagnostic_role",
        "qualification_scope",
        "gate1_eligible",
        "gate1_physical_qualification_eligible",
        "environment_equivalence_eligible",
        "height_mm",
        "artifact_root",
        "accepted_steps_path",
        "metadata_path",
        "accepted_steps_sha256",
        "expected_accepted_steps_sha256",
        "robot_usd_path",
        "expected_robot_usd_sha256",
        "environment_lock_path",
        "environment_lock_sha256",
        "telemetry_rate_hz",
        "post_run_settle_s",
        "timeout_s",
        "timeout_scale",
        "capture_video",
        "headless",
        "video_fps",
        "expected_root_state_write_count",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    from telemetry.exporters import strict_json_dumps

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                strict_json_dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class WorkerRecordingGateRequest:
    request_path: Path
    artifact_request_sha256: str
    request_id: str
    plan_id: str
    artifact_root: Path
    accepted_steps_path: Path
    metadata_path: Path
    accepted_steps_sha256: str
    source_version: str
    trial_id: int
    height_mm: int = 50
    telemetry_rate_hz: float = 120.0
    video_fps: float = 15.0
    capture_video: bool = True
    headless: bool = False
    contact_mode: str = "instrumented"
    environment_equivalence_role: str = ""
    diagnostic_role: str = ""
    qualification_scope: str = "GATE1_PHYSICAL_QUALIFICATION"
    gate1_eligible: bool = True
    gate1_physical_qualification_eligible: bool = True
    environment_equivalence_eligible: bool = False
    post_run_settle_s: float = POST_SETTLE_S
    expected_plan_sha256: str = ""
    expected_plan_event_count: int = 0
    expected_plan_segment_count: int = 0
    robot_usd_path: Path | None = None
    expected_robot_usd_sha256: str = ""
    environment_lock_path: Path | None = None
    environment_lock_sha256: str = ""
    expected_root_state_write_count: int = 0

    def preflight_payload(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA,
            "enabled": True,
            "request_id": self.request_id,
            "request_path": str(self.request_path),
            "artifact_request_sha256": self.artifact_request_sha256,
            "artifact_root": str(self.artifact_root),
            "source_version": self.source_version,
            "trial_id": self.trial_id,
            "height_mm": self.height_mm,
            "accepted_steps_path": str(self.accepted_steps_path),
            "accepted_steps_sha256": self.accepted_steps_sha256,
            "metadata_path": str(self.metadata_path),
            "contact_mode": self.contact_mode,
            "environment_equivalence_role": self.environment_equivalence_role,
            "diagnostic_role": self.diagnostic_role,
            "qualification_scope": self.qualification_scope,
            "gate1_eligible": self.gate1_eligible,
            "gate1_physical_qualification_eligible": (
                self.gate1_physical_qualification_eligible
            ),
            "environment_equivalence_eligible": (
                self.environment_equivalence_eligible
            ),
            "telemetry_rate_hz": self.telemetry_rate_hz,
            "video_fps": self.video_fps,
            "capture_video": self.capture_video,
            "post_run_settle_s": self.post_run_settle_s,
            "expected_plan_sha256": self.expected_plan_sha256,
            "expected_plan_event_count": self.expected_plan_event_count,
            "expected_plan_segment_count": self.expected_plan_segment_count,
            "plan_id": self.plan_id,
            "expected_root_state_write_count": self.expected_root_state_write_count,
            "preflight_ok": True,
        }


def load_worker_recording_gate_request(
    request_path: str | Path | None,
) -> WorkerRecordingGateRequest | None:
    """Load the explicit opt-in request; an empty path keeps UI behavior off."""

    text = str(request_path or "").strip()
    if not text:
        return None
    path = Path(text).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    raw = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(raw, dict):
        raise ValueError("FSM50 worker gate request must be a JSON object")
    missing_keys = sorted(REQUEST_REQUIRED_KEYS - set(raw))
    unexpected_keys = sorted(set(raw) - REQUEST_REQUIRED_KEYS)
    if missing_keys:
        raise ValueError(
            "FSM50 worker gate request is missing required keys: "
            + ", ".join(missing_keys)
        )
    if unexpected_keys:
        raise ValueError(
            "FSM50 worker gate request has unexpected v1 keys: "
            + ", ".join(unexpected_keys)
        )

    def required_text(key: str, *, allow_empty: bool = False) -> str:
        value = raw[key]
        if type(value) is not str:
            raise ValueError(f"FSM50 worker gate request {key} must be a string")
        if not allow_empty and not value.strip():
            raise ValueError(f"FSM50 worker gate request is missing {key}")
        return value

    def required_bool(key: str) -> bool:
        value = raw[key]
        if type(value) is not bool:
            raise ValueError(f"FSM50 worker gate request {key} must be a JSON boolean")
        return value

    def required_int(key: str) -> int:
        value = raw[key]
        if type(value) is not int:
            raise ValueError(f"FSM50 worker gate request {key} must be a JSON integer")
        return value

    def required_float(key: str) -> float:
        value = raw[key]
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"FSM50 worker gate request {key} must be a finite JSON float")
        return value

    if required_text("schema_version") != REQUEST_SCHEMA:
        raise ValueError(
            f"unsupported FSM50 worker gate request schema: {raw['schema_version']!r}"
        )
    if required_bool("enabled") is not True:
        raise ValueError("FSM50 worker gate request must contain enabled=true")

    request_id = required_text("request_id").strip()
    plan_id = required_text("plan_id").strip()
    artifact_root = Path(required_text("artifact_root")).resolve()
    steps_path = Path(required_text("accepted_steps_path")).resolve()
    metadata_path = Path(required_text("metadata_path")).resolve()
    expected_sha = required_text("accepted_steps_sha256").strip().lower()
    source_version = required_text("source_version").strip()
    expected_plan_sha = required_text("expected_plan_sha256").strip().lower()
    declared_plan_sha = required_text("plan_sha256").strip().lower()
    if declared_plan_sha != expected_plan_sha:
        raise ValueError("plan_sha256 and expected_plan_sha256 disagree")
    if required_text("artifact_owner") != "sim_worker_process":
        raise ValueError("FSM50 worker gate request must bind artifact_owner=sim_worker_process")
    def require_sha256(value: str, label: str) -> str:
        lowered = str(value).lower()
        if len(lowered) != 64 or any(
            c not in "0123456789abcdef" for c in lowered
        ):
            raise ValueError(f"{label} must be a SHA-256 hex digest")
        return lowered

    expected_sha = require_sha256(expected_sha, "accepted_steps_sha256")
    expected_plan_sha = require_sha256(expected_plan_sha, "expected_plan_sha256")
    aliased_expected_sha = required_text(
        "expected_accepted_steps_sha256"
    ).strip().lower()
    if aliased_expected_sha != expected_sha:
        raise ValueError(
            "accepted_steps_sha256 and expected_accepted_steps_sha256 disagree"
        )
    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    actual_sha = _sha256_file(steps_path).lower()
    if actual_sha != expected_sha:
        raise ValueError(
            f"accepted_steps hash mismatch expected={expected_sha} actual={actual_sha}"
        )
    if artifact_root.exists():
        raise FileExistsError(
            "worker artifact_root must not exist; the worker exclusively creates its per-run subtree: "
            + str(artifact_root)
        )
    if artifact_root == Path(artifact_root.anchor):
        raise ValueError("artifact_root cannot be a filesystem root")
    try:
        artifact_root.relative_to(path.parent)
    except ValueError as exc:
        raise ValueError(
            "artifact_root must stay below the request/batch directory"
        ) from exc
    if artifact_root == path.parent:
        raise ValueError("artifact_root cannot alias the request/batch directory")
    height_mm = required_int("height_mm")
    if height_mm != 50:
        raise ValueError(f"Gate-1 worker recording requires height_mm=50, got {height_mm}")
    contact_mode_raw = required_text("contact_mode")
    contact_mode = contact_mode_raw.strip().lower()
    if contact_mode_raw != contact_mode:
        raise ValueError("contact_mode must use its canonical lowercase spelling")
    equivalence_role_raw = required_text(
        "environment_equivalence_role", allow_empty=True
    )
    equivalence_role = equivalence_role_raw.strip().upper()
    if equivalence_role_raw != equivalence_role:
        raise ValueError(
            "environment_equivalence_role must use its canonical uppercase spelling"
        )
    diagnostic_role_raw = required_text("diagnostic_role", allow_empty=True)
    diagnostic_role = diagnostic_role_raw.strip().upper()
    if diagnostic_role_raw != diagnostic_role:
        raise ValueError("diagnostic_role must use its canonical uppercase spelling")
    if equivalence_role and equivalence_role not in ENVIRONMENT_EQUIVALENCE_ROLES:
        raise ValueError(
            "environment-equivalence role must be one of A1, A2, or B"
        )
    if diagnostic_role and diagnostic_role not in ORDINARY_UI_DIAGNOSTIC_ROLES:
        raise ValueError("diagnostic_role must be U when present")
    if diagnostic_role and equivalence_role:
        raise ValueError(
            "diagnostic_role U and environment-equivalence roles are mutually exclusive"
        )
    expected_contact_mode = (
        "disabled"
        if diagnostic_role == "U"
        else "formal"
        if equivalence_role in {"A1", "A2"}
        else "instrumented"
    )
    if contact_mode != expected_contact_mode:
        if diagnostic_role == "U":
            raise ValueError(
                "ordinary-UI diagnostic role U requires contact_mode=disabled"
            )
        if equivalence_role:
            raise ValueError(
                f"environment-equivalence role {equivalence_role} requires "
                f"contact_mode={expected_contact_mode}"
            )
        raise ValueError(
            "Gate-1 worker recording requires contact_mode=instrumented; "
            "formal capture requires an explicit environment-equivalence role"
        )
    expected_scope = (
        ORDINARY_UI_QUALIFICATION_SCOPE
        if diagnostic_role == "U"
        else "TRAJECTORY_COMPARISON"
        if equivalence_role
        else "GATE1_PHYSICAL_QUALIFICATION"
    )
    qualification_scope = required_text("qualification_scope")
    expected_gate1_eligible = bool(not diagnostic_role and not equivalence_role)
    expected_environment_eligible = bool(equivalence_role and not diagnostic_role)
    if qualification_scope != expected_scope:
        raise ValueError(
            f"recording request qualification_scope must be {expected_scope}"
        )
    if required_bool(
        "gate1_physical_qualification_eligible"
    ) is not expected_gate1_eligible:
        raise ValueError(
            "recording request gate1_physical_qualification_eligible is inconsistent"
        )
    if required_bool("gate1_eligible") is not expected_gate1_eligible:
        raise ValueError("recording request gate1_eligible is inconsistent")
    if required_bool(
        "environment_equivalence_eligible"
    ) is not expected_environment_eligible:
        raise ValueError(
            "recording request environment_equivalence_eligible is inconsistent"
        )
    post_settle = required_float("post_run_settle_s")
    if abs(post_settle - POST_SETTLE_S) > 1.0e-9:
        raise ValueError("Gate-1 worker recording requires exactly 0.5 s post-settle")
    capture_video = required_bool("capture_video")
    if not capture_video:
        raise ValueError("Gate-1 worker recording requires direct viewport video capture")
    headless = required_bool("headless")
    if headless:
        raise ValueError("Gate-1 worker recording requires the active GUI viewport")
    trial_id = required_int("trial_id")
    if trial_id < 1:
        raise ValueError("trial_id must be a positive integer")
    telemetry_rate_hz = required_float("telemetry_rate_hz")
    if abs(telemetry_rate_hz - 120.0) > 1.0e-9:
        raise ValueError(
            "Gate-1 worker recording requires 120 Hz per-physics-tick telemetry"
        )
    video_fps = required_float("video_fps")
    if abs(video_fps - 15.0) > 1.0e-9:
        raise ValueError("Gate-1 worker recording requires exactly 15 fps video")
    timeout_s = required_float("timeout_s")
    timeout_scale = required_float("timeout_scale")
    if timeout_s <= 0.0 or timeout_scale <= 0.0:
        raise ValueError("timeout_s and timeout_scale must be positive")
    plan_event_count = required_int("plan_event_count")
    plan_segment_count = required_int("plan_segment_count")
    if plan_event_count <= 0 or plan_segment_count <= 0:
        raise ValueError("plan event and segment counts must be positive")
    expected_root_state_write_count = required_int(
        "expected_root_state_write_count"
    )
    if expected_root_state_write_count != 0:
        raise ValueError("expected_root_state_write_count must be exactly zero")

    robot_usd_path = Path(required_text("robot_usd_path")).resolve()
    environment_lock_path = Path(required_text("environment_lock_path")).resolve()
    expected_robot_sha = required_text("expected_robot_usd_sha256").lower()
    environment_lock_sha = required_text("environment_lock_sha256").lower()
    expected_robot_sha = require_sha256(
        expected_robot_sha, "expected_robot_usd_sha256"
    )
    environment_lock_sha = require_sha256(
        environment_lock_sha, "environment_lock_sha256"
    )
    for source, expected, label in (
        (robot_usd_path, expected_robot_sha, "robot USD"),
        (environment_lock_path, environment_lock_sha, "environment lock"),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = _sha256_file(source).lower()
        if actual != expected:
            raise ValueError(f"{label} hash mismatch expected={expected} actual={actual}")
    return WorkerRecordingGateRequest(
        request_path=path,
        artifact_request_sha256=_sha256_file(path).lower(),
        request_id=request_id,
        plan_id=plan_id,
        artifact_root=artifact_root,
        accepted_steps_path=steps_path,
        metadata_path=metadata_path,
        accepted_steps_sha256=expected_sha,
        source_version=source_version,
        trial_id=trial_id,
        height_mm=height_mm,
        telemetry_rate_hz=telemetry_rate_hz,
        video_fps=video_fps,
        capture_video=capture_video,
        headless=headless,
        contact_mode=contact_mode,
        environment_equivalence_role=equivalence_role,
        diagnostic_role=diagnostic_role,
        qualification_scope=qualification_scope,
        gate1_eligible=expected_gate1_eligible,
        gate1_physical_qualification_eligible=expected_gate1_eligible,
        environment_equivalence_eligible=expected_environment_eligible,
        post_run_settle_s=post_settle,
        expected_plan_sha256=expected_plan_sha,
        expected_plan_event_count=plan_event_count,
        expected_plan_segment_count=plan_segment_count,
        robot_usd_path=robot_usd_path,
        expected_robot_usd_sha256=expected_robot_sha,
        environment_lock_path=environment_lock_path,
        environment_lock_sha256=environment_lock_sha,
        expected_root_state_write_count=expected_root_state_write_count,
    )


def configure_scene_for_worker_recording(scene_config: Any, request: WorkerRecordingGateRequest | None) -> Any:
    """Install the request-bound evidence sensor before ``create_scene``.

    A1/A2 retain the production aggregate ``ContactSensor`` constructor.
    Gate-1 and the explicit B role retain the existing combined filtered
    wheel/non-wheel bank.  Only observation plumbing changes here; no scene,
    actuator, solver, or pose parameter is modified.
    """

    if request is None:
        return scene_config
    if request.diagnostic_role == "U":
        if request.environment_equivalence_role or request.contact_mode != "disabled":
            raise ValueError("ordinary-UI U role identity is inconsistent")
        # Preserve the production UI default exactly.  This also keeps USD
        # activation of contact-report APIs disabled, rather than merely
        # discarding an already-created sensor after scene construction.
        if scene_config.telemetry_contact_sensors_enabled is not False:
            raise ValueError(
                "ordinary-UI U role requires the incoming production default "
                "telemetry_contact_sensors_enabled=false"
            )
        if scene_config.contact_sensor_factory is not None:
            raise ValueError(
                "ordinary-UI U role requires the incoming production default "
                "contact_sensor_factory=None"
            )
        return scene_config
    if request.contact_mode == "formal":
        if request.environment_equivalence_role not in {"A1", "A2"}:
            raise ValueError(
                "formal worker recording requires environment-equivalence role A1 or A2"
            )
        scene_config.telemetry_contact_sensors_enabled = True
        scene_config.contact_sensor_factory = None
        return scene_config
    if request.contact_mode != "instrumented":
        raise ValueError(
            f"unsupported worker recording contact_mode={request.contact_mode!r}"
        )
    from .filtered_wheel_contact import make_filtered_wheel_contact_sensor_factory
    from .nonwheel_obstacle_contact import configure_scene_for_wheel_and_nonwheel_contacts

    scene_config.telemetry_contact_sensors_enabled = True
    configure_scene_for_wheel_and_nonwheel_contacts(
        scene_config,
        wheel_factory=make_filtered_wheel_contact_sensor_factory(force_threshold_n=1.0),
        force_threshold_n=1.0,
    )
    return scene_config


def environment_equivalence_diagnostic_status(
    *,
    role: str,
    contact_mode: str,
    artifact_valid: bool,
    scheduler_complete: bool,
    dispatch_complete: bool,
    source_integrity_ok: bool,
) -> dict[str, Any]:
    """Classify capture closure independently from physical qualification."""

    normalized_role = str(role or "").strip().upper()
    normalized_mode = str(contact_mode or "").strip().lower()
    if normalized_role not in {"", "A1", "A2", "B"}:
        raise ValueError("environment-equivalence role must be empty, A1, A2, or B")
    expected_mode = (
        "formal" if normalized_role in {"A1", "A2"} else "instrumented"
    )
    if normalized_mode != expected_mode:
        raise ValueError(
            f"environment-equivalence role {normalized_role or '<none>'} "
            f"requires contact_mode={expected_mode}"
        )
    diagnostic = bool(normalized_role)
    complete = bool(
        diagnostic
        and artifact_valid
        and scheduler_complete
        and dispatch_complete
        and source_integrity_ok
    )
    return {
        "environment_equivalence_role": normalized_role,
        "environment_equivalence_diagnostic": diagnostic,
        "environment_equivalence_diagnostic_complete": complete,
        "qualification_scope": (
            "TRAJECTORY_COMPARISON"
            if diagnostic
            else "GATE1_PHYSICAL_QUALIFICATION"
        ),
    }


def ordinary_ui_diagnostic_status(
    *,
    role: str,
    contact_mode: str,
    artifact_valid: bool,
    scheduler_complete: bool,
    dispatch_complete: bool,
    wheel_target_integral_verdict: str,
    trajectory_valid: bool,
    contact_sensor_disabled: bool,
    source_integrity_ok: bool,
    root_state_write_count: int,
) -> dict[str, Any]:
    """Classify a sensor-free ordinary-UI trajectory artifact fail closed."""

    normalized_role = str(role or "").strip().upper()
    normalized_mode = str(contact_mode or "").strip().lower()
    if normalized_role != "U" or normalized_mode != "disabled":
        raise ValueError("ordinary-UI diagnostic requires role=U/contact_mode=disabled")
    complete = bool(
        artifact_valid
        and scheduler_complete
        and dispatch_complete
        and str(wheel_target_integral_verdict or "").upper() == "PASS"
        and trajectory_valid
        and contact_sensor_disabled
        and source_integrity_ok
        and type(root_state_write_count) is int
        and root_state_write_count == 0
    )
    return {
        "diagnostic_role": "U",
        "ordinary_ui_diagnostic": True,
        "ordinary_ui_diagnostic_complete": complete,
        "environment_equivalence_role": "",
        "environment_equivalence_diagnostic": False,
        "environment_equivalence_diagnostic_complete": False,
        "qualification_scope": ORDINARY_UI_QUALIFICATION_SCOPE,
        "gate1_eligible": False,
        "gate1_physical_qualification_eligible": False,
        "environment_equivalence_eligible": False,
        "physical_qualification_eligible": False,
    }


def apply_ordinary_ui_diagnostic_classification(
    result: dict[str, Any],
    *,
    role: str,
) -> None:
    """Keep an unavailable physical verdict from becoming a failure label."""

    if str(role or "").strip().upper() != "U":
        return
    result["classification_before_diagnostic_scope"] = str(
        result.get("classification", "") or ""
    )
    result["first_failure_phase_before_diagnostic_scope"] = str(
        result.get("first_failure_phase", "") or ""
    )
    result["classification"] = "TRAJECTORY_DIAGNOSTIC_ONLY"
    result["first_failure_phase"] = ""


def validate_worker_plan_binding(
    request: WorkerRecordingGateRequest,
    *,
    request_id: str,
    plan_id: str,
    worker_session_id: str,
    expected_worker_session_id: str,
) -> list[str]:
    """Return fail-closed identity errors for one formal start-plan message."""

    errors: list[str] = []
    if str(request_id) != request.request_id:
        errors.append(
            f"FSM50 artifact request_id mismatch: expected={request.request_id} received={request_id}"
        )
    if str(plan_id) != request.plan_id:
        errors.append(
            f"FSM50 artifact plan_id mismatch: expected={request.plan_id} received={plan_id}"
        )
    if str(worker_session_id) != str(expected_worker_session_id):
        errors.append(
            "FSM50 artifact worker_session_id mismatch: "
            f"expected={expected_worker_session_id} received={worker_session_id or '<missing>'}"
        )
    return errors


class WorkerRecordingSession:
    """Lifecycle hooks called only by the production worker's single tick loop."""

    def __init__(
        self,
        request: WorkerRecordingGateRequest,
        *,
        worker_session_id: str,
        finalize_callback: Callable[["WorkerRecordingSession"], dict[str, Any]] | None = None,
    ) -> None:
        self.request = request
        self.worker_session_id = str(worker_session_id)
        self.finalize_callback = finalize_callback
        self.state = "preflight"
        self.error = ""
        self.plan: Any = None
        self.service: Any = None
        self.adapter: Any = None
        self.scene_handle: Any = None
        self.collector: Any = None
        self.video_capture: Any = None
        self.video: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.plan_rows: list[dict[str, Any]] = []
        self.compiled_plan: Any = None
        self.item: Any = None
        self.expected_plan_identity: dict[str, Any] = {}
        self.admitted_plan_identity: dict[str, Any] = {}
        self.direct_dispatch_attempts: list[dict[str, Any]] = []
        self.run_dir = request.artifact_root
        self.source_freeze_pre: dict[str, Any] = {}
        self.motion_start_readiness: dict[str, Any] = {}
        self.pre_first_dispatch_readiness: dict[str, Any] = {}
        self.artifact_preflight_ready = False
        self.readiness_token = ""
        self.readiness_frames: list[dict[str, Any]] = []
        self.boundary_frame: dict[str, Any] = {}
        self.required_readiness_frames = 10
        self.startup_ground: dict[str, Any] = {}
        self.live_obstacle: dict[str, Any] = {}
        self.live_baseline: dict[str, Any] = {}
        self.environment_equivalence: dict[str, Any] = {}
        self.post_settle_deadline_s: float | None = None
        self.completion_pending = False
        self.terminal_message: dict[str, Any] | None = None
        self.created_utc = datetime.now(timezone.utc).isoformat()

    @property
    def enabled(self) -> bool:
        return True

    @property
    def terminal(self) -> bool:
        return self.state in {"complete", "failed"}

    def motion_start_ready_for_role(
        self,
        rich_readiness: Mapping[str, Any],
        shared_worker_readiness: Mapping[str, Any],
    ) -> bool:
        """Apply the role-specific, sensor-independent U admission rule."""

        rich_ready = dict(rich_readiness or {}).get("ready") is True
        if self.request.diagnostic_role == "U":
            return rich_ready
        return bool(
            rich_ready
            and dict(shared_worker_readiness or {}).get(
                "motion_start_ready"
            )
            is True
        )

    def status_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA,
            "enabled": True,
            "request_id": self.request.request_id,
            "worker_session_id": self.worker_session_id,
            "source_version": self.request.source_version,
            "trial_id": self.request.trial_id,
            "state": self.state,
            "error": self.error,
            "artifact_root": str(self.request.artifact_root),
            "run_dir": str(self.run_dir),
            "post_settle_deadline_s": self.post_settle_deadline_s,
            "terminal": self.terminal,
            "artifact_preflight_ready": bool(self.artifact_preflight_ready),
            "readiness_frame_count": len(self.readiness_frames),
            "readiness_frame_count_required": self.required_readiness_frames,
            "readiness_sample_stride_physics_ticks": int(
                getattr(getattr(self.scene_handle, "config", None), "render_interval", 0)
                or 0
            ),
            "motion_start_ready": bool(
                self.motion_start_readiness.get("ready", False)
            ),
            "adapter_runtime_instance_id": str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            "root_state_write_count": int(
                getattr(self.adapter, "root_state_write_count", 0) or 0
            ),
            "contact_mode": self.request.contact_mode,
            "environment_equivalence_role": (
                self.request.environment_equivalence_role
            ),
            "diagnostic_role": self.request.diagnostic_role,
            "qualification_scope": self.request.qualification_scope,
            "gate1_eligible": self.request.gate1_eligible,
            "gate1_physical_qualification_eligible": (
                self.request.gate1_physical_qualification_eligible
            ),
            "environment_equivalence_eligible": (
                self.request.environment_equivalence_eligible
            ),
            "rejected_direct_dispatch_attempts": list(
                self.direct_dispatch_attempts
            ),
            "contact_sensor_type": (
                ""
                if self.scene_handle is None
                or getattr(self.scene_handle, "contact_sensor", None) is None
                else type(getattr(self.scene_handle, "contact_sensor", None)).__name__
            ),
            "contact_sensor_error": str(
                getattr(self.scene_handle, "contact_sensor_error", "") or ""
            ),
            "environment_equivalence": {
                "ok": bool(self.environment_equivalence.get("ok", False)),
                "schema_version": str(
                    self.environment_equivalence.get("schema_version", "") or ""
                ),
                "failed_checks": [
                    str(row.get("name", ""))
                    for row in list(self.environment_equivalence.get("checks", []) or [])
                    if row.get("ok") is not True
                ],
            },
        }

    def _build_plan_identity(self, plan: Any) -> dict[str, Any]:
        """Build the same immutable identity used by the audited direct path."""

        if self.item is None:
            raise RuntimeError("recording source identity is unavailable")
        from .run_fsm50 import _motion_start_plan_identity

        identity = _motion_start_plan_identity(
            item=self.item,
            plan=plan,
            plan_id=self.request.plan_id,
            request_id=self.request.request_id,
            worker_session_id=self.worker_session_id,
        )
        if (
            str(identity.get("source_sha256", "") or "").lower()
            != self.request.accepted_steps_sha256.lower()
        ):
            raise RuntimeError(
                "worker plan identity source hash differs from the immutable request"
            )
        return identity

    def prepare_after_grounding(
        self,
        *,
        adapter: Any,
        scene_handle: Any,
        startup_ground: dict[str, Any],
        robot_usd: str | Path,
    ) -> None:
        """Compile v003 and attach observers before the 10-frame gate window."""

        if self.state != "preflight":
            raise RuntimeError(f"artifact session cannot prepare from state={self.state}")
        from sequence_model import load_steps_jsonl
        from .recording_audit import VersionFiles
        from .recording_fast_plan import fast_plan_rows
        from .run_fsm50 import (
            FSM50TelemetryCollector,
            _RecordingReplayViewportCapture,
            _source_freeze,
            _surface_obstacle,
            _telemetry_config,
            _wheel_forward_sign_by_leg,
            _runtime_environment_equivalence,
        )
        from motion_speed import load_motion_reference
        from sim_obstacle_scene import measure_obstacle_geometry, measure_scene_baseline

        if Path(robot_usd).resolve() != Path(self.request.robot_usd_path).resolve():
            raise RuntimeError(
                "worker robot USD path differs from the immutable artifact request"
            )
        if _sha256_file(self.request.accepted_steps_path).lower() != self.request.accepted_steps_sha256:
            raise RuntimeError("accepted_steps changed after worker request preflight")
        self.steps = load_steps_jsonl(self.request.accepted_steps_path)
        self.compiled_plan, self.plan_rows = fast_plan_rows(
            source_version=self.request.source_version,
            steps=self.steps,
            max_wheel_speed=float(adapter.max_wheel_speed),
        )
        expected_plan_sha = self.request.expected_plan_sha256 or str(self.compiled_plan.plan_sha256)
        if str(self.compiled_plan.plan_sha256) != str(expected_plan_sha):
            raise RuntimeError(
                "worker preflight compilation differs from requested plan: "
                f"expected={expected_plan_sha} compiled={self.compiled_plan.plan_sha256}"
            )
        if self.request.expected_plan_event_count != len(self.compiled_plan.events):
            raise RuntimeError(
                "worker preflight event count differs from request: "
                f"expected={self.request.expected_plan_event_count} "
                f"compiled={len(self.compiled_plan.events)}"
            )
        if self.request.expected_plan_segment_count != len(self.compiled_plan.segments):
            raise RuntimeError(
                "worker preflight segment count differs from request: "
                f"expected={self.request.expected_plan_segment_count} "
                f"compiled={len(self.compiled_plan.segments)}"
            )
        self.compiled_plan.timing["requires_motion_start_readiness_token"] = True
        self.compiled_plan.timing["requires_verified_motion_batch_ack"] = True

        root = self.request.artifact_root
        root.mkdir(parents=True, exist_ok=False)
        (root / ".partial").write_text("running\n", encoding="utf-8")
        self.item = VersionFiles(
            version_id=self.request.source_version,
            directory=self.request.accepted_steps_path.parent,
            steps_path=self.request.accepted_steps_path,
            metadata_path=self.request.metadata_path,
        )
        self.expected_plan_identity = self._build_plan_identity(self.compiled_plan)
        self.source_freeze_pre = _source_freeze(self.item, robot_usd=Path(robot_usd))
        _write_json(root / "source_freeze_pre.json", self.source_freeze_pre)
        _write_json(root / "source_freeze.json", self.source_freeze_pre)
        contact_sensor = getattr(scene_handle, "contact_sensor", None)
        contact_sensor_error = str(
            getattr(scene_handle, "contact_sensor_error", "") or ""
        )
        if self.request.diagnostic_role == "U":
            if (
                bool(scene_handle.config.telemetry_contact_sensors_enabled)
                or scene_handle.config.contact_sensor_factory is not None
                or contact_sensor is not None
                or contact_sensor_error
            ):
                raise RuntimeError(
                    "ordinary-UI U role did not preserve the sensor-disabled production scene"
                )
        else:
            if contact_sensor is None or contact_sensor_error:
                raise RuntimeError(
                    f"{self.request.contact_mode} contact sensor is unavailable: "
                    + str(contact_sensor_error or "missing")
                )
            filtered_bank = bool(
                getattr(contact_sensor, "is_filtered_wheel_contact_bank", False)
            )
            nonwheel_bank = bool(
                getattr(contact_sensor, "is_nonwheel_obstacle_contact_bank", False)
            )
            if self.request.contact_mode == "formal":
                if filtered_bank or nonwheel_bank:
                    raise RuntimeError(
                        "formal A-role unexpectedly received instrumented contact banks"
                    )
            elif not (filtered_bank and nonwheel_bank):
                raise RuntimeError(
                    "instrumented B/Gate-1 combined contact bank is unavailable"
                )
            contact_sensor.reset()
        baseline = measure_scene_baseline(scene_handle, adapter)
        self.live_baseline = baseline
        self.live_obstacle = measure_obstacle_geometry(scene_handle)
        environment_lock = json.loads(
            Path(self.request.environment_lock_path).read_text(encoding="utf-8")
        )
        self.environment_equivalence = _runtime_environment_equivalence(
            lock=environment_lock,
            scene_config=scene_handle.config,
            live_baseline=baseline,
            live_obstacle=self.live_obstacle,
            motion=load_motion_reference(),
            robot_usd=Path(robot_usd),
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )
        if not self.environment_equivalence.get("ok", False):
            raise RuntimeError("worker Gate-1 runtime environment equivalence failed")
        obstacle = _surface_obstacle(baseline)
        collector_args = SimpleNamespace(headless=self.request.headless, output_dir=str(root))
        kwargs: dict[str, Any] = {
            "args": collector_args,
            "scene_handle": scene_handle,
            "obstacle": obstacle,
            "wheel_radius_m": float(baseline.get("wheel_radius_m") or 0.04998999834060672),
            "source_version": self.request.source_version,
            "contact_mode": self.request.contact_mode,
            "environment_equivalence_role": (
                self.request.environment_equivalence_role
            ),
            "diagnostic_role": self.request.diagnostic_role,
            "plan": self.compiled_plan,
            "force_threshold_n": 2.0,
            "unload_force_n": 1.0,
            "load_confirm_force_n": 2.0,
            "top_load_dwell_s": 0.10,
            "loaded_front_face_rotation_limit_rad": 0.15,
            "wheel_forward_sign": _wheel_forward_sign_by_leg(),
        }
        if "plan_rows" in inspect.signature(FSM50TelemetryCollector).parameters:
            kwargs["plan_rows"] = self.plan_rows
        telemetry_config = _telemetry_config(root, self.request.telemetry_rate_hz)
        if self.request.diagnostic_role == "U":
            telemetry_config.telemetry.enable_contact_sensor = False
        self.collector = FSM50TelemetryCollector(telemetry_config, **kwargs)
        self.collector.start_episode(
            adapter=adapter,
            scene_handle=scene_handle,
            obstacle_height_cm=5,
            obstacle_height_m=0.05,
            sequence_label=f"worker_recording_fast_{self.request.source_version}",
            source="production_worker_ipc_gate",
        )
        self.run_dir = self.collector.run_dir or root
        video_args = SimpleNamespace(
            no_video=not self.request.capture_video,
            capture_video=self.request.capture_video,
            video_fps=self.request.video_fps,
            headless=self.request.headless,
        )
        self.video_capture = _RecordingReplayViewportCapture(
            self.run_dir,
            video_args,
            contact_mode=self.request.contact_mode,
            adapter=adapter,
        )
        self.video_capture.start()
        adapter.attach_telemetry(self.collector)
        self.adapter = adapter
        self.scene_handle = scene_handle
        self.startup_ground = dict(startup_ground or {})
        self.state = "readiness_window"

    def attach_verified_plan(
        self,
        *,
        plan: Any,
        service: Any,
        adapter: Any,
        scene_handle: Any,
        motion_start_readiness: dict[str, Any],
        robot_usd: str | Path,
    ) -> None:
        """Verify the formal IPC payload; never compile or run another path."""

        if self.state != "ready_for_plan":
            raise RuntimeError(f"artifact session is not preflight-ready: state={self.state}")
        expected = self.compiled_plan
        mismatches = []
        if str(plan.plan_sha256) != str(expected.plan_sha256):
            mismatches.append(f"sha expected={expected.plan_sha256} IPC={plan.plan_sha256}")
        if len(plan.events) != len(expected.events):
            mismatches.append(f"events expected={len(expected.events)} IPC={len(plan.events)}")
        if len(plan.segments) != len(expected.segments):
            mismatches.append(f"segments expected={len(expected.segments)} IPC={len(plan.segments)}")
        if abs(float(plan.final_time_s) - float(expected.final_time_s)) > 1.0e-9:
            mismatches.append(f"final_time expected={expected.final_time_s} IPC={plan.final_time_s}")
        if str(service.plan_id) != self.request.plan_id:
            mismatches.append(f"plan_id expected={self.request.plan_id} IPC={service.plan_id}")
        if str(service.request_id) != self.request.request_id:
            mismatches.append(
                f"request_id expected={self.request.request_id} IPC={service.request_id}"
            )
        if mismatches:
            raise RuntimeError("worker artifact plan is not the accepted v003 fast plan: " + "; ".join(mismatches))
        from .run_fsm50 import _strict_json_equal

        admitted_identity = self._build_plan_identity(plan)
        if not _strict_json_equal(admitted_identity, self.expected_plan_identity):
            raise RuntimeError(
                "worker admitted plan identity differs from independently compiled identity"
            )
        self.admitted_plan_identity = admitted_identity
        self.plan = plan
        self.service = service
        shared = dict(motion_start_readiness or {})
        if not self.motion_start_ready_for_role(
            self.motion_start_readiness,
            shared,
        ):
            raise RuntimeError("formal worker shared MOTION_START_READY is false at plan admission")
        self.motion_start_readiness["shared_production_worker_gate"] = shared
        _write_json(self.run_dir / "motion_start_readiness.json", self.motion_start_readiness)
        self.state = "attached"

    def record_start_boundary(self) -> None:
        if self.state != "attached":
            return
        from .motion_start_readiness import capture_live_motion_start_snapshot

        self.boundary_frame = capture_live_motion_start_snapshot(
            self.adapter, self.scene_handle, self.live_obstacle
        )

    def record_pre_first_dispatch(self, readiness: dict[str, Any]) -> bool:
        if self.state not in {"attached", "running"}:
            return False
        from .motion_start_readiness import capture_live_motion_start_snapshot
        from .run_fsm50 import (
            _build_motion_start_readiness_evidence,
            _evaluate_motion_start_window,
        )

        shared = dict(readiness or {})
        post_boundary_frame = capture_live_motion_start_snapshot(
            self.adapter, self.scene_handle, self.live_obstacle
        )
        identity = dict(self.admitted_plan_identity or {})
        if not identity:
            raise RuntimeError("worker admitted plan identity is unavailable")
        motion_batches = [
            dict(row or {})
            for row in list(self.service.timing_trace.get("motion_batches", []) or [])
        ]
        source_batches = [
            row
            for row in motion_batches
            if str(row.get("dispatch_kind", "") or "") == "source_segment_start"
        ]
        command_dispatch_evidence = {
            "direct_dispatch_attempt_count": len(self.direct_dispatch_attempts),
            "direct_dispatch_attempts": list(self.direct_dispatch_attempts),
            "motion_batch_count": len(motion_batches),
            "motion_batch_dispatch_kinds": [
                str(row.get("dispatch_kind", "") or "") for row in motion_batches
            ],
            "source_segment_start_count": len(source_batches),
            "allowed_pre_first_dispatch_kinds": ["playback_start_boundary"],
        }
        command_dispatch_idle = bool(
            not self.direct_dispatch_attempts
            and not source_batches
            and motion_batches
            and all(
                str(row.get("dispatch_kind", "") or "")
                == "playback_start_boundary"
                for row in motion_batches
            )
        )
        window = [
            *self.readiness_frames[-8:],
            self.boundary_frame,
            post_boundary_frame,
        ]
        rich = _evaluate_motion_start_window(
            adapter=self.adapter,
            startup_ground=self.startup_ground,
            frames=window,
            plan_identity=identity,
            required_frames=self.required_readiness_frames,
            command_dispatch_idle=command_dispatch_idle,
            command_dispatch_evidence=command_dispatch_evidence,
            diagnostic_role=self.request.diagnostic_role,
        )
        pre_first_step = int(getattr(self.adapter, "sim_steps", 0) or 0)
        candidate = {
            **rich,
            "ready": self.motion_start_ready_for_role(rich, shared),
            "shared_production_worker_gate": shared,
            "start_boundary_frame": self.boundary_frame,
            "post_boundary_frame": post_boundary_frame,
            "start_boundary_ack": dict(
                self.service.last_motion_batch_ack or {}
            ),
            "pre_first_dispatch_sim_step": pre_first_step,
            "source_command_dispatch_count": len(source_batches),
            "boundary_batch_is_not_source_command": not source_batches,
        }
        candidate["status"] = "PASS" if candidate["ready"] else "FAIL"
        if candidate["ready"]:
            evidence, token = _build_motion_start_readiness_evidence(
                readiness=candidate,
                source_version=self.request.source_version,
                trial_id=int(self.request.trial_id),
                plan_identity=identity,
                adapter_runtime_instance_id=str(
                    getattr(self.adapter, "runtime_instance_id", "") or ""
                ),
                root_state_write_count=int(
                    getattr(self.adapter, "root_state_write_count", 0) or 0
                ),
                pre_first_dispatch_sim_step=pre_first_step,
            )
            token_bound = bool(
                self.service.bind_motion_start_readiness(
                    token,
                    current_sim_step=pre_first_step,
                )
            )
            evidence["readiness_token_bound"] = token_bound
            if not token_bound:
                evidence["ready"] = False
                evidence["status"] = "FAIL"
                evidence.setdefault("window_failed_checks", []).append(
                    str(
                        getattr(self.service, "last_error", "")
                        or "pre-first MOTION_START_READY token binding failed"
                    )
                )
            self.pre_first_dispatch_readiness = evidence
            self.readiness_token = token if token_bound else ""
        else:
            candidate["readiness_token_sha256"] = ""
            candidate["readiness_token_bound"] = False
            self.pre_first_dispatch_readiness = candidate
            self.readiness_token = ""
        _write_json(
            self.run_dir / "motion_start_pre_first_dispatch.json",
            self.pre_first_dispatch_readiness,
        )
        return bool(
            self.pre_first_dispatch_readiness.get("ready", False)
            and self.pre_first_dispatch_readiness.get(
                "readiness_token_bound", False
            )
            and self.readiness_token
        )

    def reject_direct_dispatch_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Record and reject motion/environment mutations outside the bound plan."""

        if self.terminal:
            return []
        forbidden = {
            "command",
            "apply_motion_batch",
            "pause_playback",
            "resume_playback",
            "stop_playback",
            "set_height",
            "set_height_respawn",
            "respawn",
            "recalibrate_ground_reference",
            "restore_sim_state",
        }
        rejected = [
            message
            for message in messages
            if str(message.get("type", "") or "") in forbidden
        ]
        for message in rejected:
            self.direct_dispatch_attempts.append(
                {
                    "type": str(message.get("type", "") or ""),
                    "rejected": True,
                    "request_id": str(message.get("request_id", "") or ""),
                    "command_id": str(message.get("command_id", "") or ""),
                    "batch_id": str(message.get("batch_id", "") or ""),
                    "sim_step": int(getattr(self.adapter, "sim_steps", 0) or 0),
                }
            )
        return rejected

    def before_adapter_step(self) -> None:
        if self.state not in {
            "readiness_window",
            "ready_for_plan",
            "attached",
            "running",
            "postsettle",
        }:
            return
        if self.state in {"readiness_window", "ready_for_plan"}:
            from .motion_start_readiness import capture_live_motion_start_snapshot

            self.readiness_frames.append(
                capture_live_motion_start_snapshot(
                    self.adapter, self.scene_handle, self.live_obstacle
                )
            )
            if len(self.readiness_frames) > self.required_readiness_frames:
                del self.readiness_frames[: -self.required_readiness_frames]
            self.collector.set_runtime_context(
                fsm_state="MOTION_START_READY",
                macro_state_cursor="MOTION_START_READY",
                command_cursor=0,
                segment_cursor=0,
                source_command="",
                source_event_index=None,
                planned_dispatch_time_s=None,
                actual_dispatch_time_s=None,
                atomic_batch_id="",
                dispatch_kind="",
                segment_index=None,
                source_step=None,
                scheduler_phase="pre_dispatch_evidence",
            )
            return
        if self.state == "postsettle":
            self.collector.set_runtime_context(
                fsm_state="RECORDING_POST_SETTLE",
                macro_state_cursor="RECORDING_POST_SETTLE",
                scheduler_phase="post_settle",
            )
            return
        if not self.service.active and self.state == "attached":
            self.collector.set_runtime_context(
                fsm_state="MOTION_START_READY",
                macro_state_cursor="MOTION_START_READY",
                scheduler_phase="verified_start_boundary",
            )
            return
        if (
            self.state == "attached"
            and self.service.active
            and not self.service.started
        ):
            self.collector.set_runtime_context(
                fsm_state="MOTION_START_READY",
                macro_state_cursor="MOTION_START_READY",
                command_cursor=0,
                segment_cursor=0,
                atomic_batch_id=str(
                    self.service.last_motion_batch_ack.get("batch_id", "") or ""
                ),
                dispatch_kind="playback_start_boundary",
                segment_index=None,
                source_step=None,
                scheduler_phase="verified_start_boundary",
            )
            return
        if not self.service.active and self.state == "running":
            # The scheduler may complete in update() on the same tick as its
            # final source batch.  Let that batch own the next physics frame;
            # enter post-settle only after adapter.step returns.
            self.completion_pending = True
        if not self.service.active and self.state not in {"running"}:
            self.state = "postsettle"
            self.post_settle_deadline_s = float(self.adapter.sim_time) + self.request.post_run_settle_s
            self.before_adapter_step()
            return
        from .run_fsm50 import _batch_owned_command_context, _batch_owned_segment_cursor

        timing_commands = list(self.service.timing_trace.get("commands", []) or [])
        batches = [
            dict(row or {})
            for row in list(self.service.timing_trace.get("motion_batches", []) or [])
            if row.get("scheduler_sim_step") is not None
            and int(row.get("scheduler_sim_step")) == int(self.adapter.sim_steps)
        ]
        batch = batches[-1] if batches else {}
        source_event_by_plan_index = {
            int(provenance["plan_event_index"]): provenance.get("source_event_index")
            for row in self.plan_rows
            for provenance in list(row.get("command_provenance", []) or [])
            if provenance.get("plan_event_index") is not None
        }
        segment_index, source_step = _batch_owned_segment_cursor(
            scheduler_segment_index=int(self.service.segment_index),
            current_motion_batch=batch,
            plan=self.plan,
        )
        command = _batch_owned_command_context(
            current_motion_batch=batch,
            timing_commands=timing_commands,
            source_event_by_plan_index=source_event_by_plan_index,
        )
        self.collector.set_runtime_context(
            fsm_state="RECORDING_FAST_REPLAY",
            macro_state_cursor="RECORDING_FAST_REPLAY",
            segment_index=segment_index,
            segment_cursor=segment_index,
            source_step=source_step,
            command_cursor=command["command_cursor"],
            source_command=command["source_command"],
            source_event_index=command["source_event_index"],
            planned_dispatch_time_s=command["planned_dispatch_time_s"],
            actual_dispatch_time_s=command["actual_dispatch_time_s"],
            atomic_batch_id=command["atomic_batch_id"],
            dispatch_kind=command["dispatch_kind"],
            motion_start_readiness_token=self.readiness_token,
            scheduler_phase=self.service.progress.command_phase,
        )
        self.state = "running"

    def after_adapter_step(self) -> dict[str, Any] | None:
        if self.completion_pending:
            self.completion_pending = False
            self.state = "postsettle"
            self.post_settle_deadline_s = (
                float(self.adapter.sim_time) + self.request.post_run_settle_s
            )
            return None
        if (
            self.state == "readiness_window"
            and len(self.readiness_frames) >= self.required_readiness_frames
        ):
            from sim_worker_runtime import capture_worker_motion_start_readiness
            from .run_fsm50 import _evaluate_motion_start_window

            identity = dict(self.expected_plan_identity or {})
            if not identity:
                return self.fail("worker expected plan identity is unavailable")
            command_dispatch_evidence = {
                "direct_dispatch_attempt_count": len(self.direct_dispatch_attempts),
                "direct_dispatch_attempts": list(self.direct_dispatch_attempts),
                "motion_batch_count": 0,
                "source_segment_start_count": 0,
            }
            rich = _evaluate_motion_start_window(
                adapter=self.adapter,
                startup_ground=self.startup_ground,
                frames=self.readiness_frames[: self.required_readiness_frames],
                plan_identity=identity,
                required_frames=self.required_readiness_frames,
                command_dispatch_idle=not self.direct_dispatch_attempts,
                command_dispatch_evidence=command_dispatch_evidence,
                diagnostic_role=self.request.diagnostic_role,
            )
            shared = capture_worker_motion_start_readiness(
                self.adapter,
                runtime_ready=True,
                current_sim_step=int(self.adapter.sim_steps),
                worker_session_id=self.worker_session_id,
                request_identity=identity,
            )
            rich["shared_production_worker_gate"] = shared
            rich["ready"] = bool(
                self.motion_start_ready_for_role(rich, shared)
                and int(getattr(self.adapter, "root_state_write_count", 0) or 0)
                == int(self.request.expected_root_state_write_count)
            )
            self.motion_start_readiness = rich
            _write_json(self.run_dir / "motion_start_readiness.json", rich)
            if rich["ready"]:
                self.artifact_preflight_ready = True
                self.state = "ready_for_plan"
            else:
                return self.fail(
                    "10-frame MOTION_START_READY preflight failed: "
                    + str(rich.get("window_failed_checks", []) or shared.get("rejection_reason", ""))
                )
        if self.state == "postsettle" and self.post_settle_deadline_s is not None:
            if float(self.adapter.sim_time) + 1.0e-9 >= self.post_settle_deadline_s:
                return self.finalize()
        return None

    def finalize(self) -> dict[str, Any]:
        if self.terminal_message is not None:
            return dict(self.terminal_message)
        try:
            payload = (
                self.finalize_callback(self)
                if self.finalize_callback is not None
                else self._finalize_artifacts()
            )
            if payload.get("finalization_complete") is not True:
                raise RuntimeError("worker artifact finalizer did not close the artifact")
            # A closed diagnostic artifact may truthfully report physical FAIL
            # or artifact_valid=false.  Completion is about evidence closure,
            # never about manufacturing a PASS.
            self.state = "complete"
            kind = "artifact_complete"
            self.terminal_message = {"type": kind, **self.status_dict(), **dict(payload)}
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "failed"
            try:
                payload = self._finalize_failure(self.error)
            except Exception as failure_exc:
                payload = {
                    "artifact_valid": False,
                    "finalization_error": f"{type(failure_exc).__name__}: {failure_exc}",
                }
            self.terminal_message = {"type": "artifact_failed", **self.status_dict(), **payload}
        return dict(self.terminal_message)

    def fail(self, error: str) -> dict[str, Any]:
        self.error = str(error)
        if self.terminal_message is None:
            self.state = "failed"
            try:
                payload = self._finalize_failure(self.error)
            except Exception as exc:
                payload = {"artifact_valid": False, "finalization_error": str(exc)}
            self.terminal_message = {"type": "artifact_failed", **self.status_dict(), **payload}
        return dict(self.terminal_message)

    def _finalize_artifacts(self) -> dict[str, Any]:
        # Kept on the session object so the worker, never the outer controller,
        # owns every file below artifact_root.  Heavy helpers are imported only
        # for an explicitly enabled Gate-1 request.
        from telemetry.exporters import strict_json_dumps, write_json
        from .recording_audit import sha256_file
        from .recording_fast_plan import write_source_dispatch_ledger
        from .wheel_integral_evidence import evaluate_wheel_integral_evidence
        from .ordinary_ui_trajectory import (
            IDENTITY_SCHEMA as ORDINARY_UI_IDENTITY_SCHEMA,
            MANIFEST_FILENAME as ORDINARY_UI_MANIFEST_FILENAME,
            SEAL_FILENAME as ORDINARY_UI_SEAL_FILENAME,
            TRAJECTORY_FILENAME as ORDINARY_UI_TRAJECTORY_FILENAME,
            ordinary_ui_diagnostic_complete,
            write_ordinary_ui_trajectory,
        )
        from .run_fsm50 import (
            _apply_recording_artifact_policy,
            _compare_source_freezes,
            _copy_run_inputs,
            _generate_fsm50_visualization,
            _invalidate_result_for_source_drift,
            _jsonable,
            _mark_artifact_root,
            _recording_visual_manifest,
            _result_payload,
            _runtime_versions,
            _source_freeze,
            _viewport_preclose_evidence_paths,
            _write_checksums,
        )

        success = bool(self.service.stop_reason == "complete")
        immutable_sources = (
            (
                self.request.request_path,
                self.request.artifact_request_sha256,
                "artifact request",
            ),
            (
                self.request.accepted_steps_path,
                self.request.accepted_steps_sha256,
                "accepted steps",
            ),
            (
                Path(self.request.robot_usd_path),
                self.request.expected_robot_usd_sha256,
                "robot USD",
            ),
            (
                Path(self.request.environment_lock_path),
                self.request.environment_lock_sha256,
                "environment lock",
            ),
        )
        for source_path, expected_sha, label in immutable_sources:
            actual_sha = _sha256_file(source_path).lower()
            if actual_sha != str(expected_sha).lower():
                raise RuntimeError(
                    f"{label} changed during worker artifact session: "
                    f"expected={expected_sha} actual={actual_sha}"
                )
        self.collector.finish_episode(
            success=success,
            reason="scheduler complete; strict physical result is evaluated separately"
            if success
            else str(self.service.stop_reason or "scheduler failed"),
        )
        self.adapter.attach_telemetry(None)
        self.video = self.video_capture.finalize()
        run_dir = Path(self.collector.run_dir or self.run_dir).resolve()
        self.run_dir = run_dir
        _copy_run_inputs(
            self.item,
            run_dir,
            self.steps,
            float(self.adapter.max_wheel_speed),
        )
        shutil.copy2(
            self.request.request_path,
            run_dir / "input" / "worker_artifact_request.json",
        )
        dispatch_ledger = write_source_dispatch_ledger(
            csv_path=run_dir / "V003_DISPATCH_TRACE.csv",
            json_path=run_dir / "V003_DISPATCH_TRACE.json",
            source_version=self.request.source_version,
            steps=self.steps,
            plan=self.plan,
            timing_trace=self.service.timing_trace,
        )
        wheel_integral = evaluate_wheel_integral_evidence(
            plan=self.plan,
            timing_trace=self.service.timing_trace,
            telemetry_rows=self.collector.fsm50_rows,
            wheel_direction=float(getattr(self.adapter, "wheel_direction", 1.0)),
        )
        write_json(run_dir / "V003_WHEEL_INTEGRAL_EVIDENCE.json", wheel_integral)
        write_json(run_dir / "production_dispatch_timing.json", self.service.timing_trace)
        write_json(
            run_dir / "runtime_environment.json",
            {
                "source_version": self.request.source_version,
                "execution_path": "sim_worker_process_ipc",
                "artifact_owner": "sim_worker_process",
                "artifact_request_sha256": self.request.artifact_request_sha256,
                "contact_mode": self.request.contact_mode,
                "environment_equivalence_role": (
                    self.request.environment_equivalence_role
                ),
                "diagnostic_role": self.request.diagnostic_role,
                "qualification_scope": self.request.qualification_scope,
                "gate1_eligible": self.request.gate1_eligible,
                "gate1_physical_qualification_eligible": (
                    self.request.gate1_physical_qualification_eligible
                ),
                "environment_equivalence_eligible": (
                    self.request.environment_equivalence_eligible
                ),
                "actual_viewport_video": bool(
                    self.video.get("actual_viewport_video", False)
                ),
                "video_path": str(self.video.get("video_path", "") or ""),
                "video_sha256": str(self.video.get("video_sha256", "") or ""),
                "viewport_video_manifest_path": str(
                    self.video.get("manifest_path", "") or ""
                ),
                "viewport_video_manifest_sha256": str(
                    self.video.get("manifest_sha256", "") or ""
                ),
                "video": _jsonable(self.video),
                "runtime": _runtime_versions(),
                "scene_config": _jsonable(asdict(self.scene_handle.config)),
                "live_scene_baseline": _jsonable(self.live_baseline),
                "live_obstacle_geometry": _jsonable(self.live_obstacle),
                "environment_equivalence": _jsonable(self.environment_equivalence),
                "physics_dt_s": float(self.scene_handle.sim.get_physics_dt()),
                "render_interval": int(self.scene_handle.config.render_interval),
                "contact_sensor_type": (
                    ""
                    if self.scene_handle.contact_sensor is None
                    else type(self.scene_handle.contact_sensor).__name__
                ),
                "contact_sensor_error": str(
                    self.scene_handle.contact_sensor_error or ""
                ),
            },
        )
        respawn = {
            "ok": True,
            "respawned": False,
            "root_pose_written": False,
            "reason": "fresh production worker; cached-root respawn forbidden",
            "adapter_runtime_instance_id": str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            "root_state_write_count": int(
                getattr(self.adapter, "root_state_write_count", 0) or 0
            ),
        }
        result = _result_payload(
            item=self.item,
            service=self.service,
            collector=self.collector,
            respawn=respawn,
            plan=self.plan,
            run_dir=run_dir,
            timed_out=False,
            motion_start_readiness=self.motion_start_readiness,
            pre_first_dispatch_readiness=self.pre_first_dispatch_readiness,
            dispatch_ledger=dispatch_ledger,
            wheel_integral_evidence=wheel_integral,
            trial_id=self.request.trial_id,
            diagnostic_role=self.request.diagnostic_role,
        )
        result.update(
            artifact_owner="sim_worker_process",
            artifact_request_sha256=self.request.artifact_request_sha256,
            request_id=self.request.request_id,
            plan_id=self.request.plan_id,
            artifact_root=str(self.request.artifact_root),
            run_dir=str(run_dir),
            execution_path="sim_worker_process_ipc",
            worker_pid=os.getpid(),
            worker_session_id=self.worker_session_id,
            adapter_runtime_instance_id=str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            root_state_write_count=int(
                getattr(self.adapter, "root_state_write_count", 0) or 0
            ),
            simulation_app_stopped=False,
            contact_mode=self.request.contact_mode,
            environment_equivalence_role=(
                self.request.environment_equivalence_role
            ),
            diagnostic_role=self.request.diagnostic_role,
            qualification_scope=self.request.qualification_scope,
            gate1_eligible=self.request.gate1_eligible,
            gate1_physical_qualification_eligible=(
                self.request.gate1_physical_qualification_eligible
            ),
            environment_equivalence_eligible=(
                self.request.environment_equivalence_eligible
            ),
            physical_qualification_eligible=bool(
                self.request.gate1_physical_qualification_eligible
            ),
            environment_equivalence=self.environment_equivalence,
            video=self.video,
            actual_viewport_video=bool(
                self.video.get("actual_viewport_video", False)
            ),
            video_path=str(self.video.get("video_path", "") or ""),
            video_sha256=str(self.video.get("video_sha256", "") or ""),
            viewport_video_manifest_path=str(
                self.video.get("manifest_path", "") or ""
            ),
            viewport_video_manifest_sha256=str(
                self.video.get("manifest_sha256", "") or ""
            ),
        )
        post_freeze = _source_freeze(
            self.item,
            robot_usd=Path(self.request.robot_usd_path),
        )
        source_comparison = _compare_source_freezes(
            self.source_freeze_pre, post_freeze
        )
        write_json(self.request.artifact_root / "source_freeze_post.json", post_freeze)
        write_json(self.request.artifact_root / "source_integrity.json", source_comparison)
        write_json(run_dir / "source_integrity.json", source_comparison)
        result["source_integrity"] = {
            "ok": bool(source_comparison.get("equal", False)),
            "scope": "recording_version",
            "comparison": source_comparison,
        }
        if not source_comparison.get("equal", False):
            _invalidate_result_for_source_drift(
                result,
                drift=source_comparison,
                scope="worker_recording_session",
            )
        try:
            visualization = _generate_fsm50_visualization(
                run_dir,
                fsm50_rows=self.collector.fsm50_rows,
                state_timeline_rows=self.collector.state_timeline_rows,
                strict_result=result,
            )
        except Exception as exc:
            visualization = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result["visualization"] = visualization
        artifact_valid = _apply_recording_artifact_policy(
            result,
            video=self.video,
            visualization=visualization,
        )
        apply_ordinary_ui_diagnostic_classification(
            result,
            role=self.request.diagnostic_role,
        )
        contact_sensor_disabled = bool(
            self.request.diagnostic_role == "U"
            and self.scene_handle.config.telemetry_contact_sensors_enabled is False
            and self.scene_handle.config.contact_sensor_factory is None
            and self.scene_handle.contact_sensor is None
            and not str(self.scene_handle.contact_sensor_error or "")
        )
        if self.request.diagnostic_role == "U":
            diagnostic_status = ordinary_ui_diagnostic_status(
                role=self.request.diagnostic_role,
                contact_mode=self.request.contact_mode,
                artifact_valid=artifact_valid,
                scheduler_complete=result.get("scheduler_complete") is True,
                dispatch_complete=result.get("dispatch_complete") is True,
                wheel_target_integral_verdict=str(
                    result.get("wheel_target_integral_verdict", "") or ""
                ),
                trajectory_valid=True,
                contact_sensor_disabled=contact_sensor_disabled,
                source_integrity_ok=(
                    dict(result.get("source_integrity", {}) or {}).get("ok") is True
                ),
                root_state_write_count=int(result.get("root_state_write_count", -1)),
            )
        else:
            diagnostic_status = environment_equivalence_diagnostic_status(
                role=self.request.environment_equivalence_role,
                contact_mode=self.request.contact_mode,
                artifact_valid=artifact_valid,
                scheduler_complete=result.get("scheduler_complete") is True,
                dispatch_complete=result.get("dispatch_complete") is True,
                source_integrity_ok=(
                    dict(result.get("source_integrity", {}) or {}).get("ok") is True
                ),
            )
            diagnostic_status.update(
                diagnostic_role="",
                ordinary_ui_diagnostic=False,
                ordinary_ui_diagnostic_complete=False,
                gate1_physical_qualification_eligible=bool(
                    not self.request.environment_equivalence_role
                ),
                gate1_eligible=bool(
                    not self.request.environment_equivalence_role
                ),
                environment_equivalence_eligible=bool(
                    self.request.environment_equivalence_role
                ),
                physical_qualification_eligible=bool(
                    not self.request.environment_equivalence_role
                ),
            )
        result.update(diagnostic_status)
        environment_diagnostic_role = str(
            diagnostic_status["environment_equivalence_role"]
        )
        environment_diagnostic_complete = bool(
            diagnostic_status["environment_equivalence_diagnostic_complete"]
        )
        ordinary_diagnostic_complete = bool(
            diagnostic_status.get("ordinary_ui_diagnostic_complete", False)
        )
        result["contact_sensor_disabled"] = contact_sensor_disabled
        result["contact_sensor_type"] = (
            ""
            if self.scene_handle.contact_sensor is None
            else type(self.scene_handle.contact_sensor).__name__
        )
        result["contact_sensor_error"] = str(
            self.scene_handle.contact_sensor_error or ""
        )
        visual_path = run_dir / "visual_recording_manifest.json"
        write_json(
            visual_path,
            _recording_visual_manifest(
                video=self.video,
                visualization=visualization,
                contact_mode=self.request.contact_mode,
                artifact_valid=artifact_valid,
            ),
        )
        write_json(
            run_dir / "failure_diagnostics.json",
            {
                "classification": result.get("classification"),
                "first_failure_phase": result.get("first_failure_phase"),
                "scheduler_error": self.service.last_error,
                "scheduler_info": self.service.last_info,
                "servo_residual_warnings": self.service.servo_residual_warnings,
                "strict_physical_evidence": result.get("physical_evidence", {}),
                "wheel_integral_evidence": wheel_integral,
                "source_integrity": result.get("source_integrity", {}),
                "contact_mode": self.request.contact_mode,
                "environment_equivalence_role": environment_diagnostic_role,
                "environment_equivalence_diagnostic": bool(
                    environment_diagnostic_role
                ),
                "environment_equivalence_diagnostic_complete": (
                    environment_diagnostic_complete
                ),
                "diagnostic_role": self.request.diagnostic_role,
                "ordinary_ui_diagnostic": self.request.diagnostic_role == "U",
                "ordinary_ui_diagnostic_complete": ordinary_diagnostic_complete,
                "qualification_scope": str(
                    diagnostic_status["qualification_scope"]
                ),
                "gate1_physical_qualification_eligible": bool(
                    diagnostic_status["gate1_physical_qualification_eligible"]
                ),
                "gate1_eligible": bool(diagnostic_status["gate1_eligible"]),
                "environment_equivalence_eligible": bool(
                    diagnostic_status["environment_equivalence_eligible"]
                ),
                "contact_sensor_disabled": contact_sensor_disabled,
                "video": self.video,
                "artifact_valid": artifact_valid,
            },
        )
        required_names = [
            "result.json",
            "failure_diagnostics.json",
            "physical_evidence.json",
            "telemetry_finalization.json",
            "fsm50_telemetry.csv",
            "fsm50_telemetry.jsonl",
            "state_timeline.csv",
            "motion_start_readiness.json",
            "motion_start_pre_first_dispatch.json",
            "V003_DISPATCH_TRACE.csv",
            "V003_DISPATCH_TRACE.json",
            "V003_WHEEL_INTEGRAL_EVIDENCE.json",
            "production_dispatch_timing.json",
            "runtime_environment.json",
            "visual_recording_manifest.json",
            "source_integrity.json",
            "worker_recording_session.json",
            "input/worker_artifact_request.json",
            "checksums.sha256",
        ]
        if self.request.diagnostic_role == "U":
            required_names.extend(
                [
                    ORDINARY_UI_TRAJECTORY_FILENAME,
                    ORDINARY_UI_MANIFEST_FILENAME,
                    ORDINARY_UI_SEAL_FILENAME,
                ]
            )
        result["required_evidence_paths"] = [
            str((run_dir / name).resolve()) for name in required_names
        ]
        result["required_evidence_paths"].extend(
            str(path)
            for path in _viewport_preclose_evidence_paths(run_dir, self.video)
        )
        if self.request.diagnostic_role == "U":
            result["ordinary_ui_trajectory"] = {
                "schema_version": "fsm50.ordinary_ui.result_binding.v1",
                "diagnostic_complete": ordinary_diagnostic_complete,
                "trajectory_path": str(
                    (run_dir / ORDINARY_UI_TRAJECTORY_FILENAME).resolve()
                ),
                "manifest_path": str(
                    (run_dir / ORDINARY_UI_MANIFEST_FILENAME).resolve()
                ),
                "seal_path": str((run_dir / ORDINARY_UI_SEAL_FILENAME).resolve()),
                "durable_result_binding": (
                    "identity_envelope.evidence.durable_result.artifact_sha256"
                ),
                "full_physical_verdict": "NOT_EVALUABLE",
                "gate1_eligible": False,
                "environment_equivalence_eligible": False,
            }
        pointer_path = self.request.artifact_root / "artifact_pointer.json"
        write_json(pointer_path, {"run_dir": str(run_dir)})
        session_manifest = {
            **self.status_dict(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "finalization_complete": True,
            "artifact_owner": "sim_worker_process",
            "artifact_request_sha256": self.request.artifact_request_sha256,
            "plan_id": self.request.plan_id,
            "plan_sha256": str(self.plan.plan_sha256),
            "scheduler_stop_reason": str(self.service.stop_reason or ""),
            "scheduler_complete": success,
            "classification": result.get("classification"),
            "artifact_valid": artifact_valid,
            "environment_equivalence_role": environment_diagnostic_role,
            "environment_equivalence_diagnostic": bool(
                environment_diagnostic_role
            ),
            "environment_equivalence_diagnostic_complete": (
                environment_diagnostic_complete
            ),
            "diagnostic_role": self.request.diagnostic_role,
            "ordinary_ui_diagnostic": self.request.diagnostic_role == "U",
            "ordinary_ui_diagnostic_complete": ordinary_diagnostic_complete,
            "qualification_scope": str(
                diagnostic_status["qualification_scope"]
            ),
            "gate1_physical_qualification_eligible": bool(
                diagnostic_status["gate1_physical_qualification_eligible"]
            ),
            "gate1_eligible": bool(diagnostic_status["gate1_eligible"]),
            "environment_equivalence_eligible": bool(
                diagnostic_status["environment_equivalence_eligible"]
            ),
            "contact_sensor_disabled": contact_sensor_disabled,
            "video": self.video,
        }
        write_json(run_dir / "worker_recording_session.json", session_manifest)
        result_path = run_dir / "result.json"
        ordinary_receipt: dict[str, Any] = {}
        predicted_result_sha256 = ""
        if self.request.diagnostic_role == "U":
            result_payload = (
                strict_json_dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            predicted_result_sha256 = hashlib.sha256(result_payload).hexdigest()
            readiness_path = run_dir / "motion_start_pre_first_dispatch.json"
            dispatch_path = run_dir / "V003_DISPATCH_TRACE.json"
            readiness_artifact = json.loads(
                readiness_path.read_text(encoding="utf-8")
            )
            dispatch_artifact = json.loads(
                dispatch_path.read_text(encoding="utf-8")
            )
            readiness_token_sha256 = str(
                readiness_artifact.get(
                    "readiness_token_sha256", ""
                )
                or ""
            ).lower()
            adapter_runtime_instance_id = str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            )
            identity = {
                "source_version": self.request.source_version,
                "accepted_steps_sha256": self.request.accepted_steps_sha256,
                "plan_sha256": str(self.plan.plan_sha256),
                "plan_id": self.request.plan_id,
                "request_id": self.request.request_id,
                "worker_session_id": self.worker_session_id,
                "adapter_runtime_instance_id": adapter_runtime_instance_id,
                "readiness_token_sha256": readiness_token_sha256,
                "root_state_write_count": 0,
            }
            readiness_plan_identity = dict(
                readiness_artifact.get("plan_identity", {}) or {}
            )
            expected_readiness_fields = {
                "source_version": identity["source_version"],
                "source_sha256": identity["accepted_steps_sha256"],
                "plan_sha256": identity["plan_sha256"],
                "plan_id": identity["plan_id"],
                "request_id": identity["request_id"],
                "worker_session_id": identity["worker_session_id"],
            }
            for key, expected_value in expected_readiness_fields.items():
                if readiness_plan_identity.get(key) != expected_value:
                    raise RuntimeError(
                        f"ordinary-UI readiness {key} differs from immutable identity"
                    )
            if (
                str(
                    readiness_artifact.get(
                        "adapter_runtime_instance_id", ""
                    )
                    or ""
                )
                != adapter_runtime_instance_id
                or readiness_artifact.get("root_state_write_count") != 0
                or str(
                    dispatch_artifact.get("source_version", "") or ""
                )
                != identity["source_version"]
                or str(
                    dispatch_artifact.get(
                        "motion_start_readiness_token", ""
                    )
                    or ""
                ).lower()
                != readiness_token_sha256
            ):
                raise RuntimeError(
                    "ordinary-UI readiness/dispatch durable identity is inconsistent"
                )
            identity_envelope = {
                "schema_version": ORDINARY_UI_IDENTITY_SCHEMA,
                **identity,
                "evidence": {
                    "durable_result": {
                        "artifact_sha256": predicted_result_sha256,
                        "identity_fields": dict(identity),
                    },
                    "immutable_request": {
                        "artifact_sha256": self.request.artifact_request_sha256,
                        "identity_fields": {
                            key: identity[key]
                            for key in (
                                "source_version",
                                "accepted_steps_sha256",
                                "plan_sha256",
                                "plan_id",
                                "request_id",
                                "root_state_write_count",
                            )
                        },
                    },
                    "readiness": {
                        "artifact_sha256": sha256_file(readiness_path).lower(),
                        "identity_fields": {
                            key: identity[key]
                            for key in (
                                "source_version",
                                "plan_sha256",
                                "plan_id",
                                "request_id",
                                "worker_session_id",
                                "adapter_runtime_instance_id",
                                "readiness_token_sha256",
                                "root_state_write_count",
                            )
                        },
                    },
                    "dispatch_ledger": {
                        "artifact_sha256": sha256_file(dispatch_path).lower(),
                        "identity_fields": {
                            key: identity[key]
                            for key in (
                                "source_version",
                                "plan_id",
                                "readiness_token_sha256",
                            )
                        },
                    },
                },
            }
            ordinary_receipt = write_ordinary_ui_trajectory(
                run_dir,
                self.collector.fsm50_rows,
                identity_envelope=identity_envelope,
            )
            if (
                not ordinary_ui_diagnostic_complete(ordinary_receipt)
                or ordinary_receipt.get("diagnostic_complete") is not True
                or ordinary_receipt.get("full_physical_verdict")
                != "NOT_EVALUABLE"
                or ordinary_receipt.get("gate1_eligible") is not False
                or ordinary_receipt.get("environment_equivalence_eligible")
                is not False
            ):
                raise RuntimeError(
                    "ordinary-UI trajectory receipt failed strict readback validation"
                )
        write_json(result_path, result)
        if (
            self.request.diagnostic_role == "U"
            and sha256_file(result_path).lower() != predicted_result_sha256
        ):
            raise RuntimeError(
                "durable result bytes differ from the ordinary-UI identity binding"
            )
        _write_checksums(self.run_dir)
        _mark_artifact_root(self.request.artifact_root, valid=artifact_valid)
        checksums_path = run_dir / "checksums.sha256"
        telemetry_path = run_dir / "telemetry_finalization.json"
        return {
            "finalization_complete": True,
            "artifact_valid": bool(result.get("artifact_valid", False)),
            "classification": str(result.get("classification", "")),
            "scheduler_complete": bool(result.get("scheduler_complete", False)),
            "physical_success": bool(result.get("physical_success", False)),
            "strict_full_success": bool(result.get("strict_full_success", False)),
            "artifact_owner": "sim_worker_process",
            "artifact_request_sha256": self.request.artifact_request_sha256,
            "request_id": self.request.request_id,
            "plan_id": self.request.plan_id,
            "plan_sha256": str(self.plan.plan_sha256),
            "source_version": self.request.source_version,
            "trial_id": self.request.trial_id,
            "contact_mode": self.request.contact_mode,
            "environment_equivalence_role": environment_diagnostic_role,
            "environment_equivalence_diagnostic": bool(
                environment_diagnostic_role
            ),
            "environment_equivalence_diagnostic_complete": (
                environment_diagnostic_complete
            ),
            "diagnostic_role": self.request.diagnostic_role,
            "ordinary_ui_diagnostic": self.request.diagnostic_role == "U",
            "ordinary_ui_diagnostic_complete": ordinary_diagnostic_complete,
            "qualification_scope": str(
                diagnostic_status["qualification_scope"]
            ),
            "gate1_physical_qualification_eligible": bool(
                diagnostic_status["gate1_physical_qualification_eligible"]
            ),
            "gate1_eligible": bool(diagnostic_status["gate1_eligible"]),
            "environment_equivalence_eligible": bool(
                diagnostic_status["environment_equivalence_eligible"]
            ),
            "physical_qualification_eligible": bool(
                diagnostic_status["physical_qualification_eligible"]
            ),
            "contact_sensor_disabled": contact_sensor_disabled,
            "ordinary_ui_trajectory_receipt": ordinary_receipt,
            "accepted_steps_sha256": self.request.accepted_steps_sha256,
            "adapter_runtime_instance_id": str(
                getattr(self.adapter, "runtime_instance_id", "") or ""
            ),
            "root_state_write_count": int(
                getattr(self.adapter, "root_state_write_count", 0) or 0
            ),
            "artifact_root": str(self.request.artifact_root),
            "run_dir": str(run_dir),
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path).lower(),
            "artifact_pointer_path": str(pointer_path),
            "artifact_pointer_sha256": sha256_file(pointer_path).lower(),
            "checksums_path": str(checksums_path),
            "checksums_sha256": sha256_file(checksums_path).lower(),
            "telemetry_finalization_path": str(telemetry_path),
            "telemetry_finalization_sha256": sha256_file(telemetry_path).lower(),
            "visual_manifest_path": str(visual_path),
            "visual_manifest_sha256": sha256_file(visual_path).lower(),
            "error": "",
        }

    def _finalize_failure(self, error: str) -> dict[str, Any]:
        try:
            if self.adapter is not None:
                self.adapter.attach_telemetry(None)
        except Exception:
            pass
        try:
            if self.collector is not None:
                self.collector.finish_episode(success=False, reason=str(error))
                self.run_dir = self.collector.run_dir or self.run_dir
        except Exception:
            pass
        try:
            if self.video_capture is not None:
                self.video = self.video_capture.finalize()
        except Exception:
            self.video = {}
        self.request.artifact_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            **self.status_dict(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
            "artifact_valid": False,
            "video": self.video,
        }
        _write_json(Path(self.run_dir) / "worker_recording_session.json", manifest)
        partial = self.request.artifact_root / ".partial"
        (self.request.artifact_root / ".finalized").unlink(missing_ok=True)
        if partial.exists():
            partial.replace(self.request.artifact_root / ".failed")
        else:
            (self.request.artifact_root / ".failed").write_text("failed\n", encoding="utf-8")
        return manifest
