"""Minimal Gate-A recording capture inside the production replay worker.

This is an explicit normal-development hook.  It observes the existing
``SimTimePlaybackService -> SimRobotAdapter`` execution path; it neither
compiles commands nor enables the strict A/B contact-sensor bank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from telemetry.com_metrics import quat_wxyz_to_rpy

from .support_classifier import (
    LEGS,
    ObstacleGeometry,
    WheelObservation,
    classify_wheel_contact,
)
from .viewport_buffer_video import ActiveViewportBufferVideoRecorder


REQUEST_SCHEMA = "fsm50.worker_task_replay_request.v1"
SESSION_SCHEMA = "fsm50.worker_task_replay_session.v1"
TASK_INPUTS_SCHEMA = "fsm50.replay_task_inputs.v1"
TELEMETRY_SCHEMA = "fsm50.minimal_task_telemetry.v1"
DEFAULT_TELEMETRY_HZ = 15.0
DEFAULT_VIDEO_FPS = 15.0
DEFAULT_POST_SETTLE_S = 0.5
WHEEL_RADIUS_M = 0.04998999834060672
WHEEL_TARGET_NUMERIC_TOLERANCE = 1.0e-12
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
        "plan_id",
        "plan_sha256",
        "plan_event_count",
        "plan_segment_count",
        "source_version",
        "height_mm",
        "step_count",
        "run_dir",
        "accepted_steps_path",
        "accepted_steps_sha256",
        "telemetry_hz",
        "video_fps",
        "capture_video",
        "post_run_settle_s",
        "timeout_s",
        "filtered_contact_bank_enabled",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError("task replay request must be a JSON object")
    return value


def _jsonable(value: Any) -> Any:
    """Return strict-JSON data, mapping unavailable/non-finite values to null."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
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


def _write_telemetry_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sim_time_s",
        "sim_step",
        "source_step",
        "command_cursor",
        "segment_cursor",
        "event_cursor",
        "scheduler_state",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_roll_rad",
        "base_pitch_rad",
        "base_yaw_rad",
        "robot_state_finite",
        "active_contact_count_available",
        "geometry_support_candidate_count",
        "stability_state",
        "joint_limit_violation",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            position = dict(row.get("base_position_m", {}) or {})
            flat = {
                **row,
                "base_x_m": position.get("x"),
                "base_y_m": position.get("y"),
                "base_z_m": position.get("z"),
            }
            writer.writerow({key: _jsonable(flat.get(key)) for key in fields})


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if type(value) is not str or not value.strip():
        raise ValueError(f"task replay request {key} must be a non-empty string")
    return value.strip()


def _required_int(raw: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = raw.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"task replay request {key} must be an integer >= {minimum}")
    return value


def _required_float(
    raw: Mapping[str, Any], key: str, *, minimum: float = 0.0
) -> float:
    value = raw.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"task replay request {key} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= minimum:
        raise ValueError(f"task replay request {key} must be finite and > {minimum}")
    return parsed


@dataclass(frozen=True)
class WorkerTaskReplayRequest:
    request_path: Path
    request_id: str
    plan_id: str
    plan_sha256: str
    plan_event_count: int
    plan_segment_count: int
    source_version: str
    height_mm: int
    step_count: int
    run_dir: Path
    accepted_steps_path: Path
    accepted_steps_sha256: str
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
            "plan_id": self.plan_id,
            "source_version": self.source_version,
            "height_mm": self.height_mm,
            "step_count": self.step_count,
            "run_dir": str(self.run_dir),
            "telemetry_hz": self.telemetry_hz,
            "video_fps": self.video_fps,
            "filtered_contact_bank_enabled": False,
            "preflight_ok": True,
        }


def load_worker_task_replay_request(
    request_path: str | Path | None,
) -> WorkerTaskReplayRequest | None:
    """Load an explicit normal-development request; empty means fully disabled."""

    text = str(request_path or "").strip()
    if not text:
        return None
    path = Path(text).resolve()
    raw = _strict_json(path)
    missing = sorted(_REQUIRED_KEYS - set(raw))
    unexpected = sorted(set(raw) - _REQUIRED_KEYS)
    if missing:
        raise ValueError("task replay request is missing keys: " + ", ".join(missing))
    if unexpected:
        raise ValueError("task replay request has unexpected keys: " + ", ".join(unexpected))
    if raw.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("unsupported task replay request schema")
    if raw.get("enabled") is not True:
        raise ValueError("task replay request must set enabled=true")
    if raw.get("execution_mode") != "normal_development":
        raise ValueError("task replay request must use normal_development mode")
    if raw.get("capture_video") is not True:
        raise ValueError("normal task replay requires viewport video")
    if raw.get("filtered_contact_bank_enabled") is not False:
        raise ValueError("normal task replay forbids the filtered A/B contact bank")

    run_dir = Path(_required_text(raw, "run_dir")).resolve()
    steps_path = Path(_required_text(raw, "accepted_steps_path")).resolve()
    if not steps_path.is_file():
        raise FileNotFoundError(steps_path)
    claimed_steps_sha = _required_text(raw, "accepted_steps_sha256").lower()
    actual_steps_sha = _sha256_file(steps_path).lower()
    if claimed_steps_sha != actual_steps_sha:
        raise ValueError("accepted_steps_sha256 does not match the selected recording")
    plan_sha = _required_text(raw, "plan_sha256").lower()
    if len(plan_sha) != 64 or any(ch not in "0123456789abcdef" for ch in plan_sha):
        raise ValueError("plan_sha256 must be a lowercase SHA-256 digest")
    telemetry_hz = _required_float(raw, "telemetry_hz")
    video_fps = _required_float(raw, "video_fps")
    if not 10.0 <= telemetry_hz <= 30.0:
        raise ValueError("normal task telemetry_hz must remain in the minimal 10-30 Hz range")
    if abs(video_fps - DEFAULT_VIDEO_FPS) > 1.0e-9:
        raise ValueError("normal task viewport video must use 15 fps")

    return WorkerTaskReplayRequest(
        request_path=path,
        request_id=_required_text(raw, "request_id"),
        plan_id=_required_text(raw, "plan_id"),
        plan_sha256=plan_sha,
        plan_event_count=_required_int(raw, "plan_event_count", minimum=1),
        plan_segment_count=_required_int(raw, "plan_segment_count", minimum=1),
        source_version=_required_text(raw, "source_version"),
        height_mm=_required_int(raw, "height_mm", minimum=1),
        step_count=_required_int(raw, "step_count", minimum=1),
        run_dir=run_dir,
        accepted_steps_path=steps_path,
        accepted_steps_sha256=actual_steps_sha,
        telemetry_hz=telemetry_hz,
        video_fps=video_fps,
        post_run_settle_s=_required_float(raw, "post_run_settle_s"),
        timeout_s=_required_float(raw, "timeout_s"),
        capture_video=True,
    )


def validate_worker_task_plan_binding(
    request: WorkerTaskReplayRequest,
    *,
    plan: Any,
    request_id: str,
    plan_id: str,
) -> list[str]:
    errors: list[str] = []
    if str(request_id) != request.request_id:
        errors.append("task replay request_id does not match the worker request")
    if str(plan_id) != request.plan_id:
        errors.append("task replay plan_id does not match the worker request")
    if str(getattr(plan, "profile", "") or "") != "motion_only":
        errors.append(
            "task replay requires the canonical production Fast profile motion_only"
        )
    if str(getattr(plan, "plan_sha256", "") or "").lower() != request.plan_sha256:
        errors.append("task replay plan SHA-256 does not match the worker request")
    if len(list(getattr(plan, "events", []) or [])) != request.plan_event_count:
        errors.append("task replay plan event count does not match the worker request")
    if len(list(getattr(plan, "segments", []) or [])) != request.plan_segment_count:
        errors.append("task replay plan segment count does not match the worker request")
    return errors


def _matrix(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=float)
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)
    except Exception:
        return np.empty((0,), dtype=float)


def _first_row(value: Any) -> np.ndarray:
    array = _matrix(value)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array


def _exact_single_vector(value: Any, length: int) -> tuple[np.ndarray, bool]:
    """Accept exactly one environment vector, never a flattened multi-env row."""

    array = _matrix(value)
    if array.shape == (length,):
        return array, True
    if array.shape == (1, length):
        return array[0], True
    flattened = array.reshape(-1) if array.size else np.empty((0,), dtype=float)
    return flattened[:length], False


class WorkerTaskReplaySession:
    """Collect small task evidence while the production scheduler owns motion."""

    def __init__(
        self,
        request: WorkerTaskReplayRequest,
        *,
        worker_session_id: str,
        recorder_factory: Callable[..., Any] = ActiveViewportBufferVideoRecorder,
    ) -> None:
        self.request = request
        self.worker_session_id = str(worker_session_id)
        self.recorder_factory = recorder_factory
        self.state = "ready_for_plan"
        self.error = ""
        self.adapter: Any | None = None
        self.scene_handle: Any | None = None
        self.service: Any | None = None
        self.recorder: Any | None = None
        self.video: dict[str, Any] = {}
        self.video_writer_quiesced = False
        self.observer_attached = False
        self.telemetry_attached = False
        self.rows: list[dict[str, Any]] = []
        self.attach_sim_time_s = 0.0
        self.playback_finished_sim_time_s: float | None = None
        self.playback_success: bool | None = None
        self.playback_reason = ""
        self.infrastructure_failure = False
        self.simulation_app_stopped = False
        self.replay_callback_started = False
        self.replay_callback_label = ""
        self.replay_callback_expected_event_count = 0
        self.replay_callback_final_time_s = 0.0
        self.replay_callback_started_sim_time_s = 0.0
        self.replay_event_indices_recorded: list[int] = []
        self.replay_event_indices_seen: set[int] = set()
        self.last_replay_event: dict[str, Any] = {}
        self.command_callback_count = 0
        self.last_command_callback: dict[str, Any] = {}
        self.next_sample_sim_time_s = 0.0
        self.peak_roll_rad = 0.0
        self.peak_pitch_rad = 0.0
        self.nonfinite_state_detected = False
        self.joint_limit_violation_detected = False
        self.joint_limit_evidence_available = True
        self.unsafe_joint_target_detected: bool | None = None
        self.plan_target_audit: dict[str, Any] = {
            "available": False,
            "source": "UNAVAILABLE",
            "unsafe": None,
            "errors": ["plan target audit has not run"],
        }
        self.traversal = {
            leg: {
                "airborne_seen_before_top": False,
                "airborne_first_s": None,
                "front_face_crossing_s": None,
                "top_seen": False,
                "top_first_s": None,
                "illegal_drive_up": False,
            }
            for leg in LEGS
        }
        self.terminal_payload: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.state in {"complete", "failed"}

    @property
    def fast_close_ready(self) -> bool:
        return bool(
            self.state in {"complete", "failed"}
            and self.video_writer_quiesced
            and self.terminal_payload
            and Path(
                str(self.terminal_payload.get("worker_result_path", "") or "")
            ).is_file()
        )

    def status_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA,
            "enabled": True,
            "execution_mode": "normal_development",
            "request_id": self.request.request_id,
            "source_version": self.request.source_version,
            "state": self.state,
            "terminal": self.terminal,
            "fast_close_ready": self.fast_close_ready,
            "video_writer_quiesced": self.video_writer_quiesced,
            "sample_count": len(self.rows),
            "replay_callback_started": self.replay_callback_started,
            "replay_event_callback_count": len(
                self.replay_event_indices_recorded
            ),
            "replay_event_callback_expected_count": (
                self.replay_callback_expected_event_count
            ),
            "filtered_contact_bank_enabled": False,
            "error": self.error,
        }

    def attach_verified_plan(
        self,
        *,
        plan: Any,
        service: Any,
        adapter: Any,
        scene_handle: Any,
    ) -> None:
        if self.state != "ready_for_plan":
            raise RuntimeError(f"task replay session is not ready: state={self.state}")
        binding_errors = validate_worker_task_plan_binding(
            self.request,
            plan=plan,
            request_id=str(getattr(service, "request_id", "") or ""),
            plan_id=str(getattr(service, "plan_id", "") or ""),
        )
        if binding_errors:
            raise RuntimeError("; ".join(binding_errors))
        if getattr(adapter, "telemetry_collector", None) is not None:
            raise RuntimeError("normal task replay found another telemetry collector")
        if getattr(adapter, "artifact_render_observer", None) is not None:
            raise RuntimeError("normal task replay found another viewport observer")

        self.request.run_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.scene_handle = scene_handle
        self.service = service
        self.plan_target_audit = self._audit_plan_targets(plan, adapter)
        self.unsafe_joint_target_detected = self.plan_target_audit.get("unsafe")
        if self.unsafe_joint_target_detected is None:
            raise RuntimeError(
                "production plan target-limit audit is unavailable: "
                + "; ".join(
                    str(value)
                    for value in list(
                        self.plan_target_audit.get("errors", []) or []
                    )
                )
            )
        if self.unsafe_joint_target_detected is True:
            raise RuntimeError(
                "production plan contains an unsafe applied target: "
                + json.dumps(
                    list(self.plan_target_audit.get("violations", []) or []),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        self.attach_sim_time_s = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        self.next_sample_sim_time_s = self.attach_sim_time_s
        self.recorder = self.recorder_factory(
            self.request.run_dir,
            enabled=True,
            fps=self.request.video_fps,
        )
        if self.recorder.start() is not True or str(
            getattr(self.recorder, "error", "") or ""
        ):
            raise RuntimeError(
                "viewport recorder did not start: "
                + str(getattr(self.recorder, "error", "") or "unknown error")
            )
        adapter.attach_artifact_render_observer(self.recorder)
        self.observer_attached = True
        adapter.attach_telemetry(self)
        self.telemetry_attached = True
        self.state = "recording"
        self._capture(adapter, force=True)

    def record_start_boundary(self) -> None:
        # Kept as a small explicit worker hook for status/debugging.  Admission
        # remains owned by SimTimePlaybackService.
        if self.state != "recording":
            raise RuntimeError(f"task replay start boundary in state={self.state}")

    def start_replay(
        self,
        *,
        label: str,
        event_count: int,
        final_time_s: float,
        started_sim_time_s: float,
    ) -> None:
        """Implement the production telemetry-collector replay lifecycle."""

        self.replay_callback_started = True
        self.replay_callback_label = str(label or "production playback")
        self.replay_callback_expected_event_count = int(event_count)
        self.replay_callback_final_time_s = float(final_time_s)
        self.replay_callback_started_sim_time_s = float(started_sim_time_s)

    def record_replay_event(
        self, adapter: Any, event: Any, event_index: int
    ) -> None:
        """Retain the compact event cursor without sampling pre-dispatch state."""

        index = int(event_index)
        self.replay_event_indices_recorded.append(index)
        self.replay_event_indices_seen.add(index)
        self.last_replay_event = {
            "event_index": index,
            "global_command_index": getattr(event, "global_command_index", None),
            "source_step": getattr(event, "source_step", None),
            "segment_index": getattr(event, "segment_index", None),
            "command": str(getattr(event, "command", "") or ""),
            "sim_time_s": float(getattr(adapter, "sim_time", 0.0) or 0.0),
        }

    def _replay_callback_admission(self) -> dict[str, Any]:
        """Validate the observer callbacks without changing dispatch semantics."""

        expected_count = int(self.request.plan_event_count)
        recorded = list(self.replay_event_indices_recorded)
        expected_indices = list(range(expected_count))
        counts: dict[int, int] = {}
        for index in recorded:
            counts[index] = counts.get(index, 0) + 1
        missing = [index for index in expected_indices if index not in counts]
        duplicates = sorted(
            index for index, count in counts.items() if count > 1
        )
        out_of_range = sorted(
            index
            for index in counts
            if index < 0 or index >= expected_count
        )
        errors: list[str] = []
        if not self.replay_callback_started:
            errors.append("start_replay callback was not observed")
        if self.replay_callback_expected_event_count != expected_count:
            errors.append(
                "start_replay event_count does not match the admitted plan: "
                f"callback={self.replay_callback_expected_event_count} "
                f"plan={expected_count}"
            )
        if missing:
            errors.append(f"missing replay event indices: {missing}")
        if duplicates:
            errors.append(f"duplicate replay event indices: {duplicates}")
        if out_of_range:
            errors.append(f"out-of-range replay event indices: {out_of_range}")
        if recorded != expected_indices and not (missing or duplicates or out_of_range):
            errors.append("replay event indices were observed out of order")
        return {
            "valid": not errors,
            "replay_started": self.replay_callback_started,
            "expected_event_count": self.replay_callback_expected_event_count,
            "plan_event_count": expected_count,
            "recorded_event_count": len(recorded),
            "recorded_event_indices": recorded,
            "missing_event_indices": missing,
            "duplicate_event_indices": duplicates,
            "out_of_range_event_indices": out_of_range,
            "errors": errors,
        }

    def record_command(self, adapter: Any, message: Any, command: str) -> None:
        """Accept SimRobotAdapter's adjacent command callback contract."""

        self.command_callback_count += 1
        self.last_command_callback = {
            "command": str(command or ""),
            "source": str(getattr(message, "source", "") or ""),
            "playback_event_index": getattr(
                message, "playback_event_index", None
            ),
            "source_step": getattr(message, "source_step", None),
            "sim_time_s": float(getattr(adapter, "sim_time", 0.0) or 0.0),
        }

    def finish_replay(self, *, success: bool, reason: str, sim_time_s: float) -> None:
        self.playback_success = bool(success)
        self.playback_reason = "complete" if success else str(reason or "stopped")
        self.playback_finished_sim_time_s = float(sim_time_s)
        if self.state == "recording":
            self.state = "settling" if success else "failed_pending_finalize"

    def on_step(self, adapter: Any, _dt: float) -> None:
        if self.state not in {"recording", "settling", "failed_pending_finalize"}:
            return
        self._capture(adapter)

    def before_adapter_step(self) -> None:
        return

    def after_adapter_step(self) -> dict[str, Any] | None:
        if self.terminal_payload is not None:
            return None
        adapter = self.adapter
        if adapter is None:
            return None
        current_sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        # SimTimePlaybackService normally calls our collector ``finish_replay``
        # from its own _finish path before this physics step.  This read-only
        # fallback makes the lifecycle robust to compatible service fakes or a
        # future caller that reports the same terminal status without invoking
        # the collector callback.
        if self.state == "recording" and self.service is not None:
            service_active = bool(getattr(self.service, "active", False))
            service_reason = str(getattr(self.service, "stop_reason", "") or "")
            if not service_active and service_reason:
                completed_at = float(
                    getattr(self.service, "completed_at_sim_s", current_sim_time)
                    or current_sim_time
                )
                self.finish_replay(
                    success=service_reason == "complete",
                    reason=service_reason,
                    sim_time_s=completed_at,
                )
        if current_sim_time - self.attach_sim_time_s > self.request.timeout_s:
            return self.fail(
                "task replay session timed out",
                infrastructure_failure=True,
            )
        if self.state == "failed_pending_finalize":
            return self.fail(self.playback_reason or "production playback stopped")
        if self.state == "settling" and self.playback_finished_sim_time_s is not None:
            if (
                current_sim_time - self.playback_finished_sim_time_s
                + 1.0e-9
                >= self.request.post_run_settle_s
            ):
                return self._finalize(success=True, error="")
        return None

    def fail(
        self,
        error: str,
        *,
        infrastructure_failure: bool = False,
        simulation_app_stopped: bool = False,
    ) -> dict[str, Any]:
        if self.terminal_payload is not None:
            return dict(self.terminal_payload)
        self.error = str(error or "normal task replay failed")
        self.infrastructure_failure = bool(
            self.infrastructure_failure or infrastructure_failure
        )
        self.simulation_app_stopped = bool(
            self.simulation_app_stopped or simulation_app_stopped
        )
        return self._finalize(success=False, error=self.error)

    def _playback_status(self) -> dict[str, Any]:
        if self.service is None:
            return {}
        return dict(
            self.service.status_dict(
                current_sim_time_s=float(getattr(self.adapter, "sim_time", 0.0) or 0.0),
                current_wall_time_s=time.time(),
                compact=True,
            )
            or {}
        )

    def _root_state(
        self, adapter: Any
    ) -> tuple[list[float], list[float], bool]:
        data = getattr(getattr(adapter, "robot", None), "data", None)
        pose, pose_shape_ok = _exact_single_vector(
            getattr(data, "root_pose_w", None), 7
        )
        velocity, velocity_shape_ok = _exact_single_vector(
            getattr(data, "root_vel_w", None), 6
        )
        return pose.tolist(), velocity.tolist(), bool(
            pose_shape_ok and velocity_shape_ok
        )

    def _joint_state(
        self, adapter: Any
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
        expected_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
        positions, positions_shape_ok = _exact_single_vector(
            getattr(data, "joint_pos", None), len(expected_names)
        )
        velocities, velocities_shape_ok = _exact_single_vector(
            getattr(data, "joint_vel", None), len(expected_names)
        )
        return (
            {
                name: float(positions[index])
                for index, name in enumerate(names[: positions.size])
            },
            {
                name: float(velocities[index])
                for index, name in enumerate(names[: velocities.size])
            },
            bool(
                positions_shape_ok
                and velocities_shape_ok
                and len(names) == len(expected_names)
                and set(names) == set(expected_names)
            ),
        )

    def _wheel_centers(self, adapter: Any) -> dict[str, tuple[float, float, float]]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
        states = _first_row(getattr(data, "body_link_state_w", None))
        if states.size == 0:
            states = _first_row(getattr(data, "body_state_w", None))
        # ``_first_row`` removes only the environment dimension, leaving
        # [body, state] for real robot tensors.
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
        violation = False
        for name in SERVO_JOINT_NAMES:
            record_value = records.get(name)
            if not isinstance(record_value, Mapping):
                return None, False
            record = dict(record_value)
            value = joint_q.get(name)
            if value is None or not math.isfinite(float(value)):
                return None, False
            try:
                minimum = float(record["min_rad"])
                maximum = float(record["max_rad"])
            except (KeyError, TypeError, ValueError):
                return None, False
            if not (
                math.isfinite(minimum)
                and math.isfinite(maximum)
                and minimum < maximum
            ):
                return None, False
            if float(value) < minimum - 1.0e-3 or float(value) > maximum + 1.0e-3:
                violation = True
        return violation, True

    def _audit_plan_targets(self, plan: Any, adapter: Any) -> dict[str, Any]:
        """Audit every canonical segment target before production dispatch.

        This intentionally evaluates the un-applied plan values.  Scheduler
        completion cannot prove safety because the adapter clamps commands at
        application time.
        """

        errors: list[str] = []
        violations: list[dict[str, Any]] = []
        clamped_wheel_targets: list[dict[str, Any]] = []
        servo_target_count = 0
        wheel_target_count = 0
        segments = list(getattr(plan, "segments", []) or [])
        if len(segments) != self.request.plan_segment_count:
            errors.append("canonical segment collection is incomplete")

        command_to_actual = getattr(adapter, "command_to_actual_target_deg", None)
        final_limits = getattr(adapter, "get_final_target_limits_deg", None)
        try:
            max_wheel_speed = float(getattr(adapter, "max_wheel_speed"))
        except (TypeError, ValueError, AttributeError):
            max_wheel_speed = float("nan")
        if not callable(command_to_actual) or not callable(final_limits):
            errors.append("production servo target-limit transforms are unavailable")
        if not math.isfinite(max_wheel_speed) or max_wheel_speed <= 0.0:
            errors.append("production wheel target limit is unavailable")

        for fallback_index, segment in enumerate(segments):
            segment_index = getattr(segment, "segment_index", fallback_index)
            servo_raw = getattr(segment, "servo_targets", None)
            wheel_requested_raw = getattr(
                segment, "wheel_requested_velocity_rad_s", None
            )
            wheel_applied_raw = getattr(
                segment, "wheel_applied_target_rad_s", None
            )
            if not isinstance(servo_raw, Mapping):
                errors.append(f"segment {segment_index} servo targets are unavailable")
                servo_targets: dict[str, Any] = {}
            else:
                servo_targets = dict(servo_raw)
            if not isinstance(wheel_requested_raw, Mapping) or not isinstance(
                wheel_applied_raw, Mapping
            ):
                errors.append(f"segment {segment_index} wheel targets are unavailable")
                wheel_requested: dict[str, Any] = {}
                wheel_applied: dict[str, Any] = {}
            else:
                wheel_requested = dict(wheel_requested_raw)
                wheel_applied = dict(wheel_applied_raw)

            unknown_servos = sorted(set(servo_targets) - set(SERVO_JOINT_NAMES))
            if unknown_servos:
                violations.append(
                    {
                        "segment_index": segment_index,
                        "kind": "unknown_servo_target",
                        "joints": unknown_servos,
                    }
                )
            for name, raw_value in servo_targets.items():
                servo_target_count += 1
                try:
                    command_deg = float(raw_value)
                except (TypeError, ValueError):
                    command_deg = float("nan")
                if not math.isfinite(command_deg):
                    violations.append(
                        {
                            "segment_index": segment_index,
                            "kind": "nonfinite_servo_target",
                            "joint": str(name),
                        }
                    )
                    continue
                if name not in SERVO_JOINT_NAMES or not callable(command_to_actual) or not callable(final_limits):
                    continue
                try:
                    actual_deg = float(command_to_actual(name, command_deg))
                    minimum_deg, maximum_deg = (
                        float(value) for value in final_limits(name)
                    )
                except Exception as exc:
                    errors.append(
                        f"segment {segment_index} {name} target audit failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not (
                    math.isfinite(actual_deg)
                    and math.isfinite(minimum_deg)
                    and math.isfinite(maximum_deg)
                    and minimum_deg < maximum_deg
                ):
                    errors.append(
                        f"segment {segment_index} {name} target audit is non-finite"
                    )
                    continue
                if actual_deg < minimum_deg - 1.0e-9 or actual_deg > maximum_deg + 1.0e-9:
                    violations.append(
                        {
                            "segment_index": segment_index,
                            "kind": "servo_target_outside_safe_envelope",
                            "joint": name,
                            "command_target_deg": command_deg,
                            "actual_target_deg": actual_deg,
                            "safe_actual_min_deg": minimum_deg,
                            "safe_actual_max_deg": maximum_deg,
                        }
                    )

            if set(wheel_requested) != set(WHEEL_JOINT_NAMES) or set(
                wheel_applied
            ) != set(WHEEL_JOINT_NAMES):
                errors.append(
                    f"segment {segment_index} does not expose all four wheel targets"
                )
            for name in WHEEL_JOINT_NAMES:
                if name not in wheel_requested or name not in wheel_applied:
                    continue
                wheel_target_count += 1
                try:
                    requested = float(wheel_requested[name])
                    applied = float(wheel_applied[name])
                except (TypeError, ValueError):
                    requested = applied = float("nan")
                if not math.isfinite(requested) or not math.isfinite(applied):
                    violations.append(
                        {
                            "segment_index": segment_index,
                            "kind": "nonfinite_wheel_target",
                            "joint": name,
                        }
                    )
                    continue
                if math.isfinite(max_wheel_speed):
                    expected_applied = max(
                        -max_wheel_speed,
                        min(max_wheel_speed, requested),
                    )
                    requested_was_clamped = not math.isclose(
                        requested,
                        expected_applied,
                        rel_tol=0.0,
                        abs_tol=WHEEL_TARGET_NUMERIC_TOLERANCE,
                    )
                    if requested_was_clamped:
                        clamped_wheel_targets.append(
                            {
                                "segment_index": segment_index,
                                "joint": name,
                                "requested_target_rad_s": requested,
                                "expected_clamped_target_rad_s": expected_applied,
                                "applied_target_rad_s": applied,
                                "safe_abs_max_rad_s": max_wheel_speed,
                            }
                        )
                    applied_outside = (
                        abs(applied)
                        > max_wheel_speed + WHEEL_TARGET_NUMERIC_TOLERANCE
                    )
                    applied_mismatch = not math.isclose(
                        applied,
                        expected_applied,
                        rel_tol=0.0,
                        abs_tol=WHEEL_TARGET_NUMERIC_TOLERANCE,
                    )
                    if applied_outside or applied_mismatch:
                        violations.append(
                            {
                                "segment_index": segment_index,
                                "kind": (
                                    "wheel_applied_target_outside_safe_envelope"
                                    if applied_outside
                                    else "wheel_applied_target_mismatch"
                                ),
                                "joint": name,
                                "requested_target_rad_s": requested,
                                "expected_clamped_target_rad_s": expected_applied,
                                "applied_target_rad_s": applied,
                                "safe_abs_max_rad_s": max_wheel_speed,
                                "requested_was_clamped": requested_was_clamped,
                            }
                        )

        available = not errors
        unsafe: bool | None = True if violations else False if available else None
        return _jsonable(
            {
                "available": available,
                "source": "production_motion_only_segment_target_limit_audit",
                "unsafe": unsafe,
                "segment_count": len(segments),
                "servo_target_count": servo_target_count,
                "wheel_target_count": wheel_target_count,
                "clamped_wheel_target_count": len(clamped_wheel_targets),
                "clamped_wheel_targets": clamped_wheel_targets,
                "errors": errors,
                "violations": violations,
            }
        )

    def _capture(self, adapter: Any, *, force: bool = False) -> None:
        sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        if not force and sim_time + 1.0e-9 < self.next_sample_sim_time_s:
            return
        period = 1.0 / self.request.telemetry_hz
        self.next_sample_sim_time_s = max(
            self.next_sample_sim_time_s + period,
            sim_time + period,
        )
        pose, root_velocity, root_shape_available = self._root_state(adapter)
        joint_q, joint_qd, joint_shape_available = self._joint_state(adapter)
        centers = self._wheel_centers(adapter)
        obstacle = self._obstacle()
        expected_joint_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
        exact_state_available = bool(
            root_shape_available
            and joint_shape_available
            and len(pose) == 7
            and len(root_velocity) == 6
            and all(name in joint_q for name in expected_joint_names)
            and all(name in joint_qd for name in expected_joint_names)
        )
        finite_values = [
            *pose[:7],
            *root_velocity[:6],
            *(joint_q.get(name, float("nan")) for name in expected_joint_names),
            *(joint_qd.get(name, float("nan")) for name in expected_joint_names),
        ]
        state_finite = bool(
            exact_state_available
            and all(math.isfinite(float(value)) for value in finite_values)
        )
        self.nonfinite_state_detected = self.nonfinite_state_detected or not state_finite

        roll = pitch = yaw = float("nan")
        if len(pose) >= 7 and all(math.isfinite(float(v)) for v in pose[3:7]):
            roll, pitch, yaw = (
                float(value) for value in quat_wxyz_to_rpy(pose[3:7])
            )
            self.peak_roll_rad = max(self.peak_roll_rad, abs(roll))
            self.peak_pitch_rad = max(self.peak_pitch_rad, abs(pitch))

        wheel_classes: dict[str, str] = {}
        face_clearances: dict[str, float] = {}
        top_clearances: dict[str, float] = {}
        relative_s = max(0.0, sim_time - self.attach_sim_time_s)
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
            face_clearances[leg] = float(contact.front_face_clearance_m)
            top_clearances[leg] = float(contact.clearance_over_top_m)
            evidence = self.traversal[leg]
            before_cross = evidence["front_face_crossing_s"] is None
            before_top = not bool(evidence["top_seen"])
            lifted = bool(
                contact.contact_class.value == "AIR"
                or center[2] - WHEEL_RADIUS_M
                > obstacle.bottom_z_m + 0.012
            )
            if before_cross and before_top and lifted:
                evidence["airborne_seen_before_top"] = True
                if evidence["airborne_first_s"] is None:
                    evidence["airborne_first_s"] = relative_s
            if float(contact.front_face_clearance_m) > 0.0 and before_cross:
                evidence["front_face_crossing_s"] = relative_s
                evidence["illegal_drive_up"] = not bool(
                    evidence["airborne_seen_before_top"]
                )
            if contact.contact_class.value == "TOP" and not evidence["top_seen"]:
                evidence["top_seen"] = True
                evidence["top_first_s"] = relative_s

        violation, limit_evidence_available = self._joint_limit_violation(
            adapter, joint_q
        )
        self.joint_limit_evidence_available = bool(
            self.joint_limit_evidence_available and limit_evidence_available
        )
        if violation is True:
            self.joint_limit_violation_detected = True
        geometry_support_count = sum(
            value not in {"AIR", "UNKNOWN"} for value in wheel_classes.values()
        )
        attitude_recoverable = bool(
            math.isfinite(roll)
            and math.isfinite(pitch)
            and abs(roll) < math.radians(70.0)
            and abs(pitch) < math.radians(70.0)
        )
        velocity_stable = bool(
            state_finite
            and max((abs(float(value)) for value in root_velocity[:3]), default=0.0)
            <= 0.02
            and max((abs(float(value)) for value in root_velocity[3:6]), default=0.0)
            <= 0.05
            and max((abs(float(value)) for value in joint_qd.values()), default=0.0)
            <= 0.10
        )
        stability_state = (
            "stable"
            if attitude_recoverable
            and geometry_support_count >= 2
            and velocity_stable
            else "recoverable"
            if attitude_recoverable and geometry_support_count >= 2
            else "fallen"
            if math.isfinite(roll)
            and math.isfinite(pitch)
            and (abs(roll) >= math.radians(85.0) or abs(pitch) >= math.radians(85.0))
            else "unknown"
        )
        status = self._playback_status()
        progress = dict(status.get("progress_detail", {}) or {})
        position = pose[:3] if len(pose) >= 3 else [float("nan")] * 3
        row = {
            "schema_version": TELEMETRY_SCHEMA,
            "source_version": self.request.source_version,
            "sim_time_s": sim_time,
            "sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
            "source_step": progress.get("current_step_index"),
            "command_cursor": progress.get("global_command_index"),
            "segment_cursor": int(status.get("segment_index", 0) or 0),
            "event_cursor": int(status.get("index", 0) or 0),
            "scheduler_state": str(progress.get("playback_state", "") or ""),
            "base_position_m": {
                "x": position[0],
                "y": position[1],
                "z": position[2],
            },
            "base_quaternion_wxyz": pose[3:7],
            # Frozen classifier core-state names.  Keep the compact aliases
            # below as well because they are useful to humans and CSV output.
            "root_pose_w": pose[:7],
            "root_position_w": pose[:3],
            "root_orientation_wxyz": pose[3:7],
            "root_linear_velocity_w": root_velocity[:3],
            "root_angular_velocity_w": root_velocity[3:6],
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "base_yaw_rad": yaw,
            "base_linear_velocity_m_s": root_velocity[:3],
            "base_angular_velocity_rad_s": root_velocity[3:6],
            "joint_q_rad": joint_q,
            "joint_qd_rad_s": joint_qd,
            "measured_joint_position_rad": joint_q,
            "measured_joint_velocity_rad_s": joint_qd,
            "servo_targets_deg": dict(
                getattr(adapter, "servo_applied_command_deg", {}) or {}
            ),
            "wheel_targets_rad_s": dict(getattr(adapter, "wheel_speeds", {}) or {}),
            "command_space_servo_target_deg": dict(
                getattr(adapter, "servo_applied_command_deg", {}) or {}
            ),
            "wheel_command_velocity_rad_s": dict(
                getattr(adapter, "wheel_speeds", {}) or {}
            ),
            "wheel_center_w_m": centers,
            "wheel_contact_classes": wheel_classes,
            "wheel_front_face_clearance_m": face_clearances,
            "wheel_top_clearance_m": top_clearances,
            "wheel_contact_load_n": {leg: None for leg in LEGS},
            "wheel_contact_load_available": False,
            "wheel_contact_confidence": "GEOMETRY_ONLY",
            "filtered_contact_bank_enabled": False,
            "robot_state_finite": state_finite,
            "joint_limit_violation": violation,
            "joint_limit_evidence_available": limit_evidence_available,
            "active_contact_count_available": False,
            "geometry_support_candidate_count": geometry_support_count,
            "recoverability_evidence_source": "GEOMETRY_AND_ATTITUDE_HEURISTIC",
            "stability_state": stability_state,
            "final_velocity_stable": velocity_stable,
        }
        self.rows.append(_jsonable(row))

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
                value for value in (str(video.get("error", "") or ""), detach_error) if value
            )
        self.video = _jsonable(video)
        return self.video

    def _task_inputs(self, *, success: bool, error: str) -> dict[str, Any]:
        status = self._playback_status()
        callback_admission = self._replay_callback_admission()
        final = dict(self.rows[-1] if self.rows else {})
        wheel_classes = dict(final.get("wheel_contact_classes", {}) or {})
        clearances = dict(final.get("wheel_front_face_clearance_m", {}) or {})
        base_position = dict(final.get("base_position_m", {}) or {})
        crossed_wheels = sum(
            type(value) in (int, float) and float(value) > 0.0
            for value in clearances.values()
        )
        front = float(self._obstacle().front_face_x_m)
        base_x = base_position.get("x")
        body_crossed = bool(
            type(base_x) in (int, float)
            and float(base_x) > front
            and crossed_wheels >= 3
        )
        lift_complete = all(
            record.get("airborne_seen_before_top") is True
            and type(record.get("front_face_crossing_s")) in (int, float)
            for record in self.traversal.values()
        )
        any_illegal = any(
            record.get("illegal_drive_up") is True for record in self.traversal.values()
        )
        stability = str(final.get("stability_state", "unknown") or "unknown")
        joint_limit_evidence_available = bool(
            self.rows and self.joint_limit_evidence_available
        )
        geometry_support_count = int(
            final.get("geometry_support_candidate_count", 0) or 0
        )
        final_recoverable = bool(
            body_crossed
            and stability in {"safe", "stable", "recoverable"}
            and geometry_support_count >= 2
        )
        try:
            sent_events = int(status.get("events_sent", -1))
        except (TypeError, ValueError):
            sent_events = -1
        try:
            completed_segments = int(status.get("segment_index", -1))
        except (TypeError, ValueError):
            completed_segments = -1
        scheduler_complete = bool(
            status.get("stop_reason") == "complete"
            and sent_events == self.request.plan_event_count
            and completed_segments == self.request.plan_segment_count
        )
        video_path = str(
            self.video.get("video_path", "")
            or getattr(self.recorder, "video_path", "")
            or ""
        )
        motion_started: bool | None = bool(
            status.get("first_command_applied", False)
        )
        targets_applied: bool | None = motion_started
        if self.unsafe_joint_target_detected is True:
            # The hard-safety gate rejected this plan before the first
            # production boundary; absence of motion is intentional, not an
            # infrastructure readiness failure that should mask the unsafe
            # target verdict.
            motion_started = None
            targets_applied = None
        completed_result = {
            "source_version": self.request.source_version,
            "step_count": self.request.step_count,
            "plan_event_count": self.request.plan_event_count,
            "plan_segment_count": self.request.plan_segment_count,
            "expected_event_count": self.request.plan_event_count,
            "sent_event_count": sent_events,
            "expected_segment_count": self.request.plan_segment_count,
            "completed_segment_count": completed_segments,
            "dispatch_complete": scheduler_complete,
            "scheduler_complete": scheduler_complete,
            "scheduler_stop_reason": str(status.get("stop_reason", "") or self.playback_reason),
            "scheduler_status": status,
            "timed_out": "timed out" in str(error).lower(),
            "simulation_app_stopped": self.simulation_app_stopped,
            "lifecycle": {
                "failed": bool(
                    self.infrastructure_failure
                    or self.unsafe_joint_target_detected is None
                ),
                "failure_kind": (
                    "INFRASTRUCTURE"
                    if self.infrastructure_failure
                    else "PLAN_TARGET_EVIDENCE_UNAVAILABLE"
                    if self.unsafe_joint_target_detected is None
                    else ""
                ),
            },
            "motion_start_ready": motion_started,
            "actuator_targets_applied": targets_applied,
            "nonfinite_core_state_detected": self.nonfinite_state_detected,
            "plan_target_audit": self.plan_target_audit,
            "telemetry_callback_status": {
                **callback_admission,
                "event_count_complete": callback_admission["valid"],
                "command_callback_count": self.command_callback_count,
                "last_replay_event": self.last_replay_event,
                "last_command_callback": self.last_command_callback,
            },
            "body_crossed_front_face": body_crossed,
            "final_recoverable": final_recoverable,
            "maximum_abs_roll_rad": self.peak_roll_rad,
            "maximum_abs_pitch_rad": self.peak_pitch_rad,
            "video_path": video_path,
            "video": self.video,
            "video_valid": bool(self.video.get("valid") is True),
            "video_full_decode_valid": bool(
                dict(self.video.get("full_decode", {}) or {}).get("valid") is True
                and self.video.get("full_decode_all_frames") is True
            ),
            "manual_video_reviewed": False,
            "second_simulator_process_detected": None,
            "single_simulator_preflight_available": False,
        }
        if self.recorder is not None:
            completed_result["artifact_valid"] = bool(
                self.video.get("valid") is True
            )
        criteria: dict[str, Any] = {
            "no_illegal_drive_up": {"passed": not any_illegal},
        }
        if stability in {"safe", "stable", "recoverable"}:
            criteria["attitude_safe"] = {"passed": True}
        elif stability == "fallen":
            criteria["attitude_safe"] = {"passed": False}
        robot_fell = (
            True
            if stability == "fallen"
            else False
            if stability in {"safe", "stable", "recoverable"}
            else None
        )
        if joint_limit_evidence_available:
            criteria["joint_limits_safe"] = {
                "passed": not self.joint_limit_violation_detected
            }
        physical_evidence = {
            "source_version": self.request.source_version,
            "body_crossed_front_face": body_crossed,
            "required_leg_lift_completed": lift_complete,
            "final_recoverable": final_recoverable,
            "robot_fell": robot_fell,
            "body_stuck": False if body_crossed else None,
            "wheel_drive_up_without_required_lift": any_illegal,
            "dangerous_collision": None,
            "dangerous_collision_count": None,
            "dangerous_collision_available": False,
            "joint_limit_violation": self.joint_limit_violation_detected
            if joint_limit_evidence_available
            else None,
            "joint_limit_violation_count": int(self.joint_limit_violation_detected)
            if joint_limit_evidence_available
            else None,
            "joint_limit_evidence_available": (
                joint_limit_evidence_available
            ),
            "nonfinite_core_state_detected": self.nonfinite_state_detected,
            "unsafe_joint_target": self.unsafe_joint_target_detected,
            "unsafe_joint_target_evidence_available": bool(
                self.plan_target_audit.get("available") is True
            ),
            "unsafe_joint_target_validation_source": (
                str(self.plan_target_audit.get("source", "") or "UNAVAILABLE")
            ),
            "plan_target_audit": self.plan_target_audit,
            "active_leg_trapped": False
            if scheduler_complete and lift_complete
            else None,
            "active_leg_trapped_validation_source": (
                "four_leg_airborne_before_crossing_sequence"
                if scheduler_complete and lift_complete
                else "UNAVAILABLE"
            ),
            "severe_penetration": None,
            "penetration_evidence_available": False,
            "final_wheel_contact_classes": wheel_classes,
            "final_all_top": bool(
                len(wheel_classes) == len(LEGS)
                and all(value == "TOP" for value in wheel_classes.values())
            ),
            "final_all_loaded": None,
            "final_load_available": False,
            "final_velocity_stable": self._final_velocity_stable(final),
            "final_posture_complete": bool(
                len(wheel_classes) == len(LEGS)
                and all(value == "TOP" for value in wheel_classes.values())
                and stability in {"safe", "stable"}
            ),
            "maximum_abs_roll_rad": self.peak_roll_rad,
            "maximum_abs_pitch_rad": self.peak_pitch_rad,
            "traversal": {
                "legs": self.traversal,
                "any_illegal_drive_up": any_illegal,
            },
            "criteria": criteria,
            "observation_scope": {
                "contact_load_available": False,
                "wheel_classification": "GEOMETRY_ONLY",
                "filtered_contact_bank_enabled": False,
                "strict_rest_blocking": False,
                "contact_drift_blocking": False,
                "final_all_top_blocking": False,
                "recoverability_evidence_source": (
                    "GEOMETRY_AND_ATTITUDE_HEURISTIC"
                ),
            },
        }
        final.update(
            {
                "source_version": self.request.source_version,
                "body_crossed_front_face": body_crossed,
                "required_leg_lift_completed": lift_complete,
                "final_recoverable": final_recoverable,
                "robot_fell": robot_fell,
                "body_stuck": False if body_crossed else None,
                "wheel_drive_up_without_required_lift": any_illegal,
                "dangerous_collision": None,
                "dangerous_collision_available": False,
                "joint_limit_violation": self.joint_limit_violation_detected
                if joint_limit_evidence_available
                else None,
                "joint_limit_evidence_available": (
                    joint_limit_evidence_available
                ),
                "nonfinite_core_state_detected": self.nonfinite_state_detected,
                "unsafe_joint_target": self.unsafe_joint_target_detected,
                "unsafe_joint_target_evidence_available": bool(
                    self.plan_target_audit.get("available") is True
                ),
                "unsafe_joint_target_validation_source": (
                    str(
                        self.plan_target_audit.get("source", "")
                        or "UNAVAILABLE"
                    )
                ),
                "plan_target_audit": self.plan_target_audit,
                "active_leg_trapped": False
                if scheduler_complete and lift_complete
                else None,
                "active_leg_trapped_validation_source": (
                    "four_leg_airborne_before_crossing_sequence"
                    if scheduler_complete and lift_complete
                    else "UNAVAILABLE"
                ),
                "severe_penetration": None,
                "penetration_evidence_available": False,
                "final_wheel_contact_classes": wheel_classes,
            }
        )
        return _jsonable(
            {
                "schema_version": TASK_INPUTS_SCHEMA,
                "completed_result": completed_result,
                "physical_evidence": physical_evidence,
                "final_telemetry_row": final,
            }
        )

    @staticmethod
    def _final_velocity_stable(final: Mapping[str, Any]) -> bool | None:
        linear = list(final.get("base_linear_velocity_m_s", []) or [])
        angular = list(final.get("base_angular_velocity_rad_s", []) or [])
        joint_qd = dict(final.get("joint_qd_rad_s", {}) or {})
        values = [*linear, *angular, *joint_qd.values()]
        if not values or any(
            type(value) not in (int, float) or not math.isfinite(float(value))
            for value in values
        ):
            return None
        return bool(
            max((abs(float(value)) for value in linear), default=0.0) <= 0.02
            and max((abs(float(value)) for value in angular), default=0.0) <= 0.05
            and max((abs(float(value)) for value in joint_qd.values()), default=0.0)
            <= 0.10
        )

    def _finalize(self, *, success: bool, error: str) -> dict[str, Any]:
        if self.terminal_payload is not None:
            return dict(self.terminal_payload)
        if self.adapter is not None:
            self._capture(self.adapter, force=True)
        if self.telemetry_attached:
            if getattr(self.adapter, "telemetry_collector", None) is self:
                self.adapter.attach_telemetry(None)
            self.telemetry_attached = False
        video = self._detach_and_finalize_video()
        video_ok = bool(video.get("valid") is True)
        if success and self.nonfinite_state_detected:
            success = False
            error = "non-finite robot state observed during task replay"
        callback_admission = self._replay_callback_admission()
        if success and not callback_admission["valid"]:
            success = False
            self.infrastructure_failure = True
            error = "production telemetry callback admission failed: " + "; ".join(
                callback_admission["errors"]
            )
        if success and not video_ok:
            success = False
            self.infrastructure_failure = True
            error = "viewport video finalization failed: " + str(
                video.get("error", "invalid viewport video") or "invalid viewport video"
            )
        self.error = str(error or "")
        self.state = "complete" if success else "failed"

        _write_jsonl(self.request.run_dir / "minimal_telemetry.jsonl", self.rows)
        _write_telemetry_csv(self.request.run_dir / "minimal_telemetry.csv", self.rows)
        inputs = self._task_inputs(success=success, error=self.error)
        inputs_path = self.request.run_dir / "task_inputs.json"
        _atomic_write_json(inputs_path, inputs)
        worker_result = {
            "schema_version": SESSION_SCHEMA,
            "execution_mode": "normal_development",
            "source_version": self.request.source_version,
            "request_id": self.request.request_id,
            "plan_id": self.request.plan_id,
            "run_dir": str(self.request.run_dir),
            "task_inputs_path": str(inputs_path),
            "task_replay_complete": success,
            "sample_count": len(self.rows),
            "video": video,
            "video_writer_quiesced": self.video_writer_quiesced,
            "filtered_contact_bank_enabled": False,
            "error": self.error,
        }
        result_path = self.request.run_dir / "worker_task_replay_result.json"
        _atomic_write_json(result_path, worker_result)
        self.terminal_payload = {
            "type": "task_replay_complete" if success else "task_replay_failed",
            "operation": "task_replay",
            "phase": "TASK_REPLAY_COMPLETE" if success else "TASK_REPLAY_FAILED",
            "accepted": success,
            "task_replay_complete": success,
            "request_id": self.request.request_id,
            "plan_id": self.request.plan_id,
            "source_version": self.request.source_version,
            "run_dir": str(self.request.run_dir),
            "task_inputs_path": str(inputs_path),
            "worker_result_path": str(result_path),
            "video_path": str(
                video.get("video_path", "")
                or getattr(self.recorder, "video_path", "")
                or ""
            ),
            "video_writer_quiesced": self.video_writer_quiesced,
            "filtered_contact_bank_enabled": False,
            "error": self.error,
            "task_inputs": inputs,
        }
        return dict(self.terminal_payload)


__all__ = [
    "DEFAULT_POST_SETTLE_S",
    "DEFAULT_TELEMETRY_HZ",
    "DEFAULT_VIDEO_FPS",
    "REQUEST_SCHEMA",
    "WorkerTaskReplayRequest",
    "WorkerTaskReplaySession",
    "load_worker_task_replay_request",
    "validate_worker_task_plan_binding",
]
