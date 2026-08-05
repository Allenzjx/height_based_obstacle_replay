"""Real-robot-style UI controller backed by Isaac Sim commands."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shlex
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from command_model import (
    HIP_LIMIT_DEG,
    KNEE_JOINT_NAMES,
    KNEE_LIMIT_DEG,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_NAME_TO_SHORT,
    WHEEL_SHORT_NAMES,
    CommandMessage,
    split_semicolon_commands,
    validate_motion_command,
)
from height_manifest import (
    SUPPORTED_HEIGHTS_MM,
    legacy_cm_to_mm,
    normalize_height_mm,
    obstacle_height_m_mm,
)
from height_version_store import HeightVersionStore
from motion_speed import load_motion_reference
from operation_coordinator import OperationCoordinator, OperationState
from playback import PlaybackManager, plan_from_steps
from playback_availability import PlaybackAvailability, evaluate_playback_availability
from sequence_manager import SequenceManager
from sequence_model import (
    apply_command_to_state,
    clone_command_state,
    coalesce_record_events,
    format_step_json,
    make_event,
    make_step,
    normalize_step,
    step_summary,
)
from sim_obstacle_scene import (
    OBSTACLE_FRONT_FACE_X_M,
    OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M,
    SimSceneConfig,
)
from sim_process_client import SimProcessClient, run_launch_preflight_for_args
from sim_robot_adapter import NullSimRobotAdapter
from sim_transport import SimTransport
from robot_ground_diagnostics import (
    GROUND_STATE_FAIL,
    GROUND_STATE_PASS,
    GROUND_STATE_PASS_WITH_VISUAL_WARNING,
    GROUND_STATE_UNVERIFIED,
    ground_state_from_diagnostics,
    motion_status_from_worker_status,
    respawn_status_from_worker_status,
)
from recording_baseline import load_baseline, validate_recording_baseline
from playback_progress import PlaybackState


MODE_TEST = "TEST"
MODE_RECORDING_STEP = "RECORDING_STEP"
MODE_PENDING_RECORDED_STEP = "PENDING_RECORDED_STEP"
MODE_REPLACE_STEP_READY = "REPLACE_STEP_READY"
MODE_REPLACING_STEP = "REPLACING_STEP"
MODE_PENDING_REPLACEMENT = "PENDING_REPLACEMENT"
MODE_PLAYBACK = "PLAYBACK"
MODE_PLAYBACK_PAUSED = "PLAYBACK_PAUSED"
MODE_E_STOP = "E_STOP"


class DirtyHeightSwitchError(RuntimeError):
    pass


def _yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def _file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def set_text_preserving_view(widget: Any, text: str, *, follow_bottom: bool = False) -> None:
    now = time.monotonic()
    try:
        old_yview = tuple(float(value) for value in widget.yview())
    except Exception:
        old_yview = (0.0, 1.0)
    try:
        insert_index = str(widget.index("insert"))
    except Exception:
        insert_index = "1.0"
    at_bottom = len(old_yview) >= 2 and old_yview[1] >= 0.999
    last_scroll = float(getattr(widget, "_last_user_scroll_at", 0.0) or 0.0)
    dragging = bool(getattr(widget, "_scrollbar_dragging", False))
    user_recently_scrolled = dragging or (last_scroll > 0.0 and now - last_scroll < 2.0)
    state = ""
    try:
        state = str(widget.cget("state"))
    except Exception:
        state = ""
    try:
        widget.configure(state="normal")
    except Exception:
        pass
    try:
        widget.delete("1.0", "end")
        widget.insert("1.0", str(text))
    finally:
        try:
            if follow_bottom and at_bottom and not user_recently_scrolled:
                widget.yview_moveto(1.0)
            else:
                widget.yview_moveto(old_yview[0] if old_yview else 0.0)
        except Exception:
            pass
        try:
            widget.mark_set("insert", insert_index)
        except Exception:
            pass
        try:
            if state:
                widget.configure(state=state)
        except Exception:
            pass


class HeightReplayController:
    """Command controller with real_robot_ui_controller-like behavior."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.no_sim = bool(getattr(args, "no_sim", False))
        self.motion_reference = load_motion_reference()
        self.store = HeightVersionStore(
            getattr(args, "store_root", None) or None,
            robot_asset_path=getattr(args, "robot_usd", None),
        )
        self.recording_baseline = load_baseline()
        self.recording_baseline_status: dict[str, Any] = {}
        self._recording_baseline_detail_pending = False
        self.manifest_revision = 0
        self.refresh_manifest()
        height_mm_arg = getattr(args, "height_mm", None)
        if height_mm_arg is None:
            legacy_height = getattr(args, "height_cm", None)
            height_mm_arg = legacy_cm_to_mm(legacy_height) if legacy_height in {5, 10} else 50
        self._current_height_mm = normalize_height_mm(height_mm_arg)
        self.current_version_id = ""
        self.current_version_metadata: dict[str, Any] = {}
        self.manager = SequenceManager(self._current_steps_path())
        self.adapter: Any = NullSimRobotAdapter()
        self.transport = SimTransport(self.adapter)
        self.sim_client: SimProcessClient | None = None
        self.sim_connection_enabled = True
        self.sim_launch_mode = "disabled" if self.no_sim else str(getattr(args, "sim_launch_mode", "subprocess")).lower()
        if self.sim_launch_mode == "disabled":
            self.no_sim = True
        self.pending_height_mm: int | None = None
        self.pending_height_request_id = ""
        self.pending_height_requested_revision = 0
        self.pending_height_with_respawn = False
        self.sim_ready = False
        self.loaded_sim_height_mm: int | None = None
        self.latest_sim_status: dict[str, Any] = {}
        self.ui_refresh_hz = 0.0
        self._last_ui_refresh_wall = time.monotonic()
        self.mode = MODE_TEST
        self.status = "Initialized. Isaac Sim not started." if not self.no_sim else "Initialized in --no-sim mode."
        self.status_log: list[str] = []
        self.last_preflight_report: dict[str, Any] = {}
        self._warned_gui_headless = False
        self._command_timeout_stopped = False
        self.detail_text = ""
        self.last_exported_step_json: Path | None = None
        self.selected_step_index: int | None = None
        self.last_motion_command = ""
        self.last_wheel_targets: tuple[float, ...] = ()
        self.last_motion_warnings: tuple[str, ...] = ()
        self.last_wheel_stop_request: dict[str, Any] = {}
        self.last_save_report: dict[str, Any] = {}
        self._sequence_valid_cache_revision = -1
        self._sequence_valid_cache = False
        self._visible_rows_cache_revision = -1
        self._visible_rows_cache: list[dict[str, Any]] = []
        self._sequence_summary_cache_key: tuple[Any, ...] | None = None
        self._sequence_summary_cache: dict[str, Any] = {}

        self.record_start_wall_time = 0.0
        self.record_start_sim_time: float | None = None
        self.record_command_state_before: dict[str, Any] | None = None
        self.record_reference_state: dict[str, Any] | None = None
        self.record_sim_state_before: dict[str, Any] | None = None
        self.record_events: list[dict[str, Any]] = []
        self.last_record_coalesce_stats: dict[str, Any] = {}
        self.pending_step: dict[str, Any] | None = None
        self.pending_replacement: dict[str, Any] | None = None
        self.replace_target_index: int | None = None
        self.recording_kind = ""
        self.record_stop_pending: dict[str, Any] | None = None

        self.combine_mode_enabled = False
        self.combine_selected_indices: set[int] = set()
        self.combine_preview_text = ""
        self.allow_combine_conflicts = False
        self.operation = OperationCoordinator()
        self.scene_update_requested_at = 0.0
        self.scene_update_timeout_s = max(1.0, float(getattr(args, "scene_update_timeout_s", 5.0)))
        self.scene_update_sent = False
        interval_from_ms = max(0.0, float(getattr(args, "record_event_min_interval_ms", 50)) / 1000.0)
        record_event_max_hz = max(0.0, float(getattr(args, "record_event_max_hz", 20.0)))
        interval_from_hz = (1.0 / record_event_max_hz) if record_event_max_hz > 0.0 else 0.0
        self.record_event_min_interval_s = max(interval_from_ms, interval_from_hz)
        self.record_max_events_per_step = max(1, int(getattr(args, "record_max_events_per_step", 2000)))
        self.record_coalesce_slider_events = bool(getattr(args, "record_coalesce_slider_events", True))
        self.max_text_widget_chars = max(1000, int(getattr(args, "max_text_widget_chars", 200000)))
        self.restore_step_start_state_before_selected_playback = bool(getattr(args, "restore_step_start_state_before_selected_playback", True))
        self.restore_full_sim_pose_if_available = bool(getattr(args, "restore_full_sim_pose_if_available", True))
        self.fallback_to_command_state_before = bool(getattr(args, "fallback_to_command_state_before", True))
        self.playback_pre_step_settle_s = max(0.0, float(getattr(args, "playback_pre_step_settle_s", 0.30)))
        self.respawn_play_settle_s = max(0.0, float(getattr(args, "respawn_play_settle_s", 0.30)))
        self.servo_wheel_last_launch_time: float | None = None
        self.servo_wheel_staging_active = False
        self.servo_wheel_staged_state = clone_command_state(None)
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_last_launch_status: dict[str, Any] = {}
        self._motion_batch_acks_merged: set[str] = set()
        self.pending_selected_playback: dict[str, Any] | None = None
        self.last_selected_restore_result: dict[str, Any] = {}
        self.selected_fast_click_id = ""

        self.playback = PlaybackManager(self)
        self.playback.set_profile(str(getattr(args, "profile", self.playback.profile) or self.playback.profile))

    @property
    def current_height_mm(self) -> int:
        return self._current_height_mm

    @current_height_mm.setter
    def current_height_mm(self, value: int | float | str) -> None:
        self._current_height_mm = normalize_height_mm(value)

    @property
    def current_height_cm(self) -> int | float:
        """Read-only display compatibility; all authoritative state is integer mm."""

        value = self._current_height_mm / 10.0
        return int(value) if value.is_integer() else value

    @current_height_cm.setter
    def current_height_cm(self, value: int | float | str) -> None:
        numeric = float(value)
        self._current_height_mm = normalize_height_mm(int(round(numeric * 10.0)))

    @property
    def loaded_sim_height_cm(self) -> int | float | None:
        if self.loaded_sim_height_mm is None:
            return None
        value = self.loaded_sim_height_mm / 10.0
        return int(value) if value.is_integer() else value

    @loaded_sim_height_cm.setter
    def loaded_sim_height_cm(self, value: int | float | None) -> None:
        self.loaded_sim_height_mm = None if value is None else int(round(float(value) * 10.0))

    def _current_steps_path(self) -> Path:
        if self.current_version_id and not self.current_version_id.startswith("legacy_"):
            return self.store.version_steps_path(self.current_height_mm, self.current_version_id)
        if self.current_version_id.startswith("legacy_"):
            legacy = self.store.legacy_steps_path(self.current_height_mm)
            if legacy is not None:
                return legacy
        return self.store.height_dir(self.current_height_mm) / "unsaved_accepted_steps.jsonl"

    @property
    def operation_busy(self) -> bool:
        return not self.operation.idle

    @property
    def busy_name(self) -> str:
        return self.operation.reason

    @property
    def sim_connected(self) -> bool:
        if self.no_sim:
            return True
        if self.sim_client is not None:
            return bool(self.sim_connection_enabled and self.sim_client.connected)
        return False

    @property
    def current_sequence_is_valid(self) -> bool:
        if self._sequence_valid_cache_revision == self.manager.revision:
            return self._sequence_valid_cache
        if self.manager.count <= 0:
            self._sequence_valid_cache_revision = self.manager.revision
            self._sequence_valid_cache = False
            return False
        try:
            for index, step in enumerate(self.manager.steps, start=1):
                normalize_step(step, index=index)
        except Exception:
            self._sequence_valid_cache_revision = self.manager.revision
            self._sequence_valid_cache = False
            return False
        self._sequence_valid_cache_revision = self.manager.revision
        self._sequence_valid_cache = True
        return self._sequence_valid_cache

    def playback_availability(self, *, selected_index: int | None = None) -> PlaybackAvailability:
        selected = self.selected_step_index if selected_index is None else selected_index
        availability = evaluate_playback_availability(
            sim_connected=self.sim_connected,
            sim_ready=bool(self.no_sim or self.runtime_ready),
            sequence_valid=self.current_sequence_is_valid,
            sequence_count=self.manager.count,
            selected_step_valid=bool(selected is not None and 1 <= int(selected) <= self.manager.count),
            operation_state=self.operation.state,
            playback_active=bool(self.playback.active or self.playback.start_requested),
            playback_paused=bool(self.playback.paused),
            playback_scheduled=bool(self.playback.active and self.playback.scheduled_start_at > 0.0),
        )
        return availability

    @property
    def max_wheel_speed(self) -> float:
        return float(getattr(self.args, "max_wheel_speed_rad_s", getattr(self.args, "max_wheel_speed", self.motion_reference.wheel_velocity_limit_rad_s)))

    @property
    def default_wheel_speed(self) -> float:
        return float(getattr(self.args, "default_wheel_speed_rad_s", getattr(self.args, "default_wheel_speed", self.max_wheel_speed * 0.25)))

    @property
    def runtime_ready(self) -> bool:
        if self.no_sim:
            return True
        if str(self.latest_sim_status.get("phase", "") or "") == "running" and not str(self.latest_sim_status.get("traceback", "") or ""):
            return True
        return bool(self.latest_sim_status.get("runtime_ready", self.sim_ready))

    def ground_state(self) -> str:
        ground = self.latest_sim_status.get("robot_ground", {})
        return ground_state_from_diagnostics(ground if isinstance(ground, dict) else {})

    def motion_readiness(self) -> tuple[bool, str]:
        if self.no_sim:
            return True, ""
        has_explicit_motion_status = any(
            key in self.latest_sim_status
            for key in (
                "motion_ready",
                "motion_block_reason",
                "ground_state",
                "grounded_reference_valid",
            )
        )
        ground = self.latest_sim_status.get("robot_ground")
        if isinstance(ground, dict) and ground:
            has_explicit_motion_status = True
        if not has_explicit_motion_status:
            return bool(self.runtime_ready), "" if self.runtime_ready else "Isaac runtime is not ready."
        status = {
            **dict(self.latest_sim_status),
            "runtime_ready": self.runtime_ready,
            "ready": self.runtime_ready,
        }
        ready, reason, _ground_state = motion_status_from_worker_status(status)
        explicit = str(self.latest_sim_status.get("motion_block_reason", "") or "")
        return bool(ready), explicit or reason

    @property
    def motion_ready(self) -> bool:
        return self.motion_readiness()[0]

    def manual_motion_readiness(self, *, allow_staging: bool = False) -> tuple[bool, str]:
        """One readiness policy for manual servo and wheel commands."""

        if self.mode == MODE_E_STOP:
            return False, "E-stop is active. Return to TEST mode before commanding motion."
        if not self.sim_connected:
            return False, "Simulation is disconnected."
        if not self.runtime_ready:
            return False, "Simulation runtime is not ready."
        motion_ok, motion_reason = self.motion_readiness()
        if not motion_ok:
            return False, motion_reason or "Robot motion is not ready."
        if self.playback.active or self.playback.start_requested:
            return False, "Playback is active or starting."
        if self.operation.state not in {OperationState.IDLE, OperationState.RECORDING}:
            return False, self.operation.reason or f"Operation {self.operation.state.value} blocks manual motion."
        if self.servo_wheel_staging_active and not allow_staging:
            return False, "Servo-Wheel Mode is staging targets; use Launch or Cancel."
        return True, ""

    def respawn_readiness(self) -> tuple[bool, str]:
        if self.no_sim:
            return True, ""
        has_explicit_respawn_status = any(
            key in self.latest_sim_status
            for key in (
                "respawn_ready",
                "respawn_block_reason",
                "ground_reference_block_reason",
                "grounded_reference_valid",
                "grounded_reference_stable",
                "ground_state",
            )
        )
        ground = self.latest_sim_status.get("robot_ground")
        if isinstance(ground, dict) and ground:
            has_explicit_respawn_status = True
        if not has_explicit_respawn_status:
            return bool(self.runtime_ready), "" if self.runtime_ready else "Isaac runtime is not ready."
        status = {
            **dict(self.latest_sim_status),
            "runtime_ready": self.runtime_ready,
            "ready": self.runtime_ready,
        }
        computed_ready, computed_reason = respawn_status_from_worker_status(status)
        if "respawn_ready" in self.latest_sim_status:
            ready = bool(self.latest_sim_status.get("respawn_ready", False))
            reason = str(self.latest_sim_status.get("respawn_block_reason", self.latest_sim_status.get("ground_reference_block_reason", "")) or "")
            if ready:
                return True, reason
            if computed_ready:
                return True, computed_reason
            return False, reason or computed_reason
        return computed_ready, computed_reason

    @property
    def respawn_ready(self) -> bool:
        return self.respawn_readiness()[0]

    def playback_readiness(self, *, respawn_first: bool = False) -> tuple[bool, str]:
        availability = self.playback_availability()
        allowed = availability.can_respawn_start if respawn_first else availability.can_start
        return bool(allowed), "" if allowed else availability.reason

    @property
    def recording_active(self) -> bool:
        return self.mode in {MODE_RECORDING_STEP, MODE_REPLACING_STEP}

    def can_start_recording(self) -> tuple[bool, str]:
        if not self.no_sim:
            if not self.runtime_ready:
                return False, "Simulation worker is not ready."
            motion_ready, motion_reason = self.motion_readiness()
            if not motion_ready:
                return False, motion_reason or "Robot motion is not ready."
        if self.operation_busy:
            return False, self.busy_name
        playback = self.playback.status_dict()
        if bool(playback.get("active", False)) or bool(playback.get("scheduled", False)):
            return False, "Playback is active."
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard the pending step before recording."
        if self.mode not in {MODE_TEST, MODE_REPLACE_STEP_READY}:
            return False, f"step_record start is only valid in TEST or REPLACE_STEP_READY. Current mode={self.mode}"
        return True, ""

    def validate_recording_baseline(self) -> dict[str, Any]:
        detailed_status = dict(getattr(self.sim_client, "latest_detailed_status", {}) or {}) if self.sim_client is not None else {}
        if not self.no_sim and self.sim_client is not None and not detailed_status:
            self._recording_baseline_detail_pending = True
            self.transport.request_state(detailed=True)
            status = {
                "passed": False,
                "pending": True,
                "mismatches": ["Detailed state requested; validate again after the worker acknowledgment."],
                "baseline_id": self.recording_baseline.get("baseline_version", ""),
            }
            self.recording_baseline_status = status
            self.status = "Detailed state requested for recording baseline validation."
            return status
        status = validate_recording_baseline(
            self.recording_baseline,
            args=self.args,
            worker_status=detailed_status or self.latest_sim_status,
            output_root=self.store.root,
            no_sim=self.no_sim,
        )
        self.recording_baseline_status = status
        self._recording_baseline_detail_pending = False
        self.detail_text = json.dumps(status, indent=2, ensure_ascii=False, default=str)
        if status["passed"]:
            self.status = f"Recording baseline PASS: {status['baseline_id']}"
        else:
            self.status = "Recording baseline FAIL: " + "; ".join(status["mismatches"][:5])
        return status

    def can_accept_recorded_step(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.mode != MODE_PENDING_RECORDED_STEP or self.pending_step is None:
            return False, "No pending recorded step to accept."
        return True, ""

    def can_prepare_replacement(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.mode != MODE_TEST:
            return False, f"Replace is only valid in TEST or SERVO_WHEEL. Current mode={self.mode}"
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before replacement."
        return True, ""

    def can_accept_replacement(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.mode != MODE_PENDING_REPLACEMENT or self.pending_replacement is None or self.replace_target_index is None:
            return False, "No pending replacement to accept."
        return True, ""

    def can_playback(self, *, allow_respawn_first_motion_gate: bool = False) -> tuple[bool, str]:
        availability = self.playback_availability()
        allowed = availability.can_respawn_start if allow_respawn_first_motion_gate else availability.can_start
        return bool(allowed), "" if allowed else availability.reason

    def can_combine(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.mode != MODE_TEST:
            return False, f"Combine is only valid in TEST. Current mode={self.mode}"
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before combine."
        if len(self.combine_selected_indices) < 2:
            return False, "Select at least two accepted steps to combine."
        if not self.combine_selection_is_contiguous():
            return False, "Combine selection must be contiguous. Use Select Contiguous Range."
        return True, ""

    def combine_selection_is_contiguous(self) -> bool:
        selected = sorted(self.combine_selected_indices)
        return len(selected) >= 2 and selected == list(range(selected[0], selected[-1] + 1))

    def can_save(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before saving."
        return True, ""

    def can_switch_height(self, *, discard_dirty: bool = False) -> tuple[bool, str]:
        if self.operation_busy:
            return False, self.busy_name
        if self.recording_active:
            return False, "Stop recording before switching height."
        if self.playback.active or self.playback.start_requested:
            return False, "Stop playback before switching height."
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before switching height."
        if self.manager.dirty and not discard_dirty:
            return False, "Current sequence has unsaved changes. Save or confirm discard before switching height."
        return True, ""

    def active_steps_view(self) -> str:
        return "height"

    def get_visible_steps(self) -> list[dict[str, Any]]:
        return list(self.manager.steps)

    def get_visible_step(self, index: int) -> dict[str, Any]:
        steps = self.get_visible_steps()
        idx = int(index)
        if idx < 1 or idx > len(steps):
            raise IndexError(f"Accepted step index out of range: {idx}")
        return steps[idx - 1]

    def get_visible_steps_source(self) -> str:
        version = self.current_version_id or "no version"
        return f"Height - {self.current_height_mm} mm / {version}"

    def get_visible_steps_path(self) -> str:
        return str(self._current_steps_path())

    def get_visible_steps_summary(self) -> dict[str, Any]:
        path_text = self.get_visible_steps_path()
        metadata = dict(self.current_version_metadata)
        file_exists = bool(self.current_version_id)
        file_size = int(metadata.get("file_size_bytes", 0) or 0)
        mtime = 0.0
        cache_key = (self.manager.revision, path_text, self.current_version_id, metadata.get("accepted_steps_sha256", ""))
        if self._sequence_summary_cache_key == cache_key:
            return self._sequence_summary_cache
        steps = self.get_visible_steps()
        event_count = 0
        duration = 0.0
        for index, step in enumerate(steps, start=1):
            normalized = normalize_step(step, index=index)
            events = normalized.get("events", []) or []
            event_count += len(events)
            duration += float(normalized.get("duration", 0.0) or 0.0)
        sha256 = str(metadata.get("accepted_steps_sha256", "") or "")
        result = {
            "source": self.get_visible_steps_source(),
            "path": path_text,
            "file_exists": file_exists,
            "file_size_bytes": file_size,
            "sha256": sha256,
            "step_count": len(steps),
            "event_count": event_count,
            "total_duration_s": duration,
            "last_modified": mtime,
            "loaded": bool(steps),
            "read_only": bool(metadata.get("read_only", False)),
            "last_error": "",
        }
        self._sequence_summary_cache_key = cache_key
        self._sequence_summary_cache = result
        return result

    def visible_steps_are_read_only(self) -> bool:
        return bool(self.current_version_metadata.get("read_only", False))

    def visible_step_rows(self) -> list[dict[str, Any]]:
        if self._visible_rows_cache_revision == self.manager.revision:
            return self._visible_rows_cache
        rows = []
        for index, step in enumerate(self.get_visible_steps(), start=1):
            rows.append(self._step_to_row(step, index))
        self._visible_rows_cache_revision = self.manager.revision
        self._visible_rows_cache = rows
        return rows

    def _step_to_row(self, step: dict[str, Any], index: int) -> dict[str, Any]:
        normalized = normalize_step(step, index=index)
        return {
            "index": int(index),
            "name": str(normalized.get("name", "")),
            "type": str(normalized.get("type", normalized.get("step_type", ""))),
            "duration": float(normalized.get("duration", 0.0) or 0.0),
            "events_count": len(normalized.get("events", []) or []),
            "note": str(normalized.get("note", "")),
        }

    def _run_operation(self, name: str, func: Any) -> Any:
        started = time.perf_counter()
        self.status = f"{name}..."
        self.manager.last_operation_report = {}
        try:
            result = func()
            return result
        except Exception as exc:
            details = traceback.format_exc()
            self._warn(f"[WARN] {name} failed: {exc}\n{details}")
            return None
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            report = dict(getattr(self.manager, "last_operation_report", {}) or {})
            suffix = ""
            if report:
                manager_ms = report.get("elapsed_ms")
                if manager_ms is not None:
                    suffix = f" manager={report.get('operation', '')} {float(manager_ms):.1f}ms"
            message = f"[PERF] {name} took {elapsed_ms:.1f}ms{suffix}"
            if elapsed_ms > 500.0:
                self._warn("[WARN] " + message)
            else:
                self.status_log.append(message)
                print(message)

    def command_state_summary(self, state: dict[str, Any] | None) -> str:
        normalized = clone_command_state(state)
        servo_parts = [f"{name}={float(value):.1f}" for name, value in sorted(normalized["servos"].items())]
        wheel_parts = []
        for name, value in sorted(normalized["wheels"].items()):
            short = WHEEL_NAME_TO_SHORT.get(name, name)
            wheel_parts.append(f"{short}={float(value):.2f}")
        return "servos: " + ", ".join(servo_parts) + "\n" + "wheels: " + ", ".join(wheel_parts)

    def capture_current_sim_state(self) -> dict[str, Any]:
        state = self.transport.capture_sim_state()
        if not isinstance(state, dict):
            state = {}
        state = dict(state)
        state.setdefault("command_state", self.transport.capture_command_state())
        state["height_mm"] = self.current_height_mm
        state["height_cm"] = self.current_height_cm
        state["sim_time"] = self.latest_sim_status.get("sim_time", state.get("sim_time", 0.0))
        state.setdefault("joint_names", list(self.latest_sim_status.get("robot_joint_names", []) or []))
        return state

    def _current_record_sim_time(self) -> float | None:
        candidates = [
            self.latest_sim_status.get("sim_time"),
            (self.latest_sim_status.get("sim_state") or {}).get("sim_time")
            if isinstance(self.latest_sim_status.get("sim_state"), dict)
            else None,
            (self.transport.last_worker_status.get("sim_state") or {}).get("sim_time")
            if isinstance(getattr(self.transport, "last_worker_status", {}).get("sim_state"), dict)
            else None,
        ]
        for value in candidates:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return parsed
        return None

    def _record_elapsed_actual_s(self) -> tuple[float, str]:
        current_sim_time = self._current_record_sim_time()
        if self.record_start_sim_time is not None and current_sim_time is not None:
            delta = current_sim_time - self.record_start_sim_time
            if delta >= 0.0:
                return delta, "simulation_time"
        return max(0.0, time.monotonic() - self.record_start_wall_time), "monotonic_fallback"

    def _record_reference_time_s(self) -> tuple[float, float, str]:
        actual_s, source = self._record_elapsed_actual_s()
        return actual_s, actual_s, source

    def _manual_reference_state(self) -> dict[str, Any]:
        return clone_command_state(self.transport.capture_command_state())

    def _record_event_states(self, commands: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        before = clone_command_state(self.record_reference_state or self.record_command_state_before)
        after = clone_command_state(before)
        for command in commands:
            apply_command_to_state(after, command)
        self.record_reference_state = clone_command_state(after)
        return before, after

    def _motion_event_metadata(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        batch_id: str = "",
    ) -> dict[str, Any]:
        changed_servos = {
            name: float(after["servos"].get(name, 0.0))
            for name in SERVO_JOINT_NAMES
            if not math.isclose(
                float(before["servos"].get(name, 0.0)),
                float(after["servos"].get(name, 0.0)),
                abs_tol=1.0e-12,
            )
        }
        canonical_wheels = {name: float(after["wheels"].get(name, 0.0)) for name in WHEEL_JOINT_NAMES}
        return {
            "canonical_servo_target_deg": changed_servos,
            "canonical_servo_velocity_deg_s": self.motion_reference.servo_reference_velocity_deg_s,
            "canonical_wheel_velocity_rad_s": canonical_wheels,
            "command_start_sim_time": self._current_record_sim_time(),
            "wheel_active_duration_s": None,
            "batch_id": str(batch_id or ""),
            "actuator_command_semantics": "fixed-100-percent-direct-v1",
        }

    @staticmethod
    def _flat_joint_positions(sim_state: dict[str, Any] | None) -> dict[str, float]:
        state = dict(sim_state or {})
        names = [str(name) for name in (state.get("joint_names") or [])]
        values = state.get("joint_pos")
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        if not names or not isinstance(values, list) or len(values) < len(names):
            return {}
        result: dict[str, float] = {}
        for index, name in enumerate(names):
            try:
                result[name] = float(values[index])
            except (TypeError, ValueError, IndexError):
                continue
        return result

    @staticmethod
    def _root_xy(sim_state: dict[str, Any] | None) -> tuple[float, float] | None:
        values: Any = dict(sim_state or {}).get("root_pose")
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError, IndexError):
            return None

    def _recording_motion_semantics(
        self,
        events: list[dict[str, Any]],
        reference_duration_s: float,
        sim_state_after: dict[str, Any],
    ) -> dict[str, Any]:
        start_positions = self._flat_joint_positions(self.record_sim_state_before)
        end_positions = self._flat_joint_positions(sim_state_after)
        measured_wheel_displacement = {
            name: end_positions[name] - start_positions[name]
            for name in WHEEL_JOINT_NAMES
            if name in start_positions and name in end_positions
        }

        state = clone_command_state(self.record_command_state_before)
        cursor = 0.0
        derived = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        effective_derived = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        wheel_active_duration = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        wheel_velocity_history = {
            name: [float(state["wheels"].get(name, 0.0))] for name in WHEEL_JOINT_NAMES
        }
        effective_wheel_velocity_history = {
            name: [
                max(
                    -self.motion_reference.wheel_velocity_limit_rad_s,
                    min(
                        self.motion_reference.wheel_velocity_limit_rad_s,
                        float(state["wheels"].get(name, 0.0)),
                    ),
                )
            ]
            for name in WHEEL_JOINT_NAMES
        }
        for event in sorted(events, key=lambda row: float(row.get("time", 0.0))):
            event_time = max(cursor, min(reference_duration_s, float(event.get("time", 0.0))))
            interval = event_time - cursor
            for name in WHEEL_JOINT_NAMES:
                velocity = float(state["wheels"].get(name, 0.0))
                derived[name] += velocity * interval
                effective_velocity = max(-self.motion_reference.wheel_velocity_limit_rad_s, min(self.motion_reference.wheel_velocity_limit_rad_s, velocity))
                effective_derived[name] += effective_velocity * interval
                if abs(velocity) > 1.0e-12:
                    wheel_active_duration[name] += interval
            for command in (event.get("expanded_commands") or [event.get("command", "")]):
                apply_command_to_state(state, str(command))
            for name in WHEEL_JOINT_NAMES:
                value = float(state["wheels"].get(name, 0.0))
                if not math.isclose(value, wheel_velocity_history[name][-1], abs_tol=1.0e-12):
                    wheel_velocity_history[name].append(value)
            for name in WHEEL_JOINT_NAMES:
                value = max(
                    -self.motion_reference.wheel_velocity_limit_rad_s,
                    min(self.motion_reference.wheel_velocity_limit_rad_s, float(state["wheels"].get(name, 0.0))),
                )
                if not math.isclose(value, effective_wheel_velocity_history[name][-1], abs_tol=1.0e-12):
                    effective_wheel_velocity_history[name].append(value)
            cursor = event_time
        final_interval = max(0.0, reference_duration_s - cursor)
        for name in WHEEL_JOINT_NAMES:
            velocity = float(state["wheels"].get(name, 0.0))
            derived[name] += velocity * final_interval
            effective_velocity = max(-self.motion_reference.wheel_velocity_limit_rad_s, min(self.motion_reference.wheel_velocity_limit_rad_s, velocity))
            effective_derived[name] += effective_velocity * final_interval
            if abs(velocity) > 1.0e-12:
                wheel_active_duration[name] += final_interval

        actual_joint_state = dict(sim_state_after.get("actual_joint_state", {}) or {})
        target_joint_state = dict(sim_state_after.get("target_joint_state", {}) or {})
        servo_records: list[dict[str, Any]] = []
        for name in SERVO_JOINT_NAMES:
            start_deg = float((self.record_command_state_before or {}).get("servos", {}).get(name, 0.0))
            target_deg = float((self.record_reference_state or {}).get("servos", {}).get(name, start_deg))
            measured_servo_position = dict(actual_joint_state.get("servos", {}).get(name, {}) or {}).get("deg")
            target_actual = dict(target_joint_state.get("servos", {}).get(name, {}) or {}).get("target_actual_deg")
            command_times = [
                float(event.get("time", 0.0))
                for event in events
                if float(dict(event.get("command_state_before", {}) or {}).get("servos", {}).get(name, start_deg))
                != float(dict(event.get("command_state_after", {}) or {}).get("servos", {}).get(name, start_deg))
            ]
            servo_records.append(
                {
                    "joint_name": name,
                    "start_position_deg": start_deg,
                    "target_position_deg": target_deg,
                    "command_simulation_time_s": command_times[-1] if command_times else 0.0,
                    "reference_profile_id": self.recording_baseline["servo_profile"]["profile_id"],
                    "requested_target_deg": target_deg,
                    "measured_final_position_deg": measured_servo_position,
                    "target_error_deg": None if measured_servo_position is None or target_actual is None else float(measured_servo_position) - float(target_actual),
                    "completion_time_s": reference_duration_s,
                }
            )
        wheel_records = [
            {
                "wheel_joint_name": name,
                "requested_velocity_rad_s": next((value for value in reversed(wheel_velocity_history[name]) if abs(value) > 1.0e-12), 0.0),
                "requested_velocity_history_rad_s": list(wheel_velocity_history[name]),
                "applied_velocity_target_rad_s": next((value for value in reversed(effective_wheel_velocity_history[name]) if abs(value) > 1.0e-12), 0.0),
                "effective_velocity_history_rad_s": list(effective_wheel_velocity_history[name]),
                "start_simulation_time_s": 0.0,
                "stop_simulation_time_s": reference_duration_s,
                "active_duration_s": wheel_active_duration[name],
                "signed_joint_displacement_rad": float((measured_wheel_displacement or effective_derived).get(name, 0.0)),
                "measured_average_velocity_rad_s": (
                    float((measured_wheel_displacement or effective_derived).get(name, 0.0)) / wheel_active_duration[name]
                    if wheel_active_duration[name] > 0.0
                    else 0.0
                ),
                "final_stop_command": "wheel stop",
                "data_source": "articulation_joint_state" if measured_wheel_displacement else "effective_command_integration",
            }
            for name in WHEEL_JOINT_NAMES
        ]

        start_xy = self._root_xy(self.record_sim_state_before)
        end_xy = self._root_xy(sim_state_after)
        body_displacement_m = (
            math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
            if start_xy is not None and end_xy is not None
            else None
        )
        radius = self.motion_reference.wheel_radius_m
        rolling_path = {
            name: (None if radius is None else float(effective_derived[name]) * float(radius))
            for name in WHEEL_JOINT_NAMES
        }

        return {
            "actuator_command_semantics": "fixed-100-percent-direct-v1",
            "reference_duration_s": reference_duration_s,
            "actual_recording_duration_s": reference_duration_s,
            "wheel_active_duration_clock": "actual_recording_time",
            "servo_start_deg": dict((self.record_command_state_before or {}).get("servos", {})),
            "servo_target_deg": dict((self.record_reference_state or {}).get("servos", {})),
            "wheel_angular_displacement_rad": measured_wheel_displacement or effective_derived,
            "wheel_displacement_source": "articulation_joint_state" if measured_wheel_displacement else "command_velocity_x_simulation_duration",
            "canonical_wheel_angular_displacement_rad": derived,
            "derived_wheel_angular_displacement_rad": effective_derived,
            "theoretical_rolling_path_m": rolling_path,
            "robot_body_displacement_m": body_displacement_m,
            "slip_ratio": None,
            "slip_verification": "UNVERIFIED: real wheel radius/transmission are unavailable" if radius is None else "derived when body displacement is available",
            "servo_records": servo_records,
            "wheel_records": wheel_records,
        }

    def compact_step_details(self, step: dict[str, Any], *, title: str | None = None) -> str:
        normalized = normalize_step(step)
        events = normalized.get("events", [])
        heading = title or f"Step {int(normalized['index']):03d}"
        height = normalized.get("height_cm", self.current_height_cm)
        lines = [
            heading,
            f"name={normalized['name']}",
            f"type={normalized['type']} duration={float(normalized['duration']):.3f}s events={len(events)} height={height}cm",
            f"note={normalized.get('note', '')}",
            "",
            "command_state_after:",
            self.command_state_summary(normalized.get("command_state_after")),
        ]
        coalesce = normalized.get("record_coalesce")
        if isinstance(coalesce, dict) and coalesce.get("coalesced"):
            lines.extend(
                [
                    "",
                    "recording event coalescing:",
                    f"{coalesce.get('original_count')} -> {coalesce.get('coalesced_count')} events, dropped={coalesce.get('dropped_count', 0)}",
                ]
            )
        return "\n".join(lines)

    def step_json_text(self, step: dict[str, Any], *, max_chars: int | None = None) -> str:
        limit = self.max_text_widget_chars if max_chars is None else max(1000, int(max_chars))
        text = format_step_json(step)
        if len(text) <= limit:
            return text
        return (
            f"[TRUNCATED] Step JSON is {len(text)} chars; showing first {limit} chars. "
            "Use Export Step JSON for the full file.\n\n"
            + text[:limit]
        )

    def set_step_summary_detail(self, step: dict[str, Any], *, title: str | None = None) -> None:
        self.detail_text = self.compact_step_details(step, title=title)

    def export_step_json(self, index: int) -> Path | None:
        try:
            step = normalize_step(self.get_visible_step(index), index=index)
        except Exception as exc:
            self._warn(f"[WARN] Could not export step JSON: {exc}")
            return None
        path = self.store.height_dir(self.current_height_mm) / f"step_{int(index):03d}_full.json"
        path.write_text(format_step_json(step) + "\n", encoding="utf-8")
        self.last_exported_step_json = path
        self._info(f"[INFO] Exported full step JSON: {path}")
        return path

    def refresh_manifest(self) -> None:
        self.store.refresh_manifest()
        self.manifest_revision += 1

    def start_sim_if_needed(self) -> None:
        if self.no_sim:
            self.sim_ready = False
            self.transport.attach(self.adapter, ready=False)
            self._info("[INFO] --no-sim mode: Isaac Sim startup skipped.")
            return
        if self.sim_launch_mode == "subprocess":
            if self.sim_client is None:
                self._info("[INFO] Starting Isaac Sim subprocess worker...")
                self.sim_client = SimProcessClient(self.args)
                self.sim_connection_enabled = True
                self.transport.attach_process_client(self.sim_client)
                self.sim_client.start()
                self.status = "Isaac Sim subprocess worker launched; waiting for status."
            return
        if self.sim_launch_mode == "main":
            self._warn("[WARN] --sim-launch-mode main is not used with the Tk UI. Use subprocess or --auto-play.")
            return

    def run_isaac_preflight(self) -> dict[str, Any]:
        if self.no_sim:
            report = {"preflight_ok": False, "preflight_error": "--no-sim mode: preflight skipped."}
        else:
            report = run_launch_preflight_for_args(self.args)
        self.last_preflight_report = dict(report)
        self.latest_sim_status = {
            **self.latest_sim_status,
            "phase": "preflight_completed",
            "preflight": report,
            "preflight_ok": bool(report.get("preflight_ok", False)),
            "preflight_error": str(report.get("preflight_error", "") or ""),
            "candidate_reports": report.get("candidate_reports", []),
        }
        if report.get("preflight_ok"):
            selected = report.get("selected_report", {}) or {}
            self.status = f"Isaac preflight OK: {selected.get('executable', selected.get('candidate_path', ''))}"
            self._info("[INFO] " + self.status)
        else:
            self.status = f"[WARN] Isaac preflight failed: {report.get('preflight_error', '')}"
            self._warn(self.status)
        return report

    def restart_sim_worker(self) -> None:
        if self.no_sim:
            self._info("[INFO] --no-sim mode: restart skipped.")
            return
        if self.sim_launch_mode != "subprocess":
            self._warn("[WARN] Restart Isaac Worker is only available in subprocess mode.")
            return
        if self.sim_client is None:
            self.start_sim_if_needed()
            return
        self._info("[INFO] Restarting Isaac subprocess worker...")
        self.sim_client.restart()
        self.status = "Isaac subprocess worker restart requested."

    def connect_to_sim_worker(self) -> bool:
        if self.sim_client is None:
            self._warn("[WARN] No simulation worker process exists. Use Run Manager > Start Sim Worker first.")
            return False
        self.sim_connection_enabled = True
        self.transport.attach_process_client(self.sim_client)
        self._info("[INFO] Simulation worker connection enabled.")
        return True

    def disconnect_from_sim_worker(self) -> None:
        self.playback.stop(silent=True, reason="simulation_disconnected")
        self.stop_wheels(reason="simulation_disconnect")
        self.sim_connection_enabled = False
        self.sim_ready = False
        self.transport.attach(self.adapter, ready=False)
        self._info("[INFO] Disconnected UI command transport from the simulation worker; process remains managed by Run Manager.")

    def stop_sim_worker(self) -> None:
        self.stop_wheels(reason="worker_shutdown")
        if self.sim_client is not None:
            self._info("[INFO] Stopping Isaac subprocess worker...")
            self.sim_client.shutdown()
            self.sim_client = None
        self.sim_ready = False
        self.sim_connection_enabled = False
        self.transport.attach(self.adapter, ready=False)

    def open_worker_log_folder(self) -> str:
        if self.sim_client is None:
            self._warn("[WARN] Isaac subprocess worker has not been started yet.")
            return ""
        return self.sim_client.open_log_folder()

    def copy_worker_display_command(self) -> str:
        if self.sim_client is None:
            return ""
        return self.sim_client.copy_display_command()

    def generate_or_update_height_obstacle(self, *, respawn: bool = False) -> str:
        """Queue one verified USD geometry transaction."""

        target_operation = OperationState.RESPAWNING if respawn else OperationState.SCENE_UPDATE
        detail = f"Updating obstacle to {self.current_height_mm} mm"
        if not self.operation.begin(target_operation, detail=detail):
            self._warn(f"[WARN] Cannot update obstacle: {self.operation.reason}")
            return ""

        self.pending_height_mm = self.current_height_mm
        self.pending_height_request_id = uuid.uuid4().hex
        current_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0)
        self.pending_height_requested_revision = current_revision + 1
        self.pending_height_with_respawn = bool(respawn)
        self.scene_update_requested_at = time.monotonic()
        self.scene_update_sent = False
        if self.no_sim:
            self.loaded_sim_height_mm = self.current_height_mm
            self.status = f"No-sim: current height set to {self.current_height_mm} mm."
            self.pending_height_mm = None
            self.operation.finish(target_operation)
            return self.pending_height_request_id
        try:
            if self.sim_launch_mode == "subprocess":
                if self.sim_client is None:
                    self.start_sim_if_needed()
                if not self.runtime_ready:
                    self.status = f"Queued obstacle height {self.current_height_mm} mm until Isaac worker is ready."
                    self._info(f"[INFO] {self.status}")
                    return self.pending_height_request_id
                if self.sim_client is not None:
                    if respawn:
                        self.sim_client.set_height_respawn(
                            self.current_height_mm,
                            request_id=self.pending_height_request_id,
                            requested_revision=self.pending_height_requested_revision,
                            source="ui",
                        )
                    else:
                        self.sim_client.set_height_mm(
                            self.current_height_mm,
                            request_id=self.pending_height_request_id,
                            requested_revision=self.pending_height_requested_revision,
                            source="ui",
                        )
                    self.scene_update_sent = True
                    action = "Generate + Respawn" if respawn else "Generate / Update"
                    self.status = f"{action} requested for {self.current_height_mm} mm."
                    self._info(f"[INFO] {self.status}")
                return self.pending_height_request_id
        except Exception:
            self.pending_height_mm = None
            self.scene_update_sent = False
            self.operation.finish(target_operation)
            raise
        return self.pending_height_request_id

    def generate_height_and_respawn(self) -> str:
        return self.generate_or_update_height_obstacle(respawn=True)

    def recalibrate_ground_reference(self) -> str:
        request_id = uuid.uuid4().hex
        if self.no_sim:
            self.status = "No-sim: ground recalibration skipped."
            return request_id
        self.transport.recalibrate(request_id=request_id, source="ui")
        self.status = "Ground reference recalibration requested."
        return request_id

    def config_for_height(self, height_mm: int) -> SimSceneConfig:
        return SimSceneConfig(
            obstacle_height_m=obstacle_height_m_mm(height_mm),
            robot_usd=Path(self.args.robot_usd),
            save_usd=Path(self.args.save_usd),
            spawn_z=float(self.args.spawn_z),
            obstacle_x=float(self.args.obstacle_x),
            obstacle_width=self.args.obstacle_width,
            obstacle_length=self.args.obstacle_length,
            infer_obstacle_size=bool(self.args.infer_obstacle_size),
            robot_width=float(self.args.robot_width),
            robot_length=float(self.args.robot_length),
            physics_dt=float(self.args.physics_dt),
            render_interval=int(self.args.render_interval),
            device=str(getattr(self.args, "device", "cuda:0")),
            max_wheel_speed=self.max_wheel_speed,
            default_wheel_speed=self.default_wheel_speed,
            wheel_direction=float(self.args.wheel_direction),
            servo_stiffness=float(self.args.servo_stiffness),
            servo_damping=float(self.args.servo_damping),
            wheel_damping=float(self.args.wheel_damping),
            save_scene=bool(self.args.save_scene),
            telemetry_contact_sensors_enabled=False,
        )

    def set_current_height(
        self,
        value: int | float | str,
        *,
        discard_dirty: bool = False,
        load_steps: bool = True,
        generate_obstacle: bool = True,
    ) -> int:
        try:
            height = normalize_height_mm(value)
        except ValueError:
            height = legacy_cm_to_mm(value)
        if height == self.current_height_mm:
            if load_steps:
                self.load_steps_for_current_height(discard_dirty=discard_dirty)
            if generate_obstacle:
                self.generate_or_update_height_obstacle()
            return height
        ok, reason = self.can_switch_height(discard_dirty=discard_dirty)
        if not ok:
            raise DirtyHeightSwitchError(reason)
        self.current_height_mm = height
        self.args.height_mm = height
        self.current_version_id = ""
        self.current_version_metadata = {}
        self.manager.accepted_path = self._current_steps_path()
        self.manager.adopt_steps([], dirty=False)
        self.pending_step = None
        self.pending_replacement = None
        self.record_events = []
        self.record_command_state_before = None
        self.selected_step_index = None
        self.detail_text = ""
        self.mode = MODE_TEST
        loaded_count: int | None = None
        if load_steps:
            loaded_count = self.load_steps_for_current_height()
        if generate_obstacle:
            self.generate_or_update_height_obstacle()
        if not load_steps or bool(loaded_count):
            self.status = f"Current height={height} mm."
        return height

    def load_steps_for_current_height(self, *, discard_dirty: bool = False, version_id: str | None = None) -> int:
        if self.manager.dirty and not discard_dirty:
            raise DirtyHeightSwitchError("Current sequence has unsaved changes. Save or confirm discard before reloading.")

        def work() -> int:
            selected = str(version_id or self.current_version_id or self.store.active_version_id(self.current_height_mm) or "")
            if not selected:
                legacy = self.store.list_versions(self.current_height_mm, include_legacy=True)
                selected = str(legacy[0].get("version_id", "")) if legacy else ""
            if not selected:
                self.current_version_id = ""
                self.current_version_metadata = {}
                self.manager.accepted_path = self._current_steps_path()
                self.manager.adopt_steps([], dirty=False)
                self.selected_step_index = None
                self.detail_text = ""
                self._info(f"[INFO] No version exists for {self.current_height_mm} mm; ready for a new recording.")
                return 0
            try:
                steps, metadata = self.store.load_version(self.current_height_mm, selected)
            except Exception as exc:
                self.manager.accepted_path = self._current_steps_path()
                self.manager.adopt_steps([], dirty=False)
                self.selected_step_index = None
                self.detail_text = ""
                self._warn(f"[WARN] Could not load version {selected} for {self.current_height_mm} mm: {exc}")
                return 0
            self.current_version_id = selected
            self.current_version_metadata = dict(metadata)
            path = Path(str(metadata.get("accepted_steps_path", self._current_steps_path())))
            self.manager.accepted_path = path
            self.manager.adopt_steps(steps, dirty=False)
            self.selected_step_index = 1 if steps else None
            self.set_step_summary_detail(steps[0], title=f"Loaded step {int(steps[0].get('index', 1)):03d}") if steps else setattr(self, "detail_text", "")
            if steps:
                self._info(f"[INFO] Loaded {len(steps)} steps from {selected} at {self.current_height_mm} mm.")
            else:
                self._info(f"[INFO] Loaded empty version {selected} at {self.current_height_mm} mm.")
            self.manifest_revision += 1
            return len(steps)

        result = self._run_operation("Load steps for current height", work)
        return int(result or 0)

    def save_steps_for_current_height(
        self,
        *,
        allow_empty: bool = False,
        version_name: str = "manual",
        note: str = "",
    ) -> Path | None:
        """Compatibility entry point: Save always means Save New Version."""

        version_dir = self.save_new_version(allow_empty=allow_empty, version_name=version_name, note=note)
        return None if version_dir is None else version_dir / "accepted_steps.jsonl"

    def save_new_version(
        self,
        *,
        allow_empty: bool = False,
        version_name: str = "manual",
        note: str = "",
    ) -> Path | None:
        ok, reason = self.can_save()
        if not ok:
            self._warn("[WARN] " + reason)
            return None

        def work() -> Path | None:
            if self.manager.count <= 0 and not allow_empty:
                self._warn(f"[WARN] Current {self.current_height_mm} mm sequence has no accepted steps to save.")
                return None
            parent = self.current_version_id
            path = self.store.save_new_version(
                self.current_height_mm,
                self.manager.steps,
                version_name=version_name,
                note=note,
                parent_version_id=parent,
                metadata={
                    "actuator_baseline_id": self.recording_baseline.get("baseline_version", ""),
                    "environment_baseline_id": "fixed-wide-obstacle-v2",
                    "motion_profile_id": self.motion_reference.profile_id,
                    "motion_profile_mode": "fixed_100_percent",
                    "obstacle_width_m": OBSTACLE_WIDTH_M,
                    "obstacle_length_m": OBSTACLE_LENGTH_M,
                    "obstacle_front_face_x_m": OBSTACLE_FRONT_FACE_X_M,
                    "robot_respawn_pose": self.latest_sim_status.get("grounded_respawn_root_pose"),
                    "actuator_command_semantics": "fixed-100-percent-direct-v1",
                },
            )
            self.current_version_id = str(self.store.last_save_report.get("version_id", ""))
            _steps, self.current_version_metadata = self.store.load_version(self.current_height_mm, self.current_version_id)
            self.manager.accepted_path = self._current_steps_path()
            self.manager.dirty = False
            self.last_save_report = dict(self.store.last_save_report)
            self.manifest_revision += 1
            report = self.last_save_report
            self._info(
                f"[INFO] Saved {report.get('step_count', self.manager.count)} steps / "
                f"{report.get('command_count', 0)} commands at {report.get('saved_at', '')}: {path}"
            )
            return path

        result = self._run_operation("Save New Version", work)
        return result if isinstance(result, Path) else None

    def save_current_version(self, *, confirmed: bool) -> Path | None:
        if not self.current_version_id:
            self._warn("[WARN] No current version exists; use Save New Version.")
            return None
        try:
            path = self.store.save_current_version(
                self.current_height_mm,
                self.current_version_id,
                self.manager.steps,
                confirmed=confirmed,
            )
            _steps, self.current_version_metadata = self.store.load_version(self.current_height_mm, self.current_version_id)
            self.manager.accepted_path = self._current_steps_path()
            self.manager.dirty = False
            self.last_save_report = dict(self.store.last_save_report)
            self.manifest_revision += 1
            return path
        except Exception as exc:
            self._warn(f"[WARN] Save Current Version failed: {exc}")
            return None

    def new_empty_sequence_for_current_height(self, *, discard_dirty: bool = False) -> None:
        if self.manager.dirty and not discard_dirty:
            raise DirtyHeightSwitchError("Current sequence has unsaved changes. Confirm discard before creating a new empty sequence.")
        self.current_version_id = ""
        self.current_version_metadata = {}
        self.manager.accepted_path = self._current_steps_path()
        self.manager.adopt_steps([], dirty=False)
        self.pending_step = None
        self.pending_replacement = None
        self.record_events = []
        self.selected_step_index = None
        self.detail_text = ""
        self.mode = MODE_TEST
        self._info(f"[INFO] New empty sequence for {self.current_height_mm} mm.")

    def handle_command(self, message: CommandMessage | str) -> None:
        text = message.text if isinstance(message, CommandMessage) else str(message)
        source = message.source if isinstance(message, CommandMessage) else "ui"
        command_message = message if isinstance(message, CommandMessage) else None
        for command in split_semicolon_commands(text):
            try:
                tokens = command.split()
                if not tokens:
                    continue
                self._handle_tokens(tokens, command, source, command_message)
            except Exception as exc:
                failed_operation = self.operation.state
                self.operation.finish()
                self.stop_wheels(reason="command_error")
                if failed_operation is OperationState.RECORDING:
                    self.mode = MODE_TEST
                    self.record_events = []
                    self.record_command_state_before = None
                    self.record_sim_state_before = None
                self._warn(f"[WARN] Command failed: {command}: {exc}\n{traceback.format_exc()}")

    def stop_wheels(self, *, reason: str = "controller_stop") -> dict[str, Any]:
        result = self.transport.stop_wheels(reason=reason)
        if self.servo_wheel_staging_active:
            self.servo_wheel_staged_state["wheels"] = {name: 0.0 for name in WHEEL_JOINT_NAMES}
            self.servo_wheel_staged_dirty = True
        self.last_wheel_stop_request = dict(result)
        self.status = "Stopping wheels..."
        return result

    def _merge_motion_batch_ack(self, ack: dict[str, Any]) -> None:
        batch_id = str(ack.get("batch_id", "") or "")
        if not batch_id or batch_id in self._motion_batch_acks_merged:
            return
        self._motion_batch_acks_merged.add(batch_id)
        if batch_id == str(self.servo_wheel_last_launch_status.get("batch_id", "") or ""):
            self.servo_wheel_last_launch_status = {
                "state": "error" if str(ack.get("error", "") or "") else "applied",
                "batch_id": batch_id,
                "applied_sim_time": ack.get("applied_sim_time", ack.get("batch_applied_sim_time")),
                "applied_sim_step": ack.get("applied_sim_step", ack.get("first_physics_step")),
                "servo_targets_applied": dict(ack.get("servo_targets_applied", ack.get("canonical_servo_targets_deg", {})) or {}),
                "wheel_targets_applied": dict(ack.get("wheel_targets_applied", ack.get("canonical_wheel_velocity_rad_s", {})) or {}),
                "motion_start_skew_s": ack.get("motion_start_skew_s"),
                "error": str(ack.get("error", "") or ""),
            }
        for event in reversed(self.record_events):
            if str(event.get("batch_id", "") or "") != batch_id:
                continue
            event["batch_ack"] = dict(ack)
            event["applied_sim_time"] = ack.get("applied_sim_time", ack.get("batch_applied_sim_time"))
            event["applied_sim_step"] = ack.get("applied_sim_step", ack.get("first_physics_step"))
            break

    def shutdown(self) -> None:
        self._cancel_pending_selected_playback(reason="UI shutdown")
        self.playback.stop(silent=True, stop_wheels=False)
        try:
            self.stop_wheels(reason="ui_close")
        except Exception:
            pass
        if self.sim_client is not None and self.sim_connection_enabled:
            self.sim_client.shutdown()
            self.sim_client = None

    def update(self) -> None:
        self.playback.update()
        if self.playback.active or self.playback.start_requested:
            self.mode = MODE_PLAYBACK_PAUSED if self.playback.paused else MODE_PLAYBACK
        elif self.mode in {MODE_PLAYBACK, MODE_PLAYBACK_PAUSED}:
            self.mode = MODE_TEST
        if self.sim_client is not None:
            self.sim_client.poll()
            status = self.sim_client.status()
            self.latest_sim_status = status
            self.transport.update_worker_status(status)
            last_ack = dict(status.get("last_operation_ack", {}) or {})
            if str(last_ack.get("operation", "") or "") == "apply_motion_batch":
                self._merge_motion_batch_ack(last_ack)
            self.playback.sync_worker_status(
                status.get("worker_playback") if isinstance(status.get("worker_playback"), dict) else None,
                operation_ack=last_ack,
                worker_status_age_s=max(0.0, time.monotonic() - float(getattr(self.sim_client, "last_status_time", time.monotonic()) or time.monotonic())),
            )
            self.sim_ready = bool(status.get("runtime_ready", status.get("ready", False)))
            self._update_pending_selected_playback()
            if status.get("height_mm") is not None:
                self.loaded_sim_height_mm = int(status["height_mm"])
            elif status.get("height_cm") is not None:
                self.loaded_sim_height_cm = float(status["height_cm"])
            error = str(status.get("error", "") or "")
            warning = str(status.get("startup_timeout_warning", "") or status.get("status_timeout_warning", "") or "")
            if error:
                self.status = f"[WARN] Isaac subprocess worker: {error}"
            elif warning:
                self.status = f"[WARN] {warning}"
                if "No Isaac worker status" in warning and not self._command_timeout_stopped:
                    self.stop_wheels(reason="command_timeout")
                    self._command_timeout_stopped = True
            else:
                self._command_timeout_stopped = False
            if (
                self._recording_baseline_detail_pending
                and self.sim_client.latest_detailed_status
            ):
                self.validate_recording_baseline()
            if bool(getattr(self.args, "ui", False)) and bool(status.get("effective_headless", False)) and not self._warned_gui_headless:
                self._warn("[WARN] GUI was requested but AppLauncher resolved to headless mode.")
                self._warned_gui_headless = True
            if self.sim_ready and self.pending_height_mm is not None and not self.scene_update_sent:
                if self.pending_height_with_respawn:
                    self.sim_client.set_height_respawn(
                        self.pending_height_mm,
                        request_id=self.pending_height_request_id,
                        requested_revision=self.pending_height_requested_revision,
                        source="ui",
                    )
                else:
                    self.sim_client.set_height_mm(
                        self.pending_height_mm,
                        request_id=self.pending_height_request_id,
                        requested_revision=self.pending_height_requested_revision,
                        source="ui",
                    )
                self.scene_update_sent = True
        if self.pending_height_mm is not None:
            ack = dict(self.latest_sim_status.get("last_operation_ack", {}) or {})
            ack_matches = (
                str(ack.get("request_id", "") or "") == self.pending_height_request_id
                and str(ack.get("operation", "") or "") in {"set_height", "set_height_respawn"}
            )
            scene_height = self.latest_sim_status.get("scene_height_mm", self.latest_sim_status.get("height_mm"))
            if ack_matches:
                requested = int(self.pending_height_mm)
                measured = float(ack.get("measured_height_mm", -1.0) or -1.0)
                revision = int(ack.get("obstacle_revision", ack.get("revision", 0)) or 0)
                accepted = (
                    bool(ack.get("accepted", False))
                    and bool(ack.get("prim_valid", False))
                    and abs(measured - requested) <= 1.0
                    and bool(ack.get("visual_updated", False))
                    and bool(ack.get("collision_updated", False))
                    and revision >= int(self.pending_height_requested_revision)
                    and bool(ack.get("control_ready", False))
                    and not str(ack.get("error", "") or "")
                )
                elapsed = max(0.0, time.monotonic() - self.scene_update_requested_at)
                error = str(ack.get("error", "") or "")
                self.pending_height_mm = None
                self.scene_update_sent = False
                expected_operation = OperationState.RESPAWNING if self.pending_height_with_respawn else OperationState.SCENE_UPDATE
                self.operation.finish(expected_operation)
                if not accepted:
                    reason = error or (
                        f"geometry verification failed: requested={requested}mm measured={measured:.3f}mm "
                        f"prim_valid={bool(ack.get('prim_valid', False))} visual={bool(ack.get('visual_updated', False))} "
                        f"collision={bool(ack.get('collision_updated', False))} revision={revision}/"
                        f"{self.pending_height_requested_revision} control_ready={bool(ack.get('control_ready', False))}"
                    )
                    self.status = f"ERROR: Obstacle update failed: {reason}"
                    self._warn(f"[ERROR] Obstacle update failed after {elapsed:.3f}s: {reason}")
                else:
                    self.loaded_sim_height_mm = int(round(measured))
                    self._info(
                        f"[INFO] Obstacle geometry verified at {measured:.3f} mm, "
                        f"width={float(ack.get('measured_width_m', 0.0) or 0.0):.3f} m, "
                        f"revision={revision}, in {elapsed:.3f}s; control ready."
                    )
            elif self.scene_update_sent and time.monotonic() - self.scene_update_requested_at > self.scene_update_timeout_s:
                target = self.pending_height_mm
                self.pending_height_mm = None
                self.scene_update_sent = False
                expected_operation = OperationState.RESPAWNING if self.pending_height_with_respawn else OperationState.SCENE_UPDATE
                self.operation.finish(expected_operation)
                self.status = f"ERROR: Obstacle update timed out for {target} mm."
                self._warn(f"[ERROR] Obstacle update timed out for {target} mm; operation lock released.")
        self._try_finalize_record_stop()

    def snapshot(self) -> dict[str, Any]:
        playback = self.playback.status_dict()
        availability = self.playback_availability()
        state = self.transport.capture_command_state()
        display_state = clone_command_state(self.servo_wheel_staged_state if self.servo_wheel_staging_active else state)
        wheels_short = {
            short: display_state["wheels"].get(full, 0.0)
            for short, full in WHEEL_SHORT_NAMES.items()
        }
        visible_rows = self.visible_step_rows()
        sequence_summary = self.get_visible_steps_summary()
        motion_ready, motion_block_reason = self.motion_readiness()
        respawn_ready, respawn_block_reason = self.respawn_readiness()
        manifest_warning = ""
        try:
            manifest_rows = self.store.status_rows()
            manifest_warning = getattr(self.store, "last_status_warning", "")
        except Exception as exc:
            manifest_rows = getattr(self.store, "_last_status_rows", [])
            manifest_warning = f"Could not read height manifest status rows: {exc}"
            if self.status != "[WARN] " + manifest_warning:
                self._warn("[WARN] " + manifest_warning)
        sim_status = dict(self.latest_sim_status)
        wheel_command = dict(sim_status.get("wheel_command", {}) or {})
        if not wheel_command:
            wheel_command = dict(getattr(self.transport.adapter, "wheel_command_status", {}) or {})
        requested_id = str(self.last_wheel_stop_request.get("command_id", "") or "")
        if requested_id and str(wheel_command.get("command_id", "") or "") != requested_id:
            wheel_command = {
                **wheel_command,
                "command_id": requested_id,
                "state": "stopping",
                "stop_command_received": False,
                "zero_target_applied": False,
                "physically_stopped": False,
            }
        motion_profile = dict(sim_status.get("motion_profile", {}) or {}) or {
            "mode": "fixed_100_percent",
            "profile_id": self.motion_reference.profile_id,
            "servo_velocity_deg_s": self.motion_reference.servo_reference_velocity_deg_s,
            "wheel_reference_velocity_rad_s": self.motion_reference.wheel_reference_velocity_rad_s,
            "wheel_velocity_limit_rad_s": self.motion_reference.wheel_velocity_limit_rad_s,
        }
        return {
            "sim": {
                **sim_status,
                "connected": self.sim_connected,
                "ready": self.sim_ready,
                "runtime_ready": self.runtime_ready,
                "motion_ready": motion_ready,
                "motion_block_reason": motion_block_reason,
                "respawn_ready": respawn_ready,
                "respawn_block_reason": respawn_block_reason,
                "ground_state": self.ground_state(),
                "no_sim": self.no_sim,
                "loaded_height_mm": self.loaded_sim_height_mm,
                "loaded_height_cm": self.loaded_sim_height_cm,
                "transport": self.transport.status(),
                "worker_status": sim_status,
                "physics_dt": float(sim_status.get("physics_dt", 0.0) or 0.0),
                "sim_step_hz": float(sim_status.get("sim_step_hz", 0.0) or 0.0),
                "real_time_factor": float(sim_status.get("real_time_factor", 0.0) or 0.0),
                "worker_loop_hz": float(sim_status.get("worker_loop_hz", 0.0) or 0.0),
                "ui_refresh_hz": self.ui_refresh_hz,
                "current_max_wheel_speed_rad_s": float(sim_status.get("current_max_wheel_speed_rad_s", self.max_wheel_speed) or self.max_wheel_speed),
                "default_wheel_speed_rad_s": float(sim_status.get("default_wheel_speed_rad_s", self.default_wheel_speed) or self.default_wheel_speed),
                "phase": sim_status.get("phase", ""),
                "worker_pid": sim_status.get("worker_pid", sim_status.get("pid", "")),
                "worker_returncode": sim_status.get("worker_returncode", ""),
                "ipc_connected": sim_status.get("ipc_connected", sim_status.get("connected", False)),
                "target_joint_state": sim_status.get("target_joint_state"),
                "actual_joint_state": sim_status.get("actual_joint_state"),
                "joint_diagnostics": sim_status.get("joint_diagnostics", []),
                "joint_catalog": sim_status.get("joint_catalog", []),
                "robot_joint_names": sim_status.get("robot_joint_names", []),
                "robot_ground": sim_status.get("robot_ground", {}),
                "telemetry": sim_status.get("telemetry", {"enabled": False, "run_dir": ""}),
            },
            "height": {
                "current_mm": self.current_height_mm,
                "current_cm": self.current_height_cm,
                "current_m": obstacle_height_m_mm(self.current_height_mm),
                "scene_mm": self.loaded_sim_height_mm,
                "scene_cm": self.loaded_sim_height_cm,
                "measured_height_mm": dict(sim_status.get("last_set_height_result", {}) or {}).get("measured_height_mm", sim_status.get("measured_height_mm")),
                "measured_width_m": dict(sim_status.get("last_set_height_result", {}) or {}).get("measured_width_m", sim_status.get("measured_width_m")),
                "measured_bounds": dict(dict(sim_status.get("last_set_height_result", {}) or {}).get("measured_bounds", sim_status.get("measured_bounds", {})) or {}),
                "obstacle_revision": int(sim_status.get("obstacle_revision", 0) or 0),
                "steps_path": str(self._current_steps_path()),
                "current_version_id": self.current_version_id,
                "current_version_metadata": dict(self.current_version_metadata),
                "versions": self.store.list_versions(self.current_height_mm, include_legacy=True),
                "manifest_rows": manifest_rows,
                "manifest_warning": manifest_warning,
            },
            "sequence": {
                "rows": visible_rows,
                "count": self.manager.count,
                "valid": self.current_sequence_is_valid,
                "dirty": self.manager.dirty,
                "accepted_path": str(self.manager.accepted_path or self._current_steps_path()),
                "revision": self.manager.revision,
                "view": "height",
                "source": self.get_visible_steps_source(),
                "summary": sequence_summary,
                "event_count": sequence_summary["event_count"],
                "total_duration_s": sequence_summary["total_duration_s"],
                "last_modified": sequence_summary["last_modified"],
                "read_only": self.visible_steps_are_read_only(),
            },
            "height_sequence": {
                "rows": visible_rows,
                "count": self.manager.count,
                "dirty": self.manager.dirty,
                "accepted_path": str(self.manager.accepted_path or self._current_steps_path()),
                "revision": self.manager.revision,
            },
            "operation": {
                "state": self.operation.state.value,
                "idle": self.operation.idle,
                "reason": self.operation.reason,
            },
            "recording": {
                "active": self.recording_active,
                "mode": self.mode,
                "pending": self.pending_step is not None,
                "pending_replacement": self.pending_replacement is not None,
                "stop_pending": self.record_stop_pending is not None,
                "events": len(self.record_events),
                "output_root": str(self.store.root),
                "height_folder": str(self.store.height_dir(self.current_height_mm)),
                "accepted_steps_path": str(self._current_steps_path()),
                "version_id": self.current_version_id,
                "dirty": bool(self.manager.dirty),
                "baseline": dict(self.recording_baseline_status),
            },
            "playback": {
                **playback,
                "can_start": availability.can_start,
                "can_respawn_start": availability.can_respawn_start,
                "can_play_selected": availability.can_play_selected,
                "can_pause": availability.can_pause,
                "can_resume": availability.can_resume,
                "can_stop": availability.can_stop,
                "can_analyze": availability.can_analyze,
                "can_export": availability.can_export,
                "unavailable_reason": availability.reason,
            },
            "servos": display_state["servos"],
            "wheels": wheels_short,
            "target_joint_state": sim_status.get("target_joint_state"),
            "actual_joint_state": sim_status.get("actual_joint_state"),
            "wheel_command": wheel_command,
            "joint_diagnostics": sim_status.get("joint_diagnostics", []),
            "joint_catalog": sim_status.get("joint_catalog", []),
            "detail_text": self.detail_text,
            "status_text": "\n".join(self.status_log[-250:]),
            "selected_step_index": self.selected_step_index,
            "combine": {
                "enabled": self.combine_mode_enabled,
                "selected_indices": sorted(self.combine_selected_indices),
                "selected_count": len(self.combine_selected_indices),
                "contiguous": self.combine_selection_is_contiguous(),
                "preview": self.combine_preview_text,
                "allow_conflicts": self.allow_combine_conflicts,
            },
            "servo_wheel": {
                "active": self.servo_wheel_staging_active,
                "dirty": self.servo_wheel_staged_dirty,
                "staged_state": clone_command_state(self.servo_wheel_staged_state),
                "last_launch_time": self.servo_wheel_last_launch_time,
                "last_launch_status": dict(self.servo_wheel_last_launch_status),
                "status": "Staging never moves the robot; Launch sends one MotionBatch and applies both channels in one physics tick.",
            },
            "actuator_command": {
                "semantics": "fixed-100-percent-direct-v1",
                "wheel_unit": "rad/s",
                "last_command": self.last_motion_command,
                "last_wheel_targets": list(self.last_wheel_targets),
                "last_warnings": list(self.last_motion_warnings),
            },
            "motion_profile": motion_profile,
            "revisions": {
                "sequence": self.manager.revision,
                "height_sequence": self.manager.revision,
                "manifest": self.manifest_revision,
                "status": len(self.status_log),
            },
        }

    def status_line(self) -> str:
        availability = self.playback_availability()
        return (
            f"Connected={self.sim_connected} Runtime Ready={self.runtime_ready} no_sim={self.no_sim} | "
            f"phase={self.latest_sim_status.get('phase', 'none')} pid={self.latest_sim_status.get('worker_pid', self.latest_sim_status.get('pid', ''))} | "
            f"Current height={self.current_height_mm} mm version={self.current_version_id or 'none'} | "
            f"Steps source={self.get_visible_steps_source()} | "
            f"Loaded steps path={self.get_visible_steps_path() or '-'} | "
            f"Sequence count={self.manager.count} valid={self.current_sequence_is_valid} | "
            f"Operation={self.operation.state.value} | "
            f"Playback active={self.playback.active} scheduled={self.playback.scheduled_start_at > 0.0} paused={self.playback.paused} | "
            f"Recording active={self.recording_active} | "
            "Servo+Wheel=atomic batch | "
            f"Dirty={self.manager.dirty} | "
            f"RTF={float(self.latest_sim_status.get('real_time_factor', 0.0) or 0.0):.2f} | "
            f"Playback unavailable={availability.reason or '-'}"
        )

    def _handle_tokens(self, tokens: list[str], command: str, source: str, message: CommandMessage | None = None) -> None:
        verb = tokens[0].lower()
        if verb == "mode":
            self.mode = MODE_TEST
            self._info("[INFO] Mode set to TEST; Servo+Wheel uses one stateless MotionBatch.")
        elif verb in {"test_mode"}:
            self.mode = MODE_TEST
            self._info("[INFO] Mode set to TEST.")
        elif verb == "status":
            self._info("[STATUS] " + self.status_line())
        elif verb == "validate_recording_baseline":
            result = self.validate_recording_baseline()
            if result["passed"]:
                self._info(f"[INFO] Recording baseline PASS: {result['baseline_id']}")
            else:
                self._warn("[WARN] Recording baseline FAIL: " + "; ".join(result["mismatches"][:5]))
        elif verb in {"e_stop", "estop", "space"}:
            self._cancel_pending_selected_playback(reason="emergency stop")
            self.playback.stop(silent=True)
            self.stop_wheels(reason="emergency_stop")
            self.mode = MODE_E_STOP
            self.operation.finish()
            self._warn("[WARN] Emergency stop: playback stopped and wheels zeroed.")
        elif verb == "respawn":
            self.respawn_robot(source="manual")
        elif verb == "servo_wheel":
            self._handle_servo_wheel(tokens[1:])
        elif verb in {"servo", "angle", "wheel", "wheels", "speed", "w", "s", "a", "d", "x", "stop", "home", "reset"}:
            self._apply_motion_command(command, source, message=message)
        elif verb in {"step_record", "sr"}:
            self._handle_step_record(tokens[1:])
        elif verb in {"accept", "mark"}:
            self.accept_pending_step()
        elif verb == "replace_step":
            self._handle_replace_step(tokens[1:])
        elif verb == "delete_step" and len(tokens) == 2:
            self.delete_step(int(tokens[1]))
        elif verb == "undo":
            self.undo_step()
        elif verb == "clear_steps":
            self._warn("[WARN] Check Confirm Clear All and run clear_steps_confirmed to clear all accepted steps.")
        elif verb == "clear_steps_confirmed":
            self.clear_steps()
        elif verb == "show_step":
            self.show_step_command(tokens[1:])
        elif verb == "show_step_summary":
            self.show_step_summary_command(tokens[1:])
        elif verb == "inspect_step":
            self.inspect_step_command(tokens[1:])
        elif verb == "export_step_json" and len(tokens) == 2:
            self.export_step_json(int(tokens[1]))
        elif verb in {"play", "play_all"}:
            fast = "fast" in [token.lower() for token in tokens[1:]]
            self.play_all(fast=fast)
        elif verb in {"respawn_play", "respawn_and_play"}:
            self._handle_respawn_play(tokens[1:])
        elif verb == "play_step":
            self._handle_play_step(tokens[1:])
        elif verb == "play_to_step":
            self._handle_play_to_step(tokens[1:])
        elif verb == "playback_debug_selected":
            self.playback_debug_selected()
        elif verb == "pause_play":
            self.playback.pause()
        elif verb == "resume_play":
            self.playback.resume()
        elif verb in {"stop_play", "stop_playback"}:
            if not self._cancel_pending_selected_playback(reason="Stop Play clicked during restore"):
                self.playback.stop()
        elif verb == "analyze_playback_timing":
            self.detail_text = self.playback.analyze_steps(self.manager.steps)
            self._info("[INFO] " + self.detail_text)
        elif verb == "export_motion_txt":
            self.export_motion_txt()
        elif verb in {"combine", "combine_steps", "merge_steps", "merge"}:
            self._handle_combine(tokens[1:])
        else:
            self._warn(f"[WARN] Unsupported command: {command}")

    def _handle_servo_wheel(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        if sub in {"start", "mode"}:
            self.start_servo_wheel_mode()
        elif sub in {"apply", "launch"}:
            self.launch_servo_wheel()
        elif sub in {"clear", "clear_staged"}:
            self.clear_servo_wheel_staged()
        elif sub == "cancel":
            self.cancel_servo_wheel_mode()
        else:
            self._info(
                f"[SERVO-WHEEL] active={self.servo_wheel_staging_active} "
                f"dirty={self.servo_wheel_staged_dirty} last={self.servo_wheel_last_launch_status}"
            )

    def start_servo_wheel_mode(self) -> bool:
        ok, reason = self.manual_motion_readiness(allow_staging=True)
        if not ok:
            self._warn("[WARN] Cannot start Servo-Wheel Mode: " + reason)
            return False
        self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
        self.servo_wheel_staging_active = True
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_last_launch_status = {"state": "staging", "error": ""}
        self._info("[INFO] Servo-Wheel Mode started; sliders now stage targets without moving the robot.")
        return True

    def stage_servo_wheel_servo(self, joint_name: str, value: float) -> bool:
        if not self.servo_wheel_staging_active:
            return False
        command = validate_motion_command(
            f"servo {joint_name} {float(value):.9g}",
            default_wheel_speed_rad_s=self.default_wheel_speed,
            max_wheel_speed_rad_s=self.max_wheel_speed,
        ).command
        apply_command_to_state(self.servo_wheel_staged_state, command)
        self.servo_wheel_staged_dirty = True
        return True

    def stage_servo_wheel_wheel(self, wheel_name: str, value: float) -> bool:
        if not self.servo_wheel_staging_active:
            return False
        short = WHEEL_NAME_TO_SHORT.get(wheel_name, wheel_name)
        command = validate_motion_command(
            f"wheel {short} {float(value):.9g}",
            default_wheel_speed_rad_s=self.default_wheel_speed,
            max_wheel_speed_rad_s=self.max_wheel_speed,
        ).command
        apply_command_to_state(self.servo_wheel_staged_state, command)
        self.servo_wheel_staged_dirty = True
        return True

    def clear_servo_wheel_staged(self) -> bool:
        if not self.servo_wheel_staging_active:
            self._warn("[WARN] Start Servo-Wheel Mode before clearing staged targets.")
            return False
        self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_last_launch_status = {"state": "staging_reset_to_live", "error": ""}
        self._info("[INFO] Staged targets reset to current live state; no actuator command was sent.")
        return True

    def launch_servo_wheel(self) -> str:
        if not self.servo_wheel_staging_active:
            self._warn("[WARN] Start Servo-Wheel Mode before Launch.")
            return ""
        ok, reason = self.manual_motion_readiness(allow_staging=True)
        if not ok:
            self._warn("[WARN] Servo-Wheel Launch blocked: " + reason)
            return ""
        staged_before = clone_command_state(self.transport.capture_command_state())
        staged_after = clone_command_state(self.servo_wheel_staged_state)
        batch_id = self.apply_servo_wheel_together(
            staged_after["servos"],
            staged_after["wheels"],
            source="servo_wheel_launch",
            staged_state_before=staged_before,
            staged_state_after=staged_after,
        )
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_last_launch_status = {"state": "launch_requested", "batch_id": batch_id, "error": ""}
        return batch_id

    def cancel_servo_wheel_mode(self) -> bool:
        if self.sim_connected:
            self.stop_wheels(reason="servo_wheel_cancel")
        self.servo_wheel_staging_active = False
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
        self.servo_wheel_last_launch_status = {"state": "cancelled", "error": ""}
        if not self.recording_active:
            self.mode = MODE_TEST
        self._info("[INFO] Servo-Wheel Mode cancelled; staged targets cleared and wheels stopped.")
        return True

    def respawn_robot(self, *, source: str = "manual") -> bool:
        if not self.no_sim and not self.runtime_ready:
            self._warn("[WARN] Isaac runtime is not ready.")
            return False
        if self.recording_active:
            self._warn("[WARN] Stop recording before Respawn.")
            return False
        if self.pending_step is not None or self.pending_replacement is not None:
            self._warn("[WARN] Accept or discard pending step/replacement before Respawn.")
            return False
        if not self.operation.begin(OperationState.RESPAWNING, detail="Robot respawn is in progress."):
            self._warn(f"[WARN] Cannot respawn: {self.operation.reason}")
            return False
        respawn_ok, respawn_reason = self.respawn_readiness()
        if not self.no_sim and not respawn_ok:
            self.operation.finish(OperationState.RESPAWNING)
            self._warn("[WARN] " + (respawn_reason or "Respawn is not ready. Calibrate a valid ground reference first."))
            return False
        source_name = str(source or "manual")
        playback_managed = source_name == "playback"
        was_estop = self.mode == MODE_E_STOP
        if not playback_managed:
            self.playback.stop(silent=True, reason=f"{source_name}_respawn")
        try:
            self.stop_wheels(reason="respawn")
            self.transport.respawn()
            self.transport.request_state()
        except Exception:
            self.operation.finish(OperationState.RESPAWNING)
            raise
        if not playback_managed:
            if was_estop:
                self.mode = MODE_E_STOP
            else:
                self.mode = MODE_TEST
            self.operation.finish(OperationState.RESPAWNING)
            self._info(f"[INFO] Respawn requested by {source_name}; playback stopped and wheels stopped.")
        else:
            self._info("[INFO] Playback Respawn requested; wheels stopped.")
        return True

    def is_motion_command(self, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        return tokens[0].lower() in {"servo", "angle", "wheel", "wheels", "speed", "w", "a", "s", "d", "x", "home"}

    def is_immediate_stop_command(self, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        lowered = [token.lower() for token in tokens]
        return lowered == ["stop"] or lowered == ["wheel", "stop"] or lowered == ["wheels", "stop"]

    def apply_servo_wheel_together(
        self,
        servo_targets_deg: dict[str, float],
        wheel_targets_rad_s: dict[str, float],
        *,
        source: str = "ui",
        staged_state_before: dict[str, Any] | None = None,
        staged_state_after: dict[str, Any] | None = None,
    ) -> str:
        before_state = clone_command_state(self.transport.capture_command_state())
        after_state = clone_command_state(before_state)
        canonical_servos: dict[str, float] = {}
        canonical_wheels: dict[str, float] = {}
        commands: list[str] = []
        for name in SERVO_JOINT_NAMES:
            target = float(servo_targets_deg.get(name, after_state["servos"].get(name, 0.0)))
            validated = validate_motion_command(
                f"servo {name} {target:.9g}",
                default_wheel_speed_rad_s=self.default_wheel_speed,
                max_wheel_speed_rad_s=self.max_wheel_speed,
            )
            apply_command_to_state(after_state, validated.command)
            canonical_servos[name] = float(after_state["servos"][name])
            commands.append(validated.command)
        for name in WHEEL_JOINT_NAMES:
            short = WHEEL_NAME_TO_SHORT.get(name, name)
            target = wheel_targets_rad_s.get(name, wheel_targets_rad_s.get(short, after_state["wheels"].get(name, 0.0)))
            validated = validate_motion_command(
                f"wheel {short} {float(target):.9g}",
                default_wheel_speed_rad_s=self.default_wheel_speed,
                max_wheel_speed_rad_s=self.max_wheel_speed,
            )
            apply_command_to_state(after_state, validated.command)
            canonical_wheels[name] = float(after_state["wheels"][name])
            commands.append(validated.command)
        batch_id = uuid.uuid4().hex
        payload = {
            "batch_id": batch_id,
            "source": str(source or "ui"),
            "requested_sim_boundary": "next_physics_tick",
            "servo_targets_deg": canonical_servos,
            "wheel_targets_rad_s": canonical_wheels,
            "wheel_generation": self.transport.wheel_generation,
            "requested_wall_time": time.time(),
            "recording_active": self.recording_active,
            "source_step": (
                int(self.replace_target_index)
                if self.replace_target_index is not None
                else int(self.manager.count + 1) if self.recording_active else None
            ),
            "metadata": {
                "height_mm": self.current_height_mm,
                "version_id": self.current_version_id,
                "atomicity": "single_worker_cycle_single_articulation_write",
            },
            "recording_metadata": {
                "recording_active": self.recording_active,
                "height_mm": self.current_height_mm,
                "version_id": self.current_version_id,
                "actuator_command_semantics": "fixed-100-percent-direct-v1",
            },
            "high_priority": False,
        }
        self.transport.apply_motion_batch(payload)
        self.last_motion_command = "apply_motion_batch"
        self.last_wheel_targets = tuple(canonical_wheels[name] for name in WHEEL_JOINT_NAMES)
        self.servo_wheel_last_launch_time = time.time()
        if self.recording_active and source != "playback":
            event_time, actual_time, timing_source = self._record_reference_time_s()
            self.record_reference_state = clone_command_state(after_state)
            is_launch = source == "servo_wheel_launch"
            event = make_event(
                event_time,
                "servo_wheel launch" if is_launch else "apply_motion_batch",
                kind="servo_wheel_launch" if is_launch else "motion_batch",
                command_state_before=before_state,
                command_state_after=after_state,
                expanded_commands=commands,
            )
            event.update(
                actual_recording_time_s=actual_time,
                recording_timing_source=timing_source,
                staged_state_before=clone_command_state(staged_state_before or before_state),
                staged_state_after=clone_command_state(staged_state_after or after_state),
                batch_ack=None,
                **self._motion_event_metadata(before_state, after_state, batch_id=batch_id),
            )
            self.record_events.append(event)
        self.status = f"MotionBatch {batch_id} queued: servo+wheel apply on one physics tick."
        return batch_id

    def _apply_motion_command(self, command: str, source: str, *, message: CommandMessage | None = None) -> None:
        if self.is_immediate_stop_command(command):
            before_state = clone_command_state(self.transport.capture_command_state())
            self.stop_wheels(reason=f"{source}_stop")
            after_state = clone_command_state(self.transport.capture_command_state())
            if self.recording_active and source != "playback":
                event_time, actual_time, timing_source = self._record_reference_time_s()
                reference_before, reference_after = self._record_event_states([command])
                event = make_event(
                    event_time,
                    command,
                    kind=source,
                    command_state_before=reference_before,
                    command_state_after=reference_after,
                )
                event.update(
                    executed_command=command,
                    actual_recording_time_s=actual_time,
                    recording_timing_source=timing_source,
                    **self._motion_event_metadata(reference_before, reference_after),
                )
                self.record_events.append(event)
            self.status = f"Command: {command}"
            return
        if self.record_stop_pending is not None and source != "playback":
            self._warn("[WARN] Recording is stopping; new manual actuator commands are rejected.")
            return
        if source != "playback":
            manual_ok, manual_reason = self.manual_motion_readiness()
            if not manual_ok:
                self._warn(f"[WARN] Manual motion is disabled: {manual_reason}")
                return
        before_state = clone_command_state(self.transport.capture_command_state())
        validated = validate_motion_command(
            command,
            default_wheel_speed_rad_s=self.default_wheel_speed,
            max_wheel_speed_rad_s=self.max_wheel_speed,
        )
        outgoing_command = validated.command
        self.last_motion_command = outgoing_command
        self.last_wheel_targets = validated.applied_wheel_values_rad_s
        self.last_motion_warnings = validated.warnings
        outgoing_message = message or CommandMessage(text=outgoing_command, source=source)
        outgoing_message.text = outgoing_command
        self.transport.send(outgoing_command, source=source, message=outgoing_message)
        after_state = clone_command_state(self.transport.capture_command_state())
        if self.recording_active and source != "playback":
            event_time, actual_time, timing_source = self._record_reference_time_s()
            reference_before, reference_after = self._record_event_states([command])
            event = make_event(
                event_time,
                command,
                kind=source,
                command_state_before=reference_before,
                command_state_after=reference_after,
            )
            event.update(
                executed_command=outgoing_command,
                actual_recording_time_s=actual_time,
                recording_timing_source=timing_source,
                **self._motion_event_metadata(reference_before, reference_after),
            )
            self.record_events.append(event)
        self.status = f"Command: {outgoing_command}"
        if self.last_motion_warnings:
            self.status += " | " + "; ".join(self.last_motion_warnings)

    def _handle_step_record(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        if sub == "start":
            self.start_step_recording()
        elif sub == "stop":
            self.stop_step_recording()
        elif sub == "accept":
            if self.pending_replacement is not None or self.mode == MODE_PENDING_REPLACEMENT:
                self.accept_replacement()
            else:
                self.accept_pending_step()
        elif sub == "discard":
            self.discard_pending_step(restore_before=False)
        elif sub == "discard_restore_before":
            self.discard_pending_step(restore_before=True)
        else:
            self._info(
                f"[RECORD] mode={self.mode} events={len(self.record_events)} "
                f"pending={self.pending_step is not None} replacement={self.pending_replacement is not None}"
            )

    def start_step_recording(self) -> None:
        ok, reason = self.can_start_recording()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        if not self.operation.begin(OperationState.RECORDING, detail="Recording is active."):
            self._warn(f"[WARN] Cannot start recording: {self.operation.reason}")
            return
        if self.mode == MODE_TEST:
            self.recording_kind = "recorded"
            self.mode = MODE_RECORDING_STEP
        elif self.mode == MODE_REPLACE_STEP_READY and self.replace_target_index is not None:
            self.recording_kind = "replacement"
            self.mode = MODE_REPLACING_STEP
        else:
            self._warn(f"[WARN] step_record start is only valid in TEST or REPLACE_STEP_READY. Current mode={self.mode}")
            return
        self.record_start_wall_time = time.monotonic()
        self.transport.request_state()
        self.record_start_sim_time = self._current_record_sim_time()
        self.record_command_state_before = clone_command_state(self.transport.capture_command_state())
        self.record_reference_state = clone_command_state(self.record_command_state_before)
        self.record_sim_state_before = self.capture_current_sim_state()
        self.record_events = []
        self._info(f"[INFO] Step recording started. mode={self.mode}; actuator commands use the fixed 100% profile.")

    def stop_step_recording(self) -> None:
        if not self.recording_active or self.record_command_state_before is None:
            self._warn("[WARN] step_record stop is only valid while recording.")
            return
        duration, actual_duration, timing_source = self._record_reference_time_s()
        before, after = self._record_event_states(["wheel stop"])
        stop_event = make_event(
            duration,
            "wheel stop",
            kind="recording_stop_boundary",
            command_state_before=before,
            command_state_after=after,
        )
        stop_event.update(
            executed_command="wheel stop",
            actual_recording_time_s=actual_duration,
            recording_timing_source=timing_source,
            final_stop_command=True,
            **self._motion_event_metadata(before, after),
        )
        self.record_events.append(stop_event)
        stop_request = self.stop_wheels(reason="stop_recording")
        self.record_stop_pending = {
            "duration": duration,
            "actual_duration": actual_duration,
            "timing_source": timing_source,
            "command_id": str(stop_request.get("command_id", "") or ""),
            "requested_at": time.monotonic(),
        }
        self._info("[INFO] Stopping wheels... recording finalize waits for zero-target acknowledgment.")
        self._try_finalize_record_stop()

    def _try_finalize_record_stop(self) -> None:
        pending = self.record_stop_pending
        if pending is None:
            return
        status = dict(self.latest_sim_status.get("wheel_command", {}) or {})
        if not status:
            status = dict(getattr(self.transport.adapter, "wheel_command_status", {}) or {})
        command_matches = not pending["command_id"] or str(status.get("command_id", "") or "") == pending["command_id"]
        if command_matches and bool(status.get("zero_target_applied", False)):
            self.record_stop_pending = None
            self._info("[INFO] Zero target applied; finalizing recording.")
            self._finalize_step_recording(
                duration=float(pending["duration"]),
                actual_duration=float(pending["actual_duration"]),
                timing_source=str(pending["timing_source"]),
                wheel_stop_status=status,
            )
        elif time.monotonic() - float(pending["requested_at"]) > 3.0:
            self._warn("[WARN] Recording remains open: worker did not acknowledge zero wheel target within 3s.")

    def _finalize_step_recording(
        self,
        *,
        duration: float,
        actual_duration: float,
        timing_source: str,
        wheel_stop_status: dict[str, Any],
    ) -> None:
        self.transport.request_state()
        after_state = clone_command_state(self.transport.capture_command_state())
        sim_state_after = self.capture_current_sim_state()
        events = self.record_events
        self.last_record_coalesce_stats = {
            "original_count": len(events),
            "coalesced_count": len(events),
            "coalesced": False,
            "dropped_count": 0,
        }
        if self.record_coalesce_slider_events:
            events, self.last_record_coalesce_stats = coalesce_record_events(
                self.record_events,
                min_interval_s=self.record_event_min_interval_s,
                max_events=self.record_max_events_per_step,
            )
        is_replacement = self.mode == MODE_REPLACING_STEP and self.replace_target_index is not None
        index = self.replace_target_index if is_replacement else self.manager.count + 1
        step = make_step(
            index=int(index),
            step_type="replacement" if is_replacement else "recorded",
            duration=duration,
            events=events,
            command_state_before=self.record_command_state_before,
            command_state_after=after_state,
            name=f"step_{int(index):03d}_{'replacement' if is_replacement else 'recorded'}_height_{self.current_height_mm:03d}mm_dur_{duration:.2f}s",
            note=f"height={self.current_height_mm}mm",
            extra={
                "height_mm": self.current_height_mm,
                "height_m": obstacle_height_m_mm(self.current_height_mm),
                "recording_baseline": {
                    "baseline_id": self.recording_baseline_status.get("baseline_id", ""),
                    "baseline_version": self.recording_baseline_status.get("baseline_version", ""),
                    "baseline_sha256": self.recording_baseline_status.get("baseline_sha256", ""),
                    "physics_configuration_id": self.recording_baseline_status.get("baseline_id", ""),
                    "actuator_configuration_id": self.recording_baseline_status.get("baseline_id", ""),
                    "scene_baseline": copy.deepcopy(self.recording_baseline_status.get("scene_baseline", {})),
                },
                "record_coalesce": dict(self.last_record_coalesce_stats),
                "sim_state_before": self.record_sim_state_before,
                "sim_state_after": sim_state_after,
                "wheel_stop_status": dict(wheel_stop_status),
                "recording_timing": {
                    "actuator_command_semantics": "fixed-100-percent-direct-v1",
                    "actuator_command_semantics": "fixed-100-percent-direct-v1",
                    "source": timing_source,
                    "actual_duration_s": actual_duration,
                    "wheel_active_duration_s": duration,
                },
                "motion_semantics": self._recording_motion_semantics(events, duration, sim_state_after),
            },
        )
        if is_replacement:
            self.pending_replacement = step
            self.mode = MODE_PENDING_REPLACEMENT
            self.set_step_summary_detail(step, title=f"Pending replacement for step {int(index):03d}")
            self._info(f"[INFO] Replacement recording stopped; pending replacement has {len(events)} events.")
        else:
            self.pending_step = step
            self.mode = MODE_PENDING_RECORDED_STEP
            self.set_step_summary_detail(step, title=f"Pending recorded step {int(index):03d}")
            self._info(f"[INFO] Step recording stopped; pending step has {len(events)} events.")
        if not events:
            self._warn("[WARN] No motion was recorded.")
        if self.last_record_coalesce_stats.get("coalesced"):
            self._warn(
                "[WARN] Recorded step has many events; coalesced from "
                f"{self.last_record_coalesce_stats.get('original_count')} to "
                f"{self.last_record_coalesce_stats.get('coalesced_count')}."
            )
        self.record_events = []
        self.record_command_state_before = None
        self.record_reference_state = None
        self.record_sim_state_before = None
        self.record_start_sim_time = None
        self.recording_kind = ""
        self.record_stop_pending = None
        self.operation.finish(OperationState.RECORDING)

    def accept_pending_step(self) -> None:
        ok, reason = self.can_accept_recorded_step()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any] | None:
            if self.pending_step is None:
                return None
            accepted = self.manager.add_step(self.pending_step)
            self.pending_step = None
            self.mode = MODE_TEST
            self.selected_step_index = int(accepted["index"])
            self.set_step_summary_detail(accepted, title=f"Step {int(accepted['index']):03d} accepted")
            self._info(f"[INFO] Accepted recorded step {int(accepted['index']):03d}: {accepted['name']}")
            return accepted

        self._run_operation("Accept recorded step", work)

    def discard_pending_step(self, *, restore_before: bool) -> None:
        pending = self.pending_step or self.pending_replacement
        if pending is None:
            self._warn("[WARN] No pending step to discard.")
            return
        if restore_before:
            self.apply_command_state(pending.get("command_state_before"), keep_wheels=False)
        self.pending_step = None
        self.pending_replacement = None
        self.mode = MODE_TEST
        self._info("[INFO] Pending step discarded.")

    def _handle_replace_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: replace_step <index>|start|stop|accept|discard|cancel")
            return
        sub = args[0].lower()
        if sub.isdigit():
            self.prepare_replacement(int(sub))
        elif sub == "start":
            self.start_step_recording()
        elif sub == "stop":
            self.stop_step_recording()
        elif sub == "accept":
            self.accept_replacement()
        elif sub == "discard":
            self.discard_pending_step(restore_before=False)
        elif sub == "cancel":
            self.pending_replacement = None
            self.replace_target_index = None
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self.mode = MODE_TEST
            self._info("[INFO] Replacement cancelled.")

    def prepare_replacement(self, index: int) -> None:
        ok, reason = self.can_prepare_replacement()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        self.manager.get_step(index)
        self.replace_target_index = index
        self.mode = MODE_REPLACE_STEP_READY
        self._info(f"[INFO] Replacement prepared for step {index:03d}. Press Start Replacement Recording.")

    def accept_replacement(self) -> None:
        ok, reason = self.can_accept_replacement()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any] | None:
            if self.pending_replacement is None or self.replace_target_index is None:
                return None
            target = self.replace_target_index
            replaced = self.manager.replace_step(target, self.pending_replacement)
            self.selected_step_index = target
            self.set_step_summary_detail(replaced, title=f"Step {int(replaced['index']):03d} replaced")
            self.pending_replacement = None
            self.replace_target_index = None
            self.mode = MODE_TEST
            self._info(f"[INFO] Replaced step {int(replaced['index']):03d}.")
            return replaced

        self._run_operation("Accept replacement", work)

    def delete_step(self, index: int) -> None:
        removed = self.manager.delete_step(index)
        self.selected_step_index = min(index, self.manager.count) if self.manager.count else None
        self._info(f"[INFO] Deleted accepted step {index:03d}: {removed.get('name', '')}")

    def undo_step(self) -> None:
        removed = self.manager.undo()
        if removed is None:
            self._warn("[WARN] No accepted step to undo.")
            return
        self.selected_step_index = self.manager.count if self.manager.count else None
        self._info(f"[INFO] Undid accepted step {int(removed.get('index', 0)):03d}.")

    def clear_steps(self) -> None:
        self.manager.clear()
        self.selected_step_index = None
        self.detail_text = ""
        self._info("[INFO] Cleared all accepted steps in memory.")

    def show_step_command(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: show_step <index> [before|after]")
            return
        index = int(args[0])
        when = "after"
        for token in args[1:]:
            if token.lower() in {"before", "after"}:
                when = token.lower()
        self.show_step(index, when=when)

    def show_step(self, index: int, *, when: str) -> None:
        if self.recording_active:
            self._warn("[WARN] Stop recording before Show Selected Before/After.")
            return
        if self.playback.active or self.playback.start_requested:
            self._warn("[WARN] Stop playback before Show Selected Before/After.")
            return
        if not self.no_sim and not self.runtime_ready:
            self._warn("[WARN] Isaac runtime is not ready before applying a step state.")
            return
        step = normalize_step(self.get_visible_step(index), index=index)
        state_key = "command_state_before" if when == "before" else "command_state_after"
        self.apply_command_state(step.get(state_key), keep_wheels=False)
        self.selected_step_index = index
        self.detail_text = (
            f"Applied {when} command state for step {index:03d}.\n\n"
            + self.command_state_summary(step.get(state_key))
            + "\n\n"
            + self.compact_step_details(step, title=f"Step {index:03d} summary")
        )
        self._info(f"[INFO] Applied {when} state for step {index:03d}.")

    def show_step_summary_command(self, args: list[str]) -> None:
        if len(args) != 1:
            self._warn("[WARN] Usage: show_step_summary <index>")
            return
        index = int(args[0])
        step = normalize_step(self.get_visible_step(index), index=index)
        self.selected_step_index = index
        self.set_step_summary_detail(step, title=f"Step {index:03d} summary")
        self._info(f"[INFO] Showing compact summary for step {index:03d}.")

    def inspect_step_command(self, args: list[str]) -> None:
        if len(args) < 1:
            self._warn("[WARN] Usage: inspect_step <index>")
            return
        index = int(args[0])
        step = normalize_step(self.get_visible_step(index), index=index)
        self.selected_step_index = index
        self.detail_text = self.step_json_text(step)
        self._info(f"[INFO] Inspecting step {index:03d} JSON (truncated if needed).")

    def apply_command_state(self, state: dict[str, Any] | None, *, keep_wheels: bool) -> None:
        normalized = clone_command_state(state)
        if not keep_wheels:
            normalized["wheels"] = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self.apply_servo_wheel_together(
            normalized["servos"],
            normalized["wheels"],
            source="show_step",
        )

    def play_all(self, *, fast: bool) -> bool:
        ok, reason = self.playback_readiness()
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        steps = self.get_visible_steps()
        if not steps:
            return False
        profile = "fast" if fast else self.playback.profile
        label = f"{self.current_height_mm}mm accepted steps"
        return self.start_playback(steps, label=label, profile=profile)

    def _handle_respawn_play(self, args: list[str]) -> None:
        tokens = [arg.lower() for arg in args]
        fast = "fast" in tokens
        if "step" in tokens or "selected" in tokens:
            index = self.selected_step_index
            for token in tokens:
                if token.isdigit():
                    index = int(token)
            if index is None:
                self._warn("[WARN] Select an accepted step first.")
                return
            profile = "fast" if fast else self.playback.profile
            self.start_playback(
                [self.get_visible_step(index)],
                label=f"{self.current_height_mm}mm step {int(index):03d}",
                profile=profile,
                restore_start_state=True,
                respawn_first=True,
            )
            return
        if "to" in tokens:
            index = self.selected_step_index
            for token in tokens:
                if token.isdigit():
                    index = int(token)
            if index is None:
                self._warn("[WARN] Select an accepted step first.")
                return
            profile = "fast" if fast else self.playback.profile
            steps = self.get_visible_steps()
            self.start_playback(
                steps[: int(index)],
                label=f"{self.current_height_mm}mm steps 1..{int(index)}",
                profile=profile,
                respawn_first=True,
            )
            return
        steps = self.get_visible_steps()
        if not steps:
            return
        self.start_playback(
            steps,
            label=f"{self.current_height_mm}mm accepted steps",
            profile="fast" if fast else self.playback.profile,
            respawn_first=True,
        )

    def _handle_play_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: play_step <index> [fast|raw]")
            return
        ok, reason = self.can_playback()
        self._info(
            "[PLAYBACK DEBUG] play_step click "
            f"args={args} can_playback={ok} reason={reason or 'ok'} "
            f"restore_start_state={self.restore_step_start_state_before_selected_playback} "
            f"mode={self.mode} sim_ready={self.sim_ready} no_sim={self.no_sim}"
        )
        if not ok:
            self._warn("[WARN] " + reason)
            return
        index = int(args[0])
        raw_step = self.get_visible_step(index)
        normalized = normalize_step(raw_step, index=index)
        self._info(
            "[PLAYBACK DEBUG] selected step "
            f"index={index} sim_state_before={isinstance(normalized.get('sim_state_before'), dict)} "
            f"command_state_before={isinstance(normalized.get('command_state_before'), dict)} "
            f"events={len(normalized.get('events', []))}"
        )
        profile = "fast" if "fast" in [arg.lower() for arg in args[1:]] else self.playback.profile
        self.start_selected_step_playback(index, profile=profile)

    @staticmethod
    def _saved_state_is_available(value: Any) -> bool:
        return isinstance(value, dict) and bool(value)

    @staticmethod
    def _nested_values_close(left: Any, right: Any, *, tolerance: float = 1.0e-6) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            common = set(left).intersection(right)
            return bool(common) and all(
                HeightReplayController._nested_values_close(left[key], right[key], tolerance=tolerance)
                for key in common
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                HeightReplayController._nested_values_close(a, b, tolerance=tolerance)
                for a, b in zip(left, right)
            )
        try:
            return abs(float(left) - float(right)) <= tolerance
        except (TypeError, ValueError):
            return left == right

    @classmethod
    def _continuity_states_match(cls, left: dict[str, Any], right: dict[str, Any]) -> bool:
        continuity_keys = ("root_pose", "root_velocity", "joint_pos", "joint_vel", "command_state")
        common = [key for key in continuity_keys if key in left and key in right]
        if common:
            return all(cls._nested_values_close(left[key], right[key]) for key in common)
        return cls._nested_values_close(left, right)

    def resolve_selected_step_restore_state(self, selected_index: int) -> dict[str, Any]:
        """Resolve, without mutation, the authoritative saved state for Selected playback."""

        index = int(selected_index)
        if index < 1 or index > self.manager.count:
            raise IndexError(f"Selected step index {index} is outside 1..{self.manager.count}.")
        selected = self.manager.get_step(index)
        previous = self.manager.get_step(index - 1) if index > 1 else None
        source_step_index = index
        source_field = ""
        source_value: dict[str, Any] | None = None
        fallback_used = False

        if previous is not None and self._saved_state_is_available(previous.get("sim_state_after")):
            source_step_index = index - 1
            source_field = "sim_state_after"
            source_value = previous["sim_state_after"]
        elif previous is not None and self._saved_state_is_available(previous.get("command_state_after")):
            source_step_index = index - 1
            source_field = "command_state_after"
            source_value = previous["command_state_after"]
        elif self._saved_state_is_available(selected.get("sim_state_before")):
            source_field = "sim_state_before"
            source_value = selected["sim_state_before"]
            fallback_used = index > 1
        elif self._saved_state_is_available(selected.get("command_state_before")):
            source_field = "command_state_before"
            source_value = selected["command_state_before"]
            fallback_used = True

        continuity_warning = ""
        continuity = "NOT_AVAILABLE"
        if previous is not None:
            previous_sim = previous.get("sim_state_after")
            selected_sim = selected.get("sim_state_before")
            previous_command = previous.get("command_state_after")
            selected_command = selected.get("command_state_before")
            if self._saved_state_is_available(previous_sim) and self._saved_state_is_available(selected_sim):
                continuity = "PASS" if self._continuity_states_match(previous_sim, selected_sim) else "WARNING"
            elif self._saved_state_is_available(previous_command) and self._saved_state_is_available(selected_command):
                continuity = "PASS" if self._continuity_states_match(previous_command, selected_command) else "WARNING"
            if continuity == "WARNING":
                continuity_warning = (
                    f"Selected playback continuity warning: Step {index - 1} saved end state differs "
                    f"from Step {index} saved start state. Restoring authoritative previous-step end state."
                )

        restore_sim_state: dict[str, Any] | None = None
        restore_command_state: dict[str, Any] | None = None
        if source_value is not None:
            if source_field.startswith("sim_state"):
                restore_sim_state = copy.deepcopy(source_value)
                embedded = source_value.get("command_state")
                if isinstance(embedded, dict):
                    restore_command_state = clone_command_state(embedded)
            else:
                restore_command_state = clone_command_state(source_value)
                restore_sim_state = {"command_state": clone_command_state(source_value)}
        return {
            "selected_step": selected,
            "selected_step_index": index,
            "restore_source_step_index": source_step_index if source_field else None,
            "restore_source_field": source_field,
            "restore_sim_state": restore_sim_state,
            "restore_command_state": restore_command_state,
            "fallback_used": fallback_used,
            "continuity": continuity,
            "continuity_warning": continuity_warning,
        }

    def start_selected_step_playback(self, selected_index: int, *, profile: str) -> bool:
        ok, reason = self.can_playback()
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            self._warn("[WARN] Dirty Servo-Wheel staging blocks Selected playback; Launch, Clear, or Cancel first.")
            return False
        try:
            resolved = self.resolve_selected_step_restore_state(selected_index)
        except Exception as exc:
            self._warn(f"[WARN] Cannot play selected step {int(selected_index)}: {exc}")
            return False
        index = int(resolved["selected_step_index"])
        if not resolved["restore_source_field"] or not resolved["restore_sim_state"]:
            message = (
                f"Cannot play selected step {index}: no saved previous-step end state "
                "or selected-step start state is available."
            )
            self.playback.last_error = message
            self.playback.last_info = message
            self.playback.active = False
            self.playback.start_requested = False
            self.playback.scheduled_start_at = 0.0
            self.playback.progress.playback_state = PlaybackState.ERROR.value
            self.playback.progress.status_text = "Error"
            self.playback.progress.last_error = message
            self.detail_text = message
            self.operation.finish()
            self.mode = MODE_TEST
            self._warn("[ERROR] " + message)
            return False
        if not self.operation.begin(
            OperationState.PLAYBACK,
            detail=f"Restoring saved state before Selected Step {index}.",
        ):
            self._warn("[WARN] " + (self.operation.reason or "Another operation is active."))
            return False

        now = time.monotonic()
        request_id = uuid.uuid4().hex
        source_index = int(resolved["restore_source_step_index"])
        source_field = str(resolved["restore_source_field"])
        self.playback.set_profile(profile)
        self.playback.active = False
        self.playback.start_requested = True
        self.playback.scheduled_start_at = 0.0
        self.playback.plan = None
        self.playback.progress.playback_state = PlaybackState.RESTORING.value
        self.playback.progress.status_text = f"Restoring saved end state from Step {source_index}..."
        self.playback.progress.current_step_index = index
        self.playback.progress.total_steps = self.manager.count
        self.playback.progress.selected_playback = True
        self.playback.progress.command_phase = "restore_previous_saved_state"
        self.playback.last_error = ""
        self.playback.last_info = (
            f"Restoring Step {source_index}.{source_field} before Selected Step {index}."
        )
        self.mode = MODE_PLAYBACK
        self.pending_selected_playback = {
            **resolved,
            "profile": str(profile),
            "selected_fast_click_id": self.selected_fast_click_id if str(profile) == "fast" else "",
            "label": f"{self.current_height_mm}mm step {index:03d}",
            "request_id": request_id,
            "restore_count_before": int(self.latest_sim_status.get("restore_count", 0) or 0),
            "requested_at": now,
            "deadline": now + max(2.0, float(self.playback.worker_ack_timeout_s)),
            "verification_requested": False,
            "manager_revision": self.manager.revision,
            "height_mm": self.current_height_mm,
            "version_id": self.current_version_id,
            "trace": [{"event": "operation_acquired", "monotonic_s": now}],
        }
        self.status = (
            f"Restoring saved end state from Step {source_index}... "
            f"Restore source: Step {source_index}.{source_field}; Selected target: Step {index}."
        )
        self.detail_text = self.status
        if resolved["continuity_warning"]:
            self._warn("[WARN] " + str(resolved["continuity_warning"]))
        try:
            self.stop_wheels(reason="selected_restore_boundary")
            self.pending_selected_playback["trace"].append(
                {"event": "stop_wheels_sent", "monotonic_s": time.monotonic()}
            )
            self.transport.restore_sim_state(resolved["restore_sim_state"], request_id=request_id)
            self.pending_selected_playback["trace"].append(
                {"event": "restore_sent", "monotonic_s": time.monotonic(), "request_id": request_id}
            )
            self.status = (
                f"Restoring saved end state from Step {source_index}... "
                f"Restore source: Step {source_index}.{source_field}; Selected target: Step {index}."
            )
            self.detail_text = self.status
        except Exception as exc:
            self._fail_pending_selected_playback(f"previous saved state restore failed: {exc}")
            return False

        if self.no_sim:
            if hasattr(self.transport.adapter, "stop_wheels"):
                self.transport.adapter.stop_wheels()
            self.pending_selected_playback["trace"].append(
                {"event": "restore_acknowledged", "monotonic_s": time.monotonic(), "request_id": request_id}
            )
            observed = self.transport.capture_sim_state()
            verified, verify_detail = self._verify_selected_restore_state(
                self.pending_selected_playback,
                observed,
            )
            if not verified:
                self._fail_pending_selected_playback("restored state verification failed: " + verify_detail)
                return False
            self.pending_selected_playback["trace"].append(
                {"event": "state_verified", "monotonic_s": time.monotonic(), "detail": verify_detail}
            )
            return self._start_pending_selected_playback()
        return True

    @staticmethod
    def _flatten_numeric_values(value: Any) -> list[float]:
        if isinstance(value, (list, tuple)):
            flattened: list[float] = []
            for item in value:
                flattened.extend(HeightReplayController._flatten_numeric_values(item))
            return flattened
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []

    def _verify_selected_restore_state(
        self,
        pending: dict[str, Any],
        observed_sim_state: dict[str, Any],
    ) -> tuple[bool, str]:
        if not isinstance(observed_sim_state, dict) or not observed_sim_state:
            return False, "worker did not return a detailed sim_state after restore acknowledgment"
        comparisons: list[str] = []
        expected_command = pending.get("restore_command_state")
        observed_command = observed_sim_state.get("command_state")
        if isinstance(expected_command, dict) and isinstance(observed_command, dict):
            expected_servos = dict(expected_command.get("servos", {}) or {})
            observed_servos = dict(observed_command.get("servos", {}) or {})
            for name, target in expected_servos.items():
                if name not in observed_servos or abs(float(observed_servos[name]) - float(target)) > 1.0e-4:
                    return False, f"restored servo command mismatch for {name}"
            comparisons.append("servo_command_state")
        expected_sim = pending.get("restore_sim_state")
        if str(pending.get("restore_source_field", "")).startswith("sim_state") and isinstance(expected_sim, dict):
            for key, tolerance in (("root_pose", 0.05), ("joint_pos", 0.20)):
                expected_values = self._flatten_numeric_values(expected_sim.get(key))
                observed_values = self._flatten_numeric_values(observed_sim_state.get(key))
                if expected_values and observed_values:
                    if len(expected_values) != len(observed_values):
                        return False, f"restored {key} size mismatch"
                    if max(abs(left - right) for left, right in zip(expected_values, observed_values)) > tolerance:
                        return False, f"restored {key} differs beyond tolerance {tolerance}"
                    comparisons.append(key)
        if not comparisons:
            comparisons.append("worker_detailed_state_received")
        return True, "+".join(comparisons)

    def _selected_restore_transaction_is_current(self, pending: dict[str, Any]) -> bool:
        return bool(
            self.pending_selected_playback is pending
            and self.operation.state is OperationState.PLAYBACK
            and self.playback.start_requested
            and self.manager.revision == int(pending["manager_revision"])
            and self.current_height_mm == int(pending["height_mm"])
            and self.current_version_id == str(pending["version_id"])
        )

    def _update_pending_selected_playback(self) -> None:
        pending = self.pending_selected_playback
        if pending is None or self.no_sim:
            return
        if not self._selected_restore_transaction_is_current(pending):
            self._cancel_pending_selected_playback(reason="selected restore transaction was cancelled")
            return
        if time.monotonic() > float(pending["deadline"]):
            self._fail_pending_selected_playback("previous saved state restore timed out")
            return
        status = dict(self.latest_sim_status)
        matching_restore = (
            str(status.get("last_restore_request_id", "") or "") == str(pending["request_id"])
            and int(status.get("restore_count", 0) or 0) > int(pending["restore_count_before"])
        )
        if matching_restore and str(status.get("last_restore_result", "") or "") != "ok":
            error = str(status.get("last_restore_error", "") or status.get("error", "") or "unknown restore error")
            self._fail_pending_selected_playback("previous saved state restore failed: " + error)
            return
        if not matching_restore or not bool(status.get("runtime_ready", status.get("ready", False))):
            return
        if not pending["verification_requested"]:
            pending["trace"].append(
                {
                    "event": "restore_acknowledged",
                    "monotonic_s": time.monotonic(),
                    "request_id": pending["request_id"],
                    "restore_count": status.get("restore_count"),
                }
            )
            pending["verification_requested"] = True
            self.transport.request_state(detailed=True)
            pending["trace"].append({"event": "state_requested", "monotonic_s": time.monotonic()})
            return
        detailed = dict(getattr(self.sim_client, "latest_detailed_status", {}) or {})
        detailed_matches = (
            str(detailed.get("last_restore_request_id", "") or "") == str(pending["request_id"])
            and int(detailed.get("restore_count", 0) or 0) > int(pending["restore_count_before"])
            and str(detailed.get("last_restore_result", "") or "") == "ok"
            and bool(detailed.get("runtime_ready", detailed.get("ready", False)))
        )
        if not detailed_matches:
            return
        verified, verify_detail = self._verify_selected_restore_state(
            pending,
            dict(detailed.get("sim_state", {}) or {}),
        )
        if not verified:
            self._fail_pending_selected_playback("restored state verification failed: " + verify_detail)
            return
        pending["trace"].append(
            {"event": "state_verified", "monotonic_s": time.monotonic(), "detail": verify_detail}
        )
        self._start_pending_selected_playback()

    def _start_pending_selected_playback(self) -> bool:
        pending = self.pending_selected_playback
        if pending is None or not self._selected_restore_transaction_is_current(pending):
            return False
        index = int(pending["selected_step_index"])
        step = self.manager.get_step(index)
        try:
            plan = plan_from_steps(
                [step],
                profile=str(pending["profile"]),
                max_wheel_speed=self.max_wheel_speed,
                label=str(pending["label"]),
                sequence_total_steps=self.manager.count,
            )
        except Exception as exc:
            self._fail_pending_selected_playback(f"selected step plan failed after restore: {exc}")
            return False
        plan.selected_playback = True
        if not plan.events or any(int(event.source_step or 0) != index for event in plan.events):
            self._fail_pending_selected_playback(
                f"selected step plan is empty or contains events outside Step {index}"
            )
            return False
        plan.timing["selected_restore_source_step_index"] = pending["restore_source_step_index"]
        plan.timing["selected_restore_source_field"] = pending["restore_source_field"]
        plan.timing["selected_restore_request_id"] = pending["request_id"]
        plan.timing["selected_restore_trace"] = copy.deepcopy(pending["trace"])
        self.playback.progress.status_text = f"Restore complete. Starting selected Step {index}..."
        self.status = f"Restore complete. Starting selected Step {index}..."
        pending["trace"].append({"event": "playback_start_requested", "monotonic_s": time.monotonic()})
        plan.timing["selected_restore_trace"] = copy.deepcopy(pending["trace"])
        if self._use_worker_playback_scheduler():
            ok = self.playback.start_worker_plan(
                plan,
                start_delay_s=self.playback_pre_step_settle_s,
                operation_already_owned=True,
                operation_owner_id=str(pending["request_id"]),
            )
        else:
            ok = self.playback.start_plan(plan, start_delay_s=self.playback_pre_step_settle_s)
        result = {
            key: copy.deepcopy(value)
            for key, value in pending.items()
            if key not in {"selected_step", "restore_sim_state", "restore_command_state"}
        }
        result.update(
            {
                "started": bool(ok),
                "plan_event_count": len(plan.events),
                "plan_source_steps": sorted({int(event.source_step or 0) for event in plan.events}),
                "plan_selected_playback": plan.selected_playback,
            }
        )
        self.last_selected_restore_result = result
        self.pending_selected_playback = None
        if not ok:
            self._fail_pending_selected_playback(
                self.playback.last_error or "worker rejected selected playback start",
                pending_override=pending,
            )
            return False
        self.mode = MODE_PLAYBACK
        self._info(
            f"[INFO] Restore complete from Step {pending['restore_source_step_index']}."
            f"{pending['restore_source_field']}; starting only Selected Step {index}."
        )
        return True

    def _fail_pending_selected_playback(
        self,
        reason: str,
        *,
        pending_override: dict[str, Any] | None = None,
    ) -> None:
        pending = pending_override or self.pending_selected_playback or {}
        index = int(pending.get("selected_step_index", self.selected_step_index or 0) or 0)
        self.pending_selected_playback = None
        self.playback.active = False
        self.playback.start_requested = False
        self.playback.scheduled_start_at = 0.0
        self.playback.worker_managed = False
        self.playback.plan = None
        self.playback.last_error = str(reason)
        self.playback.last_info = str(reason)
        self.playback.progress.playback_state = PlaybackState.ERROR.value
        self.playback.progress.status_text = "Error"
        self.playback.progress.current_step_index = index
        self.playback.progress.total_steps = self.manager.count
        self.playback.progress.selected_playback = True
        self.playback.progress.command_phase = "restore_failed"
        self.playback.progress.last_error = str(reason)
        self.operation.finish(OperationState.PLAYBACK)
        self.mode = MODE_TEST
        self.last_selected_restore_result = {
            "selected_step_index": index,
            "restore_source_step_index": pending.get("restore_source_step_index"),
            "restore_source_field": pending.get("restore_source_field", ""),
            "request_id": pending.get("request_id", ""),
            "started": False,
            "error": str(reason),
            "trace": copy.deepcopy(pending.get("trace", [])),
        }
        message = f"Selected Step {index} was not started: {reason}."
        self.detail_text = message
        self._warn("[ERROR] " + message)

    def _cancel_pending_selected_playback(self, *, reason: str = "stopped during restore") -> bool:
        pending = self.pending_selected_playback
        if pending is None:
            return False
        pending["trace"].append({"event": "restore_cancelled", "monotonic_s": time.monotonic(), "reason": reason})
        self.last_selected_restore_result = {
            "selected_step_index": pending.get("selected_step_index"),
            "restore_source_step_index": pending.get("restore_source_step_index"),
            "restore_source_field": pending.get("restore_source_field", ""),
            "request_id": pending.get("request_id", ""),
            "started": False,
            "cancelled": True,
            "trace": copy.deepcopy(pending.get("trace", [])),
        }
        self.pending_selected_playback = None
        self.playback.active = False
        self.playback.start_requested = False
        self.playback.scheduled_start_at = 0.0
        self.playback.plan = None
        self.playback.progress.playback_state = PlaybackState.IDLE.value
        self.playback.progress.status_text = "Stopped"
        self.playback.progress.command_phase = "restore_cancelled"
        self.operation.finish(OperationState.PLAYBACK)
        self.mode = MODE_TEST
        self.stop_wheels(reason="selected_restore_cancelled")
        self._info(f"[INFO] Selected playback restore cancelled: {reason}; no plan was started.")
        return True

    def _handle_play_to_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: play_to_step <index> [fast|raw]")
            return
        ok, reason = self.can_playback()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        index = int(args[0])
        profile = "fast" if "fast" in [arg.lower() for arg in args[1:]] else self.playback.profile
        steps = self.get_visible_steps()
        self.start_playback(steps[:index], label=f"{self.current_height_mm}mm steps 1..{index}", profile=profile)

    def start_playback(
        self,
        steps: list[dict[str, Any]],
        *,
        label: str,
        profile: str | None = None,
        restore_start_state: bool = False,
        respawn_first: bool = False,
    ) -> bool:
        button_clicked_time = time.monotonic()
        if profile is not None:
            self.playback.set_profile(profile)
        try:
            plan = plan_from_steps(
                steps,
                profile=self.playback.profile,
                max_wheel_speed=self.max_wheel_speed,
                label=label,
                sequence_total_steps=self.manager.count or len(steps),
            )
            plan.timing["button_clicked_time"] = button_clicked_time
        except Exception as exc:
            self.playback.last_error = str(exc)
            self._warn(f"[WARN] Playback plan failed: {exc}")
            return False
        first_commands = [event.command for event in plan.events[:5]]
        self._info(
            "[PLAYBACK DEBUG] plan "
            f"label={label} events={len(plan.events)} final_time={plan.final_time_s:.3f}s "
            f"profile={self.playback.profile} "
            f"first_commands={first_commands}"
        )
        if not plan.events:
            self.playback.start_plan(plan)
            step_index = "unknown"
            if len(steps) == 1:
                try:
                    step_index = f"{int(normalize_step(steps[0]).get('index', 0)):03d}"
                except Exception:
                    step_index = "unknown"
            message = f"Selected step {step_index} has no motion events to play."
            self.playback.last_error = message
            self.playback.last_info = message
            self.detail_text = message
            self._warn("[WARN] " + message)
            return False
        ok, reason = self.playback_readiness(respawn_first=respawn_first)
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        motion_ok, motion_reason = self.motion_readiness()
        if respawn_first and not self.no_sim and not motion_ok:
            suffix = f": {motion_reason}" if motion_reason else "."
            self._info("[INFO] Playback will Respawn from the validated ground reference before replay" + suffix)
        self.stop_wheels(reason="playback_start_boundary")
        self._info("[PLAYBACK DEBUG] stop_wheels sent before playback restore/start.")
        pre_start_delay_s = 0.0
        restore_started = 0.0
        if respawn_first:
            self.playback.progress.playback_state = PlaybackState.RESTORING.value
            self.playback.progress.status_text = "Restoring start state..."
            restore_started = time.monotonic()
            if not self.respawn_robot(source="playback"):
                self.operation.finish(OperationState.RESPAWNING)
                self.playback.last_error = "Respawn failed before playback."
                return False
            pre_start_delay_s = max(pre_start_delay_s, self.respawn_play_settle_s)
        if restore_start_state and steps:
            self.playback.progress.playback_state = PlaybackState.RESTORING.value
            self.playback.progress.status_text = "Restoring start state..."
            restore_started = restore_started or time.monotonic()
            if self.apply_step_start_state(steps[0]):
                pre_start_delay_s = max(pre_start_delay_s, self.playback_pre_step_settle_s)
        if restore_started:
            plan.timing["restore_start"] = restore_started
            plan.timing["restore_end"] = time.monotonic()
        if self._use_worker_playback_scheduler():
            ok = self.playback.start_worker_plan(plan, start_delay_s=pre_start_delay_s)
        else:
            ok = self.playback.start_plan(plan, start_delay_s=pre_start_delay_s)
        if ok:
            self.mode = MODE_PLAYBACK
            if self.playback.worker_managed:
                self._info(
                    f"[INFO] Playback start requested: {label}; waiting for explicit worker acceptance "
                    f"request={self.playback.worker_request_id[:8]} plan={self.playback.worker_plan_id}."
                )
            elif pre_start_delay_s > 0.0:
                self._info(f"[INFO] Playback scheduled in {pre_start_delay_s:.2f}s: {label}")
            else:
                self._info(f"[INFO] Playback started: {label}")
            debug = (
                "[PLAYBACK DEBUG] manager "
                f"active={self.playback.active} scheduled={self.playback.scheduled_start_at > 0.0} "
                f"index={self.playback.index} count={len(plan.events)} "
                f"last_info={self.playback.last_info}"
            )
            self.status_log.append(debug)
            print(debug)
        else:
            self._warn(f"[WARN] Playback failed: {self.playback.last_error or 'unknown reason'}")
        return ok

    def _use_worker_playback_scheduler(self) -> bool:
        if self.no_sim:
            return False
        return bool(getattr(self.transport, "process_client", None) is not None or getattr(self.transport, "worker", None) is not None)

    def apply_step_start_state(self, step: dict[str, Any]) -> bool:
        normalized = normalize_step(step)
        index = int(normalized.get("index", 0))
        sim_state = normalized.get("sim_state_before")
        command_state = normalized.get("command_state_before")
        raw_has_command_state = isinstance(step, dict) and any(
            key in step for key in ("command_state_before", "state_before", "robot_state_before")
        )
        has_sim_state = isinstance(sim_state, dict) and bool(sim_state)
        has_command_state = raw_has_command_state and isinstance(command_state, dict) and bool(command_state)
        self._info(
            "[PLAYBACK DEBUG] start-state restore "
            f"step={index:03d} restore_full_sim_pose={self.restore_full_sim_pose_if_available} "
            f"fallback_command_state={self.fallback_to_command_state_before} "
            f"sim_state_before={has_sim_state} command_state_before={has_command_state}"
        )
        if self.restore_full_sim_pose_if_available and isinstance(sim_state, dict):
            try:
                self.transport.restore_sim_state(sim_state)
                self.stop_wheels(reason="restore_start_state")
                self.transport.request_state()
                self.detail_text = f"Restored sim_state_before for step {index:03d} before playback.\n\n" + self.compact_step_details(normalized)
                self._info(f"[INFO] Restored sim_state_before for step {index:03d} before playback.")
                return True
            except Exception as exc:
                self._warn(f"[WARN] restore_sim_state failed for step {index:03d}: {exc}")
        if not self.fallback_to_command_state_before:
            self._warn(f"[WARN] No sim_state_before saved for step {index:03d}; fallback disabled.")
            return False
        if not has_command_state:
            self._warn(f"[WARN] No start state saved for step {index:03d}; playing from current robot state.")
            return False
        if self.restore_full_sim_pose_if_available:
            self._warn(f"[WARN] No sim_state_before saved for this step; using command_state_before only.")
        self.apply_command_state(command_state, keep_wheels=False)
        self.detail_text = f"Applied command_state_before for step {index:03d} before playback.\n\n" + self.compact_step_details(normalized)
        self._info(f"[INFO] Applied command_state_before for step {index:03d} before playback.")
        return True

    def playback_debug_selected(self) -> None:
        index = self.selected_step_index
        lines = [
            "Playback selected debug",
            f"controller_selected_index={index}",
            f"manager_count={self.manager.count}",
            f"can_playback={self.can_playback()}",
            f"mode={self.mode} sim_ready={self.sim_ready} no_sim={self.no_sim}",
            f"playback_status={self.playback.status_dict()}",
        ]
        if index is not None:
            try:
                step = normalize_step(self.get_visible_step(index), index=index)
                plan = plan_from_steps(
                    [step],
                    profile=self.playback.profile,
                    max_wheel_speed=self.max_wheel_speed,
                    label=f"{self.current_height_mm}mm step {index:03d}",
                    sequence_total_steps=self.manager.count,
                )
                lines.extend(
                    [
                        "",
                        self.compact_step_details(step, title=f"Step {index:03d} summary"),
                        "",
                        f"plan_count={len(plan.events)} final_time={plan.final_time_s:.3f}s",
                        f"plan_first_commands={[event.command for event in plan.events[:5]]}",
                    ]
                )
            except Exception as exc:
                lines.append(f"selected_step_error={exc}")
        self.detail_text = "\n".join(lines)
        self._info("[PLAYBACK DEBUG] " + self.detail_text.replace("\n", " | "))

    def _handle_combine(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "mode"
        if sub == "mode":
            self.combine_mode_enabled = True
            self._info("[INFO] Combine mode enabled. Select contiguous steps, then Combine Selected Steps.")
        elif sub == "cancel":
            self.combine_mode_enabled = False
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self._info("[INFO] Combine mode cancelled.")
        elif sub == "clear":
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
        elif sub == "add":
            self.add_to_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub == "remove":
            self.remove_from_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub == "toggle":
            self.toggle_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub in {"range", "select_range", "contiguous"}:
            self.select_contiguous_combine_range()
        elif sub == "preview":
            self.preview_combine_steps()
        elif sub in {"commit", "selected"}:
            self.commit_combine_steps()

    def set_combine_selection(self, indices: list[int]) -> None:
        self.combine_selected_indices = set(int(index) for index in indices)
        if self.combine_mode_enabled:
            self.combine_preview_text = self.combine_selection_status_text()

    def add_to_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            if 1 <= int(index) <= self.manager.count:
                self.combine_selected_indices.add(int(index))
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def remove_from_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            self.combine_selected_indices.discard(int(index))
        self.combine_preview_text = self.combine_selection_status_text()

    def toggle_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            index = int(index)
            if index in self.combine_selected_indices:
                self.combine_selected_indices.remove(index)
            elif 1 <= index <= self.manager.count:
                self.combine_selected_indices.add(index)
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def select_contiguous_combine_range(self) -> None:
        selected = sorted(self.combine_selected_indices)
        if len(selected) < 2:
            self.combine_preview_text = "Select at least two steps before selecting a contiguous range."
            return
        self.combine_selected_indices = set(range(selected[0], selected[-1] + 1))
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def combine_selection_status_text(self) -> str:
        selected = sorted(self.combine_selected_indices)
        contiguous = self.combine_selection_is_contiguous()
        return (
            f"Selected combine indices: {selected}\n"
            f"Selected count: {len(selected)}\n"
            f"Contiguous: {contiguous}\n"
            + (
                "Click Preview Combined Step to compute a compact preview."
                if len(selected) >= 2 and contiguous
                else "Combine selection must be contiguous. Use Select Contiguous Range."
                if len(selected) >= 2
                else "Select at least two steps."
            )
        )

    def preview_combine_steps(self, *, silent: bool = False) -> None:
        selected = sorted(self.combine_selected_indices)
        if len(selected) < 2:
            self.combine_preview_text = "Select at least two steps."
            return
        ok, reason = self.can_combine()
        if not ok:
            self.combine_preview_text = reason
            if not silent:
                self._warn("[WARN] " + reason)
            return
        try:
            steps = [self.manager.get_step(index) for index in selected]
            from sequence_model import build_combined_step

            combined = build_combined_step(steps, allow_conflicts=self.allow_combine_conflicts)
            self.combine_preview_text = self.compact_step_details(combined, title=f"Combined preview {selected[0]:03d}..{selected[-1]:03d}")
            if not silent:
                self._info(f"[INFO] Combine preview ready for steps {selected}.")
        except Exception as exc:
            self.combine_preview_text = f"Combine preview failed: {exc}"
            if not silent:
                self._warn("[WARN] " + self.combine_preview_text)

    def commit_combine_steps(self) -> None:
        selected = sorted(self.combine_selected_indices)
        ok, reason = self.can_combine()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any]:
            combined = self.manager.replace_step_range_with_combined(selected, allow_conflicts=self.allow_combine_conflicts)
            self.selected_step_index = int(combined["index"])
            self.set_step_summary_detail(combined, title=f"Combined step {int(combined['index']):03d}")
            self.combine_mode_enabled = False
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self._info(f"[INFO] Combined steps {selected} into step {int(combined['index']):03d}.")
            return combined

        self._run_operation("Combine selected steps", work)

    def export_motion_txt(self) -> Path | None:
        if self.manager.count == 0:
            self._warn("[WARN] No accepted steps to export.")
            return None
        path = self.store.height_dir(self.current_height_mm) / "accepted_motion.txt"
        lines: list[str] = []
        for step in self.manager.steps:
            normalized = normalize_step(step)
            lines.append(f"# step {int(normalized['index']):03d} {normalized['name']}")
            for event in normalized.get("events", []):
                command = str(event.get("command", "")).strip()
                if command:
                    lines.append(f"at {float(event.get('time', 0.0)):.3f} {command}")
            lines.append(f"wait {float(normalized.get('duration', 0.0)):.3f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._info(f"[INFO] Exported motion TXT: {path}")
        return path

    def _info(self, text: str) -> None:
        self.status = text
        self.status_log.append(text)
        print(text)

    def _warn(self, text: str) -> None:
        self.status = text
        self.status_log.append(text)
        print(text)


class RealRobotStyleHeightReplayUi:
    """Tk UI that mirrors the real_robot_ui_controller layout."""

    def __init__(self, controller: HeightReplayController, *, smoke_test_ms: int = 0):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.controller = controller
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("Height Based Obstacle Replay - Isaac Sim")
        self.root.protocol("WM_DELETE_WINDOW", self._window_close)
        self.root.bind("<space>", lambda _event: self._post("e_stop") or "break")
        self.root.bind("<Control-s>", self._save_shortcut)
        self.root.bind("<Control-S>", self._save_shortcut)
        self.updating = False
        self.slider_dragging: set[str] = set()
        self.slider_pending: dict[str, tuple[str, float, float, bool]] = {}
        self.slider_after_ids: dict[str, Any] = {}
        self.slider_last_sent: dict[str, tuple[float, float]] = {}
        self.selected_step_index: int | None = None
        self.refreshing = False
        self._last_playback_guard_reason = ""
        self.ui_refresh_ms = max(20, int(getattr(controller.args, "ui_refresh_ms", 100)))
        self.sim_status_refresh_ms = max(50, int(getattr(controller.args, "sim_status_refresh_ms", 250)))
        self.full_refresh_ms = max(250, int(getattr(controller.args, "full_refresh_ms", 1000)))
        self.max_text_widget_chars = max(1000, int(getattr(controller.args, "max_text_widget_chars", 200000)))
        self.disable_auto_sim_state_json = bool(getattr(controller.args, "disable_auto_sim_state_json", False))
        self.sim_state_json_on_demand = bool(getattr(controller.args, "sim_state_json_on_demand", True))
        self._last_medium_refresh = 0.0
        self._last_full_refresh = 0.0
        self._last_sequence_revision: int | None = None
        self._last_manifest_revision: int | None = None
        self._last_detail_text = ""
        self._last_combine_preview = ""
        self._text_widget_cache: dict[int, tuple[int, int]] = {}
        self._guarded_buttons: list[tuple[Any, str]] = []
        self._guarded_button_state: dict[int, str] = {}
        self._poll_after_id: Any | None = None
        self._smoke_after_ids: list[Any] = []
        self._closing = False
        self.wheel_stop_latched = False
        self.save_in_progress = False

        self.servo_vars: dict[str, Any] = {}
        self.servo_output_label_vars: dict[str, Any] = {}
        self.wheel_vars: dict[str, Any] = {}
        self.playback_buttons: list[Any] = []
        self.playback_buttons_by_label: dict[str, Any] = {}
        self.last_selected_fast_click_id = ""
        self.last_selected_fast_click_started = 0.0
        self.last_selected_fast_feedback_ms = 0.0
        self.selected_fast_feedback_pending = False
        self.pending_selected_fast_feedback_text = ""
        self.selected_fast_click_trace: list[dict[str, Any]] = []

        self.summary_var = tk.StringVar(value=controller.status_line())
        self.sim_label_var = tk.StringVar(value="Isaac Sim: starting")
        self.mode_label_var = tk.StringVar(value="Mode: TEST")
        self.sequence_label_var = tk.StringVar(value="")
        self.accepted_steps_source_var = tk.StringVar(value="")
        self.accepted_steps_path_var = tk.StringVar(value=str(controller.manager.accepted_path or controller._current_steps_path()))
        self.save_status_var = tk.StringVar(value="No unsaved changes.")
        self.playback_label_var = tk.StringVar(value="Playback: inactive")
        self.playback_unavailable_var = tk.StringVar(value="")
        self.record_label_var = tk.StringVar(value="Record: idle")
        self.servo_wheel_mode_var = tk.StringVar(value="Servo-Wheel Mode: inactive")
        self.servo_wheel_preview_var = tk.StringVar(value="Staged/live delta: none")
        self.servo_wheel_launch_var = tk.StringVar(value="Last launch: none")
        self.recording_path_var = tk.StringVar(value=f"Recording output: {controller._current_steps_path()}")
        self.baseline_status_var = tk.StringVar(value="Recording baseline: not validated")
        self.busy_label_var = tk.StringVar(value="")
        self.pending_label_var = tk.StringVar(value="Pending: none")
        self.replacement_label_var = tk.StringVar(value="Replacement: none")
        self.wheel_status_var = tk.StringVar(value="Wheel command: fl=0 fr=0 rl=0 rr=0")
        self.measured_wheel_status_var = tk.StringVar(value="Measured wheel velocity: unavailable")
        self.command_var = tk.StringVar(value="")
        default_speed = min(0.30, float(controller.default_wheel_speed))
        self.left_speed_var = tk.StringVar(value=f"{default_speed:.2f}")
        self.right_speed_var = tk.StringVar(value=f"{default_speed:.2f}")
        self.height_var = tk.StringVar(value=f"{controller.current_height_mm} mm")
        self.motion_profile_var = tk.StringVar(value="Motion profile: Fixed 100%")
        self.play_profile_var = tk.StringVar(value=str(controller.playback.profile))
        self.restore_step_start_var = tk.BooleanVar(value=True)
        self.restore_full_sim_pose_var = tk.BooleanVar(value=controller.restore_full_sim_pose_if_available)
        self.fallback_command_state_var = tk.BooleanVar(value=controller.fallback_to_command_state_before)
        self._last_playback_highlight_step = 0
        self.clear_confirm_var = tk.BooleanVar(value=False)
        self.combine_allow_conflicts_var = tk.BooleanVar(value=False)
        self.combine_selection_var = tk.StringVar(value="Selected combine indices: []")
        self.quick_hint_var = tk.StringVar(value="")
        self._build()
        self._poll()
        if bool(getattr(controller.args, "smoke_accept_recording", False)):
            self._smoke_after_ids.append(self.root.after(120, self._smoke_accept_recording))
        if smoke_test_ms > 0:
            self._smoke_after_ids.append(self.root.after(smoke_test_ms, lambda: self._window_close(force=True)))

    def run(self) -> None:
        self.root.mainloop()

    def _window_close(self, *, force: bool = False) -> None:
        if self._closing:
            return
        if not force and not self._resolve_unsaved_changes("closing the UI"):
            return
        self._closing = True
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        for after_id in list(self.slider_after_ids.values()):
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self.slider_after_ids.clear()
        for after_id in self._smoke_after_ids:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        self._smoke_after_ids.clear()
        self.controller.shutdown()
        self.root.destroy()

    def _post(self, text: str, *, kind: str = "command", target: str = "", quiet: bool = False) -> None:
        self.busy_label_var.set("Busy..." if self.controller.operation_busy else "")
        if text.strip().split()[:1] == ["playback_debug_selected"]:
            try:
                self.controller._info(
                    "[PLAYBACK DEBUG] ui selected "
                    f"tree_selection={list(self.steps_tree.selection())} "
                    f"tree_indices={self._selected_indices()} "
                    f"controller_selected={self.controller.selected_step_index} "
                    f"manager_count={self.controller.manager.count}"
                )
            except Exception as exc:
                self.controller._warn(f"[WARN] UI selected debug failed: {exc}")
        self.controller.handle_command(CommandMessage(text=text, source="ui", kind=kind, target=target, quiet=quiet))
        self._refresh(force=False)

    def _smoke_accept_recording(self) -> None:
        for command in [
            "step_record start",
            "servo front_left_hip 5",
            "wheel fl 0.1",
            "step_record stop",
            "step_record accept",
        ]:
            self.controller.handle_command(CommandMessage(text=command, source="ui_smoke", quiet=True))
        if self.controller.manager.count < 1:
            self.controller._warn("[WARN] Smoke accept recording did not create an accepted step.")
        else:
            self.controller._info("[INFO] Smoke accept recording created one accepted step.")
        self._refresh(force=False)

    def _post_slider(self, key: str, text: str, value: float, threshold: float, *, final: bool = False) -> None:
        now = time.monotonic()
        last = self.slider_last_sent.get(key)
        if not final and last is not None:
            last_time, last_value = last
            if abs(value - last_value) < threshold and now - last_time < 0.05:
                self.slider_pending[key] = (text, value, threshold, final)
                return
        self.slider_last_sent[key] = (now, value)
        self.controller.handle_command(CommandMessage(text=text, source="ui", kind="slider", target=key, log_history=final, quiet=True))

    def _schedule_slider(self, key: str, text: str, value: float, threshold: float) -> None:
        self.slider_pending[key] = (text, value, threshold, False)
        if key not in self.slider_after_ids:
            self.slider_after_ids[key] = self.root.after(40, lambda slider_key=key: self._flush_slider(slider_key))

    def _flush_slider(self, key: str, *, final: bool = False) -> None:
        after_id = self.slider_after_ids.pop(key, None)
        if after_id is not None and final:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        pending = self.slider_pending.pop(key, None)
        if pending is None:
            return
        text, value, threshold, _was_final = pending
        self._post_slider(key, text, value, threshold, final=final)

    def _slider_release(self, key: str) -> None:
        self.slider_dragging.discard(key)
        self._flush_slider(key, final=True)

    def _build(self) -> None:
        self.root.minsize(1500, 860)
        self.root.geometry("1700x980")
        container = self.ttk.Frame(self.root, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        summary = self.ttk.Frame(container)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for col, var in enumerate(
            [
                self.sim_label_var,
                self.mode_label_var,
                self.sequence_label_var,
                self.playback_label_var,
                self.record_label_var,
                self.busy_label_var,
            ]
        ):
            self.ttk.Label(summary, textvariable=var).grid(row=0, column=col, sticky="w", padx=(0, 14))
        self.ttk.Label(summary, textvariable=self.summary_var).grid(row=1, column=0, columnspan=7, sticky="ew")
        summary.columnconfigure(6, weight=1)

        main = self.ttk.PanedWindow(container, orient="horizontal")
        self.main_paned = main
        main.grid(row=1, column=0, sticky="nsew")
        left = self._make_scroll_column(main, width=455)
        center = self._make_scroll_column(main, width=650)
        right = self.ttk.Frame(main, width=620)
        self._add_pane(main, right, weight=3)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_live_column(left)
        self._build_steps_column(center)
        self._build_right_notebook(right)

    def _add_pane(self, paned: Any, child: Any, *, weight: int) -> None:
        try:
            paned.add(child, weight=weight)
        except Exception:
            paned.add(child)

    def _make_scroll_column(self, parent: Any, *, width: int) -> Any:
        shell = self.ttk.Frame(parent, width=width)
        self._add_pane(parent, shell, weight=2)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(shell, highlightthickness=0)
        ybar = self.ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        xbar = self.ttk.Scrollbar(shell, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        body = self.ttk.Frame(canvas, padding=(0, 0, 4, 0))
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def sync(_event: Any | None = None) -> None:
            canvas.itemconfigure(window_id, width=max(canvas.winfo_width(), body.winfo_reqwidth()))
            canvas.configure(scrollregion=canvas.bbox("all"))

        body.bind("<Configure>", sync)
        canvas.bind("<Configure>", sync)
        canvas.bind("<MouseWheel>", lambda event, c=canvas: self._scroll_canvas(c, event))
        body.columnconfigure(0, weight=1)
        return body

    def _register_guarded_button(self, button: Any, guard: str) -> Any:
        self._guarded_buttons.append((button, guard))
        return button

    def _button_allowed(self, guard: str, *, availability: PlaybackAvailability | None = None, selected_index: int | None = None) -> bool:
        selected = self._selected_index() if selected_index is None else selected_index
        availability = availability or self.controller.playback_availability(selected_index=selected)
        if guard == "e_stop":
            return True
        if guard == "manual_respawn":
            return (
                (self.controller.no_sim or self.controller.runtime_ready)
                and self.controller.operation.idle
                and not self.controller.recording_active
            )
        if guard == "start_record":
            return self.controller.can_start_recording()[0]
        if guard == "accept_record":
            return self.controller.can_accept_recorded_step()[0]
        if guard == "prepare_replace":
            return self.controller.can_prepare_replacement()[0] and selected is not None
        if guard == "accept_replace":
            return self.controller.can_accept_replacement()[0]
        if guard == "play_all":
            return availability.can_start
        if guard == "respawn_play":
            return availability.can_respawn_start
        if guard == "play_selected":
            return availability.can_play_selected
        if guard == "respawn_selected":
            return availability.can_respawn_start and selected is not None
        if guard == "pause_play":
            return availability.can_pause
        if guard == "resume_play":
            return availability.can_resume
        if guard == "stop_play":
            return availability.can_stop
        if guard == "analyze_playback":
            return availability.can_analyze
        if guard == "export_motion":
            return availability.can_export
        if guard == "debug_selected":
            return availability.can_analyze and selected is not None
        if guard == "combine":
            return self.controller.can_combine()[0]
        if guard == "save":
            return self.controller.manager.dirty and self.controller.can_save()[0]
        if guard == "show_step":
            return selected is not None and not self.controller.recording_active and not self.controller.playback.active
        if guard == "height_edit":
            return self.controller.operation.state not in {
                OperationState.RECORDING,
                OperationState.PLAYBACK,
                OperationState.RESPAWNING,
            }
        if guard == "idle":
            return self.controller.operation.idle
        return True

    def _button_guard_reason(self, guard: str, *, availability: PlaybackAvailability | None = None, selected_index: int | None = None) -> str:
        selected = self._selected_index() if selected_index is None else selected_index
        availability = availability or self.controller.playback_availability(selected_index=selected)
        if guard in {
            "play_all",
            "respawn_play",
            "play_selected",
            "respawn_selected",
            "pause_play",
            "resume_play",
            "stop_play",
            "analyze_playback",
            "export_motion",
            "debug_selected",
        }:
            if guard in {"play_selected", "respawn_selected", "debug_selected"} and selected is None:
                return "Select an accepted step first."
            return availability.reason
        if guard == "manual_respawn":
            if not self.controller.no_sim and not self.controller.runtime_ready:
                return "Simulation worker is not ready."
            return self.controller.operation.reason
        if guard == "start_record":
            return self.controller.can_start_recording()[1]
        if guard == "accept_record":
            return self.controller.can_accept_recorded_step()[1]
        if guard == "prepare_replace":
            ok, reason = self.controller.can_prepare_replacement()
            return "Select an accepted step first." if ok and selected is None else reason
        if guard == "accept_replace":
            return self.controller.can_accept_replacement()[1]
        if guard == "combine":
            return self.controller.can_combine()[1]
        if guard == "save":
            return "No unsaved changes." if not self.controller.manager.dirty else self.controller.can_save()[1]
        if guard == "show_step" and selected is None:
            return "Select an accepted step first."
        if guard == "idle":
            return self.controller.operation.reason
        return ""

    def _refresh_button_states(self) -> None:
        playback_guard_reason = ""
        selected = self._selected_index()
        availability = self.controller.playback_availability(selected_index=selected)
        for button, guard in self._guarded_buttons:
            try:
                allowed = self._button_allowed(guard, availability=availability, selected_index=selected)
                state = "normal" if allowed else "disabled"
                if self._guarded_button_state.get(id(button)) != state:
                    button.configure(state=state)
                    self._guarded_button_state[id(button)] = state
                if guard in {"play_all", "respawn_play", "play_selected", "respawn_selected"} and not allowed and not playback_guard_reason:
                    playback_guard_reason = self._button_guard_reason(guard, availability=availability, selected_index=selected)
            except Exception as exc:
                self.controller._warn(f"[WARN] Button state refresh failed for {guard}: {exc}\n{traceback.format_exc()}")
        self._last_playback_guard_reason = playback_guard_reason

    def _build_live_column(self, parent: Any) -> None:
        servo_frame = self.ttk.LabelFrame(parent, text="Servos")
        wheel_frame = self.ttk.LabelFrame(parent, text="Wheels")
        quick_frame = self.ttk.LabelFrame(parent, text="Quick Commands")
        servo_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        wheel_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        quick_frame.grid(row=2, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        self._build_servo_panel(servo_frame, slider_length=220)
        self._build_wheel_panel(wheel_frame, slider_length=220)
        self._build_quick_commands(quick_frame)

    def _build_servo_panel(self, parent: Any, *, slider_length: int) -> None:
        for row, name in enumerate(SERVO_JOINT_NAMES):
            low, high = KNEE_LIMIT_DEG if name in KNEE_JOINT_NAMES else HIP_LIMIT_DEG
            var = self.tk.DoubleVar(value=0.0)
            out_var = self.tk.StringVar(value="out 0.0")
            self.servo_vars[name] = var
            self.servo_output_label_vars[name] = out_var
            self.ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            scale = self.tk.Scale(
                parent,
                from_=low,
                to=high,
                orient="horizontal",
                resolution=0.1,
                length=slider_length,
                variable=var,
                command=lambda value, joint=name: self._servo_slider(joint, value),
            )
            scale.grid(row=row, column=1, sticky="ew", pady=1)
            self.ttk.Label(parent, textvariable=out_var, width=14).grid(row=row, column=2, sticky="w", padx=(4, 2), pady=1)
            scale.bind("<ButtonPress-1>", lambda _event, key=name: self.slider_dragging.add(key))
            scale.bind("<ButtonRelease-1>", lambda _event, key=name: self._slider_release(key))
            scale.bind("<MouseWheel>", self._on_scale_mousewheel)
        parent.columnconfigure(1, weight=1)

    def _build_wheel_panel(self, parent: Any, *, slider_length: int) -> None:
        for row, short_name in enumerate(("fl", "fr", "rl", "rr")):
            var = self.tk.DoubleVar(value=0.0)
            self.wheel_vars[short_name] = var
            self.ttk.Label(parent, text=f"{short_name} wheel").grid(row=row, column=0, sticky="w", padx=4, pady=1)
            scale = self.tk.Scale(
                parent,
                from_=-float(self.controller.max_wheel_speed),
                to=float(self.controller.max_wheel_speed),
                orient="horizontal",
                resolution=0.01,
                length=slider_length,
                variable=var,
                command=lambda value, wheel=short_name: self._wheel_slider(wheel, value),
            )
            scale.grid(row=row, column=1, sticky="ew", pady=1)
            scale.bind("<ButtonPress-1>", lambda _event, key=short_name: self._begin_wheel_drag(key))
            scale.bind("<ButtonRelease-1>", lambda _event, key=short_name: self._slider_release(key))
            scale.bind("<MouseWheel>", self._on_scale_mousewheel)
        parent.columnconfigure(1, weight=1)
        self.ttk.Button(parent, text="Stop Wheels", command=self._stop_wheels_ui).grid(row=5, column=0, sticky="ew", padx=3, pady=(10, 2))
        self.ttk.Button(parent, text="All Forward", command=lambda: self._post(f"wheel all {float(self.left_speed_var.get()):.3f}")).grid(row=5, column=1, sticky="ew", padx=3, pady=(10, 2))
        self.ttk.Button(parent, text="All Backward", command=lambda: self._post(f"wheel all {-float(self.left_speed_var.get()):.3f}")).grid(row=6, column=1, sticky="ew", padx=3, pady=2)
        self.ttk.Label(parent, text="left").grid(row=7, column=0, sticky="e")
        self.ttk.Entry(parent, textvariable=self.left_speed_var, width=8).grid(row=7, column=1, sticky="w")
        self.ttk.Label(parent, text="right").grid(row=8, column=0, sticky="e")
        self.ttk.Entry(parent, textvariable=self.right_speed_var, width=8).grid(row=8, column=1, sticky="w")
        self.ttk.Button(parent, text="Set Pair", command=lambda: self._post(f"wheel {float(self.left_speed_var.get()):.3f} {float(self.right_speed_var.get()):.3f}")).grid(row=9, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.ttk.Label(parent, textvariable=self.wheel_status_var, wraplength=slider_length + 160, foreground="#8a3b00").grid(row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 2))
        self.ttk.Label(parent, textvariable=self.measured_wheel_status_var, wraplength=slider_length + 160).grid(row=11, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 2))

    def _build_quick_commands(self, parent: Any) -> None:
        self.ttk.Entry(parent, textvariable=self.command_var).grid(row=0, column=0, columnspan=4, sticky="ew", padx=3, pady=3)
        self.ttk.Button(parent, text="Run Command", command=self._run_command).grid(row=0, column=4, sticky="ew", padx=3, pady=3)
        buttons = [
            ("TEST Mode", "mode test", ""),
            ("Home", "home", ""),
            ("Status", "status", ""),
            ("E-stop", "e_stop", "e_stop"),
            ("↻ Respawn", "respawn", "manual_respawn"),
        ]
        for col, (label, command, guard) in enumerate(buttons):
            button = self.ttk.Button(
                parent,
                text=label,
                command=lambda text=command: self._post(text),
            )
            if label.endswith("Respawn"):
                button.configure(command=lambda text=command: self._post(text))
                button.bind("<Enter>", lambda _event: self.quick_hint_var.set("Respawn robot to its initial simulation pose. This does not clear E-stop."))
                button.bind("<Leave>", lambda _event: self.quick_hint_var.set(""))
            if guard:
                self._register_guarded_button(button, guard)
            button.grid(row=1, column=col, sticky="ew", padx=3, pady=3)
        self.ttk.Label(parent, textvariable=self.quick_hint_var, wraplength=420).grid(row=2, column=0, columnspan=5, sticky="ew", padx=3, pady=(1, 3))
        for col in range(5):
            parent.columnconfigure(col, weight=1, uniform="quick")

    def _build_steps_column(self, parent: Any) -> None:
        steps_frame = self.ttk.LabelFrame(parent, text="Accepted Steps")
        actions_frame = self.ttk.LabelFrame(parent, text="Accepted Step Actions")
        details_frame = self.ttk.LabelFrame(parent, text="Step Details")
        steps_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        actions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        details_frame.grid(row=2, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(2, weight=2)
        self._build_steps_tree(steps_frame)
        self._build_step_actions(actions_frame)
        self.detail_text = self._make_scrolled_text(details_frame, row=0, height=13, width=74)

    def _build_steps_tree(self, parent: Any) -> None:
        columns = ("index", "name", "type", "duration", "events", "note")
        self.ttk.Label(parent, textvariable=self.accepted_steps_source_var).grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=(0, 3))
        self.ttk.Label(parent, textvariable=self.accepted_steps_path_var, wraplength=680).grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        save_row = self.ttk.Frame(parent)
        save_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.save_modified_steps_button = self.ttk.Button(save_row, text="💾 Save New Version", command=self._save_new_version_async)
        self._register_guarded_button(self.save_modified_steps_button, "save")
        self.save_modified_steps_button.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.ttk.Label(save_row, textvariable=self.save_status_var, wraplength=500).grid(row=0, column=1, sticky="ew")
        save_row.columnconfigure(1, weight=1)
        self.steps_tree = self.ttk.Treeview(parent, columns=columns, show="headings", selectmode="extended", height=15)
        headings = {"index": "#", "name": "name", "type": "type", "duration": "duration", "events": "events count", "note": "note"}
        widths = {"index": 44, "name": 190, "type": 80, "duration": 76, "events": 88, "note": 220}
        for column in columns:
            self.steps_tree.heading(column, text=headings[column])
            self.steps_tree.column(column, width=widths[column], stretch=column in {"name", "note"})
        tree_y = self.ttk.Scrollbar(parent, orient="vertical", command=self.steps_tree.yview)
        tree_x = self.ttk.Scrollbar(parent, orient="horizontal", command=self.steps_tree.xview)
        self.steps_tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.steps_tree.grid(row=3, column=0, sticky="nsew")
        tree_y.grid(row=3, column=1, sticky="ns")
        tree_x.grid(row=4, column=0, sticky="ew")
        self.steps_tree.bind("<<TreeviewSelect>>", self._on_step_selected)
        self.steps_tree.bind("<Double-1>", lambda _event: self._selected_step_command("inspect_step {index}"))
        self.steps_tree.bind("<MouseWheel>", self._on_tree_mousewheel)
        self.steps_tree.tag_configure("combine_selected", background="#d8ecff")
        self.steps_tree.tag_configure("playback_current", background="#ffe08a", foreground="#202020")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

    def _build_step_actions(self, parent: Any) -> None:
        action_buttons = [
            ("Show Selected Before", lambda: self._selected_step_command("show_step {index} before")),
            ("Show Selected After", lambda: self._selected_step_command("show_step {index} after")),
            ("Show Step Summary", lambda: self._selected_step_command("show_step_summary {index}")),
            ("Show Step JSON", lambda: self._selected_step_command("inspect_step {index}")),
            ("Show JSON Truncated", lambda: self._selected_step_command("inspect_step {index}")),
            ("Export Step JSON", lambda: self._selected_step_command("export_step_json {index}")),
            ("Replace Selected Step", self._replace_selected),
            ("Delete Selected Step", self._delete_selected),
            ("Undo", lambda: self._post("undo")),
            ("Clear All Steps", self._clear_all_steps),
        ]
        guard_by_label = {
            "Show Selected Before": "show_step",
            "Show Selected After": "show_step",
            "Show Step Summary": "show_step",
            "Show Step JSON": "show_step",
            "Show JSON Truncated": "show_step",
            "Export Step JSON": "show_step",
            "Replace Selected Step": "prepare_replace",
            "Delete Selected Step": "height_edit",
            "Undo": "height_edit",
            "Clear All Steps": "height_edit",
        }
        for index, (label, callback) in enumerate(action_buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in guard_by_label:
                self._register_guarded_button(button, guard_by_label[label])
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=2, pady=2)
        self.ttk.Checkbutton(parent, text="Confirm Clear All", variable=self.clear_confirm_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        for col in range(3):
            parent.columnconfigure(col, weight=1)

    def _build_right_notebook(self, parent: Any) -> None:
        notebook = self.ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.right_notebook = notebook
        for tab_name, builder in [
            ("Sim Connection", self._build_connection_tab),
            ("Run Manager", self._build_run_manager_tab),
            ("Record / Servo+Wheel", self._build_record_servo_wheel_tab),
            ("Playback", self._build_playback_tab),
            ("Height Generate", self._build_height_generate_tab),
            ("Combine", self._build_combine_tab),
            ("Sim State", self._build_sim_state_tab),
        ]:
            tab = self.ttk.Frame(notebook, padding=2)
            notebook.add(tab, text=tab_name)
            body = self._make_scrollable_tab_body(tab)
            builder(body)

    def _make_scrollable_tab_body(self, parent: Any) -> Any:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(parent, highlightthickness=0)
        ybar = self.ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=ybar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        body = self.ttk.Frame(canvas, padding=6)
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def sync(_event: Any | None = None) -> None:
            canvas.itemconfigure(window_id, width=max(canvas.winfo_width(), body.winfo_reqwidth()))
            canvas.configure(scrollregion=canvas.bbox("all"))

        body.bind("<Configure>", sync)
        canvas.bind("<Configure>", sync)
        canvas.bind("<MouseWheel>", lambda event, c=canvas: self._scroll_canvas(c, event))
        body.columnconfigure(0, weight=1)
        return body

    def _build_connection_tab(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Isaac Sim Connection")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Run Isaac Preflight", self._run_isaac_preflight),
            ("Connect To Sim Worker", self._connect_worker),
            ("Disconnect From Sim Worker", self._disconnect_worker),
            ("Refresh Connection Status", lambda: self._refresh(force=True)),
        ]
        for index, (label, callback) in enumerate(buttons):
            self.ttk.Button(frame, text=label, command=callback).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.connection_text = self._make_scrolled_text(parent, row=1, height=16, width=72)

    def _build_run_manager_tab(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Simulation Process")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Start Sim Worker", self._ensure_sim),
            ("Stop Sim Worker", self._stop_worker),
            ("Restart Sim Worker", self._restart_isaac_worker),
            ("Open Worker Log Folder", self._open_worker_log_folder),
            ("Copy Worker Command", self._copy_worker_command),
        ]
        guard_by_label: dict[str, str] = {}
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(frame, text=label, command=callback)
            if label in guard_by_label:
                self._register_guarded_button(button, guard_by_label[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.run_manager_text = self._make_scrolled_text(parent, row=1, height=16, width=72)

    def _build_record_servo_wheel_tab(self, parent: Any) -> None:
        sw = self.ttk.LabelFrame(parent, text="Servo-Wheel Mode")
        sw.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Start Servo-Wheel Mode", self._start_servo_wheel_mode_ui),
            ("Launch Servo-Wheel", self._launch_servo_wheel_ui),
            ("Clear Staged", self._clear_servo_wheel_ui),
            ("Cancel Servo-Wheel Mode", self._cancel_servo_wheel_ui),
        ]
        for index, (label, callback) in enumerate(buttons):
            self.ttk.Button(sw, text=label, command=callback).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2
            )
        self.ttk.Label(sw, text="Reads the visible canonical servo and wheel targets, sends one MotionBatch, and applies one articulation write on the next physics tick.", wraplength=540).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=(4, 2)
        )
        self.ttk.Label(sw, textvariable=self.servo_wheel_mode_var, wraplength=560).grid(row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.ttk.Label(sw, textvariable=self.servo_wheel_preview_var, wraplength=560).grid(row=4, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.ttk.Label(sw, textvariable=self.servo_wheel_launch_var, wraplength=560).grid(row=5, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.ttk.Label(sw, textvariable=self.motion_profile_var, wraplength=560).grid(row=6, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        sw.columnconfigure(0, weight=1)
        sw.columnconfigure(1, weight=1)
        record = self.ttk.LabelFrame(parent, text="Record")
        record.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        record_buttons = [
            ("Validate Recording Baseline", "validate_recording_baseline"),
            ("Start Record Step", "step_record start"),
            ("Stop Record Step", "step_record stop"),
            ("Accept Recorded Step", "step_record accept"),
            ("Discard Pending Step", "step_record discard"),
            ("Discard Pending + Restore Before", "step_record discard_restore_before"),
        ]
        record_guards = {
            "Start Record Step": "start_record",
            "Accept Recorded Step": "accept_record",
        }
        for index, (label, command) in enumerate(record_buttons):
            button = self.ttk.Button(record, text=label, command=lambda text=command: self._post(text))
            if label in record_guards:
                self._register_guarded_button(button, record_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(record, textvariable=self.record_label_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        self.ttk.Label(record, textvariable=self.recording_path_var, wraplength=560).grid(row=4, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        self.ttk.Label(record, textvariable=self.baseline_status_var, wraplength=560).grid(row=5, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        self.ttk.Label(record, textvariable=self.pending_label_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        replace = self.ttk.LabelFrame(parent, text="Replacement")
        replace.grid(row=2, column=0, sticky="ew")
        replace_buttons = [
            ("Prepare Replacement", self._replace_selected),
            ("Start Replacement Recording", lambda: self._post("replace_step start")),
            ("Stop Replacement Recording", lambda: self._post("replace_step stop")),
            ("Accept Replacement", lambda: self._post("replace_step accept")),
            ("Discard Replacement", lambda: self._post("replace_step discard")),
            ("Cancel Replacement", lambda: self._post("replace_step cancel")),
        ]
        replace_guards = {
            "Prepare Replacement": "prepare_replace",
            "Start Replacement Recording": "start_record",
            "Accept Replacement": "accept_replace",
        }
        for index, (label, callback) in enumerate(replace_buttons):
            button = self.ttk.Button(replace, text=label, command=callback)
            if label in replace_guards:
                self._register_guarded_button(button, replace_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(replace, textvariable=self.replacement_label_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        record.columnconfigure(0, weight=1)
        record.columnconfigure(1, weight=1)
        replace.columnconfigure(0, weight=1)
        replace.columnconfigure(1, weight=1)

    def _start_servo_wheel_mode_ui(self) -> None:
        self.controller.start_servo_wheel_mode()
        self._refresh(force=False)

    def _launch_servo_wheel_ui(self) -> None:
        self.controller.launch_servo_wheel()
        self._refresh(force=False)

    def _clear_servo_wheel_ui(self) -> None:
        self.controller.clear_servo_wheel_staged()
        self._refresh(force=False)

    def _cancel_servo_wheel_ui(self) -> None:
        self.controller.cancel_servo_wheel_mode()
        self._refresh(force=False)

    def _build_playback_tab(self, parent: Any) -> None:
        self.ttk.Label(parent, text="Fast removes implicit UI idle only; actuator commands are unchanged.", wraplength=560).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))
        self.ttk.Checkbutton(
            parent,
            text="Selected playback always restores previous saved step end state",
            variable=self.restore_step_start_var,
            command=self._apply_playback_options,
            state="disabled",
        ).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 2)
        )
        self.ttk.Checkbutton(parent, text="Restore full sim pose if available", variable=self.restore_full_sim_pose_var, command=self._apply_playback_options).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2
        )
        self.ttk.Checkbutton(parent, text="Fallback to command_state_before", variable=self.fallback_command_state_var, command=self._apply_playback_options).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(2, 6)
        )
        buttons = [
            ("Play All", lambda: self._post_playback_command("play")),
            ("Play All Fast", lambda: self._post_playback_command("play fast")),
            ("Respawn And Play All", lambda: self._post_playback_command("respawn_play all")),
            ("Respawn And Play All Fast", lambda: self._post_playback_command("respawn_play all fast")),
            ("Play Selected Step", lambda: self._selected_step_playback_command("play_step {index}")),
            ("Play Selected Fast", lambda: self._selected_step_playback_command("play_step {index} fast")),
            ("Respawn And Play Selected Step", lambda: self._selected_step_playback_command("respawn_play selected {index}")),
            ("Respawn And Play Selected Fast", lambda: self._selected_step_playback_command("respawn_play selected {index} fast")),
            ("Play To Selected From Start", lambda: self._selected_step_playback_command("play_to_step {index}")),
            ("Respawn And Play To Selected From Start", lambda: self._selected_step_playback_command("respawn_play to {index}")),
            ("Pause Play", lambda: self._post("pause_play")),
            ("Resume Play", lambda: self._post("resume_play")),
            ("Stop Play", lambda: self._post("stop_play")),
            ("Analyze Playback Timing", lambda: self._post("analyze_playback_timing")),
            ("Debug Selected Playback", lambda: self._post("playback_debug_selected")),
            ("Export Motion TXT", lambda: self._post("export_motion_txt")),
        ]
        playback_guards = {
            "Play All": "play_all",
            "Play All Fast": "play_all",
            "Respawn And Play All": "respawn_play",
            "Respawn And Play All Fast": "respawn_play",
            "Play Selected Step": "play_selected",
            "Play Selected Fast": "play_selected",
            "Respawn And Play Selected Step": "respawn_selected",
            "Respawn And Play Selected Fast": "respawn_selected",
            "Play To Selected From Start": "play_selected",
            "Respawn And Play To Selected From Start": "respawn_selected",
            "Pause Play": "pause_play",
            "Resume Play": "resume_play",
            "Stop Play": "stop_play",
            "Analyze Playback Timing": "analyze_playback",
            "Debug Selected Playback": "debug_selected",
            "Export Motion TXT": "export_motion",
        }
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in playback_guards:
                self._register_guarded_button(button, playback_guards[label])
            self.playback_buttons.append(button)
            self.playback_buttons_by_label[label] = button
            button.grid(row=5 + index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.root.bind("<ButtonPress-1>", self._play_selected_fast_mouse_press, add="+")
        self.ttk.Label(parent, textvariable=self.playback_label_var, wraplength=560, justify="left").grid(row=13, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.ttk.Label(parent, textvariable=self.playback_unavailable_var, wraplength=560, foreground="#8a3b00").grid(row=14, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def _build_height_generate_tab(self, parent: Any) -> None:
        from height_generate_panel import HeightGeneratePanel

        self.height_generate_panel = HeightGeneratePanel(self, parent)

    def _build_combine_tab(self, parent: Any) -> None:
        buttons = [
            ("Combine Mode", lambda: self._post("combine mode")),
            ("Cancel Combine Mode", lambda: self._post("combine cancel")),
            ("Add Selected To Combine", self._combine_add_selected),
            ("Remove Selected From Combine", self._combine_remove_selected),
            ("Toggle Selected For Combine", self._combine_toggle_selected),
            ("Select Contiguous Range", self._combine_select_contiguous_range),
            ("Clear Combine Selection", lambda: self._post("combine clear")),
            ("Preview Combined Step", lambda: self._post("combine preview")),
            ("Combine Selected Steps", lambda: self._post("combine commit")),
        ]
        combine_guards = {
            "Preview Combined Step": "combine",
            "Combine Selected Steps": "combine",
        }
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in combine_guards:
                self._register_guarded_button(button, combine_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Checkbutton(parent, text="Allow conflicts, later step wins", variable=self.combine_allow_conflicts_var, command=self._set_combine_allow_conflicts).grid(row=5, column=0, columnspan=2, sticky="w")
        self.ttk.Label(parent, textvariable=self.combine_selection_var, wraplength=560).grid(row=6, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        self.combine_preview_text = self._make_scrolled_text(parent, row=7, columnspan=2, height=9, width=58)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def _build_sim_state_tab(self, parent: Any) -> None:
        self.ttk.Button(parent, text="Refresh Full Sim State", command=self._refresh_full_sim_state).grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.sim_state_text = self._make_scrolled_text(parent, row=1, height=23, width=62)

    def _refresh_full_sim_state(self) -> None:
        self.controller.transport.request_state(detailed=True)
        self.controller.status = "Detailed simulation state requested."
        self.root.after(300, lambda: self._refresh(force=False, sim_state=True))

    def _build_manifest_tree(self, parent: Any, *, row: int) -> Any:
        frame = self.ttk.LabelFrame(parent, text="Manifest")
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 6))
        columns = ("height", "recorded", "steps", "saved", "marker")
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", height=10)
        widths = {"height": 80, "recorded": 80, "steps": 70, "saved": 180, "marker": 90}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths[col], stretch=col == "saved")
        ybar = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ybar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        return tree

    def _make_scrolled_text(self, parent: Any, *, row: int, column: int = 0, columnspan: int = 1, height: int, width: int) -> Any:
        frame = self.ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=4, pady=4)
        text = self.tk.Text(frame, height=height, width=width, wrap="none")
        ybar = self.ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xbar = self.ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set, state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        self._bind_inner_scroll_widget(text, ybar)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(row, weight=1)
        return text

    def _servo_slider(self, joint_name: str, value: str) -> None:
        if self.updating:
            return
        angle = float(value)
        if self.controller.servo_wheel_staging_active:
            self.controller.stage_servo_wheel_servo(joint_name, angle)
            return
        self._schedule_slider(joint_name, f"servo {joint_name} {angle:.1f}", angle, 0.5)

    def _wheel_slider(self, short_name: str, value: str) -> None:
        if self.updating or self.wheel_stop_latched:
            return
        speed = float(value)
        if self.controller.servo_wheel_staging_active:
            self.controller.stage_servo_wheel_wheel(short_name, speed)
            return
        self._schedule_slider(short_name, f"wheel {short_name} {speed:.3f}", speed, 0.02)

    def _begin_wheel_drag(self, short_name: str) -> None:
        self.wheel_stop_latched = False
        self.slider_dragging.add(short_name)

    def _stop_wheels_ui(self) -> None:
        self.wheel_stop_latched = True
        for key in ("fl", "fr", "rl", "rr"):
            after_id = self.slider_after_ids.pop(key, None)
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
            self.slider_pending.pop(key, None)
            self.slider_dragging.discard(key)
            if key in self.wheel_vars:
                self.wheel_vars[key].set(0.0)
        self.busy_label_var.set("Stopping wheels...")
        self.controller.handle_command(CommandMessage(text="wheel stop", source="ui", command_id=uuid.uuid4().hex))
        self._refresh(force=False)

    def _run_command(self) -> None:
        text = self.command_var.get().strip()
        if not text:
            return
        self._post(text)
        self.command_var.set("")

    def _selected_indices(self) -> list[int]:
        values: list[int] = []
        for item in self.steps_tree.selection():
            try:
                values.append(int(str(item).split("_")[-1]))
            except ValueError:
                pass
        return sorted(values)

    def _selected_index(self) -> int | None:
        indices = self._selected_indices()
        return indices[0] if indices else self.selected_step_index

    def resolve_playback_selected_index(self) -> int | None:
        """Resolve the playback selection from the visible tree, then the shared controller."""

        indices = self._selected_indices()
        candidate = indices[0] if indices else self.controller.selected_step_index
        try:
            index = int(candidate) if candidate is not None else 0
        except (TypeError, ValueError):
            return None
        return index if 1 <= index <= self.controller.manager.count else None

    def _selected_fast_boundary(self, event: str, **details: Any) -> None:
        now = time.perf_counter()
        self.selected_fast_click_trace.append(
            {
                "event": str(event),
                "click_id": self.last_selected_fast_click_id,
                "relative_ms": max(0.0, (now - self.last_selected_fast_click_started) * 1000.0),
                **copy.deepcopy(details),
            }
        )

    def _play_selected_fast_mouse_press(self, event: Any) -> None:
        fast_button = self.playback_buttons_by_label.get("Play Selected Fast")
        if fast_button is None or getattr(event, "widget", None) != fast_button:
            return
        self.last_selected_fast_click_id = uuid.uuid4().hex
        self.last_selected_fast_click_started = time.perf_counter()
        self.last_selected_fast_feedback_ms = 0.0
        self.selected_fast_feedback_pending = False
        self.pending_selected_fast_feedback_text = ""
        self.selected_fast_click_trace = []
        self.controller.selected_fast_click_id = self.last_selected_fast_click_id
        self._selected_fast_boundary(
            "physical_mouse_press",
            widget=str(getattr(event, "widget", "")),
            x_root=int(getattr(event, "x_root", 0) or 0),
            y_root=int(getattr(event, "y_root", 0) or 0),
        )
        index = self.resolve_playback_selected_index()
        ok, reason = self.controller.playback_readiness(respawn_first=False)
        if index is None:
            feedback = "Select an accepted step first."
        elif not ok:
            feedback = f"Play Selected Fast blocked: {reason}"
        elif index > 1:
            feedback = (
                f"Play Selected Fast received: Step {index}\n"
                f"Restoring saved end state from Step {index - 1}..."
            )
        else:
            feedback = "Play Selected Fast received: Step 1\nRestoring Step 1 saved start state..."
        self.pending_selected_fast_feedback_text = feedback
        if index is None or not ok:
            self.selected_fast_feedback_pending = True
            self.playback_label_var.set(feedback)
            self.last_selected_fast_feedback_ms = (
                time.perf_counter() - self.last_selected_fast_click_started
            ) * 1000.0
            self._selected_fast_boundary(
                "immediate_visible_feedback",
                selected_index=index,
                allowed=False,
                reason=reason,
                text=feedback,
                feedback_ms=self.last_selected_fast_feedback_ms,
            )

    def _selected_step_command(self, template: str) -> None:
        index = self._selected_index()
        if index is None:
            self.controller._warn("[WARN] Select an accepted step first. No command posted.")
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        self._post(template.format(index=index))

    def _selected_step_playback_command(self, template: str) -> None:
        is_fast = template.strip().endswith(" fast") and template.strip().startswith("play_step")
        if is_fast:
            if not self.last_selected_fast_click_id or time.perf_counter() - self.last_selected_fast_click_started > 2.0:
                self.last_selected_fast_click_id = uuid.uuid4().hex
                self.last_selected_fast_click_started = time.perf_counter()
                self.selected_fast_click_trace = []
                self.controller.selected_fast_click_id = self.last_selected_fast_click_id
            self._selected_fast_boundary("tk_button_command_entered", template=template)
        tree_selection = list(self.steps_tree.selection())
        tree_indices = self._selected_indices()
        index = self.resolve_playback_selected_index()
        respawn_first = template.strip().startswith("respawn_play")
        ok, reason = self.controller.playback_readiness(respawn_first=respawn_first)
        command = template.format(index=index) if index is not None else ""
        if is_fast:
            feedback = self.pending_selected_fast_feedback_text
            if not feedback:
                if index is None:
                    feedback = "Select an accepted step first."
                elif not ok:
                    feedback = f"Play Selected Fast blocked: {reason}"
                elif index > 1:
                    feedback = (
                        f"Play Selected Fast received: Step {index}\n"
                        f"Restoring saved end state from Step {index - 1}..."
                    )
                else:
                    feedback = "Play Selected Fast received: Step 1\nRestoring Step 1 saved start state..."
            self.selected_fast_feedback_pending = True
            self.playback_label_var.set(feedback)
            self.busy_label_var.set("Restoring..." if ok and index is not None else "")
            self.root.update_idletasks()
            self.last_selected_fast_feedback_ms = (
                time.perf_counter() - self.last_selected_fast_click_started
            ) * 1000.0
            self._selected_fast_boundary(
                "immediate_visible_feedback",
                selected_index=index,
                allowed=bool(ok and index is not None),
                reason=reason,
                text=feedback,
                feedback_ms=self.last_selected_fast_feedback_ms,
            )
            self._selected_fast_boundary(
                "selected_index_resolved",
                tree_indices=tree_indices,
                ui_selected_step_index=self.selected_step_index,
                controller_selected_step_index=self.controller.selected_step_index,
                selected_index=index,
            )
            self._selected_fast_boundary("playback_readiness_evaluated", allowed=ok, reason=reason)
            self._selected_fast_boundary("generated_command", command=command)
            self.selected_fast_feedback_pending = False
            self.pending_selected_fast_feedback_text = ""
        self._apply_playback_options()
        self.controller._info(
            "[PLAYBACK DEBUG] ui play-selected "
            f"tree_selection={tree_selection} tree_indices={tree_indices} "
            f"controller_selected={self.controller.selected_step_index} selected_index={index} "
            f"generated_command={command or '<none>'} can_playback={ok} reason={reason or 'ok'}"
        )
        if index is None:
            self.controller._warn("[WARN] Select an accepted step first. No playback command posted.")
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        if not ok:
            self.controller._warn("[WARN] " + reason)
            return
        self._post(command)

    def _post_playback_command(self, command: str) -> None:
        self._apply_playback_options()
        self._post(command)

    def _replace_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        self._post(f"replace_step {index}")

    def _delete_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        if self.messagebox.askyesno("Delete Step", f"Delete accepted step {index:03d} from memory?"):
            self._post(f"delete_step {index}")

    def _clear_all_steps(self) -> None:
        if not self.clear_confirm_var.get():
            self.messagebox.showwarning("Confirm Clear All", "Check Confirm Clear All before clearing accepted steps.")
            self._post("clear_steps")
            return
        if self.messagebox.askyesno("Clear All Steps", "Clear all accepted steps in memory? Saved files are not deleted."):
            self._post("clear_steps_confirmed")
            self.clear_confirm_var.set(False)

    def _on_step_selected(self, _event: Any) -> None:
        indices = self._selected_indices()
        if self.controller.visible_steps_are_read_only():
            self.controller.combine_selected_indices.clear()
            self.controller.combine_preview_text = ""
        elif self.controller.combine_mode_enabled:
            if len(indices) > 1:
                self.controller.set_combine_selection(indices)
        else:
            self.controller.set_combine_selection(indices)
        if not indices:
            return
        self.selected_step_index = indices[0]
        self.controller.selected_step_index = indices[0]
        try:
            step = self.controller.get_visible_step(indices[0])
            self.controller.set_step_summary_detail(step, title=f"Step {indices[0]:03d} summary")
        except Exception:
            pass
        self._refresh(force=False)

    def _run_isaac_preflight(self) -> None:
        self.controller.status = "Running Isaac preflight..."
        self._refresh(force=True)

        def work() -> None:
            try:
                result = self.controller.run_isaac_preflight()
                error = ""
            except Exception as exc:
                result = {}
                error = str(exc)

            def done() -> None:
                if error:
                    self.controller._warn(f"[WARN] Isaac preflight failed: {error}")
                else:
                    self.controller.latest_sim_status = {
                        **self.controller.latest_sim_status,
                        "preflight": result,
                        "preflight_ok": bool(result.get("preflight_ok", False)),
                        "preflight_error": str(result.get("preflight_error", "") or ""),
                        "candidate_reports": result.get("candidate_reports", []),
                    }
                self._refresh(force=True)

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _restart_isaac_worker(self) -> None:
        try:
            self.controller.restart_sim_worker()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not restart Isaac worker: {exc}")
        self._refresh(force=True)

    def _connect_worker(self) -> None:
        self.controller.connect_to_sim_worker()
        self._refresh(force=True)

    def _disconnect_worker(self) -> None:
        self.controller.disconnect_from_sim_worker()
        self._refresh(force=True)

    def _open_worker_log_folder(self) -> None:
        try:
            path = self.controller.open_worker_log_folder()
            if path:
                self.controller._info(f"[INFO] Worker log folder: {path}")
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not open worker log folder: {exc}")
        self._refresh(force=True)

    def _copy_worker_command(self) -> None:
        command = self.controller.copy_worker_display_command()
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(command)
            self.controller._info("[INFO] Worker command copied to clipboard.")
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not copy worker command: {exc}")
        self._refresh(force=True)

    def _stop_worker(self) -> None:
        try:
            self.controller.stop_sim_worker()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not stop Isaac worker: {exc}")
        self._refresh(force=True)

    def _ensure_sim(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not start Isaac Sim: {exc}")
        self._refresh(force=False)

    def _generate_current_obstacle(self) -> None:
        try:
            self.controller.generate_or_update_height_obstacle()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not generate obstacle: {exc}")
        self._refresh(force=False)

    def _load_current_height_steps(self) -> None:
        if not self._resolve_unsaved_changes("reloading the current sequence"):
            return
        self.controller.load_steps_for_current_height(discard_dirty=True)
        self._refresh(force=False)

    def _save_shortcut(self, _event: Any | None = None) -> str:
        if self.controller.manager.dirty:
            self._save_new_version_async()
        return "break"

    def _save_new_version_async(self) -> None:
        if self.save_in_progress:
            return
        allow_empty = False
        if self.controller.manager.count <= 0:
            allow_empty = self.messagebox.askyesno(
                "Save Empty Version",
                "Current sequence has no accepted steps. Create a new immutable empty version?",
            )
            if not allow_empty:
                return
        self.save_in_progress = True
        self.save_status_var.set("Saving a new immutable version…")
        try:
            self.save_modified_steps_button.configure(state="disabled")
        except Exception:
            pass

        def work() -> None:
            try:
                path = self.controller.save_new_version(allow_empty=allow_empty, version_name="manual")
                error = ""
            except Exception as exc:
                path = None
                error = str(exc)

            def done() -> None:
                self.save_in_progress = False
                report = self.controller.last_save_report
                if path is not None and not self.controller.manager.dirty:
                    self.save_status_var.set(
                        f"Saved {report.get('version_id', '')} at {report.get('saved_at', '')} | "
                        f"{report.get('step_count', 0)} steps / {report.get('command_count', 0)} commands | "
                        f"SHA-256 {report.get('accepted_steps_sha256', '')} | {path}"
                    )
                else:
                    self.save_status_var.set(f"Save failed: {error or self.controller.status}")
                self._refresh(force=False)

            self.root.after(0, done)

        threading.Thread(target=work, name="HeightVersionSave", daemon=True).start()

    def _save_current_height_steps(self) -> Path | None:
        allow_empty = False
        if self.controller.manager.count <= 0:
            allow_empty = self.messagebox.askyesno(
                "Save Empty Steps",
                "Current sequence has no accepted steps. Create a new immutable empty version?",
            )
            if not allow_empty:
                return None
        path = self.controller.save_steps_for_current_height(allow_empty=allow_empty)
        if path is not None and not self.controller.manager.dirty:
            report = self.controller.last_save_report
            self.save_status_var.set(
                f"Saved {report.get('saved_at', '')} | {report.get('step_count', 0)} steps | "
                f"{report.get('command_count', 0)} commands"
            )
        else:
            self.save_status_var.set(f"Save failed: {self.controller.status}")
        self._refresh(force=False)
        return path

    def _resolve_unsaved_changes(self, action: str) -> bool:
        if not self.controller.manager.dirty:
            return True
        decision = self.messagebox.askyesnocancel(
            "Unsaved changes",
            f"The current sequence has unsaved changes before {action}.\n\n"
            "Yes = Save New Version, No = Discard, Cancel = keep editing.",
        )
        if decision is None:
            return False
        if decision:
            saved = self._save_current_height_steps()
            return saved is not None and not self.controller.manager.dirty
        return True

    def _new_empty_current_height(self) -> None:
        if not self._resolve_unsaved_changes("creating a new empty sequence"):
            return
        self.controller.new_empty_sequence_for_current_height(discard_dirty=True)
        self._refresh(force=False)

    def _refresh_manifest(self) -> None:
        self.controller.refresh_manifest()
        self._refresh(force=False)

    def _combine_add_selected(self) -> None:
        self.controller.add_to_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_remove_selected(self) -> None:
        self.controller.remove_from_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_toggle_selected(self) -> None:
        self.controller.toggle_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_select_contiguous_range(self) -> None:
        indices = self._selected_indices()
        if len(indices) >= 2:
            self.controller.add_to_combine_selection([indices[0], indices[-1]])
        self.controller.select_contiguous_combine_range()
        self._refresh(force=False)

    def _set_combine_allow_conflicts(self) -> None:
        self.controller.allow_combine_conflicts = bool(self.combine_allow_conflicts_var.get())
        if self.controller.combine_mode_enabled:
            selected = sorted(self.controller.combine_selected_indices)
            self.controller.combine_preview_text = (
                f"Conflict option changed for selected steps {selected}. Click Preview Combined Step to recompute."
                if len(selected) >= 2
                else "Select at least two steps."
            )
        self._refresh(force=False)

    def _apply_playback_options(self) -> None:
        self.restore_step_start_var.set(True)
        self.controller.restore_step_start_state_before_selected_playback = True
        self.controller.restore_full_sim_pose_if_available = bool(self.restore_full_sim_pose_var.get())
        self.controller.fallback_to_command_state_before = bool(self.fallback_command_state_var.get())

    def _scroll_canvas(self, canvas: Any, event: Any) -> str:
        delta = self._wheel_units(event)
        if delta:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _on_tree_mousewheel(self, event: Any) -> str:
        widget = getattr(event, "widget", self.steps_tree)
        try:
            setattr(widget, "_last_user_scroll_at", time.monotonic())
        except Exception:
            pass
        delta = self._wheel_units(event)
        if delta:
            widget.yview_scroll(delta, "units")
        return "break"

    def _on_scale_mousewheel(self, event: Any) -> str:
        return "break"

    def _on_text_mousewheel(self, event: Any) -> str:
        widget = getattr(event, "widget", None)
        if widget is not None:
            try:
                setattr(widget, "_last_user_scroll_at", time.monotonic())
            except Exception:
                pass
            delta = self._wheel_units(event)
            if delta:
                try:
                    widget.yview_scroll(delta, "units")
                except Exception:
                    pass
        return "break"

    def _wheel_units(self, event: Any) -> int:
        number = int(getattr(event, "num", 0) or 0)
        if number == 4:
            return -1
        if number == 5:
            return 1
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        return int(-1 * (delta / 120))

    def _bind_inner_scroll_widget(self, widget: Any, ybar: Any | None = None) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(sequence, self._on_text_mousewheel)
            except Exception:
                pass
        if ybar is not None:
            try:
                ybar.bind("<ButtonPress-1>", lambda _event, w=widget: setattr(w, "_scrollbar_dragging", True))
                ybar.bind(
                    "<ButtonRelease-1>",
                    lambda _event, w=widget: (setattr(w, "_scrollbar_dragging", False), setattr(w, "_last_user_scroll_at", time.monotonic())),
                )
            except Exception:
                pass

    def _poll(self) -> None:
        if self._closing:
            return
        try:
            self.controller.update()
            self._refresh(force=False)
        except Exception as exc:
            self.controller._warn(f"[WARN] UI poll failed: {exc}\n{traceback.format_exc()}")
        if not self._closing:
            self._poll_after_id = self.root.after(self.ui_refresh_ms, self._poll)

    def _right_tab_is_visible(self, tab_name: str) -> bool:
        try:
            current = self.right_notebook.select()
            return str(self.right_notebook.tab(current, "text")) == tab_name
        except Exception:
            return False

    def open_right_tab(self, tab_name: str) -> bool:
        for tab_id in self.right_notebook.tabs():
            if str(self.right_notebook.tab(tab_id, "text")) == str(tab_name):
                self.right_notebook.select(tab_id)
                return True
        return False

    def _refresh(self, *, force: bool = True, sim_state: bool = False) -> None:
        if self.refreshing or self._closing:
            return
        refresh_started = time.perf_counter()
        now = time.monotonic()
        do_medium = force or (now - self._last_medium_refresh) * 1000.0 >= self.sim_status_refresh_ms
        do_full = force or (now - self._last_full_refresh) * 1000.0 >= self.full_refresh_ms
        auto_sim_state = (
            self._right_tab_is_visible("Sim State")
            and do_full
            and not self.disable_auto_sim_state_json
            and not self.sim_state_json_on_demand
        )
        snapshot = self.controller.snapshot()
        elapsed = now - self.controller._last_ui_refresh_wall
        if elapsed > 0.0:
            self.controller.ui_refresh_hz = 1.0 / elapsed
        self.controller._last_ui_refresh_wall = now
        sequence_revision = int(snapshot["revisions"]["sequence"])
        manifest_revision = int(snapshot["revisions"]["manifest"])
        sequence_changed = self._last_sequence_revision != sequence_revision
        manifest_changed = self._last_manifest_revision != manifest_revision
        detail_changed = self._last_detail_text != snapshot["detail_text"]
        combine_changed = self._last_combine_preview != snapshot["combine"]["preview"]
        self.refreshing = True
        self.updating = True
        try:
            self.summary_var.set(self.controller.status_line())
            self.sim_label_var.set(
                f"Connected={_yes_no(snapshot['sim']['connected'])} "
                f"Ready={_yes_no(snapshot['sim']['runtime_ready'])} "
                f"phase={snapshot['sim'].get('phase', '')} "
                f"RTF={float(snapshot['sim']['real_time_factor']):.2f}"
            )
            self.mode_label_var.set(f"Mode: {snapshot['recording']['mode']}")
            self.sequence_label_var.set(
                f"Sequence: {snapshot['sequence']['source']} steps={snapshot['sequence']['count']} "
                f"valid={snapshot['sequence']['valid']} | "
                + ("Unsaved changes" if snapshot['sequence']['dirty'] else "Saved")
            )
            self.accepted_steps_source_var.set(f"Accepted Steps Source: {snapshot['sequence']['source']}")
            self.accepted_steps_path_var.set(str(Path(snapshot['sequence']['accepted_path']).resolve()))
            desired_title = "Height Based Obstacle Replay - Isaac Sim" + (" — Unsaved changes" if snapshot['sequence']['dirty'] else "")
            if self.root.title() != desired_title:
                self.root.title(desired_title)
            if snapshot['sequence']['dirty']:
                if not self.save_status_var.get().startswith("Save failed:"):
                    self.save_status_var.set("Unsaved changes — press Ctrl+S or click Save New Version.")
            elif self.save_status_var.get().startswith("Unsaved changes"):
                self.save_status_var.set("No unsaved changes.")
            playback = snapshot["playback"]
            progress = dict(playback.get("progress_detail", {}) or {})
            step_index = int(progress.get("current_step_index", 0) or 0)
            total_steps = int(progress.get("total_steps", 0) or 0)
            selected_prefix = "Selected Step" if progress.get("selected_playback") else "Playing Step"
            state_text = str(progress.get("status_text", progress.get("playback_state", "Idle")) or "Idle")
            profile_text = "Motion-only" if str(progress.get("playback_profile", playback.get("profile", "raw"))) in {"fast", "motion_only"} else "Raw"
            hold_fast_feedback = bool(
                self.selected_fast_feedback_pending
                and time.perf_counter() - self.last_selected_fast_click_started < 1.0
            )
            if self.selected_fast_feedback_pending and not hold_fast_feedback:
                self.selected_fast_feedback_pending = False
            if not hold_fast_feedback:
                self.playback_label_var.set(
                    f"{state_text} | {selected_prefix} {step_index} / {total_steps}\n"
                    f"{progress.get('current_step_id', '') or '-'}\n"
                    f"Command {int(progress.get('current_command_index_in_step', 0) or 0)} / {int(progress.get('commands_in_current_step', 0) or 0)} | "
                    f"Total command {int(progress.get('global_command_index', 0) or 0)} / {int(progress.get('total_commands', playback.get('count', 0)) or 0)}\n"
                    f"Elapsed {float(progress.get('elapsed_time', 0.0) or 0.0):.3f}s | Remaining {float(progress.get('estimated_remaining_time', 0.0) or 0.0):.3f}s\n"
                    f"Profile: {profile_text} | direct recorded actuator commands"
                )
            self.playback_unavailable_var.set(
                "Playback unavailable reason: " + (playback.get("unavailable_reason") or "Available.")
            )
            self.record_label_var.set(
                f"Record: active={snapshot['recording']['active']} stop_pending={snapshot['recording']['stop_pending']} "
                f"events={snapshot['recording']['events']} dirty={snapshot['recording']['dirty']}"
            )
            sw = dict(snapshot.get("servo_wheel", {}) or {})
            self.servo_wheel_mode_var.set(
                f"Servo-Wheel Mode: {'ACTIVE' if sw.get('active') else 'inactive'} | "
                f"staged dirty={bool(sw.get('dirty', False))}"
            )
            staged = clone_command_state(sw.get("staged_state"))
            live = clone_command_state(self.controller.transport.capture_command_state())
            servo_delta = sum(
                1 for name in SERVO_JOINT_NAMES
                if not math.isclose(float(staged["servos"].get(name, 0.0)), float(live["servos"].get(name, 0.0)), abs_tol=1.0e-6)
            )
            wheel_delta = sum(
                1 for name in WHEEL_JOINT_NAMES
                if not math.isclose(float(staged["wheels"].get(name, 0.0)), float(live["wheels"].get(name, 0.0)), abs_tol=1.0e-9)
            )
            self.servo_wheel_preview_var.set(
                f"Staged/live delta: servos={servo_delta}, wheels={wheel_delta}; sliders do not dispatch while staging."
            )
            launch = dict(sw.get("last_launch_status", {}) or {})
            self.servo_wheel_launch_var.set(
                f"Last launch: {launch.get('state', 'none')} | batch={str(launch.get('batch_id', '') or '-')[:12]} "
                f"step={launch.get('applied_sim_step', '-')} error={launch.get('error', '') or '-'}"
            )
            self.recording_path_var.set(
                f"Output root: {snapshot['recording']['output_root']}\n"
                f"Height folder: {snapshot['recording']['height_folder']}\n"
                f"accepted_steps.jsonl: {snapshot['recording']['accepted_steps_path']}"
            )
            baseline = dict(snapshot["recording"].get("baseline", {}) or {})
            self.baseline_status_var.set(
                f"Recording baseline: {'PASS' if baseline.get('passed') else 'FAIL / not validated'} | "
                f"{baseline.get('baseline_id', '-')}"
            )
            self.busy_label_var.set(
                f"Operation: {snapshot['operation']['state']}"
                + (f" - {snapshot['operation']['reason']}" if snapshot["operation"]["reason"] else "")
            )
            self.pending_label_var.set(
                f"Pending: recorded={snapshot['recording']['pending']} "
                f"replacement={snapshot['recording']['pending_replacement']}"
            )
            self.replacement_label_var.set(f"Replacement target: {self.controller.replace_target_index or 'none'}")
            self.combine_selection_var.set(
                f"Selected combine indices: {snapshot['combine']['selected_indices']} | "
                f"count={snapshot['combine']['selected_count']} contiguous={snapshot['combine']['contiguous']}"
            )
            self.height_var.set(f"{snapshot['height']['current_mm']} mm")

            motion_profile = dict(snapshot.get("motion_profile", {}) or {})
            self.motion_profile_var.set(
                "Motion profile: Fixed 100% | "
                f"Servo {float(motion_profile.get('servo_velocity_deg_s', self.controller.motion_reference.servo_reference_velocity_deg_s)):.3f} deg/s | "
                f"Wheel ref {float(motion_profile.get('wheel_reference_velocity_rad_s', self.controller.motion_reference.wheel_reference_velocity_rad_s)):.6f} rad/s"
            )

            servo_display = snapshot["servos"]
            wheel_display = snapshot["wheels"]
            for name, value in servo_display.items():
                if name not in self.slider_dragging and name in self.servo_vars:
                    self.servo_vars[name].set(float(value))
                if name in self.servo_output_label_vars:
                    self.servo_output_label_vars[name].set(
                        f"out {float(value):.1f}"
                    )
            wheel_parts = []
            for short_name, value in wheel_display.items():
                if short_name not in self.slider_dragging and short_name in self.wheel_vars:
                    self.wheel_vars[short_name].set(float(value))
                wheel_parts.append(f"{short_name}={float(value):.2f}")
            self.wheel_status_var.set(
                "Wheel target (rad/s): "
                + " ".join(wheel_parts)
                + " | "
                + (
                    "Wheels physically stopped"
                    if snapshot["wheel_command"].get("physically_stopped")
                    else "Zero target applied; wheels decelerating"
                    if snapshot["wheel_command"].get("zero_target_applied")
                    else "Stopping wheels..."
                    if self.controller.last_wheel_stop_request
                    else str(snapshot["wheel_command"].get("state", "idle"))
                )
            )
            measured_rows = dict((snapshot.get("actual_joint_state") or {}).get("wheels", {}) or {})
            measured_parts = []
            for short_name, full_name in WHEEL_SHORT_NAMES.items():
                value = dict(measured_rows.get(full_name, {}) or {}).get("rad_s")
                measured_parts.append(f"{short_name}={'?' if value is None else f'{float(value):.3f}'}")
            self.measured_wheel_status_var.set("Measured wheel velocity (rad/s): " + " ".join(measured_parts))

            if force or sequence_changed:
                self._refresh_steps_tree(snapshot["sequence"]["rows"], snapshot["combine"]["selected_indices"])
                self._last_sequence_revision = sequence_revision
            self._highlight_playback_step(step_index)
            if force or detail_changed:
                self._set_text(self.detail_text, snapshot["detail_text"])
                self._last_detail_text = snapshot["detail_text"]
            if do_medium:
                connection_json = json.dumps(snapshot["sim"], indent=2, ensure_ascii=False, default=str)
                self._set_text(self.connection_text, connection_json)
                self._set_text(self.run_manager_text, connection_json)
                self._last_medium_refresh = now
            if sim_state or auto_sim_state:
                explicit_detail = (
                    dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
                    if sim_state and self.controller.sim_client is not None
                    else {}
                )
                state_for_display = {**snapshot, "detailed_worker_state": explicit_detail} if explicit_detail else snapshot
                self._set_text(self.sim_state_text, json.dumps(state_for_display, indent=2, ensure_ascii=False, default=str))
            if force or combine_changed:
                self._set_text(self.combine_preview_text, snapshot["combine"]["preview"])
                self._last_combine_preview = snapshot["combine"]["preview"]
            if force or manifest_changed:
                self._refresh_manifest_trees(snapshot["height"]["manifest_rows"])
                self._last_manifest_revision = manifest_revision
            if do_full:
                self._last_full_refresh = now
            if hasattr(self, "height_generate_panel") and (force or do_medium or manifest_changed or sequence_changed):
                self.height_generate_panel.refresh(
                    snapshot,
                    sync_versions=bool(force or manifest_changed),
                )
            self._refresh_button_states()
        finally:
            self.updating = False
            self.refreshing = False
            elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
            if elapsed_ms > 500.0:
                self.controller._warn(f"[WARN] [PERF] UI refresh took {elapsed_ms:.1f}ms")

    def _refresh_steps_tree(self, rows: list[dict[str, Any]], combine_selected: list[int]) -> None:
        existing = set(self.steps_tree.get_children())
        wanted = {f"step_{int(row['index'])}" for row in rows}
        for item in existing - wanted:
            self.steps_tree.delete(item)
        for row in rows:
            index = int(row["index"])
            item = f"step_{index}"
            values = (
                index,
                row["name"],
                row["type"],
                f"{float(row['duration']):.3f}",
                row["events_count"],
                row["note"],
            )
            tags = []
            if index in combine_selected:
                tags.append("combine_selected")
            if index == self._last_playback_highlight_step:
                tags.append("playback_current")
            if item in existing:
                self.steps_tree.item(item, values=values, tags=tuple(tags))
            else:
                self.steps_tree.insert("", "end", iid=item, values=values, tags=tuple(tags))
        if self.controller.combine_mode_enabled and combine_selected:
            items = [f"step_{int(index)}" for index in combine_selected if f"step_{int(index)}" in self.steps_tree.get_children()]
            if items:
                try:
                    self.steps_tree.selection_set(items)
                    self.steps_tree.see(items[-1])
                except Exception:
                    pass
        elif self.controller.selected_step_index:
            item = f"step_{self.controller.selected_step_index}"
            if item in self.steps_tree.get_children():
                try:
                    self.steps_tree.selection_set(item)
                    self.steps_tree.see(item)
                except Exception:
                    pass

    def _highlight_playback_step(self, step_index: int) -> None:
        step_index = int(step_index or 0)
        if step_index <= 0 or step_index == self._last_playback_highlight_step:
            return
        old_item = f"step_{self._last_playback_highlight_step}"
        new_item = f"step_{step_index}"
        children = set(self.steps_tree.get_children())
        if old_item in children:
            tags = tuple(tag for tag in self.steps_tree.item(old_item, "tags") if tag != "playback_current")
            self.steps_tree.item(old_item, tags=tags)
        if new_item in children:
            tags = tuple(self.steps_tree.item(new_item, "tags"))
            if "playback_current" not in tags:
                self.steps_tree.item(new_item, tags=tags + ("playback_current",))
            try:
                self.steps_tree.see(new_item)
            except Exception:
                pass
        self._last_playback_highlight_step = step_index

    def _refresh_manifest_trees(self, rows: list[dict[str, Any]]) -> None:
        for tree in [getattr(getattr(self, "height_generate_panel", None), "manifest_tree", None)]:
            if tree is None:
                continue
            tree.delete(*tree.get_children())
            for row in rows:
                height = int(row["height_mm"])
                marker = "current" if height == self.controller.current_height_mm else ""
                tree.insert(
                    "",
                    "end",
                    iid=f"manifest_{height}_{id(tree)}",
                    values=(
                        f"{height} mm",
                        "yes" if int(row.get("version_count", 0)) > 0 or row.get("legacy_available") else "no",
                        row.get("version_count", 0),
                        row.get("last_saved_at", ""),
                        marker,
                    ),
                )

    def _set_text(self, widget: Any, value: str, *, max_chars: int | None = None) -> None:
        limit = self.max_text_widget_chars if max_chars is None else max(1000, int(max_chars))
        text = str(value)
        if len(text) > limit:
            text = (
                f"[TRUNCATED] Text is {len(text)} chars; showing first {limit} chars. "
                "Use an export action for full data.\n\n"
                + text[:limit]
            )
        cache_key = id(widget)
        signature = (len(text), hash(text))
        if self._text_widget_cache.get(cache_key) == signature:
            return
        set_text_preserving_view(widget, text, follow_bottom=False)
        self._text_widget_cache[cache_key] = signature
