"""Visible, single-worker Isaac/Tk acceptance for the six completion fixes.

The formal 50 mm source is opened read-only.  Recording/version writes are
redirected below the report directory supplied by the caller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from command_model import KNEE_JOINT_NAMES, SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from playback import (
    plan_from_steps,
    playback_plan_from_payload,
    playback_plan_to_payload,
    validate_plan_integrity,
)
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


FORMAL_50_PATH = PROJECT_ROOT / "saved_height_steps" / "height_05cm" / "accepted_steps.jsonl"


class CompletionFixGuiE2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir, timeout_s=1500.0)
        self.height_targets = [50, 75, 100, 50, 100, 75]
        self.height_index = 0
        self.active_height_request = ""
        self.manual_started_sim = 0.0
        self.staging_started_step = 0
        self.active_batch_id = ""
        self.active_recalibrate_request = ""
        self.recalibrate_settle_started = 0.0
        self.recalibrate_attempt = 0
        self.formal_rows: list[dict[str, Any]] = []
        self.mini_rows: list[dict[str, Any]] = []
        self.mini_stop_plan_id = ""
        self.replay_trace_key: tuple[str, int, int, int] | None = None
        self.active_replay_name = ""
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "formal_source_path": str(FORMAL_50_PATH.resolve()),
            "formal_source_sha256_before": hashlib.sha256(FORMAL_50_PATH.read_bytes()).hexdigest(),
            "temporary_version_root": str(self.version_root),
            "stages": [],
            "screenshots": {},
            "height_transactions": [],
            "manual_wheel_states": [],
            "servo_wheel": {},
            "mini_playback": {},
            "knee_test": {},
            "formal_plan_integrity": {},
            "formal_replay_trace": [],
            "formal_raw": {},
            "formal_fast": {},
            "playback_handshakes": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        tabs = [str(self.ui.right_notebook.tab(tab_id, "text")) for tab_id in self.ui.right_notebook.tabs()]
        expected = [
            "Sim Connection",
            "Run Manager",
            "Record / Servo+Wheel",
            "Playback",
            "Height Generate",
            "Combine",
            "Sim State",
        ]
        if tabs != expected:
            raise AssertionError(f"unexpected tabs: {tabs}")
        self.result["tabs"] = tabs
        self.result["speed_scale_absent"] = "Speed Scale" not in tabs
        self.result["worker_pid"] = int(self.controller.latest_sim_status.get("pid", 0) or 0)
        self.result["worker_session_id"] = str(self.controller.latest_sim_status.get("worker_session_id", "") or "")
        self.result["startup_geometry"] = {
            key: copy.deepcopy(self.controller.latest_sim_status.get(key))
            for key in (
                "height_mm",
                "measured_height_mm",
                "measured_width_m",
                "measured_length_m",
                "measured_bounds",
                "visual_updated",
                "collision_updated",
                "obstacle_prim_valid",
                "obstacle_revision",
            )
        }
        self.ui.open_right_tab("Record / Servo+Wheel")
        self._capture_ui("servo_wheel_controls", "servo_wheel_controls.png")
        self._capture_isaac("isaac_startup", "isaac_startup.png")
        self._advance("HEIGHT_START")

    def _stage_height_start(self) -> None:
        target = self.height_targets[self.height_index]
        self.controller.current_height_mm = target
        started = time.perf_counter()
        request_id = self.controller.generate_or_update_height_obstacle()
        callback_ms = (time.perf_counter() - started) * 1000.0
        if not request_id:
            raise AssertionError(f"height request {target} was not queued: {self.controller.status}")
        self.active_height_request = request_id
        self.result["height_transactions"].append(
            {
                "requested_height_mm": target,
                "request_id": request_id,
                "requested_revision": self.controller.pending_height_requested_revision,
                "ipc_payload": {
                    "type": "set_height",
                    "height_mm": target,
                    "request_id": request_id,
                    "requested_revision": self.controller.pending_height_requested_revision,
                    "source": "ui",
                },
                "callback_ms": callback_ms,
                "operation_after_callback": self.controller.operation.state.value,
                "started_monotonic": time.monotonic(),
            }
        )
        self._advance("HEIGHT_WAIT")

    def _stage_height_wait(self) -> None:
        ack = dict(self.controller.latest_sim_status.get("last_operation_ack", {}) or {})
        if str(ack.get("request_id", "") or "") != self.active_height_request:
            return
        if self.controller.pending_height_mm is not None:
            return
        row = self.result["height_transactions"][-1]
        row.update(
            ack=copy.deepcopy(ack),
            ack_wall_ms=(time.monotonic() - row["started_monotonic"]) * 1000.0,
            operation_after_ack=self.controller.operation.state.value,
            ui_loaded_height_mm=self.controller.loaded_sim_height_mm,
            worker_scene_height_mm=self.controller.latest_sim_status.get("scene_height_mm"),
        )
        required = (
            bool(ack.get("accepted", False))
            and bool(ack.get("prim_valid", False))
            and bool(ack.get("visual_updated", False))
            and bool(ack.get("collision_updated", False))
            and bool(ack.get("control_ready", False))
            and abs(float(ack.get("measured_height_mm", -1)) - float(row["requested_height_mm"])) <= 1.0
            and abs(float(ack.get("measured_width_m", -1)) - 2.0) <= 0.001
            and not str(ack.get("error", "") or "")
        )
        if not required or row["operation_after_ack"] != "IDLE":
            raise AssertionError(f"height geometry transaction failed: {row}")
        self._capture_ui(f"height_{self.height_index}", f"height_{self.height_index}_{row['requested_height_mm']}mm.png")
        self.height_index += 1
        if self.height_index < len(self.height_targets):
            self._advance("HEIGHT_START")
        else:
            self._capture_isaac("isaac_75mm_wide", "isaac_75mm_wide.png")
            self._advance("MANUAL_IDLE_START")

    def _begin_wheel(self, label: str, velocity: float) -> None:
        before = copy.deepcopy(self.controller.transport.capture_command_state())
        generation_before = int(self.controller.transport.wheel_generation)
        self.controller.handle_command(f"wheel all {velocity}")
        self.manual_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self.result["manual_wheel_states"].append(
            {
                "label": label,
                "velocity": velocity,
                "ui_callback_triggered": True,
                "controller_readiness": self.controller.manual_motion_readiness(),
                "operation": self.controller.operation.state.value,
                "generation_before": generation_before,
                "generation_sent": int(self.controller.transport.wheel_generation),
                "command_state_before": before,
                "started_wall": time.time(),
            }
        )

    def _finish_wheel_observation(self) -> bool:
        row = self.result["manual_wheel_states"][-1]
        wheel = dict(self.controller.latest_sim_status.get("wheel_command", {}) or {})
        measured = dict(wheel.get("measured_velocity_rad_s", {}) or {})
        elapsed_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0) - self.manual_started_sim
        if elapsed_sim < 0.30 or max([abs(float(value)) for value in measured.values()] or [0.0]) < 0.05:
            return False
        row.update(
            worker_received=True,
            command_state_after=copy.deepcopy(self.controller.latest_sim_status.get("command_state", {})),
            applied_target_rad_s=copy.deepcopy(wheel.get("applied_target_rad_s", {})),
            measured_velocity_rad_s=measured,
            stale_command_rejected=bool(wheel.get("stale_command_rejected", False)),
            worker_generation=int(wheel.get("generation", 0) or 0),
        )
        if row["stale_command_rejected"]:
            raise AssertionError(f"manual wheel was rejected as stale: {row}")
        return True

    def _stage_manual_idle_start(self) -> None:
        self._begin_wheel("ordinary_idle_before_recording", 0.30)
        self._advance("MANUAL_IDLE_WAIT")

    def _stage_manual_idle_wait(self) -> None:
        if not self._finish_wheel_observation():
            return
        self.controller.stop_wheels(reason="e2e_idle_boundary")
        self._advance("MANUAL_RECORD_START")

    def _stage_manual_record_start(self) -> None:
        self.controller.start_step_recording()
        if not self.controller.recording_active:
            raise AssertionError(f"recording did not start: {self.controller.status}")
        self._begin_wheel("recording", 0.32)
        self._advance("MANUAL_RECORD_WAIT")

    def _stage_manual_record_wait(self) -> None:
        if not self._finish_wheel_observation():
            return
        self.controller.stop_step_recording()
        self._advance("MANUAL_RECORD_STOP_WAIT")

    def _stage_manual_record_stop_wait(self) -> None:
        if self.controller.recording_active or self.controller.record_stop_pending is not None:
            return
        if self.controller.pending_step is None:
            return
        self.controller.discard_pending_step(restore_before=False)
        self._begin_wheel("ordinary_idle_after_recording", 0.28)
        self._advance("MANUAL_AFTER_WAIT")

    def _stage_manual_after_wait(self) -> None:
        if not self._finish_wheel_observation():
            return
        self.controller.stop_wheels(reason="e2e_after_record_boundary")
        self._capture_ui("manual_wheel_three_states", "manual_wheel_three_states.png")
        self._advance("STAGING_START")

    def _stage_staging_start(self) -> None:
        live_before = copy.deepcopy(self.controller.transport.capture_command_state())
        actual_before = copy.deepcopy(self.controller.latest_sim_status.get("actual_joint_state", {}))
        if not self.controller.start_servo_wheel_mode():
            raise AssertionError(self.controller.status)
        self.controller.stage_servo_wheel_servo("front_left_hip", 10.0)
        for name in WHEEL_JOINT_NAMES:
            self.controller.stage_servo_wheel_wheel(name, 0.25)
        self.staging_started_step = int(self.controller.latest_sim_status.get("sim_steps", 0) or 0)
        self.result["servo_wheel"] = {
            "live_before": live_before,
            "actual_before": actual_before,
            "staged_state": copy.deepcopy(self.controller.servo_wheel_staged_state),
            "staging_started_sim_step": self.staging_started_step,
        }
        self._capture_ui("servo_wheel_staged", "servo_wheel_staged.png")
        self._advance("STAGING_HOLD")

    def _stage_staging_hold(self) -> None:
        current_step = int(self.controller.latest_sim_status.get("sim_steps", 0) or 0)
        if current_step - self.staging_started_step < 24:
            return
        live_after_hold = copy.deepcopy(self.controller.transport.capture_command_state())
        self.result["servo_wheel"].update(
            live_after_staging_hold=live_after_hold,
            actual_after_staging_hold=copy.deepcopy(self.controller.latest_sim_status.get("actual_joint_state", {})),
            staging_hold_steps=current_step - self.staging_started_step,
        )
        if live_after_hold != self.result["servo_wheel"]["live_before"]:
            raise AssertionError("staging changed live command state before Launch")
        self.controller.start_step_recording()
        if not self.controller.recording_active:
            raise AssertionError(f"recording after staging did not start: {self.controller.status}")
        self.active_batch_id = self.controller.launch_servo_wheel()
        if not self.active_batch_id:
            raise AssertionError(self.controller.status)
        self.result["servo_wheel"]["first_batch_id"] = self.active_batch_id
        self._advance("STAGING_LAUNCH1_WAIT")

    def _stage_staging_launch1_wait(self) -> None:
        launch = dict(self.controller.servo_wheel_last_launch_status or {})
        if str(launch.get("batch_id", "") or "") != self.active_batch_id or launch.get("state") not in {"applied", "error"}:
            return
        if launch.get("state") != "applied" or str(launch.get("error", "") or ""):
            raise AssertionError(f"first atomic Launch failed: {launch}")
        self.result["servo_wheel"]["first_ack"] = copy.deepcopy(launch)
        self.controller.stage_servo_wheel_servo("front_left_hip", 14.0)
        for name in WHEEL_JOINT_NAMES:
            self.controller.stage_servo_wheel_wheel(name, 0.18)
        self.active_batch_id = self.controller.launch_servo_wheel()
        self.result["servo_wheel"]["second_batch_id"] = self.active_batch_id
        self._advance("STAGING_LAUNCH2_WAIT")

    def _stage_staging_launch2_wait(self) -> None:
        launch = dict(self.controller.servo_wheel_last_launch_status or {})
        if str(launch.get("batch_id", "") or "") != self.active_batch_id or launch.get("state") not in {"applied", "error"}:
            return
        if launch.get("state") != "applied" or str(launch.get("error", "") or ""):
            raise AssertionError(f"second atomic Launch failed: {launch}")
        self.result["servo_wheel"]["second_ack"] = copy.deepcopy(launch)
        if float(launch.get("motion_start_skew_s", -1.0)) != 0.0:
            raise AssertionError(f"servo/wheel start skew is not zero: {launch}")
        self._capture_ui("servo_wheel_launched", "servo_wheel_launched.png")
        self.controller.stop_step_recording()
        self._advance("STAGING_RECORD_STOP_WAIT")

    def _stage_staging_record_stop_wait(self) -> None:
        if self.controller.recording_active or self.controller.record_stop_pending is not None:
            return
        step = self.controller.pending_step
        if step is None:
            return
        launches = [row for row in list(step.get("events", []) or []) if row.get("kind") == "servo_wheel_launch"]
        if len(launches) != 2 or any(not isinstance(row.get("batch_ack"), dict) for row in launches):
            raise AssertionError(f"recorded atomic launch events are incomplete: {launches}")
        self.result["servo_wheel"]["recorded_launch_events"] = copy.deepcopy(launches)
        self.controller.accept_pending_step()
        self.controller.cancel_servo_wheel_mode()
        saved = self.controller.save_new_version(version_name="e2e-servo-wheel", note="temporary completion-fix validation")
        if saved is None:
            raise AssertionError(f"temporary version save failed: {self.controller.status}")
        version_id = self.controller.current_version_id
        self.controller.load_steps_for_current_height(discard_dirty=True, version_id=version_id)
        reloaded = copy.deepcopy(self.controller.manager.steps)
        reload_launches = [
            event
            for step_row in reloaded
            for event in list(step_row.get("events", []) or [])
            if event.get("kind") == "servo_wheel_launch"
        ]
        if len(reload_launches) != 2:
            raise AssertionError("saved/reloaded version did not preserve atomic launch events")
        self.result["servo_wheel"].update(
            temporary_version_path=str(saved.resolve()),
            temporary_version_id=version_id,
            reloaded_launch_count=len(reload_launches),
        )
        self.mini_rows = reloaded
        self._start_replay(reloaded, "temporary servo-wheel playback", "fast", result_key="mini_playback")

    def _start_replay(self, rows: list[dict[str, Any]], label: str, profile: str, *, result_key: str) -> None:
        ok = self.controller.start_playback(rows, label=label, profile=profile)
        local = copy.deepcopy(self.controller.playback.status_dict())
        if not ok:
            raise AssertionError(self.controller.playback.last_error or f"{label} start failed")
        if bool(local.get("active", False)) or not bool(local.get("start_requested", False)):
            raise AssertionError(f"worker playback bypassed START_REQUESTED: {local}")
        self.result[result_key] = {
            "label": label,
            "profile": profile,
            "local_immediately_after_request": local,
            "request_id": self.controller.playback.worker_request_id,
            "plan_id": self.controller.playback.worker_plan_id,
            "plan_sha256": self.controller.playback.plan.plan_sha256 if self.controller.playback.plan else "",
        }
        self.active_replay_name = result_key
        self.replay_trace_key = None
        self._advance("REPLAY_WAIT")

    def _stage_replay_wait(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if str(worker.get("plan_id", "") or "") != str(self.controller.playback.worker_plan_id or worker.get("plan_id", "")):
            return
        row = self.result[self.active_replay_name]
        if self.controller.playback.worker_acknowledged and "accepted_state" not in row:
            ack = dict(self.controller.latest_sim_status.get("last_operation_ack", {}) or {})
            row["accepted_state"] = copy.deepcopy(self.controller.playback.status_dict())
            row["start_ack"] = copy.deepcopy(ack)
            self.result["playback_handshakes"].append(copy.deepcopy(ack))
        if self.active_replay_name == "mini_pause_resume" and bool(worker.get("active", False)):
            if bool(worker.get("first_command_applied", False)) and not row.get("pause_requested"):
                self.controller.playback.pause()
                row["pause_requested"] = True
                row["pause_requested_wall"] = time.monotonic()
                return
            if row.get("pause_requested") and bool(worker.get("paused", False)) and not row.get("resume_requested"):
                row["worker_paused"] = copy.deepcopy(worker)
                if time.monotonic() - float(row["pause_requested_wall"]) < 0.25:
                    return
                self.controller.playback.resume()
                row["resume_requested"] = True
                return
        if (
            self.active_replay_name == "mini_stop"
            and bool(worker.get("active", False))
            and bool(worker.get("first_command_applied", False))
            and not row.get("stop_requested")
        ):
            row["worker_before_stop"] = copy.deepcopy(worker)
            row["stop_requested"] = True
            self.mini_stop_plan_id = str(worker.get("plan_id", "") or "")
            self.controller.playback.stop(reason="e2e_stop_validation")
            self._advance("MINI_STOP_WAIT")
            return
        progress = dict(worker.get("progress_detail", {}) or {})
        key = (
            self.active_replay_name,
            int(progress.get("current_step_index", 0) or 0),
            int(worker.get("segment_index", 0) or 0),
            int(progress.get("global_command_index", 0) or 0),
        )
        if key != self.replay_trace_key:
            self.replay_trace_key = key
            self.stage_started = time.monotonic()
            trace = {
                "replay": key[0],
                "wall_time": time.time(),
                "step": key[1],
                "segment": key[2],
                "global_command": key[3],
                "active": bool(worker.get("active", False)),
                "started": bool(worker.get("started", False)),
                "first_command_applied": bool(worker.get("first_command_applied", False)),
                "first_command_applied_sim_step": worker.get("first_command_applied_sim_step"),
                "servo_errors": copy.deepcopy(worker.get("current_servo_errors", {})),
                "residual_warning": copy.deepcopy(worker.get("last_servo_residual_warning", {})),
            }
            if self.active_replay_name in {"formal_raw", "formal_fast"}:
                self.result["formal_replay_trace"].append(trace)
                if key[1] in {23, 24, 25, 35}:
                    shot_key = f"{self.active_replay_name}_step_{key[1]}"
                    if shot_key not in self.result["screenshots"]:
                        self._capture_ui(shot_key, f"{shot_key}.png")
        if bool(worker.get("active", False)) or not str(worker.get("stop_reason", "") or ""):
            return
        row["worker_final"] = copy.deepcopy(worker)
        row["local_final"] = copy.deepcopy(self.controller.playback.status_dict())
        if str(worker.get("stop_reason", "") or "") != "complete":
            raise AssertionError(f"{self.active_replay_name} did not complete: {worker}")
        if self.active_replay_name == "mini_playback":
            self._advance("MINI_PAUSE_START")
        elif self.active_replay_name == "mini_pause_resume":
            if not row.get("resume_requested"):
                raise AssertionError("mini playback completed before Pause/Resume was observed")
            self._advance("MINI_STOP_START")
        elif self.active_replay_name == "mini_after_stop":
            self._advance("KNEE_START")
        elif self.active_replay_name == "formal_raw":
            self._advance("FAST_RESPAWN_START")
        elif self.active_replay_name == "formal_fast":
            self._advance("FINISH")

    def _stage_mini_pause_start(self) -> None:
        rows = copy.deepcopy(self.mini_rows)
        if rows:
            rows[-1]["duration"] = max(2.0, float(rows[-1].get("duration", 0.0) or 0.0))
        self._start_replay(rows, "temporary batch Pause/Resume", "raw", result_key="mini_pause_resume")

    def _stage_mini_stop_start(self) -> None:
        rows = copy.deepcopy(self.mini_rows)
        if rows:
            rows[-1]["duration"] = max(2.0, float(rows[-1].get("duration", 0.0) or 0.0))
        self._start_replay(rows, "temporary batch Stop", "raw", result_key="mini_stop")

    def _stage_mini_stop_wait(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if str(worker.get("plan_id", "") or "") != self.mini_stop_plan_id or bool(worker.get("active", False)):
            return
        if str(worker.get("stop_reason", "") or "") != "e2e_stop_validation":
            raise AssertionError(f"worker Stop acknowledgment mismatch: {worker}")
        self.result["mini_stop"]["worker_after_stop"] = copy.deepcopy(worker)
        self._advance("MINI_AFTER_STOP_START")

    def _stage_mini_after_stop_start(self) -> None:
        self._start_replay(self.mini_rows, "temporary batch immediate replay after Stop", "fast", result_key="mini_after_stop")

    def _stage_knee_start(self) -> None:
        servos = {name: (-60.0 if name in KNEE_JOINT_NAMES else 0.0) for name in SERVO_JOINT_NAMES}
        wheels = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self.active_batch_id = self.controller.apply_servo_wheel_together(servos, wheels, source="e2e_knee_negative")
        self.result["knee_test"] = {"batch_id": self.active_batch_id, "requested": servos}
        self.manual_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self._advance("KNEE_WAIT")

    def _stage_knee_wait(self) -> None:
        ack = dict(self.controller.latest_sim_status.get("last_operation_ack", {}) or {})
        elapsed = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0) - self.manual_started_sim
        if str(ack.get("batch_id", "") or "") != self.active_batch_id or elapsed < 0.5:
            return
        self.result["knee_test"].update(
            ack=copy.deepcopy(ack),
            actual_joint_state=copy.deepcopy(self.controller.latest_sim_status.get("actual_joint_state", {})),
        )
        if str(ack.get("error", "") or ""):
            raise AssertionError(f"negative knee batch failed: {ack}")
        self._capture_ui("negative_knees", "negative_knees.png")
        self.controller.stop_wheels(reason="e2e_before_ground_recalibration")
        self.recalibrate_settle_started = time.monotonic()
        self._advance("RECALIBRATE_SETTLE")

    def _stage_recalibrate_settle(self) -> None:
        if time.monotonic() - self.recalibrate_settle_started < 0.75:
            return
        self.recalibrate_attempt += 1
        self.active_recalibrate_request = self.controller.recalibrate_ground_reference()
        self.result.setdefault("ground_reference_recalibration", {"attempts": []})["attempts"].append({
            "attempt": self.recalibrate_attempt,
            "request_id": self.active_recalibrate_request,
            "status_before": {
                "grounded_reference_valid": self.controller.latest_sim_status.get("grounded_reference_valid"),
                "grounded_reference_stable": self.controller.latest_sim_status.get("grounded_reference_stable"),
                "robot_ground": copy.deepcopy(self.controller.latest_sim_status.get("robot_ground", {})),
            },
        })
        self._advance("RECALIBRATE_WAIT")

    def _stage_recalibrate_wait(self) -> None:
        ack = dict(self.controller.latest_sim_status.get("last_operation_ack", {}) or {})
        if (
            str(ack.get("operation", "") or "") != "recalibrate_ground_reference"
            or str(ack.get("request_id", "") or "") != self.active_recalibrate_request
        ):
            return
        attempt = self.result["ground_reference_recalibration"]["attempts"][-1]
        attempt.update(
            ack=copy.deepcopy(ack),
            status_after={
                "grounded_reference_valid": self.controller.latest_sim_status.get("grounded_reference_valid"),
                "grounded_reference_stable": self.controller.latest_sim_status.get("grounded_reference_stable"),
                "grounded_reference_physics_valid": self.controller.latest_sim_status.get("grounded_reference_physics_valid"),
                "respawn_ready": self.controller.latest_sim_status.get("respawn_ready"),
                "respawn_block_reason": self.controller.latest_sim_status.get("respawn_block_reason"),
            },
        )
        if not bool(ack.get("control_ready", False)) or str(ack.get("error", "") or ""):
            if self.recalibrate_attempt >= 8 or str(ack.get("error", "") or ""):
                raise AssertionError(f"ground reference recalibration failed after {self.recalibrate_attempt} attempts: {ack}")
            self.recalibrate_settle_started = time.monotonic()
            self._advance("RECALIBRATE_SETTLE")
            return
        self.result["ground_reference_recalibration"]["successful_attempt"] = self.recalibrate_attempt
        self._advance("RAW_RESPAWN_START")

    def _begin_height_respawn(self, next_stage: str) -> None:
        self.controller.current_height_mm = 50
        request_id = self.controller.generate_height_and_respawn()
        if not request_id:
            raise AssertionError(f"Generate+Respawn could not start: {self.controller.status}")
        self.active_height_request = request_id
        self.result.setdefault("height_respawn_transactions", []).append(
            {"request_id": request_id, "requested_revision": self.controller.pending_height_requested_revision}
        )
        self.result["height_respawn_transactions"][-1]["next_stage"] = next_stage
        self._advance("HEIGHT_RESPAWN_WAIT")

    def _stage_raw_respawn_start(self) -> None:
        self._begin_height_respawn("FORMAL_RAW_START")

    def _stage_fast_respawn_start(self) -> None:
        self._begin_height_respawn("FORMAL_FAST_START")

    def _stage_height_respawn_wait(self) -> None:
        ack = dict(self.controller.latest_sim_status.get("last_operation_ack", {}) or {})
        if str(ack.get("request_id", "") or "") != self.active_height_request or self.controller.pending_height_mm is not None:
            return
        row = self.result["height_respawn_transactions"][-1]
        row["ack"] = copy.deepcopy(ack)
        row["operation_after_ack"] = self.controller.operation.state.value
        if not bool(ack.get("accepted", False)) or not bool(ack.get("respawned", False)) or str(ack.get("error", "") or ""):
            raise AssertionError(f"verified height+respawn failed: {ack}")
        self._advance(str(row["next_stage"]))

    def _prepare_formal(self) -> None:
        self.formal_rows = [json.loads(line) for line in FORMAL_50_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.controller.current_height_mm = 50
        self.controller.current_version_id = "legacy_5cm_readonly"
        self.controller.current_version_metadata = {
            "legacy": True,
            "read_only": True,
            "accepted_steps_path": str(FORMAL_50_PATH.resolve()),
        }
        self.controller.manager.adopt_steps(self.formal_rows, dirty=False)
        raw_plan = plan_from_steps(
            self.formal_rows,
            profile="raw",
            max_wheel_speed=self.controller.max_wheel_speed,
            label="formal 50mm current version",
            sequence_total_steps=len(self.formal_rows),
        )
        payload = playback_plan_to_payload(raw_plan)
        decoded = playback_plan_from_payload(payload)
        integrity = validate_plan_integrity(
            decoded,
            expected_plan_sha256=raw_plan.plan_sha256,
            expected_event_count=len(raw_plan.events),
            expected_segment_count=len(raw_plan.segments),
        )
        self.result["formal_plan_integrity"] = {
            "source_path": str(FORMAL_50_PATH.resolve()),
            "source_sha256": hashlib.sha256(FORMAL_50_PATH.read_bytes()).hexdigest(),
            "input_steps": len(self.formal_rows),
            "input_raw_events": sum(len(row.get("events", []) or []) for row in self.formal_rows),
            "planner_events": len(raw_plan.events),
            "planner_segments": len(raw_plan.segments),
            "payload_events": len(payload.get("events", [])),
            "payload_segments": len(payload.get("segments", [])),
            "decoded_events": len(decoded.events),
            "decoded_segments": len(decoded.segments),
            "plan_sha256": raw_plan.plan_sha256,
            "integrity": integrity,
            "segment_128_policy": {
                "source_step": raw_plan.segments[128].source_step,
                "servo_tolerance_deg": raw_plan.segments[128].servo_tolerance_deg,
                "recorded_servo_residual_deg": raw_plan.segments[128].recorded_servo_residual_deg,
            },
        }
        if not integrity["ok"]:
            raise AssertionError(f"formal plan integrity failed: {integrity}")

    def _stage_formal_raw_start(self) -> None:
        self._prepare_formal()
        self._capture_ui("formal_50_before_raw", "formal_50_before_raw.png")
        self._start_replay(self.formal_rows, "formal 50mm Raw completion", "raw", result_key="formal_raw")

    def _stage_formal_fast_start(self) -> None:
        if not self.formal_rows:
            self._prepare_formal()
        self._start_replay(self.formal_rows, "formal 50mm Fast completion", "fast", result_key="formal_fast")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="e2e_complete")
        self.result["formal_source_sha256_after"] = hashlib.sha256(FORMAL_50_PATH.read_bytes()).hexdigest()
        if self.result["formal_source_sha256_after"] != self.result["formal_source_sha256_before"]:
            raise AssertionError("formal 50 mm source was modified")
        self.result["final_operation"] = self.controller.operation.state.value
        self.result["final_worker_playback"] = copy.deepcopy(self.controller.latest_sim_status.get("worker_playback", {}))
        self.result["success"] = True
        self._capture_ui("final_completed", "final_completed.png")
        self._capture_isaac("isaac_final", "isaac_final.png")
        self._save_gif()
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(500, lambda: self.ui._window_close(force=True))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return CompletionFixGuiE2E(Path(args.output)).run()


if __name__ == "__main__":
    raise SystemExit(main())
