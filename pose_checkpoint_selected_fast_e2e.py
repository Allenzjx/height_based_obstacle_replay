"""Visible real-Isaac capture/restore/Selected Raw+Fast transaction acceptance."""

from __future__ import annotations

import argparse
import copy
import ctypes
import ctypes.wintypes
import json
import time
from pathlib import Path
from typing import Any

from command_model import WHEEL_JOINT_NAMES
from operation_coordinator import OperationState
from sim_state_validation import FULL_VALID, validate_full_sim_pose_state
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


class PoseCheckpointSelectedFastE2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path, *, timeout_s: float) -> None:
        # Tk otherwise reports DPI-virtualized coordinates while SetCursorPos
        # consumes physical pixels, making a real mouse click miss on scaled
        # Windows desktops. This process-local setting must precede Tk creation.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        super().__init__(output_dir, timeout_s=timeout_s)
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "temporary_version_root": str(self.version_root),
            "formal_recordings_read_only": True,
            "stages": [],
            "screenshots": {},
            "recorded_steps": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }
        self.record_index = 1
        self.record_requested = False
        self.motion_started_sim = 0.0
        self.stop_requested = False
        self.perturb_started_sim = 0.0
        self.perturb_batch_id = ""
        self.detail_request_id = ""
        self.selected_seen_active = False
        self.selected_attempt = 1
        self.selected_profiles = ["raw", "fast"]
        self.selected_profile_index = 0
        self.selected_runs: list[dict[str, Any]] = []
        self.completed_selected_worker: dict[str, Any] = {}
        self.completed_selected_result: dict[str, Any] = {}
        self.completed_worker_verification: dict[str, Any] = {}
        self.completed_represented: list[int] = []
        self.pause_requested_at = 0.0
        self.pause_seen = False
        self.resume_requested = False
        self.selected_click_retry = 0
        self.conflicts_checked = False
        self.stop_restore_request_id = ""
        self.stop_restore_cancelled_at = 0.0

    def _tick(self) -> None:
        if self.ui._closing:
            return
        if bool(getattr(self, "_e2e_tick_active", False)):
            return
        self._e2e_tick_active = True
        try:
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError(f"overall timeout at {self.stage}")
            if self.stage != "WAIT_READY" and time.monotonic() - self.stage_started > 120.0:
                raise TimeoutError(f"stage timeout: {self.stage}")
            phase = str(self.controller.latest_sim_status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(self.controller.latest_sim_status.get("error", phase)))
            self._sample_worker()
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(exc)
        finally:
            self._e2e_tick_active = False
        if not self.ui._closing:
            self.ui.root.after(50, self._tick)

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        if not self.detail_request_id:
            self.detail_request_id = self.controller.transport.request_state(
                detailed=True,
                purpose="selected_e2e_home",
            )
            return
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        if str(detailed.get("state_capture_request_id", "") or "") != self.detail_request_id:
            return
        self.result["home_state"] = copy.deepcopy(detailed.get("sim_state", {}))
        self.detail_request_id = ""
        self.controller.current_height_mm = 50
        self.controller.current_version_id = "transient_pose_checkpoint_e2e"
        self.controller.current_version_metadata = {"temporary": True, "read_only": False}
        self.controller.manager.adopt_steps([], dirty=False)
        self.ui.open_right_tab("Record / Servo+Wheel")
        self.ui._refresh(force=True)
        self.result.update(
            worker_pid=int(self.controller.latest_sim_status.get("pid", 0) or 0),
            worker_session_id=str(self.controller.latest_sim_status.get("worker_session_id", "") or ""),
        )
        self._capture_ui("ready", "pose_checkpoint_ready.png")
        self._advance("RECORD_START")

    def _stage_record_start(self) -> None:
        if self.record_requested:
            self._advance("WAIT_RECORDING_ACTIVE")
            return
        if self.controller.operation.state is not OperationState.IDLE:
            return
        self.controller.start_step_recording()
        self.record_requested = True
        if self.controller.record_command_state_before is not None:
            raise AssertionError("recording exposed a start state before detailed worker acknowledgment")
        self._advance("WAIT_RECORDING_ACTIVE")

    def _stage_wait_recording_active(self) -> None:
        if not self.controller.recording_active:
            return
        start_meta = dict(self.controller.recording_capture_metadata.get("start", {}) or {})
        if str(dict(start_meta.get("validation", {}) or {}).get("classification", "")) != FULL_VALID:
            raise AssertionError(f"recording start checkpoint is not FULL_VALID: {start_meta}")
        state = self.controller.transport.capture_command_state()
        servos = dict(state.get("servos", {}) or {})
        servos["front_left_hip"] = 5.0 if self.record_index == 1 else 8.0
        batch_id = self.controller.apply_servo_wheel_together(
            servos,
            {name: 0.0 for name in WHEEL_JOINT_NAMES},
            source=f"pose_checkpoint_record_step_{self.record_index}",
        )
        self.motion_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self.result.setdefault("record_batch_ids", []).append(batch_id)
        self._advance("RECORD_MOTION")

    def _stage_record_motion(self) -> None:
        now_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        if now_sim - self.motion_started_sim < 0.75:
            return
        if not self.stop_requested:
            self.controller.stop_step_recording()
            self.stop_requested = True
        self._advance("WAIT_PENDING")

    def _stage_wait_pending(self) -> None:
        if self.controller.pending_step is None:
            return
        pending = copy.deepcopy(self.controller.pending_step)
        before_validation = validate_full_sim_pose_state(pending.get("sim_state_before"))
        after_validation = validate_full_sim_pose_state(pending.get("sim_state_after"))
        if not before_validation["valid"] or not after_validation["valid"]:
            raise AssertionError(
                f"recorded Step {self.record_index} contains an invalid checkpoint: "
                f"before={before_validation} after={after_validation}"
            )
        capture = dict(pending.get("pose_checkpoint_capture", {}) or {})
        self.result["recorded_steps"].append(
            {
                "step_index": self.record_index,
                "event_count": len(pending.get("events", []) or []),
                "before_validation": before_validation,
                "after_validation": after_validation,
                "capture": capture,
            }
        )
        self.controller.accept_pending_step()
        if self.controller.pending_step is not None:
            raise AssertionError("pending step did not accept")
        self._capture_ui(f"recorded_step_{self.record_index}", f"recorded_step_{self.record_index}.png")
        if self.record_index == 1:
            self.record_index = 2
            self.record_requested = False
            self.stop_requested = False
            self._advance("RECORD_START")
        else:
            self._advance("PERTURB")

    def _stage_perturb(self) -> None:
        if self.perturb_batch_id:
            self._advance("WAIT_PERTURB")
            return
        state = self.controller.transport.capture_command_state()
        servos = dict(state.get("servos", {}) or {})
        servos["front_left_hip"] = -10.0
        self.perturb_batch_id = self.controller.apply_servo_wheel_together(
            servos,
            {name: 0.0 for name in WHEEL_JOINT_NAMES},
            source="pose_checkpoint_pre_selected_perturb",
        )
        self.perturb_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self._advance("WAIT_PERTURB")

    def _stage_wait_perturb(self) -> None:
        now_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        if now_sim - self.perturb_started_sim < 0.75:
            return
        if not self.detail_request_id:
            self.detail_request_id = self.controller.transport.request_state(
                detailed=True,
                purpose="pre_selected_perturbed",
            )
            return
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        if str(detailed.get("state_capture_request_id", "") or "") != self.detail_request_id:
            return
        self.result["perturbed_state_before_restore"] = copy.deepcopy(detailed.get("sim_state", {}))
        self.detail_request_id = ""
        self._capture_ui("perturbed_before_selected", "perturbed_before_selected.png")
        self._advance("START_SELECTED")

    def _stage_start_selected(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        self.controller.selected_step_index = 2
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        profile = self.selected_profiles[self.selected_profile_index]
        button_label = "Play Selected Fast" if profile == "fast" else "Play Selected Step"
        button = self.ui.playback_buttons_by_label[button_label]
        playback_canvas = button.master.master
        if hasattr(playback_canvas, "yview_moveto"):
            playback_canvas.yview_moveto(0.0)
        self.ui.root.lift()
        self.ui.root.update()
        reported_x = int(button.winfo_rootx() + button.winfo_width() / 2)
        reported_y = int(button.winfo_rooty() + button.winfo_height() / 2)
        # Scrollable Tk canvases can report the content-frame Y rather than the
        # visible viewport Y. Calibrate against the widget that Tk says is
        # actually under each screen point before issuing the Win32 click.
        matching_y = [
            candidate_y
            for candidate_y in range(max(0, reported_y - 220), reported_y + 221)
            if self.ui.root.winfo_containing(reported_x, candidate_y) == button
        ]
        if not matching_y:
            raise AssertionError(
                f"could not locate visible screen pixels for {button_label}; "
                f"reported=({reported_x}, {reported_y})"
            )
        x = reported_x
        y = matching_y[len(matching_y) // 2]
        hit_widget = self.ui.root.winfo_containing(x, y)
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        desktop_dc = user32.GetDC(0)
        try:
            logical_width = max(1, int(gdi32.GetDeviceCaps(desktop_dc, 8)))
            logical_height = max(1, int(gdi32.GetDeviceCaps(desktop_dc, 10)))
            physical_width = max(1, int(gdi32.GetDeviceCaps(desktop_dc, 118)))
            physical_height = max(1, int(gdi32.GetDeviceCaps(desktop_dc, 117)))
        finally:
            user32.ReleaseDC(0, desktop_dc)
        scale_x = physical_width / logical_width
        scale_y = physical_height / logical_height
        physical_x = int(round(x * scale_x))
        physical_y = int(round(y * scale_y))
        root_hwnd = user32.GetAncestor(self.ui.root.winfo_id(), 2)
        self.ui.root.attributes("-topmost", True)
        self.ui.root.deiconify()
        self.ui.root.lift()
        self.ui.root.focus_force()
        user32.ShowWindow(root_hwnd, 9)
        user32.BringWindowToTop(root_hwnd)
        user32.SetForegroundWindow(root_hwnd)
        self.ui.root.update()
        foreground_before_click = int(user32.GetForegroundWindow())
        cursor_set = bool(user32.SetCursorPos(physical_x, physical_y))
        cursor_point = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(cursor_point))
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
        self.ui.root.update()
        time.sleep(0.10)
        self.ui.root.update()
        release_reported_x = int(button.winfo_rootx() + button.winfo_width() / 2)
        release_reported_y = int(button.winfo_rooty() + button.winfo_height() / 2)
        release_matching_y = [
            candidate_y
            for candidate_y in range(max(0, release_reported_y - 220), release_reported_y + 221)
            if self.ui.root.winfo_containing(release_reported_x, candidate_y) == button
        ]
        if release_matching_y:
            release_x = release_reported_x
            release_y = release_matching_y[len(release_matching_y) // 2]
            release_physical_x = int(round(release_x * scale_x))
            release_physical_y = int(round(release_y * scale_y))
            user32.SetCursorPos(release_physical_x, release_physical_y)
        else:
            release_x = x
            release_y = y
            release_physical_x = physical_x
            release_physical_y = physical_y
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
        self.ui.root.update()
        self.ui.root.after(500, lambda: self.ui.root.attributes("-topmost", False))
        self.result.setdefault("physical_clicks", []).append(
            {
                "attempt": self.selected_attempt,
                "retry": self.selected_click_retry,
                "profile": profile,
                "button": button_label,
                "reported_x": reported_x,
                "reported_y": reported_y,
                "x": x,
                "y": y,
                "physical_x": physical_x,
                "physical_y": physical_y,
                "logical_desktop": [logical_width, logical_height],
                "physical_desktop": [physical_width, physical_height],
                "dpi_scale": [scale_x, scale_y],
                "root_hwnd": int(root_hwnd),
                "foreground_before_click": foreground_before_click,
                "cursor_set": cursor_set,
                "cursor_after_set": [int(cursor_point.x), int(cursor_point.y)],
                "release_x": release_x,
                "release_y": release_y,
                "release_physical_x": release_physical_x,
                "release_physical_y": release_physical_y,
                "release_hit_verified": bool(release_matching_y),
                "hit_widget": str(hit_widget),
                "hit_verified": hit_widget == button,
                "clicked_at": time.time(),
            }
        )
        self.result["live_ui_click_trace"] = copy.deepcopy(self.ui.selected_fast_click_trace)
        self._advance("WAIT_SELECTED_START")

    def _stage_wait_selected_start(self) -> None:
        if self.controller.pending_selected_playback is None and not self.controller.playback.start_requested:
            if self.controller.playback.last_error:
                raise AssertionError(self.controller.playback.last_error)
            if time.monotonic() - self.stage_started >= 2.0 and self.selected_click_retry < 2:
                self.selected_click_retry += 1
                self._advance("START_SELECTED")
            return
        if not self.conflicts_checked:
            operation_before = self.controller.operation.state.value
            self.controller.start_step_recording()
            recording_blocked = not self.controller.recording_active and self.controller.operation.state.value == operation_before
            height_request_id = self.controller.generate_or_update_height_obstacle()
            height_blocked = not bool(height_request_id) and self.controller.operation.state.value == operation_before
            self.result["conflict_checks"] = {
                "operation_before": operation_before,
                "recording_blocked": recording_blocked,
                "height_generate_blocked": height_blocked,
                "operation_after": self.controller.operation.state.value,
            }
            if not recording_blocked or not height_blocked:
                raise AssertionError(f"restore conflict gate failed: {self.result['conflict_checks']}")
            self.conflicts_checked = True
        self._advance("WAIT_SELECTED")

    def _stage_wait_selected(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if bool(worker.get("active", False)):
            self.selected_seen_active = True
            profile = self.selected_profiles[self.selected_profile_index]
            if profile == "fast" and not self.pause_requested_at:
                self.controller.handle_command("pause_play")
                self.pause_requested_at = time.monotonic()
                self.result["pause_resume"] = {
                    "profile": profile,
                    "pause_requested": True,
                    "pause_requested_at": time.time(),
                }
                return
            if profile == "fast" and bool(worker.get("paused", False)):
                self.pause_seen = True
                self.result["pause_resume"]["worker_pause_seen"] = True
                if not self.resume_requested and time.monotonic() - self.pause_requested_at >= 0.25:
                    self.controller.handle_command("resume_play")
                    self.resume_requested = True
                    self.result["pause_resume"]["resume_requested"] = True
                    self.result["pause_resume"]["resume_requested_at"] = time.time()
                return
            profile = self.selected_profiles[self.selected_profile_index]
            screenshot_key = f"selected_{profile}_running"
            if screenshot_key not in self.result["screenshots"]:
                self._capture_ui(screenshot_key, f"selected_{profile}_running.png")
            return
        if not self.selected_seen_active or not str(worker.get("stop_reason", "") or ""):
            return
        if str(worker.get("stop_reason", "")) != "complete":
            raise AssertionError(f"Selected playback stopped early: {worker.get('last_error')}")
        if self.selected_profiles[self.selected_profile_index] == "fast":
            if not self.pause_seen or not self.resume_requested:
                raise AssertionError(f"Pause/Resume was not observed during Selected Fast: {self.result.get('pause_resume')}")
            self.result["pause_resume"].update(
                completed_after_resume=True,
                final_stop_reason=str(worker.get("stop_reason", "") or ""),
            )
        selected_result = copy.deepcopy(self.controller.last_selected_restore_result)
        verification = dict(selected_result.get("restore_verification", {}) or {})
        worker_verification = dict(self.controller.latest_sim_status.get("last_restore_verification", {}) or {})
        if not verification.get("verified") or not worker_verification.get("verified"):
            raise AssertionError(
                f"restore verification missing: controller={verification} worker={worker_verification}"
            )
        if selected_result.get("restore_source_step_index") != 1:
            raise AssertionError(f"wrong restore source: {selected_result}")
        plan = self.controller.playback.plan
        represented = sorted({int(event.source_step or 0) for event in list(plan.events if plan else [])})
        if represented != [2]:
            raise AssertionError(f"Selected playback represented steps are not exactly [2]: {represented}")
        if not bool(worker.get("first_command_applied", False)) or int(worker.get("events_sent", 0) or 0) <= 0:
            raise AssertionError(f"Selected playback reported no applied actuator command: {worker}")
        self.completed_selected_worker = copy.deepcopy(worker)
        self.completed_selected_result = selected_result
        self.completed_worker_verification = worker_verification
        self.completed_represented = represented
        if not self.detail_request_id:
            self.detail_request_id = self.controller.transport.request_state(
                detailed=True,
                purpose=f"selected_{self.selected_profiles[self.selected_profile_index]}_completed",
            )
        self._advance("CAPTURE_SELECTED_FINAL")

    def _stage_capture_selected_final(self) -> None:
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        if str(detailed.get("state_capture_request_id", "") or "") != self.detail_request_id:
            return
        expected = copy.deepcopy(self.controller.manager.steps[0].get("sim_state_after", {}))
        actual = copy.deepcopy(detailed.get("sim_state", {}))
        expected_servos = dict(dict(expected.get("actual_joint_state", {}) or {}).get("servos", {}) or {})
        actual_servos = dict(dict(actual.get("actual_joint_state", {}) or {}).get("servos", {}) or {})
        joint_deltas: dict[str, float] = {}
        for name, expected_row in expected_servos.items():
            actual_row = actual_servos.get(name, {})
            if "deg" in expected_row and "deg" in actual_row:
                joint_deltas[name] = abs(float(actual_row["deg"]) - float(expected_row["deg"]))
        max_actual_change_deg = max(joint_deltas.values(), default=0.0)
        if max_actual_change_deg <= 0.25:
            raise AssertionError(
                "Selected command completed but the captured Isaac actual joint state did not change "
                f"from the restored checkpoint: max_change={max_actual_change_deg:.6f} deg"
            )
        profile = self.selected_profiles[self.selected_profile_index]
        selected_run = {
            "attempt": self.selected_attempt,
            "profile": profile,
            "selected_step_index": 2,
            "seen_active": self.selected_seen_active,
            "final_worker": copy.deepcopy(self.completed_selected_worker),
            "restore_result": copy.deepcopy(self.completed_selected_result),
            "worker_restore_verification": copy.deepcopy(self.completed_worker_verification),
            "represented_step_indices": list(self.completed_represented),
            "actual_motion": {
                "captured_after_completion": True,
                "max_servo_change_from_restored_checkpoint_deg": max_actual_change_deg,
                "per_joint_change_deg": joint_deltas,
                "first_command_applied": bool(self.completed_selected_worker.get("first_command_applied", False)),
                "events_sent": int(self.completed_selected_worker.get("events_sent", 0) or 0),
            },
            "ui_click_trace": copy.deepcopy(self.ui.selected_fast_click_trace),
        }
        self.selected_runs.append(selected_run)
        self.result["selected_raw_fast_runs"] = copy.deepcopy(self.selected_runs)
        self._capture_ui(
            f"selected_{profile}_complete",
            f"selected_{profile}_complete.png",
        )
        if self.selected_profile_index + 1 < len(self.selected_profiles):
            self.selected_profile_index += 1
            self.selected_attempt += 1
            self.selected_seen_active = False
            self.perturb_batch_id = ""
            self.detail_request_id = ""
            self.completed_selected_worker = {}
            self.completed_selected_result = {}
            self.completed_worker_verification = {}
            self.completed_represented = []
            self.pause_requested_at = 0.0
            self.pause_seen = False
            self.resume_requested = False
            self.selected_click_retry = 0
            self._advance("PERTURB")
            return
        self._advance("START_STOP_RESTORE")

    def _stage_start_stop_restore(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        self.controller.selected_step_index = 2
        if not self.controller.start_selected_step_playback(2, profile="fast"):
            raise AssertionError(self.controller.playback.last_error)
        self._advance("WAIT_STOP_RESTORE")

    def _stage_wait_stop_restore(self) -> None:
        pending = self.controller.pending_selected_playback
        if self.stop_restore_cancelled_at <= 0.0:
            if pending is None:
                return
            self.stop_restore_request_id = str(pending.get("request_id", "") or "")
            self.controller.handle_command("stop_play")
            self.stop_restore_cancelled_at = time.monotonic()
            return
        if time.monotonic() - self.stop_restore_cancelled_at < 0.75:
            return
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        cancelled = dict(self.controller.last_selected_restore_result)
        self.result["stop_during_restore"] = {
            "request_id": self.stop_restore_request_id,
            "cancelled": bool(cancelled.get("cancelled", False)),
            "pending_cleared": self.controller.pending_selected_playback is None,
            "operation": self.controller.operation.state.value,
            "worker_active": bool(worker.get("active", False)),
            "result": cancelled,
        }
        if (
            not self.result["stop_during_restore"]["cancelled"]
            or not self.result["stop_during_restore"]["pending_cleared"]
            or self.controller.operation.state is not OperationState.IDLE
            or bool(worker.get("active", False))
        ):
            raise AssertionError(f"Stop during restore did not cleanly cancel: {self.result['stop_during_restore']}")
        self.result.update(
            selected_raw=copy.deepcopy(next(row for row in self.selected_runs if row["profile"] == "raw")),
            selected_fast=copy.deepcopy(next(row for row in self.selected_runs if row["profile"] == "fast")),
            repeat_count=len(self.selected_runs),
            final_operation=self.controller.operation.state.value,
        )
        self._advance("FINISH")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="pose_checkpoint_selected_fast_e2e_complete")
        self.result["success"] = True
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(300, lambda: self.ui._window_close(force=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    return PoseCheckpointSelectedFastE2E(Path(args.output), timeout_s=args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
