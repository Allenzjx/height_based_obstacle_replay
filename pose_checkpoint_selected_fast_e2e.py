"""Visible real-Isaac capture/restore/Selected-Fast transaction acceptance."""

from __future__ import annotations

import argparse
import copy
import ctypes
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
        self.selected_runs: list[dict[str, Any]] = []
        self.conflicts_checked = False
        self.stop_restore_request_id = ""
        self.stop_restore_cancelled_at = 0.0

    def _tick(self) -> None:
        if self.ui._closing:
            return
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
        self._capture_ui("perturbed_before_selected", "perturbed_before_selected.png")
        self._advance("START_SELECTED")

    def _stage_start_selected(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        self.controller.selected_step_index = 2
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        button = self.ui.playback_buttons_by_label["Play Selected Fast"]
        self.ui.root.lift()
        self.ui.root.update()
        x = int(button.winfo_rootx() + button.winfo_width() / 2)
        y = int(button.winfo_rooty() + button.winfo_height() / 2)
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
        self.result.setdefault("physical_clicks", []).append(
            {"attempt": self.selected_attempt, "x": x, "y": y, "clicked_at": time.time()}
        )
        self._advance("WAIT_SELECTED_START")

    def _stage_wait_selected_start(self) -> None:
        if self.controller.pending_selected_playback is None and not self.controller.playback.start_requested:
            if self.controller.playback.last_error:
                raise AssertionError(self.controller.playback.last_error)
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
            if "selected_running" not in self.result["screenshots"]:
                self._capture_ui("selected_running", "selected_fast_running.png")
            return
        if not self.selected_seen_active or not str(worker.get("stop_reason", "") or ""):
            return
        if str(worker.get("stop_reason", "")) != "complete":
            raise AssertionError(f"Selected Fast stopped early: {worker.get('last_error')}")
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
            raise AssertionError(f"Selected Fast represented steps are not exactly [2]: {represented}")
        selected_run = {
                "attempt": self.selected_attempt,
                "selected_step_index": 2,
                "seen_active": self.selected_seen_active,
                "final_worker": copy.deepcopy(worker),
                "restore_result": selected_result,
                "worker_restore_verification": worker_verification,
                "represented_step_indices": represented,
            }
        self.selected_runs.append(selected_run)
        self.result["selected_fast_runs"] = copy.deepcopy(self.selected_runs)
        self._capture_ui(
            f"selected_complete_{self.selected_attempt}",
            f"selected_fast_complete_{self.selected_attempt}.png",
        )
        if self.selected_attempt < 3:
            self.selected_attempt += 1
            self.selected_seen_active = False
            self.perturb_batch_id = ""
            self.detail_request_id = ""
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
            selected_fast=copy.deepcopy(self.selected_runs[-1]),
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
