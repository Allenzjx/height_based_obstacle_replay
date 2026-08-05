"""Visible, single-worker acceptance for previous-saved-state Selected playback."""

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

from operation_coordinator import OperationState
from playback_progress import PlaybackState
from sequence_model import load_steps_jsonl
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


FORMAL_SOURCE = PROJECT_ROOT / "saved_height_steps" / "height_05cm" / "accepted_steps.jsonl"


class SelectedPreviousStateGuiE2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir, timeout_s=600.0)
        self.formal_sha_before = hashlib.sha256(FORMAL_SOURCE.read_bytes()).hexdigest()
        self.manual_target = -20.0
        self.manual_started_sim = 0.0
        self.active_profile = ""
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "reused_existing_process": False,
            "formal_source_path": str(FORMAL_SOURCE.resolve()),
            "formal_source_sha256_before": self.formal_sha_before,
            "stages": [],
            "screenshots": {},
            "selected_runs": {},
            "conflicts": {},
            "other_playback_controls": {},
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        rows = load_steps_jsonl(FORMAL_SOURCE)
        if len(rows) < 5:
            raise AssertionError(f"formal source has only {len(rows)} steps")
        self.controller.current_height_mm = 50
        self.controller.current_version_id = "formal_50mm_read_only_e2e"
        self.controller.current_version_metadata = {
            "read_only": True,
            "accepted_steps_path": str(FORMAL_SOURCE.resolve()),
            "accepted_steps_sha256": self.formal_sha_before,
        }
        self.controller.manager.adopt_steps(rows, dirty=False)
        self.controller.selected_step_index = 5
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        item = "step_5"
        if item in self.ui.steps_tree.get_children():
            self.ui.steps_tree.selection_set(item)
            self.ui.steps_tree.see(item)
        step4 = self.controller.manager.get_step(4)
        step5 = self.controller.manager.get_step(5)
        resolved = self.controller.resolve_selected_step_restore_state(5)
        if resolved["restore_source_step_index"] != 4 or resolved["restore_source_field"] != "sim_state_after":
            raise AssertionError(f"Step 5 did not resolve Step 4.sim_state_after: {resolved}")
        self.result.update(
            worker_pid=int(self.controller.latest_sim_status.get("pid", 0) or 0),
            worker_session_id=str(self.controller.latest_sim_status.get("worker_session_id", "") or ""),
            isaac_version=str(self.controller.latest_sim_status.get("isaac_version", "") or ""),
            loaded_step_count=len(rows),
            selected_step=5,
            previous_step=4,
            step4_sim_state_after=copy.deepcopy(step4.get("sim_state_after")),
            step4_command_state_after=copy.deepcopy(step4.get("command_state_after")),
            step5_sim_state_before=copy.deepcopy(step5.get("sim_state_before")),
            step5_command_state_before=copy.deepcopy(step5.get("command_state_before")),
            initial_resolution={
                key: copy.deepcopy(value)
                for key, value in resolved.items()
                if key not in {"selected_step", "restore_sim_state", "restore_command_state"}
            },
        )
        self._advance("MANUAL_RAW_START")

    def _stage_manual_raw_start(self) -> None:
        self.controller.handle_command(f"servo front_left_hip {self.manual_target}")
        self.manual_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self._advance("MANUAL_RAW_WAIT")

    def _stage_manual_raw_wait(self) -> None:
        state = dict(self.controller.latest_sim_status.get("command_state", {}) or {})
        servos = dict(state.get("servos", {}) or {})
        elapsed = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0) - self.manual_started_sim
        if abs(float(servos.get("front_left_hip", 0.0)) - self.manual_target) > 1.0e-6 or elapsed < 0.5:
            return
        self.controller.transport.request_state(detailed=True)
        self.result["manual_state_before_raw"] = copy.deepcopy(state)
        step4_state = dict(self.result["step4_command_state_after"] or {})
        if state == step4_state:
            raise AssertionError("manual state is not visibly different from Step 4 saved end state")
        self._capture_ui("current_robot_different", "current_robot_different.png")
        self._capture_isaac("isaac_current_robot_different", "isaac_current_robot_different.png")
        self._advance("RAW_START")

    def _start_selected(self, profile: str) -> None:
        self.controller.selected_step_index = 5
        self.ui.steps_tree.selection_set("step_5")
        self.active_profile = profile
        self.result["selected_runs"][profile] = {
            "started_at": time.time(),
            "invoked_button": "Play Selected Fast" if profile == "fast" else "Play Selected Step",
            "worker_plan_before": str(
                dict(self.controller.latest_sim_status.get("worker_playback", {}) or {}).get("plan_id", "") or ""
            ),
        }
        button = self.ui.playback_buttons_by_label[self.result["selected_runs"][profile]["invoked_button"]]
        if str(button.cget("state")) == "disabled":
            raise AssertionError(f"selected playback button unexpectedly disabled: {self.controller.can_playback()[1]}")
        button.invoke()
        pending = self.controller.pending_selected_playback
        if pending is None:
            raise AssertionError("Selected playback did not enter RESTORING")
        self.result["selected_runs"][profile].update(
            restoring_status=self.controller.status,
            restore_request_id=pending["request_id"],
            restore_source_step_index=pending["restore_source_step_index"],
            restore_source_field=pending["restore_source_field"],
            operation_while_restoring=self.controller.operation.state.value,
            playback_state_while_restoring=self.controller.playback.progress.playback_state,
        )

    def _stage_raw_start(self) -> None:
        self._start_selected("raw")
        self.ui._refresh(force=True)
        self._capture_ui("restoring_previous_step", "restoring_saved_end_from_step4.png")
        self._advance("RAW_WAIT_RESTORE")

    def _record_restore_started(self, profile: str) -> bool:
        result = self.controller.last_selected_restore_result
        if not result.get("started") or result.get("profile") != profile:
            return False
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        run = self.result["selected_runs"][profile]
        run.update(
            restore_result=copy.deepcopy(result),
            detailed_state_after_restore=copy.deepcopy(detailed.get("sim_state", {})),
            worker_after_start=copy.deepcopy(worker),
            expected_worker_plan_id=str(self.controller.playback.worker_plan_id or ""),
            expected_worker_request_id=str(self.controller.playback.worker_request_id or ""),
            plan_event_count=len(self.controller.playback.plan.events) if self.controller.playback.plan else 0,
            plan_source_steps=sorted(
                {int(event.source_step or 0) for event in self.controller.playback.plan.events}
            )
            if self.controller.playback.plan
            else [],
            plan_selected_playback=bool(self.controller.playback.plan and self.controller.playback.plan.selected_playback),
        )
        if result["restore_source_step_index"] != 4 or result["restore_source_field"] != "sim_state_after":
            raise AssertionError(f"wrong restore source: {result}")
        if not run["expected_worker_plan_id"] or not run["expected_worker_request_id"]:
            raise AssertionError(f"selected playback did not allocate worker identity: {run}")
        if run["plan_source_steps"] != [5] or not run["plan_selected_playback"]:
            raise AssertionError(f"plan includes a non-selected step: {run}")
        events = [row["event"] for row in result["trace"]]
        required = [
            "operation_acquired",
            "stop_wheels_sent",
            "restore_sent",
            "restore_acknowledged",
            "state_requested",
            "state_verified",
            "playback_start_requested",
        ]
        if any(name not in events for name in required):
            raise AssertionError(f"restore trace incomplete: {events}")
        if [events.index(name) for name in required] != sorted(events.index(name) for name in required):
            raise AssertionError(f"restore trace out of order: {events}")
        return True

    def _stage_raw_wait_restore(self) -> None:
        if not self._record_restore_started("raw"):
            return
        self._capture_ui("restore_complete", "restore_complete_starting_step5.png")
        self._advance("RAW_WAIT_PLAYING")

    def _stage_raw_wait_playing(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        run = self.result["selected_runs"]["raw"]
        if (
            str(worker.get("plan_id", "") or "") != run["expected_worker_plan_id"]
            or str(worker.get("request_id", "") or "") != run["expected_worker_request_id"]
        ):
            return
        progress = dict(worker.get("progress_detail", {}) or {})
        if not worker.get("active") or progress.get("playback_state") != PlaybackState.PLAYING.value:
            return
        if int(progress.get("current_step_index", 0) or 0) != 5:
            raise AssertionError(f"Raw progress does not highlight Step 5: {progress}")
        self.result["selected_runs"]["raw"]["playing_worker"] = copy.deepcopy(worker)
        self._capture_ui("selected_step_started", "selected_step5_playing.png")
        self._advance("RAW_WAIT_COMPLETE")

    def _stage_raw_wait_complete(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        run = self.result["selected_runs"]["raw"]
        if (
            str(worker.get("plan_id", "") or "") != run["expected_worker_plan_id"]
            or str(worker.get("request_id", "") or "") != run["expected_worker_request_id"]
        ):
            return
        if worker.get("active") or str(worker.get("stop_reason", "") or "") != "complete":
            return
        progress = dict(worker.get("progress_detail", {}) or {})
        if int(progress.get("current_step_index", 0) or 0) != 5:
            raise AssertionError(f"Raw completion lost selected Step 5 progress: {progress}")
        self.result["selected_runs"]["raw"]["final_worker"] = copy.deepcopy(worker)
        self._capture_ui("selected_step_completed", "selected_step5_completed.png")
        self._advance("MANUAL_FAST_START")

    def _stage_manual_fast_start(self) -> None:
        self.controller.handle_command("servo front_left_hip -15")
        self.manual_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self._advance("MANUAL_FAST_WAIT")

    def _stage_manual_fast_wait(self) -> None:
        servos = dict(dict(self.controller.latest_sim_status.get("command_state", {}) or {}).get("servos", {}) or {})
        elapsed = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0) - self.manual_started_sim
        if abs(float(servos.get("front_left_hip", 0.0)) + 15.0) > 1.0e-6 or elapsed < 0.4:
            return
        self.result["manual_state_before_fast"] = copy.deepcopy(self.controller.latest_sim_status.get("command_state", {}))
        self._advance("FAST_START")

    def _stage_fast_start(self) -> None:
        self._start_selected("fast")
        self._advance("FAST_WAIT_RESTORE")

    def _stage_fast_wait_restore(self) -> None:
        if not self._record_restore_started("fast"):
            return
        self._advance("FAST_WAIT_COMPLETE")

    def _stage_fast_wait_complete(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        run = self.result["selected_runs"]["fast"]
        if (
            str(worker.get("plan_id", "") or "") != run["expected_worker_plan_id"]
            or str(worker.get("request_id", "") or "") != run["expected_worker_request_id"]
        ):
            return
        if worker.get("active") or str(worker.get("stop_reason", "") or "") != "complete":
            return
        self.result["selected_runs"]["fast"]["final_worker"] = copy.deepcopy(worker)
        raw_plan = self.result["selected_runs"]["raw"]["restore_result"]
        fast_plan = self.result["selected_runs"]["fast"]["restore_result"]
        if raw_plan["plan_source_steps"] != fast_plan["plan_source_steps"]:
            raise AssertionError("Raw/Fast selected plan source differs")
        self._capture_ui("selected_fast_completed", "selected_step5_fast_completed.png")
        self._advance("STEP1_START")

    def _stage_step1_start(self) -> None:
        self.controller.selected_step_index = 1
        self.ui.steps_tree.selection_set("step_1")
        button = self.ui.playback_buttons_by_label["Play Selected Fast"]
        if str(button.cget("state")) == "disabled":
            raise AssertionError(f"Step 1 selected button unexpectedly disabled: {self.controller.can_playback()[1]}")
        button.invoke()
        if self.controller.pending_selected_playback is None:
            raise AssertionError(self.controller.status)
        self.result["step1_invoked_button"] = "Play Selected Fast"
        self.result["step1_expected_worker"] = {}
        self._advance("STEP1_WAIT_RESTORE")

    def _stage_step1_wait_restore(self) -> None:
        result = self.controller.last_selected_restore_result
        if not result.get("started") or result.get("selected_step_index") != 1:
            return
        if result.get("restore_source_step_index") != 1 or result.get("restore_source_field") != "sim_state_before":
            raise AssertionError(f"Step 1 wrong restore policy: {result}")
        if result.get("plan_source_steps") != [1]:
            raise AssertionError(f"Step 1 plan source mismatch: {result}")
        self.result["step1_restore"] = copy.deepcopy(result)
        self.result["step1_expected_worker"] = {
            "plan_id": str(self.controller.playback.worker_plan_id or ""),
            "request_id": str(self.controller.playback.worker_request_id or ""),
        }
        if not all(self.result["step1_expected_worker"].values()):
            raise AssertionError("Step 1 selected playback did not allocate worker identity")
        self._advance("STEP1_WAIT_COMPLETE")

    def _stage_step1_wait_complete(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        expected = self.result["step1_expected_worker"]
        if (
            str(worker.get("plan_id", "") or "") != expected["plan_id"]
            or str(worker.get("request_id", "") or "") != expected["request_id"]
        ):
            return
        if worker.get("active") or str(worker.get("stop_reason", "") or "") != "complete":
            return
        self.result["step1_final_worker"] = copy.deepcopy(worker)
        self._advance("STOP_DURING_RESTORE")

    def _stage_stop_during_restore(self) -> None:
        self.controller.selected_step_index = 5
        self.ui.steps_tree.selection_set("step_5")
        prior_plan_id = str(dict(self.controller.latest_sim_status.get("worker_playback", {}) or {}).get("plan_id", "") or "")
        play_button = self.ui.playback_buttons_by_label["Play Selected Step"]
        if str(play_button.cget("state")) == "disabled":
            raise AssertionError(f"Stop test selected button unexpectedly disabled: {self.controller.can_playback()[1]}")
        play_button.invoke()
        pending = self.controller.pending_selected_playback
        if pending is None:
            raise AssertionError("restore was not pending before Stop")
        request_id = str(pending["request_id"])
        stop_button = self.ui.playback_buttons_by_label["Stop Play"]
        if str(stop_button.cget("state")) == "disabled":
            raise AssertionError("Stop Play button was disabled during selected restore")
        stop_button.invoke()
        if self.controller.pending_selected_playback is not None or self.controller.operation.state is not OperationState.IDLE:
            raise AssertionError("Stop did not cancel pending selected restore")
        self.result["stop_during_restore"] = {
            "request_id": request_id,
            "worker_plan_id_before": prior_plan_id,
            "cancel_result": copy.deepcopy(self.controller.last_selected_restore_result),
            "stopped_at": time.time(),
        }
        self._advance("STOP_DURING_RESTORE_WAIT")

    def _stage_stop_during_restore_wait(self) -> None:
        row = self.result["stop_during_restore"]
        if str(self.controller.latest_sim_status.get("last_restore_request_id", "") or "") != row["request_id"]:
            return
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if worker.get("active"):
            raise AssertionError("worker started playback after Stop during restore")
        row["worker_after_restore_ack"] = copy.deepcopy(worker)
        row["operation_after"] = self.controller.operation.state.value
        row["pending_after"] = self.controller.pending_selected_playback is not None
        self._advance("RECORD_CONFLICT_START")

    def _stage_record_conflict_start(self) -> None:
        self.controller.start_step_recording()
        if not self.controller.recording_active or self.controller.operation.state is not OperationState.RECORDING:
            raise AssertionError(f"could not start temporary recording conflict: {self.controller.status}")
        allowed = self.controller.start_selected_step_playback(5, profile="raw")
        self.result["conflicts"]["recording"] = {
            "selected_started": allowed,
            "operation": self.controller.operation.state.value,
            "status": self.controller.status,
        }
        if allowed:
            raise AssertionError("Selected playback started during Recording")
        self.controller.stop_step_recording()
        self._advance("RECORD_CONFLICT_WAIT")

    def _stage_record_conflict_wait(self) -> None:
        if self.controller.recording_active or self.controller.record_stop_pending is not None:
            return
        if self.controller.pending_step is not None:
            self.controller.discard_pending_step(restore_before=False)
        self._advance("HEIGHT_CONFLICT_START")

    def _stage_height_conflict_start(self) -> None:
        self.controller.current_height_mm = 50
        request_id = self.controller.generate_or_update_height_obstacle()
        if not request_id:
            raise AssertionError(self.controller.status)
        allowed = self.controller.start_selected_step_playback(5, profile="raw")
        self.result["conflicts"]["height_generate"] = {
            "request_id": request_id,
            "selected_started": allowed,
            "operation": self.controller.operation.state.value,
            "status": self.controller.status,
        }
        if allowed:
            raise AssertionError("Selected playback started during Height Generate")
        self._advance("HEIGHT_CONFLICT_WAIT")

    def _stage_height_conflict_wait(self) -> None:
        if self.controller.pending_height_mm is not None:
            return
        self.result["conflicts"]["height_generate"]["operation_after"] = self.controller.operation.state.value
        labels = [
            "Play All",
            "Play All Fast",
            "Respawn And Play Selected Step",
            "Respawn And Play Selected Fast",
            "Play To Selected From Start",
            "Respawn And Play To Selected From Start",
            "Pause Play",
            "Resume Play",
            "Stop Play",
        ]
        self.result["other_playback_controls"] = {
            "labels_expected": labels,
            "unchanged_by_source_diff": True,
            "unit_regression_coverage": "tests.test_selected_step_previous_saved_state.ConflictAndRegressionTest",
        }
        self._advance("FINISH")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="selected_e2e_complete")
        self.result["formal_source_sha256_after"] = hashlib.sha256(FORMAL_SOURCE.read_bytes()).hexdigest()
        if self.result["formal_source_sha256_after"] != self.formal_sha_before:
            raise AssertionError("formal saved sequence changed")
        if self.controller.operation.state is not OperationState.IDLE:
            raise AssertionError(f"final operation is {self.controller.operation.state.value}")
        self.result["final_operation"] = self.controller.operation.state.value
        self.result["final_playback"] = copy.deepcopy(self.controller.latest_sim_status.get("worker_playback", {}))
        self.result["success"] = True
        self._capture_ui("final", "final_selected_validation.png")
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
    return SelectedPreviousStateGuiE2E(Path(args.output)).run()


if __name__ == "__main__":
    raise SystemExit(main())
