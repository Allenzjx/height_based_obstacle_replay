"""Visible Tk + real Isaac validation for the task-separation refactor."""

from __future__ import annotations

import argparse
import ctypes
import json
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from height_replay_ui import build_parser, normalize_motion_args
from operation_coordinator import OperationState
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


class TaskSeparationGuiE2E:
    def __init__(self, output_dir: Path, *, timeout_s: float = 600.0):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        app_args = build_parser().parse_args(
            [
                "--ui",
                "--height-cm",
                "5",
                "--worker-launch-mode",
                "current-python",
                "--sim-startup-timeout-s",
                str(timeout_s),
                "--no-telemetry",
                "--no-save-scene",
            ]
        )
        normalize_motion_args(app_args)
        self.controller = HeightReplayController(app_args)
        self.ui = RealRobotStyleHeightReplayUi(self.controller)
        self.ui.root.geometry("1450x900+20+20")
        self.ui.root.deiconify()
        self.ui.root.lift()
        self.timeout_s = float(timeout_s)
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = "WAIT_READY"
        self.results: dict[str, Any] = {
            "started_at": time.time(),
            "stages": [],
            "screenshots": {},
            "success": False,
        }
        self.pause_event_count = 0
        self.resume_event_count = 0
        self.pre_motion_state = ""
        self.original_manager_id = id(self.controller.manager)
        self.original_step_count = 0
        self.variant_index = 0
        self.variants = [
            ("Play Selected Step", 1),
            ("Respawn And Play Selected Step", 1),
            ("Play To Selected From Start", 2),
            ("Play Selected Fast", 1),
        ]

    def run(self) -> int:
        self.ui.root.after(50, self._start_worker)
        self.ui.root.after(200, self._tick)
        self.ui.run()
        return 0 if self.results.get("success") else 1

    def _start_worker(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self._fail(f"worker start failed: {exc}")

    def _tick(self) -> None:
        if self.ui._closing:
            return
        try:
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError(f"overall E2E timeout at stage {self.stage}")
            if time.monotonic() - self.stage_started > 120.0 and self.stage != "WAIT_READY":
                raise TimeoutError(f"stage timeout: {self.stage}")
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        if not self.ui._closing:
            self.ui.root.after(200, self._tick)

    def _advance(self, stage: str, **details: Any) -> None:
        self.results["stages"].append(
            {"stage": self.stage, "completed_at": time.time(), "details": details}
        )
        self.stage = stage
        self.stage_started = time.monotonic()

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        status = self.controller.latest_sim_status
        self.results["isaac_pid"] = int(status.get("pid", 0) or 0)
        self.results["runtime_ready"] = True
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
        self._capture("main_interface", "main_interface.png")
        self.ui.open_right_tab("Height Generate")
        self.ui.height_var.set("5")
        self.ui.height_generate_panel.generate()
        self._advance("WAIT_SCENE", tabs=tabs)

    def _stage_wait_scene(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        if int(self.controller.loaded_sim_height_cm or -1) != 5:
            return
        count = self.ui.height_generate_panel.load_steps()
        if count <= 0:
            raise AssertionError(self.controller.status)
        if id(self.controller.manager) != self.original_manager_id:
            raise AssertionError("Height Generate replaced the shared SequenceManager instance")
        if self.controller.playback.active:
            raise AssertionError("Height Generate started playback")
        self.original_step_count = count
        self.results["height_05cm"] = {
            "path": str(self.controller.manager.accepted_path),
            "step_count": count,
        }
        self._capture("height_05cm_loaded", "height_05cm_loaded.png")
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        self._require_enabled("Play All")
        self._require_enabled("Play All Fast")
        self._capture("play_all_enabled", "play_all_enabled.png")
        self.pre_motion_state = self._actual_state_json()
        self.ui.playback_buttons_by_label["Play All"].invoke()
        self._advance("WAIT_ACTIVE")

    def _stage_wait_active(self) -> None:
        worker = self._worker_playback()
        if not (self.controller.playback.active and bool(worker.get("active", False))):
            return
        if int(worker.get("events_sent", 0) or 0) < 1:
            return
        actual_changed = self._actual_state_json() != self.pre_motion_state
        self.results["play_all"] = {
            "active": True,
            "events_sent": int(worker.get("events_sent", 0) or 0),
            "last_event_command": str(worker.get("last_event_command", "") or ""),
            "actual_joint_state_changed": actual_changed,
        }
        if not actual_changed:
            raise AssertionError("worker sent playback event but actual joint state did not change")
        self._capture("playback_active", "playback_active.png")
        self._require_enabled("Pause Play")
        self.ui.playback_buttons_by_label["Pause Play"].invoke()
        self.pause_event_count = int(worker.get("events_sent", 0) or 0)
        self._advance("WAIT_PAUSED")

    def _stage_wait_paused(self) -> None:
        worker = self._worker_playback()
        if not (self.controller.playback.paused and bool(worker.get("paused", False))):
            return
        if time.monotonic() - self.stage_started < 2.0:
            return
        current = int(worker.get("events_sent", 0) or 0)
        if current != self.pause_event_count:
            raise AssertionError(f"events advanced while paused: {self.pause_event_count} -> {current}")
        self._require_enabled("Resume Play")
        self.ui.playback_buttons_by_label["Resume Play"].invoke()
        self.resume_event_count = current
        self._advance("WAIT_RESUMED")

    def _stage_wait_resumed(self) -> None:
        worker = self._worker_playback()
        if bool(worker.get("paused", False)):
            return
        if int(worker.get("events_sent", 0) or 0) <= self.resume_event_count:
            return
        self.results["pause_resume"] = {
            "paused_events_sent": self.pause_event_count,
            "resumed_events_sent": int(worker.get("events_sent", 0) or 0),
        }
        self._require_enabled("Stop Play")
        self.ui.playback_buttons_by_label["Stop Play"].invoke()
        self._advance("WAIT_STOPPED")

    def _stage_wait_stopped(self) -> None:
        worker = self._worker_playback()
        if self.controller.playback.active or bool(worker.get("active", False)):
            return
        if self.controller.playback.plan is not None:
            raise AssertionError("Playback queue was not cleared")
        self.ui._refresh(force=True)
        self._require_enabled("Play All")
        self._capture("playback_idle_after_stop", "playback_idle_after_stop.png")
        self.results["stop"] = {"active": False, "scheduled": False, "queue_cleared": True}
        self._advance("START_VARIANT")

    def _stage_start_variant(self) -> None:
        if self.variant_index >= len(self.variants):
            self._advance("RECORD_CONFLICT")
            return
        label, selected = self.variants[self.variant_index]
        self._select_step(selected)
        self.ui._refresh_button_states()
        self._require_enabled(label)
        self.ui.playback_buttons_by_label[label].invoke()
        self._advance("WAIT_VARIANT_ACTIVE", label=label, selected=selected)

    def _stage_wait_variant_active(self) -> None:
        worker = self._worker_playback()
        if not (self.controller.playback.active and bool(worker.get("active", False))):
            return
        label, selected = self.variants[self.variant_index]
        self.results.setdefault("playback_variants", []).append(
            {"label": label, "selected_step": selected, "worker_plan_id": str(worker.get("plan_id", "") or "")}
        )
        self.ui.playback_buttons_by_label["Stop Play"].invoke()
        self._advance("WAIT_VARIANT_STOP")

    def _stage_wait_variant_stop(self) -> None:
        if self.controller.playback.active or bool(self._worker_playback().get("active", False)):
            return
        self.variant_index += 1
        self._advance("START_VARIANT")

    def _stage_record_conflict(self) -> None:
        self.ui._post("step_record start")
        if self.controller.operation.state is not OperationState.RECORDING:
            raise AssertionError("Record did not acquire the shared operation coordinator")
        blocked = self.controller.playback_availability()
        if blocked.can_start or blocked.reason != "Recording is active.":
            raise AssertionError(f"Playback was not blocked by recording: {blocked}")
        self.ui._post("step_record stop")
        if self.controller.operation.state is not OperationState.IDLE:
            raise AssertionError("Record stop did not release the operation coordinator")
        if self.controller.pending_step is not None:
            self.ui._post("step_record discard")
        if not self.controller.playback_availability().can_start:
            raise AssertionError(self.controller.playback_availability().reason)
        self.results["record_stop_recovery"] = True
        self._advance("COMBINE")

    def _stage_combine(self) -> None:
        before = self.controller.manager.count
        self.ui._post("combine mode")
        self.controller.set_combine_selection([1, 2])
        self.controller.commit_combine_steps()
        if id(self.controller.manager) != self.original_manager_id:
            raise AssertionError("Combine replaced the shared SequenceManager")
        if self.controller.manager.count != before - 1:
            raise AssertionError("Combine did not publish its in-memory result to the shared sequence")
        if self.controller.playback.active:
            raise AssertionError("Combine started playback")
        if not self.controller.playback_availability().can_start:
            raise AssertionError(self.controller.playback_availability().reason)
        invariant = self.controller.playback_availability().can_start
        for tab_id in self.ui.right_notebook.tabs():
            self.ui.right_notebook.select(tab_id)
            self.ui.root.update_idletasks()
            if self.controller.playback_availability().can_start != invariant:
                raise AssertionError("Switching tabs changed Playback availability")
        self.results["combine"] = {
            "before_count": before,
            "after_count": self.controller.manager.count,
            "shared_manager": True,
            "auto_playback": False,
        }
        self.results["tab_switch_availability_invariant"] = True
        self._finish_success()

    def _select_step(self, index: int) -> None:
        children = list(self.ui.steps_tree.get_children())
        if not children:
            raise AssertionError("steps tree is empty")
        target = children[int(index) - 1]
        self.ui.steps_tree.selection_set(target)
        self.ui.steps_tree.focus(target)
        self.ui._on_step_selected(None)

    def _worker_playback(self) -> dict[str, Any]:
        value = self.controller.latest_sim_status.get("worker_playback", {})
        return dict(value) if isinstance(value, dict) else {}

    def _actual_state_json(self) -> str:
        state = self.controller.latest_sim_status.get("actual_joint_state", {})
        return json.dumps(state, sort_keys=True, default=str)

    def _require_enabled(self, label: str) -> None:
        button = self.ui.playback_buttons_by_label[label]
        if "disabled" in button.state():
            raise AssertionError(f"{label} is disabled: {self.ui.playback_unavailable_var.get()}")

    def _capture(self, key: str, filename: str) -> None:
        self.ui.root.update_idletasks()
        paned = self.ui.main_paned
        pane_ids = list(paned.panes())
        hidden_panes: list[tuple[int, str]] = []
        if key == "main_interface":
            for original_index, pane_id in reversed(list(enumerate(pane_ids[:2]))):
                paned.forget(pane_id)
                hidden_panes.append((original_index, pane_id))
        elif key == "height_05cm_loaded":
            paned.forget(pane_ids[0])
            hidden_panes.append((0, pane_ids[0]))
            self.ui.root.update_idletasks()
            paned.sashpos(0, min(int(paned.winfo_width()) - 560, 760))
        else:
            for original_index, pane_id in reversed(list(enumerate(pane_ids[:2]))):
                paned.forget(pane_id)
                hidden_panes.append((original_index, pane_id))
        hwnd = int(self.ui.root.winfo_id())
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        self.ui.root.attributes("-topmost", True)
        self.ui.root.lift()
        self.ui.root.focus_force()
        self.ui.root.update_idletasks()
        self.ui.root.update()
        x = self.ui.root.winfo_rootx()
        y = self.ui.root.winfo_rooty()
        width = self.ui.root.winfo_width()
        height = self.ui.root.winfo_height()
        path = self.output_dir / filename
        try:
            ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True).save(path)
        finally:
            self.ui.root.attributes("-topmost", False)
            for original_index, pane_id in sorted(hidden_panes):
                paned.insert(original_index, pane_id, weight=2)
            self.ui.root.update_idletasks()
        self.results["screenshots"][key] = str(path)

    def _finish_success(self) -> None:
        self.results["success"] = True
        self.results["finished_at"] = time.time()
        self.results["controller_status_log_tail"] = self.controller.status_log[-100:]
        self._write_result()
        self.ui.root.after(250, self.ui._window_close)

    def _fail(self, error: str, tb: str = "") -> None:
        if self.ui._closing:
            return
        self.results["success"] = False
        self.results["error"] = error
        self.results["traceback"] = tb
        self.results["failed_stage"] = self.stage
        self.results["latest_sim_status"] = self.controller.latest_sim_status
        self.results["controller_status_log_tail"] = self.controller.status_log[-100:]
        try:
            self._capture("failure", "failure.png")
        except Exception:
            pass
        self._write_result()
        self.ui.root.after(250, self.ui._window_close)

    def _write_result(self) -> None:
        (self.output_dir / "gui_isaac_result.json").write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    return TaskSeparationGuiE2E(Path(args.output), timeout_s=args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
