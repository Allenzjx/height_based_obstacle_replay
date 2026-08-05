"""Visible single-worker physical-click acceptance for Play Selected Fast."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operation_coordinator import OperationState
from playback import PlaybackPlan, plan_from_steps
from sequence_model import load_steps_jsonl
from ui_motion_speed_height_version_e2e import RefactorGuiE2E


FORMAL_SOURCE = PROJECT_ROOT / "saved_height_steps" / "height_05cm" / "accepted_steps.jsonl"


class PhysicalFastClickE2E(RefactorGuiE2E):
    def __init__(self, output_dir: Path, *, repeat_count: int = 1) -> None:
        super().__init__(output_dir, timeout_s=600.0)
        self.repeat_count = max(1, int(repeat_count))
        self.run_number = 0
        self.click_id = ""
        self.click_started = 0.0
        self.trace: list[dict[str, Any]] = []
        self.trace_names: set[str] = set()
        self.restore_observed_state: dict[str, Any] = {}
        self.actual_motion_state: dict[str, Any] = {}
        self.expected_worker_plan_id = ""
        self.expected_worker_request_id = ""
        self.last_detail_request = 0.0
        self.manual_started_sim = 0.0
        self.physical_click_in_progress = False
        self.result = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "single_worker_requested": True,
            "physical_mouse_required": True,
            "formal_source_path": str(FORMAL_SOURCE.resolve()),
            "formal_source_sha256_before": hashlib.sha256(FORMAL_SOURCE.read_bytes()).hexdigest(),
            "stages": [],
            "screenshots": {},
            "runs": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }

    def _record(self, event: str, *, once: bool = True, **details: Any) -> None:
        if once and event in self.trace_names:
            return
        now = time.monotonic()
        row = {
            "event": event,
            "monotonic_s": now,
            "relative_ms": (now - self.click_started) * 1000.0 if self.click_started else None,
            **copy.deepcopy(details),
        }
        self.trace.append(row)
        self.trace_names.add(event)

    def _record_existing(self, event: str, monotonic_s: float, **details: Any) -> None:
        if event in self.trace_names:
            return
        self.trace.append(
            {
                "event": event,
                "monotonic_s": float(monotonic_s),
                "relative_ms": (float(monotonic_s) - self.click_started) * 1000.0,
                **copy.deepcopy(details),
            }
        )
        self.trace_names.add(event)

    @staticmethod
    def _plan_summary(plan: PlaybackPlan) -> dict[str, Any]:
        return {
            "profile": plan.profile,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "source_steps": sorted({int(event.source_step or 0) for event in plan.events}),
            "command_signature": [str(event.command) for event in plan.events],
            "servo_targets": [list(event.servo_targets) for event in plan.events],
            "wheel_targets": [list(event.wheel_applied_target_rad_s) for event in plan.events],
            "wheel_durations": [
                float(event.wheel_active_duration_s)
                for event in plan.events
                if event.wheel_applied_target_rad_s
            ],
            "final_time_s": float(plan.final_time_s),
            "plan_sha256": str(plan.plan_sha256),
        }

    @staticmethod
    def _actual_wheels(sim_state: dict[str, Any]) -> dict[str, float]:
        rows = dict(dict(sim_state.get("actual_joint_state", {}) or {}).get("wheels", {}) or {})
        result: dict[str, float] = {}
        for name, row in rows.items():
            try:
                result[str(name)] = float(dict(row or {}).get("rad_s", 0.0))
            except (TypeError, ValueError):
                continue
        return result

    def _install_boundary_wrappers(self) -> None:
        original_ui_command = self.ui._selected_step_playback_command

        def ui_command(template: str) -> None:
            self._record("tk_button_command_entered", template=template)
            tree_indices = self.ui._selected_indices()
            index = self.ui._selected_index()
            self._record(
                "selected_index_resolved",
                tree_indices=tree_indices,
                ui_selected_step_index=self.ui.selected_step_index,
                controller_selected_step_index=self.controller.selected_step_index,
                resolved_index=index,
            )
            ok, reason = self.controller.playback_readiness(respawn_first=False)
            self._record("playback_readiness_evaluated", allowed=ok, reason=reason)
            command = template.format(index=index) if index is not None else ""
            self._record("generated_command", command=command)
            original_ui_command(template)

        self.ui._selected_step_playback_command = ui_command

        original_handle = self.controller.handle_command

        def handle_command(message: Any) -> Any:
            text = str(getattr(message, "text", message) or "")
            if text.startswith("play_step"):
                self._record("controller_command_received", command=text)
            return original_handle(message)

        self.controller.handle_command = handle_command

        original_selected_start = self.controller.start_selected_step_playback

        def selected_start(selected_index: int, *, profile: str) -> bool:
            self._record("profile_resolved", selected_index=int(selected_index), profile=str(profile))
            return original_selected_start(selected_index, profile=profile)

        self.controller.start_selected_step_playback = selected_start

        original_begin = self.controller.operation.begin

        def begin(state: OperationState, *, detail: str = "") -> bool:
            before = self.controller.operation.state.value
            result = original_begin(state, detail=detail)
            if OperationState(state) is OperationState.PLAYBACK:
                self._record(
                    "operation_acquired",
                    state_before=before,
                    state_after=self.controller.operation.state.value,
                    acquired=bool(result),
                    begin_call_count=sum(1 for row in self.trace if row["event"] == "operation_acquired") + 1,
                )
            return result

        self.controller.operation.begin = begin

        original_enter = self.controller.operation.enter_playback

        def enter_playback(*, detail: str = "") -> bool:
            before = self.controller.operation.state.value
            result = original_enter(detail=detail)
            self._record(
                "playback_manager_enter_operation",
                once=False,
                state_before=before,
                state_after=self.controller.operation.state.value,
                accepted=bool(result),
            )
            return result

        self.controller.operation.enter_playback = enter_playback

        original_finish = self.controller.operation.finish

        def finish(expected: OperationState | None = None) -> bool:
            before = self.controller.operation.state.value
            result = original_finish(expected)
            self._record(
                "operation_finished",
                once=False,
                expected=expected.value if isinstance(expected, OperationState) else None,
                state_before=before,
                state_after=self.controller.operation.state.value,
                accepted=bool(result),
            )
            return result

        self.controller.operation.finish = finish

        original_restore = self.controller.transport.restore_sim_state

        def restore_sim_state(sim_state: dict[str, Any], *, request_id: str = "") -> str:
            result = original_restore(sim_state, request_id=request_id)
            self._record("restore_request_sent", request_id=result)
            return result

        self.controller.transport.restore_sim_state = restore_sim_state

        original_request_state = self.controller.transport.request_state

        def request_state(*, detailed: bool = False) -> Any:
            result = original_request_state(detailed=detailed)
            if detailed:
                self._record("detailed_state_requested")
            return result

        self.controller.transport.request_state = request_state

        original_verify = self.controller._verify_selected_restore_state

        def verify(pending: dict[str, Any], observed: dict[str, Any]) -> tuple[bool, str]:
            result = original_verify(pending, observed)
            if result[0]:
                self.restore_observed_state = copy.deepcopy(observed)
                self._record("restored_state_verified", verification=result[1])
            return result

        self.controller._verify_selected_restore_state = verify

        original_worker_start = self.controller.playback.start_worker_plan

        def start_worker_plan(plan: PlaybackPlan, **kwargs: Any) -> bool:
            self._record("fast_plan_built", plan=self._plan_summary(plan))
            return original_worker_start(plan, **kwargs)

        self.controller.playback.start_worker_plan = start_worker_plan

        original_transport_start = self.controller.transport.start_playback_plan

        def transport_start(plan: PlaybackPlan, **kwargs: Any) -> Any:
            self._record(
                "worker_start_request_sent",
                plan_id=str(kwargs.get("plan_id", "")),
                request_id=str(kwargs.get("request_id", "")),
                plan_sha256=str(kwargs.get("plan_sha256", "")),
                event_count=len(plan.events),
                segment_count=len(plan.segments),
                profile=plan.profile,
            )
            return original_transport_start(plan, **kwargs)

        self.controller.transport.start_playback_plan = transport_start

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        rows = load_steps_jsonl(FORMAL_SOURCE)
        if len(rows) < 5:
            raise AssertionError("formal 50 mm sequence has fewer than five steps")
        self.controller.current_height_mm = 50
        self.controller.current_version_id = "formal_50mm_physical_click_read_only"
        self.controller.current_version_metadata = {
            "read_only": True,
            "accepted_steps_path": str(FORMAL_SOURCE.resolve()),
        }
        self.controller.manager.adopt_steps(rows, dirty=False)
        self.controller.selected_step_index = 5
        self.ui.open_right_tab("Playback")
        self.ui._refresh(force=True)
        self.ui.steps_tree.selection_set("step_5")
        self.ui.steps_tree.see("step_5")
        self.ui._refresh(force=True)
        step = self.controller.manager.get_step(5)
        raw = plan_from_steps([step], profile="raw", max_wheel_speed=self.controller.max_wheel_speed, sequence_total_steps=len(rows))
        fast = plan_from_steps([step], profile="fast", max_wheel_speed=self.controller.max_wheel_speed, sequence_total_steps=len(rows))
        self.result.update(
            worker_pid=int(self.controller.latest_sim_status.get("pid", 0) or 0),
            worker_session_id=str(self.controller.latest_sim_status.get("worker_session_id", "") or ""),
            loaded_step_count=len(rows),
            selected_step=5,
            previous_step=4,
            raw_fast_plan_comparison={"raw": self._plan_summary(raw), "fast": self._plan_summary(fast)},
        )
        self._install_boundary_wrappers()
        self._advance("MANUAL_START")

    def _stage_manual_start(self) -> None:
        self.controller.handle_command("servo front_left_hip -20")
        self.manual_started_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self._advance("MANUAL_WAIT")

    def _stage_manual_wait(self) -> None:
        state = dict(self.controller.latest_sim_status.get("command_state", {}) or {})
        servos = dict(state.get("servos", {}) or {})
        elapsed = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0) - self.manual_started_sim
        if abs(float(servos.get("front_left_hip", 0.0)) + 20.0) > 1.0e-6 or elapsed < 0.5:
            return
        self.result["manual_state_before_click"] = copy.deepcopy(state)
        self._capture_ui("button_before_click", f"run_{self.run_number + 1}_play_selected_fast_button.png")
        self._advance("PHYSICAL_CLICK")

    def _stage_physical_click(self) -> None:
        if self.physical_click_in_progress:
            return
        self.physical_click_in_progress = True
        self.run_number += 1
        self.click_id = uuid.uuid4().hex
        physical_driver_click_id = self.click_id
        self.click_started = time.monotonic()
        self.trace = []
        self.trace_names = set()
        self.restore_observed_state = {}
        self.actual_motion_state = {}
        self.expected_worker_plan_id = ""
        self.expected_worker_request_id = ""
        button = self.ui.playback_buttons_by_label["Play Selected Fast"]
        raw_button = self.ui.playback_buttons_by_label["Play Selected Step"]
        self.ui.root.deiconify()
        self.ui.root.lift()
        self.ui.root.attributes("-topmost", True)
        self.ui.root.focus_force()
        self.ui.root.update_idletasks()
        self.ui.root.update()
        x = button.winfo_rootx() + button.winfo_width() // 2
        y = button.winfo_rooty() + button.winfo_height() // 2
        target = self.ui.root.winfo_containing(x, y)
        geometry = {
            "widget": str(button),
            "raw_widget": str(raw_button),
            "button_state": list(button.state()),
            "raw_button_state": list(raw_button.state()),
            "same_guard_state": ("disabled" in button.state()) == ("disabled" in raw_button.state()),
            "button_rect": [button.winfo_rootx(), button.winfo_rooty(), button.winfo_width(), button.winfo_height()],
            "center": [x, y],
            "screen": [self.ui.root.winfo_screenwidth(), self.ui.root.winfo_screenheight()],
            "winfo_containing": str(target),
            "target_matches": target == button,
        }
        self.result.setdefault("physical_click_geometry", []).append(geometry)
        if "disabled" in button.state():
            raise AssertionError(f"Play Selected Fast disabled before physical click: {self.controller.can_playback()[1]}")
        if "disabled" in raw_button.state() or not geometry["same_guard_state"]:
            raise AssertionError(f"Raw/Fast guard mismatch: {geometry}")
        if target != button:
            raise AssertionError(f"button center does not hit visible Fast widget: {geometry}")
        user32 = ctypes.windll.user32
        user32.SetForegroundWindow(self.ui.root.winfo_id())
        user32.SetCursorPos(x, y)
        self._record("physical_mouse_press", click_id=self.click_id, x=x, y=y, widget=str(button))
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        self.physical_click_context = {
            "physical_driver_click_id": physical_driver_click_id,
            "button": str(button),
            "pressed_at": self.click_started,
        }
        self.ui.root.after(20, self._send_physical_release)
        self._advance("PHYSICAL_CALLBACK_WAIT")

    def _send_physical_release(self) -> None:
        product_click_id = str(self.ui.last_selected_fast_click_id or "")
        if not product_click_id:
            self.physical_click_context["release_error"] = "physical mouse press did not create selected_fast_click_id"
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
            return
        press_to_release_ms = (time.monotonic() - self.click_started) * 1000.0
        self.click_id = product_click_id
        for row in self.trace:
            if row.get("event") == "physical_mouse_press":
                row["click_id"] = product_click_id
                row["physical_driver_click_id"] = self.physical_click_context["physical_driver_click_id"]
        self.physical_click_context["press_to_release_ms"] = press_to_release_ms
        self.physical_click_context["released_at"] = time.monotonic()
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
        self._advance("PHYSICAL_CALLBACK_WAIT")

    def _stage_physical_callback_wait(self) -> None:
        if self.physical_click_context.get("release_error"):
            raise AssertionError(str(self.physical_click_context["release_error"]))
        if "released_at" not in self.physical_click_context:
            return
        if "tk_button_command_entered" not in self.trace_names:
            if time.monotonic() - float(self.physical_click_context["released_at"]) < 0.75:
                return
            raise AssertionError("physical click did not enter the Tk button callback")
        immediate_ms = (time.monotonic() - self.click_started) * 1000.0
        physical_feedback_render_ms = float(self.ui.last_selected_fast_feedback_ms)
        self.result.setdefault("immediate_feedback", []).append(
            {
                "click_id": self.click_id,
                "elapsed_ms": immediate_ms,
                "visible_feedback_ms": self.ui.last_selected_fast_feedback_ms,
                "physical_feedback_render_ms": physical_feedback_render_ms,
                "press_to_release_ms": self.physical_click_context["press_to_release_ms"],
                "visible_text": self.ui.playback_label_var.get(),
                "status": self.controller.status,
                "playback_state": self.controller.playback.progress.playback_state,
                "pending": self.controller.pending_selected_playback is not None,
            }
        )
        self._capture_ui("restoring_immediate", f"run_{self.run_number}_physical_click_restoring.png")
        if self.controller.pending_selected_playback is None:
            raise AssertionError(f"Fast callback entered but no restore transaction exists: {self.controller.status}")
        if self.ui.last_selected_fast_feedback_ms >= 100.0:
            raise AssertionError(f"visible Fast feedback took {self.ui.last_selected_fast_feedback_ms:.3f} ms")
        if physical_feedback_render_ms >= 100.0:
            raise AssertionError(f"physical Fast feedback render took {physical_feedback_render_ms:.3f} ms")
        if self.controller.pending_selected_playback.get("selected_fast_click_id") != self.click_id:
            raise AssertionError("Selected restore request did not preserve the physical click ID")
        self.physical_click_in_progress = False
        self._advance("WAIT_START")

    def _stage_wait_start(self) -> None:
        source = self.controller.pending_selected_playback or self.controller.last_selected_restore_result
        for row in list(source.get("trace", []) or []):
            mapping = {
                "restore_acknowledged": "restore_acknowledged",
                "state_requested": "detailed_state_requested",
                "state_verified": "restored_state_verified",
            }
            if row.get("event") in mapping:
                self._record_existing(
                    mapping[str(row["event"])],
                    float(row.get("monotonic_s", time.monotonic())),
                    request_id=row.get("request_id", ""),
                    detail=row.get("detail", ""),
                )
        result = self.controller.last_selected_restore_result
        if result.get("started") and result.get("profile") == "fast":
            self.expected_worker_plan_id = str(self.controller.playback.worker_plan_id or "")
            self.expected_worker_request_id = str(self.controller.playback.worker_request_id or "")
            if result.get("plan_source_steps") != [5]:
                raise AssertionError(f"Fast selected plan source mismatch: {result}")
            self._capture_ui("restore_complete", f"run_{self.run_number}_restore_complete.png")
            self._advance("WAIT_MOTION")

    def _stage_wait_motion(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if (
            str(worker.get("plan_id", "") or "") != self.expected_worker_plan_id
            or str(worker.get("request_id", "") or "") != self.expected_worker_request_id
        ):
            return
        if worker.get("started") and (worker.get("active") or worker.get("first_command_applied")):
            self._record(
                "worker_start_accepted",
                request_id=worker.get("request_id"),
                plan_id=worker.get("plan_id"),
                plan_sha256=worker.get("plan_sha256"),
                event_count=worker.get("event_count"),
                segment_count=worker.get("segment_count"),
                profile=worker.get("profile"),
                active=worker.get("active"),
            )
        if worker.get("first_command_applied"):
            self._record(
                "first_command_applied",
                sim_step=worker.get("first_command_applied_sim_step"),
                sim_time_s=worker.get("first_command_applied_sim_time_s"),
                events_sent=worker.get("events_sent"),
            )
            now = time.monotonic()
            if now - self.last_detail_request > 0.15:
                self.controller.transport.request_state(detailed=True)
                self.last_detail_request = now
        detailed = dict(getattr(self.controller.sim_client, "latest_detailed_status", {}) or {})
        observed_state = dict(detailed.get("sim_state", {}) or {})
        restore_wheels = self._actual_wheels(self.restore_observed_state)
        actual_wheels = self._actual_wheels(observed_state)
        maximum = max([abs(value) for value in actual_wheels.values()] + [0.0])
        restore_maximum = max([abs(value) for value in restore_wheels.values()] + [0.0])
        if worker.get("first_command_applied") and maximum > max(0.10, restore_maximum + 0.05):
            self.actual_motion_state = copy.deepcopy(observed_state)
            self._record(
                "actual_motion_observed",
                restore_measured_wheels=restore_wheels,
                motion_measured_wheels=actual_wheels,
                maximum_abs_rad_s=maximum,
            )
            self._capture_ui("first_command_motion", f"run_{self.run_number}_first_command_actual_motion.png")
        if worker.get("active"):
            return
        if str(worker.get("stop_reason", "") or "") != "complete":
            return
        if "first_command_applied" not in self.trace_names or "actual_motion_observed" not in self.trace_names:
            raise AssertionError(f"worker completed without proven first command/motion: {worker}")
        if self.controller.operation.state is not OperationState.IDLE:
            return
        if sum(1 for row in self.trace if row.get("event") == "operation_acquired") != 1:
            raise AssertionError("Selected Fast did not acquire PLAYBACK exactly once")
        if any(row.get("event") == "playback_manager_enter_operation" for row in self.trace):
            raise AssertionError("PlaybackManager attempted a second PLAYBACK operation entry")
        successful_finishes = [
            row
            for row in self.trace
            if row.get("event") == "operation_finished" and row.get("accepted")
        ]
        if len(successful_finishes) != 1:
            raise AssertionError(f"Selected Fast released PLAYBACK {len(successful_finishes)} times")
        if str(worker.get("profile", "")) != "motion_only":
            raise AssertionError(f"worker Fast profile was lost: {worker.get('profile')}")
        if str(dict(worker.get("progress_detail", {}) or {}).get("playback_profile", "")) != "motion_only":
            raise AssertionError("worker progress did not preserve motion_only profile")
        self.ui._refresh(force=True)
        button = self.ui.playback_buttons_by_label["Play Selected Fast"]
        if "disabled" in button.state():
            return
        self._record(
            "completed",
            stop_reason=worker.get("stop_reason"),
            events_sent=worker.get("events_sent"),
            operation=self.controller.operation.state.value,
            button_state=list(button.state()),
        )
        ordered_trace = sorted(self.trace, key=lambda row: float(row.get("monotonic_s", 0.0) or 0.0))
        self.result["runs"].append(
            {
                "click_id": self.click_id,
                "trace": copy.deepcopy(ordered_trace),
                "ui_click_trace": copy.deepcopy(self.ui.selected_fast_click_trace),
                "restore_result": copy.deepcopy(self.controller.last_selected_restore_result),
                "final_worker": copy.deepcopy(worker),
                "operation": self.controller.operation.state.value,
                "button_state": list(button.state()),
            }
        )
        self._capture_ui("completed", f"run_{self.run_number}_completed_button_enabled.png")
        if self.run_number < self.repeat_count:
            self._advance("MANUAL_START")
        else:
            self._advance("FINISH")

    def _stage_finish(self) -> None:
        self.controller.stop_wheels(reason="physical_fast_e2e_complete")
        self.result["formal_source_sha256_after"] = hashlib.sha256(FORMAL_SOURCE.read_bytes()).hexdigest()
        if self.result["formal_source_sha256_after"] != self.result["formal_source_sha256_before"]:
            raise AssertionError("formal saved sequence changed")
        self.result["success"] = True
        self._capture_isaac("isaac_final", "isaac_final.png")
        self._save_gif()
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(500, lambda: self.ui._window_close(force=True))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-count", type=int, default=1)
    args = parser.parse_args()
    return PhysicalFastClickE2E(Path(args.output), repeat_count=args.repeat_count).run()


if __name__ == "__main__":
    raise SystemExit(main())
