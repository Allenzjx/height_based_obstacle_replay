from __future__ import annotations

import copy
import tempfile
import time
import types
import unittest
from pathlib import Path

from operation_coordinator import OperationState
from playback import SimTimePlaybackService, plan_from_steps
from sim_ui_controller import RealRobotStyleHeightReplayUi
from tests.test_selected_step_previous_saved_state import (
    asynchronous_controller,
    controller_with_three_steps,
    make_args,
)


class FastButtonAndIndexTest(unittest.TestCase):
    def test_fast_widget_binding_guard_and_immediate_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.selected_step_index = 2
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                fast = ui.playback_buttons_by_label["Play Selected Fast"]
                raw = ui.playback_buttons_by_label["Play Selected Step"]
                ui._refresh_button_states()
                self.assertTrue(str(fast.cget("command")))
                self.assertTrue(ui.root.bind("<ButtonPress-1>"))
                self.assertFalse(fast.bind("<ButtonPress-1>"))
                self.assertEqual("disabled" in fast.state(), "disabled" in raw.state())
                self.assertIs(ui.resolve_playback_selected_index(), 2)
                event = types.SimpleNamespace(widget=fast, x_root=100, y_root=100)
                ui._play_selected_fast_mouse_press(event)
                self.assertTrue(ui.last_selected_fast_click_id)
                posted: list[str] = []
                ui._post = lambda text, **_kwargs: posted.append(text)  # type: ignore[method-assign]
                ui._selected_step_playback_command("play_step {index} fast")
                self.assertLess(ui.last_selected_fast_feedback_ms, 100.0)
                self.assertIn("Play Selected Fast received: Step 2", ui.playback_label_var.get())
                self.assertEqual(posted, ["play_step 2 fast"])
                self.assertEqual(ui.selected_fast_click_trace[0]["event"], "physical_mouse_press")
                self.assertIn("immediate_visible_feedback", [row["event"] for row in ui.selected_fast_click_trace])
            finally:
                ui._window_close(force=True)

    def test_tree_selection_precedes_controller_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.selected_step_index = 3
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                ui._refresh(force=True)
                ui.steps_tree.selection_set("step_2")
                self.assertEqual(ui.resolve_playback_selected_index(), 2)
                ui.steps_tree.selection_remove("step_2")
                self.assertEqual(ui.resolve_playback_selected_index(), 3)
                controller.selected_step_index = 99
                self.assertIsNone(ui.resolve_playback_selected_index())
            finally:
                ui._window_close(force=True)


class FastPlanAndProfileTest(unittest.TestCase):
    def test_raw_fast_actuator_signature_and_source_are_equal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            selected = controller.manager.get_step(3)
            raw = plan_from_steps([selected], profile="raw", sequence_total_steps=3)
            fast = plan_from_steps([selected], profile="fast", sequence_total_steps=3)
            self.assertGreater(len(raw.events), 0)
            self.assertEqual([event.command for event in raw.events], [event.command for event in fast.events])
            self.assertEqual(len(raw.events), len(fast.events))
            self.assertEqual(len(raw.segments), len(fast.segments))
            self.assertEqual({event.source_step for event in raw.events}, {3})
            self.assertEqual({event.source_step for event in fast.events}, {3})
            self.assertEqual(raw.profile, "raw")
            self.assertEqual(fast.profile, "motion_only")

    def test_worker_compact_status_preserves_fast_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            plan = plan_from_steps([controller.manager.get_step(3)], profile="fast", sequence_total_steps=3)
            plan.selected_playback = True
            service = SimTimePlaybackService()
            self.assertTrue(
                service.start_plan(
                    plan,
                    current_sim_time_s=1.0,
                    current_wall_time_s=2.0,
                    plan_id="plan-fast",
                    request_id="request-fast",
                    worker_session_id="worker-fast",
                )
            )
            status = service.status_dict(current_sim_time_s=1.0, current_wall_time_s=2.0, compact=True)
            self.assertEqual(status["profile"], "motion_only")
            self.assertEqual(status["progress_detail"]["playback_profile"], "motion_only")
            self.assertTrue(status["progress_detail"]["selected_playback"])


class SingleOperationOwnershipTest(unittest.TestCase):
    def _ack_restore(self, controller, fake) -> None:
        request_id = fake.restore_calls[0]["request_id"]
        fake.latest_status.update(
            restore_count=1,
            last_restore_result="ok",
            last_restore_error="",
            last_restore_request_id=request_id,
            last_restore_verification={"verified": True, "request_id": request_id},
        )
        fake.latest_detailed_status = {
            **fake.latest_status,
            "detail_status": True,
            "sim_state": copy.deepcopy(fake.restore_calls[0]["sim_state"]),
        }
        controller.update()
        controller.update()

    def test_selected_restore_enters_playback_once_and_worker_start_reuses_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            begin_calls = 0
            enter_calls = 0
            successful_finish_calls = 0
            original_begin = controller.operation.begin
            original_enter = controller.operation.enter_playback
            original_finish = controller.operation.finish

            def begin(state, *, detail=""):
                nonlocal begin_calls
                begin_calls += 1
                return original_begin(state, detail=detail)

            def enter(*, detail=""):
                nonlocal enter_calls
                enter_calls += 1
                return original_enter(detail=detail)

            def finish(expected=None):
                nonlocal successful_finish_calls
                result = original_finish(expected)
                if result:
                    successful_finish_calls += 1
                return result

            controller.operation.begin = begin
            controller.operation.enter_playback = enter
            controller.operation.finish = finish
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            owner = fake.restore_calls[0]["request_id"]
            self._ack_restore(controller, fake)
            self.assertEqual(begin_calls, 1)
            self.assertEqual(enter_calls, 0)
            self.assertEqual(len(fake.start_calls), 1)
            self.assertEqual(controller.playback.operation_owner_id, owner)
            self.assertIs(controller.operation.state, OperationState.PLAYBACK)
            controller.playback.stop(reason="test_complete", stop_wheels=False)
            self.assertEqual(successful_finish_calls, 1)
            self.assertIs(controller.operation.state, OperationState.IDLE)

    def test_existing_operation_owner_must_match_restore_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            plan = plan_from_steps([controller.manager.get_step(3)], profile="fast", sequence_total_steps=3)
            plan.timing["selected_restore_request_id"] = "owner-a"
            self.assertTrue(controller.operation.begin(OperationState.PLAYBACK))
            self.assertFalse(
                controller.playback.start_worker_plan(
                    plan,
                    operation_already_owned=True,
                    operation_owner_id="owner-b",
                )
            )
            self.assertIn("does not own", controller.playback.last_error)
            self.assertIs(controller.operation.state, OperationState.PLAYBACK)
            controller.operation.finish(OperationState.PLAYBACK)

    def test_selected_fast_click_id_and_profile_are_immutable_in_pending_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, _fake = asynchronous_controller(Path(tmp))
            controller.selected_fast_click_id = "click-123"
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            pending = controller.pending_selected_playback
            self.assertIsNotNone(pending)
            self.assertEqual(pending["profile"], "fast")
            self.assertEqual(pending["selected_fast_click_id"], "click-123")
            controller.playback.set_profile("raw")
            self.assertEqual(pending["profile"], "fast")
            controller.handle_command("stop_play")
            self.assertIsNone(controller.pending_selected_playback)
            self.assertIs(controller.operation.state, OperationState.IDLE)


class FailureCleanupTest(unittest.TestCase):
    @staticmethod
    def _ack_restore(controller, fake) -> None:
        request_id = fake.restore_calls[0]["request_id"]
        fake.latest_status.update(
            restore_count=1,
            last_restore_result="ok",
            last_restore_error="",
            last_restore_request_id=request_id,
            last_restore_verification={"verified": True, "request_id": request_id},
        )
        fake.latest_detailed_status = {
            **fake.latest_status,
            "detail_status": True,
            "sim_state": copy.deepcopy(fake.restore_calls[0]["sim_state"]),
        }
        controller.update()
        controller.update()

    @staticmethod
    def _accepted_ack(controller) -> dict[str, object]:
        plan = controller.playback.plan
        assert plan is not None
        return {
            "operation": "start_playback_plan",
            "request_id": controller.playback.worker_request_id,
            "accepted": True,
            "plan_id": controller.playback.worker_plan_id,
            "plan_sha256": plan.plan_sha256,
            "event_count": len(plan.events),
            "segment_count": len(plan.segments),
            "worker_session_id": "worker-test",
        }

    def test_no_selection_has_visible_feedback_without_starting_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = controller_with_three_steps(Path(tmp))
            controller.selected_step_index = None
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                ui._refresh(force=True)
                for item in ui.steps_tree.selection():
                    ui.steps_tree.selection_remove(item)
                fast = ui.playback_buttons_by_label["Play Selected Fast"]
                ui._play_selected_fast_mouse_press(types.SimpleNamespace(widget=fast, x_root=10, y_root=10))
                self.assertEqual(ui.playback_label_var.get(), "Select an accepted step first.")
                self.assertLess(ui.last_selected_fast_feedback_ms, 100.0)
                self.assertIsNone(controller.pending_selected_playback)
                self.assertIs(controller.operation.state, OperationState.IDLE)
            finally:
                ui._window_close(force=True)

    def test_restore_timeout_cleans_pending_and_reenables_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            assert controller.pending_selected_playback is not None
            controller.pending_selected_playback["deadline"] = time.monotonic() - 1.0
            controller.update()
            self.assertIsNone(controller.pending_selected_playback)
            self.assertFalse(controller.playback.active)
            self.assertFalse(controller.playback.start_requested)
            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertIn("timed out", controller.playback.last_error)
            self.assertTrue(controller.can_playback()[0])
            self.assertEqual(fake.start_calls, [])

    def test_worker_rejection_releases_selected_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            self._ack_restore(controller, fake)
            controller.playback.sync_worker_status(
                {},
                operation_ack={
                    "operation": "start_playback_plan",
                    "request_id": controller.playback.worker_request_id,
                    "accepted": False,
                    "rejection_reason": "synthetic worker rejection",
                },
            )
            self.assertFalse(controller.playback.active)
            self.assertFalse(controller.playback.start_requested)
            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertIn("synthetic worker rejection", controller.playback.last_error)
            self.assertTrue(controller.can_playback()[0])

    def test_first_command_watchdog_releases_selected_operation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, fake = asynchronous_controller(Path(tmp))
            self.assertTrue(controller.start_selected_step_playback(3, profile="fast"))
            self._ack_restore(controller, fake)
            controller.playback.sync_worker_status({}, operation_ack=self._accepted_ack(controller))
            controller.playback.sync_worker_status(
                {
                    "active": True,
                    "started": True,
                    "plan_id": controller.playback.worker_plan_id,
                    "request_id": controller.playback.worker_request_id,
                    "worker_session_id": "worker-test",
                    "first_motion_planned_s": 0.0,
                    "first_command_applied": False,
                    "sim_elapsed_s": controller.playback.first_command_watchdog_s + 0.1,
                }
            )
            self.assertFalse(controller.playback.active)
            self.assertFalse(controller.playback.start_requested)
            self.assertEqual(controller.playback.stop_reason, "first_command_watchdog")
            self.assertIs(controller.operation.state, OperationState.IDLE)
            self.assertIn("First motion command", controller.playback.last_error)
            self.assertTrue(controller.can_playback()[0])


if __name__ == "__main__":
    unittest.main()
