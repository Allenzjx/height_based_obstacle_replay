from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from operation_coordinator import OperationCoordinator, OperationState
from playback_availability import evaluate_playback_availability
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi
from tests.controller_test_utils import make_args, motion_step


def availability(**overrides):
    values = {
        "sim_connected": True,
        "sim_ready": True,
        "sequence_valid": True,
        "sequence_count": 269,
        "selected_step_valid": True,
        "operation_state": OperationState.IDLE,
        "playback_active": False,
        "playback_paused": False,
        "playback_scheduled": False,
    }
    values.update(overrides)
    return evaluate_playback_availability(**values)


class PlaybackAvailabilityMatrixTest(unittest.TestCase):
    def test_case_a_ready_nonempty_idle_enables_play(self) -> None:
        state = availability()
        self.assertTrue(state.can_start)
        self.assertTrue(state.can_play_selected)

    def test_case_b_empty_sequence_disables_play_with_reason(self) -> None:
        state = availability(sequence_valid=False, sequence_count=0)
        self.assertFalse(state.can_start)
        self.assertEqual(state.reason, "No valid sequence is loaded.")

    def test_case_c_recording_disables_play_with_reason(self) -> None:
        state = availability(operation_state=OperationState.RECORDING)
        self.assertFalse(state.can_start)
        self.assertEqual(state.reason, "Recording is active.")

    def test_case_d_active_enables_pause_and_stop_only(self) -> None:
        state = availability(operation_state=OperationState.PLAYBACK, playback_active=True)
        self.assertFalse(state.can_start)
        self.assertTrue(state.can_pause)
        self.assertFalse(state.can_resume)
        self.assertTrue(state.can_stop)

    def test_case_e_paused_enables_resume_and_stop(self) -> None:
        state = availability(
            operation_state=OperationState.PLAYBACK,
            playback_active=True,
            playback_paused=True,
        )
        self.assertFalse(state.can_pause)
        self.assertTrue(state.can_resume)
        self.assertTrue(state.can_stop)

    def test_analysis_and_export_need_sequence_but_not_connection(self) -> None:
        state = availability(sim_connected=False, sim_ready=False)
        self.assertTrue(state.can_analyze)
        self.assertTrue(state.can_export)
        self.assertFalse(state.can_start)


class OperationConflictTest(unittest.TestCase):
    def test_operations_are_mutually_exclusive_and_recover(self) -> None:
        coordinator = OperationCoordinator()
        self.assertTrue(coordinator.begin(OperationState.RECORDING))
        self.assertFalse(coordinator.begin(OperationState.PLAYBACK))
        self.assertEqual(coordinator.reason, "Recording is active.")
        self.assertTrue(coordinator.finish(OperationState.RECORDING))
        self.assertTrue(coordinator.begin(OperationState.SCENE_UPDATE))
        self.assertFalse(coordinator.enter_playback())
        self.assertTrue(coordinator.finish(OperationState.SCENE_UPDATE))
        self.assertTrue(coordinator.idle)

    def test_case_f_stop_clears_queue_and_reenables_play(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=False)
            self.assertTrue(controller.start_playback(controller.manager.steps, label="case-f"))
            self.assertTrue(controller.playback.active)
            self.assertIsNotNone(controller.playback.plan)
            controller.playback.stop(reason="unit-stop")
            status = controller.playback.status_dict()
            self.assertFalse(status["active"])
            self.assertFalse(status["scheduled"])
            self.assertEqual(status["count"], 0)
            self.assertIsNone(controller.playback.plan)
            self.assertTrue(controller.playback_availability().can_start)


class HeightSequenceSharingTest(unittest.TestCase):
    def test_case_g_load_5cm_updates_permanent_shared_sequence_without_playing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = HeightReplayController(make_args(root))
            manager_id = id(controller.manager)
            controller.store.save_steps(5, [motion_step(height_cm=5)])

            controller.set_current_height(5, discard_dirty=True, load_steps=True, generate_obstacle=False)

            self.assertEqual(id(controller.manager), manager_id)
            self.assertEqual(controller.manager.count, 1)
            self.assertEqual(controller.manager.accepted_path, controller.store.steps_path(5))
            self.assertFalse(controller.playback.active)
            self.assertTrue(controller.playback_availability().can_start)

    def test_case_h_missing_then_valid_load_recovers_without_busy_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller = HeightReplayController(make_args(root))
            manager_id = id(controller.manager)

            controller.set_current_height(5, discard_dirty=True, load_steps=True, generate_obstacle=False)
            self.assertEqual(controller.manager.count, 0)
            self.assertIn("No version exists for 50 mm", controller.status)
            self.assertTrue(controller.operation.idle)
            self.assertFalse(controller.playback_availability().can_start)

            controller.store.save_steps(5, [motion_step(height_cm=5)])
            self.assertEqual(controller.load_steps_for_current_height(), 1)
            self.assertEqual(id(controller.manager), manager_id)
            self.assertTrue(controller.operation.idle)
            self.assertTrue(controller.playback_availability().can_start)


class UiStructureTest(unittest.TestCase):
    def test_ui_has_only_retained_tabs_and_one_poll_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            ui = RealRobotStyleHeightReplayUi(controller)
            ui.root.withdraw()
            try:
                tabs = [str(ui.right_notebook.tab(tab_id, "text")) for tab_id in ui.right_notebook.tabs()]
                self.assertEqual(
                    tabs,
                    [
                        "Sim Connection",
                        "Run Manager",
                        "Record / Servo+Wheel",
                        "Playback",
                        "Height Generate",
                        "Combine",
                        "Sim State",
                    ],
                )
                self.assertIsNotNone(ui._poll_after_id)
                self.assertEqual(len(ui.playback_buttons_by_label), 16)
                controller.manager.adopt_steps([motion_step(height_cm=10)], dirty=False)
                controller.selected_step_index = 1
                ui._refresh_button_states()
                self.assertNotIn("disabled", ui.playback_buttons_by_label["Play All"].state())
                self.assertNotIn("disabled", ui.playback_buttons_by_label["Play All Fast"].state())
                self.assertNotIn("disabled", ui.playback_buttons_by_label["Play Selected Step"].state())
            finally:
                ui._window_close()
            self.assertIsNone(ui._poll_after_id)
            self.assertEqual(ui.slider_after_ids, {})


if __name__ == "__main__":
    unittest.main()
