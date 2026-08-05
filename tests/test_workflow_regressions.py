from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sequence_model import apply_events_to_state, empty_command_state, load_steps_jsonl, make_event, make_step  # noqa: E402
from sim_ui_controller import HeightReplayController, MODE_PENDING_REPLACEMENT, MODE_TEST, RealRobotStyleHeightReplayUi  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        sim_launch_mode="disabled",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        record_event_min_interval_ms=50.0,
        record_event_max_hz=20.0,
        record_coalesce_slider_events=True,
        record_max_events_per_step=2000,
        max_text_widget_chars=2000,
        disable_auto_sim_state_json=False,
        sim_state_json_on_demand=True,
        playback_pre_step_settle_s=0.30,
        respawn_play_settle_s=0.30,
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
    )


def make_named_step(index: int, name: str, command: str) -> dict[str, object]:
    before = empty_command_state()
    events = [make_event(0.0, command, kind="test")]
    after = apply_events_to_state(before, events)
    return make_step(
        index=index,
        step_type="recorded",
        duration=0.1,
        events=events,
        command_state_before=before,
        command_state_after=after,
        name=name,
        note="height=10cm",
        extra={"height_cm": 10},
    )


class WorkflowRegressionTest(unittest.TestCase):
    def test_save_creates_immutable_versions_and_keeps_old_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "old_one", "servo front_left_hip 1"))
            controller.manager.add_step(make_named_step(2, "old_two", "servo front_right_hip 2"))
            old_path = controller.save_steps_for_current_height()
            self.assertEqual(len(load_steps_jsonl(old_path)), 2)

            controller.manager.clear()
            controller.manager.add_step(make_named_step(1, "new_one", "servo rear_left_hip 3"))
            new_path = controller.save_steps_for_current_height()
            loaded = load_steps_jsonl(new_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["name"], "new_one")
            text = Path(new_path).read_text(encoding="utf-8")
            self.assertNotIn("old_two", text)
            self.assertNotEqual(old_path, new_path)
            self.assertEqual(len(load_steps_jsonl(old_path)), 2)
            self.assertEqual(controller.store.status_rows()[-1]["version_count"], 2)

    def test_save_empty_sequence_creates_new_version_without_clearing_old(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "old_one", "servo front_left_hip 1"))
            old_path = controller.save_steps_for_current_height()
            controller.manager.clear()
            path = controller.save_steps_for_current_height(allow_empty=True)
            self.assertEqual(load_steps_jsonl(path), [])
            self.assertEqual(len(load_steps_jsonl(old_path)), 1)
            self.assertEqual(controller.store.status_rows()[-1]["version_count"], 2)

    def test_replace_save_file_contains_only_replacement_and_clears_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "old_step", "servo front_left_hip 1"))
            controller.pending_replacement = make_named_step(1, "replacement_step", "servo front_left_hip 9")
            controller.replace_target_index = 1
            controller.mode = MODE_PENDING_REPLACEMENT
            controller.accept_replacement()
            self.assertEqual(controller.manager.steps[0]["name"], "replacement_step")
            self.assertIsNone(controller.pending_replacement)
            self.assertIsNone(controller.replace_target_index)
            path = controller.save_steps_for_current_height()
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("replacement_step", text)
            self.assertNotIn("old_step", text)

    def test_replace_twice_only_latest_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "old_step", "servo front_left_hip 1"))
            for name, value in [("replacement_one", 5), ("replacement_two", 7)]:
                controller.pending_replacement = make_named_step(1, name, f"servo front_left_hip {value}")
                controller.replace_target_index = 1
                controller.mode = MODE_PENDING_REPLACEMENT
                controller.accept_replacement()
            self.assertEqual(controller.manager.steps[0]["name"], "replacement_two")

    def test_play_selected_applies_command_state_before_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            first = controller.manager.add_step(make_named_step(1, "first", "servo front_left_hip 12"))
            second = controller.manager.add_step(make_named_step(2, "second", "servo front_right_hip 20"))
            controller.start_playback([second], label="step 2", restore_start_state=True)
            state = controller.transport.capture_command_state()
            self.assertEqual(state["servos"]["front_left_hip"], first["command_state_after"]["servos"]["front_left_hip"])
            self.assertIn("command_state_before", controller.detail_text)

    def test_play_selected_restores_sim_state_before_if_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            step = make_named_step(1, "with_sim_state", "servo front_left_hip 5")
            sim_before = empty_command_state()
            sim_before["servos"]["rear_right_hip"] = 33.0
            step["sim_state_before"] = {"command_state": sim_before, "root_pose": None}
            controller.manager.add_step(step)
            controller.start_playback([controller.manager.get_step(1)], label="step 1", restore_start_state=True)
            self.assertEqual(controller.transport.capture_command_state()["servos"]["rear_right_hip"], 33.0)
            self.assertIn("sim_state_before", controller.detail_text)

    def test_respawn_and_play_calls_respawn_before_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "one", "servo front_left_hip 5"))
            calls: list[str] = []
            original_respawn = controller.transport.respawn

            def respawn_spy() -> None:
                calls.append("respawn")
                original_respawn()

            controller.transport.respawn = respawn_spy  # type: ignore[method-assign]
            controller.start_playback(controller.manager.steps, label="all", respawn_first=True)
            self.assertEqual(calls, ["respawn"])
            self.assertTrue(controller.playback.active)

    def test_servo_wheel_commands_drive_staging_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo front_left_hip 10")
            controller.handle_command("servo_wheel mode")
            controller.stage_servo_wheel_servo("front_left_hip", 20.0)
            self.assertEqual(controller.transport.capture_command_state()["servos"]["front_left_hip"], 10.0)
            controller.handle_command("servo_wheel launch")
            self.assertEqual(controller.mode, MODE_TEST)
            self.assertEqual(controller.transport.capture_command_state()["servos"]["front_left_hip"], 20.0)
            controller.handle_command("servo_wheel cancel")
            self.assertEqual(controller.mode, MODE_TEST)

    def test_start_record_allowed_after_entering_servo_wheel_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo_wheel mode")
            controller.handle_command("step_record start")
            self.assertTrue(controller.recording_active)

    def test_combine_explicit_selection_and_contiguous_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            for index in range(1, 5):
                controller.manager.add_step(make_named_step(index, f"step_{index}", f"servo front_left_hip {index}"))
            controller.add_to_combine_selection([1, 3])
            self.assertFalse(controller.can_combine()[0])
            self.assertIn("contiguous", controller.can_combine()[1].lower())
            controller.select_contiguous_combine_range()
            self.assertEqual(sorted(controller.combine_selected_indices), [1, 2, 3])
            self.assertTrue(controller.can_combine()[0])
            controller.toggle_combine_selection([2])
            self.assertEqual(sorted(controller.combine_selected_indices), [1, 3])
            controller.remove_from_combine_selection([1])
            self.assertEqual(sorted(controller.combine_selected_indices), [3])

    def test_combine_commit_removes_old_steps_and_reindexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(make_named_step(1, "one", "servo front_left_hip 1"))
            controller.manager.add_step(make_named_step(2, "two", "servo front_right_hip 2"))
            controller.manager.add_step(make_named_step(3, "three", "servo rear_left_hip 3"))
            controller.add_to_combine_selection([1, 2])
            controller.commit_combine_steps()
            self.assertEqual(controller.manager.count, 2)
            self.assertEqual([step["index"] for step in controller.manager.steps], [1, 2])
            names = [step["name"] for step in controller.manager.steps]
            self.assertNotIn("one", names)
            self.assertNotIn("two", names)
            self.assertEqual(controller.selected_step_index, 1)

    def test_refresh_steps_tree_preserves_combine_multi_selection_source(self) -> None:
        source = inspect.getsource(RealRobotStyleHeightReplayUi._refresh_steps_tree)
        self.assertIn("combine_mode_enabled", source)
        self.assertIn("selection_set(items)", source)


if __name__ == "__main__":
    unittest.main()
