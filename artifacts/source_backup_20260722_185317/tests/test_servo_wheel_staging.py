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

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES, WHEEL_NAME_TO_SHORT  # noqa: E402
from height_sequence_store import HeightSequenceStore  # noqa: E402
from sequence_model import event_playback_commands  # noqa: E402
from sim_ui_controller import HeightReplayController, MODE_SERVO_WHEEL, RealRobotStyleHeightReplayUi  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        sim_launch_mode="disabled",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        global_motion_speed_scale=1.0,
        wheel_speed_scale=1.0,
        servo_command_scale=1.0,
        playback_speed_scale=1.0,
        preserve_wheel_distance=True,
        apply_speed_scale_to_manual=True,
        apply_speed_scale_to_playback=True,
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


class ServoWheelStagingTest(unittest.TestCase):
    def test_entering_mode_stages_without_sending_slider_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            sent: list[str] = []
            original_send = controller.transport.send

            def send_spy(command: str, *, source: str = "ui") -> None:
                sent.append(command)
                original_send(command, source=source)

            controller.transport.send = send_spy  # type: ignore[method-assign]
            controller.handle_command("servo_wheel mode")
            controller.handle_command("servo front_left_hip 15")
            controller.handle_command("wheel fl 0.5")
            self.assertEqual(controller.mode, MODE_SERVO_WHEEL)
            self.assertTrue(controller.servo_wheel_staging_active)
            self.assertEqual(sent, [])
            self.assertEqual(controller.transport.capture_command_state()["servos"]["front_left_hip"], 0.0)
            self.assertEqual(controller.servo_wheel_staged_state["servos"]["front_left_hip"], 15.0)
            self.assertEqual(controller.servo_wheel_staged_state["wheels"]["front_left_ankle"], 0.5)

    def test_launch_sends_servo_commands_before_wheel_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            sent: list[str] = []
            original_send = controller.transport.send

            def send_spy(command: str, *, source: str = "ui") -> None:
                sent.append(command)
                original_send(command, source=source)

            controller.transport.send = send_spy  # type: ignore[method-assign]
            controller.handle_command("servo_wheel mode")
            controller.handle_command("servo front_left_hip -12")
            controller.handle_command("wheel fl -0.75")
            controller.handle_command("servo_wheel launch")
            self.assertEqual(len(sent), len(SERVO_JOINT_NAMES) + len(WHEEL_JOINT_NAMES))
            self.assertTrue(all(command.startswith("servo ") for command in sent[: len(SERVO_JOINT_NAMES)]))
            self.assertTrue(all(command.startswith("wheel ") for command in sent[len(SERVO_JOINT_NAMES) :]))
            self.assertIn("servo front_left_hip -12", sent)
            self.assertIn("wheel fl -0.75", sent)
            self.assertFalse(controller.servo_wheel_staged_dirty)

    def test_clear_staged_resets_to_live_state_without_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo front_left_hip 22")
            sent: list[str] = []
            original_send = controller.transport.send

            def send_spy(command: str, *, source: str = "ui") -> None:
                sent.append(command)
                original_send(command, source=source)

            controller.transport.send = send_spy  # type: ignore[method-assign]
            controller.handle_command("servo_wheel mode")
            controller.handle_command("servo front_left_hip 5")
            controller.handle_command("servo_wheel clear_staged")
            self.assertNotIn("home", sent)
            self.assertEqual(controller.servo_wheel_staged_state["servos"]["front_left_hip"], 22.0)

    def test_recording_only_records_launch_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo_wheel mode")
            controller.handle_command("step_record start")
            controller.handle_command("servo front_left_hip 10")
            controller.handle_command("wheel fl 0.2")
            self.assertEqual(controller.record_events, [])
            controller.handle_command("servo_wheel launch")
            self.assertEqual(len(controller.record_events), 1)
            event = controller.record_events[0]
            self.assertEqual(event["command"], "servo_wheel launch")
            self.assertGreater(len(event["expanded_commands"]), 0)
            self.assertEqual(event_playback_commands(event), event["expanded_commands"])
            controller.handle_command("step_record stop")
            self.assertEqual(len(controller.pending_step["events"]), 1)

    def test_stop_wheels_bypasses_staging_and_zeros_staged_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo_wheel mode")
            controller.handle_command("w")
            self.assertTrue(any(abs(value) > 0 for value in controller.servo_wheel_staged_state["wheels"].values()))
            controller.handle_command("wheel stop")
            self.assertTrue(all(abs(value) == 0.0 for value in controller.transport.capture_command_state()["wheels"].values()))
            self.assertTrue(all(abs(value) == 0.0 for value in controller.servo_wheel_staged_state["wheels"].values()))

    def test_state_machine_blocks_playback_save_switch_when_staged_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo_wheel mode")
            controller.handle_command("servo front_left_hip 10")
            self.assertFalse(controller.can_playback()[0])
            self.assertFalse(controller.can_save()[0])
            self.assertFalse(controller.can_switch_height(discard_dirty=True)[0])
            controller.handle_command("e_stop")
            self.assertEqual(controller.mode, "E_STOP")

    def test_preview_and_refresh_sources_show_staged_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("servo_wheel mode")
            controller.handle_command("servo front_left_hip 10")
            text = controller.servo_wheel_preview_text()
            self.assertIn("live command_state", text)
            self.assertIn("staged command_state", text)
            self.assertIn("launch commands", text)
            source = inspect.getsource(RealRobotStyleHeightReplayUi._refresh)
            self.assertIn("staged_active", source)
            self.assertIn("staged_state", source)

    def test_manifest_path_collision_recovers_and_status_rows_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            collision = root / "height_20cm"
            collision.write_text("not a directory", encoding="utf-8")
            store = HeightSequenceStore(root)
            self.assertTrue(collision.is_dir())
            renamed = list(root.glob("height_20cm.invalid_file_*"))
            self.assertEqual(len(renamed), 1)
            rows = store.status_rows()
            self.assertTrue(any(row["height_cm"] == 20 for row in rows))
            controller = HeightReplayController(make_args(root))
            snapshot = controller.snapshot()
            self.assertTrue(any(row["height_cm"] == 20 for row in snapshot["height"]["manifest_rows"]))


if __name__ == "__main__":
    unittest.main()
