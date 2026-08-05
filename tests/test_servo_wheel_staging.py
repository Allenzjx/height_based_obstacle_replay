from __future__ import annotations

import argparse
import inspect
import tempfile
import unittest
from pathlib import Path

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from sim_robot_adapter import NullSimRobotAdapter
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_mm=50,
        sim_launch_mode="disabled",
        max_wheel_speed_rad_s=2.0943951023931953,
        default_wheel_speed_rad_s=0.5235987755982988,
        ui_refresh_ms=100,
        sim_status_refresh_ms=125,
        full_refresh_ms=1000,
        record_event_min_interval_ms=50.0,
        record_event_max_hz=20.0,
        record_coalesce_slider_events=True,
        record_max_events_per_step=2000,
        max_text_widget_chars=2000,
        sim_state_json_on_demand=True,
        playback_pre_step_settle_s=0.30,
        respawn_play_settle_s=0.30,
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
    )


class ServoWheelAtomicBatchTest(unittest.TestCase):
    def test_one_transport_call_carries_all_twelve_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            calls: list[dict] = []
            original = controller.transport.apply_motion_batch

            def spy(payload: dict) -> str:
                calls.append(dict(payload))
                return original(payload)

            controller.transport.apply_motion_batch = spy  # type: ignore[method-assign]
            servos = {name: (-60.0 if "knee" in name else 12.0) for name in SERVO_JOINT_NAMES}
            wheels = {name: 0.3 for name in WHEEL_JOINT_NAMES}
            batch_id = controller.apply_servo_wheel_together(servos, wheels)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["batch_id"], batch_id)
            self.assertEqual(set(calls[0]["servo_targets_deg"]), set(SERVO_JOINT_NAMES))
            self.assertEqual(set(calls[0]["wheel_targets_rad_s"]), set(WHEEL_JOINT_NAMES))
            self.assertEqual(calls[0]["requested_sim_boundary"], "next_physics_tick")

    def test_adapter_ack_declares_one_tick_and_zero_channel_skew(self) -> None:
        adapter = NullSimRobotAdapter()
        ack = adapter.apply_motion_batch(
            {
                "batch_id": "atomic-1",
                "servo_targets_deg": {"front_left_knee": -60.0},
                "wheel_targets_rad_s": {name: 0.3 for name in WHEEL_JOINT_NAMES},
            }
        )
        self.assertTrue(ack["servo_applied"])
        self.assertTrue(ack["wheel_applied"])
        self.assertEqual(ack["servo_motion_start_sim_time"], ack["wheel_motion_start_sim_time"])
        self.assertEqual(ack["motion_start_skew_s"], 0.0)
        self.assertEqual(ack["first_physics_step"], 1)

    def test_recording_stores_one_canonical_batch_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.handle_command("step_record start")
            self.assertTrue(controller.start_servo_wheel_mode())
            controller.stage_servo_wheel_servo("front_left_hip", -20.0)
            for name in WHEEL_JOINT_NAMES:
                controller.stage_servo_wheel_wheel(name, 0.25)
            controller.launch_servo_wheel()
            self.assertEqual(len(controller.record_events), 1)
            event = controller.record_events[0]
            self.assertEqual(event["command"], "servo_wheel launch")
            self.assertEqual(event["kind"], "servo_wheel_launch")
            self.assertNotIn("recorded_speed_percent", event)
            self.assertEqual(event["canonical_servo_target_deg"]["front_left_hip"], -20.0)
            self.assertEqual(event["canonical_wheel_velocity_rad_s"]["front_left_ankle"], 0.25)
            self.assertEqual(len(event["expanded_commands"]), 12)
            self.assertIn("staged_state_before", event)
            self.assertIn("staged_state_after", event)

    def test_stop_wheels_preempts_and_zeros_all_wheels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.apply_servo_wheel_together(
                {name: 0.0 for name in SERVO_JOINT_NAMES},
                {name: 0.4 for name in WHEEL_JOINT_NAMES},
            )
            controller.stop_wheels(reason="test")
            self.assertTrue(all(value == 0.0 for value in controller.transport.capture_command_state()["wheels"].values()))

    def test_runtime_source_has_visible_staging_and_no_looped_ipc(self) -> None:
        source = inspect.getsource(HeightReplayController)
        self.assertIn("servo_wheel_staging_active", source)
        self.assertIn("servo_wheel_staged_state", source)
        apply_source = inspect.getsource(HeightReplayController.apply_servo_wheel_together)
        self.assertEqual(apply_source.count("self.transport.apply_motion_batch"), 1)
        self.assertNotIn("self.transport.send(", apply_source)
        ui_source = inspect.getsource(RealRobotStyleHeightReplayUi._build_record_servo_wheel_tab)
        for label in ("Start Servo-Wheel Mode", "Launch Servo-Wheel", "Clear Staged", "Cancel Servo-Wheel Mode"):
            self.assertIn(label, ui_source)

    def test_staging_changes_no_live_state_until_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            before = controller.transport.capture_command_state()
            self.assertTrue(controller.start_servo_wheel_mode())
            controller.stage_servo_wheel_servo("front_left_hip", 15.0)
            controller.stage_servo_wheel_wheel("front_left_ankle", 0.4)
            self.assertEqual(controller.transport.capture_command_state(), before)
            controller.launch_servo_wheel()
            after = controller.transport.capture_command_state()
            self.assertEqual(after["servos"]["front_left_hip"], 15.0)
            self.assertEqual(after["wheels"]["front_left_ankle"], 0.4)


if __name__ == "__main__":
    unittest.main()
