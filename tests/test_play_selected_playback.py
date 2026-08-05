from __future__ import annotations

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from playback import PlaybackEvent, PlaybackManager, PlaybackPlan  # noqa: E402
from sequence_model import empty_command_state, make_event, make_step  # noqa: E402
from sim_ipc_protocol import decode_line, encode_message, make_message  # noqa: E402
from sim_process_client import SimProcessClient  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


def make_args(store_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        no_sim=True,
        store_root=store_root,
        height_cm=10,
        robot_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd",
        save_usd="C:/robotics_sim/wlr_robot/usd/wlr_robot_height_replay_env.usd",
        spawn_z=0.04,
        obstacle_x=1.55,
        obstacle_width=None,
        obstacle_length=None,
        infer_obstacle_size=True,
        robot_width=0.80,
        robot_length=0.55,
        physics_dt=1.0 / 120.0,
        render_interval=2,
        device="cuda:0",
        max_wheel_speed_rad_s=3.0,
        default_wheel_speed_rad_s=1.0,
        apply_physx_joint_limits=False,
        ui_refresh_ms=100,
        sim_status_refresh_ms=250,
        full_refresh_ms=1000,
        no_continuous_sim_step=False,
        wheel_direction=1.0,
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        save_scene=False,
        sim_launch_mode="disabled",
        restore_step_start_state_before_selected_playback=True,
        restore_full_sim_pose_if_available=True,
        fallback_to_command_state_before=True,
        playback_pre_step_settle_s=0.30,
        respawn_play_settle_s=0.30,
    )


def motion_step(index: int = 1) -> dict[str, Any]:
    state = empty_command_state()
    return make_step(
        index=index,
        step_type="test",
        duration=0.2,
        events=[make_event(0.0, "servo front_left_hip 10")],
        command_state_before=state,
        command_state_after=state,
        name=f"step_{index:03d}",
    )


class FakePlaybackController:
    max_wheel_speed = 3.0

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.stop_wheels_count = 0

    def handle_command(self, message: Any) -> None:
        self.commands.append(str(message.text))

    def stop_wheels(self) -> None:
        self.stop_wheels_count += 1


class PlaySelectedPlaybackTest(unittest.TestCase):
    def test_restore_sim_state_is_valid_ipc_message(self) -> None:
        message = make_message("restore_sim_state", sim_state={"command_state": {"servos": {}, "wheels": {}}})
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_process_client_queues_restore_sim_state_before_connect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = SimProcessClient(make_args(Path(tmp)))
            client.restore_sim_state({"command_state": {"servos": {}, "wheels": {}}})
            self.assertEqual(client.pending_messages[-1]["type"], "restore_sim_state")

    def test_playback_manager_scheduled_delay_is_visible_and_cancellable(self) -> None:
        fake = FakePlaybackController()
        manager = PlaybackManager(fake)
        plan = PlaybackPlan(path=None, events=[PlaybackEvent(0.0, "wheel all 1.0")], final_time_s=0.0, label="unit")
        self.assertTrue(manager.start_plan(plan, start_delay_s=0.20))
        status = manager.status_dict()
        self.assertTrue(status["active"])
        self.assertTrue(status["scheduled"])
        self.assertGreater(status["starts_in_s"], 0.0)
        manager.update()
        self.assertEqual(fake.commands, [])
        manager.stop()
        status = manager.status_dict()
        self.assertFalse(status["active"])
        self.assertFalse(status["scheduled"])
        self.assertEqual(status["stop_reason"], "stopped")

    def test_playback_manager_sends_events_after_schedule(self) -> None:
        fake = FakePlaybackController()
        manager = PlaybackManager(fake)
        plan = PlaybackPlan(path=None, events=[PlaybackEvent(0.0, "wheel all 1.0")], final_time_s=0.0, label="unit")
        self.assertTrue(manager.start_plan(plan, start_delay_s=0.01))
        time.sleep(0.03)
        manager.update()
        self.assertEqual(fake.commands, ["wheel all 1.0"])
        self.assertEqual(manager.events_sent, 1)
        self.assertEqual(manager.last_event_command, "wheel all 1.0")
        self.assertEqual(manager.stop_reason, "complete")

    def test_empty_selected_plan_warns_instead_of_silent_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            state = empty_command_state()
            empty_step = make_step(
                index=1,
                step_type="test",
                duration=0.0,
                events=[],
                command_state_before=state,
                command_state_after=state,
                name="empty",
            )
            ok = controller.start_playback([empty_step], label="10cm step 001", restore_start_state=True)
            self.assertFalse(ok)
            self.assertFalse(controller.playback.active)
            self.assertIn("no motion events", controller.status)

    def test_play_selected_step_does_not_respawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(motion_step(1))
            respawn_calls: list[str] = []
            controller.respawn_robot = lambda: respawn_calls.append("respawn")  # type: ignore[method-assign]
            controller._handle_play_step(["1"])
            self.assertEqual(respawn_calls, [])
            self.assertTrue(controller.playback.active)
            self.assertTrue(controller.playback.status_dict()["scheduled"])
            self.assertIn("Playback scheduled", controller.status)

    def test_playback_debug_selected_reports_plan_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            controller.manager.add_step(motion_step(1))
            controller.selected_step_index = 1
            controller.handle_command("playback_debug_selected")
            self.assertIn("controller_selected_index=1", controller.detail_text)
            self.assertIn("plan_count=1", controller.detail_text)


if __name__ == "__main__":
    unittest.main()
