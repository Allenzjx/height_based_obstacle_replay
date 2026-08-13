from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_worker_runtime import execute_viewport_action_with_guard  # noqa: E402


class CameraViewportPhysicsInvarianceTest(unittest.TestCase):
    def test_guard_failure_stops_wheels_and_does_not_restore_state(self) -> None:
        adapter = FakeAdapter()
        vision = FakeVision(adapter)
        args = argparse.Namespace(
            viewport_physics_guard=True,
            camera_view_active_fallback=False,
            camera_view_pending_timeout_s=2.0,
            camera_view_pending_max_retries=30,
            robot_ground_penetration_tolerance_m=0.003,
            headless=False,
        )

        ok, text, status = execute_viewport_action_with_guard(
            args=args,
            adapter=adapter,
            scene_handle=FakeScene(),
            vision_processor=vision,
            action="open_camera_viewport",
            payload={"request_id": "test", "action_revision": 1},
            allow_state_restore=False,
        )

        self.assertFalse(ok)
        self.assertIn("Viewport physics guard failed", text)
        self.assertEqual(adapter.restore_calls, 0)
        self.assertGreater(adapter.stop_calls, 0)
        self.assertFalse(status["physics_guard_passed"])


class FakeData:
    def __init__(self) -> None:
        self.root_pose_w = [[0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0]]
        self.root_vel_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        self.joint_pos = [[0.0, 0.0]]
        self.joint_vel = [[0.0, 0.0]]


class FakeRobot:
    def __init__(self) -> None:
        self.data = FakeData()

    def write_data_to_sim(self) -> None:
        return None


class FakeAdapter:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.sim_time = 0.0
        self.sim_steps = 0
        self.restore_calls = 0
        self.stop_calls = 0

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {"servos": {}, "wheels": {"front_left_ankle": 0.0, "front_right_ankle": 0.0, "rear_left_ankle": 0.0, "rear_right_ankle": 0.0}}

    def capture_sim_state(self) -> dict[str, object]:
        return {
            "root_pose": self.robot.data.root_pose_w,
            "root_velocity": self.robot.data.root_vel_w,
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
            "command_state": self.capture_command_state(),
        }

    def restore_sim_state(self, _state: dict[str, object]) -> None:
        self.restore_calls += 1

    def stop_wheels(self) -> None:
        self.stop_calls += 1

    def apply_commands_to_robot(self) -> None:
        return None


class FakeVision:
    def __init__(self, adapter: FakeAdapter) -> None:
        self.adapter = adapter
        self.camera_view_status: dict[str, object] = {}

    def handle_control(self, action: str, **_payload: object) -> tuple[bool, str]:
        self.adapter.robot.data.root_pose_w = [[0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]]
        self.camera_view_status = {"requested_action": action, "supported": True, "active": True, "completed": True}
        return True, "opened"


class FakeConfig:
    ground_z_m = 0.0


class FakeScene:
    config = FakeConfig()
    fake_wheel_ground_rows = [
        {
            "wheel_name": "front_left_ankle",
            "collision_prim_paths": ["/collision"],
            "collision_api_present": True,
            "collision_enabled": True,
            "visual_ground_clearance_m": 0.001,
            "collision_ground_clearance_m": 0.001,
        }
    ]


if __name__ == "__main__":
    unittest.main()
