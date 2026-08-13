from __future__ import annotations

import argparse
import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_process_client import build_worker_command, build_worker_config  # noqa: E402
from sim_robot_adapter import SimRobotAdapter  # noqa: E402
from robot_ground_diagnostics import run_viewport_action_with_physics_guard  # noqa: E402


class GroundedRespawnAndViewportGuardTest(unittest.TestCase):
    def test_viewport_guard_does_not_call_mutating_robot_operations_when_state_is_stable(self) -> None:
        adapter = GuardFakeAdapter()
        scene = GuardFakeScene()

        result, guard = run_viewport_action_with_physics_guard(
            adapter=adapter,
            scene_handle=scene,
            action="open_camera_viewport",
            callback=lambda: "opened",
            enabled=True,
            allow_restore_on_failure=True,
        )

        self.assertEqual(result, "opened")
        self.assertTrue(guard.passed)
        self.assertEqual(adapter.calls["respawn"], 0)
        self.assertEqual(adapter.calls["reset"], 0)
        self.assertEqual(adapter.calls["restore"], 0)
        self.assertEqual(adapter.calls["root_write"], 0)
        self.assertEqual(adapter.calls["joint_write"], 0)
        self.assertEqual(adapter.calls["sim_step"], 0)

    def test_respawn_source_blocks_raw_spawn_fallback_when_reference_invalid(self) -> None:
        source = inspect.getsource(SimRobotAdapter.respawn_robot)

        self.assertIn("raw spawn fallback was not used", source)
        self.assertIn("root_pose = self.grounded_respawn_root_pose", source)
        self.assertIn("joint_pos = self.grounded_respawn_joint_pos", source)
        self.assertIn("write_root_velocity_to_sim(self.zero_root_velocity)", source)

    def test_worker_config_and_cli_include_ground_and_viewport_flags(self) -> None:
        args = argparse.Namespace(
            height_cm=5,
            robot_usd="robot.usd",
            save_usd="scene.usd",
            viewport_physics_guard=True,
            robot_auto_ground_correction=True,
            camera_view_active_fallback=True,
            worker_smoke_camera_view_ground_contact=True,
            robot_ground_settle_s=0.5,
            robot_ground_settle_max_steps=50,
            robot_ground_stable_frames=3,
            robot_ground_vertical_speed_threshold_m_s=0.02,
            robot_ground_joint_speed_threshold_rad_s=0.03,
            robot_ground_clearance_m=0.004,
            robot_ground_penetration_tolerance_m=0.005,
            robot_max_ground_correction_m=0.08,
        )

        config = build_worker_config(args, host="127.0.0.1", port=1234)
        command = build_worker_command(args, host="127.0.0.1", port=1234)

        self.assertTrue(config["viewport_physics_guard"])
        self.assertTrue(config["robot_auto_ground_correction"])
        self.assertTrue(config["worker_smoke_camera_view_ground_contact"])
        self.assertIn("--viewport-physics-guard", command)
        self.assertIn("--robot-auto-ground-correction", command)
        self.assertIn("--camera-view-active-fallback", command)
        self.assertIn("--worker-smoke-camera-view-ground-contact", command)
        self.assertIn("--robot-ground-settle-s", command)


class GuardFakeSim:
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls

    def get_physics_dt(self) -> float:
        return 1.0 / 120.0

    def step(self) -> None:
        self.calls["sim_step"] += 1


class GuardFakeData:
    root_pose_w = [[0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0]]
    root_vel_w = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    joint_pos = [[0.0, 0.0]]
    joint_vel = [[0.0, 0.0]]


class GuardFakeRobot:
    def __init__(self, calls: dict[str, int]) -> None:
        self.calls = calls
        self.data = GuardFakeData()

    def reset(self) -> None:
        self.calls["reset"] += 1

    def write_root_pose_to_sim(self, _value: object) -> None:
        self.calls["root_write"] += 1

    def write_joint_state_to_sim(self, _pos: object, _vel: object) -> None:
        self.calls["joint_write"] += 1


class GuardFakeAdapter:
    def __init__(self) -> None:
        self.calls = {name: 0 for name in ("respawn", "reset", "restore", "root_write", "joint_write", "sim_step")}
        self.robot = GuardFakeRobot(self.calls)
        self.sim = GuardFakeSim(self.calls)
        self.sim_time = 0.0
        self.sim_steps = 0
        self.grounded_reference_valid = True

    def capture_sim_state(self) -> dict[str, object]:
        return {
            "root_pose": self.robot.data.root_pose_w,
            "root_velocity": self.robot.data.root_vel_w,
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
            "command_state": self.capture_command_state(),
        }

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {"servos": {}, "wheels": {"front_left_ankle": 0.0, "front_right_ankle": 0.0, "rear_left_ankle": 0.0, "rear_right_ankle": 0.0}}

    def restore_sim_state(self, _state: dict[str, object]) -> None:
        self.calls["restore"] += 1

    def stop_wheels(self) -> None:
        pass


class GuardFakeConfig:
    ground_z_m = 0.0


class GuardFakeScene:
    config = GuardFakeConfig()
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
