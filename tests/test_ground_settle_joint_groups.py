from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES  # noqa: E402
from sim_robot_adapter import SimRobotAdapter, SimRobotAdapterConfig  # noqa: E402


def make_adapter(joint_velocities: dict[str, float], wheel_targets: dict[str, float] | None = None) -> SimRobotAdapter:
    adapter = SimRobotAdapter.__new__(SimRobotAdapter)
    adapter.config = SimRobotAdapterConfig(
        ground_stable_frames=3,
        ground_settle_max_steps=10,
        ground_vertical_speed_threshold_m_s=0.01,
        ground_joint_speed_threshold_rad_s=0.02,
        ground_servo_speed_threshold_rad_s=0.02,
        ground_wheel_speed_threshold_rad_s=0.20,
    )
    adapter.sim = SimpleNamespace(get_physics_dt=lambda: 0.1, step=lambda: None)
    adapter.robot = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda _dt: None)
    adapter.sim_time = 0.0
    adapter.sim_steps = 0
    adapter.apply_commands_to_robot = lambda: None  # type: ignore[method-assign]
    adapter._current_root_z = lambda: 0.10  # type: ignore[method-assign]
    adapter._ground_root_velocity_snapshot = lambda: {  # type: ignore[method-assign]
        "valid": True,
        "error": "",
        "values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    adapter._joint_velocity_by_name = lambda: dict(joint_velocities)  # type: ignore[method-assign]
    adapter._joint_velocity_vector = lambda: list(joint_velocities.values())  # type: ignore[method-assign]
    resolved_wheel_targets = dict(
        wheel_targets or {name: 0.0 for name in WHEEL_JOINT_NAMES}
    )
    adapter._ground_joint_state_snapshot = lambda: {  # type: ignore[method-assign]
        "valid": True,
        "error": "",
        "joint_position_by_name": {name: 0.0 for name in joint_velocities},
        "joint_velocity_vector": list(joint_velocities.values()),
        "joint_velocity_by_name": dict(joint_velocities),
        "joint_position_target_by_name": {name: 0.0 for name in joint_velocities},
        "joint_position_target_buffer_by_name": {
            name: 0.0 for name in joint_velocities
        },
        "joint_velocity_target_by_name": {
            name: resolved_wheel_targets.get(name, 0.0)
            for name in joint_velocities
        },
        "joint_velocity_target_buffer_by_name": {
            name: resolved_wheel_targets.get(name, 0.0)
            for name in joint_velocities
        },
        "joint_target_minus_position_by_name": {name: 0.0 for name in joint_velocities},
        "servo_command_target_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
        "servo_command_to_readback_error_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
    }
    adapter._wheel_velocity_target_by_name = lambda: dict(resolved_wheel_targets)  # type: ignore[method-assign]
    adapter.validate_robot_ground_contact = lambda apply_correction=False: {  # type: ignore[method-assign]
        "checked": True,
        "classification": "OK",
        "ground_state": "PASS",
        "physical_ground_safe": True,
        "visual_ground_safe": True,
        "missing_collision_wheels": [],
        "unresolved_collision_wheels": [],
        "maximum_collision_penetration_m": 0.0,
        "wheels": [
            {
                "wheel_name": name,
                "joint_name": name,
                "bounds_valid": True,
                "bounds_finite": True,
                "collision_ground_clearance_m": 0.0,
                "collision_penetration_m": 0.0,
            }
            for name in WHEEL_JOINT_NAMES
        ],
    }
    return adapter


class GroundSettleJointGroupsTest(unittest.TestCase):
    def test_wheel_jitter_below_wheel_threshold_does_not_block_chassis_stability(self) -> None:
        velocities = {name: 0.001 for name in SERVO_JOINT_NAMES}
        velocities.update({name: 0.05 for name in WHEEL_JOINT_NAMES})
        adapter = make_adapter(velocities)

        result = adapter.settle_robot_on_ground(label="unit")

        self.assertTrue(result["stable"])
        self.assertTrue(result["servo_pose_stable"])
        self.assertTrue(result["wheel_state_stable"])
        self.assertAlmostEqual(result["final_wheel_speed"], 0.05)
        self.assertIn("joint_velocity_by_name", result)
        self.assertEqual(result["offending_joint_names"], [])

    def test_high_wheel_velocity_blocks_wheel_state(self) -> None:
        velocities = {name: 0.001 for name in SERVO_JOINT_NAMES}
        velocities.update({name: 0.5 for name in WHEEL_JOINT_NAMES})
        adapter = make_adapter(velocities)

        result = adapter.settle_robot_on_ground(label="unit")

        self.assertFalse(result["stable"])
        self.assertTrue(result["servo_pose_stable"])
        self.assertFalse(result["wheel_state_stable"])
        self.assertGreater(result["final_wheel_speed"], result["wheel_speed_threshold_rad_s"])
        self.assertIn("front_left_ankle", result["offending_joint_names"])


if __name__ == "__main__":
    unittest.main()
