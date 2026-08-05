from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_robot_adapter import SimRobotAdapter, SimRobotAdapterConfig  # noqa: E402


class Cloneable:
    def clone(self) -> "Cloneable":
        return self


def make_adapter(settle: dict) -> SimRobotAdapter:
    adapter = SimRobotAdapter.__new__(SimRobotAdapter)
    adapter.config = SimRobotAdapterConfig()
    adapter.sim = SimpleNamespace(get_physics_dt=lambda: 1.0 / 120.0)
    adapter.robot = SimpleNamespace(data=SimpleNamespace(root_pose_w=Cloneable(), joint_pos=Cloneable()), write_data_to_sim=lambda: None)
    adapter.zero_root_velocity = Cloneable()
    adapter.command_zero_joint_vel = Cloneable()
    adapter.grounded_reference_valid = False
    adapter.grounded_reference_physics_valid = False
    adapter.grounded_reference_visual_valid = False
    adapter.grounded_reference_stable = False
    adapter.respawn_ready = False
    adapter.ground_reference_block_reason = ""
    adapter.grounded_reference_diagnostics = {}
    adapter.robot_ground_diagnostics = {}
    adapter.stop_wheels = lambda: None  # type: ignore[method-assign]
    adapter._targets_from_command_angles = lambda: {}  # type: ignore[method-assign]
    adapter.capture_command_state = lambda: {"servos": {}, "wheels": {}}  # type: ignore[method-assign]
    adapter.apply_commands_to_robot = lambda: None  # type: ignore[method-assign]
    adapter.settle_robot_on_ground = lambda label="": settle  # type: ignore[method-assign]
    return adapter


class GroundSettleTerminalWindowTest(unittest.TestCase):
    def test_peak_velocity_does_not_reject_final_stable_reference(self) -> None:
        adapter = make_adapter(
            {
                "stable": True,
                "stable_frames": 10,
                "stable_frames_required": 10,
                "max_abs_vertical_speed_m_s": 1.0,
                "max_abs_joint_velocity_rad_s": 1.0,
                "final_window_max_vertical_speed_m_s": 0.001,
                "final_window_max_joint_velocity_rad_s": 0.001,
                "final_window_max_root_z_delta_m": 0.0,
                "final_window_root_z_delta_threshold_m": 0.001,
                "ground_diagnostics": {
                    "checked": True,
                    "classification": "OK",
                    "ground_state": "PASS",
                    "physical_ground_safe": True,
                    "visual_ground_safe": True,
                    "missing_collision_wheels": [],
                    "unresolved_collision_wheels": [],
                    "maximum_collision_penetration_m": 0.0,
                },
            }
        )

        result = adapter.initialize_grounded_respawn_reference()

        self.assertTrue(result["grounded_reference_valid"])
        self.assertTrue(adapter.respawn_ready)

    def test_final_window_velocity_rejects_reference(self) -> None:
        adapter = make_adapter(
            {
                "stable": True,
                "stable_frames": 10,
                "stable_frames_required": 10,
                "max_abs_vertical_speed_m_s": 1.0,
                "max_abs_joint_velocity_rad_s": 1.0,
                "final_window_max_vertical_speed_m_s": 0.5,
                "final_window_max_joint_velocity_rad_s": 0.001,
                "final_window_max_root_z_delta_m": 0.0,
                "final_window_root_z_delta_threshold_m": 0.001,
                "ground_diagnostics": {
                    "checked": True,
                    "classification": "OK",
                    "ground_state": "PASS",
                    "physical_ground_safe": True,
                    "visual_ground_safe": True,
                    "missing_collision_wheels": [],
                    "unresolved_collision_wheels": [],
                    "maximum_collision_penetration_m": 0.0,
                },
            }
        )

        result = adapter.initialize_grounded_respawn_reference()

        self.assertFalse(result["grounded_reference_valid"])
        self.assertFalse(adapter.respawn_ready)
        self.assertIn("final root vertical speed", adapter.ground_reference_block_reason)


if __name__ == "__main__":
    unittest.main()
