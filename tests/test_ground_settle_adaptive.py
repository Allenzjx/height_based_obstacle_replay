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


def make_adapter(
    *,
    unstable_ticks: int,
    max_steps: int = 180,
    snapshot_valid: bool = True,
    root_velocity_valid: bool = True,
) -> SimRobotAdapter:
    adapter = SimRobotAdapter.__new__(SimRobotAdapter)
    adapter.config = SimRobotAdapterConfig(
        ground_settle_s=0.75,
        ground_settle_max_steps=max_steps,
        ground_stable_frames=10,
        ground_vertical_speed_threshold_m_s=0.01,
        ground_joint_speed_threshold_rad_s=0.02,
        ground_servo_speed_threshold_rad_s=0.02,
        ground_wheel_speed_threshold_rad_s=0.20,
    )
    adapter.sim_time = 0.0
    adapter.sim_steps = 0
    adapter.sim = SimpleNamespace(
        get_physics_dt=lambda: 1.0 / 120.0,
        step=lambda render=True: None,
        render=lambda: None,
    )
    adapter.robot = SimpleNamespace(
        write_data_to_sim=lambda: None,
        update=lambda _dt: None,
    )
    adapter.apply_commands_to_robot = lambda: None  # type: ignore[method-assign]
    adapter._render_step_timing = lambda: (8.0 / 120.0, 8)  # type: ignore[method-assign]
    adapter._current_root_z = lambda: 0.10  # type: ignore[method-assign]

    def unstable() -> bool:
        return int(adapter.sim_steps) <= int(unstable_ticks)

    adapter._ground_root_velocity_snapshot = lambda: {  # type: ignore[method-assign]
        "valid": root_velocity_valid,
        "error": "" if root_velocity_valid else "root_vel_w missing",
        "values": (
            [
                0.0,
                0.0,
                0.5 if unstable() else 0.001,
                0.0,
                0.0,
                0.0,
            ]
            if root_velocity_valid
            else []
        ),
    }

    def snapshot():
        velocities = {
            name: (0.5 if unstable() else 0.001)
            for name in SERVO_JOINT_NAMES
        }
        velocities.update({name: 0.05 for name in WHEEL_JOINT_NAMES})
        all_names = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
        return {
            "valid": snapshot_valid,
            "error": "" if snapshot_valid else "joint_pos_target missing",
            "joint_position_by_name": {name: 0.0 for name in all_names},
            "joint_velocity_vector": [velocities[name] for name in all_names],
            "joint_velocity_by_name": velocities,
            "joint_position_target_by_name": {name: 0.0 for name in all_names},
            "joint_target_minus_position_by_name": {name: 0.0 for name in all_names},
            "servo_command_target_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
            "servo_command_to_readback_error_by_name": {name: 0.0 for name in SERVO_JOINT_NAMES},
        }

    adapter._ground_joint_state_snapshot = snapshot  # type: ignore[method-assign]
    adapter._wheel_velocity_target_by_name = lambda: {  # type: ignore[method-assign]
        name: 0.0 for name in WHEEL_JOINT_NAMES
    }
    adapter._servo_position_error_by_name = lambda: {  # type: ignore[method-assign]
        name: 0.0 for name in SERVO_JOINT_NAMES
    }
    adapter.validate_robot_ground_contact = lambda apply_correction=False: {  # type: ignore[method-assign]
        "checked": True,
        "classification": "OK",
        "physical_ground_safe": True,
        "visual_ground_safe": True,
        "missing_collision_wheels": [],
        "unresolved_collision_wheels": [],
        "maximum_collision_penetration_m": 0.0,
    }
    return adapter


class GroundSettleAdaptiveTest(unittest.TestCase):
    def test_full_180_tick_budget_and_terminal_ten_tick_acceptance(self) -> None:
        adapter = make_adapter(unstable_ticks=170)
        observed = []

        result = adapter.settle_robot_on_ground(
            label="adaptive",
            tick_observer=lambda frame: observed.append(dict(frame)) or frame,
        )

        self.assertEqual(result["steps_run"], 180)
        self.assertEqual(result["step_budget"], 180)
        self.assertEqual(result["consecutive_stable_ticks"], 10)
        self.assertEqual(result["stop_reason"], "consecutive_stable")
        self.assertTrue(result["stable"])
        self.assertEqual(len(result["settle_tick_trace"]), 180)
        self.assertEqual(len(observed), 180)
        self.assertEqual(result["rolling_window_capacity"], 60)
        # The retained 60-frame diagnostic still contains landing transients,
        # but it cannot reject the authoritative final ten strict ticks.
        self.assertFalse(result["final_window_stable"])

    def test_stable_from_first_tick_stops_at_ten(self) -> None:
        result = make_adapter(unstable_ticks=0).settle_robot_on_ground(
            label="early"
        )

        self.assertEqual(result["steps_run"], 10)
        self.assertTrue(result["stopped_early"])
        self.assertTrue(result["stable"])

    def test_terminal_heuristic_cannot_synthesize_stable_ticks(self) -> None:
        result = make_adapter(unstable_ticks=180).settle_robot_on_ground(
            label="never"
        )

        self.assertEqual(result["steps_run"], 180)
        self.assertEqual(result["consecutive_stable_ticks"], 0)
        self.assertEqual(result["stop_reason"], "max_steps_exhausted")
        self.assertFalse(result["stable"])

    def test_grounding_budget_is_hard_capped_at_180_ticks(self) -> None:
        result = make_adapter(
            unstable_ticks=250,
            max_steps=250,
        ).settle_robot_on_ground(label="hard-cap")

        self.assertEqual(result["steps_run"], 180)
        self.assertEqual(result["step_budget"], 180)
        self.assertEqual(result["stop_reason"], "max_steps_exhausted")

    def test_explicit_short_duration_keeps_its_cap(self) -> None:
        result = make_adapter(unstable_ticks=180).settle_robot_on_ground(
            label="short",
            duration_s=0.30,
        )

        self.assertEqual(result["steps_run"], 36)
        self.assertEqual(result["stop_reason"], "explicit_duration_exhausted")

    def test_missing_joint_target_evidence_fails_closed(self) -> None:
        result = make_adapter(
            unstable_ticks=0,
            snapshot_valid=False,
        ).settle_robot_on_ground(label="missing")

        self.assertEqual(result["steps_run"], 180)
        self.assertFalse(result["stable"])
        self.assertFalse(result["acceptance_window_evidence_valid"])
        self.assertEqual(result["consecutive_stable_ticks"], 0)

    def test_missing_root_velocity_evidence_fails_closed(self) -> None:
        result = make_adapter(
            unstable_ticks=0,
            root_velocity_valid=False,
        ).settle_robot_on_ground(label="missing-root-velocity")

        self.assertEqual(result["steps_run"], 180)
        self.assertFalse(result["stable"])
        self.assertEqual(result["consecutive_stable_ticks"], 0)
        self.assertFalse(
            result["settle_tick_trace"][-1]["root_velocity_evidence_valid"]
        )

    def test_live_root_velocity_reader_rejects_missing_and_nonfinite_data(self) -> None:
        adapter = SimRobotAdapter.__new__(SimRobotAdapter)
        adapter.robot = SimpleNamespace(data=SimpleNamespace(root_vel_w=None))
        self.assertFalse(adapter._ground_root_velocity_snapshot()["valid"])

        adapter.robot.data.root_vel_w = [
            [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
        ]
        snapshot = adapter._ground_root_velocity_snapshot()
        self.assertFalse(snapshot["valid"])
        self.assertIn("non-finite", snapshot["error"])

    def test_diagnostic_observer_cannot_change_canonical_acceptance(self) -> None:
        def corrupt_diagnostic_copy(frame):
            frame["strict_tick_stable"] = False
            frame["stable_count"] = 0
            frame["joint_state_evidence_valid"] = False
            return frame

        result = make_adapter(unstable_ticks=0).settle_robot_on_ground(
            label="observer-isolation",
            tick_observer=corrupt_diagnostic_copy,
        )

        self.assertTrue(result["stable"])
        self.assertEqual(result["steps_run"], 10)
        self.assertEqual(result["consecutive_stable_ticks"], 10)
        self.assertTrue(result["acceptance_window_evidence_valid"])
        self.assertFalse(result["settle_tick_trace"][-1]["strict_tick_stable"])


if __name__ == "__main__":
    unittest.main()
