from __future__ import annotations

import argparse
import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_process  # noqa: E402
from sim_worker_runtime import create_adapter_config_from_args, initialize_adapter_ground_reference  # noqa: E402


class SubprocessGroundReferenceTest(unittest.TestCase):
    def test_adapter_config_from_args_carries_ground_thresholds(self) -> None:
        args = argparse.Namespace(
            max_wheel_speed_rad_s=9.0,
            default_wheel_speed_rad_s=2.0,
            wheel_direction=-1.0,
            apply_safe_servo_joint_limits=True,
            apply_physx_joint_limits=False,
            robot_ground_settle_s=0.4,
            robot_ground_settle_max_steps=44,
            robot_ground_stable_frames=7,
            robot_ground_vertical_speed_threshold_m_s=0.012,
            robot_ground_joint_speed_threshold_rad_s=0.034,
            robot_ground_clearance_m=0.005,
            robot_ground_penetration_tolerance_m=0.006,
            robot_auto_ground_correction=True,
            robot_max_ground_correction_m=0.09,
        )

        cfg = create_adapter_config_from_args(args)

        self.assertEqual(cfg.ground_settle_max_steps, 44)
        self.assertEqual(cfg.ground_stable_frames, 7)
        self.assertAlmostEqual(cfg.ground_vertical_speed_threshold_m_s, 0.012)
        self.assertAlmostEqual(cfg.ground_joint_speed_threshold_rad_s, 0.034)
        self.assertAlmostEqual(cfg.ground_penetration_tolerance_m, 0.006)
        self.assertTrue(cfg.auto_ground_correction)

    def test_subprocess_initializes_ground_reference_before_adapter_ready(self) -> None:
        source = inspect.getsource(sim_worker_process.run_worker)

        self.assertLess(source.index("initialize_adapter_ground_reference(adapter)"), source.index('set_phase("adapter_ready")'))

    def test_initialize_adapter_ground_reference_delegates_to_adapter(self) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.called = False

            def initialize_grounded_respawn_reference(self) -> dict[str, object]:
                self.called = True
                return {"grounded_reference_valid": True}

        adapter = Adapter()
        result = initialize_adapter_ground_reference(adapter)

        self.assertTrue(adapter.called)
        self.assertTrue(result["grounded_reference_valid"])

    def test_initialize_adapter_ground_reference_forwards_diagnostic_observer(self) -> None:
        observer = object()

        class Adapter:
            received = None

            def initialize_grounded_respawn_reference(self, *, tick_observer=None):
                self.received = tick_observer
                return {"grounded_reference_valid": True}

        adapter = Adapter()
        result = initialize_adapter_ground_reference(
            adapter,
            tick_observer=observer,
        )

        self.assertIs(adapter.received, observer)
        self.assertTrue(result["grounded_reference_valid"])


if __name__ == "__main__":
    unittest.main()
