from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_robot_adapter import SimRobotAdapterConfig  # noqa: E402


class RuntimePhysxLimitDefaultTest(unittest.TestCase):
    def test_adapter_config_default_writes_authoritative_joint_limits(self) -> None:
        self.assertTrue(SimRobotAdapterConfig().apply_joint_limits_to_sim)
        self.assertTrue(SimRobotAdapterConfig().apply_safe_servo_joint_limits)

    def test_cli_default_applies_physx_joint_limits(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        self.assertTrue(args.apply_physx_joint_limits)
        self.assertTrue(args.apply_safe_servo_joint_limits)

    def test_cli_flag_enables_physx_joint_limits(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim", "--apply-physx-joint-limits"])
        normalize_motion_args(args)
        self.assertTrue(args.apply_physx_joint_limits)

    def test_explicit_opt_out_is_available_for_diagnostics(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim", "--no-apply-physx-joint-limits"])
        normalize_motion_args(args)
        self.assertFalse(args.apply_physx_joint_limits)


if __name__ == "__main__":
    unittest.main()
