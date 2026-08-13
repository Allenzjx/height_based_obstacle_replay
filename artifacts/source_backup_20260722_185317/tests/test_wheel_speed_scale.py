from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S  # noqa: E402
from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from speed_scale import MotionScaleConfig, scale_manual_motion_command  # noqa: E402


class WheelSpeedScaleTest(unittest.TestCase):
    def test_manual_w_shortcut_uses_scaled_default_rad_s(self) -> None:
        scaled = scale_manual_motion_command(
            "w",
            default_wheel_speed=0.5,
            max_wheel_speed=2.0,
            wheel_speed_scale=2.0,
        )
        self.assertEqual(scaled.scaled_command, "wheel all 1")
        self.assertEqual(scaled.raw_speed_values, (0.5,))
        self.assertEqual(scaled.scaled_speed_values, (1.0,))

    def test_explicit_wheel_command_scales_and_clamps(self) -> None:
        scaled = scale_manual_motion_command(
            "wheel fl 2.0",
            default_wheel_speed=0.5,
            max_wheel_speed=3.0,
            wheel_speed_scale=2.0,
        )
        self.assertEqual(scaled.scaled_command, "wheel fl 3")

    def test_pair_wheel_command_scales_both_sides(self) -> None:
        scaled = scale_manual_motion_command(
            "wheel 1 -1",
            default_wheel_speed=0.5,
            max_wheel_speed=3.0,
            wheel_speed_scale=2.0,
        )
        self.assertEqual(scaled.scaled_command, "wheel 2 -2")

    def test_motion_scale_can_be_disabled_for_manual_control(self) -> None:
        config = MotionScaleConfig(global_motion_speed_scale=2.0, wheel_speed_scale=2.0)
        self.assertEqual(config.manual_wheel_scale, 4.0)
        config.apply_to_manual_control = False
        self.assertEqual(config.manual_wheel_scale, 1.0)

    def test_cli_wheel_speed_defaults_and_legacy_alias_are_rad_s(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        self.assertAlmostEqual(args.max_wheel_speed_rad_s, DEFAULT_MAX_WHEEL_SPEED_RAD_S)
        self.assertAlmostEqual(args.default_wheel_speed_rad_s, DEFAULT_MAX_WHEEL_SPEED_RAD_S * 0.25)

        args = build_parser().parse_args(["--ui", "--no-sim", "--max-wheel-speed", "4.0"])
        normalize_motion_args(args)
        self.assertEqual(args.max_wheel_speed_rad_s, 4.0)
        self.assertEqual(args.default_wheel_speed_rad_s, 1.0)


if __name__ == "__main__":
    unittest.main()
