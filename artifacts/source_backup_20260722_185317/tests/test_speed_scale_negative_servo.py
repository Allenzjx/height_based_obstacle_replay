from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from speed_scale import scale_manual_motion_command  # noqa: E402


class SpeedScaleNegativeServoTest(unittest.TestCase):
    def test_negative_servo_scale_one_stays_negative(self) -> None:
        scaled = scale_manual_motion_command(
            "servo front_left_knee -30",
            default_wheel_speed=1.0,
            max_wheel_speed=3.0,
            wheel_speed_scale=1.0,
            servo_command_scale=1.0,
        )
        self.assertEqual(scaled.scaled_command, "servo front_left_knee -30")
        self.assertEqual(scaled.scaled_speed_values, (-30.0,))

    def test_negative_servo_scale_half_stays_negative(self) -> None:
        scaled = scale_manual_motion_command(
            "servo front_left_knee -30",
            default_wheel_speed=1.0,
            max_wheel_speed=3.0,
            wheel_speed_scale=1.0,
            servo_command_scale=0.5,
        )
        self.assertEqual(scaled.scaled_command, "servo front_left_knee -15")
        self.assertLess(scaled.scaled_speed_values[0], 0.0)

    def test_negative_servo_scale_zero_is_explicit_zero_with_warning(self) -> None:
        scaled = scale_manual_motion_command(
            "servo front_left_knee -30",
            default_wheel_speed=1.0,
            max_wheel_speed=3.0,
            wheel_speed_scale=1.0,
            servo_command_scale=0.0,
        )
        self.assertEqual(scaled.scaled_command, "servo front_left_knee 0")
        self.assertEqual(scaled.scaled_speed_values, (0.0,))
        self.assertTrue(scaled.warnings)


if __name__ == "__main__":
    unittest.main()
