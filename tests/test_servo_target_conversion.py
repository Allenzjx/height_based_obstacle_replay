from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_robot_adapter import actual_target_deg_for_command, desired_actual_limits_deg  # noqa: E402


class ServoTargetConversionTest(unittest.TestCase):
    def test_front_knee_sign_positive(self) -> None:
        standing = 12.0
        self.assertEqual(actual_target_deg_for_command("front_left_knee", -30.0, standing), standing - 30.0)
        self.assertEqual(actual_target_deg_for_command("front_left_knee", 30.0, standing), standing + 30.0)

    def test_rear_knee_sign_negative(self) -> None:
        standing = -8.0
        self.assertEqual(actual_target_deg_for_command("rear_left_knee", -30.0, standing), standing + 30.0)
        self.assertEqual(actual_target_deg_for_command("rear_left_knee", 30.0, standing), standing - 30.0)

    def test_final_target_limits_are_sorted_for_sign_positive(self) -> None:
        self.assertEqual(desired_actual_limits_deg("front_left_knee", 10.0), (-50.0, 220.0))

    def test_final_target_limits_are_sorted_for_sign_negative(self) -> None:
        self.assertEqual(desired_actual_limits_deg("rear_left_knee", 10.0), (-200.0, 70.0))


if __name__ == "__main__":
    unittest.main()
