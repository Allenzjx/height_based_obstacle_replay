from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from command_model import KNEE_LIMIT_DEG, clamp_servo_command  # noqa: E402


class KneeNegativeCommandSpaceTest(unittest.TestCase):
    def test_knee_limit_range_keeps_negative_side(self) -> None:
        self.assertEqual(KNEE_LIMIT_DEG, (-60.0, 210.0))

    def test_front_knee_negative_command_is_allowed(self) -> None:
        self.assertEqual(clamp_servo_command("front_left_knee", -30.0), -30.0)
        self.assertEqual(clamp_servo_command("front_left_knee", -80.0), -60.0)

    def test_rear_knee_negative_command_is_allowed(self) -> None:
        self.assertEqual(clamp_servo_command("rear_right_knee", -30.0), -30.0)


if __name__ == "__main__":
    unittest.main()
