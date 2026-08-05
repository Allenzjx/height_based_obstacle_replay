from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_robot_adapter import (  # noqa: E402
    PHYSX_SAFE_LIMIT_MAX_RAD,
    PHYSX_SAFE_LIMIT_MIN_RAD,
    desired_actual_limits_deg,
    safe_rad_limits_for_actual_deg,
)


class SafeJointLimitMathTest(unittest.TestCase):
    def test_desired_limits_sort_for_front_knee(self) -> None:
        self.assertEqual(desired_actual_limits_deg("front_left_knee", 0.0), (-60.0, 210.0))

    def test_desired_limits_sort_for_rear_knee(self) -> None:
        self.assertEqual(desired_actual_limits_deg("rear_left_knee", 0.0), (-210.0, 60.0))

    def test_safe_rad_limits_clamp_to_two_pi(self) -> None:
        record = safe_rad_limits_for_actual_deg(-1000.0, 1000.0)
        self.assertEqual(record["min_rad"], PHYSX_SAFE_LIMIT_MIN_RAD)
        self.assertEqual(record["max_rad"], PHYSX_SAFE_LIMIT_MAX_RAD)
        self.assertTrue(record["clamped"])
        self.assertTrue(record["warnings"])
        self.assertFalse(record["skip"])

    def test_safe_rad_limits_keep_min_less_than_max(self) -> None:
        record = safe_rad_limits_for_actual_deg(-60.0, 210.0)
        self.assertLess(record["min_rad"], record["max_rad"])
        self.assertAlmostEqual(record["min_rad"], math.radians(-60.0))
        self.assertAlmostEqual(record["max_rad"], math.radians(210.0))


if __name__ == "__main__":
    unittest.main()
