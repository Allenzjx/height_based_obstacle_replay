from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    COLLIDER_RESOLUTION_FAILED,
    GROUND_STATE_UNVERIFIED,
    WheelGroundDiagnostics,
    build_robot_ground_diagnostics,
    sanitize_world_bound,
)


class FakeBound:
    def __init__(self, minimum: tuple[float, float, float], maximum: tuple[float, float, float], *, empty: bool = False) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.empty = empty

    def IsEmpty(self) -> bool:
        return self.empty

    def GetMin(self) -> tuple[float, float, float]:
        return self.minimum

    def GetMax(self) -> tuple[float, float, float]:
        return self.maximum


class GroundBoundSanitizationTest(unittest.TestCase):
    def test_normal_aabb_is_valid(self) -> None:
        info = sanitize_world_bound(FakeBound((-0.1, -0.2, 0.01), (0.1, 0.2, 0.2)))
        self.assertTrue(info["valid"])
        self.assertEqual(info["min"][2], 0.01)

    def test_empty_range_is_rejected(self) -> None:
        info = sanitize_world_bound(FakeBound((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), empty=True))
        self.assertFalse(info["valid"])
        self.assertIn("empty", info["rejection_reason"])

    def test_float_max_sentinel_is_rejected(self) -> None:
        info = sanitize_world_bound(FakeBound((3.402823466e38, 0.0, 0.0), (3.402823466e38, 1.0, 1.0)))
        self.assertFalse(info["valid"])
        self.assertIn("sentinel", info["rejection_reason"])

    def test_negative_float_max_sentinel_is_rejected(self) -> None:
        info = sanitize_world_bound(FakeBound((-3.402823466e38, 0.0, 0.0), (0.0, 1.0, 1.0)))
        self.assertFalse(info["valid"])
        self.assertIn("sentinel", info["rejection_reason"])

    def test_inf_nan_min_greater_than_max_and_huge_extent_are_rejected(self) -> None:
        self.assertFalse(sanitize_world_bound(FakeBound((math.inf, 0.0, 0.0), (1.0, 1.0, 1.0)))["valid"])
        self.assertFalse(sanitize_world_bound(FakeBound((math.nan, 0.0, 0.0), (1.0, 1.0, 1.0)))["valid"])
        self.assertFalse(sanitize_world_bound(FakeBound((2.0, 0.0, 0.0), (1.0, 1.0, 1.0)))["valid"])
        self.assertFalse(sanitize_world_bound(FakeBound((0.0, 0.0, 0.0), (1001.0, 1.0, 1.0)))["valid"])

    def test_invalid_collision_bound_does_not_pass_visual_warning(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_api_present=True,
                    collision_enabled=True,
                    collision_prim_paths=["/collision"],
                    collision_resolution_state=COLLIDER_RESOLUTION_FAILED,
                    visual_ground_clearance_m=-0.01,
                    collision_ground_clearance_m=None,
                    bounds_valid=False,
                    bounds_rejection_reason="FLT_MAX sentinel",
                )
            ]
        )
        self.assertEqual(diag.ground_state, GROUND_STATE_UNVERIFIED)
        self.assertFalse(diag.physical_ground_safe)
        self.assertEqual(diag.minimum_collision_clearance_m, None)


if __name__ == "__main__":
    unittest.main()
