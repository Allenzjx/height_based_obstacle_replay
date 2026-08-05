from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    COLLISION_PENETRATION,
    GROUND_OK,
    MISSING_WHEEL_COLLISION,
    VISUAL_ONLY_INTERSECTION,
    WheelGroundDiagnostics,
    build_robot_ground_diagnostics,
    compute_bounded_ground_correction,
)


class RobotGroundDiagnosticsTest(unittest.TestCase):
    def test_visual_only_intersection_does_not_allow_correction(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    visual_aabb_min_z=-0.01,
                    collision_aabb_min_z=0.001,
                    visual_ground_clearance_m=-0.01,
                    collision_ground_clearance_m=0.001,
                )
            ],
            penetration_tolerance_m=0.003,
        )

        allowed, dz, reason = compute_bounded_ground_correction(diag)

        self.assertEqual(diag.classification, VISUAL_ONLY_INTERSECTION)
        self.assertFalse(allowed)
        self.assertEqual(dz, 0.0)
        self.assertIn("COLLISION_PENETRATION", reason)

    def test_collision_penetration_computes_bounded_positive_correction(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    collision_aabb_min_z=-0.02,
                    collision_ground_clearance_m=-0.02,
                    collision_penetration_m=0.02,
                )
            ],
            penetration_tolerance_m=0.003,
        )

        allowed, dz, _reason = compute_bounded_ground_correction(diag, target_clearance_m=0.002, max_correction_m=0.01)

        self.assertEqual(diag.classification, COLLISION_PENETRATION)
        self.assertTrue(allowed)
        self.assertAlmostEqual(dz, 0.01)

    def test_missing_wheel_collision_is_explicit_failure(self) -> None:
        diag = build_robot_ground_diagnostics(
            [WheelGroundDiagnostics(wheel_name="front_left_ankle", visual_aabb_min_z=0.0, visual_ground_clearance_m=0.0)]
        )

        self.assertEqual(diag.classification, MISSING_WHEEL_COLLISION)
        self.assertEqual(diag.missing_collision_wheels, ["front_left_ankle"])

    def test_ok_classification_when_collision_clearance_is_non_negative(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_prim_paths=["/collision"],
                    collision_api_present=True,
                    collision_enabled=True,
                    visual_ground_clearance_m=0.001,
                    collision_ground_clearance_m=0.001,
                )
            ]
        )

        self.assertEqual(diag.classification, GROUND_OK)

if __name__ == "__main__":
    unittest.main()
