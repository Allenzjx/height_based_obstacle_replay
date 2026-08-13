from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    COLLIDER_CONFIRMED_MISSING,
    COLLIDER_RESOLUTION_FAILED,
    COLLISION_PENETRATION,
    GROUND_OK,
    WheelGroundDiagnostics,
    build_robot_ground_diagnostics,
)


class GroundDiagnosticStatesTest(unittest.TestCase):
    def test_resolver_failure_is_unverified_not_missing(self) -> None:
        diag = build_robot_ground_diagnostics(
            [WheelGroundDiagnostics(wheel_name="front_left_ankle", collision_resolution_state=COLLIDER_RESOLUTION_FAILED)]
        )

        self.assertEqual(diag.classification, COLLIDER_RESOLUTION_FAILED)
        self.assertEqual(diag.ground_state, "UNVERIFIED")
        self.assertEqual(diag.missing_collision_wheels, [])
        self.assertEqual(diag.unresolved_collision_wheels, ["front_left_ankle"])

    def test_confirmed_missing_and_penetration_are_fail(self) -> None:
        missing = build_robot_ground_diagnostics(
            [WheelGroundDiagnostics(wheel_name="front_left_ankle", body_prim_path="/wheel", collision_resolution_state=COLLIDER_CONFIRMED_MISSING)]
        )
        penetrating = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_resolution_state=GROUND_OK,
                    collision_api_present=True,
                    collision_prim_paths=["/collider"],
                    collision_ground_clearance_m=-0.01,
                    collision_penetration_m=0.01,
                )
            ]
        )

        self.assertEqual(missing.classification, COLLIDER_CONFIRMED_MISSING)
        self.assertEqual(missing.ground_state, "FAIL")
        self.assertEqual(penetrating.classification, COLLISION_PENETRATION)
        self.assertEqual(penetrating.ground_state, "FAIL")


if __name__ == "__main__":
    unittest.main()
