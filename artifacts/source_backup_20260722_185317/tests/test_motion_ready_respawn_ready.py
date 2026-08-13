from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import (  # noqa: E402
    GROUND_STATE_PASS_WITH_VISUAL_WARNING,
    VISUAL_ONLY_INTERSECTION,
    WheelGroundDiagnostics,
    build_robot_ground_diagnostics,
    motion_status_from_worker_status,
    respawn_status_from_worker_status,
)
from sim_worker_runtime import enrich_runtime_readiness  # noqa: E402


class MotionReadyRespawnReadyTest(unittest.TestCase):
    def test_visual_warning_with_reliable_collision_is_motion_ready(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_api_present=True,
                    collision_enabled=True,
                    collision_prim_paths=["/collision"],
                    collision_resolution_state="OK",
                    collision_world_min_z=0.002,
                    collision_ground_clearance_m=0.002,
                    visual_ground_clearance_m=-0.01,
                    bounds_valid=True,
                )
            ]
        ).to_dict()
        status = enrich_runtime_readiness({"runtime_ready": True, "robot_ground": diag, "grounded_reference_valid": False})

        self.assertEqual(status["ground_state"], GROUND_STATE_PASS_WITH_VISUAL_WARNING)
        self.assertTrue(status["motion_ready"])
        self.assertFalse(status["respawn_ready"])
        self.assertIn("grounded respawn reference", status["respawn_block_reason"])

    def test_respawn_ready_requires_valid_stable_reference(self) -> None:
        diag = build_robot_ground_diagnostics(
            [
                WheelGroundDiagnostics(
                    wheel_name="front_left_ankle",
                    collision_api_present=True,
                    collision_enabled=True,
                    collision_prim_paths=["/collision"],
                    collision_resolution_state="OK",
                    collision_world_min_z=0.001,
                    collision_ground_clearance_m=0.001,
                    bounds_valid=True,
                )
            ],
            grounded_respawn_reference_valid=True,
        ).to_dict()
        status = {
            "runtime_ready": True,
            "robot_ground": diag,
            "grounded_reference_valid": True,
            "grounded_reference_stable": True,
        }

        self.assertTrue(motion_status_from_worker_status(status)[0])
        self.assertTrue(respawn_status_from_worker_status(status)[0])

    def test_unverified_never_motion_ready(self) -> None:
        status = {
            "runtime_ready": True,
            "robot_ground": {
                "checked": True,
                "classification": "COLLIDER_RESOLUTION_FAILED",
                "ground_state": "UNVERIFIED",
                "physical_ground_safe": False,
            },
            "grounded_reference_valid": True,
            "grounded_reference_stable": True,
        }

        self.assertFalse(motion_status_from_worker_status(status)[0])
        self.assertFalse(respawn_status_from_worker_status(status)[0])

    def test_ground_reference_result_allows_visual_warning_when_physical_safe(self) -> None:
        from sim_worker_runtime import ground_reference_result_is_valid

        self.assertTrue(
            ground_reference_result_is_valid(
                {
                    "grounded_reference_valid": True,
                    "grounded_reference_physics_valid": True,
                    "grounded_reference_stable": True,
                    "grounded_reference_diagnostics": {
                        "checked": True,
                        "classification": VISUAL_ONLY_INTERSECTION,
                        "physical_ground_safe": True,
                    },
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
