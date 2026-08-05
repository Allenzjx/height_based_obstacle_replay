from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from robot_ground_diagnostics import GROUND_OK, VISUAL_ONLY_INTERSECTION  # noqa: E402
from sim_robot_adapter import SimRobotAdapter  # noqa: E402
from sim_worker_runtime import ground_reference_result_is_valid  # noqa: E402


class GroundReferenceStabilityTest(unittest.TestCase):
    def test_ground_reference_requires_stability_and_ok_classification(self) -> None:
        source = inspect.getsource(SimRobotAdapter.initialize_grounded_respawn_reference)

        self.assertIn('settle.get("stable"', source)
        self.assertIn("maximum_collision_penetration_m", source)
        self.assertIn("missing_collision_wheels", source)
        self.assertIn("ground_vertical_speed_threshold_m_s", source)
        self.assertIn("ground_joint_speed_threshold_rad_s", source)
        self.assertIn("classification != GROUND_OK", source)

    def test_visual_only_reference_is_valid_when_physical_ground_is_safe(self) -> None:
        result = {
            "grounded_reference_valid": True,
            "grounded_reference_physics_valid": True,
            "grounded_reference_stable": True,
            "grounded_reference_diagnostics": {
                "checked": True,
                "classification": VISUAL_ONLY_INTERSECTION,
                "physical_ground_safe": True,
            },
        }

        self.assertTrue(ground_reference_result_is_valid(result))

    def test_ok_reference_result_is_valid(self) -> None:
        result = {
            "grounded_reference_valid": True,
            "grounded_reference_physics_valid": True,
            "grounded_reference_stable": True,
            "grounded_reference_diagnostics": {
                "checked": True,
                "classification": GROUND_OK,
            },
        }

        self.assertTrue(ground_reference_result_is_valid(result))


if __name__ == "__main__":
    unittest.main()
