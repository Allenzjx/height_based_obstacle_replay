from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.stability_metrics import compute_equilibrium_region  # noqa: E402


class EquilibriumRegionTest(unittest.TestCase):
    def test_square_contacts_make_center_com_feasible(self) -> None:
        contacts = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
        normals = [[0.0, 0.0, 1.0]] * 4
        result = compute_equilibrium_region(
            contacts,
            normals,
            [1.0, 1.0, 1.0, 1.0],
            mass_kg=10.0,
            com_w=[0.0, 0.0, 0.5],
            direction_samples=16,
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["current_state_feasible"])
        self.assertAlmostEqual(result["equilibrium_stability_margin_m"], 0.5)

    def test_no_contacts_reports_failure_status(self) -> None:
        result = compute_equilibrium_region([], [], [], mass_kg=10.0, com_w=[0.0, 0.0, 0.5])
        self.assertEqual(result["status"], "no_contacts")
        self.assertFalse(result["current_state_feasible"])


if __name__ == "__main__":
    unittest.main()
