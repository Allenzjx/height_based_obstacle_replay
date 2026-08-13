from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.stability_metrics import compute_geometric_stability, signed_distance_to_support_region  # noqa: E402


class StabilityMarginTest(unittest.TestCase):
    def test_com_inside_square_has_positive_margin(self) -> None:
        contacts = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
        result = compute_geometric_stability(contacts, [0.0, 0.0, 0.4])
        self.assertEqual(result.state, "safe")
        self.assertAlmostEqual(result.margin_m, 0.5)
        self.assertAlmostEqual(result.area_m2, 1.0)

    def test_com_outside_square_has_negative_margin(self) -> None:
        contacts = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
        result = compute_geometric_stability(contacts, [0.75, 0.0, 0.4])
        self.assertEqual(result.state, "outside")
        self.assertAlmostEqual(result.margin_m, -0.25)

    def test_degenerate_segment_support_reports_negative_off_segment_distance(self) -> None:
        margin, degenerate = signed_distance_to_support_region([0.0, 0.25], [[-0.5, 0.0], [0.5, 0.0]])
        self.assertTrue(degenerate)
        self.assertAlmostEqual(margin, -0.25)


if __name__ == "__main__":
    unittest.main()
