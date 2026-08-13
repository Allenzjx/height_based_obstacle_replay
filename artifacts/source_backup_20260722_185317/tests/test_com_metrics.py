from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from telemetry.com_metrics import body_com_positions_from_link_pose, compute_whole_body_com  # noqa: E402


class ComMetricsTest(unittest.TestCase):
    def test_mass_weighted_whole_body_com(self) -> None:
        result = compute_whole_body_com(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            np.asarray([1.0, 2.0, 1.0]),
        )
        self.assertAlmostEqual(result.total_mass_kg, 4.0)
        np.testing.assert_allclose(result.com_w, [0.5, 0.5, 0.0])
        np.testing.assert_allclose(result.contribution.sum(axis=0), result.com_w)

    def test_mass_count_mismatch_is_reported_without_crashing(self) -> None:
        result = compute_whole_body_com(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([2.0]),
        )
        self.assertIn("mass count does not match body count", result.warnings)
        np.testing.assert_allclose(result.com_w, [0.0, 0.0, 0.0])

    def test_body_com_positions_from_link_pose_uses_wxyz_quaternions(self) -> None:
        positions = np.asarray([[1.0, 2.0, 3.0]])
        quats = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        offsets = np.asarray([[0.1, 0.2, 0.3]])
        np.testing.assert_allclose(body_com_positions_from_link_pose(positions, quats, offsets), [[1.1, 2.2, 3.3]])


if __name__ == "__main__":
    unittest.main()
