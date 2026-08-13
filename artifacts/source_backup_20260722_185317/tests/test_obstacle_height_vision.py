from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from obstacle_height_vision import (  # noqa: E402
    HeightDetection,
    TemporalHeightFilter,
    VisionHeightConfig,
    estimate_height_from_pointcloud,
    quantize_supported_height,
)


class ObstacleHeightVisionTest(unittest.TestCase):
    def test_quantizes_supported_heights_with_noise(self) -> None:
        for height in range(0, 41, 5):
            for noise in (-1.5, 0.0, 1.5):
                detected, error = quantize_supported_height(height + noise, tolerance_cm=2.0)
                self.assertEqual(detected, height)
                self.assertLessEqual(error or 0.0, 2.0)

    def test_quantization_rejects_outside_tolerance_and_boundary(self) -> None:
        self.assertEqual(quantize_supported_height(2.49, tolerance_cm=2.0)[0], None)
        self.assertEqual(quantize_supported_height(2.50, tolerance_cm=2.0)[0], None)
        self.assertEqual(quantize_supported_height(2.00, tolerance_cm=2.0)[0], 0)
        self.assertEqual(quantize_supported_height(3.00, tolerance_cm=2.0)[0], 5)
        self.assertEqual(quantize_supported_height(42.6, tolerance_cm=2.0)[0], None)

    def test_estimates_height_from_world_pointcloud(self) -> None:
        points = self._flat_ground_points()
        points[18:34, 28:52, 2] = 0.20
        detection = estimate_height_from_pointcloud(
            points,
            valid_mask=np.ones(points.shape[:2], dtype=bool),
            obstacle_x_m=1.55,
            config=VisionHeightConfig(minimum_obstacle_points=20),
            timestamp=1.0,
        )
        self.assertTrue(detection.valid, detection)
        self.assertEqual(detection.detected_height_cm, 20)
        self.assertAlmostEqual(detection.raw_height_cm or 0.0, 20.0, delta=0.3)

    def test_temporal_filter_requires_stable_frames_and_latches_revision(self) -> None:
        config = VisionHeightConfig(stable_frames_required=3, temporal_window_size=5, minimum_confidence=0.75)
        filt = TemporalHeightFilter(config)
        detections = [self._detection(10, revision_time=index) for index in range(2)]
        for detection in detections:
            self.assertFalse(filt.update(detection).stable)
        stable = filt.update(self._detection(10, revision_time=3))
        self.assertTrue(stable.stable)
        self.assertEqual(stable.detected_height_cm, 10)
        self.assertEqual(stable.detection_revision, 1)
        again = filt.update(self._detection(10, revision_time=4))
        self.assertTrue(again.stable)
        self.assertEqual(again.detection_revision, 1)

    def test_temporal_filter_rejects_height_jitter_and_increments_for_new_height(self) -> None:
        config = VisionHeightConfig(stable_frames_required=3, temporal_window_size=5, minimum_confidence=0.75)
        filt = TemporalHeightFilter(config)
        for index, height in enumerate([10, 15, 10, 15, 10]):
            result = filt.update(self._detection(height, revision_time=index))
        self.assertFalse(result.stable)
        for index in range(3):
            result = filt.update(self._detection(20, revision_time=10 + index))
        self.assertTrue(result.stable)
        self.assertEqual(result.detected_height_cm, 20)
        self.assertEqual(result.detection_revision, 1)

    def test_zero_cm_requires_temporal_stability_and_reset_can_emit_again(self) -> None:
        config = VisionHeightConfig(stable_frames_required=3, temporal_window_size=5, minimum_confidence=0.75)
        filt = TemporalHeightFilter(config)
        zero = HeightDetection(True, 0.0, 0, 0.85, 0, 0.0, 0.0, "ground", 1.0)
        self.assertFalse(filt.update(zero).stable)
        self.assertFalse(filt.update(zero).stable)
        stable = filt.update(zero)
        self.assertTrue(stable.stable)
        self.assertEqual(stable.detected_height_cm, 0)
        self.assertEqual(stable.detection_revision, 1)
        filt.reset()
        for _index in range(3):
            stable = filt.update(zero)
        self.assertTrue(stable.stable)
        self.assertEqual(stable.detection_revision, 2)

    def test_ground_only_pointcloud_returns_zero_candidate(self) -> None:
        points = self._flat_ground_points()
        detection = estimate_height_from_pointcloud(
            points,
            valid_mask=np.ones(points.shape[:2], dtype=bool),
            obstacle_x_m=1.55,
            config=VisionHeightConfig(minimum_obstacle_points=20),
            timestamp=1.0,
        )
        self.assertTrue(detection.valid)
        self.assertEqual(detection.detected_height_cm, 0)

    def _detection(self, height_cm: int, *, revision_time: float) -> HeightDetection:
        return HeightDetection(
            valid=True,
            raw_height_cm=float(height_cm),
            detected_height_cm=int(height_cm),
            confidence=0.9,
            point_count=200,
            top_plane_mad_m=0.002,
            quantization_error_cm=0.0,
            reason="ok",
            timestamp=float(revision_time),
        )

    def _flat_ground_points(self) -> np.ndarray:
        height, width = 48, 80
        xs = np.full((height, width), 1.55, dtype=float)
        ys = np.tile(np.linspace(-0.45, 0.45, width), (height, 1))
        zs = np.zeros((height, width), dtype=float)
        return np.stack((xs, ys, zs), axis=-1)


if __name__ == "__main__":
    unittest.main()
