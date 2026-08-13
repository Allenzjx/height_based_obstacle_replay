from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from obstacle_height_vision import VisionHeightConfig, estimate_height_from_pointcloud  # noqa: E402
from sim_onboard_camera import OnboardCameraProcessor  # noqa: E402


class VisionRoiSourceTest(unittest.TestCase):
    def test_generated_mode_uses_obstacle_x_prior(self) -> None:
        processor = OnboardCameraProcessor(self._scene())
        processor.set_source_mode("generated")

        self.assertEqual(processor.roi_source, "generated_scene_x_prior")
        self.assertEqual(processor._current_roi_obstacle_x(), 1.55)

    def test_external_mode_uses_camera_forward_auto_roi(self) -> None:
        processor = OnboardCameraProcessor(self._scene())
        processor.set_source_mode("external")

        self.assertEqual(processor.roi_source, "camera_forward_auto")
        self.assertIsNone(processor._current_roi_obstacle_x())

    def test_external_auto_roi_detects_moved_forward_obstacle(self) -> None:
        cfg = VisionHeightConfig(minimum_confidence=0.1, minimum_obstacle_points=20, quantization_tolerance_cm=2.0)
        points = self._pointcloud(obstacle_x_m=2.30, obstacle_height_m=0.10)
        valid = np.ones(points.shape[:2], dtype=bool)

        generated_prior = estimate_height_from_pointcloud(
            points,
            valid_mask=valid,
            config=cfg,
            obstacle_x_m=1.55,
            camera_position_w=(0.0, 0.0, 0.2),
        )
        external_auto = estimate_height_from_pointcloud(
            points,
            valid_mask=valid,
            config=cfg,
            obstacle_x_m=None,
            camera_position_w=(0.0, 0.0, 0.2),
        )

        self.assertNotEqual(generated_prior.detected_height_cm, 10)
        self.assertEqual(external_auto.detected_height_cm, 10)
        self.assertTrue(external_auto.valid)

    def _scene(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(onboard_camera_enabled=True, obstacle_x=1.55),
            camera=None,
            camera_error="",
        )

    def _pointcloud(self, *, obstacle_x_m: float, obstacle_height_m: float) -> np.ndarray:
        points = np.zeros((12, 12, 3), dtype=float)
        points[..., 0] = obstacle_x_m
        points[..., 1] = np.linspace(-0.25, 0.25, 12, dtype=float)[None, :]
        points[3:10, :, 2] = obstacle_height_m
        return points


if __name__ == "__main__":
    unittest.main()
