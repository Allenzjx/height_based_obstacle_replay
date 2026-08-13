from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_onboard_camera  # noqa: E402
from obstacle_height_vision import VisionHeightConfig  # noqa: E402
from sim_onboard_camera import FRAMING_GOOD, FRAMING_POINTS_BACKWARD, FRAMING_TOO_TOP_DOWN  # noqa: E402


class CameraGeometryDiagnosticsTest(unittest.TestCase):
    def test_camera_optical_axis_points_robot_forward(self) -> None:
        frame, points = self._frame_and_points(top_only=False)
        diagnostic = self._inspect_with_points(frame, points, obstacle_x_m=1.55)

        self.assertTrue(diagnostic.checked)
        self.assertGreater(diagnostic.center_ray_direction_w[0], 0.0)
        self.assertGreater(diagnostic.optical_forward_w[0], 0.0)
        self.assertIsNotNone(diagnostic.ground_intersection_w)

    def test_oblique_frame_is_good_when_ground_obstacle_and_top_are_visible(self) -> None:
        frame, points = self._frame_and_points(top_only=False)
        diagnostic = self._inspect_with_points(frame, points, obstacle_x_m=1.55)

        self.assertEqual(diagnostic.framing_state, FRAMING_GOOD)
        self.assertGreater(diagnostic.ground_point_fraction, 0.05)
        self.assertGreater(diagnostic.obstacle_point_fraction, 0.0)
        self.assertGreater(diagnostic.top_point_fraction, 0.0)
        self.assertLess(diagnostic.top_point_fraction, 0.85)

    def test_top_only_frame_warns_too_top_down(self) -> None:
        frame, points = self._frame_and_points(top_only=True)
        diagnostic = self._inspect_with_points(frame, points, obstacle_x_m=1.55)

        self.assertEqual(diagnostic.framing_state, FRAMING_TOO_TOP_DOWN)
        self.assertGreater(diagnostic.top_point_fraction, 0.85)

    def test_camera_pointing_backward_is_reported(self) -> None:
        frame, points = self._frame_and_points(top_only=False, backward=True)
        diagnostic = self._inspect_with_points(frame, points, obstacle_x_m=1.55)

        self.assertEqual(diagnostic.framing_state, FRAMING_POINTS_BACKWARD)

    def _inspect_with_points(self, frame: dict[str, Any], points: np.ndarray, *, obstacle_x_m: float) -> Any:
        original = sim_onboard_camera._transform_points
        sim_onboard_camera._transform_points = lambda *_args, **_kwargs: points  # type: ignore[assignment]
        try:
            return sim_onboard_camera.inspect_camera_geometry(
                SimpleNamespace(config=SimpleNamespace(ground_z_m=0.0)),
                frame,
                VisionHeightConfig(minimum_obstacle_points=5),
                roi_source="generated_scene_x_prior",
                obstacle_x_m=obstacle_x_m,
            )
        finally:
            sim_onboard_camera._transform_points = original  # type: ignore[assignment]

    def _frame_and_points(self, *, top_only: bool, backward: bool = False) -> tuple[dict[str, Any], np.ndarray]:
        height = 10
        width = 10
        pos = (0.0, 0.0, 0.35)
        target = (-1.0, 0.0, 0.0) if backward else (1.55, 0.0, 0.0)
        quat = sim_onboard_camera.camera_look_at_quat_wxyz(camera_position=pos, target_position=target, convention="ros")
        frame = {
            "depth": np.ones((height, width), dtype=float),
            "intrinsics": np.array([[100.0, 0.0, width / 2.0], [0.0, 100.0, height / 2.0], [0.0, 0.0, 1.0]], dtype=float),
            "pos_w": np.asarray(pos, dtype=float),
            "quat_w": np.asarray(quat, dtype=float),
        }
        points = np.zeros((height, width, 3), dtype=float)
        points[..., 0] = 1.55
        points[..., 1] = np.linspace(-0.25, 0.25, width, dtype=float)[None, :]
        if top_only:
            points[..., 2] = 0.10
        else:
            points[..., 2] = 0.0
            values = np.linspace(0.03, 0.10, 6, dtype=float)
            points[2:8, 2:8, 2] = values[:, None]
        return frame, points


if __name__ == "__main__":
    unittest.main()
