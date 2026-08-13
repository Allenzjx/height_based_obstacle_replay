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
from obstacle_height_vision import HeightDetection  # noqa: E402
from sim_onboard_camera import FRAMING_TOO_TOP_DOWN, CameraGeometryDiagnostics, OnboardCameraProcessor  # noqa: E402


class FakeCamera:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            output={
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "distance_to_image_plane": np.ones((8, 8), dtype=float),
            },
            intrinsic_matrices=np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]], dtype=float),
            pos_w=np.array([0.0, 0.0, 0.3], dtype=float),
            quat_w_ros=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    def update(self, _dt: float) -> None:
        return None


class CameraCoverageTest(unittest.TestCase):
    def test_coverage_warning_does_not_block_detector_by_default(self) -> None:
        scene = self._scene(strict=False)
        processor = OnboardCameraProcessor(scene)
        calls: list[str] = []
        original_geometry = sim_onboard_camera.inspect_camera_geometry
        original_estimate = sim_onboard_camera.estimate_height_from_depth

        def fake_geometry(*_args: Any, **_kwargs: Any) -> CameraGeometryDiagnostics:
            return CameraGeometryDiagnostics(checked=True, framing_state=FRAMING_TOO_TOP_DOWN, reasons=["top-only test"])

        def fake_estimate(*_args: Any, **_kwargs: Any) -> HeightDetection:
            calls.append("estimate")
            return HeightDetection(True, 5.0, 5, 0.95, 100, 0.001, 0.0, "ok", 1.0)

        sim_onboard_camera.inspect_camera_geometry = fake_geometry  # type: ignore[assignment]
        sim_onboard_camera.estimate_height_from_depth = fake_estimate  # type: ignore[assignment]
        try:
            processor.update(dt=0.1, sim_time=0.2, wall_time=1.0)
        finally:
            sim_onboard_camera.inspect_camera_geometry = original_geometry  # type: ignore[assignment]
            sim_onboard_camera.estimate_height_from_depth = original_estimate  # type: ignore[assignment]

        self.assertEqual(calls, ["estimate"])
        self.assertTrue(processor.last_detection.valid)
        self.assertEqual(processor.camera_geometry_diagnostics["framing_state"], FRAMING_TOO_TOP_DOWN)

    def test_coverage_strict_marks_bad_framing_invalid(self) -> None:
        scene = self._scene(strict=True)
        processor = OnboardCameraProcessor(scene)
        calls: list[str] = []
        original_geometry = sim_onboard_camera.inspect_camera_geometry
        original_estimate = sim_onboard_camera.estimate_height_from_depth

        def fake_geometry(*_args: Any, **_kwargs: Any) -> CameraGeometryDiagnostics:
            return CameraGeometryDiagnostics(checked=True, framing_state=FRAMING_TOO_TOP_DOWN, reasons=["top-only test"])

        def fake_estimate(*_args: Any, **_kwargs: Any) -> HeightDetection:
            calls.append("estimate")
            return HeightDetection(True, 5.0, 5, 0.95, 100, 0.001, 0.0, "ok", 1.0)

        sim_onboard_camera.inspect_camera_geometry = fake_geometry  # type: ignore[assignment]
        sim_onboard_camera.estimate_height_from_depth = fake_estimate  # type: ignore[assignment]
        try:
            processor.update(dt=0.1, sim_time=0.2, wall_time=1.0)
        finally:
            sim_onboard_camera.inspect_camera_geometry = original_geometry  # type: ignore[assignment]
            sim_onboard_camera.estimate_height_from_depth = original_estimate  # type: ignore[assignment]

        self.assertEqual(calls, [])
        self.assertFalse(processor.last_detection.valid)
        self.assertIn("camera coverage strict", processor.last_detection.reason)

    def _scene(self, *, strict: bool) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                onboard_camera_enabled=True,
                camera_update_period_s=0.0,
                camera_near_clip_m=0.05,
                camera_far_clip_m=6.0,
                obstacle_x=1.55,
                camera_coverage_strict=strict,
            ),
            camera=FakeCamera(),
            camera_error="",
            camera_parent_prim="/World/WLRRobot/base_link",
            camera_prim_path="/World/WLRRobot/base_link/onboard_rgbd_camera",
            stage=None,
        )


if __name__ == "__main__":
    unittest.main()
