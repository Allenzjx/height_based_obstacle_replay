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
from obstacle_height_vision import HeightDetection, VisionHeightConfig, estimate_height_from_depth, estimate_height_from_pointcloud  # noqa: E402
from sim_onboard_camera import OnboardCameraProcessor, build_height_measurement_evidence, build_height_provenance  # noqa: E402


class FakeCamera:
    def __init__(self) -> None:
        self.depth = np.linspace(0.4, 1.0, 64, dtype=float).reshape(8, 8)
        self.intrinsics = np.array([[120.0, 0.0, 4.0], [0.0, 120.0, 4.0], [0.0, 0.0, 1.0]], dtype=float)
        self.pos = np.array([0.0, 0.0, 0.2], dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data = SimpleNamespace(
            output={"rgb": np.zeros((8, 8, 3), dtype=np.uint8), "distance_to_image_plane": self.depth},
            intrinsic_matrices=self.intrinsics,
            pos_w=self.pos,
            quat_w_ros=self.quat,
        )

    def update(self, _dt: float) -> None:
        return None


class HeightProvenanceTest(unittest.TestCase):
    def test_provenance_records_rgbd_geometry_inputs_and_rejects_forbidden_keys(self) -> None:
        scene = self._scene()
        frame = {
            "depth": scene.camera.depth,
            "intrinsics": scene.camera.intrinsics,
            "pos_w": scene.camera.pos,
            "quat_w": scene.camera.quat,
        }

        ok = build_height_provenance(
            scene_handle=scene,
            frame=frame,
            frame_revision=3,
            roi_source="generated_scene_x_prior",
            obstacle_x_m=1.55,
            detector_input_keys={"depth_image", "intrinsic_matrix", "camera_position_w", "camera_quat_wxyz", "obstacle_x_m"},
        )
        bad = build_height_provenance(
            scene_handle=scene,
            frame=frame,
            frame_revision=4,
            roi_source="generated_scene_x_prior",
            obstacle_x_m=1.55,
            detector_input_keys={"depth_image", "expected_height_cm"},
        )

        self.assertEqual(ok["height_source"], "isaac_rgbd_depth_geometry")
        self.assertTrue(ok["intrinsics_used"])
        self.assertTrue(ok["camera_world_pose_used"])
        self.assertTrue(ok["obstacle_x_prior_used"])
        self.assertFalse(ok["expected_height_used_by_detector"])
        self.assertFalse(ok["scene_obstacle_height_used_by_detector"])
        self.assertTrue(ok["detector_input_audit_passed"])
        self.assertFalse(bad["detector_input_audit_passed"])
        self.assertIn("expected_height_cm", bad["forbidden_input_reason"])

    def test_detector_receives_depth_intrinsics_pose_not_expected_or_scene_height(self) -> None:
        calls: list[dict[str, Any]] = []
        original = sim_onboard_camera.estimate_height_from_depth

        def spy(*args: Any, **kwargs: Any) -> HeightDetection:
            calls.append({"args": args, "kwargs": kwargs})
            return HeightDetection(True, 10.0, 10, 0.95, 100, 0.001, 0.0, "ok", 1.0)

        sim_onboard_camera.estimate_height_from_depth = spy  # type: ignore[assignment]
        try:
            processor = OnboardCameraProcessor(self._scene())
            processor.request_height_validation(5)
            processor.update(dt=0.1, sim_time=0.2, wall_time=1.0)
        finally:
            sim_onboard_camera.estimate_height_from_depth = original  # type: ignore[assignment]

        self.assertEqual(len(calls), 1)
        args = calls[0]["args"]
        kwargs = calls[0]["kwargs"]
        self.assertIs(args[0], processor.scene_handle.camera.depth)
        self.assertIs(args[1], processor.scene_handle.camera.intrinsics)
        self.assertIs(args[2], processor.scene_handle.camera.pos)
        self.assertIs(args[3], processor.scene_handle.camera.quat)
        self.assertIn("config", kwargs)
        self.assertIn("obstacle_x_m", kwargs)
        forbidden = {"expected_height_cm", "generated_height_cm", "scene_obstacle_height_m", "height_cm", "obstacle_size"}
        self.assertTrue(forbidden.isdisjoint(kwargs))
        self.assertTrue(processor.height_provenance["detector_input_audit_passed"])

    def test_depth_none_is_invalid_and_metadata_mismatch_follows_pointcloud_height(self) -> None:
        cfg = VisionHeightConfig(minimum_confidence=0.1, minimum_obstacle_points=20, quantization_tolerance_cm=2.0)
        missing = estimate_height_from_depth(None, np.eye(3), (0.0, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0), config=cfg)
        points = self._pointcloud(obstacle_height_m=0.10, obstacle_x_m=1.55)
        detection = estimate_height_from_pointcloud(
            points,
            valid_mask=np.ones(points.shape[:2], dtype=bool),
            config=cfg,
            obstacle_x_m=1.55,
            camera_position_w=(0.0, 0.0, 0.2),
        )

        self.assertFalse(missing.valid)
        self.assertEqual(detection.detected_height_cm, 10)
        self.assertIsNotNone(detection.raw_height_cm)
        self.assertAlmostEqual(float(detection.raw_height_cm), 10.0, delta=0.5)

    def test_height_measurement_evidence_contains_small_camera_audit_fields(self) -> None:
        scene = self._scene()
        frame = {
            "depth": scene.camera.depth,
            "intrinsics": scene.camera.intrinsics,
            "pos_w": scene.camera.pos,
            "quat_w": scene.camera.quat,
        }
        detection = HeightDetection(
            True,
            5.1,
            5,
            0.95,
            120,
            0.001,
            0.1,
            "ok",
            1.0,
            ground_z_m=0.0,
            top_z_m=0.051,
            top_point_count=30,
            obstacle_point_count=120,
            ground_point_count=50,
        )
        provenance = build_height_provenance(
            scene_handle=scene,
            frame=frame,
            frame_revision=7,
            roi_source="generated_scene_x_prior",
            obstacle_x_m=1.55,
            detector_input_keys={"depth_image", "intrinsic_matrix", "camera_position_w", "camera_quat_wxyz", "obstacle_x_m"},
        )

        evidence = build_height_measurement_evidence(
            scene_handle=scene,
            frame=frame,
            frame_revision=7,
            detection=detection,
            filtered_detection=detection,
            roi_source="generated_scene_x_prior",
            obstacle_x_m=1.55,
            height_provenance=provenance,
        )

        self.assertEqual(evidence["height_source"], "isaac_rgbd_depth_geometry")
        self.assertEqual(evidence["depth_shape"], [8, 8])
        self.assertTrue(evidence["depth_fingerprint"])
        self.assertTrue(evidence["intrinsics_fingerprint"])
        self.assertEqual(evidence["top_z_m"], 0.051)
        self.assertEqual(evidence["ground_z_m"], 0.0)
        self.assertEqual(evidence["top_point_count"], 30)
        self.assertFalse(evidence["expected_height_used_by_detector"])
        self.assertTrue(evidence["detector_input_audit_passed"])

    def _scene(self) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                onboard_camera_enabled=True,
                camera_update_period_s=0.0,
                camera_near_clip_m=0.05,
                camera_far_clip_m=6.0,
                obstacle_x=1.55,
                obstacle_height_m=0.05,
                ground_z_m=0.0,
                camera_coverage_strict=False,
            ),
            camera=FakeCamera(),
            camera_error="",
            camera_parent_prim="/World/WLRRobot/base_link",
            camera_prim_path="/World/WLRRobot/base_link/onboard_rgbd_camera",
            stage=None,
        )

    def _pointcloud(self, *, obstacle_height_m: float, obstacle_x_m: float) -> np.ndarray:
        points = np.zeros((12, 12, 3), dtype=float)
        points[..., 0] = obstacle_x_m
        points[..., 1] = np.linspace(-0.25, 0.25, 12, dtype=float)[None, :]
        points[3:10, :, 2] = obstacle_height_m
        return points


if __name__ == "__main__":
    unittest.main()
