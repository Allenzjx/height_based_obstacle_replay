from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_onboard_camera import inspect_camera_health  # noqa: E402


class FakePrim:
    def __init__(self, *, valid: bool = True, type_name: str = "Xform", rigid: bool = False):
        self.valid = valid
        self.type_name = type_name
        self.has_rigid_body_api = rigid

    def IsValid(self) -> bool:
        return self.valid

    def GetTypeName(self) -> str:
        return self.type_name

    def HasAPI(self, _api: object) -> bool:
        return self.has_rigid_body_api


class FakeStage:
    def __init__(self, prims: dict[str, FakePrim]):
        self.prims = prims

    def GetPrimAtPath(self, path: str) -> FakePrim:
        return self.prims.get(path, FakePrim(valid=False))


class FakeData:
    def __init__(self, *, rgb: object | None = None, depth: object | None = None):
        output = {}
        if rgb is not None:
            output["rgb"] = rgb
        if depth is not None:
            output["distance_to_image_plane"] = depth
        self.output = output
        self.intrinsic_matrices = np.eye(3, dtype=float)
        self.intrinsic_matrices[0, 0] = 220.0
        self.intrinsic_matrices[1, 1] = 220.0
        self.pos_w = np.array([0.0, 0.0, 0.2], dtype=float)
        self.quat_w_ros = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


class Camera:
    def __init__(self, data: FakeData):
        self.data = data


Camera.__module__ = "isaaclab.sensors.camera.camera"


class FakeOtherCamera(Camera):
    pass


FakeOtherCamera.__module__ = "not_isaac.camera"


class CameraHealthTest(unittest.TestCase):
    def test_camera_class_module_and_valid_prims_pass(self) -> None:
        scene = self._scene()
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertTrue(health.ok, health)
        self.assertEqual(health.backend_module, "isaaclab.sensors.camera.camera")
        self.assertEqual(health.backend_class, "Camera")
        self.assertTrue(health.is_isaaclab_camera)

    def test_invalid_camera_prim_fails(self) -> None:
        scene = self._scene(camera_prim=FakePrim(valid=False, type_name="Camera"))
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertFalse(health.ok)
        self.assertFalse(health.camera_prim_valid)

    def test_parent_without_rigid_body_fails(self) -> None:
        scene = self._scene(parent_prim=FakePrim(valid=True, type_name="Xform", rigid=False))
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertFalse(health.ok)
        self.assertFalse(health.parent_has_rigid_body_api)

    def test_rgb_missing_fails(self) -> None:
        scene = self._scene(data=FakeData(rgb=None, depth=self._depth()))
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertFalse(health.ok)
        self.assertFalse(health.rgb_available)

    def test_depth_missing_fails(self) -> None:
        scene = self._scene(data=FakeData(rgb=self._rgb(), depth=None))
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertFalse(health.ok)
        self.assertFalse(health.depth_available)

    def test_nan_depth_fails(self) -> None:
        scene = self._scene(data=FakeData(rgb=self._rgb(), depth=np.full((8, 8), np.nan)))
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=2)
        self.assertFalse(health.ok)
        self.assertEqual(health.depth_finite_ratio, 0.0)

    def test_frame_revision_not_advanced_is_stale_fail(self) -> None:
        scene = self._scene()
        health = inspect_camera_health(scene, self._processor(), frames_advanced_during_check=0)
        self.assertFalse(health.ok)
        self.assertTrue(health.stale_frame)

    def _scene(
        self,
        *,
        data: FakeData | None = None,
        camera_prim: FakePrim | None = None,
        parent_prim: FakePrim | None = None,
    ) -> SimpleNamespace:
        camera_path = "/World/WLRRobot/base_link/onboard_rgbd_camera"
        parent_path = "/World/WLRRobot/base_link"
        stage = FakeStage(
            {
                camera_path: camera_prim or FakePrim(valid=True, type_name="Camera"),
                parent_path: parent_prim or FakePrim(valid=True, type_name="Xform", rigid=True),
            }
        )
        camera = Camera(data or FakeData(rgb=self._rgb(), depth=self._depth()))
        return SimpleNamespace(camera=camera, camera_prim_path=camera_path, camera_parent_prim=parent_path, stage=stage)

    def _processor(self) -> SimpleNamespace:
        return SimpleNamespace(frame_revision=3, last_frame_at=time.time())

    def _rgb(self) -> np.ndarray:
        return np.full((8, 8, 3), 127, dtype=np.uint8)

    def _depth(self) -> np.ndarray:
        values = np.linspace(0.5, 1.5, 64, dtype=float)
        return values.reshape(8, 8)


if __name__ == "__main__":
    unittest.main()
