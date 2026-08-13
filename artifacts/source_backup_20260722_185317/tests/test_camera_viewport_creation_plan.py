from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_camera_viewport import CameraViewportManager  # noqa: E402


class FakeViewport:
    def __init__(self) -> None:
        self.camera_path = ""

    def set_camera_prim_path(self, path: str) -> None:
        self.camera_path = str(path)


class FakeWindow:
    def __init__(self, viewport: FakeViewport) -> None:
        self.viewport_api = viewport
        self.visible = True


class FakeUtility:
    def __init__(self) -> None:
        self.main = FakeViewport()
        self.secondary = FakeViewport()
        self.create_kwargs: dict = {}

    def get_active_viewport(self) -> FakeViewport:
        return self.main

    def get_viewport_from_window_name(self, _name: str) -> None:
        return None

    def create_viewport_window(self, name: str = "", **kwargs: object) -> FakeWindow:
        self.create_kwargs = {"name": name, **kwargs}
        return FakeWindow(self.secondary)


class CameraViewportCreationPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {name: sys.modules.get(name) for name in ("omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility")}
        self.utility = FakeUtility()
        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        viewport_mod = types.ModuleType("omni.kit.viewport")
        utility_mod = types.ModuleType("omni.kit.viewport.utility")
        utility_mod.get_active_viewport = self.utility.get_active_viewport  # type: ignore[attr-defined]
        utility_mod.get_viewport_from_window_name = self.utility.get_viewport_from_window_name  # type: ignore[attr-defined]
        utility_mod.create_viewport_window = self.utility.create_viewport_window  # type: ignore[attr-defined]
        sys.modules["omni"] = omni
        sys.modules["omni.kit"] = kit
        sys.modules["omni.kit.viewport"] = viewport_mod
        sys.modules["omni.kit.viewport.utility"] = utility_mod

    def tearDown(self) -> None:
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_create_viewport_window_receives_camera_path(self) -> None:
        camera_path = "/World/WLRRobot/base_link/onboard_rgbd_camera"
        status = CameraViewportManager().open_onboard_camera_viewport(camera_path)

        self.assertTrue(status.supported)
        self.assertTrue(status.camera_path_verified)
        self.assertIn("camera_path", self.utility.create_kwargs)
        self.assertEqual(str(self.utility.create_kwargs["camera_path"]), camera_path)
        self.assertEqual(status.bound_camera_path, camera_path)


if __name__ == "__main__":
    unittest.main()
