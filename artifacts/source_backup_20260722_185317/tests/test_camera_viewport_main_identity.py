from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_camera_viewport import CAMERA_VIEWPORT_NAME, CameraViewportManager  # noqa: E402


class FakeViewport:
    def __init__(self, name: str, camera_path: str = "") -> None:
        self.name = name
        self.id = name.lower()
        self.camera_path = camera_path
        self.calls: list[str] = []

    def set_camera_prim_path(self, path: str | None) -> None:
        value = "" if path is None else str(path)
        self.calls.append(value)
        self.camera_path = value


class FakeWindow:
    def __init__(self, viewport: FakeViewport) -> None:
        self.viewport_api = viewport


class ActiveSwitchingUtility:
    def __init__(self) -> None:
        self.main = FakeViewport("Main", "")
        self.secondary: FakeViewport | None = None
        self.active = self.main

    def get_active_viewport(self) -> FakeViewport:
        return self.active

    def get_viewport_from_window_name(self, name: str) -> FakeViewport | None:
        if name == CAMERA_VIEWPORT_NAME:
            return self.secondary
        return None

    def create_viewport_window(self, name: str) -> FakeWindow:
        self.secondary = FakeViewport(name, "")
        self.active = self.secondary
        return FakeWindow(self.secondary)


class CameraViewportMainIdentityTest(unittest.TestCase):
    def tearDown(self) -> None:
        self._restore_modules()

    def test_return_uses_saved_main_viewport_after_secondary_becomes_active(self) -> None:
        utility = ActiveSwitchingUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        opened = manager.open_onboard_camera_viewport("/World/WLRRobot/base/onboard_rgbd_camera")
        returned = manager.return_main_view_to_perspective()

        self.assertTrue(opened.supported)
        self.assertEqual(manager.main_viewport_api, utility.main)
        self.assertEqual(utility.active, utility.secondary)
        self.assertEqual(utility.main.calls, [""])
        self.assertEqual(utility.secondary.camera_path, "/World/WLRRobot/base/onboard_rgbd_camera")
        self.assertTrue(returned.perspective_restore_verified)
        self.assertEqual(returned.main_viewport_name, "Main")

    def _install_utility(self, utility_obj: ActiveSwitchingUtility) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ["omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility"]}
        sys.modules["omni"] = types.ModuleType("omni")
        sys.modules["omni.kit"] = types.ModuleType("omni.kit")
        sys.modules["omni.kit.viewport"] = types.ModuleType("omni.kit.viewport")
        utility = types.ModuleType("omni.kit.viewport.utility")
        utility.get_active_viewport = utility_obj.get_active_viewport  # type: ignore[attr-defined]
        utility.get_viewport_from_window_name = utility_obj.get_viewport_from_window_name  # type: ignore[attr-defined]
        utility.create_viewport_window = utility_obj.create_viewport_window  # type: ignore[attr-defined]
        sys.modules["omni.kit.viewport.utility"] = utility

    def _restore_modules(self) -> None:
        for name, module in getattr(self, "_saved_modules", {}).items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
