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
    def __init__(self, name: str, camera_path: str, *, ignore_set: bool = False) -> None:
        self.name = name
        self.id = name
        self.camera_path = camera_path
        self.ignore_set = ignore_set

    def set_camera_prim_path(self, path: str | None) -> None:
        if not self.ignore_set:
            self.camera_path = "" if path is None else str(path)


class FakeWindow:
    def __init__(self, viewport: FakeViewport) -> None:
        self.viewport_api = viewport


class Utility:
    def __init__(self, main: FakeViewport) -> None:
        self.main = main
        self.secondary: FakeViewport | None = None

    def get_active_viewport(self) -> FakeViewport:
        return self.main

    def get_viewport_from_window_name(self, name: str) -> FakeViewport | None:
        return self.secondary if name == CAMERA_VIEWPORT_NAME else None

    def create_viewport_window(self, name: str) -> FakeWindow:
        self.secondary = FakeViewport(name, "")
        return FakeWindow(self.secondary)


class CameraViewportPerspectiveRestoreTest(unittest.TestCase):
    def tearDown(self) -> None:
        self._restore_modules()

    def test_saved_perspective_camera_path_is_restored_and_verified(self) -> None:
        main = FakeViewport("Main", "/OmniverseKit_Persp")
        self._install_utility(Utility(main))
        manager = CameraViewportManager()

        manager.open_onboard_camera_viewport("/World/WLRRobot/base/onboard_rgbd_camera")
        status = manager.return_main_view_to_perspective()

        self.assertTrue(status.supported)
        self.assertTrue(status.perspective_restore_verified)
        self.assertEqual(status.perspective_restore_method, "restore_saved_main_camera_path")
        self.assertEqual(status.main_camera_path_after, "/OmniverseKit_Persp")

    def test_restore_reports_postcondition_error_if_main_remains_on_onboard_camera(self) -> None:
        main = FakeViewport("Main", "/World/WLRRobot/base/onboard_rgbd_camera", ignore_set=True)
        self._install_utility(Utility(main))
        manager = CameraViewportManager()
        manager.main_viewport_api = main
        manager.main_viewport_camera_path = ""
        manager.main_viewport_was_perspective = True

        status = manager.return_main_view_to_perspective()

        self.assertFalse(status.supported)
        self.assertFalse(status.perspective_restore_verified)
        self.assertIn("postcondition", status.postcondition_error)

    def _install_utility(self, utility_obj: Utility) -> None:
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
