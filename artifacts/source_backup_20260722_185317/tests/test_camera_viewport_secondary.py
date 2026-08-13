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
        self.camera_path = camera_path
        self.calls: list[str] = []
        self.closed = False

    def set_camera_prim_path(self, path: str | None) -> None:
        value = "" if path is None else str(path)
        self.calls.append(value)
        self.camera_path = value

    def close(self) -> None:
        self.closed = True


class FakeWindow:
    def __init__(self, viewport: FakeViewport) -> None:
        self.viewport_api = viewport
        self.closed = False

    def close(self) -> None:
        self.closed = True


class SecondaryViewportUtility:
    def __init__(self) -> None:
        self.main = FakeViewport("Main", "")
        self.secondary: FakeViewport | None = None
        self.window: FakeWindow | None = None
        self.create_count = 0

    def get_active_viewport(self) -> FakeViewport:
        return self.main

    def get_viewport_from_window_name(self, name: str) -> FakeViewport | None:
        if name == CAMERA_VIEWPORT_NAME:
            return self.secondary
        return None

    def create_viewport_window(self, name: str) -> FakeWindow:
        self.create_count += 1
        self.secondary = FakeViewport(name, "")
        self.window = FakeWindow(self.secondary)
        return self.window


class PendingViewportUtility(SecondaryViewportUtility):
    def __init__(self) -> None:
        super().__init__()
        self.window = None

    def create_viewport_window(self, name: str) -> object:
        self.create_count += 1
        self.window = object()
        return self.window


class CameraViewportSecondaryTest(unittest.TestCase):
    def tearDown(self) -> None:
        self._restore_modules()

    def test_open_prefers_second_viewport_and_keeps_main_perspective(self) -> None:
        utility = SecondaryViewportUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        status = manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")

        self.assertTrue(status.supported)
        self.assertTrue(status.active)
        self.assertEqual(status.mode, "secondary_viewport")
        self.assertTrue(status.main_view_unchanged)
        self.assertEqual(utility.main.camera_path, "")
        self.assertEqual(utility.secondary.camera_path, "/World/WLRRobot/base_link/onboard_rgbd_camera")
        self.assertEqual(utility.create_count, 1)

    def test_repeated_open_reuses_second_viewport(self) -> None:
        utility = SecondaryViewportUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")
        manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")

        self.assertEqual(utility.create_count, 1)
        self.assertEqual(utility.secondary.calls, ["/World/WLRRobot/base_link/onboard_rgbd_camera", "/World/WLRRobot/base_link/onboard_rgbd_camera"])

    def test_close_only_closes_project_camera_viewport(self) -> None:
        utility = SecondaryViewportUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")
        status = manager.close_camera_viewport()

        self.assertTrue(status.completed)
        self.assertTrue(status.supported)
        self.assertFalse(utility.main.closed)
        self.assertTrue(utility.window.closed)

    def test_close_does_not_close_reused_existing_onboard_camera_window(self) -> None:
        utility = SecondaryViewportUtility()
        utility.secondary = FakeViewport(CAMERA_VIEWPORT_NAME)
        self._install_utility(utility)
        manager = CameraViewportManager()

        manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")
        status = manager.close_camera_viewport()

        self.assertFalse(status.supported)
        self.assertFalse(utility.secondary.closed)

    def test_secondary_viewport_pending_does_not_fallback_to_active_by_default(self) -> None:
        utility = PendingViewportUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        status = manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")

        self.assertTrue(status.pending)
        self.assertFalse(status.completed)
        self.assertTrue(status.main_view_unchanged)
        self.assertFalse(status.active_fallback_used)
        self.assertEqual(utility.main.camera_path, "")

    def test_pending_retry_binds_when_viewport_api_becomes_ready(self) -> None:
        utility = PendingViewportUtility()
        self._install_utility(utility)
        manager = CameraViewportManager()

        pending = manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")
        utility.secondary = FakeViewport(CAMERA_VIEWPORT_NAME)
        completed = manager.service_pending_camera_viewport()

        self.assertTrue(pending.pending)
        self.assertTrue(completed.completed)
        self.assertTrue(completed.active)
        self.assertEqual(utility.secondary.camera_path, "/World/WLRRobot/base_link/onboard_rgbd_camera")
        self.assertEqual(utility.main.camera_path, "")

    def _install_utility(self, utility_obj: SecondaryViewportUtility) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ["omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility"]}
        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        viewport_mod = types.ModuleType("omni.kit.viewport")
        utility = types.ModuleType("omni.kit.viewport.utility")
        utility.get_active_viewport = utility_obj.get_active_viewport  # type: ignore[attr-defined]
        utility.get_viewport_from_window_name = utility_obj.get_viewport_from_window_name  # type: ignore[attr-defined]
        utility.create_viewport_window = utility_obj.create_viewport_window  # type: ignore[attr-defined]
        sys.modules["omni"] = omni
        sys.modules["omni.kit"] = kit
        sys.modules["omni.kit.viewport"] = viewport_mod
        sys.modules["omni.kit.viewport.utility"] = utility

    def _restore_modules(self) -> None:
        saved = getattr(self, "_saved_modules", None)
        if not saved:
            return
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self._saved_modules = None


if __name__ == "__main__":
    unittest.main()
