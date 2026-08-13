from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_camera_viewport import CameraViewportManager, ViewportViewState  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402
from tests.controller_test_utils import FakeTransport, make_args  # noqa: E402


class FakeViewport:
    def __init__(self, name: str = "Main", camera_path: str = "") -> None:
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


class FakeUtility:
    def __init__(self, active: FakeViewport, *, secondary_supported: bool = False) -> None:
        self.active = active
        self.secondary_supported = secondary_supported

    def get_active_viewport(self) -> FakeViewport:
        return self.active


class CameraViewportRestoreTest(unittest.TestCase):
    def tearDown(self) -> None:
        self._restore_modules()

    def test_perspective_without_camera_prim_path_can_restore(self) -> None:
        active = FakeViewport(camera_path="")
        self._install_utility(FakeUtility(active))
        manager = CameraViewportManager()
        manager.previous_main_view = ViewportViewState(mode="perspective", camera_prim_path="", was_free_perspective=True)
        manager.main_viewport_api = active
        manager.main_viewport_was_perspective = True

        status = manager.return_main_view_to_perspective()

        self.assertTrue(status.completed)
        self.assertTrue(status.supported)
        self.assertEqual(status.error, "")
        self.assertEqual(active.calls, [""])

    def test_empty_previous_path_is_not_a_restore_failure(self) -> None:
        active = FakeViewport(camera_path="/World/WLRRobot/base_link/onboard_rgbd_camera")
        self._install_utility(FakeUtility(active))
        manager = CameraViewportManager()
        manager.previous_main_view = ViewportViewState(mode="perspective", camera_prim_path="", was_free_perspective=True)
        manager.main_viewport_api = active
        manager.main_viewport_was_perspective = True

        status = manager.restore_previous_view()

        self.assertTrue(status.completed)
        self.assertNotIn("no previous viewport camera path saved", status.error)
        self.assertEqual(active.camera_path, "")

    def test_active_viewport_fallback_can_return_to_perspective(self) -> None:
        active = FakeViewport(camera_path="")
        self._install_utility(FakeUtility(active))
        manager = CameraViewportManager()

        opened = manager.open_onboard_camera_viewport(
            "/World/WLRRobot/base_link/onboard_rgbd_camera",
            active_fallback_allowed=True,
        )
        restored = manager.return_main_view_to_perspective()

        self.assertEqual(opened.mode, "active_viewport_fallback")
        self.assertFalse(opened.main_view_unchanged)
        self.assertEqual(active.camera_path, "")
        self.assertTrue(restored.supported)
        self.assertEqual(active.calls, ["/World/WLRRobot/base_link/onboard_rgbd_camera", ""])

    def test_active_viewport_fallback_is_disabled_by_default(self) -> None:
        active = FakeViewport(camera_path="")
        self._install_utility(FakeUtility(active))
        manager = CameraViewportManager()

        opened = manager.open_onboard_camera_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")

        self.assertFalse(opened.supported)
        self.assertFalse(opened.active_fallback_used)
        self.assertTrue(opened.main_view_unchanged)
        self.assertEqual(active.camera_path, "")
        self.assertEqual(active.calls, [])

    def test_controller_waits_for_worker_ack_and_surfaces_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp)))
            transport = FakeTransport()
            controller.transport = transport  # type: ignore[assignment]
            controller.sim_ready = True
            controller.latest_sim_status = {
                "effective_headless": False,
                "vision": {
                    "camera_ready": True,
                    "camera_prim_path": "/World/WLRRobot/base_link/onboard_rgbd_camera",
                },
            }

            self.assertTrue(controller.open_onboard_camera_viewport())
            self.assertIn("waiting for worker ACK", controller.status_log[-1])
            self.assertFalse(any("action completed" in line for line in controller.status_log))
            payload = transport.calls[-1][1]
            revision = int(payload["action_revision"])

            controller._sync_camera_view_ack(
                {
                    "camera_view": {
                        "requested_action": "open_camera_viewport",
                        "completed_revision": revision,
                        "error": "create_viewport_window unavailable",
                    }
                }
            )

            self.assertIn("Camera viewport action failed", controller.status_log[-1])
            self.assertIn("create_viewport_window unavailable", controller.status_log[-1])

    def _install_utility(self, utility_obj: FakeUtility) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ["omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility"]}
        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        viewport_mod = types.ModuleType("omni.kit.viewport")
        utility = types.ModuleType("omni.kit.viewport.utility")
        utility.get_active_viewport = utility_obj.get_active_viewport  # type: ignore[attr-defined]
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
