from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_camera_viewport  # noqa: E402
from sim_camera_viewport import restore_previous_isaac_viewport, show_camera_in_isaac_viewport  # noqa: E402
from sim_onboard_camera import OnboardCameraProcessor  # noqa: E402
from sim_transport import SimTransport  # noqa: E402


class FakeViewport:
    def __init__(self) -> None:
        self.name = "Viewport"
        self.camera_path = "/OmniverseKit/Perspective"
        self.calls: list[str] = []

    def set_camera_prim_path(self, path: str) -> None:
        self.calls.append(path)
        self.camera_path = path


class CameraViewportControlTest(unittest.TestCase):
    def tearDown(self) -> None:
        self._restore_modules()

    def test_show_and_restore_use_lazy_viewport_utility(self) -> None:
        viewport = FakeViewport()
        self._install_fake_viewport_module(viewport)

        shown = show_camera_in_isaac_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera", active_fallback_allowed=True)
        restored = restore_previous_isaac_viewport()

        self.assertTrue(shown.supported)
        self.assertTrue(shown.active)
        self.assertEqual(shown.previous_camera_prim_path, "/OmniverseKit/Perspective")
        self.assertEqual(restored.camera_prim_path, "/OmniverseKit/Perspective")
        self.assertEqual(viewport.calls, ["/World/WLRRobot/base_link/onboard_rgbd_camera", "/OmniverseKit/Perspective"])

    def test_missing_viewport_api_is_fail_soft(self) -> None:
        self._install_fake_viewport_module(None)
        status = show_camera_in_isaac_viewport("/World/WLRRobot/base_link/onboard_rgbd_camera")
        self.assertTrue(status.requested)
        self.assertFalse(status.active)
        self.assertFalse(status.supported)
        self.assertIn("unavailable", status.error)

    def test_headless_worker_action_returns_unsupported_without_disabling_detector(self) -> None:
        scene = SimpleNamespace(
            config=SimpleNamespace(onboard_camera_enabled=True),
            camera_prim_path="/World/WLRRobot/base_link/onboard_rgbd_camera",
        )
        processor = OnboardCameraProcessor(scene)
        ok, message = processor.handle_control("show_camera_view", headless=True)
        self.assertFalse(ok)
        self.assertIn("headless", message)
        self.assertTrue(processor.enabled)
        self.assertTrue(processor.camera_view_status["requested"])
        self.assertFalse(processor.camera_view_status["supported"])

    def test_transport_camera_view_action_does_not_send_image_payloads(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, Any]]] = []

            def vision_control(self, action: str, **payload: Any) -> None:
                self.calls.append((action, dict(payload)))

        client = Client()
        transport = SimTransport()
        transport.attach_process_client(client)
        transport.show_camera_view(camera_prim_path="/World/WLRRobot/base_link/onboard_rgbd_camera")
        transport.restore_camera_view()

        self.assertEqual(client.calls[0][0], "open_camera_viewport")
        self.assertEqual(client.calls[1][0], "restore_camera_view")
        forbidden = {"rgb", "depth", "image", "png", "base64", "point_cloud"}
        for _action, payload in client.calls:
            self.assertTrue(forbidden.isdisjoint(payload))

    def _install_fake_viewport_module(self, viewport: FakeViewport | None) -> None:
        self._saved_modules = {name: sys.modules.get(name) for name in ["omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility"]}
        omni = types.ModuleType("omni")
        kit = types.ModuleType("omni.kit")
        viewport_mod = types.ModuleType("omni.kit.viewport")
        utility = types.ModuleType("omni.kit.viewport.utility")
        utility.get_active_viewport = lambda: viewport  # type: ignore[attr-defined]
        sys.modules["omni"] = omni
        sys.modules["omni.kit"] = kit
        sys.modules["omni.kit.viewport"] = viewport_mod
        sys.modules["omni.kit.viewport.utility"] = utility
        sim_camera_viewport._PREVIOUS_CAMERA_PRIM_PATH = ""  # type: ignore[attr-defined]
        sim_camera_viewport._LAST_VIEWPORT_NAME = ""  # type: ignore[attr-defined]

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
