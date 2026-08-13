from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_camera_viewport import CameraViewportManager  # noqa: E402


class IgnoringViewport:
    camera_path = "/OmniverseKit/Perspective"

    def set_camera_prim_path(self, _path: str) -> None:
        self.camera_path = "/OmniverseKit/Perspective"


class Utility:
    def __init__(self) -> None:
        self.main = IgnoringViewport()
        self.secondary = IgnoringViewport()

    def get_active_viewport(self) -> IgnoringViewport:
        return self.main

    def get_viewport_from_window_name(self, _name: str) -> IgnoringViewport | None:
        return self.secondary

    def create_viewport_window(self, **_kwargs: object) -> object:
        return object()


class CameraViewportCameraPathVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {name: sys.modules.get(name) for name in ("omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility")}
        self.utility = Utility()
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

    def test_camera_path_postcondition_failure_is_reported(self) -> None:
        status = CameraViewportManager().open_onboard_camera_viewport("/World/Camera")

        self.assertFalse(status.supported)
        self.assertFalse(status.camera_path_verified)
        self.assertIn("postcondition", status.error)
        self.assertIn("/OmniverseKit/Perspective", status.bound_camera_path)


if __name__ == "__main__":
    unittest.main()
