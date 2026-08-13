from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_camera_viewport import CameraViewportStatus, _viewport_api_diagnostics, _window_docked, _window_visible  # noqa: E402


class FakeUtility:
    __name__ = "fake.viewport.utility"
    __file__ = "fake_viewport_utility.py"

    def create_viewport_window(self, window_name: str, width: int = 800, height: int = 450, camera_path: object | None = None) -> object:
        return object()

    def get_viewport_from_window_name(self, name: str) -> object | None:
        return None


class FakeWindow:
    visible = True
    docked = False


class CameraViewportRuntimeDiagnosticsTest(unittest.TestCase):
    def test_api_diagnostics_records_signatures(self) -> None:
        diagnostics = _viewport_api_diagnostics(FakeUtility())

        self.assertTrue(diagnostics["available"])
        self.assertEqual(diagnostics["module_file"], "fake_viewport_utility.py")
        self.assertIn("camera_path", diagnostics["functions"]["create_viewport_window"]["signature"])

    def test_status_has_visibility_fields(self) -> None:
        status = CameraViewportStatus(window_visible=True, window_docked=False, api_diagnostics={"available": True}).to_dict()

        self.assertTrue(status["window_visible"])
        self.assertFalse(status["window_docked"])
        self.assertTrue(status["api_diagnostics"]["available"])

    def test_window_visible_and_docked_helpers(self) -> None:
        self.assertTrue(_window_visible(FakeWindow()))
        self.assertFalse(_window_docked(FakeWindow()))


if __name__ == "__main__":
    unittest.main()
