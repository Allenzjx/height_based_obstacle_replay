from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from isaac_launch_preflight import (  # noqa: E402
    IsaacInterpreterReport,
    detect_eula_prompt,
    format_preflight_error,
    validate_isaac_python_compatibility,
)


class IsaacLaunchPreflightTest(unittest.TestCase):
    def test_isaac5_requires_python311(self) -> None:
        ok, reason = validate_isaac_python_compatibility("3.10.13", "5.0.0")
        self.assertFalse(ok)
        self.assertIn("Python 3.11", reason)

    def test_isaac4_requires_python310(self) -> None:
        ok, reason = validate_isaac_python_compatibility("3.11.8", "4.5.0")
        self.assertFalse(ok)
        self.assertIn("Python 3.10", reason)

    def test_missing_isaacsim_error_is_actionable(self) -> None:
        report = IsaacInterpreterReport(
            executable="python.exe",
            python_version="3.13.0",
            isaacsim_importable=False,
            isaaclab_importable=False,
            app_launcher_importable=False,
            error="No module named 'isaacsim'",
            error_category="missing_isaacsim",
        )
        self.assertIn("cannot import isaacsim", format_preflight_error(report))

    def test_eula_prompt_detection(self) -> None:
        self.assertTrue(detect_eula_prompt("Do you accept the EULA? [y/N]"))
        self.assertFalse(detect_eula_prompt("ordinary startup log"))


if __name__ == "__main__":
    unittest.main()
