from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_process_client import build_windows_batch_launcher, quote_windows_cmd_arg  # noqa: E402


class WindowsBatchLauncherTest(unittest.TestCase):
    def test_cmd_quote_wraps_spaces_parentheses_chinese_and_ampersand(self) -> None:
        value = r"C:\tmp\space dir\(case)\中文 & more\file.py"
        quoted = quote_windows_cmd_arg(value)
        self.assertTrue(quoted.startswith('"') and quoted.endswith('"'))
        self.assertIn("&", quoted)
        self.assertIn("中文", quoted)

    def test_wrapper_uses_minimal_config_file_arguments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="space (中文) & ") as tmp:
            root = Path(tmp)
            bat = root / "isaac lab & launcher.bat"
            script = root / "sim worker (中文).py"
            config = root / "worker config & data.json"
            bat.write_text("@echo off\r\n", encoding="utf-8")
            script.write_text("print('worker')\n", encoding="utf-8")
            config.write_text(json.dumps({"args": {"height_cm": 10}}), encoding="utf-8")

            plan = build_windows_batch_launcher(
                isaaclab_bat=bat,
                worker_script=script,
                config_path=config,
                cwd=root,
                env={},
            )
            self.assertNotIn("", plan.argv)
            self.assertIn("cmd.exe", plan.argv[0].lower())
            wrapper_text = Path(plan.wrapper_path).read_text(encoding="utf-8")
            self.assertIn("-p", wrapper_text)
            self.assertIn("--worker-config-file", wrapper_text)
            self.assertIn(quote_windows_cmd_arg(str(bat)), wrapper_text)
            self.assertIn(quote_windows_cmd_arg(str(script)), wrapper_text)
            self.assertIn(quote_windows_cmd_arg(str(config)), wrapper_text)


if __name__ == "__main__":
    unittest.main()
