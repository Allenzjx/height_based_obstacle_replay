from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_process_client import SimProcessClient  # noqa: E402
from worker_startup_diagnostics import classify_startup_error, diagnose_startup_phase, last_meaningful_line  # noqa: E402


class WorkerStartupDiagnosticsTest(unittest.TestCase):
    def test_error_classification(self) -> None:
        self.assertEqual(classify_startup_error('"" was unexpected at this time.'), "batch_parse_error")
        self.assertEqual(classify_startup_error("ModuleNotFoundError: No module named 'isaacsim'"), "missing_isaacsim")
        self.assertEqual(classify_startup_error("ModuleNotFoundError: No module named 'isaaclab'"), "missing_isaaclab")
        self.assertEqual(classify_startup_error("Do you accept the EULA?"), "eula_required")

    def test_last_meaningful_line(self) -> None:
        self.assertEqual(last_meaningful_line(["", "  ", "real error"]), "real error")

    def test_phase_diagnosis_without_ipc_points_at_launcher(self) -> None:
        diagnosis = diagnose_startup_phase(phase="waiting_for_ipc", connected=False, ready=False)
        self.assertIn("No IPC connection", diagnosis)

    def test_process_exit_before_ipc_reads_stderr_tail(self) -> None:
        args = build_parser().parse_args(["--ui"])
        normalize_motion_args(args)
        client = SimProcessClient(args)
        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "stderr.log"
            stdout = Path(tmp) / "stdout.log"
            stderr.write_text("first\nModuleNotFoundError: No module named 'isaacsim'\n", encoding="utf-8")
            stdout.write_text("stdout line\n", encoding="utf-8")
            client.stderr_path = stderr
            client.stdout_path = stdout
            client.returncode = 1
            client._mark_exited_before_ipc()
        self.assertEqual(client.latest_status["phase"], "exited_before_ipc")
        self.assertEqual(client.latest_status["error_category"], "missing_isaacsim")
        self.assertIn("No module named 'isaacsim'", client.latest_status["error"])


if __name__ == "__main__":
    unittest.main()
