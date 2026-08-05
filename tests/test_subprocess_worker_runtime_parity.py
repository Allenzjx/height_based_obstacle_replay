from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_process  # noqa: E402
import sim_worker_runtime  # noqa: E402


class SubprocessWorkerRuntimeParityTest(unittest.TestCase):
    def test_single_subprocess_uses_shared_runtime_entry_points(self) -> None:
        process_source = inspect.getsource(sim_worker_process)
        required = [
            "create_adapter_config_from_args",
            "initialize_adapter_ground_reference",
            "handle_set_height",
            "handle_respawn",
            "build_common_worker_status",
        ]

        for name in required:
            self.assertTrue(hasattr(sim_worker_runtime, name), name)
            self.assertIn(name, process_source, name)
        self.assertFalse((MODULE_ROOT / "sim_worker.py").exists())

    def test_worker_loops_do_not_contain_removed_task_policy(self) -> None:
        process_source = inspect.getsource(sim_worker_process.run_worker)
        for removed in ("vision_control", "service_pending_viewport", "stability_session_id"):
            self.assertNotIn(removed, process_source)


if __name__ == "__main__":
    unittest.main()
