from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker  # noqa: E402
import sim_worker_process  # noqa: E402
import sim_worker_runtime  # noqa: E402


class SubprocessWorkerRuntimeParityTest(unittest.TestCase):
    def test_thread_and_subprocess_use_same_runtime_entry_points(self) -> None:
        thread_source = inspect.getsource(sim_worker)
        process_source = inspect.getsource(sim_worker_process)
        required = [
            "create_adapter_config_from_args",
            "initialize_adapter_ground_reference",
            "handle_set_height",
            "handle_respawn",
            "handle_vision_control",
            "service_pending_viewport",
            "build_common_worker_status",
        ]

        for name in required:
            self.assertTrue(hasattr(sim_worker_runtime, name), name)
            self.assertIn(name, thread_source, name)
            self.assertIn(name, process_source, name)

    def test_worker_loops_do_not_inline_viewport_guard_policy(self) -> None:
        thread_source = inspect.getsource(sim_worker)
        process_source = inspect.getsource(sim_worker_process.run_worker)

        self.assertNotIn("allow_restore_on_failure=(action", thread_source)
        self.assertNotIn("allow_restore_on_failure=(action", process_source)
        self.assertNotIn("can_change_camera_view(", thread_source)
        self.assertNotIn("can_change_camera_view(", process_source)


if __name__ == "__main__":
    unittest.main()
