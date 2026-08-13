from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_onboard_camera import OnboardCameraProcessor  # noqa: E402
import sim_worker  # noqa: E402
import sim_worker_process  # noqa: E402


class CameraViewportPendingSingleOwnerTest(unittest.TestCase):
    def test_camera_update_does_not_service_pending_viewport(self) -> None:
        source = inspect.getsource(OnboardCameraProcessor.update)

        self.assertNotIn("service_pending_camera_viewport", source)

    def test_worker_loops_are_the_pending_viewport_owner(self) -> None:
        self.assertIn("service_pending_viewport", inspect.getsource(sim_worker.SimWorker._run))
        self.assertIn("service_pending_viewport", inspect.getsource(sim_worker_process.run_worker))


if __name__ == "__main__":
    unittest.main()
