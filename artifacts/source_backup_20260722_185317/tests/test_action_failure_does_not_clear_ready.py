from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for path in (MODULE_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from controller_test_utils import make_args  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class ReadyStatusClient:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = dict(status)

    def poll(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return dict(self._status)


class ActionFailureDoesNotClearReadyTest(unittest.TestCase):
    def test_status_error_from_action_keeps_runtime_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = HeightReplayController(make_args(Path(tmp), no_sim=False))
            controller.sim_client = ReadyStatusClient(
                {
                    "ready": True,
                    "runtime_ready": True,
                    "phase": "running",
                    "error": "validation blocked: Waiting for a new Camera frame.",
                    "vision": {"camera_ready": True},
                }
            )  # type: ignore[assignment]

            controller.update()

            self.assertTrue(controller.sim_ready)
            self.assertTrue(controller.runtime_ready)
            self.assertIn("validation blocked", controller.status)


if __name__ == "__main__":
    unittest.main()
