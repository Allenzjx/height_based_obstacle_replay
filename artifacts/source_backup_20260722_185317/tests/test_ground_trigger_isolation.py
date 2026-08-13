from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_worker_process  # noqa: E402


class GroundTriggerIsolationTest(unittest.TestCase):
    def test_camera_view_ground_smoke_has_fixed_stage_names(self) -> None:
        source = inspect.getsource(sim_worker_process.run_camera_view_ground_contact_smoke)
        stages = [
            "startup_baseline",
            "after_set_height_respawn",
            "immediately_before_open",
            "immediately_after_open",
            "after_open_1_step",
            "after_open_10_steps",
            "after_open_60_steps",
            "before_return_perspective",
            "after_return_perspective",
            "after_return_60_steps",
            "after_close_camera_viewport",
        ]

        for stage in stages:
            self.assertIn(stage, source)

    def test_camera_view_ground_smoke_does_not_restore_state_around_open(self) -> None:
        source = inspect.getsource(sim_worker_process.run_camera_view_ground_contact_smoke)

        self.assertNotIn("restore_sim_state", source)
        self.assertIn("trigger_stage", source)
        self.assertIn("allow_state_restore=False", source)


if __name__ == "__main__":
    unittest.main()
