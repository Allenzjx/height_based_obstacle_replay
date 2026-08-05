from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_ui_controller import HeightReplayController  # noqa: E402


class NoThreadDefaultTest(unittest.TestCase):
    def test_default_sim_launch_mode_is_subprocess(self) -> None:
        args = build_parser().parse_args(["--ui"])
        normalize_motion_args(args)
        self.assertEqual(args.sim_launch_mode, "subprocess")
        self.assertFalse(args.no_sim)

    def test_thread_worker_branch_and_choice_are_removed(self) -> None:
        source = inspect.getsource(HeightReplayController.start_sim_if_needed)
        self.assertIn('self.sim_launch_mode == "subprocess"', source)
        self.assertNotIn('self.sim_launch_mode == "thread"', source)
        action = next(row for row in build_parser()._actions if row.dest == "sim_launch_mode")
        self.assertNotIn("thread", action.choices)

    def test_no_sim_opens_without_worker(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        controller = HeightReplayController(args)
        controller.start_sim_if_needed()
        self.assertTrue(controller.no_sim)
        self.assertIsNone(controller.sim_client)
        self.assertFalse(hasattr(controller, "sim_worker"))


if __name__ == "__main__":
    unittest.main()
