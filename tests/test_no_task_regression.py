from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_manifest import SUPPORTED_HEIGHTS_CM  # noqa: E402
from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from height_sequence_store import HeightSequenceStore  # noqa: E402
from playback import plan_from_steps  # noqa: E402
from sequence_model import empty_command_state, make_event, make_step  # noqa: E402


class NoTaskRegressionTest(unittest.TestCase):
    def test_height_sequence_store_paths_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HeightSequenceStore(Path(tmp))
            self.assertEqual(store.steps_path(10), Path(tmp) / "height_10cm" / "accepted_steps.jsonl")
            self.assertEqual(store.steps_path(0), Path(tmp) / "height_00cm" / "accepted_steps.jsonl")

    def test_manifest_supported_heights_unchanged(self) -> None:
        self.assertEqual(list(SUPPORTED_HEIGHTS_CM), list(range(0, 45, 5)))

    def test_playback_plan_timing_is_direct(self) -> None:
        step = make_step(
            index=1,
            step_type="test",
            duration=1.0,
            events=[make_event(0.5, "servo front_left_knee -30")],
            command_state_before=empty_command_state(),
            command_state_after=empty_command_state(),
        )
        plan = plan_from_steps([step], profile="fast")
        self.assertEqual(len(plan.events), 1)
        self.assertAlmostEqual(plan.events[0].time_s, 0.0)

    def test_sim_launch_mode_defaults_remain_subprocess_and_no_sim_disabled(self) -> None:
        args = build_parser().parse_args(["--ui"])
        normalize_motion_args(args)
        self.assertEqual(args.sim_launch_mode, "subprocess")

        args = build_parser().parse_args(["--ui", "--no-sim"])
        normalize_motion_args(args)
        self.assertTrue(args.no_sim)
        self.assertEqual(args.sim_launch_mode, "disabled")


if __name__ == "__main__":
    unittest.main()
