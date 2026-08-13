from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from playback import plan_from_steps  # noqa: E402
from sequence_model import empty_command_state, make_event, make_step  # noqa: E402


def sample_steps() -> list[dict]:
    return [
        make_step(
            index=1,
            step_type="test",
            duration=2.0,
            events=[
                make_event(0.0, "servo front_left_hip 10"),
                make_event(1.0, "wheel all 1.0"),
            ],
            command_state_before=empty_command_state(),
            command_state_after=empty_command_state(),
            name="sample",
        )
    ]


class PlaybackSpeedScaleTest(unittest.TestCase):
    def test_playback_speed_changes_timing(self) -> None:
        plan = plan_from_steps(sample_steps(), profile="fast", speed=2.0, trailing_pad=0.0)
        self.assertAlmostEqual(plan.events[0].time_s, 0.0)
        self.assertAlmostEqual(plan.events[1].time_s, 0.5)
        self.assertAlmostEqual(plan.final_time_s, 0.5)

    def test_preserve_wheel_distance_scales_wheel_speed(self) -> None:
        plan = plan_from_steps(
            sample_steps(),
            profile="fast",
            speed=2.0,
            trailing_pad=0.0,
            preserve_wheel_distance=True,
            max_wheel_speed=3.0,
        )
        self.assertEqual(plan.events[1].command, "wheel all 2")

    def test_preserve_wheel_distance_clamps_scaled_wheel_speed(self) -> None:
        plan = plan_from_steps(
            sample_steps(),
            profile="fast",
            speed=4.0,
            trailing_pad=0.0,
            preserve_wheel_distance=True,
            max_wheel_speed=3.0,
        )
        self.assertEqual(plan.events[1].command, "wheel all 3")

    def test_without_preserve_wheel_distance_keeps_wheel_speed(self) -> None:
        plan = plan_from_steps(
            sample_steps(),
            profile="fast",
            speed=2.0,
            trailing_pad=0.0,
            preserve_wheel_distance=False,
            max_wheel_speed=3.0,
        )
        self.assertEqual(plan.events[1].command, "wheel all 1.0")


if __name__ == "__main__":
    unittest.main()
