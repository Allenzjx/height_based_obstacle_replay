from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from command_model import HIP_LIMIT_DEG, KNEE_LIMIT_DEG, clamp_servo_command  # noqa: E402
from sequence_model import apply_command_to_state, empty_command_state  # noqa: E402


class ServoLimitsTest(unittest.TestCase):
    def test_requested_command_limits(self) -> None:
        self.assertEqual(HIP_LIMIT_DEG, (-135.0, 135.0))
        self.assertEqual(KNEE_LIMIT_DEG, (-60.0, 210.0))

    def test_servo_commands_are_clamped_in_command_space(self) -> None:
        self.assertEqual(clamp_servo_command("front_left_hip", 200.0), 135.0)
        self.assertEqual(clamp_servo_command("front_left_hip", -200.0), -135.0)
        self.assertEqual(clamp_servo_command("front_left_knee", -90.0), -60.0)
        self.assertEqual(clamp_servo_command("front_left_knee", 250.0), 210.0)

    def test_sequence_state_clamps_out_of_range_servo_commands(self) -> None:
        state = empty_command_state()
        apply_command_to_state(state, "servo front_left_hip 180")
        apply_command_to_state(state, "servo front knee -90")
        self.assertEqual(state["servos"]["front_left_hip"], 135.0)
        self.assertEqual(state["servos"]["front_left_knee"], -60.0)
        self.assertEqual(state["servos"]["front_right_knee"], -60.0)

    def test_command_zero_remains_standing_pose_command(self) -> None:
        state = empty_command_state()
        apply_command_to_state(state, "servo all 0")
        self.assertTrue(all(value == 0.0 for value in state["servos"].values()))


if __name__ == "__main__":
    unittest.main()
