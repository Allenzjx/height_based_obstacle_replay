from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_ipc_protocol import decode_line, encode_message, make_message  # noqa: E402
from sim_onboard_camera import OnboardCameraProcessor  # noqa: E402
from sim_process_client import build_worker_command  # noqa: E402


class FakeConfig:
    onboard_camera_enabled = True
    camera_update_period_s = 0.1


class FakeScene:
    config = FakeConfig()
    camera = None
    camera_error = ""
    camera_parent_prim = ""
    camera_prim_path = ""


class VisionIpcTest(unittest.TestCase):
    def test_vision_control_round_trip(self) -> None:
        message = make_message("vision_control", action="reset_filter")
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_unknown_vision_action_is_diagnostic(self) -> None:
        processor = OnboardCameraProcessor(FakeScene())
        ok, reason = processor.handle_control("wat")
        self.assertFalse(ok)
        self.assertIn("unknown vision_control action", reason)

    def test_existing_message_types_still_round_trip(self) -> None:
        message = make_message("command", command="wheel all 1.0", source="ui")
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_build_worker_command_propagates_camera_args_and_enable_cameras(self) -> None:
        args = build_parser().parse_args(["--ui", "--height-cm", "10", "--camera-width", "320", "--camera-height", "240"])
        normalize_motion_args(args)
        command = build_worker_command(args, host="127.0.0.1", port=45678)
        self.assertIn("--camera-width", command)
        self.assertIn("320", command)
        self.assertIn("--camera-height", command)
        self.assertIn("240", command)
        self.assertIn("--onboard-camera", command)
        self.assertIn("--enable_cameras", command)

    def test_build_worker_command_does_not_force_cameras_when_disabled(self) -> None:
        args = build_parser().parse_args(["--ui", "--no-onboard-camera"])
        normalize_motion_args(args)
        command = build_worker_command(args, host="127.0.0.1", port=45678)
        self.assertIn("--no-onboard-camera", command)
        self.assertNotIn("--enable_cameras", command)


if __name__ == "__main__":
    unittest.main()
