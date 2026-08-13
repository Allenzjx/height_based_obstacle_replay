from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_ipc_protocol import decode_line, encode_message, make_message  # noqa: E402
from sim_onboard_camera import OnboardCameraProcessor  # noqa: E402
from sim_process_client import SimProcessClient  # noqa: E402


class FakeConfig:
    onboard_camera_enabled = True
    camera_update_period_s = 0.1


class FakeScene:
    config = FakeConfig()
    camera = None
    camera_error = ""
    camera_parent_prim = ""
    camera_prim_path = ""


class CameraValidationIpcTest(unittest.TestCase):
    def test_validate_current_height_round_trip_has_only_small_fields(self) -> None:
        message = make_message("vision_control", action="validate_current_height", expected_height_cm=5)
        decoded = decode_line(encode_message(message))
        self.assertEqual(decoded, message)
        payload_text = encode_message(message).decode("utf-8")
        self.assertNotIn("rgb", payload_text.lower())
        self.assertNotIn("depth_image", payload_text.lower())
        self.assertNotIn("base64", payload_text.lower())

    def test_processor_accepts_new_validation_actions(self) -> None:
        processor = OnboardCameraProcessor(FakeScene())
        ok, text = processor.handle_control("validate_current_height", expected_height_cm=5)
        self.assertTrue(ok, text)
        self.assertEqual(processor._height_validation_expected_cm, 5)
        ok, text = processor.handle_control("save_rgbd_diagnostic", expected_height_cm=5)
        self.assertTrue(ok, text)
        self.assertTrue(processor.save_debug_requested)
        ok, text = processor.handle_control("clear_validation_result")
        self.assertTrue(ok, text)
        self.assertIsNone(processor._height_validation_expected_cm)

    def test_process_client_queues_validation_without_images(self) -> None:
        args = SimpleNamespace()
        client = SimProcessClient(args)
        client.validate_current_height(5)
        client.save_rgbd_diagnostic(5)
        for message in client.pending_messages:
            self.assertEqual(message["type"], "vision_control")
            self.assertNotIn("rgb", message)
            self.assertNotIn("depth", message)
            self.assertNotIn("pointcloud", message)
            self.assertNotIn("base64", message)


if __name__ == "__main__":
    unittest.main()
