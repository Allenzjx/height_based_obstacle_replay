from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from sim_ipc_protocol import JsonLineBuffer, decode_line, encode_message, make_message  # noqa: E402


class IpcProtocolTest(unittest.TestCase):
    def test_command_round_trip(self) -> None:
        message = make_message("command", command="wheel all 1.0", source="ui")
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_status_round_trip(self) -> None:
        message = make_message("status", ready=True, phase="running", physics_dt=1.0 / 120.0)
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_error_round_trip(self) -> None:
        message = make_message("error", error="boom", traceback="trace")
        self.assertEqual(decode_line(encode_message(message)), message)

    def test_invalid_json_is_safe_error_message(self) -> None:
        decoded = decode_line("{not-json")
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["type"], "error")
        self.assertIn("Invalid IPC JSON", decoded["error"])

    def test_incremental_line_buffer(self) -> None:
        first = encode_message(make_message("hello", phase="process_started"))
        second = encode_message(make_message("shutdown"))
        buffer = JsonLineBuffer()
        self.assertEqual(buffer.feed(first[:5]), [])
        messages = buffer.feed(first[5:] + second)
        self.assertEqual([message["type"] for message in messages], ["hello", "shutdown"])


if __name__ == "__main__":
    unittest.main()
