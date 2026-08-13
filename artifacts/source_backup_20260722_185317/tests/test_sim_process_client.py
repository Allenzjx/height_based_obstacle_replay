from __future__ import annotations

import inspect
import sys
import tempfile
import time
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import sim_process_client  # noqa: E402
from height_replay_ui import build_parser, normalize_motion_args  # noqa: E402
from sim_ipc_protocol import make_message  # noqa: E402
from sim_process_client import SimProcessClient, build_worker_command, tail_file  # noqa: E402
from sim_transport import SimTransport  # noqa: E402


def make_args():
    args = build_parser().parse_args(["--ui", "--height-cm", "10", "--max-wheel-speed-rad-s", "4.0"])
    normalize_motion_args(args)
    return args


class SimProcessClientTest(unittest.TestCase):
    def test_build_worker_command_contains_ipc_and_motion_args(self) -> None:
        command = build_worker_command(make_args(), host="127.0.0.1", port=45678)
        joined = " ".join(command)
        self.assertIn("sim_worker_process.py", joined)
        self.assertIn("--ipc-host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--ipc-port", command)
        self.assertIn("45678", command)
        self.assertIn("--height-cm", command)
        self.assertIn("10", command)
        self.assertIn("--max-wheel-speed-rad-s", command)
        self.assertIn("4.0", command)

    def test_subprocess_client_does_not_use_threading_thread(self) -> None:
        source = inspect.getsource(sim_process_client)
        self.assertNotIn("threading.Thread", source)

    def test_transport_can_queue_command_messages_before_connect(self) -> None:
        client = SimProcessClient(make_args())
        transport = SimTransport()
        transport.attach_process_client(client)
        transport.send("wheel all 1.0")
        self.assertEqual(client.pending_messages[-1]["type"], "command")
        self.assertEqual(client.pending_messages[-1]["command"], "wheel all 1.0")

    def test_worker_status_parsing(self) -> None:
        client = SimProcessClient(make_args())
        client._handle_message(
            make_message(
                "status",
                ready=True,
                phase="running",
                height_cm=10,
                command_state={"servos": {}, "wheels": {}},
                physics_dt=1.0 / 120.0,
            )
        )
        status = client.status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["phase"], "running")
        self.assertEqual(status["height_cm"], 10)

    def test_stdout_stderr_tail_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.log"
            path.write_text("\n".join(f"line {index}" for index in range(10)), encoding="utf-8")
            self.assertEqual(tail_file(path, 3), ["line 7", "line 8", "line 9"])

    def test_timeout_status(self) -> None:
        args = make_args()
        args.sim_startup_timeout_s = 120.0
        client = SimProcessClient(args)
        now = time.monotonic()
        client.start_time = now - 130.0
        client.last_status_time = now - 11.0
        client.conn = object()
        client._apply_timeouts()
        self.assertIn("did not become ready", client.latest_status["startup_timeout_warning"])
        self.assertIn("No Isaac worker status", client.latest_status["status_timeout_warning"])


if __name__ == "__main__":
    unittest.main()
