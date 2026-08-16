from __future__ import annotations

from collections import deque
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import height_replay_ui  # noqa: E402
from sim_ipc_protocol import decode_line, encode_message, make_message  # noqa: E402
from sim_process_client import (  # noqa: E402
    SimProcessClient,
    _append_worker_config_as_cli,
    _request_matched_shutdown_evidence,
    build_worker_config,
)
from sim_worker_process import (  # noqa: E402
    WorkerIpc,
    _finalize_active_task_session_for_worker_exit,
    _wait_for_close_receipt,
    _wait_for_task_exception_fast_shutdown,
    build_parser as build_worker_parser,
)


class _BlockedSocket:
    def send(self, _payload):
        raise BlockingIOError


class _PollingIpc:
    def __init__(self):
        self.messages = deque(
            [
                [{"type": "shutdown", "mode": "normal", "request_id": "normal"}],
                [{"type": "shutdown", "mode": "fast", "request_id": "fast"}],
            ]
        )
        self.sent = []
        self.flush_calls = 0

    def poll(self):
        return self.messages.popleft() if self.messages else []

    def send(self, message):
        self.sent.append(dict(message))

    def flush(self):
        self.flush_calls += 1


class _ExitedProcess:
    pid = 321

    @staticmethod
    def poll():
        return 0


class WorkerTaskReplayWiringTests(unittest.TestCase):
    def test_no_request_ordinary_worker_config_is_unchanged(self):
        args = height_replay_ui.build_parser().parse_args([])
        ordinary = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertNotIn("fsm50_task_request_path", ordinary)
        args.fsm50_task_request_path = "C:/task/request.json"
        configured = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertEqual(
            configured["fsm50_task_request_path"], "C:/task/request.json"
        )
        command = ["python", "sim_worker_process.py"]
        _append_worker_config_as_cli(command, configured)
        index = command.index("--fsm50-task-request-path")
        self.assertEqual(command[index + 1], "C:/task/request.json")

    def test_worker_parser_has_separate_normal_and_strict_opt_in_flags(self):
        args = build_worker_parser().parse_args(
            [
                "--fsm50-task-request-path",
                "task.json",
                "--fsm50-gate-request-path",
                "strict.json",
            ]
        )
        self.assertEqual(args.fsm50_task_request_path, "task.json")
        self.assertEqual(args.fsm50_gate_request_path, "strict.json")

    def test_task_terminal_types_round_trip_and_client_retains_ack(self):
        for kind in ("task_replay_complete", "task_replay_failed"):
            message = make_message(kind, request_id="request-v003")
            self.assertEqual(decode_line(encode_message(message)), message)

        args = height_replay_ui.build_parser().parse_args([])
        client = SimProcessClient(args)
        terminal = make_message(
            "task_replay_complete",
            request_id="request-v003",
            task_replay_complete=True,
        )
        client._handle_message(terminal)
        ack = make_message(
            "operation_ack",
            operation="task_replay",
            request_id="request-v003",
            accepted=True,
        )
        client._handle_message(ack)
        client._handle_message(
            make_message("status", ready=True, starting=False, phase="running")
        )
        self.assertEqual(
            client.latest_status["last_task_replay_terminal"]["request_id"],
            "request-v003",
        )
        self.assertEqual(
            client.latest_status["last_task_replay_ack"]["operation"],
            "task_replay",
        )

    def test_task_terminal_is_critical_under_worker_backlog(self):
        ipc = WorkerIpc.__new__(WorkerIpc)
        ipc.host = "127.0.0.1"
        ipc.port = 1
        ipc.sock = _BlockedSocket()
        ipc.buffer = None
        ipc.outbound = deque()
        ipc.current_outbound = None
        ipc.max_backlog = 4
        ipc.status_replaced = 0
        ipc.frames_sent = 0
        ipc.bytes_sent = 0
        ipc.status_enqueued = 0
        ipc.first_status_wall = 0.0
        ipc.last_status_wall = 0.0
        ipc.send_call_max_ms = 0.0
        ipc.send_call_total_ms = 0.0
        ipc.send_call_count = 0
        for index in range(5):
            ipc.send(make_message("log", text=f"row-{index}"))
        ipc.send(make_message("task_replay_complete", request_id="request-v003"))
        queued = [ipc.current_outbound, *ipc.outbound]
        task_rows = [row for row in queued if row and row["kind"] == "task_replay_complete"]
        self.assertEqual(len(task_rows), 1)
        self.assertTrue(task_rows[0]["critical"])

    def test_exception_path_waits_for_explicit_fast_shutdown_not_normal_close(self):
        ipc = _PollingIpc()
        session = SimpleNamespace(
            fast_close_ready=True,
            request=SimpleNamespace(request_id="task-request"),
        )
        adapter = SimpleNamespace(
            runtime_instance_id="adapter",
            root_state_write_count=0,
        )
        request_id = _wait_for_task_exception_fast_shutdown(
            ipc,
            task_session=session,
            adapter=adapter,
            worker_session_id="worker-session",
        )
        self.assertEqual(request_id, "fast")
        self.assertEqual(
            [(row["accepted"], row["request_id"]) for row in ipc.sent],
            [(False, "normal"), (True, "fast")],
        )
        self.assertTrue(all(row["type"] == "operation_ack" for row in ipc.sent))
        self.assertFalse(any(row["type"] == "close_receipt" for row in ipc.sent))
        self.assertFalse(any(row["type"] == "close_returned" for row in ipc.sent))
        self.assertEqual(
            ipc.sent[-1]["close_kwargs"],
            {"wait_for_replicator": False, "skip_cleanup": True},
        )

    def test_task_close_receipt_echoes_task_identity_and_worker_matches_it(self):
        args = height_replay_ui.build_parser().parse_args([])
        client = SimProcessClient(args)
        close_event = make_message(
            "close_requested",
            request_id="shutdown-request",
            mode="fast",
            accepted=True,
            error="",
            worker_pid=321,
            worker_session_id="worker-session",
            adapter_runtime_instance_id="adapter",
            artifact_request_id="",
            task_replay_request_id="task-request",
            root_state_write_count=0,
            close_kwargs={"wait_for_replicator": False, "skip_cleanup": True},
            runtime_version="5.1.0",
        )
        client._handle_message(close_event)
        receipt = client.latest_status["close_requested_receipt"]
        self.assertEqual(receipt["task_replay_request_id"], "task-request")

        polling = SimpleNamespace(
            poll=lambda: [
                {**receipt, "task_replay_request_id": "wrong-task"},
                receipt,
            ]
        )
        matched = _wait_for_close_receipt(
            polling,
            close_event,
            timeout_s=0.1,
        )
        self.assertEqual(matched["task_replay_request_id"], "task-request")

        strict_client = SimProcessClient(args)
        strict_client._handle_message(
            make_message(
                "close_requested",
                request_id="strict-shutdown",
                mode="fast",
                accepted=True,
                artifact_request_id="strict-artifact",
            )
        )
        self.assertNotIn(
            "task_replay_request_id",
            strict_client.latest_status["close_requested_receipt"],
        )

    def test_shutdown_outcome_uses_only_request_matched_ack_and_receipt(self):
        args = height_replay_ui.build_parser().parse_args([])
        client = SimProcessClient(args)
        common = {
            "mode": "fast",
            "accepted": True,
            "error": "",
            "task_replay_request_id": "task-request",
        }
        client._handle_message(
            make_message(
                "operation_ack",
                operation="shutdown",
                request_id="shutdown-request",
                **common,
            )
        )
        client._handle_message(
            make_message(
                "close_requested",
                request_id="shutdown-request",
                worker_pid=321,
                worker_session_id="worker-session",
                adapter_runtime_instance_id="adapter",
                artifact_request_id="",
                root_state_write_count=0,
                close_kwargs={
                    "wait_for_replicator": False,
                    "skip_cleanup": True,
                },
                runtime_version="5.1.0",
                **common,
            )
        )
        # Newer unrelated traffic must not be mistaken for this shutdown.
        client._handle_message(
            make_message(
                "operation_ack",
                operation="shutdown",
                request_id="stale-other-request",
                **common,
            )
        )
        client._handle_message(
            make_message(
                "close_requested",
                request_id="stale-other-request",
                **common,
            )
        )
        evidence = _request_matched_shutdown_evidence(
            client.latest_status,
            request_id="shutdown-request",
        )
        self.assertEqual(evidence["shutdown_ack"]["request_id"], "shutdown-request")
        self.assertEqual(
            evidence["close_requested_receipt"]["request_id"],
            "shutdown-request",
        )

        client.process = _ExitedProcess()
        outcome = client.shutdown(
            mode="fast",
            request_id="shutdown-request",
            force_on_timeout=False,
        )
        self.assertEqual(outcome["shutdown_ack"]["request_id"], "shutdown-request")
        self.assertEqual(
            outcome["close_requested_ack"]["request_id"], "shutdown-request"
        )
        self.assertEqual(
            outcome["close_requested_receipt"]["task_replay_request_id"],
            "task-request",
        )
        self.assertEqual(outcome["close_requested"], outcome["close_requested_ack"])
        self.assertEqual(outcome["close_returned"], {})
        self.assertEqual(outcome["close_returned_ack"], {})

    def test_unexpected_worker_loop_exit_terminalizes_active_task(self):
        published = []

        class _Session:
            def __init__(self):
                self.calls = []

            def fail(self, reason, **kwargs):
                self.calls.append((reason, kwargs))
                return {"type": "task_replay_failed", "error": reason}

        session = _Session()
        sent = _finalize_active_task_session_for_worker_exit(
            session,
            terminal_sent=False,
            publish_terminal=lambda row: published.append(dict(row)),
            reason="simulation app stopped before task replay terminal",
        )
        self.assertTrue(sent)
        self.assertEqual(len(published), 1)
        self.assertTrue(session.calls[0][1]["infrastructure_failure"])
        self.assertTrue(session.calls[0][1]["simulation_app_stopped"])


if __name__ == "__main__":
    unittest.main()
