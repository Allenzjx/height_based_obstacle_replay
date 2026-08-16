from __future__ import annotations

from collections import deque
import inspect
from pathlib import Path
import unittest
from types import SimpleNamespace

import height_replay_ui
from sim_ipc_protocol import (
    MACRO_FAST_CLOSE_SCHEMA,
    decode_line,
    encode_message,
    make_message,
)
from sim_process_client import SimProcessClient, _append_worker_config_as_cli, build_worker_config
from sim_robot_adapter import SimRobotAdapter
from sim_worker_process import (
    WorkerIpc,
    _finalize_active_macro_session_for_worker_exit,
    _wait_for_close_receipt,
    build_parser as build_worker_parser,
)


class _BlockedSocket:
    def send(self, _payload):
        raise BlockingIOError


class WorkerMacroFSMWiringTests(unittest.TestCase):
    def test_macro_request_is_strictly_opt_in_and_cli_round_trips(self):
        args = height_replay_ui.build_parser().parse_args([])
        ordinary = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertNotIn("fsm50_macro_request_path", ordinary)
        args.fsm50_macro_request_path = "C:/macro/request.json"
        configured = build_worker_config(args, host="127.0.0.1", port=1234)
        self.assertEqual(
            configured["fsm50_macro_request_path"], "C:/macro/request.json"
        )
        command = ["python", "sim_worker_process.py"]
        _append_worker_config_as_cli(command, configured)
        index = command.index("--fsm50-macro-request-path")
        self.assertEqual(command[index + 1], "C:/macro/request.json")
        parsed = build_worker_parser().parse_args(
            ["--fsm50-macro-request-path", "request.json"]
        )
        self.assertEqual(parsed.fsm50_macro_request_path, "request.json")

    def test_protocol_client_keeps_start_terminal_and_terminal_ack(self):
        for kind in ("start_macro_fsm", "macro_fsm_complete", "macro_fsm_failed"):
            message = make_message(kind, request_id="macro-request")
            self.assertEqual(decode_line(encode_message(message)), message)
        args = height_replay_ui.build_parser().parse_args([])
        client = SimProcessClient(args)
        client.start_macro_fsm(
            request_id="macro-request",
            worker_session_id="worker-session",
            source_version="v003",
            profile_id="profiles",
            graph_id="graph",
            graph_sha256="a" * 64,
            profile_library_sha256="b" * 64,
            bundle_sha256="c" * 64,
        )
        queued = client.pending_messages[-1]
        self.assertEqual(queued["type"], "start_macro_fsm")
        self.assertEqual(queued["bundle_sha256"], "c" * 64)
        terminal = make_message(
            "macro_fsm_complete",
            operation="macro_fsm",
            phase="MACRO_FSM_COMPLETE",
            request_id="macro-request",
        )
        client._handle_message(terminal)
        client._handle_message(
            make_message(
                "operation_ack",
                operation="macro_fsm",
                phase="MACRO_FSM_COMPLETE",
                request_id="macro-request",
                accepted=True,
            )
        )
        client._handle_message(make_message("status", ready=True, starting=False))
        self.assertEqual(
            client.latest_status["last_macro_fsm_terminal"]["request_id"],
            "macro-request",
        )
        self.assertEqual(
            client.latest_status["last_macro_fsm_ack"]["operation"], "macro_fsm"
        )

    def test_macro_terminal_is_critical_under_worker_backlog(self):
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
        ipc.send(make_message("macro_fsm_complete", request_id="macro-request"))
        queued = [ipc.current_outbound, *ipc.outbound]
        rows = [row for row in queued if row and row["kind"] == "macro_fsm_complete"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["critical"])

    def test_close_receipt_preserves_macro_request_identity(self):
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
            macro_fsm_request_id="macro-request",
            root_state_write_count=0,
            close_kwargs={"wait_for_replicator": False, "skip_cleanup": True},
            runtime_version="5.1.0",
            schema_version=MACRO_FAST_CLOSE_SCHEMA,
        )
        client._handle_message(close_event)
        receipt = client.latest_status["close_requested_receipt"]
        self.assertEqual(receipt["macro_fsm_request_id"], "macro-request")
        self.assertEqual(receipt["schema_version"], MACRO_FAST_CLOSE_SCHEMA)
        polling = SimpleNamespace(
            poll=lambda: [
                {**receipt, "schema_version": "wrong"},
                {**receipt, "macro_fsm_request_id": "wrong"},
                receipt,
            ]
        )
        matched = _wait_for_close_receipt(polling, close_event, timeout_s=0.1)
        self.assertEqual(matched["macro_fsm_request_id"], "macro-request")

    def test_unexpected_loop_exit_terminalizes_macro_before_close(self):
        published = []

        class _Session:
            def __init__(self):
                self.calls = []

            def fail(self, reason, **kwargs):
                self.calls.append((reason, kwargs))
                return {"type": "macro_fsm_failed", "error": reason}

        session = _Session()
        sent = _finalize_active_macro_session_for_worker_exit(
            session,
            terminal_sent=False,
            publish_terminal=lambda row: published.append(dict(row)),
            reason="simulation app stopped before Macro FSM terminal",
        )
        self.assertTrue(sent)
        self.assertEqual(len(published), 1)
        self.assertTrue(session.calls[0][1]["infrastructure_failure"])
        self.assertTrue(session.calls[0][1]["simulation_app_stopped"])

    def test_existing_adapter_hook_is_per_physics_substep_post_update(self):
        source = inspect.getsource(SimRobotAdapter.step)
        self.assertIn("for _substep in range(substeps):", source)
        self.assertIn("collector.on_step(self, physics_dt)", source)
        self.assertLess(source.index("self.robot.update(physics_dt)"), source.index("collector.on_step"))
        # Rendering occurs once after all substeps when render_interval > 1;
        # the collector remains inside the loop and is therefore not 15 Hz.
        self.assertLess(source.index("collector.on_step"), source.rindex("self.sim.render()"))

    def test_macro_runtime_has_no_playback_or_legacy_action_import(self):
        module = Path(__file__).resolve().parents[1]
        text = "\n".join(
            (module / name).read_text(encoding="utf-8")
            for name in ("worker_macro_fsm_session.py", "fsm50_macro_runner.py")
        )
        self.assertNotIn("from playback import", text)
        self.assertNotIn("import playback", text)
        self.assertNotIn("SimTimePlaybackService", text)
        self.assertNotIn("fsm_states_57", text)


if __name__ == "__main__":
    unittest.main()
