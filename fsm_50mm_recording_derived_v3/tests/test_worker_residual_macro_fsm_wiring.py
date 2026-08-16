from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import height_replay_ui
from sim_ipc_protocol import decode_line, encode_message, make_message
from sim_process_client import (
    SimProcessClient,
    _append_worker_config_as_cli,
    build_worker_config,
)
from sim_worker_process import (
    _build_macro_start_operation_ack,
    _load_exclusive_fsm50_worker_requests,
    _macro_route_ack_identity,
    _validate_macro_route_start_binding,
    _wait_for_close_receipt,
    build_parser as build_worker_parser,
)


def _client_args(**updates):
    args = height_replay_ui.build_parser().parse_args([])
    for key, value in {
        "fsm50_gate_request_path": "",
        "fsm50_task_request_path": "",
        "fsm50_macro_request_path": "",
        "fsm50_residual_macro_request_path": "",
        **updates,
    }.items():
        setattr(args, key, value)
    return args


def test_residual_request_path_is_strictly_opt_in_and_reaches_child_cli():
    missing = build_worker_config(
        _client_args(), host="127.0.0.1", port=1234
    )
    assert "fsm50_residual_macro_request_path" not in missing

    exact_path = "C:/gate_e/request.json"
    configured = build_worker_config(
        _client_args(fsm50_residual_macro_request_path=exact_path),
        host="127.0.0.1",
        port=1234,
    )
    assert configured["fsm50_residual_macro_request_path"] == exact_path
    command = ["python", "sim_worker_process.py"]
    _append_worker_config_as_cli(command, configured)
    index = command.index("--fsm50-residual-macro-request-path")
    assert command[index + 1] == exact_path

    parsed = build_worker_parser().parse_args(command[index : index + 2])
    assert parsed.fsm50_residual_macro_request_path == exact_path


def test_empty_residual_option_leaves_old_worker_cli_byte_for_byte_unchanged():
    args_without_attribute = height_replay_ui.build_parser().parse_args([])
    args_with_empty_attribute = height_replay_ui.build_parser().parse_args([])
    args_with_empty_attribute.fsm50_residual_macro_request_path = ""
    old_config = build_worker_config(
        args_without_attribute, host="127.0.0.1", port=1234
    )
    empty_config = build_worker_config(
        args_with_empty_attribute, host="127.0.0.1", port=1234
    )
    assert empty_config == old_config

    old_command = ["python", "sim_worker_process.py"]
    empty_command = ["python", "sim_worker_process.py"]
    _append_worker_config_as_cli(old_command, old_config)
    _append_worker_config_as_cli(empty_command, empty_config)
    assert empty_command == old_command


@pytest.mark.parametrize(
    "old_key,old_flag",
    (
        ("fsm50_gate_request_path", "--fsm50-gate-request-path"),
        ("fsm50_task_request_path", "--fsm50-task-request-path"),
        ("fsm50_macro_request_path", "--fsm50-macro-request-path"),
    ),
)
def test_old_request_path_flags_retain_their_exact_mapping(old_key, old_flag):
    config = {old_key: "C:/old/request.json"}
    command = ["python", "sim_worker_process.py"]
    _append_worker_config_as_cli(command, config)
    assert command == [
        "python",
        "sim_worker_process.py",
        old_flag,
        "C:/old/request.json",
        "--save-scene",
        "--apply-safe-servo-joint-limits",
        "--apply-physx-joint-limits",
        "--defer-first-visible-render",
    ]


def test_unknown_residual_cli_alias_is_not_admitted():
    with pytest.raises(SystemExit):
        build_worker_parser().parse_args(
            ["--fsm50-residual-request-path", "request.json"]
        )


def _route_args(**updates):
    values = {
        "fsm50_gate_request_path": "",
        "fsm50_task_request_path": "",
        "fsm50_macro_request_path": "",
        "fsm50_residual_macro_request_path": "",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _path_loader(label, accepted_path):
    def load(path):
        if not path:
            return None
        if path != accepted_path:
            raise ValueError(f"{label} unknown path")
        return label

    return load


def test_exact_residual_path_selects_only_the_dedicated_loader():
    path = "C:/gate_e/request.json"
    requests = _load_exclusive_fsm50_worker_requests(
        _route_args(fsm50_residual_macro_request_path=path),
        gate_loader=_path_loader("gate", "C:/gate.json"),
        task_loader=_path_loader("task", "C:/task.json"),
        macro_loader=_path_loader("macro", "C:/macro.json"),
        residual_loader=_path_loader("residual", path),
    )
    assert requests == (None, None, None, "residual")


def test_missing_unknown_or_dual_residual_route_fails_closed():
    with pytest.raises(ValueError, match="residual unknown path"):
        _load_exclusive_fsm50_worker_requests(
            _route_args(fsm50_residual_macro_request_path="C:/unknown.json"),
            gate_loader=lambda _path: None,
            task_loader=lambda _path: None,
            macro_loader=lambda _path: None,
            residual_loader=_path_loader("residual", "C:/exact.json"),
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _load_exclusive_fsm50_worker_requests(
            _route_args(
                fsm50_macro_request_path="C:/macro.json",
                fsm50_residual_macro_request_path="C:/residual.json",
            ),
            gate_loader=lambda _path: None,
            task_loader=lambda _path: None,
            macro_loader=_path_loader("macro", "C:/macro.json"),
            residual_loader=_path_loader("residual", "C:/residual.json"),
        )

    assert _load_exclusive_fsm50_worker_requests(
        _route_args(),
        gate_loader=lambda _path: None,
        task_loader=lambda _path: None,
        macro_loader=lambda _path: None,
        residual_loader=lambda _path: None,
    ) == (None, None, None, None)


def test_start_transport_is_unchanged_but_route_validator_is_exclusive():
    transport = make_message(
        "start_macro_fsm",
        schema_version="fsm50.start_residual_macro_fsm.v1",
        operation="residual_macro_fsm",
    )
    assert decode_line(encode_message(transport)) == transport
    with pytest.raises(ValueError, match="Unsupported IPC message type"):
        make_message("start_residual_macro_fsm")

    base_request = object()
    residual_request = object()
    base_session = SimpleNamespace(request=base_request)
    residual_session = SimpleNamespace(
        request=base_request,
        residual_request=residual_request,
    )
    calls = []

    def macro_validator(request, message, **kwargs):
        calls.append(("macro", request, message, kwargs))
        return []

    def residual_validator(request, message, **kwargs):
        calls.append(("residual", request, message, kwargs))
        return []

    assert _validate_macro_route_start_binding(
        base_session,
        None,
        transport,
        expected_worker_session_id="worker",
        macro_validator=macro_validator,
        residual_validator=residual_validator,
    ) == []
    assert calls[-1][0] == "macro"
    assert _validate_macro_route_start_binding(
        residual_session,
        residual_request,
        transport,
        expected_worker_session_id="worker",
        macro_validator=macro_validator,
        residual_validator=residual_validator,
    ) == []
    assert calls[-1][0] == "residual"
    assert _validate_macro_route_start_binding(
        base_session,
        residual_request,
        transport,
        expected_worker_session_id="worker",
        residual_validator=residual_validator,
    ) == ["Gate-E residual session/request route binding mismatch"]


def test_residual_route_ack_uses_outer_and_preserves_base_identity():
    identity_calls = []
    residual_request = SimpleNamespace(
        request_id="outer-gate-e",
        gate_e_identity=lambda *, payload_role: identity_calls.append(
            payload_role
        )
        or {"payload_role": payload_role, "policy_kind": "ZERO"},
    )
    session = SimpleNamespace(
        request=SimpleNamespace(request_id="base-macro"),
        residual_request=residual_request,
    )
    identity = _macro_route_ack_identity(session, payload_role="shutdown_ack")
    assert identity["macro_fsm_request_id"] == "outer-gate-e"
    assert identity["residual_macro_fsm_request_id"] == "outer-gate-e"
    assert identity["base_macro_fsm_request_id"] == "base-macro"
    assert identity["gate_e_zero_residual"] == {
        "payload_role": "shutdown_ack",
        "policy_kind": "ZERO",
    }
    assert identity_calls == ["shutdown_ack"]


def test_accepted_residual_start_builds_shared_transport_ack_without_collision():
    start_payload = {
        "accepted": True,
        "operation": "residual_macro_fsm",
        "request_id": "outer-gate-e",
        "request_identity_sha256": "a" * 64,
        "residual_macro_fsm_request_id": "outer-gate-e",
        "base_macro_fsm_request_id": "base-macro",
        "gate_e_zero_residual": {
            "schema_version": "fsm50.gate_e_zero_residual_identity.v1",
            "payload_role": "start_ack",
            "policy_kind": "ZERO",
        },
        "error": "",
    }
    ack = _build_macro_start_operation_ack(
        message={"request_id": "outer-gate-e"},
        accepted=True,
        rejection_reason="",
        adapter=SimpleNamespace(
            runtime_instance_id="adapter-instance",
            root_state_write_count=0,
        ),
        worker_session_id="worker-session",
        start_payload=start_payload,
    )
    assert ack["type"] == "operation_ack"
    assert ack["operation"] == "start_macro_fsm"
    assert ack["accepted"] is True
    assert ack["request_id"] == "outer-gate-e"
    assert ack["request_identity_sha256"] == "a" * 64
    assert ack["residual_macro_fsm_request_id"] == "outer-gate-e"
    assert ack["base_macro_fsm_request_id"] == "base-macro"
    assert ack["gate_e_zero_residual"]["payload_role"] == "start_ack"


def test_gate_e_close_receipt_echoes_outer_base_and_identity_exactly():
    client = SimProcessClient(_client_args())
    gate_identity = {
        "schema_version": "fsm50.gate_e_zero_residual_identity.v1",
        "payload_role": "close_requested",
        "request_id": "outer-gate-e",
        "base_request_id": "base-macro",
        "policy_kind": "ZERO",
    }
    try:
        client._handle_message(
            make_message(
                "close_requested",
                request_id="shutdown-request",
                macro_fsm_request_id="outer-gate-e",
                residual_macro_fsm_request_id="outer-gate-e",
                base_macro_fsm_request_id="base-macro",
                gate_e_zero_residual=gate_identity,
            )
        )
        receipt = client.pending_messages[-1]
        assert receipt["type"] == "close_receipt"
        assert receipt["close_event_type"] == "close_requested"
        assert receipt["macro_fsm_request_id"] == "outer-gate-e"
        assert receipt["residual_macro_fsm_request_id"] == "outer-gate-e"
        assert receipt["base_macro_fsm_request_id"] == "base-macro"
        assert receipt["gate_e_zero_residual"] == gate_identity
    finally:
        client.close()


def test_gate_e_worker_close_barrier_requires_exact_outer_base_and_identity():
    gate_identity = {
        "schema_version": "fsm50.gate_e_zero_residual_identity.v1",
        "payload_role": "close_requested",
        "request_id": "outer-gate-e",
        "base_request_id": "base-macro",
        "policy_kind": "ZERO",
    }
    close_event = make_message(
        "close_requested",
        request_id="shutdown-request",
        macro_fsm_request_id="outer-gate-e",
        residual_macro_fsm_request_id="outer-gate-e",
        base_macro_fsm_request_id="base-macro",
        gate_e_zero_residual=gate_identity,
    )
    exact = make_message(
        "close_receipt",
        close_event_type="close_requested",
        received=True,
        request_id="shutdown-request",
        mode="",
        accepted=None,
        error="",
        worker_pid=None,
        worker_session_id="",
        adapter_runtime_instance_id="",
        artifact_request_id="",
        root_state_write_count=None,
        close_kwargs={},
        runtime_version="",
        macro_fsm_request_id="outer-gate-e",
        residual_macro_fsm_request_id="outer-gate-e",
        base_macro_fsm_request_id="base-macro",
        gate_e_zero_residual=gate_identity,
    )

    class _Polling:
        def __init__(self, rows):
            self.rows = list(rows)

        def poll(self):
            rows, self.rows = self.rows, []
            return rows

    tampered = dict(exact)
    tampered["base_macro_fsm_request_id"] = "wrong-base"
    assert _wait_for_close_receipt(
        _Polling([tampered]), close_event, timeout_s=0.001
    ) == {}
    assert _wait_for_close_receipt(
        _Polling([exact]), close_event, timeout_s=0.1
    ) == exact


def _residual_start_kwargs():
    return {
        "request_id": "outer-request",
        "request_identity_sha256": "a" * 64,
        "worker_session_id": "worker-session",
        "source_version": "v003_20260805_224517_157723_manual",
        "profile_id": "fsm50-gate-c-successful-recording-profiles-v1",
        "graph_id": "fsm50-recording-derived-macro-v1",
        "graph_sha256": "b" * 64,
        "profile_library_sha256": "c" * 64,
        "bundle_sha256": "d" * 64,
        "policy_kind": "ZERO",
        "policy_sha256": "e" * 64,
        "residual_core_sha256": "f" * 64,
        "envelope_canonical_sha256": "1" * 64,
    }


def test_public_residual_start_method_queues_the_exact_outer_schema():
    client = SimProcessClient(_client_args())
    try:
        client.start_residual_macro_fsm(**_residual_start_kwargs())
        message = client.pending_messages[-1]
        assert set(message) == {
            "type",
            "schema_version",
            "operation",
            "request_id",
            "request_identity_sha256",
            "worker_session_id",
            "source_version",
            "profile_id",
            "graph_id",
            "graph_sha256",
            "profile_library_sha256",
            "bundle_sha256",
            "policy_kind",
            "policy_sha256",
            "residual_core_sha256",
            "envelope_canonical_sha256",
            "enqueued_wall_time",
        }
        assert message["type"] == "start_macro_fsm"
        assert message["schema_version"] == "fsm50.start_residual_macro_fsm.v1"
        assert message["operation"] == "residual_macro_fsm"
        assert message["policy_kind"] == "ZERO"
    finally:
        client.close()


def test_public_residual_start_method_uses_existing_immediate_send_path():
    class _Socket:
        def __init__(self):
            self.payload = bytearray()

        def send(self, payload):
            self.payload.extend(payload)
            return len(payload)

        def close(self):
            pass

    client = SimProcessClient(_client_args())
    socket = _Socket()
    try:
        client.conn = socket
        client.start_residual_macro_fsm(**_residual_start_kwargs())
        assert client.pending_messages == []
        assert client.pending_send_buffer == bytearray()
        message = decode_line(bytes(socket.payload))
        assert message is not None
        assert message["type"] == "start_macro_fsm"
        assert message["request_id"] == "outer-request"
        assert message["policy_kind"] == "ZERO"
    finally:
        client.conn = None
        client.close()


def test_public_residual_start_rejects_nonzero_and_old_start_shape_is_unchanged():
    client = SimProcessClient(_client_args())
    try:
        kwargs = _residual_start_kwargs()
        kwargs["policy_kind"] = "PPO"
        with pytest.raises(ValueError, match="policy_kind=ZERO"):
            client.start_residual_macro_fsm(**kwargs)

        client.start_macro_fsm(
            request_id="base-request",
            worker_session_id="worker-session",
            source_version="v003",
            profile_id="profiles",
            graph_id="graph",
            graph_sha256="a" * 64,
            profile_library_sha256="b" * 64,
            bundle_sha256="c" * 64,
        )
        old = client.pending_messages[-1]
        assert set(old) == {
            "type",
            "request_id",
            "worker_session_id",
            "source_version",
            "profile_id",
            "graph_id",
            "graph_sha256",
            "profile_library_sha256",
            "bundle_sha256",
            "enqueued_wall_time",
        }
        assert "schema_version" not in old
        assert "operation" not in old
        assert "policy_kind" not in old
    finally:
        client.close()
