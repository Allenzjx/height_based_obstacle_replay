from __future__ import annotations

import pytest

from fsm_50mm_recording_derived_v3.shutdown_contract import (
    validate_shutdown_outcome,
)


def modern_outcome(status: str) -> dict:
    fast = status == "FAST_EXIT_VERIFIED"
    return {
        "schema_version": "fsm50.shutdown_outcome.v1",
        "status": status,
        "preclose_observed": True,
        "preclose_verification": {"ok": True},
        "process_returned_normally": True,
        "handshake_state": (
            "FAST_EXIT_REQUESTED" if fast else "GRACEFUL_CLOSE_RETURNED"
        ),
        "shutdown_mode": "fast" if fast else "graceful",
        "close_kwargs": {
            "wait_for_replicator": False,
            "skip_cleanup": fast,
        },
        "runtime_version": "5.1.0.0",
        "intended_returncode": 0 if fast else 1,
        "child_returncode": 0 if fast else 1,
        "close_error": "",
    }


def formal_worker_outcome(*, close_returned: bool = False) -> dict:
    outcome = modern_outcome("FAST_EXIT_VERIFIED")
    worker_pid = 54001
    child_pid = 44001
    session = "formal-session"
    adapter = "formal-adapter"
    artifact_request = "artifact-request"
    artifact_sha = "a" * 64
    shutdown_request = "shutdown-request"
    runtime = "5.1.0.0"
    common = {
        "request_id": shutdown_request,
        "worker_pid": worker_pid,
        "worker_session_id": session,
        "adapter_runtime_instance_id": adapter,
        "artifact_request_id": artifact_request,
        "root_state_write_count": 0,
        "mode": "fast",
        "accepted": True,
        "error": "",
        "close_kwargs": {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        },
        "runtime_version": runtime,
    }
    outcome.update(
        child_pid=child_pid,
        handshake_state="FAST_WORKER_PROCESS_RETURNED",
        formal_worker_pid=worker_pid,
        formal_worker_session_id=session,
        adapter_runtime_instance_id=adapter,
        artifact_request_id=artifact_request,
        artifact_request_sha256=artifact_sha,
        worker_shutdown_request_id=shutdown_request,
        worker_returncode=0,
        worker_process_returned_normally=True,
        worker_shutdown_accepted=True,
        worker_close_requested=True,
        worker_close_returned=close_returned,
        worker_forced_termination=False,
        worker_shutdown_ack={
            **common,
            "type": "operation_ack",
            "operation": "shutdown",
        },
        worker_close_requested_ack={
            **common,
            "type": "close_requested",
        },
        worker_close_returned_ack=(
            {
                **common,
                "type": "close_returned",
            }
            if close_returned
            else {}
        ),
        preclose_verification={
            "ok": True,
            "formal_worker_identity": {
                "worker_pid": worker_pid,
                "worker_session_id": session,
                "adapter_runtime_instance_id": adapter,
                "artifact_request_id": artifact_request,
                "artifact_request_sha256": artifact_sha,
            },
        },
    )
    return outcome


@pytest.mark.parametrize("status", ["FAST_EXIT_VERIFIED", "GRACEFUL_EXIT"])
def test_exact_modern_shutdown_contract(status: str) -> None:
    validated = validate_shutdown_outcome(modern_outcome(status))

    assert validated["status"] == status
    assert validated["contract_kind"] != "legacy_normal_exit"


def test_fast_status_cannot_hide_old_close_returned_handshake() -> None:
    outcome = modern_outcome("FAST_EXIT_VERIFIED")
    outcome["handshake_state"] = "CLOSE_RETURNED"

    with pytest.raises(ValueError, match="handshake state"):
        validate_shutdown_outcome(outcome)


def test_fast_contract_requires_explicit_isaac_5_1_and_matching_return() -> None:
    outcome = modern_outcome("FAST_EXIT_VERIFIED")
    outcome["runtime_version"] = "unknown"
    with pytest.raises(ValueError, match="Isaac 5.1"):
        validate_shutdown_outcome(outcome)


def test_formal_worker_requires_preclose_acks_and_exact_process_return() -> None:
    outcome = formal_worker_outcome()
    validated = validate_shutdown_outcome(outcome)
    assert validated["contract_kind"] == "isaac_5_1_worker_fast_exit"

    for mutation in (
        lambda row: row.update(handshake_state="FAST_CLOSE_RETURNED"),
        lambda row: row.update(worker_forced_termination=True),
        lambda row: row.update(worker_shutdown_accepted=False),
        lambda row: row.update(worker_close_requested=False),
        lambda row: row.pop("worker_close_requested_ack"),
        lambda row: row.pop("formal_worker_pid"),
        lambda row: row.update(formal_worker_pid=0),
        lambda row: row.update(formal_worker_pid=54001.0),
        lambda row: row.update(child_pid=0),
        lambda row: row.update(
            formal_worker_pid=row["child_pid"]
        ),
        lambda row: row["worker_close_requested_ack"].update(
            adapter_runtime_instance_id="other"
        ),
        lambda row: row.update(worker_returncode=False),
        lambda row: row["worker_close_requested_ack"].update(accepted=1),
        lambda row: row.update(worker_close_returned=True),
        lambda row: row["preclose_verification"][
            "formal_worker_identity"
        ].update(worker_session_id="other"),
    ):
        tampered = formal_worker_outcome()
        mutation(tampered)
        with pytest.raises(ValueError, match="formal worker"):
            validate_shutdown_outcome(tampered)

    returned = formal_worker_outcome(close_returned=True)
    assert (
        validate_shutdown_outcome(returned)["contract_kind"]
        == "isaac_5_1_worker_fast_exit"
    )
    returned["worker_close_returned_ack"]["accepted"] = 1
    with pytest.raises(ValueError, match="close_returned"):
        validate_shutdown_outcome(returned)

    outcome = modern_outcome("FAST_EXIT_VERIFIED")
    outcome["child_returncode"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        validate_shutdown_outcome(outcome)


def test_legacy_normal_exit_has_a_separate_contract() -> None:
    outcome = {
        "schema_version": "fsm50.shutdown_outcome.v1",
        "status": "NORMAL_EXIT",
        "preclose_observed": True,
        "handshake_state": "CLOSE_RETURNED",
        "child_returncode": 0,
    }
    assert (
        validate_shutdown_outcome(outcome)["contract_kind"]
        == "legacy_normal_exit"
    )
    with pytest.raises(ValueError, match="legacy NORMAL_EXIT"):
        validate_shutdown_outcome(outcome, allow_legacy_normal_exit=False)

    outcome["child_returncode"] = None
    with pytest.raises(ValueError, match="missing/invalid"):
        validate_shutdown_outcome(outcome)
