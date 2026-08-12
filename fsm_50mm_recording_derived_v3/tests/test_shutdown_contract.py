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
