"""Pure-Python validation for supervised Isaac shutdown outcomes."""

from __future__ import annotations

import json
from typing import Any, Mapping


SUCCESSFUL_SHUTDOWN_STATUSES = frozenset(
    {"GRACEFUL_EXIT", "FAST_EXIT_VERIFIED", "NORMAL_EXIT"}
)


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact JSON integer")
    return value


def _strict_json_equal(left: Any, right: Any) -> bool:
    try:
        options = {
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": (",", ":"),
            "allow_nan": False,
        }
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def _validate_formal_worker_fast_outcome(outcome: Mapping[str, Any]) -> None:
    worker_pid = _exact_int(
        outcome.get("formal_worker_pid"), "formal worker PID"
    )
    child_pid = _exact_int(outcome.get("child_pid"), "controller child PID")
    session_id = str(outcome.get("formal_worker_session_id", "") or "")
    adapter_id = str(outcome.get("adapter_runtime_instance_id", "") or "")
    artifact_request_id = str(outcome.get("artifact_request_id", "") or "")
    artifact_request_sha256 = str(
        outcome.get("artifact_request_sha256", "") or ""
    ).lower()
    shutdown_request_id = str(
        outcome.get("worker_shutdown_request_id", "") or ""
    )
    runtime_version = str(outcome.get("runtime_version", "") or "")
    if child_pid <= 0 or worker_pid <= 0 or worker_pid == child_pid:
        raise ValueError("formal worker PID is invalid or aliases controller child")
    if not session_id or not adapter_id or not artifact_request_id:
        raise ValueError("formal worker identity is incomplete")
    if (
        len(artifact_request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_request_sha256)
    ):
        raise ValueError("formal worker artifact request SHA is invalid")
    if not shutdown_request_id:
        raise ValueError("formal worker shutdown request id is missing")
    if _exact_int(outcome.get("worker_returncode"), "formal worker return code") != 0:
        raise ValueError("formal worker return code is not zero")
    if outcome.get("worker_process_returned_normally") is not True:
        raise ValueError("formal worker did not return normally")
    if outcome.get("worker_forced_termination") is not False:
        raise ValueError("formal worker forced-termination evidence is not false")
    if (
        str(outcome.get("handshake_state", "") or "")
        != "FAST_WORKER_PROCESS_RETURNED"
    ):
        raise ValueError("formal worker requires FAST_WORKER_PROCESS_RETURNED")
    if not runtime_version.startswith("5.1."):
        raise ValueError("formal worker requires explicit Isaac 5.1 runtime")

    preclose = dict(outcome.get("preclose_verification", {}) or {})
    if dict(preclose.get("formal_worker_identity", {}) or {}) != {
        "worker_pid": worker_pid,
        "worker_session_id": session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": artifact_request_id,
        "artifact_request_sha256": artifact_request_sha256,
    }:
        raise ValueError("formal worker preclose identity differs from shutdown")
    fast_kwargs = {
        "wait_for_replicator": False,
        "skip_cleanup": True,
    }
    common = {
        "request_id": shutdown_request_id,
        "worker_pid": worker_pid,
        "worker_session_id": session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": artifact_request_id,
        "root_state_write_count": 0,
        "mode": "fast",
        "accepted": True,
        "error": "",
        "close_kwargs": fast_kwargs,
        "runtime_version": runtime_version,
    }
    for label, key, specific in (
        ("shutdown", "worker_shutdown_ack", {"type": "operation_ack", "operation": "shutdown"}),
        ("close_requested", "worker_close_requested_ack", {"type": "close_requested"}),
    ):
        row = dict(outcome.get(key, {}) or {})
        for field, expected in {**common, **specific}.items():
            if not _strict_json_equal(row.get(field), expected):
                raise ValueError(
                    f"formal worker {label} ACK {field} mismatch"
                )
    if outcome.get("worker_shutdown_accepted") is not True:
        raise ValueError("formal worker shutdown accepted evidence is not true")
    if outcome.get("worker_close_requested") is not True:
        raise ValueError("formal worker close-requested evidence is not true")
    close_returned_observed = outcome.get("worker_close_returned")
    if type(close_returned_observed) is not bool:
        raise ValueError(
            "formal worker close-returned observation is not an exact boolean"
        )
    close_returned = dict(
        outcome.get("worker_close_returned_ack", {}) or {}
    )
    if close_returned_observed:
        for field, expected in {
            **common,
            "type": "close_returned",
        }.items():
            if not _strict_json_equal(close_returned.get(field), expected):
                raise ValueError(
                    f"formal worker close_returned ACK {field} mismatch"
                )
    elif close_returned:
        raise ValueError(
            "formal worker close_returned ACK exists without observed return"
        )


def validate_shutdown_outcome(
    payload: Mapping[str, Any],
    *,
    allow_legacy_normal_exit: bool = True,
) -> dict[str, Any]:
    """Return a normalized copy only when the exact exit contract is valid."""

    outcome = dict(payload or {})
    if outcome.get("schema_version") != "fsm50.shutdown_outcome.v1":
        raise ValueError("shutdown_outcome schema is invalid")
    status = str(outcome.get("status", "") or "")
    if status not in SUCCESSFUL_SHUTDOWN_STATUSES:
        raise ValueError(
            "shutdown_outcome status is not a verified graceful/fast exit: "
            + status
        )
    if outcome.get("preclose_observed") is not True:
        raise ValueError("shutdown_outcome did not observe PRECLOSE_COMPLETE")

    if status == "NORMAL_EXIT":
        if not allow_legacy_normal_exit:
            raise ValueError("legacy NORMAL_EXIT is not allowed by this gate")
        if str(outcome.get("handshake_state", "") or "") != "CLOSE_RETURNED":
            raise ValueError("legacy NORMAL_EXIT handshake is not CLOSE_RETURNED")
        try:
            legacy_returncode = _exact_int(
                outcome["child_returncode"], "legacy child return code"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "legacy NORMAL_EXIT child return code is missing/invalid"
            ) from exc
        if legacy_returncode != 0:
            raise ValueError("legacy NORMAL_EXIT child return code is not zero")
        normalized = dict(outcome)
        normalized["contract_kind"] = "legacy_normal_exit"
        return normalized

    preclose = outcome.get("preclose_verification")
    if not isinstance(preclose, Mapping) or preclose.get("ok") is not True:
        raise ValueError("modern shutdown lacks successful preclose verification")
    if outcome.get("process_returned_normally") is not True:
        raise ValueError("supervised child did not return normally")
    try:
        intended = _exact_int(
            outcome["intended_returncode"], "shutdown intended return code"
        )
        actual = _exact_int(
            outcome["child_returncode"], "shutdown child return code"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("shutdown return-code evidence is missing/invalid") from exc
    if intended < 0 or actual < 0 or intended != actual:
        raise ValueError(
            f"shutdown intended/actual return code mismatch: {intended}/{actual}"
        )
    if str(outcome.get("close_error", "") or ""):
        raise ValueError("successful shutdown outcome contains close_error")

    mode = str(outcome.get("shutdown_mode", "") or "")
    state = str(outcome.get("handshake_state", "") or "")
    kwargs = dict(outcome.get("close_kwargs", {}) or {})
    if status == "FAST_EXIT_VERIFIED":
        if intended != 0 or actual != 0:
            raise ValueError(
                "FAST_EXIT_VERIFIED requires intended/actual return code zero"
            )
        if mode != "fast":
            raise ValueError("FAST_EXIT_VERIFIED shutdown_mode is not fast")
        formal_worker_markers = (
            outcome.get("formal_worker_pid"),
            outcome.get("formal_worker_session_id"),
            outcome.get("adapter_runtime_instance_id"),
            outcome.get("artifact_request_id"),
            outcome.get("artifact_request_sha256"),
            outcome.get("worker_shutdown_request_id"),
            outcome.get("worker_shutdown_accepted"),
            outcome.get("worker_close_requested"),
            outcome.get("worker_close_returned"),
            outcome.get("worker_shutdown_ack"),
            outcome.get("worker_close_requested_ack"),
            outcome.get("worker_close_returned_ack"),
            dict(outcome.get("preclose_verification", {}) or {}).get(
                "formal_worker_identity"
            ),
        )
        formal_worker = any(bool(value) for value in formal_worker_markers)
        if state not in {
            "FAST_EXIT_REQUESTED",
            "FAST_CLOSE_RETURNED",
            "FAST_WORKER_PROCESS_RETURNED",
        }:
            raise ValueError("FAST_EXIT_VERIFIED handshake state is invalid")
        if kwargs != {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }:
            raise ValueError("FAST_EXIT_VERIFIED close kwargs are invalid")
        runtime_version = str(outcome.get("runtime_version", "") or "")
        if not runtime_version.startswith("5.1."):
            raise ValueError(
                "FAST_EXIT_VERIFIED requires an explicit Isaac 5.1 runtime"
            )
        if formal_worker:
            _validate_formal_worker_fast_outcome(outcome)
            contract_kind = "isaac_5_1_worker_fast_exit"
        else:
            if state == "FAST_WORKER_PROCESS_RETURNED":
                raise ValueError(
                    "FAST_WORKER_PROCESS_RETURNED requires formal worker evidence"
                )
            contract_kind = "isaac_5_1_fast_exit"
    else:
        if mode != "graceful":
            raise ValueError("GRACEFUL_EXIT shutdown_mode is not graceful")
        if state != "GRACEFUL_CLOSE_RETURNED":
            raise ValueError("GRACEFUL_EXIT handshake state is invalid")
        if kwargs != {
            "wait_for_replicator": False,
            "skip_cleanup": False,
        }:
            raise ValueError("GRACEFUL_EXIT close kwargs are invalid")
        contract_kind = "graceful_exit"

    normalized = dict(outcome)
    normalized["contract_kind"] = contract_kind
    return normalized


__all__ = ["SUCCESSFUL_SHUTDOWN_STATUSES", "validate_shutdown_outcome"]
