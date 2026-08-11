"""Pure-Python validation for supervised Isaac shutdown outcomes."""

from __future__ import annotations

from typing import Any, Mapping


SUCCESSFUL_SHUTDOWN_STATUSES = frozenset(
    {"GRACEFUL_EXIT", "FAST_EXIT_VERIFIED", "NORMAL_EXIT"}
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
            legacy_returncode = int(outcome["child_returncode"])
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
        intended = int(outcome["intended_returncode"])
        actual = int(outcome["child_returncode"])
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
        if mode != "fast":
            raise ValueError("FAST_EXIT_VERIFIED shutdown_mode is not fast")
        if state not in {"FAST_EXIT_REQUESTED", "FAST_CLOSE_RETURNED"}:
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
