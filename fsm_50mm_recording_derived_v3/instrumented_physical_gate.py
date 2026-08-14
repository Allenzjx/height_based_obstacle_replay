"""Pure-Python admission gate for three independent instrumented v003 runs.

The gate is intentionally read-only and path explicit.  It does not discover a
``latest`` run, launch Isaac, write a report, or repair incomplete evidence.
Every admitted replay is first passed through the durable artifact loader used
by the environment A/A--A/B gate; this module then adds the stricter physical,
clean-reset, identity, and three-independent-process requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from fsm_50mm_recording_derived_v3.environment_ab_artifacts import (
    ArtifactValidationError,
    ReplayArtifact,
    _strict_json,
    _validate_checksums,
    load_completed_replay_artifact,
)
from fsm_50mm_recording_derived_v3.environment_equivalence import sha256_file
from fsm_50mm_recording_derived_v3.shutdown_contract import (
    validate_shutdown_outcome,
)


SCHEMA_VERSION = "fsm50.instrumented_physical_gate.v1"
ENVIRONMENT_REPORT_SCHEMA = "fsm50.environment_equivalence_report.v1"
EXPECTED_SOURCE_VERSION = "v003_20260805_224517_157723_manual"
WHEEL_INTEGRAL_EVIDENCE_FILENAME = "V003_WHEEL_INTEGRAL_EVIDENCE.json"
REQUIRED_RUN_COUNT = 3
SHA256_HEX = frozenset("0123456789abcdef")
ALLOWED_SHUTDOWN_STATUSES = frozenset({"GRACEFUL_EXIT", "FAST_EXIT_VERIFIED"})
REQUIRED_PHYSICAL_CRITERIA = frozenset(
    {
        "contact_evidence_valid",
        "all_legs_linkage_lift_valid",
        "no_illegal_drive_up",
        "attitude_safe",
        "joint_limits_safe",
        "collision_safe",
        "penetration_safe",
        "contact_drift_safe",
        "final_all_top",
        "final_all_loaded",
        "final_velocity_stable",
    }
)
PHYSICAL_CRITERION_SCOPES = {
    "contact_evidence_valid": "INSTRUMENTED_ONLY",
    "all_legs_linkage_lift_valid": "COMMON",
    "no_illegal_drive_up": "COMMON",
    "attitude_safe": "COMMON",
    "joint_limits_safe": "COMMON",
    "collision_safe": "INSTRUMENTED_ONLY",
    "penetration_safe": "COMMON",
    "contact_drift_safe": "INSTRUMENTED_ONLY",
    "final_all_top": "COMMON",
    "final_all_loaded": "COMMON",
    "final_velocity_stable": "COMMON",
}


class InstrumentedPhysicalGateError(ValueError):
    """The supplied evidence cannot establish the strict physical gate."""

    def __init__(self, failures: str | Sequence[str]):
        rows = [
            str(row)
            for row in ([failures] if isinstance(failures, str) else failures)
            if str(row)
        ]
        self.failures = tuple(rows)
        super().__init__("; ".join(rows) or "instrumented physical gate failed")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InstrumentedPhysicalGateError(f"{label}: expected an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise InstrumentedPhysicalGateError(f"{label}: expected an array")
    return list(value)


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in SHA256_HEX for character in digest):
        raise InstrumentedPhysicalGateError(f"{label}: invalid SHA-256")
    return digest


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise InstrumentedPhysicalGateError(f"{label}: boolean is not an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InstrumentedPhysicalGateError(
            f"{label}: missing or invalid integer"
        ) from exc
    if not math.isfinite(number) or number != int(number) or int(number) <= 0:
        raise InstrumentedPhysicalGateError(
            f"{label}: expected a positive integer"
        )
    return int(number)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _require_true(payload: Mapping[str, Any], key: str, label: str) -> None:
    if payload.get(key) is not True:
        raise InstrumentedPhysicalGateError(f"{label}.{key} is not true")


def _require_false(payload: Mapping[str, Any], key: str, label: str) -> None:
    if payload.get(key) is not False:
        raise InstrumentedPhysicalGateError(f"{label}.{key} is not false")


def _explicit_artifact_path(path: str | Path, label: str) -> Path:
    """Accept an exact run directory or its exact version artifact directory."""

    requested = Path(path).resolve()
    if not requested.is_dir():
        raise InstrumentedPhysicalGateError(
            f"{label}: artifact directory does not exist: {requested}"
        )
    if (requested / "result.json").is_file():
        return requested
    pointer_path = requested / "artifact_pointer.json"
    if not pointer_path.is_file():
        raise InstrumentedPhysicalGateError(
            f"{label}: expected result.json or artifact_pointer.json directly in "
            f"{requested}; recursive/latest discovery is forbidden"
        )
    try:
        pointer = _mapping(_strict_json(pointer_path), str(pointer_path))
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(exc.failures) from exc
    run_text = str(pointer.get("run_dir", "") or "")
    if not run_text:
        raise InstrumentedPhysicalGateError(
            f"{pointer_path}: run_dir pointer is missing"
        )
    run_dir = Path(run_text).resolve()
    try:
        run_dir.relative_to(requested)
    except ValueError as exc:
        raise InstrumentedPhysicalGateError(
            f"{pointer_path}: run_dir escapes the explicit artifact"
        ) from exc
    if not (run_dir / "result.json").is_file():
        raise InstrumentedPhysicalGateError(
            f"{pointer_path}: pointed result.json is missing"
        )
    return requested


def _strict_report(
    environment_report_path: str | Path,
    environment_report_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    report_path = Path(environment_report_path).resolve()
    if not report_path.is_file():
        raise InstrumentedPhysicalGateError(
            f"environment report is missing: {report_path}"
        )
    expected_sha = _sha256(
        environment_report_sha256, "environment_report_sha256"
    )
    actual_sha = sha256_file(report_path).lower()
    if actual_sha != expected_sha:
        raise InstrumentedPhysicalGateError(
            "environment report SHA-256 differs from the explicit trust anchor"
        )
    try:
        report = _mapping(_strict_json(report_path), str(report_path))
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(exc.failures) from exc
    failures: list[str] = []
    if str(report.get("schema_version", "")) != ENVIRONMENT_REPORT_SCHEMA:
        failures.append("environment report schema is unsupported")
    if str(report.get("status", "")) != "PASS":
        failures.append("environment report status is not PASS")
    if report.get("environment_equivalent") is not True:
        failures.append("environment report environment_equivalent is not true")
    for key in ("instrumentation_comparison", "trajectory_comparison"):
        try:
            check = _mapping(report.get(key), f"environment_report.{key}")
        except InstrumentedPhysicalGateError as exc:
            failures.extend(exc.failures)
            continue
        if check.get("ok") is not True:
            failures.append(f"environment report {key}.ok is not true")
    try:
        readback = _mapping(
            report.get("runtime_readback"), "environment_report.runtime_readback"
        )
        if readback.get("ok") is not True:
            failures.append("environment report runtime_readback.ok is not true")
        if readback.get("readback_complete") is not True:
            failures.append(
                "environment report runtime_readback.readback_complete is not true"
            )
        runs = _mapping(readback.get("runs"), "environment_report.runtime_readback.runs")
        b_run = _mapping(runs.get("B"), "environment_report.runtime_readback.runs.B")
        b_provenance = _mapping(
            b_run.get("provenance"),
            "environment_report.runtime_readback.runs.B.provenance",
        )
        if str(b_provenance.get("contact_mode", "")) != "instrumented":
            failures.append("environment report B contact_mode is not instrumented")
    except InstrumentedPhysicalGateError as exc:
        failures.extend(exc.failures)
    try:
        extra = _mapping(report.get("extra"), "environment_report.extra")
        conversion = _mapping(
            extra.get("artifact_conversion"),
            "environment_report.extra.artifact_conversion",
        )
        if conversion.get("ok") is not True:
            failures.append("environment report artifact_conversion.ok is not true")
    except InstrumentedPhysicalGateError as exc:
        failures.extend(exc.failures)
    try:
        _mapping(report.get("static_fingerprint"), "environment_report.static_fingerprint")
    except InstrumentedPhysicalGateError as exc:
        failures.extend(exc.failures)
    if failures:
        raise InstrumentedPhysicalGateError(failures)
    return report_path, actual_sha, report


def _verdict_status(value: Any, label: str) -> str:
    if isinstance(value, str):
        status = value.strip().upper()
    elif isinstance(value, Mapping):
        row = dict(value)
        statuses = [
            str(row[key]).strip().upper()
            for key in ("status", "verdict", "result")
            if key in row and row[key] is not None
        ]
        if not statuses or len(set(statuses)) != 1:
            raise InstrumentedPhysicalGateError(
                f"{label}: missing/conflicting explicit status/verdict/result"
            )
        status = next(iter(set(statuses)))
        if "available" in row and row.get("available") is not True:
            raise InstrumentedPhysicalGateError(f"{label}.available is not true")
        if "availability" in row:
            availability = row.get("availability")
            if isinstance(availability, str):
                available = availability.strip().upper() in {
                    "AVAILABLE",
                    "COMPLETE",
                    "PRESENT",
                }
            else:
                available = availability is True
            if not available:
                raise InstrumentedPhysicalGateError(
                    f"{label}.availability is not available"
                )
        if "passed" in row and row.get("passed") is not True:
            raise InstrumentedPhysicalGateError(f"{label}.passed is not true")
    else:
        raise InstrumentedPhysicalGateError(
            f"{label}: missing explicit PASS/FAIL/NE verdict"
        )
    if status != "PASS":
        raise InstrumentedPhysicalGateError(
            f"{label}: verdict is {status or 'MISSING'}, not PASS"
        )
    return status


def _require_pass_criteria(value: Any, label: str) -> None:
    criteria = _mapping(value, label)
    if not criteria:
        raise InstrumentedPhysicalGateError(f"{label}: criteria are empty")
    failures: list[str] = []
    for name, raw in criteria.items():
        criterion_label = f"{label}.{name}"
        if isinstance(raw, bool):
            if raw is not True:
                failures.append(f"{criterion_label} is false")
            continue
        try:
            _verdict_status(raw, criterion_label)
        except InstrumentedPhysicalGateError as exc:
            failures.extend(exc.failures)
    if failures:
        raise InstrumentedPhysicalGateError(failures)


def _require_v2_physical_criteria(physical: Mapping[str, Any], label: str) -> None:
    criteria = _mapping(physical.get("criteria"), f"{label}.criteria")
    if set(criteria) != set(REQUIRED_PHYSICAL_CRITERIA):
        raise InstrumentedPhysicalGateError(
            f"{label}.criteria does not contain the exact 11 v2 criteria"
        )
    normalized: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name, raw in criteria.items():
        row = _mapping(raw, f"{label}.criteria.{name}")
        normalized[name] = row
        if str(row.get("name", "")) != name:
            failures.append(f"{label}.criteria.{name}.name mismatch")
        if str(row.get("scope", "")) != PHYSICAL_CRITERION_SCOPES[name]:
            failures.append(f"{label}.criteria.{name}.scope is invalid")
        if str(row.get("availability", "")) != "AVAILABLE":
            failures.append(f"{label}.criteria.{name}.availability is not AVAILABLE")
        if row.get("passed") is not True:
            failures.append(f"{label}.criteria.{name}.passed is not true")
        if str(row.get("reason", "") or ""):
            failures.append(f"{label}.criteria.{name}.reason is not empty")

    records = _sequence(physical.get("criterion_records"), f"{label}.criterion_records")
    if len(records) != len(REQUIRED_PHYSICAL_CRITERIA):
        failures.append(f"{label}.criterion_records does not contain exactly 11 rows")
    else:
        listed: dict[str, dict[str, Any]] = {}
        for index, raw in enumerate(records):
            row = _mapping(raw, f"{label}.criterion_records[{index}]")
            name = str(row.get("name", ""))
            if not name or name in listed:
                failures.append(
                    f"{label}.criterion_records[{index}] has a missing/duplicate name"
                )
                continue
            listed[name] = row
        if listed != normalized:
            failures.append(f"{label}.criterion_records differs from criteria")

    strict = _mapping(physical.get("strict_criteria"), f"{label}.strict_criteria")
    if strict != {name: True for name in criteria}:
        failures.append(f"{label}.strict_criteria differs from the PASS records")
    for key in (
        "role_capture_verdict_reasons",
        "full_physical_verdict_reasons",
    ):
        reasons = physical.get(key)
        if not isinstance(reasons, list) or reasons:
            failures.append(f"{label}.{key} is missing/non-empty")
    if failures:
        raise InstrumentedPhysicalGateError(failures)


def _runtime_instance_id(result: Mapping[str, Any], label: str) -> str:
    readiness = _mapping(
        result.get("motion_start_readiness"), f"{label}.motion_start_readiness"
    )
    pre_first = _mapping(
        result.get("motion_start_pre_first_dispatch"),
        f"{label}.motion_start_pre_first_dispatch",
    )
    respawn = _mapping(result.get("respawn"), f"{label}.respawn")
    candidates: list[tuple[str, str]] = []
    for name, payload in (
        ("result", result),
        ("motion_start_readiness", readiness),
        ("motion_start_pre_first_dispatch", pre_first),
        ("respawn", respawn),
    ):
        if payload.get("adapter_runtime_instance_id") is not None:
            candidates.append(
                (name, str(payload.get("adapter_runtime_instance_id", "") or ""))
            )
    for name in ("final", "token_payload"):
        nested = pre_first.get(name)
        if isinstance(nested, Mapping) and nested.get("adapter_runtime_instance_id") is not None:
            candidates.append(
                (
                    f"motion_start_pre_first_dispatch.{name}",
                    str(nested.get("adapter_runtime_instance_id", "") or ""),
                )
            )
    values = {value for _name, value in candidates if value}
    if len(values) != 1 or any(not value for _name, value in candidates):
        raise InstrumentedPhysicalGateError(
            f"{label}: missing/conflicting adapter runtime instance ids: {candidates!r}"
        )
    runtime_id = next(iter(values))
    if len(runtime_id) < 8:
        raise InstrumentedPhysicalGateError(
            f"{label}: adapter runtime instance id is implausibly short"
        )
    return runtime_id


def _require_no_root_state_writes(value: Any, label: str) -> None:
    """Reject contradictory nested clean-reset evidence, not just top flags."""

    count_keys = {
        "root_state_write_count",
        "root_state_write_count_before",
        "root_state_write_count_after",
    }
    false_keys = {
        "root_pose_written",
        "writes_robot_state",
        "state_writes_performed",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_label = f"{label}.{key}"
            if key in count_keys and nested != 0:
                raise InstrumentedPhysicalGateError(
                    f"{nested_label} is not zero"
                )
            if key in false_keys and nested is not False:
                raise InstrumentedPhysicalGateError(
                    f"{nested_label} is not false"
                )
            if key == "root_state_write_events":
                if not isinstance(nested, list) or nested:
                    raise InstrumentedPhysicalGateError(
                        f"{nested_label} is missing/non-empty"
                    )
            _require_no_root_state_writes(nested, nested_label)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_no_root_state_writes(nested, f"{label}[{index}]")


def _validate_reset_readiness_dispatch(
    result: Mapping[str, Any], *, label: str, trial_id: int
) -> str:
    _require_true(result, "fresh_process_clean_reset", label)
    _require_true(result, "motion_start_ready", label)
    _require_true(result, "dispatch_complete", label)

    respawn = _mapping(result.get("respawn"), f"{label}.respawn")
    _require_true(respawn, "ok", f"{label}.respawn")
    _require_false(respawn, "respawned", f"{label}.respawn")
    _require_false(respawn, "root_pose_written", f"{label}.respawn")
    if respawn.get("root_state_write_count") != 0:
        raise InstrumentedPhysicalGateError(
            f"{label}.respawn.root_state_write_count is not zero"
        )
    _require_no_root_state_writes(respawn, f"{label}.respawn")

    readiness = _mapping(
        result.get("motion_start_readiness"), f"{label}.motion_start_readiness"
    )
    _require_true(readiness, "ready", f"{label}.motion_start_readiness")
    _require_false(
        readiness, "writes_robot_state", f"{label}.motion_start_readiness"
    )
    if str(readiness.get("status", "")) != "PASS":
        raise InstrumentedPhysicalGateError(
            f"{label}.motion_start_readiness.status is not PASS"
        )
    if readiness.get("root_state_write_count") != 0:
        raise InstrumentedPhysicalGateError(
            f"{label}.motion_start_readiness.root_state_write_count is not zero"
        )
    _require_no_root_state_writes(
        readiness, f"{label}.motion_start_readiness"
    )

    pre_first = _mapping(
        result.get("motion_start_pre_first_dispatch"),
        f"{label}.motion_start_pre_first_dispatch",
    )
    _require_true(pre_first, "ready", f"{label}.motion_start_pre_first_dispatch")
    _require_false(
        pre_first,
        "writes_robot_state",
        f"{label}.motion_start_pre_first_dispatch",
    )
    if str(pre_first.get("status", "")) != "PASS":
        raise InstrumentedPhysicalGateError(
            f"{label}.motion_start_pre_first_dispatch.status is not PASS"
        )
    if pre_first.get("root_state_write_count") != 0:
        raise InstrumentedPhysicalGateError(
            f"{label}.motion_start_pre_first_dispatch.root_state_write_count is not zero"
        )
    token = _mapping(
        pre_first.get("token_payload"),
        f"{label}.motion_start_pre_first_dispatch.token_payload",
    )
    if str(token.get("source_version", "")) != EXPECTED_SOURCE_VERSION:
        raise InstrumentedPhysicalGateError(
            f"{label}: readiness token source version is not v003"
        )
    if _positive_integer(token.get("trial_id"), f"{label}.token.trial_id") != trial_id:
        raise InstrumentedPhysicalGateError(
            f"{label}: readiness token trial id differs from result"
        )
    if token.get("root_state_write_count") != 0:
        raise InstrumentedPhysicalGateError(
            f"{label}: readiness token root_state_write_count is not zero"
        )
    _require_no_root_state_writes(
        pre_first, f"{label}.motion_start_pre_first_dispatch"
    )

    ledger = _mapping(result.get("dispatch_ledger"), f"{label}.dispatch_ledger")
    _require_true(ledger, "complete", f"{label}.dispatch_ledger")
    if str(ledger.get("schema_version", "")) != "fsm50.source_dispatch_ledger.v1":
        raise InstrumentedPhysicalGateError(
            f"{label}: dispatch ledger schema is unsupported"
        )
    if str(ledger.get("source_version", "")) != EXPECTED_SOURCE_VERSION:
        raise InstrumentedPhysicalGateError(
            f"{label}: dispatch ledger source version is not v003"
        )
    errors = ledger.get("errors")
    if not isinstance(errors, list) or errors:
        raise InstrumentedPhysicalGateError(
            f"{label}: dispatch ledger errors are missing/non-empty"
        )
    _require_true(
        ledger, "one_motion_batch_per_physics_tick", f"{label}.dispatch_ledger"
    )
    if str(ledger.get("motion_start_readiness_token", "")).lower() != str(
        pre_first.get("readiness_token_sha256", "")
    ).lower():
        raise InstrumentedPhysicalGateError(
            f"{label}: dispatch/readiness token identity mismatch"
        )
    return _runtime_instance_id(result, label)


def _validate_physical_pass(
    result: Mapping[str, Any], *, label: str, telemetry_count: int
) -> tuple[str, str]:
    for key in (
        "artifact_valid",
        "scheduler_complete",
        "physical_success",
        "strict_full_success",
    ):
        _require_true(result, key, label)
    if str(result.get("scheduler_stop_reason", "")) != "complete":
        raise InstrumentedPhysicalGateError(
            f"{label}.scheduler_stop_reason is not complete"
        )
    if str(result.get("classification", "")) != "FULL_SUCCESS":
        raise InstrumentedPhysicalGateError(
            f"{label}.classification is not FULL_SUCCESS"
        )
    if result.get("timed_out") is True or result.get("simulation_app_stopped") is True:
        raise InstrumentedPhysicalGateError(f"{label}: replay timed out/stopped")
    lifecycle = _mapping(result.get("lifecycle"), f"{label}.lifecycle")
    _require_true(lifecycle, "finalized", f"{label}.lifecycle")
    _require_false(lifecycle, "failed", f"{label}.lifecycle")
    _require_true(lifecycle, "strict_success", f"{label}.lifecycle")

    physical = _mapping(result.get("physical_evidence"), f"{label}.physical_evidence")
    if str(physical.get("schema_version", "")) != "fsm50.physical_evidence.v2":
        raise InstrumentedPhysicalGateError(
            f"{label}: physical evidence schema is not fsm50.physical_evidence.v2"
        )
    if str(physical.get("contact_mode", "")) != "instrumented":
        raise InstrumentedPhysicalGateError(
            f"{label}: physical evidence contact_mode is not instrumented"
        )
    _require_true(physical, "physical_success", f"{label}.physical_evidence")
    _require_true(physical, "evidence_complete", f"{label}.physical_evidence")
    if str(physical.get("source_version", "")) != EXPECTED_SOURCE_VERSION:
        raise InstrumentedPhysicalGateError(
            f"{label}: physical evidence source version is not v003"
        )
    if _positive_integer(
        physical.get("sample_count"), f"{label}.physical_evidence.sample_count"
    ) != telemetry_count:
        raise InstrumentedPhysicalGateError(
            f"{label}: physical sample_count differs from telemetry"
        )
    reasons = physical.get("not_evaluable_reasons")
    if not isinstance(reasons, list) or reasons:
        raise InstrumentedPhysicalGateError(
            f"{label}: physical not_evaluable_reasons is missing/non-empty"
        )
    role_status = _verdict_status(
        physical.get("role_capture_verdict"),
        f"{label}.physical_evidence.role_capture_verdict",
    )
    full_status = _verdict_status(
        physical.get("full_physical_verdict"),
        f"{label}.physical_evidence.full_physical_verdict",
    )
    for key, expected in (
        ("role_capture_verdict", role_status),
        ("full_physical_verdict", full_status),
    ):
        if key in result and _verdict_status(result.get(key), f"{label}.{key}") != expected:
            raise InstrumentedPhysicalGateError(
                f"{label}.{key} differs from physical evidence"
            )
    _require_v2_physical_criteria(physical, f"{label}.physical_evidence")
    for optional_key in (
        "criterion_status",
        "role_criteria",
        "full_physical_criteria",
    ):
        if optional_key in physical:
            _require_pass_criteria(
                physical.get(optional_key),
                f"{label}.physical_evidence.{optional_key}",
            )
    return role_status, full_status


def _validate_wheel_integral_pass(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_checksums_sha256: Any,
    label: str,
) -> dict[str, Any]:
    if result.get("wheel_target_integral_complete") is not True:
        raise InstrumentedPhysicalGateError(
            f"{label}.wheel_target_integral_complete is not true"
        )
    if str(result.get("wheel_target_integral_verdict", "")) != "PASS":
        raise InstrumentedPhysicalGateError(
            f"{label}.wheel_target_integral_verdict is not PASS"
        )
    embedded = _mapping(
        result.get("wheel_integral_evidence"), f"{label}.wheel_integral_evidence"
    )
    if str(embedded.get("target_integral_verdict", "")) != "PASS":
        raise InstrumentedPhysicalGateError(
            f"{label}.wheel_integral_evidence.target_integral_verdict is not PASS"
        )
    for key in ("structural_errors", "target_not_evaluable_reasons"):
        rows = embedded.get(key)
        if not isinstance(rows, list) or rows:
            raise InstrumentedPhysicalGateError(
                f"{label}.wheel_integral_evidence.{key} is missing/non-empty"
            )

    evidence_path = run_dir / WHEEL_INTEGRAL_EVIDENCE_FILENAME
    if not evidence_path.is_file():
        raise InstrumentedPhysicalGateError(
            f"{label}: {WHEEL_INTEGRAL_EVIDENCE_FILENAME} is missing"
        )
    try:
        durable = _mapping(_strict_json(evidence_path), str(evidence_path))
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(exc.failures) from exc
    if durable != embedded:
        raise InstrumentedPhysicalGateError(
            f"{label}: {WHEEL_INTEGRAL_EVIDENCE_FILENAME} differs from "
            "result.wheel_integral_evidence"
        )
    try:
        manifest = _validate_checksums(run_dir, [evidence_path])
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(
            [f"{label}: {failure}" for failure in exc.failures]
        ) from exc
    relative = evidence_path.relative_to(run_dir).as_posix()
    actual_sha = sha256_file(evidence_path).lower()
    checksum_manifest_path = run_dir / "checksums.sha256"
    checksum_manifest_sha = sha256_file(checksum_manifest_path).lower()
    if checksum_manifest_sha != _sha256(
        expected_checksums_sha256, f"{label}.loader_checksums_sha256"
    ):
        raise InstrumentedPhysicalGateError(
            f"{label}: checksum manifest changed after reliable loader validation"
        )
    if manifest.get(relative) != actual_sha:
        # _validate_checksums already enforces this.  Keep the explicit branch
        # so the gate output can never be formed from unbound loader evidence.
        raise InstrumentedPhysicalGateError(
            f"{label}: loader checksum coverage for {relative} is missing/stale"
        )
    return {
        "path": str(evidence_path.resolve()),
        "sha256": actual_sha,
        "loader_checksum_relative_path": relative,
        "loader_checksum_covered": True,
        "loader_checksum_manifest_sha256": checksum_manifest_sha,
        "target_integral_verdict": "PASS",
        "wheel_target_integral_complete": True,
        "structural_error_count": 0,
        "target_not_evaluable_reason_count": 0,
    }


def _artifact_identities(artifact: ReplayArtifact) -> dict[str, Any]:
    try:
        physics_dt_s = float(artifact.provenance.get("physics_dt_s"))
        runtime_dt_s = float(artifact.runtime_environment.get("physics_dt_s"))
        scene_config = _mapping(
            artifact.runtime_environment.get("scene_config"),
            "artifact.runtime_environment.scene_config",
        )
        scene_dt_s = float(scene_config.get("physics_dt"))
        render_interval = int(artifact.provenance.get("render_interval"))
    except (TypeError, ValueError) as exc:
        raise InstrumentedPhysicalGateError(
            "artifact physics-dt/render identity is missing or invalid"
        ) from exc
    expected_dt_s = 1.0 / 120.0
    if not all(
        math.isclose(value, expected_dt_s, rel_tol=0.0, abs_tol=1.0e-15)
        for value in (physics_dt_s, runtime_dt_s, scene_dt_s)
    ):
        raise InstrumentedPhysicalGateError(
            "artifact physics dt is not consistently 1/120 s"
        )
    if render_interval <= 0:
        raise InstrumentedPhysicalGateError(
            "artifact render interval is not positive"
        )
    environment_identity = {
        "environment_equivalence": artifact.runtime_environment.get(
            "environment_equivalence"
        ),
        "scene_config": artifact.runtime_environment.get("scene_config"),
        "live_obstacle_geometry": artifact.runtime_environment.get(
            "live_obstacle_geometry"
        ),
        "motion_reference": artifact.runtime_environment.get("motion_reference"),
    }
    physics_identity = {
        "physics_dt_s": physics_dt_s,
        "render_interval": render_interval,
        "scene_config": scene_config,
        "runtime_versions": artifact.provenance.get("runtime_versions"),
    }
    source_identity = {
        "source_version": artifact.provenance.get("source_version"),
        "source_git_head": artifact.provenance.get("source_git_head"),
        "source_files_sha256": artifact.provenance.get("source_files_sha256"),
    }
    return {
        "accepted_steps_sha256": _sha256(
            artifact.provenance.get("accepted_steps_sha256"),
            "artifact.accepted_steps_sha256",
        ),
        "plan_sha256": _sha256(
            artifact.provenance.get("plan_sha256"), "artifact.plan_sha256"
        ),
        "source_identity_sha256": _canonical_sha256(source_identity),
        "environment_identity_sha256": _canonical_sha256(environment_identity),
        "physics_identity_sha256": _canonical_sha256(physics_identity),
        "source_git_head": str(artifact.provenance.get("source_git_head", "")),
        "source_files_sha256": _sha256(
            artifact.provenance.get("source_files_sha256"),
            "artifact.source_files_sha256",
        ),
        "physics_dt_s": physics_dt_s,
        "render_interval": render_interval,
        "runtime_versions": artifact.provenance.get("runtime_versions"),
    }


def _report_b_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    readback = _mapping(report.get("runtime_readback"), "environment_report.runtime_readback")
    runs = _mapping(readback.get("runs"), "environment_report.runtime_readback.runs")
    b_run = _mapping(runs.get("B"), "environment_report.runtime_readback.runs.B")
    provenance = _mapping(
        b_run.get("provenance"), "environment_report.runtime_readback.runs.B.provenance"
    )
    artifact_text = str(b_run.get("artifact_root", "") or "")
    run_text = str(b_run.get("run_dir", "") or "")
    if not artifact_text or not run_text:
        raise InstrumentedPhysicalGateError(
            "environment report B artifact_root/run_dir binding is missing"
        )
    return {
        "artifact_root": str(Path(artifact_text).resolve()),
        "run_dir": str(Path(run_text).resolve()),
        "provenance": provenance,
    }


def _require_common_identity(
    run_rows: Sequence[Mapping[str, Any]], report_b: Mapping[str, Any]
) -> dict[str, Any]:
    identity_fields = (
        "accepted_steps_sha256",
        "plan_sha256",
        "source_identity_sha256",
        "environment_identity_sha256",
        "physics_identity_sha256",
        "source_git_head",
        "source_files_sha256",
        "physics_dt_s",
        "render_interval",
        "runtime_versions",
    )
    failures: list[str] = []
    for field in identity_fields:
        values = [_canonical(row["identity"][field]) for row in run_rows]
        if len(set(values)) != 1:
            failures.append(f"{field} differs across the three runs")
    if failures:
        raise InstrumentedPhysicalGateError(failures)
    common = dict(run_rows[0]["identity"])

    provenance = _mapping(report_b.get("provenance"), "environment_report.B.provenance")
    report_pairs = {
        "accepted_steps_sha256": provenance.get("accepted_steps_sha256"),
        "plan_sha256": provenance.get("plan_sha256"),
        "source_git_head": provenance.get("source_git_head"),
        "source_files_sha256": provenance.get("source_files_sha256"),
        "physics_dt_s": provenance.get("physics_dt_s"),
        "render_interval": provenance.get("render_interval"),
        "runtime_versions": provenance.get("runtime_versions"),
    }
    for field, value in report_pairs.items():
        if _canonical(value) != _canonical(common[field]):
            failures.append(
                f"environment report B {field} differs from the instrumented gate runs"
            )
    if str(provenance.get("source_version", "")) != EXPECTED_SOURCE_VERSION:
        failures.append("environment report B source_version is not v003")
    if str(provenance.get("contact_mode", "")) != "instrumented":
        failures.append("environment report B contact_mode is not instrumented")
    if failures:
        raise InstrumentedPhysicalGateError(failures)
    return common


def _load_one(
    explicit_path: Path,
    *,
    index: int,
) -> tuple[ReplayArtifact, dict[str, Any]]:
    label = f"run[{index}]"
    try:
        artifact = load_completed_replay_artifact(explicit_path, role=f"I{index + 1}")
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(
            [f"{label}: {failure}" for failure in exc.failures]
        ) from exc
    if (explicit_path / "result.json").is_file():
        if artifact.run_dir.resolve() != explicit_path:
            raise InstrumentedPhysicalGateError(
                f"{label}: loader resolved a different run than the explicit path"
            )
    elif artifact.artifact_root.resolve() != explicit_path:
        raise InstrumentedPhysicalGateError(
            f"{label}: loader resolved a different artifact than the explicit path"
        )

    result_path = artifact.run_dir / "result.json"
    try:
        durable_result = _mapping(_strict_json(result_path), str(result_path))
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(exc.failures) from exc
    if durable_result != artifact.result:
        raise InstrumentedPhysicalGateError(
            f"{label}: result.json changed or differs from the validated result"
        )
    result = artifact.result
    if str(result.get("source_version", "")) != EXPECTED_SOURCE_VERSION:
        raise InstrumentedPhysicalGateError(f"{label}: source is not the exact v003 recording")
    if str(result.get("contact_mode", "")) != "instrumented":
        raise InstrumentedPhysicalGateError(f"{label}: result contact_mode is not instrumented")
    if str(artifact.provenance.get("contact_mode", "")) != "instrumented":
        raise InstrumentedPhysicalGateError(
            f"{label}: validated provenance contact_mode is not instrumented"
        )
    if str(artifact.runtime_environment.get("contact_mode", "")) != "instrumented":
        raise InstrumentedPhysicalGateError(
            f"{label}: runtime contact_mode is not instrumented"
        )

    gate1_identity = {
        "environment_equivalence_role": "",
        "environment_equivalence_diagnostic": False,
        "environment_equivalence_diagnostic_complete": False,
        "diagnostic_role": "",
        "ordinary_ui_diagnostic": False,
        "ordinary_ui_diagnostic_complete": False,
        "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
        "gate1_eligible": True,
        "gate1_physical_qualification_eligible": True,
        "environment_equivalence_eligible": False,
        "physical_qualification_eligible": True,
    }
    for field, expected in gate1_identity.items():
        if _canonical(result.get(field)) != _canonical(expected):
            raise InstrumentedPhysicalGateError(
                f"{label}: result {field} is not the exact Gate-1 eligibility value"
            )
    runtime_gate1_identity = {
        "environment_equivalence_role": "",
        "diagnostic_role": "",
        "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
        "gate1_eligible": True,
        "gate1_physical_qualification_eligible": True,
        "environment_equivalence_eligible": False,
    }
    for field, expected in runtime_gate1_identity.items():
        if _canonical(artifact.runtime_environment.get(field)) != _canonical(expected):
            raise InstrumentedPhysicalGateError(
                f"{label}: runtime {field} is not the exact Gate-1 eligibility value"
            )

    trial_id = _positive_integer(result.get("trial_id"), f"{label}.trial_id")
    runtime_instance_id = _validate_reset_readiness_dispatch(
        result, label=label, trial_id=trial_id
    )
    wheel_integral = _validate_wheel_integral_pass(
        result,
        run_dir=artifact.run_dir,
        expected_checksums_sha256=artifact.provenance.get("checksums_sha256"),
        label=label,
    )
    role_status, full_status = _validate_physical_pass(
        result, label=label, telemetry_count=len(artifact.telemetry_rows)
    )

    batch_root = Path(str(artifact.provenance.get("batch_root", "") or "")).resolve()
    if not batch_root.is_dir():
        raise InstrumentedPhysicalGateError(f"{label}: owning batch root is missing")
    batch_request_path = batch_root / "batch_request.json"
    shutdown_path = batch_root / "shutdown_outcome.json"
    try:
        batch_request = _mapping(_strict_json(batch_request_path), str(batch_request_path))
        shutdown = _mapping(_strict_json(shutdown_path), str(shutdown_path))
    except ArtifactValidationError as exc:
        raise InstrumentedPhysicalGateError(exc.failures) from exc
    if _positive_integer(
        batch_request.get("trial_id"), f"{label}.batch_request.trial_id"
    ) != trial_id:
        raise InstrumentedPhysicalGateError(
            f"{label}: batch request trial id differs from result"
        )
    batch_args = _mapping(
        batch_request.get("args"), f"{label}.batch_request.args"
    )
    batch_gate1_identity = {
        "environment_equivalence_role": "",
        "diagnostic_role": "",
        "qualification_scope": "GATE1_PHYSICAL_QUALIFICATION",
        "gate1_physical_qualification_eligible": True,
        "environment_equivalence_eligible": False,
    }
    for field, expected in batch_gate1_identity.items():
        if _canonical(batch_request.get(field)) != _canonical(expected):
            raise InstrumentedPhysicalGateError(
                f"{label}: batch request {field} is not the exact Gate-1 eligibility value"
            )
    for field, expected in {
        "environment_equivalence_role": "",
        "diagnostic_role": "",
        "contact_mode": "instrumented",
    }.items():
        if _canonical(batch_args.get(field)) != _canonical(expected):
            raise InstrumentedPhysicalGateError(
                f"{label}: batch request args {field} is not the exact Gate-1 value"
            )
    try:
        normalized_shutdown = validate_shutdown_outcome(
            shutdown, allow_legacy_normal_exit=False
        )
    except (TypeError, ValueError) as exc:
        raise InstrumentedPhysicalGateError(f"{label}: {exc}") from exc
    shutdown_status = str(normalized_shutdown.get("status", ""))
    if shutdown_status not in ALLOWED_SHUTDOWN_STATUSES:
        raise InstrumentedPhysicalGateError(
            f"{label}: shutdown status is not GRACEFUL_EXIT/FAST_EXIT_VERIFIED"
        )
    if (
        normalized_shutdown.get("process_returned_normally") is not True
        or int(normalized_shutdown.get("intended_returncode", -1)) != 0
        or int(normalized_shutdown.get("child_returncode", -1)) != 0
    ):
        raise InstrumentedPhysicalGateError(
            f"{label}: successful gate requires a normal zero process return"
        )
    closure = _mapping(
        artifact.provenance.get("batch_shutdown_closure"),
        f"{label}.batch_shutdown_closure",
    )
    if (
        str(closure.get("status", "")) != shutdown_status
        or str(closure.get("phase", "")) != "SHUTDOWN_COMPLETE"
    ):
        raise InstrumentedPhysicalGateError(
            f"{label}: shutdown closure identity/status mismatch"
        )
    live_checksum_count = _positive_integer(
        closure.get("live_checksum_entry_count"),
        f"{label}.batch_shutdown_closure.live_checksum_entry_count",
    )
    preclose_checksum_count = _positive_integer(
        closure.get("preclose_checksum_entry_count"),
        f"{label}.batch_shutdown_closure.preclose_checksum_entry_count",
    )
    preclose_evidence = _mapping(
        closure.get("preclose_evidence_files"),
        f"{label}.batch_shutdown_closure.preclose_evidence_files",
    )
    if not preclose_evidence:
        raise InstrumentedPhysicalGateError(
            f"{label}: preclose evidence-file closure is empty"
        )

    source_closure = _mapping(
        artifact.provenance.get("source_closure"), f"{label}.source_closure"
    )
    if str(source_closure.get("git_head", "")) != str(
        artifact.provenance.get("source_git_head", "")
    ) or str(source_closure.get("current_git_head", "")) != str(
        artifact.provenance.get("source_git_head", "")
    ):
        raise InstrumentedPhysicalGateError(
            f"{label}: frozen/current source git identity mismatch"
        )
    source_closure_hashes = {
        key: _sha256(source_closure.get(key), f"{label}.source_closure.{key}")
        for key in (
            "source_freeze_pre_sha256",
            "source_freeze_post_sha256",
            "source_integrity_sha256",
        )
    }
    if _sha256(
        source_closure.get("files_sha256"), f"{label}.source_closure.files_sha256"
    ) != _sha256(
        artifact.provenance.get("source_files_sha256"),
        f"{label}.source_files_sha256",
    ):
        raise InstrumentedPhysicalGateError(
            f"{label}: source closure file-map identity mismatch"
        )

    telemetry = _mapping(
        artifact.provenance.get("telemetry_finalization"),
        f"{label}.telemetry_finalization",
    )
    viewport = _mapping(
        artifact.provenance.get("viewport_video"), f"{label}.viewport_video"
    )
    row = {
        "ordinal": index + 1,
        "explicit_artifact_path": str(explicit_path),
        "artifact_root": str(artifact.artifact_root.resolve()),
        "run_dir": str(artifact.run_dir.resolve()),
        "batch_root": str(batch_root),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path).lower(),
        "trial_id": trial_id,
        "adapter_runtime_instance_id": runtime_instance_id,
        "source_version": EXPECTED_SOURCE_VERSION,
        "contact_mode": "instrumented",
        "role_capture_verdict": role_status,
        "full_physical_verdict": full_status,
        "artifact_valid": True,
        "physical_success": True,
        "strict_full_success": True,
        "scheduler_complete": True,
        "motion_start_ready": True,
        "dispatch_complete": True,
        "wheel_integral": wheel_integral,
        "shutdown_status": shutdown_status,
        "shutdown_outcome_sha256": sha256_file(shutdown_path).lower(),
        "batch_shutdown_closure_sha256": _sha256(
            closure.get("closure_sha256"), f"{label}.closure_sha256"
        ),
        "preclose_checksum_closure": {
            "live_checksum_entry_count": live_checksum_count,
            "preclose_checksum_entry_count": preclose_checksum_count,
            "preclose_evidence_file_count": len(preclose_evidence),
        },
        "source_closure": {
            "git_head": str(source_closure["git_head"]),
            "files_sha256": str(source_closure["files_sha256"]),
            **source_closure_hashes,
        },
        "telemetry": {
            "sample_count": len(artifact.telemetry_rows),
            "fsm50_telemetry_sha256": _sha256(
                artifact.provenance.get("artifact_hashes", {}).get(
                    "fsm50_telemetry.jsonl"
                ),
                f"{label}.fsm50_telemetry_sha256",
            ),
            "finalization_marker_sha256": _sha256(
                telemetry.get("marker_sha256"),
                f"{label}.telemetry_finalization.marker_sha256",
            ),
            "checksums_sha256": _sha256(
                artifact.provenance.get("checksums_sha256"),
                f"{label}.checksums_sha256",
            ),
        },
        "video": {
            "manifest_sha256": _sha256(
                viewport.get("manifest_sha256"), f"{label}.video.manifest_sha256"
            ),
            "video_sha256": _sha256(
                viewport.get("video_sha256"), f"{label}.video.video_sha256"
            ),
            "frame_count": _positive_integer(
                viewport.get("frame_count"), f"{label}.video.frame_count"
            ),
        },
        "identity": _artifact_identities(artifact),
    }
    return artifact, row


def build_instrumented_physical_gate(
    *,
    artifact_paths: Sequence[str | Path],
    environment_report_path: str | Path,
    environment_report_sha256: str,
) -> dict[str, Any]:
    """Read and admit exactly three explicit instrumented v003 artifacts.

    The returned object is deterministic for fixed artifact bytes.  The
    function returns only ``status=PASS``; malformed, missing, duplicated,
    tampered, ``NE``, or ``FAIL`` evidence raises
    :class:`InstrumentedPhysicalGateError`.
    """

    paths = _sequence(artifact_paths, "artifact_paths")
    if len(paths) != REQUIRED_RUN_COUNT:
        raise InstrumentedPhysicalGateError(
            f"artifact_paths must contain exactly {REQUIRED_RUN_COUNT} explicit paths"
        )
    explicit_paths = [
        _explicit_artifact_path(path, f"artifact_paths[{index}]")
        for index, path in enumerate(paths)
    ]
    if len(set(explicit_paths)) != REQUIRED_RUN_COUNT:
        raise InstrumentedPhysicalGateError("artifact paths are duplicated")

    report_path, report_sha, report = _strict_report(
        environment_report_path, environment_report_sha256
    )
    report_b = _report_b_binding(report)

    artifacts: list[ReplayArtifact] = []
    run_rows: list[dict[str, Any]] = []
    for index, explicit_path in enumerate(explicit_paths):
        artifact, row = _load_one(explicit_path, index=index)
        artifacts.append(artifact)
        run_rows.append(row)

    distinct_fields = {
        "run_dir": [row["run_dir"] for row in run_rows],
        "batch_root": [row["batch_root"] for row in run_rows],
        "result_sha256": [row["result_sha256"] for row in run_rows],
        "trial_id": [row["trial_id"] for row in run_rows],
        "adapter_runtime_instance_id": [
            row["adapter_runtime_instance_id"] for row in run_rows
        ],
    }
    failures = [
        f"{field} is not distinct across all three runs"
        for field, values in distinct_fields.items()
        if len(set(values)) != REQUIRED_RUN_COUNT
    ]
    if failures:
        raise InstrumentedPhysicalGateError(failures)

    common_identity = _require_common_identity(run_rows, report_b)
    b_run_dir = str(report_b["run_dir"])
    b_artifact_root = str(report_b["artifact_root"])
    b_matches: list[int] = []
    for row in run_rows:
        run_match = row["run_dir"] == b_run_dir
        artifact_match = row["artifact_root"] == b_artifact_root
        if run_match != artifact_match:
            raise InstrumentedPhysicalGateError(
                "environment report B run_dir/artifact_root binding is inconsistent"
            )
        if run_match:
            b_matches.append(int(row["ordinal"]))
    if len(b_matches) > 1:
        raise InstrumentedPhysicalGateError(
            "environment report B was counted more than once"
        )

    report_static = _mapping(
        report.get("static_fingerprint"), "environment_report.static_fingerprint"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "gate_passed": True,
        "fail_closed": True,
        "required_run_count": REQUIRED_RUN_COUNT,
        "admitted_run_count": REQUIRED_RUN_COUNT,
        "source_version": EXPECTED_SOURCE_VERSION,
        "environment_report": {
            "path": str(report_path),
            "sha256": report_sha,
            "schema_version": ENVIRONMENT_REPORT_SCHEMA,
            "status": "PASS",
            "environment_equivalent": True,
            "static_fingerprint_sha256": _canonical_sha256(report_static),
            "b_artifact_root": b_artifact_root,
            "b_run_dir": b_run_dir,
            "b_counted_run_ordinals": b_matches,
            "b_counted_at_most_once": True,
        },
        "common_identity": common_identity,
        "distinctness": {
            field: {
                "all_distinct": True,
                "values": values,
            }
            for field, values in distinct_fields.items()
        },
        "runs": run_rows,
    }


def validate_instrumented_physical_gate(
    payload: Mapping[str, Any],
    *,
    artifact_paths: Sequence[str | Path],
    environment_report_path: str | Path,
    environment_report_sha256: str,
) -> dict[str, Any]:
    """Re-read all evidence and require ``payload`` to equal the live gate."""

    supplied = _mapping(payload, "instrumented_physical_gate")
    if str(supplied.get("schema_version", "")) != SCHEMA_VERSION:
        raise InstrumentedPhysicalGateError(
            f"instrumented physical gate schema must be {SCHEMA_VERSION}"
        )
    rebuilt = build_instrumented_physical_gate(
        artifact_paths=artifact_paths,
        environment_report_path=environment_report_path,
        environment_report_sha256=environment_report_sha256,
    )
    try:
        equal = _canonical(supplied) == _canonical(rebuilt)
    except (TypeError, ValueError) as exc:
        raise InstrumentedPhysicalGateError(
            f"instrumented physical gate is not strict finite JSON: {exc}"
        ) from exc
    if not equal:
        raise InstrumentedPhysicalGateError(
            "instrumented physical gate payload differs from current artifact bytes"
        )
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate three explicit, independent instrumented v003 physical runs "
            "without launching Isaac or writing artifacts."
        )
    )
    parser.add_argument("artifacts", nargs=3, type=Path)
    parser.add_argument("--environment-report", required=True, type=Path)
    parser.add_argument("--environment-report-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        gate = build_instrumented_physical_gate(
            artifact_paths=args.artifacts,
            environment_report_path=args.environment_report,
            environment_report_sha256=args.environment_report_sha256,
        )
    except InstrumentedPhysicalGateError as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAIL",
                    "gate_passed": False,
                    "fail_closed": True,
                    "failures": list(exc.failures),
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    print(
        json.dumps(
            gate,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_SHUTDOWN_STATUSES",
    "ENVIRONMENT_REPORT_SCHEMA",
    "EXPECTED_SOURCE_VERSION",
    "InstrumentedPhysicalGateError",
    "PHYSICAL_CRITERION_SCOPES",
    "REQUIRED_PHYSICAL_CRITERIA",
    "REQUIRED_RUN_COUNT",
    "SCHEMA_VERSION",
    "WHEEL_INTEGRAL_EVIDENCE_FILENAME",
    "build_instrumented_physical_gate",
    "validate_instrumented_physical_gate",
]
