"""Fail-closed, Isaac-free inventory of the complete 50 mm recording corpus.

The source of truth for membership is the set of directories that physically
exists below ``height_050mm/versions``.  ``active_version.json`` and the global
manifest are audited as pointers; neither is allowed to select a recording.

This module does not mutate recordings and does not claim physical success.
It materializes a deterministic static index with enough detail to audit raw
event timing, command targets, atomic batches, snapshots and wheel integrals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from command_model import (
    DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    resolve_servo_targets_for_command,
    resolve_servo_targets_for_group_part,
    resolve_wheel_name,
)
from sequence_model import apply_command_to_state, event_playback_commands
from sim_state_validation import FULL_VALID, validate_full_sim_pose_state

from .recording_fast_plan import fast_plan_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDING_ROOT = (
    PROJECT_ROOT
    / "saved_height_steps_fsm_reference_v2"
    / "height_050mm"
)
DEFAULT_REPORT_ROOT = MODULE_ROOT / "reports"

AUDIT_SCHEMA = "fsm50.recording_corpus_audit.v1"
INDEX_SCHEMA = "fsm50.recording_corpus_index.v1"
PRIMARY_VERSION_PREFIX = "v003"
PRIMARY_PRIORITY = "PRIMARY_BASELINE"
REFERENCE_PRIORITY = "CROSS_VERSION_REFERENCE"
STATIC_ONLY_STATUS = "NOT_RUN"
EPSILON = 1.0e-8

SERVO_NAMES = tuple(str(name) for name in SERVO_JOINT_NAMES)
WHEEL_NAMES = tuple(str(name) for name in WHEEL_JOINT_NAMES)

METADATA_REQUIRED_FIELDS = frozenset(
    {
        "accepted_steps_sha256",
        "actuator_baseline_id",
        "actuator_command_semantics",
        "command_count",
        "created_at",
        "environment_baseline_id",
        "height_mm",
        "motion_profile_id",
        "motion_profile_mode",
        "obstacle_front_face_x_m",
        "obstacle_length_m",
        "obstacle_width_m",
        "robot_asset_path",
        "robot_asset_sha256",
        "schema_version",
        "step_count",
        "updated_at",
        "version_id",
        "version_name",
    }
)

STEP_REQUIRED_FIELDS = frozenset(
    {
        "command_state_after",
        "command_state_before",
        "duration",
        "events",
        "height_m",
        "height_mm",
        "index",
        "motion_semantics",
        "name",
        "recording_timing",
        "sim_state_after",
        "sim_state_before",
        "type",
        "wheel_stop_status",
    }
)

EVENT_REQUIRED_FIELDS = frozenset(
    {
        "actual_recording_time_s",
        "actuator_command_semantics",
        "command",
        "command_start_sim_time",
        "command_state_after",
        "command_state_before",
        "kind",
        "recording_timing_source",
        "time",
    }
)

MOTION_REQUIRED_FIELDS = frozenset(
    {
        "actual_recording_duration_s",
        "actuator_command_semantics",
        "canonical_wheel_angular_displacement_rad",
        "derived_wheel_angular_displacement_rad",
        "reference_duration_s",
        "servo_records",
        "servo_start_deg",
        "servo_target_deg",
        "wheel_active_duration_clock",
        "wheel_angular_displacement_rad",
        "wheel_displacement_source",
        "wheel_records",
    }
)

TIMING_REQUIRED_FIELDS = frozenset(
    {
        "actual_duration_s",
        "actuator_command_semantics",
        "source",
        "wheel_active_duration_s",
    }
)

INDEX_COLUMNS = (
    "version_id",
    "version_directory",
    "priority",
    "priority_rank",
    "active_pointer",
    "accepted_steps_present",
    "metadata_present",
    "accepted_steps_sha256",
    "metadata_accepted_steps_sha256",
    "sha256_matches",
    "metadata_schema_version",
    "metadata_schema_valid",
    "step_count",
    "metadata_step_count",
    "source_event_count",
    "metadata_command_count",
    "decoded_command_count",
    "servo_command_count",
    "wheel_command_count",
    "atomic_event_count",
    "atomic_valid_count",
    "same_tick_servo_wheel_atomic_count",
    "source_event_index_valid_count",
    "source_event_index_error_count",
    "source_duration_s",
    "timestamp_error_count",
    "duration_error_count",
    "target_state_error_count",
    "snapshot_full_valid_count",
    "snapshot_count",
    "snapshot_incomplete_count",
    "source_wheel_segment_count",
    "fast_wheel_segment_count",
    "wheel_integral_theta_rad_json",
    "wheel_integral_max_error_rad",
    "duplicate_step_count",
    "duplicate_event_count",
    "empty_step_count",
    "empty_wait_count",
    "same_timestamp_overlap_count",
    "snapshot_time_overlap_count",
    "mid_step_wheel_stop_count",
    "missing_required_field_count",
    "nonfinite_numeric_count",
    "schema_drift_count",
    "static_integrity_status",
    "physical_replay_status",
    "issue_codes_json",
)


@dataclass(frozen=True)
class VersionSource:
    version_id: str
    directory: Path
    accepted_steps_path: Path
    metadata_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_cell(value: Any) -> str:
    return _canonical_json(value)


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _int_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _json_safe_value(value: Any) -> Any:
    """Replace non-finite report leaves with null while retaining their paths.

    ``nonfinite_numeric_path_counts`` records every affected source location,
    so this is explicit evidence sanitization and never a numeric fallback.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _close(left: Any, right: Any, tolerance: float = EPSILON) -> bool:
    return bool(
        _finite(left)
        and _finite(right)
        and abs(float(left) - float(right)) <= tolerance
    )


def _version_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.name.split("_", 1)[0].lower()
    digits = "".join(character for character in prefix if character.isdigit())
    return (int(digits) if digits else 10**9, path.name.lower())


def discover_versions(recording_root: Path = DEFAULT_RECORDING_ROOT) -> list[VersionSource]:
    """Enumerate every physical version directory; no version count is fixed."""

    versions_root = Path(recording_root).resolve() / "versions"
    if not versions_root.is_dir():
        raise FileNotFoundError(f"recording versions directory is missing: {versions_root}")
    directories = sorted(
        (path for path in versions_root.iterdir() if path.is_dir()),
        key=_version_sort_key,
    )
    return [
        VersionSource(
            version_id=directory.name,
            directory=directory.resolve(),
            accepted_steps_path=(directory / "accepted_steps.jsonl").resolve(),
            metadata_path=(directory / "metadata.json").resolve(),
        )
        for directory in directories
    ]


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        issues.append(
            {
                "severity": "ERROR",
                "code": "JSON_OBJECT_INVALID",
                "path": str(path),
                "message": str(exc),
            }
        )
        return None, issues
    if not isinstance(value, dict):
        issues.append(
            {
                "severity": "ERROR",
                "code": "JSON_OBJECT_WRONG_TYPE",
                "path": str(path),
                "message": "top-level JSON value is not an object",
            }
        )
        return None, issues
    return value, issues


def _read_jsonl_objects(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [
            {
                "severity": "ERROR",
                "code": "JSONL_READ_FAILED",
                "path": str(path),
                "message": str(exc),
            }
        ]
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            issues.append(
                {
                    "severity": "WARN",
                    "code": "JSONL_BLANK_LINE",
                    "path": str(path),
                    "line_number": line_number,
                    "message": "blank JSONL row",
                }
            )
            continue
        try:
            value = json.loads(raw_line)
        except ValueError as exc:
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "JSONL_ROW_INVALID",
                    "path": str(path),
                    "line_number": line_number,
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(value, dict):
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "JSONL_ROW_WRONG_TYPE",
                    "path": str(path),
                    "line_number": line_number,
                    "message": "JSONL row is not an object",
                }
            )
            continue
        rows.append(value)
    return rows, issues


def _normalized_path(path: str) -> str:
    parts = []
    for part in path.split("."):
        parts.append("[]" if part.isdigit() else part)
    return ".".join(parts)


def _value_anomalies(value: Any, path: str = "") -> tuple[list[str], list[str]]:
    """Return all null paths and non-finite numeric paths."""

    nulls: list[str] = []
    nonfinite: list[str] = []
    if value is None:
        nulls.append(_normalized_path(path or "<root>"))
    elif isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(_normalized_path(path or "<root>"))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            child_nulls, child_nonfinite = _value_anomalies(item, child)
            nulls.extend(child_nulls)
            nonfinite.extend(child_nonfinite)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            child_nulls, child_nonfinite = _value_anomalies(item, child)
            nulls.extend(child_nulls)
            nonfinite.extend(child_nonfinite)
    return nulls, nonfinite


def _key_signature(value: Any) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in value)) if isinstance(value, Mapping) else ()


def _modal_signature(signatures: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    counts = Counter(signatures)
    if not counts:
        return ()
    maximum = max(counts.values())
    return min(signature for signature, count in counts.items() if count == maximum)


def _derive_schema_baseline(
    loaded: Sequence[tuple[VersionSource, dict[str, Any] | None, list[dict[str, Any]]]]
) -> dict[str, Any]:
    metadata_signatures = []
    step_signatures = []
    motion_signatures = []
    timing_signatures = []
    event_signatures: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for _source, metadata, steps in loaded:
        if metadata is not None:
            metadata_signatures.append(_key_signature(metadata))
        for step in steps:
            step_signatures.append(_key_signature(step))
            motion_signatures.append(_key_signature(step.get("motion_semantics")))
            timing_signatures.append(_key_signature(step.get("recording_timing")))
            events = step.get("events")
            if not isinstance(events, list):
                continue
            for event in events:
                if isinstance(event, dict):
                    event_signatures[str(event.get("kind", "") or "<missing>")].append(
                        _key_signature(event)
                    )
    return {
        "metadata_keys": list(_modal_signature(metadata_signatures)),
        "step_keys": list(_modal_signature(step_signatures)),
        "motion_semantics_keys": list(_modal_signature(motion_signatures)),
        "recording_timing_keys": list(_modal_signature(timing_signatures)),
        "event_keys_by_kind": {
            kind: list(_modal_signature(signatures))
            for kind, signatures in sorted(event_signatures.items())
        },
    }


def _schema_difference(
    actual: Any,
    expected: Sequence[str],
    *,
    location: str,
) -> dict[str, Any] | None:
    actual_keys = set(_key_signature(actual))
    expected_keys = {str(key) for key in expected}
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if not missing and not extra:
        return None
    return {"location": location, "missing_keys": missing, "extra_keys": extra}


def _command_channel(command: str) -> str:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return "invalid"
    if not tokens:
        return "empty"
    verb = tokens[0].lower()
    if verb in {"servo", "angle"}:
        return "servo"
    if verb in {"wheel", "wheels", "speed", "w", "a", "s", "d", "x", "stop"}:
        return "wheel"
    if verb in {"wait", "hold", "sleep", "pause"}:
        return "timing"
    if verb in {"record_start", "record_stop"}:
        return "marker"
    if verb == "home":
        return "combined"
    return "other"


def _command_syntax_error(command: str) -> str | None:
    """Validate the command grammar consumed by sequence/playback code."""

    try:
        tokens = shlex.split(str(command))
    except ValueError as exc:
        return f"shell tokenization failed: {exc}"
    if not tokens:
        return "command is empty"
    verb = tokens[0].lower()
    try:
        if verb in {"servo", "angle"}:
            if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
                resolve_servo_targets_for_group_part(tokens[1], tokens[2])
                if not _finite(tokens[3]):
                    return "servo target is non-finite"
                return None
            if len(tokens) == 3:
                resolve_servo_targets_for_command(tokens[1])
                if not _finite(tokens[2]):
                    return "servo target is non-finite"
                return None
            return "servo command has invalid arity"
        if verb in {"wheels", "speed"}:
            if len(tokens) != 3 or not all(_finite(value) for value in tokens[1:]):
                return "left/right wheel command requires two finite speeds"
            return None
        if verb == "wheel":
            if len(tokens) == 2 and tokens[1].lower() == "stop":
                return None
            if len(tokens) != 3 or not _finite(tokens[2]):
                return "wheel command requires a target and one finite speed"
            if _finite(tokens[1]):
                return None
            if tokens[1].lower() == "all":
                return None
            resolve_wheel_name(tokens[1])
            return None
        if verb in {"w", "a", "s", "d", "x", "stop", "home"}:
            return None if len(tokens) == 1 else f"{verb} command has invalid arity"
        if verb in {"wait", "hold", "sleep", "pause"}:
            if len(tokens) != 2 or not _finite(tokens[1]):
                return "timing command requires one finite duration"
            return None
        if verb in {"record_start", "record_stop"}:
            return None if len(tokens) == 1 else f"{verb} marker has invalid arity"
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        return str(exc)
    return f"unknown command verb: {verb}"


def _control_keys(command: str) -> set[str]:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return {f"invalid:{command}"}
    if not tokens:
        return set()
    verb = tokens[0].lower()
    try:
        if verb in {"servo", "angle"}:
            if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
                names = resolve_servo_targets_for_group_part(tokens[1], tokens[2])
            elif len(tokens) == 3:
                names = resolve_servo_targets_for_command(tokens[1])
            else:
                return {f"malformed:{command}"}
            return {f"servo:{name}" for name in names}
        if verb in {"wheels", "speed", "w", "a", "s", "d", "x", "stop"}:
            return {f"wheel:{name}" for name in WHEEL_NAMES}
        if verb == "wheel":
            if len(tokens) >= 2 and tokens[1].lower() in {"all", "stop"}:
                return {f"wheel:{name}" for name in WHEEL_NAMES}
            if len(tokens) == 3 and _finite(tokens[1]):
                return {f"wheel:{name}" for name in WHEEL_NAMES}
            if len(tokens) == 3:
                return {f"wheel:{resolve_wheel_name(tokens[1])}"}
    except (KeyError, TypeError, ValueError):
        return {f"malformed:{command}"}
    return set()


def _wait_duration(command: str) -> float | None:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return None
    if not tokens or tokens[0].lower() not in {"wait", "hold", "sleep", "pause"}:
        return None
    if len(tokens) != 2 or not _finite(tokens[1]):
        return float("nan")
    return float(tokens[1])


def _strict_target_state(value: Any) -> tuple[dict[str, dict[str, float]], list[str]]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {"servos": {}, "wheels": {}}, ["state is not an object"]
    servos = value.get("servos")
    wheels = value.get("wheels")
    if not isinstance(servos, Mapping):
        errors.append("servos is not an object")
        servos = {}
    if not isinstance(wheels, Mapping):
        errors.append("wheels is not an object")
        wheels = {}
    servo_keys = {str(key) for key in servos}
    wheel_keys = {str(key) for key in wheels}
    if servo_keys != set(SERVO_NAMES):
        errors.append(
            "servo keys differ: missing={} extra={}".format(
                sorted(set(SERVO_NAMES) - servo_keys),
                sorted(servo_keys - set(SERVO_NAMES)),
            )
        )
    if wheel_keys != set(WHEEL_NAMES):
        errors.append(
            "wheel keys differ: missing={} extra={}".format(
                sorted(set(WHEEL_NAMES) - wheel_keys),
                sorted(wheel_keys - set(WHEEL_NAMES)),
            )
        )
    normalized_servos: dict[str, float] = {}
    normalized_wheels: dict[str, float] = {}
    for name in SERVO_NAMES:
        if name not in servos or not _finite(servos.get(name)):
            errors.append(f"servos.{name} is missing or non-finite")
        else:
            normalized_servos[name] = float(servos[name])
    for name in WHEEL_NAMES:
        if name not in wheels or not _finite(wheels.get(name)):
            errors.append(f"wheels.{name} is missing or non-finite")
        else:
            normalized_wheels[name] = float(wheels[name])
    return {"servos": normalized_servos, "wheels": normalized_wheels}, errors


def _target_states_equal(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    left_state, left_errors = _strict_target_state(left)
    right_state, right_errors = _strict_target_state(right)
    if left_errors or right_errors:
        return False
    return all(
        abs(left_state[group][name] - right_state[group][name]) <= tolerance
        for group, names in (("servos", SERVO_NAMES), ("wheels", WHEEL_NAMES))
        for name in names
    )


def _state_copy(value: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "servos": {name: float(value["servos"][name]) for name in SERVO_NAMES},
        "wheels": {name: float(value["wheels"][name]) for name in WHEEL_NAMES},
    }


def _atomic_evidence(
    event: Mapping[str, Any],
    *,
    step_index: int,
    event_index: int,
) -> dict[str, Any]:
    commands = event_playback_commands(dict(event))
    channels = {
        _command_channel(command)
        for command in commands
        if _command_syntax_error(command) is None
    }
    atomic = "servo" in channels and "wheel" in channels
    result: dict[str, Any] = {
        "source_step_index": step_index,
        "source_event_index": event_index,
        "atomic_servo_wheel": atomic,
        "valid": False,
        "errors": [],
    }
    if not atomic:
        return result
    errors: list[str] = []
    ack = event.get("batch_ack")
    if not isinstance(ack, Mapping):
        errors.append("batch_ack is missing")
        ack = {}
    if not str(event.get("batch_id", "") or ""):
        errors.append("batch_id is missing")
    if str(event.get("batch_id", "") or "") != str(ack.get("batch_id", "") or ""):
        errors.append("event batch_id differs from batch_ack batch_id")
    if str(ack.get("operation", "") or "") != "apply_motion_batch":
        errors.append("batch operation is not apply_motion_batch")
    if ack.get("servo_applied") is not True or ack.get("wheel_applied") is not True:
        errors.append("batch did not acknowledge both servo and wheel")
    if str(ack.get("error", "") or ""):
        errors.append(f"batch_ack error: {ack.get('error')}")
    applied_step = ack.get("applied_sim_step")
    first_step = ack.get("first_physics_step")
    applied_step_number = _int_or_none(applied_step)
    first_step_number = _int_or_none(first_step)
    event_applied_step_number = _int_or_none(event.get("applied_sim_step"))
    if (
        applied_step_number is None
        or first_step_number is None
        or first_step_number != applied_step_number + 1
    ):
        errors.append("first_physics_step is not applied_sim_step + 1")
    if (
        event_applied_step_number is None
        or applied_step_number is None
        or event_applied_step_number != applied_step_number
    ):
        errors.append("event applied_sim_step differs from batch_ack applied_sim_step")
    event_applied_time = _float_or_none(event.get("applied_sim_time"))
    ack_applied_time = _float_or_none(ack.get("applied_sim_time"))
    batch_applied_time = _float_or_none(ack.get("batch_applied_sim_time"))
    physics_dt = _float_or_none(ack.get("physics_dt_s"))
    if not _close(event_applied_time, ack_applied_time, tolerance=1.0e-10):
        errors.append("event applied_sim_time differs from batch_ack applied_sim_time")
    if not _close(batch_applied_time, ack_applied_time, tolerance=1.0e-10):
        errors.append("batch_applied_sim_time differs from batch_ack applied_sim_time")
    if physics_dt is None or physics_dt <= 0.0:
        errors.append("physics_dt_s is missing, non-finite, or non-positive")
    servo_start = ack.get("servo_motion_start_sim_time")
    wheel_start = ack.get("wheel_motion_start_sim_time")
    if not _close(servo_start, wheel_start, tolerance=1.0e-10):
        errors.append("servo and wheel motion start times differ")
    skew = ack.get("motion_start_skew_s")
    if not _finite(skew) or abs(float(skew)) > 1.0e-10:
        errors.append("motion_start_skew_s is missing, non-finite, or non-zero")
    if (
        ack_applied_time is None
        or physics_dt is None
        or physics_dt <= 0.0
        or not _close(
            servo_start,
            ack_applied_time + physics_dt,
            tolerance=1.0e-10,
        )
        or not _close(
            wheel_start,
            ack_applied_time + physics_dt,
            tolerance=1.0e-10,
        )
    ):
        errors.append("servo/wheel motion start is not applied_sim_time + physics_dt_s")
    applied_targets, applied_errors = _strict_target_state(
        {
            "servos": ack.get("servo_targets_applied"),
            "wheels": ack.get("wheel_targets_applied"),
        }
    )
    canonical_targets, canonical_errors = _strict_target_state(
        {
            "servos": ack.get("canonical_servo_targets_deg"),
            "wheels": ack.get("canonical_wheel_velocity_rad_s"),
        }
    )
    if applied_errors:
        errors.append("batch applied target layout is invalid: " + "; ".join(applied_errors))
    if canonical_errors:
        errors.append("batch canonical target layout is invalid: " + "; ".join(canonical_errors))
    if not applied_errors and not _target_states_equal(
        applied_targets, event.get("command_state_after"), tolerance=1.0e-5
    ):
        errors.append("batch applied targets differ from event command_state_after")
    if (
        not applied_errors
        and not canonical_errors
        and not _target_states_equal(
            canonical_targets, applied_targets, tolerance=1.0e-9
        )
    ):
        errors.append("batch canonical targets differ from applied targets")
    if not canonical_errors and not _target_states_equal(
        canonical_targets, event.get("command_state_after"), tolerance=1.0e-5
    ):
        errors.append("batch canonical targets differ from event command_state_after")
    result.update(
        valid=not errors,
        errors=errors,
        batch_id=str(event.get("batch_id", "") or ""),
        event_applied_sim_step=event_applied_step_number,
        applied_sim_step=applied_step_number,
        first_physics_step=first_step_number,
        event_applied_sim_time=event_applied_time,
        batch_applied_sim_time=ack_applied_time,
        physics_dt_s=physics_dt,
        servo_motion_start_sim_time=float(servo_start) if _finite(servo_start) else None,
        wheel_motion_start_sim_time=float(wheel_start) if _finite(wheel_start) else None,
        motion_start_skew_s=float(skew) if _finite(skew) else None,
    )
    return result


def _source_wheel_segments(
    step: Mapping[str, Any],
    *,
    step_index: int,
) -> tuple[list[dict[str, Any]], dict[str, float | None], list[dict[str, Any]]]:
    """Integrate exact source wheel targets over the recorded event clock."""

    issues: list[dict[str, Any]] = []
    initial, initial_errors = _strict_target_state(step.get("command_state_before"))
    if initial_errors:
        return [], {name: None for name in WHEEL_NAMES}, [
            {
                "severity": "ERROR",
                "code": "STEP_INITIAL_TARGET_INVALID",
                "source_step_index": step_index,
                "message": "; ".join(initial_errors),
            }
        ]
    duration = _float_or_none(step.get("duration"))
    if duration is None or duration < 0.0:
        return [], {name: None for name in WHEEL_NAMES}, [
            {
                "severity": "ERROR",
                "code": "STEP_DURATION_INVALID",
                "source_step_index": step_index,
                "message": "duration is missing, non-finite, or negative",
            }
        ]
    state = _state_copy(initial)
    integral = {name: 0.0 for name in WHEEL_NAMES}
    segments: list[dict[str, Any]] = []
    previous_time = 0.0
    establishing_event_index: int | None = None
    events_value = step.get("events")
    events = events_value if isinstance(events_value, list) else []
    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping) or not _finite(event.get("time")):
            continue
        event_time = min(duration, max(0.0, float(event["time"])))
        interval = event_time - previous_time
        if interval < -EPSILON:
            continue
        interval = max(0.0, interval)
        speeds = {name: float(state["wheels"][name]) for name in WHEEL_NAMES}
        theta = {name: speeds[name] * interval for name in WHEEL_NAMES}
        for name in WHEEL_NAMES:
            integral[name] += theta[name]
        if interval > EPSILON and any(abs(value) > EPSILON for value in speeds.values()):
            segments.append(
                {
                    "source_step_index": step_index,
                    "source_segment_index": len(segments),
                    "source_event_start_index": establishing_event_index,
                    "source_event_end_index": event_index,
                    "start_s": previous_time,
                    "end_s": event_time,
                    "duration_s": interval,
                    "speed_rad_s": speeds,
                    "theta_rad": theta,
                }
            )
        for command in event_playback_commands(dict(event)):
            apply_command_to_state(state, command)
        batch_ack = event.get("batch_ack")
        canonical_wheels = (
            batch_ack.get("canonical_wheel_velocity_rad_s")
            if isinstance(batch_ack, Mapping)
            and isinstance(batch_ack.get("canonical_wheel_velocity_rad_s"), Mapping)
            else event.get("canonical_wheel_velocity_rad_s")
        )
        if (
            isinstance(canonical_wheels, Mapping)
            and {str(key) for key in canonical_wheels} == set(WHEEL_NAMES)
            and all(_finite(canonical_wheels.get(name)) for name in WHEEL_NAMES)
        ):
            # The expanded human-readable command is rounded in a few legacy
            # rows (for example 2.0944), while this map preserves the exact
            # target that was applied.  Wheel integration must use the latter.
            state["wheels"] = {
                name: float(canonical_wheels[name]) for name in WHEEL_NAMES
            }
        if any(_command_channel(command) == "wheel" for command in event_playback_commands(dict(event))):
            establishing_event_index = event_index
        previous_time = event_time
    final_interval = duration - previous_time
    if final_interval >= -EPSILON:
        final_interval = max(0.0, final_interval)
        speeds = {name: float(state["wheels"][name]) for name in WHEEL_NAMES}
        theta = {name: speeds[name] * final_interval for name in WHEEL_NAMES}
        for name in WHEEL_NAMES:
            integral[name] += theta[name]
        if final_interval > EPSILON and any(abs(value) > EPSILON for value in speeds.values()):
            segments.append(
                {
                    "source_step_index": step_index,
                    "source_segment_index": len(segments),
                    "source_event_start_index": establishing_event_index,
                    "source_event_end_index": None,
                    "start_s": previous_time,
                    "end_s": duration,
                    "duration_s": final_interval,
                    "speed_rad_s": speeds,
                    "theta_rad": theta,
                }
            )
    final_state, final_errors = _strict_target_state(step.get("command_state_after"))
    if final_errors or not _target_states_equal(state, final_state):
        issues.append(
            {
                "severity": "ERROR",
                "code": "STEP_TARGET_RECONSTRUCTION_MISMATCH",
                "source_step_index": step_index,
                "message": "; ".join(final_errors) if final_errors else "decoded events do not reproduce command_state_after",
            }
        )
    return segments, integral, issues


def _step_fingerprint(step: Mapping[str, Any]) -> str:
    before, _ = _strict_target_state(step.get("command_state_before"))
    after, _ = _strict_target_state(step.get("command_state_after"))
    duration = _float_or_none(step.get("duration"))
    events_value = step.get("events")
    events = events_value if isinstance(events_value, list) else []
    payload = {
        "duration_s": round(duration, 9) if duration is not None else None,
        "commands": [
            command
            for event in events
            if isinstance(event, dict)
            for command in event_playback_commands(event)
        ],
        "initial_targets": before,
        "final_targets": after,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_event_fingerprint(event: Mapping[str, Any]) -> str:
    event_time = _float_or_none(event.get("time"))
    payload = {
        "time_s": round(event_time, 9) if event_time is not None else None,
        "commands": event_playback_commands(dict(event)),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_wheel_stop(command: str) -> bool:
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return False
    if not tokens:
        return False
    verb = tokens[0].lower()
    if verb in {"stop", "x"}:
        return True
    return bool(
        verb in {"wheel", "wheels", "speed"}
        and len(tokens) >= 2
        and tokens[1].lower() == "stop"
    )


def _required_missing(value: Any, required: Iterable[str]) -> list[str]:
    if not isinstance(value, Mapping):
        return sorted(str(key) for key in required)
    return sorted(
        str(key)
        for key in required
        if str(key) not in value or value.get(str(key)) is None
    )


class RecordingCorpusAudit:
    """Build a dynamic, read-only index for every physical version directory."""

    def __init__(
        self,
        recording_root: Path = DEFAULT_RECORDING_ROOT,
        report_root: Path = DEFAULT_REPORT_ROOT,
        *,
        primary_version_prefix: str = PRIMARY_VERSION_PREFIX,
    ) -> None:
        self.recording_root = Path(recording_root).resolve()
        self.report_root = Path(report_root).resolve()
        self.primary_version_prefix = str(primary_version_prefix).strip().lower()

    def enumerate_versions(self) -> list[VersionSource]:
        return discover_versions(self.recording_root)

    def _active_pointer(self) -> dict[str, Any]:
        path = self.recording_root / "active_version.json"
        if not path.is_file():
            return {"path": str(path), "present": False, "version_id": "", "valid": False}
        value, issues = _read_json_object(path)
        version_id = str((value or {}).get("version_id", "") or "")
        return {
            "path": str(path),
            "present": True,
            "version_id": version_id,
            "valid": bool(value is not None and version_id and not issues),
            "issues": issues,
        }

    def _manifest_inventory(self, disk_ids: Sequence[str]) -> dict[str, Any]:
        path = self.recording_root.parent / "manifest.json"
        if not path.is_file():
            return {
                "path": str(path),
                "present": False,
                "version_ids": [],
                "manifest_only_version_ids": [],
                "disk_only_version_ids": list(disk_ids),
            }
        value, issues = _read_json_object(path)
        heights = (value or {}).get("heights")
        if not isinstance(heights, Mapping):
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "MANIFEST_HEIGHTS_WRONG_TYPE",
                    "path": str(path),
                    "message": "manifest heights is not an object",
                }
            )
            heights = {}
        height_value = heights.get("50")
        if not isinstance(height_value, Mapping):
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "MANIFEST_HEIGHT_ROW_WRONG_TYPE",
                    "path": str(path),
                    "message": "manifest heights.50 is not an object",
                }
            )
            height_value = {}
        height_row = dict(height_value)
        rows_value = height_row.get("versions")
        if not isinstance(rows_value, list):
            issues.append(
                {
                    "severity": "ERROR",
                    "code": "MANIFEST_VERSIONS_WRONG_TYPE",
                    "path": str(path),
                    "message": "manifest heights.50.versions is not a list",
                }
            )
            rows_value = []
        rows = rows_value
        manifest_ids = [
            str(row.get("version_id", "") or "")
            for row in rows
            if isinstance(row, Mapping) and str(row.get("version_id", "") or "")
        ]
        disk_set = set(disk_ids)
        manifest_set = set(manifest_ids)
        return {
            "path": str(path),
            "present": True,
            "schema_version": str((value or {}).get("schema_version", "") or ""),
            "declared_version_count": height_row.get("version_count"),
            "version_ids": manifest_ids,
            "manifest_only_version_ids": sorted(manifest_set - disk_set),
            "disk_only_version_ids": sorted(disk_set - manifest_set),
            "issues": issues,
        }

    def _priority(self, version_id: str) -> tuple[str, int]:
        prefix = version_id.split("_", 1)[0].lower()
        if prefix == self.primary_version_prefix:
            return PRIMARY_PRIORITY, 0
        return REFERENCE_PRIORITY, 1

    def _audit_version(
        self,
        source: VersionSource,
        metadata: dict[str, Any] | None,
        steps: list[dict[str, Any]],
        initial_issues: list[dict[str, Any]],
        schema_baseline: Mapping[str, Any],
        *,
        active_version_id: str,
    ) -> dict[str, Any]:
        issues = list(initial_issues)

        def add_issue(
            severity: str,
            code: str,
            message: str,
            **details: Any,
        ) -> None:
            row = {"severity": severity, "code": code, "message": message}
            row.update(details)
            issues.append(row)

        accepted_present = source.accepted_steps_path.is_file()
        metadata_present = source.metadata_path.is_file()
        priority, priority_rank = self._priority(source.version_id)
        accepted_sha = sha256_file(source.accepted_steps_path) if accepted_present else ""
        expected_sha = str((metadata or {}).get("accepted_steps_sha256", "") or "").lower()
        sha_matches = bool(accepted_sha and expected_sha and accepted_sha.lower() == expected_sha)
        if not accepted_present:
            add_issue("ERROR", "ACCEPTED_STEPS_MISSING", "accepted_steps.jsonl is missing")
        if not metadata_present:
            add_issue("ERROR", "METADATA_MISSING", "metadata.json is missing")
        if accepted_present and metadata_present and not sha_matches:
            add_issue(
                "ERROR",
                "ACCEPTED_STEPS_HASH_MISMATCH",
                "accepted_steps SHA-256 differs from metadata",
                actual_sha256=accepted_sha,
                metadata_sha256=expected_sha,
            )

        metadata_missing = _required_missing(metadata, METADATA_REQUIRED_FIELDS)
        for field in metadata_missing:
            add_issue(
                "ERROR",
                "METADATA_REQUIRED_FIELD_MISSING",
                f"metadata required field is missing: {field}",
                field=field,
            )
        metadata_schema = str((metadata or {}).get("schema_version", "") or "")
        metadata_schema_valid = metadata_schema == "height-steps-versions.v2"
        if metadata is not None and not metadata_schema_valid:
            add_issue(
                "ERROR",
                "METADATA_SCHEMA_UNSUPPORTED",
                f"unsupported metadata schema: {metadata_schema!r}",
            )
        if metadata is not None and str(metadata.get("version_id", "") or "") != source.version_id:
            add_issue(
                "ERROR",
                "METADATA_VERSION_ID_MISMATCH",
                "metadata version_id differs from directory name",
                metadata_version_id=str(metadata.get("version_id", "") or ""),
            )
        if metadata is not None and _int_or_none(metadata.get("height_mm")) != 50:
            add_issue("ERROR", "METADATA_HEIGHT_MISMATCH", "metadata height_mm is not 50")

        schema_drift: list[dict[str, Any]] = []
        difference = _schema_difference(
            metadata,
            list(schema_baseline.get("metadata_keys", []) or []),
            location="metadata",
        )
        if difference:
            schema_drift.append(difference)

        null_paths: list[str] = []
        nonfinite_paths: list[str] = []
        if metadata is not None:
            found_nulls, found_nonfinite = _value_anomalies(metadata, "metadata")
            null_paths.extend(found_nulls)
            nonfinite_paths.extend(found_nonfinite)

        source_events: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        atomic_rows: list[dict[str, Any]] = []
        source_wheel_segments: list[dict[str, Any]] = []
        wheel_integral = {name: 0.0 for name in WHEEL_NAMES}
        wheel_integral_errors: list[dict[str, Any]] = []
        snapshot_rows: list[dict[str, Any]] = []
        step_fingerprints: dict[str, list[int]] = defaultdict(list)
        event_fingerprints: dict[str, list[dict[str, int]]] = defaultdict(list)
        duplicate_indices: dict[int, list[int]] = defaultdict(list)
        duplicate_names: dict[str, list[int]] = defaultdict(list)
        same_timestamp_overlaps: list[dict[str, Any]] = []
        snapshot_time_overlaps: list[dict[str, Any]] = []
        mid_step_stops: list[dict[str, Any]] = []
        empty_waits: list[dict[str, Any]] = []
        empty_steps: list[int] = []
        event_before_target_mismatches: list[dict[str, Any]] = []
        source_event_index_errors: list[dict[str, Any]] = []
        timestamp_errors: list[dict[str, Any]] = []
        duration_errors: list[dict[str, Any]] = []
        target_state_errors: list[dict[str, Any]] = []
        command_counts = Counter()
        previous_snapshot_after_time: float | None = None
        raw_duration_total = 0.0

        for ordinal, step in enumerate(steps, start=1):
            step_index_value = step.get("index")
            recorded_step_index = _int_or_none(step_index_value)
            step_index = recorded_step_index if recorded_step_index is not None else ordinal
            duplicate_indices[step_index].append(ordinal)
            duplicate_names[str(step.get("name", "") or "")].append(step_index)
            if recorded_step_index != ordinal:
                source_event_index_errors.append(
                    {
                        "source_step_index": recorded_step_index,
                        "recorded_source_step_index": _float_or_none(step_index_value),
                        "expected_step_index": ordinal,
                        "reason": "step index is not contiguous 1-based corpus order",
                    }
                )
            missing = _required_missing(step, STEP_REQUIRED_FIELDS)
            for field in missing:
                add_issue(
                    "ERROR",
                    "STEP_REQUIRED_FIELD_MISSING",
                    f"step {step_index} required field is missing: {field}",
                    source_step_index=step_index,
                    field=field,
                )
            difference = _schema_difference(
                step,
                list(schema_baseline.get("step_keys", []) or []),
                location=f"steps[{ordinal - 1}]",
            )
            if difference:
                schema_drift.append(difference)
            difference = _schema_difference(
                step.get("motion_semantics"),
                list(schema_baseline.get("motion_semantics_keys", []) or []),
                location=f"steps[{ordinal - 1}].motion_semantics",
            )
            if difference:
                schema_drift.append(difference)
            difference = _schema_difference(
                step.get("recording_timing"),
                list(schema_baseline.get("recording_timing_keys", []) or []),
                location=f"steps[{ordinal - 1}].recording_timing",
            )
            if difference:
                schema_drift.append(difference)
            for field in _required_missing(step.get("motion_semantics"), MOTION_REQUIRED_FIELDS):
                add_issue(
                    "ERROR",
                    "MOTION_REQUIRED_FIELD_MISSING",
                    f"step {step_index} motion_semantics field is missing: {field}",
                    source_step_index=step_index,
                    field=field,
                )
            for field in _required_missing(step.get("recording_timing"), TIMING_REQUIRED_FIELDS):
                add_issue(
                    "ERROR",
                    "TIMING_REQUIRED_FIELD_MISSING",
                    f"step {step_index} recording_timing field is missing: {field}",
                    source_step_index=step_index,
                    field=field,
                )
            found_nulls, found_nonfinite = _value_anomalies(step, f"steps.{ordinal - 1}")
            null_paths.extend(found_nulls)
            nonfinite_paths.extend(found_nonfinite)

            duration = _float_or_none(step.get("duration"))
            if duration is None or duration < 0.0:
                duration_errors.append(
                    {"source_step_index": step_index, "reason": "invalid step duration"}
                )
                duration = 0.0
            raw_duration_total += duration
            events_value = step.get("events")
            if not isinstance(events_value, list):
                add_issue(
                    "ERROR",
                    "STEP_EVENTS_WRONG_TYPE",
                    f"step {step_index} events is not a list",
                    source_step_index=step_index,
                )
                events: list[Any] = []
            else:
                # Retain the raw array positions.  Filtering first would
                # silently renumber source_event_index after a malformed row.
                events = list(events_value)

            initial_targets, initial_target_errors = _strict_target_state(
                step.get("command_state_before")
            )
            final_targets, final_target_errors = _strict_target_state(
                step.get("command_state_after")
            )
            if initial_target_errors:
                target_state_errors.append(
                    {
                        "source_step_index": step_index,
                        "field": "command_state_before",
                        "errors": initial_target_errors,
                    }
                )
            if final_target_errors:
                target_state_errors.append(
                    {
                        "source_step_index": step_index,
                        "field": "command_state_after",
                        "errors": final_target_errors,
                    }
                )

            event_times: list[float] = []
            command_time_origins: list[float] = []
            running_state = _state_copy(initial_targets) if not initial_target_errors else None
            by_timestamp: dict[float, list[dict[str, Any]]] = defaultdict(list)
            decoded_in_step = 0
            step_servo_count = 0
            step_wheel_count = 0
            step_timing_count = 0
            step_other_count = 0
            actionable_command_count = 0
            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    add_issue(
                        "ERROR",
                        "EVENT_WRONG_TYPE",
                        f"step {step_index} event {event_index} is not an object",
                        source_step_index=step_index,
                        source_event_index=event_index,
                        value_type=type(event).__name__,
                    )
                    source_event_index_errors.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "reason": "source event is not an object",
                        }
                    )
                    source_events.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "source_event_id": (
                                f"{source.version_id}:s{step_index:03d}:e{event_index:03d}"
                            ),
                            "source_event_valid": False,
                            "source_event_error": "source event is not an object",
                            "value_type": type(event).__name__,
                            "time_s": None,
                            "actual_recording_time_s": None,
                            "duration_to_next_event_s": None,
                            "expanded_commands": [],
                            "channels": [],
                            "control_keys": [],
                            "atomic_servo_wheel": False,
                            "atomic_evidence_valid": None,
                        }
                    )
                    continue
                event_kind = str(event.get("kind", "") or "<missing>")
                expected_event_keys = dict(
                    schema_baseline.get("event_keys_by_kind", {}) or {}
                ).get(event_kind, [])
                difference = _schema_difference(
                    event,
                    expected_event_keys,
                    location=f"steps[{ordinal - 1}].events[{event_index}]",
                )
                if difference:
                    schema_drift.append(difference)
                for field in _required_missing(event, EVENT_REQUIRED_FIELDS):
                    add_issue(
                        "ERROR",
                        "EVENT_REQUIRED_FIELD_MISSING",
                        f"step {step_index} event {event_index} required field is missing: {field}",
                        source_step_index=step_index,
                        source_event_index=event_index,
                        field=field,
                    )
                commands = event_playback_commands(event)
                syntax_errors = [_command_syntax_error(command) for command in commands]
                channels = [
                    _command_channel(command) if syntax_error is None else "invalid"
                    for command, syntax_error in zip(commands, syntax_errors)
                ]
                decoded_in_step += len(commands)
                actionable_command_count += sum(
                    channel in {"servo", "wheel", "timing", "combined"}
                    for channel in channels
                )
                for channel in channels:
                    command_counts[channel] += 1
                    if channel == "servo":
                        step_servo_count += 1
                    elif channel == "wheel":
                        step_wheel_count += 1
                    elif channel == "timing":
                        step_timing_count += 1
                    elif channel not in {"marker"}:
                        step_other_count += 1
                invalid_commands = [
                    {"command": command, "error": syntax_error}
                    for command, syntax_error in zip(commands, syntax_errors)
                    if syntax_error is not None
                ]
                if invalid_commands:
                    add_issue(
                        "ERROR",
                        "COMMAND_INVALID",
                        f"step {step_index} event {event_index} has an unknown or malformed command",
                        source_step_index=step_index,
                        source_event_index=event_index,
                        commands=invalid_commands,
                    )
                event_time = _float_or_none(event.get("time"))
                actual_time = _float_or_none(event.get("actual_recording_time_s"))
                command_start_time = _float_or_none(event.get("command_start_sim_time"))
                if event_time is None:
                    timestamp_errors.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "reason": "event time is missing or non-finite",
                        }
                    )
                    event_time = 0.0
                event_times.append(event_time)
                if command_start_time is not None:
                    command_time_origins.append(command_start_time - event_time)
                if actual_time is None or not _close(event_time, actual_time):
                    timestamp_errors.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "reason": "time differs from actual_recording_time_s",
                        }
                    )
                if event_time < -EPSILON or event_time > duration + EPSILON:
                    timestamp_errors.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "reason": "event time lies outside [0, duration]",
                            "time_s": event_time,
                            "duration_s": duration,
                        }
                    )
                if str(event.get("recording_timing_source", "") or "") != "simulation_time":
                    timestamp_errors.append(
                        {
                            "source_step_index": step_index,
                            "source_event_index": event_index,
                            "reason": "recording_timing_source is not simulation_time",
                        }
                    )
                event_controls = sorted(
                    set().union(*(_control_keys(command) for command in commands))
                    if commands
                    else set()
                )
                by_timestamp[event_time].append(
                    {
                        "source_event_index": event_index,
                        "commands": commands,
                        "control_keys": event_controls,
                    }
                )
                atomic = _atomic_evidence(
                    event,
                    step_index=step_index,
                    event_index=event_index,
                )
                if atomic["atomic_servo_wheel"]:
                    atomic_rows.append(atomic)
                    if not atomic["valid"]:
                        add_issue(
                            "ERROR",
                            "ATOMIC_EVIDENCE_INVALID",
                            f"step {step_index} event {event_index} atomic evidence is invalid",
                            source_step_index=step_index,
                            source_event_index=event_index,
                            errors=atomic["errors"],
                        )
                for command in commands:
                    wait_duration = _wait_duration(command)
                    if wait_duration is not None and (
                        not math.isfinite(wait_duration) or wait_duration <= EPSILON
                    ):
                        empty_waits.append(
                            {
                                "source_step_index": step_index,
                                "source_event_index": event_index,
                                "command": command,
                                "duration_s": wait_duration if math.isfinite(wait_duration) else None,
                            }
                        )
                    if _is_wheel_stop(command) and event_time < duration - EPSILON:
                        mid_step_stops.append(
                            {
                                "source_step_index": step_index,
                                "source_event_index": event_index,
                                "time_s": event_time,
                                "step_duration_s": duration,
                                "command": command,
                            }
                        )
                if running_state is not None:
                    if not _target_states_equal(running_state, event.get("command_state_before")):
                        event_before_target_mismatches.append(
                            {
                                "source_step_index": step_index,
                                "source_event_index": event_index,
                            }
                        )
                    for command in commands:
                        apply_command_to_state(running_state, command)
                    if not _target_states_equal(running_state, event.get("command_state_after")):
                        target_state_errors.append(
                            {
                                "source_step_index": step_index,
                                "source_event_index": event_index,
                                "field": "event.command_state_after",
                                "errors": ["decoded command state differs"],
                            }
                        )
                event_fingerprints[_source_event_fingerprint(event)].append(
                    {"source_step_index": step_index, "source_event_index": event_index}
                )
                next_time = duration
                for following_event in events[event_index + 1 :]:
                    if not isinstance(following_event, Mapping):
                        continue
                    following_time = _float_or_none(following_event.get("time"))
                    if following_time is not None:
                        next_time = following_time
                        break
                source_events.append(
                    {
                        "source_step_index": step_index,
                        "source_event_index": event_index,
                        "source_event_id": f"{source.version_id}:s{step_index:03d}:e{event_index:03d}",
                        "source_event_valid": True,
                        "source_event_error": "",
                        "time_s": event_time,
                        "actual_recording_time_s": actual_time,
                        "duration_to_next_event_s": (
                            max(0.0, float(next_time) - event_time)
                            if next_time is not None
                            else None
                        ),
                        "command_start_sim_time": _float_or_none(event.get("command_start_sim_time")),
                        "applied_sim_time": _float_or_none(event.get("applied_sim_time")),
                        "applied_sim_step": (
                            _int_or_none(event.get("applied_sim_step"))
                        ),
                        "kind": event_kind,
                        "base_command": str(event.get("command", "") or ""),
                        "expanded_commands": commands,
                        "channels": sorted(set(channels)),
                        "control_keys": event_controls,
                        "atomic_servo_wheel": bool(atomic["atomic_servo_wheel"]),
                        "atomic_evidence_valid": bool(atomic["valid"])
                        if atomic["atomic_servo_wheel"]
                        else None,
                    }
                )

            if actionable_command_count == 0:
                empty_steps.append(step_index)

            for left, right in zip(event_times, event_times[1:]):
                if right < left - EPSILON:
                    timestamp_errors.append(
                        {
                            "source_step_index": step_index,
                            "reason": "source event timestamps are not monotonic",
                            "left_s": left,
                            "right_s": right,
                        }
                    )
            if command_time_origins and (
                max(command_time_origins) - min(command_time_origins) > 1.0e-6
            ):
                timestamp_errors.append(
                    {
                        "source_step_index": step_index,
                        "reason": "command_start_sim_time - event time is not constant within the step",
                        "minimum_origin_s": min(command_time_origins),
                        "maximum_origin_s": max(command_time_origins),
                    }
                )
            if event_times and not _close(max(event_times), duration, tolerance=1.0e-6):
                duration_errors.append(
                    {
                        "source_step_index": step_index,
                        "reason": "duration differs from maximum event timestamp",
                        "duration_s": duration,
                        "maximum_event_time_s": max(event_times),
                    }
                )
            timing = step.get("recording_timing") if isinstance(step.get("recording_timing"), Mapping) else {}
            motion = step.get("motion_semantics") if isinstance(step.get("motion_semantics"), Mapping) else {}
            for location, value in (
                ("recording_timing.actual_duration_s", timing.get("actual_duration_s")),
                ("motion_semantics.actual_recording_duration_s", motion.get("actual_recording_duration_s")),
                ("motion_semantics.reference_duration_s", motion.get("reference_duration_s")),
            ):
                if not _close(value, duration, tolerance=1.0e-6):
                    duration_errors.append(
                        {
                            "source_step_index": step_index,
                            "reason": f"{location} differs from step duration",
                            "value": _float_or_none(value),
                            "duration_s": duration,
                        }
                    )
            if str(timing.get("source", "") or "") != "simulation_time":
                timestamp_errors.append(
                    {
                        "source_step_index": step_index,
                        "reason": "recording_timing.source is not simulation_time",
                    }
                )

            for timestamp, group in sorted(by_timestamp.items()):
                if len(group) < 2:
                    continue
                overlaps: set[str] = set()
                for left_index, left in enumerate(group):
                    left_keys = set(left["control_keys"])
                    for right in group[left_index + 1 :]:
                        overlaps.update(left_keys.intersection(right["control_keys"]))
                if overlaps:
                    same_timestamp_overlaps.append(
                        {
                            "source_step_index": step_index,
                            "time_s": timestamp,
                            "overlapping_control_keys": sorted(overlaps),
                            "events": group,
                            "resolution": "JSONL event order is authoritative; overlap is not atomic",
                        }
                    )

            for field in ("sim_state_before", "sim_state_after"):
                validation = validate_full_sim_pose_state(step.get(field))
                snapshot_rows.append(
                    {
                        "source_step_index": step_index,
                        "field": field,
                        "classification": str(validation.get("classification", "INVALID")),
                        "valid": bool(validation.get("valid", False)),
                        "reason": str(validation.get("reason", "") or ""),
                        "missing_fields": list(validation.get("missing_fields", []) or []),
                        "invalid_fields": list(validation.get("invalid_fields", []) or []),
                    }
                )
            before_time = _float_or_none(
                (step.get("sim_state_before") or {}).get("sim_time")
                if isinstance(step.get("sim_state_before"), Mapping)
                else None
            )
            after_time = _float_or_none(
                (step.get("sim_state_after") or {}).get("sim_time")
                if isinstance(step.get("sim_state_after"), Mapping)
                else None
            )
            if before_time is not None and after_time is not None and after_time < before_time - EPSILON:
                timestamp_errors.append(
                    {
                        "source_step_index": step_index,
                        "reason": "sim_state_after.sim_time precedes sim_state_before.sim_time",
                    }
                )
            if (
                previous_snapshot_after_time is not None
                and before_time is not None
                and before_time < previous_snapshot_after_time - EPSILON
            ):
                snapshot_time_overlaps.append(
                    {
                        "source_step_index": step_index,
                        "previous_after_sim_time_s": previous_snapshot_after_time,
                        "current_before_sim_time_s": before_time,
                        "overlap_s": previous_snapshot_after_time - before_time,
                    }
                )
            if after_time is not None:
                previous_snapshot_after_time = after_time

            segments, integral, integration_issues = _source_wheel_segments(
                step,
                step_index=step_index,
            )
            source_wheel_segments.extend(segments)
            issues.extend(integration_issues)
            for name in WHEEL_NAMES:
                if _finite(integral.get(name)):
                    wheel_integral[name] += float(integral[name])
            expected_integral = motion.get("derived_wheel_angular_displacement_rad")
            if not isinstance(expected_integral, Mapping):
                expected_integral = motion.get("canonical_wheel_angular_displacement_rad")
            for name in WHEEL_NAMES:
                expected_value = expected_integral.get(name) if isinstance(expected_integral, Mapping) else None
                actual_value = integral.get(name)
                if not _close(expected_value, actual_value, tolerance=1.0e-6):
                    wheel_integral_errors.append(
                        {
                            "source_step_index": step_index,
                            "wheel_joint_name": name,
                            "computed_theta_rad": _float_or_none(actual_value),
                            "recorded_theta_rad": _float_or_none(expected_value),
                            "error_rad": (
                                abs(float(actual_value) - float(expected_value))
                                if _finite(actual_value) and _finite(expected_value)
                                else None
                            ),
                        }
                    )
            step_fingerprints[_step_fingerprint(step)].append(step_index)
            step_rows.append(
                {
                    "source_step_index": step_index,
                    "name": str(step.get("name", "") or ""),
                    "type": str(step.get("type", "") or ""),
                    "duration_s": duration,
                    "source_event_count": len(events),
                    "decoded_command_count": decoded_in_step,
                    "servo_command_count": step_servo_count,
                    "wheel_command_count": step_wheel_count,
                    "timing_command_count": step_timing_count,
                    "other_command_count": step_other_count,
                    "initial_joint_targets": initial_targets,
                    "final_joint_targets": final_targets,
                    "before_snapshot": snapshot_rows[-2],
                    "after_snapshot": snapshot_rows[-1],
                    "source_wheel_segment_indices": list(
                        range(
                            len(source_wheel_segments) - len(segments),
                            len(source_wheel_segments),
                        )
                    ),
                    "source_wheel_theta_rad": integral,
                    "step_fingerprint_sha256": _step_fingerprint(step),
                }
            )

        for details in timestamp_errors:
            add_issue("ERROR", "TIMESTAMP_INVALID", details["reason"], **details)
        for details in duration_errors:
            add_issue("ERROR", "DURATION_INVALID", details["reason"], **details)
        for details in target_state_errors:
            add_issue(
                "ERROR",
                "TARGET_STATE_INVALID",
                f"target-state evidence invalid at step {details.get('source_step_index')}",
                **details,
            )
        if source_event_index_errors:
            add_issue(
                "ERROR",
                "SOURCE_EVENT_INDEX_INVALID",
                "one or more source step/event indices are invalid or unmappable",
                count=len(source_event_index_errors),
            )
        if wheel_integral_errors:
            add_issue(
                "ERROR",
                "WHEEL_INTEGRAL_MISMATCH",
                "computed sum(omega * dt) differs from recorded canonical displacement",
                mismatch_count=len(wheel_integral_errors),
            )
        if nonfinite_paths:
            add_issue(
                "ERROR",
                "NONFINITE_NUMERIC_VALUE",
                "recording contains NaN or infinity",
                count=len(nonfinite_paths),
            )
        if schema_drift:
            add_issue(
                "ERROR",
                "SCHEMA_DRIFT",
                "one or more objects differ from the corpus modal key schema",
                count=len(schema_drift),
            )

        duplicate_step_groups = [
            {"fingerprint_sha256": fingerprint, "source_step_indices": indices}
            for fingerprint, indices in sorted(step_fingerprints.items())
            if len(indices) > 1
        ]
        duplicate_index_groups = [
            {"source_step_index": index, "row_ordinals": ordinals}
            for index, ordinals in sorted(duplicate_indices.items())
            if len(ordinals) > 1
        ]
        duplicate_name_groups = [
            {"name": name, "source_step_indices": indices}
            for name, indices in sorted(duplicate_names.items())
            if name and len(indices) > 1
        ]
        duplicate_event_groups = [
            {"fingerprint_sha256": fingerprint, "locations": locations}
            for fingerprint, locations in sorted(event_fingerprints.items())
            if len(locations) > 1
        ]
        duplicate_step_count = sum(len(row["source_step_indices"]) - 1 for row in duplicate_step_groups)
        duplicate_event_count = sum(len(row["locations"]) - 1 for row in duplicate_event_groups)
        if duplicate_index_groups:
            add_issue("ERROR", "DUPLICATE_STEP_INDEX", "duplicate step indices exist")
        if duplicate_step_groups:
            add_issue(
                "WARN",
                "DUPLICATE_STEP_SEMANTICS",
                "semantically duplicate steps exist within this version",
                count=duplicate_step_count,
            )
        if same_timestamp_overlaps:
            add_issue(
                "WARN",
                "SAME_TIMESTAMP_CONTROL_OVERLAP",
                "separate source events command the same actuator at one timestamp",
                count=len(same_timestamp_overlaps),
            )
        if snapshot_time_overlaps:
            add_issue(
                "WARN",
                "SNAPSHOT_TIME_OVERLAP",
                "curated step snapshot clocks overlap or jump backwards",
                count=len(snapshot_time_overlaps),
            )
        if event_before_target_mismatches:
            add_issue(
                "WARN",
                "EVENT_BEFORE_STATE_STAGING_MISMATCH",
                "event command_state_before differs from ordered decoded state; event order remains explicit",
                count=len(event_before_target_mismatches),
            )

        full_snapshot_count = sum(
            row["classification"] == FULL_VALID and row["valid"] for row in snapshot_rows
        )
        incomplete_snapshot_count = len(snapshot_rows) - full_snapshot_count
        if incomplete_snapshot_count:
            add_issue(
                "ERROR",
                "SNAPSHOT_INCOMPLETE",
                "one or more before/after snapshots are not FULL_VALID",
                full_valid_count=full_snapshot_count,
                snapshot_count=len(snapshot_rows),
            )

        metadata_step_count = _int_or_none((metadata or {}).get("step_count"))
        metadata_command_count = _int_or_none((metadata or {}).get("command_count"))
        source_event_count = len(source_events)
        if metadata_step_count != len(steps):
            add_issue(
                "ERROR",
                "METADATA_STEP_COUNT_MISMATCH",
                "metadata step_count differs from accepted JSONL rows",
                metadata_step_count=metadata_step_count,
                actual_step_count=len(steps),
            )
        if metadata_command_count != source_event_count:
            add_issue(
                "ERROR",
                "METADATA_COMMAND_COUNT_MISMATCH",
                "metadata command_count differs from source event count",
                metadata_command_count=metadata_command_count,
                source_event_count=source_event_count,
            )

        fast_plan: dict[str, Any] = {
            "built": False,
            "source_mapping_error_count": 0,
            "wheel_segments": [],
        }
        if steps and not any(issue["code"].startswith("JSONL_") for issue in issues):
            try:
                plan, fast_rows = fast_plan_rows(
                    source_version=source.version_id,
                    steps=steps,
                    max_wheel_speed=DEFAULT_MAX_WHEEL_SPEED_RAD_S,
                )
                provenance = [
                    row
                    for segment in fast_rows
                    for row in list(segment.get("command_provenance", []) or [])
                ]
                mapping_errors = [row for row in provenance if row.get("mapping_error")]
                fast_wheel_segments = []
                fast_theta = {name: 0.0 for name in WHEEL_NAMES}
                for row in fast_rows:
                    speeds = {
                        name: float((row.get("wheel_target_rad_s", {}) or {}).get(name, 0.0))
                        for name in WHEEL_NAMES
                    }
                    duration = float(row.get("wheel_duration_s", 0.0) or 0.0)
                    if duration <= EPSILON or not any(abs(value) > EPSILON for value in speeds.values()):
                        continue
                    theta = {name: speeds[name] * duration for name in WHEEL_NAMES}
                    for name in WHEEL_NAMES:
                        fast_theta[name] += theta[name]
                    fast_wheel_segments.append(
                        {
                            "decoded_segment_index": int(row["decoded_segment_index"]),
                            "source_step_index": int(row["source_step_index"]),
                            "source_event_indices": list(row["source_event_indices"]),
                            "start_s": float(row["command_start_s"]),
                            "end_s": float(row["command_end_s"]),
                            "duration_s": duration,
                            "speed_rad_s": speeds,
                            "theta_rad": theta,
                            "concurrent_servo_wheel": bool(row["concurrent"]),
                        }
                    )
                fast_plan = {
                    "built": True,
                    "profile": str(plan.profile),
                    "plan_sha256": str(plan.plan_sha256),
                    "source_event_count": len(plan.events),
                    "segment_count": len(plan.segments),
                    "duration_s": float(plan.final_time_s),
                    "source_mapping_valid_count": len(provenance) - len(mapping_errors),
                    "source_mapping_error_count": len(mapping_errors),
                    "source_mapping_errors": mapping_errors,
                    "wheel_segment_count": len(fast_wheel_segments),
                    "wheel_theta_rad": fast_theta,
                    "wheel_segments": fast_wheel_segments,
                }
                if mapping_errors:
                    add_issue(
                        "ERROR",
                        "FAST_PLAN_SOURCE_MAPPING_ERROR",
                        "authoritative Fast plan lost source event indices",
                        count=len(mapping_errors),
                    )
                    source_event_index_errors.extend(mapping_errors)
            except Exception as exc:  # audit must persist the failure, not select a fallback
                add_issue(
                    "ERROR",
                    "FAST_PLAN_BUILD_FAILED",
                    f"authoritative Fast planner failed: {type(exc).__name__}: {exc}",
                )

        error_codes = sorted({issue["code"] for issue in issues if issue["severity"] == "ERROR"})
        warning_codes = sorted({issue["code"] for issue in issues if issue["severity"] == "WARN"})
        static_status = "PASS" if not error_codes else "FAIL"
        wheel_integral_max_error = max(
            [
                float(row["error_rad"])
                for row in wheel_integral_errors
                if row.get("error_rad") is not None
            ]
            + [0.0]
        )
        null_counts = Counter(null_paths)
        nonfinite_counts = Counter(nonfinite_paths)
        return {
            "schema_version": INDEX_SCHEMA,
            "version_id": source.version_id,
            "version_directory": str(source.directory),
            "priority": priority,
            "priority_rank": priority_rank,
            "active_pointer": source.version_id == active_version_id,
            "selection_eligible": False,
            "selection_reason": "static audit never selects a physical primitive",
            "accepted_steps_path": str(source.accepted_steps_path),
            "metadata_path": str(source.metadata_path),
            "accepted_steps_present": accepted_present,
            "metadata_present": metadata_present,
            "accepted_steps_sha256": accepted_sha,
            "metadata_accepted_steps_sha256": expected_sha,
            "sha256_matches": sha_matches,
            "metadata_schema_version": metadata_schema,
            "metadata_schema_valid": metadata_schema_valid,
            "step_count": len(steps),
            "metadata_step_count": metadata_step_count,
            "source_event_count": source_event_count,
            "metadata_command_count": metadata_command_count,
            "decoded_command_count": sum(command_counts.values()),
            "servo_command_count": int(command_counts["servo"]),
            "wheel_command_count": int(command_counts["wheel"]),
            "timing_command_count": int(command_counts["timing"]),
            "other_command_count": int(
                command_counts["other"]
                + command_counts["invalid"]
                + command_counts["empty"]
                + command_counts["combined"]
            ),
            "atomic_event_count": len(atomic_rows),
            "atomic_valid_count": sum(bool(row["valid"]) for row in atomic_rows),
            "same_tick_servo_wheel_atomic_count": sum(bool(row["valid"]) for row in atomic_rows),
            "source_event_index_valid_count": sum(
                row.get("source_event_valid") is True for row in source_events
            ),
            "source_event_index_error_count": len(source_event_index_errors),
            "source_duration_s": raw_duration_total,
            "timestamp_error_count": len(timestamp_errors),
            "duration_error_count": len(duration_errors),
            "target_state_error_count": len(target_state_errors),
            "snapshot_full_valid_count": full_snapshot_count,
            "snapshot_count": len(snapshot_rows),
            "snapshot_incomplete_count": incomplete_snapshot_count,
            "source_wheel_segment_count": len(source_wheel_segments),
            "fast_wheel_segment_count": int(fast_plan.get("wheel_segment_count", 0) or 0),
            "wheel_integral_theta_rad": wheel_integral,
            "wheel_integral_mismatch_count": len(wheel_integral_errors),
            "wheel_integral_max_error_rad": wheel_integral_max_error,
            "duplicate_step_count": duplicate_step_count,
            "duplicate_event_count": duplicate_event_count,
            "empty_step_count": len(empty_steps),
            "empty_wait_count": len(empty_waits),
            "same_timestamp_overlap_count": len(same_timestamp_overlaps),
            "snapshot_time_overlap_count": len(snapshot_time_overlaps),
            "mid_step_wheel_stop_count": len(mid_step_stops),
            "missing_required_field_count": sum(
                issue["code"].endswith("REQUIRED_FIELD_MISSING") for issue in issues
            ),
            "null_value_count": len(null_paths),
            "null_value_path_counts": dict(sorted(null_counts.items())),
            "nonfinite_numeric_count": len(nonfinite_paths),
            "nonfinite_numeric_path_counts": dict(sorted(nonfinite_counts.items())),
            "schema_drift_count": len(schema_drift),
            "static_integrity_status": static_status,
            "physical_replay_status": STATIC_ONLY_STATUS,
            "issue_codes": error_codes + warning_codes,
            "error_codes": error_codes,
            "warning_codes": warning_codes,
            "issues": issues,
            "schema_drift": schema_drift,
            "steps": step_rows,
            "source_events": source_events,
            "source_wheel_segments": source_wheel_segments,
            "fast_plan": fast_plan,
            "snapshots": snapshot_rows,
            "atomic_events": atomic_rows,
            "duplicates": {
                "step_fingerprint_groups": duplicate_step_groups,
                "step_index_groups": duplicate_index_groups,
                "step_name_groups": duplicate_name_groups,
                "event_groups": duplicate_event_groups,
            },
            "empty_steps": empty_steps,
            "empty_waits": empty_waits,
            "same_timestamp_overlaps": same_timestamp_overlaps,
            "snapshot_time_overlaps": snapshot_time_overlaps,
            "mid_step_wheel_stops": mid_step_stops,
            "event_before_target_mismatches": event_before_target_mismatches,
            "wheel_integral_errors": wheel_integral_errors,
        }

    def audit(self) -> dict[str, Any]:
        sources = self.enumerate_versions()
        loaded: list[
            tuple[
                VersionSource,
                dict[str, Any] | None,
                list[dict[str, Any]],
                list[dict[str, Any]],
            ]
        ] = []
        for source in sources:
            issues: list[dict[str, Any]] = []
            metadata: dict[str, Any] | None = None
            steps: list[dict[str, Any]] = []
            if source.metadata_path.is_file():
                metadata, read_issues = _read_json_object(source.metadata_path)
                issues.extend(read_issues)
            if source.accepted_steps_path.is_file():
                steps, read_issues = _read_jsonl_objects(source.accepted_steps_path)
                issues.extend(read_issues)
            loaded.append((source, metadata, steps, issues))

        schema_baseline = _derive_schema_baseline(
            [(source, metadata, steps) for source, metadata, steps, _issues in loaded]
        )
        active_pointer = self._active_pointer()
        active_version_id = str(active_pointer.get("version_id", "") or "")
        versions = [
            self._audit_version(
                source,
                metadata,
                steps,
                issues,
                schema_baseline,
                active_version_id=active_version_id,
            )
            for source, metadata, steps, issues in loaded
        ]
        disk_ids = [source.version_id for source in sources]
        primary_matches = [
            version["version_id"]
            for version in versions
            if version["version_id"].split("_", 1)[0].lower()
            == self.primary_version_prefix
        ]
        priority_valid = len(primary_matches) == 1
        selection_policy = {
            "schema_version": "fsm50.recording_corpus_selection_policy.v1",
            "primary_version_prefix": self.primary_version_prefix,
            "primary_priority": PRIMARY_PRIORITY,
            "reference_priority": REFERENCE_PRIORITY,
            "primary_version_ids": primary_matches,
            "priority_policy_valid": priority_valid,
            "active_pointer_version_id": active_version_id,
            "active_pointer_is_selection": False,
            "automatic_selected_version_id": None,
            "physical_primitive_selected": False,
            "no_implicit_active_latest": True,
            "selection_policy": "NO_IMPLICIT_ACTIVE_LATEST",
            "rule": (
                "v003 receives the highest static audit priority because it is the first "
                "complete action sequence. All other physical directories are cross-version "
                "references. The active/latest pointer, fewer steps, and version number never "
                "select a primitive; physical replay evidence is required."
            ),
        }

        accepted_hash_groups: dict[str, list[str]] = defaultdict(list)
        cross_step_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for version in versions:
            if version["accepted_steps_sha256"]:
                accepted_hash_groups[version["accepted_steps_sha256"]].append(
                    version["version_id"]
                )
            for step in version["steps"]:
                cross_step_groups[step["step_fingerprint_sha256"]].append(
                    {
                        "version_id": version["version_id"],
                        "source_step_index": step["source_step_index"],
                    }
                )
        duplicate_accepted_hash_groups = [
            {"accepted_steps_sha256": digest, "version_ids": version_ids}
            for digest, version_ids in sorted(accepted_hash_groups.items())
            if len(version_ids) > 1
        ]
        cross_version_duplicate_step_groups = [
            {"step_fingerprint_sha256": digest, "locations": locations}
            for digest, locations in sorted(cross_step_groups.items())
            if len({row["version_id"] for row in locations}) > 1
        ]

        summary_fields = (
            "step_count",
            "source_event_count",
            "decoded_command_count",
            "servo_command_count",
            "wheel_command_count",
            "atomic_event_count",
            "atomic_valid_count",
            "same_tick_servo_wheel_atomic_count",
            "snapshot_full_valid_count",
            "snapshot_count",
            "snapshot_incomplete_count",
            "source_wheel_segment_count",
            "fast_wheel_segment_count",
            "duplicate_step_count",
            "duplicate_event_count",
            "empty_step_count",
            "empty_wait_count",
            "same_timestamp_overlap_count",
            "snapshot_time_overlap_count",
            "mid_step_wheel_stop_count",
            "missing_required_field_count",
            "nonfinite_numeric_count",
            "schema_drift_count",
            "timestamp_error_count",
            "duration_error_count",
            "target_state_error_count",
        )
        aggregate_counts = {
            field: sum(int(version.get(field, 0) or 0) for version in versions)
            for field in summary_fields
        }
        issue_code_counts = Counter(
            issue["code"]
            for version in versions
            for issue in version["issues"]
        )
        static_pass_count = sum(
            version["static_integrity_status"] == "PASS" for version in versions
        )
        corpus_status = (
            "PASS"
            if versions and static_pass_count == len(versions) and priority_valid
            else "STATIC_INCOMPLETE"
        )
        payload = {
            "schema_version": AUDIT_SCHEMA,
            "scope": {
                "recording_root": str(self.recording_root),
                "versions_root": str((self.recording_root / "versions").resolve()),
                "report_root": str(self.report_root),
                "membership_source": "physical_directories_only",
                "recordings_mutated": False,
                "isaac_started": False,
                "physical_replay_status": STATIC_ONLY_STATUS,
            },
            "selection_policy": selection_policy,
            "active_pointer": active_pointer,
            "manifest_inventory": self._manifest_inventory(disk_ids),
            "schema_baseline": schema_baseline,
            "aggregate": {
                "version_count_discovered": len(versions),
                "static_pass_count": static_pass_count,
                "static_fail_count": len(versions) - static_pass_count,
                "corpus_static_status": corpus_status,
                "physical_replay_status": STATIC_ONLY_STATUS,
                "counts": aggregate_counts,
                "issue_code_counts": dict(sorted(issue_code_counts.items())),
                "duplicate_accepted_hash_groups": duplicate_accepted_hash_groups,
                "cross_version_duplicate_step_group_count": len(
                    cross_version_duplicate_step_groups
                ),
            },
            "cross_version_duplicate_step_groups": cross_version_duplicate_step_groups,
            "versions": versions,
        }
        return _json_safe_value(payload)

    @staticmethod
    def _csv_row(version: Mapping[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for column in INDEX_COLUMNS:
            if column == "wheel_integral_theta_rad_json":
                value = version.get("wheel_integral_theta_rad", {})
            elif column == "issue_codes_json":
                value = version.get("issue_codes", [])
            else:
                value = version.get(column, "")
            row[column] = _json_cell(value) if isinstance(value, (dict, list, tuple)) else value
        return row

    @staticmethod
    def _markdown(payload: Mapping[str, Any]) -> str:
        aggregate = dict(payload.get("aggregate", {}) or {})
        counts = dict(aggregate.get("counts", {}) or {})
        selection = dict(payload.get("selection_policy", {}) or {})
        manifest = dict(payload.get("manifest_inventory", {}) or {})
        versions = list(payload.get("versions", []) or [])
        lines = [
            "# 50 mm Recording Corpus Static Audit",
            "",
            f"Status: **{aggregate.get('corpus_static_status', 'UNKNOWN')}**",
            "",
            (
                "This is a static, Isaac-free audit of the directories physically present in "
                "`height_050mm/versions`. It does not establish physical replay success, contact "
                "validity, primitive quality, or FSM success."
            ),
            "",
            "## Membership and selection policy",
            "",
            f"- Physical version directories discovered dynamically: **{aggregate.get('version_count_discovered', 0)}**.",
            f"- Highest audit priority: `{PRIMARY_PRIORITY}` for `{', '.join(selection.get('primary_version_ids', []) or ['MISSING'])}`.",
            f"- Active pointer observed: `{selection.get('active_pointer_version_id', '')}`; it is **not** a selection input.",
            "- No recording is automatically selected. Version recency, the active pointer, or fewer steps cannot promote v012 or any other version.",
            "- v003 is the first complete action sequence and remains the primary corpus priority; its snapshot limitations are reported rather than hidden.",
            "",
            "Manifest discrepancies are inventory evidence, not membership overrides:",
            "",
            f"- Manifest-only IDs: `{', '.join(manifest.get('manifest_only_version_ids', []) or ['none'])}`",
            f"- Disk-only IDs: `{', '.join(manifest.get('disk_only_version_ids', []) or ['none'])}`",
            "",
            "## Per-version index",
            "",
            "| Priority | Version | Static | SHA | Steps | Source / decoded commands | Servo / wheel | Atomic valid/total | FULL snapshots | Wheel segments source/fast | Duplicates | Empty waits | Overlap | Mid-stop | Schema drift |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for version in versions:
            lines.append(
                "| {priority} | `{version}` | {status} | {sha} | {steps} | {events}/{decoded} | {servo}/{wheel} | {atomic_valid}/{atomic} | {snap}/{snap_total} | {source_segments}/{fast_segments} | {duplicate} | {empty_wait} | {overlap} | {mid_stop} | {drift} |".format(
                    priority=version.get("priority"),
                    version=version.get("version_id"),
                    status=version.get("static_integrity_status"),
                    sha="PASS" if version.get("sha256_matches") else "FAIL",
                    steps=version.get("step_count"),
                    events=version.get("source_event_count"),
                    decoded=version.get("decoded_command_count"),
                    servo=version.get("servo_command_count"),
                    wheel=version.get("wheel_command_count"),
                    atomic_valid=version.get("atomic_valid_count"),
                    atomic=version.get("atomic_event_count"),
                    snap=version.get("snapshot_full_valid_count"),
                    snap_total=version.get("snapshot_count"),
                    source_segments=version.get("source_wheel_segment_count"),
                    fast_segments=version.get("fast_wheel_segment_count"),
                    duplicate=version.get("duplicate_step_count"),
                    empty_wait=version.get("empty_wait_count"),
                    overlap=version.get("same_timestamp_overlap_count"),
                    mid_stop=version.get("mid_step_wheel_stop_count"),
                    drift=version.get("schema_drift_count"),
                )
            )
        lines.extend(
            [
                "",
                "Counts use these precise meanings:",
                "",
                "- `Source commands` are accepted JSONL events and are checked against `metadata.command_count`; `decoded commands` expand atomic source events.",
                "- `Atomic` means one source event contains both servo and wheel commands and its batch acknowledgement proves a common next physics tick.",
                "- `Overlap` means separate events at the same source timestamp command at least one common actuator. JSONL order is retained, but this is not treated as atomic.",
                "- `Mid-stop` means a wheel-stop command occurs before the end of the step. It is preserved as timing evidence, not silently discarded.",
                "- Every source and authoritative Fast wheel segment is stored in the JSON index with speed, duration, source event indices, and `theta = omega * dt`.",
                "",
                "## Aggregate evidence",
                "",
                f"- Steps: **{counts.get('step_count', 0)}**",
                f"- Source events / decoded commands: **{counts.get('source_event_count', 0)} / {counts.get('decoded_command_count', 0)}**",
                f"- Servo / wheel decoded commands: **{counts.get('servo_command_count', 0)} / {counts.get('wheel_command_count', 0)}**",
                f"- Same-tick atomic batches: **{counts.get('atomic_valid_count', 0)} / {counts.get('atomic_event_count', 0)} valid**",
                f"- Before/after snapshots: **{counts.get('snapshot_full_valid_count', 0)} / {counts.get('snapshot_count', 0)} FULL_VALID**",
                f"- Source / Fast wheel segments: **{counts.get('source_wheel_segment_count', 0)} / {counts.get('fast_wheel_segment_count', 0)}**",
                f"- Missing required fields / non-finite numerics: **{counts.get('missing_required_field_count', 0)} / {counts.get('nonfinite_numeric_count', 0)}**",
                f"- Empty steps / empty waits: **{counts.get('empty_step_count', 0)} / {counts.get('empty_wait_count', 0)}**",
                f"- Same-timestamp overlaps / snapshot-clock overlaps: **{counts.get('same_timestamp_overlap_count', 0)} / {counts.get('snapshot_time_overlap_count', 0)}**",
                "",
                "## Fail-closed findings",
                "",
            ]
        )
        error_versions = [version for version in versions if version.get("error_codes")]
        if not error_versions:
            lines.append("- No static integrity errors were found.")
        else:
            for version in error_versions:
                lines.append(
                    f"- `{version['version_id']}`: `{', '.join(version['error_codes'])}`"
                )
        lines.extend(
            [
                "",
                "The v003 priority does not turn incomplete evidence into a pass. A priority is an audit/repair order, not a physical primitive selection.",
                "",
                "## Schema and evidence boundary",
                "",
                "The JSON index contains the corpus modal key schema and every deviation. Required metadata, step, timing, motion and event fields are checked independently, and all numeric leaves are scanned for NaN/Infinity. Optional historical `null` values are counted by normalized path rather than converted to zero.",
                "",
                "The recordings contain accepted commands and endpoint snapshots, not continuous contact/COM telemetry. Therefore every version remains physical replay `NOT_RUN` until a separately finalized physical replay supplies that evidence.",
                "",
            ]
        )
        return "\n".join(lines)

    def write_reports(self, payload: Mapping[str, Any]) -> dict[str, Path]:
        self.report_root.mkdir(parents=True, exist_ok=True)
        csv_path = self.report_root / "RECORDING_CORPUS_INDEX_50MM.csv"
        json_path = self.report_root / "RECORDING_CORPUS_INDEX_50MM.json"
        markdown_path = self.report_root / "RECORDING_CORPUS_AUDIT_50MM.md"
        json_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(INDEX_COLUMNS))
            writer.writeheader()
            for version in list(payload.get("versions", []) or []):
                writer.writerow(self._csv_row(version))
        markdown_path.write_text(self._markdown(payload), encoding="utf-8", newline="\n")
        return {"csv": csv_path, "json": json_path, "markdown": markdown_path}

    def run(self) -> dict[str, Any]:
        payload = self.audit()
        paths = self.write_reports(payload)
        return {"payload": payload, "paths": paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit all physical 50 mm recording version directories without Isaac"
    )
    parser.add_argument(
        "--recording-root",
        type=Path,
        default=DEFAULT_RECORDING_ROOT,
        help="height_050mm root containing versions/",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="destination for the three corpus reports",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = RecordingCorpusAudit(
        recording_root=arguments.recording_root,
        report_root=arguments.report_root,
    ).run()
    payload = result["payload"]
    print(
        json.dumps(
            {
                "corpus_static_status": payload["aggregate"]["corpus_static_status"],
                "version_count_discovered": payload["aggregate"]["version_count_discovered"],
                "reports": {key: str(path) for key, path in result["paths"].items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["aggregate"]["corpus_static_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
