"""Sensor-independent trajectory evidence for an ordinary-UI diagnostic run.

This module is deliberately Isaac-free and contact-free.  It projects an
existing per-physics-tick telemetry stream into root, COM, joint-drive, wheel,
and dispatch evidence.  Identity values that are not present in every
telemetry row are never inferred: callers must supply a sealed identity
envelope cross-bound to durable result, immutable request, readiness, and
dispatch-ledger evidence.

An ordinary-UI trajectory can be *diagnostically complete*.  It can never be a
physical-success, Gate-1, or environment-equivalence artifact because contact,
collision, penetration, and support evidence are intentionally unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES


IDENTITY_SCHEMA = "fsm50.ordinary_ui.identity_envelope.v1"
ROW_SCHEMA = "fsm50.ordinary_ui.trajectory_row.v1"
MANIFEST_SCHEMA = "fsm50.ordinary_ui.trajectory_manifest.v1"
SEAL_SCHEMA = "fsm50.ordinary_ui.trajectory_seal.v1"
BUNDLE_SCHEMA = "fsm50.ordinary_ui.trajectory_bundle.v1"
RECEIPT_SCHEMA = "fsm50.ordinary_ui.trajectory_receipt.v1"

TRAJECTORY_FILENAME = "ordinary_ui_trajectory.jsonl"
MANIFEST_FILENAME = "ordinary_ui_trajectory_manifest.json"
SEAL_FILENAME = "ordinary_ui_trajectory_seal.json"

NOT_EVALUABLE = "NOT_EVALUABLE"
PHYSICS_RATE_HZ = 120
PHYSICS_DT_S = float(Fraction(1, PHYSICS_RATE_HZ))

JOINT_NAMES = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
SERVO_NAMES = tuple(SERVO_JOINT_NAMES)
WHEEL_NAMES = tuple(WHEEL_JOINT_NAMES)
LEGS = ("FL", "FR", "RL", "RR")
LEG_TO_WHEEL = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_FIELDS = (
    "source_version",
    "accepted_steps_sha256",
    "plan_sha256",
    "plan_id",
    "request_id",
    "worker_session_id",
    "adapter_runtime_instance_id",
    "readiness_token_sha256",
    "root_state_write_count",
)
_HASH_IDENTITY_FIELDS = {
    "accepted_steps_sha256",
    "plan_sha256",
    "readiness_token_sha256",
}
_EVIDENCE_NAMES = (
    "durable_result",
    "immutable_request",
    "readiness",
    "dispatch_ledger",
)
_EVIDENCE_REQUIRED_FIELDS = {
    "durable_result": frozenset(_IDENTITY_FIELDS),
    "immutable_request": frozenset(
        {
            "source_version",
            "accepted_steps_sha256",
            "plan_sha256",
            "plan_id",
            "request_id",
            "root_state_write_count",
        }
    ),
    "readiness": frozenset(
        {
            "source_version",
            "plan_sha256",
            "plan_id",
            "request_id",
            "worker_session_id",
            "adapter_runtime_instance_id",
            "root_state_write_count",
        }
    ),
    "dispatch_ledger": frozenset(
        {
            "source_version",
            "plan_id",
            "readiness_token_sha256",
        }
    ),
}

_TOP_ROW_KEYS = frozenset(
    {
        "schema_version",
        "trace_index",
        "sample_index",
        "sim_step",
        "time_s",
        "physics_dt_s",
        "source_version",
        "root",
        "com",
        "joints",
        "wheels",
        "dispatch",
    }
)
_ROOT_KEYS = frozenset(
    {"pose_w", "linear_velocity_w", "angular_velocity_w"}
)
_COM_KEYS = frozenset({"position_w", "velocity_w"})
_JOINT_KEYS = frozenset(
    {
        "measured_position_rad",
        "measured_velocity_rad_s",
        "physx_position_target_rad",
        "position_target_buffer_rad",
        "physx_velocity_target_rad_s",
        "velocity_target_buffer_rad_s",
        "servo_command_target_rad",
        "physx_drive_target_evidence_valid",
    }
)
_WHEEL_KEYS = frozenset(
    {
        "command_velocity_rad_s",
        "canonical_forward_angle_rad",
        "canonical_forward_velocity_rad_s",
        "logical_target_rad_s_by_leg",
        "physx_target_rad_s_by_leg",
        "forward_sign_by_joint",
        "wheel_direction",
    }
)
_DISPATCH_KEYS = frozenset(
    {
        "fsm_state",
        "scheduler_phase",
        "macro_state_cursor",
        "command_cursor",
        "segment_cursor",
        "source_command",
        "source_event_index",
        "source_event_indices",
        "source_fast_segment",
        "source_step",
        "planned_dispatch_time_s",
        "actual_dispatch_time_s",
        "atomic_batch_id",
        "dispatch_kind",
        "atomic_concurrent",
        "planned_servo_target_deg",
        "planned_wheel_target_rad_s",
    }
)


class OrdinaryUITrajectoryError(ValueError):
    """Raised when ordinary-UI diagnostic evidence fails closed."""


def _reject_constant(value: str) -> None:
    raise OrdinaryUITrajectoryError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OrdinaryUITrajectoryError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except OrdinaryUITrajectoryError:
        raise
    except Exception as exc:
        raise OrdinaryUITrajectoryError(
            f"{label} is not strict JSON: {type(exc).__name__}: {exc}"
        ) from exc


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OrdinaryUITrajectoryError(
            f"value is not strict finite JSON: {type(exc).__name__}: {exc}"
        ) from exc


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality coercions."""

    return _json_bytes(left) == _json_bytes(right)


def _require_exact_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrdinaryUITrajectoryError(f"{label} must be an object")
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise OrdinaryUITrajectoryError(
            f"{label} keys differ: missing={missing} extra={extra}"
        )
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrdinaryUITrajectoryError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise OrdinaryUITrajectoryError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return text


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OrdinaryUITrajectoryError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise OrdinaryUITrajectoryError(f"{label} must be >= {minimum}")
    return result


def _require_optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, label, minimum=0)


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OrdinaryUITrajectoryError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OrdinaryUITrajectoryError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise OrdinaryUITrajectoryError(f"{label} must be finite")
    return result


def _require_optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _require_finite(value, label)


def _finite_vector(value: Any, width: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrdinaryUITrajectoryError(f"{label} must be a {width}-element array")
    if len(value) != width:
        raise OrdinaryUITrajectoryError(
            f"{label} must contain exactly {width} values, got {len(value)}"
        )
    return [_require_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _finite_map(value: Any, names: Sequence[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise OrdinaryUITrajectoryError(f"{label} must be an object")
    expected = set(names)
    observed = set(value)
    if observed != expected:
        raise OrdinaryUITrajectoryError(
            f"{label} identity differs: missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )
    return {name: _require_finite(value[name], f"{label}.{name}") for name in names}


def _finite_subset_map(value: Any, names: Sequence[str], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise OrdinaryUITrajectoryError(f"{label} must be an object")
    allowed = set(names)
    extra = set(value) - allowed
    if extra:
        raise OrdinaryUITrajectoryError(f"{label} contains unknown joints: {sorted(extra)}")
    return {
        str(name): _require_finite(raw, f"{label}.{name}")
        for name, raw in value.items()
    }


def _identity_value(value: Any, field: str, label: str) -> Any:
    if field in _HASH_IDENTITY_FIELDS:
        return _require_sha256(value, label)
    if field == "root_state_write_count":
        count = _require_int(value, label, minimum=0)
        if count != 0:
            raise OrdinaryUITrajectoryError(
                f"{label} must be exactly 0 for an ordinary-UI diagnostic"
            )
        return count
    return _require_text(value, label)


def validate_identity_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the external sealed identity envelope.

    Each evidence source carries its own immutable artifact SHA and an explicit
    projection of the identity fields extracted by the caller.  The projection
    is checked against the envelope; absent telemetry fields are never guessed.
    """

    expected_top = frozenset({"schema_version", *_IDENTITY_FIELDS, "evidence"})
    raw = _require_exact_keys(envelope, expected_top, "identity_envelope")
    if raw.get("schema_version") != IDENTITY_SCHEMA:
        raise OrdinaryUITrajectoryError(
            f"identity_envelope.schema_version must equal {IDENTITY_SCHEMA!r}"
        )

    identity = {
        field: _identity_value(raw.get(field), field, f"identity_envelope.{field}")
        for field in _IDENTITY_FIELDS
    }
    evidence_raw = _require_exact_keys(
        raw.get("evidence"), frozenset(_EVIDENCE_NAMES), "identity_envelope.evidence"
    )
    canonical_evidence: dict[str, Any] = {}
    coverage = {field: [] for field in _IDENTITY_FIELDS}
    for evidence_name in _EVIDENCE_NAMES:
        item = _require_exact_keys(
            evidence_raw[evidence_name],
            frozenset({"artifact_sha256", "identity_fields"}),
            f"identity_envelope.evidence.{evidence_name}",
        )
        artifact_sha = _require_sha256(
            item.get("artifact_sha256"),
            f"identity_envelope.evidence.{evidence_name}.artifact_sha256",
        )
        fields = item.get("identity_fields")
        if not isinstance(fields, Mapping):
            raise OrdinaryUITrajectoryError(
                f"identity_envelope.evidence.{evidence_name}.identity_fields must be an object"
            )
        unknown = set(fields) - set(_IDENTITY_FIELDS)
        if unknown:
            raise OrdinaryUITrajectoryError(
                f"identity_envelope.evidence.{evidence_name}.identity_fields "
                f"contains unknown fields: {sorted(unknown)}"
            )
        missing = set(_EVIDENCE_REQUIRED_FIELDS[evidence_name]) - set(fields)
        if missing:
            raise OrdinaryUITrajectoryError(
                f"identity_envelope.evidence.{evidence_name}.identity_fields "
                f"is missing required fields: {sorted(missing)}"
            )
        canonical_fields: dict[str, Any] = {}
        for field, value in fields.items():
            canonical = _identity_value(
                value,
                field,
                f"identity_envelope.evidence.{evidence_name}.identity_fields.{field}",
            )
            if canonical != identity[field]:
                raise OrdinaryUITrajectoryError(
                    f"identity evidence mismatch for {field}: "
                    f"envelope={identity[field]!r} {evidence_name}={canonical!r}"
                )
            canonical_fields[field] = canonical
            coverage[field].append(evidence_name)
        canonical_evidence[evidence_name] = {
            "artifact_sha256": artifact_sha,
            "identity_fields": {
                field: canonical_fields[field] for field in _IDENTITY_FIELDS if field in canonical_fields
            },
        }

    undercovered = {
        field: sources for field, sources in coverage.items() if len(sources) < 2
    }
    if undercovered:
        raise OrdinaryUITrajectoryError(
            "every identity field must be cross-bound by at least two external "
            f"evidence sources; undercovered={undercovered}"
        )
    return {
        "schema_version": IDENTITY_SCHEMA,
        **identity,
        "evidence": canonical_evidence,
    }


def _cross_check_raw_row_identity(
    row: Mapping[str, Any], identity: Mapping[str, Any], row_index: int
) -> None:
    aliases = {
        "source_version": "source_version",
        "accepted_steps_sha256": "accepted_steps_sha256",
        "plan_sha256": "plan_sha256",
        "plan_id": "plan_id",
        "request_id": "request_id",
        "worker_session_id": "worker_session_id",
        "adapter_runtime_instance_id": "adapter_runtime_instance_id",
        "readiness_token_sha256": "readiness_token_sha256",
        "motion_start_readiness_token": "readiness_token_sha256",
        "root_state_write_count": "root_state_write_count",
    }
    for row_field, identity_field in aliases.items():
        if row_field not in row or row[row_field] in (None, ""):
            continue
        observed = _identity_value(
            row[row_field], identity_field, f"telemetry[{row_index}].{row_field}"
        )
        if observed != identity[identity_field]:
            raise OrdinaryUITrajectoryError(
                f"telemetry[{row_index}].{row_field} differs from sealed identity: "
                f"observed={observed!r} expected={identity[identity_field]!r}"
            )


def _required_raw(row: Mapping[str, Any], key: str, row_index: int) -> Any:
    if key not in row:
        raise OrdinaryUITrajectoryError(f"telemetry[{row_index}] is missing {key}")
    return row[key]


def _dispatch_projection(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    def text(key: str) -> str:
        value = _required_raw(row, key, row_index)
        if not isinstance(value, str):
            raise OrdinaryUITrajectoryError(f"telemetry[{row_index}].{key} must be a string")
        return value

    indices = _required_raw(row, "source_event_indices", row_index)
    if not isinstance(indices, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in indices
    ):
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].source_event_indices must be an array of non-negative integers"
        )
    atomic_concurrent = _required_raw(row, "atomic_concurrent", row_index)
    if not isinstance(atomic_concurrent, bool):
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].atomic_concurrent must be boolean"
        )
    return {
        "fsm_state": text("fsm_state"),
        "scheduler_phase": text("scheduler_phase"),
        "macro_state_cursor": text("macro_state_cursor"),
        "command_cursor": _require_optional_int(
            _required_raw(row, "command_cursor", row_index),
            f"telemetry[{row_index}].command_cursor",
        ),
        "segment_cursor": _require_optional_int(
            _required_raw(row, "segment_cursor", row_index),
            f"telemetry[{row_index}].segment_cursor",
        ),
        "source_command": text("source_command"),
        "source_event_index": _require_optional_int(
            _required_raw(row, "source_event_index", row_index),
            f"telemetry[{row_index}].source_event_index",
        ),
        "source_event_indices": list(indices),
        "source_fast_segment": _require_optional_int(
            _required_raw(row, "source_fast_segment", row_index),
            f"telemetry[{row_index}].source_fast_segment",
        ),
        "source_step": _require_optional_int(
            _required_raw(row, "source_step", row_index),
            f"telemetry[{row_index}].source_step",
        ),
        "planned_dispatch_time_s": _require_optional_finite(
            _required_raw(row, "planned_dispatch_time_s", row_index),
            f"telemetry[{row_index}].planned_dispatch_time_s",
        ),
        "actual_dispatch_time_s": _require_optional_finite(
            _required_raw(row, "actual_dispatch_time_s", row_index),
            f"telemetry[{row_index}].actual_dispatch_time_s",
        ),
        "atomic_batch_id": text("atomic_batch_id"),
        "dispatch_kind": text("dispatch_kind"),
        "atomic_concurrent": atomic_concurrent,
        "planned_servo_target_deg": _finite_subset_map(
            _required_raw(row, "planned_servo_target_deg", row_index),
            SERVO_NAMES,
            f"telemetry[{row_index}].planned_servo_target_deg",
        ),
        "planned_wheel_target_rad_s": _finite_subset_map(
            _required_raw(row, "planned_wheel_target_rad_s", row_index),
            WHEEL_NAMES,
            f"telemetry[{row_index}].planned_wheel_target_rad_s",
        ),
    }


def _project_raw_row(
    raw: Mapping[str, Any], identity: Mapping[str, Any], row_index: int
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise OrdinaryUITrajectoryError(f"telemetry[{row_index}] must be an object")
    _cross_check_raw_row_identity(raw, identity, row_index)

    robot_joint_names = _required_raw(raw, "robot_joint_names", row_index)
    if not isinstance(robot_joint_names, list) or tuple(robot_joint_names) != JOINT_NAMES:
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].robot_joint_names must equal the canonical 12-joint order"
        )
    drive_valid = _required_raw(raw, "physx_drive_target_evidence_valid", row_index)
    if drive_valid is not True:
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].physx_drive_target_evidence_valid must be true"
        )

    joints = {
        "measured_position_rad": _finite_map(
            _required_raw(raw, "measured_joint_position_rad", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].measured_joint_position_rad",
        ),
        "measured_velocity_rad_s": _finite_map(
            _required_raw(raw, "measured_joint_velocity_rad_s", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].measured_joint_velocity_rad_s",
        ),
        "physx_position_target_rad": _finite_map(
            _required_raw(raw, "physx_joint_position_target_rad", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].physx_joint_position_target_rad",
        ),
        "position_target_buffer_rad": _finite_map(
            _required_raw(raw, "joint_position_target_buffer_rad", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].joint_position_target_buffer_rad",
        ),
        "physx_velocity_target_rad_s": _finite_map(
            _required_raw(raw, "physx_joint_velocity_target_rad_s", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].physx_joint_velocity_target_rad_s",
        ),
        "velocity_target_buffer_rad_s": _finite_map(
            _required_raw(raw, "joint_velocity_target_buffer_rad_s", row_index),
            JOINT_NAMES,
            f"telemetry[{row_index}].joint_velocity_target_buffer_rad_s",
        ),
        "servo_command_target_rad": _finite_map(
            _required_raw(raw, "servo_command_target_rad", row_index),
            SERVO_NAMES,
            f"telemetry[{row_index}].servo_command_target_rad",
        ),
        "physx_drive_target_evidence_valid": True,
    }
    wheels = {
        "command_velocity_rad_s": _finite_map(
            _required_raw(raw, "wheel_command_velocity_rad_s", row_index),
            WHEEL_NAMES,
            f"telemetry[{row_index}].wheel_command_velocity_rad_s",
        ),
        "canonical_forward_angle_rad": _finite_map(
            _required_raw(raw, "wheel_canonical_forward_angle_rad", row_index),
            LEGS,
            f"telemetry[{row_index}].wheel_canonical_forward_angle_rad",
        ),
        "canonical_forward_velocity_rad_s": _finite_map(
            _required_raw(raw, "wheel_canonical_forward_velocity_rad_s", row_index),
            LEGS,
            f"telemetry[{row_index}].wheel_canonical_forward_velocity_rad_s",
        ),
        "logical_target_rad_s_by_leg": _finite_map(
            _required_raw(raw, "wheel_logical_target_rad_s_by_leg", row_index),
            LEGS,
            f"telemetry[{row_index}].wheel_logical_target_rad_s_by_leg",
        ),
        "physx_target_rad_s_by_leg": _finite_map(
            _required_raw(raw, "wheel_physx_target_rad_s_by_leg", row_index),
            LEGS,
            f"telemetry[{row_index}].wheel_physx_target_rad_s_by_leg",
        ),
        "forward_sign_by_joint": _finite_map(
            _required_raw(raw, "wheel_forward_sign", row_index),
            WHEEL_NAMES,
            f"telemetry[{row_index}].wheel_forward_sign",
        ),
        "wheel_direction": _require_finite(
            _required_raw(raw, "wheel_direction", row_index),
            f"telemetry[{row_index}].wheel_direction",
        ),
    }
    if abs(wheels["wheel_direction"]) != 1.0:
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].wheel_direction must be exactly -1 or 1"
        )
    if any(abs(value) != 1.0 for value in wheels["forward_sign_by_joint"].values()):
        raise OrdinaryUITrajectoryError(
            f"telemetry[{row_index}].wheel_forward_sign values must be exactly -1 or 1"
        )

    result = {
        "schema_version": ROW_SCHEMA,
        "trace_index": row_index,
        "sample_index": _require_int(
            _required_raw(raw, "sample_index", row_index),
            f"telemetry[{row_index}].sample_index",
            minimum=0,
        ),
        "sim_step": _require_int(
            _required_raw(raw, "sim_step", row_index),
            f"telemetry[{row_index}].sim_step",
            minimum=0,
        ),
        "time_s": _require_finite(
            _required_raw(raw, "time_s", row_index), f"telemetry[{row_index}].time_s"
        ),
        "physics_dt_s": _require_finite(
            _required_raw(raw, "physics_dt_s", row_index),
            f"telemetry[{row_index}].physics_dt_s",
        ),
        "source_version": str(identity["source_version"]),
        "root": {
            "pose_w": _finite_vector(
                _required_raw(raw, "root_pose_w", row_index),
                7,
                f"telemetry[{row_index}].root_pose_w",
            ),
            "linear_velocity_w": _finite_vector(
                _required_raw(raw, "root_linear_velocity_w", row_index),
                3,
                f"telemetry[{row_index}].root_linear_velocity_w",
            ),
            "angular_velocity_w": _finite_vector(
                _required_raw(raw, "root_angular_velocity_w", row_index),
                3,
                f"telemetry[{row_index}].root_angular_velocity_w",
            ),
        },
        "com": {
            "position_w": _finite_vector(
                _required_raw(raw, "com_position_w", row_index),
                3,
                f"telemetry[{row_index}].com_position_w",
            ),
            "velocity_w": _finite_vector(
                _required_raw(raw, "com_velocity_w", row_index),
                3,
                f"telemetry[{row_index}].com_velocity_w",
            ),
        },
        "joints": joints,
        "wheels": wheels,
        "dispatch": _dispatch_projection(raw, row_index),
    }
    _validate_cross_readback(result, row_index)
    return result


def _validate_cross_readback(row: Mapping[str, Any], row_index: int) -> None:
    joints = row["joints"]
    wheels = row["wheels"]
    for leg, joint_name in LEG_TO_WHEEL.items():
        logical = wheels["logical_target_rad_s_by_leg"][leg]
        command = wheels["command_velocity_rad_s"][joint_name]
        if logical != command:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{row_index}] logical wheel target differs from command for {leg}"
            )
        physx = wheels["physx_target_rad_s_by_leg"][leg]
        readback = joints["physx_velocity_target_rad_s"][joint_name]
        if physx != readback:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{row_index}] PhysX wheel target differs from joint readback for {leg}"
            )
        sign = wheels["forward_sign_by_joint"][joint_name]
        direction = wheels["wheel_direction"]
        expected_angle = direction * sign * joints["measured_position_rad"][joint_name]
        expected_velocity = direction * sign * joints["measured_velocity_rad_s"][joint_name]
        if wheels["canonical_forward_angle_rad"][leg] != expected_angle:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{row_index}] canonical wheel angle differs for {leg}"
            )
        if wheels["canonical_forward_velocity_rad_s"][leg] != expected_velocity:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{row_index}] canonical wheel velocity differs for {leg}"
            )


def _validate_grid(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise OrdinaryUITrajectoryError("ordinary-UI trajectory must contain at least one row")
    first_sample = _require_int(rows[0]["sample_index"], "trajectory[0].sample_index", minimum=0)
    if first_sample != 0:
        raise OrdinaryUITrajectoryError("ordinary-UI trajectory sample_index must start at 0")
    first_step = _require_int(rows[0]["sim_step"], "trajectory[0].sim_step", minimum=0)
    first_time = _require_finite(rows[0]["time_s"], "trajectory[0].time_s")
    for index, row in enumerate(rows):
        trace_index = _require_int(row["trace_index"], f"trajectory[{index}].trace_index", minimum=0)
        sample_index = _require_int(row["sample_index"], f"trajectory[{index}].sample_index", minimum=0)
        sim_step = _require_int(row["sim_step"], f"trajectory[{index}].sim_step", minimum=0)
        physics_dt = _require_finite(row["physics_dt_s"], f"trajectory[{index}].physics_dt_s")
        time_s = _require_finite(row["time_s"], f"trajectory[{index}].time_s")
        if trace_index != index:
            raise OrdinaryUITrajectoryError(
                f"trajectory trace_index is not continuous: expected={index} got={trace_index}"
            )
        if sample_index != first_sample + index:
            raise OrdinaryUITrajectoryError(
                f"trajectory sample_index is not continuous at row {index}: "
                f"expected={first_sample + index} got={sample_index}"
            )
        if sim_step != first_step + index:
            raise OrdinaryUITrajectoryError(
                f"trajectory sim_step is not a continuous 120 Hz grid at row {index}: "
                f"expected={first_step + index} got={sim_step}"
            )
        if physics_dt != PHYSICS_DT_S:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{index}].physics_dt_s must be exactly 1/120"
            )
        expected_time = first_time + index * PHYSICS_DT_S
        # Only IEEE-754 representation accumulation is tolerated.  This is
        # not a physical or scheduler threshold: sim_step remains exact.
        ulp = max(
            math.ulp(first_time),
            math.ulp(time_s),
            math.ulp(expected_time),
            math.ulp(PHYSICS_DT_S),
        )
        representation_bound = float(index + 2) * ulp
        if abs(time_s - expected_time) > representation_bound:
            raise OrdinaryUITrajectoryError(
                f"trajectory time is off the exact 120 Hz grid at row {index}: "
                f"observed={time_s!r} expected={expected_time!r} "
                f"representation_bound={representation_bound!r}"
            )


def _validate_projected_rows(
    rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _require_exact_keys(raw, _TOP_ROW_KEYS, f"trajectory[{index}]")
        if row.get("schema_version") != ROW_SCHEMA:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{index}].schema_version must equal {ROW_SCHEMA!r}"
            )
        if row.get("source_version") != identity["source_version"]:
            raise OrdinaryUITrajectoryError(
                f"trajectory[{index}].source_version differs from sealed identity"
            )
        root = _require_exact_keys(row.get("root"), _ROOT_KEYS, f"trajectory[{index}].root")
        com = _require_exact_keys(row.get("com"), _COM_KEYS, f"trajectory[{index}].com")
        joints = _require_exact_keys(
            row.get("joints"), _JOINT_KEYS, f"trajectory[{index}].joints"
        )
        wheels = _require_exact_keys(
            row.get("wheels"), _WHEEL_KEYS, f"trajectory[{index}].wheels"
        )
        dispatch = _require_exact_keys(
            row.get("dispatch"), _DISPATCH_KEYS, f"trajectory[{index}].dispatch"
        )
        # Reconstruct a raw-shaped row and pass it through the same projection
        # path.  Exact equality then validates types, finite values, joint
        # identity, wheel/readback consistency, and dispatch layout.
        reconstructed = {
            "sample_index": row.get("sample_index"),
            "sim_step": row.get("sim_step"),
            "time_s": row.get("time_s"),
            "physics_dt_s": row.get("physics_dt_s"),
            "source_version": row.get("source_version"),
            "robot_joint_names": list(JOINT_NAMES),
            "root_pose_w": root.get("pose_w"),
            "root_linear_velocity_w": root.get("linear_velocity_w"),
            "root_angular_velocity_w": root.get("angular_velocity_w"),
            "com_position_w": com.get("position_w"),
            "com_velocity_w": com.get("velocity_w"),
            "measured_joint_position_rad": joints.get("measured_position_rad"),
            "measured_joint_velocity_rad_s": joints.get("measured_velocity_rad_s"),
            "physx_joint_position_target_rad": joints.get("physx_position_target_rad"),
            "joint_position_target_buffer_rad": joints.get("position_target_buffer_rad"),
            "physx_joint_velocity_target_rad_s": joints.get("physx_velocity_target_rad_s"),
            "joint_velocity_target_buffer_rad_s": joints.get("velocity_target_buffer_rad_s"),
            "servo_command_target_rad": joints.get("servo_command_target_rad"),
            "physx_drive_target_evidence_valid": joints.get(
                "physx_drive_target_evidence_valid"
            ),
            "wheel_command_velocity_rad_s": wheels.get("command_velocity_rad_s"),
            "wheel_canonical_forward_angle_rad": wheels.get(
                "canonical_forward_angle_rad"
            ),
            "wheel_canonical_forward_velocity_rad_s": wheels.get(
                "canonical_forward_velocity_rad_s"
            ),
            "wheel_logical_target_rad_s_by_leg": wheels.get(
                "logical_target_rad_s_by_leg"
            ),
            "wheel_physx_target_rad_s_by_leg": wheels.get(
                "physx_target_rad_s_by_leg"
            ),
            "wheel_forward_sign": wheels.get("forward_sign_by_joint"),
            "wheel_direction": wheels.get("wheel_direction"),
            **dict(dispatch),
        }
        rebuilt = _project_raw_row(reconstructed, identity, index)
        if not _strict_json_equal(rebuilt, dict(row)):
            raise OrdinaryUITrajectoryError(
                f"trajectory[{index}] is not the canonical sensor-independent projection"
            )
        canonical.append(rebuilt)
    _validate_grid(canonical)
    return canonical


def _manifest_for(
    rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]
) -> dict[str, Any]:
    trace_payload = _jsonl_bytes(rows)
    identity_payload = _json_bytes(identity)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "diagnostic_kind": "ordinary_ui_sensor_independent_trajectory",
        "diagnostic_complete": True,
        "identity_envelope": dict(identity),
        "identity_envelope_sha256": _sha256_bytes(identity_payload),
        "trajectory": {
            "filename": TRAJECTORY_FILENAME,
            "schema_version": ROW_SCHEMA,
            "sha256": _sha256_bytes(trace_payload),
            "size_bytes": len(trace_payload),
            "row_count": len(rows),
            "physics_rate_hz": PHYSICS_RATE_HZ,
            "physics_dt_fraction": "1/120",
            "first_sim_step": rows[0]["sim_step"],
            "last_sim_step": rows[-1]["sim_step"],
            "first_time_s": rows[0]["time_s"],
            "last_time_s": rows[-1]["time_s"],
            "joint_names": list(JOINT_NAMES),
            "servo_joint_names": list(SERVO_NAMES),
            "wheel_joint_names": list(WHEEL_NAMES),
        },
        "projection": {
            "sensor_independent": True,
            "categories": ["root", "com", "joint_target_readback", "wheel", "dispatch"],
            "contact_fields_projected": [],
            "physical_fields_projected": [],
        },
        "contact_evidence": {
            "available": False,
            "verdict": NOT_EVALUABLE,
            "reason": "ordinary-UI trajectory intentionally contains no contact evidence",
        },
        "physical_evidence": {
            "available": False,
            "full_verdict": NOT_EVALUABLE,
            "reason": (
                "contact, collision, penetration, support, and linkage-lift evidence "
                "is unavailable"
            ),
        },
        "eligibility": {
            "gate1": False,
            "environment_equivalence": False,
            "physical_success_claim": False,
        },
    }


def build_ordinary_ui_trajectory(
    rows: Iterable[Mapping[str, Any]], *, identity_envelope: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a canonical in-memory ordinary-UI trajectory bundle."""

    identity = validate_identity_envelope(identity_envelope)
    projected = [
        _project_raw_row(row, identity, index) for index, row in enumerate(rows)
    ]
    _validate_grid(projected)
    manifest = _manifest_for(projected, identity)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "rows": projected,
        "manifest": manifest,
    }


def _seal_for(trace_payload: bytes, manifest_payload: bytes, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SEAL_SCHEMA,
        "diagnostic_complete": True,
        "identity_envelope_sha256": manifest["identity_envelope_sha256"],
        "files": {
            TRAJECTORY_FILENAME: {
                "sha256": _sha256_bytes(trace_payload),
                "size_bytes": len(trace_payload),
            },
            MANIFEST_FILENAME: {
                "sha256": _sha256_bytes(manifest_payload),
                "size_bytes": len(manifest_payload),
            },
        },
        "full_physical_verdict": NOT_EVALUABLE,
        "gate1_eligible": False,
        "environment_equivalence_eligible": False,
    }


def _write_new_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise OrdinaryUITrajectoryError(f"refusing to overwrite sealed evidence: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise OrdinaryUITrajectoryError(f"refusing to overwrite sealed evidence: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _receipt_for(output_dir: Path, manifest: Mapping[str, Any], seal_payload: bytes) -> dict[str, Any]:
    trajectory = manifest["trajectory"]
    manifest_payload = _json_bytes(manifest)
    return {
        "schema_version": RECEIPT_SCHEMA,
        "output_dir": str(output_dir.resolve()),
        "diagnostic_complete": True,
        "row_count": trajectory["row_count"],
        "identity_envelope_sha256": manifest["identity_envelope_sha256"],
        "trajectory_path": str((output_dir / TRAJECTORY_FILENAME).resolve()),
        "trajectory_sha256": trajectory["sha256"],
        "trajectory_size_bytes": trajectory["size_bytes"],
        "manifest_path": str((output_dir / MANIFEST_FILENAME).resolve()),
        "manifest_sha256": _sha256_bytes(manifest_payload),
        "manifest_size_bytes": len(manifest_payload),
        "seal_path": str((output_dir / SEAL_FILENAME).resolve()),
        "seal_sha256": _sha256_bytes(seal_payload),
        "seal_size_bytes": len(seal_payload),
        "full_physical_verdict": NOT_EVALUABLE,
        "gate1_eligible": False,
        "environment_equivalence_eligible": False,
    }


def write_ordinary_ui_trajectory(
    output_dir: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    identity_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Write and immediately revalidate a three-file diagnostic seal."""

    bundle = build_ordinary_ui_trajectory(rows, identity_envelope=identity_envelope)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trace_payload = _jsonl_bytes(bundle["rows"])
    manifest_payload = _json_bytes(bundle["manifest"])
    seal = _seal_for(trace_payload, manifest_payload, bundle["manifest"])
    seal_payload = _json_bytes(seal)
    for filename in (TRAJECTORY_FILENAME, MANIFEST_FILENAME, SEAL_FILENAME):
        if (directory / filename).exists():
            raise OrdinaryUITrajectoryError(
                f"refusing to overwrite sealed evidence: {directory / filename}"
            )
    _write_new_atomic(directory / TRAJECTORY_FILENAME, trace_payload)
    _write_new_atomic(directory / MANIFEST_FILENAME, manifest_payload)
    # The seal is the last close marker.
    _write_new_atomic(directory / SEAL_FILENAME, seal_payload)
    receipt = validate_ordinary_ui_trajectory(directory)
    expected = _receipt_for(directory, bundle["manifest"], seal_payload)
    if not _strict_json_equal(receipt, expected):
        raise OrdinaryUITrajectoryError("ordinary-UI trajectory write/readback receipt mismatch")
    return receipt


def _read_file(path: Path, label: str) -> bytes:
    if not path.is_file():
        raise OrdinaryUITrajectoryError(f"{label} is missing: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OrdinaryUITrajectoryError(f"could not read {label}: {exc}") from exc


def _load_jsonl(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory JSONL must be non-empty and newline-terminated"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OrdinaryUITrajectoryError("ordinary-UI trajectory is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        if not line:
            raise OrdinaryUITrajectoryError(
                f"ordinary-UI trajectory contains a blank JSONL row at line {index + 1}"
            )
        value = _strict_loads(line, label=f"ordinary-UI trajectory line {index + 1}")
        if not isinstance(value, dict):
            raise OrdinaryUITrajectoryError(
                f"ordinary-UI trajectory line {index + 1} must be an object"
            )
        rows.append(value)
    return rows


def validate_ordinary_ui_trajectory(output_dir: str | Path) -> dict[str, Any]:
    """Strictly reload, hash, and semantically validate a diagnostic seal."""

    directory = Path(output_dir)
    trace_payload = _read_file(directory / TRAJECTORY_FILENAME, "trajectory")
    manifest_payload = _read_file(directory / MANIFEST_FILENAME, "manifest")
    seal_payload = _read_file(directory / SEAL_FILENAME, "seal")
    try:
        manifest_text = manifest_payload.decode("utf-8")
        seal_text = seal_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OrdinaryUITrajectoryError("manifest/seal must be UTF-8") from exc
    manifest = _strict_loads(manifest_text, label="ordinary-UI trajectory manifest")
    seal = _strict_loads(seal_text, label="ordinary-UI trajectory seal")
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise OrdinaryUITrajectoryError(
            f"manifest.schema_version must equal {MANIFEST_SCHEMA!r}"
        )
    if not isinstance(seal, Mapping) or seal.get("schema_version") != SEAL_SCHEMA:
        raise OrdinaryUITrajectoryError(f"seal.schema_version must equal {SEAL_SCHEMA!r}")
    identity = validate_identity_envelope(manifest.get("identity_envelope"))
    rows = _load_jsonl(trace_payload)
    canonical_rows = _validate_projected_rows(rows, identity)
    canonical_trace_payload = _jsonl_bytes(canonical_rows)
    if trace_payload != canonical_trace_payload:
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory bytes differ from canonical JSONL serialization"
        )
    expected_manifest = _manifest_for(canonical_rows, identity)
    if not _strict_json_equal(dict(manifest), expected_manifest):
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory manifest differs from canonical recomputation"
        )
    canonical_manifest_payload = _json_bytes(expected_manifest)
    if manifest_payload != canonical_manifest_payload:
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory manifest bytes differ from canonical serialization"
        )
    expected_seal = _seal_for(trace_payload, manifest_payload, expected_manifest)
    if not _strict_json_equal(dict(seal), expected_seal):
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory seal differs from file hashes or fixed eligibility verdicts"
        )
    if seal_payload != _json_bytes(expected_seal):
        raise OrdinaryUITrajectoryError(
            "ordinary-UI trajectory seal bytes differ from canonical serialization"
        )
    return _receipt_for(directory, expected_manifest, seal_payload)


def ordinary_ui_diagnostic_complete(value: Mapping[str, Any] | str | Path) -> bool:
    """Return true only after full in-memory or on-disk diagnostic validation."""

    try:
        if isinstance(value, (str, Path)):
            validate_ordinary_ui_trajectory(value)
            return True
        if not isinstance(value, Mapping):
            return False
        schema = value.get("schema_version")
        if schema == RECEIPT_SCHEMA:
            observed = validate_ordinary_ui_trajectory(
                _require_text(value.get("output_dir"), "receipt.output_dir")
            )
            return _strict_json_equal(observed, dict(value))
        if schema == BUNDLE_SCHEMA:
            if set(value) != {"schema_version", "rows", "manifest"}:
                return False
            manifest = value.get("manifest")
            rows = value.get("rows")
            if not isinstance(manifest, Mapping) or not isinstance(rows, list):
                return False
            identity = validate_identity_envelope(manifest.get("identity_envelope"))
            canonical_rows = _validate_projected_rows(rows, identity)
            return _strict_json_equal(
                dict(manifest), _manifest_for(canonical_rows, identity)
            )
        return False
    except (OrdinaryUITrajectoryError, OSError, TypeError, ValueError):
        return False


__all__ = [
    "BUNDLE_SCHEMA",
    "IDENTITY_SCHEMA",
    "JOINT_NAMES",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "NOT_EVALUABLE",
    "OrdinaryUITrajectoryError",
    "PHYSICS_DT_S",
    "PHYSICS_RATE_HZ",
    "RECEIPT_SCHEMA",
    "ROW_SCHEMA",
    "SEAL_FILENAME",
    "SEAL_SCHEMA",
    "TRAJECTORY_FILENAME",
    "build_ordinary_ui_trajectory",
    "ordinary_ui_diagnostic_complete",
    "validate_identity_envelope",
    "validate_ordinary_ui_trajectory",
    "write_ordinary_ui_trajectory",
]
