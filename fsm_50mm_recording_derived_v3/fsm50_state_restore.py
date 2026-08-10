"""Fail-closed, Isaac-free provenance validation for non-A0 test starts.

The module deliberately performs no restore and no prefix replay.  It validates
one of two durable evidence packages and returns the exact two values a caller
needs: ``restore_provenance`` for :class:`FSM50Controller`, and a normalized
``sim_state`` for ``adapter.restore_sim_state`` (``None`` for prefix replay).

Trust is anchored by caller-supplied SHA256 values.  Paths and hashes merely
declared inside an untrusted JSON document are never sufficient by themselves.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TRUSTED_SIM_STATE_SCHEMA = "fsm50.trusted_sim_state_before.v1"
VERIFIED_PREFIX_REPLAY_SCHEMA = "fsm50.verified_prefix_replay_manifest.v1"
A0_STATE_ID = "A0_RESET_AND_SETTLE"


class RestoreProvenanceError(ValueError):
    """Raised when restore/prefix evidence cannot be trusted completely."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RestoreProvenanceError(f"{label} must be a 64-character SHA256")
    return digest


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RestoreProvenanceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise RestoreProvenanceError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except RestoreProvenanceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestoreProvenanceError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RestoreProvenanceError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_declared_path(value: Any, *, base: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RestoreProvenanceError(f"{label} is required")
    declared = Path(text)
    return (declared if declared.is_absolute() else base / declared).resolve()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _declared_environment_hash(payload: Mapping[str, Any]) -> str:
    direct = str(payload.get("environment_fingerprint_sha256", "") or "").strip()
    nested = payload.get("environment_fingerprint")
    nested_hash = (
        str(nested.get("sha256", "") or "").strip()
        if isinstance(nested, Mapping)
        else ""
    )
    values = {value.lower() for value in (direct, nested_hash) if value}
    if len(values) != 1:
        raise RestoreProvenanceError(
            "exactly one consistent environment fingerprint SHA256 is required"
        )
    return _validated_sha256(values.pop(), label="environment_fingerprint_sha256")


def _matching_state_id(payload: Mapping[str, Any], expected: str, *, label: str) -> str:
    values = {
        str(payload.get(key, "") or "").strip()
        for key in ("state_id", "target_state_id")
        if str(payload.get(key, "") or "").strip()
    }
    if values != {expected}:
        raise RestoreProvenanceError(
            f"{label} state id mismatch: expected {expected!r}, observed {sorted(values)!r}"
        )
    return expected


def _verify_file_hash(path: Path, expected: Any, *, label: str) -> str:
    expected_digest = _validated_sha256(expected, label=f"expected {label} SHA256")
    if not path.is_file():
        raise RestoreProvenanceError(f"{label} file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise RestoreProvenanceError(
            f"{label} SHA256 mismatch: expected={expected_digest} actual={actual}"
        )
    return actual


def _single_finite_row(value: Any, *, width: int | None, label: str) -> list[float]:
    row = value
    while (
        isinstance(row, (list, tuple))
        and len(row) == 1
        and isinstance(row[0], (list, tuple))
    ):
        row = row[0]
    if not isinstance(row, (list, tuple)):
        raise RestoreProvenanceError(f"sim_state_before.{label} must be an array")
    if width is not None and len(row) != width:
        raise RestoreProvenanceError(
            f"sim_state_before.{label} must contain {width} values"
        )
    try:
        result = [float(item) for item in row]
    except (TypeError, ValueError) as exc:
        raise RestoreProvenanceError(
            f"sim_state_before.{label} must contain only numbers"
        ) from exc
    if not result or not all(math.isfinite(item) for item in result):
        raise RestoreProvenanceError(
            f"sim_state_before.{label} must contain finite values"
        )
    return result


def _finite_command_mapping(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise RestoreProvenanceError(f"sim_state_before.command_state.{label} is required")
    result: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name).strip()
        if not name or name in result:
            raise RestoreProvenanceError(
                f"sim_state_before.command_state.{label} has an invalid joint name"
            )
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RestoreProvenanceError(
                f"sim_state_before.command_state.{label}.{name} is not numeric"
            ) from exc
        if not math.isfinite(number):
            raise RestoreProvenanceError(
                f"sim_state_before.command_state.{label}.{name} is not finite"
            )
        result[name] = number
    return result


def _normalize_complete_sim_state(
    value: Any,
    *,
    expected_joint_names: Sequence[str] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RestoreProvenanceError("sim_state_before must be a JSON object")
    state = dict(value)
    if state.get("pose_restore_eligible") is False:
        raise RestoreProvenanceError("sim_state_before is explicitly not pose-restore eligible")
    capture_source = str(state.get("capture_source", "") or "").lower()
    if capture_source in {"nullsimrobotadapter", "no_sim", "no-sim"}:
        raise RestoreProvenanceError("NullSim/no-sim checkpoints cannot be restored")

    root_pose = _single_finite_row(state.get("root_pose"), width=7, label="root_pose")
    root_velocity = _single_finite_row(
        state.get("root_velocity"), width=6, label="root_velocity"
    )
    if math.sqrt(sum(component * component for component in root_pose[3:7])) <= 1.0e-12:
        raise RestoreProvenanceError("sim_state_before.root_pose quaternion is degenerate")

    raw_names = state.get("joint_names")
    if not isinstance(raw_names, (list, tuple)) or not raw_names:
        raise RestoreProvenanceError("sim_state_before.joint_names is required")
    joint_names = [str(name).strip() for name in raw_names]
    if any(not name for name in joint_names) or len(joint_names) != len(set(joint_names)):
        raise RestoreProvenanceError(
            "sim_state_before.joint_names must be non-empty and unique"
        )
    joint_pos = _single_finite_row(
        state.get("joint_pos"), width=len(joint_names), label="joint_pos"
    )
    joint_vel = _single_finite_row(
        state.get("joint_vel"), width=len(joint_names), label="joint_vel"
    )
    if expected_joint_names is not None:
        expected = [str(name) for name in expected_joint_names]
        if (
            not expected
            or len(expected) != len(set(expected))
            or set(expected) != set(joint_names)
        ):
            raise RestoreProvenanceError(
                "sim_state_before joint names do not match the current articulation"
            )

    command_state = state.get("command_state")
    if not isinstance(command_state, Mapping):
        raise RestoreProvenanceError("sim_state_before.command_state is required")
    servos = _finite_command_mapping(command_state.get("servos"), label="servos")
    wheels = _finite_command_mapping(command_state.get("wheels"), label="wheels")
    overlap = set(servos) & set(wheels)
    commanded = set(servos) | set(wheels)
    if overlap or commanded != set(joint_names):
        raise RestoreProvenanceError(
            "command_state must cover every joint name exactly once across servos/wheels"
        )

    normalized = dict(state)
    normalized.update(
        {
            "root_pose": [root_pose],
            "root_velocity": [root_velocity],
            "joint_pos": [joint_pos],
            "joint_vel": [joint_vel],
            "joint_names": joint_names,
            "command_state": {"servos": servos, "wheels": wheels},
        }
    )
    return normalized


def _verify_finalized_source_result(
    evidence: Mapping[str, Any],
    *,
    evidence_path: Path,
    target_state_id: str,
    environment_fingerprint_sha256: str,
) -> dict[str, Any]:
    source_run = _resolve_declared_path(
        evidence.get("source_run_directory"),
        base=evidence_path.parent,
        label="source_run_directory",
    )
    if not source_run.is_dir():
        raise RestoreProvenanceError(f"source run directory is missing: {source_run}")
    source_result_value = evidence.get("source_result_path", "result.json")
    source_result_declared = Path(str(source_result_value or "result.json"))
    source_result = (
        source_result_declared.resolve()
        if source_result_declared.is_absolute()
        else (source_run / source_result_declared).resolve()
    )
    if source_result.name != "result.json" or source_result.parent != source_run:
        raise RestoreProvenanceError(
            "source_result_path must be the source run's direct result.json"
        )
    result_hash = _verify_file_hash(
        source_result,
        evidence.get("source_result_sha256", evidence.get("source_sha256")),
        label="source result",
    )
    result = _read_json_mapping(source_result, label="source result")
    if not str(result.get("schema_version", "") or "").strip():
        raise RestoreProvenanceError("source result schema_version is required")
    captured_state_ids = result.get("captured_state_ids")
    if isinstance(captured_state_ids, list):
        normalized_captured = [str(item).strip() for item in captured_state_ids]
        if (
            any(not item for item in normalized_captured)
            or len(normalized_captured) != len(set(normalized_captured))
            or target_state_id not in normalized_captured
        ):
            raise RestoreProvenanceError(
                "source result does not list the target in captured_state_ids"
            )
    else:
        # Backward-compatible path for a state-specific source result.  A full
        # controller run legitimately contains snapshots for many states and
        # therefore uses the explicit captured_state_ids membership above.
        _matching_state_id(result, target_state_id, label="source result")
    result_environment = _declared_environment_hash(result)
    if result_environment != environment_fingerprint_sha256:
        raise RestoreProvenanceError("source result environment fingerprint mismatch")
    result_run = _resolve_declared_path(
        result.get("run_dir"), base=source_result.parent, label="source result run_dir"
    )
    if result_run != source_run:
        raise RestoreProvenanceError("source result run_dir does not match source_run_directory")
    if result.get("artifact_valid") is not True:
        raise RestoreProvenanceError("source result artifact_valid must be true")
    lifecycle = result.get("lifecycle")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("finalized") is not True
        or lifecycle.get("failed") is True
    ):
        raise RestoreProvenanceError("source result lifecycle is not finalized")
    artifact_root = _resolve_declared_path(
        result.get("artifact_root"),
        base=source_result.parent,
        label="source result artifact_root",
    )
    if (
        not artifact_root.is_dir()
        or not _path_is_within(source_run, artifact_root)
        or not _path_is_within(evidence_path, artifact_root)
        or not (artifact_root / ".finalized").is_file()
        or (artifact_root / ".partial").exists()
        or (artifact_root / ".failed").exists()
    ):
        raise RestoreProvenanceError(
            "source artifact must be finalized and contain the run/evidence"
        )
    return {
        "source_run_directory": str(source_run),
        "source_result_path": str(source_result),
        "source_result_sha256": result_hash,
        "artifact_root": str(artifact_root),
        "source_result_schema_version": str(result["schema_version"]),
    }


def _validate_trusted_sim_state(
    *,
    target_state_id: str,
    environment_fingerprint_sha256: str,
    path: Path,
    expected_sha256: str,
    expected_joint_names: Sequence[str] | None,
) -> dict[str, Any]:
    evidence_sha256 = _verify_file_hash(
        path, expected_sha256, label="sim_state_before evidence"
    )
    evidence = _read_json_mapping(path, label="sim_state_before evidence")
    if str(evidence.get("schema_version", "")) != TRUSTED_SIM_STATE_SCHEMA:
        raise RestoreProvenanceError("unsupported sim_state_before evidence schema")
    _matching_state_id(evidence, target_state_id, label="sim_state_before evidence")
    evidence_environment = _declared_environment_hash(evidence)
    if evidence_environment != environment_fingerprint_sha256:
        raise RestoreProvenanceError("sim_state_before environment fingerprint mismatch")
    source = _verify_finalized_source_result(
        evidence,
        evidence_path=path,
        target_state_id=target_state_id,
        environment_fingerprint_sha256=environment_fingerprint_sha256,
    )
    sim_state = _normalize_complete_sim_state(
        evidence.get("sim_state_before"),
        expected_joint_names=expected_joint_names,
    )
    provenance = {
        "method": "TRUSTED_SIM_STATE_RESTORE",
        "validated": True,
        "state_id": target_state_id,
        "source_run_directory": source["source_run_directory"],
        "source_sha256": evidence_sha256,
        "source_result_path": source["source_result_path"],
        "source_result_sha256": source["source_result_sha256"],
        "artifact_root": source["artifact_root"],
        "environment_fingerprint_sha256": environment_fingerprint_sha256,
        "evidence_path": str(path),
        "evidence_schema_version": TRUSTED_SIM_STATE_SCHEMA,
    }
    return {"restore_provenance": provenance, "sim_state": sim_state}


def _validate_verified_prefix(
    *,
    target_state_id: str,
    environment_fingerprint_sha256: str,
    path: Path,
    expected_sha256: str,
    state_order: Sequence[str] | None,
) -> dict[str, Any]:
    evidence_sha256 = _verify_file_hash(
        path, expected_sha256, label="prefix replay manifest"
    )
    manifest = _read_json_mapping(path, label="prefix replay manifest")
    if str(manifest.get("schema_version", "")) != VERIFIED_PREFIX_REPLAY_SCHEMA:
        raise RestoreProvenanceError("unsupported prefix replay manifest schema")
    if manifest.get("verified") is not True or manifest.get("prefix_complete") is not True:
        raise RestoreProvenanceError(
            "prefix replay manifest requires verified=true and prefix_complete=true"
        )
    _matching_state_id(manifest, target_state_id, label="prefix replay manifest")
    manifest_environment = _declared_environment_hash(manifest)
    if manifest_environment != environment_fingerprint_sha256:
        raise RestoreProvenanceError("prefix replay environment fingerprint mismatch")

    order = [str(state_id) for state_id in list(state_order or [])]
    if (
        not order
        or order[0] != A0_STATE_ID
        or len(order) != len(set(order))
        or target_state_id not in order
    ):
        raise RestoreProvenanceError(
            "state_order must be unique, begin at A0, and contain the target state"
        )
    completed = manifest.get("completed_prefix")
    if not isinstance(completed, list) or any(not str(item).strip() for item in completed):
        raise RestoreProvenanceError("completed_prefix must be a JSON array of state ids")
    completed_prefix = [str(item) for item in completed]
    expected_prefix = order[: order.index(target_state_id)]
    if completed_prefix != expected_prefix:
        raise RestoreProvenanceError(
            f"completed_prefix mismatch: expected={expected_prefix!r} "
            f"observed={completed_prefix!r}"
        )
    source = _verify_finalized_source_result(
        manifest,
        evidence_path=path,
        target_state_id=target_state_id,
        environment_fingerprint_sha256=environment_fingerprint_sha256,
    )
    provenance = {
        "method": "VERIFIED_PREFIX_REPLAY",
        "validated": True,
        "state_id": target_state_id,
        "source_run_directory": source["source_run_directory"],
        "source_sha256": evidence_sha256,
        "source_result_path": source["source_result_path"],
        "source_result_sha256": source["source_result_sha256"],
        "artifact_root": source["artifact_root"],
        "environment_fingerprint_sha256": environment_fingerprint_sha256,
        "evidence_path": str(path),
        "evidence_schema_version": VERIFIED_PREFIX_REPLAY_SCHEMA,
        "completed_prefix": completed_prefix,
    }
    return {"restore_provenance": provenance, "sim_state": None}


def validate_state_restore(
    *,
    target_state_id: str,
    environment_fingerprint_sha256: str = "",
    sim_state_before_path: str | Path | None = None,
    sim_state_before_sha256: str = "",
    prefix_replay_manifest_path: str | Path | None = None,
    prefix_replay_manifest_sha256: str = "",
    state_order: Sequence[str] | None = None,
    expected_joint_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate exactly one state-start method and return controller inputs.

    A0 may start from a clean reset with no evidence.  Every other state must
    supply exactly one externally hashed direct-state or prefix-replay file.
    """

    target_state_id = str(target_state_id or "").strip()
    if not target_state_id:
        raise RestoreProvenanceError("target_state_id is required")
    direct = sim_state_before_path is not None
    prefix = prefix_replay_manifest_path is not None
    if direct and prefix:
        raise RestoreProvenanceError(
            "choose exactly one of sim_state_before or verified prefix replay"
        )
    if not direct and not prefix:
        if target_state_id != A0_STATE_ID:
            raise RestoreProvenanceError(
                "non-A0 state requires trusted sim_state_before or verified prefix replay"
            )
        return {
            "restore_provenance": {
                "method": "CLEAN_A0_RESET",
                "validated": True,
                "state_id": target_state_id,
            },
            "sim_state": None,
        }

    environment_hash = _validated_sha256(
        environment_fingerprint_sha256,
        label="current environment_fingerprint_sha256",
    )
    if direct:
        path = Path(sim_state_before_path).resolve()  # type: ignore[arg-type]
        return _validate_trusted_sim_state(
            target_state_id=target_state_id,
            environment_fingerprint_sha256=environment_hash,
            path=path,
            expected_sha256=sim_state_before_sha256,
            expected_joint_names=expected_joint_names,
        )
    path = Path(prefix_replay_manifest_path).resolve()  # type: ignore[arg-type]
    return _validate_verified_prefix(
        target_state_id=target_state_id,
        environment_fingerprint_sha256=environment_hash,
        path=path,
        expected_sha256=prefix_replay_manifest_sha256,
        state_order=state_order,
    )


# Explicit aliases make the intent clear at call sites and keep the module easy
# to adopt without coupling it to a particular runner implementation.
load_validated_state_restore = validate_state_restore
validate_state_restore_provenance = validate_state_restore


__all__ = [
    "A0_STATE_ID",
    "RestoreProvenanceError",
    "TRUSTED_SIM_STATE_SCHEMA",
    "VERIFIED_PREFIX_REPLAY_SCHEMA",
    "load_validated_state_restore",
    "sha256_file",
    "validate_state_restore",
    "validate_state_restore_provenance",
]
