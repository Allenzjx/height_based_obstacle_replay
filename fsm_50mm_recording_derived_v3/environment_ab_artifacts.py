"""Convert finalized replay artifacts into the environment A/A--A/B gate.

This module is deliberately Isaac-free.  It consumes durable artifacts only;
it never imports a simulator package and never reconstructs missing telemetry.
Malformed, partial, camera-only, source-mismatched, or non-finite required
evidence fails closed and is still recorded in the requested JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from command_model import SERVO_JOINT_NAMES
from fsm_50mm_recording_derived_v3.environment_equivalence import (
    build_static_environment_fingerprint,
    compare_instrumentation_configs,
    compare_trajectory_equivalence,
    sha256_file,
    write_environment_equivalence_report,
)
from fsm_50mm_recording_derived_v3.shutdown_contract import (
    validate_shutdown_outcome,
)


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = MODULE_ROOT / "reports" / "ENVIRONMENT_EQUIVALENCE_REPORT.json"
RESULT_SCHEMA = "fsm50.recording_replay_result.v1"
CONVERSION_SCHEMA = "fsm50.environment_ab_artifacts.v1"
TELEMETRY_FINALIZATION_SCHEMA = "telemetry.canonical_finalization.v1"
LEGS = ("FL", "FR", "RL", "RR")
CONTACT_CLASSES = frozenset({"GROUND", "FRONT_FACE", "TOP", "AIR", "UNKNOWN"})
COMMON_WHEEL_FORCE_SOURCE = "isaaclab.ContactSensor.net_forces_w"
SHA256_HEX = frozenset("0123456789abcdef")
DIRECT_VIEWPORT_CAPTURE_BACKEND = (
    "active_viewport_ldr_byte_buffer_to_omni_videoencoding"
)
DIRECT_VIEWPORT_BUFFER_SCHEMA = "fsm50.active_viewport_buffer_video.v1"
DIRECT_VIEWPORT_SOURCE = "actual_active_isaac_gui_viewport_render_product"
FORMAL_WORKER_ARTIFACT_OWNER = "sim_worker_process"
FORMAL_WORKER_EXECUTION_PATH = "sim_worker_process_ipc"
TRAJECTORY_DIAGNOSTIC_SCOPE = "TRAJECTORY_DIAGNOSTIC_ONLY"
TRAJECTORY_DIAGNOSTIC_PHYSICAL_CLAIM = "NO_PHYSICAL_CLAIM"
TRAJECTORY_COMPARISON_QUALIFICATION_SCOPE = "TRAJECTORY_COMPARISON"
TRAJECTORY_DIAGNOSTIC_ROLE_CONTACT_MODES = {
    "A1": "formal",
    "A2": "formal",
    "B": "instrumented",
}
TRAJECTORY_DIAGNOSTIC_DISPATCH_FILENAMES = (
    "motion_start_readiness.json",
    "motion_start_pre_first_dispatch.json",
    "V003_DISPATCH_TRACE.csv",
    "V003_DISPATCH_TRACE.json",
    "production_dispatch_timing.json",
)
SENSOR_INDEPENDENT_TRAJECTORY_METRICS = (
    "root_trajectory",
    "joint_trajectory",
    "wheel_rotation",
    "wheel_travel",
    "final_pose",
    "obstacle_geometry",
)
OBSERVATION_TRAJECTORY_METRICS = ("contact_class", "contact_force")
WORKER_BATCH_CONTROL_FILENAMES = (
    "worker_artifact_request.json",
    "worker_startup_binding.json",
    "worker_preplay_stop_ack.json",
    "worker_playback_start_ack.json",
    "worker_artifact_complete_ack.json",
)

REQUIRED_RUN_FILENAMES = (
    "result.json",
    "fsm50_telemetry.jsonl",
    "runtime_environment.json",
    "visual_recording_manifest.json",
    "physical_evidence.json",
    "failure_diagnostics.json",
    "telemetry_finalization.json",
)

REQUIRED_CANONICAL_TELEMETRY_FILENAMES = frozenset(
    {
        "telemetry_samples.csv",
        "body_com_timeseries.csv",
        "joint_timeseries.csv",
        "contacts.csv",
        "events.jsonl",
        "stability_summary.json",
        "fsm50_telemetry.csv",
        "fsm50_telemetry.jsonl",
        "wheel_filtered_contacts.jsonl",
        "nonwheel_obstacle_contacts.csv",
        "nonwheel_obstacle_contacts.jsonl",
        "state_timeline.csv",
        "physical_evidence.json",
    }
)


class ArtifactValidationError(ValueError):
    """One or more replay artifacts cannot be admitted to the A/B gate."""

    def __init__(self, failures: str | Sequence[str]):
        rows = [str(failure) for failure in ([failures] if isinstance(failures, str) else failures)]
        self.failures = tuple(row for row in rows if row)
        super().__init__("; ".join(self.failures) or "artifact validation failed")


@dataclass(frozen=True)
class ReplayArtifact:
    role: str
    artifact_root: Path
    run_dir: Path
    result: dict[str, Any]
    runtime_environment: dict[str, Any]
    visual_manifest: dict[str, Any]
    fast_plan: dict[str, Any]
    telemetry_rows: tuple[dict[str, Any], ...]
    trajectory_metrics: dict[str, Any]
    provenance: dict[str, Any]


def _decode_strict_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def _type_strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values while preserving bool/int/float type distinctions."""

    try:
        encode = lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return encode(left) == encode(right)
    except (TypeError, ValueError):
        return False


def _strict_json(path: Path) -> Any:
    try:
        return _decode_strict_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactValidationError(f"{path}: invalid strict JSON: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(f"{path}: unreadable telemetry JSONL: {exc}") from exc
    if not raw_lines:
        raise ArtifactValidationError(f"{path}: telemetry is empty")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            raise ArtifactValidationError(f"{path}:{line_number}: blank JSONL row")
        try:
            row = _decode_strict_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"{path}:{line_number}: invalid strict JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ArtifactValidationError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{label}: expected object")
    return dict(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{label}: boolean is not numeric evidence")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{label}: missing or non-numeric value {value!r}") from exc
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{label}: non-finite value {value!r}")
    return number


def _integer(value: Any, label: str) -> int:
    number = _finite(value, label)
    integer = int(number)
    if number != integer:
        raise ArtifactValidationError(f"{label}: expected integer, got {value!r}")
    return integer


def _exact_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ArtifactValidationError(
            f"{label}: expected exact JSON integer, got {value!r}"
        )
    return value


def _vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ArtifactValidationError(f"{label}: expected {length}-element sequence")
    if len(value) != length:
        raise ArtifactValidationError(f"{label}: expected length {length}, got {len(value)}")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in SHA256_HEX for character in digest):
        raise ArtifactValidationError(f"{label}: expected lowercase/uppercase SHA-256 hex")
    return digest


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_telemetry_finalization(
    path: Path,
    payload: dict[str, Any],
    *,
    run_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    if str(payload.get("schema_version", "")) != TELEMETRY_FINALIZATION_SCHEMA:
        raise ArtifactValidationError(
            f"{path}: unsupported telemetry finalization schema"
        )
    if payload.get("canonical_export_attempted") is not True:
        raise ArtifactValidationError(
            f"{path}: canonical_export_attempted is not true"
        )
    if payload.get("canonical_complete") is not True:
        raise ArtifactValidationError(f"{path}: canonical_complete is not true")
    errors = payload.get("errors")
    if not isinstance(errors, list) or errors:
        raise ArtifactValidationError(f"{path}: finalization errors are not empty")
    journal = _mapping(payload.get("journal"), f"{path}.journal")
    journal_errors = journal.get("errors")
    if not isinstance(journal_errors, list) or journal_errors:
        raise ArtifactValidationError(f"{path}: journal errors are not empty")
    if journal.get("removed_after_success") is not True:
        raise ArtifactValidationError(
            f"{path}: journal was not removed after canonical success"
        )
    if (run_dir / ".telemetry_journal").exists():
        raise ArtifactValidationError(
            f"{path}: successful run retains .telemetry_journal"
        )
    files = _mapping(payload.get("canonical_files"), f"{path}.canonical_files")
    missing = sorted(REQUIRED_CANONICAL_TELEMETRY_FILENAMES - set(files))
    if missing:
        raise ArtifactValidationError(
            f"{path}: canonical telemetry files are missing: {missing}"
        )
    normalized_files: dict[str, Any] = {}
    for relative, raw in files.items():
        name = str(relative or "").replace("\\", "/")
        candidate = (run_dir / name).resolve()
        if not name or Path(name).is_absolute() or not _is_within(candidate, run_dir):
            raise ArtifactValidationError(
                f"{path}: canonical telemetry path escapes run_dir: {relative!r}"
            )
        row = _mapping(raw, f"{path}.canonical_files[{name!r}]")
        expected_sha = _sha256(row.get("sha256"), f"{path}.{name}.sha256")
        expected_size = _integer(
            row.get("size_bytes"), f"{path}.{name}.size_bytes"
        )
        record_count = _integer(
            row.get("record_count"), f"{path}.{name}.record_count"
        )
        if expected_size < 0 or record_count < 0:
            raise ArtifactValidationError(
                f"{path}: negative size/count for canonical file {name}"
            )
        if not candidate.is_file():
            raise ArtifactValidationError(
                f"{path}: canonical telemetry file is missing: {candidate}"
            )
        if (
            candidate.stat().st_size != expected_size
            or sha256_file(candidate).lower() != expected_sha
        ):
            raise ArtifactValidationError(
                f"{path}: canonical telemetry hash/size mismatch: {name}"
            )
        normalized_files[name] = {
            "sha256": expected_sha,
            "size_bytes": expected_size,
            "record_count": record_count,
        }

    result_finalization = _mapping(
        result.get("telemetry_finalization"), "result.telemetry_finalization"
    )
    marker_path = Path(str(result_finalization.get("marker_path", "") or "")).resolve()
    if marker_path != path.resolve():
        raise ArtifactValidationError(
            "result.telemetry_finalization.marker_path differs from marker"
        )
    if _sha256(
        result_finalization.get("marker_sha256"),
        "result.telemetry_finalization.marker_sha256",
    ) != sha256_file(path).lower():
        raise ArtifactValidationError(
            "result.telemetry_finalization.marker_sha256 differs from marker"
        )
    for key in (
        "schema_version",
        "canonical_export_attempted",
        "canonical_complete",
        "stream_counts",
        "canonical_files",
        "journal",
        "errors",
    ):
        if result_finalization.get(key) != payload.get(key):
            raise ArtifactValidationError(
                f"result.telemetry_finalization.{key} differs from marker"
            )
    return {
        "canonical_files": normalized_files,
        "stream_counts": _mapping(
            payload.get("stream_counts"), f"{path}.stream_counts"
        ),
        "marker_sha256": sha256_file(path).lower(),
    }


def _resolve_run_dir(path: str | Path) -> Path:
    requested = Path(path).resolve()
    if not requested.is_dir():
        raise ArtifactValidationError(f"run path is not a directory: {requested}")
    if (requested / "result.json").is_file():
        return requested
    pointer_path = requested / "artifact_pointer.json"
    if pointer_path.is_file():
        pointer = _mapping(_strict_json(pointer_path), str(pointer_path))
        pointed = Path(str(pointer.get("run_dir", ""))).resolve()
        if pointed.is_dir() and (pointed / "result.json").is_file() and _is_within(pointed, requested):
            return pointed
        raise ArtifactValidationError(f"{pointer_path}: invalid or escaping run_dir pointer")
    candidates = sorted(requested.rglob("result.json"))
    if len(candidates) != 1:
        raise ArtifactValidationError(
            f"{requested}: expected exactly one nested result.json, found {len(candidates)}"
        )
    return candidates[0].parent.resolve()


def _resolve_contact_mode(
    *,
    role: str,
    result: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    artifact_root: Path,
) -> str:
    candidates: list[tuple[str, str]] = []
    for source, payload in (("result", result), ("runtime_environment", runtime_environment)):
        if payload.get("contact_mode") is not None:
            candidates.append((source, str(payload.get("contact_mode", "")).strip().lower()))
    # Production stores the mode in the containing batch readback.  Search a
    # bounded ancestor chain and accept only a readback whose batch_root points
    # to its own directory (when the field is present).
    for directory in (artifact_root, *artifact_root.parents[:4]):
        readback_path = directory / "runtime_environment_readback.json"
        if not readback_path.is_file():
            continue
        readback = _mapping(_strict_json(readback_path), str(readback_path))
        batch_root = str(readback.get("batch_root", "") or "")
        if batch_root and Path(batch_root).resolve() != directory.resolve():
            raise ArtifactValidationError(f"{readback_path}: batch_root pointer mismatch")
        if readback.get("contact_mode") is not None:
            candidates.append(
                (str(readback_path), str(readback.get("contact_mode", "")).strip().lower())
            )
        break
    if not candidates:
        raise ArtifactValidationError(f"{role}: contact_mode evidence is missing")
    modes = {value for _source, value in candidates}
    if len(modes) != 1 or not modes <= {"formal", "instrumented"}:
        raise ArtifactValidationError(f"{role}: conflicting/unsupported contact_mode evidence {candidates!r}")
    mode = next(iter(modes))
    expected = {"A1": "formal", "A2": "formal", "B": "instrumented"}.get(
        str(role).strip().upper()
    )
    if expected is not None and mode != expected:
        raise ArtifactValidationError(
            f"{role}: expected contact_mode={expected}, artifact reports {mode}"
        )
    return mode


def _checksum_manifest_entries(root: Path, manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        raise ArtifactValidationError(f"{manifest_path}: checksum manifest is required")
    manifest: dict[str, str] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(f"{manifest_path}: unreadable: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        digest, separator, relative = raw.partition("  ")
        digest = digest.strip().lower()
        relative = relative.strip().replace("\\", "/")
        if (
            not separator
            or len(digest) != 64
            or any(character not in SHA256_HEX for character in digest)
            or not relative
            or relative in manifest
        ):
            raise ArtifactValidationError(f"{manifest_path}:{line_number}: malformed/duplicate entry")
        candidate = (root / relative).resolve()
        if not _is_within(candidate, root):
            raise ArtifactValidationError(f"{manifest_path}:{line_number}: path escapes run_dir")
        manifest[relative] = digest
    if not manifest:
        raise ArtifactValidationError(f"{manifest_path}: checksum manifest is empty")
    return manifest


def _validate_checksum_manifest(
    root: Path,
    manifest_path: Path,
    required_paths: Sequence[Path],
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    root = root.resolve()
    manifest = _checksum_manifest_entries(root, manifest_path)
    overrides = {str(key): Path(value).resolve() for key, value in (path_overrides or {}).items()}
    for relative, digest in manifest.items():
        candidate = overrides.get(relative, (root / relative).resolve())
        if not _is_within(candidate, root):
            raise ArtifactValidationError(
                f"{manifest_path}: checksum target escapes root: {candidate}"
            )
        if not candidate.is_file():
            raise ArtifactValidationError(
                f"{manifest_path}: listed file is missing: {relative} -> {candidate}"
            )
        actual = sha256_file(candidate).lower()
        if actual != digest:
            raise ArtifactValidationError(
                f"{manifest_path}: checksum mismatch for {relative}"
            )
    for path in required_paths:
        resolved = path.resolve()
        if not _is_within(resolved, root):
            raise ArtifactValidationError(f"required checksum file is outside root: {resolved}")
        relative = resolved.relative_to(root).as_posix()
        if relative not in manifest:
            raise ArtifactValidationError(f"{manifest_path}: required file is not covered: {relative}")
    return manifest


def _validate_checksums(run_dir: Path, required_paths: Sequence[Path]) -> dict[str, str]:
    return _validate_checksum_manifest(
        run_dir,
        run_dir / "checksums.sha256",
        required_paths,
    )


def _resolve_batch_root(artifact_root: Path, run_dir: Path) -> Path:
    candidates = [
        parent.resolve()
        for parent in artifact_root.parents
        if (parent / "batch_request.json").is_file()
    ]
    if len(candidates) != 1:
        raise ArtifactValidationError(
            f"{artifact_root}: expected exactly one owning batch_root with "
            f"batch_request.json, found {len(candidates)}"
        )
    batch_root = candidates[0]
    if artifact_root.parent.parent.resolve() != batch_root:
        raise ArtifactValidationError(
            f"{artifact_root}: version artifact is not at batch_root/version/artifact depth"
        )
    if not _is_within(artifact_root, batch_root) or not _is_within(run_dir, batch_root):
        raise ArtifactValidationError("version artifact/run_dir escapes its owning batch_root")
    return batch_root


def _require_evidence_file(
    evidence_files: Mapping[str, Any],
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    matches: list[tuple[str, Any]] = []
    for name, value in evidence_files.items():
        try:
            if Path(str(name)).resolve() == resolved:
                matches.append((str(name), value))
        except (OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ArtifactValidationError(
            f"{label}: expected exactly one evidence_files row for {resolved}, found {len(matches)}"
        )
    row = _mapping(matches[0][1], f"{label}.evidence_files[{matches[0][0]!r}]")
    if not resolved.is_file():
        raise ArtifactValidationError(f"{label}: evidence file is missing: {resolved}")
    expected_sha = _sha256(row.get("sha256"), f"{label}.{resolved.name}.sha256")
    expected_size = _integer(row.get("size_bytes"), f"{label}.{resolved.name}.size_bytes")
    if expected_size < 0:
        raise ArtifactValidationError(f"{label}: negative size for {resolved}")
    if resolved.stat().st_size != expected_size or sha256_file(resolved).lower() != expected_sha:
        raise ArtifactValidationError(f"{label}: evidence hash/size mismatch for {resolved}")
    return {"path": str(resolved), "sha256": expected_sha, "size_bytes": expected_size}


def _is_formal_worker_result(result: Mapping[str, Any]) -> bool:
    """Recognize the production worker identity without accepting half-bindings."""

    owner = str(result.get("artifact_owner", "") or "")
    execution_path = str(result.get("execution_path", "") or "")
    if not owner and not execution_path:
        return False
    if owner != FORMAL_WORKER_ARTIFACT_OWNER:
        raise ArtifactValidationError(
            "result.artifact_owner is present but is not sim_worker_process"
        )
    if execution_path != FORMAL_WORKER_EXECUTION_PATH:
        raise ArtifactValidationError(
            "result.execution_path is present but is not sim_worker_process_ipc"
        )
    return True


def _require_same_path(
    payload: Mapping[str, Any],
    key: str,
    expected: Path,
    *,
    label: str,
) -> None:
    text = str(payload.get(key, "") or "")
    if not text or Path(text).resolve() != Path(expected).resolve():
        raise ArtifactValidationError(f"{label}.{key} path mismatch")


def _load_formal_worker_batch_closure(
    *,
    named_paths: Mapping[str, Path],
    batch_root: Path,
    artifact_root: Path,
    run_dir: Path,
    result_path: Path,
    result: Mapping[str, Any],
    batch_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the worker request, launch, stop/start boundary, and terminal ACK."""

    request_path = named_paths["worker_artifact_request"]
    startup_path = named_paths["worker_startup_binding"]
    stop_path = named_paths["worker_preplay_stop_ack"]
    start_path = named_paths["worker_playback_start_ack"]
    complete_path = named_paths["worker_artifact_complete_ack"]
    request = _mapping(_strict_json(request_path), str(request_path))
    startup = _mapping(_strict_json(startup_path), str(startup_path))
    stop_ack = _mapping(_strict_json(stop_path), str(stop_path))
    start_ack = _mapping(_strict_json(start_path), str(start_path))
    complete_ack = _mapping(_strict_json(complete_path), str(complete_path))
    request_sha = sha256_file(request_path).lower()

    if str(request.get("schema_version", "")) != "fsm50.worker_recording_gate_request.v1":
        raise ArtifactValidationError("worker_artifact_request schema is invalid")
    if request.get("enabled") is not True:
        raise ArtifactValidationError("worker_artifact_request.enabled is not true")
    if str(request.get("artifact_owner", "")) != FORMAL_WORKER_ARTIFACT_OWNER:
        raise ArtifactValidationError("worker_artifact_request artifact_owner mismatch")
    if _exact_integer(
        request.get("expected_root_state_write_count"),
        "worker_artifact_request.expected_root_state_write_count",
    ) != 0:
        raise ArtifactValidationError("worker artifact request permits root-state writes")

    if str(batch_request.get("schema_version", "")) != "fsm50.formal_worker_recording_batch.v1":
        raise ArtifactValidationError("formal worker batch_request schema is invalid")
    _require_same_path(
        batch_request,
        "artifact_request_path",
        request_path,
        label="batch_request",
    )
    if _sha256(
        batch_request.get("artifact_request_sha256"),
        "batch_request.artifact_request_sha256",
    ) != request_sha:
        raise ArtifactValidationError("batch_request does not bind the actual worker request bytes")

    source_version = str(result.get("source_version", "") or "")
    request_id = str(request.get("request_id", "") or "")
    plan_id = str(request.get("plan_id", "") or "")
    plan_sha = _sha256(request.get("plan_sha256"), "worker request plan_sha256")
    expected_plan_sha = _sha256(
        request.get("expected_plan_sha256"),
        "worker request expected_plan_sha256",
    )
    if expected_plan_sha != plan_sha:
        raise ArtifactValidationError("worker request expected/actual plan SHA mismatch")
    accepted_sha = _sha256(
        request.get(
            "expected_accepted_steps_sha256",
            request.get("accepted_steps_sha256"),
        ),
        "worker request accepted_steps_sha256",
    )
    trial_id = _exact_integer(request.get("trial_id"), "worker request trial_id")
    plan_event_count = _exact_integer(
        request.get("plan_event_count"), "worker request plan_event_count"
    )
    plan_segment_count = _exact_integer(
        request.get("plan_segment_count"), "worker request plan_segment_count"
    )
    contact_mode = str(request.get("contact_mode", "") or "")
    diagnostic_role = str(
        request.get("environment_equivalence_role", "") or ""
    )
    if diagnostic_role not in {"", "A1", "A2", "B"}:
        raise ArtifactValidationError(
            "worker request environment_equivalence_role is invalid"
        )
    expected_diagnostic_contact_mode = (
        TRAJECTORY_DIAGNOSTIC_ROLE_CONTACT_MODES.get(diagnostic_role)
        if diagnostic_role
        else "instrumented"
    )
    if not request_id or not plan_id or trial_id < 1:
        raise ArtifactValidationError("worker request identity is incomplete")
    if contact_mode not in {"formal", "instrumented"}:
        raise ArtifactValidationError("worker request contact_mode is invalid")
    if contact_mode != expected_diagnostic_contact_mode:
        raise ArtifactValidationError(
            "worker request role/contact_mode contract is invalid"
        )
    _require_same_path(
        request, "artifact_root", artifact_root, label="worker_artifact_request"
    )
    request_expected = {
        "source_version": source_version,
        "plan_sha256": _sha256(result.get("plan_sha256"), "result.plan_sha256"),
        "accepted_steps_sha256": _sha256(
            result.get("accepted_steps_sha256"), "result.accepted_steps_sha256"
        ),
        "trial_id": _exact_integer(result.get("trial_id"), "result.trial_id"),
        "contact_mode": str(result.get("contact_mode", "") or ""),
    }
    actual_request_identity = {
        "source_version": str(request.get("source_version", "") or ""),
        "plan_sha256": plan_sha,
        "accepted_steps_sha256": accepted_sha,
        "trial_id": trial_id,
        "contact_mode": contact_mode,
    }
    if not _type_strict_json_equal(actual_request_identity, request_expected):
        raise ArtifactValidationError("worker request/result source, plan, trial, or contact identity mismatch")
    if (
        plan_event_count
        != _exact_integer(result.get("plan_event_count"), "result.plan_event_count")
        or plan_segment_count
        != _exact_integer(result.get("plan_segment_count"), "result.plan_segment_count")
    ):
        raise ArtifactValidationError("worker request/result plan counts mismatch")
    batch_identity = {
        "trial_id": _exact_integer(batch_request.get("trial_id"), "batch_request.trial_id"),
        "plan_id": str(batch_request.get("plan_id", "") or ""),
        "plan_sha256": _sha256(
            batch_request.get("plan_sha256"), "batch_request.plan_sha256"
        ),
    }
    if not _type_strict_json_equal(batch_identity, {
        "trial_id": trial_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha,
    }):
        raise ArtifactValidationError("batch_request worker trial/plan identity mismatch")
    batch_args = _mapping(batch_request.get("args"), "batch_request.args")
    expected_qualification_scope = (
        TRAJECTORY_COMPARISON_QUALIFICATION_SCOPE
        if diagnostic_role
        else "GATE1_PHYSICAL_QUALIFICATION"
    )
    if not _type_strict_json_equal(
        {
            "environment_equivalence_role": batch_request.get(
                "environment_equivalence_role"
            ),
            "qualification_scope": batch_request.get("qualification_scope"),
            "args_environment_equivalence_role": batch_args.get(
                "environment_equivalence_role"
            ),
            "args_contact_mode": batch_args.get("contact_mode"),
        },
        {
            "environment_equivalence_role": diagnostic_role,
            "qualification_scope": expected_qualification_scope,
            "args_environment_equivalence_role": diagnostic_role,
            "args_contact_mode": contact_mode,
        },
    ):
        raise ArtifactValidationError(
            "batch_request diagnostic role/scope/contact_mode differs from worker request"
        )
    if str(batch_args.get("contact_mode", "") or "") != contact_mode:
        raise ArtifactValidationError("batch_request contact_mode differs from worker request")
    if _mapping(
        batch_request.get("prelaunch_environment_validation"),
        "batch_request.prelaunch_environment_validation",
    ).get("ok") is not True:
        raise ArtifactValidationError("formal worker prelaunch environment validation is not true")
    if str(result.get("request_id", "") or "") != request_id:
        raise ArtifactValidationError("worker result request_id mismatch")
    if str(result.get("plan_id", "") or "") != plan_id:
        raise ArtifactValidationError("worker result plan_id mismatch")
    if _sha256(
        result.get("artifact_request_sha256"),
        "result.artifact_request_sha256",
    ) != request_sha:
        raise ArtifactValidationError("worker result does not bind the actual request bytes")
    _require_same_path(result, "artifact_root", artifact_root, label="result")
    _require_same_path(result, "run_dir", run_dir, label="result")
    if str(result.get("artifact_owner", "")) != FORMAL_WORKER_ARTIFACT_OWNER:
        raise ArtifactValidationError("worker result artifact_owner mismatch")
    if str(result.get("execution_path", "")) != FORMAL_WORKER_EXECUTION_PATH:
        raise ArtifactValidationError("worker result execution_path mismatch")
    diagnostic_complete = result.get(
        "environment_equivalence_diagnostic_complete"
    )
    if type(diagnostic_complete) is not bool:
        raise ArtifactValidationError(
            "worker result diagnostic_complete must be an exact JSON boolean"
        )
    result_diagnostic_expected = {
        "environment_equivalence_role": diagnostic_role,
        "environment_equivalence_diagnostic": bool(diagnostic_role),
        "qualification_scope": expected_qualification_scope,
    }
    if not _type_strict_json_equal(
        {
            key: result.get(key) for key in result_diagnostic_expected
        },
        result_diagnostic_expected,
    ):
        raise ArtifactValidationError(
            "worker result diagnostic role/scope identity is invalid"
        )

    worker_pid = _exact_integer(result.get("worker_pid"), "result.worker_pid")
    worker_session_id = str(result.get("worker_session_id", "") or "")
    adapter_id = str(result.get("adapter_runtime_instance_id", "") or "")
    if worker_pid <= 0 or not worker_session_id or not adapter_id:
        raise ArtifactValidationError("worker result process/session/adapter identity is incomplete")
    if _exact_integer(result.get("root_state_write_count"), "result.root_state_write_count") != 0:
        raise ArtifactValidationError("worker result reports a root-state write")
    respawn = _mapping(result.get("respawn"), "result.respawn")
    if (
        _exact_integer(respawn.get("root_state_write_count"), "result.respawn.root_state_write_count") != 0
        or str(respawn.get("adapter_runtime_instance_id", "") or "") != adapter_id
        or respawn.get("root_pose_written") is not False
    ):
        raise ArtifactValidationError("worker result respawn/root-write evidence is invalid")

    if str(startup.get("artifact_request_id", "") or "") != request_id:
        raise ArtifactValidationError("worker startup binding request_id mismatch")
    if _sha256(
        startup.get("artifact_request_sha256"),
        "worker startup artifact_request_sha256",
    ) != request_sha:
        raise ArtifactValidationError("worker startup binding request SHA mismatch")
    if _exact_integer(startup.get("worker_pid"), "worker startup PID") != worker_pid:
        raise ArtifactValidationError("worker startup/result PID mismatch")
    if str(startup.get("worker_session_id", "") or "") != worker_session_id:
        raise ArtifactValidationError("worker startup/result session mismatch")
    if str(startup.get("adapter_runtime_instance_id", "") or "") != adapter_id:
        raise ArtifactValidationError("worker startup/result adapter mismatch")
    status = _mapping(startup.get("status"), "worker_startup_binding.status")
    session = _mapping(
        status.get("worker_artifact_session"),
        "worker_startup_binding.status.worker_artifact_session",
    )
    preflight = _mapping(
        status.get("worker_artifact_preflight"),
        "worker_startup_binding.status.worker_artifact_preflight",
    )
    if status.get("ready") is not True or status.get("artifact_preflight_ready") is not True:
        raise ArtifactValidationError("worker startup was not artifact-preflight ready")
    status_expected = {
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
    }
    if not _type_strict_json_equal(
        {key: status.get(key) for key in status_expected}, status_expected
    ):
        raise ArtifactValidationError("worker startup status PID/session mismatch")
    session_expected = {
        "request_id": request_id,
        "source_version": source_version,
        "trial_id": trial_id,
        "contact_mode": contact_mode,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "root_state_write_count": 0,
        "motion_start_ready": True,
    }
    if not _type_strict_json_equal(
        {key: session.get(key) for key in session_expected}, session_expected
    ):
        raise ArtifactValidationError("worker startup session identity/readiness mismatch")
    if not _type_strict_json_equal(
        session.get("environment_equivalence_role"), diagnostic_role
    ) or not _type_strict_json_equal(
        preflight.get("environment_equivalence_role"), diagnostic_role
    ):
        raise ArtifactValidationError(
            "worker startup diagnostic role differs from worker request"
        )
    _require_same_path(session, "artifact_root", artifact_root, label="worker startup session")
    if _sha256(
        preflight.get("artifact_request_sha256"),
        "worker startup preflight request SHA",
    ) != request_sha:
        raise ArtifactValidationError("worker startup preflight request SHA mismatch")
    if _sha256(
        preflight.get("expected_plan_sha256"),
        "worker startup preflight plan SHA",
    ) != plan_sha:
        raise ArtifactValidationError("worker startup preflight plan SHA mismatch")
    if (
        _exact_integer(session.get("readiness_frame_count_required"), "worker readiness required")
        != 10
        or _exact_integer(session.get("readiness_frame_count"), "worker readiness count") < 10
        or _exact_integer(
            session.get("readiness_sample_stride_physics_ticks"),
            "worker readiness stride",
        )
        != 8
    ):
        raise ArtifactValidationError("worker startup 10-frame readiness window is invalid")
    environment = _mapping(
        session.get("environment_equivalence"),
        "worker startup session environment_equivalence",
    )
    if environment.get("ok") is not True or list(environment.get("failed_checks", []) or []):
        raise ArtifactValidationError("worker startup environment equivalence failed")
    if (
        not str(session.get("contact_sensor_type", "") or "")
        or str(session.get("contact_sensor_type", "") or "") == "NoneType"
        or str(session.get("contact_sensor_error", "") or "")
    ):
        raise ArtifactValidationError("worker startup contact sensor evidence is invalid")

    if str(stop_ack.get("type", "")) != "stop_ack":
        raise ArtifactValidationError("worker pre-play stop ACK type is invalid")
    command_id = str(stop_ack.get("command_id", "") or "")
    if not command_id:
        raise ArtifactValidationError("worker pre-play stop command_id is missing")
    if (
        str(stop_ack.get("reason", "") or "") != "playback_start_boundary"
        or stop_ack.get("zero_target_applied") is not True
        or str(stop_ack.get("error", "") or "")
        or _exact_integer(stop_ack.get("worker_pid"), "pre-play stop worker_pid") != worker_pid
        or str(stop_ack.get("worker_session_id", "") or "") != worker_session_id
        or str(stop_ack.get("adapter_runtime_instance_id", "") or "") != adapter_id
        or str(stop_ack.get("artifact_request_id", "") or "") != request_id
        or _exact_integer(stop_ack.get("root_state_write_count"), "pre-play stop root writes") != 0
    ):
        raise ArtifactValidationError("worker pre-play stop ACK is not a zero-target worker ACK")
    stop_applied_wall = _finite(
        stop_ack.get("target_applied_wall_time"),
        "worker pre-play stop target_applied_wall_time",
    )

    start_expected = {
        "type": "operation_ack",
        "operation": "start_playback_plan",
        "request_id": request_id,
        "artifact_request_id": request_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha,
        "event_count": plan_event_count,
        "segment_count": plan_segment_count,
        "accepted": True,
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "root_state_write_count": 0,
        "motion_start_ready": True,
    }
    if not _type_strict_json_equal(
        {key: start_ack.get(key) for key in start_expected}, start_expected
    ):
        raise ArtifactValidationError("worker playback-start ACK identity/readiness mismatch")
    if str(start_ack.get("error", "") or "") or str(
        start_ack.get("rejection_reason", "") or ""
    ):
        raise ArtifactValidationError("worker playback-start ACK records rejection/error")
    start_accepted_wall = _finite(
        start_ack.get("accepted_wall_time"),
        "worker playback-start accepted_wall_time",
    )
    if stop_applied_wall > start_accepted_wall:
        raise ArtifactValidationError("worker pre-play stop ACK occurred after playback-start ACK")
    if command_id == request_id:
        raise ArtifactValidationError("worker stop command_id aliases playback request_id")

    complete_expected = {
        "type": "operation_ack",
        "operation": "recording_artifact",
        "phase": "ARTIFACT_COMPLETE",
        "artifact_owner": FORMAL_WORKER_ARTIFACT_OWNER,
        "accepted": True,
        "artifact_complete": True,
        "request_id": request_id,
        "artifact_request_id": request_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha,
        "artifact_request_sha256": request_sha,
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "source_version": source_version,
        "trial_id": trial_id,
        "contact_mode": contact_mode,
        "accepted_steps_sha256": accepted_sha,
        "adapter_runtime_instance_id": adapter_id,
        "root_state_write_count": 0,
        "environment_equivalence_role": diagnostic_role,
        "environment_equivalence_diagnostic_complete": diagnostic_complete,
    }
    if not _type_strict_json_equal(
        {key: complete_ack.get(key) for key in complete_expected},
        complete_expected,
    ):
        raise ArtifactValidationError("worker artifact-complete ACK identity mismatch")
    if str(complete_ack.get("error", "") or ""):
        raise ArtifactValidationError("worker artifact-complete ACK records an error")
    declared_paths = {
        "result_path": (result_path, "result_sha256"),
        "artifact_pointer_path": (artifact_root / "artifact_pointer.json", "artifact_pointer_sha256"),
        "checksums_path": (run_dir / "checksums.sha256", "checksums_sha256"),
        "telemetry_finalization_path": (
            run_dir / "telemetry_finalization.json",
            "telemetry_finalization_sha256",
        ),
        "visual_manifest_path": (
            run_dir / "visual_recording_manifest.json",
            "visual_manifest_sha256",
        ),
    }
    for path_key, (expected_path, sha_key) in declared_paths.items():
        _require_same_path(complete_ack, path_key, expected_path, label="worker complete ACK")
        if not expected_path.is_file():
            raise ArtifactValidationError(
                f"required artifact is missing: {expected_path}"
            )
        if _sha256(complete_ack.get(sha_key), f"worker complete ACK {sha_key}") != sha256_file(expected_path).lower():
            raise ArtifactValidationError(f"worker artifact-complete ACK {sha_key} mismatch")
    _require_same_path(
        complete_ack, "artifact_root", artifact_root, label="worker complete ACK"
    )
    _require_same_path(complete_ack, "run_dir", run_dir, label="worker complete ACK")
    for key in (
        "artifact_valid",
        "classification",
        "scheduler_complete",
        "physical_success",
        "strict_full_success",
    ):
        if not _type_strict_json_equal(complete_ack.get(key), result.get(key)):
            raise ArtifactValidationError(f"worker complete ACK/result {key} mismatch")

    copied_request = run_dir / "input" / "worker_artifact_request.json"
    if not copied_request.is_file() or sha256_file(copied_request).lower() != request_sha:
        raise ArtifactValidationError("run-local worker artifact request differs from sealed request")
    required_evidence = list(result.get("required_evidence_paths", []) or [])
    if not required_evidence:
        raise ArtifactValidationError("worker result.required_evidence_paths is empty")
    _validate_checksum_manifest(
        run_dir,
        run_dir / "checksums.sha256",
        [
            copied_request,
            *(
                Path(str(raw)).resolve()
                for raw in required_evidence
                if Path(str(raw)).resolve() != run_dir / "checksums.sha256"
            ),
        ],
    )
    return {
        "artifact_owner": FORMAL_WORKER_ARTIFACT_OWNER,
        "execution_path": FORMAL_WORKER_EXECUTION_PATH,
        "request_id": request_id,
        "request_sha256": request_sha,
        "plan_id": plan_id,
        "plan_sha256": plan_sha,
        "trial_id": trial_id,
        "contact_mode": contact_mode,
        "environment_equivalence_role": diagnostic_role,
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "preplay_stop_command_id": command_id,
        "preplay_stop_applied_wall_time": stop_applied_wall,
        "playback_start_accepted_wall_time": start_accepted_wall,
    }


def _validate_formal_worker_shutdown_binding(
    shutdown: Mapping[str, Any],
    worker_closure: Mapping[str, Any],
) -> None:
    """Require exact pre-close ACKs and normal exits from both owned processes."""

    if (
        str(shutdown.get("status", "")) != "FAST_EXIT_VERIFIED"
        or str(shutdown.get("handshake_state", ""))
        != "FAST_WORKER_PROCESS_RETURNED"
        or str(shutdown.get("shutdown_mode", "")) != "fast"
        or _exact_integer(
            shutdown.get("child_returncode"), "shutdown child_returncode"
        )
        != 0
        or _exact_integer(
            shutdown.get("intended_returncode"), "shutdown intended_returncode"
        )
        != 0
        or shutdown.get("process_returned_normally") is not True
        or _exact_integer(
            shutdown.get("worker_returncode"), "shutdown worker_returncode"
        )
        != 0
        or shutdown.get("worker_process_returned_normally") is not True
        or shutdown.get("worker_shutdown_accepted") is not True
        or shutdown.get("worker_close_requested") is not True
        or type(shutdown.get("worker_close_returned")) is not bool
    ):
        raise ArtifactValidationError("formal worker shutdown is not a verified double-process fast exit")
    worker_pid = int(worker_closure["worker_pid"])
    worker_session_id = str(worker_closure["worker_session_id"])
    adapter_id = str(worker_closure["adapter_runtime_instance_id"])
    artifact_request_id = str(worker_closure["request_id"])
    artifact_request_sha256 = str(worker_closure["request_sha256"])
    if (
        _exact_integer(
            shutdown.get("formal_worker_pid"), "shutdown formal_worker_pid"
        )
        != worker_pid
        or str(shutdown.get("formal_worker_session_id", "") or "")
        != worker_session_id
        or str(shutdown.get("adapter_runtime_instance_id", "") or "")
        != adapter_id
        or str(shutdown.get("artifact_request_id", "") or "")
        != artifact_request_id
        or _sha256(
            shutdown.get("artifact_request_sha256"),
            "shutdown artifact_request_sha256",
        )
        != artifact_request_sha256
        or shutdown.get("worker_forced_termination") is not False
    ):
        raise ArtifactValidationError("shutdown formal-worker identity/termination binding mismatch")
    shutdown_request_id = str(shutdown.get("worker_shutdown_request_id", "") or "")
    if not shutdown_request_id:
        raise ArtifactValidationError("shutdown worker request_id is missing")
    close_kwargs = {"wait_for_replicator": False, "skip_cleanup": True}
    runtime_version = str(shutdown.get("runtime_version", "") or "")
    if not runtime_version.startswith("5.1."):
        raise ArtifactValidationError("formal worker shutdown runtime is not Isaac 5.1")
    common_expected = {
        "request_id": shutdown_request_id,
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": artifact_request_id,
        "root_state_write_count": 0,
        "close_kwargs": close_kwargs,
        "runtime_version": runtime_version,
        "accepted": True,
        "error": "",
    }
    shutdown_ack = _mapping(
        shutdown.get("worker_shutdown_ack"), "shutdown.worker_shutdown_ack"
    )
    if (
        shutdown_ack.get("type") != "operation_ack"
        or shutdown_ack.get("operation") != "shutdown"
        or shutdown_ack.get("mode") != "fast"
        or not _type_strict_json_equal(
            {key: shutdown_ack.get(key) for key in common_expected},
            common_expected,
        )
    ):
        raise ArtifactValidationError("raw worker shutdown ACK identity/contract mismatch")
    close_requested_ack = _mapping(
        shutdown.get("worker_close_requested_ack"),
        "shutdown.worker_close_requested_ack",
    )
    if (
        close_requested_ack.get("type") != "close_requested"
        or close_requested_ack.get("mode") != "fast"
        or not _type_strict_json_equal(
            {key: close_requested_ack.get(key) for key in common_expected},
            common_expected,
        )
    ):
        raise ArtifactValidationError(
            "raw worker_close_requested_ack identity/contract mismatch"
        )
    close_returned_observed = shutdown.get("worker_close_returned")
    raw_close_returned = shutdown.get("worker_close_returned_ack", {})
    if close_returned_observed:
        close_returned_ack = _mapping(
            raw_close_returned,
            "shutdown.worker_close_returned_ack",
        )
        if (
            close_returned_ack.get("type") != "close_returned"
            or close_returned_ack.get("mode") != "fast"
            or not _type_strict_json_equal(
                {key: close_returned_ack.get(key) for key in common_expected},
                common_expected,
            )
        ):
            raise ArtifactValidationError(
                "raw worker_close_returned_ack identity/contract mismatch"
            )
    elif raw_close_returned not in ({}, None):
        raise ArtifactValidationError(
            "raw worker_close_returned_ack exists without observed return"
        )


def _load_batch_shutdown_closure(
    *,
    batch_root: Path,
    artifact_root: Path,
    run_dir: Path,
    result_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only a normally closed supervised batch with durable evidence."""

    result_claims_worker = _is_formal_worker_result(result)

    if not (batch_root / ".finalized").is_file():
        raise ArtifactValidationError(f"{batch_root}: batch .finalized marker is missing")
    if (batch_root / ".partial").exists() or (batch_root / ".failed").exists():
        raise ArtifactValidationError(f"{batch_root}: batch partial/failed marker is present")

    named_paths = {
        "batch_request": batch_root / "batch_request.json",
        "shutdown_outcome": batch_root / "shutdown_outcome.json",
        "batch_finalization": batch_root / "batch_finalization.json",
        "batch_results": batch_root / "batch_results.json",
        "preclose_complete": batch_root / "preclose_complete.json",
        "batch_results_preclose": batch_root / "batch_results.preclose.json",
        "batch_finalization_preclose": batch_root / "batch_finalization.preclose.json",
        "checksums_preclose": batch_root / "checksums.preclose.sha256",
        "batch_checksums": batch_root / "checksums.sha256",
        "batch_source_integrity": batch_root / "source_integrity.json",
        "run_checksums": run_dir / "checksums.sha256",
    }
    missing = [str(path) for path in named_paths.values() if not path.is_file()]
    if missing:
        raise ArtifactValidationError(
            [f"required shutdown-closure artifact is missing: {path}" for path in missing]
        )

    request = _mapping(_strict_json(named_paths["batch_request"]), "batch_request.json")
    formal_schema = str(request.get("schema_version", "")) == (
        "fsm50.formal_worker_recording_batch.v1"
    )
    any_worker_control = any(
        (batch_root / name).exists() for name in WORKER_BATCH_CONTROL_FILENAMES
    )
    versions = request.get("versions")
    source_version = str(result.get("source_version", "") or "")
    worker_owned = bool(result_claims_worker or formal_schema or any_worker_control)
    if worker_owned:
        named_paths.update(
            {
                Path(name).stem: batch_root / name
                for name in WORKER_BATCH_CONTROL_FILENAMES
            }
        )
        missing_controls = [
            str(named_paths[Path(name).stem])
            for name in WORKER_BATCH_CONTROL_FILENAMES
            if not named_paths[Path(name).stem].is_file()
        ]
        if missing_controls:
            raise ArtifactValidationError(
                [
                    f"required formal-worker shutdown-closure artifact is missing: {path}"
                    for path in missing_controls
                ]
            )
        if not result_claims_worker:
            raise ArtifactValidationError(
                "formal-worker batch/control evidence cannot be downgraded to a legacy result"
            )
        if versions != [source_version]:
            raise ArtifactValidationError(
                "formal-worker batch_request.versions must equal [source_version] exactly"
            )
    if not isinstance(versions, list) or versions.count(source_version) != 1:
        raise ArtifactValidationError(
            "batch_request.versions does not contain the current source version exactly once"
        )

    worker_closure: dict[str, Any] | None = None
    if worker_owned:
        worker_closure = _load_formal_worker_batch_closure(
            named_paths=named_paths,
            batch_root=batch_root,
            artifact_root=artifact_root,
            run_dir=run_dir,
            result_path=result_path,
            result=result,
            batch_request=request,
        )

    shutdown = _mapping(
        _strict_json(named_paths["shutdown_outcome"]), "shutdown_outcome.json"
    )
    try:
        validated_shutdown = validate_shutdown_outcome(
            shutdown,
            allow_legacy_normal_exit=not worker_owned,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(str(exc)) from exc
    if worker_owned:
        assert worker_closure is not None
        _validate_formal_worker_shutdown_binding(shutdown, worker_closure)
        if str(validated_shutdown.get("contract_kind", "")) != (
            "isaac_5_1_worker_fast_exit"
        ):
            raise ArtifactValidationError("formal worker shutdown contract kind is invalid")

    finalization = _mapping(
        _strict_json(named_paths["batch_finalization"]), "batch_finalization.json"
    )
    finalization_failures: list[str] = []
    if Path(str(finalization.get("artifact_root", ""))).resolve() != batch_root:
        finalization_failures.append("artifact_root mismatch")
    if finalization.get("finalized") is not True or finalization.get("failed") is not False:
        finalization_failures.append("finalized/failed flags are not true/false")
    if str(finalization.get("phase", "")) != "SHUTDOWN_COMPLETE":
        finalization_failures.append("phase is not SHUTDOWN_COMPLETE")
    if str(finalization.get("close_error", "") or ""):
        finalization_failures.append("close_error is not empty")
    if str(finalization.get("batch_error", "") or ""):
        finalization_failures.append("batch_error is not empty")
    if list(finalization.get("finalization_errors", []) or []):
        finalization_failures.append("finalization_errors is not empty")
    if not _type_strict_json_equal(
        _mapping(
            finalization.get("shutdown_outcome"),
            "batch_finalization.shutdown_outcome",
        ),
        shutdown,
    ):
        finalization_failures.append("embedded shutdown_outcome differs from file")
    if finalization_failures:
        raise ArtifactValidationError(
            [f"{named_paths['batch_finalization']}: {failure}" for failure in finalization_failures]
        )

    batch_results_raw = _strict_json(named_paths["batch_results"])
    if not isinstance(batch_results_raw, list) or not batch_results_raw:
        raise ArtifactValidationError("batch_results.json must be a non-empty list")
    batch_results = [
        _mapping(item, f"batch_results[{index}]")
        for index, item in enumerate(batch_results_raw)
    ]
    if worker_owned and not _type_strict_json_equal(
        batch_results_raw, [dict(result)]
    ):
        raise ArtifactValidationError(
            "formal-worker batch_results must equal [durable worker result] exactly"
        )
    matching = [
        item
        for item in batch_results
        if str(item.get("source_version", "")) == source_version
        and Path(str(item.get("run_dir", ""))).resolve() == run_dir
        and Path(str(item.get("artifact_root", ""))).resolve() == artifact_root
    ]
    if len(matching) != 1 or not _type_strict_json_equal(
        matching[0], dict(result)
    ):
        raise ArtifactValidationError(
            "batch_results does not contain exactly the current durable result/version/run/artifact"
        )

    preclose_results = _strict_json(named_paths["batch_results_preclose"])
    if not _type_strict_json_equal(preclose_results, batch_results_raw):
        raise ArtifactValidationError(
            "batch_results.preclose.json differs from normal-exit live batch_results.json"
        )
    preclose_finalization = _mapping(
        _strict_json(named_paths["batch_finalization_preclose"]),
        "batch_finalization.preclose.json",
    )
    expected_preclose_close_error = (
        "PENDING_FORMAL_WORKER_CLOSE"
        if worker_owned
        else "PENDING_SIMULATION_CLOSE"
    )
    if (
        Path(str(preclose_finalization.get("artifact_root", ""))).resolve() != batch_root
        or preclose_finalization.get("finalized") is not True
        or preclose_finalization.get("failed") is not False
        or str(preclose_finalization.get("phase", "")) != "PRECLOSE_FINALIZED"
        or str(preclose_finalization.get("close_error", ""))
        != expected_preclose_close_error
        or str(preclose_finalization.get("batch_error", "") or "")
        or list(preclose_finalization.get("finalization_errors", []) or [])
    ):
        raise ArtifactValidationError(
            "batch_finalization.preclose.json is not a valid successful preclose snapshot"
        )

    marker = _mapping(
        _strict_json(named_paths["preclose_complete"]), "preclose_complete.json"
    )
    if str(marker.get("schema_version", "")) != "fsm50.preclose_complete.v1":
        raise ArtifactValidationError("preclose_complete schema is invalid")
    if Path(str(marker.get("batch_root", ""))).resolve() != batch_root:
        raise ArtifactValidationError("preclose_complete.batch_root mismatch")
    evidence = _mapping(marker.get("evidence"), "preclose_complete.evidence")
    if (
        "manifest_error" in evidence
        or list(evidence.get("immutable_preclose_errors", []) or [])
        or str(evidence.get("batch_marker_error", "") or "")
    ):
        raise ArtifactValidationError("preclose_complete records invalid/incomplete evidence")
    if _integer(evidence.get("physics_result_count"), "preclose.physics_result_count") != len(
        batch_results
    ):
        raise ArtifactValidationError("preclose physics_result_count differs from batch_results")
    if not _type_strict_json_equal(
        _mapping(
            evidence.get("batch_finalization"),
            "preclose.batch_finalization",
        ),
        preclose_finalization,
    ):
        raise ArtifactValidationError(
            "preclose evidence batch_finalization differs from immutable snapshot"
        )
    batch_source_integrity = _mapping(
        _strict_json(named_paths["batch_source_integrity"]),
        "batch source_integrity.json",
    )
    if batch_source_integrity.get("equal") is not True:
        raise ArtifactValidationError("batch source_integrity.equal is not true")
    if not _type_strict_json_equal(
        _mapping(evidence.get("source_integrity"), "preclose.source_integrity"),
        batch_source_integrity,
    ):
        raise ArtifactValidationError("preclose source_integrity differs from durable file")

    evidence_files = _mapping(evidence.get("evidence_files"), "preclose.evidence_files")
    stable_preclose_paths = (
        named_paths["batch_results_preclose"],
        named_paths["batch_finalization_preclose"],
        named_paths["checksums_preclose"],
        named_paths["batch_source_integrity"],
        named_paths["batch_request"],
        result_path,
        named_paths["run_checksums"],
    )
    if worker_owned:
        stable_preclose_paths = stable_preclose_paths + tuple(
            named_paths[Path(name).stem] for name in WORKER_BATCH_CONTROL_FILENAMES
        )
    evidence_rows = {
        str(path.resolve()): _require_evidence_file(
            evidence_files, path, label="preclose_complete"
        )
        for path in stable_preclose_paths
    }

    preclose_required = [
        named_paths["batch_results"],
        named_paths["batch_finalization"],
        named_paths["batch_request"],
        named_paths["batch_source_integrity"],
        result_path,
    ]
    if worker_owned:
        preclose_required.extend(
            named_paths[Path(name).stem] for name in WORKER_BATCH_CONTROL_FILENAMES
        )
    preclose_manifest = _validate_checksum_manifest(
        batch_root,
        named_paths["checksums_preclose"],
        preclose_required,
        path_overrides={
            "batch_results.json": named_paths["batch_results_preclose"],
            "batch_finalization.json": named_paths["batch_finalization_preclose"],
        },
    )
    live_required = [
        named_paths["shutdown_outcome"],
        named_paths["batch_finalization"],
        named_paths["batch_results"],
        named_paths["preclose_complete"],
        named_paths["batch_results_preclose"],
        named_paths["batch_finalization_preclose"],
        named_paths["checksums_preclose"],
        named_paths["batch_request"],
        named_paths["batch_source_integrity"],
        result_path,
    ]
    if worker_owned:
        live_required.extend(
            named_paths[Path(name).stem] for name in WORKER_BATCH_CONTROL_FILENAMES
        )
    live_manifest = _validate_checksum_manifest(
        batch_root,
        named_paths["batch_checksums"],
        live_required,
    )
    closure_files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path).lower(),
        }
        for name, path in named_paths.items()
    }
    closure_files["result"] = {
        "path": str(result_path.resolve()),
        "sha256": sha256_file(result_path).lower(),
    }
    closure_digest = hashlib.sha256(
        json.dumps(
            closure_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "batch_root": str(batch_root),
        "status": str(validated_shutdown["status"]),
        "shutdown_mode": str(validated_shutdown.get("shutdown_mode", "") or ""),
        "shutdown_contract_kind": str(
            validated_shutdown.get("contract_kind", "") or ""
        ),
        "phase": "SHUTDOWN_COMPLETE",
        "files": closure_files,
        "closure_sha256": closure_digest,
        "live_checksum_entry_count": len(live_manifest),
        "preclose_checksum_entry_count": len(preclose_manifest),
        "preclose_evidence_files": evidence_rows,
        "formal_worker_closure": worker_closure,
    }


def _validate_result(
    result: dict[str, Any],
    *,
    result_path: Path,
    run_dir: Path,
    artifact_root: Path,
) -> None:
    failures: list[str] = []
    if str(result.get("schema_version", "")) != RESULT_SCHEMA:
        failures.append(f"result schema must be {RESULT_SCHEMA}")
    if Path(str(result.get("run_dir", ""))).resolve() != run_dir:
        failures.append("result.run_dir does not identify the supplied run")
    if Path(str(result.get("artifact_root", ""))).resolve() != artifact_root:
        failures.append("result.artifact_root does not identify the containing artifact")
    if result.get("artifact_valid") is not True:
        failures.append("result.artifact_valid is not true")
    lifecycle = _mapping(result.get("lifecycle", {}), f"{result_path}: lifecycle")
    if lifecycle.get("finalized") is not True or lifecycle.get("failed") is not False:
        failures.append("result lifecycle is not finalized/non-failed")
    if result.get("scheduler_complete") is not True:
        failures.append("scheduler_complete is not true")
    if str(result.get("scheduler_stop_reason", "")) != "complete":
        failures.append("scheduler_stop_reason is not complete")
    if result.get("timed_out") is not False:
        failures.append("result timed_out is not false")
    if result.get("simulation_app_stopped") is not False:
        failures.append("result simulation_app_stopped is not false")
    if _mapping(result.get("source_integrity", {}), "source_integrity").get("ok") is not True:
        failures.append("source_integrity.ok is not true")
    if _mapping(result.get("visualization", {}), "visualization").get("ok") is not True:
        failures.append("visualization.ok is not true")
    if str(result.get("requested_profile", "")).lower() != "fast":
        failures.append("requested_profile is not fast")
    if str(result.get("canonical_profile", "")).lower() != "motion_only":
        failures.append("canonical_profile is not motion_only")
    if not str(result.get("source_version", "")):
        failures.append("source_version is missing")
    if not str(result.get("classification", "")):
        failures.append("classification is missing")
    try:
        _sha256(result.get("accepted_steps_sha256"), "accepted_steps_sha256")
        _sha256(result.get("plan_sha256"), "plan_sha256")
        if "expected_preflight_steps_sha256" in result and _sha256(
            result.get("expected_preflight_steps_sha256"),
            "expected_preflight_steps_sha256",
        ) != _sha256(result.get("accepted_steps_sha256"), "accepted_steps_sha256"):
            failures.append("accepted_steps_sha256 differs from preflight hash")
        if _integer(result.get("plan_event_count"), "plan_event_count") <= 0:
            failures.append("plan_event_count must be positive")
        if _integer(result.get("plan_segment_count"), "plan_segment_count") <= 0:
            failures.append("plan_segment_count must be positive")
        if _finite(result.get("plan_final_time_s"), "plan_final_time_s") <= 0.0:
            failures.append("plan_final_time_s must be positive")
    except ArtifactValidationError as exc:
        failures.extend(exc.failures)
    if failures:
        raise ArtifactValidationError([f"{result_path}: {failure}" for failure in failures])


def _validate_visual_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    contact_mode: str,
) -> None:
    """Require the formal combined viewport-video/telemetry manifest."""

    failures: list[str] = []
    if str(payload.get("schema_version", "")) != "fsm50.recording_visual_evidence.v2":
        failures.append(
            "schema is not fsm50.recording_visual_evidence.v2; legacy telemetry-only "
            "visualization is not actual viewport video evidence"
        )
    if str(payload.get("kind", "")) != (
        "actual_active_gui_viewport_video_with_telemetry_visualization"
    ):
        failures.append("kind does not identify active-GUI viewport video")
    if str(payload.get("contact_mode", "")) != contact_mode:
        failures.append("contact_mode differs from the replay role")
    if payload.get("artifact_valid") is not True:
        failures.append("artifact_valid is not true")
    if payload.get("actual_viewport_video") is not True:
        failures.append("actual_viewport_video is not true")
    if payload.get("not_camera_video") is not False:
        failures.append("not_camera_video must be false")
    basis = payload.get("basis", [])
    basis_text = " ".join(str(item).lower() for item in basis) if isinstance(basis, list) else ""
    if "viewport" not in basis_text or "telemetry" not in basis_text:
        failures.append("basis must name both viewport frames and telemetry")
    if failures:
        raise ArtifactValidationError([f"{path}: {failure}" for failure in failures])


def _load_viewport_video_evidence(
    run_dir: Path,
    *,
    contact_mode: str,
    result: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    visual_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the complete direct active-viewport evidence closure."""

    # Recording replay production writes one capture per finalized run.  Do
    # not walk to a batch/ancestor, which could reuse a different run's movie.
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "viewport_video_manifest.json"
    buffer_manifest_path = run_dir / "viewport_buffer_video_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError(
            f"{run_dir}: actual viewport_video_manifest.json is missing; "
            "visual_recording_manifest.json with not_camera_video=true is not video evidence"
        )
    if not buffer_manifest_path.is_file():
        raise ArtifactValidationError(
            f"{buffer_manifest_path}: required direct viewport buffer manifest is missing"
        )
    manifest = _mapping(_strict_json(manifest_path), str(manifest_path))
    buffer_manifest = _mapping(
        _strict_json(buffer_manifest_path), str(buffer_manifest_path)
    )
    failures: list[str] = []

    if str(manifest.get("schema_version", "")) != "fsm50.recording_viewport_video.v1":
        failures.append("unsupported viewport video schema")
    if str(buffer_manifest.get("schema_version", "")) != DIRECT_VIEWPORT_BUFFER_SCHEMA:
        failures.append("unsupported direct viewport buffer schema")
    for flag in ("valid", "artifact_valid", "actual_viewport_video"):
        if manifest.get(flag) is not True:
            failures.append(f"viewport video {flag} is not true")
        if buffer_manifest.get(flag) is not True:
            failures.append(f"viewport buffer video {flag} is not true")
    if manifest.get("capture_requested") is not True:
        failures.append("viewport video capture_requested is not true")
    if manifest.get("diagnostic_only") is not False:
        failures.append("diagnostic-only capture cannot establish equivalence")
    for payload_name, payload in (
        ("viewport manifest", manifest),
        ("viewport buffer manifest", buffer_manifest),
    ):
        if payload.get("not_camera_video") is not False:
            failures.append(f"{payload_name} does not identify actual camera video")
        if str(payload.get("source", "")) != DIRECT_VIEWPORT_SOURCE:
            failures.append(f"{payload_name} source is not the active Isaac GUI render product")
        if str(payload.get("capture_backend", "")) != DIRECT_VIEWPORT_CAPTURE_BACKEND:
            failures.append(f"{payload_name} capture_backend is not the direct LdrColor buffer backend")
        if payload.get("render_product_unchanged") is not True:
            failures.append(f"{payload_name} render_product_unchanged is not true")
        if payload.get("active_render_product_identity_proven") is not True:
            failures.append(
                f"{payload_name} active render-product identity is not proven"
            )
        if payload.get("capture_graph_created") is not False:
            failures.append(f"{payload_name} reports a capture graph")
        if payload.get("render_observer_only") is not True:
            failures.append(f"{payload_name} is not an existing-render observer")
        if payload.get("frame_ledger_complete") is not True:
            failures.append(f"{payload_name} frame ledger is not complete")
        if payload.get("full_decode_all_frames") is not True:
            failures.append(f"{payload_name} did not fully decode every frame")
        if str(payload.get("error", "") or ""):
            failures.append(f"{payload_name} reports an error")
        if str(payload.get("checkpoint_error", "") or ""):
            failures.append(f"{payload_name} reports a checkpoint error")
        try:
            if _integer(
                payload.get("extra_app_update_count"),
                f"{payload_name}.extra_app_update_count",
            ) != 0:
                failures.append(f"{payload_name} invoked extra app.update calls")
            if _integer(
                payload.get("extra_render_count"),
                f"{payload_name}.extra_render_count",
            ) != 0:
                failures.append(f"{payload_name} invoked extra renders")
            if _integer(
                payload.get("maximum_pending_captures"),
                f"{payload_name}.maximum_pending_captures",
            ) != 1:
                failures.append(f"{payload_name} did not enforce one pending capture")
        except ArtifactValidationError as exc:
            failures.extend(exc.failures)

    if str(manifest.get("contact_mode", "")) != contact_mode:
        failures.append("viewport contact_mode differs from the replay role")
    render_product_path = str(manifest.get("render_product_path", "") or "")
    if not render_product_path:
        failures.append("render_product_path is missing")

    frame_count = 0
    fps = 0.0
    viewport_identity = 0
    viewport_identity_check_count = 0
    try:
        frame_count = _integer(manifest.get("frame_count"), "viewport.frame_count")
        if frame_count < 2:
            failures.append("viewport video has fewer than two frames")
        fps = _finite(manifest.get("fps"), "viewport.fps")
        if fps <= 0.0:
            failures.append("viewport fps must be positive")
        viewport_identity = _integer(
            manifest.get("viewport_identity"), "viewport.viewport_identity"
        )
        if viewport_identity <= 0:
            failures.append("viewport_identity must be positive")
        viewport_identity_check_count = _integer(
            manifest.get("viewport_identity_check_count"),
            "viewport.viewport_identity_check_count",
        )
        if viewport_identity_check_count < 2 * frame_count + 1:
            failures.append(
                "viewport identity was not rechecked before/after every frame and at finalize"
            )
    except ArtifactValidationError as exc:
        failures.extend(exc.failures)

    direct_fields = (
        "valid",
        "artifact_valid",
        "actual_viewport_video",
        "not_camera_video",
        "capture_backend",
        "source",
        "render_product_path",
        "viewport_identity",
        "viewport_identity_check_count",
        "render_product_unchanged",
        "active_render_product_identity_proven",
        "capture_graph_created",
        "extra_app_update_count",
        "extra_render_count",
        "render_observer_only",
        "maximum_pending_captures",
        "fps",
        "frame_count",
        "frame_ledger_complete",
        "ledger_path",
        "ledger_sha256",
        "video_path",
        "video_sha256",
        "video_size",
        "first_frame_path",
        "first_frame_sha256",
        "last_frame_path",
        "last_frame_sha256",
        "full_decode",
        "full_decode_all_frames",
        "error",
        "checkpoint_error",
    )
    for field in direct_fields:
        if manifest.get(field) != buffer_manifest.get(field):
            failures.append(
                f"viewport wrapper {field} differs from the direct buffer manifest"
            )

    evidence_specs = {
        "ledger": (
            "ledger_path",
            "ledger_sha256",
            run_dir / "viewport_frame_ledger.jsonl",
            ".jsonl",
        ),
        "first_frame": (
            "first_frame_path",
            "first_frame_sha256",
            run_dir / "viewport_first_frame.png",
            ".png",
        ),
        "last_frame": (
            "last_frame_path",
            "last_frame_sha256",
            run_dir / "viewport_last_frame.png",
            ".png",
        ),
        "video": (
            "video_path",
            "video_sha256",
            run_dir / "actual_viewport_video.mp4",
            ".mp4",
        ),
    }
    evidence_paths: dict[str, Path] = {}
    evidence_hashes: dict[str, str] = {}
    for label, (path_field, sha_field, expected_path, suffix) in evidence_specs.items():
        path_text = str(manifest.get(path_field, "") or "")
        candidate = Path(path_text).resolve() if path_text else Path()
        expected = expected_path.resolve()
        evidence_paths[label] = expected
        if not path_text:
            failures.append(f"viewport {label} path is missing")
            continue
        if candidate != expected:
            failures.append(f"viewport {label} path is not the canonical run-local path")
            continue
        if not _is_within(candidate, run_dir):
            failures.append(f"viewport {label} escapes the run directory")
            continue
        if candidate.suffix.lower() != suffix:
            failures.append(f"viewport {label} has the wrong file extension")
            continue
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            failures.append(f"viewport {label} is missing or empty")
            continue
        try:
            expected_sha = _sha256(manifest.get(sha_field), f"viewport.{sha_field}")
            actual_sha = sha256_file(candidate).lower()
            evidence_hashes[label] = actual_sha
            if actual_sha != expected_sha:
                failures.append(f"viewport {label} SHA-256 mismatch")
        except (OSError, ArtifactValidationError) as exc:
            failures.extend(
                exc.failures
                if isinstance(exc, ArtifactValidationError)
                else [f"viewport {label} is unreadable: {exc}"]
            )

    video_path = evidence_paths["video"]
    if video_path.is_file():
        try:
            header = video_path.read_bytes()[:32]
            if len(header) < 12 or header[4:8] != b"ftyp":
                failures.append("viewport MP4 lacks an ISO base-media ftyp header")
            if _integer(manifest.get("video_size"), "viewport.video_size") != video_path.stat().st_size:
                failures.append("viewport video_size differs from the MP4")
        except (OSError, ArtifactValidationError) as exc:
            failures.extend(
                exc.failures
                if isinstance(exc, ArtifactValidationError)
                else [f"viewport MP4 is unreadable: {exc}"]
            )
    for label in ("first_frame", "last_frame"):
        path = evidence_paths[label]
        if path.is_file():
            try:
                if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                    failures.append(f"viewport {label} lacks a PNG signature")
            except OSError as exc:
                failures.append(f"viewport {label} is unreadable: {exc}")

    ledger_rows: list[dict[str, Any]] = []
    ledger_path = evidence_paths["ledger"]
    if ledger_path.is_file():
        try:
            ledger_rows = _jsonl(ledger_path)
        except ArtifactValidationError as exc:
            failures.extend(exc.failures)
    if len(ledger_rows) != frame_count:
        failures.append("viewport frame ledger count differs from frame_count")
    ledger_dimensions: set[tuple[int, int]] = set()
    for index, row in enumerate(ledger_rows):
        label = f"viewport.ledger[{index}]"
        try:
            if _integer(row.get("render_sequence"), f"{label}.render_sequence") != index:
                failures.append(f"{label}.render_sequence is not contiguous")
            if _integer(
                row.get("encoded_frame_index"), f"{label}.encoded_frame_index"
            ) != index:
                failures.append(f"{label}.encoded_frame_index is not contiguous")
            if _integer(row.get("sim_step"), f"{label}.sim_step") < 0:
                failures.append(f"{label}.sim_step is negative")
            _finite(row.get("sim_time_s"), f"{label}.sim_time_s")
            width = _integer(row.get("width"), f"{label}.width")
            height = _integer(row.get("height"), f"{label}.height")
            if width < 1 or height < 1:
                failures.append(f"{label} dimensions are invalid")
            else:
                ledger_dimensions.add((width, height))
            if _integer(
                row.get("rgba_buffer_size"), f"{label}.rgba_buffer_size"
            ) != width * height * 4:
                failures.append(f"{label}.rgba_buffer_size is not RGBA8-sized")
            if "RGBA8_UNORM" not in str(row.get("byte_format", "")):
                failures.append(f"{label}.byte_format is not RGBA8_UNORM")
            if str(row.get("capture_backend", "")) != DIRECT_VIEWPORT_CAPTURE_BACKEND:
                failures.append(f"{label}.capture_backend mismatch")
            if str(row.get("render_product_path", "")) != render_product_path:
                failures.append(f"{label}.render_product_path mismatch")
            if _integer(
                row.get("viewport_identity"), f"{label}.viewport_identity"
            ) != viewport_identity:
                failures.append(f"{label}.viewport_identity mismatch")
        except ArtifactValidationError as exc:
            failures.extend(exc.failures)

    try:
        full_decode = _mapping(manifest.get("full_decode"), "viewport.full_decode")
        if full_decode.get("valid") is not True:
            failures.append("viewport full_decode.valid is not true")
        decoded_count = _integer(
            full_decode.get("decoded_frame_count"),
            "viewport.full_decode.decoded_frame_count",
        )
        decoded_width = _integer(
            full_decode.get("decoded_width"), "viewport.full_decode.decoded_width"
        )
        decoded_height = _integer(
            full_decode.get("decoded_height"), "viewport.full_decode.decoded_height"
        )
        decoded_channels = _integer(
            full_decode.get("decoded_channels"),
            "viewport.full_decode.decoded_channels",
        )
        if decoded_count != frame_count:
            failures.append("full decoded frame count differs from the ledger")
        if decoded_width < 1 or decoded_height < 1 or decoded_channels not in (3, 4):
            failures.append("full decoded frame dimensions/channels are invalid")
        if ledger_dimensions != {(decoded_width, decoded_height)}:
            failures.append("full decoded frame dimensions differ from the ledger")
    except ArtifactValidationError as exc:
        failures.extend(exc.failures)

    if failures:
        raise ArtifactValidationError(
            [f"{manifest_path}: {failure}" for failure in failures]
        )

    actual_video_sha = evidence_hashes["video"]
    manifest_sha = sha256_file(manifest_path).lower()
    buffer_manifest_sha = sha256_file(buffer_manifest_path).lower()
    expected_paths = {
        "video_path": str(video_path),
        "video_sha256": actual_video_sha,
        "viewport_video_manifest_path": str(manifest_path.resolve()),
        "viewport_video_manifest_sha256": manifest_sha,
    }
    for source_name, payload in (
        ("result", result),
        ("runtime_environment", runtime_environment),
        ("visual_recording_manifest", visual_manifest),
    ):
        if payload.get("actual_viewport_video") is not True:
            failures.append(f"{source_name}.actual_viewport_video is not true")
        if str(payload.get("contact_mode", "")) != contact_mode:
            failures.append(f"{source_name}.contact_mode mismatch")
        for field, expected in expected_paths.items():
            actual = str(payload.get(field, "") or "")
            if field.endswith("sha256"):
                actual = actual.lower()
                expected = expected.lower()
            if actual != expected:
                failures.append(f"{source_name}.{field} mismatch")
    if failures:
        raise ArtifactValidationError(
            [f"{manifest_path}: {failure}" for failure in failures]
        )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "buffer_manifest_path": str(buffer_manifest_path.resolve()),
        "buffer_manifest_sha256": buffer_manifest_sha,
        "video_path": str(video_path),
        "video_sha256": actual_video_sha,
        "ledger_path": str(ledger_path),
        "ledger_sha256": evidence_hashes["ledger"],
        "first_frame_path": str(evidence_paths["first_frame"]),
        "first_frame_sha256": evidence_hashes["first_frame"],
        "last_frame_path": str(evidence_paths["last_frame"]),
        "last_frame_sha256": evidence_hashes["last_frame"],
        "frame_count": frame_count,
        "fps": fps,
        "capture_backend": DIRECT_VIEWPORT_CAPTURE_BACKEND,
        "render_product_path": render_product_path,
        "render_product_unchanged": True,
        "active_render_product_identity_proven": True,
        "capture_graph_created": False,
        "render_observer_only": True,
        "extra_app_update_count": 0,
        "extra_render_count": 0,
        "maximum_pending_captures": 1,
        "frame_ledger_complete": True,
        "full_decode_all_frames": True,
        "full_decode": dict(manifest["full_decode"]),
        "source": DIRECT_VIEWPORT_SOURCE,
    }


def _current_git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MODULE_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        raise ArtifactValidationError(f"current git HEAD is unavailable: {exc}") from exc
    head = str(completed.stdout or "").strip().lower()
    if (
        completed.returncode != 0
        or len(head) not in {40, 64}
        or any(character not in SHA256_HEX for character in head)
    ):
        raise ArtifactValidationError(
            f"current git HEAD is unavailable/invalid: {completed.stderr.strip()!r}"
        )
    return head


def _load_source_closure(
    artifact_root: Path,
    *,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen source closure and return cross-run provenance."""

    pre_path = artifact_root / "source_freeze_pre.json"
    alias_path = artifact_root / "source_freeze.json"
    post_path = artifact_root / "source_freeze_post.json"
    integrity_path = artifact_root / "source_integrity.json"
    required = (pre_path, alias_path, post_path, integrity_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ArtifactValidationError(
            [f"required source-closure artifact is missing: {path}" for path in missing]
        )
    pre = _mapping(_strict_json(pre_path), str(pre_path))
    alias = _mapping(_strict_json(alias_path), str(alias_path))
    post = _mapping(_strict_json(post_path), str(post_path))
    integrity = _mapping(_strict_json(integrity_path), str(integrity_path))
    if alias != pre:
        raise ArtifactValidationError(f"{alias_path}: differs from source_freeze_pre.json")

    def normalize_files(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
        files = _mapping(payload.get("files"), f"{label}.files")
        if not files:
            raise ArtifactValidationError(f"{label}.files is empty")
        normalized: dict[str, Any] = {}
        for source_path, raw in sorted(files.items(), key=lambda item: str(item[0])):
            name = str(source_path or "")
            if not name:
                raise ArtifactValidationError(f"{label}.files has an empty path")
            row = _mapping(raw, f"{label}.files[{name!r}]")
            normalized[name] = {
                "sha256": _sha256(row.get("sha256"), f"{label}.files[{name!r}].sha256"),
                "size_bytes": _integer(
                    row.get("size_bytes"), f"{label}.files[{name!r}].size_bytes"
                ),
            }
            if normalized[name]["size_bytes"] < 0:
                raise ArtifactValidationError(
                    f"{label}.files[{name!r}].size_bytes is negative"
                )
        return normalized

    pre_files = normalize_files(pre, "source_freeze_pre")
    post_files = normalize_files(post, "source_freeze_post")
    if pre_files != post_files:
        raise ArtifactValidationError("source freeze files changed within the replay")
    pre_git = _mapping(pre.get("git"), "source_freeze_pre.git")
    post_git = _mapping(post.get("git"), "source_freeze_post.git")
    pre_head = str(pre_git.get("head", "") or "").strip().lower()
    post_head = str(post_git.get("head", "") or "").strip().lower()
    if not pre_head or pre_head.startswith("unavailable") or pre_head != post_head:
        raise ArtifactValidationError("source freeze git HEAD is missing/unavailable or changed")
    current_head = _current_git_head()
    if current_head != pre_head:
        raise ArtifactValidationError(
            f"current git HEAD {current_head} differs from frozen replay HEAD {pre_head}"
        )
    for source_text, frozen in pre_files.items():
        source_path = Path(source_text)
        if not source_path.is_absolute():
            raise ArtifactValidationError(
                f"source closure path is not absolute: {source_text!r}"
            )
        if not source_path.is_file():
            raise ArtifactValidationError(
                f"source closure file is missing at report time: {source_path}"
            )
        current_size = source_path.stat().st_size
        current_sha = sha256_file(source_path).lower()
        if current_size != frozen["size_bytes"] or current_sha != frozen["sha256"]:
            raise ArtifactValidationError(
                f"source closure file changed after replay: {source_path}"
            )
    pre_created = str(pre.get("created_utc", "") or "")
    post_created = str(post.get("created_utc", "") or "")
    expected_integrity = {
        "equal": True,
        "changed": [],
        "missing": [],
        "added": [],
        "before_created_utc": pre_created,
        "after_created_utc": post_created,
    }
    if not pre_created or not post_created or integrity != expected_integrity:
        raise ArtifactValidationError(
            f"{integrity_path}: does not prove an unchanged pre/post source closure"
        )
    result_integrity = _mapping(result.get("source_integrity"), "result.source_integrity")
    if (
        result_integrity.get("ok") is not True
        or str(result_integrity.get("scope", "")) != "recording_version"
        or _mapping(result_integrity.get("comparison"), "result.source_integrity.comparison")
        != integrity
    ):
        raise ArtifactValidationError(
            "result.source_integrity does not match the durable source-integrity artifact"
        )
    files_json = json.dumps(
        pre_files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "git_head": pre_head,
        "current_git_head": current_head,
        "files": pre_files,
        "files_sha256": hashlib.sha256(files_json).hexdigest(),
        "source_freeze_pre_sha256": sha256_file(pre_path),
        "source_freeze_post_sha256": sha256_file(post_path),
        "source_integrity_sha256": sha256_file(integrity_path),
    }


def _validate_fast_plan(path: Path, plan: dict[str, Any], result: dict[str, Any]) -> None:
    failures: list[str] = []
    exact = {
        "source_version": str(result["source_version"]),
        "profile_requested": "fast",
        "profile_normalized": "motion_only",
        "plan_sha256": _sha256(result["plan_sha256"], "result.plan_sha256"),
        "event_count": _integer(result["plan_event_count"], "result.plan_event_count"),
        "segment_count": _integer(result["plan_segment_count"], "result.plan_segment_count"),
    }
    for key, expected in exact.items():
        actual = plan.get(key)
        if key == "plan_sha256":
            try:
                actual = _sha256(actual, f"fast_plan.{key}")
            except ArtifactValidationError as exc:
                failures.extend(exc.failures)
                continue
        elif key in {"event_count", "segment_count"}:
            try:
                actual = _integer(actual, f"fast_plan.{key}")
            except ArtifactValidationError as exc:
                failures.extend(exc.failures)
                continue
        if actual != expected:
            failures.append(f"fast plan {key} mismatch: {actual!r} != {expected!r}")
    try:
        plan_time = _finite(plan.get("final_time_s"), "fast_plan.final_time_s")
        result_time = _finite(result.get("plan_final_time_s"), "result.plan_final_time_s")
        if not math.isclose(plan_time, result_time, rel_tol=0.0, abs_tol=1.0e-12):
            failures.append("fast plan final_time_s differs from result")
    except ArtifactValidationError as exc:
        failures.extend(exc.failures)
    if failures:
        raise ArtifactValidationError([f"{path}: {failure}" for failure in failures])


def _root_pose(row: Mapping[str, Any], row_label: str) -> list[float]:
    keys = ("base_x_m", "base_y_m", "base_z_m", "base_qw", "base_qx", "base_qy", "base_qz")
    return [_finite(row.get(key), f"{row_label}.{key}") for key in keys]


def _named_finite_mapping(
    value: Any,
    names: Sequence[str],
    label: str,
) -> dict[str, float]:
    values = _mapping(value, label)
    missing = [name for name in names if name not in values]
    if missing:
        raise ArtifactValidationError(f"{label}: missing keys {missing}")
    return {name: _finite(values[name], f"{label}.{name}") for name in names}


def _obstacle_geometry(runtime_environment: Mapping[str, Any], label: str) -> dict[str, Any]:
    geometry = _mapping(runtime_environment.get("live_obstacle_geometry"), f"{label}.live_obstacle_geometry")
    prim_path = str(geometry.get("prim_path", "") or "")
    if not prim_path:
        raise ArtifactValidationError(f"{label}.live_obstacle_geometry.prim_path is missing")
    for flag in ("prim_valid", "visual_valid", "collision_valid"):
        if geometry.get(flag) is not True:
            raise ArtifactValidationError(f"{label}.live_obstacle_geometry.{flag} is not true")
    measured = _mapping(geometry.get("measured_bounds"), f"{label}.measured_bounds")
    collision = _mapping(geometry.get("collision_bounds"), f"{label}.collision_bounds")
    numeric_names = (
        "height_m",
        "width_m",
        "length_m",
        "front_face_x_m",
        "center_y_m",
        "bottom_z_m",
        "top_z_m",
        "collision_height_m",
        "collision_width_m",
    )
    dimensions = {
        name: _finite(geometry.get(name), f"{label}.live_obstacle_geometry.{name}")
        for name in numeric_names
    }
    measured_min = _vector(measured.get("min"), 3, f"{label}.measured_bounds.min")
    measured_max = _vector(measured.get("max"), 3, f"{label}.measured_bounds.max")
    collision_min = _vector(collision.get("min"), 3, f"{label}.collision_bounds.min")
    collision_max = _vector(collision.get("max"), 3, f"{label}.collision_bounds.max")
    if any(upper < lower for lower, upper in zip(measured_min, measured_max)):
        raise ArtifactValidationError(f"{label}.measured_bounds: inverted bounds")
    if any(upper < lower for lower, upper in zip(collision_min, collision_max)):
        raise ArtifactValidationError(f"{label}.collision_bounds: inverted bounds")
    return {
        "prim_path": prim_path,
        "prim_valid": True,
        "visual_valid": True,
        "collision_valid": True,
        "measured_bounds": {"min": measured_min, "max": measured_max},
        "collision_bounds": {"min": collision_min, "max": collision_max},
        "dimensions": dimensions,
    }


def extract_trajectory_metrics(
    telemetry_rows: Sequence[Mapping[str, Any]],
    runtime_environment: Mapping[str, Any],
    *,
    contact_mode: str,
    label: str = "run",
) -> dict[str, Any]:
    """Extract the eight common metrics and validate role-specific evidence.

    ``formal`` uses aggregate contact rows and is not required to expose any
    filtered/non-wheel instrumentation fields.  ``instrumented`` must prove
    that both extra sensor banks are complete, but the compared force metric is
    still the common per-wheel upward/total force recorded in both modes.
    """

    if len(telemetry_rows) < 2:
        raise ArtifactValidationError(f"{label}: at least two telemetry samples are required")
    mode = str(contact_mode or "").strip().lower()
    if mode not in {"formal", "instrumented"}:
        raise ArtifactValidationError(f"{label}: unsupported contact_mode {contact_mode!r}")
    root_trajectory: list[list[float]] = []
    joint_trajectory = {name: [] for name in SERVO_JOINT_NAMES}
    wheel_rotation = {leg: [] for leg in LEGS}
    wheel_travel = {leg: [] for leg in LEGS}
    contact_class = {leg: [] for leg in LEGS}
    contact_force = {
        leg: {"upward_n": [], "total_n": []}
        for leg in LEGS
    }

    for index, row in enumerate(telemetry_rows):
        row_label = f"{label}.telemetry[{index}]"
        root_trajectory.append(_root_pose(row, row_label))
        positions = _named_finite_mapping(
            row.get("measured_joint_position_rad"),
            SERVO_JOINT_NAMES,
            f"{row_label}.measured_joint_position_rad",
        )
        rotations = _named_finite_mapping(
            row.get("wheel_integrated_rotation_rad"),
            LEGS,
            f"{row_label}.wheel_integrated_rotation_rad",
        )
        travels = _named_finite_mapping(
            row.get("wheel_integrated_travel_m"),
            LEGS,
            f"{row_label}.wheel_integrated_travel_m",
        )
        classes = _mapping(row.get("wheel_contact_classes"), f"{row_label}.wheel_contact_classes")
        if row.get("wheel_net_force_layout_valid") is not True:
            raise ArtifactValidationError(
                f"{row_label}.wheel_net_force_layout_valid is not true"
            )
        if row.get("wheel_net_force_valid") is not True:
            raise ArtifactValidationError(f"{row_label}.wheel_net_force_valid is not true")
        if str(row.get("wheel_net_force_error", "") or ""):
            raise ArtifactValidationError(f"{row_label}.wheel_net_force_error is not empty")
        if str(row.get("wheel_contact_force_common_source", "") or "") != (
            COMMON_WHEEL_FORCE_SOURCE
        ):
            raise ArtifactValidationError(
                f"{row_label}.wheel_contact_force_common_source is not "
                f"{COMMON_WHEEL_FORCE_SOURCE!r}"
            )
        upward_forces = _named_finite_mapping(
            row.get("wheel_contact_force_up_n"),
            LEGS,
            f"{row_label}.wheel_contact_force_up_n",
        )
        total_forces = _named_finite_mapping(
            row.get("wheel_contact_force_total_n"),
            LEGS,
            f"{row_label}.wheel_contact_force_total_n",
        )
        if mode == "instrumented":
            for flag in (
                "filtered_contact_layout_valid",
                "filtered_contact_force_valid",
                "filtered_contact_geometry_valid",
                "filtered_contact_consistency_valid",
                "filtered_contact_available",
                "collision_evidence_valid",
            ):
                if row.get(flag) is not True:
                    raise ArtifactValidationError(f"{row_label}.{flag} is not true")
            if str(row.get("filtered_contact_error", "") or ""):
                raise ArtifactValidationError(f"{row_label}.filtered_contact_error is not empty")
            if str(row.get("filtered_contact_consistency_error", "") or ""):
                raise ArtifactValidationError(
                    f"{row_label}.filtered_contact_consistency_error is not empty"
                )
            if str(row.get("collision_evidence_error", "") or ""):
                raise ArtifactValidationError(f"{row_label}.collision_evidence_error is not empty")
            filtered_rows = row.get("wheel_filtered_contacts")
            if not isinstance(filtered_rows, list) or len(filtered_rows) != 8:
                raise ArtifactValidationError(
                    f"{row_label}.wheel_filtered_contacts must contain all eight wheel/surface rows"
                )
            nonwheel_rows = row.get("nonwheel_obstacle_contacts")
            if not isinstance(nonwheel_rows, list) or not nonwheel_rows:
                raise ArtifactValidationError(
                    f"{row_label}.nonwheel_obstacle_contacts must contain sensor rows"
                )
            if not str(row.get("collision_evidence_source", "") or ""):
                raise ArtifactValidationError(
                    f"{row_label}.collision_evidence_source is missing"
                )
        for name in SERVO_JOINT_NAMES:
            joint_trajectory[name].append(positions[name])
        for leg in LEGS:
            wheel_rotation[leg].append(rotations[leg])
            wheel_travel[leg].append(travels[leg])
            value = str(classes.get(leg, ""))
            if value not in CONTACT_CLASSES:
                raise ArtifactValidationError(
                    f"{row_label}.wheel_contact_classes.{leg}: invalid/missing class {value!r}"
                )
            contact_class[leg].append(value)
            contact_force[leg]["upward_n"].append(upward_forces[leg])
            contact_force[leg]["total_n"].append(total_forces[leg])

    return {
        "root_trajectory": root_trajectory,
        "joint_trajectory": joint_trajectory,
        "wheel_rotation": wheel_rotation,
        "wheel_travel": wheel_travel,
        "final_pose": list(root_trajectory[-1]),
        "obstacle_geometry": _obstacle_geometry(runtime_environment, label),
        "contact_class": contact_class,
        "contact_force": contact_force,
    }


def _sample_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    physics_dt_s: float,
    label: str,
) -> dict[str, Any]:
    if len(rows) < 2:
        raise ArtifactValidationError(f"{label}: sample grid requires at least two rows")
    times = [_finite(row.get("time_s"), f"{label}[{index}].time_s") for index, row in enumerate(rows)]
    steps = [_integer(row.get("sim_step"), f"{label}[{index}].sim_step") for index, row in enumerate(rows)]
    samples = [_integer(row.get("sample_index"), f"{label}[{index}].sample_index") for index, row in enumerate(rows)]
    for index in range(1, len(rows)):
        if not times[index] > times[index - 1]:
            raise ArtifactValidationError(f"{label}: time_s is not strictly increasing at row {index}")
        if not steps[index] > steps[index - 1]:
            raise ArtifactValidationError(f"{label}: sim_step is not strictly increasing at row {index}")
        if samples[index] != samples[index - 1] + 1:
            raise ArtifactValidationError(f"{label}: sample_index is not contiguous at row {index}")
    dt = _finite(physics_dt_s, f"{label}.physics_dt_s")
    if dt <= 0.0:
        raise ArtifactValidationError(f"{label}.physics_dt_s must be positive")
    signature: list[list[int]] = []
    time_tolerance = max(1.0e-10, dt * 1.0e-6)
    for index, (time_s, sim_step, sample_index) in enumerate(zip(times, steps, samples)):
        relative_time = time_s - times[0]
        time_ticks = int(round(relative_time / dt))
        if not math.isclose(relative_time, time_ticks * dt, rel_tol=0.0, abs_tol=time_tolerance):
            raise ArtifactValidationError(
                f"{label}[{index}]: time_s is off the physics-dt grid"
            )
        relative_step = sim_step - steps[0]
        if time_ticks != relative_step:
            raise ArtifactValidationError(
                f"{label}[{index}]: sim-time tick {time_ticks} != relative sim_step {relative_step}"
            )
        signature.append([sample_index - samples[0], relative_step, time_ticks])
    return {
        "sample_count": len(rows),
        "physics_dt_s": dt,
        "start_time_s": times[0],
        "end_time_s": times[-1],
        "duration_s": times[-1] - times[0],
        "start_sim_step": steps[0],
        "end_sim_step": steps[-1],
        "signature": signature,
    }


def load_completed_replay_artifact(path: str | Path, *, role: str) -> ReplayArtifact:
    """Load and validate one finalized replay run before metric conversion."""

    run_dir = _resolve_run_dir(path)
    result_path = run_dir / "result.json"
    result = _mapping(_strict_json(result_path), str(result_path))
    worker_owned = _is_formal_worker_result(result)
    if str(role).strip().upper() == "B" and not worker_owned:
        raise ArtifactValidationError(
            "B: legacy direct artifact is diagnostic-only; formal B requires "
            "artifact_owner=sim_worker_process and execution_path=sim_worker_process_ipc"
        )
    artifact_root_text = str(result.get("artifact_root", "") or "")
    if not artifact_root_text:
        raise ArtifactValidationError(f"{result_path}: artifact_root is missing")
    artifact_root = Path(artifact_root_text).resolve()
    if not artifact_root.is_dir() or not _is_within(run_dir, artifact_root):
        raise ArtifactValidationError(f"{result_path}: artifact_root is invalid or does not contain run_dir")
    if any(".partial" in part.lower() for part in (*artifact_root.parts, *run_dir.relative_to(artifact_root).parts)):
        raise ArtifactValidationError(f"{artifact_root}: partial path component is forbidden")
    if (artifact_root / ".partial").exists() or (run_dir / ".partial").exists():
        raise ArtifactValidationError(f"{artifact_root}: .partial marker is present")
    if not (artifact_root / ".finalized").is_file() or (artifact_root / ".failed").exists():
        raise ArtifactValidationError(f"{artifact_root}: finalized marker missing or failed marker present")
    _validate_result(result, result_path=result_path, run_dir=run_dir, artifact_root=artifact_root)

    pointer_path = artifact_root / "artifact_pointer.json"
    pointer = _mapping(_strict_json(pointer_path), str(pointer_path))
    if Path(str(pointer.get("run_dir", ""))).resolve() != run_dir:
        raise ArtifactValidationError(f"{pointer_path}: run_dir pointer mismatch")
    batch_root = _resolve_batch_root(artifact_root, run_dir)
    batch_shutdown_closure = _load_batch_shutdown_closure(
        batch_root=batch_root,
        artifact_root=artifact_root,
        run_dir=run_dir,
        result_path=result_path,
        result=result,
    )
    source_closure = _load_source_closure(artifact_root, result=result)

    paths = {name: run_dir / name for name in REQUIRED_RUN_FILENAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactValidationError([f"required artifact is missing: {path}" for path in missing])
    runtime_environment = _mapping(_strict_json(paths["runtime_environment.json"]), str(paths["runtime_environment.json"]))
    contact_mode = _resolve_contact_mode(
        role=role,
        result=result,
        runtime_environment=runtime_environment,
        artifact_root=artifact_root,
    )
    visual_manifest = _mapping(_strict_json(paths["visual_recording_manifest.json"]), str(paths["visual_recording_manifest.json"]))
    _validate_visual_manifest(
        paths["visual_recording_manifest.json"],
        visual_manifest,
        contact_mode=contact_mode,
    )
    viewport_video = _load_viewport_video_evidence(
        run_dir,
        contact_mode=contact_mode,
        result=result,
        runtime_environment=runtime_environment,
        visual_manifest=visual_manifest,
    )
    physical_evidence = _mapping(_strict_json(paths["physical_evidence.json"]), str(paths["physical_evidence.json"]))
    if physical_evidence != _mapping(result.get("physical_evidence"), "result.physical_evidence"):
        raise ArtifactValidationError("physical_evidence.json differs from result.physical_evidence")
    telemetry_finalization_payload = _mapping(
        _strict_json(paths["telemetry_finalization.json"]),
        str(paths["telemetry_finalization.json"]),
    )
    telemetry_finalization = _validate_telemetry_finalization(
        paths["telemetry_finalization.json"],
        telemetry_finalization_payload,
        run_dir=run_dir,
        result=result,
    )
    failure_diagnostics = _mapping(_strict_json(paths["failure_diagnostics.json"]), str(paths["failure_diagnostics.json"]))
    if str(failure_diagnostics.get("classification", "")) != str(result.get("classification", "")):
        raise ArtifactValidationError("failure_diagnostics classification differs from result")

    fast_plan_paths = sorted((run_dir / "input" / "fast_plan").glob("*_fast_plan.json"))
    if len(fast_plan_paths) != 1:
        raise ArtifactValidationError(
            f"{run_dir}: expected exactly one input/fast_plan/*_fast_plan.json, found {len(fast_plan_paths)}"
        )
    fast_plan_path = fast_plan_paths[0]
    fast_plan = _mapping(_strict_json(fast_plan_path), str(fast_plan_path))
    _validate_fast_plan(fast_plan_path, fast_plan, result)

    accepted_steps_path = run_dir / "input" / "accepted_steps.jsonl"
    if not accepted_steps_path.is_file():
        raise ArtifactValidationError(f"{accepted_steps_path}: copied recording input is required")
    accepted_steps_sha = sha256_file(accepted_steps_path).lower()
    result_recording_sha = _sha256(
        result["accepted_steps_sha256"], "accepted_steps_sha256"
    )
    if accepted_steps_sha != result_recording_sha:
        raise ArtifactValidationError(
            f"{accepted_steps_path}: file SHA-256 differs from result.accepted_steps_sha256"
        )

    required_checksum_paths = [
        *paths.values(),
        accepted_steps_path,
        fast_plan_path,
        Path(viewport_video["manifest_path"]),
        Path(viewport_video["buffer_manifest_path"]),
        Path(viewport_video["video_path"]),
        Path(viewport_video["ledger_path"]),
        Path(viewport_video["first_frame_path"]),
        Path(viewport_video["last_frame_path"]),
    ]
    checksum_manifest = _validate_checksums(run_dir, required_checksum_paths)
    telemetry_rows = _jsonl(paths["fsm50_telemetry.jsonl"])
    source_version = str(result["source_version"])
    for index, row in enumerate(telemetry_rows):
        if str(row.get("source_version", "")) != source_version:
            raise ArtifactValidationError(
                f"telemetry[{index}].source_version differs from result: {row.get('source_version')!r}"
            )
    if str(physical_evidence.get("source_version", "")) != source_version:
        raise ArtifactValidationError("physical_evidence.source_version differs from result")
    if _integer(physical_evidence.get("sample_count"), "physical_evidence.sample_count") != len(
        telemetry_rows
    ):
        raise ArtifactValidationError("physical_evidence.sample_count differs from telemetry")
    canonical_fsm = _mapping(
        telemetry_finalization["canonical_files"].get("fsm50_telemetry.jsonl"),
        "telemetry_finalization.fsm50_telemetry.jsonl",
    )
    if _integer(
        canonical_fsm.get("record_count"),
        "telemetry_finalization.fsm50_telemetry.jsonl.record_count",
    ) != len(telemetry_rows):
        raise ArtifactValidationError(
            "telemetry finalization FSM record_count differs from telemetry"
        )
    if _integer(
        telemetry_finalization["stream_counts"].get("fsm50_telemetry"),
        "telemetry_finalization.stream_counts.fsm50_telemetry",
    ) != len(telemetry_rows):
        raise ArtifactValidationError(
            "telemetry finalization FSM stream count differs from telemetry"
        )

    runtime_equivalence = _mapping(
        runtime_environment.get("environment_equivalence"),
        "runtime_environment.environment_equivalence",
    )
    if runtime_equivalence.get("ok") is not True:
        raise ArtifactValidationError("runtime_environment.environment_equivalence.ok is not true")
    scene_config = _mapping(runtime_environment.get("scene_config"), "runtime_environment.scene_config")
    device = str(scene_config.get("device", "") or "")
    if not device:
        raise ArtifactValidationError("runtime_environment.scene_config.device is missing")
    physics_dt = _finite(runtime_environment.get("physics_dt_s"), "runtime_environment.physics_dt_s")
    if physics_dt <= 0.0:
        raise ArtifactValidationError("runtime_environment.physics_dt_s must be positive")
    render_interval = _integer(
        runtime_environment.get("render_interval"),
        "runtime_environment.render_interval",
    )
    if render_interval <= 0:
        raise ArtifactValidationError("runtime_environment.render_interval must be positive")
    metrics = extract_trajectory_metrics(
        telemetry_rows,
        runtime_environment,
        contact_mode=contact_mode,
        label=str(role),
    )
    grid = _sample_grid(
        telemetry_rows,
        physics_dt_s=physics_dt,
        label=f"{role}.sample_grid",
    )
    accepted_sha = _sha256(result["accepted_steps_sha256"], "accepted_steps_sha256")
    plan_sha = _sha256(result["plan_sha256"], "plan_sha256")
    runtime_versions = _mapping(
        runtime_environment.get("runtime"), "runtime_environment.runtime"
    )
    if not runtime_versions:
        raise ArtifactValidationError("runtime_environment.runtime is empty")
    provenance = {
        "role": str(role),
        "source_version": source_version,
        "accepted_steps_sha256": accepted_sha,
        "plan_sha256": plan_sha,
        "requested_profile": "fast",
        "canonical_profile": "motion_only",
        "contact_mode": contact_mode,
        "artifact_owner": str(result.get("artifact_owner", "") or "legacy_direct"),
        "execution_path": str(result.get("execution_path", "") or "legacy_direct"),
        "plan_event_count": _integer(result["plan_event_count"], "plan_event_count"),
        "plan_segment_count": _integer(result["plan_segment_count"], "plan_segment_count"),
        "plan_final_time_s": _finite(result["plan_final_time_s"], "plan_final_time_s"),
        "device": device,
        "physics_dt_s": physics_dt,
        "render_interval": render_interval,
        "runtime_versions": runtime_versions,
        "source_git_head": source_closure["git_head"],
        "source_files_sha256": source_closure["files_sha256"],
        "source_closure": source_closure,
        "batch_root": str(batch_root),
        "batch_shutdown_status": batch_shutdown_closure["status"],
        "batch_finalization_phase": batch_shutdown_closure["phase"],
        "batch_shutdown_closure": batch_shutdown_closure,
        "formal_worker_closure": batch_shutdown_closure.get(
            "formal_worker_closure"
        ),
        "viewport_video": viewport_video,
        "telemetry_finalization": telemetry_finalization,
        "sample_grid": grid,
        "artifact_hashes": {
            path.relative_to(run_dir).as_posix(): checksum_manifest[path.relative_to(run_dir).as_posix()]
            for path in required_checksum_paths
        },
        "checksums_sha256": sha256_file(run_dir / "checksums.sha256"),
        "metrics_sha256": hashlib.sha256(
            json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return ReplayArtifact(
        role=str(role),
        artifact_root=artifact_root,
        run_dir=run_dir,
        result=result,
        runtime_environment=runtime_environment,
        visual_manifest=visual_manifest,
        fast_plan=fast_plan,
        telemetry_rows=tuple(telemetry_rows),
        trajectory_metrics=metrics,
        provenance=provenance,
    )


def _require_exact_diagnostic_role(
    payload: Mapping[str, Any],
    *,
    role: str,
    label: str,
) -> None:
    if not _type_strict_json_equal(
        payload.get("environment_equivalence_role"), role
    ):
        raise ArtifactValidationError(
            f"{label}.environment_equivalence_role must equal {role!r} exactly"
        )


def _validate_trajectory_diagnostic_dispatch(
    artifact: ReplayArtifact,
) -> dict[str, Any]:
    """Revalidate the sealed source-dispatch evidence used by diagnostics."""

    run_dir = artifact.run_dir
    paths = {
        name: run_dir / name for name in TRAJECTORY_DIAGNOSTIC_DISPATCH_FILENAMES
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactValidationError(
            [f"required trajectory-diagnostic dispatch artifact is missing: {path}" for path in missing]
        )
    _validate_checksum_manifest(run_dir, run_dir / "checksums.sha256", list(paths.values()))

    result = artifact.result
    if result.get("dispatch_complete") is not True:
        raise ArtifactValidationError("result.dispatch_complete is not true")
    result_ledger = _mapping(result.get("dispatch_ledger"), "result.dispatch_ledger")
    ledger = _mapping(
        _strict_json(paths["V003_DISPATCH_TRACE.json"]),
        str(paths["V003_DISPATCH_TRACE.json"]),
    )
    expected_ledger = {
        key: ledger.get(key)
        for key in result_ledger
        if key not in {"csv_path", "json_path"}
    }
    result_ledger_without_paths = {
        key: value
        for key, value in result_ledger.items()
        if key not in {"csv_path", "json_path"}
    }
    if not _type_strict_json_equal(
        result_ledger_without_paths, expected_ledger
    ):
        raise ArtifactValidationError(
            "result.dispatch_ledger differs from V003_DISPATCH_TRACE.json"
        )
    for key, expected in (
        ("json_path", paths["V003_DISPATCH_TRACE.json"]),
        ("csv_path", paths["V003_DISPATCH_TRACE.csv"]),
    ):
        _require_same_path(result_ledger, key, expected, label="result.dispatch_ledger")

    result_plan_events = _exact_integer(
        result.get("plan_event_count"), "result.plan_event_count"
    )
    result_plan_segments = _exact_integer(
        result.get("plan_segment_count"), "result.plan_segment_count"
    )
    ledger_expected = {
        "schema_version": "fsm50.source_dispatch_ledger.v1",
        "complete": True,
        "source_version": str(result.get("source_version", "") or ""),
        "plan_event_count": result_plan_events,
        "retained_plan_event_count": result_plan_events,
        "live_timing_command_count": result_plan_events,
        "plan_segment_count": result_plan_segments,
        "primary_motion_batch_count": result_plan_segments,
        "one_motion_batch_per_physics_tick": True,
        "motion_start_readiness_token_count": 1,
        "playback_start_boundary_count": 1,
        "final_safety_stop_count": 1,
    }
    if not _type_strict_json_equal(
        {key: ledger.get(key) for key in ledger_expected}, ledger_expected
    ):
        raise ArtifactValidationError(
            "trajectory-diagnostic dispatch ledger identity/count contract is invalid"
        )
    if list(ledger.get("errors", []) or []):
        raise ArtifactValidationError("trajectory-diagnostic dispatch ledger records errors")
    rows = ledger.get("rows")
    motion_batches = ledger.get("motion_batches")
    if (
        not isinstance(rows, list)
        or not isinstance(motion_batches, list)
        or len(rows) < result_plan_events
        or len(motion_batches) < result_plan_segments
    ):
        raise ArtifactValidationError(
            "trajectory-diagnostic dispatch ledger rows/batches are incomplete"
        )

    adapter_id = str(result.get("adapter_runtime_instance_id", "") or "")
    readiness = _mapping(
        _strict_json(paths["motion_start_readiness.json"]),
        str(paths["motion_start_readiness.json"]),
    )
    pre_first = _mapping(
        _strict_json(paths["motion_start_pre_first_dispatch.json"]),
        str(paths["motion_start_pre_first_dispatch.json"]),
    )
    readiness_expected = {
        "schema_version": "fsm50.motion_start_readiness_window.v1",
        "ready": True,
        "status": "PASS",
        "gate": "MOTION_START_READY",
        "root_state_write_count": 0,
        "writes_robot_state": False,
        "command_dispatch_idle": True,
        "adapter_runtime_instance_id": adapter_id,
    }
    for label, payload in (("motion_start_readiness", readiness), ("motion_start_pre_first_dispatch", pre_first)):
        if not _type_strict_json_equal(
            {key: payload.get(key) for key in readiness_expected},
            readiness_expected,
        ) or list(payload.get("window_failed_checks", []) or []):
            raise ArtifactValidationError(f"{label} is not a sealed idle PASS")
    if (
        pre_first.get("readiness_token_bound") is not True
        or _exact_integer(
            pre_first.get("source_command_dispatch_count"),
            "motion_start_pre_first_dispatch.source_command_dispatch_count",
        )
        != 0
        or pre_first.get("boundary_batch_is_not_source_command") is not True
    ):
        raise ArtifactValidationError(
            "motion_start_pre_first_dispatch does not prove a bound token before source dispatch"
        )

    timing = _mapping(
        _strict_json(paths["production_dispatch_timing.json"]),
        str(paths["production_dispatch_timing.json"]),
    )
    commands = timing.get("commands")
    if not isinstance(commands, list) or len(commands) < result_plan_events:
        raise ArtifactValidationError(
            "production_dispatch_timing does not cover every retained plan event"
        )
    return {
        "dispatch_complete": True,
        "source_version": ledger_expected["source_version"],
        "plan_event_count": result_plan_events,
        "plan_segment_count": result_plan_segments,
        "readiness_sha256": sha256_file(paths["motion_start_readiness.json"]).lower(),
        "pre_first_dispatch_sha256": sha256_file(
            paths["motion_start_pre_first_dispatch.json"]
        ).lower(),
        "dispatch_json_sha256": sha256_file(paths["V003_DISPATCH_TRACE.json"]).lower(),
        "dispatch_csv_sha256": sha256_file(paths["V003_DISPATCH_TRACE.csv"]).lower(),
        "dispatch_timing_sha256": sha256_file(
            paths["production_dispatch_timing.json"]
        ).lower(),
    }


def load_sealed_trajectory_diagnostic_artifact(
    path: str | Path,
    *,
    role: str,
) -> ReplayArtifact:
    """Load one formal A1/A2/B artifact for trajectory diagnosis only.

    The qualification loader remains the single sealed-artifact validator.
    This thin wrapper adds the diagnostic role, dispatch, and scope bindings;
    it deliberately does not require or infer a physical PASS.
    """

    normalized_role = str(role).strip().upper()
    expected_contact_mode = TRAJECTORY_DIAGNOSTIC_ROLE_CONTACT_MODES.get(
        normalized_role
    )
    if expected_contact_mode is None:
        raise ArtifactValidationError(
            f"trajectory diagnostic role must be one of A1, A2, B; got {role!r}"
        )
    artifact = load_completed_replay_artifact(path, role=normalized_role)
    result = artifact.result
    if (
        artifact.provenance.get("artifact_owner") != FORMAL_WORKER_ARTIFACT_OWNER
        or artifact.provenance.get("execution_path")
        != FORMAL_WORKER_EXECUTION_PATH
        or artifact.provenance.get("contact_mode") != expected_contact_mode
    ):
        raise ArtifactValidationError(
            f"{normalized_role}: trajectory diagnostic requires the formal worker/contact-mode contract"
        )
    _require_exact_diagnostic_role(
        result, role=normalized_role, label="result"
    )
    if (
        result.get("environment_equivalence_diagnostic") is not True
        or result.get("environment_equivalence_diagnostic_complete") is not True
        or result.get("qualification_scope")
        != TRAJECTORY_COMPARISON_QUALIFICATION_SCOPE
        or result.get("artifact_valid") is not True
        or result.get("scheduler_complete") is not True
        or result.get("dispatch_complete") is not True
        or _mapping(result.get("source_integrity"), "result.source_integrity").get("ok")
        is not True
    ):
        raise ArtifactValidationError(
            "result is not a complete TRAJECTORY_COMPARISON diagnostic"
        )

    batch_root = Path(str(artifact.provenance["batch_root"])).resolve()
    batch_request = _mapping(
        _strict_json(batch_root / "batch_request.json"), "batch_request.json"
    )
    worker_request = _mapping(
        _strict_json(batch_root / "worker_artifact_request.json"),
        "worker_artifact_request.json",
    )
    startup = _mapping(
        _strict_json(batch_root / "worker_startup_binding.json"),
        "worker_startup_binding.json",
    )
    complete_ack = _mapping(
        _strict_json(batch_root / "worker_artifact_complete_ack.json"),
        "worker_artifact_complete_ack.json",
    )
    _require_exact_diagnostic_role(
        batch_request, role=normalized_role, label="batch_request"
    )
    if batch_request.get("qualification_scope") != TRAJECTORY_COMPARISON_QUALIFICATION_SCOPE:
        raise ArtifactValidationError(
            "batch_request.qualification_scope is not TRAJECTORY_COMPARISON"
        )
    batch_args = _mapping(batch_request.get("args"), "batch_request.args")
    _require_exact_diagnostic_role(
        batch_args, role=normalized_role, label="batch_request.args"
    )
    _require_exact_diagnostic_role(
        worker_request, role=normalized_role, label="worker_artifact_request"
    )
    for label, payload in (
        ("batch_request.args", batch_args),
        ("worker_artifact_request", worker_request),
    ):
        if payload.get("contact_mode") != expected_contact_mode:
            raise ArtifactValidationError(
                f"{label}.contact_mode does not match diagnostic role {normalized_role}"
            )
    startup_status = _mapping(startup.get("status"), "worker_startup_binding.status")
    startup_session = _mapping(
        startup_status.get("worker_artifact_session"),
        "worker_startup_binding.status.worker_artifact_session",
    )
    startup_preflight = _mapping(
        startup_status.get("worker_artifact_preflight"),
        "worker_startup_binding.status.worker_artifact_preflight",
    )
    _require_exact_diagnostic_role(
        startup_session, role=normalized_role, label="worker startup session"
    )
    _require_exact_diagnostic_role(
        startup_preflight, role=normalized_role, label="worker startup preflight"
    )
    _require_exact_diagnostic_role(
        complete_ack, role=normalized_role, label="worker artifact-complete ACK"
    )
    if complete_ack.get("environment_equivalence_diagnostic_complete") is not True:
        raise ArtifactValidationError(
            "worker artifact-complete ACK diagnostic_complete is not true"
        )

    runtime = artifact.runtime_environment
    _require_exact_diagnostic_role(
        runtime, role=normalized_role, label="runtime_environment"
    )
    failure_diagnostics = _mapping(
        _strict_json(artifact.run_dir / "failure_diagnostics.json"),
        "failure_diagnostics.json",
    )
    _require_exact_diagnostic_role(
        failure_diagnostics,
        role=normalized_role,
        label="failure_diagnostics",
    )
    session_manifest = _mapping(
        _strict_json(artifact.run_dir / "worker_recording_session.json"),
        "worker_recording_session.json",
    )
    _require_exact_diagnostic_role(
        session_manifest,
        role=normalized_role,
        label="worker_recording_session",
    )
    for label, payload in (
        ("failure_diagnostics", failure_diagnostics),
        ("worker_recording_session", session_manifest),
    ):
        if payload.get("environment_equivalence_diagnostic_complete") is not True:
            raise ArtifactValidationError(f"{label} diagnostic_complete is not true")

    dispatch = _validate_trajectory_diagnostic_dispatch(artifact)
    physical_observation = {
        "classification": str(result.get("classification", "") or ""),
        "physical_success": result.get("physical_success"),
        "strict_full_success": result.get("strict_full_success"),
        "used_for_diagnostic_admission": False,
    }
    provenance = {
        **artifact.provenance,
        "role": normalized_role,
        "environment_equivalence_role": normalized_role,
        "qualification_scope": TRAJECTORY_COMPARISON_QUALIFICATION_SCOPE,
        "diagnostic_scope": TRAJECTORY_DIAGNOSTIC_SCOPE,
        "physical_claim": TRAJECTORY_DIAGNOSTIC_PHYSICAL_CLAIM,
        "observed_physical_outcome": physical_observation,
        "dispatch_evidence": dispatch,
    }
    return ReplayArtifact(
        role=normalized_role,
        artifact_root=artifact.artifact_root,
        run_dir=artifact.run_dir,
        result=artifact.result,
        runtime_environment=artifact.runtime_environment,
        visual_manifest=artifact.visual_manifest,
        fast_plan=artifact.fast_plan,
        telemetry_rows=artifact.telemetry_rows,
        trajectory_metrics=artifact.trajectory_metrics,
        provenance=provenance,
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_artifact_triplet(
    a1: ReplayArtifact,
    a2: ReplayArtifact,
    b: ReplayArtifact,
) -> dict[str, Any]:
    """Validate shared provenance, normalized config, and the sample grid."""

    artifacts = (a1, a2, b)
    failures: list[str] = []
    if len({artifact.run_dir.resolve() for artifact in artifacts}) != 3:
        failures.append("A1, A2, and B must be three distinct run directories")
    contact_modes = [artifact.provenance.get("contact_mode") for artifact in artifacts]
    if contact_modes != ["formal", "formal", "instrumented"]:
        failures.append(
            "role/contact-mode contract requires A1=formal, A2=formal, "
            f"B=instrumented; got {contact_modes!r}"
        )
    common_fields = (
        "accepted_steps_sha256",
        "plan_sha256",
        "source_version",
        "requested_profile",
        "canonical_profile",
        "artifact_owner",
        "execution_path",
        "plan_event_count",
        "plan_segment_count",
        "device",
        "render_interval",
        "source_git_head",
        "source_files_sha256",
        "batch_shutdown_status",
        "batch_finalization_phase",
    )
    for field in common_fields:
        values = [artifact.provenance[field] for artifact in artifacts]
        if values[1:] != values[:-1]:
            failures.append(f"{field} differs across A1/A2/B: {values!r}")
    for field in ("physics_dt_s", "plan_final_time_s"):
        values = [float(artifact.provenance[field]) for artifact in artifacts]
        if not all(math.isclose(values[0], value, rel_tol=0.0, abs_tol=1.0e-12) for value in values[1:]):
            failures.append(f"{field} differs across A1/A2/B: {values!r}")
    runtime_versions = [_canonical(artifact.provenance["runtime_versions"]) for artifact in artifacts]
    if runtime_versions[1:] != runtime_versions[:-1]:
        failures.append("runtime package versions differ across A1/A2/B")
    source_file_maps = [
        _canonical(artifact.provenance["source_closure"]["files"])
        for artifact in artifacts
    ]
    if source_file_maps[1:] != source_file_maps[:-1]:
        failures.append("frozen source file closure differs across A1/A2/B")
    grid_signatures = [_canonical(artifact.provenance["sample_grid"]["signature"]) for artifact in artifacts]
    if grid_signatures[1:] != grid_signatures[:-1]:
        failures.append("normalized simulation-time/sample grid differs across A1/A2/B")
    for field in ("sample_count", "start_sim_step", "end_sim_step"):
        values = [artifact.provenance["sample_grid"][field] for artifact in artifacts]
        if values[1:] != values[:-1]:
            failures.append(f"sample-grid {field} differs across A1/A2/B: {values!r}")
    for field in ("start_time_s", "end_time_s", "duration_s"):
        values = [float(artifact.provenance["sample_grid"][field]) for artifact in artifacts]
        if not all(
            math.isclose(values[0], value, rel_tol=0.0, abs_tol=1.0e-9)
            for value in values[1:]
        ):
            failures.append(f"sample-grid {field} differs across A1/A2/B: {values!r}")

    scene_configs = [
        _mapping(artifact.runtime_environment.get("scene_config"), f"{artifact.role}.scene_config")
        for artifact in artifacts
    ]
    if _canonical(scene_configs[0]) != _canonical(scene_configs[1]):
        failures.append("A1 and A2 scene configs are not exact repeats")
    sensor_readbacks = [
        {
            "contact_sensor_type": artifact.runtime_environment.get("contact_sensor_type"),
            "contact_sensor_error": artifact.runtime_environment.get("contact_sensor_error"),
        }
        for artifact in artifacts
    ]
    ab1 = compare_instrumentation_configs(
        scene_configs[0],
        scene_configs[2],
        baseline_sensor_readback=sensor_readbacks[0],
        instrumented_sensor_readback=sensor_readbacks[2],
    )
    ab2 = compare_instrumentation_configs(
        scene_configs[1],
        scene_configs[2],
        baseline_sensor_readback=sensor_readbacks[1],
        instrumented_sensor_readback=sensor_readbacks[2],
    )
    if not ab1["ok"] or not ab2["ok"]:
        failures.append("B changes physical scene configuration outside the instrumentation allow-list")
    allowed_b_differences = sorted(
        set(ab1["allowed_instrumentation_differences"])
        | set(ab2["allowed_instrumentation_differences"])
        | set(ab1["allowed_sensor_readback_differences"])
        | set(ab2["allowed_sensor_readback_differences"])
    )
    if not allowed_b_differences:
        failures.append("B has no explicit instrumentation/readback difference from A")

    return {
        "schema_version": "fsm50.environment_ab_triplet_validation.v1",
        "ok": not failures,
        "failures": failures,
        "common_provenance": {
            field: a1.provenance[field] for field in (*common_fields, "physics_dt_s", "plan_final_time_s")
        },
        "contact_modes": {artifact.role: artifact.provenance["contact_mode"] for artifact in artifacts},
        "sample_grid": {
            artifact.role: artifact.provenance["sample_grid"] for artifact in artifacts
        },
        "A1_vs_B": ab1,
        "A2_vs_B": ab2,
        "allowed_B_differences": allowed_b_differences,
    }


def _failure_check(schema: str, failures: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "ok": False,
        "fail_closed": True,
        "failures": [str(failure) for failure in failures],
    }


def compare_sealed_trajectory_diagnostics(
    *,
    baseline_a1: str | Path,
    baseline_a2: str | Path,
    instrumented_b: str | Path,
    self_error_multiplier: float = 3.0,
    absolute_floors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare a sealed A1/A2/B matrix without making a physical claim."""

    artifact_admission: dict[str, Any]
    trajectory: dict[str, Any]
    triplet: dict[str, Any]
    artifacts: tuple[ReplayArtifact, ReplayArtifact, ReplayArtifact] | None = None
    try:
        artifacts = (
            load_sealed_trajectory_diagnostic_artifact(
                baseline_a1, role="A1"
            ),
            load_sealed_trajectory_diagnostic_artifact(
                baseline_a2, role="A2"
            ),
            load_sealed_trajectory_diagnostic_artifact(
                instrumented_b, role="B"
            ),
        )
        triplet = validate_artifact_triplet(*artifacts)
        if triplet.get("ok") is True:
            trajectory = compare_trajectory_equivalence(
                artifacts[0].trajectory_metrics,
                artifacts[1].trajectory_metrics,
                artifacts[2].trajectory_metrics,
                self_error_multiplier=self_error_multiplier,
                absolute_floors=absolute_floors,
            )
        else:
            trajectory = _failure_check(
                "fsm50.trajectory_equivalence.v1",
                [
                    "trajectory comparison was blocked by invalid A1/A2/B provenance",
                    *triplet.get("failures", []),
                ],
            )
        metric_rows = trajectory.get("metrics")
        if isinstance(metric_rows, Mapping):
            for metric, interpretation in {
                "contact_class": (
                    "observation-semantics-sensitive: formal A aggregate fallback and "
                    "instrumented B surface-aware labels may legitimately differ in overlap; "
                    "this categorical result is not a dynamics or physical-success claim"
                ),
                "contact_force": (
                    "observation-semantics-sensitive: both modes use common net_forces_w, "
                    "but sensor construction and aggregation layout may affect the readback; "
                    "this result is not a dynamics or physical-success claim"
                ),
            }.items():
                if metric in metric_rows:
                    metric_rows[metric]["interpretation"] = interpretation
            trajectory["sensor_independent_trajectory_ok"] = all(
                isinstance(metric_rows.get(metric), Mapping)
                and metric_rows[metric].get("ok") is True
                for metric in SENSOR_INDEPENDENT_TRAJECTORY_METRICS
            )
            trajectory["observation_metrics_ok"] = all(
                isinstance(metric_rows.get(metric), Mapping)
                and metric_rows[metric].get("ok") is True
                for metric in OBSERVATION_TRAJECTORY_METRICS
            )
            trajectory["metric_groups"] = {
                "sensor_independent_trajectory": list(
                    SENSOR_INDEPENDENT_TRAJECTORY_METRICS
                ),
                "observation_sensitive": list(OBSERVATION_TRAJECTORY_METRICS),
            }
        artifact_admission = {
            "schema_version": "fsm50.sealed_trajectory_diagnostic_admission.v1",
            "ok": True,
            "fail_closed": True,
            "roles": {
                artifact.role: {
                    "artifact_root": str(artifact.artifact_root),
                    "run_dir": str(artifact.run_dir),
                    "artifact_owner": artifact.provenance["artifact_owner"],
                    "execution_path": artifact.provenance["execution_path"],
                    "contact_mode": artifact.provenance["contact_mode"],
                    "qualification_scope": artifact.provenance[
                        "qualification_scope"
                    ],
                    "closure_sha256": artifact.provenance[
                        "batch_shutdown_closure"
                    ]["closure_sha256"],
                    "metrics_sha256": artifact.provenance["metrics_sha256"],
                    "video_sha256": artifact.provenance["viewport_video"][
                        "video_sha256"
                    ],
                    "dispatch_evidence": artifact.provenance[
                        "dispatch_evidence"
                    ],
                    "observed_physical_outcome": artifact.provenance[
                        "observed_physical_outcome"
                    ],
                }
                for artifact in artifacts
            },
        }
    except (ArtifactValidationError, ValueError) as exc:
        failures = (
            list(exc.failures)
            if isinstance(exc, ArtifactValidationError)
            else [str(exc)]
        )
        artifact_admission = _failure_check(
            "fsm50.sealed_trajectory_diagnostic_admission.v1", failures
        )
        triplet = _failure_check(
            "fsm50.environment_ab_triplet_validation.v1", failures
        )
        trajectory = _failure_check(
            "fsm50.trajectory_equivalence.v1", failures
        )

    return {
        "schema_version": "fsm50.sealed_trajectory_diagnostic_comparison.v1",
        "ok": bool(
            artifact_admission.get("ok") is True
            and triplet.get("ok") is True
            and trajectory.get("ok") is True
        ),
        "fail_closed": True,
        "diagnostic_scope": TRAJECTORY_DIAGNOSTIC_SCOPE,
        "physical_claim": TRAJECTORY_DIAGNOSTIC_PHYSICAL_CLAIM,
        "qualification_eligible": False,
        "interpretation": (
            "A1/A2 self-error and A/B trajectory-diagnostic comparison only; "
            "physical PASS/FAIL/NOT_EVALUABLE is neither required nor promoted"
        ),
        "artifact_admission": artifact_admission,
        "triplet_validation": triplet,
        "trajectory_comparison": trajectory,
        "sensor_independent_trajectory_ok": (
            trajectory.get("sensor_independent_trajectory_ok")
            if isinstance(trajectory, Mapping)
            else None
        ),
        "observation_metrics_ok": (
            trajectory.get("observation_metrics_ok")
            if isinstance(trajectory, Mapping)
            else None
        ),
    }


def compare_environment_run_artifacts(
    *,
    baseline_a1: str | Path,
    baseline_a2: str | Path,
    instrumented_b: str | Path,
    self_error_multiplier: float = 3.0,
    absolute_floors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Validate and compare A1/A2/B without writing a report.

    Parameters are keyword-only paths to two ``contact_mode=formal`` runs and
    one ``contact_mode=instrumented`` run.  The returned object always has the
    keys ``instrumentation_comparison``, ``trajectory_comparison``,
    ``runtime_readback``, and ``artifact_conversion``.  Artifact defects are
    represented as ``ok=false`` fail-closed checks instead of invented data.
    """

    artifacts: tuple[ReplayArtifact, ReplayArtifact, ReplayArtifact] | None = None
    try:
        artifacts = (
            load_completed_replay_artifact(baseline_a1, role="A1"),
            load_completed_replay_artifact(baseline_a2, role="A2"),
            load_completed_replay_artifact(instrumented_b, role="B"),
        )
        triplet = validate_artifact_triplet(*artifacts)
        conversion = {
            "schema_version": CONVERSION_SCHEMA,
            "ok": True,
            "metric_names": sorted(artifacts[0].trajectory_metrics),
            "metric_manifests": {
                artifact.role: {
                    "sample_count": len(artifact.telemetry_rows),
                    "metrics_sha256": artifact.provenance["metrics_sha256"],
                    "telemetry_sha256": artifact.provenance["artifact_hashes"][
                        "fsm50_telemetry.jsonl"
                    ],
                }
                for artifact in artifacts
            },
        }
        runtime_readback = {
            "schema_version": "fsm50.environment_ab_runtime_readback.v1",
            "ok": True,
            "readback_complete": True,
            "triplet_validation": triplet,
            "runs": {
                artifact.role: {
                    "artifact_root": str(artifact.artifact_root),
                    "run_dir": str(artifact.run_dir),
                    "provenance": artifact.provenance,
                }
                for artifact in artifacts
            },
        }
        if triplet["ok"]:
            trajectory = compare_trajectory_equivalence(
                artifacts[0].trajectory_metrics,
                artifacts[1].trajectory_metrics,
                artifacts[2].trajectory_metrics,
                self_error_multiplier=self_error_multiplier,
                absolute_floors=absolute_floors,
            )
        else:
            trajectory = _failure_check(
                "fsm50.trajectory_equivalence.v1",
                ["trajectory comparison was blocked by invalid A/A-B provenance", *triplet["failures"]],
            )
    except ArtifactValidationError as exc:
        failures = list(exc.failures)
        triplet = _failure_check("fsm50.environment_ab_triplet_validation.v1", failures)
        trajectory = _failure_check("fsm50.trajectory_equivalence.v1", failures)
        runtime_readback = {
            "schema_version": "fsm50.environment_ab_runtime_readback.v1",
            "ok": False,
            "status": "FAIL",
            "readback_complete": False,
            "failures": failures,
        }
        conversion = _failure_check(CONVERSION_SCHEMA, failures)

    return {
        "schema_version": "fsm50.environment_ab_comparison.v1",
        "ok": bool(
            triplet.get("ok", False)
            and trajectory.get("ok", False)
            and runtime_readback.get("readback_complete", False)
        ),
        "fail_closed": True,
        "instrumentation_comparison": triplet,
        "trajectory_comparison": trajectory,
        "runtime_readback": runtime_readback,
        "artifact_conversion": conversion,
    }


def generate_environment_equivalence_report(
    *,
    a1_run: str | Path,
    a2_run: str | Path,
    b_run: str | Path,
    output_path: str | Path = DEFAULT_REPORT_PATH,
    fingerprint: Mapping[str, Any] | None = None,
    self_error_multiplier: float = 3.0,
    absolute_floors: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Write and return the complete environment-equivalence report object.

    ``a1_run`` and ``a2_run`` must identify finalized formal runs; ``b_run``
    must identify a finalized instrumented run.  The return value is the exact
    JSON object written to ``output_path`` (schema
    ``fsm50.environment_equivalence_report.v1``), not merely the inner A/B
    comparison object.
    """

    static_fingerprint = dict(fingerprint or build_static_environment_fingerprint())
    comparison = compare_environment_run_artifacts(
        baseline_a1=a1_run,
        baseline_a2=a2_run,
        instrumented_b=b_run,
        self_error_multiplier=self_error_multiplier,
        absolute_floors=absolute_floors,
    )

    write_environment_equivalence_report(
        output_path,
        fingerprint=static_fingerprint,
        instrumentation_comparison=comparison["instrumentation_comparison"],
        trajectory_comparison=comparison["trajectory_comparison"],
        runtime_readback=comparison["runtime_readback"],
        extra={"artifact_conversion": comparison["artifact_conversion"]},
    )
    report = _strict_json(Path(output_path))
    if not isinstance(report, dict):
        raise ArtifactValidationError(f"{output_path}: written report is not an object")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert three finalized replay artifacts into an environment A/A--A/B report."
    )
    parser.add_argument("--a1", required=True, type=Path, help="First baseline run/artifact directory")
    parser.add_argument("--a2", required=True, type=Path, help="Second baseline run/artifact directory")
    parser.add_argument("--b", required=True, type=Path, help="Instrumented run/artifact directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--self-error-multiplier", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_environment_equivalence_report(
        a1_run=args.a1,
        a2_run=args.a2,
        b_run=args.b,
        output_path=args.output,
        self_error_multiplier=args.self_error_multiplier,
    )
    print(
        json.dumps(
            {
                "report": str(Path(args.output).resolve()),
                "status": report.get("status"),
                "environment_equivalent": report.get("environment_equivalent"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArtifactValidationError",
    "COMMON_WHEEL_FORCE_SOURCE",
    "CONVERSION_SCHEMA",
    "DEFAULT_REPORT_PATH",
    "ReplayArtifact",
    "compare_environment_run_artifacts",
    "compare_sealed_trajectory_diagnostics",
    "extract_trajectory_metrics",
    "generate_environment_equivalence_report",
    "load_completed_replay_artifact",
    "load_sealed_trajectory_diagnostic_artifact",
    "main",
    "TRAJECTORY_DIAGNOSTIC_PHYSICAL_CLAIM",
    "TRAJECTORY_DIAGNOSTIC_SCOPE",
    "validate_artifact_triplet",
]
