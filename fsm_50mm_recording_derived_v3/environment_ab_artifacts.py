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


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = MODULE_ROOT / "reports" / "ENVIRONMENT_EQUIVALENCE_REPORT.json"
RESULT_SCHEMA = "fsm50.recording_replay_result.v1"
CONVERSION_SCHEMA = "fsm50.environment_ab_artifacts.v1"
LEGS = ("FL", "FR", "RL", "RR")
CONTACT_CLASSES = frozenset({"GROUND", "FRONT_FACE", "TOP", "AIR", "UNKNOWN"})
COMMON_WHEEL_FORCE_SOURCE = "isaaclab.ContactSensor.net_forces_w"
SHA256_HEX = frozenset("0123456789abcdef")

REQUIRED_RUN_FILENAMES = (
    "result.json",
    "fsm50_telemetry.jsonl",
    "runtime_environment.json",
    "visual_recording_manifest.json",
    "physical_evidence.json",
    "failure_diagnostics.json",
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


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
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
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
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


def _load_batch_shutdown_closure(
    *,
    batch_root: Path,
    artifact_root: Path,
    run_dir: Path,
    result_path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only a normally closed supervised batch with durable evidence."""

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
    versions = request.get("versions")
    source_version = str(result.get("source_version", "") or "")
    if not isinstance(versions, list) or versions.count(source_version) != 1:
        raise ArtifactValidationError(
            "batch_request.versions does not contain the current source version exactly once"
        )

    shutdown = _mapping(
        _strict_json(named_paths["shutdown_outcome"]), "shutdown_outcome.json"
    )
    if str(shutdown.get("schema_version", "")) != "fsm50.shutdown_outcome.v1":
        raise ArtifactValidationError("shutdown_outcome schema is invalid")
    if str(shutdown.get("status", "")) != "NORMAL_EXIT":
        raise ArtifactValidationError(
            f"shutdown_outcome status is not NORMAL_EXIT: {shutdown.get('status')!r}"
        )

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
    if _mapping(finalization.get("shutdown_outcome"), "batch_finalization.shutdown_outcome") != shutdown:
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
    matching = [
        item
        for item in batch_results
        if str(item.get("source_version", "")) == source_version
        and Path(str(item.get("run_dir", ""))).resolve() == run_dir
        and Path(str(item.get("artifact_root", ""))).resolve() == artifact_root
    ]
    if len(matching) != 1 or matching[0] != dict(result):
        raise ArtifactValidationError(
            "batch_results does not contain exactly the current durable result/version/run/artifact"
        )

    preclose_results = _strict_json(named_paths["batch_results_preclose"])
    if preclose_results != batch_results_raw:
        raise ArtifactValidationError(
            "batch_results.preclose.json differs from normal-exit live batch_results.json"
        )
    preclose_finalization = _mapping(
        _strict_json(named_paths["batch_finalization_preclose"]),
        "batch_finalization.preclose.json",
    )
    if (
        Path(str(preclose_finalization.get("artifact_root", ""))).resolve() != batch_root
        or preclose_finalization.get("finalized") is not True
        or preclose_finalization.get("failed") is not False
        or str(preclose_finalization.get("phase", "")) != "PRECLOSE_FINALIZED"
        or str(preclose_finalization.get("close_error", "")) != "PENDING_SIMULATION_CLOSE"
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
    if _mapping(evidence.get("batch_finalization"), "preclose.batch_finalization") != preclose_finalization:
        raise ArtifactValidationError(
            "preclose evidence batch_finalization differs from immutable snapshot"
        )
    batch_source_integrity = _mapping(
        _strict_json(named_paths["batch_source_integrity"]),
        "batch source_integrity.json",
    )
    if batch_source_integrity.get("equal") is not True:
        raise ArtifactValidationError("batch source_integrity.equal is not true")
    if _mapping(evidence.get("source_integrity"), "preclose.source_integrity") != batch_source_integrity:
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
    evidence_rows = {
        str(path.resolve()): _require_evidence_file(
            evidence_files, path, label="preclose_complete"
        )
        for path in stable_preclose_paths
    }

    preclose_manifest = _validate_checksum_manifest(
        batch_root,
        named_paths["checksums_preclose"],
        [
            named_paths["batch_results"],
            named_paths["batch_finalization"],
            named_paths["batch_request"],
            named_paths["batch_source_integrity"],
            result_path,
        ],
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
        "status": "NORMAL_EXIT",
        "phase": "SHUTDOWN_COMPLETE",
        "files": closure_files,
        "closure_sha256": closure_digest,
        "live_checksum_entry_count": len(live_manifest),
        "preclose_checksum_entry_count": len(preclose_manifest),
        "preclose_evidence_files": evidence_rows,
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
    if not bool(result.get("artifact_valid", False)):
        failures.append("result.artifact_valid is not true")
    lifecycle = _mapping(result.get("lifecycle", {}), f"{result_path}: lifecycle")
    if lifecycle.get("finalized") is not True or lifecycle.get("failed") is not False:
        failures.append("result lifecycle is not finalized/non-failed")
    if result.get("scheduler_complete") is not True:
        failures.append("scheduler_complete is not true")
    if str(result.get("scheduler_stop_reason", "")) != "complete":
        failures.append("scheduler_stop_reason is not complete")
    if result.get("timed_out") is True:
        failures.append("result is timed out")
    if result.get("simulation_app_stopped") is True:
        failures.append("simulation app stopped during replay")
    if not bool(_mapping(result.get("source_integrity", {}), "source_integrity").get("ok", False)):
        failures.append("source_integrity.ok is not true")
    if not bool(_mapping(result.get("visualization", {}), "visualization").get("ok", False)):
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
    """Require a real active-viewport MP4; telemetry plots are insufficient."""

    # Recording replay production writes one capture per finalized run.  Do
    # not walk to a batch/ancestor, which could reuse a different run's movie.
    manifest_path = run_dir / "viewport_video_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactValidationError(
            f"{run_dir}: actual viewport_video_manifest.json is missing; "
            "visual_recording_manifest.json with not_camera_video=true is not video evidence"
        )
    manifest = _mapping(_strict_json(manifest_path), str(manifest_path))
    failures: list[str] = []
    if str(manifest.get("schema_version", "")) != "fsm50.recording_viewport_video.v1":
        failures.append("unsupported viewport video schema")
    for flag in ("valid", "artifact_valid", "actual_viewport_video", "capture_requested"):
        if manifest.get(flag) is not True:
            failures.append(f"viewport video {flag} is not true")
    if manifest.get("diagnostic_only") is not False:
        failures.append("diagnostic-only capture cannot establish equivalence")
    if manifest.get("not_camera_video") is not False:
        failures.append("viewport manifest does not identify actual camera video")
    if str(manifest.get("contact_mode", "")) != contact_mode:
        failures.append("viewport contact_mode differs from the replay role")
    if str(manifest.get("source", "")) != "actual_active_isaac_gui_viewport_render_product":
        failures.append("viewport source is not the active Isaac GUI render product")
    if not str(manifest.get("render_product_path", "") or ""):
        failures.append("render_product_path is missing")
    if str(manifest.get("error", "") or ""):
        failures.append("viewport recorder reports an error")
    try:
        if _integer(manifest.get("frame_count"), "viewport.frame_count") < 2:
            failures.append("viewport video has fewer than two frames")
        if _finite(manifest.get("fps"), "viewport.fps") <= 0.0:
            failures.append("viewport fps must be positive")
    except ArtifactValidationError as exc:
        failures.extend(exc.failures)
    video_text = str(manifest.get("video_path", "") or "")
    video_path = Path(video_text).resolve() if video_text else Path()
    if not video_text or not video_path.is_file() or video_path.stat().st_size <= 0:
        failures.append("viewport MP4 is missing or empty")
    elif not _is_within(video_path, manifest_path.parent):
        failures.append("viewport MP4 escapes its manifest directory")
    elif video_path.suffix.lower() != ".mp4":
        failures.append("viewport video is not an MP4 file")
    else:
        try:
            with video_path.open("rb") as stream:
                header = stream.read(32)
            if len(header) < 12 or header[4:8] != b"ftyp":
                failures.append("viewport MP4 lacks an ISO base-media ftyp header")
            expected_sha = _sha256(manifest.get("video_sha256"), "viewport.video_sha256")
            if sha256_file(video_path).lower() != expected_sha:
                failures.append("viewport MP4 SHA-256 mismatch")
        except OSError as exc:
            failures.append(f"viewport MP4 is unreadable: {exc}")
        except ArtifactValidationError as exc:
            failures.extend(exc.failures)
    if failures:
        raise ArtifactValidationError([f"{manifest_path}: {failure}" for failure in failures])
    actual_video_sha = sha256_file(video_path)
    manifest_sha = sha256_file(manifest_path)
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
        raise ArtifactValidationError([f"{manifest_path}: {failure}" for failure in failures])
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "video_path": str(video_path),
        "video_sha256": actual_video_sha,
        "frame_count": int(manifest["frame_count"]),
        "fps": float(manifest["fps"]),
        "render_product_path": str(manifest["render_product_path"]),
        "source": str(manifest["source"]),
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
                "filtered_contact_available",
                "collision_evidence_valid",
            ):
                if row.get(flag) is not True:
                    raise ArtifactValidationError(f"{row_label}.{flag} is not true")
            if str(row.get("filtered_contact_error", "") or ""):
                raise ArtifactValidationError(f"{row_label}.filtered_contact_error is not empty")
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
        Path(viewport_video["video_path"]),
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
        "viewport_video": viewport_video,
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
    "extract_trajectory_metrics",
    "generate_environment_equivalence_report",
    "load_completed_replay_artifact",
    "main",
    "validate_artifact_triplet",
]
