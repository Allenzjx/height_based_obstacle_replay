"""Isaac-facing orchestration for the event-gated 50 mm controller.

This module is imported only by the supervised child after SimulationApp is
allowed to start.  The scene, adapter, grounding, and telemetry construction
match the production Replay Fast path; controller logic remains in the pure
Python modules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .fsm50_controller import ControllerStatus, FSM50Controller
from .fsm50_observation import FSM50Observation
from .fsm50_state_model import FSM50StateTable
from .fsm50_telemetry import FSM50TelemetryCollector
from .filtered_wheel_contact import make_filtered_wheel_contact_sensor_factory
from .nonwheel_obstacle_contact import configure_scene_for_wheel_and_nonwheel_contacts
from .shutdown_contract import validate_shutdown_outcome


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Read a report without accepting duplicate keys or non-finite values."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"environment A/B report is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("environment A/B report must contain an object")
    return payload


def _gate_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"environment A/B report is missing object {label}")
    return dict(value)


def _canonical_gate_value(value: Any, label: str) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"environment A/B report has non-canonical {label}: {exc}"
        ) from exc


def _reject_pending_gate_value(value: Any, label: str) -> None:
    """Reject pending placeholders in the required dynamic gate evidence."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_pending_gate_value(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_pending_gate_value(item, f"{label}[{index}]")
        return
    if isinstance(value, str):
        normalized = value.strip().upper()
        # Do not treat a filesystem path whose directory happens to contain
        # "pending" as a placeholder.  Runtime-version/status placeholders are
        # scalar tokens such as PENDING_* or unknown_pending_runtime_readback.
        if (
            "PENDING" in normalized
            and "/" not in normalized
            and "\\" not in normalized
        ):
            raise RuntimeError(
                f"environment A/B report contains pending value at {label}: {value!r}"
            )


def _validated_checksum_rows(
    root: Path,
    manifest_name: str = "checksums.sha256",
) -> dict[str, str]:
    checksum_path = root / manifest_name
    if not checksum_path.is_file():
        raise RuntimeError(f"batch shutdown closure is missing {checksum_path}")
    rows: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"batch checksum manifest is unreadable: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        digest, separator, relative = raw.partition("  ")
        digest = digest.strip().lower()
        relative = relative.strip().replace("\\", "/")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in rows
        ):
            raise RuntimeError(
                f"invalid batch checksum row {checksum_path}:{line_number}"
            )
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"batch checksum path escapes closure root: {relative}"
            ) from exc
        rows[relative] = digest
    return rows


def _require_current_checksum(
    root: Path,
    checksum_rows: Mapping[str, str],
    path: Path,
) -> None:
    if not path.is_file():
        raise RuntimeError(f"batch shutdown closure file is missing: {path}")
    relative = path.resolve().relative_to(root).as_posix()
    if checksum_rows.get(relative) != _sha256(path):
        raise RuntimeError(
            f"batch shutdown closure checksum is missing or stale for {relative}"
        )


def _find_batch_shutdown_root(artifact_root: Path) -> Path:
    closure_names = (
        "shutdown_outcome.json",
        "batch_finalization.json",
        "preclose_complete.json",
        "checksums.preclose.sha256",
    )
    for candidate in artifact_root.resolve().parents:
        if any((candidate / name).exists() for name in closure_names):
            return candidate
    raise RuntimeError(
        f"no batch shutdown closure contains artifact {artifact_root.resolve()}"
    )


def _current_git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"current git HEAD is unavailable: {exc}") from exc
    head = str(completed.stdout or "").strip().lower()
    if completed.returncode != 0 or not head:
        detail = str(completed.stderr or "").strip()
        raise RuntimeError(f"current git HEAD is unavailable: {detail}")
    return head


def _validate_live_source_closure(provenance: Mapping[str, Any], role: str) -> None:
    """Bind a historical PASS to the source files and git HEAD used now."""

    closure = _gate_mapping(
        provenance.get("source_closure"), f"{role}.provenance.source_closure"
    )
    files = _gate_mapping(
        closure.get("files"), f"{role}.provenance.source_closure.files"
    )
    if not files:
        raise RuntimeError(f"{role} source closure has no files")
    for source_name, raw in files.items():
        source_path = Path(str(source_name))
        if not source_path.is_absolute():
            raise RuntimeError(
                f"{role} source closure path is not absolute: {source_name}"
            )
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise RuntimeError(
                f"{role} current source closure file is missing: {source_path}"
            )
        embedded = _gate_mapping(
            raw, f"{role}.provenance.source_closure.files[{source_name!r}]"
        )
        expected_sha = str(embedded.get("sha256", "") or "").strip().lower()
        expected_size = embedded.get("size_bytes")
        if (
            len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
        ):
            raise RuntimeError(f"{role} source closure metadata is invalid: {source_name}")
        if source_path.stat().st_size != expected_size or _sha256(source_path) != expected_sha:
            raise RuntimeError(
                f"{role} current source file differs from the reported closure: {source_path}"
            )

    expected_head = str(provenance.get("source_git_head", "") or "").strip().lower()
    closure_head = str(closure.get("git_head", "") or "").strip().lower()
    current_head = _current_git_head()
    if not expected_head or expected_head != closure_head or current_head != expected_head:
        raise RuntimeError(
            f"{role} current git HEAD differs from reported source_git_head: "
            f"expected={expected_head!r} current={current_head!r}"
        )


def _validate_batch_shutdown_closure(artifact_root: Path) -> Path:
    """Revalidate parent-owned normal shutdown and immutable preclose evidence."""

    artifact_root = Path(artifact_root).resolve()
    batch_root = _find_batch_shutdown_root(artifact_root)
    if (
        not (batch_root / ".finalized").is_file()
        or (batch_root / ".partial").exists()
        or (batch_root / ".failed").exists()
    ):
        raise RuntimeError(
            f"batch shutdown closure marker is not finalized: {batch_root}"
        )

    shutdown_path = batch_root / "shutdown_outcome.json"
    finalization_path = batch_root / "batch_finalization.json"
    results_path = batch_root / "batch_results.json"
    preclose_path = batch_root / "preclose_complete.json"
    snapshot_results_path = batch_root / "batch_results.preclose.json"
    snapshot_finalization_path = batch_root / "batch_finalization.preclose.json"
    snapshot_checksums_path = batch_root / "checksums.preclose.sha256"
    shutdown = _strict_json_object(shutdown_path)
    finalization = _strict_json_object(finalization_path)
    preclose = _strict_json_object(preclose_path)

    try:
        validate_shutdown_outcome(shutdown)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "batch shutdown outcome is not a verified graceful/fast exit: "
            f"{shutdown_path}: {exc}"
        ) from exc
    if (
        finalization.get("finalized") is not True
        or finalization.get("failed") is not False
        or finalization.get("phase") != "SHUTDOWN_COMPLETE"
        or str(finalization.get("close_error", "") or "")
        or str(finalization.get("artifact_root", "") or "") != str(batch_root)
    ):
        raise RuntimeError(
            f"batch finalization is not finalized SHUTDOWN_COMPLETE: {finalization_path}"
        )
    embedded_shutdown = _gate_mapping(
        finalization.get("shutdown_outcome"),
        "batch_finalization.shutdown_outcome",
    )
    if _canonical_gate_value(
        embedded_shutdown, "batch_finalization.shutdown_outcome"
    ) != _canonical_gate_value(shutdown, "shutdown_outcome"):
        raise RuntimeError(
            "batch_finalization.shutdown_outcome differs from shutdown_outcome.json"
        )

    if (
        preclose.get("schema_version") != "fsm50.preclose_complete.v1"
        or str(preclose.get("batch_root", "") or "") != str(batch_root)
        or not str(preclose.get("created_utc", "") or "")
    ):
        raise RuntimeError(f"invalid preclose_complete marker: {preclose_path}")
    evidence = _gate_mapping(preclose.get("evidence"), "preclose_complete.evidence")
    if any(
        evidence.get(key)
        for key in (
            "manifest_error",
            "immutable_preclose_errors",
            "batch_marker_error",
        )
    ):
        raise RuntimeError("preclose_complete contains closure errors")
    try:
        physics_result_count = int(evidence.get("physics_result_count", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("preclose_complete physics_result_count is invalid") from exc
    if physics_result_count <= 0:
        raise RuntimeError("preclose_complete has no physics results")
    source_integrity = _gate_mapping(
        evidence.get("source_integrity"), "preclose_complete.evidence.source_integrity"
    )
    if source_integrity.get("equal") is not True:
        raise RuntimeError("preclose_complete source integrity is not equal")
    preclose_finalization = _gate_mapping(
        evidence.get("batch_finalization"),
        "preclose_complete.evidence.batch_finalization",
    )
    if (
        preclose_finalization.get("finalized") is not True
        or preclose_finalization.get("failed") is not False
        or preclose_finalization.get("phase") != "PRECLOSE_FINALIZED"
    ):
        raise RuntimeError("preclose_complete batch finalization is invalid")
    evidence_files = _gate_mapping(
        evidence.get("evidence_files"), "preclose_complete.evidence.evidence_files"
    )

    for snapshot_path in (
        snapshot_results_path,
        snapshot_finalization_path,
        snapshot_checksums_path,
    ):
        embedded = _gate_mapping(
            evidence_files.get(str(snapshot_path.resolve())),
            f"preclose evidence file {snapshot_path.name}",
        )
        if not snapshot_path.is_file():
            raise RuntimeError(f"immutable preclose snapshot is missing: {snapshot_path}")
        if (
            str(embedded.get("sha256", "") or "").lower()
            != _sha256(snapshot_path)
            or embedded.get("size_bytes") != snapshot_path.stat().st_size
        ):
            raise RuntimeError(
                f"immutable preclose snapshot differs from marker: {snapshot_path}"
            )
    snapshot_finalization = _strict_json_object(snapshot_finalization_path)
    if _canonical_gate_value(
        snapshot_finalization, "batch_finalization.preclose"
    ) != _canonical_gate_value(
        preclose_finalization, "preclose embedded batch_finalization"
    ):
        raise RuntimeError(
            "batch_finalization.preclose.json differs from preclose marker"
        )

    preclose_checksum_rows = _validated_checksum_rows(
        batch_root, "checksums.preclose.sha256"
    )
    for original_name, snapshot_path in (
        ("batch_results.json", snapshot_results_path),
        ("batch_finalization.json", snapshot_finalization_path),
    ):
        if preclose_checksum_rows.get(original_name) != _sha256(snapshot_path):
            raise RuntimeError(
                f"preclose checksum for {original_name} differs from immutable snapshot"
            )

    checksum_rows = _validated_checksum_rows(batch_root)
    for closure_path in (
        shutdown_path,
        finalization_path,
        results_path,
        preclose_path,
        snapshot_results_path,
        snapshot_finalization_path,
        snapshot_checksums_path,
    ):
        _require_current_checksum(batch_root, checksum_rows, closure_path)
    return batch_root


def _load_environment_gate(path: Path) -> dict[str, Any]:
    """Load and revalidate the durable A1/A2/B evidence before SimulationApp."""

    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise RuntimeError(f"environment A/B report is missing: {report_path}")
    payload = _strict_json_object(report_path)
    if payload.get("schema_version") != "fsm50.environment_equivalence_report.v1":
        raise RuntimeError("environment A/B report schema is missing or unsupported")
    if not str(payload.get("created_utc", "") or ""):
        raise RuntimeError("environment A/B report created_utc is missing")
    _gate_mapping(payload.get("static_fingerprint"), "static_fingerprint")
    if (
        payload.get("environment_equivalent") is not True
        or payload.get("status") != "PASS"
    ):
        raise RuntimeError(
            "full/state FSM is blocked until environment A/A-B status is PASS"
        )
    _reject_pending_gate_value(payload, "report")

    instrumentation = _gate_mapping(
        payload.get("instrumentation_comparison"),
        "instrumentation_comparison",
    )
    trajectory = _gate_mapping(
        payload.get("trajectory_comparison"),
        "trajectory_comparison",
    )
    runtime_readback = _gate_mapping(
        payload.get("runtime_readback"),
        "runtime_readback",
    )
    extra = _gate_mapping(payload.get("extra"), "extra")
    conversion = _gate_mapping(
        extra.get("artifact_conversion"),
        "extra.artifact_conversion",
    )
    for label, check, schema in (
        (
            "instrumentation_comparison",
            instrumentation,
            "fsm50.environment_ab_triplet_validation.v1",
        ),
        (
            "trajectory_comparison",
            trajectory,
            "fsm50.trajectory_equivalence.v1",
        ),
        (
            "runtime_readback",
            runtime_readback,
            "fsm50.environment_ab_runtime_readback.v1",
        ),
        (
            "extra.artifact_conversion",
            conversion,
            "fsm50.environment_ab_artifacts.v1",
        ),
    ):
        if check.get("schema_version") != schema:
            raise RuntimeError(
                f"environment A/B report {label}.schema_version is unsupported"
            )
        if check.get("ok") is not True:
            raise RuntimeError(f"environment A/B report {label}.ok is not true")
        _reject_pending_gate_value(check, label)
    if runtime_readback.get("readback_complete") is not True:
        raise RuntimeError(
            "environment A/B report runtime_readback.readback_complete is not true"
        )

    runs = _gate_mapping(runtime_readback.get("runs"), "runtime_readback.runs")
    required_roles = ("A1", "A2", "B")
    if set(runs) != set(required_roles):
        raise RuntimeError(
            "environment A/B report runtime_readback.runs must contain exactly A1, A2, and B"
        )

    try:
        from .environment_ab_artifacts import (
            compare_environment_run_artifacts,
            load_completed_replay_artifact,
            validate_artifact_triplet,
        )

        loaded = []
        artifact_paths: dict[str, str] = {}
        for role in required_roles:
            embedded = _gate_mapping(
                runs.get(role), f"runtime_readback.runs.{role}"
            )
            artifact_root = str(embedded.get("artifact_root", "") or "")
            run_dir = str(embedded.get("run_dir", "") or "")
            provenance = _gate_mapping(
                embedded.get("provenance"),
                f"runtime_readback.runs.{role}.provenance",
            )
            if not artifact_root or not run_dir:
                raise RuntimeError(
                    f"runtime_readback.runs.{role} is missing artifact_root or run_dir"
                )
            for required_key in (
                "source_closure",
                "batch_root",
                "batch_shutdown_status",
                "batch_finalization_phase",
                "batch_shutdown_closure",
                "artifact_hashes",
                "checksums_sha256",
                "metrics_sha256",
                "viewport_video",
            ):
                if required_key not in provenance:
                    raise RuntimeError(
                        f"runtime_readback.runs.{role}.provenance is missing {required_key}"
                    )
            artifact = load_completed_replay_artifact(artifact_root, role=role)
            if str(artifact.artifact_root) != artifact_root:
                raise RuntimeError(
                    f"runtime_readback.runs.{role}.artifact_root differs from the reloaded artifact"
                )
            if str(artifact.run_dir) != run_dir:
                raise RuntimeError(
                    f"runtime_readback.runs.{role}.run_dir differs from the reloaded artifact"
                )
            if _canonical_gate_value(
                artifact.provenance, f"reloaded {role} provenance"
            ) != _canonical_gate_value(provenance, f"embedded {role} provenance"):
                raise RuntimeError(
                    f"runtime_readback.runs.{role}.provenance differs from current disk evidence"
                )
            _validate_live_source_closure(artifact.provenance, role)
            loaded.append(artifact)
            artifact_paths[role] = artifact_root

        validated_batch_roots: set[Path] = set()
        for artifact in loaded:
            batch_root = _find_batch_shutdown_root(artifact.artifact_root)
            if batch_root not in validated_batch_roots:
                _validate_batch_shutdown_closure(artifact.artifact_root)
                validated_batch_roots.add(batch_root)

        revalidated_triplet = validate_artifact_triplet(*loaded)
        if revalidated_triplet.get("ok") is not True:
            raise RuntimeError(
                "current A1/A2/B artifacts fail triplet validation: "
                + "; ".join(
                    str(item)
                    for item in revalidated_triplet.get("failures", [])
                )
            )
        if _canonical_gate_value(
            revalidated_triplet, "revalidated instrumentation comparison"
        ) != _canonical_gate_value(
            instrumentation, "embedded instrumentation comparison"
        ):
            raise RuntimeError(
                "instrumentation_comparison differs from current A1/A2/B artifacts"
            )
        embedded_triplet = _gate_mapping(
            runtime_readback.get("triplet_validation"),
            "runtime_readback.triplet_validation",
        )
        if _canonical_gate_value(
            embedded_triplet, "runtime_readback.triplet_validation"
        ) != _canonical_gate_value(
            revalidated_triplet, "revalidated triplet"
        ):
            raise RuntimeError(
                "runtime_readback.triplet_validation differs from current artifacts"
            )

        trajectory_metrics = _gate_mapping(
            trajectory.get("metrics"), "trajectory_comparison.metrics"
        )
        absolute_floors: dict[str, float] = {}
        for metric, raw_metric in trajectory_metrics.items():
            metric_row = _gate_mapping(
                raw_metric, f"trajectory_comparison.metrics.{metric}"
            )
            if "absolute_floor" not in metric_row:
                raise RuntimeError(
                    f"trajectory_comparison.metrics.{metric}.absolute_floor is missing"
                )
            absolute_floors[str(metric)] = float(metric_row["absolute_floor"])
        multiplier = float(trajectory["self_error_multiplier"])
        recomputed = compare_environment_run_artifacts(
            baseline_a1=artifact_paths["A1"],
            baseline_a2=artifact_paths["A2"],
            instrumented_b=artifact_paths["B"],
            self_error_multiplier=multiplier,
            absolute_floors=absolute_floors,
        )
        if recomputed.get("ok") is not True:
            raise RuntimeError("current A1/A2/B artifact comparison is not ok")
        embedded_checks = {
            "instrumentation_comparison": instrumentation,
            "trajectory_comparison": trajectory,
            "runtime_readback": runtime_readback,
            "artifact_conversion": conversion,
        }
        for key, embedded in embedded_checks.items():
            current = recomputed.get(key)
            if _canonical_gate_value(
                current, f"recomputed {key}"
            ) != _canonical_gate_value(embedded, f"embedded {key}"):
                raise RuntimeError(
                    f"environment A/B report {key} differs from current disk artifacts"
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"environment A/B artifact revalidation failed: {type(exc).__name__}: {exc}"
        ) from exc
    return payload


class ViewportVideoRecorder:
    """Capture the actual active viewport render product and encode an MP4."""

    def __init__(self, root: Path, *, enabled: bool, fps: float) -> None:
        self.root = root
        self.frames = root / "viewport_frames"
        self.video_path = root / "fsm50_viewport.mp4"
        self.enabled = bool(enabled)
        self.fps = max(1.0, float(fps))
        self.error = ""
        self.render_product_path = ""
        self.started = False

    def start(self) -> None:
        if not self.enabled:
            self.error = "viewport capture disabled (headless or --no-video)"
            return
        try:
            self.frames.mkdir(parents=True, exist_ok=False)
            from omni.kit.viewport.utility import get_active_viewport  # type: ignore
            import isaacsim.kit.scripts.movie_capture as movie_capture  # type: ignore

            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("active GUI viewport is unavailable")
            self.render_product_path = str(viewport.render_product_path)
            movie_capture.basePath = str(self.frames.resolve()).replace("\\", "/")
            # An explicit fileName makes Isaac 5.1 overwrite the same PNG on
            # every render.  Empty delegates naming to AOV_FrameNumber so the
            # capture contains distinct viewport frames.
            movie_capture.baseFilename = ""
            movie_capture.inflightFileIO = 4
            movie_capture.attach_post_process_save_to_disk()
            self.started = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def finalize(self) -> dict[str, Any]:
        if self.started:
            try:
                import omni.kit.renderer_capture  # type: ignore

                omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
            except Exception as exc:
                self.error = self.error or f"capture flush failed: {type(exc).__name__}: {exc}"
        frames = sorted(self.frames.rglob("*.png")) if self.frames.is_dir() else []
        if len(frames) >= 2:
            try:
                import imageio.v2 as imageio  # type: ignore

                writer = imageio.get_writer(
                    self.video_path,
                    fps=self.fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=None,
                )
                try:
                    for frame in frames:
                        writer.append_data(imageio.imread(frame))
                finally:
                    writer.close()
            except Exception as exc:
                self.error = self.error or f"video encode failed: {type(exc).__name__}: {exc}"
        valid = bool(
            len(frames) >= 2
            and self.video_path.is_file()
            and self.video_path.stat().st_size > 0
            and not self.error
        )
        manifest = {
            "schema_version": "fsm50.viewport_video.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "valid": valid,
            "not_camera_video": False,
            "source": "actual_active_isaac_gui_viewport_render_product",
            "render_product_path": self.render_product_path,
            "frame_directory": str(self.frames),
            "frame_count": len(frames),
            "fps": self.fps,
            "video_path": str(self.video_path),
            "video_sha256": _sha256(self.video_path) if valid else "",
            "error": self.error,
        }
        _write_json(self.root / "viewport_video_manifest.json", manifest)
        return manifest


def _observation_from_latest(
    collector: FSM50TelemetryCollector,
    controller: FSM50Controller | None,
) -> FSM50Observation:
    if not collector.fsm50_rows:
        return FSM50Observation.from_mapping({}, strict=False)
    row = dict(collector.fsm50_rows[-1])
    if (
        controller is not None
        and controller.state is not None
        and controller.state.target_com_leg is not None
    ):
        row["target_com_leg"] = controller.state.target_com_leg.value
    return FSM50Observation.from_mapping(row, strict=False)


def _state_snapshot(
    adapter: Any,
    *,
    state_id: str,
    environment_fingerprint_sha256: str,
    run_dir: Path,
) -> Path:
    destination = run_dir / "state_snapshots" / state_id / "sim_state_before.json"
    envelope = {
        "schema_version": "fsm50.trusted_sim_state_before.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state_id": state_id,
        "environment_fingerprint_sha256": environment_fingerprint_sha256,
        "source_run_directory": str(run_dir),
        "source_result_path": "PENDING_FINAL_RESULT",
        "source_result_sha256": "PENDING_FINAL_RESULT",
        "sim_state_before": _jsonable(adapter.capture_sim_state()),
    }
    _write_json(destination, envelope)
    return destination


def _finalize_state_snapshots(
    run_dir: Path,
    *,
    result_path: Path,
    result_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "state_snapshots").glob("*/sim_state_before.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_result_path"] = str(result_path)
        payload["source_result_sha256"] = result_sha256
        _write_json(path, payload)
        rows.append(
            {
                "state_id": str(payload.get("state_id", "")),
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    return rows


def _refresh_restore_artifact_hashes(run_dir: Path, result_path: Path) -> None:
    """Bind restore envelopes to the immutable final result bytes."""

    result_sha256 = _sha256(result_path)
    snapshots = _finalize_state_snapshots(
        run_dir,
        result_path=result_path,
        result_sha256=result_sha256,
    )
    prefix_path = run_dir / "verified_prefix_replay_manifest.json"
    prefix_sha256 = ""
    if prefix_path.is_file():
        prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
        prefix["source_result_path"] = str(result_path)
        prefix["source_result_sha256"] = result_sha256
        _write_json(prefix_path, prefix)
        prefix_sha256 = _sha256(prefix_path)
    _write_json(
        run_dir / "state_snapshot_index.json",
        {
            "schema_version": "fsm50.state_snapshot_index.v1",
            "source_result_path": str(result_path),
            "source_result_sha256": result_sha256,
            "prefix_manifest_path": str(prefix_path),
            "prefix_manifest_sha256": prefix_sha256,
            "snapshots": snapshots,
        },
    )


def _write_controller_timeline(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(run_dir / "fsm_controller_timeline.json", rows)
    flat: list[dict[str, Any]] = []
    for row in rows:
        flat.append(
            {
                "event_index": row.get("event_index"),
                "event": row.get("event"),
                "state_id": row.get("state_id"),
                "time_s": row.get("time_s"),
                "outcome": row.get("outcome", ""),
                "reason": row.get("reason", row.get("transition_reason", "")),
                "retry_index": row.get("retry_index", 0),
                "payload_json": json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True),
            }
        )
    fields = (
        "event_index",
        "event",
        "state_id",
        "time_s",
        "outcome",
        "reason",
        "retry_index",
        "payload_json",
    )
    with (run_dir / "fsm_controller_timeline.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _safe_state_test_criteria(physical: Mapping[str, Any]) -> dict[str, bool]:
    criteria = dict(physical.get("strict_criteria", {}) or {})
    return {
        "evidence_complete": bool(physical.get("evidence_complete", False)),
        "contact_evidence_valid": bool(criteria.get("contact_evidence_valid", False)),
        "no_illegal_drive_up": bool(criteria.get("no_illegal_drive_up", False)),
        "attitude_safe": bool(criteria.get("attitude_safe", False)),
        "joint_limits_safe": bool(criteria.get("joint_limits_safe", False)),
        "collision_safe": bool(criteria.get("collision_safe", False)),
        "contact_drift_safe": bool(criteria.get("contact_drift_safe", False)),
    }


def _run_one_controller_episode(
    *,
    run_index: int,
    args: Any,
    adapter: Any,
    scene_handle: Any,
    batch_root: Path,
    environment_gate: Mapping[str, Any],
    environment_gate_sha256: str,
    environment_lock: Mapping[str, Any],
    robot_usd: Path,
    video_manifest_path: Path,
) -> dict[str, Any]:
    from sim_obstacle_scene import measure_obstacle_geometry, measure_scene_baseline
    from sim_state_validation import verify_restored_full_sim_pose
    from sim_worker_runtime import handle_respawn
    from . import run_fsm50 as runner

    mode = str(args.command)
    run_root = runner._new_directory(
        batch_root / "runs", f"{mode}_run_{run_index:02d}"
    )
    partial = run_root / ".partial"
    partial.write_text("running\n", encoding="utf-8")
    collector: FSM50TelemetryCollector | None = None
    controller: FSM50Controller | None = None
    result: dict[str, Any] = {}
    try:
        adapter.attach_telemetry(None)
        respawn = handle_respawn(adapter=adapter)
        if not bool(respawn.get("ok", False)):
            raise RuntimeError(str(respawn.get("error", "clean respawn failed")))
        sensor = getattr(scene_handle, "contact_sensor", None)
        if sensor is None or str(getattr(scene_handle, "contact_sensor_error", "") or ""):
            raise RuntimeError("instrumented contact sensor is unavailable")
        sensor.reset()

        restore_bundle = getattr(args, "restore_bundle", None)
        restore_provenance: Mapping[str, Any] | None = None
        start_state_id = "A0_RESET_AND_SETTLE"
        target_state_id = str(getattr(args, "state_id", "") or "")
        prefix_mode = bool(getattr(args, "replay_prefix", False))
        restore_result: dict[str, Any] | None = None
        if mode == "test-state" and target_state_id != "A0_RESET_AND_SETTLE":
            if restore_bundle is not None:
                restore_values = dict(restore_bundle)
                restore_provenance = dict(
                    restore_values.get("restore_provenance", {}) or {}
                )
                restore_method = str(
                    restore_provenance.get("method", "")
                ).upper()
                if restore_method == "VERIFIED_PREFIX_REPLAY":
                    prefix_mode = True
                    start_state_id = "A0_RESET_AND_SETTLE"
                else:
                    sim_state = restore_values.get("sim_state")
                    if not isinstance(sim_state, Mapping):
                        raise RuntimeError("trusted restore bundle has no sim_state")
                    expected_state = dict(sim_state)
                    restore_result = dict(
                        adapter.restore_sim_state(expected_state) or {}
                    )
                    # Match the production worker's restore boundary: saved
                    # pose/velocities are restored, wheel commands are made
                    # safe, one physics tick is completed, and the measured
                    # pose must verify before control is allowed to start.
                    adapter.stop_wheels()
                    adapter.apply_commands_to_robot()
                    adapter.robot.write_data_to_sim()
                    adapter.step()
                    measured_state = dict(adapter.capture_sim_state() or {})
                    verification = verify_restored_full_sim_pose(
                        expected_state,
                        measured_state,
                        list(getattr(adapter.robot, "joint_names", []) or []),
                    )
                    restore_result["verification"] = verification
                    restore_result["measured_state"] = measured_state
                    restore_result["safe_wheel_boundary"] = True
                    if not verification.get("verified", False):
                        raise RuntimeError(
                            "post-restore pose verification failed: "
                            + str(verification.get("reason", "unknown error"))
                        )
                    sensor.reset()
                    start_state_id = target_state_id
            elif prefix_mode:
                start_state_id = "A0_RESET_AND_SETTLE"
            else:
                raise RuntimeError(
                    "non-A0 test-state requires trusted restore or --replay-prefix"
                )

        live_baseline = measure_scene_baseline(scene_handle, adapter)
        live_obstacle = measure_obstacle_geometry(scene_handle)
        motion = runner.load_motion_reference()
        runtime_environment = runner._runtime_environment_equivalence(
            lock=dict(environment_lock),
            scene_config=scene_handle.config,
            live_baseline=live_baseline,
            live_obstacle=live_obstacle,
            motion=motion,
            robot_usd=robot_usd,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )
        if not runtime_environment.get("ok", False):
            raise RuntimeError("per-run runtime environment lock comparison failed")

        _write_json(
            run_root / "runtime_environment.pre_collector.json",
            {
                "runtime": runner._runtime_versions(),
                "scene_config": asdict(scene_handle.config),
                "live_scene_baseline": live_baseline,
                "live_obstacle_geometry": live_obstacle,
                "environment_equivalence": runtime_environment,
                "physics_dt_s": float(scene_handle.sim.get_physics_dt()),
                "render_interval": int(scene_handle.config.render_interval),
                "contact_mode": "instrumented",
                "contact_sensor_type": type(scene_handle.contact_sensor).__name__,
                "contact_sensor_error": str(scene_handle.contact_sensor_error or ""),
            },
        )

        config = runner._telemetry_config(run_root, float(args.telemetry_rate))
        obstacle = runner._surface_obstacle(live_baseline)
        wheel_radius = float(
            live_baseline.get("wheel_radius_m") or 0.04998999834060672
        )
        collector = FSM50TelemetryCollector(
            config,
            args=SimpleNamespace(
                headless=bool(args.headless), output_dir=str(run_root)
            ),
            scene_handle=scene_handle,
            obstacle=obstacle,
            wheel_radius_m=wheel_radius,
            source_version="fsm50_event_controller",
            plan=None,
            plan_rows=[],
            force_threshold_n=2.0,
            unload_force_n=1.0,
            load_confirm_force_n=2.0,
            top_load_dwell_s=0.10,
            loaded_front_face_rotation_limit_rad=0.15,
            wheel_forward_sign=runner._wheel_forward_sign_by_leg(),
        )
        collector.start_episode(
            adapter=adapter,
            scene_handle=scene_handle,
            obstacle_height_cm=5,
            obstacle_height_m=0.05,
            sequence_label=f"fsm50_{mode}_{run_index:02d}",
            source="fsm50_event_controller",
        )
        run_dir = collector.run_dir or run_root
        adapter.attach_telemetry(collector)
        table = FSM50StateTable.load(Path(args.config))
        controller = FSM50Controller(
            adapter,
            table,
            start_state_id=start_state_id,
            restore_provenance=restore_provenance,
            require_physically_verified=mode in {"run-fsm", "validate-5"},
        )
        deadline = float(adapter.sim_time) + float(args.timeout_s)
        target_entered = False
        target_completed = False
        snapshot_states: set[str] = set()
        last_controller_result: Any = None

        while float(adapter.sim_time) <= deadline:
            if not bool(scene_handle.app_is_running()):
                raise RuntimeError("SimulationApp stopped during FSM")
            collector.set_runtime_context(
                fsm_state=controller.current_state_id or "FSM_PRESTART",
                scheduler_phase="EVENT_GATED_CONTROLLER",
                source_step=None,
                segment_index=None,
            )
            adapter.step()
            observation = _observation_from_latest(collector, controller)
            last_controller_result = controller.update(observation)
            current_state = controller.current_state_id
            if current_state and current_state not in snapshot_states:
                _state_snapshot(
                    adapter,
                    state_id=current_state,
                    environment_fingerprint_sha256=environment_gate_sha256,
                    run_dir=run_dir,
                )
                snapshot_states.add(current_state)
            if target_state_id and current_state == target_state_id:
                target_entered = True
            if (
                mode == "test-state"
                and target_entered
                and last_controller_result.transitioned
                and last_controller_result.previous_state_id == target_state_id
                and current_state != target_state_id
            ):
                target_completed = True
                break
            if controller.status in {ControllerStatus.SUCCEEDED, ControllerStatus.SAFE_STOP}:
                break
        else:
            raise RuntimeError("FSM global simulation-time deadline exceeded")

        settle_target = float(adapter.sim_time) + max(
            0.0, float(args.post_run_settle_s)
        )
        while (
            float(adapter.sim_time) + 1.0e-9 < settle_target
            and bool(scene_handle.app_is_running())
        ):
            collector.set_runtime_context(
                fsm_state=controller.current_state_id,
                scheduler_phase="FSM_POST_RESULT_SETTLE",
                source_step=None,
                segment_index=None,
            )
            adapter.step()
            if controller.status == ControllerStatus.SAFE_STOP:
                controller.update(_observation_from_latest(collector, controller))

        controller_success = controller.status == ControllerStatus.SUCCEEDED
        state_test_success = bool(
            mode == "test-state"
            and target_completed
            and controller.status != ControllerStatus.SAFE_STOP
        )
        collector.finish_episode(
            success=controller_success or state_test_success,
            reason=(
                "strict controller terminal guard satisfied"
                if controller_success
                else "target state physical exit guard satisfied"
                if state_test_success
                else controller.safe_stop_reason or "controller incomplete"
            ),
        )
        run_dir = collector.run_dir or run_root
        physical = collector.physical_evidence()
        state_criteria = _safe_state_test_criteria(physical)
        state_physical_success = bool(
            state_test_success and state_criteria and all(state_criteria.values())
        )
        full_physical_success = bool(
            controller_success and physical.get("physical_success", False)
        )
        _write_controller_timeline(run_dir, controller.timeline)
        runtime_environment_payload = {
            "runtime": runner._runtime_versions(),
            "scene_config": asdict(scene_handle.config),
            "live_scene_baseline": live_baseline,
            "live_obstacle_geometry": live_obstacle,
            "environment_equivalence": runtime_environment,
            "environment_gate_path": str(args.environment_report),
            "environment_fingerprint_sha256": environment_gate_sha256,
            "physics_dt_s": float(scene_handle.sim.get_physics_dt()),
            "render_interval": int(scene_handle.config.render_interval),
            "contact_mode": "instrumented",
            "contact_sensor_type": type(scene_handle.contact_sensor).__name__,
            "contact_sensor_error": str(scene_handle.contact_sensor_error or ""),
        }
        _write_json(run_dir / "runtime_environment.json", runtime_environment_payload)
        prefix_completed = [
            state_id
            for state_id in controller.completed_state_ids
            if state_id != "SAFE_STOP"
        ]
        prefix_manifest = {
            "schema_version": "fsm50.verified_prefix_replay_manifest.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "verified": bool(controller.status != ControllerStatus.SAFE_STOP),
            # A direct mid-graph restore can verify the requested state, but
            # it did not execute the earlier states in this episode and must
            # never be advertised as a verified live prefix.
            "prefix_complete": bool(
                start_state_id == "A0_RESET_AND_SETTLE"
                and controller.status != ControllerStatus.SAFE_STOP
            ),
            "target_state_id": controller.current_state_id,
            "environment_fingerprint_sha256": environment_gate_sha256,
            "source_run_directory": str(run_dir),
            "source_result_path": "PENDING_FINAL_RESULT",
            "source_result_sha256": "PENDING_FINAL_RESULT",
            "completed_prefix": prefix_completed,
        }
        prefix_path = run_dir / "verified_prefix_replay_manifest.json"
        _write_json(prefix_path, prefix_manifest)
        result = {
            "schema_version": "fsm50.controller_run_result.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "run_index": run_index,
            "run_dir": str(run_dir),
            "artifact_root": str(run_root),
            "artifact_valid": False,
            "environment_fingerprint_sha256": environment_gate_sha256,
            "start_state_id": start_state_id,
            "target_state_id": target_state_id,
            "target_entered": target_entered,
            "target_completed": target_completed,
            "controller_status": controller.status.value,
            "controller_success": controller_success,
            "state_test_success": state_test_success,
            "state_physical_success": state_physical_success,
            "full_physical_success": full_physical_success,
            "strict_success_before_video": (
                state_physical_success if mode == "test-state" else full_physical_success
            ),
            "strict_success": False,
            "classification": (
                "STATE_PHYSICALLY_VERIFIED"
                if state_physical_success
                else "FSM_PHYSICALLY_SUCCEEDED"
                if full_physical_success
                else "SAFE_STOP"
                if controller.status == ControllerStatus.SAFE_STOP
                else "PHYSICAL_EVIDENCE_INCOMPLETE"
            ),
            "first_failure_state": (
                controller.current_state_id
                if not (state_physical_success or full_physical_success)
                else ""
            ),
            "failure_reason": (
                controller.safe_stop_reason
                if controller.status == ControllerStatus.SAFE_STOP
                else "" if (state_physical_success or full_physical_success)
                else "controller or strict physical evidence incomplete"
            ),
            "physical_evidence": physical,
            "state_test_criteria": state_criteria,
            "controller_last_result": (
                None
                if last_controller_result is None
                else last_controller_result.to_mapping()
            ),
            "completed_state_ids": prefix_completed,
            "retry_counts": dict(controller.retry_counts),
            "restore_provenance": _jsonable(restore_provenance),
            "restore_result": _jsonable(restore_result),
            "environment_gate": {
                "path": str(args.environment_report),
                "sha256": environment_gate_sha256,
                "status": environment_gate.get("status"),
            },
            "runtime_environment_equivalence": runtime_environment,
            "video_manifest_path": str(video_manifest_path),
            "state_snapshot_count": len(snapshot_states),
            "captured_state_ids": sorted(snapshot_states),
            "prefix_manifest_path": str(prefix_path),
            "lifecycle": {"finalized": False, "failed": False},
        }
        try:
            visualization = runner._generate_fsm50_visualization(
                run_dir,
                fsm50_rows=collector.fsm50_rows,
                state_timeline_rows=collector.state_timeline_rows,
                strict_result=result,
            )
        except Exception as exc:
            visualization = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["visualization"] = visualization
        _write_json(
            run_dir / "visual_recording_manifest.json",
            {
                "kind": "fsm50_equivalent_telemetry_visualization",
                "not_camera_only": True,
                "actual_viewport_video": "PENDING_BATCH_VIDEO_FINALIZATION",
                "visualization": visualization,
                "artifact_valid": False,
                "basis": [
                    "fsm50_telemetry.csv",
                    "state_timeline.csv",
                    "fsm_controller_timeline.json",
                ],
            },
        )
        _write_json(run_root / "artifact_pointer.json", {"run_dir": str(run_dir)})
        result_path = run_dir / "result.json"
        _write_json(result_path, result)
        provisional_result_sha = _sha256(result_path)
        snapshots = _finalize_state_snapshots(
            run_dir,
            result_path=result_path,
            result_sha256=provisional_result_sha,
        )
        prefix_manifest["source_result_path"] = str(result_path)
        prefix_manifest["source_result_sha256"] = provisional_result_sha
        _write_json(prefix_path, prefix_manifest)
        _write_json(
            run_dir / "state_snapshot_index.json",
            {
                "schema_version": "fsm50.state_snapshot_index.v1",
                "source_result_path": str(result_path),
                "source_result_sha256": provisional_result_sha,
                "prefix_manifest_path": str(prefix_path),
                "prefix_manifest_sha256": _sha256(prefix_path),
                "snapshots": snapshots,
            },
        )
        _write_json(
            run_dir / "failure_diagnostics.json",
            {
                **result,
                "state_snapshot_index": str(run_dir / "state_snapshot_index.json"),
            },
        )
        return result
    except Exception as exc:
        run_dir = (
            collector.run_dir
            if collector is not None and collector.run_dir is not None
            else run_root
        )
        if collector is not None:
            try:
                collector.finish_episode(success=False, reason=str(exc))
            except Exception:
                pass
        failure = {
            "schema_version": "fsm50.controller_run_result.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "run_index": run_index,
            "run_dir": str(run_dir),
            "artifact_root": str(run_root),
            "artifact_valid": False,
            "environment_fingerprint_sha256": environment_gate_sha256,
            "strict_success": False,
            "classification": "RUNNER_EXCEPTION",
            "first_failure_state": "" if controller is None else controller.current_state_id,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "controller_status": (
                ControllerStatus.NOT_STARTED.value
                if controller is None
                else controller.status.value
            ),
            "controller_timeline": [] if controller is None else controller.timeline,
            "lifecycle": {"finalized": False, "failed": True},
        }
        if controller is not None:
            _write_controller_timeline(run_dir, controller.timeline)
        _write_json(run_dir / "failure_diagnostics.json", failure)
        _write_json(run_dir / "result.json", failure)
        _write_json(run_root / "artifact_pointer.json", {"run_dir": str(run_dir)})
        (run_root / ".failed").write_text("failed\n", encoding="utf-8")
        partial.unlink(missing_ok=True)
        return failure
    finally:
        try:
            adapter.attach_telemetry(None)
        except Exception:
            pass


def _materialize_episode_video(
    run_dir: Path,
    video: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the actual batch viewport recording into a checksummed run."""

    materialized = dict(video)
    materialized["scope"] = "batch_viewport_contains_all_episodes"
    if not bool(video.get("valid", False)):
        materialized["valid"] = False
        return materialized
    try:
        source = Path(str(video.get("video_path", ""))).resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"viewport video is missing or empty: {source}")
        declared_sha = str(video.get("video_sha256", "") or "").lower()
        source_sha = _sha256(source)
        if not declared_sha or source_sha != declared_sha:
            raise RuntimeError("viewport video SHA256 does not match its manifest")
        destination = run_dir / "actual_viewport_video.mp4"
        if source != destination.resolve():
            shutil.copy2(source, destination)
        destination_sha = _sha256(destination)
        if destination_sha != source_sha:
            raise RuntimeError("materialized viewport video SHA256 mismatch")
        materialized.update(
            {
                "valid": True,
                "actual_viewport_video": True,
                "not_camera_video": False,
                "source_video_path": str(source),
                "video_path": str(destination),
                "video_sha256": destination_sha,
                "size_bytes": destination.stat().st_size,
            }
        )
    except Exception as exc:
        materialized.update(
            {
                "valid": False,
                "actual_viewport_video": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    _write_json(run_dir / "viewport_video_manifest.json", materialized)
    return materialized


def _finalize_episode_result(
    result: dict[str, Any],
    *,
    video: Mapping[str, Any],
    source_integrity: Mapping[str, Any],
    runner: Any,
) -> None:
    run_dir = Path(str(result.get("run_dir", "")))
    if not run_dir.is_dir():
        return
    episode_video = _materialize_episode_video(run_dir, video)
    video_valid = bool(episode_video.get("valid", False))
    source_ok = bool(source_integrity.get("equal", False))
    visualization_ok = bool(
        dict(result.get("visualization", {}) or {}).get("ok", False)
    )
    required_paths = (
        run_dir / "physical_evidence.json",
        run_dir / "fsm50_telemetry.csv",
        run_dir / "fsm50_telemetry.jsonl",
        run_dir / "state_timeline.csv",
        run_dir / "fsm_controller_timeline.json",
        run_dir / "fsm_controller_timeline.csv",
        run_dir / "runtime_environment.json",
        run_dir / "fsm50_equivalent_visualization.png",
        run_dir / "fsm50_equivalent_visualization.html",
        run_dir / "viewport_video_manifest.json",
        run_dir / "actual_viewport_video.mp4",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    runner_exception = str(result.get("classification", "")) == "RUNNER_EXCEPTION"
    artifact_valid = bool(
        not runner_exception
        and source_ok
        and visualization_ok
        and video_valid
        and not missing
    )
    result["video"] = episode_video
    result["source_integrity"] = {
        "ok": source_ok,
        "scope": "fsm_batch_source_freeze",
        "comparison": dict(source_integrity),
    }
    result["artifact_valid"] = artifact_valid
    result["strict_success"] = bool(
        result.get("strict_success_before_video", False) and artifact_valid
    )
    if result["strict_success"]:
        result["classification"] = (
            "STATE_PHYSICALLY_VERIFIED"
            if result.get("mode") == "test-state"
            else "FSM_PHYSICALLY_SUCCEEDED"
        )
    if not artifact_valid:
        prior = str(result.get("classification", "") or "UNKNOWN")
        if prior != "ARTIFACT_INVALID":
            result["classification_before_artifact_validation"] = prior
        result["classification"] = "ARTIFACT_INVALID"
        reasons = []
        if runner_exception:
            reasons.append("runner exception")
        if not source_ok:
            reasons.append("source drift")
        if not visualization_ok:
            reasons.append("telemetry visualization unavailable")
        if not video_valid:
            reasons.append(
                str(episode_video.get("error", "actual viewport video unavailable"))
            )
        if missing:
            reasons.append("missing required files: " + ", ".join(missing))
        result["artifact_validation_failures"] = reasons
        result["failure_reason"] = "; ".join(reasons)
    result["lifecycle"] = {
        "finalized": artifact_valid,
        "failed": not artifact_valid,
        "strict_success": bool(result["strict_success"]),
    }
    visual_manifest_path = run_dir / "visual_recording_manifest.json"
    visual_manifest = (
        json.loads(visual_manifest_path.read_text(encoding="utf-8"))
        if visual_manifest_path.is_file()
        else {}
    )
    visual_manifest.update(
        {
            "kind": "fsm50_equivalent_telemetry_visualization",
            "not_camera_only": True,
            "not_camera_video": False,
            "actual_viewport_video": video_valid,
            "video": episode_video,
            "artifact_valid": artifact_valid,
        }
    )
    _write_json(visual_manifest_path, visual_manifest)
    _write_json(run_dir / "result.json", result)
    _refresh_restore_artifact_hashes(run_dir, run_dir / "result.json")
    _write_json(run_dir / "failure_diagnostics.json", result)
    runner._write_checksums(run_dir)
    artifact_root = Path(str(result.get("artifact_root", run_dir))).resolve()
    runner._mark_artifact_root(artifact_root, valid=artifact_valid)


def run_fsm_locked(
    args: Any,
    *,
    process_snapshot: list[dict[str, Any]],
    supervisor: Any | None = None,
) -> int:
    """Run one/state/five FSM episodes inside one supervised SimulationApp."""

    from sim_obstacle_scene import (
        DEFAULT_ROBOT_USD_PATH,
        OBSTACLE_LENGTH_M,
        OBSTACLE_WIDTH_M,
        SimSceneConfig,
        create_scene,
        ensure_simulation_app,
        finalize_scene_after_grounding,
        measure_obstacle_geometry,
        measure_scene_baseline,
    )
    from sim_robot_adapter import SimRobotAdapter, SimRobotAdapterConfig
    from sim_worker_runtime import (
        ground_reference_result_is_valid,
        initialize_adapter_ground_reference,
    )
    from . import run_fsm50 as runner

    output_root = Path(args.output_root).resolve()
    batch_root = runner._new_directory(output_root, str(args.command).replace("-", "_"))
    if supervisor is not None:
        supervisor.announce_batch(batch_root)
    partial = batch_root / ".partial"
    partial.write_text("running\n", encoding="utf-8")
    environment_report_path = Path(args.environment_report).resolve()
    environment_gate = _load_environment_gate(environment_report_path)
    environment_gate_sha256 = _sha256(environment_report_path)
    environment_lock_path = Path(args.report_root).resolve() / "environment_lock_50mm.json"
    environment_lock = runner._load_environment_lock(environment_lock_path)
    robot_usd = Path(args.robot_usd or DEFAULT_ROBOT_USD_PATH).resolve()
    source_freeze = runner._source_freeze(robot_usd=robot_usd)
    run_count = 5 if args.command == "validate-5" else 1
    _write_json(
        batch_root / "batch_request.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "args": vars(args),
            "run_count": run_count,
            "environment_report": str(environment_report_path),
            "environment_report_sha256": environment_gate_sha256,
            "source_freeze": source_freeze,
            "process_preflight": process_snapshot,
        },
    )
    motion = runner.load_motion_reference()
    scene_config = SimSceneConfig(
        obstacle_height_m=0.05,
        robot_usd=robot_usd,
        save_usd=runner.PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd",
        spawn_z=0.04,
        obstacle_x=1.55,
        obstacle_width=OBSTACLE_WIDTH_M,
        obstacle_length=OBSTACLE_LENGTH_M,
        infer_obstacle_size=False,
        robot_width=0.80,
        robot_length=0.55,
        physics_dt=1.0 / 120.0,
        render_interval=8,
        device=str(args.device),
        max_wheel_speed=float(motion.wheel_velocity_limit_rad_s),
        default_wheel_speed=float(motion.wheel_reference_velocity_rad_s),
        wheel_direction=1.0,
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        save_scene=False,
        defer_first_visible_render=True,
    )
    configure_scene_for_wheel_and_nonwheel_contacts(
        scene_config,
        wheel_factory=make_filtered_wheel_contact_sensor_factory(force_threshold_n=1.0),
        force_threshold_n=1.0,
    )
    simulation_app = None
    scene_handle = None
    results: list[dict[str, Any]] = []
    exit_code = 1
    batch_error = ""
    video = ViewportVideoRecorder(
        batch_root,
        enabled=bool(not args.headless and not getattr(args, "no_video", False)),
        fps=float(getattr(args, "video_fps", 15.0)),
    )
    try:
        simulation_app = ensure_simulation_app(args)
        scene_handle = create_scene(scene_config, simulation_app=simulation_app)
        adapter = SimRobotAdapter(
            scene_handle,
            SimRobotAdapterConfig(
                max_wheel_speed=float(motion.wheel_velocity_limit_rad_s),
                default_wheel_speed=float(motion.wheel_reference_velocity_rad_s),
                wheel_direction=1.0,
                apply_safe_servo_joint_limits=True,
                apply_joint_limits_to_sim=True,
                ground_settle_s=0.75,
                ground_settle_max_steps=180,
                ground_stable_frames=10,
                ground_vertical_speed_threshold_m_s=0.01,
                ground_joint_speed_threshold_rad_s=0.02,
                ground_servo_speed_threshold_rad_s=0.02,
                ground_wheel_speed_threshold_rad_s=0.20,
                ground_clearance_m=0.002,
                ground_penetration_tolerance_m=0.003,
                auto_ground_correction=False,
                max_ground_correction_m=0.10,
            ),
        )
        # Formal worker-equivalent path.  Historical environment-lock poses
        # are comparison evidence and are never written before grounding.
        ground = initialize_adapter_ground_reference(adapter)
        _write_json(batch_root / "ground_initialization.json", ground)
        if not ground_reference_result_is_valid(ground):
            raise RuntimeError(f"ground reference validation failed: {ground}")
        finalize_scene_after_grounding(scene_handle)
        live_baseline = measure_scene_baseline(scene_handle, adapter)
        live_obstacle = measure_obstacle_geometry(scene_handle)
        runtime_environment = runner._runtime_environment_equivalence(
            lock=dict(environment_lock),
            scene_config=scene_config,
            live_baseline=live_baseline,
            live_obstacle=live_obstacle,
            motion=motion,
            robot_usd=robot_usd,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )
        runtime_readback = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment_gate": environment_gate,
            "environment_gate_sha256": environment_gate_sha256,
            "scene_config": asdict(scene_config),
            "live_scene_baseline": live_baseline,
            "live_obstacle_geometry": live_obstacle,
            "ground_reference": ground,
            "runtime": runner._runtime_versions(),
            "runtime_environment_equivalence": runtime_environment,
            "readback_complete": True,
            "ok": bool(runtime_environment.get("ok", False)),
        }
        _write_json(batch_root / "runtime_environment_readback.json", runtime_readback)
        if not runtime_environment.get("ok", False):
            raise RuntimeError("runtime environment differs from environment lock")
        if scene_handle.contact_sensor is None or scene_handle.contact_sensor_error:
            raise RuntimeError(
                "filtered contact instrumentation unavailable: "
                + str(scene_handle.contact_sensor_error)
            )
        video.start()
        video_manifest_path = batch_root / "viewport_video_manifest.json"
        for run_index in range(1, run_count + 1):
            result = _run_one_controller_episode(
                run_index=run_index,
                args=args,
                adapter=adapter,
                scene_handle=scene_handle,
                batch_root=batch_root,
                environment_gate=environment_gate,
                environment_gate_sha256=environment_gate_sha256,
                environment_lock=environment_lock,
                robot_usd=robot_usd,
                video_manifest_path=video_manifest_path,
            )
            results.append(result)
            _write_json(batch_root / "batch_results.json", results)
            if result.get("classification") == "RUNNER_EXCEPTION":
                break
        video_result = video.finalize()
        source_post = runner._source_freeze(robot_usd=robot_usd)
        source_integrity = runner._compare_source_freezes(source_freeze, source_post)
        for result in results:
            _finalize_episode_result(
                result,
                video=video_result,
                source_integrity=source_integrity,
                runner=runner,
            )
        strict_count = sum(bool(result.get("strict_success", False)) for result in results)
        required_count = run_count
        batch_success = bool(
            len(results) == required_count
            and strict_count == required_count
            and source_integrity.get("equal", False)
            and video_result.get("valid", False)
        )
        summary = {
            "schema_version": "fsm50.validation_summary.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "required_runs": required_count,
            "completed_runs": len(results),
            "strict_success_count": strict_count,
            "strict_success": batch_success,
            "results": results,
            "source_integrity": source_integrity,
            "video": video_result,
        }
        _write_json(batch_root / "batch_results.json", results)
        _write_json(batch_root / "validation_summary.json", summary)
        _write_json(batch_root / "five_run_summary.json", summary if run_count == 5 else {})
        exit_code = 0 if batch_success else 1
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
        _write_json(
            batch_root / "batch_failure.json",
            {"error": batch_error, "results_so_far": results},
        )
        exit_code = 1
    finally:
        if not (batch_root / "viewport_video_manifest.json").is_file():
            try:
                video_result = video.finalize()
            except Exception as exc:
                video_result = {"valid": False, "error": str(exc)}
        else:
            video_result = json.loads(
                (batch_root / "viewport_video_manifest.json").read_text(encoding="utf-8")
            )
        try:
            batch_source_comparison = runner._compare_source_freezes(
                source_freeze, runner._source_freeze(robot_usd=robot_usd)
            )
        except Exception as exc:
            batch_source_comparison = {
                "equal": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        for result in results:
            lifecycle = dict(result.get("lifecycle", {}) or {})
            if lifecycle.get("finalized") is not True and lifecycle.get("failed") is not True:
                try:
                    _finalize_episode_result(
                        result,
                        video=video_result,
                        source_integrity=batch_source_comparison,
                        runner=runner,
                    )
                except Exception as exc:
                    result["artifact_valid"] = False
                    result["strict_success"] = False
                    result["classification"] = "ARTIFACT_INVALID"
                    result["failure_reason"] = (
                        f"late result finalization failed: {type(exc).__name__}: {exc}"
                    )
                    result["lifecycle"] = {"finalized": False, "failed": True}
        _write_json(batch_root / "source_integrity.json", batch_source_comparison)
        batch_artifact_valid = bool(
            not batch_error
            and len(results) == run_count
            and batch_source_comparison.get("equal", False)
            and bool(video_result.get("valid", False))
            and all(bool(row.get("artifact_valid", False)) for row in results)
        )
        finalization = {
            "schema_version": "fsm50.batch_finalization.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "finalized": batch_artifact_valid,
            "failed": not batch_artifact_valid,
            "artifact_valid": batch_artifact_valid,
            "strict_success": bool(
                batch_artifact_valid
                and results
                and all(bool(row.get("strict_success", False)) for row in results)
            ),
            "batch_error": batch_error,
            "result_count": len(results),
            "video": video_result,
            "source_integrity": batch_source_comparison,
            "phase": "PRECLOSE_FINALIZED",
        }
        _write_json(batch_root / "batch_finalization.json", finalization)
        _write_json(batch_root / "batch_results.json", results)
        _write_json(batch_root / "batch_results.preclose.json", results)
        _write_json(batch_root / "batch_finalization.preclose.json", finalization)
        runner._write_checksums(batch_root)
        checksums = batch_root / "checksums.sha256"
        if checksums.is_file():
            runner._atomic_copy_file(
                checksums,
                batch_root / "checksums.preclose.sha256",
            )
        runner._mark_artifact_root(batch_root, valid=batch_artifact_valid)
        evidence_manifest = runner._preclose_evidence_manifest(
            batch_root,
            results=results,
            batch_source_comparison=batch_source_comparison,
            batch_finalization=finalization,
        )
        if supervisor is not None:
            supervisor.mark_preclose(evidence_manifest)
        close_error = ""
        shutdown_mode = str(getattr(args, "shutdown_mode", "fast") or "fast")
        try:
            shutdown_mode = runner._close_simulation_with_explicit_policy(
                simulation_app=simulation_app,
                scene_handle=scene_handle,
                args=args,
                supervisor=supervisor,
                intended_returncode=(
                    0 if shutdown_mode == "fast" else exit_code
                ),
            )
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            # Preserve immutable preclose bytes.  The supervising parent owns
            # the post-return shutdown outcome and lifecycle rewrite.
        if supervisor is not None and close_error:
            supervisor.mark_close_error(
                shutdown_mode=shutdown_mode,
                error=close_error,
            )
    print(
        json.dumps(
            {"batch_root": str(batch_root), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return runner._process_returncode_after_close(
        command_exit_code=exit_code,
        shutdown_mode=shutdown_mode,
        close_error=close_error,
        supervised=supervisor is not None,
    )


__all__ = ["ViewportVideoRecorder", "run_fsm_locked"]
