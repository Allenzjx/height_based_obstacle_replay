"""CLI for audited 50 mm recording replays and the derived deterministic FSM.

The recording replay path deliberately uses the repository's authoritative
``plan_from_steps(..., profile="fast")`` plus ``SimTimePlaybackService``.  A
scheduler completion is recorded separately from strict physical traversal
evidence; the former can never make the latter true.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from command_model import SERVO_JOINT_NAMES, WHEEL_FORWARD_SIGN, WHEEL_JOINT_NAMES
from motion_speed import load_motion_reference
from playback import SimTimePlaybackService
from sequence_model import load_steps_jsonl
from telemetry.config import RuntimeTelemetryConfig
from telemetry.exporters import strict_json_dumps, write_csv, write_json

from .filtered_wheel_contact import make_filtered_wheel_contact_sensor_factory
from .fsm50_telemetry import FSM50TelemetryCollector
from .nonwheel_obstacle_contact import configure_scene_for_wheel_and_nonwheel_contacts
from .recording_audit import (
    DEFAULT_RECORDING_ROOT,
    DEFAULT_REPORT_ROOT,
    RecordingAudit,
    VersionFiles,
    sha256_file,
)
from .recording_fast_plan import (
    fast_plan_rows,
    write_fast_plan,
    write_source_dispatch_ledger,
)
from .wheel_integral_evidence import evaluate_wheel_integral_evidence
from .motion_start_readiness import (
    capture_live_motion_start_snapshot,
    evaluate_motion_start_ready,
    rest_qualification_summary,
)
from .shutdown_contract import SUCCESSFUL_SHUTDOWN_STATUSES
from .support_classifier import ObstacleGeometry
from .viewport_buffer_video import (
    CAPTURE_BACKEND as ACTIVE_VIEWPORT_BUFFER_BACKEND,
    ActiveViewportBufferVideoRecorder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = MODULE_ROOT / "runs"
ENVIRONMENT_LOCK_PATH = DEFAULT_REPORT_ROOT / "environment_lock_50mm.json"
VERSION_MATRIX_PATH = DEFAULT_REPORT_ROOT / "RECORDING_VERSION_MATRIX_50MM.csv"
DIAGONAL_TIMELINE_PATH = DEFAULT_REPORT_ROOT / "DIAGONAL_SUPPORT_TIMELINE.csv"
FIVE_RUN_SUMMARY_PATH = DEFAULT_REPORT_ROOT / "FSM50_5_RUN_SUMMARY.csv"
SINGLETON_LOCK_PATH = (
    Path(tempfile.gettempdir())
    / f"fsm50_replay_{hashlib.sha256(str(PROJECT_ROOT).encode('utf-8')).hexdigest()[:16]}.pid.lock"
)
SUPERVISED_CHILD_SENTINEL = "__replay-recordings-child"
SIMULATION_CLOSE_GRACE_S = 60.0
SUPERVISED_CHILD_SUCCESS_RETURNCODE = 0
ATOMIC_REPLACE_RETRY_S = 5.0


def _replace_with_windows_retry(source: Path, destination: Path) -> None:
    """Retry transient Windows sharing violations during atomic replacement."""

    deadline = time.monotonic() + float(ATOMIC_REPLACE_RETRY_S)
    delay_s = 0.01
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient_windows_error = bool(
                os.name == "nt"
                and (
                    isinstance(exc, PermissionError)
                    or getattr(exc, "winerror", None) in {5, 32}
                )
            )
            if not transient_windows_error or time.monotonic() >= deadline:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2.0, 0.10)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace one small supervisor/control JSON document."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        encoded = strict_json_dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_windows_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Durably snapshot a finalized evidence file without later rewriting it."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        _replace_with_windows_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


_PRECLOSE_SNAPSHOT_FILES: tuple[tuple[str, str], ...] = (
    ("batch_results.json", "batch_results.preclose.json"),
    ("batch_finalization.json", "batch_finalization.preclose.json"),
    ("checksums.sha256", "checksums.preclose.sha256"),
)

_WORKER_BATCH_CONTROL_FILES: tuple[str, ...] = (
    "worker_artifact_request.json",
    "worker_startup_binding.json",
    "worker_preplay_stop_ack.json",
    "worker_playback_start_ack.json",
    "worker_artifact_complete_ack.json",
)


def _snapshot_preclose_files(batch_root: Path) -> list[str]:
    """Copy the durable batch evidence set and report every failed copy."""

    errors: list[str] = []
    for source_name, destination_name in _PRECLOSE_SNAPSHOT_FILES:
        try:
            _atomic_copy_file(
                batch_root / source_name,
                batch_root / destination_name,
            )
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
    return errors


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""

    try:
        options = {
            "ensure_ascii": False,
            "sort_keys": True,
            "separators": (",", ":"),
        }
        return strict_json_dumps(left, **options) == strict_json_dumps(
            right, **options
        )
    except (TypeError, ValueError):
        return False


def _exact_json_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an exact JSON integer")
    return value


def _strict_json_read(path: Path) -> Any:
    """Read one JSON file while rejecting duplicates and non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError(f"duplicate JSON object key {key!r}")
            decoded[key] = value
        return decoded

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


class ChildSupervisorHandshake:
    """Child-owned atomic handshake consumed only by its exact parent PID."""

    def __init__(self, path: Path, *, token: str, parent_pid: int) -> None:
        self.path = Path(path).resolve()
        self.token = str(token)
        self.parent_pid = int(parent_pid)
        self.child_pid = os.getpid()
        self.batch_root: Path | None = None
        self.sequence = 0
        self.worker_binding: dict[str, Any] = {}

    def _write(self, state: str, **extra: Any) -> None:
        self.sequence += 1
        _atomic_write_json(
            self.path,
            {
                "schema_version": "fsm50.supervisor_handshake.v1",
                "token": self.token,
                "parent_pid": self.parent_pid,
                "child_pid": self.child_pid,
                "batch_root": "" if self.batch_root is None else str(self.batch_root),
                "state": str(state),
                "sequence": self.sequence,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                **_jsonable(self.worker_binding),
                **_jsonable(extra),
            },
        )

    def announce_batch(self, batch_root: Path) -> None:
        self.batch_root = Path(batch_root).resolve()
        self._write("BATCH_ALLOCATED")

    def mark_preclose(self, evidence: dict[str, Any]) -> Path:
        if self.batch_root is None:
            raise RuntimeError("cannot mark preclose before announcing batch_root")
        marker = self.batch_root / "preclose_complete.json"
        payload = {
            "schema_version": "fsm50.preclose_complete.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "token": self.token,
            "parent_pid": self.parent_pid,
            "child_pid": self.child_pid,
            "batch_root": str(self.batch_root),
            "evidence": _jsonable(evidence),
        }
        _atomic_write_json(marker, payload)
        self._write("PRECLOSE_COMPLETE", preclose_marker=str(marker))
        return marker

    def bind_worker(
        self,
        *,
        worker_pid: int,
        worker_session_id: str,
        adapter_runtime_instance_id: str,
        artifact_request_id: str,
        artifact_request_sha256: str,
    ) -> None:
        if int(worker_pid) <= 0 or not str(worker_session_id):
            raise ValueError("formal worker binding requires PID and session id")
        self.worker_binding = {
            "formal_worker_pid": int(worker_pid),
            "formal_worker_session_id": str(worker_session_id),
            "adapter_runtime_instance_id": str(adapter_runtime_instance_id),
            "artifact_request_id": str(artifact_request_id),
            "artifact_request_sha256": str(artifact_request_sha256).lower(),
        }
        self._write("FORMAL_WORKER_BOUND")

    def mark_graceful_close_returned(
        self,
        *,
        intended_returncode: int,
        worker_returncode: int | None = None,
        worker_process_returned_normally: bool | None = None,
        worker_shutdown_accepted: bool | None = None,
        worker_shutdown_request_id: str = "",
    ) -> None:
        self._write(
            "GRACEFUL_CLOSE_RETURNED",
            shutdown_mode="graceful",
            close_kwargs={
                "wait_for_replicator": False,
                "skip_cleanup": False,
            },
            intended_returncode=int(intended_returncode),
            worker_returncode=worker_returncode,
            worker_process_returned_normally=worker_process_returned_normally,
            worker_shutdown_accepted=worker_shutdown_accepted,
            worker_shutdown_request_id=str(worker_shutdown_request_id),
        )

    def mark_fast_exit_requested(
        self,
        *,
        intended_returncode: int,
        runtime_version: str,
        worker_shutdown_request_id: str = "",
    ) -> None:
        self._write(
            "FAST_EXIT_REQUESTED",
            shutdown_mode="fast",
            close_kwargs={
                "wait_for_replicator": False,
                "skip_cleanup": True,
            },
            intended_returncode=int(intended_returncode),
            runtime_version=str(runtime_version),
            worker_shutdown_request_id=str(worker_shutdown_request_id),
        )

    def mark_fast_close_returned(
        self,
        *,
        intended_returncode: int,
        runtime_version: str,
        worker_returncode: int | None = None,
        worker_process_returned_normally: bool | None = None,
        worker_shutdown_accepted: bool | None = None,
        worker_close_requested: bool | None = None,
        worker_close_returned: bool | None = None,
        worker_forced_termination: bool | None = None,
        worker_shutdown_request_id: str = "",
        worker_shutdown_ack: dict[str, Any] | None = None,
        worker_close_requested_ack: dict[str, Any] | None = None,
        worker_close_returned_ack: dict[str, Any] | None = None,
    ) -> None:
        # This state is deliberately not named CLOSE_RETURNED: a fast close
        # does not claim that graceful close_stage cleanup completed.
        self._write(
            "FAST_CLOSE_RETURNED",
            shutdown_mode="fast",
            close_kwargs={
                "wait_for_replicator": False,
                "skip_cleanup": True,
            },
            intended_returncode=int(intended_returncode),
            runtime_version=str(runtime_version),
            worker_returncode=worker_returncode,
            worker_process_returned_normally=worker_process_returned_normally,
            worker_shutdown_accepted=worker_shutdown_accepted,
            worker_close_requested=worker_close_requested,
            worker_close_returned=worker_close_returned,
            worker_forced_termination=worker_forced_termination,
            worker_shutdown_request_id=str(worker_shutdown_request_id),
            worker_shutdown_ack=_jsonable(worker_shutdown_ack or {}),
            worker_close_requested_ack=_jsonable(
                worker_close_requested_ack or {}
            ),
            worker_close_returned_ack=_jsonable(worker_close_returned_ack or {}),
        )

    def mark_fast_worker_process_returned(
        self,
        *,
        intended_returncode: int,
        runtime_version: str,
        worker_returncode: int,
        worker_process_returned_normally: bool,
        worker_shutdown_accepted: bool,
        worker_close_requested: bool,
        worker_close_returned: bool,
        worker_forced_termination: bool,
        worker_shutdown_request_id: str,
        worker_shutdown_ack: dict[str, Any],
        worker_close_requested_ack: dict[str, Any],
        worker_close_returned_ack: dict[str, Any] | None = None,
    ) -> None:
        """Record controller-observed worker exit after fast close dispatch.

        Isaac's ``skip_cleanup=True`` path may terminate the worker process
        without returning from ``SimulationApp.close``.  Therefore this state
        claims only that the controller received the pre-close ACK chain and
        then observed the exact worker process return normally.  A genuine
        post-close ACK is retained when available, but is never synthesized.
        """

        self._write(
            "FAST_WORKER_PROCESS_RETURNED",
            shutdown_mode="fast",
            close_kwargs={
                "wait_for_replicator": False,
                "skip_cleanup": True,
            },
            intended_returncode=int(intended_returncode),
            runtime_version=str(runtime_version),
            worker_returncode=worker_returncode,
            worker_process_returned_normally=worker_process_returned_normally,
            worker_shutdown_accepted=worker_shutdown_accepted,
            worker_close_requested=worker_close_requested,
            worker_close_returned=worker_close_returned,
            worker_forced_termination=worker_forced_termination,
            worker_shutdown_request_id=str(worker_shutdown_request_id),
            worker_shutdown_ack=_jsonable(worker_shutdown_ack),
            worker_close_requested_ack=_jsonable(worker_close_requested_ack),
            worker_close_returned_ack=_jsonable(
                worker_close_returned_ack or {}
            ),
        )

    def mark_close_error(self, *, shutdown_mode: str, error: str) -> None:
        self._write(
            "CLOSE_ERROR",
            shutdown_mode=str(shutdown_mode),
            close_error=str(error),
        )

    def mark_failed(self, error: str) -> None:
        self._write("CHILD_FAILED", error=str(error))


def _viewport_preclose_evidence_paths(
    run_dir: Path,
    video: dict[str, Any],
) -> list[Path]:
    """Return fail-closed viewport evidence without breaking failure shutdown.

    A valid-video claim requires the complete direct-buffer closure.  When
    capture itself fails, the wrapper manifest and any low-level diagnostic
    files that were actually finalized remain required evidence, while
    nonexistent PNG/MP4 outputs are not invented as required paths.  The run
    remains ARTIFACT_INVALID, but its diagnostic artifact can still reach
    PRECLOSE_COMPLETE and a separately verified Isaac shutdown.
    """

    root = Path(run_dir).resolve()
    wrapper_text = str(video.get("manifest_path", "") or "")
    wrapper = (
        Path(wrapper_text).resolve()
        if wrapper_text
        else root / "viewport_video_manifest.json"
    )
    paths = [wrapper]
    diagnostic_names = (
        "viewport_buffer_video_manifest.json",
        "viewport_frame_ledger.jsonl",
    )
    for name in diagnostic_names:
        candidate = root / name
        if candidate.is_file():
            paths.append(candidate)
    if video.get("actual_viewport_video") is True:
        paths.extend(
            [
                root / "viewport_buffer_video_manifest.json",
                root / "viewport_frame_ledger.jsonl",
                root / "viewport_first_frame.png",
                root / "viewport_last_frame.png",
                Path(
                    str(video.get("video_path", "") or root / "actual_viewport_video.mp4")
                ).resolve(),
            ]
        )
    return list(dict.fromkeys(paths))


def _preclose_evidence_manifest(
    batch_root: Path,
    *,
    results: list[dict[str, Any]],
    batch_source_comparison: dict[str, Any],
    batch_finalization: dict[str, Any],
    include_global_analysis_reports: bool = True,
) -> dict[str, Any]:
    paths: list[Path] = [
        batch_root / "batch_request.json",
        batch_root / "batch_results.json",
        batch_root / "batch_finalization.json",
        batch_root / "batch_results.preclose.json",
        batch_root / "batch_finalization.preclose.json",
        batch_root / "source_integrity.json",
        batch_root / "checksums.sha256",
        batch_root / "checksums.preclose.sha256",
    ]
    if include_global_analysis_reports:
        paths.extend([VERSION_MATRIX_PATH, DIAGONAL_TIMELINE_PATH])
    paths.extend(
        candidate
        for name in _WORKER_BATCH_CONTROL_FILES
        if (candidate := batch_root / name).is_file()
    )
    if any(_result_is_worker_owned(dict(result or {})) for result in results):
        paths.extend(batch_root / name for name in _WORKER_BATCH_CONTROL_FILES)
    for result in results:
        run_dir_text = str(result.get("run_dir", "") or "")
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        declared = list(result.get("required_evidence_paths", []) or [])
        if declared:
            paths.extend(Path(str(path)).resolve() for path in declared)
        else:
            video_record = dict(result.get("video", {}) or {})
            paths.extend(
                [
                    run_dir / "result.json",
                    run_dir / "failure_diagnostics.json",
                    run_dir / "physical_evidence.json",
                    run_dir / "fsm50_telemetry.csv",
                    run_dir / "fsm50_telemetry.jsonl",
                    run_dir / "state_timeline.csv",
                    run_dir / "telemetry_finalization.json",
                    run_dir / "visual_recording_manifest.json",
                    run_dir / "checksums.sha256",
                ]
            )
            paths.extend(_viewport_preclose_evidence_paths(run_dir, video_record))
    missing = [str(path.resolve()) for path in sorted(set(paths)) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "preclose evidence is incomplete; missing required files: "
            + ", ".join(missing)
        )
    evidence_files = {
        str(path.resolve()): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(paths))
    }
    return {
        "physics_result_count": len(results),
        "source_integrity": _jsonable(batch_source_comparison),
        "batch_finalization": _jsonable(batch_finalization),
        "evidence_files": evidence_files,
    }


class ReplaySingletonLock:
    """Atomic, ownership-checked singleton lock for Isaac replay processes."""

    def __init__(self, path: Path = SINGLETON_LOCK_PATH) -> None:
        self.path = Path(path).resolve()
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "token": self.token,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
        }
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    str(self.path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf-8"))
                    existing_pid = int(existing.get("pid", 0) or 0)
                except Exception as exc:
                    raise RuntimeError(
                        f"FSM50 singleton lock exists but is unreadable: {self.path}: {exc}"
                    ) from exc
                if existing_pid > 0 and _pid_is_alive(existing_pid):
                    raise RuntimeError(
                        f"FSM50 replay singleton is already held by pid={existing_pid}: {self.path}"
                    )
                # A dead owner left a stale lock.  Remove only the path whose
                # recorded owner was just checked, then retry O_EXCL once.
                try:
                    current = json.loads(self.path.read_text(encoding="utf-8"))
                    if (
                        int(current.get("pid", 0) or 0) != existing_pid
                        or str(current.get("token", ""))
                        != str(existing.get("token", ""))
                    ):
                        raise RuntimeError(
                            f"FSM50 singleton lock owner changed during stale-lock check: {self.path}"
                        )
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            else:
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                        stream.flush()
                        os.fsync(stream.fileno())
                except Exception:
                    try:
                        self.path.unlink()
                    except Exception:
                        pass
                    raise
                self.acquired = True
                return
        raise RuntimeError(f"could not acquire FSM50 singleton lock: {self.path}")

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                int(existing.get("pid", 0) or 0) == os.getpid()
                and str(existing.get("token", "")) == self.token
            ):
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False


def _pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if int(pid) == os.getpid():
        return True
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a reliable existence probe on Windows;
        # some dead PIDs produce a CPython SystemError wrapping WinError 87.
        # Query a non-mutating process handle and its exit code instead.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        handle = open_process(process_query_limited_information, False, int(pid))
        if not handle:
            # Access denied means the process exists but is protected. Other
            # errors (including invalid/dead PID) are safely treated as absent.
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return True
            return int(exit_code.value) == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _os_process_snapshot() -> list[dict[str, Any]]:
    """Return PID/name/command line using only operating-system facilities."""

    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OS process enumeration failed: "
                + str(completed.stderr or completed.stdout).strip()
            )
        raw = json.loads(completed.stdout or "[]")
        values = raw if isinstance(raw, list) else [raw]
        return [
            {
                "pid": int(row.get("ProcessId", 0) or 0),
                "name": str(row.get("Name", "") or ""),
                "command_line": str(row.get("CommandLine", "") or ""),
            }
            for row in values
            if isinstance(row, dict)
        ]
    completed = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OS process enumeration failed: "
            + str(completed.stderr or completed.stdout).strip()
        )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "command_line": parts[2] if len(parts) >= 3 else parts[1],
                }
            )
    return rows


def _process_looks_like_simulator(row: dict[str, Any]) -> bool:
    if int(row.get("pid", 0) or 0) == os.getpid():
        return False
    name = Path(str(row.get("name", "") or "")).name.lower()
    command = str(row.get("command_line", "") or "").lower().replace("\\", "/")
    if name in {"kit", "kit.exe", "isaac-sim", "isaac-sim.exe", "isaacsim"}:
        return True
    simulator_tokens = (
        "isaac-sim",
        "isaacsim.exp",
        "isaacsim_",
        "omni.isaac.sim",
        "/kit/kit.exe",
        "/kit/kit ",
    )
    worker_tokens = (
        " sim_worker.py",
        " -m sim_worker ",
        " -m sim_worker_runtime ",
        "/sim_worker_runtime.py",
    )
    return any(token in command for token in simulator_tokens + worker_tokens)


def _existing_simulator_processes(
    snapshot: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row) for row in snapshot if _process_looks_like_simulator(dict(row))]


def _read_supervisor_handshake(
    path: Path,
    *,
    token: str,
    parent_pid: int,
    child_pid: int,
    output_root: Path,
) -> tuple[dict[str, Any] | None, Path | None, bool]:
    if not Path(path).is_file():
        return None, None, False
    payload = _strict_json_read(Path(path))
    expected = {
        "schema_version": "fsm50.supervisor_handshake.v1",
        "token": str(token),
        "parent_pid": int(parent_pid),
        "child_pid": int(child_pid),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"supervisor handshake {key} mismatch: expected={value!r} "
                f"actual={payload.get(key)!r}"
            )
    batch_text = str(payload.get("batch_root", "") or "")
    batch_root = Path(batch_text).resolve() if batch_text else None
    if batch_root is not None and not _path_is_within(batch_root, output_root):
        raise RuntimeError(
            f"child batch_root escapes requested output root: {batch_root}"
        )
    preclose_valid = False
    if batch_root is not None:
        marker_path = batch_root / "preclose_complete.json"
        if marker_path.is_file():
            marker = _strict_json_read(marker_path)
            marker_expected = {
                "schema_version": "fsm50.preclose_complete.v1",
                "token": str(token),
                "parent_pid": int(parent_pid),
                "child_pid": int(child_pid),
                "batch_root": str(batch_root),
            }
            for key, value in marker_expected.items():
                if marker.get(key) != value:
                    raise RuntimeError(
                        f"preclose marker {key} mismatch: expected={value!r} "
                        f"actual={marker.get(key)!r}"
                    )
            preclose_valid = True
    return payload, batch_root, preclose_valid


def _validate_preclose_closure(batch_root: Path) -> dict[str, Any]:
    """Re-read every durable preclose byte before admitting any close mode."""

    root = Path(batch_root).resolve()
    marker_path = root / "preclose_complete.json"
    if not marker_path.is_file():
        raise RuntimeError("preclose_complete.json is missing")
    marker = _strict_json_read(marker_path)
    evidence = marker.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("preclose marker evidence is not a mapping")
    if (
        evidence.get("manifest_error")
        or evidence.get("immutable_preclose_errors")
        or evidence.get("batch_marker_error")
    ):
        raise RuntimeError("preclose marker records evidence/snapshot errors")
    evidence_files = evidence.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        raise RuntimeError("preclose evidence_files is empty or invalid")
    verified_files: dict[str, dict[str, Any]] = {}
    for raw_path, raw_record in evidence_files.items():
        if not isinstance(raw_record, dict):
            raise RuntimeError(f"preclose evidence row is invalid: {raw_path!r}")
        path = Path(str(raw_path)).resolve()
        if not path.is_file():
            raise RuntimeError(f"preclose evidence file is missing: {path}")
        expected_sha = str(raw_record.get("sha256", "") or "").lower()
        expected_size = int(raw_record.get("size_bytes", -1) or -1)
        if (
            len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
            or expected_size < 0
        ):
            raise RuntimeError(f"preclose evidence metadata is invalid: {path}")
        actual_sha = sha256_file(path).lower()
        actual_size = path.stat().st_size
        if actual_sha != expected_sha or actual_size != expected_size:
            raise RuntimeError(f"preclose evidence hash/size mismatch: {path}")
        verified_files[str(path)] = {
            "sha256": actual_sha,
            "size_bytes": actual_size,
        }

    required = (
        root / "batch_request.json",
        root / "batch_results.json",
        root / "batch_finalization.json",
        root / "batch_results.preclose.json",
        root / "batch_finalization.preclose.json",
        root / "source_integrity.json",
        root / "checksums.sha256",
        root / "checksums.preclose.sha256",
    )
    missing_evidence = [
        str(path)
        for path in required
        if str(path.resolve()) not in verified_files
    ]
    if missing_evidence:
        raise RuntimeError(
            "preclose evidence manifest does not cover required closure files: "
            + ", ".join(missing_evidence)
        )
    if sha256_file(root / "batch_results.json") != sha256_file(
        root / "batch_results.preclose.json"
    ):
        raise RuntimeError("batch_results preclose snapshot differs from live bytes")
    if sha256_file(root / "batch_finalization.json") != sha256_file(
        root / "batch_finalization.preclose.json"
    ):
        raise RuntimeError(
            "batch_finalization preclose snapshot differs from live bytes"
        )
    if sha256_file(root / "checksums.sha256") != sha256_file(
        root / "checksums.preclose.sha256"
    ):
        raise RuntimeError("preclose checksum snapshot differs from live bytes")

    batch_results = _strict_json_read(root / "batch_results.preclose.json")
    if not isinstance(batch_results, list):
        raise RuntimeError("batch_results preclose snapshot is not a list")
    try:
        physics_result_count = int(evidence.get("physics_result_count", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("preclose physics_result_count is invalid") from exc
    if physics_result_count != len(batch_results):
        raise RuntimeError("preclose physics_result_count differs from batch_results")
    worker_results = [
        dict(row or {})
        for row in batch_results
        if isinstance(row, dict) and _result_is_worker_owned(dict(row or {}))
    ]
    batch_request = _strict_json_read(root / "batch_request.json")
    formal_worker_schema = bool(
        isinstance(batch_request, dict)
        and batch_request.get("schema_version")
        == "fsm50.formal_worker_recording_batch.v1"
    )
    formal_worker_controls_present = any(
        (root / name).exists() for name in _WORKER_BATCH_CONTROL_FILES
    )
    formal_worker_batch = bool(
        formal_worker_schema
        or worker_results
        or formal_worker_controls_present
    )
    if formal_worker_batch and (
        not formal_worker_schema or not worker_results
    ):
        raise RuntimeError(
            "formal-worker batch request/result ownership is inconsistent"
        )
    formal_worker_identity: dict[str, Any] | None = None
    if formal_worker_batch:
        if len(batch_results) != 1 or len(worker_results) != 1:
            raise RuntimeError(
                "formal-worker batch_results must contain exactly one durable result"
            )
        for name in _WORKER_BATCH_CONTROL_FILES:
            control_path = (root / name).resolve()
            if str(control_path) not in verified_files:
                raise RuntimeError(
                    f"preclose formal-worker evidence does not cover {name}"
                )
        request = _strict_json_read(root / "worker_artifact_request.json")
        request_sha = sha256_file(root / "worker_artifact_request.json").lower()
        startup = _strict_json_read(root / "worker_startup_binding.json")
        stop_ack = _strict_json_read(root / "worker_preplay_stop_ack.json")
        start_ack = _strict_json_read(root / "worker_playback_start_ack.json")
        ack = _strict_json_read(root / "worker_artifact_complete_ack.json")
        if (
            ack.get("type") != "operation_ack"
            or ack.get("operation") != "recording_artifact"
            or ack.get("phase") != "ARTIFACT_COMPLETE"
            or ack.get("accepted") is not True
            or ack.get("artifact_complete") is not True
        ):
            raise RuntimeError("preclose worker artifact ACK is not complete")
        request_identity = {
            "request_id": str(request.get("request_id", "") or ""),
            "plan_id": str(request.get("plan_id", "") or ""),
            "plan_sha256": str(request.get("plan_sha256", "") or ""),
            "source_version": str(request.get("source_version", "") or ""),
            "trial_id": _exact_json_int(
                request.get("trial_id"), "worker request trial_id"
            ),
            "contact_mode": str(request.get("contact_mode", "") or ""),
            "environment_equivalence_role": str(
                request.get("environment_equivalence_role", "") or ""
            ),
            "diagnostic_role": str(request.get("diagnostic_role", "") or ""),
            "qualification_scope": str(
                request.get("qualification_scope", "") or ""
            ),
            "gate1_eligible": request.get("gate1_eligible"),
            "gate1_physical_qualification_eligible": request.get(
                "gate1_physical_qualification_eligible"
            ),
            "environment_equivalence_eligible": request.get(
                "environment_equivalence_eligible"
            ),
            "artifact_root": str(
                Path(str(request.get("artifact_root", "") or "")).resolve()
            ),
            "artifact_request_sha256": request_sha,
        }
        if (
            batch_request.get("schema_version")
            != "fsm50.formal_worker_recording_batch.v1"
            or list(batch_request.get("versions", []) or [])
            != [request_identity["source_version"]]
            or str(batch_request.get("artifact_request_path", "") or "")
            != str((root / "worker_artifact_request.json").resolve())
            or str(
                batch_request.get("artifact_request_sha256", "") or ""
            ).lower()
            != request_sha
            or str(batch_request.get("plan_id", "") or "")
            != request_identity["plan_id"]
            or str(batch_request.get("plan_sha256", "") or "")
            != request_identity["plan_sha256"]
            or _exact_json_int(
                batch_request.get("trial_id"), "batch request trial_id"
            )
            != request_identity["trial_id"]
            or str(
                dict(batch_request.get("args", {}) or {}).get(
                    "contact_mode", ""
                )
                or ""
            )
            != request_identity["contact_mode"]
            or str(
                dict(batch_request.get("args", {}) or {}).get(
                    "environment_equivalence_role", ""
                )
                or ""
            )
            != request_identity["environment_equivalence_role"]
            or str(
                batch_request.get("environment_equivalence_role", "") or ""
            )
            != request_identity["environment_equivalence_role"]
            or str(
                dict(batch_request.get("args", {}) or {}).get(
                    "diagnostic_role", ""
                )
                or ""
            )
            != request_identity["diagnostic_role"]
            or str(batch_request.get("diagnostic_role", "") or "")
            != request_identity["diagnostic_role"]
            or str(batch_request.get("qualification_scope", "") or "")
            != request_identity["qualification_scope"]
            or batch_request.get("gate1_eligible")
            is not request_identity["gate1_eligible"]
            or batch_request.get("gate1_physical_qualification_eligible")
            is not request_identity["gate1_physical_qualification_eligible"]
            or batch_request.get("environment_equivalence_eligible")
            is not request_identity["environment_equivalence_eligible"]
            or dict(
                batch_request.get("prelaunch_environment_validation", {}) or {}
            ).get("ok")
            is not True
        ):
            raise RuntimeError("preclose batch request differs from worker request")
        worker_identity = {
            "worker_pid": _exact_json_int(
                startup.get("worker_pid"), "preclose worker PID"
            ),
            "worker_session_id": str(startup.get("worker_session_id", "") or ""),
            "adapter_runtime_instance_id": str(
                startup.get("adapter_runtime_instance_id", "") or ""
            ),
        }
        if (
            worker_identity["worker_pid"] <= 0
            or not worker_identity["worker_session_id"]
            or not worker_identity["adapter_runtime_instance_id"]
            or str(startup.get("artifact_request_id", "") or "")
            != request_identity["request_id"]
            or str(startup.get("artifact_request_sha256", "") or "").lower()
            != request_sha
        ):
            raise RuntimeError("preclose worker startup binding is incomplete")
        replayed_startup = _validate_recording_worker_ready_status(
            dict(startup.get("status", {}) or {}),
            request=request,
            request_sha256=request_sha,
            worker_pid=worker_identity["worker_pid"],
        )
        for key, value in worker_identity.items():
            if replayed_startup.get(key) != value:
                raise RuntimeError(
                    f"preclose replayed startup {key} differs from binding"
                )
        formal_worker_identity = {
            **worker_identity,
            "artifact_request_id": request_identity["request_id"],
            "artifact_request_sha256": request_sha,
        }
        if (
            stop_ack.get("type") != "stop_ack"
            or stop_ack.get("zero_target_applied") is not True
            or str(stop_ack.get("error", "") or "")
            or str(stop_ack.get("reason", "") or "")
            != "playback_start_boundary"
            or not str(stop_ack.get("command_id", "") or "")
            or _exact_json_int(
                stop_ack.get("root_state_write_count"),
                "pre-play stop root_state_write_count",
            )
            != 0
            or str(stop_ack.get("artifact_request_id", "") or "")
            != request_identity["request_id"]
        ):
            raise RuntimeError("preclose formal-worker pre-play stop is invalid")
        if (
            start_ack.get("type") != "operation_ack"
            or start_ack.get("operation") != "start_playback_plan"
            or start_ack.get("accepted") is not True
            or start_ack.get("motion_start_ready") is not True
            or str(start_ack.get("error", "") or "")
            or str(start_ack.get("artifact_request_id", "") or "")
            != request_identity["request_id"]
            or _exact_json_int(
                start_ack.get("root_state_write_count"),
                "playback start root_state_write_count",
            )
            != 0
            or _exact_json_int(
                start_ack.get("event_count"), "playback start event_count"
            )
            != _exact_json_int(
                request.get("plan_event_count"), "worker request plan_event_count"
            )
            or _exact_json_int(
                start_ack.get("segment_count"), "playback start segment_count"
            )
            != _exact_json_int(
                request.get("plan_segment_count"),
                "worker request plan_segment_count",
            )
        ):
            raise RuntimeError("preclose formal-worker playback start is invalid")
        for label, row in (
            ("pre-play stop", stop_ack),
            ("playback start", start_ack),
            ("artifact complete", ack),
        ):
            for key, value in worker_identity.items():
                if not _strict_json_equal(row.get(key), value):
                    raise RuntimeError(
                        f"preclose {label} {key} differs from worker binding"
                    )
            for key in (
                "contact_mode",
                "environment_equivalence_role",
                "diagnostic_role",
                "qualification_scope",
                "gate1_eligible",
                "gate1_physical_qualification_eligible",
                "environment_equivalence_eligible",
            ):
                if not _strict_json_equal(row.get(key), request_identity[key]):
                    raise RuntimeError(
                        f"preclose {label} {key} differs from worker request"
                    )
        for key, value in request_identity.items():
            actual = ack.get(key)
            if key == "artifact_root" and actual:
                actual = str(Path(str(actual)).resolve())
            elif key == "artifact_request_sha256" and actual:
                actual = str(actual).lower()
            if actual != value:
                raise RuntimeError(
                    f"preclose artifact ACK {key} differs from request"
                )
        for key in ("request_id", "plan_id", "plan_sha256"):
            if start_ack.get(key) != request_identity[key]:
                raise RuntimeError(
                    f"preclose playback start {key} differs from request"
                )
        try:
            stop_applied_wall = float(stop_ack["target_applied_wall_time"])
            start_accepted_wall = float(start_ack["accepted_wall_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "preclose stop/start ordering timestamps are missing or invalid"
            ) from exc
        if (
            not math.isfinite(stop_applied_wall)
            or not math.isfinite(start_accepted_wall)
            or stop_applied_wall > start_accepted_wall
        ):
            raise RuntimeError(
                "preclose playback start preceded the applied zero-wheel stop"
            )
        durable_result = _validate_worker_artifact_complete_ack(
            ack,
            request=request,
            request_sha256=request_sha,
            batch_root=root,
            worker_pid=worker_identity["worker_pid"],
            worker_session_id=worker_identity["worker_session_id"],
            adapter_runtime_instance_id=worker_identity[
                "adapter_runtime_instance_id"
            ],
        )
        if not _strict_json_equal(batch_results, [durable_result]):
            raise RuntimeError(
                "batch_results is not byte-semantically equal to the worker result"
            )
        for key, value in {
            **worker_identity,
            **request_identity,
            "artifact_owner": "sim_worker_process",
            "execution_path": "sim_worker_process_ipc",
            "root_state_write_count": 0,
        }.items():
            if not _strict_json_equal(durable_result.get(key), value):
                raise RuntimeError(
                    f"preclose durable worker result {key} differs from closure"
                )
    snapshot_finalization = _strict_json_read(
        root / "batch_finalization.preclose.json"
    )
    if formal_worker_batch:
        durable_result = worker_results[0]
        environment_diagnostic_role = str(
            durable_result.get("environment_equivalence_role", "") or ""
        )
        ordinary_diagnostic_role = str(
            durable_result.get("diagnostic_role", "") or ""
        )
        expected_batch_diagnostic = bool(
            environment_diagnostic_role
            and durable_result.get(
                "environment_equivalence_diagnostic_complete"
            )
            is True
        )
        expected_ordinary_diagnostic = bool(
            ordinary_diagnostic_role == "U"
            and durable_result.get("ordinary_ui_diagnostic_complete") is True
        )
        expected_batch_scope = (
            "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
            if ordinary_diagnostic_role == "U"
            else "TRAJECTORY_COMPARISON"
            if environment_diagnostic_role
            else "GATE1_PHYSICAL_QUALIFICATION"
        )
        expected_batch_command_success = bool(
            expected_ordinary_diagnostic
            if ordinary_diagnostic_role == "U"
            else expected_batch_diagnostic
            if environment_diagnostic_role
            else durable_result.get("strict_full_success") is True
        )
        expected_batch_fields = {
            "environment_equivalence_role": environment_diagnostic_role,
            "environment_equivalence_diagnostic_complete": (
                expected_batch_diagnostic
            ),
            "diagnostic_role": ordinary_diagnostic_role,
            "ordinary_ui_diagnostic_complete": expected_ordinary_diagnostic,
            "qualification_scope": expected_batch_scope,
            "gate1_physical_qualification_eligible": bool(
                not ordinary_diagnostic_role and not environment_diagnostic_role
            ),
            "gate1_eligible": bool(
                not ordinary_diagnostic_role and not environment_diagnostic_role
            ),
            "environment_equivalence_eligible": bool(
                environment_diagnostic_role and not ordinary_diagnostic_role
            ),
            "command_success": expected_batch_command_success,
        }
        for key, value in expected_batch_fields.items():
            if not _strict_json_equal(snapshot_finalization.get(key), value):
                raise RuntimeError(
                    f"preclose batch diagnostic {key} differs from worker result"
                )
    if not _strict_json_equal(
        evidence.get("batch_finalization"), snapshot_finalization
    ):
        raise RuntimeError(
            "preclose embedded finalization differs from immutable snapshot"
        )
    marker_state = {
        "finalized": (root / ".finalized").is_file(),
        "failed": (root / ".failed").is_file(),
        "partial": (root / ".partial").exists(),
    }
    expected_finalized = snapshot_finalization.get("finalized") is True
    expected_failed = snapshot_finalization.get("failed") is True
    if marker_state["partial"] or expected_finalized == expected_failed:
        raise RuntimeError(
            "preclose lifecycle state is ambiguous or still partial"
        )
    if (
        marker_state["finalized"] != expected_finalized
        or marker_state["failed"] != expected_failed
        or marker_state["finalized"] == marker_state["failed"]
    ):
        raise RuntimeError(
            "preclose lifecycle marker does not match batch finalization"
        )
    source_integrity = evidence.get("source_integrity")
    if not isinstance(source_integrity, dict) or source_integrity.get("equal") is not True:
        raise RuntimeError("preclose source integrity is not equal=true")

    checksum_rows: dict[str, str] = {}
    checksum_path = root / "checksums.preclose.sha256"
    for line_number, raw in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative = raw.partition("  ")
        digest = digest.strip().lower()
        relative = relative.strip().replace("\\", "/")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in checksum_rows
        ):
            raise RuntimeError(
                f"invalid/duplicate preclose checksum row {line_number}"
            )
        target = (root / relative).resolve()
        if not _path_is_within(target, root):
            raise RuntimeError(f"preclose checksum path escapes batch root: {relative}")
        if not target.is_file() or sha256_file(target).lower() != digest:
            raise RuntimeError(f"preclose checksum mismatch: {relative}")
        checksum_rows[relative] = digest
    if not checksum_rows:
        raise RuntimeError("preclose checksum manifest is empty")
    for path in required[:3] + (root / "source_integrity.json",):
        relative = path.resolve().relative_to(root).as_posix()
        if relative not in checksum_rows:
            raise RuntimeError(
                f"preclose checksum manifest does not cover {relative}"
            )
    checksum_exclusions = {
        "checksums.sha256",
        "checksums.preclose.sha256",
        "batch_results.preclose.json",
        "batch_finalization.preclose.json",
    }
    uncovered_evidence: list[str] = []
    for absolute_text in verified_files:
        evidence_path = Path(absolute_text).resolve()
        if not _path_is_within(evidence_path, root):
            continue
        relative = evidence_path.relative_to(root).as_posix()
        if relative in checksum_exclusions:
            continue
        if checksum_rows.get(relative) != sha256_file(evidence_path).lower():
            uncovered_evidence.append(relative)
    if uncovered_evidence:
        raise RuntimeError(
            "preclose checksum manifest does not cover current evidence: "
            + ", ".join(sorted(uncovered_evidence))
        )
    return {
        "ok": True,
        "marker_path": str(marker_path),
        "marker_sha256": sha256_file(marker_path),
        "verified_file_count": len(verified_files),
        "checksum_row_count": len(checksum_rows),
        "checksums_preclose_sha256": sha256_file(checksum_path),
        "formal_worker_identity": _jsonable(formal_worker_identity),
    }


def _validate_formal_worker_fast_shutdown(
    *,
    handshake: dict[str, Any],
    preclose_identity: dict[str, Any],
    child_pid: int,
) -> dict[str, Any]:
    """Cross-bind the worker producer to pre-close ACKs and process exit."""

    fast_kwargs = {
        "wait_for_replicator": False,
        "skip_cleanup": True,
    }
    worker_pid = _exact_json_int(
        handshake.get("formal_worker_pid"), "formal worker PID"
    )
    controller_child_pid = _exact_json_int(
        child_pid, "controller child PID"
    )
    worker_session_id = str(
        handshake.get("formal_worker_session_id", "") or ""
    )
    adapter_id = str(
        handshake.get("adapter_runtime_instance_id", "") or ""
    )
    artifact_request_id = str(
        handshake.get("artifact_request_id", "") or ""
    )
    artifact_request_sha256 = str(
        handshake.get("artifact_request_sha256", "") or ""
    ).lower()
    expected_preclose = {
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": artifact_request_id,
        "artifact_request_sha256": artifact_request_sha256,
    }
    if controller_child_pid <= 0 or worker_pid <= 0 or worker_pid == controller_child_pid:
        raise RuntimeError("formal worker PID is missing or aliases controller child")
    if not worker_session_id or not adapter_id or not artifact_request_id:
        raise RuntimeError("formal worker shutdown identity is incomplete")
    if (
        len(artifact_request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_request_sha256)
    ):
        raise RuntimeError("formal worker artifact request SHA is invalid")
    if not _strict_json_equal(dict(preclose_identity or {}), expected_preclose):
        raise RuntimeError(
            "formal worker shutdown identity differs from preclose artifact producer"
        )

    shutdown_request_id = str(
        handshake.get("worker_shutdown_request_id", "") or ""
    )
    runtime_version = str(handshake.get("runtime_version", "") or "")
    if not shutdown_request_id or not runtime_version.startswith("5.1."):
        raise RuntimeError("formal worker fast shutdown request/runtime is invalid")
    common = {
        "request_id": shutdown_request_id,
        "worker_pid": worker_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": artifact_request_id,
        "root_state_write_count": 0,
        "mode": "fast",
        "accepted": True,
        "error": "",
        "close_kwargs": fast_kwargs,
        "runtime_version": runtime_version,
    }
    rows = (
        (
            "shutdown",
            dict(handshake.get("worker_shutdown_ack", {}) or {}),
            {"type": "operation_ack", "operation": "shutdown"},
        ),
        (
            "close_requested",
            dict(handshake.get("worker_close_requested_ack", {}) or {}),
            {"type": "close_requested"},
        ),
    )
    for label, row, specific in rows:
        for key, value in {**common, **specific}.items():
            if not _strict_json_equal(row.get(key), value):
                raise RuntimeError(
                    f"formal worker {label} ACK {key} mismatch: "
                    f"expected={value!r} actual={row.get(key)!r}"
                )
    if handshake.get("worker_shutdown_accepted") is not True:
        raise RuntimeError("formal worker shutdown accepted evidence is not true")
    if handshake.get("worker_close_requested") is not True:
        raise RuntimeError("formal worker close-requested evidence is not true")
    close_returned_observed = handshake.get("worker_close_returned")
    if type(close_returned_observed) is not bool:
        raise RuntimeError(
            "formal worker close-returned observation must be an exact boolean"
        )
    close_returned = dict(
        handshake.get("worker_close_returned_ack", {}) or {}
    )
    if close_returned_observed:
        for key, value in {**common, "type": "close_returned"}.items():
            if not _strict_json_equal(close_returned.get(key), value):
                raise RuntimeError(
                    f"formal worker close_returned ACK {key} mismatch: "
                    f"expected={value!r} actual={close_returned.get(key)!r}"
                )
    elif close_returned:
        raise RuntimeError(
            "formal worker close_returned ACK exists without an observed return"
        )
    if _exact_json_int(
        handshake.get("worker_returncode"), "formal worker return code"
    ) != 0:
        raise RuntimeError("formal worker return code is not zero")
    if handshake.get("worker_process_returned_normally") is not True:
        raise RuntimeError("formal worker did not return normally")
    if handshake.get("worker_forced_termination") is not False:
        raise RuntimeError("formal worker shutdown used or omitted forced-termination evidence")
    return {
        **expected_preclose,
        "worker_shutdown_request_id": shutdown_request_id,
        "runtime_version": runtime_version,
        "close_kwargs": fast_kwargs,
        "worker_returncode": 0,
        "worker_process_returned_normally": True,
        "worker_forced_termination": False,
        "worker_close_returned_observed": close_returned_observed,
    }


def _close_simulation_with_explicit_policy(
    *,
    simulation_app: Any | None,
    scene_handle: Any | None,
    args: Any,
    supervisor: ChildSupervisorHandshake | None,
    intended_returncode: int,
) -> str:
    """Execute exactly one preselected close policy after durable preclose."""

    mode = str(getattr(args, "shutdown_mode", "fast") or "fast").lower()
    if mode not in {"graceful", "fast"}:
        raise ValueError(f"unsupported shutdown mode: {mode}")
    runtime_versions = _runtime_versions()
    runtime_version = str(
        dict(runtime_versions.get("packages", {}) or {}).get(
            "isaacsim", "unknown"
        )
    )
    if simulation_app is None:
        if scene_handle is not None:
            scene_handle.close()
        if supervisor is not None:
            supervisor.mark_graceful_close_returned(
                intended_returncode=intended_returncode
            )
        return "graceful"
    if mode == "fast":
        if not runtime_version.startswith("5.1."):
            raise RuntimeError(
                "fast shutdown is restricted to the verified Isaac 5.1 path; "
                f"runtime={runtime_version}"
            )
        if supervisor is not None:
            supervisor.mark_fast_exit_requested(
                intended_returncode=intended_returncode,
                runtime_version=runtime_version,
            )
        simulation_app.close(
            wait_for_replicator=False,
            skip_cleanup=True,
        )
        if supervisor is not None:
            supervisor.mark_fast_close_returned(
                intended_returncode=intended_returncode,
                runtime_version=runtime_version,
            )
        return "fast"
    simulation_app.close(
        wait_for_replicator=False,
        skip_cleanup=False,
    )
    if supervisor is not None:
        supervisor.mark_graceful_close_returned(
            intended_returncode=intended_returncode
        )
    return "graceful"


def _terminate_owned_child_tree(child: Any) -> dict[str, Any]:
    """Terminate only the exact Popen-owned child and its descendants."""

    pid = int(child.pid)
    if child.poll() is not None:
        return {"requested": False, "pid": pid, "already_exited": True}
    if os.name == "nt":
        detail: dict[str, Any] = {
            "requested": True,
            "method": "taskkill_exact_owned_pid_tree",
            "pid": pid,
        }
        try:
            completed = subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
            )
            detail.update(
                {
                    "returncode": completed.returncode,
                    "stdout": str(completed.stdout or "").strip(),
                    "stderr": str(completed.stderr or "").strip(),
                }
            )
        except Exception as exc:
            detail["taskkill_error"] = f"{type(exc).__name__}: {exc}"
    else:
        import signal

        detail = {
            "requested": True,
            "method": "killpg_owned_child_session",
            "pid": pid,
        }
        try:
            process_group = os.getpgid(pid)
            detail["process_group"] = process_group
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            detail["already_exited_during_termination"] = True
    try:
        returncode = child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        returncode = child.wait(timeout=15)
        detail["root_kill_fallback"] = True
    detail["child_returncode"] = int(returncode)
    return detail


def _monitor_supervised_child(
    child: Any,
    *,
    handshake_path: Path,
    token: str,
    parent_pid: int,
    output_root: Path,
    close_grace_s: float = SIMULATION_CLOSE_GRACE_S,
    poll_interval_s: float = 0.20,
    clock: Any = time.monotonic,
    sleep: Any = time.sleep,
    terminate_tree: Any = _terminate_owned_child_tree,
) -> tuple[dict[str, Any], Path | None]:
    """Wait indefinitely for physics, then bound only post-marker shutdown."""

    started = clock()
    preclose_seen_at: float | None = None
    batch_root: Path | None = None
    last_state = ""
    while True:
        handshake, observed_root, preclose_valid = _read_supervisor_handshake(
            handshake_path,
            token=token,
            parent_pid=parent_pid,
            child_pid=int(child.pid),
            output_root=output_root,
        )
        if observed_root is not None:
            batch_root = observed_root
        if handshake is not None:
            last_state = str(handshake.get("state", "") or "")
        now = clock()
        if preclose_valid and preclose_seen_at is None:
            preclose_seen_at = now
        returncode = child.poll()
        if returncode is not None:
            # The child may publish its terminal handshake immediately before
            # exiting, between the poll-loop read and ``poll()``.  Re-read once
            # after observing the return code so that state is never lost to
            # this race.
            final_handshake, final_root, final_preclose = _read_supervisor_handshake(
                handshake_path,
                token=token,
                parent_pid=parent_pid,
                child_pid=int(child.pid),
                output_root=output_root,
            )
            if final_root is not None:
                batch_root = final_root
            if final_handshake is not None:
                handshake = final_handshake
                last_state = str(final_handshake.get("state", "") or "")
            preclose_valid = bool(final_preclose)
            preclose_verification: dict[str, Any] = {
                "ok": False,
                "error": "preclose marker was not verified",
            }
            if preclose_valid and batch_root is not None:
                try:
                    preclose_verification = _validate_preclose_closure(batch_root)
                except Exception as exc:
                    preclose_verification = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            intended_returncode = (
                None
                if handshake is None
                else handshake.get("intended_returncode")
            )
            returncode_matches = bool(
                type(returncode) is int
                and type(intended_returncode) is int
                and returncode == intended_returncode
            )
            shutdown_mode = str(
                "" if handshake is None else handshake.get("shutdown_mode", "")
            )
            close_kwargs = dict(
                {} if handshake is None else handshake.get("close_kwargs", {}) or {}
            )
            runtime_version = str(
                "" if handshake is None else handshake.get("runtime_version", "")
            )
            close_error = str(
                "" if handshake is None else handshake.get("close_error", "")
            )
            try:
                formal_worker_pid = int(
                    0
                    if handshake is None
                    else handshake.get("formal_worker_pid", 0) or 0
                )
            except (TypeError, ValueError):
                formal_worker_pid = 0
            preclose_worker_identity = dict(
                preclose_verification.get("formal_worker_identity", {}) or {}
            )
            worker_bound = bool(
                preclose_worker_identity
                or formal_worker_pid
                or (
                    handshake is not None
                    and any(
                        bool(handshake.get(key))
                        for key in (
                            "formal_worker_session_id",
                            "adapter_runtime_instance_id",
                            "artifact_request_id",
                            "artifact_request_sha256",
                            "worker_shutdown_request_id",
                            "worker_shutdown_accepted",
                            "worker_close_requested",
                            "worker_close_returned",
                            "worker_shutdown_ack",
                            "worker_close_requested_ack",
                            "worker_close_returned_ack",
                        )
                    )
                )
            )
            artifact_request_id = str(
                ""
                if handshake is None
                else handshake.get("artifact_request_id", "") or ""
            )
            formal_worker_shutdown_verification: dict[str, Any] = {}
            if not preclose_valid:
                status = "CHILD_EXIT_BEFORE_PRECLOSE"
            elif preclose_verification.get("ok") is not True:
                status = "PRECLOSE_CLOSURE_INVALID"
            elif last_state == "CLOSE_ERROR":
                status = "SIMULATION_CLOSE_ERROR"
            elif last_state in {
                "FAST_EXIT_REQUESTED",
                "FAST_CLOSE_RETURNED",
                "FAST_WORKER_PROCESS_RETURNED",
            }:
                worker_shutdown_request_id = str(
                    ""
                    if handshake is None
                    else handshake.get("worker_shutdown_request_id", "") or ""
                )
                worker_shutdown_ack = dict(
                    {} if handshake is None else handshake.get("worker_shutdown_ack", {}) or {}
                )
                worker_close_requested_ack = dict(
                    {}
                    if handshake is None
                    else handshake.get("worker_close_requested_ack", {}) or {}
                )
                worker_close_returned_ack = dict(
                    {}
                    if handshake is None
                    else handshake.get("worker_close_returned_ack", {}) or {}
                )
                formal_worker_session_id = str(
                    ""
                    if handshake is None
                    else handshake.get("formal_worker_session_id", "") or ""
                )
                formal_worker_shutdown_verification: dict[str, Any] = {
                    "ok": not worker_bound,
                    "error": "",
                }
                if worker_bound:
                    try:
                        verified_shutdown = _validate_formal_worker_fast_shutdown(
                            handshake=dict(handshake or {}),
                            preclose_identity=preclose_worker_identity,
                            child_pid=int(child.pid),
                        )
                        formal_worker_shutdown_verification = {
                            "ok": True,
                            "error": "",
                            **verified_shutdown,
                        }
                    except Exception as exc:
                        formal_worker_shutdown_verification = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                worker_close_verified = bool(
                    (
                        not worker_bound
                        and last_state
                        in {"FAST_EXIT_REQUESTED", "FAST_CLOSE_RETURNED"}
                    )
                    or (
                        worker_bound
                        and last_state == "FAST_WORKER_PROCESS_RETURNED"
                        and handshake is not None
                        and _strict_json_equal(
                            handshake.get("worker_returncode"), 0
                        )
                        and handshake.get("worker_process_returned_normally") is True
                        and handshake.get("worker_shutdown_accepted") is True
                        and handshake.get("worker_close_requested") is True
                        and type(handshake.get("worker_close_returned")) is bool
                        and handshake.get("worker_forced_termination") is False
                        and str(handshake.get("formal_worker_session_id", "") or "")
                        and str(handshake.get("artifact_request_id", "") or "")
                        and formal_worker_shutdown_verification.get("ok") is True
                    )
                )
                fast_contract = bool(
                    shutdown_mode == "fast"
                    and runtime_version.startswith("5.1.")
                    and close_kwargs
                    == {
                        "wait_for_replicator": False,
                        "skip_cleanup": True,
                    }
                    and worker_close_verified
                )
                status = (
                    "FAST_EXIT_VERIFIED"
                    if fast_contract and returncode_matches
                    else "FAST_EXIT_FAILED"
                )
            elif last_state == "GRACEFUL_CLOSE_RETURNED":
                worker_close_verified = bool(
                    not worker_bound
                    or (
                        handshake is not None
                        and handshake.get("worker_returncode") == 0
                        and handshake.get("worker_process_returned_normally") is True
                        and handshake.get("worker_shutdown_accepted") is True
                        and str(handshake.get("worker_shutdown_request_id", "") or "")
                        and str(handshake.get("formal_worker_session_id", "") or "")
                    )
                )
                graceful_contract = bool(
                    shutdown_mode == "graceful"
                    and close_kwargs
                    == {
                        "wait_for_replicator": False,
                        "skip_cleanup": False,
                    }
                    and worker_close_verified
                )
                status = (
                    "GRACEFUL_EXIT"
                    if graceful_contract and returncode_matches
                    else "CHILD_EXIT_UNEXPECTED_RETURNCODE"
                )
            else:
                status = "CHILD_EXIT_BEFORE_CLOSE_RETURNED"
            return (
                {
                    "schema_version": "fsm50.shutdown_outcome.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "parent_pid": int(parent_pid),
                    "child_pid": int(child.pid),
                    "child_returncode": int(returncode),
                    "preclose_observed": bool(preclose_valid),
                    "preclose_verification": preclose_verification,
                    "handshake_state": last_state,
                    "shutdown_mode": shutdown_mode,
                    "close_kwargs": close_kwargs,
                    "intended_returncode": intended_returncode,
                    "process_returned_normally": True,
                    "runtime_version": runtime_version,
                    "close_error": close_error,
                    "formal_worker_pid": (
                        None
                        if handshake is None
                        else handshake.get("formal_worker_pid")
                    ),
                    "formal_worker_session_id": (
                        ""
                        if handshake is None
                        else str(handshake.get("formal_worker_session_id", "") or "")
                    ),
                    "worker_returncode": (
                        None
                        if handshake is None
                        else handshake.get("worker_returncode")
                    ),
                    "worker_process_returned_normally": (
                        None
                        if handshake is None
                        else handshake.get("worker_process_returned_normally")
                    ),
                    "worker_shutdown_accepted": (
                        None
                        if handshake is None
                        else handshake.get("worker_shutdown_accepted")
                    ),
                    "worker_close_requested": (
                        None
                        if handshake is None
                        else handshake.get("worker_close_requested")
                    ),
                    "worker_close_returned": (
                        None
                        if handshake is None
                        else handshake.get("worker_close_returned")
                    ),
                    "worker_shutdown_request_id": (
                        ""
                        if handshake is None
                        else str(handshake.get("worker_shutdown_request_id", "") or "")
                    ),
                    "worker_shutdown_ack": (
                        {}
                        if handshake is None
                        else _jsonable(handshake.get("worker_shutdown_ack", {}) or {})
                    ),
                    "worker_close_requested_ack": (
                        {}
                        if handshake is None
                        else _jsonable(
                            handshake.get("worker_close_requested_ack", {}) or {}
                        )
                    ),
                    "worker_close_returned_ack": (
                        {}
                        if handshake is None
                        else _jsonable(
                            handshake.get("worker_close_returned_ack", {}) or {}
                        )
                    ),
                    "formal_worker_shutdown_verification": (
                        {}
                        if not worker_bound
                        else _jsonable(formal_worker_shutdown_verification)
                    ),
                    "adapter_runtime_instance_id": (
                        ""
                        if handshake is None
                        else str(
                            handshake.get("adapter_runtime_instance_id", "") or ""
                        )
                    ),
                    "artifact_request_id": artifact_request_id,
                    "artifact_request_sha256": (
                        ""
                        if handshake is None
                        else str(
                            handshake.get("artifact_request_sha256", "") or ""
                        ).lower()
                    ),
                    "worker_forced_termination": (
                        None
                        if handshake is None
                        else handshake.get("worker_forced_termination")
                    ),
                    "elapsed_s": float(now - started),
                    "close_wait_s": None
                    if preclose_seen_at is None
                    else float(now - preclose_seen_at),
                },
                batch_root,
            )
        if (
            preclose_seen_at is not None
            and now - preclose_seen_at >= float(close_grace_s)
        ):
            termination = terminate_tree(child)
            return (
                {
                    "schema_version": "fsm50.shutdown_outcome.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "SIMULATION_CLOSE_TIMEOUT",
                    "parent_pid": int(parent_pid),
                    "child_pid": int(child.pid),
                    "child_returncode": child.poll(),
                    "preclose_observed": True,
                    "handshake_state": last_state,
                    "elapsed_s": float(now - started),
                    "close_wait_s": float(now - preclose_seen_at),
                    "close_grace_s": float(close_grace_s),
                    "termination": _jsonable(termination),
                },
                batch_root,
            )
        sleep(float(poll_interval_s))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _new_directory(parent: Path, label: str) -> Path:
    path = parent / f"{_utc_stamp()}_{label}_{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _result_is_worker_owned(result: dict[str, Any]) -> bool:
    """Return whether immutable per-run bytes belong to the formal worker.

    A completed production-worker artifact has exactly one writer.  Batch
    source-integrity and shutdown failures remain qualification failures, but
    they must never rewrite telemetry, result, checksum, or lifecycle-marker
    bytes that the worker already finalized.
    """

    return str(result.get("artifact_owner", "") or "") == "sim_worker_process"


def _record_shutdown_outcome(
    batch_root: Path,
    outcome: dict[str, Any],
) -> None:
    """Persist supervisor outcome; lifecycle changes never rewrite physics class."""

    batch_root = Path(batch_root).resolve()
    status = str(outcome.get("status", "") or "")
    shutdown_failed = status not in SUCCESSFUL_SHUTDOWN_STATUSES
    _atomic_write_json(batch_root / "shutdown_outcome.json", outcome)
    results_path = batch_root / "batch_results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        results = []
    if not isinstance(results, list):
        results = []
    if shutdown_failed:
        for result in results:
            if not isinstance(result, dict):
                continue
            if _result_is_worker_owned(result):
                # Preserve the worker-owned artifact byte-for-byte.  The
                # batch finalization below records the shutdown failure and
                # is the authoritative qualification boundary.
                continue
            # Deliberately preserve classification, physical_success, and
            # strict_full_success: shutdown is an artifact lifecycle failure,
            # not a retroactive physics observation.
            lifecycle = dict(result.get("lifecycle", {}) or {})
            lifecycle.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "failure_reason": status,
                }
            )
            result["lifecycle"] = lifecycle
            result["artifact_valid"] = False
            result["shutdown_outcome"] = {
                "status": status,
                "path": str(batch_root / "shutdown_outcome.json"),
            }
            run_dir_text = str(result.get("run_dir", "") or "")
            if run_dir_text:
                run_dir = Path(run_dir_text).resolve()
                if run_dir.is_dir() and _path_is_within(run_dir, batch_root):
                    _atomic_write_json(run_dir / "result.json", result)
                    _atomic_write_json(
                        run_dir / "failure_diagnostics.json", result
                    )
                    _write_checksums(run_dir)
            artifact_text = str(result.get("artifact_root", "") or "")
            if artifact_text:
                artifact_root = Path(artifact_text).resolve()
                if artifact_root.is_dir() and _path_is_within(
                    artifact_root, batch_root
                ):
                    _mark_artifact_root(artifact_root, valid=False)
        _atomic_write_json(results_path, results)
    finalization_path = batch_root / "batch_finalization.json"
    try:
        batch_finalization = json.loads(
            finalization_path.read_text(encoding="utf-8")
        )
    except Exception:
        batch_finalization = {"artifact_root": str(batch_root)}
    batch_finalization["shutdown_outcome"] = _jsonable(outcome)
    if shutdown_failed:
        batch_finalization.update(
            {
                "finalized": False,
                "failed": True,
                "strict_success": False,
                "environment_equivalence_diagnostic_complete": False,
                "ordinary_ui_diagnostic_complete": False,
                "command_success": False,
                "failure_reason": status,
                "close_error": str(outcome.get("close_error", "") or status),
                "phase": status,
            }
        )
    else:
        batch_finalization.update(
            {
                "close_error": "",
                "phase": "SHUTDOWN_COMPLETE",
            }
        )
    _atomic_write_json(finalization_path, batch_finalization)
    _write_checksums(batch_root)
    if shutdown_failed:
        _mark_artifact_root(batch_root, valid=False)


def _batch_command_exit_code(batch_root: Path, *, fallback: int) -> int:
    """Separate command qualification from the fast-close process return."""

    try:
        finalization = json.loads(
            (Path(batch_root).resolve() / "batch_finalization.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return int(fallback)
    if not isinstance(finalization, dict):
        return int(fallback)
    root = Path(batch_root).resolve()
    if (
        str(finalization.get("phase", "") or "") != "SHUTDOWN_COMPLETE"
        or finalization.get("finalized") is not True
        or finalization.get("failed") is not False
        or bool(str(finalization.get("batch_error", "") or ""))
        or bool(str(finalization.get("close_error", "") or ""))
        or bool(list(finalization.get("finalization_errors", []) or []))
        or not (root / ".finalized").is_file()
        or (root / ".partial").exists()
        or (root / ".failed").exists()
    ):
        return 1
    diagnostic_role = str(
        finalization.get("environment_equivalence_role", "") or ""
    )
    ordinary_role = str(finalization.get("diagnostic_role", "") or "")
    if ordinary_role == "U":
        return (
            0
            if finalization.get("ordinary_ui_diagnostic_complete") is True
            and finalization.get("command_success") is True
            else 1
        )
    if diagnostic_role:
        return (
            0
            if finalization.get(
                "environment_equivalence_diagnostic_complete"
            )
            is True
            and finalization.get("command_success") is True
            else 1
        )
    return 0 if finalization.get("strict_success") is True else 1


def _process_returncode_after_close(
    *,
    command_exit_code: int,
    shutdown_mode: str,
    close_error: str,
    supervised: bool,
) -> int:
    """Separate a verified fast child exit from the command result.

    Isaac 5.1 may terminate inside ``SimulationApp.close(skip_cleanup=True)``
    or may return to Python.  In both cases a supervised fast-close child must
    return zero so the parent can verify a normal process return.  The parent
    then restores the user-facing PASS/FAIL status from the finalized batch;
    an unsupervised invocation retains the command's original exit code.
    """

    if close_error:
        return 1
    if supervised and str(shutdown_mode) == "fast":
        return 0
    return int(command_exit_code)


def _serialize_replay_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in vars(args).items()}


def _normalize_recording_replay_role(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Bind the opt-in A1/A2/B role to exactly one contact mode.

    An absent role is the existing Gate-1 contract and remains instrumented.
    Formal aggregate-sensor capture is reachable only through an explicit A
    role; this prevents a plain physical-qualification invocation from being
    silently downgraded to trajectory-comparison evidence.
    """

    if str(getattr(args, "command", "") or "") != "replay-recordings":
        return args
    role = str(
        getattr(args, "environment_equivalence_role", "") or ""
    ).strip().upper()
    diagnostic_role = str(
        getattr(args, "diagnostic_role", "") or ""
    ).strip().upper()
    if role not in {"", "A1", "A2", "B"}:
        raise RuntimeError(
            "--environment-equivalence-role must be A1, A2, or B"
        )
    if diagnostic_role not in {"", "U"}:
        raise RuntimeError("--diagnostic-role must be U")
    if diagnostic_role and role:
        raise RuntimeError(
            "--diagnostic-role U and --environment-equivalence-role are mutually exclusive"
        )
    requested_mode = str(getattr(args, "contact_mode", "") or "").strip().lower()
    expected_mode = (
        "disabled"
        if diagnostic_role == "U"
        else "formal"
        if role in {"A1", "A2"}
        else "instrumented"
    )
    if not role and requested_mode == "formal":
        raise RuntimeError(
            "formal capture requires an explicit --environment-equivalence-role A1 or A2"
        )
    if requested_mode and requested_mode != expected_mode:
        raise RuntimeError(
            f"environment-equivalence role {role or '<none>'} requires "
            f"--contact-mode {expected_mode}"
        )
    if (role or diagnostic_role) and bool(getattr(args, "resume", False)):
        raise RuntimeError(
            "diagnostic captures must be fresh; --resume is forbidden"
        )
    args.environment_equivalence_role = role
    args.diagnostic_role = diagnostic_role
    args.contact_mode = expected_mode
    return args


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        strict_json_dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _recording_worker_args(
    args: argparse.Namespace,
    *,
    robot_usd: Path,
    motion: Any,
    artifact_request_path: Path,
) -> SimpleNamespace:
    """Build the exact production-worker launch surface for Gate-1 replay."""

    from sim_obstacle_scene import OBSTACLE_LENGTH_M, OBSTACLE_WIDTH_M

    return SimpleNamespace(
        worker_launch_mode=str(
            getattr(args, "worker_launch_mode", "auto") or "auto"
        ),
        worker_python_exe=str(
            getattr(args, "worker_python_exe", "") or ""
        ),
        isaaclab_bat=str(
            getattr(args, "isaaclab_bat", "C:/robotics_sim/IsaacLab/isaaclab.bat")
        ),
        preflight_timeout_s=float(
            getattr(args, "preflight_timeout_s", 30.0)
        ),
        sim_startup_timeout_s=float(
            getattr(args, "sim_startup_timeout_s", 600.0)
        ),
        sim_worker_status_timeout_s=float(
            getattr(args, "sim_worker_status_timeout_s", 10.0)
        ),
        sim_worker_log_lines=int(
            getattr(args, "sim_worker_log_lines", 200)
        ),
        accept_isaac_eula=bool(
            getattr(args, "accept_isaac_eula", False)
        ),
        height_mm=50,
        height_cm=None,
        robot_usd=str(Path(robot_usd).resolve()),
        save_usd=str(
            (PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd").resolve()
        ),
        save_scene=False,
        spawn_z=0.04,
        obstacle_x=1.55,
        obstacle_width=float(OBSTACLE_WIDTH_M),
        obstacle_length=float(OBSTACLE_LENGTH_M),
        infer_obstacle_size=False,
        robot_width=0.80,
        robot_length=0.55,
        physics_dt=1.0 / 120.0,
        render_interval=8,
        wheel_direction=1.0,
        max_wheel_speed_rad_s=float(motion.wheel_velocity_limit_rad_s),
        default_wheel_speed_rad_s=float(motion.wheel_reference_velocity_rad_s),
        servo_stiffness=600.0,
        servo_damping=60.0,
        wheel_damping=20.0,
        device=str(getattr(args, "device", "cuda:0") or "cuda:0"),
        sim_status_refresh_ms=125,
        headless=bool(getattr(args, "headless", False)),
        livestream=int(getattr(args, "livestream", 0) or 0),
        experience=str(getattr(args, "experience", "") or ""),
        apply_safe_servo_joint_limits=True,
        apply_physx_joint_limits=True,
        no_continuous_sim_step=False,
        worker_smoke_negative_knee_test=False,
        worker_smoke_ground_structure=False,
        worker_smoke_ground_calibration=False,
        worker_smoke_output="",
        worker_smoke_test_s=0.0,
        defer_first_visible_render=True,
        robot_ground_settle_s=0.75,
        robot_ground_settle_max_steps=180,
        robot_ground_stable_frames=10,
        robot_ground_vertical_speed_threshold_m_s=0.01,
        robot_ground_joint_speed_threshold_rad_s=0.02,
        robot_ground_servo_speed_threshold_rad_s=0.02,
        robot_ground_wheel_speed_threshold_rad_s=0.20,
        robot_ground_clearance_m=0.002,
        robot_ground_penetration_tolerance_m=0.003,
        robot_auto_ground_correction=False,
        robot_max_ground_correction_m=0.10,
        fsm50_gate_request_path=str(Path(artifact_request_path).resolve()),
    )


def _recording_artifact_request(
    *,
    item: VersionFiles,
    batch_root: Path,
    args: argparse.Namespace,
    robot_usd: Path,
    environment_lock_path: Path,
    expected_steps_sha256: str,
    plan: Any,
    request_id: str,
    plan_id: str,
    trial_id: int,
) -> dict[str, Any]:
    """Create the immutable, pre-launch formal-worker artifact request."""

    # Allocate only a name.  The production worker is the exclusive creator
    # and writer of the version/artifact subtree.
    artifact_root = (
        Path(batch_root).resolve()
        / item.version_id
        / f"{_utc_stamp()}_clean_fast_replay_{uuid.uuid4().hex[:10]}"
    )
    equivalence_role = str(
        getattr(args, "environment_equivalence_role", "") or ""
    )
    diagnostic_role = str(getattr(args, "diagnostic_role", "") or "")
    qualification_scope = (
        "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
        if diagnostic_role == "U"
        else "TRAJECTORY_COMPARISON"
        if equivalence_role
        else "GATE1_PHYSICAL_QUALIFICATION"
    )
    request = {
        "schema_version": "fsm50.worker_recording_gate_request.v1",
        "enabled": True,
        "artifact_owner": "sim_worker_process",
        "request_id": str(request_id),
        "plan_id": str(plan_id),
        "plan_sha256": str(plan.plan_sha256),
        "plan_event_count": len(plan.events),
        "plan_segment_count": len(plan.segments),
        "source_version": item.version_id,
        "trial_id": int(trial_id),
        "contact_mode": str(
            getattr(args, "contact_mode", "instrumented") or "instrumented"
        ),
        "environment_equivalence_role": equivalence_role,
        "diagnostic_role": diagnostic_role,
        "qualification_scope": qualification_scope,
        "gate1_eligible": bool(not diagnostic_role and not equivalence_role),
        "gate1_physical_qualification_eligible": bool(
            not diagnostic_role and not equivalence_role
        ),
        "environment_equivalence_eligible": bool(
            equivalence_role and not diagnostic_role
        ),
        "height_mm": 50,
        "artifact_root": str(artifact_root),
        "accepted_steps_path": str(item.steps_path.resolve()),
        "metadata_path": str(item.metadata_path.resolve()),
        "accepted_steps_sha256": str(expected_steps_sha256).lower(),
        "expected_accepted_steps_sha256": str(expected_steps_sha256).lower(),
        "robot_usd_path": str(Path(robot_usd).resolve()),
        "expected_robot_usd_sha256": sha256_file(robot_usd).lower(),
        "environment_lock_path": str(Path(environment_lock_path).resolve()),
        "environment_lock_sha256": sha256_file(environment_lock_path).lower(),
        "telemetry_rate_hz": float(args.telemetry_rate),
        "post_run_settle_s": float(args.post_run_settle_s),
        "timeout_s": float(args.timeout_s),
        "timeout_scale": float(args.timeout_scale),
        "capture_video": bool(
            not bool(getattr(args, "no_video", False))
            and not bool(getattr(args, "headless", False))
        ),
        "headless": bool(getattr(args, "headless", False)),
        "video_fps": float(args.video_fps),
        "expected_root_state_write_count": 0,
        "expected_plan_sha256": str(plan.plan_sha256),
    }
    return request


def _parse_checksum_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
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
                f"invalid/duplicate checksum row {line_number}: {path}"
            )
        rows[relative] = digest
    if not rows:
        raise RuntimeError(f"checksum manifest is empty: {path}")
    return rows


def _validate_worker_artifact_complete_ack(
    ack: dict[str, Any],
    *,
    request: dict[str, Any],
    request_sha256: str,
    batch_root: Path,
    worker_pid: int,
    worker_session_id: str,
    adapter_runtime_instance_id: str,
) -> dict[str, Any]:
    """Re-read a sealed worker artifact without mutating its subtree."""

    requested_equivalence_role = str(
        request.get("environment_equivalence_role", "") or ""
    )
    requested_diagnostic_role = str(request.get("diagnostic_role", "") or "")
    expected_scope = (
        "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
        if requested_diagnostic_role == "U"
        else "TRAJECTORY_COMPARISON"
        if requested_equivalence_role
        else "GATE1_PHYSICAL_QUALIFICATION"
    )
    expected = {
        "type": "operation_ack",
        "operation": "recording_artifact",
        "phase": "ARTIFACT_COMPLETE",
        "artifact_owner": "sim_worker_process",
        "request_id": str(request["request_id"]),
        "artifact_request_id": str(request["request_id"]),
        "plan_id": str(request["plan_id"]),
        "plan_sha256": str(request["plan_sha256"]),
        "artifact_request_sha256": str(request_sha256),
        "worker_pid": int(worker_pid),
        "worker_session_id": str(worker_session_id),
        "adapter_runtime_instance_id": str(adapter_runtime_instance_id),
        "source_version": str(request["source_version"]),
        "trial_id": int(request["trial_id"]),
        "accepted_steps_sha256": str(
            request.get(
                "expected_accepted_steps_sha256",
                request.get("accepted_steps_sha256", ""),
            )
        ),
        "contact_mode": str(request["contact_mode"]),
        "environment_equivalence_role": str(
            request.get("environment_equivalence_role", "") or ""
        ),
        "environment_equivalence_diagnostic": bool(
            requested_equivalence_role
        ),
        "diagnostic_role": requested_diagnostic_role,
        "ordinary_ui_diagnostic": requested_diagnostic_role == "U",
        "qualification_scope": expected_scope,
        "gate1_eligible": bool(
            not requested_diagnostic_role and not requested_equivalence_role
        ),
        "gate1_physical_qualification_eligible": bool(
            not requested_diagnostic_role and not requested_equivalence_role
        ),
        "environment_equivalence_eligible": bool(
            requested_equivalence_role and not requested_diagnostic_role
        ),
    }
    for key, value in expected.items():
        if not _strict_json_equal(ack.get(key), value):
            raise RuntimeError(
                f"worker artifact ACK {key} mismatch: "
                f"expected={value!r} actual={ack.get(key)!r}"
            )
    if ack.get("accepted") is not True or ack.get("artifact_complete") is not True:
        raise RuntimeError("worker artifact ACK is not accepted and complete")
    if str(ack.get("error", "") or ""):
        raise RuntimeError(f"worker artifact ACK contains error: {ack['error']}")
    try:
        root_write_count = _exact_json_int(
            ack.get("root_state_write_count"),
            "worker artifact root_state_write_count",
        )
    except (TypeError, ValueError):
        root_write_count = -1
    if root_write_count != 0:
        raise RuntimeError("worker artifact reports a root-state write")

    root = Path(batch_root).resolve()
    artifact_root = Path(str(ack.get("artifact_root", "") or "")).resolve()
    run_dir = Path(str(ack.get("run_dir", "") or "")).resolve()
    result_path = Path(str(ack.get("result_path", "") or "")).resolve()
    if not _path_is_within(artifact_root, root) or artifact_root == root:
        raise RuntimeError("worker artifact_root escapes or aliases batch_root")
    if artifact_root != Path(str(request.get("artifact_root", "") or "")).resolve():
        raise RuntimeError("worker artifact_root differs from the sealed request")
    if not _path_is_within(run_dir, artifact_root):
        raise RuntimeError("worker run_dir escapes artifact_root")
    if result_path != run_dir / "result.json":
        raise RuntimeError("worker result_path is not run_dir/result.json")
    if (artifact_root / ".partial").exists():
        raise RuntimeError("worker artifact is still partial")
    finalized = (artifact_root / ".finalized").is_file()
    failed = (artifact_root / ".failed").is_file()
    if finalized == failed:
        raise RuntimeError("worker artifact lifecycle marker is ambiguous")

    declared_paths = {
        "result_path": "result_sha256",
        "artifact_pointer_path": "artifact_pointer_sha256",
        "checksums_path": "checksums_sha256",
        "telemetry_finalization_path": "telemetry_finalization_sha256",
        "visual_manifest_path": "visual_manifest_sha256",
    }
    resolved_paths: dict[str, Path] = {}
    for path_key, sha_key in declared_paths.items():
        path = Path(str(ack.get(path_key, "") or "")).resolve()
        if not path.is_file() or not _path_is_within(path, artifact_root):
            raise RuntimeError(f"worker ACK path missing/escaping: {path_key}")
        if sha256_file(path).lower() != str(ack.get(sha_key, "") or "").lower():
            raise RuntimeError(f"worker ACK digest mismatch: {path_key}")
        resolved_paths[path_key] = path

    pointer = _strict_json_read(resolved_paths["artifact_pointer_path"])
    if Path(str(pointer.get("run_dir", "") or "")).resolve() != run_dir:
        raise RuntimeError("worker artifact pointer does not bind run_dir")
    result = _strict_json_read(result_path)
    if not isinstance(result, dict):
        raise RuntimeError("worker result is not a mapping")
    equivalence_role = str(
        request.get("environment_equivalence_role", "") or ""
    )
    diagnostic_role = str(request.get("diagnostic_role", "") or "")
    contact_mode = str(request.get("contact_mode", "") or "")
    expected_contact_mode = (
        "disabled"
        if diagnostic_role == "U"
        else "formal"
        if equivalence_role in {"A1", "A2"}
        else "instrumented"
    )
    if (
        equivalence_role not in {"", "A1", "A2", "B"}
        or diagnostic_role not in {"", "U"}
        or bool(equivalence_role and diagnostic_role)
        or contact_mode != expected_contact_mode
        or str(request.get("qualification_scope", "") or "") != expected_scope
    ):
        raise RuntimeError(
            "worker request environment-equivalence role/contact-mode matrix is invalid"
        )
    result_expected = {
        "artifact_owner": "sim_worker_process",
        "execution_path": "sim_worker_process_ipc",
        "artifact_request_sha256": str(request_sha256),
        "request_id": str(request["request_id"]),
        "plan_id": str(request["plan_id"]),
        "source_version": str(request["source_version"]),
        "trial_id": int(request["trial_id"]),
        "plan_sha256": str(request["plan_sha256"]),
        "contact_mode": str(request["contact_mode"]),
        "environment_equivalence_role": str(
            request.get("environment_equivalence_role", "") or ""
        ),
        "diagnostic_role": diagnostic_role,
        "qualification_scope": expected_scope,
        "gate1_eligible": bool(not diagnostic_role and not equivalence_role),
        "gate1_physical_qualification_eligible": bool(
            not diagnostic_role and not equivalence_role
        ),
        "environment_equivalence_eligible": bool(
            equivalence_role and not diagnostic_role
        ),
        "accepted_steps_sha256": str(
            request.get(
                "expected_accepted_steps_sha256",
                request.get("accepted_steps_sha256", ""),
            )
        ),
        "run_dir": str(run_dir),
        "artifact_root": str(artifact_root),
        "worker_pid": int(worker_pid),
        "worker_session_id": str(worker_session_id),
        "adapter_runtime_instance_id": str(adapter_runtime_instance_id),
        "root_state_write_count": 0,
    }
    for key, value in result_expected.items():
        if not _strict_json_equal(result.get(key), value):
            raise RuntimeError(
                f"worker result {key} mismatch: "
                f"expected={value!r} actual={result.get(key)!r}"
            )
    lifecycle = dict(result.get("lifecycle", {}) or {})
    if finalized != (lifecycle.get("finalized") is True):
        raise RuntimeError("worker lifecycle finalized flag/marker mismatch")
    if failed != (lifecycle.get("failed") is True):
        raise RuntimeError("worker lifecycle failed flag/marker mismatch")
    artifact_valid = result.get("artifact_valid") is True
    if artifact_valid != finalized or artifact_valid == failed:
        raise RuntimeError("worker artifact_valid/lifecycle marker mismatch")
    if lifecycle.get("strict_success") is not bool(
        artifact_valid and result.get("strict_full_success") is True
    ):
        raise RuntimeError("worker lifecycle strict_success is inconsistent")
    if _exact_json_int(
        dict(result.get("respawn", {}) or {}).get("root_state_write_count"),
        "worker result respawn root_state_write_count",
    ) != 0:
        raise RuntimeError("worker result does not prove zero root-state writes")
    policy_valid = bool(
        dict(result.get("source_integrity", {}) or {}).get("ok") is True
        and dict(result.get("visualization", {}) or {}).get("ok") is True
        and result.get("actual_viewport_video") is True
        and dict(result.get("video", {}) or {}).get("actual_viewport_video")
        is True
        and result.get("motion_start_ready") is True
        and result.get("dispatch_complete") is True
        and str(
            result.get("wheel_target_integral_verdict", "NOT_EVALUABLE")
            or "NOT_EVALUABLE"
        ).upper()
        in {"PASS", "FAIL"}
        and _telemetry_finalization_valid(result)
    )
    if artifact_valid != policy_valid:
        raise RuntimeError("worker artifact_valid differs from re-evaluated policy")
    expected_diagnostic_complete = bool(
        equivalence_role
        and artifact_valid
        and result.get("scheduler_complete") is True
        and result.get("dispatch_complete") is True
        and dict(result.get("source_integrity", {}) or {}).get("ok") is True
    )
    ordinary_trace_valid = False
    contact_sensor_disabled = False
    ordinary_receipt: dict[str, Any] = {}
    if diagnostic_role == "U":
        from .ordinary_ui_trajectory import (
            ordinary_ui_diagnostic_complete,
            validate_ordinary_ui_trajectory,
        )

        ordinary_receipt = validate_ordinary_ui_trajectory(run_dir)
        ordinary_trace_valid = ordinary_ui_diagnostic_complete(ordinary_receipt)
        runtime_environment = _strict_json_read(run_dir / "runtime_environment.json")
        scene_config = dict(runtime_environment.get("scene_config", {}) or {})
        contact_sensor_disabled = bool(
            scene_config.get("telemetry_contact_sensors_enabled") is False
            and scene_config.get("contact_sensor_factory") is None
            and str(runtime_environment.get("contact_sensor_type", "") or "") == ""
            and str(runtime_environment.get("contact_sensor_error", "") or "") == ""
            and result.get("contact_sensor_disabled") is True
            and str(result.get("contact_sensor_type", "") or "") == ""
            and str(result.get("contact_sensor_error", "") or "") == ""
        )
        manifest = _strict_json_read(
            run_dir / "ordinary_ui_trajectory_manifest.json"
        )
        identity_envelope = dict(manifest.get("identity_envelope", {}) or {})
        readiness_path = run_dir / "motion_start_pre_first_dispatch.json"
        readiness = _strict_json_read(readiness_path)
        dispatch_path = run_dir / "V003_DISPATCH_TRACE.json"
        dispatch = _strict_json_read(dispatch_path)
        evidence = dict(identity_envelope.get("evidence", {}) or {})
        actual_evidence_hashes = {
            "durable_result": sha256_file(result_path).lower(),
            "immutable_request": str(request_sha256).lower(),
            "readiness": sha256_file(readiness_path).lower(),
            "dispatch_ledger": sha256_file(dispatch_path).lower(),
        }
        for label, actual_sha in actual_evidence_hashes.items():
            declared_sha = str(
                dict(evidence.get(label, {}) or {}).get("artifact_sha256", "")
                or ""
            ).lower()
            if declared_sha != actual_sha:
                raise RuntimeError(
                    f"ordinary-UI identity evidence {label} hash is not durable"
                )
        expected_identity = {
            "source_version": str(request["source_version"]),
            "accepted_steps_sha256": str(
                request.get(
                    "expected_accepted_steps_sha256",
                    request.get("accepted_steps_sha256", ""),
                )
            ).lower(),
            "plan_sha256": str(request["plan_sha256"]),
            "plan_id": str(request["plan_id"]),
            "request_id": str(request["request_id"]),
            "worker_session_id": str(worker_session_id),
            "adapter_runtime_instance_id": str(adapter_runtime_instance_id),
            "readiness_token_sha256": str(
                readiness.get("readiness_token_sha256", "") or ""
            ).lower(),
            "root_state_write_count": 0,
        }
        readiness_plan_identity = dict(
            readiness.get("plan_identity", {}) or {}
        )
        expected_readiness_identity = {
            "source_version": expected_identity["source_version"],
            "source_sha256": expected_identity[
                "accepted_steps_sha256"
            ],
            "plan_sha256": expected_identity["plan_sha256"],
            "plan_id": expected_identity["plan_id"],
            "request_id": expected_identity["request_id"],
            "worker_session_id": expected_identity["worker_session_id"],
        }
        for key, value in expected_readiness_identity.items():
            if not _strict_json_equal(
                readiness_plan_identity.get(key), value
            ):
                raise RuntimeError(
                    f"ordinary-UI readiness artifact {key} differs from closure"
                )
        if (
            str(
                readiness.get("adapter_runtime_instance_id", "") or ""
            )
            != expected_identity["adapter_runtime_instance_id"]
            or readiness.get("root_state_write_count") != 0
            or str(dispatch.get("source_version", "") or "")
            != expected_identity["source_version"]
            or str(
                dispatch.get("motion_start_readiness_token", "") or ""
            ).lower()
            != expected_identity["readiness_token_sha256"]
        ):
            raise RuntimeError(
                "ordinary-UI readiness/dispatch artifacts differ from closure"
            )
        for key, value in expected_identity.items():
            if not _strict_json_equal(identity_envelope.get(key), value):
                raise RuntimeError(
                    f"ordinary-UI identity envelope {key} differs from closure"
                )
        physical = dict(result.get("physical_evidence", {}) or {})
        if (
            result.get("classification") != "TRAJECTORY_DIAGNOSTIC_ONLY"
            or str(result.get("first_failure_phase", "") or "")
            or result.get("motion_start_ready") is not True
            or result.get("motion_start_readiness_scope")
            != "SENSOR_INDEPENDENT_TRAJECTORY_ADMISSION"
            or result.get("physical_motion_start_verdict")
            != "NOT_EVALUABLE"
            or
            result.get("physical_success") is not False
            or result.get("strict_full_success") is not False
            or str(result.get("full_physical_verdict", "") or "")
            != "NOT_EVALUABLE"
            or str(physical.get("full_physical_verdict", "") or "")
            != "NOT_EVALUABLE"
            or physical.get("physical_qualification_eligible") is not False
            or physical.get("environment_equivalence_eligible") is not False
        ):
            raise RuntimeError(
                "ordinary-UI diagnostic contains an ineligible physical claim"
            )
    expected_ordinary_complete = bool(
        diagnostic_role == "U"
        and artifact_valid
        and result.get("scheduler_complete") is True
        and result.get("dispatch_complete") is True
        and str(result.get("wheel_target_integral_verdict", "") or "").upper()
        == "PASS"
        and ordinary_trace_valid
        and contact_sensor_disabled
        and dict(result.get("source_integrity", {}) or {}).get("ok") is True
        and result.get("root_state_write_count") == 0
    )
    expected_diagnostic = {
        "environment_equivalence_diagnostic": bool(equivalence_role),
        "environment_equivalence_diagnostic_complete": (
            expected_diagnostic_complete
        ),
        "diagnostic_role": diagnostic_role,
        "ordinary_ui_diagnostic": diagnostic_role == "U",
        "qualification_scope": expected_scope,
        "gate1_eligible": bool(not diagnostic_role and not equivalence_role),
        "gate1_physical_qualification_eligible": bool(
            not diagnostic_role and not equivalence_role
        ),
        "environment_equivalence_eligible": bool(
            equivalence_role and not diagnostic_role
        ),
    }
    if diagnostic_role == "U":
        expected_diagnostic.update(
            ordinary_ui_diagnostic_complete=expected_ordinary_complete,
            physical_qualification_eligible=False,
            contact_sensor_disabled=True,
        )
    for key, value in expected_diagnostic.items():
        if not _strict_json_equal(result.get(key), value):
            raise RuntimeError(
                f"worker result diagnostic {key} mismatch: "
                f"expected={value!r} actual={result.get(key)!r}"
            )
    summary_fields = [
        "artifact_valid",
        "classification",
        "scheduler_complete",
        "physical_success",
        "strict_full_success",
        "environment_equivalence_diagnostic_complete",
    ]
    if diagnostic_role == "U":
        summary_fields.append("ordinary_ui_diagnostic_complete")
        if not _strict_json_equal(
            ack.get("ordinary_ui_trajectory_receipt"), ordinary_receipt
        ):
            raise RuntimeError(
                "worker ACK ordinary-UI trajectory receipt differs from strict readback"
            )
    for key in summary_fields:
        if not _strict_json_equal(ack.get(key), result.get(key)):
            raise RuntimeError(f"worker ACK/result summary mismatch: {key}")

    checksum_path = resolved_paths["checksums_path"]
    checksum_rows = _parse_checksum_manifest(checksum_path)
    required = list(result.get("required_evidence_paths", []) or [])
    if not required:
        raise RuntimeError("worker result required_evidence_paths is empty")
    for raw in required:
        path = Path(str(raw)).resolve()
        if not path.is_file() or not _path_is_within(path, artifact_root):
            raise RuntimeError(f"worker required evidence missing/escaping: {path}")
        if path == checksum_path:
            continue
        relative = path.relative_to(run_dir).as_posix() if _path_is_within(path, run_dir) else path.relative_to(artifact_root).as_posix()
        manifest_relative = (
            relative
            if _path_is_within(path, run_dir)
            else relative
        )
        # Run checksums are rooted at run_dir.  Artifact-root control files
        # are authenticated directly by the ACK and are not silently treated
        # as run checksum rows.
        if _path_is_within(path, run_dir):
            actual = sha256_file(path).lower()
            if checksum_rows.get(manifest_relative) != actual:
                raise RuntimeError(
                    f"worker checksum omits/mismatches required evidence: {path}"
                )
    return result


def _validate_recording_worker_ready_status(
    status: dict[str, Any],
    *,
    request: dict[str, Any],
    request_sha256: str,
    worker_pid: int,
) -> dict[str, Any]:
    """Bind the exact production worker before any playback request is sent."""

    session = dict(status.get("worker_artifact_session", {}) or {})
    preflight = dict(status.get("worker_artifact_preflight", {}) or {})
    if status.get("ready") is not True or status.get("artifact_preflight_ready") is not True:
        raise RuntimeError("formal worker artifact preflight is not ready")
    expected = {
        "request_id": str(request["request_id"]),
        "source_version": str(request["source_version"]),
        "trial_id": int(request["trial_id"]),
        "contact_mode": str(request["contact_mode"]),
        "environment_equivalence_role": str(
            request.get("environment_equivalence_role", "") or ""
        ),
        "diagnostic_role": str(request.get("diagnostic_role", "") or ""),
        "qualification_scope": str(request.get("qualification_scope", "") or ""),
        "gate1_eligible": request.get("gate1_eligible"),
        "gate1_physical_qualification_eligible": request.get(
            "gate1_physical_qualification_eligible"
        ),
        "environment_equivalence_eligible": request.get(
            "environment_equivalence_eligible"
        ),
        "artifact_root": str(Path(str(request["artifact_root"])).resolve()),
    }
    for key, value in expected.items():
        actual = session.get(key)
        if key == "artifact_root" and actual:
            actual = str(Path(str(actual)).resolve())
        if not _strict_json_equal(actual, value):
            raise RuntimeError(
                f"formal worker ready status {key} mismatch: "
                f"expected={value!r} actual={actual!r}"
            )
    if str(preflight.get("artifact_request_sha256", "") or "").lower() != str(
        request_sha256
    ).lower():
        raise RuntimeError("formal worker did not bind the exact artifact request bytes")
    if str(preflight.get("expected_plan_sha256", "") or "") != str(
        request["plan_sha256"]
    ):
        raise RuntimeError("formal worker preflight plan SHA mismatch")
    if str(
        preflight.get("environment_equivalence_role", "") or ""
    ) != str(request.get("environment_equivalence_role", "") or ""):
        raise RuntimeError(
            "formal worker preflight environment-equivalence role mismatch"
        )
    for key in (
        "diagnostic_role",
        "qualification_scope",
        "gate1_eligible",
        "gate1_physical_qualification_eligible",
        "environment_equivalence_eligible",
    ):
        if not _strict_json_equal(preflight.get(key), request.get(key)):
            raise RuntimeError(f"formal worker preflight {key} mismatch")
    if _exact_json_int(status.get("worker_pid"), "worker ready PID") != _exact_json_int(
        worker_pid, "launched worker PID"
    ):
        raise RuntimeError("formal worker status PID differs from the launched process")
    worker_session_id = str(status.get("worker_session_id", "") or "")
    if not worker_session_id or worker_session_id != str(
        session.get("worker_session_id", "") or ""
    ):
        raise RuntimeError("formal worker session identity is missing or inconsistent")
    adapter_id = str(session.get("adapter_runtime_instance_id", "") or "")
    if not adapter_id:
        raise RuntimeError("formal worker adapter runtime identity is missing")
    if _exact_json_int(
        session.get("root_state_write_count"),
        "worker ready root_state_write_count",
    ) != _exact_json_int(
        request.get("expected_root_state_write_count"),
        "worker request expected_root_state_write_count",
    ):
        raise RuntimeError("formal worker performed or misreported a root-state write")
    if session.get("motion_start_ready") is not True:
        raise RuntimeError("formal worker 10-frame motion-start preflight is not ready")
    if _exact_json_int(
        session.get("readiness_frame_count"), "worker readiness frame_count"
    ) < _exact_json_int(
        session.get("readiness_frame_count_required"),
        "worker readiness required frame_count",
    ):
        raise RuntimeError("formal worker readiness window is incomplete")
    if _exact_json_int(
        session.get("readiness_sample_stride_physics_ticks"),
        "worker readiness sample stride",
    ) != 8:
        raise RuntimeError("formal worker readiness cadence differs from render_interval=8")
    environment = dict(session.get("environment_equivalence", {}) or {})
    if environment.get("ok") is not True or list(environment.get("failed_checks", []) or []):
        raise RuntimeError("formal worker runtime environment equivalence failed")
    diagnostic_role = str(request.get("diagnostic_role", "") or "")
    if str(session.get("contact_sensor_error", "") or ""):
        raise RuntimeError("formal worker contact sensor has an error")
    sensor_type = str(session.get("contact_sensor_type", "") or "")
    role = str(request.get("environment_equivalence_role", "") or "")
    expected_sensor_type = (
        ""
        if diagnostic_role == "U"
        else "ContactSensor"
        if role in {"A1", "A2"}
        else "WheelAndNonWheelContactSensorBank"
    )
    if sensor_type != expected_sensor_type:
        raise RuntimeError(
            "formal worker contact sensor role mismatch: "
            f"expected={expected_sensor_type} actual={sensor_type or '<missing>'}"
        )
    return {
        "worker_pid": int(worker_pid),
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_id,
        "artifact_request_id": str(request["request_id"]),
        "artifact_request_sha256": str(request_sha256),
        "contact_mode": str(request.get("contact_mode", "") or ""),
        "environment_equivalence_role": str(
            request.get("environment_equivalence_role", "") or ""
        ),
        "diagnostic_role": str(request.get("diagnostic_role", "") or ""),
        "qualification_scope": str(request.get("qualification_scope", "") or ""),
        "gate1_eligible": request.get("gate1_eligible"),
        "gate1_physical_qualification_eligible": request.get(
            "gate1_physical_qualification_eligible"
        ),
        "environment_equivalence_eligible": request.get(
            "environment_equivalence_eligible"
        ),
        "status": _jsonable(status),
    }


def _wait_for_recording_worker_ready(
    client: Any,
    *,
    request: dict[str, Any],
    request_sha256: str,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        client.poll()
        last_status = dict(client.status() or {})
        artifact_acks = list(last_status.get("artifact_ack_history", []) or [])
        latest_artifact_ack = dict(last_status.get("last_artifact_ack", {}) or {})
        if latest_artifact_ack:
            artifact_acks.append(latest_artifact_ack)
        for row in reversed(artifact_acks):
            ack = dict(row or {})
            if (
                str(ack.get("type", "") or "") == "operation_ack"
                and str(ack.get("operation", "") or "")
                == "recording_artifact"
                and str(ack.get("request_id", "") or "")
                == str(request.get("request_id", "") or "")
                and str(ack.get("phase", "") or "") == "ARTIFACT_FAILED"
                and ack.get("accepted") is False
                and ack.get("artifact_complete") is False
            ):
                raise RuntimeError(
                    "formal worker artifact preflight failed: "
                    + str(ack.get("error", "") or ack)
                )
        artifact_session = dict(
            last_status.get("worker_artifact_session", {}) or {}
        )
        if (
            str(artifact_session.get("request_id", "") or "")
            == str(request.get("request_id", "") or "")
            and artifact_session.get("terminal") is True
            and str(artifact_session.get("state", "") or "") == "failed"
        ):
            raise RuntimeError(
                "formal worker artifact preflight failed: "
                + str(artifact_session.get("error", "") or artifact_session)
            )
        process = getattr(client, "process", None)
        returncode = None if process is None else process.poll()
        if returncode is not None:
            raise RuntimeError(
                "formal worker exited before artifact preflight completed: "
                f"returncode={returncode} status={last_status}"
            )
        if (
            last_status.get("ready") is True
            and last_status.get("artifact_preflight_ready") is True
        ):
            pid = int(getattr(client, "pid", 0) or 0)
            if pid <= 0:
                raise RuntimeError("formal worker has no launched process PID")
            return _validate_recording_worker_ready_status(
                last_status,
                request=request,
                request_sha256=request_sha256,
                worker_pid=pid,
            )
        error = str(last_status.get("error", "") or "")
        if error and last_status.get("starting") is not True:
            raise RuntimeError(f"formal worker startup failed: {error}")
        time.sleep(max(0.005, min(0.25, float(poll_interval_s))))
    raise RuntimeError(
        "formal worker artifact preflight timed out: " + repr(last_status)
    )


def _wait_for_worker_operation_ack(
    client: Any,
    *,
    operation: str,
    request_id: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    """Find one exact critical operation ACK without guessing from latest status."""

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        client.poll()
        status = dict(client.status() or {})
        history = list(status.get("operation_ack_history", []) or [])
        latest = dict(status.get("last_operation_ack", {}) or {})
        if latest:
            history.append(latest)
        for row in reversed(history):
            ack = dict(row or {})
            if (
                str(ack.get("operation", "") or "") == str(operation)
                and str(ack.get("request_id", "") or "") == str(request_id)
            ):
                return ack
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"formal worker exited before {operation} ACK: "
                f"returncode={process.poll()}"
            )
        time.sleep(max(0.005, min(0.10, float(poll_interval_s))))
    raise RuntimeError(
        f"timed out waiting for formal worker {operation} ACK request={request_id}"
    )


def _wait_for_worker_stop_ack(
    client: Any,
    *,
    command_id: str,
    worker_pid: int,
    worker_session_id: str,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.02,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        client.poll()
        status = dict(client.status() or {})
        history = list(status.get("operation_ack_history", []) or [])
        latest = dict(status.get("last_operation_ack", {}) or {})
        if latest:
            history.append(latest)
        for row in reversed(history):
            ack = dict(row or {})
            if (
                str(ack.get("type", "") or "") == "stop_ack"
                and str(ack.get("command_id", "") or "") == str(command_id)
            ):
                if ack.get("zero_target_applied") is not True or str(
                    ack.get("error", "") or ""
                ):
                    raise RuntimeError("formal worker pre-play stop was not applied")
                if int(ack.get("worker_pid", 0) or 0) != int(worker_pid):
                    raise RuntimeError("formal worker pre-play stop PID mismatch")
                if str(ack.get("worker_session_id", "") or "") != str(
                    worker_session_id
                ):
                    raise RuntimeError("formal worker pre-play stop session mismatch")
                if int(ack.get("root_state_write_count", -1)) != 0:
                    raise RuntimeError("formal worker pre-play stop reports a root write")
                return ack
        process = getattr(client, "process", None)
        if process is not None and process.poll() is not None:
            raise RuntimeError("formal worker exited before pre-play stop ACK")
        time.sleep(max(0.005, min(0.10, float(poll_interval_s))))
    raise RuntimeError("timed out waiting for formal worker pre-play stop ACK")


def _deserialize_replay_args(payload: dict[str, Any]) -> argparse.Namespace:
    values = dict(payload)
    for key in (
        "recording_root",
        "report_root",
        "output_root",
        "config",
        "environment_report",
        "sim_state_before",
        "prefix_manifest",
    ):
        if key in values:
            values[key] = Path(str(values[key]))
    return argparse.Namespace(**values)


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (completed.stdout or completed.stderr).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain=v1"),
    }


def _source_files(robot_usd: Path | None = None) -> list[Path]:
    # Top-level Python modules are the replay/adapter command closure.  The
    # telemetry tree is included recursively because its support/contact
    # interpretation participates directly in strict success.
    candidates = list(PROJECT_ROOT.glob("*.py"))
    candidates.extend(sorted((PROJECT_ROOT / "telemetry").rglob("*.py")))
    candidates.extend(sorted((PROJECT_ROOT / "config").rglob("*.yaml")))
    candidates.extend(sorted((PROJECT_ROOT / "config").rglob("*.json")))
    candidates.extend(sorted(MODULE_ROOT.glob("*.py")))
    candidates.extend(sorted(MODULE_ROOT.glob("*.yaml")))
    if robot_usd is not None:
        candidates.append(Path(robot_usd).resolve())
    return [path.resolve() for path in candidates if path.is_file()]


def _source_freeze(
    item: VersionFiles | None = None,
    *,
    robot_usd: Path | None = None,
) -> dict[str, Any]:
    files = _source_files(robot_usd)
    if robot_usd is not None and not Path(robot_usd).resolve().is_file():
        raise FileNotFoundError(f"robot USD is absent from the source closure: {robot_usd}")
    if item is not None:
        files.extend([item.steps_path.resolve(), item.metadata_path.resolve()])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_snapshot(),
        "files": {
            str(path): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(set(files))
        },
    }


def _compare_source_freezes(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_files = dict(before.get("files", {}) or {})
    after_files = dict(after.get("files", {}) or {})
    before_names = set(before_files)
    after_names = set(after_files)
    changed = [
        name
        for name in sorted(before_names & after_names)
        if dict(before_files[name]) != dict(after_files[name])
    ]
    missing = sorted(before_names - after_names)
    added = sorted(after_names - before_names)
    return {
        "equal": not changed and not missing and not added,
        "changed": changed,
        "missing": missing,
        "added": added,
        "before_created_utc": before.get("created_utc"),
        "after_created_utc": after.get("created_utc"),
    }


def _fail_closed_recording_audits(
    audit: RecordingAudit,
    selected: Iterable[VersionFiles],
) -> list[dict[str, Any]]:
    """Recompute every selected accepted-steps hash before Isaac starts."""

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in selected:
        try:
            row = audit.audit_version(item)
        except Exception as exc:
            raise RuntimeError(
                f"recording preflight audit failed closed for {item.version_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        rows.append(row)
        reasons: list[str] = []
        if not bool(row.get("valid", False)):
            reasons.append("audit_version.valid is false")
        if not bool(row.get("metadata_sha256_matches", False)):
            reasons.append("accepted_steps hash differs from metadata")
        if list(row.get("missing_required_files", []) or []):
            reasons.append("required recording files are missing")
        actual_hash = str(row.get("accepted_steps_sha256", "") or "").lower()
        try:
            immediate_hash = sha256_file(item.steps_path).lower()
        except Exception as exc:
            reasons.append(f"accepted_steps cannot be rehashed: {exc}")
        else:
            if not actual_hash or immediate_hash != actual_hash:
                reasons.append("accepted_steps changed during preflight audit")
        if int(row.get("actual_step_count", 0) or 0) <= 0:
            reasons.append("accepted_steps is empty")
        if reasons:
            failures.append(f"{item.version_id}: " + "; ".join(reasons))
    if failures:
        raise RuntimeError("recording preflight rejected: " + " | ".join(failures))
    return rows


_RELIABLE_REPLAY_CLASSIFICATIONS = frozenset(
    {
        "FULL_SUCCESS",
        "PARTIAL_SUCCESS",
        "PHYSICAL_FAILURE",
        "SCHEDULER_FAILURE",
        "WHEEL_INTEGRAL_FAILURE",
    }
)

_ACTUAL_VIEWPORT_VIDEO_SOURCE = "actual_active_isaac_gui_viewport_render_product"


def _recording_video_capture_requested(args: argparse.Namespace) -> bool:
    return bool(
        not bool(getattr(args, "headless", False))
        and not bool(getattr(args, "no_video", False))
    )


def _recording_video_disabled_reason(args: argparse.Namespace) -> str:
    if bool(getattr(args, "no_video", False)):
        return "--no-video diagnostic mode disables reliable/strict completion"
    if bool(getattr(args, "headless", False)):
        return "headless diagnostic mode has no active GUI viewport"
    return ""


def _mp4_has_container_signature(path: Path) -> bool:
    try:
        header = Path(path).read_bytes()[:32]
    except OSError:
        return False
    return bool(len(header) >= 12 and header[4:8] == b"ftyp")


def _finalize_recording_viewport_video_contract(
    run_dir: Path,
    raw_video: dict[str, Any],
    *,
    contact_mode: str,
    capture_requested: bool,
    disabled_reason: str = "",
    capture_error: str = "",
) -> dict[str, Any]:
    """Validate and persist one per-version active-viewport video manifest."""

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = dict(raw_video or {})
    errors: list[str] = []
    for reason in (disabled_reason, capture_error, str(raw.get("error", "") or "")):
        if reason and reason not in errors:
            errors.append(reason)
    video_text = str(raw.get("video_path", "") or "")
    video_path = (
        Path(video_text).resolve()
        if video_text
        else (run_dir / "actual_viewport_video.mp4").resolve()
    )
    video_exists = bool(
        video_path.is_file()
        and video_path.stat().st_size > 0
        and video_path.suffix.lower() == ".mp4"
        and _path_is_within(video_path, run_dir)
    )
    video_sha256 = sha256_file(video_path).lower() if video_exists else ""
    claimed_sha256 = str(raw.get("video_sha256", "") or "").lower()
    try:
        frame_count = int(raw.get("frame_count", 0) or 0)
    except (TypeError, ValueError):
        frame_count = 0
    if not capture_requested:
        errors.append("viewport video capture was not requested")
    if not bool(raw.get("valid", False)):
        errors.append("viewport recorder did not finalize a valid video")
    if bool(raw.get("not_camera_video", False)):
        errors.append("telemetry visualization is not camera video")
    if str(raw.get("source", "") or "") != _ACTUAL_VIEWPORT_VIDEO_SOURCE:
        errors.append("video source is not the active GUI viewport render product")
    if str(raw.get("capture_backend", "") or "") != ACTIVE_VIEWPORT_BUFFER_BACKEND:
        errors.append("viewport video was not produced by the direct LdrColor buffer backend")
    if not str(raw.get("render_product_path", "") or ""):
        errors.append("active viewport render product path is missing")
    if raw.get("render_product_unchanged") is not True:
        errors.append("active viewport render product identity was not preserved")
    if raw.get("active_render_product_identity_proven") is not True:
        errors.append("active viewport identity evidence is incomplete")
    if raw.get("capture_graph_created") is not False:
        errors.append("a viewport capture graph was created")
    if raw.get("render_observer_only") is not True:
        errors.append("viewport capture was not an existing-render observer")
    if raw.get("extra_app_update_count") != 0:
        errors.append("viewport capture invoked extra app.update calls")
    if raw.get("extra_render_count") != 0:
        errors.append("viewport capture invoked extra renders")
    if raw.get("maximum_pending_captures") != 1:
        errors.append("viewport capture did not enforce one pending buffer")
    if frame_count < 2:
        errors.append("fewer than two viewport frames were captured")

    ledger_text = str(raw.get("ledger_path", "") or "")
    ledger_path = Path(ledger_text).resolve() if ledger_text else None
    ledger_rows: list[dict[str, Any]] = []
    ledger_sha256 = ""
    if (
        ledger_path is None
        or not ledger_path.is_file()
        or not _path_is_within(ledger_path, run_dir)
        or ledger_path.suffix.lower() != ".jsonl"
    ):
        errors.append("viewport frame ledger is missing or outside the run")
    else:
        ledger_sha256 = sha256_file(ledger_path).lower()
        claimed_ledger_sha256 = str(raw.get("ledger_sha256", "") or "").lower()
        if not claimed_ledger_sha256 or claimed_ledger_sha256 != ledger_sha256:
            errors.append("viewport frame ledger SHA256 is missing or mismatched")
        try:
            for line_number, line in enumerate(
                ledger_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    raise ValueError(f"blank line {line_number}")
                decoded = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite constant {value}")
                    ),
                )
                if not isinstance(decoded, dict):
                    raise ValueError(f"line {line_number} is not an object")
                ledger_rows.append(decoded)
        except Exception as exc:
            errors.append(
                "viewport frame ledger is not valid JSONL: "
                f"{type(exc).__name__}: {exc}"
            )
    if len(ledger_rows) != frame_count:
        errors.append("viewport frame ledger count does not match frame_count")
    elif ledger_rows:
        if [row.get("render_sequence") for row in ledger_rows] != list(
            range(frame_count)
        ):
            errors.append("viewport frame ledger render sequence is not contiguous")
        if [row.get("encoded_frame_index") for row in ledger_rows] != list(
            range(frame_count)
        ):
            errors.append("viewport frame ledger encoded sequence is not contiguous")
        if any(
            str(row.get("capture_backend", "") or "")
            != ACTIVE_VIEWPORT_BUFFER_BACKEND
            or str(row.get("render_product_path", "") or "")
            != str(raw.get("render_product_path", "") or "")
            for row in ledger_rows
        ):
            errors.append("viewport frame ledger backend/render product is inconsistent")
    if raw.get("frame_ledger_complete") is not True:
        errors.append("viewport frame ledger was not finalized as complete")

    full_decode = dict(raw.get("full_decode", {}) or {})
    try:
        decoded_frame_count = int(full_decode.get("decoded_frame_count", -1))
        decoded_width = int(full_decode.get("decoded_width", 0))
        decoded_height = int(full_decode.get("decoded_height", 0))
        decoded_channels = int(full_decode.get("decoded_channels", 0))
    except (TypeError, ValueError):
        decoded_frame_count = -1
        decoded_width = decoded_height = decoded_channels = 0
    if (
        raw.get("full_decode_all_frames") is not True
        or full_decode.get("valid") is not True
        or decoded_frame_count != frame_count
        or decoded_width < 1
        or decoded_height < 1
        or decoded_channels not in (3, 4)
    ):
        errors.append("full MP4 decode does not exactly match the frame ledger")

    for checkpoint_name in ("first_frame", "last_frame"):
        checkpoint_text = str(raw.get(f"{checkpoint_name}_path", "") or "")
        checkpoint_path = Path(checkpoint_text).resolve() if checkpoint_text else None
        checkpoint_exists = bool(
            checkpoint_path is not None
            and checkpoint_path.is_file()
            and checkpoint_path.stat().st_size > 0
            and _path_is_within(checkpoint_path, run_dir)
            and checkpoint_path.suffix.lower() == ".png"
        )
        claimed_checkpoint_sha = str(
            raw.get(f"{checkpoint_name}_sha256", "") or ""
        ).lower()
        if (
            not checkpoint_exists
            or not claimed_checkpoint_sha
            or claimed_checkpoint_sha != sha256_file(checkpoint_path).lower()
        ):
            errors.append(f"viewport {checkpoint_name} checkpoint is invalid")
    if not video_exists:
        errors.append("viewport MP4 is missing, empty, outside the run, or not MP4")
    elif not _mp4_has_container_signature(video_path):
        errors.append("viewport MP4 container signature is invalid")
    if not claimed_sha256 or claimed_sha256 != video_sha256:
        errors.append("viewport MP4 SHA256 is missing or mismatched")
    errors = list(dict.fromkeys(errors))
    actual_viewport_video = not errors
    manifest_path = run_dir / "viewport_video_manifest.json"
    manifest = {
        **_jsonable(raw),
        "schema_version": "fsm50.recording_viewport_video.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contact_mode": str(contact_mode),
        "capture_requested": bool(capture_requested),
        "diagnostic_only": not bool(capture_requested),
        "valid": actual_viewport_video,
        "artifact_valid": actual_viewport_video,
        "actual_viewport_video": actual_viewport_video,
        "not_camera_video": False,
        "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
        "frame_count": frame_count,
        "ledger_path": "" if ledger_path is None else str(ledger_path),
        "ledger_sha256": ledger_sha256,
        "video_path": str(video_path),
        "video_sha256": video_sha256,
        "error": "; ".join(errors),
    }
    _atomic_write_json(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path).lower(),
    }


class _RecordingReplayViewportCapture:
    """Per-version direct-buffer observer of the adapter's existing renders.

    The recorder never rebinds the active viewport, creates a post-process
    graph, calls ``app.update``, or requests a render.  It is attached only to
    :class:`SimRobotAdapter`'s already-required render hook and is detached
    before its encoder and evidence files are finalized.
    """

    def __init__(
        self,
        run_dir: Path,
        args: argparse.Namespace,
        *,
        contact_mode: str,
        adapter: Any | None = None,
        recorder: Any | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.args = args
        self.contact_mode = str(contact_mode)
        self.capture_requested = _recording_video_capture_requested(args)
        if recorder is None:
            recorder = ActiveViewportBufferVideoRecorder(
                self.run_dir,
                enabled=self.capture_requested,
                fps=float(getattr(args, "video_fps", 15.0)),
            )
        self.recorder = recorder
        self.adapter = adapter
        self.original_render_product_path = ""
        self.observed_render_product_path = ""
        self.observer_attached = False
        self.capture_error = ""
        self._finalized: dict[str, Any] | None = None

    def start(self) -> None:
        try:
            started = self.recorder.start()
        except Exception as exc:
            started = False
            self.capture_error = self.capture_error or (
                f"viewport recorder start failed: {type(exc).__name__}: {exc}"
            )
        if not self.capture_requested:
            return
        self.original_render_product_path = str(
            getattr(self.recorder, "render_product_path", "") or ""
        )
        self.observed_render_product_path = self.original_render_product_path
        if started is False or str(getattr(self.recorder, "error", "") or ""):
            self.capture_error = self.capture_error or (
                "direct viewport recorder did not start successfully"
            )
            return
        if not self.original_render_product_path:
            self.capture_error = self.capture_error or (
                "active viewport render product path is unavailable"
            )
            return
        if self.adapter is None:
            self.capture_error = self.capture_error or (
                "adapter render hook is unavailable"
            )
            return
        try:
            self.adapter.attach_artifact_render_observer(self.recorder)
            self.observer_attached = True
        except Exception as exc:
            self.capture_error = self.capture_error or (
                "viewport observer attach failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _detach_observer(self) -> str:
        if not self.observer_attached:
            return ""
        try:
            if self.adapter is None:
                raise RuntimeError("attached viewport observer lost its adapter")
            self.adapter.detach_artifact_render_observer(self.recorder)
            return ""
        except Exception as exc:
            return f"viewport observer detach failed: {type(exc).__name__}: {exc}"
        finally:
            self.observer_attached = False

    def finalize(self) -> dict[str, Any]:
        if self._finalized is not None:
            return dict(self._finalized)
        detach_error = ""
        raw: dict[str, Any]
        try:
            detach_error = self._detach_observer()
        finally:
            try:
                raw = dict(self.recorder.finalize() or {})
            except Exception as exc:
                raw = {
                    "valid": False,
                    "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
                    "video_path": str(self.run_dir / "actual_viewport_video.mp4"),
                    "video_sha256": "",
                    "frame_count": 0,
                    "error": (
                        "viewport recorder finalize failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
        observed_path = str(raw.get("render_product_path", "") or "")
        if (
            self.capture_requested
            and self.original_render_product_path
            and observed_path != self.original_render_product_path
        ):
            self.capture_error = self.capture_error or (
                "active viewport render product identity changed during capture"
            )
        combined_error = "; ".join(
            reason
            for reason in (self.capture_error, detach_error)
            if reason
        )
        self._finalized = _finalize_recording_viewport_video_contract(
            self.run_dir,
            raw,
            contact_mode=self.contact_mode,
            capture_requested=self.capture_requested,
            disabled_reason=_recording_video_disabled_reason(self.args),
            capture_error=combined_error,
        )
        return dict(self._finalized)


def _missing_recording_viewport_video(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    contact_mode: str,
    error: str,
) -> dict[str, Any]:
    return _finalize_recording_viewport_video_contract(
        run_dir,
        {
            "valid": False,
            "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
            "render_product_path": "",
            "frame_count": 0,
            "video_path": str(
                Path(run_dir).resolve() / "actual_viewport_video.mp4"
            ),
            "video_sha256": "",
            "error": str(error),
        },
        contact_mode=contact_mode,
        capture_requested=_recording_video_capture_requested(args),
        disabled_reason=_recording_video_disabled_reason(args),
    )


def _recording_visual_manifest(
    *,
    video: dict[str, Any],
    visualization: dict[str, Any],
    contact_mode: str,
    artifact_valid: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "fsm50.recording_visual_evidence.v2",
        "kind": "actual_active_gui_viewport_video_with_telemetry_visualization",
        "contact_mode": str(contact_mode),
        "actual_viewport_video": bool(video.get("actual_viewport_video", False)),
        "not_camera_video": False,
        "video_path": str(video.get("video_path", "") or ""),
        "video_sha256": str(video.get("video_sha256", "") or ""),
        "viewport_video_manifest_path": str(video.get("manifest_path", "") or ""),
        "viewport_video_manifest_sha256": str(
            video.get("manifest_sha256", "") or ""
        ),
        "video": _jsonable(video),
        "telemetry_visualization": _jsonable(visualization),
        "artifact_valid": bool(artifact_valid),
        "basis": [
            "actual active Isaac GUI viewport frames",
            "fsm50_telemetry.csv",
            "state_timeline.csv",
            "result.json strict result fields",
        ],
    }


def _apply_recording_artifact_policy(
    result: dict[str, Any],
    *,
    video: dict[str, Any],
    visualization: dict[str, Any],
) -> bool:
    """Bind strict/finalized status to source, visualization, and real video."""

    source_ok = bool(dict(result.get("source_integrity", {}) or {}).get("ok", False))
    visualization_ok = bool(visualization.get("ok", False))
    video_ok = bool(video.get("actual_viewport_video", False))
    motion_start_ok = bool(result.get("motion_start_ready") is True)
    dispatch_ok = bool(result.get("dispatch_complete") is True)
    wheel_integral_verdict = str(
        result.get("wheel_target_integral_verdict", "NOT_EVALUABLE")
        or "NOT_EVALUABLE"
    ).upper()
    wheel_integral_evaluable = wheel_integral_verdict in {"PASS", "FAIL"}
    telemetry_ok = _telemetry_finalization_valid(result)
    artifact_valid = bool(
        source_ok
        and visualization_ok
        and video_ok
        and motion_start_ok
        and dispatch_ok
        and wheel_integral_evaluable
        and telemetry_ok
    )
    if not artifact_valid and source_ok:
        result["classification_before_artifact_validation"] = result.get(
            "classification"
        )
        result["classification"] = "ARTIFACT_INVALID"
        result["first_failure_phase"] = (
            "VIEWPORT_VIDEO_MISSING_OR_INVALID"
            if not video_ok
            else "MOTION_START_EVIDENCE_INVALID"
            if not motion_start_ok
            else "SOURCE_DISPATCH_LEDGER_INCOMPLETE"
            if not dispatch_ok
            else "WHEEL_TARGET_INTEGRAL_NOT_EVALUABLE"
            if not wheel_integral_evaluable
            else "TELEMETRY_FINALIZATION_INCOMPLETE"
            if not telemetry_ok
            else "VISUALIZATION_FAILED"
        )
        result["strict_full_success"] = False
        result["strict_success"] = False
    result["artifact_valid"] = artifact_valid
    result["lifecycle"] = {
        "finalized": artifact_valid,
        "failed": not artifact_valid,
        "strict_success": bool(
            result.get("strict_full_success", False) and artifact_valid
        ),
    }
    return artifact_valid


def _telemetry_finalization_valid(result: dict[str, Any]) -> bool:
    finalization = dict(result.get("telemetry_finalization", {}) or {})
    journal = dict(finalization.get("journal", {}) or {})
    if (
        str(finalization.get("schema_version", ""))
        != "telemetry.canonical_finalization.v1"
        or finalization.get("canonical_export_attempted") is not True
        or finalization.get("canonical_complete") is not True
        or list(finalization.get("errors", []) or [])
        or journal.get("removed_after_success") is not True
        or list(journal.get("errors", []) or [])
    ):
        return False
    run_dir_text = str(result.get("run_dir", "") or "")
    marker_text = str(finalization.get("marker_path", "") or "")
    marker_sha = str(finalization.get("marker_sha256", "") or "").lower()
    if not run_dir_text or not marker_text or len(marker_sha) != 64:
        return False
    run_dir = Path(run_dir_text).resolve()
    marker_path = Path(marker_text).resolve()
    if marker_path != run_dir / "telemetry_finalization.json":
        return False
    if (
        not marker_path.is_file()
        or sha256_file(marker_path).lower() != marker_sha
        or (run_dir / ".telemetry_journal").exists()
    ):
        return False
    try:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value}")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            decoded: dict[str, Any] = {}
            for key, value in pairs:
                if key in decoded:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                decoded[key] = value
            return decoded

        marker = json.loads(
            marker_path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(marker, dict):
        return False
    for key in (
        "schema_version",
        "canonical_export_attempted",
        "canonical_complete",
        "stream_counts",
        "canonical_files",
        "journal",
        "errors",
    ):
        if finalization.get(key) != marker.get(key):
            return False
    return True


def _reliable_recording_video_files(
    result: dict[str, Any], run_dir: Path
) -> tuple[Path, ...] | None:
    """Return required video evidence files only for a coherent real capture."""

    run_dir = Path(run_dir).resolve()
    contact_mode = str(result.get("contact_mode", "") or "")
    video = dict(result.get("video", {}) or {})
    if (
        contact_mode not in {"formal", "instrumented"}
        or result.get("actual_viewport_video") is not True
        or video.get("actual_viewport_video") is not True
        or video.get("valid") is not True
        or video.get("diagnostic_only") is True
        or video.get("not_camera_video") is True
    ):
        return None
    video_path = Path(str(video.get("video_path", "") or "")).resolve()
    manifest_path = Path(str(video.get("manifest_path", "") or "")).resolve()
    runtime_path = run_dir / "runtime_environment.json"
    visual_path = run_dir / "visual_recording_manifest.json"
    if (
        manifest_path != run_dir / "viewport_video_manifest.json"
        or not video_path.is_file()
        or not manifest_path.is_file()
        or not runtime_path.is_file()
        or not visual_path.is_file()
        or not _path_is_within(video_path, run_dir)
        or not _mp4_has_container_signature(video_path)
    ):
        return None
    actual_video_sha = sha256_file(video_path).lower()
    actual_manifest_sha = sha256_file(manifest_path).lower()
    if (
        str(video.get("video_sha256", "") or "").lower() != actual_video_sha
        or str(video.get("manifest_sha256", "") or "").lower()
        != actual_manifest_sha
        or str(result.get("video_path", "") or "") != str(video_path)
        or str(result.get("video_sha256", "") or "").lower() != actual_video_sha
    ):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    for row in (manifest, runtime, visual):
        if (
            not isinstance(row, dict)
            or str(row.get("contact_mode", "") or "") != contact_mode
            or row.get("actual_viewport_video") is not True
            or str(row.get("video_path", "") or "") != str(video_path)
            or str(row.get("video_sha256", "") or "").lower()
            != actual_video_sha
        ):
            return None
    if (
        manifest.get("valid") is not True
        or manifest.get("not_camera_video") is True
        or str(manifest.get("source", "") or "")
        != _ACTUAL_VIEWPORT_VIDEO_SOURCE
        or int(manifest.get("frame_count", 0) or 0) < 2
    ):
        return None
    return (manifest_path, runtime_path, visual_path, video_path)


def _checksums_match_required_files(
    run_dir: Path,
    required_files: Iterable[Path],
) -> bool:
    """Return true only when the run checksum manifest covers current files."""

    manifest_path = Path(run_dir).resolve() / "checksums.sha256"
    if not manifest_path.is_file():
        return False
    manifest: dict[str, str] = {}
    try:
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, separator, relative = raw_line.partition("  ")
            digest = digest.strip().lower()
            relative = relative.strip().replace("\\", "/")
            if (
                not separator
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not relative
            ):
                return False
            manifest[relative] = digest
        for path in required_files:
            resolved = Path(path).resolve()
            relative = resolved.relative_to(Path(run_dir).resolve()).as_posix()
            if manifest.get(relative) != sha256_file(resolved).lower():
                return False
    except (OSError, UnicodeError, ValueError):
        return False
    return True


def _reliable_replay_completion(
    result_path: Path,
    *,
    output_root: Path,
    expected_version: str,
    expected_steps_sha256: str,
) -> dict[str, Any] | None:
    """Validate one durable, source-matched, version-level completion.

    A physical or scheduler failure is still a completed experiment and may be
    resumed past.  Runner/source/artifact failures are intentionally not: they
    retain their CRASH diagnostics but must be retried.  In particular, a
    directory carrying ``.partial`` can never satisfy this predicate.
    """

    output_root = Path(output_root).resolve()
    result_path = Path(result_path).resolve()
    if not result_path.is_file() or not _path_is_within(result_path, output_root):
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None
    if str(result.get("schema_version", "")) != "fsm50.recording_replay_result.v1":
        return None
    if str(result.get("source_version", "")) != str(expected_version):
        return None
    if str(result.get("accepted_steps_sha256", "")).lower() != str(
        expected_steps_sha256
    ).lower():
        return None
    if str(result.get("classification", "")) not in _RELIABLE_REPLAY_CLASSIFICATIONS:
        return None
    if not bool(result.get("artifact_valid", False)):
        return None
    lifecycle = dict(result.get("lifecycle", {}) or {})
    if not bool(lifecycle.get("finalized", False)) or bool(
        lifecycle.get("failed", False)
    ):
        return None
    if not bool(dict(result.get("source_integrity", {}) or {}).get("ok", False)):
        return None
    if not bool(dict(result.get("visualization", {}) or {}).get("ok", False)):
        return None
    if not _telemetry_finalization_valid(result):
        return None

    run_dir_text = str(result.get("run_dir", "") or "")
    artifact_root_text = str(result.get("artifact_root", "") or "")
    if not run_dir_text or not artifact_root_text:
        return None
    run_dir = Path(run_dir_text).resolve()
    artifact_root = Path(artifact_root_text).resolve()
    if (
        result_path.parent != run_dir
        or not run_dir.is_dir()
        or not artifact_root.is_dir()
        or not _path_is_within(run_dir, artifact_root)
        or not _path_is_within(artifact_root, output_root)
    ):
        return None
    if (
        (artifact_root / ".partial").exists()
        or not (artifact_root / ".finalized").is_file()
        or (artifact_root / ".failed").exists()
    ):
        return None

    video_files = _reliable_recording_video_files(result, run_dir)
    if video_files is None:
        return None

    diagnostics_path = run_dir / "failure_diagnostics.json"
    telemetry_finalization_path = run_dir / "telemetry_finalization.json"
    pointer_path = artifact_root / "artifact_pointer.json"
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(diagnostics, dict) or not isinstance(pointer, dict):
        return None
    if str(diagnostics.get("classification", "")) != str(
        result.get("classification", "")
    ):
        return None
    pointer_run_dir = str(pointer.get("run_dir", "") or "")
    if not pointer_run_dir or Path(pointer_run_dir).resolve() != run_dir:
        return None
    if not _checksums_match_required_files(
        run_dir,
        (
            result_path,
            diagnostics_path,
            telemetry_finalization_path,
            *video_files,
        ),
    ):
        return None
    try:
        # Keep resume admission aligned with the environment-artifact gate.  A
        # version marker alone is not durable proof of completion: the child
        # writes it before SimulationApp.close() and the supervising parent
        # records a verified graceful/fast exit only after process return.
        # Import lazily so the
        # replay CLI does not load artifact-conversion code on ordinary runs.
        from .environment_ab_artifacts import (
            _load_batch_shutdown_closure,
            _resolve_batch_root,
        )

        batch_root = _resolve_batch_root(artifact_root, run_dir)
        batch_shutdown_closure = _load_batch_shutdown_closure(
            batch_root=batch_root,
            artifact_root=artifact_root,
            run_dir=run_dir,
            result_path=result_path,
            result=result,
        )
    except Exception:
        # Resume is an optimization.  Any malformed, incomplete, ambiguous,
        # or subsequently tampered closure must therefore fail closed and be
        # replayed instead of surfacing as a CLI crash.
        return None
    if (
        str(batch_shutdown_closure.get("status", ""))
        not in SUCCESSFUL_SHUTDOWN_STATUSES
        or str(batch_shutdown_closure.get("phase", ""))
        != "SHUTDOWN_COMPLETE"
    ):
        return None
    return {
        "source_version": str(expected_version),
        "accepted_steps_sha256": str(expected_steps_sha256).lower(),
        "classification": str(result.get("classification", "")),
        "strict_full_success": bool(result.get("strict_full_success", False)),
        "batch_root": str(batch_root),
        "batch_shutdown_closure_sha256": str(
            batch_shutdown_closure.get("closure_sha256", "")
        ),
        "artifact_root": str(artifact_root),
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "failure_diagnostics_path": str(diagnostics_path),
        "created_utc": str(result.get("created_utc", "") or ""),
    }


def _find_reliable_completed_replays(
    output_root: Path,
    selected: Iterable[VersionFiles],
    expected_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Find the newest trustworthy completion for each selected recording."""

    output_root = Path(output_root).resolve()
    if not output_root.is_dir():
        return {}
    expected = {
        item.version_id: str(expected_hashes.get(item.version_id, "")).lower()
        for item in selected
        if str(expected_hashes.get(item.version_id, ""))
    }
    completed: dict[str, tuple[tuple[int, str], dict[str, Any]]] = {}
    try:
        result_paths = output_root.rglob("result.json")
        for result_path in result_paths:
            try:
                source_version = str(
                    json.loads(result_path.read_text(encoding="utf-8")).get(
                        "source_version", ""
                    )
                )
            except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
                continue
            expected_hash = expected.get(source_version)
            if not expected_hash:
                continue
            record = _reliable_replay_completion(
                result_path,
                output_root=output_root,
                expected_version=source_version,
                expected_steps_sha256=expected_hash,
            )
            if record is None:
                continue
            try:
                sort_key = (result_path.stat().st_mtime_ns, str(result_path))
            except OSError:
                continue
            previous = completed.get(source_version)
            if previous is None or sort_key > previous[0]:
                completed[source_version] = (sort_key, record)
    except OSError:
        return {}
    return {version: row for version, (_key, row) in completed.items()}


def _compact_recording_audit(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in row.items() if key != "steps"}


def _load_environment_lock(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"environment lock is required before Isaac launch: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema_version", "")) != "fsm50.environment_lock.v1":
        raise RuntimeError(f"unsupported environment lock schema: {path}")
    if not isinstance(payload.get("selected_environment"), dict):
        raise RuntimeError(f"environment lock has no selected_environment: {path}")
    return payload


def _verify_locked_source_hashes(lock: dict[str, Any]) -> dict[str, Any]:
    locked = {
        str(Path(str(raw_path)).resolve()): str(expected).lower()
        for raw_path, expected in dict(lock.get("source_sha256", {}) or {}).items()
    }
    robot_text = str(
        dict(lock.get("selected_environment", {}) or {}).get("robot_usd", "")
        or ""
    )
    required_paths = {
        str(path.resolve())
        for path in _source_files(Path(robot_text) if robot_text else None)
    }
    missing_from_lock = sorted(required_paths - set(locked))
    rows: list[dict[str, Any]] = []
    for raw_path, expected in sorted(locked.items()):
        path = Path(raw_path).resolve()
        actual = sha256_file(path) if path.is_file() else ""
        rows.append(
            {
                "path": str(path),
                "expected_sha256": str(expected).lower(),
                "actual_sha256": actual.lower(),
                "ok": bool(actual and actual.lower() == str(expected).lower()),
            }
        )
    return {
        "ok": bool(rows)
        and not missing_from_lock
        and all(row["ok"] for row in rows),
        "files": rows,
        "required_source_file_count": len(required_paths),
        "locked_source_file_count": len(locked),
        "missing_from_lock": missing_from_lock,
        "source_closure_complete": not missing_from_lock,
    }


def _runtime_environment_equivalence(
    *,
    lock: dict[str, Any],
    scene_config: Any,
    live_baseline: dict[str, Any],
    live_obstacle: dict[str, Any],
    motion: Any,
    robot_usd: Path,
    physics_dt_s: float,
) -> dict[str, Any]:
    """Compare every runtime-readable invariant to the audited lock."""

    expected = dict(lock.get("selected_environment", {}) or {})
    checks: list[dict[str, Any]] = []

    def numeric(name: str, actual: Any, expected_key: str, tolerance: float) -> None:
        try:
            actual_value = float(actual)
            expected_value = float(expected[expected_key])
            ok = bool(
                math.isfinite(actual_value)
                and math.isfinite(expected_value)
                and abs(actual_value - expected_value) <= float(tolerance)
            )
        except Exception:
            actual_value = actual
            expected_value = expected.get(expected_key)
            ok = False
        checks.append(
            {
                "name": name,
                "expected_key": expected_key,
                "expected": expected_value,
                "actual": actual_value,
                "tolerance": float(tolerance),
                "ok": ok,
            }
        )

    actual_robot_hash = sha256_file(Path(robot_usd)) if Path(robot_usd).is_file() else ""
    expected_robot_hash = str(expected.get("robot_usd_sha256", "") or "").lower()
    checks.append(
        {
            "name": "robot_usd_sha256",
            "expected": expected_robot_hash,
            "actual": actual_robot_hash.lower(),
            "tolerance": 0.0,
            "ok": bool(expected_robot_hash and actual_robot_hash.lower() == expected_robot_hash),
        }
    )
    numeric("physics_dt_s", physics_dt_s, "physics_dt_s", 1.0e-12)
    numeric(
        "render_interval_physics_steps",
        getattr(scene_config, "render_interval", None),
        "render_interval_physics_steps",
        0.0,
    )
    numeric("servo_stiffness", getattr(scene_config, "servo_stiffness", None), "servo_stiffness", 1.0e-9)
    numeric("servo_damping", getattr(scene_config, "servo_damping", None), "servo_damping", 1.0e-9)
    numeric("wheel_damping", getattr(scene_config, "wheel_damping", None), "wheel_damping", 1.0e-9)
    numeric(
        "wheel_maximum_velocity_rad_s",
        getattr(scene_config, "max_wheel_speed", None),
        "wheel_maximum_velocity_rad_s",
        1.0e-9,
    )
    numeric(
        "wheel_reference_velocity_rad_s",
        motion.wheel_reference_velocity_rad_s,
        "wheel_reference_velocity_rad_s",
        1.0e-9,
    )
    numeric(
        "servo_reference_velocity_deg_s",
        motion.servo_reference_velocity_deg_s,
        "servo_reference_velocity_deg_s",
        1.0e-9,
    )
    numeric("obstacle_height_m", live_obstacle.get("height_m"), "obstacle_height_m", 1.0e-3)
    numeric(
        "obstacle_front_face_x_m",
        live_obstacle.get("front_face_x_m"),
        "obstacle_front_face_x_m",
        1.0e-3,
    )
    numeric("obstacle_length_m", live_obstacle.get("length_m"), "obstacle_length_m", 1.0e-3)
    numeric("obstacle_width_m", live_obstacle.get("width_m"), "obstacle_width_m", 1.0e-3)
    numeric("obstacle_bottom_z_m", live_obstacle.get("bottom_z_m"), "obstacle_bottom_z_m", 1.0e-3)

    root = list(live_baseline.get("robot_root_pose", []) or [])
    expected_position = list(expected.get("robot_initial_root_position_m", []) or [])
    expected_orientation = list(expected.get("robot_initial_root_orientation_wxyz", []) or [])
    position_error = (
        math.sqrt(sum((float(root[i]) - float(expected_position[i])) ** 2 for i in range(3)))
        if len(root) >= 3 and len(expected_position) >= 3
        else float("nan")
    )
    checks.append(
        {
            "name": "robot_initial_root_position_norm_m",
            "expected": expected_position,
            "actual": root[:3],
            "error": position_error,
            "tolerance": 0.015,
            "ok": bool(math.isfinite(position_error) and position_error <= 0.015),
        }
    )
    orientation_error = float("nan")
    if len(root) >= 7 and len(expected_orientation) >= 4:
        actual_q = [float(value) for value in root[3:7]]
        expected_q = [float(value) for value in expected_orientation[:4]]
        actual_norm = math.sqrt(sum(value * value for value in actual_q))
        expected_norm = math.sqrt(sum(value * value for value in expected_q))
        if actual_norm > 1.0e-12 and expected_norm > 1.0e-12:
            dot = abs(
                sum(actual_q[i] * expected_q[i] for i in range(4))
                / (actual_norm * expected_norm)
            )
            orientation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
    checks.append(
        {
            "name": "robot_initial_root_orientation_error_rad",
            "expected": expected_orientation,
            "actual": root[3:7],
            "error": orientation_error,
            "tolerance": 0.035,
            "ok": bool(math.isfinite(orientation_error) and orientation_error <= 0.035),
        }
    )
    geometry_flags = {
        key: bool(live_obstacle.get(key, False))
        for key in ("prim_valid", "visual_valid", "collision_valid")
    }
    checks.append(
        {
            "name": "obstacle_geometry_valid",
            "expected": {key: True for key in geometry_flags},
            "actual": geometry_flags,
            "tolerance": 0.0,
            "ok": bool(geometry_flags and all(geometry_flags.values())),
        }
    )
    source_hashes = _verify_locked_source_hashes(lock)
    return {
        "schema_version": "fsm50.runtime_environment_equivalence.v1",
        "ok": bool(checks and all(row["ok"] for row in checks) and source_hashes["ok"]),
        "checks": checks,
        "locked_source_hashes": source_hashes,
        "source_locked_not_live_readable": {
            key: expected.get(key)
            for key in (
                "ground_static_friction",
                "ground_dynamic_friction",
                "obstacle_static_friction",
                "obstacle_dynamic_friction",
                "joint_command_signs",
                "joint_command_limits_deg",
                "wheel_direction_mapping",
            )
        },
    }


def _runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in ("isaacsim", "isaaclab", "torch", "numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except Exception:
            packages[name] = "unavailable"
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def _select_versions(items: list[VersionFiles], requested: Iterable[str]) -> list[VersionFiles]:
    tokens = [token.strip() for value in requested for token in str(value).split(",") if token.strip()]
    if not tokens or any(token.lower() == "all" for token in tokens):
        return items
    selected: list[VersionFiles] = []
    for token in tokens:
        matches = [
            item
            for item in items
            if item.version_id == token or item.version_id.startswith(token + "_")
        ]
        if len(matches) != 1:
            raise ValueError(f"version selector {token!r} matched {len(matches)} directories")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def _surface_obstacle(live: dict[str, Any]) -> ObstacleGeometry:
    front = float(live.get("obstacle_front_face_x_m", live.get("front_face_x_m", 0.521312174)))
    top = float(live.get("obstacle_top_z_m", live.get("top_z_m", 0.05)))
    bottom = float(live.get("obstacle_bottom_z_m", live.get("bottom_z_m", 0.0)))
    length = float(live.get("obstacle_length_m", live.get("length_m", 2.057375557)))
    width = float(live.get("obstacle_width_m", live.get("width_m", 2.0)))
    center = live.get("obstacle_center_m", None)
    center_y = float(center[1]) if isinstance(center, (list, tuple)) and len(center) >= 2 else float(live.get("center_y_m", 0.0))
    return ObstacleGeometry(
        front_face_x_m=front,
        top_z_m=top,
        bottom_z_m=bottom,
        rear_face_x_m=front + length,
        center_y_m=center_y,
        width_m=width,
    )


def _telemetry_config(output_root: Path, sample_hz: float) -> RuntimeTelemetryConfig:
    config = RuntimeTelemetryConfig()
    config.telemetry.enabled = True
    config.telemetry.sample_hz = float(sample_hz)
    config.telemetry.flush_interval_s = 15.0
    config.telemetry.output_root = str(output_root)
    config.telemetry.save_npz = False
    config.telemetry.save_csv = True
    config.telemetry.save_contacts = True
    config.telemetry.save_events = True
    config.telemetry.enable_contact_sensor = True
    config.telemetry.max_full_samples = 250000
    config.visualization.live_enabled = False
    # The FSM has an explicit two-contact corridor.  The generic equilibrium
    # solver rejects degenerate support, so it is not used as a substitute.
    config.stability.equilibrium_enabled = False
    config.stability.contact_force_threshold_n = 2.0
    return config


def _wheel_forward_sign_by_leg() -> dict[str, float]:
    return {
        "FL": float(WHEEL_FORWARD_SIGN["front_left_ankle"]),
        "FR": float(WHEEL_FORWARD_SIGN["front_right_ankle"]),
        "RL": float(WHEEL_FORWARD_SIGN["rear_left_ankle"]),
        "RR": float(WHEEL_FORWARD_SIGN["rear_right_ankle"]),
    }


def _first_failure(evidence: dict[str, Any]) -> str:
    legs = dict(evidence.get("traversal", {}).get("legs", {}) or {})
    for leg in ("FR", "FL", "RR", "RL"):
        row = dict(legs.get(leg, {}) or {})
        if row.get("unload_start_s") is None:
            return f"{leg}_UNLOAD_NOT_OBSERVED"
        if row.get("airborne_start_s") is None:
            return f"{leg}_AIRBORNE_NOT_OBSERVED"
        if row.get("front_face_crossing_s") is None:
            return f"{leg}_FRONT_FACE_CROSSING_NOT_OBSERVED"
        if row.get("illegal_reasons"):
            return f"{leg}_ILLEGAL_DRIVE_UP"
        if row.get("top_contact_s") is None:
            return f"{leg}_TOP_CONTACT_NOT_OBSERVED"
        if row.get("top_load_confirm_s") is None:
            return f"{leg}_TOP_LOAD_NOT_CONFIRMED"
    if not evidence.get("final_all_top", False):
        return "FINAL_ALL_TOP_NOT_CONFIRMED"
    if not evidence.get("final_all_loaded", False):
        return "FINAL_ALL_LOADED_NOT_CONFIRMED"
    if int(evidence.get("dangerous_collision_count", 0) or 0):
        return "DANGEROUS_COLLISION"
    return ""


def _finite_extreme(rows: list[dict[str, Any]], key: str, *, maximum: bool) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key))
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return max(values) if maximum else min(values)


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _row_mapping(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key, {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _generate_fsm50_visualization(
    run_dir: Path,
    *,
    fsm50_rows: list[dict[str, Any]],
    state_timeline_rows: list[dict[str, Any]],
    strict_result: dict[str, Any],
) -> dict[str, Any]:
    """Create the dedicated FSM50 evidence plot and self-contained HTML index."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    png_path = run_dir / "fsm50_equivalent_visualization.png"
    html_path = run_dir / "fsm50_equivalent_visualization.html"
    times = [_float_or_nan(row.get("time_s")) for row in fsm50_rows]
    fig, axes = plt.subplots(4, 1, figsize=(13.0, 12.0), sharex=True)
    if fsm50_rows:
        axes[0].plot(
            times,
            [_float_or_nan(row.get("base_roll_rad")) for row in fsm50_rows],
            label="base roll",
        )
        axes[0].plot(
            times,
            [_float_or_nan(row.get("base_pitch_rad")) for row in fsm50_rows],
            label="base pitch",
        )
        axes[0].axhline(0.0, color="black", linewidth=0.5)
        axes[0].set_ylabel("attitude [rad]")
        axes[0].legend(loc="upper right")

        axes[1].plot(
            times,
            [
                _float_or_nan(row.get("wheel_support_polygon_margin_m"))
                for row in fsm50_rows
            ],
            label="support polygon margin",
        )
        axes[1].plot(
            times,
            [
                _float_or_nan(row.get("two_leg_corridor_distance_m"))
                for row in fsm50_rows
            ],
            label="two-leg corridor distance",
        )
        axes[1].axhline(0.0, color="black", linewidth=0.5)
        axes[1].set_ylabel("margin [m]")
        axes[1].legend(loc="upper right")

        for leg in ("FL", "FR", "RL", "RR"):
            axes[2].plot(
                times,
                [
                    _float_or_nan(_row_mapping(row, "wheel_contact_force_up_n").get(leg))
                    for row in fsm50_rows
                ],
                label=leg,
            )
        axes[2].set_ylabel("upward load [N]")
        axes[2].legend(loc="upper right", ncol=4)

        source_steps = [_float_or_nan(row.get("source_step")) for row in fsm50_rows]
        axes[3].step(times, source_steps, where="post", label="source step")
        axes[3].set_ylabel("recording step")
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No FSM50 telemetry samples", ha="center", va="center")
    for row in state_timeline_rows:
        start = _float_or_nan(row.get("start_time_s"))
        if math.isfinite(start):
            axes[3].axvline(start, color="#808080", alpha=0.18, linewidth=0.6)
    axes[3].set_xlabel("simulation time [s]")
    classification = str(strict_result.get("classification", "UNKNOWN"))
    strict_success = bool(strict_result.get("strict_full_success", False))
    fig.suptitle(
        f"FSM50 deterministic replay evidence | {classification} | "
        f"strict_success={strict_success}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    timeline_table = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(key, '')))}</td>"
            for key in (
                "start_time_s",
                "end_time_s",
                "source_fast_segment",
                "source_step",
                "support_legs",
                "primary_diagonal",
                "wheel_contact_classes",
            )
        )
        + "</tr>"
        for row in state_timeline_rows
    )
    strict_json = html.escape(
        json.dumps(_jsonable(strict_result), ensure_ascii=False, indent=2), quote=False
    )
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>FSM50 equivalent visualization</title>"
        "<style>body{font-family:system-ui;margin:2rem}img{max-width:100%}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.3rem}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style></head><body>"
        f"<h1>FSM50 replay: {html.escape(classification)}</h1>"
        f"<p>Strict success: <strong>{str(strict_success).lower()}</strong>; "
        f"telemetry rows: {len(fsm50_rows)}; timeline rows: {len(state_timeline_rows)}.</p>"
        f"<img src='{html.escape(png_path.name)}' alt='FSM50 replay evidence plot'>"
        "<h2>Strict result</h2>"
        f"<pre>{strict_json}</pre>"
        "<h2>State timeline</h2><table><thead><tr>"
        "<th>start</th><th>end</th><th>segment</th><th>step</th>"
        "<th>support legs</th><th>diagonal</th><th>contacts</th>"
        f"</tr></thead><tbody>{timeline_table}</tbody></table></body></html>",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "kind": "fsm50_equivalent_telemetry_visualization",
        "png": str(png_path),
        "html": str(html_path),
        "fsm50_row_count": len(fsm50_rows),
        "state_timeline_row_count": len(state_timeline_rows),
        "strict_full_success": strict_success,
        "classification": classification,
    }


def _generate_existing_fsm50_visualization(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    strict_result = json.loads(result_path.read_text(encoding="utf-8"))
    return _generate_fsm50_visualization(
        run_dir,
        fsm50_rows=_load_csv_rows(run_dir / "fsm50_telemetry.csv"),
        state_timeline_rows=_load_csv_rows(run_dir / "state_timeline.csv"),
        strict_result=strict_result,
    )


def _mark_artifact_root(artifact_root: Path, *, valid: bool) -> None:
    partial = artifact_root / ".partial"
    destination = artifact_root / (".finalized" if valid else ".failed")
    if partial.exists():
        partial.replace(destination)
    else:
        destination.write_text(
            ("finalized" if valid else "failed") + "\n",
            encoding="utf-8",
        )
    obsolete = artifact_root / (".failed" if valid else ".finalized")
    obsolete.unlink(missing_ok=True)


def _invalidate_result_for_source_drift(
    result: dict[str, Any],
    *,
    drift: dict[str, Any],
    scope: str,
) -> None:
    if str(result.get("classification", "")) != "SOURCE_DRIFT":
        result["classification_before_source_drift"] = result.get("classification")
    result["classification"] = "SOURCE_DRIFT"
    result["first_failure_phase"] = "SOURCE_DRIFT"
    result["strict_full_success"] = False
    result["strict_success"] = False
    result["artifact_valid"] = False
    result["source_integrity"] = {
        "ok": False,
        "scope": str(scope),
        "comparison": _jsonable(drift),
    }
    result["lifecycle"] = {
        "finalized": False,
        "failed": True,
        "strict_success": False,
    }


def _leg_summary(evidence: dict[str, Any], leg: str) -> tuple[bool, bool]:
    row = dict(evidence.get("traversal", {}).get("legs", {}).get(leg, {}) or {})
    lift = bool(
        row.get("unload_start_s") is not None
        and row.get("airborne_start_s") is not None
        and row.get("front_face_crossing_s") is not None
        and not row.get("illegal_reasons")
    )
    place = bool(lift and row.get("top_load_confirm_s") is not None)
    return lift, place


def _result_payload(
    *,
    item: VersionFiles,
    service: SimTimePlaybackService,
    collector: FSM50TelemetryCollector,
    respawn: dict[str, Any],
    plan: Any,
    run_dir: Path,
    timed_out: bool,
    motion_start_readiness: dict[str, Any],
    pre_first_dispatch_readiness: dict[str, Any],
    dispatch_ledger: dict[str, Any],
    wheel_integral_evidence: dict[str, Any],
    trial_id: int,
    diagnostic_role: str = "",
) -> dict[str, Any]:
    normalized_diagnostic_role = str(diagnostic_role or "").strip().upper()
    if normalized_diagnostic_role not in {"", "U"}:
        raise ValueError("diagnostic_role must be empty or U")
    last_sim_time_s = float((collector.last_row or {}).get("time_s", 0.0) or 0.0)
    scheduler = service.status_dict(
        current_sim_time_s=last_sim_time_s,
        current_wall_time_s=time.time(),
        compact=False,
    )
    physical = collector.physical_evidence()
    role_capture_verdict = str(
        physical.get("role_capture_verdict", "NOT_EVALUABLE")
        or "NOT_EVALUABLE"
    ).upper()
    full_physical_verdict = str(
        physical.get("full_physical_verdict", "NOT_EVALUABLE")
        or "NOT_EVALUABLE"
    ).upper()
    scheduler_complete = bool(service.stop_reason == "complete" and not timed_out)
    rich_motion_start_ready = motion_start_readiness.get("ready") is True
    shared_physical_motion_start_ready = (
        dict(
            motion_start_readiness.get(
                "shared_production_worker_gate", {}
            )
            or {}
        ).get("motion_start_ready")
        is True
    )
    motion_start_ready = bool(
        rich_motion_start_ready
        and pre_first_dispatch_readiness.get("ready") is True
        and (
            normalized_diagnostic_role == "U"
            or shared_physical_motion_start_ready
        )
    )
    dispatch_complete = bool(dispatch_ledger.get("complete") is True)
    wheel_target_integral_verdict = str(
        wheel_integral_evidence.get(
            "target_integral_verdict", "NOT_EVALUABLE"
        )
        or "NOT_EVALUABLE"
    ).upper()
    wheel_target_integral_complete = bool(
        wheel_target_integral_verdict == "PASS"
    )
    strict_success = bool(
        scheduler_complete
        and motion_start_ready
        and dispatch_complete
        and wheel_target_integral_complete
        and physical.get("physical_success", False)
    )
    valid_legs = [
        leg
        for leg, row in dict(physical.get("traversal", {}).get("legs", {}) or {}).items()
        if bool(dict(row or {}).get("linkage_lift_valid", False))
    ]
    classification = (
        "FULL_SUCCESS"
        if strict_success
        else "SCHEDULER_FAILURE"
        if not scheduler_complete
        else "MOTION_START_BLOCKED"
        if not motion_start_ready
        else "DISPATCH_FAILURE"
        if not dispatch_complete
        else "WHEEL_INTEGRAL_FAILURE"
        if not wheel_target_integral_complete
        else "PARTIAL_SUCCESS"
        if valid_legs
        else "PHYSICAL_FAILURE"
    )
    first_failure_phase = (
        ""
        if strict_success
        else "SCHEDULER"
        if not scheduler_complete
        else "MOTION_START_READY"
        if not motion_start_ready
        else "SOURCE_DISPATCH_LEDGER"
        if not dispatch_complete
        else "WHEEL_TARGET_INTEGRAL"
        if not wheel_target_integral_complete
        else _first_failure(physical)
    )
    rows = collector.fsm50_rows
    final_home_error = None
    if rows:
        positions = dict(rows[-1].get("measured_joint_position_rad", {}) or {})
        targets = dict(rows[-1].get("actual_joint_target_rad", {}) or {})
        errors = [
            abs(float(positions[name]) - float(targets[name]))
            for name in SERVO_JOINT_NAMES
            if name in targets and name in positions
        ]
        final_home_error = max(errors) if errors else None
    return {
        "schema_version": "fsm50.recording_replay_result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_version": item.version_id,
        "trial_id": int(trial_id),
        "accepted_steps_sha256": sha256_file(item.steps_path),
        "run_dir": str(run_dir),
        "requested_profile": "fast",
        "canonical_profile": str(plan.profile),
        "plan_sha256": str(plan.plan_sha256),
        "plan_event_count": len(plan.events),
        "plan_segment_count": len(plan.segments),
        "plan_final_time_s": float(plan.final_time_s),
        "respawn": _jsonable(respawn),
        "fresh_process_clean_reset": True,
        "motion_start_readiness": _jsonable(motion_start_readiness),
        "motion_start_pre_first_dispatch": _jsonable(
            pre_first_dispatch_readiness
        ),
        "motion_start_ready": motion_start_ready,
        "motion_start_readiness_scope": (
            "SENSOR_INDEPENDENT_TRAJECTORY_ADMISSION"
            if normalized_diagnostic_role == "U"
            else "PHYSICAL_MOTION_ADMISSION"
        ),
        "physical_motion_start_verdict": (
            "NOT_EVALUABLE"
            if normalized_diagnostic_role == "U"
            else "PASS"
            if shared_physical_motion_start_ready
            else "FAIL"
        ),
        "dispatch_ledger": _jsonable(dispatch_ledger),
        "dispatch_complete": dispatch_complete,
        "wheel_integral_evidence": _jsonable(wheel_integral_evidence),
        "wheel_target_integral_verdict": wheel_target_integral_verdict,
        "wheel_target_integral_complete": wheel_target_integral_complete,
        "measured_wheel_tracking_verdict": str(
            wheel_integral_evidence.get(
                "measured_tracking_verdict", "NOT_EVALUABLE"
            )
            or "NOT_EVALUABLE"
        ).upper(),
        "scheduler_complete": scheduler_complete,
        "scheduler_stop_reason": str(service.stop_reason or ""),
        "scheduler_status": _jsonable(scheduler),
        "timed_out": bool(timed_out),
        "physical_evidence": physical,
        "role_capture_verdict": role_capture_verdict,
        "role_capture_success": role_capture_verdict == "PASS",
        "full_physical_verdict": full_physical_verdict,
        "telemetry_finalization": _jsonable(
            getattr(collector, "telemetry_finalization_status", lambda: {})()
        ),
        "physical_success": bool(physical.get("physical_success", False)),
        "strict_full_success": strict_success,
        "classification": classification,
        "valid_linkage_lift_legs": valid_legs,
        "first_failure_phase": first_failure_phase,
        "maximum_abs_roll_rad": _finite_extreme(
            [{**row, "abs_roll": abs(float(row.get("base_roll_rad", math.nan)))} for row in rows],
            "abs_roll",
            maximum=True,
        ),
        "maximum_abs_pitch_rad": _finite_extreme(
            [{**row, "abs_pitch": abs(float(row.get("base_pitch_rad", math.nan)))} for row in rows],
            "abs_pitch",
            maximum=True,
        ),
        "minimum_support_polygon_margin_m": _finite_extreme(rows, "wheel_support_polygon_margin_m", maximum=False),
        "maximum_contact_drift_m": physical.get("maximum_contact_drift_m"),
        "minimum_two_leg_corridor_margin_m": physical.get("minimum_two_leg_corridor_margin_m"),
        "final_joint_target_error_rad": final_home_error,
        "success_semantics": (
            "MOTION_START_READY, complete source/semantic-noop dispatch ledger, "
            "scheduler completion, a PASS authoritative plan-to-PhysX wheel "
            "target integral, and strict per-leg "
            "UNLOAD->AIR->CLEAR_FACE->TOP->LOAD evidence are independent; "
            "only their conjunction can be success"
        ),
    }


def _write_checksums(
    root: Path,
    *,
    exclude_preclose_snapshots: bool = False,
) -> None:
    root = Path(root).resolve()
    destination = root / "checksums.sha256"
    preclose_destinations = {
        destination_name
        for _source_name, destination_name in _PRECLOSE_SNAPSHOT_FILES
    }
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        relative_parts = path.relative_to(root).parts
        if path.resolve() == destination or path.name in {
            ".partial",
            ".complete",
            ".finalized",
            ".failed",
        }:
            continue
        if ".telemetry_journal" in relative_parts or (
            path.name.startswith(".") and ".tmp" in path.name
        ):
            continue
        if exclude_preclose_snapshots and relative in preclose_destinations:
            continue
        rows.append(f"{sha256_file(path)}  {relative}")
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(rows) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_run_inputs(item: VersionFiles, run_dir: Path, steps: list[dict[str, Any]], max_wheel_speed: float) -> None:
    inputs = run_dir / "input"
    inputs.mkdir(parents=True, exist_ok=False)
    shutil.copy2(item.steps_path, inputs / "accepted_steps.jsonl")
    shutil.copy2(item.metadata_path, inputs / "metadata.json")
    write_fast_plan(
        output_dir=inputs / "fast_plan",
        source_version=item.version_id,
        steps=steps,
        max_wheel_speed=max_wheel_speed,
    )


def _settle(adapter: Any, duration_s: float, collector: FSM50TelemetryCollector, label: str) -> None:
    target = float(adapter.sim_time) + max(0.0, float(duration_s))
    collector.set_runtime_context(
        fsm_state=label,
        macro_state_cursor=label,
        source_step=None,
        segment_index=None,
        segment_cursor=None,
        command_cursor=None,
        source_command="",
        source_event_index=None,
        planned_dispatch_time_s=None,
        actual_dispatch_time_s=None,
        atomic_batch_id="",
        dispatch_kind="",
        scheduler_phase="post_settle",
    )
    while float(adapter.sim_time) + 1.0e-9 < target:
        adapter.step()


def _motion_start_plan_identity(
    *,
    item: VersionFiles,
    plan: Any,
    plan_id: str,
    request_id: str,
    worker_session_id: str,
) -> dict[str, Any]:
    initial_command_state = copy.deepcopy(
        dict(plan.timing.get("source_initial_command_state", {}) or {})
    )
    initial_command_state_sha256 = str(
        plan.timing.get("source_initial_command_state_sha256", "") or ""
    )
    return {
        "source_version": str(item.version_id),
        "source_sha256": sha256_file(item.steps_path),
        "plan_sha256": str(plan.plan_sha256),
        "plan_id": str(plan_id),
        "request_id": str(request_id),
        "worker_session_id": str(worker_session_id),
        "event_count": len(plan.events),
        "segment_count": len(plan.segments),
        "validated_plan_sha256": str(plan.plan_sha256),
        "validated_event_count": len(plan.events),
        "validated_segment_count": len(plan.segments),
        "integrity_ok": True,
        "requested_worker_session_id": str(worker_session_id),
        "source_initial_command_state": initial_command_state,
        "source_initial_command_state_sha256": initial_command_state_sha256,
    }


def _build_motion_start_readiness_evidence(
    *,
    readiness: dict[str, Any],
    source_version: str,
    trial_id: int,
    plan_identity: dict[str, Any],
    adapter_runtime_instance_id: str,
    root_state_write_count: int,
    pre_first_dispatch_sim_step: int,
) -> tuple[dict[str, Any], str]:
    """Return a JSON-safe evidence document and its immutable binding token."""

    def clone(value: Any) -> Any:
        return json.loads(
            json.dumps(
                _jsonable(copy.deepcopy(value)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )

    snapshot = clone(readiness)
    token_payload = {
        "schema_version": "fsm50.motion_start_readiness_token.v1",
        "source_version": str(source_version),
        "trial_id": int(trial_id),
        "plan_identity": clone(plan_identity),
        "adapter_runtime_instance_id": str(adapter_runtime_instance_id),
        "root_state_write_count": int(root_state_write_count),
        "pre_first_dispatch_sim_step": int(pre_first_dispatch_sim_step),
        "pre_first_dispatch_readiness": snapshot,
    }
    token = hashlib.sha256(
        json.dumps(
            token_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence = {
        **snapshot,
        "readiness_token_sha256": token,
        "token_payload": clone(token_payload),
    }
    # Exercise the same encoder used by artifact writers here so a future
    # accidental self-reference fails before SimulationApp is launched.
    json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)
    return evidence, token


def _batch_owned_segment_cursor(
    *,
    scheduler_segment_index: int,
    current_motion_batch: dict[str, Any],
    plan: Any,
) -> tuple[int | None, int | None]:
    """Bind the next physics frame to its applied batch, not an advanced cursor."""

    if current_motion_batch:
        raw_segment_index = current_motion_batch.get("segment_index")
        raw_source_step = current_motion_batch.get("source_step_index")
        return (
            None if raw_segment_index is None else int(raw_segment_index),
            None if raw_source_step is None else int(raw_source_step),
        )
    segment_index = int(scheduler_segment_index)
    if 0 <= segment_index < len(plan.segments):
        return segment_index, int(plan.segments[segment_index].source_step)
    if plan.segments and segment_index == len(plan.segments):
        final_segment = plan.segments[-1]
        return len(plan.segments) - 1, int(final_segment.source_step)
    return segment_index, None


def _batch_owned_command_context(
    *,
    current_motion_batch: dict[str, Any],
    timing_commands: list[dict[str, Any]],
    source_event_by_plan_index: dict[int, Any],
) -> dict[str, Any]:
    """Select command provenance for the batch owning the next physics frame."""

    latest = dict(timing_commands[-1] if timing_commands else {})
    if current_motion_batch:
        global_indices = list(
            current_motion_batch.get("global_command_indices", []) or []
        )
        plan_indices = list(
            current_motion_batch.get("plan_event_indices", []) or []
        )
        # Scheduler-generated boundaries (including final_safety_stop) are real
        # motion batches but are not source commands.  They must not inherit the
        # last recording cursor merely because the service trace retains it.
        if not global_indices or not plan_indices:
            return {
                "command_cursor": None,
                "source_command": "",
                "source_event_index": None,
                "planned_dispatch_time_s": None,
                "actual_dispatch_time_s": None,
                "atomic_batch_id": str(
                    current_motion_batch.get("batch_id", "") or ""
                ),
                "dispatch_kind": str(
                    current_motion_batch.get("dispatch_kind", "") or ""
                ),
            }
        command_cursor = int(global_indices[-1])
        plan_event_index = int(plan_indices[-1])
        command_row = next(
            (
                dict(row)
                for row in reversed(timing_commands)
                if int(row.get("global_command_index", -1)) == command_cursor
            ),
            {},
        )
        return {
            "command_cursor": command_cursor,
            "source_command": str(command_row.get("command", "") or ""),
            "source_event_index": source_event_by_plan_index.get(
                plan_event_index
            ),
            "planned_dispatch_time_s": command_row.get(
                "planned_start_sim_time"
            ),
            "actual_dispatch_time_s": command_row.get(
                "actual_start_sim_time"
            ),
            "atomic_batch_id": str(
                current_motion_batch.get("batch_id", "") or ""
            ),
            "dispatch_kind": str(
                current_motion_batch.get("dispatch_kind", "") or ""
            ),
        }

    command_cursor = latest.get("global_command_index")
    try:
        plan_event_index = int(command_cursor) - 1
    except (TypeError, ValueError):
        plan_event_index = None
    return {
        "command_cursor": command_cursor,
        "source_command": str(latest.get("command", "") or ""),
        "source_event_index": (
            None
            if plan_event_index is None
            else source_event_by_plan_index.get(plan_event_index)
        ),
        "planned_dispatch_time_s": latest.get("planned_start_sim_time"),
        "actual_dispatch_time_s": latest.get("actual_start_sim_time"),
        "atomic_batch_id": str(latest.get("atomic_batch_id", "") or ""),
        "dispatch_kind": "",
    }


def _fresh_ground_evidence_for_motion_start(
    startup_ground: dict[str, Any],
    frame: dict[str, Any],
) -> dict[str, Any]:
    """Bind startup REST evidence to current contact/penetration observations."""

    live = copy.deepcopy(dict(startup_ground or {}))
    penetration = dict(frame.get("penetration", {}) or {})
    forces = dict(frame.get("wheel_net_forces_w", {}) or {})
    support_errors: list[str] = []
    for leg in ("FL", "FR", "RL", "RR"):
        vector = list(forces.get(leg, []) or [])
        try:
            parsed = [float(value) for value in vector]
        except (TypeError, ValueError):
            parsed = []
        if (
            len(parsed) != 3
            or not all(math.isfinite(value) for value in parsed)
            or parsed[2] <= 0.0
        ):
            support_errors.append(f"{leg} current upward support is unavailable")
    classification = str(penetration.get("classification", "") or "")
    penetration_safe = bool(
        penetration.get("valid") is True
        and penetration.get("physical_ground_safe") is True
        and classification in {"OK", "VISUAL_ONLY_INTERSECTION"}
    )
    live["ground_contact_resolved"] = bool(
        frame.get("wheel_force_evidence_valid") is True
        and not support_errors
        and penetration_safe
    )
    live["ground_support_clearance_evidence"] = {
        "valid": penetration_safe,
        "error": str(penetration.get("error", "") or ""),
        "wheel_clearance_by_name": dict(
            penetration.get("wheel_clearance_m", {}) or {}
        ),
        "source": "fresh pre-dispatch penetration_snapshot",
    }
    diagnostics = dict(live.get("grounded_reference_diagnostics", {}) or {})
    diagnostics.update(
        {
            "checked": penetration.get("valid") is True,
            "physical_ground_safe": penetration.get("physical_ground_safe") is True,
            "classification": classification,
            "fresh_motion_start_sim_step": frame.get("sim_step"),
        }
    )
    live["grounded_reference_diagnostics"] = diagnostics
    live["motion_start_support_errors"] = support_errors
    return live


def _evaluate_motion_start_window(
    *,
    adapter: Any,
    startup_ground: dict[str, Any],
    frames: list[dict[str, Any]],
    plan_identity: dict[str, Any],
    required_frames: int,
    command_dispatch_idle: bool = True,
    command_dispatch_evidence: dict[str, Any] | None = None,
    diagnostic_role: str = "",
) -> dict[str, Any]:
    normalized_diagnostic_role = str(diagnostic_role or "").strip().upper()
    if normalized_diagnostic_role not in {"", "U"}:
        raise ValueError("diagnostic_role must be empty or U")
    selected = list(frames[-max(1, int(required_frames)) :])
    frame_results: list[dict[str, Any]] = []
    expected_instance = str(getattr(adapter, "runtime_instance_id", "") or "")
    for frame in selected:
        frame_result = evaluate_motion_start_ready(
            ground_reference=_fresh_ground_evidence_for_motion_start(
                startup_ground, frame
            ),
            snapshot=frame,
            production_runtime_ready=True,
            expected_sim_step=int(frame.get("sim_step", -1)),
            expected_adapter_runtime_instance_id=expected_instance,
            plan_identity=plan_identity,
            command_dispatch_idle=bool(command_dispatch_idle),
            root_seed_applied=False,
            vertical_speed_limit_m_s=float(
                adapter.config.ground_vertical_speed_threshold_m_s
            ),
            wheel_speed_limit_rad_s=float(
                adapter.config.ground_wheel_speed_threshold_rad_s
            ),
            penetration_limit_m=float(
                adapter.config.ground_penetration_tolerance_m
            ),
        )
        if normalized_diagnostic_role == "U":
            # U proves that the ordinary production UI trajectory can be
            # dispatched with its contact sensor plumbing left disabled.  A
            # missing contact observation is therefore expected and must not
            # be converted into either a physical PASS or a readiness FAIL.
            # Retain only identity/runtime/kinematic admission checks; mark
            # every physical/contact check explicitly NOT_EVALUABLE.
            applicable_checks = {
                "production_runtime_ready",
                "no_historical_root_seed",
                "fresh_adapter_instance",
                "fresh_snapshot",
                "plan_identity_bound_and_no_prior_dispatch",
                "live_command_state_matches_source_initial_state",
                "root_motion_evidence_complete",
                "attitude_evidence_complete",
                "obstacle_relative_pose_complete",
                "joint_and_physx_position_targets_valid",
                "wheel_measured_motion_safe",
                "wheel_targets_zero_and_physx_verified",
            }
            raw_checks = dict(frame_result.get("checks", {}) or {})
            role_checks: dict[str, dict[str, Any]] = {}
            for name, raw_check in raw_checks.items():
                if name in applicable_checks:
                    role_checks[name] = dict(raw_check or {})
                else:
                    role_checks[name] = {
                        "passed": None,
                        "verdict": "NOT_EVALUABLE",
                        "availability": "UNAVAILABLE_BY_ROLE",
                        "reason": (
                            "ordinary-UI diagnostic role U runs with contact "
                            "sensors disabled and cannot make physical claims"
                        ),
                    }
            applicable_ready = bool(
                applicable_checks
                and applicable_checks.issubset(role_checks)
                and all(
                    role_checks[name].get("passed") is True
                    for name in applicable_checks
                )
            )
            frame_result.update(
                diagnostic_role="U",
                qualification_scope=(
                    "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
                ),
                checks=role_checks,
                ready=applicable_ready,
                status="PASS" if applicable_ready else "FAIL",
                classification=(
                    "MOTION_START_READY"
                    if applicable_ready
                    else "MOTION_START_BLOCKED"
                ),
                failed_checks=[
                    name
                    for name in sorted(applicable_checks)
                    if role_checks.get(name, {}).get("passed") is not True
                ],
                physical_readiness_verdict="NOT_EVALUABLE",
                production_worker_motion_ready=None,
                production_worker_motion_reason="",
                production_worker_ground_state="NOT_EVALUABLE",
            )
        frame_results.append(frame_result)
    steps = [int(frame.get("sim_step", -1)) for frame in selected]
    contiguous_physics_ticks = bool(
        steps
        and steps == list(range(steps[0], steps[0] + len(steps)))
    )
    production_step_stride: int | None = None
    production_cadence_error = ""
    try:
        _render_elapsed_s, raw_step_stride = adapter._render_step_timing()
        parsed_step_stride = int(raw_step_stride)
        if parsed_step_stride < 1:
            raise ValueError("physics-step stride must be positive")
        production_step_stride = parsed_step_stride
    except (AttributeError, TypeError, ValueError) as exc:
        production_cadence_error = (
            "production render-loop cadence unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    observed_step_deltas = [
        int(current - previous)
        for previous, current in zip(steps, steps[1:])
    ]
    production_sampling_cadence_valid = bool(
        steps
        and production_step_stride is not None
        and all(step >= 0 for step in steps)
        and all(
            delta == production_step_stride for delta in observed_step_deltas
        )
    )
    window_failed_checks: list[str] = []
    if len(selected) != int(required_frames):
        window_failed_checks.append(
            "observed frame count does not match required production samples"
        )
    if not production_sampling_cadence_valid:
        if production_cadence_error:
            window_failed_checks.append(production_cadence_error)
        else:
            window_failed_checks.append(
                "motion-start samples do not follow the production render-loop "
                f"cadence of {production_step_stride} physics tick(s)"
            )
    for index, row in enumerate(frame_results):
        if row.get("ready") is not True:
            failed = list(row.get("failed_checks", []) or [])
            window_failed_checks.append(
                f"sample {index} is not MOTION_START_READY"
                + (f": {failed}" if failed else "")
            )
    ready = bool(
        len(selected) == int(required_frames)
        and production_sampling_cadence_valid
        and frame_results
        and all(row.get("ready") is True for row in frame_results)
    )
    return {
        "schema_version": "fsm50.motion_start_readiness_window.v1",
        "gate": "MOTION_START_READY",
        "ready": ready,
        "status": "PASS" if ready else "FAIL",
        "required_consecutive_frames": int(required_frames),
        "observed_frame_count": len(selected),
        "sim_steps": steps,
        # ``SimRobotAdapter.step`` is one production render-loop iteration.  At
        # render_interval=8 it advances eight 1/120 s physics ticks before the
        # next scheduler/readiness observation.  Preserve the literal physics
        # contiguity result instead of mislabelling these samples as +1 ticks,
        # while qualifying the exact cadence used by the formal worker.
        "contiguous_sim_steps": contiguous_physics_ticks,
        "production_sampling_cadence_valid": production_sampling_cadence_valid,
        "production_step_stride_physics_ticks": production_step_stride,
        "production_sampling_cadence_error": production_cadence_error,
        "observed_sim_step_deltas": observed_step_deltas,
        "sampling_semantics": (
            "one readiness snapshot per production render-loop iteration"
        ),
        "window_failed_checks": window_failed_checks,
        "plan_identity": dict(plan_identity),
        "command_dispatch_idle": bool(command_dispatch_idle),
        "command_dispatch_evidence": dict(command_dispatch_evidence or {}),
        "adapter_runtime_instance_id": expected_instance,
        "root_state_write_count": int(
            getattr(adapter, "root_state_write_count", 0) or 0
        ),
        "rest_qualification": rest_qualification_summary(startup_ground),
        "frame_results": frame_results,
        "frames": selected,
        "final": frame_results[-1] if frame_results else {},
        "strict_rest_is_required": False,
        "envelope_status": "PENDING_THREE_SUCCESSFUL_V003_FAST_REPLAYS",
        "writes_robot_state": False,
        "diagnostic_role": normalized_diagnostic_role,
        "qualification_scope": (
            "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
            if normalized_diagnostic_role == "U"
            else "GATE1_PHYSICAL_QUALIFICATION"
        ),
        "physical_readiness_verdict": (
            "NOT_EVALUABLE" if normalized_diagnostic_role == "U" else "EVALUATED"
        ),
    }


def _run_recording_version(
    *,
    item: VersionFiles,
    adapter: Any,
    scene_handle: Any,
    live_baseline: dict[str, Any],
    batch_root: Path,
    args: argparse.Namespace,
    robot_usd: Path,
    expected_steps_sha256: str,
    environment_lock: dict[str, Any],
    startup_ground: dict[str, Any],
    trial_id: int,
) -> dict[str, Any]:
    from sim_obstacle_scene import measure_obstacle_geometry, measure_scene_baseline

    version_root = batch_root / item.version_id
    version_root.mkdir(parents=True, exist_ok=True)
    artifact_root = _new_directory(version_root, "clean_fast_replay")
    partial = artifact_root / ".partial"
    partial.write_text("running\n", encoding="utf-8")
    source_freeze: dict[str, Any] = {}
    immediate_steps_sha256 = ""
    recording_changed_after_preflight = False
    steps: list[dict[str, Any]] = []
    motion: Any = None
    plan: Any = None
    _plan_rows: list[dict[str, Any]] = []
    service = SimTimePlaybackService()
    collector: FSM50TelemetryCollector | None = None
    run_dir = artifact_root
    contact_mode = str(
        getattr(args, "contact_mode", "instrumented") or "instrumented"
    )
    video_capture: _RecordingReplayViewportCapture | None = None
    video: dict[str, Any] = {}
    version_live_baseline: dict[str, Any] = {}
    version_live_obstacle: dict[str, Any] = {}
    version_environment_equivalence: dict[str, Any] = {}
    timed_out = False
    app_stopped = False
    respawn: dict[str, Any] = {}
    motion_start_readiness: dict[str, Any] = {}
    pre_first_dispatch_readiness: dict[str, Any] = {}
    dispatch_ledger: dict[str, Any] = {}
    wheel_integral_evidence: dict[str, Any] = {}
    try:
        source_freeze = _source_freeze(item, robot_usd=robot_usd)
        write_json(artifact_root / "source_freeze_pre.json", source_freeze)
        write_json(artifact_root / "source_freeze.json", source_freeze)
        immediate_steps_sha256 = sha256_file(item.steps_path)
        recording_changed_after_preflight = (
            immediate_steps_sha256.lower()
            != str(expected_steps_sha256).lower()
        )
        if recording_changed_after_preflight:
            raise RuntimeError(
                f"accepted_steps changed after prelaunch audit for {item.version_id}: "
                f"expected={expected_steps_sha256} actual={immediate_steps_sha256}"
            )
        steps = load_steps_jsonl(item.steps_path)
        motion = load_motion_reference()
        plan, _plan_rows = fast_plan_rows(
            source_version=item.version_id,
            steps=steps,
            max_wheel_speed=float(motion.wheel_velocity_limit_rad_s),
        )
        plan.timing["requires_motion_start_readiness_token"] = True
        plan.timing["requires_verified_motion_batch_ack"] = True
        config = _telemetry_config(artifact_root, args.telemetry_rate)
        collector_args = SimpleNamespace(
            headless=bool(args.headless), output_dir=str(artifact_root)
        )
        # Gate-1 runs one selected version in one fresh supervised child.  Do
        # not respawn here: respawn_robot writes the cached grounded root pose
        # and would no longer be equivalent to ordinary production Play All
        # Fast from the just-created articulation.
        adapter.attach_telemetry(None)
        respawn = {
            "ok": True,
            "respawned": False,
            "root_pose_written": False,
            "reason": "fresh supervised child; per-version cached-root respawn forbidden",
            "adapter_runtime_instance_id": str(
                getattr(adapter, "runtime_instance_id", "") or ""
            ),
            "root_state_write_count": int(
                getattr(adapter, "root_state_write_count", 0) or 0
            ),
        }
        filtered_sensor = getattr(scene_handle, "contact_sensor", None)
        if filtered_sensor is None or str(getattr(scene_handle, "contact_sensor_error", "") or ""):
            raise RuntimeError(
                "filtered contact sensor unavailable after respawn: "
                + str(getattr(scene_handle, "contact_sensor_error", "") or "missing sensor")
            )
        filtered_sensor.reset()
        version_live_baseline = measure_scene_baseline(scene_handle, adapter)
        version_live_obstacle = measure_obstacle_geometry(scene_handle)
        version_environment_equivalence = _runtime_environment_equivalence(
            lock=environment_lock,
            scene_config=scene_handle.config,
            live_baseline=version_live_baseline,
            live_obstacle=version_live_obstacle,
            motion=motion,
            robot_usd=robot_usd,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )
        write_json(
            artifact_root / "runtime_environment_pre_action.json",
            {
                "source_version": item.version_id,
                "contact_mode": contact_mode,
                "video_capture_requested": _recording_video_capture_requested(args),
                "live_scene_baseline": version_live_baseline,
                "live_obstacle_geometry": version_live_obstacle,
                "environment_equivalence": version_environment_equivalence,
            },
        )
        if not version_environment_equivalence.get("ok", False):
            raise RuntimeError(
                f"per-version runtime environment drift before {item.version_id}"
            )
        obstacle = _surface_obstacle(version_live_baseline)
        wheel_radius = float(
            version_live_baseline.get("wheel_radius_m") or 0.04998999834060672
        )
        collector_kwargs: dict[str, Any] = {
            "args": collector_args,
            "scene_handle": scene_handle,
            "obstacle": obstacle,
            "wheel_radius_m": wheel_radius,
            "source_version": item.version_id,
            "contact_mode": contact_mode,
            "plan": plan,
            "force_threshold_n": 2.0,
            "unload_force_n": 1.0,
            "load_confirm_force_n": 2.0,
            "top_load_dwell_s": 0.10,
            "loaded_front_face_rotation_limit_rad": 0.15,
            "wheel_forward_sign": _wheel_forward_sign_by_leg(),
        }
        if "plan_rows" in inspect.signature(FSM50TelemetryCollector).parameters:
            collector_kwargs["plan_rows"] = _plan_rows
        collector = FSM50TelemetryCollector(config, **collector_kwargs)
        collector.start_episode(
            adapter=adapter,
            scene_handle=scene_handle,
            obstacle_height_cm=5,
            obstacle_height_m=0.05,
            sequence_label=f"recording_fast_{item.version_id}",
            source="fsm50_recording_audit",
        )
        run_dir = collector.run_dir or artifact_root
        video_capture = _RecordingReplayViewportCapture(
            run_dir,
            args,
            contact_mode=contact_mode,
            adapter=adapter,
        )
        video_capture.start()
        adapter.attach_telemetry(collector)
        collector.record_event(
            float(adapter.sim_time),
            "fresh_process_motion_start_evidence_begin",
            severity="info",
            message=f"Fresh-process startup evidence for {item.version_id}",
            extra={"respawn": _jsonable(respawn)},
        )
        plan_id = f"recording-{item.version_id}-trial-{int(trial_id):02d}"
        request_id = uuid.uuid4().hex
        worker_session_id = (
            f"fsm50-{getattr(adapter, 'runtime_instance_id', uuid.uuid4().hex)[:12]}"
        )
        plan_identity = _motion_start_plan_identity(
            item=item,
            plan=plan,
            plan_id=plan_id,
            request_id=request_id,
            worker_session_id=worker_session_id,
        )
        readiness_frames: list[dict[str, Any]] = []
        required_readiness_frames = 10
        while len(readiness_frames) < required_readiness_frames:
            frame = capture_live_motion_start_snapshot(
                adapter, scene_handle, version_live_obstacle
            )
            readiness_frames.append(frame)
            collector.set_runtime_context(
                fsm_state="MOTION_START_READY",
                macro_state_cursor="MOTION_START_READY",
                command_cursor=0,
                segment_cursor=0,
                source_command="",
                source_event_index=None,
                planned_dispatch_time_s=None,
                actual_dispatch_time_s=None,
                atomic_batch_id="",
                dispatch_kind="",
                segment_index=None,
                source_step=None,
                scheduler_phase="pre_dispatch_evidence",
            )
            adapter.step()
        motion_start_readiness = _evaluate_motion_start_window(
            adapter=adapter,
            startup_ground=startup_ground,
            frames=readiness_frames,
            plan_identity=plan_identity,
            required_frames=required_readiness_frames,
        )
        write_json(run_dir / "motion_start_readiness.json", motion_start_readiness)
        if not motion_start_readiness.get("ready", False):
            raise RuntimeError(
                "MOTION_START_READY failed before production start boundary: "
                + json.dumps(
                    motion_start_readiness.get("window_failed_checks", []),
                    ensure_ascii=False,
                )
            )
        from sim_worker_runtime import capture_worker_motion_start_readiness

        shared_worker_readiness = capture_worker_motion_start_readiness(
            adapter,
            runtime_ready=True,
            current_sim_step=int(adapter.sim_steps),
            worker_session_id=worker_session_id,
            request_identity=plan_identity,
        )
        motion_start_readiness["shared_production_worker_gate"] = (
            shared_worker_readiness
        )
        write_json(run_dir / "motion_start_readiness.json", motion_start_readiness)
        if not shared_worker_readiness.get("motion_start_ready", False):
            raise RuntimeError(
                "shared production worker MOTION_START_READY failed: "
                + str(shared_worker_readiness.get("rejection_reason", "") or "")
            )
        one_tick_start_delay_s = float(scene_handle.sim.get_physics_dt())
        started = service.start_plan(
            plan,
            current_sim_time_s=float(adapter.sim_time),
            current_wall_time_s=time.time(),
            start_delay_sim_s=one_tick_start_delay_s,
            plan_id=plan_id,
            request_id=request_id,
            worker_session_id=worker_session_id,
        )
        if not started:
            raise RuntimeError(service.last_error or "Fast replay plan did not start")
        if not service.apply_playback_start_boundary(
            adapter,
            current_sim_time_s=float(adapter.sim_time),
            current_sim_step=int(adapter.sim_steps),
        ):
            raise RuntimeError(
                service.last_error or "production playback start boundary failed"
            )
        boundary_frame = capture_live_motion_start_snapshot(
            adapter, scene_handle, version_live_obstacle
        )
        start_boundary_ack = copy.deepcopy(service.last_motion_batch_ack)
        collector.set_runtime_context(
            fsm_state="MOTION_START_READY",
            macro_state_cursor="MOTION_START_READY",
            command_cursor=0,
            segment_cursor=0,
            source_command="",
            source_event_index=None,
            planned_dispatch_time_s=None,
            actual_dispatch_time_s=None,
            atomic_batch_id=str(
                service.last_motion_batch_ack.get("batch_id", "") or ""
            ),
            dispatch_kind="playback_start_boundary",
            segment_index=None,
            source_step=None,
            scheduler_phase="verified_start_boundary",
        )
        # Let the zero-wheel start boundary own and cross its declared first
        # physics step.  The authoritative pre-first snapshot is captured only
        # afterwards, on the exact tick where the first source batch may be
        # dispatched, matching the production worker loop.
        adapter.step()
        post_boundary_frame = capture_live_motion_start_snapshot(
            adapter, scene_handle, version_live_obstacle
        )
        pre_first_dispatch_readiness = _evaluate_motion_start_window(
            adapter=adapter,
            startup_ground=startup_ground,
            frames=[
                *readiness_frames[2:],
                boundary_frame,
                post_boundary_frame,
            ],
            plan_identity=plan_identity,
            required_frames=required_readiness_frames,
        )
        pre_first_shared_worker_readiness = capture_worker_motion_start_readiness(
            adapter,
            runtime_ready=True,
            current_sim_step=int(adapter.sim_steps),
            worker_session_id=worker_session_id,
            request_identity=plan_identity,
        )
        pre_first_dispatch_readiness.update(
            {
                "start_boundary_ack": start_boundary_ack,
                "start_boundary_applied_sim_step": start_boundary_ack.get(
                    "applied_sim_step"
                ),
                "start_boundary_first_physics_step": start_boundary_ack.get(
                    "first_physics_step"
                ),
                "pre_first_dispatch_sim_step": int(adapter.sim_steps),
                "shared_production_worker_gate": (
                    pre_first_shared_worker_readiness
                ),
                "source_command_dispatch_count": 0,
                "boundary_batch_is_not_source_command": True,
            }
        )
        if not pre_first_dispatch_readiness.get("ready", False) or not bool(
            pre_first_shared_worker_readiness.get("motion_start_ready", False)
        ):
            raise RuntimeError(
                "MOTION_START_READY became stale/invalid on the exact "
                "pre-first-dispatch physics tick"
            )
        # Freeze the evidence before constructing the token envelope.  The
        # helper deliberately produces two independent JSON trees so attaching
        # the envelope cannot create a readiness -> payload -> readiness cycle.
        pre_first_dispatch_readiness, readiness_token = (
            _build_motion_start_readiness_evidence(
                readiness=pre_first_dispatch_readiness,
                source_version=item.version_id,
                trial_id=int(trial_id),
                plan_identity=plan_identity,
                adapter_runtime_instance_id=str(
                    getattr(adapter, "runtime_instance_id", "") or ""
                ),
                root_state_write_count=int(
                    getattr(adapter, "root_state_write_count", 0) or 0
                ),
                pre_first_dispatch_sim_step=int(adapter.sim_steps),
            )
        )
        if not service.bind_motion_start_readiness(
            readiness_token,
            current_sim_step=int(adapter.sim_steps),
        ):
            raise RuntimeError(
                service.last_error
                or "pre-first MOTION_START_READY token binding failed"
            )
        write_json(
            run_dir / "motion_start_pre_first_dispatch.json",
            pre_first_dispatch_readiness,
        )
        run_start_sim = float(adapter.sim_time)
        source_event_by_plan_index = {
            int(provenance["plan_event_index"]): provenance.get(
                "source_event_index"
            )
            for plan_row in _plan_rows
            for provenance in list(
                plan_row.get("command_provenance", []) or []
            )
            if provenance.get("plan_event_index") is not None
        }
        deadline_sim = run_start_sim + max(
            float(args.timeout_s),
            one_tick_start_delay_s
            + float(plan.final_time_s) * float(args.timeout_scale),
        )
        while service.active:
            if not bool(scene_handle.app_is_running()):
                app_stopped = True
                service.stop(
                    adapter,
                    current_sim_time_s=float(adapter.sim_time),
                    current_wall_time_s=time.time(),
                    reason="simulation_app_stopped",
                )
                break
            service.update(
                adapter,
                current_sim_time_s=float(adapter.sim_time),
                current_sim_step=int(adapter.sim_steps),
                current_wall_time_s=time.time(),
            )
            timing_commands = list(
                service.timing_trace.get("commands", []) or []
            )
            current_step_batches = [
                dict(row or {})
                for row in list(
                    service.timing_trace.get("motion_batches", []) or []
                )
                if row.get("scheduler_sim_step") is not None
                and int(row.get("scheduler_sim_step")) == int(adapter.sim_steps)
            ]
            current_motion_batch = (
                current_step_batches[-1] if current_step_batches else {}
            )
            scheduler_segment_index = int(service.segment_index)
            # A zero-duration segment can complete in the same scheduler call
            # in which its source batch is applied.  In that case the service
            # cursor already points at N+1 while the command that will own the
            # next physics frame is still segment N.  Bind telemetry to the
            # applied batch whenever one exists; use the scheduler cursor only
            # on ticks without a new batch.
            segment_index, source_step = _batch_owned_segment_cursor(
                scheduler_segment_index=scheduler_segment_index,
                current_motion_batch=current_motion_batch,
                plan=plan,
            )
            command_context = _batch_owned_command_context(
                current_motion_batch=current_motion_batch,
                timing_commands=timing_commands,
                source_event_by_plan_index=source_event_by_plan_index,
            )
            collector.set_runtime_context(
                fsm_state="RECORDING_FAST_REPLAY",
                macro_state_cursor="RECORDING_FAST_REPLAY",
                segment_index=segment_index,
                segment_cursor=segment_index,
                source_step=source_step,
                command_cursor=command_context["command_cursor"],
                source_command=command_context["source_command"],
                source_event_index=command_context["source_event_index"],
                planned_dispatch_time_s=command_context[
                    "planned_dispatch_time_s"
                ],
                actual_dispatch_time_s=command_context[
                    "actual_dispatch_time_s"
                ],
                atomic_batch_id=command_context["atomic_batch_id"],
                dispatch_kind=command_context["dispatch_kind"],
                motion_start_readiness_token=readiness_token,
                scheduler_phase=service.progress.command_phase,
            )
            adapter.step()
            if float(adapter.sim_time) > deadline_sim:
                timed_out = True
                service.stop(
                    adapter,
                    current_sim_time_s=float(adapter.sim_time),
                    current_wall_time_s=time.time(),
                    reason="runner_timeout",
                )
                break
        if not app_stopped and bool(scene_handle.app_is_running()):
            _settle(adapter, args.post_run_settle_s, collector, "RECORDING_POST_SETTLE")
        scheduler_complete = bool(service.stop_reason == "complete" and not timed_out)
        collector.finish_episode(
            success=scheduler_complete,
            reason="scheduler complete; strict physical result is evaluated separately"
            if scheduler_complete
            else str(service.stop_reason or "scheduler failed"),
        )
        video = video_capture.finalize()
        run_dir = collector.run_dir or artifact_root
        _copy_run_inputs(item, run_dir, steps, float(motion.wheel_velocity_limit_rad_s))
        dispatch_ledger = write_source_dispatch_ledger(
            csv_path=run_dir / "V003_DISPATCH_TRACE.csv",
            json_path=run_dir / "V003_DISPATCH_TRACE.json",
            source_version=item.version_id,
            steps=steps,
            plan=plan,
            timing_trace=service.timing_trace,
        )
        wheel_integral_evidence = evaluate_wheel_integral_evidence(
            plan=plan,
            timing_trace=service.timing_trace,
            telemetry_rows=collector.fsm50_rows,
            wheel_direction=float(getattr(adapter, "wheel_direction", 1.0)),
        )
        write_json(
            run_dir / "V003_WHEEL_INTEGRAL_EVIDENCE.json",
            wheel_integral_evidence,
        )
        write_json(
            run_dir / "production_dispatch_timing.json",
            service.timing_trace,
        )
        write_json(
            run_dir / "runtime_environment.json",
            {
                "source_version": item.version_id,
                "contact_mode": contact_mode,
                "actual_viewport_video": bool(
                    video.get("actual_viewport_video", False)
                ),
                "video_path": str(video.get("video_path", "") or ""),
                "video_sha256": str(video.get("video_sha256", "") or ""),
                "viewport_video_manifest_path": str(
                    video.get("manifest_path", "") or ""
                ),
                "viewport_video_manifest_sha256": str(
                    video.get("manifest_sha256", "") or ""
                ),
                "video": _jsonable(video),
                "runtime": _runtime_versions(),
                "scene_config": _jsonable(asdict(scene_handle.config)),
                "batch_live_scene_baseline": _jsonable(live_baseline),
                "live_scene_baseline": _jsonable(version_live_baseline),
                "live_obstacle_geometry": _jsonable(version_live_obstacle),
                "environment_equivalence": version_environment_equivalence,
                "motion_reference": motion.to_dict(),
                "physics_dt_s": float(scene_handle.sim.get_physics_dt()),
                "render_interval": int(scene_handle.config.render_interval),
                "contact_sensor_type": type(scene_handle.contact_sensor).__name__,
                "contact_sensor_error": str(scene_handle.contact_sensor_error or ""),
            },
        )
        result = _result_payload(
            item=item,
            service=service,
            collector=collector,
            respawn=respawn,
            plan=plan,
            run_dir=run_dir,
            timed_out=timed_out,
            motion_start_readiness=motion_start_readiness,
            pre_first_dispatch_readiness=pre_first_dispatch_readiness,
            dispatch_ledger=dispatch_ledger,
            wheel_integral_evidence=wheel_integral_evidence,
            trial_id=trial_id,
            diagnostic_role="",
        )
        result["artifact_root"] = str(artifact_root)
        result["run_dir"] = str(run_dir)
        result["expected_preflight_steps_sha256"] = str(expected_steps_sha256)
        result["simulation_app_stopped"] = bool(app_stopped)
        result["environment_equivalence"] = version_environment_equivalence
        result["contact_mode"] = contact_mode
        result["video"] = _jsonable(video)
        result["actual_viewport_video"] = bool(
            video.get("actual_viewport_video", False)
        )
        result["video_path"] = str(video.get("video_path", "") or "")
        result["video_sha256"] = str(video.get("video_sha256", "") or "")
        result["viewport_video_manifest_path"] = str(
            video.get("manifest_path", "") or ""
        )
        result["viewport_video_manifest_sha256"] = str(
            video.get("manifest_sha256", "") or ""
        )
        result["required_evidence_paths"] = [
            str((run_dir / name).resolve())
            for name in (
                "result.json",
                "failure_diagnostics.json",
                "physical_evidence.json",
                "telemetry_finalization.json",
                "fsm50_telemetry.csv",
                "fsm50_telemetry.jsonl",
                "state_timeline.csv",
                "motion_start_readiness.json",
                "motion_start_pre_first_dispatch.json",
                "V003_DISPATCH_TRACE.csv",
                "V003_DISPATCH_TRACE.json",
                "V003_WHEEL_INTEGRAL_EVIDENCE.json",
                "production_dispatch_timing.json",
                "runtime_environment.json",
                "visual_recording_manifest.json",
                "checksums.sha256",
            )
        ]
        result["required_evidence_paths"].extend(
            str(path)
            for path in _viewport_preclose_evidence_paths(run_dir, video)
        )
        post_source_freeze = _source_freeze(item, robot_usd=robot_usd)
        source_comparison = _compare_source_freezes(source_freeze, post_source_freeze)
        write_json(artifact_root / "source_freeze_post.json", post_source_freeze)
        write_json(artifact_root / "source_integrity.json", source_comparison)
        result["source_integrity"] = {
            "ok": bool(source_comparison["equal"]),
            "scope": "recording_version",
            "comparison": source_comparison,
        }
        if not source_comparison["equal"]:
            _invalidate_result_for_source_drift(
                result,
                drift=source_comparison,
                scope="recording_version",
            )
        result["strict_success"] = bool(result.get("strict_full_success", False))
        try:
            visualization = _generate_fsm50_visualization(
                run_dir,
                fsm50_rows=collector.fsm50_rows,
                state_timeline_rows=collector.state_timeline_rows,
                strict_result=result,
            )
        except Exception as exc:
            visualization = {"ok": False, "error": str(exc)}
        result["visualization"] = visualization
        video_ok = bool(video.get("actual_viewport_video", False))
        artifact_valid = _apply_recording_artifact_policy(
            result,
            video=video,
            visualization=visualization,
        )
        write_json(
            run_dir / "visual_recording_manifest.json",
            _recording_visual_manifest(
                video=video,
                visualization=visualization,
                contact_mode=contact_mode,
                artifact_valid=artifact_valid,
            ),
        )
        write_json(
            run_dir / "failure_diagnostics.json",
            {
                "classification": result["classification"],
                "first_failure_phase": result["first_failure_phase"],
                "scheduler_error": service.last_error,
                "scheduler_info": service.last_info,
                "servo_residual_warnings": service.servo_residual_warnings,
                "strict_physical_evidence": result["physical_evidence"],
                "wheel_integral_evidence": result["wheel_integral_evidence"],
                "wheel_target_integral_verdict": result[
                    "wheel_target_integral_verdict"
                ],
                "source_integrity": result["source_integrity"],
                "contact_mode": contact_mode,
                "video": _jsonable(video),
                "actual_viewport_video": video_ok,
                "artifact_valid": artifact_valid,
            },
        )
        write_json(artifact_root / "artifact_pointer.json", {"run_dir": str(run_dir)})
        # result.json is written only after artifact_root, visualization,
        # source-integrity, and lifecycle fields have their final values.
        write_json(run_dir / "result.json", result)
        _write_checksums(run_dir)
        _mark_artifact_root(artifact_root, valid=artifact_valid)
        return result
    except Exception as exc:
        try:
            service.stop(
                adapter,
                current_sim_time_s=float(adapter.sim_time),
                current_wall_time_s=time.time(),
                reason="runner_exception",
            )
        except Exception:
            pass
        finish_error = ""
        if collector is not None:
            try:
                collector.finish_episode(success=False, reason=str(exc))
            except Exception as finish_exc:
                finish_error = f"{type(finish_exc).__name__}: {finish_exc}"
        run_dir = (collector.run_dir if collector is not None else None) or artifact_root
        try:
            video = (
                video_capture.finalize()
                if video_capture is not None
                else _missing_recording_viewport_video(
                    run_dir,
                    args,
                    contact_mode=contact_mode,
                    error="recording replay failed before viewport capture started",
                )
            )
        except Exception as video_exc:
            try:
                video = _missing_recording_viewport_video(
                    Path(run_dir),
                    args,
                    contact_mode=contact_mode,
                    error=(
                        "viewport finalize raised while preserving failure "
                        f"artifact: {type(video_exc).__name__}: {video_exc}"
                    ),
                )
            except Exception as manifest_exc:
                video = {
                    "valid": False,
                    "artifact_valid": False,
                    "actual_viewport_video": False,
                    "not_camera_video": False,
                    "contact_mode": contact_mode,
                    "video_path": str(
                        Path(run_dir) / "actual_viewport_video.mp4"
                    ),
                    "video_sha256": "",
                    "manifest_path": str(
                        Path(run_dir) / "viewport_video_manifest.json"
                    ),
                    "manifest_sha256": "",
                    "error": (
                        "video failure artifact write failed: "
                        f"{type(manifest_exc).__name__}: {manifest_exc}; "
                        f"original={type(video_exc).__name__}: {video_exc}"
                    ),
                }
        gate_failure = bool(
            "MOTION_START_READY" in str(exc)
            or "production worker MOTION_START_READY" in str(exc)
        )
        if not motion_start_readiness:
            motion_start_readiness = {
                "schema_version": "fsm50.motion_start_readiness_window.v1",
                "gate": "MOTION_START_READY",
                "ready": False,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "rest_qualification": rest_qualification_summary(startup_ground),
                "envelope_status": (
                    "PENDING_THREE_SUCCESSFUL_V003_FAST_REPLAYS"
                ),
            }
        if not pre_first_dispatch_readiness:
            pre_first_dispatch_readiness = {
                "schema_version": "fsm50.motion_start_readiness_window.v1",
                "gate": "MOTION_START_READY_PRE_FIRST_DISPATCH",
                "ready": False,
                "status": "NOT_REACHED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        dispatch_error = ""
        if steps and plan is not None:
            try:
                dispatch_ledger = write_source_dispatch_ledger(
                    csv_path=Path(run_dir) / "V003_DISPATCH_TRACE.csv",
                    json_path=Path(run_dir) / "V003_DISPATCH_TRACE.json",
                    source_version=item.version_id,
                    steps=steps,
                    plan=plan,
                    timing_trace=service.timing_trace,
                )
            except Exception as dispatch_exc:
                dispatch_error = (
                    f"{type(dispatch_exc).__name__}: {dispatch_exc}"
                )
        if not dispatch_ledger:
            dispatch_ledger = {
                "schema_version": "fsm50.source_dispatch_ledger.v1",
                "source_version": item.version_id,
                "complete": False,
                "errors": [
                    dispatch_error
                    or "dispatch ledger unavailable before runner failure"
                ],
            }
            write_csv(Path(run_dir) / "V003_DISPATCH_TRACE.csv", [])
            write_json(
                Path(run_dir) / "V003_DISPATCH_TRACE.json",
                {**dispatch_ledger, "rows": [], "motion_batches": []},
            )
        if not wheel_integral_evidence and plan is not None:
            try:
                wheel_integral_evidence = evaluate_wheel_integral_evidence(
                    plan=plan,
                    timing_trace=service.timing_trace,
                    telemetry_rows=(
                        [] if collector is None else collector.fsm50_rows
                    ),
                    wheel_direction=float(
                        getattr(adapter, "wheel_direction", 1.0)
                    ),
                )
            except Exception as integral_exc:
                wheel_integral_evidence = {
                    "schema_version": 1,
                    "target_integral_verdict": "NOT_EVALUABLE",
                    "measured_tracking_verdict": "NOT_EVALUABLE",
                    "overall_verdict": "NOT_EVALUABLE",
                    "physical_success": False,
                    "structural_errors": [
                        f"{type(integral_exc).__name__}: {integral_exc}"
                    ],
                }
        write_json(
            Path(run_dir) / "V003_WHEEL_INTEGRAL_EVIDENCE.json",
            wheel_integral_evidence,
        )
        write_json(
            Path(run_dir) / "motion_start_readiness.json",
            motion_start_readiness,
        )
        write_json(
            Path(run_dir) / "motion_start_pre_first_dispatch.json",
            pre_first_dispatch_readiness,
        )
        write_json(
            Path(run_dir) / "production_dispatch_timing.json",
            service.timing_trace,
        )
        physical: dict[str, Any] = {}
        if collector is not None:
            try:
                physical = collector.physical_evidence()
            except Exception:
                physical = {}
        failure = {
            "schema_version": "fsm50.recording_replay_result.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_version": item.version_id,
            "accepted_steps_sha256": immediate_steps_sha256,
            "expected_preflight_steps_sha256": str(expected_steps_sha256),
            "trial_id": int(trial_id),
            "classification": (
                "MOTION_START_BLOCKED" if gate_failure else "RUNNER_EXCEPTION"
            ),
            "strict_full_success": False,
            "physical_success": False,
            "scheduler_complete": False,
            "first_failure_phase": (
                "MOTION_START_READY" if gate_failure else "RUNNER_EXCEPTION"
            ),
            "error": f"{type(exc).__name__}: {exc}",
            "telemetry_finish_error": finish_error,
            "artifact_root": str(artifact_root),
            "run_dir": str(run_dir),
            "contact_mode": contact_mode,
            "video": _jsonable(video),
            "actual_viewport_video": bool(
                video.get("actual_viewport_video", False)
            ),
            "video_path": str(video.get("video_path", "") or ""),
            "video_sha256": str(video.get("video_sha256", "") or ""),
            "viewport_video_manifest_path": str(
                video.get("manifest_path", "") or ""
            ),
            "viewport_video_manifest_sha256": str(
                video.get("manifest_sha256", "") or ""
            ),
            "physical_evidence": physical,
            "telemetry_finalization": (
                {}
                if collector is None
                else getattr(
                    collector,
                    "telemetry_finalization_status",
                    lambda: {},
                )()
            ),
            "respawn": _jsonable(respawn),
            "fresh_process_clean_reset": True,
            "motion_start_readiness": _jsonable(motion_start_readiness),
            "motion_start_pre_first_dispatch": _jsonable(
                pre_first_dispatch_readiness
            ),
            "motion_start_ready": False,
            "dispatch_ledger": _jsonable(dispatch_ledger),
            "dispatch_complete": False,
            "wheel_integral_evidence": _jsonable(
                wheel_integral_evidence
            ),
            "wheel_target_integral_verdict": str(
                wheel_integral_evidence.get(
                    "target_integral_verdict", "NOT_EVALUABLE"
                )
            ),
            "measured_wheel_tracking_verdict": str(
                wheel_integral_evidence.get(
                    "measured_tracking_verdict", "NOT_EVALUABLE"
                )
            ),
            "visualization": {"ok": False, "error": "not generated"},
            "artifact_valid": False,
            "strict_success": False,
            "lifecycle": {
                "finalized": False,
                "failed": True,
                "strict_success": False,
            },
        }
        try:
            post_source_freeze = _source_freeze(item, robot_usd=robot_usd)
            source_comparison = _compare_source_freezes(source_freeze, post_source_freeze)
            write_json(artifact_root / "source_freeze_post.json", post_source_freeze)
            write_json(artifact_root / "source_integrity.json", source_comparison)
            failure["source_integrity"] = {
                "ok": bool(source_comparison["equal"]),
                "scope": "recording_version",
                "comparison": source_comparison,
            }
            if not source_comparison["equal"]:
                failure["runner_error"] = failure["error"]
                _invalidate_result_for_source_drift(
                    failure,
                    drift=source_comparison,
                    scope="recording_version",
                )
        except Exception as freeze_exc:
            failure["source_integrity"] = {
                "ok": False,
                "scope": "recording_version",
                "error": f"{type(freeze_exc).__name__}: {freeze_exc}",
            }
        if (
            not bool(video.get("actual_viewport_video", False))
            and bool(dict(failure.get("source_integrity", {}) or {}).get("ok", False))
        ):
            failure["classification_before_artifact_validation"] = failure.get(
                "classification"
            )
            failure["classification"] = "ARTIFACT_INVALID"
            failure["first_failure_phase"] = "VIEWPORT_VIDEO_MISSING_OR_INVALID"
        try:
            visualization = _generate_fsm50_visualization(
                Path(run_dir),
                fsm50_rows=[] if collector is None else collector.fsm50_rows,
                state_timeline_rows=[] if collector is None else collector.state_timeline_rows,
                strict_result=failure,
            )
        except Exception as visual_exc:
            visualization = {"ok": False, "error": str(visual_exc)}
        failure["visualization"] = visualization
        required_names = (
            "result.json",
            "failure_diagnostics.json",
            "motion_start_readiness.json",
            "motion_start_pre_first_dispatch.json",
            "V003_DISPATCH_TRACE.csv",
            "V003_DISPATCH_TRACE.json",
            "V003_WHEEL_INTEGRAL_EVIDENCE.json",
            "production_dispatch_timing.json",
            "runtime_environment.json",
            "visual_recording_manifest.json",
            "checksums.sha256",
        )
        optional_names = (
            "physical_evidence.json",
            "telemetry_finalization.json",
            "fsm50_telemetry.csv",
            "fsm50_telemetry.jsonl",
            "state_timeline.csv",
        )
        failure["required_evidence_paths"] = [
            str((Path(run_dir) / name).resolve()) for name in required_names
        ] + [
            str((Path(run_dir) / name).resolve())
            for name in optional_names
            if (Path(run_dir) / name).is_file()
        ]
        failure["required_evidence_paths"].extend(
            str(path)
            for path in _viewport_preclose_evidence_paths(Path(run_dir), video)
        )
        runtime_environment = {
            "source_version": item.version_id,
            "contact_mode": contact_mode,
            "actual_viewport_video": bool(
                video.get("actual_viewport_video", False)
            ),
            "video_path": str(video.get("video_path", "") or ""),
            "video_sha256": str(video.get("video_sha256", "") or ""),
            "viewport_video_manifest_path": str(
                video.get("manifest_path", "") or ""
            ),
            "viewport_video_manifest_sha256": str(
                video.get("manifest_sha256", "") or ""
            ),
            "video": _jsonable(video),
            "runtime": _runtime_versions(),
            "scene_config": _jsonable(asdict(scene_handle.config)),
            "batch_live_scene_baseline": _jsonable(live_baseline),
            "live_scene_baseline": _jsonable(version_live_baseline),
            "live_obstacle_geometry": _jsonable(version_live_obstacle),
            "environment_equivalence": _jsonable(
                version_environment_equivalence
            ),
            "motion_reference": (
                {} if motion is None else _jsonable(motion.to_dict())
            ),
            "failure": f"{type(exc).__name__}: {exc}",
        }
        finalization_errors: list[str] = []
        for label, action in (
            (
                "runtime_environment",
                lambda: write_json(
                    Path(run_dir) / "runtime_environment.json",
                    runtime_environment,
                ),
            ),
            (
                "visual_manifest",
                lambda: write_json(
                    Path(run_dir) / "visual_recording_manifest.json",
                    _recording_visual_manifest(
                        video=video,
                        visualization=visualization,
                        contact_mode=contact_mode,
                        artifact_valid=False,
                    ),
                ),
            ),
            ("failure_diagnostics", lambda: write_json(Path(run_dir) / "failure_diagnostics.json", failure)),
            ("result", lambda: write_json(Path(run_dir) / "result.json", failure)),
            ("checksums", lambda: _write_checksums(Path(run_dir))),
        ):
            try:
                action()
            except Exception as finalize_exc:
                finalization_errors.append(
                    f"{label}: {type(finalize_exc).__name__}: {finalize_exc}"
                )
        if finalization_errors:
            failure["finalization_errors"] = finalization_errors
            try:
                write_json(Path(run_dir) / "result.json", failure)
                _write_checksums(Path(run_dir))
            except Exception:
                pass
        try:
            write_json(artifact_root / "artifact_pointer.json", {"run_dir": str(run_dir)})
        except Exception:
            pass
        _mark_artifact_root(artifact_root, valid=False)
        return failure
    finally:
        if service.active:
            try:
                service.stop(
                    adapter,
                    current_sim_time_s=float(adapter.sim_time),
                    current_wall_time_s=time.time(),
                    reason="runner_finally",
                )
            except Exception:
                pass
        try:
            adapter.attach_telemetry(None)
        except Exception:
            pass


def _append_runtime_lock(
    readback: dict[str, Any], *, path: Path = ENVIRONMENT_LOCK_PATH
) -> None:
    path = Path(path).resolve()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    rows = list(payload.get("runtime_readbacks", []) or [])
    rows.append(readback)
    payload["runtime_readbacks"] = rows
    payload["status"] = "runtime_readback_available"
    write_json(path, payload)


def _update_required_reports(results: list[dict[str, Any]]) -> None:
    existing: list[dict[str, str]] = []
    fields: list[str] = []
    if VERSION_MATRIX_PATH.exists():
        with VERSION_MATRIX_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            existing = list(reader)
    by_version = {str(row.get("source_version")): row for row in results}
    for row in existing:
        result = by_version.get(str(row.get("version")))
        if not result:
            continue
        physical = dict(result.get("physical_evidence", {}) or {})
        for leg in ("FR", "FL", "RR", "RL"):
            lift, place = _leg_summary(physical, leg)
            row[f"{leg}_lift_success"] = str(lift).upper()
            row[f"{leg}_place_success"] = str(place).upper()
        row["full_success"] = str(bool(result.get("strict_full_success", False))).upper()
        row["first_failure_phase"] = str(result.get("first_failure_phase", ""))
        row["final_recovery_success"] = str(
            bool(physical.get("final_all_top", False) and physical.get("final_all_loaded", False))
        ).upper()
        row["maximum_roll_rad"] = result.get("maximum_abs_roll_rad", "")
        row["maximum_pitch_rad"] = result.get("maximum_abs_pitch_rad", "")
        row["minimum_support_margin_m"] = result.get("minimum_support_polygon_margin_m", "")
        row["maximum_contact_drift_m"] = result.get("maximum_contact_drift_m", "")
        rotations = [
            float(item.get("loaded_front_face_rotation_rad", 0.0) or 0.0)
            for item in dict(physical.get("traversal", {}).get("legs", {}) or {}).values()
        ]
        row["loaded_front_face_wheel_rotation_rad"] = max(rotations or [0.0])
        row["replay_evidence"] = str(result.get("run_dir", ""))
        row["comments"] = f"{result.get('classification', '')}; scheduler_complete={result.get('scheduler_complete', False)}; physical_success={result.get('physical_success', False)}"
    if fields and existing:
        with VERSION_MATRIX_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
    timeline: list[dict[str, Any]] = []
    for result in results:
        path = Path(str(result.get("run_dir", ""))) / "state_timeline.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                timeline.append(
                    {
                        "source_version": result.get("source_version"),
                        "run_dir": result.get("run_dir"),
                        **row,
                    }
                )
    write_csv(DIAGONAL_TIMELINE_PATH, timeline)


def _apply_batch_source_drift(
    results: list[dict[str, Any]], drift: dict[str, Any]
) -> None:
    for result in results:
        if _result_is_worker_owned(result):
            # A post-run batch source mismatch invalidates the batch, not the
            # already sealed worker artifact.  The caller records the mismatch
            # in batch source_integrity/finalization and returns failure.
            continue
        _invalidate_result_for_source_drift(result, drift=drift, scope="batch")
        run_dir = Path(str(result.get("run_dir", "") or ""))
        artifact_root = Path(str(result.get("artifact_root", "") or ""))
        if not run_dir.is_dir():
            continue
        try:
            result["visualization"] = _generate_fsm50_visualization(
                run_dir,
                fsm50_rows=_load_csv_rows(run_dir / "fsm50_telemetry.csv"),
                state_timeline_rows=_load_csv_rows(run_dir / "state_timeline.csv"),
                strict_result=result,
            )
        except Exception as exc:
            result["visualization"] = {"ok": False, "error": str(exc)}
        write_json(
            run_dir / "visual_recording_manifest.json",
            {
                **_recording_visual_manifest(
                    video=dict(result.get("video", {}) or {}),
                    visualization=dict(result.get("visualization", {}) or {}),
                    contact_mode=str(result.get("contact_mode", "") or ""),
                    artifact_valid=False,
                ),
                "source_integrity": result["source_integrity"],
            },
        )
        write_json(run_dir / "failure_diagnostics.json", result)
        write_json(run_dir / "result.json", result)
        _write_checksums(run_dir)
        if artifact_root.is_dir():
            _mark_artifact_root(artifact_root, valid=False)


def _run_grounding_only_locked(
    args: argparse.Namespace,
    *,
    process_snapshot: list[dict[str, Any]],
    supervisor: ChildSupervisorHandshake | None = None,
) -> int:
    """Run one fresh-process formal grounding probe and preserve failures."""

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
    from .grounding_diagnostics import (
        GroundingTraceWriter,
        analyze_joint_trace,
        write_grounding_trace_csv,
    )

    output_root = Path(args.output_root).resolve()
    batch_root = _new_directory(
        output_root,
        f"grounding_only_trial_{int(getattr(args, 'trial_id', 0) or 0):02d}",
    )
    if supervisor is not None:
        supervisor.announce_batch(batch_root)
    (batch_root / ".partial").write_text("running\n", encoding="utf-8")
    diagnostic_dir = batch_root / "diagnostic_artifact"
    qualification_dir = batch_root / "qualification_artifact"
    diagnostic_dir.mkdir()
    qualification_dir.mkdir()

    robot_usd = Path(args.robot_usd or DEFAULT_ROBOT_USD_PATH).resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot USD not found: {robot_usd}")
    environment_lock_path = Path(args.report_root).resolve() / "environment_lock_50mm.json"
    environment_lock = _load_environment_lock(environment_lock_path)
    locked_source_check = _verify_locked_source_hashes(environment_lock)
    robot_hash = sha256_file(robot_usd).lower()
    locked_robot_hash = str(
        dict(environment_lock.get("selected_environment", {}) or {}).get(
            "robot_usd_sha256", ""
        )
        or ""
    ).lower()
    prelaunch_environment_ok = bool(
        locked_source_check.get("ok", False)
        and locked_robot_hash
        and robot_hash == locked_robot_hash
    )
    batch_source_freeze = _source_freeze(robot_usd=robot_usd)
    trial_id = int(getattr(args, "trial_id", 0) or 0)
    _atomic_write_json(
        batch_root / "batch_request.json",
        {
            "schema_version": "fsm50.grounding_only_request.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": "grounding-only",
            "trial_id": trial_id,
            "args": vars(args),
            "source_freeze": batch_source_freeze,
            "environment_lock_path": str(environment_lock_path),
            "prelaunch_environment_validation": {
                "ok": prelaunch_environment_ok,
                "locked_source_hashes": locked_source_check,
                "locked_robot_usd_sha256": locked_robot_hash,
                "actual_robot_usd_sha256": robot_hash,
            },
            "process_preflight": _jsonable(process_snapshot),
            "clean_reset_contract": {
                "fresh_supervised_child": True,
                "historical_root_pose_seed_allowed": False,
                "trial_id": trial_id,
            },
        },
    )

    motion = load_motion_reference()
    scene_config = SimSceneConfig(
        obstacle_height_m=0.05,
        robot_usd=robot_usd,
        save_usd=PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd",
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
    # Formal mode retains the production aggregate sensor constructor.  It is
    # observation-only and supplies full-body net_forces_w for this diagnostic.
    scene_config.telemetry_contact_sensors_enabled = True
    scene_config.contact_sensor_factory = None

    simulation_app = None
    scene_handle = None
    adapter = None
    trace_writer = None
    capture: _RecordingReplayViewportCapture | None = None
    ground: dict[str, Any] = {}
    video: dict[str, Any] = {}
    run_error = ""
    live_baseline: dict[str, Any] = {}
    live_obstacle: dict[str, Any] = {}
    try:
        if not prelaunch_environment_ok:
            raise RuntimeError(
                "environment lock prelaunch validation failed; see batch_request.json"
            )
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
        capture = _RecordingReplayViewportCapture(
            diagnostic_dir,
            args,
            contact_mode="formal_grounding_diagnostic",
            adapter=adapter,
        )
        capture.start()
        trace_writer = GroundingTraceWriter(
            diagnostic_dir / "grounding_telemetry.jsonl",
            adapter=adapter,
            scene_handle=scene_handle,
        )
        ground = initialize_adapter_ground_reference(
            adapter,
            tick_observer=trace_writer,
        )
        ground["locked_ground_seed"] = {
            "applied": False,
            "reason": "historical locked pose is read-only comparison evidence",
            "initialization_path": "formal_worker_unseeded_ground_reference",
        }
        finalize_scene_after_grounding(scene_handle)
        live_baseline = measure_scene_baseline(scene_handle, adapter)
        live_obstacle = measure_obstacle_geometry(scene_handle)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
    finally:
        if trace_writer is not None:
            try:
                trace_writer.close()
            except Exception as exc:
                run_error = run_error or f"trace close failed: {type(exc).__name__}: {exc}"
        if capture is not None:
            try:
                video = capture.finalize()
            except Exception as exc:
                video = _missing_recording_viewport_video(
                    diagnostic_dir,
                    args,
                    contact_mode="formal_grounding_diagnostic",
                    error=(
                        "capture finalize failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
        else:
            video = _missing_recording_viewport_video(
                diagnostic_dir,
                args,
                contact_mode="formal_grounding_diagnostic",
                error="viewport capture was not constructed",
            )

    rows = [] if trace_writer is None else list(trace_writer.rows)
    if not (diagnostic_dir / "grounding_telemetry.jsonl").is_file():
        (diagnostic_dir / "grounding_telemetry.jsonl").write_text(
            "", encoding="utf-8"
        )
    write_grounding_trace_csv(
        diagnostic_dir / "grounding_telemetry.csv",
        rows,
    )
    joint_analysis = analyze_joint_trace(
        rows,
        servo_threshold_rad_s=0.02,
        wheel_threshold_rad_s=0.20,
    )
    _atomic_write_json(
        diagnostic_dir / "joint_velocity_analysis.json",
        joint_analysis,
    )
    _atomic_write_json(
        diagnostic_dir / "ground_initialization.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "trial_id": trial_id,
            "ground_reference": _jsonable(ground),
        },
    )
    source_post = _source_freeze(robot_usd=robot_usd)
    source_integrity = _compare_source_freezes(batch_source_freeze, source_post)
    _atomic_write_json(batch_root / "source_integrity.json", source_integrity)
    physical_pass = bool(ground and ground_reference_result_is_valid(ground))
    grounded_diagnostics = dict(
        ground.get(
            "grounded_reference_diagnostics",
            ground.get("ground_diagnostics", {}),
        )
        or {}
    )
    ground_failure_reason = str(
        grounded_diagnostics.get("ground_reference_block_reason", "")
        or ground.get("ground_reference_block_reason", "")
        or "; ".join(grounded_diagnostics.get("reasons", []) or [])
        or ""
    )
    trace_complete = bool(
        rows
        and int(ground.get("steps_run", -1) or -1) == len(rows)
        and len(rows) <= 180
        and all(row.get("diagnostic_evidence_valid") is True for row in rows)
        and not list(ground.get("diagnostic_observer_errors", []) or [])
    )
    video_valid = bool(video.get("valid", False))
    diagnostic_valid = bool(
        trace_complete
        and video_valid
        and source_integrity.get("equal", False)
        and not run_error
    )
    runtime_readback = {
        "schema_version": "fsm50.grounding_runtime_readback.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial_id": trial_id,
        "runtime": _runtime_versions(),
        "scene_config": _jsonable(asdict(scene_config)),
        "physics_dt_s": (
            None
            if scene_handle is None
            else float(scene_handle.sim.get_physics_dt())
        ),
        "live_scene_baseline": _jsonable(live_baseline),
        "live_obstacle_geometry": _jsonable(live_obstacle),
        "contact_sensor_type": (
            ""
            if scene_handle is None or scene_handle.contact_sensor is None
            else type(scene_handle.contact_sensor).__name__
        ),
        "contact_sensor_error": (
            "" if scene_handle is None else str(scene_handle.contact_sensor_error or "")
        ),
        "locked_ground_seed_applied": False,
    }
    _atomic_write_json(
        diagnostic_dir / "runtime_environment_readback.json",
        runtime_readback,
    )
    diagnostic_result = {
        "schema_version": "fsm50.grounding_diagnostic_result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "GROUNDING_DIAGNOSTIC",
        "can_satisfy_grounding_qualification": False,
        "trial_id": trial_id,
        "run_dir": str(diagnostic_dir),
        "artifact_valid": diagnostic_valid,
        "physical_grounding_pass": physical_pass,
        "classification": (
            "GROUNDING_DIAGNOSTIC_COMPLETE"
            if diagnostic_valid
            else "GROUNDING_DIAGNOSTIC_INVALID"
        ),
        "grounding_result": "PASS" if physical_pass else "FAIL",
        "run_error": run_error,
        "trace_complete": trace_complete,
        "trace_count": len(rows),
        "video": video,
        "source_integrity": source_integrity,
        "joint_velocity_analysis": joint_analysis,
        "ground_reference": _jsonable(ground),
        "lifecycle": {
            "finalized": diagnostic_valid,
            "failed": not diagnostic_valid,
            "strict_success": False,
        },
    }
    _atomic_write_json(diagnostic_dir / "result.json", diagnostic_result)
    _atomic_write_json(
        diagnostic_dir / "failure_diagnostics.json",
        {
            **diagnostic_result,
            "failure_reason": (
                ""
                if physical_pass and diagnostic_valid
                else run_error
                or ground_failure_reason
                or "grounding did not satisfy strict qualification"
            ),
        },
    )
    _write_checksums(diagnostic_dir)
    _mark_artifact_root(diagnostic_dir, valid=diagnostic_valid)

    evidence_paths: list[Path] = [
        diagnostic_dir / "result.json",
        diagnostic_dir / "failure_diagnostics.json",
        diagnostic_dir / "ground_initialization.json",
        diagnostic_dir / "grounding_telemetry.jsonl",
        diagnostic_dir / "grounding_telemetry.csv",
        diagnostic_dir / "joint_velocity_analysis.json",
        diagnostic_dir / "runtime_environment_readback.json",
        diagnostic_dir / "viewport_video_manifest.json",
        diagnostic_dir / "checksums.sha256",
    ]
    video_path = Path(
        str(
            video.get(
                "video_path",
                diagnostic_dir / "actual_viewport_video.mp4",
            )
        )
    ).resolve()
    if video_valid and video_path.is_file():
        evidence_paths.append(video_path)
    qualification_pass = bool(physical_pass and diagnostic_valid)
    evidence_manifest = {
        str(path.resolve()): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in evidence_paths
        if path.is_file()
    }
    _atomic_write_json(
        qualification_dir / "evidence_manifest.json",
        {
            "schema_version": "fsm50.grounding_qualification_evidence.v1",
            "trial_id": trial_id,
            "diagnostic_artifact_root": str(diagnostic_dir),
            "evidence_files": evidence_manifest,
        },
    )
    qualification_result = {
        "schema_version": "fsm50.grounding_qualification_result.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_kind": "GROUNDING_QUALIFICATION",
        "trial_id": trial_id,
        "run_dir": str(qualification_dir),
        "artifact_root": str(qualification_dir),
        "qualified": qualification_pass,
        "strict_success": qualification_pass,
        "classification": "GROUNDING_PASS" if qualification_pass else "GROUNDING_FAIL",
        "diagnostic_artifact_root": str(diagnostic_dir),
        "diagnostic_artifact_valid": diagnostic_valid,
        "physical_grounding_pass": physical_pass,
        "trace_count": len(rows),
        "steps_run": ground.get("steps_run"),
        "consecutive_stable_ticks": ground.get("consecutive_stable_ticks"),
        "stop_reason": ground.get("stop_reason"),
        "video_valid": video_valid,
        "failure_reason": (
            ""
            if qualification_pass
            else run_error
            or ground_failure_reason
            or "grounding qualification failed"
        ),
        "lifecycle": {
            "finalized": qualification_pass,
            "failed": not qualification_pass,
            "strict_success": qualification_pass,
        },
    }
    _atomic_write_json(qualification_dir / "result.json", qualification_result)
    _atomic_write_json(
        qualification_dir / "failure_diagnostics.json",
        qualification_result,
    )
    _write_checksums(qualification_dir)
    qualification_required = [
        qualification_dir / "result.json",
        qualification_dir / "failure_diagnostics.json",
        qualification_dir / "evidence_manifest.json",
        qualification_dir / "checksums.sha256",
        *evidence_paths,
    ]
    qualification_result["required_evidence_paths"] = [
        str(path.resolve()) for path in qualification_required
    ]
    # Bind the required-path contract into the final result bytes, then refresh
    # its checksum before batch preclose is built.
    _atomic_write_json(qualification_dir / "result.json", qualification_result)
    _atomic_write_json(
        qualification_dir / "failure_diagnostics.json",
        qualification_result,
    )
    _write_checksums(qualification_dir)
    _mark_artifact_root(qualification_dir, valid=qualification_pass)

    batch_results = [qualification_result]
    _atomic_write_json(batch_root / "batch_results.json", batch_results)
    batch_finalization = {
        "schema_version": "fsm50.batch_finalization.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(batch_root),
        "command": "grounding-only",
        "trial_id": trial_id,
        "finalized": diagnostic_valid,
        "failed": not diagnostic_valid,
        "strict_success": qualification_pass,
        "physical_result": "PASS" if physical_pass else "FAIL",
        "diagnostic_artifact_valid": diagnostic_valid,
        "batch_error": run_error,
        "finalization_errors": [],
        "phase": "PRECLOSE_FINALIZED",
        "close_error": "",
    }
    _atomic_write_json(batch_root / "batch_finalization.json", batch_finalization)
    _write_checksums(batch_root)
    batch_artifact_valid = bool(diagnostic_valid)
    retry_snapshot_errors: list[str] = []
    snapshot_errors = _snapshot_preclose_files(batch_root)
    if snapshot_errors:
        batch_artifact_valid = False
        qualification_pass = False
        qualification_result.update(
            {
                "qualified": False,
                "strict_success": False,
                "classification": "GROUNDING_FAIL",
                "failure_reason": "preclose snapshot failed: "
                + "; ".join(snapshot_errors),
                "lifecycle": {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                },
            }
        )
        _atomic_write_json(
            qualification_dir / "result.json", qualification_result
        )
        _atomic_write_json(
            qualification_dir / "failure_diagnostics.json",
            qualification_result,
        )
        _write_checksums(qualification_dir)
        _mark_artifact_root(qualification_dir, valid=False)
        batch_results = [qualification_result]
        _atomic_write_json(batch_root / "batch_results.json", batch_results)
        batch_finalization.update(
            {
                "finalized": False,
                "failed": True,
                "strict_success": False,
                "finalization_errors": snapshot_errors,
            }
        )
        _atomic_write_json(batch_root / "batch_finalization.json", batch_finalization)
        _write_checksums(batch_root, exclude_preclose_snapshots=True)
        retry_snapshot_errors = _snapshot_preclose_files(batch_root)
        if retry_snapshot_errors:
            batch_finalization["finalization_errors"] = [
                *snapshot_errors,
                *[
                    f"snapshot_retry: {error}"
                    for error in retry_snapshot_errors
                ],
            ]
    _mark_artifact_root(batch_root, valid=batch_artifact_valid)
    preclose_evidence = _preclose_evidence_manifest(
        batch_root,
        results=batch_results,
        batch_source_comparison=source_integrity,
        batch_finalization=batch_finalization,
        include_global_analysis_reports=False,
    )
    if retry_snapshot_errors:
        preclose_evidence["immutable_preclose_errors"] = [
            f"snapshot_retry: {error}" for error in retry_snapshot_errors
        ]
    if supervisor is not None:
        supervisor.mark_preclose(preclose_evidence)
    else:
        _atomic_write_json(
            batch_root / "preclose_complete.json",
            {
                "schema_version": "fsm50.preclose_complete.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "token": "",
                "parent_pid": os.getppid(),
                "child_pid": os.getpid(),
                "batch_root": str(batch_root),
                "evidence": preclose_evidence,
            },
        )

    exit_code = 0 if qualification_pass else 1
    close_error = ""
    shutdown_mode = str(getattr(args, "shutdown_mode", "fast") or "fast")
    try:
        shutdown_mode = _close_simulation_with_explicit_policy(
            simulation_app=simulation_app,
            scene_handle=scene_handle,
            args=args,
            supervisor=supervisor,
            intended_returncode=(0 if shutdown_mode == "fast" else exit_code),
        )
    except Exception as exc:
        close_error = f"{type(exc).__name__}: {exc}"
        exit_code = 1
        if supervisor is not None:
            supervisor.mark_close_error(
                shutdown_mode=shutdown_mode,
                error=close_error,
            )
    print(
        json.dumps(
            {
                "batch_root": str(batch_root),
                "trial_id": trial_id,
                "grounding_result": "PASS" if qualification_pass else "FAIL",
                "diagnostic_artifact_valid": diagnostic_valid,
                "close_error": close_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return _process_returncode_after_close(
        command_exit_code=exit_code,
        shutdown_mode=shutdown_mode,
        close_error=close_error,
        supervised=supervisor is not None,
    )


def _run_recording_replays_direct_legacy_locked(
    args: argparse.Namespace,
    *,
    process_snapshot: list[dict[str, Any]],
    supervisor: ChildSupervisorHandshake | None = None,
) -> int:
    # Isaac imports occur only after AppLauncher has started SimulationApp.
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
    from sim_worker_runtime import ground_reference_result_is_valid, initialize_adapter_ground_reference

    audit = RecordingAudit(Path(args.recording_root), Path(args.report_root))
    selected = _select_versions(audit.enumerate_versions(), args.versions)
    if not selected:
        raise RuntimeError("no recording versions selected")
    if len(selected) != 1:
        raise RuntimeError(
            "recording replay requires exactly one selected version per fresh "
            "supervised SimulationApp process; invoke versions serially"
        )
    trial_id = int(getattr(args, "trial_id", 0) or 0)
    if trial_id < 1:
        raise RuntimeError("recording replay requires a positive --trial-id")
    preflight_audits = _fail_closed_recording_audits(audit, selected)
    expected_hashes = {
        str(row["version"]): str(row["accepted_steps_sha256"])
        for row in preflight_audits
    }
    robot_usd = Path(args.robot_usd or DEFAULT_ROBOT_USD_PATH).resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot USD not found: {robot_usd}")
    environment_lock_path = Path(args.report_root).resolve() / "environment_lock_50mm.json"
    environment_lock = _load_environment_lock(environment_lock_path)
    locked_source_check = _verify_locked_source_hashes(environment_lock)
    robot_hash = sha256_file(robot_usd).lower()
    metadata_robot_checks = [
        {
            "version": str(row.get("version", "")),
            "expected_robot_asset_sha256": str(
                dict(row.get("metadata", {}) or {}).get("robot_asset_sha256", "") or ""
            ).lower(),
            "actual_robot_usd_sha256": robot_hash,
        }
        for row in preflight_audits
    ]
    for row in metadata_robot_checks:
        row["ok"] = bool(
            row["expected_robot_asset_sha256"]
            and row["expected_robot_asset_sha256"] == robot_hash
        )
    prelaunch_environment_ok = bool(
        locked_source_check.get("ok", False)
        and metadata_robot_checks
        and all(row["ok"] for row in metadata_robot_checks)
        and str(
            dict(environment_lock.get("selected_environment", {}) or {}).get(
                "robot_usd_sha256", ""
            )
        ).lower()
        == robot_hash
    )
    batch_source_freeze = _source_freeze(robot_usd=robot_usd)
    batch_root = _new_directory(Path(args.output_root).resolve(), "recording_replays")
    if supervisor is not None:
        supervisor.announce_batch(batch_root)
    batch_partial = batch_root / ".partial"
    batch_partial.write_text("running\n", encoding="utf-8")
    write_json(
        batch_root / "batch_request.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": [item.version_id for item in selected],
            "trial_id": trial_id,
            "resume": bool(getattr(args, "resume", False)),
            "resume_skipped": _jsonable(
                list(getattr(args, "resume_skipped", []) or [])
            ),
            "args": vars(args),
            "source_freeze": batch_source_freeze,
            "recording_preflight_audits": [
                _compact_recording_audit(row) for row in preflight_audits
            ],
            "environment_lock_path": str(environment_lock_path),
            "prelaunch_environment_validation": {
                "ok": prelaunch_environment_ok,
                "locked_source_hashes": locked_source_check,
                "metadata_robot_checks": metadata_robot_checks,
                "locked_robot_usd_sha256": str(
                    dict(environment_lock.get("selected_environment", {}) or {}).get(
                        "robot_usd_sha256", ""
                    )
                ).lower(),
            },
            "process_preflight": [
                {
                    "pid": row.get("pid"),
                    "name": row.get("name"),
                    "command_line_sha256": hashlib.sha256(
                        str(row.get("command_line", "")).encode("utf-8")
                    ).hexdigest(),
                }
                for row in process_snapshot
            ],
        },
    )
    motion = load_motion_reference()
    scene_config = SimSceneConfig(
        obstacle_height_m=0.05,
        robot_usd=robot_usd,
        save_usd=PROJECT_ROOT / "usd" / "wlr_robot_height_replay_env.usd",
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
    contact_mode = str(getattr(args, "contact_mode", "instrumented") or "instrumented")
    if contact_mode == "formal":
        # Environment-equivalence baseline A: retain the production scene's
        # original aggregate ContactSensor constructor.  The telemetry flag
        # only activates observation and is an explicitly allowed A/B field.
        scene_config.telemetry_contact_sensors_enabled = True
        scene_config.contact_sensor_factory = None
    elif contact_mode == "instrumented":
        configure_scene_for_wheel_and_nonwheel_contacts(
            scene_config,
            wheel_factory=make_filtered_wheel_contact_sensor_factory(
                force_threshold_n=1.0
            ),
            force_threshold_n=1.0,
        )
    else:
        raise ValueError(f"unsupported contact_mode: {contact_mode}")
    simulation_app = None
    scene_handle = None
    results: list[dict[str, Any]] = []
    exit_code = 0
    batch_error = ""
    try:
        if not prelaunch_environment_ok:
            raise RuntimeError(
                "environment lock prelaunch validation failed; see batch_request.json"
            )
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
        # Keep the formal worker's initialization path byte-for-byte in spirit:
        # scene -> adapter -> initialize_adapter_ground_reference.  The
        # environment lock contains an earlier measured grounded pose, but it
        # is comparison evidence, not an initial-state command.  Writing that
        # historical root pose here changes reset/grounding behavior and makes
        # the replay environment non-equivalent to the production worker.
        locked_ground_seed = {
            "applied": False,
            "reason": "historical locked pose is read-only comparison evidence",
            "initialization_path": "formal_worker_unseeded_ground_reference",
        }
        ground = initialize_adapter_ground_reference(adapter)
        ground["locked_ground_seed"] = locked_ground_seed
        write_json(
            batch_root / "ground_initialization.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "locked_ground_seed": locked_ground_seed,
                "ground_reference": _jsonable(ground),
            },
        )
        rest_qualification = rest_qualification_summary(ground)
        write_json(batch_root / "rest_qualification.json", rest_qualification)
        finalize_scene_after_grounding(scene_handle)
        live_baseline = measure_scene_baseline(scene_handle, adapter)
        live_obstacle = measure_obstacle_geometry(scene_handle)
        environment_equivalence = _runtime_environment_equivalence(
            lock=environment_lock,
            scene_config=scene_config,
            live_baseline=live_baseline,
            live_obstacle=live_obstacle,
            motion=motion,
            robot_usd=robot_usd,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
        )
        readback = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "batch_root": str(batch_root),
            "selected_versions": [item.version_id for item in selected],
            "scene_config": _jsonable(asdict(scene_config)),
            "live_scene_baseline": _jsonable(live_baseline),
            "live_obstacle_geometry": _jsonable(live_obstacle),
            "ground_reference": _jsonable(ground),
            "rest_qualification": rest_qualification,
            "rest_qualification_required_for_motion_start": False,
            "runtime": _runtime_versions(),
            "contact_sensor_type": type(scene_handle.contact_sensor).__name__,
            "contact_sensor_error": str(scene_handle.contact_sensor_error or ""),
            "contact_mode": contact_mode,
            "environment_equivalence": environment_equivalence,
        }
        write_json(batch_root / "runtime_environment_readback.json", readback)
        if not environment_equivalence.get("ok", False):
            raise RuntimeError(
                "runtime environment differs from environment_lock_50mm.json; "
                "see runtime_environment_readback.json"
            )
        if scene_handle.contact_sensor is None or scene_handle.contact_sensor_error:
            raise RuntimeError(
                "filtered contact sensor unavailable: " + str(scene_handle.contact_sensor_error)
            )
        # The audited environment lock is immutable input for all three Gate-1
        # trials.  Runtime readback belongs to this batch artifact and must not
        # rewrite the shared qualification lock between clean processes.
        for item in selected:
            print(f"[FSM50] clean Fast replay starting: {item.version_id}", flush=True)
            result = _run_recording_version(
                item=item,
                adapter=adapter,
                scene_handle=scene_handle,
                live_baseline=live_baseline,
                batch_root=batch_root,
                args=args,
                robot_usd=robot_usd,
                expected_steps_sha256=expected_hashes[item.version_id],
                environment_lock=environment_lock,
                startup_ground=ground,
                trial_id=trial_id,
            )
            results.append(result)
            write_json(batch_root / "batch_results.json", results)
            print(
                f"[FSM50] {item.version_id}: scheduler={result.get('scheduler_complete')} "
                f"physical={result.get('physical_success')} class={result.get('classification')}",
                flush=True,
            )
            if result.get("classification") in {
                "RUNNER_EXCEPTION",
                "SOURCE_DRIFT",
                "ARTIFACT_INVALID",
            }:
                exit_code = 1
                if not args.continue_on_error:
                    break
    except Exception as exc:
        exit_code = 1
        batch_error = f"{type(exc).__name__}: {exc}"
        write_json(
            batch_root / "batch_failure.json",
            {"error": batch_error, "results_so_far": results},
        )
        print(f"[FSM50] batch failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        # Finalize all physics, source-integrity, visualization, and report
        # evidence before entering SimulationApp.close(), which may hang in
        # native shutdown.  The parent starts its only timeout after the
        # durable preclose marker below becomes visible.
        batch_source_comparison: dict[str, Any]
        try:
            batch_source_post = _source_freeze(robot_usd=robot_usd)
            batch_source_comparison = _compare_source_freezes(
                batch_source_freeze, batch_source_post
            )
            write_json(batch_root / "source_freeze_post.json", batch_source_post)
            write_json(
                batch_root / "source_integrity.json", batch_source_comparison
            )
            if not batch_source_comparison["equal"]:
                exit_code = 1
                _apply_batch_source_drift(results, batch_source_comparison)
        except Exception as exc:
            batch_source_comparison = {
                "equal": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            exit_code = 1
        finalization_errors: list[str] = []
        batch_valid = bool(
            not batch_error
            and batch_source_comparison.get("equal", False)
            and not finalization_errors
            and results
            and all(
                dict(result.get("lifecycle", {}) or {}).get("finalized") is True
                and not bool(
                    dict(result.get("lifecycle", {}) or {}).get("failed", False)
                )
                for result in results
            )
        )
        batch_finalization = {
            "artifact_root": str(batch_root),
            "finalized": batch_valid,
            "failed": not batch_valid,
            "strict_success": bool(
                batch_valid
                and all(bool(row.get("strict_full_success", False)) for row in results)
            ),
            "batch_error": batch_error,
            "close_error": "PENDING_SIMULATION_CLOSE",
            "phase": "PRECLOSE_FINALIZED",
            "source_integrity": batch_source_comparison,
            "finalization_errors": finalization_errors,
        }
        for label, action in (
            ("batch_results", lambda: write_json(batch_root / "batch_results.json", results)),
            (
                "batch_finalization",
                lambda: write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                ),
            ),
            ("checksums", lambda: _write_checksums(batch_root)),
        ):
            try:
                action()
            except Exception as exc:
                finalization_errors.append(
                    f"{label}: {type(exc).__name__}: {exc}"
                )
                exit_code = 1
                batch_valid = False
        if finalization_errors:
            batch_finalization.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "finalization_errors": list(finalization_errors),
                }
            )
            try:
                write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                )
                _write_checksums(batch_root)
            except Exception as exc:
                print(
                    f"[FSM50] preclose finalization rewrite failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        # A batch marker is a reliability claim.  Do not create it until all
        # immutable preclose evidence has been durably copied.  If a copy
        # fails, rewrite both the live and snapshot finalization as failed.
        immutable_preclose_errors = _snapshot_preclose_files(batch_root)
        if immutable_preclose_errors:
            exit_code = 1
            batch_valid = False
            finalization_errors.extend(
                f"immutable_preclose: {error}"
                for error in immutable_preclose_errors
            )
            batch_finalization.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "finalization_errors": list(finalization_errors),
                }
            )
            try:
                write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                )
                _write_checksums(
                    batch_root,
                    exclude_preclose_snapshots=True,
                )
                refresh_errors = _snapshot_preclose_files(batch_root)
                if refresh_errors:
                    immutable_preclose_errors.extend(
                        f"refresh: {error}" for error in refresh_errors
                    )
            except Exception as exc:
                immutable_preclose_errors.append(
                    f"failure_rewrite: {type(exc).__name__}: {exc}"
                )
        try:
            evidence_manifest = _preclose_evidence_manifest(
                batch_root,
                results=results,
                batch_source_comparison=batch_source_comparison,
                batch_finalization=batch_finalization,
                include_global_analysis_reports=False,
            )
        except Exception as exc:
            evidence_manifest = {
                "physics_result_count": len(results),
                "manifest_error": f"{type(exc).__name__}: {exc}",
                "source_integrity": _jsonable(batch_source_comparison),
                "batch_finalization": _jsonable(batch_finalization),
            }
            exit_code = 1
            batch_valid = False
            finalization_errors.append(
                f"preclose_evidence_manifest: {evidence_manifest['manifest_error']}"
            )
            batch_finalization.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "finalization_errors": list(finalization_errors),
                }
            )
            try:
                write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                )
                _write_checksums(
                    batch_root,
                    exclude_preclose_snapshots=True,
                )
                immutable_preclose_errors.extend(
                    f"manifest_failure_refresh: {error}"
                    for error in _snapshot_preclose_files(batch_root)
                )
            except Exception as rewrite_exc:
                immutable_preclose_errors.append(
                    "manifest_failure_rewrite: "
                    f"{type(rewrite_exc).__name__}: {rewrite_exc}"
                )
            # The fallback marker must embed the same finalization payload as
            # the freshly rewritten immutable preclose snapshot.  Keeping the
            # pre-rewrite copy here makes an otherwise complete diagnostic
            # shutdown fail parent-side closure verification.
            evidence_manifest["batch_finalization"] = _jsonable(
                batch_finalization
            )
        if immutable_preclose_errors:
            evidence_manifest["immutable_preclose_errors"] = list(
                immutable_preclose_errors
            )
        try:
            _mark_artifact_root(batch_root, valid=batch_valid)
        except Exception as exc:
            marker_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[FSM50] batch marker failed: {marker_error}",
                file=sys.stderr,
                flush=True,
            )
            exit_code = 1
            batch_valid = False
            finalization_errors.append(f"batch_marker: {marker_error}")
            batch_finalization.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "finalization_errors": list(finalization_errors),
                }
            )
            evidence_manifest["batch_marker_error"] = marker_error
            evidence_manifest["batch_finalization"] = _jsonable(batch_finalization)
            try:
                write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                )
                _write_checksums(
                    batch_root,
                    exclude_preclose_snapshots=True,
                )
                _snapshot_preclose_files(batch_root)
                _mark_artifact_root(batch_root, valid=False)
            except Exception as recovery_exc:
                print(
                    f"[FSM50] failed-marker recovery failed: "
                    f"{type(recovery_exc).__name__}: {recovery_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if supervisor is not None:
            supervisor.mark_preclose(evidence_manifest)
        else:
            _atomic_write_json(
                batch_root / "preclose_complete.json",
                {
                    "schema_version": "fsm50.preclose_complete.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "token": "",
                    "parent_pid": os.getppid(),
                    "child_pid": os.getpid(),
                    "batch_root": str(batch_root),
                    "evidence": evidence_manifest,
                },
            )

        close_error = ""
        shutdown_mode = str(getattr(args, "shutdown_mode", "fast") or "fast")
        try:
            shutdown_mode = _close_simulation_with_explicit_policy(
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
            # Preclose bytes are immutable once PRECLOSE_COMPLETE is written.
            # The parent records the close failure and performs all lifecycle
            # mutation after it has verified that frozen closure.
        if supervisor is not None and close_error:
            supervisor.mark_close_error(
                shutdown_mode=shutdown_mode,
                error=close_error,
            )
    print(json.dumps({"batch_root": str(batch_root), "results": results}, ensure_ascii=False, indent=2))
    return _process_returncode_after_close(
        command_exit_code=exit_code,
        shutdown_mode=shutdown_mode,
        close_error=close_error,
        supervised=supervisor is not None,
    )


def _finalize_worker_recording_batch_preclose(
    batch_root: Path,
    *,
    artifact_request: dict[str, Any],
    results: list[dict[str, Any]],
    batch_error: str,
    source_integrity: dict[str, Any],
    supervisor: ChildSupervisorHandshake,
) -> tuple[dict[str, Any], bool]:
    """Freeze controller-owned batch bytes without touching worker run bytes."""

    root = Path(batch_root).resolve()
    finalization_errors: list[str] = []
    worker_result = dict(results[0] or {}) if len(results) == 1 else {}
    equivalence_role = str(
        artifact_request.get("environment_equivalence_role", "") or ""
    )
    diagnostic_role = str(artifact_request.get("diagnostic_role", "") or "")
    qualification_scope = (
        "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
        if diagnostic_role == "U"
        else "TRAJECTORY_COMPARISON"
        if equivalence_role
        else "GATE1_PHYSICAL_QUALIFICATION"
    )
    gate1_eligible = bool(not diagnostic_role and not equivalence_role)
    environment_equivalence_eligible = bool(
        equivalence_role and not diagnostic_role
    )
    expected_role_identity = {
        "environment_equivalence_role": equivalence_role,
        "diagnostic_role": diagnostic_role,
        "qualification_scope": qualification_scope,
        "gate1_physical_qualification_eligible": gate1_eligible,
        "gate1_eligible": gate1_eligible,
        "environment_equivalence_eligible": environment_equivalence_eligible,
    }
    if (
        equivalence_role not in {"", "A1", "A2", "B"}
        or diagnostic_role not in {"", "U"}
        or bool(equivalence_role and diagnostic_role)
    ):
        finalization_errors.append(
            "artifact_request_role_identity: invalid diagnostic role matrix"
        )
    for key, expected in expected_role_identity.items():
        if not _strict_json_equal(artifact_request.get(key), expected):
            finalization_errors.append(
                "artifact_request_role_identity: "
                f"{key} expected={expected!r} "
                f"actual={artifact_request.get(key)!r}"
            )
        if worker_result and not _strict_json_equal(
            worker_result.get(key), expected
        ):
            finalization_errors.append(
                "worker_result_role_identity: "
                f"{key} expected={expected!r} "
                f"actual={worker_result.get(key)!r}"
            )
    worker_result_complete = bool(
        len(results) == 1
        and _result_is_worker_owned(worker_result)
        and dict(results[0].get("lifecycle", {}) or {}).get("finalized") is True
        and dict(results[0].get("lifecycle", {}) or {}).get("failed") is not True
    )
    batch_valid = bool(
        not batch_error
        and source_integrity.get("equal") is True
        and worker_result_complete
        and not finalization_errors
    )
    environment_diagnostic_complete = bool(
        batch_valid
        and equivalence_role
        and worker_result.get(
            "environment_equivalence_diagnostic_complete"
        )
        is True
    )
    ordinary_diagnostic_complete = bool(
        batch_valid
        and diagnostic_role == "U"
        and worker_result.get("ordinary_ui_diagnostic_complete") is True
    )
    batch_finalization = {
        "artifact_root": str(root),
        "artifact_owner": "formal_worker_batch_controller",
        "finalized": batch_valid,
        "failed": not batch_valid,
        "strict_success": bool(
            batch_valid
            and len(results) == 1
            and results[0].get("strict_full_success") is True
        ),
        "environment_equivalence_role": equivalence_role,
        "environment_equivalence_diagnostic_complete": (
            environment_diagnostic_complete
        ),
        "diagnostic_role": diagnostic_role,
        "ordinary_ui_diagnostic_complete": ordinary_diagnostic_complete,
        "command_success": bool(
            ordinary_diagnostic_complete
            if diagnostic_role == "U"
            else environment_diagnostic_complete
            if equivalence_role
            else (
                batch_valid
                and len(results) == 1
                and results[0].get("strict_full_success") is True
            )
        ),
        "qualification_scope": qualification_scope,
        "gate1_physical_qualification_eligible": gate1_eligible,
        "gate1_eligible": gate1_eligible,
        "environment_equivalence_eligible": environment_equivalence_eligible,
        "batch_error": str(batch_error or ""),
        "close_error": "PENDING_FORMAL_WORKER_CLOSE",
        "phase": "PRECLOSE_FINALIZED",
        "source_integrity": _jsonable(source_integrity),
        "finalization_errors": finalization_errors,
        "worker_artifact_result_count": len(results),
    }

    def persist_live() -> None:
        _atomic_write_json(root / "batch_results.json", results)
        _atomic_write_json(root / "batch_finalization.json", batch_finalization)
        _write_checksums(root, exclude_preclose_snapshots=True)

    try:
        persist_live()
    except Exception as exc:
        finalization_errors.append(
            f"live_batch_bytes: {type(exc).__name__}: {exc}"
        )
        batch_valid = False
    if finalization_errors:
        batch_finalization.update(
            finalized=False,
            failed=True,
            strict_success=False,
            environment_equivalence_diagnostic_complete=False,
            ordinary_ui_diagnostic_complete=False,
            command_success=False,
            finalization_errors=list(finalization_errors),
        )
        try:
            persist_live()
        except Exception as exc:
            finalization_errors.append(
                f"failure_live_rewrite: {type(exc).__name__}: {exc}"
            )

    immutable_errors = _snapshot_preclose_files(root)
    if immutable_errors:
        batch_valid = False
        finalization_errors.extend(
            f"immutable_preclose: {error}" for error in immutable_errors
        )
        batch_finalization.update(
            finalized=False,
            failed=True,
            strict_success=False,
            environment_equivalence_diagnostic_complete=False,
            ordinary_ui_diagnostic_complete=False,
            command_success=False,
            finalization_errors=list(finalization_errors),
        )
        try:
            persist_live()
            immutable_errors.extend(
                f"refresh: {error}" for error in _snapshot_preclose_files(root)
            )
        except Exception as exc:
            immutable_errors.append(
                f"failure_rewrite: {type(exc).__name__}: {exc}"
            )

    try:
        evidence = _preclose_evidence_manifest(
            root,
            results=results,
            batch_source_comparison=source_integrity,
            batch_finalization=batch_finalization,
            include_global_analysis_reports=False,
        )
    except Exception as exc:
        batch_valid = False
        finalization_errors.append(
            f"preclose_evidence_manifest: {type(exc).__name__}: {exc}"
        )
        batch_finalization.update(
            finalized=False,
            failed=True,
            strict_success=False,
            environment_equivalence_diagnostic_complete=False,
            ordinary_ui_diagnostic_complete=False,
            command_success=False,
            finalization_errors=list(finalization_errors),
        )
        try:
            persist_live()
            immutable_errors.extend(
                f"manifest_failure_refresh: {error}"
                for error in _snapshot_preclose_files(root)
            )
        except Exception as rewrite_exc:
            immutable_errors.append(
                "manifest_failure_rewrite: "
                f"{type(rewrite_exc).__name__}: {rewrite_exc}"
            )
        evidence = {
            "physics_result_count": len(results),
            "manifest_error": f"{type(exc).__name__}: {exc}",
            "source_integrity": _jsonable(source_integrity),
            "batch_finalization": _jsonable(batch_finalization),
        }
    if immutable_errors:
        evidence["immutable_preclose_errors"] = list(immutable_errors)

    try:
        _mark_artifact_root(root, valid=batch_valid)
    except Exception as exc:
        batch_valid = False
        marker_error = f"{type(exc).__name__}: {exc}"
        evidence["batch_marker_error"] = marker_error
        finalization_errors.append(f"batch_marker: {marker_error}")
        batch_finalization.update(
            finalized=False,
            failed=True,
            strict_success=False,
            environment_equivalence_diagnostic_complete=False,
            ordinary_ui_diagnostic_complete=False,
            command_success=False,
            finalization_errors=list(finalization_errors),
        )
        evidence["batch_finalization"] = _jsonable(batch_finalization)
        try:
            persist_live()
            _snapshot_preclose_files(root)
            _mark_artifact_root(root, valid=False)
        except Exception:
            pass
    supervisor.mark_preclose(evidence)
    return batch_finalization, batch_valid


def _select_exact_operation_ack(
    status: dict[str, Any], *, operation: str, request_id: str
) -> dict[str, Any]:
    history = list(status.get("operation_ack_history", []) or [])
    latest = dict(status.get("last_operation_ack", {}) or {})
    if latest:
        history.append(latest)
    for row in reversed(history):
        ack = dict(row or {})
        if (
            str(ack.get("operation", "") or "") == str(operation)
            and str(ack.get("request_id", "") or "") == str(request_id)
        ):
            return ack
    return {}


def _failed_formal_worker_cleanup_identity(
    client: Any,
    process: Any,
    artifact_request: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind failed-worker cleanup to its exact terminal ACK without startup claims."""

    if not isinstance(artifact_request, dict):
        raise RuntimeError(
            "failed formal worker shutdown requires its immutable artifact request"
        )
    request_id = str(artifact_request.get("request_id", "") or "")
    if not request_id:
        raise RuntimeError("failed formal worker artifact request has no request_id")
    failed_ack: dict[str, Any] = {}
    # A failed session status and its critical terminal ACK travel as separate
    # ordered IPC messages.  Drain before requesting shutdown and accept only
    # the exact request-matched terminal ACK.
    for attempt in range(101):
        try:
            client.poll()
        except Exception:
            pass
        pre_shutdown_status = dict(client.status() or {})
        artifact_acks = list(
            pre_shutdown_status.get("artifact_ack_history", []) or []
        )
        latest_artifact_ack = dict(
            pre_shutdown_status.get("last_artifact_ack", {}) or {}
        )
        if latest_artifact_ack:
            artifact_acks.append(latest_artifact_ack)
        for row in reversed(artifact_acks):
            candidate = dict(row or {})
            if (
                candidate.get("type") == "operation_ack"
                and candidate.get("operation") == "recording_artifact"
                and candidate.get("phase") == "ARTIFACT_FAILED"
                and candidate.get("accepted") is False
                and candidate.get("artifact_complete") is False
                and candidate.get("request_id") == request_id
            ):
                failed_ack = candidate
                break
        if failed_ack or attempt >= 100:
            break
        time.sleep(0.01)
    if not failed_ack:
        raise RuntimeError(
            "failed formal worker has no exact terminal ARTIFACT_FAILED ACK"
        )
    launched_pid = _exact_json_int(
        getattr(client, "pid", None), "launched formal worker PID"
    )
    process_pid = _exact_json_int(
        getattr(process, "pid", None), "formal worker process PID"
    )
    if launched_pid <= 0:
        raise RuntimeError("launched formal worker PID must be positive")
    if launched_pid != process_pid:
        raise RuntimeError(
            "formal worker client/process PID mismatch: "
            f"client={launched_pid} process={process_pid}"
        )
    worker_session_id = failed_ack.get("worker_session_id")
    adapter_runtime_instance_id = failed_ack.get("adapter_runtime_instance_id")
    if not isinstance(worker_session_id, str) or not worker_session_id:
        raise RuntimeError(
            "failed formal worker ACK has no exact worker session identity"
        )
    if not isinstance(adapter_runtime_instance_id, str):
        raise RuntimeError(
            "failed formal worker ACK adapter identity must be a string"
        )
    expected_identity = {
        "worker_pid": launched_pid,
        "worker_session_id": worker_session_id,
        "adapter_runtime_instance_id": adapter_runtime_instance_id,
        "artifact_request_id": request_id,
        "root_state_write_count": 0,
        "contact_mode": artifact_request.get("contact_mode"),
        "environment_equivalence_role": artifact_request.get(
            "environment_equivalence_role"
        ),
        "diagnostic_role": artifact_request.get("diagnostic_role"),
        "qualification_scope": artifact_request.get("qualification_scope"),
        "gate1_eligible": artifact_request.get("gate1_eligible"),
        "gate1_physical_qualification_eligible": artifact_request.get(
            "gate1_physical_qualification_eligible"
        ),
        "environment_equivalence_eligible": artifact_request.get(
            "environment_equivalence_eligible"
        ),
    }
    for key, value in {
        "request_id": request_id,
        **expected_identity,
    }.items():
        if not _strict_json_equal(failed_ack.get(key), value):
            raise RuntimeError(
                f"failed formal worker terminal ACK {key} mismatch: "
                f"expected={value!r} actual={failed_ack.get(key)!r}"
            )
    if not str(failed_ack.get("error", "") or ""):
        raise RuntimeError("failed formal worker terminal ACK has no error")
    return expected_identity


def _cleanup_owned_formal_worker_without_claim(client: Any, process: Any) -> None:
    """Reap one launched worker without creating any lifecycle success claim."""

    if process.poll() is None:
        try:
            client.request_shutdown(mode="normal", request_id=uuid.uuid4().hex)
        except Exception:
            pass
        # The outer parent owns the 60-second tree timeout.  Keeping this child
        # alive while waiting ensures that timeout can reap the whole PID tree.
        while process.poll() is None:
            try:
                client.poll()
            except Exception:
                pass
            time.sleep(0.02)
    # Drain any bytes queued immediately before process exit, but never promote
    # them to an accepted shutdown/close handshake on this cleanup-only path.
    for _attempt in range(101):
        try:
            client.poll()
        except Exception:
            break
        time.sleep(0.01)
    client.close()


def _close_formal_recording_worker(
    client: Any,
    *,
    supervisor: ChildSupervisorHandshake,
    worker_binding: dict[str, Any],
    artifact_complete: bool,
    artifact_request: dict[str, Any] | None = None,
) -> str:
    """Request close; only the outer parent owns the 60-second kill policy."""

    process = getattr(client, "process", None)
    if process is None:
        raise RuntimeError("formal worker process was never launched")
    expected_identity: dict[str, Any]
    required_binding_keys = (
        "worker_pid",
        "worker_session_id",
        "adapter_runtime_instance_id",
        "artifact_request_id",
    )
    cleanup_only = not all(
        key in worker_binding for key in required_binding_keys
    )
    if not cleanup_only:
        if process.poll() is not None:
            raise RuntimeError(
                "formal worker exited before controller-requested shutdown: "
                f"returncode={process.poll()}"
            )
        expected_identity = {
            "worker_pid": _exact_json_int(
                worker_binding["worker_pid"], "formal worker binding PID"
            ),
            "worker_session_id": str(worker_binding["worker_session_id"]),
            "adapter_runtime_instance_id": str(
                worker_binding["adapter_runtime_instance_id"]
            ),
            "artifact_request_id": str(worker_binding["artifact_request_id"]),
            "root_state_write_count": 0,
        }
        for key in (
            "contact_mode",
            "environment_equivalence_role",
            "diagnostic_role",
            "qualification_scope",
            "gate1_eligible",
            "gate1_physical_qualification_eligible",
            "environment_equivalence_eligible",
        ):
            if key in worker_binding:
                expected_identity[key] = worker_binding[key]
    else:
        if artifact_complete:
            _cleanup_owned_formal_worker_without_claim(client, process)
            raise RuntimeError(
                "completed formal worker artifact has no durable startup binding"
            )
        try:
            expected_identity = _failed_formal_worker_cleanup_identity(
                client,
                process,
                artifact_request,
            )
        except Exception:
            _cleanup_owned_formal_worker_without_claim(client, process)
            raise
        if process.poll() is not None:
            # The worker's top-level preflight exception can close its own Kit
            # process before the controller requests shutdown.  The terminal
            # ACK above proves which owned worker failed; closing the client is
            # cleanup only and must not synthesize a graceful-close handshake.
            failed_returncode = process.poll()
            client.close()
            if _strict_json_equal(failed_returncode, 0):
                raise RuntimeError(
                    "failed formal worker exited zero without accepting normal shutdown"
                )
            return "normal"
    mode = "fast" if artifact_complete else "normal"
    shutdown_request_id = uuid.uuid4().hex
    runtime_version = ""
    if mode == "fast":
        # The exact runtime is learned from the worker's accepted close ACK;
        # do not synthesize a version string in the controller.
        runtime_version = "PENDING_FORMAL_WORKER_ACK"
    client.request_shutdown(mode=mode, request_id=shutdown_request_id)
    while process.poll() is None:
        client.poll()
        time.sleep(0.02)
    for _attempt in range(101):
        try:
            client.poll()
        except Exception:
            break
        time.sleep(0.005)
    returncode = process.poll()
    status = dict(client.status() or {})
    shutdown_ack = _select_exact_operation_ack(
        status,
        operation="shutdown",
        request_id=shutdown_request_id,
    )
    if cleanup_only and not shutdown_ack:
        # A top-level failed worker may self-exit while the controller's normal
        # shutdown request is in flight.  Exact ARTIFACT_FAILED identity was
        # already verified, so release local IPC resources while preserving the
        # original batch failure.  Do not claim that shutdown was accepted.
        client.close()
        if _strict_json_equal(returncode, 0):
            raise RuntimeError(
                "failed formal worker exited zero without accepting normal shutdown"
            )
        return mode
    expected_identity = {
        "request_id": shutdown_request_id,
        **expected_identity,
    }
    for key, value in expected_identity.items():
        if not _strict_json_equal(shutdown_ack.get(key), value):
            raise RuntimeError(
                f"formal worker shutdown ACK {key} mismatch: "
                f"expected={value!r} actual={shutdown_ack.get(key)!r}"
            )
    if (
        shutdown_ack.get("type") != "operation_ack"
        or shutdown_ack.get("operation") != "shutdown"
        or shutdown_ack.get("accepted") is not True
        or str(shutdown_ack.get("error", "") or "")
        or str(shutdown_ack.get("mode", "") or "") != mode
        or returncode != 0
    ):
        raise RuntimeError(
            "formal worker shutdown was not accepted/normal: "
            f"ack={shutdown_ack!r} returncode={returncode!r}"
        )
    if mode == "fast":
        close_requested = dict(status.get("close_requested_ack", {}) or {})
        close_returned = dict(status.get("close_returned_ack", {}) or {})
        runtime_version = str(shutdown_ack.get("runtime_version", "") or "")
        fast_close_kwargs = {
            "wait_for_replicator": False,
            "skip_cleanup": True,
        }
        if (
            not runtime_version.startswith("5.1.")
            or not _strict_json_equal(
                shutdown_ack.get("close_kwargs"), fast_close_kwargs
            )
        ):
            raise RuntimeError("formal worker fast-shutdown contract is invalid")
        for label, ack in (("close_requested", close_requested),):
            if str(ack.get("type", "") or "") != label:
                raise RuntimeError(f"formal worker {label} ACK is missing")
            for key, value in {
                **expected_identity,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": fast_close_kwargs,
                "runtime_version": runtime_version,
            }.items():
                if not _strict_json_equal(ack.get(key), value):
                    raise RuntimeError(
                        f"formal worker {label} {key} mismatch: "
                        f"expected={value!r} actual={ack.get(key)!r}"
                    )
        close_returned_observed = bool(close_returned)
        if close_returned_observed:
            if str(close_returned.get("type", "") or "") != "close_returned":
                raise RuntimeError("formal worker close_returned ACK type is invalid")
            for key, value in {
                **expected_identity,
                "mode": "fast",
                "accepted": True,
                "error": "",
                "close_kwargs": fast_close_kwargs,
                "runtime_version": runtime_version,
            }.items():
                if not _strict_json_equal(close_returned.get(key), value):
                    raise RuntimeError(
                        f"formal worker close_returned {key} mismatch: "
                        f"expected={value!r} actual={close_returned.get(key)!r}"
                    )
        supervisor.mark_fast_worker_process_returned(
            intended_returncode=0,
            runtime_version=runtime_version,
            worker_returncode=0,
            worker_process_returned_normally=True,
            worker_shutdown_accepted=True,
            worker_close_requested=True,
            worker_close_returned=close_returned_observed,
            worker_forced_termination=False,
            worker_shutdown_request_id=shutdown_request_id,
            worker_shutdown_ack=shutdown_ack,
            worker_close_requested_ack=close_requested,
            worker_close_returned_ack=close_returned,
        )
    elif not cleanup_only:
        supervisor.mark_graceful_close_returned(
            intended_returncode=0,
            worker_returncode=0,
            worker_process_returned_normally=True,
            worker_shutdown_accepted=True,
            worker_shutdown_request_id=shutdown_request_id,
        )
    client.close()
    return mode


def _run_recording_replays_locked(
    args: argparse.Namespace,
    *,
    process_snapshot: list[dict[str, Any]],
    supervisor: ChildSupervisorHandshake | None = None,
) -> int:
    """Run Gate-1 through the production IPC worker, never a local adapter."""

    if supervisor is None:
        raise RuntimeError(
            "formal recording replay requires the supervised child/worker topology"
        )
    from playback import playback_plan_to_payload
    from sim_obstacle_scene import DEFAULT_ROBOT_USD_PATH
    from sim_process_client import SimProcessClient
    from sim_transport import SimTransport

    audit = RecordingAudit(Path(args.recording_root), Path(args.report_root))
    selected = _select_versions(audit.enumerate_versions(), args.versions)
    if len(selected) != 1:
        raise RuntimeError(
            "formal worker recording replay requires exactly one selected version"
        )
    item = selected[0]
    trial_id = int(getattr(args, "trial_id", 0) or 0)
    if trial_id < 1:
        raise RuntimeError("recording replay requires a positive --trial-id")
    _normalize_recording_replay_role(args)
    contact_mode = str(getattr(args, "contact_mode", "") or "")
    equivalence_role = str(
        getattr(args, "environment_equivalence_role", "") or ""
    )
    diagnostic_role = str(getattr(args, "diagnostic_role", "") or "")
    if not equivalence_role and not diagnostic_role and contact_mode != "instrumented":
        raise RuntimeError(
            "Gate-1 formal worker replay requires --contact-mode instrumented"
        )
    if bool(getattr(args, "headless", False)) or bool(
        getattr(args, "no_video", False)
    ):
        raise RuntimeError("Gate-1 formal worker replay requires the actual GUI viewport")

    preflight_audits = _fail_closed_recording_audits(audit, selected)
    expected_steps_sha256 = str(preflight_audits[0]["accepted_steps_sha256"])
    robot_usd = Path(args.robot_usd or DEFAULT_ROBOT_USD_PATH).resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot USD not found: {robot_usd}")
    environment_lock_path = (
        Path(args.report_root).resolve() / "environment_lock_50mm.json"
    )
    environment_lock = _load_environment_lock(environment_lock_path)
    locked_source_check = _verify_locked_source_hashes(environment_lock)
    robot_hash = sha256_file(robot_usd).lower()
    metadata_robot_sha = str(
        dict(preflight_audits[0].get("metadata", {}) or {}).get(
            "robot_asset_sha256", ""
        )
        or ""
    ).lower()
    locked_robot_sha = str(
        dict(environment_lock.get("selected_environment", {}) or {}).get(
            "robot_usd_sha256", ""
        )
        or ""
    ).lower()
    prelaunch_environment_ok = bool(
        locked_source_check.get("ok") is True
        and metadata_robot_sha == robot_hash
        and locked_robot_sha == robot_hash
    )

    motion = load_motion_reference()
    steps = load_steps_jsonl(item.steps_path)
    plan, _plan_rows = fast_plan_rows(
        source_version=item.version_id,
        steps=steps,
        max_wheel_speed=float(motion.wheel_velocity_limit_rad_s),
    )
    request_id = uuid.uuid4().hex
    plan_scope = (
        "ordinary-ui-u"
        if diagnostic_role == "U"
        else f"environment-{equivalence_role.lower()}"
        if equivalence_role
        else "gate1"
    )
    plan_id = (
        f"fsm50-{plan_scope}-{item.version_id}-trial{trial_id}-"
        f"{uuid.uuid4().hex[:10]}"
    )
    batch_source_pre = _source_freeze(robot_usd=robot_usd)
    batch_root = _new_directory(
        Path(args.output_root).resolve(), "recording_replays"
    )
    supervisor.announce_batch(batch_root)
    (batch_root / ".partial").write_text("running\n", encoding="utf-8")
    artifact_request = _recording_artifact_request(
        item=item,
        batch_root=batch_root,
        args=args,
        robot_usd=robot_usd,
        environment_lock_path=environment_lock_path,
        expected_steps_sha256=expected_steps_sha256,
        plan=plan,
        request_id=request_id,
        plan_id=plan_id,
        trial_id=trial_id,
    )
    request_path = batch_root / "worker_artifact_request.json"
    _atomic_write_json(request_path, artifact_request)
    artifact_request_sha256 = sha256_file(request_path).lower()
    _atomic_write_json(
        batch_root / "batch_request.json",
        {
            "schema_version": "fsm50.formal_worker_recording_batch.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "versions": [item.version_id],
            "trial_id": trial_id,
            "environment_equivalence_role": equivalence_role,
            "diagnostic_role": diagnostic_role,
            "qualification_scope": (
                "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
                if diagnostic_role == "U"
                else "TRAJECTORY_COMPARISON"
                if equivalence_role
                else "GATE1_PHYSICAL_QUALIFICATION"
            ),
            "gate1_physical_qualification_eligible": bool(
                not diagnostic_role and not equivalence_role
            ),
            "gate1_eligible": bool(
                not diagnostic_role and not equivalence_role
            ),
            "environment_equivalence_eligible": bool(
                equivalence_role and not diagnostic_role
            ),
            "args": _serialize_replay_args(args),
            "source_freeze": batch_source_pre,
            "recording_preflight_audits": [
                _compact_recording_audit(row) for row in preflight_audits
            ],
            "environment_lock_path": str(environment_lock_path),
            "prelaunch_environment_validation": {
                "ok": prelaunch_environment_ok,
                "locked_source_hashes": locked_source_check,
                "metadata_robot_sha256": metadata_robot_sha,
                "locked_robot_usd_sha256": locked_robot_sha,
                "actual_robot_usd_sha256": robot_hash,
            },
            "artifact_request_path": str(request_path),
            "artifact_request_sha256": artifact_request_sha256,
            "plan_id": plan_id,
            "plan_sha256": str(plan.plan_sha256),
            "process_preflight": [
                {
                    "pid": row.get("pid"),
                    "name": row.get("name"),
                    "command_line_sha256": hashlib.sha256(
                        str(row.get("command_line", "")).encode("utf-8")
                    ).hexdigest(),
                }
                for row in process_snapshot
            ],
        },
    )

    client: Any | None = None
    worker_binding: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    batch_error = ""
    artifact_complete = False
    command_exit_code = 0
    try:
        if not prelaunch_environment_ok:
            raise RuntimeError(
                "environment/source lock prelaunch validation failed; "
                "see batch_request.json"
            )
        worker_args = _recording_worker_args(
            args,
            robot_usd=robot_usd,
            motion=motion,
            artifact_request_path=request_path,
        )
        client = SimProcessClient(worker_args)
        client.start()
        worker_binding = _wait_for_recording_worker_ready(
            client,
            request=artifact_request,
            request_sha256=artifact_request_sha256,
            timeout_s=float(worker_args.sim_startup_timeout_s),
        )
        _atomic_write_json(
            batch_root / "worker_startup_binding.json", worker_binding
        )
        supervisor.bind_worker(
            worker_pid=int(worker_binding["worker_pid"]),
            worker_session_id=str(worker_binding["worker_session_id"]),
            adapter_runtime_instance_id=str(
                worker_binding["adapter_runtime_instance_id"]
            ),
            artifact_request_id=request_id,
            artifact_request_sha256=artifact_request_sha256,
        )
        transport = SimTransport()
        transport.attach_process_client(client)
        preplay_stop = transport.stop_wheels(reason="playback_start_boundary")
        preplay_stop_ack = _wait_for_worker_stop_ack(
            client,
            command_id=str(preplay_stop["command_id"]),
            worker_pid=int(worker_binding["worker_pid"]),
            worker_session_id=str(worker_binding["worker_session_id"]),
            timeout_s=30.0,
        )
        _atomic_write_json(
            batch_root / "worker_preplay_stop_ack.json", preplay_stop_ack
        )
        transport.start_playback_plan(
            plan,
            start_delay_sim_s=0.0,
            plan_id=plan_id,
            request_id=request_id,
            plan_sha256=str(plan.plan_sha256),
            worker_session_id=str(worker_binding["worker_session_id"]),
        )
        start_ack = _wait_for_worker_operation_ack(
            client,
            operation="start_playback_plan",
            request_id=request_id,
            timeout_s=30.0,
        )
        _atomic_write_json(
            batch_root / "worker_playback_start_ack.json", start_ack
        )
        expected_start = {
            "accepted": True,
            "request_id": request_id,
            "plan_id": plan_id,
            "plan_sha256": str(plan.plan_sha256),
            "worker_session_id": str(worker_binding["worker_session_id"]),
            "motion_start_ready": True,
            "contact_mode": contact_mode,
            "environment_equivalence_role": equivalence_role,
            "diagnostic_role": diagnostic_role,
            "qualification_scope": str(
                artifact_request["qualification_scope"]
            ),
            "gate1_eligible": artifact_request["gate1_eligible"],
            "gate1_physical_qualification_eligible": artifact_request[
                "gate1_physical_qualification_eligible"
            ],
            "environment_equivalence_eligible": artifact_request[
                "environment_equivalence_eligible"
            ],
        }
        for key, value in expected_start.items():
            if start_ack.get(key) != value:
                raise RuntimeError(
                    f"formal worker playback-start ACK {key} mismatch: "
                    f"expected={value!r} actual={start_ack.get(key)!r}"
                )
        if str(start_ack.get("error", "") or ""):
            raise RuntimeError(
                "formal worker rejected playback: " + str(start_ack["error"])
            )

        artifact_ack = client.wait_for_artifact(
            timeout_s=None,
            request_id=request_id,
        )
        _atomic_write_json(
            batch_root / "worker_artifact_complete_ack.json", artifact_ack
        )
        artifact_complete = bool(
            artifact_ack.get("operation") == "recording_artifact"
            and artifact_ack.get("phase") == "ARTIFACT_COMPLETE"
            and artifact_ack.get("accepted") is True
            and artifact_ack.get("artifact_complete") is True
        )
        result = _validate_worker_artifact_complete_ack(
            artifact_ack,
            request=artifact_request,
            request_sha256=artifact_request_sha256,
            batch_root=batch_root,
            worker_pid=int(worker_binding["worker_pid"]),
            worker_session_id=str(worker_binding["worker_session_id"]),
            adapter_runtime_instance_id=str(
                worker_binding["adapter_runtime_instance_id"]
            ),
        )
        results = [result]
        _atomic_write_json(batch_root / "batch_results.json", results)
        command_exit_code = (
            0
            if (
                result.get("ordinary_ui_diagnostic_complete") is True
                if diagnostic_role == "U"
                else result.get("environment_equivalence_diagnostic_complete")
                is True
                if equivalence_role
                else result.get("strict_full_success") is True
            )
            else 1
        )
    except Exception as exc:
        command_exit_code = 1
        batch_error = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(
            batch_root / "batch_failure.json",
            {
                "error": batch_error,
                "results_so_far": results,
                "worker_binding": worker_binding,
            },
        )
        print(f"[FSM50] formal worker batch failed: {batch_error}", file=sys.stderr, flush=True)

    close_error = ""
    shutdown_mode = "fast" if artifact_complete else "graceful"
    try:
        try:
            batch_source_post = _source_freeze(robot_usd=robot_usd)
            source_integrity = _compare_source_freezes(
                batch_source_pre, batch_source_post
            )
            _atomic_write_json(
                batch_root / "source_freeze_post.json", batch_source_post
            )
        except Exception as exc:
            source_integrity = {
                "equal": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            command_exit_code = 1
        _atomic_write_json(batch_root / "source_integrity.json", source_integrity)
        if source_integrity.get("equal") is not True:
            command_exit_code = 1
            _apply_batch_source_drift(results, source_integrity)
        _batch_finalization, _batch_valid = (
            _finalize_worker_recording_batch_preclose(
                batch_root,
                artifact_request=artifact_request,
                results=results,
                batch_error=batch_error,
                source_integrity=source_integrity,
                supervisor=supervisor,
            )
        )
    except Exception as exc:
        command_exit_code = 1
        postprocessing_error = f"{type(exc).__name__}: {exc}"
        batch_error = (
            f"{batch_error}; postprocessing: {postprocessing_error}"
            if batch_error
            else f"postprocessing: {postprocessing_error}"
        )
        try:
            _atomic_write_json(
                batch_root / "batch_failure.json",
                {
                    "error": batch_error,
                    "results_so_far": results,
                    "worker_binding": worker_binding,
                    "postprocessing_error": postprocessing_error,
                },
            )
        except Exception as persist_exc:
            print(
                "[FSM50] failed to persist postprocessing failure: "
                f"{type(persist_exc).__name__}: {persist_exc}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        try:
            if client is None:
                raise RuntimeError("formal worker client was not created")
            shutdown_mode = _close_formal_recording_worker(
                client,
                supervisor=supervisor,
                worker_binding=worker_binding,
                artifact_complete=artifact_complete,
                artifact_request=artifact_request,
            )
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
            command_exit_code = 1
            supervisor.mark_close_error(
                shutdown_mode=shutdown_mode,
                error=close_error,
            )
    print(
        json.dumps(
            {
                "batch_root": str(batch_root),
                "execution_path": "sim_worker_process_ipc",
                "results": results,
                "close_error": close_error,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return _process_returncode_after_close(
        command_exit_code=command_exit_code,
        shutdown_mode=shutdown_mode,
        close_error=close_error,
        supervised=True,
    )


def run_recording_replays(args: argparse.Namespace) -> int:
    _normalize_recording_replay_role(args)
    singleton = ReplaySingletonLock()
    singleton.acquire()
    try:
        if bool(getattr(args, "resume", False)):
            audit = RecordingAudit(
                Path(args.recording_root),
                Path(args.report_root),
            )
            selected = _select_versions(audit.enumerate_versions(), args.versions)
            if not selected:
                raise RuntimeError("no recording versions selected")
            preflight_audits = _fail_closed_recording_audits(audit, selected)
            expected_hashes = {
                str(row["version"]): str(row["accepted_steps_sha256"])
                for row in preflight_audits
            }
            completed = _find_reliable_completed_replays(
                Path(args.output_root),
                selected,
                expected_hashes,
            )
            pending = [
                item for item in selected if item.version_id not in completed
            ]
            resume_summary = {
                "resume": True,
                "requested_versions": [item.version_id for item in selected],
                "skipped_reliable_versions": list(completed),
                "pending_versions": [item.version_id for item in pending],
                "reliable_artifacts": list(completed.values()),
            }
            print(json.dumps(resume_summary, ensure_ascii=False, indent=2), flush=True)
            if not pending:
                return 0
            # The child receives full version ids, so it cannot accidentally
            # widen a short selector after the parent completed the audit.
            args = copy.copy(args)
            args.versions = [item.version_id for item in pending]
            args.resume_skipped = list(completed.values())
        process_snapshot = _os_process_snapshot()
        conflicts = _existing_simulator_processes(process_snapshot)
        if conflicts:
            summary = [
                {
                    "pid": row.get("pid"),
                    "name": row.get("name"),
                    "command_line": row.get("command_line"),
                }
                for row in conflicts
            ]
            raise RuntimeError(
                "existing Kit/Isaac/sim worker process detected; refusing a second "
                "SimulationApp: " + json.dumps(summary, ensure_ascii=False)
            )
        output_root = Path(args.output_root).resolve()
        control_parent = output_root / ".supervisor"
        control_parent.mkdir(parents=True, exist_ok=True)
        control_root = _new_directory(control_parent, "parent_child")
        request_path = control_root / "child_request.json"
        handshake_path = control_root / "child_handshake.json"
        token = uuid.uuid4().hex
        parent_pid = os.getpid()
        _atomic_write_json(
            request_path,
            {
                "schema_version": "fsm50.supervised_child_request.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "token": token,
                "parent_pid": parent_pid,
                "handshake_path": str(handshake_path),
                "args": _serialize_replay_args(args),
                "process_snapshot": _jsonable(process_snapshot),
            },
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "fsm_50mm_recording_derived_v3.run_fsm50",
            SUPERVISED_CHILD_SENTINEL,
            "--request",
            str(request_path),
            "--handshake",
            str(handshake_path),
            "--token",
            token,
            "--parent-pid",
            str(parent_pid),
        ]
        popen_kwargs: dict[str, Any] = {
            "cwd": str(PROJECT_ROOT),
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = int(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True
        child = subprocess.Popen(command, **popen_kwargs)
        _atomic_write_json(
            control_root / "supervisor_parent.json",
            {
                "schema_version": "fsm50.supervisor_parent.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "parent_pid": parent_pid,
                "child_pid": int(child.pid),
                "token": token,
                "command": command,
                "singleton_lock_path": str(singleton.path),
                "global_run_timeout": None,
                "post_preclose_shutdown_grace_s": SIMULATION_CLOSE_GRACE_S,
            },
        )
        try:
            outcome, batch_root = _monitor_supervised_child(
                child,
                handshake_path=handshake_path,
                token=token,
                parent_pid=parent_pid,
                output_root=output_root,
            )
        except Exception as exc:
            termination: dict[str, Any] = {}
            if child.poll() is None:
                termination = _terminate_owned_child_tree(child)
            outcome = {
                "schema_version": "fsm50.shutdown_outcome.v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "SUPERVISOR_HANDSHAKE_INVALID",
                "parent_pid": parent_pid,
                "child_pid": int(child.pid),
                "child_returncode": child.poll(),
                "error": f"{type(exc).__name__}: {exc}",
                "termination": termination,
            }
            batch_root = None
            try:
                _handshake, observed, _preclose = _read_supervisor_handshake(
                    handshake_path,
                    token=token,
                    parent_pid=parent_pid,
                    child_pid=int(child.pid),
                    output_root=output_root,
                )
                batch_root = observed
            except Exception:
                pass
        _atomic_write_json(control_root / "shutdown_outcome.json", outcome)
        if batch_root is not None:
            _record_shutdown_outcome(batch_root, outcome)
        print(
            json.dumps(
                {
                    "supervisor_control_root": str(control_root),
                    "batch_root": "" if batch_root is None else str(batch_root),
                    "shutdown_outcome": outcome,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if str(outcome.get("status")) not in SUCCESSFUL_SHUTDOWN_STATUSES:
            return 1
        if batch_root is None:
            return int(outcome.get("child_returncode", 1) or 0)
        return _batch_command_exit_code(
            batch_root,
            fallback=int(outcome.get("child_returncode", 1) or 0),
        )
    finally:
        singleton.release()


def _supervised_child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--handshake", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    internal = parser.parse_args(argv)
    request = json.loads(internal.request.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "fsm50.supervised_child_request.v1",
        "token": str(internal.token),
        "parent_pid": int(internal.parent_pid),
        "handshake_path": str(internal.handshake.resolve()),
    }
    actual = {
        "schema_version": request.get("schema_version"),
        "token": request.get("token"),
        "parent_pid": int(request.get("parent_pid", 0) or 0),
        "handshake_path": str(Path(request.get("handshake_path", "")).resolve()),
    }
    if actual != expected:
        raise RuntimeError(
            f"supervised child request validation failed: expected={expected} actual={actual}"
        )
    if os.getppid() != int(internal.parent_pid):
        raise RuntimeError(
            f"supervised child parent PID changed before startup: "
            f"expected={internal.parent_pid} actual={os.getppid()}"
        )
    supervisor = ChildSupervisorHandshake(
        internal.handshake,
        token=internal.token,
        parent_pid=internal.parent_pid,
    )
    try:
        args = _deserialize_replay_args(dict(request.get("args", {}) or {}))
        process_snapshot = list(request.get("process_snapshot", []) or [])
        if str(getattr(args, "command", "")) == "grounding-only":
            return _run_grounding_only_locked(
                args,
                process_snapshot=process_snapshot,
                supervisor=supervisor,
            )
        if str(getattr(args, "command", "")) in {
            "run-fsm",
            "test-state",
            "validate-5",
        }:
            from .fsm50_isaac_runtime import run_fsm_locked

            return run_fsm_locked(
                args,
                process_snapshot=process_snapshot,
                supervisor=supervisor,
            )
        return _run_recording_replays_locked(
            args,
            process_snapshot=process_snapshot,
            supervisor=supervisor,
        )
    except Exception as exc:
        try:
            supervisor.mark_failed(f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        print(
            f"[FSM50] supervised child failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Run the non-mutating recording/Fast-plan audit.")
    audit_parser.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    audit_parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)

    grounding = subparsers.add_parser(
        "grounding-only",
        help=(
            "Run one fresh-process formal clean-reset grounding diagnostic; "
            "invoke it three times for the 3/3 gate"
        ),
    )
    grounding.add_argument("--trial-id", type=int, required=True)
    grounding.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    grounding.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT / "grounding_only",
    )
    grounding.add_argument("--robot-usd", type=str, default="")
    grounding.add_argument("--device", type=str, default="cuda:0")
    grounding.add_argument("--headless", action="store_true")
    grounding.add_argument("--livestream", type=int, default=0)
    grounding.add_argument("--experience", type=str, default="")
    grounding.add_argument("--video-fps", type=float, default=15.0)
    grounding.add_argument("--no-video", action="store_true")
    grounding.add_argument(
        "--shutdown-mode",
        choices=("fast", "graceful"),
        default="fast",
    )

    replay = subparsers.add_parser(
        "replay-recordings",
        help="Replay one recording version in one fresh supervised Isaac process.",
    )
    replay.add_argument("--versions", nargs="+", default=["all"])
    replay.add_argument(
        "--trial-id",
        type=int,
        required=True,
        help="Positive clean-process trial identifier; one version per invocation.",
    )
    replay.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    replay.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    replay.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT / "v003_fast_replay_baseline",
    )
    replay.add_argument("--robot-usd", type=str, default="")
    replay.add_argument("--device", type=str, default="cuda:0")
    replay.add_argument(
        "--headless",
        action="store_true",
        help="Diagnostic only; no active GUI viewport means artifacts cannot be reliable.",
    )
    replay.add_argument("--livestream", type=int, default=0)
    replay.add_argument("--experience", type=str, default="")
    replay.add_argument("--telemetry-rate", type=float, default=120.0)
    replay.add_argument(
        "--contact-mode",
        choices=("formal", "instrumented"),
        default="",
        help=(
            "Explicit sensor mode. Omit for Gate-1 instrumented capture or to "
            "derive the exact mode from --environment-equivalence-role."
        ),
    )
    replay.add_argument(
        "--environment-equivalence-role",
        choices=("A1", "A2", "B"),
        default="",
        help=(
            "Explicit trajectory-comparison capture role: A1/A2 use the "
            "production aggregate ContactSensor; B uses the instrumented "
            "filtered wheel/non-wheel bank. Physical Gate-1 qualification "
            "remains separate."
        ),
    )
    replay.add_argument(
        "--diagnostic-role",
        choices=("U",),
        default="",
        help=(
            "Explicit sensor-free ordinary-production-UI trajectory diagnostic. "
            "U derives contact_mode=disabled and is never Gate-1 or "
            "environment-equivalence qualification evidence."
        ),
    )
    replay.add_argument(
        "--playback-pre-step-settle-s",
        type=float,
        default=0.30,
        help=(
            "Recorded for UI-path provenance only; Gate-1 dispatch begins on "
            "the next physics tick after the verified start boundary."
        ),
    )
    replay.add_argument("--post-run-settle-s", type=float, default=0.50)
    replay.add_argument("--video-fps", type=float, default=15.0)
    replay.add_argument(
        "--shutdown-mode",
        choices=("fast", "graceful"),
        default="fast",
        help=(
            "fast uses the explicit Isaac 5.1 skip_cleanup path after durable "
            "preclose; graceful is retained for close_stage diagnostics"
        ),
    )
    replay.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Diagnostic only; each version is ARTIFACT_INVALID and cannot be "
            "strictly or reliably completed without actual viewport video."
        ),
    )
    replay.add_argument("--timeout-s", type=float, default=30.0)
    replay.add_argument("--timeout-scale", type=float, default=3.0)
    replay.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip only source-matched, checksummed, finalized recording-version "
            "artifacts; partial/crashed/invalid runs are retried."
        ),
    )
    replay.add_argument("--continue-on-error", action="store_true", default=True)
    replay.add_argument("--fail-fast", dest="continue_on_error", action="store_false")

    def add_fsm_runtime_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--config",
            type=Path,
            default=MODULE_ROOT / "fsm50_config.yaml",
        )
        command_parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
        command_parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT / "fsm_runs")
        command_parser.add_argument(
            "--environment-report",
            type=Path,
            default=DEFAULT_REPORT_ROOT / "ENVIRONMENT_EQUIVALENCE_REPORT.json",
        )
        command_parser.add_argument("--robot-usd", type=str, default="")
        command_parser.add_argument("--device", type=str, default="cuda:0")
        command_parser.add_argument("--headless", action="store_true")
        command_parser.add_argument("--livestream", type=int, default=0)
        command_parser.add_argument("--experience", type=str, default="")
        command_parser.add_argument("--telemetry-rate", type=float, default=120.0)
        command_parser.add_argument(
            "--timeout-s",
            type=float,
            default=600.0,
            help="Per-run simulation-time deadline; state transitions remain physical-event gated.",
        )
        command_parser.add_argument("--post-run-settle-s", type=float, default=0.30)
        command_parser.add_argument("--video-fps", type=float, default=15.0)
        command_parser.add_argument(
            "--shutdown-mode",
            choices=("fast", "graceful"),
            default="fast",
        )
        command_parser.add_argument(
            "--no-video",
            action="store_true",
            help="Diagnostic only; strict success is impossible without actual viewport video.",
        )

    run_fsm = subparsers.add_parser(
        "run-fsm", help="Run the real event-gated A0-to-F5 controller."
    )
    add_fsm_runtime_arguments(run_fsm)

    test_state = subparsers.add_parser(
        "test-state", help="Run one state from a trusted restore or verified live prefix."
    )
    add_fsm_runtime_arguments(test_state)
    test_state.add_argument("--state-id", required=True)
    restore_group = test_state.add_mutually_exclusive_group()
    restore_group.add_argument("--sim-state-before", type=Path)
    restore_group.add_argument("--prefix-manifest", type=Path)
    restore_group.add_argument(
        "--replay-prefix",
        action="store_true",
        help="Execute a fresh live A0 prefix to the target state in this run.",
    )
    test_state.add_argument("--sim-state-before-sha256", default="")
    test_state.add_argument("--prefix-manifest-sha256", default="")

    validate_five = subparsers.add_parser(
        "validate-5", help="Run five consecutive clean-reset full FSM validations."
    )
    add_fsm_runtime_arguments(validate_five)
    validate_five.set_defaults(output_root=DEFAULT_RUN_ROOT / "validate_5")

    validate_environment = subparsers.add_parser(
        "validate-environment",
        help="Build/compare the environment fingerprint and real A/A-B run artifacts.",
    )
    validate_environment.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "ENVIRONMENT_EQUIVALENCE_REPORT.json",
    )
    validate_environment.add_argument("--baseline-a1", type=Path)
    validate_environment.add_argument("--baseline-a2", type=Path)
    validate_environment.add_argument("--instrumented-b", type=Path)
    validate_environment.add_argument("--robot-usd", type=Path)

    report = subparsers.add_parser("report", help="Regenerate telemetry visualizations for existing run directories.")
    report.add_argument("run_dirs", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == SUPERVISED_CHILD_SENTINEL:
        return _supervised_child_main(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
    _normalize_recording_replay_role(args)
    if args.command == "audit":
        result = RecordingAudit(args.recording_root, args.report_root).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "grounding-only":
        return run_recording_replays(args)
    if args.command == "replay-recordings":
        return run_recording_replays(args)
    if args.command in {"run-fsm", "validate-5"}:
        return run_recording_replays(args)
    if args.command == "test-state":
        state_id = str(args.state_id)
        if state_id != "A0_RESET_AND_SETTLE" and not any(
            (
                args.sim_state_before,
                args.prefix_manifest,
                args.replay_prefix,
            )
        ):
            raise RuntimeError(
                "non-A0 test-state requires --sim-state-before, "
                "--prefix-manifest, or --replay-prefix"
            )
        if args.sim_state_before or args.prefix_manifest:
            from .fsm50_state_restore import validate_state_restore
            from .fsm50_controller import validate_happy_path_reachability

            environment_path = Path(args.environment_report).resolve()
            if not environment_path.is_file():
                raise RuntimeError(
                    f"environment report is missing: {environment_path}"
                )
            args.restore_bundle = validate_state_restore(
                target_state_id=state_id,
                environment_fingerprint_sha256=sha256_file(environment_path),
                sim_state_before_path=args.sim_state_before,
                sim_state_before_sha256=args.sim_state_before_sha256,
                prefix_replay_manifest_path=args.prefix_manifest,
                prefix_replay_manifest_sha256=args.prefix_manifest_sha256,
                state_order=list(
                    validate_happy_path_reachability(Path(args.config))
                ),
            )
            if args.prefix_manifest:
                args.replay_prefix = True
        else:
            args.restore_bundle = None
        return run_recording_replays(args)
    if args.command == "validate-environment":
        from .environment_equivalence import (
            build_static_environment_fingerprint,
            write_environment_equivalence_report,
        )

        fingerprint = build_static_environment_fingerprint(
            robot_usd_path=args.robot_usd
        )
        if args.baseline_a1 or args.baseline_a2 or args.instrumented_b:
            if not all((args.baseline_a1, args.baseline_a2, args.instrumented_b)):
                raise RuntimeError(
                    "A/A-B comparison requires --baseline-a1, --baseline-a2, "
                    "and --instrumented-b together"
                )
            from .environment_ab_artifacts import (
                generate_environment_equivalence_report,
            )

            payload = generate_environment_equivalence_report(
                a1_run=args.baseline_a1,
                a2_run=args.baseline_a2,
                b_run=args.instrumented_b,
                output_path=args.output,
                fingerprint=fingerprint,
            )
        else:
            # A static fingerprint is useful evidence, but it cannot satisfy
            # the runtime A/A--A/B gate on its own.
            write_environment_equivalence_report(
                args.output,
                fingerprint=fingerprint,
                instrumentation_comparison=None,
                trajectory_comparison=None,
                runtime_readback=None,
            )
            payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("status") == "PASS" else 1
    if args.command == "report":
        results = []
        for run_dir in args.run_dirs:
            results.append(_generate_existing_fsm50_visualization(run_dir))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
