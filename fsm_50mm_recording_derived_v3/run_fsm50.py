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
from sim_state_validation import verify_restored_full_sim_pose
from telemetry.config import RuntimeTelemetryConfig
from telemetry.exporters import write_csv, write_json

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
from .recording_fast_plan import fast_plan_rows, write_fast_plan
from .support_classifier import ObstacleGeometry


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


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace one small supervisor/control JSON document."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=str,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
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
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


_PRECLOSE_SNAPSHOT_FILES: tuple[tuple[str, str], ...] = (
    ("batch_results.json", "batch_results.preclose.json"),
    ("batch_finalization.json", "batch_finalization.preclose.json"),
    ("checksums.sha256", "checksums.preclose.sha256"),
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


class ChildSupervisorHandshake:
    """Child-owned atomic handshake consumed only by its exact parent PID."""

    def __init__(self, path: Path, *, token: str, parent_pid: int) -> None:
        self.path = Path(path).resolve()
        self.token = str(token)
        self.parent_pid = int(parent_pid)
        self.child_pid = os.getpid()
        self.batch_root: Path | None = None
        self.sequence = 0

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

    def mark_close_returned(self, *, close_error: str = "") -> None:
        self._write(
            "CLOSE_RETURNED" if not close_error else "CLOSE_ERROR",
            close_error=str(close_error),
        )

    def mark_failed(self, error: str) -> None:
        self._write("CHILD_FAILED", error=str(error))


def _preclose_evidence_manifest(
    batch_root: Path,
    *,
    results: list[dict[str, Any]],
    batch_source_comparison: dict[str, Any],
    batch_finalization: dict[str, Any],
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
        VERSION_MATRIX_PATH,
        DIAGONAL_TIMELINE_PATH,
    ]
    for result in results:
        run_dir_text = str(result.get("run_dir", "") or "")
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        paths.extend(
            [
                run_dir / "result.json",
                run_dir / "physical_evidence.json",
                run_dir / "fsm50_telemetry.csv",
                run_dir / "state_timeline.csv",
                run_dir / "visual_recording_manifest.json",
                run_dir / "checksums.sha256",
            ]
        )
    evidence_files = {
        str(path.resolve()): {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(paths))
        if path.is_file()
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
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
            if not preclose_valid:
                status = "CHILD_EXIT_BEFORE_PRECLOSE"
            elif last_state == "CLOSE_ERROR":
                status = "SIMULATION_CLOSE_ERROR"
            elif last_state != "CLOSE_RETURNED":
                status = "CHILD_EXIT_BEFORE_CLOSE_RETURNED"
            elif int(returncode) != SUPERVISED_CHILD_SUCCESS_RETURNCODE:
                status = "CHILD_EXIT_UNEXPECTED_RETURNCODE"
            else:
                status = "NORMAL_EXIT"
            return (
                {
                    "schema_version": "fsm50.shutdown_outcome.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "status": status,
                    "parent_pid": int(parent_pid),
                    "child_pid": int(child.pid),
                    "child_returncode": int(returncode),
                    "preclose_observed": bool(preclose_valid),
                    "handshake_state": last_state,
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


def _record_shutdown_outcome(
    batch_root: Path,
    outcome: dict[str, Any],
) -> None:
    """Persist supervisor outcome; lifecycle changes never rewrite physics class."""

    batch_root = Path(batch_root).resolve()
    status = str(outcome.get("status", "") or "")
    shutdown_failed = status != "NORMAL_EXIT"
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
                "failure_reason": status,
                "close_error": status,
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


def _serialize_replay_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in vars(args).items()}


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
        else (run_dir / "fsm50_viewport.mp4").resolve()
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
    if not str(raw.get("render_product_path", "") or ""):
        errors.append("active viewport render product path is missing")
    if frame_count < 2:
        errors.append("fewer than two viewport frames were captured")
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
    """Per-version scope around the shared Isaac viewport recorder.

    Isaac's movie-capture helper leaves its temporary render product attached.
    Restoring and deleting that graph after each version is required so the next
    version receives a new frame directory instead of silently reusing the
    previous version's global ``basePath``.
    """

    def __init__(
        self,
        run_dir: Path,
        args: argparse.Namespace,
        *,
        contact_mode: str,
        recorder: Any | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.args = args
        self.contact_mode = str(contact_mode)
        self.capture_requested = _recording_video_capture_requested(args)
        if recorder is None:
            from .fsm50_isaac_runtime import ViewportVideoRecorder

            recorder = ViewportVideoRecorder(
                self.run_dir,
                enabled=self.capture_requested,
                fps=float(getattr(args, "video_fps", 15.0)),
            )
        self.recorder = recorder
        self.viewport: Any | None = None
        self.original_render_product_path = ""
        self.capture_render_product_path = ""
        self.capture_error = ""
        self._finalized: dict[str, Any] | None = None

    def start(self) -> None:
        if self.capture_requested:
            try:
                from omni.kit.viewport.utility import get_active_viewport  # type: ignore

                self.viewport = get_active_viewport()
                if self.viewport is None:
                    raise RuntimeError("active GUI viewport is unavailable")
                self.original_render_product_path = str(
                    self.viewport.render_product_path
                )
            except Exception as exc:
                self.capture_error = f"viewport preflight failed: {type(exc).__name__}: {exc}"
        try:
            self.recorder.start()
        except Exception as exc:
            self.capture_error = self.capture_error or (
                f"viewport recorder start failed: {type(exc).__name__}: {exc}"
            )
        if self.capture_requested and self.viewport is not None:
            self.capture_render_product_path = str(
                self.viewport.render_product_path
            )
            if (
                not self.capture_render_product_path
                or self.capture_render_product_path
                == self.original_render_product_path
            ):
                self.capture_error = self.capture_error or (
                    "movie capture did not attach a distinct viewport render product"
                )

    def _release_capture_graph(self) -> str:
        if (
            not self.capture_requested
            or self.viewport is None
            or not self.original_render_product_path
            or not self.capture_render_product_path
            or self.capture_render_product_path == self.original_render_product_path
        ):
            return ""
        try:
            import isaacsim.kit.scripts.movie_capture as movie_capture  # type: ignore

            postfix = str(movie_capture.rpPrimPathPostFix)
            expected_capture = self.original_render_product_path + postfix
            if self.capture_render_product_path != expected_capture:
                raise RuntimeError(
                    "unexpected movie-capture render product: "
                    f"{self.capture_render_product_path}"
                )
            self.viewport.render_product_path = self.original_render_product_path
            stage = movie_capture.omni.usd.get_context().get_stage()
            movie_capture.remove_existing_graph(
                stage,
                self.capture_render_product_path,
                self.capture_render_product_path + str(movie_capture.ogNodePath),
            )
            return ""
        except Exception as exc:
            return f"viewport capture cleanup failed: {type(exc).__name__}: {exc}"

    def finalize(self) -> dict[str, Any]:
        if self._finalized is not None:
            return dict(self._finalized)
        try:
            raw = dict(self.recorder.finalize() or {})
        except Exception as exc:
            raw = {
                "valid": False,
                "source": _ACTUAL_VIEWPORT_VIDEO_SOURCE,
                "video_path": str(self.run_dir / "fsm50_viewport.mp4"),
                "video_sha256": "",
                "frame_count": 0,
                "error": f"viewport recorder finalize failed: {type(exc).__name__}: {exc}",
            }
        cleanup_error = self._release_capture_graph()
        combined_error = "; ".join(
            reason
            for reason in (self.capture_error, cleanup_error)
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
            "video_path": str(Path(run_dir).resolve() / "fsm50_viewport.mp4"),
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
    artifact_valid = bool(source_ok and visualization_ok and video_ok)
    if not artifact_valid and source_ok:
        result["classification_before_artifact_validation"] = result.get(
            "classification"
        )
        result["classification"] = "ARTIFACT_INVALID"
        result["first_failure_phase"] = (
            "VIEWPORT_VIDEO_MISSING_OR_INVALID"
            if not video_ok
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
        (result_path, diagnostics_path, *video_files),
    ):
        return None
    try:
        # Keep resume admission aligned with the environment-artifact gate.  A
        # version marker alone is not durable proof of completion: the child
        # writes it before SimulationApp.close() and the supervising parent
        # records NORMAL_EXIT only after close returns.  Import lazily so the
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
        str(batch_shutdown_closure.get("status", "")) != "NORMAL_EXIT"
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


def _seed_adapter_from_locked_ground_pose(
    adapter: Any,
    lock: dict[str, Any],
) -> dict[str, Any]:
    """Restore the audited recording-start pose before the strict ground settle.

    The raw USD spawn pose is intentionally not treated as recording evidence.
    Joint positions stay at the articulation's captured standing defaults, while
    the root pose comes from the audited 50 mm environment lock and every root
    and joint velocity is explicitly zeroed.
    """

    expected = dict(lock.get("selected_environment", {}) or {})
    position = list(expected.get("robot_initial_root_position_m", []) or [])
    orientation = list(expected.get("robot_initial_root_orientation_wxyz", []) or [])
    home = dict(expected.get("standing_home_command_state_deg", {}) or {})
    try:
        target_pose = [float(value) for value in position + orientation]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("locked recording-start root pose is non-numeric") from exc
    if len(position) != 3 or len(orientation) != 4 or not all(
        math.isfinite(value) for value in target_pose
    ):
        raise RuntimeError("locked recording-start root pose must contain 3+4 finite values")
    quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in orientation))
    if abs(quaternion_norm - 1.0) > 1.0e-3:
        raise RuntimeError(
            "locked recording-start orientation must be a unit quaternion; "
            f"norm={quaternion_norm}"
        )
    if set(home) != set(SERVO_JOINT_NAMES) or not all(
        math.isfinite(float(value)) for value in home.values()
    ):
        raise RuntimeError("locked standing-home command state is incomplete or non-finite")

    captured = dict(adapter.capture_sim_state() or {})
    if not bool(captured.get("pose_restore_eligible", False)):
        raise RuntimeError("adapter capture is not eligible for locked ground-pose restore")
    joint_names = [str(name) for name in list(captured.get("joint_names", []) or [])]
    joint_pos = captured.get("joint_pos")
    if not joint_names or joint_pos in (None, [], {}):
        raise RuntimeError("adapter capture has no complete standing joint pose")
    command_state = dict(captured.get("command_state", {}) or {})
    captured_wheels = dict(command_state.get("wheels", {}) or {})
    if set(captured_wheels) != set(WHEEL_JOINT_NAMES):
        raise RuntimeError("adapter capture has an incomplete wheel command state")

    seed_state = copy.deepcopy(captured)
    seed_state["root_pose"] = [target_pose]
    seed_state["root_velocity"] = [[0.0] * 6]
    seed_state["joint_vel"] = [[0.0] * len(joint_names)]
    seed_state["command_state"] = {
        "servos": {name: float(home[name]) for name in SERVO_JOINT_NAMES},
        "wheels": {name: 0.0 for name in WHEEL_JOINT_NAMES},
    }
    restore = dict(adapter.restore_sim_state(seed_state) or {})
    validation = dict(restore.get("validation", {}) or {})
    if not bool(validation.get("valid", False)):
        raise RuntimeError(
            "locked recording-start pose restore validation failed: "
            + str(validation.get("reason", "unknown validation failure"))
        )
    measured_state = dict(adapter.capture_sim_state() or {})
    post_write_verification = verify_restored_full_sim_pose(
        seed_state,
        measured_state,
        joint_names,
    )

    def finite_row(value: Any) -> list[float]:
        row = value
        while (
            isinstance(row, (list, tuple))
            and len(row) == 1
            and isinstance(row[0], (list, tuple))
        ):
            row = row[0]
        if not isinstance(row, (list, tuple)):
            return []
        try:
            numbers = [float(item) for item in row]
        except (TypeError, ValueError):
            return []
        return numbers if all(math.isfinite(item) for item in numbers) else []

    measured_root_velocity = finite_row(measured_state.get("root_velocity"))
    measured_joint_velocity = finite_row(measured_state.get("joint_vel"))
    root_velocity_max_abs = max([abs(value) for value in measured_root_velocity] + [float("inf")]) if not measured_root_velocity else max(abs(value) for value in measured_root_velocity)
    joint_velocity_max_abs = max([abs(value) for value in measured_joint_velocity] + [float("inf")]) if not measured_joint_velocity else max(abs(value) for value in measured_joint_velocity)
    velocity_tolerance = 1.0e-6
    velocities_verified = bool(
        len(measured_root_velocity) == 6
        and len(measured_joint_velocity) == len(joint_names)
        and root_velocity_max_abs <= velocity_tolerance
        and joint_velocity_max_abs <= velocity_tolerance
    )
    if not bool(post_write_verification.get("verified", False)) or not velocities_verified:
        raise RuntimeError(
            "locked recording-start pose was not verified after write: "
            f"pose={post_write_verification.get('reason', 'unknown')}; "
            f"root_velocity_max_abs={root_velocity_max_abs}; "
            f"joint_velocity_max_abs={joint_velocity_max_abs}"
        )
    raw_pose = captured.get("root_pose")
    return {
        "schema_version": "fsm50.locked_ground_seed.v1",
        "source": "environment_lock_50mm.json:selected_environment",
        "raw_spawn_root_pose": _jsonable(raw_pose),
        "requested_root_pose": target_pose,
        "requested_quaternion_norm": quaternion_norm,
        "root_velocity_written": [0.0] * 6,
        "joint_velocity_written_zero": True,
        "joint_count": len(joint_names),
        "joint_names": joint_names,
        "standing_joint_position_source": "captured_articulation_default",
        "standing_home_command_deg": {
            name: float(home[name]) for name in SERVO_JOINT_NAMES
        },
        "wheel_target_rad_s": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        "restore_validation": _jsonable(validation),
        "restore_trace": _jsonable(restore.get("trace", [])),
        "post_write_pose_verification": _jsonable(post_write_verification),
        "post_write_velocity_verification": {
            "verified": velocities_verified,
            "tolerance": velocity_tolerance,
            "root_velocity_max_abs": root_velocity_max_abs,
            "joint_velocity_max_abs": joint_velocity_max_abs,
        },
    }


def _verify_locked_source_hashes(lock: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw_path, expected in sorted(dict(lock.get("source_sha256", {}) or {}).items()):
        path = Path(str(raw_path)).resolve()
        actual = sha256_file(path) if path.is_file() else ""
        rows.append(
            {
                "path": str(path),
                "expected_sha256": str(expected).lower(),
                "actual_sha256": actual.lower(),
                "ok": bool(actual and actual.lower() == str(expected).lower()),
            }
        )
    return {"ok": bool(rows) and all(row["ok"] for row in rows), "files": rows}


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
) -> dict[str, Any]:
    last_sim_time_s = float((collector.last_row or {}).get("time_s", 0.0) or 0.0)
    scheduler = service.status_dict(
        current_sim_time_s=last_sim_time_s,
        current_wall_time_s=time.time(),
        compact=False,
    )
    physical = collector.physical_evidence()
    scheduler_complete = bool(service.stop_reason == "complete" and not timed_out)
    strict_success = bool(scheduler_complete and physical.get("physical_success", False))
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
        else "PARTIAL_SUCCESS"
        if valid_legs
        else "PHYSICAL_FAILURE"
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
        "accepted_steps_sha256": sha256_file(item.steps_path),
        "run_dir": str(run_dir),
        "requested_profile": "fast",
        "canonical_profile": str(plan.profile),
        "plan_sha256": str(plan.plan_sha256),
        "plan_event_count": len(plan.events),
        "plan_segment_count": len(plan.segments),
        "plan_final_time_s": float(plan.final_time_s),
        "respawn": _jsonable(respawn),
        "scheduler_complete": scheduler_complete,
        "scheduler_stop_reason": str(service.stop_reason or ""),
        "scheduler_status": _jsonable(scheduler),
        "timed_out": bool(timed_out),
        "physical_evidence": physical,
        "physical_success": bool(physical.get("physical_success", False)),
        "strict_full_success": strict_success,
        "classification": classification,
        "valid_linkage_lift_legs": valid_legs,
        "first_failure_phase": "" if strict_success else _first_failure(physical),
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
        "success_semantics": "scheduler completion and strict per-leg UNLOAD->AIR->CLEAR_FACE->TOP->LOAD evidence are independent; only their conjunction can be success",
    }


def _write_checksums(root: Path) -> None:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {
            "checksums.sha256",
            ".partial",
            ".complete",
            ".finalized",
            ".failed",
        }:
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


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
    collector.set_runtime_context(fsm_state=label, source_step=None, segment_index=None)
    while float(adapter.sim_time) + 1.0e-9 < target:
        adapter.step()


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
) -> dict[str, Any]:
    from sim_worker_runtime import handle_respawn
    from sim_obstacle_scene import measure_obstacle_geometry, measure_scene_baseline

    version_root = batch_root / item.version_id
    version_root.mkdir(parents=True, exist_ok=True)
    artifact_root = _new_directory(version_root, "clean_fast_replay")
    partial = artifact_root / ".partial"
    partial.write_text("running\n", encoding="utf-8")
    source_freeze = _source_freeze(item, robot_usd=robot_usd)
    write_json(artifact_root / "source_freeze_pre.json", source_freeze)
    write_json(artifact_root / "source_freeze.json", source_freeze)
    immediate_steps_sha256 = sha256_file(item.steps_path)
    recording_changed_after_preflight = (
        immediate_steps_sha256.lower() != str(expected_steps_sha256).lower()
    )
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
    try:
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
        config = _telemetry_config(artifact_root, args.telemetry_rate)
        collector_args = SimpleNamespace(
            headless=bool(args.headless), output_dir=str(artifact_root)
        )
        # No collector is attached across a respawn.  The new episode begins
        # only after both robot state and filtered-sensor history are clean.
        adapter.attach_telemetry(None)
        respawn = handle_respawn(adapter=adapter)
        if not bool(respawn.get("ok", False)):
            raise RuntimeError(str(respawn.get("error", "clean respawn failed")))
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
        )
        video_capture.start()
        adapter.attach_telemetry(collector)
        collector.record_event(
            float(adapter.sim_time),
            "clean_respawn_complete",
            severity="info",
            message=f"Clean respawn for {item.version_id}",
            extra={"respawn": _jsonable(respawn)},
        )
        started = service.start_plan(
            plan,
            current_sim_time_s=float(adapter.sim_time),
            current_wall_time_s=time.time(),
            start_delay_sim_s=float(args.post_respawn_settle_s),
            plan_id=f"recording-{item.version_id}-{uuid.uuid4().hex[:8]}",
            request_id=uuid.uuid4().hex,
            worker_session_id=f"fsm50-{uuid.uuid4().hex[:12]}",
        )
        if not started:
            raise RuntimeError(service.last_error or "Fast replay plan did not start")
        run_start_sim = float(adapter.sim_time)
        deadline_sim = run_start_sim + max(
            float(args.timeout_s),
            float(args.post_respawn_settle_s) + float(plan.final_time_s) * float(args.timeout_scale),
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
            segment_index = int(service.segment_index)
            source_step = None
            if 0 <= segment_index < len(plan.segments):
                source_step = int(plan.segments[segment_index].source_step)
            collector.set_runtime_context(
                fsm_state="RECORDING_FAST_REPLAY",
                segment_index=segment_index,
                source_step=source_step,
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
            video = {
                "valid": False,
                "artifact_valid": False,
                "actual_viewport_video": False,
                "not_camera_video": False,
                "contact_mode": contact_mode,
                "video_path": str(Path(run_dir) / "fsm50_viewport.mp4"),
                "video_sha256": "",
                "manifest_path": str(Path(run_dir) / "viewport_video_manifest.json"),
                "manifest_sha256": "",
                "error": f"video failure artifact write failed: {type(video_exc).__name__}: {video_exc}",
            }
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
            "classification": "RUNNER_EXCEPTION",
            "strict_full_success": False,
            "physical_success": False,
            "scheduler_complete": False,
            "first_failure_phase": "RUNNER_EXCEPTION",
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


def _run_recording_replays_locked(
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
        if not ground_reference_result_is_valid(ground):
            raise RuntimeError(f"ground reference validation failed: {ground}")
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
        _append_runtime_lock(readback, path=environment_lock_path)
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
        try:
            _update_required_reports(results)
        except Exception as exc:
            finalization_errors.append(
                f"required_reports: {type(exc).__name__}: {exc}"
            )
            exit_code = 1
        batch_valid = bool(
            not batch_error
            and batch_source_comparison.get("equal", False)
            and not finalization_errors
            and all(
                not bool(dict(result.get("lifecycle", {}) or {}).get("failed", False))
                for result in results
            )
        )
        batch_finalization = {
            "artifact_root": str(batch_root),
            "finalized": batch_valid,
            "failed": not batch_valid,
            "strict_success": bool(
                results
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
                _write_checksums(batch_root)
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
                _write_checksums(batch_root)
                immutable_preclose_errors.extend(
                    f"manifest_failure_refresh: {error}"
                    for error in _snapshot_preclose_files(batch_root)
                )
            except Exception as rewrite_exc:
                immutable_preclose_errors.append(
                    "manifest_failure_rewrite: "
                    f"{type(rewrite_exc).__name__}: {rewrite_exc}"
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
                _write_checksums(batch_root)
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
        try:
            if simulation_app is not None:
                simulation_app.close(wait_for_replicator=False)
            elif scene_handle is not None:
                scene_handle.close()
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
            exit_code = 1
            batch_finalization.update(
                {
                    "finalized": False,
                    "failed": True,
                    "strict_success": False,
                    "close_error": close_error,
                    "phase": "SIMULATION_CLOSE_ERROR",
                    "failure_reason": "SIMULATION_CLOSE_ERROR",
                }
            )
            try:
                _atomic_write_json(
                    batch_root / "batch_finalization.json", batch_finalization
                )
                _write_checksums(batch_root)
                _mark_artifact_root(batch_root, valid=False)
            except Exception as finalize_exc:
                print(
                    f"[FSM50] close-error evidence failed: "
                    f"{type(finalize_exc).__name__}: {finalize_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        if supervisor is not None:
            supervisor.mark_close_returned(close_error=close_error)
    print(json.dumps({"batch_root": str(batch_root), "results": results}, ensure_ascii=False, indent=2))
    return exit_code


def run_recording_replays(args: argparse.Namespace) -> int:
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
        if str(outcome.get("status")) != "NORMAL_EXIT":
            return 1
        return int(outcome.get("child_returncode", 1) or 0)
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

    replay = subparsers.add_parser("replay-recordings", help="Replay recording versions in one Isaac instance.")
    replay.add_argument("--versions", nargs="+", default=["all"])
    replay.add_argument("--recording-root", type=Path, default=DEFAULT_RECORDING_ROOT)
    replay.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    replay.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT / "recording_replays")
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
        default="instrumented",
        help=(
            "formal uses the production aggregate ContactSensor; instrumented "
            "uses only the filtered wheel/non-wheel telemetry factory."
        ),
    )
    replay.add_argument("--post-respawn-settle-s", type=float, default=0.30)
    replay.add_argument("--post-run-settle-s", type=float, default=0.50)
    replay.add_argument("--video-fps", type=float, default=15.0)
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
    if args.command == "audit":
        result = RecordingAudit(args.recording_root, args.report_root).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
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
