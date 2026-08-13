"""Subprocess client and Windows-safe launcher for the Isaac Sim worker.

The Tk process owns this client. Isaac Sim itself lives in a separate Python
process and communicates through a localhost newline-JSON socket.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from isaac_launch_preflight import (
    IsaacInterpreterReport,
    format_preflight_error,
    inspect_launcher_environment,
    quote_windows_cmd_arg,
    run_interpreter_preflight,
    run_isaaclab_bat_preflight,
)
from sim_ipc_protocol import JsonLineBuffer, encode_message, make_message
from worker_startup_diagnostics import (
    classify_startup_error,
    diagnose_startup_phase,
    last_meaningful_line,
    summarize_environment,
)


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_ISAACLAB_BAT = Path("C:/robotics_sim/IsaacLab/isaaclab.bat")
WORKER_SCRIPT = MODULE_ROOT / "sim_worker_process.py"
CONFIG_DIR = Path(tempfile.gettempdir()) / "height_replay_worker_configs"
LOG_DIR = Path(tempfile.gettempdir()) / "height_replay_worker_logs"


VALUE_FLAGS = {
    "--ipc-host",
    "--ipc-port",
    "--height-cm",
    "--robot-usd",
    "--save-usd",
    "--spawn-z",
    "--obstacle-x",
    "--obstacle-width",
    "--obstacle-length",
    "--infer-obstacle-size",
    "--robot-width",
    "--robot-length",
    "--physics-dt",
    "--render-interval",
    "--wheel-direction",
    "--max-wheel-speed-rad-s",
    "--default-wheel-speed-rad-s",
    "--servo-stiffness",
    "--servo-damping",
    "--wheel-damping",
    "--device",
    "--sim-status-refresh-ms",
    "--sim-worker-log-lines",
    "--camera-parent-prim",
    "--camera-width",
    "--camera-height",
    "--camera-update-period-s",
    "--camera-offset-x",
    "--camera-offset-y",
    "--camera-offset-z",
    "--camera-pitch-deg",
    "--camera-aim-mode",
    "--camera-target-x",
    "--camera-target-y",
    "--camera-target-z",
    "--camera-target-frame",
    "--camera-look-at-roll-deg",
    "--camera-focal-length",
    "--camera-horizontal-aperture",
    "--camera-near-clip-m",
    "--camera-far-clip-m",
    "--vision-confidence-threshold",
    "--vision-stable-frames",
    "--vision-window-size",
    "--vision-height-tolerance-cm",
    "--worker-smoke-camera-height-cm",
    "--worker-smoke-camera-validation-s",
    "--worker-smoke-camera-output",
    "--worker-smoke-camera-counterfactual-output",
    "--worker-smoke-output",
    "--camera-view-pending-timeout-s",
    "--camera-view-pending-max-retries",
    "--robot-ground-settle-s",
    "--robot-ground-settle-max-steps",
    "--robot-ground-stable-frames",
    "--robot-ground-vertical-speed-threshold-m-s",
    "--robot-ground-joint-speed-threshold-rad-s",
    "--robot-ground-servo-speed-threshold-rad-s",
    "--robot-ground-wheel-speed-threshold-rad-s",
    "--robot-ground-clearance-m",
    "--robot-ground-penetration-tolerance-m",
    "--robot-max-ground-correction-m",
    "--livestream",
    "--experience",
    "--telemetry-rate",
    "--visualized-env-id",
    "--output-dir",
    "--telemetry-config",
    "--worker-config-file",
}


class TailBuffer:
    def __init__(self, max_lines: int = 200):
        self.max_lines = max(1, int(max_lines))
        self.lines: deque[str] = deque(maxlen=self.max_lines)

    def extend(self, lines: list[str]) -> None:
        for line in lines:
            self.lines.append(str(line))

    def snapshot(self) -> list[str]:
        return list(self.lines)


@dataclass
class WorkerLaunchPlan:
    argv: list[str] = field(default_factory=list)
    cwd: str = str(PROJECT_ROOT)
    env: dict[str, str] = field(default_factory=dict)
    display_command: str = ""
    launch_mode: str = ""
    resolved_python: str = ""
    config_path: str = ""
    wrapper_path: str = ""
    warnings: list[str] = field(default_factory=list)
    requested_launch_mode: str = "auto"
    resolved_launch_mode: str = ""
    python_version: str = ""
    isaacsim_version: str = ""
    isaaclab_version: str = ""
    preflight_ok: bool = False
    preflight_error: str = ""
    error_category: str = ""
    candidate_reports: list[dict[str, Any]] = field(default_factory=list)
    key_environment: dict[str, str] = field(default_factory=dict)
    requested_headless: bool = False
    effective_headless: bool = False
    requested_livestream: int = 0
    effective_livestream: int = 0
    requested_enable_cameras: bool = False
    effective_enable_cameras: bool = False
    selected_experience: str = ""
    eula_source: str = "unset"

    def to_status(self) -> dict[str, Any]:
        data = asdict(self)
        data["env"] = summarize_environment(self.env)
        return data


def tail_file(path: str | Path | None, max_lines: int) -> list[str]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    try:
        return source.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, int(max_lines)) :]
    except Exception as exc:
        return [f"[tail-error] {source}: {exc}"]


def build_worker_command(args: Any, *, host: str, port: int) -> list[str]:
    """Build a direct worker CLI for compatibility and unit tests.

    The real subprocess launcher uses ``--worker-config-file`` to avoid long
    Windows batch argv strings. This function remains useful for old tests and
    for users who want to inspect the equivalent direct CLI.
    """

    command = [sys.executable, "-u", str(WORKER_SCRIPT), "--worker"]
    config = build_worker_config(args, host=host, port=port)
    _append_worker_config_as_cli(command, config)
    validate_worker_argv(command)
    return command


def build_worker_config(args: Any, *, host: str, port: int) -> dict[str, Any]:
    servo_threshold = getattr(args, "robot_ground_servo_speed_threshold_rad_s", None)
    if servo_threshold is None:
        servo_threshold = getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)
    telemetry_enabled = getattr(args, "telemetry_effective_enabled", getattr(args, "telemetry_enabled", None))
    if telemetry_enabled is None:
        telemetry_enabled = True
    config: dict[str, Any] = {
        "ipc_host": str(host),
        "ipc_port": int(port),
        "height_cm": int(getattr(args, "height_cm", 0)),
        "robot_usd": str(getattr(args, "robot_usd")),
        "save_usd": str(getattr(args, "save_usd")),
        "spawn_z": float(getattr(args, "spawn_z", 0.04)),
        "obstacle_x": float(getattr(args, "obstacle_x", 1.55)),
        "infer_obstacle_size": bool(getattr(args, "infer_obstacle_size", True)),
        "robot_width": float(getattr(args, "robot_width", 0.80)),
        "robot_length": float(getattr(args, "robot_length", 0.55)),
        "physics_dt": float(getattr(args, "physics_dt", 1.0 / 120.0)),
        "render_interval": int(getattr(args, "render_interval", 2)),
        "wheel_direction": float(getattr(args, "wheel_direction", 1.0)),
        "max_wheel_speed_rad_s": float(
            getattr(args, "max_wheel_speed_rad_s", getattr(args, "max_wheel_speed", 2.0943951023931953))
        ),
        "default_wheel_speed_rad_s": float(
            getattr(args, "default_wheel_speed_rad_s", getattr(args, "default_wheel_speed", 0.5235987755982988))
        ),
        "servo_stiffness": float(getattr(args, "servo_stiffness", 600.0)),
        "servo_damping": float(getattr(args, "servo_damping", 60.0)),
        "wheel_damping": float(getattr(args, "wheel_damping", 20.0)),
        "device": str(getattr(args, "device", "cuda:0")),
        "sim_status_refresh_ms": int(getattr(args, "sim_status_refresh_ms", 250)),
        "sim_worker_log_lines": int(getattr(args, "sim_worker_log_lines", 200)),
        "save_scene": bool(getattr(args, "save_scene", True)),
        "onboard_camera": bool(getattr(args, "onboard_camera", True)),
        "camera_width": int(getattr(args, "camera_width", 424)),
        "camera_height": int(getattr(args, "camera_height", 240)),
        "camera_update_period_s": float(getattr(args, "camera_update_period_s", 0.10)),
        "camera_offset_x": float(getattr(args, "camera_offset_x", 0.35)),
        "camera_offset_y": float(getattr(args, "camera_offset_y", 0.0)),
        "camera_offset_z": float(getattr(args, "camera_offset_z", 0.18)),
        "camera_pitch_deg": float(getattr(args, "camera_pitch_deg", 14.0)),
        "camera_aim_mode": str(getattr(args, "camera_aim_mode", "pitch") or "pitch"),
        "camera_target_x": float(getattr(args, "camera_target_x", 1.55)),
        "camera_target_y": float(getattr(args, "camera_target_y", 0.0)),
        "camera_target_z": float(getattr(args, "camera_target_z", 0.02)),
        "camera_target_frame": str(getattr(args, "camera_target_frame", "world") or "world"),
        "camera_look_at_roll_deg": float(getattr(args, "camera_look_at_roll_deg", 0.0)),
        "camera_coverage_strict": bool(getattr(args, "camera_coverage_strict", False)),
        "camera_focal_length": float(getattr(args, "camera_focal_length", 24.0)),
        "camera_horizontal_aperture": float(getattr(args, "camera_horizontal_aperture", 20.955)),
        "camera_near_clip_m": float(getattr(args, "camera_near_clip_m", 0.05)),
        "camera_far_clip_m": float(getattr(args, "camera_far_clip_m", 6.0)),
        "vision_confidence_threshold": float(getattr(args, "vision_confidence_threshold", 0.75)),
        "vision_stable_frames": int(getattr(args, "vision_stable_frames", 5)),
        "vision_window_size": int(getattr(args, "vision_window_size", 7)),
        "vision_height_tolerance_cm": float(getattr(args, "vision_height_tolerance_cm", 2.0)),
        "headless": bool(getattr(args, "headless", False)),
        "livestream": int(getattr(args, "livestream", 0) or 0),
        "enable_cameras": bool(getattr(args, "onboard_camera", True)) or bool(getattr(args, "enable_cameras", False)),
        "apply_safe_servo_joint_limits": bool(getattr(args, "apply_safe_servo_joint_limits", True)),
        "apply_physx_joint_limits": bool(getattr(args, "apply_physx_joint_limits", False)),
        "no_continuous_sim_step": bool(getattr(args, "no_continuous_sim_step", False)),
        "worker_smoke_negative_knee_test": bool(getattr(args, "worker_smoke_negative_knee_test", False)),
        "worker_smoke_camera_detection": bool(getattr(args, "worker_smoke_camera_detection", False)),
        "worker_smoke_camera_provenance": bool(getattr(args, "worker_smoke_camera_provenance", False)),
        "worker_smoke_camera_pose_ab": bool(getattr(args, "worker_smoke_camera_pose_ab", False)),
        "worker_smoke_camera_counterfactual": bool(getattr(args, "worker_smoke_camera_counterfactual", False)),
        "worker_smoke_camera_view_ground_contact": bool(getattr(args, "worker_smoke_camera_view_ground_contact", False)),
        "worker_smoke_ground_structure": bool(getattr(args, "worker_smoke_ground_structure", False)),
        "worker_smoke_ground_calibration": bool(getattr(args, "worker_smoke_ground_calibration", False)),
        "worker_smoke_vision_playback": bool(getattr(args, "worker_smoke_vision_playback", False)),
        "worker_smoke_camera_height_cm": getattr(args, "worker_smoke_camera_height_cm", None),
        "worker_smoke_camera_validation_s": float(getattr(args, "worker_smoke_camera_validation_s", 10.0)),
        "worker_smoke_camera_output": str(getattr(args, "worker_smoke_camera_output", "") or ""),
        "worker_smoke_camera_counterfactual_output": str(getattr(args, "worker_smoke_camera_counterfactual_output", "") or ""),
        "worker_smoke_output": str(getattr(args, "worker_smoke_output", "") or ""),
        "worker_smoke_test_s": float(getattr(args, "worker_smoke_test_s", 0.0)),
        "viewport_physics_guard": bool(getattr(args, "viewport_physics_guard", True)),
        "defer_first_visible_render": bool(getattr(args, "defer_first_visible_render", True)),
        "camera_view_active_fallback": bool(getattr(args, "camera_view_active_fallback", False)),
        "camera_view_pending_timeout_s": float(getattr(args, "camera_view_pending_timeout_s", 10.0)),
        "camera_view_pending_max_retries": int(getattr(args, "camera_view_pending_max_retries", 30)),
        "robot_ground_settle_s": float(getattr(args, "robot_ground_settle_s", 0.75)),
        "robot_ground_settle_max_steps": int(getattr(args, "robot_ground_settle_max_steps", 180)),
        "robot_ground_stable_frames": int(getattr(args, "robot_ground_stable_frames", 10)),
        "robot_ground_vertical_speed_threshold_m_s": float(getattr(args, "robot_ground_vertical_speed_threshold_m_s", 0.01)),
        "robot_ground_joint_speed_threshold_rad_s": float(getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)),
        "robot_ground_servo_speed_threshold_rad_s": float(servo_threshold),
        "robot_ground_wheel_speed_threshold_rad_s": float(getattr(args, "robot_ground_wheel_speed_threshold_rad_s", 0.20)),
        "robot_ground_clearance_m": float(getattr(args, "robot_ground_clearance_m", 0.002)),
        "robot_ground_penetration_tolerance_m": float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)),
        "robot_auto_ground_correction": bool(getattr(args, "robot_auto_ground_correction", False)),
        "robot_max_ground_correction_m": float(getattr(args, "robot_max_ground_correction_m", 0.10)),
        "accept_isaac_eula": bool(getattr(args, "accept_isaac_eula", False)),
        "telemetry_enabled": bool(telemetry_enabled),
        "live_viz_enabled": bool(getattr(args, "live_viz_effective_enabled", True)),
        "telemetry_report": bool(getattr(args, "telemetry_report_effective_enabled", True)),
        "equilibrium_region_enabled": bool(getattr(args, "equilibrium_region_effective_enabled", True)),
    }
    optional_values = {
        "obstacle_width": getattr(args, "obstacle_width", None),
        "obstacle_length": getattr(args, "obstacle_length", None),
        "camera_parent_prim": str(getattr(args, "camera_parent_prim", "") or "").strip(),
        "experience": str(getattr(args, "experience", "") or "").strip(),
        "telemetry_rate": getattr(args, "telemetry_rate", None),
        "visualized_env_id": getattr(args, "visualized_env_id", None),
        "output_dir": str(getattr(args, "output_dir", "") or "").strip(),
        "telemetry_config": str(getattr(args, "telemetry_config", "") or "").strip(),
    }
    for key, value in optional_values.items():
        if value is not None and value != "":
            config[key] = value
    return {key: value for key, value in config.items() if value is not None}


def write_worker_config(config: dict[str, Any]) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    path = CONFIG_DIR / f"worker_{stamp}.json"
    payload = {
        "created_at": time.time(),
        "worker_script": str(WORKER_SCRIPT),
        "args": dict(config),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_worker_argv(argv: list[str], *, config_path: str | Path | None = None) -> None:
    if not isinstance(argv, list) or not argv:
        raise ValueError("Worker argv is empty.")
    for index, item in enumerate(argv):
        if item is None:
            raise ValueError(f"Worker argv contains None at index {index}: {argv!r}")
        if str(item) == "":
            raise ValueError(f"Worker argv contains an empty string at index {index}: {argv!r}")
    for index, item in enumerate(argv):
        if item in VALUE_FLAGS:
            if index + 1 >= len(argv):
                raise ValueError(f"Worker argv flag {item} is missing its value: {argv!r}")
            if str(argv[index + 1]) == "":
                raise ValueError(f"Worker argv flag {item} has an empty value: {argv!r}")
    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"Worker script does not exist: {WORKER_SCRIPT}")
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Worker config JSON does not exist: {path}")
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Worker config JSON is unreadable: {path}: {exc}") from exc


def build_direct_python_launcher(
    *,
    python_exe: str | Path,
    worker_script: str | Path,
    config_path: str | Path,
    cwd: str | Path,
    env: dict[str, str],
) -> WorkerLaunchPlan:
    python_path = Path(python_exe)
    if not python_path.exists():
        raise FileNotFoundError(f"Python executable does not exist: {python_path}")
    script = Path(worker_script)
    if not script.exists():
        raise FileNotFoundError(f"Worker script does not exist: {script}")
    config = Path(config_path)
    if not config.exists():
        raise FileNotFoundError(f"Worker config JSON does not exist: {config}")
    argv = [str(python_path), "-u", str(script), "--worker", "--worker-config-file", str(config)]
    validate_worker_argv(argv, config_path=config)
    return WorkerLaunchPlan(
        argv=argv,
        cwd=str(cwd),
        env=dict(env),
        display_command=_display_command(argv),
        launch_mode="direct-python",
        resolved_launch_mode="current-python",
        resolved_python=str(python_path),
        config_path=str(config),
    )


def build_windows_batch_launcher(
    *,
    isaaclab_bat: str | Path,
    worker_script: str | Path,
    config_path: str | Path,
    cwd: str | Path,
    env: dict[str, str],
) -> WorkerLaunchPlan:
    bat = Path(isaaclab_bat)
    if not bat.exists():
        raise FileNotFoundError(f"IsaacLab batch launcher does not exist: {bat}")
    script = Path(worker_script)
    if not script.exists():
        raise FileNotFoundError(f"Worker script does not exist: {script}")
    config = Path(config_path)
    if not config.exists():
        raise FileNotFoundError(f"Worker config JSON does not exist: {config}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = CONFIG_DIR / f"launch_worker_{int(time.time() * 1000)}.cmd"
    wrapper_lines = [
        "@echo off",
        "setlocal EnableExtensions",
        "call "
        + " ".join(
            [
                quote_windows_cmd_arg(str(bat)),
                "-p",
                quote_windows_cmd_arg(str(script)),
                "--worker",
                "--worker-config-file",
                quote_windows_cmd_arg(str(config)),
            ]
        ),
        "exit /b %ERRORLEVEL%",
    ]
    wrapper.write_text("\r\n".join(wrapper_lines) + "\r\n", encoding="utf-8")
    argv = ["cmd.exe", "/d", "/c", str(wrapper)]
    validate_worker_argv(argv, config_path=config)
    return WorkerLaunchPlan(
        argv=argv,
        cwd=str(cwd),
        env=dict(env),
        display_command=_display_command(argv) + f"  # wrapper: {wrapper}",
        launch_mode="isaaclab-bat",
        resolved_launch_mode="isaaclab-bat",
        resolved_python="",
        config_path=str(config),
        wrapper_path=str(wrapper),
    )


def build_worker_launch_plan(
    args: Any,
    *,
    host: str,
    port: int,
    run_preflight_checks: bool = True,
) -> WorkerLaunchPlan:
    requested = resolve_requested_launch_mode(args)
    env, env_status = build_child_environment(args)
    config = build_worker_config(args, host=host, port=port)
    config_path = write_worker_config(config)
    candidate_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    preflight_timeout = float(getattr(args, "preflight_timeout_s", 30.0))
    selected: tuple[str, str | Path, IsaacInterpreterReport] | None = None

    for candidate_mode, candidate_path in iter_launch_candidates(args, requested):
        report = _run_candidate_preflight(
            candidate_mode,
            candidate_path,
            args=args,
            env=env,
            timeout_s=preflight_timeout,
            enabled=run_preflight_checks,
        )
        report_data = report.to_dict()
        report_data["candidate_launch_mode"] = candidate_mode
        report_data["candidate_path"] = str(candidate_path)
        candidate_reports.append(report_data)
        if _report_is_launchable(report):
            selected = (candidate_mode, candidate_path, report)
            break

    if selected is None:
        preflight_error = _summarize_candidate_failures(candidate_reports)
        return WorkerLaunchPlan(
            argv=[],
            cwd=str(PROJECT_ROOT),
            env=env,
            display_command="",
            launch_mode=requested,
            resolved_launch_mode="",
            resolved_python="",
            config_path=str(config_path),
            warnings=warnings,
            requested_launch_mode=requested,
            preflight_ok=False,
            preflight_error=preflight_error,
            error_category=classify_startup_error(preflight_error),
            candidate_reports=candidate_reports,
            **env_status,
        )

    selected_mode, selected_path, report = selected
    if selected_mode == "isaaclab-bat":
        plan = build_windows_batch_launcher(
            isaaclab_bat=selected_path,
            worker_script=WORKER_SCRIPT,
            config_path=config_path,
            cwd=PROJECT_ROOT,
            env=env,
        )
        plan.resolved_python = report.executable
    else:
        plan = build_direct_python_launcher(
            python_exe=selected_path,
            worker_script=WORKER_SCRIPT,
            config_path=config_path,
            cwd=PROJECT_ROOT,
            env=env,
        )
        plan.resolved_launch_mode = selected_mode
    plan.requested_launch_mode = requested
    plan.launch_mode = plan.resolved_launch_mode or selected_mode
    plan.preflight_ok = True
    plan.preflight_error = ""
    plan.candidate_reports = candidate_reports
    plan.warnings = warnings
    plan.python_version = report.python_version
    plan.isaacsim_version = report.isaacsim_version
    plan.isaaclab_version = report.isaaclab_version
    plan.error_category = ""
    for key, value in env_status.items():
        setattr(plan, key, value)
    plan.key_environment = summarize_environment(env)
    return plan


def run_launch_preflight_for_args(args: Any) -> dict[str, Any]:
    env, env_status = build_child_environment(args)
    requested = resolve_requested_launch_mode(args)
    reports: list[dict[str, Any]] = []
    selected_report: dict[str, Any] | None = None
    timeout_s = float(getattr(args, "preflight_timeout_s", 30.0))
    for candidate_mode, candidate_path in iter_launch_candidates(args, requested):
        report = _run_candidate_preflight(
            candidate_mode,
            candidate_path,
            args=args,
            env=env,
            timeout_s=timeout_s,
            enabled=True,
        )
        data = report.to_dict()
        data["candidate_launch_mode"] = candidate_mode
        data["candidate_path"] = str(candidate_path)
        data["launchable"] = _report_is_launchable(report)
        reports.append(data)
        if data["launchable"] and selected_report is None:
            selected_report = data
            break
    return {
        "requested_launch_mode": requested,
        "preflight_ok": selected_report is not None,
        "selected_report": selected_report or {},
        "candidate_reports": reports,
        "preflight_error": "" if selected_report else _summarize_candidate_failures(reports),
        "environment": summarize_environment(env),
        **env_status,
    }


def resolve_requested_launch_mode(args: Any) -> str:
    legacy = getattr(args, "worker_python_mode", None)
    mode = str(getattr(args, "worker_launch_mode", "") or "").strip() or "auto"
    if legacy:
        mode = str(legacy)
    if mode not in {"auto", "current-python", "isaaclab-bat", "explicit-python"}:
        return "auto"
    if mode == "explicit-python" and not str(getattr(args, "worker_python_exe", "") or "").strip():
        return "auto"
    return mode


def iter_launch_candidates(args: Any, requested: str) -> list[tuple[str, str | Path]]:
    candidates: list[tuple[str, str | Path]] = []
    explicit = str(getattr(args, "worker_python_exe", "") or "").strip()
    isaaclab_bat = Path(str(getattr(args, "isaaclab_bat", DEFAULT_ISAACLAB_BAT)))
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_python = Path(conda_prefix) / "python.exe" if conda_prefix else None
    bundled_dir = isaaclab_bat.parent / "_isaac_sim"
    bundled_candidates = [bundled_dir / "python.bat", bundled_dir / "python.exe"]
    if requested == "explicit-python":
        return [("explicit-python", explicit)]
    if requested == "current-python":
        return [("current-python", sys.executable)]
    if requested == "isaaclab-bat":
        return [("isaaclab-bat", isaaclab_bat)]
    if explicit:
        candidates.append(("explicit-python", explicit))
    candidates.append(("current-python", sys.executable))
    if conda_python is not None:
        candidates.append(("conda-python", conda_python))
    for path in bundled_candidates:
        candidates.append(("bundled-python", path))
    candidates.append(("isaaclab-bat", isaaclab_bat))
    deduped: list[tuple[str, str | Path]] = []
    seen: set[str] = set()
    for mode, path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((mode, path))
    return deduped


def build_child_environment(args: Any) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ)
    requested_headless = bool(getattr(args, "headless", False))
    requested_livestream = int(getattr(args, "livestream", 0) or 0)
    requested_enable_cameras = bool(getattr(args, "onboard_camera", True)) or bool(getattr(args, "enable_cameras", False))
    effective_livestream = requested_livestream if requested_livestream >= 0 else 0
    effective_headless = bool(requested_headless)
    if effective_livestream in {1, 2}:
        effective_headless = True
    effective_enable_cameras = bool(requested_enable_cameras)
    env["HEADLESS"] = "1" if effective_headless else "0"
    env["LIVESTREAM"] = str(effective_livestream)
    env["ENABLE_CAMERAS"] = "1" if effective_enable_cameras else "0"
    eula_source = "unset"
    if bool(getattr(args, "accept_isaac_eula", False)):
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        eula_source = "cli"
    elif env.get("OMNI_KIT_ACCEPT_EULA"):
        eula_source = "environment"
    status = {
        "key_environment": summarize_environment(env),
        "requested_headless": requested_headless,
        "effective_headless": effective_headless,
        "requested_livestream": requested_livestream,
        "effective_livestream": effective_livestream,
        "requested_enable_cameras": requested_enable_cameras,
        "effective_enable_cameras": effective_enable_cameras,
        "selected_experience": str(getattr(args, "experience", "") or ""),
        "eula_source": eula_source,
    }
    return env, status


class SimProcessClient:
    def __init__(self, args: Any):
        self.args = args
        self.host = "127.0.0.1"
        self.server: socket.socket | None = None
        self.conn: socket.socket | None = None
        self.process: subprocess.Popen[Any] | None = None
        self.stdout_path: Path | None = None
        self.stderr_path: Path | None = None
        self._stdout_handle: Any | None = None
        self._stderr_handle: Any | None = None
        self.line_buffer = JsonLineBuffer()
        self.pending_messages: list[dict[str, Any]] = []
        self.launch_plan: WorkerLaunchPlan | None = None
        self.latest_status: dict[str, Any] = {
            "ready": False,
            "starting": False,
            "phase": "not_started",
            "error": "",
            "traceback": "",
        }
        self.start_time = 0.0
        self.last_status_time = 0.0
        self.returncode: int | None = None
        self.last_error = ""
        self.last_log_activity_at = 0.0
        self.last_log_activity_monotonic = 0.0
        self._last_log_sizes: dict[str, int] = {}

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def connected(self) -> bool:
        return self.conn is not None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self._reset_socket()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, 0))
        self.server.listen(1)
        self.server.setblocking(False)
        port = int(self.server.getsockname()[1])
        self.latest_status = {
            "ready": False,
            "starting": True,
            "phase": "preflight_started",
            "error": "",
            "traceback": "",
            "requested_launch_mode": resolve_requested_launch_mode(self.args),
        }
        self.start_time = time.monotonic()
        self.last_status_time = self.start_time
        try:
            self.launch_plan = build_worker_launch_plan(self.args, host=self.host, port=port)
        except Exception as exc:
            self.latest_status.update(
                {
                    "ready": False,
                    "starting": False,
                    "phase": "launch_plan_failed",
                    "error": str(exc),
                    "error_category": classify_startup_error(str(exc)),
                }
            )
            self.close()
            return
        self.latest_status.update(self.launch_plan.to_status())
        if not self.launch_plan.preflight_ok:
            self.latest_status.update(
                {
                    "ready": False,
                    "starting": False,
                    "phase": "preflight_failed",
                    "error": self.launch_plan.preflight_error,
                    "error_category": self.launch_plan.error_category or classify_startup_error(self.launch_plan.preflight_error),
                    "startup_diagnosis": diagnose_startup_phase(
                        phase="preflight_failed",
                        connected=False,
                        ready=False,
                        error_category=self.launch_plan.error_category,
                        error=self.launch_plan.preflight_error,
                    ),
                }
            )
            self.close()
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        self.stdout_path = LOG_DIR / f"worker_{stamp}_stdout.log"
        self.stderr_path = LOG_DIR / f"worker_{stamp}_stderr.log"
        self._stdout_handle = self.stdout_path.open("w", encoding="utf-8", errors="replace")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8", errors="replace")
        try:
            self.process = subprocess.Popen(
                self.launch_plan.argv,
                cwd=self.launch_plan.cwd,
                env=self.launch_plan.env,
                stdout=self._stdout_handle,
                stderr=self._stderr_handle,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:
            self.latest_status.update(
                {
                    "ready": False,
                    "starting": False,
                    "phase": "process_spawn_failed",
                    "error": str(exc),
                    "error_category": classify_startup_error(str(exc)),
                }
            )
            self.close()
            return
        self.start_time = time.monotonic()
        self.last_status_time = self.start_time
        self.last_log_activity_at = time.time()
        self.last_log_activity_monotonic = self.start_time
        self.latest_status.update(
            {
                "ready": False,
                "starting": True,
                "phase": "process_spawned",
                "error": "",
                "traceback": "",
                "worker_pid": self.pid,
                "worker_command": list(self.launch_plan.argv),
                "display_command": self.launch_plan.display_command,
                "worker_cwd": self.launch_plan.cwd,
                "stdout_path": str(self.stdout_path),
                "stderr_path": str(self.stderr_path),
                "worker_config_path": self.launch_plan.config_path,
                "worker_wrapper_path": self.launch_plan.wrapper_path,
                "worker_env": summarize_environment(self.launch_plan.env),
            }
        )

    def poll(self) -> list[dict[str, Any]]:
        self._update_log_activity()
        if self.server is not None and self.conn is None:
            try:
                self.conn, _addr = self.server.accept()
                self.conn.setblocking(False)
                self.latest_status["phase"] = "ipc_connected"
                self._flush_pending()
            except BlockingIOError:
                self.latest_status.setdefault("phase", "waiting_for_ipc")
        messages: list[dict[str, Any]] = []
        if self.conn is not None:
            while True:
                try:
                    chunk = self.conn.recv(65536)
                except BlockingIOError:
                    break
                except OSError as exc:
                    self.last_error = str(exc)
                    break
                if not chunk:
                    break
                for message in self.line_buffer.feed(chunk):
                    self._handle_message(message)
                    messages.append(dict(self.latest_status))
        self._detect_log_startup_failure()
        if self.process is not None:
            self.returncode = self.process.poll()
            if self.returncode is not None and not self.latest_status.get("ready", False):
                self._mark_exited_before_ipc()
        self._apply_timeouts()
        return messages

    def send_command(self, command: str, *, source: str = "ui", **metadata: Any) -> None:
        message = make_message("command", command=str(command), source=source)
        for key in ("playback_label", "playback_event_index", "playback_event_count", "playback_final_time_s", "source_step"):
            if key in metadata and metadata[key] is not None:
                message[key] = metadata[key]
        self._send_or_queue(message)

    def set_height(self, height_cm: int, **payload: Any) -> None:
        message = make_message("set_height", height_cm=int(height_cm))
        message.update(payload)
        self._send_or_queue(message)

    def respawn(self) -> None:
        self._send_or_queue(make_message("respawn"))

    def restore_sim_state(self, sim_state: dict[str, Any]) -> None:
        self._send_or_queue(make_message("restore_sim_state", sim_state=dict(sim_state or {})))

    def stop_wheels(self) -> None:
        self._send_or_queue(make_message("stop_wheels"))

    def request_state(self) -> None:
        self._send_or_queue(make_message("request_state"))

    def vision_control(self, action: str, **payload: Any) -> None:
        message = make_message("vision_control", action=str(action))
        message.update(payload)
        self._send_or_queue(message)

    def set_vision_enabled(self, enabled: bool) -> None:
        self.vision_control("enable" if enabled else "disable")

    def request_vision_detection_once(self) -> None:
        self.vision_control("detect_once")

    def reset_vision_filter(self) -> None:
        self.vision_control("reset_filter")

    def save_vision_debug_frame(self) -> None:
        self.vision_control("save_debug_frame")

    def validate_camera(self) -> None:
        self.vision_control("validate_camera")

    def validate_current_height(self, expected_height_cm: int) -> None:
        self.vision_control("validate_current_height", expected_height_cm=int(expected_height_cm))

    def save_rgbd_diagnostic(self, expected_height_cm: int | None = None) -> None:
        payload: dict[str, Any] = {}
        if expected_height_cm is not None:
            payload["expected_height_cm"] = int(expected_height_cm)
        self.vision_control("save_rgbd_diagnostic", **payload)

    def clear_validation_result(self) -> None:
        self.vision_control("clear_validation_result")

    def validate_mount_after_next_respawn(self) -> None:
        self.vision_control("validate_mount_after_next_respawn")

    def show_camera_view(self, **payload: Any) -> None:
        self.vision_control("open_camera_viewport", **payload)

    def open_camera_viewport(self, **payload: Any) -> None:
        self.vision_control("open_camera_viewport", **payload)

    def return_main_view_to_perspective(self, **payload: Any) -> None:
        self.vision_control("return_main_view_to_perspective", **payload)

    def close_camera_viewport(self, **payload: Any) -> None:
        self.vision_control("close_camera_viewport", **payload)

    def restore_camera_view(self, **payload: Any) -> None:
        self.vision_control("restore_camera_view", **payload)

    def set_vision_source_mode(self, source_mode: str) -> None:
        self.vision_control("set_source_mode", source_mode=str(source_mode))

    def validate_camera_geometry(self) -> None:
        self.vision_control("validate_camera_geometry")

    def restart(self) -> None:
        self.shutdown(timeout_s=3.0)
        self.start()

    def restart_without_camera(self) -> None:
        setattr(self.args, "onboard_camera", False)
        setattr(self.args, "enable_cameras", False)
        self.restart()

    def run_preflight(self) -> dict[str, Any]:
        result = run_launch_preflight_for_args(self.args)
        self.latest_status.update({"preflight": result, "phase": "preflight_completed"})
        return result

    def open_log_folder(self) -> str:
        folder = str(LOG_DIR)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        return folder

    def copy_display_command(self) -> str:
        if self.launch_plan is not None:
            return self.launch_plan.display_command
        return str(self.latest_status.get("display_command", "") or "")

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        try:
            self._send_or_queue(make_message("shutdown"))
            self._flush_pending()
        except Exception:
            pass
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=max(0.1, float(timeout_s)))
            except subprocess.TimeoutExpired:
                self._terminate_process_tree()
        self.close()

    def close(self) -> None:
        for sock in (self.conn, self.server):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        self.conn = None
        self.server = None
        for handle in (self._stdout_handle, self._stderr_handle):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        self._stdout_handle = None
        self._stderr_handle = None
        wrapper = self.launch_plan.wrapper_path if self.launch_plan is not None else ""
        if wrapper:
            try:
                Path(wrapper).unlink(missing_ok=True)
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        self._update_log_activity()
        status = dict(self.latest_status)
        now = time.monotonic()
        stdout_tail = tail_file(self.stdout_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        stderr_tail = tail_file(self.stderr_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        error_category = str(status.get("error_category", "") or classify_startup_error("\n".join(stderr_tail + stdout_tail)))
        status.update(
            {
                "worker": True,
                "worker_pid": self.pid,
                "worker_returncode": self.returncode,
                "startup_elapsed_s": max(0.0, now - self.start_time) if self.start_time else 0.0,
                "connected": self.connected,
                "ipc_connected": self.connected,
                "stdout_path": str(self.stdout_path) if self.stdout_path else str(status.get("stdout_path", "") or ""),
                "stderr_path": str(self.stderr_path) if self.stderr_path else str(status.get("stderr_path", "") or ""),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "last_meaningful_stdout": last_meaningful_line(stdout_tail),
                "last_meaningful_stderr": last_meaningful_line(stderr_tail),
                "last_log_activity_at": float(self.last_log_activity_at),
                "startup_progress_message": self._startup_progress_message(),
                "error_category": error_category,
                "startup_diagnosis": diagnose_startup_phase(
                    phase=str(status.get("phase", "")),
                    connected=self.connected,
                    ready=bool(status.get("ready", False)),
                    error_category=error_category,
                    error=str(status.get("error", "") or ""),
                    effective_headless=bool(status.get("effective_headless", False)),
                ),
            }
        )
        return status

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "hello":
            self.latest_status.update(message)
            self.latest_status.setdefault("phase", "ipc_connected")
        elif message_type == "status":
            self.latest_status.update(message)
            self.latest_status["error"] = str(message.get("error", "") or "")
            self.last_status_time = time.monotonic()
        elif message_type == "error":
            self.latest_status.update(message)
            self.latest_status["ready"] = False
            self.latest_status["phase"] = message.get("phase", self.latest_status.get("phase", "error"))
            self.latest_status["error"] = str(message.get("error", "worker error"))
            self.latest_status["error_category"] = classify_startup_error(
                self.latest_status["error"] + "\n" + str(message.get("traceback", "") or "")
            )
            self.last_status_time = time.monotonic()
        elif message_type == "log":
            tail = list(self.latest_status.get("latest_worker_log_tail", []) or [])
            tail.append(str(message.get("text", "")))
            self.latest_status["latest_worker_log_tail"] = tail[-int(getattr(self.args, "sim_worker_log_lines", 200)) :]
        self.latest_status["worker_pid"] = self.pid
        self.latest_status["worker_returncode"] = self.returncode

    def _send_or_queue(self, message: dict[str, Any]) -> None:
        if self.conn is None:
            self.pending_messages.append(message)
            return
        self.conn.sendall(encode_message(message))

    def _flush_pending(self) -> None:
        if self.conn is None:
            return
        pending = list(self.pending_messages)
        self.pending_messages.clear()
        for message in pending:
            self.conn.sendall(encode_message(message))

    def _apply_timeouts(self) -> None:
        if not self.start_time or self.latest_status.get("ready", False):
            return
        now = time.monotonic()
        startup_timeout = float(getattr(self.args, "sim_startup_timeout_s", 600.0))
        status_timeout = float(getattr(self.args, "sim_worker_status_timeout_s", 10.0))
        recent_log = self.last_log_activity_monotonic and (now - self.last_log_activity_monotonic) < min(60.0, startup_timeout)
        if now - self.start_time > startup_timeout and not recent_log and not self.latest_status.get("startup_timeout_warning"):
            self.latest_status["startup_timeout_warning"] = (
                f"Isaac worker did not become ready within {startup_timeout:g}s and worker logs stopped changing. Check worker log."
            )
            self.latest_status["error_category"] = "startup_timeout"
        elif now - self.start_time > startup_timeout and recent_log:
            self.latest_status["startup_progress_message"] = (
                "Isaac worker is still producing logs; first extension download or shader compilation may still be progressing."
            )
        if now - self.last_status_time > status_timeout and self.connected:
            self.latest_status["status_timeout_warning"] = (
                f"No Isaac worker status received for {status_timeout:g}s."
            )

    def _detect_log_startup_failure(self) -> None:
        if self.connected or self.latest_status.get("ready", False) or self.process is None:
            return
        stdout_tail = tail_file(self.stdout_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        stderr_tail = tail_file(self.stderr_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        combined = "\n".join(stderr_tail + stdout_tail)
        category = classify_startup_error(combined)
        if category == "eula_required":
            self.latest_status.update(
                {
                    "ready": False,
                    "starting": False,
                    "phase": "eula_required",
                    "error_category": "eula_required",
                    "error": (
                        "Isaac Sim requested EULA input. Background worker stdin is disabled; "
                        "run Isaac Sim once in the foreground or pass --accept-isaac-eula only if you explicitly agree."
                    ),
                }
            )
            self._terminate_process_tree()
        elif category == "batch_parse_error":
            self.latest_status.update(
                {
                    "ready": False,
                    "starting": False,
                    "phase": "batch_parse_error",
                    "error_category": "batch_parse_error",
                    "error": last_meaningful_line(stderr_tail + stdout_tail) or 'Windows batch parser reported: "" was unexpected at this time.',
                }
            )

    def _mark_exited_before_ipc(self) -> None:
        if self.connected and self.latest_status.get("phase") not in {"process_spawned", "waiting_for_ipc"}:
            return
        stdout_tail = tail_file(self.stdout_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        stderr_tail = tail_file(self.stderr_path, int(getattr(self.args, "sim_worker_log_lines", 200)))
        combined = "\n".join(stderr_tail + stdout_tail)
        category = classify_startup_error(combined)
        last_line = last_meaningful_line(stderr_tail) or last_meaningful_line(stdout_tail)
        self.latest_status.update(
            {
                "ready": False,
                "starting": False,
                "phase": "exited_before_ipc" if not self.connected else "exited",
                "error_category": category,
                "error": (
                    f"Isaac worker exited before IPC connection (returncode {self.returncode}, category={category}). "
                    f"{last_line}".strip()
                ),
                "traceback": combined,
                "last_meaningful_stdout": last_meaningful_line(stdout_tail),
                "last_meaningful_stderr": last_meaningful_line(stderr_tail),
            }
        )

    def _update_log_activity(self) -> None:
        for path in (self.stdout_path, self.stderr_path):
            if path is None:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            key = str(path)
            if size != self._last_log_sizes.get(key):
                self._last_log_sizes[key] = size
                self.last_log_activity_at = time.time()
                self.last_log_activity_monotonic = time.monotonic()

    def _startup_progress_message(self) -> str:
        if not self.start_time:
            return ""
        if self.last_log_activity_monotonic and time.monotonic() - self.last_log_activity_monotonic < 10.0:
            return "Worker logs are still changing."
        return str(self.latest_status.get("startup_progress_message", "") or "")

    def _reset_socket(self) -> None:
        for sock in (self.conn, self.server):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        self.conn = None
        self.server = None

    def _terminate_process_tree(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        pid = self.process.pid
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                self.process.wait(timeout=3.0)
                return
            except Exception:
                pass
        try:
            self.process.terminate()
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)
        except Exception:
            pass


def _append_worker_config_as_cli(command: list[str], config: dict[str, Any]) -> None:
    flag_map = {
        "ipc_host": "--ipc-host",
        "ipc_port": "--ipc-port",
        "height_cm": "--height-cm",
        "robot_usd": "--robot-usd",
        "save_usd": "--save-usd",
        "spawn_z": "--spawn-z",
        "obstacle_x": "--obstacle-x",
        "obstacle_width": "--obstacle-width",
        "obstacle_length": "--obstacle-length",
        "infer_obstacle_size": "--infer-obstacle-size",
        "robot_width": "--robot-width",
        "robot_length": "--robot-length",
        "physics_dt": "--physics-dt",
        "render_interval": "--render-interval",
        "wheel_direction": "--wheel-direction",
        "max_wheel_speed_rad_s": "--max-wheel-speed-rad-s",
        "default_wheel_speed_rad_s": "--default-wheel-speed-rad-s",
        "servo_stiffness": "--servo-stiffness",
        "servo_damping": "--servo-damping",
        "wheel_damping": "--wheel-damping",
        "device": "--device",
        "sim_status_refresh_ms": "--sim-status-refresh-ms",
        "sim_worker_log_lines": "--sim-worker-log-lines",
        "camera_parent_prim": "--camera-parent-prim",
        "camera_width": "--camera-width",
        "camera_height": "--camera-height",
        "camera_update_period_s": "--camera-update-period-s",
        "camera_offset_x": "--camera-offset-x",
        "camera_offset_y": "--camera-offset-y",
        "camera_offset_z": "--camera-offset-z",
        "camera_pitch_deg": "--camera-pitch-deg",
        "camera_aim_mode": "--camera-aim-mode",
        "camera_target_x": "--camera-target-x",
        "camera_target_y": "--camera-target-y",
        "camera_target_z": "--camera-target-z",
        "camera_target_frame": "--camera-target-frame",
        "camera_look_at_roll_deg": "--camera-look-at-roll-deg",
        "camera_focal_length": "--camera-focal-length",
        "camera_horizontal_aperture": "--camera-horizontal-aperture",
        "camera_near_clip_m": "--camera-near-clip-m",
        "camera_far_clip_m": "--camera-far-clip-m",
        "vision_confidence_threshold": "--vision-confidence-threshold",
        "vision_stable_frames": "--vision-stable-frames",
        "vision_window_size": "--vision-window-size",
        "vision_height_tolerance_cm": "--vision-height-tolerance-cm",
        "worker_smoke_camera_height_cm": "--worker-smoke-camera-height-cm",
        "worker_smoke_camera_validation_s": "--worker-smoke-camera-validation-s",
        "worker_smoke_camera_output": "--worker-smoke-camera-output",
        "worker_smoke_camera_counterfactual_output": "--worker-smoke-camera-counterfactual-output",
        "worker_smoke_output": "--worker-smoke-output",
        "camera_view_pending_timeout_s": "--camera-view-pending-timeout-s",
        "camera_view_pending_max_retries": "--camera-view-pending-max-retries",
        "robot_ground_settle_s": "--robot-ground-settle-s",
        "robot_ground_settle_max_steps": "--robot-ground-settle-max-steps",
        "robot_ground_stable_frames": "--robot-ground-stable-frames",
        "robot_ground_vertical_speed_threshold_m_s": "--robot-ground-vertical-speed-threshold-m-s",
        "robot_ground_joint_speed_threshold_rad_s": "--robot-ground-joint-speed-threshold-rad-s",
        "robot_ground_servo_speed_threshold_rad_s": "--robot-ground-servo-speed-threshold-rad-s",
        "robot_ground_wheel_speed_threshold_rad_s": "--robot-ground-wheel-speed-threshold-rad-s",
        "robot_ground_clearance_m": "--robot-ground-clearance-m",
        "robot_ground_penetration_tolerance_m": "--robot-ground-penetration-tolerance-m",
        "robot_max_ground_correction_m": "--robot-max-ground-correction-m",
        "livestream": "--livestream",
        "experience": "--experience",
        "telemetry_rate": "--telemetry-rate",
        "visualized_env_id": "--visualized-env-id",
        "output_dir": "--output-dir",
        "telemetry_config": "--telemetry-config",
    }
    for key, flag in flag_map.items():
        value = config.get(key)
        if value is None or value == "":
            continue
        command.extend([flag, str(value)])
    _append_bool(command, bool(config.get("save_scene", True)), "--save-scene", "--no-save-scene")
    _append_bool(command, bool(config.get("onboard_camera", True)), "--onboard-camera", "--no-onboard-camera")
    if bool(config.get("enable_cameras", False)):
        command.append("--enable_cameras")
    if bool(config.get("headless", False)):
        command.append("--headless")
    _append_bool(command, bool(config.get("apply_safe_servo_joint_limits", True)), "--apply-safe-servo-joint-limits", "--no-apply-safe-servo-joint-limits")
    if bool(config.get("apply_physx_joint_limits", False)):
        command.append("--apply-physx-joint-limits")
    if bool(config.get("no_continuous_sim_step", False)):
        command.append("--no-continuous-sim-step")
    if bool(config.get("worker_smoke_negative_knee_test", False)):
        command.append("--worker-smoke-negative-knee-test")
    if bool(config.get("worker_smoke_camera_detection", False)):
        command.append("--worker-smoke-camera-detection")
    if bool(config.get("worker_smoke_camera_provenance", False)):
        command.append("--worker-smoke-camera-provenance")
    if bool(config.get("worker_smoke_camera_pose_ab", False)):
        command.append("--worker-smoke-camera-pose-ab")
    if bool(config.get("worker_smoke_camera_counterfactual", False)):
        command.append("--worker-smoke-camera-counterfactual")
    if bool(config.get("worker_smoke_camera_view_ground_contact", False)):
        command.append("--worker-smoke-camera-view-ground-contact")
    if bool(config.get("worker_smoke_ground_structure", False)):
        command.append("--worker-smoke-ground-structure")
    if bool(config.get("worker_smoke_ground_calibration", False)):
        command.append("--worker-smoke-ground-calibration")
    if bool(config.get("worker_smoke_vision_playback", False)):
        command.append("--worker-smoke-vision-playback")
    if bool(config.get("viewport_physics_guard", True)):
        command.append("--viewport-physics-guard")
    else:
        command.append("--no-viewport-physics-guard")
    if bool(config.get("defer_first_visible_render", True)):
        command.append("--defer-first-visible-render")
    else:
        command.append("--no-defer-first-visible-render")
    if bool(config.get("camera_view_active_fallback", False)):
        command.append("--camera-view-active-fallback")
    if bool(config.get("robot_auto_ground_correction", False)):
        command.append("--robot-auto-ground-correction")
    if bool(config.get("camera_coverage_strict", False)):
        command.append("--camera-coverage-strict")
    _append_bool(command, bool(config.get("telemetry_enabled", False)), "--telemetry", "--no-telemetry")
    _append_bool(command, bool(config.get("live_viz_enabled", True)), "--live-viz", "--no-live-viz")
    _append_bool(command, bool(config.get("telemetry_report", True)), "--report", "--no-report")
    _append_bool(command, bool(config.get("equilibrium_region_enabled", True)), "--equilibrium-region", "--no-equilibrium-region")
    if float(config.get("worker_smoke_test_s", 0.0) or 0.0) > 0.0:
        command.extend(["--worker-smoke-test-s", str(float(config["worker_smoke_test_s"]))])


def _append_bool(command: list[str], value: bool, true_flag: str, false_flag: str | None = None) -> None:
    if value:
        command.append(true_flag)
    elif false_flag:
        command.append(false_flag)


def _run_candidate_preflight(
    candidate_mode: str,
    candidate_path: str | Path,
    *,
    args: Any,
    env: dict[str, str],
    timeout_s: float,
    enabled: bool,
) -> IsaacInterpreterReport:
    if not enabled:
        return IsaacInterpreterReport(
            executable=str(candidate_path),
            python_version=sys.version.replace("\n", " "),
            isaacsim_importable=True,
            isaaclab_importable=True,
            app_launcher_importable=True,
            isaacsim_version="preflight-skipped",
            isaaclab_version="preflight-skipped",
            compatible_python=True,
            environment=inspect_launcher_environment(env),
        )
    if candidate_mode == "isaaclab-bat":
        return run_isaaclab_bat_preflight(
            candidate_path,
            timeout_s=timeout_s,
            env=env,
            isaaclab_root=Path(str(getattr(args, "isaaclab_bat", DEFAULT_ISAACLAB_BAT))).parent,
        )
    return run_interpreter_preflight(
        candidate_path,
        timeout_s=timeout_s,
        env=env,
        isaaclab_root=Path(str(getattr(args, "isaaclab_bat", DEFAULT_ISAACLAB_BAT))).parent,
    )


def _report_is_launchable(report: IsaacInterpreterReport) -> bool:
    return (
        bool(report.compatible_python)
        and bool(report.isaacsim_importable)
        and bool(report.isaaclab_importable)
        and bool(report.app_launcher_importable)
        and report.error_category not in {"eula_required"}
    )


def _summarize_candidate_failures(reports: list[dict[str, Any]]) -> str:
    if not reports:
        return "No Isaac Python candidates were available."
    parts: list[str] = []
    for report in reports:
        candidate = f"{report.get('candidate_launch_mode')}: {report.get('candidate_path')}"
        error = format_preflight_error(report) or str(report.get("error", "") or report.get("error_category", "") or "not launchable")
        parts.append(f"{candidate} -> {error}")
    return "No launchable Isaac interpreter found. " + " | ".join(parts)


def _display_command(argv: list[str]) -> str:
    try:
        return subprocess.list2cmdline([str(item) for item in argv])
    except Exception:
        return " ".join(str(item) for item in argv)
