"""Small runtime helpers for the sole subprocess Isaac worker.

Task-specific camera, stability, report, and periodic telemetry services are
intentionally absent from the formal UI path.
"""

from __future__ import annotations

import time
from typing import Any

from command_model import SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from height_manifest import obstacle_height_m, obstacle_height_m_mm, normalize_height_mm
from robot_ground_diagnostics import (
    GROUND_OK,
    VISUAL_ONLY_INTERSECTION,
    default_robot_ground_diagnostics,
    ground_state_from_diagnostics,
    motion_status_from_worker_status,
    respawn_status_from_worker_status,
)
from sim_obstacle_scene import measure_obstacle_geometry, update_obstacle_height
from sim_robot_adapter import SimRobotAdapterConfig


RESPAWN_GROUND_FAILURE_TEXT = (
    "grounded respawn reference is invalid; robot stopped. "
    "Use Respawn And Validate Ground after fixing the ground contact issue."
)


def create_adapter_config_from_args(args: Any) -> SimRobotAdapterConfig:
    servo_threshold = getattr(args, "robot_ground_servo_speed_threshold_rad_s", None)
    if servo_threshold is None:
        servo_threshold = getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)
    return SimRobotAdapterConfig(
        max_wheel_speed=float(getattr(args, "max_wheel_speed_rad_s", getattr(args, "max_wheel_speed", 0.0))),
        default_wheel_speed=float(getattr(args, "default_wheel_speed_rad_s", getattr(args, "default_wheel_speed", 0.0))),
        wheel_direction=float(getattr(args, "wheel_direction", 1.0)),
        apply_safe_servo_joint_limits=bool(getattr(args, "apply_safe_servo_joint_limits", True)),
        apply_joint_limits_to_sim=bool(getattr(args, "apply_physx_joint_limits", True)),
        ground_settle_s=float(getattr(args, "robot_ground_settle_s", 0.75)),
        ground_settle_max_steps=int(getattr(args, "robot_ground_settle_max_steps", 180)),
        ground_stable_frames=int(getattr(args, "robot_ground_stable_frames", 10)),
        ground_vertical_speed_threshold_m_s=float(getattr(args, "robot_ground_vertical_speed_threshold_m_s", 0.01)),
        ground_joint_speed_threshold_rad_s=float(getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)),
        ground_servo_speed_threshold_rad_s=float(servo_threshold),
        ground_wheel_speed_threshold_rad_s=float(getattr(args, "robot_ground_wheel_speed_threshold_rad_s", 0.20)),
        ground_clearance_m=float(getattr(args, "robot_ground_clearance_m", 0.002)),
        ground_penetration_tolerance_m=float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)),
        auto_ground_correction=bool(getattr(args, "robot_auto_ground_correction", False)),
        max_ground_correction_m=float(getattr(args, "robot_max_ground_correction_m", 0.10)),
    )


def initialize_adapter_ground_reference(
    adapter: Any,
    *,
    tick_observer: Any | None = None,
) -> dict[str, Any]:
    if adapter is None or not hasattr(adapter, "initialize_grounded_respawn_reference"):
        return {"grounded_reference_valid": False, "error": "adapter does not support grounded reference initialization"}
    if tick_observer is None:
        return dict(adapter.initialize_grounded_respawn_reference() or {})
    return dict(
        adapter.initialize_grounded_respawn_reference(
            tick_observer=tick_observer
        )
        or {}
    )


def handle_set_height(
    *,
    adapter: Any,
    scene_handle: Any,
    height_cm: int | None = None,
    height_mm: int | None = None,
    source: str = "ui",
    request_id: str = "",
    obstacle_revision: int = 0,
    respawn_policy: str = "never",
) -> dict[str, Any]:
    resolved_mm = normalize_height_mm(height_mm if height_mm is not None else int(height_cm or 0) * 10)
    update_started = time.perf_counter()
    try:
        transaction = update_obstacle_height(scene_handle, obstacle_height_m_mm(resolved_mm))
        physics_dt = float(scene_handle.sim.get_physics_dt())
        for _tick in range(2):
            adapter.step(physics_dt)
        if hasattr(scene_handle.sim, "render"):
            scene_handle.sim.render()
        measured = measure_obstacle_geometry(scene_handle)
        update_error = str(measured.get("error", "") or "")
    except Exception as exc:
        transaction = {"old_geometry": measure_obstacle_geometry(scene_handle), "update_mode": "error"}
        measured = measure_obstacle_geometry(scene_handle)
        update_error = str(exc)
    obstacle_update_s = time.perf_counter() - update_started
    old_geometry = dict(transaction.get("old_geometry", {}) or {})
    measured_height_mm = float(measured.get("height_m", -1.0) or -1.0) * 1000.0
    requested_height_m = obstacle_height_m_mm(resolved_mm)
    visual_updated = bool(measured.get("visual_valid", False)) and abs(float(measured.get("height_m", -1.0)) - requested_height_m) <= 0.001
    collision_updated = bool(measured.get("collision_valid", False)) and abs(float(measured.get("collision_height_m", -1.0)) - requested_height_m) <= 0.001
    geometry_ok = (
        bool(measured.get("prim_valid", False))
        and visual_updated
        and collision_updated
        and abs(measured_height_mm - float(resolved_mm)) <= 1.0
        and not update_error
    )
    policy = str(respawn_policy or "required").lower()
    if policy not in {"required", "if_motion_ready", "never"}:
        policy = "required"
    motion_ready = _adapter_motion_ready(adapter)
    respawn_result: dict[str, Any] = {"ok": True, "respawned": False, "skipped": True}
    warning = ""
    if geometry_ok and (policy == "required" or (policy == "if_motion_ready" and motion_ready)):
        respawn_result = _respawn_with_ground_policy(adapter)
    elif policy == "if_motion_ready":
        warning = "respawn skipped because motion_ready=false"
        respawn_result = {
            "ok": True,
            "respawned": False,
            "skipped": True,
            "warning": warning,
            "ground_diagnostics": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        }
    return {
        "ok": geometry_ok and (bool(respawn_result.get("ok", True)) if policy == "required" else True),
        "accepted": geometry_ok,
        "old_height_mm": None if old_geometry.get("height_m") is None else float(old_geometry["height_m"]) * 1000.0,
        "requested_height_mm": resolved_mm,
        "measured_height_mm": measured_height_mm,
        "height_mm": resolved_mm,
        "height_cm": resolved_mm / 10.0,
        "source": str(source or ""),
        "request_id": str(request_id or ""),
        "obstacle_revision": int(obstacle_revision),
        "obstacle_updated": geometry_ok,
        "visual_updated": visual_updated,
        "collision_updated": collision_updated,
        "prim_path": str(measured.get("prim_path", "/World/Obstacle")),
        "prim_valid": bool(measured.get("prim_valid", False)),
        "measured_bounds": dict(measured.get("measured_bounds", {}) or {}),
        "visual_bounds": dict(measured.get("visual_bounds", {}) or {}),
        "collision_bounds": dict(measured.get("collision_bounds", {}) or {}),
        "measured_width_m": measured.get("width_m"),
        "measured_length_m": measured.get("length_m"),
        "front_face_x_m": measured.get("front_face_x_m"),
        "center_y_m": measured.get("center_y_m"),
        "bottom_z_m": measured.get("bottom_z_m"),
        "top_z_m": measured.get("top_z_m"),
        "update_mode": str(transaction.get("update_mode", "")),
        "obstacle_update_s": obstacle_update_s,
        "scene_height_mm": resolved_mm,
        "scene_height_cm": resolved_mm / 10.0,
        "scene_ready": geometry_ok,
        "respawn_policy": policy,
        "respawn_requested": policy != "never",
        "respawned": bool(respawn_result.get("respawned", False)),
        "motion_ready": bool(motion_ready),
        "respawn_warning": warning or str(respawn_result.get("warning", "")),
        "respawn_result": respawn_result,
        "control_ready": bool(geometry_ok and _adapter_motion_ready(adapter)),
        "error": update_error or (str(respawn_result.get("error", "")) if policy == "required" else ""),
    }


def handle_respawn(*, adapter: Any) -> dict[str, Any]:
    return _respawn_with_ground_policy(adapter)


def build_common_worker_status(
    *,
    args: Any,
    adapter: Any,
    scene_handle: Any | None = None,
    detailed: bool = False,
) -> dict[str, Any]:
    if not detailed:
        return _build_light_worker_status(args=args, adapter=adapter, scene_handle=scene_handle)
    if adapter is None:
        catalog = _joint_catalog_from_adapter(None)
        return {
            "command_state": None,
            "target_joint_state": None,
            "actual_joint_state": None,
            "joint_diagnostics": [],
            "joint_catalog": catalog,
            "robot_joint_names": [row["joint_name"] for row in catalog],
            "robot_ground": default_robot_ground_diagnostics("adapter unavailable"),
            "grounded_reference_valid": False,
            "grounded_reference_physics_valid": False,
            "grounded_reference_visual_valid": False,
            "grounded_reference_stable": False,
            "respawn_ready": False,
            "respawn_block_reason": "adapter unavailable",
            "ground_reference_block_reason": "adapter unavailable",
            "grounded_reference_diagnostics": default_robot_ground_diagnostics("adapter unavailable"),
            "last_ground_settle_result": {},
            "last_ground_correction_z_m": 0.0,
            "negative_knee_smoke_result": None,
            "physics_dt": float(getattr(args, "physics_dt", 0.0) or 0.0),
            "sim_step_hz": 0.0,
            "current_max_wheel_speed_rad_s": float(getattr(args, "max_wheel_speed_rad_s", 0.0) or 0.0),
            "default_wheel_speed_rad_s": float(getattr(args, "default_wheel_speed_rad_s", 0.0) or 0.0),
            "adapter": "",
            "telemetry": {"enabled": False, "run_dir": ""},
            "wheel_command": {},
            "scene_baseline": {},
        }
    physics_dt = _safe_physics_dt(adapter, args)
    catalog = _joint_catalog_from_adapter(adapter)
    return {
        "command_state": _safe_capture_command_state(adapter),
        "sim_state": adapter.capture_sim_state() if hasattr(adapter, "capture_sim_state") else {"command_state": _safe_capture_command_state(adapter)},
        "target_joint_state": adapter.get_target_joint_state() if hasattr(adapter, "get_target_joint_state") else {},
        "actual_joint_state": adapter.get_actual_joint_state() if hasattr(adapter, "get_actual_joint_state") else {},
        "joint_diagnostics": adapter.get_joint_diagnostics() if hasattr(adapter, "get_joint_diagnostics") else [],
        "joint_catalog": catalog,
        "robot_joint_names": [row["joint_name"] for row in catalog],
        "robot_ground": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
        "grounded_reference_physics_valid": bool(getattr(adapter, "grounded_reference_physics_valid", False)),
        "grounded_reference_visual_valid": bool(getattr(adapter, "grounded_reference_visual_valid", False)),
        "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
        "respawn_ready": bool(getattr(adapter, "respawn_ready", False)),
        "ground_reference_block_reason": str(getattr(adapter, "ground_reference_block_reason", "") or ""),
        "grounded_reference_diagnostics": dict(getattr(adapter, "grounded_reference_diagnostics", default_robot_ground_diagnostics("not checked"))),
        "last_ground_settle_result": dict(getattr(adapter, "last_ground_settle_result", {}) or {}),
        "last_ground_correction_z_m": float(getattr(adapter, "last_ground_correction_z_m", 0.0) or 0.0),
        "negative_knee_smoke_result": getattr(adapter, "negative_knee_smoke_result", None),
        "physics_dt": physics_dt,
        "sim_step_hz": 1.0 / physics_dt if physics_dt > 0 else 0.0,
        "current_max_wheel_speed_rad_s": float(getattr(adapter, "max_wheel_speed", 0.0) or 0.0),
        "default_wheel_speed_rad_s": float(getattr(adapter, "default_wheel_speed", 0.0) or 0.0),
        "adapter": type(adapter).__name__,
        "telemetry": _safe_telemetry_status(adapter),
        "wheel_command": dict(getattr(adapter, "wheel_command_status", {}) or {}),
        "scene_baseline": dict(getattr(adapter, "scene_baseline_metrics", {}) or {}),
    }


def _build_light_worker_status(*, args: Any, adapter: Any, scene_handle: Any | None = None) -> dict[str, Any]:
    """Small, allocation-bounded heartbeat; detailed state is explicit only."""

    if adapter is None:
        return {
            "command_state": {"servos": {}, "wheels": {}},
            "servo_target_deg": [],
            "servo_actual_deg": [],
            "wheel_canonical_rad_s": [],
            "wheel_effective_rad_s": [],
            "wheel_measured_rad_s": [],
            "motion_ready": False,
            "respawn_ready": False,
            "control_ready": False,
            "ground_state": "UNVERIFIED",
            "wheel_command": {},
            "motion_profile": {},
            "motion_batch": {},
            "physics_dt": float(getattr(args, "physics_dt", 0.0) or 0.0),
        }
    command_state = _safe_capture_command_state(adapter)
    target_state = adapter.get_target_joint_state() if hasattr(adapter, "get_target_joint_state") else {}
    actual_state = adapter.get_actual_joint_state() if hasattr(adapter, "get_actual_joint_state") else {}
    target_servos = dict(target_state.get("servos", {}) or {})
    actual_servos = dict(actual_state.get("servos", {}) or {})
    target_wheels = dict(target_state.get("wheels", {}) or {})
    actual_wheels = dict(actual_state.get("wheels", {}) or {})
    ground = dict(getattr(adapter, "robot_ground_diagnostics", {}) or {})
    ground_compact = {
        key: ground.get(key)
        for key in (
            "classification",
            "checked",
            "physical_ground_safe",
            "minimum_clearance_m",
            "maximum_collision_penetration_m",
            "root_z_m",
        )
        if key in ground
    }
    readiness = {
        "ready": True,
        "runtime_ready": True,
        "robot_ground": ground_compact,
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
        "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
    }
    motion_ready, motion_reason, ground_state = motion_status_from_worker_status(readiness)
    respawn_ready, respawn_reason = respawn_status_from_worker_status(readiness)
    motion_reference = getattr(adapter, "motion_reference", None)
    motion_profile = {
        "mode": "fixed_100_percent",
        "profile_id": str(getattr(motion_reference, "profile_id", "")),
        "servo_velocity_deg_s": float(getattr(motion_reference, "servo_reference_velocity_deg_s", 0.0) or 0.0),
        "wheel_reference_velocity_rad_s": float(getattr(motion_reference, "wheel_reference_velocity_rad_s", 0.0) or 0.0),
        "wheel_velocity_limit_rad_s": float(getattr(motion_reference, "wheel_velocity_limit_rad_s", 0.0) or 0.0),
    }
    return {
        "command_state": command_state,
        "servo_target_deg": [float(command_state.get("servos", {}).get(name, 0.0)) for name in SERVO_JOINT_NAMES],
        "servo_actual_deg": [_row_float(actual_servos.get(name), "deg") for name in SERVO_JOINT_NAMES],
        "wheel_canonical_rad_s": [float(command_state.get("wheels", {}).get(name, 0.0)) for name in WHEEL_JOINT_NAMES],
        "wheel_effective_rad_s": [_row_float(target_wheels.get(name), "target_rad_s") for name in WHEEL_JOINT_NAMES],
        "wheel_measured_rad_s": [_row_float(actual_wheels.get(name), "rad_s") for name in WHEEL_JOINT_NAMES],
        # Compatibility views are compact and contain only the 12 commandable joints.
        "target_joint_state": {
            "servos": target_servos,
            "wheels": target_wheels,
        },
        "actual_joint_state": {
            "servos": actual_servos,
            "wheels": actual_wheels,
        },
        "robot_ground": ground_compact,
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
        "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
        "motion_ready": bool(motion_ready),
        "motion_block_reason": str(motion_reason or ""),
        "respawn_ready": bool(respawn_ready),
        "respawn_block_reason": str(respawn_reason or ""),
        "control_ready": bool(motion_ready),
        "ground_state": ground_state,
        "wheel_command": dict(getattr(adapter, "wheel_command_status", {}) or {}),
        "motion_profile": motion_profile,
        "motion_batch": dict(getattr(adapter, "motion_batch_status", {}) or {}),
        "physics_dt": _safe_physics_dt(adapter, args),
        "sim_step_hz": 1.0 / _safe_physics_dt(adapter, args) if _safe_physics_dt(adapter, args) > 0.0 else 0.0,
    }


def _row_float(row: Any, key: str) -> float | None:
    value = row.get(key) if isinstance(row, dict) else None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _joint_catalog_from_adapter(adapter: Any | None) -> list[dict[str, Any]]:
    robot = getattr(adapter, "robot", None)
    articulation_names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
    diagnostics = getattr(adapter, "get_joint_diagnostics", lambda: [])()
    diagnostic_by_name = {str(row.get("joint_name", "")): dict(row) for row in diagnostics if isinstance(row, dict) and str(row.get("joint_name", ""))}
    servo_ids = dict(getattr(adapter, "servo_name_to_id", {}) or {})
    wheel_ids = dict(getattr(adapter, "wheel_name_to_id", {}) or {})
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, kind: str, commandable: bool) -> None:
        joint_id = servo_ids.get(name, wheel_ids.get(name))
        diagnostic = diagnostic_by_name.get(name, {})
        if joint_id is None and name in articulation_names:
            joint_id = articulation_names.index(name)
        rows.append({"joint_name": name, "kind": kind, "commandable": bool(commandable), "available": bool(name in articulation_names or name in diagnostic_by_name or joint_id is not None), "joint_id": None if joint_id is None else int(joint_id), "source": "worker_articulation", "diagnostic": diagnostic})
        seen.add(name)

    for name in SERVO_JOINT_NAMES:
        add(name, "servo", True)
    for name in WHEEL_JOINT_NAMES:
        add(name, "wheel", True)
    for name in articulation_names:
        if name not in seen:
            add(name, "articulation", False)
    return rows


def enrich_runtime_readiness(status: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(status)
    runtime_ready = bool(enriched.get("runtime_ready", enriched.get("ready", False)))
    if str(enriched.get("phase", "") or "") == "running" and not str(enriched.get("traceback", "") or ""):
        runtime_ready = True
    enriched["runtime_ready"] = runtime_ready
    enriched["ready"] = runtime_ready
    enriched.setdefault("ready_source", "runtime")
    enriched.setdefault("last_ready_change_reason", "runtime status")
    enriched.setdefault("last_ready_change_message_type", "status")
    motion_ready, motion_block_reason, ground_state = motion_status_from_worker_status(enriched)
    respawn_ready, respawn_block_reason = respawn_status_from_worker_status(enriched)
    enriched["ground_state"] = ground_state
    ground = dict(enriched.get("robot_ground", {}) if isinstance(enriched.get("robot_ground"), dict) else {})
    ground["ground_state"] = ground_state_from_diagnostics(ground)
    ground.setdefault("physical_ground_safe", bool(motion_ready))
    ground.setdefault("respawn_ready", bool(respawn_ready))
    enriched["robot_ground"] = ground
    enriched["motion_ready"] = bool(motion_ready)
    enriched["motion_block_reason"] = str(motion_block_reason or "")
    enriched["respawn_ready"] = bool(respawn_ready)
    enriched["respawn_block_reason"] = str(respawn_block_reason or "")
    return enriched


def ground_reference_result_is_valid(result: dict[str, Any]) -> bool:
    diagnostics = result.get("grounded_reference_diagnostics", result.get("ground_diagnostics", {}))
    classification = str(diagnostics.get("classification", ""))
    physical_ground_safe = bool(diagnostics.get("physical_ground_safe", classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}))
    return bool(result.get("grounded_reference_valid", False) and result.get("grounded_reference_physics_valid", False) and result.get("grounded_reference_stable", False) and diagnostics.get("checked", False) and physical_ground_safe and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION})


def _respawn_with_ground_policy(adapter: Any) -> dict[str, Any]:
    if adapter is None or not hasattr(adapter, "respawn_robot"):
        return {"ok": False, "respawned": False, "error": "adapter does not support respawn_robot"}
    status = _ground_status(adapter)
    respawn_ready, _ = respawn_status_from_worker_status(status)
    if not respawn_ready and hasattr(adapter, "initialize_grounded_respawn_reference"):
        initialize_adapter_ground_reference(adapter)
    respawn_ready, respawn_reason = respawn_status_from_worker_status(_ground_status(adapter))
    if not respawn_ready:
        _stop_wheels_preserving_pose(adapter)
        return {"ok": False, "respawned": False, "error": respawn_reason or RESPAWN_GROUND_FAILURE_TEXT, "ground_diagnostics": dict(getattr(adapter, "grounded_reference_diagnostics", default_robot_ground_diagnostics("invalid grounded reference"))), "settle": dict(getattr(adapter, "last_ground_settle_result", {}) or {})}
    result = dict(adapter.respawn_robot(settle=True) or {})
    settle = dict(result.get("settle", {}) or {})
    settle_ok = bool(
        settle.get("stable") is True
        and settle.get("ground_contact_resolved") is True
        and settle.get("acceptance_window_evidence_valid") is True
    )
    explicit_ok = result.get("ok") is True and result.get("respawned") is True
    ok = bool(explicit_ok and settle_ok)
    result["ok"] = ok
    result["respawned"] = ok
    if not ok:
        _stop_wheels_preserving_pose(adapter)
        result["error"] = str(
            result.get("error", "")
            or "respawn settle failed strict grounding checks"
        )
    return result


def _ground_status(adapter: Any) -> dict[str, Any]:
    return {"ready": True, "runtime_ready": True, "robot_ground": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))), "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)), "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False))}


def _adapter_motion_ready(adapter: Any) -> bool:
    if adapter is None:
        return False
    ready, _, _ = motion_status_from_worker_status(_ground_status(adapter))
    return bool(ready)


def _stop_wheels_preserving_pose(adapter: Any) -> None:
    try:
        adapter.stop_wheels()
        if hasattr(adapter, "apply_commands_to_robot"):
            adapter.apply_commands_to_robot()
        writer = getattr(getattr(adapter, "robot", None), "write_data_to_sim", None)
        if callable(writer):
            writer()
    except Exception:
        pass


def _safe_capture_command_state(adapter: Any) -> dict[str, Any]:
    try:
        return dict(adapter.capture_command_state() or {})
    except Exception:
        return {"servos": {}, "wheels": {}}


def _safe_physics_dt(adapter: Any, args: Any) -> float:
    try:
        return float(adapter.sim.get_physics_dt())
    except Exception:
        return float(getattr(args, "physics_dt", 0.0) or 0.0)


def _safe_telemetry_status(adapter: Any) -> dict[str, Any]:
    collector = getattr(adapter, "telemetry_collector", None)
    if collector is None:
        return {"enabled": False, "run_dir": ""}
    try:
        return dict(collector.status())
    except Exception as exc:
        return {"enabled": True, "run_dir": "", "error": str(exc)}
