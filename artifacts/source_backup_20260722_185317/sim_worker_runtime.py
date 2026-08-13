"""Shared runtime helpers for thread and subprocess Isaac workers.

The UI has two worker backends.  Keeping viewport, respawn, and grounded
reference policy here prevents the two loops from drifting in exactly the
places that affect physics state.
"""

from __future__ import annotations

import time
from typing import Any

from height_manifest import obstacle_height_m
from sim_camera_viewport import close_onboard_camera_viewport, default_camera_viewport_status
from sim_obstacle_scene import update_obstacle_height
from sim_robot_adapter import SimRobotAdapterConfig
from robot_ground_diagnostics import (
    GROUND_OK,
    VISUAL_ONLY_INTERSECTION,
    VIEWPORT_ACTIONS,
    can_change_camera_view,
    default_robot_ground_diagnostics,
    detect_fabric_or_viewport_warning,
    ground_state_from_diagnostics,
    merge_guard_into_camera_view_status,
    motion_status_from_worker_status,
    respawn_status_from_worker_status,
    run_viewport_action_with_physics_guard,
)


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
        apply_joint_limits_to_sim=bool(getattr(args, "apply_physx_joint_limits", False)),
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


def initialize_adapter_ground_reference(adapter: Any) -> dict[str, Any]:
    if adapter is None or not hasattr(adapter, "initialize_grounded_respawn_reference"):
        return {"grounded_reference_valid": False, "error": "adapter does not support grounded reference initialization"}
    result = adapter.initialize_grounded_respawn_reference()
    return dict(result or {})


def handle_set_height(
    *,
    adapter: Any,
    scene_handle: Any,
    vision_processor: Any | None,
    height_cm: int,
    source: str = "ui",
    request_id: str = "",
    obstacle_revision: int = 0,
    respawn_policy: str = "required",
) -> dict[str, Any]:
    update_obstacle_height(scene_handle, obstacle_height_m(int(height_cm)))
    if vision_processor is not None:
        vision_processor.reset_filter()
    policy = str(respawn_policy or "required").lower()
    if policy not in {"required", "if_motion_ready", "never"}:
        policy = "required"
    respawn_requested = policy != "never"
    motion_ready = _adapter_motion_ready(adapter)
    respawn_result: dict[str, Any] = {"ok": True, "respawned": False, "skipped": True}
    respawn_warning = ""
    if policy == "required" or (policy == "if_motion_ready" and motion_ready):
        respawn_result = _respawn_with_ground_policy(adapter, vision_processor=vision_processor)
    elif policy == "if_motion_ready":
        respawn_warning = "respawn skipped because motion_ready=false"
        respawn_result = {
            "ok": True,
            "respawned": False,
            "skipped": True,
            "warning": respawn_warning,
            "ground_diagnostics": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        }
    return {
        "ok": bool(respawn_result.get("ok", True)) if policy == "required" else True,
        "height_cm": int(height_cm),
        "source": str(source or ""),
        "request_id": str(request_id or ""),
        "obstacle_revision": int(obstacle_revision),
        "obstacle_updated": True,
        "scene_height_cm": int(height_cm),
        "scene_ready": True,
        "respawn_policy": policy,
        "respawn_requested": bool(respawn_requested),
        "respawned": bool(respawn_result.get("respawned", True)),
        "motion_ready": bool(motion_ready),
        "respawn_warning": respawn_warning or str(respawn_result.get("warning", "")),
        "respawn_result": respawn_result,
        "error": str(respawn_result.get("error", "")) if policy == "required" else "",
    }


def handle_respawn(*, adapter: Any, vision_processor: Any | None = None, reset_filter: bool = True) -> dict[str, Any]:
    if vision_processor is not None and bool(reset_filter):
        vision_processor.reset_filter()
    return _respawn_with_ground_policy(adapter, vision_processor=vision_processor)


def handle_vision_control(
    *,
    args: Any,
    adapter: Any,
    scene_handle: Any,
    vision_processor: Any | None,
    action: str,
    payload: dict[str, Any] | None = None,
    logger_tail_provider: Any | None = None,
    runtime_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_name = str(action or "")
    details = dict(payload or {})
    if action_name == "validate_robot_ground_contact":
        result = adapter.validate_robot_ground_contact(apply_correction=False)
        return {"ok": True, "text": "robot ground contact validated", "ground": result}
    if action_name == "calibrate_ground_reference":
        if hasattr(adapter, "calibrate_grounded_reference"):
            result = adapter.calibrate_grounded_reference()
        else:
            result = initialize_adapter_ground_reference(adapter)
        return {
            "ok": bool(result.get("grounded_reference_valid", False)),
            "text": str(result.get("error", "") or "ground reference calibrated"),
            "ground_calibration": result,
        }
    if action_name == "respawn_validate_ground":
        result = handle_respawn(adapter=adapter, vision_processor=vision_processor, reset_filter=True)
        return {
            "ok": bool(result.get("ok", True)),
            "text": str(result.get("error", "") or "respawned and validated ground"),
            "respawned": bool(result.get("respawned", True)),
            "respawn_result": result,
        }
    if vision_processor is None:
        return {"ok": False, "text": "vision processor unavailable"}
    details.setdefault("headless", bool(getattr(args, "headless", False)))
    details.setdefault("active_fallback_allowed", bool(getattr(args, "camera_view_active_fallback", False)))
    details.setdefault("camera_view_pending_timeout_s", float(getattr(args, "camera_view_pending_timeout_s", 10.0)))
    details.setdefault("camera_view_pending_max_retries", int(getattr(args, "camera_view_pending_max_retries", 30)))
    ok, text, status = execute_viewport_action_with_guard(
        args=args,
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision_processor,
        action=action_name,
        payload=details,
        logger_tail_provider=logger_tail_provider,
        runtime_metrics=runtime_metrics,
        allow_state_restore=False,
    )
    return {"ok": bool(ok), "text": str(text or ""), "camera_view": dict(status or {})}


def execute_viewport_action_with_guard(
    *,
    args: Any,
    adapter: Any,
    scene_handle: Any,
    vision_processor: Any,
    action: str,
    payload: dict[str, Any] | None = None,
    logger_tail_provider: Any | None = None,
    runtime_metrics: dict[str, Any] | None = None,
    allow_state_restore: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    action_name = str(action or "")
    details = dict(payload or {})
    if action_name not in VIEWPORT_ACTIONS or not bool(getattr(args, "viewport_physics_guard", True)):
        ok, text = vision_processor.handle_control(action_name, **details)
        return bool(ok), str(text or ""), dict(getattr(vision_processor, "camera_view_status", {}) or {})

    command_state = _safe_capture_command_state(adapter)
    idle_guard = can_change_camera_view(
        flags=details.get("camera_view_guard_flags", {}),
        wheel_command_values=command_state.get("wheels", {}),
    )
    if action_name != "close_camera_viewport" and not idle_guard.allowed:
        status = _idle_guard_failure_status(action_name, details, idle_guard.reasons)
        vision_processor.camera_view_status = status
        return False, "camera view idle guard failed: " + "; ".join(idle_guard.reasons), status

    before_count, before_tail = _tail_count(logger_tail_provider)

    def _call_viewport_action() -> tuple[bool, str]:
        return vision_processor.handle_control(action_name, **details)

    started_wall = float((runtime_metrics or {}).get("started_wall", time.monotonic()))
    sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
    rtf = sim_time / max(1.0e-9, time.monotonic() - started_wall)
    (ok, text), guard = run_viewport_action_with_physics_guard(
        adapter=adapter,
        scene_handle=scene_handle,
        action=action_name,
        callback=_call_viewport_action,
        enabled=True,
        allow_restore_on_failure=bool(allow_state_restore),
        rtf_before=rtf,
        rtf_after=rtf,
        penetration_tolerance_m=float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)),
    )
    new_lines = _new_tail_lines(logger_tail_provider, before_count, before_tail)
    guard.fabric_warning_detected = detect_fabric_or_viewport_warning(new_lines)
    status = merge_guard_into_camera_view_status(getattr(vision_processor, "camera_view_status", {}), guard)
    status["trigger_stage"] = str(status.get("trigger_stage", "") or action_name)
    if not guard.passed:
        status = _mark_viewport_guard_failure(status, guard)
        _stop_wheels_preserving_pose(adapter)
        _close_project_camera_viewport_after_guard_failure(details)
        vision_processor.camera_view_status = status
        return False, str(status.get("error", "")), status
    vision_processor.camera_view_status = status
    return bool(ok), str(text or ""), status


def service_pending_viewport(
    *,
    args: Any,
    adapter: Any,
    scene_handle: Any,
    vision_processor: Any | None,
    logger_tail_provider: Any | None = None,
    runtime_metrics: dict[str, Any] | None = None,
) -> bool:
    if vision_processor is None:
        return False
    if not bool(getattr(vision_processor, "camera_view_status", {}).get("pending", False)):
        return False
    if not bool(getattr(args, "viewport_physics_guard", True)):
        vision_processor.service_pending_camera_viewport()
        return True

    before_count, before_tail = _tail_count(logger_tail_provider)

    def _service_pending() -> tuple[bool, str]:
        vision_processor.service_pending_camera_viewport()
        status = getattr(vision_processor, "camera_view_status", {})
        return bool(status.get("supported", False) or status.get("pending", False)), str(
            status.get("error", "") or "camera viewport pending"
        )

    started_wall = float((runtime_metrics or {}).get("started_wall", time.monotonic()))
    sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
    rtf = sim_time / max(1.0e-9, time.monotonic() - started_wall)
    (_ok, _text), guard = run_viewport_action_with_physics_guard(
        adapter=adapter,
        scene_handle=scene_handle,
        action="open_camera_viewport",
        callback=_service_pending,
        enabled=True,
        allow_restore_on_failure=False,
        rtf_before=rtf,
        rtf_after=rtf,
        penetration_tolerance_m=float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)),
    )
    new_lines = _new_tail_lines(logger_tail_provider, before_count, before_tail)
    guard.fabric_warning_detected = detect_fabric_or_viewport_warning(new_lines)
    status = merge_guard_into_camera_view_status(getattr(vision_processor, "camera_view_status", {}), guard)
    status["trigger_stage"] = str(status.get("trigger_stage", "") or "open_camera_viewport_pending_service")
    if not guard.passed:
        status = _mark_viewport_guard_failure(status, guard)
        _stop_wheels_preserving_pose(adapter)
        _close_project_camera_viewport_after_guard_failure(getattr(vision_processor, "camera_view_status", {}))
    vision_processor.camera_view_status = status
    return True


def build_common_worker_status(*, args: Any, adapter: Any, scene_handle: Any | None = None) -> dict[str, Any]:
    if adapter is None:
        return {
            "command_state": None,
            "target_joint_state": None,
            "actual_joint_state": None,
            "joint_diagnostics": [],
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
            "camera_detection_smoke_result": None,
            "camera_view_ground_contact_smoke_result": None,
            "physics_dt": float(getattr(args, "physics_dt", 0.0) or 0.0),
            "sim_step_hz": 0.0,
            "current_max_wheel_speed_rad_s": float(getattr(args, "max_wheel_speed_rad_s", 0.0) or 0.0),
            "default_wheel_speed_rad_s": float(getattr(args, "default_wheel_speed_rad_s", 0.0) or 0.0),
            "adapter": "",
            "telemetry": {"enabled": False, "run_dir": ""},
        }
    physics_dt = _safe_physics_dt(adapter, args)
    return {
        "command_state": _safe_capture_command_state(adapter),
        "sim_state": adapter.capture_sim_state() if hasattr(adapter, "capture_sim_state") else {"command_state": _safe_capture_command_state(adapter)},
        "target_joint_state": adapter.get_target_joint_state() if hasattr(adapter, "get_target_joint_state") else {},
        "actual_joint_state": adapter.get_actual_joint_state() if hasattr(adapter, "get_actual_joint_state") else {},
        "joint_diagnostics": adapter.get_joint_diagnostics() if hasattr(adapter, "get_joint_diagnostics") else [],
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
        "camera_detection_smoke_result": getattr(adapter, "camera_detection_smoke_result", None),
        "camera_view_ground_contact_smoke_result": getattr(adapter, "camera_view_ground_contact_smoke_result", None),
        "physics_dt": physics_dt,
        "sim_step_hz": 1.0 / physics_dt if physics_dt > 0 else 0.0,
        "current_max_wheel_speed_rad_s": float(getattr(adapter, "max_wheel_speed", 0.0) or 0.0),
        "default_wheel_speed_rad_s": float(getattr(adapter, "default_wheel_speed", 0.0) or 0.0),
        "adapter": type(adapter).__name__,
        "telemetry": _safe_telemetry_status(adapter),
    }


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
    vision = dict(enriched.get("vision", {}) if isinstance(enriched.get("vision"), dict) else {})
    camera_ready = bool(vision.get("camera_ready", enriched.get("camera_ready", False)))
    camera_prim = str(vision.get("camera_prim_path", "") or enriched.get("camera_prim_path", "") or "")
    headless = bool(enriched.get("effective_headless", False))
    camera_view_ready = runtime_ready and camera_ready and bool(camera_prim) and not headless
    camera_view_reason = ""
    if not camera_view_ready:
        parts: list[str] = []
        if not runtime_ready:
            parts.append("runtime is not ready")
        if headless:
            parts.append("headless mode")
        if not camera_ready:
            parts.append("camera is not ready")
        if not camera_prim:
            parts.append("camera prim path is unavailable")
        camera_view_reason = "; ".join(parts)
    enriched["camera_ready"] = camera_ready
    enriched["camera_view_ready"] = bool(camera_view_ready)
    enriched["camera_view_block_reason"] = camera_view_reason
    enriched["vision_detection_ready"] = bool(runtime_ready and camera_ready)
    enriched["vision_generation_ready"] = bool(runtime_ready)
    enriched.setdefault("validation_ready", False)
    enriched.setdefault("validation_block_reason", "")
    vision["camera_view_ready"] = bool(camera_view_ready)
    vision["camera_view_block_reason"] = camera_view_reason
    enriched["vision"] = vision
    return enriched


def ground_reference_result_is_valid(result: dict[str, Any]) -> bool:
    diagnostics = result.get("grounded_reference_diagnostics", result.get("ground_diagnostics", {}))
    classification = str(diagnostics.get("classification", ""))
    physical_ground_safe = bool(diagnostics.get("physical_ground_safe", classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}))
    return (
        bool(result.get("grounded_reference_valid", False))
        and bool(result.get("grounded_reference_physics_valid", False))
        and bool(result.get("grounded_reference_stable", False))
        and bool(diagnostics.get("checked", False))
        and physical_ground_safe
        and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}
    )


def _respawn_with_ground_policy(adapter: Any, *, vision_processor: Any | None) -> dict[str, Any]:
    if adapter is None or not hasattr(adapter, "respawn_robot"):
        return {"ok": False, "respawned": False, "error": "adapter does not support respawn_robot"}
    status = {
        "ready": True,
        "runtime_ready": True,
        "robot_ground": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
        "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
    }
    respawn_ready, _respawn_reason = respawn_status_from_worker_status(status)
    if not respawn_ready and hasattr(adapter, "initialize_grounded_respawn_reference"):
        initialize_adapter_ground_reference(adapter)
    status = {
        "ready": True,
        "runtime_ready": True,
        "robot_ground": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
        "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
    }
    respawn_ready, respawn_reason = respawn_status_from_worker_status(status)
    if not respawn_ready:
        _stop_wheels_preserving_pose(adapter)
        diagnostics = dict(getattr(adapter, "grounded_reference_diagnostics", default_robot_ground_diagnostics("invalid grounded reference")))
        return {
            "ok": False,
            "respawned": False,
            "error": respawn_reason or RESPAWN_GROUND_FAILURE_TEXT,
            "ground_diagnostics": diagnostics,
            "settle": dict(getattr(adapter, "last_ground_settle_result", {}) or {}),
        }
    result = dict(adapter.respawn_robot(settle=True) or {})
    if vision_processor is not None:
        vision_processor.notify_respawn()
    result.setdefault("ok", True)
    result.setdefault("respawned", True)
    return result


def _adapter_motion_ready(adapter: Any) -> bool:
    if adapter is None:
        return False
    status = {
        "ready": True,
        "runtime_ready": True,
        "robot_ground": dict(getattr(adapter, "robot_ground_diagnostics", default_robot_ground_diagnostics("not checked"))),
        "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
    }
    motion_ready, _reason, _ground_state = motion_status_from_worker_status(status)
    return bool(motion_ready)


def _idle_guard_failure_status(action: str, payload: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    revision = int(payload.get("action_revision", payload.get("request_revision", 0)) or 0)
    status = default_camera_viewport_status("; ".join(reasons))
    status.update(
        {
            "requested": True,
            "requested_action": str(action),
            "request_id": str(payload.get("request_id", "") or ""),
            "request_revision": revision,
            "completed_revision": revision,
            "completed": True,
            "physics_guard_checked": True,
            "physics_guard_passed": False,
            "trigger_stage": str(action),
        }
    )
    return status


def _mark_viewport_guard_failure(status: dict[str, Any], guard: Any) -> dict[str, Any]:
    merged = dict(status or {})
    reasons = list(getattr(guard, "reasons", []) or [])
    prompt = "Use Respawn And Validate Ground before continuing."
    merged.update(
        {
            "supported": False,
            "active": False,
            "completed": True,
            "pending": False,
            "physics_guard_checked": True,
            "physics_guard_passed": False,
            "postcondition_error": "Viewport physics guard failed",
            "error": "Viewport physics guard failed: " + ("; ".join(reasons) if reasons else "state changed") + f". {prompt}",
        }
    )
    return merged


def _close_project_camera_viewport_after_guard_failure(payload: dict[str, Any]) -> None:
    try:
        close_onboard_camera_viewport(
            request_id=str(payload.get("request_id", "") or "viewport_guard_failure"),
            action_revision=int(payload.get("action_revision", payload.get("request_revision", 0)) or 0),
        )
    except Exception:
        pass


def _stop_wheels_preserving_pose(adapter: Any) -> None:
    try:
        adapter.stop_wheels()
        if hasattr(adapter, "apply_commands_to_robot"):
            adapter.apply_commands_to_robot()
        robot = getattr(adapter, "robot", None)
        writer = getattr(robot, "write_data_to_sim", None)
        if callable(writer):
            writer()
    except Exception:
        pass


def _safe_capture_command_state(adapter: Any) -> dict[str, Any]:
    try:
        state = adapter.capture_command_state()
        return dict(state or {})
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


def _tail_count(logger_tail_provider: Any | None) -> tuple[int, list[str]]:
    if logger_tail_provider is None:
        return 0, []
    try:
        tail = list(logger_tail_provider())
        return len(tail), tail
    except Exception:
        return 0, []


def _new_tail_lines(logger_tail_provider: Any | None, before_count: int, before_tail: list[str]) -> list[str]:
    if logger_tail_provider is None:
        return []
    try:
        after_tail = list(logger_tail_provider())
    except Exception:
        return []
    if before_count and len(after_tail) >= before_count:
        return after_tail[before_count:]
    if before_tail and after_tail[: len(before_tail)] == before_tail:
        return after_tail[len(before_tail) :]
    return after_tail
