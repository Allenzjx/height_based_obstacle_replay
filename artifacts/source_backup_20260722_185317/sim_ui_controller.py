"""Real-robot-style UI controller backed by Isaac Sim commands."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from command_model import (
    DEFAULT_MAX_WHEEL_SPEED_RAD_S,
    HIP_LIMIT_DEG,
    KNEE_JOINT_NAMES,
    KNEE_LIMIT_DEG,
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WHEEL_NAME_TO_SHORT,
    WHEEL_SHORT_NAMES,
    CommandMessage,
    split_semicolon_commands,
)
from height_manifest import SUPPORTED_HEIGHTS_CM, normalize_height_cm, obstacle_height_m
from height_sequence_store import HeightSequenceStore
from playback import PlaybackManager, plan_from_steps
from sequence_manager import SequenceManager
from sequence_model import (
    apply_command_to_state,
    clone_command_state,
    coalesce_record_events,
    empty_command_state,
    format_step_json,
    make_event,
    make_step,
    normalize_step,
    step_summary,
)
from sim_obstacle_scene import SimSceneConfig
from sim_onboard_camera import camera_pitch_quat_wxyz
from sim_process_client import SimProcessClient, run_launch_preflight_for_args
from sim_robot_adapter import NullSimRobotAdapter
from sim_transport import SimTransport
from robot_ground_diagnostics import can_change_camera_view as ground_can_change_camera_view
from robot_ground_diagnostics import (
    GROUND_STATE_FAIL,
    GROUND_STATE_PASS,
    GROUND_STATE_PASS_WITH_VISUAL_WARNING,
    GROUND_STATE_UNVERIFIED,
    ground_state_from_diagnostics,
    motion_status_from_worker_status,
    respawn_status_from_worker_status,
)
from speed_scale import MotionScaleConfig, scale_manual_motion_command
from vision_task_state import (
    PHASE_BLOCKED,
    PHASE_DETECTING,
    PHASE_GENERATING_OBSTACLE,
    PHASE_PLAYING,
    PHASE_READY,
    PHASE_STABLE_DETECTED,
    PHASE_STEPS_READY,
    PHASE_STEPS_LOADING,
    PHASE_VALIDATED,
    PHASE_VALIDATING,
    PHASE_VALIDATION_FAILED,
    PHASE_WAITING_FOR_SCENE,
    VISION_SOURCE_EXTERNAL,
    VISION_SOURCE_GENERATED,
    VisionTaskState,
    inactive_state,
    normalize_source_mode,
    ready_state,
)


MODE_TEST = "TEST"
MODE_RECORDING_STEP = "RECORDING_STEP"
MODE_PENDING_RECORDED_STEP = "PENDING_RECORDED_STEP"
MODE_REPLACE_STEP_READY = "REPLACE_STEP_READY"
MODE_REPLACING_STEP = "REPLACING_STEP"
MODE_PENDING_REPLACEMENT = "PENDING_REPLACEMENT"
MODE_SERVO_WHEEL = "SERVO_WHEEL"
MODE_PLAYBACK = "PLAYBACK"
MODE_PLAYBACK_PAUSED = "PLAYBACK_PAUSED"
MODE_E_STOP = "E_STOP"


class DirtyHeightSwitchError(RuntimeError):
    pass


def _format_vision_status_summary(vision: dict[str, Any]) -> str:
    health = vision.get("camera_health", {}) if isinstance(vision.get("camera_health", {}), dict) else {}
    validation = vision.get("height_validation", {}) if isinstance(vision.get("height_validation", {}), dict) else {}
    mount = vision.get("camera_mount_validation", {}) if isinstance(vision.get("camera_mount_validation", {}), dict) else {}
    camera_view = vision.get("camera_view", {}) if isinstance(vision.get("camera_view", {}), dict) else {}
    ground = vision.get("robot_ground", {}) if isinstance(vision.get("robot_ground", {}), dict) else {}
    surface = ground.get("ground_surface", {}) if isinstance(ground.get("ground_surface", {}), dict) else {}
    if not surface and isinstance(vision.get("ground_surface"), dict):
        surface = dict(vision.get("ground_surface", {}))
    physics_guard = camera_view.get("physics_guard", {}) if isinstance(camera_view.get("physics_guard", {}), dict) else {}
    geometry = vision.get("camera_geometry", {}) if isinstance(vision.get("camera_geometry", {}), dict) else {}
    provenance = vision.get("height_provenance", {}) if isinstance(vision.get("height_provenance", {}), dict) else {}
    evidence = vision.get("height_measurement_evidence", {}) if isinstance(vision.get("height_measurement_evidence", {}), dict) else {}
    backend = ".".join(part for part in [str(health.get("backend_module", "") or ""), str(health.get("backend_class", "") or "")] if part)
    if not backend or backend == ".":
        backend = "-"
    rgb_shape = health.get("rgb_shape") or ()
    depth_shape = health.get("depth_shape") or ()
    expected = validation.get("expected_height_cm") if bool(validation.get("checked", False)) else None
    measured = validation.get("raw_height_cm") if validation.get("raw_height_cm") is not None else vision.get("raw_height_cm")
    detected = validation.get("detected_height_cm") if validation.get("detected_height_cm") is not None else vision.get("detected_height_cm")
    error_mm = validation.get("absolute_error_mm")
    validation_state = "NOT CHECKED"
    if bool(validation.get("checked", False)):
        validation_state = "PASS" if bool(validation.get("passed", False)) else "FAIL"
    frame_live = bool(health.get("checked", False)) and int(health.get("frames_advanced_during_check", 0) or 0) > 0 and not bool(health.get("stale_frame", True))
    lines = [
        f"Camera Backend: {backend}",
        f"Isaac Camera: {_yes_no(health.get('is_isaaclab_camera', False))}",
        f"Camera Prim: {'VALID' if bool(health.get('camera_prim_valid', False)) else 'INVALID'} {vision.get('camera_prim_path', '') or health.get('camera_prim_path', '') or ''}",
        f"Parent Rigid Body: {'VALID' if bool(health.get('parent_has_rigid_body_api', False)) else 'INVALID'} {vision.get('camera_parent_prim', '') or health.get('parent_prim_path', '') or ''}",
        f"RGB: {_shape_text(rgb_shape)}",
        f"Depth: {_shape_text(depth_shape)}",
        f"Depth Finite: {_percent(health.get('depth_finite_ratio'))}",
        f"Frame Live: {_yes_no(frame_live)}",
        f"Frame Age: {_seconds(health.get('frame_age_s'))}",
        f"Camera Health: {'PASS' if bool(health.get('ok', False)) else ('FAIL' if bool(health.get('checked', False)) else 'NOT CHECKED')}",
        f"Camera Viewport: action={camera_view.get('requested_action', '') or '-'} completed={_yes_no(camera_view.get('completed', False))} "
        f"active={_yes_no(camera_view.get('active', False))} supported={_yes_no(camera_view.get('supported', False))} mode={camera_view.get('mode', '') or '-'}",
        f"Camera View Pending: {_yes_no(camera_view.get('pending', False))} retry={int(camera_view.get('retry_count', 0) or 0)} "
        f"active_fallback_used={_yes_no(camera_view.get('active_fallback_used', False))}",
        f"Main Viewport: id={camera_view.get('main_viewport_id', '') or '-'} name={camera_view.get('main_viewport_name', '') or '-'} "
        f"camera_before={camera_view.get('main_camera_path_before', '') or '-'} camera_after={camera_view.get('main_camera_path_after', '') or '-'}",
        f"Secondary Viewport: name={camera_view.get('secondary_viewport_name', camera_view.get('camera_viewport_name', '')) or '-'} "
        f"camera={camera_view.get('secondary_camera_path', camera_view.get('camera_prim_path', '')) or '-'}",
        f"Perspective Restore: {_yes_no(camera_view.get('perspective_restore_verified', False))} "
        f"method={camera_view.get('perspective_restore_method', '') or '-'} postcondition={camera_view.get('postcondition_error', '') or '-'}",
        f"Camera View Physics Guard: {'PASS' if bool(camera_view.get('physics_guard_passed', False)) else ('FAIL' if bool(camera_view.get('physics_guard_checked', False)) else 'NOT CHECKED')}",
        f"Camera View Root Delta: {_meters(camera_view.get('root_pose_delta_m'))} rot={_degrees(camera_view.get('root_rotation_delta_deg'))} "
        f"joint={_meters(camera_view.get('max_joint_delta_rad'))} sim_dt={camera_view.get('sim_time_delta', 0)} steps={camera_view.get('sim_steps_delta', 0)}",
        f"Camera View Trigger Stage: {camera_view.get('trigger_stage', '') or '-'}",
        f"Camera View Error: {camera_view.get('error', '') or '-'}",
        f"Ground Contact: {_ground_contact_label(ground)}",
        f"Ground Diagnostic State: {ground.get('ground_state', vision.get('ground_state', '-')) or '-'}",
        f"Configured Ground Z: {_meters(surface.get('configured_ground_z_m'))}",
        f"Actual Ground Z: {_meters(surface.get('actual_ground_z_m'))}",
        f"Ground Z Delta: {_meters(surface.get('ground_z_delta_m'))}",
        f"Ground Prim: {surface.get('ground_prim_path', '-') or '-'}",
        f"Ground Collision Prim: {surface.get('ground_collision_prim_path', '-') or '-'}",
        f"Root Z: {_meters(ground.get('root_z_m'))}",
        f"Minimum Collision Clearance: {_meters(ground.get('minimum_collision_clearance_m'))}",
        f"Maximum Collision Penetration: {_meters(ground.get('maximum_collision_penetration_m'))}",
        f"Minimum Visual Clearance: {_meters(ground.get('minimum_visual_clearance_m'))}",
        f"Missing Collision Wheels: {', '.join(str(item) for item in ground.get('missing_collision_wheels', []) or []) or '-'}",
        f"Grounded Respawn Reference: {'VALID' if bool(ground.get('grounded_respawn_reference_valid', False) or vision.get('grounded_reference_valid', False)) else 'INVALID'} "
        f"physics={_yes_no(vision.get('grounded_reference_physics_valid', ground.get('grounded_reference_physics_valid', False)))} "
        f"visual={_yes_no(vision.get('grounded_reference_visual_valid', ground.get('grounded_reference_visual_valid', False)))} "
        f"stable={_yes_no(vision.get('grounded_reference_stable', ground.get('grounded_reference_stable', False)))}",
        f"Last Ground Correction: {_meters(vision.get('last_ground_correction_z_m', ground.get('correction_z_m')))}",
        f"Fabric/Render Warning: {_yes_no(camera_view.get('fabric_warning_detected', physics_guard.get('fabric_warning_detected', False)))}",
        f"Camera Mount: {_mount_state(mount)}",
        f"Mount Error: pos={_meters(mount.get('position_error_m'))} rot={_degrees(mount.get('orientation_error_deg'))}",
        f"Camera Geometry: {geometry.get('framing_state', 'NOT_CHECKED')} axis={geometry.get('center_ray_direction_w', '-')}",
        f"Coverage: ground={_percent(geometry.get('ground_point_fraction'))} obstacle={_percent(geometry.get('obstacle_point_fraction'))} top={_percent(geometry.get('top_point_fraction'))}",
        "Measurement Model: Top-plane world-Z from RGB-D point cloud",
        "Formula: height = top_z - ground_z",
        "Top-plane visibility is required by the current measurement model.",
        f"Height Source: {provenance.get('height_source', 'isaac_rgbd_depth_geometry')}",
        f"Uses Depth: {_yes_no(provenance.get('depth_data_type') == 'distance_to_image_plane')}",
        f"Uses Intrinsics: {_yes_no(provenance.get('intrinsics_used', False))}",
        f"Uses Camera Pose: {_yes_no(provenance.get('camera_world_pose_used', False))}",
        f"Uses Ground Reference: {_yes_no(evidence.get('ground_z_m') is not None)}",
        f"Uses Generated Obstacle X As ROI: {_yes_no(evidence.get('obstacle_x_prior_used', provenance.get('obstacle_x_prior_used', False)))}",
        f"Uses Expected Height In Detector: {_yes_no(provenance.get('expected_height_used_by_detector', False))}",
        f"Uses Generated Height In Detector: {_yes_no(provenance.get('generated_height_used_by_detector', False))}",
        f"Uses Scene Obstacle Height In Detector: {_yes_no(provenance.get('scene_obstacle_height_used_by_detector', False))}",
        f"Top Z: {_meters(evidence.get('top_z_m'))}",
        f"Ground Z Used By Detector: {_meters(evidence.get('ground_z_m'))}",
        f"Depth Evidence: rev={evidence.get('depth_frame_revision', vision.get('frame_revision', 0))} shape={evidence.get('depth_shape', [])} finite={_percent(evidence.get('depth_finite_ratio'))} hash={evidence.get('depth_fingerprint', '') or '-'}",
        f"Intrinsics Evidence: used={_yes_no(evidence.get('intrinsics_used', provenance.get('intrinsics_used', False)))} hash={evidence.get('intrinsics_fingerprint', '') or '-'}",
        f"Detector Audit: {'PASS' if bool(evidence.get('detector_input_audit_passed', provenance.get('detector_input_audit_passed', False))) else 'FAIL'}",
        f"ROI Prior: {provenance.get('roi_source', vision.get('roi_source', '-'))}",
        f"Expected Height: {_cm(expected)}",
        f"Measured Height: {_cm(measured)}",
        f"Detected Height: {_bucket_cm(detected)}",
        f"Error: {_mm(error_mm)}",
        f"Confidence: {_percent(vision.get('confidence'))}",
        f"Stable Frames: {int(vision.get('stable_count', 0) or 0)}/{int(vision.get('stable_required', 0) or 0)}",
        f"Height Validation: {validation_state}",
        f"Debug Image: {vision.get('debug_image_path', '') or '-'}",
        f"Debug JSON: {vision.get('debug_sidecar_path', '') or '-'}",
    ]
    reasons = []
    if isinstance(health.get("reasons"), list) and health.get("reasons"):
        reasons.append("Camera: " + "; ".join(str(item) for item in health.get("reasons", [])))
    if isinstance(validation.get("reasons"), list) and validation.get("reasons"):
        reasons.append("Validation: " + "; ".join(str(item) for item in validation.get("reasons", [])))
    if isinstance(mount.get("reasons"), list) and mount.get("reasons"):
        reasons.append("Mount: " + "; ".join(str(item) for item in mount.get("reasons", [])))
    if bool(camera_view.get("requested", False)) and not bool(camera_view.get("active", False)):
        reasons.append(
            "Manual Camera View: Click the Camera button in the upper-left Isaac Viewport and select Perspective; "
            "or open Window > Viewports > Viewport 2, then select Cameras > onboard_rgbd_camera. "
            "Camera Inspector path: Tools > Sensors > Camera Inspector; Refresh; Select /World/WLRRobot/base_link/onboard_rgbd_camera; Create Viewport."
        )
    if isinstance(geometry.get("reasons"), list) and geometry.get("reasons"):
        reasons.append("Geometry: " + "; ".join(str(item) for item in geometry.get("reasons", [])))
    if isinstance(ground.get("reasons"), list) and ground.get("reasons"):
        reasons.append("Ground: " + "; ".join(str(item) for item in ground.get("reasons", [])))
    if isinstance(physics_guard.get("reasons"), list) and physics_guard.get("reasons"):
        reasons.append("Camera Guard: " + "; ".join(str(item) for item in physics_guard.get("reasons", [])))
    if str(provenance.get("forbidden_input_reason", "") or ""):
        reasons.append("Provenance: " + str(provenance.get("forbidden_input_reason")))
    if reasons:
        lines.append("Reasons:")
        lines.extend(reasons)
    return "\n".join(lines)


def _yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def _ground_contact_label(ground: dict[str, Any]) -> str:
    classification = str(ground.get("classification", "") or "UNKNOWN")
    if classification == "OK":
        return "PASS"
    if classification == "VISUAL_ONLY_INTERSECTION":
        return "PASS WITH VISUAL WARNING"
    if classification in {"COLLISION_PENETRATION", "MISSING_WHEEL_COLLISION", "COLLIDER_CONFIRMED_MISSING"}:
        return "FAIL"
    if classification == "COLLIDER_RESOLUTION_FAILED":
        return "UNVERIFIED"
    if classification == "RENDER_OR_FABRIC_DESYNC_SUSPECTED":
        return "VISUAL/FABRIC SUSPECTED"
    return classification or "UNKNOWN"


def _shape_text(shape: Any) -> str:
    if not shape:
        return "-"
    try:
        values = [int(v) for v in shape]
    except Exception:
        return str(shape)
    if len(values) >= 2:
        return f"{values[1]}x{values[0]}"
    return "x".join(str(v) for v in values)


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _seconds(value: Any) -> str:
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return "-"


def _cm(value: Any) -> str:
    try:
        return f"{float(value):.3f}cm"
    except (TypeError, ValueError):
        return "-"


def _mm(value: Any) -> str:
    try:
        return f"{float(value):.2f}mm"
    except (TypeError, ValueError):
        return "-"


def _meters(value: Any) -> str:
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return "invalid"
        if abs(numeric) >= 1.0e12:
            return "invalid"
        return f"{numeric:.4f}m"
    except (TypeError, ValueError):
        return "unavailable"


def _degrees(value: Any) -> str:
    try:
        return f"{float(value):.3f}deg"
    except (TypeError, ValueError):
        return "-"


def _mount_state(mount: dict[str, Any]) -> str:
    if bool(mount.get("pending", False)):
        return "PENDING RESPAWN"
    if bool(mount.get("checked", False)):
        return "PASS" if bool(mount.get("passed", False)) else "FAIL"
    return "NOT CHECKED"


def _bucket_cm(value: Any) -> str:
    try:
        return f"{int(value)}cm"
    except (TypeError, ValueError):
        return "-"


def set_text_preserving_view(widget: Any, text: str, *, follow_bottom: bool = False) -> None:
    now = time.monotonic()
    try:
        old_yview = tuple(float(value) for value in widget.yview())
    except Exception:
        old_yview = (0.0, 1.0)
    try:
        insert_index = str(widget.index("insert"))
    except Exception:
        insert_index = "1.0"
    at_bottom = len(old_yview) >= 2 and old_yview[1] >= 0.999
    last_scroll = float(getattr(widget, "_last_user_scroll_at", 0.0) or 0.0)
    dragging = bool(getattr(widget, "_scrollbar_dragging", False))
    user_recently_scrolled = dragging or (last_scroll > 0.0 and now - last_scroll < 2.0)
    state = ""
    try:
        state = str(widget.cget("state"))
    except Exception:
        state = ""
    try:
        widget.configure(state="normal")
    except Exception:
        pass
    try:
        widget.delete("1.0", "end")
        widget.insert("1.0", str(text))
    finally:
        try:
            if follow_bottom and at_bottom and not user_recently_scrolled:
                widget.yview_moveto(1.0)
            else:
                widget.yview_moveto(old_yview[0] if old_yview else 0.0)
        except Exception:
            pass
        try:
            widget.mark_set("insert", insert_index)
        except Exception:
            pass
        try:
            if state:
                widget.configure(state=state)
        except Exception:
            pass


class HeightReplayController:
    """Command controller with real_robot_ui_controller-like behavior."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.no_sim = bool(getattr(args, "no_sim", False))
        self.store = HeightSequenceStore(getattr(args, "store_root", None) or None)
        self.manifest_revision = 0
        self.refresh_manifest()
        self.current_height_cm = normalize_height_cm(getattr(args, "height_cm", 0))
        self.manager = SequenceManager(self.store.steps_path(self.current_height_cm))
        self.adapter: Any = NullSimRobotAdapter()
        self.transport = SimTransport(self.adapter)
        self.sim_worker: Any | None = None
        self.sim_client: SimProcessClient | None = None
        self.sim_launch_mode = "disabled" if self.no_sim else str(getattr(args, "sim_launch_mode", "subprocess")).lower()
        if self.sim_launch_mode == "disabled":
            self.no_sim = True
        self.pending_height_cm: int | None = None
        self.sim_ready = False
        self.loaded_sim_height_cm: int | None = None
        self.latest_sim_status: dict[str, Any] = {}
        self.ui_refresh_hz = 0.0
        self._last_ui_refresh_wall = time.monotonic()
        self.mode = MODE_TEST
        self.status = "Initialized. Isaac Sim not started." if not self.no_sim else "Initialized in --no-sim mode."
        self.status_log: list[str] = []
        self.last_preflight_report: dict[str, Any] = {}
        self._warned_gui_headless = False
        self.detail_text = ""
        self.last_exported_step_json: Path | None = None
        self.selected_step_index: int | None = None
        self.last_raw_motion_command = ""
        self.last_scaled_motion_command = ""
        self.last_raw_wheel_speeds: tuple[float, ...] = ()
        self.last_scaled_wheel_speeds: tuple[float, ...] = ()
        self.last_motion_warnings: tuple[str, ...] = ()

        self.record_start_wall_time = 0.0
        self.record_command_state_before: dict[str, Any] | None = None
        self.record_sim_state_before: dict[str, Any] | None = None
        self.record_events: list[dict[str, Any]] = []
        self.last_record_coalesce_stats: dict[str, Any] = {}
        self.pending_step: dict[str, Any] | None = None
        self.pending_replacement: dict[str, Any] | None = None
        self.replace_target_index: int | None = None
        self.recording_kind = ""

        self.combine_mode_enabled = False
        self.combine_selected_indices: set[int] = set()
        self.combine_preview_text = ""
        self.allow_combine_conflicts = False
        self.operation_busy = False
        self.busy_name = ""
        interval_from_ms = max(0.0, float(getattr(args, "record_event_min_interval_ms", 50)) / 1000.0)
        record_event_max_hz = max(0.0, float(getattr(args, "record_event_max_hz", 20.0)))
        interval_from_hz = (1.0 / record_event_max_hz) if record_event_max_hz > 0.0 else 0.0
        self.record_event_min_interval_s = max(interval_from_ms, interval_from_hz)
        self.record_max_events_per_step = max(1, int(getattr(args, "record_max_events_per_step", 2000)))
        self.record_coalesce_slider_events = bool(getattr(args, "record_coalesce_slider_events", True))
        self.max_text_widget_chars = max(1000, int(getattr(args, "max_text_widget_chars", 200000)))
        self.restore_step_start_state_before_selected_playback = bool(getattr(args, "restore_step_start_state_before_selected_playback", True))
        self.restore_full_sim_pose_if_available = bool(getattr(args, "restore_full_sim_pose_if_available", True))
        self.fallback_to_command_state_before = bool(getattr(args, "fallback_to_command_state_before", True))
        self.playback_pre_step_settle_s = max(0.0, float(getattr(args, "playback_pre_step_settle_s", 0.30)))
        self.respawn_play_settle_s = max(0.0, float(getattr(args, "respawn_play_settle_s", 0.30)))
        self.servo_wheel_staging_active = False
        self.servo_wheel_staged_state: dict[str, Any] = empty_command_state()
        self.servo_wheel_staged_dirty = False
        self.servo_wheel_last_launch_commands: list[str] = []
        self.servo_wheel_last_launch_time: float | None = None

        self.playback = PlaybackManager(self)
        self.motion_scale = MotionScaleConfig(
            global_motion_speed_scale=float(getattr(args, "global_motion_speed_scale", 1.0)),
            wheel_speed_scale=float(getattr(args, "wheel_speed_scale", 1.0)),
            servo_command_scale=float(getattr(args, "servo_command_scale", 1.0)),
            playback_speed_scale=float(getattr(args, "playback_speed_scale", 1.0)),
            apply_to_manual_control=bool(getattr(args, "apply_speed_scale_to_manual", True)),
            apply_to_playback=bool(getattr(args, "apply_speed_scale_to_playback", True)),
            preserve_wheel_distance=bool(getattr(args, "preserve_wheel_distance", True)),
        )
        self.playback.preserve_wheel_distance = self.motion_scale.preserve_wheel_distance
        self.height_task_mode = "recording"
        self.height_task_active = False
        self.height_task_heights: list[int] = []
        self.height_task_index = -1
        self.vision_enabled = bool(getattr(args, "onboard_camera", True))
        self.vision_auto_replay_enabled = bool(getattr(args, "vision_auto_replay", False))
        self.vision_auto_replay_armed = False
        self.vision_respawn_before_replay = bool(getattr(args, "vision_respawn_before_replay", True))
        self.vision_last_detection_revision = 0
        self.vision_last_consumed_detection_revision = 0
        self.vision_last_detected_height_cm: int | None = None
        self.vision_last_auto_action = "idle"
        self.vision_last_block_reason = ""
        self.vision_auto_replay_cooldown_s = max(0.0, float(getattr(args, "vision_auto_replay_cooldown_s", 2.0)))
        self.vision_last_replay_started_at = 0.0
        self.vision_task = inactive_state()
        self.vision_task_active = False
        self.vision_source_mode = VISION_SOURCE_GENERATED
        self.vision_generated_height_cm: int | None = None
        self.vision_generation_revision = 0
        self.vision_generation_request_id = ""
        self.vision_detection_baseline_revision = 0
        self.vision_frame_baseline_revision = 0
        self.vision_generation_requested_at = 0.0
        self.vision_scene_ready = False
        self.vision_scene_obstacle_revision = 0
        self.vision_steps: list[dict[str, Any]] = []
        self.vision_steps_path: Path | None = None
        self.vision_steps_height_cm: int | None = None
        self.vision_steps_detection_revision = 0
        self.vision_steps_ready = False
        self.vision_steps_error = ""
        self.vision_steps_revision = 0
        self.vision_validated_height_cm: int | None = None
        self.vision_validated_detection_revision = 0
        self.vision_pending_validation_revision = 0
        self.vision_pending_validation_expected_cm: int | None = None
        self.camera_view_action_revision = 0
        self.camera_view_pending_revision = 0
        self.camera_view_pending_request_id = ""
        self.camera_view_pending_action = ""
        self.camera_view_last_completed_revision = 0

    @property
    def max_wheel_speed(self) -> float:
        return float(getattr(self.args, "max_wheel_speed_rad_s", getattr(self.args, "max_wheel_speed", DEFAULT_MAX_WHEEL_SPEED_RAD_S)))

    @property
    def default_wheel_speed(self) -> float:
        return float(getattr(self.args, "default_wheel_speed_rad_s", getattr(self.args, "default_wheel_speed", self.max_wheel_speed * 0.25)))

    @property
    def runtime_ready(self) -> bool:
        if self.no_sim:
            return True
        if str(self.latest_sim_status.get("phase", "") or "") == "running" and not str(self.latest_sim_status.get("traceback", "") or ""):
            return True
        return bool(self.latest_sim_status.get("runtime_ready", self.sim_ready))

    def ground_state(self) -> str:
        ground = self.latest_sim_status.get("robot_ground", {})
        return ground_state_from_diagnostics(ground if isinstance(ground, dict) else {})

    def motion_readiness(self) -> tuple[bool, str]:
        if self.no_sim:
            return True, ""
        has_explicit_motion_status = any(
            key in self.latest_sim_status
            for key in (
                "motion_ready",
                "motion_block_reason",
                "ground_state",
                "grounded_reference_valid",
            )
        )
        ground = self.latest_sim_status.get("robot_ground")
        if isinstance(ground, dict) and ground:
            has_explicit_motion_status = True
        if not has_explicit_motion_status:
            return bool(self.runtime_ready), "" if self.runtime_ready else "Isaac runtime is not ready."
        status = {
            **dict(self.latest_sim_status),
            "runtime_ready": self.runtime_ready,
            "ready": self.runtime_ready,
        }
        ready, reason, _ground_state = motion_status_from_worker_status(status)
        explicit = str(self.latest_sim_status.get("motion_block_reason", "") or "")
        return bool(ready), explicit or reason

    @property
    def motion_ready(self) -> bool:
        return self.motion_readiness()[0]

    def respawn_readiness(self) -> tuple[bool, str]:
        if self.no_sim:
            return True, ""
        has_explicit_respawn_status = any(
            key in self.latest_sim_status
            for key in (
                "respawn_ready",
                "respawn_block_reason",
                "ground_reference_block_reason",
                "grounded_reference_valid",
                "grounded_reference_stable",
                "ground_state",
            )
        )
        ground = self.latest_sim_status.get("robot_ground")
        if isinstance(ground, dict) and ground:
            has_explicit_respawn_status = True
        if not has_explicit_respawn_status:
            return bool(self.runtime_ready), "" if self.runtime_ready else "Isaac runtime is not ready."
        status = {
            **dict(self.latest_sim_status),
            "runtime_ready": self.runtime_ready,
            "ready": self.runtime_ready,
        }
        if "respawn_ready" in self.latest_sim_status:
            ready = bool(self.latest_sim_status.get("respawn_ready", False))
            reason = str(self.latest_sim_status.get("respawn_block_reason", self.latest_sim_status.get("ground_reference_block_reason", "")) or "")
            return ready, reason
        return respawn_status_from_worker_status(status)

    @property
    def respawn_ready(self) -> bool:
        return self.respawn_readiness()[0]

    def playback_readiness(self, *, respawn_first: bool = False) -> tuple[bool, str]:
        ok, reason = self.can_playback()
        if not ok:
            return ok, reason
        if respawn_first:
            respawn_ok, respawn_reason = self.respawn_readiness()
            if not respawn_ok:
                if self.vision_respawn_before_replay:
                    return False, (respawn_reason or "Respawn is not ready.") + " Disable Respawn Before Auto Replay or calibrate a valid ground reference."
                return False, respawn_reason or "Respawn is not ready."
        return True, ""

    def camera_view_readiness(self) -> tuple[bool, str]:
        vision = self.latest_sim_status.get("vision", {}) if isinstance(self.latest_sim_status.get("vision"), dict) else {}
        runtime_ready = self.runtime_ready
        headless = bool(self.latest_sim_status.get("effective_headless", bool(getattr(self.args, "headless", False))))
        camera_ready = bool(vision.get("camera_ready", self.latest_sim_status.get("camera_ready", False)))
        camera_path = str(vision.get("camera_prim_path", "") or self.latest_sim_status.get("camera_prim_path", "") or "")
        idle_ok, idle_reason = self.can_change_camera_view()
        reasons: list[str] = []
        if not runtime_ready:
            reasons.append("runtime is not ready")
        if headless:
            reasons.append("headless mode")
        if not camera_ready:
            reasons.append("camera is not ready")
        if not camera_path:
            reasons.append("camera prim path is unavailable")
        if not idle_ok:
            reasons.append(idle_reason or "robot is not idle")
        return not reasons, "; ".join(reasons)

    @property
    def recording_active(self) -> bool:
        return self.mode in {MODE_RECORDING_STEP, MODE_REPLACING_STEP}

    def can_start_recording(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before recording Height Task steps."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.playback.active:
            return False, "Stop playback before recording."
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard the pending step before recording."
        if self.mode not in {MODE_TEST, MODE_SERVO_WHEEL, MODE_REPLACE_STEP_READY}:
            return False, f"step_record start is only valid in TEST, SERVO_WHEEL, or REPLACE_STEP_READY. Current mode={self.mode}"
        return True, ""

    def can_accept_recorded_step(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before accepting recorded steps."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.mode != MODE_PENDING_RECORDED_STEP or self.pending_step is None:
            return False, "No pending recorded step to accept."
        return True, ""

    def can_prepare_replacement(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before replacing Height Task steps."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before replacement."
        if self.mode not in {MODE_TEST, MODE_SERVO_WHEEL}:
            return False, f"Replace is only valid in TEST or SERVO_WHEEL. Current mode={self.mode}"
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before replacement."
        return True, ""

    def can_accept_replacement(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before accepting replacements."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.mode != MODE_PENDING_REPLACEMENT or self.pending_replacement is None or self.replace_target_index is None:
            return False, "No pending replacement to accept."
        return True, ""

    def can_playback(self) -> tuple[bool, str]:
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before playback."
        if self.mode not in {MODE_TEST, MODE_SERVO_WHEEL}:
            return False, f"Playback is only valid in TEST or SERVO_WHEEL. Current mode={self.mode}"
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before playback."
        if not self.no_sim and not self.runtime_ready:
            return False, "Isaac runtime is not ready."
        motion_ok, motion_reason = self.motion_readiness()
        if not self.no_sim and not motion_ok:
            return False, motion_reason or "Motion is blocked by ground safety."
        return True, ""

    def can_combine(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before combining Height Task steps."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before combine."
        if self.mode != MODE_TEST:
            return False, f"Combine is only valid in TEST. Current mode={self.mode}"
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before combine."
        if len(self.combine_selected_indices) < 2:
            return False, "Select at least two accepted steps to combine."
        if not self.combine_selection_is_contiguous():
            return False, "Combine selection must be contiguous. Use Select Contiguous Range."
        return True, ""

    def combine_selection_is_contiguous(self) -> bool:
        selected = sorted(self.combine_selected_indices)
        return len(selected) >= 2 and selected == list(range(selected[0], selected[-1] + 1))

    def can_save(self) -> tuple[bool, str]:
        if self.visible_steps_are_read_only():
            return False, "Vision steps are read-only. Finish Vision Task before saving Height Task steps."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before saving."
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before saving."
        return True, ""

    def can_switch_height(self, *, discard_dirty: bool = False) -> tuple[bool, str]:
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.recording_active:
            return False, "Stop recording before switching height."
        if self.playback.active:
            return False, "Stop playback before switching height."
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before switching height."
        if self.pending_step is not None or self.pending_replacement is not None:
            return False, "Accept or discard pending step before switching height."
        if self.manager.dirty and not discard_dirty:
            return False, "Current sequence has unsaved changes. Save or confirm discard before switching height."
        return True, ""

    def active_steps_view(self) -> str:
        return "vision" if self.vision_task_active else "height"

    def get_visible_steps(self) -> list[dict[str, Any]]:
        if self.active_steps_view() == "vision":
            return list(self.vision_steps) if self.vision_steps_ready else []
        return list(self.manager.steps)

    def get_visible_step(self, index: int) -> dict[str, Any]:
        steps = self.get_visible_steps()
        idx = int(index)
        if idx < 1 or idx > len(steps):
            raise IndexError(f"Accepted step index out of range: {idx}")
        return steps[idx - 1]

    def get_visible_steps_source(self) -> str:
        if self.active_steps_view() != "vision":
            return f"Height - {self.current_height_cm}cm"
        if not self.vision_steps_ready:
            return "Vision - waiting for validation"
        return f"Vision - detected {int(self.vision_steps_height_cm or 0)}cm"

    def visible_steps_are_read_only(self) -> bool:
        return self.active_steps_view() == "vision"

    def visible_step_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, step in enumerate(self.get_visible_steps(), start=1):
            rows.append(self._step_to_row(step, index))
        return rows

    def can_start_vision_task(self) -> tuple[bool, str]:
        if self.height_task_active:
            return False, "Finish the active Height Task before starting Vision Task."
        if self.recording_active:
            return False, "Stop recording before starting Vision Task."
        if self.pending_step is not None:
            return False, "Accept or discard the pending recorded step before starting Vision Task."
        if self.pending_replacement is not None:
            return False, "Accept or discard the pending replacement before starting Vision Task."
        if self.manager.dirty:
            return False, "Save or discard unsaved Height Task steps before starting Vision Task."
        if self.playback.active:
            return False, "Stop playback before starting Vision Task."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        return True, ""

    def _set_vision_blocked(self, reason: str) -> None:
        self.vision_task.phase = PHASE_BLOCKED
        self.vision_task.block_reason = str(reason)
        self.vision_task.last_action = "blocked"
        self.vision_last_block_reason = str(reason)

    def _clear_vision_steps(self, reason: str = "") -> None:
        self.vision_steps = []
        self.vision_steps_path = None
        self.vision_steps_height_cm = None
        self.vision_steps_detection_revision = 0
        self.vision_steps_ready = False
        self.vision_steps_error = str(reason or "")
        self.vision_steps_revision += 1
        self.vision_validated_height_cm = None
        self.vision_validated_detection_revision = 0

    def _sync_vision_task_state(self) -> None:
        vision = self.latest_sim_status.get("vision", {})
        validation = vision.get("height_validation", {}) if isinstance(vision, dict) and isinstance(vision.get("height_validation"), dict) else {}
        detected = vision.get("detected_height_cm") if isinstance(vision, dict) else None
        detection_revision = int(vision.get("detection_revision", 0) or 0) if isinstance(vision, dict) else 0
        scene_height = self.latest_sim_status.get("scene_height_cm", self.latest_sim_status.get("height_cm"))
        obstacle_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0)
        self.vision_task.active = bool(self.vision_task_active)
        self.vision_task.source_mode = self.vision_source_mode
        self.vision_task.requested_height_cm = self.vision_generated_height_cm
        self.vision_task.generated_height_cm = self.vision_generated_height_cm
        self.vision_task.scene_height_cm = int(scene_height) if scene_height is not None else None
        self.vision_task.obstacle_revision = obstacle_revision
        self.vision_task.generation_request_id = self.vision_generation_request_id
        self.vision_task.generation_frame_baseline = self.vision_frame_baseline_revision
        self.vision_task.generation_detection_baseline = self.vision_detection_baseline_revision
        self.vision_task.detected_height_cm = int(detected) if detected is not None else None
        self.vision_task.detected_revision = detection_revision
        self.vision_task.validation_checked = bool(validation.get("checked", False))
        self.vision_task.validation_passed = bool(validation.get("passed", False))
        self.vision_task.validated_height_cm = self.vision_validated_height_cm
        self.vision_task.validated_detection_revision = self.vision_validated_detection_revision
        self.vision_task.steps_ready = bool(self.vision_steps_ready)
        self.vision_task.steps_height_cm = self.vision_steps_height_cm
        self.vision_task.steps_path = str(self.vision_steps_path or "")
        self.vision_task.steps_count = len(self.vision_steps)

    def _step_to_row(self, step: dict[str, Any], index: int) -> dict[str, Any]:
        normalized = normalize_step(step, index=index)
        return {
            "index": int(index),
            "name": str(normalized.get("name", "")),
            "type": str(normalized.get("type", normalized.get("step_type", ""))),
            "duration": float(normalized.get("duration", 0.0) or 0.0),
            "events_count": len(normalized.get("events", []) or []),
            "note": str(normalized.get("note", "")),
        }

    def _run_operation(self, name: str, func: Any) -> Any:
        if self.operation_busy:
            self._warn(f"[WARN] Operation busy: {self.busy_name}")
            return None
        started = time.perf_counter()
        self.operation_busy = True
        self.busy_name = name
        self.status = f"{name}..."
        self.manager.last_operation_report = {}
        try:
            result = func()
            return result
        except Exception as exc:
            self._warn(f"[WARN] {name} failed: {exc}")
            return None
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            report = dict(getattr(self.manager, "last_operation_report", {}) or {})
            suffix = ""
            if report:
                manager_ms = report.get("elapsed_ms")
                if manager_ms is not None:
                    suffix = f" manager={report.get('operation', '')} {float(manager_ms):.1f}ms"
            message = f"[PERF] {name} took {elapsed_ms:.1f}ms{suffix}"
            if elapsed_ms > 500.0:
                self._warn("[WARN] " + message)
            else:
                self.status_log.append(message)
                print(message)
            self.operation_busy = False
            self.busy_name = ""

    def command_state_summary(self, state: dict[str, Any] | None) -> str:
        normalized = clone_command_state(state)
        servo_parts = [f"{name}={float(value):.1f}" for name, value in sorted(normalized["servos"].items())]
        wheel_parts = []
        for name, value in sorted(normalized["wheels"].items()):
            short = WHEEL_NAME_TO_SHORT.get(name, name)
            wheel_parts.append(f"{short}={float(value):.2f}")
        return "servos: " + ", ".join(servo_parts) + "\n" + "wheels: " + ", ".join(wheel_parts)

    def capture_current_sim_state(self) -> dict[str, Any]:
        state = self.transport.capture_sim_state()
        if not isinstance(state, dict):
            state = {}
        state = dict(state)
        state.setdefault("command_state", self.transport.capture_command_state())
        state["height_cm"] = self.current_height_cm
        state["sim_time"] = self.latest_sim_status.get("sim_time", state.get("sim_time", 0.0))
        return state

    def compact_step_details(self, step: dict[str, Any], *, title: str | None = None) -> str:
        normalized = normalize_step(step)
        events = normalized.get("events", [])
        heading = title or f"Step {int(normalized['index']):03d}"
        height = normalized.get("height_cm", self.current_height_cm)
        lines = [
            heading,
            f"name={normalized['name']}",
            f"type={normalized['type']} duration={float(normalized['duration']):.3f}s events={len(events)} height={height}cm",
            f"note={normalized.get('note', '')}",
            "",
            "command_state_after:",
            self.command_state_summary(normalized.get("command_state_after")),
        ]
        coalesce = normalized.get("record_coalesce")
        if isinstance(coalesce, dict) and coalesce.get("coalesced"):
            lines.extend(
                [
                    "",
                    "recording event coalescing:",
                    f"{coalesce.get('original_count')} -> {coalesce.get('coalesced_count')} events, dropped={coalesce.get('dropped_count', 0)}",
                ]
            )
        return "\n".join(lines)

    def step_json_text(self, step: dict[str, Any], *, max_chars: int | None = None) -> str:
        limit = self.max_text_widget_chars if max_chars is None else max(1000, int(max_chars))
        text = format_step_json(step)
        if len(text) <= limit:
            return text
        return (
            f"[TRUNCATED] Step JSON is {len(text)} chars; showing first {limit} chars. "
            "Use Export Step JSON for the full file.\n\n"
            + text[:limit]
        )

    def set_step_summary_detail(self, step: dict[str, Any], *, title: str | None = None) -> None:
        self.detail_text = self.compact_step_details(step, title=title)

    def export_step_json(self, index: int) -> Path | None:
        try:
            step = normalize_step(self.get_visible_step(index), index=index)
        except Exception as exc:
            self._warn(f"[WARN] Could not export step JSON: {exc}")
            return None
        export_height = self.vision_steps_height_cm if self.visible_steps_are_read_only() and self.vision_steps_height_cm is not None else self.current_height_cm
        path = self.store.height_dir(export_height) / f"step_{int(index):03d}_full.json"
        path.write_text(format_step_json(step) + "\n", encoding="utf-8")
        self.last_exported_step_json = path
        self._info(f"[INFO] Exported full step JSON: {path}")
        return path

    def refresh_manifest(self) -> None:
        self.store.refresh_manifest()
        self.manifest_revision += 1

    @property
    def current_task_height(self) -> int | None:
        if not self.height_task_active or self.height_task_index < 0:
            return None
        if self.height_task_index >= len(self.height_task_heights):
            return None
        return self.height_task_heights[self.height_task_index]

    def start_sim_if_needed(self) -> None:
        if self.no_sim:
            self.sim_ready = False
            self.transport.attach(self.adapter, ready=False)
            self._info("[INFO] --no-sim mode: Isaac Sim startup skipped.")
            return
        if self.sim_launch_mode == "subprocess":
            if self.sim_client is None:
                self._info("[INFO] Starting Isaac Sim subprocess worker...")
                self.sim_client = SimProcessClient(self.args)
                self.transport.attach_process_client(self.sim_client)
                self.sim_client.start()
                self.status = "Isaac Sim subprocess worker launched; waiting for status."
            return
        if self.sim_launch_mode == "thread" and self.sim_worker is None:
            from sim_worker import SimWorker

            self._info("[INFO] Starting Isaac Sim worker...")
            self.sim_worker = SimWorker(
                self.args,
                config_factory=self.config_for_height,
                initial_height_cm=self.current_height_cm,
                status_interval_s=float(getattr(self.args, "sim_status_refresh_ms", 250)) / 1000.0,
                no_continuous_sim_step=bool(getattr(self.args, "no_continuous_sim_step", False)),
            )
            self.transport.attach_worker(self.sim_worker)
            self.sim_worker.start()
            return
        if self.sim_launch_mode == "main":
            self._warn("[WARN] --sim-launch-mode main is not used with the Tk UI. Use subprocess or --auto-play.")
            return

    def run_isaac_preflight(self) -> dict[str, Any]:
        if self.no_sim:
            report = {"preflight_ok": False, "preflight_error": "--no-sim mode: preflight skipped."}
        else:
            report = run_launch_preflight_for_args(self.args)
        self.last_preflight_report = dict(report)
        self.latest_sim_status = {
            **self.latest_sim_status,
            "phase": "preflight_completed",
            "preflight": report,
            "preflight_ok": bool(report.get("preflight_ok", False)),
            "preflight_error": str(report.get("preflight_error", "") or ""),
            "candidate_reports": report.get("candidate_reports", []),
        }
        if report.get("preflight_ok"):
            selected = report.get("selected_report", {}) or {}
            self.status = f"Isaac preflight OK: {selected.get('executable', selected.get('candidate_path', ''))}"
            self._info("[INFO] " + self.status)
        else:
            self.status = f"[WARN] Isaac preflight failed: {report.get('preflight_error', '')}"
            self._warn(self.status)
        return report

    def restart_sim_worker(self) -> None:
        if self.no_sim:
            self._info("[INFO] --no-sim mode: restart skipped.")
            return
        if self.sim_launch_mode != "subprocess":
            self._warn("[WARN] Restart Isaac Worker is only available in subprocess mode.")
            return
        if self.sim_client is None:
            self.start_sim_if_needed()
            return
        self._info("[INFO] Restarting Isaac subprocess worker...")
        self.sim_client.restart()
        self.status = "Isaac subprocess worker restart requested."

    def restart_sim_worker_without_camera(self) -> None:
        if self.no_sim:
            self._info("[INFO] --no-sim mode: restart without camera skipped.")
            return
        setattr(self.args, "onboard_camera", False)
        setattr(self.args, "enable_cameras", False)
        if self.sim_client is None:
            self.start_sim_if_needed()
            return
        self._info("[INFO] Restarting Isaac subprocess worker without onboard camera...")
        self.sim_client.restart_without_camera()
        self.status = "Isaac subprocess worker restart without onboard camera requested."

    def stop_sim_worker(self) -> None:
        if self.sim_client is not None:
            self._info("[INFO] Stopping Isaac subprocess worker...")
            self.sim_client.shutdown()
            self.sim_client = None
        if self.sim_worker is not None:
            self._info("[INFO] Stopping Isaac thread worker...")
            self.sim_worker.stop()
            self.sim_worker = None
        self.sim_ready = False
        self.transport.attach(self.adapter, ready=False)

    def open_worker_log_folder(self) -> str:
        if self.sim_client is None:
            self._warn("[WARN] Isaac subprocess worker has not been started yet.")
            return ""
        return self.sim_client.open_log_folder()

    def copy_worker_display_command(self) -> str:
        if self.sim_client is None:
            return ""
        return self.sim_client.copy_display_command()

    def generate_or_update_height_obstacle(self) -> None:
        if self.no_sim:
            self.loaded_sim_height_cm = self.current_height_cm
            self.status = f"No-sim: current height set to {self.current_height_cm}cm."
            return
        if self.sim_launch_mode == "subprocess":
            if self.sim_client is None:
                self.start_sim_if_needed()
            if not self.runtime_ready:
                self.pending_height_cm = self.current_height_cm
                self.status = f"Queued obstacle height {self.current_height_cm}cm until Isaac worker is ready."
                self._info(f"[INFO] {self.status}")
                return
            if self.sim_client is not None:
                self.sim_client.set_height(self.current_height_cm)
                self.loaded_sim_height_cm = self.current_height_cm
                self.status = f"Requested obstacle height {self.current_height_cm}cm from Isaac subprocess worker."
                self._info(f"[INFO] {self.status}")
            return
        if self.sim_worker is None:
            self.start_sim_if_needed()
            return
        self.sim_worker.set_height(self.current_height_cm)
        self.loaded_sim_height_cm = self.current_height_cm
        self.status = f"Requested obstacle height {self.current_height_cm}cm from Isaac Sim worker."
        self._info(f"[INFO] {self.status}")

    def config_for_height(self, height_cm: int) -> SimSceneConfig:
        return SimSceneConfig(
            obstacle_height_m=obstacle_height_m(height_cm),
            robot_usd=Path(self.args.robot_usd),
            save_usd=Path(self.args.save_usd),
            spawn_z=float(self.args.spawn_z),
            obstacle_x=float(self.args.obstacle_x),
            obstacle_width=self.args.obstacle_width,
            obstacle_length=self.args.obstacle_length,
            infer_obstacle_size=bool(self.args.infer_obstacle_size),
            robot_width=float(self.args.robot_width),
            robot_length=float(self.args.robot_length),
            physics_dt=float(self.args.physics_dt),
            render_interval=int(self.args.render_interval),
            device=str(getattr(self.args, "device", "cuda:0")),
            max_wheel_speed=self.max_wheel_speed,
            default_wheel_speed=self.default_wheel_speed,
            wheel_direction=float(self.args.wheel_direction),
            servo_stiffness=float(self.args.servo_stiffness),
            servo_damping=float(self.args.servo_damping),
            wheel_damping=float(self.args.wheel_damping),
            save_scene=bool(self.args.save_scene),
            onboard_camera_enabled=bool(getattr(self.args, "onboard_camera", True)),
            camera_parent_prim=str(getattr(self.args, "camera_parent_prim", "") or ""),
            camera_width=int(getattr(self.args, "camera_width", 424)),
            camera_height=int(getattr(self.args, "camera_height", 240)),
            camera_update_period_s=float(getattr(self.args, "camera_update_period_s", 0.10)),
            camera_offset_pos=(
                float(getattr(self.args, "camera_offset_x", 0.35)),
                float(getattr(self.args, "camera_offset_y", 0.0)),
                float(getattr(self.args, "camera_offset_z", 0.18)),
            ),
            camera_offset_rot=camera_pitch_quat_wxyz(float(getattr(self.args, "camera_pitch_deg", 14.0))),
            camera_offset_convention="world",
            camera_aim_mode=str(getattr(self.args, "camera_aim_mode", "pitch") or "pitch"),
            camera_target_x=float(getattr(self.args, "camera_target_x", getattr(self.args, "obstacle_x", 1.55))),
            camera_target_y=float(getattr(self.args, "camera_target_y", 0.0)),
            camera_target_z=float(getattr(self.args, "camera_target_z", 0.02)),
            camera_target_frame=str(getattr(self.args, "camera_target_frame", "world") or "world"),
            camera_look_at_roll_deg=float(getattr(self.args, "camera_look_at_roll_deg", 0.0)),
            camera_focal_length=float(getattr(self.args, "camera_focal_length", 24.0)),
            camera_horizontal_aperture=float(getattr(self.args, "camera_horizontal_aperture", 20.955)),
            camera_near_clip_m=float(getattr(self.args, "camera_near_clip_m", 0.05)),
            camera_far_clip_m=float(getattr(self.args, "camera_far_clip_m", 6.0)),
            camera_coverage_strict=bool(getattr(self.args, "camera_coverage_strict", False)),
            telemetry_contact_sensors_enabled=bool(getattr(self.args, "telemetry_contact_sensors_enabled", False)),
        )

    def set_current_height(
        self,
        value: int | float | str,
        *,
        discard_dirty: bool = False,
        load_steps: bool = True,
        generate_obstacle: bool = True,
    ) -> int:
        height = normalize_height_cm(value)
        if height == self.current_height_cm:
            if generate_obstacle:
                self.generate_or_update_height_obstacle()
            return height
        ok, reason = self.can_switch_height(discard_dirty=discard_dirty)
        if not ok:
            raise DirtyHeightSwitchError(reason)
        self.current_height_cm = height
        self.manager = SequenceManager(self.store.steps_path(height))
        self.pending_step = None
        self.pending_replacement = None
        self.record_events = []
        self.record_command_state_before = None
        self.selected_step_index = None
        self.detail_text = ""
        self.mode = MODE_TEST
        if load_steps:
            self.load_steps_for_current_height()
        if generate_obstacle:
            self.generate_or_update_height_obstacle()
        self.status = f"Current height={height}cm."
        return height

    def load_steps_for_current_height(self) -> int:
        def work() -> int:
            steps = self.store.load_steps(self.current_height_cm)
            self.manager = SequenceManager(self.store.steps_path(self.current_height_cm))
            self.manager.adopt_steps(steps, dirty=False)
            self.selected_step_index = 1 if steps else None
            self.set_step_summary_detail(steps[0], title=f"Loaded step {int(steps[0].get('index', 1)):03d}") if steps else setattr(self, "detail_text", "")
            if steps:
                self._info(f"[INFO] Loaded {len(steps)} saved steps for {self.current_height_cm}cm.")
            else:
                self._warn(f"No saved steps found for {self.current_height_cm}cm. Please record steps first.")
            return len(steps)

        result = self._run_operation("Load steps for current height", work)
        return int(result or 0)

    def save_steps_for_current_height(self, *, allow_empty: bool = False) -> Path | None:
        ok, reason = self.can_save()
        if not ok:
            self._warn("[WARN] " + reason)
            return None

        def work() -> Path | None:
            if self.manager.count <= 0:
                if not allow_empty:
                    self._warn(f"[WARN] Current {self.current_height_cm}cm sequence has no accepted steps to save.")
                    return None
                path = self.store.clear_saved_steps(self.current_height_cm)
                self.manager.accepted_path = path
                self.manager.dirty = False
                self.refresh_manifest()
                self._info(f"[INFO] Cleared saved steps for {self.current_height_cm}cm; previous file overwritten.")
                return path
            path = self.store.save_steps(self.current_height_cm, self.manager.steps)
            self.manager.accepted_path = path
            self.manager.dirty = False
            self.refresh_manifest()
            self._info(f"[INFO] Saved {self.manager.count} steps for {self.current_height_cm}cm; previous file overwritten: {path}")
            return path

        result = self._run_operation("Save steps for current height", work)
        return result if isinstance(result, Path) else None

    def new_empty_sequence_for_current_height(self, *, discard_dirty: bool = False) -> None:
        if self.visible_steps_are_read_only():
            raise RuntimeError("Finish Vision Task before creating a new Height Task sequence.")
        if self.manager.dirty and not discard_dirty:
            raise DirtyHeightSwitchError("Current sequence has unsaved changes. Confirm discard before creating a new empty sequence.")
        self.manager = SequenceManager(self.store.steps_path(self.current_height_cm))
        self.pending_step = None
        self.pending_replacement = None
        self.record_events = []
        self.selected_step_index = None
        self.detail_text = ""
        self.mode = MODE_TEST
        self._info(f"[INFO] New empty sequence for {self.current_height_cm}cm.")

    def start_height_task(self, mode: str, heights: list[int]) -> None:
        if self.vision_task_active:
            raise RuntimeError("Finish Vision Task before starting Height Task.")
        selected = [normalize_height_cm(height) for height in heights]
        if not selected:
            raise ValueError("Select at least one supported height.")
        self.height_task_mode = "replay" if str(mode).lower().startswith("replay") else "recording"
        self.height_task_heights = selected
        self.height_task_index = 0
        self.height_task_active = True
        self.set_current_height(selected[0], discard_dirty=True, load_steps=True, generate_obstacle=True)
        self._info(
            f"[INFO] Started Height {self.height_task_mode.title()} Task: "
            f"{len(selected)} height(s), current={selected[0]}cm."
        )

    def finish_height_task(self) -> None:
        self.height_task_active = False
        self.height_task_index = -1
        self._info("[INFO] Height task finished.")

    def start_vision_task(self, source_mode: str | None = None) -> bool:
        ok, reason = self.can_start_vision_task()
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        self.vision_source_mode = normalize_source_mode(source_mode or self.vision_source_mode)
        self.vision_task = ready_state(self.vision_source_mode)
        self.vision_task_active = True
        self.transport.set_vision_source_mode("external" if self.vision_source_mode == VISION_SOURCE_EXTERNAL else "generated")
        self._clear_vision_steps("Waiting for a validated Vision detection.")
        self.vision_generated_height_cm = None
        self.vision_generation_request_id = ""
        self.vision_scene_ready = False
        self.vision_scene_obstacle_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0)
        self.vision_detection_baseline_revision = int((self.latest_sim_status.get("vision", {}) or {}).get("detection_revision", 0) or 0) if isinstance(self.latest_sim_status.get("vision", {}), dict) else 0
        self.vision_frame_baseline_revision = int((self.latest_sim_status.get("vision", {}) or {}).get("frame_revision", 0) or 0) if isinstance(self.latest_sim_status.get("vision", {}), dict) else 0
        self.selected_step_index = None
        self.vision_last_auto_action = "vision task started"
        self._info("[INFO] Vision Task started.")
        return True

    def finish_vision_task(self) -> None:
        self.vision_task = inactive_state(self.vision_source_mode)
        self.vision_task_active = False
        self._clear_vision_steps("Vision Task inactive.")
        self.vision_generated_height_cm = None
        self.vision_generation_request_id = ""
        self.vision_scene_ready = False
        self.selected_step_index = 1 if self.manager.count else None
        self.vision_auto_replay_armed = False
        self.vision_last_auto_action = "vision task finished"
        self._info("[INFO] Vision Task finished; Height Task steps restored.")

    def _next_camera_view_request(self, action: str, *, active_fallback_allowed: bool | None = None) -> dict[str, Any]:
        self.camera_view_action_revision += 1
        revision = int(self.camera_view_action_revision)
        request_id = f"camera-view-{revision}-{uuid.uuid4().hex[:8]}"
        self.camera_view_pending_revision = revision
        self.camera_view_pending_request_id = request_id
        self.camera_view_pending_action = str(action)
        allowed, reason = self.can_change_camera_view()
        return {
            "request_id": request_id,
            "action_revision": revision,
            "active_fallback_allowed": bool(getattr(self.args, "camera_view_active_fallback", False)) if active_fallback_allowed is None else bool(active_fallback_allowed),
            "camera_view_pending_timeout_s": float(getattr(self.args, "camera_view_pending_timeout_s", 10.0)),
            "camera_view_pending_max_retries": int(getattr(self.args, "camera_view_pending_max_retries", 30)),
            "camera_view_guard_flags": self._camera_view_guard_flags(),
            "camera_view_guard_allowed": bool(allowed),
            "camera_view_guard_reason": str(reason),
        }

    def _camera_view_guard_flags(self) -> dict[str, Any]:
        return {
            "recording_active": bool(self.recording_active),
            "playback_active": bool(self.playback.active),
            "playback_scheduled": bool(getattr(self.playback, "scheduled_start_at", 0.0) or 0.0),
            "operation_busy": bool(self.operation_busy),
            "pending_step": self.pending_step is not None,
            "pending_replacement": self.pending_replacement is not None,
            "e_stop_exception": False,
        }

    def can_change_camera_view(self) -> tuple[bool, str]:
        if not self.runtime_ready:
            return False, "Isaac runtime is not ready."
        if bool(self.latest_sim_status.get("effective_headless", bool(getattr(self.args, "headless", False)))):
            return False, "Camera viewport is unavailable in headless mode."
        command_state = self.transport.capture_command_state()
        guard = ground_can_change_camera_view(
            flags=self._camera_view_guard_flags(),
            wheel_command_values=command_state.get("wheels", {}),
        )
        return bool(guard.allowed), "; ".join(guard.reasons)

    def _sync_camera_view_ack(self, vision: dict[str, Any]) -> None:
        camera_view = vision.get("camera_view", {}) if isinstance(vision.get("camera_view"), dict) else {}
        completed = int(camera_view.get("completed_revision", 0) or 0)
        if self.camera_view_pending_revision <= 0 or completed < self.camera_view_pending_revision:
            return
        if completed == self.camera_view_last_completed_revision:
            return
        self.camera_view_last_completed_revision = completed
        self.camera_view_pending_revision = 0
        error = str(camera_view.get("error", "") or "")
        action = str(camera_view.get("requested_action", self.camera_view_pending_action) or self.camera_view_pending_action)
        if error:
            self._warn("[WARN] Camera viewport action failed: " + error)
        else:
            self._info(f"[INFO] Camera viewport action completed: {action}.")

    def open_onboard_camera_viewport(self) -> bool:
        vision = self.latest_sim_status.get("vision", {}) if isinstance(self.latest_sim_status.get("vision"), dict) else {}
        ready, reason = self.camera_view_readiness()
        if not ready:
            self._warn("[WARN] Camera viewport is not ready: " + reason)
            return False
        camera_path = str(vision.get("camera_prim_path", "") or self.latest_sim_status.get("camera_prim_path", "") or "")
        payload = self._next_camera_view_request("open_camera_viewport")
        self.transport.open_camera_viewport(camera_prim_path=camera_path, **payload)
        self.vision_last_auto_action = "camera viewport open requested"
        self._info("[INFO] Requested onboard camera viewport; waiting for worker ACK.")
        return True

    def show_camera_in_isaac_sim(self) -> bool:
        return self.open_onboard_camera_viewport()

    def show_camera_in_main_view_fallback(self) -> bool:
        vision = self.latest_sim_status.get("vision", {}) if isinstance(self.latest_sim_status.get("vision"), dict) else {}
        ready, reason = self.camera_view_readiness()
        if not ready:
            self._warn("[WARN] Camera viewport fallback is not ready: " + reason)
            return False
        camera_path = str(vision.get("camera_prim_path", "") or self.latest_sim_status.get("camera_prim_path", "") or "")
        payload = self._next_camera_view_request("open_camera_viewport", active_fallback_allowed=True)
        self.transport.open_camera_viewport(camera_prim_path=camera_path, **payload)
        self.vision_last_auto_action = "camera main-view fallback requested"
        self._info("[INFO] Requested explicit main-view camera fallback; waiting for worker ACK.")
        return True

    def return_main_view_to_perspective(self) -> bool:
        payload = self._next_camera_view_request("return_main_view_to_perspective")
        self.transport.return_main_view_to_perspective(**payload)
        self.vision_last_auto_action = "return main view requested"
        self._info("[INFO] Requested main Viewport return to Perspective; waiting for worker ACK.")
        return True

    def close_onboard_camera_viewport(self) -> bool:
        payload = self._next_camera_view_request("close_camera_viewport")
        self.transport.close_camera_viewport(**payload)
        self.vision_last_auto_action = "close camera viewport requested"
        self._info("[INFO] Requested Onboard Camera viewport close; waiting for worker ACK.")
        return True

    def restore_camera_view(self) -> bool:
        payload = self._next_camera_view_request("restore_camera_view")
        self.transport.restore_camera_view(**payload)
        self.vision_last_auto_action = "restore camera viewport requested"
        self._info("[INFO] Requested Isaac viewport restore; waiting for worker ACK.")
        return True

    def validate_robot_ground_contact(self) -> bool:
        self.transport.validate_robot_ground_contact()
        self.vision_last_auto_action = "robot ground validation requested"
        self._info("[INFO] Requested robot ground-contact validation.")
        return True

    def calibrate_ground_reference(self) -> bool:
        if self.recording_active or self.playback.active or self.operation_busy:
            self._warn("[WARN] Calibrate Ground Reference requires idle recording/playback state.")
            return False
        self.transport.calibrate_ground_reference()
        self.vision_last_auto_action = "ground reference calibration requested"
        self._info("[INFO] Requested ground reference calibration.")
        return True

    def respawn_and_validate_ground(self) -> bool:
        if self.recording_active or self.playback.active or self.operation_busy:
            self._warn("[WARN] Respawn And Validate Ground requires idle recording/playback state.")
            return False
        self.transport.respawn_and_validate_ground()
        self.vision_last_auto_action = "respawn and ground validation requested"
        self._info("[INFO] Requested robot respawn and ground-contact validation.")
        return True

    def validate_camera_geometry(self) -> bool:
        self.transport.validate_camera_geometry()
        self.vision_last_auto_action = "camera geometry validation requested"
        self._info("[INFO] Isaac camera geometry validation requested.")
        return True

    def apply_recommended_oblique_camera_pose(self) -> bool:
        target_x = float(getattr(self.args, "obstacle_x", 1.55))
        setattr(self.args, "camera_aim_mode", "look-at")
        setattr(self.args, "camera_target_x", target_x)
        setattr(self.args, "camera_target_y", 0.0)
        setattr(self.args, "camera_target_z", 0.02)
        setattr(self.args, "camera_target_frame", "world")
        setattr(self.args, "camera_look_at_roll_deg", 0.0)
        setattr(self.args, "onboard_camera", True)
        setattr(self.args, "enable_cameras", True)
        self.vision_last_auto_action = "recommended oblique camera pose staged"
        self._info(
            "[INFO] Staged recommended oblique camera pose: "
            f"--camera-aim-mode look-at --camera-target-x {target_x:.3f} --camera-target-y 0.000 --camera-target-z 0.020. "
            "Restart Worker With Camera Pose to recreate the camera sensor."
        )
        return True

    def restart_sim_worker_with_camera_pose(self) -> None:
        setattr(self.args, "onboard_camera", True)
        setattr(self.args, "enable_cameras", True)
        self._info("[INFO] Restarting Isaac worker with staged camera pose parameters...")
        self.restart_sim_worker()

    def generate_vision_test_obstacle(self, height_cm: int) -> bool:
        if not self.vision_task_active:
            self._warn("[WARN] Start Vision Task before generating a Vision test obstacle.")
            return False
        if self.vision_source_mode != VISION_SOURCE_GENERATED:
            self._warn("[WARN] Vision obstacle generation is only available in Generated Test Obstacle mode.")
            return False
        try:
            height = normalize_height_cm(height_cm)
        except Exception as exc:
            self._warn("[WARN] " + str(exc))
            return False
        ok, reason = self._can_generate_vision_obstacle()
        if not ok:
            self._warn("[WARN] " + reason)
            self._set_vision_blocked(reason)
            return False
        self.stop_wheels()
        self.vision_auto_replay_armed = False
        self.reset_vision_filter(clear_steps=True)
        self.clear_vision_validation_result(clear_steps=True)
        self._clear_vision_steps("Waiting for a validated Vision detection.")
        self.vision_generation_revision += 1
        self.vision_generation_request_id = f"vision-{self.vision_generation_revision}-{uuid.uuid4().hex[:8]}"
        self.vision_generated_height_cm = height
        self.vision_generation_requested_at = time.time()
        vision = self.latest_sim_status.get("vision", {}) if isinstance(self.latest_sim_status.get("vision"), dict) else {}
        self.vision_detection_baseline_revision = int(vision.get("detection_revision", 0) or 0)
        self.vision_frame_baseline_revision = int(vision.get("frame_revision", 0) or 0)
        self.vision_scene_obstacle_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0) + 1
        self.vision_scene_ready = False
        self.vision_task.phase = PHASE_GENERATING_OBSTACLE
        self.vision_task.last_action = f"generate {height}cm"
        if self.no_sim:
            self.loaded_sim_height_cm = height
            self.vision_scene_ready = True
            self.vision_task.phase = PHASE_DETECTING
            self._info(f"[INFO] No-sim: Vision test obstacle marked as {height}cm.")
            return True
        if self.sim_launch_mode == "subprocess":
            if self.sim_client is None:
                self.start_sim_if_needed()
            if not self.runtime_ready:
                self._warn("[WARN] Isaac worker is not ready for Vision obstacle generation.")
                self.vision_task.phase = PHASE_WAITING_FOR_SCENE
                return False
            if self.sim_client is not None:
                self.sim_client.set_height(
                    height,
                    source="vision_task",
                    request_id=self.vision_generation_request_id,
                    generation_revision=self.vision_generation_revision,
                    respawn_policy="if_motion_ready",
                )
        elif self.sim_worker is not None:
            self.sim_worker.set_height(
                height,
                source="vision_task",
                request_id=self.vision_generation_request_id,
                generation_revision=self.vision_generation_revision,
                respawn_policy="if_motion_ready",
            )
        else:
            self.start_sim_if_needed()
            return False
        self.loaded_sim_height_cm = height
        self.vision_task.phase = PHASE_WAITING_FOR_SCENE
        self._info(f"[INFO] Requested Vision test obstacle {height}cm.")
        return True

    def _can_generate_vision_obstacle(self) -> tuple[bool, str]:
        if self.recording_active:
            return False, "Stop recording before generating a Vision test obstacle."
        if self.pending_step is not None:
            return False, "Accept or discard the pending recorded step before generating a Vision test obstacle."
        if self.pending_replacement is not None:
            return False, "Accept or discard the pending replacement before generating a Vision test obstacle."
        if self.playback.active:
            return False, "Stop playback before generating a Vision test obstacle."
        if self.mode == MODE_E_STOP:
            return False, "E-stop is active. Clear E-stop before generating a Vision test obstacle."
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before generating a Vision test obstacle."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        return True, ""

    def load_validated_vision_steps(self, height_cm: int, detection_revision: int) -> int:
        height = normalize_height_cm(height_cm)
        self.vision_task.phase = PHASE_STEPS_LOADING
        path = self.store.steps_path(height)
        steps = self.store.load_steps(height)
        self.vision_steps_path = path
        self.vision_steps_height_cm = height
        self.vision_steps_detection_revision = int(detection_revision)
        self.vision_steps = [normalize_step(step, index=index) for index, step in enumerate(steps, start=1)]
        self.vision_steps_ready = bool(self.vision_steps)
        self.vision_steps_error = "" if self.vision_steps_ready else f"No saved steps found for {height}cm. Please record steps first."
        self.vision_steps_revision += 1
        self.selected_step_index = 1 if self.vision_steps_ready else None
        self.vision_task.phase = PHASE_STEPS_READY if self.vision_steps_ready else PHASE_VALIDATED
        if self.vision_steps_ready:
            self._info(f"[INFO] Loaded {len(self.vision_steps)} Vision step(s) for validated {height}cm: {path}")
        else:
            self._warn(self.vision_steps_error)
        return len(self.vision_steps)

    def next_height(self, *, save_current: bool = False) -> bool:
        if save_current:
            if self.save_steps_for_current_height() is None:
                return False
        if not self.height_task_active:
            self._warn("[WARN] No active height task.")
            return False
        if self.height_task_index + 1 >= len(self.height_task_heights):
            self._warn("[WARN] Already at the last selected height.")
            return False
        self.height_task_index += 1
        self.set_current_height(self.height_task_heights[self.height_task_index], discard_dirty=True, load_steps=True, generate_obstacle=True)
        return True

    def previous_height(self) -> bool:
        if not self.height_task_active:
            self._warn("[WARN] No active height task.")
            return False
        if self.height_task_index <= 0:
            self._warn("[WARN] Already at the first selected height.")
            return False
        self.height_task_index -= 1
        self.set_current_height(self.height_task_heights[self.height_task_index], discard_dirty=True, load_steps=True, generate_obstacle=True)
        return True

    def auto_replay_current_height(self) -> bool:
        self.generate_or_update_height_obstacle()
        if self.manager.count == 0:
            self.load_steps_for_current_height()
        if self.manager.count == 0:
            return False
        return self.start_playback(self.manager.steps, label=f"{self.current_height_cm}cm accepted steps")

    def respawn_and_auto_replay_current_height(self) -> bool:
        self.generate_or_update_height_obstacle()
        if self.manager.count == 0:
            self.load_steps_for_current_height()
        if self.manager.count == 0:
            return False
        return self.start_playback(
            self.manager.steps,
            label=f"{self.current_height_cm}cm accepted steps",
            respawn_first=True,
        )

    def auto_replay_selected_heights(self, heights: list[int]) -> None:
        for height in heights:
            self.set_current_height(height, discard_dirty=True, load_steps=True, generate_obstacle=True)
            if self.manager.count == 0:
                self._warn(f"No saved steps found for {height}cm. Please record steps first.")
                continue
            self.start_playback(self.manager.steps, label=f"{height}cm accepted steps")
            break

    def set_vision_enabled(self, enabled: bool) -> None:
        self.vision_enabled = bool(enabled)
        self.transport.set_vision_enabled(self.vision_enabled)
        self.vision_last_auto_action = "camera enable requested" if self.vision_enabled else "camera disable requested"
        self._info(f"[INFO] Vision camera {'enabled' if self.vision_enabled else 'disabled'}.")

    def request_vision_detection_once(self) -> None:
        self.transport.request_vision_detection_once()
        self.vision_last_auto_action = "detect_once requested"
        self._info("[INFO] Vision detect_once requested.")

    def reset_vision_filter(self, *, clear_steps: bool = True) -> None:
        self.transport.reset_vision_filter()
        self.vision_last_auto_action = "filter reset requested"
        self.vision_last_block_reason = ""
        self.vision_pending_validation_revision = 0
        self.vision_pending_validation_expected_cm = None
        if clear_steps:
            self._clear_vision_steps("Waiting for a validated Vision detection.")
        self._info("[INFO] Vision detection filter reset.")

    def request_vision_debug_frame(self) -> None:
        self.transport.save_vision_debug_frame()
        self.vision_last_auto_action = "debug frame requested"
        self._info("[INFO] Vision debug frame save requested.")

    def validate_isaac_camera(self) -> None:
        self.transport.validate_camera()
        self.vision_last_auto_action = "camera validation requested"
        self._info("[INFO] Isaac camera validation requested.")

    def validate_current_generated_height(self) -> bool:
        ok, reason = self.can_validate_current_generated_height()
        if not ok:
            self._set_vision_blocked(reason)
            self.vision_last_auto_action = "height validation blocked"
            self._warn("[WARN] " + reason)
            return False
        expected = int(self.vision_generated_height_cm)
        vision = self.latest_sim_status.get("vision", {}) if isinstance(self.latest_sim_status.get("vision"), dict) else {}
        self.vision_pending_validation_revision = int(vision.get("detection_revision", 0) or 0)
        self.vision_pending_validation_expected_cm = expected
        self.vision_task.phase = PHASE_VALIDATING
        self.transport.validate_current_height(expected)
        self.vision_last_auto_action = f"height validation requested for {expected}cm"
        self._info(f"[INFO] Height validation requested for Vision generated height {expected}cm.")
        return True

    def can_validate_current_generated_height(self) -> tuple[bool, str]:
        if not self.vision_task_active:
            return False, "Start Vision Task first."
        if self.vision_source_mode != VISION_SOURCE_GENERATED:
            return False, "Generated-height validation is disabled in External / Unknown Obstacle mode."
        if self.vision_generated_height_cm is None:
            return False, "Generate a Vision test obstacle first."
        vision = self.latest_sim_status.get("vision", {})
        if not isinstance(vision, dict):
            return False, "Vision status unavailable."
        if not bool(vision.get("camera_ready", False)):
            return False, "Camera health not ready."
        if not bool(vision.get("stable", False)):
            return False, "Waiting for a new stable detection."
        detected = vision.get("detected_height_cm")
        if detected is None:
            return False, "Stable detected height is missing."
        confidence = float(vision.get("confidence", 0.0) or 0.0)
        threshold = float(getattr(self.args, "vision_confidence_threshold", 0.75))
        if confidence < threshold:
            return False, f"Confidence below threshold: {confidence:.2f} < {threshold:.2f}."
        detection_revision = int(vision.get("detection_revision", 0) or 0)
        frame_revision = int(vision.get("frame_revision", 0) or 0)
        if detection_revision <= int(self.vision_detection_baseline_revision):
            return False, "Waiting for a new stable detection."
        if frame_revision <= int(self.vision_frame_baseline_revision):
            return False, "Waiting for a new Camera frame."
        scene_height = self.latest_sim_status.get("scene_height_cm", self.latest_sim_status.get("height_cm"))
        if scene_height is None or int(scene_height) != int(self.vision_generated_height_cm):
            return False, "Waiting for worker scene confirmation."
        obstacle_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0)
        if obstacle_revision < int(self.vision_scene_obstacle_revision):
            return False, "Waiting for worker scene confirmation."
        request_id = str(self.latest_sim_status.get("last_set_height_request_id", "") or "")
        if self.vision_generation_request_id and request_id not in {"", self.vision_generation_request_id}:
            return False, "Worker scene confirmation belongs to a different Vision generation request."
        if int(detected) != int(self.vision_generated_height_cm):
            return False, f"Detected height differs from generated height: {int(detected)}cm != {int(self.vision_generated_height_cm)}cm."
        return True, ""

    def request_rgbd_diagnostic(self) -> None:
        self.transport.save_rgbd_diagnostic()
        self.vision_last_auto_action = "RGB-D diagnostic requested"
        self._info("[INFO] RGB-D diagnostic save requested.")

    def clear_vision_validation_result(self, *, clear_steps: bool = True) -> None:
        self.transport.clear_validation_result()
        self.vision_last_auto_action = "validation result cleared"
        self.vision_last_block_reason = ""
        self.vision_pending_validation_revision = 0
        self.vision_pending_validation_expected_cm = None
        if clear_steps:
            self._clear_vision_steps("Waiting for a validated Vision detection.")
        self._info("[INFO] Vision validation result cleared.")

    def validate_mount_after_next_respawn(self) -> None:
        self.transport.validate_mount_after_next_respawn()
        self.vision_last_auto_action = "mount validation armed"
        self._info("[INFO] Camera mount validation armed. Use Respawn when ready to complete the check.")

    def open_vision_debug_folder(self) -> str:
        vision = self.latest_sim_status.get("vision", {})
        folder = ""
        if isinstance(vision, dict):
            folder = str(vision.get("debug_folder", "") or "")
            if not folder:
                debug_path = str(vision.get("debug_image_path", "") or vision.get("debug_sidecar_path", "") or "")
                if debug_path:
                    folder = str(Path(debug_path).parent)
        if not folder:
            folder = str(Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug")
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            self._warn(f"[WARN] Could not open vision debug folder: {exc}")
        self._info(f"[INFO] Vision debug folder: {path}")
        return str(path)

    def set_vision_auto_replay_enabled(self, enabled: bool) -> None:
        self.vision_auto_replay_enabled = bool(enabled)
        if not self.vision_auto_replay_enabled:
            self.disarm_vision_auto_replay()
        self.vision_last_auto_action = "auto replay enabled" if enabled else "auto replay disabled"
        self._info(f"[INFO] Vision auto replay {'enabled' if enabled else 'disabled'}.")

    def arm_vision_auto_replay(self) -> None:
        if self.mode == MODE_E_STOP:
            self.vision_last_block_reason = "E-stop is active. Re-arm after clearing E-stop."
            self.vision_last_auto_action = "arm blocked"
            self._warn("[WARN] " + self.vision_last_block_reason)
            return
        self.vision_auto_replay_armed = True
        self.vision_last_block_reason = ""
        self.vision_last_auto_action = "armed"
        self._info("[INFO] Vision auto replay armed.")

    def disarm_vision_auto_replay(self) -> None:
        self.vision_auto_replay_armed = False
        self.vision_last_auto_action = "disarmed"

    def can_auto_replay_detected_height(
        self,
        height_cm: int | None,
        detection_revision: int | None,
        *,
        require_auto_replay: bool = True,
    ) -> tuple[bool, str]:
        if require_auto_replay and not self.vision_auto_replay_enabled:
            return False, "Vision auto replay is disabled."
        if require_auto_replay and not self.vision_auto_replay_armed:
            return False, "Vision auto replay is disarmed."
        if self.no_sim or not self.runtime_ready:
            return False, "Isaac runtime is not ready."
        motion_ok, motion_reason = self.motion_readiness()
        if not motion_ok:
            return False, motion_reason or "Motion is blocked by ground safety."
        if bool(self.vision_respawn_before_replay):
            respawn_ok, respawn_reason = self.respawn_readiness()
            if not respawn_ok:
                return False, (respawn_reason or "Respawn is not ready.") + " Disable Respawn Before Auto Replay or calibrate a valid ground reference."
        vision = self.latest_sim_status.get("vision", {})
        if not isinstance(vision, dict):
            return False, "Vision status unavailable."
        if not bool(vision.get("camera_ready", False)):
            return False, "Vision camera is not ready."
        if not bool(vision.get("stable", False)):
            return False, "Vision detection is not stable yet."
        revision = int(detection_revision or 0)
        if revision <= 0:
            return False, "Vision detection revision is not ready."
        if require_auto_replay and revision == self.vision_last_consumed_detection_revision:
            return False, "Vision detection revision was already consumed."
        if height_cm not in SUPPORTED_HEIGHTS_CM:
            return False, f"Detected height is not supported: {height_cm}"
        confidence = float(vision.get("confidence", 0.0) or 0.0)
        threshold = float(getattr(self.args, "vision_confidence_threshold", 0.75))
        if confidence < threshold:
            return False, f"Vision confidence {confidence:.2f} is below {threshold:.2f}."
        if self.recording_active:
            return False, "Recording is active."
        if self.pending_step is not None:
            return False, "Accept or discard the pending recorded step first."
        if self.pending_replacement is not None:
            return False, "Accept or discard the pending replacement first."
        if self.operation_busy:
            return False, f"Operation busy: {self.busy_name}"
        if self.playback.active:
            return False, "Playback is already active."
        if self.mode == MODE_E_STOP:
            return False, "E-stop is active. Re-arm after clearing E-stop."
        if self.mode not in {MODE_TEST, MODE_SERVO_WHEEL}:
            return False, f"Playback is only valid in TEST or SERVO_WHEEL. Current mode={self.mode}"
        if self.servo_wheel_staging_active and self.servo_wheel_staged_dirty:
            return False, "Launch, clear, or cancel staged Servo+Wheel command before auto replay."
        generated_requires_validation = self.vision_task_active and self.vision_source_mode == VISION_SOURCE_GENERATED
        if generated_requires_validation:
            validation = vision.get("height_validation", {}) if isinstance(vision.get("height_validation"), dict) else {}
            if not bool(validation.get("checked", False)) or not bool(validation.get("passed", False)):
                return False, "Generated Test Obstacle mode requires validation PASS before auto replay."
            if int(validation.get("expected_height_cm", -1)) != int(height_cm):
                return False, "Validation expected height does not match detected height."
            if self.vision_steps_detection_revision != revision:
                return False, "Vision steps are not loaded for this detection revision."
        if not self.vision_steps_ready:
            if not generated_requires_validation and self.store.has_saved_steps(int(height_cm)):
                self.load_validated_vision_steps(int(height_cm), revision)
            if not self.vision_steps_ready:
                return False, self.vision_steps_error or f"No saved steps found for {int(height_cm)}cm. Please record steps first."
        if self.vision_steps_height_cm != int(height_cm):
            return False, "Loaded Vision steps height does not match detected height."
        cooldown = float(self.vision_auto_replay_cooldown_s)
        if require_auto_replay and cooldown > 0.0 and self.vision_last_replay_started_at > 0.0:
            elapsed = time.monotonic() - self.vision_last_replay_started_at
            if elapsed < cooldown:
                return False, f"Vision auto replay cooldown: {cooldown - elapsed:.1f}s remaining."
        return True, ""

    def replay_detected_height(self, height_cm: int, detection_revision: int) -> bool:
        return self.replay_validated_vision_steps(height_cm, detection_revision, require_auto_replay=True)

    def replay_validated_vision_steps(self, height_cm: int, detection_revision: int, *, require_auto_replay: bool = False) -> bool:
        height = normalize_height_cm(height_cm)
        revision = int(detection_revision)
        if require_auto_replay:
            ok, reason = self.can_auto_replay_detected_height(height, revision, require_auto_replay=True)
            if not ok:
                self.vision_last_block_reason = reason
                self.vision_last_auto_action = "blocked"
                self._warn("[WARN] " + reason)
                return False
        else:
            ok, reason = self.playback_readiness(respawn_first=bool(self.vision_respawn_before_replay))
            if not ok:
                self.vision_last_block_reason = reason
                self.vision_last_auto_action = "manual play blocked"
                self._warn("[WARN] " + reason)
                return False
            if self.vision_task_active and self.vision_source_mode == VISION_SOURCE_GENERATED:
                validation = (self.latest_sim_status.get("vision", {}) or {}).get("height_validation", {}) if isinstance(self.latest_sim_status.get("vision", {}), dict) else {}
                if not bool(validation.get("checked", False)) or not bool(validation.get("passed", False)):
                    reason = "Generated Test Obstacle mode requires validation PASS before playback."
                    self.vision_last_block_reason = reason
                    self.vision_last_auto_action = "manual play blocked"
                    self._warn("[WARN] " + reason)
                    return False
                if int(validation.get("expected_height_cm", -1)) != int(height):
                    reason = "Validation expected height does not match requested Vision steps."
                    self.vision_last_block_reason = reason
                    self.vision_last_auto_action = "manual play blocked"
                    self._warn("[WARN] " + reason)
                    return False
        self.stop_wheels()
        if not self.vision_steps_ready:
            count = self.load_validated_vision_steps(height, revision)
            if count <= 0:
                reason = self.vision_steps_error or f"No saved steps found for {height}cm. Please record steps first."
                self.vision_last_block_reason = reason
                self.vision_last_auto_action = "blocked"
                self._warn("[WARN] " + reason)
                return False
        if not self.vision_steps:
            reason = f"No Vision steps loaded for {height}cm."
            self.vision_last_block_reason = reason
            self.vision_last_auto_action = "blocked"
            self._warn("[WARN] " + reason)
            return False
        started = self.start_playback(
            self.vision_steps,
            label=f"{height}cm validated vision steps",
            respawn_first=bool(self.vision_respawn_before_replay),
        )
        if started:
            self.vision_last_consumed_detection_revision = revision
            self.vision_last_replay_started_at = time.monotonic()
            self.vision_last_detected_height_cm = height
            self.vision_last_block_reason = ""
            self.vision_auto_replay_armed = False
            self.vision_task.phase = PHASE_PLAYING
            self.vision_last_auto_action = f"started Vision playback for {height}cm revision {revision}; disarmed"
        return started

    def _maybe_load_steps_after_validation(self, vision: dict[str, Any]) -> None:
        if not self.vision_task_active or self.vision_source_mode != VISION_SOURCE_GENERATED:
            return
        validation = vision.get("height_validation", {}) if isinstance(vision.get("height_validation"), dict) else {}
        if not bool(validation.get("checked", False)):
            return
        expected = validation.get("expected_height_cm")
        detected = validation.get("detected_height_cm")
        revision = int(validation.get("detection_revision", vision.get("detection_revision", 0)) or 0)
        if not bool(validation.get("passed", False)):
            self.vision_task.phase = PHASE_VALIDATION_FAILED
            self._clear_vision_steps("Vision validation failed.")
            self.vision_last_block_reason = "; ".join(str(item) for item in validation.get("reasons", [])) or "Vision validation failed."
            return
        if expected is None or detected is None or self.vision_generated_height_cm is None:
            return
        if int(expected) != int(self.vision_generated_height_cm) or int(detected) != int(expected):
            self.vision_task.phase = PHASE_VALIDATION_FAILED
            self._clear_vision_steps("Validation height did not match generated obstacle.")
            return
        if revision <= int(self.vision_detection_baseline_revision):
            self.vision_task.phase = PHASE_VALIDATION_FAILED
            self._clear_vision_steps("Validation used a stale detection revision.")
            return
        if self.vision_steps_ready and self.vision_steps_detection_revision == revision and self.vision_steps_height_cm == int(expected):
            return
        self.vision_validated_height_cm = int(expected)
        self.vision_validated_detection_revision = revision
        self.vision_task.phase = PHASE_VALIDATED
        self.load_validated_vision_steps(int(expected), revision)
        if self.vision_steps_ready and self.vision_auto_replay_enabled and self.vision_auto_replay_armed:
            self.replay_validated_vision_steps(int(expected), revision, require_auto_replay=True)

    def _maybe_auto_replay_from_vision(self) -> None:
        vision = self.latest_sim_status.get("vision", {})
        if not isinstance(vision, dict):
            return
        revision = int(vision.get("detection_revision", 0) or 0)
        height = vision.get("detected_height_cm")
        if revision <= 0 or height is None:
            return
        self.vision_last_detection_revision = revision
        self.vision_last_detected_height_cm = int(height)
        if self.vision_task_active and self.vision_source_mode == VISION_SOURCE_GENERATED:
            self._maybe_load_steps_after_validation(vision)
            return
        if (not self.vision_task_active or self.vision_source_mode == VISION_SOURCE_EXTERNAL) and self.vision_auto_replay_enabled and self.vision_auto_replay_armed:
            if not self.vision_steps_ready or self.vision_steps_height_cm != int(height) or self.vision_steps_detection_revision != revision:
                self.load_validated_vision_steps(int(height), revision)
        ok, reason = self.can_auto_replay_detected_height(int(height), revision, require_auto_replay=True)
        if not ok:
            self.vision_last_block_reason = reason
            return
        self.replay_validated_vision_steps(int(height), revision, require_auto_replay=True)

    def handle_command(self, message: CommandMessage | str) -> None:
        text = message.text if isinstance(message, CommandMessage) else str(message)
        source = message.source if isinstance(message, CommandMessage) else "ui"
        command_message = message if isinstance(message, CommandMessage) else None
        for command in split_semicolon_commands(text):
            try:
                tokens = command.split()
                if not tokens:
                    continue
                self._handle_tokens(tokens, command, source, command_message)
            except Exception as exc:
                self._warn(f"[WARN] Command failed: {command}: {exc}")

    def stop_wheels(self) -> None:
        self.transport.stop_wheels()

    def shutdown(self) -> None:
        self.playback.stop(silent=True, stop_wheels=False)
        try:
            self.transport.stop_wheels()
        except Exception:
            pass
        if self.sim_client is not None:
            self.sim_client.shutdown()
            self.sim_client = None
        if self.sim_worker is not None:
            self.sim_worker.stop()
            self.sim_worker = None

    def update(self) -> None:
        self.playback.update()
        if self.playback.active:
            self.mode = MODE_PLAYBACK_PAUSED if self.playback.paused else MODE_PLAYBACK
        elif self.mode in {MODE_PLAYBACK, MODE_PLAYBACK_PAUSED}:
            self.mode = MODE_TEST
        if self.sim_client is not None:
            self.sim_client.poll()
            status = self.sim_client.status()
            self.latest_sim_status = status
            self.transport.update_worker_status(status)
            self.sim_ready = bool(status.get("runtime_ready", status.get("ready", False)))
            if status.get("height_cm") is not None:
                self.loaded_sim_height_cm = int(status["height_cm"])
            error = str(status.get("error", "") or "")
            warning = str(status.get("startup_timeout_warning", "") or status.get("status_timeout_warning", "") or "")
            if error:
                self.status = f"[WARN] Isaac subprocess worker: {error}"
            elif warning:
                self.status = f"[WARN] {warning}"
            if bool(getattr(self.args, "ui", False)) and bool(status.get("effective_headless", False)) and not self._warned_gui_headless:
                self._warn("[WARN] GUI was requested but AppLauncher resolved to headless mode.")
                self._warned_gui_headless = True
            if self.sim_ready and self.pending_height_cm is not None:
                pending = self.pending_height_cm
                self.pending_height_cm = None
                self.sim_client.set_height(pending)
                self.loaded_sim_height_cm = pending
        if self.sim_worker is not None:
            statuses = self.sim_worker.drain_status()
            if statuses:
                status = statuses[-1]
                self.latest_sim_status = status
                self.transport.update_worker_status(status)
                self.sim_ready = bool(status.get("runtime_ready", status.get("ready", False)))
                if status.get("height_cm") is not None:
                    self.loaded_sim_height_cm = int(status["height_cm"])
                error = str(status.get("error", "") or "")
                if error:
                    self.status = f"[WARN] Isaac Sim worker: {error}"
        vision = self.latest_sim_status.get("vision", {})
        if isinstance(vision, dict):
            self.vision_enabled = bool(vision.get("enabled", self.vision_enabled))
            revision = int(vision.get("detection_revision", 0) or 0)
            if revision > 0:
                self.vision_last_detection_revision = revision
            detected = vision.get("detected_height_cm")
            if detected is not None:
                try:
                    self.vision_last_detected_height_cm = int(detected)
                except (TypeError, ValueError):
                    pass
            self._sync_camera_view_ack(vision)
        if self.vision_task_active and self.vision_generated_height_cm is not None:
            scene_height = self.latest_sim_status.get("scene_height_cm", self.latest_sim_status.get("height_cm"))
            obstacle_revision = int(self.latest_sim_status.get("obstacle_revision", 0) or 0)
            request_id = str(self.latest_sim_status.get("last_set_height_request_id", "") or "")
            self.vision_scene_ready = (
                scene_height is not None
                and int(scene_height) == int(self.vision_generated_height_cm)
                and obstacle_revision >= int(self.vision_scene_obstacle_revision)
                and (not self.vision_generation_request_id or request_id in {"", self.vision_generation_request_id})
            )
            if self.vision_scene_ready and self.vision_task.phase in {PHASE_GENERATING_OBSTACLE, PHASE_WAITING_FOR_SCENE}:
                self.vision_task.phase = PHASE_DETECTING
        self._sync_vision_task_state()
        self._maybe_auto_replay_from_vision()

    def snapshot(self) -> dict[str, Any]:
        playback = self.playback.status_dict()
        state = self.transport.capture_command_state()
        wheels_short = {
            short: state["wheels"].get(full, 0.0)
            for short, full in WHEEL_SHORT_NAMES.items()
        }
        task_total = len(self.height_task_heights)
        task_pos = self.height_task_index + 1 if self.height_task_active and self.height_task_index >= 0 else 0
        visible_rows = self.visible_step_rows()
        visible_revision = self.vision_steps_revision if self.active_steps_view() == "vision" else self.manager.revision
        visible_count = len(visible_rows)
        visible_source = self.get_visible_steps_source()
        visible_read_only = self.visible_steps_are_read_only()
        runtime_ready = self.runtime_ready
        motion_ready, motion_block_reason = self.motion_readiness()
        respawn_ready, respawn_block_reason = self.respawn_readiness()
        camera_view_ready, camera_view_block_reason = self.camera_view_readiness()
        validation_ready, validation_block_reason = self.can_validate_current_generated_height()
        ground_state = self.ground_state()
        manifest_warning = ""
        try:
            manifest_rows = self.store.status_rows()
            manifest_warning = getattr(self.store, "last_status_warning", "")
        except Exception as exc:
            manifest_rows = getattr(self.store, "_last_status_rows", [])
            manifest_warning = f"Could not read height manifest status rows: {exc}"
            if self.status != "[WARN] " + manifest_warning:
                self._warn("[WARN] " + manifest_warning)
        return {
            "sim": {
                "ready": self.sim_ready,
                "runtime_ready": runtime_ready,
                "ready_source": self.latest_sim_status.get("ready_source", "runtime"),
                "last_ready_change_reason": self.latest_sim_status.get("last_ready_change_reason", ""),
                "last_ready_change_message_type": self.latest_sim_status.get("last_ready_change_message_type", ""),
                "motion_ready": motion_ready,
                "motion_block_reason": motion_block_reason,
                "respawn_ready": respawn_ready,
                "respawn_block_reason": respawn_block_reason,
                "ground_state": ground_state,
                "no_sim": self.no_sim,
                "loaded_height_cm": self.loaded_sim_height_cm,
                "transport": self.transport.status(),
                "worker_status": dict(self.latest_sim_status),
                "physics_dt": float(self.latest_sim_status.get("physics_dt", 0.0) or 0.0),
                "sim_step_hz": float(self.latest_sim_status.get("sim_step_hz", 0.0) or 0.0),
                "real_time_factor": float(self.latest_sim_status.get("real_time_factor", 0.0) or 0.0),
                "worker_loop_hz": float(self.latest_sim_status.get("worker_loop_hz", 0.0) or 0.0),
                "ui_refresh_hz": self.ui_refresh_hz,
                "current_max_wheel_speed_rad_s": float(self.latest_sim_status.get("current_max_wheel_speed_rad_s", self.max_wheel_speed) or self.max_wheel_speed),
                "default_wheel_speed_rad_s": float(self.latest_sim_status.get("default_wheel_speed_rad_s", self.default_wheel_speed) or self.default_wheel_speed),
                "phase": self.latest_sim_status.get("phase", ""),
                "worker_pid": self.latest_sim_status.get("worker_pid", self.latest_sim_status.get("pid", "")),
                "worker_returncode": self.latest_sim_status.get("worker_returncode", ""),
                "requested_launch_mode": self.latest_sim_status.get("requested_launch_mode", getattr(self.args, "worker_launch_mode", "")),
                "resolved_launch_mode": self.latest_sim_status.get("resolved_launch_mode", self.latest_sim_status.get("launch_mode", "")),
                "resolved_python": self.latest_sim_status.get("resolved_python", ""),
                "python_version": self.latest_sim_status.get("python_version", ""),
                "isaacsim_version": self.latest_sim_status.get("isaacsim_version", ""),
                "isaaclab_version": self.latest_sim_status.get("isaaclab_version", ""),
                "preflight_ok": self.latest_sim_status.get("preflight_ok", False),
                "preflight_error": self.latest_sim_status.get("preflight_error", ""),
                "candidate_reports": self.latest_sim_status.get("candidate_reports", []),
                "display_command": self.latest_sim_status.get("display_command", self.latest_sim_status.get("worker_command", "")),
                "worker_command": self.latest_sim_status.get("worker_command", []),
                "worker_cwd": self.latest_sim_status.get("worker_cwd", ""),
                "worker_config_path": self.latest_sim_status.get("worker_config_path", self.latest_sim_status.get("config_path", "")),
                "worker_wrapper_path": self.latest_sim_status.get("worker_wrapper_path", self.latest_sim_status.get("wrapper_path", "")),
                "ipc_connected": self.latest_sim_status.get("ipc_connected", self.latest_sim_status.get("connected", False)),
                "phase_elapsed_s": float(self.latest_sim_status.get("phase_elapsed_s", 0.0) or 0.0),
                "startup_phase_history": self.latest_sim_status.get("startup_phase_history", []),
                "last_log_activity_at": self.latest_sim_status.get("last_log_activity_at", 0.0),
                "startup_progress_message": self.latest_sim_status.get("startup_progress_message", ""),
                "requested_headless": self.latest_sim_status.get("requested_headless", bool(getattr(self.args, "headless", False))),
                "effective_headless": self.latest_sim_status.get("effective_headless", False),
                "requested_livestream": self.latest_sim_status.get("requested_livestream", getattr(self.args, "livestream", 0)),
                "effective_livestream": self.latest_sim_status.get("effective_livestream", 0),
                "requested_enable_cameras": self.latest_sim_status.get("requested_enable_cameras", bool(getattr(self.args, "onboard_camera", True))),
                "effective_enable_cameras": self.latest_sim_status.get("effective_enable_cameras", bool(getattr(self.args, "enable_cameras", False))),
                "selected_experience": self.latest_sim_status.get("selected_experience", getattr(self.args, "experience", "")),
                "eula_source": self.latest_sim_status.get("eula_source", ""),
                "worker_env": self.latest_sim_status.get("worker_env", self.latest_sim_status.get("key_environment", {})),
                "last_meaningful_stdout": self.latest_sim_status.get("last_meaningful_stdout", ""),
                "last_meaningful_stderr": self.latest_sim_status.get("last_meaningful_stderr", ""),
                "error_category": self.latest_sim_status.get("error_category", ""),
                "startup_diagnosis": self.latest_sim_status.get("startup_diagnosis", ""),
                "startup_elapsed_s": float(self.latest_sim_status.get("startup_elapsed_s", 0.0) or 0.0),
                "stdout_tail": self.latest_sim_status.get("stdout_tail", []),
                "stderr_tail": self.latest_sim_status.get("stderr_tail", []),
                "traceback": self.latest_sim_status.get("traceback", ""),
                "target_joint_state": self.latest_sim_status.get("target_joint_state"),
                "actual_joint_state": self.latest_sim_status.get("actual_joint_state"),
                "joint_diagnostics": self.latest_sim_status.get("joint_diagnostics", []),
                "robot_ground": self.latest_sim_status.get("robot_ground", {}),
                "ground_surface": self.latest_sim_status.get(
                    "ground_surface",
                    (self.latest_sim_status.get("robot_ground", {}) or {}).get("ground_surface", {})
                    if isinstance(self.latest_sim_status.get("robot_ground", {}), dict)
                    else {},
                ),
                "ground_resolution_state": self.latest_sim_status.get(
                    "ground_resolution_state",
                    (self.latest_sim_status.get("robot_ground", {}) or {}).get("ground_resolution_state", "")
                    if isinstance(self.latest_sim_status.get("robot_ground", {}), dict)
                    else "",
                ),
                "grounded_reference_valid": self.latest_sim_status.get("grounded_reference_valid", False),
                "grounded_reference_physics_valid": self.latest_sim_status.get("grounded_reference_physics_valid", False),
                "grounded_reference_visual_valid": self.latest_sim_status.get("grounded_reference_visual_valid", False),
                "grounded_reference_stable": self.latest_sim_status.get("grounded_reference_stable", False),
                "ground_reference_block_reason": self.latest_sim_status.get("ground_reference_block_reason", ""),
                "grounded_reference_diagnostics": self.latest_sim_status.get("grounded_reference_diagnostics", {}),
                "last_ground_correction_z_m": self.latest_sim_status.get("last_ground_correction_z_m", 0.0),
                "telemetry": self.latest_sim_status.get("telemetry", {"enabled": False, "run_dir": ""}),
            },
            "height": {
                "current_cm": self.current_height_cm,
                "current_m": obstacle_height_m(self.current_height_cm),
                "steps_path": str(self.store.steps_path(self.current_height_cm)),
                "manifest_rows": manifest_rows,
                "manifest_warning": manifest_warning,
            },
            "task": {
                "active": self.height_task_active,
                "mode": self.height_task_mode,
                "index": task_pos,
                "total": task_total,
                "heights": list(self.height_task_heights),
                "current_height": self.current_task_height,
            },
            "sequence": {
                "rows": visible_rows,
                "count": visible_count,
                "dirty": self.manager.dirty if not visible_read_only else False,
                "accepted_path": str(self.vision_steps_path or "") if self.active_steps_view() == "vision" else str(self.store.steps_path(self.current_height_cm)),
                "revision": visible_revision,
                "view": self.active_steps_view(),
                "source": visible_source,
                "read_only": visible_read_only,
            },
            "height_sequence": {
                "rows": self.manager.rows(),
                "count": self.manager.count,
                "dirty": self.manager.dirty,
                "accepted_path": str(self.store.steps_path(self.current_height_cm)),
                "revision": self.manager.revision,
            },
            "recording": {
                "active": self.recording_active,
                "mode": self.mode,
                "pending": self.pending_step is not None,
                "pending_replacement": self.pending_replacement is not None,
                "events": len(self.record_events),
            },
            "playback": playback,
            "servos": state["servos"],
            "wheels": wheels_short,
            "target_joint_state": self.latest_sim_status.get("target_joint_state"),
            "actual_joint_state": self.latest_sim_status.get("actual_joint_state"),
            "joint_diagnostics": self.latest_sim_status.get("joint_diagnostics", []),
            "detail_text": self.detail_text,
            "status_text": "\n".join(self.status_log[-250:]),
            "selected_step_index": self.selected_step_index,
            "combine": {
                "enabled": self.combine_mode_enabled,
                "selected_indices": sorted(self.combine_selected_indices),
                "selected_count": len(self.combine_selected_indices),
                "contiguous": self.combine_selection_is_contiguous(),
                "preview": self.combine_preview_text,
                "allow_conflicts": self.allow_combine_conflicts,
            },
            "servo_wheel": {
                "staging_active": self.servo_wheel_staging_active,
                "staged_dirty": self.servo_wheel_staged_dirty,
                "staged_state": clone_command_state(self.servo_wheel_staged_state),
                "last_launch_commands": list(self.servo_wheel_last_launch_commands),
                "last_launch_time": self.servo_wheel_last_launch_time,
                "preview": self.servo_wheel_preview_text(),
            },
            "speed_scale": {
                "global_motion_speed_scale": self.motion_scale.global_motion_speed_scale,
                "wheel_speed_scale": self.motion_scale.wheel_speed_scale,
                "servo_command_scale": self.motion_scale.servo_command_scale,
                "playback_speed_scale": self.motion_scale.playback_speed_scale,
                "manual_wheel_scale": self.motion_scale.manual_wheel_scale,
                "manual_servo_scale": self.motion_scale.manual_servo_scale,
                "playback_scale": self.motion_scale.playback_scale,
                "apply_to_manual_control": self.motion_scale.apply_to_manual_control,
                "apply_to_playback": self.motion_scale.apply_to_playback,
                "preserve_wheel_distance": self.motion_scale.preserve_wheel_distance,
                "last_raw_motion_command": self.last_raw_motion_command,
                "last_scaled_motion_command": self.last_scaled_motion_command,
                "last_raw_values": list(self.last_raw_wheel_speeds),
                "last_scaled_values": list(self.last_scaled_wheel_speeds),
                "last_warnings": list(self.last_motion_warnings),
            },
            "vision": {
                **(dict(self.latest_sim_status.get("vision", {})) if isinstance(self.latest_sim_status.get("vision"), dict) else {}),
                "runtime_ready": runtime_ready,
                "camera_view_ready": camera_view_ready,
                "camera_view_block_reason": camera_view_block_reason,
                "vision_detection_ready": bool(runtime_ready and (dict(self.latest_sim_status.get("vision", {})).get("camera_ready", False) if isinstance(self.latest_sim_status.get("vision"), dict) else False)),
                "vision_generation_ready": bool(runtime_ready and self.vision_task_active and self.vision_source_mode == VISION_SOURCE_GENERATED),
                "validation_ready": validation_ready,
                "validation_block_reason": validation_block_reason,
                "motion_ready": motion_ready,
                "motion_block_reason": motion_block_reason,
                "respawn_ready": respawn_ready,
                "respawn_block_reason": respawn_block_reason,
                "ground_state": ground_state,
                "auto_replay_enabled": self.vision_auto_replay_enabled,
                "auto_replay_armed": self.vision_auto_replay_armed,
                "respawn_before_replay": self.vision_respawn_before_replay,
                "last_detection_revision": self.vision_last_detection_revision,
                "last_consumed_detection_revision": self.vision_last_consumed_detection_revision,
                "last_detected_height_cm": self.vision_last_detected_height_cm,
                "last_auto_action": self.vision_last_auto_action,
                "last_block_reason": self.vision_last_block_reason,
                "auto_replay_cooldown_s": self.vision_auto_replay_cooldown_s,
                "last_replay_started_at": self.vision_last_replay_started_at,
                "task": self.vision_task.to_dict(),
                "steps_ready": self.vision_steps_ready,
                "steps_height_cm": self.vision_steps_height_cm,
                "steps_path": str(self.vision_steps_path or ""),
                "steps_count": len(self.vision_steps),
                "steps_error": self.vision_steps_error,
                "steps_revision": self.vision_steps_revision,
                "source_mode": self.vision_source_mode,
                "generated_height_cm": self.vision_generated_height_cm,
                "generation_revision": self.vision_generation_revision,
                "generation_request_id": self.vision_generation_request_id,
                "generation_detection_baseline": self.vision_detection_baseline_revision,
                "generation_frame_baseline": self.vision_frame_baseline_revision,
                "scene_ready": self.vision_scene_ready,
                "robot_ground": self.latest_sim_status.get("robot_ground", {}),
                "grounded_reference_valid": self.latest_sim_status.get("grounded_reference_valid", False),
                "grounded_reference_physics_valid": self.latest_sim_status.get("grounded_reference_physics_valid", False),
                "grounded_reference_visual_valid": self.latest_sim_status.get("grounded_reference_visual_valid", False),
                "grounded_reference_stable": self.latest_sim_status.get("grounded_reference_stable", False),
                "ground_reference_block_reason": self.latest_sim_status.get("ground_reference_block_reason", ""),
                "grounded_reference_diagnostics": self.latest_sim_status.get("grounded_reference_diagnostics", {}),
                "last_ground_correction_z_m": self.latest_sim_status.get("last_ground_correction_z_m", 0.0),
            },
            "revisions": {
                "sequence": visible_revision,
                "height_sequence": self.manager.revision,
                "vision_steps": self.vision_steps_revision,
                "manifest": self.manifest_revision,
                "status": len(self.status_log),
            },
        }

    def status_line(self) -> str:
        task_total = len(self.height_task_heights)
        task_pos = self.height_task_index + 1 if self.height_task_active and self.height_task_index >= 0 else 0
        motion_ready, motion_reason = self.motion_readiness()
        respawn_ready, respawn_reason = self.respawn_readiness()
        return (
            f"Runtime Ready={self.runtime_ready} Motion Ready={motion_ready} Respawn Ready={respawn_ready} no_sim={self.no_sim} | "
            f"phase={self.latest_sim_status.get('phase', 'none')} pid={self.latest_sim_status.get('worker_pid', self.latest_sim_status.get('pid', ''))} | "
            f"Current height={self.current_height_cm}cm | "
            f"Height task active={self.height_task_active} mode={self.height_task_mode} "
            f"index={task_pos}/{task_total} | "
            f"Loaded steps path={self.store.steps_path(self.current_height_cm)} | "
            f"Playback active={self.playback.active} paused={self.playback.paused} | "
            f"Recording active={self.recording_active} | "
            f"Servo+Wheel staging={self.servo_wheel_staging_active} dirty={self.servo_wheel_staged_dirty} | "
            f"Vision auto={self.vision_auto_replay_enabled} armed={self.vision_auto_replay_armed} | "
            f"Dirty={self.manager.dirty} | "
            f"RTF={float(self.latest_sim_status.get('real_time_factor', 0.0) or 0.0):.2f} | "
            f"wheel_scale={self.motion_scale.manual_wheel_scale:.2f} | "
            f"motion_block={motion_reason or '-'} | respawn_block={respawn_reason or '-'}"
        )

    def _handle_tokens(self, tokens: list[str], command: str, source: str, message: CommandMessage | None = None) -> None:
        verb = tokens[0].lower()
        if verb == "mode":
            if len(tokens) > 1 and tokens[1].lower() in {"servo_wheel", "servo+wheel", "sw"}:
                self.mode = MODE_SERVO_WHEEL
                self._info("[INFO] Mode set to SERVO_WHEEL.")
            else:
                self.mode = MODE_TEST
                self._info("[INFO] Mode set to TEST.")
        elif verb in {"test_mode"}:
            self.mode = MODE_TEST
            self._info("[INFO] Mode set to TEST.")
        elif verb == "status":
            self._info("[STATUS] " + self.status_line())
        elif verb in {"e_stop", "estop", "space"}:
            self.playback.stop(silent=True)
            self.transport.stop_wheels()
            self.mode = MODE_E_STOP
            self.vision_auto_replay_armed = False
            self.vision_last_block_reason = "E-stop is active. Re-arm after clearing E-stop."
            self.vision_last_auto_action = "disarmed by E-stop"
            extra = " Staged Servo+Wheel state preserved." if self.servo_wheel_staging_active else ""
            self._warn("[WARN] Emergency stop: playback stopped and wheels zeroed." + extra)
        elif verb == "respawn":
            self.respawn_robot(source="manual")
        elif verb == "servo_wheel":
            self._handle_servo_wheel(tokens[1:])
        elif verb in {"servo", "angle", "wheel", "wheels", "speed", "w", "s", "a", "d", "x", "stop", "home", "reset"}:
            self._apply_motion_command(command, source, message=message)
        elif verb in {"step_record", "sr"}:
            self._handle_step_record(tokens[1:])
        elif verb in {"accept", "mark"}:
            self.accept_pending_step()
        elif verb == "replace_step":
            self._handle_replace_step(tokens[1:])
        elif verb == "delete_step" and len(tokens) == 2:
            self.delete_step(int(tokens[1]))
        elif verb == "undo":
            self.undo_step()
        elif verb == "clear_steps":
            self._warn("[WARN] Check Confirm Clear All and run clear_steps_confirmed to clear all accepted steps.")
        elif verb == "clear_steps_confirmed":
            self.clear_steps()
        elif verb == "show_step":
            self.show_step_command(tokens[1:])
        elif verb == "show_step_summary":
            self.show_step_summary_command(tokens[1:])
        elif verb == "inspect_step":
            self.inspect_step_command(tokens[1:])
        elif verb == "export_step_json" and len(tokens) == 2:
            self.export_step_json(int(tokens[1]))
        elif verb in {"play", "play_all"}:
            fast = "fast" in [token.lower() for token in tokens[1:]]
            self.play_all(fast=fast)
        elif verb in {"respawn_play", "respawn_and_play"}:
            self._handle_respawn_play(tokens[1:])
        elif verb == "play_step":
            self._handle_play_step(tokens[1:])
        elif verb == "play_to_step":
            self._handle_play_to_step(tokens[1:])
        elif verb == "playback_debug_selected":
            self.playback_debug_selected()
        elif verb == "pause_play":
            self.playback.pause()
        elif verb == "resume_play":
            self.playback.resume()
        elif verb in {"stop_play", "stop_playback"}:
            self.playback.stop()
        elif verb == "analyze_playback_timing":
            self.detail_text = self.playback.analyze_steps(self.manager.steps)
            self._info("[INFO] " + self.detail_text)
        elif verb == "export_motion_txt":
            self.export_motion_txt()
        elif verb in {"combine", "combine_steps", "merge_steps", "merge"}:
            self._handle_combine(tokens[1:])
        elif verb == "vision":
            self._handle_vision_command(tokens[1:])
        else:
            self._warn(f"[WARN] Unsupported command: {command}")

    def _handle_vision_command(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        if sub == "start":
            source = " ".join(args[1:]) if len(args) > 1 else None
            self.start_vision_task(source)
        elif sub == "finish":
            self.finish_vision_task()
        elif sub == "source" and len(args) > 1:
            self.vision_source_mode = normalize_source_mode(" ".join(args[1:]))
            self.vision_task.source_mode = self.vision_source_mode
            self.transport.set_vision_source_mode("external" if self.vision_source_mode == VISION_SOURCE_EXTERNAL else "generated")
            self._info(f"[INFO] Vision source set to {self.vision_source_mode}.")
        elif sub in {"generate", "generate_obstacle", "generate_test_obstacle"} and len(args) > 1:
            self.generate_vision_test_obstacle(int(args[1]))
        elif sub in {"play_validated", "play_validated_steps"}:
            if self.vision_steps_height_cm is None:
                self._warn("[WARN] Vision steps are not ready.")
            else:
                self.replay_validated_vision_steps(int(self.vision_steps_height_cm), int(self.vision_steps_detection_revision or self.vision_last_detection_revision or 0))
        elif sub == "enable":
            self.set_vision_enabled(True)
        elif sub == "disable":
            self.set_vision_enabled(False)
        elif sub == "detect_once":
            self.request_vision_detection_once()
        elif sub == "reset_filter":
            self.reset_vision_filter()
        elif sub == "save_debug_frame":
            self.request_vision_debug_frame()
        elif sub == "validate_camera":
            self.validate_isaac_camera()
        elif sub in {"show_camera_view", "open_camera_viewport"}:
            self.open_onboard_camera_viewport()
        elif sub == "return_main_view_to_perspective":
            self.return_main_view_to_perspective()
        elif sub == "close_camera_viewport":
            self.close_onboard_camera_viewport()
        elif sub == "restore_camera_view":
            self.restore_camera_view()
        elif sub == "validate_camera_geometry":
            self.transport.validate_camera_geometry()
        elif sub == "validate_current_height":
            self.validate_current_generated_height()
        elif sub == "save_rgbd_diagnostic":
            self.request_rgbd_diagnostic()
        elif sub == "clear_validation_result":
            self.clear_vision_validation_result()
        elif sub == "validate_mount_after_next_respawn":
            self.validate_mount_after_next_respawn()
        elif sub == "auto_enable":
            self.set_vision_auto_replay_enabled(True)
        elif sub == "auto_disable":
            self.set_vision_auto_replay_enabled(False)
        elif sub == "arm":
            self.arm_vision_auto_replay()
        elif sub == "disarm":
            self.disarm_vision_auto_replay()
            self._info("[INFO] Vision auto replay disarmed.")
        else:
            self._info("[VISION] " + json.dumps(self.snapshot().get("vision", {}), ensure_ascii=False, default=str))

    def _handle_servo_wheel(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "mode"
        if sub == "mode":
            self.mode = MODE_SERVO_WHEEL
            self.servo_wheel_staging_active = True
            self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
            self.servo_wheel_staged_dirty = False
            self.detail_text = self.servo_wheel_preview_text()
            self._info("[INFO] Servo+Wheel staging mode active. Drag sliders to stage; click Launch Servo+Wheel to apply.")
        elif sub == "launch":
            self.launch_servo_wheel_staged()
        elif sub == "cancel":
            self.transport.stop_wheels()
            self.servo_wheel_staging_active = False
            self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
            self.servo_wheel_staged_dirty = False
            self.mode = MODE_TEST
            self._info("[INFO] Servo+Wheel mode cancelled; wheels stopped.")
        elif sub in {"clear", "clear_staged"}:
            self.reset_servo_wheel_staged_to_current()
        else:
            self._warn("[WARN] Usage: servo_wheel mode|launch|cancel|clear_staged")

    def respawn_robot(self, *, source: str = "manual") -> bool:
        if not self.no_sim and not self.runtime_ready:
            self._warn("[WARN] Isaac runtime is not ready.")
            return False
        if self.recording_active:
            self._warn("[WARN] Stop recording before Respawn.")
            return False
        if self.pending_step is not None or self.pending_replacement is not None:
            self._warn("[WARN] Accept or discard pending step/replacement before Respawn.")
            return False
        if self.operation_busy:
            self._warn(f"[WARN] Operation busy: {self.busy_name}")
            return False
        respawn_ok, respawn_reason = self.respawn_readiness()
        if not self.no_sim and not respawn_ok:
            self._warn("[WARN] " + (respawn_reason or "Respawn is not ready. Calibrate a valid ground reference first."))
            return False
        manual = str(source or "manual") == "manual"
        was_estop = self.mode == MODE_E_STOP
        if manual:
            self.playback.stop(silent=True, reason="manual_respawn")
            self.vision_auto_replay_armed = False
            self.vision_last_auto_action = "disarmed by manual respawn"
        self.transport.stop_wheels()
        self.transport.respawn()
        self.transport.request_state()
        if manual:
            self.vision_last_consumed_detection_revision = 0
            self.vision_last_detection_revision = 0
            self.vision_last_detected_height_cm = None
            self.reset_vision_filter(clear_steps=True)
            self.clear_vision_validation_result(clear_steps=True)
            if was_estop:
                self.mode = MODE_E_STOP
                self.vision_last_block_reason = "E-stop is active. Respawn did not clear it."
            else:
                self.mode = MODE_TEST
            self._info("[INFO] Manual Respawn requested; playback stopped, wheels stopped, Vision auto replay disarmed.")
        else:
            self._info("[INFO] Playback Respawn requested; wheels stopped.")
        return True

    def is_motion_command(self, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        return tokens[0].lower() in {"servo", "angle", "wheel", "wheels", "speed", "w", "a", "s", "d", "x", "home"}

    def is_immediate_stop_command(self, command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if not tokens:
            return False
        lowered = [token.lower() for token in tokens]
        return lowered == ["stop"] or lowered == ["wheel", "stop"] or lowered == ["wheels", "stop"]

    def should_stage_motion_command(self, command: str, source: str) -> bool:
        if not self.servo_wheel_staging_active:
            return False
        if source in {"playback", "show_step", "servo_wheel_launch", "transport"}:
            return False
        if self.is_immediate_stop_command(command):
            return False
        return self.is_motion_command(command)

    def stage_motion_command(self, command: str, source: str) -> None:
        outgoing_command = command
        if source != "playback":
            scaled = scale_manual_motion_command(
                command,
                default_wheel_speed=self.default_wheel_speed,
                max_wheel_speed=self.max_wheel_speed,
                wheel_speed_scale=self.motion_scale.manual_wheel_scale,
                servo_command_scale=self.motion_scale.manual_servo_scale,
            )
            outgoing_command = scaled.scaled_command
            self.last_raw_motion_command = scaled.raw_command
            self.last_scaled_motion_command = scaled.scaled_command
            self.last_raw_wheel_speeds = scaled.raw_speed_values
            self.last_scaled_wheel_speeds = scaled.scaled_speed_values
            self.last_motion_warnings = scaled.warnings
        apply_command_to_state(self.servo_wheel_staged_state, outgoing_command)
        self.servo_wheel_staged_dirty = True
        self.detail_text = self.servo_wheel_preview_text()
        self.status = (
            f"Staged Servo+Wheel command: {command}"
            if outgoing_command == command
            else f"Staged Servo+Wheel command: {command} -> {outgoing_command}"
        )
        if self.last_motion_warnings:
            self.status += " | " + "; ".join(self.last_motion_warnings)

    def build_commands_from_command_state(self, state: dict[str, Any]) -> list[str]:
        normalized = clone_command_state(state)
        commands: list[str] = []
        for name in SERVO_JOINT_NAMES:
            commands.append(f"servo {name} {float(normalized['servos'].get(name, 0.0)):.6g}")
        for name in WHEEL_JOINT_NAMES:
            short = WHEEL_NAME_TO_SHORT.get(name, name)
            commands.append(f"wheel {short} {float(normalized['wheels'].get(name, 0.0)):.6g}")
        return commands

    def launch_servo_wheel_staged(self) -> None:
        if not self.servo_wheel_staging_active:
            self.servo_wheel_staging_active = True
            self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
            self.servo_wheel_staged_dirty = False
        self.mode = MODE_SERVO_WHEEL if not self.recording_active else self.mode
        before_state = clone_command_state(self.transport.capture_command_state())
        commands = self.build_commands_from_command_state(self.servo_wheel_staged_state)
        for command in commands:
            self.transport.send(command, source="servo_wheel_launch")
        after_state = clone_command_state(self.servo_wheel_staged_state)
        self.servo_wheel_last_launch_commands = list(commands)
        self.servo_wheel_last_launch_time = time.time()
        self.servo_wheel_staged_dirty = False
        if self.recording_active:
            event_time = max(0.0, time.monotonic() - self.record_start_wall_time)
            self.record_events.append(
                make_event(
                    event_time,
                    "servo_wheel launch",
                    kind="servo_wheel_launch",
                    command_state_before=before_state,
                    command_state_after=after_state,
                    expanded_commands=commands,
                )
            )
        self.detail_text = self.servo_wheel_preview_text()
        self._info("[INFO] Launched staged Servo+Wheel command.")

    def reset_servo_wheel_staged_to_current(self) -> None:
        self.servo_wheel_staged_state = clone_command_state(self.transport.capture_command_state())
        self.servo_wheel_staged_dirty = False
        self.detail_text = self.servo_wheel_preview_text()
        self._info("[INFO] Servo+Wheel staged state reset to current live state.")

    def servo_wheel_preview_text(self) -> str:
        live_state = clone_command_state(self.transport.capture_command_state())
        staged_state = clone_command_state(self.servo_wheel_staged_state)
        commands = self.build_commands_from_command_state(staged_state)
        delta_lines: list[str] = []
        for name in SERVO_JOINT_NAMES:
            live = float(live_state["servos"].get(name, 0.0))
            staged = float(staged_state["servos"].get(name, 0.0))
            if abs(staged - live) > 1.0e-9:
                delta_lines.append(f"{name}: {live:.3g} -> {staged:.3g}")
        for name in WHEEL_JOINT_NAMES:
            live = float(live_state["wheels"].get(name, 0.0))
            staged = float(staged_state["wheels"].get(name, 0.0))
            if abs(staged - live) > 1.0e-9:
                delta_lines.append(f"{WHEEL_NAME_TO_SHORT.get(name, name)}: {live:.3g} -> {staged:.3g}")
        last_launch = "-" if self.servo_wheel_last_launch_time is None else time.strftime("%H:%M:%S", time.localtime(self.servo_wheel_last_launch_time))
        return (
            "Servo+Wheel staging preview\n"
            f"staging_active={self.servo_wheel_staging_active} staged_dirty={self.servo_wheel_staged_dirty}\n"
            f"last_launch_time={last_launch}\n\n"
            "live command_state:\n"
            + self.command_state_summary(live_state)
            + "\n\nstaged command_state:\n"
            + self.command_state_summary(staged_state)
            + "\n\ndelta:\n"
            + ("\n".join(delta_lines) if delta_lines else "no staged delta")
            + "\n\nlaunch commands:\n"
            + "\n".join(commands)
        )

    def _apply_motion_command(self, command: str, source: str, *, message: CommandMessage | None = None) -> None:
        if self.is_immediate_stop_command(command):
            if self.servo_wheel_staging_active:
                for name in WHEEL_JOINT_NAMES:
                    self.servo_wheel_staged_state["wheels"][name] = 0.0
                self.servo_wheel_staged_dirty = True
            before_state = clone_command_state(self.transport.capture_command_state())
            self.transport.send(command, source=source, message=message)
            after_state = clone_command_state(self.transport.capture_command_state())
            if self.recording_active and source != "playback":
                self.record_events.append(
                    make_event(
                        max(0.0, time.monotonic() - self.record_start_wall_time),
                        command,
                        kind=source,
                        command_state_before=before_state,
                        command_state_after=after_state,
                    )
                )
            self.status = f"Command: {command}"
            return
        if self.should_stage_motion_command(command, source):
            self.stage_motion_command(command, source)
            return
        before_state = clone_command_state(self.transport.capture_command_state())
        outgoing_command = command
        if source != "playback":
            scaled = scale_manual_motion_command(
                command,
                default_wheel_speed=self.default_wheel_speed,
                max_wheel_speed=self.max_wheel_speed,
                wheel_speed_scale=self.motion_scale.manual_wheel_scale,
                servo_command_scale=self.motion_scale.manual_servo_scale,
            )
            outgoing_command = scaled.scaled_command
            self.last_raw_motion_command = scaled.raw_command
            self.last_scaled_motion_command = scaled.scaled_command
            self.last_raw_wheel_speeds = scaled.raw_speed_values
            self.last_scaled_wheel_speeds = scaled.scaled_speed_values
            self.last_motion_warnings = scaled.warnings
        self.transport.send(outgoing_command, source=source, message=message)
        after_state = clone_command_state(self.transport.capture_command_state())
        if self.recording_active and source != "playback":
            event_time = max(0.0, time.monotonic() - self.record_start_wall_time)
            self.record_events.append(
                make_event(
                    event_time,
                    outgoing_command,
                    kind=source,
                    command_state_before=before_state,
                    command_state_after=after_state,
                )
            )
        self.status = f"Command: {command}" if outgoing_command == command else f"Command: {command} -> {outgoing_command}"
        if self.last_motion_warnings:
            self.status += " | " + "; ".join(self.last_motion_warnings)

    def _handle_step_record(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        if sub == "start":
            self.start_step_recording()
        elif sub == "stop":
            self.stop_step_recording()
        elif sub == "accept":
            if self.pending_replacement is not None or self.mode == MODE_PENDING_REPLACEMENT:
                self.accept_replacement()
            else:
                self.accept_pending_step()
        elif sub == "discard":
            self.discard_pending_step(restore_before=False)
        elif sub == "discard_restore_before":
            self.discard_pending_step(restore_before=True)
        else:
            self._info(
                f"[RECORD] mode={self.mode} events={len(self.record_events)} "
                f"pending={self.pending_step is not None} replacement={self.pending_replacement is not None}"
            )

    def start_step_recording(self) -> None:
        ok, reason = self.can_start_recording()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        if self.mode in {MODE_TEST, MODE_SERVO_WHEEL}:
            self.recording_kind = "recorded"
            self.mode = MODE_RECORDING_STEP
        elif self.mode == MODE_REPLACE_STEP_READY and self.replace_target_index is not None:
            self.recording_kind = "replacement"
            self.mode = MODE_REPLACING_STEP
        else:
            self._warn(f"[WARN] step_record start is only valid in TEST or REPLACE_STEP_READY. Current mode={self.mode}")
            return
        self.record_start_wall_time = time.monotonic()
        self.transport.request_state()
        self.record_command_state_before = clone_command_state(self.transport.capture_command_state())
        self.record_sim_state_before = self.capture_current_sim_state()
        self.record_events = []
        if self.servo_wheel_staging_active:
            self._info("[INFO] Recording in Servo+Wheel staging mode. Slider changes will be staged; Launch records the motion.")
        else:
            self._info(f"[INFO] Step recording started. mode={self.mode}")

    def stop_step_recording(self) -> None:
        if not self.recording_active or self.record_command_state_before is None:
            self._warn("[WARN] step_record stop is only valid while recording.")
            return
        duration = max(0.0, time.monotonic() - self.record_start_wall_time)
        self.transport.request_state()
        after_state = clone_command_state(self.transport.capture_command_state())
        sim_state_after = self.capture_current_sim_state()
        events = self.record_events
        self.last_record_coalesce_stats = {
            "original_count": len(events),
            "coalesced_count": len(events),
            "coalesced": False,
            "dropped_count": 0,
        }
        if self.record_coalesce_slider_events:
            events, self.last_record_coalesce_stats = coalesce_record_events(
                self.record_events,
                min_interval_s=self.record_event_min_interval_s,
                max_events=self.record_max_events_per_step,
            )
        is_replacement = self.mode == MODE_REPLACING_STEP and self.replace_target_index is not None
        index = self.replace_target_index if is_replacement else self.manager.count + 1
        step = make_step(
            index=int(index),
            step_type="replacement" if is_replacement else "recorded",
            duration=duration,
            events=events,
            command_state_before=self.record_command_state_before,
            command_state_after=after_state,
            name=f"step_{int(index):03d}_{'replacement' if is_replacement else 'recorded'}_height_{self.current_height_cm:02d}cm_dur_{duration:.2f}s",
            note=f"height={self.current_height_cm}cm",
            extra={
                "height_cm": self.current_height_cm,
                "height_m": obstacle_height_m(self.current_height_cm),
                "record_coalesce": dict(self.last_record_coalesce_stats),
                "sim_state_before": self.record_sim_state_before,
                "sim_state_after": sim_state_after,
            },
        )
        if is_replacement:
            self.pending_replacement = step
            self.mode = MODE_PENDING_REPLACEMENT
            self.set_step_summary_detail(step, title=f"Pending replacement for step {int(index):03d}")
            self._info(f"[INFO] Replacement recording stopped; pending replacement has {len(events)} events.")
        else:
            self.pending_step = step
            self.mode = MODE_PENDING_RECORDED_STEP
            self.set_step_summary_detail(step, title=f"Pending recorded step {int(index):03d}")
            self._info(f"[INFO] Step recording stopped; pending step has {len(events)} events.")
        if self.servo_wheel_staging_active and not events:
            self._warn("[WARN] No launched motion was recorded. Click Launch Servo+Wheel while recording to record motion.")
        if self.last_record_coalesce_stats.get("coalesced"):
            self._warn(
                "[WARN] Recorded step has many events; coalesced from "
                f"{self.last_record_coalesce_stats.get('original_count')} to "
                f"{self.last_record_coalesce_stats.get('coalesced_count')}."
            )
        self.record_events = []
        self.record_command_state_before = None
        self.record_sim_state_before = None
        self.recording_kind = ""

    def accept_pending_step(self) -> None:
        ok, reason = self.can_accept_recorded_step()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any] | None:
            if self.pending_step is None:
                return None
            accepted = self.manager.add_step(self.pending_step)
            self.pending_step = None
            self.mode = MODE_TEST
            self.selected_step_index = int(accepted["index"])
            self.set_step_summary_detail(accepted, title=f"Step {int(accepted['index']):03d} accepted")
            self._info(f"[INFO] Accepted recorded step {int(accepted['index']):03d}: {accepted['name']}")
            return accepted

        self._run_operation("Accept recorded step", work)

    def discard_pending_step(self, *, restore_before: bool) -> None:
        pending = self.pending_step or self.pending_replacement
        if pending is None:
            self._warn("[WARN] No pending step to discard.")
            return
        if restore_before:
            self.apply_command_state(pending.get("command_state_before"), keep_wheels=False)
        self.pending_step = None
        self.pending_replacement = None
        self.mode = MODE_TEST
        self._info("[INFO] Pending step discarded.")

    def _handle_replace_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: replace_step <index>|start|stop|accept|discard|cancel")
            return
        sub = args[0].lower()
        if sub.isdigit():
            self.prepare_replacement(int(sub))
        elif sub == "start":
            self.start_step_recording()
        elif sub == "stop":
            self.stop_step_recording()
        elif sub == "accept":
            self.accept_replacement()
        elif sub == "discard":
            self.discard_pending_step(restore_before=False)
        elif sub == "cancel":
            self.pending_replacement = None
            self.replace_target_index = None
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self.mode = MODE_TEST
            self._info("[INFO] Replacement cancelled.")

    def prepare_replacement(self, index: int) -> None:
        ok, reason = self.can_prepare_replacement()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        self.manager.get_step(index)
        self.replace_target_index = index
        self.mode = MODE_REPLACE_STEP_READY
        self._info(f"[INFO] Replacement prepared for step {index:03d}. Press Start Replacement Recording.")

    def accept_replacement(self) -> None:
        ok, reason = self.can_accept_replacement()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any] | None:
            if self.pending_replacement is None or self.replace_target_index is None:
                return None
            target = self.replace_target_index
            replaced = self.manager.replace_step(target, self.pending_replacement)
            self.selected_step_index = target
            self.set_step_summary_detail(replaced, title=f"Step {int(replaced['index']):03d} replaced")
            self.pending_replacement = None
            self.replace_target_index = None
            self.mode = MODE_TEST
            self._info(f"[INFO] Replaced step {int(replaced['index']):03d}.")
            return replaced

        self._run_operation("Accept replacement", work)

    def delete_step(self, index: int) -> None:
        if self.visible_steps_are_read_only():
            self._warn("[WARN] Vision steps are read-only. Finish Vision Task before deleting Height Task steps.")
            return
        removed = self.manager.delete_step(index)
        self.selected_step_index = min(index, self.manager.count) if self.manager.count else None
        self._info(f"[INFO] Deleted accepted step {index:03d}: {removed.get('name', '')}")

    def undo_step(self) -> None:
        if self.visible_steps_are_read_only():
            self._warn("[WARN] Vision steps are read-only. Finish Vision Task before editing Height Task steps.")
            return
        removed = self.manager.undo()
        if removed is None:
            self._warn("[WARN] No accepted step to undo.")
            return
        self.selected_step_index = self.manager.count if self.manager.count else None
        self._info(f"[INFO] Undid accepted step {int(removed.get('index', 0)):03d}.")

    def clear_steps(self) -> None:
        if self.visible_steps_are_read_only():
            self._warn("[WARN] Vision steps are read-only. Finish Vision Task before clearing Height Task steps.")
            return
        self.manager.clear()
        self.selected_step_index = None
        self.detail_text = ""
        self._info("[INFO] Cleared all accepted steps in memory.")

    def show_step_command(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: show_step <index> [before|after]")
            return
        index = int(args[0])
        when = "after"
        for token in args[1:]:
            if token.lower() in {"before", "after"}:
                when = token.lower()
        self.show_step(index, when=when)

    def show_step(self, index: int, *, when: str) -> None:
        if self.recording_active:
            self._warn("[WARN] Stop recording before Show Selected Before/After.")
            return
        if self.playback.active:
            self._warn("[WARN] Stop playback before Show Selected Before/After.")
            return
        if not self.no_sim and not self.runtime_ready:
            self._warn("[WARN] Isaac runtime is not ready before applying a step state.")
            return
        step = normalize_step(self.get_visible_step(index), index=index)
        state_key = "command_state_before" if when == "before" else "command_state_after"
        self.apply_command_state(step.get(state_key), keep_wheels=False)
        self.selected_step_index = index
        self.detail_text = (
            f"Applied {when} command state for step {index:03d}.\n\n"
            + self.command_state_summary(step.get(state_key))
            + "\n\n"
            + self.compact_step_details(step, title=f"Step {index:03d} summary")
        )
        self._info(f"[INFO] Applied {when} state for step {index:03d}.")

    def show_step_summary_command(self, args: list[str]) -> None:
        if len(args) != 1:
            self._warn("[WARN] Usage: show_step_summary <index>")
            return
        index = int(args[0])
        step = normalize_step(self.get_visible_step(index), index=index)
        self.selected_step_index = index
        self.set_step_summary_detail(step, title=f"Step {index:03d} summary")
        self._info(f"[INFO] Showing compact summary for step {index:03d}.")

    def inspect_step_command(self, args: list[str]) -> None:
        if len(args) < 1:
            self._warn("[WARN] Usage: inspect_step <index>")
            return
        index = int(args[0])
        step = normalize_step(self.get_visible_step(index), index=index)
        self.selected_step_index = index
        self.detail_text = self.step_json_text(step)
        self._info(f"[INFO] Inspecting step {index:03d} JSON (truncated if needed).")

    def apply_command_state(self, state: dict[str, Any] | None, *, keep_wheels: bool) -> None:
        normalized = clone_command_state(state)
        for name, value in normalized["servos"].items():
            self.transport.send(f"servo {name} {float(value):.6g}", source="show_step")
        if not keep_wheels:
            self.transport.send("wheel stop", source="show_step")
            return
        for name, value in normalized["wheels"].items():
            short = WHEEL_NAME_TO_SHORT.get(name, name)
            self.transport.send(f"wheel {short} {float(value):.6g}", source="show_step")

    def play_all(self, *, fast: bool) -> bool:
        ok, reason = self.playback_readiness(respawn_first=bool(respawn_first))
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        if self.active_steps_view() == "height" and self.manager.count == 0:
            self.load_steps_for_current_height()
        steps = self.get_visible_steps()
        if not steps:
            return False
        profile = "fast" if fast else self.playback.profile
        label = (
            f"{self.vision_steps_height_cm}cm validated vision steps"
            if self.active_steps_view() == "vision"
            else f"{self.current_height_cm}cm accepted steps"
        )
        return self.start_playback(steps, label=label, profile=profile)

    def _handle_respawn_play(self, args: list[str]) -> None:
        tokens = [arg.lower() for arg in args]
        fast = "fast" in tokens
        if "step" in tokens or "selected" in tokens:
            index = self.selected_step_index
            for token in tokens:
                if token.isdigit():
                    index = int(token)
            if index is None:
                self._warn("[WARN] Select an accepted step first.")
                return
            profile = "fast" if fast else self.playback.profile
            source_height = self.vision_steps_height_cm if self.active_steps_view() == "vision" else self.current_height_cm
            self.start_playback(
                [self.get_visible_step(index)],
                label=f"{source_height}cm step {int(index):03d}",
                profile=profile,
                restore_start_state=True,
                respawn_first=True,
            )
            return
        if "to" in tokens:
            index = self.selected_step_index
            for token in tokens:
                if token.isdigit():
                    index = int(token)
            if index is None:
                self._warn("[WARN] Select an accepted step first.")
                return
            profile = "fast" if fast else self.playback.profile
            steps = self.get_visible_steps()
            source_height = self.vision_steps_height_cm if self.active_steps_view() == "vision" else self.current_height_cm
            self.start_playback(
                steps[: int(index)],
                label=f"{source_height}cm steps 1..{int(index)}",
                profile=profile,
                respawn_first=True,
            )
            return
        if self.active_steps_view() == "height" and self.manager.count == 0:
            self.load_steps_for_current_height()
        steps = self.get_visible_steps()
        if not steps:
            return
        source_height = self.vision_steps_height_cm if self.active_steps_view() == "vision" else self.current_height_cm
        self.start_playback(
            steps,
            label=f"{source_height}cm accepted steps",
            profile="fast" if fast else self.playback.profile,
            respawn_first=True,
        )

    def _handle_play_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: play_step <index> [fast|raw]")
            return
        ok, reason = self.can_playback()
        self._info(
            "[PLAYBACK DEBUG] play_step click "
            f"args={args} can_playback={ok} reason={reason or 'ok'} "
            f"restore_start_state={self.restore_step_start_state_before_selected_playback} "
            f"mode={self.mode} sim_ready={self.sim_ready} no_sim={self.no_sim}"
        )
        if not ok:
            self._warn("[WARN] " + reason)
            return
        index = int(args[0])
        raw_step = self.get_visible_step(index)
        normalized = normalize_step(raw_step, index=index)
        self._info(
            "[PLAYBACK DEBUG] selected step "
            f"index={index} sim_state_before={isinstance(normalized.get('sim_state_before'), dict)} "
            f"command_state_before={isinstance(normalized.get('command_state_before'), dict)} "
            f"events={len(normalized.get('events', []))}"
        )
        profile = "fast" if "fast" in [arg.lower() for arg in args[1:]] else self.playback.profile
        self.start_playback(
            [raw_step],
            label=(
                f"{self.vision_steps_height_cm}cm step {index:03d}"
                if self.active_steps_view() == "vision"
                else f"{self.current_height_cm}cm step {index:03d}"
            ),
            profile=profile,
            restore_start_state=self.restore_step_start_state_before_selected_playback,
        )

    def _handle_play_to_step(self, args: list[str]) -> None:
        if not args:
            self._warn("[WARN] Usage: play_to_step <index> [fast|raw]")
            return
        ok, reason = self.can_playback()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        index = int(args[0])
        profile = "fast" if "fast" in [arg.lower() for arg in args[1:]] else self.playback.profile
        steps = self.get_visible_steps()
        source_height = self.vision_steps_height_cm if self.active_steps_view() == "vision" else self.current_height_cm
        self.start_playback(steps[:index], label=f"{source_height}cm steps 1..{index}", profile=profile)

    def start_playback(
        self,
        steps: list[dict[str, Any]],
        *,
        label: str,
        profile: str | None = None,
        restore_start_state: bool = False,
        respawn_first: bool = False,
    ) -> bool:
        ok, reason = self.can_playback()
        if not ok:
            self._warn("[WARN] " + reason)
            return False
        if profile is not None:
            self.playback.set_profile(profile)
        try:
            plan = plan_from_steps(
                steps,
                profile=self.playback.profile,
                speed=self.playback.speed,
                trailing_pad=self.playback.trailing_pad,
                max_idle_gap=self.playback.max_idle_gap,
                preserve_wheel_distance=self.playback.preserve_wheel_distance,
                max_wheel_speed=self.max_wheel_speed,
                label=label,
            )
        except Exception as exc:
            self.playback.last_error = str(exc)
            self._warn(f"[WARN] Playback plan failed: {exc}")
            return False
        first_commands = [event.command for event in plan.events[:5]]
        self._info(
            "[PLAYBACK DEBUG] plan "
            f"label={label} events={len(plan.events)} final_time={plan.final_time_s:.3f}s "
            f"profile={self.playback.profile} speed={self.playback.speed:.2f} "
            f"first_commands={first_commands}"
        )
        if not plan.events:
            self.playback.start_plan(plan)
            step_index = "unknown"
            if len(steps) == 1:
                try:
                    step_index = f"{int(normalize_step(steps[0]).get('index', 0)):03d}"
                except Exception:
                    step_index = "unknown"
            message = f"Selected step {step_index} has no motion events to play."
            self.playback.last_error = message
            self.playback.last_info = message
            self.detail_text = message
            self._warn("[WARN] " + message)
            return False
        self.transport.stop_wheels()
        self._info("[PLAYBACK DEBUG] stop_wheels sent before playback restore/start.")
        pre_start_delay_s = 0.0
        if respawn_first:
            self.respawn_robot(source="playback")
            pre_start_delay_s = max(pre_start_delay_s, self.respawn_play_settle_s)
        if restore_start_state and steps:
            if self.apply_step_start_state(steps[0]):
                pre_start_delay_s = max(pre_start_delay_s, self.playback_pre_step_settle_s)
        ok = self.playback.start_plan(plan, start_delay_s=pre_start_delay_s)
        if ok:
            self.mode = MODE_PLAYBACK
            if pre_start_delay_s > 0.0:
                self._info(f"[INFO] Playback scheduled in {pre_start_delay_s:.2f}s: {label}")
            else:
                self._info(f"[INFO] Playback started: {label}")
            debug = (
                "[PLAYBACK DEBUG] manager "
                f"active={self.playback.active} scheduled={self.playback.scheduled_start_at > 0.0} "
                f"index={self.playback.index} count={len(plan.events)} "
                f"last_info={self.playback.last_info}"
            )
            self.status_log.append(debug)
            print(debug)
        else:
            self._warn(f"[WARN] Playback failed: {self.playback.last_error or 'unknown reason'}")
        return ok

    def apply_step_start_state(self, step: dict[str, Any]) -> bool:
        normalized = normalize_step(step)
        index = int(normalized.get("index", 0))
        sim_state = normalized.get("sim_state_before")
        command_state = normalized.get("command_state_before")
        raw_has_command_state = isinstance(step, dict) and any(
            key in step for key in ("command_state_before", "state_before", "robot_state_before")
        )
        has_sim_state = isinstance(sim_state, dict) and bool(sim_state)
        has_command_state = raw_has_command_state and isinstance(command_state, dict) and bool(command_state)
        self._info(
            "[PLAYBACK DEBUG] start-state restore "
            f"step={index:03d} restore_full_sim_pose={self.restore_full_sim_pose_if_available} "
            f"fallback_command_state={self.fallback_to_command_state_before} "
            f"sim_state_before={has_sim_state} command_state_before={has_command_state}"
        )
        if self.restore_full_sim_pose_if_available and isinstance(sim_state, dict):
            try:
                self.transport.restore_sim_state(sim_state)
                self.transport.stop_wheels()
                self.transport.request_state()
                self.detail_text = f"Restored sim_state_before for step {index:03d} before playback.\n\n" + self.compact_step_details(normalized)
                self._info(f"[INFO] Restored sim_state_before for step {index:03d} before playback.")
                return True
            except Exception as exc:
                self._warn(f"[WARN] restore_sim_state failed for step {index:03d}: {exc}")
        if not self.fallback_to_command_state_before:
            self._warn(f"[WARN] No sim_state_before saved for step {index:03d}; fallback disabled.")
            return False
        if not has_command_state:
            self._warn(f"[WARN] No start state saved for step {index:03d}; playing from current robot state.")
            return False
        if self.restore_full_sim_pose_if_available:
            self._warn(f"[WARN] No sim_state_before saved for this step; using command_state_before only.")
        self.apply_command_state(command_state, keep_wheels=False)
        self.detail_text = f"Applied command_state_before for step {index:03d} before playback.\n\n" + self.compact_step_details(normalized)
        self._info(f"[INFO] Applied command_state_before for step {index:03d} before playback.")
        return True

    def playback_debug_selected(self) -> None:
        index = self.selected_step_index
        lines = [
            "Playback selected debug",
            f"controller_selected_index={index}",
            f"manager_count={self.manager.count}",
            f"can_playback={self.can_playback()}",
            f"mode={self.mode} sim_ready={self.sim_ready} no_sim={self.no_sim}",
            f"playback_status={self.playback.status_dict()}",
        ]
        if index is not None:
            try:
                step = normalize_step(self.get_visible_step(index), index=index)
                plan = plan_from_steps(
                    [step],
                    profile=self.playback.profile,
                    speed=self.playback.speed,
                    trailing_pad=self.playback.trailing_pad,
                    max_idle_gap=self.playback.max_idle_gap,
                    preserve_wheel_distance=self.playback.preserve_wheel_distance,
                    max_wheel_speed=self.max_wheel_speed,
                    label=(
                        f"{self.vision_steps_height_cm}cm step {index:03d}"
                        if self.active_steps_view() == "vision"
                        else f"{self.current_height_cm}cm step {index:03d}"
                    ),
                )
                lines.extend(
                    [
                        "",
                        self.compact_step_details(step, title=f"Step {index:03d} summary"),
                        "",
                        f"plan_count={len(plan.events)} final_time={plan.final_time_s:.3f}s",
                        f"plan_first_commands={[event.command for event in plan.events[:5]]}",
                    ]
                )
            except Exception as exc:
                lines.append(f"selected_step_error={exc}")
        self.detail_text = "\n".join(lines)
        self._info("[PLAYBACK DEBUG] " + self.detail_text.replace("\n", " | "))

    def _handle_combine(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "mode"
        if sub == "mode":
            self.combine_mode_enabled = True
            self._info("[INFO] Combine mode enabled. Select contiguous steps, then Combine Selected Steps.")
        elif sub == "cancel":
            self.combine_mode_enabled = False
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self._info("[INFO] Combine mode cancelled.")
        elif sub == "clear":
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
        elif sub == "add":
            self.add_to_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub == "remove":
            self.remove_from_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub == "toggle":
            self.toggle_combine_selection([int(token) for token in args[1:] if token.isdigit()])
        elif sub in {"range", "select_range", "contiguous"}:
            self.select_contiguous_combine_range()
        elif sub == "preview":
            self.preview_combine_steps()
        elif sub == "play_preview":
            self.play_combine_preview()
        elif sub in {"commit", "selected"}:
            self.commit_combine_steps()

    def set_combine_selection(self, indices: list[int]) -> None:
        self.combine_selected_indices = set(int(index) for index in indices)
        if self.combine_mode_enabled:
            self.combine_preview_text = self.combine_selection_status_text()

    def add_to_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            if 1 <= int(index) <= self.manager.count:
                self.combine_selected_indices.add(int(index))
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def remove_from_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            self.combine_selected_indices.discard(int(index))
        self.combine_preview_text = self.combine_selection_status_text()

    def toggle_combine_selection(self, indices: list[int]) -> None:
        for index in indices:
            index = int(index)
            if index in self.combine_selected_indices:
                self.combine_selected_indices.remove(index)
            elif 1 <= index <= self.manager.count:
                self.combine_selected_indices.add(index)
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def select_contiguous_combine_range(self) -> None:
        selected = sorted(self.combine_selected_indices)
        if len(selected) < 2:
            self.combine_preview_text = "Select at least two steps before selecting a contiguous range."
            return
        self.combine_selected_indices = set(range(selected[0], selected[-1] + 1))
        self.combine_mode_enabled = True
        self.combine_preview_text = self.combine_selection_status_text()

    def combine_selection_status_text(self) -> str:
        selected = sorted(self.combine_selected_indices)
        contiguous = self.combine_selection_is_contiguous()
        return (
            f"Selected combine indices: {selected}\n"
            f"Selected count: {len(selected)}\n"
            f"Contiguous: {contiguous}\n"
            + (
                "Click Preview Combined Step to compute a compact preview."
                if len(selected) >= 2 and contiguous
                else "Combine selection must be contiguous. Use Select Contiguous Range."
                if len(selected) >= 2
                else "Select at least two steps."
            )
        )

    def preview_combine_steps(self, *, silent: bool = False) -> None:
        selected = sorted(self.combine_selected_indices)
        if len(selected) < 2:
            self.combine_preview_text = "Select at least two steps."
            return
        ok, reason = self.can_combine()
        if not ok:
            self.combine_preview_text = reason
            if not silent:
                self._warn("[WARN] " + reason)
            return
        try:
            steps = [self.manager.get_step(index) for index in selected]
            from sequence_model import build_combined_step

            combined = build_combined_step(steps, allow_conflicts=self.allow_combine_conflicts)
            self.combine_preview_text = self.compact_step_details(combined, title=f"Combined preview {selected[0]:03d}..{selected[-1]:03d}")
            if not silent:
                self._info(f"[INFO] Combine preview ready for steps {selected}.")
        except Exception as exc:
            self.combine_preview_text = f"Combine preview failed: {exc}"
            if not silent:
                self._warn("[WARN] " + self.combine_preview_text)

    def commit_combine_steps(self) -> None:
        selected = sorted(self.combine_selected_indices)
        ok, reason = self.can_combine()
        if not ok:
            self._warn("[WARN] " + reason)
            return

        def work() -> dict[str, Any]:
            combined = self.manager.replace_step_range_with_combined(selected, allow_conflicts=self.allow_combine_conflicts)
            self.selected_step_index = int(combined["index"])
            self.set_step_summary_detail(combined, title=f"Combined step {int(combined['index']):03d}")
            self.combine_mode_enabled = False
            self.combine_selected_indices.clear()
            self.combine_preview_text = ""
            self._info(f"[INFO] Combined steps {selected} into step {int(combined['index']):03d}.")
            return combined

        self._run_operation("Combine selected steps", work)

    def play_combine_preview(self) -> None:
        selected = sorted(self.combine_selected_indices)
        if len(selected) < 2:
            self._warn("[WARN] Select at least two steps to play a combined preview.")
            return
        ok, reason = self.can_playback()
        if not ok:
            self._warn("[WARN] " + reason)
            return
        try:
            from sequence_model import build_combined_step

            combined = build_combined_step(
                [self.manager.get_step(index) for index in selected],
                allow_conflicts=self.allow_combine_conflicts,
            )
            self.start_playback(
                [combined],
                label=f"combined preview {selected}",
                profile=self.playback.profile,
                restore_start_state=self.restore_step_start_state_before_selected_playback,
            )
        except Exception as exc:
            self._warn(f"[WARN] Could not play combined preview: {exc}")

    def export_motion_txt(self) -> Path | None:
        if self.manager.count == 0:
            self._warn("[WARN] No accepted steps to export.")
            return None
        path = self.store.height_dir(self.current_height_cm) / "accepted_motion.txt"
        lines: list[str] = []
        for step in self.manager.steps:
            normalized = normalize_step(step)
            lines.append(f"# step {int(normalized['index']):03d} {normalized['name']}")
            for event in normalized.get("events", []):
                command = str(event.get("command", "")).strip()
                if command:
                    lines.append(f"at {float(event.get('time', 0.0)):.3f} {command}")
            lines.append(f"wait {float(normalized.get('duration', 0.0)):.3f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._info(f"[INFO] Exported motion TXT: {path}")
        return path

    def set_playback_options(
        self,
        *,
        profile: str,
        speed: float,
        trailing_pad: float,
        max_idle_gap: float | None,
        preserve_wheel_distance: bool | None = None,
    ) -> None:
        if preserve_wheel_distance is not None:
            self.motion_scale.preserve_wheel_distance = bool(preserve_wheel_distance)
            self.playback.preserve_wheel_distance = self.motion_scale.preserve_wheel_distance
        effective_speed = float(speed) * self.motion_scale.playback_scale
        effective_speed = max(0.1, min(5.0, effective_speed))
        self.playback.set_profile(profile)
        self.playback.set_speed(effective_speed)
        self.playback.set_trailing_pad(trailing_pad)
        self.playback.set_max_idle_gap(max_idle_gap)

    def set_motion_scale(
        self,
        *,
        global_motion_speed_scale: float | None = None,
        wheel_speed_scale: float | None = None,
        servo_command_scale: float | None = None,
        playback_speed_scale: float | None = None,
        apply_to_manual_control: bool | None = None,
        apply_to_playback: bool | None = None,
        preserve_wheel_distance: bool | None = None,
    ) -> None:
        if global_motion_speed_scale is not None:
            self.motion_scale.global_motion_speed_scale = max(0.0, float(global_motion_speed_scale))
        if wheel_speed_scale is not None:
            self.motion_scale.wheel_speed_scale = max(0.0, float(wheel_speed_scale))
        if servo_command_scale is not None:
            self.motion_scale.servo_command_scale = max(0.0, float(servo_command_scale))
        if playback_speed_scale is not None:
            self.motion_scale.playback_speed_scale = max(0.0, float(playback_speed_scale))
        if apply_to_manual_control is not None:
            self.motion_scale.apply_to_manual_control = bool(apply_to_manual_control)
        if apply_to_playback is not None:
            self.motion_scale.apply_to_playback = bool(apply_to_playback)
        if preserve_wheel_distance is not None:
            self.motion_scale.preserve_wheel_distance = bool(preserve_wheel_distance)
        self.playback.preserve_wheel_distance = self.motion_scale.preserve_wheel_distance

    def reset_motion_scale(self) -> None:
        self.motion_scale = MotionScaleConfig()
        self.playback.preserve_wheel_distance = self.motion_scale.preserve_wheel_distance

    def _info(self, text: str) -> None:
        self.status = text
        self.status_log.append(text)
        print(text)

    def _warn(self, text: str) -> None:
        self.status = text
        self.status_log.append(text)
        print(text)


class RealRobotStyleHeightReplayUi:
    """Tk UI that mirrors the real_robot_ui_controller layout."""

    def __init__(self, controller: HeightReplayController, *, smoke_test_ms: int = 0):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.controller = controller
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title("Height Based Obstacle Replay - Isaac Sim")
        self.root.protocol("WM_DELETE_WINDOW", self._window_close)
        self.root.bind("<space>", lambda _event: self._post("e_stop") or "break")
        self.updating = False
        self.slider_dragging: set[str] = set()
        self.slider_pending: dict[str, tuple[str, float, float, bool]] = {}
        self.slider_after_ids: dict[str, Any] = {}
        self.slider_last_sent: dict[str, tuple[float, float]] = {}
        self.selected_step_index: int | None = None
        self.refreshing = False
        self._last_playback_guard_reason = ""
        self.ui_refresh_ms = max(20, int(getattr(controller.args, "ui_refresh_ms", 100)))
        self.sim_status_refresh_ms = max(50, int(getattr(controller.args, "sim_status_refresh_ms", 250)))
        self.full_refresh_ms = max(250, int(getattr(controller.args, "full_refresh_ms", 1000)))
        self.max_text_widget_chars = max(1000, int(getattr(controller.args, "max_text_widget_chars", 200000)))
        self.disable_auto_sim_state_json = bool(getattr(controller.args, "disable_auto_sim_state_json", False))
        self.sim_state_json_on_demand = bool(getattr(controller.args, "sim_state_json_on_demand", True))
        self._last_medium_refresh = 0.0
        self._last_full_refresh = 0.0
        self._last_sequence_revision: int | None = None
        self._last_manifest_revision: int | None = None
        self._last_detail_text = ""
        self._last_combine_preview = ""
        self._text_widget_cache: dict[int, tuple[int, int]] = {}
        self._guarded_buttons: list[tuple[Any, str]] = []

        self.servo_vars: dict[str, Any] = {}
        self.servo_output_label_vars: dict[str, Any] = {}
        self.wheel_vars: dict[str, Any] = {}
        self.playback_buttons: list[Any] = []

        self.summary_var = tk.StringVar(value=controller.status_line())
        self.sim_label_var = tk.StringVar(value="Isaac Sim: starting")
        self.mode_label_var = tk.StringVar(value="Mode: TEST")
        self.sequence_label_var = tk.StringVar(value="")
        self.accepted_steps_source_var = tk.StringVar(value="")
        self.task_label_var = tk.StringVar(value="")
        self.playback_label_var = tk.StringVar(value="Playback: inactive")
        self.record_label_var = tk.StringVar(value="Record: idle")
        self.busy_label_var = tk.StringVar(value="")
        self.pending_label_var = tk.StringVar(value="Pending: none")
        self.replacement_label_var = tk.StringVar(value="Replacement: none")
        self.wheel_status_var = tk.StringVar(value="Wheel command: fl=0 fr=0 rl=0 rr=0")
        self.command_var = tk.StringVar(value="")
        default_speed = min(0.30, float(controller.default_wheel_speed))
        self.left_speed_var = tk.StringVar(value=f"{default_speed:.2f}")
        self.right_speed_var = tk.StringVar(value=f"{default_speed:.2f}")
        self.height_var = tk.StringVar(value=str(controller.current_height_cm))
        self.play_profile_var = tk.StringVar(value="fast")
        self.play_speed_var = tk.StringVar(value="1.0")
        self.play_trailing_pad_var = tk.StringVar(value="0.05")
        self.play_max_idle_gap_var = tk.StringVar(value="none")
        self.restore_step_start_var = tk.BooleanVar(value=controller.restore_step_start_state_before_selected_playback)
        self.restore_full_sim_pose_var = tk.BooleanVar(value=controller.restore_full_sim_pose_if_available)
        self.fallback_command_state_var = tk.BooleanVar(value=controller.fallback_to_command_state_before)
        self.preserve_distance_var = tk.BooleanVar(value=controller.motion_scale.preserve_wheel_distance)
        self.global_scale_var = tk.StringVar(value=f"{controller.motion_scale.global_motion_speed_scale:.2f}")
        self.wheel_scale_var = tk.StringVar(value=f"{controller.motion_scale.wheel_speed_scale:.2f}")
        self.servo_scale_var = tk.StringVar(value=f"{controller.motion_scale.servo_command_scale:.2f}")
        self.playback_scale_var = tk.StringVar(value=f"{controller.motion_scale.playback_speed_scale:.2f}")
        self.apply_manual_scale_var = tk.BooleanVar(value=controller.motion_scale.apply_to_manual_control)
        self.apply_playback_scale_var = tk.BooleanVar(value=controller.motion_scale.apply_to_playback)
        self.speed_scale_label_var = tk.StringVar(value="")
        self.clear_confirm_var = tk.BooleanVar(value=False)
        self.combine_allow_conflicts_var = tk.BooleanVar(value=False)
        self.combine_selection_var = tk.StringVar(value="Selected combine indices: []")
        self.quick_hint_var = tk.StringVar(value="")
        self.vision_source_var = tk.StringVar(value="Generated Test Obstacle")
        self.vision_test_height_var = tk.StringVar(value="5")
        self.vision_task_status_var = tk.StringVar(value="Vision Task Active: false")
        self.vision_detection_status_var = tk.StringVar(value="")
        self.vision_steps_status_var = tk.StringVar(value="")
        self.vision_high_level_status_var = tk.StringVar(value="")
        self.vision_block_reasons_var = tk.StringVar(value="")
        self.vision_respawn_before_replay_var = tk.BooleanVar(value=controller.vision_respawn_before_replay)
        self.vision_details_auto_refresh_var = tk.BooleanVar(value=True)
        self.vision_details_paused = False

        self.height_mode_var = tk.StringVar(value="recording")
        self.height_selected_vars = {height: tk.BooleanVar(value=height == controller.current_height_cm) for height in SUPPORTED_HEIGHTS_CM}

        self._build()
        self._poll()
        if bool(getattr(controller.args, "smoke_accept_recording", False)):
            self.root.after(120, self._smoke_accept_recording)
        if smoke_test_ms > 0:
            self.root.after(smoke_test_ms, self._window_close)

    def run(self) -> None:
        self.root.mainloop()

    def _window_close(self) -> None:
        self.controller.shutdown()
        self.root.destroy()

    def _post(self, text: str, *, kind: str = "command", target: str = "", quiet: bool = False) -> None:
        self.busy_label_var.set("Busy..." if self.controller.operation_busy else "")
        self.root.update_idletasks()
        if text.strip().split()[:1] == ["playback_debug_selected"]:
            try:
                self.controller._info(
                    "[PLAYBACK DEBUG] ui selected "
                    f"tree_selection={list(self.steps_tree.selection())} "
                    f"tree_indices={self._selected_indices()} "
                    f"controller_selected={self.controller.selected_step_index} "
                    f"manager_count={self.controller.manager.count}"
                )
            except Exception as exc:
                self.controller._warn(f"[WARN] UI selected debug failed: {exc}")
        self.controller.handle_command(CommandMessage(text=text, source="ui", kind=kind, target=target, quiet=quiet))
        self._refresh(force=False)

    def _smoke_accept_recording(self) -> None:
        for command in [
            "step_record start",
            "servo front_left_hip 5",
            "wheel fl 0.1",
            "step_record stop",
            "step_record accept",
        ]:
            self.controller.handle_command(CommandMessage(text=command, source="ui_smoke", quiet=True))
        if self.controller.manager.count < 1:
            self.controller._warn("[WARN] Smoke accept recording did not create an accepted step.")
        else:
            self.controller._info("[INFO] Smoke accept recording created one accepted step.")
        self._refresh(force=False)

    def _post_slider(self, key: str, text: str, value: float, threshold: float, *, final: bool = False) -> None:
        now = time.monotonic()
        last = self.slider_last_sent.get(key)
        if not final and last is not None:
            last_time, last_value = last
            if abs(value - last_value) < threshold and now - last_time < 0.05:
                self.slider_pending[key] = (text, value, threshold, final)
                return
        self.slider_last_sent[key] = (now, value)
        self.controller.handle_command(CommandMessage(text=text, source="ui", kind="slider", target=key, log_history=final, quiet=True))

    def _schedule_slider(self, key: str, text: str, value: float, threshold: float) -> None:
        self.slider_pending[key] = (text, value, threshold, False)
        if key not in self.slider_after_ids:
            self.slider_after_ids[key] = self.root.after(40, lambda slider_key=key: self._flush_slider(slider_key))

    def _flush_slider(self, key: str, *, final: bool = False) -> None:
        after_id = self.slider_after_ids.pop(key, None)
        if after_id is not None and final:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
        pending = self.slider_pending.pop(key, None)
        if pending is None:
            return
        text, value, threshold, _was_final = pending
        self._post_slider(key, text, value, threshold, final=final)

    def _slider_release(self, key: str) -> None:
        self.slider_dragging.discard(key)
        self._flush_slider(key, final=True)

    def _build(self) -> None:
        self.root.minsize(1500, 860)
        self.root.geometry("1700x980")
        container = self.ttk.Frame(self.root, padding=8)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        summary = self.ttk.Frame(container)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for col, var in enumerate(
            [
                self.sim_label_var,
                self.mode_label_var,
                self.sequence_label_var,
                self.task_label_var,
                self.playback_label_var,
                self.record_label_var,
                self.busy_label_var,
            ]
        ):
            self.ttk.Label(summary, textvariable=var).grid(row=0, column=col, sticky="w", padx=(0, 14))
        self.ttk.Label(summary, textvariable=self.summary_var).grid(row=1, column=0, columnspan=7, sticky="ew")
        summary.columnconfigure(6, weight=1)

        main = self.ttk.PanedWindow(container, orient="horizontal")
        main.grid(row=1, column=0, sticky="nsew")
        left = self._make_scroll_column(main, width=455)
        center = self._make_scroll_column(main, width=650)
        right = self.ttk.Frame(main, width=620)
        self._add_pane(main, right, weight=3)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._build_live_column(left)
        self._build_steps_column(center)
        self._build_right_notebook(right)

    def _add_pane(self, paned: Any, child: Any, *, weight: int) -> None:
        try:
            paned.add(child, weight=weight)
        except Exception:
            paned.add(child)

    def _make_scroll_column(self, parent: Any, *, width: int) -> Any:
        shell = self.ttk.Frame(parent, width=width)
        self._add_pane(parent, shell, weight=2)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(shell, highlightthickness=0)
        ybar = self.ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        xbar = self.ttk.Scrollbar(shell, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        body = self.ttk.Frame(canvas, padding=(0, 0, 4, 0))
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def sync(_event: Any | None = None) -> None:
            canvas.itemconfigure(window_id, width=max(canvas.winfo_width(), body.winfo_reqwidth()))
            canvas.configure(scrollregion=canvas.bbox("all"))

        body.bind("<Configure>", sync)
        canvas.bind("<Configure>", sync)
        canvas.bind("<MouseWheel>", lambda event, c=canvas: self._scroll_canvas(c, event))
        body.columnconfigure(0, weight=1)
        return body

    def _register_guarded_button(self, button: Any, guard: str) -> Any:
        self._guarded_buttons.append((button, guard))
        return button

    def _button_allowed(self, guard: str) -> bool:
        if guard == "e_stop":
            return True
        if guard == "manual_respawn":
            return (
                (self.controller.no_sim or self.controller.runtime_ready)
                and not self.controller.recording_active
                and self.controller.pending_step is None
                and self.controller.pending_replacement is None
                and not self.controller.operation_busy
            )
        if self.controller.operation_busy:
            return False
        if guard == "start_record":
            return self.controller.can_start_recording()[0]
        if guard == "accept_record":
            return self.controller.can_accept_recorded_step()[0]
        if guard == "prepare_replace":
            return self.controller.can_prepare_replacement()[0] and self._selected_index() is not None
        if guard == "accept_replace":
            return self.controller.can_accept_replacement()[0]
        if guard == "playback":
            return self.controller.can_playback()[0]
        if guard == "combine":
            return self.controller.can_combine()[0]
        if guard == "save":
            return self.controller.can_save()[0]
        if guard == "show_step":
            return self._selected_index() is not None and not self.controller.recording_active and not self.controller.playback.active
        if guard == "height_edit":
            return not self.controller.visible_steps_are_read_only()
        if guard == "start_vision_task":
            return self.controller.can_start_vision_task()[0]
        if guard == "generate_vision_obstacle":
            return (
                self.controller.vision_task_active
                and self.controller.vision_source_mode == VISION_SOURCE_GENERATED
                and self.controller._can_generate_vision_obstacle()[0]
            )
        if guard == "validate_generated_height":
            return bool(self.controller.vision_task_active and self.controller.vision_source_mode == VISION_SOURCE_GENERATED)
        if guard == "play_vision_steps":
            return bool(
                self.controller.vision_steps_ready
                and self.controller.vision_steps
                and self.controller.playback_readiness(respawn_first=bool(self.controller.vision_respawn_before_replay))[0]
            )
        if guard == "arm_vision_auto":
            return self.controller.mode != MODE_E_STOP and not self.controller.playback.active and not self.controller.operation_busy
        if guard == "show_camera_view":
            return self.controller.camera_view_readiness()[0]
        if guard == "idle":
            return not self.controller.recording_active and not self.controller.playback.active
        return True

    def _button_guard_reason(self, guard: str) -> str:
        if guard == "e_stop":
            return "ok"
        if guard == "manual_respawn":
            if not self.controller.no_sim and not self.controller.runtime_ready:
                return "Isaac runtime is not ready."
            if self.controller.recording_active:
                return "Stop recording before Respawn."
            if self.controller.pending_step is not None or self.controller.pending_replacement is not None:
                return "Accept or discard pending step/replacement before Respawn."
            if self.controller.operation_busy:
                return f"Operation busy: {self.controller.busy_name}"
            return ""
        if self.controller.operation_busy:
            return f"Operation busy: {self.controller.busy_name}"
        if guard == "start_record":
            return self.controller.can_start_recording()[1]
        if guard == "accept_record":
            return self.controller.can_accept_recorded_step()[1]
        if guard == "prepare_replace":
            ok, reason = self.controller.can_prepare_replacement()
            if ok and self._selected_index() is None:
                return "Select an accepted step first."
            return reason
        if guard == "accept_replace":
            return self.controller.can_accept_replacement()[1]
        if guard == "playback":
            return self.controller.can_playback()[1]
        if guard == "combine":
            return self.controller.can_combine()[1]
        if guard == "save":
            return self.controller.can_save()[1]
        if guard == "show_step":
            if self._selected_index() is None:
                return "Select an accepted step first."
            if self.controller.recording_active:
                return "Stop recording first."
            if self.controller.playback.active:
                return "Stop playback first."
        if guard == "height_edit" and self.controller.visible_steps_are_read_only():
            return "Vision steps are read-only."
        if guard == "start_vision_task":
            return self.controller.can_start_vision_task()[1]
        if guard == "generate_vision_obstacle":
            if not self.controller.vision_task_active:
                return "Start Vision Task first."
            if self.controller.vision_source_mode != VISION_SOURCE_GENERATED:
                return "Generated obstacle is disabled in External / Unknown mode."
            return self.controller._can_generate_vision_obstacle()[1]
        if guard == "validate_generated_height":
            return self.controller.can_validate_current_generated_height()[1]
        if guard == "play_vision_steps" and not self.controller.vision_steps_ready:
            return self.controller.vision_steps_error or "Vision steps are not ready."
        if guard == "play_vision_steps":
            return self.controller.playback_readiness(respawn_first=bool(self.controller.vision_respawn_before_replay))[1]
        if guard == "arm_vision_auto" and self.controller.mode == MODE_E_STOP:
            return "E-stop is active."
        if guard == "show_camera_view":
            return self.controller.camera_view_readiness()[1]
        if guard == "idle" and (self.controller.recording_active or self.controller.playback.active):
            return "Wait until recording/playback is idle."
        return ""

    def _refresh_button_states(self) -> None:
        playback_guard_reason = ""
        for button, guard in self._guarded_buttons:
            try:
                allowed = self._button_allowed(guard)
                button.configure(state="normal" if allowed else "disabled")
                if guard == "playback" and not allowed and not playback_guard_reason:
                    playback_guard_reason = self._button_guard_reason(guard)
            except Exception:
                pass
        self._last_playback_guard_reason = playback_guard_reason

    def _build_live_column(self, parent: Any) -> None:
        servo_frame = self.ttk.LabelFrame(parent, text="Servos")
        wheel_frame = self.ttk.LabelFrame(parent, text="Wheels")
        quick_frame = self.ttk.LabelFrame(parent, text="Quick Commands")
        servo_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        wheel_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        quick_frame.grid(row=2, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        self._build_servo_panel(servo_frame, slider_length=220)
        self._build_wheel_panel(wheel_frame, slider_length=220)
        self._build_quick_commands(quick_frame)

    def _build_servo_panel(self, parent: Any, *, slider_length: int) -> None:
        for row, name in enumerate(SERVO_JOINT_NAMES):
            low, high = KNEE_LIMIT_DEG if name in KNEE_JOINT_NAMES else HIP_LIMIT_DEG
            var = self.tk.DoubleVar(value=0.0)
            out_var = self.tk.StringVar(value="out 0.0")
            self.servo_vars[name] = var
            self.servo_output_label_vars[name] = out_var
            self.ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4, pady=1)
            scale = self.tk.Scale(
                parent,
                from_=low,
                to=high,
                orient="horizontal",
                resolution=0.1,
                length=slider_length,
                variable=var,
                command=lambda value, joint=name: self._servo_slider(joint, value),
            )
            scale.grid(row=row, column=1, sticky="ew", pady=1)
            self.ttk.Label(parent, textvariable=out_var, width=14).grid(row=row, column=2, sticky="w", padx=(4, 2), pady=1)
            scale.bind("<ButtonPress-1>", lambda _event, key=name: self.slider_dragging.add(key))
            scale.bind("<ButtonRelease-1>", lambda _event, key=name: self._slider_release(key))
            scale.bind("<MouseWheel>", self._on_scale_mousewheel)
        parent.columnconfigure(1, weight=1)

    def _build_wheel_panel(self, parent: Any, *, slider_length: int) -> None:
        for row, short_name in enumerate(("fl", "fr", "rl", "rr")):
            var = self.tk.DoubleVar(value=0.0)
            self.wheel_vars[short_name] = var
            self.ttk.Label(parent, text=f"{short_name} wheel").grid(row=row, column=0, sticky="w", padx=4, pady=1)
            scale = self.tk.Scale(
                parent,
                from_=-float(self.controller.max_wheel_speed),
                to=float(self.controller.max_wheel_speed),
                orient="horizontal",
                resolution=0.01,
                length=slider_length,
                variable=var,
                command=lambda value, wheel=short_name: self._wheel_slider(wheel, value),
            )
            scale.grid(row=row, column=1, sticky="ew", pady=1)
            scale.bind("<ButtonPress-1>", lambda _event, key=short_name: self.slider_dragging.add(key))
            scale.bind("<ButtonRelease-1>", lambda _event, key=short_name: self._slider_release(key))
            scale.bind("<MouseWheel>", self._on_scale_mousewheel)
        parent.columnconfigure(1, weight=1)
        self.ttk.Button(parent, text="Stop Wheels", command=lambda: self._post("wheel stop")).grid(row=5, column=0, sticky="ew", padx=3, pady=(10, 2))
        self.ttk.Button(parent, text="All Forward", command=lambda: self._post(f"wheel all {float(self.left_speed_var.get()):.3f}")).grid(row=5, column=1, sticky="ew", padx=3, pady=(10, 2))
        self.ttk.Button(parent, text="All Backward", command=lambda: self._post(f"wheel all {-float(self.left_speed_var.get()):.3f}")).grid(row=6, column=1, sticky="ew", padx=3, pady=2)
        self.ttk.Label(parent, text="left").grid(row=7, column=0, sticky="e")
        self.ttk.Entry(parent, textvariable=self.left_speed_var, width=8).grid(row=7, column=1, sticky="w")
        self.ttk.Label(parent, text="right").grid(row=8, column=0, sticky="e")
        self.ttk.Entry(parent, textvariable=self.right_speed_var, width=8).grid(row=8, column=1, sticky="w")
        self.ttk.Button(parent, text="Set Pair", command=lambda: self._post(f"wheel {float(self.left_speed_var.get()):.3f} {float(self.right_speed_var.get()):.3f}")).grid(row=9, column=0, columnspan=2, sticky="ew", padx=3, pady=2)
        self.ttk.Label(parent, textvariable=self.wheel_status_var, wraplength=slider_length + 160, foreground="#8a3b00").grid(row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 2))

    def _build_quick_commands(self, parent: Any) -> None:
        self.ttk.Entry(parent, textvariable=self.command_var).grid(row=0, column=0, columnspan=4, sticky="ew", padx=3, pady=3)
        self.ttk.Button(parent, text="Run Command", command=self._run_command).grid(row=0, column=4, sticky="ew", padx=3, pady=3)
        buttons = [
            ("TEST Mode", "mode test", ""),
            ("Home", "home", ""),
            ("Status", "status", ""),
            ("E-stop", "e_stop", "e_stop"),
            ("↻ Respawn", "respawn", "manual_respawn"),
        ]
        for col, (label, command, guard) in enumerate(buttons):
            button = self.ttk.Button(
                parent,
                text=label,
                command=lambda text=command: self._post(text),
            )
            if label.endswith("Respawn"):
                button.configure(command=lambda text=command: self._post(text))
                button.bind("<Enter>", lambda _event: self.quick_hint_var.set("Respawn robot to its initial simulation pose. This does not clear E-stop."))
                button.bind("<Leave>", lambda _event: self.quick_hint_var.set(""))
            if guard:
                self._register_guarded_button(button, guard)
            button.grid(row=1, column=col, sticky="ew", padx=3, pady=3)
        self.ttk.Label(parent, textvariable=self.quick_hint_var, wraplength=420).grid(row=2, column=0, columnspan=5, sticky="ew", padx=3, pady=(1, 3))
        for col in range(5):
            parent.columnconfigure(col, weight=1, uniform="quick")

    def _build_steps_column(self, parent: Any) -> None:
        steps_frame = self.ttk.LabelFrame(parent, text="Accepted Steps")
        actions_frame = self.ttk.LabelFrame(parent, text="Accepted Step Actions")
        details_frame = self.ttk.LabelFrame(parent, text="Step Details")
        steps_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        actions_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        details_frame.grid(row=2, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(2, weight=2)
        self._build_steps_tree(steps_frame)
        self._build_step_actions(actions_frame)
        self.detail_text = self._make_scrolled_text(details_frame, row=0, height=13, width=74)

    def _build_steps_tree(self, parent: Any) -> None:
        columns = ("index", "name", "type", "duration", "events", "note")
        self.ttk.Label(parent, textvariable=self.accepted_steps_source_var).grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=(0, 3))
        self.steps_tree = self.ttk.Treeview(parent, columns=columns, show="headings", selectmode="extended", height=15)
        headings = {"index": "#", "name": "name", "type": "type", "duration": "duration", "events": "events count", "note": "note"}
        widths = {"index": 44, "name": 190, "type": 80, "duration": 76, "events": 88, "note": 220}
        for column in columns:
            self.steps_tree.heading(column, text=headings[column])
            self.steps_tree.column(column, width=widths[column], stretch=column in {"name", "note"})
        tree_y = self.ttk.Scrollbar(parent, orient="vertical", command=self.steps_tree.yview)
        tree_x = self.ttk.Scrollbar(parent, orient="horizontal", command=self.steps_tree.xview)
        self.steps_tree.configure(yscrollcommand=tree_y.set, xscrollcommand=tree_x.set)
        self.steps_tree.grid(row=1, column=0, sticky="nsew")
        tree_y.grid(row=1, column=1, sticky="ns")
        tree_x.grid(row=2, column=0, sticky="ew")
        self.steps_tree.bind("<<TreeviewSelect>>", self._on_step_selected)
        self.steps_tree.bind("<Double-1>", lambda _event: self._selected_step_command("inspect_step {index}"))
        self.steps_tree.bind("<MouseWheel>", self._on_tree_mousewheel)
        self.steps_tree.tag_configure("combine_selected", background="#d8ecff")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

    def _build_step_actions(self, parent: Any) -> None:
        action_buttons = [
            ("Show Selected Before", lambda: self._selected_step_command("show_step {index} before")),
            ("Show Selected After", lambda: self._selected_step_command("show_step {index} after")),
            ("Show Step Summary", lambda: self._selected_step_command("show_step_summary {index}")),
            ("Show Step JSON", lambda: self._selected_step_command("inspect_step {index}")),
            ("Show JSON Truncated", lambda: self._selected_step_command("inspect_step {index}")),
            ("Export Step JSON", lambda: self._selected_step_command("export_step_json {index}")),
            ("Replace Selected Step", self._replace_selected),
            ("Delete Selected Step", self._delete_selected),
            ("Undo", lambda: self._post("undo")),
            ("Clear All Steps", self._clear_all_steps),
        ]
        guard_by_label = {
            "Show Selected Before": "show_step",
            "Show Selected After": "show_step",
            "Show Step Summary": "show_step",
            "Show Step JSON": "show_step",
            "Show JSON Truncated": "show_step",
            "Export Step JSON": "show_step",
            "Replace Selected Step": "prepare_replace",
            "Delete Selected Step": "height_edit",
            "Undo": "height_edit",
            "Clear All Steps": "height_edit",
        }
        for index, (label, callback) in enumerate(action_buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in guard_by_label:
                self._register_guarded_button(button, guard_by_label[label])
            button.grid(row=index // 3, column=index % 3, sticky="ew", padx=2, pady=2)
        self.ttk.Checkbutton(parent, text="Confirm Clear All", variable=self.clear_confirm_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        for col in range(3):
            parent.columnconfigure(col, weight=1)

    def _build_right_notebook(self, parent: Any) -> None:
        notebook = self.ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.right_notebook = notebook
        for tab_name, builder in [
            ("Sim Connection", self._build_connection_tab),
            ("Run Manager", self._build_run_manager_tab),
            ("Record / Servo+Wheel", self._build_record_servo_wheel_tab),
            ("Speed Scale", self._build_speed_scale_tab),
            ("Playback", self._build_playback_tab),
            ("Height Task", self._build_height_task_tab),
            ("Combine", self._build_combine_tab),
            ("Vision Auto Replay", self._build_vision_auto_replay_tab),
            ("Sim State", self._build_sim_state_tab),
        ]:
            tab = self.ttk.Frame(notebook, padding=2)
            notebook.add(tab, text=tab_name)
            body = self._make_scrollable_tab_body(tab)
            builder(body)

    def _make_scrollable_tab_body(self, parent: Any) -> Any:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = self.tk.Canvas(parent, highlightthickness=0)
        ybar = self.ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=ybar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        body = self.ttk.Frame(canvas, padding=6)
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def sync(_event: Any | None = None) -> None:
            canvas.itemconfigure(window_id, width=max(canvas.winfo_width(), body.winfo_reqwidth()))
            canvas.configure(scrollregion=canvas.bbox("all"))

        body.bind("<Configure>", sync)
        canvas.bind("<Configure>", sync)
        canvas.bind("<MouseWheel>", lambda event, c=canvas: self._scroll_canvas(c, event))
        body.columnconfigure(0, weight=1)
        return body

    def _build_connection_tab(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Isaac Sim Connection")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Run Isaac Preflight", self._run_isaac_preflight),
            ("Start / Ensure Isaac Sim", self._ensure_sim),
            ("Restart Isaac Worker", self._restart_isaac_worker),
            ("Restart Without Onboard Camera", self._restart_without_onboard_camera),
            ("Generate / Load Current Height Obstacle", self._generate_current_obstacle),
            ("Respawn Robot", lambda: self._post("respawn")),
            ("Open Worker Log Folder", self._open_worker_log_folder),
            ("Copy Worker Command", self._copy_worker_command),
            ("Refresh Worker Logs", lambda: self._refresh(force=True)),
            ("Stop Worker", self._stop_worker),
            ("Stop Wheels", lambda: self._post("wheel stop")),
        ]
        for index, (label, callback) in enumerate(buttons):
            self.ttk.Button(frame, text=label, command=callback).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        self.connection_text = self._make_scrolled_text(parent, row=1, height=16, width=72)

    def _build_run_manager_tab(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Height Step Store")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        buttons = [
            ("Load Steps For Current Height", self._load_current_height_steps),
            ("Save Steps For Current Height", self._save_current_height_steps),
            ("New Empty Sequence For Current Height", self._new_empty_current_height),
            ("Refresh Manifest", self._refresh_manifest),
        ]
        guard_by_label = {
            "Load Steps For Current Height": "idle",
            "Save Steps For Current Height": "save",
            "New Empty Sequence For Current Height": "idle",
        }
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(frame, text=label, command=callback)
            if label in guard_by_label:
                self._register_guarded_button(button, guard_by_label[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.manifest_tree = self._build_manifest_tree(parent, row=1)

    def _build_record_servo_wheel_tab(self, parent: Any) -> None:
        sw = self.ttk.LabelFrame(parent, text="Servo+Wheel")
        sw.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        for index, (label, command) in enumerate(
            [
                ("Servo+Wheel Mode", "servo_wheel mode"),
                ("Launch Servo+Wheel", "servo_wheel launch"),
                ("Cancel Servo+Wheel Mode", "servo_wheel cancel"),
                ("Clear Servo+Wheel Staged", "servo_wheel clear_staged"),
            ]
        ):
            self.ttk.Button(sw, text=label, command=lambda text=command: self._post(text)).grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(sw, text="In Servo+Wheel Mode, sliders are staged only. Robot moves on Launch.", wraplength=540).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=(4, 2)
        )
        self.servo_wheel_preview_text = self._make_scrolled_text(parent, row=1, height=9, width=58)
        record = self.ttk.LabelFrame(parent, text="Record")
        record.grid(row=2, column=0, sticky="ew", pady=(6, 4))
        record_buttons = [
            ("Start Record Step", "step_record start"),
            ("Stop Record Step", "step_record stop"),
            ("Accept Recorded Step", "step_record accept"),
            ("Discard Pending Step", "step_record discard"),
            ("Discard Pending + Restore Before", "step_record discard_restore_before"),
        ]
        record_guards = {
            "Start Record Step": "start_record",
            "Accept Recorded Step": "accept_record",
        }
        for index, (label, command) in enumerate(record_buttons):
            button = self.ttk.Button(record, text=label, command=lambda text=command: self._post(text))
            if label in record_guards:
                self._register_guarded_button(button, record_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(record, textvariable=self.record_label_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        self.ttk.Label(record, textvariable=self.pending_label_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        replace = self.ttk.LabelFrame(parent, text="Replacement")
        replace.grid(row=3, column=0, sticky="ew")
        replace_buttons = [
            ("Prepare Replacement", self._replace_selected),
            ("Start Replacement Recording", lambda: self._post("replace_step start")),
            ("Stop Replacement Recording", lambda: self._post("replace_step stop")),
            ("Accept Replacement", lambda: self._post("replace_step accept")),
            ("Discard Replacement", lambda: self._post("replace_step discard")),
            ("Cancel Replacement", lambda: self._post("replace_step cancel")),
        ]
        replace_guards = {
            "Prepare Replacement": "prepare_replace",
            "Start Replacement Recording": "start_record",
            "Accept Replacement": "accept_replace",
        }
        for index, (label, callback) in enumerate(replace_buttons):
            button = self.ttk.Button(replace, text=label, command=callback)
            if label in replace_guards:
                self._register_guarded_button(button, replace_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(replace, textvariable=self.replacement_label_var).grid(row=3, column=0, columnspan=2, sticky="w", padx=3, pady=2)
        record.columnconfigure(0, weight=1)
        record.columnconfigure(1, weight=1)
        replace.columnconfigure(0, weight=1)
        replace.columnconfigure(1, weight=1)

    def _build_speed_scale_tab(self, parent: Any) -> None:
        frame = self.ttk.LabelFrame(parent, text="Speed Scale")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        rows = [
            ("global", self.global_scale_var),
            ("wheel", self.wheel_scale_var),
            ("servo command", self.servo_scale_var),
            ("playback", self.playback_scale_var),
        ]
        for row, (label, var) in enumerate(rows):
            self.ttk.Label(frame, text=label).grid(row=row, column=0, sticky="e", padx=3, pady=2)
            self.ttk.Entry(frame, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=3, pady=2)
        self.ttk.Checkbutton(frame, text="Apply to manual control", variable=self.apply_manual_scale_var, command=self._apply_speed_scale_controls).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=3, pady=(6, 2)
        )
        self.ttk.Checkbutton(frame, text="Apply to playback", variable=self.apply_playback_scale_var, command=self._apply_speed_scale_controls).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=3, pady=2
        )
        self.ttk.Checkbutton(frame, text="Preserve wheel distance", variable=self.preserve_distance_var, command=self._apply_speed_scale_controls).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=3, pady=2
        )
        self.ttk.Button(frame, text="Apply", command=self._apply_speed_scale_controls).grid(row=7, column=0, sticky="ew", padx=3, pady=(8, 2))
        self.ttk.Button(frame, text="Reset", command=self._reset_speed_scale_controls).grid(row=7, column=1, sticky="ew", padx=3, pady=(8, 2))
        self.ttk.Label(parent, textvariable=self.speed_scale_label_var, wraplength=560).grid(row=1, column=0, sticky="ew", padx=3, pady=4)
        parent.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_playback_tab(self, parent: Any) -> None:
        self.ttk.Radiobutton(parent, text="Motion-only / Fast", variable=self.play_profile_var, value="fast").grid(row=0, column=0, sticky="w")
        self.ttk.Radiobutton(parent, text="Raw", variable=self.play_profile_var, value="raw").grid(row=0, column=1, sticky="w")
        for row, (label, var) in enumerate(
            [("speed", self.play_speed_var), ("trailing pad", self.play_trailing_pad_var), ("max idle gap", self.play_max_idle_gap_var)],
            start=1,
        ):
            self.ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=2, pady=2)
            self.ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=2, pady=2)
        self.ttk.Checkbutton(parent, text="Preserve wheel distance", variable=self.preserve_distance_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 6))
        self.ttk.Checkbutton(parent, text="Restore step start state before selected playback", variable=self.restore_step_start_var, command=self._apply_playback_options).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(2, 2)
        )
        self.ttk.Checkbutton(parent, text="Restore full sim pose if available", variable=self.restore_full_sim_pose_var, command=self._apply_playback_options).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=2
        )
        self.ttk.Checkbutton(parent, text="Fallback to command_state_before", variable=self.fallback_command_state_var, command=self._apply_playback_options).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(2, 6)
        )
        buttons = [
            ("Play All", lambda: self._post_playback_command("play")),
            ("Play All Fast", lambda: self._post_playback_command("play fast")),
            ("Respawn And Play All", lambda: self._post_playback_command("respawn_play all")),
            ("Respawn And Play All Fast", lambda: self._post_playback_command("respawn_play all fast")),
            ("Play Selected Step", lambda: self._selected_step_playback_command("play_step {index}")),
            ("Play Selected Fast", lambda: self._selected_step_playback_command("play_step {index} fast")),
            ("Respawn And Play Selected Step", lambda: self._selected_step_playback_command("respawn_play selected {index}")),
            ("Respawn And Play Selected Fast", lambda: self._selected_step_playback_command("respawn_play selected {index} fast")),
            ("Play To Selected From Start", lambda: self._selected_step_playback_command("play_to_step {index}")),
            ("Respawn And Play To Selected From Start", lambda: self._selected_step_playback_command("respawn_play to {index}")),
            ("Pause Play", lambda: self._post("pause_play")),
            ("Resume Play", lambda: self._post("resume_play")),
            ("Stop Play", lambda: self._post("stop_play")),
            ("Analyze Playback Timing", lambda: self._post("analyze_playback_timing")),
            ("Debug Selected Playback", lambda: self._post("playback_debug_selected")),
            ("Export Motion TXT", lambda: self._post("export_motion_txt")),
        ]
        playback_guards = {
            "Play All": "playback",
            "Play All Fast": "playback",
            "Respawn And Play All": "playback",
            "Respawn And Play All Fast": "playback",
            "Play Selected Step": "playback",
            "Play Selected Fast": "playback",
            "Respawn And Play Selected Step": "playback",
            "Respawn And Play Selected Fast": "playback",
            "Play To Selected From Start": "playback",
            "Respawn And Play To Selected From Start": "playback",
        }
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in playback_guards:
                self._register_guarded_button(button, playback_guards[label])
            button.grid(row=8 + index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(parent, textvariable=self.playback_label_var).grid(row=16, column=0, columnspan=2, sticky="w", pady=(6, 0))
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def _build_height_task_tab(self, parent: Any) -> None:
        from height_task_panel import HeightTaskPanel

        self.height_task_panel = HeightTaskPanel(self, parent)

    def _build_combine_tab(self, parent: Any) -> None:
        buttons = [
            ("Combine Mode", lambda: self._post("combine mode")),
            ("Cancel Combine Mode", lambda: self._post("combine cancel")),
            ("Add Selected To Combine", self._combine_add_selected),
            ("Remove Selected From Combine", self._combine_remove_selected),
            ("Toggle Selected For Combine", self._combine_toggle_selected),
            ("Select Contiguous Range", self._combine_select_contiguous_range),
            ("Clear Combine Selection", lambda: self._post("combine clear")),
            ("Preview Combined Step", lambda: self._post("combine preview")),
            ("Play Combined Preview", lambda: self._post("combine play_preview")),
            ("Combine Selected Steps", lambda: self._post("combine commit")),
        ]
        combine_guards = {
            "Preview Combined Step": "combine",
            "Play Combined Preview": "playback",
            "Combine Selected Steps": "combine",
        }
        for index, (label, callback) in enumerate(buttons):
            button = self.ttk.Button(parent, text=label, command=callback)
            if label in combine_guards:
                self._register_guarded_button(button, combine_guards[label])
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Checkbutton(parent, text="Allow conflicts, later step wins", variable=self.combine_allow_conflicts_var, command=self._set_combine_allow_conflicts).grid(row=5, column=0, columnspan=2, sticky="w")
        self.ttk.Label(parent, textvariable=self.combine_selection_var, wraplength=560).grid(row=6, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        self.combine_preview_text = self._make_scrolled_text(parent, row=7, columnspan=2, height=9, width=58)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def _build_vision_auto_replay_tab(self, parent: Any) -> None:
        high = self.ttk.LabelFrame(parent, text="Vision Readiness")
        task = self.ttk.LabelFrame(parent, text="Vision Task")
        camera = self.ttk.LabelFrame(parent, text="Camera Controls")
        detection = self.ttk.LabelFrame(parent, text="Detection And Validation")
        steps = self.ttk.LabelFrame(parent, text="Vision Steps")
        auto = self.ttk.LabelFrame(parent, text="Auto Replay")
        high.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        task.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        camera.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        detection.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        steps.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        auto.grid(row=5, column=0, sticky="ew", pady=(0, 6))

        self.ttk.Label(high, textvariable=self.vision_high_level_status_var, wraplength=560).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        self.ttk.Label(high, textvariable=self.vision_block_reasons_var, wraplength=560, foreground="#8a3b00").grid(row=1, column=0, sticky="ew", padx=3, pady=(0, 3))
        high.columnconfigure(0, weight=1)

        self.ttk.Label(task, text="Vision Source").grid(row=0, column=0, sticky="w", padx=3, pady=2)
        source = self.ttk.Combobox(
            task,
            textvariable=self.vision_source_var,
            values=["Generated Test Obstacle", "External / Unknown Obstacle"],
            state="readonly",
            width=28,
        )
        source.grid(row=0, column=1, sticky="ew", padx=3, pady=2)
        source.bind("<<ComboboxSelected>>", lambda _event: self._vision_source_changed())
        self.ttk.Label(task, text="Test Obstacle Height").grid(row=1, column=0, sticky="w", padx=3, pady=2)
        self.ttk.Combobox(
            task,
            textvariable=self.vision_test_height_var,
            values=[str(height) for height in SUPPORTED_HEIGHTS_CM],
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="w", padx=3, pady=2)
        task_buttons = [
            ("Start Vision Task", self._vision_start_task, "start_vision_task"),
            ("Start And Generate Selected Vision Test", self._vision_start_and_generate_selected, ""),
            ("Generate Vision Test Obstacle", self._vision_generate_test_obstacle, "generate_vision_obstacle"),
            ("Finish Vision Task", self._vision_finish_task, ""),
        ]
        for index, (label, callback, guard) in enumerate(task_buttons):
            button = self.ttk.Button(task, text=label, command=callback)
            if guard:
                self._register_guarded_button(button, guard)
            button.grid(row=2 + index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Label(task, textvariable=self.vision_task_status_var, wraplength=560).grid(row=4, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        task.columnconfigure(1, weight=1)

        camera_buttons = [
            ("Enable Onboard Camera", lambda: self._vision_set_enabled(True), ""),
            ("Disable Onboard Camera", lambda: self._vision_set_enabled(False), ""),
            ("Detect Once", self._vision_detect_once, ""),
            ("Reset Detection Filter", self._vision_reset_filter, ""),
            ("Validate Isaac Camera", self._vision_validate_camera, ""),
            ("Validate Camera Geometry", self._vision_validate_camera_geometry, ""),
            ("Apply Recommended Oblique Camera Pose", self._vision_apply_recommended_camera_pose, ""),
            ("Restart Worker With Camera Pose", self._vision_restart_worker_with_camera_pose, ""),
            ("Open Onboard Camera Viewport", self._vision_open_camera_viewport, "show_camera_view"),
            ("Show Camera In Main View - Fallback", self._vision_show_camera_main_fallback, "show_camera_view"),
            ("Return Main View To Perspective", self._vision_return_main_view, ""),
            ("Close Onboard Camera Viewport", self._vision_close_camera_viewport, ""),
            ("Restore Previous View", self._vision_restore_camera_view, ""),
            ("Calibrate Ground Reference", self._vision_calibrate_ground_reference, ""),
            ("Validate Robot Ground Contact", self._vision_validate_robot_ground_contact, ""),
            ("Respawn And Validate Ground", self._vision_respawn_and_validate_ground, "manual_respawn"),
            ("Save RGB-D Diagnostic", self._vision_save_rgbd_diagnostic, ""),
            ("Open Vision Debug Folder", self._vision_open_debug_folder, ""),
            ("Validate Mount After Next Respawn", self._vision_validate_mount_after_next_respawn, ""),
        ]
        for index, (label, callback, guard) in enumerate(camera_buttons):
            button = self.ttk.Button(camera, text=label, command=callback)
            if guard:
                self._register_guarded_button(button, guard)
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        camera.columnconfigure(0, weight=1)
        camera.columnconfigure(1, weight=1)

        self.ttk.Label(detection, textvariable=self.vision_detection_status_var, wraplength=560).grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        validate_button = self.ttk.Button(detection, text="Validate Current Generated Height", command=self._vision_validate_current_height)
        self._register_guarded_button(validate_button, "validate_generated_height")
        validate_button.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        self.ttk.Button(detection, text="Clear Validation Result", command=self._vision_clear_validation_result).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        detection.columnconfigure(0, weight=1)
        detection.columnconfigure(1, weight=1)

        self.ttk.Label(steps, textvariable=self.vision_steps_status_var, wraplength=560).grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=4)
        play_button = self.ttk.Button(steps, text="Play Validated Vision Steps", command=self._vision_play_validated_steps)
        self._register_guarded_button(play_button, "play_vision_steps")
        play_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        steps.columnconfigure(0, weight=1)
        steps.columnconfigure(1, weight=1)

        auto_buttons = [
            ("Enable Auto Replay", lambda: self._vision_set_auto_replay(True), ""),
            ("Disable Auto Replay", lambda: self._vision_set_auto_replay(False), ""),
            ("Arm Auto Replay", self._vision_arm, "arm_vision_auto"),
            ("Disarm Auto Replay", self._vision_disarm, ""),
        ]
        for index, (label, callback, guard) in enumerate(auto_buttons):
            button = self.ttk.Button(auto, text=label, command=callback)
            if guard:
                self._register_guarded_button(button, guard)
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=2)
        self.ttk.Checkbutton(
            auto,
            text="Respawn Before Auto Replay",
            variable=self.vision_respawn_before_replay_var,
            command=self._vision_apply_respawn_option,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=3, pady=(6, 2))
        auto.columnconfigure(0, weight=1)
        auto.columnconfigure(1, weight=1)

        self.vision_status_tree = self._make_vision_status_tree(parent, row=6)
        raw = self.ttk.LabelFrame(parent, text="Raw Vision Diagnostics")
        raw.grid(row=7, column=0, sticky="nsew", pady=(0, 6))
        controls = self.ttk.Frame(raw)
        controls.grid(row=0, column=0, sticky="ew", padx=3, pady=(2, 0))
        self.ttk.Checkbutton(controls, text="Auto Refresh Details", variable=self.vision_details_auto_refresh_var).grid(row=0, column=0, sticky="w", padx=2)
        self.ttk.Button(controls, text="Refresh Details", command=self._vision_refresh_details_once).grid(row=0, column=1, sticky="ew", padx=2)
        self.ttk.Button(controls, text="Pause Details", command=self._vision_pause_details).grid(row=0, column=2, sticky="ew", padx=2)
        controls.columnconfigure(2, weight=1)
        self.vision_status_text = self._make_scrolled_text(raw, row=1, height=12, width=62)
        raw.columnconfigure(0, weight=1)
        raw.rowconfigure(1, weight=1)

    def _build_sim_state_tab(self, parent: Any) -> None:
        self.ttk.Button(parent, text="Refresh Sim State", command=lambda: self._refresh(force=False, sim_state=True)).grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 4))
        self.sim_state_text = self._make_scrolled_text(parent, row=1, height=23, width=62)

    def _build_manifest_tree(self, parent: Any, *, row: int) -> Any:
        frame = self.ttk.LabelFrame(parent, text="Manifest")
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 6))
        columns = ("height", "recorded", "steps", "saved", "marker")
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", height=10)
        widths = {"height": 80, "recorded": 80, "steps": 70, "saved": 180, "marker": 90}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=widths[col], stretch=col == "saved")
        ybar = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ybar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        return tree

    def _make_vision_status_tree(self, parent: Any, *, row: int) -> Any:
        frame = self.ttk.LabelFrame(parent, text="Structured Vision Status")
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 6))
        columns = ("field", "value")
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse", height=12)
        tree.heading("field", text="Field")
        tree.heading("value", text="Value")
        tree.column("field", width=190, stretch=False)
        tree.column("value", width=360, stretch=True)
        ybar = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ybar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        self._bind_inner_scroll_widget(tree, ybar)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
        return tree

    def _make_scrolled_text(self, parent: Any, *, row: int, column: int = 0, columnspan: int = 1, height: int, width: int) -> Any:
        frame = self.ttk.Frame(parent)
        frame.grid(row=row, column=column, columnspan=columnspan, sticky="nsew", padx=4, pady=4)
        text = self.tk.Text(frame, height=height, width=width, wrap="none")
        ybar = self.ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        xbar = self.ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set, state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        self._bind_inner_scroll_widget(text, ybar)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(row, weight=1)
        return text

    def _servo_slider(self, joint_name: str, value: str) -> None:
        if self.updating:
            return
        angle = float(value)
        self._schedule_slider(joint_name, f"servo {joint_name} {angle:.1f}", angle, 0.5)

    def _wheel_slider(self, short_name: str, value: str) -> None:
        if self.updating:
            return
        speed = float(value)
        self._schedule_slider(short_name, f"wheel {short_name} {speed:.3f}", speed, 0.02)

    def _run_command(self) -> None:
        text = self.command_var.get().strip()
        if not text:
            return
        self._post(text)
        self.command_var.set("")

    def _selected_indices(self) -> list[int]:
        values: list[int] = []
        for item in self.steps_tree.selection():
            try:
                values.append(int(str(item).split("_")[-1]))
            except ValueError:
                pass
        return sorted(values)

    def _selected_index(self) -> int | None:
        indices = self._selected_indices()
        return indices[0] if indices else self.selected_step_index

    def _selected_step_command(self, template: str) -> None:
        index = self._selected_index()
        if index is None:
            self.controller._warn("[WARN] Select an accepted step first. No command posted.")
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        self._post(template.format(index=index))

    def _selected_step_playback_command(self, template: str) -> None:
        self._apply_playback_options()
        tree_selection = list(self.steps_tree.selection())
        tree_indices = self._selected_indices()
        index = self._selected_index()
        ok, reason = self.controller.can_playback()
        command = template.format(index=index) if index is not None else ""
        self.controller._info(
            "[PLAYBACK DEBUG] ui play-selected "
            f"tree_selection={tree_selection} tree_indices={tree_indices} "
            f"controller_selected={self.controller.selected_step_index} selected_index={index} "
            f"generated_command={command or '<none>'} can_playback={ok} reason={reason or 'ok'}"
        )
        if index is None:
            self.controller._warn("[WARN] Select an accepted step first. No playback command posted.")
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        if not ok:
            self.controller._warn("[WARN] " + reason)
            return
        self._post(command)

    def _post_playback_command(self, command: str) -> None:
        self._apply_playback_options()
        self._post(command)

    def _replace_selected(self) -> None:
        if self.controller.visible_steps_are_read_only():
            self.messagebox.showwarning("Read Only", "Vision steps are read-only. Finish Vision Task before replacing Height Task steps.")
            return
        index = self._selected_index()
        if index is None:
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        self._post(f"replace_step {index}")

    def _delete_selected(self) -> None:
        if self.controller.visible_steps_are_read_only():
            self.messagebox.showwarning("Read Only", "Vision steps are read-only. Finish Vision Task before deleting Height Task steps.")
            return
        index = self._selected_index()
        if index is None:
            self.messagebox.showwarning("No Selection", "Select an accepted step first.")
            return
        if self.messagebox.askyesno("Delete Step", f"Delete accepted step {index:03d} from memory?"):
            self._post(f"delete_step {index}")

    def _clear_all_steps(self) -> None:
        if self.controller.visible_steps_are_read_only():
            self.messagebox.showwarning("Read Only", "Vision steps are read-only. Finish Vision Task before clearing Height Task steps.")
            return
        if not self.clear_confirm_var.get():
            self.messagebox.showwarning("Confirm Clear All", "Check Confirm Clear All before clearing accepted steps.")
            self._post("clear_steps")
            return
        if self.messagebox.askyesno("Clear All Steps", "Clear all accepted steps in memory? Saved files are not deleted."):
            self._post("clear_steps_confirmed")
            self.clear_confirm_var.set(False)

    def _on_step_selected(self, _event: Any) -> None:
        indices = self._selected_indices()
        if self.controller.visible_steps_are_read_only():
            self.controller.combine_selected_indices.clear()
            self.controller.combine_preview_text = ""
        elif self.controller.combine_mode_enabled:
            if len(indices) > 1:
                self.controller.set_combine_selection(indices)
        else:
            self.controller.set_combine_selection(indices)
        if not indices:
            return
        self.selected_step_index = indices[0]
        self.controller.selected_step_index = indices[0]
        try:
            step = self.controller.get_visible_step(indices[0])
            self.controller.set_step_summary_detail(step, title=f"Step {indices[0]:03d} summary")
        except Exception:
            pass
        self._refresh(force=False)

    def _run_isaac_preflight(self) -> None:
        self.controller.status = "Running Isaac preflight..."
        self._refresh(force=True)

        def work() -> None:
            try:
                result = self.controller.run_isaac_preflight()
                error = ""
            except Exception as exc:
                result = {}
                error = str(exc)

            def done() -> None:
                if error:
                    self.controller._warn(f"[WARN] Isaac preflight failed: {error}")
                else:
                    self.controller.latest_sim_status = {
                        **self.controller.latest_sim_status,
                        "preflight": result,
                        "preflight_ok": bool(result.get("preflight_ok", False)),
                        "preflight_error": str(result.get("preflight_error", "") or ""),
                        "candidate_reports": result.get("candidate_reports", []),
                    }
                self._refresh(force=True)

            self.root.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _restart_isaac_worker(self) -> None:
        try:
            self.controller.restart_sim_worker()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not restart Isaac worker: {exc}")
        self._refresh(force=True)

    def _restart_without_onboard_camera(self) -> None:
        try:
            self.controller.restart_sim_worker_without_camera()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not restart without onboard camera: {exc}")
        self._refresh(force=True)

    def _open_worker_log_folder(self) -> None:
        try:
            path = self.controller.open_worker_log_folder()
            if path:
                self.controller._info(f"[INFO] Worker log folder: {path}")
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not open worker log folder: {exc}")
        self._refresh(force=True)

    def _copy_worker_command(self) -> None:
        command = self.controller.copy_worker_display_command()
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(command)
            self.controller._info("[INFO] Worker command copied to clipboard.")
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not copy worker command: {exc}")
        self._refresh(force=True)

    def _stop_worker(self) -> None:
        try:
            self.controller.stop_sim_worker()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not stop Isaac worker: {exc}")
        self._refresh(force=True)

    def _ensure_sim(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not start Isaac Sim: {exc}")
        self._refresh(force=False)

    def _generate_current_obstacle(self) -> None:
        try:
            self.controller.generate_or_update_height_obstacle()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not generate obstacle: {exc}")
        self._refresh(force=False)

    def _load_current_height_steps(self) -> None:
        self.controller.load_steps_for_current_height()
        self._refresh(force=False)

    def _save_current_height_steps(self) -> None:
        allow_empty = False
        if self.controller.manager.count <= 0:
            allow_empty = self.messagebox.askyesno(
                "Save Empty Steps",
                "Current sequence has no accepted steps. Overwrite this height's saved file as empty?",
            )
            if not allow_empty:
                return
        self.controller.save_steps_for_current_height(allow_empty=allow_empty)
        self._refresh(force=False)

    def _new_empty_current_height(self) -> None:
        if self.controller.manager.dirty and not self.messagebox.askyesno("Discard Unsaved Changes", "Discard unsaved current-height steps?"):
            return
        self.controller.new_empty_sequence_for_current_height(discard_dirty=True)
        self._refresh(force=False)

    def _refresh_manifest(self) -> None:
        self.controller.refresh_manifest()
        self._refresh(force=False)

    def _vision_set_enabled(self, enabled: bool) -> None:
        self.controller.set_vision_enabled(enabled)
        self._refresh(force=True)

    def _vision_source_mode_from_var(self) -> str:
        return normalize_source_mode(self.vision_source_var.get())

    def _vision_source_changed(self) -> None:
        self.controller.vision_source_mode = self._vision_source_mode_from_var()
        self.controller.transport.set_vision_source_mode("external" if self.controller.vision_source_mode == VISION_SOURCE_EXTERNAL else "generated")
        if self.controller.vision_task_active:
            self.controller.vision_task.source_mode = self.controller.vision_source_mode
            self.controller._clear_vision_steps("Waiting for a validated Vision detection.")
        self._refresh(force=True)

    def _vision_start_task(self) -> None:
        self.controller.start_vision_task(self._vision_source_mode_from_var())
        self._refresh(force=True)

    def _vision_generate_test_obstacle(self) -> None:
        try:
            height = int(self.vision_test_height_var.get())
        except (TypeError, ValueError):
            self.controller._warn("[WARN] Select a valid Vision test obstacle height.")
            self._refresh(force=True)
            return
        self.controller.generate_vision_test_obstacle(height)
        self._refresh(force=True)

    def _vision_finish_task(self) -> None:
        self.controller.finish_vision_task()
        self._refresh(force=True)

    def _vision_start_and_generate_selected(self) -> None:
        if not self.controller.vision_task_active:
            self.controller.start_vision_task(self._vision_source_mode_from_var())
        self._vision_generate_test_obstacle()

    def _vision_detect_once(self) -> None:
        self.controller.request_vision_detection_once()
        self._refresh(force=True)

    def _vision_reset_filter(self) -> None:
        self.controller.reset_vision_filter()
        self._refresh(force=True)

    def _vision_set_auto_replay(self, enabled: bool) -> None:
        self.controller.set_vision_auto_replay_enabled(enabled)
        self._refresh(force=True)

    def _vision_arm(self) -> None:
        self.controller.arm_vision_auto_replay()
        self._refresh(force=True)

    def _vision_disarm(self) -> None:
        self.controller.disarm_vision_auto_replay()
        self.controller._info("[INFO] Vision auto replay disarmed.")
        self._refresh(force=True)

    def _vision_apply_respawn_option(self) -> None:
        self.controller.vision_respawn_before_replay = bool(self.vision_respawn_before_replay_var.get())
        self.controller.vision_last_auto_action = (
            "respawn before auto replay enabled"
            if self.controller.vision_respawn_before_replay
            else "respawn before auto replay disabled"
        )
        self._refresh(force=True)

    def _vision_save_debug_frame(self) -> None:
        self.controller.request_vision_debug_frame()
        self._refresh(force=True)

    def _vision_validate_camera(self) -> None:
        self.controller.validate_isaac_camera()
        self._refresh(force=True)

    def _vision_validate_camera_geometry(self) -> None:
        self.controller.validate_camera_geometry()
        self._refresh(force=True)

    def _vision_apply_recommended_camera_pose(self) -> None:
        self.controller.apply_recommended_oblique_camera_pose()
        self._refresh(force=True)

    def _vision_restart_worker_with_camera_pose(self) -> None:
        try:
            self.controller.restart_sim_worker_with_camera_pose()
        except Exception as exc:
            self.controller._warn(f"[WARN] Could not restart worker with camera pose: {exc}")
        self._refresh(force=True)

    def _vision_open_camera_viewport(self) -> None:
        ok = self.controller.open_onboard_camera_viewport()
        if not ok:
            self.controller._warn(
                "[WARN] Camera viewport auto-switch unavailable. Manual path: "
                "Click the Camera button in the Isaac Viewport and select Perspective, or open Window > Viewports > Viewport 2 and select Cameras > onboard_rgbd_camera."
            )
        self._refresh(force=True)

    def _vision_show_camera_main_fallback(self) -> None:
        self.controller.show_camera_in_main_view_fallback()
        self._refresh(force=True)

    def _vision_return_main_view(self) -> None:
        self.controller.return_main_view_to_perspective()
        self._refresh(force=True)

    def _vision_close_camera_viewport(self) -> None:
        self.controller.close_onboard_camera_viewport()
        self._refresh(force=True)

    def _vision_restore_camera_view(self) -> None:
        self.controller.restore_camera_view()
        self._refresh(force=True)

    def _vision_validate_robot_ground_contact(self) -> None:
        self.controller.validate_robot_ground_contact()
        self._refresh(force=True)

    def _vision_calibrate_ground_reference(self) -> None:
        self.controller.calibrate_ground_reference()
        self._refresh(force=True)

    def _vision_respawn_and_validate_ground(self) -> None:
        self.controller.respawn_and_validate_ground()
        self._refresh(force=True)

    def _vision_validate_current_height(self) -> None:
        self.controller.validate_current_generated_height()
        self._refresh(force=True)

    def _vision_save_rgbd_diagnostic(self) -> None:
        self.controller.request_rgbd_diagnostic()
        self._refresh(force=True)

    def _vision_clear_validation_result(self) -> None:
        self.controller.clear_vision_validation_result()
        self._refresh(force=True)

    def _vision_validate_mount_after_next_respawn(self) -> None:
        self.controller.validate_mount_after_next_respawn()
        self._refresh(force=True)

    def _vision_open_debug_folder(self) -> None:
        self.controller.open_vision_debug_folder()
        self._refresh(force=True)

    def _vision_refresh_details_once(self) -> None:
        self.vision_details_paused = False
        self.vision_details_auto_refresh_var.set(True)
        self._refresh(force=True)

    def _vision_pause_details(self) -> None:
        self.vision_details_paused = True
        self.vision_details_auto_refresh_var.set(False)
        self.controller._info("[INFO] Raw Vision Diagnostics auto refresh paused.")
        self._refresh(force=False)

    def _vision_play_validated_steps(self) -> None:
        if not self.controller.vision_steps_ready or self.controller.vision_steps_height_cm is None:
            self.controller._warn("[WARN] Vision steps are not ready.")
            self._refresh(force=True)
            return
        revision = int(self.controller.vision_steps_detection_revision or self.controller.vision_last_detection_revision or 0)
        self.controller.replay_validated_vision_steps(int(self.controller.vision_steps_height_cm), revision)
        self._refresh(force=True)

    def _combine_add_selected(self) -> None:
        self.controller.add_to_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_remove_selected(self) -> None:
        self.controller.remove_from_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_toggle_selected(self) -> None:
        self.controller.toggle_combine_selection(self._selected_indices())
        self._refresh(force=False)

    def _combine_select_contiguous_range(self) -> None:
        indices = self._selected_indices()
        if len(indices) >= 2:
            self.controller.add_to_combine_selection([indices[0], indices[-1]])
        self.controller.select_contiguous_combine_range()
        self._refresh(force=False)

    def _set_combine_allow_conflicts(self) -> None:
        self.controller.allow_combine_conflicts = bool(self.combine_allow_conflicts_var.get())
        if self.controller.combine_mode_enabled:
            selected = sorted(self.controller.combine_selected_indices)
            self.controller.combine_preview_text = (
                f"Conflict option changed for selected steps {selected}. Click Preview Combined Step to recompute."
                if len(selected) >= 2
                else "Select at least two steps."
            )
        self._refresh(force=False)

    def _apply_speed_scale_controls(self) -> None:
        try:
            self.controller.set_motion_scale(
                global_motion_speed_scale=float(self.global_scale_var.get()),
                wheel_speed_scale=float(self.wheel_scale_var.get()),
                servo_command_scale=float(self.servo_scale_var.get()),
                playback_speed_scale=float(self.playback_scale_var.get()),
                apply_to_manual_control=bool(self.apply_manual_scale_var.get()),
                apply_to_playback=bool(self.apply_playback_scale_var.get()),
                preserve_wheel_distance=bool(self.preserve_distance_var.get()),
            )
            self._sync_speed_scale_vars()
        except Exception as exc:
            self.controller._warn(f"[WARN] Invalid speed scale settings: {exc}")
        self._refresh(force=True)

    def _reset_speed_scale_controls(self) -> None:
        self.controller.reset_motion_scale()
        self._sync_speed_scale_vars()
        self._refresh(force=True)

    def _sync_speed_scale_vars(self) -> None:
        scale = self.controller.motion_scale
        self.global_scale_var.set(f"{scale.global_motion_speed_scale:.2f}")
        self.wheel_scale_var.set(f"{scale.wheel_speed_scale:.2f}")
        self.servo_scale_var.set(f"{scale.servo_command_scale:.2f}")
        self.playback_scale_var.set(f"{scale.playback_speed_scale:.2f}")
        self.apply_manual_scale_var.set(scale.apply_to_manual_control)
        self.apply_playback_scale_var.set(scale.apply_to_playback)
        self.preserve_distance_var.set(scale.preserve_wheel_distance)

    def _apply_playback_options(self) -> None:
        idle_text = self.play_max_idle_gap_var.get().strip().lower()
        idle_gap = None if idle_text in {"", "none", "null"} else float(idle_text)
        self.controller.set_playback_options(
            profile=self.play_profile_var.get(),
            speed=float(self.play_speed_var.get()),
            trailing_pad=float(self.play_trailing_pad_var.get()),
            max_idle_gap=idle_gap,
            preserve_wheel_distance=bool(self.preserve_distance_var.get()),
        )
        self.controller.restore_step_start_state_before_selected_playback = bool(self.restore_step_start_var.get())
        self.controller.restore_full_sim_pose_if_available = bool(self.restore_full_sim_pose_var.get())
        self.controller.fallback_to_command_state_before = bool(self.fallback_command_state_var.get())

    def _scroll_canvas(self, canvas: Any, event: Any) -> str:
        delta = self._wheel_units(event)
        if delta:
            canvas.yview_scroll(delta, "units")
        return "break"

    def _on_tree_mousewheel(self, event: Any) -> str:
        widget = getattr(event, "widget", self.steps_tree)
        try:
            setattr(widget, "_last_user_scroll_at", time.monotonic())
        except Exception:
            pass
        delta = self._wheel_units(event)
        if delta:
            widget.yview_scroll(delta, "units")
        return "break"

    def _on_scale_mousewheel(self, event: Any) -> str:
        return "break"

    def _on_text_mousewheel(self, event: Any) -> str:
        widget = getattr(event, "widget", None)
        if widget is not None:
            try:
                setattr(widget, "_last_user_scroll_at", time.monotonic())
            except Exception:
                pass
            delta = self._wheel_units(event)
            if delta:
                try:
                    widget.yview_scroll(delta, "units")
                except Exception:
                    pass
        return "break"

    def _wheel_units(self, event: Any) -> int:
        number = int(getattr(event, "num", 0) or 0)
        if number == 4:
            return -1
        if number == 5:
            return 1
        delta = int(getattr(event, "delta", 0) or 0)
        if delta == 0:
            return 0
        return int(-1 * (delta / 120))

    def _bind_inner_scroll_widget(self, widget: Any, ybar: Any | None = None) -> None:
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                widget.bind(sequence, self._on_text_mousewheel)
            except Exception:
                pass
        if ybar is not None:
            try:
                ybar.bind("<ButtonPress-1>", lambda _event, w=widget: setattr(w, "_scrollbar_dragging", True))
                ybar.bind(
                    "<ButtonRelease-1>",
                    lambda _event, w=widget: (setattr(w, "_scrollbar_dragging", False), setattr(w, "_last_user_scroll_at", time.monotonic())),
                )
            except Exception:
                pass

    def _poll(self) -> None:
        self.controller.update()
        self._refresh(force=False)
        self.root.after(self.ui_refresh_ms, self._poll)

    def _right_tab_is_visible(self, tab_name: str) -> bool:
        try:
            current = self.right_notebook.select()
            return str(self.right_notebook.tab(current, "text")) == tab_name
        except Exception:
            return False

    def _refresh(self, *, force: bool = True, sim_state: bool = False) -> None:
        if self.refreshing:
            return
        refresh_started = time.perf_counter()
        now = time.monotonic()
        do_medium = force or (now - self._last_medium_refresh) * 1000.0 >= self.sim_status_refresh_ms
        do_full = (now - self._last_full_refresh) * 1000.0 >= self.full_refresh_ms
        sim_state_visible = self._right_tab_is_visible("Sim State")
        auto_sim_state = (
            sim_state_visible
            and do_full
            and not self.disable_auto_sim_state_json
            and not self.sim_state_json_on_demand
        )
        snapshot = self.controller.snapshot()
        elapsed = now - self.controller._last_ui_refresh_wall
        if elapsed > 0.0:
            self.controller.ui_refresh_hz = 1.0 / elapsed
        self.controller._last_ui_refresh_wall = now
        sequence_revision = int(snapshot["revisions"]["sequence"])
        manifest_revision = int(snapshot["revisions"]["manifest"])
        sequence_changed = self._last_sequence_revision != sequence_revision
        manifest_changed = self._last_manifest_revision != manifest_revision
        detail_changed = self._last_detail_text != snapshot["detail_text"]
        combine_changed = self._last_combine_preview != snapshot["combine"]["preview"]
        self.refreshing = True
        self.updating = True
        try:
            self.summary_var.set(self.controller.status_line())
            self.sim_label_var.set(
                f"Runtime Ready={_yes_no(snapshot['sim'].get('runtime_ready', snapshot['sim']['ready']))} "
                f"Motion Ready={_yes_no(snapshot['sim'].get('motion_ready', False))} no_sim={snapshot['sim']['no_sim']} "
                f"phase={snapshot['sim'].get('phase', '')} RTF={float(snapshot['sim']['real_time_factor']):.2f}"
            )
            self.mode_label_var.set(f"Mode: {snapshot['recording']['mode']}")
            self.sequence_label_var.set(
                f"View: {snapshot['sequence']['source']} steps={snapshot['sequence']['count']} "
                f"dirty={snapshot['height_sequence']['dirty']}"
            )
            self.accepted_steps_source_var.set(
                f"Accepted Steps Source: {snapshot['sequence']['source']}"
                + (" (read-only)" if snapshot["sequence"].get("read_only") else "")
            )
            self.task_label_var.set(
                f"Height Task active={snapshot['task']['active']} index={snapshot['task']['index']}/{snapshot['task']['total']}"
            )
            playback_guard_ok, playback_guard_reason = self.controller.can_playback()
            self.playback_label_var.set(
                f"Playback: active={snapshot['playback']['active']} scheduled={snapshot['playback'].get('scheduled', False)} "
                f"starts_in={float(snapshot['playback'].get('starts_in_s', 0.0)):.2f}s paused={snapshot['playback']['paused']} "
                f"idx={snapshot['playback']['index']}/{snapshot['playback']['count']} sent={snapshot['playback'].get('events_sent', 0)} "
                f"profile={snapshot['playback']['profile']} speed={float(snapshot['playback']['speed']):.2f} "
                f"last={snapshot['playback'].get('last_event_command', '') or '-'} "
                f"stop={snapshot['playback'].get('stop_reason', '') or '-'} "
                f"info={snapshot['playback'].get('last_error') or snapshot['playback'].get('last_info') or ('' if playback_guard_ok else playback_guard_reason)}"
            )
            self.record_label_var.set(
                f"Record: active={snapshot['recording']['active']} events={snapshot['recording']['events']}"
            )
            self.busy_label_var.set(f"Busy: {self.controller.busy_name}" if self.controller.operation_busy else "")
            self.pending_label_var.set(
                f"Pending: recorded={snapshot['recording']['pending']} replacement={snapshot['recording']['pending_replacement']}"
            )
            self.replacement_label_var.set(f"Replacement target: {self.controller.replace_target_index or 'none'}")
            self.combine_selection_var.set(
                f"Selected combine indices: {snapshot['combine']['selected_indices']} | "
                f"count={snapshot['combine']['selected_count']} contiguous={snapshot['combine']['contiguous']}"
            )
            self.height_var.set(str(snapshot["height"]["current_cm"]))

            staged_active = bool(snapshot["servo_wheel"]["staging_active"])
            staged_state = snapshot["servo_wheel"]["staged_state"]
            servo_display = staged_state["servos"] if staged_active else snapshot["servos"]
            wheel_display = {
                short: staged_state["wheels"].get(full, 0.0)
                for short, full in WHEEL_SHORT_NAMES.items()
            } if staged_active else snapshot["wheels"]
            for name, value in servo_display.items():
                if name not in self.slider_dragging and name in self.servo_vars:
                    self.servo_vars[name].set(float(value))
                if name in self.servo_output_label_vars:
                    prefix = "staged" if staged_active else "out"
                    self.servo_output_label_vars[name].set(f"{prefix} {float(value):.1f}")
            wheel_parts = []
            for short_name, value in wheel_display.items():
                if short_name not in self.slider_dragging and short_name in self.wheel_vars:
                    self.wheel_vars[short_name].set(float(value))
                wheel_parts.append(f"{short_name}={float(value):.2f}")
            self.wheel_status_var.set(("Staged wheel: " if staged_active else "Wheel command: ") + " ".join(wheel_parts))
            self.speed_scale_label_var.set(
                "manual wheel scale="
                f"{float(snapshot['speed_scale']['manual_wheel_scale']):.2f}, "
                f"manual servo scale={float(snapshot['speed_scale']['manual_servo_scale']):.2f}, "
                f"playback scale={float(snapshot['speed_scale']['playback_scale']):.2f}, "
                f"max wheel={float(snapshot['sim']['current_max_wheel_speed_rad_s']):.3f} rad/s, "
                f"default={float(snapshot['sim']['default_wheel_speed_rad_s']):.3f} rad/s\n"
                f"last: {snapshot['speed_scale']['last_raw_motion_command'] or '-'}"
                f" -> {snapshot['speed_scale']['last_scaled_motion_command'] or '-'}"
            )
            self.vision_respawn_before_replay_var.set(bool(snapshot["vision"].get("respawn_before_replay", True)))
            source_label = "External / Unknown Obstacle" if snapshot["vision"].get("source_mode") == VISION_SOURCE_EXTERNAL else "Generated Test Obstacle"
            self.vision_source_var.set(source_label)
            generated_height = snapshot["vision"].get("generated_height_cm")
            if generated_height is not None:
                self.vision_test_height_var.set(str(int(generated_height)))
            task = snapshot["vision"].get("task", {}) if isinstance(snapshot["vision"].get("task"), dict) else {}
            validation = snapshot["vision"].get("height_validation", {}) if isinstance(snapshot["vision"].get("height_validation"), dict) else {}
            validation_state = "NOT CHECKED"
            if bool(validation.get("checked", False)):
                validation_state = "PASS" if bool(validation.get("passed", False)) else "FAIL"
            validation_ready = bool(snapshot["vision"].get("validation_ready", False))
            validation_block = str(snapshot["vision"].get("validation_block_reason", "") or "")
            camera_block = str(snapshot["vision"].get("camera_view_block_reason", "") or "")
            ground_block = str(snapshot["sim"].get("motion_block_reason", "") or "")
            respawn_block = str(snapshot["sim"].get("respawn_block_reason", "") or snapshot["vision"].get("respawn_block_reason", "") or "")
            robot_ground = snapshot["vision"].get("robot_ground", {}) if isinstance(snapshot["vision"].get("robot_ground"), dict) else {}
            playback_guard_text = self.controller.playback_readiness(respawn_first=bool(snapshot["vision"].get("respawn_before_replay", True)))[1]
            self.vision_high_level_status_var.set(
                " | ".join(
                    [
                        f"Runtime Ready={_yes_no(snapshot['sim'].get('runtime_ready', False))}",
                        f"Camera Ready={_yes_no(snapshot['vision'].get('camera_ready', False))}",
                        f"Ground State={snapshot['vision'].get('ground_state', '-') or '-'}",
                        f"Physical Ground Safe={_yes_no(robot_ground.get('physical_ground_safe', False))}",
                        f"Visual Ground Safe={_yes_no(robot_ground.get('visual_ground_safe', False))}",
                        f"Motion Ready={_yes_no(snapshot['vision'].get('motion_ready', False))}",
                        f"Respawn Ready={_yes_no(snapshot['vision'].get('respawn_ready', False))}",
                        f"Vision Task Active={_yes_no(task.get('active', False))}",
                        f"Generated={snapshot['vision'].get('generated_height_cm', '-') or '-'}cm",
                        f"Scene={task.get('scene_height_cm', '-') or '-'}cm",
                        f"Detected={snapshot['vision'].get('detected_height_cm', '-') or '-'}cm",
                        f"Validation Ready={_yes_no(validation_ready)}",
                        f"Validation State={validation_state}",
                        f"Steps Ready={_yes_no(snapshot['vision'].get('steps_ready', False))}",
                    ]
                )
            )
            self.vision_block_reasons_var.set(
                " | ".join(
                    part
                    for part in [
                        f"Camera View Block: {camera_block}" if camera_block else "Camera View Block: -",
                        f"Ground Block: {ground_block}" if ground_block else "Ground Block: -",
                        f"Respawn Block: {respawn_block}" if respawn_block else "Respawn Block: -",
                        f"Validation Block: {validation_block}" if validation_block else "Validation Block: -",
                        f"Playback Block: {playback_guard_text}" if playback_guard_text else "Playback Block: -",
                    ]
                    if part
                )
            )
            self.vision_task_status_var.set(
                f"Vision Task Active: {bool(task.get('active', False))} | "
                f"Source: {source_label} | Phase: {task.get('phase', 'INACTIVE')}\n"
                f"Generated Height: {task.get('generated_height_cm', '-') or '-'}cm | "
                f"Scene Height: {task.get('scene_height_cm', '-') or '-'}cm | "
                f"Generation Revision: {snapshot['vision'].get('generation_revision', 0)} | "
                f"Detection Baseline: {task.get('generation_detection_baseline', 0)}"
            )
            self.vision_detection_status_var.set(
                f"Generated Height: {snapshot['vision'].get('generated_height_cm', '-') or '-'}cm | "
                f"Measured Height: {snapshot['vision'].get('raw_height_cm', '-') or '-'} | "
                f"Candidate Height: {snapshot['vision'].get('candidate_height_cm', '-') or '-'}cm | "
                f"Stable Detected Height: {snapshot['vision'].get('detected_height_cm', '-') or '-'}cm\n"
                f"Confidence: {float(snapshot['vision'].get('confidence', 0.0) or 0.0):.3f} | "
                f"Stable Frames: {int(snapshot['vision'].get('stable_count', 0) or 0)}/{int(snapshot['vision'].get('stable_required', 0) or 0)} | "
                f"Detection Revision: {int(snapshot['vision'].get('detection_revision', 0) or 0)} | "
                f"Validation State: {validation_state}"
            )
            self.vision_steps_status_var.set(
                f"Steps Source: {snapshot['sequence']['source']} | "
                f"Validated Height: {snapshot['vision'].get('steps_height_cm', '-') or '-'}cm | "
                f"Step Count: {snapshot['vision'].get('steps_count', 0)} | "
                f"Steps Ready: {bool(snapshot['vision'].get('steps_ready', False))}\n"
                f"Steps Path: {snapshot['vision'].get('steps_path', '') or '-'}\n"
                f"{snapshot['vision'].get('steps_error', '') or ''}"
            )

            if force or do_full or sequence_changed:
                combine_indices = [] if snapshot["sequence"].get("read_only") else snapshot["combine"]["selected_indices"]
                self._refresh_steps_tree(snapshot["sequence"]["rows"], combine_indices)
                self._last_sequence_revision = sequence_revision
            if force or detail_changed:
                self._set_text(self.detail_text, snapshot["detail_text"])
                self._last_detail_text = snapshot["detail_text"]
            if do_medium:
                self._set_text(self.connection_text, json.dumps(snapshot["sim"], indent=2, ensure_ascii=False))
                self._set_text(self.servo_wheel_preview_text, snapshot["servo_wheel"]["preview"])
                if hasattr(self, "vision_status_text"):
                    self._refresh_vision_status_tree(snapshot)
                    if bool(self.vision_details_auto_refresh_var.get()) and not self.vision_details_paused:
                        self._set_text(self.vision_status_text, _format_vision_status_summary(snapshot["vision"]))
                self._last_medium_refresh = now
            if sim_state or auto_sim_state:
                self._set_text(self.sim_state_text, json.dumps(snapshot, indent=2, ensure_ascii=False, default=str))
            if force or combine_changed:
                self._set_text(self.combine_preview_text, snapshot["combine"]["preview"])
                self._last_combine_preview = snapshot["combine"]["preview"]
            if do_full or manifest_changed:
                self._refresh_manifest_trees(snapshot["height"]["manifest_rows"], snapshot["task"])
                self._last_manifest_revision = manifest_revision
            if hasattr(self, "height_task_panel") and (force or do_full or manifest_changed or sequence_changed):
                self.height_task_panel.refresh(snapshot)
            if do_full:
                self._last_full_refresh = now
            self._refresh_button_states()
        finally:
            self.updating = False
            self.refreshing = False
            elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
            if elapsed_ms > 500.0:
                message = f"[WARN] [PERF] UI refresh took {elapsed_ms:.1f}ms"
                self.controller.status = message
                self.controller.status_log.append(message)
                print(message)

    def _refresh_vision_status_tree(self, snapshot: dict[str, Any]) -> None:
        tree = getattr(self, "vision_status_tree", None)
        if tree is None:
            return
        try:
            yview = tree.yview()
        except Exception:
            yview = (0.0, 1.0)
        try:
            selection = tuple(tree.selection())
        except Exception:
            selection = ()
        rows = self._vision_status_tree_rows(snapshot)
        existing = set(tree.get_children())
        wanted = {iid for iid, _field, _value in rows}
        for iid in existing - wanted:
            tree.delete(iid)
        for iid, field, value in rows:
            values = (field, value)
            if iid in existing:
                if tuple(tree.item(iid, "values")) != values:
                    tree.item(iid, values=values)
            else:
                tree.insert("", "end", iid=iid, values=values)
        try:
            if selection:
                tree.selection_set([item for item in selection if item in tree.get_children()])
            tree.yview_moveto(float(yview[0] if yview else 0.0))
        except Exception:
            pass

    def _vision_status_tree_rows(self, snapshot: dict[str, Any]) -> list[tuple[str, str, str]]:
        vision = snapshot.get("vision", {}) if isinstance(snapshot.get("vision"), dict) else {}
        sim = snapshot.get("sim", {}) if isinstance(snapshot.get("sim"), dict) else {}
        ground = vision.get("robot_ground", {}) if isinstance(vision.get("robot_ground"), dict) else {}
        camera_view = vision.get("camera_view", {}) if isinstance(vision.get("camera_view"), dict) else {}
        health = vision.get("camera_health", {}) if isinstance(vision.get("camera_health"), dict) else {}
        validation = vision.get("height_validation", {}) if isinstance(vision.get("height_validation"), dict) else {}
        evidence = vision.get("height_measurement_evidence", {}) if isinstance(vision.get("height_measurement_evidence"), dict) else {}
        fields = [
            ("runtime_ready", "Runtime Ready", _yes_no(sim.get("runtime_ready", False))),
            ("camera_ready", "Camera Ready", _yes_no(vision.get("camera_ready", False))),
            ("camera_backend", "Camera Backend", ".".join(part for part in [str(health.get("backend_module", "") or ""), str(health.get("backend_class", "") or "")] if part) or "-"),
            ("camera_health", "Camera Health", "PASS" if bool(health.get("ok", False)) else ("FAIL" if bool(health.get("checked", False)) else "NOT CHECKED")),
            ("frame_age", "Frame Age", _seconds(health.get("frame_age_s"))),
            ("ground_contact", "Ground Contact", _ground_contact_label(ground)),
            ("ground_state", "Ground State", str(vision.get("ground_state", ground.get("ground_state", "-")) or "-")),
            ("physical_ground_safe", "Physical Ground Safe", _yes_no(ground.get("physical_ground_safe", False))),
            ("visual_ground_safe", "Visual Ground Safe", _yes_no(ground.get("visual_ground_safe", False))),
            ("bounds_valid", "Ground Bounds Valid", _yes_no(all(bool(w.get("bounds_valid", False)) for w in ground.get("wheels", []) if isinstance(w, dict))) if ground.get("wheels") else "-"),
            ("root_z", "Root Z", _meters(ground.get("root_z_m"))),
            ("collision_clearance", "Collision Clearance", _meters(ground.get("minimum_collision_clearance_m"))),
            ("motion_ready", "Motion Ready", _yes_no(vision.get("motion_ready", False))),
            ("respawn_ready", "Respawn Ready", _yes_no(vision.get("respawn_ready", sim.get("respawn_ready", False)))),
            ("ground_reference_stable", "Ground Reference Stable", _yes_no(vision.get("grounded_reference_stable", sim.get("grounded_reference_stable", False)))),
            ("ground_reference_reject", "Ground Reference Reject Reason", str(ground.get("ground_reference_block_reason", sim.get("ground_reference_block_reason", "")) or "-")),
            ("camera_viewport", "Camera Viewport", f"ready={_yes_no(vision.get('camera_view_ready', False))} active={_yes_no(camera_view.get('active', False))} pending={_yes_no(camera_view.get('pending', False))}"),
            ("camera_view_error", "Camera View Error", str(camera_view.get("error", "") or "-")),
            ("camera_view_window", "Camera View Window Class", str(camera_view.get("window_class", "") or "-")),
            ("camera_view_api", "Camera Viewport Class", str(camera_view.get("viewport_class", "") or "-")),
            ("camera_bound_path", "Bound Camera Path", str(camera_view.get("bound_camera_path", camera_view.get("secondary_camera_path", "")) or "-")),
            ("camera_path_verified", "Camera Path Verified", _yes_no(camera_view.get("camera_path_verified", False))),
            ("camera_frame_ready", "Camera View Frame Ready", _yes_no(camera_view.get("frame_ready", False))),
            ("measurement_model", "Measurement Model", "Top-plane world-Z from RGB-D point cloud"),
            ("measurement_formula", "Formula", "height = top_z - ground_z"),
            ("uses_camera_depth", "Uses Camera Depth", _yes_no(evidence.get("height_source", "isaac_rgbd_depth_geometry") == "isaac_rgbd_depth_geometry")),
            ("uses_intrinsics", "Uses Camera Intrinsics", _yes_no(evidence.get("intrinsics_used", False))),
            ("uses_camera_pose", "Uses Camera World Pose", _yes_no(evidence.get("camera_world_pose_used", False))),
            ("uses_expected_height", "Uses Expected Height In Detector", _yes_no(evidence.get("expected_height_used_by_detector", False))),
            ("uses_generated_height", "Uses Generated Height In Detector", _yes_no(evidence.get("generated_height_used_by_detector", False))),
            ("uses_scene_height", "Uses Scene Obstacle Height In Detector", _yes_no(evidence.get("scene_obstacle_height_used_by_detector", False))),
            ("uses_roi_x", "Uses Generated Obstacle X As ROI", _yes_no(evidence.get("obstacle_x_prior_used", False))),
            ("top_z", "Top Z", _meters(evidence.get("top_z_m"))),
            ("ground_z_used", "Ground Z Used", _meters(evidence.get("ground_z_m"))),
            ("depth_evidence", "Depth Evidence", f"rev={evidence.get('depth_frame_revision', vision.get('frame_revision', 0))} finite={_percent(evidence.get('depth_finite_ratio'))} hash={evidence.get('depth_fingerprint', '') or '-'}"),
            ("detector_audit", "Detector Input Audit", "PASS" if bool(evidence.get("detector_input_audit_passed", False)) else "FAIL"),
            ("measured_height", "Measured Height", _cm(vision.get("raw_height_cm"))),
            ("detected_height", "Detected Height", _bucket_cm(vision.get("detected_height_cm"))),
            ("confidence", "Confidence", _percent(vision.get("confidence"))),
            ("validation_ready", "Validation Ready", _yes_no(vision.get("validation_ready", False))),
            ("validation_state", "Validation State", "PASS" if bool(validation.get("checked", False)) and bool(validation.get("passed", False)) else ("FAIL" if bool(validation.get("checked", False)) else "NOT CHECKED")),
        ]
        return [(iid, field, str(value)) for iid, field, value in fields]

    def _refresh_steps_tree(self, rows: list[dict[str, Any]], combine_selected: list[int]) -> None:
        existing = set(self.steps_tree.get_children())
        wanted = {f"step_{int(row['index'])}" for row in rows}
        for item in existing - wanted:
            self.steps_tree.delete(item)
        for row in rows:
            index = int(row["index"])
            item = f"step_{index}"
            values = (
                index,
                row["name"],
                row["type"],
                f"{float(row['duration']):.3f}",
                row["events_count"],
                row["note"],
            )
            tags = ("combine_selected",) if index in combine_selected else ()
            if item in existing:
                self.steps_tree.item(item, values=values, tags=tags)
            else:
                self.steps_tree.insert("", "end", iid=item, values=values, tags=tags)
        if self.controller.combine_mode_enabled and combine_selected:
            items = [f"step_{int(index)}" for index in combine_selected if f"step_{int(index)}" in self.steps_tree.get_children()]
            if items:
                try:
                    self.steps_tree.selection_set(items)
                    self.steps_tree.see(items[-1])
                except Exception:
                    pass
        elif self.controller.selected_step_index:
            item = f"step_{self.controller.selected_step_index}"
            if item in self.steps_tree.get_children():
                try:
                    self.steps_tree.selection_set(item)
                    self.steps_tree.see(item)
                except Exception:
                    pass

    def _refresh_manifest_trees(self, rows: list[dict[str, Any]], task: dict[str, Any]) -> None:
        for tree in [getattr(self, "manifest_tree", None), getattr(getattr(self, "height_task_panel", None), "manifest_tree", None)]:
            if tree is None:
                continue
            tree.delete(*tree.get_children())
            task_height = task.get("current_height")
            for row in rows:
                height = int(row["height_cm"])
                marker = "current" if height == self.controller.current_height_cm else ""
                if task_height == height:
                    marker = "task"
                tree.insert(
                    "",
                    "end",
                    iid=f"manifest_{height}_{id(tree)}",
                    values=(
                        f"{height}cm",
                        "yes" if row["recorded"] else "no",
                        row["step_count"],
                        row["last_saved_at"],
                        marker,
                    ),
                )

    def _set_text(self, widget: Any, value: str, *, max_chars: int | None = None) -> None:
        limit = self.max_text_widget_chars if max_chars is None else max(1000, int(max_chars))
        text = str(value)
        if len(text) > limit:
            text = (
                f"[TRUNCATED] Text is {len(text)} chars; showing first {limit} chars. "
                "Use an export action for full data.\n\n"
                + text[:limit]
            )
        cache_key = id(widget)
        signature = (len(text), hash(text))
        if self._text_widget_cache.get(cache_key) == signature:
            return
        set_text_preserving_view(widget, text, follow_bottom=False)
        self._text_widget_cache[cache_key] = signature
