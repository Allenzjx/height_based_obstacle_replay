"""Standalone Isaac Sim worker process for height replay.

This script is launched by the Tk UI subprocess client. It owns SimulationApp,
scene creation, SimRobotAdapter, and the continuous sim loop in its main thread.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

from command_model import DEFAULT_MAX_WHEEL_SPEED_RAD_S, CommandMessage
from camera_validation import compare_detection_to_expected, default_camera_health_status, default_height_validation_result
from height_manifest import SUPPORTED_HEIGHTS_CM, normalize_height_cm, obstacle_height_m
from height_sequence_store import HeightSequenceStore
from obstacle_height_vision import VisionHeightConfig, estimate_height_from_depth, estimate_height_from_pointcloud
from playback import plan_from_steps
from sim_camera_viewport import default_camera_viewport_status
from sim_ipc_protocol import JsonLineBuffer, encode_message, make_message
from sim_obstacle_scene import (
    DEFAULT_ROBOT_USD_PATH,
    DEFAULT_SCENE_SAVE_PATH,
    SimSceneConfig,
    add_app_launcher_args,
    create_scene,
    ensure_simulation_app,
    finalize_scene_after_grounding,
    update_obstacle_height,
)
from sim_onboard_camera import OnboardCameraProcessor, camera_pitch_quat_wxyz, reset_camera
from sim_robot_adapter import SimRobotAdapter
from sim_worker_runtime import (
    build_common_worker_status,
    create_adapter_config_from_args,
    enrich_runtime_readiness,
    execute_viewport_action_with_guard,
    handle_respawn,
    handle_set_height,
    handle_vision_control,
    initialize_adapter_ground_reference,
    service_pending_viewport,
)
from robot_ground_diagnostics import GROUND_OK, default_robot_ground_diagnostics, motion_status_from_worker_status, respawn_status_from_worker_status
from telemetry import create_telemetry_collector
from telemetry.config import add_telemetry_args, load_telemetry_config


class WorkerIpc:
    def __init__(self, host: str, port: int):
        self.host = str(host or "")
        self.port = int(port or 0)
        self.sock: socket.socket | None = None
        self.buffer = JsonLineBuffer()
        if self.host and self.port:
            self.sock = socket.create_connection((self.host, self.port), timeout=10.0)
            self.sock.setblocking(False)

    def send(self, message: dict[str, Any]) -> None:
        payload = encode_message(message)
        if self.sock is None:
            sys.stdout.buffer.write(payload)
            sys.stdout.flush()
            return
        try:
            self.sock.setblocking(True)
            self.sock.sendall(payload)
        finally:
            try:
                self.sock.setblocking(False)
            except Exception:
                pass

    def poll(self) -> list[dict[str, Any]]:
        if self.sock is None:
            return []
        messages: list[dict[str, Any]] = []
        while True:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            messages.extend(self.buffer.feed(chunk))
        return messages

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


class WorkerLogger:
    def __init__(self, max_lines: int):
        self.lines: deque[str] = deque(maxlen=max(1, int(max_lines)))

    def log(self, text: str) -> None:
        line = str(text)
        self.lines.append(line)
        print(line, flush=True)

    def tail(self) -> list[str]:
        return list(self.lines)


def config_from_args(args: argparse.Namespace, height_cm: int) -> SimSceneConfig:
    return SimSceneConfig(
        obstacle_height_m=obstacle_height_m(height_cm),
        robot_usd=Path(args.robot_usd),
        save_usd=Path(args.save_usd),
        spawn_z=float(args.spawn_z),
        obstacle_x=float(args.obstacle_x),
        obstacle_width=args.obstacle_width,
        obstacle_length=args.obstacle_length,
        infer_obstacle_size=bool(args.infer_obstacle_size),
        robot_width=float(args.robot_width),
        robot_length=float(args.robot_length),
        physics_dt=float(args.physics_dt),
        render_interval=int(args.render_interval),
        device=str(getattr(args, "device", "cuda:0")),
        max_wheel_speed=float(args.max_wheel_speed_rad_s),
        default_wheel_speed=float(args.default_wheel_speed_rad_s),
        wheel_direction=float(args.wheel_direction),
        servo_stiffness=float(args.servo_stiffness),
        servo_damping=float(args.servo_damping),
        wheel_damping=float(args.wheel_damping),
        save_scene=bool(args.save_scene),
        onboard_camera_enabled=bool(getattr(args, "onboard_camera", True)),
        camera_parent_prim=str(getattr(args, "camera_parent_prim", "") or ""),
        camera_width=int(getattr(args, "camera_width", 424)),
        camera_height=int(getattr(args, "camera_height", 240)),
        camera_update_period_s=float(getattr(args, "camera_update_period_s", 0.10)),
        camera_offset_pos=(
            float(getattr(args, "camera_offset_x", 0.35)),
            float(getattr(args, "camera_offset_y", 0.0)),
            float(getattr(args, "camera_offset_z", 0.18)),
        ),
        camera_offset_rot=camera_pitch_quat_wxyz(float(getattr(args, "camera_pitch_deg", 14.0))),
        camera_offset_convention="world",
        camera_aim_mode=str(getattr(args, "camera_aim_mode", "pitch") or "pitch"),
        camera_target_x=float(getattr(args, "camera_target_x", getattr(args, "obstacle_x", 1.55))),
        camera_target_y=float(getattr(args, "camera_target_y", 0.0)),
        camera_target_z=float(getattr(args, "camera_target_z", 0.02)),
        camera_target_frame=str(getattr(args, "camera_target_frame", "world") or "world"),
        camera_look_at_roll_deg=float(getattr(args, "camera_look_at_roll_deg", 0.0)),
        camera_focal_length=float(getattr(args, "camera_focal_length", 24.0)),
        camera_horizontal_aperture=float(getattr(args, "camera_horizontal_aperture", 20.955)),
        camera_near_clip_m=float(getattr(args, "camera_near_clip_m", 0.05)),
        camera_far_clip_m=float(getattr(args, "camera_far_clip_m", 6.0)),
        camera_coverage_strict=bool(getattr(args, "camera_coverage_strict", False)),
        telemetry_contact_sensors_enabled=bool(getattr(args, "telemetry_contact_sensors_enabled", False)),
        defer_first_visible_render=bool(getattr(args, "defer_first_visible_render", True)),
    )


def vision_config_from_args(args: argparse.Namespace) -> VisionHeightConfig:
    return VisionHeightConfig(
        quantization_tolerance_cm=float(getattr(args, "vision_height_tolerance_cm", 2.0)),
        minimum_confidence=float(getattr(args, "vision_confidence_threshold", 0.75)),
        temporal_window_size=int(getattr(args, "vision_window_size", 7)),
        stable_frames_required=int(getattr(args, "vision_stable_frames", 5)),
    )


def run_negative_knee_smoke(adapter: SimRobotAdapter, logger: WorkerLogger) -> dict[str, Any]:
    knees = [
        "front_left_knee",
        "front_right_knee",
        "rear_left_knee",
        "rear_right_knee",
    ]
    command_deg = -30.0
    logger.log("[worker] negative knee smoke: sending command-space -30 deg to all knees")
    for name in knees:
        adapter.handle_command(CommandMessage(text=f"servo {name} {command_deg:g}", source="worker_smoke"))
    dt = float(adapter.sim.get_physics_dt())
    for _index in range(120):
        adapter.step(dt)
        time.sleep(0.0)
    actual = adapter.get_actual_joint_state().get("servos", {})
    diagnostics = {row["joint_name"]: row for row in adapter.get_joint_diagnostics()}
    rows: list[dict[str, Any]] = []
    ok = True
    for name in knees:
        standing = float(adapter.standing_pose_deg[name])
        measured = actual.get(name, {}).get("deg")
        expected_delta = float(adapter.servo_target_debug[name]["target_actual_deg"]) - standing
        measured_delta = None if measured is None else float(measured) - standing
        moved_expected_direction = False
        if measured_delta is not None:
            moved_expected_direction = (expected_delta < 0.0 and measured_delta < -0.5) or (expected_delta > 0.0 and measured_delta > 0.5)
        row = {
            "joint_name": name,
            "standing_pose_deg": standing,
            "command_deg": command_deg,
            "target_actual_deg": float(adapter.servo_target_debug[name]["target_actual_deg"]),
            "measured_actual_deg": measured,
            "expected_delta_deg": expected_delta,
            "measured_delta_deg": measured_delta,
            "target_inside_limit": diagnostics.get(name, {}).get("target_inside_current_limit", "unknown"),
            "limit_deg": diagnostics.get(name, {}).get("current_physx_or_usd_limit_deg"),
            "pass": moved_expected_direction,
        }
        rows.append(row)
        ok = ok and moved_expected_direction
        logger.log(f"[worker] negative knee smoke {name}: {'PASS' if moved_expected_direction else 'FAIL'} {row}")
    result = {"ok": ok, "command_deg": command_deg, "rows": rows}
    adapter.negative_knee_smoke_result = result
    logger.log(f"[worker] negative knee smoke result: {'PASS' if ok else 'FAIL'}")
    return result


def run_camera_detection_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    requested_height = getattr(args, "worker_smoke_camera_height_cm", None)
    heights = [normalize_height_cm(requested_height)] if requested_height is not None else list(SUPPORTED_HEIGHTS_CM)
    logger.log(f"[worker] camera detection smoke: starting heights={heights}")
    rows: list[dict[str, Any]] = []
    ok = True
    dt = float(scene_handle.sim.get_physics_dt())
    validation_s = max(2.0, float(getattr(args, "worker_smoke_camera_validation_s", 0.0) or 0.0))
    if validation_s <= 2.0:
        validation_s = max(2.0, float(getattr(args, "worker_smoke_test_s", 20.0)) / max(1, len(heights)))
    for target_cm in heights:
        update_obstacle_height(scene_handle, obstacle_height_m(target_cm))
        adapter.respawn_robot()
        reset_camera(scene_handle)
        vision.reset_filter()
        vision.request_camera_validation(duration_s=min(1.5, validation_s), min_frames=2)
        height_deadline = time.monotonic() + validation_s
        stable_status: dict[str, Any] | None = None
        while time.monotonic() < height_deadline and scene_handle.app_is_running():
            adapter.step(dt)
            status = vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
            if bool(status.get("stable")) and status.get("detected_height_cm") is not None:
                stable_status = status
                health = status.get("camera_health", {})
                if isinstance(health, dict) and bool(health.get("checked", False)):
                    break
            time.sleep(0.0)
        final_status = vision.status()
        validation = compare_detection_to_expected(
            final_status,
            int(target_cm),
            raw_height_tolerance_cm=1.0,
            minimum_confidence=float(getattr(args, "vision_confidence_threshold", 0.75)),
            minimum_obstacle_points=int(getattr(vision.config, "minimum_obstacle_points", 80)),
            quantization_tolerance_cm=float(getattr(args, "vision_height_tolerance_cm", 2.0)),
        )
        vision.height_validation_result = validation.to_dict()
        vision.save_debug_frame(expected_height_cm=int(target_cm))
        adapter.step(dt)
        vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
        final_status = vision.status()
        detected = validation.detected_height_cm
        row_ok = bool(validation.passed)
        health = final_status.get("camera_health", {})
        provenance = final_status.get("height_provenance", {})
        geometry = final_status.get("camera_geometry", {})
        provenance_ok = (
            isinstance(provenance, dict)
            and provenance.get("height_source") == "isaac_rgbd_depth_geometry"
            and bool(provenance.get("intrinsics_used", False))
            and bool(provenance.get("camera_world_pose_used", False))
            and not bool(provenance.get("expected_height_used_by_detector", True))
            and not bool(provenance.get("generated_height_used_by_detector", True))
            and not bool(provenance.get("scene_obstacle_height_used_by_detector", True))
            and bool(provenance.get("detector_input_audit_passed", False))
        )
        row = {
            "target_cm": int(target_cm),
            "expected_height_cm": int(target_cm),
            "camera_health_ok": bool(health.get("ok", False)) if isinstance(health, dict) else False,
            "camera_backend": (
                f"{health.get('backend_module', '')}.{health.get('backend_class', '')}"
                if isinstance(health, dict)
                else ""
            ),
            "rgb_available": bool(health.get("rgb_available", False)) if isinstance(health, dict) else False,
            "depth_available": bool(health.get("depth_available", False)) if isinstance(health, dict) else False,
            "frames_advanced_during_check": int(health.get("frames_advanced_during_check", 0)) if isinstance(health, dict) else 0,
            "raw_height_cm": validation.raw_height_cm,
            "absolute_error_cm": validation.absolute_error_cm,
            "absolute_error_mm": validation.absolute_error_mm,
            "detected_height_cm": detected,
            "candidate_height_cm": validation.candidate_height_cm,
            "confidence": validation.confidence,
            "stable_count": validation.stable_count,
            "valid_point_count": validation.valid_point_count,
            "top_plane_mad_m": validation.top_plane_mad_m,
            "pass": row_ok,
            "provenance_pass": bool(provenance_ok),
            "source_mode": str(final_status.get("source_mode", "")),
            "roi_source": str(final_status.get("roi_source", "")),
            "camera_geometry": dict(geometry) if isinstance(geometry, dict) else {},
            "height_provenance": dict(provenance) if isinstance(provenance, dict) else {},
            "failure_reasons": list(validation.reasons),
            "debug_image_path": final_status.get("debug_image_path", ""),
            "debug_sidecar_path": final_status.get("debug_sidecar_path", ""),
        }
        rows.append(row)
        ok = ok and row_ok
        logger.log(f"[worker] camera detection smoke {target_cm:02d}cm: {'PASS' if row_ok else 'FAIL'} {row}")
    result = {"ok": ok, "rows": rows}
    output = str(getattr(args, "worker_smoke_camera_output", "") or "").strip()
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        logger.log(f"[worker] camera detection smoke wrote {out_path}")
    logger.log(f"[worker] camera detection smoke result: {'PASS' if ok else 'FAIL'}")
    return result


def run_camera_provenance_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    """Run the real camera smoke plus small pure-detector provenance checks."""

    smoke_args = argparse.Namespace(**vars(args))
    if getattr(smoke_args, "worker_smoke_camera_height_cm", None) is None:
        smoke_args.worker_smoke_camera_height_cm = 5
    smoke_args.worker_smoke_camera_output = ""
    detection_result = run_camera_detection_smoke(smoke_args, scene_handle, adapter, vision, logger)
    rows = list(detection_result.get("rows", []))
    row_provenance_ok = all(bool(row.get("provenance_pass", False)) for row in rows)
    detector_audit = _run_pure_camera_provenance_audit()
    ok = bool(detection_result.get("ok", False)) and row_provenance_ok and bool(detector_audit.get("ok", False))
    result = {
        "ok": ok,
        "mode": "camera_provenance_smoke",
        "real_camera_detection": detection_result,
        "detector_input_audit": detector_audit,
        "rows": rows,
    }
    output = str(getattr(args, "worker_smoke_camera_output", "") or "").strip()
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        logger.log(f"[worker] camera provenance smoke wrote {out_path}")
    logger.log(f"[worker] camera provenance smoke result: {'PASS' if ok else 'FAIL'}")
    return result


def run_camera_counterfactual_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    """Real-camera provenance checks that vary geometry and metadata separately."""

    logger.log("[worker] camera counterfactual smoke: starting")
    dt = float(scene_handle.sim.get_physics_dt())
    validation_s = max(3.0, float(getattr(args, "worker_smoke_camera_validation_s", 10.0) or 10.0))

    def detect_geometry(*, geometry_cm: int, expected_cm: int, label: str) -> dict[str, Any]:
        update_obstacle_height(scene_handle, obstacle_height_m(int(geometry_cm)))
        adapter.respawn_robot()
        reset_camera(scene_handle)
        vision.reset_filter()
        vision.request_camera_validation(duration_s=min(1.5, validation_s), min_frames=2)
        deadline = time.monotonic() + validation_s
        while time.monotonic() < deadline and scene_handle.app_is_running():
            adapter.step(dt)
            status = vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
            if bool(status.get("stable")) and status.get("detected_height_cm") is not None:
                health = status.get("camera_health", {})
                if isinstance(health, dict) and bool(health.get("checked", False)):
                    break
            time.sleep(0.0)
        status = vision.status()
        validation = compare_detection_to_expected(
            status,
            int(expected_cm),
            raw_height_tolerance_cm=1.0,
            minimum_confidence=float(getattr(args, "vision_confidence_threshold", 0.75)),
            minimum_obstacle_points=int(getattr(vision.config, "minimum_obstacle_points", 80)),
            quantization_tolerance_cm=float(getattr(args, "vision_height_tolerance_cm", 2.0)),
        )
        evidence = status.get("height_measurement_evidence", {})
        provenance = status.get("height_provenance", {})
        row = {
            "label": str(label),
            "geometry_cm": int(geometry_cm),
            "metadata_expected_cm": int(expected_cm),
            "raw_height_cm": status.get("raw_height_cm"),
            "detected_height_cm": status.get("detected_height_cm"),
            "candidate_height_cm": status.get("candidate_height_cm"),
            "confidence": status.get("confidence"),
            "stable": bool(status.get("stable", False)),
            "stable_count": int(status.get("stable_count", 0) or 0),
            "validation_passed_against_metadata": bool(validation.passed),
            "validation_reasons": list(validation.reasons),
            "height_measurement_evidence": dict(evidence) if isinstance(evidence, dict) else {},
            "height_provenance": dict(provenance) if isinstance(provenance, dict) else {},
        }
        logger.log(f"[worker] camera counterfactual {label}: {row}")
        return row

    normal_5cm = detect_geometry(geometry_cm=5, expected_cm=5, label="normal_5cm")
    metadata_mismatch = detect_geometry(geometry_cm=10, expected_cm=5, label="metadata_5cm_geometry_10cm")
    same_frame_status = vision.status()
    validation_20 = compare_detection_to_expected(
        same_frame_status,
        20,
        raw_height_tolerance_cm=1.0,
        minimum_confidence=float(getattr(args, "vision_confidence_threshold", 0.75)),
        minimum_obstacle_points=int(getattr(vision.config, "minimum_obstacle_points", 80)),
        quantization_tolerance_cm=float(getattr(args, "vision_height_tolerance_cm", 2.0)),
    )
    expected_change_only = {
        "label": "expected_20cm_same_depth_frame",
        "raw_height_cm": same_frame_status.get("raw_height_cm"),
        "detected_height_cm": same_frame_status.get("detected_height_cm"),
        "candidate_height_cm": same_frame_status.get("candidate_height_cm"),
        "depth_frame_revision": (same_frame_status.get("height_measurement_evidence", {}) or {}).get("depth_frame_revision")
        if isinstance(same_frame_status.get("height_measurement_evidence"), dict)
        else same_frame_status.get("frame_revision"),
        "validation_passed_against_expected_20cm": bool(validation_20.passed),
        "validation_reasons": list(validation_20.reasons),
    }
    frame = dict(getattr(vision, "_last_frame_sample", {}) or {})
    depth_unavailable = estimate_height_from_depth(
        None,
        frame.get("intrinsics"),
        frame.get("pos_w", (0.0, 0.0, 0.0)),
        frame.get("quat_w", (1.0, 0.0, 0.0, 0.0)),
        config=vision.config,
        obstacle_x_m=float(getattr(scene_handle.config, "obstacle_x", 1.55)),
        timestamp=time.time(),
        near_clip_m=float(getattr(scene_handle.config, "camera_near_clip_m", 0.05)),
        far_clip_m=float(getattr(scene_handle.config, "camera_far_clip_m", 6.0)),
    )
    depth_unavailable_row = {
        "label": "depth_unavailable",
        "valid": bool(depth_unavailable.valid),
        "detected_height_cm": depth_unavailable.detected_height_cm,
        "raw_height_cm": depth_unavailable.raw_height_cm,
        "reason": str(depth_unavailable.reason),
    }
    frozen_before_revision = int(same_frame_status.get("detection_revision", 0) or 0)
    frozen_before_fingerprint = ""
    if isinstance(same_frame_status.get("height_measurement_evidence"), dict):
        frozen_before_fingerprint = str(same_frame_status["height_measurement_evidence"].get("depth_fingerprint", "") or "")
    update_obstacle_height(scene_handle, obstacle_height_m(5))
    frozen_after_status = vision.status()
    frozen_depth_row = {
        "label": "depth_frozen_without_new_camera_update",
        "detection_revision_before": frozen_before_revision,
        "detection_revision_after": int(frozen_after_status.get("detection_revision", 0) or 0),
        "depth_fingerprint_before": frozen_before_fingerprint,
        "depth_fingerprint_after": (
            str(frozen_after_status.get("height_measurement_evidence", {}).get("depth_fingerprint", "") or "")
            if isinstance(frozen_after_status.get("height_measurement_evidence"), dict)
            else ""
        ),
        "published_new_detection": int(frozen_after_status.get("detection_revision", 0) or 0) != frozen_before_revision,
    }
    ok = (
        bool(normal_5cm.get("validation_passed_against_metadata", False))
        and int(normal_5cm.get("detected_height_cm") or -1) == 5
        and int(metadata_mismatch.get("detected_height_cm") or -1) == 10
        and not bool(metadata_mismatch.get("validation_passed_against_metadata", True))
        and int(expected_change_only.get("detected_height_cm") or -1) == 10
        and not bool(expected_change_only.get("validation_passed_against_expected_20cm", True))
        and not bool(depth_unavailable_row.get("valid", True))
        and not bool(frozen_depth_row.get("published_new_detection", True))
    )
    result = {
        "ok": bool(ok),
        "mode": "camera_counterfactual",
        "normal_5cm": normal_5cm,
        "metadata_mismatch": metadata_mismatch,
        "expected_change_only": expected_change_only,
        "depth_unavailable": depth_unavailable_row,
        "depth_frozen": frozen_depth_row,
    }
    output = str(getattr(args, "worker_smoke_camera_counterfactual_output", "") or "").strip()
    if not output:
        output = str(getattr(args, "worker_smoke_output", "") or "").strip()
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        logger.log(f"[worker] camera counterfactual smoke wrote {out_path}")
    logger.log(f"[worker] camera counterfactual smoke result: {'PASS' if ok else 'FAIL'}")
    return result


def run_camera_view_ground_contact_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    logger.log("[worker] camera view ground-contact smoke: starting")
    dt = float(scene_handle.sim.get_physics_dt())
    report: dict[str, Any] = {
        "ok": False,
        "mode": "camera_view_ground_contact",
        "rows": [],
        "debug_paths": [],
        "trigger_stage": "",
    }

    def row(name: str, payload: dict[str, Any] | None = None) -> None:
        payload = dict(payload or {})
        payload.setdefault("trigger_stage", name)
        item = {"name": name, **payload}
        report["rows"].append(item)
        logger.log(f"[worker] camera view ground smoke {name}: {item}")

    def ground(stage: str) -> dict[str, Any]:
        diagnostics = adapter.validate_robot_ground_contact(apply_correction=False)
        diagnostics["trigger_stage"] = stage
        return diagnostics

    def step_frames(count: int, label: str) -> dict[str, Any]:
        for _index in range(max(0, int(count))):
            service_pending_viewport(
                args=args,
                adapter=adapter,
                scene_handle=scene_handle,
                vision_processor=vision,
                logger_tail_provider=logger.tail,
                runtime_metrics={"started_wall": time.monotonic()},
            )
            adapter.step(dt)
            vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
        return ground(label)

    startup_ground = ground("startup_baseline")
    row("startup_baseline", {"ground": startup_ground, "camera_view": dict(vision.camera_view_status)})

    set_height_result = handle_set_height(
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision,
        height_cm=5,
        source="worker_smoke_camera_view_ground_contact",
        request_id="worker_smoke_set_height_5cm",
        obstacle_revision=1,
    )
    reset_camera(scene_handle)
    after_height_ground = ground("after_set_height_respawn")
    row("after_set_height_respawn", {"ground": after_height_ground, "set_height": set_height_result})

    camera_path = str(getattr(scene_handle, "camera_prim_path", "") or "")
    payload = {
        "request_id": "worker_smoke_camera_view_ground_contact",
        "action_revision": 1,
        "headless": bool(getattr(args, "headless", False)),
        "camera_prim_path": camera_path,
        "active_fallback_allowed": bool(getattr(args, "camera_view_active_fallback", False)),
        "camera_view_pending_timeout_s": float(getattr(args, "camera_view_pending_timeout_s", 10.0)),
        "camera_view_pending_max_retries": int(getattr(args, "camera_view_pending_max_retries", 30)),
    }

    before_open_ground = ground("immediately_before_open")
    row("immediately_before_open", {"ground": before_open_ground, "camera_view": dict(vision.camera_view_status)})
    open_ok, open_text, open_status = execute_viewport_action_with_guard(
        args=args,
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision,
        action="open_camera_viewport",
        payload=payload,
        logger_tail_provider=logger.tail,
        runtime_metrics={"started_wall": time.monotonic()},
        allow_state_restore=False,
    )
    after_open_ground = ground("immediately_after_open")
    row("immediately_after_open", {"ok": open_ok, "text": open_text, "ground": after_open_ground, "camera_view": open_status})

    after_open_1 = step_frames(1, "after_open_1_step")
    row("after_open_1_step", {"ground": after_open_1, "camera_view": dict(vision.camera_view_status)})
    after_open_10 = step_frames(9, "after_open_10_steps")
    row("after_open_10_steps", {"ground": after_open_10, "camera_view": dict(vision.camera_view_status)})
    after_open_60 = step_frames(50, "after_open_60_steps")
    row("after_open_60_steps", {"ground": after_open_60, "camera_view": dict(vision.camera_view_status)})

    before_return_ground = ground("before_return_perspective")
    row("before_return_perspective", {"ground": before_return_ground, "camera_view": dict(vision.camera_view_status)})
    return_payload = dict(payload)
    return_payload["action_revision"] = 2
    return_ok, return_text, return_status = execute_viewport_action_with_guard(
        args=args,
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision,
        action="return_main_view_to_perspective",
        payload=return_payload,
        logger_tail_provider=logger.tail,
        runtime_metrics={"started_wall": time.monotonic()},
        allow_state_restore=False,
    )
    after_return_ground = ground("after_return_perspective")
    row("after_return_perspective", {"ok": return_ok, "text": return_text, "ground": after_return_ground, "camera_view": return_status})

    after_return_60 = step_frames(60, "after_return_60_steps")
    row("after_return_60_steps", {"ground": after_return_60, "camera_view": dict(vision.camera_view_status)})

    close_payload = dict(payload)
    close_payload["action_revision"] = 3
    close_ok, close_text, close_status = execute_viewport_action_with_guard(
        args=args,
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision,
        action="close_camera_viewport",
        payload=close_payload,
        logger_tail_provider=logger.tail,
        runtime_metrics={"started_wall": time.monotonic()},
        allow_state_restore=False,
    )
    after_close_ground = ground("after_close_camera_viewport")
    row("after_close_camera_viewport", {"ok": close_ok, "text": close_text, "ground": after_close_ground, "camera_view": close_status})

    control_root_z = startup_ground.get("root_z_m")
    camera_root_z = after_open_60.get("root_z_m")
    root_z_diff = None
    if control_root_z is not None and camera_root_z is not None:
        root_z_diff = abs(float(control_root_z) - float(camera_root_z))
    validation_ok = True
    reasons: list[str] = []
    if not bool(open_status.get("physics_guard_passed", False)):
        validation_ok = False
        reasons.append("open_camera_viewport immediate physics guard failed")
    if not bool(return_status.get("perspective_restore_verified", False)):
        validation_ok = False
        reasons.append("return_main_view_to_perspective postcondition was not verified")
    if root_z_diff is not None and root_z_diff > 0.002:
        validation_ok = False
        reasons.append(f"startup/open60 root z diff {root_z_diff:.6f}m exceeds 0.002m")
    tolerance = float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003))
    trigger_stage = ""
    for item in report["rows"]:
        stage = str(item.get("name", ""))
        ground_diag = item.get("ground", {})
        camera_view = item.get("camera_view", {})
        if isinstance(ground_diag, dict) and float(ground_diag.get("maximum_collision_penetration_m", 0.0) or 0.0) > tolerance:
            validation_ok = False
            reasons.append(f"{stage} collision penetration exceeds tolerance")
            trigger_stage = trigger_stage or stage
        if isinstance(ground_diag, dict) and ground_diag.get("missing_collision_wheels"):
            validation_ok = False
            reasons.append(f"{stage} missing wheel collision: {ground_diag.get('missing_collision_wheels')}")
            trigger_stage = trigger_stage or stage
        if isinstance(ground_diag, dict) and str(ground_diag.get("classification", "")) not in {"", GROUND_OK}:
            trigger_stage = trigger_stage or stage
        if isinstance(camera_view, dict) and bool(camera_view.get("physics_guard_checked", False)) and not bool(camera_view.get("physics_guard_passed", False)):
            trigger_stage = trigger_stage or stage
    trigger_stage = trigger_stage or ("none" if validation_ok else "unknown")
    report.update(
        {
            "ok": bool(validation_ok),
            "reasons": reasons,
            "control_root_z_m": control_root_z,
            "camera_root_z_m": camera_root_z,
            "control_camera_root_z_diff_m": root_z_diff,
            "camera_view": dict(vision.camera_view_status),
            "baseline_root_z_m": startup_ground.get("root_z_m"),
            "trigger_stage": trigger_stage,
            "grounded_reference_valid": bool(getattr(adapter, "grounded_reference_valid", False)),
            "grounded_reference_physics_valid": bool(getattr(adapter, "grounded_reference_physics_valid", False)),
            "grounded_reference_visual_valid": bool(getattr(adapter, "grounded_reference_visual_valid", False)),
            "grounded_reference_stable": bool(getattr(adapter, "grounded_reference_stable", False)),
        }
    )
    output_dir = Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"camera_view_ground_contact_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)
    logger.log(f"[worker] camera view ground-contact smoke wrote {out_path}")
    logger.log(f"[worker] camera view ground-contact smoke result: {'PASS' if validation_ok else 'FAIL'}")
    return report


def run_ground_structure_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    logger: WorkerLogger,
) -> dict[str, Any]:
    logger.log("[worker] ground structure smoke: collecting runtime USD/articulation diagnostics")
    ground = adapter.validate_robot_ground_contact(apply_correction=False)
    report = {
        "ok": bool(ground.get("checked", False)),
        "mode": "ground_structure",
        "robot_usd": str(getattr(args, "robot_usd", "") or ""),
        "robot_usd_exists": Path(str(getattr(args, "robot_usd", "") or "")).exists(),
        "runtime_robot_prim_path": str(getattr(scene_handle, "robot_prim_path", "") or "/World/WLRRobot"),
        "robot_joint_names": list(getattr(adapter.robot, "joint_names", []) or []),
        "robot_body_names": list(getattr(adapter.robot, "body_names", []) or []),
        "joint_name_to_id": {str(name): int(index) for index, name in enumerate(list(getattr(adapter.robot, "joint_names", []) or []))},
        "body_name_to_id": {str(name): int(index) for index, name in enumerate(list(getattr(adapter.robot, "body_names", []) or []))},
        "ground": ground,
        "ground_surface": dict(ground.get("ground_surface", {}) or {}),
        "wheel_rows": list(ground.get("wheels", []) or []),
        "prim_tree": _runtime_robot_prim_tree(scene_handle),
        "last_ground_settle_result": dict(getattr(adapter, "last_ground_settle_result", {}) or {}),
        "status": enrich_runtime_readiness(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle)),
    }
    path = _write_worker_smoke_report(args, report, logger, "ground_structure")
    report["report_path"] = str(path)
    logger.log(f"[worker] ground structure smoke result: {'PASS' if report['ok'] else 'FAIL'}")
    return report


def run_ground_calibration_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    logger: WorkerLogger,
) -> dict[str, Any]:
    logger.log("[worker] ground calibration smoke: settling and calibrating grounded reference")
    before = enrich_runtime_readiness(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle))
    settle = adapter.settle_robot_on_ground(label="worker_smoke_ground_calibration")
    calibration = adapter.calibrate_grounded_reference()
    after = enrich_runtime_readiness(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle))
    motion_ready, motion_reason, ground_state = motion_status_from_worker_status(after)
    respawn_ready, respawn_reason = respawn_status_from_worker_status(after)
    ok = bool(motion_ready and respawn_ready and calibration.get("grounded_reference_valid", False))
    report = {
        "ok": ok,
        "mode": "ground_calibration",
        "before": before,
        "settle": settle,
        "calibration": calibration,
        "after": after,
        "ground_state": ground_state,
        "motion_ready": bool(motion_ready),
        "motion_block_reason": motion_reason,
        "respawn_ready": bool(respawn_ready),
        "respawn_block_reason": respawn_reason,
        "joint_velocity_by_name": dict(settle.get("joint_velocity_by_name", {}) or {}),
        "final_window_joint_velocity_by_name": list(settle.get("final_window_joint_velocity_by_name", []) or []),
        "offending_joint_names": list(settle.get("offending_joint_names", []) or []),
        "wheel_rows": list((settle.get("ground_diagnostics", {}) or {}).get("wheels", []) or []),
    }
    path = _write_worker_smoke_report(args, report, logger, "ground_calibration")
    report["report_path"] = str(path)
    logger.log(f"[worker] ground calibration smoke result: {'PASS' if ok else 'FAIL'}")
    return report


def run_vision_playback_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    height_cm = normalize_height_cm(getattr(args, "worker_smoke_camera_height_cm", None) or 5)
    logger.log(f"[worker] vision playback smoke: generated {height_cm}cm, validate, load steps, gated playback probe")
    dt = float(scene_handle.sim.get_physics_dt())
    set_height_result = handle_set_height(
        adapter=adapter,
        scene_handle=scene_handle,
        vision_processor=vision,
        height_cm=height_cm,
        source="worker_smoke_vision_playback",
        request_id=f"worker_smoke_vision_playback_{height_cm}",
        obstacle_revision=1,
        respawn_policy="if_motion_ready",
    )
    reset_camera(scene_handle)
    vision.reset_filter()
    validation_s = max(2.0, float(getattr(args, "worker_smoke_camera_validation_s", 10.0) or 10.0))
    vision.request_camera_validation(duration_s=min(1.5, validation_s), min_frames=2)
    deadline = time.monotonic() + validation_s
    while time.monotonic() < deadline and scene_handle.app_is_running():
        adapter.step(dt)
        vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
        status = vision.status()
        if bool(status.get("stable")) and status.get("detected_height_cm") == height_cm:
            health = status.get("camera_health", {})
            if isinstance(health, dict) and bool(health.get("checked", False)):
                break
        time.sleep(0.0)
    final_status = vision.status()
    validation = compare_detection_to_expected(
        final_status,
        int(height_cm),
        raw_height_tolerance_cm=1.0,
        minimum_confidence=float(getattr(args, "vision_confidence_threshold", 0.75)),
        minimum_obstacle_points=int(getattr(vision.config, "minimum_obstacle_points", 80)),
        quantization_tolerance_cm=float(getattr(args, "vision_height_tolerance_cm", 2.0)),
    )
    vision.height_validation_result = validation.to_dict()
    store = HeightSequenceStore()
    steps = store.load_steps(height_cm)
    steps_path = store.steps_path(height_cm)
    common_status = enrich_runtime_readiness(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle))
    motion_ready, motion_reason, _ground_state = motion_status_from_worker_status(common_status)
    respawn_ready, respawn_reason = respawn_status_from_worker_status(common_status)
    gate_ok = bool(validation.passed and steps and motion_ready and respawn_ready)
    playback_started = False
    playback_error = ""
    if gate_ok:
        respawn = adapter.respawn_robot(settle=True)
        if not bool(respawn.get("ok", True)):
            gate_ok = False
            playback_error = str(respawn.get("error", "respawn failed before playback"))
        else:
            try:
                plan = plan_from_steps(
                    steps,
                    profile="fast",
                    speed=1.0,
                    preserve_wheel_distance=True,
                    max_wheel_speed=adapter.max_wheel_speed,
                    label=f"{height_cm}cm worker vision playback smoke",
                )
                start_sim_time = float(adapter.sim_time)
                event_index = 0
                playback_started = True
                while scene_handle.app_is_running() and float(adapter.sim_time) - start_sim_time < 0.5:
                    elapsed = float(adapter.sim_time) - start_sim_time
                    while event_index < len(plan.events) and plan.events[event_index].time_s <= elapsed + 1.0e-6:
                        adapter.handle_command(plan.events[event_index].command)
                        event_index += 1
                    adapter.step(dt)
                    vision.update(dt=dt, sim_time=float(adapter.sim_time), wall_time=time.time())
                    time.sleep(0.0)
            except Exception as exc:
                playback_error = str(exc)
            finally:
                adapter.stop_wheels()
                adapter.apply_commands_to_robot()
                adapter.robot.write_data_to_sim()
    report = {
        "ok": bool(validation.passed and len(steps) == 35 and playback_started and not playback_error),
        "mode": "vision_playback",
        "height_cm": int(height_cm),
        "set_height": set_height_result,
        "vision": vision.status(),
        "height_validation": validation.to_dict(),
        "steps_path": str(steps_path),
        "steps_count": len(steps),
        "motion_ready": bool(motion_ready),
        "motion_block_reason": motion_reason,
        "respawn_ready": bool(respawn_ready),
        "respawn_block_reason": respawn_reason,
        "playback_gate_ok": bool(gate_ok),
        "playback_started": bool(playback_started),
        "playback_error": playback_error,
        "wheel_targets_after_stop": adapter.capture_command_state().get("wheels", {}),
        "ground": adapter.validate_robot_ground_contact(apply_correction=False),
    }
    if not gate_ok and not playback_error:
        report["playback_block_reason"] = "; ".join(
            reason
            for reason in [
                "" if validation.passed else "height validation did not pass",
                "" if steps else "validated steps are missing",
                "" if motion_ready else motion_reason or "motion is not ready",
                "" if respawn_ready else respawn_reason or "respawn is not ready",
            ]
            if reason
        )
    path = _write_worker_smoke_report(args, report, logger, "vision_playback")
    report["report_path"] = str(path)
    logger.log(f"[worker] vision playback smoke result: {'PASS' if report['ok'] else 'FAIL'}")
    return report


def run_camera_pose_ab_smoke(
    args: argparse.Namespace,
    scene_handle: Any,
    adapter: SimRobotAdapter,
    vision: OnboardCameraProcessor,
    logger: WorkerLogger,
) -> dict[str, Any]:
    smoke_args = argparse.Namespace(**vars(args))
    if getattr(smoke_args, "worker_smoke_camera_height_cm", None) is None:
        smoke_args.worker_smoke_camera_height_cm = 5
    smoke_args.worker_smoke_camera_output = ""
    detection = run_camera_detection_smoke(smoke_args, scene_handle, adapter, vision, logger)
    report = {
        "ok": bool(detection.get("ok", False)),
        "mode": "camera_pose_ab",
        "current_camera_aim_mode": str(getattr(args, "camera_aim_mode", "pitch") or "pitch"),
        "current_camera_target": {
            "x": float(getattr(args, "camera_target_x", 0.0) or 0.0),
            "y": float(getattr(args, "camera_target_y", 0.0) or 0.0),
            "z": float(getattr(args, "camera_target_z", 0.0) or 0.0),
            "frame": str(getattr(args, "camera_target_frame", "world") or "world"),
        },
        "detection": detection,
        "note": "This worker branch validates the currently launched camera pose. Full pitch-vs-look-at A/B requires launching the worker twice with different camera pose CLI args.",
    }
    path = _write_worker_smoke_report(args, report, logger, "camera_pose_ab")
    report["report_path"] = str(path)
    logger.log(f"[worker] camera pose smoke result: {'PASS' if report['ok'] else 'FAIL'}")
    return report


def _runtime_robot_prim_tree(scene_handle: Any) -> list[dict[str, Any]]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore
        try:
            from pxr import PhysxSchema  # type: ignore
        except Exception:
            PhysxSchema = None  # type: ignore
    except Exception as exc:
        return [{"error": f"pxr unavailable: {exc}"}]
    stage = None
    for attr in ("stage", "usd_stage"):
        stage = getattr(scene_handle, attr, None)
        if stage is not None:
            break
    if stage is None:
        try:
            stage = scene_handle.sim.stage
        except Exception:
            stage = None
    if stage is None:
        return [{"error": "stage unavailable"}]
    root_path = str(getattr(scene_handle, "robot_prim_path", "") or "/World/WLRRobot")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return [{"error": f"robot root unavailable: {root_path}"}]
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows: list[dict[str, Any]] = []
    try:
        prims = [root] + list(root.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        queue = list(prims)
        seen: set[str] = set()
        while queue:
            prim = queue.pop(0)
            path = str(prim.GetPath())
            if path in seen:
                continue
            seen.add(path)
            queue.extend(list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies())))
            rows.append(_prim_diagnostic_row(prim, xform_cache, UsdPhysics, PhysxSchema))
            if len(rows) > 5000:
                rows.append({"warning": "prim tree truncated at 5000 rows"})
                break
    except Exception as exc:
        rows.append({"error": f"prim traversal failed: {exc}"})
    return rows


def _prim_diagnostic_row(prim: Any, xform_cache: Any, UsdPhysics: Any, PhysxSchema: Any | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "name": str(prim.GetName()),
        "type": str(prim.GetTypeName()),
        "is_instance": _safe_call_bool(prim, "IsInstance"),
        "is_instance_proxy": _safe_call_bool(prim, "IsInstanceProxy"),
        "prototype_path": "",
        "applied_schemas": [],
        "has_rigid_body_api": False,
        "has_collision_api": False,
        "has_physx_collision_api": False,
        "collision_enabled": None,
        "local_transform": [],
        "world_transform": [],
        "relationship_targets": {},
    }
    try:
        proto = prim.GetPrototype()
        if proto:
            row["prototype_path"] = str(proto.GetPath())
    except Exception:
        pass
    try:
        row["applied_schemas"] = [str(item) for item in prim.GetAppliedSchemas()]
    except Exception:
        pass
    try:
        row["has_rigid_body_api"] = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))
    except Exception:
        pass
    try:
        row["has_collision_api"] = bool(prim.HasAPI(UsdPhysics.CollisionAPI))
        api = UsdPhysics.CollisionAPI(prim)
        attr = api.GetCollisionEnabledAttr()
        if attr:
            value = attr.Get()
            row["collision_enabled"] = True if value is None else bool(value)
    except Exception:
        pass
    if PhysxSchema is not None:
        try:
            row["has_physx_collision_api"] = bool(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))
        except Exception:
            pass
    try:
        xform = UsdGeom.Xformable(prim)
        row["local_transform"] = _matrix_to_rows(xform.GetLocalTransformation())
    except Exception:
        pass
    try:
        row["world_transform"] = _matrix_to_rows(xform_cache.GetLocalToWorldTransform(prim))
    except Exception:
        pass
    for rel_name in ("physics:body0", "physics:body1", "body0", "body1"):
        try:
            rel = prim.GetRelationship(rel_name)
            if rel:
                row["relationship_targets"][rel_name] = [str(target) for target in rel.GetTargets()]
        except Exception:
            pass
    return row


def _matrix_to_rows(matrix: Any) -> list[list[float]]:
    try:
        return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]
    except Exception:
        return []


def _safe_call_bool(obj: Any, name: str) -> bool:
    try:
        method = getattr(obj, name)
        return bool(method())
    except Exception:
        return False


def _write_worker_smoke_report(args: argparse.Namespace, report: dict[str, Any], logger: WorkerLogger, mode: str) -> Path:
    explicit = str(getattr(args, "worker_smoke_output", "") or "").strip()
    if not explicit:
        explicit = str(getattr(args, "worker_smoke_camera_output", "") or "").strip()
    if explicit:
        path = Path(explicit)
    else:
        output_dir = Path(__file__).resolve().parent / "saved_height_steps" / "vision_debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    logger.log(f"[worker] {mode} smoke wrote {path}")
    return path


def _run_pure_camera_provenance_audit() -> dict[str, Any]:
    import numpy as np  # type: ignore

    cfg = VisionHeightConfig(
        minimum_confidence=0.1,
        minimum_obstacle_points=20,
        quantization_tolerance_cm=2.0,
        temporal_window_size=3,
        stable_frames_required=2,
    )
    points_10 = _synthetic_camera_pointcloud(obstacle_height_m=0.10, obstacle_x_m=1.55)
    valid = np.ones(points_10.shape[:2], dtype=bool)
    metadata_mismatch = estimate_height_from_pointcloud(
        points_10,
        valid_mask=valid,
        config=cfg,
        obstacle_x_m=1.55,
        camera_position_w=(0.0, 0.0, 0.2),
        timestamp=1.0,
    )
    depth_missing = estimate_height_from_depth(
        None,
        np.eye(3, dtype=float),
        (0.0, 0.0, 0.2),
        (1.0, 0.0, 0.0, 0.0),
        config=cfg,
        timestamp=1.0,
    )
    outside_prior = estimate_height_from_pointcloud(
        _synthetic_camera_pointcloud(obstacle_height_m=0.10, obstacle_x_m=2.30),
        valid_mask=valid,
        config=cfg,
        obstacle_x_m=1.55,
        camera_position_w=(0.0, 0.0, 0.2),
        timestamp=1.0,
    )
    external_auto = estimate_height_from_pointcloud(
        _synthetic_camera_pointcloud(obstacle_height_m=0.10, obstacle_x_m=2.30),
        valid_mask=valid,
        config=cfg,
        obstacle_x_m=None,
        camera_position_w=(0.0, 0.0, 0.2),
        timestamp=1.0,
    )
    checks = {
        "metadata_5cm_pointcloud_10cm_outputs_10cm": bool(
            metadata_mismatch.detected_height_cm == 10
            and metadata_mismatch.raw_height_cm is not None
            and abs(float(metadata_mismatch.raw_height_cm) - 10.0) <= 0.5
        ),
        "depth_none_invalid": not bool(depth_missing.valid),
        "generated_x_prior_rejects_moved_obstacle": outside_prior.detected_height_cm != 10,
        "external_auto_roi_accepts_moved_obstacle": bool(external_auto.valid and external_auto.detected_height_cm == 10),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "metadata_mismatch_raw_height_cm": metadata_mismatch.raw_height_cm,
        "metadata_mismatch_detected_height_cm": metadata_mismatch.detected_height_cm,
        "depth_none_reason": depth_missing.reason,
        "generated_prior_reason": outside_prior.reason,
        "external_auto_raw_height_cm": external_auto.raw_height_cm,
        "external_auto_detected_height_cm": external_auto.detected_height_cm,
        "scope": "pure detector audit; real Isaac camera row above proves live RGB-D source",
    }


def _synthetic_camera_pointcloud(*, obstacle_height_m: float, obstacle_x_m: float) -> Any:
    import numpy as np  # type: ignore

    height = 12
    width = 12
    points = np.zeros((height, width, 3), dtype=float)
    points[..., 0] = float(obstacle_x_m)
    points[..., 1] = np.linspace(-0.25, 0.25, width, dtype=float)[None, :]
    points[..., 2] = 0.0
    points[3:10, :, 2] = float(obstacle_height_m)
    return points


def run_worker(args: argparse.Namespace) -> int:
    ipc = WorkerIpc(args.ipc_host, int(args.ipc_port))
    logger = WorkerLogger(args.sim_worker_log_lines)
    height_cm = normalize_height_cm(args.height_cm)
    phase = "process_started"
    started_wall = time.monotonic()
    phase_started_wall = started_wall
    phase_history: list[dict[str, Any]] = []
    startup_pose_trace: list[dict[str, Any]] = []
    simulation_app = None
    scene_handle = None
    adapter = None
    telemetry_collector = None
    telemetry_finish_success = False
    telemetry_finish_reason = "worker did not complete"
    sim_time = 0.0
    sim_steps = 0
    last_status_wall = 0.0
    respawn_count = 0
    last_respawn_at = 0.0
    restore_count = 0
    last_restore_at = 0.0
    last_restore_result = ""
    last_restore_error = ""
    obstacle_revision = 0
    last_set_height_source = "startup"
    last_set_height_request_id = ""
    last_set_height_result: dict[str, Any] = {}
    shutdown_requested = False
    vision_processor: OnboardCameraProcessor | None = None
    vision_control_error = ""
    smoke_deadline = started_wall + float(args.worker_smoke_test_s) if float(args.worker_smoke_test_s) > 0 else None

    def publish_status(*, ready: bool, starting: bool, error: str = "", tb: str = "") -> None:
        status = build_status(
            args,
            adapter,
            phase=phase,
            phase_started_wall=phase_started_wall,
            phase_history=phase_history,
            startup_pose_trace=startup_pose_trace,
            ready=ready,
            starting=starting,
            height_cm=height_cm,
            sim_time=sim_time,
            sim_steps=sim_steps,
            started_wall=started_wall,
            error=error,
            tb=tb,
            log_tail=logger.tail(),
            respawn_count=respawn_count,
            last_respawn_at=last_respawn_at,
            restore_count=restore_count,
            last_restore_at=last_restore_at,
            last_restore_result=last_restore_result,
            last_restore_error=last_restore_error,
            obstacle_revision=obstacle_revision,
            last_set_height_source=last_set_height_source,
            last_set_height_request_id=last_set_height_request_id,
            last_set_height_result=last_set_height_result,
            scene_handle=scene_handle,
            vision_status=vision_processor.status(failure_reason=vision_control_error or None) if vision_processor is not None else default_vision_status(args),
        )
        ipc.send(make_message("status", **status))

    def set_phase(name: str, details: dict[str, Any] | None = None, *, publish: bool = True) -> None:
        nonlocal phase, phase_started_wall
        phase = str(name)
        phase_started_wall = time.monotonic()
        row = {
            "phase": phase,
            "wall_time": time.time(),
            "elapsed_s": max(0.0, phase_started_wall - started_wall),
            "details": dict(details or {}),
        }
        phase_history.append(row)
        del phase_history[:-50]
        startup_pose_trace.append(
            _startup_pose_trace_row(
                phase,
                scene_handle=scene_handle,
                adapter=adapter,
                sim_time=sim_time,
                sim_steps=sim_steps,
                visible_rendered=bool(scene_handle is not None and getattr(scene_handle, "first_visible_render_completed", False)),
            )
        )
        del startup_pose_trace[:-120]
        logger.log(f"[worker] phase={phase} details={row['details']}")
        if publish:
            try:
                publish_status(ready=False, starting=True)
            except Exception:
                pass

    try:
        ipc.send(make_message("hello", pid=os.getpid(), phase=phase, ready=False, starting=True))
        logger.log(f"[worker] pid={os.getpid()} connected; initial height={height_cm}cm")

        set_phase("starting_app")
        logger.log("[worker] starting Isaac SimulationApp")
        if bool(getattr(args, "onboard_camera", True)):
            setattr(args, "enable_cameras", True)
        simulation_app = ensure_simulation_app(args)

        set_phase("app_created")
        logger.log("[worker] SimulationApp created")

        logger.log(f"[worker] creating scene for {height_cm}cm")
        scene_handle = create_scene(
            config_from_args(args, height_cm),
            simulation_app=simulation_app,
            phase_callback=lambda name, details=None: set_phase(name, details),
        )

        vision_processor = OnboardCameraProcessor(scene_handle, vision_config_from_args(args))
        publish_status(ready=False, starting=True)
        logger.log("[worker] scene created")

        set_phase("adapter_initializing")
        adapter = SimRobotAdapter(scene_handle, create_adapter_config_from_args(args))
        telemetry_collector = create_telemetry_collector(args, scene_handle=scene_handle)
        if telemetry_collector is not None:
            adapter.attach_telemetry(telemetry_collector)
            run_dir = telemetry_collector.start_episode(
                adapter=adapter,
                scene_handle=scene_handle,
                obstacle_height_cm=height_cm,
                obstacle_height_m=obstacle_height_m(height_cm),
                sequence_label=f"{height_cm}cm worker session",
                source="subprocess_worker",
            )
            logger.log(f"[worker] telemetry enabled run_dir={run_dir}")
        set_phase("pre_grounding_started")
        set_phase("grounded_reference_initializing")
        ground_init = initialize_adapter_ground_reference(adapter)
        sim_time = float(getattr(adapter, "sim_time", sim_time))
        sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
        logger.log(f"[worker] grounded reference init: {ground_init}")
        set_phase("hidden_ground_settle_completed")
        set_phase("grounded_reference_saved", {"valid": bool(ground_init.get("grounded_reference_valid", False))})
        finalize_scene_after_grounding(scene_handle, phase_callback=lambda name, details=None: set_phase(name, details))
        set_phase("adapter_ready")
        publish_status(ready=True, starting=False)
        logger.log("[worker] SimRobotAdapter ready")
        if bool(args.worker_smoke_negative_knee_test):
            negative_result = run_negative_knee_smoke(adapter, logger)
            publish_status(ready=True, starting=False)
            if not negative_result["ok"]:
                raise RuntimeError("Negative knee smoke test failed")
        if bool(getattr(args, "worker_smoke_ground_structure", False)):
            ground_structure_result = run_ground_structure_smoke(args, scene_handle, adapter, logger)
            adapter.ground_structure_smoke_result = ground_structure_result
            publish_status(ready=True, starting=False)
            if not ground_structure_result["ok"]:
                raise RuntimeError("Ground structure smoke test failed")
        if bool(getattr(args, "worker_smoke_ground_calibration", False)):
            ground_calibration_result = run_ground_calibration_smoke(args, scene_handle, adapter, logger)
            adapter.ground_calibration_smoke_result = ground_calibration_result
            publish_status(ready=True, starting=False)
            if not ground_calibration_result["ok"]:
                raise RuntimeError("Ground calibration smoke test failed")
        if bool(getattr(args, "worker_smoke_camera_view_ground_contact", False)):
            camera_view_ground_result = run_camera_view_ground_contact_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.camera_view_ground_contact_smoke_result = camera_view_ground_result
            publish_status(ready=True, starting=False)
            if not camera_view_ground_result["ok"]:
                raise RuntimeError("Camera viewport ground-contact smoke test failed")
        if bool(getattr(args, "worker_smoke_camera_pose_ab", False)):
            camera_pose_result = run_camera_pose_ab_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.camera_pose_ab_smoke_result = camera_pose_result
            publish_status(ready=True, starting=False)
            if not camera_pose_result["ok"]:
                raise RuntimeError("Camera pose A/B smoke test failed")
        if bool(getattr(args, "worker_smoke_camera_counterfactual", False)):
            camera_counterfactual_result = run_camera_counterfactual_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.camera_counterfactual_smoke_result = camera_counterfactual_result
            publish_status(ready=True, starting=False)
            if not camera_counterfactual_result["ok"]:
                raise RuntimeError("Camera counterfactual smoke test failed")
        if bool(getattr(args, "worker_smoke_vision_playback", False)):
            vision_playback_result = run_vision_playback_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.vision_playback_smoke_result = vision_playback_result
            publish_status(ready=True, starting=False)
            if not vision_playback_result["ok"]:
                raise RuntimeError("Vision playback smoke test failed")
        if bool(getattr(args, "worker_smoke_camera_provenance", False)):
            camera_result = run_camera_provenance_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.camera_detection_smoke_result = camera_result
            publish_status(ready=True, starting=False)
            if not camera_result["ok"]:
                raise RuntimeError("Camera provenance smoke test failed")
        elif bool(getattr(args, "worker_smoke_camera_detection", False)):
            camera_result = run_camera_detection_smoke(args, scene_handle, adapter, vision_processor, logger)
            adapter.camera_detection_smoke_result = camera_result
            publish_status(ready=True, starting=False)
            if not camera_result["ok"]:
                raise RuntimeError("Camera detection smoke test failed")

        dt = float(scene_handle.sim.get_physics_dt())
        set_phase("running", publish=False)
        while not shutdown_requested and scene_handle.app_is_running():
            for message in ipc.poll():
                kind = str(message.get("type", ""))
                if kind == "shutdown":
                    shutdown_requested = True
                    break
                if kind == "command":
                    adapter.handle_command(_command_message_from_ipc(message))
                elif kind == "set_height":
                    height_cm = normalize_height_cm(message.get("height_cm", height_cm))
                    last_set_height_source = str(message.get("source", "ui") or "ui")
                    last_set_height_request_id = str(message.get("request_id", "") or "")
                    obstacle_revision += 1
                    logger.log(f"[worker] set_height {height_cm}cm source={last_set_height_source} revision={obstacle_revision}")
                    result = handle_set_height(
                        adapter=adapter,
                        scene_handle=scene_handle,
                        vision_processor=vision_processor,
                        height_cm=height_cm,
                        source=last_set_height_source,
                        request_id=last_set_height_request_id,
                        obstacle_revision=obstacle_revision,
                        respawn_policy=str(message.get("respawn_policy", "required") or "required"),
                    )
                    last_set_height_result = dict(result)
                    if telemetry_collector is not None:
                        telemetry_collector.update_obstacle_context(
                            height_cm=height_cm,
                            height_m=obstacle_height_m(height_cm),
                            source=last_set_height_source,
                            request_id=last_set_height_request_id,
                        )
                    sim_time = float(getattr(adapter, "sim_time", sim_time))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                    if bool(result.get("respawned", False)):
                        respawn_count += 1
                        last_respawn_at = time.time()
                    if not bool(result.get("ok", True)):
                        vision_control_error = str(result.get("error", ""))
                        logger.log(f"[worker] set_height ground failure: {vision_control_error}")
                elif kind == "respawn":
                    logger.log("[worker] respawn")
                    result = handle_respawn(adapter=adapter, vision_processor=vision_processor, reset_filter=True)
                    sim_time = float(getattr(adapter, "sim_time", sim_time))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                    if bool(result.get("respawned", False)):
                        respawn_count += 1
                        last_respawn_at = time.time()
                    if not bool(result.get("ok", True)):
                        vision_control_error = str(result.get("error", ""))
                        logger.log(f"[worker] respawn ground failure: {vision_control_error}")
                elif kind == "stop_wheels":
                    adapter.stop_wheels()
                elif kind == "restore_sim_state":
                    logger.log("[worker] restore_sim_state")
                    restore_count += 1
                    last_restore_at = time.time()
                    if hasattr(adapter, "restore_sim_state"):
                        try:
                            adapter.restore_sim_state(message.get("sim_state", {}))
                            adapter.stop_wheels()
                            if vision_processor is not None:
                                vision_processor.reset_filter()
                            last_restore_result = "ok"
                            last_restore_error = ""
                            publish_status(ready=True, starting=False)
                        except Exception as exc:
                            last_restore_result = "error"
                            last_restore_error = str(exc)
                            logger.log(f"[worker] restore_sim_state ERROR: {exc}")
                            publish_status(ready=True, starting=False, error=last_restore_error)
                    else:
                        last_restore_result = "unsupported"
                        last_restore_error = "Adapter does not support restore_sim_state."
                        publish_status(ready=True, starting=False, error=last_restore_error)
                elif kind == "request_state":
                    publish_status(ready=True, starting=False)
                elif kind == "vision_control":
                    action = str(message.get("action", ""))
                    payload = dict(message)
                    payload.pop("type", None)
                    payload.pop("action", None)
                    result = handle_vision_control(
                        args=args,
                        adapter=adapter,
                        scene_handle=scene_handle,
                        vision_processor=vision_processor,
                        action=action,
                        payload=payload,
                        logger_tail_provider=logger.tail,
                        runtime_metrics={"started_wall": started_wall},
                    )
                    if bool(result.get("respawned", False)):
                        sim_time = float(getattr(adapter, "sim_time", sim_time))
                        sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                        respawn_count += 1
                        last_respawn_at = time.time()
                    ok = bool(result.get("ok", False))
                    text = str(result.get("text", ""))
                    vision_control_error = "" if ok else text
                    logger.log(f"[worker] vision_control {action}: {text}")
                    publish_status(ready=True, starting=False)

            if shutdown_requested:
                break
            service_pending_viewport(
                args=args,
                adapter=adapter,
                scene_handle=scene_handle,
                vision_processor=vision_processor,
                logger_tail_provider=logger.tail,
                runtime_metrics={"started_wall": started_wall},
            )
            if not bool(args.no_continuous_sim_step):
                adapter.step(dt)
                sim_time = float(getattr(adapter, "sim_time", sim_time + dt))
                sim_steps = int(getattr(adapter, "sim_steps", sim_steps + 1))
                if vision_processor is not None:
                    vision_processor.update(dt=dt, sim_time=sim_time, wall_time=time.time())
            else:
                time.sleep(min(dt, 0.02))
                if vision_processor is not None:
                    vision_processor.update(dt=dt, sim_time=sim_time, wall_time=time.time())
            now = time.monotonic()
            if now - last_status_wall >= max(0.05, float(args.sim_status_refresh_ms) / 1000.0):
                publish_status(ready=True, starting=False)
                last_status_wall = now
            if smoke_deadline is not None and now >= smoke_deadline:
                logger.log("[worker] smoke test complete")
                break

        set_phase("shutdown", publish=False)
        publish_status(ready=False, starting=False)
        telemetry_finish_success = True
        telemetry_finish_reason = "worker shutdown"
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logger.log(f"[worker] ERROR: {exc}")
        logger.log(tb)
        telemetry_finish_reason = str(exc)
        try:
            ipc.send(make_message("error", phase=phase, error=str(exc), traceback=tb, ready=False, starting=False))
        except Exception:
            pass
        return 1
    finally:
        try:
            if telemetry_collector is not None:
                status = telemetry_collector.finish_episode(success=telemetry_finish_success, reason=telemetry_finish_reason)
                logger.log(f"[worker] telemetry saved: {status.get('run_dir', '')}")
        except Exception as exc:
            logger.log(f"[worker] telemetry finish failed: {exc}")
        try:
            if adapter is not None:
                adapter.stop_wheels()
                adapter.apply_commands_to_robot()
                adapter.robot.write_data_to_sim()
        except Exception:
            pass
        try:
            if scene_handle is not None:
                scene_handle.close()
            elif simulation_app is not None:
                simulation_app.close()
        except Exception:
            pass
        ipc.close()


def _startup_pose_trace_row(
    phase: str,
    *,
    scene_handle: Any | None,
    adapter: Any | None,
    sim_time: float,
    sim_steps: int,
    visible_rendered: bool,
) -> dict[str, Any]:
    ground_z = 0.0
    if scene_handle is not None:
        surface = getattr(scene_handle, "ground_surface_info", {}) or {}
        if isinstance(surface, dict):
            try:
                ground_z = float(surface.get("actual_ground_z_m", surface.get("configured_ground_z_m", 0.0)) or 0.0)
            except Exception:
                ground_z = 0.0
    root_z = None
    root_vertical_velocity = None
    if adapter is not None:
        try:
            sig = adapter.physics_signature()
            root_z = sig.get("root_z_m")
            root_velocity = sig.get("root_linear_velocity", [])
            if isinstance(root_velocity, list) and len(root_velocity) >= 3:
                root_vertical_velocity = float(root_velocity[2])
        except Exception:
            root_z = None
    return {
        "phase": str(phase),
        "timestamp": time.time(),
        "sim_time": float(sim_time),
        "sim_steps": int(sim_steps),
        "root_z_m": root_z,
        "root_vertical_velocity_m_s": root_vertical_velocity,
        "minimum_collision_z_m": None,
        "minimum_visual_z_m": None,
        "ground_z_m": float(ground_z),
        "visible_rendered": bool(visible_rendered),
    }


def build_status(
    args: argparse.Namespace,
    adapter: Any,
    *,
    phase: str,
    phase_started_wall: float,
    phase_history: list[dict[str, Any]],
    startup_pose_trace: list[dict[str, Any]],
    ready: bool,
    starting: bool,
    height_cm: int,
    sim_time: float,
    sim_steps: int,
    started_wall: float,
    error: str,
    tb: str,
    log_tail: list[str],
    respawn_count: int = 0,
    last_respawn_at: float = 0.0,
    restore_count: int = 0,
    last_restore_at: float = 0.0,
    last_restore_result: str = "",
    last_restore_error: str = "",
    obstacle_revision: int = 0,
    last_set_height_source: str = "",
    last_set_height_request_id: str = "",
    last_set_height_result: dict[str, Any] | None = None,
    scene_handle: Any | None = None,
    vision_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed_wall = max(1.0e-9, time.monotonic() - started_wall)
    phase_elapsed = max(0.0, time.monotonic() - float(phase_started_wall or started_wall))
    effective_livestream = int(getattr(args, "livestream", 0) or os.environ.get("LIVESTREAM", 0) or 0)
    effective_headless = bool(getattr(args, "headless", False)) or effective_livestream in {1, 2} or os.environ.get("HEADLESS", "0") == "1"
    effective_enable_cameras = bool(getattr(args, "enable_cameras", False)) or os.environ.get("ENABLE_CAMERAS", "0") == "1"
    eula_source = "cli" if bool(getattr(args, "accept_isaac_eula", False)) else ("environment" if os.environ.get("OMNI_KIT_ACCEPT_EULA") else "unset")
    status: dict[str, Any] = {
        "ready": bool(ready),
        "starting": bool(starting),
        "phase": str(phase),
        "phase_elapsed_s": phase_elapsed,
        "startup_phase_history": list(phase_history),
        "startup_pose_trace": list(startup_pose_trace),
        "height_cm": int(height_cm),
        "scene_height_cm": int(height_cm),
        "obstacle_revision": int(obstacle_revision),
        "last_set_height_source": str(last_set_height_source or ""),
        "last_set_height_request_id": str(last_set_height_request_id or ""),
        "last_set_height_result": dict(last_set_height_result or {}),
        "obstacle_updated": bool((last_set_height_result or {}).get("obstacle_updated", False)),
        "scene_ready": bool((last_set_height_result or {}).get("scene_ready", False)),
        "respawn_requested": bool((last_set_height_result or {}).get("respawn_requested", False)),
        "respawned": bool((last_set_height_result or {}).get("respawned", False)),
        "respawn_warning": str((last_set_height_result or {}).get("respawn_warning", "") or ""),
        "error": str(error or ""),
        "traceback": str(tb or ""),
        "pid": os.getpid(),
        "startup_elapsed_s": elapsed_wall,
        "sim_time": float(sim_time),
        "sim_steps": int(sim_steps),
        "real_time_factor": float(sim_time) / elapsed_wall,
        "latest_worker_log_tail": list(log_tail),
        "respawn_count": int(respawn_count),
        "last_respawn_at": float(last_respawn_at),
        "restore_count": int(restore_count),
        "last_restore_at": float(last_restore_at),
        "last_restore_result": str(last_restore_result or ""),
        "last_restore_error": str(last_restore_error or ""),
        "vision": dict(vision_status or default_vision_status(args)),
        "requested_headless": bool(getattr(args, "headless", False)),
        "effective_headless": bool(effective_headless),
        "requested_livestream": int(getattr(args, "livestream", 0) or 0),
        "effective_livestream": int(effective_livestream),
        "requested_enable_cameras": bool(getattr(args, "onboard_camera", True)),
        "effective_enable_cameras": bool(effective_enable_cameras),
        "selected_experience": str(getattr(args, "experience", "") or ""),
        "eula_source": eula_source,
        "launcher_environment": {
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "HEADLESS": os.environ.get("HEADLESS", ""),
            "LIVESTREAM": os.environ.get("LIVESTREAM", ""),
            "ENABLE_CAMERAS": os.environ.get("ENABLE_CAMERAS", ""),
            "OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", ""),
            "ISAAC_PATH": os.environ.get("ISAAC_PATH", ""),
            "EXP_PATH": os.environ.get("EXP_PATH", ""),
            "CARB_APP_PATH": os.environ.get("CARB_APP_PATH", ""),
        },
        "camera_ready": bool(scene_handle is not None and getattr(scene_handle, "camera", None) is not None and not getattr(scene_handle, "camera_error", "")),
        "camera_error": str(getattr(scene_handle, "camera_error", "") if scene_handle is not None else ""),
        "camera_parent_prim": str(getattr(scene_handle, "camera_parent_prim", "") if scene_handle is not None else ""),
        "camera_prim_path": str(getattr(scene_handle, "camera_prim_path", "") if scene_handle is not None else ""),
        "worker_config_file": str(getattr(args, "worker_config_file", "") or ""),
    }
    if adapter is not None:
        status.update(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle))
    else:
        status.update(build_common_worker_status(args=args, adapter=None, scene_handle=scene_handle))
    return enrich_runtime_readiness(status)


def default_vision_status(args: argparse.Namespace) -> dict[str, Any]:
    enabled = bool(getattr(args, "onboard_camera", True))
    return {
        "enabled": enabled,
        "camera_ready": False,
        "camera_parent_prim": "",
        "camera_prim_path": "",
        "frame_revision": 0,
        "detection_revision": 0,
        "last_frame_at": 0.0,
        "raw_height_cm": None,
        "detected_height_cm": None,
        "candidate_height_cm": None,
        "confidence": 0.0,
        "stable": False,
        "stable_count": 0,
        "stable_required": int(getattr(args, "vision_stable_frames", 5)),
        "valid_point_count": 0,
        "top_plane_mad_m": None,
        "quantization_error_cm": None,
        "method": "rgbd_geometry",
        "failure_reason": "camera unavailable" if enabled else "disabled",
        "debug_image_path": "",
        "debug_sidecar_path": "",
        "debug_folder": "",
        "camera_health": default_camera_health_status(reason="camera unavailable" if enabled else "disabled"),
        "height_validation": default_height_validation_result(reason="not checked"),
        "height_validation_summary": "Height Validation: NOT CHECKED",
        "camera_mount_validation": {"checked": False, "passed": False, "pending": False, "reasons": ["not checked"]},
        "camera_view": default_camera_viewport_status("not requested"),
        "camera_geometry": {"checked": False, "framing_state": "NOT_CHECKED", "reasons": ["camera unavailable" if enabled else "disabled"]},
        "camera_coverage": {"checked": False, "framing_state": "NOT_CHECKED", "reasons": ["camera unavailable" if enabled else "disabled"]},
        "ground_surface": {},
        "height_provenance": {
            "height_source": "isaac_rgbd_depth_geometry",
            "depth_data_type": "distance_to_image_plane",
            "intrinsics_used": False,
            "camera_world_pose_used": False,
            "expected_height_used_by_detector": False,
            "generated_height_used_by_detector": False,
            "scene_obstacle_height_used_by_detector": False,
            "usd_geometry_height_used_by_detector": False,
            "detector_input_audit_passed": False,
        },
        "height_measurement_evidence": {
            "height_source": "isaac_rgbd_depth_geometry",
            "depth_frame_revision": 0,
            "depth_timestamp": 0.0,
            "depth_fingerprint": "",
            "depth_shape": [],
            "depth_finite_ratio": 0.0,
            "intrinsics_fingerprint": "",
            "intrinsics_used": False,
            "camera_position_w": [],
            "camera_quaternion_w": [],
            "camera_world_pose_used": False,
            "roi_source": "generated_scene_x_prior",
            "obstacle_x_prior_used": False,
            "obstacle_x_prior_m": None,
            "ground_reference_source": "configured_world_ground_z",
            "ground_z_m": 0.0,
            "top_z_m": None,
            "raw_height_cm": None,
            "candidate_height_cm": None,
            "detected_height_cm": None,
            "top_point_count": 0,
            "obstacle_point_count": 0,
            "ground_point_count": 0,
            "expected_height_used_by_detector": False,
            "generated_height_used_by_detector": False,
            "scene_obstacle_height_used_by_detector": False,
            "usd_geometry_height_used_by_detector": False,
            "detector_input_audit_passed": False,
            "forbidden_input_reason": "",
        },
        "source_mode": "generated",
        "roi_source": "generated_scene_x_prior",
        "consecutive_errors": 0,
    }


def _command_message_from_ipc(message: dict[str, Any]) -> CommandMessage:
    return CommandMessage(
        text=str(message.get("command", "")),
        source=str(message.get("source", "ui")),
        playback_label=str(message.get("playback_label", "") or ""),
        playback_event_index=_optional_int(message.get("playback_event_index")),
        playback_event_count=int(message.get("playback_event_count", 0) or 0),
        playback_final_time_s=float(message.get("playback_final_time_s", 0.0) or 0.0),
        source_step=_optional_int(message.get("source_step")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Height replay Isaac worker process.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-config-file", "--worker_config_file", dest="worker_config_file", type=str, default="")
    parser.add_argument("--ipc-host", default="")
    parser.add_argument("--ipc-port", type=int, default=0)
    parser.add_argument("--worker-smoke-test-s", type=float, default=0.0)
    parser.add_argument("--worker-smoke-camera-detection", "--worker_smoke_camera_detection", dest="worker_smoke_camera_detection", action="store_true")
    parser.add_argument("--worker-smoke-camera-provenance", "--worker_smoke_camera_provenance", dest="worker_smoke_camera_provenance", action="store_true")
    parser.add_argument("--worker-smoke-camera-pose-ab", "--worker_smoke_camera_pose_ab", dest="worker_smoke_camera_pose_ab", action="store_true")
    parser.add_argument("--worker-smoke-camera-counterfactual", "--worker_smoke_camera_counterfactual", dest="worker_smoke_camera_counterfactual", action="store_true")
    parser.add_argument("--worker-smoke-camera-view-ground-contact", "--worker_smoke_camera_view_ground_contact", dest="worker_smoke_camera_view_ground_contact", action="store_true")
    parser.add_argument("--worker-smoke-ground-structure", "--worker_smoke_ground_structure", dest="worker_smoke_ground_structure", action="store_true")
    parser.add_argument("--worker-smoke-ground-calibration", "--worker_smoke_ground_calibration", dest="worker_smoke_ground_calibration", action="store_true")
    parser.add_argument("--worker-smoke-vision-playback", "--worker_smoke_vision_playback", dest="worker_smoke_vision_playback", action="store_true")
    parser.add_argument("--worker-smoke-camera-height-cm", "--worker_smoke_camera_height_cm", dest="worker_smoke_camera_height_cm", type=int, default=None)
    parser.add_argument("--worker-smoke-camera-validation-s", "--worker_smoke_camera_validation_s", dest="worker_smoke_camera_validation_s", type=float, default=10.0)
    parser.add_argument("--worker-smoke-camera-output", "--worker_smoke_camera_output", dest="worker_smoke_camera_output", type=str, default="")
    parser.add_argument("--worker-smoke-camera-counterfactual-output", "--worker_smoke_camera_counterfactual_output", dest="worker_smoke_camera_counterfactual_output", type=str, default="")
    parser.add_argument("--worker-smoke-output", "--worker_smoke_output", dest="worker_smoke_output", type=str, default="")
    parser.add_argument("--height-cm", "--height_cm", dest="height_cm", type=int, default=0)
    parser.add_argument("--robot-usd", "--robot_usd", dest="robot_usd", type=str, default=str(DEFAULT_ROBOT_USD_PATH))
    parser.add_argument("--save-usd", "--save_usd", dest="save_usd", type=str, default=str(DEFAULT_SCENE_SAVE_PATH))
    parser.add_argument("--save-scene", "--save_scene", dest="save_scene", action="store_true", default=True)
    parser.add_argument("--no-save-scene", dest="save_scene", action="store_false")
    parser.add_argument("--spawn-z", "--spawn_z", dest="spawn_z", type=float, default=0.04)
    parser.add_argument("--obstacle-x", "--obstacle_x", dest="obstacle_x", type=float, default=1.55)
    parser.add_argument("--obstacle-width", "--obstacle_width", dest="obstacle_width", type=float, default=None)
    parser.add_argument("--obstacle-length", "--obstacle_length", dest="obstacle_length", type=float, default=None)
    parser.add_argument("--infer-obstacle-size", "--infer_obstacle_size", dest="infer_obstacle_size", type=_bool_arg, nargs="?", const=True, default=True)
    parser.add_argument("--robot-width", "--robot_width", dest="robot_width", type=float, default=0.80)
    parser.add_argument("--robot-length", "--robot_length", dest="robot_length", type=float, default=0.55)
    parser.add_argument("--physics-dt", "--physics_dt", dest="physics_dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--render-interval", "--render_interval", dest="render_interval", type=int, default=2)
    parser.add_argument("--wheel-direction", "--wheel_direction", dest="wheel_direction", type=float, default=1.0)
    parser.add_argument("--max-wheel-speed-rad-s", "--max-wheel-speed", "--max_wheel_speed", dest="max_wheel_speed_rad_s", type=float, default=DEFAULT_MAX_WHEEL_SPEED_RAD_S)
    parser.add_argument("--default-wheel-speed-rad-s", "--default-wheel-speed", "--default_wheel_speed", dest="default_wheel_speed_rad_s", type=float, default=DEFAULT_MAX_WHEEL_SPEED_RAD_S * 0.25)
    parser.add_argument("--servo-stiffness", "--servo_stiffness", dest="servo_stiffness", type=float, default=600.0)
    parser.add_argument("--servo-damping", "--servo_damping", dest="servo_damping", type=float, default=60.0)
    parser.add_argument("--wheel-damping", "--wheel_damping", dest="wheel_damping", type=float, default=20.0)
    parser.add_argument("--onboard-camera", "--onboard_camera", dest="onboard_camera", action="store_true", default=True)
    parser.add_argument("--no-onboard-camera", "--no_onboard_camera", dest="onboard_camera", action="store_false")
    parser.add_argument("--camera-parent-prim", "--camera_parent_prim", dest="camera_parent_prim", type=str, default="")
    parser.add_argument("--camera-width", "--camera_width", dest="camera_width", type=int, default=424)
    parser.add_argument("--camera-height", "--camera_height", dest="camera_height", type=int, default=240)
    parser.add_argument("--camera-update-period-s", "--camera_update_period_s", dest="camera_update_period_s", type=float, default=0.10)
    parser.add_argument("--camera-offset-x", "--camera_offset_x", dest="camera_offset_x", type=float, default=0.35)
    parser.add_argument("--camera-offset-y", "--camera_offset_y", dest="camera_offset_y", type=float, default=0.0)
    parser.add_argument("--camera-offset-z", "--camera_offset_z", dest="camera_offset_z", type=float, default=0.18)
    parser.add_argument("--camera-pitch-deg", "--camera_pitch_deg", dest="camera_pitch_deg", type=float, default=14.0)
    parser.add_argument("--camera-aim-mode", "--camera_aim_mode", dest="camera_aim_mode", choices=["pitch", "look-at"], default="pitch")
    parser.add_argument("--camera-target-x", "--camera_target_x", dest="camera_target_x", type=float, default=1.55)
    parser.add_argument("--camera-target-y", "--camera_target_y", dest="camera_target_y", type=float, default=0.0)
    parser.add_argument("--camera-target-z", "--camera_target_z", dest="camera_target_z", type=float, default=0.02)
    parser.add_argument("--camera-target-frame", "--camera_target_frame", dest="camera_target_frame", choices=["world", "parent"], default="world")
    parser.add_argument("--camera-look-at-roll-deg", "--camera_look_at_roll_deg", dest="camera_look_at_roll_deg", type=float, default=0.0)
    parser.add_argument("--camera-coverage-strict", "--camera_coverage_strict", dest="camera_coverage_strict", action="store_true", default=False)
    parser.add_argument("--camera-focal-length", "--camera_focal_length", dest="camera_focal_length", type=float, default=24.0)
    parser.add_argument("--camera-horizontal-aperture", "--camera_horizontal_aperture", dest="camera_horizontal_aperture", type=float, default=20.955)
    parser.add_argument("--camera-near-clip-m", "--camera_near_clip_m", dest="camera_near_clip_m", type=float, default=0.05)
    parser.add_argument("--camera-far-clip-m", "--camera_far_clip_m", dest="camera_far_clip_m", type=float, default=6.0)
    parser.add_argument("--vision-confidence-threshold", "--vision_confidence_threshold", dest="vision_confidence_threshold", type=float, default=0.75)
    parser.add_argument("--vision-stable-frames", "--vision_stable_frames", dest="vision_stable_frames", type=int, default=5)
    parser.add_argument("--vision-window-size", "--vision_window_size", dest="vision_window_size", type=int, default=7)
    parser.add_argument("--vision-height-tolerance-cm", "--vision_height_tolerance_cm", dest="vision_height_tolerance_cm", type=float, default=2.0)
    parser.add_argument("--sim-status-refresh-ms", "--sim_status_refresh_ms", dest="sim_status_refresh_ms", type=int, default=250)
    parser.add_argument("--sim-worker-log-lines", "--sim_worker_log_lines", dest="sim_worker_log_lines", type=int, default=200)
    parser.add_argument("--worker-smoke-negative-knee-test", "--worker_smoke_negative_knee_test", dest="worker_smoke_negative_knee_test", action="store_true")
    parser.add_argument("--apply-safe-servo-joint-limits", "--apply_safe_servo_joint_limits", dest="apply_safe_servo_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-safe-servo-joint-limits", dest="apply_safe_servo_joint_limits", action="store_false")
    parser.add_argument("--apply-physx-joint-limits", "--apply_physx_joint_limits", dest="apply_physx_joint_limits", action="store_true")
    parser.add_argument("--no-continuous-sim-step", "--no_continuous_sim_step", dest="no_continuous_sim_step", action="store_true")
    parser.add_argument("--viewport-physics-guard", "--viewport_physics_guard", dest="viewport_physics_guard", action="store_true", default=True)
    parser.add_argument("--no-viewport-physics-guard", "--no_viewport_physics_guard", dest="viewport_physics_guard", action="store_false")
    parser.add_argument("--defer-first-visible-render", "--defer_first_visible_render", dest="defer_first_visible_render", action="store_true", default=True)
    parser.add_argument("--no-defer-first-visible-render", "--no_defer_first_visible_render", dest="defer_first_visible_render", action="store_false")
    parser.add_argument("--camera-view-active-fallback", "--camera_view_active_fallback", dest="camera_view_active_fallback", action="store_true")
    parser.add_argument("--camera-view-pending-timeout-s", "--camera_view_pending_timeout_s", dest="camera_view_pending_timeout_s", type=float, default=10.0)
    parser.add_argument("--camera-view-pending-max-retries", "--camera_view_pending_max_retries", dest="camera_view_pending_max_retries", type=int, default=30)
    parser.add_argument("--robot-ground-settle-s", "--robot_ground_settle_s", dest="robot_ground_settle_s", type=float, default=0.75)
    parser.add_argument("--robot-ground-settle-max-steps", "--robot_ground_settle_max_steps", dest="robot_ground_settle_max_steps", type=int, default=180)
    parser.add_argument("--robot-ground-stable-frames", "--robot_ground_stable_frames", dest="robot_ground_stable_frames", type=int, default=10)
    parser.add_argument("--robot-ground-vertical-speed-threshold-m-s", "--robot_ground_vertical_speed_threshold_m_s", dest="robot_ground_vertical_speed_threshold_m_s", type=float, default=0.01)
    parser.add_argument("--robot-ground-joint-speed-threshold-rad-s", "--robot_ground_joint_speed_threshold_rad_s", dest="robot_ground_joint_speed_threshold_rad_s", type=float, default=0.02)
    parser.add_argument("--robot-ground-servo-speed-threshold-rad-s", "--robot_ground_servo_speed_threshold_rad_s", dest="robot_ground_servo_speed_threshold_rad_s", type=float, default=None)
    parser.add_argument("--robot-ground-wheel-speed-threshold-rad-s", "--robot_ground_wheel_speed_threshold_rad_s", dest="robot_ground_wheel_speed_threshold_rad_s", type=float, default=0.20)
    parser.add_argument("--robot-ground-clearance-m", "--robot_ground_clearance_m", dest="robot_ground_clearance_m", type=float, default=0.002)
    parser.add_argument("--robot-ground-penetration-tolerance-m", "--robot_ground_penetration_tolerance_m", dest="robot_ground_penetration_tolerance_m", type=float, default=0.003)
    parser.add_argument("--robot-auto-ground-correction", "--robot_auto_ground_correction", dest="robot_auto_ground_correction", action="store_true")
    parser.add_argument("--robot-max-ground-correction-m", "--robot_max_ground_correction_m", dest="robot_max_ground_correction_m", type=float, default=0.10)
    if not add_app_launcher_args(parser):
        if not _parser_has_option(parser, "--device"):
            parser.add_argument("--device", type=str, default="cuda:0")
        if not _parser_has_option(parser, "--headless"):
            parser.add_argument("--headless", action="store_true")
        if not _parser_has_option(parser, "--livestream"):
            parser.add_argument("--livestream", type=int, default=0)
        if not _parser_has_option(parser, "--enable_cameras"):
            parser.add_argument("--enable_cameras", action="store_true")
        if not _parser_has_option(parser, "--experience"):
            parser.add_argument("--experience", type=str, default="")
    parser.add_argument("--accept-isaac-eula", "--accept_isaac_eula", dest="accept_isaac_eula", action="store_true", default=False)
    parser.add_argument("--no-accept-isaac-eula", "--no_accept_isaac_eula", dest="accept_isaac_eula", action="store_false")
    add_telemetry_args(parser)
    return parser


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_worker_config_file(args)
    args.max_wheel_speed_rad_s = abs(float(args.max_wheel_speed_rad_s))
    max_speed = float(args.max_wheel_speed_rad_s)
    args.default_wheel_speed_rad_s = max(-max_speed, min(max_speed, float(args.default_wheel_speed_rad_s)))
    args.camera_width = max(16, int(getattr(args, "camera_width", 424)))
    args.camera_height = max(16, int(getattr(args, "camera_height", 240)))
    args.camera_update_period_s = max(0.01, float(getattr(args, "camera_update_period_s", 0.10)))
    args.camera_near_clip_m = max(0.001, float(getattr(args, "camera_near_clip_m", 0.05)))
    args.camera_far_clip_m = max(args.camera_near_clip_m + 0.01, float(getattr(args, "camera_far_clip_m", 6.0)))
    args.vision_confidence_threshold = max(0.0, min(1.0, float(getattr(args, "vision_confidence_threshold", 0.75))))
    args.vision_stable_frames = max(1, int(getattr(args, "vision_stable_frames", 5)))
    args.vision_window_size = max(args.vision_stable_frames, int(getattr(args, "vision_window_size", 7)))
    args.vision_height_tolerance_cm = max(0.1, float(getattr(args, "vision_height_tolerance_cm", 2.0)))
    args.robot_ground_settle_s = max(0.0, float(getattr(args, "robot_ground_settle_s", 0.75)))
    args.robot_ground_settle_max_steps = max(1, int(getattr(args, "robot_ground_settle_max_steps", 180)))
    args.robot_ground_stable_frames = max(1, int(getattr(args, "robot_ground_stable_frames", 10)))
    args.robot_ground_vertical_speed_threshold_m_s = max(0.0, float(getattr(args, "robot_ground_vertical_speed_threshold_m_s", 0.01)))
    args.robot_ground_joint_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_joint_speed_threshold_rad_s", 0.02)))
    if getattr(args, "robot_ground_servo_speed_threshold_rad_s", None) is None:
        args.robot_ground_servo_speed_threshold_rad_s = args.robot_ground_joint_speed_threshold_rad_s
    args.robot_ground_servo_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_servo_speed_threshold_rad_s", args.robot_ground_joint_speed_threshold_rad_s)))
    args.robot_ground_wheel_speed_threshold_rad_s = max(0.0, float(getattr(args, "robot_ground_wheel_speed_threshold_rad_s", 0.20)))
    args.robot_ground_clearance_m = max(0.0, float(getattr(args, "robot_ground_clearance_m", 0.002)))
    args.robot_ground_penetration_tolerance_m = max(0.0, float(getattr(args, "robot_ground_penetration_tolerance_m", 0.003)))
    args.robot_max_ground_correction_m = max(0.0, float(getattr(args, "robot_max_ground_correction_m", 0.10)))
    args.camera_view_pending_timeout_s = max(0.05, float(getattr(args, "camera_view_pending_timeout_s", 10.0)))
    args.camera_view_pending_max_retries = max(1, int(getattr(args, "camera_view_pending_max_retries", 30)))
    telemetry_config = load_telemetry_config(args)
    args.telemetry_runtime_config = telemetry_config
    args.telemetry_effective_enabled = bool(telemetry_config.telemetry.enabled)
    args.live_viz_effective_enabled = bool(telemetry_config.visualization.live_enabled)
    args.telemetry_report_effective_enabled = bool(telemetry_config.telemetry.report_on_finish)
    args.equilibrium_region_effective_enabled = bool(telemetry_config.stability.equilibrium_enabled)
    args.telemetry_contact_sensors_enabled = bool(telemetry_config.telemetry.enabled and telemetry_config.telemetry.enable_contact_sensor)
    if getattr(args, "worker_smoke_camera_height_cm", None) is not None:
        args.worker_smoke_camera_height_cm = normalize_height_cm(args.worker_smoke_camera_height_cm)
    args.worker_smoke_camera_validation_s = max(1.0, float(getattr(args, "worker_smoke_camera_validation_s", 10.0)))
    if bool(getattr(args, "onboard_camera", True)):
        setattr(args, "enable_cameras", True)
    else:
        setattr(args, "enable_cameras", False)
    livestream = max(0, int(getattr(args, "livestream", 0) or 0))
    setattr(args, "livestream", livestream)
    os.environ["HEADLESS"] = "1" if bool(getattr(args, "headless", False)) or livestream in {1, 2} else "0"
    os.environ["LIVESTREAM"] = str(livestream)
    os.environ["ENABLE_CAMERAS"] = "1" if bool(getattr(args, "enable_cameras", False)) else "0"
    if bool(getattr(args, "accept_isaac_eula", False)):
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return run_worker(args)


def load_worker_config_file(args: argparse.Namespace) -> None:
    config_path = str(getattr(args, "worker_config_file", "") or "").strip()
    if not config_path:
        return
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("args", data)
    if not isinstance(values, dict):
        raise ValueError(f"Worker config file does not contain an args object: {path}")
    for key, value in values.items():
        setattr(args, key, value)
    setattr(args, "worker_config_file", str(path))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
