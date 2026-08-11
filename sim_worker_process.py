"""Standalone Isaac Sim worker process for height replay.

This script is launched by the Tk UI subprocess client. It owns SimulationApp,
scene creation, SimRobotAdapter, and the continuous sim loop in its main thread.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from command_model import JOINT_COMMAND_SIGN, KNEE_JOINT_NAMES, CommandMessage
from height_manifest import (
    SUPPORTED_HEIGHTS_MM,
    legacy_cm_to_mm,
    normalize_height_mm,
    obstacle_height_m_mm,
)
from playback import SimTimePlaybackService, playback_plan_from_payload, validate_plan_integrity
from motion_speed import load_motion_reference
from sim_ipc_protocol import JsonLineBuffer, encode_message, make_message
from sim_obstacle_scene import (
    DEFAULT_ROBOT_USD_PATH,
    DEFAULT_SCENE_SAVE_PATH,
    OBSTACLE_LENGTH_M,
    OBSTACLE_WIDTH_M,
    SimSceneConfig,
    add_app_launcher_args,
    create_scene,
    ensure_simulation_app,
    finalize_scene_after_grounding,
    measure_obstacle_geometry,
    measure_scene_baseline,
)
from sim_robot_adapter import SimRobotAdapter
from sim_state_validation import validate_full_sim_pose_state, verify_restored_full_sim_pose
from sim_worker_runtime import (
    build_common_worker_status,
    create_adapter_config_from_args,
    enrich_runtime_readiness,
    handle_respawn,
    handle_set_height,
    initialize_adapter_ground_reference,
)
from robot_ground_diagnostics import GROUND_OK, default_robot_ground_diagnostics, motion_status_from_worker_status, respawn_status_from_worker_status


class WorkerIpc:
    def __init__(self, host: str, port: int):
        self.host = str(host or "")
        self.port = int(port or 0)
        self.sock: socket.socket | None = None
        self.buffer = JsonLineBuffer()
        self.outbound: deque[dict[str, Any]] = deque()
        self.current_outbound: dict[str, Any] | None = None
        self.max_backlog = 32
        self.status_replaced = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self.status_enqueued = 0
        self.first_status_wall = 0.0
        self.last_status_wall = 0.0
        self.send_call_max_ms = 0.0
        self.send_call_total_ms = 0.0
        self.send_call_count = 0
        if self.host and self.port:
            self.sock = socket.create_connection((self.host, self.port), timeout=10.0)
            self.sock.setblocking(False)

    def send(self, message: dict[str, Any]) -> None:
        call_started = time.perf_counter()
        payload = encode_message(message)
        if self.sock is None:
            sys.stdout.buffer.write(payload)
            sys.stdout.flush()
            return
        kind = str(message.get("type", "") or "")
        if kind == "status":
            now = time.monotonic()
            self.first_status_wall = self.first_status_wall or now
            self.last_status_wall = now
            self.status_enqueued += 1
        critical = kind in {"error", "stop_ack", "operation_ack", "save_result"}
        if kind == "status":
            if (
                self.current_outbound is not None
                and self.current_outbound["kind"] == "status"
                and int(self.current_outbound["offset"]) == 0
            ):
                self.current_outbound = None
                self.status_replaced += 1
            kept = deque(row for row in self.outbound if row["kind"] != "status")
            self.status_replaced += len(self.outbound) - len(kept)
            self.outbound = kept
        if len(self.outbound) >= self.max_backlog and not critical:
            self.status_replaced += 1
            return
        if critical and len(self.outbound) >= self.max_backlog:
            for index, row in enumerate(self.outbound):
                if not row["critical"]:
                    del self.outbound[index]
                    break
        self.outbound.append({"kind": kind, "critical": critical, "payload": payload, "offset": 0})
        self.flush()
        elapsed_ms = (time.perf_counter() - call_started) * 1000.0
        self.send_call_count += 1
        self.send_call_total_ms += elapsed_ms
        self.send_call_max_ms = max(self.send_call_max_ms, elapsed_ms)

    def flush(self) -> None:
        if self.sock is None:
            return
        while self.current_outbound is not None or self.outbound:
            if self.current_outbound is None:
                self.current_outbound = self.outbound.popleft()
            row = self.current_outbound
            try:
                sent = self.sock.send(row["payload"][int(row["offset"]) :])
            except BlockingIOError:
                return
            if sent <= 0:
                return
            row["offset"] = int(row["offset"]) + int(sent)
            self.bytes_sent += int(sent)
            if int(row["offset"]) >= len(row["payload"]):
                self.frames_sent += 1
                self.current_outbound = None

    def poll(self) -> list[dict[str, Any]]:
        if self.sock is None:
            return []
        self.flush()
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

    def status(self) -> dict[str, Any]:
        duration = max(1.0e-9, self.last_status_wall - self.first_status_wall) if self.status_enqueued > 1 else 0.0
        return {
            "outbound_backlog": len(self.outbound) + (1 if self.current_outbound is not None else 0),
            "status_replaced": self.status_replaced,
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "status_enqueued": self.status_enqueued,
            "status_send_hz": (self.status_enqueued - 1) / duration if duration > 0.0 else 0.0,
            "socket_send_blocking_ms": 0.0,
            "send_call_average_ms": self.send_call_total_ms / max(1, self.send_call_count),
            "send_call_max_ms": self.send_call_max_ms,
        }

    def start_runtime_status_window(self) -> None:
        """Exclude startup phase bursts from the formal heartbeat-rate metric."""

        self.status_enqueued = 0
        self.first_status_wall = 0.0
        self.last_status_wall = 0.0

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


def _pose_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Bounded restore evidence; intentionally excludes unrelated diagnostics."""

    return {
        "capture_source": str(state.get("capture_source", "") or ""),
        "root_pose": copy.deepcopy(state.get("root_pose")),
        "root_velocity": copy.deepcopy(state.get("root_velocity")),
        "joint_pos": copy.deepcopy(state.get("joint_pos")),
        "joint_vel": copy.deepcopy(state.get("joint_vel")),
        "joint_names": list(state.get("joint_names", []) or []),
        "command_state": copy.deepcopy(state.get("command_state", {})),
        "actual_joint_state": copy.deepcopy(state.get("actual_joint_state", {})),
        "adapter_sim_time": state.get("adapter_sim_time"),
        "adapter_sim_steps": state.get("adapter_sim_steps"),
    }


def config_from_args(args: argparse.Namespace, height_mm: int) -> SimSceneConfig:
    return SimSceneConfig(
        obstacle_height_m=obstacle_height_m_mm(height_mm),
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
        telemetry_contact_sensors_enabled=bool(getattr(args, "telemetry_contact_sensors_enabled", False)),
        defer_first_visible_render=bool(getattr(args, "defer_first_visible_render", True)),
    )


def run_negative_knee_smoke(adapter: SimRobotAdapter, logger: WorkerLogger) -> dict[str, Any]:
    knees = [name for name in adapter.robot.joint_names if name in KNEE_JOINT_NAMES]
    missing = sorted(KNEE_JOINT_NAMES.difference(knees))
    if missing:
        raise RuntimeError(f"Negative knee smoke missing articulation joints: {missing}; available={adapter.robot.joint_names}")
    saved_state = adapter.capture_sim_state()
    lifted_state = copy.deepcopy(saved_state)
    root_pose = lifted_state.get("root_pose")
    if not isinstance(root_pose, list) or not root_pose or len(root_pose[0]) < 3:
        raise RuntimeError("Negative knee smoke could not capture a liftable articulation root pose")
    root_pose[0][2] = float(root_pose[0][2]) + 0.50
    lifted_state["root_velocity"] = [[0.0] * 6]
    adapter.restore_sim_state(lifted_state)
    logger.log("[worker] negative knee smoke: safe no-contact root lift +0.50m")
    dt = float(adapter.sim.get_physics_dt())
    rows: list[dict[str, Any]] = []
    ok = True
    for name in knees:
        for other in knees:
            adapter.handle_command(CommandMessage(text=f"servo {other} 0", source="worker_smoke"))
        for requested_deg in (0.0, -30.0, -60.0, 0.0):
            adapter.handle_command(CommandMessage(text=f"servo {name} {requested_deg:g}", source="worker_smoke"))
            stable_frames = 0
            measured_actual_deg: float | None = None
            measured_command_deg: float | None = None
            steps_used = 0
            for step_index in range(480):
                adapter.step(dt)
                steps_used = step_index + 1
                measured_actual = adapter.get_actual_joint_state().get("servos", {}).get(name, {}).get("deg")
                measured_actual_deg = None if measured_actual is None else float(measured_actual)
                if measured_actual_deg is None:
                    stable_frames = 0
                    continue
                measured_command_deg = (measured_actual_deg - float(adapter.standing_pose_deg[name])) / float(JOINT_COMMAND_SIGN[name])
                if abs(measured_command_deg - requested_deg) <= 0.5:
                    stable_frames += 1
                    if stable_frames >= 10:
                        break
                else:
                    stable_frames = 0
            diagnostics = {row["joint_name"]: row for row in adapter.get_joint_diagnostics()}
            diagnostic = diagnostics[name]
            worker_command_deg = float(adapter.servo_target_debug[name]["command_deg"])
            error_deg = None if measured_command_deg is None else measured_command_deg - requested_deg
            stage_pass = bool(
                worker_command_deg == requested_deg
                and diagnostic.get("target_inside_current_limit") is True
                and error_deg is not None
                and abs(error_deg) <= 1.0
            )
            row = {
                "test": "safe_no_contact",
                "joint_name": name,
                "requested_command_deg": requested_deg,
                "worker_command_deg": worker_command_deg,
                "target_actual_deg": float(adapter.servo_target_debug[name]["target_actual_deg"]),
                "measured_actual_deg": measured_actual_deg,
                "measured_command_deg": measured_command_deg,
                "steady_state_error_deg": error_deg,
                "target_inside_limit": diagnostic.get("target_inside_current_limit", "unknown"),
                "runtime_limit_deg": diagnostic.get("current_physx_or_usd_limit_deg"),
                "runtime_limit_rad": diagnostic.get("current_physx_or_usd_limit_rad"),
                "limit_source": diagnostic.get("current_limit_source"),
                "limit_write": diagnostic.get("safe_limit_record", {}),
                "physics_steps": steps_used,
                "pass": stage_pass,
            }
            rows.append(row)
            ok = ok and stage_pass
            logger.log(f"[worker] negative knee smoke {name} {requested_deg:g}deg: {'PASS' if stage_pass else 'FAIL'} {row}")

    adapter.restore_sim_state(saved_state)
    normal_rows: list[dict[str, Any]] = []
    for name in knees:
        adapter.handle_command(CommandMessage(text=f"servo {name} -60", source="worker_smoke_normal_scene"))
        for _index in range(240):
            adapter.step(dt)
        measured_actual = adapter.get_actual_joint_state().get("servos", {}).get(name, {}).get("deg")
        measured_actual_deg = None if measured_actual is None else float(measured_actual)
        measured_command_deg = None if measured_actual_deg is None else (
            measured_actual_deg - float(adapter.standing_pose_deg[name])
        ) / float(JOINT_COMMAND_SIGN[name])
        normal_rows.append(
            {
                "test": "normal_obstacle_scene",
                "joint_name": name,
                "requested_command_deg": -60.0,
                "worker_command_deg": float(adapter.servo_target_debug[name]["command_deg"]),
                "measured_command_deg": measured_command_deg,
                "steady_state_error_deg": None if measured_command_deg is None else measured_command_deg + 60.0,
                "ground_classification": str(getattr(adapter, "robot_ground_diagnostics", {}).get("classification", "")),
            }
        )
        adapter.handle_command(CommandMessage(text=f"servo {name} 0", source="worker_smoke_normal_scene"))
    adapter.restore_sim_state(saved_state)
    # The smoke flag is used only by the dedicated evidence run.  Leave all
    # four knees at the verified -60 command long enough for the visible GUI
    # screenshot; the worker is shut down immediately after capture.
    for name in knees:
        adapter.handle_command(CommandMessage(text=f"servo {name} -60", source="worker_smoke_screenshot"))
    for _index in range(240):
        adapter.step(dt)
    screenshot_rows: list[dict[str, Any]] = []
    for name in knees:
        measured_actual = adapter.get_actual_joint_state().get("servos", {}).get(name, {}).get("deg")
        measured_actual_deg = None if measured_actual is None else float(measured_actual)
        measured_command_deg = None if measured_actual_deg is None else (
            measured_actual_deg - float(adapter.standing_pose_deg[name])
        ) / float(JOINT_COMMAND_SIGN[name])
        screenshot_rows.append(
            {
                "joint_name": name,
                "requested_command_deg": -60.0,
                "worker_command_deg": float(adapter.servo_target_debug[name]["command_deg"]),
                "measured_command_deg": measured_command_deg,
                "steady_state_error_deg": None if measured_command_deg is None else measured_command_deg + 60.0,
            }
        )
    result = {
        "ok": ok,
        "command_chain": [0.0, -30.0, -60.0, 0.0],
        "rows": rows,
        "normal_scene_rows": normal_rows,
        "visible_screenshot_pose": screenshot_rows,
    }
    adapter.negative_knee_smoke_result = result
    logger.log(f"[worker] negative knee smoke result: {'PASS' if ok else 'FAIL'}")
    return result










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
    if explicit:
        path = Path(explicit)
    else:
        output_dir = Path(__file__).resolve().parent / "reports" / "worker_smoke"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    logger.log(f"[worker] {mode} smoke wrote {path}")
    return path






def run_worker(args: argparse.Namespace) -> int:
    ipc = WorkerIpc(args.ipc_host, int(args.ipc_port))
    logger = WorkerLogger(args.sim_worker_log_lines)
    height_mm = normalize_height_mm(args.height_mm)
    phase = "process_started"
    started_wall = time.monotonic()
    phase_started_wall = started_wall
    phase_history: list[dict[str, Any]] = []
    startup_pose_trace: list[dict[str, Any]] = []
    simulation_app = None
    scene_handle = None
    adapter = None
    sim_time = 0.0
    sim_steps = 0
    last_status_wall = 0.0
    running_started_wall = 0.0
    running_started_sim = 0.0
    rtf_last_wall = 0.0
    rtf_last_sim = 0.0
    rtf_window: deque[float] = deque(maxlen=24)
    respawn_count = 0
    last_respawn_at = 0.0
    restore_count = 0
    last_restore_at = 0.0
    last_restore_result = ""
    last_restore_error = ""
    last_restore_request_id = ""
    last_restore_verification: dict[str, Any] = {}
    obstacle_revision = 0
    last_set_height_source = "startup"
    last_set_height_request_id = ""
    last_set_height_result: dict[str, Any] = {}
    shutdown_requested = False
    worker_error = ""
    smoke_deadline = started_wall + float(args.worker_smoke_test_s) if float(args.worker_smoke_test_s) > 0 else None
    playback_service = SimTimePlaybackService()
    worker_session_id = uuid.uuid4().hex

    def publish_status(
        *,
        ready: bool,
        starting: bool,
        error: str = "",
        tb: str = "",
        detailed: bool = False,
        state_capture_request_id: str = "",
        state_capture_purpose: str = "",
    ) -> None:
        nonlocal rtf_last_wall, rtf_last_sim
        published_wall = time.monotonic()
        if phase == "running" and rtf_last_wall > 0.0:
            wall_delta = published_wall - rtf_last_wall
            sim_delta = float(sim_time) - float(rtf_last_sim)
            if wall_delta > 0.0 and sim_delta > 0.0:
                rtf_window.append(max(0.0, sim_delta / wall_delta))
        if phase == "running":
            rtf_last_wall = published_wall
            rtf_last_sim = float(sim_time)
        status = build_status(
            args,
            adapter,
            phase=phase,
            phase_started_wall=phase_started_wall,
            phase_history=phase_history,
            startup_pose_trace=startup_pose_trace,
            ready=ready,
            starting=starting,
            height_mm=height_mm,
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
            last_restore_request_id=last_restore_request_id,
            obstacle_revision=obstacle_revision,
            last_set_height_source=last_set_height_source,
            last_set_height_request_id=last_set_height_request_id,
            last_set_height_result=last_set_height_result,
            scene_handle=scene_handle,
            detailed=detailed,
            ipc_status=ipc.status(),
            running_started_wall=running_started_wall,
            running_started_sim=running_started_sim,
        )
        status["real_time_factor_lifetime"] = float(status.get("real_time_factor", 0.0) or 0.0)
        if rtf_window:
            ordered_rtf = sorted(rtf_window)
            status["real_time_factor"] = float(ordered_rtf[len(ordered_rtf) // 2])
        status["real_time_factor_window_samples"] = len(rtf_window)
        status["worker_playback"] = playback_service.status_dict(
            current_sim_time_s=sim_time,
            current_wall_time_s=time.time(),
            compact=not detailed,
        )
        status["worker_session_id"] = worker_session_id
        status["last_restore_verification"] = copy.deepcopy(last_restore_verification)
        if detailed:
            expected_names = list(getattr(getattr(adapter, "robot", None), "joint_names", []) or [])
            validation = validate_full_sim_pose_state(status.get("sim_state"), expected_names)
            status.update(
                state_capture_request_id=str(state_capture_request_id or ""),
                state_capture_purpose=str(state_capture_purpose or ""),
                state_capture_worker_session_id=worker_session_id,
                state_capture_sim_step=int(getattr(adapter, "sim_steps", sim_steps) or sim_steps),
                state_capture_sim_time=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                state_capture_validation=validation,
            )
        envelope = make_message("status", **status)
        envelope["status_payload_bytes"] = len(encode_message(envelope))
        ipc.send(envelope)

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
        logger.log(f"[worker] pid={os.getpid()} connected; initial height={height_mm}mm")

        set_phase("starting_app")
        logger.log("[worker] starting Isaac SimulationApp")
        simulation_app = ensure_simulation_app(args)

        set_phase("app_created")
        logger.log("[worker] SimulationApp created")

        logger.log(f"[worker] creating scene for {height_mm}mm")
        scene_handle = create_scene(
            config_from_args(args, height_mm),
            simulation_app=simulation_app,
            phase_callback=lambda name, details=None: set_phase(name, details),
        )

        publish_status(ready=False, starting=True)
        logger.log("[worker] scene created")

        set_phase("adapter_initializing")
        adapter = SimRobotAdapter(scene_handle, create_adapter_config_from_args(args))
        set_phase("pre_grounding_started")
        set_phase("grounded_reference_initializing")
        ground_init = initialize_adapter_ground_reference(adapter)
        sim_time = float(getattr(adapter, "sim_time", sim_time))
        sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
        logger.log(
            "[worker] grounded reference init: "
            + str(
                {
                    "grounded_reference_valid": ground_init.get(
                        "grounded_reference_valid"
                    ),
                    "stable": ground_init.get("stable"),
                    "steps_run": ground_init.get("steps_run"),
                    "step_budget": ground_init.get("step_budget"),
                    "consecutive_stable_ticks": ground_init.get(
                        "consecutive_stable_ticks"
                    ),
                    "stop_reason": ground_init.get("stop_reason"),
                    "classification": dict(
                        ground_init.get("grounded_reference_diagnostics", {})
                        or {}
                    ).get("classification"),
                }
            )
        )
        set_phase("hidden_ground_settle_completed")
        set_phase("grounded_reference_saved", {"valid": bool(ground_init.get("grounded_reference_valid", False))})
        finalize_scene_after_grounding(scene_handle, phase_callback=lambda name, details=None: set_phase(name, details))
        startup_geometry = measure_obstacle_geometry(scene_handle)
        obstacle_revision = 1
        last_set_height_result = {
            "accepted": bool(startup_geometry.get("prim_valid", False))
            and bool(startup_geometry.get("visual_valid", False))
            and bool(startup_geometry.get("collision_valid", False)),
            "old_height_mm": None,
            "requested_height_mm": height_mm,
            "measured_height_mm": float(startup_geometry.get("height_m", 0.0) or 0.0) * 1000.0,
            "measured_width_m": startup_geometry.get("width_m"),
            "measured_length_m": startup_geometry.get("length_m"),
            "measured_bounds": dict(startup_geometry.get("measured_bounds", {}) or {}),
            "visual_bounds": dict(startup_geometry.get("visual_bounds", {}) or {}),
            "collision_bounds": dict(startup_geometry.get("collision_bounds", {}) or {}),
            "visual_updated": bool(startup_geometry.get("visual_valid", False)),
            "collision_updated": bool(startup_geometry.get("collision_valid", False)),
            "prim_valid": bool(startup_geometry.get("prim_valid", False)),
            "prim_path": str(startup_geometry.get("prim_path", "/World/Obstacle")),
            "obstacle_revision": obstacle_revision,
            "control_ready": bool(ground_init.get("grounded_reference_valid", False)),
            "scene_ready": True,
            "update_mode": "startup_create_verified",
            "error": str(startup_geometry.get("error", "") or ""),
        }
        adapter.scene_baseline_metrics = {
            "measurement": "cached-lightweight-startup",
            "height_mm": height_mm,
            "obstacle_height_m": obstacle_height_m_mm(height_mm),
            "obstacle_width_m": float(scene_handle.config.obstacle_width),
            "obstacle_length_m": float(scene_handle.config.obstacle_length),
            "obstacle_front_face_x_m": float(scene_handle.config.obstacle_front_x),
            "obstacle_bottom_z_m": float(scene_handle.config.ground_z_m),
        }
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
        dt = float(scene_handle.sim.get_physics_dt())
        set_phase("running", publish=False)
        running_started_wall = time.monotonic()
        running_started_sim = float(sim_time)
        rtf_last_wall = running_started_wall
        rtf_last_sim = running_started_sim
        ipc.start_runtime_status_window()
        while not shutdown_requested and scene_handle.app_is_running():
            polled_messages = ipc.poll()
            # Safety stop is serviced before ordinary FIFO work.  The adapter's
            # generation check then rejects delayed non-zero wheel commands.
            for stop_message in [row for row in polled_messages if str(row.get("type", "")) == "stop_wheels"]:
                adapter.stop_wheels(
                    generation=_optional_int(stop_message.get("wheel_generation")),
                    command_id=str(stop_message.get("command_id", "") or ""),
                    requested_wall_time=float(stop_message.get("requested_wall_time", 0.0) or 0.0),
                    enqueued_wall_time=float(stop_message.get("enqueued_wall_time", 0.0) or 0.0),
                )
                adapter.apply_commands_to_robot()
                adapter.robot.write_data_to_sim()
                ipc.send(
                    make_message(
                        "stop_ack",
                        command_id=str(stop_message.get("command_id", "") or ""),
                        received_wall_time=float(adapter.wheel_command_status.get("received_wall_time", time.time())),
                        target_applied_wall_time=float(adapter.wheel_command_status.get("target_applied_wall_time", time.time())),
                        target_applied_sim_time=float(adapter.wheel_command_status.get("target_applied_sim_time", sim_time)),
                        zero_target_applied=True,
                        error="",
                    )
                )
                publish_status(ready=True, starting=False)
            for message in polled_messages:
                kind = str(message.get("type", ""))
                if kind == "stop_wheels":
                    continue
                if kind == "shutdown":
                    shutdown_requested = True
                    break
                if kind == "command":
                    adapter.handle_command(_command_message_from_ipc(message))
                elif kind == "apply_motion_batch":
                    ack = adapter.apply_motion_batch(message)
                    ipc.send(make_message("operation_ack", operation="apply_motion_batch", **ack))
                    publish_status(ready=True, starting=False)
                elif kind == "start_playback_plan":
                    request_id = str(message.get("request_id", "") or "")
                    requested_plan_id = str(message.get("plan_id", "") or "")
                    requested_sha = str(message.get("plan_sha256", "") or "")
                    plan_payload = dict(message.get("plan", {}) or {})
                    plan = playback_plan_from_payload(plan_payload)
                    integrity = validate_plan_integrity(
                        plan,
                        expected_plan_sha256=requested_sha,
                        expected_event_count=int(message.get("event_count", -1) or -1),
                        expected_segment_count=int(message.get("segment_count", -1) or -1),
                    )
                    rejection_reasons = list(integrity.get("errors", []) or [])
                    if not request_id:
                        rejection_reasons.append("missing playback_request_id")
                    if not requested_plan_id:
                        rejection_reasons.append("missing plan_id")
                    if playback_service.active:
                        rejection_reasons.append(
                            f"worker playback already active plan={playback_service.plan_id} request={playback_service.request_id}"
                        )
                    ok = not rejection_reasons
                    if ok:
                        plan.plan_sha256 = str(integrity["plan_sha256"])
                        ok = playback_service.start_plan(
                            plan,
                            current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                            current_wall_time_s=time.time(),
                            start_delay_sim_s=float(message.get("start_delay_sim_s", 0.0) or 0.0),
                            plan_id=requested_plan_id,
                            request_id=request_id,
                            worker_session_id=worker_session_id,
                        )
                        if not ok:
                            rejection_reasons.append(playback_service.last_error or "scheduler rejected plan")
                    rejection_reason = "; ".join(rejection_reasons)
                    ipc.send(
                        make_message(
                            "operation_ack",
                            operation="start_playback_plan",
                            request_id=request_id,
                            accepted=bool(ok),
                            rejection_reason=rejection_reason,
                            error=rejection_reason,
                            plan_id=requested_plan_id,
                            plan_sha256=str(integrity.get("plan_sha256", "") or ""),
                            profile=str(plan.profile or "raw"),
                            event_count=int(integrity.get("event_count", 0) or 0),
                            segment_count=int(integrity.get("segment_count", 0) or 0),
                            input_step_count=int(integrity.get("input_step_count", 0) or 0),
                            represented_step_indices=list(integrity.get("represented_step_indices", []) or []),
                            missing_required_step_indices=list(integrity.get("missing_required_step_indices", []) or []),
                            worker_session_id=worker_session_id,
                            accepted_wall_time=time.time(),
                        )
                    )
                    if not ok:
                        logger.log(f"[worker] playback plan rejected: {rejection_reason}")
                    else:
                        logger.log(
                            "[worker] playback plan scheduled "
                            f"id={playback_service.plan_id} events={len(plan.events)} "
                            f"final_time={plan.final_time_s:.3f}s sha={plan.plan_sha256}"
                        )
                    publish_status(ready=True, starting=False)
                elif kind == "pause_playback":
                    playback_service.pause(
                        current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                        adapter=adapter,
                    )
                    publish_status(ready=True, starting=False)
                elif kind == "resume_playback":
                    playback_service.resume(current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time))
                    publish_status(ready=True, starting=False)
                elif kind == "stop_playback":
                    playback_service.stop(
                        adapter,
                        current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                        current_wall_time_s=time.time(),
                        reason=str(message.get("reason", "stopped") or "stopped"),
                        stop_wheels=bool(message.get("stop_wheels", True)),
                    )
                    publish_status(ready=True, starting=False)
                elif kind in {"set_height", "set_height_respawn"}:
                    operation_started = time.perf_counter()
                    if message.get("height_mm") is not None:
                        requested_height_mm = normalize_height_mm(message.get("height_mm"))
                    else:
                        requested_height_mm = legacy_cm_to_mm(message.get("height_cm"))
                    last_set_height_source = str(message.get("source", "ui") or "ui")
                    last_set_height_request_id = str(message.get("request_id", "") or "")
                    requested_revision = int(message.get("requested_revision", obstacle_revision + 1) or (obstacle_revision + 1))
                    if requested_revision <= obstacle_revision:
                        requested_revision = obstacle_revision + 1
                    logger.log(
                        f"[worker] {kind} {requested_height_mm}mm source={last_set_height_source} "
                        f"requested_revision={requested_revision} current_revision={obstacle_revision}"
                    )
                    result = handle_set_height(
                        adapter=adapter,
                        scene_handle=scene_handle,
                        height_mm=requested_height_mm,
                        source=last_set_height_source,
                        request_id=last_set_height_request_id,
                        obstacle_revision=requested_revision,
                        respawn_policy="required" if kind == "set_height_respawn" else "never",
                    )
                    last_set_height_result = dict(result)
                    if bool(result.get("accepted", False)):
                        height_mm = requested_height_mm
                        obstacle_revision = requested_revision
                        adapter.scene_baseline_metrics.update(
                            height_mm=height_mm,
                            obstacle_height_m=obstacle_height_m_mm(height_mm),
                            obstacle_width_m=result.get("measured_width_m"),
                            obstacle_revision=obstacle_revision,
                            obstacle_bounds=result.get("measured_bounds", {}),
                        )
                    sim_time = float(getattr(adapter, "sim_time", sim_time))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                    if bool(result.get("respawned", False)):
                        respawn_count += 1
                        last_respawn_at = time.time()
                    if not bool(result.get("ok", True)):
                        worker_error = str(result.get("error", ""))
                        logger.log(f"[worker] set_height ground failure: {worker_error}")
                    ack_payload = {
                        **result,
                        "request_id": last_set_height_request_id,
                        "accepted": bool(result.get("accepted", False)),
                        "updated": bool(result.get("obstacle_updated", False)),
                        "height_mm": int(round(float(result.get("measured_height_mm", height_mm) or height_mm))),
                        "revision": obstacle_revision,
                        "obstacle_revision": obstacle_revision,
                        "control_ready": bool(result.get("control_ready", False)),
                        "worker_operation_s": time.perf_counter() - operation_started,
                        "request_enqueued_wall_time": float(message.get("enqueued_wall_time", 0.0) or 0.0),
                        "error": str(result.get("error", "") or ""),
                    }
                    ipc.send(make_message("operation_ack", operation=kind, **ack_payload))
                elif kind == "respawn":
                    logger.log("[worker] respawn")
                    result = handle_respawn(adapter=adapter)
                    sim_time = float(getattr(adapter, "sim_time", sim_time))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                    if bool(result.get("respawned", False)):
                        respawn_count += 1
                        last_respawn_at = time.time()
                    if not bool(result.get("ok", True)):
                        worker_error = str(result.get("error", ""))
                        logger.log(f"[worker] respawn ground failure: {worker_error}")
                    ipc.send(
                        make_message(
                            "operation_ack",
                            operation="respawn",
                            request_id=str(message.get("request_id", "") or ""),
                            respawned=bool(result.get("respawned", False)),
                            control_ready=bool(result.get("ok", False)),
                            error=str(result.get("error", "") or ""),
                        )
                    )
                elif kind == "recalibrate_ground_reference":
                    logger.log("[worker] explicit ground reference recalibration")
                    ground_result = initialize_adapter_ground_reference(adapter)
                    adapter.scene_baseline_metrics = measure_scene_baseline(scene_handle, adapter)
                    sim_time = float(getattr(adapter, "sim_time", sim_time))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
                    ipc.send(
                        make_message(
                            "operation_ack",
                            operation="recalibrate_ground_reference",
                            request_id=str(message.get("request_id", "") or ""),
                            control_ready=bool(ground_result.get("grounded_reference_valid", False)),
                            grounded_reference_valid=bool(ground_result.get("grounded_reference_valid", False)),
                            grounded_reference_stable=bool(ground_result.get("grounded_reference_stable", False)),
                            grounded_reference_physics_valid=bool(ground_result.get("grounded_reference_physics_valid", False)),
                            stable_frames=int(ground_result.get("stable_frames", 0) or 0),
                            stable_frames_required=int(ground_result.get("stable_frames_required", 0) or 0),
                            ground_reference_block_reason=str(
                                dict(ground_result.get("grounded_reference_diagnostics", {}) or {}).get(
                                    "ground_reference_block_reason", ""
                                )
                                or ""
                            ),
                            error=str(ground_result.get("error", "") or ""),
                        )
                    )
                elif kind == "restore_sim_state":
                    logger.log("[worker] restore_sim_state")
                    restore_count += 1
                    last_restore_at = time.time()
                    last_restore_request_id = str(message.get("request_id", "") or "")
                    last_restore_verification = {}
                    if hasattr(adapter, "restore_sim_state"):
                        try:
                            expected_state = dict(message.get("sim_state", {}) or {})
                            expected_names = list(getattr(adapter.robot, "joint_names", []) or [])
                            expected_validation = validate_full_sim_pose_state(expected_state, expected_names)
                            if not expected_validation["valid"]:
                                raise ValueError("restore source is not FULL_VALID: " + str(expected_validation["reason"]))
                            if playback_service.active:
                                playback_service.stop(
                                    adapter,
                                    current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                                    current_wall_time_s=time.time(),
                                    reason="restore_boundary",
                                    stop_wheels=True,
                                )
                            restore_result = dict(adapter.restore_sim_state(expected_state) or {})
                            adapter.stop_wheels()
                            adapter.apply_commands_to_robot()
                            adapter.robot.write_data_to_sim()
                            restore_trace = list(restore_result.get("trace", []) or [])
                            restore_trace.append(
                                {"event": "safe_wheel_boundary_applied", "sim_step": int(getattr(adapter, "sim_steps", sim_steps) or sim_steps)}
                            )
                            adapter.step(dt)
                            sim_time = float(getattr(adapter, "sim_time", sim_time + dt))
                            sim_steps = int(getattr(adapter, "sim_steps", sim_steps + 1))
                            restore_trace.append({"event": "physics_boundary_completed", "sim_step": sim_steps, "sim_time": sim_time})
                            measured_state = dict(adapter.capture_sim_state() or {})
                            verification = verify_restored_full_sim_pose(expected_state, measured_state, expected_names)
                            verification.update(
                                request_id=last_restore_request_id,
                                worker_session_id=worker_session_id,
                                worker_sim_step=sim_steps,
                                worker_sim_time=sim_time,
                                restore_trace=restore_trace,
                                expected_state_summary=_pose_state_summary(expected_state),
                                measured_state_summary=_pose_state_summary(measured_state),
                            )
                            last_restore_verification = verification
                            if not verification["verified"]:
                                raise RuntimeError("post-restore pose verification failed: " + str(verification["reason"]))
                            last_restore_result = "ok"
                            last_restore_error = ""
                            publish_status(ready=True, starting=False)
                        except Exception as exc:
                            last_restore_result = "error"
                            last_restore_error = str(exc)
                            if not last_restore_verification:
                                last_restore_verification = {
                                    "verified": False,
                                    "request_id": last_restore_request_id,
                                    "worker_session_id": worker_session_id,
                                    "reason": str(exc),
                                }
                            logger.log(f"[worker] restore_sim_state ERROR: {exc}")
                            publish_status(ready=True, starting=False, error=last_restore_error)
                    else:
                        last_restore_result = "unsupported"
                        last_restore_error = "Adapter does not support restore_sim_state."
                        publish_status(ready=True, starting=False, error=last_restore_error)
                elif kind == "request_state":
                    publish_status(
                        ready=True,
                        starting=False,
                        detailed=bool(message.get("detailed", False)),
                        state_capture_request_id=str(message.get("request_id", "") or ""),
                        state_capture_purpose=str(message.get("purpose", "") or ""),
                    )
            if shutdown_requested:
                break
            if not bool(args.no_continuous_sim_step):
                playback_service.update(
                    adapter,
                    current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                    current_sim_step=int(getattr(adapter, "sim_steps", sim_steps) or sim_steps),
                    current_wall_time_s=time.time(),
                )
                adapter.step(dt)
                sim_time = float(getattr(adapter, "sim_time", sim_time + dt))
                sim_steps = int(getattr(adapter, "sim_steps", sim_steps + 1))
            else:
                time.sleep(min(dt, 0.02))
            now = time.monotonic()
            if now - last_status_wall >= max(0.05, float(args.sim_status_refresh_ms) / 1000.0):
                publish_status(ready=True, starting=False)
                last_status_wall = now
            ipc.flush()
            if smoke_deadline is not None and now >= smoke_deadline:
                logger.log("[worker] smoke test complete")
                break

        set_phase("shutdown", publish=False)
        publish_status(ready=False, starting=False)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logger.log(f"[worker] ERROR: {exc}")
        logger.log(tb)
        try:
            ipc.send(make_message("error", phase=phase, error=str(exc), traceback=tb, ready=False, starting=False))
        except Exception:
            pass
        return 1
    finally:
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
    height_mm: int,
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
    last_restore_request_id: str = "",
    obstacle_revision: int = 0,
    last_set_height_source: str = "",
    last_set_height_request_id: str = "",
    last_set_height_result: dict[str, Any] | None = None,
    scene_handle: Any | None = None,
    detailed: bool = False,
    ipc_status: dict[str, Any] | None = None,
    running_started_wall: float = 0.0,
    running_started_sim: float = 0.0,
) -> dict[str, Any]:
    elapsed_wall = max(1.0e-9, time.monotonic() - started_wall)
    phase_elapsed = max(0.0, time.monotonic() - float(phase_started_wall or started_wall))
    if running_started_wall > 0.0:
        rtf_wall = max(1.0e-9, time.monotonic() - running_started_wall)
        rtf = max(0.0, float(sim_time) - float(running_started_sim)) / rtf_wall
    else:
        rtf = float(sim_time) / elapsed_wall
    result = dict(last_set_height_result or {})
    status: dict[str, Any] = {
        "ready": bool(ready),
        "runtime_ready": bool(ready),
        "starting": bool(starting),
        "phase": str(phase),
        "phase_elapsed_s": phase_elapsed,
        "height_mm": int(height_mm),
        "height_cm": float(height_mm) / 10.0,
        "scene_height_mm": int(height_mm),
        "obstacle_revision": int(obstacle_revision),
        "last_set_height_source": str(last_set_height_source or ""),
        "last_set_height_request_id": str(last_set_height_request_id or ""),
        "obstacle_updated": bool(result.get("obstacle_updated", False)),
        "measured_height_mm": result.get("measured_height_mm"),
        "measured_width_m": result.get("measured_width_m"),
        "measured_length_m": result.get("measured_length_m"),
        "measured_bounds": dict(result.get("measured_bounds", {}) or {}),
        "visual_updated": bool(result.get("visual_updated", False)),
        "collision_updated": bool(result.get("collision_updated", False)),
        "obstacle_prim_valid": bool(result.get("prim_valid", False)),
        "scene_ready": bool(result.get("scene_ready", ready)),
        "respawned": bool(result.get("respawned", False)),
        "error": str(error or ""),
        "traceback": str(tb or "") if error else "",
        "pid": os.getpid(),
        "sim_time": float(sim_time),
        "sim_steps": int(sim_steps),
        "real_time_factor": float(rtf),
        "respawn_count": int(respawn_count),
        "last_respawn_at": float(last_respawn_at),
        "restore_count": int(restore_count),
        "last_restore_result": str(last_restore_result or ""),
        "last_restore_error": str(last_restore_error or ""),
        "last_restore_request_id": str(last_restore_request_id or ""),
        "perf": dict(ipc_status or {}),
    }
    status.update(build_common_worker_status(args=args, adapter=adapter, scene_handle=scene_handle, detailed=detailed))
    if detailed:
        effective_livestream = int(getattr(args, "livestream", 0) or os.environ.get("LIVESTREAM", 0) or 0)
        effective_headless = bool(getattr(args, "headless", False)) or effective_livestream in {1, 2} or os.environ.get("HEADLESS", "0") == "1"
        status.update(
            detail_status=True,
            startup_elapsed_s=elapsed_wall,
            startup_phase_history=list(phase_history),
            startup_pose_trace=list(startup_pose_trace),
            latest_worker_log_tail=list(log_tail),
            requested_headless=bool(getattr(args, "headless", False)),
            effective_headless=effective_headless,
            requested_livestream=int(getattr(args, "livestream", 0) or 0),
            effective_livestream=effective_livestream,
            first_visible_render_completed=bool(scene_handle is not None and getattr(scene_handle, "first_visible_render_completed", False)),
            selected_experience=str(getattr(args, "experience", "") or ""),
            worker_config_file=str(getattr(args, "worker_config_file", "") or ""),
            last_set_height_result=result,
            last_restore_at=float(last_restore_at),
        )
    return enrich_runtime_readiness(status)




def _command_message_from_ipc(message: dict[str, Any]) -> CommandMessage:
    return CommandMessage(
        text=str(message.get("command", "")),
        source=str(message.get("source", "ui")),
        playback_label=str(message.get("playback_label", "") or ""),
        playback_event_index=_optional_int(message.get("playback_event_index")),
        playback_event_count=int(message.get("playback_event_count", 0) or 0),
        playback_final_time_s=float(message.get("playback_final_time_s", 0.0) or 0.0),
        source_step=_optional_int(message.get("source_step")),
        wheel_generation=_optional_int(message.get("wheel_generation")),
        command_id=str(message.get("command_id", "") or ""),
        high_priority=bool(message.get("high_priority", False)),
        requested_wall_time=float(message.get("requested_wall_time", 0.0) or 0.0),
        enqueued_wall_time=float(message.get("enqueued_wall_time", 0.0) or 0.0),
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
    parser.add_argument("--worker-smoke-ground-structure", "--worker_smoke_ground_structure", dest="worker_smoke_ground_structure", action="store_true")
    parser.add_argument("--worker-smoke-ground-calibration", "--worker_smoke_ground_calibration", dest="worker_smoke_ground_calibration", action="store_true")
    parser.add_argument("--worker-smoke-output", "--worker_smoke_output", dest="worker_smoke_output", type=str, default="")
    parser.add_argument("--height-mm", "--height_mm", dest="height_mm", type=int, default=50)
    parser.add_argument("--height-cm", "--height_cm", dest="height_cm", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--robot-usd", "--robot_usd", dest="robot_usd", type=str, default=str(DEFAULT_ROBOT_USD_PATH))
    parser.add_argument("--save-usd", "--save_usd", dest="save_usd", type=str, default=str(DEFAULT_SCENE_SAVE_PATH))
    parser.add_argument("--save-scene", "--save_scene", dest="save_scene", action="store_true", default=True)
    parser.add_argument("--no-save-scene", dest="save_scene", action="store_false")
    parser.add_argument("--spawn-z", "--spawn_z", dest="spawn_z", type=float, default=0.04)
    parser.add_argument("--obstacle-x", "--obstacle_x", dest="obstacle_x", type=float, default=1.55)
    parser.add_argument("--obstacle-width", "--obstacle_width", dest="obstacle_width", type=float, default=OBSTACLE_WIDTH_M)
    parser.add_argument("--obstacle-length", "--obstacle_length", dest="obstacle_length", type=float, default=OBSTACLE_LENGTH_M)
    parser.add_argument("--infer-obstacle-size", "--infer_obstacle_size", dest="infer_obstacle_size", type=_bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--robot-width", "--robot_width", dest="robot_width", type=float, default=0.80)
    parser.add_argument("--robot-length", "--robot_length", dest="robot_length", type=float, default=0.55)
    parser.add_argument("--physics-dt", "--physics_dt", dest="physics_dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--render-interval", "--render_interval", dest="render_interval", type=int, default=8)
    parser.add_argument("--wheel-direction", "--wheel_direction", dest="wheel_direction", type=float, default=1.0)
    parser.add_argument("--max-wheel-speed-rad-s", "--max-wheel-speed", "--max_wheel_speed", dest="max_wheel_speed_rad_s", type=float, default=load_motion_reference().wheel_velocity_limit_rad_s)
    parser.add_argument(
        "--default-wheel-speed-rad-s",
        "--default-wheel-speed",
        "--default_wheel_speed",
        dest="default_wheel_speed_rad_s",
        type=float,
        default=load_motion_reference().wheel_reference_velocity_rad_s,
    )
    parser.add_argument("--servo-stiffness", "--servo_stiffness", dest="servo_stiffness", type=float, default=600.0)
    parser.add_argument("--servo-damping", "--servo_damping", dest="servo_damping", type=float, default=60.0)
    parser.add_argument("--wheel-damping", "--wheel_damping", dest="wheel_damping", type=float, default=20.0)
    parser.add_argument("--sim-status-refresh-ms", "--sim_status_refresh_ms", dest="sim_status_refresh_ms", type=int, default=125)
    parser.add_argument("--sim-worker-log-lines", "--sim_worker_log_lines", dest="sim_worker_log_lines", type=int, default=200)
    parser.add_argument("--worker-smoke-negative-knee-test", "--worker_smoke_negative_knee_test", dest="worker_smoke_negative_knee_test", action="store_true")
    parser.add_argument("--apply-safe-servo-joint-limits", "--apply_safe_servo_joint_limits", dest="apply_safe_servo_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-safe-servo-joint-limits", dest="apply_safe_servo_joint_limits", action="store_false")
    parser.add_argument("--apply-physx-joint-limits", "--apply_physx_joint_limits", dest="apply_physx_joint_limits", action="store_true", default=True)
    parser.add_argument("--no-apply-physx-joint-limits", dest="apply_physx_joint_limits", action="store_false")
    parser.add_argument("--no-continuous-sim-step", "--no_continuous_sim_step", dest="no_continuous_sim_step", action="store_true")
    parser.add_argument("--defer-first-visible-render", "--defer_first_visible_render", dest="defer_first_visible_render", action="store_true", default=True)
    parser.add_argument("--no-defer-first-visible-render", "--no_defer_first_visible_render", dest="defer_first_visible_render", action="store_false")
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
        if not _parser_has_option(parser, "--experience"):
            parser.add_argument("--experience", type=str, default="")
    parser.add_argument("--accept-isaac-eula", "--accept_isaac_eula", dest="accept_isaac_eula", action="store_true", default=False)
    parser.add_argument("--no-accept-isaac-eula", "--no_accept_isaac_eula", dest="accept_isaac_eula", action="store_false")
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
    if getattr(args, "height_cm", None) is not None:
        args.height_mm = legacy_cm_to_mm(args.height_cm)
    args.height_mm = normalize_height_mm(args.height_mm)
    args.max_wheel_speed_rad_s = abs(float(args.max_wheel_speed_rad_s))
    max_speed = float(args.max_wheel_speed_rad_s)
    args.default_wheel_speed_rad_s = max(-max_speed, min(max_speed, float(args.default_wheel_speed_rad_s)))
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
    args.telemetry_effective_enabled = False
    args.live_viz_effective_enabled = False
    args.equilibrium_region_effective_enabled = False
    args.telemetry_contact_sensors_enabled = False
    livestream = max(0, int(getattr(args, "livestream", 0) or 0))
    setattr(args, "livestream", livestream)
    os.environ["HEADLESS"] = "1" if bool(getattr(args, "headless", False)) or livestream in {1, 2} else "0"
    os.environ["LIVESTREAM"] = str(livestream)
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
    known = set(vars(args))
    for key, value in values.items():
        if key in known:
            setattr(args, key, value)
    setattr(args, "worker_config_file", str(path))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
