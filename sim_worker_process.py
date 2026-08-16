"""Standalone Isaac Sim worker process for height replay.

This script is launched by the Tk UI subprocess client. It owns SimulationApp,
scene creation, SimRobotAdapter, and the continuous sim loop in its main thread.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.metadata
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
from sim_ipc_protocol import (
    MACRO_FAST_CLOSE_SCHEMA,
    JsonLineBuffer,
    encode_message,
    make_message,
)
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
    capture_worker_motion_start_readiness,
    create_adapter_config_from_args,
    enrich_runtime_readiness,
    handle_respawn,
    handle_set_height,
    initialize_adapter_ground_reference,
)
from robot_ground_diagnostics import GROUND_OK, default_robot_ground_diagnostics, motion_status_from_worker_status, respawn_status_from_worker_status
from fsm_50mm_recording_derived_v3.worker_recording_session import (
    WorkerRecordingSession,
    configure_scene_for_worker_recording,
    load_worker_recording_gate_request,
    validate_worker_plan_binding,
)
from fsm_50mm_recording_derived_v3.worker_task_replay_session import (
    WorkerTaskReplaySession,
    load_worker_task_replay_request,
    validate_worker_task_plan_binding,
)
from fsm_50mm_recording_derived_v3.worker_macro_fsm_session import (
    WorkerMacroFSMSession,
    load_worker_macro_fsm_request,
    validate_worker_macro_start_binding,
)
from fsm_50mm_recording_derived_v3.worker_residual_macro_fsm_session import (
    GateEResidualMacroFSMSession,
    GateEResidualMacroFSMSessionFactory,
    WorkerResidualMacroFSMRequest,
    load_worker_residual_macro_fsm_request,
    validate_worker_residual_start_binding,
)


@functools.lru_cache(maxsize=1)
def _worker_runtime_version() -> str:
    for distribution in ("isaacsim", "isaac-sim"):
        try:
            return str(importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unavailable"


def _worker_ack_identity(
    adapter: Any | None,
    *,
    worker_session_id: str,
    artifact_request_id: str,
    artifact_request: Any | None = None,
) -> dict[str, Any]:
    """Return the exact runtime identity required on every Gate-1 ACK."""

    identity = {
        "worker_pid": os.getpid(),
        "worker_session_id": str(worker_session_id or ""),
        "adapter_runtime_instance_id": str(
            getattr(adapter, "runtime_instance_id", "") or ""
        ),
        "artifact_request_id": str(artifact_request_id or ""),
        "root_state_write_count": int(
            getattr(adapter, "root_state_write_count", 0) or 0
        ),
    }
    if artifact_request is not None:
        identity.update(
            contact_mode=str(getattr(artifact_request, "contact_mode", "") or ""),
            environment_equivalence_role=str(
                getattr(artifact_request, "environment_equivalence_role", "") or ""
            ),
            diagnostic_role=str(
                getattr(artifact_request, "diagnostic_role", "") or ""
            ),
            qualification_scope=str(
                getattr(artifact_request, "qualification_scope", "") or ""
            ),
            gate1_eligible=bool(
                getattr(artifact_request, "gate1_eligible", False)
            ),
            gate1_physical_qualification_eligible=bool(
                getattr(
                    artifact_request,
                    "gate1_physical_qualification_eligible",
                    False,
                )
            ),
            environment_equivalence_eligible=bool(
                getattr(artifact_request, "environment_equivalence_eligible", False)
            ),
        )
    return identity


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
        critical = kind in {
            "error",
            "stop_ack",
            "operation_ack",
            "save_result",
            "artifact_complete",
            "artifact_failed",
            "task_replay_complete",
            "task_replay_failed",
            "macro_fsm_complete",
            "macro_fsm_failed",
            "close_requested",
            "close_returned",
        }
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

    def flush_until_empty(self, timeout_s: float = 0.5) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while self.current_outbound is not None or self.outbound:
            self.flush()
            if self.current_outbound is None and not self.outbound:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

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


def _wait_for_close_receipt(
    ipc: WorkerIpc,
    close_event: dict[str, Any],
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Wait until the controller confirms decoding one exact close event.

    Formal fast-close uses no worker-local timeout: the outer supervisor owns
    the sole 60-second process-tree timeout.  This prevents native Kit teardown
    from terminating the process before the controller has retained the
    shutdown ACK and the ordered ``close_requested`` event.
    """

    deadline = (
        None
        if timeout_s is None or float(timeout_s) <= 0.0
        else time.monotonic() + float(timeout_s)
    )
    expected = {
        "close_event_type": str(close_event.get("type", "") or ""),
        "request_id": str(close_event.get("request_id", "") or ""),
        "mode": str(close_event.get("mode", "") or ""),
        "accepted": close_event.get("accepted"),
        "error": str(close_event.get("error", "") or ""),
        "worker_pid": close_event.get("worker_pid"),
        "worker_session_id": str(
            close_event.get("worker_session_id", "") or ""
        ),
        "adapter_runtime_instance_id": str(
            close_event.get("adapter_runtime_instance_id", "") or ""
        ),
        "artifact_request_id": str(
            close_event.get("artifact_request_id", "") or ""
        ),
        "root_state_write_count": close_event.get("root_state_write_count"),
        "close_kwargs": dict(close_event.get("close_kwargs", {}) or {}),
        "runtime_version": str(close_event.get("runtime_version", "") or ""),
    }
    if "schema_version" in close_event:
        expected["schema_version"] = str(
            close_event.get("schema_version", "") or ""
        )
    if "task_replay_request_id" in close_event:
        expected["task_replay_request_id"] = str(
            close_event.get("task_replay_request_id", "") or ""
        )
    if "macro_fsm_request_id" in close_event:
        expected["macro_fsm_request_id"] = str(
            close_event.get("macro_fsm_request_id", "") or ""
        )
    if "residual_macro_fsm_request_id" in close_event:
        expected["residual_macro_fsm_request_id"] = str(
            close_event.get("residual_macro_fsm_request_id", "") or ""
        )
    if "base_macro_fsm_request_id" in close_event:
        expected["base_macro_fsm_request_id"] = str(
            close_event.get("base_macro_fsm_request_id", "") or ""
        )
    if "gate_e_zero_residual" in close_event:
        expected["gate_e_zero_residual"] = close_event.get(
            "gate_e_zero_residual"
        )
    while deadline is None or time.monotonic() <= deadline:
        for message in ipc.poll():
            if str(message.get("type", "") or "") != "close_receipt":
                continue
            if message.get("received") is not True:
                continue
            if all(
                key in message
                and _strict_json_value_equal(message[key], value)
                for key, value in expected.items()
            ):
                return dict(message)
        time.sleep(0.002)
    return {}


def _wait_for_task_exception_fast_shutdown(
    ipc: Any,
    *,
    task_session: Any,
    adapter: Any,
    worker_session_id: str,
) -> str:
    """Hold a failed task worker for the controller-owned fast-close request.

    A normal Kit close is known to stall in the supported Isaac version.  This
    path does not manufacture a receipt or close the app itself: it accepts one
    explicit fast shutdown only after the task result is durable and its video
    writer is quiesced.  The regular finally block then performs the existing
    close_requested/receipt/native-close/close_returned sequence.
    """

    if task_session is None or not bool(task_session.fast_close_ready):
        return ""
    while True:
        for message in ipc.poll():
            if str(message.get("type", "") or "") != "shutdown":
                continue
            request_id = str(message.get("request_id", "") or "")
            mode = str(message.get("mode", "") or "").strip().lower()
            if mode != "fast":
                ipc.send(
                    make_message(
                        "operation_ack",
                        operation="shutdown",
                        accepted=False,
                        request_id=request_id,
                        mode=mode or "normal",
                        error=(
                            "terminal task-replay exception requires the verified "
                            "fast-close receipt path"
                        ),
                    )
                )
                continue
            if not request_id:
                ipc.send(
                    make_message(
                        "operation_ack",
                        operation="shutdown",
                        accepted=False,
                        request_id="",
                        mode="fast",
                        error="fast shutdown request_id is required",
                    )
                )
                continue
            ipc.send(
                make_message(
                    "operation_ack",
                    operation="shutdown",
                    accepted=True,
                    request_id=request_id,
                    mode="fast",
                    **_worker_ack_identity(
                        adapter,
                        worker_session_id=worker_session_id,
                        artifact_request_id="",
                    ),
                    task_replay_request_id=task_session.request.request_id,
                    close_kwargs={
                        "wait_for_replicator": False,
                        "skip_cleanup": True,
                    },
                    runtime_version=_worker_runtime_version(),
                    error="",
                )
            )
            return request_id
        ipc.flush()
        time.sleep(0.002)


def _wait_for_macro_exception_fast_shutdown(
    ipc: Any,
    *,
    macro_session: Any,
    adapter: Any,
    worker_session_id: str,
) -> str:
    """Hold a terminal Macro worker for its controller-owned fast close."""

    if macro_session is None or not bool(macro_session.fast_close_ready):
        return ""
    while True:
        for message in ipc.poll():
            if str(message.get("type", "") or "") != "shutdown":
                continue
            request_id = str(message.get("request_id", "") or "")
            mode = str(message.get("mode", "") or "").strip().lower()
            if mode != "fast" or not request_id:
                ipc.send(
                    make_message(
                        "operation_ack",
                        operation="shutdown",
                        accepted=False,
                        request_id=request_id,
                        mode=mode or "normal",
                        error=(
                            "terminal Macro FSM requires a non-empty request_id "
                            "on the verified fast-close receipt path"
                        ),
                        **_macro_route_ack_identity(
                            macro_session, payload_role="shutdown_ack"
                        ),
                    )
                )
                continue
            ipc.send(
                make_message(
                    "operation_ack",
                    operation="shutdown",
                    accepted=True,
                    request_id=request_id,
                    mode="fast",
                    **_worker_ack_identity(
                        adapter,
                        worker_session_id=worker_session_id,
                        artifact_request_id="",
                    ),
                    **_macro_route_ack_identity(
                        macro_session, payload_role="shutdown_ack"
                    ),
                    close_kwargs={
                        "wait_for_replicator": False,
                        "skip_cleanup": True,
                    },
                    runtime_version=_worker_runtime_version(),
                    schema_version=MACRO_FAST_CLOSE_SCHEMA,
                    error="",
                )
            )
            return request_id
        ipc.flush()
        time.sleep(0.002)


def _finalize_active_task_session_for_worker_exit(
    task_session: Any,
    *,
    terminal_sent: bool,
    publish_terminal: Any,
    reason: str,
) -> bool:
    """Make an unexpected worker-loop exit durable before any Kit close."""

    if task_session is None or bool(terminal_sent):
        return bool(terminal_sent)
    resolved_reason = str(reason or "worker loop exited")
    app_stopped = "simulation app stopped" in resolved_reason.lower()
    publish_terminal(
        task_session.fail(
            resolved_reason,
            infrastructure_failure=True,
            simulation_app_stopped=app_stopped,
        )
    )
    return True


def _finalize_active_macro_session_for_worker_exit(
    macro_session: Any,
    *,
    terminal_sent: bool,
    publish_terminal: Any,
    reason: str,
) -> bool:
    """Make an unexpected Macro worker-loop exit durable before Kit close."""

    if macro_session is None or bool(terminal_sent):
        return bool(terminal_sent)
    terminal = macro_session.fail(
        str(reason or "worker loop exited"),
        infrastructure_failure=True,
        simulation_app_stopped=True,
    )
    if terminal is None:
        raise RuntimeError(
            "Macro worker exit did not produce a durable terminal result"
        )
    publish_terminal(terminal)
    return True


def _load_exclusive_fsm50_worker_requests(
    args: argparse.Namespace,
    *,
    gate_loader: Any | None = None,
    task_loader: Any | None = None,
    macro_loader: Any | None = None,
    residual_loader: Any | None = None,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Load exactly zero or one worker-owned FSM50 request route."""

    gate_fn = gate_loader or load_worker_recording_gate_request
    task_fn = task_loader or load_worker_task_replay_request
    macro_fn = macro_loader or load_worker_macro_fsm_request
    residual_fn = residual_loader or load_worker_residual_macro_fsm_request
    gate_request = gate_fn(getattr(args, "fsm50_gate_request_path", ""))
    task_request = task_fn(getattr(args, "fsm50_task_request_path", ""))
    macro_request = macro_fn(getattr(args, "fsm50_macro_request_path", ""))
    residual_request = residual_fn(
        getattr(args, "fsm50_residual_macro_request_path", "")
    )
    if sum(
        request is not None
        for request in (
            gate_request,
            task_request,
            macro_request,
            residual_request,
        )
    ) > 1:
        raise ValueError(
            "Gate-1 recording, normal task replay, Macro FSM, and Gate-E "
            "residual Macro FSM requests are mutually exclusive"
        )
    return gate_request, task_request, macro_request, residual_request


def _macro_route_ack_identity(macro_session: Any, *, payload_role: str) -> dict[str, Any]:
    residual_request = getattr(macro_session, "residual_request", None)
    if residual_request is None:
        return {"macro_fsm_request_id": macro_session.request.request_id}
    return {
        "macro_fsm_request_id": residual_request.request_id,
        "residual_macro_fsm_request_id": residual_request.request_id,
        "base_macro_fsm_request_id": macro_session.request.request_id,
        "gate_e_zero_residual": residual_request.gate_e_identity(
            payload_role=payload_role
        ),
    }


def _validate_macro_route_start_binding(
    macro_session: Any,
    residual_request: Any | None,
    message: dict[str, Any],
    *,
    expected_worker_session_id: str,
    macro_validator: Any | None = None,
    residual_validator: Any | None = None,
) -> list[str]:
    if residual_request is None:
        validator = macro_validator or validate_worker_macro_start_binding
        return list(
            validator(
                macro_session.request,
                message,
                expected_worker_session_id=expected_worker_session_id,
            )
        )
    if getattr(macro_session, "residual_request", None) is not residual_request:
        return ["Gate-E residual session/request route binding mismatch"]
    validator = residual_validator or validate_worker_residual_start_binding
    return list(
        validator(
            residual_request,
            message,
            expected_worker_session_id=expected_worker_session_id,
        )
    )


def _build_macro_start_operation_ack(
    *,
    message: dict[str, Any],
    accepted: bool,
    rejection_reason: str,
    adapter: Any,
    worker_session_id: str,
    start_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the shared-transport Macro start ACK without overlay collisions."""

    excluded = {
        "accepted",
        "type",
        "operation",
        "request_id",
        "rejection_reason",
        "error",
        "worker_pid",
        "worker_session_id",
        "adapter_runtime_instance_id",
        "artifact_request_id",
        "root_state_write_count",
    }
    return make_message(
        "operation_ack",
        operation="start_macro_fsm",
        request_id=str(message.get("request_id", "") or ""),
        accepted=bool(accepted),
        rejection_reason=str(rejection_reason or ""),
        error=str(rejection_reason or ""),
        **_worker_ack_identity(
            adapter,
            worker_session_id=worker_session_id,
            artifact_request_id="",
        ),
        **{
            key: value
            for key, value in start_payload.items()
            if key not in excluded
        },
    )


def _strict_json_value_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_json_value_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_value_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return bool(actual == expected)


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
    shutdown_mode = ""
    shutdown_request_id = ""
    worker_error = ""
    smoke_deadline = started_wall + float(args.worker_smoke_test_s) if float(args.worker_smoke_test_s) > 0 else None
    playback_service = SimTimePlaybackService()
    worker_session_id = uuid.uuid4().hex
    last_motion_start_readiness: dict[str, Any] = {}
    gate_request = None
    artifact_session: WorkerRecordingSession | None = None
    artifact_terminal_sent = False
    task_request = None
    task_session: WorkerTaskReplaySession | None = None
    task_terminal_sent = False
    macro_request = None
    residual_request: WorkerResidualMacroFSMRequest | None = None
    macro_session: WorkerMacroFSMSession | GateEResidualMacroFSMSession | None = None
    macro_terminal_sent = False

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
        status["runtime_version"] = _worker_runtime_version()
        status["worker_artifact_session"] = (
            {"enabled": False}
            if artifact_session is None
            else artifact_session.status_dict()
        )
        status["worker_artifact_preflight"] = (
            {"enabled": False}
            if gate_request is None
            else gate_request.preflight_payload()
        )
        if task_request is not None:
            status["worker_task_replay_session"] = (
                {"enabled": False}
                if task_session is None
                else task_session.status_dict()
            )
            status["worker_task_replay_preflight"] = (
                task_request.preflight_payload()
            )
            status["task_replay_preflight_ready"] = bool(
                task_session is not None and task_session.state == "ready_for_plan"
            )
            status["task_replay_request_id"] = task_request.request_id
            status["adapter_runtime_instance_id"] = str(
                getattr(adapter, "runtime_instance_id", "") or ""
            )
            status["root_state_write_count"] = int(
                getattr(adapter, "root_state_write_count", 0) or 0
            )
        if macro_request is not None:
            status["worker_macro_fsm_session"] = (
                {"enabled": False}
                if macro_session is None
                else macro_session.status_dict()
            )
            status["worker_macro_fsm_preflight"] = macro_request.preflight_payload()
            status["macro_fsm_preflight_ready"] = bool(
                macro_session is not None and macro_session.state == "ready_for_start"
            )
            status["macro_fsm_request_id"] = macro_request.request_id
            status["adapter_runtime_instance_id"] = str(
                getattr(adapter, "runtime_instance_id", "") or ""
            )
            status["root_state_write_count"] = int(
                getattr(adapter, "root_state_write_count", 0) or 0
            )
        if residual_request is not None:
            status["worker_residual_macro_fsm_session"] = (
                {"enabled": False}
                if macro_session is None
                else macro_session.status_dict()
            )
            status["worker_residual_macro_fsm_preflight"] = (
                residual_request.preflight_payload()
            )
            status["residual_macro_fsm_preflight_ready"] = bool(
                macro_session is not None
                and macro_session.state == "ready_for_start"
            )
            status["residual_macro_fsm_request_id"] = residual_request.request_id
            status["base_macro_fsm_request_id"] = (
                residual_request.base_request.request_id
            )
            status["adapter_runtime_instance_id"] = str(
                getattr(adapter, "runtime_instance_id", "") or ""
            )
            status["root_state_write_count"] = int(
                getattr(adapter, "root_state_write_count", 0) or 0
            )
        artifact_status = dict(status["worker_artifact_session"] or {})
        if (
            artifact_session is not None
            and artifact_status.get("terminal") is True
            and str(artifact_status.get("state", "") or "") == "failed"
        ):
            status["ready"] = False
            status["starting"] = False
            status["error"] = str(
                artifact_status.get("error", "")
                or "formal worker artifact session failed"
            )
        status["artifact_preflight_ready"] = bool(
            artifact_status.get("artifact_preflight_ready", False)
        )
        status["artifact_request_id"] = str(
            artifact_status.get("request_id", "") or ""
        )
        if detailed:
            status["motion_start_readiness"] = copy.deepcopy(
                last_motion_start_readiness
            )
        else:
            status["motion_start_readiness"] = {
                key: copy.deepcopy(last_motion_start_readiness.get(key))
                for key in (
                    "schema_version",
                    "motion_start_ready",
                    "decision",
                    "rejection_reason",
                    "current_sim_step",
                    "ground_state",
                    "adapter_runtime_instance_id",
                    "root_state_write_count",
                    "scheduler_accepted",
                    "scheduler_plan_id",
                    "scheduler_request_id",
                    "scheduler_worker_session_id",
                    "playback_start_boundary",
                )
                if key in last_motion_start_readiness
            }
        status["motion_start_ready"] = bool(
            last_motion_start_readiness.get("motion_start_ready", False)
        )
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

    def send_artifact_terminal(terminal: dict[str, Any]) -> None:
        """Publish raw diagnostics plus the sole outer-consumable terminal ACK."""

        nonlocal artifact_terminal_sent
        if artifact_terminal_sent:
            return
        ipc.send(dict(terminal))
        complete = str(terminal.get("type", "") or "") == "artifact_complete"
        ipc.send(
            make_message(
                "operation_ack",
                operation="recording_artifact",
                phase="ARTIFACT_COMPLETE" if complete else "ARTIFACT_FAILED",
                artifact_owner="sim_worker_process",
                accepted=complete,
                artifact_complete=complete,
                **_worker_ack_identity(
                    adapter,
                    worker_session_id=worker_session_id,
                    artifact_request_id=(
                        ""
                        if artifact_session is None
                        else artifact_session.request.request_id
                    ),
                    artifact_request=(
                        None if artifact_session is None else artifact_session.request
                    ),
                ),
                **{
                    key: value
                    for key, value in terminal.items()
                    if key
                    not in {
                        "type",
                        "operation",
                        "phase",
                        "artifact_owner",
                        "accepted",
                        "artifact_complete",
                        "worker_pid",
                        "worker_session_id",
                        "adapter_runtime_instance_id",
                        "artifact_request_id",
                        "root_state_write_count",
                        "contact_mode",
                        "environment_equivalence_role",
                        "diagnostic_role",
                        "qualification_scope",
                        "gate1_eligible",
                        "gate1_physical_qualification_eligible",
                        "environment_equivalence_eligible",
                    }
                },
            )
        )
        artifact_terminal_sent = True

    def send_task_terminal(terminal: dict[str, Any]) -> None:
        """Publish one durable normal-development task terminal and ACK."""

        nonlocal task_terminal_sent
        if task_terminal_sent:
            return
        ipc.send(dict(terminal))
        # The durable terminal itself is the critical delivery.  Mark it sent
        # before publishing the convenience operation ACK so an ACK failure
        # cannot cause the terminal frame to be emitted twice.
        task_terminal_sent = True
        complete = str(terminal.get("type", "") or "") == "task_replay_complete"
        excluded = {
            "type",
            "operation",
            "phase",
            "accepted",
            "task_replay_complete",
            "worker_pid",
            "worker_session_id",
            "adapter_runtime_instance_id",
            "artifact_request_id",
            "root_state_write_count",
        }
        ipc.send(
            make_message(
                "operation_ack",
                operation="task_replay",
                phase="TASK_REPLAY_COMPLETE" if complete else "TASK_REPLAY_FAILED",
                accepted=complete,
                task_replay_complete=complete,
                **_worker_ack_identity(
                    adapter,
                    worker_session_id=worker_session_id,
                    artifact_request_id="",
                ),
                **{
                    key: value
                    for key, value in terminal.items()
                    if key not in excluded
                },
            )
        )

    def send_macro_terminal(terminal: dict[str, Any]) -> None:
        """Publish one durable normal-development Macro terminal and ACK."""

        nonlocal macro_terminal_sent
        if macro_terminal_sent:
            return
        ipc.send(dict(terminal))
        macro_terminal_sent = True
        complete = str(terminal.get("type", "") or "") == "macro_fsm_complete"
        excluded = {
            "type",
            "operation",
            "phase",
            "accepted",
            "macro_fsm_complete",
            "worker_pid",
            "worker_session_id",
            "adapter_runtime_instance_id",
            "artifact_request_id",
            "root_state_write_count",
        }
        ipc.send(
            make_message(
                "operation_ack",
                operation="macro_fsm",
                phase="MACRO_FSM_COMPLETE" if complete else "MACRO_FSM_FAILED",
                accepted=complete,
                macro_fsm_complete=complete,
                **_worker_ack_identity(
                    adapter,
                    worker_session_id=worker_session_id,
                    artifact_request_id="",
                ),
                **{key: value for key, value in terminal.items() if key not in excluded},
            )
        )

    def request_macro_failure(
        reason: str,
        *,
        infrastructure_failure: bool,
        simulation_app_stopped: bool = False,
    ) -> None:
        """Request the session's atomic stop; publish only after verification."""

        if macro_session is None or macro_terminal_sent:
            return
        terminal = macro_session.fail(
            reason,
            infrastructure_failure=infrastructure_failure,
            simulation_app_stopped=simulation_app_stopped,
        )
        if terminal is not None:
            send_macro_terminal(terminal)

    try:
        ipc.send(make_message("hello", pid=os.getpid(), phase=phase, ready=False, starting=True))
        logger.log(f"[worker] pid={os.getpid()} connected; initial height={height_mm}mm")

        (
            gate_request,
            task_request,
            macro_request,
            residual_request,
        ) = _load_exclusive_fsm50_worker_requests(args)
        if gate_request is not None:
            if height_mm != gate_request.height_mm:
                raise ValueError(
                    f"worker height {height_mm}mm does not match Gate-1 request "
                    f"{gate_request.height_mm}mm"
                )
            artifact_session = WorkerRecordingSession(
                gate_request,
                worker_session_id=worker_session_id,
            )
            logger.log(
                "[worker] FSM50 Gate-1 artifact request accepted pre-scene "
                f"request_id={gate_request.request_id}"
            )
        if task_request is not None:
            if height_mm != task_request.height_mm:
                raise ValueError(
                    f"worker height {height_mm}mm does not match task replay request "
                    f"{task_request.height_mm}mm"
                )
            task_session = WorkerTaskReplaySession(
                task_request,
                worker_session_id=worker_session_id,
            )
            logger.log(
                "[worker] FSM50 normal task replay request accepted pre-scene "
                f"request_id={task_request.request_id}"
            )
        if macro_request is not None:
            if bool(getattr(args, "no_continuous_sim_step", False)):
                raise ValueError(
                    "Macro FSM requires the existing continuous production physics loop"
                )
            if height_mm != macro_request.height_mm:
                raise ValueError(
                    f"worker height {height_mm}mm does not match Macro FSM request "
                    f"{macro_request.height_mm}mm"
                )
            macro_session = WorkerMacroFSMSession(
                macro_request,
                worker_session_id=worker_session_id,
            )
            logger.log(
                "[worker] FSM50 normal Macro FSM request accepted pre-scene "
                f"request_id={macro_request.request_id}"
            )
        if residual_request is not None:
            if bool(getattr(args, "no_continuous_sim_step", False)):
                raise ValueError(
                    "Gate-E residual Macro FSM requires the existing continuous "
                    "production physics loop"
                )
            if height_mm != residual_request.base_request.height_mm:
                raise ValueError(
                    f"worker height {height_mm}mm does not match Gate-E residual "
                    f"Macro request {residual_request.base_request.height_mm}mm"
                )
            macro_session = GateEResidualMacroFSMSessionFactory(
                residual_request
            ).build_session(worker_session_id=worker_session_id)
            logger.log(
                "[worker] FSM50 Gate-E R0 ZERO residual Macro request accepted "
                f"pre-scene request_id={residual_request.request_id} "
                f"base_request_id={residual_request.base_request.request_id}"
            )

        set_phase("starting_app")
        logger.log("[worker] starting Isaac SimulationApp")
        simulation_app = ensure_simulation_app(args)

        set_phase("app_created")
        logger.log("[worker] SimulationApp created")

        logger.log(f"[worker] creating scene for {height_mm}mm")
        scene_config = config_from_args(args, height_mm)
        configure_scene_for_worker_recording(scene_config, gate_request)
        scene_handle = create_scene(
            scene_config,
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
        if artifact_session is not None:
            artifact_session.prepare_after_grounding(
                adapter=adapter,
                scene_handle=scene_handle,
                startup_ground=ground_init,
                robot_usd=Path(args.robot_usd),
            )
        if macro_session is not None:
            macro_session.prepare_after_adapter(
                adapter=adapter,
                scene_handle=scene_handle,
                project_root=Path(__file__).resolve().parent,
            )
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
            if artifact_session is not None and not artifact_session.terminal:
                rejected_messages = artifact_session.reject_direct_dispatch_messages(
                    polled_messages
                )
                if rejected_messages:
                    rejected_ids = {id(message) for message in rejected_messages}
                    polled_messages = [
                        message
                        for message in polled_messages
                        if id(message) not in rejected_ids
                    ]
                    rejected_kinds = [
                        str(message.get("type", "") or "")
                        for message in rejected_messages
                    ]
                    send_artifact_terminal(
                        artifact_session.fail(
                            "direct dispatch is forbidden during formal worker recording: "
                            + ", ".join(rejected_kinds)
                        )
                    )
                    publish_status(
                        ready=False,
                        starting=False,
                        error=artifact_session.error,
                    )
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
                        reason=str(stop_message.get("reason", "") or ""),
                        wheel_generation=int(
                            adapter.wheel_command_status.get(
                                "wheel_generation",
                                stop_message.get("wheel_generation", 0),
                            )
                            or 0
                        ),
                        **_worker_ack_identity(
                            adapter,
                            worker_session_id=worker_session_id,
                            artifact_request_id=(
                                ""
                                if artifact_session is None
                                else artifact_session.request.request_id
                            ),
                            artifact_request=(
                                None
                                if artifact_session is None
                                else artifact_session.request
                            ),
                        ),
                        received_wall_time=float(adapter.wheel_command_status.get("received_wall_time", time.time())),
                        target_applied_wall_time=float(adapter.wheel_command_status.get("target_applied_wall_time", time.time())),
                        target_applied_sim_time=float(adapter.wheel_command_status.get("target_applied_sim_time", sim_time)),
                        zero_target_applied=True,
                        error="",
                    )
                )
                if (
                    macro_session is not None
                    and macro_session.state
                    in {
                        "boundary_pending",
                        "running",
                        "terminal_command_pending_readback",
                        "safe_stop_pending_readback",
                        "settling",
                    }
                    and not macro_terminal_sent
                ):
                    request_macro_failure(
                        "external safety stop interrupted Macro FSM execution",
                        infrastructure_failure=False,
                    )
                publish_status(ready=True, starting=False)
            for message in polled_messages:
                kind = str(message.get("type", ""))
                if kind == "stop_wheels":
                    continue
                if kind == "shutdown":
                    requested_shutdown_mode = str(message.get("mode", "") or "").strip().lower()
                    if (
                        requested_shutdown_mode == "fast"
                        and artifact_session is None
                        and task_session is None
                        and macro_session is None
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                error=(
                                    "fast shutdown is restricted to an explicit FSM50 "
                                    "artifact, normal task-replay, or Macro FSM worker"
                                ),
                            )
                        )
                        continue
                    if (
                        requested_shutdown_mode == "fast"
                        and artifact_session is not None
                        and artifact_session.state != "complete"
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                mode="fast",
                                error=(
                                    "fast shutdown requires an ARTIFACT_COMPLETE "
                                    f"worker session; state={artifact_session.state}"
                                ),
                            )
                        )
                        continue
                    if (
                        requested_shutdown_mode == "fast"
                        and task_session is not None
                        and not task_session.fast_close_ready
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                mode="fast",
                                error=(
                                    "fast shutdown requires a durable terminal normal task "
                                    "replay with quiesced video writer; "
                                    f"state={task_session.state}"
                                ),
                            )
                        )
                        continue
                    if (
                        requested_shutdown_mode == "fast"
                        and macro_session is not None
                        and not macro_session.fast_close_ready
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                mode="fast",
                                **_macro_route_ack_identity(
                                    macro_session, payload_role="shutdown_ack"
                                ),
                                error=(
                                    "fast shutdown requires a durable terminal Macro FSM "
                                    "with quiesced video writer; "
                                    f"state={macro_session.state}"
                                ),
                            )
                        )
                        continue
                    if (
                        task_session is not None
                        and requested_shutdown_mode != "fast"
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                mode=requested_shutdown_mode or "normal",
                                task_replay_request_id=(
                                    task_session.request.request_id
                                ),
                                error=(
                                    "normal task replay requires the verified fast-close "
                                    "receipt path after a durable terminal result"
                                ),
                            )
                        )
                        continue
                    if macro_session is not None and requested_shutdown_mode != "fast":
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                mode=requested_shutdown_mode or "normal",
                                **_macro_route_ack_identity(
                                    macro_session, payload_role="shutdown_ack"
                                ),
                                error=(
                                    "normal Macro FSM requires the verified fast-close "
                                    "receipt path after a durable terminal result"
                                ),
                            )
                        )
                        continue
                    if (
                        task_session is not None
                        and requested_shutdown_mode == "fast"
                        and not str(message.get("request_id", "") or "")
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id="",
                                mode="fast",
                                task_replay_request_id=(
                                    task_session.request.request_id
                                ),
                                error="fast shutdown request_id is required",
                            )
                        )
                        continue
                    if (
                        macro_session is not None
                        and requested_shutdown_mode == "fast"
                        and not str(message.get("request_id", "") or "")
                    ):
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id="",
                                mode="fast",
                                **_macro_route_ack_identity(
                                    macro_session, payload_role="shutdown_ack"
                                ),
                                error="fast shutdown request_id is required",
                            )
                        )
                        continue
                    if requested_shutdown_mode not in {"", "normal", "fast"}:
                        ipc.send(
                            make_message(
                                "operation_ack",
                                operation="shutdown",
                                accepted=False,
                                request_id=str(message.get("request_id", "") or ""),
                                error=f"unsupported shutdown mode: {requested_shutdown_mode}",
                            )
                        )
                        continue
                    shutdown_mode = requested_shutdown_mode
                    shutdown_request_id = str(message.get("request_id", "") or "")
                    close_kwargs = (
                        {
                            "wait_for_replicator": False,
                            "skip_cleanup": True,
                        }
                        if shutdown_mode == "fast"
                        else {}
                    )
                    ipc.send(
                        make_message(
                            "operation_ack",
                            operation="shutdown",
                            accepted=True,
                            request_id=shutdown_request_id,
                            mode=shutdown_mode or "normal",
                            **_worker_ack_identity(
                                adapter,
                                worker_session_id=worker_session_id,
                                artifact_request_id=(
                                    ""
                                    if artifact_session is None
                                    else artifact_session.request.request_id
                                ),
                                artifact_request=(
                                    None
                                    if artifact_session is None
                                    else artifact_session.request
                                ),
                            ),
                            **(
                                {}
                                if task_session is None
                                else {
                                    "task_replay_request_id": (
                                        task_session.request.request_id
                                    )
                                }
                            ),
                            **(
                                {}
                                if macro_session is None
                                else {
                                    **_macro_route_ack_identity(
                                        macro_session,
                                        payload_role="shutdown_ack",
                                    ),
                                    "schema_version": MACRO_FAST_CLOSE_SCHEMA,
                                }
                            ),
                            close_kwargs=close_kwargs,
                            runtime_version=_worker_runtime_version(),
                            error="",
                        )
                    )
                    shutdown_requested = True
                    break
                if kind == "start_macro_fsm":
                    rejection_reasons: list[str] = []
                    if macro_session is None:
                        rejection_reasons.append(
                            "worker was not launched with an explicit Macro FSM request"
                        )
                    else:
                        if macro_session.state != "ready_for_start":
                            rejection_reasons.append(
                                "Macro FSM preflight is not ready: "
                                f"state={macro_session.state}"
                            )
                        if playback_service.active:
                            rejection_reasons.append(
                                "production playback already owns the dispatch slot"
                            )
                        rejection_reasons.extend(
                            _validate_macro_route_start_binding(
                                macro_session,
                                residual_request,
                                message,
                                expected_worker_session_id=worker_session_id,
                            )
                        )
                    accepted = not rejection_reasons
                    start_payload: dict[str, Any] = {}
                    if accepted and macro_session is not None:
                        try:
                            start_payload = dict(macro_session.start())
                        except Exception as macro_exc:
                            rejection_reasons.append(
                                "Macro FSM start failed: "
                                f"{type(macro_exc).__name__}: {macro_exc}"
                            )
                            accepted = False
                    rejection_reason = "; ".join(rejection_reasons)
                    if not accepted and macro_session is not None and not macro_terminal_sent:
                        request_macro_failure(
                            rejection_reason or "Macro FSM start rejected",
                            infrastructure_failure=True,
                        )
                    ipc.send(
                        _build_macro_start_operation_ack(
                            message=message,
                            accepted=accepted,
                            rejection_reason=rejection_reason,
                            adapter=adapter,
                            worker_session_id=worker_session_id,
                            start_payload=start_payload,
                        )
                    )
                    publish_status(ready=True, starting=False)
                    continue
                if macro_session is not None and kind in {
                    "command",
                    "apply_motion_batch",
                    "start_playback_plan",
                    "pause_playback",
                    "resume_playback",
                    "stop_playback",
                    "set_height",
                    "set_height_respawn",
                    "recalibrate_ground_reference",
                    "respawn",
                    "restore_sim_state",
                }:
                    rejection_reason = (
                        f"{kind} is forbidden while the Macro FSM worker owns "
                        "the production dispatch slot"
                    )
                    if not macro_terminal_sent:
                        request_macro_failure(
                            rejection_reason,
                            infrastructure_failure=False,
                        )
                    ipc.send(
                        make_message(
                            "operation_ack",
                            operation=kind,
                            request_id=str(message.get("request_id", "") or ""),
                            accepted=False,
                            **_macro_route_ack_identity(
                                macro_session, payload_role="shutdown_ack"
                            ),
                            error=rejection_reason,
                        )
                    )
                    publish_status(ready=False, starting=False, error=rejection_reason)
                    continue
                if kind == "command":
                    adapter.handle_command(_command_message_from_ipc(message))
                elif kind == "apply_motion_batch":
                    if playback_service.active:
                        ack = {
                            "batch_id": str(message.get("batch_id", "") or ""),
                            "source": str(message.get("source", "ui") or "ui"),
                            "applied_sim_step": None,
                            "first_physics_step": None,
                            "servo_targets_applied": {},
                            "wheel_targets_applied": {},
                            "servo_applied": False,
                            "wheel_applied": False,
                            "motion_start_skew_s": None,
                            "error": (
                                "direct motion batch rejected while production "
                                "playback owns the per-tick dispatch slot"
                            ),
                        }
                    else:
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
                    if (
                        artifact_session is not None
                        and artifact_session.state != "ready_for_plan"
                    ):
                        rejection_reasons.append(
                            "FSM50 artifact preflight is not ready: "
                            f"state={artifact_session.state}"
                        )
                    if (
                        task_session is not None
                        and task_session.state != "ready_for_plan"
                    ):
                        rejection_reasons.append(
                            "FSM50 normal task replay preflight is not ready: "
                            f"state={task_session.state}"
                        )
                    if artifact_session is not None:
                        rejection_reasons.extend(
                            validate_worker_plan_binding(
                                artifact_session.request,
                                request_id=request_id,
                                plan_id=requested_plan_id,
                                worker_session_id=str(
                                    message.get("worker_session_id", "") or ""
                                ),
                                expected_worker_session_id=worker_session_id,
                            )
                        )
                    if task_session is not None:
                        rejection_reasons.extend(
                            validate_worker_task_plan_binding(
                                task_session.request,
                                plan=plan,
                                request_id=request_id,
                                plan_id=requested_plan_id,
                            )
                        )
                    current_motion_start_step = int(
                        getattr(adapter, "sim_steps", sim_steps) or sim_steps
                    )
                    last_adapter_batch = dict(
                        getattr(adapter, "motion_batch_status", {}) or {}
                    )
                    try:
                        adapter_batch_owns_current_step = bool(
                            last_adapter_batch
                            and not str(last_adapter_batch.get("error", "") or "")
                            and int(last_adapter_batch.get("applied_sim_step"))
                            == current_motion_start_step
                        )
                    except (TypeError, ValueError):
                        adapter_batch_owns_current_step = False
                    if adapter_batch_owns_current_step:
                        rejection_reasons.append(
                            "current physics tick already contains a motion batch"
                        )
                    request_identity = {
                        "request_id": request_id,
                        "plan_id": requested_plan_id,
                        "plan_sha256": requested_sha,
                        "validated_plan_sha256": str(
                            integrity.get("plan_sha256", "") or ""
                        ),
                        "event_count": message.get("event_count"),
                        "validated_event_count": integrity.get("event_count"),
                        "segment_count": message.get("segment_count"),
                        "validated_segment_count": integrity.get("segment_count"),
                        "integrity_ok": integrity.get("ok") is True,
                        "requested_worker_session_id": str(
                            message.get("worker_session_id", "") or ""
                        ),
                        "source_initial_command_state": copy.deepcopy(
                            dict(
                                plan.timing.get("source_initial_command_state", {})
                                or {}
                            )
                        ),
                        "source_initial_command_state_sha256": str(
                            plan.timing.get(
                                "source_initial_command_state_sha256", ""
                            )
                            or ""
                        ),
                    }
                    if (
                        artifact_session is not None
                        and artifact_session.expected_plan_identity
                    ):
                        request_identity = dict(
                            artifact_session.expected_plan_identity
                        )
                    last_motion_start_readiness = capture_worker_motion_start_readiness(
                        adapter,
                        runtime_ready=bool(phase == "running"),
                        current_sim_step=current_motion_start_step,
                        worker_session_id=worker_session_id,
                        request_identity=request_identity,
                    )
                    if (
                        artifact_session is not None
                        and artifact_session.request.diagnostic_role == "U"
                    ):
                        physical_motion_start_ready = bool(
                            last_motion_start_readiness.get(
                                "motion_start_ready", False
                            )
                        )
                        role_motion_start_ready = (
                            artifact_session.motion_start_ready_for_role(
                                artifact_session.motion_start_readiness,
                                last_motion_start_readiness,
                            )
                        )
                        last_motion_start_readiness = {
                            **last_motion_start_readiness,
                            "motion_start_ready": role_motion_start_ready,
                            "diagnostic_role": "U",
                            "qualification_scope": (
                                "PRODUCTION_DEFAULT_TRAJECTORY_DIAGNOSTIC"
                            ),
                            "physical_motion_start_ready": (
                                physical_motion_start_ready
                            ),
                            "physical_motion_start_verdict": "NOT_EVALUABLE",
                            "sensor_independent_trajectory_admission": (
                                role_motion_start_ready
                            ),
                            "rejection_reason": (
                                ""
                                if role_motion_start_ready
                                else "sensor-independent U trajectory readiness failed"
                            ),
                        }
                    if not bool(last_motion_start_readiness.get("motion_start_ready", False)):
                        rejection_reasons.append(
                            "MOTION_START_READY failed: "
                            + str(
                                last_motion_start_readiness.get(
                                    "rejection_reason",
                                    "missing readiness evidence",
                                )
                                or "missing readiness evidence"
                            )
                        )
                    ok = not rejection_reasons
                    if ok:
                        plan.plan_sha256 = str(integrity["plan_sha256"])
                        # Only the production worker and the audited v003
                        # runner can satisfy this live, pre-first-dispatch
                        # contract.  The generic compiler remains usable by
                        # offline/direct scheduler tests without claiming it.
                        plan.timing[
                            "requires_motion_start_readiness_token"
                        ] = True
                        plan.timing["requires_verified_motion_batch_ack"] = True
                        requested_start_delay_s = float(
                            message.get("start_delay_sim_s", 0.0) or 0.0
                        )
                        effective_start_delay_s = max(
                            requested_start_delay_s,
                            float(dt),
                        )
                        ok = playback_service.start_plan(
                            plan,
                            current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                            current_wall_time_s=time.time(),
                            start_delay_sim_s=effective_start_delay_s,
                            plan_id=requested_plan_id,
                            request_id=request_id,
                            worker_session_id=worker_session_id,
                        )
                        if not ok:
                            rejection_reasons.append(playback_service.last_error or "scheduler rejected plan")
                        else:
                            if artifact_session is not None:
                                try:
                                    artifact_session.attach_verified_plan(
                                        plan=plan,
                                        service=playback_service,
                                        adapter=adapter,
                                        scene_handle=scene_handle,
                                        motion_start_readiness=last_motion_start_readiness,
                                        robot_usd=Path(args.robot_usd),
                                    )
                                except Exception as artifact_exc:
                                    rejection_reasons.append(
                                        "FSM50 worker artifact plan admission failed: "
                                        f"{type(artifact_exc).__name__}: {artifact_exc}"
                                    )
                                    playback_service.stop(
                                        adapter,
                                        current_sim_time_s=float(
                                            getattr(adapter, "sim_time", sim_time)
                                            or sim_time
                                        ),
                                        current_wall_time_s=time.time(),
                                        reason="artifact_plan_admission_failed",
                                        stop_wheels=True,
                                    )
                                    send_artifact_terminal(
                                        artifact_session.fail(rejection_reasons[-1])
                                    )
                                    ok = False
                            if ok and task_session is not None:
                                try:
                                    task_session.attach_verified_plan(
                                        plan=plan,
                                        service=playback_service,
                                        adapter=adapter,
                                        scene_handle=scene_handle,
                                    )
                                except Exception as task_exc:
                                    rejection_reasons.append(
                                        "FSM50 normal task replay plan admission failed: "
                                        f"{type(task_exc).__name__}: {task_exc}"
                                    )
                                    playback_service.stop(
                                        adapter,
                                        current_sim_time_s=float(
                                            getattr(adapter, "sim_time", sim_time)
                                            or sim_time
                                        ),
                                        current_wall_time_s=time.time(),
                                        reason="task_replay_plan_admission_failed",
                                        stop_wheels=True,
                                    )
                                    send_task_terminal(
                                        task_session.fail(rejection_reasons[-1])
                                    )
                                    ok = False
                            if not ok:
                                boundary_ok = False
                            else:
                                boundary_ok = playback_service.apply_playback_start_boundary(
                                    adapter,
                                    current_sim_time_s=float(
                                        getattr(adapter, "sim_time", sim_time)
                                        or sim_time
                                    ),
                                    current_sim_step=current_motion_start_step,
                                )
                            last_motion_start_readiness = {
                                **last_motion_start_readiness,
                                "requested_start_delay_sim_s": requested_start_delay_s,
                                "effective_start_delay_sim_s": effective_start_delay_s,
                                "start_boundary_minimum_physics_ticks": 1,
                                "playback_start_boundary": {
                                    "applied": bool(boundary_ok),
                                    "ack": copy.deepcopy(
                                        playback_service.last_motion_batch_ack
                                    ),
                                    "sim_step": current_motion_start_step,
                                    "error": ""
                                    if boundary_ok
                                    else str(
                                        playback_service.last_error
                                        or "playback zero-wheel start boundary failed"
                                    ),
                                },
                            }
                            if boundary_ok and artifact_session is not None:
                                artifact_session.record_start_boundary()
                            if boundary_ok and task_session is not None:
                                task_session.record_start_boundary()
                            if not boundary_ok:
                                rejection_reasons.append(
                                    "playback zero-wheel start boundary failed: "
                                    + str(
                                        playback_service.last_error
                                        or "missing verified batch ACK"
                                    )
                                )
                                playback_service.stop(
                                    adapter,
                                    current_sim_time_s=float(
                                        getattr(adapter, "sim_time", sim_time)
                                        or sim_time
                                    ),
                                    current_wall_time_s=time.time(),
                                    reason="motion_start_boundary_failed",
                                    stop_wheels=True,
                                )
                                ok = False
                    rejection_reason = "; ".join(rejection_reasons)
                    last_motion_start_readiness = {
                        **last_motion_start_readiness,
                        "scheduler_accepted": bool(ok),
                        "scheduler_rejection_reason": rejection_reason,
                        "scheduler_plan_id": requested_plan_id,
                        "scheduler_request_id": request_id,
                        "scheduler_worker_session_id": worker_session_id,
                    }
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
                            **_worker_ack_identity(
                                adapter,
                                worker_session_id=worker_session_id,
                                artifact_request_id=(
                                    ""
                                    if artifact_session is None
                                    else artifact_session.request.request_id
                                ),
                                artifact_request=(
                                    None
                                    if artifact_session is None
                                    else artifact_session.request
                                ),
                            ),
                            accepted_wall_time=time.time(),
                            motion_start_ready=bool(
                                last_motion_start_readiness.get(
                                    "motion_start_ready", False
                                )
                            ),
                            motion_start_readiness=copy.deepcopy(
                                last_motion_start_readiness
                            ),
                        )
                    )
                    if not ok:
                        logger.log(f"[worker] playback plan rejected: {rejection_reason}")
                        if (
                            task_session is not None
                            and not task_session.terminal
                            and not task_terminal_sent
                        ):
                            send_task_terminal(
                                task_session.fail(
                                    rejection_reason
                                    or "production scheduler rejected task replay"
                                )
                            )
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
                if (
                    playback_service.active
                    and not playback_service.started
                    and not playback_service.motion_start_readiness_token
                ):
                    boundary = dict(
                        last_motion_start_readiness.get(
                            "playback_start_boundary", {}
                        )
                        or {}
                    )
                    boundary_ack = dict(boundary.get("ack", {}) or {})
                    try:
                        boundary_first_step = int(
                            boundary_ack.get("first_physics_step")
                        )
                    except (TypeError, ValueError):
                        boundary_first_step = -1
                    current_pre_first_step = int(
                        getattr(adapter, "sim_steps", sim_steps) or sim_steps
                    )
                    if (
                        boundary.get("applied") is True
                        and current_pre_first_step >= boundary_first_step >= 0
                    ):
                        identity = dict(
                            last_motion_start_readiness.get("identity", {}) or {}
                        )
                        pre_first_readiness = capture_worker_motion_start_readiness(
                            adapter,
                            runtime_ready=bool(phase == "running"),
                            current_sim_step=current_pre_first_step,
                            worker_session_id=worker_session_id,
                            request_identity=identity,
                        )
                        if artifact_session is not None:
                            artifact_pre_first_ready = (
                                artifact_session.record_pre_first_dispatch(
                                    {
                                        **pre_first_readiness,
                                        "playback_start_boundary": boundary,
                                        "pre_first_dispatch": True,
                                        "pre_first_dispatch_sim_step": (
                                            current_pre_first_step
                                        ),
                                    }
                                )
                            )
                            readiness_token = str(
                                artifact_session.readiness_token or ""
                            )
                            token_bound = bool(
                                artifact_pre_first_ready and readiness_token
                            )
                        else:
                            token_payload = {
                                "schema_version": (
                                    "production.motion_start_readiness_token.v1"
                                ),
                                "worker_session_id": worker_session_id,
                                "plan_id": playback_service.plan_id,
                                "request_id": playback_service.request_id,
                                "plan_sha256": str(
                                    playback_service.plan.plan_sha256
                                    if playback_service.plan is not None
                                    else ""
                                ),
                                "boundary_ack": boundary_ack,
                                "pre_first_readiness": pre_first_readiness,
                                "pre_first_dispatch_sim_step": (
                                    current_pre_first_step
                                ),
                            }
                            readiness_token = hashlib.sha256(
                                json.dumps(
                                    token_payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    default=str,
                                ).encode("utf-8")
                            ).hexdigest()
                            token_bound = bool(
                                pre_first_readiness.get(
                                    "motion_start_ready", False
                                )
                                and playback_service.bind_motion_start_readiness(
                                    readiness_token,
                                    current_sim_step=current_pre_first_step,
                                )
                            )
                        last_motion_start_readiness = {
                            **pre_first_readiness,
                            "playback_start_boundary": boundary,
                            "pre_first_dispatch": True,
                            "pre_first_dispatch_sim_step": current_pre_first_step,
                            "readiness_token_sha256": (
                                readiness_token if token_bound else ""
                            ),
                            "readiness_token_bound": token_bound,
                        }
                        if artifact_session is not None:
                            if not artifact_pre_first_ready:
                                token_bound = False
                                playback_service.stop(
                                    adapter,
                                    current_sim_time_s=float(
                                        getattr(adapter, "sim_time", sim_time)
                                        or sim_time
                                    ),
                                    current_wall_time_s=time.time(),
                                    reason="artifact_rich_pre_first_dispatch_failed",
                                    stop_wheels=True,
                                )
                                if not artifact_terminal_sent:
                                    send_artifact_terminal(
                                        artifact_session.fail(
                                            "rich 10-frame pre-first-dispatch readiness failed"
                                        )
                                    )
                        if not token_bound:
                            playback_service.stop(
                                adapter,
                                current_sim_time_s=float(
                                    getattr(adapter, "sim_time", sim_time)
                                    or sim_time
                                ),
                                current_wall_time_s=time.time(),
                                reason="motion_start_pre_first_dispatch_failed",
                                stop_wheels=True,
                            )
                playback_service.update(
                    adapter,
                    current_sim_time_s=float(getattr(adapter, "sim_time", sim_time) or sim_time),
                    current_sim_step=int(getattr(adapter, "sim_steps", sim_steps) or sim_steps),
                    current_wall_time_s=time.time(),
                )
                if artifact_session is not None:
                    artifact_session.before_adapter_step()
                if task_session is not None and not task_session.terminal:
                    task_session.before_adapter_step()
                if macro_session is not None and not macro_session.terminal:
                    macro_session.before_adapter_step()
                adapter.step(dt)
                sim_time = float(getattr(adapter, "sim_time", sim_time + dt))
                sim_steps = int(getattr(adapter, "sim_steps", sim_steps + 1))
                if artifact_session is not None:
                    terminal = artifact_session.after_adapter_step()
                    if terminal is not None and not artifact_terminal_sent:
                        send_artifact_terminal(terminal)
                if task_session is not None and not task_session.terminal:
                    terminal = task_session.after_adapter_step()
                    if terminal is not None and not task_terminal_sent:
                        send_task_terminal(terminal)
                if macro_session is not None and not macro_session.terminal:
                    terminal = macro_session.after_adapter_step()
                    if terminal is not None and not macro_terminal_sent:
                        send_macro_terminal(terminal)
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

        if task_session is not None and not task_terminal_sent:
            if shutdown_requested:
                task_exit_reason = "worker shutdown requested before task replay terminal"
            elif smoke_deadline is not None and time.monotonic() >= smoke_deadline:
                task_exit_reason = "worker smoke deadline reached before task replay terminal"
            else:
                task_exit_reason = "simulation app stopped before task replay terminal"
            task_terminal_sent = _finalize_active_task_session_for_worker_exit(
                task_session,
                terminal_sent=task_terminal_sent,
                publish_terminal=send_task_terminal,
                reason=task_exit_reason,
            )
        if macro_session is not None and not macro_terminal_sent:
            if shutdown_requested:
                macro_exit_reason = "worker shutdown requested before Macro FSM terminal"
            elif smoke_deadline is not None and time.monotonic() >= smoke_deadline:
                macro_exit_reason = "worker smoke deadline reached before Macro FSM terminal"
            else:
                macro_exit_reason = "simulation app stopped before Macro FSM terminal"
            macro_terminal_sent = _finalize_active_macro_session_for_worker_exit(
                macro_session,
                terminal_sent=macro_terminal_sent,
                publish_terminal=send_macro_terminal,
                reason=macro_exit_reason,
            )
        if (
            task_session is not None
            and task_session.fast_close_ready
            and shutdown_mode != "fast"
            and getattr(ipc, "sock", None) is not None
        ):
            shutdown_request_id = _wait_for_task_exception_fast_shutdown(
                ipc,
                task_session=task_session,
                adapter=adapter,
                worker_session_id=worker_session_id,
            )
            if shutdown_request_id:
                shutdown_mode = "fast"
        if (
            macro_session is not None
            and macro_session.fast_close_ready
            and shutdown_mode != "fast"
            and getattr(ipc, "sock", None) is not None
        ):
            shutdown_request_id = _wait_for_macro_exception_fast_shutdown(
                ipc,
                macro_session=macro_session,
                adapter=adapter,
                worker_session_id=worker_session_id,
            )
            if shutdown_request_id:
                shutdown_mode = "fast"
        set_phase("shutdown", publish=False)
        publish_status(ready=False, starting=False)
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        logger.log(f"[worker] ERROR: {exc}")
        logger.log(tb)
        if artifact_session is not None and not artifact_terminal_sent:
            try:
                send_artifact_terminal(
                    artifact_session.fail(f"{type(exc).__name__}: {exc}")
                )
            except Exception:
                pass
        if task_session is not None and not task_terminal_sent:
            try:
                send_task_terminal(
                    task_session.fail(
                        f"{type(exc).__name__}: {exc}",
                        infrastructure_failure=True,
                    )
                )
            except Exception:
                pass
        if macro_session is not None and not macro_terminal_sent:
            try:
                request_macro_failure(
                    f"{type(exc).__name__}: {exc}",
                    infrastructure_failure=True,
                    simulation_app_stopped=True,
                )
            except Exception:
                pass
        try:
            ipc.send(make_message("error", phase=phase, error=str(exc), traceback=tb, ready=False, starting=False))
        except Exception:
            pass
        if (
            task_session is not None
            and task_session.fast_close_ready
            and getattr(ipc, "sock", None) is not None
        ):
            shutdown_request_id = _wait_for_task_exception_fast_shutdown(
                ipc,
                task_session=task_session,
                adapter=adapter,
                worker_session_id=worker_session_id,
            )
            if shutdown_request_id:
                shutdown_mode = "fast"
        if (
            macro_session is not None
            and macro_session.fast_close_ready
            and getattr(ipc, "sock", None) is not None
        ):
            shutdown_request_id = _wait_for_macro_exception_fast_shutdown(
                ipc,
                macro_session=macro_session,
                adapter=adapter,
                worker_session_id=worker_session_id,
            )
            if shutdown_request_id:
                shutdown_mode = "fast"
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
            if shutdown_mode == "fast" and (
                artifact_session is not None
                or task_session is not None
                or macro_session is not None
            ):
                fast_artifact_request = (
                    None if artifact_session is None else artifact_session.request
                )
                fast_artifact_request_id = (
                    ""
                    if fast_artifact_request is None
                    else fast_artifact_request.request_id
                )
                fast_task_request_id = (
                    "" if task_session is None else task_session.request.request_id
                )
                close_requested = make_message(
                    "close_requested",
                    mode="fast",
                    accepted=True,
                    error="",
                    request_id=shutdown_request_id,
                    **_worker_ack_identity(
                        adapter,
                        worker_session_id=worker_session_id,
                        artifact_request_id=fast_artifact_request_id,
                        artifact_request=fast_artifact_request,
                    ),
                    **(
                        {}
                        if task_session is None
                        else {"task_replay_request_id": fast_task_request_id}
                    ),
                    **(
                        {}
                        if macro_session is None
                        else {
                            **_macro_route_ack_identity(
                                macro_session,
                                payload_role="close_requested",
                            ),
                            "schema_version": MACRO_FAST_CLOSE_SCHEMA,
                        }
                    ),
                    close_kwargs={
                        "wait_for_replicator": False,
                        "skip_cleanup": True,
                    },
                    runtime_version=_worker_runtime_version(),
                )
                ipc.send(close_requested)
                close_receipt = _wait_for_close_receipt(
                    ipc,
                    close_requested,
                    timeout_s=None,
                )
                logger.log(
                    "[worker] verified close_requested receipt accepted "
                    f"request_id={close_receipt.get('request_id', '')}"
                )
                if simulation_app is not None:
                    simulation_app.close(
                        wait_for_replicator=False,
                        skip_cleanup=True,
                    )
                ipc.send(
                    make_message(
                        "close_returned",
                        mode="fast",
                        accepted=True,
                        error="",
                        request_id=shutdown_request_id,
                        **_worker_ack_identity(
                            adapter,
                            worker_session_id=worker_session_id,
                            artifact_request_id=fast_artifact_request_id,
                            artifact_request=fast_artifact_request,
                        ),
                        **(
                            {}
                            if task_session is None
                            else {"task_replay_request_id": fast_task_request_id}
                        ),
                        **(
                            {}
                            if macro_session is None
                            else {
                                **_macro_route_ack_identity(
                                    macro_session,
                                    payload_role="close_returned",
                                ),
                                "schema_version": MACRO_FAST_CLOSE_SCHEMA,
                            }
                        ),
                        close_kwargs={
                            "wait_for_replicator": False,
                            "skip_cleanup": True,
                        },
                        runtime_version=_worker_runtime_version(),
                    )
                )
                ipc.flush_until_empty(0.5)
            elif scene_handle is not None:
                scene_handle.close()
            elif simulation_app is not None:
                simulation_app.close()
        except Exception as close_exc:
            if shutdown_mode == "fast" and (
                artifact_session is not None
                or task_session is not None
                or macro_session is not None
            ):
                try:
                    logger.log(
                        "[worker] verified fast close ERROR: "
                        f"{type(close_exc).__name__}: {close_exc}"
                    )
                except Exception:
                    pass
                ipc.close()
                # Override the pending successful return: a Python exception
                # from receipt handling or SimulationApp.close is a non-zero
                # worker exit, never a verified native fast-close result.
                raise
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
    parser.add_argument(
        "--fsm50-gate-request-path",
        "--fsm50_gate_request_path",
        dest="fsm50_gate_request_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--fsm50-task-request-path",
        "--fsm50_task_request_path",
        dest="fsm50_task_request_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--fsm50-macro-request-path",
        "--fsm50_macro_request_path",
        dest="fsm50_macro_request_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--fsm50-residual-macro-request-path",
        "--fsm50_residual_macro_request_path",
        dest="fsm50_residual_macro_request_path",
        type=str,
        default="",
    )
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
    # Every worker enters scene configuration from the ordinary production UI
    # default.  The validated request hook runs after request loading and is
    # the sole authority that may enable/configure A1/A2/B sensor plumbing;
    # U must observe this exact False value and leave it unchanged.
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
