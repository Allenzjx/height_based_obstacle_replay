"""Background Isaac Sim loop for the Tk UI.

This worker keeps SimulationApp, scene creation, command application, and
``sim.step()`` away from Tk callbacks. It is intentionally thread-based for the
current desktop workflow; the UI process communicates only through queues.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from command_model import CommandMessage
from height_manifest import obstacle_height_m
from camera_validation import default_camera_health_status, default_height_validation_result
from obstacle_height_vision import VisionHeightConfig
from sim_camera_viewport import default_camera_viewport_status
from sim_obstacle_scene import create_scene, ensure_simulation_app
from sim_robot_adapter import SimRobotAdapter
from sim_onboard_camera import OnboardCameraProcessor
from sim_worker_runtime import (
    build_common_worker_status,
    create_adapter_config_from_args,
    enrich_runtime_readiness,
    handle_respawn,
    handle_set_height,
    handle_vision_control,
    initialize_adapter_ground_reference,
    service_pending_viewport,
)
from telemetry import create_telemetry_collector


@dataclass
class SimWorkerCommand:
    kind: str
    payload: Any = None


class SimWorker:
    def __init__(
        self,
        args: Any,
        *,
        config_factory: Any,
        initial_height_cm: int,
        status_interval_s: float = 0.25,
        no_continuous_sim_step: bool = False,
    ):
        self.args = args
        self.config_factory = config_factory
        self.initial_height_cm = int(initial_height_cm)
        self.status_interval_s = max(0.05, float(status_interval_s))
        self.no_continuous_sim_step = bool(no_continuous_sim_step)
        self.command_queue: "queue.Queue[SimWorkerCommand]" = queue.Queue()
        self.status_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ready = False
        self.last_status: dict[str, Any] = {
            "ready": False,
            "starting": False,
            "error": "",
            "height_cm": self.initial_height_cm,
        }
        self.respawn_count = 0
        self.last_respawn_at = 0.0
        self.restore_count = 0
        self.last_restore_at = 0.0
        self.last_restore_result = ""
        self.last_restore_error = ""
        self.vision_processor: OnboardCameraProcessor | None = None
        self.obstacle_revision = 0
        self.last_set_height_result: dict[str, Any] = {}
        self.last_set_height_source = "startup"
        self.last_set_height_request_id = ""

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="HeightReplaySimWorker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.command_queue.put(SimWorkerCommand("stop"))

    def send_command(self, command: str, *, source: str = "ui", **metadata: Any) -> None:
        payload = {
            key: metadata[key]
            for key in ("playback_label", "playback_event_index", "playback_event_count", "playback_final_time_s", "source_step")
            if key in metadata and metadata[key] is not None
        }
        self.command_queue.put(SimWorkerCommand("command", CommandMessage(text=str(command), source=source, **payload)))

    def set_height(self, height_cm: int, **payload: Any) -> None:
        message = {"height_cm": int(height_cm)}
        message.update(payload)
        self.command_queue.put(SimWorkerCommand("set_height", message))

    def respawn(self) -> None:
        self.command_queue.put(SimWorkerCommand("respawn"))

    def request_state(self) -> None:
        self.command_queue.put(SimWorkerCommand("request_state"))

    def restore_sim_state(self, sim_state: dict[str, Any]) -> None:
        self.command_queue.put(SimWorkerCommand("restore_sim_state", dict(sim_state or {})))

    def vision_control(self, action: str, **payload: Any) -> None:
        message = {"action": str(action)}
        message.update(payload)
        self.command_queue.put(SimWorkerCommand("vision_control", message))

    def drain_status(self) -> list[dict[str, Any]]:
        statuses: list[dict[str, Any]] = []
        while True:
            try:
                status = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self.last_status = status
            statuses.append(status)
        return statuses

    def _publish(self, status: dict[str, Any]) -> None:
        self.last_status = status
        try:
            self.status_queue.put_nowait(status)
        except queue.Full:
            pass

    def _run(self) -> None:
        simulation_app = None
        scene_handle = None
        adapter = None
        telemetry_collector = None
        telemetry_finish_success = False
        telemetry_finish_reason = "worker did not complete"
        vision_processor = None
        height_cm = self.initial_height_cm
        sim_time = 0.0
        sim_steps = 0
        started_wall = time.monotonic()
        self._started_wall_for_guard = started_wall
        last_status_wall = 0.0
        last_step_wall = time.monotonic()
        self._publish({"ready": False, "starting": True, "height_cm": height_cm, "error": ""})
        try:
            simulation_app = ensure_simulation_app(self.args)
            scene_handle = create_scene(self.config_factory(height_cm), simulation_app=simulation_app)
            vision_processor = OnboardCameraProcessor(scene_handle, self._vision_config())
            self.vision_processor = vision_processor
            adapter = SimRobotAdapter(scene_handle, create_adapter_config_from_args(self.args))
            telemetry_collector = create_telemetry_collector(self.args, scene_handle=scene_handle)
            if telemetry_collector is not None:
                adapter.attach_telemetry(telemetry_collector)
                telemetry_collector.start_episode(
                    adapter=adapter,
                    scene_handle=scene_handle,
                    obstacle_height_cm=height_cm,
                    obstacle_height_m=obstacle_height_m(height_cm),
                    sequence_label=f"{height_cm}cm thread worker session",
                    source="thread_worker",
                )
            initialize_adapter_ground_reference(adapter)
            sim_time = float(getattr(adapter, "sim_time", sim_time))
            sim_steps = int(getattr(adapter, "sim_steps", sim_steps))
            self.ready = True
            self._publish(self._status(adapter, height_cm, sim_time, sim_steps, started_wall, "", starting=False))
            dt = float(scene_handle.sim.get_physics_dt())
            while not self.stop_event.is_set() and scene_handle.app_is_running():
                self._drain_commands(adapter, scene_handle, height_cm_ref=[height_cm])
                # _drain_commands updates the mutable ref so local height stays current.
                height_cm = int(getattr(self, "_height_cm", height_cm))
                now = time.monotonic()
                service_pending_viewport(
                    args=self.args,
                    adapter=adapter,
                    scene_handle=scene_handle,
                    vision_processor=vision_processor,
                    runtime_metrics={"started_wall": started_wall},
                )
                if not self.no_continuous_sim_step:
                    adapter.step(dt)
                    sim_time = float(getattr(adapter, "sim_time", sim_time + dt))
                    sim_steps = int(getattr(adapter, "sim_steps", sim_steps + 1))
                    if vision_processor is not None:
                        vision_processor.update(dt=dt, sim_time=sim_time, wall_time=time.time())
                else:
                    time.sleep(min(dt, 0.02))
                    if vision_processor is not None:
                        vision_processor.update(dt=dt, sim_time=sim_time, wall_time=time.time())
                if now - last_status_wall >= self.status_interval_s:
                    elapsed_wall = max(1.0e-9, now - started_wall)
                    status = self._status(adapter, height_cm, sim_time, sim_steps, started_wall, "", starting=False)
                    status["real_time_factor"] = sim_time / elapsed_wall
                    status["worker_loop_hz"] = 1.0 / max(1.0e-9, now - last_step_wall)
                    self._publish(status)
                    last_status_wall = now
                last_step_wall = now
            telemetry_finish_success = True
            telemetry_finish_reason = "worker shutdown"
        except Exception as exc:
            telemetry_finish_reason = str(exc)
            self.ready = False
            self._publish({"ready": False, "starting": False, "height_cm": height_cm, "error": str(exc)})
        finally:
            try:
                if telemetry_collector is not None:
                    telemetry_collector.finish_episode(success=telemetry_finish_success, reason=telemetry_finish_reason)
            except Exception:
                pass
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

    def _drain_commands(self, adapter: Any, scene_handle: Any, *, height_cm_ref: list[int]) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            if command.kind == "stop":
                self.stop_event.set()
                return
            if command.kind == "command":
                adapter.handle_command(command.payload)
            elif command.kind == "set_height":
                payload = command.payload if isinstance(command.payload, dict) else {"height_cm": int(command.payload)}
                height_cm = int(payload.get("height_cm", height_cm_ref[0]))
                self.last_set_height_source = str(payload.get("source", "ui") or "ui")
                self.last_set_height_request_id = str(payload.get("request_id", "") or "")
                self.obstacle_revision += 1
                result = handle_set_height(
                    adapter=adapter,
                    scene_handle=scene_handle,
                    vision_processor=self.vision_processor,
                    height_cm=height_cm,
                    source=self.last_set_height_source,
                    request_id=self.last_set_height_request_id,
                    obstacle_revision=self.obstacle_revision,
                    respawn_policy=str(payload.get("respawn_policy", "required") or "required"),
                )
                self.last_set_height_result = dict(result)
                collector = getattr(adapter, "telemetry_collector", None)
                if collector is not None:
                    collector.update_obstacle_context(
                        height_cm=height_cm,
                        height_m=obstacle_height_m(height_cm),
                        source=self.last_set_height_source,
                        request_id=self.last_set_height_request_id,
                    )
                if bool(result.get("respawned", False)):
                    self.respawn_count += 1
                    self.last_respawn_at = time.time()
                if not bool(result.get("ok", True)):
                    self.last_restore_error = str(result.get("error", ""))
                self._height_cm = height_cm
                height_cm_ref[0] = height_cm
            elif command.kind == "respawn":
                result = handle_respawn(adapter=adapter, vision_processor=self.vision_processor, reset_filter=True)
                if bool(result.get("respawned", False)):
                    self.respawn_count += 1
                    self.last_respawn_at = time.time()
                if not bool(result.get("ok", True)):
                    self.last_restore_error = str(result.get("error", ""))
            elif command.kind == "restore_sim_state":
                self.restore_count += 1
                self.last_restore_at = time.time()
                if hasattr(adapter, "restore_sim_state"):
                    try:
                        adapter.restore_sim_state(command.payload)
                        adapter.stop_wheels()
                        if self.vision_processor is not None:
                            self.vision_processor.reset_filter()
                        self.last_restore_result = "ok"
                        self.last_restore_error = ""
                        self._publish(self._status(adapter, height_cm_ref[0], 0.0, 0, time.monotonic(), "", starting=False))
                    except Exception as exc:
                        self.last_restore_result = "error"
                        self.last_restore_error = str(exc)
                        self._publish(self._status(adapter, height_cm_ref[0], 0.0, 0, time.monotonic(), self.last_restore_error, starting=False))
                else:
                    self.last_restore_result = "unsupported"
                    self.last_restore_error = "Adapter does not support restore_sim_state."
                    self._publish(self._status(adapter, height_cm_ref[0], 0.0, 0, time.monotonic(), self.last_restore_error, starting=False))
            elif command.kind == "request_state":
                self._publish(self._status(adapter, height_cm_ref[0], 0.0, 0, time.monotonic(), "", starting=False))
            elif command.kind == "vision_control":
                payload = command.payload if isinstance(command.payload, dict) else {"action": str(command.payload)}
                action = str(payload.get("action", ""))
                details = dict(payload)
                details.pop("action", None)
                result = handle_vision_control(
                    args=self.args,
                    adapter=adapter,
                    scene_handle=scene_handle,
                    vision_processor=self.vision_processor,
                    action=action,
                    payload=details,
                    runtime_metrics={"started_wall": getattr(self, "_started_wall_for_guard", time.monotonic())},
                )
                if bool(result.get("respawned", False)):
                    self.respawn_count += 1
                    self.last_respawn_at = time.time()
                ok = bool(result.get("ok", False))
                text = str(result.get("text", ""))
                if not ok:
                    self.last_restore_error = text
                self._publish(self._status(adapter, height_cm_ref[0], 0.0, 0, time.monotonic(), "" if ok else text, starting=False))

    def _status(
        self,
        adapter: Any,
        height_cm: int,
        sim_time: float,
        sim_steps: int,
        started_wall: float,
        error: str,
        *,
        starting: bool,
    ) -> dict[str, Any]:
        elapsed_wall = max(1.0e-9, time.monotonic() - started_wall)
        sim_time = float(getattr(adapter, "sim_time", sim_time) or sim_time)
        sim_steps = int(getattr(adapter, "sim_steps", sim_steps) or sim_steps)
        status = {
            "ready": True,
            "starting": starting,
            "height_cm": int(height_cm),
            "scene_height_cm": int(height_cm),
            "obstacle_revision": int(self.obstacle_revision),
            "last_set_height_source": str(self.last_set_height_source or ""),
            "last_set_height_request_id": str(self.last_set_height_request_id or ""),
            "last_set_height_result": dict(self.last_set_height_result),
            "obstacle_updated": bool(self.last_set_height_result.get("obstacle_updated", False)),
            "scene_ready": bool(self.last_set_height_result.get("scene_ready", False)),
            "respawn_requested": bool(self.last_set_height_result.get("respawn_requested", False)),
            "respawned": bool(self.last_set_height_result.get("respawned", False)),
            "respawn_warning": str(self.last_set_height_result.get("respawn_warning", "") or ""),
            "error": error,
            "respawn_count": int(self.respawn_count),
            "last_respawn_at": float(self.last_respawn_at),
            "restore_count": int(self.restore_count),
            "last_restore_at": float(self.last_restore_at),
            "last_restore_result": str(self.last_restore_result or ""),
            "last_restore_error": str(self.last_restore_error or ""),
            "sim_time": float(sim_time),
            "sim_steps": int(sim_steps),
            "real_time_factor": float(sim_time) / elapsed_wall,
            "vision": self.vision_processor.status() if self.vision_processor is not None else self._default_vision_status(),
        }
        status.update(build_common_worker_status(args=self.args, adapter=adapter, scene_handle=None))
        return enrich_runtime_readiness(status)

    def _vision_config(self) -> VisionHeightConfig:
        return VisionHeightConfig(
            quantization_tolerance_cm=float(getattr(self.args, "vision_height_tolerance_cm", 2.0)),
            minimum_confidence=float(getattr(self.args, "vision_confidence_threshold", 0.75)),
            temporal_window_size=int(getattr(self.args, "vision_window_size", 7)),
            stable_frames_required=int(getattr(self.args, "vision_stable_frames", 5)),
        )

    def _default_vision_status(self) -> dict[str, Any]:
        enabled = bool(getattr(self.args, "onboard_camera", True))
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
            "stable_required": int(getattr(self.args, "vision_stable_frames", 5)),
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
            "source_mode": "generated",
            "roi_source": "generated_scene_x_prior",
            "consecutive_errors": 0,
        }
