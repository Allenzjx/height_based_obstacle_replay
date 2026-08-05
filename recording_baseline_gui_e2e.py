"""One visible Tk window and one real Isaac worker baseline evidence run."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from PIL import ImageGrab

from command_model import KNEE_JOINT_NAMES, WHEEL_JOINT_NAMES
from height_manifest import SUPPORTED_HEIGHTS_CM
from height_replay_ui import build_parser, normalize_motion_args
from height_sequence_store import HeightSequenceStore
from operation_coordinator import OperationState
from playback import plan_from_steps
from sequence_manager import SequenceManager
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


FORMAL_RECORDING_HEIGHTS_CM = tuple(height for height in SUPPORTED_HEIGHTS_CM if height > 0)


class RecordingBaselineGuiE2E:
    def __init__(self, output_dir: Path, *, timeout_s: float):
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_recording_root = Path(tempfile.mkdtemp(prefix="wlr_recording_baseline_e2e_"))
        args = build_parser().parse_args(
            [
                "--ui",
                "--height-cm",
                "5",
                "--worker-launch-mode",
                "explicit-python",
                "--worker-python-exe",
                r"C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe",
                "--sim-startup-timeout-s",
                str(timeout_s),
                "--no-telemetry",
                "--no-save-scene",
            ]
        )
        normalize_motion_args(args)
        self.controller = HeightReplayController(args)
        self.ui = RealRobotStyleHeightReplayUi(self.controller)
        self.ui.root.geometry("1700x980+0+0")
        self.ui.root.deiconify()
        self.ui.root.lift()
        self.timeout_s = float(timeout_s)
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = "WAIT_READY"
        self.result: dict[str, Any] = {
            "success": False,
            "started_at": time.time(),
            "visible_gui_requested": True,
            "reused_existing_process": False,
            "temporary_recording_root": str(self.temp_recording_root),
            "stages": [],
            "screenshots": {},
            "stop_rows": [],
            "height_rows": [],
            "respawn_rows": [],
        }
        self.stop_index = 0
        self.stop_context: dict[str, Any] = {}
        self.record_start_sim_time = 0.0
        self.motion_start_sim_time = 0.0
        self.pending_recorded_step: dict[str, Any] | None = None
        self.raw_plan = None
        self.fast_plan = None
        self.playback_runs: dict[str, Any] = {}
        self.playback_seen_active = False
        self.height_index = 0
        self.respawn_iteration = 0
        self.respawn_requested_count = 0

    def run(self) -> int:
        self.ui.root.after(50, self._start_worker)
        self.ui.root.after(200, self._tick)
        self.ui.run()
        self.result["ui_closed"] = True
        self.result["worker_reference_cleared"] = self.controller.sim_client is None
        self.result["finished_at"] = time.time()
        self._write_outputs()
        return 0 if self.result.get("success") else 1

    def _start_worker(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self._fail(exc)

    def _tick(self) -> None:
        if self.ui._closing:
            return
        try:
            elapsed = time.monotonic() - self.started
            if elapsed > self.timeout_s:
                raise TimeoutError(f"overall timeout at {self.stage} after {elapsed:.1f}s")
            if self.stage != "WAIT_READY" and time.monotonic() - self.stage_started > 180.0:
                raise TimeoutError(f"stage timeout: {self.stage}")
            phase = str(self.controller.latest_sim_status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(self.controller.latest_sim_status.get("error", phase)))
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(exc)
        if not self.ui._closing:
            self.ui.root.after(100, self._tick)

    def _advance(self, stage: str, **details: Any) -> None:
        self.result["stages"].append(
            {"stage": self.stage, "completed_at": time.time(), "details": details}
        )
        self.stage = stage
        self.stage_started = time.monotonic()

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        status = self.controller.latest_sim_status
        if not bool(status.get("first_visible_render_completed", False)):
            return
        self.result["isaac_pid"] = int(status.get("pid", 0) or 0)
        self.result["requested_headless"] = bool(status.get("requested_headless", True))
        self.result["effective_headless"] = bool(status.get("effective_headless", True))
        tabs = [str(self.ui.right_notebook.tab(tab_id, "text")) for tab_id in self.ui.right_notebook.tabs()]
        expected = [
            "Sim Connection",
            "Run Manager",
            "Record / Servo+Wheel",
            "Playback",
            "Height Generate",
            "Combine",
            "Sim State",
        ]
        if tabs != expected:
            raise AssertionError(f"unexpected tabs: {tabs}")
        retired_tab = "Speed" + " Scale"
        if retired_tab in tabs:
            raise AssertionError("retired percentage task is visible")
        self.result["tabs"] = tabs
        self.result["retired_percentage_tab_absent"] = True
        self.ui.open_right_tab("Record / Servo+Wheel")
        self.ui._refresh(force=True, sim_state=True)
        self._capture("ui_without_retired_task", "ui_without_retired_task.png")
        self._capture("new_recording_output_root", "new_recording_output_root.png")
        baseline = self.controller.validate_recording_baseline()
        self.result["authoritative_baseline_validation"] = copy.deepcopy(baseline)
        if not baseline.get("passed"):
            raise AssertionError("baseline validation failed: " + "; ".join(baseline.get("mismatches", [])))
        self.ui._refresh(force=True, sim_state=True)
        self._capture("baseline_validation_pass", "baseline_validation_pass.png")
        self._advance("START_STOP_CYCLE", tabs=tabs, baseline_id=baseline["baseline_id"])

    def _stage_start_stop_cycle(self) -> None:
        if self.stop_index >= 20:
            self._advance("PREPARE_TEMP_RECORDING")
            return
        self.stop_index += 1
        self.stop_context = {
            "test_index": self.stop_index,
            "requested_velocity_rad_s": 0.3,
            "start_requested_wall_time": time.time(),
        }
        self.ui._post("wheel all 0.3")
        self._advance("WAIT_WHEEL_MOVING", test_index=self.stop_index)

    def _stage_wait_wheel_moving(self) -> None:
        wheel = self._wheel_status()
        targets = dict(wheel.get("applied_target_rad_s", {}) or {})
        if not targets or max(abs(float(value)) for value in targets.values()) < 0.25:
            return
        self.stop_context["start_generation"] = int(wheel.get("generation", 0) or 0)
        self.stop_context["measured_velocity_before_stop_rad_s"] = copy.deepcopy(
            wheel.get("measured_velocity_rad_s", {})
        )
        if self.stop_index == 1:
            self.ui.open_right_tab("Record / Servo+Wheel")
            self.ui._refresh(force=True, sim_state=True)
            self._capture("wheel_actual_velocity_control", "wheel_actual_velocity_control.png")
        callback_started = time.perf_counter()
        click_wall = time.time()
        self.ui._stop_wheels_ui()
        callback_latency = time.perf_counter() - callback_started
        request = dict(self.controller.last_wheel_stop_request)
        self.stop_context.update(
            stop_click_wall_time=click_wall,
            ui_callback_latency_s=callback_latency,
            stop_command_id=str(request.get("command_id", "") or ""),
            stop_generation=int(request.get("wheel_generation", 0) or 0),
        )
        client = self.controller.sim_client
        if client is None:
            raise AssertionError("subprocess client disappeared during stop test")
        client.send_command(
            "wheel all 0.3",
            source="stale_generation_probe",
            wheel_generation=int(self.stop_context["start_generation"]),
            command_id=f"stale-{self.stop_index}-{uuid.uuid4().hex}",
            requested_wall_time=time.time(),
        )
        self._advance("WAIT_WHEEL_STOPPED", test_index=self.stop_index)

    def _stage_wait_wheel_stopped(self) -> None:
        wheel = self._wheel_status()
        if str(wheel.get("command_id", "") or "") != self.stop_context["stop_command_id"]:
            return
        if not bool(wheel.get("zero_target_applied", False)):
            return
        if not bool(wheel.get("physically_stopped", False)):
            return
        if not bool(wheel.get("stale_command_rejected", False)):
            return
        targets = dict(wheel.get("applied_target_rad_s", {}) or {})
        if not targets or any(abs(float(value)) > 1.0e-12 for value in targets.values()):
            raise AssertionError(f"zero target did not persist: {targets}")
        requested = float(wheel.get("requested_wall_time", 0.0) or 0.0)
        enqueued = float(wheel.get("enqueued_wall_time", 0.0) or 0.0)
        applied = float(wheel.get("target_applied_wall_time", 0.0) or 0.0)
        stopped = float(wheel.get("measured_stop_wall_time", 0.0) or 0.0)
        row = {
            **self.stop_context,
            "stop_command_requested_wall_time": requested,
            "zero_command_enqueue_wall_time": enqueued,
            "zero_target_applied_wall_time": applied,
            "measured_stop_wall_time": stopped,
            "ipc_enqueue_latency_s": max(0.0, enqueued - requested),
            "worker_apply_latency_s": max(0.0, applied - enqueued),
            "command_stop_latency_s": max(0.0, applied - float(self.stop_context["stop_click_wall_time"])),
            "physical_stop_latency_s": max(0.0, stopped - applied),
            "stop_tolerance_rad_s": float(wheel.get("stop_tolerance_rad_s", 0.0) or 0.0),
            "measured_velocity_after_stop_rad_s": copy.deepcopy(wheel.get("measured_velocity_rad_s", {})),
            "stale_command_detected": True,
            "pass": True,
        }
        self.result["stop_rows"].append(row)
        if self.stop_index == 1:
            self.controller.detail_text = json.dumps(wheel, indent=2, ensure_ascii=False, default=str)
            self.ui.open_right_tab("Sim State")
            self.ui._refresh(force=True, sim_state=True)
            self._capture("stop_acknowledgment", "stop_acknowledgment.png")
        self._advance("START_STOP_CYCLE", completed=self.stop_index)

    def _stage_prepare_temp_recording(self) -> None:
        self.controller.stop_wheels(reason="recording_test_prepare")
        self.respawn_requested_count = int(
            self.controller.latest_sim_status.get("respawn_count", 0) or 0
        )
        if not self.controller.respawn_robot(source="recording_test_prepare"):
            raise AssertionError(self.controller.status)
        self._advance("WAIT_TEMP_RECORDING_READY")

    def _stage_wait_temp_recording_ready(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        if int(self.controller.latest_sim_status.get("respawn_count", 0) or 0) <= self.respawn_requested_count:
            return
        wheel = self._wheel_status()
        if not (wheel.get("zero_target_applied") and wheel.get("physically_stopped")):
            return
        test_baseline = copy.deepcopy(self.controller.recording_baseline)
        test_baseline["recording_output_root"] = str(self.temp_recording_root)
        self.controller.recording_baseline = test_baseline
        self.controller.store = HeightSequenceStore(self.temp_recording_root)
        self.controller.manager = SequenceManager(
            self.controller.store.steps_path(self.controller.current_height_cm)
        )
        validation = self.controller.validate_recording_baseline()
        self.result["temporary_recording_validation"] = copy.deepcopy(validation)
        if not validation.get("passed"):
            raise AssertionError("temporary recording gate failed: " + "; ".join(validation.get("mismatches", [])))
        self.ui.open_right_tab("Record / Servo+Wheel")
        self.ui._refresh(force=True, sim_state=True)
        self.ui._post("step_record start")
        if self.controller.operation.state is not OperationState.RECORDING:
            raise AssertionError(self.controller.status)
        self.record_start_sim_time = self._sim_time()
        self._advance("WAIT_RECORD_IDLE")

    def _stage_wait_record_idle(self) -> None:
        if self._sim_time() - self.record_start_sim_time < 0.50:
            return
        self.ui._post("servo front_left_hip 10")
        self.ui._post("wheel all 0.3")
        self.motion_start_sim_time = self._sim_time()
        self._advance("WAIT_RECORD_MOTION")

    def _stage_wait_record_motion(self) -> None:
        if self._sim_time() - self.motion_start_sim_time < 1.20:
            return
        self.ui._stop_wheels_ui()
        self._advance("WAIT_RECORD_WHEEL_STOP")

    def _stage_wait_record_wheel_stop(self) -> None:
        wheel = self._wheel_status()
        if not (wheel.get("zero_target_applied") and wheel.get("physically_stopped")):
            return
        self.ui._post("step_record stop")
        self._advance("WAIT_RECORD_FINALIZED")

    def _stage_wait_record_finalized(self) -> None:
        if self.controller.record_stop_pending is not None or self.controller.pending_step is None:
            return
        step = copy.deepcopy(self.controller.pending_step)
        if step["events"][-1]["command"] != "wheel stop":
            raise AssertionError("recording has no final wheel stop boundary")
        if not step.get("wheel_stop_status", {}).get("zero_target_applied"):
            raise AssertionError("recording finalized without zero target acknowledgment")
        self.pending_recorded_step = step
        self.raw_plan = plan_from_steps([step], profile="raw", max_wheel_speed=self.controller.max_wheel_speed)
        self.fast_plan = plan_from_steps([step], profile="fast", max_wheel_speed=self.controller.max_wheel_speed)
        raw_signature = self._plan_signature(self.raw_plan)
        fast_signature = self._plan_signature(self.fast_plan)
        if raw_signature != fast_signature:
            raise AssertionError("Raw/Fast actuator signatures differ")
        if not self.fast_plan.final_time_s < self.raw_plan.final_time_s:
            raise AssertionError("Fast did not remove the intentional initial UI idle")
        self.result["recording"] = {
            "step": step,
            "raw_plan_signature": raw_signature,
            "fast_plan_signature": fast_signature,
            "raw_duration_s": self.raw_plan.final_time_s,
            "fast_duration_s": self.fast_plan.final_time_s,
            "output_root": str(self.controller.store.root),
            "saved_to_disk": False,
        }
        self.ui._post("step_record accept")
        if self.controller.manager.count != 1:
            raise AssertionError("temporary recorded step was not accepted in memory")
        self.controller.playback.set_profile("raw")
        self.playback_seen_active = False
        self.ui._post("play")
        self._advance("WAIT_RAW_PLAYBACK")

    def _stage_wait_raw_playback(self) -> None:
        worker = self._worker_playback()
        if bool(worker.get("active", False)):
            self.playback_seen_active = True
            return
        if not self.playback_seen_active:
            return
        if str(worker.get("stop_reason", "") or "") != "complete":
            raise AssertionError(f"Raw playback did not complete: {worker}")
        self.playback_runs["raw"] = copy.deepcopy(worker)
        self.controller.respawn_robot(source="between_playback_profiles")
        self.controller.playback.set_profile("fast")
        self.playback_seen_active = False
        self.ui._post("play fast")
        self._advance("WAIT_FAST_PLAYBACK")

    def _stage_wait_fast_playback(self) -> None:
        worker = self._worker_playback()
        if bool(worker.get("active", False)):
            self.playback_seen_active = True
            return
        if not self.playback_seen_active:
            return
        if str(worker.get("stop_reason", "") or "") != "complete":
            raise AssertionError(f"Fast playback did not complete: {worker}")
        self.playback_runs["fast"] = copy.deepcopy(worker)
        raw_commands = [row.get("command") for row in self.playback_runs["raw"].get("timing", {}).get("commands", [])]
        fast_commands = [row.get("command") for row in self.playback_runs["fast"].get("timing", {}).get("commands", [])]
        if raw_commands != fast_commands:
            raise AssertionError(f"executed Raw/Fast commands differ: {raw_commands} vs {fast_commands}")
        self.result["playback_runs"] = copy.deepcopy(self.playback_runs)
        self.controller.respawn_robot(source="knee_test_prepare")
        for name in sorted(KNEE_JOINT_NAMES):
            self.ui._post(f"servo {name} -60")
        self.motion_start_sim_time = self._sim_time()
        self._advance("WAIT_KNEES")

    def _stage_wait_knees(self) -> None:
        if self._sim_time() - self.motion_start_sim_time < 3.0:
            return
        status = self.controller.latest_sim_status
        targets = dict(status.get("target_joint_state", {}).get("servos", {}) or {})
        actual = dict(status.get("actual_joint_state", {}).get("servos", {}) or {})
        rows = []
        for name in sorted(KNEE_JOINT_NAMES):
            target_row = dict(targets.get(name, {}) or {})
            actual_deg = dict(actual.get(name, {}) or {}).get("deg")
            command_deg = target_row.get("command_deg")
            row = {
                "joint_name": name,
                "requested_command_deg": -60.0,
                "worker_command_deg": command_deg,
                "measured_actual_deg": actual_deg,
                "target_actual_deg": target_row.get("target_actual_deg"),
            }
            if command_deg is None or abs(float(command_deg) + 60.0) > 1.0e-6:
                raise AssertionError(f"knee command was clamped: {row}")
            rows.append(row)
        self.result["knee_minus_60_rows"] = rows
        self.controller.detail_text = json.dumps(rows, indent=2, ensure_ascii=False, default=str)
        self.ui.open_right_tab("Sim State")
        self.ui._refresh(force=True, sim_state=True)
        self._capture("four_knees_minus_60", "four_knees_minus_60.png")
        self.respawn_requested_count = int(
            self.controller.latest_sim_status.get("respawn_count", 0) or 0
        )
        if not self.controller.respawn_robot(source="height_geometry_prepare"):
            raise AssertionError(self.controller.status)
        self._advance("WAIT_HEIGHT_GEOMETRY_RESPAWN")

    def _stage_wait_height_geometry_respawn(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        if int(self.controller.latest_sim_status.get("respawn_count", 0) or 0) <= self.respawn_requested_count:
            return
        wheel = self._wheel_status()
        if not (wheel.get("zero_target_applied") and wheel.get("physically_stopped")):
            return
        self.controller.manager.adopt_steps([], dirty=False)
        self.height_index = 0
        self._advance("START_HEIGHT")

    def _stage_start_height(self) -> None:
        if self.height_index >= len(FORMAL_RECORDING_HEIGHTS_CM):
            self.controller.set_current_height(5, discard_dirty=True, load_steps=False, generate_obstacle=True)
            self._advance("WAIT_RESPAWN_BASE_HEIGHT")
            return
        height = int(FORMAL_RECORDING_HEIGHTS_CM[self.height_index])
        self.controller.set_current_height(
            height,
            discard_dirty=True,
            load_steps=False,
            generate_obstacle=True,
        )
        self._advance("WAIT_HEIGHT", height_cm=height)

    def _stage_wait_height(self) -> None:
        height = int(FORMAL_RECORDING_HEIGHTS_CM[self.height_index])
        if self.controller.operation.state is not OperationState.IDLE:
            return
        status = self.controller.latest_sim_status
        if int(status.get("scene_height_cm", -1) or -1) != height:
            return
        scene = dict(status.get("scene_baseline", {}) or {})
        if not scene.get("available"):
            raise AssertionError(f"scene metrics unavailable at {height}cm: {scene}")
        row = {"height_cm": height, **scene}
        self.result["height_rows"].append(row)
        self.result["respawn_rows"].append({"height_cm": height, "respawn_iteration": 1, **scene})
        if height == int(FORMAL_RECORDING_HEIGHTS_CM[-1]):
            self.controller.detail_text = json.dumps(row, indent=2, ensure_ascii=False, default=str)
            self.ui.open_right_tab("Sim State")
            self.ui._refresh(force=True, sim_state=True)
            self._capture("obstacle_distance", "obstacle_distance.png")
        self.height_index += 1
        self._advance("START_HEIGHT")

    def _stage_wait_respawn_base_height(self) -> None:
        if self.controller.operation.state is not OperationState.IDLE:
            return
        if int(self.controller.latest_sim_status.get("scene_height_cm", -1) or -1) != 5:
            return
        self.respawn_iteration = 1
        self._advance("START_RESPAWN")

    def _stage_start_respawn(self) -> None:
        if self.respawn_iteration >= 10:
            self._finish_success()
            return
        self.respawn_iteration += 1
        self.respawn_requested_count = int(self.controller.latest_sim_status.get("respawn_count", 0) or 0)
        self.controller.respawn_robot(source=f"determinism_{self.respawn_iteration}")
        self._advance("WAIT_RESPAWN", iteration=self.respawn_iteration)

    def _stage_wait_respawn(self) -> None:
        status = self.controller.latest_sim_status
        if int(status.get("respawn_count", 0) or 0) <= self.respawn_requested_count:
            return
        scene = dict(status.get("scene_baseline", {}) or {})
        if not scene.get("available"):
            raise AssertionError(f"scene metrics unavailable after respawn: {scene}")
        self.result["respawn_rows"].append(
            {"height_cm": 5, "respawn_iteration": self.respawn_iteration, **scene}
        )
        if self.respawn_iteration == 10:
            self.controller.detail_text = json.dumps(scene, indent=2, ensure_ascii=False, default=str)
            self.ui.open_right_tab("Sim State")
            self.ui._refresh(force=True, sim_state=True)
            self._capture("robot_respawn_pose", "robot_respawn_pose.png")
        self._advance("START_RESPAWN")

    def _finish_success(self) -> None:
        stops = list(self.result["stop_rows"])
        heights = list(self.result["height_rows"])
        respawns = list(self.result["respawn_rows"])
        formal_respawns = [row for row in respawns if int(row.get("height_cm", -1)) == 5]
        if len(stops) != 20 or not all(row["pass"] for row in stops):
            raise AssertionError("not all 20 real wheel stop cycles passed")
        front_values = [float(row["obstacle_front_face_x_m"]) for row in heights]
        root_distances = [float(row["root_to_obstacle_front_m"]) for row in heights]
        bottoms = [float(row["obstacle_bottom_z_m"]) for row in heights]
        roots = [list(row["robot_root_pose"]) for row in heights]
        formal_roots = [list(row["robot_root_pose"]) for row in formal_respawns]
        reference_root = list(
            self.result.get("authoritative_baseline_validation", {})
            .get("scene_baseline", {})
            .get("robot_root_pose", [])
            or []
        )
        reference_respawn_error = self._max_position_spread(
            [reference_root, formal_roots[0]] if formal_roots else []
        )
        collision_clearances = [
            float(list(row.get("robot_collision_bounds_min_m", [0.0, 0.0, -math.inf]))[2])
            for row in formal_respawns
        ]
        self.result["geometry_summary"] = {
            "front_face_max_error_m": max(front_values) - min(front_values),
            "approach_distance_max_error_m": max(root_distances) - min(root_distances),
            "bottom_max_error_m": max(bottoms) - min(bottoms),
            "root_position_max_pairwise_error_m": self._max_position_spread(roots),
            "height_count": len(heights),
            "respawn_count": len(formal_respawns),
            "respawn_root_position_max_pairwise_error_m": self._max_position_spread(formal_roots),
            "reference_to_respawn_position_error_m": reference_respawn_error,
            "minimum_robot_collision_clearance_m": min(collision_clearances) if collision_clearances else -math.inf,
        }
        if self.result["geometry_summary"]["front_face_max_error_m"] > 1.0e-4:
            raise AssertionError(f"obstacle front face moved: {front_values}")
        if self.result["geometry_summary"]["bottom_max_error_m"] > 1.0e-4:
            raise AssertionError(f"obstacle bottom moved: {bottoms}")
        if len(formal_respawns) != 10:
            raise AssertionError(f"expected 10 formal 5cm respawns, got {len(formal_respawns)}")
        if self.result["geometry_summary"]["minimum_robot_collision_clearance_m"] < -0.003:
            raise AssertionError(f"robot collision penetrated ground: {collision_clearances}")
        if self.result["geometry_summary"]["reference_to_respawn_position_error_m"] > 0.003:
            raise AssertionError(
                "respawn settled pose differs from authoritative reference: "
                f"{self.result['geometry_summary']['reference_to_respawn_position_error_m']:.6f}m"
            )
        self.result["success"] = True
        self.result["controller_status_log_tail"] = self.controller.status_log[-150:]
        self._write_outputs()
        self.ui.root.after(250, lambda: self.ui._window_close(force=True))

    def _fail(self, exc: Exception) -> None:
        if self.ui._closing:
            return
        self.result.update(
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            failed_stage=self.stage,
            latest_sim_status=copy.deepcopy(self.controller.latest_sim_status),
            controller_status_log_tail=self.controller.status_log[-150:],
        )
        try:
            self._capture("failure", "failure.png")
        except Exception:
            pass
        self._write_outputs()
        self.ui.root.after(250, lambda: self.ui._window_close(force=True))

    def _wheel_status(self) -> dict[str, Any]:
        return dict(self.controller.latest_sim_status.get("wheel_command", {}) or {})

    def _worker_playback(self) -> dict[str, Any]:
        return dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})

    def _sim_time(self) -> float:
        return float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)

    @staticmethod
    def _plan_signature(plan) -> list[dict[str, Any]]:
        return [
            {
                "command": event.command,
                "channel": event.channel,
                "planned_duration_s": event.planned_duration_s,
                "servo_targets": list(event.servo_targets),
                "servo_reference_velocity_deg_s": event.servo_base_velocity_deg_s,
                "wheel_requested_velocity_rad_s": list(event.wheel_requested_velocity_rad_s),
                "wheel_applied_target_rad_s": list(event.wheel_applied_target_rad_s),
                "wheel_active_duration_s": event.wheel_active_duration_s,
                "dispatch_command": event.dispatch_command,
            }
            for event in plan.events
        ]

    @staticmethod
    def _max_position_spread(poses: list[list[float]]) -> float:
        valid = [row for row in poses if len(row) >= 3]
        maximum = 0.0
        for left in valid:
            for right in valid:
                maximum = max(
                    maximum,
                    math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))),
                )
        return maximum

    def _capture(self, key: str, filename: str) -> None:
        self.ui.root.update_idletasks()
        self.ui.root.attributes("-topmost", True)
        self.ui.root.lift()
        self.ui.root.focus_force()
        self.ui.root.update()
        path = self.output_dir / filename
        try:
            ImageGrab.grab(window=int(self.ui.root.winfo_id())).save(path)
        finally:
            self.ui.root.attributes("-topmost", False)
        self.result["screenshots"][key] = str(path)

    def _write_outputs(self) -> None:
        (self.output_dir / "real_isaac_result.json").write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        self._write_csv(self.output_dir / "wheel_stop_latency_real.csv", self.result.get("stop_rows", []))
        self._write_csv(self.output_dir / "height_geometry_real.csv", self.result.get("height_rows", []))
        self._write_csv(self.output_dir / "respawn_distance_real.csv", self.result.get("respawn_rows", []))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    args = parser.parse_args()
    return RecordingBaselineGuiE2E(args.output, timeout_s=args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
