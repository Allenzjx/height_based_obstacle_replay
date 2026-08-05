"""Visible one-worker Isaac/Tk acceptance run for the speed/height/version refactor.

All recordings and version writes are redirected below the caller's report
directory.  The formal legacy recording roots are read-only for this run.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image, ImageGrab

from command_model import KNEE_JOINT_NAMES, SERVO_JOINT_NAMES, WHEEL_JOINT_NAMES
from height_replay_ui import build_parser, normalize_motion_args
from sim_ui_controller import HeightReplayController, RealRobotStyleHeightReplayUi


def _percentile(values: list[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(finite) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return finite[low]
    return finite[low] * (high - rank) + finite[high] * (rank - low)


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else 0.0


class RefactorGuiE2E:
    def __init__(self, output_dir: Path, *, timeout_s: float):
        self.output_dir = output_dir.resolve()
        self.screenshot_dir = self.output_dir / "screenshots"
        self.version_root = self.output_dir / "temporary_version_store_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        app_args = build_parser().parse_args(
            [
                "--ui",
                "--height-mm",
                "50",
                "--store-root",
                str(self.version_root),
                "--worker-launch-mode",
                "explicit-python",
                "--worker-python-exe",
                r"C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe",
                "--sim-startup-timeout-s",
                str(timeout_s),
                "--sim-worker-status-timeout-s",
                "30",
                "--no-save-scene",
                "--sim-state-json-on-demand",
            ]
        )
        normalize_motion_args(app_args)
        self.controller = HeightReplayController(app_args)
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
            "one_worker_requested": True,
            "reused_existing_process": False,
            "temporary_version_root": str(self.version_root),
            "stages": [],
            "screenshots": {},
            "callback_rows": [],
            "height_rows": [],
            "atomic_rows": [],
            "servo_rows": [],
            "wheel_rows": [],
            "record_play_rows": [],
            "version_rows": [],
            "ui_after_probe_ms": [],
            "ui_after_probe_rows": [],
            "rtf_samples": [],
            "worker_loop_hz_samples": [],
            "status_payload_bytes": [],
        }
        self.capture_in_progress = False
        self.ordinary_probe_enabled = False
        self.probe_after_id: Any | None = None
        self.last_probe = time.perf_counter()
        self.last_worker_sample: tuple[float, int] | None = None
        self.height_index = 0
        self.height_context: dict[str, Any] = {}
        self.atomic_context: dict[str, Any] = {}
        self.servo_percent_index = 0
        self.servo_percents = [100.0, 200.0]
        self.servo_context: dict[str, Any] = {}
        self.wheel_percent_index = 0
        self.wheel_percents = [100.0, 200.0]
        self.wheel_context: dict[str, Any] = {}
        self.record_percent_index = 0
        self.record_percents = [100.0, 200.0]
        self.record_context: dict[str, Any] = {}
        self.saved_version_ids: list[str] = []
        self.gif_frames: list[Image.Image] = []
        self.profile_tab_index = 0
        self.profile_speed_index = 0

    def run(self) -> int:
        self.ui.root.after(20, self._start_worker)
        self.probe_after_id = self.ui.root.after(10, self._probe_ui)
        self.ui.root.after(100, self._tick)
        self.ui.run()
        self.result["ui_closed"] = True
        self.result["worker_reference_cleared"] = self.controller.sim_client is None
        self.result["finished_at"] = time.time()
        self._write_result()
        return 0 if self.result.get("success") else 1

    def _start_worker(self) -> None:
        try:
            self.controller.start_sim_if_needed()
        except Exception as exc:
            self._fail(exc)

    def _probe_ui(self) -> None:
        if self.ui._closing:
            return
        now = time.perf_counter()
        gap_ms = (now - self.last_probe) * 1000.0
        self.last_probe = now
        if self.ordinary_probe_enabled and not self.capture_in_progress:
            self.result["ui_after_probe_ms"].append(gap_ms)
            self.result["ui_after_probe_rows"].append({"stage": self.stage, "gap_ms": gap_ms})
        self.probe_after_id = self.ui.root.after(10, self._probe_ui)

    def _sample_worker(self) -> None:
        status = self.controller.latest_sim_status
        if not status:
            return
        payload = status.get("status_payload_bytes")
        if payload is not None:
            self.result["status_payload_bytes"].append(float(payload))
        rtf = status.get("real_time_factor")
        if rtf is not None and str(status.get("phase", "")) == "running":
            self.result["rtf_samples"].append(float(rtf))
        try:
            steps = int(status.get("sim_steps", 0) or 0)
        except (TypeError, ValueError):
            return
        now = time.monotonic()
        if self.last_worker_sample is not None and steps != self.last_worker_sample[1]:
            dt = now - self.last_worker_sample[0]
            if dt > 0.0:
                self.result["worker_loop_hz_samples"].append((steps - self.last_worker_sample[1]) / dt)
        self.last_worker_sample = (now, steps)

    def _tick(self) -> None:
        if self.ui._closing:
            return
        try:
            elapsed = time.monotonic() - self.started
            if elapsed > self.timeout_s:
                raise TimeoutError(f"overall timeout at {self.stage} after {elapsed:.1f}s")
            if self.stage != "WAIT_READY" and time.monotonic() - self.stage_started > 150.0:
                raise TimeoutError(f"stage timeout: {self.stage}")
            phase = str(self.controller.latest_sim_status.get("phase", "") or "")
            if phase in {"preflight_failed", "launch_plan_failed", "process_spawn_failed", "runtime_failed"}:
                raise RuntimeError(str(self.controller.latest_sim_status.get("error", phase)))
            self._sample_worker()
            getattr(self, f"_stage_{self.stage.lower()}")()
        except Exception as exc:
            self._fail(exc)
        if not self.ui._closing:
            self.ui.root.after(50, self._tick)

    def _advance(self, stage: str, **details: Any) -> None:
        self.result["stages"].append(
            {"stage": self.stage, "completed_at": time.time(), "details": details}
        )
        self.stage = stage
        self.stage_started = time.monotonic()
        self.ordinary_probe_enabled = stage in {
            "PROFILE_UI",
            "GENERATE_START",
            "GENERATE_WAIT",
            "RESPAWN_START",
            "RESPAWN_WAIT",
            "ATOMIC_START",
            "ATOMIC_WAIT",
            "SERVO_RESET",
            "SERVO_WAIT_RESET",
            "SERVO_START",
            "SERVO_WAIT",
            "KNEE_START",
            "KNEE_WAIT",
            "RECORD_RUN",
            "RECORD_WAIT_PLAY",
        }

    def _stage_wait_ready(self) -> None:
        if not (self.controller.sim_connected and self.controller.runtime_ready):
            return
        status = self.controller.latest_sim_status
        tabs = [str(self.ui.right_notebook.tab(tab_id, "text")) for tab_id in self.ui.right_notebook.tabs()]
        expected = [
            "Sim Connection",
            "Run Manager",
            "Record / Servo+Wheel",
            "Speed Scale",
            "Playback",
            "Height Generate",
            "Combine",
            "Sim State",
        ]
        if tabs != expected:
            raise AssertionError(f"unexpected tabs: {tabs}")
        if any(name in tabs for name in ("Vision Auto Replay", "Stability Replay")):
            raise AssertionError("retired Vision/Stability tab is present")
        self.result.update(
            isaac_pid=int(status.get("pid", status.get("worker_pid", 0)) or 0),
            tabs=tabs,
            runtime_ready=True,
            control_ready=bool(status.get("control_ready", False)),
            requested_launch_mode=str(status.get("requested_launch_mode", "")),
        )
        self._capture_ui("main_ui", "main_ui.png")
        self.ui.open_right_tab("Speed Scale")
        self.ui._refresh(force=True)
        self._capture_ui("speed_scale", "speed_scale.png")
        self.ui.open_right_tab("Height Generate")
        self.ui._refresh(force=True)
        self._capture_ui("height_and_versions", "height_and_versions.png")
        self._capture_isaac("wider_obstacle", "wider_obstacle.png")
        # Startup and explicit screenshot work are not ordinary UI callbacks.
        self.result["ui_after_probe_ms"] = []
        self.result["rtf_samples"] = []
        self.result["worker_loop_hz_samples"] = []
        self.result["status_payload_bytes"] = []
        self.last_probe = time.perf_counter()
        self.last_worker_sample = None
        self._advance("PROFILE_UI")

    def _stage_profile_ui(self) -> None:
        tabs = list(self.ui.right_notebook.tabs())
        if self.profile_tab_index < 100:
            for _ in range(min(5, 100 - self.profile_tab_index)):
                index = self.profile_tab_index
                self.profile_tab_index += 1
                started = time.perf_counter()
                self.ui.right_notebook.select(tabs[index % len(tabs)])
                self.ui.root.update_idletasks()
                self._callback("tab_switch", started)
            return
        if self.profile_speed_index < 100:
            for _ in range(min(5, 100 - self.profile_speed_index)):
                index = self.profile_speed_index
                self.profile_speed_index += 1
                started = time.perf_counter()
                self.ui.speed_percent_var.set(float((index * 3) % 301))
                self.ui._schedule_speed_scale(str((index * 3) % 301))
                self._callback("speed_slider_debounce", started)
            return
        self.ui.speed_percent_var.set(100.0)
        self.ui._schedule_speed_scale("100")
        self.result["speed_slider_pending_after_100_drags"] = int(self.ui.speed_scale_after_id is not None)
        self.ui.root.geometry("1690x970+5+5")
        self.ui.root.update_idletasks()
        self.ui.root.geometry("1700x980+0+0")
        self.height_index = 0
        self._advance("GENERATE_START")

    def _stage_generate_start(self) -> None:
        if self.height_index >= 3:
            self._advance("RESPAWN_START")
            return
        height = (50, 75, 100)[self.height_index]
        self.controller.set_current_height(height, discard_dirty=True, load_steps=False, generate_obstacle=False)
        self.ui.height_var.set(f"{height} mm")
        status_before = dict(self.controller.latest_sim_status)
        started = time.perf_counter()
        request_id = self.controller.generate_or_update_height_obstacle()
        callback_ms = self._callback("generate_height", started, height_mm=height)
        self.height_context = {
            "height_mm": height,
            "request_id": request_id,
            "requested_wall": time.monotonic(),
            "callback_ms": callback_ms,
            "respawn_count_before": int(status_before.get("respawn_count", 0) or 0),
        }
        self._advance("GENERATE_WAIT", height_mm=height, request_id=request_id)

    def _stage_generate_wait(self) -> None:
        ack = self._find_ack("set_height", request_id=self.height_context["request_id"])
        if not ack:
            return
        wall_ms = (time.monotonic() - self.height_context["requested_wall"]) * 1000.0
        row = {
            **self.height_context,
            "worker_operation_ms": float(ack.get("worker_operation_s", 0.0) or 0.0) * 1000.0,
            "obstacle_update_ms": float(ack.get("obstacle_update_s", 0.0) or 0.0) * 1000.0,
            "ack_wall_ms": wall_ms,
            "control_ready_ms": wall_ms if bool(ack.get("control_ready", False)) else None,
            "control_ready": bool(ack.get("control_ready", False)),
            "respawned": bool(ack.get("respawned", False)),
            "respawn_count_after": int(self.controller.latest_sim_status.get("respawn_count", 0) or 0),
            "baseline_scan_ran": False,
            "revision": int(ack.get("revision", 0) or 0),
            "error": str(ack.get("error", "") or ""),
        }
        if row["error"]:
            raise AssertionError(row["error"])
        if row["respawned"] or row["respawn_count_after"] != row["respawn_count_before"]:
            raise AssertionError("ordinary Generate unexpectedly respawned the robot")
        if wall_ms >= 1500.0 or not row["control_ready"]:
            raise AssertionError(f"height update did not return ready in one-second class: {row}")
        self.result["height_rows"].append(row)
        if self.height_index == 2:
            self.ui.open_right_tab("Height Generate")
            self.ui._refresh(force=True)
            self._capture_ui("generate_control_ready", "generate_control_ready.png")
        self.height_index += 1
        self._advance("GENERATE_START")

    def _stage_respawn_start(self) -> None:
        self.controller.set_current_height(75, discard_dirty=True, load_steps=False, generate_obstacle=False)
        self.ui.height_var.set("75 mm")
        started = time.perf_counter()
        request_id = self.controller.generate_height_and_respawn()
        callback_ms = self._callback("generate_height_respawn", started, height_mm=75)
        self.height_context = {
            "height_mm": 75,
            "request_id": request_id,
            "requested_wall": time.monotonic(),
            "callback_ms": callback_ms,
        }
        self._advance("RESPAWN_WAIT", request_id=request_id)

    def _stage_respawn_wait(self) -> None:
        ack = self._find_ack("set_height_respawn", request_id=self.height_context["request_id"])
        if not ack:
            return
        wall_ms = (time.monotonic() - self.height_context["requested_wall"]) * 1000.0
        self.result["height_respawn"] = {
            **self.height_context,
            "ack_wall_ms": wall_ms,
            "respawned": bool(ack.get("respawned", False)),
            "control_ready": bool(ack.get("control_ready", False)),
            "error": str(ack.get("error", "") or ""),
        }
        if str(ack.get("error", "") or "") or wall_ms >= 3500.0 or not bool(ack.get("control_ready", False)):
            raise AssertionError(f"Generate + Respawn failed acceptance: {self.result['height_respawn']}")
        self._advance("ATOMIC_START")

    def _stage_atomic_start(self) -> None:
        self.controller.set_speed_percent(100.0)
        state = self.controller.transport.capture_command_state()
        servos = dict(state["servos"])
        servos["front_left_hip"] = -20.0 if servos.get("front_left_hip", 0.0) > -10.0 else 20.0
        wheels = {name: 0.30 for name in WHEEL_JOINT_NAMES}
        baseline = list(self.controller.latest_sim_status.get("servo_actual_deg", []) or [])
        started = time.perf_counter()
        batch_id = self.controller.apply_servo_wheel_together(servos, wheels, source="real_e2e")
        callback_ms = self._callback("apply_servo_wheel_together", started)
        self.atomic_context = {
            "batch_id": batch_id,
            "callback_ms": callback_ms,
            "baseline_servo": baseline,
            "observed_servo_start_sim": None,
            "observed_wheel_start_sim": None,
            "requested_wall": time.monotonic(),
        }
        self._advance("ATOMIC_WAIT", batch_id=batch_id)

    def _stage_atomic_wait(self) -> None:
        status = self.controller.latest_sim_status
        sim_time = float(status.get("sim_time", 0.0) or 0.0)
        actual = list(status.get("servo_actual_deg", []) or [])
        baseline = self.atomic_context["baseline_servo"]
        if self.atomic_context["observed_servo_start_sim"] is None and baseline and len(actual) == len(baseline):
            if any(a is not None and b is not None and abs(float(a) - float(b)) > 0.05 for a, b in zip(actual, baseline)):
                self.atomic_context["observed_servo_start_sim"] = sim_time
        measured = [value for value in list(status.get("wheel_measured_rad_s", []) or []) if value is not None]
        if self.atomic_context["observed_wheel_start_sim"] is None and measured and max(abs(float(value)) for value in measured) > 0.03:
            self.atomic_context["observed_wheel_start_sim"] = sim_time
        ack = self._find_ack("apply_motion_batch", batch_id=self.atomic_context["batch_id"])
        if not ack or self.atomic_context["observed_servo_start_sim"] is None or self.atomic_context["observed_wheel_start_sim"] is None:
            return
        row = {
            "batch_id": self.atomic_context["batch_id"],
            "callback_ms": self.atomic_context["callback_ms"],
            "servo_apply_tick": int(ack.get("first_physics_step", 0) or 0),
            "wheel_apply_tick": int(ack.get("first_physics_step", 0) or 0),
            "ack_servo_start_sim_time": float(ack.get("servo_motion_start_sim_time", 0.0) or 0.0),
            "ack_wheel_start_sim_time": float(ack.get("wheel_motion_start_sim_time", 0.0) or 0.0),
            "ack_start_difference_s": abs(float(ack.get("motion_start_skew_s", 0.0) or 0.0)),
            "observed_servo_start_sim_time": self.atomic_context["observed_servo_start_sim"],
            "observed_wheel_start_sim_time": self.atomic_context["observed_wheel_start_sim"],
            "observed_start_difference_s": abs(self.atomic_context["observed_servo_start_sim"] - self.atomic_context["observed_wheel_start_sim"]),
            "physics_dt_s": float(ack.get("physics_dt_s", 0.0) or 0.0),
            "servo_applied": bool(ack.get("servo_applied", False)),
            "wheel_applied": bool(ack.get("wheel_applied", False)),
            "pass": bool(ack.get("servo_applied", False) and ack.get("wheel_applied", False) and float(ack.get("motion_start_skew_s", 1.0)) <= float(ack.get("physics_dt_s", 0.0)) + 1e-9),
        }
        if not row["pass"]:
            raise AssertionError(f"atomic batch failed: {row}")
        self.result["atomic_rows"].append(row)
        self.ui.open_right_tab("Record / Servo+Wheel")
        self.ui._refresh(force=True)
        self._capture_ui("servo_wheel_simultaneous", "servo_wheel_simultaneous.png")
        self.controller.stop_wheels(reason="atomic_e2e_complete")
        self._advance("SERVO_RESET")

    def _stage_servo_reset(self) -> None:
        if self.servo_percent_index >= len(self.servo_percents):
            self._advance("WHEEL_BEFORE")
            return
        self.controller.set_speed_percent(100.0)
        state = self.controller.transport.capture_command_state()
        servos = dict(state["servos"])
        servos["front_left_hip"] = 0.0
        self.controller.apply_servo_wheel_together(servos, {name: 0.0 for name in WHEEL_JOINT_NAMES}, source="servo_reset")
        self.servo_context = {"reset_started": time.monotonic()}
        self._advance("SERVO_WAIT_RESET")

    def _stage_servo_wait_reset(self) -> None:
        actual = list(self.controller.latest_sim_status.get("servo_actual_deg", []) or [])
        if not actual or actual[0] is None or abs(float(actual[0])) > 2.0:
            if time.monotonic() - self.servo_context["reset_started"] < 4.0:
                return
        percent = self.servo_percents[self.servo_percent_index]
        self.controller.set_speed_percent(percent)
        state = self.controller.transport.capture_command_state()
        servos = dict(state["servos"])
        servos["front_left_hip"] = 60.0
        start_actual = float(actual[0]) if actual and actual[0] is not None else 0.0
        start_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        batch_id = self.controller.apply_servo_wheel_together(servos, {name: 0.0 for name in WHEEL_JOINT_NAMES}, source=f"servo_{percent:g}")
        self.servo_context = {
            "percent": percent,
            "batch_id": batch_id,
            "start_actual": start_actual,
            "start_sim": start_sim,
            "samples": [(start_sim, start_actual)],
            "velocity_samples": [],
            "applied_torque_samples": [],
            "computed_torque_samples": [],
            "effort_limit_samples": [],
        }
        self._advance("SERVO_RUN")

    def _stage_servo_run(self) -> None:
        status = self.controller.latest_sim_status
        actual = list(status.get("servo_actual_deg", []) or [])
        if not actual or actual[0] is None:
            return
        sim_time = float(status.get("sim_time", 0.0) or 0.0)
        value = float(actual[0])
        actual_state = dict(status.get("actual_joint_state", {}) or {})
        servo_state = dict(dict(actual_state.get("servos", {}) or {}).get("front_left_hip", {}) or {})
        for source_key, target_key in (
            ("velocity_deg_s", "velocity_samples"),
            ("applied_torque_nm", "applied_torque_samples"),
            ("computed_torque_nm", "computed_torque_samples"),
            ("effort_limit_nm", "effort_limit_samples"),
        ):
            sample = servo_state.get(source_key)
            if sample is not None:
                self.servo_context[target_key].append(float(sample))
        samples = self.servo_context["samples"]
        if not samples or sim_time > samples[-1][0]:
            samples.append((sim_time, value))
        elapsed = sim_time - self.servo_context["start_sim"]
        if abs(value - 60.0) > 2.0 and elapsed < 2.5:
            return
        rates = [
            abs((b[1] - a[1]) / (b[0] - a[0]))
            for a, b in zip(samples, samples[1:])
            if b[0] > a[0] and abs(b[1] - a[1]) > 0.01
        ]
        moving_velocity = [abs(value) for value in self.servo_context["velocity_samples"] if abs(value) > 1.0]
        applied_torque = [abs(value) for value in self.servo_context["applied_torque_samples"]]
        computed_torque = [abs(value) for value in self.servo_context["computed_torque_samples"]]
        effort_limits = [abs(value) for value in self.servo_context["effort_limit_samples"] if abs(value) > 0.0]
        effort_limit = max(effort_limits) if effort_limits else None
        applied_peak = max(applied_torque or [0.0])
        computed_peak = max(computed_torque or [0.0])
        ack = self._find_ack("apply_motion_batch", batch_id=self.servo_context["batch_id"])
        row = {
            "speed_percent": self.servo_context["percent"],
            "target_deg": 60.0,
            "start_actual_deg": self.servo_context["start_actual"],
            "final_actual_deg": value,
            "target_error_deg": value - 60.0,
            "elapsed_sim_s": elapsed,
            "actual_average_velocity_deg_s": _mean(moving_velocity),
            "actual_peak_velocity_deg_s": max(moving_velocity or rates or [0.0]),
            "position_net_average_velocity_deg_s": abs(value - self.servo_context["start_actual"]) / max(elapsed, 1e-9),
            "applied_torque_peak_nm": applied_peak,
            "computed_torque_peak_nm": computed_peak,
            "effort_limit_nm": effort_limit,
            "effort_saturated": bool(effort_limit is not None and (applied_peak >= 0.98 * effort_limit or computed_peak > effort_limit)),
            "requested_velocity_deg_s": float(ack.get("requested_servo_velocity_deg_s", 0.0) or 0.0),
            "effective_velocity_deg_s": float(ack.get("effective_servo_velocity_deg_s", 0.0) or 0.0),
            "actuator_limit": self.controller.motion_reference.servo_velocity_limit_deg_s,
            "samples": [{"sim_time": sample[0], "actual_deg": sample[1]} for sample in samples],
        }
        self.result["servo_rows"].append(row)
        self.servo_percent_index += 1
        self._advance("SERVO_RESET")

    def _stage_wheel_before(self) -> None:
        if self.wheel_percent_index >= len(self.wheel_percents):
            self._advance("KNEE_START")
            return
        stop_request = self.controller.stop_wheels(reason="wheel_path_boundary")
        self.controller.set_speed_percent(self.wheel_percents[self.wheel_percent_index])
        self.wheel_context = {
            "percent": self.wheel_percents[self.wheel_percent_index],
            "boundary_command_id": str(stop_request.get("command_id", "") or ""),
        }
        self._advance("WHEEL_WAIT_BOUNDARY")

    def _stage_wheel_wait_boundary(self) -> None:
        if not self._find_stop_ack(self.wheel_context["boundary_command_id"]):
            return
        self.controller.transport.request_state(detailed=True)
        self.wheel_context["detail_requested"] = time.monotonic()
        self._advance("WHEEL_WAIT_BEFORE")

    def _stage_wheel_wait_before(self) -> None:
        detail = self._detailed_status()
        if not detail:
            return
        before = dict(detail.get("sim_state", {}) or {})
        state = self.controller.transport.capture_command_state()
        batch_id = self.controller.apply_servo_wheel_together(
            dict(state["servos"]),
            {name: 0.30 for name in WHEEL_JOINT_NAMES},
            source=f"wheel_{self.wheel_percents[self.wheel_percent_index]:g}",
        )
        self.wheel_context.update(before=before, batch_id=batch_id, measured=[])
        self._advance("WHEEL_WAIT_APPLY")

    def _stage_wheel_wait_apply(self) -> None:
        ack = self._find_ack("apply_motion_batch", batch_id=self.wheel_context["batch_id"])
        if not ack:
            return
        self.wheel_context["apply_ack"] = ack
        self.wheel_context["start_sim"] = float(ack.get("batch_applied_sim_time", 0.0) or 0.0)
        self._advance("WHEEL_RUN")

    def _stage_wheel_run(self) -> None:
        status = self.controller.latest_sim_status
        sim_time = float(status.get("sim_time", 0.0) or 0.0)
        measured = [float(value) for value in list(status.get("wheel_measured_rad_s", []) or []) if value is not None]
        if measured:
            self.wheel_context["measured"].append(_mean([abs(value) for value in measured]))
        if sim_time - self.wheel_context["start_sim"] < 1.0:
            return
        stop_request = self.controller.stop_wheels(reason="wheel_path_duration_complete")
        self.wheel_context["stop_command_id"] = str(stop_request.get("command_id", "") or "")
        self.wheel_context["stop_requested"] = time.monotonic()
        self._advance("WHEEL_WAIT_STOP")

    def _stage_wheel_wait_stop(self) -> None:
        stop_ack = self._find_stop_ack(self.wheel_context["stop_command_id"])
        if not stop_ack:
            return
        self.wheel_context["stop_ack"] = stop_ack
        self.controller.transport.request_state(detailed=True)
        self._advance("WHEEL_WAIT_AFTER")

    def _stage_wheel_wait_after(self) -> None:
        detail = self._detailed_status()
        if not detail:
            return
        after = dict(detail.get("sim_state", {}) or {})
        before_positions = self._joint_positions(self.wheel_context["before"])
        after_positions = self._joint_positions(after)
        signed = {
            name: after_positions[name] - before_positions[name]
            for name in WHEEL_JOINT_NAMES
            if name in before_positions and name in after_positions
        }
        before_xy = self._root_xy(self.wheel_context["before"])
        after_xy = self._root_xy(after)
        body = math.hypot(after_xy[0] - before_xy[0], after_xy[1] - before_xy[1]) if before_xy and after_xy else None
        ack = dict(self.wheel_context.get("apply_ack", {}) or {})
        active_duration = (
            float(dict(self.wheel_context.get("stop_ack", {}) or {}).get("target_applied_sim_time", 0.0) or 0.0)
            - float(self.wheel_context["start_sim"])
        )
        effective_velocity = abs(float(dict(ack.get("effective_wheel_velocity_rad_s", {}) or {}).get(WHEEL_JOINT_NAMES[0], 0.0)))
        row = {
            "speed_percent": self.wheel_context["percent"],
            "canonical_velocity_rad_s": 0.30,
            "commanded_effective_velocity_rad_s": effective_velocity,
            "measured_velocity_rad_s": _mean(self.wheel_context["measured"]),
            "active_duration_s": active_duration,
            "signed_joint_displacement_rad": signed,
            "mean_abs_joint_displacement_rad": _mean([abs(value) for value in signed.values()]),
            "theoretical_joint_displacement_rad": effective_velocity * active_duration,
            "theoretical_rolling_path_m": None,
            "robot_body_displacement_m": body,
            "slip_ratio": None,
            "slip_status": "UNVERIFIED: wheel radius/transmission unavailable",
        }
        self.result["wheel_rows"].append(row)
        self.wheel_percent_index += 1
        self._advance("WHEEL_BEFORE")

    def _stage_knee_start(self) -> None:
        self.controller.set_speed_percent(100.0)
        state = self.controller.transport.capture_command_state()
        servos = dict(state["servos"])
        for name in KNEE_JOINT_NAMES:
            servos[name] = -60.0
        batch_id = self.controller.apply_servo_wheel_together(
            servos,
            {name: 0.0 for name in WHEEL_JOINT_NAMES},
            source="knee_minus_60_real_e2e",
        )
        self.result["knee_batch_id"] = batch_id
        self._advance("KNEE_WAIT")

    def _stage_knee_wait(self) -> None:
        ack = self._find_ack("apply_motion_batch", batch_id=self.result["knee_batch_id"])
        if not ack:
            return
        canonical = dict(ack.get("canonical_servo_targets_deg", {}) or {})
        values = {name: canonical.get(name) for name in sorted(KNEE_JOINT_NAMES)}
        passed = all(value is not None and abs(float(value) + 60.0) < 1e-9 for value in values.values())
        self.result["knee_minus_60"] = {"targets_deg": values, "all_four_applied": passed}
        if not passed:
            raise AssertionError(f"knee targets were not all -60: {values}")
        self.ui.open_right_tab("Record / Servo+Wheel")
        self.ui._refresh(force=True)
        self._capture_ui("four_knees_minus_60", "four_knees_minus_60.png")
        self.record_percent_index = 0
        self._advance("VERSION_HEIGHT_START")

    def _stage_version_height_start(self) -> None:
        self.controller.set_current_height(50, discard_dirty=True, load_steps=False, generate_obstacle=False)
        self.ui.height_var.set("50 mm")
        request_id = self.controller.generate_or_update_height_obstacle()
        self.height_context = {"height_mm": 50, "request_id": request_id}
        self._advance("VERSION_HEIGHT_WAIT")

    def _stage_version_height_wait(self) -> None:
        ack = self._find_ack("set_height", request_id=self.height_context["request_id"])
        if not ack:
            return
        if str(ack.get("error", "") or "") or not bool(ack.get("control_ready", False)):
            raise AssertionError(f"50 mm version setup failed: {ack}")
        self._advance("RECORD_START")

    def _stage_record_start(self) -> None:
        if self.record_percent_index >= len(self.record_percents):
            self._advance("LOAD_VERSIONS")
            return
        worker_playback = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if self.controller.mode in {"PLAYBACK", "PLAYBACK_PAUSED"} or bool(worker_playback.get("active", False)):
            return
        percent = self.record_percents[self.record_percent_index]
        self.controller.set_speed_percent(percent)
        self.controller.start_step_recording()
        if not self.controller.recording_active:
            raise AssertionError(self.controller.status)
        state = self.controller.transport.capture_command_state()
        servos = dict(state["servos"])
        servos["front_left_hip"] = 15.0 if self.record_percent_index == 0 else 30.0
        batch_id = self.controller.apply_servo_wheel_together(
            servos,
            {name: 0.25 for name in WHEEL_JOINT_NAMES},
            source="real_recording",
        )
        self.record_context = {
            "percent": percent,
            "batch_id": batch_id,
            "start_sim": float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0),
        }
        self._advance("RECORD_RUN")

    def _stage_record_run(self) -> None:
        now_sim = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        if now_sim - self.record_context["start_sim"] < 2.0:
            return
        self.controller.stop_step_recording()
        self._advance("RECORD_WAIT_PENDING")

    def _stage_record_wait_pending(self) -> None:
        if self.controller.pending_step is None:
            return
        pending = json.loads(json.dumps(self.controller.pending_step, default=str))
        batch_events = [event for event in pending.get("events", []) if event.get("kind") == "motion_batch"]
        if len(batch_events) != 1:
            raise AssertionError(f"recorded Servo+Wheel batch count={len(batch_events)}")
        event = batch_events[0]
        canonical = dict(event.get("canonical_wheel_velocity_rad_s", {}) or {})
        effective = dict(event.get("effective_wheel_velocity_rad_s", {}) or {})
        self.controller.accept_pending_step()
        if self.controller.pending_step is not None:
            raise AssertionError("recorded step did not accept")
        callback_started = time.perf_counter()
        self.ui._save_new_version_async()
        callback_ms = self._callback("save_new_version_icon", callback_started, speed_percent=self.record_context["percent"])
        self.record_context.update(
            pending=pending,
            recorded_event=event,
            canonical=canonical,
            effective=effective,
            save_callback_ms=callback_ms,
        )
        self._advance("RECORD_WAIT_SAVE")

    def _stage_record_wait_save(self) -> None:
        if self.ui.save_in_progress:
            return
        report = dict(self.controller.last_save_report)
        version_id = str(report.get("version_id", "") or "")
        if not version_id or version_id in self.saved_version_ids:
            return
        self.saved_version_ids.append(version_id)
        self.result["version_rows"].append(dict(report))
        step = self.controller.manager.steps[-1]
        if not self.controller.start_playback([step], label=f"recorded-{self.record_context['percent']:g}", profile="raw"):
            raise AssertionError(self.controller.status)
        self.record_context["version_id"] = version_id
        self.record_context["play_start_sim"] = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        self.record_context["play_seen_active"] = False
        self._advance("RECORD_WAIT_PLAY")

    def _stage_record_wait_play(self) -> None:
        worker = dict(self.controller.latest_sim_status.get("worker_playback", {}) or {})
        if bool(worker.get("active", False)):
            self.record_context["play_seen_active"] = True
            live_batch = dict(self.controller.latest_sim_status.get("motion_batch", {}) or {})
            live_canonical = dict(live_batch.get("canonical_wheel_velocity_rad_s", {}) or {})
            if (
                str(live_batch.get("source", "") or "") == "playback"
                and live_canonical == dict(self.record_context["recorded_event"].get("canonical_wheel_velocity_rad_s", {}) or {})
            ):
                self.record_context["playback_motion_batch"] = live_batch
            if self.record_percent_index == 1 and not self.record_context.get("interaction_done"):
                if "interaction_index" not in self.record_context:
                    self.record_context["interaction_index"] = 0
                    self.record_context["interaction_sim_before"] = float(
                        self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0
                    )
                tabs = list(self.ui.right_notebook.tabs())
                start_index = int(self.record_context["interaction_index"])
                for index in range(start_index, min(100, start_index + 5)):
                    self.ui.right_notebook.select(tabs[index % len(tabs)])
                    self.ui.speed_percent_var.set(150.0 + float(index % 2) * 50.0)
                    self.ui._schedule_speed_scale(str(self.ui.speed_percent_var.get()))
                self.record_context["interaction_index"] = min(100, start_index + 5)
                if self.record_context["interaction_index"] >= 100:
                    self.ui.root.geometry("1680x960+10+10")
                    self.ui.root.update_idletasks()
                    self.ui.root.geometry("1700x980+0+0")
                    self.record_context["interaction_done"] = True
            return
        if not self.record_context.get("play_seen_active"):
            return
        play_end = float(self.controller.latest_sim_status.get("sim_time", 0.0) or 0.0)
        event = self.record_context["recorded_event"]
        playback_batch = dict(self.record_context.get("playback_motion_batch", {}) or {})
        if not playback_batch:
            raise AssertionError("playback canonical MotionBatch was not observed in lightweight status")
        canonical_recorded = dict(event.get("canonical_wheel_velocity_rad_s", {}) or {})
        canonical_playback = dict(playback_batch.get("canonical_wheel_velocity_rad_s", {}) or {})
        effective_playback = dict(playback_batch.get("effective_wheel_velocity_rad_s", {}) or {})
        executor_scale = float(playback_batch.get("executor_speed_percent", 100.0) or 100.0) / 100.0
        double_scaled = any(
            abs(float(effective_playback.get(name, 0.0)) - float(value) * executor_scale) > 1.0e-9
            for name, value in canonical_playback.items()
        )
        row = {
            "speed_percent": self.record_context["percent"],
            "version_id": self.record_context["version_id"],
            "recorded_batch_id": self.record_context["batch_id"],
            "recorded_canonical_wheel_rad_s": canonical_recorded,
            "recorded_effective_wheel_rad_s": self.record_context["effective"],
            "playback_canonical_wheel_rad_s": canonical_playback,
            "playback_effective_wheel_rad_s": effective_playback,
            "playback_executor_speed_percent": playback_batch.get("executor_speed_percent"),
            "playback_elapsed_sim_s": play_end - self.record_context["play_start_sim"],
            "double_scaled": double_scaled,
            "canonical_match": canonical_playback == canonical_recorded,
        }
        if row["double_scaled"]:
            raise AssertionError(f"playback double-scaled: {row}")
        if self.record_context.get("interaction_done"):
            row["ui_interaction_sim_advanced_s"] = play_end - float(self.record_context["interaction_sim_before"])
        self.result["record_play_rows"].append(row)
        self.record_percent_index += 1
        self._advance("RECORD_START")

    def _stage_load_versions(self) -> None:
        if len(self.saved_version_ids) != 2:
            raise AssertionError(f"expected two versions, found {self.saved_version_ids}")
        loaded: list[dict[str, Any]] = []
        for version_id in self.saved_version_ids:
            count = self.controller.load_steps_for_current_height(discard_dirty=True, version_id=version_id)
            metadata = dict(self.controller.current_version_metadata)
            loaded.append(
                {
                    "version_id": version_id,
                    "count": count,
                    "sha256": metadata.get("accepted_steps_sha256"),
                    "path": str(self.controller.store.version_dir(50, version_id)),
                }
            )
        if loaded[0]["sha256"] == loaded[1]["sha256"]:
            raise AssertionError("v001 and v002 hashes unexpectedly match")
        if not all(Path(row["path"]).is_dir() for row in loaded):
            raise AssertionError(f"saved version missing: {loaded}")
        self.result["version_load_validation"] = loaded
        self.ui.open_right_tab("Height Generate")
        self.ui._refresh(force=True)
        self._capture_ui("two_immutable_versions", "two_immutable_versions.png")
        self._advance("FINALIZE")

    def _stage_finalize(self) -> None:
        self.controller.stop_wheels(reason="final_e2e_safety_stop")
        status = self.controller.latest_sim_status
        probes = list(self.result["ui_after_probe_ms"])
        ordinary = [row["elapsed_ms"] for row in self.result["callback_rows"]]
        rtf = list(self.result["rtf_samples"])
        loop_hz = list(self.result["worker_loop_hz_samples"])
        sizes = list(self.result["status_payload_bytes"])
        perf = dict(status.get("perf", {}) or {})
        servo_rows = list(self.result["servo_rows"])
        wheel_rows = list(self.result["wheel_rows"])
        wheel_path_ratio = (
            float(wheel_rows[1]["mean_abs_joint_displacement_rad"])
            / max(1.0e-9, float(wheel_rows[0]["mean_abs_joint_displacement_rad"]))
            if len(wheel_rows) == 2
            else 0.0
        )
        servo_average_speed_ratio = (
            float(servo_rows[1]["actual_average_velocity_deg_s"])
            / max(1.0e-9, float(servo_rows[0]["actual_average_velocity_deg_s"]))
            if len(servo_rows) == 2
            else 0.0
        )
        servo_peak_speed_ratio = (
            float(servo_rows[1]["actual_peak_velocity_deg_s"])
            / max(1.0e-9, float(servo_rows[0]["actual_peak_velocity_deg_s"]))
            if len(servo_rows) == 2
            else 0.0
        )
        self.result["performance_summary"] = {
            "ordinary_callback_count": len(ordinary),
            "ordinary_callback_average_ms": _mean(ordinary),
            "ordinary_callback_p95_ms": _percentile(ordinary, 95.0),
            "ordinary_callback_max_ms": max(ordinary or [0.0]),
            "ui_after_probe_average_ms": _mean(probes),
            "ui_after_probe_p95_ms": _percentile(probes, 95.0),
            "ui_after_probe_max_ms": max(probes or [0.0]),
            "real_time_factor_average": _mean(rtf[-80:]),
            "real_time_factor_p50": _percentile(rtf[-80:], 50.0),
            "worker_loop_hz_average": _mean(loop_hz[-80:]),
            "status_payload_bytes_average": _mean(sizes[-80:]),
            "status_payload_bytes_max": max(sizes or [0.0]),
            "wheel_200_to_100_joint_path_ratio": wheel_path_ratio,
            "servo_200_to_100_average_speed_ratio": servo_average_speed_ratio,
            "servo_200_to_100_peak_speed_ratio": servo_peak_speed_ratio,
            **perf,
        }
        criteria = {
            "callback_p95_under_30ms": self.result["performance_summary"]["ordinary_callback_p95_ms"] < 30.0,
            "callback_max_under_100ms": self.result["performance_summary"]["ordinary_callback_max_ms"] < 100.0,
            "no_ui_probe_over_200ms": self.result["performance_summary"]["ui_after_probe_max_ms"] < 200.0,
            "status_under_16kb": self.result["performance_summary"]["status_payload_bytes_max"] < 16 * 1024,
            "rtf_at_least_0_9": self.result["performance_summary"]["real_time_factor_p50"] >= 0.9,
            "all_generate_under_1_5s": all(row["ack_wall_ms"] < 1500.0 for row in self.result["height_rows"]),
            "atomic_batch_pass": all(row["pass"] for row in self.result["atomic_rows"]),
            "two_versions": len(self.saved_version_ids) == 2,
            "four_knees": bool(self.result.get("knee_minus_60", {}).get("all_four_applied", False)),
            "servo_target_error_within_2_5deg": len(servo_rows) == 2 and all(abs(float(row["target_error_deg"])) <= 2.5 for row in servo_rows),
            "servo_200_actual_speed_increased": len(servo_rows) == 2 and (
                servo_average_speed_ratio > 1.2 or servo_peak_speed_ratio > 1.2
            ),
            "wheel_200_path_ratio_within_10pct": 1.8 <= wheel_path_ratio <= 2.2,
            "record_play_single_scaling": len(self.result["record_play_rows"]) == 2 and all(row["canonical_match"] and not row["double_scaled"] for row in self.result["record_play_rows"]),
        }
        self.result["acceptance_criteria"] = criteria
        if not all(criteria.values()):
            raise AssertionError(f"acceptance criteria failed: {criteria}")
        self.result["final_wheel_stop_requested"] = True
        self.result["success"] = True
        self._save_gif()
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(400, lambda: self.ui._window_close(force=True))

    def _callback(self, name: str, started: float, **extra: Any) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.result["callback_rows"].append({"name": name, "elapsed_ms": elapsed_ms, **extra})
        return elapsed_ms

    def _find_ack(self, operation: str, *, request_id: str = "", batch_id: str = "") -> dict[str, Any]:
        status = self.controller.latest_sim_status
        rows = list(status.get("operation_ack_history", []) or [])
        latest = status.get("last_operation_ack")
        if isinstance(latest, dict):
            rows.append(latest)
        for row in reversed(rows):
            if str(row.get("operation", "") or "") != operation:
                continue
            if request_id and str(row.get("request_id", "") or "") != request_id:
                continue
            if batch_id and str(row.get("batch_id", "") or "") != batch_id:
                continue
            return dict(row)
        return {}

    def _find_stop_ack(self, command_id: str) -> dict[str, Any]:
        status = self.controller.latest_sim_status
        rows = list(status.get("operation_ack_history", []) or [])
        latest = status.get("last_operation_ack")
        if isinstance(latest, dict):
            rows.append(latest)
        for row in reversed(rows):
            if str(row.get("type", "") or "") != "stop_ack":
                continue
            if str(row.get("command_id", "") or "") == str(command_id or ""):
                return dict(row)
        return {}

    def _detailed_status(self) -> dict[str, Any]:
        client = self.controller.sim_client
        return dict(getattr(client, "latest_detailed_status", {}) or {}) if client is not None else {}

    @staticmethod
    def _joint_positions(sim_state: dict[str, Any]) -> dict[str, float]:
        names = [str(name) for name in list(sim_state.get("joint_names", []) or [])]
        values: Any = sim_state.get("joint_pos")
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        if not names or not isinstance(values, list):
            return {}
        return {name: float(values[index]) for index, name in enumerate(names) if index < len(values)}

    @staticmethod
    def _root_xy(sim_state: dict[str, Any]) -> tuple[float, float] | None:
        values: Any = sim_state.get("root_pose")
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError, IndexError):
            return None

    def _capture_ui(self, key: str, filename: str) -> None:
        self.capture_in_progress = True
        try:
            self.ui.root.update_idletasks()
            self.ui.root.lift()
            self.ui.root.attributes("-topmost", True)
            self.ui.root.update()
            x = self.ui.root.winfo_rootx()
            y = self.ui.root.winfo_rooty()
            width = self.ui.root.winfo_width()
            height = self.ui.root.winfo_height()
            image = ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True)
            path = self.screenshot_dir / filename
            image.save(path)
            self.result["screenshots"][key] = str(path)
            if len(self.gif_frames) < 12:
                self.gif_frames.append(image.copy().convert("P", palette=Image.Palette.ADAPTIVE))
        finally:
            try:
                self.ui.root.attributes("-topmost", False)
            except Exception:
                pass
            self.capture_in_progress = False
            self.last_probe = time.perf_counter()

    def _capture_isaac(self, key: str, filename: str) -> None:
        if not hasattr(ctypes, "windll"):
            return
        user32 = ctypes.windll.user32
        matches: list[tuple[int, str]] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def enum_window(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value
            lowered = title.lower()
            if "isaac sim" in lowered and "height based obstacle replay" not in lowered:
                matches.append((int(hwnd), title))
            return True

        user32.EnumWindows(enum_window, 0)
        if not matches:
            self.result["isaac_window_capture_warning"] = "No separate Isaac window title was found."
            return
        hwnd, title = matches[0]
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if rect.right <= rect.left or rect.bottom <= rect.top:
            return
        self.capture_in_progress = True
        try:
            user32.ShowWindow(hwnd, 9)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            self.ui.root.update_idletasks()
            image = ImageGrab.grab(
                bbox=(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)),
                all_screens=True,
            )
            path = self.screenshot_dir / filename
            image.save(path)
            self.result["screenshots"][key] = str(path)
            self.result["isaac_window_title"] = title
        finally:
            self.capture_in_progress = False
            self.last_probe = time.perf_counter()
            self.ui.root.lift()

    def _save_gif(self) -> None:
        if len(self.gif_frames) < 2:
            return
        path = self.output_dir / "validation_motion.gif"
        frames = [frame.resize((850, 490)) for frame in self.gif_frames]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=650, loop=0)
        self.result["video_or_animation"] = str(path)

    def _cancel_probe(self) -> None:
        if self.probe_after_id is None:
            return
        try:
            self.ui.root.after_cancel(self.probe_after_id)
        except Exception:
            pass
        self.probe_after_id = None

    def _fail(self, exc: Exception) -> None:
        if self.ui._closing:
            return
        self.result.update(
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            failed_stage=self.stage,
            traceback=traceback.format_exc(),
            latest_sim_status=self.controller.latest_sim_status,
            controller_status_log_tail=self.controller.status_log[-120:],
        )
        try:
            self.controller.stop_wheels(reason="e2e_failure")
            self._capture_ui("failure", "failure.png")
        except Exception:
            pass
        self._write_result()
        self._cancel_probe()
        self.ui.root.after(300, lambda: self.ui._window_close(force=True))

    def _write_result(self) -> None:
        (self.output_dir / "real_isaac_result.json").write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    return RefactorGuiE2E(Path(args.output), timeout_s=args.timeout_s).run()


if __name__ == "__main__":
    raise SystemExit(main())
