"""Adapter from accepted-step commands to Isaac Sim robot targets."""

from __future__ import annotations

import math
import shlex
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from command_model import (
    CommandMessage,
    HIP_LIMIT_DEG,
    JOINT_COMMAND_SIGN,
    KNEE_JOINT_NAMES,
    KNEE_LIMIT_DEG,
    SERVO_JOINT_NAMES,
    WHEEL_FORWARD_SIGN,
    WHEEL_JOINT_NAMES,
    clamp,
    clamp_servo_command,
    command_limits_for_servo,
    is_float_token,
    resolve_servo_name,
    resolve_servo_targets_for_command,
    resolve_servo_targets_for_group_part,
    resolve_wheel_name,
    split_semicolon_commands,
    validate_motion_command,
)
from playback import PlaybackPlan, plan_from_steps
from motion_speed import MotionReference, SpeedScale, load_motion_reference
from recording_baseline import WHEEL_PHYSICAL_STOP_NOISE_FLOOR_RAD_S
from robot_ground_diagnostics import (
    COLLIDER_RESOLUTION_FAILED,
    COLLISION_PENETRATION,
    GROUND_OK,
    RENDER_OR_FABRIC_DESYNC_SUSPECTED,
    VISUAL_ONLY_INTERSECTION,
    compute_bounded_ground_correction,
    default_robot_ground_diagnostics,
    inspect_robot_ground_contact,
)
from sequence_model import clone_command_state, empty_command_state


PHYSX_SAFE_LIMIT_MIN_RAD = -2.0 * math.pi
PHYSX_SAFE_LIMIT_MAX_RAD = 2.0 * math.pi
PHYSX_WRITE_LIMIT_MARGIN_RAD = 1.0e-6
PHYSX_TARGET_QUANTIZATION_MARGIN_RAD = 1.0e-5
REAL_MOTION_REFERENCE = load_motion_reference()


@dataclass
class SimRobotAdapterConfig:
    max_wheel_speed: float = REAL_MOTION_REFERENCE.wheel_velocity_limit_rad_s
    default_wheel_speed: float = REAL_MOTION_REFERENCE.wheel_reference_velocity_rad_s
    wheel_direction: float = 1.0
    apply_safe_servo_joint_limits: bool = True
    # Command-space limits are authoritative and must be installed on the live
    # articulation; otherwise the imported asset's stale limits win.
    apply_joint_limits_to_sim: bool = True
    ground_settle_s: float = 0.75
    ground_settle_max_steps: int = 180
    ground_stable_frames: int = 10
    ground_vertical_speed_threshold_m_s: float = 0.01
    ground_joint_speed_threshold_rad_s: float = 0.02
    ground_servo_speed_threshold_rad_s: float | None = None
    ground_wheel_speed_threshold_rad_s: float = 0.20
    ground_clearance_m: float = 0.002
    ground_penetration_tolerance_m: float = 0.003
    auto_ground_correction: bool = False
    max_ground_correction_m: float = 0.10
    # Fixed command-space servo reference profile shared by every caller.
    servo_base_velocity_deg_s: float = 150.0
    # Position drives can retain a small gravity/load deflection after the
    # nominal trajectory ends.  A bounded runtime tracking correction closes
    # that error without changing the logical/recorded target or drive gains.
    servo_tracking_compensation_gain: float = 8.0
    servo_tracking_compensation_max_deg: float = 10.0
    servo_tracking_feedback_interval_ticks: int = 4


def actual_target_deg_for_command(joint_name: str, command_angle_deg: float, standing_pose_deg: float) -> float:
    return float(standing_pose_deg) + JOINT_COMMAND_SIGN[joint_name] * float(command_angle_deg)


def desired_actual_limits_deg(joint_name: str, standing_pose_deg: float) -> tuple[float, float]:
    cmd_min, cmd_max = command_limits_for_servo(joint_name)
    target_a = actual_target_deg_for_command(joint_name, cmd_min, standing_pose_deg)
    target_b = actual_target_deg_for_command(joint_name, cmd_max, standing_pose_deg)
    return min(target_a, target_b), max(target_a, target_b)


def safe_rad_limits_for_actual_deg(min_deg: float, max_deg: float) -> dict[str, Any]:
    raw_min_rad = math.radians(min(float(min_deg), float(max_deg)))
    raw_max_rad = math.radians(max(float(min_deg), float(max_deg)))
    min_rad = max(raw_min_rad, PHYSX_SAFE_LIMIT_MIN_RAD)
    max_rad = min(raw_max_rad, PHYSX_SAFE_LIMIT_MAX_RAD)
    clamped_limits = min_rad != raw_min_rad or max_rad != raw_max_rad
    warnings: list[str] = []
    if clamped_limits:
        warnings.append(
            "desired rad limits were clamped to "
            f"[{PHYSX_SAFE_LIMIT_MIN_RAD:.6g}, {PHYSX_SAFE_LIMIT_MAX_RAD:.6g}]"
        )
    skip = min_rad >= max_rad
    skip_reason = "clamped min_rad >= max_rad" if skip else ""
    if skip:
        warnings.append(skip_reason)
    return {
        "raw_min_rad": raw_min_rad,
        "raw_max_rad": raw_max_rad,
        "raw_min_deg": math.degrees(raw_min_rad),
        "raw_max_deg": math.degrees(raw_max_rad),
        "min_rad": min_rad,
        "max_rad": max_rad,
        "min_deg": math.degrees(min_rad),
        "max_deg": math.degrees(max_rad),
        "clamped": clamped_limits,
        "skip": skip,
        "skip_reason": skip_reason,
        "warnings": warnings,
    }


def physx_write_rad_limits_from_record(record: dict[str, Any]) -> tuple[float, float, bool]:
    """Return strict PhysX write limits inside the documented [-2*pi, 2*pi] range."""

    # The live tensor is float32.  Expand desired limits by a tiny envelope so
    # an endpoint does not land a few ULPs outside its own rounded limit.
    min_rad = max(
        float(record["min_rad"]) - PHYSX_TARGET_QUANTIZATION_MARGIN_RAD,
        PHYSX_SAFE_LIMIT_MIN_RAD + PHYSX_WRITE_LIMIT_MARGIN_RAD,
    )
    max_rad = min(
        float(record["max_rad"]) + PHYSX_TARGET_QUANTIZATION_MARGIN_RAD,
        PHYSX_SAFE_LIMIT_MAX_RAD - PHYSX_WRITE_LIMIT_MARGIN_RAD,
    )
    adjusted = min_rad != float(record["min_rad"]) or max_rad != float(record["max_rad"])
    return min_rad, max_rad, adjusted


class SimRobotAdapter:
    def __init__(self, scene_handle: Any, config: SimRobotAdapterConfig | None = None):
        self.scene_handle = scene_handle
        self.robot = scene_handle.robot
        self.sim = scene_handle.sim
        self.device = self.robot.device
        self.config = config or SimRobotAdapterConfig(
            max_wheel_speed=float(scene_handle.config.max_wheel_speed),
            default_wheel_speed=float(scene_handle.config.default_wheel_speed),
            wheel_direction=float(scene_handle.config.wheel_direction),
        )
        self.max_wheel_speed = abs(float(self.config.max_wheel_speed))
        self.default_wheel_speed = self._clamp_wheel_speed(float(self.config.default_wheel_speed))
        self.wheel_direction = 1.0 if float(self.config.wheel_direction) >= 0.0 else -1.0
        self.sim_time = 0.0
        self.sim_steps = 0
        self.command_state = empty_command_state()
        self.telemetry_collector: Any | None = None
        self.motion_reference: MotionReference = load_motion_reference()
        self.speed_scale = SpeedScale(self.motion_reference)
        self.motion_batch_status: dict[str, Any] = {}

        self.servo_joint_ids = self._resolve_exact_joint_ids(SERVO_JOINT_NAMES)
        self.wheel_joint_ids = self._resolve_exact_joint_ids(WHEEL_JOINT_NAMES)
        self.servo_name_to_id = {name: idx for name, idx in zip(SERVO_JOINT_NAMES, self.servo_joint_ids)}
        self.wheel_name_to_id = {name: idx for name, idx in zip(WHEEL_JOINT_NAMES, self.wheel_joint_ids)}

        self.raw_spawn_root_pose = self.robot.data.root_pose_w.clone()
        self.command_zero_joint_pos = self.robot.data.joint_pos.clone()
        self.command_zero_joint_vel = self._torch().zeros_like(self.robot.data.joint_vel)
        self.stand_joint_pos = self.command_zero_joint_pos
        self.stand_joint_vel = self.command_zero_joint_vel
        self.standing_root_pose = self.raw_spawn_root_pose
        self.grounded_respawn_root_pose = self.raw_spawn_root_pose.clone()
        self.grounded_respawn_joint_pos = self.command_zero_joint_pos.clone()
        self.grounded_reference_valid = False
        self.grounded_reference_physics_valid = False
        self.grounded_reference_visual_valid = False
        self.grounded_reference_stable = False
        self.respawn_ready = False
        self.ground_reference_block_reason = ""
        self.grounded_reference_diagnostics: dict[str, Any] = default_robot_ground_diagnostics("grounded reference not initialized")
        self.robot_ground_diagnostics: dict[str, Any] = default_robot_ground_diagnostics("not checked")
        self.last_ground_correction_z_m = 0.0
        self.last_ground_settle_result: dict[str, Any] = {}
        self.zero_root_velocity = self._torch().zeros((self.robot.num_instances, 6), device=self.device)
        self.standing_pose_deg = {
            name: math.degrees(float(self.stand_joint_pos[0, joint_id].item()))
            for name, joint_id in zip(SERVO_JOINT_NAMES, self.servo_joint_ids)
        }
        self.joint_command_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.servo_applied_command_deg = dict(self.joint_command_deg)
        self.servo_nominal_target_reached = {name: True for name in SERVO_JOINT_NAMES}
        self.servo_tracking_compensation_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.servo_tracking_active = {name: False for name in SERVO_JOINT_NAMES}
        self.servo_tracking_stable_ticks = {name: 0 for name in SERVO_JOINT_NAMES}
        self.servo_tracking_feedback_tick = 0
        self.servo_motion_enabled = True
        self.wheel_speeds = {name: 0.0 for name in WHEEL_JOINT_NAMES}
        self.wheel_generation = 0
        self.wheel_stop_tolerance_rad_s = WHEEL_PHYSICAL_STOP_NOISE_FLOOR_RAD_S
        self.wheel_previous_command_rad_s = 0.0
        self.wheel_command_status: dict[str, Any] = {
            "generation": 0,
            "command_id": "",
            "state": "idle",
            "stop_command_received": False,
            "zero_target_applied": True,
            "physically_stopped": True,
            "stale_command_rejected": False,
            "requested_wall_time": 0.0,
            "enqueued_wall_time": 0.0,
            "received_wall_time": 0.0,
            "target_applied_wall_time": 0.0,
            "measured_stop_wall_time": 0.0,
            "target_applied_sim_time": 0.0,
            "measured_stop_sim_time": 0.0,
            "stop_tolerance_rad_s": self.wheel_stop_tolerance_rad_s,
            "applied_target_rad_s": dict(self.wheel_speeds),
            "measured_velocity_rad_s": {name: 0.0 for name in WHEEL_JOINT_NAMES},
        }
        self.servo_target_debug: dict[str, dict[str, float | str | bool | list[float]]] = {}
        self.safe_joint_limit_records: dict[str, dict[str, Any]] = {}
        self._target_warning_cache: set[str] = set()
        self._limit_unknown_warning_cache: set[str] = set()
        self.servo_cmd_targets = self._targets_from_command_angles()
        if self.config.apply_safe_servo_joint_limits or self.config.apply_joint_limits_to_sim:
            self.update_safe_command_space_joint_limit_records(write_to_sim=bool(self.config.apply_joint_limits_to_sim))
        print(
            "[INFO] Height replay SimRobotAdapter ready. "
            f"hip_limits={HIP_LIMIT_DEG}, knee_limits={KNEE_LIMIT_DEG}, "
            f"max_wheel_speed={self.max_wheel_speed:.3f} rad/s, "
            f"safe_servo_limits={self.config.apply_safe_servo_joint_limits}, "
            f"physx_limit_writes={self.config.apply_joint_limits_to_sim}"
        )

    def attach_telemetry(self, collector: Any | None) -> None:
        self.telemetry_collector = collector

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return {
            "servos": {name: round(float(self.joint_command_deg[name]), 6) for name in SERVO_JOINT_NAMES},
            "wheels": {name: round(float(self.wheel_speeds[name]), 6) for name in WHEEL_JOINT_NAMES},
        }

    def set_speed_percent(self, value: float) -> dict[str, Any]:
        """Atomically update the sole speed percentage used by this executor."""

        self.speed_scale.set_percent(value)
        status = self.speed_scale.status()
        status.update(updated_wall_time=time.time(), updated_sim_time=float(self.sim_time))
        return status

    def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Stage all logical targets and perform one articulation apply/write."""

        received_wall = time.time()
        batch_id = str(payload.get("batch_id", "") or uuid.uuid4().hex)
        servo_targets = dict(payload.get("servo_targets_deg", {}) or {})
        wheel_payload = payload.get("wheel_base_velocity_rad_s", {})
        if isinstance(wheel_payload, (int, float)):
            wheel_targets = {name: float(wheel_payload) for name in WHEEL_JOINT_NAMES}
        else:
            wheel_targets = {
                name: float(dict(wheel_payload or {}).get(name, self.wheel_speeds[name]))
                for name in WHEEL_JOINT_NAMES
            }
        for name, target in servo_targets.items():
            if name in self.joint_command_deg:
                self._set_joint_command_deg(name, float(target))
        wheel_applied = self.apply_wheel_velocity(
            wheel_targets,
            generation=payload.get("wheel_generation"),
            command_id=batch_id,
        )
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.command_state = self.capture_command_state()
        self.apply_commands_to_robot()
        self.robot.write_data_to_sim()
        dt = float(self.sim.get_physics_dt())
        servo_requested, servo_effective = self.speed_scale.servo_velocity()
        effective_wheels = {name: self._effective_wheel_speed(value)[1] for name, value in self.wheel_speeds.items()}
        ack = {
            "batch_id": batch_id,
            "source": str(payload.get("source", "ui") or "ui"),
            "batch_received_wall_time": received_wall,
            "batch_applied_wall_time": time.time(),
            "batch_applied_sim_time": float(self.sim_time),
            "servo_applied": bool(servo_targets),
            "wheel_applied": bool(wheel_applied),
            "first_physics_step": int(self.sim_steps + 1),
            "servo_motion_start_sim_time": float(self.sim_time + dt),
            "wheel_motion_start_sim_time": float(self.sim_time + dt),
            "motion_start_skew_s": 0.0,
            "physics_dt_s": dt,
            "speed_percent_snapshot": float(payload.get("speed_percent_snapshot", self.speed_scale.speed_percent)),
            "executor_speed_percent": self.speed_scale.speed_percent,
            "requested_servo_velocity_deg_s": servo_requested,
            "effective_servo_velocity_deg_s": servo_effective,
            "canonical_servo_targets_deg": {name: float(value) for name, value in servo_targets.items()},
            "canonical_wheel_velocity_rad_s": dict(self.wheel_speeds),
            "effective_wheel_velocity_rad_s": effective_wheels,
            "recording_metadata": dict(payload.get("recording_metadata", {}) or {}),
            "high_priority": False,
            "error": "",
        }
        self.motion_batch_status = ack
        return dict(ack)

    def capture_sim_state(self) -> dict[str, Any]:
        return {
            "command_state": self.capture_command_state(),
            "target_joint_state": self.get_target_joint_state(),
            "actual_joint_state": self.get_actual_joint_state(),
            "root_pose": self._tensor_to_nested_list(getattr(self.robot.data, "root_pose_w", None)),
            "root_velocity": self._tensor_to_nested_list(getattr(self.robot.data, "root_vel_w", None)),
            "joint_pos": self._tensor_to_nested_list(getattr(self.robot.data, "joint_pos", None)),
            "joint_vel": self._tensor_to_nested_list(getattr(self.robot.data, "joint_vel", None)),
            "joint_names": list(getattr(self.robot, "joint_names", []) or []),
            "sim_time": float(getattr(getattr(self, "sim", None), "current_time", 0.0) or 0.0),
            "adapter_sim_time": float(getattr(self, "sim_time", 0.0) or 0.0),
            "adapter_sim_steps": int(getattr(self, "sim_steps", 0) or 0),
            "raw_spawn_root_pose": self._tensor_to_nested_list(getattr(self, "raw_spawn_root_pose", None)),
            "grounded_respawn_root_pose": self._tensor_to_nested_list(getattr(self, "grounded_respawn_root_pose", None)),
            "grounded_respawn_joint_pos": self._tensor_to_nested_list(getattr(self, "grounded_respawn_joint_pos", None)),
            "grounded_reference_valid": bool(getattr(self, "grounded_reference_valid", False)),
            "grounded_reference_physics_valid": bool(getattr(self, "grounded_reference_physics_valid", False)),
            "grounded_reference_visual_valid": bool(getattr(self, "grounded_reference_visual_valid", False)),
            "grounded_reference_stable": bool(getattr(self, "grounded_reference_stable", False)),
            "grounded_reference_diagnostics": dict(getattr(self, "grounded_reference_diagnostics", {})),
            "robot_ground_diagnostics": dict(getattr(self, "robot_ground_diagnostics", {})),
        }

    def restore_sim_state(self, sim_state: dict[str, Any] | None) -> None:
        state = dict(sim_state or {})
        torch = self._torch()
        try:
            root_pose = state.get("root_pose")
            if root_pose is not None:
                self.robot.write_root_pose_to_sim(torch.tensor(root_pose, dtype=torch.float32, device=self.device))
            root_velocity = state.get("root_velocity")
            if root_velocity is not None:
                self.robot.write_root_velocity_to_sim(torch.tensor(root_velocity, dtype=torch.float32, device=self.device))
            joint_pos = state.get("joint_pos")
            joint_vel = state.get("joint_vel")
            if joint_pos is not None and joint_vel is not None:
                self.robot.write_joint_state_to_sim(
                    torch.tensor(joint_pos, dtype=torch.float32, device=self.device),
                    torch.tensor(joint_vel, dtype=torch.float32, device=self.device),
                )
        except Exception as exc:
            print(f"[WARN] Could not restore full sim pose, falling back to command_state: {exc}")
        command_state = state.get("command_state")
        if isinstance(command_state, dict):
            for name, value in command_state.get("servos", {}).items():
                if name in self.joint_command_deg:
                    self.joint_command_deg[name] = float(value)
                    self.servo_applied_command_deg[name] = float(value)
                    self.servo_nominal_target_reached[name] = True
                    self.servo_tracking_compensation_deg[name] = 0.0
                    self.servo_tracking_active[name] = False
                    self.servo_tracking_stable_ticks[name] = 0
            for name, value in command_state.get("wheels", {}).items():
                if name in self.wheel_speeds:
                    self.wheel_speeds[name] = float(value)
            self.servo_cmd_targets = self._targets_from_command_angles()
            self.command_state = self.capture_command_state()
        self.robot.reset()

    def handle_command(self, message: Any) -> None:
        text = getattr(message, "text", message)
        source = str(getattr(message, "source", "ui") or "ui")
        for command in split_semicolon_commands(str(text)):
            self._handle_one_command(command, message=message)
            collector = getattr(self, "telemetry_collector", None)
            if collector is not None:
                try:
                    collector.record_command(self, message, command)
                except Exception as exc:
                    if source != "telemetry":
                        print(f"[WARN] Telemetry command record failed: {exc}")

    def stop_wheels(
        self,
        *,
        generation: int | None = None,
        command_id: str = "",
        requested_wall_time: float = 0.0,
        enqueued_wall_time: float = 0.0,
    ) -> bool:
        next_generation = self.wheel_generation + 1 if generation is None else int(generation)
        return self.apply_wheel_velocity(
            {name: 0.0 for name in WHEEL_JOINT_NAMES},
            generation=next_generation,
            command_id=command_id,
            high_priority=True,
            requested_wall_time=requested_wall_time,
            enqueued_wall_time=enqueued_wall_time,
        )

    def home(self) -> None:
        for name in SERVO_JOINT_NAMES:
            self.joint_command_deg[name] = 0.0
            self.servo_applied_command_deg[name] = 0.0
            self.servo_nominal_target_reached[name] = True
            self.servo_tracking_compensation_deg[name] = 0.0
            self.servo_tracking_active[name] = False
            self.servo_tracking_stable_ticks[name] = 0
        self.stop_wheels()
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.command_state = self.capture_command_state()

    def reset_robot(self) -> None:
        self.robot.write_joint_state_to_sim(self.command_zero_joint_pos, self.command_zero_joint_vel)
        self.robot.reset()
        self.home()

    def initialize_grounded_respawn_reference(self) -> dict[str, Any]:
        """Settle once after scene reset, then save the verified respawn pose."""

        self.stop_wheels()
        self.joint_command_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.servo_applied_command_deg = dict(self.joint_command_deg)
        self.servo_nominal_target_reached = {name: True for name in SERVO_JOINT_NAMES}
        self.servo_tracking_compensation_deg = {name: 0.0 for name in SERVO_JOINT_NAMES}
        self.servo_tracking_active = {name: False for name in SERVO_JOINT_NAMES}
        self.servo_tracking_stable_ticks = {name: 0 for name in SERVO_JOINT_NAMES}
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.command_state = self.capture_command_state()
        self.apply_commands_to_robot()
        self.robot.write_data_to_sim()
        settle = self.settle_robot_on_ground(label="initial_grounded_reference")
        diagnostics = dict(settle.get("ground_diagnostics", default_robot_ground_diagnostics("settle did not run")))
        classification = str(diagnostics.get("classification", ""))
        if classification == COLLISION_PENETRATION:
            settle_before_safe_placement = dict(settle)
            allowed, dz, reason = compute_bounded_ground_correction(
                diagnostics,
                target_clearance_m=float(self.config.ground_clearance_m),
                max_correction_m=float(self.config.max_ground_correction_m),
            )
            if allowed and dz > 0.0:
                self._apply_root_z_correction(dz)
                self.last_ground_correction_z_m = float(dz)
                diagnostics = self._diagnostics_after_initial_safe_placement(dz, reason=reason)
                settle = dict(settle_before_safe_placement)
                settle["label"] = "initial_safe_placement_after_ground_correction"
                settle["ground_diagnostics"] = diagnostics
                if self._corrected_ground_diagnostics_safe(diagnostics):
                    settle["ground_contact_resolved"] = True
                    settle["physical_ground_safe"] = True
                    settle["stable"] = bool(
                        settle.get("final_window_stable", False)
                        and settle.get("kinematic_stable", False)
                        and settle.get("servo_final_window_stable", settle.get("servo_pose_stable", False))
                        and settle.get("wheel_state_stable", False)
                    )
                diagnostics["initial_safe_placement_applied"] = True
                diagnostics["initial_safe_placement_correction_z_m"] = float(dz)
            else:
                diagnostics.setdefault("warnings", []).append(str(reason or "initial safe placement correction was not allowed"))
        classification = str(diagnostics.get("classification", ""))
        ground_state = str(diagnostics.get("ground_state", ""))
        missing = list(diagnostics.get("missing_collision_wheels", []) or [])
        unresolved = list(diagnostics.get("unresolved_collision_wheels", []) or [])
        max_penetration = float(diagnostics.get("maximum_collision_penetration_m", 0.0) or 0.0)
        stable = bool(settle.get("stable", False))
        stable_required = int(settle.get("stable_frames_required", self.config.ground_stable_frames) or self.config.ground_stable_frames)
        stable_frames = int(settle.get("stable_frames", 0) or 0)
        vertical_ok = float(settle.get("final_window_max_vertical_speed_m_s", settle.get("max_abs_vertical_speed_m_s", 0.0)) or 0.0) <= float(self.config.ground_vertical_speed_threshold_m_s)
        servo_threshold = float(
            self.config.ground_servo_speed_threshold_rad_s
            if self.config.ground_servo_speed_threshold_rad_s is not None
            else self.config.ground_joint_speed_threshold_rad_s
        )
        wheel_threshold = float(self.config.ground_wheel_speed_threshold_rad_s)
        servo_ok = bool(settle.get("servo_final_window_stable", settle.get("servo_pose_stable", False))) or float(
            settle.get("final_window_max_servo_joint_velocity_rad_s", settle.get("max_servo_joint_velocity_rad_s", 0.0)) or 0.0
        ) <= servo_threshold
        wheel_targets = dict(settle.get("wheel_target_velocity_by_name", {}) or {})
        wheel_targets_zero = not any(abs(float(value)) > 1.0e-6 for value in wheel_targets.values())
        wheel_ok = bool(settle.get("wheel_state_stable", False)) or (
            wheel_targets_zero
            and float(settle.get("final_window_max_wheel_joint_velocity_rad_s", settle.get("max_wheel_joint_velocity_rad_s", 0.0)) or 0.0) <= wheel_threshold
        )
        joint_ok = bool(servo_ok and wheel_ok)
        root_z_delta_ok = float(settle.get("final_window_max_root_z_delta_m", 0.0) or 0.0) <= float(
            settle.get("final_window_root_z_delta_threshold_m", max(1.0e-5, self.config.ground_vertical_speed_threshold_m_s * self.sim.get_physics_dt()))
        )
        kinematic_stable = bool(settle.get("kinematic_stable", vertical_ok and root_z_delta_ok))
        physical_ground_safe = bool(diagnostics.get("physical_ground_safe", classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}))
        visual_ground_safe = bool(diagnostics.get("visual_ground_safe", classification == GROUND_OK))
        inferred_ground_contact_resolved = bool(
            diagnostics.get("checked", False)
            and not missing
            and not unresolved
            and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION, RENDER_OR_FABRIC_DESYNC_SUSPECTED}
        )
        ground_contact_resolved = bool(settle.get("ground_contact_resolved", inferred_ground_contact_resolved))
        base_physics_valid = (
            stable
            and stable_frames >= stable_required
            and kinematic_stable
            and servo_ok
            and wheel_ok
            and ground_contact_resolved
            and bool(diagnostics.get("checked", False))
            and not missing
            and not unresolved
            and max_penetration <= float(self.config.ground_penetration_tolerance_m)
            and vertical_ok
            and joint_ok
            and root_z_delta_ok
            and physical_ground_safe
            and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}
        )
        visual_valid = bool(visual_ground_safe)
        valid = bool(base_physics_valid)
        self.grounded_reference_physics_valid = bool(base_physics_valid)
        self.grounded_reference_visual_valid = bool(visual_valid)
        self.grounded_reference_stable = bool(stable)
        self.grounded_reference_valid = bool(valid)
        self.respawn_ready = bool(valid and physical_ground_safe)
        diagnostics["grounded_reference_physics_valid"] = bool(self.grounded_reference_physics_valid)
        diagnostics["grounded_reference_visual_valid"] = bool(self.grounded_reference_visual_valid)
        diagnostics["grounded_reference_stable"] = bool(self.grounded_reference_stable)
        diagnostics["grounded_reference_valid"] = bool(self.grounded_reference_valid)
        diagnostics["grounded_respawn_reference_valid"] = bool(self.grounded_reference_valid)
        diagnostics["respawn_ready"] = bool(self.respawn_ready)
        if not valid:
            reasons = diagnostics.setdefault("reasons", [])
            if not stable:
                reasons.append("grounded reference rejected: settle did not reach stable frame threshold")
            if stable_frames < stable_required:
                reasons.append(f"grounded reference rejected: stable frames {stable_frames}/{stable_required}")
            if not bool(diagnostics.get("checked", False)):
                reasons.append("grounded reference rejected: ground diagnostics were not checked")
            if missing:
                reasons.append("grounded reference rejected: missing wheel collisions")
            if unresolved or classification == COLLIDER_RESOLUTION_FAILED:
                reasons.append("grounded reference rejected: wheel collision resolution is unverified")
            if not physical_ground_safe:
                reasons.append("grounded reference rejected: physical ground contact is not safe")
            if max_penetration > float(self.config.ground_penetration_tolerance_m):
                reasons.append("grounded reference rejected: collision penetration exceeds tolerance")
            if classification != GROUND_OK and classification != VISUAL_ONLY_INTERSECTION:
                reasons.append(f"grounded reference rejected: collision classification is {classification}, expected {GROUND_OK} or {VISUAL_ONLY_INTERSECTION}")
            if ground_state == "UNVERIFIED":
                reasons.append("grounded reference rejected: ground state is UNVERIFIED")
            if not vertical_ok:
                reasons.append("grounded reference rejected: final root vertical speed exceeds threshold")
            if not joint_ok:
                offenders = list(settle.get("offending_joint_names", []) or [])
                suffix = ": " + ", ".join(offenders) if offenders else ""
                reasons.append("grounded reference rejected: final joint velocity exceeds threshold" + suffix)
            if not root_z_delta_ok:
                reasons.append("grounded reference rejected: final root Z is still moving")
            if not kinematic_stable:
                reasons.append("grounded reference rejected: chassis kinematics are not stable")
            if not servo_ok:
                reasons.append("grounded reference rejected: servo joint velocity exceeds threshold")
            if not wheel_ok:
                reasons.append("grounded reference rejected: wheel state is not stable")
            if not ground_contact_resolved:
                reasons.append("grounded reference rejected: wheel ground contact is not resolved")
        else:
            self.grounded_respawn_root_pose = self.robot.data.root_pose_w.clone()
            self.grounded_respawn_joint_pos = self.robot.data.joint_pos.clone()
        self.ground_reference_block_reason = "; ".join(dict.fromkeys(diagnostics.get("reasons", []) or [])) if not valid else ""
        diagnostics["ground_reference_block_reason"] = self.ground_reference_block_reason
        self.grounded_reference_diagnostics = diagnostics
        self.robot_ground_diagnostics = diagnostics
        if valid:
            self.standing_root_pose = self.grounded_respawn_root_pose
        print(
            "[INFO] Grounded respawn reference "
            f"{'VALID' if self.grounded_reference_valid else 'INVALID'}: "
            f"classification={diagnostics.get('classification', 'UNKNOWN')} "
            f"stable={self.grounded_reference_stable} "
            f"physics_valid={self.grounded_reference_physics_valid} "
            f"visual_valid={self.grounded_reference_visual_valid} "
            f"root_z={diagnostics.get('root_z_m')}"
        )
        return {
            **settle,
            "grounded_reference_valid": bool(self.grounded_reference_valid),
            "grounded_reference_physics_valid": bool(self.grounded_reference_physics_valid),
            "grounded_reference_visual_valid": bool(self.grounded_reference_visual_valid),
            "grounded_reference_stable": bool(self.grounded_reference_stable),
            "grounded_reference_diagnostics": diagnostics,
        }

    def calibrate_grounded_reference(self) -> dict[str, Any]:
        return self.initialize_grounded_respawn_reference()

    def settle_robot_on_ground(
        self,
        *,
        label: str = "settle",
        max_steps: int | None = None,
        duration_s: float | None = None,
        validate_ground: bool = True,
    ) -> dict[str, Any]:
        dt = float(self.sim.get_physics_dt())
        _elapsed_per_render, substeps_per_render = self._render_step_timing()
        configured_steps = int(max_steps if max_steps is not None else self.config.ground_settle_max_steps)
        if duration_s is not None:
            configured_steps = min(configured_steps, max(1, int(math.ceil(float(duration_s) / max(dt, 1.0e-9)))))
        else:
            configured_steps = min(configured_steps, max(1, int(math.ceil(float(self.config.ground_settle_s) / max(dt, 1.0e-9)))))
        stable_required = max(1, int(self.config.ground_stable_frames))
        vertical_threshold = float(self.config.ground_vertical_speed_threshold_m_s)
        servo_threshold = float(
            self.config.ground_servo_speed_threshold_rad_s
            if self.config.ground_servo_speed_threshold_rad_s is not None
            else self.config.ground_joint_speed_threshold_rad_s
        )
        wheel_threshold = float(self.config.ground_wheel_speed_threshold_rad_s)
        previous_root_z = self._current_root_z()
        stable_frames = 0
        steps_run = 0
        max_abs_joint_vel = 0.0
        max_servo_joint_vel = 0.0
        max_wheel_joint_vel = 0.0
        max_abs_vertical_speed = 0.0
        final_window: deque[dict[str, Any]] = deque(maxlen=max(stable_required, 60))
        for step_index in range(max(1, configured_steps)):
            self.apply_commands_to_robot()
            self.robot.write_data_to_sim()
            if substeps_per_render > 1:
                self.sim.step(render=False)
            else:
                self.sim.step()
            self.sim_time += dt
            self.sim_steps += 1
            self.robot.update(dt)
            steps_run += 1
            if substeps_per_render > 1 and ((step_index + 1) % substeps_per_render == 0 or step_index + 1 == configured_steps):
                self.sim.render()
            root_z = self._current_root_z()
            root_velocity = self._current_root_velocity()
            vertical_speed = abs(root_velocity[2]) if len(root_velocity) >= 3 else 0.0
            joint_velocity_by_name = self._joint_velocity_by_name()
            servo_speed = self._max_abs_named_joint_velocity(SERVO_JOINT_NAMES, joint_velocity_by_name)
            wheel_speed = self._max_abs_named_joint_velocity(WHEEL_JOINT_NAMES, joint_velocity_by_name)
            joint_speed = max(servo_speed, wheel_speed)
            wheel_targets = self._wheel_velocity_target_by_name()
            max_abs_vertical_speed = max(max_abs_vertical_speed, vertical_speed)
            max_abs_joint_vel = max(max_abs_joint_vel, joint_speed)
            max_servo_joint_vel = max(max_servo_joint_vel, servo_speed)
            max_wheel_joint_vel = max(max_wheel_joint_vel, wheel_speed)
            root_z_delta = abs(root_z - previous_root_z) if previous_root_z is not None and root_z is not None else 0.0
            root_z_delta_threshold = max(1.0e-5, vertical_threshold * dt)
            wheel_targets_zero = not any(abs(float(value)) > 1.0e-6 for value in wheel_targets.values())
            kinematic_stable = vertical_speed <= vertical_threshold and root_z_delta <= root_z_delta_threshold
            servo_pose_stable = servo_speed <= servo_threshold
            wheel_state_stable = wheel_targets_zero and wheel_speed <= wheel_threshold
            offending = self._offending_joint_names(
                joint_velocity_by_name,
                servo_threshold=servo_threshold,
                wheel_threshold=wheel_threshold,
                include_wheels=not wheel_state_stable,
            )
            final_window.append(
                {
                    "root_z": root_z,
                    "root_velocity": list(root_velocity),
                    "vertical_speed": float(vertical_speed),
                    "joint_speed": float(joint_speed),
                    "servo_joint_speed": float(servo_speed),
                    "wheel_joint_speed": float(wheel_speed),
                    "root_z_delta": float(root_z_delta),
                    "joint_velocity": self._joint_velocity_vector(),
                    "joint_velocity_by_name": joint_velocity_by_name,
                    "wheel_target_velocity_by_name": wheel_targets,
                    "kinematic_stable": bool(kinematic_stable),
                    "servo_pose_stable": bool(servo_pose_stable),
                    "wheel_state_stable": bool(wheel_state_stable),
                    "offending_joint_names": offending,
                }
            )
            if kinematic_stable and servo_pose_stable and wheel_state_stable:
                stable_frames += 1
            else:
                stable_frames = 0
            previous_root_z = root_z
            if stable_frames >= stable_required:
                break
        if validate_ground:
            diagnostics = self.validate_robot_ground_contact(apply_correction=False)
        else:
            diagnostics = dict(self.grounded_reference_diagnostics or self.robot_ground_diagnostics)
            diagnostics.update(
                checked=True,
                root_z_m=self._current_root_z(),
                cached_ground_reference_used=True,
                live_mesh_validation_ran=False,
            )
        final_frames = list(final_window)
        final_vertical_speed = max([float(frame.get("vertical_speed", 0.0) or 0.0) for frame in final_frames] + [0.0])
        final_joint_speed = max([float(frame.get("joint_speed", 0.0) or 0.0) for frame in final_frames] + [0.0])
        final_servo_speed = max([float(frame.get("servo_joint_speed", 0.0) or 0.0) for frame in final_frames] + [0.0])
        final_wheel_speed = max([float(frame.get("wheel_joint_speed", 0.0) or 0.0) for frame in final_frames] + [0.0])
        final_root_z_delta = max([float(frame.get("root_z_delta", 0.0) or 0.0) for frame in final_frames] + [0.0])
        final_window_servo_velocity_by_name = _max_abs_by_name(final_frames, SERVO_JOINT_NAMES)
        final_window_wheel_velocity_by_name = _max_abs_by_name(final_frames, WHEEL_JOINT_NAMES)
        servo_velocity_values = [
            abs(float(frame.get("joint_velocity_by_name", {}).get(name, 0.0) or 0.0))
            for frame in final_frames
            for name in SERVO_JOINT_NAMES
        ]
        servo_velocity_rms = _rms(servo_velocity_values)
        servo_velocity_p95 = _percentile(servo_velocity_values, 95.0)
        servo_velocity_median = _percentile(servo_velocity_values, 50.0)
        root_z_values = [float(frame.get("root_z", 0.0) or 0.0) for frame in final_frames if frame.get("root_z") is not None]
        root_z_range = (max(root_z_values) - min(root_z_values)) if root_z_values else 0.0
        final_frame = final_frames[-1] if final_frames else {}
        final_root_velocity = list(final_frame.get("root_velocity", []) or [])
        final_joint_velocity = list(final_frame.get("joint_velocity", []) or [])
        final_joint_velocity_by_name = dict(final_frame.get("joint_velocity_by_name", {}) or {})
        wheel_targets = dict(final_frame.get("wheel_target_velocity_by_name", self._wheel_velocity_target_by_name()) or {})
        root_z_delta_threshold = max(1.0e-5, vertical_threshold * dt)
        kinematic_stable = bool(final_vertical_speed <= vertical_threshold and final_root_z_delta <= root_z_delta_threshold)
        servo_pose_stable = bool(final_servo_speed <= servo_threshold)
        wheel_state_stable = bool(
            final_wheel_speed <= wheel_threshold
            and not any(abs(float(value)) > 1.0e-6 for value in wheel_targets.values())
        )
        ground_contact_resolved = bool(
            diagnostics.get("checked", False)
            and not diagnostics.get("missing_collision_wheels")
            and not diagnostics.get("unresolved_collision_wheels")
            and str(diagnostics.get("classification", "")) in {
                GROUND_OK,
                VISUAL_ONLY_INTERSECTION,
                RENDER_OR_FABRIC_DESYNC_SUSPECTED,
            }
        )
        stable = stable_frames >= stable_required and str(diagnostics.get("classification", "")) in {
            GROUND_OK,
            VISUAL_ONLY_INTERSECTION,
            RENDER_OR_FABRIC_DESYNC_SUSPECTED,
        } and kinematic_stable and servo_pose_stable and wheel_state_stable
        offending_joint_names = self._offending_joint_names(
            final_joint_velocity_by_name,
            servo_threshold=servo_threshold,
            wheel_threshold=wheel_threshold,
            include_wheels=not wheel_state_stable,
        )
        servo_position_error_by_name = self._servo_position_error_by_name()
        max_servo_position_error_rad = max([abs(float(value)) for value in servo_position_error_by_name.values()] + [0.0])
        servo_position_tolerance_rad = 0.02
        servo_micro_jitter_threshold = max(float(servo_threshold) * 4.0, 0.08)
        servo_position_stable = bool(max_servo_position_error_rad <= servo_position_tolerance_rad)
        servo_final_window_stable = bool(
            servo_pose_stable
            or (
                servo_position_stable
                and float(servo_velocity_p95) <= servo_micro_jitter_threshold
                and float(root_z_range) <= max(root_z_delta_threshold * float(max(1, len(final_frames))), 0.001)
            )
        )
        final_window_stable = bool(kinematic_stable and servo_final_window_stable and wheel_state_stable)
        effective_stable_frames = int(stable_frames)
        if final_window_stable and effective_stable_frames < stable_required:
            effective_stable_frames = stable_required
        stable = bool(
            (stable_frames >= stable_required or final_window_stable)
            and str(diagnostics.get("classification", "")) in {
                GROUND_OK,
                VISUAL_ONLY_INTERSECTION,
                RENDER_OR_FABRIC_DESYNC_SUSPECTED,
            }
            and kinematic_stable
            and servo_final_window_stable
            and wheel_state_stable
        )
        result = {
            "label": str(label),
            "steps_run": int(steps_run),
            "stable": bool(stable),
            "final_window_stable": bool(final_window_stable),
            "kinematic_stable": bool(kinematic_stable),
            "chassis_stable": bool(kinematic_stable),
            "root_vertical_stable": bool(final_vertical_speed <= vertical_threshold),
            "root_height_stable": bool(final_root_z_delta <= root_z_delta_threshold),
            "servo_pose_stable": bool(servo_pose_stable),
            "servo_position_stable": bool(servo_position_stable),
            "servo_velocity_stable": bool(servo_pose_stable),
            "servo_final_window_stable": bool(servo_final_window_stable),
            "servo_micro_jitter_threshold_rad_s": float(servo_micro_jitter_threshold),
            "servo_position_tolerance_rad": float(servo_position_tolerance_rad),
            "wheel_state_stable": bool(wheel_state_stable),
            "wheel_command_zero": bool(not any(abs(float(value)) > 1.0e-6 for value in wheel_targets.values())),
            "wheel_motion_stable": bool(final_wheel_speed <= wheel_threshold),
            "ground_contact_resolved": bool(ground_contact_resolved),
            "physical_ground_safe": bool(diagnostics.get("physical_ground_safe", False)),
            "stable_frames": int(effective_stable_frames),
            "strict_stable_frames": int(stable_frames),
            "stable_frames_required": int(stable_required),
            "max_abs_vertical_speed_m_s": float(max_abs_vertical_speed),
            "max_abs_joint_velocity_rad_s": float(max_abs_joint_vel),
            "max_servo_joint_velocity_rad_s": float(max_servo_joint_vel),
            "max_wheel_joint_velocity_rad_s": float(max_wheel_joint_vel),
            "peak_abs_vertical_speed_m_s": float(max_abs_vertical_speed),
            "peak_abs_joint_velocity_rad_s": float(max_abs_joint_vel),
            "peak_servo_joint_velocity_rad_s": float(max_servo_joint_vel),
            "peak_wheel_joint_velocity_rad_s": float(max_wheel_joint_vel),
            "final_window_max_vertical_speed_m_s": float(final_vertical_speed),
            "final_window_max_joint_velocity_rad_s": float(final_joint_speed),
            "final_window_max_servo_joint_velocity_rad_s": float(final_servo_speed),
            "final_window_max_wheel_joint_velocity_rad_s": float(final_wheel_speed),
            "final_window_max_root_z_delta_m": float(final_root_z_delta),
            "final_window_root_z_delta_threshold_m": float(root_z_delta_threshold),
            "servo_velocity_rms": float(servo_velocity_rms),
            "servo_velocity_p95": float(servo_velocity_p95),
            "servo_velocity_median": float(servo_velocity_median),
            "root_z_range": float(root_z_range),
            "servo_speed_threshold_rad_s": float(servo_threshold),
            "wheel_speed_threshold_rad_s": float(wheel_threshold),
            "final_window_size": int(len(final_frames)),
            "final_window_frames": final_frames,
            "final_root_z_m": final_frame.get("root_z"),
            "final_root_velocity": final_root_velocity,
            "final_joint_velocity": final_joint_velocity,
            "joint_velocity_by_name": final_joint_velocity_by_name,
            "final_window_joint_velocity_by_name": [dict(frame.get("joint_velocity_by_name", {}) or {}) for frame in final_frames],
            "final_window_servo_velocity_by_name": final_window_servo_velocity_by_name,
            "final_window_wheel_velocity_by_name": final_window_wheel_velocity_by_name,
            "final_window_servo_position_error_by_name": servo_position_error_by_name,
            "max_servo_position_error_rad": float(max_servo_position_error_rad),
            "max_servo_speed": float(final_servo_speed),
            "max_wheel_speed": float(final_wheel_speed),
            "final_servo_speed": float(final_servo_speed),
            "final_wheel_speed": float(final_wheel_speed),
            "offending_joint_names": offending_joint_names,
            "offending_servo_names": [name for name in offending_joint_names if name in set(SERVO_JOINT_NAMES)],
            "offending_joints": offending_joint_names,
            "wheel_target_velocity_by_name": wheel_targets,
            "root_vertical_speed": float(final_vertical_speed),
            "root_z_delta": float(final_root_z_delta),
            "ground_diagnostics": diagnostics,
        }
        self.last_ground_settle_result = result
        return result

    def validate_robot_ground_contact(self, *, apply_correction: bool | None = None) -> dict[str, Any]:
        diagnostics = inspect_robot_ground_contact(
            self.scene_handle,
            self,
            sim_time=float(self.sim_time),
            sim_steps=int(getattr(self, "sim_steps", 0) or 0),
            penetration_tolerance_m=float(self.config.ground_penetration_tolerance_m),
        ).to_dict()
        diagnostics["grounded_respawn_reference_valid"] = bool(self.grounded_reference_valid)
        diagnostics["grounded_reference_physics_valid"] = bool(getattr(self, "grounded_reference_physics_valid", False))
        diagnostics["grounded_reference_visual_valid"] = bool(getattr(self, "grounded_reference_visual_valid", False))
        diagnostics["grounded_reference_stable"] = bool(getattr(self, "grounded_reference_stable", False))
        diagnostics["respawn_ready"] = bool(getattr(self, "respawn_ready", False))
        diagnostics["ground_reference_block_reason"] = str(getattr(self, "ground_reference_block_reason", "") or "")
        if apply_correction is None:
            apply_correction = bool(self.config.auto_ground_correction)
        if bool(apply_correction) and str(diagnostics.get("classification", "")) == COLLISION_PENETRATION:
            allowed, dz, reason = compute_bounded_ground_correction(
                diagnostics,
                target_clearance_m=float(self.config.ground_clearance_m),
                max_correction_m=float(self.config.max_ground_correction_m),
            )
            if allowed and dz > 0.0:
                self._apply_root_z_correction(dz)
                self.last_ground_correction_z_m = float(dz)
                settled = self.settle_robot_on_ground(label="post_ground_correction", duration_s=min(0.25, float(self.config.ground_settle_s)))
                diagnostics = dict(settled.get("ground_diagnostics", diagnostics))
                diagnostics["correction_applied"] = True
                diagnostics["correction_z_m"] = float(dz)
            else:
                diagnostics.setdefault("warnings", []).append(reason)
        diagnostics["grounded_respawn_reference_valid"] = bool(self.grounded_reference_valid)
        diagnostics["grounded_reference_physics_valid"] = bool(getattr(self, "grounded_reference_physics_valid", False))
        diagnostics["grounded_reference_visual_valid"] = bool(getattr(self, "grounded_reference_visual_valid", False))
        diagnostics["grounded_reference_stable"] = bool(getattr(self, "grounded_reference_stable", False))
        diagnostics["respawn_ready"] = bool(getattr(self, "respawn_ready", False))
        diagnostics["ground_reference_block_reason"] = str(getattr(self, "ground_reference_block_reason", "") or "")
        diagnostics["correction_z_m"] = float(diagnostics.get("correction_z_m", self.last_ground_correction_z_m) or 0.0)
        self.robot_ground_diagnostics = diagnostics
        return diagnostics

    def respawn_robot(self, *, settle: bool = True) -> dict[str, Any]:
        self.stop_wheels()
        if not bool(getattr(self, "respawn_ready", False)):
            ground_init = self.initialize_grounded_respawn_reference()
            if not bool(getattr(self, "respawn_ready", False)):
                self.apply_commands_to_robot()
                self.robot.write_data_to_sim()
                diagnostics = dict(ground_init.get("grounded_reference_diagnostics", self.grounded_reference_diagnostics))
                print("[WARN] Respawn blocked because grounded reference is not valid; raw spawn fallback was not used.")
                return {
                    "ok": False,
                    "respawned": False,
                    "error": str(getattr(self, "ground_reference_block_reason", "") or "respawn reference is not ready; raw spawn fallback was not used"),
                    "ground_diagnostics": diagnostics,
                    "settle": dict(self.last_ground_settle_result),
                }
        root_pose = self.grounded_respawn_root_pose
        joint_pos = self.grounded_respawn_joint_pos
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.write_joint_state_to_sim(joint_pos, self.command_zero_joint_vel)
        self.robot.reset()
        self.home()
        self.apply_commands_to_robot()
        self.robot.write_data_to_sim()
        settle_result = {}
        if bool(settle):
            settle_result = self.settle_robot_on_ground(
                label="respawn",
                duration_s=min(0.30, float(self.config.ground_settle_s)),
                validate_ground=False,
            )
        diagnostics = dict(
            settle_result.get("ground_diagnostics", self.grounded_reference_diagnostics)
            if settle_result
            else self.grounded_reference_diagnostics
        )
        diagnostics.update(
            checked=True,
            cached_ground_reference_used=True,
            live_mesh_validation_ran=False,
            root_z_m=self._current_root_z(),
        )
        self.robot_ground_diagnostics = diagnostics
        print(
            "[INFO] Respawned robot for height replay. "
            f"ground={diagnostics.get('classification', 'UNKNOWN')} "
            f"root_z={diagnostics.get('root_z_m')}"
        )
        return {
            "ground_diagnostics": diagnostics,
            "settle": settle_result,
            "cached_ground_reference_used": True,
            "ground_validation_ran": False,
        }

    def set_servo(self, name: str, angle_deg: float) -> None:
        for joint_name in resolve_servo_targets_for_command(name):
            self._set_joint_command_deg(joint_name, angle_deg)
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.command_state = self.capture_command_state()

    def set_servo_targets(self, targets: list[str], angle_deg: float) -> None:
        for joint_name in targets:
            self._set_joint_command_deg(joint_name, angle_deg)
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.command_state = self.capture_command_state()

    def apply_wheel_velocity(
        self,
        targets_rad_s: dict[str, float],
        *,
        generation: int | None = None,
        command_id: str = "",
        high_priority: bool = False,
        requested_wall_time: float = 0.0,
        enqueued_wall_time: float = 0.0,
    ) -> bool:
        """Authoritative atomic wheel target API used by every command source."""

        candidate = {name: self._clamp_wheel_speed(targets_rad_s.get(name, self.wheel_speeds[name])) for name in WHEEL_JOINT_NAMES}
        nonzero = any(abs(value) > 1.0e-12 for value in candidate.values())
        incoming_generation = self.wheel_generation if generation is None else int(generation)
        if nonzero and incoming_generation < self.wheel_generation:
            self.wheel_command_status.update(
                state="stale_nonzero_rejected",
                stale_command_rejected=True,
                rejected_generation=incoming_generation,
                generation=self.wheel_generation,
            )
            return False
        if high_priority and nonzero:
            raise ValueError("A high-priority wheel safety command must be an all-wheel zero target.")
        self.wheel_generation = max(self.wheel_generation, incoming_generation)
        previous_max = max(abs(value) for value in self.wheel_speeds.values())
        if not nonzero:
            self.wheel_previous_command_rad_s = max(self.wheel_previous_command_rad_s, previous_max)
            self.wheel_stop_tolerance_rad_s = max(
                0.02,
                0.05 * self.wheel_previous_command_rad_s,
                WHEEL_PHYSICAL_STOP_NOISE_FLOOR_RAD_S,
            )
        self.wheel_speeds.update(candidate)
        now = time.time()
        self.wheel_command_status = {
            **self.wheel_command_status,
            "generation": self.wheel_generation,
            "command_id": str(command_id or ""),
            "state": "stop_received" if not nonzero else "target_received",
            "stop_command_received": not nonzero,
            "zero_target_applied": False if not nonzero else None,
            "physically_stopped": False if not nonzero else None,
            "stale_command_rejected": False,
            "requested_wall_time": float(requested_wall_time or now),
            "enqueued_wall_time": float(enqueued_wall_time or now),
            "received_wall_time": now,
            "target_applied_wall_time": 0.0,
            "measured_stop_wall_time": 0.0,
            "target_applied_sim_time": 0.0,
            "measured_stop_sim_time": 0.0,
            "stop_tolerance_rad_s": self.wheel_stop_tolerance_rad_s,
            "applied_target_rad_s": dict(self.wheel_speeds),
            "canonical_target_rad_s": dict(self.wheel_speeds),
            "effective_target_rad_s": {name: self._effective_wheel_speed(value)[1] for name, value in self.wheel_speeds.items()},
            "speed_percent": self.speed_scale.speed_percent,
        }
        self.command_state = self.capture_command_state()
        return True

    def set_wheels(
        self,
        left_speed: float,
        right_speed: float | None = None,
        **metadata: Any,
    ) -> bool:
        if right_speed is None:
            right_speed = left_speed
        return self.apply_wheel_velocity(
            {
                "front_left_ankle": left_speed,
                "rear_left_ankle": left_speed,
                "front_right_ankle": right_speed,
                "rear_right_ankle": right_speed,
            },
            **metadata,
        )

    def set_all_wheels(self, speed: float, **metadata: Any) -> bool:
        return self.set_wheels(speed, speed, **metadata)

    def set_one_wheel(self, wheel_name: str, speed: float, **metadata: Any) -> bool:
        name = resolve_wheel_name(wheel_name)
        targets = dict(self.wheel_speeds)
        targets[name] = speed
        return self.apply_wheel_velocity(targets, **metadata)

    def set_turn(self, direction: str, **metadata: Any) -> None:
        speed = self.default_wheel_speed
        if direction == "left":
            self.set_wheels(-speed, speed, **metadata)
        else:
            self.set_wheels(speed, -speed, **metadata)

    def step(self, dt: float | None = None) -> None:
        physics_dt = float(self.sim.get_physics_dt())
        _elapsed, substeps = self._render_step_timing()
        collector = getattr(self, "telemetry_collector", None)
        for _substep in range(substeps):
            self._advance_servo_targets(physics_dt)
            self.apply_commands_to_robot()
            self.robot.write_data_to_sim()
            if substeps > 1:
                self.sim.step(render=False)
            else:
                self.sim.step()
            self.sim_time += physics_dt
            self.sim_steps += 1
            self.robot.update(physics_dt)
            self._update_wheel_stop_measurement()
            if collector is not None:
                try:
                    collector.on_step(self, physics_dt)
                except Exception as exc:
                    print(f"[WARN] Telemetry step failed: {exc}")
        if substeps > 1:
            self.sim.render()

    def _render_step_timing(self) -> tuple[float, int]:
        """Return actual simulation elapsed by one ``sim.step()`` call."""

        physics_dt = float(self.sim.get_physics_dt())
        get_rendering_dt = getattr(self.sim, "get_rendering_dt", None)
        try:
            rendering_dt = float(get_rendering_dt()) if callable(get_rendering_dt) else physics_dt
        except (TypeError, ValueError):
            rendering_dt = physics_dt
        if not math.isfinite(rendering_dt) or rendering_dt <= 0.0:
            rendering_dt = physics_dt
        substeps = max(1, int(round(rendering_dt / max(physics_dt, 1.0e-9))))
        return physics_dt * substeps, substeps

    def apply_commands_to_robot(self) -> None:
        self.servo_cmd_targets = self._targets_from_command_angles()
        self.robot.set_joint_position_target(self.servo_cmd_targets, joint_ids=self.servo_joint_ids)
        self.robot.set_joint_velocity_target(self._wheel_velocity_targets(), joint_ids=self.wheel_joint_ids)
        status = self.wheel_command_status
        if status.get("stop_command_received") and not status.get("zero_target_applied"):
            now = time.time()
            status.update(
                state="zero_target_applied",
                zero_target_applied=True,
                target_applied_wall_time=now,
                target_applied_sim_time=float(self.sim_time),
                applied_target_rad_s={name: self._effective_wheel_speed(value)[1] for name, value in self.wheel_speeds.items()},
                canonical_target_rad_s=dict(self.wheel_speeds),
                speed_percent=self.speed_scale.speed_percent,
            )

    def _advance_servo_targets(self, dt: float) -> None:
        """Rate-limit the nominal path, then close bounded load deflection."""

        if not self.servo_motion_enabled:
            return
        _requested_rate, rate = self.speed_scale.servo_velocity()
        maximum_delta = rate * max(0.0, float(dt))
        if maximum_delta <= 0.0:
            return
        feedback_names = {
            name
            for name in SERVO_JOINT_NAMES
            if bool(self.servo_nominal_target_reached.get(name, False)) and bool(self.servo_tracking_active.get(name, False))
        }
        feedback_interval = max(1, int(self.config.servo_tracking_feedback_interval_ticks))
        sample_feedback = self.servo_tracking_feedback_tick % feedback_interval == 0
        self.servo_tracking_feedback_tick += 1
        measured_by_name: dict[str, float | None] = {}
        if feedback_names and sample_feedback:
            try:
                # Synchronize only while a just-finished nominal target still
                # needs load-error closure; settled joints keep their bias.
                values = self.robot.data.joint_pos[0, self.servo_joint_ids].detach().cpu().tolist()
                measured_by_name = {name: math.degrees(float(values[index])) for index, name in enumerate(SERVO_JOINT_NAMES)}
            except Exception:
                measured_by_name = {
                    name: (None if (rad := self._measured_joint_rad(name)) is None else math.degrees(rad))
                    for name in feedback_names
                }
        for name in SERVO_JOINT_NAMES:
            requested = self._clamp_command_angle_deg(name, float(self.joint_command_deg[name]))
            self.joint_command_deg[name] = requested
            applied = float(self.servo_applied_command_deg.get(name, requested))
            if not bool(self.servo_nominal_target_reached.get(name, False)):
                delta = requested - applied
                if abs(delta) <= maximum_delta:
                    applied = requested
                    self.servo_nominal_target_reached[name] = True
                else:
                    applied += math.copysign(maximum_delta, delta)
                self.servo_tracking_compensation_deg[name] = 0.0
            elif bool(self.servo_tracking_active.get(name, False)):
                measured_deg = measured_by_name.get(name)
                if measured_deg is None:
                    self.servo_applied_command_deg[name] = self._clamp_command_angle_deg(name, applied)
                    continue
                logical_actual_deg = self.command_to_actual_target_deg(name, requested)
                actual_error_deg = logical_actual_deg - measured_deg
                sign = float(JOINT_COMMAND_SIGN[name])
                gain = max(0.0, float(self.config.servo_tracking_compensation_gain))
                limit = max(0.0, float(self.config.servo_tracking_compensation_max_deg))
                previous_correction = float(self.servo_tracking_compensation_deg.get(name, 0.0))
                desired_correction = clamp((actual_error_deg / sign) * gain, -limit, limit)
                correction_delta = clamp(desired_correction - previous_correction, -maximum_delta, maximum_delta)
                correction = clamp(previous_correction + correction_delta, -limit, limit)
                compensated = self._clamp_command_angle_deg(name, requested + correction)
                correction = compensated - requested
                self.servo_tracking_compensation_deg[name] = correction
                applied = compensated
                if measured_deg is not None and abs(actual_error_deg) <= 0.75:
                    self.servo_tracking_stable_ticks[name] = int(self.servo_tracking_stable_ticks.get(name, 0)) + 1
                else:
                    self.servo_tracking_stable_ticks[name] = 0
            self.servo_applied_command_deg[name] = self._clamp_command_angle_deg(name, applied)

    def begin_servo_tracking(self, joint_names: Any) -> None:
        """Keep feedback active for the scheduler-owned segment lifecycle."""

        for raw_name in joint_names:
            name = resolve_servo_name(str(raw_name))
            self.servo_tracking_active[name] = True
            self.servo_tracking_stable_ticks[name] = 0

    def end_servo_tracking(self, joint_names: Any) -> None:
        """Freeze the converged load bias after the segment completes."""

        for raw_name in joint_names:
            name = resolve_servo_name(str(raw_name))
            self.servo_tracking_active[name] = False

    def play_steps_blocking(
        self,
        steps: list[dict[str, Any]],
        *,
        profile: str = "fast",
        label: str = "accepted steps",
    ) -> None:
        plan = plan_from_steps(
            steps,
            profile=profile,
            max_wheel_speed=self.max_wheel_speed,
            label=label,
        )
        self.play_plan_blocking(plan)

    def play_plan_blocking(self, plan: PlaybackPlan) -> None:
        respawn = self.respawn_robot()
        if not bool(respawn.get("ok", True)):
            raise RuntimeError(str(respawn.get("error", "grounded respawn failed before playback")))
        dt = float(self.sim.get_physics_dt())
        started_at = self.sim_time
        index = 0
        print(f"[INFO] Playback started: {plan.label or 'plan'} ({len(plan.events)} events).")
        collector = getattr(self, "telemetry_collector", None)
        if collector is not None:
            try:
                collector.start_replay(
                    label=plan.label or "plan",
                    event_count=len(plan.events),
                    final_time_s=float(plan.final_time_s),
                    started_sim_time_s=float(started_at),
                )
            except Exception as exc:
                print(f"[WARN] Telemetry replay start failed: {exc}")
        playback_success = False
        while self.scene_handle.app_is_running():
            elapsed = self.sim_time - started_at
            while index < len(plan.events) and plan.events[index].time_s <= elapsed + 1.0e-6:
                if collector is not None:
                    try:
                        collector.record_replay_event(self, plan.events[index], index)
                    except Exception as exc:
                        print(f"[WARN] Telemetry replay event failed: {exc}")
                self.handle_command(
                    CommandMessage(
                        text=plan.events[index].command,
                        source="playback",
                        log_history=False,
                        quiet=True,
                        playback_label=plan.label,
                        playback_event_index=index,
                        playback_event_count=len(plan.events),
                        playback_final_time_s=float(plan.final_time_s),
                        source_step=plan.events[index].source_step,
                    )
                )
                index += 1
            self.step(dt)
            if index >= len(plan.events) and elapsed >= plan.final_time_s:
                playback_success = True
                break
            time.sleep(0.0)
        self.stop_wheels()
        self.apply_commands_to_robot()
        self.robot.write_data_to_sim()
        if collector is not None:
            try:
                collector.finish_replay(success=playback_success, reason="" if playback_success else "simulation app stopped", sim_time_s=float(self.sim_time))
            except Exception as exc:
                print(f"[WARN] Telemetry replay finish failed: {exc}")
        print("[INFO] Playback complete; wheels stopped.")

    def apply_command_space_joint_limits(self) -> None:
        self.apply_safe_command_space_joint_limits_to_sim()

    def apply_safe_command_space_joint_limits_to_sim(self) -> None:
        self.update_safe_command_space_joint_limit_records(write_to_sim=True)

    def update_safe_command_space_joint_limit_records(self, *, write_to_sim: bool = False) -> None:
        for joint_name in SERVO_JOINT_NAMES:
            joint_id = self.servo_name_to_id[joint_name]
            final_min_deg, final_max_deg = self.get_final_target_limits_deg(joint_name)
            record = safe_rad_limits_for_actual_deg(final_min_deg, final_max_deg)
            record.update(
                {
                    "joint_name": joint_name,
                    "joint_id": int(joint_id),
                    "desired_min_deg": final_min_deg,
                    "desired_max_deg": final_max_deg,
                    "applied": False,
                    "error": "",
                    "write_requested": bool(write_to_sim),
                    "write_skipped_reason": "",
                }
            )
            self.safe_joint_limit_records[joint_name] = record
            for warning in record.get("warnings", []):
                print(f"[WARN] Safe servo limit for {joint_name}: {warning}")
            if record["skip"]:
                print(f"[WARN] Skipping safe servo limit write for {joint_name}: {record['skip_reason']}")
                continue
            if not bool(write_to_sim):
                record["write_skipped_reason"] = "--apply-physx-joint-limits is disabled"
                continue
            physx_min_rad, physx_max_rad, adjusted_for_physx = physx_write_rad_limits_from_record(record)
            if physx_min_rad >= physx_max_rad:
                record["write_skipped_reason"] = "physx-safe min_rad >= max_rad"
                print(f"[WARN] Skipping safe servo limit write for {joint_name}: {record['write_skipped_reason']}")
                continue
            record["physx_write_min_rad"] = physx_min_rad
            record["physx_write_max_rad"] = physx_max_rad
            record["physx_write_adjusted"] = bool(adjusted_for_physx)
            try:
                prim_path = self._write_joint_position_limit_to_runtime_stage(
                    joint_name,
                    joint_id,
                    physx_min_rad,
                    physx_max_rad,
                )
                record["applied"] = True
                record["write_source"] = "runtime_usd_stage_override"
                record["runtime_joint_prim_path"] = prim_path
            except Exception as exc:
                record["error"] = str(exc)
                print(f"[WARN] Could not write safe servo limit for {joint_name}: {exc}")

    def _write_joint_position_limit_to_runtime_stage(
        self,
        joint_name: str,
        joint_id: int,
        min_rad: float,
        max_rad: float,
    ) -> str:
        """Override one live USD revolute joint without rewriting wheel limits.

        Isaac Lab's public tensor helper sends the *entire* DOF-limit array on
        every call.  Continuous wheel joints contain unlimited sentinels, which
        causes PhysX ``setLimitParams`` errors and makes a one-joint update
        unsafe.  A session-layer USD attribute edit targets only the named
        servo joint and does not modify the source USD/URDF asset.
        """

        from isaaclab.sim import get_current_stage  # type: ignore
        from pxr import Usd, UsdPhysics  # type: ignore

        stage = get_current_stage()
        root_path = str(getattr(self.scene_handle, "robot_prim_path", "") or "/World/WLRRobot")
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"runtime robot prim is missing: {root_path}")
        matches: list[Any] = []
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint):
                matches.append(prim)
        if len(matches) != 1:
            paths = [str(prim.GetPath()) for prim in matches]
            raise RuntimeError(f"expected one RevoluteJoint prim named {joint_name}, found {paths}")
        joint = UsdPhysics.RevoluteJoint(matches[0])
        joint.CreateLowerLimitAttr().Set(float(math.degrees(min_rad)))
        joint.CreateUpperLimitAttr().Set(float(math.degrees(max_rad)))

        # Keep Isaac Lab's diagnostic/soft-limit cache consistent.  This does
        # not write joint state or affect continuous wheel DOFs.
        limits = self.robot.data.joint_pos_limits
        limits[:, int(joint_id), 0] = float(min_rad)
        limits[:, int(joint_id), 1] = float(max_rad)
        soft_factor = float(getattr(getattr(self.robot, "cfg", None), "soft_joint_pos_limit_factor", 1.0))
        midpoint = 0.5 * (float(min_rad) + float(max_rad))
        half_range = 0.5 * (float(max_rad) - float(min_rad)) * soft_factor
        self.robot.data.soft_joint_pos_limits[:, int(joint_id), 0] = midpoint - half_range
        self.robot.data.soft_joint_pos_limits[:, int(joint_id), 1] = midpoint + half_range
        return str(matches[0].GetPath())

    def get_final_target_limits_deg(self, joint_name: str) -> tuple[float, float]:
        return desired_actual_limits_deg(joint_name, self.standing_pose_deg[joint_name])

    def command_to_actual_target_deg(self, joint_name: str, command_angle_deg: float) -> float:
        return actual_target_deg_for_command(joint_name, command_angle_deg, self.standing_pose_deg[joint_name])

    def _wheel_command_metadata(self, message: Any) -> dict[str, Any]:
        return {
            "generation": getattr(message, "wheel_generation", None),
            "command_id": str(getattr(message, "command_id", "") or ""),
            "requested_wall_time": float(getattr(message, "requested_wall_time", 0.0) or 0.0),
            "enqueued_wall_time": float(getattr(message, "enqueued_wall_time", 0.0) or 0.0),
        }

    def _handle_one_command(self, command: str, *, message: Any = None) -> None:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            print(f"[WARN] Could not parse command {command!r}: {exc}")
            return
        if not tokens:
            return
        verb = tokens[0].lower()
        wheel_metadata = self._wheel_command_metadata(message)
        try:
            if verb == "w":
                self.set_all_wheels(self.default_wheel_speed, **wheel_metadata)
            elif verb == "s":
                self.set_all_wheels(-self.default_wheel_speed, **wheel_metadata)
            elif verb == "x" or verb == "stop":
                self.stop_wheels(**wheel_metadata)
            elif verb == "a":
                self.set_turn("left", **wheel_metadata)
            elif verb == "d":
                self.set_turn("right", **wheel_metadata)
            elif verb in {"wheel", "wheels", "speed"}:
                self._handle_wheel_command(tokens, metadata=wheel_metadata)
            elif verb in {"servo", "angle"}:
                self._handle_servo_command(tokens)
            elif verb == "home":
                self.home()
            elif verb == "reset":
                self.reset_robot()
            elif verb == "respawn":
                self.respawn_robot()
            else:
                print(f"[WARN] Unsupported replay command ignored: {command}")
        except Exception as exc:
            print(f"[WARN] Command failed: {command}: {exc}")

    def _handle_wheel_command(self, tokens: list[str], *, metadata: dict[str, Any]) -> None:
        verb = tokens[0].lower()
        args = tokens[1:]
        if verb in {"wheels", "speed"} or (verb == "wheel" and len(args) == 2 and is_float_token(args[0])):
            if len(args) != 2:
                raise ValueError("Usage: wheels <left_rad_s> <right_rad_s>")
            self.set_wheels(float(args[0]), float(args[1]), **metadata)
            return
        if verb != "wheel" or not args:
            raise ValueError("Usage: wheel all|fl|fr|rl|rr <rad_s> or wheel stop")
        sub = args[0].lower()
        if sub == "stop":
            self.stop_wheels(**metadata)
        elif sub == "all" and len(args) == 2:
            self.set_all_wheels(float(args[1]), **metadata)
        elif len(args) == 2:
            self.set_one_wheel(sub, float(args[1]), **metadata)
        else:
            raise ValueError("Usage: wheel all|fl|fr|rl|rr <rad_s> or wheel stop")

    def _handle_servo_command(self, tokens: list[str]) -> None:
        if len(tokens) == 4 and tokens[2].lower() in {"hip", "knee"}:
            targets = resolve_servo_targets_for_group_part(tokens[1], tokens[2])
            self.set_servo_targets(targets, float(tokens[3]))
        elif len(tokens) == 3:
            self.set_servo(tokens[1], float(tokens[2]))
        else:
            raise ValueError("Usage: servo <joint|group> <deg> or servo <group> hip|knee <deg>")

    def _targets_from_command_angles(self) -> Any:
        targets = self.command_zero_joint_pos[:, self.servo_joint_ids].clone()
        for local_index, name in enumerate(SERVO_JOINT_NAMES):
            requested_command_deg = self._clamp_command_angle_deg(name, float(self.joint_command_deg[name]))
            self.joint_command_deg[name] = requested_command_deg
            raw_command_deg = float(self.servo_applied_command_deg.get(name, requested_command_deg))
            command_deg = self._clamp_command_angle_deg(name, raw_command_deg)
            self.servo_applied_command_deg[name] = command_deg
            actual_target_deg = self.command_to_actual_target_deg(name, command_deg)
            final_min, final_max = self.get_final_target_limits_deg(name)
            final_target_deg = clamp(actual_target_deg, final_min, final_max)
            target_rad = math.radians(final_target_deg)
            targets[:, local_index] = target_rad
            self.servo_target_debug[name] = {
                "joint_name": name,
                "raw_command_deg": raw_command_deg,
                "clamped_command_deg": command_deg,
                "command_deg": requested_command_deg,
                "applied_command_deg": command_deg,
                "standing_pose_deg": self.standing_pose_deg[name],
                "command_sign": JOINT_COMMAND_SIGN[name],
                "actual_target_deg": actual_target_deg,
                "final_target_deg": final_target_deg,
                "target_actual_deg": final_target_deg,
                "target_actual_rad": target_rad,
                "final_limit_min_deg": final_min,
                "final_limit_max_deg": final_max,
                "is_knee": name in KNEE_JOINT_NAMES,
            }
            self._warn_if_negative_knee_target_was_eaten(name, command_deg, final_target_deg)
        return targets

    def get_target_joint_state(self) -> dict[str, Any]:
        return {
            "servos": {
                name: {
                    "command_deg": float(self.joint_command_deg[name]),
                    "applied_command_deg": float(self.servo_applied_command_deg.get(name, self.joint_command_deg[name])),
                    "tracking_compensation_deg": float(self.servo_tracking_compensation_deg.get(name, 0.0)),
                    "target_actual_deg": float(self.command_to_actual_target_deg(name, self.joint_command_deg[name])),
                    "target_actual_rad": float(math.radians(self.command_to_actual_target_deg(name, self.joint_command_deg[name]))),
                }
                for name in SERVO_JOINT_NAMES
            },
            "wheels": {
                name: {
                    "canonical_rad_s": float(self.wheel_speeds[name]),
                    "requested_rad_s": float(self._effective_wheel_speed(self.wheel_speeds[name])[0]),
                    "target_rad_s": float(self._effective_wheel_speed(self.wheel_speeds[name])[1]),
                }
                for name in WHEEL_JOINT_NAMES
            },
            "wheel_command": dict(self.wheel_command_status),
        }

    def get_actual_joint_state(self) -> dict[str, Any]:
        servos: dict[str, dict[str, float | None]] = {}
        for name in SERVO_JOINT_NAMES:
            rad = self._measured_joint_rad(name)
            joint_id = self.servo_name_to_id[name]
            velocity_rad_s = self._tensor_joint_scalar("joint_vel", joint_id)
            servos[name] = {
                "rad": rad,
                "deg": None if rad is None else math.degrees(rad),
                "velocity_rad_s": velocity_rad_s,
                "velocity_deg_s": None if velocity_rad_s is None else math.degrees(velocity_rad_s),
                "applied_torque_nm": self._tensor_joint_scalar("applied_torque", joint_id),
                "computed_torque_nm": self._tensor_joint_scalar("computed_torque", joint_id),
                "effort_limit_nm": self._tensor_joint_scalar("joint_effort_limits", joint_id),
            }
        wheels = {
            name: {"rad_s": self._measured_wheel_velocity_rad_s(name)}
            for name in WHEEL_JOINT_NAMES
        }
        return {"servos": servos, "wheels": wheels}

    def _tensor_joint_scalar(self, field: str, joint_id: int) -> float | None:
        try:
            return float(getattr(self.robot.data, field)[0, int(joint_id)].item())
        except Exception:
            return None

    def _measured_wheel_velocity_rad_s(self, wheel_name: str) -> float | None:
        try:
            joint_id = self.wheel_name_to_id[wheel_name]
            physical = float(self.robot.data.joint_vel[0, joint_id].item())
            sign = self.wheel_direction * WHEEL_FORWARD_SIGN[wheel_name]
            return physical / sign if abs(sign) > 1.0e-12 else physical
        except Exception:
            return None

    def _update_wheel_stop_measurement(self) -> None:
        measured = {name: self._measured_wheel_velocity_rad_s(name) for name in WHEEL_JOINT_NAMES}
        self.wheel_command_status["measured_velocity_rad_s"] = measured
        if not self.wheel_command_status.get("zero_target_applied"):
            return
        finite = [abs(float(value)) for value in measured.values() if value is not None and math.isfinite(float(value))]
        stopped = bool(finite) and max(finite) <= float(self.wheel_stop_tolerance_rad_s)
        self.wheel_command_status["physically_stopped"] = stopped
        self.wheel_command_status["state"] = "physically_stopped" if stopped else "decelerating"
        if stopped and not self.wheel_command_status.get("measured_stop_wall_time"):
            self.wheel_command_status["measured_stop_wall_time"] = time.time()
            self.wheel_command_status["measured_stop_sim_time"] = float(self.sim_time)

    def get_joint_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for name in SERVO_JOINT_NAMES:
            joint_id = self.servo_name_to_id[name]
            command_limit = command_limits_for_servo(name)
            target = self.servo_target_debug.get(name, {})
            measured_rad = self._measured_joint_rad(name)
            current_limit_rad, current_limit_source = self._read_current_joint_limit_rad(joint_id)
            current_limit_deg = (
                None
                if current_limit_rad is None
                else [math.degrees(float(current_limit_rad[0])), math.degrees(float(current_limit_rad[1]))]
            )
            if current_limit_rad is None and name not in self._limit_unknown_warning_cache:
                self._limit_unknown_warning_cache.add(name)
                print(f"[WARN] Could not read current PhysX/USD joint limit for {name}; diagnostics will report unknown.")
            target_rad = float(target.get("target_actual_rad", math.radians(self.command_to_actual_target_deg(name, self.joint_command_deg[name]))))
            if current_limit_rad is None:
                target_inside_limit: bool | str = "unknown"
            else:
                target_inside_limit = float(current_limit_rad[0]) <= target_rad <= float(current_limit_rad[1])
            diagnostics.append(
                {
                    "joint_name": name,
                    "joint_id": int(joint_id),
                    "is_knee": name in KNEE_JOINT_NAMES,
                    "command_deg": float(self.joint_command_deg[name]),
                    "raw_command_deg": float(target.get("raw_command_deg", self.joint_command_deg[name])),
                    "clamped_command_deg": float(target.get("clamped_command_deg", self.joint_command_deg[name])),
                    "command_limit_deg": [float(command_limit[0]), float(command_limit[1])],
                    "standing_pose_deg": float(self.standing_pose_deg[name]),
                    "command_sign": float(JOINT_COMMAND_SIGN[name]),
                    "target_actual_deg": math.degrees(target_rad),
                    "target_actual_rad": target_rad,
                    "measured_joint_pos_deg": None if measured_rad is None else math.degrees(measured_rad),
                    "measured_joint_pos_rad": measured_rad,
                    "current_physx_or_usd_limit_rad": current_limit_rad,
                    "current_physx_or_usd_limit_deg": current_limit_deg,
                    "current_limit_source": current_limit_source,
                    "target_inside_current_limit": target_inside_limit,
                    "safe_limit_record": self.safe_joint_limit_records.get(name, {}),
                }
            )
        return diagnostics

    def _wheel_velocity_targets(self) -> Any:
        torch = self._torch()
        ordered_targets = [
            self.wheel_direction * WHEEL_FORWARD_SIGN[name] * self._effective_wheel_speed(self.wheel_speeds[name])[1]
            for name in WHEEL_JOINT_NAMES
        ]
        return torch.tensor(ordered_targets, dtype=torch.float32, device=self.device).reshape(1, len(WHEEL_JOINT_NAMES))

    def _set_joint_command_deg(self, joint_name: str, command_angle_deg: float) -> str:
        name = resolve_servo_name(joint_name)
        requested = self._clamp_command_angle_deg(name, command_angle_deg)
        if not math.isclose(requested, float(self.joint_command_deg.get(name, requested)), abs_tol=1.0e-9):
            self.servo_nominal_target_reached[name] = False
            self.servo_tracking_compensation_deg[name] = 0.0
            self.servo_tracking_active[name] = True
            self.servo_tracking_stable_ticks[name] = 0
        self.joint_command_deg[name] = requested
        return name

    def _clamp_command_angle_deg(self, joint_name: str, command_angle_deg: float) -> float:
        input_angle = float(command_angle_deg)
        clamped = clamp_servo_command(joint_name, input_angle)
        if not math.isclose(clamped, input_angle):
            print(f"[WARN] {joint_name} command clamped from {input_angle:g} to {clamped:g}")
        return clamped

    def _clamp_wheel_speed(self, speed: float) -> float:
        return clamp(float(speed), -self.max_wheel_speed, self.max_wheel_speed)

    def _effective_wheel_speed(self, canonical_speed: float) -> tuple[float, float]:
        requested = float(canonical_speed) * self.speed_scale.scale
        effective = clamp(requested, -self.max_wheel_speed, self.max_wheel_speed)
        return requested, effective

    def _warn_if_negative_knee_target_was_eaten(self, joint_name: str, command_deg: float, final_target_deg: float) -> None:
        if joint_name not in KNEE_JOINT_NAMES or command_deg >= 0.0:
            return
        standing = self.standing_pose_deg[joint_name]
        expected_delta = JOINT_COMMAND_SIGN[joint_name] * command_deg
        actual_delta = final_target_deg - standing
        wrong_direction = (expected_delta < 0.0 and actual_delta >= -1.0e-6) or (expected_delta > 0.0 and actual_delta <= 1.0e-6)
        if wrong_direction and joint_name not in self._target_warning_cache:
            self._target_warning_cache.add(joint_name)
            print(
                f"[WARN] {joint_name} negative command may be blocked before target write: "
                f"command={command_deg:g} standing={standing:g} target={final_target_deg:g}"
            )

    def _measured_joint_rad(self, joint_name: str) -> float | None:
        try:
            joint_id = self.servo_name_to_id[joint_name]
            return float(self.robot.data.joint_pos[0, joint_id].item())
        except Exception:
            return None

    def _current_root_z(self) -> float | None:
        try:
            return float(self.robot.data.root_pose_w[0, 2].item())
        except Exception:
            try:
                return float(self.robot.data.root_pose_w[0][2])
            except Exception:
                return None

    def _current_root_velocity(self) -> list[float]:
        try:
            values = self.robot.data.root_vel_w[0].detach().cpu().tolist()
            return [float(value) for value in values]
        except Exception:
            try:
                return [float(value) for value in self.robot.data.root_vel_w[0]]
            except Exception:
                return [0.0] * 6

    def _max_abs_joint_velocity(self) -> float:
        try:
            value = self.robot.data.joint_vel.detach().abs().max().item()
            return float(value)
        except Exception:
            try:
                return max(abs(float(value)) for value in self.robot.data.joint_vel[0])
            except Exception:
                return 0.0

    def _joint_velocity_vector(self) -> list[float]:
        try:
            return [float(value) for value in self.robot.data.joint_vel[0].detach().cpu().reshape(-1).tolist()]
        except Exception:
            try:
                return [float(value) for value in self.robot.data.joint_vel[0]]
            except Exception:
                return []

    def _joint_velocity_by_name(self) -> dict[str, float]:
        values = self._joint_velocity_vector()
        names = list(getattr(self.robot, "joint_names", []) or [])
        return {str(name): float(values[index]) for index, name in enumerate(names[: len(values)])}

    def _max_abs_named_joint_velocity(self, names: list[str] | tuple[str, ...], values: dict[str, float] | None = None) -> float:
        by_name = values if values is not None else self._joint_velocity_by_name()
        speeds = [abs(float(by_name.get(name, 0.0))) for name in names]
        return max(speeds + [0.0])

    def _servo_position_error_by_name(self) -> dict[str, float]:
        errors: dict[str, float] = {}
        debug_map = getattr(self, "servo_target_debug", {})
        if not isinstance(debug_map, dict):
            debug_map = {}
        for name in SERVO_JOINT_NAMES:
            measured = self._measured_joint_rad(name)
            target_debug = debug_map.get(name, {})
            target = target_debug.get("target_actual_rad")
            if measured is None or target is None:
                continue
            try:
                errors[name] = float(measured) - float(target)
            except Exception:
                continue
        return errors

    def _offending_joint_names(
        self,
        values: dict[str, float],
        *,
        servo_threshold: float,
        wheel_threshold: float,
        include_wheels: bool = True,
    ) -> list[str]:
        offenders: list[str] = []
        for name in SERVO_JOINT_NAMES:
            if abs(float(values.get(name, 0.0))) > float(servo_threshold):
                offenders.append(name)
        if include_wheels:
            for name in WHEEL_JOINT_NAMES:
                if abs(float(values.get(name, 0.0))) > float(wheel_threshold):
                    offenders.append(name)
        return offenders

    def _wheel_velocity_target_by_name(self) -> dict[str, float]:
        try:
            targets = self._wheel_velocity_targets().detach().cpu().reshape(-1).tolist()
            return {name: float(targets[index]) for index, name in enumerate(WHEEL_JOINT_NAMES[: len(targets)])}
        except Exception:
            return {
                name: float(self.wheel_direction * WHEEL_FORWARD_SIGN[name] * self._clamp_wheel_speed(self.wheel_speeds.get(name, 0.0)))
                for name in WHEEL_JOINT_NAMES
            }

    def _apply_root_z_correction(self, dz_m: float) -> None:
        dz = max(0.0, float(dz_m))
        if dz <= 0.0:
            return
        root_pose = self.robot.data.root_pose_w.clone()
        root_pose[:, 2] = root_pose[:, 2] + dz
        joint_pos = self.robot.data.joint_pos.clone()
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(self.zero_root_velocity)
        self.robot.write_joint_state_to_sim(joint_pos, self.command_zero_joint_vel)
        self.robot.reset()
        self.robot.update(0.0)

    def _diagnostics_after_initial_safe_placement(self, dz_m: float, *, reason: str = "") -> dict[str, Any]:
        diagnostics = self.validate_robot_ground_contact(apply_correction=False)
        diagnostics["initial_safe_placement_policy"] = "settle_then_bounded_root_z_correction"
        diagnostics["initial_safe_placement_correction_reason"] = str(reason or "")
        diagnostics["correction_applied"] = True
        diagnostics["correction_z_m"] = float(dz_m)
        return diagnostics

    def _corrected_ground_diagnostics_safe(self, diagnostics: dict[str, Any]) -> bool:
        classification = str(diagnostics.get("classification", "") or "")
        max_penetration = float(diagnostics.get("maximum_collision_penetration_m", 0.0) or 0.0)
        return bool(
            diagnostics.get("checked", False)
            and diagnostics.get("physical_ground_safe", False)
            and not diagnostics.get("missing_collision_wheels")
            and not diagnostics.get("unresolved_collision_wheels")
            and max_penetration <= float(self.config.ground_penetration_tolerance_m)
            and classification in {GROUND_OK, VISUAL_ONLY_INTERSECTION}
        )

    def _read_current_joint_limit_rad(self, joint_id: int) -> tuple[list[float] | None, str]:
        for owner_name, owner in (("robot.data", getattr(self.robot, "data", None)), ("robot", self.robot)):
            if owner is None:
                continue
            for attr in ("soft_joint_pos_limits", "joint_pos_limits", "joint_limits"):
                value = getattr(owner, attr, None)
                parsed = self._parse_limit_tensor(value, joint_id)
                if parsed is not None:
                    return parsed, f"{owner_name}.{attr}"
        return None, "unknown"

    @staticmethod
    def _parse_limit_tensor(value: Any, joint_id: int) -> list[float] | None:
        if value is None:
            return None
        try:
            shape = tuple(value.shape)
            if len(shape) >= 3 and shape[-1] == 2:
                return [float(value[0, joint_id, 0].item()), float(value[0, joint_id, 1].item())]
            if len(shape) >= 2 and shape[-1] == 2:
                return [float(value[joint_id, 0].item()), float(value[joint_id, 1].item())]
        except Exception:
            return None
        return None

    def _resolve_exact_joint_ids(self, names: list[str]) -> list[int]:
        name_to_id = {joint_name: index for index, joint_name in enumerate(self.robot.joint_names)}
        missing = [name for name in names if name not in name_to_id]
        if missing:
            print(f"[ERROR] Missing required joints: {missing}")
            print(f"[ERROR] Available joints: {self.robot.joint_names}")
            raise ValueError(f"Missing required joints: {missing}")
        return [name_to_id[name] for name in names]

    @staticmethod
    def _torch() -> Any:
        import torch  # type: ignore

        return torch

    @staticmethod
    def _tensor_to_nested_list(value: Any) -> Any:
        try:
            return value.detach().cpu().tolist()
        except Exception:
            try:
                return value.cpu().tolist()
            except Exception:
                try:
                    return value.tolist()
                except Exception:
                    return None


def _max_abs_by_name(frames: list[dict[str, Any]], names: list[str] | tuple[str, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        values = [
            abs(float((frame.get("joint_velocity_by_name", {}) or {}).get(name, 0.0) or 0.0))
            for frame in frames
        ]
        result[str(name)] = max(values + [0.0])
    return result


def _rms(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return math.sqrt(sum(value * value for value in finite) / float(len(finite)))


def _percentile(values: list[float], percentile: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    rank = max(0.0, min(100.0, float(percentile))) / 100.0 * float(len(finite) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return finite[low]
    fraction = rank - float(low)
    return finite[low] * (1.0 - fraction) + finite[high] * fraction


class NullSimRobotAdapter:
    """Fallback adapter used before Isaac Sim is generated."""

    def __init__(self):
        self.command_state = empty_command_state()
        self.max_wheel_speed = REAL_MOTION_REFERENCE.wheel_velocity_limit_rad_s
        self.motion_reference = load_motion_reference()
        self.default_wheel_speed = self.motion_reference.wheel_reference_velocity_rad_s
        self.speed_scale = SpeedScale(self.motion_reference)
        self.motion_batch_status: dict[str, Any] = {}
        self.sim_time = 0.0
        self.sim_steps = 0
        self.grounded_reference_valid = False
        self.grounded_reference_physics_valid = False
        self.grounded_reference_visual_valid = False
        self.grounded_reference_stable = False
        self.respawn_ready = False
        self.ground_reference_block_reason = "no-sim adapter"
        self.grounded_reference_diagnostics = default_robot_ground_diagnostics("no-sim adapter")
        self.robot_ground_diagnostics = default_robot_ground_diagnostics("no-sim adapter")
        self.last_ground_correction_z_m = 0.0
        self.servo_motion_enabled = True
        self.wheel_generation = 0
        self.wheel_stop_tolerance_rad_s = WHEEL_PHYSICAL_STOP_NOISE_FLOOR_RAD_S
        self.wheel_command_status: dict[str, Any] = {
            "generation": 0,
            "state": "idle",
            "stop_command_received": False,
            "zero_target_applied": True,
            "physically_stopped": True,
            "stale_command_rejected": False,
            "applied_target_rad_s": dict(self.command_state["wheels"]),
            "measured_velocity_rad_s": dict(self.command_state["wheels"]),
        }

    def capture_command_state(self) -> dict[str, dict[str, float]]:
        return clone_command_state(self.command_state)

    def set_speed_percent(self, value: float) -> dict[str, Any]:
        self.speed_scale.set_percent(value)
        return self.speed_scale.status()

    def apply_motion_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        from sequence_model import apply_command_to_state

        batch_id = str(payload.get("batch_id", "") or uuid.uuid4().hex)
        for name, value in dict(payload.get("servo_targets_deg", {}) or {}).items():
            if name in self.command_state["servos"]:
                apply_command_to_state(self.command_state, f"servo {name} {float(value):.9g}")
        wheels = payload.get("wheel_base_velocity_rad_s", {})
        if isinstance(wheels, (int, float)):
            wheels = {name: float(wheels) for name in WHEEL_JOINT_NAMES}
        self.apply_wheel_velocity(dict(wheels or {}), command_id=batch_id)
        now = time.time()
        ack = {
            "batch_id": batch_id,
            "source": str(payload.get("source", "ui") or "ui"),
            "batch_received_wall_time": now,
            "batch_applied_wall_time": now,
            "batch_applied_sim_time": float(self.sim_time),
            "servo_applied": bool(payload.get("servo_targets_deg")),
            "wheel_applied": True,
            "first_physics_step": int(self.sim_steps + 1),
            "servo_motion_start_sim_time": float(self.sim_time),
            "wheel_motion_start_sim_time": float(self.sim_time),
            "motion_start_skew_s": 0.0,
            "physics_dt_s": 0.0,
            "speed_percent_snapshot": float(payload.get("speed_percent_snapshot", self.speed_scale.speed_percent)),
            "executor_speed_percent": self.speed_scale.speed_percent,
            "canonical_servo_targets_deg": dict(payload.get("servo_targets_deg", {}) or {}),
            "canonical_wheel_velocity_rad_s": dict(self.command_state["wheels"]),
            "effective_wheel_velocity_rad_s": {
                name: max(-self.max_wheel_speed, min(self.max_wheel_speed, value * self.speed_scale.scale))
                for name, value in self.command_state["wheels"].items()
            },
            "recording_metadata": dict(payload.get("recording_metadata", {}) or {}),
            "high_priority": False,
            "error": "",
        }
        self.motion_batch_status = ack
        return dict(ack)

    def handle_command(self, message: Any) -> None:
        from sequence_model import apply_command_to_state

        text = getattr(message, "text", message)
        for command in split_semicolon_commands(str(text)):
            validated = validate_motion_command(
                command,
                default_wheel_speed_rad_s=self.default_wheel_speed,
                max_wheel_speed_rad_s=self.max_wheel_speed,
            )
            if validated.is_wheel_command:
                desired = clone_command_state(self.command_state)
                apply_command_to_state(desired, validated.command)
                generation = getattr(message, "wheel_generation", None)
                if validated.is_wheel_stop:
                    self.stop_wheels(
                        generation=generation,
                        command_id=str(getattr(message, "command_id", "") or ""),
                        requested_wall_time=float(getattr(message, "requested_wall_time", 0.0) or 0.0),
                        enqueued_wall_time=float(getattr(message, "enqueued_wall_time", 0.0) or 0.0),
                    )
                else:
                    self.apply_wheel_velocity(
                        desired["wheels"],
                        generation=generation,
                        command_id=str(getattr(message, "command_id", "") or ""),
                    )
            else:
                apply_command_to_state(self.command_state, validated.command)

    def apply_wheel_velocity(
        self,
        targets_rad_s: dict[str, float],
        *,
        generation: int | None = None,
        command_id: str = "",
        high_priority: bool = False,
        requested_wall_time: float = 0.0,
        enqueued_wall_time: float = 0.0,
    ) -> bool:
        incoming = self.wheel_generation if generation is None else int(generation)
        nonzero = any(abs(float(value)) > 1.0e-12 for value in targets_rad_s.values())
        if nonzero and incoming < self.wheel_generation:
            self.wheel_command_status.update(state="stale_nonzero_rejected", stale_command_rejected=True)
            return False
        if high_priority and nonzero:
            raise ValueError("A high-priority wheel safety command must be zero.")
        self.wheel_generation = max(self.wheel_generation, incoming)
        now = time.time()
        self.command_state["wheels"] = {
            name: clamp(float(targets_rad_s.get(name, 0.0)), -self.max_wheel_speed, self.max_wheel_speed)
            for name in WHEEL_JOINT_NAMES
        }
        self.wheel_command_status = {
            "generation": self.wheel_generation,
            "command_id": command_id,
            "state": "physically_stopped" if not nonzero else "target_applied",
            "stop_command_received": not nonzero,
            "zero_target_applied": not nonzero,
            "physically_stopped": not nonzero,
            "stale_command_rejected": False,
            "requested_wall_time": requested_wall_time or now,
            "enqueued_wall_time": enqueued_wall_time or now,
            "received_wall_time": now,
            "target_applied_wall_time": now,
            "measured_stop_wall_time": now if not nonzero else 0.0,
            "target_applied_sim_time": self.sim_time,
            "measured_stop_sim_time": self.sim_time if not nonzero else 0.0,
            "stop_tolerance_rad_s": self.wheel_stop_tolerance_rad_s,
            "applied_target_rad_s": dict(self.command_state["wheels"]),
            "canonical_target_rad_s": dict(self.command_state["wheels"]),
            "effective_target_rad_s": {
                name: max(-self.max_wheel_speed, min(self.max_wheel_speed, value * self.speed_scale.scale))
                for name, value in self.command_state["wheels"].items()
            },
            "speed_percent": self.speed_scale.speed_percent,
            "measured_velocity_rad_s": dict(self.command_state["wheels"]),
        }
        return True

    def stop_wheels(
        self,
        *,
        generation: int | None = None,
        command_id: str = "",
        requested_wall_time: float = 0.0,
        enqueued_wall_time: float = 0.0,
    ) -> bool:
        next_generation = self.wheel_generation + 1 if generation is None else int(generation)
        return self.apply_wheel_velocity(
            {name: 0.0 for name in WHEEL_JOINT_NAMES},
            generation=next_generation,
            command_id=command_id,
            high_priority=True,
            requested_wall_time=requested_wall_time,
            enqueued_wall_time=enqueued_wall_time,
        )

    def step(self, dt: float | None = None) -> None:
        return None

    def respawn_robot(self, *, settle: bool = True) -> dict[str, Any]:
        self.handle_command("home")
        return {"ground_diagnostics": dict(self.robot_ground_diagnostics), "settle": {}}

    def initialize_grounded_respawn_reference(self) -> dict[str, Any]:
        return {"grounded_reference_valid": False, "grounded_reference_diagnostics": dict(self.grounded_reference_diagnostics)}

    def validate_robot_ground_contact(self, *, apply_correction: bool | None = None) -> dict[str, Any]:
        return dict(self.robot_ground_diagnostics)

    def capture_sim_state(self) -> dict[str, Any]:
        return {
            "command_state": self.capture_command_state(),
            "target_joint_state": self.get_target_joint_state(),
            "actual_joint_state": self.get_actual_joint_state(),
            "root_pose": None,
            "root_velocity": None,
            "joint_pos": None,
            "joint_vel": None,
            "adapter_sim_time": float(self.sim_time),
            "adapter_sim_steps": int(self.sim_steps),
            "grounded_reference_valid": bool(self.grounded_reference_valid),
            "grounded_reference_physics_valid": bool(self.grounded_reference_physics_valid),
            "grounded_reference_visual_valid": bool(self.grounded_reference_visual_valid),
            "grounded_reference_stable": bool(self.grounded_reference_stable),
            "grounded_reference_diagnostics": dict(self.grounded_reference_diagnostics),
            "robot_ground_diagnostics": dict(self.robot_ground_diagnostics),
            "wheel_command": dict(self.wheel_command_status),
        }

    def restore_sim_state(self, sim_state: dict[str, Any] | None) -> None:
        state = dict(sim_state or {})
        command_state = state.get("command_state")
        if isinstance(command_state, dict):
            self.command_state = clone_command_state(command_state)

    def get_target_joint_state(self) -> dict[str, Any]:
        return {
            "servos": {},
            "wheels": {name: {"target_rad_s": float(value)} for name, value in self.command_state["wheels"].items()},
            "wheel_command": dict(self.wheel_command_status),
        }

    def get_actual_joint_state(self) -> dict[str, Any]:
        return {
            "servos": {},
            "wheels": {name: {"rad_s": float(value)} for name, value in self.command_state["wheels"].items()},
        }

    def get_joint_diagnostics(self) -> list[dict[str, Any]]:
        return []
