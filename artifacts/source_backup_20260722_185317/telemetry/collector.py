"""Runtime telemetry collector for height-based obstacle replay."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .com_metrics import (
    WholeBodyComResult,
    body_com_positions_from_link_pose,
    compute_whole_body_com,
    quat_wxyz_to_rpy,
    to_numpy,
)
from .config import RuntimeTelemetryConfig, load_telemetry_config
from .contact_metrics import ContactMetricsResult, compute_contact_metrics
from .event_recorder import EventRecorder
from .exporters import create_run_dir, python_runtime_metadata, summarize_run, write_csv, write_json, write_jsonl, write_metadata, write_npz
from .joint_metrics import JointMetricState, JointMetricsResult, compute_joint_metrics
from .model_audit import audit_robot_model, write_model_audit
from .replay_metrics import ReplayContext, command_targets_from_adapter
from .stability_metrics import StabilityResult, compute_equilibrium_region, compute_geometric_stability, gravity_aligned_support_plane
from .visualization.live_overlay import LiveTelemetryOverlay


class TelemetryCollector:
    def __init__(self, config: RuntimeTelemetryConfig, *, args: Any | None = None, scene_handle: Any | None = None):
        self.config = config
        self.args = args
        self.scene_handle = scene_handle
        self.enabled = bool(config.telemetry.enabled)
        self.run_dir: Path | None = None
        self.event_recorder: EventRecorder | None = None
        self.overlay = LiveTelemetryOverlay(config.visualization, headless=bool(getattr(args, "headless", False)))
        self.rows: list[dict[str, Any]] = []
        self.body_rows: list[dict[str, Any]] = []
        self.joint_rows: list[dict[str, Any]] = []
        self.contact_rows: list[dict[str, Any]] = []
        self.joint_state = JointMetricState()
        self.replay_context = ReplayContext()
        self.started_wall = time.time()
        self.finished_wall = 0.0
        self.sample_index = 0
        self.dropped_samples = 0
        self.next_sample_time_s = 0.0
        self.last_flush_time_s = -math.inf
        self.last_equilibrium_time_s = -math.inf
        self.last_equilibrium: dict[str, Any] | None = None
        self.last_status: dict[str, Any] = {"enabled": self.enabled, "run_dir": ""}
        self.last_com_w: np.ndarray | None = None
        self.last_com_time_s: float | None = None
        self.last_row: dict[str, Any] | None = None
        self.overhead_total_s = 0.0
        self.overhead_max_s = 0.0
        self.sample_calls = 0
        self.step_calls = 0
        self.obstacle_height_cm: int | None = None
        self.obstacle_height_m: float | None = None
        self.sequence_label = "session"
        self.source = "runtime"

    def start_episode(
        self,
        *,
        adapter: Any | None = None,
        scene_handle: Any | None = None,
        obstacle_height_cm: int | None = None,
        obstacle_height_m: float | None = None,
        sequence_label: str = "session",
        source: str = "runtime",
    ) -> Path | None:
        if not self.enabled:
            return None
        if scene_handle is not None:
            self.scene_handle = scene_handle
        self.obstacle_height_cm = None if obstacle_height_cm is None else int(obstacle_height_cm)
        self.obstacle_height_m = None if obstacle_height_m is None else float(obstacle_height_m)
        self.sequence_label = str(sequence_label or "session")
        self.source = str(source or "runtime")
        self.started_wall = time.time()
        self.run_dir = create_run_dir(
            self.config.telemetry.output_root,
            height_cm=self.obstacle_height_cm,
            sequence_label=self.sequence_label,
        )
        self.event_recorder = EventRecorder(self.run_dir / "events.jsonl")
        metadata = self._metadata(adapter)
        write_metadata(self.run_dir, metadata, self.config)
        try:
            audit = audit_robot_model(self.scene_handle, adapter)
            write_model_audit(self.run_dir, audit)
        except Exception as exc:
            self.record_event(0.0, "model_audit_failed", severity="warning", message=str(exc))
        self.record_event(0.0, "telemetry_started", severity="info", message=f"Telemetry run started: {self.sequence_label}")
        self.last_status.update({"run_dir": str(self.run_dir), "sequence_label": self.sequence_label})
        return self.run_dir

    def on_step(self, adapter: Any, dt_s: float) -> None:
        if not self.enabled:
            return
        self.step_calls += 1
        sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        if self.run_dir is None:
            self.start_episode(
                adapter=adapter,
                scene_handle=getattr(adapter, "scene_handle", self.scene_handle),
                obstacle_height_cm=self.obstacle_height_cm,
                obstacle_height_m=self.obstacle_height_m,
                sequence_label=self.sequence_label,
                source=self.source,
            )
        interval = 1.0 / max(0.1, float(self.config.telemetry.sample_hz))
        if sim_time + 1.0e-9 < self.next_sample_time_s:
            return
        self.next_sample_time_s = sim_time + interval
        max_samples = int(self.config.telemetry.max_full_samples)
        if max_samples > 0 and len(self.rows) >= max_samples:
            self.dropped_samples += 1
            return
        started = time.perf_counter()
        try:
            row, body_rows, joint_result, contacts, stability, equilibrium = self._sample(adapter, dt_s, sim_time)
            elapsed_s = time.perf_counter() - started
            row["sample_overhead_ms"] = elapsed_s * 1000.0
            self.overhead_total_s += elapsed_s
            self.overhead_max_s = max(self.overhead_max_s, elapsed_s)
            self.sample_calls += 1
            self.rows.append(row)
            self.body_rows.extend(body_rows)
            self.joint_rows.extend(joint_result.rows)
            self.contact_rows.extend(contacts.contacts)
            self.last_row = row
            self._record_warning_events(row, joint_result, contacts, stability, equilibrium)
            self.overlay.update(row, stability, contacts, equilibrium)
            if sim_time - self.last_flush_time_s >= float(self.config.telemetry.flush_interval_s):
                self.flush()
                self.last_flush_time_s = sim_time
        except Exception as exc:
            self.record_event(sim_time, "telemetry_sample_failed", severity="warning", message=str(exc))

    def flush(self) -> None:
        if not self.enabled or self.run_dir is None:
            return
        if bool(self.config.telemetry.save_csv):
            write_csv(self.run_dir / "telemetry_samples.csv", self.rows)
            write_csv(self.run_dir / "body_com_timeseries.csv", self.body_rows)
            write_csv(self.run_dir / "joint_timeseries.csv", self.joint_rows)
            if bool(self.config.telemetry.save_contacts):
                write_csv(self.run_dir / "contacts.csv", self.contact_rows)
        if bool(self.config.telemetry.save_npz):
            write_npz(self.run_dir / "telemetry_timeseries.npz", self.rows)
        if bool(self.config.telemetry.save_events) and self.event_recorder is not None:
            self.event_recorder.flush()
        summary = summarize_run(
            self.rows,
            self.body_rows,
            self.joint_rows,
            self.contact_rows,
            self.events,
            started_wall=self.started_wall,
            finished_wall=self.finished_wall or time.time(),
            dropped_samples=self.dropped_samples,
        )
        write_json(self.run_dir / "stability_summary.json", summary)
        self.last_status.update(
            {
                "enabled": True,
                "run_dir": str(self.run_dir),
                "sample_count": len(self.rows),
                "event_count": len(self.events),
                "last_static_margin_m": (self.last_row or {}).get("static_stability_margin_m"),
                "last_stability_state": (self.last_row or {}).get("stability_state", ""),
                "mean_sample_overhead_ms": summary.get("mean_sample_overhead_ms"),
                "max_sample_overhead_ms": summary.get("max_sample_overhead_ms"),
            }
        )

    def finish_episode(self, *, success: bool = True, reason: str = "") -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        sim_time = float((self.last_row or {}).get("time_s", 0.0) or 0.0)
        self.finish_replay(success=success, reason=reason, sim_time_s=sim_time)
        self.finished_wall = time.time()
        self.record_event(
            sim_time,
            "telemetry_finished",
            severity="info" if success else "warning",
            message=str(reason or "Telemetry run finished"),
        )
        self.flush()
        report_status: dict[str, Any] = {}
        if self.run_dir is not None and bool(self.config.telemetry.report_on_finish):
            try:
                from .visualization.report_generator import generate_report

                report_status = generate_report(self.run_dir)
            except Exception as exc:
                report_status = {"ok": False, "error": str(exc)}
                write_json(self.run_dir / "report_status.json", report_status)
        self.overlay.clear()
        status = self.status()
        status["report"] = report_status
        return status

    def status(self) -> dict[str, Any]:
        status = dict(self.last_status)
        status.update(
            {
                "enabled": bool(self.enabled),
                "overlay": self.overlay.status(),
                "sample_calls": int(self.sample_calls),
                "step_calls": int(self.step_calls),
                "dropped_samples": int(self.dropped_samples),
            }
        )
        return status

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self.event_recorder.events if self.event_recorder is not None else [])

    def record_event(self, simulation_time_s: float, event_type: str, **kwargs: Any) -> None:
        if not self.enabled:
            return
        if self.event_recorder is None and self.run_dir is not None:
            self.event_recorder = EventRecorder(self.run_dir / "events.jsonl")
        if self.event_recorder is not None:
            self.event_recorder.record(float(simulation_time_s), event_type, **kwargs)

    def record_command(self, adapter: Any, message: Any, command: str) -> None:
        if not self.enabled:
            return
        sim_time = float(getattr(adapter, "sim_time", 0.0) or 0.0)
        source = str(getattr(message, "source", "ui") or "ui")
        if source == "playback" and not self.replay_context.active:
            self.start_replay(
                label=str(getattr(message, "playback_label", "") or "ui playback"),
                event_count=int(getattr(message, "playback_event_count", 0) or 0),
                final_time_s=float(getattr(message, "playback_final_time_s", 0.0) or 0.0),
                started_sim_time_s=sim_time,
            )
        event_index = getattr(message, "playback_event_index", None)
        source_step = getattr(message, "source_step", None)
        if event_index is not None:
            self.replay_context.event_index = int(event_index)
        if source_step is not None:
            self.replay_context.step_index = int(source_step)
        self.replay_context.last_command = str(command)
        self.replay_context.phase = "command"
        self.record_event(
            sim_time,
            "command_applied",
            severity="info",
            key=f"{source}:{event_index if event_index is not None else sim_time:.6f}:{command}",
            message=str(command),
            step_sequence=self.replay_context.sequence_name,
            phase=self.replay_context.phase,
            extra={
                "command_source": source,
                "playback_event_index": event_index,
                "source_step": source_step,
                "command_kind": str(getattr(message, "kind", "command") or "command"),
                "command_target": str(getattr(message, "target", "") or ""),
            },
        )

    def start_replay(self, *, label: str, event_count: int, final_time_s: float, started_sim_time_s: float) -> None:
        if not self.enabled:
            return
        self.replay_context = ReplayContext(
            active=True,
            sequence_name=str(label or "playback"),
            started_sim_time_s=float(started_sim_time_s),
            final_time_s=float(final_time_s),
            event_count=int(event_count),
            phase="started",
        )
        self.record_event(
            float(started_sim_time_s),
            "replay_started",
            severity="info",
            message=f"Replay started: {label}",
            step_sequence=str(label or "playback"),
        )

    def record_replay_event(self, adapter: Any, event: Any, event_index: int) -> None:
        if not self.enabled:
            return
        self.replay_context.event_index = int(event_index)
        self.replay_context.step_index = getattr(event, "source_step", None)
        self.replay_context.last_command = str(getattr(event, "command", ""))
        self.replay_context.phase = "command"
        self.record_event(
            float(getattr(adapter, "sim_time", 0.0) or 0.0),
            "replay_command",
            severity="info",
            key=f"replay:{event_index}:{self.replay_context.last_command}",
            message=self.replay_context.last_command,
            step_sequence=self.replay_context.sequence_name,
            phase=self.replay_context.phase,
            extra={"playback_event_index": int(event_index), "source_step": self.replay_context.step_index},
        )

    def finish_replay(self, *, success: bool, reason: str = "", sim_time_s: float | None = None) -> None:
        if not self.enabled or not self.replay_context.active:
            return
        end_time = float(sim_time_s if sim_time_s is not None else (self.last_row or {}).get("time_s", 0.0) or 0.0)
        self.replay_context.active = False
        self.replay_context.success = bool(success)
        self.replay_context.failure_reason = str(reason or "")
        self.replay_context.phase = "complete" if success else "failed"
        self.record_event(
            end_time,
            "replay_finished",
            severity="info" if success else "warning",
            message=str(reason or ("Replay complete" if success else "Replay failed")),
            step_sequence=self.replay_context.sequence_name,
            phase=self.replay_context.phase,
        )

    def update_obstacle_context(self, *, height_cm: int, height_m: float | None = None, source: str = "", request_id: str = "") -> None:
        self.obstacle_height_cm = int(height_cm)
        self.obstacle_height_m = None if height_m is None else float(height_m)
        self.record_event(
            float((self.last_row or {}).get("time_s", 0.0) or 0.0),
            "obstacle_height_changed",
            severity="info",
            obstacle=f"{int(height_cm)}cm",
            message=f"Obstacle height set to {int(height_cm)}cm",
            extra={"source": str(source or ""), "request_id": str(request_id or ""), "height_m": self.obstacle_height_m},
        )

    def _sample(
        self,
        adapter: Any,
        dt_s: float,
        sim_time: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], JointMetricsResult, ContactMetricsResult, StabilityResult, dict[str, Any] | None]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        joint_names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
        body_names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
        env_id = int(self.config.telemetry.env_id)
        root_pose = _env_row(getattr(data, "root_pose_w", None), env_id, 7, fill=np.nan)
        root_vel = _env_row(getattr(data, "root_vel_w", None), env_id, 6, fill=np.nan)
        roll, pitch, yaw = quat_wxyz_to_rpy(root_pose[3:7])
        body_com_pos_w, body_com_source = _body_com_positions_w(data, env_id)
        masses = _body_masses(data, env_id, len(body_names))
        com = compute_whole_body_com(body_com_pos_w, masses, source=body_com_source)
        com_velocity = self._com_velocity(data, env_id, masses, com, sim_time, dt_s)
        body_rows = self._body_rows(body_names, body_com_pos_w, masses, com, sim_time)
        contacts = compute_contact_metrics(
            adapter=adapter,
            scene_handle=getattr(adapter, "scene_handle", self.scene_handle),
            contact_sensor=getattr(getattr(adapter, "scene_handle", self.scene_handle), "contact_sensor", None),
            env_id=env_id,
            simulation_time_s=sim_time,
            force_threshold_n=float(self.config.stability.contact_force_threshold_n),
            friction_default=float(self.config.stability.friction_coefficient_default),
            wheel_radius_m=None,
            slip_warning_threshold=float(self.config.warnings.wheel_slip_ratio),
        )
        stability = compute_geometric_stability(
            contacts.support_points_w,
            com.com_w,
            normal_forces_n=contacts.normal_forces_n,
            contact_force_threshold_n=float(self.config.stability.contact_force_threshold_n),
            safe_margin_m=float(self.config.stability.safe_margin_m),
            warning_margin_m=float(self.config.stability.warning_margin_m),
        )
        predicted_com = com.com_w + np.asarray(com_velocity, dtype=float) * 0.20
        dynamic_stability = compute_geometric_stability(
            contacts.support_points_w,
            predicted_com,
            normal_forces_n=contacts.normal_forces_n,
            contact_force_threshold_n=float(self.config.stability.contact_force_threshold_n),
            safe_margin_m=float(self.config.stability.safe_margin_m),
            warning_margin_m=float(self.config.stability.warning_margin_m),
        )
        equilibrium = self._equilibrium(sim_time, contacts, com, stability)
        commanded_pos, commanded_vel, command_source = command_targets_from_adapter(adapter, joint_names)
        joint_pos_target = _first_available(data, "joint_pos_target")
        joint_vel_target = _first_available(data, "joint_vel_target")
        joint_result = compute_joint_metrics(
            joint_names=joint_names,
            joint_pos=getattr(data, "joint_pos", None),
            joint_vel=getattr(data, "joint_vel", None),
            commanded_pos=joint_pos_target if joint_pos_target is not None else commanded_pos,
            commanded_vel=joint_vel_target if joint_vel_target is not None else commanded_vel,
            commanded_torque=_first_available(data, "joint_effort_target"),
            applied_torque=_first_available(data, "applied_torque", "computed_torque"),
            position_limits=_first_available(data, "soft_joint_pos_limits", "joint_pos_limits"),
            velocity_limits=_first_available(data, "soft_joint_vel_limits", "joint_vel_limits"),
            effort_limits=_first_available(data, "soft_joint_effort_limits", "joint_effort_limits"),
            state=self.joint_state,
            time_s=sim_time,
            dt_s=dt_s,
            acceleration_filter_enabled=bool(self.config.filters.acceleration_filter_enabled),
            acceleration_cutoff_hz=float(self.config.filters.acceleration_cutoff_hz),
            torque_warning_threshold=float(self.config.warnings.torque_utilization),
        )
        replay = self.replay_context.snapshot(sim_time)
        row = {
            "sample_index": int(self.sample_index),
            "time_s": float(sim_time),
            "sim_step": int(getattr(adapter, "sim_steps", 0) or 0),
            "obstacle_height_cm": self.obstacle_height_cm,
            "obstacle_height_m": self.obstacle_height_m,
            "base_x_m": _float(root_pose[0]),
            "base_y_m": _float(root_pose[1]),
            "base_z_m": _float(root_pose[2]),
            "base_qw": _float(root_pose[3]),
            "base_qx": _float(root_pose[4]),
            "base_qy": _float(root_pose[5]),
            "base_qz": _float(root_pose[6]),
            "base_roll_rad": float(roll),
            "base_pitch_rad": float(pitch),
            "base_yaw_rad": float(yaw),
            "base_vx_m_s": _float(root_vel[0]),
            "base_vy_m_s": _float(root_vel[1]),
            "base_vz_m_s": _float(root_vel[2]),
            "base_wx_rad_s": _float(root_vel[3]),
            "base_wy_rad_s": _float(root_vel[4]),
            "base_wz_rad_s": _float(root_vel[5]),
            "total_mass_kg": float(com.total_mass_kg),
            "com_x_m": _float(com.com_w[0]),
            "com_y_m": _float(com.com_w[1]),
            "com_z_m": _float(com.com_w[2]),
            "com_vx_m_s": _float(com_velocity[0]),
            "com_vy_m_s": _float(com_velocity[1]),
            "com_vz_m_s": _float(com_velocity[2]),
            "com_source": com.source,
            "com_warning": "; ".join(com.warnings),
            "com_projection_x_m": _component(stability.com_projection_w, 0),
            "com_projection_y_m": _component(stability.com_projection_w, 1),
            "com_projection_z_m": _component(stability.com_projection_w, 2),
            "static_stability_margin_m": float(stability.margin_m),
            "dynamic_stability_margin_m": float(dynamic_stability.margin_m),
            "stability_state": stability.state,
            "dynamic_stability_state": dynamic_stability.state,
            "support_area_m2": float(stability.area_m2),
            "support_perimeter_m": float(stability.perimeter_m),
            "active_contact_count": int(stability.active_contact_count),
            "support_polygon_w": stability.support_vertices_w,
            "support_degenerate": bool(stability.degenerate_support),
            "contact_geometry_source": contacts.contact_geometry_source,
            "max_contact_force_n": float(contacts.max_contact_force_n),
            "max_friction_utilization": float(contacts.max_friction_utilization),
            "max_wheel_slip_ratio": float(contacts.max_wheel_slip_ratio),
            "equilibrium_status": str((equilibrium or {}).get("status", "")),
            "equilibrium_stability_margin_m": _float((equilibrium or {}).get("equilibrium_stability_margin_m")),
            "equilibrium_area_m2": _float((equilibrium or {}).get("area_m2")),
            "equilibrium_solver_time_ms": _float((equilibrium or {}).get("solver_time_ms")),
            "equilibrium_vertices_w": (equilibrium or {}).get("vertices_w", []),
            "tracking_rmse_rad": float(joint_result.tracking_rmse_rad),
            "max_abs_tracking_error_rad": float(joint_result.max_abs_tracking_error_rad),
            "max_torque_utilization": float(joint_result.max_torque_utilization),
            "max_torque_joint": joint_result.max_torque_joint,
            "total_positive_work_j": float(joint_result.total_positive_work_j),
            "command_target_source": command_source,
            "warnings": "; ".join(com.warnings + contacts.warnings + stability.warnings + list((equilibrium or {}).get("warnings", []) or [])),
        }
        row.update(replay)
        self.sample_index += 1
        return row, body_rows, joint_result, contacts, stability, equilibrium

    def _body_rows(
        self,
        body_names: list[str],
        body_com_pos_w: Any,
        masses: Any,
        com: WholeBodyComResult,
        sim_time: float,
    ) -> list[dict[str, Any]]:
        positions = np.asarray(to_numpy(body_com_pos_w), dtype=float)
        if positions.ndim == 3:
            positions = positions[0]
        mass_arr = np.asarray(to_numpy(masses), dtype=float).reshape(-1)
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(body_names[: positions.shape[0] if positions.ndim == 2 else 0]):
            contribution = com.contribution[index] if index < len(com.contribution) else np.full(3, np.nan)
            mass = _float(mass_arr[index]) if index < mass_arr.size else float("nan")
            rows.append(
                {
                    "time_s": float(sim_time),
                    "body_index": int(index),
                    "body_name": name,
                    "mass_kg": mass,
                    "body_com_x_m": _float(positions[index, 0]),
                    "body_com_y_m": _float(positions[index, 1]),
                    "body_com_z_m": _float(positions[index, 2]),
                    "contribution_x_m": _float(contribution[0]),
                    "contribution_y_m": _float(contribution[1]),
                    "contribution_z_m": _float(contribution[2]),
                    "mass_fraction": mass / com.total_mass_kg if math.isfinite(mass) and math.isfinite(com.total_mass_kg) and com.total_mass_kg > 0.0 else float("nan"),
                    "valid_for_total_com": bool(index < len(com.valid_mask) and com.valid_mask[index]),
                }
            )
        return rows

    def _com_velocity(
        self,
        data: Any,
        env_id: int,
        masses: Any,
        com: WholeBodyComResult,
        sim_time: float,
        dt_s: float,
    ) -> np.ndarray:
        state = _env_matrix(getattr(data, "body_com_state_w", None), env_id)
        if state.ndim == 2 and state.shape[-1] >= 10:
            velocities = np.asarray(state[:, 7:10], dtype=float)
            mass_arr = np.asarray(to_numpy(masses), dtype=float).reshape(-1)
            count = min(len(velocities), mass_arr.size)
            mass_full = np.full(len(velocities), np.nan, dtype=float)
            if count:
                mass_full[:count] = mass_arr[:count]
            valid = np.zeros(len(velocities), dtype=bool)
            if count:
                valid[:count] = np.isfinite(velocities[:count]).all(axis=1) & np.isfinite(mass_full[:count]) & (mass_full[:count] > 0.0)
            if np.any(valid):
                return np.sum(velocities[valid] * mass_full[valid, None], axis=0) / max(float(np.sum(mass_full[valid])), 1.0e-12)
        if self.last_com_w is None or self.last_com_time_s is None:
            velocity = np.zeros(3, dtype=float)
        else:
            denom = max(float(sim_time) - float(self.last_com_time_s), max(float(dt_s), 1.0e-12))
            velocity = (np.asarray(com.com_w, dtype=float) - self.last_com_w) / denom
        self.last_com_w = np.asarray(com.com_w, dtype=float).copy()
        self.last_com_time_s = float(sim_time)
        return velocity

    def _equilibrium(
        self,
        sim_time: float,
        contacts: ContactMetricsResult,
        com: WholeBodyComResult,
        stability: StabilityResult,
    ) -> dict[str, Any] | None:
        if not bool(self.config.stability.equilibrium_enabled):
            return None
        interval = 1.0 / max(0.01, float(self.config.stability.equilibrium_hz))
        if self.last_equilibrium is not None and sim_time - self.last_equilibrium_time_s < interval:
            return self.last_equilibrium
        plane = gravity_aligned_support_plane(contacts.support_points_w, contacts.normal_forces_n)
        result = compute_equilibrium_region(
            contacts.support_points_w,
            contacts.support_normals_w,
            contacts.friction_coefficients,
            mass_kg=float(com.total_mass_kg),
            com_w=com.com_w,
            support_plane=plane,
            friction_pyramid_sides=int(self.config.stability.friction_pyramid_sides),
            direction_samples=int(self.config.stability.direction_samples),
            solver_tolerance=float(self.config.stability.solver_tolerance),
        )
        if stability.degenerate_support and result.get("status") == "success":
            result.setdefault("warnings", []).append("geometric support polygon is degenerate")
        self.last_equilibrium = result
        self.last_equilibrium_time_s = float(sim_time)
        return result

    def _record_warning_events(
        self,
        row: dict[str, Any],
        joint_result: JointMetricsResult,
        contacts: ContactMetricsResult,
        stability: StabilityResult,
        equilibrium: dict[str, Any] | None,
    ) -> None:
        if self.event_recorder is None:
            return
        time_s = float(row.get("time_s", 0.0) or 0.0)
        self.event_recorder.record_state(
            time_s,
            "stability_margin_low",
            str(stability.state) in {"critical", "outside", "airborne"},
            key="static",
            enter_severity="critical" if str(stability.state) in {"outside", "airborne"} else "warning",
            enter_message=f"Static stability state is {stability.state}; margin={stability.margin_m:.4g}m",
            exit_message="Static stability margin recovered.",
            value=stability.margin_m,
            threshold=0.0,
            step_sequence=self.replay_context.sequence_name,
            phase=self.replay_context.phase,
        )
        roll = abs(_float(row.get("base_roll_rad")))
        pitch = abs(_float(row.get("base_pitch_rad")))
        self.event_recorder.record_state(
            time_s,
            "roll_limit_warning",
            math.isfinite(roll) and roll >= float(self.config.warnings.roll_rad),
            key="base_roll",
            enter_message=f"Base roll exceeded {self.config.warnings.roll_rad:g} rad",
            exit_message="Base roll recovered.",
            value=roll,
            threshold=float(self.config.warnings.roll_rad),
        )
        self.event_recorder.record_state(
            time_s,
            "pitch_limit_warning",
            math.isfinite(pitch) and pitch >= float(self.config.warnings.pitch_rad),
            key="base_pitch",
            enter_message=f"Base pitch exceeded {self.config.warnings.pitch_rad:g} rad",
            exit_message="Base pitch recovered.",
            value=pitch,
            threshold=float(self.config.warnings.pitch_rad),
        )
        for joint in joint_result.rows:
            name = str(joint.get("joint_name", ""))
            torque_util = _float(joint.get("torque_utilization"))
            self.event_recorder.record_state(
                time_s,
                "torque_limit_warning",
                bool(joint.get("torque_limit_warning", False)),
                key=name,
                joint=name,
                enter_message=f"{name} torque utilization exceeded threshold",
                exit_message=f"{name} torque utilization recovered.",
                value=torque_util,
                threshold=float(self.config.warnings.torque_utilization),
            )
            self.event_recorder.record_state(
                time_s,
                "joint_position_limit_warning",
                bool(joint.get("position_limit_warning", False)),
                key=name,
                joint=name,
                enter_message=f"{name} is close to position limit",
                exit_message=f"{name} moved away from position limit.",
                value=joint.get("normalized_position_limit_margin"),
                threshold=0.1,
            )
        for contact in contacts.contacts:
            body = str(contact.get("body_name", ""))
            force = _float(contact.get("total_force_n"))
            self.event_recorder.record_state(
                time_s,
                "impact_force_warning",
                math.isfinite(force) and force >= float(self.config.warnings.impact_force_n),
                key=body,
                body=body,
                enter_message=f"{body} contact force exceeded impact threshold",
                exit_message=f"{body} contact force recovered.",
                value=force,
                threshold=float(self.config.warnings.impact_force_n),
            )
            slip = _float(contact.get("slip_ratio"))
            self.event_recorder.record_state(
                time_s,
                "wheel_slip_warning",
                bool(contact.get("slip_warning", False)),
                key=body,
                body=body,
                enter_message=f"{body} slip ratio exceeded threshold",
                exit_message=f"{body} slip ratio recovered.",
                value=slip,
                threshold=float(self.config.warnings.wheel_slip_ratio),
            )
        if equilibrium:
            eq_margin = _float(equilibrium.get("equilibrium_stability_margin_m"))
            self.event_recorder.record_state(
                time_s,
                "equilibrium_margin_low",
                math.isfinite(eq_margin) and eq_margin < 0.0,
                key="equilibrium",
                enter_severity="critical",
                enter_message=f"Friction-aware equilibrium region excludes current COM; margin={eq_margin:.4g}m",
                exit_message="Friction-aware equilibrium margin recovered.",
                value=eq_margin,
                threshold=0.0,
            )

    def _metadata(self, adapter: Any | None) -> dict[str, Any]:
        scene_config = getattr(self.scene_handle, "config", None)
        return {
            "schema_version": "height_replay_telemetry_run.v1",
            "run_label": self.sequence_label,
            "source": self.source,
            "created_wall_time": self.started_wall,
            "obstacle_height_cm": self.obstacle_height_cm,
            "obstacle_height_m": self.obstacle_height_m,
            "physics_dt": float(getattr(scene_config, "physics_dt", getattr(self.args, "physics_dt", 0.0)) or 0.0),
            "render_interval": int(getattr(scene_config, "render_interval", getattr(self.args, "render_interval", 0)) or 0),
            "robot_usd": str(getattr(scene_config, "robot_usd", getattr(self.args, "robot_usd", ""))),
            "robot_body_count": len(getattr(getattr(adapter, "robot", None), "body_names", []) or []),
            "robot_joint_count": len(getattr(getattr(adapter, "robot", None), "joint_names", []) or []),
            "runtime": python_runtime_metadata(),
        }


def create_telemetry_collector(
    args: Any | None = None,
    *,
    scene_handle: Any | None = None,
    config: RuntimeTelemetryConfig | None = None,
) -> TelemetryCollector | None:
    cfg = config or getattr(args, "telemetry_runtime_config", None) or load_telemetry_config(args)
    if not bool(cfg.telemetry.enabled):
        return None
    return TelemetryCollector(cfg, args=args, scene_handle=scene_handle)


def _body_com_positions_w(data: Any, env_id: int) -> tuple[np.ndarray, str]:
    state = _env_matrix(getattr(data, "body_com_state_w", None), env_id)
    if state.ndim == 2 and state.shape[-1] >= 3:
        return np.asarray(state[:, :3], dtype=float), "isaaclab.ArticulationData.body_com_state_w"
    link_state = _env_matrix(getattr(data, "body_link_state_w", None), env_id)
    if link_state.ndim != 2 or link_state.shape[-1] < 7:
        link_state = _env_matrix(getattr(data, "body_state_w", None), env_id)
    local_com = _env_matrix(getattr(data, "body_com_pose_b", None), env_id)
    if link_state.ndim == 2 and link_state.shape[-1] >= 7 and local_com.ndim == 2 and local_com.shape[-1] >= 3:
        return (
            body_com_positions_from_link_pose(link_state[:, :3], link_state[:, 3:7], local_com[:, :3]),
            "link_state_w + body_com_pose_b",
        )
    if link_state.ndim == 2 and link_state.shape[-1] >= 3:
        return np.asarray(link_state[:, :3], dtype=float), "link_position_fallback"
    return np.empty((0, 3), dtype=float), "unavailable"


def _body_masses(data: Any, env_id: int, count: int) -> np.ndarray:
    masses = to_numpy(getattr(data, "default_mass", None))
    if masses.ndim >= 2:
        masses = masses[min(max(0, env_id), masses.shape[0] - 1)]
    masses = masses.reshape(-1) if masses.size else masses
    out = np.full(count, np.nan, dtype=float)
    if masses.size:
        out[: min(count, masses.size)] = masses[: min(count, masses.size)]
    return out


def _env_row(value: Any, env_id: int, count: int, *, fill: float) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim >= 2:
        arr = arr[min(max(0, env_id), arr.shape[0] - 1)]
    arr = arr.reshape(-1) if arr.size else arr
    out = np.full(count, fill, dtype=float)
    if arr.size:
        out[: min(count, arr.size)] = arr[: min(count, arr.size)]
    return out


def _env_matrix(value: Any, env_id: int) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim >= 3:
        arr = arr[min(max(0, env_id), arr.shape[0] - 1)]
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return np.asarray(arr, dtype=float)


def _first_available(owner: Any, *names: str) -> Any:
    for name in names:
        value = getattr(owner, name, None)
        if to_numpy(value).size:
            return value
    return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _component(values: Any, index: int) -> float:
    try:
        return float(values[index])
    except Exception:
        return float("nan")
