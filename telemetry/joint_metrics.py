"""Joint telemetry calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .com_metrics import first_order_low_pass, to_numpy


@dataclass
class JointMetricState:
    previous_time_s: float | None = None
    previous_position: np.ndarray | None = None
    previous_velocity: np.ndarray | None = None
    filtered_acceleration: np.ndarray | None = None
    cumulative_positive_work_j: np.ndarray | None = None
    cumulative_absolute_energy_j: np.ndarray | None = None


@dataclass
class JointMetricsResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    tracking_rmse_rad: float = float("nan")
    max_abs_tracking_error_rad: float = float("nan")
    max_torque_utilization: float = float("nan")
    max_torque_joint: str = ""
    total_positive_work_j: float = 0.0
    warnings: list[str] = field(default_factory=list)


def compute_joint_metrics(
    *,
    joint_names: list[str],
    joint_pos: Any,
    joint_vel: Any,
    commanded_pos: Any | None,
    commanded_vel: Any | None,
    commanded_torque: Any | None,
    applied_torque: Any | None,
    position_limits: Any | None,
    velocity_limits: Any | None,
    effort_limits: Any | None,
    state: JointMetricState,
    time_s: float,
    dt_s: float,
    acceleration_filter_enabled: bool,
    acceleration_cutoff_hz: float,
    torque_warning_threshold: float,
) -> JointMetricsResult:
    names = [str(name) for name in joint_names]
    pos = _row_vector(joint_pos, len(names), fill=np.nan)
    vel = _row_vector(joint_vel, len(names), fill=np.nan)
    cmd_pos = _optional_row_vector(commanded_pos, len(names))
    cmd_vel = _optional_row_vector(commanded_vel, len(names))
    cmd_tau = _optional_row_vector(commanded_torque, len(names))
    tau = _optional_row_vector(applied_torque, len(names))
    pos_limits = _limit_matrix(position_limits, len(names))
    vel_limits = _limit_vector(velocity_limits, len(names))
    effort = _limit_vector(effort_limits, len(names))
    if state.cumulative_positive_work_j is None or len(state.cumulative_positive_work_j) != len(names):
        state.cumulative_positive_work_j = np.zeros(len(names), dtype=float)
    if state.cumulative_absolute_energy_j is None or len(state.cumulative_absolute_energy_j) != len(names):
        state.cumulative_absolute_energy_j = np.zeros(len(names), dtype=float)
    if state.previous_velocity is None:
        raw_acc = np.zeros(len(names), dtype=float)
    else:
        denom = max(float(dt_s), 1.0e-12)
        raw_acc = (vel - state.previous_velocity) / denom
    if acceleration_filter_enabled:
        filtered_acc = first_order_low_pass(
            state.filtered_acceleration,
            raw_acc,
            dt=max(float(dt_s), 0.0),
            cutoff_hz=float(acceleration_cutoff_hz),
        )
    else:
        filtered_acc = raw_acc
    power = tau * vel if tau is not None else np.full(len(names), np.nan, dtype=float)
    positive_power = np.where(np.isfinite(power) & (power > 0.0), power, 0.0)
    absolute_power = np.where(np.isfinite(power), np.abs(power), 0.0)
    state.cumulative_positive_work_j += positive_power * max(float(dt_s), 0.0)
    state.cumulative_absolute_energy_j += absolute_power * max(float(dt_s), 0.0)
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    max_util = float("nan")
    max_util_joint = ""
    warnings: list[str] = []
    for index, name in enumerate(names):
        lower = float(pos_limits[index, 0]) if np.isfinite(pos_limits[index, 0]) else float("nan")
        upper = float(pos_limits[index, 1]) if np.isfinite(pos_limits[index, 1]) else float("nan")
        position = float(pos[index])
        velocity = float(vel[index])
        cpos = float(cmd_pos[index]) if cmd_pos is not None and np.isfinite(cmd_pos[index]) else float("nan")
        cvel = float(cmd_vel[index]) if cmd_vel is not None and np.isfinite(cmd_vel[index]) else float("nan")
        ctau = float(cmd_tau[index]) if cmd_tau is not None and np.isfinite(cmd_tau[index]) else float("nan")
        applied = float(tau[index]) if tau is not None and np.isfinite(tau[index]) else float("nan")
        effort_limit = float(effort[index]) if np.isfinite(effort[index]) else float("nan")
        velocity_limit = float(vel_limits[index]) if np.isfinite(vel_limits[index]) else float("nan")
        position_error = position - cpos if math.isfinite(position) and math.isfinite(cpos) else float("nan")
        velocity_error = velocity - cvel if math.isfinite(velocity) and math.isfinite(cvel) else float("nan")
        torque_error = applied - ctau if math.isfinite(applied) and math.isfinite(ctau) else float("nan")
        if math.isfinite(position_error):
            errors.append(position_error)
        pos_margin = _position_limit_margin(position, lower, upper)
        velocity_margin = abs(velocity) / velocity_limit if math.isfinite(velocity_limit) and velocity_limit > 0.0 and math.isfinite(velocity) else float("nan")
        torque_util = abs(applied) / effort_limit if math.isfinite(effort_limit) and effort_limit > 0.0 and math.isfinite(applied) else float("nan")
        if math.isfinite(torque_util) and (not math.isfinite(max_util) or torque_util > max_util):
            max_util = torque_util
            max_util_joint = name
        row = {
            "time_s": float(time_s),
            "joint_name": name,
            "position_rad": position,
            "velocity_rad_s": velocity,
            "acceleration_rad_s2": float(filtered_acc[index]),
            "raw_acceleration_rad_s2": float(raw_acc[index]),
            "commanded_position_rad": cpos,
            "commanded_velocity_rad_s": cvel,
            "commanded_torque_nm": ctau,
            "applied_torque_nm": applied,
            "position_tracking_error_rad": position_error,
            "velocity_tracking_error_rad_s": velocity_error,
            "torque_tracking_error_nm": torque_error,
            "position_limit_lower_rad": lower,
            "position_limit_upper_rad": upper,
            "normalized_position_limit_margin": pos_margin,
            "velocity_limit_rad_s": velocity_limit,
            "effort_limit_nm": effort_limit,
            "torque_utilization": torque_util,
            "mechanical_power_w": float(power[index]),
            "positive_power_w": float(positive_power[index]),
            "negative_power_w": float(power[index]) if math.isfinite(float(power[index])) and float(power[index]) < 0.0 else 0.0,
            "cumulative_positive_work_j": float(state.cumulative_positive_work_j[index]),
            "cumulative_absolute_energy_j": float(state.cumulative_absolute_energy_j[index]),
            "actuator_saturation": bool(math.isfinite(torque_util) and torque_util >= 1.0),
            "position_limit_warning": bool(math.isfinite(pos_margin) and pos_margin <= 0.1),
            "velocity_limit_warning": bool(math.isfinite(velocity_margin) and velocity_margin >= 0.9),
            "torque_limit_warning": bool(math.isfinite(torque_util) and torque_util >= float(torque_warning_threshold)),
            "torque_source": "isaaclab.ArticulationData.applied_torque" if tau is not None else "unavailable",
        }
        rows.append(row)
    state.previous_time_s = float(time_s)
    state.previous_position = pos.copy()
    state.previous_velocity = vel.copy()
    state.filtered_acceleration = filtered_acc.copy()
    rmse = math.sqrt(float(np.mean(np.square(errors)))) if errors else float("nan")
    max_abs = float(np.max(np.abs(errors))) if errors else float("nan")
    return JointMetricsResult(
        rows=rows,
        tracking_rmse_rad=rmse,
        max_abs_tracking_error_rad=max_abs,
        max_torque_utilization=float(max_util),
        max_torque_joint=max_util_joint,
        total_positive_work_j=float(np.sum(state.cumulative_positive_work_j)),
        warnings=warnings,
    )


def _row_vector(value: Any, count: int, *, fill: float) -> np.ndarray:
    arr = np.asarray(to_numpy(value), dtype=float)
    if arr.ndim >= 2:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        arr = np.nanmax(np.abs(arr[:, :2]), axis=1)
    arr = arr.reshape(-1)
    out = np.full(count, fill, dtype=float)
    out[: min(count, arr.size)] = arr[: min(count, arr.size)]
    return out


def _optional_row_vector(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = _row_vector(value, count, fill=np.nan)
    return arr


def _limit_matrix(value: Any, count: int) -> np.ndarray:
    arr = np.asarray(to_numpy(value), dtype=float)
    if arr.ndim >= 3:
        arr = arr[0]
    out = np.full((count, 2), np.nan, dtype=float)
    if arr.ndim == 2 and arr.shape[-1] == 2:
        rows = min(count, arr.shape[0])
        out[:rows, :] = arr[:rows, :2]
    return out


def _limit_vector(value: Any, count: int) -> np.ndarray:
    if value is None:
        return np.full(count, np.nan, dtype=float)
    arr = np.asarray(to_numpy(value), dtype=float)
    if arr.ndim >= 2:
        arr = arr[0]
    arr = arr.reshape(-1)
    out = np.full(count, np.nan, dtype=float)
    out[: min(count, arr.size)] = arr[: min(count, arr.size)]
    return out


def _position_limit_margin(position: float, lower: float, upper: float) -> float:
    if not all(math.isfinite(value) for value in (position, lower, upper)) or upper <= lower:
        return float("nan")
    return min(position - lower, upper - position) / max(upper - lower, 1.0e-12)
