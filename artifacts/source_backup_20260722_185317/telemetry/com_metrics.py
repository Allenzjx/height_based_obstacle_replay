"""Center-of-mass and frame transform helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class WholeBodyComResult:
    total_mass_kg: float
    com_w: np.ndarray
    contribution: np.ndarray
    valid_mask: np.ndarray
    warnings: list[str]
    source: str


def to_numpy(value: Any, *, dtype: Any = float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    try:
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    except Exception:
        try:
            return value.cpu().numpy().astype(dtype, copy=False)
        except Exception:
            return np.asarray(value, dtype=dtype)


def compute_whole_body_com(
    body_com_pos_w: Any,
    masses_kg: Any,
    *,
    source: str = "isaaclab.ArticulationData.body_com_state_w",
) -> WholeBodyComResult:
    positions = np.asarray(to_numpy(body_com_pos_w), dtype=float)
    masses = np.asarray(to_numpy(masses_kg), dtype=float).reshape(-1)
    if positions.ndim == 3:
        positions = positions[0]
    if positions.ndim != 2 or positions.shape[-1] != 3:
        return WholeBodyComResult(float("nan"), np.full(3, np.nan), np.empty((0, 3)), np.zeros(0, dtype=bool), ["body COM positions unavailable"], source)
    warnings: list[str] = []
    if masses.size != positions.shape[0]:
        resized = np.full(positions.shape[0], np.nan, dtype=float)
        count = min(masses.size, positions.shape[0])
        if count:
            resized[:count] = masses[:count]
        masses = resized
        warnings.append("mass count does not match body count")
    finite_pos = np.isfinite(positions).all(axis=1)
    finite_mass = np.isfinite(masses)
    positive_mass = masses > 0.0
    valid = finite_pos & finite_mass & positive_mass
    if np.any(~finite_pos):
        warnings.append("one or more body COM positions are NaN/Inf")
    if np.any(finite_mass & ~positive_mass):
        warnings.append("one or more bodies have non-positive mass")
    if not np.any(valid):
        return WholeBodyComResult(float("nan"), np.full(3, np.nan), np.empty((0, 3)), valid, warnings + ["no valid positive masses"], source)
    total = float(np.sum(masses[valid]))
    contribution = np.zeros_like(positions, dtype=float)
    contribution[valid] = (masses[valid, None] * positions[valid]) / total
    com = np.sum(contribution[valid], axis=0)
    return WholeBodyComResult(total, com, contribution, valid, warnings, source)


def body_com_positions_from_link_pose(link_pos_w: Any, link_quat_wxyz: Any, local_com_pos_b: Any) -> np.ndarray:
    positions = np.asarray(to_numpy(link_pos_w), dtype=float)
    quats = np.asarray(to_numpy(link_quat_wxyz), dtype=float)
    offsets = np.asarray(to_numpy(local_com_pos_b), dtype=float)
    if positions.ndim == 3:
        positions = positions[0]
    if quats.ndim == 3:
        quats = quats[0]
    if offsets.ndim == 3:
        offsets = offsets[0]
    result = np.full_like(positions, np.nan, dtype=float)
    for index in range(min(len(positions), len(quats), len(offsets))):
        result[index] = positions[index] + quat_rotate_wxyz(quats[index], offsets[index])
    return result


def quat_rotate_wxyz(quat: Any, vec: Any) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    v = np.asarray(vec, dtype=float)
    w, x, y, z = q
    qvec = np.asarray([x, y, z], dtype=float)
    t = 2.0 * np.cross(qvec, v)
    return v + w * t + np.cross(qvec, t)


def quat_inverse_rotate_wxyz(quat: Any, vec: Any) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    return quat_rotate_wxyz(np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=float), vec)


def normalize_quat_wxyz(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=float).reshape(-1)[:4]
    if q.size != 4 or not np.isfinite(q).all():
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_wxyz_to_rpy(quat: Any) -> tuple[float, float, float]:
    w, x, y, z = normalize_quat_wxyz(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def first_order_low_pass(previous: Any, raw: Any, *, dt: float, cutoff_hz: float) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=float)
    if previous is None:
        return raw_arr
    prev_arr = np.asarray(previous, dtype=float)
    if dt <= 0.0 or cutoff_hz <= 0.0:
        return raw_arr
    rc = 1.0 / (2.0 * math.pi * float(cutoff_hz))
    alpha = float(dt) / (rc + float(dt))
    return prev_arr + alpha * (raw_arr - prev_arr)
