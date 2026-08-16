"""Strict whole-body centroidal and physical support evidence for FSM50.

The numerical API in this module is intentionally independent of Isaac Sim.
Array-like inputs may be Python sequences, NumPy arrays, or detached Torch
tensors.  The two ``measure_isaac_*`` helpers are lazy adapters: they inspect
an already-running project's adapter/robot/contact bank without importing or
starting Isaac.

No function in this module substitutes a root/base position for missing body
center-of-mass data.  Missing, stale, ambiguous, or non-finite evidence is
reported as unavailable or ``NOT_PROVEN``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
DIAGONALS: tuple[tuple[str, str], ...] = (("FL", "RR"), ("FR", "RL"))
LEG_TO_WHEEL_BODY: Mapping[str, str] = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}
SUPPORT_SURFACES = frozenset({"GROUND", "OBSTACLE_TOP"})
ALL_SURFACES = frozenset({"GROUND", "OBSTACLE_TOP", "FRONT_FACE", "AIR", "UNKNOWN"})
CONTACT_MOMENT_MODELS = frozenset(
    {"MEASURED", "POINT_CONTACT_ZERO_CONSERVATIVE"}
)
RAW_CONTACT_FORCE_AGGREGATE_ATOL_N = 1.0e-5
RAW_CONTACT_POINT_AGGREGATE_ATOL_M = 1.0e-6
RAW_CONTACT_NORMAL_UNIT_TOLERANCE = 1.0e-5
COM_FIELD_NAMES = frozenset({"body_names", "masses", "positions", "velocities", "accelerations"})
ANGULAR_MOMENTUM_FIELD_NAMES = frozenset(
    {
        "body_names",
        "masses",
        "positions",
        "linear_accelerations",
        "angular_velocities",
        "angular_accelerations",
        "com_principal_orientations",
        "inertias",
    }
)
CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA = "fsm50.centroidal_support_evidence.v1"


class EvidenceStatus(str, Enum):
    PROVEN = "PROVEN"
    NOT_PROVEN = "NOT_PROVEN"
    UNAVAILABLE = "UNAVAILABLE"


class SupportModel(str, Enum):
    NONE = "NONE"
    STRICT_COPLANAR_CONVEX_HULL = "STRICT_COPLANAR_CONVEX_HULL"
    DIAGONAL_LINE_SEGMENT = "DIAGONAL_LINE_SEGMENT"
    MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED = "MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED"


class TransferMethod(str, Enum):
    IMPULSE_BASED_COM_TRANSFER = "IMPULSE_BASED_COM_TRANSFER"
    SUPPORT_ANGLE_COM_TRANSFER = "SUPPORT_ANGLE_COM_TRANSFER"
    HYBRID_COM_TRANSFER = "HYBRID_COM_TRANSFER"
    NOT_YET_PROVEN = "NOT_YET_PROVEN"


Vec3 = tuple[float, float, float]


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{label} must be finite{suffix}")
    return result


def _array(value: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is unavailable")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        result = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric array-like") from exc
    if result.shape != shape:
        raise ValueError(f"{label} shape {result.shape} does not equal {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _vec3(value: Any, label: str) -> np.ndarray:
    return _array(value, label, (3,))


def _unit_vec3(value: Any, label: str, *, tolerance: float = 1.0e-6) -> np.ndarray:
    result = _vec3(value, label)
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12 or abs(norm - 1.0) > tolerance:
        raise ValueError(f"{label} must be a unit vector")
    return result


def _strict_time(
    physics_tick: Any,
    sim_time_s: Any,
    physics_dt_s: Any,
) -> tuple[int, float, float]:
    tick = _exact_int(physics_tick, "physics_tick")
    sim_time = _finite_float(sim_time_s, "sim_time_s", minimum=0.0)
    dt = _finite_float(physics_dt_s, "physics_dt_s", minimum=0.0)
    if dt <= 0.0:
        raise ValueError("physics_dt_s must be positive")
    if not math.isclose(sim_time, tick * dt, rel_tol=0.0, abs_tol=max(1.0e-9, dt * 1.0e-6)):
        raise ValueError("sim_time_s does not identify physics_tick at physics_dt_s")
    return tick, sim_time, dt


def _valid_body_names(
    body_names: Sequence[str], expected_body_names: Sequence[str] | None
) -> tuple[str, ...]:
    if isinstance(body_names, (str, bytes)):
        raise ValueError("body_names must be an ordered sequence")
    names = tuple(body_names)
    if not names:
        raise ValueError("body_names is empty")
    if any(type(name) is not str or not name.strip() or name != name.strip() for name in names):
        raise ValueError("body_names contains an invalid name")
    if len(set(names)) != len(names):
        raise ValueError("body_names contains duplicates")
    if expected_body_names is not None and names != tuple(expected_body_names):
        raise ValueError("body_names/order differs from the expected articulation order")
    return names


@dataclass(frozen=True)
class WholeBodyCOMMeasurement:
    com_measurement_available: bool
    acceleration_available: bool
    physics_tick: int | None
    sim_time_s: float | None
    physics_dt_s: float | None
    body_names: tuple[str, ...]
    body_masses_kg: tuple[float, ...]
    total_mass_kg: float | None
    position_w_m: Vec3 | None
    velocity_w_m_s: Vec3 | None
    acceleration_w_m_s2: Vec3 | None
    source: str
    errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.com_measurement_available


def unavailable_whole_body_com(*errors: str, source: str) -> WholeBodyCOMMeasurement:
    reasons = tuple(str(item) for item in errors if str(item)) or ("whole-body COM evidence unavailable",)
    return WholeBodyCOMMeasurement(
        com_measurement_available=False,
        acceleration_available=False,
        physics_tick=None,
        sim_time_s=None,
        physics_dt_s=None,
        body_names=(),
        body_masses_kg=(),
        total_mass_kg=None,
        position_w_m=None,
        velocity_w_m_s=None,
        acceleration_w_m_s2=None,
        source=source,
        errors=reasons,
    )


def measure_whole_body_com(
    *,
    body_names: Sequence[str],
    body_masses_kg: Any,
    body_com_positions_w_m: Any,
    body_com_velocities_w_m_s: Any,
    body_com_accelerations_w_m_s2: Any,
    physics_tick: int,
    sim_time_s: float,
    physics_dt_s: float,
    field_physics_ticks: Mapping[str, int],
    expected_body_names: Sequence[str] | None = None,
    source: str = "array.body_com_state_w",
) -> WholeBodyCOMMeasurement:
    """Compute mass-weighted whole-articulation COM from one exact tick.

    Acceleration is required by this strict API.  A caller lacking it receives
    an unavailable result; the function never differentiates a base proxy.
    """

    try:
        tick, time_s, dt = _strict_time(physics_tick, sim_time_s, physics_dt_s)
        names = _valid_body_names(body_names, expected_body_names)
        if not isinstance(field_physics_ticks, Mapping) or set(field_physics_ticks) != COM_FIELD_NAMES:
            raise ValueError("field_physics_ticks keys are not exact")
        for field, field_tick in field_physics_ticks.items():
            if _exact_int(field_tick, f"field_physics_ticks.{field}") != tick:
                raise ValueError(f"{field} does not come from physics_tick")
        count = len(names)
        masses = _array(body_masses_kg, "body_masses_kg", (count,))
        if np.any(masses <= 0.0):
            raise ValueError("every body mass must be positive")
        total_mass = float(np.sum(masses, dtype=np.float64))
        if not math.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("total body mass must be finite and positive")
        positions = _array(body_com_positions_w_m, "body_com_positions_w_m", (count, 3))
        velocities = _array(body_com_velocities_w_m_s, "body_com_velocities_w_m_s", (count, 3))
        accelerations = _array(
            body_com_accelerations_w_m_s2,
            "body_com_accelerations_w_m_s2",
            (count, 3),
        )
        weights = masses[:, None]
        position = np.sum(weights * positions, axis=0, dtype=np.float64) / total_mass
        velocity = np.sum(weights * velocities, axis=0, dtype=np.float64) / total_mass
        acceleration = np.sum(weights * accelerations, axis=0, dtype=np.float64) / total_mass
        if not all(np.isfinite(value).all() for value in (position, velocity, acceleration)):
            raise ValueError("mass-weighted COM output is non-finite")
    except (TypeError, ValueError, OverflowError) as exc:
        return unavailable_whole_body_com(str(exc), source=source)
    return WholeBodyCOMMeasurement(
        com_measurement_available=True,
        acceleration_available=True,
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        body_names=names,
        body_masses_kg=tuple(float(value) for value in masses),
        total_mass_kg=total_mass,
        position_w_m=tuple(float(value) for value in position),
        velocity_w_m_s=tuple(float(value) for value in velocity),
        acceleration_w_m_s2=tuple(float(value) for value in acceleration),
        source=source,
    )


def _slice_env(value: Any, env_id: int, trailing_shape: tuple[int, ...], label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is unavailable")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=float)
    if result.ndim != len(trailing_shape) + 1 or tuple(result.shape[1:]) != trailing_shape:
        raise ValueError(f"{label} has invalid batched shape {result.shape}")
    if not 0 <= env_id < result.shape[0]:
        raise ValueError(f"env_id={env_id} is out of range for {label}")
    row = np.asarray(result[env_id], dtype=float)
    if not np.isfinite(row).all():
        raise ValueError(f"{label}[{env_id}] contains non-finite values")
    return row


def _tensor_float_array(value: Any, label: str) -> np.ndarray:
    """Detach a tensor-like value without accepting Boolean payloads.

    Raw PhysX contact buffers may contain unspecified values outside the
    count/start slices, so finiteness is checked only on referenced rows.
    """

    if value is None:
        raise ValueError(f"{label} is unavailable")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        untyped = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric array-like") from exc
    if np.issubdtype(untyped.dtype, np.bool_):
        raise ValueError(f"{label} must not contain Boolean values")
    try:
        return np.asarray(untyped, dtype=float)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric array-like") from exc


def _raw_index_array(value: Any, label: str, shape: tuple[int, ...]) -> np.ndarray:
    result = _tensor_float_array(value, label)
    if result.shape != shape:
        raise ValueError(f"{label} shape {result.shape} does not equal {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    if np.any(result < 0.0) or np.any(result != np.floor(result)):
        raise ValueError(f"{label} must contain non-negative exact integers")
    if np.any(result > float(2**53 - 1)):
        raise ValueError(f"{label} exceeds the exact floating-point integer range")
    return result.astype(np.int64)


def _validate_raw_buffer_layout(
    counts: np.ndarray,
    starts: np.ndarray,
    capacity: int,
    label: str,
) -> None:
    intervals: list[tuple[int, int, str]] = []
    total_count = 0
    for sensor_index in range(counts.shape[0]):
        for filter_index in range(counts.shape[1]):
            count = int(counts[sensor_index, filter_index])
            start = int(starts[sensor_index, filter_index])
            end = start + count
            pair = f"sensor={sensor_index},filter={filter_index}"
            if start > capacity or end > capacity:
                raise ValueError(f"{label} {pair} slice exceeds capacity={capacity}")
            total_count += count
            if count:
                intervals.append((start, end, pair))
    if total_count >= capacity:
        raise ValueError(
            f"{label} capacity is exhausted: used={total_count}, capacity={capacity}"
        )
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"{label} slices overlap: {previous[2]} and {current[2]}"
            )


def _used_raw_slices(
    values: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
    label: str,
) -> None:
    for sensor_index in range(counts.shape[0]):
        for filter_index in range(counts.shape[1]):
            count = int(counts[sensor_index, filter_index])
            if not count:
                continue
            start = int(starts[sensor_index, filter_index])
            if not np.isfinite(values[start : start + count]).all():
                raise ValueError(
                    f"{label} contains non-finite used rows for "
                    f"sensor={sensor_index},filter={filter_index}"
                )


def _measure_raw_contact_moment_at_reference(
    sensor: Any,
    *,
    leg: str,
    env_id: int,
    physics_dt_s: float,
    aggregate_sensor_count: int,
    aggregate_normal_forces_w_n: np.ndarray,
    aggregate_friction_forces_w_n: np.ndarray,
    aggregate_contact_points_w_m: np.ndarray,
    whole_body_com_position_w_m: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    ...,
]:
    """Measure every filter-pair wrench from strict raw PhysX buffer slices."""

    view = getattr(sensor, "contact_physx_view", None)
    if view is None:
        raise ValueError(f"{leg} contact_physx_view is unavailable")
    checker = getattr(view, "check", None)
    if not callable(checker) or not bool(checker()):
        raise ValueError(f"{leg} contact_physx_view is invalid")
    sensor_count = _exact_int(
        getattr(view, "sensor_count", None),
        f"{leg} contact_physx_view.sensor_count",
        minimum=1,
    )
    filter_count = _exact_int(
        getattr(view, "filter_count", None),
        f"{leg} contact_physx_view.filter_count",
        minimum=1,
    )
    capacity = _exact_int(
        getattr(view, "max_contact_data_count", None),
        f"{leg} contact_physx_view.max_contact_data_count",
        minimum=1,
    )
    if sensor_count != aggregate_sensor_count:
        raise ValueError(
            f"{leg} raw sensor_count does not match aggregate leading dimension"
        )
    if filter_count != 2:
        raise ValueError(f"{leg} raw filter_count is not exact ground/obstacle")
    if not 0 <= env_id < sensor_count:
        raise ValueError(f"{leg} raw contact pair index is out of range")
    for value, label in (
        (aggregate_normal_forces_w_n, "aggregate normal forces"),
        (aggregate_friction_forces_w_n, "aggregate friction forces"),
        (aggregate_contact_points_w_m, "aggregate contact points"),
    ):
        if value.shape != (filter_count, 3):
            raise ValueError(f"{leg} {label} shape is not exact")

    get_contact_data = getattr(view, "get_contact_data", None)
    get_friction_data = getattr(view, "get_friction_data", None)
    if not callable(get_contact_data) or not callable(get_friction_data):
        raise ValueError(f"{leg} raw contact/friction accessors are unavailable")
    contact_payload = get_contact_data(dt=physics_dt_s)
    friction_payload = get_friction_data(dt=physics_dt_s)
    if not isinstance(contact_payload, (tuple, list)) or len(contact_payload) != 6:
        raise ValueError(f"{leg} get_contact_data payload is malformed")
    if not isinstance(friction_payload, (tuple, list)) or len(friction_payload) != 4:
        raise ValueError(f"{leg} get_friction_data payload is malformed")

    force_magnitudes = _tensor_float_array(contact_payload[0], f"{leg} raw contact force magnitudes")
    contact_points = _tensor_float_array(contact_payload[1], f"{leg} raw contact points")
    contact_normals = _tensor_float_array(contact_payload[2], f"{leg} raw contact normals")
    separation_distances = _tensor_float_array(contact_payload[3], f"{leg} raw separation distances")
    expected_scalar_shape = (capacity, 1)
    expected_vec_shape = (capacity, 3)
    for value, expected, label in (
        (force_magnitudes, expected_scalar_shape, "contact force magnitudes"),
        (contact_points, expected_vec_shape, "contact points"),
        (contact_normals, expected_vec_shape, "contact normals"),
        (separation_distances, expected_scalar_shape, "separation distances"),
    ):
        if value.shape != expected:
            raise ValueError(
                f"{leg} raw {label} shape {value.shape} does not equal {expected}"
            )
    contact_counts = _raw_index_array(
        contact_payload[4],
        f"{leg} raw contact counts",
        (sensor_count, filter_count),
    )
    contact_starts = _raw_index_array(
        contact_payload[5],
        f"{leg} raw contact starts",
        (sensor_count, filter_count),
    )
    _validate_raw_buffer_layout(contact_counts, contact_starts, capacity, f"{leg} raw contact")
    for value, label in (
        (force_magnitudes, "raw contact force magnitudes"),
        (contact_points, "raw contact points"),
        (contact_normals, "raw contact normals"),
        (separation_distances, "raw separation distances"),
    ):
        _used_raw_slices(value, contact_counts, contact_starts, f"{leg} {label}")

    friction_forces = _tensor_float_array(friction_payload[0], f"{leg} raw friction forces")
    friction_points = _tensor_float_array(friction_payload[1], f"{leg} raw friction points")
    for value, label in (
        (friction_forces, "friction forces"),
        (friction_points, "friction points"),
    ):
        if value.shape != expected_vec_shape:
            raise ValueError(
                f"{leg} raw {label} shape {value.shape} does not equal {expected_vec_shape}"
            )
    friction_counts = _raw_index_array(
        friction_payload[2],
        f"{leg} raw friction counts",
        (sensor_count, filter_count),
    )
    friction_starts = _raw_index_array(
        friction_payload[3],
        f"{leg} raw friction starts",
        (sensor_count, filter_count),
    )
    _validate_raw_buffer_layout(friction_counts, friction_starts, capacity, f"{leg} raw friction")
    _used_raw_slices(friction_forces, friction_counts, friction_starts, f"{leg} raw friction forces")
    _used_raw_slices(friction_points, friction_counts, friction_starts, f"{leg} raw friction points")

    for sensor_index in range(sensor_count):
        for raw_filter_index in range(filter_count):
            count = int(contact_counts[sensor_index, raw_filter_index])
            if not count:
                continue
            start = int(contact_starts[sensor_index, raw_filter_index])
            magnitudes = force_magnitudes[start : start + count, 0]
            normals = contact_normals[start : start + count]
            if np.any(magnitudes < 0.0):
                raise ValueError(f"{leg} raw contact force magnitude is negative")
            normal_lengths = np.linalg.norm(normals, axis=1)
            if np.any(
                np.abs(normal_lengths - 1.0)
                > RAW_CONTACT_NORMAL_UNIT_TOLERANCE
            ):
                raise ValueError(f"{leg} raw contact normal is not unit length")

    pair_results: list[
        tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
    ] = []
    for pair_filter_index in range(filter_count):
        contact_count = int(contact_counts[env_id, pair_filter_index])
        contact_start = int(contact_starts[env_id, pair_filter_index])
        pair_points = contact_points[contact_start : contact_start + contact_count]
        pair_normal_forces = (
            force_magnitudes[contact_start : contact_start + contact_count]
            * contact_normals[contact_start : contact_start + contact_count]
        )
        raw_normal_force = np.sum(
            pair_normal_forces,
            axis=0,
            dtype=np.float64,
        )
        if not np.allclose(
            raw_normal_force,
            aggregate_normal_forces_w_n[pair_filter_index],
            rtol=0.0,
            atol=RAW_CONTACT_FORCE_AGGREGATE_ATOL_N,
        ):
            raise ValueError(
                f"{leg} raw/aggregate normal force mismatch for "
                f"filter={pair_filter_index}"
            )

        friction_count = int(friction_counts[env_id, pair_filter_index])
        friction_start = int(friction_starts[env_id, pair_filter_index])
        pair_friction_forces = friction_forces[
            friction_start : friction_start + friction_count
        ]
        pair_friction_points = friction_points[
            friction_start : friction_start + friction_count
        ]
        raw_friction_force = np.sum(
            pair_friction_forces,
            axis=0,
            dtype=np.float64,
        )
        if not np.allclose(
            raw_friction_force,
            aggregate_friction_forces_w_n[pair_filter_index],
            rtol=0.0,
            atol=RAW_CONTACT_FORCE_AGGREGATE_ATOL_N,
        ):
            raise ValueError(
                f"{leg} raw/aggregate friction force mismatch for "
                f"filter={pair_filter_index}"
            )

        if not contact_count:
            if friction_count:
                raise ValueError(
                    f"{leg} raw friction rows have no contact-point reference "
                    f"for filter={pair_filter_index}"
                )
            pair_results.append(
                (raw_normal_force, raw_friction_force, None, None)
            )
            continue
        raw_mean_point = np.mean(pair_points, axis=0, dtype=np.float64)
        if not np.isfinite(
            aggregate_contact_points_w_m[pair_filter_index]
        ).all() or not np.allclose(
            raw_mean_point,
            aggregate_contact_points_w_m[pair_filter_index],
            rtol=0.0,
            atol=RAW_CONTACT_POINT_AGGREGATE_ATOL_M,
        ):
            raise ValueError(
                f"{leg} raw/aggregate contact point mismatch for "
                f"filter={pair_filter_index}"
            )

        raw_moment_about_com = np.sum(
            np.cross(
                pair_points - whole_body_com_position_w_m,
                pair_normal_forces,
            ),
            axis=0,
            dtype=np.float64,
        ) + np.sum(
            np.cross(
                pair_friction_points - whole_body_com_position_w_m,
                pair_friction_forces,
            ),
            axis=0,
            dtype=np.float64,
        )
        raw_force = raw_normal_force + raw_friction_force
        moment_at_raw_mean_point = raw_moment_about_com - np.cross(
            raw_mean_point - whole_body_com_position_w_m,
            raw_force,
        )
        if not np.isfinite(moment_at_raw_mean_point).all():
            raise ValueError(f"{leg} measured raw contact moment is non-finite")
        pair_results.append(
            (
                raw_normal_force,
                raw_friction_force,
                raw_mean_point,
                moment_at_raw_mean_point,
            )
        )

    return tuple(pair_results)


def _buffer_timestamp(data: Any, buffer_name: str) -> float:
    buffer = getattr(data, buffer_name, None)
    return _finite_float(getattr(buffer, "timestamp", None), f"{buffer_name}.timestamp", minimum=0.0)


def measure_isaac_whole_body_com(
    adapter: Any,
    *,
    env_id: int = 0,
    expected_body_names: Sequence[str] | None = None,
) -> WholeBodyCOMMeasurement:
    """Read strict current-tick COM evidence from an existing SimRobotAdapter.

    This helper deliberately has no root-position fallback and performs no
    simulator step or update.
    """

    source = "isaaclab.ArticulationData.body_com_pos/lin_vel/lin_acc_w"
    try:
        env_index = _exact_int(env_id, "env_id")
        robot = getattr(adapter, "robot")
        data = getattr(robot, "data")
        tick, time_s, dt = _strict_time(
            getattr(adapter, "sim_steps"),
            getattr(adapter, "sim_time"),
            adapter.sim.get_physics_dt(),
        )
        names = _valid_body_names(getattr(robot, "body_names"), expected_body_names)
        if tuple(getattr(data, "body_names")) != names:
            raise ValueError("ArticulationData.body_names/order differs from Articulation.body_names")
        physx_names = tuple(robot.root_physx_view.shared_metatype.link_names)
        if physx_names != names:
            raise ValueError("PhysX link_names/order differs from Articulation.body_names")
        if _exact_int(getattr(robot, "num_bodies"), "robot.num_bodies", minimum=1) != len(names):
            raise ValueError("robot.num_bodies differs from body_names count")
        data_timestamp = _finite_float(getattr(data, "_sim_timestamp", None), "ArticulationData._sim_timestamp")
        if not math.isclose(data_timestamp, time_s, rel_tol=0.0, abs_tol=max(1.0e-9, dt * 1.0e-6)):
            raise ValueError("ArticulationData is not at the adapter current physics tick")

        # Reading the properties refreshes their lazy buffers at _sim_timestamp.
        positions_raw = data.body_com_pos_w
        velocities_raw = data.body_com_lin_vel_w
        accelerations_raw = data.body_com_lin_acc_w
        for buffer_name in ("_body_com_pose_w", "_body_com_vel_w", "_body_com_acc_w"):
            if not math.isclose(
                _buffer_timestamp(data, buffer_name),
                data_timestamp,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            ):
                raise ValueError(f"{buffer_name} is stale")
        masses_raw = robot.root_physx_view.get_masses()
        masses_all = np.asarray(
            masses_raw.detach().cpu().numpy() if hasattr(masses_raw, "detach") else masses_raw,
            dtype=float,
        )
        if masses_all.shape != (getattr(robot, "num_instances"), len(names)):
            raise ValueError(f"PhysX masses shape {masses_all.shape} is invalid")
        masses = _slice_env(masses_all, env_index, (len(names),), "PhysX masses")
        positions = _slice_env(positions_raw, env_index, (len(names), 3), "body_com_pos_w")
        velocities = _slice_env(velocities_raw, env_index, (len(names), 3), "body_com_lin_vel_w")
        accelerations = _slice_env(accelerations_raw, env_index, (len(names), 3), "body_com_lin_acc_w")
    except Exception as exc:
        return unavailable_whole_body_com(f"{type(exc).__name__}: {exc}", source=source)
    return measure_whole_body_com(
        body_names=names,
        body_masses_kg=masses,
        body_com_positions_w_m=positions,
        body_com_velocities_w_m_s=velocities,
        body_com_accelerations_w_m_s2=accelerations,
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        field_physics_ticks={name: tick for name in COM_FIELD_NAMES},
        expected_body_names=expected_body_names,
        source=source,
    )


@dataclass(frozen=True)
class CentroidalAngularMomentumRateMeasurement:
    available: bool
    physics_tick: int | None
    sim_time_s: float | None
    body_names: tuple[str, ...]
    angular_momentum_rate_w_nm: Vec3 | None
    source: str
    errors: tuple[str, ...] = ()


def unavailable_centroidal_angular_momentum_rate(
    *errors: str,
    source: str,
) -> CentroidalAngularMomentumRateMeasurement:
    reasons = tuple(str(item) for item in errors if str(item)) or (
        "NOT_PROVEN_MISSING_ANGULAR_MOMENTUM_RATE",
    )
    return CentroidalAngularMomentumRateMeasurement(
        available=False,
        physics_tick=None,
        sim_time_s=None,
        body_names=(),
        angular_momentum_rate_w_nm=None,
        source=source,
        errors=reasons,
    )


def _quat_wxyz_to_rotation(value: Any, label: str) -> np.ndarray:
    quaternion = _array(value, label, (4,))
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-5:
        raise ValueError(f"{label} must be a normalized wxyz quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def measure_centroidal_angular_momentum_rate(
    *,
    whole_body_com: WholeBodyCOMMeasurement,
    body_names: Sequence[str],
    body_masses_kg: Any,
    body_com_positions_w_m: Any,
    body_com_linear_accelerations_w_m_s2: Any,
    body_angular_velocities_w_rad_s: Any,
    body_angular_accelerations_w_rad_s2: Any,
    body_com_principal_quaternions_wxyz: Any,
    body_inertias_com_principal_kg_m2: Any,
    physics_tick: int,
    sim_time_s: float,
    physics_dt_s: float,
    field_physics_ticks: Mapping[str, int],
    expected_body_names: Sequence[str] | None = None,
    source: str = "array.centroidal_angular_momentum_rate",
) -> CentroidalAngularMomentumRateMeasurement:
    """Compute independent whole-body angular-momentum rate about COM.

    Each inertia tensor must be about its body COM and diagonal in that body's
    principal-inertia frame.  The matching COM/principal-axis quaternion
    rotates it to world before evaluating ``I*alpha + w x I*w``.  A full
    non-diagonal tensor is rejected because its frame is ambiguous here.
    """

    try:
        tick, time_s, _dt = _strict_time(physics_tick, sim_time_s, physics_dt_s)
        if (
            not whole_body_com.available
            or not whole_body_com.acceleration_available
            or whole_body_com.physics_tick != tick
        ):
            raise ValueError("current whole-body COM/acceleration is unavailable")
        names = _valid_body_names(body_names, expected_body_names)
        if names != whole_body_com.body_names:
            raise ValueError("angular-momentum body names/order differ from COM evidence")
        if not isinstance(field_physics_ticks, Mapping) or set(field_physics_ticks) != ANGULAR_MOMENTUM_FIELD_NAMES:
            raise ValueError("angular-momentum field_physics_ticks keys are not exact")
        for field, field_tick in field_physics_ticks.items():
            if _exact_int(field_tick, f"field_physics_ticks.{field}") != tick:
                raise ValueError(f"{field} does not come from physics_tick")
        count = len(names)
        masses = _array(body_masses_kg, "body_masses_kg", (count,))
        if np.any(masses <= 0.0) or tuple(float(value) for value in masses) != whole_body_com.body_masses_kg:
            raise ValueError("angular-momentum masses differ from COM evidence")
        positions = _array(body_com_positions_w_m, "body_com_positions_w_m", (count, 3))
        linear_accelerations = _array(
            body_com_linear_accelerations_w_m_s2,
            "body_com_linear_accelerations_w_m_s2",
            (count, 3),
        )
        angular_velocities = _array(
            body_angular_velocities_w_rad_s,
            "body_angular_velocities_w_rad_s",
            (count, 3),
        )
        angular_accelerations = _array(
            body_angular_accelerations_w_rad_s2,
            "body_angular_accelerations_w_rad_s2",
            (count, 3),
        )
        quaternions = _array(
            body_com_principal_quaternions_wxyz,
            "body_com_principal_quaternions_wxyz",
            (count, 4),
        )
        inertia_values = np.asarray(body_inertias_com_principal_kg_m2, dtype=float)
        if inertia_values.shape == (count, 9):
            inertia_values = inertia_values.reshape(count, 3, 3)
        if inertia_values.shape != (count, 3, 3) or not np.isfinite(inertia_values).all():
            raise ValueError("body inertias must have exact finite shape (B,3,3) or (B,9)")
        com_position = _vec3(whole_body_com.position_w_m, "whole-body COM position")
        com_acceleration = _vec3(
            whole_body_com.acceleration_w_m_s2,
            "whole-body COM acceleration",
        )
        rate = np.zeros(3, dtype=np.float64)
        for index, name in enumerate(names):
            inertia_principal = np.asarray(inertia_values[index], dtype=np.float64)
            if not np.allclose(inertia_principal, inertia_principal.T, rtol=0.0, atol=1.0e-9):
                raise ValueError(f"{name} inertia tensor is not symmetric")
            if not np.allclose(
                inertia_principal,
                np.diag(np.diag(inertia_principal)),
                rtol=0.0,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"{name} inertia tensor is not diagonal in the COM principal frame"
                )
            eigenvalues = np.diag(inertia_principal)
            if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
                raise ValueError(f"{name} inertia tensor is not positive definite")
            rotation = _quat_wxyz_to_rotation(
                quaternions[index], f"{name} COM principal quaternion"
            )
            inertia_world = rotation @ inertia_principal @ rotation.T
            omega = angular_velocities[index]
            alpha = angular_accelerations[index]
            rotational = inertia_world @ alpha + np.cross(
                omega, inertia_world @ omega
            )
            orbital = np.cross(
                positions[index] - com_position,
                masses[index] * (linear_accelerations[index] - com_acceleration),
            )
            rate += rotational + orbital
        if not np.isfinite(rate).all():
            raise ValueError("centroidal angular-momentum rate is non-finite")
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        return unavailable_centroidal_angular_momentum_rate(str(exc), source=source)
    return CentroidalAngularMomentumRateMeasurement(
        available=True,
        physics_tick=tick,
        sim_time_s=time_s,
        body_names=names,
        angular_momentum_rate_w_nm=tuple(float(value) for value in rate),
        source=source,
    )


def measure_isaac_centroidal_angular_momentum_rate(
    adapter: Any,
    whole_body_com: WholeBodyCOMMeasurement,
    *,
    env_id: int = 0,
    expected_body_names: Sequence[str] | None = None,
) -> CentroidalAngularMomentumRateMeasurement:
    """Read current PhysX body inertia/kinematics and compute independent Ldot."""

    source = "isaaclab.ArticulationData+PhysX.get_inertias.centroidal_Ldot"
    try:
        env_index = _exact_int(env_id, "env_id")
        robot = adapter.robot
        data = robot.data
        tick, time_s, dt = _strict_time(
            adapter.sim_steps,
            adapter.sim_time,
            adapter.sim.get_physics_dt(),
        )
        names = _valid_body_names(robot.body_names, expected_body_names)
        if tuple(data.body_names) != names or tuple(robot.root_physx_view.shared_metatype.link_names) != names:
            raise ValueError("Isaac/PhysX body names/order are inconsistent")
        data_timestamp = _finite_float(data._sim_timestamp, "ArticulationData._sim_timestamp")
        if not math.isclose(
            data_timestamp, time_s, rel_tol=0.0, abs_tol=max(1.0e-9, dt * 1.0e-6)
        ):
            raise ValueError("ArticulationData is not current")
        com_poses_raw = data.body_com_pose_w
        velocities_raw = data.body_com_vel_w
        accelerations_raw = data.body_com_acc_w
        for buffer_name in (
            "_body_com_pose_w",
            "_body_com_vel_w",
            "_body_com_acc_w",
        ):
            if not math.isclose(
                _buffer_timestamp(data, buffer_name),
                data_timestamp,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            ):
                raise ValueError(f"{buffer_name} is stale")
        instances = _exact_int(robot.num_instances, "robot.num_instances", minimum=1)
        count = len(names)
        masses_raw = robot.root_physx_view.get_masses()
        inertias_raw = robot.root_physx_view.get_inertias()
        masses = _slice_env(masses_raw, env_index, (count,), "PhysX masses")
        inertias = _slice_env(inertias_raw, env_index, (count, 9), "PhysX inertias")
        if env_index >= instances:
            raise ValueError("env_id exceeds robot.num_instances")
        com_poses = _slice_env(com_poses_raw, env_index, (count, 7), "body_com_pose_w")
        positions = com_poses[:, :3]
        principal_quaternions = np.asarray(com_poses[:, 3:7], dtype=float)
        velocities = _slice_env(velocities_raw, env_index, (count, 6), "body_com_vel_w")
        accelerations = _slice_env(accelerations_raw, env_index, (count, 6), "body_com_acc_w")
    except Exception as exc:
        return unavailable_centroidal_angular_momentum_rate(
            f"{type(exc).__name__}: {exc}", source=source
        )
    return measure_centroidal_angular_momentum_rate(
        whole_body_com=whole_body_com,
        body_names=names,
        body_masses_kg=masses,
        body_com_positions_w_m=positions,
        body_com_linear_accelerations_w_m_s2=accelerations[:, :3],
        body_angular_velocities_w_rad_s=velocities[:, 3:6],
        body_angular_accelerations_w_rad_s2=accelerations[:, 3:6],
        body_com_principal_quaternions_wxyz=principal_quaternions,
        body_inertias_com_principal_kg_m2=inertias,
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        field_physics_ticks={name: tick for name in ANGULAR_MOMENTUM_FIELD_NAMES},
        expected_body_names=expected_body_names,
        source=source,
    )


@dataclass(frozen=True)
class WheelContactMeasurement:
    leg: str
    wheel_body_name: str
    physics_tick: int
    sim_time_s: float
    surface_kind: str
    surface_height_m: float | None
    surface_normal_w: Vec3 | None
    active: bool
    contact_point_w_m: Vec3 | None
    normal_force_w_n: Vec3 | None
    friction_force_w_n: Vec3 | None
    contact_moment_w_nm: Vec3 | None
    contact_moment_model: str
    dwell_s: float | None
    surface_dwell_verified: bool
    slip_speed_m_s: float | None
    contact_drift_speed_m_s: float | None
    friction_coefficient: float | None
    finite_patch_radius_m: float | None
    source: str


@dataclass(frozen=True)
class ValidatedWheelContact:
    measurement: WheelContactMeasurement
    evidence_available: bool
    support_qualified: bool
    normal_load_n: float | None
    tangential_load_n: float | None
    friction_utilization: float | None
    errors: tuple[str, ...]

    @property
    def leg(self) -> str:
        return self.measurement.leg


@dataclass(frozen=True)
class WheelContactFrame:
    available: bool
    physics_tick: int
    sim_time_s: float
    physics_dt_s: float
    contacts: tuple[ValidatedWheelContact, ...]
    thresholds: "SupportThresholds"
    errors: tuple[str, ...]

    def by_leg(self) -> dict[str, ValidatedWheelContact]:
        return {row.leg: row for row in self.contacts}


@dataclass(frozen=True)
class SupportThresholds:
    minimum_normal_force_n: float = 2.0
    minimum_dwell_s: float = 0.08
    maximum_slip_speed_m_s: float = 0.05
    maximum_contact_drift_speed_m_s: float = 0.05
    coplanar_height_tolerance_m: float = 0.005
    normal_alignment_tolerance_n: float = 1.0e-5
    friction_normal_tolerance_n: float = 1.0e-5
    minimum_diagonal_load_ratio: float = 0.70
    maximum_active_leg_load_n: float = 1.0

    def validated(self) -> "SupportThresholds":
        values = (
            _finite_float(self.minimum_normal_force_n, "minimum_normal_force_n", minimum=0.0),
            _finite_float(self.minimum_dwell_s, "minimum_dwell_s", minimum=0.0),
            _finite_float(self.maximum_slip_speed_m_s, "maximum_slip_speed_m_s", minimum=0.0),
            _finite_float(
                self.maximum_contact_drift_speed_m_s,
                "maximum_contact_drift_speed_m_s",
                minimum=0.0,
            ),
            _finite_float(self.coplanar_height_tolerance_m, "coplanar_height_tolerance_m", minimum=0.0),
            _finite_float(self.normal_alignment_tolerance_n, "normal_alignment_tolerance_n", minimum=0.0),
            _finite_float(self.friction_normal_tolerance_n, "friction_normal_tolerance_n", minimum=0.0),
            _finite_float(self.minimum_diagonal_load_ratio, "minimum_diagonal_load_ratio", minimum=0.0),
            _finite_float(self.maximum_active_leg_load_n, "maximum_active_leg_load_n", minimum=0.0),
        )
        if not 0.5 < values[7] <= 1.0:
            raise ValueError("minimum_diagonal_load_ratio must be in (0.5, 1]")
        return self


def validate_wheel_contact_frame(
    measurements: Sequence[WheelContactMeasurement],
    *,
    physics_tick: int,
    sim_time_s: float,
    physics_dt_s: float,
    thresholds: SupportThresholds = SupportThresholds(),
) -> WheelContactFrame:
    """Validate an exact four-wheel contact frame and qualify support legs."""

    tick, time_s, dt = _strict_time(physics_tick, sim_time_s, physics_dt_s)
    thresholds.validated()
    rows = tuple(measurements)
    frame_errors: list[str] = []
    if len(rows) != len(LEGS) or {row.leg for row in rows} != set(LEGS):
        frame_errors.append("measurements must contain exactly one FL/FR/RL/RR row")
    if len({row.leg for row in rows}) != len(rows):
        frame_errors.append("wheel contact legs are duplicated")
    validated: list[ValidatedWheelContact] = []
    for row in sorted(rows, key=lambda item: LEGS.index(item.leg) if item.leg in LEGS else len(LEGS)):
        errors: list[str] = []
        try:
            if row.leg not in LEGS:
                raise ValueError(f"unknown leg {row.leg!r}")
            if row.wheel_body_name != LEG_TO_WHEEL_BODY[row.leg]:
                raise ValueError("wheel body identity does not match leg")
            if _exact_int(row.physics_tick, f"{row.leg}.physics_tick") != tick:
                raise ValueError("contact is not from the current physics tick")
            row_time = _finite_float(row.sim_time_s, f"{row.leg}.sim_time_s", minimum=0.0)
            if not math.isclose(row_time, time_s, rel_tol=0.0, abs_tol=max(1.0e-9, dt * 1.0e-6)):
                raise ValueError("contact sim_time_s is not current")
            if type(row.active) is not bool:
                raise ValueError("active must be an exact bool")
            if type(row.surface_dwell_verified) is not bool:
                raise ValueError("surface_dwell_verified must be an exact bool")
            surface = str(row.surface_kind)
            if surface not in ALL_SURFACES:
                raise ValueError(f"unsupported surface_kind {surface!r}")
            if surface in SUPPORT_SURFACES:
                height = _finite_float(row.surface_height_m, f"{row.leg}.surface_height_m")
                normal = _unit_vec3(row.surface_normal_w, f"{row.leg}.surface_normal_w")
                if abs(float(normal[2])) < 1.0 - 1.0e-6:
                    raise ValueError("GROUND/OBSTACLE_TOP must provide a horizontal surface normal")
            else:
                height = None
                normal = None

            if row.active:
                if surface not in SUPPORT_SURFACES:
                    raise ValueError("active support evidence is not ground or obstacle top")
                point = _vec3(row.contact_point_w_m, f"{row.leg}.contact_point_w_m")
                normal_force = _vec3(row.normal_force_w_n, f"{row.leg}.normal_force_w_n")
                friction_force = _vec3(row.friction_force_w_n, f"{row.leg}.friction_force_w_n")
                if abs(float(point[2]) - float(height)) > thresholds.coplanar_height_tolerance_m:
                    raise ValueError("contact point is inconsistent with its surface height")
                normal_load = float(np.dot(normal_force, normal))
                if normal_load <= 0.0:
                    raise ValueError("active contact normal load must be positive")
                normal_tangent = normal_force - normal_load * normal
                if float(np.linalg.norm(normal_tangent)) > thresholds.normal_alignment_tolerance_n:
                    raise ValueError("normal force is not aligned with surface normal")
                friction_normal = float(np.dot(friction_force, normal))
                if abs(friction_normal) > thresholds.friction_normal_tolerance_n:
                    raise ValueError("friction force has a normal component")
                tangential = friction_force - friction_normal * normal
                tangential_load = float(np.linalg.norm(tangential))
                mu = _finite_float(row.friction_coefficient, f"{row.leg}.friction_coefficient", minimum=0.0)
                if mu <= 0.0:
                    raise ValueError("friction_coefficient must be positive")
                dwell = _finite_float(row.dwell_s, f"{row.leg}.dwell_s", minimum=0.0)
                slip = _finite_float(row.slip_speed_m_s, f"{row.leg}.slip_speed_m_s", minimum=0.0)
                drift = _finite_float(
                    row.contact_drift_speed_m_s,
                    f"{row.leg}.contact_drift_speed_m_s",
                    minimum=0.0,
                )
                if row.finite_patch_radius_m is not None:
                    _finite_float(
                        row.finite_patch_radius_m,
                        f"{row.leg}.finite_patch_radius_m",
                        minimum=0.0,
                    )
                if row.contact_moment_w_nm is not None:
                    contact_moment = _vec3(
                        row.contact_moment_w_nm,
                        f"{row.leg}.contact_moment_w_nm",
                    )
                    if row.contact_moment_model not in CONTACT_MOMENT_MODELS:
                        raise ValueError("contact_moment_model is invalid")
                    if (
                        row.contact_moment_model
                        == "POINT_CONTACT_ZERO_CONSERVATIVE"
                        and np.any(contact_moment != 0.0)
                    ):
                        raise ValueError(
                            "point-contact conservative moment must be exact zero"
                        )
                elif row.contact_moment_model:
                    raise ValueError("contact_moment_model exists without contact moment")
                utilization = tangential_load / (mu * normal_load) if normal_load > 0.0 else math.inf
                support = bool(
                    normal_load >= thresholds.minimum_normal_force_n
                    and row.surface_dwell_verified
                    and dwell >= thresholds.minimum_dwell_s
                    and slip <= thresholds.maximum_slip_speed_m_s
                    and drift <= thresholds.maximum_contact_drift_speed_m_s
                    and utilization <= 1.0
                )
            else:
                # An inactive row may omit point/dwell/slip, but any supplied force
                # must be finite and must not contradict the inactive flag.
                normal_load = 0.0
                tangential_load = 0.0
                utilization = 0.0
                for value, label in (
                    (row.normal_force_w_n, "normal_force_w_n"),
                    (row.friction_force_w_n, "friction_force_w_n"),
                ):
                    if value is not None and float(np.linalg.norm(_vec3(value, f"{row.leg}.{label}"))) > 1.0e-9:
                        raise ValueError("inactive contact carries nonzero force")
                support = False
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(str(exc))
            normal_load = None
            tangential_load = None
            utilization = None
            support = False
        validated.append(
            ValidatedWheelContact(
                measurement=row,
                evidence_available=not errors,
                support_qualified=support,
                normal_load_n=normal_load,
                tangential_load_n=tangential_load,
                friction_utilization=utilization,
                errors=tuple(errors),
            )
        )
    for row in validated:
        frame_errors.extend(f"{row.leg}: {error}" for error in row.errors)
    return WheelContactFrame(
        available=not frame_errors,
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        contacts=tuple(validated),
        thresholds=thresholds,
        errors=tuple(frame_errors),
    )


@dataclass(frozen=True)
class SupportRegionAssessment:
    status: EvidenceStatus
    model: SupportModel
    physics_tick: int | None
    support_legs: tuple[str, ...]
    contact_points_w_m: tuple[Vec3, ...]
    hull_xy_m: tuple[tuple[float, float], ...]
    signed_margin_m: float | None
    diagonal: str | None
    line_parameter: float | None
    line_distance_m: float | None
    between_contacts: bool | None
    finite_patch_approximation: bool
    corridor_half_width_m: float | None
    corridor_signed_margin_m: float | None
    reasons: tuple[str, ...]


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(row[0]), float(row[1])) for row in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _unavailable_support(*reasons: str, model: SupportModel = SupportModel.NONE) -> SupportRegionAssessment:
    return SupportRegionAssessment(
        status=EvidenceStatus.UNAVAILABLE,
        model=model,
        physics_tick=None,
        support_legs=(),
        contact_points_w_m=(),
        hull_xy_m=(),
        signed_margin_m=None,
        diagonal=None,
        line_parameter=None,
        line_distance_m=None,
        between_contacts=None,
        finite_patch_approximation=False,
        corridor_half_width_m=None,
        corridor_signed_margin_m=None,
        reasons=tuple(reasons) or ("support evidence unavailable",),
    )


def assess_support_region(
    com: WholeBodyCOMMeasurement,
    contacts: WheelContactFrame,
    *,
    thresholds: SupportThresholds = SupportThresholds(),
) -> SupportRegionAssessment:
    """Assess strict coplanar hull or degenerate diagonal line support."""

    thresholds.validated()
    if not com.available or com.position_w_m is None:
        return _unavailable_support("whole-body COM is unavailable")
    if not contacts.available:
        return _unavailable_support(*contacts.errors)
    if com.physics_tick != contacts.physics_tick or not math.isclose(
        float(com.sim_time_s), contacts.sim_time_s, rel_tol=0.0, abs_tol=max(1.0e-9, contacts.physics_dt_s * 1.0e-6)
    ):
        return _unavailable_support("COM and contact frame are not from the same physics tick")
    supporting = tuple(row for row in contacts.contacts if row.support_qualified)
    legs = tuple(row.leg for row in supporting)
    points = tuple(row.measurement.contact_point_w_m for row in supporting)
    if any(point is None for point in points):
        return _unavailable_support("qualified support has no contact point")
    point_values = tuple(point for point in points if point is not None)
    if len(supporting) in (3, 4):
        heights = [float(row.measurement.surface_height_m) for row in supporting]
        if max(heights) - min(heights) > thresholds.coplanar_height_tolerance_m:
            return SupportRegionAssessment(
                status=EvidenceStatus.NOT_PROVEN,
                model=SupportModel.MULTI_HEIGHT_OR_DYNAMIC_WRENCH_REQUIRED,
                physics_tick=contacts.physics_tick,
                support_legs=legs,
                contact_points_w_m=point_values,
                hull_xy_m=(),
                signed_margin_m=None,
                diagonal=None,
                line_parameter=None,
                line_distance_m=None,
                between_contacts=None,
                finite_patch_approximation=False,
                corridor_half_width_m=None,
                corridor_signed_margin_m=None,
                reasons=("support contacts are not coplanar; contact-wrench evidence is required",),
            )
        xy = np.asarray([point[:2] for point in point_values], dtype=float)
        hull = _convex_hull_xy(xy)
        if hull.shape[0] < 3:
            return _unavailable_support("coplanar support hull is degenerate")
        com_xy = np.asarray(com.position_w_m[:2], dtype=float)
        signed_distances: list[float] = []
        for index, start in enumerate(hull):
            end = hull[(index + 1) % len(hull)]
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length <= 1.0e-12:
                return _unavailable_support("coplanar support hull contains a zero-length edge")
            signed_distances.append(float(np.cross(edge, com_xy - start)) / length)
        margin = min(signed_distances)
        return SupportRegionAssessment(
            status=EvidenceStatus.PROVEN,
            model=SupportModel.STRICT_COPLANAR_CONVEX_HULL,
            physics_tick=contacts.physics_tick,
            support_legs=legs,
            contact_points_w_m=point_values,
            hull_xy_m=tuple((float(row[0]), float(row[1])) for row in hull),
            signed_margin_m=margin,
            diagonal=None,
            line_parameter=None,
            line_distance_m=None,
            between_contacts=None,
            finite_patch_approximation=False,
            corridor_half_width_m=None,
            corridor_signed_margin_m=None,
            reasons=() if margin >= 0.0 else ("COM projection is outside the strict coplanar hull",),
        )
    if len(supporting) == 2 and frozenset(legs) in {frozenset(item) for item in DIAGONALS}:
        first = np.asarray(point_values[0][:2], dtype=float)
        second = np.asarray(point_values[1][:2], dtype=float)
        com_xy = np.asarray(com.position_w_m[:2], dtype=float)
        direction = second - first
        length_sq = float(np.dot(direction, direction))
        if length_sq <= 1.0e-12:
            return _unavailable_support("diagonal contact points coincide")
        parameter = float(np.dot(com_xy - first, direction) / length_sq)
        closest = first + min(1.0, max(0.0, parameter)) * direction
        line_distance = abs(float(np.cross(direction, com_xy - first))) / math.sqrt(length_sq)
        segment_distance = float(np.linalg.norm(com_xy - closest))
        between = 0.0 <= parameter <= 1.0
        radii = [row.measurement.finite_patch_radius_m for row in supporting]
        corridor_width = None
        corridor_margin = None
        approximation = False
        if all(radius is not None and float(radius) > 0.0 for radius in radii):
            corridor_width = min(float(radius) for radius in radii if radius is not None)
            corridor_margin = corridor_width - segment_distance
            approximation = True
        diagonal = "FL_RR" if frozenset(legs) == frozenset(("FL", "RR")) else "FR_RL"
        return SupportRegionAssessment(
            status=EvidenceStatus.PROVEN,
            model=SupportModel.DIAGONAL_LINE_SEGMENT,
            physics_tick=contacts.physics_tick,
            support_legs=legs,
            contact_points_w_m=point_values,
            hull_xy_m=(),
            signed_margin_m=(-segment_distance),
            diagonal=diagonal,
            line_parameter=parameter,
            line_distance_m=line_distance,
            between_contacts=between,
            finite_patch_approximation=approximation,
            corridor_half_width_m=corridor_width,
            corridor_signed_margin_m=corridor_margin,
            reasons=(
                "point-contact support is a line segment; corridor is a finite-contact-patch approximation"
                if approximation
                else "point-contact support is a line segment; no finite patch corridor is available"
            ,),
        )
    return _unavailable_support("support set is neither a 3/4-point coplanar hull nor a two-leg diagonal")


@dataclass(frozen=True)
class ContactWrenchFeasibility:
    status: EvidenceStatus
    proven_feasible: bool
    physics_tick: int | None
    witness_kind: str
    multiple_contact_heights: bool | None
    dynamic: bool | None
    angular_momentum_rate_w_nm: Vec3 | None
    angular_momentum_rate_source: str
    force_residual_w_n: Vec3 | None
    force_residual_norm_n: float | None
    moment_residual_w_nm: Vec3 | None
    moment_residual_norm_nm: float | None
    maximum_friction_utilization: float | None
    reasons: tuple[str, ...]


def assess_contact_wrench_feasibility(
    com: WholeBodyCOMMeasurement,
    contacts: WheelContactFrame,
    *,
    angular_momentum_rate: CentroidalAngularMomentumRateMeasurement | None,
    gravity_w_m_s2: Sequence[float] = (0.0, 0.0, -9.81),
    force_residual_tolerance_n: float = 1.0,
    moment_residual_tolerance_nm: float = 0.25,
    dynamic_acceleration_threshold_m_s2: float = 0.1,
) -> ContactWrenchFeasibility:
    """Conservatively certify the measured contact wrench as one witness.

    Failure to certify is ``NOT_PROVEN`` rather than a claim of infeasibility;
    this routine does not silently replace missing moments or solve for
    unmeasured forces.
    """

    reasons: list[str] = []
    if (
        angular_momentum_rate is None
        or not angular_momentum_rate.available
        or angular_momentum_rate.angular_momentum_rate_w_nm is None
    ):
        return ContactWrenchFeasibility(
            status=EvidenceStatus.NOT_PROVEN,
            proven_feasible=False,
            physics_tick=contacts.physics_tick if contacts.available else None,
            witness_kind="MEASURED_CONTACT_WRENCH_SUFFICIENT_CERTIFICATE",
            multiple_contact_heights=None,
            dynamic=None,
            angular_momentum_rate_w_nm=None,
            angular_momentum_rate_source=(
                angular_momentum_rate.source if angular_momentum_rate is not None else ""
            ),
            force_residual_w_n=None,
            force_residual_norm_n=None,
            moment_residual_w_nm=None,
            moment_residual_norm_nm=None,
            maximum_friction_utilization=None,
            reasons=("NOT_PROVEN_MISSING_ANGULAR_MOMENTUM_RATE",),
        )
    try:
        if not com.available or not com.acceleration_available:
            raise ValueError("whole-body COM position/acceleration is unavailable")
        if not contacts.available or com.physics_tick != contacts.physics_tick:
            raise ValueError("contact frame is unavailable or not COM-current")
        mass = _finite_float(com.total_mass_kg, "total_mass_kg", minimum=0.0)
        if mass <= 0.0:
            raise ValueError("total_mass_kg must be positive")
        com_pos = _vec3(com.position_w_m, "COM position")
        com_acc = _vec3(com.acceleration_w_m_s2, "COM acceleration")
        gravity = _vec3(gravity_w_m_s2, "gravity_w_m_s2")
        if (
            angular_momentum_rate.physics_tick != contacts.physics_tick
            or angular_momentum_rate.body_names != com.body_names
            or not math.isclose(
                float(angular_momentum_rate.sim_time_s),
                contacts.sim_time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, contacts.physics_dt_s * 1.0e-6),
            )
        ):
            raise ValueError("angular-momentum rate is not current or body-identity-bound")
        ldot = _vec3(
            angular_momentum_rate.angular_momentum_rate_w_nm,
            "angular_momentum_rate_w_nm",
        )
        force_tol = _finite_float(force_residual_tolerance_n, "force residual tolerance", minimum=0.0)
        moment_tol = _finite_float(moment_residual_tolerance_nm, "moment residual tolerance", minimum=0.0)
        dynamic_threshold = _finite_float(
            dynamic_acceleration_threshold_m_s2,
            "dynamic acceleration threshold",
            minimum=0.0,
        )
        active = [row for row in contacts.contacts if row.measurement.active]
        if not active:
            raise ValueError("no active current-tick contact wrench exists")
        total_force = np.zeros(3, dtype=float)
        total_moment = np.zeros(3, dtype=float)
        utilizations: list[float] = []
        heights: list[float] = []
        moment_models: set[str] = set()
        for row in active:
            if not row.evidence_available:
                raise ValueError(f"{row.leg} contact evidence is unavailable")
            measurement = row.measurement
            if (
                measurement.contact_moment_w_nm is None
                or measurement.contact_moment_model not in CONTACT_MOMENT_MODELS
            ):
                raise ValueError(f"{row.leg} contact moment evidence/model is unavailable")
            point = _vec3(measurement.contact_point_w_m, f"{row.leg} contact point")
            normal_force = _vec3(measurement.normal_force_w_n, f"{row.leg} normal force")
            friction_force = _vec3(measurement.friction_force_w_n, f"{row.leg} friction force")
            contact_moment = _vec3(measurement.contact_moment_w_nm, f"{row.leg} contact moment")
            if (
                measurement.contact_moment_model
                == "POINT_CONTACT_ZERO_CONSERVATIVE"
                and np.any(contact_moment != 0.0)
            ):
                raise ValueError(
                    f"{row.leg} point-contact conservative moment is not exact zero"
                )
            if row.normal_load_n is None or row.normal_load_n < 0.0:
                raise ValueError(f"{row.leg} normal load is invalid")
            if row.friction_utilization is None or not math.isfinite(row.friction_utilization):
                raise ValueError(f"{row.leg} friction utilization is invalid")
            if row.friction_utilization > 1.0:
                reasons.append(f"{row.leg} measured force is outside its friction cone")
            force = normal_force + friction_force
            total_force += force
            total_moment += np.cross(point - com_pos, force) + contact_moment
            utilizations.append(row.friction_utilization)
            heights.append(float(measurement.surface_height_m))
            moment_models.add(measurement.contact_moment_model)
        force_residual = total_force + mass * gravity - mass * com_acc
        moment_residual = total_moment - ldot
        force_norm = float(np.linalg.norm(force_residual))
        moment_norm = float(np.linalg.norm(moment_residual))
        if force_norm > force_tol:
            reasons.append("measured force witness does not close centroidal force balance")
        if moment_norm > moment_tol:
            reasons.append("measured wrench witness does not close centroidal moment balance")
        point_contact_zero = "POINT_CONTACT_ZERO_CONSERVATIVE" in moment_models
        if point_contact_zero:
            reasons.append(
                "NOT_PROVEN_POINT_CONTACT_ZERO_MOMENT_IS_NOT_MEASURED"
            )
        proven = not reasons
        return ContactWrenchFeasibility(
            status=EvidenceStatus.PROVEN if proven else EvidenceStatus.NOT_PROVEN,
            proven_feasible=proven,
            physics_tick=contacts.physics_tick,
            witness_kind=(
                "POINT_CONTACT_ZERO_MOMENT_DIAGNOSTIC_ONLY"
                if point_contact_zero
                else "MEASURED_FORCE_AND_MOMENT_SUFFICIENT_CERTIFICATE"
            ),
            multiple_contact_heights=(max(heights) - min(heights) > 1.0e-6),
            dynamic=(float(np.linalg.norm(com_acc)) > dynamic_threshold),
            angular_momentum_rate_w_nm=tuple(float(value) for value in ldot),
            angular_momentum_rate_source=angular_momentum_rate.source,
            force_residual_w_n=tuple(float(value) for value in force_residual),
            force_residual_norm_n=force_norm,
            moment_residual_w_nm=tuple(float(value) for value in moment_residual),
            moment_residual_norm_nm=moment_norm,
            maximum_friction_utilization=max(utilizations),
            reasons=tuple(reasons),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return ContactWrenchFeasibility(
            status=EvidenceStatus.UNAVAILABLE,
            proven_feasible=False,
            physics_tick=None,
            witness_kind="MEASURED_CONTACT_WRENCH_SUFFICIENT_CERTIFICATE",
            multiple_contact_heights=None,
            dynamic=None,
            angular_momentum_rate_w_nm=None,
            angular_momentum_rate_source=angular_momentum_rate.source,
            force_residual_w_n=None,
            force_residual_norm_n=None,
            moment_residual_w_nm=None,
            moment_residual_norm_nm=None,
            maximum_friction_utilization=None,
            reasons=(str(exc),),
        )


@dataclass(frozen=True)
class DiagonalSupportEvidence:
    status: EvidenceStatus
    classification: str
    physics_tick: int | None
    primary_diagonal: str | None
    load_fl_rr_n: float | None
    load_fr_rl_n: float | None
    load_ratio_fl_rr: float | None
    load_ratio_fr_rl: float | None
    active_swing_leg: str | None
    active_swing_leg_load_n: float | None
    support_region: SupportRegionAssessment
    wrench_feasibility_proven: bool
    reasons: tuple[str, ...]


def assess_primary_diagonal_support(
    com: WholeBodyCOMMeasurement,
    contacts: WheelContactFrame,
    *,
    active_swing_leg: str,
    wrench_feasibility: ContactWrenchFeasibility | None,
    thresholds: SupportThresholds = SupportThresholds(),
    require_wrench_feasibility: bool = True,
) -> DiagonalSupportEvidence:
    thresholds.validated()
    region = assess_support_region(com, contacts, thresholds=thresholds)
    wrench_proven = bool(wrench_feasibility and wrench_feasibility.proven_feasible)
    if active_swing_leg not in LEGS:
        return DiagonalSupportEvidence(
            EvidenceStatus.UNAVAILABLE,
            "NOT_PROVEN",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            region,
            wrench_proven,
            ("active_swing_leg is invalid",),
        )
    if not contacts.available:
        return DiagonalSupportEvidence(
            EvidenceStatus.UNAVAILABLE,
            "NOT_PROVEN",
            None,
            None,
            None,
            None,
            None,
            None,
            active_swing_leg,
            None,
            region,
            wrench_proven,
            contacts.errors,
        )
    by_leg = contacts.by_leg()
    loads = {
        leg: float(by_leg[leg].normal_load_n or 0.0)
        if by_leg[leg].evidence_available and by_leg[leg].measurement.active
        else 0.0
        for leg in LEGS
    }
    load_a = loads["FL"] + loads["RR"]
    load_b = loads["FR"] + loads["RL"]
    total = load_a + load_b
    if total <= 1.0e-12:
        ratios = (None, None)
        candidate = None
    else:
        ratios = (load_a / total, load_b / total)
        candidate = "FL_RR" if ratios[0] >= ratios[1] else "FR_RL"
    reasons: list[str] = []
    pair = ("FL", "RR") if candidate == "FL_RR" else ("FR", "RL") if candidate == "FR_RL" else ()
    ratio = ratios[0] if candidate == "FL_RR" else ratios[1] if candidate == "FR_RL" else None
    if candidate is None or ratio is None or ratio < thresholds.minimum_diagonal_load_ratio:
        reasons.append("neither diagonal carries the required normal-load ratio")
    if pair and not all(by_leg[leg].support_qualified for leg in pair):
        reasons.append("both candidate diagonal legs are not dwell/slip/load-qualified")
    if active_swing_leg in pair:
        reasons.append("active swing leg belongs to the candidate support diagonal")
    active_load = loads[active_swing_leg]
    if active_load > thresholds.maximum_active_leg_load_n:
        reasons.append("active swing leg is not unloaded")
    if (
        region.model != SupportModel.DIAGONAL_LINE_SEGMENT
        or region.diagonal != candidate
        or region.between_contacts is not True
        or not region.finite_patch_approximation
        or region.corridor_signed_margin_m is None
        or region.corridor_signed_margin_m < 0.0
    ):
        reasons.append("whole-body COM is not inside the explicit finite-patch diagonal corridor")
    if require_wrench_feasibility and not wrench_proven:
        reasons.append("current-tick contact-wrench feasibility is not proven")
    proven = not reasons
    return DiagonalSupportEvidence(
        status=EvidenceStatus.PROVEN if proven else EvidenceStatus.NOT_PROVEN,
        classification="PRIMARY_DIAGONAL_SUPPORT" if proven else "NOT_PROVEN",
        physics_tick=contacts.physics_tick,
        primary_diagonal=candidate,
        load_fl_rr_n=load_a,
        load_fr_rl_n=load_b,
        load_ratio_fl_rr=ratios[0],
        load_ratio_fr_rl=ratios[1],
        active_swing_leg=active_swing_leg,
        active_swing_leg_load_n=active_load,
        support_region=region,
        wrench_feasibility_proven=wrench_proven,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class TransferSample:
    com: WholeBodyCOMMeasurement
    contacts: WheelContactFrame
    support_joint_angles_rad: Mapping[str, float]
    body_recoverable: bool
    wrench_feasibility: ContactWrenchFeasibility | None = None


@dataclass(frozen=True)
class TransferThresholds:
    baseline_sample_count: int = 2
    minimum_preload_increase_n: float = 2.0
    release_fraction_of_peak: float = 0.75
    minimum_target_acceleration_m_s2: float = 0.02
    minimum_target_velocity_change_m_s: float = 0.01
    minimum_target_displacement_m: float = 0.005
    settle_acceleration_m_s2: float = 0.02
    impulse_velocity_absolute_tolerance_m_s: float = 0.03
    impulse_velocity_relative_tolerance: float = 0.50
    minimum_support_joint_change_rad: float = 0.01
    minimum_joint_path_efficiency: float = 0.60
    minimum_active_leg_unload_n: float = 1.0
    maximum_final_active_leg_load_n: float = 1.0


@dataclass(frozen=True)
class TransferMethodEvidence:
    status: EvidenceStatus
    method: TransferMethod
    impulse_status: EvidenceStatus
    support_angle_status: EvidenceStatus
    impulse_w_ns: Vec3 | None
    predicted_delta_velocity_w_m_s: Vec3 | None
    measured_delta_velocity_w_m_s: Vec3 | None
    preload_tick: int | None
    release_tick: int | None
    acceleration_tick: int | None
    velocity_tick: int | None
    displacement_tick: int | None
    settle_tick: int | None
    reasons: tuple[str, ...]


def _validate_transfer_samples(samples: Sequence[TransferSample]) -> tuple[TransferSample, ...]:
    rows = tuple(samples)
    if len(rows) < 7:
        raise ValueError("at least seven transfer samples are required")
    previous_tick: int | None = None
    body_identity: tuple[tuple[str, ...], tuple[float, ...], float] | None = None
    for row in rows:
        if type(row.body_recoverable) is not bool:
            raise ValueError("body_recoverable must be an exact bool")
        if not row.com.available or not row.com.acceleration_available or not row.contacts.available:
            raise ValueError("every transfer sample requires current COM/contact evidence")
        if row.com.physics_tick != row.contacts.physics_tick:
            raise ValueError("transfer COM/contact tick mismatch")
        tick = int(row.com.physics_tick)
        if previous_tick is not None and tick != previous_tick + 1:
            raise ValueError("transfer samples must cover consecutive physics ticks")
        previous_tick = tick
        identity = (row.com.body_names, row.com.body_masses_kg, float(row.com.total_mass_kg))
        if body_identity is None:
            body_identity = identity
        elif identity != body_identity:
            raise ValueError("whole-body identity/masses changed within transfer window")
        if not isinstance(row.support_joint_angles_rad, Mapping):
            raise ValueError("support_joint_angles_rad is not a mapping")
        for name, value in row.support_joint_angles_rad.items():
            if type(name) is not str or not name:
                raise ValueError("support joint name is invalid")
            _finite_float(value, f"support joint {name}")
    return rows


def classify_com_transfer(
    samples: Sequence[TransferSample],
    *,
    target_direction_w: Sequence[float],
    support_legs: Sequence[str],
    active_swing_leg: str,
    support_joint_names: Sequence[str],
    thresholds: TransferThresholds = TransferThresholds(),
) -> TransferMethodEvidence:
    """Classify impulse/support-angle transfer only from ordered causal evidence."""

    try:
        rows = _validate_transfer_samples(samples)
        target = _vec3(target_direction_w, "target_direction_w")
        target[2] = 0.0
        target_norm = float(np.linalg.norm(target))
        if target_norm <= 1.0e-12:
            raise ValueError("target_direction_w has no horizontal component")
        target /= target_norm
        support = tuple(support_legs)
        if len(support) < 2 or len(set(support)) != len(support) or any(leg not in LEGS for leg in support):
            raise ValueError("support_legs is invalid")
        if active_swing_leg not in LEGS or active_swing_leg in support:
            raise ValueError("active_swing_leg is invalid for support_legs")
        joint_names = tuple(support_joint_names)
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("support_joint_names must be unique and nonempty")
        baseline_count = _exact_int(thresholds.baseline_sample_count, "baseline_sample_count", minimum=1)
        if baseline_count >= len(rows) - 4:
            raise ValueError("baseline window leaves insufficient causal samples")
        release_fraction = _finite_float(thresholds.release_fraction_of_peak, "release_fraction_of_peak")
        if not 0.0 <= release_fraction < 1.0:
            raise ValueError("release_fraction_of_peak must be in [0, 1)")
        for row in rows:
            by_leg = row.contacts.by_leg()
            if not all(by_leg[leg].support_qualified for leg in support):
                raise ValueError("support legs are not continuously dwell/slip/load-qualified")
            if not row.body_recoverable:
                raise ValueError("body recoverability was lost during transfer")
            if set(row.support_joint_angles_rad) != set(joint_names):
                raise ValueError("support joint angle keys are not exact")

        positions = np.asarray([row.com.position_w_m for row in rows], dtype=float)
        velocities = np.asarray([row.com.velocity_w_m_s for row in rows], dtype=float)
        accelerations = np.asarray([row.com.acceleration_w_m_s2 for row in rows], dtype=float)
        masses = np.asarray([row.com.total_mass_kg for row in rows], dtype=float)
        dt = float(rows[0].com.physics_dt_s)
        if not np.allclose(masses, masses[0], rtol=0.0, atol=0.0):
            raise ValueError("total mass changed within transfer window")
        total_forces: list[np.ndarray] = []
        support_loads: list[float] = []
        active_loads: list[float] = []
        for row in rows:
            by_leg = row.contacts.by_leg()
            force = np.zeros(3, dtype=float)
            for leg in support:
                measurement = by_leg[leg].measurement
                force += _vec3(measurement.normal_force_w_n, f"{leg} normal force")
                force += _vec3(measurement.friction_force_w_n, f"{leg} friction force")
            total_forces.append(force)
            support_loads.append(sum(float(by_leg[leg].normal_load_n) for leg in support))
            active_loads.append(float(by_leg[active_swing_leg].normal_load_n or 0.0))
        force_values = np.asarray(total_forces)
        baseline_force = np.mean(force_values[:baseline_count], axis=0)
        baseline_load = float(np.mean(support_loads[:baseline_count]))
        load_excess = np.asarray(support_loads) - baseline_load
        search = load_excess[baseline_count:]
        peak_index = baseline_count + int(np.argmax(search))
        peak_excess = float(load_excess[peak_index])
        impulse_reasons: list[str] = []
        preload_index: int | None = None
        release_index: int | None = None
        acceleration_index: int | None = None
        velocity_index: int | None = None
        displacement_index: int | None = None
        settle_index: int | None = None
        if any(
            row.wrench_feasibility is None
            or not row.wrench_feasibility.proven_feasible
            or row.wrench_feasibility.physics_tick != row.com.physics_tick
            for row in rows
        ):
            impulse_reasons.append(
                "independent current-tick contact-wrench feasibility is not proven throughout the window"
            )
        if peak_excess < thresholds.minimum_preload_increase_n:
            impulse_reasons.append("support preload increase is below threshold")
        else:
            preload_index = next(
                index
                for index in range(baseline_count, peak_index + 1)
                if load_excess[index] >= thresholds.minimum_preload_increase_n
            )
            release_candidates = [
                index
                for index in range(peak_index + 1, len(rows))
                if load_excess[index] <= peak_excess * release_fraction
            ]
            if release_candidates:
                release_index = release_candidates[0]
            else:
                impulse_reasons.append("no post-preload release is observed")
        target_acc = accelerations @ target
        target_velocity_delta = (velocities - np.mean(velocities[:baseline_count], axis=0)) @ target
        target_displacement = (positions - np.mean(positions[:baseline_count], axis=0)) @ target
        if release_index is not None:
            candidates = [i for i in range(release_index + 1, len(rows)) if target_acc[i] >= thresholds.minimum_target_acceleration_m_s2]
            acceleration_index = candidates[0] if candidates else None
            if acceleration_index is None:
                impulse_reasons.append("no target-directed COM acceleration follows release")
        if acceleration_index is not None:
            candidates = [i for i in range(acceleration_index + 1, len(rows)) if target_velocity_delta[i] >= thresholds.minimum_target_velocity_change_m_s]
            velocity_index = candidates[0] if candidates else None
            if velocity_index is None:
                impulse_reasons.append("no target-directed COM velocity follows acceleration")
        if velocity_index is not None:
            candidates = [i for i in range(velocity_index + 1, len(rows)) if target_displacement[i] >= thresholds.minimum_target_displacement_m]
            displacement_index = candidates[0] if candidates else None
            if displacement_index is None:
                impulse_reasons.append("no target-directed COM displacement follows velocity")
        if displacement_index is not None:
            candidates = [i for i in range(displacement_index + 1, len(rows)) if abs(float(target_acc[i])) <= thresholds.settle_acceleration_m_s2]
            settle_index = candidates[0] if candidates else None
            if settle_index is None:
                impulse_reasons.append("no settle/coast sample follows displacement")

        integration_end = settle_index if settle_index is not None else len(rows) - 1
        impulse = np.sum(force_values[: integration_end + 1] - baseline_force, axis=0) * dt
        predicted_dv = impulse / masses[0]
        measured_dv = velocities[integration_end] - np.mean(velocities[:baseline_count], axis=0)
        predicted_target = float(np.dot(predicted_dv, target))
        measured_target = float(np.dot(measured_dv, target))
        tolerance = thresholds.impulse_velocity_absolute_tolerance_m_s + thresholds.impulse_velocity_relative_tolerance * max(abs(predicted_target), abs(measured_target))
        if predicted_target <= 0.0 or measured_target <= 0.0:
            impulse_reasons.append("impulse and measured delta velocity are not both target-directed")
        elif float(np.linalg.norm(predicted_dv - measured_dv)) > tolerance:
            impulse_reasons.append("impulse/M does not agree with measured COM delta velocity")
        if active_loads[0] - active_loads[-1] < thresholds.minimum_active_leg_unload_n:
            impulse_reasons.append("active swing-leg load did not decrease after the impulse")
        if active_loads[-1] > thresholds.maximum_final_active_leg_load_n:
            impulse_reasons.append("active swing leg remains loaded after the impulse")
        impulse_proven = not impulse_reasons

        angle_reasons: list[str] = []
        if any(
            row.wrench_feasibility is None
            or not row.wrench_feasibility.proven_feasible
            or row.wrench_feasibility.physics_tick != row.com.physics_tick
            for row in rows
        ):
            angle_reasons.append(
                "independent current-tick contact-wrench feasibility is not proven throughout the window"
            )
        angle_values = np.asarray(
            [[float(row.support_joint_angles_rad[name]) for name in joint_names] for row in rows],
            dtype=float,
        )
        net_angle = angle_values[-1] - angle_values[0]
        path_length = float(np.sum(np.linalg.norm(np.diff(angle_values, axis=0), axis=1)))
        net_length = float(np.linalg.norm(net_angle))
        if net_length < thresholds.minimum_support_joint_change_rad:
            angle_reasons.append("support-joint change is below threshold")
        if path_length <= 1.0e-12 or net_length / path_length < thresholds.minimum_joint_path_efficiency:
            angle_reasons.append("support-joint change is not sufficiently progressive")
        if target_displacement[-1] < thresholds.minimum_target_displacement_m:
            angle_reasons.append("COM did not move toward the target support region")
        if active_loads[0] - active_loads[-1] < thresholds.minimum_active_leg_unload_n:
            angle_reasons.append("active swing-leg load did not decrease sufficiently")
        if active_loads[-1] > thresholds.maximum_final_active_leg_load_n:
            angle_reasons.append("active swing leg remains loaded")
        angle_proven = not angle_reasons
        if impulse_proven and angle_proven:
            method = TransferMethod.HYBRID_COM_TRANSFER
        elif impulse_proven:
            method = TransferMethod.IMPULSE_BASED_COM_TRANSFER
        elif angle_proven:
            method = TransferMethod.SUPPORT_ANGLE_COM_TRANSFER
        else:
            method = TransferMethod.NOT_YET_PROVEN
        return TransferMethodEvidence(
            status=EvidenceStatus.PROVEN if method != TransferMethod.NOT_YET_PROVEN else EvidenceStatus.NOT_PROVEN,
            method=method,
            impulse_status=EvidenceStatus.PROVEN if impulse_proven else EvidenceStatus.NOT_PROVEN,
            support_angle_status=EvidenceStatus.PROVEN if angle_proven else EvidenceStatus.NOT_PROVEN,
            impulse_w_ns=tuple(float(value) for value in impulse),
            predicted_delta_velocity_w_m_s=tuple(float(value) for value in predicted_dv),
            measured_delta_velocity_w_m_s=tuple(float(value) for value in measured_dv),
            preload_tick=int(rows[preload_index].com.physics_tick) if preload_index is not None else None,
            release_tick=int(rows[release_index].com.physics_tick) if release_index is not None else None,
            acceleration_tick=int(rows[acceleration_index].com.physics_tick) if acceleration_index is not None else None,
            velocity_tick=int(rows[velocity_index].com.physics_tick) if velocity_index is not None else None,
            displacement_tick=int(rows[displacement_index].com.physics_tick) if displacement_index is not None else None,
            settle_tick=int(rows[settle_index].com.physics_tick) if settle_index is not None else None,
            reasons=tuple(impulse_reasons + angle_reasons),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        return TransferMethodEvidence(
            status=EvidenceStatus.UNAVAILABLE,
            method=TransferMethod.NOT_YET_PROVEN,
            impulse_status=EvidenceStatus.UNAVAILABLE,
            support_angle_status=EvidenceStatus.UNAVAILABLE,
            impulse_w_ns=None,
            predicted_delta_velocity_w_m_s=None,
            measured_delta_velocity_w_m_s=None,
            preload_tick=None,
            release_tick=None,
            acceleration_tick=None,
            velocity_tick=None,
            displacement_tick=None,
            settle_tick=None,
            reasons=(str(exc),),
        )


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _com_mapping(value: WholeBodyCOMMeasurement) -> dict[str, Any]:
    return {
        "com_measurement_available": value.com_measurement_available,
        "acceleration_available": value.acceleration_available,
        "physics_tick": value.physics_tick,
        "sim_time_s": value.sim_time_s,
        "physics_dt_s": value.physics_dt_s,
        "body_names": list(value.body_names),
        "body_masses_kg": list(value.body_masses_kg),
        "total_mass_kg": value.total_mass_kg,
        "position_w_m": list(value.position_w_m) if value.position_w_m is not None else None,
        "velocity_w_m_s": list(value.velocity_w_m_s) if value.velocity_w_m_s is not None else None,
        "acceleration_w_m_s2": (
            list(value.acceleration_w_m_s2) if value.acceleration_w_m_s2 is not None else None
        ),
        "source": value.source,
        "errors": list(value.errors),
    }


def _angular_rate_mapping(
    value: CentroidalAngularMomentumRateMeasurement,
) -> dict[str, Any]:
    return {
        "available": value.available,
        "physics_tick": value.physics_tick,
        "sim_time_s": value.sim_time_s,
        "body_names": list(value.body_names),
        "angular_momentum_rate_w_nm": (
            list(value.angular_momentum_rate_w_nm)
            if value.angular_momentum_rate_w_nm is not None
            else None
        ),
        "source": value.source,
        "errors": list(value.errors),
    }


def _measurement_mapping(value: WheelContactMeasurement) -> dict[str, Any]:
    return {
        "leg": value.leg,
        "wheel_body_name": value.wheel_body_name,
        "physics_tick": value.physics_tick,
        "sim_time_s": value.sim_time_s,
        "surface_kind": value.surface_kind,
        "surface_height_m": value.surface_height_m,
        "surface_normal_w": list(value.surface_normal_w) if value.surface_normal_w is not None else None,
        "active": value.active,
        "contact_point_w_m": list(value.contact_point_w_m) if value.contact_point_w_m is not None else None,
        "normal_force_w_n": list(value.normal_force_w_n) if value.normal_force_w_n is not None else None,
        "friction_force_w_n": list(value.friction_force_w_n) if value.friction_force_w_n is not None else None,
        "contact_moment_w_nm": (
            list(value.contact_moment_w_nm) if value.contact_moment_w_nm is not None else None
        ),
        "contact_moment_model": value.contact_moment_model,
        "dwell_s": value.dwell_s,
        "surface_dwell_verified": value.surface_dwell_verified,
        "slip_speed_m_s": value.slip_speed_m_s,
        "contact_drift_speed_m_s": value.contact_drift_speed_m_s,
        "friction_coefficient": value.friction_coefficient,
        "finite_patch_radius_m": value.finite_patch_radius_m,
        "source": value.source,
    }


def _contact_frame_mapping(value: WheelContactFrame) -> dict[str, Any]:
    return {
        "available": value.available,
        "physics_tick": value.physics_tick,
        "sim_time_s": value.sim_time_s,
        "physics_dt_s": value.physics_dt_s,
        "contacts": [
            {
                "measurement": _measurement_mapping(row.measurement),
                "evidence_available": row.evidence_available,
                "support_qualified": row.support_qualified,
                "normal_load_n": row.normal_load_n,
                "tangential_load_n": row.tangential_load_n,
                "friction_utilization": row.friction_utilization,
                "errors": list(row.errors),
            }
            for row in value.contacts
        ],
        "thresholds": {
            "minimum_normal_force_n": value.thresholds.minimum_normal_force_n,
            "minimum_dwell_s": value.thresholds.minimum_dwell_s,
            "maximum_slip_speed_m_s": value.thresholds.maximum_slip_speed_m_s,
            "maximum_contact_drift_speed_m_s": value.thresholds.maximum_contact_drift_speed_m_s,
            "coplanar_height_tolerance_m": value.thresholds.coplanar_height_tolerance_m,
            "normal_alignment_tolerance_n": value.thresholds.normal_alignment_tolerance_n,
            "friction_normal_tolerance_n": value.thresholds.friction_normal_tolerance_n,
            "minimum_diagonal_load_ratio": value.thresholds.minimum_diagonal_load_ratio,
            "maximum_active_leg_load_n": value.thresholds.maximum_active_leg_load_n,
        },
        "errors": list(value.errors),
    }


def _support_mapping(value: SupportRegionAssessment) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "model": value.model.value,
        "physics_tick": value.physics_tick,
        "support_legs": list(value.support_legs),
        "contact_points_w_m": [list(point) for point in value.contact_points_w_m],
        "hull_xy_m": [list(point) for point in value.hull_xy_m],
        "signed_margin_m": value.signed_margin_m,
        "diagonal": value.diagonal,
        "line_parameter": value.line_parameter,
        "line_distance_m": value.line_distance_m,
        "between_contacts": value.between_contacts,
        "finite_patch_approximation": value.finite_patch_approximation,
        "corridor_half_width_m": value.corridor_half_width_m,
        "corridor_signed_margin_m": value.corridor_signed_margin_m,
        "reasons": list(value.reasons),
    }


def _wrench_mapping(value: ContactWrenchFeasibility) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "proven_feasible": value.proven_feasible,
        "physics_tick": value.physics_tick,
        "witness_kind": value.witness_kind,
        "multiple_contact_heights": value.multiple_contact_heights,
        "dynamic": value.dynamic,
        "angular_momentum_rate_w_nm": (
            list(value.angular_momentum_rate_w_nm)
            if value.angular_momentum_rate_w_nm is not None
            else None
        ),
        "angular_momentum_rate_source": value.angular_momentum_rate_source,
        "force_residual_w_n": (
            list(value.force_residual_w_n) if value.force_residual_w_n is not None else None
        ),
        "force_residual_norm_n": value.force_residual_norm_n,
        "moment_residual_w_nm": (
            list(value.moment_residual_w_nm) if value.moment_residual_w_nm is not None else None
        ),
        "moment_residual_norm_nm": value.moment_residual_norm_nm,
        "maximum_friction_utilization": value.maximum_friction_utilization,
        "reasons": list(value.reasons),
    }


def _diagonal_mapping(value: DiagonalSupportEvidence) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "classification": value.classification,
        "physics_tick": value.physics_tick,
        "primary_diagonal": value.primary_diagonal,
        "load_fl_rr_n": value.load_fl_rr_n,
        "load_fr_rl_n": value.load_fr_rl_n,
        "load_ratio_fl_rr": value.load_ratio_fl_rr,
        "load_ratio_fr_rl": value.load_ratio_fr_rl,
        "active_swing_leg": value.active_swing_leg,
        "active_swing_leg_load_n": value.active_swing_leg_load_n,
        "wrench_feasibility_proven": value.wrench_feasibility_proven,
        "reasons": list(value.reasons),
    }


def _centroidal_payload(
    *,
    sim_step: int,
    physics_time_s: float,
    physics_dt_s: float,
    whole_body_com: WholeBodyCOMMeasurement,
    centroidal_angular_momentum_rate: CentroidalAngularMomentumRateMeasurement,
    wheel_contacts: WheelContactFrame,
    support_region: SupportRegionAssessment,
    contact_wrench_feasibility: ContactWrenchFeasibility,
    diagonal_support: DiagonalSupportEvidence,
) -> dict[str, Any]:
    return {
        "sim_step": sim_step,
        "physics_time_s": physics_time_s,
        "physics_dt_s": physics_dt_s,
        "whole_body_com": _com_mapping(whole_body_com),
        "centroidal_angular_momentum_rate": _angular_rate_mapping(
            centroidal_angular_momentum_rate
        ),
        "wheel_contacts": _contact_frame_mapping(wheel_contacts),
        "support_region": _support_mapping(support_region),
        "contact_wrench_feasibility": _wrench_mapping(contact_wrench_feasibility),
        "diagonal_support": _diagonal_mapping(diagonal_support),
    }


@dataclass(frozen=True)
class CentroidalSupportEvidence:
    """One SHA-bound immutable current-tick observation envelope."""

    schema_version: str
    sim_step: int
    physics_time_s: float
    physics_dt_s: float
    whole_body_com: WholeBodyCOMMeasurement
    centroidal_angular_momentum_rate: CentroidalAngularMomentumRateMeasurement
    wheel_contacts: WheelContactFrame
    support_region: SupportRegionAssessment
    contact_wrench_feasibility: ContactWrenchFeasibility
    diagonal_support: DiagonalSupportEvidence
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA:
            raise ValueError("centroidal support schema_version mismatch")
        tick, time_s, dt = _strict_time(
            self.sim_step, self.physics_time_s, self.physics_dt_s
        )
        if (
            self.wheel_contacts.physics_tick != tick
            or not math.isclose(
                self.wheel_contacts.sim_time_s,
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
            or not math.isclose(self.wheel_contacts.physics_dt_s, dt, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("wheel contact frame is not envelope-current")
        if self.whole_body_com.physics_tick is not None and (
            self.whole_body_com.physics_tick != tick
            or not math.isclose(
                float(self.whole_body_com.sim_time_s),
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
        ):
            raise ValueError("whole-body COM is not envelope-current")
        angular_rate = self.centroidal_angular_momentum_rate
        if angular_rate.physics_tick is not None and (
            angular_rate.physics_tick != tick
            or angular_rate.body_names != self.whole_body_com.body_names
            or not math.isclose(
                float(angular_rate.sim_time_s),
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
        ):
            raise ValueError("centroidal angular-momentum rate is not envelope-current")
        if self.contact_wrench_feasibility.proven_feasible and (
            not angular_rate.available
            or angular_rate.angular_momentum_rate_w_nm
            != self.contact_wrench_feasibility.angular_momentum_rate_w_nm
            or angular_rate.source
            != self.contact_wrench_feasibility.angular_momentum_rate_source
        ):
            raise ValueError("proven wrench does not bind the independent angular-momentum rate")
        for label, component_tick in (
            ("support_region", self.support_region.physics_tick),
            ("contact_wrench_feasibility", self.contact_wrench_feasibility.physics_tick),
            ("diagonal_support", self.diagonal_support.physics_tick),
        ):
            if component_tick is not None and component_tick != tick:
                raise ValueError(f"{label} is not envelope-current")
        if self.diagonal_support.support_region != self.support_region:
            raise ValueError("diagonal support does not bind the envelope support region")
        if (
            self.diagonal_support.wrench_feasibility_proven
            != self.contact_wrench_feasibility.proven_feasible
        ):
            raise ValueError("diagonal support does not bind the envelope wrench result")
        if type(self.payload_sha256) is not str or len(self.payload_sha256) != 64:
            raise ValueError("payload_sha256 is invalid")
        try:
            int(self.payload_sha256, 16)
        except ValueError as exc:
            raise ValueError("payload_sha256 is invalid") from exc
        if self.payload_sha256 != _canonical_json_sha256(self._payload_mapping()):
            raise ValueError("centroidal support payload SHA mismatch")

    @classmethod
    def create(
        cls,
        *,
        sim_step: int,
        physics_time_s: float,
        physics_dt_s: float,
        whole_body_com: WholeBodyCOMMeasurement,
        centroidal_angular_momentum_rate: CentroidalAngularMomentumRateMeasurement,
        wheel_contacts: WheelContactFrame,
        support_region: SupportRegionAssessment,
        contact_wrench_feasibility: ContactWrenchFeasibility,
        diagonal_support: DiagonalSupportEvidence,
    ) -> "CentroidalSupportEvidence":
        tick, time_s, dt = _strict_time(sim_step, physics_time_s, physics_dt_s)
        payload = _centroidal_payload(
            sim_step=tick,
            physics_time_s=time_s,
            physics_dt_s=dt,
            whole_body_com=whole_body_com,
            centroidal_angular_momentum_rate=centroidal_angular_momentum_rate,
            wheel_contacts=wheel_contacts,
            support_region=support_region,
            contact_wrench_feasibility=contact_wrench_feasibility,
            diagonal_support=diagonal_support,
        )
        return cls(
            schema_version=CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA,
            sim_step=tick,
            physics_time_s=time_s,
            physics_dt_s=dt,
            whole_body_com=whole_body_com,
            centroidal_angular_momentum_rate=centroidal_angular_momentum_rate,
            wheel_contacts=wheel_contacts,
            support_region=support_region,
            contact_wrench_feasibility=contact_wrench_feasibility,
            diagonal_support=diagonal_support,
            payload_sha256=_canonical_json_sha256(payload),
        )

    def _payload_mapping(self) -> dict[str, Any]:
        return _centroidal_payload(
            sim_step=self.sim_step,
            physics_time_s=self.physics_time_s,
            physics_dt_s=self.physics_dt_s,
            whole_body_com=self.whole_body_com,
            centroidal_angular_momentum_rate=self.centroidal_angular_momentum_rate,
            wheel_contacts=self.wheel_contacts,
            support_region=self.support_region,
            contact_wrench_feasibility=self.contact_wrench_feasibility,
            diagonal_support=self.diagonal_support,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload_sha256": self.payload_sha256,
            "payload": self._payload_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CentroidalSupportEvidence":
        root = _json_mapping(value, "centroidal support evidence")
        _json_exact_keys(root, {"schema_version", "payload_sha256", "payload"}, "centroidal support evidence")
        if _json_text(root["schema_version"], "schema_version") != CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA:
            raise ValueError("centroidal support schema_version mismatch")
        digest = _json_text(root["payload_sha256"], "payload_sha256")
        payload = _json_mapping(root["payload"], "payload")
        _json_exact_keys(
            payload,
            {
                "sim_step",
                "physics_time_s",
                "physics_dt_s",
                "whole_body_com",
                "centroidal_angular_momentum_rate",
                "wheel_contacts",
                "support_region",
                "contact_wrench_feasibility",
                "diagonal_support",
            },
            "payload",
        )
        if digest != _canonical_json_sha256(payload):
            raise ValueError("centroidal support payload SHA mismatch")
        tick, time_s, dt = _strict_time(
            payload["sim_step"], payload["physics_time_s"], payload["physics_dt_s"]
        )
        com = _com_from_mapping(payload["whole_body_com"])
        angular_rate = _angular_rate_from_mapping(
            payload["centroidal_angular_momentum_rate"]
        )
        contacts = _contact_frame_from_mapping(payload["wheel_contacts"])
        support = _support_from_mapping(payload["support_region"])
        wrench = _wrench_from_mapping(payload["contact_wrench_feasibility"])
        diagonal = _diagonal_from_mapping(payload["diagonal_support"], support_region=support)
        return cls(
            schema_version=CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA,
            sim_step=tick,
            physics_time_s=time_s,
            physics_dt_s=dt,
            whole_body_com=com,
            centroidal_angular_momentum_rate=angular_rate,
            wheel_contacts=contacts,
            support_region=support,
            contact_wrench_feasibility=wrench,
            diagonal_support=diagonal,
            payload_sha256=digest,
        )


def _json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{label} contains a non-string key")
    return value


def _json_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys are not exact")


def _json_text(value: Any, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    return value


def _json_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact bool")
    return value


def _json_optional_bool(value: Any, label: str) -> bool | None:
    return None if value is None else _json_bool(value, label)


def _json_optional_float(value: Any, label: str) -> float | None:
    return None if value is None else _finite_float(value, label)


def _json_optional_int(value: Any, label: str) -> int | None:
    return None if value is None else _exact_int(value, label)


def _json_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON list")
    return value


def _json_strings(value: Any, label: str) -> tuple[str, ...]:
    rows = _json_list(value, label)
    return tuple(_json_text(item, label) for item in rows)


def _json_optional_vec(value: Any, size: int, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    rows = _json_list(value, label)
    array = _array(rows, label, (size,))
    return tuple(float(item) for item in array)


def _json_status(value: Any, label: str) -> EvidenceStatus:
    try:
        return EvidenceStatus(_json_text(value, label))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc


def _com_from_mapping(value: Any) -> WholeBodyCOMMeasurement:
    row = _json_mapping(value, "whole_body_com")
    keys = {
        "com_measurement_available",
        "acceleration_available",
        "physics_tick",
        "sim_time_s",
        "physics_dt_s",
        "body_names",
        "body_masses_kg",
        "total_mass_kg",
        "position_w_m",
        "velocity_w_m_s",
        "acceleration_w_m_s2",
        "source",
        "errors",
    }
    _json_exact_keys(row, keys, "whole_body_com")
    result = WholeBodyCOMMeasurement(
        com_measurement_available=_json_bool(row["com_measurement_available"], "com_measurement_available"),
        acceleration_available=_json_bool(row["acceleration_available"], "acceleration_available"),
        physics_tick=_json_optional_int(row["physics_tick"], "COM physics_tick"),
        sim_time_s=_json_optional_float(row["sim_time_s"], "COM sim_time_s"),
        physics_dt_s=_json_optional_float(row["physics_dt_s"], "COM physics_dt_s"),
        body_names=_json_strings(row["body_names"], "COM body_names"),
        body_masses_kg=tuple(
            _finite_float(item, "COM body mass", minimum=0.0)
            for item in _json_list(row["body_masses_kg"], "COM body_masses_kg")
        ),
        total_mass_kg=_json_optional_float(row["total_mass_kg"], "COM total_mass_kg"),
        position_w_m=_json_optional_vec(row["position_w_m"], 3, "COM position_w_m"),
        velocity_w_m_s=_json_optional_vec(row["velocity_w_m_s"], 3, "COM velocity_w_m_s"),
        acceleration_w_m_s2=_json_optional_vec(
            row["acceleration_w_m_s2"], 3, "COM acceleration_w_m_s2"
        ),
        source=_json_text(row["source"], "COM source"),
        errors=_json_strings(row["errors"], "COM errors"),
    )
    if result.available:
        if not result.acceleration_available:
            raise ValueError("available COM must include acceleration")
        _valid_body_names(result.body_names, None)
        if len(result.body_masses_kg) != len(result.body_names) or any(
            mass <= 0.0 for mass in result.body_masses_kg
        ):
            raise ValueError("available COM body masses are invalid")
        total = sum(result.body_masses_kg)
        if result.total_mass_kg is None or not math.isclose(
            total, result.total_mass_kg, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError("available COM total mass is inconsistent")
        if any(
            item is None
            for item in (result.physics_tick, result.sim_time_s, result.physics_dt_s, result.position_w_m, result.velocity_w_m_s, result.acceleration_w_m_s2)
        ):
            raise ValueError("available COM mapping omits required evidence")
        _strict_time(result.physics_tick, result.sim_time_s, result.physics_dt_s)
    elif any(
        item is not None
        for item in (result.physics_tick, result.sim_time_s, result.physics_dt_s, result.total_mass_kg, result.position_w_m, result.velocity_w_m_s, result.acceleration_w_m_s2)
    ):
        raise ValueError("unavailable COM mapping carries pseudo-measurement values")
    if not result.available and not result.errors:
        raise ValueError("unavailable COM mapping must explain the missing evidence")
    return result


def _angular_rate_from_mapping(
    value: Any,
) -> CentroidalAngularMomentumRateMeasurement:
    row = _json_mapping(value, "centroidal_angular_momentum_rate")
    _json_exact_keys(
        row,
        {
            "available",
            "physics_tick",
            "sim_time_s",
            "body_names",
            "angular_momentum_rate_w_nm",
            "source",
            "errors",
        },
        "centroidal_angular_momentum_rate",
    )
    result = CentroidalAngularMomentumRateMeasurement(
        available=_json_bool(row["available"], "angular rate available"),
        physics_tick=_json_optional_int(row["physics_tick"], "angular rate physics_tick"),
        sim_time_s=_json_optional_float(row["sim_time_s"], "angular rate sim_time_s"),
        body_names=_json_strings(row["body_names"], "angular rate body_names"),
        angular_momentum_rate_w_nm=_json_optional_vec(
            row["angular_momentum_rate_w_nm"],
            3,
            "angular_momentum_rate_w_nm",
        ),
        source=_json_text(row["source"], "angular rate source"),
        errors=_json_strings(row["errors"], "angular rate errors"),
    )
    if result.available:
        _valid_body_names(result.body_names, None)
        if any(
            item is None
            for item in (
                result.physics_tick,
                result.sim_time_s,
                result.angular_momentum_rate_w_nm,
            )
        ):
            raise ValueError("available angular-momentum rate omits required evidence")
    elif any(
        item is not None
        for item in (
            result.physics_tick,
            result.sim_time_s,
            result.angular_momentum_rate_w_nm,
        )
    ) or result.body_names:
        raise ValueError("unavailable angular-momentum rate carries pseudo-measurement values")
    if not result.available and not result.errors:
        raise ValueError(
            "unavailable angular-momentum rate must explain the missing evidence"
        )
    return result


def _measurement_from_mapping(value: Any) -> WheelContactMeasurement:
    row = _json_mapping(value, "wheel contact measurement")
    keys = {
        "leg",
        "wheel_body_name",
        "physics_tick",
        "sim_time_s",
        "surface_kind",
        "surface_height_m",
        "surface_normal_w",
        "active",
        "contact_point_w_m",
        "normal_force_w_n",
        "friction_force_w_n",
        "contact_moment_w_nm",
        "contact_moment_model",
        "dwell_s",
        "surface_dwell_verified",
        "slip_speed_m_s",
        "contact_drift_speed_m_s",
        "friction_coefficient",
        "finite_patch_radius_m",
        "source",
    }
    _json_exact_keys(row, keys, "wheel contact measurement")
    return WheelContactMeasurement(
        leg=_json_text(row["leg"], "contact leg"),
        wheel_body_name=_json_text(row["wheel_body_name"], "wheel_body_name"),
        physics_tick=_exact_int(row["physics_tick"], "contact physics_tick"),
        sim_time_s=_finite_float(row["sim_time_s"], "contact sim_time_s", minimum=0.0),
        surface_kind=_json_text(row["surface_kind"], "surface_kind"),
        surface_height_m=_json_optional_float(row["surface_height_m"], "surface_height_m"),
        surface_normal_w=_json_optional_vec(row["surface_normal_w"], 3, "surface_normal_w"),
        active=_json_bool(row["active"], "contact active"),
        contact_point_w_m=_json_optional_vec(row["contact_point_w_m"], 3, "contact_point_w_m"),
        normal_force_w_n=_json_optional_vec(row["normal_force_w_n"], 3, "normal_force_w_n"),
        friction_force_w_n=_json_optional_vec(row["friction_force_w_n"], 3, "friction_force_w_n"),
        contact_moment_w_nm=_json_optional_vec(row["contact_moment_w_nm"], 3, "contact_moment_w_nm"),
        contact_moment_model=_json_text(row["contact_moment_model"], "contact_moment_model"),
        dwell_s=_json_optional_float(row["dwell_s"], "dwell_s"),
        surface_dwell_verified=_json_bool(row["surface_dwell_verified"], "surface_dwell_verified"),
        slip_speed_m_s=_json_optional_float(row["slip_speed_m_s"], "slip_speed_m_s"),
        contact_drift_speed_m_s=_json_optional_float(
            row["contact_drift_speed_m_s"], "contact_drift_speed_m_s"
        ),
        friction_coefficient=_json_optional_float(row["friction_coefficient"], "friction_coefficient"),
        finite_patch_radius_m=_json_optional_float(row["finite_patch_radius_m"], "finite_patch_radius_m"),
        source=_json_text(row["source"], "contact source"),
    )


def _contact_frame_from_mapping(value: Any) -> WheelContactFrame:
    row = _json_mapping(value, "wheel_contacts")
    _json_exact_keys(
        row,
        {
            "available",
            "physics_tick",
            "sim_time_s",
            "physics_dt_s",
            "contacts",
            "thresholds",
            "errors",
        },
        "wheel_contacts",
    )
    tick, time_s, dt = _strict_time(row["physics_tick"], row["sim_time_s"], row["physics_dt_s"])
    threshold_row = _json_mapping(row["thresholds"], "support thresholds")
    threshold_keys = {
        "minimum_normal_force_n",
        "minimum_dwell_s",
        "maximum_slip_speed_m_s",
        "maximum_contact_drift_speed_m_s",
        "coplanar_height_tolerance_m",
        "normal_alignment_tolerance_n",
        "friction_normal_tolerance_n",
        "minimum_diagonal_load_ratio",
        "maximum_active_leg_load_n",
    }
    _json_exact_keys(threshold_row, threshold_keys, "support thresholds")
    thresholds = SupportThresholds(
        minimum_normal_force_n=_finite_float(
            threshold_row["minimum_normal_force_n"], "minimum_normal_force_n", minimum=0.0
        ),
        minimum_dwell_s=_finite_float(
            threshold_row["minimum_dwell_s"], "minimum_dwell_s", minimum=0.0
        ),
        maximum_slip_speed_m_s=_finite_float(
            threshold_row["maximum_slip_speed_m_s"], "maximum_slip_speed_m_s", minimum=0.0
        ),
        maximum_contact_drift_speed_m_s=_finite_float(
            threshold_row["maximum_contact_drift_speed_m_s"],
            "maximum_contact_drift_speed_m_s",
            minimum=0.0,
        ),
        coplanar_height_tolerance_m=_finite_float(
            threshold_row["coplanar_height_tolerance_m"],
            "coplanar_height_tolerance_m",
            minimum=0.0,
        ),
        normal_alignment_tolerance_n=_finite_float(
            threshold_row["normal_alignment_tolerance_n"],
            "normal_alignment_tolerance_n",
            minimum=0.0,
        ),
        friction_normal_tolerance_n=_finite_float(
            threshold_row["friction_normal_tolerance_n"],
            "friction_normal_tolerance_n",
            minimum=0.0,
        ),
        minimum_diagonal_load_ratio=_finite_float(
            threshold_row["minimum_diagonal_load_ratio"],
            "minimum_diagonal_load_ratio",
            minimum=0.0,
        ),
        maximum_active_leg_load_n=_finite_float(
            threshold_row["maximum_active_leg_load_n"],
            "maximum_active_leg_load_n",
            minimum=0.0,
        ),
    )
    thresholds.validated()
    contacts_raw = _json_list(row["contacts"], "wheel contacts")
    parsed: list[ValidatedWheelContact] = []
    for item in contacts_raw:
        contact = _json_mapping(item, "validated wheel contact")
        _json_exact_keys(
            contact,
            {
                "measurement",
                "evidence_available",
                "support_qualified",
                "normal_load_n",
                "tangential_load_n",
                "friction_utilization",
                "errors",
            },
            "validated wheel contact",
        )
        parsed.append(
            ValidatedWheelContact(
                measurement=_measurement_from_mapping(contact["measurement"]),
                evidence_available=_json_bool(contact["evidence_available"], "evidence_available"),
                support_qualified=_json_bool(contact["support_qualified"], "support_qualified"),
                normal_load_n=_json_optional_float(contact["normal_load_n"], "normal_load_n"),
                tangential_load_n=_json_optional_float(contact["tangential_load_n"], "tangential_load_n"),
                friction_utilization=_json_optional_float(
                    contact["friction_utilization"], "friction_utilization"
                ),
                errors=_json_strings(contact["errors"], "validated contact errors"),
            )
        )
    result = WheelContactFrame(
        available=_json_bool(row["available"], "wheel contacts available"),
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        contacts=tuple(parsed),
        thresholds=thresholds,
        errors=_json_strings(row["errors"], "wheel contacts errors"),
    )
    recomputed = validate_wheel_contact_frame(
        [item.measurement for item in parsed],
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        thresholds=thresholds,
    )
    if recomputed != result:
        raise ValueError("serialized wheel contact validation fields are not reproducible")
    return result


def _support_from_mapping(value: Any) -> SupportRegionAssessment:
    row = _json_mapping(value, "support_region")
    keys = {
        "status",
        "model",
        "physics_tick",
        "support_legs",
        "contact_points_w_m",
        "hull_xy_m",
        "signed_margin_m",
        "diagonal",
        "line_parameter",
        "line_distance_m",
        "between_contacts",
        "finite_patch_approximation",
        "corridor_half_width_m",
        "corridor_signed_margin_m",
        "reasons",
    }
    _json_exact_keys(row, keys, "support_region")
    try:
        model = SupportModel(_json_text(row["model"], "support model"))
    except ValueError as exc:
        raise ValueError("support model is invalid") from exc
    points = tuple(
        _json_optional_vec(item, 3, "support contact point")
        for item in _json_list(row["contact_points_w_m"], "support contact points")
    )
    hull = tuple(
        _json_optional_vec(item, 2, "support hull point")
        for item in _json_list(row["hull_xy_m"], "support hull")
    )
    if any(item is None for item in points) or any(item is None for item in hull):
        raise ValueError("support points cannot be null")
    diagonal = row["diagonal"]
    if diagonal is not None and diagonal not in {"FL_RR", "FR_RL"}:
        raise ValueError("support diagonal is invalid")
    return SupportRegionAssessment(
        status=_json_status(row["status"], "support status"),
        model=model,
        physics_tick=_json_optional_int(row["physics_tick"], "support physics_tick"),
        support_legs=_json_strings(row["support_legs"], "support legs"),
        contact_points_w_m=tuple(item for item in points if item is not None),
        hull_xy_m=tuple(item for item in hull if item is not None),
        signed_margin_m=_json_optional_float(row["signed_margin_m"], "signed_margin_m"),
        diagonal=diagonal,
        line_parameter=_json_optional_float(row["line_parameter"], "line_parameter"),
        line_distance_m=_json_optional_float(row["line_distance_m"], "line_distance_m"),
        between_contacts=_json_optional_bool(row["between_contacts"], "between_contacts"),
        finite_patch_approximation=_json_bool(
            row["finite_patch_approximation"], "finite_patch_approximation"
        ),
        corridor_half_width_m=_json_optional_float(
            row["corridor_half_width_m"], "corridor_half_width_m"
        ),
        corridor_signed_margin_m=_json_optional_float(
            row["corridor_signed_margin_m"], "corridor_signed_margin_m"
        ),
        reasons=_json_strings(row["reasons"], "support reasons"),
    )


def _wrench_from_mapping(value: Any) -> ContactWrenchFeasibility:
    row = _json_mapping(value, "contact_wrench_feasibility")
    keys = {
        "status",
        "proven_feasible",
        "physics_tick",
        "witness_kind",
        "multiple_contact_heights",
        "dynamic",
        "angular_momentum_rate_w_nm",
        "angular_momentum_rate_source",
        "force_residual_w_n",
        "force_residual_norm_n",
        "moment_residual_w_nm",
        "moment_residual_norm_nm",
        "maximum_friction_utilization",
        "reasons",
    }
    _json_exact_keys(row, keys, "contact_wrench_feasibility")
    result = ContactWrenchFeasibility(
        status=_json_status(row["status"], "wrench status"),
        proven_feasible=_json_bool(row["proven_feasible"], "proven_feasible"),
        physics_tick=_json_optional_int(row["physics_tick"], "wrench physics_tick"),
        witness_kind=_json_text(row["witness_kind"], "witness_kind"),
        multiple_contact_heights=_json_optional_bool(
            row["multiple_contact_heights"], "multiple_contact_heights"
        ),
        dynamic=_json_optional_bool(row["dynamic"], "dynamic"),
        angular_momentum_rate_w_nm=_json_optional_vec(
            row["angular_momentum_rate_w_nm"],
            3,
            "angular_momentum_rate_w_nm",
        ),
        angular_momentum_rate_source=_json_text(
            row["angular_momentum_rate_source"],
            "angular_momentum_rate_source",
        ),
        force_residual_w_n=_json_optional_vec(row["force_residual_w_n"], 3, "force_residual_w_n"),
        force_residual_norm_n=_json_optional_float(
            row["force_residual_norm_n"], "force_residual_norm_n"
        ),
        moment_residual_w_nm=_json_optional_vec(
            row["moment_residual_w_nm"], 3, "moment_residual_w_nm"
        ),
        moment_residual_norm_nm=_json_optional_float(
            row["moment_residual_norm_nm"], "moment_residual_norm_nm"
        ),
        maximum_friction_utilization=_json_optional_float(
            row["maximum_friction_utilization"], "maximum_friction_utilization"
        ),
        reasons=_json_strings(row["reasons"], "wrench reasons"),
    )
    if result.proven_feasible != (result.status == EvidenceStatus.PROVEN):
        raise ValueError("wrench status/proven_feasible are inconsistent")
    if result.proven_feasible:
        if result.physics_tick is None or not result.angular_momentum_rate_source:
            raise ValueError("proven wrench omits its current independent source")
        if any(
            item is None
            for item in (
                result.multiple_contact_heights,
                result.dynamic,
                result.angular_momentum_rate_w_nm,
                result.force_residual_w_n,
                result.force_residual_norm_n,
                result.moment_residual_w_nm,
                result.moment_residual_norm_nm,
                result.maximum_friction_utilization,
            )
        ) or result.reasons:
            raise ValueError("proven wrench mapping is incomplete or contradictory")
    elif not result.reasons:
        raise ValueError("unproven wrench mapping must explain the missing proof")
    return result


def _diagonal_from_mapping(
    value: Any, *, support_region: SupportRegionAssessment
) -> DiagonalSupportEvidence:
    row = _json_mapping(value, "diagonal_support")
    keys = {
        "status",
        "classification",
        "physics_tick",
        "primary_diagonal",
        "load_fl_rr_n",
        "load_fr_rl_n",
        "load_ratio_fl_rr",
        "load_ratio_fr_rl",
        "active_swing_leg",
        "active_swing_leg_load_n",
        "wrench_feasibility_proven",
        "reasons",
    }
    _json_exact_keys(row, keys, "diagonal_support")
    classification = _json_text(row["classification"], "diagonal classification")
    if classification not in {"PRIMARY_DIAGONAL_SUPPORT", "NOT_PROVEN"}:
        raise ValueError("diagonal classification is invalid")
    primary = row["primary_diagonal"]
    if primary is not None and primary not in {"FL_RR", "FR_RL"}:
        raise ValueError("primary_diagonal is invalid")
    active = row["active_swing_leg"]
    if active is not None and active not in LEGS:
        raise ValueError("active_swing_leg is invalid")
    status = _json_status(row["status"], "diagonal status")
    reasons = _json_strings(row["reasons"], "diagonal reasons")
    if (classification == "PRIMARY_DIAGONAL_SUPPORT") != (
        status == EvidenceStatus.PROVEN
    ):
        raise ValueError("diagonal classification/status are inconsistent")
    if status == EvidenceStatus.PROVEN and reasons:
        raise ValueError("proven diagonal support carries failure reasons")
    if status != EvidenceStatus.PROVEN and not reasons:
        raise ValueError("unproven diagonal support must explain the missing proof")
    return DiagonalSupportEvidence(
        status=status,
        classification=classification,
        physics_tick=_json_optional_int(row["physics_tick"], "diagonal physics_tick"),
        primary_diagonal=primary,
        load_fl_rr_n=_json_optional_float(row["load_fl_rr_n"], "load_fl_rr_n"),
        load_fr_rl_n=_json_optional_float(row["load_fr_rl_n"], "load_fr_rl_n"),
        load_ratio_fl_rr=_json_optional_float(row["load_ratio_fl_rr"], "load_ratio_fl_rr"),
        load_ratio_fr_rl=_json_optional_float(row["load_ratio_fr_rl"], "load_ratio_fr_rl"),
        active_swing_leg=active,
        active_swing_leg_load_n=_json_optional_float(
            row["active_swing_leg_load_n"], "active_swing_leg_load_n"
        ),
        support_region=support_region,
        wrench_feasibility_proven=_json_bool(
            row["wrench_feasibility_proven"], "wrench_feasibility_proven"
        ),
        reasons=reasons,
    )


def measure_isaac_wheel_contacts(
    adapter: Any,
    sensor_bank: Any,
    *,
    env_id: int = 0,
    surface_kind_by_leg: Mapping[str, str],
    surface_height_m_by_leg: Mapping[str, float],
    friction_coefficient_by_leg: Mapping[str, float],
    surface_dwell_s_by_leg: Mapping[str, float] | None = None,
    surface_dwell_kind_by_leg: Mapping[str, str] | None = None,
    previous_frame: WheelContactFrame | None = None,
    finite_patch_radius_m_by_leg: Mapping[str, float] | None = None,
    whole_body_com: WholeBodyCOMMeasurement | None = None,
    contact_moment_model: str = "MEASURED",
    thresholds: SupportThresholds = SupportThresholds(),
) -> WheelContactFrame:
    """Lazily read exact filtered wheel contacts from an existing bank.

    ``ContactSensor.current_contact_time`` is body-wide and cannot prove that
    the selected ground/top surface remained unchanged.  Therefore support
    qualification requires caller-supplied per-surface dwell plus the exact
    surface identity tracked for that dwell.  A front-face to top transition
    therefore cannot inherit the obstacle aggregate's contact time.  Without
    both mappings the contact remains observable but
    ``surface_dwell_verified`` is false.
    Slip is computed from rigid-body COM linear/angular velocity at the
    measured contact point.  In the default ``MEASURED`` mode, each child
    sensor's raw PhysX normal/friction force and application-point buffers are
    strictly reconciled with the aggregate sensor fields.  Raw net force and
    moment are summed about the current whole-body COM; the stored contact
    moment is the equivalent couple at the raw mean contact point.  The
    explicit ``POINT_CONTACT_ZERO_CONSERVATIVE`` mode supplies an exact zero
    only as a diagnostic; it can never prove wrench feasibility.
    """

    tick, time_s, dt = _strict_time(
        getattr(adapter, "sim_steps"),
        getattr(adapter, "sim_time"),
        adapter.sim.get_physics_dt(),
    )
    env_index = _exact_int(env_id, "env_id")
    if (
        type(contact_moment_model) is not str
        or contact_moment_model not in CONTACT_MOMENT_MODELS
    ):
        raise ValueError("contact_moment_model is invalid")
    force_threshold = _finite_float(
        getattr(sensor_bank, "force_threshold_n", None),
        "sensor_bank.force_threshold_n",
        minimum=0.0,
    )
    if (
        set(surface_kind_by_leg) != set(LEGS)
        or set(surface_height_m_by_leg) != set(LEGS)
        or set(friction_coefficient_by_leg) != set(LEGS)
    ):
        raise ValueError("surface/friction mappings must have exact FL/FR/RL/RR keys")
    if surface_dwell_s_by_leg is not None and set(surface_dwell_s_by_leg) != set(LEGS):
        raise ValueError("surface_dwell_s_by_leg keys are not exact")
    if surface_dwell_kind_by_leg is not None and set(surface_dwell_kind_by_leg) != set(LEGS):
        raise ValueError("surface_dwell_kind_by_leg keys are not exact")
    if (surface_dwell_s_by_leg is None) != (surface_dwell_kind_by_leg is None):
        raise ValueError("surface dwell seconds and surface identity must be supplied together")
    if finite_patch_radius_m_by_leg is not None and set(finite_patch_radius_m_by_leg) != set(LEGS):
        raise ValueError("finite_patch_radius_m_by_leg keys are not exact")
    specs = tuple(getattr(sensor_bank, "specs"))
    if tuple(spec.leg for spec in specs) != LEGS:
        raise ValueError("filtered sensor bank leg order is not exact FL/FR/RL/RR")
    if tuple(spec.body_name for spec in specs) != tuple(LEG_TO_WHEEL_BODY[leg] for leg in LEGS):
        raise ValueError("filtered sensor bank wheel body order is invalid")
    if any(
        type(getattr(spec, "prim_path", None)) is not str
        or not spec.prim_path.startswith("/")
        for spec in specs
    ):
        raise ValueError("filtered sensor bank wheel prim identities are invalid")
    filters = tuple(getattr(sensor_bank, "filter_surfaces"))
    if tuple(name for name, _path in filters) != ("ground", "obstacle"):
        raise ValueError("filtered sensor surface order is not exact ground/obstacle")
    expected_filter_paths = tuple(path for _name, path in filters)

    robot = adapter.robot
    data = robot.data
    body_names = tuple(robot.body_names)
    body_index = {name: index for index, name in enumerate(body_names)}
    whole_body_com_position = None
    if contact_moment_model == "MEASURED":
        if (
            whole_body_com is None
            or not whole_body_com.available
            or whole_body_com.position_w_m is None
        ):
            raise ValueError("current whole-body COM is required for measured contact moments")
        if (
            whole_body_com.physics_tick != tick
            or tuple(whole_body_com.body_names) != body_names
            or not math.isclose(
                _finite_float(whole_body_com.sim_time_s, "whole_body_com.sim_time_s", minimum=0.0),
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
            or not math.isclose(
                _finite_float(
                    whole_body_com.physics_dt_s,
                    "whole_body_com.physics_dt_s",
                    minimum=0.0,
                ),
                dt,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
        ):
            raise ValueError("whole-body COM is not current or body-identity-bound")
        whole_body_com_position = _vec3(
            whole_body_com.position_w_m,
            "whole_body_com.position_w_m",
        )
    data_timestamp = _finite_float(
        getattr(data, "_sim_timestamp", None), "ArticulationData._sim_timestamp", minimum=0.0
    )
    if not math.isclose(
        data_timestamp,
        time_s,
        rel_tol=0.0,
        abs_tol=max(1.0e-9, dt * 1.0e-6),
    ):
        raise ValueError("ArticulationData is stale for wheel slip measurement")
    com_pos_all = data.body_com_pos_w
    com_vel_all = data.body_com_vel_w
    for buffer_name in ("_body_com_pose_w", "_body_com_vel_w"):
        if not math.isclose(
            _buffer_timestamp(data, buffer_name),
            data_timestamp,
            rel_tol=0.0,
            abs_tol=max(1.0e-9, dt * 1.0e-6),
        ):
            raise ValueError(f"{buffer_name} is stale for wheel slip measurement")
    previous_by_leg = previous_frame.by_leg() if previous_frame is not None else {}
    rows: list[WheelContactMeasurement] = []
    for spec in specs:
        leg = spec.leg
        sensor = sensor_bank.sensors[leg]
        sensor_cfg = getattr(sensor, "cfg", None)
        if getattr(sensor_cfg, "prim_path", None) != getattr(spec, "prim_path", None):
            raise ValueError(f"{leg} child sensor prim identity does not match its spec")
        configured_filter_paths = getattr(
            sensor_cfg,
            "filter_prim_paths_expr",
            None,
        )
        if isinstance(configured_filter_paths, (str, bytes)) or tuple(
            configured_filter_paths or ()
        ) != expected_filter_paths:
            raise ValueError(
                f"{leg} child sensor filter identity/order does not match the bank"
            )
        snapshot = sensor.data
        sensor_clock = _slice_env(
            getattr(sensor, "_timestamp", None), env_index, (), f"{leg} sensor clock"
        )
        sensor_time = _slice_env(
            getattr(sensor, "_timestamp_last_update", None),
            env_index,
            (),
            f"{leg} sensor timestamp",
        )
        if not math.isclose(
            float(sensor_clock),
            time_s,
            rel_tol=0.0,
            abs_tol=max(1.0e-9, dt * 1.0e-6),
        ) or not math.isclose(
            float(sensor_time),
            time_s,
            rel_tol=0.0,
            abs_tol=max(1.0e-9, dt * 1.0e-6),
        ):
            raise ValueError(f"{leg} contact sensor is stale")
        surface = str(surface_kind_by_leg[leg])
        if surface not in ALL_SURFACES:
            raise ValueError(f"{leg} surface_kind is invalid")
        if surface == "GROUND":
            filter_index = 0
        elif surface in {"OBSTACLE_TOP", "FRONT_FACE"}:
            filter_index = 1
        else:
            filter_index = None
        force_matrix_raw = _tensor_float_array(
            snapshot.force_matrix_w,
            f"{leg} force_matrix_w",
        )
        points_raw = _tensor_float_array(snapshot.contact_pos_w, f"{leg} contact_pos_w")
        friction_raw = _tensor_float_array(
            snapshot.friction_forces_w,
            f"{leg} friction_forces_w",
        )
        if (
            force_matrix_raw.ndim != 4
            or force_matrix_raw.shape[1:] != (1, 2, 3)
            or points_raw.shape != force_matrix_raw.shape
            or friction_raw.shape != force_matrix_raw.shape
        ):
            raise ValueError(f"{leg} aggregate contact layout is invalid")
        aggregate_sensor_count = force_matrix_raw.shape[0]
        if not 0 <= env_index < aggregate_sensor_count:
            raise ValueError(f"env_id={env_index} is out of range for {leg} sensor")
        forces = np.asarray(force_matrix_raw[env_index], dtype=float)
        measured_contact_moment = None
        raw_pairs = None
        if contact_moment_model == "MEASURED":
            if whole_body_com_position is None:
                raise ValueError("whole-body COM position is unavailable")
            raw_pairs = _measure_raw_contact_moment_at_reference(
                sensor,
                leg=leg,
                env_id=env_index,
                physics_dt_s=dt,
                aggregate_sensor_count=aggregate_sensor_count,
                aggregate_normal_forces_w_n=forces[0],
                aggregate_friction_forces_w_n=friction_raw[env_index, 0],
                aggregate_contact_points_w_m=points_raw[env_index, 0],
                whole_body_com_position_w_m=whole_body_com_position,
            )
            raw_normal_norms = tuple(
                float(np.linalg.norm(pair[0])) for pair in raw_pairs
            )
            if surface == "AIR" and any(
                normal_norm > force_threshold for normal_norm in raw_normal_norms
            ):
                raise ValueError(f"{leg} AIR classification has above-threshold raw contact")
            if filter_index is not None and any(
                pair_index != filter_index and normal_norm > force_threshold
                for pair_index, normal_norm in enumerate(raw_normal_norms)
            ):
                raise ValueError(
                    f"{leg} above-threshold raw contact exists on a non-selected surface"
                )
        if filter_index is None:
            normal_force = np.zeros(3, dtype=float)
            friction_force = np.zeros(3, dtype=float)
            point = None
            active = False
            slip = None
            drift = None
        else:
            aggregate_normal_force = np.asarray(forces[0, filter_index], dtype=float)
            aggregate_friction_force = np.asarray(
                friction_raw[env_index, 0, filter_index],
                dtype=float,
            )
            aggregate_point = np.asarray(
                points_raw[env_index, 0, filter_index],
                dtype=float,
            )
            if not np.isfinite(aggregate_normal_force).all() or not np.isfinite(
                aggregate_friction_force
            ).all():
                raise ValueError(f"{leg} selected aggregate force is non-finite")
            if contact_moment_model == "MEASURED":
                if raw_pairs is None:
                    raise ValueError(f"{leg} raw contact pair is unavailable")
                (
                    normal_force,
                    friction_force,
                    raw_mean_point,
                    measured_contact_moment,
                ) = raw_pairs[filter_index]
                point_value = raw_mean_point
            else:
                normal_force = aggregate_normal_force
                friction_force = aggregate_friction_force
                point_value = aggregate_point
            active = bool(
                np.linalg.norm(normal_force) > force_threshold
            )
            point = (
                point_value
                if active
                and point_value is not None
                and np.isfinite(point_value).all()
                else None
            )
            if active and point is None:
                raise ValueError(f"{leg} active filtered contact point is unavailable")
            if (
                active
                and contact_moment_model == "MEASURED"
                and measured_contact_moment is None
            ):
                raise ValueError(f"{leg} active aggregate has no raw contact rows")
            if active:
                index = body_index.get(spec.body_name)
                if index is None:
                    raise ValueError(f"{leg} wheel body is absent from articulation body_names")
                body_pos = _slice_env(com_pos_all, env_index, (len(body_names), 3), "body_com_pos_w")[index]
                body_vel = _slice_env(com_vel_all, env_index, (len(body_names), 6), "body_com_vel_w")[index]
                contact_velocity = body_vel[:3] + np.cross(body_vel[3:6], point - body_pos)
                normal = np.asarray((0.0, 0.0, 1.0), dtype=float)
                slip = float(np.linalg.norm(contact_velocity - np.dot(contact_velocity, normal) * normal))
                previous = previous_by_leg.get(leg)
                if (
                    previous is not None
                    and previous.measurement.contact_point_w_m is not None
                    and previous.measurement.physics_tick == tick - 1
                ):
                    drift = float(
                        np.linalg.norm(point - np.asarray(previous.measurement.contact_point_w_m, dtype=float)) / dt
                    )
                else:
                    drift = None
            else:
                slip = None
                drift = None
        sensor_clock_after = _slice_env(
            getattr(sensor, "_timestamp", None),
            env_index,
            (),
            f"{leg} sensor clock after raw read",
        )
        sensor_time_after = _slice_env(
            getattr(sensor, "_timestamp_last_update", None),
            env_index,
            (),
            f"{leg} sensor timestamp after raw read",
        )
        if (
            float(sensor_clock_after) != float(sensor_clock)
            or float(sensor_time_after) != float(sensor_time)
            or not math.isclose(
                float(sensor_clock_after),
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
            or not math.isclose(
                float(sensor_time_after),
                time_s,
                rel_tol=0.0,
                abs_tol=max(1.0e-9, dt * 1.0e-6),
            )
        ):
            raise ValueError(f"{leg} contact sensor changed tick during raw read")
        if active:
            if contact_moment_model == "MEASURED":
                row_contact_moment = tuple(
                    float(value) for value in measured_contact_moment
                )
            else:
                row_contact_moment = (0.0, 0.0, 0.0)
            row_contact_moment_model = contact_moment_model
        else:
            row_contact_moment = None
            row_contact_moment_model = ""
        rows.append(
            WheelContactMeasurement(
                leg=leg,
                wheel_body_name=spec.body_name,
                physics_tick=tick,
                sim_time_s=time_s,
                surface_kind=surface,
                surface_height_m=float(surface_height_m_by_leg[leg]) if surface in SUPPORT_SURFACES else None,
                surface_normal_w=(0.0, 0.0, 1.0) if surface in SUPPORT_SURFACES else None,
                active=active,
                contact_point_w_m=tuple(float(value) for value in point) if point is not None else None,
                normal_force_w_n=tuple(float(value) for value in normal_force),
                friction_force_w_n=tuple(float(value) for value in friction_force),
                contact_moment_w_nm=row_contact_moment,
                contact_moment_model=row_contact_moment_model,
                dwell_s=(float(surface_dwell_s_by_leg[leg]) if surface_dwell_s_by_leg is not None else None),
                surface_dwell_verified=bool(
                    surface_dwell_s_by_leg is not None
                    and surface_dwell_kind_by_leg[leg] == surface
                ),
                slip_speed_m_s=slip,
                contact_drift_speed_m_s=drift,
                friction_coefficient=float(friction_coefficient_by_leg[leg]),
                finite_patch_radius_m=(
                    float(finite_patch_radius_m_by_leg[leg])
                    if finite_patch_radius_m_by_leg is not None
                    else None
                ),
                source=(
                    "isaaclab.RigidContactView.raw_contact_and_friction_wrench.current_tick"
                    if contact_moment_model == "MEASURED"
                    else "isaaclab.FilteredWheelContactSensorBank.point_contact_zero.current_tick"
                ),
            )
        )
    return validate_wheel_contact_frame(
        rows,
        physics_tick=tick,
        sim_time_s=time_s,
        physics_dt_s=dt,
        thresholds=thresholds,
    )


__all__ = [
    "ALL_SURFACES",
    "ANGULAR_MOMENTUM_FIELD_NAMES",
    "CENTROIDAL_SUPPORT_EVIDENCE_SCHEMA",
    "COM_FIELD_NAMES",
    "CONTACT_MOMENT_MODELS",
    "DIAGONALS",
    "EvidenceStatus",
    "LEG_TO_WHEEL_BODY",
    "LEGS",
    "RAW_CONTACT_FORCE_AGGREGATE_ATOL_N",
    "RAW_CONTACT_NORMAL_UNIT_TOLERANCE",
    "RAW_CONTACT_POINT_AGGREGATE_ATOL_M",
    "SUPPORT_SURFACES",
    "ContactWrenchFeasibility",
    "CentroidalAngularMomentumRateMeasurement",
    "CentroidalSupportEvidence",
    "DiagonalSupportEvidence",
    "SupportModel",
    "SupportRegionAssessment",
    "SupportThresholds",
    "TransferMethod",
    "TransferMethodEvidence",
    "TransferSample",
    "TransferThresholds",
    "ValidatedWheelContact",
    "WheelContactFrame",
    "WheelContactMeasurement",
    "WholeBodyCOMMeasurement",
    "assess_contact_wrench_feasibility",
    "assess_primary_diagonal_support",
    "assess_support_region",
    "classify_com_transfer",
    "measure_isaac_whole_body_com",
    "measure_isaac_centroidal_angular_momentum_rate",
    "measure_isaac_wheel_contacts",
    "measure_centroidal_angular_momentum_rate",
    "measure_whole_body_com",
    "unavailable_centroidal_angular_momentum_rate",
    "unavailable_whole_body_com",
    "validate_wheel_contact_frame",
]
