"""Contact and wheel telemetry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from command_model import WHEEL_JOINT_NAMES
from .com_metrics import to_numpy


@dataclass
class ContactMetricsResult:
    contacts: list[dict[str, Any]] = field(default_factory=list)
    support_points_w: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    support_normals_w: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=float))
    normal_forces_n: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    friction_coefficients: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    max_contact_force_n: float = 0.0
    max_friction_utilization: float = float("nan")
    max_wheel_slip_ratio: float = float("nan")
    contact_geometry_source: str = "unavailable"
    contact_force_source: str = "unavailable"
    contact_force_valid: bool = False
    contact_force_reason: str = "contact force unavailable"
    friction_utilization_valid: bool = False
    friction_utilization_reason: str = "friction utilization requires valid contact forces"
    wheel_slip_state: str = "unavailable"
    wheel_slip_source: str = "unavailable"
    warnings: list[str] = field(default_factory=list)


def compute_contact_metrics(
    *,
    adapter: Any,
    scene_handle: Any,
    contact_sensor: Any | None,
    env_id: int,
    simulation_time_s: float,
    force_threshold_n: float,
    friction_default: float,
    wheel_radius_m: float | None = None,
    slip_warning_threshold: float = 0.2,
) -> ContactMetricsResult:
    sensor_result = _contacts_from_sensor(
        contact_sensor=contact_sensor,
        adapter=adapter,
        env_id=env_id,
        simulation_time_s=simulation_time_s,
        force_threshold_n=force_threshold_n,
        friction_default=friction_default,
        wheel_radius_m=wheel_radius_m,
        slip_warning_threshold=slip_warning_threshold,
    )
    if sensor_result.contacts:
        return sensor_result
    return _contacts_from_wheel_geometry(
        adapter=adapter,
        scene_handle=scene_handle,
        simulation_time_s=simulation_time_s,
        force_threshold_n=force_threshold_n,
        friction_default=friction_default,
        wheel_radius_m=wheel_radius_m,
        slip_warning_threshold=slip_warning_threshold,
        warnings=sensor_result.warnings,
    )


def _contacts_from_sensor(
    *,
    contact_sensor: Any | None,
    adapter: Any,
    env_id: int,
    simulation_time_s: float,
    force_threshold_n: float,
    friction_default: float,
    wheel_radius_m: float | None,
    slip_warning_threshold: float,
) -> ContactMetricsResult:
    if contact_sensor is None:
        return ContactMetricsResult(warnings=["contact sensor unavailable"])
    try:
        sim = getattr(adapter, "sim", None)
        dt = float(sim.get_physics_dt()) if sim is not None and hasattr(sim, "get_physics_dt") else 0.0
        contact_sensor.update(dt, force_recompute=True)
    except Exception as exc:
        return ContactMetricsResult(warnings=[f"contact sensor update failed: {exc}"])
    data = getattr(contact_sensor, "data", None)
    net_forces = to_numpy(getattr(data, "net_forces_w", None))
    if net_forces.size == 0:
        return ContactMetricsResult(warnings=["contact sensor net_forces_w unavailable"])
    if net_forces.ndim == 3:
        forces = net_forces[min(max(0, env_id), net_forces.shape[0] - 1)]
    else:
        forces = net_forces.reshape(-1, 3)
    body_names = list(getattr(contact_sensor, "body_names", []) or getattr(getattr(adapter, "robot", None), "body_names", []) or [])
    body_positions = _body_positions(adapter, env_id)
    points: list[list[float]] = []
    normals: list[list[float]] = []
    magnitudes: list[float] = []
    contacts: list[dict[str, Any]] = []
    max_force = 0.0
    for index, force in enumerate(forces):
        if force.shape[0] < 3 or not np.isfinite(force[:3]).all():
            continue
        magnitude = float(np.linalg.norm(force[:3]))
        if magnitude < float(force_threshold_n):
            continue
        normal_force = max(0.0, float(force[2]))
        normal = np.asarray(force[:3], dtype=float) / max(magnitude, 1.0e-12)
        body_name = str(body_names[index]) if index < len(body_names) else f"body_{index}"
        point = _approx_contact_point_for_body(body_name, body_positions.get(body_name), wheel_radius_m)
        slip, slip_state = _wheel_slip_ratio(adapter, body_name, wheel_radius_m)
        points.append(point.tolist())
        normals.append(normal.tolist())
        magnitudes.append(normal_force if normal_force > 0.0 else magnitude)
        max_force = max(max_force, magnitude)
        contacts.append(
            {
                "time_s": float(simulation_time_s),
                "body_name": body_name,
                "other_body": "ground_or_obstacle",
                "active": True,
                "contact_point_w": point.tolist(),
                "contact_normal_w": normal.tolist(),
                "normal_force_n": normal_force if normal_force > 0.0 else magnitude,
                "tangential_force_n": 0.0,
                "total_force_n": magnitude,
                "friction_coefficient": float(friction_default),
                "friction_utilization": float("nan"),
                "slip_ratio": slip,
                "slip_state": slip_state,
                "slip_warning": False,
                "impact_warning": False,
                "wheel_contact": _is_wheel_name(body_name),
                "source": "isaaclab.ContactSensor.net_forces_w",
                "geometry_source": "wheel_geometry_approximation",
            }
        )
    return ContactMetricsResult(
        contacts=contacts,
        support_points_w=np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float),
        support_normals_w=np.asarray(normals, dtype=float) if normals else np.empty((0, 3), dtype=float),
        normal_forces_n=np.asarray(magnitudes, dtype=float),
        friction_coefficients=np.full(len(points), float(friction_default), dtype=float),
        max_contact_force_n=max_force,
        contact_geometry_source="sensor_force_with_wheel_geometry_approximation",
        contact_force_source="isaaclab.ContactSensor.net_forces_w",
        contact_force_valid=bool(contacts),
        contact_force_reason="" if contacts else "contact sensor reported no active contacts",
        friction_utilization_valid=False,
        friction_utilization_reason="tangential contact forces unavailable from net_forces_w",
        wheel_slip_state=_aggregate_slip_state(contacts),
        wheel_slip_source="joint velocity vs base velocity",
        warnings=[] if contacts else ["contact sensor reported no active contacts"],
    )


def _contacts_from_wheel_geometry(
    *,
    adapter: Any,
    scene_handle: Any,
    simulation_time_s: float,
    force_threshold_n: float,
    friction_default: float,
    wheel_radius_m: float | None,
    slip_warning_threshold: float,
    warnings: list[str],
) -> ContactMetricsResult:
    ground_z = _ground_z(scene_handle)
    positions = _body_positions(adapter, 0)
    radius = float(wheel_radius_m if wheel_radius_m is not None and wheel_radius_m > 0.0 else 0.05)
    points: list[list[float]] = []
    normals: list[list[float]] = []
    normal_forces: list[float] = []
    contacts: list[dict[str, Any]] = []
    for wheel_name in WHEEL_JOINT_NAMES:
        body_pos = positions.get(_wheel_body_name_guess(wheel_name), positions.get(wheel_name))
        if body_pos is None:
            continue
        center = np.asarray(body_pos, dtype=float)
        point = np.asarray([center[0], center[1], ground_z], dtype=float)
        active = bool(center[2] - radius <= ground_z + 0.01)
        if not active:
            continue
        slip, slip_state = _wheel_slip_ratio(adapter, wheel_name, radius)
        normal_force = float("nan")
        points.append(point.tolist())
        normals.append([0.0, 0.0, 1.0])
        normal_forces.append(force_threshold_n)
        contacts.append(
            {
                "time_s": float(simulation_time_s),
                "body_name": wheel_name,
                "other_body": "ground_plane",
                "active": True,
                "contact_point_w": point.tolist(),
                "contact_normal_w": [0.0, 0.0, 1.0],
                "normal_force_n": normal_force,
                "tangential_force_n": float("nan"),
                "total_force_n": normal_force,
                "friction_coefficient": float(friction_default),
                "friction_utilization": float("nan"),
                "slip_ratio": slip,
                "slip_state": slip_state,
                "slip_warning": bool(math.isfinite(slip) and slip >= float(slip_warning_threshold)),
                "impact_warning": False,
                "wheel_contact": True,
                "source": "wheel_geometry_approximation",
                "geometry_source": "wheel_geometry_approximation",
            }
        )
    return ContactMetricsResult(
        contacts=contacts,
        support_points_w=np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float),
        support_normals_w=np.asarray(normals, dtype=float) if normals else np.empty((0, 3), dtype=float),
        normal_forces_n=np.asarray(normal_forces, dtype=float),
        friction_coefficients=np.full(len(points), float(friction_default), dtype=float),
        max_contact_force_n=float("nan"),
        max_friction_utilization=float("nan"),
        max_wheel_slip_ratio=max([row["slip_ratio"] for row in contacts if math.isfinite(float(row["slip_ratio"]))] + [float("nan")]),
        contact_geometry_source="wheel_geometry_approximation",
        contact_force_source="disabled",
        contact_force_valid=False,
        contact_force_reason="contact sensor disabled or unavailable; support contacts are geometry-estimated",
        friction_utilization_valid=False,
        friction_utilization_reason="friction utilization requires valid normal and tangential contact forces",
        wheel_slip_state=_aggregate_slip_state(contacts),
        wheel_slip_source="joint velocity vs base velocity with idle deadband",
        warnings=list(warnings) + ["contact forces unavailable; wheel support points are approximate"] if contacts else list(warnings),
    )


def _body_positions(adapter: Any, env_id: int) -> dict[str, np.ndarray]:
    robot = getattr(adapter, "robot", None)
    names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
    states = to_numpy(getattr(getattr(robot, "data", None), "body_link_state_w", None))
    if states.size == 0:
        states = to_numpy(getattr(getattr(robot, "data", None), "body_state_w", None))
    if states.ndim == 3:
        states = states[min(max(0, env_id), states.shape[0] - 1)]
    result: dict[str, np.ndarray] = {}
    if states.ndim == 2 and states.shape[-1] >= 3:
        for index, name in enumerate(names[: states.shape[0]]):
            result[name] = np.asarray(states[index, :3], dtype=float)
    return result


def _approx_contact_point_for_body(body_name: str, body_position: np.ndarray | None, wheel_radius_m: float | None) -> np.ndarray:
    if body_position is None:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)
    point = np.asarray(body_position, dtype=float).copy()
    if _is_wheel_name(body_name):
        point[2] -= float(wheel_radius_m if wheel_radius_m is not None and wheel_radius_m > 0.0 else 0.05)
    return point


def _wheel_slip_ratio(adapter: Any, name: str, wheel_radius_m: float | None) -> tuple[float, str]:
    radius = float(wheel_radius_m if wheel_radius_m is not None and wheel_radius_m > 0.0 else 0.05)
    joint_name = name if name in WHEEL_JOINT_NAMES else _wheel_joint_for_body(name)
    try:
        joint_id = getattr(adapter, "wheel_name_to_id", {}).get(joint_name)
        omega = float(adapter.robot.data.joint_vel[0, joint_id].item()) if joint_id is not None else float("nan")
    except Exception:
        omega = float("nan")
    try:
        root_v = to_numpy(adapter.robot.data.root_vel_w)[0, :3]
        tangent_speed = float(np.linalg.norm(root_v[:2]))
    except Exception:
        tangent_speed = float("nan")
    surface_speed = abs(omega * radius) if math.isfinite(omega) else float("nan")
    if not math.isfinite(surface_speed) or not math.isfinite(tangent_speed):
        return float("nan"), "unavailable"
    idle_deadband_m_s = 0.01
    if abs(tangent_speed) < idle_deadband_m_s and abs(surface_speed) < idle_deadband_m_s:
        return 0.0, "idle_deadband"
    denom = max(abs(tangent_speed), abs(surface_speed), 1.0e-6)
    return abs(tangent_speed - surface_speed) / denom, "moving"


def _aggregate_slip_state(contacts: list[dict[str, Any]]) -> str:
    states = {str(row.get("slip_state", "") or "") for row in contacts}
    if not states:
        return "unavailable"
    if "moving" in states:
        return "moving"
    if "idle_deadband" in states:
        return "idle_deadband"
    return sorted(states)[0] if states else "unavailable"


def _ground_z(scene_handle: Any) -> float:
    surface = getattr(scene_handle, "ground_surface_info", {}) or {}
    if isinstance(surface, dict):
        for key in ("actual_ground_z_m", "configured_ground_z_m"):
            try:
                value = surface.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                pass
    try:
        return float(getattr(getattr(scene_handle, "config", None), "ground_z_m", 0.0))
    except Exception:
        return 0.0


def _wheel_body_name_guess(wheel_name: str) -> str:
    return str(wheel_name).replace("_ankle", "_wheel")


def _wheel_joint_for_body(body_name: str) -> str:
    lower = str(body_name).lower()
    for name in WHEEL_JOINT_NAMES:
        prefix = name.replace("_ankle", "")
        if prefix in lower:
            return name
    return str(body_name)


def _is_wheel_name(name: str) -> bool:
    lower = str(name).lower()
    return "wheel" in lower or lower in set(WHEEL_JOINT_NAMES)
