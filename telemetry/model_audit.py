"""Runtime model audit for telemetry provenance."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from command_model import WHEEL_JOINT_NAMES

from .com_metrics import to_numpy
from .exporters import write_json


ISAACLAB_VERSION_FILE = Path("C:/robotics_sim/IsaacLab/VERSION")


def audit_robot_model(scene_handle: Any, adapter: Any | None = None) -> dict[str, Any]:
    robot = getattr(adapter, "robot", None) if adapter is not None else getattr(scene_handle, "robot", None)
    data = getattr(robot, "data", None)
    body_names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
    joint_names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
    mass_rows, mass_warnings = _mass_rows(data, body_names)
    joint_rows, joint_warnings = _joint_limit_rows(data, joint_names)
    usd = _usd_audit(scene_handle)
    contact_sensor = getattr(scene_handle, "contact_sensor", None)
    warnings = list(mass_warnings) + list(joint_warnings) + list(usd.get("warnings", []))
    if contact_sensor is None:
        warnings.append("ContactSensor unavailable; contact points may use wheel geometry approximation.")
    return {
        "schema_version": "height_replay_model_audit.v1",
        "isaac_lab_version": _read_text(ISAACLAB_VERSION_FILE),
        "robot_usd": str(getattr(getattr(scene_handle, "config", None), "robot_usd", "")),
        "robot_prim_path": "/World/WLRRobot",
        "body_count": len(body_names),
        "joint_count": len(joint_names),
        "body_names": body_names,
        "joint_names": joint_names,
        "masses": mass_rows,
        "joints": joint_rows,
        "wheel_joint_names": list(WHEEL_JOINT_NAMES),
        "contact_sensor": {
            "requested": bool(getattr(getattr(scene_handle, "config", None), "telemetry_contact_sensors_enabled", False)),
            "available": contact_sensor is not None,
            "error": str(getattr(scene_handle, "contact_sensor_error", "") or ""),
            "body_names": [str(name) for name in (getattr(contact_sensor, "body_names", []) or [])] if contact_sensor is not None else [],
        },
        "friction_sources": {
            "ground_static_friction": 1.25,
            "ground_dynamic_friction": 1.05,
            "obstacle_static_friction": 1.20,
            "obstacle_dynamic_friction": 1.00,
            "telemetry_default_mu": 1.05,
        },
        "usd": usd,
        "warnings": warnings,
    }


def write_model_audit(run_dir: str | Path, audit: dict[str, Any]) -> None:
    destination = Path(run_dir)
    write_json(destination / "model_audit.json", audit)
    (destination / "model_audit.txt").write_text(_audit_text(audit), encoding="utf-8")


def _mass_rows(data: Any, body_names: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    masses = to_numpy(getattr(data, "default_mass", None))
    if masses.ndim >= 2:
        masses = masses[0]
    masses = masses.reshape(-1) if masses.size else masses
    local_com = to_numpy(getattr(data, "body_com_pose_b", None))
    if local_com.ndim >= 3:
        local_com = local_com[0]
    rows: list[dict[str, Any]] = []
    if not masses.size:
        warnings.append("robot.data.default_mass unavailable")
    for index, name in enumerate(body_names):
        mass = _float(masses[index]) if index < masses.size else float("nan")
        row = {
            "body_index": index,
            "body_name": name,
            "mass_kg": mass,
            "local_com_xyz_m": _vec(local_com[index, :3]) if local_com.ndim == 2 and index < local_com.shape[0] and local_com.shape[1] >= 3 else [],
        }
        rows.append(row)
    return rows, warnings


def _joint_limit_rows(data: Any, joint_names: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    pos_limits = _limit_matrix(
        _first_available(
            data,
            "soft_joint_pos_limits",
            "joint_pos_limits",
            "default_joint_pos_limits",
        ),
        len(joint_names),
    )
    vel_limits = _limit_vector(
        _first_available(data, "soft_joint_vel_limits", "joint_vel_limits", "default_joint_vel_limits"),
        len(joint_names),
    )
    effort_limits = _limit_vector(
        _first_available(data, "soft_joint_effort_limits", "joint_effort_limits", "default_joint_effort_limits"),
        len(joint_names),
    )
    if not pos_limits:
        warnings.append("joint position limits unavailable from ArticulationData")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(joint_names):
        lower, upper = (pos_limits[index] if index < len(pos_limits) else [float("nan"), float("nan")])
        rows.append(
            {
                "joint_index": index,
                "joint_name": name,
                "position_limit_lower_rad": lower,
                "position_limit_upper_rad": upper,
                "velocity_limit_rad_s": vel_limits[index] if index < len(vel_limits) else float("nan"),
                "effort_limit_nm": effort_limits[index] if index < len(effort_limits) else float("nan"),
            }
        )
    return rows, warnings


def _usd_audit(scene_handle: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "stage_available": False,
        "collision_prim_count_under_robot": 0,
        "mass_api_prim_count_under_robot": 0,
        "joint_prim_count_under_robot": 0,
        "warnings": [],
    }
    try:
        from pxr import UsdPhysics  # type: ignore
    except Exception as exc:
        result["warnings"].append(f"pxr.UsdPhysics unavailable: {exc}")
        return result
    try:
        stage = _resolve_stage(scene_handle)
        if stage is None:
            result["warnings"].append("USD stage unavailable")
            return result
        result["stage_available"] = True
        collision_count = 0
        mass_count = 0
        joint_count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/WLRRobot"):
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_count += 1
            if prim.IsA(UsdPhysics.Joint):
                joint_count += 1
        result["collision_prim_count_under_robot"] = collision_count
        result["mass_api_prim_count_under_robot"] = mass_count
        result["joint_prim_count_under_robot"] = joint_count
    except Exception as exc:
        result["warnings"].append(f"USD audit failed: {exc}")
    return result


def _resolve_stage(scene_handle: Any) -> Any | None:
    try:
        import isaaclab.sim as sim_utils  # type: ignore

        return sim_utils.get_current_stage()
    except Exception:
        pass
    try:
        return scene_handle.sim.stage
    except Exception:
        return None


def _first_available(owner: Any, *names: str) -> Any:
    for name in names:
        value = getattr(owner, name, None)
        arr = to_numpy(value)
        if arr.size:
            return value
    return None


def _limit_matrix(value: Any, count: int) -> list[list[float]]:
    arr = to_numpy(value)
    if arr.ndim >= 3:
        arr = arr[0]
    rows: list[list[float]] = []
    if arr.ndim == 2 and arr.shape[-1] == 2:
        for index in range(min(count, arr.shape[0])):
            rows.append([_float(arr[index, 0]), _float(arr[index, 1])])
    return rows


def _limit_vector(value: Any, count: int) -> list[float]:
    arr = to_numpy(value)
    if arr.ndim >= 2:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[-1] == 2:
        arr = np.nanmax(np.abs(arr[:, :2]), axis=1)
    arr = arr.reshape(-1) if arr.size else arr
    return [_float(arr[index]) if index < arr.size else float("nan") for index in range(count)]


def _vec(value: Any) -> list[float]:
    try:
        return [float(item) for item in value]
    except Exception:
        return []


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _audit_text(audit: dict[str, Any]) -> str:
    lines = [
        "Height Replay Telemetry Model Audit",
        f"Isaac Lab: {audit.get('isaac_lab_version', '') or 'unknown'}",
        f"Robot USD: {audit.get('robot_usd', '')}",
        f"Bodies: {audit.get('body_count', 0)}",
        f"Joints: {audit.get('joint_count', 0)}",
        f"ContactSensor available: {bool((audit.get('contact_sensor') or {}).get('available', False))}",
        "",
        "Warnings:",
    ]
    warnings = list(audit.get("warnings", []) or [])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    return "\n".join(lines) + "\n"
