"""Compact live telemetry packets for lightweight consumers."""

from __future__ import annotations

import json
from typing import Any


LIVE_PACKET_FIELDS = (
    "sample_index",
    "time_s",
    "sim_step",
    "com_x_m",
    "com_y_m",
    "com_z_m",
    "com_vx_m_s",
    "com_vy_m_s",
    "com_vz_m_s",
    "com_projection_x_m",
    "com_projection_y_m",
    "static_stability_margin_m",
    "dynamic_stability_margin_m",
    "equilibrium_stability_margin_m",
    "stability_state",
    "dynamic_stability_state",
    "support_area_m2",
    "support_degenerate",
    "equilibrium_status",
    "base_roll_rad",
    "base_pitch_rad",
    "base_yaw_rad",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "base_vx_m_s",
    "base_vy_m_s",
    "base_vz_m_s",
    "base_wx_rad_s",
    "base_wy_rad_s",
    "base_wz_rad_s",
    "active_contact_count",
    "max_contact_force_n",
    "max_friction_utilization",
    "max_wheel_slip_ratio",
    "max_torque_utilization",
    "max_torque_joint",
    "tracking_rmse_rad",
    "max_abs_tracking_error_rad",
    "total_positive_work_j",
    "sample_overhead_ms",
    "warnings",
    "replay_state",
    "step_progress",
    "replay_event_index",
    "replay_event_count",
)

FIELD_UNITS = {
    "time_s": "s",
    "com_x_m": "m",
    "com_y_m": "m",
    "com_z_m": "m",
    "com_vx_m_s": "m/s",
    "com_vy_m_s": "m/s",
    "com_vz_m_s": "m/s",
    "com_projection_x_m": "m",
    "com_projection_y_m": "m",
    "static_stability_margin_m": "m",
    "dynamic_stability_margin_m": "m",
    "equilibrium_stability_margin_m": "m",
    "support_area_m2": "m^2",
    "base_roll_rad": "rad",
    "base_pitch_rad": "rad",
    "base_yaw_rad": "rad",
    "base_x_m": "m",
    "base_y_m": "m",
    "base_z_m": "m",
    "base_vx_m_s": "m/s",
    "base_vy_m_s": "m/s",
    "base_vz_m_s": "m/s",
    "base_wx_rad_s": "rad/s",
    "base_wy_rad_s": "rad/s",
    "base_wz_rad_s": "rad/s",
    "active_contact_count": "count",
    "max_contact_force_n": "N",
    "tracking_rmse_rad": "rad",
    "max_abs_tracking_error_rad": "rad",
    "total_positive_work_j": "J",
    "sample_overhead_ms": "ms",
}


def live_packet_from_row(
    row: dict[str, Any] | None,
    *,
    live_frame_id: int,
    session_id: str = "",
    trial_id: int = 0,
    generation_id: int = 0,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": "telemetry.live.v1",
        "live_frame_id": int(live_frame_id),
        "session_id": str(session_id or ""),
        "trial_id": int(trial_id or 0),
        "generation_id": int(generation_id or 0),
    }
    if not isinstance(row, dict):
        packet["empty"] = True
        packet["approx_size_bytes"] = len(json.dumps(packet, separators=(",", ":"), default=str).encode("utf-8"))
        return packet
    for key in LIVE_PACKET_FIELDS:
        value = row.get(key)
        if isinstance(value, (int, float, str, bool)) or value is None:
            packet[key] = value
    for key in (
        "joint_catalog",
        "robot_joint_names",
        "active_contact_count_label",
        "contact_geometry_source",
        "contact_force_source",
        "contact_force_valid",
        "contact_force_reason",
        "friction_utilization_valid",
        "friction_utilization_reason",
        "wheel_slip_state",
        "wheel_slip_source",
        "equilibrium_margin_valid",
        "equilibrium_reason",
    ):
        if key in row:
            packet[key] = row.get(key)
    packet["field_metadata"] = _field_metadata(row, packet)
    packet["replay_active"] = str(row.get("replay_state", "") or "") == "active"
    packet["replay_progress"] = row.get("step_progress", 0.0)
    packet["approx_size_bytes"] = len(json.dumps(packet, separators=(",", ":"), default=str).encode("utf-8"))
    return packet


def _field_metadata(row: dict[str, Any], packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    timestamp = row.get("time_s")
    frame = packet.get("live_frame_id")
    for key in LIVE_PACKET_FIELDS:
        value = packet.get(key)
        valid = _value_valid(value)
        reason = "" if valid else "unavailable"
        if key == "max_contact_force_n":
            valid = bool(row.get("contact_force_valid", valid))
            reason = str(row.get("contact_force_reason", "" if valid else "contact force unavailable") or "")
        elif key == "max_friction_utilization":
            valid = bool(row.get("friction_utilization_valid", valid))
            reason = str(row.get("friction_utilization_reason", "" if valid else "friction utilization unavailable") or "")
        elif key == "equilibrium_stability_margin_m":
            valid = bool(row.get("equilibrium_margin_valid", valid))
            reason = str(row.get("equilibrium_reason", "" if valid else "equilibrium margin unavailable") or "")
        metadata[key] = {
            "unit": FIELD_UNITS.get(key, ""),
            "valid": bool(valid),
            "source": str(row.get(f"{key}_source", _default_source(key)) or _default_source(key)),
            "reason": reason,
            "timestamp_s": timestamp,
            "frame": frame,
            "confidence": row.get(f"{key}_confidence", None),
        }
    return metadata


def _value_valid(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return value == value and value not in {float("inf"), float("-inf")}
    if isinstance(value, (int, bool)):
        return True
    if isinstance(value, str):
        return bool(value)
    return False


def _default_source(key: str) -> str:
    if key.startswith("com_"):
        return "whole_body_com"
    if "contact" in key or key in {"max_friction_utilization", "max_wheel_slip_ratio"}:
        return "contact_metrics"
    if "equilibrium" in key:
        return "equilibrium_solver"
    if key.startswith("base_"):
        return "isaaclab.ArticulationData.root"
    if "torque" in key or "tracking" in key:
        return "joint_metrics"
    return "telemetry_collector"
