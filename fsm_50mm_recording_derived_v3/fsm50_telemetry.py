"""Extended telemetry and physical evidence capture for 50 mm replay/FSM runs."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from command_model import SERVO_JOINT_NAMES, WHEEL_FORWARD_SIGN, WHEEL_JOINT_NAMES
from telemetry.collector import TelemetryCollector
from telemetry.com_metrics import to_numpy
from telemetry.exporters import write_csv, write_json, write_jsonl

from .filtered_wheel_contact import FILTERED_SURFACES, filtered_contact_rows
from .nonwheel_obstacle_contact import (
    NONWHEEL_FORCE_SOURCE,
    nonwheel_obstacle_contact_rows,
)
from .support_classifier import (
    LEGS,
    ContactClass,
    ContactPersistenceTracker,
    ObstacleGeometry,
    TraversalEvidenceTracker,
    WheelObservation,
    classify_diagonal_support,
    classify_wheel_contact,
    diagonal_support_corridor,
    polygon_support_margin,
)


LEG_TO_WHEEL_JOINT = {
    "FL": "front_left_ankle",
    "FR": "front_right_ankle",
    "RL": "rear_left_ankle",
    "RR": "rear_right_ankle",
}
LEG_TO_WHEEL_BODY = {
    "FL": "front_left_wheel",
    "FR": "front_right_wheel",
    "RL": "rear_left_wheel",
    "RR": "rear_right_wheel",
}

FILTERED_FORCE_SOURCE = "isaaclab.ContactSensor.force_matrix_w"
FILTERED_GEOMETRY_SOURCE = "isaaclab.ContactSensor.contact_pos_w"
FILTERED_FRICTION_SOURCE = "isaaclab.ContactSensor.friction_forces_w"
COMMON_WHEEL_FORCE_SOURCE = "isaaclab.ContactSensor.net_forces_w"


def _wheel_sign(value: Any, *, label: str) -> float:
    sign = _safe_float(value)
    if not math.isfinite(sign) or abs(sign) < 1.0e-12:
        raise ValueError(f"{label} must be a finite, non-zero direction sign")
    return -1.0 if sign < 0.0 else 1.0


def canonical_wheel_values(
    values: Mapping[str, float],
    *,
    wheel_direction: float = 1.0,
    wheel_forward_sign: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Convert raw joint values so positive always means robot-forward motion."""

    configured = dict(wheel_forward_sign or {})
    global_sign = _wheel_sign(wheel_direction, label="wheel_direction")
    result: dict[str, float] = {}
    for leg in LEGS:
        joint_name = LEG_TO_WHEEL_JOINT[leg]
        joint_sign = _wheel_sign(
            configured.get(leg, WHEEL_FORWARD_SIGN[joint_name]),
            label=f"wheel_forward_sign[{leg}]",
        )
        result[leg] = global_sign * joint_sign * _safe_float(values.get(leg))
    return result


def _as_row(value: Any, env_id: int = 0) -> np.ndarray:
    array = np.asarray(to_numpy(value), dtype=float)
    if array.ndim >= 2:
        array = array[min(max(0, env_id), array.shape[0] - 1)]
    return array


def _leg_for_body(body_name: str) -> str | None:
    lowered = str(body_name).lower()
    for leg, expected in LEG_TO_WHEEL_BODY.items():
        if expected in lowered:
            return leg
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


class FSM50TelemetryCollector(TelemetryCollector):
    """Augment the existing telemetry collector without replacing it.

    The base collector remains the source for COM, base attitude, joint
    diagnostics, and raw ContactSensor rows.  This class adds the per-wheel
    obstacle-relative states, diagonal/corridor logic, integrated wheel travel,
    and strict linkage-lift evidence needed by the 50 mm task.
    """

    def __init__(
        self,
        config: Any,
        *,
        args: Any | None,
        scene_handle: Any,
        obstacle: ObstacleGeometry,
        wheel_radius_m: float,
        source_version: str,
        plan: Any | None = None,
        plan_rows: list[dict[str, Any]] | None = None,
        force_threshold_n: float = 2.0,
        unload_force_n: float = 1.0,
        load_confirm_force_n: float = 2.0,
        top_load_dwell_s: float = 0.10,
        loaded_front_face_rotation_limit_rad: float = 0.15,
        wheel_forward_sign: Mapping[str, float] | None = None,
        maximum_roll_rad: float = math.radians(45.0),
        maximum_pitch_rad: float = math.radians(45.0),
        maximum_angular_velocity_rad_s: float = 3.0,
        maximum_contact_drift_m: float = 0.03,
        final_linear_velocity_m_s: float = 0.03,
        final_angular_velocity_rad_s: float = 0.15,
        final_stable_dwell_s: float = 0.25,
        dangerous_nonwheel_contact_force_n: float = 5.0,
    ) -> None:
        super().__init__(config, args=args, scene_handle=scene_handle)
        self.obstacle = obstacle
        self.wheel_radius_m = float(wheel_radius_m)
        self.source_version = str(source_version)
        self.plan = plan
        self.plan_rows_by_segment = {
            int(row["decoded_segment_index"]): dict(row)
            for row in list(plan_rows or [])
            if row.get("decoded_segment_index") is not None
        }
        self.force_threshold_n = float(force_threshold_n)
        self.maximum_roll_rad = float(maximum_roll_rad)
        self.maximum_pitch_rad = float(maximum_pitch_rad)
        self.maximum_angular_velocity_rad_s = float(maximum_angular_velocity_rad_s)
        self.maximum_allowed_contact_drift_m = float(maximum_contact_drift_m)
        self.final_linear_velocity_m_s = float(final_linear_velocity_m_s)
        self.final_angular_velocity_rad_s = float(final_angular_velocity_rad_s)
        self.final_stable_dwell_s = float(final_stable_dwell_s)
        self.dangerous_nonwheel_contact_force_n = float(
            dangerous_nonwheel_contact_force_n
        )
        requested_signs = dict(wheel_forward_sign or {})
        self.wheel_forward_sign = {
            leg: _wheel_sign(
                requested_signs.get(
                    leg,
                    WHEEL_FORWARD_SIGN[LEG_TO_WHEEL_JOINT[leg]],
                ),
                label=f"wheel_forward_sign[{leg}]",
            )
            for leg in LEGS
        }
        self.persistence = ContactPersistenceTracker()
        self.traversal = TraversalEvidenceTracker(
            unload_force_n=unload_force_n,
            load_confirm_force_n=load_confirm_force_n,
            top_load_dwell_s=top_load_dwell_s,
            loaded_front_face_rotation_limit_rad=loaded_front_face_rotation_limit_rad,
            # This collector passes already-canonicalized angles to the
            # traversal tracker, so a second left/right sign correction would
            # be incorrect.
            wheel_forward_sign={leg: 1.0 for leg in LEGS},
        )
        self.fsm50_rows: list[dict[str, Any]] = []
        self.state_timeline_rows: list[dict[str, Any]] = []
        self.runtime_context: dict[str, Any] = {}
        self.last_timeline_key: tuple[Any, ...] | None = None
        self.previous_wheel_angle: dict[str, float] = {}
        self.integrated_wheel_rotation: dict[str, float] = {leg: 0.0 for leg in LEGS}
        self.integrated_wheel_travel: dict[str, float] = {leg: 0.0 for leg in LEGS}
        self.maximum_contact_drift_m = 0.0
        self.minimum_corridor_margin_m = float("inf")
        self.dangerous_collision_rows: list[dict[str, Any]] = []
        self.nonwheel_obstacle_rows: list[dict[str, Any]] = []
        self.collision_evidence_errors: list[str] = []
        self.joint_limit_violation_rows: list[dict[str, Any]] = []
        self.filtered_surface_rows: list[dict[str, Any]] = []
        self.filtered_contact_errors: list[str] = []

    def set_runtime_context(self, **values: Any) -> None:
        self.runtime_context.update(values)

    def on_step(self, adapter: Any, dt_s: float) -> None:
        before_rows = len(self.rows)
        before_contacts = len(self.contact_rows)
        super().on_step(adapter, dt_s)
        if len(self.rows) == before_rows:
            return
        base_row = self.rows[-1]
        contacts = self.contact_rows[before_contacts:]
        extended = self._extend_row(adapter, base_row, contacts)
        filtered_rows = list(extended.get("wheel_filtered_contacts", []) or [])
        filtered_evidence = self._install_filtered_contact_sample(
            base_row=base_row,
            filtered_rows=filtered_rows,
            before_contacts=before_contacts,
        )
        extended.update(filtered_evidence)
        base_row.update(
            {
                "primary_diagonal": extended["primary_diagonal"],
                "support_legs": extended["support_legs"],
                "light_support_legs": extended["light_support_legs"],
                "diagonal_load_fl_rr_n": extended["diagonal_load_fl_rr_n"],
                "diagonal_load_fr_rl_n": extended["diagonal_load_fr_rl_n"],
                "two_leg_corridor_distance_m": extended[
                    "two_leg_corridor_distance_m"
                ],
                "wheel_contact_classes": extended["wheel_contact_classes"],
                **filtered_evidence,
            }
        )
        self.fsm50_rows.append(extended)
        self._record_timeline(extended)

    @staticmethod
    def _filtered_layout_valid(rows: list[dict[str, Any]]) -> bool:
        expected = {
            (leg, surface)
            for leg in LEGS
            for surface, _prim_path in FILTERED_SURFACES
        }
        actual = {
            (str(row.get("leg", "")), str(row.get("surface", "")))
            for row in rows
        }
        return len(rows) == len(expected) and actual == expected

    def _install_filtered_contact_sample(
        self,
        *,
        base_row: dict[str, Any],
        filtered_rows: list[dict[str, Any]],
        before_contacts: int,
    ) -> dict[str, Any]:
        """Replace generic contacts with the eight labelled wheel/surface rows."""

        layout_valid = self._filtered_layout_valid(filtered_rows)
        if not layout_valid:
            return {
                "filtered_contact_layout_valid": False,
                "filtered_contact_force_valid": False,
                "filtered_contact_geometry_valid": False,
            }

        time_s = float(base_row.get("time_s", 0.0) or 0.0)
        csv_rows = [self._contact_csv_row(time_s, row) for row in filtered_rows]
        # Keep the base collector and its contacts.csv output coherent: one
        # sample is always FL/FR/RL/RR x ground/obstacle, including inactive
        # pairs, rather than a second set of aggregate net-force rows.
        self.contact_rows[before_contacts:] = csv_rows

        force_valid = all(bool(row.get("force_valid", False)) for row in filtered_rows)
        active_rows = [row for row in filtered_rows if bool(row.get("active", False))]
        geometry_valid = all(
            bool(row.get("contact_point_valid", False)) for row in active_rows
        )
        total_forces = [
            _safe_float(row.get("total_force_n"))
            for row in filtered_rows
            if bool(row.get("force_valid", False))
            and math.isfinite(_safe_float(row.get("total_force_n")))
        ]
        max_force = max(total_forces, default=0.0)
        force_reason = (
            ""
            if force_valid
            else "one or more filtered force_matrix_w wheel/surface rows are non-finite"
        )
        return {
            "filtered_contact_layout_valid": True,
            "filtered_contact_force_valid": force_valid,
            "filtered_contact_geometry_valid": geometry_valid,
            "active_contact_count": len(active_rows),
            "active_contact_count_label": f"{len(active_rows)} (sensor-confirmed)",
            "contact_geometry_source": FILTERED_GEOMETRY_SOURCE,
            "contact_force_source": FILTERED_FORCE_SOURCE,
            "contact_force_valid": force_valid,
            "contact_force_reason": force_reason,
            "max_contact_force_n": max_force,
            "max_contact_force_n_source": FILTERED_FORCE_SOURCE,
        }

    def _contact_csv_row(
        self,
        time_s: float,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        normal = np.asarray(row.get("normal_force_w", []), dtype=float).reshape(-1)
        normal_n = _safe_float(row.get("normal_force_n"))
        contact_normal = [float("nan")] * 3
        if normal.size >= 3 and np.isfinite(normal[:3]).all() and normal_n > 1.0e-12:
            contact_normal = (normal[:3] / normal_n).tolist()
        return {
            "time_s": float(time_s),
            "source_version": self.source_version,
            **dict(row),
            "body_name": str(row.get("wheel_body_name", "")),
            "other_body": str(row.get("other_prim_path", "")),
            "contact_normal_w": contact_normal,
            "tangential_force_n": _safe_float(row.get("friction_force_n")),
            "wheel_contact": True,
            "source": FILTERED_FORCE_SOURCE,
            "geometry_source": FILTERED_GEOMETRY_SOURCE,
            "friction_source": FILTERED_FRICTION_SOURCE,
        }

    def _body_positions(self, adapter: Any) -> dict[str, tuple[float, float, float]]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        names = [str(name) for name in (getattr(robot, "body_names", []) or [])]
        states = _as_row(getattr(data, "body_link_state_w", None))
        if states.size == 0:
            states = _as_row(getattr(data, "body_state_w", None))
        result: dict[str, tuple[float, float, float]] = {}
        if states.ndim == 2 and states.shape[1] >= 3:
            for index, name in enumerate(names[: states.shape[0]]):
                result[name] = tuple(float(value) for value in states[index, :3])
        return result

    def _joint_vectors(self, adapter: Any) -> tuple[dict[str, float], dict[str, float]]:
        robot = getattr(adapter, "robot", None)
        data = getattr(robot, "data", None)
        names = [str(name) for name in (getattr(robot, "joint_names", []) or [])]
        positions = _as_row(getattr(data, "joint_pos", None)).reshape(-1)
        velocities = _as_row(getattr(data, "joint_vel", None)).reshape(-1)
        pos = {
            name: float(positions[index])
            for index, name in enumerate(names[: positions.size])
        }
        vel = {
            name: float(velocities[index])
            for index, name in enumerate(names[: velocities.size])
        }
        return pos, vel

    @staticmethod
    def _force_by_leg(raw_contacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Aggregate legacy contact rows using their historical field meanings."""

        result = {
            leg: {
                "upward_force_n": float("nan"),
                "total_force_n": float("nan"),
                "source": "unavailable",
            }
            for leg in LEGS
        }
        for row in raw_contacts:
            if row.get("active") is False:
                continue
            leg = _leg_for_body(str(row.get("body_name", "")))
            if leg not in LEGS:
                continue
            upward = _safe_float(row.get("normal_force_n"))
            total = _safe_float(row.get("total_force_n"))
            FSM50TelemetryCollector._accumulate_force(
                result,
                leg=leg,
                upward=upward,
                total=total,
                source=str(row.get("source", "unavailable") or "unavailable"),
            )
        return result

    @staticmethod
    def _filtered_force_by_leg(
        filtered_contacts: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate filtered pairs by explicit leg and world-up force only."""

        result = {
            leg: {
                "upward_force_n": float("nan"),
                "total_force_n": float("nan"),
                "source": "unavailable",
            }
            for leg in LEGS
        }
        for row in filtered_contacts:
            # All eight filtered rows carry a real force vector, including an
            # inactive pair whose finite vector is zero (or merely below the
            # activity threshold).  Dropping those rows turns a sensor-proven
            # zero into NaN and makes the common A/B force signal unusable.
            if row.get("force_valid") is False:
                continue
            leg = str(row.get("leg", "") or "").upper()
            if leg not in LEGS:
                continue
            # Do not substitute the force-vector norm here: a large horizontal
            # front-face normal is not vertical support load.
            upward = _safe_float(row.get("upward_force_n"))
            total = _safe_float(row.get("total_force_n"))
            FSM50TelemetryCollector._accumulate_force(
                result,
                leg=leg,
                upward=upward,
                total=total,
                source=str(row.get("source", FILTERED_FORCE_SOURCE) or FILTERED_FORCE_SOURCE),
            )
        return result

    @staticmethod
    def _common_wheel_net_force_by_leg(
        contact_sensor: Any,
        *,
        env_id: int = 0,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Read the shared four-wheel force signal from ``net_forces_w``.

        Formal aggregate sensors expose many robot bodies, while the filtered
        sensor facade exposes exactly four.  Both nevertheless provide the
        same full per-body ``net_forces_w`` tensor.  Reading that tensor is the
        only sound way to distinguish a finite, sensor-observed zero force from
        a missing active-contact row.  Any ambiguous layout, invalid tensor
        shape, or non-finite wheel vector fails closed for all four legs.
        """

        def empty_forces() -> dict[str, dict[str, Any]]:
            return {
                leg: {
                    "upward_force_n": float("nan"),
                    "total_force_n": float("nan"),
                    "source": COMMON_WHEEL_FORCE_SOURCE,
                }
                for leg in LEGS
            }

        def invalid(
            error: str,
            *,
            layout_valid: bool = False,
        ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
            return empty_forces(), {
                "wheel_net_force_layout_valid": bool(layout_valid),
                "wheel_net_force_valid": False,
                "wheel_net_force_error": str(error),
                "wheel_contact_force_common_source": COMMON_WHEEL_FORCE_SOURCE,
            }

        if contact_sensor is None:
            return invalid("contact sensor is unavailable")

        body_names = [str(name) for name in (getattr(contact_sensor, "body_names", []) or [])]
        if not body_names:
            return invalid("contact sensor body_names is empty")
        normalized_names = [
            name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
            for name in body_names
        ]
        body_index_by_leg: dict[str, int] = {}
        layout_errors: list[str] = []
        for leg, expected_body in LEG_TO_WHEEL_BODY.items():
            matches = [
                index
                for index, normalized in enumerate(normalized_names)
                if normalized == expected_body.lower()
            ]
            if len(matches) != 1:
                layout_errors.append(
                    f"{leg}/{expected_body} resolved {len(matches)} bodies; exactly one is required"
                )
            else:
                body_index_by_leg[leg] = matches[0]
        if layout_errors:
            return invalid("; ".join(layout_errors))

        try:
            data = getattr(contact_sensor, "data", None)
            net_forces = np.asarray(
                to_numpy(getattr(data, "net_forces_w", None)),
                dtype=float,
            )
        except Exception as exc:
            return invalid(f"net_forces_w read failed: {type(exc).__name__}: {exc}")
        if net_forces.ndim != 3:
            return invalid(
                f"net_forces_w expected [env, body, xyz], got shape={net_forces.shape}"
            )
        if net_forces.shape[1] != len(body_names) or net_forces.shape[2] != 3:
            return invalid(
                "net_forces_w/body_names layout mismatch: "
                f"shape={net_forces.shape} body_names={len(body_names)}"
            )
        selected_env = int(env_id)
        if selected_env < 0 or selected_env >= net_forces.shape[0]:
            return invalid(
                f"env_id={selected_env} outside [0, {net_forces.shape[0]})"
            )

        wheel_vectors = {
            leg: np.asarray(net_forces[selected_env, body_index], dtype=float).reshape(-1)
            for leg, body_index in body_index_by_leg.items()
        }
        nonfinite_legs = [
            leg
            for leg, vector in wheel_vectors.items()
            if vector.size != 3 or not np.isfinite(vector).all()
        ]
        if nonfinite_legs:
            return invalid(
                "non-finite net_forces_w vector for " + ", ".join(sorted(nonfinite_legs)),
                layout_valid=True,
            )

        forces = {
            leg: {
                "upward_force_n": max(0.0, float(vector[2])),
                "total_force_n": float(np.linalg.norm(vector)),
                "source": COMMON_WHEEL_FORCE_SOURCE,
            }
            for leg, vector in wheel_vectors.items()
        }
        return forces, {
            "wheel_net_force_layout_valid": True,
            "wheel_net_force_valid": True,
            "wheel_net_force_error": "",
            "wheel_contact_force_common_source": COMMON_WHEEL_FORCE_SOURCE,
        }

    @staticmethod
    def _accumulate_force(
        result: dict[str, dict[str, Any]],
        *,
        leg: str,
        upward: float,
        total: float,
        source: str,
    ) -> None:
        current_up = result[leg]["upward_force_n"]
        current_total = result[leg]["total_force_n"]
        if math.isfinite(upward):
            result[leg]["upward_force_n"] = (
                max(0.0, upward)
                if not math.isfinite(current_up)
                else current_up + max(0.0, upward)
            )
        if math.isfinite(total):
            result[leg]["total_force_n"] = (
                max(0.0, total)
                if not math.isfinite(current_total)
                else current_total + max(0.0, total)
            )
        result[leg]["source"] = str(source or "unavailable")

    def _filtered_contacts(self) -> tuple[list[dict[str, Any]], str]:
        sensor = getattr(self.scene_handle, "contact_sensor", None)
        if not bool(getattr(sensor, "is_filtered_wheel_contact_bank", False)):
            return [], "filtered per-wheel ContactSensor bank unavailable"
        try:
            return (
                filtered_contact_rows(
                    sensor,
                    force_threshold_n=self.force_threshold_n,
                ),
                "",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.filtered_contact_errors or self.filtered_contact_errors[-1] != error:
                self.filtered_contact_errors.append(error)
            return [], error

    def _nonwheel_collision_sample(
        self, time_s: float
    ) -> tuple[list[dict[str, Any]], bool, bool, str]:
        sensor = getattr(self.scene_handle, "contact_sensor", None)
        try:
            rows = nonwheel_obstacle_contact_rows(
                sensor,
                force_threshold_n=1.0,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.collision_evidence_errors or self.collision_evidence_errors[-1] != error:
                self.collision_evidence_errors.append(error)
            return [], False, False, error
        valid = bool(rows) and all(
            bool(row.get("force_valid", False))
            and (
                not bool(row.get("active", False))
                or bool(row.get("contact_point_valid", False))
            )
            for row in rows
        )
        dangerous = False
        stamped: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "time_s": float(time_s),
                "source_version": self.source_version,
                **row,
            }
            force = _safe_float(row.get("normal_force_n"))
            item["dangerous"] = bool(
                valid
                and bool(row.get("active", False))
                and math.isfinite(force)
                and force >= self.dangerous_nonwheel_contact_force_n
            )
            dangerous = bool(dangerous or item["dangerous"])
            stamped.append(item)
            self.nonwheel_obstacle_rows.append(item)
            if item["dangerous"]:
                self.dangerous_collision_rows.append(item)
        return stamped, valid, dangerous, "" if valid else "invalid non-wheel contact layout/force evidence"

    @staticmethod
    def _install_measured_contact_points(
        classified: Mapping[str, Any],
        filtered_contacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replace geometric projections with sensor-measured contact points.

        The wheel center remains the geometry source for surface class and
        clearance, while support polygons, diagonal corridors, and anchor
        drift use ``ContactSensor.contact_pos_w``.  At the upper front corner
        both filters may be active, so the row matching the classified surface
        is preferred and the largest-force valid row is the deterministic
        fallback.
        """

        result = dict(classified)
        for leg, contact in list(result.items()):
            desired_surface = (
                "ground"
                if contact.contact_class == ContactClass.GROUND
                else "obstacle"
                if contact.contact_class in {ContactClass.FRONT_FACE, ContactClass.TOP}
                else ""
            )
            candidates = [
                row
                for row in filtered_contacts
                if str(row.get("leg", "")).upper() == leg
                and bool(row.get("active", False))
                and bool(row.get("contact_point_valid", False))
            ]
            preferred = [
                row for row in candidates if str(row.get("surface", "")) == desired_surface
            ]
            pool = preferred or candidates
            if not pool:
                continue
            selected = max(
                pool,
                key=lambda row: (
                    _safe_float(row.get("total_force_n"))
                    if math.isfinite(_safe_float(row.get("total_force_n")))
                    else -math.inf
                ),
            )
            point = np.asarray(selected.get("contact_point_w", []), dtype=float).reshape(-1)
            if point.size < 3 or not np.isfinite(point[:3]).all():
                continue
            result[leg] = replace(
                contact,
                contact_point_w=tuple(float(value) for value in point[:3]),
                source=f"{contact.source}; {FILTERED_GEOMETRY_SOURCE}",
            )
        return result

    def _segment_context(self) -> dict[str, Any]:
        segment_index = self.runtime_context.get("segment_index")
        segment = None
        if self.plan is not None and segment_index is not None:
            try:
                index = int(segment_index)
                if 0 <= index < len(self.plan.segments):
                    segment = self.plan.segments[index]
            except Exception:
                segment = None
        if segment is None:
            return {
                "source_fast_segment": segment_index,
                "source_step": self.runtime_context.get("source_step"),
                "plan_event_start_index": None,
                "source_event_indices": [],
                "source_command_provenance": [],
                "planned_servo_target_deg": {},
                "planned_wheel_target_rad_s": {},
                "atomic_concurrent": False,
            }
        aligned = self.plan_rows_by_segment.get(int(segment.segment_index), {})
        source_event_indices = [
            int(value) for value in list(aligned.get("source_event_indices", []) or [])
        ]
        return {
            "source_fast_segment": int(segment.segment_index),
            "source_step": int(segment.source_step),
            # ``event_start_index`` belongs to the decoded plan.  It is not an
            # accepted_steps source-event index and must never be presented as
            # one.  The recording mapping comes from recording_fast_plan.
            "plan_event_start_index": int(segment.event_start_index),
            "source_event_indices": source_event_indices,
            "source_event_start": (
                source_event_indices[0] if source_event_indices else None
            ),
            "source_event_count": len(source_event_indices),
            "source_command_provenance": list(
                aligned.get("command_provenance", []) or []
            ),
            "planned_servo_target_deg": dict(segment.servo_targets),
            "planned_wheel_target_rad_s": dict(segment.wheel_applied_target_rad_s),
            "atomic_concurrent": bool(
                segment.servo_targets
                and any(
                    abs(float(value)) > 1.0e-12
                    for value in segment.wheel_applied_target_rad_s.values()
                )
            ),
        }

    def _joint_safety_evidence(
        self,
        adapter: Any,
        joint_pos: Mapping[str, float],
        actual_target: Mapping[str, float],
    ) -> dict[str, Any]:
        try:
            diagnostics = list(adapter.get_joint_diagnostics())
        except Exception as exc:
            return {
                "joint_limit_evidence_valid": False,
                "joint_limit_evidence_error": f"{type(exc).__name__}: {exc}",
                "joint_limit_margin_rad": {},
                "minimum_joint_limit_margin_rad": float("nan"),
                "joint_limit_violation": True,
                "actuator_target_error_rad": {},
            }
        margins: dict[str, float] = {}
        errors: dict[str, float] = {}
        valid = True
        violations: list[str] = []
        for row in diagnostics:
            name = str(row.get("joint_name", ""))
            if name not in SERVO_JOINT_NAMES:
                continue
            measured = _safe_float(joint_pos.get(name))
            target = _safe_float(actual_target.get(name))
            limits = list(row.get("current_physx_or_usd_limit_rad", []) or [])
            if len(limits) < 2 or not all(math.isfinite(_safe_float(v)) for v in limits):
                valid = False
                margins[name] = float("nan")
            elif not math.isfinite(measured):
                valid = False
                margins[name] = float("nan")
            else:
                lower, upper = float(limits[0]), float(limits[1])
                margin = min(measured - lower, upper - measured)
                margins[name] = margin
                if margin < -1.0e-5:
                    violations.append(name)
            errors[name] = (
                target - measured
                if math.isfinite(target) and math.isfinite(measured)
                else float("nan")
            )
        if set(margins) != set(SERVO_JOINT_NAMES):
            valid = False
        finite_margins = [value for value in margins.values() if math.isfinite(value)]
        violation = bool(violations or not valid)
        if violation:
            self.joint_limit_violation_rows.append(
                {
                    "time_s": float(getattr(adapter, "sim_time", 0.0) or 0.0),
                    "violating_joints": violations,
                    "evidence_valid": valid,
                }
            )
        return {
            "joint_limit_evidence_valid": valid,
            "joint_limit_evidence_error": "" if valid else "one or more servo limits/positions unavailable",
            "joint_limit_margin_rad": margins,
            "minimum_joint_limit_margin_rad": min(finite_margins, default=float("nan")),
            "joint_limit_violation": violation,
            "actuator_target_error_rad": errors,
            "maximum_actuator_target_error_rad": max(
                (abs(value) for value in errors.values() if math.isfinite(value)),
                default=float("nan"),
            ),
        }

    def _extend_row(
        self,
        adapter: Any,
        base_row: dict[str, Any],
        raw_contacts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        time_s = float(base_row.get("time_s", 0.0) or 0.0)
        body_positions = self._body_positions(adapter)
        joint_pos, joint_vel = self._joint_vectors(adapter)
        filtered_contacts, filtered_contact_error = self._filtered_contacts()
        (
            nonwheel_contacts,
            collision_evidence_valid,
            dangerous_collision,
            collision_evidence_error,
        ) = self._nonwheel_collision_sample(time_s)
        forces, common_force_evidence = self._common_wheel_net_force_by_leg(
            getattr(self.scene_handle, "contact_sensor", None)
        )
        for contact in filtered_contacts:
            self.filtered_surface_rows.append(
                {
                    "time_s": time_s,
                    "source_version": self.source_version,
                    **contact,
                }
            )
        surface_loads = {
            leg: {
                surface: sum(
                    max(0.0, _safe_float(row.get("normal_force_n")))
                    for row in filtered_contacts
                    if row.get("active") is True
                    and str(row.get("leg", "")) == leg
                    and str(row.get("surface", "")) == surface
                    and math.isfinite(_safe_float(row.get("normal_force_n")))
                )
                for surface in ("ground", "obstacle")
            }
            for leg in LEGS
        }
        wheel_observations: dict[str, WheelObservation] = {}
        wheel_angles: dict[str, float] = {}
        for leg in LEGS:
            body_name = LEG_TO_WHEEL_BODY[leg]
            center = body_positions.get(body_name, (float("nan"),) * 3)
            wheel_joint = LEG_TO_WHEEL_JOINT[leg]
            wheel_angles[leg] = joint_pos.get(wheel_joint, float("nan"))
            wheel_observations[leg] = WheelObservation(
                leg=leg,
                center_w=center,
                upward_force_n=float(forces[leg]["upward_force_n"]),
                total_force_n=float(forces[leg]["total_force_n"]),
                wheel_angle_rad=wheel_angles[leg],
                wheel_velocity_rad_s=joint_vel.get(wheel_joint, float("nan")),
                force_source=str(forces[leg]["source"]),
            )
        classified = {
            leg: classify_wheel_contact(
                observation,
                self.obstacle,
                wheel_radius_m=self.wheel_radius_m,
                force_threshold_n=self.force_threshold_n,
            )
            for leg, observation in wheel_observations.items()
        }
        classified = self._install_measured_contact_points(
            classified, filtered_contacts
        )
        persistence, drift = self.persistence.update(time_s, classified)
        if drift:
            self.maximum_contact_drift_m = max(
                self.maximum_contact_drift_m, max(drift.values())
            )
        diagonal = classify_diagonal_support(
            classified,
            support_force_threshold_n=self.force_threshold_n,
            persistence_s=persistence,
            contact_drift_m=drift,
        )
        support_points = [
            classified[leg].contact_point_w[:2] for leg in diagonal.support_legs
        ]
        com_xy = (
            _safe_float(base_row.get("com_x_m")),
            _safe_float(base_row.get("com_y_m")),
        )
        polygon = polygon_support_margin(support_points, com_xy)
        corridor_distance = float("nan")
        corridor_fraction = float("nan")
        corridor_valid = False
        if len(diagonal.support_legs) == 2:
            left, right = diagonal.support_legs
            corridor = diagonal_support_corridor(
                classified[left].contact_point_w[:2],
                classified[right].contact_point_w[:2],
                com_xy,
                corridor_half_width_m=0.06,
            )
            corridor_distance = corridor.perpendicular_distance_m
            corridor_fraction = corridor.segment_fraction
            corridor_valid = corridor.valid
            if math.isfinite(corridor_distance):
                self.minimum_corridor_margin_m = min(
                    self.minimum_corridor_margin_m, 0.06 - corridor_distance
                )
        wheel_direction = getattr(adapter, "wheel_direction", 1.0)
        canonical_angles = canonical_wheel_values(
            wheel_angles,
            wheel_direction=wheel_direction,
            wheel_forward_sign=self.wheel_forward_sign,
        )
        canonical_velocities = canonical_wheel_values(
            {
                leg: joint_vel.get(LEG_TO_WHEEL_JOINT[leg], float("nan"))
                for leg in LEGS
            },
            wheel_direction=wheel_direction,
            wheel_forward_sign=self.wheel_forward_sign,
        )
        for leg, canonical in canonical_angles.items():
            previous = self.previous_wheel_angle.get(leg)
            if previous is not None and math.isfinite(canonical):
                delta = canonical - previous
                self.integrated_wheel_rotation[leg] += delta
                self.integrated_wheel_travel[leg] += delta * self.wheel_radius_m
            if math.isfinite(canonical):
                self.previous_wheel_angle[leg] = canonical
        self.traversal.update(time_s, classified, canonical_angles)

        command_deg = {
            name: _safe_float(dict(getattr(adapter, "joint_command_deg", {}) or {}).get(name))
            for name in SERVO_JOINT_NAMES
        }
        actual_target_rad = _as_row(
            getattr(getattr(getattr(adapter, "robot", None), "data", None), "joint_pos_target", None)
        ).reshape(-1)
        joint_names = [
            str(name) for name in (getattr(getattr(adapter, "robot", None), "joint_names", []) or [])
        ]
        actual_target = {
            name: float(actual_target_rad[index])
            for index, name in enumerate(joint_names[: actual_target_rad.size])
        }
        joint_safety = self._joint_safety_evidence(
            adapter, joint_pos, actual_target
        )
        wheel_command = dict(getattr(adapter, "wheel_speeds", {}) or {})
        segment = self._segment_context()
        row: dict[str, Any] = {
            **dict(base_row),
            "source_version": self.source_version,
            "fsm_state": str(self.runtime_context.get("fsm_state", "")),
            "scheduler_phase": str(self.runtime_context.get("scheduler_phase", "")),
            **segment,
            "command_space_servo_target_deg": command_deg,
            "actual_joint_target_rad": actual_target,
            "measured_joint_position_rad": joint_pos,
            "measured_joint_velocity_rad_s": joint_vel,
            "wheel_command_velocity_rad_s": wheel_command,
            "wheel_canonical_forward_angle_rad": canonical_angles,
            "wheel_canonical_forward_velocity_rad_s": canonical_velocities,
            "wheel_forward_sign": dict(self.wheel_forward_sign),
            "wheel_direction": _wheel_sign(
                wheel_direction,
                label="adapter.wheel_direction",
            ),
            "wheel_integrated_rotation_rad": dict(self.integrated_wheel_rotation),
            "wheel_integrated_travel_m": dict(self.integrated_wheel_travel),
            "wheel_contact_classes": {
                leg: classified[leg].contact_class.value for leg in LEGS
            },
            "wheel_centers_w": {leg: classified[leg].center_w for leg in LEGS},
            "wheel_obstacle_relative": {
                leg: classified[leg].obstacle_relative for leg in LEGS
            },
            "wheel_clearance_over_top_m": {
                leg: classified[leg].clearance_over_top_m for leg in LEGS
            },
            "wheel_front_face_clearance_m": {
                leg: classified[leg].front_face_clearance_m for leg in LEGS
            },
            "wheel_contact_points_w": {
                leg: classified[leg].contact_point_w for leg in LEGS
            },
            "wheel_contact_force_up_n": {
                leg: classified[leg].upward_force_n for leg in LEGS
            },
            "wheel_contact_force_total_n": {
                leg: classified[leg].total_force_n for leg in LEGS
            },
            **common_force_evidence,
            "wheel_contact_confidence": {
                leg: classified[leg].confidence for leg in LEGS
            },
            "filtered_contact_available": bool(filtered_contacts),
            "filtered_contact_error": filtered_contact_error,
            "wheel_filtered_contacts": filtered_contacts,
            "wheel_surface_normal_force_n": surface_loads,
            "wheel_contact_persistence_s": persistence,
            "wheel_contact_drift_m": drift,
            "support_legs": list(diagonal.support_legs),
            "light_support_legs": list(diagonal.light_support_legs),
            "persistent_support_legs": list(diagonal.persistent_support_legs),
            "stable_contact_legs": list(diagonal.stable_contact_legs),
            "primary_diagonal": diagonal.primary.value,
            "diagonal_load_fl_rr_n": diagonal.load_fl_rr_n,
            "diagonal_load_fr_rl_n": diagonal.load_fr_rl_n,
            "diagonal_load_share_fl_rr": diagonal.load_share_fl_rr,
            "diagonal_load_share_fr_rl": diagonal.load_share_fr_rl,
            "wheel_support_polygon_margin_m": polygon.signed_margin_m,
            "wheel_support_polygon_valid": polygon.valid,
            "wheel_support_polygon_degenerate": polygon.degenerate,
            "wheel_support_polygon": polygon.hull,
            "two_leg_corridor_distance_m": corridor_distance,
            "two_leg_corridor_fraction": corridor_fraction,
            "two_leg_corridor_valid": corridor_valid,
            "maximum_contact_drift_m_so_far": self.maximum_contact_drift_m,
            **joint_safety,
            "ik_evidence_applicable": False,
            "ik_evidence_valid": None,
            "ik_evidence_reason": "direct recording replay dispatches recorded joint commands and does not invoke IK",
            "nonwheel_obstacle_contacts": nonwheel_contacts,
            "collision_evidence_source": NONWHEEL_FORCE_SOURCE,
            "collision_evidence_valid": collision_evidence_valid,
            "collision_evidence_error": collision_evidence_error,
            "dangerous_collision": (
                dangerous_collision if collision_evidence_valid else None
            ),
        }
        return row

    def _record_timeline(self, row: dict[str, Any]) -> None:
        key = (
            row.get("source_fast_segment"),
            row.get("source_step"),
            tuple(row.get("source_event_indices", [])),
            tuple(row.get("support_legs", [])),
            row.get("primary_diagonal"),
            tuple(sorted(dict(row.get("wheel_contact_classes", {})).items())),
        )
        if key == self.last_timeline_key:
            if self.state_timeline_rows:
                self.state_timeline_rows[-1]["end_time_s"] = row.get("time_s")
                self.state_timeline_rows[-1]["sample_count"] += 1
            return
        self.last_timeline_key = key
        self.state_timeline_rows.append(
            {
                "start_time_s": row.get("time_s"),
                "end_time_s": row.get("time_s"),
                "sample_count": 1,
                "source_version": self.source_version,
                "source_fast_segment": row.get("source_fast_segment"),
                "source_step": row.get("source_step"),
                "source_event_indices": row.get("source_event_indices"),
                "support_legs": row.get("support_legs"),
                "primary_diagonal": row.get("primary_diagonal"),
                "wheel_contact_classes": row.get("wheel_contact_classes"),
            }
        )

    @staticmethod
    def _filtered_samples_valid(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(
            bool(row.get("filtered_contact_layout_valid", False))
            and bool(row.get("filtered_contact_force_valid", False))
            and bool(row.get("filtered_contact_geometry_valid", False))
            for row in rows
        )

    def physical_evidence(self) -> dict[str, Any]:
        traversal = self.traversal.result()
        last = self.fsm50_rows[-1] if self.fsm50_rows else {}
        contact_force_valid = any(
            bool(row.get("contact_force_valid", False)) for row in self.fsm50_rows
        )
        filtered_contact_valid = self._filtered_samples_valid(self.fsm50_rows)
        final_classes = dict(last.get("wheel_contact_classes", {}) or {})
        final_all_top = bool(
            final_classes
            and all(final_classes.get(leg) == ContactClass.TOP.value for leg in LEGS)
        )
        final_loads = dict(last.get("wheel_contact_force_up_n", {}) or {})
        final_all_loaded = bool(
            final_loads
            and all(
                math.isfinite(_safe_float(final_loads.get(leg)))
                and _safe_float(final_loads.get(leg)) >= self.force_threshold_n
                for leg in LEGS
            )
        )
        def finite_values(key: str) -> list[float]:
            values: list[float] = []
            for row in self.fsm50_rows:
                value = _safe_float(row.get(key))
                if math.isfinite(value):
                    values.append(value)
            return values

        rolls = finite_values("base_roll_rad")
        pitches = finite_values("base_pitch_rad")
        angular_speeds = [
            math.sqrt(wx * wx + wy * wy + wz * wz)
            for row in self.fsm50_rows
            for wx, wy, wz in [
                (
                    _safe_float(row.get("base_wx_rad_s")),
                    _safe_float(row.get("base_wy_rad_s")),
                    _safe_float(row.get("base_wz_rad_s")),
                )
            ]
            if all(math.isfinite(value) for value in (wx, wy, wz))
        ]
        attitude_evidence_valid = bool(
            self.fsm50_rows
            and len(rolls) == len(self.fsm50_rows)
            and len(pitches) == len(self.fsm50_rows)
            and len(angular_speeds) == len(self.fsm50_rows)
        )
        maximum_abs_roll = max((abs(value) for value in rolls), default=float("nan"))
        maximum_abs_pitch = max((abs(value) for value in pitches), default=float("nan"))
        maximum_angular_speed = max(angular_speeds, default=float("nan"))
        attitude_safe = bool(
            attitude_evidence_valid
            and maximum_abs_roll <= self.maximum_roll_rad
            and maximum_abs_pitch <= self.maximum_pitch_rad
            and maximum_angular_speed <= self.maximum_angular_velocity_rad_s
        )
        joint_limit_evidence_valid = bool(
            self.fsm50_rows
            and all(bool(row.get("joint_limit_evidence_valid", False)) for row in self.fsm50_rows)
        )
        joint_limit_safe = bool(
            joint_limit_evidence_valid
            and not any(bool(row.get("joint_limit_violation", True)) for row in self.fsm50_rows)
        )
        final_time = _safe_float(last.get("time_s"))
        stable_rows = [
            row
            for row in self.fsm50_rows
            if math.isfinite(final_time)
            and _safe_float(row.get("time_s")) >= final_time - self.final_stable_dwell_s
        ]
        final_stable = bool(stable_rows)
        for row in stable_rows:
            linear = math.sqrt(
                sum(_safe_float(row.get(key)) ** 2 for key in ("base_vx_m_s", "base_vy_m_s", "base_vz_m_s"))
            )
            angular = math.sqrt(
                sum(_safe_float(row.get(key)) ** 2 for key in ("base_wx_rad_s", "base_wy_rad_s", "base_wz_rad_s"))
            )
            if (
                not math.isfinite(linear)
                or not math.isfinite(angular)
                or linear > self.final_linear_velocity_m_s
                or angular > self.final_angular_velocity_rad_s
            ):
                final_stable = False
                break
        if stable_rows:
            stable_span = _safe_float(stable_rows[-1].get("time_s")) - _safe_float(
                stable_rows[0].get("time_s")
            )
            final_stable = bool(final_stable and stable_span + 1.0e-9 >= self.final_stable_dwell_s)
        collision_valid = bool(
            self.fsm50_rows
            and all(bool(row.get("collision_evidence_valid", False)) for row in self.fsm50_rows)
        )
        collision_safe = bool(
            collision_valid
            and not any(bool(row.get("dangerous_collision", False)) for row in self.fsm50_rows)
        )
        evidence_complete = bool(
            filtered_contact_valid
            and attitude_evidence_valid
            and joint_limit_evidence_valid
            and collision_valid
        )
        not_evaluable_reasons: list[str] = []
        if not filtered_contact_valid:
            not_evaluable_reasons.append("filtered wheel force/contact-point evidence incomplete")
        if not attitude_evidence_valid:
            not_evaluable_reasons.append("base attitude/angular-velocity evidence incomplete")
        if not joint_limit_evidence_valid:
            not_evaluable_reasons.append("per-servo joint-limit evidence incomplete")
        if not collision_valid:
            not_evaluable_reasons.append("non-wheel link/chassis obstacle-collision evidence unavailable")
        criteria = {
            "contact_evidence_valid": filtered_contact_valid,
            "all_legs_linkage_lift_valid": bool(traversal["all_legs_valid"]),
            "no_illegal_drive_up": not bool(traversal["any_illegal_drive_up"]),
            "attitude_safe": attitude_safe,
            "joint_limits_safe": joint_limit_safe,
            "collision_safe": collision_safe,
            "contact_drift_safe": bool(
                self.maximum_contact_drift_m <= self.maximum_allowed_contact_drift_m
            ),
            "final_all_top": final_all_top,
            "final_all_loaded": final_all_loaded,
            "final_velocity_stable": final_stable,
        }
        physical_success = bool(evidence_complete and all(criteria.values()))
        return {
            "source_version": self.source_version,
            "sample_count": len(self.fsm50_rows),
            "contact_force_valid": contact_force_valid,
            "filtered_contact_valid": filtered_contact_valid,
            "filtered_contact_error_count": len(self.filtered_contact_errors),
            "filtered_contact_errors": list(self.filtered_contact_errors),
            "evidence_complete": evidence_complete,
            "not_evaluable_reasons": not_evaluable_reasons,
            "strict_criteria": criteria,
            "traversal": traversal,
            "final_wheel_contact_classes": final_classes,
            "final_all_top": final_all_top,
            "final_all_loaded": final_all_loaded,
            "maximum_contact_drift_m": self.maximum_contact_drift_m,
            "minimum_two_leg_corridor_margin_m": None
            if not math.isfinite(self.minimum_corridor_margin_m)
            else self.minimum_corridor_margin_m,
            "dangerous_collision_count": len(self.dangerous_collision_rows),
            "collision_evidence_valid": collision_valid,
            "joint_limit_evidence_valid": joint_limit_evidence_valid,
            "joint_limit_violation_count": len(self.joint_limit_violation_rows),
            "maximum_abs_roll_rad": maximum_abs_roll,
            "maximum_abs_pitch_rad": maximum_abs_pitch,
            "maximum_angular_velocity_rad_s": maximum_angular_speed,
            "final_velocity_stable": final_stable,
            "physical_success": physical_success,
        }

    def flush(self) -> None:
        super().flush()
        if self.run_dir is None:
            return
        write_csv(self.run_dir / "fsm50_telemetry.csv", self.fsm50_rows)
        write_jsonl(self.run_dir / "fsm50_telemetry.jsonl", self.fsm50_rows)
        write_jsonl(
            self.run_dir / "wheel_filtered_contacts.jsonl",
            self.filtered_surface_rows,
        )
        write_csv(
            self.run_dir / "nonwheel_obstacle_contacts.csv",
            self.nonwheel_obstacle_rows,
        )
        write_jsonl(
            self.run_dir / "nonwheel_obstacle_contacts.jsonl",
            self.nonwheel_obstacle_rows,
        )
        write_csv(self.run_dir / "state_timeline.csv", self.state_timeline_rows)
        write_json(self.run_dir / "physical_evidence.json", self.physical_evidence())
