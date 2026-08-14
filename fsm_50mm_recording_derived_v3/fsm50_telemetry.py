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
from .grounding_diagnostics import (
    penetration_snapshot as wheel_ground_aabb_proxy_snapshot,
)
from .nonwheel_obstacle_contact import (
    NONWHEEL_FORCE_SOURCE,
    nonwheel_obstacle_contact_rows,
)
from .physx_contact_separation import PHYSX_SEPARATION_SOURCE
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


def _time_window_numeric_guard_s(
    rows: list[dict[str, Any]],
    *,
    final_time_s: float,
    cutoff_time_s: float,
) -> float:
    """Bound timestamp roundoff accumulated over the recorded physics ticks."""

    if not math.isfinite(final_time_s) or not math.isfinite(cutoff_time_s):
        return 0.0
    last = rows[-1] if rows else {}
    physics_dt_s = _safe_float(last.get("physics_dt_s"))
    sim_step = _safe_float(last.get("sim_step"))
    accumulated_steps = max(1, len(rows) - 1)
    if math.isfinite(sim_step):
        accumulated_steps = max(accumulated_steps, abs(int(sim_step)))
    elif math.isfinite(physics_dt_s) and physics_dt_s > 0.0:
        accumulated_steps = max(
            accumulated_steps,
            int(math.ceil(abs(final_time_s) / physics_dt_s)),
        )

    # Each repeated dt addition can round by at most one ulp at the current
    # timestamp magnitude.  The extra endpoint ulps cover the final subtraction
    # used to form cutoff_time_s.  This is a representation-error bound, not a
    # physical timing tolerance.
    timestamp_ulp = max(math.ulp(final_time_s), math.ulp(cutoff_time_s))
    guard_s = float(accumulated_steps + 2) * timestamp_ulp
    if math.isfinite(physics_dt_s) and physics_dt_s > 0.0:
        guard_s += float(accumulated_steps) * math.ulp(physics_dt_s)
    return guard_s


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
        contact_mode: str,
        environment_equivalence_role: str = "",
        diagnostic_role: str = "",
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
        maximum_penetration_m: float = 0.003,
    ) -> None:
        super().__init__(config, args=args, scene_handle=scene_handle)
        # The base sample must not checkpoint before this collector has added
        # the same frame's FSM evidence.  ``on_step`` performs the shared
        # cadence check after the extended row and timeline are complete.
        self.defer_periodic_checkpoint = True
        self.obstacle = obstacle
        self.wheel_radius_m = float(wheel_radius_m)
        self.source_version = str(source_version)
        normalized_contact_mode = str(contact_mode).strip().lower()
        if normalized_contact_mode not in {"disabled", "formal", "instrumented"}:
            raise ValueError(
                "contact_mode must be explicitly set to disabled, formal, or instrumented"
            )
        self.contact_mode = normalized_contact_mode
        normalized_role = str(environment_equivalence_role or "").strip().upper()
        if normalized_role not in {"", "A1", "A2", "B"}:
            raise ValueError(
                "environment_equivalence_role must be empty, A1, A2, or B"
            )
        if normalized_role in {"A1", "A2"} and normalized_contact_mode != "formal":
            raise ValueError("A1/A2 evidence requires contact_mode=formal")
        if normalized_role == "B" and normalized_contact_mode != "instrumented":
            raise ValueError("B evidence requires contact_mode=instrumented")
        normalized_diagnostic_role = str(diagnostic_role or "").strip().upper()
        if normalized_diagnostic_role not in {"", "U"}:
            raise ValueError("diagnostic_role must be empty or U")
        if normalized_diagnostic_role == "U":
            if normalized_role:
                raise ValueError(
                    "U diagnostic evidence cannot carry an environment-equivalence role"
                )
            if normalized_contact_mode != "disabled":
                raise ValueError("U diagnostic evidence requires contact_mode=disabled")
        elif normalized_contact_mode == "disabled":
            raise ValueError("contact_mode=disabled requires diagnostic_role=U")
        self.environment_equivalence_role = normalized_role
        self.diagnostic_role = normalized_diagnostic_role
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
        self.maximum_penetration_m = float(maximum_penetration_m)
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
        self.persistence = ContactPersistenceTracker(
            load_confirm_force_n=load_confirm_force_n
        )
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
        self.integrated_wheel_force_impulse_w: dict[str, list[float]] = {
            leg: [0.0, 0.0, 0.0] for leg in LEGS
        }
        self.integrated_wheel_upward_impulse_n_s: dict[str, float] = {
            leg: 0.0 for leg in LEGS
        }
        self.initial_com_position_w: tuple[float, float, float] | None = None
        self.previous_sample_time_s: float | None = None
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
        extended = self._extend_row(adapter, base_row, contacts, dt_s=float(dt_s))
        filtered_rows = list(extended.get("wheel_filtered_contacts", []) or [])
        filtered_evidence = self._install_filtered_contact_sample(
            base_row=base_row,
            filtered_rows=filtered_rows,
            before_contacts=before_contacts,
            qualification_row=extended,
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
        self._maybe_checkpoint(
            float(base_row.get("time_s", getattr(adapter, "sim_time", 0.0)) or 0.0)
        )

    @staticmethod
    def _filtered_row_identity_valid(
        row: Mapping[str, Any],
        *,
        leg: str,
        surface: str,
        filter_index: int,
        other_prim_path: str,
    ) -> bool:
        """Validate the exact wheel/filter identity before trusting its label."""

        def normalized_path(value: Any) -> str:
            return str(value or "").replace("\\", "/").rstrip("/")

        wheel_body = LEG_TO_WHEEL_BODY[leg]
        wheel_prim_path = normalized_path(row.get("wheel_prim_path"))
        filter_value = row.get("filter_index")
        return bool(
            str(row.get("leg", "")).upper() == leg
            and str(row.get("surface", "")) == surface
            and isinstance(filter_value, int)
            and not isinstance(filter_value, bool)
            and filter_value == filter_index
            and str(row.get("wheel_body_name", "")) == wheel_body
            and wheel_prim_path == f"/World/WLRRobot/{wheel_body}"
            and normalized_path(row.get("other_prim_path"))
            == normalized_path(other_prim_path)
            and str(row.get("source", "")) == FILTERED_FORCE_SOURCE
        )

    @classmethod
    def _exact_filtered_surface_evidence_by_leg(
        cls,
        rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]] | None:
        """Build the two exact, identity-checked surface rows for every leg.

        ``None`` means the filtered bank is unavailable, preserving the common
        aggregate/geometry fallback.  A non-empty but malformed layout remains
        available and emits ``identity_valid=False`` so classification fails
        closed instead of silently taking that fallback.
        """

        if not rows:
            return None
        expected_pairs = {
            (leg, surface)
            for leg in LEGS
            for surface, _other_prim_path in FILTERED_SURFACES
        }
        actual_pairs = [
            (str(row.get("leg", "")).upper(), str(row.get("surface", "")))
            for row in rows
        ]
        pair_layout_valid = bool(
            len(rows) == len(expected_pairs)
            and len(set(actual_pairs)) == len(actual_pairs)
            and set(actual_pairs) == expected_pairs
        )
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for leg in LEGS:
            result[leg] = {}
            for filter_index, (surface, other_prim_path) in enumerate(
                FILTERED_SURFACES
            ):
                matches = [
                    row
                    for row in rows
                    if str(row.get("leg", "")).upper() == leg
                    and str(row.get("surface", "")) == surface
                ]
                row = matches[0] if len(matches) == 1 else {}
                result[leg][surface] = {
                    "leg": str(row.get("leg", "")),
                    "surface": str(row.get("surface", "")),
                    "identity_valid": bool(
                        pair_layout_valid
                        and len(matches) == 1
                        and cls._filtered_row_identity_valid(
                            row,
                            leg=leg,
                            surface=surface,
                            filter_index=filter_index,
                            other_prim_path=other_prim_path,
                        )
                    ),
                    "force_valid": row.get("force_valid"),
                    "contact_point_valid": row.get("contact_point_valid"),
                    "active": row.get("active"),
                    "normal_force_n": row.get("normal_force_n"),
                    "source": row.get("source"),
                }
        return result

    @classmethod
    def _filtered_layout_valid(cls, rows: list[dict[str, Any]]) -> bool:
        expected = {
            (leg, surface)
            for leg in LEGS
            for surface, _prim_path in FILTERED_SURFACES
        }
        actual = {
            (str(row.get("leg", "")), str(row.get("surface", "")))
            for row in rows
        }
        if len(rows) != len(expected) or actual != expected:
            return False
        evidence = cls._exact_filtered_surface_evidence_by_leg(rows)
        return bool(
            evidence is not None
            and all(
                row.get("identity_valid") is True
                for surfaces in evidence.values()
                for row in surfaces.values()
            )
        )

    def _install_filtered_contact_sample(
        self,
        *,
        base_row: dict[str, Any],
        filtered_rows: list[dict[str, Any]],
        before_contacts: int,
        qualification_row: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replace generic contacts with the eight labelled wheel/surface rows."""

        layout_valid = self._filtered_layout_valid(filtered_rows)
        if not layout_valid:
            return {
                "filtered_contact_layout_valid": False,
                "filtered_contact_force_valid": False,
                "filtered_contact_geometry_valid": False,
                "filtered_contact_consistency_valid": False,
                "filtered_contact_consistency_error": (
                    "filtered wheel/surface layout is incomplete"
                ),
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
        consistency_valid, consistency_error = self._filtered_contact_consistency(
            qualification_row, filtered_rows
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
            "filtered_contact_geometry_valid": bool(
                geometry_valid and consistency_valid
            ),
            "filtered_contact_consistency_valid": consistency_valid,
            "filtered_contact_consistency_error": consistency_error,
            "active_contact_count": len(active_rows),
            "active_contact_count_label": f"{len(active_rows)} (sensor-confirmed)",
            "contact_geometry_source": FILTERED_GEOMETRY_SOURCE,
            "contact_force_source": FILTERED_FORCE_SOURCE,
            "contact_force_valid": force_valid,
            "contact_force_reason": force_reason,
            "max_contact_force_n": max_force,
            "max_contact_force_n_source": FILTERED_FORCE_SOURCE,
        }

    def _filtered_contact_consistency(
        self,
        qualification_row: Mapping[str, Any] | None,
        filtered_rows: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """Require a loaded common contact to exist in its exact filter pair."""

        if qualification_row is None:
            return True, ""
        classes = dict(qualification_row.get("wheel_contact_classes", {}) or {})
        upward = dict(qualification_row.get("wheel_contact_force_up_n", {}) or {})
        total = dict(qualification_row.get("wheel_contact_force_total_n", {}) or {})
        errors: list[str] = []
        for leg in LEGS:
            contact_class = str(classes.get(leg, "") or "")
            surface = (
                "ground"
                if contact_class == ContactClass.GROUND.value
                else "obstacle"
                if contact_class
                in {ContactClass.FRONT_FACE.value, ContactClass.TOP.value}
                else ""
            )
            if not surface:
                continue
            common_load_values = [
                value
                for value in (
                    _safe_float(upward.get(leg)),
                    _safe_float(total.get(leg)),
                )
                if math.isfinite(value)
            ]
            if not common_load_values:
                continue
            common_load = max(common_load_values)
            if common_load < self.force_threshold_n:
                continue
            matching = [
                row
                for row in filtered_rows
                if str(row.get("leg", "")) == leg
                and str(row.get("surface", "")) == surface
            ]
            if len(matching) != 1:
                errors.append(
                    f"{leg}/{surface}: expected one filtered pair, got {len(matching)}"
                )
                continue
            row = matching[0]
            filtered_force = _safe_float(row.get("normal_force_n"))
            if not (
                row.get("force_valid") is True
                and row.get("active") is True
                and math.isfinite(filtered_force)
                and filtered_force >= self.force_threshold_n
                and row.get("contact_point_valid") is True
            ):
                errors.append(
                    f"{leg}/{surface}: common_load={common_load:.9g} N but "
                    "filtered force/contact-point evidence is zero or invalid"
                )
        return not errors, "; ".join(errors)

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
                    "vector_w": [float("nan")] * 3,
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
                "vector_w": [float(value) for value in vector],
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

    @staticmethod
    def _measured_contact_point_validity(
        classified: Mapping[str, Any],
        filtered_contacts: list[dict[str, Any]],
    ) -> dict[str, bool]:
        """Report whether each installed point came from live sensor geometry.

        ``classify_wheel_contact`` always provides a geometric projection.  It
        must not be confused with the measured ``contact_pos_w`` required by
        anchored-drift evidence, so validity is reconstructed from the same
        filtered-row selection contract used by
        :meth:`_install_measured_contact_points`.
        """

        validity = {leg: False for leg in LEGS}

        def finite_point(row: Mapping[str, Any]) -> bool:
            try:
                point = np.asarray(row.get("contact_point_w", []), dtype=float).reshape(-1)
            except Exception:
                return False
            return bool(point.size >= 3 and np.isfinite(point[:3]).all())

        for leg in LEGS:
            contact = classified.get(leg)
            if contact is None:
                continue
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
                and row.get("active") is True
                and row.get("contact_point_valid") is True
                and finite_point(row)
            ]
            preferred = [
                row
                for row in candidates
                if str(row.get("surface", "")) == desired_surface
            ]
            validity[leg] = bool(preferred or candidates)
        return validity

    def _physx_separation_sample(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Read the cached exact-pair signed separation evidence fail closed."""

        sensor = getattr(self.scene_handle, "contact_sensor", None)
        unavailable = {
            "schema_version": "fsm50.physx_contact_separation.v1",
            "valid": False,
            "status": "UNKNOWN",
            "pair_count": 0,
            "pair_ids": [],
            "unknown_pair_ids": [],
            "maximum_physx_penetration_m": None,
            "maximum_by_scope_m": None,
            "source": PHYSX_SEPARATION_SOURCE,
            "errors": [],
        }
        # Qualification needs wheel-ground, wheel-obstacle, and every
        # non-wheel-obstacle pair.  A wheel-only bank is deliberately not
        # accepted as whole-robot penetration evidence.
        if not (
            bool(getattr(sensor, "is_filtered_wheel_contact_bank", False))
            and bool(getattr(sensor, "is_nonwheel_obstacle_contact_bank", False))
        ):
            unavailable["errors"] = [
                "combined wheel/non-wheel exact-pair PhysX contact views are unavailable"
            ]
            return [], unavailable
        observations = getattr(sensor, "separation_observations", None)
        evidence_reader = getattr(sensor, "separation_evidence", None)
        if not callable(observations) or not callable(evidence_reader):
            unavailable["errors"] = [
                "contact sensor does not expose signed-separation observations/evidence"
            ]
            return [], unavailable
        try:
            rows = [dict(row) for row in list(observations(env_id=0) or [])]
            evidence = dict(evidence_reader(env_id=0) or {})
        except Exception as exc:
            unavailable["errors"] = [
                f"signed-separation read failed: {type(exc).__name__}: {exc}"
            ]
            return [], unavailable
        maximum = _safe_float(evidence.get("maximum_physx_penetration_m"))
        pair_ids = [str(row.get("pair_id", "") or "") for row in rows]
        reported_pair_ids = [
            str(value) for value in list(evidence.get("pair_ids", []) or [])
        ]
        row_maxima = [_safe_float(row.get("maximum_penetration_m")) for row in rows]
        computed_maximum = max(row_maxima, default=float("nan"))
        try:
            reported_pair_count = int(evidence.get("pair_count", -1))
        except (TypeError, ValueError):
            reported_pair_count = -1
        structurally_valid = bool(
            evidence.get("schema_version") == "fsm50.physx_contact_separation.v1"
            and evidence.get("valid") is True
            and evidence.get("status") == "AVAILABLE"
            and evidence.get("source") == PHYSX_SEPARATION_SOURCE
            and rows
            and all(row.get("valid") is True for row in rows)
            and all(pair_ids)
            and len(pair_ids) == len(set(pair_ids))
            and reported_pair_ids == pair_ids
            and reported_pair_count == len(rows)
            and not list(evidence.get("unknown_pair_ids", []) or [])
            and all(math.isfinite(value) and value >= 0.0 for value in row_maxima)
            and math.isfinite(maximum)
            and maximum >= 0.0
            and maximum == computed_maximum
        )
        if not structurally_valid:
            evidence["valid"] = False
            evidence["status"] = "UNKNOWN"
            evidence["maximum_physx_penetration_m"] = None
            errors = list(evidence.get("errors", []) or [])
            if not errors:
                errors.append("signed-separation evidence is incomplete or non-finite")
            evidence["errors"] = errors
        return rows, evidence

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
        *,
        dt_s: float | None = None,
    ) -> dict[str, Any]:
        time_s = float(base_row.get("time_s", 0.0) or 0.0)
        body_positions = self._body_positions(adapter)
        joint_pos, joint_vel = self._joint_vectors(adapter)
        drive_evidence: dict[str, Any] = {}
        drive_evidence_error = ""
        try:
            capture = getattr(adapter, "capture_joint_drive_evidence")
            drive_evidence = dict(capture() or {})
            drive_evidence_error = str(drive_evidence.get("error", "") or "")
        except Exception as exc:
            drive_evidence_error = (
                "joint/PhysX drive evidence unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
        if drive_evidence:
            joint_pos = dict(drive_evidence.get("joint_position_by_name", {}) or {})
            joint_vel = dict(drive_evidence.get("joint_velocity_by_name", {}) or {})
        actual_target = dict(
            drive_evidence.get("joint_position_target_by_name", {}) or {}
        )
        buffered_position_target = dict(
            drive_evidence.get("joint_position_target_buffer_by_name", {}) or {}
        )
        physx_velocity_target = dict(
            drive_evidence.get("joint_velocity_target_by_name", {}) or {}
        )
        buffered_velocity_target = dict(
            drive_evidence.get("joint_velocity_target_buffer_by_name", {}) or {}
        )
        servo_command_target_rad = dict(
            drive_evidence.get("servo_command_target_by_name", {}) or {}
        )
        servo_command_to_physx_error = dict(
            drive_evidence.get("servo_command_to_readback_error_by_name", {}) or {}
        )
        target_minus_position = dict(
            drive_evidence.get("joint_target_minus_position_by_name", {}) or {}
        )
        drive_evidence_valid = bool(drive_evidence.get("valid", False))
        filtered_contacts, filtered_contact_error = self._filtered_contacts()
        filtered_surface_evidence_by_leg = (
            self._exact_filtered_surface_evidence_by_leg(filtered_contacts)
        )
        (
            nonwheel_contacts,
            collision_evidence_valid,
            dangerous_collision,
            collision_evidence_error,
        ) = self._nonwheel_collision_sample(time_s)
        forces, common_force_evidence = self._common_wheel_net_force_by_leg(
            getattr(self.scene_handle, "contact_sensor", None)
        )
        wheel_net_forces_w = {
            leg: list(forces[leg].get("vector_w", [float("nan")] * 3))
            for leg in LEGS
        }
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
                filtered_surface_evidence=(
                    None
                    if filtered_surface_evidence_by_leg is None
                    else filtered_surface_evidence_by_leg[leg]
                ),
            )
            for leg, observation in wheel_observations.items()
        }
        measured_contact_point_valid = self._measured_contact_point_validity(
            classified, filtered_contacts
        )
        classified = self._install_measured_contact_points(classified, filtered_contacts)
        wheel_command = dict(getattr(adapter, "wheel_speeds", {}) or {})
        logical_wheel_target_by_leg = {
            leg: _safe_float(wheel_command.get(LEG_TO_WHEEL_JOINT[leg]))
            for leg in LEGS
        }
        physx_wheel_target_by_leg = {
            leg: _safe_float(physx_velocity_target.get(LEG_TO_WHEEL_JOINT[leg]))
            for leg in LEGS
        }
        wheel_drive_evidence_valid = {
            leg: bool(
                drive_evidence_valid
                and math.isfinite(logical_wheel_target_by_leg[leg])
                and math.isfinite(physx_wheel_target_by_leg[leg])
            )
            for leg in LEGS
        }
        persistence, drift = self.persistence.update(
            time_s,
            classified,
            logical_wheel_target_rad_s=logical_wheel_target_by_leg,
            physx_wheel_target_rad_s=physx_wheel_target_by_leg,
            measured_contact_point_valid=measured_contact_point_valid,
            drive_evidence_valid=wheel_drive_evidence_valid,
        )
        finite_zero_target_contact_displacement = [
            float(value) for value in drift.values() if math.isfinite(_safe_float(value))
        ]
        if finite_zero_target_contact_displacement:
            self.maximum_contact_drift_m = max(
                self.maximum_contact_drift_m,
                max(finite_zero_target_contact_displacement),
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

        physics_dt = _safe_float(dt_s)
        observed_sample_dt = float("nan")
        if self.previous_sample_time_s is not None and math.isfinite(time_s):
            observed_dt = float(time_s) - float(self.previous_sample_time_s)
            if observed_dt > 0.0 and math.isfinite(observed_dt):
                observed_sample_dt = observed_dt
        impulse_dt = (
            observed_sample_dt
            if math.isfinite(observed_sample_dt)
            else physics_dt
        )
        impulse_evidence_valid = bool(
            common_force_evidence.get("wheel_net_force_valid") is True
            and math.isfinite(impulse_dt)
            and impulse_dt > 0.0
            and all(
                len(wheel_net_forces_w[leg]) == 3
                and all(
                    math.isfinite(_safe_float(value))
                    for value in wheel_net_forces_w[leg]
                )
                for leg in LEGS
            )
        )
        if impulse_evidence_valid:
            for leg in LEGS:
                vector = wheel_net_forces_w[leg]
                for axis in range(3):
                    self.integrated_wheel_force_impulse_w[leg][axis] += (
                        float(vector[axis]) * impulse_dt
                    )
                self.integrated_wheel_upward_impulse_n_s[leg] += (
                    max(0.0, float(vector[2])) * impulse_dt
                )
        self.previous_sample_time_s = float(time_s) if math.isfinite(time_s) else None

        com_position_w = [
            _safe_float(base_row.get("com_x_m")),
            _safe_float(base_row.get("com_y_m")),
            _safe_float(base_row.get("com_z_m")),
        ]
        if self.initial_com_position_w is None and all(
            math.isfinite(value) for value in com_position_w
        ):
            self.initial_com_position_w = tuple(com_position_w)
        com_displacement_w = (
            [
                float(com_position_w[index] - self.initial_com_position_w[index])
                for index in range(3)
            ]
            if self.initial_com_position_w is not None
            and all(math.isfinite(value) for value in com_position_w)
            else [float("nan")] * 3
        )
        self.traversal.update(time_s, classified, canonical_angles)

        command_deg = {
            name: _safe_float(dict(getattr(adapter, "joint_command_deg", {}) or {}).get(name))
            for name in SERVO_JOINT_NAMES
        }
        joint_safety = self._joint_safety_evidence(
            adapter, joint_pos, actual_target
        )
        segment = self._segment_context()
        try:
            wheel_ground_aabb_proxy = wheel_ground_aabb_proxy_snapshot(adapter)
        except Exception as exc:
            wheel_ground_aabb_proxy = {
                "valid": False,
                "error": f"AABB proxy capture failed: {type(exc).__name__}: {exc}",
                "maximum_collision_penetration_m": float("nan"),
                "wheel_penetration_m": {},
            }
        wheel_ground_aabb_proxy = {
            **wheel_ground_aabb_proxy,
            "evidence_class": "wheel_ground_aabb_proxy",
            "physical_qualification_eligible": False,
        }
        separation_observations, separation_evidence = (
            self._physx_separation_sample()
        )
        root_pose_w = [
            _safe_float(base_row.get(key))
            for key in (
                "base_x_m",
                "base_y_m",
                "base_z_m",
                "base_qw",
                "base_qx",
                "base_qy",
                "base_qz",
            )
        ]
        root_velocity_w = [
            _safe_float(base_row.get(key))
            for key in (
                "base_vx_m_s",
                "base_vy_m_s",
                "base_vz_m_s",
                "base_wx_rad_s",
                "base_wy_rad_s",
                "base_wz_rad_s",
            )
        ]
        row: dict[str, Any] = {
            **dict(base_row),
            "source_version": self.source_version,
            "fsm_state": str(self.runtime_context.get("fsm_state", "")),
            "scheduler_phase": str(self.runtime_context.get("scheduler_phase", "")),
            "macro_state_cursor": str(
                self.runtime_context.get("macro_state_cursor", "RECORDING_FAST_REPLAY")
            ),
            "command_cursor": self.runtime_context.get("command_cursor"),
            "segment_cursor": self.runtime_context.get("segment_cursor"),
            "source_command": str(self.runtime_context.get("source_command", "")),
            "source_event_index": self.runtime_context.get("source_event_index"),
            "planned_dispatch_time_s": self.runtime_context.get(
                "planned_dispatch_time_s"
            ),
            "actual_dispatch_time_s": self.runtime_context.get(
                "actual_dispatch_time_s"
            ),
            "atomic_batch_id": str(self.runtime_context.get("atomic_batch_id", "")),
            "dispatch_kind": str(self.runtime_context.get("dispatch_kind", "")),
            "motion_start_readiness_token": str(
                self.runtime_context.get("motion_start_readiness_token", "")
            ),
            **segment,
            # This is the authoritative simulator physics step supplied by the
            # caller.  Telemetry cadence belongs in ``observed_sample_dt_s``;
            # replacing physics_dt_s with a timestamp delta corrupts exact
            # wheel-target integration evidence.
            "physics_dt_s": physics_dt,
            "observed_sample_dt_s": observed_sample_dt,
            "root_pose_w": root_pose_w,
            "root_position_w": root_pose_w[:3],
            "root_orientation_wxyz": root_pose_w[3:7],
            "root_linear_velocity_w": root_velocity_w[:3],
            "root_angular_velocity_w": root_velocity_w[3:6],
            "com_position_w": com_position_w,
            "com_velocity_w": [
                _safe_float(base_row.get("com_vx_m_s")),
                _safe_float(base_row.get("com_vy_m_s")),
                _safe_float(base_row.get("com_vz_m_s")),
            ],
            "com_displacement_w": com_displacement_w,
            "command_space_servo_target_deg": command_deg,
            "servo_command_target_rad": servo_command_target_rad,
            "actual_joint_target_rad": actual_target,
            "physx_joint_position_target_rad": actual_target,
            "joint_position_target_buffer_rad": buffered_position_target,
            "physx_joint_velocity_target_rad_s": physx_velocity_target,
            "joint_velocity_target_buffer_rad_s": buffered_velocity_target,
            "joint_target_minus_position_rad": target_minus_position,
            "servo_command_to_physx_target_error_rad": servo_command_to_physx_error,
            "physx_drive_target_evidence_valid": drive_evidence_valid,
            "physx_drive_target_evidence_error": drive_evidence_error,
            "measured_joint_position_rad": joint_pos,
            "measured_joint_velocity_rad_s": joint_vel,
            "wheel_command_velocity_rad_s": wheel_command,
            "wheel_canonical_forward_angle_rad": canonical_angles,
            "wheel_canonical_forward_velocity_rad_s": canonical_velocities,
            "wheel_forward_sign": {
                name: float(WHEEL_FORWARD_SIGN[name]) for name in WHEEL_JOINT_NAMES
            },
            "wheel_forward_sign_by_leg": dict(self.wheel_forward_sign),
            "wheel_direction": _wheel_sign(
                wheel_direction,
                label="adapter.wheel_direction",
            ),
            "wheel_integrated_rotation_rad": dict(self.integrated_wheel_rotation),
            "wheel_integrated_travel_m": dict(self.integrated_wheel_travel),
            "wheel_legacy_simple_integration_authoritative": False,
            "wheel_legacy_simple_integration_role": "diagnostic_only",
            "wheel_legacy_simple_integration_reason": (
                "authoritative target/measured evidence is emitted by "
                "V003_WHEEL_INTEGRAL_EVIDENCE.json"
            ),
            "wheel_net_forces_w": wheel_net_forces_w,
            "wheel_force_impulse_w_n_s": {
                leg: list(values)
                for leg, values in self.integrated_wheel_force_impulse_w.items()
            },
            "wheel_upward_force_impulse_n_s": dict(
                self.integrated_wheel_upward_impulse_n_s
            ),
            "force_impulse_evidence_valid": impulse_evidence_valid,
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
            "wheel_filtered_surface_identity_valid_by_leg": {
                leg: bool(
                    filtered_surface_evidence_by_leg is not None
                    and all(
                        row.get("identity_valid") is True
                        for row in filtered_surface_evidence_by_leg[leg].values()
                    )
                )
                for leg in LEGS
            },
            "wheel_filtered_contacts": filtered_contacts,
            "wheel_surface_normal_force_n": surface_loads,
            "wheel_contact_persistence_s": persistence,
            "wheel_contact_drift_m": drift,
            "wheel_contact_persistence_mode": {
                leg: self.persistence.last_samples[leg].mode.value
                for leg in LEGS
            },
            "wheel_contact_persistence_evidence_valid": {
                leg: self.persistence.last_samples[leg].evidence_valid
                for leg in LEGS
            },
            "wheel_contact_persistence_reason": {
                leg: self.persistence.last_samples[leg].reason for leg in LEGS
            },
            "wheel_contact_persistence_diagnostic": {
                leg: self.persistence.last_samples[leg].as_dict() for leg in LEGS
            },
            "wheel_logical_target_rad_s_by_leg": logical_wheel_target_by_leg,
            "wheel_physx_target_rad_s_by_leg": physx_wheel_target_by_leg,
            "wheel_drive_evidence_valid_by_leg": wheel_drive_evidence_valid,
            "wheel_measured_contact_point_valid_by_leg": measured_contact_point_valid,
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
            "maximum_contact_drift_m_so_far": (
                max(
                    self.persistence.maximum_zero_target_contact_point_displacement_m.values()
                )
                if self.persistence.maximum_zero_target_contact_point_displacement_m
                else None
            ),
            "maximum_zero_target_contact_point_displacement_m_so_far": (
                max(
                    self.persistence.maximum_zero_target_contact_point_displacement_m.values()
                )
                if self.persistence.maximum_zero_target_contact_point_displacement_m
                else None
            ),
            "contact_point_displacement_semantics": (
                "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
            ),
            "physical_anchoring_proven": False,
            "material_point_identity_available": False,
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
            "contact_mode": self.contact_mode,
            "environment_equivalence_role": str(
                getattr(self, "environment_equivalence_role", "") or ""
            ),
            "diagnostic_role": str(
                getattr(self, "diagnostic_role", "") or ""
            ),
            "physx_signed_separation_observations": separation_observations,
            "physx_signed_separation_evidence": separation_evidence,
            "penetration": separation_evidence,
            "penetration_evidence_source": separation_evidence.get("source"),
            "penetration_evidence_valid": separation_evidence.get("valid") is True,
            "penetration_evidence_error": "; ".join(
                str(value) for value in list(separation_evidence.get("errors", []) or [])
            ),
            "maximum_collision_penetration_m": separation_evidence.get(
                "maximum_physx_penetration_m"
            ),
            "wheel_ground_aabb_proxy": wheel_ground_aabb_proxy,
            "wheel_ground_aabb_proxy_valid": (
                wheel_ground_aabb_proxy.get("valid") is True
            ),
            "wheel_ground_aabb_proxy_maximum_overlap_m": (
                wheel_ground_aabb_proxy.get("maximum_collision_penetration_m")
            ),
        }
        if str(getattr(self, "diagnostic_role", "") or "").upper() == "U":
            # U is a kinematic trajectory capture in the ordinary production
            # scene.  Keep joint/root/wheel-center measurements, but erase all
            # derived contact assertions so geometry cannot masquerade as a
            # disabled ContactSensor observation.
            row.update(
                wheel_net_forces_w={leg: [None, None, None] for leg in LEGS},
                wheel_contact_force_up_n={leg: None for leg in LEGS},
                wheel_contact_force_total_n={leg: None for leg in LEGS},
                wheel_net_force_valid=False,
                wheel_net_force_layout_valid=False,
                wheel_net_force_error="contact sensors deliberately disabled for U diagnostic",
                wheel_contact_classes={leg: ContactClass.UNKNOWN.value for leg in LEGS},
                wheel_contact_confidence={leg: "UNAVAILABLE_SENSOR_DISABLED" for leg in LEGS},
                wheel_contact_points_w={leg: [None, None, None] for leg in LEGS},
                wheel_measured_contact_point_valid_by_leg={leg: False for leg in LEGS},
                wheel_filtered_contacts=[],
                filtered_contact_available=False,
                filtered_contact_error="contact sensors deliberately disabled for U diagnostic",
                filtered_contact_layout_valid=False,
                filtered_contact_force_valid=False,
                filtered_contact_geometry_valid=False,
                filtered_contact_consistency_valid=False,
                nonwheel_obstacle_contacts=[],
                collision_evidence_source="",
                collision_evidence_valid=False,
                collision_evidence_error="contact sensors deliberately disabled for U diagnostic",
                dangerous_collision=None,
                force_impulse_evidence_valid=False,
                support_legs=[],
                light_support_legs=[],
                persistent_support_legs=[],
                stable_contact_legs=[],
                primary_diagonal="NONE",
                wheel_support_polygon=[],
                wheel_support_polygon_valid=False,
                wheel_support_polygon_degenerate=True,
                wheel_support_polygon_margin_m=None,
                two_leg_corridor_distance_m=None,
                two_leg_corridor_fraction=None,
                two_leg_corridor_valid=False,
                physx_signed_separation_observations=[],
                penetration_evidence_source="",
                penetration_evidence_valid=False,
                penetration_evidence_error="contact sensors deliberately disabled for U diagnostic",
                maximum_collision_penetration_m=None,
            )
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
            and bool(row.get("filtered_contact_consistency_valid", False))
            for row in rows
        )

    def physical_evidence(self) -> dict[str, Any]:
        if str(getattr(self, "diagnostic_role", "") or "").upper() == "U":
            criterion_scopes = {
                "contact_evidence_valid": "INSTRUMENTED_ONLY",
                "all_legs_linkage_lift_valid": "COMMON",
                "no_illegal_drive_up": "COMMON",
                "attitude_safe": "COMMON",
                "joint_limits_safe": "COMMON",
                "collision_safe": "INSTRUMENTED_ONLY",
                "penetration_safe": "COMMON",
                "contact_drift_safe": "INSTRUMENTED_ONLY",
                "final_all_top": "COMMON",
                "final_all_loaded": "COMMON",
                "final_velocity_stable": "COMMON",
            }
            reason = (
                "ordinary-UI U diagnostic deliberately disables all contact "
                "sensors and is not physical-qualification evidence"
            )
            records = {
                name: {
                    "name": name,
                    "scope": scope,
                    "availability": "UNAVAILABLE_BY_ROLE",
                    "passed": None,
                    "reason": reason,
                }
                for name, scope in criterion_scopes.items()
            }
            unavailable = [f"{name}: {reason}" for name in records]
            return {
                "schema_version": "fsm50.physical_evidence.v2",
                "source_version": self.source_version,
                "contact_mode": "disabled",
                "environment_equivalence_role": "",
                "diagnostic_role": "U",
                "sample_count": len(self.fsm50_rows),
                "contact_force_valid": False,
                "common_contact_evidence_valid": False,
                "filtered_contact_valid": False,
                "filtered_contact_error_count": 0,
                "filtered_contact_errors": [],
                "evidence_complete": False,
                "not_evaluable_reasons": unavailable,
                "criteria": records,
                "criterion_records": list(records.values()),
                "strict_criteria": {name: None for name in records},
                "role_capture_verdict": "NOT_EVALUABLE",
                "full_physical_verdict": "NOT_EVALUABLE",
                "role_capture_verdict_reasons": unavailable,
                "full_physical_verdict_reasons": unavailable,
                "traversal": {
                    "evidence_available": False,
                    "all_legs_valid": False,
                    "any_illegal_drive_up": False,
                    "legs": {},
                },
                "final_wheel_contact_classes": {},
                "final_all_top": False,
                "final_all_loaded": False,
                "maximum_contact_drift_m": None,
                "minimum_two_leg_corridor_margin_m": None,
                "dangerous_collision_count": 0,
                "collision_evidence_valid": False,
                "physx_drive_target_evidence_valid": False,
                "penetration_evidence_valid": False,
                "penetration_evidence_source": "",
                "maximum_collision_penetration_m": None,
                "maximum_allowed_penetration_m": self.maximum_penetration_m,
                "force_impulse_evidence_valid": False,
                "final_wheel_force_impulse_w_n_s": {},
                "final_wheel_upward_force_impulse_n_s": {},
                "joint_limit_evidence_valid": False,
                "joint_limit_violation_count": 0,
                "maximum_abs_roll_rad": None,
                "maximum_abs_pitch_rad": None,
                "maximum_angular_velocity_rad_s": None,
                "final_velocity_stable": False,
                "contact_drift_evidence_valid": False,
                "zero_target_loaded_contact_displacement_sample_count": 0,
                "contact_point_displacement_semantics": (
                    "UNAVAILABLE_BY_ROLE"
                ),
                "physical_anchoring_proven": False,
                "material_point_identity_available": False,
                "wheel_legacy_simple_integration_authoritative": False,
                "wheel_integral_authoritative_artifact": (
                    "V003_WHEEL_INTEGRAL_EVIDENCE.json"
                ),
                "physical_qualification_eligible": False,
                "environment_equivalence_eligible": False,
                "physical_success": False,
            }
        traversal = self.traversal.result()
        last = self.fsm50_rows[-1] if self.fsm50_rows else {}
        sample_count = len(self.fsm50_rows)

        def finite_mapping(
            row: Mapping[str, Any], key: str, names: tuple[str, ...] = LEGS
        ) -> bool:
            values = dict(row.get(key, {}) or {})
            return set(values) == set(names) and all(
                math.isfinite(_safe_float(values.get(name))) for name in names
            )

        def finite_vectors(
            row: Mapping[str, Any], key: str, names: tuple[str, ...] = LEGS
        ) -> bool:
            values = dict(row.get(key, {}) or {})
            if set(values) != set(names):
                return False
            for name in names:
                try:
                    vector = np.asarray(values.get(name, []), dtype=float).reshape(-1)
                except Exception:
                    return False
                if vector.size < 3 or not np.isfinite(vector[:3]).all():
                    return False
            return True

        contact_force_valid = bool(
            self.fsm50_rows
            and all(row.get("wheel_net_force_valid") is True for row in self.fsm50_rows)
        )
        common_contact_evidence_valid = bool(
            contact_force_valid
            and all(
                finite_mapping(row, "wheel_contact_force_up_n")
                and finite_mapping(row, "wheel_canonical_forward_angle_rad")
                and finite_vectors(row, "wheel_centers_w")
                and set(dict(row.get("wheel_contact_classes", {}) or {})) == set(LEGS)
                and all(
                    dict(row.get("wheel_contact_classes", {}) or {}).get(leg)
                    in {item.value for item in ContactClass if item != ContactClass.UNKNOWN}
                    for leg in LEGS
                )
                for row in self.fsm50_rows
            )
        )
        filtered_contact_valid = self._filtered_samples_valid(self.fsm50_rows)
        final_classes = dict(last.get("wheel_contact_classes", {}) or {})
        final_contact_classes_valid = bool(
            set(final_classes) == set(LEGS)
            and all(final_classes.get(leg) != ContactClass.UNKNOWN.value for leg in LEGS)
        )
        final_all_top = bool(
            final_contact_classes_valid
            and all(final_classes.get(leg) == ContactClass.TOP.value for leg in LEGS)
        )
        final_loads = dict(last.get("wheel_contact_force_up_n", {}) or {})
        final_loads_valid = bool(
            set(final_loads) == set(LEGS)
            and all(math.isfinite(_safe_float(final_loads.get(leg))) for leg in LEGS)
        )
        final_all_loaded = bool(
            final_loads_valid
            and all(
                _safe_float(final_loads.get(leg)) >= self.force_threshold_n
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
        known_joint_limit_violation = any(
            row.get("joint_limit_evidence_valid") is True
            and row.get("joint_limit_violation") is True
            for row in self.fsm50_rows
        )
        joint_limit_safe = bool(
            joint_limit_evidence_valid
            and not known_joint_limit_violation
        )
        final_time = _safe_float(last.get("time_s"))
        stable_cutoff = final_time - self.final_stable_dwell_s
        stable_cutoff_guard = _time_window_numeric_guard_s(
            self.fsm50_rows,
            final_time_s=final_time,
            cutoff_time_s=stable_cutoff,
        )
        stable_rows = [
            row
            for row in self.fsm50_rows
            if math.isfinite(final_time)
            and _safe_float(row.get("time_s"))
            >= stable_cutoff - stable_cutoff_guard
        ]
        kinematics_evidence_valid = bool(
            self.fsm50_rows
            and all(
                math.isfinite(_safe_float(row.get(key)))
                for row in self.fsm50_rows
                for key in (
                    "time_s",
                    "base_vx_m_s",
                    "base_vy_m_s",
                    "base_vz_m_s",
                    "base_wx_rad_s",
                    "base_wy_rad_s",
                    "base_wz_rad_s",
                )
            )
        )
        final_stable = bool(stable_rows and kinematics_evidence_valid)
        known_final_velocity_violation = False
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
                if math.isfinite(linear) and math.isfinite(angular):
                    known_final_velocity_violation = True
                break
        if stable_rows:
            stable_span = _safe_float(stable_rows[-1].get("time_s")) - _safe_float(
                stable_rows[0].get("time_s")
            )
            final_stable = bool(
                final_stable and stable_span >= self.final_stable_dwell_s
            )
        collision_valid = bool(
            self.fsm50_rows
            and all(bool(row.get("collision_evidence_valid", False)) for row in self.fsm50_rows)
        )
        known_dangerous_collision = any(
            row.get("collision_evidence_valid") is True
            and row.get("dangerous_collision") is True
            for row in self.fsm50_rows
        )
        collision_safe = bool(
            collision_valid
            and not known_dangerous_collision
        )
        drive_target_evidence_valid = bool(
            self.fsm50_rows
            and all(
                row.get("physx_drive_target_evidence_valid") is True
                for row in self.fsm50_rows
            )
        )
        penetration_evidence_valid = bool(
            self.fsm50_rows
            and all(
                row.get("penetration_evidence_valid") is True
                and str(row.get("penetration_evidence_source", "") or "")
                == PHYSX_SEPARATION_SOURCE
                for row in self.fsm50_rows
            )
        )
        penetration_values = [
            _safe_float(row.get("maximum_collision_penetration_m"))
            for row in self.fsm50_rows
        ]
        maximum_penetration = max(
            (value for value in penetration_values if math.isfinite(value)),
            default=float("nan"),
        )
        known_penetration_violation = any(
            row.get("penetration_evidence_valid") is True
            and math.isfinite(_safe_float(row.get("maximum_collision_penetration_m")))
            and _safe_float(row.get("maximum_collision_penetration_m"))
            > self.maximum_penetration_m
            for row in self.fsm50_rows
        )
        penetration_safe = bool(
            penetration_evidence_valid
            and len(penetration_values) == len(self.fsm50_rows)
            and all(math.isfinite(value) for value in penetration_values)
            and maximum_penetration <= self.maximum_penetration_m
        )
        force_impulse_evidence_valid = bool(
            self.fsm50_rows
            and all(
                row.get("force_impulse_evidence_valid") is True
                for row in self.fsm50_rows
            )
        )
        zero_target_contact_displacement_samples: list[dict[str, Any]] = []
        drift_relevant_invalid = False
        drift_layout_valid = bool(self.fsm50_rows)
        for row in self.fsm50_rows:
            diagnostics = dict(row.get("wheel_contact_persistence_diagnostic", {}) or {})
            if set(diagnostics) != set(LEGS):
                drift_layout_valid = False
                continue
            for leg in LEGS:
                sample = dict(diagnostics.get(leg, {}) or {})
                mode = str(sample.get("mode", "") or "")
                contact_class = str(sample.get("contact_class", "") or "")
                if mode == "ZERO_TARGET_LOADED_CONTACT_EPOCH":
                    value = _safe_float(
                        sample.get("zero_target_contact_point_displacement_m")
                    )
                    semantics_valid = bool(
                        sample.get("physical_anchoring_proven") is False
                        and sample.get("material_point_identity_available") is False
                        and sample.get("contact_point_displacement_semantics")
                        == "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
                    )
                    if (
                        sample.get("evidence_valid") is True
                        and semantics_valid
                        and math.isfinite(value)
                    ):
                        zero_target_contact_displacement_samples.append(sample)
                    else:
                        drift_relevant_invalid = True
                elif mode == "ANCHORED":
                    # Old rows used a zero command as an anchoring label.  A
                    # new producer must never accept that semantic claim.
                    drift_layout_valid = False
                    drift_relevant_invalid = True
                elif mode == "INVALID_EVIDENCE" and contact_class in {
                    ContactClass.GROUND.value,
                    ContactClass.TOP.value,
                    ContactClass.UNKNOWN.value,
                }:
                    drift_relevant_invalid = True
                elif mode not in {
                    "NO_CONTACT",
                    "UNLOADED",
                    "ACTIVE_ROLLING",
                    "INVALID_EVIDENCE",
                }:
                    drift_layout_valid = False
        zero_target_contact_displacement_values = [
            _safe_float(sample.get("zero_target_contact_point_displacement_m"))
            for sample in zero_target_contact_displacement_samples
        ]
        contact_drift_evidence_valid = bool(
            drift_layout_valid
            and not drift_relevant_invalid
            and zero_target_contact_displacement_samples
            and all(
                math.isfinite(value)
                for value in zero_target_contact_displacement_values
            )
        )
        known_contact_drift_violation = any(
            value > self.maximum_allowed_contact_drift_m
            for value in zero_target_contact_displacement_values
            if math.isfinite(value)
        )
        contact_drift_safe = bool(
            contact_drift_evidence_valid and not known_contact_drift_violation
        )

        records: dict[str, dict[str, Any]] = {}

        def add_record(
            name: str,
            *,
            scope: str,
            available: bool,
            passed_when_available: bool,
            known_failure: bool = False,
            missing_reason: str = "evidence is incomplete",
            failure_reason: str = "criterion failed",
        ) -> None:
            if scope == "INSTRUMENTED_ONLY" and self.contact_mode == "formal":
                availability = "UNAVAILABLE_BY_ROLE"
                passed: bool | None = None
                reason = "criterion is reserved for instrumented B-role capture"
            elif known_failure:
                # A directly observed violation remains a FAIL even when some
                # other ticks are missing.  This is the fail-before-unknown
                # rule used by both verdict aggregators.
                availability = "AVAILABLE" if available else "MISSING"
                passed = False
                reason = failure_reason
            elif available:
                availability = "AVAILABLE"
                passed = bool(passed_when_available)
                reason = "" if passed else failure_reason
            else:
                availability = "MISSING"
                passed = None
                reason = missing_reason
            records[name] = {
                "name": name,
                "scope": scope,
                "availability": availability,
                "passed": passed,
                "reason": reason,
            }

        add_record(
            "contact_evidence_valid",
            scope="INSTRUMENTED_ONLY",
            available=filtered_contact_valid,
            passed_when_available=filtered_contact_valid,
            missing_reason="all-tick filtered wheel force/contact-point evidence is incomplete",
        )
        add_record(
            "all_legs_linkage_lift_valid",
            scope="COMMON",
            available=common_contact_evidence_valid,
            passed_when_available=bool(traversal["all_legs_valid"]),
            missing_reason="all-tick common wheel force/geometry/angle evidence is incomplete",
            failure_reason="one or more legs lack a valid linkage-lift episode",
        )
        add_record(
            "no_illegal_drive_up",
            scope="COMMON",
            available=common_contact_evidence_valid,
            passed_when_available=not bool(traversal["any_illegal_drive_up"]),
            known_failure=bool(traversal["any_illegal_drive_up"]),
            missing_reason="all-tick common traversal evidence is incomplete",
            failure_reason="an illegal drive-up event was observed",
        )
        add_record(
            "attitude_safe",
            scope="COMMON",
            available=attitude_evidence_valid,
            passed_when_available=attitude_safe,
            known_failure=bool(
                (math.isfinite(maximum_abs_roll) and maximum_abs_roll > self.maximum_roll_rad)
                or (math.isfinite(maximum_abs_pitch) and maximum_abs_pitch > self.maximum_pitch_rad)
                or (
                    math.isfinite(maximum_angular_speed)
                    and maximum_angular_speed > self.maximum_angular_velocity_rad_s
                )
            ),
            missing_reason="all-tick base attitude/angular-velocity evidence is incomplete",
            failure_reason="base attitude or angular velocity exceeded an existing limit",
        )
        add_record(
            "joint_limits_safe",
            scope="COMMON",
            available=joint_limit_evidence_valid,
            passed_when_available=joint_limit_safe,
            known_failure=known_joint_limit_violation,
            missing_reason="all-tick per-servo joint-limit evidence is incomplete",
            failure_reason="a servo joint-limit violation was observed",
        )
        add_record(
            "collision_safe",
            scope="INSTRUMENTED_ONLY",
            available=collision_valid,
            passed_when_available=collision_safe,
            known_failure=known_dangerous_collision,
            missing_reason="all-tick non-wheel obstacle-contact evidence is incomplete",
            failure_reason="a dangerous non-wheel obstacle collision was observed",
        )
        add_record(
            "penetration_safe",
            scope="COMMON",
            available=penetration_evidence_valid,
            passed_when_available=penetration_safe,
            known_failure=known_penetration_violation,
            missing_reason=(
                "all-tick exact-pair PhysX signed-separation evidence is incomplete; "
                "wheel-ground AABB overlap is diagnostic-only"
            ),
            failure_reason="exact-pair PhysX penetration exceeded the existing limit",
        )
        add_record(
            "contact_drift_safe",
            scope="INSTRUMENTED_ONLY",
            available=contact_drift_evidence_valid,
            passed_when_available=contact_drift_safe,
            known_failure=known_contact_drift_violation,
            missing_reason=(
                "zero-target loaded measured contact-point displacement evidence "
                "is incomplete"
            ),
            failure_reason=(
                "zero-target loaded measured contact-point displacement exceeded "
                "the existing limit"
            ),
        )
        add_record(
            "final_all_top",
            scope="COMMON",
            available=final_contact_classes_valid,
            passed_when_available=final_all_top,
            known_failure=bool(final_contact_classes_valid and not final_all_top),
            missing_reason="final four-wheel contact classes are incomplete",
            failure_reason="the final contact class is not TOP for all four wheels",
        )
        add_record(
            "final_all_loaded",
            scope="COMMON",
            available=final_loads_valid,
            passed_when_available=final_all_loaded,
            known_failure=bool(final_loads_valid and not final_all_loaded),
            missing_reason="final four-wheel common load evidence is incomplete",
            failure_reason="one or more final wheel loads are below the existing threshold",
        )
        add_record(
            "final_velocity_stable",
            scope="COMMON",
            available=kinematics_evidence_valid,
            passed_when_available=final_stable,
            known_failure=known_final_velocity_violation,
            missing_reason="all-tick base velocity/timestamp evidence is incomplete",
            failure_reason="the final base-velocity dwell did not meet existing limits",
        )

        if len(records) != 11:
            raise RuntimeError(f"physical evidence must contain exactly 11 criteria, got {len(records)}")

        def verdict(*, full: bool) -> tuple[str, list[str]]:
            selected = list(records.values())
            if not full:
                selected = [
                    record
                    for record in selected
                    if record["availability"] != "UNAVAILABLE_BY_ROLE"
                ]
            failures = [record for record in selected if record["passed"] is False]
            if failures:
                return "FAIL", [
                    f"{record['name']}: {record['reason']}" for record in failures
                ]
            unknown = [
                record
                for record in selected
                if record["availability"] != "AVAILABLE"
                or record["passed"] is None
            ]
            if unknown:
                return "NOT_EVALUABLE", [
                    f"{record['name']}: {record['reason']}" for record in unknown
                ]
            return "PASS", []

        role_capture_verdict, role_reasons = verdict(full=False)
        full_physical_verdict, full_reasons = verdict(full=True)
        evidence_complete = all(
            record["availability"] == "AVAILABLE" for record in records.values()
        )
        strict_criteria = {
            name: record["passed"] for name, record in records.items()
        }
        not_evaluable_reasons = [
            f"{record['name']}: {record['reason']}"
            for record in records.values()
            if record["availability"] != "AVAILABLE"
            or record["passed"] is None
        ]
        physical_success = full_physical_verdict == "PASS"
        return {
            "schema_version": "fsm50.physical_evidence.v2",
            "source_version": self.source_version,
            "contact_mode": self.contact_mode,
            "environment_equivalence_role": str(
                getattr(self, "environment_equivalence_role", "") or ""
            ),
            "diagnostic_role": str(
                getattr(self, "diagnostic_role", "") or ""
            ),
            "sample_count": sample_count,
            "contact_force_valid": contact_force_valid,
            "common_contact_evidence_valid": common_contact_evidence_valid,
            "filtered_contact_valid": filtered_contact_valid,
            "filtered_contact_error_count": len(self.filtered_contact_errors),
            "filtered_contact_errors": list(self.filtered_contact_errors),
            "evidence_complete": evidence_complete,
            "not_evaluable_reasons": not_evaluable_reasons,
            "criteria": records,
            "criterion_records": list(records.values()),
            "strict_criteria": strict_criteria,
            "role_capture_verdict": role_capture_verdict,
            "full_physical_verdict": full_physical_verdict,
            "role_capture_verdict_reasons": role_reasons,
            "full_physical_verdict_reasons": full_reasons,
            "traversal": traversal,
            "final_wheel_contact_classes": final_classes,
            "final_all_top": final_all_top,
            "final_all_loaded": final_all_loaded,
            "maximum_contact_drift_m": (
                max(zero_target_contact_displacement_values)
                if zero_target_contact_displacement_values
                else None
            ),
            "maximum_zero_target_contact_point_displacement_m": (
                max(zero_target_contact_displacement_values)
                if zero_target_contact_displacement_values
                else None
            ),
            "minimum_two_leg_corridor_margin_m": None
            if not math.isfinite(self.minimum_corridor_margin_m)
            else self.minimum_corridor_margin_m,
            "dangerous_collision_count": len(self.dangerous_collision_rows),
            "collision_evidence_valid": collision_valid,
            "physx_drive_target_evidence_valid": drive_target_evidence_valid,
            "penetration_evidence_valid": penetration_evidence_valid,
            "penetration_evidence_source": PHYSX_SEPARATION_SOURCE,
            "maximum_collision_penetration_m": maximum_penetration,
            "maximum_allowed_penetration_m": self.maximum_penetration_m,
            "force_impulse_evidence_valid": force_impulse_evidence_valid,
            "final_wheel_force_impulse_w_n_s": {
                leg: list(values)
                for leg, values in self.integrated_wheel_force_impulse_w.items()
            },
            "final_wheel_upward_force_impulse_n_s": dict(
                self.integrated_wheel_upward_impulse_n_s
            ),
            "joint_limit_evidence_valid": joint_limit_evidence_valid,
            "joint_limit_violation_count": len(self.joint_limit_violation_rows),
            "maximum_abs_roll_rad": maximum_abs_roll,
            "maximum_abs_pitch_rad": maximum_abs_pitch,
            "maximum_angular_velocity_rad_s": maximum_angular_speed,
            "final_velocity_stable": final_stable,
            "contact_drift_evidence_valid": contact_drift_evidence_valid,
            "zero_target_loaded_contact_displacement_sample_count": len(
                zero_target_contact_displacement_samples
            ),
            "contact_point_displacement_semantics": (
                "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
            ),
            "physical_anchoring_proven": False,
            "material_point_identity_available": False,
            "wheel_legacy_simple_integration_authoritative": False,
            "wheel_integral_authoritative_artifact": (
                "V003_WHEEL_INTEGRAL_EVIDENCE.json"
            ),
            "physical_success": physical_success,
        }

    def _journal_stream_rows(self) -> dict[str, list[dict[str, Any]]]:
        streams = super()._journal_stream_rows()
        streams.update(
            {
                "fsm50_telemetry": self.fsm50_rows,
                "wheel_filtered_contacts": self.filtered_surface_rows,
                "nonwheel_obstacle_contacts": self.nonwheel_obstacle_rows,
            }
        )
        return streams

    def _validate_checkpoint_state(self) -> None:
        super()._validate_checkpoint_state()
        if len(self.rows) != len(self.fsm50_rows):
            raise RuntimeError(
                "base/FSM telemetry row alignment failed: "
                f"base={len(self.rows)}, fsm50={len(self.fsm50_rows)}"
            )

    def _write_canonical_artifacts(self) -> dict[str, int]:
        record_counts = super()._write_canonical_artifacts()
        if self.run_dir is None:
            raise RuntimeError("telemetry run_dir is unavailable")
        write_csv(self.run_dir / "fsm50_telemetry.csv", self.fsm50_rows)
        record_counts["fsm50_telemetry.csv"] = len(self.fsm50_rows)
        write_jsonl(self.run_dir / "fsm50_telemetry.jsonl", self.fsm50_rows)
        record_counts["fsm50_telemetry.jsonl"] = len(self.fsm50_rows)
        write_jsonl(
            self.run_dir / "wheel_filtered_contacts.jsonl",
            self.filtered_surface_rows,
        )
        record_counts["wheel_filtered_contacts.jsonl"] = len(
            self.filtered_surface_rows
        )
        write_csv(
            self.run_dir / "nonwheel_obstacle_contacts.csv",
            self.nonwheel_obstacle_rows,
        )
        record_counts["nonwheel_obstacle_contacts.csv"] = len(
            self.nonwheel_obstacle_rows
        )
        write_jsonl(
            self.run_dir / "nonwheel_obstacle_contacts.jsonl",
            self.nonwheel_obstacle_rows,
        )
        record_counts["nonwheel_obstacle_contacts.jsonl"] = len(
            self.nonwheel_obstacle_rows
        )
        write_csv(self.run_dir / "state_timeline.csv", self.state_timeline_rows)
        record_counts["state_timeline.csv"] = len(self.state_timeline_rows)
        write_json(self.run_dir / "physical_evidence.json", self.physical_evidence())
        record_counts["physical_evidence.json"] = 1
        return record_counts
