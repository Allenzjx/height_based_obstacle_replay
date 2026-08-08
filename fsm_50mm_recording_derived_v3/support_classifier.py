"""Physics-state support, contact, and traversal evidence classifiers.

All functions in this module are Isaac-independent so the exact guard logic can
be unit tested before starting the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LEGS = ("FL", "FR", "RL", "RR")
DIAGONAL_A = ("FL", "RR")
DIAGONAL_B = ("FR", "RL")


class ContactClass(str, Enum):
    GROUND = "GROUND"
    FRONT_FACE = "FRONT_FACE"
    TOP = "TOP"
    AIR = "AIR"
    UNKNOWN = "UNKNOWN"


class PrimaryDiagonal(str, Enum):
    FL_RR = "FL_RR"
    FR_RL = "FR_RL"
    BALANCED = "BALANCED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ObstacleGeometry:
    front_face_x_m: float
    top_z_m: float
    bottom_z_m: float = 0.0
    rear_face_x_m: float | None = None
    center_y_m: float = 0.0
    width_m: float = 2.0


@dataclass(frozen=True)
class WheelObservation:
    leg: str
    center_w: tuple[float, float, float]
    upward_force_n: float = float("nan")
    total_force_n: float = float("nan")
    wheel_angle_rad: float = float("nan")
    wheel_velocity_rad_s: float = float("nan")
    force_source: str = "unavailable"


@dataclass(frozen=True)
class WheelContact:
    leg: str
    contact_class: ContactClass
    active: bool
    center_w: tuple[float, float, float]
    contact_point_w: tuple[float, float, float]
    obstacle_relative: tuple[float, float, float]
    upward_force_n: float
    total_force_n: float
    clearance_over_top_m: float
    front_face_clearance_m: float
    source: str
    confidence: str


@dataclass(frozen=True)
class DiagonalSupportResult:
    primary: PrimaryDiagonal
    load_fl_rr_n: float
    load_fr_rl_n: float
    load_share_fl_rr: float
    load_share_fr_rl: float
    support_legs: tuple[str, ...]
    light_support_legs: tuple[str, ...]
    persistent_support_legs: tuple[str, ...]
    stable_contact_legs: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class DiagonalCorridorResult:
    valid: bool
    segment_fraction: float
    perpendicular_distance_m: float
    within_longitudinal_bounds: bool
    within_corridor_width: bool
    reason: str = ""


@dataclass(frozen=True)
class PolygonSupportResult:
    valid: bool
    signed_margin_m: float
    hull: tuple[tuple[float, float], ...]
    degenerate: bool
    reason: str = ""


def _finite_vector(values: Sequence[float], size: int) -> np.ndarray | None:
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if vector.size < size or not np.isfinite(vector[:size]).all():
        return None
    return vector[:size]


def classify_wheel_contact(
    observation: WheelObservation,
    obstacle: ObstacleGeometry,
    *,
    wheel_radius_m: float,
    force_threshold_n: float = 2.0,
    surface_tolerance_m: float = 0.012,
    face_tolerance_m: float = 0.012,
) -> WheelContact:
    """Classify a wheel against ground, front face, top, or air.

    Force is used when available.  If the sensor is unavailable, geometry may
    still classify a contact, but the returned confidence is explicitly
    ``GEOMETRY_ONLY`` and cannot satisfy a load-confirmation guard.
    """

    leg = str(observation.leg).upper()
    center = _finite_vector(observation.center_w, 3)
    if leg not in LEGS or center is None or not math.isfinite(float(wheel_radius_m)):
        nan3 = (float("nan"),) * 3
        return WheelContact(
            leg=leg,
            contact_class=ContactClass.UNKNOWN,
            active=False,
            center_w=nan3 if center is None else tuple(float(v) for v in center),
            contact_point_w=nan3,
            obstacle_relative=nan3,
            upward_force_n=float(observation.upward_force_n),
            total_force_n=float(observation.total_force_n),
            clearance_over_top_m=float("nan"),
            front_face_clearance_m=float("nan"),
            source="invalid_geometry",
            confidence="UNKNOWN",
        )

    x, y, z = (float(value) for value in center)
    radius = abs(float(wheel_radius_m))
    bottom = z - radius
    front_edge = x + radius
    clearance_top = bottom - float(obstacle.top_z_m)
    # Traversal proof is tied to the wheel *center* crossing the obstacle
    # plane.  ``front_edge`` is still used below to recognize a face contact,
    # but using it as crossing evidence would declare success one radius too
    # early and could hide a tire-driven climb.
    clearance_face = x - float(obstacle.front_face_x_m)
    force_available = math.isfinite(float(observation.total_force_n)) or math.isfinite(
        float(observation.upward_force_n)
    )
    measured_force = max(
        0.0,
        float(observation.total_force_n)
        if math.isfinite(float(observation.total_force_n))
        else float(observation.upward_force_n)
        if math.isfinite(float(observation.upward_force_n))
        else 0.0,
    )
    force_active = bool(force_available and measured_force >= float(force_threshold_n))
    lateral_inside = abs(y - float(obstacle.center_y_m)) <= (
        0.5 * float(obstacle.width_m) + radius + surface_tolerance_m
    )
    on_top_geometry = bool(
        lateral_inside
        and x >= float(obstacle.front_face_x_m) - radius - face_tolerance_m
        and (obstacle.rear_face_x_m is None or x <= obstacle.rear_face_x_m + radius)
        and abs(bottom - float(obstacle.top_z_m)) <= surface_tolerance_m
    )
    on_face_geometry = bool(
        lateral_inside
        and abs(front_edge - float(obstacle.front_face_x_m)) <= face_tolerance_m
        and z >= float(obstacle.bottom_z_m) + radius - surface_tolerance_m
        and bottom <= float(obstacle.top_z_m) + surface_tolerance_m
    )
    on_ground_geometry = bool(
        abs(bottom - float(obstacle.bottom_z_m)) <= surface_tolerance_m
        and x < float(obstacle.front_face_x_m) - radius + face_tolerance_m
    )

    # TOP takes precedence near the upper front corner only after the wheel
    # bottom reaches top height; this avoids calling a loaded face contact TOP.
    if on_top_geometry and (force_active or not force_available):
        contact_class = ContactClass.TOP
        point = (x, y, float(obstacle.top_z_m))
    elif on_face_geometry and (force_active or not force_available):
        contact_class = ContactClass.FRONT_FACE
        point = (float(obstacle.front_face_x_m), y, min(z, obstacle.top_z_m))
    elif on_ground_geometry and (force_active or not force_available):
        contact_class = ContactClass.GROUND
        point = (x, y, float(obstacle.bottom_z_m))
    else:
        contact_class = ContactClass.AIR
        point = (x, y, bottom)

    active = contact_class not in {ContactClass.AIR, ContactClass.UNKNOWN}
    confidence = "FORCE_AND_GEOMETRY" if force_available else "GEOMETRY_ONLY"
    if force_available and not force_active and active:
        # This path normally cannot occur, but keeping it explicit makes later
        # changes to the geometric rules fail safe.
        active = False
        contact_class = ContactClass.AIR
        confidence = "FORCE_REJECTED_GEOMETRY"
    return WheelContact(
        leg=leg,
        contact_class=contact_class,
        active=active,
        center_w=(x, y, z),
        contact_point_w=tuple(float(v) for v in point),
        obstacle_relative=(
            x - float(obstacle.front_face_x_m),
            y - float(obstacle.center_y_m),
            bottom - float(obstacle.top_z_m),
        ),
        upward_force_n=float(observation.upward_force_n),
        total_force_n=float(observation.total_force_n),
        clearance_over_top_m=clearance_top,
        front_face_clearance_m=clearance_face,
        source=str(observation.force_source or "unavailable"),
        confidence=confidence,
    )


def classify_diagonal_support(
    contacts: Mapping[str, WheelContact],
    *,
    support_force_threshold_n: float = 2.0,
    light_force_threshold_n: float = 0.5,
    persistence_s: Mapping[str, float] | None = None,
    minimum_persistence_s: float = 0.08,
    contact_drift_m: Mapping[str, float] | None = None,
    maximum_contact_drift_m: float = 0.015,
    dominance_hysteresis: float = 0.08,
) -> DiagonalSupportResult:
    persistence = persistence_s or {}
    drift = contact_drift_m or {}

    def load(leg: str) -> float:
        row = contacts.get(leg)
        if row is None or not row.active:
            return 0.0
        value = float(row.upward_force_n)
        if not math.isfinite(value):
            return 0.0
        return max(0.0, value)

    loads = {leg: load(leg) for leg in LEGS}
    support = tuple(
        leg
        for leg in LEGS
        if contacts.get(leg) is not None
        and contacts[leg].active
        and loads[leg] >= float(support_force_threshold_n)
    )
    light = tuple(
        leg
        for leg in LEGS
        if contacts.get(leg) is not None
        and contacts[leg].active
        and float(light_force_threshold_n) <= loads[leg] < float(support_force_threshold_n)
    )
    persistent = tuple(
        leg for leg in support if float(persistence.get(leg, 0.0)) >= minimum_persistence_s
    )
    stable = tuple(
        leg
        for leg in persistent
        if float(drift.get(leg, 0.0)) <= float(maximum_contact_drift_m)
    )
    load_a = loads["FL"] + loads["RR"]
    load_b = loads["FR"] + loads["RL"]
    total = load_a + load_b
    if total <= 1.0e-12:
        primary = PrimaryDiagonal.UNKNOWN
        share_a = share_b = float("nan")
        reason = "no force-valid support load"
    else:
        share_a = load_a / total
        share_b = load_b / total
        if share_a >= 0.5 + dominance_hysteresis:
            primary = PrimaryDiagonal.FL_RR
        elif share_b >= 0.5 + dominance_hysteresis:
            primary = PrimaryDiagonal.FR_RL
        else:
            primary = PrimaryDiagonal.BALANCED
        reason = "force load share with contact/persistence/drift qualifiers"
    return DiagonalSupportResult(
        primary=primary,
        load_fl_rr_n=load_a,
        load_fr_rl_n=load_b,
        load_share_fl_rr=share_a,
        load_share_fr_rl=share_b,
        support_legs=support,
        light_support_legs=light,
        persistent_support_legs=persistent,
        stable_contact_legs=stable,
        reason=reason,
    )


def diagonal_support_corridor(
    p1_xy: Sequence[float],
    p2_xy: Sequence[float],
    com_xy: Sequence[float],
    *,
    corridor_half_width_m: float,
    longitudinal_slack: float = 0.08,
) -> DiagonalCorridorResult:
    p1 = _finite_vector(p1_xy, 2)
    p2 = _finite_vector(p2_xy, 2)
    com = _finite_vector(com_xy, 2)
    if p1 is None or p2 is None or com is None:
        return DiagonalCorridorResult(False, float("nan"), float("nan"), False, False, "non-finite input")
    direction = p2 - p1
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1.0e-12:
        return DiagonalCorridorResult(False, float("nan"), float("nan"), False, False, "coincident contacts")
    relative = com - p1
    fraction = float(np.dot(relative, direction) / length_squared)
    perpendicular = abs(float(direction[0] * relative[1] - direction[1] * relative[0])) / math.sqrt(length_squared)
    longitudinal_ok = -float(longitudinal_slack) <= fraction <= 1.0 + float(longitudinal_slack)
    width_ok = perpendicular <= float(corridor_half_width_m)
    return DiagonalCorridorResult(
        valid=bool(longitudinal_ok and width_ok),
        segment_fraction=fraction,
        perpendicular_distance_m=perpendicular,
        within_longitudinal_bounds=longitudinal_ok,
        within_corridor_width=width_ok,
        reason="" if longitudinal_ok and width_ok else "COM outside diagonal support corridor",
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
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


def polygon_support_margin(
    support_points_xy: Iterable[Sequence[float]], com_xy: Sequence[float]
) -> PolygonSupportResult:
    points = np.asarray(list(support_points_xy), dtype=float)
    com = _finite_vector(com_xy, 2)
    if com is None or points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return PolygonSupportResult(False, float("nan"), (), True, "at least three finite support contacts required")
    points = points[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    hull = _convex_hull(points)
    if len(hull) < 3:
        return PolygonSupportResult(False, float("nan"), tuple(map(tuple, hull)), True, "support hull is degenerate")
    distances: list[float] = []
    inside = True
    signed_crosses: list[float] = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        relative = com - start
        signed_crosses.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
        denom = float(np.dot(edge, edge))
        t = 0.0 if denom <= 1.0e-12 else max(0.0, min(1.0, float(np.dot(relative, edge) / denom)))
        distances.append(float(np.linalg.norm(com - (start + t * edge))))
    inside = all(value >= -1.0e-10 for value in signed_crosses) or all(
        value <= 1.0e-10 for value in signed_crosses
    )
    distance = min(distances) if distances else float("nan")
    return PolygonSupportResult(
        valid=bool(inside),
        signed_margin_m=distance if inside else -distance,
        hull=tuple((float(row[0]), float(row[1])) for row in hull),
        degenerate=False,
        reason="" if inside else "COM projection outside support polygon",
    )


@dataclass
class ContactPersistenceTracker:
    """Track per-leg contact dwell, anchor drift, and class changes."""

    active_since_s: dict[str, float] = field(default_factory=dict)
    anchor_w: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    last_class: dict[str, ContactClass] = field(default_factory=dict)

    def update(
        self, time_s: float, contacts: Mapping[str, WheelContact]
    ) -> tuple[dict[str, float], dict[str, float]]:
        persistence: dict[str, float] = {}
        drift: dict[str, float] = {}
        for leg in LEGS:
            contact = contacts.get(leg)
            if contact is None or not contact.active:
                self.active_since_s.pop(leg, None)
                self.anchor_w.pop(leg, None)
                self.last_class[leg] = ContactClass.AIR if contact is None else contact.contact_class
                persistence[leg] = 0.0
                drift[leg] = 0.0
                continue
            if self.last_class.get(leg) != contact.contact_class or leg not in self.active_since_s:
                self.active_since_s[leg] = float(time_s)
                self.anchor_w[leg] = contact.contact_point_w
            self.last_class[leg] = contact.contact_class
            persistence[leg] = max(0.0, float(time_s) - self.active_since_s[leg])
            anchor = np.asarray(self.anchor_w[leg], dtype=float)
            current = np.asarray(contact.contact_point_w, dtype=float)
            drift[leg] = float(np.linalg.norm(current - anchor))
        return persistence, drift


@dataclass
class LegTraversalEvidence:
    leg: str
    minimum_airborne_s: float = 0.05
    minimum_unload_dwell_s: float = 0.025
    loaded_support_seen_s: float | None = None
    unload_candidate_s: float | None = None
    airborne_candidate_s: float | None = None
    unload_start_s: float | None = None
    unload_end_s: float | None = None
    airborne_start_s: float | None = None
    airborne_end_s: float | None = None
    maximum_clearance_m: float = -math.inf
    front_face_crossing_s: float | None = None
    top_contact_s: float | None = None
    top_load_confirm_s: float | None = None
    minimum_load_n: float = math.inf
    loaded_front_face_rotation_rad: float = 0.0
    illegal_reasons: list[str] = field(default_factory=list)
    last_angle_rad: float | None = None
    last_class: ContactClass = ContactClass.UNKNOWN
    top_load_since_s: float | None = None
    airborne_seen_before_top: bool = False

    @property
    def linkage_lift_valid(self) -> bool:
        ordered = bool(
            self.unload_start_s is not None
            and self.airborne_start_s is not None
            and self.airborne_end_s is not None
            and self.front_face_crossing_s is not None
            and self.top_contact_s is not None
            and self.top_load_confirm_s is not None
            and self.unload_start_s <= self.airborne_start_s
            and self.airborne_start_s <= self.front_face_crossing_s
            and self.front_face_crossing_s <= self.top_contact_s
            and self.top_contact_s <= self.top_load_confirm_s
        )
        return bool(
            ordered
            and float(self.airborne_end_s) - float(self.airborne_start_s)
            >= float(self.minimum_airborne_s)
            and not self.illegal_reasons
        )


class TraversalEvidenceTracker:
    """Require UNLOAD→AIR→CLEAR_FACE→TOP→LOAD for each individual leg."""

    def __init__(
        self,
        *,
        unload_force_n: float,
        load_confirm_force_n: float,
        top_load_dwell_s: float,
        loaded_front_face_rotation_limit_rad: float,
        minimum_airborne_s: float = 0.05,
        minimum_unload_dwell_s: float = 0.025,
        wheel_forward_sign: Mapping[str, float] | None = None,
    ) -> None:
        self.unload_force_n = float(unload_force_n)
        self.load_confirm_force_n = float(load_confirm_force_n)
        self.top_load_dwell_s = float(top_load_dwell_s)
        self.loaded_front_face_rotation_limit_rad = float(
            loaded_front_face_rotation_limit_rad
        )
        self.minimum_airborne_s = max(0.0, float(minimum_airborne_s))
        self.minimum_unload_dwell_s = max(0.0, float(minimum_unload_dwell_s))
        self.wheel_forward_sign = dict(wheel_forward_sign or {})
        self.legs = {
            leg: LegTraversalEvidence(
                leg,
                minimum_airborne_s=self.minimum_airborne_s,
                minimum_unload_dwell_s=self.minimum_unload_dwell_s,
            )
            for leg in LEGS
        }

    def update(
        self,
        time_s: float,
        contacts: Mapping[str, WheelContact],
        wheel_angles_rad: Mapping[str, float],
    ) -> None:
        for leg in LEGS:
            evidence = self.legs[leg]
            contact = contacts.get(leg)
            if contact is None:
                continue
            load = float(contact.upward_force_n)
            if math.isfinite(load):
                evidence.minimum_load_n = min(evidence.minimum_load_n, max(0.0, load))
                if contact.active and load >= self.load_confirm_force_n:
                    if evidence.loaded_support_seen_s is None:
                        evidence.loaded_support_seen_s = float(time_s)
                if evidence.unload_start_s is None:
                    if (
                        evidence.loaded_support_seen_s is not None
                        and load <= self.unload_force_n
                    ):
                        if evidence.unload_candidate_s is None:
                            evidence.unload_candidate_s = float(time_s)
                        if (
                            float(time_s) - float(evidence.unload_candidate_s)
                            >= evidence.minimum_unload_dwell_s
                        ):
                            evidence.unload_start_s = float(evidence.unload_candidate_s)
                    elif load > self.unload_force_n:
                        evidence.unload_candidate_s = None
            angle = float(wheel_angles_rad.get(leg, float("nan")))
            delta = 0.0
            if evidence.last_angle_rad is not None and math.isfinite(angle):
                delta = angle - evidence.last_angle_rad
            if math.isfinite(angle):
                evidence.last_angle_rad = angle
            if (
                contact.contact_class == ContactClass.FRONT_FACE
                and math.isfinite(float(contact.total_force_n))
                # A vertical front face reacts mainly along world X; world-up
                # force can be near zero even during a heavily loaded tire
                # drive-up.  Face rejection therefore uses the measured total
                # obstacle-contact force, while support/load confirmation uses
                # the world-up component.
                and float(contact.total_force_n) > self.unload_force_n
            ):
                forward = float(self.wheel_forward_sign.get(leg, 1.0)) * delta
                evidence.loaded_front_face_rotation_rad += max(0.0, forward)
                if (
                    evidence.loaded_front_face_rotation_rad
                    > self.loaded_front_face_rotation_limit_rad
                    and "loaded front-face wheel rotation exceeded limit"
                    not in evidence.illegal_reasons
                ):
                    evidence.illegal_reasons.append(
                        "loaded front-face wheel rotation exceeded limit"
                    )
            if contact.contact_class == ContactClass.AIR:
                if evidence.airborne_candidate_s is None:
                    evidence.airborne_candidate_s = float(time_s)
                # Sensor warm-up may briefly report every wheel AIR.  A lift
                # is admissible only after that wheel first demonstrated a
                # loaded support contact and then sustained an unload dwell.
                if evidence.unload_start_s is not None:
                    if evidence.airborne_start_s is None:
                        evidence.airborne_start_s = float(evidence.airborne_candidate_s)
                    evidence.airborne_seen_before_top = True
                    evidence.maximum_clearance_m = max(
                        evidence.maximum_clearance_m,
                        float(contact.clearance_over_top_m),
                    )
            elif evidence.last_class == ContactClass.AIR and evidence.airborne_end_s is None:
                if evidence.airborne_start_s is not None:
                    evidence.airborne_end_s = float(time_s)
                    if (
                        evidence.front_face_crossing_s is None
                        and contact.contact_class != ContactClass.TOP
                        and "airborne attempt ended before front-face crossing"
                        not in evidence.illegal_reasons
                    ):
                        evidence.illegal_reasons.append(
                            "airborne attempt ended before front-face crossing"
                        )
                if evidence.airborne_start_s is None:
                    evidence.airborne_candidate_s = None
            if (
                contact.front_face_clearance_m >= 0.0
                and evidence.front_face_crossing_s is None
            ):
                evidence.front_face_crossing_s = float(time_s)
                if not evidence.airborne_seen_before_top:
                    evidence.illegal_reasons.append(
                        "front face crossed without prior airborne interval"
                    )
                if contact.clearance_over_top_m < 0.0:
                    evidence.illegal_reasons.append(
                        "front face crossed before wheel bottom cleared obstacle top"
                    )
            if contact.contact_class == ContactClass.TOP:
                if evidence.top_contact_s is None:
                    evidence.top_contact_s = float(time_s)
                    if not evidence.airborne_seen_before_top:
                        evidence.illegal_reasons.append(
                            "GROUND/FRONT_FACE to TOP transition without AIR"
                        )
                    if evidence.front_face_crossing_s is None:
                        evidence.illegal_reasons.append(
                            "top contact occurred before wheel center crossed front face"
                        )
                if math.isfinite(load) and load >= self.load_confirm_force_n:
                    if evidence.top_load_since_s is None:
                        evidence.top_load_since_s = float(time_s)
                    if (
                        evidence.top_load_confirm_s is None
                        and float(time_s) - evidence.top_load_since_s
                        >= self.top_load_dwell_s
                    ):
                        evidence.top_load_confirm_s = float(time_s)
                        evidence.unload_end_s = evidence.airborne_start_s
                else:
                    evidence.top_load_since_s = None
            else:
                evidence.top_load_since_s = None
            evidence.last_class = contact.contact_class

    def result(self) -> dict[str, Any]:
        legs: dict[str, Any] = {}
        for leg, evidence in self.legs.items():
            row = {
                key: value
                for key, value in evidence.__dict__.items()
                if key
                not in {
                    "last_angle_rad",
                    "last_class",
                    "top_load_since_s",
                    "unload_candidate_s",
                    "airborne_candidate_s",
                }
            }
            row["last_class"] = evidence.last_class.value
            row["linkage_lift_valid"] = evidence.linkage_lift_valid
            if not math.isfinite(float(row["maximum_clearance_m"])):
                row["maximum_clearance_m"] = None
            if not math.isfinite(float(row["minimum_load_n"])):
                row["minimum_load_n"] = None
            legs[leg] = row
        return {
            "legs": legs,
            "all_legs_valid": all(item.linkage_lift_valid for item in self.legs.values()),
            "any_illegal_drive_up": any(item.illegal_reasons for item in self.legs.values()),
        }
