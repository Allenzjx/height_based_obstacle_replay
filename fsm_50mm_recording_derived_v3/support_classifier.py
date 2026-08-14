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


def _validated_filtered_surface_activity(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    leg: str,
    force_threshold_n: float,
) -> tuple[dict[str, bool] | None, str]:
    """Decode the exact ground/obstacle pair without inferring its identity.

    ``identity_valid`` is supplied by the sensor-layout owner.  This
    Isaac-independent layer still requires the row identity to match the
    observation and requires the complete force/activity/contact-point
    contract before it lets exact surface evidence affect classification.
    """

    if not isinstance(evidence, Mapping):
        return None, "filtered surface evidence is not a mapping"
    if set(evidence) != {"ground", "obstacle"}:
        return None, "filtered surface evidence must contain exactly ground and obstacle"

    activity: dict[str, bool] = {}
    for surface in ("ground", "obstacle"):
        row = evidence.get(surface)
        if not isinstance(row, Mapping):
            return None, f"{surface} filtered surface evidence is not a mapping"
        if row.get("identity_valid") is not True:
            return None, f"{surface} filtered surface identity is invalid"
        if str(row.get("leg", "")).upper() != leg:
            return None, f"{surface} filtered surface leg does not match {leg}"
        if str(row.get("surface", "")) != surface:
            return None, f"{surface} filtered surface label does not match its key"
        if row.get("force_valid") is not True:
            return None, f"{surface} filtered surface force is invalid"
        if not isinstance(row.get("active"), bool):
            return None, f"{surface} filtered surface active flag is invalid"
        if not isinstance(row.get("contact_point_valid"), bool):
            return None, f"{surface} filtered surface contact-point flag is invalid"
        normal_force_value = row.get("normal_force_n")
        if isinstance(normal_force_value, (bool, np.bool_)):
            return None, f"{surface} filtered surface normal force is invalid"
        try:
            normal_force_n = float(normal_force_value)
        except (TypeError, ValueError):
            return None, f"{surface} filtered surface normal force is invalid"
        if not math.isfinite(normal_force_n) or normal_force_n < 0.0:
            return None, f"{surface} filtered surface normal force is invalid"

        active = bool(row["active"])
        threshold_active = normal_force_n >= float(force_threshold_n)
        if active != threshold_active:
            return None, f"{surface} filtered surface activity/force is inconsistent"
        if active and row.get("contact_point_valid") is not True:
            return None, f"{surface} active filtered surface contact point is invalid"
        activity[surface] = active
    return activity, ""


def classify_wheel_contact(
    observation: WheelObservation,
    obstacle: ObstacleGeometry,
    *,
    wheel_radius_m: float,
    force_threshold_n: float = 2.0,
    surface_tolerance_m: float = 0.012,
    face_tolerance_m: float = 0.012,
    filtered_surface_evidence: Mapping[str, Mapping[str, Any]] | None = None,
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

    surface_evidence_available = filtered_surface_evidence is not None
    surface_activity: dict[str, bool] | None = None
    surface_evidence_error = ""
    if surface_evidence_available:
        surface_activity, surface_evidence_error = (
            _validated_filtered_surface_activity(
                filtered_surface_evidence,
                leg=leg,
                force_threshold_n=force_threshold_n,
            )
        )

    # TOP takes precedence near the upper front corner only after the wheel
    # bottom reaches top height; this avoids calling a loaded face contact TOP.
    # When an exact filtered pair is available, its surface identity resolves
    # geometry overlap.  Geometry remains necessary to split an obstacle pair
    # into TOP versus FRONT_FACE and to reject an impossible surface pairing.
    if surface_evidence_available and surface_activity is None:
        contact_class = ContactClass.UNKNOWN
        point = (x, y, bottom)
        confidence = "FILTERED_SURFACE_EVIDENCE_INVALID"
    elif surface_activity is not None:
        ground_active = surface_activity["ground"]
        obstacle_active = surface_activity["obstacle"]
        if ground_active and not obstacle_active and on_ground_geometry:
            contact_class = ContactClass.GROUND
            point = (x, y, float(obstacle.bottom_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif obstacle_active and not ground_active and on_top_geometry:
            contact_class = ContactClass.TOP
            point = (x, y, float(obstacle.top_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif obstacle_active and not ground_active and on_face_geometry:
            contact_class = ContactClass.FRONT_FACE
            point = (float(obstacle.front_face_x_m), y, min(z, obstacle.top_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif ground_active and obstacle_active and on_top_geometry:
            contact_class = ContactClass.TOP
            point = (x, y, float(obstacle.top_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif ground_active and obstacle_active and on_face_geometry:
            contact_class = ContactClass.FRONT_FACE
            point = (float(obstacle.front_face_x_m), y, min(z, obstacle.top_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif ground_active and obstacle_active and on_ground_geometry:
            contact_class = ContactClass.GROUND
            point = (x, y, float(obstacle.bottom_z_m))
            confidence = "EXACT_FILTERED_SURFACE_AND_GEOMETRY"
        elif not ground_active and not obstacle_active and not force_active:
            contact_class = ContactClass.AIR
            point = (x, y, bottom)
            confidence = "EXACT_FILTERED_SURFACE_NO_CONTACT"
        else:
            # A loaded aggregate with no active exact pair, or an active exact
            # pair incompatible with geometry, is contradictory evidence.  It
            # must not be disguised as AIR or assigned to a guessed surface.
            contact_class = ContactClass.UNKNOWN
            point = (x, y, bottom)
            confidence = "FILTERED_SURFACE_GEOMETRY_MISMATCH"
    elif on_top_geometry and (force_active or not force_available):
        contact_class = ContactClass.TOP
        point = (x, y, float(obstacle.top_z_m))
        confidence = "FORCE_AND_GEOMETRY" if force_available else "GEOMETRY_ONLY"
    elif on_face_geometry and (force_active or not force_available):
        contact_class = ContactClass.FRONT_FACE
        point = (float(obstacle.front_face_x_m), y, min(z, obstacle.top_z_m))
        confidence = "FORCE_AND_GEOMETRY" if force_available else "GEOMETRY_ONLY"
    elif on_ground_geometry and (force_active or not force_available):
        contact_class = ContactClass.GROUND
        point = (x, y, float(obstacle.bottom_z_m))
        confidence = "FORCE_AND_GEOMETRY" if force_available else "GEOMETRY_ONLY"
    else:
        contact_class = ContactClass.AIR
        point = (x, y, bottom)
        confidence = "FORCE_AND_GEOMETRY" if force_available else "GEOMETRY_ONLY"

    active = contact_class not in {ContactClass.AIR, ContactClass.UNKNOWN}
    if not surface_evidence_available and force_available and not force_active and active:
        # This path normally cannot occur, but keeping it explicit makes later
        # changes to the geometric rules fail safe.
        active = False
        contact_class = ContactClass.AIR
        confidence = "FORCE_REJECTED_GEOMETRY"
    source = str(observation.force_source or "unavailable")
    if surface_evidence_available:
        source = (
            f"{source}; exact_filtered_surface_evidence"
            if not surface_evidence_error
            else f"{source}; invalid_filtered_surface_evidence: {surface_evidence_error}"
        )
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
        source=source,
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


class ContactPersistenceMode(str, Enum):
    """Per-tick interpretation of contact-point motion."""

    NO_CONTACT = "NO_CONTACT"
    UNLOADED = "UNLOADED"
    ZERO_TARGET_LOADED_CONTACT_EPOCH = "ZERO_TARGET_LOADED_CONTACT_EPOCH"
    ACTIVE_ROLLING = "ACTIVE_ROLLING"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


@dataclass(frozen=True)
class ContactPersistenceSample:
    """One leg's contact persistence and measured contact-point evidence.

    A zero wheel target is command evidence only.  It defines when the
    displacement measurement is applicable; it does not prove that the wheel
    or a persistent material contact point is physically anchored.
    """

    leg: str
    time_s: float
    mode: ContactPersistenceMode
    contact_class: ContactClass
    persistence_s: float
    loaded: bool
    valid: bool
    evidence_valid: bool
    reason: str
    logical_wheel_target_rad_s: float | None = None
    physx_wheel_target_rad_s: float | None = None
    measured_contact_point_valid: bool = False
    zero_target_contact_epoch_start_s: float | None = None
    zero_target_contact_point_displacement_m: float | None = None
    zero_target_contact_epoch_max_displacement_m: float | None = None
    maximum_zero_target_contact_point_displacement_m_so_far: float | None = None
    physical_anchoring_proven: bool = False
    material_point_identity_available: bool = False
    contact_point_displacement_semantics: str = (
        "ZERO_TARGET_LOADED_MEASURED_CONTACT_POINT_DISPLACEMENT"
    )
    rolling_epoch_start_s: float | None = None
    rolling_displacement_m: float | None = None
    rolling_epoch_max_displacement_m: float | None = None
    maximum_rolling_displacement_m_so_far: float | None = None

    def as_dict(self) -> dict[str, Any]:
        row = dict(self.__dict__)
        row["mode"] = self.mode.value
        row["contact_class"] = self.contact_class.value
        return row


@dataclass
class ContactPersistenceTracker:
    """Track contact dwell and zero-target measured-point displacement.

    ``update`` keeps the historical ``(persistence, drift)`` return shape.  The
    drift mapping is finite only for a loaded, exact-zero-target measurement
    epoch; all other modes return NaN so missing or inapplicable evidence
    cannot look like a zero-displacement pass.  This tracker never claims
    physical anchoring or persistent material-point identity.
    """

    # This is the existing support/load-confirm threshold, exposed so the
    # caller can pass its already-configured value.  It is not a new tolerance.
    load_confirm_force_n: float = 2.0
    active_since_s: dict[str, float] = field(default_factory=dict)
    zero_target_contact_anchor_w: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    zero_target_contact_epoch_start_s: dict[str, float] = field(default_factory=dict)
    zero_target_contact_epoch_max_displacement_m: dict[str, float] = field(
        default_factory=dict
    )
    maximum_zero_target_contact_point_displacement_m: dict[str, float] = field(
        default_factory=dict
    )
    rolling_anchor_w: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    rolling_epoch_start_s: dict[str, float] = field(default_factory=dict)
    rolling_epoch_max_displacement_m: dict[str, float] = field(default_factory=dict)
    maximum_rolling_displacement_m: dict[str, float] = field(default_factory=dict)
    last_class: dict[str, ContactClass] = field(default_factory=dict)
    last_mode: dict[str, ContactPersistenceMode] = field(default_factory=dict)
    last_samples: dict[str, ContactPersistenceSample] = field(default_factory=dict)

    def __post_init__(self) -> None:
        threshold = float(self.load_confirm_force_n)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("load_confirm_force_n must be finite and non-negative")
        self.load_confirm_force_n = threshold

    @staticmethod
    def _flag_for_leg(value: Mapping[str, bool] | bool | None, leg: str) -> bool:
        if isinstance(value, Mapping):
            return value.get(leg) is True
        return value is True

    @staticmethod
    def _target_for_leg(
        values: Mapping[str, float] | None, leg: str
    ) -> float | None:
        if values is None or leg not in values:
            return None
        raw = values.get(leg)
        if isinstance(raw, bool):
            return None
        try:
            target = float(raw)
        except (TypeError, ValueError):
            return None
        return target if math.isfinite(target) else None

    def _clear_zero_target_contact_epoch(self, leg: str) -> None:
        self.zero_target_contact_anchor_w.pop(leg, None)
        self.zero_target_contact_epoch_start_s.pop(leg, None)
        self.zero_target_contact_epoch_max_displacement_m.pop(leg, None)

    def _clear_rolling_epoch(self, leg: str) -> None:
        self.rolling_anchor_w.pop(leg, None)
        self.rolling_epoch_start_s.pop(leg, None)
        self.rolling_epoch_max_displacement_m.pop(leg, None)

    def _reset_contact_epoch(self, leg: str) -> None:
        self.active_since_s.pop(leg, None)
        self._clear_zero_target_contact_epoch(leg)
        self._clear_rolling_epoch(leg)

    def _sample(
        self,
        *,
        leg: str,
        time_s: float,
        mode: ContactPersistenceMode,
        contact_class: ContactClass,
        persistence_s: float,
        loaded: bool,
        valid: bool,
        evidence_valid: bool,
        reason: str,
        logical_target: float | None,
        physx_target: float | None,
        measured_point_valid: bool,
        zero_target_contact_point_displacement_m: float | None = None,
        rolling_displacement_m: float | None = None,
    ) -> ContactPersistenceSample:
        return ContactPersistenceSample(
            leg=leg,
            time_s=float(time_s),
            mode=mode,
            contact_class=contact_class,
            persistence_s=float(persistence_s),
            loaded=bool(loaded),
            valid=bool(valid),
            evidence_valid=bool(evidence_valid),
            reason=str(reason),
            logical_wheel_target_rad_s=logical_target,
            physx_wheel_target_rad_s=physx_target,
            measured_contact_point_valid=bool(measured_point_valid),
            zero_target_contact_epoch_start_s=(
                self.zero_target_contact_epoch_start_s.get(leg)
            ),
            zero_target_contact_point_displacement_m=(
                zero_target_contact_point_displacement_m
            ),
            zero_target_contact_epoch_max_displacement_m=(
                self.zero_target_contact_epoch_max_displacement_m.get(leg)
            ),
            maximum_zero_target_contact_point_displacement_m_so_far=(
                self.maximum_zero_target_contact_point_displacement_m.get(leg)
            ),
            rolling_epoch_start_s=self.rolling_epoch_start_s.get(leg),
            rolling_displacement_m=rolling_displacement_m,
            rolling_epoch_max_displacement_m=self.rolling_epoch_max_displacement_m.get(leg),
            maximum_rolling_displacement_m_so_far=self.maximum_rolling_displacement_m.get(leg),
        )

    def update(
        self,
        time_s: float,
        contacts: Mapping[str, WheelContact],
        *,
        logical_wheel_target_rad_s: Mapping[str, float] | None = None,
        physx_wheel_target_rad_s: Mapping[str, float] | None = None,
        measured_contact_point_valid: Mapping[str, bool] | None = None,
        drive_evidence_valid: Mapping[str, bool] | bool | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        persistence: dict[str, float] = {}
        drift: dict[str, float] = {}
        for leg in LEGS:
            contact = contacts.get(leg)
            logical_target = self._target_for_leg(logical_wheel_target_rad_s, leg)
            physx_target = self._target_for_leg(physx_wheel_target_rad_s, leg)
            point_flag = self._flag_for_leg(measured_contact_point_valid, leg)
            drive_flag = self._flag_for_leg(drive_evidence_valid, leg)
            previous_class = self.last_class.get(leg)

            if contact is None:
                self._reset_contact_epoch(leg)
                self.last_class[leg] = ContactClass.UNKNOWN
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=ContactClass.UNKNOWN,
                    persistence_s=0.0,
                    loaded=False,
                    valid=False,
                    evidence_valid=False,
                    reason="wheel contact sample is missing",
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=False,
                )
                persistence[leg] = 0.0
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            contact_class = contact.contact_class
            class_changed = previous_class is not None and previous_class != contact_class
            if class_changed:
                self._reset_contact_epoch(leg)
            self.last_class[leg] = contact_class

            if contact_class == ContactClass.UNKNOWN:
                self._reset_contact_epoch(leg)
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=contact_class,
                    persistence_s=0.0,
                    loaded=False,
                    valid=False,
                    evidence_valid=False,
                    reason="wheel contact class is UNKNOWN",
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=False,
                )
                persistence[leg] = 0.0
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            if not contact.active or contact_class == ContactClass.AIR:
                self._reset_contact_epoch(leg)
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.NO_CONTACT,
                    contact_class=contact_class,
                    persistence_s=0.0,
                    loaded=False,
                    valid=False,
                    evidence_valid=True,
                    reason="wheel has no active contact",
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=False,
                )
                persistence[leg] = 0.0
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            load = float(contact.upward_force_n)
            if not math.isfinite(load):
                self._reset_contact_epoch(leg)
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=contact_class,
                    persistence_s=0.0,
                    loaded=False,
                    valid=False,
                    evidence_valid=False,
                    reason="upward contact load is non-finite",
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=point_flag,
                )
                persistence[leg] = 0.0
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            loaded = bool(load >= self.load_confirm_force_n)
            if not loaded:
                self._reset_contact_epoch(leg)
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.UNLOADED,
                    contact_class=contact_class,
                    persistence_s=0.0,
                    loaded=False,
                    valid=False,
                    evidence_valid=True,
                    reason="contact load is below the configured load-confirm threshold",
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=point_flag,
                )
                persistence[leg] = 0.0
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            if leg not in self.active_since_s:
                self.active_since_s[leg] = float(time_s)
            persistence_value = max(
                0.0, float(time_s) - float(self.active_since_s[leg])
            )
            persistence[leg] = persistence_value

            if contact_class not in {ContactClass.GROUND, ContactClass.TOP}:
                self._clear_zero_target_contact_epoch(leg)
                self._clear_rolling_epoch(leg)
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=contact_class,
                    persistence_s=persistence_value,
                    loaded=True,
                    valid=False,
                    evidence_valid=False,
                    reason=(
                        f"{contact_class.value} is not eligible for zero-target "
                        "loaded contact-point displacement evidence"
                    ),
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=point_flag,
                )
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            point = _finite_vector(contact.contact_point_w, 3)
            if not point_flag or point is None:
                self._clear_zero_target_contact_epoch(leg)
                self._clear_rolling_epoch(leg)
                reason = (
                    "measured contact-point validity is unavailable"
                    if not point_flag
                    else "measured contact point is non-finite"
                )
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=contact_class,
                    persistence_s=persistence_value,
                    loaded=True,
                    valid=False,
                    evidence_valid=False,
                    reason=reason,
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=False,
                )
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            if not drive_flag or logical_target is None or physx_target is None:
                self._clear_zero_target_contact_epoch(leg)
                self._clear_rolling_epoch(leg)
                reasons: list[str] = []
                if not drive_flag:
                    reasons.append("drive-target evidence is invalid")
                if logical_target is None:
                    reasons.append("logical wheel target is missing or non-finite")
                if physx_target is None:
                    reasons.append("independent PhysX velocity target is missing or non-finite")
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.INVALID_EVIDENCE,
                    contact_class=contact_class,
                    persistence_s=persistence_value,
                    loaded=True,
                    valid=False,
                    evidence_valid=False,
                    reason="; ".join(reasons),
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=True,
                )
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            current = tuple(float(value) for value in point)
            rolling = bool(logical_target != 0.0 or physx_target != 0.0)
            if rolling:
                self._clear_zero_target_contact_epoch(leg)
                if (
                    self.last_mode.get(leg) != ContactPersistenceMode.ACTIVE_ROLLING
                    or leg not in self.rolling_anchor_w
                ):
                    self.rolling_anchor_w[leg] = current
                    self.rolling_epoch_start_s[leg] = float(time_s)
                    self.rolling_epoch_max_displacement_m[leg] = 0.0
                rolling_anchor = np.asarray(self.rolling_anchor_w[leg], dtype=float)
                rolling_displacement = float(
                    np.linalg.norm(np.asarray(current, dtype=float) - rolling_anchor)
                )
                self.rolling_epoch_max_displacement_m[leg] = max(
                    self.rolling_epoch_max_displacement_m.get(leg, 0.0),
                    rolling_displacement,
                )
                self.maximum_rolling_displacement_m[leg] = max(
                    self.maximum_rolling_displacement_m.get(leg, 0.0),
                    rolling_displacement,
                )
                sample = self._sample(
                    leg=leg,
                    time_s=time_s,
                    mode=ContactPersistenceMode.ACTIVE_ROLLING,
                    contact_class=contact_class,
                    persistence_s=persistence_value,
                    loaded=True,
                    valid=False,
                    evidence_valid=True,
                    reason=(
                        "one or both wheel targets are non-zero; zero-target "
                        "contact-point displacement evidence is not applicable"
                    ),
                    logical_target=logical_target,
                    physx_target=physx_target,
                    measured_point_valid=True,
                    rolling_displacement_m=rolling_displacement,
                )
                drift[leg] = float("nan")
                self.last_mode[leg] = sample.mode
                self.last_samples[leg] = sample
                continue

            # Exactly-zero logical and independent PhysX targets define only a
            # loaded contact-point measurement epoch.  They do not prove that
            # the wheel is physically anchored or that ContactSensor reports a
            # persistent material point.  Measured motion is still retained as
            # fail-closed displacement evidence.
            self._clear_rolling_epoch(leg)
            if (
                self.last_mode.get(leg)
                != ContactPersistenceMode.ZERO_TARGET_LOADED_CONTACT_EPOCH
                or leg not in self.zero_target_contact_anchor_w
            ):
                self.zero_target_contact_anchor_w[leg] = current
                self.zero_target_contact_epoch_start_s[leg] = float(time_s)
                self.zero_target_contact_epoch_max_displacement_m[leg] = 0.0
            anchor = np.asarray(self.zero_target_contact_anchor_w[leg], dtype=float)
            displacement = float(
                np.linalg.norm(np.asarray(current, dtype=float) - anchor)
            )
            self.zero_target_contact_epoch_max_displacement_m[leg] = max(
                self.zero_target_contact_epoch_max_displacement_m.get(leg, 0.0),
                displacement,
            )
            self.maximum_zero_target_contact_point_displacement_m[leg] = max(
                self.maximum_zero_target_contact_point_displacement_m.get(leg, 0.0),
                displacement,
            )
            sample = self._sample(
                leg=leg,
                time_s=time_s,
                mode=ContactPersistenceMode.ZERO_TARGET_LOADED_CONTACT_EPOCH,
                contact_class=contact_class,
                persistence_s=persistence_value,
                loaded=True,
                valid=True,
                evidence_valid=True,
                reason="",
                logical_target=logical_target,
                physx_target=physx_target,
                measured_point_valid=True,
                zero_target_contact_point_displacement_m=displacement,
            )
            drift[leg] = displacement
            self.last_mode[leg] = sample.mode
            self.last_samples[leg] = sample
        return persistence, drift


class TraversalEpisodeStatus(str, Enum):
    """Lifecycle state for one continuous wheel-lift attempt."""

    ACTIVE_UNARMED = "ACTIVE_UNARMED"
    ACTIVE_PRE_FACE = "ACTIVE_PRE_FACE"
    ACTIVE_POST_FACE = "ACTIVE_POST_FACE"
    TOP_LOAD_PENDING = "TOP_LOAD_PENDING"
    INCOMPLETE_PRE_FACE = "INCOMPLETE_PRE_FACE"
    ILLEGAL_POST_FACE_NON_TOP = "ILLEGAL_POST_FACE_NON_TOP"
    ILLEGAL_SEQUENCE = "ILLEGAL_SEQUENCE"
    VALID = "VALID"
    IGNORED_UNARMED = "IGNORED_UNARMED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class TraversalEpisode:
    """Evidence belonging to one attempt; terminal episodes are immutable by use."""

    episode_index: int
    started_s: float
    status: TraversalEpisodeStatus = TraversalEpisodeStatus.ACTIVE_UNARMED
    ended_s: float | None = None
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
    top_load_since_s: float | None = None
    airborne_seen_before_top: bool = False
    illegal_reasons: list[str] = field(default_factory=list)
    evidence_reason: str = ""

    def linkage_lift_valid(
        self, *, minimum_airborne_s: float, global_illegal_reasons: Sequence[str]
    ) -> bool:
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
            >= float(minimum_airborne_s)
            and not self.illegal_reasons
            and not global_illegal_reasons
        )


@dataclass
class LegTraversalEvidence:
    leg: str
    minimum_airborne_s: float = 0.05
    minimum_unload_dwell_s: float = 0.025

    # Legacy summary fields.  They mirror the canonical valid episode when one
    # exists, otherwise the current/latest raw episode.  Keeping these fields
    # avoids silently breaking existing artifact readers while raw episodes
    # prevent evidence from separate attempts being stitched together.
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

    episodes: list[TraversalEpisode] = field(default_factory=list)
    active_episode: TraversalEpisode | None = None
    canonical_episode_index: int | None = None
    next_episode_index: int = 0

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

    @staticmethod
    def _episode_has_progress(episode: TraversalEpisode) -> bool:
        return bool(
            episode.loaded_support_seen_s is not None
            or episode.unload_candidate_s is not None
            or episode.unload_start_s is not None
            or episode.airborne_candidate_s is not None
            or episode.airborne_start_s is not None
            or episode.front_face_crossing_s is not None
            or episode.top_contact_s is not None
            or episode.illegal_reasons
        )

    @staticmethod
    def _append_illegal(
        evidence: LegTraversalEvidence,
        episode: TraversalEpisode | None,
        reason: str,
    ) -> None:
        if reason not in evidence.illegal_reasons:
            evidence.illegal_reasons.append(reason)
        if episode is not None and reason not in episode.illegal_reasons:
            episode.illegal_reasons.append(reason)

    @staticmethod
    def _episode_by_index(
        evidence: LegTraversalEvidence, episode_index: int | None
    ) -> TraversalEpisode | None:
        if episode_index is None:
            return None
        for episode in evidence.episodes:
            if episode.episode_index == episode_index:
                return episode
        active = evidence.active_episode
        if active is not None and active.episode_index == episode_index:
            return active
        return None

    def _new_episode(
        self, evidence: LegTraversalEvidence, time_s: float
    ) -> TraversalEpisode:
        episode = TraversalEpisode(
            episode_index=evidence.next_episode_index,
            started_s=float(time_s),
        )
        evidence.next_episode_index += 1
        evidence.active_episode = episode
        return episode

    @staticmethod
    def _archive_episode(
        evidence: LegTraversalEvidence,
        episode: TraversalEpisode,
        *,
        status: TraversalEpisodeStatus,
        time_s: float,
        reason: str = "",
    ) -> None:
        episode.status = status
        episode.ended_s = float(time_s)
        episode.evidence_reason = str(reason)
        evidence.episodes.append(episode)
        evidence.active_episode = None
        if status == TraversalEpisodeStatus.VALID:
            evidence.canonical_episode_index = episode.episode_index

    def _summary_episode(
        self, evidence: LegTraversalEvidence
    ) -> TraversalEpisode | None:
        canonical = self._episode_by_index(
            evidence, evidence.canonical_episode_index
        )
        if canonical is not None:
            return canonical
        if evidence.active_episode is not None:
            return evidence.active_episode
        return evidence.episodes[-1] if evidence.episodes else None

    def _sync_legacy_summary(self, evidence: LegTraversalEvidence) -> None:
        episode = self._summary_episode(evidence)
        fields = (
            "loaded_support_seen_s",
            "unload_candidate_s",
            "airborne_candidate_s",
            "unload_start_s",
            "unload_end_s",
            "airborne_start_s",
            "airborne_end_s",
            "maximum_clearance_m",
            "front_face_crossing_s",
            "top_contact_s",
            "top_load_confirm_s",
            "top_load_since_s",
            "airborne_seen_before_top",
        )
        if episode is None:
            defaults: dict[str, Any] = {
                "maximum_clearance_m": -math.inf,
                "airborne_seen_before_top": False,
            }
            for name in fields:
                setattr(evidence, name, defaults.get(name))
            return
        for name in fields:
            setattr(evidence, name, getattr(episode, name))

    def _observe_load(
        self,
        episode: TraversalEpisode,
        *,
        time_s: float,
        contact: WheelContact,
        load: float,
    ) -> None:
        if not math.isfinite(load):
            return
        episode.minimum_load_n = min(episode.minimum_load_n, max(0.0, load))
        if contact.active and load >= self.load_confirm_force_n:
            if episode.loaded_support_seen_s is None:
                episode.loaded_support_seen_s = float(time_s)
        if episode.unload_start_s is not None:
            return
        if episode.loaded_support_seen_s is not None and load <= self.unload_force_n:
            if episode.unload_candidate_s is None:
                episode.unload_candidate_s = float(time_s)
            if (
                float(time_s) - float(episode.unload_candidate_s)
                >= self.minimum_unload_dwell_s
            ):
                episode.unload_start_s = float(episode.unload_candidate_s)
        elif load > self.unload_force_n:
            episode.unload_candidate_s = None

    def _start_from_current_contact(
        self,
        evidence: LegTraversalEvidence,
        *,
        time_s: float,
        contact: WheelContact,
        load: float,
    ) -> TraversalEpisode:
        """Start a fresh attempt and use the current sample only as its baseline."""

        episode = self._new_episode(evidence, time_s)
        self._observe_load(
            episode,
            time_s=time_s,
            contact=contact,
            load=load,
        )
        evidence.last_class = contact.contact_class
        return episode

    def _invalidate_unknown(
        self, evidence: LegTraversalEvidence, *, time_s: float, reason: str
    ) -> None:
        episode = evidence.active_episode
        if episode is not None:
            status = (
                TraversalEpisodeStatus.NOT_EVALUABLE
                if self._episode_has_progress(episode)
                else TraversalEpisodeStatus.IGNORED_UNARMED
            )
            self._archive_episode(
                evidence,
                episode,
                status=status,
                time_s=time_s,
                reason=reason,
            )
        evidence.last_class = ContactClass.UNKNOWN
        self._sync_legacy_summary(evidence)

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
                self._invalidate_unknown(
                    evidence,
                    time_s=time_s,
                    reason="wheel contact sample unavailable",
                )
                continue
            angle = float(wheel_angles_rad.get(leg, float("nan")))
            delta = 0.0
            if evidence.last_angle_rad is not None and math.isfinite(angle):
                delta = angle - evidence.last_angle_rad
            if math.isfinite(angle):
                evidence.last_angle_rad = angle

            if contact.contact_class == ContactClass.UNKNOWN:
                self._invalidate_unknown(
                    evidence,
                    time_s=time_s,
                    reason="wheel contact geometry is UNKNOWN",
                )
                continue

            load = float(contact.upward_force_n)
            if math.isfinite(load):
                evidence.minimum_load_n = min(evidence.minimum_load_n, max(0.0, load))

            episode = evidence.active_episode
            canonical_complete = bool(
                evidence.canonical_episode_index is not None and episode is None
            )
            if episode is None and not canonical_complete:
                episode = self._new_episode(evidence, time_s)
            if episode is not None:
                self._observe_load(
                    episode,
                    time_s=time_s,
                    contact=contact,
                    load=load,
                )
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
                ):
                    self._append_illegal(
                        evidence,
                        episode,
                        "loaded front-face wheel rotation exceeded limit"
                    )
            if canonical_complete:
                # Freeze the successful raw episode.  Later samples still
                # contribute to the global loaded-front-face rotation safety
                # check above, but cannot rewrite or extend its evidence chain.
                evidence.last_class = contact.contact_class
                self._sync_legacy_summary(evidence)
                continue

            assert episode is not None
            if contact.contact_class == ContactClass.AIR:
                if episode.airborne_candidate_s is None:
                    episode.airborne_candidate_s = float(time_s)
                # Sensor warm-up may briefly report every wheel AIR.  A lift
                # is admissible only after that wheel first demonstrated a
                # loaded support contact and then sustained an unload dwell.
                if episode.unload_start_s is not None:
                    if episode.airborne_start_s is None:
                        episode.airborne_start_s = float(episode.airborne_candidate_s)
                    episode.airborne_seen_before_top = True
                    episode.status = TraversalEpisodeStatus.ACTIVE_PRE_FACE
                    clearance = float(contact.clearance_over_top_m)
                    if math.isfinite(clearance):
                        episode.maximum_clearance_m = max(
                            episode.maximum_clearance_m, clearance
                        )
                crossed_now = bool(
                    math.isfinite(float(contact.front_face_clearance_m))
                    and float(contact.front_face_clearance_m) >= 0.0
                )
                if crossed_now and episode.front_face_crossing_s is None:
                    episode.front_face_crossing_s = float(time_s)
                    if not episode.airborne_seen_before_top:
                        self._append_illegal(
                            evidence,
                            episode,
                            "front face crossed without prior airborne interval",
                        )
                    if (
                        not math.isfinite(float(contact.clearance_over_top_m))
                        or float(contact.clearance_over_top_m) < 0.0
                    ):
                        self._append_illegal(
                            evidence,
                            episode,
                            "front face crossed before wheel bottom cleared obstacle top",
                        )
                    if episode.airborne_seen_before_top:
                        episode.status = TraversalEpisodeStatus.ACTIVE_POST_FACE
                evidence.last_class = ContactClass.AIR
                self._sync_legacy_summary(evidence)
                continue

            if evidence.last_class == ContactClass.AIR:
                if episode.airborne_start_s is not None:
                    episode.airborne_end_s = float(time_s)
                    crossed_now = bool(
                        math.isfinite(float(contact.front_face_clearance_m))
                        and float(contact.front_face_clearance_m) >= 0.0
                    )
                    if contact.contact_class != ContactClass.TOP:
                        if episode.front_face_crossing_s is not None or crossed_now:
                            if episode.front_face_crossing_s is None:
                                episode.front_face_crossing_s = float(time_s)
                            if (
                                not math.isfinite(float(contact.clearance_over_top_m))
                                or float(contact.clearance_over_top_m) < 0.0
                            ):
                                self._append_illegal(
                                    evidence,
                                    episode,
                                    "front face crossed before wheel bottom cleared obstacle top",
                                )
                            self._append_illegal(
                                evidence,
                                episode,
                                "airborne attempt ended after front-face crossing on non-TOP contact",
                            )
                            self._archive_episode(
                                evidence,
                                episode,
                                status=TraversalEpisodeStatus.ILLEGAL_POST_FACE_NON_TOP,
                                time_s=time_s,
                                reason="post-face airborne attempt ended on a non-TOP contact",
                            )
                        else:
                            # A retry that lands before the obstacle face is
                            # diagnostic, not a dangerous drive-up.  Archive it
                            # and start from the current sample so no timestamp
                            # can leak into the next attempt.
                            self._archive_episode(
                                evidence,
                                episode,
                                status=TraversalEpisodeStatus.INCOMPLETE_PRE_FACE,
                                time_s=time_s,
                                reason="airborne attempt ended before front-face crossing",
                            )
                        self._start_from_current_contact(
                            evidence,
                            time_s=time_s,
                            contact=contact,
                            load=load,
                        )
                        self._sync_legacy_summary(evidence)
                        continue
                else:
                    # Unarmed AIR (for example sensor warm-up) cannot supply
                    # lift evidence to the next attempt.
                    self._archive_episode(
                        evidence,
                        episode,
                        status=TraversalEpisodeStatus.IGNORED_UNARMED,
                        time_s=time_s,
                        reason="AIR interval was not preceded by loaded support and unload dwell",
                    )
                    episode = self._start_from_current_contact(
                        evidence,
                        time_s=time_s,
                        contact=contact,
                        load=load,
                    )

            if (
                math.isfinite(float(contact.front_face_clearance_m))
                and contact.front_face_clearance_m >= 0.0
                and episode.front_face_crossing_s is None
            ):
                episode.front_face_crossing_s = float(time_s)
                if not episode.airborne_seen_before_top:
                    self._append_illegal(
                        evidence,
                        episode,
                        "front face crossed without prior airborne interval"
                    )
                if (
                    not math.isfinite(float(contact.clearance_over_top_m))
                    or float(contact.clearance_over_top_m) < 0.0
                ):
                    self._append_illegal(
                        evidence,
                        episode,
                        "front face crossed before wheel bottom cleared obstacle top"
                    )
                if episode.airborne_seen_before_top:
                    episode.status = TraversalEpisodeStatus.ACTIVE_POST_FACE
            if contact.contact_class == ContactClass.TOP:
                if episode.top_contact_s is None:
                    episode.top_contact_s = float(time_s)
                    if not episode.airborne_seen_before_top:
                        self._append_illegal(
                            evidence,
                            episode,
                            "GROUND/FRONT_FACE to TOP transition without AIR"
                        )
                    if episode.front_face_crossing_s is None:
                        self._append_illegal(
                            evidence,
                            episode,
                            "top contact occurred before wheel center crossed front face"
                        )
                episode.status = TraversalEpisodeStatus.TOP_LOAD_PENDING
                if math.isfinite(load) and load >= self.load_confirm_force_n:
                    if episode.top_load_since_s is None:
                        episode.top_load_since_s = float(time_s)
                    if (
                        episode.top_load_confirm_s is None
                        and float(time_s) - episode.top_load_since_s
                        >= self.top_load_dwell_s
                    ):
                        episode.top_load_confirm_s = float(time_s)
                        episode.unload_end_s = episode.airborne_start_s
                        if episode.linkage_lift_valid(
                            minimum_airborne_s=self.minimum_airborne_s,
                            global_illegal_reasons=(),
                        ):
                            self._archive_episode(
                                evidence,
                                episode,
                                status=TraversalEpisodeStatus.VALID,
                                time_s=time_s,
                            )
                        elif episode.illegal_reasons:
                            self._archive_episode(
                                evidence,
                                episode,
                                status=TraversalEpisodeStatus.ILLEGAL_SEQUENCE,
                                time_s=time_s,
                                reason="TOP load confirmed after an illegal traversal sequence",
                            )
                else:
                    episode.top_load_since_s = None
            else:
                episode.top_load_since_s = None
            evidence.last_class = contact.contact_class
            self._sync_legacy_summary(evidence)

    @staticmethod
    def _episode_row(
        episode: TraversalEpisode,
        *,
        minimum_airborne_s: float,
        global_illegal_reasons: Sequence[str],
    ) -> dict[str, Any]:
        row = {
            key: value
            for key, value in episode.__dict__.items()
            if key not in {"status"}
        }
        row["status"] = episode.status.value
        row["linkage_lift_valid"] = episode.linkage_lift_valid(
            minimum_airborne_s=minimum_airborne_s,
            global_illegal_reasons=global_illegal_reasons,
        )
        if not math.isfinite(float(row["maximum_clearance_m"])):
            row["maximum_clearance_m"] = None
        if not math.isfinite(float(row["minimum_load_n"])):
            row["minimum_load_n"] = None
        return row

    def result(self) -> dict[str, Any]:
        legs: dict[str, Any] = {}
        for leg, evidence in self.legs.items():
            self._sync_legacy_summary(evidence)
            excluded = {
                "last_angle_rad",
                "last_class",
                "top_load_since_s",
                "unload_candidate_s",
                "airborne_candidate_s",
                "episodes",
                "active_episode",
                "next_episode_index",
            }
            row = {
                key: value
                for key, value in evidence.__dict__.items()
                if key not in excluded
            }
            row["last_class"] = evidence.last_class.value
            row["linkage_lift_valid"] = evidence.linkage_lift_valid
            if not math.isfinite(float(row["maximum_clearance_m"])):
                row["maximum_clearance_m"] = None
            if not math.isfinite(float(row["minimum_load_n"])):
                row["minimum_load_n"] = None
            raw_episodes = list(evidence.episodes)
            if evidence.active_episode is not None:
                raw_episodes.append(evidence.active_episode)
            row["episodes"] = [
                self._episode_row(
                    episode,
                    minimum_airborne_s=evidence.minimum_airborne_s,
                    global_illegal_reasons=evidence.illegal_reasons,
                )
                for episode in raw_episodes
            ]
            row["active_episode_status"] = (
                evidence.active_episode.status.value
                if evidence.active_episode is not None
                else None
            )
            canonical = self._episode_by_index(
                evidence, evidence.canonical_episode_index
            )
            row["canonical_episode"] = (
                self._episode_row(
                    canonical,
                    minimum_airborne_s=evidence.minimum_airborne_s,
                    global_illegal_reasons=evidence.illegal_reasons,
                )
                if canonical is not None
                else None
            )
            legs[leg] = row
        return {
            "legs": legs,
            "all_legs_valid": all(item.linkage_lift_valid for item in self.legs.values()),
            "any_illegal_drive_up": any(item.illegal_reasons for item in self.legs.values()),
        }
