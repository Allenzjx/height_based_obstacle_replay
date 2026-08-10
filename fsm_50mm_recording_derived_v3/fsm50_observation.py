"""Isaac-independent, fail-closed observations for the 50 mm FSM runtime.

The simulator-facing adapter is expected to produce a plain mapping. This
module validates that mapping and turns it into one immutable observation per
physics sample. Invalid or unavailable evidence is never replaced with a
plausible value: FSM50Observation.control_ready becomes False and guards can
use FSM50Observation.guard_allows to fail closed.

Both the canonical field names defined by FSM50Observation and the current
FSM50TelemetryCollector row names are accepted. There are no Isaac, Isaac Lab,
torch, or numpy imports, which keeps fake-controller tests cheap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias


LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")
DIAGONALS: tuple[str, ...] = ("FL_RR", "FR_RL")

Vector3: TypeAlias = tuple[float, float, float]
QuaternionWXYZ: TypeAlias = tuple[float, float, float, float]

_LEG_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "fl": "FL",
        "front_left": "FL",
        "front_left_ankle": "FL",
        "front_left_wheel": "FL",
        "fr": "FR",
        "front_right": "FR",
        "front_right_ankle": "FR",
        "front_right_wheel": "FR",
        "rl": "RL",
        "rear_left": "RL",
        "rear_left_ankle": "RL",
        "rear_left_wheel": "RL",
        "rr": "RR",
        "rear_right": "RR",
        "rear_right_ankle": "RR",
        "rear_right_wheel": "RR",
    }
)
_MISSING = object()


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
    NONE = "NONE"
    DYNAMIC = "DYNAMIC"


@dataclass(frozen=True, slots=True)
class ObservationIssue:
    """One mapping-validation problem."""

    field: str
    code: str
    message: str
    critical: bool = True

    def __str__(self) -> str:
        return f"{self.field}: {self.code}: {self.message}"


class ObservationValidationError(ValueError):
    """Raised by strict construction when observation evidence is invalid."""

    def __init__(self, issues: Sequence[ObservationIssue]):
        self.issues = tuple(issues)
        detail = "; ".join(str(issue) for issue in self.issues)
        super().__init__(detail or "observation validation failed")


class CriticalTelemetryUnavailable(RuntimeError):
    """Raised when code explicitly requires a control-ready observation."""


@dataclass(frozen=True, slots=True)
class TwoLegCorridorObservation:
    applicable: bool
    valid: bool
    perpendicular_distance_m: float | None
    segment_fraction: float | None
    within_longitudinal_bounds: bool | None = None
    within_corridor_width: bool | None = None


@dataclass(frozen=True, slots=True)
class COMTargetDirection:
    """Geometric COM target resolved from the current measured frame."""

    target_leg: str
    target_contact_w: Vector3
    direction_w: Vector3
    direction_body: Vector3
    distance_m: float
    horizontal_only: bool
    source: str = "MEASURED_TARGET_CONTACT_GEOMETRY"


def _canonical_leg(value: Any) -> str | None:
    if isinstance(value, Enum):
        value = value.value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _LEG_ALIASES.get(key)


def _enum_text(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value).strip().upper()


def _raw_finite_vector(value: Any, size: int, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a length-{size} sequence")
    if len(value) != size:
        raise ValueError(f"{label} must have shape ({size},), got ({len(value)},)")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain real numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain only finite values")
    return result


def _normalize_quaternion(value: Any, *, label: str) -> QuaternionWXYZ:
    q = _raw_finite_vector(value, 4, label=label)
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} must have non-zero norm")
    return tuple(component / norm for component in q)  # type: ignore[return-value]


def _rotate_by_quaternion(q: QuaternionWXYZ, vector: Vector3) -> Vector3:
    """Rotate a vector by a normalized wxyz quaternion."""

    w, x, y, z = q
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def compute_world_com_target_direction(
    *,
    root_orientation_wxyz: Sequence[float],
    com_position_w: Sequence[float],
    target_contact_w: Sequence[float],
    target_leg: str,
    horizontal_only: bool = True,
) -> COMTargetDirection:
    """Resolve a world COM direction from measured geometry.

    The target-contact/COM delta is transformed into the current body frame for
    diagnostic signs and then back through the measured orientation. A YAML
    direction hint is deliberately not an input.
    """

    leg = _canonical_leg(target_leg)
    if leg is None:
        raise ValueError(f"target_leg must identify one of {LEGS}, got {target_leg!r}")
    q = _normalize_quaternion(root_orientation_wxyz, label="root_orientation_wxyz")
    com = _raw_finite_vector(com_position_w, 3, label="com_position_w")
    target = _raw_finite_vector(target_contact_w, 3, label="target_contact_w")
    delta: Vector3 = (
        target[0] - com[0],
        target[1] - com[1],
        0.0 if horizontal_only else target[2] - com[2],
    )
    distance = math.sqrt(sum(component * component for component in delta))
    if distance <= 1.0e-12:
        qualifier = "horizontal " if horizontal_only else ""
        raise ValueError(f"target contact has zero {qualifier}distance from COM")
    direct_world: Vector3 = tuple(
        component / distance for component in delta
    )  # type: ignore[assignment]
    conjugate: QuaternionWXYZ = (q[0], -q[1], -q[2], -q[3])
    direction_body = _rotate_by_quaternion(conjugate, direct_world)
    reconstructed_world = _rotate_by_quaternion(q, direction_body)
    reconstructed_norm = math.sqrt(
        sum(component * component for component in reconstructed_world)
    )
    direction_world: Vector3 = tuple(
        component / reconstructed_norm for component in reconstructed_world
    )  # type: ignore[assignment]
    return COMTargetDirection(
        target_leg=leg,
        target_contact_w=target,  # type: ignore[arg-type]
        direction_w=direction_world,
        direction_body=direction_body,
        distance_m=distance,
        horizontal_only=bool(horizontal_only),
    )


class _Builder:
    def __init__(self, source: Mapping[str, Any]):
        self.source = source
        self.issues: list[ObservationIssue] = []

    def issue(self, field: str, code: str, message: str) -> None:
        self.issues.append(ObservationIssue(field, code, message))

    def lookup(self, *names: str) -> Any:
        for name in names:
            if name in self.source:
                return self.source[name]
        return _MISSING

    def scalar(
        self,
        field: str,
        *aliases: str,
        required: bool = True,
        nonnegative: bool = False,
        raw: Any = _MISSING,
    ) -> float | None:
        value = self.lookup(field, *aliases) if raw is _MISSING else raw
        if value is _MISSING or value is None:
            if required:
                self.issue(field, "MISSING", "required finite scalar is unavailable")
            return None
        if isinstance(value, bool):
            self.issue(field, "TYPE", "boolean is not a valid real scalar")
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            self.issue(field, "TYPE", f"expected a real scalar, got {value!r}")
            return None
        if not math.isfinite(result):
            if required:
                self.issue(field, "NONFINITE", f"expected a finite scalar, got {result!r}")
            return None
        if nonnegative and result < 0.0:
            self.issue(field, "RANGE", f"expected a non-negative scalar, got {result!r}")
            return None
        return result

    def boolean(
        self,
        field: str,
        *aliases: str,
        required: bool = True,
        raw: Any = _MISSING,
    ) -> bool | None:
        value = self.lookup(field, *aliases) if raw is _MISSING else raw
        if value is _MISSING or value is None:
            if required:
                self.issue(field, "MISSING", "required boolean evidence is unavailable")
            return None
        if not isinstance(value, bool):
            self.issue(field, "TYPE", f"expected bool, got {type(value).__name__}")
            return None
        return value

    def vector(
        self,
        field: str,
        size: int,
        *aliases: str,
        components: tuple[str, ...] = (),
    ) -> tuple[float, ...] | None:
        value = self.lookup(field, *aliases)
        if value is _MISSING and components and all(
            component in self.source for component in components
        ):
            value = tuple(self.source[component] for component in components)
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", f"required shape ({size},) vector is unavailable")
            return None
        try:
            return _raw_finite_vector(value, size, label=field)
        except ValueError as exc:
            message = str(exc)
            code = "NONFINITE" if "finite" in message else "SHAPE_OR_TYPE"
            self.issue(field, code, message)
            return None

    def quaternion(
        self,
        field: str,
        *aliases: str,
        components: tuple[str, ...] = (),
    ) -> QuaternionWXYZ | None:
        value = self.lookup(field, *aliases)
        if value is _MISSING and components and all(
            component in self.source for component in components
        ):
            value = tuple(self.source[component] for component in components)
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", "required wxyz quaternion is unavailable")
            return None
        try:
            return _normalize_quaternion(value, label=field)
        except ValueError as exc:
            message = str(exc)
            code = "NONFINITE" if "finite" in message else "SHAPE_OR_TYPE"
            self.issue(field, code, message)
            return None

    def scalar_mapping_value(
        self,
        field: str,
        value: Any,
        *,
        required: bool = True,
        nonempty: bool = True,
        nonnegative: bool = False,
        excluded_keys: frozenset[str] = frozenset(),
    ) -> dict[str, float]:
        if value is _MISSING or value is None:
            if required:
                self.issue(field, "MISSING", "required scalar mapping is unavailable")
            return {}
        if not isinstance(value, Mapping):
            self.issue(field, "TYPE", f"expected mapping, got {type(value).__name__}")
            return {}
        result: dict[str, float] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key or key in excluded_keys:
                continue
            if isinstance(raw_value, bool):
                self.issue(f"{field}.{key}", "TYPE", "boolean is not a valid scalar")
                continue
            try:
                number = float(raw_value)
            except (TypeError, ValueError):
                self.issue(
                    f"{field}.{key}", "TYPE", f"expected real scalar, got {raw_value!r}"
                )
                continue
            if not math.isfinite(number):
                self.issue(f"{field}.{key}", "NONFINITE", f"got {number!r}")
                continue
            if nonnegative and number < 0.0:
                self.issue(
                    f"{field}.{key}", "RANGE", f"expected non-negative, got {number!r}"
                )
                continue
            result[key] = number
        if required and nonempty and not result:
            self.issue(field, "EMPTY", "at least one finite entry is required")
        return result

    def scalar_mapping(
        self,
        field: str,
        *aliases: str,
        required: bool = True,
        nonempty: bool = True,
        nonnegative: bool = False,
    ) -> dict[str, float]:
        return self.scalar_mapping_value(
            field,
            self.lookup(field, *aliases),
            required=required,
            nonempty=nonempty,
            nonnegative=nonnegative,
        )

    def leg_scalar_mapping_value(
        self,
        field: str,
        value: Any,
        *,
        nonnegative: bool = False,
    ) -> dict[str, float]:
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", "required four-leg scalar mapping is unavailable")
            return {}
        if not isinstance(value, Mapping):
            self.issue(field, "TYPE", f"expected mapping, got {type(value).__name__}")
            return {}
        result: dict[str, float] = {}
        for raw_key, raw_value in value.items():
            leg = _canonical_leg(raw_key)
            if leg is None:
                self.issue(f"{field}.{raw_key}", "LEG_KEY", f"expected one of {LEGS}")
                continue
            if leg in result:
                self.issue(f"{field}.{raw_key}", "DUPLICATE_LEG", f"duplicates {leg}")
                continue
            if isinstance(raw_value, bool):
                self.issue(f"{field}.{leg}", "TYPE", "boolean is not a valid scalar")
                continue
            try:
                number = float(raw_value)
            except (TypeError, ValueError):
                self.issue(
                    f"{field}.{leg}", "TYPE", f"expected real scalar, got {raw_value!r}"
                )
                continue
            if not math.isfinite(number):
                self.issue(f"{field}.{leg}", "NONFINITE", f"got {number!r}")
                continue
            if nonnegative and number < 0.0:
                self.issue(
                    f"{field}.{leg}", "RANGE", f"expected non-negative, got {number!r}"
                )
                continue
            result[leg] = number
        missing = tuple(leg for leg in LEGS if leg not in result)
        if missing:
            self.issue(field, "LEG_KEYS", f"missing canonical legs: {missing}")
        return result

    def leg_scalar_mapping(
        self, field: str, *aliases: str, nonnegative: bool = False
    ) -> dict[str, float]:
        return self.leg_scalar_mapping_value(
            field, self.lookup(field, *aliases), nonnegative=nonnegative
        )

    def leg_vector_mapping(
        self, field: str, size: int, *aliases: str
    ) -> dict[str, tuple[float, ...]]:
        value = self.lookup(field, *aliases)
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", "required four-leg vector mapping is unavailable")
            return {}
        if not isinstance(value, Mapping):
            self.issue(field, "TYPE", f"expected mapping, got {type(value).__name__}")
            return {}
        result: dict[str, tuple[float, ...]] = {}
        for raw_key, raw_value in value.items():
            leg = _canonical_leg(raw_key)
            if leg is None:
                self.issue(f"{field}.{raw_key}", "LEG_KEY", f"expected one of {LEGS}")
                continue
            if leg in result:
                self.issue(f"{field}.{raw_key}", "DUPLICATE_LEG", f"duplicates {leg}")
                continue
            try:
                result[leg] = _raw_finite_vector(
                    raw_value, size, label=f"{field}.{leg}"
                )
            except ValueError as exc:
                message = str(exc)
                code = "NONFINITE" if "finite" in message else "SHAPE_OR_TYPE"
                self.issue(f"{field}.{leg}", code, message)
        missing = tuple(leg for leg in LEGS if leg not in result)
        if missing:
            self.issue(field, "LEG_KEYS", f"missing canonical legs: {missing}")
        return result

    def contact_class_mapping(
        self, field: str, *aliases: str
    ) -> dict[str, ContactClass]:
        value = self.lookup(field, *aliases)
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", "required four-leg contact classes are unavailable")
            return {}
        if not isinstance(value, Mapping):
            self.issue(field, "TYPE", f"expected mapping, got {type(value).__name__}")
            return {}
        result: dict[str, ContactClass] = {}
        for raw_key, raw_value in value.items():
            leg = _canonical_leg(raw_key)
            if leg is None:
                self.issue(f"{field}.{raw_key}", "LEG_KEY", f"expected one of {LEGS}")
                continue
            if leg in result:
                self.issue(f"{field}.{raw_key}", "DUPLICATE_LEG", f"duplicates {leg}")
                continue
            try:
                result[leg] = ContactClass(_enum_text(raw_value))
            except ValueError:
                self.issue(
                    f"{field}.{leg}", "ENUM", f"unknown contact class {raw_value!r}"
                )
        missing = tuple(leg for leg in LEGS if leg not in result)
        if missing:
            self.issue(field, "LEG_KEYS", f"missing canonical legs: {missing}")
        return result

    def legs(self, field: str, *aliases: str) -> tuple[str, ...]:
        value = self.lookup(field, *aliases)
        if value is _MISSING or value is None:
            self.issue(field, "MISSING", "required leg sequence is unavailable")
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            self.issue(field, "TYPE", "expected a sequence of leg identifiers")
            return ()
        result: list[str] = []
        for raw_leg in value:
            leg = _canonical_leg(raw_leg)
            if leg is None:
                self.issue(field, "LEG_KEY", f"unknown leg identifier {raw_leg!r}")
                continue
            if leg in result:
                self.issue(field, "DUPLICATE_LEG", f"duplicate leg {leg}")
                continue
            result.append(leg)
        return tuple(result)


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


def _surface_force_maps(builder: _Builder) -> tuple[dict[str, float], dict[str, float]]:
    ground_raw = builder.lookup("filtered_ground_force_n")
    obstacle_raw = builder.lookup("filtered_obstacle_force_n")
    nested = builder.lookup("wheel_surface_normal_force_n")
    if (ground_raw is _MISSING or obstacle_raw is _MISSING) and isinstance(
        nested, Mapping
    ):
        ground_raw = {}
        obstacle_raw = {}
        for raw_leg, surfaces in nested.items():
            if not isinstance(surfaces, Mapping):
                builder.issue(
                    f"wheel_surface_normal_force_n.{raw_leg}",
                    "TYPE",
                    "expected ground/obstacle mapping",
                )
                continue
            ground_raw[raw_leg] = surfaces.get("ground", _MISSING)
            obstacle_raw[raw_leg] = surfaces.get("obstacle", _MISSING)
    return (
        builder.leg_scalar_mapping_value(
            "filtered_ground_force_n", ground_raw, nonnegative=True
        ),
        builder.leg_scalar_mapping_value(
            "filtered_obstacle_force_n", obstacle_raw, nonnegative=True
        ),
    )


def _servo_targets(builder: _Builder) -> dict[str, float]:
    raw = builder.lookup("servo_target_rad", "servo_targets_rad")
    filter_wheels = False
    scale = 1.0
    if raw is _MISSING:
        raw = builder.lookup("actual_joint_target_rad")
        filter_wheels = raw is not _MISSING
    if raw is _MISSING:
        raw = builder.lookup("command_space_servo_target_deg")
        scale = math.pi / 180.0
    excluded: frozenset[str] = frozenset()
    if filter_wheels and isinstance(raw, Mapping):
        excluded = frozenset(str(key) for key in raw if _canonical_leg(key) is not None)
    parsed = builder.scalar_mapping_value(
        "servo_target_rad", raw, excluded_keys=excluded
    )
    return {key: value * scale for key, value in parsed.items()}


def _diagonal_load_share(builder: _Builder) -> dict[str, float]:
    raw = builder.lookup("diagonal_load_share")
    if raw is _MISSING:
        raw = {
            "FL_RR": builder.lookup("diagonal_load_share_fl_rr"),
            "FR_RL": builder.lookup("diagonal_load_share_fr_rl"),
        }
    if not isinstance(raw, Mapping):
        builder.issue("diagonal_load_share", "TYPE", "expected mapping")
        return {}
    result: dict[str, float] = {}
    unexpected = tuple(str(key) for key in raw if _enum_text(key) not in DIAGONALS)
    if unexpected:
        builder.issue(
            "diagonal_load_share", "DIAGONAL_KEYS", f"unexpected keys: {unexpected}"
        )
    for diagonal in DIAGONALS:
        value = _MISSING
        for raw_key, raw_value in raw.items():
            if _enum_text(raw_key) == diagonal:
                value = raw_value
                break
        number = builder.scalar(
            f"diagonal_load_share.{diagonal}", raw=value, required=True
        )
        if number is not None:
            if not 0.0 <= number <= 1.0:
                builder.issue(
                    f"diagonal_load_share.{diagonal}",
                    "RANGE",
                    f"load share must be in [0, 1], got {number}",
                )
            else:
                result[diagonal] = number
    if len(result) == 2 and not math.isclose(
        sum(result.values()), 1.0, abs_tol=1.0e-5
    ):
        builder.issue(
            "diagonal_load_share",
            "SUM",
            f"FL_RR and FR_RL shares must sum to 1, got {sum(result.values())}",
        )
    return result


@dataclass(frozen=True, slots=True)
class FSM50Observation:
    """One validated, deeply immutable FSM physics observation."""

    time_s: float | None
    root_position_w: Vector3 | None
    root_orientation_wxyz: QuaternionWXYZ | None
    root_linear_velocity_w: Vector3 | None
    root_angular_velocity_w: Vector3 | None
    com_position_w: Vector3 | None
    com_velocity_w: Vector3 | None
    joint_position_rad: Mapping[str, float]
    joint_velocity_rad_s: Mapping[str, float]
    servo_target_rad: Mapping[str, float]
    wheel_target_rad_s: Mapping[str, float]
    measured_wheel_velocity_rad_s: Mapping[str, float]
    integrated_wheel_rotation_rad: Mapping[str, float]
    integrated_wheel_travel_m: Mapping[str, float]
    wheel_center_w: Mapping[str, Vector3]
    filtered_ground_force_n: Mapping[str, float]
    filtered_obstacle_force_n: Mapping[str, float]
    wheel_contact_point_w: Mapping[str, Vector3]
    wheel_contact_class: Mapping[str, ContactClass]
    support_legs: tuple[str, ...]
    light_support_legs: tuple[str, ...]
    primary_diagonal: PrimaryDiagonal | None
    diagonal_load_share: Mapping[str, float]
    support_polygon_margin_m: float | None
    support_polygon_valid: bool | None
    two_leg_corridor: TwoLegCorridorObservation
    contact_drift_m: Mapping[str, float]
    wheel_clearance_over_top_m: Mapping[str, float]
    wheel_front_face_clearance_m: Mapping[str, float]
    nonwheel_obstacle_contact: bool | None
    nonwheel_contact_evidence_valid: bool
    joint_limit_margin_rad: Mapping[str, float]
    actuator_target_error_rad: Mapping[str, float]
    com_target: COMTargetDirection | None
    issues: tuple[ObservationIssue, ...] = ()

    @property
    def control_ready(self) -> bool:
        return not any(issue.critical for issue in self.issues)

    @property
    def missing_critical_fields(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                issue.field
                for issue in self.issues
                if issue.critical and issue.code in {"MISSING", "EMPTY", "LEG_KEYS"}
            )
        )

    @property
    def fail_closed_reason(self) -> str:
        return "" if self.control_ready else "; ".join(str(issue) for issue in self.issues)

    def guard_allows(self, predicate: bool) -> bool:
        """Return False whenever telemetry is not control-ready."""

        return bool(self.control_ready and predicate)

    def require_control_ready(self) -> FSM50Observation:
        if not self.control_ready:
            raise CriticalTelemetryUnavailable(self.fail_closed_reason)
        return self

    def target_direction_for(self, target_leg: str) -> COMTargetDirection:
        """Resolve another target leg from this sample's measured geometry."""

        leg = _canonical_leg(target_leg)
        if leg is None:
            raise ValueError(f"target_leg must identify one of {LEGS}, got {target_leg!r}")
        if (
            self.root_orientation_wxyz is None
            or self.com_position_w is None
            or leg not in self.wheel_contact_point_w
        ):
            raise CriticalTelemetryUnavailable(
                f"cannot resolve COM target {leg}: orientation, COM, or contact point unavailable"
            )
        return compute_world_com_target_direction(
            root_orientation_wxyz=self.root_orientation_wxyz,
            com_position_w=self.com_position_w,
            target_contact_w=self.wheel_contact_point_w[leg],
            target_leg=leg,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mutable copy."""

        return {
            "time_s": self.time_s,
            "root_position_w": self.root_position_w,
            "root_orientation_wxyz": self.root_orientation_wxyz,
            "root_linear_velocity_w": self.root_linear_velocity_w,
            "root_angular_velocity_w": self.root_angular_velocity_w,
            "com_position_w": self.com_position_w,
            "com_velocity_w": self.com_velocity_w,
            "joint_position_rad": dict(self.joint_position_rad),
            "joint_velocity_rad_s": dict(self.joint_velocity_rad_s),
            "servo_target_rad": dict(self.servo_target_rad),
            "wheel_target_rad_s": dict(self.wheel_target_rad_s),
            "measured_wheel_velocity_rad_s": dict(self.measured_wheel_velocity_rad_s),
            "integrated_wheel_rotation_rad": dict(self.integrated_wheel_rotation_rad),
            "integrated_wheel_travel_m": dict(self.integrated_wheel_travel_m),
            "wheel_center_w": dict(self.wheel_center_w),
            "filtered_ground_force_n": dict(self.filtered_ground_force_n),
            "filtered_obstacle_force_n": dict(self.filtered_obstacle_force_n),
            "wheel_contact_point_w": dict(self.wheel_contact_point_w),
            "wheel_contact_class": {
                leg: value.value for leg, value in self.wheel_contact_class.items()
            },
            "support_legs": list(self.support_legs),
            "light_support_legs": list(self.light_support_legs),
            "primary_diagonal": (
                None if self.primary_diagonal is None else self.primary_diagonal.value
            ),
            "diagonal_load_share": dict(self.diagonal_load_share),
            "support_polygon_margin_m": self.support_polygon_margin_m,
            "support_polygon_valid": self.support_polygon_valid,
            "two_leg_corridor": {
                "applicable": self.two_leg_corridor.applicable,
                "valid": self.two_leg_corridor.valid,
                "perpendicular_distance_m": self.two_leg_corridor.perpendicular_distance_m,
                "segment_fraction": self.two_leg_corridor.segment_fraction,
                "within_longitudinal_bounds": (
                    self.two_leg_corridor.within_longitudinal_bounds
                ),
                "within_corridor_width": self.two_leg_corridor.within_corridor_width,
            },
            "contact_drift_m": dict(self.contact_drift_m),
            "wheel_clearance_over_top_m": dict(self.wheel_clearance_over_top_m),
            "wheel_front_face_clearance_m": dict(self.wheel_front_face_clearance_m),
            "nonwheel_obstacle_contact": self.nonwheel_obstacle_contact,
            "nonwheel_contact_evidence_valid": self.nonwheel_contact_evidence_valid,
            "joint_limit_margin_rad": dict(self.joint_limit_margin_rad),
            "actuator_target_error_rad": dict(self.actuator_target_error_rad),
            "com_target": (
                None
                if self.com_target is None
                else {
                    "target_leg": self.com_target.target_leg,
                    "target_contact_w": self.com_target.target_contact_w,
                    "direction_w": self.com_target.direction_w,
                    "direction_body": self.com_target.direction_body,
                    "distance_m": self.com_target.distance_m,
                    "horizontal_only": self.com_target.horizontal_only,
                    "source": self.com_target.source,
                }
            ),
            "control_ready": self.control_ready,
            "issues": [
                {
                    "field": issue.field,
                    "code": issue.code,
                    "message": issue.message,
                    "critical": issue.critical,
                }
                for issue in self.issues
            ],
        }

    @classmethod
    def from_mapping(
        cls, source: Mapping[str, Any], *, strict: bool = False
    ) -> FSM50Observation:
        """Validate a telemetry mapping and build an immutable observation."""

        if not isinstance(source, Mapping):
            raise TypeError(
                f"observation source must be a mapping, got {type(source).__name__}"
            )
        builder = _Builder(source)

        time_s = builder.scalar("time_s", "simulation_time_s", nonnegative=True)
        root_position = builder.vector(
            "root_position_w",
            3,
            "root_position_w_m",
            components=("base_x_m", "base_y_m", "base_z_m"),
        )
        root_orientation = builder.quaternion(
            "root_orientation_wxyz",
            "root_quaternion_wxyz",
            components=("base_qw", "base_qx", "base_qy", "base_qz"),
        )
        root_linear_velocity = builder.vector(
            "root_linear_velocity_w",
            3,
            "root_linear_velocity_w_m_s",
            components=("base_vx_m_s", "base_vy_m_s", "base_vz_m_s"),
        )
        root_angular_velocity = builder.vector(
            "root_angular_velocity_w",
            3,
            "root_angular_velocity_w_rad_s",
            components=("base_wx_rad_s", "base_wy_rad_s", "base_wz_rad_s"),
        )
        com_position = builder.vector(
            "com_position_w",
            3,
            "com_position_w_m",
            components=("com_x_m", "com_y_m", "com_z_m"),
        )
        com_velocity = builder.vector(
            "com_velocity_w",
            3,
            "com_velocity_w_m_s",
            components=("com_vx_m_s", "com_vy_m_s", "com_vz_m_s"),
        )

        joint_position = builder.scalar_mapping(
            "joint_position_rad", "measured_joint_position_rad"
        )
        joint_velocity = builder.scalar_mapping(
            "joint_velocity_rad_s", "measured_joint_velocity_rad_s"
        )
        servo_target = _servo_targets(builder)
        wheel_target = builder.leg_scalar_mapping(
            "wheel_target_rad_s", "wheel_command_velocity_rad_s"
        )
        measured_wheel_velocity = builder.leg_scalar_mapping(
            "measured_wheel_velocity_rad_s",
            "wheel_canonical_forward_velocity_rad_s",
        )
        integrated_rotation = builder.leg_scalar_mapping(
            "integrated_wheel_rotation_rad", "wheel_integrated_rotation_rad"
        )
        integrated_travel = builder.leg_scalar_mapping(
            "integrated_wheel_travel_m", "wheel_integrated_travel_m"
        )
        wheel_centers = builder.leg_vector_mapping(
            "wheel_center_w", 3, "wheel_centers_w"
        )
        ground_force, obstacle_force = _surface_force_maps(builder)
        contact_points = builder.leg_vector_mapping(
            "wheel_contact_point_w", 3, "wheel_contact_points_w"
        )
        contact_classes = builder.contact_class_mapping(
            "wheel_contact_class", "wheel_contact_classes"
        )
        support_legs = builder.legs("support_legs")
        light_support_legs = builder.legs("light_support_legs")
        overlap = tuple(leg for leg in support_legs if leg in light_support_legs)
        if overlap:
            builder.issue(
                "light_support_legs",
                "OVERLAP",
                f"legs cannot be both full and light support: {overlap}",
            )

        primary_raw = builder.lookup("primary_diagonal")
        primary_diagonal: PrimaryDiagonal | None = None
        if primary_raw is _MISSING or primary_raw is None:
            builder.issue(
                "primary_diagonal", "MISSING", "measured primary diagonal unavailable"
            )
        else:
            try:
                primary_diagonal = PrimaryDiagonal(_enum_text(primary_raw))
            except ValueError:
                builder.issue(
                    "primary_diagonal", "ENUM", f"unknown diagonal {primary_raw!r}"
                )
        diagonal_share = _diagonal_load_share(builder)

        polygon_margin = builder.scalar(
            "support_polygon_margin_m", "wheel_support_polygon_margin_m"
        )
        polygon_valid = builder.boolean(
            "support_polygon_valid", "wheel_support_polygon_valid"
        )

        applicable_raw = builder.lookup("two_leg_corridor_applicable")
        if applicable_raw is _MISSING:
            corridor_applicable = len(support_legs) == 2
        else:
            parsed_applicable = builder.boolean(
                "two_leg_corridor_applicable", raw=applicable_raw
            )
            corridor_applicable = bool(parsed_applicable)
        corridor_valid_value = builder.boolean("two_leg_corridor_valid")
        corridor_distance = builder.scalar(
            "two_leg_corridor_distance_m",
            raw=builder.lookup(
                "two_leg_corridor_distance_m",
                "two_leg_corridor_perpendicular_distance_m",
            ),
            required=corridor_applicable,
            nonnegative=True,
        )
        corridor_fraction = builder.scalar(
            "two_leg_corridor_fraction",
            raw=builder.lookup("two_leg_corridor_fraction"),
            required=corridor_applicable,
        )
        corridor_longitudinal = builder.boolean(
            "two_leg_corridor_within_longitudinal_bounds", required=False
        )
        corridor_width = builder.boolean(
            "two_leg_corridor_within_width", required=False
        )
        corridor = TwoLegCorridorObservation(
            applicable=corridor_applicable,
            valid=bool(corridor_valid_value),
            perpendicular_distance_m=corridor_distance,
            segment_fraction=corridor_fraction,
            within_longitudinal_bounds=corridor_longitudinal,
            within_corridor_width=corridor_width,
        )

        contact_drift = builder.leg_scalar_mapping(
            "contact_drift_m", "wheel_contact_drift_m", nonnegative=True
        )
        clearance_top = builder.leg_scalar_mapping("wheel_clearance_over_top_m")
        clearance_face = builder.leg_scalar_mapping(
            "wheel_front_face_clearance_m"
        )

        evidence_valid_value = builder.boolean(
            "nonwheel_contact_evidence_valid", "collision_evidence_valid"
        )
        evidence_valid = bool(evidence_valid_value)
        nonwheel_raw = builder.lookup(
            "nonwheel_obstacle_contact", "dangerous_collision"
        )
        if nonwheel_raw is _MISSING:
            contacts_raw = builder.lookup("nonwheel_obstacle_contacts")
            if isinstance(contacts_raw, Sequence) and not isinstance(
                contacts_raw, (str, bytes)
            ):
                nonwheel_raw = bool(contacts_raw)
        nonwheel_contact = builder.boolean(
            "nonwheel_obstacle_contact",
            raw=nonwheel_raw,
            required=evidence_valid,
        )
        if evidence_valid_value is False:
            builder.issue(
                "nonwheel_contact_evidence_valid",
                "EVIDENCE_UNAVAILABLE",
                "non-wheel collision telemetry is invalid; guards must fail closed",
            )

        joint_limit_margin = builder.scalar_mapping(
            "joint_limit_margin_rad", nonnegative=True
        )
        actuator_target_error = builder.scalar_mapping(
            "actuator_target_error_rad", nonnegative=True
        )
        unknown_margin_joints = tuple(
            name for name in joint_limit_margin if name not in joint_position
        )
        if unknown_margin_joints:
            builder.issue(
                "joint_limit_margin_rad",
                "JOINT_KEYS",
                f"margins reference unmeasured joints: {unknown_margin_joints}",
            )
        unknown_error_joints = tuple(
            name for name in actuator_target_error if name not in joint_position
        )
        if unknown_error_joints:
            builder.issue(
                "actuator_target_error_rad",
                "JOINT_KEYS",
                f"errors reference unmeasured joints: {unknown_error_joints}",
            )

        com_target: COMTargetDirection | None = None
        target_leg_raw = builder.lookup("target_com_leg")
        if target_leg_raw is not _MISSING and target_leg_raw not in (None, ""):
            target_leg = _canonical_leg(target_leg_raw)
            if target_leg is None:
                builder.issue(
                    "target_com_leg",
                    "LEG_KEY",
                    f"expected one of {LEGS}, got {target_leg_raw!r}",
                )
            elif (
                root_orientation is None
                or com_position is None
                or target_leg not in contact_points
            ):
                builder.issue(
                    "com_target",
                    "MISSING",
                    f"cannot resolve target {target_leg} without orientation, COM, and contact",
                )
            else:
                try:
                    com_target = compute_world_com_target_direction(
                        root_orientation_wxyz=root_orientation,
                        com_position_w=com_position,
                        target_contact_w=contact_points[target_leg],
                        target_leg=target_leg,
                    )
                except ValueError as exc:
                    builder.issue("com_target", "GEOMETRY", str(exc))

        observation = cls(
            time_s=time_s,
            root_position_w=root_position,  # type: ignore[arg-type]
            root_orientation_wxyz=root_orientation,
            root_linear_velocity_w=root_linear_velocity,  # type: ignore[arg-type]
            root_angular_velocity_w=root_angular_velocity,  # type: ignore[arg-type]
            com_position_w=com_position,  # type: ignore[arg-type]
            com_velocity_w=com_velocity,  # type: ignore[arg-type]
            joint_position_rad=_freeze(joint_position),
            joint_velocity_rad_s=_freeze(joint_velocity),
            servo_target_rad=_freeze(servo_target),
            wheel_target_rad_s=_freeze(wheel_target),
            measured_wheel_velocity_rad_s=_freeze(measured_wheel_velocity),
            integrated_wheel_rotation_rad=_freeze(integrated_rotation),
            integrated_wheel_travel_m=_freeze(integrated_travel),
            wheel_center_w=_freeze(wheel_centers),  # type: ignore[arg-type]
            filtered_ground_force_n=_freeze(ground_force),
            filtered_obstacle_force_n=_freeze(obstacle_force),
            wheel_contact_point_w=_freeze(contact_points),  # type: ignore[arg-type]
            wheel_contact_class=_freeze(contact_classes),  # type: ignore[arg-type]
            support_legs=support_legs,
            light_support_legs=light_support_legs,
            primary_diagonal=primary_diagonal,
            diagonal_load_share=_freeze(diagonal_share),
            support_polygon_margin_m=polygon_margin,
            support_polygon_valid=polygon_valid,
            two_leg_corridor=corridor,
            contact_drift_m=_freeze(contact_drift),
            wheel_clearance_over_top_m=_freeze(clearance_top),
            wheel_front_face_clearance_m=_freeze(clearance_face),
            nonwheel_obstacle_contact=nonwheel_contact,
            nonwheel_contact_evidence_valid=evidence_valid,
            joint_limit_margin_rad=_freeze(joint_limit_margin),
            actuator_target_error_rad=_freeze(actuator_target_error),
            com_target=com_target,
            issues=tuple(builder.issues),
        )
        if strict and observation.issues:
            raise ObservationValidationError(observation.issues)
        return observation

    @classmethod
    def fake(cls, **overrides: Any) -> FSM50Observation:
        """Build a complete deterministic sample for controller unit tests."""

        servo_joints = tuple(
            f"{prefix}_{joint}"
            for prefix in ("front_left", "front_right", "rear_left", "rear_right")
            for joint in ("hip", "knee")
        )
        wheel_joints = (
            "front_left_ankle",
            "front_right_ankle",
            "rear_left_ankle",
            "rear_right_ankle",
        )
        joint_names = servo_joints + wheel_joints
        centers = {
            "FL": (0.30, 0.20, 0.05),
            "FR": (0.30, -0.20, 0.05),
            "RL": (-0.30, 0.20, 0.05),
            "RR": (-0.30, -0.20, 0.05),
        }
        mapping: dict[str, Any] = {
            "time_s": 0.0,
            "root_position_w": (0.0, 0.0, 0.10),
            "root_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "root_linear_velocity_w": (0.0, 0.0, 0.0),
            "root_angular_velocity_w": (0.0, 0.0, 0.0),
            "com_position_w": (0.0, 0.0, 0.08),
            "com_velocity_w": (0.0, 0.0, 0.0),
            "joint_position_rad": {name: 0.0 for name in joint_names},
            "joint_velocity_rad_s": {name: 0.0 for name in joint_names},
            "servo_target_rad": {name: 0.0 for name in servo_joints},
            "wheel_target_rad_s": {leg: 0.0 for leg in LEGS},
            "measured_wheel_velocity_rad_s": {leg: 0.0 for leg in LEGS},
            "integrated_wheel_rotation_rad": {leg: 0.0 for leg in LEGS},
            "integrated_wheel_travel_m": {leg: 0.0 for leg in LEGS},
            "wheel_center_w": centers,
            "filtered_ground_force_n": {leg: 5.0 for leg in LEGS},
            "filtered_obstacle_force_n": {leg: 0.0 for leg in LEGS},
            "wheel_contact_point_w": {
                leg: (center[0], center[1], 0.0)
                for leg, center in centers.items()
            },
            "wheel_contact_class": {leg: "GROUND" for leg in LEGS},
            "support_legs": list(LEGS),
            "light_support_legs": [],
            "primary_diagonal": "BALANCED",
            "diagonal_load_share": {"FL_RR": 0.5, "FR_RL": 0.5},
            "support_polygon_margin_m": 0.1,
            "support_polygon_valid": True,
            "two_leg_corridor_applicable": False,
            "two_leg_corridor_valid": False,
            "contact_drift_m": {leg: 0.0 for leg in LEGS},
            "wheel_clearance_over_top_m": {leg: -0.05 for leg in LEGS},
            "wheel_front_face_clearance_m": {leg: -0.2 for leg in LEGS},
            "nonwheel_contact_evidence_valid": True,
            "nonwheel_obstacle_contact": False,
            "joint_limit_margin_rad": {name: 1.0 for name in servo_joints},
            "actuator_target_error_rad": {name: 0.0 for name in servo_joints},
        }
        mapping.update(overrides)
        return cls.from_mapping(mapping, strict=True)


Observation = FSM50Observation

__all__ = [
    "COMTargetDirection",
    "ContactClass",
    "CriticalTelemetryUnavailable",
    "DIAGONALS",
    "FSM50Observation",
    "LEGS",
    "Observation",
    "ObservationIssue",
    "ObservationValidationError",
    "PrimaryDiagonal",
    "TwoLegCorridorObservation",
    "compute_world_com_target_direction",
]
