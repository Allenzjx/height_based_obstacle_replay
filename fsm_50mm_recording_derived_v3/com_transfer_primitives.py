"""Isaac-independent COM-transfer primitives and physical transition guards.

The classes in this module deliberately consume measured values rather than
simulator objects.  That keeps the safety and transition logic deterministic,
unit-testable, and usable by either replay or an eventual FSM controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class Leg(str, Enum):
    FL = "FL"
    FR = "FR"
    RL = "RL"
    RR = "RR"


LEGS: tuple[Leg, ...] = tuple(Leg)


class COMTransferMethod(str, Enum):
    NONE = "NONE"
    IMPULSE_REACTION_TRANSFER = "IMPULSE_REACTION_TRANSFER"
    ANCHORED_SUPPORT_ANGLE_TRANSFER = "ANCHORED_SUPPORT_ANGLE_TRANSFER"
    WHEEL_IMPULSE_ASSIST = "WHEEL_IMPULSE_ASSIST"
    HYBRID = "HYBRID"


class ActiveLegMode(str, Enum):
    UNLOAD = "UNLOAD"
    LIFT = "LIFT"
    HOLD_LIFT = "HOLD_LIFT"
    SWING_CLEAR = "SWING_CLEAR"
    PLACE = "PLACE"
    CONFIRM = "CONFIRM"
    OTHER = "OTHER"


class ImpulseStage(str, Enum):
    PRELOAD = "PRELOAD"
    PUSH = "PUSH"
    RELEASE = "RELEASE"
    COAST = "COAST"
    SETTLE = "SETTLE"
    VERIFY = "VERIFY"


@dataclass(frozen=True)
class GuardDecision:
    satisfied: bool
    abort: bool = False
    reason: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    unmet: tuple[str, ...] = ()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _vec2(values: Sequence[float], *, label: str) -> tuple[float, float]:
    if len(values) < 2 or not _finite(values[0]) or not _finite(values[1]):
        raise ValueError(f"{label} must contain two finite values")
    return float(values[0]), float(values[1])


def _unit(values: Sequence[float], *, label: str = "direction") -> tuple[float, float]:
    x, y = _vec2(values, label=label)
    norm = math.hypot(x, y)
    if norm <= 1.0e-12:
        raise ValueError(f"{label} must be non-zero")
    return x / norm, y / norm


def _projection(values: Sequence[float], direction: tuple[float, float]) -> float:
    x, y = _vec2(values, label="vector")
    return x * direction[0] + y * direction[1]


def _lateral(values: Sequence[float], direction: tuple[float, float]) -> float:
    x, y = _vec2(values, label="vector")
    return -x * direction[1] + y * direction[0]


def _leg(value: Leg | str) -> Leg:
    return value if isinstance(value, Leg) else Leg(str(value).upper())


# ---------------------------------------------------------------------------
# Per-leg IK acceptance


@dataclass(frozen=True)
class LegIKCandidate:
    leg: Leg
    requested_targets_deg: Mapping[str, float]
    reference_targets_deg: Mapping[str, float]
    joint_limits_deg: Mapping[str, tuple[float, float]]
    solver_valid: bool = True
    solver_reason: str = ""


@dataclass(frozen=True)
class LegIKDecision:
    leg: Leg
    accepted: bool
    targets_deg: Mapping[str, float]
    preserved_reference_joints: tuple[str, ...] = ()
    boundary_clamped_joints: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class PerLegIKBatchDecision:
    decisions: Mapping[Leg, LegIKDecision]

    @property
    def accepted_targets_deg(self) -> dict[Leg, dict[str, float]]:
        return {
            leg: dict(decision.targets_deg)
            for leg, decision in self.decisions.items()
            if decision.accepted
        }

    @property
    def rejected_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg, decision in self.decisions.items() if not decision.accepted)


def accept_per_leg_ik(
    candidates: Mapping[Leg | str, LegIKCandidate],
    *,
    unchanged_tolerance_deg: float = 1.0e-6,
    boundary_roundoff_tolerance_deg: float = 1.0e-6,
) -> PerLegIKBatchDecision:
    """Validate each IK leg independently.

    An unrelated invalid leg never rejects a valid leg.  If a solver returns a
    numerically perturbed value for an unchanged joint, the exact in-limit
    reference is retained.  A genuinely out-of-limit request remains rejected.
    """

    decisions: dict[Leg, LegIKDecision] = {}
    for raw_leg, candidate in candidates.items():
        leg = _leg(raw_leg)
        if _leg(candidate.leg) != leg:
            decisions[leg] = LegIKDecision(leg, False, {}, reason="candidate leg mismatch")
            continue
        if not candidate.solver_valid:
            decisions[leg] = LegIKDecision(
                leg,
                False,
                {},
                reason=candidate.solver_reason or "IK solver rejected this leg",
            )
            continue

        accepted: dict[str, float] = {}
        preserved: list[str] = []
        clamped: list[str] = []
        error = ""
        for joint, requested_raw in candidate.requested_targets_deg.items():
            if joint not in candidate.reference_targets_deg:
                error = f"missing reference for {joint}"
                break
            if joint not in candidate.joint_limits_deg:
                error = f"missing joint limit for {joint}"
                break
            requested = float(requested_raw)
            reference = float(candidate.reference_targets_deg[joint])
            lower, upper = (float(v) for v in candidate.joint_limits_deg[joint])
            if not all(_finite(v) for v in (requested, reference, lower, upper)) or lower > upper:
                error = f"invalid numeric IK data for {joint}"
                break
            if (
                abs(requested - reference) <= float(unchanged_tolerance_deg)
                and lower <= reference <= upper
            ):
                accepted[joint] = reference
                preserved.append(joint)
                continue
            if requested < lower:
                if lower - requested <= float(boundary_roundoff_tolerance_deg):
                    accepted[joint] = lower
                    clamped.append(joint)
                else:
                    error = f"{joint} target {requested} below limit {lower}"
                    break
            elif requested > upper:
                if requested - upper <= float(boundary_roundoff_tolerance_deg):
                    accepted[joint] = upper
                    clamped.append(joint)
                else:
                    error = f"{joint} target {requested} above limit {upper}"
                    break
            else:
                accepted[joint] = requested

        if error:
            decisions[leg] = LegIKDecision(leg, False, {}, reason=error)
        else:
            decisions[leg] = LegIKDecision(
                leg,
                True,
                accepted,
                tuple(preserved),
                tuple(clamped),
                "accepted independently",
            )
    return PerLegIKBatchDecision(decisions)


# ---------------------------------------------------------------------------
# Active-leg isolation


@dataclass(frozen=True)
class CorrectionIsolationResult:
    targets_deg: Mapping[Leg, Mapping[str, float]]
    applied_corrections_deg: Mapping[Leg, Mapping[str, float]]
    scales: Mapping[Leg, float]
    ignored: Mapping[Leg, tuple[str, ...]]


def isolate_active_leg_corrections(
    base_targets_deg: Mapping[Leg | str, Mapping[str, float]],
    whole_body_corrections_deg: Mapping[Leg | str, Mapping[str, float]],
    *,
    active_leg: Leg | str | None,
    support_legs: Iterable[Leg | str],
    mode: ActiveLegMode | str,
    place_confirm_blend: float = 0.0,
) -> CorrectionIsolationResult:
    """Apply COM corrections without overwriting an active swing-leg target."""

    active = None if active_leg in (None, "", "NONE") else _leg(active_leg)
    support = {_leg(leg) for leg in support_legs}
    mode = mode if isinstance(mode, ActiveLegMode) else ActiveLegMode(str(mode).upper())
    blend = max(0.0, min(1.0, float(place_confirm_blend)))
    protected = {
        ActiveLegMode.UNLOAD,
        ActiveLegMode.LIFT,
        ActiveLegMode.HOLD_LIFT,
        ActiveLegMode.SWING_CLEAR,
    }

    normalized_base = {_leg(leg): dict(values) for leg, values in base_targets_deg.items()}
    normalized_correction = {
        _leg(leg): dict(values) for leg, values in whole_body_corrections_deg.items()
    }
    output: dict[Leg, dict[str, float]] = {}
    applied: dict[Leg, dict[str, float]] = {}
    scales: dict[Leg, float] = {}
    ignored: dict[Leg, tuple[str, ...]] = {}
    for leg in LEGS:
        base = {joint: float(value) for joint, value in normalized_base.get(leg, {}).items()}
        correction = normalized_correction.get(leg, {})
        if mode in protected:
            scale = 1.0 if leg in support and leg != active else 0.0
        elif mode in {ActiveLegMode.PLACE, ActiveLegMode.CONFIRM} and leg == active:
            scale = blend
        else:
            scale = 1.0
        scales[leg] = scale
        applied_leg: dict[str, float] = {}
        ignored_leg: list[str] = []
        result = dict(base)
        for joint, correction_raw in correction.items():
            if joint not in base:
                ignored_leg.append(joint)
                continue
            delta = float(correction_raw) * scale
            result[joint] = base[joint] + delta
            applied_leg[joint] = delta
        output[leg] = result
        applied[leg] = applied_leg
        ignored[leg] = tuple(ignored_leg)
    return CorrectionIsolationResult(output, applied, scales, ignored)


# ---------------------------------------------------------------------------
# Direction, anchors, and transfer guards


@dataclass(frozen=True)
class COMDirectionalConfig:
    target_direction_xy: tuple[float, float]
    minimum_displacement_m: float
    minimum_projected_velocity_m_s: float | None = None
    maximum_abs_projected_velocity_m_s: float | None = None
    maximum_lateral_displacement_m: float | None = None
    required_consecutive_samples: int = 1


@dataclass(frozen=True)
class COMDirectionalResult:
    satisfied: bool
    projected_displacement_m: float
    projected_velocity_m_s: float
    lateral_displacement_m: float
    consecutive_samples: int
    reason: str = ""


@dataclass
class COMDirectionalDetector:
    config: COMDirectionalConfig
    start_com_xy: tuple[float, float] | None = None
    consecutive_samples: int = 0

    def __post_init__(self) -> None:
        _unit(self.config.target_direction_xy)
        if not _finite(self.config.minimum_displacement_m) or self.config.minimum_displacement_m < 0.0:
            raise ValueError("minimum_displacement_m must be non-negative")
        if self.config.required_consecutive_samples < 1:
            raise ValueError("required_consecutive_samples must be positive")
        for name, value in (
            ("minimum_projected_velocity_m_s", self.config.minimum_projected_velocity_m_s),
            ("maximum_abs_projected_velocity_m_s", self.config.maximum_abs_projected_velocity_m_s),
            ("maximum_lateral_displacement_m", self.config.maximum_lateral_displacement_m),
        ):
            if value is not None and (not _finite(value) or (name.startswith("maximum") and value < 0.0)):
                raise ValueError(f"{name} is invalid")

    def reset(self, start_com_xy: Sequence[float] | None = None) -> None:
        self.start_com_xy = (
            None if start_com_xy is None else _vec2(start_com_xy, label="start_com_xy")
        )
        self.consecutive_samples = 0

    def update(
        self, com_xy: Sequence[float], com_velocity_xy: Sequence[float]
    ) -> COMDirectionalResult:
        com = _vec2(com_xy, label="com_xy")
        velocity = _vec2(com_velocity_xy, label="com_velocity_xy")
        direction = _unit(self.config.target_direction_xy)
        if self.start_com_xy is None:
            self.start_com_xy = com
            return COMDirectionalResult(False, 0.0, _projection(velocity, direction), 0.0, 0, "baseline captured")
        delta = (com[0] - self.start_com_xy[0], com[1] - self.start_com_xy[1])
        projected = _projection(delta, direction)
        speed = _projection(velocity, direction)
        lateral = abs(_lateral(delta, direction))
        unmet: list[str] = []
        if projected < self.config.minimum_displacement_m:
            unmet.append("directional displacement below threshold")
        if (
            self.config.minimum_projected_velocity_m_s is not None
            and speed < self.config.minimum_projected_velocity_m_s
        ):
            unmet.append("projected COM velocity below threshold")
        if (
            self.config.maximum_abs_projected_velocity_m_s is not None
            and abs(speed) > self.config.maximum_abs_projected_velocity_m_s
        ):
            unmet.append("projected COM velocity has not settled")
        if (
            self.config.maximum_lateral_displacement_m is not None
            and lateral > self.config.maximum_lateral_displacement_m
        ):
            unmet.append("lateral COM drift exceeded threshold")
        self.consecutive_samples = self.consecutive_samples + 1 if not unmet else 0
        satisfied = self.consecutive_samples >= self.config.required_consecutive_samples
        return COMDirectionalResult(
            satisfied,
            projected,
            speed,
            lateral,
            self.consecutive_samples,
            "" if satisfied else "; ".join(unmet) or "waiting for consecutive samples",
        )


@dataclass(frozen=True)
class ContactDriftResult:
    valid: bool
    drift_m: Mapping[Leg, float]
    maximum_drift_m: float
    missing_legs: tuple[Leg, ...] = ()
    reason: str = ""


@dataclass
class ContactAnchorTracker:
    anchors_xy: dict[Leg, tuple[float, float]] = field(default_factory=dict)

    def capture(
        self,
        contact_points_xy: Mapping[Leg | str, Sequence[float]],
        legs: Iterable[Leg | str],
    ) -> None:
        normalized = {_leg(leg): values for leg, values in contact_points_xy.items()}
        self.anchors_xy = {
            _leg(leg): _vec2(normalized[_leg(leg)], label=f"{_leg(leg).value} contact")
            for leg in legs
        }

    def evaluate(
        self,
        contact_points_xy: Mapping[Leg | str, Sequence[float]],
        *,
        maximum_drift_m: float,
    ) -> ContactDriftResult:
        normalized = {_leg(leg): value for leg, value in contact_points_xy.items()}
        drift: dict[Leg, float] = {}
        missing: list[Leg] = []
        for leg, anchor in self.anchors_xy.items():
            if leg not in normalized:
                missing.append(leg)
                continue
            try:
                current = _vec2(normalized[leg], label=f"{leg.value} contact")
            except ValueError:
                missing.append(leg)
                continue
            drift[leg] = math.hypot(current[0] - anchor[0], current[1] - anchor[1])
        maximum = max(drift.values(), default=float("nan"))
        valid = not missing and all(value <= maximum_drift_m for value in drift.values())
        reason = ""
        if missing:
            reason = "missing anchored contacts"
        elif not valid:
            reason = "support contact drift exceeded threshold"
        return ContactDriftResult(valid, drift, maximum, tuple(missing), reason)


@dataclass(frozen=True)
class TransferObservation:
    time_s: float
    com_xy: tuple[float, float]
    com_velocity_xy: tuple[float, float]
    contact_points_xy: Mapping[Leg, tuple[float, float]]
    contact_active: Mapping[Leg, bool]
    contact_load_n: Mapping[Leg, float]
    roll_deg: float
    pitch_deg: float
    angular_speed_deg_s: float


@dataclass(frozen=True)
class AnchoredTransferConfig:
    support_legs: tuple[Leg, ...]
    target_direction_xy: tuple[float, float]
    minimum_com_displacement_m: float
    maximum_contact_drift_m: float
    minimum_support_load_n: float
    maximum_roll_deg: float
    maximum_pitch_deg: float
    maximum_angular_speed_deg_s: float
    required_consecutive_samples: int = 2


@dataclass
class AnchoredSupportAngleGuard:
    config: AnchoredTransferConfig
    anchors: ContactAnchorTracker = field(default_factory=ContactAnchorTracker)
    detector: COMDirectionalDetector = field(init=False)
    initialized: bool = False

    def __post_init__(self) -> None:
        self.detector = COMDirectionalDetector(
            COMDirectionalConfig(
                self.config.target_direction_xy,
                self.config.minimum_com_displacement_m,
                required_consecutive_samples=self.config.required_consecutive_samples,
            )
        )

    def reset(self) -> None:
        self.anchors = ContactAnchorTracker()
        self.detector.reset()
        self.initialized = False

    def update(self, observation: TransferObservation) -> GuardDecision:
        if not all(
            _finite(value)
            for value in (
                *observation.com_xy,
                *observation.com_velocity_xy,
                observation.roll_deg,
                observation.pitch_deg,
                observation.angular_speed_deg_s,
            )
        ):
            return GuardDecision(False, True, "non-finite anchored-transfer observation")
        if not self.initialized:
            missing = tuple(
                leg for leg in self.config.support_legs if leg not in observation.contact_points_xy
            )
            if missing:
                return GuardDecision(False, True, "cannot capture missing support anchors", unmet=tuple(leg.value for leg in missing))
            self.anchors.capture(observation.contact_points_xy, self.config.support_legs)
            self.detector.reset(observation.com_xy)
            self.initialized = True
            return GuardDecision(False, reason="support anchors and COM baseline captured")

        unmet: list[str] = []
        if any(not observation.contact_active.get(leg, False) for leg in self.config.support_legs):
            return GuardDecision(False, True, "anchored support contact lost")
        if any(
            not _finite(observation.contact_load_n.get(leg))
            or float(observation.contact_load_n[leg]) < self.config.minimum_support_load_n
            for leg in self.config.support_legs
        ):
            unmet.append("support load below threshold")
        drift = self.anchors.evaluate(
            observation.contact_points_xy,
            maximum_drift_m=self.config.maximum_contact_drift_m,
        )
        if not drift.valid:
            return GuardDecision(
                False,
                True,
                drift.reason,
                metrics={"contact_drift_m": dict(drift.drift_m), "maximum_contact_drift_m": drift.maximum_drift_m},
            )
        if abs(observation.roll_deg) > self.config.maximum_roll_deg:
            return GuardDecision(False, True, "roll safety limit exceeded")
        if abs(observation.pitch_deg) > self.config.maximum_pitch_deg:
            return GuardDecision(False, True, "pitch safety limit exceeded")
        if abs(observation.angular_speed_deg_s) > self.config.maximum_angular_speed_deg_s:
            unmet.append("body angular speed has not settled")
        direction = self.detector.update(observation.com_xy, observation.com_velocity_xy)
        if not direction.satisfied:
            unmet.append(direction.reason)
        metrics = {
            "projected_com_displacement_m": direction.projected_displacement_m,
            "projected_com_velocity_m_s": direction.projected_velocity_m_s,
            "contact_drift_m": dict(drift.drift_m),
            "maximum_contact_drift_m": drift.maximum_drift_m,
        }
        return GuardDecision(not unmet, reason="" if not unmet else "; ".join(filter(None, unmet)), metrics=metrics, unmet=tuple(filter(None, unmet)))


@dataclass(frozen=True)
class ImpulseTransferConfig:
    support_legs: tuple[Leg, ...]
    impulse_leg: Leg
    target_leg: Leg
    swing_leg: Leg
    target_direction_xy: tuple[float, float]
    minimum_support_load_n: float
    minimum_preload_force_n: float
    minimum_push_velocity_m_s: float
    minimum_push_displacement_m: float
    minimum_transfer_displacement_m: float
    maximum_release_load_ratio: float
    maximum_settle_velocity_m_s: float
    maximum_settle_angular_speed_deg_s: float
    minimum_target_load_increase_n: float
    maximum_swing_load_share: float
    maximum_roll_deg: float
    maximum_pitch_deg: float
    required_consecutive_samples: int = 2


@dataclass
class ImpulseReactionGuard:
    config: ImpulseTransferConfig
    baseline_com_xy: tuple[float, float] | None = None
    baseline_load_n: dict[Leg, float] = field(default_factory=dict)
    current_stage: ImpulseStage | None = None
    consecutive_samples: int = 0

    def reset(self) -> None:
        self.baseline_com_xy = None
        self.baseline_load_n.clear()
        self.current_stage = None
        self.consecutive_samples = 0

    def _baseline(self, observation: TransferObservation) -> None:
        if self.baseline_com_xy is None:
            self.baseline_com_xy = _vec2(observation.com_xy, label="com_xy")
            self.baseline_load_n = {
                leg: max(0.0, float(observation.contact_load_n.get(leg, 0.0)))
                for leg in LEGS
            }

    def update(self, stage: ImpulseStage | str, observation: TransferObservation) -> GuardDecision:
        stage = stage if isinstance(stage, ImpulseStage) else ImpulseStage(str(stage).upper())
        required_load_legs = set(self.config.support_legs) | {
            self.config.impulse_leg,
            self.config.target_leg,
            self.config.swing_leg,
        }
        if not all(
            _finite(value)
            for value in (
                *observation.com_xy,
                *observation.com_velocity_xy,
                observation.roll_deg,
                observation.pitch_deg,
                observation.angular_speed_deg_s,
            )
        ) or any(not _finite(observation.contact_load_n.get(leg)) for leg in required_load_legs):
            return GuardDecision(False, True, "non-finite impulse-transfer observation")
        self._baseline(observation)
        if stage != self.current_stage:
            self.current_stage = stage
            self.consecutive_samples = 0
        assert self.baseline_com_xy is not None
        direction = _unit(self.config.target_direction_xy)
        delta = (
            observation.com_xy[0] - self.baseline_com_xy[0],
            observation.com_xy[1] - self.baseline_com_xy[1],
        )
        displacement = _projection(delta, direction)
        velocity = _projection(observation.com_velocity_xy, direction)
        loads = {
            leg: max(0.0, float(observation.contact_load_n.get(leg, 0.0)))
            for leg in LEGS
        }
        total_load = sum(loads.values())
        support_valid = all(
            observation.contact_active.get(leg, False)
            and loads[leg] >= self.config.minimum_support_load_n
            for leg in self.config.support_legs
        )
        if not support_valid:
            return GuardDecision(False, True, "required impulse support lost")
        if abs(observation.roll_deg) > self.config.maximum_roll_deg:
            return GuardDecision(False, True, "roll safety limit exceeded")
        if abs(observation.pitch_deg) > self.config.maximum_pitch_deg:
            return GuardDecision(False, True, "pitch safety limit exceeded")

        conditions: list[tuple[bool, str]]
        if stage == ImpulseStage.PRELOAD:
            conditions = [
                (observation.contact_active.get(self.config.impulse_leg, False), "impulse leg has no contact"),
                (loads[self.config.impulse_leg] >= self.config.minimum_preload_force_n, "impulse preload force below threshold"),
            ]
        elif stage == ImpulseStage.PUSH:
            conditions = [
                (velocity >= self.config.minimum_push_velocity_m_s, "COM velocity is not in the target direction"),
                (displacement >= self.config.minimum_push_displacement_m, "push displacement below threshold"),
            ]
        elif stage == ImpulseStage.RELEASE:
            reference = max(self.baseline_load_n.get(self.config.impulse_leg, 0.0), self.config.minimum_preload_force_n)
            conditions = [
                (loads[self.config.impulse_leg] <= reference * self.config.maximum_release_load_ratio, "impulse load has not released"),
            ]
        elif stage == ImpulseStage.COAST:
            conditions = [
                (displacement >= self.config.minimum_transfer_displacement_m, "coast displacement below target"),
                (velocity >= -self.config.maximum_settle_velocity_m_s, "COM is reversing away from target"),
            ]
        elif stage == ImpulseStage.SETTLE:
            conditions = [
                (displacement >= self.config.minimum_transfer_displacement_m, "transfer displacement below target"),
                (abs(velocity) <= self.config.maximum_settle_velocity_m_s, "COM velocity has not settled"),
                (abs(observation.angular_speed_deg_s) <= self.config.maximum_settle_angular_speed_deg_s, "body angular velocity has not settled"),
            ]
        else:
            target_gain = loads[self.config.target_leg] - self.baseline_load_n.get(self.config.target_leg, 0.0)
            swing_share = loads[self.config.swing_leg] / total_load if total_load > 1.0e-12 else 1.0
            conditions = [
                (displacement >= self.config.minimum_transfer_displacement_m, "verified COM displacement below target"),
                (target_gain >= self.config.minimum_target_load_increase_n, "target-leg load did not increase"),
                (swing_share <= self.config.maximum_swing_load_share, "swing leg did not unload"),
            ]
        unmet = tuple(reason for passed, reason in conditions if not passed)
        self.consecutive_samples = self.consecutive_samples + 1 if not unmet else 0
        satisfied = self.consecutive_samples >= self.config.required_consecutive_samples
        metrics = {
            "stage": stage.value,
            "projected_com_displacement_m": displacement,
            "projected_com_velocity_m_s": velocity,
            "loads_n": loads,
            "consecutive_samples": self.consecutive_samples,
        }
        return GuardDecision(
            satisfied,
            reason="" if satisfied else "; ".join(unmet) or "waiting for consecutive samples",
            metrics=metrics,
            unmet=unmet,
        )


# ---------------------------------------------------------------------------
# Stateful transition helpers


@dataclass(frozen=True)
class TargetLatchResult:
    target_xy: tuple[float, float]
    relatched: bool
    relatch_count: int
    frozen: bool
    reason: str


@dataclass
class SingleRelatchTarget:
    geometry_change_threshold_m: float
    maximum_relatches: int = 1
    target_xy: tuple[float, float] | None = None
    support_signature: tuple[str, ...] = ()
    relatch_count: int = 0
    convergence_started: bool = False

    def reset(self) -> None:
        self.target_xy = None
        self.support_signature = ()
        self.relatch_count = 0
        self.convergence_started = False

    def update(
        self,
        candidate_target_xy: Sequence[float],
        support_signature: Iterable[Leg | str],
        *,
        support_geometry_change_m: float,
        convergence_started: bool = False,
        reason: str = "support geometry changed",
    ) -> TargetLatchResult:
        candidate = _vec2(candidate_target_xy, label="candidate_target_xy")
        signature = tuple(sorted(_leg(leg).value for leg in support_signature))
        if not signature:
            if self.target_xy is None:
                raise ValueError("cannot latch a COM target without a support set")
            return TargetLatchResult(self.target_xy, False, self.relatch_count, self.convergence_started, "support set missing; target preserved")
        if self.target_xy is None:
            self.target_xy = candidate
            self.support_signature = signature
            self.convergence_started = bool(convergence_started)
            return TargetLatchResult(candidate, False, 0, self.convergence_started, "initial target latched")
        self.convergence_started = self.convergence_started or bool(convergence_started)
        changed = support_geometry_change_m >= self.geometry_change_threshold_m
        if changed and not self.convergence_started and self.relatch_count < self.maximum_relatches:
            self.target_xy = candidate
            self.support_signature = signature
            self.relatch_count += 1
            return TargetLatchResult(candidate, True, self.relatch_count, False, reason)
        if changed and self.convergence_started:
            outcome = "target frozen after convergence window started"
        elif changed:
            outcome = "single relatch budget exhausted"
        else:
            outcome = "support geometry did not require relatch"
        return TargetLatchResult(self.target_xy, False, self.relatch_count, self.convergence_started, outcome)


@dataclass
class ContinuousOffsetManager:
    current_offsets: dict[str, float] = field(default_factory=dict)
    target_offsets: dict[str, float] = field(default_factory=dict)
    state_id: str = ""

    def initialize(self, offsets: Mapping[str, float]) -> None:
        """Seed offsets from the actually applied controller state."""

        self.current_offsets = {str(joint): float(value) for joint, value in offsets.items()}
        self.target_offsets = dict(self.current_offsets)

    def enter_state(
        self,
        state_id: str,
        desired_offsets: Mapping[str, float] | None = None,
    ) -> Mapping[str, float]:
        """Enter without implicitly clearing offsets from the prior state."""

        self.state_id = str(state_id)
        if desired_offsets is not None:
            for joint, value in desired_offsets.items():
                self.target_offsets[str(joint)] = float(value)
                self.current_offsets.setdefault(str(joint), 0.0)
        return dict(self.current_offsets)

    def request_clear(self, joints: Iterable[str] | None = None) -> None:
        names = tuple(joints) if joints is not None else tuple(self.current_offsets)
        for joint in names:
            self.target_offsets[str(joint)] = 0.0

    def update(self, dt_s: float, *, maximum_rate_deg_s: float) -> Mapping[str, float]:
        if dt_s < 0.0 or maximum_rate_deg_s < 0.0:
            raise ValueError("dt_s and maximum_rate_deg_s must be non-negative")
        maximum_delta = float(dt_s) * float(maximum_rate_deg_s)
        for joint in set(self.current_offsets) | set(self.target_offsets):
            current = float(self.current_offsets.get(joint, 0.0))
            target = float(self.target_offsets.get(joint, current))
            delta = max(-maximum_delta, min(maximum_delta, target - current))
            self.current_offsets[joint] = current + delta
        return dict(self.current_offsets)


@dataclass
class StateScopedWheelRamp:
    ramped_state_ids: frozenset[str]
    maximum_acceleration_rad_s2: float | Mapping[str, float]
    last_output_rad_s: dict[str, float] = field(default_factory=dict)
    active_state_id: str = ""

    def _limit(self, wheel: str) -> float:
        if isinstance(self.maximum_acceleration_rad_s2, Mapping):
            return float(self.maximum_acceleration_rad_s2.get(wheel, 0.0))
        return float(self.maximum_acceleration_rad_s2)

    def update(
        self,
        state_id: str,
        desired_rad_s: Mapping[str, float],
        *,
        dt_s: float,
        entry_velocity_rad_s: Mapping[str, float] | None = None,
    ) -> Mapping[str, float]:
        state_id = str(state_id)
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        desired = {str(wheel): float(value) for wheel, value in desired_rad_s.items()}
        if state_id not in self.ramped_state_ids:
            self.active_state_id = state_id
            self.last_output_rad_s = dict(desired)
            return desired
        if state_id != self.active_state_id:
            starting = entry_velocity_rad_s if entry_velocity_rad_s is not None else self.last_output_rad_s
            self.last_output_rad_s = {
                wheel: float(starting.get(wheel, 0.0)) for wheel in desired
            }
            self.active_state_id = state_id
        output: dict[str, float] = {}
        for wheel, target in desired.items():
            current = float(self.last_output_rad_s.get(wheel, 0.0))
            maximum_delta = max(0.0, self._limit(wheel)) * max(0.0, float(dt_s))
            output[wheel] = current + max(-maximum_delta, min(maximum_delta, target - current))
        self.last_output_rad_s = output
        return dict(output)


# ---------------------------------------------------------------------------
# Lift/place/final verification


@dataclass(frozen=True)
class LiftSwingClearanceResult:
    allowed: bool
    reason: str
    retained_clearance_m: float


def evaluate_lift_to_swing_clearance(
    *,
    lift_peak_clearance_m: float,
    current_clearance_m: float,
    predicted_swing_clearance_m: float,
    minimum_clearance_m: float,
    maximum_clearance_drop_m: float,
) -> LiftSwingClearanceResult:
    values = (
        lift_peak_clearance_m,
        current_clearance_m,
        predicted_swing_clearance_m,
        minimum_clearance_m,
        maximum_clearance_drop_m,
    )
    if not all(_finite(value) for value in values):
        return LiftSwingClearanceResult(False, "non-finite clearance evidence", float("nan"))
    retained = min(float(current_clearance_m), float(predicted_swing_clearance_m))
    if lift_peak_clearance_m < minimum_clearance_m:
        return LiftSwingClearanceResult(False, "lift clearance was never established", retained)
    if retained < minimum_clearance_m:
        return LiftSwingClearanceResult(False, "LIFT to SWING would lose required clearance", retained)
    if lift_peak_clearance_m - retained > maximum_clearance_drop_m:
        return LiftSwingClearanceResult(False, "LIFT to SWING clearance drop exceeds limit", retained)
    return LiftSwingClearanceResult(True, "clearance continuity verified", retained)


@dataclass(frozen=True)
class PlaceLoadDwellResult:
    satisfied: bool
    dwell_s: float
    consecutive_samples: int
    reason: str


@dataclass
class PlaceLoadDwellGuard:
    minimum_load_n: float
    required_dwell_s: float
    required_consecutive_samples: int = 2
    load_started_at_s: float | None = None
    consecutive_samples: int = 0
    last_time_s: float | None = None

    def reset(self) -> None:
        self.load_started_at_s = None
        self.consecutive_samples = 0
        self.last_time_s = None

    def update(self, *, time_s: float, contact_class: str, load_n: float) -> PlaceLoadDwellResult:
        if not _finite(time_s) or (self.last_time_s is not None and time_s < self.last_time_s):
            self.reset()
            return PlaceLoadDwellResult(False, 0.0, 0, "non-monotonic or non-finite time")
        self.last_time_s = float(time_s)
        valid = str(contact_class).upper() == "TOP" and _finite(load_n) and float(load_n) >= self.minimum_load_n
        if not valid:
            self.reset()
            return PlaceLoadDwellResult(False, 0.0, 0, "TOP contact with sufficient load is not present")
        if self.load_started_at_s is None:
            self.load_started_at_s = float(time_s)
        self.consecutive_samples += 1
        dwell = max(0.0, float(time_s) - self.load_started_at_s)
        satisfied = dwell >= self.required_dwell_s and self.consecutive_samples >= self.required_consecutive_samples
        return PlaceLoadDwellResult(satisfied, dwell, self.consecutive_samples, "" if satisfied else "waiting for stable TOP load dwell")


@dataclass(frozen=True)
class FinalSuccessCriteria:
    minimum_top_load_n: float
    minimum_top_load_dwell_s: float
    maximum_home_error_deg: float
    maximum_roll_deg: float
    maximum_pitch_deg: float
    maximum_root_linear_speed_m_s: float
    maximum_root_angular_speed_deg_s: float
    minimum_forward_clearance_m: float


@dataclass(frozen=True)
class FinalSuccessObservation:
    leg_traversal_valid: Mapping[Leg, bool]
    leg_airborne_seen: Mapping[Leg, bool]
    illegal_drive_up: Mapping[Leg, bool]
    final_contact_class: Mapping[Leg, str]
    final_load_n: Mapping[Leg, float]
    final_top_load_dwell_s: Mapping[Leg, float]
    home_error_deg: Mapping[str, float]
    roll_deg: float
    pitch_deg: float
    root_linear_speed_m_s: float
    root_angular_speed_deg_s: float
    forward_clearance_m: float
    dangerous_collision: bool = False
    joint_limit_violation: bool = False
    ik_failure: bool = False
    state_timeout: bool = False
    critical_recovery_skipped: bool = False
    concurrent_home_recovery_verified: bool = False


def evaluate_final_success(
    observation: FinalSuccessObservation, criteria: FinalSuccessCriteria
) -> GuardDecision:
    unmet: list[str] = []
    for leg in LEGS:
        if not observation.leg_traversal_valid.get(leg, False):
            unmet.append(f"{leg.value} linkage traversal not verified")
        if not observation.leg_airborne_seen.get(leg, False):
            unmet.append(f"{leg.value} has no AIRBORNE evidence")
        if observation.illegal_drive_up.get(leg, False):
            unmet.append(f"{leg.value} illegal loaded front-face drive-up")
        if str(observation.final_contact_class.get(leg, "UNKNOWN")).upper() != "TOP":
            unmet.append(f"{leg.value} final contact is not TOP")
        load = observation.final_load_n.get(leg, float("nan"))
        if not _finite(load) or float(load) < criteria.minimum_top_load_n:
            unmet.append(f"{leg.value} final TOP load below threshold")
        if observation.final_top_load_dwell_s.get(leg, 0.0) < criteria.minimum_top_load_dwell_s:
            unmet.append(f"{leg.value} final TOP load dwell too short")
    if not observation.home_error_deg or any(
        not _finite(error) or abs(float(error)) > criteria.maximum_home_error_deg
        for error in observation.home_error_deg.values()
    ):
        unmet.append("home pose tolerance not satisfied")
    if not _finite(observation.roll_deg) or abs(observation.roll_deg) > criteria.maximum_roll_deg:
        unmet.append("final roll exceeds limit")
    if not _finite(observation.pitch_deg) or abs(observation.pitch_deg) > criteria.maximum_pitch_deg:
        unmet.append("final pitch exceeds limit")
    if not _finite(observation.root_linear_speed_m_s) or observation.root_linear_speed_m_s > criteria.maximum_root_linear_speed_m_s:
        unmet.append("root linear velocity has not settled")
    if not _finite(observation.root_angular_speed_deg_s) or observation.root_angular_speed_deg_s > criteria.maximum_root_angular_speed_deg_s:
        unmet.append("root angular velocity has not settled")
    if not _finite(observation.forward_clearance_m) or observation.forward_clearance_m < criteria.minimum_forward_clearance_m:
        unmet.append("insufficient final forward workspace")
    if observation.dangerous_collision:
        unmet.append("dangerous collision observed")
    if observation.joint_limit_violation:
        unmet.append("joint limit violation observed")
    if observation.ik_failure:
        unmet.append("critical IK failure observed")
    if observation.state_timeout:
        unmet.append("state timeout observed")
    if observation.critical_recovery_skipped:
        unmet.append("critical traversal state was skipped")
    if not observation.concurrent_home_recovery_verified:
        unmet.append("concurrent wheel+servo home recovery not verified")
    metrics = {
        "roll_deg": observation.roll_deg,
        "pitch_deg": observation.pitch_deg,
        "root_linear_speed_m_s": observation.root_linear_speed_m_s,
        "root_angular_speed_deg_s": observation.root_angular_speed_deg_s,
        "forward_clearance_m": observation.forward_clearance_m,
    }
    return GuardDecision(not unmet, reason="" if not unmet else "; ".join(unmet), metrics=metrics, unmet=tuple(unmet))


__all__ = [
    "ActiveLegMode",
    "AnchoredSupportAngleGuard",
    "AnchoredTransferConfig",
    "COMDirectionalConfig",
    "COMDirectionalDetector",
    "COMDirectionalResult",
    "COMTransferMethod",
    "ContactAnchorTracker",
    "ContactDriftResult",
    "ContinuousOffsetManager",
    "CorrectionIsolationResult",
    "FinalSuccessCriteria",
    "FinalSuccessObservation",
    "GuardDecision",
    "ImpulseReactionGuard",
    "ImpulseStage",
    "ImpulseTransferConfig",
    "LEGS",
    "Leg",
    "LegIKCandidate",
    "LegIKDecision",
    "LiftSwingClearanceResult",
    "PerLegIKBatchDecision",
    "PlaceLoadDwellGuard",
    "PlaceLoadDwellResult",
    "SingleRelatchTarget",
    "StateScopedWheelRamp",
    "TargetLatchResult",
    "TransferObservation",
    "accept_per_leg_ik",
    "evaluate_final_success",
    "evaluate_lift_to_swing_clearance",
    "isolate_active_leg_corrections",
]
