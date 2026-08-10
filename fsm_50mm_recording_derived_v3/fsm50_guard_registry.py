"""Fail-closed live guard registry for the recording-derived 50 mm FSM.

The registry consumes the immutable observation contract and never advances a
state from elapsed time or target arrival alone.  Time is used only for stable
dwell after a physical predicate has become true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

from .com_transfer_primitives import GuardDecision, Leg
from .fsm50_state_model import FSM50State, PrimaryDiagonal

if TYPE_CHECKING:
    from .fsm50_observation import FSM50Observation


LIFECYCLE_GUARDS: frozenset[str] = frozenset(
    {
        "LIVE_EVIDENCE_COMPLETE",
        "STATE_PHYSICAL_EVENT",
        "STATE_PHYSICAL_EVENT_WITH_DWELL",
        "FAIL_CLOSED_SAFETY_STOP",
    }
)

STATE_GUARDS: frozenset[str] = frozenset(
    {
        "INITIAL_STABLE",
        "ANCHORED_TRANSFER",
        "IMPULSE_PRELOAD",
        "IMPULSE_PUSH",
        "IMPULSE_RELEASE",
        "TRANSFER_SETTLED",
        "UNLOAD_READY",
        "UNLOADED",
        "AIRBORNE",
        "AIRBORNE_HOLD",
        "FACE_CLEARED",
        "OVER_TOP",
        "TOP_CONTACT",
        "TOP_LOAD",
        "SUPPORT_STABLE",
        "COM_DIRECTION",
        "FRONT_PAIR_TOP",
        "ADVANCE_GEOMETRY",
        "WHEELS_STOPPED",
        "WORKSPACE_CLEAR",
        "DIAGONAL_SUPPORT",
        "ALL_TOP",
        "HOME_AND_FORWARD",
        "FINAL_SETTLED",
        "TERMINAL_SUCCESS",
        "TERMINAL_SAFE_STOP",
    }
)


_TRAVERSAL_PHASE_ORDER: tuple[str, ...] = (
    "UNSEEN",
    "LOADED_SUPPORT",
    "UNLOADED",
    "AIRBORNE",
    "FACE_CLEARED",
    "TOP_CONTACT",
    "TOP_LOAD_CONFIRMED",
)
_TRAVERSAL_PHASE_RANK: Mapping[str, int] = {
    name: index for index, name in enumerate(_TRAVERSAL_PHASE_ORDER)
}


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


def _leg_keyed(values: Mapping[Any, Any]) -> dict[Leg, Any]:
    result: dict[Leg, Any] = {}
    for key, value in dict(values or {}).items():
        try:
            result[key if isinstance(key, Leg) else Leg(str(getattr(key, "value", key)).upper())] = value
        except ValueError:
            continue
    return result


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _norm(values: Iterable[float]) -> float:
    parsed = [float(value) for value in values]
    return math.sqrt(sum(value * value for value in parsed))


def _unit_xy(values: Iterable[float]) -> tuple[float, float] | None:
    parsed = list(values)
    if len(parsed) < 2 or not _finite(parsed[0]) or not _finite(parsed[1]):
        return None
    length = math.hypot(float(parsed[0]), float(parsed[1]))
    if length <= 1.0e-12:
        return None
    return float(parsed[0]) / length, float(parsed[1]) / length


def _rpy_deg(quaternion_wxyz: Iterable[float]) -> tuple[float, float, float]:
    values = list(quaternion_wxyz)
    if len(values) != 4 or not all(_finite(value) for value in values):
        return float("nan"), float("nan"), float("nan")
    w, x, y, z = (float(value) for value in values)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _threshold(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if not _finite(value):
        raise ValueError(f"guard threshold {name} must be finite")
    return float(value)


@dataclass
class GuardEvaluationContext:
    state: FSM50State
    thresholds: Mapping[str, Any]
    entered_at_s: float
    baseline_com_xy: tuple[float, float]
    baseline_load_n: dict[Leg, float]
    baseline_wheel_rotation_rad: dict[Leg, float]
    baseline_root_x_m: float
    predicate_true_since_s: dict[str, float] = field(default_factory=dict)
    front_face_entry_rotation_rad: dict[Leg, float] = field(default_factory=dict)
    peak_clearance_m: dict[Leg, float] = field(default_factory=dict)
    illegal_drive_up: dict[Leg, bool] = field(
        default_factory=lambda: {leg: False for leg in Leg}
    )
    traversal_valid: dict[Leg, bool] = field(
        default_factory=lambda: {leg: False for leg in Leg}
    )
    airborne_seen: dict[Leg, bool] = field(
        default_factory=lambda: {leg: False for leg in Leg}
    )
    traversal_phase: dict[Leg, str] = field(
        default_factory=lambda: {leg: "UNSEEN" for leg in Leg}
    )
    loaded_support_seen_s: dict[Leg, float] = field(default_factory=dict)
    unload_start_s: dict[Leg, float] = field(default_factory=dict)
    unload_end_s: dict[Leg, float] = field(default_factory=dict)
    airborne_start_s: dict[Leg, float] = field(default_factory=dict)
    airborne_end_s: dict[Leg, float] = field(default_factory=dict)
    front_face_crossing_s: dict[Leg, float] = field(default_factory=dict)
    top_contact_s: dict[Leg, float] = field(default_factory=dict)
    top_load_confirm_s: dict[Leg, float] = field(default_factory=dict)
    loaded_front_face_rotation_rad: dict[Leg, float] = field(
        default_factory=lambda: {leg: 0.0 for leg in Leg}
    )
    last_wheel_rotation_rad: dict[Leg, float] = field(default_factory=dict)
    last_contact_class: dict[Leg, str] = field(default_factory=dict)
    concurrent_home_recovery_verified: bool = False
    critical_recovery_skipped: bool = False
    safe_stop_applied: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def enter(
        cls,
        state: FSM50State,
        observation: "FSM50Observation",
        thresholds: Mapping[str, Any],
        *,
        inherited: "GuardEvaluationContext | None" = None,
    ) -> "GuardEvaluationContext":
        loads = _loads(observation)
        rotations = _leg_keyed(observation.integrated_wheel_rotation_rad)
        com = observation.com_position_w
        root = observation.root_position_w
        context = cls(
            state=state,
            thresholds=dict(thresholds),
            entered_at_s=float(observation.time_s),
            baseline_com_xy=(float(com[0]), float(com[1])),
            baseline_load_n={leg: float(loads.get(leg, 0.0)) for leg in Leg},
            baseline_wheel_rotation_rad={
                leg: float(rotations.get(leg, 0.0)) for leg in Leg
            },
            baseline_root_x_m=float(root[0]),
        )
        classes = _classes(observation)
        loaded_threshold = _threshold(thresholds, "support_load_enter_n", 2.0)
        context.last_wheel_rotation_rad = {
            leg: float(value)
            for leg, value in rotations.items()
            if _finite(value)
        }
        context.last_contact_class = dict(classes)
        for leg in Leg:
            if (
                loads.get(leg, 0.0) >= loaded_threshold
                and classes.get(leg, "UNKNOWN") not in {"AIR", "UNKNOWN"}
            ):
                context.loaded_support_seen_s[leg] = float(observation.time_s)
                context.traversal_phase[leg] = "LOADED_SUPPORT"
        if inherited is not None:
            context.front_face_entry_rotation_rad = dict(
                inherited.front_face_entry_rotation_rad
            )
            context.peak_clearance_m = dict(inherited.peak_clearance_m)
            context.illegal_drive_up = dict(inherited.illegal_drive_up)
            context.traversal_valid = dict(inherited.traversal_valid)
            context.airborne_seen = dict(inherited.airborne_seen)
            context.traversal_phase = dict(inherited.traversal_phase)
            for name in (
                "loaded_support_seen_s",
                "unload_start_s",
                "unload_end_s",
                "airborne_start_s",
                "airborne_end_s",
                "front_face_crossing_s",
                "top_contact_s",
                "top_load_confirm_s",
                "loaded_front_face_rotation_rad",
                "last_wheel_rotation_rad",
                "last_contact_class",
            ):
                setattr(context, name, dict(getattr(inherited, name)))
            for leg in Leg:
                if (
                    leg not in context.loaded_support_seen_s
                    and loads.get(leg, 0.0) >= loaded_threshold
                    and classes.get(leg, "UNKNOWN") not in {"AIR", "UNKNOWN"}
                ):
                    context.loaded_support_seen_s[leg] = float(observation.time_s)
                    context.advance_traversal(leg, "LOADED_SUPPORT")
            context.concurrent_home_recovery_verified = bool(
                inherited.concurrent_home_recovery_verified
            )
            context.critical_recovery_skipped = bool(
                inherited.critical_recovery_skipped
            )
            context.safe_stop_applied = bool(inherited.safe_stop_applied)
        return context

    def phase_at_least(self, leg: Leg, phase: str) -> bool:
        current = self.traversal_phase.get(leg, "UNSEEN")
        return _TRAVERSAL_PHASE_RANK.get(current, -1) >= _TRAVERSAL_PHASE_RANK[phase]

    def advance_traversal(self, leg: Leg, phase: str) -> None:
        if phase not in _TRAVERSAL_PHASE_RANK:
            raise ValueError(f"unknown traversal phase: {phase}")
        if not self.phase_at_least(leg, phase):
            self.traversal_phase[leg] = phase

    def stable_dwell(
        self, name: str, *, time_s: float, condition: bool, required_s: float
    ) -> tuple[bool, float]:
        if not condition:
            self.predicate_true_since_s.pop(name, None)
            return False, 0.0
        start = self.predicate_true_since_s.setdefault(name, float(time_s))
        dwell = max(0.0, float(time_s) - start)
        return dwell >= max(0.0, float(required_s)), dwell


def _loads(observation: "FSM50Observation") -> dict[Leg, float]:
    ground = _leg_keyed(observation.filtered_ground_force_n)
    obstacle = _leg_keyed(observation.filtered_obstacle_force_n)
    return {
        leg: max(0.0, float(ground.get(leg, 0.0)))
        + max(0.0, float(obstacle.get(leg, 0.0)))
        for leg in Leg
    }


def _classes(observation: "FSM50Observation") -> dict[Leg, str]:
    return {leg: _enum_text(value) for leg, value in _leg_keyed(observation.wheel_contact_class).items()}


def _active_leg(state: FSM50State) -> Leg | None:
    return state.swing_leg or state.active_leg


class FSM50GuardRegistry:
    """Name-to-predicate registry with startup completeness validation."""

    def __init__(self) -> None:
        self._state_predicates: dict[
            str,
            Callable[["FSM50Observation", GuardEvaluationContext], GuardDecision],
        ] = {
            name: getattr(self, f"_guard_{name.lower()}") for name in STATE_GUARDS
        }
        if set(self._state_predicates) != set(STATE_GUARDS):
            raise RuntimeError("internal guard registry construction failed")

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._state_predicates) | LIFECYCLE_GUARDS

    def validate_states(self, states: Iterable[FSM50State]) -> None:
        unknown: list[str] = []
        for state in states:
            state_guard = str(state.guard or "").strip().upper()
            if state_guard not in STATE_GUARDS:
                unknown.append(f"{state.state_id}:{state.guard or '<empty>'}")
            for name in (
                state.entry_guard.kind,
                state.progress_guard.kind,
                state.exit_guard.kind,
                state.abort_guard.kind,
            ):
                normalized = str(name or "").strip().upper()
                if not normalized or normalized not in self.names:
                    unknown.append(f"{state.state_id}:{name or '<empty>'}")
        if unknown:
            raise ValueError("unknown or unimplemented guards: " + ", ".join(unknown))

    def evaluate(
        self,
        name: str,
        observation: "FSM50Observation",
        context: GuardEvaluationContext,
    ) -> GuardDecision:
        normalized = str(name).strip().upper()
        if normalized not in self.names:
            raise KeyError(f"unknown or unimplemented guard: {name}")
        if normalized == "LIVE_EVIDENCE_COMPLETE":
            if observation.control_ready:
                return GuardDecision(True, reason="critical live evidence complete")
            return GuardDecision(
                False,
                True,
                "critical live evidence missing",
                unmet=tuple(observation.missing_critical_fields),
            )
        if normalized in {"STATE_PHYSICAL_EVENT", "STATE_PHYSICAL_EVENT_WITH_DWELL"}:
            state_guard = str(context.state.guard or "").strip().upper()
            if state_guard not in STATE_GUARDS:
                raise KeyError(
                    "state physical-event lifecycle guard requires a concrete "
                    f"state predicate, got {context.state.guard!r}"
                )
            return self.evaluate(context.state.guard, observation, context)
        if normalized == "FAIL_CLOSED_SAFETY_STOP":
            return self._safety(observation, context)
        if not observation.control_ready:
            return GuardDecision(
                False,
                True,
                "critical telemetry unavailable",
                unmet=tuple(observation.missing_critical_fields),
            )
        self._update_traversal(observation, context)
        return self._state_predicates[normalized](observation, context)

    def _update_traversal(
        self, observation: "FSM50Observation", context: GuardEvaluationContext
    ) -> None:
        classes = _classes(observation)
        rotations = _leg_keyed(observation.integrated_wheel_rotation_rad)
        clearance = _leg_keyed(observation.wheel_clearance_over_top_m)
        loads = _loads(observation)
        loaded_limit = _threshold(
            context.thresholds, "maximum_loaded_front_face_rotation_rad", 0.15
        )
        load_threshold = _threshold(context.thresholds, "support_load_exit_n", 1.5)
        loaded_support_threshold = _threshold(
            context.thresholds, "support_load_enter_n", 2.0
        )
        time_s = float(observation.time_s)
        for leg in Leg:
            if _finite(clearance.get(leg)):
                context.peak_clearance_m[leg] = max(
                    context.peak_clearance_m.get(leg, -math.inf),
                    float(clearance[leg]),
                )
            classification = classes.get(leg, "UNKNOWN")
            rotation = rotations.get(leg)
            previous_rotation = context.last_wheel_rotation_rad.get(leg)
            if (
                loads[leg] >= loaded_support_threshold
                and classification not in {"AIR", "UNKNOWN"}
            ):
                context.loaded_support_seen_s.setdefault(leg, time_s)
                context.advance_traversal(leg, "LOADED_SUPPORT")
            if classification == "FRONT_FACE" and loads[leg] >= load_threshold:
                entry = context.front_face_entry_rotation_rad.setdefault(
                    leg, float(rotation) if _finite(rotation) else 0.0
                )
                if _finite(rotation) and _finite(previous_rotation):
                    # Integrated rotation is already expressed in canonical
                    # forward wheel coordinates by the telemetry collector.
                    delta = max(0.0, float(rotation) - float(previous_rotation))
                    context.loaded_front_face_rotation_rad[leg] = (
                        context.loaded_front_face_rotation_rad.get(leg, 0.0) + delta
                    )
                if context.loaded_front_face_rotation_rad.get(leg, 0.0) > loaded_limit:
                    context.illegal_drive_up[leg] = True
                    context.traversal_valid[leg] = False
            else:
                context.front_face_entry_rotation_rad.pop(leg, None)
            if _finite(rotation):
                context.last_wheel_rotation_rad[leg] = float(rotation)
            context.last_contact_class[leg] = classification

    def _safety(
        self, observation: "FSM50Observation", context: GuardEvaluationContext
    ) -> GuardDecision:
        if not observation.control_ready:
            return GuardDecision(
                False,
                True,
                "critical telemetry unavailable",
                unmet=tuple(observation.missing_critical_fields),
            )
        self._update_traversal(observation, context)
        unmet: list[str] = []
        if not bool(observation.nonwheel_contact_evidence_valid):
            unmet.append("nonwheel obstacle-contact evidence invalid")
        elif bool(observation.nonwheel_obstacle_contact):
            unmet.append("dangerous nonwheel obstacle contact")
        if any(context.illegal_drive_up.values()):
            unmet.append("ILLEGAL_DRIVE_UP")
        margins = [
            float(value)
            for value in dict(observation.joint_limit_margin_rad).values()
            if _finite(value)
        ]
        required_margin = math.radians(
            _threshold(context.thresholds, "minimum_joint_limit_margin_deg", 1.0)
        )
        if not margins or min(margins) < required_margin:
            unmet.append("joint-limit margin below threshold")
        roll, pitch, _yaw = _rpy_deg(observation.root_orientation_wxyz)
        if not _finite(roll) or abs(roll) > _threshold(context.thresholds, "maximum_roll_deg", 12.0):
            unmet.append("roll safety limit exceeded")
        if not _finite(pitch) or abs(pitch) > _threshold(context.thresholds, "maximum_pitch_deg", 18.0):
            unmet.append("pitch safety limit exceeded")
        angular_deg_s = math.degrees(_norm(observation.root_angular_velocity_w))
        if angular_deg_s > _threshold(
            context.thresholds, "maximum_angular_speed_deg_s", 30.0
        ):
            unmet.append("angular velocity safety limit exceeded")
        return GuardDecision(
            False,
            abort=bool(unmet),
            reason="; ".join(unmet) if unmet else "safety evidence valid",
            metrics={
                "roll_deg": roll,
                "pitch_deg": pitch,
                "angular_speed_deg_s": angular_deg_s,
                "illegal_drive_up": {
                    leg.value: value for leg, value in context.illegal_drive_up.items()
                },
            },
            unmet=tuple(unmet),
        )

    def _direction_metrics(
        self, observation: "FSM50Observation", context: GuardEvaluationContext
    ) -> tuple[tuple[float, float] | None, float, float, float]:
        state = context.state
        direction = None
        if state.target_com_leg is not None:
            try:
                target = observation.target_direction_for(state.target_com_leg)
                if _enum_text(target.target_leg) != state.target_com_leg.value:
                    return None, float("nan"), float("nan"), float("nan")
                direction = _unit_xy(target.direction_w)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                direction = None
        if direction is None:
            return None, float("nan"), float("nan"), float("nan")
        dx = float(observation.com_position_w[0]) - context.baseline_com_xy[0]
        dy = float(observation.com_position_w[1]) - context.baseline_com_xy[1]
        vx = float(observation.com_velocity_w[0])
        vy = float(observation.com_velocity_w[1])
        projected = dx * direction[0] + dy * direction[1]
        velocity = vx * direction[0] + vy * direction[1]
        lateral = abs(-dx * direction[1] + dy * direction[0])
        return direction, projected, velocity, lateral

    def _support_ok(
        self, observation: "FSM50Observation", context: GuardEvaluationContext
    ) -> tuple[bool, tuple[str, ...]]:
        state = context.state
        loads = _loads(observation)
        classes = _classes(observation)
        drift = _leg_keyed(observation.contact_drift_m)
        threshold = _threshold(context.thresholds, "support_load_enter_n", 2.0)
        maximum_drift = min(
            float(state.allowed_contact_drift),
            _threshold(context.thresholds, "maximum_contact_drift_m", 0.015),
        )
        unmet: list[str] = []
        for leg in state.support_legs:
            if classes.get(leg, "UNKNOWN") in {"AIR", "UNKNOWN"}:
                unmet.append(f"{leg.value} support contact absent")
            if loads.get(leg, 0.0) < threshold:
                unmet.append(f"{leg.value} support load low")
            if not _finite(drift.get(leg)) or float(drift[leg]) > maximum_drift:
                unmet.append(f"{leg.value} contact drift invalid")
        return not unmet, tuple(unmet)

    def _guard_initial_stable(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        classes = _classes(observation)
        contact_ok = all(classes.get(leg) in {"GROUND", "TOP"} for leg in Leg)
        linear = _norm(observation.root_linear_velocity_w)
        angular = math.degrees(_norm(observation.root_angular_velocity_w))
        condition = (
            contact_ok
            and linear <= _threshold(context.thresholds, "final_root_linear_speed_m_s", 0.03)
            and angular <= _threshold(context.thresholds, "final_root_angular_speed_deg_s", 5.0)
        )
        satisfied, dwell = context.stable_dwell(
            "initial_stable",
            time_s=observation.time_s,
            condition=condition,
            required_s=context.state.settle_duration,
        )
        return GuardDecision(satisfied, reason="" if satisfied else "initial support/body not stably settled", metrics={"dwell_s": dwell})

    def _guard_anchored_transfer(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        support_ok, unmet = self._support_ok(observation, context)
        direction, displacement, velocity, lateral = self._direction_metrics(observation, context)
        minimum = _threshold(context.thresholds, "minimum_com_displacement_m", 0.004)
        if direction is None:
            return GuardDecision(False, True, "target-leg world direction unavailable")
        all_unmet = list(unmet)
        if displacement < minimum:
            all_unmet.append("anchored COM displacement below threshold")
        if not support_ok:
            all_unmet.append("anchored support invalid")
        return GuardDecision(not all_unmet, reason="; ".join(all_unmet), metrics={"projected_com_displacement_m": displacement, "projected_com_velocity_m_s": velocity, "lateral_displacement_m": lateral}, unmet=tuple(all_unmet))

    def _guard_impulse_preload(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = context.state.impulse_leg
        if leg is None:
            return GuardDecision(False, True, "impulse leg is not configured")
        load = _loads(observation)[leg]
        required = _threshold(context.thresholds, "impulse_preload_enter_n", 3.0)
        support_ok, unmet = self._support_ok(observation, context)
        satisfied = load >= required and support_ok
        return GuardDecision(satisfied, reason="" if satisfied else "; ".join((*unmet, "impulse preload below threshold" if load < required else "" )).strip("; "), metrics={"impulse_leg": leg.value, "load_n": load})

    def _guard_impulse_push(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        direction, displacement, velocity, _lateral = self._direction_metrics(observation, context)
        if direction is None:
            return GuardDecision(False, True, "target-leg world direction unavailable")
        satisfied = (
            displacement >= _threshold(context.thresholds, "minimum_com_displacement_m", 0.004)
            and velocity >= _threshold(context.thresholds, "minimum_com_velocity_m_s", 0.005)
        )
        return GuardDecision(satisfied, reason="" if satisfied else "impulse has not produced target COM displacement/velocity", metrics={"projected_com_displacement_m": displacement, "projected_com_velocity_m_s": velocity})

    def _guard_impulse_release(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = context.state.impulse_leg
        if leg is None:
            return GuardDecision(False, True, "impulse leg is not configured")
        load = _loads(observation)[leg]
        reference = max(
            context.baseline_load_n.get(leg, 0.0),
            _threshold(context.thresholds, "impulse_preload_enter_n", 3.0),
        )
        limit = reference * _threshold(context.thresholds, "impulse_release_ratio", 0.55)
        return GuardDecision(load <= limit, reason="" if load <= limit else "impulse load has not released", metrics={"load_n": load, "release_limit_n": limit})

    def _guard_transfer_settled(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        direction, displacement, velocity, _lateral = self._direction_metrics(observation, context)
        if context.state.target_com_leg is not None and direction is None:
            return GuardDecision(False, True, "target-leg world direction unavailable")
        minimum = _threshold(context.thresholds, "minimum_com_displacement_m", 0.004)
        speed = _norm(observation.com_velocity_w[:2])
        angular = math.degrees(_norm(observation.root_angular_velocity_w))
        condition = (
            (context.state.target_com_leg is None or displacement >= minimum)
            and speed <= _threshold(context.thresholds, "maximum_settle_com_speed_m_s", 0.02)
            and angular <= _threshold(context.thresholds, "maximum_settle_angular_speed_deg_s", 8.0)
        )
        satisfied, dwell = context.stable_dwell("transfer_settled", time_s=observation.time_s, condition=condition, required_s=context.state.settle_duration)
        return GuardDecision(satisfied, reason="" if satisfied else "COM/body has not settled after transfer", metrics={"projected_com_displacement_m": displacement, "projected_com_velocity_m_s": velocity, "com_speed_m_s": speed, "angular_speed_deg_s": angular, "dwell_s": dwell})

    def _guard_unload_ready(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = _active_leg(context.state)
        if leg is None:
            return GuardDecision(False, True, "unload active leg is not configured")
        load = _loads(observation)[leg]
        support_ok, unmet = self._support_ok(observation, context)
        corridor = observation.two_leg_corridor
        corridor_ok = (
            bool(corridor.applicable and corridor.valid)
            if len(context.state.support_legs) == 2
            else bool(observation.support_polygon_valid)
        )
        satisfied = load <= _threshold(context.thresholds, "unload_load_exit_n", 0.8) and support_ok and corridor_ok
        reasons = list(unmet)
        if load > _threshold(context.thresholds, "unload_load_exit_n", 0.8):
            reasons.append("active-leg load not low enough")
        if not corridor_ok:
            reasons.append("COM support corridor/polygon invalid")
        return GuardDecision(satisfied, reason="; ".join(reasons), metrics={"active_leg": leg.value, "load_n": load, "corridor_or_polygon_ok": corridor_ok})

    def _guard_unloaded(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = _active_leg(context.state)
        if leg is None:
            return GuardDecision(False, True, "unload active leg is not configured")
        load = _loads(observation)[leg]
        loaded_support_seen = leg in context.loaded_support_seen_s
        condition = (
            loaded_support_seen
            and load <= _threshold(context.thresholds, "unload_load_enter_n", 0.5)
        )
        satisfied, dwell = context.stable_dwell("unloaded", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "minimum_unload_dwell_s", 0.025))
        if satisfied:
            context.unload_start_s.setdefault(
                leg, context.predicate_true_since_s.get("unloaded", float(observation.time_s))
            )
            context.advance_traversal(leg, "UNLOADED")
        reason = ""
        if not satisfied:
            reason = (
                "loaded support evidence missing before unload"
                if not loaded_support_seen
                else "active leg has not remained unloaded"
            )
        return GuardDecision(satisfied, reason=reason, metrics={"active_leg": leg.value, "load_n": load, "loaded_support_seen": loaded_support_seen, "dwell_s": dwell})

    def _airborne_condition(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> tuple[bool, Leg | None, float]:
        leg = _active_leg(context.state)
        if leg is None:
            return False, None, float("nan")
        classification = _classes(observation).get(leg, "UNKNOWN")
        clearance = float(_leg_keyed(observation.wheel_clearance_over_top_m).get(leg, float("nan")))
        return classification == "AIR" and _finite(clearance) and clearance >= _threshold(context.thresholds, "minimum_airborne_clearance_m", 0.003), leg, clearance

    def _guard_airborne(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        condition, leg, clearance = self._airborne_condition(observation, context)
        ordered_unload = leg is not None and context.phase_at_least(leg, "UNLOADED")
        condition = condition and ordered_unload
        satisfied, dwell = context.stable_dwell("airborne", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "minimum_airborne_dwell_s", 0.05))
        if satisfied and leg is not None:
            start_s = context.predicate_true_since_s.get(
                "airborne", float(observation.time_s)
            )
            context.airborne_start_s.setdefault(leg, start_s)
            context.unload_end_s.setdefault(leg, start_s)
            context.airborne_seen[leg] = True
            context.advance_traversal(leg, "AIRBORNE")
        reason = ""
        if not satisfied:
            reason = (
                "ordered UNLOAD evidence missing before AIR"
                if condition is False and not ordered_unload
                else "linkage lift/AIR interval not established"
            )
        return GuardDecision(satisfied, reason=reason, metrics={"active_leg": "" if leg is None else leg.value, "clearance_m": clearance, "ordered_unload": ordered_unload, "dwell_s": dwell})

    def _guard_airborne_hold(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        condition, leg, clearance = self._airborne_condition(observation, context)
        peak = context.peak_clearance_m.get(leg, clearance) if leg is not None else clearance
        retained = (
            condition
            and leg is not None
            and context.phase_at_least(leg, "AIRBORNE")
            and peak - clearance
            <= _threshold(
                context.thresholds, "maximum_lift_to_swing_clearance_drop_m", 0.004
            )
        )
        satisfied, dwell = context.stable_dwell("airborne_hold", time_s=observation.time_s, condition=retained, required_s=context.state.settle_duration)
        return GuardDecision(satisfied, reason="" if satisfied else "lift posture/clearance was not retained", metrics={"clearance_m": clearance, "peak_clearance_m": peak, "dwell_s": dwell})

    def _guard_face_cleared(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = _active_leg(context.state)
        if leg is None:
            return GuardDecision(False, True, "swing leg is not configured")
        classes = _classes(observation)
        face = _leg_keyed(observation.wheel_front_face_clearance_m).get(leg)
        top = _leg_keyed(observation.wheel_clearance_over_top_m).get(leg)
        satisfied = (
            context.phase_at_least(leg, "AIRBORNE")
            and context.airborne_seen.get(leg, False)
            and classes.get(leg) == "AIR"
            and _finite(face)
            and float(face) >= _threshold(context.thresholds, "face_crossing_enter_m", 0.004)
            and _finite(top)
            and float(top) >= _threshold(context.thresholds, "minimum_airborne_clearance_m", 0.003)
            and not context.illegal_drive_up.get(leg, False)
        )
        if satisfied:
            context.front_face_crossing_s.setdefault(leg, float(observation.time_s))
            context.advance_traversal(leg, "FACE_CLEARED")
        return GuardDecision(satisfied, reason="" if satisfied else "AIRBORNE front-face clearance not established", metrics={"active_leg": leg.value, "front_face_clearance_m": face, "top_clearance_m": top})

    def _guard_over_top(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        return self._guard_face_cleared(observation, context)

    def _guard_top_contact(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = _active_leg(context.state)
        if leg is None:
            return GuardDecision(False, True, "place leg is not configured")
        top_contact = _classes(observation).get(leg) == "TOP"
        ordered = context.phase_at_least(leg, "FACE_CLEARED")
        if top_contact and not ordered:
            context.illegal_drive_up[leg] = True
            context.traversal_valid[leg] = False
            return GuardDecision(
                False,
                True,
                "TOP contact occurred before ordered UNLOAD/AIR/CLEAR_FACE evidence",
                metrics={"active_leg": leg.value, "traversal_phase": context.traversal_phase[leg]},
                unmet=(f"{leg.value} ILLEGAL_DRIVE_UP",),
            )
        valid = top_contact and ordered and not context.illegal_drive_up.get(leg, False)
        if valid:
            context.airborne_end_s.setdefault(leg, float(observation.time_s))
            context.top_contact_s.setdefault(leg, float(observation.time_s))
            context.advance_traversal(leg, "TOP_CONTACT")
        return GuardDecision(valid, reason="" if valid else "TOP contact after ordered AIR/CLEAR_FACE is not present", metrics={"active_leg": leg.value, "traversal_phase": context.traversal_phase[leg]})

    def _guard_top_load(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = context.state.active_leg
        if leg is None:
            # Confirmation states intentionally clear swing_leg but retain the
            # traversed leg in active_leg.
            return GuardDecision(False, True, "TOP-load leg is not configured")
        load = _loads(observation)[leg]
        top_loaded = (
            _classes(observation).get(leg) == "TOP"
            and load >= _threshold(context.thresholds, "top_load_enter_n", 2.0)
        )
        ordered = context.phase_at_least(leg, "TOP_CONTACT")
        if top_loaded and not ordered:
            context.illegal_drive_up[leg] = True
            context.traversal_valid[leg] = False
            return GuardDecision(
                False,
                True,
                "TOP load occurred before ordered place evidence",
                metrics={"active_leg": leg.value, "load_n": load, "traversal_phase": context.traversal_phase[leg]},
                unmet=(f"{leg.value} ILLEGAL_DRIVE_UP",),
            )
        condition = (
            top_loaded
            and ordered
            and context.airborne_seen.get(leg, False)
            and not context.illegal_drive_up.get(leg, False)
        )
        satisfied, dwell = context.stable_dwell("top_load", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "top_load_dwell_s", 0.08))
        if satisfied:
            context.top_load_confirm_s.setdefault(leg, float(observation.time_s))
            context.advance_traversal(leg, "TOP_LOAD_CONFIRMED")
            context.traversal_valid[leg] = True
        return GuardDecision(satisfied, reason="" if satisfied else "stable ordered TOP load dwell missing", metrics={"active_leg": leg.value, "load_n": load, "traversal_phase": context.traversal_phase[leg], "dwell_s": dwell})

    def _guard_support_stable(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        valid, unmet = self._support_ok(observation, context)
        satisfied, dwell = context.stable_dwell("support_stable", time_s=observation.time_s, condition=valid, required_s=_threshold(context.thresholds, "support_persistence_s", 0.08))
        return GuardDecision(satisfied, reason="; ".join(unmet) if unmet else ("" if satisfied else "waiting for support persistence"), metrics={"dwell_s": dwell}, unmet=unmet)

    def _guard_com_direction(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        direction, displacement, velocity, lateral = self._direction_metrics(observation, context)
        if direction is None:
            return GuardDecision(False, True, "target-leg world direction unavailable")
        condition = displacement >= _threshold(context.thresholds, "minimum_com_displacement_m", 0.004) and velocity >= -_threshold(context.thresholds, "maximum_settle_com_speed_m_s", 0.02)
        return GuardDecision(condition, reason="" if condition else "COM has not moved toward the live target leg", metrics={"target_direction_world_xy": direction, "projected_com_displacement_m": displacement, "projected_com_velocity_m_s": velocity, "lateral_displacement_m": lateral})

    def _guard_front_pair_top(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        classes, loads = _classes(observation), _loads(observation)
        required = _threshold(context.thresholds, "top_load_enter_n", 2.0)
        condition = all(
            classes.get(leg) == "TOP"
            and loads[leg] >= required
            and context.traversal_valid.get(leg, False)
            for leg in (Leg.FL, Leg.FR)
        )
        satisfied, dwell = context.stable_dwell("front_pair_top", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "top_load_dwell_s", 0.08))
        return GuardDecision(satisfied, reason="" if satisfied else "front pair TOP support not stable", metrics={"dwell_s": dwell})

    def _guard_advance_geometry(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        if context.state.state_id.startswith("F"):
            displacement = float(observation.root_position_w[0]) - context.baseline_root_x_m
            required = _threshold(context.thresholds, "minimum_forward_workspace_m", 0.10)
            return GuardDecision(displacement >= required, reason="" if displacement >= required else "final forward workspace not reached", metrics={"forward_displacement_m": displacement})
        face = _leg_keyed(observation.wheel_front_face_clearance_m)
        classes = _classes(observation)
        distance = min(-float(face.get(leg, -math.inf)) for leg in (Leg.RL, Leg.RR))
        required = _threshold(context.thresholds, "approach_front_face_distance_m", 0.08)
        safe = all(classes.get(leg) not in {"FRONT_FACE", "UNKNOWN"} for leg in (Leg.RL, Leg.RR))
        return GuardDecision(distance <= required and safe, reason="" if distance <= required and safe else "rear wheels have not reached safe pre-face stop geometry", metrics={"rear_distance_to_face_m": distance, "rear_not_on_face": safe})

    def _guard_wheels_stopped(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        values = list(dict(observation.measured_wheel_velocity_rad_s).values())
        condition = bool(values) and all(_finite(value) and abs(float(value)) <= _threshold(context.thresholds, "wheel_stopped_rad_s", 0.08) for value in values)
        satisfied, dwell = context.stable_dwell("wheels_stopped", time_s=observation.time_s, condition=condition, required_s=context.state.settle_duration)
        return GuardDecision(satisfied, reason="" if satisfied else "measured wheels have not stopped", metrics={"dwell_s": dwell})

    def _guard_workspace_clear(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        leg = _active_leg(context.state)
        if leg is None:
            return GuardDecision(False, True, "workspace leg is not configured")
        face = _leg_keyed(observation.wheel_front_face_clearance_m).get(leg)
        classes = _classes(observation)
        condition = _finite(face) and float(face) >= _threshold(context.thresholds, "face_crossing_enter_m", 0.004) and classes.get(leg) in {"TOP", "AIR"}
        return GuardDecision(condition, reason="" if condition else "active-leg workspace is not clear", metrics={"active_leg": leg.value, "front_face_clearance_m": face})

    def _guard_diagonal_support(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        measured = _enum_text(observation.primary_diagonal)
        expected = context.state.primary_diagonal.value
        support_ok, unmet = self._support_ok(observation, context)
        corridor = observation.two_leg_corridor
        condition = (
            measured == expected
            and support_ok
            and bool(corridor.applicable)
            and bool(corridor.valid)
        )
        satisfied, dwell = context.stable_dwell("diagonal_support", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "support_persistence_s", 0.08))
        return GuardDecision(satisfied, reason="; ".join(unmet) if unmet else ("" if satisfied else "measured primary diagonal/corridor not stable"), metrics={"expected_primary_diagonal": expected, "measured_primary_diagonal": measured, "corridor_applicable": bool(corridor.applicable), "corridor_valid": bool(corridor.valid), "dwell_s": dwell})

    def _guard_all_top(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        classes, loads = _classes(observation), _loads(observation)
        threshold = _threshold(context.thresholds, "top_load_enter_n", 2.0)
        condition = all(classes.get(leg) == "TOP" and loads[leg] >= threshold and context.traversal_valid.get(leg, False) for leg in Leg)
        satisfied, dwell = context.stable_dwell("all_top", time_s=observation.time_s, condition=condition, required_s=_threshold(context.thresholds, "top_load_dwell_s", 0.08))
        return GuardDecision(satisfied, reason="" if satisfied else "four verified traversals/TOP loads not stable", metrics={"dwell_s": dwell})

    def _guard_home_and_forward(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        errors = list(dict(observation.actuator_target_error_rad).values())
        maximum = math.radians(_threshold(context.thresholds, "maximum_home_error_deg", 2.0))
        home = bool(errors) and all(_finite(value) and abs(float(value)) <= maximum for value in errors)
        forward = float(observation.root_position_w[0]) - context.baseline_root_x_m
        condition = home and forward >= _threshold(context.thresholds, "minimum_forward_workspace_m", 0.10)
        if condition:
            context.concurrent_home_recovery_verified = bool(context.extra.get("atomic_home_ack", False))
            condition = context.concurrent_home_recovery_verified
        return GuardDecision(condition, reason="" if condition else "home/forward/atomic recovery evidence incomplete", metrics={"home_error_max_rad": max((abs(float(value)) for value in errors), default=float("nan")), "forward_displacement_m": forward, "atomic_home_ack": context.concurrent_home_recovery_verified})

    def _guard_final_settled(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        classes = _classes(observation)
        wheels = list(dict(observation.measured_wheel_velocity_rad_s).values())
        roll, pitch, _yaw = _rpy_deg(observation.root_orientation_wxyz)
        condition = (
            all(classes.get(leg) == "TOP" for leg in Leg)
            and bool(wheels)
            and all(abs(float(value)) <= _threshold(context.thresholds, "wheel_stopped_rad_s", 0.08) for value in wheels if _finite(value))
            and _norm(observation.root_linear_velocity_w) <= _threshold(context.thresholds, "final_root_linear_speed_m_s", 0.03)
            and math.degrees(_norm(observation.root_angular_velocity_w)) <= _threshold(context.thresholds, "final_root_angular_speed_deg_s", 5.0)
            and abs(roll) <= _threshold(context.thresholds, "maximum_roll_deg", 12.0)
            and abs(pitch) <= _threshold(context.thresholds, "maximum_pitch_deg", 18.0)
        )
        satisfied, dwell = context.stable_dwell("final_settled", time_s=observation.time_s, condition=condition, required_s=context.state.settle_duration)
        return GuardDecision(satisfied, reason="" if satisfied else "final pose/velocity/TOP contacts not settled", metrics={"dwell_s": dwell, "roll_deg": roll, "pitch_deg": pitch})

    def _guard_terminal_success(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        classes = _classes(observation)
        unmet: list[str] = []
        for leg in Leg:
            if not context.traversal_valid.get(leg, False):
                unmet.append(f"{leg.value} traversal not verified")
            if not context.airborne_seen.get(leg, False):
                unmet.append(f"{leg.value} AIR interval missing")
            if context.illegal_drive_up.get(leg, False):
                unmet.append(f"{leg.value} ILLEGAL_DRIVE_UP")
            if classes.get(leg) != "TOP":
                unmet.append(f"{leg.value} final contact not TOP")
        if context.critical_recovery_skipped:
            unmet.append("critical state skipped")
        if not context.concurrent_home_recovery_verified:
            unmet.append("atomic home recovery not verified")
        return GuardDecision(not unmet, reason="; ".join(unmet), unmet=tuple(unmet))

    def _guard_terminal_safe_stop(self, observation: "FSM50Observation", context: GuardEvaluationContext) -> GuardDecision:
        wheels = list(dict(observation.measured_wheel_velocity_rad_s).values())
        stopped = context.safe_stop_applied and bool(wheels) and all(_finite(value) and abs(float(value)) <= _threshold(context.thresholds, "wheel_stopped_rad_s", 0.08) for value in wheels)
        return GuardDecision(stopped, reason="" if stopped else "SAFE_STOP commands/physical wheel stop not confirmed")


__all__ = [
    "FSM50GuardRegistry",
    "GuardEvaluationContext",
    "LIFECYCLE_GUARDS",
    "STATE_GUARDS",
]
