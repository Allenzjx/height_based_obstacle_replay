"""Pure-Python command executor for the recording-derived 50 mm FSM.

The executor deliberately depends on a very small, duck-typed adapter protocol:

``capture_command_state()``
    Return ``{"servos": {name: deg}, "wheels": {name: rad_s}}`` for the
    command targets that are actually staged on the articulation.

``apply_motion_batch(payload)``
    Atomically stage the complete servo and wheel maps and return an
    acknowledgement.  One executor update performs at most one such call.

No Isaac/Omniverse module is imported here.  The same implementation can
therefore be tested with a fake adapter before a simulator is launched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .com_transfer_primitives import (
    ActiveLegMode,
    ContinuousOffsetManager,
    Leg,
    LegIKCandidate,
    PerLegIKBatchDecision,
    accept_per_leg_ik,
    isolate_active_leg_corrections,
)


_LEG_PREFIX: Mapping[Leg, str] = MappingProxyType(
    {
        Leg.FL: "front_left_",
        Leg.FR: "front_right_",
        Leg.RL: "rear_left_",
        Leg.RR: "rear_right_",
    }
)


def _plain_kind(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, Enum):
        value = value.value
    text = str(value).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text or default


def _finite_mapping(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = {str(name): float(target) for name, target in value.items()}
    if any(not math.isfinite(target) for target in result.values()):
        raise ValueError(f"{label} contains a non-finite target")
    return result


def _freeze(value: Any) -> Any:
    """Recursively freeze evidence stored on :class:`ExecutionResult`."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _leg_for_joint(name: str) -> Leg | None:
    lower = str(name).lower()
    for leg, prefix in _LEG_PREFIX.items():
        if lower.startswith(prefix):
            return leg
    return None


def _by_leg(targets: Mapping[str, float]) -> dict[Leg, dict[str, float]]:
    result = {leg: {} for leg in Leg}
    for name, value in targets.items():
        leg = _leg_for_joint(name)
        if leg is not None:
            result[leg][str(name)] = float(value)
    return result


def _flat_by_leg(targets: Mapping[Leg, Mapping[str, float]]) -> dict[str, float]:
    return {
        str(name): float(value)
        for leg in Leg
        for name, value in targets.get(leg, {}).items()
    }


def _normalize_leg(value: Any) -> Leg | None:
    if value in (None, "", "NONE"):
        return None
    return value if isinstance(value, Leg) else Leg(str(value).upper())


def _normalize_legs(values: Iterable[Any]) -> tuple[Leg, ...]:
    return tuple(
        value if isinstance(value, Leg) else Leg(str(value).upper())
        for value in values
        if value not in (None, "", "NONE")
    )


def _active_mode(state: Any) -> ActiveLegMode:
    explicit = getattr(state, "active_leg_mode", None)
    if explicit is not None:
        return (
            explicit
            if isinstance(explicit, ActiveLegMode)
            else ActiveLegMode(_plain_kind(explicit, "OTHER"))
        )
    label = " ".join(
        str(getattr(state, name, "") or "").upper()
        for name in ("state_id", "state_name")
    )
    if "HOLD" in label and "LIFT" in label:
        return ActiveLegMode.HOLD_LIFT
    if "SWING" in label or "CLEAR" in label:
        return ActiveLegMode.SWING_CLEAR
    if "UNLOAD" in label:
        return ActiveLegMode.UNLOAD
    if "LIFT" in label:
        return ActiveLegMode.LIFT
    if "PLACE" in label:
        return ActiveLegMode.PLACE
    if "CONFIRM" in label:
        return ActiveLegMode.CONFIRM
    return ActiveLegMode.OTHER


def _curve_fraction(kind: str, normalized_time: float) -> float:
    value = max(0.0, min(1.0, float(normalized_time)))
    if kind == "HOLD":
        return 0.0
    if kind == "STEP":
        return 1.0
    if kind == "CUBIC":
        return value * value * (3.0 - 2.0 * value)
    if kind in {"LINEAR", "LINEAR_RAMP", "RECORDING_RAMP", "RECORDING_PROFILE"}:
        return value
    raise ValueError(f"unsupported trajectory type: {kind}")


def _recording_fraction(profile: Any, normalized_time: float, fallback_kind: str) -> float:
    """Evaluate optional normalized recording knots without inventing timing.

    A profile may be a callable accepting ``u`` in ``[0, 1]`` or a sequence of
    ``(u, value)`` pairs.  When no knots are supplied, recording ramp/profile
    follows its duration-normalized linear envelope and is still rate limited.
    """

    u = max(0.0, min(1.0, float(normalized_time)))
    if profile is None:
        return _curve_fraction(fallback_kind, u)
    if callable(profile):
        result = float(profile(u))
        if not math.isfinite(result):
            raise ValueError("recording profile returned a non-finite fraction")
        return max(0.0, min(1.0, result))
    if not isinstance(profile, Sequence) or isinstance(profile, (str, bytes)):
        raise TypeError("recording profile must be callable or a knot sequence")
    knots: list[tuple[float, float]] = []
    for item in profile:
        if isinstance(item, Mapping):
            x = float(item.get("time", item.get("u")))
            y = float(item.get("value", item.get("fraction")))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            x, y = float(item[0]), float(item[1])
        else:
            raise TypeError("recording profile knots must be (time, fraction) pairs")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("recording profile contains a non-finite knot")
        knots.append((x, y))
    if not knots:
        return _curve_fraction(fallback_kind, u)
    knots.sort(key=lambda row: row[0])
    if knots[0][0] > 0.0 or knots[-1][0] < 1.0:
        raise ValueError("recording profile knots must cover normalized time [0, 1]")
    if any(right[0] <= left[0] for left, right in zip(knots, knots[1:])):
        raise ValueError("recording profile knot times must be strictly increasing")
    if u <= knots[0][0]:
        return max(0.0, min(1.0, knots[0][1]))
    for left, right in zip(knots, knots[1:]):
        if u <= right[0]:
            weight = (u - left[0]) / (right[0] - left[0])
            value = left[1] + weight * (right[1] - left[1])
            return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, knots[-1][1]))


def _rate_limited(
    previous: Mapping[str, float],
    desired: Mapping[str, float],
    *,
    maximum_rate: float | Mapping[str, float],
    dt_s: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, target in desired.items():
        current = float(previous[name])
        limit = (
            float(maximum_rate.get(name, 0.0))
            if isinstance(maximum_rate, Mapping)
            else float(maximum_rate)
        )
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(f"rate limit for {name} must be finite and non-negative")
        maximum_delta = limit * max(0.0, float(dt_s))
        result[name] = current + max(-maximum_delta, min(maximum_delta, float(target) - current))
    return result


def _maximum_delta(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return max((abs(float(right[name]) - float(left[name])) for name in right), default=0.0)


def _maps_close(left: Mapping[str, float], right: Mapping[str, float], tolerance: float) -> bool:
    return set(left) == set(right) and all(
        abs(float(left[name]) - float(right[name])) <= float(tolerance) for name in left
    )


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable evidence for one executor tick."""

    state_id: str
    tick_index: int
    time_s: float
    elapsed_s: float
    dt_s: float
    requested: Mapping[str, Any]
    applied: Mapping[str, Any]
    ack: Mapping[str, Any]
    ik_evidence: Mapping[str, Any]
    isolation_evidence: Mapping[str, Any]
    continuity_evidence: Mapping[str, Any]
    atomic_evidence: Mapping[str, Any]
    endpoint_reached: bool
    hold_complete: bool
    ok: bool
    fail_closed: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        for name in (
            "requested",
            "applied",
            "ack",
            "ik_evidence",
            "isolation_evidence",
            "continuity_evidence",
            "atomic_evidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "tick_index": self.tick_index,
            "time_s": self.time_s,
            "elapsed_s": self.elapsed_s,
            "dt_s": self.dt_s,
            "requested": _thaw(self.requested),
            "applied": _thaw(self.applied),
            "ack": _thaw(self.ack),
            "ik_evidence": _thaw(self.ik_evidence),
            "isolation_evidence": _thaw(self.isolation_evidence),
            "continuity_evidence": _thaw(self.continuity_evidence),
            "atomic_evidence": _thaw(self.atomic_evidence),
            "endpoint_reached": self.endpoint_reached,
            "hold_complete": self.hold_complete,
            "ok": self.ok,
            "fail_closed": self.fail_closed,
            "error": self.error,
        }


class FSM50CommandExecutor:
    """Generate and atomically apply one complete command batch per tick."""

    def __init__(
        self,
        adapter: Any,
        *,
        servo_rate_deg_s: float | Mapping[str, float] = 150.0,
        wheel_acceleration_rad_s2: float | Mapping[str, float] = 4.0,
        correction_rate_deg_s: float = 60.0,
        endpoint_tolerance: float = 1.0e-6,
    ) -> None:
        if not callable(getattr(adapter, "capture_command_state", None)):
            raise TypeError("adapter must provide capture_command_state()")
        if not callable(getattr(adapter, "apply_motion_batch", None)):
            raise TypeError("adapter must provide apply_motion_batch(payload)")
        if not math.isfinite(float(correction_rate_deg_s)) or correction_rate_deg_s < 0.0:
            raise ValueError("correction_rate_deg_s must be finite and non-negative")
        if not math.isfinite(float(endpoint_tolerance)) or endpoint_tolerance < 0.0:
            raise ValueError("endpoint_tolerance must be finite and non-negative")
        self.adapter = adapter
        self.servo_rate_deg_s = servo_rate_deg_s
        self.wheel_acceleration_rad_s2 = wheel_acceleration_rad_s2
        self.correction_rate_deg_s = float(correction_rate_deg_s)
        self.endpoint_tolerance = float(endpoint_tolerance)
        self.offsets = ContinuousOffsetManager()
        self.state: Any | None = None
        self.state_id = ""
        self.mode = ActiveLegMode.OTHER
        self.active_leg: Leg | None = None
        self.support_legs: tuple[Leg, ...] = ()
        self.place_confirm_blend = 0.0
        self.entry_time_s = 0.0
        self.last_time_s = 0.0
        self.tick_index = 0
        self.entry_sequence = 0
        self.explicit_hold_s = 0.0
        self.endpoint_reached_at_s: float | None = None
        self.servo_kind = "HOLD"
        self.wheel_kind = "HOLD"
        self.servo_duration_s = 0.0
        self.wheel_duration_s = 0.0
        self.servo_profile: Any = None
        self.wheel_profile: Any = None
        self.entry_servo: dict[str, float] = {}
        self.entry_wheel: dict[str, float] = {}
        self.servo_profile_start: dict[str, float] = {}
        self.servo_end: dict[str, float] = {}
        self.wheel_end: dict[str, float] = {}
        self.last_servo: dict[str, float] = {}
        self.last_wheel: dict[str, float] = {}
        self.entry_offsets: dict[str, float] = {}
        self.declared_servo_start: dict[str, float] = {}
        self.declared_wheel_start: dict[str, float] = {}
        self.fault_latched = False
        self.fault_reason = ""

    @staticmethod
    def _duration(
        state: Any,
        attribute: str,
        start: Mapping[str, float],
        end: Mapping[str, float],
        rate: float | Mapping[str, float],
        kind: str,
    ) -> float:
        explicit = getattr(state, attribute, None)
        if explicit is not None:
            value = float(explicit)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{attribute} must be finite and non-negative")
            return value
        if kind == "HOLD" or _maps_close(start, end, 1.0e-12):
            return 0.0
        durations: list[float] = []
        for name, target in end.items():
            limit = float(rate.get(name, 0.0)) if isinstance(rate, Mapping) else float(rate)
            delta = abs(float(target) - float(start[name]))
            if delta > 0.0 and limit <= 0.0:
                raise ValueError(f"positive {attribute} motion for {name} has a zero rate limit")
            durations.append(delta / limit if delta > 0.0 else 0.0)
        return max(durations, default=0.0)

    def _isolation_scales(self) -> Mapping[Leg, float]:
        zeros = {leg: {name: 0.0 for name in values} for leg, values in _by_leg(self.entry_servo).items()}
        result = isolate_active_leg_corrections(
            _by_leg(self.entry_servo),
            zeros,
            active_leg=self.active_leg,
            support_legs=self.support_legs,
            mode=self.mode,
            place_confirm_blend=self.place_confirm_blend,
        )
        return result.scales

    def enter_state(
        self,
        state: Any,
        *,
        time_s: float,
        explicit_hold_s: float | None = None,
        whole_body_corrections_deg: Mapping[Any, Mapping[str, float]] | None = None,
        place_confirm_blend: float = 0.0,
    ) -> Mapping[str, Any]:
        """Capture the actual staged command and initialize a continuous state.

        Entry never writes to the adapter; the first :meth:`update` owns the
        next tick's single atomic write.
        """

        now = float(time_s)
        if not math.isfinite(now):
            raise ValueError("state entry time must be finite")
        captured = self.adapter.capture_command_state()
        if not isinstance(captured, Mapping):
            raise TypeError("capture_command_state() must return a mapping")
        servos = _finite_mapping(captured.get("servos"), label="captured servos")
        wheels = _finite_mapping(captured.get("wheels"), label="captured wheels")
        if not servos or not wheels:
            raise ValueError("captured command state must contain servo and wheel targets")

        state_id = str(getattr(state, "state_id", "") or "")
        if not state_id:
            raise ValueError("state_id is required")
        servo_endpoint = _finite_mapping(
            getattr(state, "servo_end_target", {}) or {}, label="servo_end_target"
        )
        wheel_endpoint = _finite_mapping(
            getattr(state, "wheel_end_target", {}) or {}, label="wheel_end_target"
        )
        unknown_servos = sorted(set(servo_endpoint) - set(servos))
        unknown_wheels = sorted(set(wheel_endpoint) - set(wheels))
        if unknown_servos or unknown_wheels:
            raise ValueError(
                f"state targets unknown actuators: servos={unknown_servos}, wheels={unknown_wheels}"
            )

        if not self.offsets.current_offsets:
            self.offsets.initialize({name: 0.0 for name in servos})
        else:
            for name in servos:
                self.offsets.current_offsets.setdefault(name, 0.0)
                self.offsets.target_offsets.setdefault(name, self.offsets.current_offsets[name])

        corrections = self._normalize_corrections(whole_body_corrections_deg, set(servos))
        self.offsets.enter_state(state_id, corrections if corrections is not None else None)
        self.state = state
        self.state_id = state_id
        self.mode = _active_mode(state)
        self.active_leg = _normalize_leg(
            getattr(state, "active_leg", None) or getattr(state, "swing_leg", None)
        )
        self.support_legs = _normalize_legs(getattr(state, "support_legs", ()) or ())
        self.place_confirm_blend = max(0.0, min(1.0, float(place_confirm_blend)))
        self.entry_time_s = now
        self.last_time_s = now
        self.tick_index = 0
        self.entry_sequence += 1
        hold = getattr(state, "explicit_hold_s", 0.0) if explicit_hold_s is None else explicit_hold_s
        self.explicit_hold_s = float(hold)
        if not math.isfinite(self.explicit_hold_s) or self.explicit_hold_s < 0.0:
            raise ValueError("explicit_hold_s must be finite and non-negative")
        self.endpoint_reached_at_s = None
        self.servo_kind = _plain_kind(getattr(state, "servo_trajectory_type", None), "HOLD")
        self.wheel_kind = _plain_kind(getattr(state, "wheel_trajectory_type", None), "HOLD")
        self.servo_profile = getattr(state, "servo_recording_profile", None)
        self.wheel_profile = getattr(state, "wheel_recording_profile", None)
        self.entry_servo = dict(servos)
        self.entry_wheel = dict(wheels)
        self.last_servo = dict(servos)
        self.last_wheel = dict(wheels)
        self.entry_offsets = dict(self.offsets.current_offsets)
        scales = self._isolation_scales()
        self.servo_profile_start = {
            name: float(value)
            - float(scales.get(_leg_for_joint(name), 1.0)) * float(self.entry_offsets.get(name, 0.0))
            for name, value in servos.items()
        }
        self.servo_end = dict(self.servo_profile_start)
        self.servo_end.update(servo_endpoint)
        self.wheel_end = dict(wheels)
        self.wheel_end.update(wheel_endpoint)
        self.declared_servo_start = _finite_mapping(
            getattr(state, "servo_start_target", {}) or {}, label="servo_start_target"
        )
        self.declared_wheel_start = _finite_mapping(
            getattr(state, "wheel_start_target", {}) or {}, label="wheel_start_target"
        )
        self.servo_duration_s = self._duration(
            state,
            "servo_duration_s",
            self.servo_profile_start,
            self.servo_end,
            self.servo_rate_deg_s,
            self.servo_kind,
        )
        self.wheel_duration_s = self._duration(
            state,
            "wheel_duration_s",
            self.entry_wheel,
            self.wheel_end,
            self.wheel_acceleration_rad_s2,
            self.wheel_kind,
        )
        self.fault_latched = False
        self.fault_reason = ""
        return _freeze(
            {
                "state_id": self.state_id,
                "captured_command_state": {"servos": servos, "wheels": wheels},
                "servo_profile_start_deg": self.servo_profile_start,
                "servo_endpoint_deg": self.servo_end,
                "wheel_endpoint_rad_s": self.wheel_end,
                "mode": self.mode.value,
                "active_leg": None if self.active_leg is None else self.active_leg.value,
                "support_legs": [leg.value for leg in self.support_legs],
            }
        )

    @staticmethod
    def _normalize_corrections(
        values: Mapping[Any, Mapping[str, float]] | None,
        known_servos: set[str],
    ) -> dict[str, float] | None:
        if values is None:
            return None
        if not isinstance(values, Mapping):
            raise TypeError("whole_body_corrections_deg must be a leg mapping")
        flattened: dict[str, float] = {}
        for _leg_key, joints in values.items():
            if not isinstance(joints, Mapping):
                raise TypeError("each correction leg must map joint names to values")
            for name, raw in joints.items():
                joint = str(name)
                value = float(raw)
                if joint not in known_servos:
                    raise ValueError(f"correction targets unknown servo {joint}")
                if not math.isfinite(value):
                    raise ValueError(f"correction for {joint} is non-finite")
                flattened[joint] = value
        return flattened

    def _trajectory_targets(self, elapsed_s: float) -> tuple[dict[str, float], dict[str, float]]:
        servo_u = 1.0 if self.servo_duration_s <= 0.0 else elapsed_s / self.servo_duration_s
        wheel_u = 1.0 if self.wheel_duration_s <= 0.0 else elapsed_s / self.wheel_duration_s
        servo_fraction = (
            _recording_fraction(self.servo_profile, servo_u, self.servo_kind)
            if self.servo_kind == "RECORDING_RAMP"
            else _curve_fraction(self.servo_kind, servo_u)
        )
        wheel_fraction = (
            _recording_fraction(self.wheel_profile, wheel_u, self.wheel_kind)
            if self.wheel_kind == "RECORDING_PROFILE"
            else _curve_fraction(self.wheel_kind, wheel_u)
        )
        servo = {
            name: float(start) + servo_fraction * (float(self.servo_end[name]) - float(start))
            for name, start in self.servo_profile_start.items()
        }
        wheel = {
            name: float(start) + wheel_fraction * (float(self.wheel_end[name]) - float(start))
            for name, start in self.entry_wheel.items()
        }
        return servo, wheel

    @staticmethod
    def _ik_evidence(decision: PerLegIKBatchDecision | None) -> dict[str, Any]:
        if decision is None:
            return {"used": False, "rejected_legs": [], "decisions": {}}
        return {
            "used": True,
            "rejected_legs": [leg.value for leg in decision.rejected_legs],
            "decisions": {
                leg.value: {
                    "accepted": row.accepted,
                    "targets_deg": dict(row.targets_deg),
                    "preserved_reference_joints": list(row.preserved_reference_joints),
                    "boundary_clamped_joints": list(row.boundary_clamped_joints),
                    "reason": row.reason,
                }
                for leg, row in decision.decisions.items()
            },
        }

    def _apply_ik(
        self,
        desired: dict[str, float],
        candidates: Mapping[Any, LegIKCandidate] | None,
    ) -> tuple[dict[str, float], PerLegIKBatchDecision | None]:
        if not candidates:
            return desired, None
        normalized: dict[Leg, LegIKCandidate] = {}
        for raw_leg, candidate in candidates.items():
            leg = raw_leg if isinstance(raw_leg, Leg) else Leg(str(raw_leg).upper())
            if not isinstance(candidate, LegIKCandidate):
                raise TypeError("ik_candidates values must be LegIKCandidate instances")
            unknown = sorted(set(candidate.requested_targets_deg) - set(desired))
            if unknown:
                raise ValueError(f"IK candidate for {leg.value} targets unknown servos: {unknown}")
            normalized[leg] = candidate
        decision = accept_per_leg_ik(normalized)
        output = dict(desired)
        for leg, row in decision.decisions.items():
            candidate = normalized[leg]
            if row.accepted:
                output.update({name: float(value) for name, value in row.targets_deg.items()})
            else:
                # Reject only this leg's proposal.  Other valid legs continue,
                # while the rejected joints retain their last applied target.
                for name in candidate.requested_targets_deg:
                    output[name] = float(self.last_servo[name])
        return output, decision

    @staticmethod
    def _ack_target_matches(
        ack: Mapping[str, Any],
        key: str,
        requested: Mapping[str, float],
        *,
        tolerance: float = 1.0e-6,
    ) -> bool:
        value = ack.get(key)
        if not isinstance(value, Mapping):
            return False
        try:
            applied = _finite_mapping(value, label=key)
        except (TypeError, ValueError):
            return False
        return _maps_close(applied, requested, tolerance)

    def _atomic_ack_evidence(
        self,
        ack: Mapping[str, Any],
        servo_targets: Mapping[str, float],
        wheel_targets: Mapping[str, float],
    ) -> dict[str, Any]:
        errors: list[str] = []
        if str(ack.get("error", "") or ""):
            errors.append(f"adapter ack error: {ack.get('error')}")
        if ack.get("servo_applied") is not True:
            errors.append("ack does not confirm servo application")
        if ack.get("wheel_applied") is not True:
            errors.append("ack does not confirm wheel application")
        try:
            applied_step = int(ack["applied_sim_step"])
            first_step = int(ack["first_physics_step"])
        except (KeyError, TypeError, ValueError):
            applied_step = -1
            first_step = -1
            errors.append("ack lacks shared applied_sim_step/first_physics_step")
        else:
            if first_step != applied_step + 1:
                errors.append("first_physics_step is not the tick after applied_sim_step")
        channel_steps: dict[str, int] = {}
        for key in ("servo_applied_sim_step", "wheel_applied_sim_step"):
            if key in ack:
                try:
                    channel_steps[key] = int(ack[key])
                except (TypeError, ValueError):
                    errors.append(f"{key} is not an integer")
        if channel_steps and (
            set(channel_steps) != {"servo_applied_sim_step", "wheel_applied_sim_step"}
            or len(set(channel_steps.values())) != 1
            or next(iter(channel_steps.values())) != applied_step
        ):
            errors.append("servo and wheel applied_sim_step differ")
        channel_first: dict[str, int] = {}
        for key in ("servo_first_physics_step", "wheel_first_physics_step"):
            if key in ack:
                try:
                    channel_first[key] = int(ack[key])
                except (TypeError, ValueError):
                    errors.append(f"{key} is not an integer")
        if channel_first and (
            set(channel_first) != {"servo_first_physics_step", "wheel_first_physics_step"}
            or len(set(channel_first.values())) != 1
            or next(iter(channel_first.values())) != first_step
        ):
            errors.append("servo and wheel first_physics_step differ")
        if not self._ack_target_matches(ack, "servo_targets_applied", servo_targets):
            errors.append("ack servo_targets_applied differs from requested full map")
        if not self._ack_target_matches(ack, "wheel_targets_applied", wheel_targets):
            errors.append("ack wheel_targets_applied differs from requested full map")
        skew = ack.get("motion_start_skew_s")
        if skew is None or not math.isfinite(float(skew)) or abs(float(skew)) > 1.0e-12:
            errors.append("ack does not prove zero servo/wheel motion-start skew")
        return {
            "required": True,
            "verified": not errors,
            "applied_sim_step": applied_step,
            "first_physics_step": first_step,
            "channel_applied_steps": channel_steps,
            "channel_first_physics_steps": channel_first,
            "motion_start_skew_s": skew,
            "errors": errors,
        }

    def _failure_result(
        self,
        *,
        now: float,
        elapsed: float,
        dt_s: float,
        requested: Mapping[str, Any],
        applied: Mapping[str, Any],
        ack: Mapping[str, Any],
        ik: Mapping[str, Any],
        isolation: Mapping[str, Any],
        continuity: Mapping[str, Any],
        atomic: Mapping[str, Any],
        error: str,
    ) -> ExecutionResult:
        self.fault_latched = True
        self.fault_reason = str(error)
        return ExecutionResult(
            state_id=self.state_id,
            tick_index=self.tick_index,
            time_s=now,
            elapsed_s=elapsed,
            dt_s=dt_s,
            requested=requested,
            applied=applied,
            ack=ack,
            ik_evidence=ik,
            isolation_evidence=isolation,
            continuity_evidence=continuity,
            atomic_evidence=atomic,
            endpoint_reached=False,
            hold_complete=False,
            ok=False,
            fail_closed=True,
            error=str(error),
        )

    def update(
        self,
        *,
        time_s: float,
        whole_body_corrections_deg: Mapping[Any, Mapping[str, float]] | None = None,
        ik_candidates: Mapping[Any, LegIKCandidate] | None = None,
        place_confirm_blend: float | None = None,
    ) -> ExecutionResult:
        """Generate one tick and make at most one atomic adapter call."""

        if self.state is None:
            raise RuntimeError("enter_state() must be called before update()")
        now = float(time_s)
        if not math.isfinite(now) or now + 1.0e-12 < self.last_time_s:
            raise ValueError("executor time must be finite and monotonic")
        dt_s = max(0.0, now - self.last_time_s)
        elapsed = max(0.0, now - self.entry_time_s)
        self.tick_index += 1
        if self.fault_latched:
            return self._failure_result(
                now=now,
                elapsed=elapsed,
                dt_s=dt_s,
                requested={},
                applied={},
                ack={},
                ik={"used": False},
                isolation={},
                continuity={"fault_latched": True},
                atomic={"required": bool(getattr(self.state, "atomic_concurrent", False)), "verified": False},
                error=self.fault_reason or "executor fault is latched",
            )

        if place_confirm_blend is not None:
            self.place_confirm_blend = max(0.0, min(1.0, float(place_confirm_blend)))
        corrections = self._normalize_corrections(
            whole_body_corrections_deg, set(self.entry_servo)
        )
        if corrections is not None:
            self.offsets.enter_state(self.state_id, corrections)
        current_offsets = dict(
            self.offsets.update(dt_s, maximum_rate_deg_s=self.correction_rate_deg_s)
        )

        requested: dict[str, Any] = {}
        applied: dict[str, Any] = {}
        ack: dict[str, Any] = {}
        ik_evidence: dict[str, Any] = {"used": False, "rejected_legs": [], "decisions": {}}
        isolation_evidence: dict[str, Any] = {}
        continuity: dict[str, Any] = {}
        atomic_evidence: dict[str, Any] = {
            "required": bool(getattr(self.state, "atomic_concurrent", False)),
            "verified": False,
            "errors": [],
        }
        try:
            servo_profile_target, wheel_profile_target = self._trajectory_targets(elapsed)
            isolation = isolate_active_leg_corrections(
                _by_leg(servo_profile_target),
                _by_leg(current_offsets),
                active_leg=self.active_leg,
                support_legs=self.support_legs,
                mode=self.mode,
                place_confirm_blend=self.place_confirm_blend,
            )
            isolated_servo = _flat_by_leg(isolation.targets_deg)
            # Preserve any servo whose naming scheme is not leg-addressable.
            isolated_servo.update(
                {
                    name: value
                    for name, value in servo_profile_target.items()
                    if _leg_for_joint(name) is None
                }
            )
            ik_target, ik_decision = self._apply_ik(isolated_servo, ik_candidates)
            ik_evidence = self._ik_evidence(ik_decision)
            final_servo = _rate_limited(
                self.last_servo,
                ik_target,
                maximum_rate=self.servo_rate_deg_s,
                dt_s=dt_s,
            )
            final_wheel = _rate_limited(
                self.last_wheel,
                wheel_profile_target,
                maximum_rate=self.wheel_acceleration_rad_s2,
                dt_s=dt_s,
            )
            requested = {
                "servo_endpoint_deg": dict(self.servo_end),
                "wheel_endpoint_rad_s": dict(self.wheel_end),
                "servo_profile_target_deg": servo_profile_target,
                "wheel_profile_target_rad_s": wheel_profile_target,
                "servo_after_isolation_and_ik_deg": ik_target,
                "current_correction_offsets_deg": current_offsets,
            }
            isolation_evidence = {
                "mode": self.mode.value,
                "active_leg": None if self.active_leg is None else self.active_leg.value,
                "support_legs": [leg.value for leg in self.support_legs],
                "place_confirm_blend": self.place_confirm_blend,
                "scales": {leg.value: value for leg, value in isolation.scales.items()},
                "applied_corrections_deg": {
                    leg.value: dict(values)
                    for leg, values in isolation.applied_corrections_deg.items()
                },
                "ignored": {
                    leg.value: list(values) for leg, values in isolation.ignored.items()
                },
            }
            continuity = {
                "started_from_captured_command": True,
                "entry_servo_targets_deg": self.entry_servo,
                "entry_wheel_targets_rad_s": self.entry_wheel,
                "declared_servo_start_target_deg": self.declared_servo_start,
                "declared_wheel_start_target_rad_s": self.declared_wheel_start,
                "previous_servo_targets_deg": self.last_servo,
                "previous_wheel_targets_rad_s": self.last_wheel,
                "maximum_servo_delta_this_tick_deg": _maximum_delta(self.last_servo, final_servo),
                "maximum_wheel_delta_this_tick_rad_s": _maximum_delta(self.last_wheel, final_wheel),
                "entry_correction_offsets_deg": self.entry_offsets,
                "current_correction_offsets_deg": current_offsets,
                "offsets_carried_across_state": True,
                "wheel_command_carried_from_capture": True,
            }
            payload = {
                "batch_id": f"fsm50-{self.entry_sequence}-{self.state_id}-{self.tick_index}",
                "source": "fsm50_executor",
                "state_id": self.state_id,
                "tick_index": self.tick_index,
                "atomic_concurrent": bool(getattr(self.state, "atomic_concurrent", False)),
                "servo_targets_deg": dict(final_servo),
                "wheel_targets_rad_s": dict(final_wheel),
            }
            # This is the only adapter write in this method.
            raw_ack = self.adapter.apply_motion_batch(payload)
            if not isinstance(raw_ack, Mapping):
                raise RuntimeError("apply_motion_batch() did not return an acknowledgement mapping")
            ack = dict(raw_ack)
            applied = {
                "servo_targets_deg": dict(final_servo),
                "wheel_targets_rad_s": dict(final_wheel),
                "payload": payload,
            }
            if bool(getattr(self.state, "atomic_concurrent", False)):
                atomic_evidence = self._atomic_ack_evidence(ack, final_servo, final_wheel)
                if not atomic_evidence["verified"]:
                    return self._failure_result(
                        now=now,
                        elapsed=elapsed,
                        dt_s=dt_s,
                        requested=requested,
                        applied=applied,
                        ack=ack,
                        ik=ik_evidence,
                        isolation=isolation_evidence,
                        continuity=continuity,
                        atomic=atomic_evidence,
                        error="atomic acknowledgement verification failed: "
                        + "; ".join(atomic_evidence["errors"]),
                    )
            else:
                ack_error = str(ack.get("error", "") or "")
                atomic_evidence = {
                    "required": False,
                    "verified": not ack_error,
                    "errors": [] if not ack_error else [ack_error],
                }
                if ack_error:
                    return self._failure_result(
                        now=now,
                        elapsed=elapsed,
                        dt_s=dt_s,
                        requested=requested,
                        applied=applied,
                        ack=ack,
                        ik=ik_evidence,
                        isolation=isolation_evidence,
                        continuity=continuity,
                        atomic=atomic_evidence,
                        error=f"adapter acknowledgement error: {ack_error}",
                    )
        except Exception as exc:
            return self._failure_result(
                now=now,
                elapsed=elapsed,
                dt_s=dt_s,
                requested=requested,
                applied=applied,
                ack=ack,
                ik=ik_evidence,
                isolation=isolation_evidence,
                continuity=continuity,
                atomic=atomic_evidence,
                error=f"{type(exc).__name__}: {exc}",
            )

        self.last_servo = dict(applied["servo_targets_deg"])
        self.last_wheel = dict(applied["wheel_targets_rad_s"])
        self.last_time_s = now
        servo_profile_complete = self.servo_kind == "HOLD" or elapsed + 1.0e-12 >= self.servo_duration_s
        wheel_profile_complete = self.wheel_kind == "HOLD" or elapsed + 1.0e-12 >= self.wheel_duration_s
        endpoint_reached = bool(
            servo_profile_complete
            and wheel_profile_complete
            and _maps_close(self.last_servo, requested["servo_after_isolation_and_ik_deg"], self.endpoint_tolerance)
            and _maps_close(self.last_wheel, requested["wheel_profile_target_rad_s"], self.endpoint_tolerance)
        )
        if endpoint_reached and self.endpoint_reached_at_s is None:
            self.endpoint_reached_at_s = now
        hold_elapsed = (
            0.0
            if self.endpoint_reached_at_s is None
            else max(0.0, now - self.endpoint_reached_at_s)
        )
        hold_complete = bool(
            self.endpoint_reached_at_s is not None
            and hold_elapsed + 1.0e-12 >= self.explicit_hold_s
        )
        requested["explicit_hold_s"] = self.explicit_hold_s
        requested["hold_elapsed_s"] = hold_elapsed
        return ExecutionResult(
            state_id=self.state_id,
            tick_index=self.tick_index,
            time_s=now,
            elapsed_s=elapsed,
            dt_s=dt_s,
            requested=requested,
            applied=applied,
            ack=ack,
            ik_evidence=ik_evidence,
            isolation_evidence=isolation_evidence,
            continuity_evidence=continuity,
            atomic_evidence=atomic_evidence,
            endpoint_reached=endpoint_reached,
            hold_complete=hold_complete,
            ok=True,
        )

    execute_tick = update


FSM50Executor = FSM50CommandExecutor


__all__ = [
    "ExecutionResult",
    "FSM50CommandExecutor",
    "FSM50Executor",
]
