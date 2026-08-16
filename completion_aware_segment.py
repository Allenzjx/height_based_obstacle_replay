"""Transport-neutral completion-aware execution for one playback segment.

The production adapter owns servo slew and actuator writes.  This module owns
only the measured endpoint, wheel-duration, hold-duration, and liveness state
machine so playback and higher-level controllers can share one completion
contract without synthesizing per-physics-tick commands.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_SERVO_POSITION_TOLERANCE_DEG = 1.0
SERVO_COMPLETION_VELOCITY_DEG_S = 0.5
CONTACT_RESIDUAL_GRACE_S = 0.25
CONTACT_RESIDUAL_STABLE_S = 0.10
ACTUATOR_DIVERGENCE_WINDOW_S = 1.50
CONTACT_RESIDUAL_HARD_CAP_DEG = 3.0
SERVO_STALL_WINDOW_S = 0.75
SERVO_IMPROVEMENT_EPSILON_DEG = 0.05


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _finite_map(values: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    result: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        if not name:
            raise ValueError(f"{label} contains an empty key")
        if type(raw_value) not in (int, float):
            raise ValueError(f"{label}.{name} must be finite")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}.{name} must be finite")
        result[name] = value
    return result


@dataclass(frozen=True)
class SegmentCompletionSpec:
    """Immutable execution inputs copied from one production PlaybackSegment."""

    segment_index: int
    source_step: int
    source_step_id: str
    servo_targets_deg: Mapping[str, float]
    servo_duration_s: float
    servo_tolerance_deg: float
    recorded_servo_residual_deg: Mapping[str, float]
    legacy_missing_endpoint: bool
    wheel_active_duration_s: float
    explicit_hold_s: float

    def __post_init__(self) -> None:
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("segment_index must be a non-negative int")
        if type(self.source_step) is not int or self.source_step < 0:
            raise ValueError("source_step must be a non-negative int")
        if not isinstance(self.source_step_id, str):
            raise ValueError("source_step_id must be a string")
        if type(self.legacy_missing_endpoint) is not bool:
            raise ValueError("legacy_missing_endpoint must be bool")
        targets = _finite_map(self.servo_targets_deg, label="servo_targets_deg")
        residuals = _finite_map(
            self.recorded_servo_residual_deg,
            label="recorded_servo_residual_deg",
        )
        if not targets and residuals:
            raise ValueError("a segment without servo targets cannot carry residuals")
        if self.legacy_missing_endpoint and residuals:
            raise ValueError("legacy-missing endpoint requires an empty residual map")
        if targets and not self.legacy_missing_endpoint and set(residuals) != set(
            targets
        ):
            raise ValueError(
                "non-legacy residuals must exactly match all servo targets"
            )
        duration = _finite_nonnegative(
            self.servo_duration_s, label="servo_duration_s"
        )
        if type(self.servo_tolerance_deg) not in (int, float):
            raise ValueError("servo_tolerance_deg must be finite")
        tolerance = float(self.servo_tolerance_deg)
        if not math.isfinite(tolerance):
            raise ValueError("servo_tolerance_deg must be finite")
        if not (
            DEFAULT_SERVO_POSITION_TOLERANCE_DEG
            <= tolerance
            <= CONTACT_RESIDUAL_HARD_CAP_DEG
        ):
            raise ValueError("servo_tolerance_deg must be within 1..3 degrees")
        wheel_duration = _finite_nonnegative(
            self.wheel_active_duration_s,
            label="wheel_active_duration_s",
        )
        hold = _finite_nonnegative(self.explicit_hold_s, label="explicit_hold_s")
        object.__setattr__(self, "servo_targets_deg", MappingProxyType(targets))
        object.__setattr__(
            self,
            "recorded_servo_residual_deg",
            MappingProxyType(residuals),
        )
        object.__setattr__(self, "servo_duration_s", duration)
        object.__setattr__(self, "servo_tolerance_deg", tolerance)
        object.__setattr__(self, "wheel_active_duration_s", wheel_duration)
        object.__setattr__(self, "explicit_hold_s", hold)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "source_step": self.source_step,
            "source_step_id": self.source_step_id,
            "servo_targets_deg": dict(self.servo_targets_deg),
            "servo_duration_s": self.servo_duration_s,
            "servo_tolerance_deg": self.servo_tolerance_deg,
            "recorded_servo_residual_deg": dict(
                self.recorded_servo_residual_deg
            ),
            "legacy_missing_endpoint": self.legacy_missing_endpoint,
            "wheel_active_duration_s": self.wheel_active_duration_s,
            "explicit_hold_s": self.explicit_hold_s,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SegmentCompletionSpec":
        return cls(
            segment_index=values["segment_index"],
            source_step=values["source_step"],
            source_step_id=values.get("source_step_id", ""),
            servo_targets_deg=values.get("servo_targets_deg", {}),
            servo_duration_s=values.get("servo_duration_s", 0.0),
            servo_tolerance_deg=values.get(
                "servo_tolerance_deg", DEFAULT_SERVO_POSITION_TOLERANCE_DEG
            ),
            recorded_servo_residual_deg=values.get(
                "recorded_servo_residual_deg", {}
            ),
            legacy_missing_endpoint=values.get("legacy_missing_endpoint", False),
            wheel_active_duration_s=values.get("wheel_active_duration_s", 0.0),
            explicit_hold_s=values.get("explicit_hold_s", 0.0),
        )


@dataclass(frozen=True)
class SegmentFeedback:
    """One completed-physics-step measurement in actual-joint error space.

    ``servo_errors_deg`` must be calculated as measured actual degrees minus
    ``adapter.command_to_actual_target_deg(name, command_target)``.  It is not
    a canonical-command or drive-target map.
    """

    elapsed_s: float
    sim_time_s: float
    sim_step: int
    servo_errors_deg: Mapping[str, Any]
    servo_velocity_deg_s: Mapping[str, Any]
    tracking_evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        elapsed = _finite_nonnegative(self.elapsed_s, label="elapsed_s")
        sim_time = _finite_nonnegative(self.sim_time_s, label="sim_time_s")
        if type(self.sim_step) is not int or self.sim_step < 0:
            raise ValueError("sim_step must be a non-negative int")
        if not isinstance(self.servo_errors_deg, Mapping):
            raise ValueError("servo_errors_deg must be a mapping")
        if not isinstance(self.servo_velocity_deg_s, Mapping):
            raise ValueError("servo_velocity_deg_s must be a mapping")
        evidence = self.tracking_evidence
        if evidence is not None and not isinstance(evidence, Mapping):
            raise ValueError("tracking_evidence must be a mapping or None")
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "sim_time_s", sim_time)
        object.__setattr__(self, "servo_errors_deg", dict(self.servo_errors_deg))
        object.__setattr__(
            self, "servo_velocity_deg_s", dict(self.servo_velocity_deg_s)
        )
        object.__setattr__(
            self,
            "tracking_evidence",
            copy.deepcopy(dict(evidence or {})),
        )


class SegmentDecisionKind(str, Enum):
    WAIT = "WAIT"
    WHEEL_STOP_DUE = "WHEEL_STOP_DUE"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass(frozen=True)
class SegmentDecision:
    kind: SegmentDecisionKind
    segment_index: int
    source_step: int
    source_step_id: str
    sim_time_s: float
    sim_step: int
    segment_elapsed_s: float
    servo_planned_done: bool
    reference_position_done: bool
    servo_done: bool
    servo_errors_deg: Mapping[str, float]
    servo_velocity_deg_s: Mapping[str, float]
    max_servo_error_deg: float
    servo_tolerance_deg: float
    recorded_servo_residual_deg: Mapping[str, float]
    legacy_missing_endpoint: bool
    contact_candidate: bool
    contact_extension_s: float
    contact_grace_done: bool
    contact_stable: bool
    contact_window_ready: bool
    contact_window_duration_s: float
    contact_window_cap_deg: float
    contact_window_min_deg: float
    contact_window_max_deg: float
    contact_window_slope_deg_s: float
    divergence_window_duration_s: float
    divergence_window_min_deg: float
    divergence_window_max_deg: float
    divergence_window_slope_deg_s: float
    velocities_near_zero: bool
    wheel_elapsed_s: float
    wheel_duration_s: float
    wheel_done: bool
    hold_elapsed_s: float
    hold_duration_s: float
    hold_done: bool
    segment_done: bool
    wheel_stop_due: bool
    wheel_stop_acknowledged: bool
    stalled_for_s: float
    worst_joint: str
    tracking_evidence: Mapping[str, Any]
    phase: str
    failure_reason: str
    failure_code: str

    def to_mapping(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["kind"] = self.kind.value
        payload["servo_errors_deg"] = dict(self.servo_errors_deg)
        payload["servo_velocity_deg_s"] = dict(self.servo_velocity_deg_s)
        payload["recorded_servo_residual_deg"] = dict(
            self.recorded_servo_residual_deg
        )
        payload["tracking_evidence"] = copy.deepcopy(
            dict(self.tracking_evidence)
        )
        return payload


class CompletionAwareSegmentExecutor:
    """Stateful, transport-neutral implementation of production completion."""

    def __init__(
        self,
        *,
        servo_position_tolerance_deg: float = DEFAULT_SERVO_POSITION_TOLERANCE_DEG,
        contact_residual_grace_s: float = CONTACT_RESIDUAL_GRACE_S,
        contact_residual_stable_s: float = CONTACT_RESIDUAL_STABLE_S,
        actuator_divergence_window_s: float = ACTUATOR_DIVERGENCE_WINDOW_S,
        contact_residual_hard_cap_deg: float = CONTACT_RESIDUAL_HARD_CAP_DEG,
        servo_stall_window_s: float = SERVO_STALL_WINDOW_S,
        servo_improvement_epsilon_deg: float = SERVO_IMPROVEMENT_EPSILON_DEG,
        servo_completion_velocity_deg_s: float = SERVO_COMPLETION_VELOCITY_DEG_S,
    ) -> None:
        self.servo_position_tolerance_deg = _finite_nonnegative(
            servo_position_tolerance_deg,
            label="servo_position_tolerance_deg",
        )
        self.contact_residual_grace_s = _finite_nonnegative(
            contact_residual_grace_s, label="contact_residual_grace_s"
        )
        self.contact_residual_stable_s = _finite_nonnegative(
            contact_residual_stable_s, label="contact_residual_stable_s"
        )
        self.actuator_divergence_window_s = _finite_nonnegative(
            actuator_divergence_window_s,
            label="actuator_divergence_window_s",
        )
        self.contact_residual_hard_cap_deg = _finite_nonnegative(
            contact_residual_hard_cap_deg,
            label="contact_residual_hard_cap_deg",
        )
        self.servo_stall_window_s = _finite_nonnegative(
            servo_stall_window_s, label="servo_stall_window_s"
        )
        self.servo_improvement_epsilon_deg = _finite_nonnegative(
            servo_improvement_epsilon_deg,
            label="servo_improvement_epsilon_deg",
        )
        self.servo_completion_velocity_deg_s = _finite_nonnegative(
            servo_completion_velocity_deg_s,
            label="servo_completion_velocity_deg_s",
        )
        self.servo_tracking_hard_liveness_s = (
            self.actuator_divergence_window_s + self.servo_stall_window_s
        )
        self.reset()

    def reset(self) -> None:
        self.spec: SegmentCompletionSpec | None = None
        self.start_elapsed_s = 0.0
        self.start_sim_time_s = 0.0
        self.start_sim_step = 0
        self.best_servo_error_deg = float("inf")
        self.last_servo_improvement_elapsed_s = 0.0
        self.last_servo_error_deg: float | None = None
        self.last_servo_change_elapsed_s = 0.0
        self.last_servo_worsening_elapsed_s = 0.0
        self.contact_within_tolerance_since_s: float | None = None
        self.contact_error_history: list[tuple[float, float]] = []
        self.contact_stable_error_history: list[tuple[float, float]] = []
        self.wheel_stop_ack: dict[str, Any] = {}
        self.observation_count = 0
        self.last_feedback: SegmentFeedback | None = None
        self.last_decision: SegmentDecision | None = None

    def start(
        self,
        spec: SegmentCompletionSpec,
        *,
        start_elapsed_s: float,
        start_sim_time_s: float,
        start_sim_step: int,
        servo_duration_s_override: float | None = None,
    ) -> None:
        if self.spec is not None and (
            self.last_decision is None
            or self.last_decision.kind
            not in {SegmentDecisionKind.COMPLETE, SegmentDecisionKind.FAIL}
        ):
            raise RuntimeError("a completion-aware segment is already active")
        duration = (
            spec.servo_duration_s
            if servo_duration_s_override is None
            else _finite_nonnegative(
                servo_duration_s_override,
                label="servo_duration_s_override",
            )
        )
        start_elapsed = _finite_nonnegative(
            start_elapsed_s, label="start_elapsed_s"
        )
        start_sim_time = _finite_nonnegative(
            start_sim_time_s, label="start_sim_time_s"
        )
        if type(start_sim_step) is not int or start_sim_step < 0:
            raise ValueError("start_sim_step must be a non-negative int")
        self.reset()
        self.spec = replace(spec, servo_duration_s=duration)
        self.start_elapsed_s = start_elapsed
        self.start_sim_time_s = start_sim_time
        self.start_sim_step = start_sim_step
        self.last_servo_improvement_elapsed_s = start_elapsed
        self.last_servo_change_elapsed_s = start_elapsed
        self.last_servo_worsening_elapsed_s = start_elapsed

    def acknowledge_wheel_stop(
        self,
        *,
        applied_sim_step: int,
        first_physics_step: int,
        batch_id: str,
    ) -> None:
        if self.spec is None or self.last_decision is None:
            raise RuntimeError("no active segment can acknowledge a wheel stop")
        if not self.last_decision.wheel_stop_due or self.wheel_stop_ack:
            raise RuntimeError("wheel stop is not due or was already acknowledged")
        if (
            type(applied_sim_step) is not int
            or type(first_physics_step) is not int
            or applied_sim_step != self.last_decision.sim_step
            or first_physics_step != applied_sim_step + 1
            or not isinstance(batch_id, str)
            or not batch_id
        ):
            raise ValueError("wheel-stop acknowledgement is not exact N+1")
        self.wheel_stop_ack = {
            "applied_sim_step": applied_sim_step,
            "first_physics_step": first_physics_step,
            "batch_id": batch_id,
        }

    @staticmethod
    def _parse_exact_measurements(
        raw: Mapping[str, Any],
        targets: set[str],
        *,
        label: str,
    ) -> tuple[dict[str, float], str]:
        if set(raw) != targets:
            missing = sorted(targets - set(raw))
            extra = sorted(set(raw) - targets)
            return {}, f"{label} keys are not exact; missing={missing} extra={extra}"
        parsed: dict[str, float] = {}
        for name in sorted(targets):
            if type(raw[name]) not in (int, float):
                return {}, f"{label}.{name} is not finite numeric data"
            try:
                value = float(raw[name])
            except (TypeError, ValueError):
                return {}, f"{label}.{name} is not finite numeric data"
            if not math.isfinite(value):
                return {}, f"{label}.{name} is not finite numeric data"
            parsed[name] = value
        return parsed, ""

    def observe(self, feedback: SegmentFeedback) -> SegmentDecision:
        spec = self.spec
        if spec is None:
            raise RuntimeError("no completion-aware segment is active")
        if self.last_decision is not None and self.last_decision.kind in {
            SegmentDecisionKind.COMPLETE,
            SegmentDecisionKind.FAIL,
        }:
            raise RuntimeError("a terminal segment cannot be observed again")
        if feedback.elapsed_s + 1.0e-9 < self.start_elapsed_s:
            raise ValueError("feedback elapsed_s precedes segment start")
        if feedback.sim_time_s + 1.0e-9 < self.start_sim_time_s:
            raise ValueError("feedback sim_time_s precedes segment start")
        if self.last_feedback is None and feedback.sim_step <= self.start_sim_step:
            raise ValueError("first segment feedback must be after start_sim_step")
        if self.last_feedback is not None:
            if feedback.sim_step <= self.last_feedback.sim_step:
                raise ValueError("segment feedback sim_step must strictly increase")
            if (
                feedback.elapsed_s + 1.0e-9 < self.last_feedback.elapsed_s
                or feedback.sim_time_s + 1.0e-9 < self.last_feedback.sim_time_s
            ):
                raise ValueError("segment feedback regressed")

        segment_elapsed = max(0.0, feedback.elapsed_s - self.start_elapsed_s)
        servo_planned_done = (
            segment_elapsed + 1.0e-9 >= spec.servo_duration_s
        )
        target_keys = set(spec.servo_targets_deg)
        errors: dict[str, float] = {}
        velocities: dict[str, float] = {}
        measurement_error = ""
        if not target_keys and (
            feedback.servo_errors_deg or feedback.servo_velocity_deg_s
        ):
            measurement_error = (
                "servo measurement maps must be exactly empty without targets"
            )
        elif target_keys and servo_planned_done:
            errors, measurement_error = self._parse_exact_measurements(
                feedback.servo_errors_deg,
                target_keys,
                label="servo_errors_deg",
            )
            if not measurement_error:
                velocities, measurement_error = self._parse_exact_measurements(
                    feedback.servo_velocity_deg_s,
                    target_keys,
                    label="servo_velocity_deg_s",
                )

        max_error = max([abs(value) for value in errors.values()] or [0.0])
        reference_done = bool(
            not target_keys
            or (
                servo_planned_done
                and not measurement_error
                and max_error <= spec.servo_tolerance_deg
            )
        )
        previous_error = self.last_servo_error_deg
        if previous_error is None or abs(max_error - previous_error) > self.servo_improvement_epsilon_deg:
            self.last_servo_change_elapsed_s = feedback.elapsed_s
        if previous_error is not None and max_error > previous_error + self.servo_improvement_epsilon_deg:
            self.last_servo_worsening_elapsed_s = feedback.elapsed_s
        self.last_servo_error_deg = max_error

        contact_candidate = bool(
            reference_done
            and max_error > self.servo_position_tolerance_deg
            and max_error <= spec.servo_tolerance_deg
            and spec.recorded_servo_residual_deg
        )
        if contact_candidate:
            if self.contact_within_tolerance_since_s is None:
                self.contact_within_tolerance_since_s = feedback.elapsed_s
        else:
            self.contact_within_tolerance_since_s = None
        if servo_planned_done and spec.recorded_servo_residual_deg and not measurement_error:
            self.contact_error_history.append((feedback.elapsed_s, max_error))
            self.contact_stable_error_history.append((feedback.elapsed_s, max_error))
            divergence_start = feedback.elapsed_s - self.actuator_divergence_window_s
            while len(self.contact_error_history) > 1 and self.contact_error_history[1][0] <= divergence_start:
                self.contact_error_history.pop(0)
            stable_start = feedback.elapsed_s - self.contact_residual_stable_s
            while len(self.contact_stable_error_history) > 1 and self.contact_stable_error_history[1][0] <= stable_start:
                self.contact_stable_error_history.pop(0)

        contact_extension = max(0.0, segment_elapsed - spec.servo_duration_s)
        stable_errors = [value for _at, value in self.contact_stable_error_history]
        stable_duration = (
            self.contact_stable_error_history[-1][0]
            - self.contact_stable_error_history[0][0]
            if len(self.contact_stable_error_history) >= 2
            else 0.0
        )
        stable_ready = bool(
            len(self.contact_stable_error_history) >= 2
            and stable_duration >= self.contact_residual_stable_s * 0.90
        )
        divergence_errors = [value for _at, value in self.contact_error_history]
        divergence_duration = (
            self.contact_error_history[-1][0] - self.contact_error_history[0][0]
            if len(self.contact_error_history) >= 2
            else 0.0
        )
        divergence_min = min(divergence_errors or [max_error])
        divergence_max = max(divergence_errors or [max_error])
        divergence_slope = (
            (divergence_errors[-1] - divergence_errors[0]) / divergence_duration
            if divergence_duration > 1.0e-9
            else 0.0
        )
        contact_cap = min(
            self.contact_residual_hard_cap_deg,
            spec.servo_tolerance_deg + 0.5,
        )
        contact_min = min(stable_errors or [max_error])
        contact_max = max(stable_errors or [max_error])
        contact_slope = (
            (stable_errors[-1] - stable_errors[0]) / stable_duration
            if stable_duration > 1.0e-9
            else 0.0
        )
        contact_stable = bool(
            contact_candidate
            and stable_ready
            and max(stable_errors or [float("inf")]) <= contact_cap
            and stable_errors[-1] <= stable_errors[0] + 0.25
        )
        velocities_near_zero = bool(
            target_keys
            and set(velocities) == target_keys
            and all(
                abs(value) <= self.servo_completion_velocity_deg_s
                for value in velocities.values()
            )
        )
        servo_done = reference_done
        contact_grace_done = False
        if contact_candidate:
            contact_grace_done = bool(
                contact_extension >= self.contact_residual_grace_s
                and max_error
                <= min(
                    spec.servo_tolerance_deg,
                    self.contact_residual_hard_cap_deg,
                )
            )
            servo_done = contact_grace_done

        wheel_done = (
            segment_elapsed + 1.0e-9 >= spec.wheel_active_duration_s
        )
        hold_done = segment_elapsed + 1.0e-9 >= spec.explicit_hold_s
        segment_done = servo_done and wheel_done and hold_done
        wheel_stop_due = bool(
            wheel_done
            and spec.wheel_active_duration_s > 0.0
            and not segment_done
            and not self.wheel_stop_ack
        )
        stalled_for = max(
            0.0,
            feedback.elapsed_s - self.last_servo_improvement_elapsed_s,
        )
        worst_joint = (
            max(errors, key=lambda name: abs(errors[name])) if errors else "unknown"
        )
        failure_reason = ""
        failure_code = ""
        phase = "waiting"

        if measurement_error:
            failure_reason = "invalid_joint_state"
            failure_code = measurement_error
        elif not segment_done and servo_planned_done and target_keys and not servo_done:
            within_recorded_tolerance = bool(
                max_error > self.servo_position_tolerance_deg
                and max_error <= spec.servo_tolerance_deg
                and spec.recorded_servo_residual_deg
            )
            if within_recorded_tolerance:
                phase = "contact_residual_grace"
            if not reference_done:
                if max_error + self.servo_improvement_epsilon_deg < self.best_servo_error_deg:
                    self.best_servo_error_deg = max_error
                    self.last_servo_improvement_elapsed_s = feedback.elapsed_s
                    stalled_for = 0.0
                worst_velocity = velocities.get(worst_joint)
                worst_error = errors.get(worst_joint)
                low_response = bool(
                    worst_velocity is None or abs(worst_velocity) <= 5.0
                )
                not_recovering = bool(
                    worst_velocity is None
                    or abs(worst_velocity) <= self.servo_completion_velocity_deg_s
                    or (
                        worst_error is not None
                        and worst_error * worst_velocity > 0.0
                    )
                )
                worsening = bool(
                    max_error > self.contact_residual_hard_cap_deg
                    and divergence_duration
                    >= self.actuator_divergence_window_s * 0.90
                    and divergence_errors[-1] > divergence_errors[0] + 0.25
                    and divergence_errors[-1] >= divergence_max - 0.25
                    and divergence_slope > 1.0
                    and low_response
                    and not_recovering
                )
                if worsening:
                    failure_reason = "actuator_unstable"
                    failure_code = "worsening_outside_hard_tolerance"
                elif contact_extension >= self.servo_tracking_hard_liveness_s:
                    failure_reason = (
                        "actuator_limit"
                        if velocities_near_zero
                        else "actuator_unstable"
                    )
                    failure_code = (
                        "hard_liveness_near_zero"
                        if velocities_near_zero
                        else "hard_liveness_in_motion"
                    )
                elif (
                    contact_extension > 0.0
                    and stalled_for >= self.servo_stall_window_s
                    and velocities_near_zero
                ):
                    failure_reason = "actuator_limit"
                    failure_code = "stalled_near_zero"
                elif contact_extension > 0.0 and not within_recorded_tolerance:
                    phase = "servo_completion_extension"

        if failure_reason:
            kind = SegmentDecisionKind.FAIL
        elif segment_done:
            kind = SegmentDecisionKind.COMPLETE
        elif wheel_stop_due:
            kind = SegmentDecisionKind.WHEEL_STOP_DUE
        else:
            kind = SegmentDecisionKind.WAIT
        decision = SegmentDecision(
            kind=kind,
            segment_index=spec.segment_index,
            source_step=spec.source_step,
            source_step_id=spec.source_step_id,
            sim_time_s=feedback.sim_time_s,
            sim_step=feedback.sim_step,
            segment_elapsed_s=segment_elapsed,
            servo_planned_done=servo_planned_done,
            reference_position_done=reference_done,
            servo_done=servo_done,
            servo_errors_deg=MappingProxyType(dict(errors)),
            servo_velocity_deg_s=MappingProxyType(dict(velocities)),
            max_servo_error_deg=max_error,
            servo_tolerance_deg=spec.servo_tolerance_deg,
            recorded_servo_residual_deg=MappingProxyType(
                dict(spec.recorded_servo_residual_deg)
            ),
            legacy_missing_endpoint=spec.legacy_missing_endpoint,
            contact_candidate=contact_candidate,
            contact_extension_s=contact_extension,
            contact_grace_done=contact_grace_done,
            contact_stable=contact_stable,
            contact_window_ready=stable_ready,
            contact_window_duration_s=stable_duration,
            contact_window_cap_deg=contact_cap,
            contact_window_min_deg=contact_min,
            contact_window_max_deg=contact_max,
            contact_window_slope_deg_s=contact_slope,
            divergence_window_duration_s=divergence_duration,
            divergence_window_min_deg=divergence_min,
            divergence_window_max_deg=divergence_max,
            divergence_window_slope_deg_s=divergence_slope,
            velocities_near_zero=velocities_near_zero,
            wheel_elapsed_s=segment_elapsed,
            wheel_duration_s=spec.wheel_active_duration_s,
            wheel_done=wheel_done,
            hold_elapsed_s=segment_elapsed,
            hold_duration_s=spec.explicit_hold_s,
            hold_done=hold_done,
            segment_done=segment_done,
            wheel_stop_due=wheel_stop_due,
            wheel_stop_acknowledged=bool(self.wheel_stop_ack),
            stalled_for_s=stalled_for,
            worst_joint=worst_joint,
            tracking_evidence=MappingProxyType(
                copy.deepcopy(dict(feedback.tracking_evidence or {}))
            ),
            phase=phase,
            failure_reason=failure_reason,
            failure_code=failure_code,
        )
        self.observation_count += 1
        self.last_feedback = feedback
        self.last_decision = decision
        return decision

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_spec": None if self.spec is None else self.spec.to_mapping(),
            "start_elapsed_s": self.start_elapsed_s,
            "start_sim_time_s": self.start_sim_time_s,
            "start_sim_step": self.start_sim_step,
            "wheel_stop_ack": copy.deepcopy(self.wheel_stop_ack),
            "observation_count": self.observation_count,
            "contact_error_history": [list(row) for row in self.contact_error_history],
            "contact_stable_error_history": [
                list(row) for row in self.contact_stable_error_history
            ],
            "last_decision": (
                None
                if self.last_decision is None
                else self.last_decision.to_mapping()
            ),
        }
