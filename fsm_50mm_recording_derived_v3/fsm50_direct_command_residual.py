"""Isaac-free bounded 12-D direct-command residual composition.

The recording-derived Macro FSM remains the nominal command authority.  This
module only composes a phase-contracted residual with one already-validated
nominal 8-servo + 4-wheel target map.  It deliberately contains no controller,
worker, simulator, or policy-runtime imports.

Servo residuals use command-space degrees.  Wheel residuals use rad/s.  Their
rate limits therefore use degrees/s and rad/s^2 respectively.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from command_model import (
    SERVO_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    command_limits_for_servo,
)


RESIDUAL_ACTION_SCHEMA = "fsm50.direct_command_residual_action.v1"
RESIDUAL_PHASE_CONTRACT_SCHEMA = "fsm50.direct_command_residual_contract.v1"
RESIDUAL_TRANSFORM_RESULT_SCHEMA = "fsm50.direct_command_residual_result.v1"

RESIDUAL_ACTION_NAMES = tuple(SERVO_JOINT_NAMES) + tuple(WHEEL_JOINT_NAMES)
RESIDUAL_ACTION_DIM = len(RESIDUAL_ACTION_NAMES)
ZERO_RESIDUAL_ACTION = (0.0,) * RESIDUAL_ACTION_DIM

_SERVO_COUNT = len(SERVO_JOINT_NAMES)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "action_schema",
        "source_version",
        "profile_strategy",
        "macro_state",
        "subphase",
        "action_names",
        "enabled_mask",
        "residual_min_command_units",
        "residual_max_command_units",
        "maximum_rate_command_units_per_s",
    }
)


class ResidualContractError(ValueError):
    """Raised when a residual contract or transform input is not exact/safe."""


class ResidualPolicy(Protocol):
    """Minimal deployment policy interface consumed by a future worker hook."""

    policy_id: str

    @property
    def policy_sha256(self) -> str: ...

    def act(self, observation: Mapping[str, Any]) -> Sequence[float]: ...


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 of one strict JSON mapping."""

    if not isinstance(value, Mapping):
        raise ResidualContractError("canonical digest input must be a mapping")
    try:
        payload = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResidualContractError(
            "canonical digest input must contain strict finite JSON data"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _required_text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ResidualContractError(f"{label} must be non-empty text")
    return value


def _finite_vector(
    value: Any,
    *,
    label: str,
    length: int = RESIDUAL_ACTION_DIM,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResidualContractError(f"{label} must be a numeric sequence")
    if len(value) != length:
        raise ResidualContractError(f"{label} must contain exactly {length} values")
    result: list[float] = []
    for index, raw in enumerate(value):
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ResidualContractError(f"{label}[{index}] must be finite numeric data")
        result.append(float(raw))
    return tuple(result)


def _validated_target_map(
    value: Any,
    *,
    names: Sequence[str],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ResidualContractError(
            f"{label} must contain exactly the canonical target names"
        )
    result: dict[str, float] = {}
    for name in names:
        raw = value[name]
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ResidualContractError(f"{label}.{name} must be finite numeric data")
        result[name] = float(raw)
    return result


def _validated_servo_safety_limits(
    value: Mapping[str, Sequence[float]] | None,
) -> dict[str, tuple[float, float]]:
    if value is not None and (
        not isinstance(value, Mapping) or set(value) != set(SERVO_JOINT_NAMES)
    ):
        raise ResidualContractError(
            "servo_safe_command_limits_deg must contain exactly the canonical servos"
        )
    result: dict[str, tuple[float, float]] = {}
    for name in SERVO_JOINT_NAMES:
        command_lower, command_upper = (
            float(item) for item in command_limits_for_servo(name)
        )
        raw_limits: Sequence[float] = (
            (command_lower, command_upper) if value is None else value[name]
        )
        if (
            isinstance(raw_limits, (str, bytes))
            or not isinstance(raw_limits, Sequence)
            or len(raw_limits) != 2
            or any(
                type(item) not in (int, float) or not math.isfinite(float(item))
                for item in raw_limits
            )
        ):
            raise ResidualContractError(
                f"servo_safe_command_limits_deg.{name} must be two finite values"
            )
        supplied_lower, supplied_upper = (float(item) for item in raw_limits)
        if supplied_lower > supplied_upper:
            raise ResidualContractError(
                f"servo_safe_command_limits_deg.{name} is reversed"
            )
        lower = max(command_lower, supplied_lower)
        upper = min(command_upper, supplied_upper)
        if lower > upper:
            raise ResidualContractError(
                f"servo_safe_command_limits_deg.{name} does not intersect command safety"
            )
        result[name] = (lower, upper)
    return result


def _validated_latched_servo_residuals(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ResidualContractError("latched_servo_residual_deg must be a mapping")
    unknown = sorted(set(value) - set(SERVO_JOINT_NAMES))
    if unknown:
        raise ResidualContractError(
            "latched_servo_residual_deg contains non-servo names: "
            + ", ".join(unknown)
        )
    result: dict[str, float] = {}
    for name in SERVO_JOINT_NAMES:
        if name not in value:
            continue
        raw = value[name]
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ResidualContractError(
                f"latched_servo_residual_deg.{name} must be finite numeric data"
            )
        result[name] = float(raw)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _positive_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


@dataclass(frozen=True)
class ResidualPhaseContract:
    """One fail-closed residual envelope for an exact strategy/phase context."""

    source_version: str
    profile_strategy: str
    macro_state: str
    subphase: str
    enabled_mask: tuple[bool, ...]
    residual_min_command_units: tuple[float, ...]
    residual_max_command_units: tuple[float, ...]
    maximum_rate_command_units_per_s: tuple[float, ...]
    action_names: tuple[str, ...] = RESIDUAL_ACTION_NAMES

    def __post_init__(self) -> None:
        for field_name in (
            "source_version",
            "profile_strategy",
            "macro_state",
            "subphase",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        action_names = tuple(self.action_names)
        if action_names != RESIDUAL_ACTION_NAMES:
            raise ResidualContractError(
                "action_names must equal the canonical 8-servo + 4-wheel order"
            )
        object.__setattr__(self, "action_names", action_names)

        raw_mask = self.enabled_mask
        if (
            isinstance(raw_mask, (str, bytes))
            or not isinstance(raw_mask, Sequence)
            or len(raw_mask) != RESIDUAL_ACTION_DIM
            or any(type(item) is not bool for item in raw_mask)
        ):
            raise ResidualContractError(
                f"enabled_mask must contain exactly {RESIDUAL_ACTION_DIM} bool values"
            )
        mask = tuple(raw_mask)
        lower = _finite_vector(
            self.residual_min_command_units,
            label="residual_min_command_units",
        )
        upper = _finite_vector(
            self.residual_max_command_units,
            label="residual_max_command_units",
        )
        rates = _finite_vector(
            self.maximum_rate_command_units_per_s,
            label="maximum_rate_command_units_per_s",
        )
        for index, name in enumerate(RESIDUAL_ACTION_NAMES):
            if lower[index] > 0.0 or upper[index] < 0.0 or lower[index] > upper[index]:
                raise ResidualContractError(
                    f"residual bounds for {name} must satisfy min <= 0 <= max"
                )
            if rates[index] < 0.0:
                raise ResidualContractError(
                    f"maximum residual rate for {name} must be non-negative"
                )
            if not mask[index] and (
                lower[index] != 0.0
                or upper[index] != 0.0
                or rates[index] != 0.0
            ):
                raise ResidualContractError(
                    f"disabled residual channel {name} must have zero bounds and rate"
                )
        object.__setattr__(self, "enabled_mask", mask)
        object.__setattr__(self, "residual_min_command_units", lower)
        object.__setattr__(self, "residual_max_command_units", upper)
        object.__setattr__(self, "maximum_rate_command_units_per_s", rates)

    def require_context(
        self,
        *,
        source_version: str,
        profile_strategy: str,
        macro_state: str,
        subphase: str,
    ) -> None:
        actual = (source_version, profile_strategy, macro_state, subphase)
        expected = (
            self.source_version,
            self.profile_strategy,
            self.macro_state,
            self.subphase,
        )
        if actual != expected:
            raise ResidualContractError(
                "residual phase contract context mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RESIDUAL_PHASE_CONTRACT_SCHEMA,
            "action_schema": RESIDUAL_ACTION_SCHEMA,
            "source_version": self.source_version,
            "profile_strategy": self.profile_strategy,
            "macro_state": self.macro_state,
            "subphase": self.subphase,
            "action_names": list(self.action_names),
            "enabled_mask": list(self.enabled_mask),
            "residual_min_command_units": list(
                self.residual_min_command_units
            ),
            "residual_max_command_units": list(
                self.residual_max_command_units
            ),
            "maximum_rate_command_units_per_s": list(
                self.maximum_rate_command_units_per_s
            ),
        }

    @property
    def sha256(self) -> str:
        return canonical_mapping_sha256(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidualPhaseContract":
        if not isinstance(value, Mapping) or set(value) != set(_CONTRACT_KEYS):
            raise ResidualContractError("residual phase contract schema keys are not exact")
        if value.get("schema_version") != RESIDUAL_PHASE_CONTRACT_SCHEMA:
            raise ResidualContractError("unsupported residual phase contract schema")
        if value.get("action_schema") != RESIDUAL_ACTION_SCHEMA:
            raise ResidualContractError("unsupported residual action schema")
        sequence_fields: dict[str, tuple[Any, ...]] = {}
        for field_name in (
            "action_names",
            "enabled_mask",
            "residual_min_command_units",
            "residual_max_command_units",
            "maximum_rate_command_units_per_s",
        ):
            raw = value.get(field_name)
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise ResidualContractError(
                    f"residual phase contract {field_name} must be a sequence"
                )
            sequence_fields[field_name] = tuple(raw)
        return cls(
            source_version=value.get("source_version"),
            profile_strategy=value.get("profile_strategy"),
            macro_state=value.get("macro_state"),
            subphase=value.get("subphase"),
            action_names=sequence_fields["action_names"],
            enabled_mask=sequence_fields["enabled_mask"],
            residual_min_command_units=sequence_fields[
                "residual_min_command_units"
            ],
            residual_max_command_units=sequence_fields[
                "residual_max_command_units"
            ],
            maximum_rate_command_units_per_s=sequence_fields[
                "maximum_rate_command_units_per_s"
            ],
        )


@dataclass(frozen=True)
class ResidualTransformInput:
    """Validated nominal command, policy action, and runtime safety envelope."""

    source_version: str
    profile_strategy: str
    macro_state: str
    subphase: str
    nominal_servo_targets_deg: Mapping[str, float]
    nominal_wheel_targets_rad_s: Mapping[str, float]
    normalized_action: tuple[float, ...]
    previous_applied_residual: tuple[float, ...]
    decision_dt_s: float
    maximum_wheel_speed_rad_s: float
    servo_safe_command_limits_deg: Mapping[str, Sequence[float]] | None = None
    latched_servo_residual_deg: Mapping[str, float] | None = None
    force_zero_residual: bool = False
    force_zero_wheels: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "source_version",
            "profile_strategy",
            "macro_state",
            "subphase",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        servos = _validated_target_map(
            self.nominal_servo_targets_deg,
            names=SERVO_JOINT_NAMES,
            label="nominal_servo_targets_deg",
        )
        wheels = _validated_target_map(
            self.nominal_wheel_targets_rad_s,
            names=WHEEL_JOINT_NAMES,
            label="nominal_wheel_targets_rad_s",
        )
        action = _finite_vector(self.normalized_action, label="normalized_action")
        previous = _finite_vector(
            self.previous_applied_residual,
            label="previous_applied_residual",
        )
        if (
            type(self.decision_dt_s) not in (int, float)
            or not math.isfinite(float(self.decision_dt_s))
            or float(self.decision_dt_s) <= 0.0
        ):
            raise ResidualContractError("decision_dt_s must be finite and positive")
        if (
            type(self.maximum_wheel_speed_rad_s) not in (int, float)
            or not math.isfinite(float(self.maximum_wheel_speed_rad_s))
            or float(self.maximum_wheel_speed_rad_s) <= 0.0
        ):
            raise ResidualContractError(
                "maximum_wheel_speed_rad_s must be finite and positive"
            )
        if type(self.force_zero_residual) is not bool:
            raise ResidualContractError("force_zero_residual must be an exact bool")
        if type(self.force_zero_wheels) is not bool:
            raise ResidualContractError("force_zero_wheels must be an exact bool")
        safety = _validated_servo_safety_limits(
            self.servo_safe_command_limits_deg
        )
        latched = _validated_latched_servo_residuals(
            self.latched_servo_residual_deg
        )
        for name, target in servos.items():
            lower, upper = safety[name]
            if not lower <= target <= upper:
                raise ResidualContractError(
                    f"nominal servo target is outside hard safety at {name}"
                )
        wheel_limit = float(self.maximum_wheel_speed_rad_s)
        for name, target in wheels.items():
            if abs(target) > wheel_limit:
                raise ResidualContractError(
                    f"nominal wheel target is outside hard safety at {name}"
                )
        if self.force_zero_wheels and any(target != 0.0 for target in wheels.values()):
            raise ResidualContractError(
                "force_zero_wheels requires an exact zero nominal wheel map"
            )

        object.__setattr__(self, "nominal_servo_targets_deg", servos)
        object.__setattr__(self, "nominal_wheel_targets_rad_s", wheels)
        object.__setattr__(self, "normalized_action", action)
        object.__setattr__(self, "previous_applied_residual", previous)
        object.__setattr__(self, "decision_dt_s", float(self.decision_dt_s))
        object.__setattr__(
            self, "maximum_wheel_speed_rad_s", wheel_limit
        )
        object.__setattr__(self, "servo_safe_command_limits_deg", safety)
        object.__setattr__(self, "latched_servo_residual_deg", latched)


@dataclass(frozen=True)
class ResidualTransformResult:
    """Deterministic evidence and the sole final 8+4 applied target map."""

    source_version: str
    profile_strategy: str
    macro_state: str
    subphase: str
    contract_sha256: str
    raw_normalized_action: tuple[float, ...]
    clipped_normalized_action: tuple[float, ...]
    phase_mask: tuple[bool, ...]
    residual_min_command_units: tuple[float, ...]
    residual_max_command_units: tuple[float, ...]
    previous_applied_residual: tuple[float, ...]
    requested_residual: tuple[float, ...]
    rate_limited_residual: tuple[float, ...]
    applied_residual: tuple[float, ...]
    nominal_servo_targets_deg: Mapping[str, float]
    nominal_wheel_targets_rad_s: Mapping[str, float]
    applied_servo_targets_deg: Mapping[str, float]
    applied_wheel_targets_rad_s: Mapping[str, float]
    clip_reasons_by_action: tuple[tuple[str, ...], ...]
    zero_identity: bool

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RESIDUAL_TRANSFORM_RESULT_SCHEMA,
            "action_schema": RESIDUAL_ACTION_SCHEMA,
            "action_names": list(RESIDUAL_ACTION_NAMES),
            "source_version": self.source_version,
            "profile_strategy": self.profile_strategy,
            "macro_state": self.macro_state,
            "subphase": self.subphase,
            "contract_sha256": self.contract_sha256,
            "raw_normalized_action": list(self.raw_normalized_action),
            "clipped_normalized_action": list(self.clipped_normalized_action),
            "phase_mask": list(self.phase_mask),
            "residual_min_command_units": list(
                self.residual_min_command_units
            ),
            "residual_max_command_units": list(
                self.residual_max_command_units
            ),
            "previous_applied_residual": list(self.previous_applied_residual),
            "requested_residual": list(self.requested_residual),
            "rate_limited_residual": list(self.rate_limited_residual),
            "applied_residual": list(self.applied_residual),
            "nominal_servo_targets_deg": {
                name: float(self.nominal_servo_targets_deg[name])
                for name in SERVO_JOINT_NAMES
            },
            "nominal_wheel_targets_rad_s": {
                name: float(self.nominal_wheel_targets_rad_s[name])
                for name in WHEEL_JOINT_NAMES
            },
            "applied_servo_targets_deg": {
                name: float(self.applied_servo_targets_deg[name])
                for name in SERVO_JOINT_NAMES
            },
            "applied_wheel_targets_rad_s": {
                name: float(self.applied_wheel_targets_rad_s[name])
                for name in WHEEL_JOINT_NAMES
            },
            "clip_reasons_by_action": {
                name: list(self.clip_reasons_by_action[index])
                for index, name in enumerate(RESIDUAL_ACTION_NAMES)
            },
            "zero_identity": self.zero_identity,
        }

    @property
    def sha256(self) -> str:
        return canonical_mapping_sha256(self.to_mapping())


@dataclass(frozen=True)
class ZeroResidualPolicy:
    """A deployment policy whose output proves the exact identity path."""

    policy_id: str = "fsm50.zero_direct_command_residual.v1"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fsm50.zero_direct_command_residual_policy.v1",
            "policy_id": self.policy_id,
            "action_schema": RESIDUAL_ACTION_SCHEMA,
            "action_names": list(RESIDUAL_ACTION_NAMES),
            "action_dim": RESIDUAL_ACTION_DIM,
        }

    @property
    def policy_sha256(self) -> str:
        return canonical_mapping_sha256(self.to_mapping())

    def act(self, observation: Mapping[str, Any]) -> tuple[float, ...]:
        if not isinstance(observation, Mapping):
            raise ResidualContractError("policy observation must be a mapping")
        return ZERO_RESIDUAL_ACTION


def _build_result(
    *,
    transform_input: ResidualTransformInput,
    contract: ResidualPhaseContract,
    raw_action: tuple[float, ...],
    clipped_action: tuple[float, ...],
    requested: tuple[float, ...],
    rate_limited: tuple[float, ...],
    applied_residual: tuple[float, ...],
    applied_servos: Mapping[str, float],
    applied_wheels: Mapping[str, float],
    clip_reasons: Sequence[Sequence[str]],
    zero_identity: bool,
) -> ResidualTransformResult:
    return ResidualTransformResult(
        source_version=transform_input.source_version,
        profile_strategy=transform_input.profile_strategy,
        macro_state=transform_input.macro_state,
        subphase=transform_input.subphase,
        contract_sha256=contract.sha256,
        raw_normalized_action=raw_action,
        clipped_normalized_action=clipped_action,
        phase_mask=contract.enabled_mask,
        residual_min_command_units=contract.residual_min_command_units,
        residual_max_command_units=contract.residual_max_command_units,
        previous_applied_residual=transform_input.previous_applied_residual,
        requested_residual=requested,
        rate_limited_residual=rate_limited,
        applied_residual=applied_residual,
        nominal_servo_targets_deg=dict(
            transform_input.nominal_servo_targets_deg
        ),
        nominal_wheel_targets_rad_s=dict(
            transform_input.nominal_wheel_targets_rad_s
        ),
        applied_servo_targets_deg=dict(applied_servos),
        applied_wheel_targets_rad_s=dict(applied_wheels),
        clip_reasons_by_action=tuple(tuple(items) for items in clip_reasons),
        zero_identity=bool(zero_identity),
    )


def compose_direct_command_residual(
    transform_input: ResidualTransformInput,
    contract: ResidualPhaseContract,
) -> ResidualTransformResult:
    """Compose one bounded residual without changing nominal FSM semantics.

    An all-exact-zero policy action with no active nonzero completion latch has
    a dedicated identity branch: the returned applied maps are copies of the
    validated nominal maps, all applied residual entries are positive zero,
    and no clipping reason is emitted.  A nonzero sparse completion latch must
    instead pass through the normal composer so its effective endpoint remains
    frozen while every unlatched channel converges normally toward zero.
    """

    if not isinstance(transform_input, ResidualTransformInput):
        raise ResidualContractError("transform_input must be ResidualTransformInput")
    if not isinstance(contract, ResidualPhaseContract):
        raise ResidualContractError("contract must be ResidualPhaseContract")
    contract.require_context(
        source_version=transform_input.source_version,
        profile_strategy=transform_input.profile_strategy,
        macro_state=transform_input.macro_state,
        subphase=transform_input.subphase,
    )

    raw_action = transform_input.normalized_action
    latched = dict(transform_input.latched_servo_residual_deg or {})
    has_nonzero_latch = any(value != 0.0 for value in latched.values())
    empty_reasons: tuple[tuple[str, ...], ...] = tuple(
        () for _ in RESIDUAL_ACTION_NAMES
    )
    if all(value == 0.0 for value in raw_action) and not has_nonzero_latch:
        # Do not pass through multiply/add/clamp: this branch is the exact R0
        # command identity, even if a prior nonzero residual existed.  An
        # active nonzero sparse latch intentionally excludes this branch.
        return _build_result(
            transform_input=transform_input,
            contract=contract,
            raw_action=raw_action,
            clipped_action=ZERO_RESIDUAL_ACTION,
            requested=ZERO_RESIDUAL_ACTION,
            rate_limited=ZERO_RESIDUAL_ACTION,
            applied_residual=ZERO_RESIDUAL_ACTION,
            applied_servos=transform_input.nominal_servo_targets_deg,
            applied_wheels=transform_input.nominal_wheel_targets_rad_s,
            clip_reasons=empty_reasons,
            zero_identity=True,
        )

    reasons: list[list[str]] = [[] for _ in RESIDUAL_ACTION_NAMES]
    clipped_action_values: list[float] = []
    requested_values: list[float] = []
    rate_limited_values: list[float] = []
    applied_values: list[float] = []
    for index, name in enumerate(RESIDUAL_ACTION_NAMES):
        raw = raw_action[index]
        clipped_action = _clamp(raw, -1.0, 1.0)
        if clipped_action != raw:
            reasons[index].append("NORMALIZED_ACTION_CLIP")
        clipped_action_values.append(_positive_zero(clipped_action))

        enabled = contract.enabled_mask[index]
        lower = contract.residual_min_command_units[index]
        upper = contract.residual_max_command_units[index]
        if not enabled:
            requested = 0.0
        elif clipped_action < 0.0:
            requested = (-clipped_action) * lower
        else:
            requested = clipped_action * upper
        requested = _positive_zero(requested)
        requested_values.append(requested)

        previous = transform_input.previous_applied_residual[index]
        maximum_delta = (
            contract.maximum_rate_command_units_per_s[index]
            * transform_input.decision_dt_s
        )
        rate_limited = _clamp(
            requested,
            previous - maximum_delta,
            previous + maximum_delta,
        )
        if rate_limited != requested:
            reasons[index].append("RESIDUAL_RATE_LIMIT")
        rate_limited = _positive_zero(rate_limited)
        rate_limited_values.append(rate_limited)

        if transform_input.force_zero_residual:
            candidate = 0.0
            if raw != 0.0 or previous != 0.0:
                reasons[index].append("FORCE_ZERO_RESIDUAL")
        elif index < _SERVO_COUNT and name in latched:
            candidate = latched[name]
            reasons[index].append("ACTIVE_COMPLETION_LATCH")
        elif not enabled:
            candidate = 0.0
            if raw != 0.0 or previous != 0.0:
                reasons[index].append("PHASE_MASK_HARD_ZERO")
        else:
            candidate = _clamp(rate_limited, lower, upper)
            if candidate != rate_limited:
                reasons[index].append("RESIDUAL_BOUND_CLIP")

        if index < _SERVO_COUNT:
            nominal = transform_input.nominal_servo_targets_deg[name]
            safe_lower, safe_upper = transform_input.servo_safe_command_limits_deg[  # type: ignore[index]
                name
            ]
            safe_residual_lower = safe_lower - nominal
            safe_residual_upper = safe_upper - nominal
            safe_candidate = _clamp(
                candidate, safe_residual_lower, safe_residual_upper
            )
            if safe_candidate != candidate:
                reasons[index].append("SERVO_HARD_SAFETY_CLIP")
            candidate = safe_candidate
        else:
            nominal = transform_input.nominal_wheel_targets_rad_s[name]
            if transform_input.force_zero_wheels or nominal == 0.0:
                if candidate != 0.0 or raw != 0.0 or previous != 0.0:
                    reasons[index].append(
                        "FORCE_ZERO_WHEEL"
                        if transform_input.force_zero_wheels
                        else "NOMINAL_ZERO_WHEEL_HARD_ZERO"
                    )
                candidate = 0.0
            else:
                maximum = transform_input.maximum_wheel_speed_rad_s
                speed_lower = -maximum - nominal
                speed_upper = maximum - nominal
                speed_candidate = _clamp(candidate, speed_lower, speed_upper)
                if speed_candidate != candidate:
                    reasons[index].append("WHEEL_HARD_SPEED_CLIP")
                candidate = speed_candidate

                if nominal > 0.0:
                    signed_candidate = max(candidate, -nominal)
                else:
                    signed_candidate = min(candidate, -nominal)
                if signed_candidate != candidate:
                    reasons[index].append("WHEEL_SIGN_PRESERVATION")
                candidate = signed_candidate
        applied_values.append(_positive_zero(candidate))

    clipped_action_tuple = tuple(clipped_action_values)
    requested_tuple = tuple(requested_values)
    rate_limited_tuple = tuple(rate_limited_values)
    applied_tuple = tuple(applied_values)

    applied_servos: dict[str, float] = {}
    for index, name in enumerate(SERVO_JOINT_NAMES):
        nominal = transform_input.nominal_servo_targets_deg[name]
        lower, upper = transform_input.servo_safe_command_limits_deg[name]  # type: ignore[index]
        target = _clamp(nominal + applied_tuple[index], lower, upper)
        applied_servos[name] = _positive_zero(target)

    applied_wheels: dict[str, float] = {}
    for wheel_offset, name in enumerate(WHEEL_JOINT_NAMES):
        index = _SERVO_COUNT + wheel_offset
        nominal = transform_input.nominal_wheel_targets_rad_s[name]
        target = nominal + applied_tuple[index]
        target = _clamp(
            target,
            -transform_input.maximum_wheel_speed_rad_s,
            transform_input.maximum_wheel_speed_rad_s,
        )
        if nominal > 0.0:
            target = max(0.0, target)
        elif nominal < 0.0:
            target = min(0.0, target)
        else:
            target = 0.0
        applied_wheels[name] = _positive_zero(target)

    # Rebuild the final residual from the actual safe target maps so the
    # evidence always states the exact applied-minus-nominal command delta.
    final_applied = tuple(
        _positive_zero(
            applied_servos[name]
            - transform_input.nominal_servo_targets_deg[name]
        )
        for name in SERVO_JOINT_NAMES
    ) + tuple(
        _positive_zero(
            applied_wheels[name]
            - transform_input.nominal_wheel_targets_rad_s[name]
        )
        for name in WHEEL_JOINT_NAMES
    )

    return _build_result(
        transform_input=transform_input,
        contract=contract,
        raw_action=raw_action,
        clipped_action=clipped_action_tuple,
        requested=requested_tuple,
        rate_limited=rate_limited_tuple,
        applied_residual=final_applied,
        applied_servos=applied_servos,
        applied_wheels=applied_wheels,
        clip_reasons=reasons,
        zero_identity=False,
    )


__all__ = [
    "RESIDUAL_ACTION_DIM",
    "RESIDUAL_ACTION_NAMES",
    "RESIDUAL_ACTION_SCHEMA",
    "RESIDUAL_PHASE_CONTRACT_SCHEMA",
    "RESIDUAL_TRANSFORM_RESULT_SCHEMA",
    "ZERO_RESIDUAL_ACTION",
    "ResidualContractError",
    "ResidualPhaseContract",
    "ResidualPolicy",
    "ResidualTransformInput",
    "ResidualTransformResult",
    "ZeroResidualPolicy",
    "canonical_mapping_sha256",
    "compose_direct_command_residual",
]
