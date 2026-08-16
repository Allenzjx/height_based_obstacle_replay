"""Exact-eight physics-step driver for the 15 Hz residual policy cadence.

The production macro runtime evaluates source/completion work once per render
cycle while continuing safety/readback work at 120 Hz.  This small Isaac-free
driver gives a new residual environment the same cadence without relying on a
DirectRLEnv decimation setting that would hide the intermediate safety steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


EXPECTED_PHYSICS_HZ = 120
EXPECTED_POLICY_HZ = 15
EXPECTED_RENDER_SUBSTEPS = EXPECTED_PHYSICS_HZ // EXPECTED_POLICY_HZ


class OuterCycleContractError(RuntimeError):
    """Raised when the raw physics environment violates the outer-cycle API."""


@dataclass(frozen=True)
class OuterCycleContext:
    outer_cycle_index: int
    physics_substep_index: int
    source_cursor_permit: bool
    policy_update_permit: bool

    def __post_init__(self) -> None:
        if type(self.outer_cycle_index) is not int or self.outer_cycle_index < 0:
            raise OuterCycleContractError("outer_cycle_index must be a non-negative int")
        if (
            type(self.physics_substep_index) is not int
            or not 0 <= self.physics_substep_index < EXPECTED_RENDER_SUBSTEPS
        ):
            raise OuterCycleContractError("physics_substep_index is outside the exact-eight cycle")
        if type(self.source_cursor_permit) is not bool or type(self.policy_update_permit) is not bool:
            raise OuterCycleContractError("outer-cycle permits must be exact bool values")
        expected = self.physics_substep_index == 0
        if self.source_cursor_permit != expected or self.policy_update_permit != expected:
            raise OuterCycleContractError("source/policy permit is allowed only at the outer boundary")


@dataclass(frozen=True)
class RawPhysicsStepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.reward, bool) or not math.isfinite(float(self.reward)):
            raise OuterCycleContractError("raw reward must be finite")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise OuterCycleContractError("raw terminal flags must be exact bool values")
        if not isinstance(self.info, Mapping):
            raise OuterCycleContractError("raw info must be a mapping")


@dataclass(frozen=True)
class OuterCycleResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    physics_steps: int
    terminal_substep_index: int | None


class RawPhysicsStepper(Protocol):
    def __call__(
        self,
        action: tuple[float, ...],
        context: OuterCycleContext,
    ) -> RawPhysicsStepResult: ...


def _finite_action(action: Sequence[float]) -> tuple[float, ...]:
    if isinstance(action, (str, bytes, bytearray)):
        raise OuterCycleContractError("policy action must be a numeric sequence")
    result: list[float] = []
    for index, value in enumerate(action):
        if isinstance(value, bool):
            raise OuterCycleContractError(f"policy action[{index}] must not be bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise OuterCycleContractError(f"policy action[{index}] must be numeric") from exc
        if not math.isfinite(numeric):
            raise OuterCycleContractError(f"policy action[{index}] must be finite")
        result.append(numeric)
    if not result:
        raise OuterCycleContractError("policy action must not be empty")
    return tuple(result)


class Fsm50OuterCycleDriver:
    """Hold one policy action across exactly eight visible physics callbacks."""

    def __init__(self, physics_stepper: RawPhysicsStepper):
        if not callable(physics_stepper):
            raise OuterCycleContractError("physics_stepper must be callable")
        self._physics_stepper = physics_stepper
        self._outer_cycle_index = 0

    @property
    def outer_cycle_index(self) -> int:
        return self._outer_cycle_index

    def reset(self) -> None:
        self._outer_cycle_index = 0

    def step(self, action: Sequence[float]) -> OuterCycleResult:
        held_action = _finite_action(action)
        reward_sum = 0.0
        last: RawPhysicsStepResult | None = None
        terminal_substep: int | None = None
        completed_steps = 0
        for substep in range(EXPECTED_RENDER_SUBSTEPS):
            context = OuterCycleContext(
                outer_cycle_index=self._outer_cycle_index,
                physics_substep_index=substep,
                source_cursor_permit=substep == 0,
                policy_update_permit=substep == 0,
            )
            result = self._physics_stepper(held_action, context)
            if not isinstance(result, RawPhysicsStepResult):
                raise OuterCycleContractError("physics_stepper must return RawPhysicsStepResult")
            last = result
            reward_sum += float(result.reward)
            completed_steps += 1
            if result.terminated or result.truncated:
                terminal_substep = substep
                break
        if last is None:  # pragma: no cover - the exact-eight constant makes this unreachable
            raise OuterCycleContractError("outer cycle delivered no physics callbacks")
        cycle = self._outer_cycle_index
        self._outer_cycle_index += 1
        info = dict(last.info)
        info.update(
            {
                "fsm50_outer_cycle_index": cycle,
                "fsm50_outer_cycle_physics_steps": completed_steps,
                "fsm50_outer_cycle_complete": completed_steps == EXPECTED_RENDER_SUBSTEPS,
                "fsm50_terminal_substep_index": terminal_substep,
            }
        )
        return OuterCycleResult(
            observation=last.observation,
            reward=reward_sum,
            terminated=last.terminated,
            truncated=last.truncated,
            info=info,
            physics_steps=completed_steps,
            terminal_substep_index=terminal_substep,
        )


__all__ = [
    "EXPECTED_PHYSICS_HZ",
    "EXPECTED_POLICY_HZ",
    "EXPECTED_RENDER_SUBSTEPS",
    "Fsm50OuterCycleDriver",
    "OuterCycleContext",
    "OuterCycleContractError",
    "OuterCycleResult",
    "RawPhysicsStepResult",
    "RawPhysicsStepper",
]
