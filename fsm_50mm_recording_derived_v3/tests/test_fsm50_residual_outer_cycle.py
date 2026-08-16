from __future__ import annotations

import pytest

from fsm_50mm_recording_derived_v3.fsm50_residual_outer_cycle import (
    EXPECTED_RENDER_SUBSTEPS,
    Fsm50OuterCycleDriver,
    OuterCycleContractError,
    RawPhysicsStepResult,
)


def test_one_action_is_held_for_exactly_eight_physics_callbacks() -> None:
    calls = []

    def stepper(action, context):
        calls.append((action, context))
        return RawPhysicsStepResult(
            observation={"step": len(calls)},
            reward=0.25,
            terminated=False,
            truncated=False,
            info={},
        )

    driver = Fsm50OuterCycleDriver(stepper)
    result = driver.step([0.0] * 12)
    assert result.physics_steps == EXPECTED_RENDER_SUBSTEPS == 8
    assert result.reward == pytest.approx(2.0)
    assert [call[0] for call in calls] == [(0.0,) * 12] * 8
    assert [call[1].source_cursor_permit for call in calls] == [True] + [False] * 7
    assert [call[1].policy_update_permit for call in calls] == [True] + [False] * 7
    assert [call[1].physics_substep_index for call in calls] == list(range(8))
    assert result.info["fsm50_outer_cycle_complete"] is True


def test_terminal_is_latched_and_returned_without_stepping_a_reset_episode() -> None:
    calls = []

    def stepper(action, context):
        calls.append(context.physics_substep_index)
        terminal = context.physics_substep_index == 3
        return RawPhysicsStepResult(
            observation=context.physics_substep_index,
            reward=1.0,
            terminated=terminal,
            truncated=False,
            info={"terminal": terminal},
        )

    result = Fsm50OuterCycleDriver(stepper).step([0.0])
    assert calls == [0, 1, 2, 3]
    assert result.physics_steps == 4
    assert result.terminal_substep_index == 3
    assert result.terminated is True
    assert result.reward == 4.0
    assert result.info["fsm50_outer_cycle_complete"] is False


def test_cycle_index_advances_once_per_policy_action_and_reset_is_explicit() -> None:
    seen = []

    def stepper(action, context):
        seen.append(context.outer_cycle_index)
        return RawPhysicsStepResult(None, 0.0, False, False, {})

    driver = Fsm50OuterCycleDriver(stepper)
    driver.step([0.0])
    driver.step([0.0])
    assert seen == [0] * 8 + [1] * 8
    assert driver.outer_cycle_index == 2
    driver.reset()
    assert driver.outer_cycle_index == 0


@pytest.mark.parametrize("action", [[], [float("nan")], [float("inf")], [True]])
def test_action_validation_fails_closed(action) -> None:
    driver = Fsm50OuterCycleDriver(
        lambda action, context: RawPhysicsStepResult(None, 0.0, False, False, {})
    )
    with pytest.raises(OuterCycleContractError):
        driver.step(action)


def test_raw_result_schema_fails_closed() -> None:
    with pytest.raises(OuterCycleContractError, match="finite"):
        RawPhysicsStepResult(None, float("nan"), False, False, {})
    with pytest.raises(OuterCycleContractError, match="terminal flags"):
        RawPhysicsStepResult(None, 0.0, 0, False, {})
    driver = Fsm50OuterCycleDriver(lambda action, context: (None, 0.0, False, False, {}))
    with pytest.raises(OuterCycleContractError, match="RawPhysicsStepResult"):
        driver.step([0.0])
