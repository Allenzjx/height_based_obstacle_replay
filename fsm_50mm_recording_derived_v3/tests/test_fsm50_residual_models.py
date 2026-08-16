from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from fsm_50mm_recording_derived_v3.fsm50_residual_models import (
    Fsm50ResidualPolicy,
    Fsm50ResidualValue,
    RESIDUAL_ACTION_DIM,
    make_actor_space,
    make_observation_space,
)
from fsm_50mm_recording_derived_v3.fsm50_residual_observation import ACTOR_OBSERVATION_DIM


def test_actor_deterministic_mean_starts_at_exact_zero() -> None:
    observation_space = make_observation_space()
    action_space = make_actor_space()
    policy = Fsm50ResidualPolicy(observation_space, action_space, "cpu")
    observation = torch.linspace(-1.0, 1.0, ACTOR_OBSERVATION_DIM).repeat(4, 1)
    mean, extra = policy.compute({"observations": observation}, "policy")
    assert mean.shape == (4, RESIDUAL_ACTION_DIM)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert extra["log_std"].shape == mean.shape


def test_value_shape_and_finiteness() -> None:
    observation_space = make_observation_space()
    action_space = make_actor_space()
    value = Fsm50ResidualValue(observation_space, action_space, "cpu")
    result, extra = value.compute({"states": torch.zeros(3, ACTOR_OBSERVATION_DIM)}, "value")
    assert result.shape == (3, 1)
    assert torch.isfinite(result).all()
    assert extra == {}


def test_model_spaces_fail_closed() -> None:
    bad_action = gym.spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
    with pytest.raises(ValueError, match="12"):
        Fsm50ResidualPolicy(make_observation_space(), bad_action, "cpu")
    small_state = gym.spaces.Box(-1.0, 1.0, shape=(ACTOR_OBSERVATION_DIM - 1,), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot be smaller"):
        Fsm50ResidualValue(small_state, make_actor_space(), "cpu")


def test_nonfinite_inputs_are_rejected() -> None:
    policy = Fsm50ResidualPolicy(make_observation_space(), make_actor_space(), "cpu")
    observations = torch.zeros(1, ACTOR_OBSERVATION_DIM)
    observations[0, -1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        policy.compute({"observations": observations}, "policy")
