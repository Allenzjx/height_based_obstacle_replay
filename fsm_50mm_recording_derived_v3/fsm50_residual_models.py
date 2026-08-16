"""Small SKRL actor/critic models for bounded 50 mm residual control.

Only generic network structure is adapted from the retired PPO project.  No
legacy phase logic, weights, checkpoint, normalization statistics, or action
authority is imported.  The actor's mean head is initialized to exact zero so
deterministic evaluation begins at the nominal Macro command.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .fsm50_residual_observation import ACTOR_OBSERVATION_DIM


RESIDUAL_ACTION_DIM = 12
ACTOR_HIDDEN = (256, 256, 128)
CRITIC_HIDDEN = (256, 256, 128)
INITIAL_LOG_STD = -4.0
MIN_LOG_STD = -5.0
MAX_LOG_STD = -2.5


def flat_dim(space: gym.Space) -> int:
    return int(gym.spaces.flatdim(space))


def make_actor_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(RESIDUAL_ACTION_DIM,),
        dtype=np.float32,
    )


def make_observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(ACTOR_OBSERVATION_DIM,),
        dtype=np.float32,
    )


class Fsm50ResidualPolicy(GaussianMixin, Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, device: torch.device | str):
        if flat_dim(action_space) != RESIDUAL_ACTION_DIM:
            raise ValueError(f"action space must have {RESIDUAL_ACTION_DIM} elements")
        if flat_dim(observation_space) != ACTOR_OBSERVATION_DIM:
            raise ValueError(f"observation space must have {ACTOR_OBSERVATION_DIM} elements")
        normalized_action_space = make_actor_space()
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=normalized_action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=MIN_LOG_STD,
            max_log_std=MAX_LOG_STD,
            reduction="sum",
            role="policy",
        )
        self.encoder = nn.Sequential(
            nn.Linear(ACTOR_OBSERVATION_DIM, ACTOR_HIDDEN[0]),
            nn.ELU(),
            nn.Linear(ACTOR_HIDDEN[0], ACTOR_HIDDEN[1]),
            nn.ELU(),
            nn.Linear(ACTOR_HIDDEN[1], ACTOR_HIDDEN[2]),
            nn.ELU(),
        )
        self.mean_head = nn.Linear(ACTOR_HIDDEN[-1], RESIDUAL_ACTION_DIM)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std_parameter = nn.Parameter(torch.full((RESIDUAL_ACTION_DIM,), INITIAL_LOG_STD))

    def compute(self, inputs, role):
        observations = inputs["observations"].reshape(-1, ACTOR_OBSERVATION_DIM).to(self.device)
        if not torch.isfinite(observations).all():
            raise ValueError("policy observation contains non-finite values")
        mean = torch.tanh(self.mean_head(self.encoder(torch.clamp(observations, -20.0, 20.0))))
        return mean, {"log_std": self.log_std_parameter.expand_as(mean)}


class Fsm50ResidualValue(DeterministicMixin, Model):
    def __init__(self, state_space: gym.Space, action_space: gym.Space, device: torch.device | str):
        if flat_dim(state_space) < ACTOR_OBSERVATION_DIM:
            raise ValueError("critic state space cannot be smaller than the deployable actor observation")
        Model.__init__(self, observation_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        state_dim = flat_dim(state_space)
        self.value = nn.Sequential(
            nn.Linear(state_dim, CRITIC_HIDDEN[0]),
            nn.ELU(),
            nn.Linear(CRITIC_HIDDEN[0], CRITIC_HIDDEN[1]),
            nn.ELU(),
            nn.Linear(CRITIC_HIDDEN[1], CRITIC_HIDDEN[2]),
            nn.ELU(),
            nn.Linear(CRITIC_HIDDEN[-1], 1),
        )

    def compute(self, inputs, role):
        states = inputs["states"].reshape(inputs["states"].shape[0], -1).to(self.device)
        if not torch.isfinite(states).all():
            raise ValueError("critic state contains non-finite values")
        return self.value(torch.clamp(states, -20.0, 20.0)), {}


__all__ = [
    "ACTOR_HIDDEN",
    "CRITIC_HIDDEN",
    "Fsm50ResidualPolicy",
    "Fsm50ResidualValue",
    "INITIAL_LOG_STD",
    "MAX_LOG_STD",
    "MIN_LOG_STD",
    "RESIDUAL_ACTION_DIM",
    "flat_dim",
    "make_actor_space",
    "make_observation_space",
]
