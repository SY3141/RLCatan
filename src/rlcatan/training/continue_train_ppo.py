from __future__ import annotations

import os
import random
from typing import Iterable, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron.gym.envs.catanatron_env import CatanatronEnv

from catanatron.models.enums import ActionType
from catanatron.gym.action_type_filtering import (
    COMPLEX_DEV_CARD_ACTION_TYPES,
    PLAYER_TRADING_ACTION_TYPES,
)
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper
import torch


def heuristic_mask(env: RLCatanEnvWrapper, valid_indices: list[int]) -> list[int]:
    """
    Apply heuristic filters on top of already-filtered valid_indices.
    """
    # Access all important game state data via env.env
    base_env = cast(CatanatronEnv, env.env)
    game = base_env.game
    state = game.state

    # Call heuristic functions to modify valid_indices
    # TODO: Check various heuristics to drop some indices from valid_indices

    # TODO: Return the modified valid_indices, right now nothing is removed
    return valid_indices


def make_env(seed: int | None = None) -> gym.Env:
    """
    Build a single training environment:
      - CatanatronEnv (1v1 vs. RandomPlayer)
      - RLCatanEnvWrapper: filters out some ActionTypes
      - ActionMasker: gives MaskablePPO a valid-action mask
    """
    base_env = CatanatronEnv(config={"opponent_type": "RandomPlayer"})

    if seed is not None:
        base_env.reset(seed=seed)

    # Excluding complex dev card actions and player trading actions for v1
    excluded_type_groups: Iterable[Iterable[ActionType]] = [
        COMPLEX_DEV_CARD_ACTION_TYPES,
        PLAYER_TRADING_ACTION_TYPES,
    ]

    # First wrap: filter out unwanted ActionTypes
    wrapped_env = RLCatanEnvWrapper(base_env, excluded_type_groups=excluded_type_groups)

    # Masking function for SB3 MaskablePPO, I adapted it to include heuristic filtering
    def mask_fn(env: gym.Env) -> np.ndarray:
        """
        Mask function for SB3 ActionMasker.

        `env` is the wrapped environment (RLCatanEnvWrapper), so this function:
          - calls env.get_valid_actions() to get already-filtered indices
          - casts env.action_space to Discrete to get `n` for mask length
        """
        # Cast to RLCatanEnvWrapper to access get_valid_actions
        env = cast(RLCatanEnvWrapper, env)

        # Legal moves after filtering out excluded ActionTypes
        base_valid = env.get_valid_actions()

        # Further filter valid actions with heuristics
        filtered_valid = heuristic_mask(env, base_valid)

        # Cast action_space to Discrete to access `n`
        action_space = cast(Discrete, env.action_space)
        mask = np.zeros(action_space.n, dtype=bool)
        mask[filtered_valid] = True
        return mask

    # Second wrap: ActionMasker for MaskablePPO
    masked_env = ActionMasker(wrapped_env, mask_fn)

    return masked_env


def ppo_train(step_lim = 1_000):
    # Seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    env = make_env(seed=seed)

    model_path = os.path.join("..", "models", "ppo_v2.zip")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = MaskablePPO.load(model_path, env=env, device=device)
    else:
        print("Creating new model...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )

    # Might want to adjust total_timesteps based on compute resources
    total_timesteps = step_lim
    model.learn(total_timesteps=total_timesteps)

    # The model is saved to ./models/ppo_v1 so it can be imported by our player subclass
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    model.save(os.path.join("..", "models", "ppo_v2"))


if __name__ == "__main__":
    ppo_train()