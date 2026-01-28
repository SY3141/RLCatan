from __future__ import annotations

import os
import random
from typing import Iterable, cast
import torch
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.logger import configure

from catanatron.gym.envs.catanatron_env import CatanatronEnv

from catanatron.models.enums import ActionType
from catanatron.gym.action_type_filtering import (
    COMPLEX_DEV_CARD_ACTION_TYPES,
    PLAYER_TRADING_ACTION_TYPES,
)
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.mcts import MCTSPlayer  # too slow for training
from catanatron.players.playouts import GreedyPlayoutsPlayer  # broken
from catanatron.players.value import ValueFunctionPlayer  # too good for training v3
from catanatron.players.ppo_player import PPOPlayer  # goldilocks bot
from catanatron.models.player import Color

# from stable_baselines3.common.vec_env import SubprocVecEnv
from catanatron.gym.reward_wrapper import RewardWrapper
from catanatron.gym.callbacks import ResourceLogCallback


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


def make_env(seed: int | None = None, filtered_actions=[]) -> gym.Env:
    """
    Build a single training environment:
      - CatanatronEnv (1v1 vs. A chosen bot)
      - RLCatanEnvWrapper: filters out some ActionTypes
      - RewardWrapper: Adds shaping rewards for resources
      - ActionMasker: gives MaskablePPO a valid-action mask
    """
    base_env = CatanatronEnv(
        config={"enemies": [ValueFunctionPlayer(Color.RED)], "vps_to_win": 15}
    )
    print("Enemy bot:", base_env.enemies)

    if seed is not None:
        base_env.reset(seed=seed)

    # Excluding complex dev card actions and player trading actions for v1
    excluded_type_groups: Iterable[Iterable[ActionType]] = [filtered_actions]

    # First wrap: Filter out unwanted ActionTypes
    wrapped_env = RLCatanEnvWrapper(base_env, excluded_type_groups=excluded_type_groups)
    # Second wrap: Add Reward Shaping
    reward_env = RewardWrapper(
        wrapped_env,
        gain_scale=0.1,
        spend_scale=0.05,
        decay_factor=0.999,
        build_scale=0.02,
    )

    # Masking function for SB3 MaskablePPO, I adapted it to include heuristic filtering
    def mask_fn(env: gym.Env) -> np.ndarray:
        """
        Mask function for SB3 ActionMasker.

        `env` is the wrapped environment (RLCatanEnvWrapper), so this function:
          - calls env.get_valid_actions() to get already-filtered indices
          - casts env.action_space to Discrete to get `n` for mask length
        """
        # Cast to RewardWrapper to access get_valid_actions
        reward_wrapper = cast(RewardWrapper, env)
        base_valid = reward_wrapper.get_valid_actions()

        # Further filter valid actions with heuristics
        # Pass the inner env RLCatanEnvWrapper to the heuristic function
        filtered_valid = heuristic_mask(
            cast(RLCatanEnvWrapper, reward_wrapper.env), base_valid
        )

        # Cast action_space to Discrete to access `n`
        action_space = cast(Discrete, env.action_space)
        mask = np.zeros(action_space.n, dtype=bool)
        if len(filtered_valid) == 0:
            mask[:] = True
        else:
            mask[filtered_valid] = True
        return mask

    # Third wrap: ActionMasker wraps the reward_env
    masked_env = ActionMasker(reward_env, mask_fn)

    return masked_env


def ppo_train(step_lim=1_000, model_name="ppo_v3"):
    # Seed for reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # env = SubprocVecEnv([make_env for _ in range(8)]) #trying to use more cores
    env = make_env(seed=seed)
    model_path = os.path.join("..", "models", f"{model_name}.zip")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(model_path):
        print("Loading existing model...")
        model = MaskablePPO.load(model_path, env=env, device=device)

        # Create a new logger that writes to tensorboard
        new_logger = configure("./ppo_tensorboard_logs/", ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        print("Creating new model...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            device=device,
            tensorboard_log="./ppo_tensorboard_logs/",
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0002,
            vf_coef=0.5,
        )

    # Logs reward function information in Tensorboard
    callback = ResourceLogCallback()

    # Might want to adjust total_timesteps based on compute resources
    total_timesteps = 3000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # The model is saved to ./models/ppo_v1 so it can be imported by our player subclass
    os.makedirs(os.path.join("..", "models"), exist_ok=True)
    model.save(os.path.join("..", "models", model_name))


if __name__ == "__main__":
    ppo_train()
