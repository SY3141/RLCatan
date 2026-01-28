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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor  # <--- Added this

from catanatron.gym.envs.catanatron_env import CatanatronEnv

from catanatron.models.enums import ActionType
from catanatron.gym.action_type_filtering import (
    COMPLEX_DEV_CARD_ACTION_TYPES,
    PLAYER_TRADING_ACTION_TYPES,
)
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper


def heuristic_mask(env: RLCatanEnvWrapper, valid_indices: list[int]) -> list[int]:
    base_env = cast(CatanatronEnv, env.env)
    # Placeholder for future heuristics
    return valid_indices


def make_env(seed: int | None = None) -> gym.Env:
    base_env = CatanatronEnv(config={"opponent_type": "RandomPlayer"})

    if seed is not None:
        base_env.reset(seed=seed)

    # Wrap with Monitor to track episode rewards/lengths for PPO logs
    # This populates 'ep_info_buffer' which we need for the score
    monitored_env = Monitor(base_env)

    excluded_type_groups: Iterable[Iterable[ActionType]] = [
        COMPLEX_DEV_CARD_ACTION_TYPES,
        PLAYER_TRADING_ACTION_TYPES,
    ]

    wrapped_env = RLCatanEnvWrapper(
        monitored_env, excluded_type_groups=excluded_type_groups
    )

    def mask_fn(env: gym.Env) -> np.ndarray:
        env = cast(RLCatanEnvWrapper, env)
        base_valid = env.get_valid_actions()
        filtered_valid = heuristic_mask(env, base_valid)
        action_space = cast(Discrete, env.action_space)
        mask = np.zeros(action_space.n, dtype=bool)
        mask[filtered_valid] = True
        return mask

    masked_env = ActionMasker(wrapped_env, mask_fn)
    return masked_env


class RewardTrackerCallback(BaseCallback):
    """
    Callback to capture the 'ep_rew_mean' (mean reward of last 100 episodes)
    from the training buffer at the very end of training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.final_mean_reward = -100.0

    def _on_step(self) -> bool:
        # Stable Baselines3 maintains a buffer of the last 100 episode stats
        if len(self.model.ep_info_buffer) > 0:
            self.final_mean_reward = np.mean(
                [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            )
        return True


def train_ppo(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    total_timesteps=1_000_000,
    save_path=None,
):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    env = make_env(seed=seed)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
    )

    reward_tracker = RewardTrackerCallback()

    try:
        model.learn(total_timesteps=total_timesteps, callback=reward_tracker)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            print(f"Model saved to {save_path}")

        # Return the score directly for the hyperparameter search
        return reward_tracker.final_mean_reward

    except Exception as e:
        print(
            f"Training failed with params: LR={learning_rate}, Batch={batch_size}... Error: {e}"
        )
        return -100.0


if __name__ == "__main__":
    score = train_ppo(save_path=os.path.join("..", "models", "ppo_v1"))
    print(f"Final Training Score (ep_rew_mean): {score}")
