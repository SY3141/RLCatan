from __future__ import annotations

import os
import random
from typing import Iterable, cast, Any

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
from catanatron.gym.reward_wrapper import RewardWrapper
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper
from .curriculum import CurriculumManager
from .curriculum_callback import CurriculumCallback


def heuristic_mask(env: Any, valid_indices: list[int]) -> list[int]:
    # Placeholder for future heuristics
    return valid_indices


def make_env(
    seed: int | None = None,
    disabled_action_names: list[str] | None = None,
    reward_shaping: dict[str, Any] | None = None,
) -> gym.Env:
    base_env = CatanatronEnv(config={"opponent_type": "RandomPlayer"})

    if seed is not None:
        base_env.reset(seed=seed)

    # Wrap with Monitor to track episode rewards/lengths for PPO logs
    # This populates 'ep_info_buffer' which we need for the score
    monitored_env = Monitor(base_env)

    # Build excluded type groups; include curriculum-provided disabled action names as one group
    excluded_type_groups: list[Iterable[ActionType]] = [
        COMPLEX_DEV_CARD_ACTION_TYPES,
        PLAYER_TRADING_ACTION_TYPES,
    ]

    if disabled_action_names:
        disabled_set = set()
        for n in disabled_action_names:
            if n in ActionType.__members__:
                disabled_set.add(ActionType[n])
        if disabled_set:
            excluded_type_groups.append(disabled_set)

    wrapped_env = RLCatanEnvWrapper(monitored_env, excluded_type_groups=excluded_type_groups)

    # Curriculum implemented reward shaping wrapper.
    if reward_shaping and reward_shaping.get("enabled", True):
        allowed_keys = {
            "gain_scale",
            "spend_scale",
            "decay_factor",
            "build_scale",
            "player_idx",
            "resource_attr",
            "debug",
        }
        reward_kwargs = {k: v for k, v in reward_shaping.items() if k in allowed_keys}
        wrapped_env = RewardWrapper(wrapped_env, **reward_kwargs)

    def mask_fn(env: gym.Env) -> np.ndarray:
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
    curriculum_json: str | None = None,
):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    env = make_env(seed=seed)

    curriculum = None
    if curriculum_json:
        try:
            curriculum = CurriculumManager.from_json(curriculum_json)
            # Apply initial phase by adjusting env wrapper excluded sets if provided
            phase = curriculum.current_phase()
            disabled_types = phase.get("disabled_action_types", [])
            reward_shaping = phase.get("reward_shaping")
            # Rebuild env with disabled action names applied
            env = make_env(
                seed=seed,
                disabled_action_names=disabled_types,
                reward_shaping=reward_shaping,
            )
            print(f"[Curriculum] Starting with phase: {phase.get('name')}")
            print(f"[Curriculum] Target VP: {phase.get('target_vp')}, Disabled actions: {disabled_types}")
            if reward_shaping and reward_shaping.get("enabled", True):
                print(f"[Curriculum] Reward shaping enabled with config: {reward_shaping}")
        except Exception as e:
            print(f"[Curriculum] Failed to load: {e}")

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
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

    # Setup curriculum callback if curriculum is provided
    callbacks = [reward_tracker]
    if curriculum is not None:
        curriculum_callback = CurriculumCallback(
            curriculum=curriculum,
            env_factory=lambda disabled, reward: make_env(
                seed=seed,
                disabled_action_names=disabled,
                reward_shaping=reward,
            ),
            eval_freq=2048,
            verbose=1,
        )
        callbacks.append(curriculum_callback)
        print("[Curriculum] Curriculum callback enabled")

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
    # Allow quick local testing by pointing to configs/curriculum.json
    cfg_path = os.environ.get("RLCATAN_CURRICULUM_JSON", os.path.join("..", "..", "configs", "curriculum.json"))
    if os.path.exists(cfg_path):
        print(f"Using curriculum config: {cfg_path}")
        score = train_ppo(save_path=os.path.join("..", "models", "ppo_v1"), curriculum_json=cfg_path)
    else:
        score = train_ppo(save_path=os.path.join("..", "models", "ppo_v1"))
    print(f"Final Training Score (ep_rew_mean): {score}")
