# TODO Make a wrapper here of the Catanatron environment to allow for filtered observations and actions

from __future__ import annotations
from typing import Any, Dict, Iterable, List

import gymnasium as gym
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.gym.action_type_filtering import filter_action_types, COMPLEX_DEV_CARD_ACTION_TYPES, PLAYER_TRADING_ACTION_TYPES

class RLCatanEnvWrapper(gym.Wrapper):
    """
    A Gym environment wrapper for Catanatron that filters out certain action types
    to simplify the action space for reinforcement learning agents.

    This wrapper excludes complex development card actions and player trading actions
    from the set of valid actions returned by the environment.
    """

    def __init__(self, env: CatanatronEnv):
        super().__init__(env)

    def reset(self, **kwargs) -> tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and returns the initial observation and info dictionary
        with filtered valid actions.
        """
        observation, info = self.env.reset(**kwargs)

        filtered_actions = filter_action_types(
            self.env,
            info["valid_actions"],
            excluded_types=[COMPLEX_DEV_CARD_ACTION_TYPES, PLAYER_TRADING_ACTION_TYPES]
        )

        info["valid_actions"] = filtered_actions

        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the environment using the given action index.
        Returns the observation, reward, done flags, and info dictionary
        with filtered valid actions.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        filtered_actions = filter_action_types(
            self.env,
            info["valid_actions"],
            excluded_types=[COMPLEX_DEV_CARD_ACTION_TYPES, PLAYER_TRADING_ACTION_TYPES]
        )
        info["valid_actions"] = filtered_actions
        return observation, reward, terminated, truncated, info