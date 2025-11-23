from __future__ import annotations
from typing import Any, Iterable, Dict, cast, SupportsFloat

import gymnasium as gym
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.models.enums import ActionType
from catanatron.gym.action_type_filtering import filter_action_types


class RLCatanEnvWrapper(gym.Wrapper):
    """
    A Gym environment wrapper for Catanatron that filters out certain action types
    to simplify the action space for reinforcement learning agents.

    This wrapper excludes passed action type sets (Such as defined in action_type_filtering.py).
    """

    def __init__(
        self, env: CatanatronEnv, excluded_type_groups: Iterable[Iterable[ActionType]]
    ) -> None:
        super().__init__(env)
        self._excluded_type_groups = excluded_type_groups

    def get_valid_actions(self) -> list[int]:
        """
        Return filtered valid actions (with excluded ActionTypes removed).
        """
        base_env: CatanatronEnv = cast(CatanatronEnv, self.env)
        base_valid = base_env.get_valid_actions()

        return filter_action_types(
            base_env, base_valid, excluded_types=self._excluded_type_groups
        )

    def reset(self, **kwargs) -> tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and returns the initial observation and info dictionary
        with filtered valid actions.
        """
        observation, info = self.env.reset(**kwargs)
        info["valid_actions"] = self.get_valid_actions()

        return observation, info

    def step(
        self, action: int
    ) -> tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the environment using the given action index.
        Returns the observation, reward, done flags, and info dictionary
        with filtered valid actions.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        info["valid_actions"] = self.get_valid_actions()

        return observation, reward, terminated, truncated, info
