from __future__ import annotations
from typing import Any, Iterable, Dict, cast, SupportsFloat

import gymnasium as gym
from typing import Iterable, List, Dict, Any
from catanatron.models.enums import ActionType, RESOURCES
from catanatron.gym.envs.catanatron_env import CatanatronEnv, to_action_space


class RLCatanEnvWrapper(gym.Wrapper):
    """
    A Gym environment wrapper for Catanatron that filters out certain action types
    to simplify the action space for reinforcement learning agents.

    This wrapper excludes passed action type sets (Such as defined in action_type_filtering.py).

    Injects per-player resource counts and exposes filtered valid action indices.
    """
    def __init__(self, env: gym.Env, excluded_type_groups: Iterable[Iterable[ActionType]] = ()):
        super().__init__(env)
        # Flatten all excluded type groups into a single set for O(1) membership checks.
        self._excluded: set[ActionType] = set().union(*excluded_type_groups)
        self._printed_resources_once = False

    # Resource helpers
    def _inject_resource_counts(self, info: Dict[str, Any]):
        base = self.env.unwrapped
        # Obtain the canonical State object regardless of whether base is a CatanatronEnv or a wrapper around it.
        st = base.game.state if isinstance(base, CatanatronEnv) else getattr(base, "game", None).state
        # Build a {Color -> {RESOURCE -> count}} map by reading from State.player_state.
        rc = {color: _player_resources(st, color) for color in getattr(st, "colors", [])}
        if rc:
            # Expose counts via info so wrappers/callbacks can use them.
            info["resource_counts"] = rc
            # Print the structure once to confirm shape and keys during runtime.
            if not self._printed_resources_once:
                self._printed_resources_once = True
                print("[RLCatanEnvWrapper] Injected resource_counts structure:")
                for color, counts in rc.items():
                    print(f"  {getattr(color, 'value', color)}: {counts}")

    # Actions
    def _playable_actions(self) -> List[Any]:
        """
        Reads the list of legal actions from the game state.
        Returns a copy (list()) to avoid accidental in-place modification of the state's list.
        """
        base = self.env.unwrapped
        st = base.game.state if isinstance(base, CatanatronEnv) else getattr(base, "game", None).state
        return list(getattr(st, "playable_actions", []) or [])

    def get_valid_actions(self) -> List[int]:
        """
        Return filtered valid actions with excluded ActionTypes removed.
        The returned values are global Discrete action indices, via to_action_space.
        """
        actions = self._playable_actions()
        if not actions:
            # Some prompts may not require a choice, so log for diagnostics and return empty.
            base = self.env.unwrapped
            st = base.game.state if isinstance(base, CatanatronEnv) else getattr(base, "game", None).state
            print(f"[RLCatanEnvWrapper] No playable_actions (prompt={getattr(st,'current_prompt',None)})")
            return []
        # Drop actions whose action_type is in the excluded set, which simplifies the action space.
        filtered = [a for a in actions if getattr(a, "action_type", None) not in self._excluded]
        if not filtered:
            # If filtering removed everything, fall back to unfiltered list to avoid empty masks.
            print("[RLCatanEnvWrapper] Filter removed all actions; using unfiltered indices")
            filtered = actions
        # Map each domain action to its integer index used by the gym Discrete action space.
        # Common indices: 0=ROLL (before rolling), last index may be END_TURN (after rolling).
        return [to_action_space(a) for a in filtered]

    # Gym api
    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation and info dictionary
        with filtered valid actions.
        """
        obs, info = self.env.reset(**kwargs)
        # Some envs return None or non-dicts; normalize to a dict so we can attach fields.
        if not isinstance(info, dict):
            info = {}
        # Attach current per-player resource counts to info.
        self._inject_resource_counts(info)
        return obs, info

    def step(self, action):
        # Delegate to the wrapped env, then enrich the info dict.
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {}
        # Keep resource counts updated every step so downstream wrappers/callbacks see fresh data.
        self._inject_resource_counts(info)
        return obs, reward, terminated, truncated, info


def _player_resources(state: Any, color: Any) -> Dict[str, int]:
    """
        Given a State object and a player color, return a dict mapping containing the resource count for each resource
    """
    # Convert a player color to its numeric index (P{idx}_* fields).
    idx = state.color_to_index.get(color)
    # Read counts from State.player_state using the canonical key pattern:
    #   P{idx}_{RESOURCE}_IN_HAND  (e.g., P0_WOOD_IN_HAND)
    # If idx is None (unknown color), default all resource counts to 0.
    return {r: int(state.player_state.get(f"P{idx}_{r}_IN_HAND", 0)) for r in RESOURCES} if idx is not None else {r: 0 for r in RESOURCES}
