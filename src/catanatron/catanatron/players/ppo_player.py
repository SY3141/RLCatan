from __future__ import annotations

import os.path
from pathlib import Path
from typing import Iterable, List, Optional, cast

import numpy as np
from gymnasium.spaces import Discrete
from sb3_contrib.ppo_mask import MaskablePPO

from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionType
from catanatron.features import create_sample, get_feature_ordering

from catanatron.gym.envs.catanatron_env import (
    ACTIONS_ARRAY,
    to_action_space,
    from_action_space,
)

from catanatron.gym.action_type_filtering import (
    COMPLEX_DEV_CARD_ACTION_TYPES,
    PLAYER_TRADING_ACTION_TYPES,
)


class PPOPlayer(Player):
    """
    Catanatron Player that uses a trained MaskablePPO model.

    Assumes:
      - 1v1 game (PPO agent vs one opponent)
      - BASE map
      - Vector observation created with `create_sample` + `get_feature_ordering(2)`
      - Model file at `models/ppo_v1.zip`
    """

    def __init__(
        self,
        color: Color,
    ):
        super().__init__(color)

        # Load the trained PPO model (.zip extension added automatically)
        base_dir = Path(__file__).resolve().parents[3]  # adjust depth to match your layout
        model_path = base_dir / "rlcatan" / "models" / "ppo_v2"

        self.model: MaskablePPO = MaskablePPO.load(model_path, device="cpu")

        # Feature ordering must match training
        # During training, CatanatronEnv used num_players=2, BASE map.
        self.features: List[str] = get_feature_ordering(num_players=2)

        # Excluding the same ActionType groups as in training
        self.excluded_type_groups = [
            COMPLEX_DEV_CARD_ACTION_TYPES,
            PLAYER_TRADING_ACTION_TYPES,
        ]

    def _build_observation(self, game) -> np.ndarray:
        """
        Build the observation vector from the current game state,
        matching the training representation.
        """
        # Reusing create_sample from features.py to build the observation
        sample = create_sample(game, self.color)
        obs_vec = np.array([float(sample[f]) for f in self.features], dtype=np.float32)

        return obs_vec

    def _indices_from_playable_actions(self, playable_actions: Iterable[Action]) -> List[int]:
        """
        Map each playable Action to its discrete action index using the same
        encoding as CatanatronEnv (to_action_space).
        """
        return [to_action_space(a) for a in playable_actions]

    def _apply_action_type_filters(self, indices: List[int]) -> List[int]:
        """
        Filter out indices whose ACTIONS_ARRAY entry has an ActionType
        in any of the excluded groups.
        """
        if not self.excluded_type_groups:
            return indices

        filtered: List[int] = []

        for idx in indices:
            action_type, _ = ACTIONS_ARRAY[idx]

            # Changed logic from action_type_filtering to use any() over excluded groups, not sure which approach is more efficient
            if any(action_type in group for group in self.excluded_type_groups):
                continue

            filtered.append(idx)

        return filtered

    def _build_action_mask(self, valid_indices: List[int]) -> np.ndarray:
        """
        Build a boolean mask over the global action space indices,
        with True marking allowed actions.

        This mask is passed into MaskablePPO.predict(action_masks=...).
        """
        action_space = cast(Discrete, self.model.action_space)
        n_actions = action_space.n
        mask = np.zeros(n_actions, dtype=bool)
        mask[valid_indices] = True

        return mask

    # ----- Required Player API -----

    def decide(self, game, playable_actions: Iterable[Action]) -> Action:
        """
        Main entry point called by Catanatron's Game engine.

        Args:
            game: full Game instance (read-only).
            playable_actions: iterable of Action objects.

        Returns:
            One of the playable_actions chosen by the PPO policy.
        """

        # 1. Build observation from the current game state
        obs = self._build_observation(game)

        # 2. Map playable Actions -> global action indices
        playable_actions = list(playable_actions) # Saves recomputing this list later
        all_valid_indices = self._indices_from_playable_actions(playable_actions)

        # 3. Apply v1 simplification filters (no complex dev cards, no player trades)
        filtered_indices = self._apply_action_type_filters(all_valid_indices)

        # 4. Build mask over the entire action space
        action_mask = self._build_action_mask(filtered_indices)

        # 5. Query PPO policy
        action_int, _ = self.model.predict(
            obs,
            deterministic=True,
            action_masks=action_mask,
        )
        chosen_index = int(action_int)

        # 6. Map index -> concrete Action from playable_actions
        #    This will pick the first matching Action in playable_actions whose
        #    normalized (action_type, value) matches ACTIONS_ARRAY[chosen_index].
        chosen_action = from_action_space(chosen_index, list(playable_actions))

        return chosen_action

    def reset_state(self):
        """
        We can mess with this if we want to reset any internal
        per-game state. For now, it just defers to the base class.
        """
        super().reset_state()
