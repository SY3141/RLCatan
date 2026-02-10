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

# PLACEMENT IMPORTS
from collections import defaultdict


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
        model_name: Optional[str] = "ppo_v3",
    ):
        super().__init__(color)
        print(f"Initializing PPOPlayer with model '{model_name}'")
        # Load the trained PPO model (.zip extension added automatically)
        base_dir = (
            Path(__file__).resolve().parents[3]
        )  # adjust depth to match your layout
        model_path = base_dir / "rlcatan" / "models" / model_name

        self.model: MaskablePPO = MaskablePPO.load(model_path, device="cpu")

        # Feature ordering must match training
        # During training, CatanatronEnv used num_players=2, BASE map.
        self.features: List[str] = get_feature_ordering(num_players=2)

        # Excluding the same ActionType groups as in training

        self.excluded_type_groups = []

        # [     COMPLEX_DEV_CARD_ACTION_TYPES,
        #     PLAYER_TRADING_ACTION_TYPES,
        # ]
        # PLACEMENT
        self.turn = 0
        self.production_counts = {
            "WHEAT": 0,
            "ORE": 0,
            "BRICK": 0,
            "SHEEP": 0,
            "WOOD": 0,
        }  # Tracks resource production counts

    # PLACEMENT
    def compute_node_pip_totals(self, board, playable_actions):
        def number_to_pips(number):
            pip_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
            return pip_map.get(number, 0)

        # Map each node to its adjacent tiles
        node_to_tiles = defaultdict(list)
        for _, tile in board.map.land_tiles.items():
            for node_id in tile.nodes.values():
                node_to_tiles[node_id].append(tile)

        resource_order = ["WHEAT", "ORE", "BRICK", "SHEEP", "WOOD"]
        totals = {}

        for node_id, tiles in node_to_tiles.items():
            pip_list = [0, 0, 0, 0, 0]
            for tile in tiles:
                if (
                    hasattr(tile, "resource")
                    and tile.resource in resource_order
                    and isinstance(tile.number, int)
                ):
                    pips = number_to_pips(tile.number)
                    res_index = resource_order.index(tile.resource)
                    pip_list[res_index] += pips
            totals[node_id] = pip_list

        # Extract buildable node IDs from playable actions
        buildable_nodes = {
            action.value
            for action in playable_actions
            if action.action_type == ActionType.BUILD_SETTLEMENT
        }

        # Filter only buildable nodes
        buildable_totals = {
            nid: totals[nid] for nid in buildable_nodes if nid in totals
        }

        # Sort buildable nodes by total pips (sum of all resources)
        sorted_buildable = sorted(
            buildable_totals.items(), key=lambda x: sum(x[1]), reverse=True
        )

        return sorted_buildable

    # PLACEMENT
    def choose_placement(
        self, game: Game, player_color: str, playable_actions: Iterable[Action]
    ) -> Action:
        """Called once at the start of the game to read the map.
        Args:
            game (Game): complete game state. read-only.
        """
        road_actions = [
            a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD
        ]
        if road_actions:
            print(f"Chosen road action: {road_actions[0]}")
            return road_actions[0]

        most_pip_nodes = self.compute_node_pip_totals(
            game.state.board, playable_actions
        )[:10]
        if self.turn == 0:  # checks for first turn
            chosen_node = most_pip_nodes[0]
            # Determine which resource (wheat or ore) has the highest pip potential among top nodes
            max_ore = max(node[1][1] for node in most_pip_nodes)  # ore index = 1
            max_wheat = max(node[1][0] for node in most_pip_nodes)  # wheat index = 0
            if max_ore >= max_wheat:  # Sort by ore pips descendin
                most_pip_nodes.sort(key=lambda x: x[1][1], reverse=True)
            else:  # Sort by wheat pips descending
                most_pip_nodes.sort(key=lambda x: x[1][0], reverse=True)

            # Pick the top node with at least some production of the prioritized resource
            for node in most_pip_nodes:
                if node[1][0] > 0 or node[1][1] > 0:
                    chosen_node = node
                    break
        else:
            # Calculate a score for each node based on production counts
            best_score = float("-inf")
            chosen_node = most_pip_nodes[0]
            for node in most_pip_nodes:
                score = sum(
                    [
                        node[1][i] * list(self.production_counts.values())[i]
                        for i in range(5)
                    ]
                )
                if score > best_score:
                    best_score = score
                    chosen_node = node
        self.production_counts = {
            k: self.production_counts[k] + chosen_node[1][i]
            for i, k in enumerate(self.production_counts.keys())
        }  # adds production to production counts

        for action in playable_actions:
            if action.value == chosen_node[0]:
                return action

        # If not found, fall back to first available action
        print(f"Node not in playable_actions; using default.")
        return playable_actions[0]

    def _build_observation(self, game) -> np.ndarray:
        """
        Build the observation vector from the current game state,
        matching the training representation.
        """
        # Reusing create_sample from features.py to build the observation
        sample = create_sample(game, self.color)
        obs_vec = np.array([float(sample[f]) for f in self.features], dtype=np.float32)

        return obs_vec

    def _indices_from_playable_actions(
        self, playable_actions: Iterable[Action]
    ) -> List[int]:
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
        # PLACEMENT
        if self.turn in [0, 2]:  # placement phase for towns
            self.turn += 1
            return self.choose_placement(game, self.color, playable_actions)
        # PLACEMENT

        # 1. Build observation from the current game state
        obs = self._build_observation(game)

        # 2. Map playable Actions -> global action indices
        playable_actions = list(playable_actions)  # Saves recomputing this list later
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
