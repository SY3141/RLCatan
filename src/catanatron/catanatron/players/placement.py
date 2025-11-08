from collections import defaultdict
from typing import Iterable

from catanatron import Game, Action, Player
import random
from catanatron.state_functions import (
    player_key,
)
from catanatron.models.enums import ActionType


class PlacementPlayer(Player):

    def __init__(self, color: str):
        super().__init__(color)
        self.turn = 0
        self.production_counts = {
            "WHEAT": 0,
            "ORE": 0,
            "BRICK": 0,
            "SHEEP": 0,
            "WOOD": 0,
        }  # Tracks resource production counts
        # resources are 'wheat', 'ore', 'brick', 'sheep', 'wood'

    def compute_node_pip_totals(self, board, playable_actions):
        def number_to_pips(number):
            pip_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
            return pip_map.get(number, 0)

        # Map each node to its adjacent tiles
        node_to_tiles = defaultdict(list)
        for _, tile in board.map.land_tiles.items():
            for node_id in tile.nodes.values():
                node_to_tiles[node_id].append(tile)

        # Compute per-resource pip totals: [wheat, ore, brick, sheep, wood]
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

        # print("Buildable node pip totals:")
        # for nid, pip_list in sorted_buildable:
        #     print(f"Node {nid}: {pip_list} (sum={sum(pip_list)})")

        return sorted_buildable

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
        # Try to find a placement in action whose value matches the node ID
        with open("placement_log.txt", "a") as f:
            for action in playable_actions:
                f.write(f"Action: {action}, Value: {action.value}\n")
            f.write(f"NEW TURN\n")

        for action in playable_actions:
            if action.value == chosen_node[0]:
                print(f"Chosen action: {action}")
                return action

        # If not found, fall back to first available action
        print(f"Node not in playable_actions; using default.")
        return playable_actions[0]

    def decide(self, game: Game, playable_actions):
        if self.turn < 4:
            self.turn += 1
            return self.choose_placement(game, self.color, playable_actions)

        if len(playable_actions) == 1:
            return playable_actions[0]

        best_value = float("-inf")
        best_actions = []
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)

            key = player_key(game_copy.state, self.color)
            value = game_copy.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_value = value
                best_actions = [action]

        return random.choice(best_actions)
