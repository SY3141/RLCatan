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
        self.production_counts = defaultdict(int) # Tracks resource production counts
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

        # Compute pip totals for every node
        totals = {
            node_id: sum(number_to_pips(tile.number) if isinstance(tile.number, int) else 0 for tile in tiles)
            for node_id, tiles in node_to_tiles.items()
        }

        # Extract buildable node IDs from playable actions
        buildable_nodes = {
            action.value for action in playable_actions
            if action.action_type == ActionType.BUILD_SETTLEMENT
        }

        # Filter only buildable nodes from totals
        buildable_totals = {nid: totals[nid] for nid in buildable_nodes if nid in totals}

        # Sort by pip total, descending
        sorted_buildable = sorted(buildable_totals.items(), key=lambda x: x[1], reverse=True)

        print(f"Buildable node pip totals: {sorted_buildable}")
        return sorted_buildable

    def choose_placement(self, game: Game, player_color: str, playable_actions: Iterable[Action]) -> Action:
        """Called once at the start of the game to read the map.
        Args:
            game (Game): complete game state. read-only.
        """
        road_actions = [a for a in playable_actions if a.action_type == ActionType.BUILD_ROAD]
        if road_actions:
            print(f"Chosen road action: {road_actions[0]}")
            return road_actions[0]
        
        most_pip_nodes = self.compute_node_pip_totals(game.state.board, playable_actions)[:5] #node with most pips
        chosen_node = random.choice(most_pip_nodes)
        #Try to find a placement in action whose value matches the node ID
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
            self.turn+=1
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

