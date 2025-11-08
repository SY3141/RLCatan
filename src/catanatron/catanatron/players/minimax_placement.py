import time
import random
from typing import Any

from catanatron.game import Game
from catanatron import Action
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import expand_spectrum, list_prunned_actions
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
)
from typing import Iterable
from collections import defaultdict
from catanatron.models.enums import ActionType

ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 20


class AlphaBetaPlacementPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    NOTE: More than 3 levels seems to take much longer, it would be
    interesting to see this with prunning.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false"
        self.value_fn_builder_name = (
            "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.use_value_function = None
        self.epsilon = epsilon

        self.turn = 0
        self.production_counts = {
            "WHEAT": 0,
            "ORE": 0,
            "BRICK": 0,
            "SHEEP": 0,
            "WOOD": 0,
        }  # Tracks resource production counts

    def value_function(self, game, p0_color):
        raise NotImplementedError

    def get_actions(self, game):
        if self.prunning:
            return list_prunned_actions(game)
        return game.state.playable_actions

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
        if self.turn in [0, 2]:  # placement phase for towns
            self.turn += 1
            return self.choose_placement(game, self.color, playable_actions)

        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        start = time.time()
        state_id = str(len(game.state.actions))
        node = DebugStateNode(state_id, self.color)  # i think it comes from outside
        deadline = start + MAX_SEARCH_TIME_SECS
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node
        )
        # print("Decision Results:", self.depth, len(actions), time.time() - start)
        # if game.state.num_turns > 10:
        #     render_debug_tree(node)
        #     breakpoint()
        if result[0] is None:
            return playable_actions[0]
        return result[0]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        maximizingPlayer = game.state.current_color() == self.color
        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        if maximizingPlayer:
            best_action = None
            best_value = float("-inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # beta cutoff

            node.expected_value = best_value
            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # alpha cutoff

            node.expected_value = best_value
            return best_action, best_value


class DebugStateNode:
    def __init__(self, label, color):
        self.label = label
        self.children = []  # DebugActionNode[]
        self.expected_value = None
        self.color = color


class DebugActionNode:
    def __init__(self, action):
        self.action = action
        self.expected_value: Any = None
        self.children = []  # DebugStateNode[]
        self.probas = []


# def render_debug_tree(node):
#     from graphviz import Digraph

#     dot = Digraph("AlphaBetaSearch")

#     agenda = [node]

#     while len(agenda) != 0:
#         tmp = agenda.pop()
#         dot.node(
#             tmp.label,
#             label=f"<{tmp.label}<br /><font point-size='10'>{tmp.expected_value}</font>>",
#             style="filled",
#             fillcolor=tmp.color.value,
#         )
#         for child in tmp.children:
#             action_label = (
#                 f"{tmp.label} - {str(child.action).replace('<', '').replace('>', '')}"
#             )
#             dot.node(
#                 action_label,
#                 label=f"<{action_label}<br /><font point-size='10'>{child.expected_value}</font>>",
#                 shape="box",
#             )
#             dot.edge(tmp.label, action_label)
#             for action_child, proba in zip(child.children, child.probas):
#                 dot.node(
#                     action_child.label,
#                     label=f"<{action_child.label}<br /><font point-size='10'>{action_child.expected_value}</font>>",
#                 )
#                 dot.edge(action_label, action_child.label, label=str(proba))
#                 agenda.append(action_child)
#     print(dot.render())


class SameTurnAlphaBetaPlayer(AlphaBetaPlacementPlayer):
    """
    Same like AlphaBeta but only within turn
    """

    def alphabeta(self, game, depth, alpha, beta, deadline, node):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if (
            depth == 0
            or game.state.current_color() != self.color
            or game.winning_color() is not None
            or time.time() >= deadline
        ):
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        best_action = None
        best_value = float("-inf")
        for i, (action, outcomes) in enumerate(action_outcomes.items()):
            action_node = DebugActionNode(action)

            expected_value = 0
            for j, (outcome, proba) in enumerate(outcomes):
                out_node = DebugStateNode(
                    f"{node.label} {i} {j}", outcome.state.current_color()
                )

                result = self.alphabeta(
                    outcome, depth - 1, alpha, beta, deadline, out_node
                )
                value = result[1]
                expected_value += proba * value

                action_node.children.append(out_node)
                action_node.probas.append(proba)

            action_node.expected_value = expected_value
            node.children.append(action_node)

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # beta cutoff

        node.expected_value = best_value
        return best_action, best_value
