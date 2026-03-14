import random
import builtins

from enum import Enum
from catanatron.models.actions import serialize_action


class Color(Enum):
    """Enum to represent the colors in the game"""

    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    """Interface to represent a player's decision logic.

    Formulated as a class (instead of a function) so that players
    can have an initialization that can later be serialized to
    the database via pickle.
    """

    def __init__(self, color, is_bot=True):
        """Initialize the player

        Args:
            color(Color): the color of the player
            is_bot(bool): whether the player is controlled by the computer
        """
        self.color = color
        self.is_bot = is_bot
        self.last_decision_info = None # For LLM move explanations.

    def decide(self, game, playable_actions):
        """Should return one of the playable_actions or
        an OFFER_TRADE action if its your turn and you have already rolled.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options right now
        """
        raise NotImplementedError

    def decide_with_context(self, game, playable_actions):
        """Wrapper around decide() that also saves the context of the decision for later analysis."""
        playable_actions = list(playable_actions)
        action = self.decide(game, playable_actions)

        self.last_decision_info = self.build_decision_info(
            game, playable_actions, action
        )

        return action

    def build_decision_info(self, game, playable_actions, chosen_action):
        """
        Build a dictionary of information about the decision just made, for later analysis.
        get_decision_details() can be overridden by subclasses to add more specific information about the decision.
        """
        return {
            # Basic info about the game state and decision
            "bot_class": type(self).__name__,
            "actor": game.state.current_color().value,
            "current_player_index": game.state.current_player_index,
            "current_turn_index": game.state.current_turn_index,

            # Prompt is deprecated, but still useful for now. The is_* flags are its replacement, so they're included too
            "prompt": game.state.current_prompt.value, # Current prompt (e.g. "BUILD_INITIAL_SETTLEMENT", "PLAY_TURN", etc.)
            "is_initial_build_phase": game.state.is_initial_build_phase,
            "is_discarding": game.state.is_discarding,
            "is_moving_knight": game.state.is_moving_knight,
            "is_road_building": game.state.is_road_building,

            # Information about the options available to the player and the choice they made
            "playable_actions": [serialize_action(a) for a in playable_actions],
            "chosen_action": serialize_action(chosen_action),

            # Any additional info provided by the specific Player subclass about the decision
            **self.get_decision_details(game, playable_actions, chosen_action),
        }

    def get_decision_details(self, game, playable_actions, chosen_action):
        """
        Hook for subclasses to add more specific information about the decision made.
        This will be included in the decision_info dictionary built in build_decision_info().
        """
        return {}

    def reset_state(self):
        """Hook for resetting state between games"""
        self.last_decision_info = None

    def __repr__(self):
        return f"{type(self).__name__}:{self.color.value}"


class SimplePlayer(Player):
    """Simple AI player that always takes the first action in the list of playable_actions"""

    def decide(self, game, playable_actions):
        return playable_actions[0]


class HumanPlayer(Player):
    """Human player that selects which action to take using standard input"""

    def __init__(self, color, is_bot=False, input_fn=builtins.input):
        super().__init__(color, is_bot)
        self.input_fn = input_fn  # this is for testing purposes

    def decide(self, game, playable_actions):
        for i, action in enumerate(playable_actions):
            print(f"{i}: {action.action_type} {action.value}")
        i = None
        while i is None or (i < 0 or i >= len(playable_actions)):
            print("Please enter a valid index:")
            try:
                x = self.input_fn(">>> ")  # Use the input_fn
                i = int(x)
            except ValueError:
                pass

        return playable_actions[i]


class RandomPlayer(Player):
    """Random AI player that selects an action randomly from the list of playable_actions"""

    def decide(self, game, playable_actions):
        return random.choice(playable_actions)
