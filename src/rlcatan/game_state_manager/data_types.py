
from enum import Enum, auto

"""Represents the Move From Player and GameState"""
class ActionType(Enum):
    BUILD_ROAD = auto()
    BUILD_SETTLEMENT = auto()
    BUILD_CITY = auto()
    COLLECT_RESOURCES = auto()
    END_TURN = auto()

class MoveData:
    def __init__(self, player_id: str, action_type: ActionType, parameters: dict = None):
        self.player_id = player_id
        self.action_type = action_type
        self.parameters = parameters if parameters is not None else {}

class GameStateData:
    def __init__(self):
        self.board_layout: dict = {}
        self.turn_count: int = 0
        self.is_game_over: bool = False