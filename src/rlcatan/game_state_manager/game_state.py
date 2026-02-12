from rlcatan.game_state_manager.data_types import GameStateData
from rlcatan.game_state_manager.data_types import MoveData
import typing

class InvalidMoveError(Exception):
    """Raised when an illegal move is applied."""
    pass

class GameStateManager:
    MAX_PLAYERS: int = 4

    def __init__(self):
        self._current_game_state = GameStateData()
        self._player_assets: typing.Dict[str, dict] = {}
        
        for i in range(1, self.MAX_PLAYERS + 1):
            p_id = f"Player_{i}"
            self._player_assets[p_id] = {
                "resources": {"brick": 0, "lumber": 0, "wool": 0, "grain": 0, "ore": 0},
                "structures": [],
                "score": 0
            }

    def validate_move(self, move: MoveData):
        if move.player_id not in self._player_assets:
            return False
        return True

    def update_state(self, move: MoveData):
        if self.validate_move(move):
            self._apply_move(move)
        else:
            raise InvalidMoveError(f"Invalid move: {move.action_type}")

    def get_state(self):
        return self._current_game_state

    def _apply_move(self, move: MoveData):
        self._calculate_resources(move)
        self._update_scores()
        self._current_game_state.turn_count += 1
        
        if self._check_victory():
            self._current_game_state.is_game_over = True

    def _calculate_resources(self, move: MoveData):
        assets = self._player_assets[move.player_id]
        res = assets["resources"]
        structs = assets["structures"]
        
        if move.action_type == "build_road":
            res["brick"] -= 1
            res["lumber"] -= 1
            structs.append("road")
        elif move.action_type == "build_settlement":
            res["brick"] -= 1
            res["lumber"] -= 1
            res["wool"] -= 1
            res["grain"] -= 1
            structs.append("settlement")
        elif move.action_type == "build_city":
            res["ore"] -= 3
            res["grain"] -= 2
            if "settlement" in structs:
                structs.remove("settlement")
            structs.append("city")
        elif move.action_type == "collect_resources":
            r_type = move.parameters.get("type")
            amount = move.parameters.get("amount", 0)
            if r_type in res:
                res[r_type] += amount

    def _update_scores(self):
        for p_id in self._player_assets:
            structs = self._player_assets[p_id]["structures"]
            score = structs.count("settlement") * 1 + structs.count("city") * 2
            self._player_assets[p_id]["score"] = score

    def _check_victory(self):
        for p_id in self._player_assets:
            if self._player_assets[p_id]["score"] >= 10:
                return True
        return False