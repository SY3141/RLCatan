import os
import json
import logging
import traceback
from typing import List
from functools import lru_cache
from pathlib import Path

from flask import Response, Blueprint, jsonify, abort, request

from catanatron.web.models import upsert_game_state, get_game_state
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.game import Game

from catanatron.players.minimax_placement import AlphaBetaPlacementPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.placement import PlacementPlayer
from catanatron.players.ppo_player import PPOPlayer

bp = Blueprint("api", __name__, url_prefix="/api")

@lru_cache(maxsize=1)
def _load_league_bots_by_name() -> dict[str, dict]:
    """
    Loads the league json once and returns a mapping:
      bot_name -> bot_record
    """
    bots_path = os.environ.get("BOTS_JSON_PATH")
    if not bots_path:
        return {}

    data = json.loads(Path(bots_path).read_text(encoding="utf-8"))
    return {b["name"]: b for b in data}


def _resolve_model_path(raw_path: str) -> Path:
    p = Path(raw_path)

    # If this is an HPC absolute path (e.g., /nfs/.../rlcatan/...), remap into the container
    if p.is_absolute() and "rlcatan" in p.parts:
        idx = p.parts.index("rlcatan")
        p = Path("/app").joinpath(*p.parts[idx:])

    # If path is relative, treat it as relative to /app
    if not p.is_absolute():
        p = Path("/app") / p

    return p


def player_factory(player_key):
    key, color = player_key

    if player_key[0] == "CATANATRON":
        return AlphaBetaPlayer(color, 2, True)
    
    if player_key[0] == "FINAL_BOSS":
        return AlphaBetaPlacementPlayer(color, 2, True)
    
    elif player_key[0] == "VALUE_FUNCTION":
        return ValueFunctionPlayer(color, is_bot=True)
    
    elif player_key[0] == "MCTS_PLAYER":
        return MCTSPlayer(color, num_simulations=100)
    
    elif player_key[0] == "GREEDY_PLAYER":
        return GreedyPlayoutsPlayer(color, num_playouts=50)
    
    elif player_key[0] == "VP_PLAYER":
        return VictoryPointPlayer(color)
    
    elif player_key[0] == "PLACEMENT_PLAYER":
        return PlacementPlayer(color)
    
    elif player_key[0] == "WEIGHTED_RANDOM_PLAYER":
        return WeightedRandomPlayer(color)

    elif player_key[0] == "RANDOM":
        return RandomPlayer(color)

    elif player_key[0] == "HUMAN":
        return ValueFunctionPlayer(color, is_bot=False)

    # load bots by name from league.json (Their keys are expected to be in the format "BOT:bot_name")
    if isinstance(key, str) and key.startswith("BOT:"):
        bot_name = key.split(":", 1)[1]
        bots = _load_league_bots_by_name()
        bot = bots.get(bot_name)

        if bot is None:
            abort(400, description=f"Unknown bot '{bot_name}'")

        raw_path = bot.get("path")

        if not raw_path:
            abort(400, description=f"Bot '{bot_name}' has no model path")

        model_path = _resolve_model_path(raw_path)

        return PPOPlayer(color=color, model_path=str(model_path), device="cpu", deterministic=True)

    raise ValueError(f"Invalid player key: {key}")


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    if not request.is_json or request.json is None or "players" not in request.json:
        abort(400, description="Missing or invalid JSON body: 'players' key required")
    player_keys = request.json["players"]
    players = list(map(player_factory, zip(player_keys, Color)))

    game = Game(players=players)
    upsert_game_state(game)
    return jsonify({"game_id": game.id})


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    parsed_state_index = _parse_state_index(state_index)
    game = get_game_state(game_id, parsed_state_index)
    if game is None:
        abort(404, description="Resource not found")

    payload = json.dumps(game, cls=GameEncoder)
    return Response(
        response=payload,
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):
    game = get_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_color() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    # TODO: remove `or body_is_empty` when fully implement actions in FE
    body_is_empty = (not request.data) or request.json is None or request.json == {}
    if game.state.current_player().is_bot or body_is_empty:
        game.play_tick()
        upsert_game_state(game)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        upsert_game_state(game)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/stress-test", methods=["GET"])
def stress_test_endpoint():
    players = [
        AlphaBetaPlayer(Color.RED, 2, True),
        AlphaBetaPlayer(Color.BLUE, 2, True),
        AlphaBetaPlayer(Color.ORANGE, 2, True),
        AlphaBetaPlayer(Color.WHITE, 2, True),
    ]
    game = Game(players=players)
    game.play_tick()
    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route(
    "/games/<string:game_id>/states/<string:state_index>/mcts-analysis", methods=["GET"]
)
def mcts_analysis_endpoint(game_id, state_index):
    """Get MCTS analysis for specific game state."""
    logging.info(f"MCTS analysis request for game {game_id} at state {state_index}")

    # Convert 'latest' to None for consistency with get_game_state
    parsed_state_index = _parse_state_index(state_index)
    try:
        game = get_game_state(game_id, parsed_state_index)
        if game is None:
            logging.error(
                f"Game/state not found: {game_id}/{state_index}"
            )  # Use original state_index for logging
            abort(404, description="Game state not found")

        analyzer = GameAnalyzer(num_simulations=100)
        probabilities = analyzer.analyze_win_probabilities(game)

        logging.info(f"Analysis successful. Probabilities: {probabilities}")
        return Response(
            response=json.dumps(
                {
                    "success": True,
                    "probabilities": probabilities,
                    "state_index": (
                        parsed_state_index
                        if parsed_state_index is not None
                        else len(game.state.actions)
                    ),
                }
            ),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Error in MCTS analysis endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return Response(
            response=json.dumps(
                {"success": False, "error": str(e), "trace": traceback.format_exc()}
            ),
            status=500,
            mimetype="application/json",
        )


def _parse_state_index(state_index_str: str):
    """Helper function to parse and validate state_index."""
    if state_index_str == "latest":
        return None
    try:
        return int(state_index_str)
    except ValueError:
        abort(
            400,
            description="Invalid state_index format. state_index must be an integer or 'latest'.",
        )

def _load_bots():
    """
    Temporary bot source:
    - If BOTS_JSON_PATH is set and points to a JSON file, load it.
    - Otherwise return a small stub list so the UI works.
    """

    def _normalize_bot(raw):
        bot_id = raw.get("id") or raw.get("name")
        name = raw.get("name") or bot_id
        elo = raw.get("elo", 0)

        # Key is what /api/games expects in its "players" array
        key = raw.get("key")
        if not key:
            if bot_id == "random":
                key = "RANDOM"
            else:
                key = f"BOT:{bot_id}"

        return {
            "id": bot_id,
            "name": name,
            "elo": elo,
            "key": key,
            "path": raw.get("path"),
            "games": raw.get("games"),
        }

    default_bots = [
        {"id": "catanatron_ab_2", "name": "Catanatron (AlphaBeta d2)", "elo": 1500, "key": "CATANATRON"},
        {"id": "final_boss", "name": "Final Boss (AlphaBeta Placement)", "elo": 1600, "key": "FINAL_BOSS"},
        {"id": "value_function", "name": "Value Function Bot", "elo": 1400, "key": "VALUE_FUNCTION"},
        {"id": "mcts", "name": "MCTS (100 sims)", "elo": 1450, "key": "MCTS_PLAYER"},
        {"id": "greedy", "name": "Greedy Playouts (50)", "elo": 1300, "key": "GREEDY_PLAYER"},
        {"id": "vp_player", "name": "Victory Point Bot", "elo": 1200, "key": "VP_PLAYER"},
        {"id": "placement_player", "name": "Placement Only Bot", "elo": 1100, "key": "PLACEMENT_PLAYER"},
        {"id": "weighted_random", "name": "Weighted Random", "elo": 1050, "key": "WEIGHTED_RANDOM_PLAYER"},
        {"id": "random", "name": "Random", "elo": 1000, "key": "RANDOM"},
        {"id": "human", "name": "Human", "elo": None, "key": "HUMAN"},
        {"id": "ppo_v2_2026-02-07", "name": "PPO v2 (2026-02-07)", "elo": 1623, "key": "BOT:ppo_v2_2026-02-07"},
    ]

    path = os.environ.get("BOTS_JSON_PATH")

    json_bots = []
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            json_bots = [_normalize_bot(x) for x in data]
    
    return default_bots + json_bots


@bp.route("/bots", methods=("GET",))
def get_bots_endpoint():
    return jsonify(_load_bots())


# ===== Debugging Routes
# @app.route(
#     "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
# )
# def get_game_feature_vector(game_id, player_index):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     return create_sample(game, game.state.colors[player_index])


# @app.route("/games/<string:game_id>/value-function", methods=["GET"])
# def get_game_value_function(game_id):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     # model = tf.keras.models.load_model("data/models/mcts-rep-a")
#     model2 = tf.keras.models.load_model("data/models/mcts-rep-b")
#     feature_ordering = get_feature_ordering()
#     indices = [feature_ordering.index(f) for f in NUMERIC_FEATURES]
#     data = {}
#     for color in game.state.colors:
#         sample = create_sample_vector(game, color)
#         # scores = model.call(tf.convert_to_tensor([sample]))

#         inputs1 = [create_board_tensor(game, color)]
#         inputs2 = [[float(sample[i]) for i in indices]]
#         scores2 = model2.call(
#             [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
#         )
#         data[color.value] = float(scores2.numpy()[0][0])

#     return data
