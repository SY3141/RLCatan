import os
import signal

from sb3_contrib.ppo_mask import MaskablePPO

from league import League, LeagueMember
from continue_train_ppo import make_env
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.ppo_player import PPOPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.cli.accumulators import StatisticsAccumulator
from catanatron.game import Game

"""
League-based training loop to allow for self-play.
"""

STOP_REQUESTED = False

def _handle_stop(signum, frame):
    """Signal handler to request to stop the training loop."""
    print("Signal received, requesting stop. This may take a couple of minutes to finish the current phase and save.")
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_opponent(member: LeagueMember):
    """
    Build an opponent Player from a LeagueMember.
    """

    if member.name == "random":
        return RandomPlayer(Color.RED)
    if member.name == "vf":
        return ValueFunctionPlayer(Color.RED)

    # Snapshot PPO opponent
    return PPOPlayer(Color.RED, model_path=member.path, deterministic=True)


def eval_vs_opponent(main_player, opp_player, n_games: int = 20, seed_base: int = 0) -> float:
    """
    As Catan is stochastic, elo is updated based on winrate over n_games (one phase).
    This function runs n_games between main_player and opp_player, returning the winrate of main_player.
    This does mean running games that don't contribute to training, but otherwise Elo updates would be too noisy.

    Note that this function doesn't restrict the action space, despite the fact that they were trained with action masking.
    This might lead to issues but elo won't be as accurate for the true game otherwise.
    We can monitor this and see how well it works in practice.
    """

    # Reusing the existing StatisticsAccumulator to track wins
    stats = StatisticsAccumulator()

    for i in range(n_games):
        game = Game([main_player, opp_player], seed=seed_base + i)
        game.play(accumulators=[stats])

    total_tracked = sum(stats.wins.values())

    # Avoid division by zero
    if total_tracked == 0:
        return 0.0

    return stats.wins[main_player.color] / total_tracked


def main():
    """
    Main training loop for league-based training.
    1. Ensure league and baselines exist
    2. For each phase:
       a. Sample opponent from league
       b. Train main PPO against opponent for phase_steps
       c. Snapshot main PPO and add to league
       d. Evaluate main PPO vs opponent to update Elo
       e. Periodically prune league
    3. Repeat from step 2
    """

    # Setup signal handlers for graceful stopping
    signal.signal(signal.SIGTERM, _handle_stop) # Runs on termination signal (e.g. Ctrl+C)
    signal.signal(signal.SIGINT, _handle_stop)  # Runs on interrupt signal (e.g. kill command, server shutdown)

    # Find important paths and ensure league directory exists under models
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.normpath(os.path.join(base_dir, "..", "models"))
    league_dir = os.path.join(models_dir, "league")
    os.makedirs(league_dir, exist_ok=True)

    league_path = os.path.join(league_dir, "league.json")
    league = League(league_path)

    # Ensure baselines exist in league
    league.ensure_member("random", path=None, elo=800.0)
    league.ensure_member("vf", path=None, elo=1200.0)

    main_name = "main"
    main_model_path = os.path.join(models_dir, "ppo_v2.zip") # Hardcoded main model path. May want to change later
    league.ensure_member(main_name, path=main_model_path, elo=1000.0)

    # Load main PPO model
    device = "cuda"
    env = make_env(seed=42)

    model = MaskablePPO.load(main_model_path, env=env, device=device)

    # Training parameters (may want to adjust later)
    phase_steps = 50_000
    eval_games = 10

    phase = 0

    while True:
        # Note that this only checks for stop requests between phases to keep things safe
        if STOP_REQUESTED:
            print("Stop complete. Program will now exit.")
            break

        phase += 1

        # Find an opponent with similar Elo
        main_elo = league.get(main_name).elo

        opp_member = league.sample_opponent_biased_for_baseline(
            target_elo=main_elo,
            p_vf=0.2,
            p_random=0.05,
            temperature=150.0,
        )
        opponent_player = build_opponent(opp_member)

        # New env with this opponent (reusing make_env from continue_train_ppo)
        env = make_env(seed=42 + phase, enemies=[opponent_player])
        model.set_env(env)

        model.learn(total_timesteps=phase_steps, reset_num_timesteps=False)

        # Snapshot
        snap_name = f"snap_{model.num_timesteps}"
        snap_path = os.path.join(league_dir, snap_name)
        model.save(snap_path)

        # Add the new snapshot to the league and load it for evaluation
        league.add_member(LeagueMember(name=snap_name, path=snap_path, elo=main_elo))
        main_player = PPOPlayer(Color.BLUE, model_path=snap_path, deterministic=True)

        # Evaluate main vs opponent to update Elo
        winrate = eval_vs_opponent(main_player, opponent_player, n_games=eval_games, seed_base=2000 + phase)
        a_score = float(winrate)  # Elo expects 0..1
        league.update(a_name=main_name, b_name=opp_member.name, a_score=a_score)

        # Periodic pruning
        if phase % 20 == 0:
            removed = league.prune(max_members=120, keep_recent=30, keep_top=15)

            # Remove pruned snapshot files from the disk to keep things clean
            for snap in removed:
                if snap.path and os.path.exists(snap.path + ".zip") and snap.name.startswith("snap_"):
                    os.remove(snap.path + ".zip")

        # Always keep main pointing to the latest weights
        model.save(os.path.join(models_dir, "ppo_v2"))
        league.get(main_name).path = os.path.join(models_dir, "ppo_v2.zip")
        league.save()

if __name__ == "__main__":
    main()