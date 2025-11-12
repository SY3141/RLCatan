import random

import gymnasium
from gymnasium.utils.env_checker import check_env
import numpy as np

from catanatron.features import get_feature_ordering
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.gym.envs.catanatron_env import CatanatronEnv

features = get_feature_ordering(2)


def get_p0_num_settlements(obs):
    indexes = [
        i
        for i, name in enumerate(features)
        if "NODE" in name and "SETTLEMENT" in name and "P0" in name
    ]
    return sum([obs[i] for i in indexes])


def test_check_env():
    """
    Test that the CatanatronEnv adheres to the Gym API:
    https://gymnasium.farama.org/tutorials/training_agents/1_check_env/
    """

    env = CatanatronEnv()
    check_env(env)


def test_gym():
    """
    Test basic Gym API functionality of CatanatronEnv.
    1. Reset environment and check initial observation and info.
    2. Take a random valid action and check resulting observation, reward, done flags, and info.
    3. Reset environment again and verify observation is different from previous step.
    4. Check starting resources in observation.
    """

    env = CatanatronEnv()

    first_observation, info = env.reset()  # this forces advanced until p0...
    assert len(info["valid_actions"]) >= 50  # first seat at most blocked 4 nodes
    assert get_p0_num_settlements(first_observation) == 0

    action = random.choice(info["valid_actions"])
    second_observation, reward, terminated, truncated, info = env.step(action)
    assert np.any(first_observation != second_observation)
    assert reward == 0
    assert not terminated
    assert not truncated
    assert len(info["valid_actions"]) in [2, 3]

    assert second_observation[features.index("BANK_DEV_CARDS")] == 25  # type: ignore
    assert second_observation[features.index("BANK_SHEEP")] == 19  # type: ignore
    assert get_p0_num_settlements(second_observation) == 1

    reset_obs, _ = env.reset()
    assert np.any(reset_obs != second_observation)
    assert get_p0_num_settlements(reset_obs) == 0

    env.close()


def test_gym_registration_and_api_works():
    """
    Test that the Catanatron-v0 environment is properly registered
    and can be created via gymnasium.make. Also test basic Gym API functionality.
    1. Create environment using gymnasium.make.
    2. Reset environment and check initial observation and info.
    3. Take random actions until the episode ends, checking reward at the end.
    4. Ensure final reward is either -1 (loss) or 1 (win).
    """

    env = gymnasium.make("catanatron/Catanatron-v0")
    observation, info = env.reset()
    done = False
    reward = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    assert reward in [-1, 1] # These are the default terminal rewards


def test_invalid_action_reward():
    """
    Test that the environment correctly handles invalid actions
    by returning a custom invalid action reward and not terminating the episode.
    1. Create environment with custom invalid action reward.
    2. Reset environment and get initial observation and info.
    3. Identify an invalid action and take that action.
    4. Verify that the reward matches the custom invalid action reward,
       the episode is not terminated or truncated, and the observation remains unchanged.
    5. Repeat the invalid action multiple times to ensure consistent behavior,
       and verify that the episode eventually truncates after a set number of invalid actions.
    """

    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"invalid_action_reward": -1234}
    )
    first_obs, info = env.reset()
    invalid_action = next(filter(lambda i: i not in info["valid_actions"], range(1000)))
    observation, reward, terminated, truncated, info = env.step(invalid_action)
    assert reward == -1234
    assert not terminated
    assert not truncated
    assert (observation == first_obs).all()
    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(invalid_action)
        assert (observation == first_obs).all()
    assert not terminated
    assert truncated


def test_custom_reward():
    """
    Test that the environment correctly uses a custom reward function.
    1. Define a custom reward function that returns a fixed value.
    2. Create environment with the custom reward function.
    3. Reset environment and get initial observation and info.
    4. Take a random valid action.
    5. Verify that the reward returned matches the value from the custom reward function.
    """

    def custom_reward(game, p0_color):
        return 123

    env = gymnasium.make(
        "catanatron/Catanatron-v0", config={"reward_function": custom_reward}
    )
    observation, info = env.reset()
    action = random.choice(info["valid_actions"])
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 123


def test_custom_map():
    """
    Test that the environment can be created with a custom map configuration.
    1. Create environment with a specified custom map type.
    2. Reset environment and get initial observation and info.
    3. Verify that the number of valid actions and the size of the observation
       match expected values for the custom map.
    """

    env = gymnasium.make("catanatron/Catanatron-v0", config={"map_type": "MINI"})
    observation, info = env.reset()
    assert len(info["valid_actions"]) < 50 # MINI map has fewer actions
    assert len(observation) < 614
    # assert env.action_space.n == 260


def test_enemies():
    """
    Test that the environment can be created with custom enemy players
    and that the game runs to completion with those enemies.
    1. Create environment with specified enemy players.
    2. Reset environment and get initial observation and info.
    3. Verify that the observation size matches expected feature ordering.
    4. Take random valid actions until the episode ends.
    5. Verify that the expected enemy player wins and the reward reflects a loss.
    """

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [
                ValueFunctionPlayer(Color.RED), # When we make a better bot we could replace this to reduce flakiness
                RandomPlayer(Color.ORANGE),
                RandomPlayer(Color.WHITE),
            ]
        },
    )
    observation, info = env.reset()
    assert len(observation) == len(get_feature_ordering(4))

    done = False
    reward = 0
    while not done:
        action = random.choice(info["valid_actions"])
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Virtually impossible for a Random bot to beat Value Function Player
    assert env.unwrapped.game.winning_color() == Color.RED  # type: ignore
    assert reward == -1  # ensure we lost
    env.close()


def test_mixed_rep():
    """
    Test that the environment can be created with mixed representation
    and that the observation contains both board and numeric components.
    1. Create environment with mixed representation configuration.
    2. Reset environment and get initial observation and info.
    3. Verify that the observation contains both 'board' and 'numeric' keys.
    """

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config={"representation": "mixed"},
    )
    observation, info = env.reset()
    assert "board" in observation # Image/tensor part
    assert "numeric" in observation # Feature vector part

    env.close()


def test_env_runs_with_masked_random_policy():
    """
    Smoke test to ensure the environment runs correctly
    when using a masked random policy that only selects valid actions.
    1. Create environment.
    2. Reset environment and get initial observation and info.
    3. Continuously select random valid actions until the episode ends.
    4. Ensure no errors occur during the process.
    """

    import numpy as np

    env = gymnasium.make(
        "catanatron/Catanatron-v0"
    )

    obs, info = env.reset()
    done = False

    while not done:
        valid = np.array(info["valid_actions"])
        action = np.random.choice(valid)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()