import gymnasium as gym
from typing import cast
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper


class ResourceRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_scale=0.1, decay_factor=0.999):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.decay_factor = decay_factor
        self.last_resource_count = 0
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        catan_env = cast(CatanatronEnv, self.env.unwrapped)
        game = catan_env.game

        current_player = game.state.current_player
        self.last_resource_count = sum(current_player.resources.values())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        reward = float(reward)

        catan_env = cast(CatanatronEnv, self.env.unwrapped)
        game = catan_env.game

        current_player = game.state.current_player
        current_resource_count = sum(current_player.resources.values())

        delta = current_resource_count - self.last_resource_count
        resource_shaping = 0.0
        if delta > 0:
            resource_shaping = delta * self.reward_scale

        self.last_resource_count = current_resource_count

        raw_total_reward = reward + resource_shaping
        time_decay = self.decay_factor ** self.step_count
        final_reward = raw_total_reward * time_decay

        info['reward_original'] = reward
        info['reward_shaping_resource'] = resource_shaping
        info['reward_decay_factor'] = time_decay

        return obs, final_reward, terminated, truncated, info

    def get_valid_actions(self):
        rl_env = cast(RLCatanEnvWrapper, self.env)
        return rl_env.get_valid_actions()