from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ResourceLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in TensorBoard.
    It accumulates values over steps and logs the MEAN value.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.resource_rewards = []
        self.original_rewards = []
        self.decay_factors = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:

            # Collect data if present
            if "reward_shaping_resource" in info:
                self.resource_rewards.append(info["reward_shaping_resource"])

            if "reward_original" in info:
                self.original_rewards.append(info["reward_original"])

            if "reward_decay_factor" in info:
                self.decay_factors.append(info["reward_decay_factor"])

        # Log every 100 steps to avoid slowing down training and to make the graph readable
        if self.n_calls % 100 == 0:
            if self.resource_rewards:
                self.logger.record("custom/resource_shaping_mean", np.mean(self.resource_rewards))
                self.resource_rewards = []  # Reset buffer

            if self.original_rewards:
                self.logger.record("custom/original_reward_mean", np.mean(self.original_rewards))
                self.original_rewards = []  # Reset buffer

            if self.decay_factors:
                self.logger.record("custom/decay_factor_mean", np.mean(self.decay_factors))
                self.decay_factors = []  # Reset buffer

        return True