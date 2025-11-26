from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ResourceLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.original = []
        self.gain = []
        self.spend = []
        self.total = []
        self.final = []
        self.decay = []
        self.inv = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward_original" in info: self.original.append(info["reward_original"])
            if "reward_shaping_resource_gain" in info: self.gain.append(info["reward_shaping_resource_gain"])
            if "reward_shaping_resource_spend" in info: self.spend.append(info["reward_shaping_resource_spend"])
            if "reward_shaping_resource_total" in info: self.total.append(info["reward_shaping_resource_total"])
            if "reward_final" in info: self.final.append(info["reward_final"])
            if "reward_decay_factor" in info: self.decay.append(info["reward_decay_factor"])
            if "active_player_resource_total" in info: self.inv.append(info["active_player_resource_total"])

        if self.n_calls % 100 == 0:
            if self.original: self.logger.record("custom/original_reward_mean", np.mean(self.original)); self.original=[]
            if self.gain: self.logger.record("custom/resource_gain_mean", np.mean(self.gain)); self.gain=[]
            if self.spend: self.logger.record("custom/resource_spend_mean", np.mean(self.spend)); self.spend=[]
            if self.total: self.logger.record("custom/resource_total_mean", np.mean(self.total)); self.total=[]
            if self.final: self.logger.record("custom/final_reward_mean", np.mean(self.final)); self.final=[]
            if self.decay: self.logger.record("custom/decay_factor_mean", np.mean(self.decay)); self.decay=[]
            if self.inv: self.logger.record("custom/resource_inventory_mean", np.mean(self.inv)); self.inv=[]
        return True
