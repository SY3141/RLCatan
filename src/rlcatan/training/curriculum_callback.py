"""Curriculum learning callback for progressive training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable, Any
import gymnasium as gym

from .curriculum import CurriculumManager


class CurriculumCallback(BaseCallback):
    """
    Callback that advances curriculum phases based on agent performance.

    This callback monitors training metrics (e.g., win rate, average reward) and
    automatically transitions to the next curriculum phase when thresholds are met.
    """

    def __init__(
        self,
        curriculum: CurriculumManager,
        env_factory: Callable[[list[str] | None, dict[str, Any] | None], gym.Env],
        eval_freq: int = 2048,
        verbose: int = 1,
    ):
        """
        Args:
            curriculum: CurriculumManager instance with defined phases
            env_factory: Function that creates a new environment with disabled actions
            eval_freq: How often to check for phase advancement (in timesteps)
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.curriculum = curriculum
        self.env_factory = env_factory
        self.eval_freq = eval_freq
        self.episode_count = 0
        self.last_eval_step = 0
        self.phase_start_step = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Only evaluate at specified frequency
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps

        # Check if we have enough episode data
        if len(self.model.ep_info_buffer) == 0:
            return True

        # Calculate metrics from recent episodes (last 100)
        recent_episodes = list(self.model.ep_info_buffer)

        # Win rate: count episodes with positive reward
        win_rate = np.mean([ep["r"] > 0 for ep in recent_episodes])

        # Average reward
        avg_reward = np.mean([ep["r"] for ep in recent_episodes])

        # Average episode length
        avg_ep_len = np.mean([ep["l"] for ep in recent_episodes])

        # Average victory points (if available in info)
        avg_vp = np.mean([ep.get("vp", 0) for ep in recent_episodes if "vp" in ep])

        metrics = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_vp": avg_vp,
            "avg_ep_len": avg_ep_len,
        }

        # Calculate episodes since phase start
        episodes_in_phase = len(recent_episodes)  # Approximate

        current_phase = self.curriculum.current_phase()

        if self.verbose >= 2:
            print(f"\n[Curriculum] Phase: {current_phase.get('name')}")
            print(
                f"[Curriculum] Metrics: win_rate={win_rate:.3f}, avg_reward={avg_reward:.3f}, avg_vp={avg_vp:.1f}"
            )

        # Try to advance to next phase
        if self.curriculum.advance_if_needed(metrics, episodes_in_phase):
            new_phase = self.curriculum.current_phase()

            if self.verbose >= 1:
                print(f"\n{'='*60}")
                print(f"CURRICULUM PHASE ADVANCED!")
                print(f"{'='*60}")
                print(f"Previous phase: {current_phase.get('name')}")
                print(f"New phase: {new_phase.get('name')}")
                print(f"Target VP: {new_phase.get('target_vp')}")
                print(f"Disabled actions: {new_phase.get('disabled_action_types', [])}")
                print(f"Metrics at transition: {metrics}")
                print(f"{'='*60}\n")

            # Rebuild environment with new phase settings
            try:
                disabled_actions = new_phase.get("disabled_action_types", [])
                reward_shaping = new_phase.get("reward_shaping")
                new_env = self.env_factory(disabled_actions, reward_shaping)
                self.model.set_env(new_env)
                self.phase_start_step = self.num_timesteps

                if self.verbose >= 1:
                    print(f"[Curriculum] Environment updated with new phase settings")
            except Exception as e:
                print(f"[Curriculum] ERROR: Failed to update environment: {e}")
                return False

        return True

    def _on_training_start(self) -> None:
        """Called when training starts."""
        phase = self.curriculum.current_phase()
        if self.verbose >= 1:
            print(f"\n[Curriculum] Starting training with phase: {phase.get('name')}")
            print(f"[Curriculum] Target VP: {phase.get('target_vp')}")
            print(
                f"[Curriculum] Disabled actions: {phase.get('disabled_action_types', [])}\n"
            )

    def _on_training_end(self) -> None:
        """Called when training ends."""
        phase = self.curriculum.current_phase()
        if self.verbose >= 1:
            print(f"\n[Curriculum] Training ended at phase: {phase.get('name')}")
            print(f"[Curriculum] Total timesteps: {self.num_timesteps}\n")


__all__ = ["CurriculumCallback"]
