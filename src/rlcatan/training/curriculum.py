from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import json


class PhaseConfig(TypedDict, total=False):
    name: str
    target_vp: int
    disabled_action_types: List[str]
    min_episodes: int
    advance_threshold: Dict[str, Any]


class CurriculumManager:
    """Simple curriculum manager that steps through ordered phases.

    Phase structure (JSON):
    {
      "phases": [
         {"name":"phase-1","target_vp":3,"disabled_action_types":["MOVE_ROBBER","BUY_DEVELOPMENT_CARD"],"min_episodes":100,"advance_threshold":{"metric":"win_rate","value":0.8}}
      ]
    }
    """

    def __init__(self, phases: List[PhaseConfig], cycle: bool = False) -> None:
        assert phases, "curriculum must contain at least one phase"
        self.phases = phases
        self.index = 0
        self.cycle = cycle

    @classmethod
    def from_json(cls, path: str) -> "CurriculumManager":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        phases = data.get("phases", [])
        return cls(phases)

    def current_phase(self) -> PhaseConfig:
        return self.phases[self.index]

    def set_phase(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.phases):
            raise IndexError("phase index out of range")
        self.index = idx

    def advance_if_needed(self, metrics: Dict[str, float], episodes: int = 0) -> bool:
        """Advance to next phase based on provided metrics.

        metrics: dict containing metrics such as 'win_rate' or 'avg_vp'.
        episodes: total episodes seen so far (used for min_episodes checks).

        Returns True if advanced.
        """
        phase = self.current_phase()
        thr = phase.get("advance_threshold")
        min_eps = phase.get("min_episodes", 0)
        if thr is None:
            return False
        if episodes < min_eps:
            return False
        metric = thr.get("metric")
        value = thr.get("value")
        if metric is None or value is None:
            return False
        current = metrics.get(metric)
        if current is None:
            return False
        if current >= value:
            # advance
            next_idx = self.index + 1
            if next_idx >= len(self.phases):
                if self.cycle:
                    self.index = 0
                    return True
                return False
            self.index = next_idx
            return True
        return False


__all__ = ["CurriculumManager", "PhaseConfig"]
