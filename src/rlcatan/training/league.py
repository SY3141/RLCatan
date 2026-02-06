# league.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import json, os, random
from typing import List, Optional

"""
A league system for managing AI opponents with Elo ratings.
"""

@dataclass
class LeagueMember:
    name: str
    path: Optional[str]  # None for built-in baselines like Random and VF
    elo: float = 1000.0  # Starting Elo
    games: int = 0       # Number of games played

class League:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.members: List[LeagueMember] = []

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            self.load()

    def add_member(self, m: LeagueMember) -> None:
        """Add a new member to the league and save."""
        self.members.append(m)
        self.save()

    def get(self, name: str) -> LeagueMember:
        """Get a member by name."""
        return next(m for m in self.members if m.name == name)

    def ensure_member(self, name: str, path: Optional[str], elo: float = 1000.0) -> None:
        """Ensure a member exists; if not, add them."""
        if any(m.name == name for m in self.members):
            return

        self.add_member(LeagueMember(name=name, path=path, elo=elo))

    def expected(self, ra: float, rb: float) -> float:
        """Calculate the expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, a_name: str, b_name: str, a_score: float, k: float = 16.0) -> None:
        """Update Elo ratings after a match between player A and B."""
        a = next(m for m in self.members if m.name == a_name)
        b = next(m for m in self.members if m.name == b_name)

        ea = self.expected(a.elo, b.elo)
        eb = self.expected(b.elo, a.elo)

        a.elo += k * (a_score - ea)
        b.elo += k * ((1.0 - a_score) - eb)

        a.games += 1
        b.games += 1

        self.save()

    def sample_opponent(self, target_elo: float, temperature: float = 150.0) -> LeagueMember:
        """Sample an opponent based on target Elo using a softmax distribution."""
        # softmax over -distance
        weights = []

        for m in self.members:
            d = abs(m.elo - target_elo)
            w = pow(2.71828, -d / temperature)
            weights.append(w)

        return random.choices(self.members, weights=weights, k=1)[0]

    def sample_opponent_biased_for_baseline(
            self,
            target_elo: float,
            p_vf: float = 0.2, # So essentially 20% vf, 5% random, 75% elo-near
            p_random: float = 0.05,
            temperature: float = 150.0, # temperature controls the chance of sampling further from target elo
    ) -> LeagueMember:
        """Sample an opponent with bias towards baselines."""
        r = random.random()

        if r < p_random:
            return self.get("random")
        if r < p_random + p_vf:
            return self.get("vf")

        return self.sample_opponent(target_elo=target_elo, temperature=temperature)

    def save(self) -> None:
        """Save the league members to the JSON file."""
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.members], f, indent=2)

    def load(self) -> None:
        """Load the league members from the JSON file."""
        with open(self.save_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.members = [LeagueMember(**m) for m in raw]

    def prune(
            self,
            max_members: int = 120,
            keep_recent: int = 30,
            keep_top: int = 15,
    ) -> List[LeagueMember]:
        """
        Bound league size while keeping strong + recent + baselines:
        1. Always keep baselines (currently "random" and "vf").
        2. Keep the most recent `keep_recent` snapshots."
        3. Keep the top `keep_top` snapshots by Elo.
        4. If still over `max_members`, drop the lowest Elo from the remaining.

        Returns the list of removed LeagueMembers so their files can be cleaned up.
        """

        # Early exit if already within limits
        if len(self.members) <= max_members:
            return []

        # Define baseline names #TODO make configurable once we're beating vf
        baseline_names = {"main", "random", "vf"}

        # Separate baselines and snapshots
        baselines = [m for m in self.members if m.name in baseline_names]
        snaps = [m for m in self.members if m.name not in baseline_names]

        # Helper to extract the snapshot step from its name
        def _snap_step(name: str) -> int:
            if name.startswith("snap_"):
                try:
                    return int(name.split("_", 1)[1])
                except ValueError:
                    pass

            print("Warning: Unexpected snapshot name format:", name)
            return -1

        # Sort snapshots by recency (step number)
        snaps_sorted_by_recency = sorted(snaps, key=lambda m: _snap_step(m.name))

        # Find the most recent and top snapshots. These are safe from being pruned.
        recent = snaps_sorted_by_recency[-keep_recent:] if len(
            snaps_sorted_by_recency) > keep_recent else snaps_sorted_by_recency

        top = sorted(snaps, key=lambda m: m.elo, reverse=True)[:keep_top]

        keep = {m.name: m for m in baselines + recent + top}

        # If still too many, trim the middle (not baseline, not top, not recent) by lowest Elo first
        # This only occurs if keep_recent + keep_top + baselines > max_members. Not sure if that's something
        # we'll want to do, but I'm adding this to be safe.
        kept_list = list(keep.values())

        if len(kept_list) > max_members:
            protected = set(m.name for m in baselines + recent + top)
            extras = [m for m in kept_list if m.name not in protected]
            extras_sorted = sorted(extras, key=lambda m: m.elo)
            drop_count = len(kept_list) - max_members
            drop_names = set(m.name for m in extras_sorted[:drop_count])
            kept_list = [m for m in kept_list if m.name not in drop_names]

        kept_names = set(m.name for m in kept_list)
        removed = [m for m in self.members if m.name not in kept_names]

        self.members = kept_list
        self.save()

        return removed