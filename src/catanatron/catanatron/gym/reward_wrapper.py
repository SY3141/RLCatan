import gymnasium as gym
from typing import Any, Dict, Optional, cast
from collections import Counter
from catanatron.gym.envs.catanatron_env import CatanatronEnv
from catanatron.gym.rlcatan_env_wrapper import RLCatanEnvWrapper
from catanatron.models.enums import ActionType

# Canonical resource names we expect to track.
TYPICAL = {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"}


def _players(state: Any) -> list[Any]:
    """
    Return list of player objects from state (robust to missing attribute).
    """
    return getattr(state, "players", []) or []


def _is_resource_name(x: Any) -> bool:
    """
    Return True if x is a string matching one of the expected resource names.
    Simplified version (we standardized on plain strings).
    """
    return isinstance(x, str) and x.upper() in TYPICAL


def _valid_resource_dict(d: Dict[Any, Any]) -> bool:
    """
    True if d looks like a resource count mapping: {RESOURCE_NAME -> int}.
    Empty dict is considered valid (treated as zero resources).
    """
    if not isinstance(d, (dict, Counter)):
        return False
    if not d:
        return True
    return all(_is_resource_name(k) and isinstance(v, int) for k, v in d.items())


def _sum(mapping: Dict[str, int]) -> int:
    """
    Sum integer counts only for recognized resource names.
    """
    return sum(int(v) for k, v in mapping.items() if k.upper() in TYPICAL)


class RewardWrapper(gym.Wrapper):
    """
    Adds reward shaping signals based on the active player's resource inventory dynamics.

    Features:
      - Tracks total resource cards held by the current active player.
      - Positive delta (gain) → gain_scale * gained resources.
      - Negative delta (spend) → spend_scale * spent resources.
      - Small flat bonus for BUILD_* actions (roads/settlements/cities).
      - Global exponential decay applied to the shaped + base reward: (decay_factor ** step_count).
      - Automatically disables shaping until the resource mapping is discovered in info/state.
      - Injects detailed reward components into the info dict for logging (e.g., tensorboard).

    Configuration (constructor arguments):
      gain_scale: multiplier for resource gains.
      spend_scale: multiplier for resource spends.
      decay_factor: per-step multiplicative decay applied to every reward.
      build_scale: flat bonus when building infrastructure.
      player_idx: optional fixed player index (else uses state's active player).
      resource_attr: key name where per-player resource dicts may appear in info/state.
      debug: toggles discovery / missing mapping prints.
    """

    def __init__(
        self,
        env,
        gain_scale: float = 0.1,
        spend_scale: float = 0.05,
        decay_factor: float = 0.999,
        build_scale: float = 0.02,
        player_idx: Optional[int] = None,
        resource_attr: str = "resource_counts",
        debug: bool = True,
    ):
        super().__init__(env)
        # Scaling and decay parameters for shaping.
        self.gain_scale = gain_scale
        self.spend_scale = spend_scale
        self.decay_factor = decay_factor
        self.build_scale = build_scale

        # Optional override of which player's resources to track; otherwise uses active player.
        self.player_idx_override = player_idx

        # Attribute/key name where resource mappings might be found (info or state).
        self.resource_attr = resource_attr
        self.debug = debug

        # State for tracking previous total and enabling/disabling shaping.
        self.last_resource_total = 0
        self.step_count = 0
        self._discovered = (
            False  # Set True once we successfully locate a resource mapping.
        )
        self._shaping_enabled = True  # Turned off if mapping not found.
        self._missing_logged = False  # Avoid spamming a missing-resource print.

    # --- Helper methods for resolving players / resources ---

    def _active_index(self, state: Any, n_players: int) -> int:
        """
        Determine active player's index using common attribute names.
        """
        # Respect explicit override to track a fixed player (e.g., a bot).
        if (
            self.player_idx_override is not None
            and 0 <= self.player_idx_override < n_players
        ):
            return self.player_idx_override

        # Use the canonical Catanatron field, fall back to 0 if missing/out of range.
        idx = getattr(state, "active_player_idx", None)
        return idx if isinstance(idx, int) and 0 <= idx < n_players else 0

    def _active_color(self) -> Any:
        """
        Retrieve the color (identifier) of the active player for indexing resource mappings.
        """
        game = cast(CatanatronEnv, self.env.unwrapped).game
        state = game.state
        players = _players(state)
        if not players:
            return None
        p = players[self._active_index(state, len(players))]
        return getattr(p, "color", None)

    def _extract_from_info(
        self, info: Dict[str, Any], active_color: Any
    ) -> Optional[int]:
        """
        Attempt to read the active player's total resources from the info dict.
        Supports colors stored directly or via .value attribute (Enum compatibility).
        """
        container = info.get(self.resource_attr)
        if not isinstance(container, dict):
            return None
        # Try possible key variants for the active player's identifier.
        keys = [active_color]
        if hasattr(active_color, "value"):
            keys.append(getattr(active_color, "value"))
        for k in keys:
            if k in container and _valid_resource_dict(container[k]):
                return _sum(container[k])
        # Fallback: if container itself is directly a resource dict.
        if _valid_resource_dict(container):
            return _sum(container)
        return None

    def _extract_from_state(self, state: Any, active_color: Any) -> Optional[int]:
        """
        Same extraction logic, but searches attributes on the state object directly
        (covers cases where resources are stored only in state, not in info).
        """
        for attr in [
            self.resource_attr,
            "resource_counts",
            "player_resource_counts",
            "resources_by_color",
            "resource_cards_by_color",
        ]:
            if hasattr(state, attr):
                container = getattr(state, attr)
                if isinstance(container, dict):
                    if active_color in container and _valid_resource_dict(
                        container[active_color]
                    ):
                        return _sum(container[active_color])
                    if _valid_resource_dict(container):
                        return _sum(container)
        return None

    def _get_resource_total(self, info: Dict[str, Any]) -> int:
        """
        Core accessor for current active player's total resource count.
        Tries info first, then state fallback. Disables shaping gracefully until mapping is found.
        """
        game = cast(CatanatronEnv, self.env.unwrapped).game
        state = game.state
        active_color = self._active_color()
        if active_color is None:
            return 0

        total = self._extract_from_info(info, active_color)
        if total is None:
            total = self._extract_from_state(state, active_color)

        # If still not found, log once and suspend shaping.
        if total is None:
            if not self._discovered and not self._missing_logged and self.debug:
                self._missing_logged = True
                self._shaping_enabled = False
                print(
                    "[ResourceRewardWrapper] Resource mapping not found; shaping off until available."
                )
            return 0

        # First successful discovery: enable shaping and optionally log.
        if not self._discovered:
            self._discovered = True
            self._shaping_enabled = True
            if self.debug:
                print(
                    f"[ResourceRewardWrapper] Resource mapping discovered; shaping on (total={total})."
                )
        return total

    # --- Gym API methods ---

    def reset(self, **kwargs):
        """
        Reset underlying env, initialize tracking variables, and prime last_resource_total.
        """
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.last_resource_total = self._get_resource_total(info or {})
        return obs, info

    def step(self, action):
        """
        Execute one environment step, then compute shaped reward components:
          1. base_reward: original env's reward.
          2. gain_reward: scaled positive delta in resource inventory.
          3. spend_reward: scaled negative delta (resource expenditure).
          4. build_bonus: flat bonus for infrastructure build actions.
          5. decay: exponential decay applied to aggregate (base + shaping).
        Stores each component in info for logging transparency.
        """
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        base_reward = float(base_reward)

        # Inventory delta (current minus previous).
        current_total = self._get_resource_total(info or {})
        delta = current_total - self.last_resource_total

        # Positive resource gain shaping.
        gain_reward = self.gain_scale * max(delta, 0) if self._shaping_enabled else 0.0
        # Negative spend shaping.
        spend_reward = (
            self.spend_scale * max(-delta, 0) if self._shaping_enabled else 0.0
        )

        # Build bonus if the action decodes to a BUILD_* action.
        build_bonus = 0.0
        try:
            action_type, _ = cast(
                CatanatronEnv, self.env.unwrapped
            ).decode_action_index(int(action))
            if action_type in {
                ActionType.BUILD_ROAD,
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
            }:
                build_bonus = self.build_scale
        except Exception:
            # Ignore decode errors silently, no bonus applied.
            pass

        # Update internal resource snapshot.
        self.last_resource_total = current_total

        # Aggregate shaping and decayed final reward.
        shaping_total = gain_reward + spend_reward + build_bonus
        raw_total = base_reward + shaping_total
        decay = self.decay_factor ** self.step_count
        final_reward = raw_total * decay

        # Populate detailed reward diagnostics.
        info["reward_original"] = base_reward
        info["reward_shaping_resource_gain"] = gain_reward
        info["reward_shaping_resource_spend"] = spend_reward
        info["reward_shaping_build_bonus"] = build_bonus
        info["reward_shaping_resource_total"] = gain_reward + spend_reward
        info["reward_decay_factor"] = decay
        info["reward_final"] = final_reward
        info["active_player_resource_total"] = current_total
        info["resource_shaping_enabled"] = self._shaping_enabled

        return obs, final_reward, terminated, truncated, info

    def get_valid_actions(self):
        """
        Delegate to the underlying RLCatanEnvWrapper for filtered, global action indices.
        """
        return cast(RLCatanEnvWrapper, self.env).get_valid_actions()
