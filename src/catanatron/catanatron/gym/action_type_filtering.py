from __future__ import annotations
from typing import Iterable
from catanatron.models.enums import ActionType

"""
Utilities for filtering action types, e.g. to disable trading or development cards.
Can add more set-based filters here as needed and experiment with different filters.
"""

# These are just some example groups to exclude; we can add more.
# For v1 I'll just exclude complex dev cards and player trading.
SIMPLE_DEV_CARD_ACTION_TYPES: set[ActionType] = {
    ActionType.BUY_DEVELOPMENT_CARD,  # Victory point cards are played automatically.
    ActionType.PLAY_KNIGHT_CARD,
}

COMPLEX_DEV_CARD_ACTION_TYPES: set[ActionType] = {
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
    ActionType.PLAY_MONOPOLY,
}

BANK_TRADING_ACTION_TYPES: set[ActionType] = {
    ActionType.MARITIME_TRADE,
}

PLAYER_TRADING_ACTION_TYPES: set[ActionType] = {
    ActionType.OFFER_TRADE,
    ActionType.ACCEPT_TRADE,
    ActionType.REJECT_TRADE,
    ActionType.CONFIRM_TRADE,
    ActionType.CANCEL_TRADE,
}

# Not sure if it's safe to exclude this one since it may be forced by 7 rolls and knights
# ROBBER_ACTION_TYPES: set[ActionType] = {
#     ActionType.MOVE_ROBBER,
# }


def filter_action_types(
    env, indices: Iterable[int], excluded_types: Iterable[Iterable[ActionType]]
) -> list[int]:
    """
    Given an env that implements decode_action_index(action_int) -> (ActionType, value),
    an iterable of action indices, and an iterable of sets of ActionTypes to exclude,
    returns a filtered list of action indices excluding any whose ActionType is in any of the excluded sets.
    """
    filtered: list[int] = []

    for idx in indices:
        filter_action = False
        action_type, _ = env.decode_action_index(idx)

        for excluded_set in excluded_types:
            if action_type in excluded_set:
                filter_action = True
                break

        if filter_action:
            continue

        filtered.append(idx)

    return filtered
