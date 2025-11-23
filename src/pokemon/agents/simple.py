# src/pokemon/agents/simple.py
"""This module defines simple agents for the Pokemon battle game."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Union

from pokemon.action import Action, ActionType

if TYPE_CHECKING:
    from pokemon.battle import Battle

from .base_agent import BaseAgent


class InputAgent(BaseAgent):
    """Agent that requires user input for actions."""

    def __init__(self, name: str = "Input"):
        """Initialize the InputAgent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        """Return all possible actions to be displayed to the user.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            A list of all possible actions.
        """
        return battle.get_possible_actions(trainer_id)


class FirstAgent(BaseAgent):
    """Agent that always selects the first available action."""

    def __init__(self, name: str = "First"):
        """Initialize the FirstAgent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the first possible action.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The first action in the list of possible actions.
        """
        return battle.get_possible_actions(trainer_id)[0]


class RandomAgent(BaseAgent):
    """Agent that selects a random action from the available actions."""

    def __init__(self, name: str = "Random"):
        """Initialize the RandomAgent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random action.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            A randomly chosen action from the list of possible actions.
        """
        actions = battle.get_possible_actions(trainer_id)
        return random.choice(actions)


class RandomAttackAgent(BaseAgent):
    """Agent that selects a random attack action from the available actions."""

    def __init__(self, name: str = "RandomAttack"):
        """Initialize the RandomAttackAgent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random attack action.
        If no attack actions are available, a random action from all possible
        actions is returned instead.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            A randomly chosen attack action.
        """
        actions = battle.get_possible_actions(trainer_id)
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)
        return random.choice(attack_actions)
