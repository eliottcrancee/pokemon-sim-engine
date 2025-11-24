# src/pokemon/agents/heuristic.py
"""This module defines agents that use heuristics to make decisions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from pokemon.action import Action, ActionType

if TYPE_CHECKING:
    from pokemon.battle import Battle

from .base_agent import BaseAgent


def _get_best_heal_action(actions: list[Action]) -> Optional[Action]:
    """Return the best healing action from a list of actions.
    It is assumed that higher item IDs correspond to better potions.
    Args:
        actions: A list of possible actions.
    Returns:
        The best healing action, or None if no healing actions are available.
    """
    heal_actions = [
        a
        for a in actions
        if a.action_type == ActionType.USE_ITEM
        and a.item is not None
        and "Potion" in a.item.name
    ]
    if not heal_actions:
        return None
    return max(heal_actions, key=lambda a: a.item.item_id)


class RandomAttackAndPotionAgent(BaseAgent):
    """Agent that randomly attacks, but uses a potion if HP is low."""

    def __init__(
        self, name: str = "RandomAttackAndPotion", heal_threshold: float = 0.2
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            heal_threshold: The HP percentage below which the agent will try to
                use a healing item.
        """
        name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random attack or a potion action.
        - If HP is below `heal_threshold`, uses a random available "Potion" item.
        - Otherwise, performs a random attack.
        - If no attack is possible, performs a random action.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The chosen action.
        """
        trainer = battle.get_trainer_by_id(trainer_id)
        current_pokemon = trainer.pokemon_team[0]
        actions = battle.get_possible_actions(trainer_id)

        if current_pokemon.hp / current_pokemon.max_hp < self.heal_threshold:
            heal_actions = [
                a
                for a in actions
                if a.action_type == ActionType.USE_ITEM
                and a.item is not None
                and "Potion" in a.item.name
            ]
            if heal_actions:
                return random.choice(heal_actions)

        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)
        return random.choice(attack_actions)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"heal_threshold={self.heal_threshold})"
        )


class BestAttackAgent(BaseAgent):
    """Agent that selects the attack action with the highest power."""

    def __init__(self, name: str = "BestAttack"):
        """Initialize the agent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the attack action with the highest base power.
        If no attack is possible, returns a random action.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The chosen action.
        """
        actions = battle.get_possible_actions(trainer_id)
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        return max(attack_actions, key=lambda a: a.move.power if a.move else 0)


class BestAttackAndPotionAgent(BaseAgent):
    """Agent that uses the best attack, but uses a potion if HP is low."""

    def __init__(self, name: str = "BestAttackAndPotion", heal_threshold: float = 0.2):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            heal_threshold: The HP percentage below which the agent will try
                to use a healing item.
        """
        name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the best attack or a potion action.
        - If HP is below `heal_threshold`, uses the best available "Potion" item.
        - Otherwise, performs the attack with the highest base power.
        - If no attack is possible, performs a random action.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The chosen action.
        """
        trainer = battle.get_trainer_by_id(trainer_id)
        current_pokemon = trainer.pokemon_team[0]
        actions = battle.get_possible_actions(trainer_id)

        if current_pokemon.hp / current_pokemon.max_hp < self.heal_threshold:
            heal_action = _get_best_heal_action(actions)
            if heal_action:
                return heal_action

        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        return max(attack_actions, key=lambda a: a.move.power if a.move else 0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"heal_threshold={self.heal_threshold})"
        )
