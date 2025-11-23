# src/pokemon/agents/base_agent.py
"""This module defines the base classes for all agents."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Union

from pokemon.action import Action, ActionError
from pokemon.glicko2 import calculate_new_rating

if TYPE_CHECKING:
    from pokemon.battle import Battle


class BaseAgent(abc.ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str):
        """Initialize the agent.
        Args:
            name: The name of the agent.
        """
        self.name = name
        # Glicko-2 rating system parameters
        self.rating = 1500.0
        self.rating_deviation = 350.0

    @abc.abstractmethod
    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Union[Action, list[Action]]:
        """Get the action for the agent.
        This method must be implemented by subclasses.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The action(s) to take.
        """
        raise NotImplementedError

    def update_rating(self, period_results: list[tuple[float, float, float]]):
        """Update the agent's Glicko-2 rating.
        Args:
            period_results: A list of tuples, where each tuple contains
                the opponent's rating, opponent's rating deviation, and the
                outcome of the match (1 for win, 0.5 for draw, 0 for loss).
        """
        self.rating, self.rating_deviation = calculate_new_rating(
            self.rating, self.rating_deviation, period_results
        )

    @staticmethod
    def translate_action(
        action: Action,
        new_battle: Battle,
        trainer_id: int,
    ) -> Action | None:
        """Translate an action from one battle context to another.
        This is useful for simulations where battle objects are copied.
        Args:
            action: The action to translate.
            new_battle: The new battle context.
            trainer_id: The ID of the trainer performing the action.
        Returns:
            The translated action, or None if the action is None.
        Raises:
            ActionError: If an item in the action is not found in the new
                trainer's inventory.
        """
        if action is None:
            return None

        new_trainer = new_battle.get_trainer_by_id(trainer_id)
        new_opponent = new_battle.get_trainer_by_id(1 - trainer_id)

        new_item = None
        if action.item:
            new_item = new_trainer.inventory.get(action.item.name)
            if new_item is None:
                raise ActionError(
                    f"Item {action.item.name} not found in trainer "
                    f"{new_trainer.name}'s inventory"
                )

        return Action(
            action_type=action.action_type,
            trainer=new_trainer,
            opponent=new_opponent,
            move=action.move,
            target_index=action.target_index,
            item=new_item,
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class EvaluatingAgent(BaseAgent, abc.ABC):
    """Abstract base class for agents that evaluate battle states.
    Provides a common heuristic function to score a battle from a trainer's
    perspective.
    """

    def _score_trainer(self, battle: Battle, trainer_id: int) -> float:
        """Calculate a score for a single trainer based on their team's status.
        The score is a sum of:
        - The ratio of current HP to max HP for each living Pokemon.
        - A bonus for each living Pokemon.
        - A bonus for each item in the inventory.
        Args:
            battle: The battle state to score.
            trainer_id: The ID of the trainer to score.
        Returns:
            The calculated score for the trainer.
        """
        trainer = battle.get_trainer_by_id(trainer_id)
        hp_score = sum(p.hp / p.max_hp for p in trainer.pokemon_team if p.is_alive)
        alive_bonus = sum(0.5 for p in trainer.pokemon_team if p.is_alive)
        item_bonus = sum(0.2 * i.quantity for i in trainer.inventory.values())
        return hp_score + alive_bonus + item_bonus

    def _evaluate_battle(self, battle: Battle, trainer_id: int) -> float:
        """Calculate a score for the battle from a trainer's perspective.
        A positive score favors the given trainer, a negative score the opponent.
        The score is the difference between the trainer's score and the
        opponent's score.
        A large bonus/penalty is applied if the battle is won or lost.
        Args:
            battle: The battle state to evaluate.
            trainer_id: The ID of the trainer from whose perspective to evaluate.
        Returns:
            The calculated score for the battle.
        """
        if battle.winner == trainer_id:
            return 100.0
        if battle.winner is not None:
            return -100.0

        my_score = self._score_trainer(battle, trainer_id)
        opponent_score = self._score_trainer(battle, 1 - trainer_id)
        return my_score - opponent_score
