import random

from pokemon.action import Action, ActionType
from pokemon.battle import Battle

from .base_agent import BaseAgent


class InputAgent(BaseAgent):
    """Agent that requires user input for actions."""

    def __init__(self, name: str = "Input"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        """Return all possible actions to be displayed to the user."""
        return battle.get_possible_actions(trainer_id)


class FirstAgent(BaseAgent):
    """Agent that always selects the first available action."""

    def __init__(self, name: str = "First"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return the first possible action."""
        return battle.get_possible_actions(trainer_id)[0]


class RandomAgent(BaseAgent):
    """Agent that selects a random action from the available actions."""

    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random action."""
        actions = battle.get_possible_actions(trainer_id)
        return random.choice(actions)


class RandomAttackAgent(BaseAgent):
    """Agent that selects a random attack action from the available actions."""

    def __init__(self, name: str = "RandomAttack"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Return a random attack action.
        If no attack actions are available, a random action from all possible actions is returned.
        """
        actions = battle.get_possible_actions(trainer_id)
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)
        return random.choice(attack_actions)
