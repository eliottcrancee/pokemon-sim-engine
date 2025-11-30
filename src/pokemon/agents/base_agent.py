import abc

from pokemon.action import Action
from pokemon.battle import Battle
from pokemon.glicko2 import calculate_new_rating


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
    ) -> Action:
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

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
