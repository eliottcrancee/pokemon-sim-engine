# src/pokemon/agents/search.py
"""This module defines agents that use search algorithms to make decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pokemon.loguru_logger import logger

if TYPE_CHECKING:
    from pokemon.action import Action
    from pokemon.battle import Battle

from .base_agent import EvaluatingAgent


class OneStepUniformExpectimaxAgent(EvaluatingAgent):
    """Agent that evaluates actions by simulating one turn ahead.
    Uses a heuristic to score the resulting battle state after one turn.
    The action with the highest average score against all opponent responses
    is chosen.
    """

    def __init__(self, name: str = "OneStepUniformExpectimax"):
        """Initialize the agent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Choose an action by calculating the expected score.
        The score is averaged over a uniform probability distribution for the
        opponent's moves.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The action with the highest average score.
        """
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)
        avg_scores = np.zeros(len(my_actions))

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions."
            )

        for i, a0 in enumerate(my_actions):
            score = 0.0
            for a1 in op_actions:
                sim_battle = battle.copy()
                my_sim_action = self.translate_action(a0, sim_battle, trainer_id)
                op_sim_action = self.translate_action(a1, sim_battle, 1 - trainer_id)

                if trainer_id == 0:
                    sim_battle.turn(my_sim_action, op_sim_action)
                else:
                    sim_battle.turn(op_sim_action, my_sim_action)

                score += self._evaluate_battle(sim_battle, trainer_id)

                if verbose:
                    logger.info(f"Action: {a0}, Opponent Action: {a1}, Score: {score}")

            avg_scores[i] = score / len(op_actions)

        best_index = np.argmax(avg_scores)
        return my_actions[best_index]


class MinimaxAgent(EvaluatingAgent):
    """Minimax agent for Pokemon battles."""

    def __init__(self, name: str = "Minimax", depth: int = 1):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth for the Minimax algorithm.
        """
        super().__init__(name)
        self.depth = depth

    def _minimax(self, battle: Battle, trainer_id: int, current_depth: int) -> float:
        """Recursive minimax evaluation."""
        if current_depth == 0 or battle.winner is not None:
            return self._evaluate_battle(battle, trainer_id)

        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)
        best_score = -float("inf")

        for a0 in my_actions:
            min_score = float("inf")
            for a1 in op_actions:
                sim_battle = battle.copy()
                my_sim_action = self.translate_action(a0, sim_battle, trainer_id)
                op_sim_action = self.translate_action(a1, sim_battle, 1 - trainer_id)

                if trainer_id == 0:
                    sim_battle.turn(my_sim_action, op_sim_action)
                else:
                    sim_battle.turn(op_sim_action, my_sim_action)

                score = self._minimax(sim_battle, trainer_id, current_depth - 1)
                min_score = min(min_score, score)

            best_score = max(best_score, min_score)
        return best_score

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Choose an action using the Minimax algorithm.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The action that maximizes the minimum possible score.
        """
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)
        best_score = -float("inf")
        best_action = my_actions[0]

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions at depth {self.depth}."
            )

        for a0 in my_actions:
            min_score = float("inf")
            for a1 in op_actions:
                sim_battle = battle.copy()
                my_sim_action = self.translate_action(a0, sim_battle, trainer_id)
                op_sim_action = self.translate_action(a1, sim_battle, 1 - trainer_id)

                if trainer_id == 0:
                    sim_battle.turn(my_sim_action, op_sim_action)
                else:
                    sim_battle.turn(op_sim_action, my_sim_action)

                score = self._minimax(sim_battle, trainer_id, self.depth - 1)
                min_score = min(min_score, score)

                if verbose:
                    logger.info(f"Action: {a0}, Opponent Action: {a1}, Score: {score}")

            if min_score > best_score:
                best_score = min_score
                best_action = a0

        return best_action

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', depth={self.depth})"


class OneStepMinimaxAgent(MinimaxAgent):
    """Minimax agent with a depth of 1."""

    def __init__(self, name: str = "OneStepMinimax"):
        """Initialize the agent."""
        super().__init__(name, depth=1)


class TwoStepMinimaxAgent(MinimaxAgent):
    """Minimax agent with a depth of 2."""

    def __init__(self, name: str = "TwoStepMinimax"):
        """Initialize the agent."""
        super().__init__(name, depth=2)


class AlphaBetaAgent(EvaluatingAgent):
    """Minimax agent with Alpha-Beta pruning for Pokemon battles."""

    def __init__(self, name: str = "AlphaBeta", depth: int = 2):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth.
        """
        super().__init__(name)
        self.depth = depth

    def _get_max_value(
        self, battle: Battle, trainer_id: int, depth: int, alpha: float, beta: float
    ) -> float:
        """Maximizer node: Tries to maximize the score."""
        if depth == 0 or battle.winner is not None:
            return self._evaluate_battle(battle, trainer_id)

        value = -float("inf")
        my_actions = battle.get_possible_actions(trainer_id)

        for action in my_actions:
            # Pass the chosen action to the minimizer (opponent's turn)
            v = self._get_min_value(battle, action, trainer_id, depth, alpha, beta)
            value = max(value, v)
            alpha = max(alpha, value)

            # Beta Cutoff: The minimizer (opponent) has a better option elsewhere,
            # so they won't let us reach this state.
            if value >= beta:
                break

        return value

    def _get_min_value(
        self,
        battle: Battle,
        my_action: Action,
        trainer_id: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Minimizer node: Opponent tries to minimize our score."""
        value = float("inf")
        op_actions = battle.get_possible_actions(1 - trainer_id)

        for op_action in op_actions:
            sim_battle = battle.copy()
            my_sim_action = self.translate_action(my_action, sim_battle, trainer_id)
            op_sim_action = self.translate_action(op_action, sim_battle, 1 - trainer_id)

            if trainer_id == 0:
                sim_battle.turn(my_sim_action, op_sim_action)
            else:
                sim_battle.turn(op_sim_action, my_sim_action)

            # After the turn, go back to Maximizer for the next depth level
            v = self._get_max_value(sim_battle, trainer_id, depth - 1, alpha, beta)

            value = min(value, v)
            beta = min(beta, value)

            # Alpha Cutoff: We (maximizer) have a better option elsewhere,
            # so we won't choose the path leading to this bad outcome.
            if value <= alpha:
                break

        return value

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Choose an action using Minimax with Alpha-Beta pruning.
        Args:
            battle: The current battle state.
            trainer_id: The ID of the trainer for which to get the action.
            verbose: Whether to log verbose output.
        Returns:
            The best action found.
        """
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        best_score = -float("inf")
        best_action = my_actions[0]
        alpha = -float("inf")
        beta = float("inf")

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions at depth {self.depth} with Alpha-Beta."
            )

        for action in my_actions:
            # We start by calling the minimizer because we are selecting 'action',
            # and we need to know the worst-case response from the opponent.
            score = self._get_min_value(
                battle, action, trainer_id, self.depth, alpha, beta
            )

            if verbose:
                logger.info(f"Action: {action}, Score: {score}")

            if score > best_score:
                best_score = score
                best_action = action

            # Update alpha for the root level
            alpha = max(alpha, best_score)

        return best_action

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', depth={self.depth})"


class OneStepAlphaBetaAgent(AlphaBetaAgent):
    """AlphaBeta agent with a depth of 1."""

    def __init__(self, name: str = "OneStepAlphaBeta"):
        super().__init__(name, depth=1)


class TwoStepAlphaBetaAgent(AlphaBetaAgent):
    """AlphaBeta agent with a depth of 2."""

    def __init__(self, name: str = "TwoStepAlphaBeta"):
        super().__init__(name, depth=2)


class ThreeStepAlphaBetaAgent(AlphaBetaAgent):
    """AlphaBeta agent with a depth of 3."""

    def __init__(self, name: str = "ThreeStepAlphaBeta"):
        super().__init__(name, depth=3)
