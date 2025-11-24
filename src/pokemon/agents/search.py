# src/pokemon/agents/search.py
"""This module defines agents that use search algorithms to make decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed

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

    def __init__(
        self, name: str = "OneStepUniformExpectimax", parallelize: bool = False
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
        """
        super().__init__(name)
        self.parallelize = parallelize

    def _calculate_action_score(
        self,
        battle: Battle,
        trainer_id: int,
        a0: Action,
        op_actions: list[Action],
        verbose: bool,
    ) -> float:
        score = 0.0
        # Handle cases where op_actions might be empty to avoid division by zero
        if not op_actions:
            return 0.0

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
        return score / len(op_actions)

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

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions."
            )

        if self.parallelize:
            results = Parallel(n_jobs=-1)(
                delayed(self._calculate_action_score)(
                    battle, trainer_id, a0, op_actions, verbose
                )
                for a0 in my_actions
            )
            avg_scores = np.array(results)
        else:
            avg_scores = np.zeros(len(my_actions))
            for i, a0 in enumerate(my_actions):
                avg_scores[i] = self._calculate_action_score(
                    battle, trainer_id, a0, op_actions, verbose
                )

        best_index = np.argmax(avg_scores)
        return my_actions[best_index]


class MinimaxAgent(EvaluatingAgent):
    """Minimax agent for Pokemon battles."""

    def __init__(
        self, name: str = "Minimax", depth: int = 1, parallelize: bool = False
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth for the Minimax algorithm.
        """
        name = f"{name}(depth={depth})"
        super().__init__(name)
        self.depth = depth
        self.parallelize = parallelize

    def _evaluate_minimax_action(
        self,
        battle: Battle,
        trainer_id: int,
        a0: Action,
        op_actions: list[Action],
        verbose: bool,
    ) -> float:
        min_score = float("inf")
        # If no opponent actions are possible, it means the opponent has no valid moves.
        # This scenario should probably lead to a very favorable outcome for the current trainer.
        # However, to avoid division by zero and handle edge cases, ensure op_actions is not empty
        # or handle it as a special case. For now, assume op_actions won't be empty in normal play.
        if not op_actions:
            # If no opponent actions, consider this a win for our action (or similar heuristic)
            # For simplicity, returning a very high score if opponent can't move.
            # This might need refinement based on game rules.
            return float("inf")

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
        return min_score

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

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions at depth {self.depth}."
            )

        if self.parallelize:
            # When parallelizing, we get all min_scores first, then find the best action.
            scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_minimax_action)(
                    battle, trainer_id, a0, op_actions, verbose
                )
                for a0 in my_actions
            )
            # Find the index of the action that yields the maximum score
            best_action_index = np.argmax(scores)
            return my_actions[best_action_index]
        else:
            best_score = -float("inf")
            best_action = my_actions[0]  # Default to the first action

            for a0 in my_actions:
                min_score = self._evaluate_minimax_action(
                    battle, trainer_id, a0, op_actions, verbose
                )

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

    def __init__(
        self, name: str = "AlphaBeta", depth: int = 2, parallelize: bool = False
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth.
        """
        name = f"{name}(depth={depth})"
        super().__init__(name)
        self.depth = depth
        self.parallelize = parallelize

    def _evaluate_alphabeta_action(
        self,
        battle: Battle,
        action: Action,
        trainer_id: int,
        depth: int,
        alpha: float,
        beta: float,
        verbose: bool,
    ) -> float:
        # Note: The initial alpha/beta passed here for each parallel task
        # will be the global initial alpha/beta, not necessarily updated
        # by other parallel branches. This reduces some pruning efficiency
        # at the very top level but allows parallelization.
        score = self._get_min_value(battle, action, trainer_id, depth, alpha, beta)
        if verbose:
            logger.info(f"Action: {action}, Score: {score}")
        return score

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

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions at depth {self.depth} with Alpha-Beta."
            )

        if self.parallelize:
            scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_alphabeta_action)(
                    battle,
                    action,
                    trainer_id,
                    self.depth,
                    -float("inf"),
                    float("inf"),  # Initial alpha/beta for each parallel task
                    verbose,
                )
                for action in my_actions
            )
            best_action_index = np.argmax(scores)
            return my_actions[best_action_index]
        else:
            best_score = -float("inf")
            best_action = my_actions[0]
            alpha = -float("inf")
            beta = float("inf")

            for action in my_actions:
                score = self._get_min_value(
                    battle, action, trainer_id, self.depth, alpha, beta
                )

                if verbose:
                    logger.info(f"Action: {action}, Score: {score}")

                if score > best_score:
                    best_score = score
                    best_action = action

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
