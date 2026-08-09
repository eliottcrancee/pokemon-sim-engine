"""This module defines agents that use search algorithms to make decisions."""

import math
import random
import time
from collections.abc import Callable
from typing import ClassVar, cast

import numpy as np
from joblib import Parallel, delayed

from pokemon.action import Action, ActionType
from pokemon.agents.evaluation import standard_evaluation
from pokemon.battle import Battle
from pokemon.loguru_logger import logger

from .base_agent import BaseAgent


class _ParallelPoolMixin:
    """Reusable joblib pool shared across turns to avoid re-spawning workers.

    Spawning a loky process pool on Windows is expensive (~1s). Reusing a
    single pool for the whole lifetime of the agent amortises that cost and
    guarantees the CPU is saturated from the very first ``get_action`` call.
    The pool lives on the instance but is excluded from pickling
    (``__getstate__``/``__setstate__``) so parallel workers can still receive
    a copy of the agent without trying to serialize the pool itself.
    """

    _parallel_pool: Parallel | None = None
    # Every subclass sets this in __init__; declared here for static typing.
    parallelize: bool = False

    # Jobs below this threshold have so little work that the overhead of
    # spawning/communicating with parallel workers dwarfs any speedup, so we
    # fall back to sequential execution (measurements: task count * depth).
    PARALLEL_MIN_JOBS: ClassVar[int] = 16

    def _get_pool(self) -> Parallel:
        if self._parallel_pool is None:
            self._parallel_pool = Parallel(n_jobs=-1)
        return self._parallel_pool

    def _should_parallelize(self, task_count: int) -> bool:
        """Return whether parallelizing ``task_count`` jobs is worth it."""
        return self.parallelize and task_count >= self.PARALLEL_MIN_JOBS

    def __getstate__(self):
        state = self.__dict__.copy()
        # The pool is not picklable and must not travel to the workers.
        state["_parallel_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._parallel_pool = None


class OneStepUniformExpectimaxAgent(_ParallelPoolMixin, BaseAgent):
    """Agent that evaluates actions by simulating one turn ahead.
    Uses a heuristic to score the resulting battle state after one turn.
    The action with the highest average score against all opponent responses
    is chosen.
    """

    def __init__(
        self,
        name: str = "OneStepUniformExpectimax",
        parallelize: bool = False,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            evaluation_func: Custom evaluation function.
        """
        super().__init__(name)
        self.parallelize = parallelize
        self.evaluation_func = evaluation_func or standard_evaluation

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

            if trainer_id == 0:
                sim_battle.turn(a0, a1)
            else:
                sim_battle.turn(a1, a0)

            score += self.evaluation_func(sim_battle, trainer_id)

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

        if not my_actions:
            raise ValueError("No possible actions for agent.")

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions."
            )

        if self._should_parallelize(len(my_actions) * len(op_actions)):
            if not op_actions:
                return my_actions[0]
            # Flatten over (my_action x op_action) pairs so joblib can spawn
            # far more tasks than the number of our own actions and saturate
            # all CPU cores. Each task scores one single action pair.
            results = self._get_pool()(
                delayed(self._calculate_action_score)(
                    battle, trainer_id, a0, [a1], verbose
                )
                for a0 in my_actions
                for a1 in op_actions
            )
            avg_scores = (
                np.asarray(results)
                .reshape(len(my_actions), len(op_actions))
                .mean(axis=1)
            )
        else:
            avg_scores = np.zeros(len(my_actions))
            for i, a0 in enumerate(my_actions):
                avg_scores[i] = self._calculate_action_score(
                    battle, trainer_id, a0, op_actions, verbose
                )

        best_index = int(np.argmax(avg_scores))
        return my_actions[best_index]


class MinimaxAgent(_ParallelPoolMixin, BaseAgent):
    """Minimax agent for Pokemon battles."""

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int]]] = {"depth": [1, 2, 3]}

    def __init__(
        self,
        name: str = "Minimax",
        depth: int = 1,
        parallelize: bool = False,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth for the Minimax algorithm.
            evaluation_func: Custom evaluation function.
        """
        name = f"{name}(d={depth})"
        super().__init__(name)
        self.depth = depth
        self.parallelize = parallelize
        self.evaluation_func = evaluation_func or standard_evaluation

    def _apply_turn(
        self, battle: Battle, trainer_id: int, a0: Action, a1: Action
    ) -> Battle:
        """Return a copy of the battle advanced by one turn."""
        sim_battle = battle.copy()
        if trainer_id == 0:
            sim_battle.turn(a0, a1)
        else:
            sim_battle.turn(a1, a0)
        return sim_battle

    def _evaluate_minimax_action(
        self,
        battle: Battle,
        trainer_id: int,
        a0: Action,
        op_actions: list[Action],
        verbose: bool,
    ) -> float:
        min_score = float("inf")
        if not op_actions:
            return float("inf")

        for a1 in op_actions:
            sim_battle = battle.copy()

            if trainer_id == 0:
                sim_battle.turn(a0, a1)
            else:
                sim_battle.turn(a1, a0)

            score = self._minimax(sim_battle, trainer_id, self.depth - 1)
            min_score = min(min_score, score)

            if verbose:
                logger.info(f"Action: {a0}, Opponent Action: {a1}, Score: {score}")
        return min_score

    def _minimax(self, battle: Battle, trainer_id: int, current_depth: int) -> float:
        """Recursive minimax evaluation."""
        if current_depth == 0 or battle.winner is not None:
            return self.evaluation_func(battle, trainer_id)

        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)
        best_score = -float("inf")

        for a0 in my_actions:
            min_score = float("inf")
            for a1 in op_actions:
                sim_battle = battle.copy()

                if trainer_id == 0:
                    sim_battle.turn(a0, a1)
                else:
                    sim_battle.turn(a1, a0)

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

        if not my_actions:
            # This should not happen in a normal game state
            raise ValueError("No possible actions for agent.")

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions against "
                f"{len(op_actions)} opponent actions at depth {self.depth}."
            )

        if self.parallelize:
            if not op_actions:
                return my_actions[0]
            # Parallelize over (my_action x op_action) pairs instead of just
            # my_actions so there are enough tasks to use every CPU core.
            # Each delayed call evaluates a single opponent reply -> a child
            # node; we take the min across replies per move.
            pair_scores = cast(
                list[float],
                self._get_pool()(
                    delayed(self._minimax)(
                        self._apply_turn(battle, trainer_id, a0, a1),
                        trainer_id,
                        self.depth - 1,
                    )
                    for a0 in my_actions
                    for a1 in op_actions
                ),
            )
            scores = (
                np.asarray(pair_scores)
                .reshape(len(my_actions), len(op_actions))
                .min(axis=1)
            )
            best_action_index = int(np.argmax(scores))
            return my_actions[best_action_index]
        else:
            best_score = -float("inf")
            best_action = my_actions[0]

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


class AlphaBetaAgent(_ParallelPoolMixin, BaseAgent):
    """Minimax agent with Alpha-Beta pruning for Pokemon battles."""

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int]]] = {"depth": [1, 2, 3]}

    def __init__(
        self,
        name: str = "AlphaBeta",
        depth: int = 2,
        parallelize: bool = False,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        """Initialize the agent.
        Args:
            name: The name of the agent.
            depth: The search depth.
            evaluation_func: Custom evaluation function.
        """
        name = f"{name}(d={depth})"
        super().__init__(name)
        self.depth = depth
        self.parallelize = parallelize
        self.evaluation_func = evaluation_func or standard_evaluation

    def _apply_turn(
        self, battle: Battle, trainer_id: int, a0: Action, a1: Action
    ) -> Battle:
        """Return a copy of the battle advanced by one turn."""
        sim_battle = battle.copy()
        if trainer_id == 0:
            sim_battle.turn(a0, a1)
        else:
            sim_battle.turn(a1, a0)
        return sim_battle

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
        score = self._get_min_value(battle, action, trainer_id, depth, alpha, beta)
        if verbose:
            logger.info(f"Action: {action}, Score: {score}")
        return score

    def _get_max_value(
        self, battle: Battle, trainer_id: int, depth: int, alpha: float, beta: float
    ) -> float:
        """Maximizer node: Tries to maximize the score."""
        if depth == 0 or battle.winner is not None:
            return self.evaluation_func(battle, trainer_id)

        value = -float("inf")
        my_actions = battle.get_possible_actions(trainer_id)

        if not my_actions:
            return self.evaluation_func(battle, trainer_id)

        for action in my_actions:
            v = self._get_min_value(battle, action, trainer_id, depth, alpha, beta)
            value = max(value, v)
            alpha = max(alpha, value)
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

        if not op_actions:
            return self.evaluation_func(battle, trainer_id)

        for op_action in op_actions:
            sim_battle = battle.copy()

            if trainer_id == 0:
                sim_battle.turn(my_action, op_action)
            else:
                sim_battle.turn(op_action, my_action)

            v = self._get_max_value(sim_battle, trainer_id, depth - 1, alpha, beta)
            value = min(value, v)
            beta = min(beta, value)
            if value <= alpha:
                break
        return value

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        """Choose an action using Minimax with Alpha-Beta pruning."""
        my_actions = battle.get_possible_actions(trainer_id)

        if not my_actions:
            raise ValueError("No possible actions for agent.")

        if verbose:
            logger.info(f"[{self.name}] getting action for Trainer {trainer_id}")
            logger.info(
                f"Evaluating {len(my_actions)} actions at depth {self.depth} with Alpha-Beta."
            )

        if self.parallelize:
            if not my_actions:
                raise ValueError("No possible actions for agent.")

            # Young Brothers Wait Concept (YBWC): search the first child
            # sequentially to obtain a tight alpha bound, then evaluate the
            # remaining siblings in parallel using that bound. Sharing the
            # alpha bound preserves pruning (unlike a naive flatten that
            # restarts each task from [-inf, inf] and re-explores pruned
            # branches), which is what makes parallel alpha-beta actually
            # FASTER than sequential.
            op_actions = battle.get_possible_actions(1 - trainer_id)
            if len(my_actions) == 1 or not op_actions:
                return my_actions[
                    int(
                        np.argmax(
                            [
                                self._get_min_value(
                                    battle,
                                    a,
                                    trainer_id,
                                    self.depth,
                                    -float("inf"),
                                    float("inf"),
                                )
                                for a in my_actions
                            ]
                        )
                    )
                ]

            first_action = my_actions[0]
            alpha_bound = self._get_min_value(
                battle,
                first_action,
                trainer_id,
                self.depth,
                -float("inf"),
                float("inf"),
            )
            sibling_scores = cast(
                list[float],
                self._get_pool()(
                    delayed(self._get_min_value)(
                        battle,
                        action,
                        trainer_id,
                        self.depth,
                        alpha_bound,
                        float("inf"),
                    )
                    for action in my_actions[1:]
                ),
            )
            scores = [alpha_bound, *sibling_scores]
            best_action_index = int(np.argmax(scores))
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


class MonteCarloExpectiminimaxAgent(_ParallelPoolMixin, BaseAgent):
    """
    An agent that handles the randomness of Pokemon battles by simulating
    the same turn multiple times (Monte Carlo sampling) to calculate an
    Expected Value (EV) for every action pair.
    """

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int]]] = {
        "depth": [1, 2],
        "num_simulations": [20, 40, 60],
    }

    def __init__(
        self,
        name: str = "MonteCarloExpectiminimax",
        depth: int = 1,
        num_simulations: int = 20,
        parallelize: bool = False,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        """
        Args:
            name: The name of the agent.
            depth: How many turns ahead to look (usually 1 for heavy sampling).
            num_simulations: How many times to simulate the randomness of a single turn.
                             Higher = more accurate but slower.
            parallelize: Whether to use multiple CPU cores.
            evaluation_func: Custom evaluation function.
        """
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")

        name = f"{name}(d={depth}, sim={num_simulations})"
        super().__init__(name)
        self.depth = depth
        self.num_simulations = num_simulations
        self.parallelize = parallelize
        self.evaluation_func = evaluation_func or standard_evaluation

    def _simulate_outcome(
        self,
        battle: Battle,
        trainer_id: int,
        my_action: Action,
        op_action: Action,
        depth: int,
    ) -> float:
        """
        Runs N simulations for a specific pair of actions and returns the
        average score. This smooths out RNG (misses, crits, speed ties).

        ``Battle.copy()`` clones the RNG state, so every simulation must be
        re-seeded with an independent stream: otherwise all N samples resolve
        identically and the average collapses to a single deterministic turn.
        """
        total_score = 0.0

        for _ in range(self.num_simulations):
            sim_battle = battle.copy()
            # Independent RNG per sample so the outcome distribution is
            # actually explored instead of repeating one deterministic roll.
            sim_battle.reseed(random.getrandbits(64))

            if trainer_id == 0:
                sim_battle.turn(my_action, op_action)
            else:
                sim_battle.turn(op_action, my_action)

            if depth <= 1 or sim_battle.winner is not None:
                total_score += self.evaluation_func(sim_battle, trainer_id)
            else:
                total_score += self._expectimax_value(sim_battle, trainer_id, depth - 1)

        return total_score / self.num_simulations

    def _expectimax_value(self, battle: Battle, trainer_id: int, depth: int) -> float:
        """Recursive expectimax evaluation for depth > 1."""
        if depth == 0 or battle.winner is not None:
            return self.evaluation_func(battle, trainer_id)

        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        if not my_actions:
            return self.evaluation_func(battle, trainer_id)

        action_scores = [
            self._evaluate_action_node(
                battle,
                trainer_id,
                my_action,
                op_actions,
                depth,
                False,
            )
            for my_action in my_actions
        ]
        return max(action_scores)

    def _evaluate_action_node(
        self,
        battle: Battle,
        trainer_id: int,
        my_action: Action,
        op_actions: list[Action],
        depth: int,
        verbose: bool,
    ) -> float:
        """
        Evaluates 'my_action' by assuming the opponent will pick the best response
        (Minimax) based on the Expected Value of that response.
        """
        worst_case_expected_value = float("inf")

        if not op_actions:
            return self.evaluation_func(battle, trainer_id)

        for op_action in op_actions:
            # Calculate Expected Value of this specific matchup (My Move vs Op Move)
            expected_value = self._simulate_outcome(
                battle, trainer_id, my_action, op_action, depth
            )

            # The opponent wants to minimize our score
            worst_case_expected_value = min(worst_case_expected_value, expected_value)

            if verbose:
                logger.info(f"  vs Opponent {op_action}: EV = {expected_value:.2f}")

        return worst_case_expected_value

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        if not my_actions:
            raise ValueError("No possible actions.")

        if verbose:
            logger.info(
                f"[{self.name}] Evaluting {len(my_actions)} actions with {self.num_simulations} samples each."
            )

        # --- Parallel Execution ---
        if self.parallelize:
            if not op_actions:
                return my_actions[0]
            # Flatten over (my_action, op_action) so there are far more tasks
            # than my_actions alone; each task runs its own MC sampling for a
            # single pair and we take the min across opponent replies.
            pair_scores = cast(
                list[float],
                self._get_pool()(
                    delayed(self._simulate_outcome)(
                        battle, trainer_id, a0, a1, self.depth
                    )
                    for a0 in my_actions
                    for a1 in op_actions
                ),
            )
            scores = (
                np.asarray(pair_scores)
                .reshape(len(my_actions), len(op_actions))
                .min(axis=1)
            )
            best_index = int(np.argmax(scores))
            return my_actions[best_index]

        # --- Sequential Execution ---
        else:
            best_score = -float("inf")
            best_action = my_actions[0]

            for my_action in my_actions:
                if verbose:
                    logger.info(f"Analyzing My Action: {my_action}")

                # Get the Minimax value for this action (assuming average RNG)
                score = self._evaluate_action_node(
                    battle, trainer_id, my_action, op_actions, self.depth, verbose
                )

                if verbose:
                    logger.info(f"-> Worst-case EV for {my_action}: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_action = my_action

            return best_action


class RiskAwareExpectimaxAgent(_ParallelPoolMixin, BaseAgent):
    """
    Expectimax agent that explicitly models turn randomness with Monte Carlo
    sampling and applies a risk penalty to unstable outcomes.

    Utility formula per action pair:
        utility = mean(score_samples) - risk_lambda * std(score_samples)
    """

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int] | list[float]]] = {
        "depth": [1, 2],
        "num_simulations": [20, 40, 60],
        "risk_lambda": [0.05, 0.1, 0.15, 0.2],
    }

    def __init__(
        self,
        name: str = "RiskAwareExpectimax",
        depth: int = 1,
        num_simulations: int = 10,
        risk_lambda: float = 0.1,
        parallelize: bool = False,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if risk_lambda < 0:
            raise ValueError("risk_lambda must be >= 0")

        name = f"{name}(d={depth}, sim={num_simulations}, risk_lambda={risk_lambda})"
        super().__init__(name)
        self.depth = depth
        self.num_simulations = num_simulations
        self.risk_lambda = risk_lambda
        self.parallelize = parallelize
        self.evaluation_func = evaluation_func or standard_evaluation

    def _risk_adjusted_pair_value(
        self,
        battle: Battle,
        trainer_id: int,
        my_action: Action,
        op_action: Action,
        depth: int,
    ) -> float:
        samples: list[float] = []

        for _ in range(self.num_simulations):
            sim_battle = battle.copy()
            # Battle.copy() clones the RNG state; without re-seeding every
            # sample resolves identically and std_dev is always 0, which
            # silently disables the risk penalty.
            sim_battle.reseed(random.getrandbits(64))

            if trainer_id == 0:
                sim_battle.turn(my_action, op_action)
            else:
                sim_battle.turn(op_action, my_action)

            if depth == 1 or sim_battle.winner is not None:
                sample_score = self.evaluation_func(sim_battle, trainer_id)
            else:
                sample_score = self._expectimax_value(sim_battle, trainer_id, depth - 1)

            samples.append(sample_score)

        mean_score = sum(samples) / len(samples)
        variance = sum((value - mean_score) ** 2 for value in samples) / len(samples)
        std_dev = math.sqrt(variance)
        return mean_score - (self.risk_lambda * std_dev)

    def _evaluate_action_node(
        self,
        battle: Battle,
        trainer_id: int,
        my_action: Action,
        op_actions: list[Action],
        depth: int,
        verbose: bool,
    ) -> float:
        if not op_actions:
            return self.evaluation_func(battle, trainer_id)

        total_score = 0.0
        for op_action in op_actions:
            pair_score = self._risk_adjusted_pair_value(
                battle, trainer_id, my_action, op_action, depth
            )
            total_score += pair_score

            if verbose:
                logger.info(
                    f"  vs Opponent {op_action}: risk-adjusted value = {pair_score:.2f}"
                )

        return total_score / len(op_actions)

    def _expectimax_value(self, battle: Battle, trainer_id: int, depth: int) -> float:
        if depth == 0 or battle.winner is not None:
            return self.evaluation_func(battle, trainer_id)

        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        if not my_actions:
            return self.evaluation_func(battle, trainer_id)

        action_scores = [
            self._evaluate_action_node(
                battle,
                trainer_id,
                my_action,
                op_actions,
                depth,
                False,
            )
            for my_action in my_actions
        ]
        return max(action_scores)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        if not my_actions:
            raise ValueError("No possible actions.")

        if verbose:
            logger.info(
                f"[{self.name}] evaluating {len(my_actions)} actions with "
                f"{self.num_simulations} simulations and risk lambda={self.risk_lambda}."
            )

        if self.parallelize:
            if not op_actions:
                return my_actions[0]
            pair_scores = cast(
                list[float],
                self._get_pool()(
                    delayed(self._risk_adjusted_pair_value)(
                        battle, trainer_id, action, op_action, self.depth
                    )
                    for action in my_actions
                    for op_action in op_actions
                ),
            )
            scores = (
                np.asarray(pair_scores)
                .reshape(len(my_actions), len(op_actions))
                .mean(axis=1)
            )
            best_index = int(np.argmax(scores))
            return my_actions[best_index]
        else:
            best_score = -float("inf")
            best_action = my_actions[0]

            for action in my_actions:
                score = self._evaluate_action_node(
                    battle,
                    trainer_id,
                    action,
                    op_actions,
                    self.depth,
                    verbose,
                )
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action


class IterativeDeepeningAlphaBetaAgent(BaseAgent):
    """
    An advanced Alpha-Beta agent that uses Iterative Deepening and Move Ordering.

    Improvements over standard AlphaBeta:
    1. Move Ordering: Evaluates promising moves first to maximize pruning.
    2. Iterative Deepening: Searches depth 1, then 2, then 3... to always have a valid result.
    3. Time Management: Can stop searching if a time limit is reached.
    """

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int] | list[float]]] = {
        "max_depth": [2, 3, 4],
        "time_limit": [0.5, 1.0, 2.0],
    }

    def __init__(
        self,
        name: str = "IterativeAlphaBeta",
        max_depth: int = 4,
        time_limit: float = 2.0,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        """
        Args:
            name: Agent name.
            max_depth: The maximum depth to search if time permits.
            time_limit: Maximum time (in seconds) allowed per turn.
            evaluation_func: Custom evaluation function.
        """
        name = f"{name}(max_d={max_depth}, time={time_limit}s)"
        super().__init__(name)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = 0
        self.evaluation_func = evaluation_func or standard_evaluation

    def _heuristic_sort(
        self, battle: Battle, actions: list[Action], trainer_id: int
    ) -> list[Action]:
        """
        Sorts actions to try the most promising ones first.
        Pruning works best when the best move is found early.
        """

        def score_action(action):
            # Prioritize switching if active pokemon is about to die (simple heuristic)
            if action.action_type == ActionType.SWITCH:
                # If current pokemon is healthy, switching is low priority
                if battle.trainers[trainer_id].active_pokemon.hp_ratio > 0.3:
                    return -10
                return 5

            if action.action_type == ActionType.ATTACK:
                if action.move_slot_index == -1:
                    return 0  # Struggle
                move = (
                    battle.trainers[trainer_id]
                    .active_pokemon.move_slots[action.move_slot_index]
                    .move
                )
                # Heuristic: Priority > Power > Accuracy
                return move.power + (move.priority * 100)

            return 0  # Pass or Item

        # Sort descending
        return sorted(actions, key=score_action, reverse=True)

    def _is_time_up(self) -> bool:
        return (time.perf_counter() - self.start_time) > self.time_limit

    def _alphabeta(
        self,
        battle: Battle,
        trainer_id: int,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> float:
        # Check time budget occasionally (every few nodes check could be expensive,
        # so we check at start of node)
        if self._is_time_up():
            raise TimeoutError()

        # Leaf node or terminal state
        if depth == 0 or battle.winner is not None:
            return self.evaluation_func(battle, trainer_id)

        # Get actions
        current_actor = trainer_id if is_maximizing else (1 - trainer_id)
        possible_actions = battle.get_possible_actions(current_actor)

        if not possible_actions:
            return self.evaluation_func(battle, trainer_id)

        # Optimization: Sort moves to maximize pruning
        # We only sort at higher depths to save overhead on deep leaves
        if depth > 1:
            possible_actions = self._heuristic_sort(
                battle, possible_actions, current_actor
            )

        if is_maximizing:
            value = -float("inf")
            for action in possible_actions:
                # We need to simulate the opponent's response for this round
                # In this structure, we assume simultaneous resolution is split into ply
                # Max ply (Player picks) -> Min ply (Opponent picks) -> Resolution

                # However, since Battle.turn takes both, we must simulate the Min layer inside here
                # or recurse. Standard Minimax for simultaneous games assumes:
                # Max Node: I pick action A. What is the value?
                # The value is the result of the Opponent picking their BEST response B.

                # So we recurse to a Minimizing node passing my action
                val = self._min_node_simulation(
                    battle, action, trainer_id, depth, alpha, beta
                )

                value = max(value, val)
                alpha = max(alpha, value)
                if value >= beta:
                    break  # Beta Cutoff
            return value

        # This branch shouldn't strictly be reached in this structure
        # because _min_node_simulation handles the opponent,
        # but kept for standard structure completeness.
        return 0.0

    def _min_node_simulation(
        self,
        battle: Battle,
        my_action: Action,
        trainer_id: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Represents the opponent choosing their move against 'my_action'.
        """
        if self._is_time_up():
            raise TimeoutError()

        opponent_id = 1 - trainer_id
        op_actions = battle.get_possible_actions(opponent_id)

        if depth > 1:
            op_actions = self._heuristic_sort(battle, op_actions, opponent_id)

        value = float("inf")

        for op_action in op_actions:
            sim_battle = battle.copy()

            # Execute turn
            if trainer_id == 0:
                sim_battle.turn(my_action, op_action)
            else:
                sim_battle.turn(op_action, my_action)

            # Recurse back to Maximizing node (Depth - 1)
            val = self._alphabeta(sim_battle, trainer_id, depth - 1, alpha, beta, True)

            value = min(value, val)
            beta = min(beta, value)
            if value <= alpha:
                break  # Alpha Cutoff

        return value

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        self.start_time = time.perf_counter()
        my_actions = battle.get_possible_actions(trainer_id)

        if not my_actions:
            raise ValueError("No possible actions for agent.")

        # Initial Sort
        my_actions = self._heuristic_sort(battle, my_actions, trainer_id)

        best_action = my_actions[0]
        completed_depth = 0

        try:
            # Iterative Deepening Loop
            for d in range(1, self.max_depth + 1):
                if verbose:
                    logger.info(f"[{self.name}] Searching Depth {d}...")

                current_depth_best_action = None
                current_depth_best_score = -float("inf")
                alpha = -float("inf")
                beta = float("inf")

                # Root Search
                for action in my_actions:
                    score = self._min_node_simulation(
                        battle, action, trainer_id, d, alpha, beta
                    )

                    if score > current_depth_best_score:
                        current_depth_best_score = score
                        current_depth_best_action = action

                    alpha = max(alpha, current_depth_best_score)

                # If we finished the depth successfully, update the main best_action
                if current_depth_best_action is not None:
                    best_action = current_depth_best_action
                    completed_depth = d

                    # Update sorting for next depth: Put the best action first!
                    # This massively improves pruning for the next depth
                    my_actions.remove(best_action)
                    my_actions.insert(0, best_action)

                    if verbose:
                        logger.info(
                            f"Depth {d} result: {best_action} (Score: {current_depth_best_score})"
                        )
                else:
                    # If search at this depth failed, stop deepening.
                    if verbose:
                        logger.info(f"Depth {d} search failed. Halting deepening.")
                    break

        except TimeoutError:
            if verbose:
                logger.info(
                    f"[{self.name}] Time limit reached. "
                    f"Returning result from depth {completed_depth}."
                )

        return best_action


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.
    Represents a decision point for the Player.
    """

    def __init__(self, parent: "MCTSNode | None", action: Action | None):
        self.parent = parent
        self.action = action  # The action YOU took to get here
        self.children: dict[str, MCTSNode] = {}  # Map Action string -> Node
        self.visits = 0
        self.value = 0.0  # Accumulated score

        # We do NOT store 'untried_actions' here initially, because
        # valid actions depend on the state, and the state depends on
        # what the opponent did (which is stochastic in this implementation).

    def is_fully_expanded(self, possible_actions: list[Action]) -> bool:
        # A node is fully expanded if we have tried all currently possible actions as children
        return len(self.children) == len(possible_actions)

    def best_child(self, exploration_weight=1.41):
        """Selects the child with the highest Upper Confidence Bound (UCB1)."""
        best_score = -float("inf")
        best_nodes = []

        for child in self.children.values():
            if child.visits == 0:
                # Prioritize unvisited slightly to ensure coverage if added but not simulated
                score = float("inf")
            else:
                # UCB1: Exploitation + Exploration
                exploit = child.value / child.visits
                explore = math.sqrt(2 * math.log(self.visits) / child.visits)
                score = exploit + (exploration_weight * explore)

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        if not best_nodes:
            return None

        return random.choice(best_nodes)


class MCTSAgent(BaseAgent):
    """
    Open-Loop MCTS Agent for Simultaneous Stochastic Games.

    Why Open-Loop?
    We do not store the game state in the nodes. Instead, every iteration
    we re-simulate from the root. The 'Opponent' and 'RNG' are treated
    as stochastic transitions. This handles the randomness of Pokemon
    (misses, crits, speed ties) naturally by averaging results over many visits.
    """

    DEFAULT_SWEEP_PARAMS: ClassVar[dict[str, list[int] | list[float]]] = {
        "time_limit": [0.5, 1.0, 2.0],
        "rollout_depth": [4, 8, 12],
    }

    def __init__(
        self,
        name: str = "MCTS",
        time_limit: float = 1.0,
        rollout_depth: int = 8,
        evaluation_func: Callable[[Battle, int], float] | None = None,
    ):
        name = f"{name}(time={time_limit}s, rollout_d={rollout_depth})"
        super().__init__(name)
        self.time_limit = time_limit
        self.rollout_depth = rollout_depth
        self.evaluation_func = evaluation_func or standard_evaluation

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.time()

        # Root represents the CURRENT state
        root = MCTSNode(parent=None, action=None)

        iterations = 0

        # We need a string representation for actions to map them in children dict
        # We assume Action.__str__ or __repr__ is unique enough

        while time.time() - start_time < self.time_limit:
            iterations += 1
            node: MCTSNode | None = root

            # CRITICAL: We copy the battle at the start of every iteration.
            # This is "Open Loop". We assume the path down the tree is valid.
            sim_battle = battle.copy()

            # --- 1. SELECTION ---
            # Walk down the tree until we find a node that isn't fully expanded
            # OR we hit a terminal state.
            while not sim_battle.done:
                # Get legal actions for THIS state (it might differ from previous runs due to RNG)
                legal_actions = sim_battle.get_possible_actions(trainer_id)

                # If we can expand (have actions we haven't created children for yet)
                if not node.is_fully_expanded(legal_actions):
                    break  # Go to Expansion

                # Otherwise, choose best child to descend
                node = node.best_child()
                if node is None:
                    # Should not happen unless no legal actions
                    break

                # Apply the move associated with the child node
                # AND simulate an opponent move to advance the state
                my_action = node.action
                if my_action is None:
                    break
                op_action = self._get_heuristic_opponent_move(
                    sim_battle, 1 - trainer_id
                )

                if trainer_id == 0:
                    sim_battle.turn(my_action, op_action)
                else:
                    sim_battle.turn(op_action, my_action)

            # --- 2. EXPANSION ---
            if not sim_battle.done and node is not None:
                legal_actions = sim_battle.get_possible_actions(trainer_id)

                # Find actions we haven't tried yet at this node
                tried_action_strs = set(node.children.keys())
                untried = [a for a in legal_actions if str(a) not in tried_action_strs]

                if untried:
                    action = random.choice(untried)
                    new_node = MCTSNode(parent=node, action=action)
                    node.children[str(action)] = new_node
                    node = new_node

                    # Apply this new action to the simulation
                    op_action = self._get_heuristic_opponent_move(
                        sim_battle, 1 - trainer_id
                    )
                    if trainer_id == 0:
                        sim_battle.turn(action, op_action)
                    else:
                        sim_battle.turn(op_action, action)

            # --- 3. SIMULATION (Rollout) ---
            depth = 0
            while not sim_battle.done and depth < self.rollout_depth:
                # Use heuristics for rollout so it's not totally random chaos
                a0 = self._get_rollout_move(sim_battle, 0)
                a1 = self._get_rollout_move(sim_battle, 1)
                sim_battle.turn(a0, a1)
                depth += 1

            # --- 4. BACKPROPAGATION ---
            # Calculate score from the perspective of the agent (trainer_id)
            score = self._evaluate_result(sim_battle, trainer_id)

            # Backpropagate up the tree
            while node is not None:
                node.visits += 1
                node.value += score
                node = node.parent

        if verbose:
            logger.info(f"[{self.name}] {iterations} iterations, {self.time_limit}s")

        # Robust Child selection: Pick most visited, not highest value (handles RNG better)
        if not root.children:
            # Fallback if no simulations ran (e.g. time limit 0)
            return random.choice(battle.get_possible_actions(trainer_id))

        best_child = max(root.children.values(), key=lambda n: n.visits)
        if best_child.action is None:
            return random.choice(battle.get_possible_actions(trainer_id))

        if verbose:
            win_rate = best_child.value / best_child.visits
            logger.info(
                f"Best Move: {best_child.action} (Win Rate: {win_rate:.2%}, Visits: {best_child.visits})"
            )

        return best_child.action

    def _evaluate_result(self, battle: Battle, trainer_id: int) -> float:
        """
        Returns a score between 0.0 and 1.0.
        1.0 = Win
        0.0 = Loss
        0.5ish = In progress.
        """
        if battle.winner == trainer_id:
            return 1.0
        elif battle.winner == (1 - trainer_id):
            return 0.0
        elif battle.tie:
            return 0.5

        # Heuristic for unfinished games
        # Based on HP difference
        my_pokemon = battle.trainers[trainer_id].active_pokemon
        op_pokemon = battle.trainers[1 - trainer_id].active_pokemon

        my_hp_frac = my_pokemon.hp_ratio
        op_hp_frac = op_pokemon.hp_ratio

        # 0.5 base + up to 0.5 based on HP diff
        # If I have 100% and they have 0%, score is 1.0
        score = 0.5 + (my_hp_frac - op_hp_frac) * 0.5
        return max(0.0, min(1.0, score))

    def _get_heuristic_opponent_move(self, battle: Battle, trainer_id: int) -> Action:
        """
        Simulate a semi-competent opponent.
        If we assume random, the MCTS plans for the opponent to make mistakes.
        """
        actions = battle.get_possible_actions(trainer_id)

        # Filter for attacks
        attacks = [a for a in actions if a.action_type == ActionType.ATTACK]
        # switches = [a for a in actions if isinstance(a, SwitchAction)]

        # If active pokemon is dead, must switch (handled by get_possible_actions,
        # but if multiple switches exist, pick random)
        if not battle.trainers[trainer_id].active_pokemon.is_alive:
            return random.choice(actions)

        # 85% chance to attack if possible
        if attacks and random.random() < 0.85:
            # Prefer higher power moves slightly
            return max(
                attacks,
                key=lambda a: (
                    self._get_move_power(battle, trainer_id, a) + random.uniform(0, 40)
                ),
            )

        return random.choice(actions)

    def _get_rollout_move(self, battle: Battle, trainer_id: int) -> Action:
        """
        Fast heuristic for rollout phase.
        Needs to be very fast to allow many simulations.
        """
        actions = battle.get_possible_actions(trainer_id)

        # Simple heuristic: If we can attack, attack with random high power move.
        # If low HP, maybe switch? (Too complex for rollout, keep it simple)
        attacks = [a for a in actions if a.action_type == ActionType.ATTACK]

        if attacks:
            # Randomly pick an attack, weighted slightly by power?
            # Or just random attack to be fast.
            # Let's do random attack to save CPU cycles for more iterations.
            return random.choice(attacks)

        return random.choice(actions)

    def _get_move_power(self, battle: Battle, trainer_id: int, action: Action):
        move_slot_index = action.attack_move_slot_index
        if move_slot_index == -1:
            return 40  # Struggle power
        return (
            battle.trainers[trainer_id]
            .active_pokemon.move_slots[move_slot_index]
            .move.power
        )
