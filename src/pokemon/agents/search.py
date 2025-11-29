# src/pokemon/agents/search.py
"""This module defines agents that use search algorithms to make decisions."""

import math
import random
import time

import numpy as np
from joblib import Parallel, delayed

from pokemon.action import Action, ActionType
from pokemon.battle import Battle
from pokemon.loguru_logger import logger

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

            if trainer_id == 0:
                sim_battle.turn(a0, a1)
            else:
                sim_battle.turn(a1, a0)

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
        name = f"{name}(d={depth})"
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
            return self._evaluate_battle(battle, trainer_id)

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
            scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_minimax_action)(
                    battle, trainer_id, a0, op_actions, verbose
                )
                for a0 in my_actions
            )
            best_action_index = np.argmax(scores)
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
        name = f"{name}(d={depth})"
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

        if not my_actions:
            return self._evaluate_battle(battle, trainer_id)

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
            return self._evaluate_battle(battle, trainer_id)

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
            scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_alphabeta_action)(
                    battle,
                    action,
                    trainer_id,
                    self.depth,
                    -float("inf"),
                    float("inf"),
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


class MonteCarloExpectiminimaxAgent(EvaluatingAgent):
    """
    An agent that handles the randomness of Pokemon battles by simulating
    the same turn multiple times (Monte Carlo sampling) to calculate an
    Expected Value (EV) for every action pair.
    """

    def __init__(
        self,
        name: str = "MonteCarloExpectiminimax",
        depth: int = 1,
        num_simulations: int = 20,
        parallelize: bool = False,
    ):
        """
        Args:
            name: The name of the agent.
            depth: How many turns ahead to look (usually 1 for heavy sampling).
            num_simulations: How many times to simulate the randomness of a single turn.
                             Higher = more accurate but slower.
            parallelize: Whether to use multiple CPU cores.
        """
        name = f"{name}(d={depth}, sim={num_simulations})"
        super().__init__(name)
        self.depth = depth
        self.num_simulations = num_simulations
        self.parallelize = parallelize

    def _simulate_outcome(
        self, battle: Battle, trainer_id: int, my_action: Action, op_action: Action
    ) -> float:
        """
        Runs N simulations for a specific pair of actions and returns the average score.
        This smooths out RNG (misses, crits, speed ties).
        """
        total_score = 0.0

        # We run the EXACT same turn logic N times to see the distribution of outcomes
        for _ in range(self.num_simulations):
            sim_battle = battle.copy()

            # Apply moves based on ID
            if trainer_id == 0:
                sim_battle.turn(my_action, op_action)
            else:
                sim_battle.turn(op_action, my_action)

            # If we are at depth 0, evaluate. If depth > 1, we would recurse here.
            # For performance, Monte Carlo is usually done at depth 1 or with MCTS.
            # Here we stick to depth 1 evaluation for the sampling.
            total_score += self._evaluate_battle(sim_battle, trainer_id)

        return total_score / self.num_simulations

    def _evaluate_action_node(
        self,
        battle: Battle,
        trainer_id: int,
        my_action: Action,
        op_actions: list[Action],
        verbose: bool,
    ) -> float:
        """
        Evaluates 'my_action' by assuming the opponent will pick the best response
        (Minimax) based on the Expected Value of that response.
        """
        worst_case_expected_value = float("inf")

        if not op_actions:
            return self._evaluate_battle(battle, trainer_id)

        for op_action in op_actions:
            # Calculate Expected Value of this specific matchup (My Move vs Op Move)
            expected_value = self._simulate_outcome(
                battle, trainer_id, my_action, op_action
            )

            # The opponent wants to minimize our score
            if expected_value < worst_case_expected_value:
                worst_case_expected_value = expected_value

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
            # We parallelize the outer loop (My Actions)
            scores = Parallel(n_jobs=-1)(
                delayed(self._evaluate_action_node)(
                    battle, trainer_id, a0, op_actions, verbose
                )
                for a0 in my_actions
            )
            best_index = np.argmax(scores)
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
                    battle, trainer_id, my_action, op_actions, verbose
                )

                if verbose:
                    logger.info(f"-> Worst-case EV for {my_action}: {score:.2f}")

                if score > best_score:
                    best_score = score
                    best_action = my_action

            return best_action


class IterativeDeepeningAlphaBetaAgent(EvaluatingAgent):
    """
    An advanced Alpha-Beta agent that uses Iterative Deepening and Move Ordering.

    Improvements over standard AlphaBeta:
    1. Move Ordering: Evaluates promising moves first to maximize pruning.
    2. Iterative Deepening: Searches depth 1, then 2, then 3... to always have a valid result.
    3. Time Management: Can stop searching if a time limit is reached.
    """

    def __init__(
        self,
        name: str = "IterativeAlphaBeta",
        max_depth: int = 4,
        time_limit: float = 2.0,
    ):
        """
        Args:
            name: Agent name.
            max_depth: The maximum depth to search if time permits.
            time_limit: Maximum time (in seconds) allowed per turn.
        """
        name = f"{name}(max_d={max_depth}, time={time_limit}s)"
        super().__init__(name)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = 0

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
            return self._evaluate_battle(battle, trainer_id)

        # Get actions
        current_actor = trainer_id if is_maximizing else (1 - trainer_id)
        possible_actions = battle.get_possible_actions(current_actor)

        if not possible_actions:
            return self._evaluate_battle(battle, trainer_id)

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

        # Initial Sort
        my_actions = self._heuristic_sort(battle, my_actions, trainer_id)

        best_action = my_actions[0]

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
                        logger.info(
                            f"Depth {d} search failed. Halting deepening."
                        )
                    break

        except TimeoutError:
            if verbose:
                logger.info(
                    f"[{self.name}] Time limit reached. Returning result from depth {d - 1}."
                )

        return best_action


class MCTSNode:
    """
    A node in the Monte Carlo Search Tree.
    Represents a decision point for the Player.
    """

    def __init__(self, parent, action: Action):
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


class MCTSAgent(EvaluatingAgent):
    """
    Open-Loop MCTS Agent for Simultaneous Stochastic Games.

    Why Open-Loop?
    We do not store the game state in the nodes. Instead, every iteration
    we re-simulate from the root. The 'Opponent' and 'RNG' are treated
    as stochastic transitions. This handles the randomness of Pokemon
    (misses, crits, speed ties) naturally by averaging results over many visits.
    """

    def __init__(
        self, name: str = "MCTS", time_limit: float = 1.0, rollout_depth: int = 8
    ):
        name = f"{name}(time={time_limit}s, rollout_d={rollout_depth})"
        super().__init__(name)
        self.time_limit = time_limit
        self.rollout_depth = rollout_depth

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
            node = root

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
                op_action = self._get_heuristic_opponent_move(
                    sim_battle, 1 - trainer_id
                )

                if trainer_id == 0:
                    sim_battle.turn(my_action, op_action)
                else:
                    sim_battle.turn(op_action, my_action)

            # --- 2. EXPANSION ---
            if not sim_battle.done:
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
                        sim_battle.turn(node.action, op_action)
                    else:
                        sim_battle.turn(op_action, node.action)

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
                key=lambda a: self._get_move_power(battle, trainer_id, a)
                + random.uniform(0, 40),
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
        if action.move_slot_index == -1:
            return 40  # Struggle power
        return (
            battle.trainers[trainer_id]
            .active_pokemon.move_slots[action.move_slot_index]
            .move.power
        )
