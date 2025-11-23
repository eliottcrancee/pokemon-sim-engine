# agent.py

import math
import random
import time

import numpy as np

from pokemon.action import Action, ActionError, ActionType
from pokemon.battle import Battle
from pokemon.glicko2 import calculate_new_rating
from pokemon.loguru_logger import logger

CHOICE = random.choice
RANDOM = random.random


class BaseAgent:
    """Base class for all agents."""

    def __init__(self, name):
        self.name = name
        # Glicko-2 rating system parameters
        self.r = 1500.0  # Rating
        self.rd = 350.0  # Rating Deviation

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        raise NotImplementedError

    def update_rating(self, period_results):
        self.r, self.rd = calculate_new_rating(self.r, self.rd, period_results)

    @staticmethod
    def translate_action(
        action: Action,
        new_battle: Battle,
        trainer_id: int,
    ) -> Action:
        if action is None:
            return None

        new_trainer = new_battle.get_trainer_by_id(trainer_id)
        new_opponent = new_battle.get_trainer_by_id(1 - trainer_id)

        new_item = None
        if action.item:
            new_item = new_trainer.inventory.get(action.item.name, None)
            if new_item is None:
                raise ActionError(
                    f"Item {action.item.name} not found in trainer {new_trainer.name}'s inventory"
                )

        return Action(
            action_type=action.action_type,
            trainer=new_trainer,
            opponent=new_opponent,
            move=action.move,
            target_index=action.target_index,
            item=new_item,
        )


class InputAgent(BaseAgent):
    """Agent that requires user input for actions."""

    def __init__(self, name="Input"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        return actions


class FirstAgent(BaseAgent):
    """Agent that always selects the first available action."""

    def __init__(self, name="First"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        action = battle.get_possible_actions(trainer_id)[0]
        return action


class RandomAgent(BaseAgent):
    """Agent that selects a random action from the available actions."""

    def __init__(self, name="Random"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        action = random.choice(actions)
        return action


class RandomAttackAgent(BaseAgent):
    """Agent that selects a random attack action from the available actions."""

    def __init__(self, name="RandomAttack"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)
        return random.choice(attack_actions)


class RandomAttackAndPotionAgent(BaseAgent):
    """Agent that selects a random attack action from the available actions.
    If the current Pokemon's HP is below a threshold, it will try to use a healing item instead.
    """

    def __init__(self, name="RandomAttackAndPotion", heal_threshold=0.20):
        if heal_threshold != 0.2:
            name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
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


class BestAttackAgent(BaseAgent):
    """Agent that selects the attack action with the highest power."""

    def __init__(self, name="BestAttack"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        # Choose the attack with the highest power
        best_action = max(attack_actions, key=lambda a: a.move.power if a.move else 0)
        return best_action


class BestAttackAndPotionAgent(BaseAgent):
    """Agent that selects the attack action with the highest power.
    If the current Pokemon's HP is below a threshold, it will try to use a healing item instead.
    """

    def __init__(self, name="BestAttackAndPotion", heal_threshold=0.20):
        if heal_threshold != 0.2:
            name = f"{name}(heal_threshold={heal_threshold})"
        super().__init__(name)
        self.heal_threshold = heal_threshold

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
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
                # Choose the healing item with the highest quantity
                best_heal_action = max(heal_actions, key=lambda a: a.item.quantity)
                return best_heal_action

        attack_actions = [a for a in actions if a.action_type == ActionType.ATTACK]
        if not attack_actions:
            return random.choice(actions)

        # Choose the attack with the highest power
        best_attack_action = max(
            attack_actions, key=lambda a: a.move.power if a.move else 0
        )
        return best_attack_action


class OneStepUniformExpectimaxAgent(BaseAgent):
    """Agent that evaluates all possible actions by simulating one turn ahead.
    Uses a heuristic to score the resulting battle state after one turn.
    The action with the highest average score against all opponent responses is chosen.
    """

    def __init__(self, name="OneStepUniformExpectimax"):
        super().__init__(name)

    def _score_trainer(self, battle: Battle, t_id: int) -> float:
        trainer = battle.get_trainer_by_id(t_id)
        hp_score = sum(p.hp / p.max_hp for p in trainer.pokemon_team if p.is_alive)
        alive_bonus = sum(0.5 for p in trainer.pokemon_team if p.is_alive)
        item_bonus = sum(0.2 * i.quantity for i in trainer.inventory.values())
        return hp_score + alive_bonus + item_bonus

    def _evaluate_battle(self, battle: Battle, trainer_id: int) -> float:
        if battle.winner == trainer_id:
            return 100.0
        if battle.winner is not None:
            return -100.0
        return self._score_trainer(battle, trainer_id) - self._score_trainer(
            battle, 1 - trainer_id
        )

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        avg_scores = []

        for a0 in actions:
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

            avg_scores.append(score / len(op_actions))

        best_index = np.argmax(avg_scores)
        return actions[best_index]


class OneStepMinimaxAgent(BaseAgent):
    """One-step Minimax agent for a zero-sum simultaneous-move Pokemon battle.
    For each of our actions, compute the worst-case resulting evaluation
    across all opponent actions. Choose the action that maximizes this worst case.
    """

    def __init__(self, name="OneStepMinimax"):
        super().__init__(name)

    def _score_trainer(self, battle: Battle, t_id: int) -> float:
        trainer = battle.get_trainer_by_id(t_id)
        hp_score = sum(p.hp / p.max_hp for p in trainer.pokemon_team if p.is_alive)
        alive_bonus = sum(0.5 for p in trainer.pokemon_team if p.is_alive)
        item_bonus = sum(0.2 * i.quantity for i in trainer.inventory.values())
        return hp_score + alive_bonus + item_bonus

    def _evaluate_battle(self, battle: Battle, trainer_id: int) -> float:
        if battle.winner == trainer_id:
            return 100.0
        if battle.winner is not None:
            return -100.0
        return self._score_trainer(battle, trainer_id) - self._score_trainer(
            battle, 1 - trainer_id
        )

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        # All possible actions for both players
        my_actions = battle.get_possible_actions(trainer_id)
        op_actions = battle.get_possible_actions(1 - trainer_id)

        worst_case_scores = []

        # --- Minimax step ---
        for a0 in my_actions:
            # We want: min over opponent actions of resulting score
            min_score = float("inf")

            for a1 in op_actions:
                sim_battle = battle.copy()
                my_sim_action = self.translate_action(a0, sim_battle, trainer_id)
                op_sim_action = self.translate_action(a1, sim_battle, 1 - trainer_id)

                # turn() expects (action_player0, action_player1)
                if trainer_id == 0:
                    sim_battle.turn(my_sim_action, op_sim_action)
                else:
                    sim_battle.turn(op_sim_action, my_sim_action)

                score = self._evaluate_battle(sim_battle, trainer_id)

                if score < min_score:
                    min_score = score

                if verbose:
                    logger.info(f"[Minimax] My={a0}, Opp={a1}, Score={score}")

            worst_case_scores.append(min_score)

        # We pick the action that maximizes the worst case
        best_index = np.argmax(worst_case_scores)
        return my_actions[best_index]


class RolloutEvaluateAgent(BaseAgent):
    """Agent that evaluates all possible actions by performing random rollouts,
    averaging the results to choose the best action.
    """

    def __init__(self, name="RolloutEvaluateAgent", time_budget=0.005, depth=0):
        name = f"{name}(budget={time_budget}s, depth={depth})"
        super().__init__(name)
        self.time_budget = time_budget
        self.depth = depth

    def _score_trainer(self, battle: Battle, t_id: int) -> float:
        trainer = battle.get_trainer_by_id(t_id)
        hp_score = sum(p.hp / p.max_hp for p in trainer.pokemon_team if p.is_alive)
        alive_bonus = sum(0.5 for p in trainer.pokemon_team if p.is_alive)
        item_bonus = sum(0.2 * i.quantity for i in trainer.inventory.values())
        return hp_score + alive_bonus + item_bonus

    def _evaluate_battle(self, battle: Battle, trainer_id: int) -> float:
        if battle.winner == trainer_id:
            return 100.0
        if battle.winner is not None:
            return -100.0
        return self._score_trainer(battle, trainer_id) - self._score_trainer(
            battle, 1 - trainer_id
        )

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.perf_counter()
        possible_actions = battle.get_possible_actions(trainer_id)

        if len(possible_actions) == 1:
            return possible_actions[0]

        # Storage: {action_index: [list_of_scores]}
        action_scores = {i: [] for i in range(len(possible_actions))}

        action_idx = 0
        sim_count = 0

        # 1. Simulation Loop
        while (time.perf_counter() - start_time) < self.time_budget:
            action_to_test = possible_actions[action_idx]

            score = self._run_single_simulation(battle, action_to_test, trainer_id)
            action_scores[action_idx].append(score)

            action_idx = (action_idx + 1) % len(possible_actions)
            sim_count += 1

        # 2. Aggregate Results
        avg_scores = []
        counts = []

        for i in range(len(possible_actions)):
            scores = action_scores[i]
            if scores:
                avg_scores.append(np.mean(scores))
                counts.append(len(scores))
            else:
                avg_scores.append(-999.0)  # Penalty for un-simulated moves
                counts.append(0)

        # 3. Log and Decide
        if verbose:
            self._log(sim_count, possible_actions, avg_scores, counts)

        best_index = np.argmax(avg_scores)
        return possible_actions[best_index]

    def _log(self, total_sims: int, actions: list, avg_scores: list, counts: list):
        logger.info(f"[{self.name}] Total Sims: {total_sims} | Depth: {self.depth}")
        logger.info(f"{'Action':<40} | {'Visits':<8} | {'Avg Score':<10}")
        logger.info("-" * 65)

        # Sort for display so the best moves are at the top
        sorted_indices = np.argsort(avg_scores)[::-1]

        for i in sorted_indices:
            action_str = str(actions[i])[:38]  # Truncate long action names
            logger.info(f"{action_str:<40} | {counts[i]:<8} | {avg_scores[i]:.2f}")
        logger.info("-" * 65)

    def _run_single_simulation(
        self, battle: Battle, action: Action, trainer_id: int
    ) -> float:
        sim_battle = battle.copy()
        opponent_id = 1 - trainer_id

        # Translate and execute Turn 0
        my_sim_action = self.translate_action(action, sim_battle, trainer_id)

        op_possible = sim_battle.get_possible_actions(opponent_id)
        op_action_choice = random.choice(op_possible) if op_possible else None
        op_sim_action = self.translate_action(op_action_choice, sim_battle, opponent_id)

        if trainer_id == 0:
            sim_battle.turn(my_sim_action, op_sim_action)
        else:
            sim_battle.turn(op_sim_action, my_sim_action)

        # Random Rollout
        depth_count = 0
        while not sim_battle.done and depth_count < self.depth:
            p0 = sim_battle.get_possible_actions(0)
            p1 = sim_battle.get_possible_actions(1)
            a0 = random.choice(p0) if p0 else None
            a1 = random.choice(p1) if p1 else None

            if not a0 and not a1:
                break

            sim_battle.turn(a0, a1)
            depth_count += 1

        return self._evaluate_battle(sim_battle, trainer_id)


class TwoSideGreedyAgent(BaseAgent):
    def __init__(self, name="TwoSideGreedy", time_budget=0.5, depth=5):
        super().__init__(name)
        self.time_budget = time_budget
        self.depth = depth
        self.name = repr(self)

    def __repr__(self):
        return f"{self.name}(budget={self.time_budget}s, depth={self.depth})"

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.perf_counter()

        # 1. Identify all possible move combinations (My Move x Opponent Move)
        my_actions = battle.get_possible_actions(trainer_id)
        opponent_id = 1 - trainer_id
        op_actions = battle.get_possible_actions(opponent_id)

        # If forced move, return immediately
        if len(my_actions) == 1:
            return my_actions[0]

        # If opponent has no moves (struggle/wait), treat as single None action
        if not op_actions:
            op_actions = [None]

        # Data Structure: scores[my_action_idx][op_action_idx] = [list of scores]
        # We map indices to allow easy aggregation later
        score_matrix = {
            i: {j: [] for j in range(len(op_actions))} for i in range(len(my_actions))
        }

        # Create a task list of all pairs (i, j) to iterate round-robin
        # This ensures we don't spend all time on just the first action
        tasks = [(i, j) for i in range(len(my_actions)) for j in range(len(op_actions))]
        task_idx = 0
        sim_count = 0

        # 2. Simulation Loop (Time Budgeted)
        while (time.perf_counter() - start_time) < self.time_budget:
            my_idx, op_idx = tasks[task_idx]

            my_action = my_actions[my_idx]
            op_action = op_actions[op_idx]

            # Run simulation with FIXED opponent move for the first turn
            score = self._run_paired_simulation(
                battle, my_action, op_action, trainer_id
            )

            score_matrix[my_idx][op_idx].append(score)

            # Cycle to next pair
            task_idx = (task_idx + 1) % len(tasks)
            sim_count += 1

        # 3. Aggregation (Maximin)
        # We want the move where the Opponent's BEST response (Min score for me)
        # is better than the other moves' worst cases.

        final_values = []  # (my_action_index, worst_case_score, best_op_response_index)

        for i in range(len(my_actions)):
            worst_case_val = float("inf")
            worst_case_op_idx = -1

            # Check all opponent responses for my action i
            for j in range(len(op_actions)):
                outcomes = score_matrix[i][j]
                if outcomes:
                    avg_val = np.mean(outcomes)
                else:
                    # We didn't have time to simulate this pair?
                    # Assume neutral or slight penalty to encourage exploration next time
                    avg_val = -50.0

                # Track minimum (opponent plays optimally against me)
                if avg_val < worst_case_val:
                    worst_case_val = avg_val
                    worst_case_op_idx = j

            final_values.append((i, worst_case_val, worst_case_op_idx))

        # 4. Log and Select
        if verbose:
            self._log_analysis(sim_count, my_actions, op_actions, final_values)

        # Pick the action with the highest worst_case_val
        best_choice = max(final_values, key=lambda x: x[1])
        return my_actions[best_choice[0]]

    def _run_paired_simulation(
        self, battle: Battle, my_action: Action, op_action: Action, trainer_id: int
    ) -> float:
        """Simulates: My Action vs Op Action -> Then Random Rollout."""
        sim_battle = battle.copy()
        opponent_id = 1 - trainer_id

        # Translate actions to the simulation instance
        my_sim_action = self.translate_action(my_action, sim_battle, trainer_id)
        op_sim_action = self.translate_action(op_action, sim_battle, opponent_id)

        # Execute the specific Turn 0 we are testing
        if trainer_id == 0:
            sim_battle.turn(my_sim_action, op_sim_action)
        else:
            sim_battle.turn(op_sim_action, my_sim_action)

        # Standard Random Rollout from depth 1 onwards
        depth_count = 0
        while not sim_battle.done and depth_count < self.depth:
            p0 = sim_battle.get_possible_actions(0)
            p1 = sim_battle.get_possible_actions(1)
            a0 = random.choice(p0) if p0 else None
            a1 = random.choice(p1) if p1 else None

            if not a0 and not a1:
                break

            sim_battle.turn(a0, a1)
            depth_count += 1

        return self._evaluate_state(sim_battle, trainer_id)

    def _score_trainer(self, battle: Battle, t_id: int) -> float:
        trainer = battle.get_trainer_by_id(t_id)
        hp_score = sum(p.hp / p.max_hp for p in trainer.pokemon_team if p.is_alive)
        alive_bonus = sum(0.5 for p in trainer.pokemon_team if p.is_alive)
        item_bonus = sum(0.2 * i.quantity for i in trainer.inventory.values())
        return hp_score + alive_bonus + item_bonus

    def _evaluate_state(self, battle: Battle, trainer_id: int) -> float:
        # Same heuristic evaluation as before
        if battle.winner == trainer_id:
            return 100.0
        if battle.winner is not None:
            return -100.0

        return self._score_trainer(battle, trainer_id) - self._score_trainer(
            battle, 1 - trainer_id
        )

    def _log_analysis(self, total_sims, my_actions, op_actions, final_values):
        """Precise logging showing the 'Worst Case' analysis."""
        logger.info(
            f"[{self.name}] Sims: {total_sims} | Depth: {self.depth} | Strategy: Maximin"
        )
        logger.info(
            f"{'My Action':<30} | {'Worst Case (Score)':<18} | {'Caused by Op Move'}"
        )
        logger.info("-" * 85)

        # Sort by score descending (Best moves first)
        sorted_results = sorted(final_values, key=lambda x: x[1], reverse=True)

        for idx, score, op_idx in sorted_results:
            my_act_str = str(my_actions[idx])[:28]

            if op_idx != -1:
                op_act_str = str(op_actions[op_idx])
            else:
                op_act_str = "Unknown/Unsimulated"

            logger.info(f"{my_act_str:<30} | {score:<18.2f} | {op_act_str}")
        logger.info("-" * 85)


class FlatMCTSAgent(TwoSideGreedyAgent):
    def __init__(self, name="FlatMCTS", time_budget=0.5, depth=5, exploration_c=1.41):
        self.exploration_c = exploration_c
        super().__init__(name, time_budget, depth)

    def __repr__(self):
        return f"{self.name}(budget={self.time_budget}s, depth={self.depth}, C={self.exploration_c})"

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.perf_counter()

        my_actions = battle.get_possible_actions(trainer_id)
        opponent_id = 1 - trainer_id
        op_actions = battle.get_possible_actions(opponent_id)

        if len(my_actions) == 1:
            return my_actions[0]
        if not op_actions:
            op_actions = [None]

        # Data Structures
        N_my = np.zeros(len(my_actions), dtype=int)
        N_pair = {
            i: np.zeros(len(op_actions), dtype=int) for i in range(len(my_actions))
        }
        Q_pair = {
            i: np.zeros(len(op_actions), dtype=float) for i in range(len(my_actions))
        }

        # --- DYNAMIC NORMALIZATION TRACKERS ---
        # We track the global min and max mean-scores seen to normalize correctly
        min_q_val = float("inf")
        max_q_val = -float("inf")

        total_sims = 0

        # --- 1. Warm-up Phase (Expanded) ---
        # We visit every pair (My, Op) once to establish a baseline range
        # This ensures min_q_val and max_q_val are populated
        for i in range(len(my_actions)):
            for j in range(len(op_actions)):
                if (time.perf_counter() - start_time) > self.time_budget:
                    break

                score = self._run_paired_simulation(
                    battle, my_actions[i], op_actions[j], trainer_id
                )

                N_my[i] += 1
                N_pair[i][j] += 1
                Q_pair[i][j] += score

                # Update bounds
                if score < min_q_val:
                    min_q_val = score
                if score > max_q_val:
                    max_q_val = score

                total_sims += 1

        # --- 2. MCTS Loop ---
        while (time.perf_counter() - start_time) < self.time_budget:
            log_total = math.log(total_sims)

            # --- Dynamic Normalization Helper ---
            # Avoid division by zero if all scores are identical
            if max_q_val > min_q_val:

                def normalize(val):
                    return (val - min_q_val) / (max_q_val - min_q_val)
            else:

                def normalize(val):
                    return 0.5

            # A. SELECTION (My Move)
            best_my_idx = -1
            best_ucb_val = -float("inf")

            for i in range(len(my_actions)):
                # Average score for this node
                avg_score = sum(Q_pair[i]) / N_my[i]

                # Apply Dynamic Normalization
                norm_score = normalize(avg_score)

                # UCB Calculation
                ucb = norm_score + self.exploration_c * math.sqrt(log_total / N_my[i])

                if ucb > best_ucb_val:
                    best_ucb_val = ucb
                    best_my_idx = i

            # B. SELECTION (Opponent Move)
            my_idx = best_my_idx
            best_op_idx = -1
            worst_op_val = float("inf")  # Opponent wants to minimize score

            parent_log = math.log(N_my[my_idx])

            for j in range(len(op_actions)):
                n_visits = N_pair[my_idx][j]
                if n_visits == 0:
                    # If a pair inside the selected branch is unvisited, force visit it
                    best_op_idx = j
                    break

                avg_val = Q_pair[my_idx][j] / n_visits
                norm_val = normalize(avg_val)

                # Opponent UCB (Inverted)
                ucb_op = norm_val - self.exploration_c * math.sqrt(
                    parent_log / n_visits
                )

                if ucb_op < worst_op_val:
                    worst_op_val = ucb_op
                    best_op_idx = j

            # C. SIMULATION
            op_idx = best_op_idx
            my_action = my_actions[my_idx]
            op_action = op_actions[op_idx]

            score = self._run_paired_simulation(
                battle, my_action, op_action, trainer_id
            )

            # D. BACKPROPAGATION & Range Update
            N_my[my_idx] += 1
            N_pair[my_idx][op_idx] += 1
            Q_pair[my_idx][op_idx] += score
            total_sims += 1

            # Important: Update the global min/max observed averages
            # We use the new single score to update bounds to ensure the range expands if we find extreme events
            if score < min_q_val:
                min_q_val = score
            if score > max_q_val:
                max_q_val = score

        # --- 3. Final Decision ---
        final_values = []
        for i in range(len(my_actions)):
            worst_case = float("inf")
            best_response_idx = -1

            for j in range(len(op_actions)):
                if N_pair[i][j] > 0:
                    val = Q_pair[i][j] / N_pair[i][j]
                else:
                    val = -999

                if val < worst_case:
                    worst_case = val
                    best_response_idx = j

            final_values.append((i, worst_case, best_response_idx, N_my[i]))

        if verbose:
            self._log_mcts(total_sims, my_actions, op_actions, final_values, N_pair)

        best_choice = max(final_values, key=lambda x: x[1])
        return my_actions[best_choice[0]]

    def _log_mcts(self, total_sims, my_actions, op_actions, final_values, N_pair):
        logger.info(
            f"[{self.name}] Sims: {total_sims} | Depth: {self.depth} | Method: UCB"
        )
        logger.info(
            f"{'My Action':<30} | {'Visits':<6} | {'MinScore':<8} | {'Main Threat'}"
        )
        logger.info("-" * 85)

        sorted_results = sorted(final_values, key=lambda x: x[1], reverse=True)

        for idx, score, op_idx, visits in sorted_results:
            my_act_str = str(my_actions[idx])[:28]
            if op_idx != -1 and score > -900:
                op_act_str = str(op_actions[op_idx])
                # Show how many times we checked this specific threat
                threat_visits = N_pair[idx][op_idx]
                op_act_str += f" (n={threat_visits})"
            else:
                op_act_str = "Unexplored"

            logger.info(
                f"{my_act_str:<30} | {visits:<6} | {score:<8.2f} | {op_act_str}"
            )
        logger.info("-" * 85)
