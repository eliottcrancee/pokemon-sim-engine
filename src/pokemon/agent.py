# agent.py

import math
import os
import random
import sys
import time

import numpy as np

from pokemon.trainer import Trainer

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from loguru import logger

from pokemon.action import Action, ActionError
from pokemon.battle import Battle

CHOICE = random.choice
RANDOM = random.random

logger.remove()
logger.add("agent.log", rotation="1 MB", mode="w")


class BaseAgent:
    def __init__(self, name):
        self.name = name

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        raise NotImplementedError

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


class RandomAgent(BaseAgent):
    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        action = random.choice(actions)
        return action


class InputAgent(BaseAgent):
    def __init__(self, name="InputAgent"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        return actions


class GreedyEvaluateAgent(BaseAgent):
    def __init__(self, name="GreedyEvaluateAgent", time_budget=0.5, depth=5):
        super().__init__(name)
        self.time_budget = time_budget
        self.depth = depth

    @staticmethod
    def evaluate(battle: Battle, trainer_id: int) -> float:
        if battle.winner == trainer_id:
            return 100.0
        elif battle.winner is not None:
            return -100.0

        my_trainer = battle.get_trainer_by_id(trainer_id)
        op_trainer = battle.get_trainer_by_id(1 - trainer_id)

        def quick_score(trainer: Trainer) -> float:
            score = 0
            count = 0
            for p in trainer.pokemon_team:
                if p.is_alive:
                    score += p.hp / p.max_hp
                    count += 1
            for i in trainer.inventory.values():
                score += 0.4 * i.quantity  # Lesser weight for items than survival
            return score + (count * 0.5)  # Value survival heavily

        return quick_score(my_trainer) - quick_score(op_trainer)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.perf_counter()

        possible_actions = battle.get_possible_actions(trainer_id)
        if not possible_actions:
            return None
        if len(possible_actions) == 1:
            return possible_actions[0]

        # Opponent actions for sampling
        opponent_id = 1 - trainer_id
        op_possible = battle.get_possible_actions(opponent_id)
        if not op_possible:
            op_possible = [None]

        # Performance tracking
        sim_count = 0
        scores = {i: [] for i in range(len(possible_actions))}

        # Profiling timers
        t_copy = 0.0
        t_setup = 0.0
        t_turn0 = 0.0
        t_rollout = 0.0
        t_eval = 0.0

        # Loop until time budget is exhausted
        action_idx = 0
        num_actions = len(possible_actions)

        while (time.perf_counter() - start_time) < self.time_budget:
            action = possible_actions[action_idx]

            # --- SIMULATION START ---
            t0 = time.perf_counter()
            sim_battle = battle.copy()
            t1 = time.perf_counter()
            t_copy += t1 - t0

            # 2. Setup Turn 0
            # We pick a RANDOM opponent move to simulate uncertainty
            op_action_choice = CHOICE(op_possible)

            sim_action = self.translate_action(action, sim_battle, trainer_id)
            sim_op_action = self.translate_action(
                op_action_choice, sim_battle, opponent_id
            )
            t2 = time.perf_counter()
            t_setup += t2 - t1

            # Execute Turn 0
            if trainer_id == 0:
                sim_battle.turn(sim_action, sim_op_action)
            else:
                sim_battle.turn(sim_op_action, sim_action)
            t3 = time.perf_counter()
            t_turn0 += t3 - t2

            # 3. Rollout (Random Policy)
            # We inline the loop logic to avoid function call overhead
            depth_count = 0
            while not sim_battle.done and depth_count < self.depth:
                # Get moves
                p0 = sim_battle.get_possible_actions(0)
                p1 = sim_battle.get_possible_actions(1)

                # Fast Random Selection
                a0 = CHOICE(p0) if p0 else None
                a1 = CHOICE(p1) if p1 else None

                if not a0 and not a1:
                    break

                sim_battle.turn(a0, a1)
                depth_count += 1
            t4 = time.perf_counter()
            t_rollout += t4 - t3

            # 4. Score
            final_score = self.evaluate(sim_battle, trainer_id)
            scores[action_idx].append(final_score)
            t5 = time.perf_counter()
            t_eval += t5 - t4

            # --- SIMULATION END ---

            sim_count += 1
            action_idx = (action_idx + 1) % num_actions

        # Aggregate Results
        avg_scores = []
        for i in range(num_actions):
            if scores[i]:
                avg_scores.append(np.mean(scores[i]))
            else:
                avg_scores.append(
                    -999
                )  # Should not happen if time budget allows at least 1 pass

        best_index = np.argmax(avg_scores)

        if verbose:
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"MonteCarloAgent (Sims: {sim_count}, Depth: {self.depth}, Time: {elapsed_time:.4f}s)"
            )
            logger.info(
                f"Profile: Copy={t_copy:.4f}s, Setup={t_setup:.4f}s, Turn0={t_turn0:.4f}s, Rollout={t_rollout:.4f}s, Eval={t_eval:.4f}s"
            )
            for i, score in enumerate(avg_scores):
                logger.info(f"  Action: {possible_actions[i]} -> MinScore: {score:.4f}")

        return possible_actions[best_index]


class MonteCarloEvaluateAgent(BaseAgent):
    def __init__(
        self, name="TwoSidedMonteCarloAgent", time_budget=0.5, c_param=1.0, depth=2
    ):
        super().__init__(name)
        self.time_budget = time_budget
        self.c_param = c_param
        self.depth = depth  # maximum tree depth

    # ---------------- Evaluation ----------------
    @staticmethod
    def evaluate(battle: Battle, trainer_id: int) -> float:
        if battle.winner == trainer_id:
            return 1.0
        elif battle.winner is not None:
            return -1.0

        my_trainer = battle.get_trainer_by_id(trainer_id)
        op_trainer = battle.get_trainer_by_id(1 - trainer_id)

        def quick_score(trainer: Trainer, opponent: Trainer) -> float:
            from pokemon.pokemon import PokemonStatus

            score = 0.0
            alive_count = 0
            for p in trainer.pokemon_team:
                if p.is_alive:
                    hp_frac = p.hp / p.max_hp
                    alive_count += 1
                    score += 0.6 * hp_frac
                    if hp_frac < 0.8:
                        score += 0.2 * (1 - hp_frac)

                    if p.status == PokemonStatus.Burn:
                        score -= 0.2

            score += 0.3 * alive_count

            for p in opponent.pokemon_team:
                if p.is_alive:
                    score += 0.5 * (1 - p.hp / p.max_hp)
                    if p.status == PokemonStatus.Burn:
                        score += 0.2

            for item in trainer.inventory.values():
                score += 0.05 * item.quantity
            return score

        my_score = quick_score(my_trainer, op_trainer)
        op_score = quick_score(op_trainer, my_trainer)
        diff = my_score - op_score
        return np.tanh(diff)

    # ---------------- Node ----------------
    class Node:
        def __init__(
            self, battle, trainer_id, parent=None, action=None, is_opponent=False
        ):
            self.battle = battle
            self.trainer_id = trainer_id
            self.parent = parent
            self.action = action
            self.is_opponent = is_opponent
            self.children = []
            self.unexpanded_actions = battle.get_possible_actions(trainer_id)
            self.Q = 0.0
            self.N = 0

        def ucb_score(self, c_param):
            if self.N == 0:
                return float("inf")
            return self.Q / self.N + c_param * math.sqrt(
                math.log(self.parent.N) / self.N
            )

        def best_child(self, c_param, maximize=True):
            if maximize:
                return max(self.children, key=lambda c: c.ucb_score(c_param))
            else:
                return min(self.children, key=lambda c: c.ucb_score(c_param))

    # ---------------- MCTS Core ----------------
    def select(self, node):
        while True:
            if node.unexpanded_actions:
                return node
            if not node.children:
                return node
            # Maximize if it's our turn, minimize if opponent
            node = node.best_child(self.c_param, maximize=not node.is_opponent)

    def expand(self, node, action=None):
        if action is None:
            action = node.unexpanded_actions.pop()
        else:
            node.unexpanded_actions.remove(action)

        # Sample opponent action if this is our move, else sample our action if opponent node
        if node.is_opponent:
            cur_id = node.trainer_id
            opp_id = 1 - cur_id
            opp_actions = node.battle.get_possible_actions(opp_id)
            opp_action = CHOICE(opp_actions) if opp_actions else None
            new_battle = node.battle.copy()
            my_action = self.translate_action(opp_action, new_battle, opp_id)
            op_action = self.translate_action(action, new_battle, cur_id)
            if node.trainer_id == 0:
                new_battle.turn(op_action, my_action)
            else:
                new_battle.turn(my_action, op_action)
            next_node = self.Node(
                new_battle,
                node.trainer_id,
                parent=node,
                action=action,
                is_opponent=not node.is_opponent,
            )
        else:
            # It's our turn, opponent will move randomly
            op_id = 1 - node.trainer_id
            opp_actions = node.battle.get_possible_actions(op_id)
            opp_action = CHOICE(opp_actions) if opp_actions else None
            new_battle = node.battle.copy()
            my_action = self.translate_action(action, new_battle, node.trainer_id)
            op_action = self.translate_action(opp_action, new_battle, op_id)
            if node.trainer_id == 0:
                new_battle.turn(my_action, op_action)
            else:
                new_battle.turn(op_action, my_action)
            next_node = self.Node(
                new_battle,
                node.trainer_id,
                parent=node,
                action=action,
                is_opponent=True,
            )

        node.children.append(next_node)
        return next_node

    def _rollout(self, battle, trainer_id, max_depth=10):
        """Simulate a random rollout from a battle state."""
        sim_battle = battle.copy()
        depth = 0
        while not sim_battle.done and depth < max_depth:
            p0_actions = sim_battle.get_possible_actions(0)
            p1_actions = sim_battle.get_possible_actions(1)

            a0 = CHOICE(p0_actions) if p0_actions else None
            a1 = CHOICE(p1_actions) if p1_actions else None

            if not a0 and not a1:
                break

            sim_battle.turn(a0, a1)
            depth += 1

        return self.evaluate(sim_battle, trainer_id)

    def evaluate_leaf(self, node, depth=None):
        if depth is None:
            depth = self.depth

        if depth <= 0 or node.battle.done:
            return self._rollout(node.battle, node.trainer_id)

        if node.unexpanded_actions:
            values = []
            for action in node.unexpanded_actions[:]:
                child = self.expand(node, action)
                val = self.evaluate_leaf(child, depth - 1)
                values.append(val)
            return max(values) if not node.is_opponent else min(values)
        elif node.children:
            child = node.best_child(self.c_param, maximize=not node.is_opponent)
            return self.evaluate_leaf(child, depth - 1)
        else:
            return self._rollout(node.battle, node.trainer_id)

    def backpropagate(self, node, value):
        while node is not None:
            node.N += 1
            # For our nodes maximize, for opponent nodes minimize
            if node.is_opponent:
                node.Q += -value
            else:
                node.Q += value
            node = node.parent

    # ---------------- Agent API ----------------
    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> Action:
        start_time = time.perf_counter()
        actions = battle.get_possible_actions(trainer_id)
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]

        root = self.Node(battle, trainer_id, is_opponent=False)

        while time.perf_counter() - start_time < self.time_budget:
            selected = self.select(root)
            if selected.unexpanded_actions:
                leaf = self.expand(selected)
            else:
                leaf = selected
            value = self.evaluate_leaf(leaf, self.depth)
            self.backpropagate(leaf, value)

        best_child = max(root.children, key=lambda c: c.N)
        if verbose:
            logger.info(
                f"Two-sided MCTS completed: {sum(c.N for c in root.children)} sims"
            )
            for c in root.children:
                logger.info(
                    f"A={c.action}, N={c.N}, Q={c.Q:.2f}, mean={c.Q / max(1, c.N):.2f}"
                )
        return best_child.action
