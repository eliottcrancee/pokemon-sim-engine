import sys
import os

# Add src to path so we can import pokemon package
# __file__ is src/pokemon/battle_graph.py
# dirname is src/pokemon
# .. is src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque, Counter
from typing import Dict, List, Tuple

from pokemon.battle import Battle
from pokemon.battle_registry import BattleRegistry


class BattleGraph:
    def __init__(
        self,
        initial_battle: Battle,
        samples_per_step: int = 10,
        max_depth: int = 10,
        max_nodes: int = 1000,
    ):
        self.initial_battle = initial_battle
        self.samples_per_step = samples_per_step
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        # Graph structure: state_hash -> { (action_p1, action_p2) -> [(prob, next_state_hash)] }
        self.transitions: Dict[int, Dict[Tuple[str, str], List[Tuple[float, int]]]] = {}
        # Map hash to Battle object (one representative)
        self.states: Dict[int, Battle] = {}
        # Depth of each state
        self.depths: Dict[int, int] = {}
        self.stop_reason = "Unknown"

    def build(self):
        initial_state = self.initial_battle.copy()
        initial_state.headless = True
        initial_hash = hash(initial_state)

        self.states[initial_hash] = initial_state
        self.depths[initial_hash] = 0

        queue = deque([initial_hash])

        while queue:
            if len(self.states) >= self.max_nodes:
                self.stop_reason = "Max nodes reached"
                break

            current_hash = queue.popleft()
            current_battle = self.states[current_hash]
            current_depth = self.depths[current_hash]

            if current_depth >= self.max_depth:
                # We continue processing other nodes in queue that might be shallower,
                # but we don't expand this one.
                # If all nodes in queue are at max_depth, we will eventually empty the queue.
                continue

            if current_battle.done:
                continue

            # Get possible actions
            actions_p1 = current_battle.get_possible_actions(0)
            actions_p2 = current_battle.get_possible_actions(1)

            # Initialize transitions for this state
            self.transitions[current_hash] = {}

            # Iterate over all action pairs
            for a1 in actions_p1:
                for a2 in actions_p2:
                    action_pair_key = (str(a1), str(a2))
                    outcomes = Counter()

                    # Sample outcomes
                    for _ in range(self.samples_per_step):
                        next_battle = current_battle.copy()
                        next_battle.headless = True
                        next_battle.turn(a1, a2)

                        next_hash = hash(next_battle)
                        outcomes[next_hash] += 1

                        if next_hash not in self.states:
                            if len(self.states) >= self.max_nodes:
                                self.stop_reason = "Max nodes reached"
                                # We can't add this new state.
                                # We should probably stop exploring entirely or just not add this branch?
                                # If we stop entirely, we break the outer loops.
                                break

                            self.states[next_hash] = next_battle
                            self.depths[next_hash] = current_depth + 1
                            queue.append(next_hash)

                    if self.stop_reason == "Max nodes reached":
                        break

                    # Store probabilities
                    total_samples = sum(outcomes.values())
                    probs = []
                    for next_h, count in outcomes.items():
                        probs.append((count / total_samples, next_h))

                    self.transitions[current_hash][action_pair_key] = probs

                if self.stop_reason == "Max nodes reached":
                    break

        if not queue and self.stop_reason == "Unknown":
            self.stop_reason = "Queue empty (exploration complete)"

    def get_stats(self):
        if not self.states:
            return "Graph not built."

        num_nodes = len(self.states)
        max_depth_reached = max(self.depths.values())

        # Width: max nodes at any depth
        depth_counts = Counter(self.depths.values())
        max_width = max(depth_counts.values()) if depth_counts else 0

        # Probable nodes: nodes that are reached with some significant probability?
        # Or maybe just nodes that have > 1 incoming edge?
        # Let's just count nodes that are part of the graph.
        # The user asked "combien de noeud probable".
        # Maybe they mean nodes that are not extremely rare outcomes?
        # Since we sample, we only find "probable" nodes (unlikely ones might be missed).
        # So "num_nodes" is effectively "probable nodes" found by sampling.

        return {
            "num_nodes": num_nodes,
            "max_depth": max_depth_reached,
            "max_width": max_width,
            "depth_distribution": dict(depth_counts),
            "stop_reason": self.stop_reason,
        }
