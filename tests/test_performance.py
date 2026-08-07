from __future__ import annotations

import time
import random

import pytest

from joblib import Parallel, cpu_count, delayed

from pokemon import config as pokemon_config
from pokemon.agents import FirstAgent
from pokemon.agents.search import AlphaBetaAgent, MinimaxAgent
from pokemon.battle import Battle
from pokemon.battle_registry import BattleRegistry

BASELINE_BATTLE = "sinnoh_opening_sparks"


def test_performance_battle_turns(benchmark):
    """Benchmarks how many turns of a battle can be run per second."""
    agent_0 = FirstAgent()
    agent_1 = FirstAgent()
    battle = BattleRegistry.get(BASELINE_BATTLE)
    if battle is None:
        raise AssertionError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")

    def run_turn():
        if battle.done:
            battle.reset()

        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)
        battle.turn(action_0, action_1)

    benchmark(run_turn)


@pytest.mark.parametrize("use_cache", [False, True])
def test_benchmark_minimax_cache(benchmark, use_cache: bool):
    """Benchmarks Minimax action selection with and without transposition cache."""
    battle = BattleRegistry.get(BASELINE_BATTLE)
    if battle is None:
        raise AssertionError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")

    agent = MinimaxAgent(depth=2, use_cache=use_cache, parallelize=False)

    def run_get_action() -> None:
        random.seed(12345)
        agent.get_action(battle.copy(), trainer_id=0, verbose=False)

    benchmark(run_get_action)


@pytest.mark.parametrize("use_cache", [False, True])
def test_benchmark_alphabeta_cache(benchmark, use_cache: bool):
    """Benchmarks Alpha-Beta action selection with and without transposition cache."""
    battle = BattleRegistry.get(BASELINE_BATTLE)
    if battle is None:
        raise AssertionError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")

    agent = AlphaBetaAgent(depth=2, use_cache=use_cache, parallelize=False)

    def run_get_action() -> None:
        random.seed(12345)
        agent.get_action(battle.copy(), trainer_id=0, verbose=False)

    benchmark(run_get_action)


def battle_worker(duration: int, battle: Battle | None = None) -> int:
    """Worker function to run continuous battle turns for a specified duration
    and report the total number of turns completed.
    """
    if battle is None:
        battle = BattleRegistry.get(BASELINE_BATTLE)
        if battle is None:
            raise ValueError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")
    agent_0 = FirstAgent()
    agent_1 = FirstAgent()

    turns_count = 0
    start_time = time.perf_counter()

    # Run the loop until the duration is exceeded
    while time.perf_counter() - start_time < duration:
        if battle.done:
            # Crucially, resetting the battle must be fast
            battle.reset()

        # Get actions
        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)

        # Execute turn
        battle.turn(action_0, action_1)
        turns_count += 1

    # Report the final count back to the main process
    return turns_count


def battle_worker_timed(
    n_runs: int, battle: Battle | None = None
) -> tuple[float, float, float]:
    """Worker function to run continuous battle turns for a specified duration
    and report the total number of turns completed.
    """
    if battle is None:
        battle = BattleRegistry.get(BASELINE_BATTLE)
        if battle is None:
            raise ValueError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")
    agent_0 = FirstAgent()
    agent_1 = FirstAgent()

    reset_time = 0.0
    action_time = 0.0
    turn_time = 0.0

    for _ in range(n_runs):
        if battle.done:
            start_reset = time.perf_counter()
            battle.reset()
            end_reset = time.perf_counter()
            reset_time += end_reset - start_reset

        # Get actions
        start_action = time.perf_counter()
        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)
        end_action = time.perf_counter()
        action_time += end_action - start_action

        # Execute turn
        start_turn = time.perf_counter()
        battle.turn(action_0, action_1)
        end_turn = time.perf_counter()
        turn_time += end_turn - start_turn

    print(
        f"Reset Time: {reset_time:.6f}s, Action Time: {action_time:.6f}s, Turn Time: {turn_time:.6f}s"
    )

    return reset_time, action_time, turn_time


def parallel_stress_test_turns(duration: int = 20) -> None:
    """Performs a parallel stress test to determine the maximum number of battle
    turns possible across all CPU cores within a specified duration.
    """

    pokemon_config.DEBUG = False  # Disable debug mode for performance testing
    num_process = cpu_count()

    print("\n--- Parallel Battle Turn Stress Test ---")
    print(f"Target Duration: {duration}s")
    print(f"Using {num_process} CPU cores...")

    start_time = time.perf_counter()
    results = Parallel(n_jobs=num_process)(
        delayed(battle_worker)(duration) for _ in range(num_process)
    )
    end_time = time.perf_counter()

    total_turns = sum(r for r in results if isinstance(r, int))

    print(f"Test completed in {end_time - start_time:.2f} seconds.")
    print(f"Total turns completed across all {num_process} processes: {total_turns:,}")
    print(f"Overall Turns per Second: {total_turns / (end_time - start_time):,.2f} TPS")


if __name__ == "__main__":
    # battle_worker_timed(1000, battle_generator())
    parallel_stress_test_turns()
