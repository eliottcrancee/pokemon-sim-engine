# tests/test_performance.py

import os
import time

import pytest
from joblib import Parallel, cpu_count, delayed

from pokemon.agent import RandomAgent
from pokemon.battle import Battle
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer


@pytest.fixture
def battle_instance():
    """Fixture to create a standard battle instance for testing."""
    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Chimchar(level=10),
        ],
    )
    trainer2 = Trainer(name="Gary", pokemon_team=[PokemonAccessor.Piplup(level=11)])
    return Battle(trainer1, trainer2)


def test_performance_battle_turns(benchmark, battle_instance):
    """Benchmarks how many turns of a battle can be run per second."""
    agent_0 = RandomAgent()
    agent_1 = RandomAgent()
    battle = battle_instance

    def run_turn():
        if battle.done:
            battle.reset()

        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)
        battle.turn(action_0, action_1)

    benchmark(run_turn)


def battle_worker(duration: int):
    """Worker function to run continuous battle turns for a specified duration
    and report the total number of turns completed.
    """
    # Create a fresh, independent battle instance for this process
    trainer1 = Trainer(
        name=f"Ash_{os.getpid()}",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Chimchar(level=10),
        ],
    )
    trainer2 = Trainer(
        name=f"Gary_{os.getpid()}", pokemon_team=[PokemonAccessor.Piplup(level=11)]
    )
    battle = Battle(trainer1, trainer2)

    agent_0 = RandomAgent()
    agent_1 = RandomAgent()

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


# NOTE: This function does not use the 'benchmark' fixture
def test_parallel_stress_test_turns():
    """Performs a parallel stress test to determine the maximum number of battle
    turns possible across all CPU cores within 10 seconds.
    """
    # --- Configuration ---
    # Duration for the stress test (in seconds)
    DURATION = 10
    # Use all available CPU cores
    NUM_PROCESSES = cpu_count()

    global DEBUG
    DEBUG = False  # Disable debug mode for performance testing

    print("\n--- Parallel Battle Turn Stress Test ---")
    print(f"Target Duration: {DURATION}s")
    print(f"Using {NUM_PROCESSES} CPU cores...")

    # --- Start Processes ---

    start_time = time.perf_counter()
    results = Parallel(n_jobs=NUM_PROCESSES)(
        delayed(battle_worker)(DURATION) for _ in range(NUM_PROCESSES)
    )
    end_time = time.perf_counter()

    # --- Wait and Aggregate Results ---
    total_turns = sum(results)

    # --- Report ---
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
    print(
        f"Total turns completed across all {NUM_PROCESSES} processes: {total_turns:,}"
    )
    print(f"Overall Turns per Second: {total_turns / (end_time - start_time):,.2f} TPS")


if __name__ == "__main__":
    test_parallel_stress_test_turns()
