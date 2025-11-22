# tests/test_performance.py

import multiprocessing as mp
import os
import time

import pytest

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


def battle_worker(duration: int, turns_queue: mp.Queue):
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
    start_time = time.time()

    # Run the loop until the duration is exceeded
    while time.time() - start_time < duration:
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
    turns_queue.put(turns_count)


# NOTE: This function does not use the 'benchmark' fixture
def test_parallel_stress_test_turns():
    """Performs a parallel stress test to determine the maximum number of battle
    turns possible across all CPU cores within 10 seconds.
    """
    # --- Configuration ---
    # Duration for the stress test (in seconds)
    DURATION = 10
    # Use all available CPU cores
    NUM_PROCESSES = mp.cpu_count()

    global DEBUG
    DEBUG = False  # Disable debug mode for performance testing

    print("\n--- Parallel Battle Turn Stress Test ---")
    print(f"Target Duration: {DURATION}s")
    print(f"Using {NUM_PROCESSES} CPU cores...")

    # A Queue to collect the turn count from each worker process
    turns_queue = mp.Queue()
    processes = []

    # --- Start Processes ---
    for i in range(NUM_PROCESSES):
        # Pass the duration and the queue to the worker
        p = mp.Process(target=battle_worker, args=(DURATION, turns_queue))
        processes.append(p)
        p.start()

    # --- Wait and Aggregate Results ---
    # Wait for all processes to complete their DURATION time limit
    for p in processes:
        p.join()

    total_turns = 0
    # Collect the results from the queue
    while not turns_queue.empty():
        total_turns += turns_queue.get()

    # --- Report ---
    print(f"Test completed in approximately {DURATION}s.")
    print(
        f"Total turns completed across all {NUM_PROCESSES} processes: {total_turns:,}"
    )


if __name__ == "__main__":
    test_parallel_stress_test_turns()
