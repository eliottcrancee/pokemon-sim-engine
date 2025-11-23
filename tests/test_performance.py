# tests/test_performance.py

import time

from joblib import Parallel, cpu_count, delayed

from pokemon.agent import FirstAgent
from pokemon.battle import Battle
from pokemon.item import ItemAccessor
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer


def battle_generator():
    """Creates a standard battle instance for performance testing."""

    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Chimchar(level=10),
        ],
        inventory={
            ItemAccessor.Potion.name: ItemAccessor.Potion(default_quantity=1),
            ItemAccessor.SuperPotion.name: ItemAccessor.SuperPotion(default_quantity=1),
        },
    )
    trainer2 = Trainer(
        name="Gary",
        pokemon_team=[PokemonAccessor.Piplup(level=11)],
        inventory={
            ItemAccessor.Potion.name: ItemAccessor.Potion(default_quantity=1),
            ItemAccessor.SuperPotion.name: ItemAccessor.SuperPotion(default_quantity=1),
        },
    )
    return Battle(trainer1, trainer2)


def test_performance_battle_turns(benchmark):
    """Benchmarks how many turns of a battle can be run per second."""
    agent_0 = FirstAgent()
    agent_1 = FirstAgent()
    battle = battle_generator()

    def run_turn():
        if battle.done:
            battle.reset()

        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)
        battle.turn(action_0, action_1)

    benchmark(run_turn)


def battle_worker(duration: int, battle: Battle = None) -> int:
    """Worker function to run continuous battle turns for a specified duration
    and report the total number of turns completed.
    """
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


def parallel_stress_test_turns(duration: int = 10):
    """Performs a parallel stress test to determine the maximum number of battle
    turns possible across all CPU cores within a specified duration.
    """

    global DEBUG
    DEBUG = False  # Disable debug mode for performance testing
    num_process = cpu_count()

    print("\n--- Parallel Battle Turn Stress Test ---")
    print(f"Target Duration: {duration}s")
    print(f"Using {num_process} CPU cores...")

    start_time = time.perf_counter()
    results = Parallel(n_jobs=num_process)(
        delayed(battle_worker)(duration, battle_generator()) for _ in range(num_process)
    )
    end_time = time.perf_counter()

    total_turns = sum(results)

    print(f"Test completed in {end_time - start_time:.2f} seconds.")
    print(f"Total turns completed across all {num_process} processes: {total_turns:,}")
    print(f"Overall Turns per Second: {total_turns / (end_time - start_time):,.2f} TPS")


if __name__ == "__main__":
    parallel_stress_test_turns()
