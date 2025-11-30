import time
import timeit

from tqdm import tqdm

from pokemon.agents import (
    AlphaBetaAgent,
    BaseAgent,
    BestAttackAgent,
    BestAttackAndPotionAgent,
    InputAgent,
    OneStepUniformExpectimaxAgent,
    RandomAgent,
    RandomAttackAgent,
    SmarterHeuristicAgent,
)
from pokemon.battle import Battle
from pokemon.battle_registry import BattleRegistry
from pokemon.play import play_multiple, play_tournament
from pokemon.ui import play_ui

AGENT_POOL = [
    RandomAgent(),
    RandomAttackAgent(),
    BestAttackAgent(),
    BestAttackAndPotionAgent(),
    SmarterHeuristicAgent(),
    OneStepUniformExpectimaxAgent(),
    # AlphaBetaAgent(depth=1),
]


def select_from_list(prompt: str, options: list[str]) -> str:
    """Generic helper to select an item from a list by its index."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    while True:
        try:
            choice = int(input("> "))
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def select_agent(prompt: str) -> "BaseAgent":
    """Lets the user select an agent from the AGENT_POOL."""
    agent_index = select_from_list(prompt, AGENT_POOL)
    return AGENT_POOL[agent_index]


def get_battle_description(battle: "Battle") -> str:
    """Creates a descriptive string for a battle."""
    t1 = battle.trainers[0]
    t2 = battle.trainers[1]

    t1_team = ", ".join([f"{p.name} (L{p.level})" for p in t1.pokemon_team])
    t2_team = ", ".join([f"{p.name} (L{p.level})" for p in t2.pokemon_team])

    return f"{t1.name} [{t1_team}] vs. {t2.name} [{t2_team}]"


def select_battle_verbose() -> tuple[str, "Battle"]:
    """
    Lets the user select a battle, displaying verbose descriptions.
    Returns the name of the battle and the Battle object.
    """
    battle_names = BattleRegistry.list_battles()
    descriptions = [
        get_battle_description(BattleRegistry.get(name)) for name in battle_names
    ]

    print("\nChoose a battle scenario:")
    for i, name in enumerate(battle_names, 1):
        print(f"  [{i}] {name}: {descriptions[i - 1]}")

    while True:
        try:
            choice = int(input("> "))
            if 1 <= choice <= len(battle_names):
                selected_name = battle_names[choice - 1]
                return selected_name, BattleRegistry.get(selected_name)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def run_ui_mode():
    """Play an interactive game against an AI."""
    print("\n--- UI Mode ---")
    opponent = select_agent("Choose your opponent:")
    battle_name, battle = select_battle_verbose()
    player = InputAgent()

    print(f"\nStarting UI battle: You vs. {opponent.name} in '{battle_name}'")
    print("Loading...")
    time.sleep(1)

    play_ui(battle, player, opponent)


def run_agent_vs_agent_mode():
    """Run a simulation between two agents."""
    print("\n--- Agent vs. Agent Mode ---")
    agent1 = select_agent("Choose Agent 1:")
    agent2 = select_agent("Choose Agent 2:")
    battle_name, battle = select_battle_verbose()

    while True:
        try:
            n_battles = int(input("Enter number of battles to simulate: "))
            if n_battles > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print(
        f"\nRunning {n_battles} battles for '{battle_name}' between '{agent1.name}' and '{agent2.name}'..."
    )

    wins1, draws, wins2 = play_multiple(
        battle=battle,
        agent_0=agent1,
        agent_1=agent2,
        n_battles=n_battles,
        n_jobs=-1,
        verbose=True,
    )

    print("\n--- Simulation Results ---")
    print(f"{agent1.name} Wins: {wins1} ({wins1 / n_battles:.1%})")
    print(f"{agent2.name} Wins: {wins2} ({wins2 / n_battles:.1%})")
    print(f"Draws: {draws} ({draws / n_battles:.1%})")
    print("--------------------------")


def run_tournament_mode():
    """Run a tournament with a pool of agents."""
    print("\n--- Tournament Mode ---")
    battle_name, battle = select_battle_verbose()

    while True:
        try:
            n_matches = int(input("Enter total number of matches to run: "))
            n_battles_per_match = int(input("Enter battles per match: "))
            if n_matches > 0 and n_battles_per_match > 0:
                break
            else:
                print("Please enter positive numbers.")
        except ValueError:
            print("Please enter valid numbers.")

    print(f"\nStarting tournament for '{battle_name}'...")

    final_rankings = play_tournament(
        agent_pool=AGENT_POOL,
        battle=battle,
        n_matches=n_matches,
        n_battles_per_match=n_battles_per_match,
    )

    max_name_length = max(len(agent.name) for agent in final_rankings)

    print("\n--- Final Tournament Rankings ---")
    print(f"{'Rank':<5} {'Agent':<{max_name_length}} {'Rating':<10} {'Deviation':<10}")
    print("-" * (5 + max_name_length + 10 + 10 + 3))
    for i, agent in enumerate(final_rankings, 1):
        print(
            f"{i:<5} {agent.name:<{max_name_length}} {agent.rating:<10.2f} {agent.rating_deviation:<10.2f}"
        )
    print("-" * (5 + max_name_length + 10 + 10 + 3))


def _perform_agent_action(agent, battle_template):
    """Helper function to be timed by timeit."""
    battle_instance = battle_template.copy()
    _ = agent.get_action(battle_instance, 0)


def run_performance_test_mode():
    """Test the performance (execution time) of an agent's get_action method."""
    print("\n--- Agent Performance Test Mode ---")

    agent_instance = select_agent("Select agent to test performance:")
    battle_name, battle_template = select_battle_verbose()

    if battle_template is None:
        print("Invalid battle selected. Returning to main menu.")
        return

    while True:
        try:
            num_repetitions = int(
                input("Enter number of repetitions for timeit (e.g., 100): ")
            )
            if num_repetitions > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print(
        f"\nMeasuring performance of '{agent_instance.name}' on '{battle_name}' for {num_repetitions} repetitions..."
    )

    timer = timeit.Timer(lambda: _perform_agent_action(agent_instance, battle_template))

    try:
        times = []
        for _ in tqdm(range(num_repetitions), desc="Running Performance Test"):
            times.append(timer.timeit(number=1))

        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)
        std_dev = (
            (sum((x - avg_time) ** 2 for x in times) / len(times)) ** 0.5
            if len(times) > 1
            else 0.0
        )

        print("\n--- Performance Results ---")
        print(f"Agent: {agent_instance.name}")
        print(f"Battle: {battle_name}")
        print(f"Number of individual calls measured: {num_repetitions}")
        print(f"Min time per call: {min_time:.6f} seconds")
        print(f"Max time per call: {max_time:.6f} seconds")
        print(f"Average time per call: {avg_time:.6f} seconds")
        print(f"Standard deviation: {std_dev:.6f} seconds")
        print("---------------------------")

    except Exception as e:
        print(f"\nAn error occurred during performance testing: {e}")
        print(
            "Please ensure the selected agent and battle are compatible and have valid actions."
        )
        import traceback

        traceback.print_exc()


def main():
    """
    Main interactive menu for the Pokémon Playground.
    """
    while True:
        print("\nWelcome to the Pokémon Playground!")
        print("What would you like to do?")
        print("  [1] Play an interactive game (UI Mode)")
        print("  [2] Simulate Agent vs. Agent")
        print("  [3] Run a Tournament")
        print("  [4] Test Agent Performance (Timeit)")
        print("  [5] Exit")

        try:
            choice = input("> ")
            if choice == "1":
                run_ui_mode()
            elif choice == "2":
                run_agent_vs_agent_mode()
            elif choice == "3":
                run_tournament_mode()
            elif choice == "4":
                run_performance_test_mode()
            elif choice == "5":
                print("Exiting playground. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 5.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Returning to the main menu.")

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
