import os
import sys

from pokemon.agents import InputAgent, SmarterHeuristicAgent
from pokemon.battle_registry import BattleRegistry
from pokemon.ui import play_ui


def main():
    print("--- Quick Start UI Battle ---")

    # Load a battle from the registry
    battle_name = "kanto_classic"
    battle = BattleRegistry.get(battle_name)
    
    if not battle:
        print(f"Error: Battle '{battle_name}' not found in registry. Exiting quick start.")
        return

    # Create Agents (You against a Smarter Heuristic AI)
    player_agent = InputAgent()
    ai_agent = SmarterHeuristicAgent()

    print(f"\nStarting a quick UI battle: You vs. {ai_agent.name} in '{battle_name}'!")
    print("Loading...")
    
    play_ui(battle, player_agent, ai_agent)

    print("\n--- Quick Start Complete ---")
    print("For more options (Agent vs. Agent, Tournaments, Performance Tests),")
    print("run 'python src/playground.py' from your terminal.")


if __name__ == "__main__":
    main()
