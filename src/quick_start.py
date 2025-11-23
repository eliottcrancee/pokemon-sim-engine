import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pokemon.agents import InputAgent, OneStepUniformExpectimaxAgent
from pokemon.battle import Battle
from pokemon.item import ItemAccessor
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer
from pokemon.ui import play_ui


def main():
    ash = Trainer(
        name="Ash",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Charmander(level=12),
            PokemonAccessor.Squirtle(level=12),
        ],
        inventory={
            ItemAccessor.Potion.name: ItemAccessor.Potion(default_quantity=1),
            ItemAccessor.SuperPotion.name: ItemAccessor.SuperPotion(default_quantity=1),
        },
    )

    gary = Trainer(
        name="Gary",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Charmander(level=12),
            PokemonAccessor.Squirtle(level=12),
        ],
        inventory={
            ItemAccessor.Potion.name: ItemAccessor.Potion(default_quantity=1),
            ItemAccessor.SuperPotion.name: ItemAccessor.SuperPotion(default_quantity=1),
        },
    )

    # Create Agents
    ash_agent = InputAgent()
    gary_agent = OneStepUniformExpectimaxAgent()

    # Create Battle
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    print("Starting UI Battle...")
    play_ui(battle, ash_agent, gary_agent)


if __name__ == "__main__":
    main()
