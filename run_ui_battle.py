import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pokemon.agent import InputAgent
from pokemon.battle import Battle
from pokemon.item import ItemAccessor
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer
from pokemon.ui import play_ui


def main():
    # Setup Trainers and Pokemon
    pikachu_ash = PokemonAccessor.Pikachu(level=10)
    charmander_ash = PokemonAccessor.Chimchar(level=10)
    # ...
    potion_ash = ItemAccessor.Potion(default_quantity=2)

    ash = Trainer(
        name="Ash",
        pokemon_team=[pikachu_ash, charmander_ash],
        inventory={ItemAccessor.Potion: potion_ash},
    )

    squirtle_gary = PokemonAccessor.Pikachu(level=10)
    bulbasaur_gary = PokemonAccessor.Chimchar(level=10)
    potion_gary = ItemAccessor.Potion(default_quantity=2)

    gary = Trainer(
        name="Gary",
        pokemon_team=[squirtle_gary, bulbasaur_gary],
        inventory={ItemAccessor.Potion: potion_gary},
    )

    # Create Agents
    ash_agent = InputAgent(name="Ash")
    gary_agent = InputAgent(name="Gary")

    # Create Battle
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    print("Starting UI Battle...")
    play_ui(battle, ash_agent, gary_agent)


if __name__ == "__main__":
    main()
