import os
import sys


# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pokemon.battle import Battle
from pokemon.item import Items
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.trainer import Trainer


def create_trainer(name):
    return Trainer(
        name=name,
        pokemon_team=[
            Pokemon(Pokedex.Pikachu, level=10),
            Pokemon(Pokedex.Charmander, level=10),
            Pokemon(Pokedex.Squirtle, level=10),
        ],
        inventory={
            Items.Potion: 1,
            Items.SuperPotion: 1,
        },
    )


def create_battle():
    ash = create_trainer("Ash")
    gary = create_trainer("Gary")
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)
    return battle


if __name__ == "__main__":
    battle = create_battle()
    print(battle)

    actions = battle.get_possible_actions(0)

    action0 = actions[0]
    action1 = battle.get_possible_actions(1)[0]

    messages = battle.turn(action0, action1)
    for message in messages:
        print(message)

    battle_copy = battle.copy()
    actions_copy = battle_copy.get_possible_actions(0)

    for action in actions_copy:
        print(action)
