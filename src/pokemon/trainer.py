# trainer.py

from dataclasses import dataclass, field
from functools import cached_property

import torch
from pympler import asizeof

from pokemon.config import (
    DEBUG,
    MAX_ITEM_QUANTITY,
    MAX_ITEMS_PER_TRAINER,
    MAX_POKEMON_PER_TRAINER,
)
from pokemon.item import Item, ItemAccessor
from pokemon.pokemon import (
    POKEMON_ONE_HOT_DESCRIPTION,
    POKEMON_ONE_HOT_PADDING,
    POKEMON_STATUS_ONE_HOT_DESCRIPTION,
    POKEMON_STATUS_ONE_HOT_PADDING,
    Pokemon,
)

MAX_MOVES = 4

POKEMON_PADDING_CACHE = {
    i: torch.cat([POKEMON_ONE_HOT_PADDING] * (MAX_POKEMON_PER_TRAINER - i))
    for i in range(MAX_POKEMON_PER_TRAINER)
}
POKEMON_PADDING_CACHE[MAX_POKEMON_PER_TRAINER] = torch.tensor([])

POKEMON_STATUS_PADDING_CACHE = {
    i: torch.cat([POKEMON_STATUS_ONE_HOT_PADDING] * (MAX_POKEMON_PER_TRAINER - i))
    for i in range(MAX_POKEMON_PER_TRAINER)
}
POKEMON_STATUS_PADDING_CACHE[MAX_POKEMON_PER_TRAINER] = torch.tensor([])

POKEMON_ZERROS_PADDING_CACHE = {
    i: torch.cat([torch.tensor([0.0])] * (MAX_POKEMON_PER_TRAINER - i))
    for i in range(MAX_POKEMON_PER_TRAINER)
}
POKEMON_ZERROS_PADDING_CACHE[MAX_POKEMON_PER_TRAINER] = torch.tensor([])

ZEROS_TENSOR_CACHE = {
    i: torch.zeros(i)
    for i in range(MAX_POKEMON_PER_TRAINER + MAX_ITEMS_PER_TRAINER + 30)
}

EMPTY_ITEM = ItemAccessor.Potion(0)


class TrainerError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


@dataclass
class Trainer:
    name: str
    pokemon_team: list[Pokemon] = field(default_factory=list)
    inventory: dict[str, Item] = field(default_factory=dict)

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()

    def copy(self):
        cls = self.__class__
        new_trainer = cls.__new__(cls)
        new_trainer.name = self.name
        new_trainer.pokemon_team = [pokemon.copy() for pokemon in self.pokemon_team]
        new_trainer.inventory = {
            name: item.copy() for name, item in self.inventory.items()
        }
        return new_trainer

    def validate_inputs(self):
        if not isinstance(self.name, str):
            raise TrainerError(f"Name must be a string, not {type(self.name).__name__}")
        if not isinstance(self.pokemon_team, list):
            raise TrainerError(
                f"Pokemon team must be a list, not {type(self.pokemon_team).__name__}"
            )
        if not all(isinstance(pokemon, Pokemon) for pokemon in self.pokemon_team):
            raise TrainerError(
                f"Pokemon team must be a list of Pokemon, not {type(self.pokemon_team).__name__}"
            )
        if not isinstance(self.inventory, dict):
            raise TrainerError(
                f"Inventory must be a dictionary, not {type(self.inventory).__name__}"
            )
        if len(self.pokemon_team) > MAX_POKEMON_PER_TRAINER:
            raise TrainerError(f"Pokemon team cannot exceed {MAX_POKEMON_PER_TRAINER}")
        if len(self.inventory) > MAX_ITEMS_PER_TRAINER:
            raise TrainerError(f"Inventory cannot exceed {MAX_ITEMS_PER_TRAINER}")
        if not all(
            isinstance(k, str) and isinstance(v, Item)
            for k, v in self.inventory.items()
        ):
            raise TrainerError(
                "Inventory must be a dictionary of item names to Item objects."
            )

    def switch_pokemon(self, new_pokemon):
        if DEBUG:
            if not isinstance(new_pokemon, Pokemon):
                raise TrainerError(
                    f"New Pokemon must be an instance of Pokemon, not {type(new_pokemon).__name__}"
                )
            if new_pokemon not in self.pokemon_team:
                raise TrainerError("New Pokemon must be in the team")
            if not new_pokemon.is_alive:
                raise TrainerError("New Pokemon must be alive")
            if new_pokemon == self.pokemon_team[0]:
                raise TrainerError("New Pokemon must be different from the current one")
        self.pokemon_team[0].clear()
        self.pokemon_team.remove(new_pokemon)
        self.pokemon_team.insert(0, new_pokemon)

    @property
    def lowest_alive_pokemon(self) -> Pokemon:
        for pokemon in self.pokemon_team:
            if pokemon.is_alive:
                return pokemon
        return None

    @property
    def is_defeated(self) -> bool:
        return all(not pokemon.is_alive for pokemon in self.pokemon_team)

    def switch_to_lowest_alive_pokemon(self):
        self.switch_pokemon(self.lowest_alive_pokemon)

    def clear_team(self):
        for pokemon in self.pokemon_team:
            pokemon.clear()

    def reset_team(self):
        for pokemon in self.pokemon_team:
            pokemon.reset()

    def reset_inventory(self):
        for item in self.inventory.values():
            item.reset()

    def reset(self):
        self.reset_team()
        self.reset_inventory()

    def __repr__(self) -> str:
        return f"Trainer(name={self.name}, pokemons={self.pokemon_team}, items={self.inventory})"

    def __str__(self) -> str:
        return self.name

    @property
    def str_pokemon_team(self) -> str:
        text = f"{self.name}'s Pokémon Team:"
        for i, pokemon in enumerate(self.pokemon_team):
            text += f"\n[{i + 1}] {pokemon}"
        return text

    @property
    def str_inventory(self) -> str:
        text = f"{self.name}'s Inventory:"
        if not self.inventory:
            text += "\nEmpty..."
        else:
            for item in self.inventory.values():
                if item.quantity > 0:
                    text += f"\n{str(item)}"
        return text

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)

    @cached_property
    def pokemon_level_tensor(self) -> torch.Tensor:
        return torch.tensor([pokemon.level / 100 for pokemon in self.pokemon_team])

    @property
    def tensor(self) -> torch.Tensor:
        team_size = len(self.pokemon_team)

        if self.pokemon_team:
            tensor = torch.cat(
                [
                    torch.cat([pokemon.one_hot for pokemon in self.pokemon_team]),
                    POKEMON_PADDING_CACHE[team_size],
                    torch.tensor([pokemon.hp_ratio for pokemon in self.pokemon_team]),
                    POKEMON_ZERROS_PADDING_CACHE[team_size],
                    self.pokemon_level_tensor,
                    POKEMON_ZERROS_PADDING_CACHE[team_size],
                    torch.cat(
                        [pokemon.status.one_hot for pokemon in self.pokemon_team]
                    ),
                    POKEMON_STATUS_PADDING_CACHE[team_size],
                    torch.tensor(
                        [
                            int(self.pokemon_team[0]._confused),
                            int(self.pokemon_team[0]._taunted),
                        ]
                    ),
                    torch.tensor(
                        [
                            self.inventory.get(item.name, EMPTY_ITEM).quantity
                            / MAX_ITEM_QUANTITY
                            for item in ItemAccessor
                        ]
                    ),
                ]
            )

        elif not self.pokemon_team:
            tensor = torch.cat(
                [
                    POKEMON_PADDING_CACHE[0],
                    POKEMON_ZERROS_PADDING_CACHE[team_size],
                    self.pokemon_level_tensor,
                    POKEMON_ZERROS_PADDING_CACHE[team_size],
                    POKEMON_STATUS_PADDING_CACHE[0],
                    ZEROS_TENSOR_CACHE[2],
                    torch.tensor(
                        [
                            self.inventory.get(item.name, EMPTY_ITEM).quantity
                            / MAX_ITEM_QUANTITY
                            for item in ItemAccessor
                        ]
                    ),
                ]
            )

        return tensor

    @cached_property
    def tensor_description(self):
        return TRAINER_TENSOR_DESCRIPTION


TRAINER_TENSOR_DESCRIPTION = []
TRAINER_TENSOR_DESCRIPTION = []
for i in range(MAX_POKEMON_PER_TRAINER):
    TRAINER_TENSOR_DESCRIPTION += [
        f"Pokemon {i} " + text for text in POKEMON_ONE_HOT_DESCRIPTION
    ]
TRAINER_TENSOR_DESCRIPTION += [
    f"Pokemon {i} HP Ratio" for i in range(MAX_POKEMON_PER_TRAINER)
]
TRAINER_TENSOR_DESCRIPTION += [
    f"Pokemon {i} Level" for i in range(MAX_POKEMON_PER_TRAINER)
]
for i in range(MAX_POKEMON_PER_TRAINER):
    TRAINER_TENSOR_DESCRIPTION += [
        f"Pokemon {i} " + text for text in POKEMON_STATUS_ONE_HOT_DESCRIPTION
    ]
TRAINER_TENSOR_DESCRIPTION += ["Active Pokemon Confused"]
TRAINER_TENSOR_DESCRIPTION += ["Active Pokemon Taunted"]
TRAINER_TENSOR_DESCRIPTION += [f"Item {item.name} Quantity" for item in ItemAccessor]
