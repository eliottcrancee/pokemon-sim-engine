# item.py

import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
from pympler import asizeof

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from envs.pokemon.config import DEBUG, MAX_ITEM_QUANTITY
from envs.pokemon.message import Message
from envs.pokemon.pokemon import Pokemon, PokemonStatus
from envs.pokemon.tensor_cache import ONEHOTCACHE


class ItemError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ItemCategory(Enum):
    """Enum for Item categories."""

    Healing = auto()
    PokeBall = auto()
    Battle = auto()
    KeyItem = auto()

    def __repr__(self):
        return f"ItemCategory.{self.name}"

    def __str__(self):
        return self.name.capitalize()

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


@dataclass
class Item:
    item_id: int
    name: str
    category: ItemCategory
    description: str
    default_quantity: int = field(default=0)

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()
        self.quantity = self.default_quantity

    def validate_inputs(self):
        if not isinstance(self.name, str):
            raise ItemError(f"Name must be a string, not {type(self.name).__name__}")
        if not isinstance(self.category, ItemCategory):
            raise ItemError(
                f"Category must be an ItemCategory, not {type(self.category).__name__}"
            )
        if not isinstance(self.description, str):
            raise ItemError(
                f"Description must be a string, not {type(self.description).__name__}"
            )
        if not isinstance(self.default_quantity, int):
            raise ItemError(
                f"Quantity must be an integer, not {type(self.quantity).__name__}"
            )
        if self.default_quantity < 0:
            raise ItemError(
                f"Quantity must be non-negative, not {self.default_quantity}"
            )
        if self.default_quantity > MAX_ITEM_QUANTITY:
            raise ItemError(
                f"Quantity must be less than {MAX_ITEM_QUANTITY}, not {self.default_quantity}"
            )

    def validate(self, pokemon) -> bool:
        """Validate if the item can be used on the given Pokémon."""
        if DEBUG:
            if not isinstance(pokemon, Pokemon):
                raise ItemError("Target must be an instance of Pokemon")
        return True

    def use(self, pokemon) -> list[Message]:
        """Define the effect of using this item on the given Pokémon."""
        if DEBUG:
            if not isinstance(pokemon, Pokemon):
                raise ItemError("Target must be an instance of Pokemon")
            if self.quantity == 0:
                raise ItemError(f"Item {self.name} is out of stock")
        return []

    def reset(self):
        self.quantity = self.default_quantity

    def __str__(self) -> str:
        return f"{self.name} - {self.description} (x{self.quantity})"

    def __repr__(self):
        return f"Item(id={self.item_id}, name={self.name}, category={self.category}, description={self.description}, quantity={self.quantity})"

    def __hash__(self) -> int:
        return hash(self.item_id)

    def __eq__(self, other) -> bool:
        return self.item_id == other.item_id

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)

    @property
    def one_hot(self) -> torch.Tensor:
        return ONEHOTCACHE.get_one_hot(len(ITEM_LIST), self.item_id)

    @property
    def one_hot_description(self) -> torch.Tensor:
        return ITEM_ONE_HOT_DESCRIPTION


@dataclass
class Potion(Item):
    item_id: int = 0
    name: str = "Potion"
    category: ItemCategory = ItemCategory.Healing
    description: str = "Restores 20 HP"
    default_quantity: int = 1

    def __post_init__(self):
        super().__post_init__()

    def validate(self, pokemon) -> bool:
        super().validate(pokemon)
        return pokemon.hp < pokemon.max_hp and pokemon.is_alive

    def use(self, pokemon) -> list[Message]:
        super().use(pokemon)
        previous_hp = pokemon.hp
        pokemon.hp += 20
        self.quantity -= 1
        return [Message(f"{pokemon.surname} restored {pokemon.hp - previous_hp} HP.")]

    __repr__ = Item.__repr__
    __str__ = Item.__str__
    __hash__ = Item.__hash__
    __eq__ = Item.__eq__


@dataclass
class SuperPotion(Item):
    item_id: int = 1
    name: str = "Super Potion"
    category: ItemCategory = ItemCategory.Healing
    description: str = "Restores 50 HP"
    default_quantity: int = 1

    def __post_init__(self):
        super().__post_init__()

    def validate(self, pokemon) -> bool:
        super().validate(pokemon)
        return pokemon.hp < pokemon.max_hp and pokemon.is_alive

    def use(self, pokemon) -> list[Message]:
        super().use(pokemon)
        previous_hp = pokemon.hp
        pokemon.hp += 50
        self.quantity -= 1
        return [Message(f"{pokemon.surname} restored {pokemon.hp - previous_hp} HP.")]

    __repr__ = Item.__repr__
    __str__ = Item.__str__
    __hash__ = Item.__hash__
    __eq__ = Item.__eq__


@dataclass
class FullHeal(Item):
    item_id: int = 2
    name: str = "Full Heal"
    category: ItemCategory = ItemCategory.Healing
    description: str = "Cures all status conditions"
    default_quantity: int = 1

    def __post_init__(self):
        super().__post_init__()

    def validate(self, pokemon) -> bool:
        super().validate(pokemon)
        return pokemon.status != PokemonStatus.Healthy

    def use(self, pokemon) -> list[Message]:
        super().use(pokemon)
        pokemon.status = PokemonStatus.Healthy
        self.quantity -= 1
        return [Message(f"{pokemon.surname} was fully cured.")]

    __repr__ = Item.__repr__
    __str__ = Item.__str__
    __hash__ = Item.__hash__
    __eq__ = Item.__eq__


@dataclass
class Revive(Item):
    item_id: int = 3
    name: str = "Revive"
    category: ItemCategory = ItemCategory.Healing
    description: str = "Revives a fainted Pokémon with 50% HP"
    default_quantity: int = 1

    def __post_init__(self):
        super().__post_init__()

    def validate(self, pokemon) -> bool:
        super().validate(pokemon)
        return not pokemon.is_alive

    def use(self, pokemon) -> list[Message]:
        super().use(pokemon)
        pokemon.hp = pokemon.max_hp // 2
        self.quantity -= 1
        return [Message(f"{pokemon.surname} was revived with 50% HP.")]

    __repr__ = Item.__repr__
    __str__ = Item.__str__
    __hash__ = Item.__hash__
    __eq__ = Item.__eq__


ITEM_LIST = [Potion, SuperPotion, FullHeal, Revive]

ITEM_MAP = {item.name.replace(" ", "").lower(): item for item in ITEM_LIST}

assert len({item.item_id for item in ITEM_LIST}) == len(ITEM_LIST), "Duplicate Item IDs"

assert all(item.item_id == index for index, item in enumerate(ITEM_LIST)), (
    "Invalid Item IDs"
)


class _ItemAccessor:
    def __getattr__(self, attr: str) -> Item:
        t = ITEM_MAP.get(attr.lower())
        if t:
            return t
        raise AttributeError(f"No Item named '{attr}'")

    def __iter__(self):
        return iter(ITEM_LIST)

    def __len__(self):
        return len(ITEM_LIST)

    def __repr__(self):
        return "Items(" + ", ".join(t.name for t in ITEM_LIST) + ")"

    def by_id(self, item_id: int) -> Item:
        return ITEM_LIST[item_id]

    def get(self, name: str) -> Item | None:
        return ITEM_MAP.get(name)

    @property
    def names(self) -> list[str]:
        return list(ITEM_MAP.keys())


ItemAccessor = _ItemAccessor()


ITEM_ONE_HOT_DESCRIPTION = [item.name for item in ItemAccessor]

ITEM_ONE_HOT_PADDING = torch.tensor([0] * (len(ITEM_LIST)))
