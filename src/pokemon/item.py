from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import TYPE_CHECKING

from pokemon.message import Message
from pokemon.pokemon import PokemonStatus

if TYPE_CHECKING:
    from pokemon.pokemon import Pokemon


@unique
class ItemCategory(IntEnum):
    MEDICINE = 0
    POKEBALL = 1
    BATTLE = 2
    KEY_ITEM = 3
    BERRY = 4


class ItemError(Exception):
    pass


class ItemEffect(ABC):
    """Abstract base class for item effects."""

    @abstractmethod
    def apply(self, target: Pokemon) -> list[Message]:
        pass

    @abstractmethod
    def can_use(self, target: Pokemon) -> bool:
        pass


@dataclass(frozen=True, slots=True)
class HealHP(ItemEffect):
    amount: int

    def can_use(self, target: Pokemon) -> bool:
        return target.is_alive and target.hp < target.max_hp

    def apply(self, target: Pokemon) -> list[Message]:
        old_hp = target.hp
        target.hp += self.amount
        recovered = target.hp - old_hp
        return [Message(f"{target.surname} recovered {recovered} HP.")]


@dataclass(frozen=True, slots=True)
class HealStatus(ItemEffect):
    status_list: list[tuple[PokemonStatus, ...]] | None = None  # None = All

    def can_use(self, target: Pokemon) -> bool:
        if not target.is_alive:
            return False
        if target.status == PokemonStatus.HEALTHY:
            return False
        if self.status_list and target.status not in self.status_list:
            return False
        return True

    def apply(self, target: Pokemon) -> list[Message]:
        target.status = PokemonStatus.HEALTHY
        return [Message(f"{target.surname} was cured of its status.")]


@dataclass(frozen=True, slots=True)
class ReviveEffect(ItemEffect):
    percent: int  # 50 or 100

    def can_use(self, target: Pokemon) -> bool:
        return not target.is_alive

    def apply(self, target: Pokemon) -> list[Message]:
        target.is_alive = True
        target.hp = (target.max_hp * self.percent) // 100
        target.status = PokemonStatus.HEALTHY
        return [Message(f"{target.surname} was revived!")]


@dataclass(slots=True, frozen=True)
class Item:
    """Class representing an item in the game."""

    id: int
    name: str
    category: ItemCategory
    description: str
    effect: ItemEffect

    def can_use(self, target: Pokemon) -> bool:
        return self.effect.can_use(target)

    def use(self, target: Pokemon) -> list[Message]:
        """Use the item on the target Pokémon."""
        if not self.can_use(target):
            return [Message("It won't have any effect.")]
        return self.effect.apply(target)

    def __repr__(self) -> str:
        return f"<Item: {self.name}>"


# --- Registry System ---


class ItemRegistry:
    _items: list[Item] = []
    _map: dict[str, Item] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: ItemCategory,
        effect: ItemEffect,
        desc: str,
    ) -> Item:
        item_id = len(cls._items)
        item = Item(item_id, name, category, desc, effect)

        cls._items.append(item)
        key = name.replace(" ", "").lower()
        if key in cls._map:
            raise ItemError(f"Duplicate item name: {name}")
        cls._map[key] = item
        return item

    @classmethod
    def get(cls, name_or_id: str | int) -> Item | None:
        if isinstance(name_or_id, int):
            if 0 <= name_or_id < len(cls._items):
                return cls._items[name_or_id]
            return None
        return cls._map.get(str(name_or_id).replace(" ", "").lower())

    @classmethod
    def all(cls) -> list[Item]:
        return cls._items


# --- Registration ---

Potion = ItemRegistry.register(
    "Potion", ItemCategory.MEDICINE, HealHP(20), "Restores 20 HP."
)

SuperPotion = ItemRegistry.register(
    "Super Potion", ItemCategory.MEDICINE, HealHP(50), "Restores 50 HP."
)

HyperPotion = ItemRegistry.register(
    "Hyper Potion", ItemCategory.MEDICINE, HealHP(200), "Restores 200 HP."
)

FullHeal = ItemRegistry.register(
    "Full Heal",
    ItemCategory.MEDICINE,
    HealStatus(),
    "Cures all status problems.",
)

Revive = ItemRegistry.register(
    "Revive",
    ItemCategory.MEDICINE,
    ReviveEffect(50),
    "Revives a fainted Pokémon with half HP.",
)

MaxRevive = ItemRegistry.register(
    "Max Revive",
    ItemCategory.MEDICINE,
    ReviveEffect(100),
    "Revives a fainted Pokémon with full HP.",
)

# --- Accessor Helper ---


class _ItemsMeta(type):
    def __getattr__(cls, name: str) -> Item:
        item = ItemRegistry.get(name)
        if item:
            return item
        raise AttributeError(f"Item '{name}' not found")


class Items(metaclass=_ItemsMeta):
    """
    Access point for all registered items.
    Usage: Items.Potion, Items.Revive.
    """

    pass
