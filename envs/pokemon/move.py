# move.py

import os
import random
import sys
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import torch
from pympler import asizeof

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from envs.pokemon.config import DEBUG
from envs.pokemon.message import Message
from envs.pokemon.pokemon_type import PokemonType, PokemonTypeAccessor
from envs.pokemon.tensor_cache import ONEHOTCACHE

ACCURACY_MODIFIERS = [0.33, 0.38, 0.43, 0.5, 0.6, 0.75, 1, 1.33, 1.67, 2, 2.33, 2.67, 3]


class MoveError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MoveCategory(Enum):
    """Enum for Move categories."""

    Physical = auto()
    Special = auto()
    Status = auto()

    def __repr__(self):
        return f"MoveCategory.{self.name}"

    def __str__(self):
        return self.name.capitalize()

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, MoveCategory):
            return self.value == other.value
        return NotImplemented


@dataclass
class Move:
    """Dataclass representing a Pokémon move with attributes and methods."""

    move_id: int
    name: str
    category: MoveCategory
    type: PokemonType
    power: int
    accuracy: int
    pp: int

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()

    def validate_inputs(self):
        if not isinstance(self.name, str):
            raise MoveError(f"Name must be a string, not {type(self.name).__name__}")
        if not isinstance(self.move_id, int):
            raise MoveError(
                f"Move ID must be an integer, not {type(self.move_id).__name__}"
            )
        if not isinstance(self.category, MoveCategory):
            raise MoveError(
                f"Category must be a MoveCategory, not {type(self.category).__name__}"
            )
        if not isinstance(self.type, PokemonType):
            raise MoveError(
                f"Type must be a PokemonType, not {type(self.type).__name__}"
            )
        if not isinstance(self.power, int) or self.power < 0:
            raise MoveError(f"Power must be a non-negative integer, not {self.power}")
        if not isinstance(self.accuracy, int) or not 0 <= self.accuracy <= 100:
            raise MoveError(
                f"Accuracy must be an integer between 0 and 100, not {self.accuracy}"
            )
        if not isinstance(self.pp, int) or self.pp < 0:
            raise MoveError(f"PP must be a non-negative integer, not {self.pp}")

    def calculate_damage(self, user, target) -> tuple[int, int, bool, bool, float]:
        if self.power == 0 or self.category == MoveCategory.Status:
            return 0, 0, False, False, 1.0

        attack_stat = (
            user.attack if self.category == MoveCategory.Physical else user.sp_attack
        )
        defense_stat = (
            target.defense
            if self.category == MoveCategory.Physical
            else target.sp_defense
        )

        base_damage = (
            user.level_factor * self.power * (attack_stat / defense_stat)
        ) / 50 + 2
        effectiveness = self.type.effectiveness_against(target.types)
        damage = base_damage * effectiveness * random.uniform(0.85, 1.0)

        critical = random.random() < 0.0625  # 1/16 chance
        if critical:
            damage *= 1.5

        stab = self.type in user.types
        if stab:
            damage *= 1.5

        accuracy = int(
            self.accuracy * ACCURACY_MODIFIERS[user._modifiers["accuracy"] + 6]
        )
        hit = random.randrange(100) < accuracy

        return (
            int(damage) if hit else 0,
            int(base_damage),
            critical,
            stab,
            effectiveness,
        )

    def secondary_effect(
        self, user, target, damage, base_damage, critical, stab, effectiveness
    ) -> list[Message]:
        return []

    def __str__(self) -> str:
        return f"{self.name} | Type: {self.type.name} | PP: {self.pp}/{self.pp}"

    def __repr__(self):
        return f"Move(id={self.move_id}, name={self.name}, category={self.category}, type={self.type}, power={self.power}, accuracy={self.accuracy}, pp={self.pp})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return NotImplemented

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)

    @property
    def one_hot(self) -> torch.Tensor:
        return ONEHOTCACHE.get_one_hot(len(MOVE_LIST), self.move_id)

    @cached_property
    def one_hot_description(self) -> torch.Tensor:
        return MOVE_ONE_HOT_DESCRIPTION


########## Define moves ##########

SelfHit = Move(
    0, "Self Hit", MoveCategory.Physical, PokemonTypeAccessor.TypeNone, 40, 100, 1
)
Tackle = Move(
    1, "Tackle", MoveCategory.Physical, PokemonTypeAccessor.Normal, 35, 95, 35
)
Scratch = Move(
    2, "Scratch", MoveCategory.Physical, PokemonTypeAccessor.Normal, 40, 100, 35
)
RazorLeaf = Move(
    3, "Razor Leaf", MoveCategory.Physical, PokemonTypeAccessor.Grass, 55, 95, 25
)
Pound = Move(4, "Pound", MoveCategory.Physical, PokemonTypeAccessor.Normal, 40, 100, 35)
ThunderShock = Move(
    5,
    "Thunder Shock",
    MoveCategory.Physical,
    PokemonTypeAccessor.Electric,
    40,
    100,
    30,
)


class StruggleMove(Move):
    def __init__(self):
        super().__init__(
            6,
            "Struggle",
            MoveCategory.Physical,
            PokemonTypeAccessor.Normal,
            50,
            100,
            1,
        )

    def secondary_effect(
        self, user, target, damage, base_damage, critical, stab, effectiveness
    ) -> list[Message]:
        super().secondary_effect(
            user, target, damage, base_damage, critical, stab, effectiveness
        )
        try:
            user._pp[user.moves.index(self)] -= 1
        except ValueError:
            pass
        if user.hp <= 0:
            return []
        lose_hp = min(user.hp, user.max_hp // 4)
        user.hp -= lose_hp
        return [Message(f"{user.surname} lost {lose_hp} HP.")]


Struggle = StruggleMove()

########## ################# ##########

MOVE_LIST = [SelfHit, Tackle, Scratch, RazorLeaf, Pound, ThunderShock, Struggle]
MOVE_MAP = {move.name.replace(" ", "").lower(): move for move in MOVE_LIST}

assert len({move.move_id for move in MOVE_LIST}) == len(MOVE_LIST), "Duplicate Move IDs"
assert all(move.move_id == index for index, move in enumerate(MOVE_LIST)), (
    "Invalid Move IDs"
)


class _MoveAccessor:
    def __getattr__(self, attr: str) -> Move:
        t = MOVE_MAP.get(attr.lower())
        if t:
            return t
        raise AttributeError(f"No Move named '{attr}'")

    def __iter__(self):
        return iter(MOVE_LIST)

    def __len__(self):
        return len(MOVE_LIST)

    def __repr__(self):
        return "Moves(" + ", ".join(t.name for t in MOVE_LIST) + ")"

    def by_id(self, move_id: int) -> Move:
        return MOVE_LIST[move_id]

    def get(self, name: str) -> Move | None:
        return MOVE_MAP.get(name)

    @property
    def names(self) -> list[str]:
        return list(MOVE_MAP.keys())


MoveAccessor = _MoveAccessor()


MOVE_ONE_HOT_DESCRIPTION = [move.name for move in MoveAccessor]
