# pokemon.py

import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property

import torch
from pympler import asizeof

# Ensure current working directory is in path
sys.path.append(os.getcwd())  # nopep8

from pokemon.config import DEBUG
from pokemon.message import Message
from pokemon.move import Move, MoveAccessor
from pokemon.pokemon_type import PokemonType, PokemonTypeAccessor
from pokemon.tensor_cache import ONEHOTCACHE

STAT_MODIFIER = [0.25, 0.29, 0.33, 0.40, 0.50, 0.67, 1, 1.5, 2, 2.5, 3, 3.5, 4]


class PokemonError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class PokemonStatus(Enum):
    Healthy = auto()
    Burn = auto()

    def __repr__(self):
        return f"PokemonStatus.{self.name}"

    def __str__(self):
        return self.name.capitalize()

    def __eq__(self, other):
        return self.value == other.value

    @property
    def one_hot(self) -> list[int]:
        return ONEHOTCACHE.get_one_hot(len(STATUS_LIST) + 1, self.value)

    @property
    def one_hot_description(self) -> list[str]:
        return POKEMON_STATUS_ONE_HOT_DESCRIPTION


STATUS_LIST = [status for status in PokemonStatus]
STATUS_MAP = {status.name: status for status in PokemonStatus}

POKEMON_STATUS_ONE_HOT_DESCRIPTION = ["Status Padding"] + [
    status.name.capitalize() for status in PokemonStatus
]

POKEMON_STATUS_ONE_HOT_PADDING = torch.tensor([1] + [0] * len(PokemonStatus))


@dataclass
class Pokemon:
    pokemon_id: int
    name: str
    types: set
    level: int
    base_hp: int
    base_attack: int
    base_defense: int
    base_sp_attack: int
    base_sp_defense: int
    base_speed: int

    ev_hp: int = 0
    ev_attack: int = 0
    ev_defense: int = 0
    ev_sp_attack: int = 0
    ev_sp_defense: int = 0
    ev_speed: int = 0
    iv_hp: int = 10
    iv_attack: int = 10
    iv_defense: int = 10
    iv_sp_attack: int = 10
    iv_sp_defense: int = 10
    iv_speed: int = 10
    moves: list[Move] = field(default_factory=list)

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()
        self.surname = self.name.capitalize()
        self.status = PokemonStatus.Healthy
        self._hp = self.max_hp
        self._confused = False
        self._confused_turns = 0
        self._taunted = False
        self._taunted_turns = 0
        self._modifiers = {
            stat: 0
            for stat in [
                "attack",
                "defense",
                "sp_attack",
                "sp_defense",
                "speed",
                "accuracy",
            ]
        }
        self._pp = [move.pp for move in self.moves]

    def validate_inputs(self):
        if not isinstance(self.pokemon_id, int) or self.pokemon_id < 0:
            raise PokemonError("Pokemon ID must be a positive integer")
        if not isinstance(self.name, str) or not self.name:
            raise PokemonError("Name must be a non-empty string")
        if not isinstance(self.types, list) or not all(
            isinstance(t, PokemonType) for t in self.types
        ):
            raise PokemonError("Types must be a set of PokemonType")
        if not isinstance(self.level, int) or not 1 <= self.level <= 100:
            raise PokemonError("Level must be an integer between 1 and 100")
        if not isinstance(self.base_hp, int) or not 1 <= self.base_hp <= 255:
            raise PokemonError("Base HP must be an integer between 1 and 255")
        if not isinstance(self.base_attack, int) or not 1 <= self.base_attack <= 255:
            raise PokemonError("Base Attack must be an integer between 1 and 255")
        if not isinstance(self.base_defense, int) or not 1 <= self.base_defense <= 255:
            raise PokemonError("Base Defense must be an integer between 1 and 255")
        if (
            not isinstance(self.base_sp_attack, int)
            or not 1 <= self.base_sp_attack <= 255
        ):
            raise PokemonError(
                "Base Special Attack must be an integer between 1 and 255"
            )
        if (
            not isinstance(self.base_sp_defense, int)
            or not 1 <= self.base_sp_defense <= 255
        ):
            raise PokemonError(
                "Base Special Defense must be an integer between 1 and 255"
            )
        if not isinstance(self.base_speed, int) or not 1 <= self.base_speed <= 255:
            raise PokemonError("Base Speed must be an integer between 1 and 255")
        if not isinstance(self.moves, list) or not all(
            isinstance(m, Move) for m in self.moves
        ):
            raise PokemonError("Moves must be a list of Move")

    @cached_property
    def max_hp(self) -> int:
        return (
            int((2 * self.base_hp + self.iv_hp + self.ev_hp // 4) * self.level // 100)
            + self.level
            + 10
        )

    @cached_property
    def level_factor(self) -> float:
        return (2 * self.level) / 5 + 2

    @property
    def hp(self) -> int:
        return self._hp

    @hp.setter
    def hp(self, value: int):
        self._hp = max(0, min(value, self.max_hp))

    @property
    def hp_ratio(self) -> float:
        return self.hp / self.max_hp

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def is_like(self, other) -> bool:
        return self.pokemon_id == other.pokemon_id

    def calculate_stat(self, base: int, iv: int, ev: int, modifier: int) -> int:
        stat = int((2 * base + iv + ev // 4) * self.level // 100) + 5
        return int(stat * STAT_MODIFIER[modifier + 6])

    @cached_property
    def attack(self) -> int:
        return self.calculate_stat(
            self.base_attack, self.iv_attack, self.ev_attack, self._modifiers["attack"]
        )

    @cached_property
    def defense(self) -> int:
        return self.calculate_stat(
            self.base_defense,
            self.iv_defense,
            self.ev_defense,
            self._modifiers["defense"],
        )

    @cached_property
    def sp_attack(self) -> int:
        return self.calculate_stat(
            self.base_sp_attack,
            self.iv_sp_attack,
            self.ev_sp_attack,
            self._modifiers["sp_attack"],
        )

    @cached_property
    def sp_defense(self) -> int:
        return self.calculate_stat(
            self.base_sp_defense,
            self.iv_sp_defense,
            self.ev_sp_defense,
            self._modifiers["sp_defense"],
        )

    @cached_property
    def speed(self) -> int:
        return self.calculate_stat(
            self.base_speed, self.iv_speed, self.ev_speed, self._modifiers["speed"]
        )

    def reset_stat(self, stat: str):
        if DEBUG:
            if stat not in self._modifiers:
                raise PokemonError(f"Invalid stat: {stat}")
        self.__dict__.pop(stat, None)

    def apply_modifier(self, stat: str, modifier: int):
        if DEBUG:
            if stat not in self._modifiers:
                raise PokemonError(f"Invalid stat: {stat}")
        self._modifiers[stat] += modifier
        if self._modifiers[stat] > 6:
            self._modifiers[stat] = 6
        elif self._modifiers[stat] < -6:
            self._modifiers[stat] = -6
        self.reset_stat(stat)

    def reset_modifiers(self):
        for stat in self._modifiers:
            self._modifiers[stat] = 0
            self.reset_stat(stat)

    def clear(self):
        self.reset_modifiers()
        self._taunted = False
        self._taunted_turns = 0
        self._confused = False
        self._confused_turns = 0

    def reset(self):
        self.clear()
        self.hp = self.max_hp
        self._pp = [move.pp for move in self.moves]

    def can_attack(self) -> tuple[bool, list[Message]]:
        if self._confused:
            if self._confused_turns == 0:
                self._confused_turns = random.randint(2, 4)
            self._confused_turns -= 1
            if self._confused_turns == 0:
                self._confused = False
                return True, [Message(f"{self.surname} snapped out of confusion!")]
            if random.choice([True, False]):
                damage, critical, effectiveness = MoveAccessor.SelfHit.calculate_damage(
                    self, self
                )[0:3]
                self.hp -= damage
                return False, [
                    Message(
                        f"{self.surname} is confused! It hurt itself in its confusion!"
                    )
                ]
        return True, []

    def after_turn(self) -> list[Message]:
        messages = []
        if self.is_alive:
            if self.status == PokemonStatus.Burn:
                burn_damage = self.max_hp // 8
                self.hp -= burn_damage
                messages.extend(
                    [
                        Message(f"{self.surname} is hurt by its burn!"),
                        Message(f"{self.name} lost {burn_damage} HP."),
                    ]
                )
            if self._taunted:
                # Implement taunt logic
                pass
        return messages

    def __repr__(self) -> str:
        return f"Pokemon(id={self.pokemon_id}, name={self.name}, surname={self.surname}, level={self.level}, hp={self.hp}/{self.max_hp})"

    def __str__(self) -> str:
        return f"{self.surname} Lvl.{self.level} | HP: {self.hp}/{self.max_hp}"

    def __hash__(self) -> int:
        return hash(self.pokemon_id)

    @property
    def str_stats(self) -> str:
        return f"{self.surname} Lvl.{self.level} | HP: {self.hp}/{self.max_hp} | ATK: {self.attack} | DEF: {self.defense} | SP.ATK: {self.sp_attack} | SP.DEF: {self.sp_defense} | SPD: {self.speed}"

    @property
    def str_moves(self) -> str:
        text = f"{self.surname} has the following moves:"
        text += "".join(
            [
                f"\n[{i + 1}] {move.name} | Type: {move.type} | PP: {self._pp[i]}/{move.pp}"
                for i, move in enumerate(self.moves)
            ]
        )
        return text

    def get_pp(self, move_name: str) -> int:
        for i, move in enumerate(self.moves):
            if move.name == move_name:
                return self._pp[i]
        if DEBUG:
            raise PokemonError(f"Move {move_name} not found in moveset")

    def decrease_pp(self, move_name: str):
        for i, move in enumerate(self.moves):
            if move.name == move_name:
                if self._pp[i] > 0:
                    self._pp[i] -= 1
                else:
                    if DEBUG:
                        raise PokemonError(f"PP for move {move_name} is already 0")
                return
        if DEBUG:
            raise PokemonError(f"Move {move_name} not found in moveset")

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)

    @property
    def one_hot(self) -> torch.Tensor:
        return ONEHOTCACHE.get_one_hot(len(POKEMON_LIST) + 1, self.pokemon_id + 1)

    @cached_property
    def one_hot_description(self) -> torch.Tensor:
        return POKEMON_ONE_HOT_DESCRIPTION


@dataclass
class Pikachu(Pokemon):
    level: int = 5
    pokemon_id: int = 0
    name: str = "pikachu"
    types: set = field(default_factory=lambda: [PokemonTypeAccessor.Electric])
    base_hp: int = 35
    base_attack: int = 55
    base_defense: int = 40
    base_sp_attack: int = 50
    base_sp_defense: int = 50
    base_speed: int = 90
    moves: list[Move] = field(
        default_factory=lambda: [MoveAccessor.Scratch, MoveAccessor.ThunderShock]
    )

    __repr__ = Pokemon.__repr__
    __str__ = Pokemon.__str__
    __hash__ = Pokemon.__hash__


@dataclass
class Chimchar(Pokemon):
    level: int = 5
    pokemon_id: int = 1
    name: str = "chimchar"
    types: set = field(default_factory=lambda: [PokemonTypeAccessor.Fire])
    base_hp: int = 44
    base_attack: int = 58
    base_defense: int = 44
    base_sp_attack: int = 58
    base_sp_defense: int = 44
    base_speed: int = 61
    moves: list[Move] = field(default_factory=lambda: [MoveAccessor.Scratch])

    __repr__ = Pokemon.__repr__
    __str__ = Pokemon.__str__
    __hash__ = Pokemon.__hash__


@dataclass
class Piplup(Pokemon):
    level: int = 5
    pokemon_id: int = 2
    name: str = "piplup"
    types: set = field(default_factory=lambda: [PokemonTypeAccessor.Water])
    base_hp: int = 53
    base_attack: int = 51
    base_defense: int = 53
    base_sp_attack: int = 61
    base_sp_defense: int = 56
    base_speed: int = 40
    moves: list[Move] = field(default_factory=lambda: [MoveAccessor.Scratch])

    __repr__ = Pokemon.__repr__
    __str__ = Pokemon.__str__
    __hash__ = Pokemon.__hash__


POKEMON_LIST = [Pikachu, Chimchar, Piplup]

POKEMON_MAP = {pokemon.name: pokemon for pokemon in POKEMON_LIST}

assert len({pokemon.pokemon_id for pokemon in POKEMON_LIST}) == len(POKEMON_LIST), (
    "Duplicate Pokemon IDs"
)

assert all(pokemon.pokemon_id == index for index, pokemon in enumerate(POKEMON_LIST)), (
    "Invalid Move IDs"
)


class _PokemonAccessor:
    def __getattr__(self, attr: str) -> Pokemon:
        t = POKEMON_MAP.get(attr.lower())
        if t:
            return t
        raise AttributeError(f"No Pokemon named '{attr}'")

    def __iter__(self):
        return iter(POKEMON_LIST)

    def __len__(self):
        return len(POKEMON_LIST)

    def __repr__(self):
        return "Pokemons(" + ", ".join(t.name.capitalize() for t in POKEMON_LIST) + ")"

    def by_id(self, pokemon_id: int) -> Pokemon:
        return POKEMON_LIST[pokemon_id]

    def get(self, name: str) -> Pokemon | None:
        return POKEMON_MAP.get(name)

    @property
    def names(self) -> list[str]:
        return list(POKEMON_MAP.keys())


PokemonAccessor = _PokemonAccessor()


POKEMON_ONE_HOT_PADDING = torch.tensor([1] + [0] * len(POKEMON_LIST))

POKEMON_ONE_HOT_DESCRIPTION = ["Pokemon Padding"] + [
    pokemon.name.capitalize() for pokemon in PokemonAccessor
]
