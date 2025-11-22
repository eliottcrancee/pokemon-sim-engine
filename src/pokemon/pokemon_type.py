# pokemon_type.py

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from pympler import asizeof

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from pokemon.config import DEBUG


class PokemonTypeError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass()
class PokemonType:
    type_id: int
    name: str
    weaknesses: frozenset[str] = field(default_factory=frozenset)
    resistances: frozenset[str] = field(default_factory=frozenset)
    immunities: frozenset[str] = field(default_factory=frozenset)
    _relations_resolved: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if DEBUG:
            self.validate_inputs()

    def validate_inputs(self) -> None:
        if not isinstance(self.name, str):
            raise PokemonTypeError(
                f"Name must be a string, not {type(self.name).__name__}"
            )
        if not isinstance(self.type_id, int):
            raise PokemonTypeError(
                f"PokemonType ID must be an integer, not {type(self.type_id).__name__}"
            )
        for attr in ["weaknesses", "resistances", "immunities"]:
            if not isinstance(getattr(self, attr), frozenset):
                raise PokemonTypeError(
                    f"{attr.capitalize()} must be a frozenset, not {type(getattr(self, attr)).__name__}"
                )

    def effectiveness_against(self, target_types: list["PokemonType"]) -> float:
        effectiveness = 1.0
        for target_type in target_types:
            if self in target_type.immunities:
                return 0.0
            elif self in target_type.weaknesses:
                effectiveness *= 2.0
            elif self in target_type.resistances:
                effectiveness *= 0.5
        return effectiveness

    def __repr__(self) -> str:
        return f"PokemonType(id={self.type_id}, name={self.name})"

    def __str__(self) -> str:
        return self.name.capitalize()

    def __eq__(self, other: "PokemonType") -> bool:
        return self.type_id == other.type_id

    def __hash__(self) -> int:
        return hash((self.type_id, self.name))

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)


# fmt: off
TypeNone = PokemonType(0, "TypeNone")
Normal  = PokemonType(1, "Normal",
    weaknesses=frozenset({"Fighting"}),
    resistances=frozenset(),
    immunities=frozenset({"Ghost"}))
Fire    = PokemonType(2, "Fire",
    weaknesses=frozenset({"Water","Ground","Rock"}),
    resistances=frozenset({"Grass","Fire","Steel","Ice"}),
    immunities=frozenset())
Water   = PokemonType(3, "Water",
    weaknesses=frozenset({"Grass","Electric"}),
    resistances=frozenset({"Fire","Water","Ice","Steel"}),
    immunities=frozenset())
Grass   = PokemonType(4, "Grass",
    weaknesses=frozenset({"Fire","Ice","Poison","Flying","Bug"}),
    resistances=frozenset({"Water","Grass","Electric","Ground"}),
    immunities=frozenset())
Electric= PokemonType(5, "Electric",
    weaknesses=frozenset({"Ground"}),
    resistances=frozenset({"Electric","Flying","Steel"}),
    immunities=frozenset())
Ice     = PokemonType(6, "Ice",
    weaknesses=frozenset({"Fire","Fighting","Rock","Steel"}),
    resistances=frozenset({"Ice"}),
    immunities=frozenset())
Fighting= PokemonType(7, "Fighting",
    weaknesses=frozenset({"Flying","Psychic"}),
    resistances=frozenset({"Bug","Rock","Dark"}),
    immunities=frozenset())
Poison  = PokemonType(8, "Poison",
    weaknesses=frozenset({"Ground","Psychic"}),
    resistances=frozenset({"Grass","Fighting","Poison","Bug"}),
    immunities=frozenset())
Ground  = PokemonType(9, "Ground",
    weaknesses=frozenset({"Water","Grass","Ice"}),
    resistances=frozenset({"Poison","Rock"}),
    immunities=frozenset({"Electric"}))
Flying  = PokemonType(10,"Flying",
    weaknesses=frozenset({"Electric","Ice","Rock"}),
    resistances=frozenset({"Grass","Fighting","Bug"}),
    immunities=frozenset({"Ground"}))
Psychic = PokemonType(11,"Psychic",
    weaknesses=frozenset({"Bug","Ghost","Dark"}),
    resistances=frozenset({"Fighting","Psychic"}),
    immunities=frozenset())
Bug     = PokemonType(12,"Bug",
    weaknesses=frozenset({"Fire","Flying","Rock"}),
    resistances=frozenset({"Grass","Fighting","Ground"}),
    immunities=frozenset({"Psychic"}))
Rock    = PokemonType(13,"Rock",
    weaknesses=frozenset({"Water","Grass","Fighting","Ground","Steel"}),
    resistances=frozenset({"Normal","Fire","Poison","Flying"}),
    immunities=frozenset())
Ghost   = PokemonType(14,"Ghost",
    weaknesses=frozenset({"Ghost","Dark"}),
    resistances=frozenset({"Bug","Poison"}),
    immunities=frozenset({"Normal","Fighting"}))
Dragon  = PokemonType(15,"Dragon",
    weaknesses=frozenset({"Ice","Dragon"}),
    resistances=frozenset({"Fire","Water","Electric","Grass"}),
    immunities=frozenset())
Dark    = PokemonType(16,"Dark",
    weaknesses=frozenset({"Fighting","Bug"}),
    resistances=frozenset({"Ghost","Dark"}),
    immunities=frozenset())
Steel   = PokemonType(17,"Steel",
    weaknesses=frozenset({"Fire","Fighting","Ground"}),
    resistances=frozenset({"Normal","Grass","Ice","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel"}),
    immunities=frozenset({"Poison"}))
# fmt: on

_TYPE_LIST: list[PokemonType] = sorted(
    (obj for obj in globals().values() if isinstance(obj, PokemonType)),
    key=lambda t: t.type_id,
)

assert len({t.type_id for t in _TYPE_LIST}) == len(_TYPE_LIST), (
    "Duplicate PokemonType IDs"
)
assert all(t.type_id == i for i, t in enumerate(_TYPE_LIST)), "Invalid PokemonType IDs"

_TYPE_BY_NAME: dict[str, PokemonType] = {t.name: t for t in _TYPE_LIST}
_TYPE_BY_ID: dict[int, PokemonType] = {t.type_id: t for t in _TYPE_LIST}


class _PokemonTypeAccessor:
    def __getattr__(self, attr: str) -> PokemonType:
        t = _TYPE_BY_NAME.get(attr) or _TYPE_BY_NAME.get(attr.capitalize())
        if t:
            return t
        raise AttributeError(f"No PokemonType named '{attr}'")

    def __iter__(self):
        return iter(_TYPE_LIST)

    def __len__(self):
        return len(_TYPE_LIST)

    def __repr__(self):
        return "PokemonTypes(" + ", ".join(t.name for t in _TYPE_LIST) + ")"

    def by_id(self, type_id: int) -> PokemonType:
        return _TYPE_BY_ID[type_id]

    def get(self, name: str) -> PokemonType | None:
        return _TYPE_BY_NAME.get(name)

    @property
    def names(self) -> list[str]:
        return list(_TYPE_BY_NAME.keys())


PokemonTypeAccessor = _PokemonTypeAccessor()


def resolve_relations():
    name_map = {t.name: t for t in PokemonTypeAccessor}
    for t in PokemonTypeAccessor:
        for attr in ["weaknesses", "resistances", "immunities"]:
            lst = getattr(t, attr)
            if lst:
                setattr(t, attr, [name_map[n] for n in lst])
    for t in PokemonTypeAccessor:
        for attr in ["weaknesses", "resistances", "immunities"]:
            if not all(isinstance(i, PokemonType) for i in getattr(t, attr)):
                raise ValueError(f"Invalid {attr} for {t.name}")


resolve_relations()
