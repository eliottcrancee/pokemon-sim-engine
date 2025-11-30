from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, Optional

from pokemon.message import Message
from pokemon.move import Move, MoveRegistry
from pokemon.pokemon_status import PokemonStatus
from pokemon.pokemon_type import PokemonType

_STAT_STAGE_MULTIPLIERS: Final[tuple[float, ...]] = (
    0.25,  # -6 to -1
    0.29,
    0.33,
    0.40,
    0.50,
    0.67,
    1.0,  # 0
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,  # +1 to +6
)


class PokemonError(Exception):
    pass


@dataclass(slots=True)
class MoveSlot:
    """
    Represents a learned move and its current PP.
    """

    move: Move
    current_pp: int
    max_pp: int

    @classmethod
    def from_move(cls, move: Move) -> MoveSlot:
        return cls(move, move.pp, move.pp)

    def reset(self):
        self.current_pp = self.max_pp

    def copy(self) -> MoveSlot:
        return MoveSlot(self.move, self.current_pp, self.max_pp)

    def __eq__(self, other):
        if not isinstance(other, MoveSlot):
            return NotImplemented
        return self.move.id == other.move.id and self.current_pp == other.current_pp

    def __hash__(self):
        return hash((self.move.id, self.current_pp))


@dataclass(frozen=True, slots=True)
class PokemonSpecies:
    """
    Immutable definition of a Pokemon Species.
    Optimized: 'base_stats' is a tuple for O(1) indexed access matching IVS/EVS.
    Order: HP, Atk, Def, SpA, SpD, Spe.
    """

    id: int
    name: str
    types: tuple[PokemonType, ...]
    base_stats: tuple[int, int, int, int, int, int]
    default_moves: tuple[str, ...]


class Pokemon:
    """
    Highly optimized Pokemon instance.
    Uses array.array for integer stats to ensure C-like contiguous memory.
    Implements pre-calculation for stats to ensure O(1) access during battle.
    """

    __slots__ = (
        "species",
        "surname",
        "level",
        "_ivs",
        "_evs",
        "move_slots",
        "_current_hp",
        "_max_hp",
        "_status",
        "_stat_stages",
        "_raw_stats",
        "_effective_stats",
        "_level_factor",
        "is_alive",
        "_confused",
        "_confused_turns",
        "_taunted",
        "_taunted_turns",
        "_accuracy",
        "_accuracy_stage",
        "_sleep_turns",
    )

    def __init__(
        self,
        species: PokemonSpecies,
        level: int = 5,
        moves: list[Move] | None = None,
        ivs: tuple[int, ...] = (10, 10, 10, 10, 10, 10),
        evs: tuple[int, ...] = (0, 0, 0, 0, 0, 0),
        surname: str | None = None,
    ):
        self.species = species
        self.level = max(1, min(100, level))
        self.surname = surname or species.name

        self._ivs = ivs
        self._evs = evs

        # Pre-calculate Level Factor for damage formula (Math optimization)
        self._level_factor = (2 * self.level) / 5 + 2

        # Moves Initialization
        if moves:
            self.move_slots = [MoveSlot.from_move(m) for m in moves]
        else:
            self.move_slots = []
            for m_name in species.default_moves:
                m = MoveRegistry.get(m_name)
                if m:
                    self.move_slots.append(MoveSlot.from_move(m))

        # Battle State
        self._status = PokemonStatus.HEALTHY
        self.is_alive = True
        self._confused = False
        self._confused_turns = 0
        self._taunted = False
        self._taunted_turns = 0
        self._accuracy = 1
        self._accuracy_stage = 0
        self._sleep_turns = 0

        # Stats Initialization
        # 0:Atk, 1:Def, 2:SpA, 3:SpD, 4:Spe (HP is handled separately)
        self._stat_stages = [0, 0, 0, 0, 0]
        self._raw_stats = [0, 0, 0, 0, 0]
        self._effective_stats = [0, 0, 0, 0, 0]

        self._calculate_max_hp()
        self._current_hp = self._max_hp

        # Calculate initial stats
        self._recalculate_all_stats()

    def _calculate_max_hp(self):
        """Calculates HP based on Gen 3+ formula."""
        base = self.species.base_stats[0]
        iv = self._ivs[0]
        ev = self._evs[0]
        self._max_hp = (
            ((2 * base + iv + (ev // 4)) * self.level) // 100 + self.level + 10
        )

    def _recalculate_all_stats(self):
        """
        Calculates Raw stats (Nature/IV/EV/Level) and then updates Effective stats.
        Call this on Level Up or Spawn.
        """
        # Stats indices: 0=HP(skip), 1=Atk, 2=Def, 3=SpA, 4=SpD, 5=Spe
        # Map to internal arrays: 0=Atk, 1=Def, 2=SpA, 3=SpD, 4=Spe

        # Species stats tuple: (HP, Atk, Def, SpA, SpD, Spe)
        s_stats = self.species.base_stats

        for i in range(5):
            # i corresponds to internal arrays.
            # Species/IV/EV index is i + 1 (skipping HP at index 0)
            base = s_stats[i + 1]
            iv = self._ivs[i + 1]
            ev = self._evs[i + 1]

            # Gen 3+ Formula
            raw_val = ((2 * base + iv + (ev // 4)) * self.level) // 100 + 5
            self._raw_stats[i] = raw_val

        # Update effective stats based on stages
        self._update_effective_stats()

    def _update_effective_stats(self):
        """
        Updates cached battle stats based on current Raw Stats and Stages.
        This provides O(1) read access during battle logic.
        """
        for i in range(5):
            stage = self._stat_stages[i]
            # Offset index 6 maps to multiplier 1.0
            # Clamp stage between -6 and +6, then shift by +6 for array index
            idx = max(0, min(12, stage + 6))
            mult = _STAT_STAGE_MULTIPLIERS[idx]
            # Fast int conversion
            self._effective_stats[i] = int(self._raw_stats[i] * mult)

    # --- Properties for O(1) Access ---
    @property
    def id(self) -> int:
        return self.species.id

    @property
    def name(self) -> str:
        return self.species.name

    @property
    def types(self) -> tuple[PokemonType, ...]:
        return self.species.types

    @property
    def max_hp(self) -> int:
        return self._max_hp

    @property
    def hp(self) -> int:
        return self._current_hp

    @hp.setter
    def hp(self, value: int):
        # Branchless clamping
        self._current_hp = value if value < self._max_hp else self._max_hp
        if self._current_hp <= 0:
            self._current_hp = 0
            self.is_alive = False
            self._status = PokemonStatus.FAINTED
        else:
            self.is_alive = True

    @property
    def hp_ratio(self) -> float:
        return self._current_hp / self._max_hp if self._max_hp else 0.0

    @property
    def level_factor(self) -> float:
        return self._level_factor

    @property
    def status(self) -> PokemonStatus:
        return self._status

    @status.setter
    def status(self, value: PokemonStatus):
        self._status = value
        if value != PokemonStatus.SLEEP:
            self._sleep_turns = 0

    @property
    def taunted(self) -> bool:
        return self._taunted

    @taunted.setter
    def taunted(self, value: bool):
        self._taunted = value
        if not value:
            self._taunted_turns = 0

    @property
    def confused(self) -> bool:
        return self._confused

    @confused.setter
    def confused(self, value: bool):
        self._confused = value
        if not value:
            self._confused_turns = 0

    @property
    def modifiers(self) -> dict[int, int, int, int, int]:
        return (
            self._stat_stages[0],
            self._stat_stages[1],
            self._stat_stages[2],
            self._stat_stages[3],
            self._stat_stages[4],
        )

    @property
    def attack(self) -> int:
        return self._effective_stats[0]

    @property
    def defense(self) -> int:
        return self._effective_stats[1]

    @property
    def sp_attack(self) -> int:
        return self._effective_stats[2]

    @property
    def sp_defense(self) -> int:
        return self._effective_stats[3]

    @property
    def speed(self) -> int:
        return self._effective_stats[4]

    @property
    def attack_stage(self) -> int:
        return self._stat_stages[0]

    @property
    def defense_stage(self) -> int:
        return self._stat_stages[1]

    @property
    def sp_attack_stage(self) -> int:
        return self._stat_stages[2]

    @property
    def sp_defense_stage(self) -> int:
        return self._stat_stages[3]

    @property
    def speed_stage(self) -> int:
        return self._stat_stages[4]

    def reset(self):
        self.full_restore()
        for slot in self.move_slots:
            slot.reset()

    def full_restore(self):
        self.restore()
        self._current_hp = self.max_hp
        self.is_alive = True

    def restore(self):
        self._status = PokemonStatus.HEALTHY
        self._confused = False
        self._confused_turns = 0
        self._taunted = False
        self._taunted_turns = 0
        self._sleep_turns = 0
        self.reset_modifiers()

    def reset_modifiers(self):
        for i in range(5):
            self._stat_stages[i] = 0
            self._effective_stats[i] = self._raw_stats[i]
        self._accuracy_stage = 0

    def apply_modifier(self, stat_index: int, stages: int) -> bool:
        """
        stat_index: 0=Atk, 1=Def, 2=SpA, 3=SpD, 4=Spe
        Returns True if changed.
        """
        if not (0 <= stat_index <= 4):
            return False

        current = self._stat_stages[stat_index]
        if (stages > 0 and current == 6) or (stages < 0 and current == -6):
            return False

        new_stage = max(-6, min(6, current + stages))
        self._stat_stages[stat_index] = new_stage

        idx = new_stage + 6
        self._effective_stats[stat_index] = int(
            self._raw_stats[stat_index] * _STAT_STAGE_MULTIPLIERS[idx]
        )
        return True

    def apply_accuracy_modifier(self, stages: int) -> bool:
        """
        Applies accuracy modifier.
        Returns True if changed.
        """
        current = self._accuracy_stage
        if (stages > 0 and current == 6) or (stages < 0 and current == -6):
            return False

        new_stage = max(-6, min(6, current + stages))
        self._accuracy_stage = new_stage
        return True

    def decrease_pp(self, move_index: int):
        slot = self.move_slots[move_index]
        if slot.current_pp > 0:
            slot.current_pp -= 1

    def copy(self) -> Pokemon:
        """
        Fastest possible deep copy using __new__ to bypass __init__.
        """
        new_p = Pokemon.__new__(Pokemon)

        # Reference copy for immutable objects
        new_p.species = self.species
        new_p.surname = self.surname
        new_p.level = self.level
        new_p._level_factor = self._level_factor

        # Fast Array copying (memory block copy)
        new_p._ivs = self._ivs
        new_p._evs = self._evs
        new_p._stat_stages = self._stat_stages.copy()
        new_p._raw_stats = self._raw_stats.copy()
        new_p._effective_stats = self._effective_stats.copy()

        # Primitive state
        new_p._max_hp = self._max_hp
        new_p._current_hp = self._current_hp
        new_p.is_alive = self.is_alive
        new_p._status = self._status
        new_p._confused = self._confused
        new_p._confused_turns = self._confused_turns
        new_p._taunted = self._taunted
        new_p._taunted_turns = self._taunted_turns
        new_p._accuracy = self._accuracy
        new_p._accuracy_stage = self._accuracy_stage
        new_p._sleep_turns = self._sleep_turns

        # List comp for move slots
        new_p.move_slots = [ms.copy() for ms in self.move_slots]

        return new_p

    def can_attack(self) -> tuple[bool, list[Message]]:
        msgs = []

        # Optimistic check: if healthy and no volatile status, return immediately
        if self._status == PokemonStatus.HEALTHY and not self._confused:
            return True, msgs

        # STATUS CHECKS
        if self._status == PokemonStatus.FREEZE:
            if random.random() < 0.2:
                self._status = PokemonStatus.HEALTHY
                msgs.append(Message(f"{self.surname} thawed out!"))
            else:
                return False, [Message(f"{self.surname} is frozen solid!")]

        elif self._status == PokemonStatus.SLEEP:
            if self._sleep_turns <= 0:
                self._status = PokemonStatus.HEALTHY
                msgs.append(Message(f"{self.surname} woke up!"))
            else:
                msgs.append(Message(f"{self.surname} is fast asleep."))
                self._sleep_turns -= 1
                return False, msgs

        elif self._status == PokemonStatus.PARALYSIS:
            if random.random() < 0.25:
                return False, [Message(f"{self.surname} is paralyzed! It can't move!")]

        # VOLATILE CHECKS
        if self._confused:
            msgs.append(Message(f"{self.surname} is confused!"))
            self._confused_turns -= 1
            if self._confused_turns <= 0:
                self._confused = False
                msgs.append(Message(f"{self.surname} snapped out of confusion!"))
            elif random.random() < 0.50:
                atk = self.attack
                defn = self.defense
                dmg = int(((self._level_factor * 40 * atk // defn) // 50) + 2)
                self.hp -= dmg
                msgs.append(Message("It hurt itself in its confusion!"))
                return False, msgs

        return True, msgs

    def after_turn(self) -> list[Message]:
        msgs = []
        if not self.is_alive:
            return msgs

        # Use bitmask-like logic or direct checks
        status = self._status

        if status == PokemonStatus.BURN:
            dmg = max(
                1, self._max_hp >> 3
            )  # Bitwise shift for division by 8 (Gen 1-6 style 1/8)
            self.hp -= dmg
            msgs.append(Message(f"{self.surname} is hurt by its burn!"))

        elif status == PokemonStatus.POISON:
            dmg = max(1, self._max_hp >> 3)
            self.hp -= dmg
            msgs.append(Message(f"{self.surname} is hurt by poison!"))

        if self._taunted:
            self._taunted_turns -= 1
            if self._taunted_turns <= 0:
                self._taunted = False
                msgs.append(Message(f"{self.surname}'s taunt wore off!"))

        return msgs

    def __repr__(self) -> str:
        return f"<{self.surname} ({self.species.name}) L{self.level} HP:{self.hp}/{self.max_hp}>"

    def __eq__(self, other):
        if not isinstance(other, Pokemon):
            return NotImplemented
        return (
            self.species.id == other.species.id
            and self.level == other.level
            and self._current_hp == other._current_hp
            and self._status == other._status
            and self._stat_stages == other._stat_stages
            and self._confused == other._confused
            and self._confused_turns == other._confused_turns
            and self._taunted == other._taunted
            and self._taunted_turns == other._taunted_turns
            and self._accuracy_stage == other._accuracy_stage
            and self.move_slots == other.move_slots
        )

    def __hash__(self):
        return hash(
            (
                self.species.id,
                self.level,
                self._current_hp,
                self._status,
                tuple(self._stat_stages),
                self._confused,
                self._confused_turns,
                self._taunted,
                self._taunted_turns,
                self._accuracy_stage,
                tuple(self.move_slots),
            )
        )


# --- Registry System ---


class SpeciesRegistry:
    _species: list[PokemonSpecies] = []
    _map: dict[str, PokemonSpecies] = {}

    @classmethod
    def register(
        cls,
        name: str,
        types: tuple[PokemonType, ...],
        hp: int,
        atk: int,
        Def: int,
        spa: int,
        spd: int,
        spe: int,
        moves: tuple[str, ...],
    ) -> PokemonSpecies:
        s_id = len(cls._species)
        stats = (hp, atk, Def, spa, spd, spe)
        species = PokemonSpecies(s_id, name, types, stats, moves)
        cls._species.append(species)
        cls._map[name.lower()] = species
        return species

    @classmethod
    def get(cls, name_or_id: str | int) -> Optional[PokemonSpecies]:
        if isinstance(name_or_id, int):
            if 0 <= name_or_id < len(cls._species):
                return cls._species[name_or_id]
        else:
            return cls._map.get(str(name_or_id).lower())
        return None


# --- Registration ---

Pikachu = SpeciesRegistry.register(
    "Pikachu",
    (PokemonType.ELECTRIC,),
    35,
    55,
    40,
    50,
    50,
    90,
    ("Thunder Shock", "Growl", "Quick Attack", "Tail Whip"),
)

Chimchar = SpeciesRegistry.register(
    "Chimchar", (PokemonType.FIRE,), 44, 58, 44, 58, 44, 61, ("Scratch", "Ember")
)

Piplup = SpeciesRegistry.register(
    "Piplup",
    (PokemonType.WATER,),
    53,
    51,
    53,
    61,
    56,
    40,
    ("Tackle", "Water Gun", "Growl"),
)

Bulbasaur = SpeciesRegistry.register(
    "Bulbasaur",
    (PokemonType.GRASS, PokemonType.POISON),  # Corrected types
    45,
    49,
    49,
    65,
    65,
    45,
    ("Tackle", "Growl", "Vine Whip"),
)

Charmander = SpeciesRegistry.register(
    "Charmander",
    (PokemonType.FIRE,),
    39,
    52,
    43,
    60,
    50,
    65,
    ("Scratch", "Growl", "Ember"),
)

Squirtle = SpeciesRegistry.register(
    "Squirtle",
    (PokemonType.WATER,),
    44,
    48,
    65,
    50,
    64,
    43,
    ("Tackle", "Tail Whip", "Water Gun"),
)

Pidgey = SpeciesRegistry.register(
    "Pidgey",
    (PokemonType.NORMAL, PokemonType.FLYING),
    40,
    45,
    40,
    35,
    35,
    56,
    ("Tackle", "Sand Attack", "Quick Attack"),
)

Rattata = SpeciesRegistry.register(
    "Rattata",
    (PokemonType.NORMAL,),
    30,
    56,
    35,
    25,
    35,
    72,
    ("Tackle", "Tail Whip", "Quick Attack"),
)

Sandshrew = SpeciesRegistry.register(
    "Sandshrew",
    (PokemonType.GROUND,),
    50,
    75,
    85,
    20,
    30,
    40,
    ("Scratch", "Sand Attack"),
)

Eevee = SpeciesRegistry.register(
    "Eevee",
    (PokemonType.NORMAL,),
    55,
    55,
    50,
    45,
    65,
    55,
    ("Tackle", "Tail Whip", "Sand Attack"),
)

Charizard = SpeciesRegistry.register(
    "Charizard",
    (PokemonType.FIRE, PokemonType.FLYING),
    78,
    84,
    78,
    109,
    85,
    100,
    ("Flamethrower", "Wing Attack", "Slash", "Growl"),
)
Blastoise = SpeciesRegistry.register(
    "Blastoise",
    (PokemonType.WATER,),
    79,
    83,
    100,
    85,
    105,
    78,
    ("Hydro Pump", "Skull Bash", "Bite", "Tackle"),
)
Venosaur = SpeciesRegistry.register(
    "Venosaur",
    (PokemonType.GRASS, PokemonType.POISON),
    80,
    82,
    83,
    100,
    100,
    80,
    ("Solar Beam", "Razor Leaf", "Tackle", "Growl"),
)
Snorlax = SpeciesRegistry.register(
    "Snorlax",
    (PokemonType.NORMAL,),
    160,
    110,
    65,
    65,
    110,
    30,
    ("Body Slam", "Hyper Beam", "Pound", "Tackle"),
)
Lapras = SpeciesRegistry.register(
    "Lapras",
    (PokemonType.WATER, PokemonType.ICE),
    130,
    85,
    80,
    85,
    95,
    60,
    ("Blizzard", "Body Slam", "Hydro Pump", "Growl"),
)
Pidgeot = SpeciesRegistry.register(
    "Pidgeot",
    (PokemonType.NORMAL, PokemonType.FLYING),
    83,
    80,
    75,
    70,
    70,
    101,
    ("Wing Attack", "Sky Attack", "Tackle", "Sand Attack"),
)
Alakazam = SpeciesRegistry.register(
    "Alakazam",
    (PokemonType.PSYCHIC,),
    55,
    50,
    45,
    135,
    95,
    120,
    ("Psychic", "Tackle", "Pound", "Growl"),
)
Rhydon = SpeciesRegistry.register(
    "Rhydon",
    (PokemonType.GROUND, PokemonType.ROCK),
    105,
    130,
    120,
    45,
    45,
    40,
    ("Earthquake", "Rock Slide", "Pound", "Tail Whip"),
)
Gyarados = SpeciesRegistry.register(
    "Gyarados",
    (PokemonType.WATER, PokemonType.FLYING),
    95,
    125,
    79,
    60,
    100,
    81,
    ("Hydro Pump", "Hyper Beam", "Dragon Rage", "Tackle"),
)
Arcanine = SpeciesRegistry.register(
    "Arcanine",
    (PokemonType.FIRE,),
    90,
    110,
    80,
    100,
    80,
    95,
    ("Flamethrower", "Extreme Speed", "Bite", "Growl"),
)
Exeggutor = SpeciesRegistry.register(
    "Exeggutor",
    (PokemonType.GRASS, PokemonType.PSYCHIC),
    95,
    95,
    85,
    125,
    65,
    55,
    ("Psychic", "Solar Beam", "Tackle", "Growl"),
)

# --- Accessor Helper ---


class _PokedexMeta(type):
    def __getattr__(cls, name: str) -> PokemonSpecies:
        s = SpeciesRegistry.get(name)
        if s:
            return s
        raise AttributeError(f"Pokemon {name} not found")


class Pokedex(metaclass=_PokedexMeta):
    pass
