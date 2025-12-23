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

Turtwig = SpeciesRegistry.register(
    "Turtwig",
    (PokemonType.GRASS,),
    55, 68, 64, 45, 55, 31,
    ("Tackle", "Withdraw", "Absorb", "Razor Leaf"),
)

Chimchar = SpeciesRegistry.register(
    "Chimchar",
    (PokemonType.FIRE,),
    44, 58, 44, 58, 44, 61,
    ("Scratch", "Leer", "Ember", "Taunt"),
)

Piplup = SpeciesRegistry.register(
    "Piplup",
    (PokemonType.WATER,),
    53, 51, 53, 61, 56, 40,
    ("Pound", "Growl", "Bubble", "Water Gun"),
)

Starly = SpeciesRegistry.register(
    "Starly",
    (PokemonType.NORMAL, PokemonType.FLYING),
    40, 55, 30, 30, 30, 60,
    ("Tackle", "Growl", "Quick Attack", "Wing Attack"),
)

Shinx = SpeciesRegistry.register(
    "Shinx",
    (PokemonType.ELECTRIC,),
    45, 65, 34, 40, 34, 45,
    ("Tackle", "Leer", "Thunder Shock", "Spark"),
)

Garchomp = SpeciesRegistry.register(
    "Garchomp",
    (PokemonType.DRAGON, PokemonType.GROUND),
    108, 130, 95, 80, 85, 102,
    ("Dragon Rush", "Crunch", "Take Down", "Sand Attack"),
)

Lucario = SpeciesRegistry.register(
    "Lucario",
    (PokemonType.FIGHTING, PokemonType.STEEL),
    70, 110, 70, 115, 70, 90,
    ("Aura Sphere", "Mach Punch", "Extreme Speed", "Dragon Pulse"),
)

Torterra = SpeciesRegistry.register(
    "Torterra",
    (PokemonType.GRASS, PokemonType.GROUND),
    95, 109, 105, 75, 85, 56,
    ("Earthquake", "Wood Hammer", "Crunch", "Solar Beam"),
)

Infernape = SpeciesRegistry.register(
    "Infernape",
    (PokemonType.FIRE, PokemonType.FIGHTING),
    76, 104, 71, 104, 71, 108,
    ("Flare Blitz", "Low Kick", "U-turn", "Mach Punch"),
)

Empoleon = SpeciesRegistry.register(
    "Empoleon",
    (PokemonType.WATER, PokemonType.STEEL),
    84, 86, 88, 111, 101, 60,
    ("Hydro Pump", "Flash Cannon", "Ice Beam", "Aqua Jet"),
)

Staraptor = SpeciesRegistry.register(
    "Staraptor",
    (PokemonType.NORMAL, PokemonType.FLYING),
    85, 120, 70, 50, 60, 100,
    ("Brave Bird", "Low Kick", "U-turn", "Quick Attack"),
)

Luxray = SpeciesRegistry.register(
    "Luxray",
    (PokemonType.ELECTRIC,),
    80, 120, 79, 95, 79, 70,
    ("Wild Charge", "Bite", "Body Slam", "Thunderbolt"),
)

Weavile = SpeciesRegistry.register(
    "Weavile",
    (PokemonType.DARK, PokemonType.ICE),
    70, 120, 65, 45, 85, 125,
    ("Ice Shard", "Slash", "Bite", "Low Kick"),
)

Magnezone = SpeciesRegistry.register(
    "Magnezone",
    (PokemonType.ELECTRIC, PokemonType.STEEL),
    70, 70, 115, 130, 90, 60,
    ("Thunderbolt", "Flash Cannon", "Thunder Shock", "Tri Attack"),
)

Rhyperior = SpeciesRegistry.register(
    "Rhyperior",
    (PokemonType.ROCK, PokemonType.GROUND),
    115, 140, 130, 55, 55, 40,
    ("Earthquake", "Rock Slide", "Megahorn", "Stone Edge"),
)

Togekiss = SpeciesRegistry.register(
    "Togekiss",
    (PokemonType.NORMAL, PokemonType.FLYING), # Fairy type did not exist in Gen 4
    85, 50, 95, 120, 115, 80,
    ("Air Slash", "Aura Sphere", "Agility", "Nasty Plot"),
)

Dialga = SpeciesRegistry.register(
    "Dialga",
    (PokemonType.STEEL, PokemonType.DRAGON),
    100, 120, 120, 150, 100, 90,
    ("Draco Meteor", "Flash Cannon", "Fire Blast", "Rock Slide"),
)

Palkia = SpeciesRegistry.register(
    "Palkia",
    (PokemonType.WATER, PokemonType.DRAGON),
    90, 120, 100, 150, 120, 100,
    ("Spacial Rend", "Hydro Pump", "Fire Blast", "Thunder"),
)

Giratina = SpeciesRegistry.register(
    "Giratina",
    (PokemonType.GHOST, PokemonType.DRAGON),
    150, 100, 120, 100, 120, 90, # Altered Forme
    ("Shadow Claw", "Draco Meteor", "Aura Sphere", "Will-O-Wisp"),
)

Arceus = SpeciesRegistry.register(
    "Arceus",
    (PokemonType.NORMAL,),
    120, 120, 120, 120, 120, 120,
    ("Judgment", "Extreme Speed", "Swords Dance", "Shadow Claw"),
)

Gible = SpeciesRegistry.register(
    "Gible",
    (PokemonType.DRAGON, PokemonType.GROUND),
    58, 70, 45, 40, 45, 42,
    ("Tackle", "Sand Attack", "Dragon Rage", "Take Down"),
)

Riolu = SpeciesRegistry.register(
    "Riolu",
    (PokemonType.FIGHTING,),
    40, 70, 40, 35, 40, 60,
    ("Quick Attack", "Mach Punch"),
)

Hippopotas = SpeciesRegistry.register(
    "Hippopotas",
    (PokemonType.GROUND,),
    68, 72, 78, 38, 42, 32,
    ("Tackle", "Sand Attack", "Bite", "Body Slam"),
)

Croagunk = SpeciesRegistry.register(
    "Croagunk",
    (PokemonType.POISON, PokemonType.FIGHTING),
    48, 61, 40, 61, 40, 50,
    ("Poison Sting", "Mud-Slap", "Taunt"),
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
