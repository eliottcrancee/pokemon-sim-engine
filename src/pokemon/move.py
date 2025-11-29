from __future__ import annotations

import random
from abc import ABC
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Final

from pokemon.message import Message
from pokemon.pokemon_status import PokemonStatus
from pokemon.pokemon_type import PokemonType

if TYPE_CHECKING:
    from pokemon.pokemon import Pokemon


_STAT_NAME_TO_INDEX: Final[dict[str, int]] = {
    "attack": 0,
    "defense": 1,
    "sp_attack": 2,
    "sp_defense": 3,
    "speed": 4,
}


_ACCURACY_MODIFIERS: Final[tuple[float, ...]] = (
    0.33,
    0.38,
    0.43,
    0.5,
    0.6,
    0.75,
    1.0,
    1.33,
    1.67,
    2.0,
    2.33,
    2.67,
    3.0,
)


@unique
class MoveCategory(IntEnum):
    """Enumeration for move categories (Physical, Special, Status)."""

    PHYSICAL = 0
    SPECIAL = 1
    STATUS = 2


class MoveError(Exception):
    """Custom exception for move-related errors."""

    pass


def _calculate_base_damage(
    power: int,
    atk: int,
    defi: int,
    level_factor: float,
    effectiveness: float,
    stab: bool,
    accuracy: int,
    acc_stage: int,
) -> tuple[int, int, bool, bool, float]:
    """
    A pure and optimized function for damage calculation.
    Detached from the class to be testable in isolation.
    """

    # Map the accuracy stage index (-6 to +6) to the array index (0 to 12)
    acc_idx = max(0, min(12, acc_stage + 6))
    hit_threshold = (accuracy * _ACCURACY_MODIFIERS[acc_idx]) / 100.0

    if random.random() >= hit_threshold:
        return 0, 0, False, stab, effectiveness

    # Standard Pokémon damage formula (approximated)
    raw_dmg = (level_factor * power * (atk / defi)) / 50 + 2

    # Random factor [0.85, 1.0]
    multiplier = effectiveness * (0.85 + 0.15 * random.random())

    is_critical = random.random() < 0.0625
    if is_critical:
        multiplier *= 1.5

    if stab:
        multiplier *= 1.5

    final_damage = int(raw_dmg * multiplier)

    # Base damage (raw info before RNG) useful for UI or AI
    base_info = int(raw_dmg * effectiveness * (1.5 if stab else 1.0))

    return final_damage, base_info, is_critical, stab, effectiveness


class Move(ABC):
    """
    Base class for all moves, optimized for memory using __slots__.
    """

    __slots__ = (
        "id",
        "name",
        "category",
        "type",
        "power",
        "accuracy",
        "pp",
        "max_pp",
        "priority",
    )

    def __init__(
        self,
        name: str,
        category: MoveCategory,
        p_type: PokemonType,
        power: int,
        accuracy: int,
        pp: int,
        priority: int = 0,
    ):
        self.id: int = -1  # Will be assigned by the registry
        self.name = name
        self.category = category
        self.type = p_type
        self.power = power
        self.accuracy = accuracy
        self.pp = pp
        self.max_pp = pp  # To restore PP
        self.priority = priority

    def calculate_damage(
        self, user: Pokemon, target: Pokemon
    ) -> tuple[int, int, bool, bool, float]:
        """
        Orchestrates stat retrieval and calls the pure calculation logic.
        """
        if self.category == MoveCategory.STATUS or self.power == 0:
            return 0, 0, False, False, 1.0

        # Dynamic stat determination (Physical vs Special)
        if self.category == MoveCategory.PHYSICAL:
            atk, defi = user.attack, target.defense
        else:
            atk, defi = user.sp_attack, target.sp_defense

        # Type effectiveness using the precomputed chart
        effectiveness = self.type.effectiveness_against(target.types)

        stab = self.type in user.types
        acc_stage = user._accuracy_stage

        return _calculate_base_damage(
            self.power,
            atk,
            defi,
            user.level_factor,
            effectiveness,
            stab,
            self.accuracy,
            acc_stage,
        )

    def secondary_effect(
        self,
        user: Pokemon,
        target: Pokemon,
        damage_dealt: int,
        base_damage: int,
        critical: bool,
        stab: bool,
        effectiveness: float,
    ) -> list[Message]:
        """Hook for secondary effects. By default, no effect."""
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} ({self.type.name})>"

    def __str__(self) -> str:
        return (
            f"{self.name} | Type: {self.type.label} | PP: {self.max_pp}/{self.max_pp}"
        )


# --- Specific Implementations ---


class SimpleMove(Move):
    """Standard move that only deals damage."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        category: MoveCategory,
        p_type: PokemonType,
        power: int,
        accuracy: int,
        pp: int,
        priority: int = 0,
    ):
        super().__init__(name, category, p_type, power, accuracy, pp, priority)


class StatMove(Move):
    """Move that modifies a Pokémon's stats (Buff/Debuff)."""

    __slots__ = (
        "stat_index",
        "is_accuracy_modifier",
        "stages",
        "self_target",
        "_stat_name",
    )

    def __init__(
        self,
        name: str,
        p_type: PokemonType,
        pp: int,
        stat: str,
        stages: int,
        self_target: bool = False,
        accuracy: int = 100,
        priority: int = 0,
    ):
        super().__init__(name, MoveCategory.STATUS, p_type, 0, accuracy, pp, priority)
        self._stat_name = stat

        if stat.lower() == "accuracy":
            self.is_accuracy_modifier = True
            self.stat_index = -1
        else:
            self.is_accuracy_modifier = False
            self.stat_index = _STAT_NAME_TO_INDEX.get(stat.lower())
            if self.stat_index is None:
                raise ValueError(f"Invalid stat name for StatMove: {stat}")

        self.stages = stages
        self.self_target = self_target

    def secondary_effect(
        self,
        user: Pokemon,
        target: Pokemon,
        damage_dealt: int,
        base_damage: int,
        critical: bool,
        stab: bool,
        effectiveness: float,
    ) -> list[Message]:
        """Applies the stat modification."""
        affected = user if self.self_target else target

        msgs = []
        success = False
        if self.is_accuracy_modifier:
            success = affected.apply_accuracy_modifier(self.stages)
        else:
            success = affected.apply_modifier(self.stat_index, self.stages)

        stat_name_for_message = self._stat_name

        if not success:
            msgs.append(
                Message(
                    f"{affected.surname}'s {stat_name_for_message} won't go any {'higher' if self.stages > 0 else 'lower'}!"
                )
            )
        else:
            adv = "rose" if self.stages > 0 else "fell"
            qualifier = "sharply " if abs(self.stages) > 1 else ""
            msgs.append(
                Message(
                    f"{affected.surname}'s {stat_name_for_message} {qualifier}{adv}!"
                )
            )
        return msgs


class ApplyStatusMove(Move):
    """
    Move that applies a status effect (volatile or non-volatile).
    This can be a pure status move (power=0) or a damaging move with a
    chance to apply a status.
    """

    __slots__ = (
        "status_to_apply",
        "volatile_status",
        "effect_chance",
        "duration",
        "self_target",
    )

    def __init__(
        self,
        name: str,
        category: MoveCategory,
        p_type: PokemonType,
        power: int,
        accuracy: int,
        pp: int,
        priority: int = 0,
        effect_chance: float = 1.0,
        status_to_apply: PokemonStatus | None = None,
        volatile_status: str | None = None,
        duration: tuple[int, int] | None = None,
        self_target: bool = False,
    ):
        super().__init__(name, category, p_type, power, accuracy, pp, priority)
        self.status_to_apply = status_to_apply
        self.volatile_status = volatile_status
        self.effect_chance = effect_chance
        self.duration = duration
        self.self_target = self_target

    def secondary_effect(
        self,
        user: Pokemon,
        target: Pokemon,
        damage_dealt: int,
        base_damage: int,
        critical: bool,
        stab: bool,
        effectiveness: float,
    ) -> list[Message]:
        # For damaging moves, only apply effect if damage was dealt.
        if self.power > 0 and damage_dealt == 0:
            return []

        # For pure status moves, check accuracy hit in the damage calc phase
        # Here we just check the effect chance.
        if random.random() > self.effect_chance:
            return []

        affected = user if self.self_target else target
        msgs = []

        # Apply non-volatile status
        if self.status_to_apply and affected.status == PokemonStatus.HEALTHY:
            affected.status = self.status_to_apply
            status_name = self.status_to_apply.name.lower()
            msgs.append(
                Message(f"{affected.surname} was afflicted with {status_name}!")
            )
            if self.status_to_apply == PokemonStatus.SLEEP and self.duration:
                affected._sleep_turns = random.randint(*self.duration)

        # Apply volatile status
        if self.volatile_status == "confuse" and not affected.confused:
            affected.confused = True
            if self.duration:
                affected._confused_turns = random.randint(*self.duration)
            msgs.append(Message(f"{affected.surname} became confused!"))

        elif self.volatile_status == "taunt" and not affected.taunted:
            affected.taunted = True
            # Taunt has a fixed duration in later gens, but can be random here
            affected._taunted_turns = self.duration[0] if self.duration else 3
            msgs.append(Message(f"{affected.surname} was taunted!"))

        return msgs


class RecoilMove(Move):
    """A move that inflicts recoil damage on the user (e.g., Take Down)."""

    __slots__ = ("recoil_ratio",)

    def __init__(
        self,
        name: str,
        category: MoveCategory,
        p_type: PokemonType,
        power: int,
        accuracy: int,
        pp: int,
        recoil_ratio: float = 0.25,
        priority: int = 0,
    ):
        super().__init__(name, category, p_type, power, accuracy, pp, priority)
        self.recoil_ratio = recoil_ratio

    def secondary_effect(
        self,
        user: Pokemon,
        target: Pokemon,
        damage_dealt: int,
        base_damage: int,
        critical: bool,
        stab: bool,
        effectiveness: float,
    ) -> list[Message]:
        """Applies recoil damage to the user."""
        if damage_dealt > 0:
            recoil = max(1, int(damage_dealt * self.recoil_ratio))
            user.hp -= recoil
            return [Message(f"{user.surname} is hit with recoil!")]
        return []


class StruggleMove(RecoilMove):
    """Specific implementation for the move Struggle."""

    __slots__ = ()

    def __init__(self):
        super().__init__(
            "Struggle",
            MoveCategory.PHYSICAL,
            PokemonType.NORMAL,
            50,
            100,
            1,
            recoil_ratio=0.0,
        )

    def secondary_effect(
        self,
        user: Pokemon,
        target: Pokemon,
        damage_dealt: int,
        base_damage: int,
        critical: bool,
        stab: bool,
        effectiveness: float,
    ) -> list[Message]:
        """
        Struggle has a unique mechanic: 25% of max HP in recoil,
        regardless of damage dealt.
        """
        recoil = max(1, user.max_hp // 4)
        user.hp -= recoil
        return [Message(f"{user.surname} lost {recoil} HP due to recoil!")]


# --- Registry System ---


class MoveRegistry:
    """
    Singleton that manages the registration of and access to Moves.
    Guarantees unique and sequential IDs.
    """

    _moves: list[Move] = []
    _map: dict[str, Move] = {}

    @classmethod
    def register(cls, move: Move) -> Move:
        """Assigns a unique ID to the move and registers it."""
        move.id = len(cls._moves)
        cls._moves.append(move)
        # Normalized key for case/space insensitive search
        key = move.name.replace(" ", "").lower()
        if key in cls._map:
            raise MoveError(f"Duplicate move name registered: {move.name}")
        cls._map[key] = move
        return move

    @classmethod
    def get(cls, name_or_id: str | int) -> Move | None:
        """Retrieves a move by its name (str) or ID (int)."""
        if isinstance(name_or_id, int):
            return cls._moves[name_or_id] if 0 <= name_or_id < len(cls._moves) else None
        return cls._map.get(str(name_or_id).replace(" ", "").lower())

    @classmethod
    def all(cls) -> list[Move]:
        """Returns a list of all registered moves."""
        return cls._moves

    @classmethod
    def count(cls) -> int:
        """Returns the total number of registered moves."""
        return len(cls._moves)


# --- Registration ---

# System
Struggle = MoveRegistry.register(StruggleMove())

# Physical
Tackle = MoveRegistry.register(
    SimpleMove("Tackle", MoveCategory.PHYSICAL, PokemonType.NORMAL, 35, 95, 35)
)
Scratch = MoveRegistry.register(
    SimpleMove("Scratch", MoveCategory.PHYSICAL, PokemonType.NORMAL, 40, 100, 35)
)
Pound = MoveRegistry.register(
    SimpleMove("Pound", MoveCategory.PHYSICAL, PokemonType.NORMAL, 40, 100, 35)
)
Peck = MoveRegistry.register(
    SimpleMove("Peck", MoveCategory.PHYSICAL, PokemonType.FLYING, 35, 100, 35)
)
VineWhip = MoveRegistry.register(
    SimpleMove("Vine Whip", MoveCategory.PHYSICAL, PokemonType.GRASS, 35, 100, 10)
)
RazorLeaf = MoveRegistry.register(
    SimpleMove("Razor Leaf", MoveCategory.PHYSICAL, PokemonType.GRASS, 55, 95, 25)
)
QuickAttack = MoveRegistry.register(
    SimpleMove(
        "Quick Attack",
        MoveCategory.PHYSICAL,
        PokemonType.NORMAL,
        40,
        100,
        30,
        priority=1,
    )
)
WingAttack = MoveRegistry.register(
    SimpleMove("Wing Attack", MoveCategory.PHYSICAL, PokemonType.FLYING, 35, 100, 35)
)
Earthquake = MoveRegistry.register(
    SimpleMove("Earthquake", MoveCategory.PHYSICAL, PokemonType.GROUND, 100, 100, 10)
)
Slash = MoveRegistry.register(
    SimpleMove("Slash", MoveCategory.PHYSICAL, PokemonType.NORMAL, 70, 100, 20)
)
Bite = MoveRegistry.register(
    SimpleMove("Bite", MoveCategory.PHYSICAL, PokemonType.DARK, 60, 100, 25)
)
RockSlide = MoveRegistry.register(
    SimpleMove("Rock Slide", MoveCategory.PHYSICAL, PokemonType.ROCK, 75, 90, 10)
)
IronTail = MoveRegistry.register(
    SimpleMove("Iron Tail", MoveCategory.PHYSICAL, PokemonType.STEEL, 100, 75, 15)
)
BodySlam = MoveRegistry.register(
    SimpleMove("Body Slam", MoveCategory.PHYSICAL, PokemonType.NORMAL, 85, 100, 15)
)
ExtremeSpeed = MoveRegistry.register(
    SimpleMove(
        "Extreme Speed",
        MoveCategory.PHYSICAL,
        PokemonType.NORMAL,
        80,
        100,
        5,
        priority=2,
    )
)
SkullBash = MoveRegistry.register(
    SimpleMove("Skull Bash", MoveCategory.PHYSICAL, PokemonType.NORMAL, 130, 100, 10)
)
SkyAttack = MoveRegistry.register(
    SimpleMove("Sky Attack", MoveCategory.PHYSICAL, PokemonType.FLYING, 140, 90, 5)
)


# Special
ThunderShock = MoveRegistry.register(
    SimpleMove("Thunder Shock", MoveCategory.SPECIAL, PokemonType.ELECTRIC, 40, 100, 30)
)
Ember = MoveRegistry.register(
    ApplyStatusMove(
        "Ember",
        MoveCategory.SPECIAL,
        PokemonType.FIRE,
        power=40,
        accuracy=100,
        pp=25,
        effect_chance=0.1,  # 10% chance
        status_to_apply=PokemonStatus.BURN,
    )
)
WaterGun = MoveRegistry.register(
    SimpleMove("Water Gun", MoveCategory.SPECIAL, PokemonType.WATER, 40, 100, 25)
)
Flamethrower = MoveRegistry.register(
    SimpleMove("Flamethrower", MoveCategory.SPECIAL, PokemonType.FIRE, 95, 100, 15)
)
HydroPump = MoveRegistry.register(
    SimpleMove("Hydro Pump", MoveCategory.SPECIAL, PokemonType.WATER, 120, 80, 5)
)
Thunderbolt = MoveRegistry.register(
    SimpleMove("Thunderbolt", MoveCategory.SPECIAL, PokemonType.ELECTRIC, 95, 100, 15)
)
Psychic = MoveRegistry.register(
    SimpleMove("Psychic", MoveCategory.SPECIAL, PokemonType.PSYCHIC, 90, 100, 10)
)
SolarBeam = MoveRegistry.register(
    SimpleMove("Solar Beam", MoveCategory.SPECIAL, PokemonType.GRASS, 120, 100, 10)
)
HyperBeam = MoveRegistry.register(
    SimpleMove("Hyper Beam", MoveCategory.SPECIAL, PokemonType.NORMAL, 150, 90, 5)
)
Blizzard = MoveRegistry.register(
    SimpleMove("Blizzard", MoveCategory.SPECIAL, PokemonType.ICE, 110, 70, 5)
)
DragonRage = MoveRegistry.register(
    SimpleMove("Dragon Rage", MoveCategory.SPECIAL, PokemonType.DRAGON, 40, 100, 10)
)

# Status
Growl = MoveRegistry.register(StatMove("Growl", PokemonType.NORMAL, 40, "attack", -1))
TailWhip = MoveRegistry.register(
    StatMove("Tail Whip", PokemonType.NORMAL, 30, "defense", -1)
)
SandAttack = MoveRegistry.register(
    StatMove("Sand Attack", PokemonType.GROUND, 15, "accuracy", -1, accuracy=100)
)
Agility = MoveRegistry.register(
    StatMove("Agility", PokemonType.PSYCHIC, 30, "speed", 2, self_target=True)
)


SleepPowder = MoveRegistry.register(
    ApplyStatusMove(
        "Sleep Powder",
        MoveCategory.STATUS,
        PokemonType.GRASS,
        power=0,
        accuracy=75,
        pp=15,
        status_to_apply=PokemonStatus.SLEEP,
        duration=(1, 3),
    )
)

ThunderWave = MoveRegistry.register(
    ApplyStatusMove(
        "Thunder Wave",
        MoveCategory.STATUS,
        PokemonType.ELECTRIC,
        power=0,
        accuracy=90,
        pp=20,
        status_to_apply=PokemonStatus.PARALYSIS,
    )
)

WillOWisp = MoveRegistry.register(
    ApplyStatusMove(
        "Will-O-Wisp",
        MoveCategory.STATUS,
        PokemonType.FIRE,
        power=0,
        accuracy=85,
        pp=15,
        status_to_apply=PokemonStatus.BURN,
    )
)

Supersonic = MoveRegistry.register(
    ApplyStatusMove(
        "Supersonic",
        MoveCategory.STATUS,
        PokemonType.NORMAL,
        power=0,
        accuracy=55,
        pp=20,
        volatile_status="confuse",
        duration=(1, 4),
    )
)

Taunt = MoveRegistry.register(
    ApplyStatusMove(
        "Taunt",
        MoveCategory.STATUS,
        PokemonType.DARK,
        power=0,
        accuracy=100,
        pp=20,
        volatile_status="taunt",
        duration=(3, 3),
    )
)


# --- Accessor Helper ---


class _MoveAccessorMeta(type):
    """Metaclass to allow accessing moves like Moves.Tackle."""

    def __getattr__(cls, name: str) -> Move:
        move = MoveRegistry.get(name)
        if move:
            return move
        raise AttributeError(f"Move '{name}' not found in MoveRegistry.")


class Moves(metaclass=_MoveAccessorMeta):
    """
    Provides a clean, dot-notation accessor for all registered moves.
    Usage: Moves.Tackle, Moves.Ember.
    """

    pass
