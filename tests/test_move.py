import random
from unittest.mock import patch

import pytest

from pokemon.move import (
    MoveCategory,
    MoveError,
    MoveRegistry,
    Moves,
    RecoilMove,
    _calculate_base_damage,
)
from pokemon.pokemon import Pokedex, Pokemon, PokemonStatus
from pokemon.pokemon_type import PokemonType


@pytest.fixture
def charmander():
    """Creates a Fire type Pokemon (Balanced)."""
    p = Pokemon(species=Pokedex.Charmander, level=10)
    return p


@pytest.fixture
def squirtle():
    """Creates a Water type Pokemon (High Defense)."""
    p = Pokemon(species=Pokedex.Squirtle, level=10)
    return p


@pytest.fixture
def rattata():
    """Creates a Normal type Pokemon (Physical)."""
    p = Pokemon(species=Pokedex.Rattata, level=10)
    return p


def test_physical_damage_calculation(rattata: Pokemon, squirtle: Pokemon):
    """
    Test that a physical move uses Attack and Defense stats.
    Rattata (Normal) uses Tackle (Physical).
    """
    move = Moves.Tackle
    assert move.category == MoveCategory.PHYSICAL

    # We use a fixed seed to ensure deterministic calculations
    # (since the formula involves random variance)
    random.seed(42)

    damage, base_info, is_crit, stab, effectiveness = move.calculate_damage(
        rattata, squirtle
    )

    assert isinstance(damage, int)
    assert damage > 0
    assert effectiveness == 1.0
    assert stab is True


def test_type_effectiveness_super_effective(squirtle: Pokemon, charmander: Pokemon):
    """
    Test Super Effective damage (Water -> Fire).
    """
    move = Moves.WaterGun  # Water type

    _, _, _, _, effectiveness = move.calculate_damage(squirtle, charmander)

    assert effectiveness == 2.0  # Water is super effective against Fire


def test_type_effectiveness_not_effective(charmander: Pokemon, squirtle: Pokemon):
    """
    Test Not Very Effective damage (Fire -> Water).
    """
    move = Moves.Ember  # Fire type

    _, _, _, _, effectiveness = move.calculate_damage(charmander, squirtle)

    assert effectiveness == 0.5  # Fire is not very effective against Water


def test_status_move_debuff(rattata: Pokemon, squirtle: Pokemon):
    """
    Test that a Status move (Tail Whip) modifies the target's stats.
    """
    move = Moves.TailWhip

    # Pre-condition
    assert squirtle.modifiers[1] == 0 # Defense is index 1

    # Simulate application (Move.secondary_effect)
    msgs = move.secondary_effect(rattata, squirtle, 0, 0, False, False, 1.0)

    # Post-condition: Defense should be -1
    assert squirtle.modifiers[1] == -1
    assert "fell" in msgs[0].text


def test_status_move_self_buff(charmander: Pokemon):
    """
    Test that a self-targeting Status move (Agility) modifies the user's stats.
    """
    move = Moves.Agility

    # Agility raises speed by 2 stages for the user
    msgs = move.secondary_effect(charmander, charmander, 0, 0, False, False, 1.0)

    assert charmander.modifiers[4] == 2 # Speed is index 4
    assert "sharply rose" in msgs[0].text


def test_ember_secondary_effect(charmander: Pokemon, rattata: Pokemon):
    """
    Test that Ember applies the BURN status (10% chance).
    We use a mock to force the 10% chance to trigger success.
    """
    move = Moves.Ember

    # Ensure Rattata is fresh
    rattata.hp = rattata.max_hp
    rattata.status = PokemonStatus.HEALTHY

    # We simulate a hit that deals 5 damage.
    # Since Rattata is Level 10, it has > 5 HP, so it won't die.
    damage_dealt = 5

    # Simulate the HP loss that occurs before the secondary effect check
    rattata.hp -= damage_dealt

    # We mock random.random to return 0.05.
    # Since 0.05 < 0.10, the 10% chance requirement is met.
    with patch("random.random", return_value=0.05):
        msgs = move.secondary_effect(
            charmander,
            rattata,
            damage_dealt=damage_dealt,
            base_damage=damage_dealt,
            critical=False,
            stab=True,
            effectiveness=1.0,
        )

    # Assertions
    assert rattata.hp > 0, "Rattata should be alive"
    assert rattata.status == PokemonStatus.BURN, "Status should be updated to BURN"
    assert "burn" in msgs[0].text


def test_struggle_recoil_mechanic(charmander: Pokemon, squirtle: Pokemon):
    """
    Test Struggle specific recoil: user loses 1/4 of Max HP.
    """
    move = Moves.Struggle
    initial_hp = charmander.hp
    max_hp = charmander.max_hp

    # Execute secondary effect
    msgs = move.secondary_effect(charmander, squirtle, 10, 10, False, False, 1.0)

    expected_loss = max(1, max_hp // 4)
    assert charmander.hp == initial_hp - expected_loss
    assert "recoil" in msgs[0].text


def test_standard_recoil_move(charmander, squirtle):
    """
    Test generic RecoilMove logic (e.g. user takes % of damage dealt).
    Creating a custom move instance since standard registry moves might change.
    """
    recoil_move = RecoilMove(
        "TestTakedown",
        MoveCategory.PHYSICAL,
        PokemonType.NORMAL,
        90,
        100,
        20,
        recoil_ratio=0.5,
    )

    damage_dealt = 50
    initial_hp = charmander.hp

    # Apply recoil
    msgs = recoil_move.secondary_effect(
        charmander, squirtle, damage_dealt, damage_dealt, False, False, 1.0
    )

    # Should lose 50% of 50 = 25 HP
    assert charmander.hp == initial_hp - 25
    assert "recoil" in msgs[0].text


def test_registry_access_and_errors():
    """
    Test that the registry works correctly via dot notation and string lookup.
    """
    # 1. Accessor
    assert Moves.Tackle.name == "Tackle"

    # 2. String lookup (Case insensitive)
    assert MoveRegistry.get("flamethrower").name == "Flamethrower"

    # 3. Invalid Access
    with pytest.raises(AttributeError):
        _ = Moves.NonExistentMove

    with pytest.raises(MoveError):
        # Registering a duplicate name
        MoveRegistry.register(
            RecoilMove("Tackle", MoveCategory.PHYSICAL, PokemonType.NORMAL, 1, 1, 1)
        )


def test_calculation_miss_logic():
    """
    Test that damage is 0 if the move misses.
    """
    # We force accuracy failure by passing accuracy=0 to the internal calculator
    damage, _, _, _, _ = _calculate_base_damage(
        power=100,
        atk=50,
        defi=50,
        level_factor=1.0,
        effectiveness=1.0,
        stab=False,
        accuracy=0,  # Impossible to hit
        acc_stage=0,
    )
    assert damage == 0
