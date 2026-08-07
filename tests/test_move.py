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
def chimchar():
    """Creates a Fire type Pokemon (Balanced)."""
    p = Pokemon(species=Pokedex.Chimchar, level=10)
    return p


@pytest.fixture
def piplup():
    """Creates a Water type Pokemon (High Defense)."""
    p = Pokemon(species=Pokedex.Piplup, level=10)
    return p


@pytest.fixture
def starly():
    """Creates a Normal/Flying type Pokemon (Physical)."""
    p = Pokemon(species=Pokedex.Starly, level=10)
    return p


def test_physical_damage_calculation(starly: Pokemon, piplup: Pokemon):
    """
    Test that a physical move uses Attack and Defense stats.
    Starly (Normal/Flying) uses Tackle (Physical).
    """
    move = Moves.Tackle
    assert move.category == MoveCategory.PHYSICAL

    # We use a fixed seed to ensure deterministic calculations
    # (since the formula involves random variance)
    random.seed(42)

    damage, base_info, is_crit, stab, effectiveness = move.calculate_damage(
        starly, piplup
    )

    assert isinstance(damage, int)
    assert damage > 0
    assert effectiveness == 1.0
    assert stab is True


def test_type_effectiveness_super_effective(piplup: Pokemon, chimchar: Pokemon):
    """
    Test Super Effective damage (Water -> Fire).
    """
    move = Moves.WaterGun  # Water type

    _, _, _, _, effectiveness = move.calculate_damage(piplup, chimchar)

    assert effectiveness == 2.0  # Water is super effective against Fire


def test_type_effectiveness_not_effective(chimchar: Pokemon, piplup: Pokemon):
    """
    Test Not Very Effective damage (Fire -> Water).
    """
    move = Moves.Ember  # Fire type

    _, _, _, _, effectiveness = move.calculate_damage(chimchar, piplup)

    assert effectiveness == 0.5  # Fire is not very effective against Water


def test_status_move_debuff(starly: Pokemon, piplup: Pokemon):
    """
    Test that a Status move (Tail Whip) modifies the target's stats.
    """
    move = Moves.TailWhip

    # Pre-condition
    assert piplup.modifiers[1] == 0  # Defense is index 1

    # Simulate application (Move.secondary_effect)
    msgs = move.secondary_effect(starly, piplup, 0, 0, False, False, 1.0)

    # Post-condition: Defense should be -1
    assert piplup.modifiers[1] == -1
    assert "fell" in msgs[0].text


def test_status_move_self_buff(chimchar: Pokemon):
    """
    Test that a self-targeting Status move (Agility) modifies the user's stats.
    """
    move = Moves.Agility

    # Agility raises speed by 2 stages for the user
    msgs = move.secondary_effect(chimchar, chimchar, 0, 0, False, False, 1.0)

    assert chimchar.modifiers[4] == 2  # Speed is index 4
    assert "sharply rose" in msgs[0].text


def test_ember_secondary_effect(chimchar: Pokemon, starly: Pokemon):
    """
    Test that Ember applies the BURN status (10% chance).
    We use a mock to force the 10% chance to trigger success.
    """
    move = Moves.Ember

    # Ensure Starly is fresh
    starly.hp = starly.max_hp
    starly.status = PokemonStatus.HEALTHY

    # We simulate a hit that deals 5 damage.
    # Since Starly is Level 10, it has > 5 HP, so it won't die.
    damage_dealt = 5

    # Simulate the HP loss that occurs before the secondary effect check
    starly.hp -= damage_dealt

    # We mock random.random to return 0.05.
    # Since 0.05 < 0.10, the 10% chance requirement is met.
    with patch("random.random", return_value=0.05):
        msgs = move.secondary_effect(
            chimchar,
            starly,
            damage_dealt=damage_dealt,
            base_damage=damage_dealt,
            critical=False,
            stab=True,
            effectiveness=1.0,
        )

    # Assertions
    assert starly.hp > 0, "Starly should be alive"
    assert starly.status == PokemonStatus.BURN, "Status should be updated to BURN"
    assert "burn" in msgs[0].text


def test_struggle_recoil_mechanic(chimchar: Pokemon, piplup: Pokemon):
    """
    Test Struggle specific recoil: user loses 1/4 of Max HP.
    """
    move = Moves.Struggle
    initial_hp = chimchar.hp
    max_hp = chimchar.max_hp

    # Execute secondary effect
    msgs = move.secondary_effect(chimchar, piplup, 10, 10, False, False, 1.0)

    expected_loss = max(1, max_hp // 4)
    assert chimchar.hp == initial_hp - expected_loss
    assert "recoil" in msgs[0].text


def test_standard_recoil_move(chimchar, piplup):
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
    initial_hp = chimchar.hp

    # Apply recoil
    msgs = recoil_move.secondary_effect(
        chimchar, piplup, damage_dealt, damage_dealt, False, False, 1.0
    )

    # Should lose 50% of 50 = 25 HP
    assert chimchar.hp == initial_hp - 25
    assert "recoil" in msgs[0].text


def test_registry_access_and_errors():
    """
    Test that the registry works correctly via dot notation and string lookup.
    """
    # 1. Accessor
    assert Moves.Tackle.name == "Tackle"

    # 2. String lookup (Case insensitive)
    flamethrower = MoveRegistry.get("flamethrower")
    assert flamethrower is not None
    assert flamethrower.name == "Flamethrower"

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


def test_pure_status_move_accuracy_rolls(chimchar: Pokemon, piplup: Pokemon):
    """
    Regression test: pure status moves (power == 0) must honor their listed
    accuracy (e.g. Sleep Powder is 75%, not 100%).
    """
    sleep_powder = MoveRegistry.get("Sleep Powder")
    assert sleep_powder is not None
    assert sleep_powder.power == 0
    assert sleep_powder.accuracy == 75

    rng = random.Random(42)
    sleeps = 0
    for _ in range(1000):
        piplup.status = PokemonStatus.HEALTHY
        _, bd, crit, stab, eff = sleep_powder.calculate_damage(
            chimchar, piplup, rng=rng
        )
        sleep_powder.secondary_effect(chimchar, piplup, 0, bd, crit, stab, eff, rng=rng)
        if piplup.status == PokemonStatus.SLEEP:
            sleeps += 1

    # ~750/1000; wide tolerance to avoid flakes.
    assert 650 <= sleeps <= 850


def test_status_move_accuracy_scaled_by_user_stage(chimchar: Pokemon, piplup: Pokemon):
    """
    Regression test: a Pokemon whose accuracy was lowered (e.g. by Sand Attack)
    should miss pure status moves more often.
    """
    sleep_powder = MoveRegistry.get("Sleep Powder")
    chimchar.apply_accuracy_modifier(-2)  # 0.6x multiplier

    rng = random.Random(42)
    sleeps = 0
    for _ in range(1000):
        piplup.status = PokemonStatus.HEALTHY
        _, bd, crit, stab, eff = sleep_powder.calculate_damage(
            chimchar, piplup, rng=rng
        )
        sleep_powder.secondary_effect(chimchar, piplup, 0, bd, crit, stab, eff, rng=rng)
        if piplup.status == PokemonStatus.SLEEP:
            sleeps += 1

    # 75% * 0.6 = 45%; wide tolerance.
    assert 350 <= sleeps <= 550


def test_always_hits_move_ignores_accuracy_stage(chimchar: Pokemon, piplup: Pokemon):
    """
    Regression test: Aura Sphere (always_hits=True) never misses, even when
    the user's accuracy is at -6.
    """
    aura_sphere = MoveRegistry.get("Aura Sphere")
    assert aura_sphere is not None
    assert aura_sphere.always_hits is True

    chimchar.apply_accuracy_modifier(-6)  # 0.33x multiplier
    rng = random.Random(1)
    damage, _, _, _, _ = aura_sphere.calculate_damage(chimchar, piplup, rng=rng)
    assert damage > 0


def test_mud_slap_lowers_accuracy_stage(chimchar: Pokemon, piplup: Pokemon):
    """
    Regression test: Mud-Slap applies an accuracy stage drop to the target.
    """
    mud_slap = MoveRegistry.get("Mud-Slap")
    assert mud_slap is not None
    assert mud_slap.accuracy_stages == -1

    msgs = mud_slap.secondary_effect(
        chimchar, piplup, 9, 9, False, False, 1.0, rng=random.Random(1)
    )
    assert piplup._accuracy_stage == -1
    assert "fell" in msgs[0].text


def test_stat_move_debuff_misses_with_low_accuracy(chimchar: Pokemon, piplup: Pokemon):
    """
    Regression test: a non-self-targeting stat move with a low accuracy (e.g.
    70%) should miss when the accuracy roll fails, not always apply.
    """
    from pokemon.move import StatMove

    low_acc_stat = StatMove(
        "TestStatDebuff",
        PokemonType.GROUND,
        15,
        "speed",  # stat to lower
        -1,
        self_target=False,
        accuracy=70,  # 70% accuracy so it can miss
    )

    # Patch random to force a miss (0.9 > 0.70).
    with patch("random.random", return_value=0.9):
        msgs = low_acc_stat.secondary_effect(chimchar, piplup, 0, 0, False, False, 1.0)
    assert "missed" in msgs[0].text
    assert piplup.modifiers[4] == 0  # No stat change applied

    # And a hit applies it.
    with patch("random.random", return_value=0.1):
        msgs = low_acc_stat.secondary_effect(chimchar, piplup, 0, 0, False, False, 1.0)
    assert "missed" not in msgs[0].text
    assert piplup.modifiers[4] == -1


def test_effectiveness_with_three_types():
    """
    Regression test: type effectiveness must not crash when a defender has
    more than two types (the old chart lookup raised IndexError).
    """
    # A synthetic 3-type defender (no Fire, so Fire deals 2x * 2x * 1x).
    result = PokemonType.FIRE.effectiveness_against(
        (PokemonType.GRASS, PokemonType.BUG, PokemonType.ELECTRIC)
    )
    assert result == 4.0  # 2x * 2x * 1x
