from unittest.mock import patch

import pytest

from pokemon.move import MoveCategory, MoveRegistry, PokemonType
from pokemon.pokemon import (
    _STAT_STAGE_MULTIPLIERS,
    Pokedex,
    Pokemon,
    PokemonStatus,
    SpeciesRegistry,
)


@pytest.fixture(scope="module", autouse=True)
def setup_moves():
    """Ensure some moves are registered for testing purposes."""
    if not MoveRegistry.get("Thunder Shock"):
        MoveRegistry.register(
            name="Thunder Shock",
            move_type=PokemonType.ELECTRIC,
            category=MoveCategory.SPECIAL,
            power=40,
            accuracy=100,
            pp=30,
        )
    if not MoveRegistry.get("Growl"):
        MoveRegistry.register(
            name="Growl",
            move_type=PokemonType.NORMAL,
            category=MoveCategory.STATUS,
            power=0,
            accuracy=100,
            pp=40,
        )
    if not MoveRegistry.get("Quick Attack"):
        MoveRegistry.register(
            name="Quick Attack",
            move_type=PokemonType.NORMAL,
            category=MoveCategory.PHYSICAL,
            power=40,
            accuracy=100,
            pp=30,
        )
    if not MoveRegistry.get("Tail Whip"):
        MoveRegistry.register(
            name="Tail Whip",
            move_type=PokemonType.NORMAL,
            category=MoveCategory.STATUS,
            power=0,
            accuracy=100,
            pp=30,
        )


@pytest.fixture
def pikachu() -> Pokemon:
    """Provides a default level 5 Pikachu for tests."""
    return Pokemon(species=Pokedex.Pikachu, level=5)


def test_basic_creation(pikachu: Pokemon):
    assert pikachu.species.name == "Pikachu"
    assert pikachu.surname == "Pikachu"
    assert pikachu.level == 5
    assert pikachu.hp == pikachu.max_hp
    assert pikachu.is_alive is True
    assert len(pikachu.move_slots) > 0
    assert pikachu.move_slots[0].move.name == "Thunder Shock"

def test_custom_attributes():
    char = Pokemon(
        species=Pokedex.Charmander,
        level=10,
        surname="Charry",
        ivs=(31, 31, 31, 31, 31, 31),
    )
    assert char.level == 10
    assert char.surname == "Charry"
    assert char._ivs == (31, 31, 31, 31, 31, 31)

def test_stat_calculation():
    """
    Verify stats against a known calculator for a level 50 Bulbasaur.
    IVs: (31, 31, 31, 31, 31, 31), EVs: (0, 0, 0, 0, 0, 0)
    Expected HP: 120
    Expected other stats: 69, 69, 85, 85, 65.
    """
    bulbasaur = Pokemon(
        species=Pokedex.Bulbasaur, level=50, ivs=(31, 31, 31, 31, 31, 31)
    )
    assert bulbasaur.max_hp == 120
    assert bulbasaur.attack == 69
    assert bulbasaur.defense == 69
    assert bulbasaur.sp_attack == 85
    assert bulbasaur.sp_defense == 85
    assert bulbasaur.speed == 65


def test_hp_setter_and_fainting(pikachu: Pokemon):
    pikachu.hp -= 10
    assert pikachu.hp == pikachu.max_hp - 10
    assert pikachu.is_alive is True

    pikachu.hp = -5  # Should clamp to 0
    assert pikachu.hp == 0
    assert pikachu.is_alive is False
    assert pikachu.status == PokemonStatus.FAINTED

    pikachu.hp = 1000  # Should clamp to max_hp
    assert pikachu.hp == pikachu.max_hp
    assert pikachu.is_alive is True  # Fainting is reversible

def test_restore_and_reset(pikachu: Pokemon):
    pikachu.hp -= 5
    pikachu.status = PokemonStatus.BURN
    pikachu.apply_modifier(0, -2)  # -2 Attack
    pikachu.decrease_pp(0)

    assert pikachu.hp < pikachu.max_hp
    assert pikachu.attack_stage == -2
    assert pikachu.move_slots[0].current_pp < pikachu.move_slots[0].max_pp

    pikachu.full_restore()
    assert pikachu.hp == pikachu.max_hp
    assert pikachu.status == PokemonStatus.HEALTHY
    assert pikachu.attack_stage == 0
    # full_restore does not reset PP
    assert pikachu.move_slots[0].current_pp < pikachu.move_slots[0].max_pp

    pikachu.reset()
    assert pikachu.move_slots[0].current_pp == pikachu.move_slots[0].max_pp

def test_copy(pikachu: Pokemon):
    p1 = pikachu
    p1.apply_modifier(0, 2)  # +2 Attack
    p1.hp -= 5

    p2 = p1.copy()

    assert p1 is not p2
    assert p1.species is p2.species  # Should be a reference
    assert p1.hp == p2.hp
    assert p1.attack == p2.attack
    assert p1.move_slots[0].current_pp == p2.move_slots[0].current_pp

    # Modify copy and check original is unchanged
    p2.hp -= 5
    p2.apply_modifier(0, -1)  # p2 attack stage should be +1 now
    p2.decrease_pp(0)

    assert p1.hp != p2.hp
    assert p1.attack_stage == 2
    assert p2.attack_stage == 1
    assert p1.attack != p2.attack
    assert p1.move_slots[0].current_pp != p2.move_slots[0].current_pp


@pytest.mark.parametrize(
    "stat_idx, prop_name",
    [
        (0, "attack"),
        (1, "defense"),
        (2, "sp_attack"),
        (3, "sp_defense"),
        (4, "speed"),
    ],
)
def test_apply_modifier(pikachu: Pokemon, stat_idx, prop_name):
    initial_stat = getattr(pikachu, prop_name)
    raw_stat = pikachu._raw_stats[stat_idx]

    # Increase stage
    changed = pikachu.apply_modifier(stat_idx, 2)
    assert changed is True
    assert pikachu._stat_stages[stat_idx] == 2
    expected = int(raw_stat * _STAT_STAGE_MULTIPLIERS[2 + 6])
    assert getattr(pikachu, prop_name) == expected

    # Decrease stage
    changed = pikachu.apply_modifier(stat_idx, -3)
    assert changed is True
    assert pikachu._stat_stages[stat_idx] == -1
    expected = int(raw_stat * _STAT_STAGE_MULTIPLIERS[-1 + 6])
    assert getattr(pikachu, prop_name) == expected

def test_modifier_limits(pikachu: Pokemon):
    # Test upper limit
    pikachu.apply_modifier(0, 6)
    assert pikachu.attack_stage == 6
    changed = pikachu.apply_modifier(0, 1)
    assert changed is False
    assert pikachu.attack_stage == 6

    # Test lower limit
    pikachu.apply_modifier(1, -6)
    assert pikachu.defense_stage == -6
    changed = pikachu.apply_modifier(1, -1)
    assert changed is False
    assert pikachu.defense_stage == -6

def test_reset_modifiers(pikachu: Pokemon):
    initial_attack = pikachu.attack
    pikachu.apply_modifier(0, 4)
    assert pikachu.attack != initial_attack

    pikachu.reset_modifiers()
    assert pikachu.attack_stage == 0
    assert pikachu.attack == initial_attack


@patch("random.random", return_value=0.1)  # Fails paralysis/freeze check
def test_can_attack_paralyzed_fail(mock_random, pikachu: Pokemon):
    pikachu.status = PokemonStatus.PARALYSIS
    can_attack, msgs = pikachu.can_attack()
    assert can_attack is False
    assert "paralyzed" in msgs[0].text

@patch("random.random", return_value=0.9)  # Passes paralysis/freeze check
def test_can_attack_paralyzed_ok(mock_random, pikachu: Pokemon):
    pikachu.status = PokemonStatus.PARALYSIS
    can_attack, msgs = pikachu.can_attack()
    assert can_attack is True
    assert len(msgs) == 0

@patch("random.random", return_value=0.1)  # Thaws out
def test_can_attack_frozen_thaw(mock_random, pikachu: Pokemon):
    pikachu.status = PokemonStatus.FREEZE
    can_attack, msgs = pikachu.can_attack()
    assert can_attack is True
    assert "thawed out" in msgs[0].text
    assert pikachu.status == PokemonStatus.HEALTHY

@patch("random.random", return_value=0.4)  # Hurts itself
def test_can_attack_confused_fail(mock_random, pikachu: Pokemon):
    pikachu.confused = True
    pikachu._confused_turns = 3
    initial_hp = pikachu.hp
    can_attack, msgs = pikachu.can_attack()
    assert can_attack is False
    assert "confused" in msgs[0].text
    assert "hurt itself" in msgs[1].text
    assert pikachu.hp < initial_hp

@patch("random.random", return_value=0.9)  # Attacks normally
def test_can_attack_confused_ok(mock_random, pikachu: Pokemon):
    pikachu.confused = True
    pikachu._confused_turns = 3
    initial_hp = pikachu.hp
    can_attack, msgs = pikachu.can_attack()
    assert can_attack is True
    assert "confused" in msgs[0].text
    assert pikachu.hp == initial_hp

def test_after_turn_burn_damage(pikachu: Pokemon):
    pikachu.status = PokemonStatus.BURN
    initial_hp = pikachu.hp
    expected_dmg = max(1, pikachu.max_hp >> 3)
    msgs = pikachu.after_turn()
    assert pikachu.hp == initial_hp - expected_dmg
    assert "hurt by its burn" in msgs[0].text

def test_after_turn_poison_damage(pikachu: Pokemon):
    pikachu.status = PokemonStatus.POISON
    initial_hp = pikachu.hp
    expected_dmg = max(1, pikachu.max_hp >> 3)
    msgs = pikachu.after_turn()
    assert pikachu.hp == initial_hp - expected_dmg
    assert "hurt by poison" in msgs[0].text


def test_species_registry_get():
    pika_by_name = SpeciesRegistry.get("Pikachu")
    assert pika_by_name is not None
    assert pika_by_name.name == "Pikachu"
    pika_by_id = SpeciesRegistry.get(pika_by_name.id)
    assert pika_by_name is pika_by_id

def test_pokedex_getattr():
    assert Pokedex.Pikachu.name == "Pikachu"
    assert Pokedex.Charmander.types == (PokemonType.FIRE,)

def test_pokedex_not_found():
    with pytest.raises(AttributeError):
        _ = Pokedex.Mewtwo  # Assuming Mewtwo is not registered