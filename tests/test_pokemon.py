from unittest.mock import patch

import pytest

from pokemon.move import MoveCategory, MoveRegistry, PokemonType, SimpleMove
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
            SimpleMove(
                "Thunder Shock",
                MoveCategory.SPECIAL,
                PokemonType.ELECTRIC,
                40,
                100,
                30,
            )
        )
    if not MoveRegistry.get("Growl"):
        MoveRegistry.register(
            SimpleMove("Growl", MoveCategory.STATUS, PokemonType.NORMAL, 0, 100, 40)
        )
    if not MoveRegistry.get("Quick Attack"):
        MoveRegistry.register(
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
    if not MoveRegistry.get("Tail Whip"):
        MoveRegistry.register(
            SimpleMove("Tail Whip", MoveCategory.STATUS, PokemonType.NORMAL, 0, 100, 30)
        )


@pytest.fixture
def shinx() -> Pokemon:
    """Provides a default level 5 Shinx for tests."""
    return Pokemon(species=Pokedex.Shinx, level=5)


def test_basic_creation(shinx: Pokemon):
    assert shinx.species.name == "Shinx"
    assert shinx.surname == "Shinx"
    assert shinx.level == 5
    assert shinx.hp == shinx.max_hp
    assert shinx.is_alive is True
    assert len(shinx.move_slots) > 0
    assert shinx.move_slots[0].move.name == "Tackle"


def test_custom_attributes():
    char = Pokemon(
        species=Pokedex.Chimchar,
        level=10,
        surname="Charry",
        ivs=(31, 31, 31, 31, 31, 31),
    )
    assert char.level == 10
    assert char.surname == "Charry"
    assert char._ivs == (31, 31, 31, 31, 31, 31)


def test_stat_calculation():
    """Verify stats against a known calculator for a level 50 Turtwig."""
    # Base stats: HP=55, Atk=68, Def=64, SpA=45, SpD=55, Spe=31
    # IVs: (31, 31, 31, 31, 31, 31), EVs: (0, 0, 0, 0, 0, 0)
    # Formula: ((2*base + iv + ev/4) * level / 100) + 5 for non-HP
    # HP Formula: ((2*base + iv + ev/4) * level / 100) + level + 10
    turtwig = Pokemon(species=Pokedex.Turtwig, level=50, ivs=(31, 31, 31, 31, 31, 31))
    # HP: ((2*55 + 31) * 50 / 100) + 50 + 10 = (141 * 50 / 100) + 60 = 70 + 60 = 130
    assert turtwig.max_hp == 130
    # Atk: ((2*68 + 31) * 50 / 100) + 5 = (167 * 50 / 100) + 5 = 83 + 5 = 88
    assert turtwig.attack == 88
    # Def: ((2*64 + 31) * 50 / 100) + 5 = (159 * 50 / 100) + 5 = 79 + 5 = 84
    assert turtwig.defense == 84
    # SpA: ((2*45 + 31) * 50 / 100) + 5 = (121 * 50 / 100) + 5 = 60 + 5 = 65
    assert turtwig.sp_attack == 65
    # SpD: ((2*55 + 31) * 50 / 100) + 5 = (141 * 50 / 100) + 5 = 70 + 5 = 75
    assert turtwig.sp_defense == 75
    # Spe: ((2*31 + 31) * 50 / 100) + 5 = (93 * 50 / 100) + 5 = 46 + 5 = 51
    assert turtwig.speed == 51


def test_hp_setter_and_fainting(shinx: Pokemon):
    shinx.hp -= 10
    assert shinx.hp == shinx.max_hp - 10
    assert shinx.is_alive is True

    shinx.hp = -5  # Should clamp to 0
    assert shinx.hp == 0
    assert shinx.is_alive is False
    assert shinx.status == PokemonStatus.FAINTED

    shinx.hp = 1000  # Should clamp to max_hp
    assert shinx.hp == shinx.max_hp
    assert shinx.is_alive is True  # Fainting is reversible


def test_restore_and_reset(shinx: Pokemon):
    shinx.hp -= 5
    shinx.status = PokemonStatus.BURN
    shinx.apply_modifier(0, -2)  # -2 Attack
    shinx.decrease_pp(0)

    assert shinx.hp < shinx.max_hp
    assert shinx.attack_stage == -2
    assert shinx.move_slots[0].current_pp < shinx.move_slots[0].max_pp

    shinx.full_restore()
    assert shinx.hp == shinx.max_hp
    assert shinx.status == PokemonStatus.HEALTHY
    assert shinx.attack_stage == 0
    # full_restore does not reset PP
    assert shinx.move_slots[0].current_pp < shinx.move_slots[0].max_pp

    shinx.reset()
    assert shinx.move_slots[0].current_pp == shinx.move_slots[0].max_pp


def test_copy(shinx: Pokemon):
    p1 = shinx
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
def test_apply_modifier(shinx: Pokemon, stat_idx, prop_name):
    raw_stat = shinx._raw_stats[stat_idx]

    # Increase stage
    changed = shinx.apply_modifier(stat_idx, 2)
    assert changed is True
    assert shinx._stat_stages[stat_idx] == 2
    expected = int(raw_stat * _STAT_STAGE_MULTIPLIERS[2 + 6])
    assert getattr(shinx, prop_name) == expected

    # Decrease stage
    changed = shinx.apply_modifier(stat_idx, -3)
    assert changed is True
    assert shinx._stat_stages[stat_idx] == -1
    expected = int(raw_stat * _STAT_STAGE_MULTIPLIERS[-1 + 6])
    assert getattr(shinx, prop_name) == expected


def test_modifier_limits(shinx: Pokemon):
    # Test upper limit
    shinx.apply_modifier(0, 6)
    assert shinx.attack_stage == 6
    changed = shinx.apply_modifier(0, 1)
    assert changed is False
    assert shinx.attack_stage == 6

    # Test lower limit
    shinx.apply_modifier(1, -6)
    assert shinx.defense_stage == -6
    changed = shinx.apply_modifier(1, -1)
    assert changed is False
    assert shinx.defense_stage == -6


def test_reset_modifiers(shinx: Pokemon):
    initial_attack = shinx.attack
    shinx.apply_modifier(0, 4)
    assert shinx.attack != initial_attack

    shinx.reset_modifiers()
    assert shinx.attack_stage == 0
    assert shinx.attack == initial_attack


@patch("random.random", return_value=0.1)  # Fails paralysis/freeze check
def test_can_attack_paralyzed_fail(mock_random, shinx: Pokemon):
    shinx.status = PokemonStatus.PARALYSIS
    can_attack, msgs = shinx.can_attack()
    assert can_attack is False
    assert "paralyzed" in msgs[0].text


@patch("random.random", return_value=0.9)  # Passes paralysis/freeze check
def test_can_attack_paralyzed_ok(mock_random, shinx: Pokemon):
    shinx.status = PokemonStatus.PARALYSIS
    can_attack, msgs = shinx.can_attack()
    assert can_attack is True
    assert len(msgs) == 0


@patch("random.random", return_value=0.1)  # Thaws out
def test_can_attack_frozen_thaw(mock_random, shinx: Pokemon):
    shinx.status = PokemonStatus.FREEZE
    can_attack, msgs = shinx.can_attack()
    assert can_attack is True
    assert "thawed out" in msgs[0].text
    assert shinx.status == PokemonStatus.HEALTHY


@patch("random.random", return_value=0.4)  # Hurts itself
def test_can_attack_confused_fail(mock_random, shinx: Pokemon):
    shinx.confused = True
    shinx._confused_turns = 3
    initial_hp = shinx.hp
    can_attack, msgs = shinx.can_attack()
    assert can_attack is False
    assert "confused" in msgs[0].text
    assert "hurt itself" in msgs[1].text
    assert shinx.hp < initial_hp


@patch("random.random", return_value=0.9)  # Attacks normally
def test_can_attack_confused_ok(mock_random, shinx: Pokemon):
    shinx.confused = True
    shinx._confused_turns = 3
    initial_hp = shinx.hp
    can_attack, msgs = shinx.can_attack()
    assert can_attack is True
    assert "confused" in msgs[0].text
    assert shinx.hp == initial_hp


def test_after_turn_burn_damage(shinx: Pokemon):
    shinx.status = PokemonStatus.BURN
    initial_hp = shinx.hp
    expected_dmg = max(1, shinx.max_hp >> 3)
    msgs = shinx.after_turn()
    assert shinx.hp == initial_hp - expected_dmg
    assert "hurt by its burn" in msgs[0].text


def test_after_turn_poison_damage(shinx: Pokemon):
    shinx.status = PokemonStatus.POISON
    initial_hp = shinx.hp
    expected_dmg = max(1, shinx.max_hp >> 3)
    msgs = shinx.after_turn()
    assert shinx.hp == initial_hp - expected_dmg
    assert "hurt by poison" in msgs[0].text


def test_species_registry_get():
    shinx_by_name = SpeciesRegistry.get("Shinx")
    assert shinx_by_name is not None
    assert shinx_by_name.name == "Shinx"
    shinx_by_id = SpeciesRegistry.get(shinx_by_name.id)
    assert shinx_by_name is shinx_by_id


def test_pokedex_getattr():
    assert Pokedex.Shinx.name == "Shinx"
    assert Pokedex.Chimchar.types == (PokemonType.FIRE,)


def test_pokedex_not_found():
    with pytest.raises(AttributeError):
        _ = Pokedex.Mewtwo  # Assuming Mewtwo is not registered
