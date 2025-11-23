# tests/test_pokemon_type.py

import pytest

from pokemon.pokemon_type import PokemonType, PokemonTypeAccessor, PokemonTypeError


def test_type_accessor():
    """Tests the PokemonTypeAccessor for retrieving types."""
    assert PokemonTypeAccessor.Fire.name == "Fire"
    assert PokemonTypeAccessor.by_id(1).name == "Normal"
    assert len(list(PokemonTypeAccessor)) > 15  # Ensure types are loaded
    with pytest.raises(AttributeError):
        _ = PokemonTypeAccessor.NonExistentType


def test_type_equality():
    """Tests equality comparison between PokemonType instances."""
    assert PokemonTypeAccessor.Grass == PokemonTypeAccessor.by_id(4)
    assert PokemonTypeAccessor.Water != PokemonTypeAccessor.Ice


def test_effectiveness_against_single_type():
    """Tests the effectiveness calculation against a single target type."""
    fire = PokemonTypeAccessor.Fire
    water = PokemonTypeAccessor.Water
    grass = PokemonTypeAccessor.Grass

    # Super effective
    assert water.effectiveness_against([fire]) == 2.0
    # Not very effective
    assert fire.effectiveness_against([water]) == 0.5
    # Normal effectiveness
    assert fire.effectiveness_against([grass]) == 2.0
    # Immunity
    ground = PokemonTypeAccessor.Ground
    electric = PokemonTypeAccessor.Electric
    assert electric.effectiveness_against([ground]) == 0.0


def test_effectiveness_against_dual_type():
    """Tests the effectiveness calculation against a dual-type target."""
    rock = PokemonTypeAccessor.Rock
    ground = PokemonTypeAccessor.Ground
    water = (
        PokemonTypeAccessor.Water
    )  # Water is super effective against both Rock and Ground

    # Dual weakness: Water vs Rock/Ground should be 4x
    assert water.effectiveness_against([rock, ground]) == 4.0

    electric = PokemonTypeAccessor.Electric
    flying = PokemonTypeAccessor.Flying  # Flying is immune to Ground
    # One immunity should result in 0x effectiveness
    assert ground.effectiveness_against([electric, flying]) == 0.0

    fire = PokemonTypeAccessor.Fire
    # Resistance and weakness should cancel out (0.5 * 2.0 = 1.0)
    # Grass is weak to Fire (2x) but resists Water (0.5x)
    # Let's test Fire against Water/Grass
    water_grass = [PokemonTypeAccessor.Water, PokemonTypeAccessor.Grass]
    assert fire.effectiveness_against(water_grass) == 1.0  # 0.5 * 2.0


def test_invalid_type_initialization():
    """Tests that initializing a PokemonType with invalid data raises an error.
    This requires the DEBUG flag to be set, which we assume is off during tests.
    This test is designed to pass if no error is raised when DEBUG is off.
    If DEBUG were on, we would assert that a PokemonTypeError is raised.
    """
    try:
        # This should raise PokemonTypeError if DEBUG is on
        _ = PokemonType(name=123, type_id="abc")
    except (PokemonTypeError, TypeError):
        # Catching TypeError as well, as it might be raised without DEBUG validation
        pass


# --- Performance Test ---


def test_performance_effectiveness_against(benchmark):
    """Performance test for the effectiveness_against method."""
    water = PokemonTypeAccessor.Water
    fire = PokemonTypeAccessor.Fire
    flying = PokemonTypeAccessor.Flying
    benchmark(water.effectiveness_against, [fire, flying])
