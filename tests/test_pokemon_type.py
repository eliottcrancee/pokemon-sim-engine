import pytest

from pokemon.pokemon_type import (
    _EFFECTIVENESS_CHART,
    _EFFECTIVENESS_CHART_3D,
    PokemonType,
)


@pytest.mark.parametrize(
    "attacker, defender, expected",
    [
        (PokemonType.WATER, PokemonType.FIRE, 2.0),  # Weakness
        (PokemonType.FIRE, PokemonType.WATER, 0.5),  # Resistance
        (PokemonType.NORMAL, PokemonType.NORMAL, 1.0),  # Neutral
        (PokemonType.GROUND, PokemonType.FLYING, 0.0),  # Immunity
        (PokemonType.GHOST, PokemonType.NORMAL, 0.0),  # Immunity
    ],
)
def test_single_type_effectiveness(attacker, defender, expected):
    """
    Tests 1v1 interactions.
    We pad the second defender type with NONETYPE to simulate a single-type Pokemon.
    """
    defenders = (defender, PokemonType.NONETYPE)
    assert attacker.effectiveness_against(defenders) == expected


@pytest.mark.parametrize(
    "attacker, def_primary, def_secondary, expected",
    [
        # Quad Weakness (2.0 * 2.0)
        (PokemonType.ICE, PokemonType.DRAGON, PokemonType.FLYING, 4.0),
        # Neutralization (2.0 * 0.5)
        # Ground hits Poison (2.0) but Bug resists Ground (0.5) in your config
        (PokemonType.GROUND, PokemonType.POISON, PokemonType.BUG, 1.0),
        # Immunity Override (0.0 * 2.0)
        # Electric vs Ground (0.0) / Water (2.0) -> Immunity wins
        (PokemonType.ELECTRIC, PokemonType.GROUND, PokemonType.WATER, 0.0),
        # Quad Resistance (0.5 * 0.5)
        # Grass vs Fire (0.5) / Flying (0.5) -> 0.25
        (PokemonType.GRASS, PokemonType.FIRE, PokemonType.FLYING, 0.25),
    ],
)
def test_dual_type_effectiveness(attacker, def_primary, def_secondary, expected):
    """
    Tests 1v2 interactions (Dual-type defenders).
    """
    defenders = (def_primary, def_secondary)
    assert attacker.effectiveness_against(defenders) == expected


def test_order_independence():
    """
    Ensure that the order of the defender's types does not change the result.
    (Fire/Flying should be the same as Flying/Fire).
    """
    attacker = PokemonType.ROCK
    # Rock vs Fire (2.0) / Flying (2.0) -> 4.0

    res1 = attacker.effectiveness_against((PokemonType.FIRE, PokemonType.FLYING))
    res2 = attacker.effectiveness_against((PokemonType.FLYING, PokemonType.FIRE))

    assert res1 == 4.0
    assert res1 == res2


def test_custom_config_logic():
    """
    Verifies a specific rule from your custom config:
    Your config defines BUG as immune to PSYCHIC (Standard games are just resistance).
    """
    attacker = PokemonType.PSYCHIC
    defender = PokemonType.BUG

    assert attacker.effectiveness_against((defender, PokemonType.NONETYPE)) == 0.0


def test_3d_matrix_integrity():
    """
    Mathematical proof: Ensures the 3D optimization matches
    the result of performing the 2D multiplications manually.
    """
    for atk in PokemonType:
        for d1 in PokemonType:
            for d2 in PokemonType:
                # Calculate manually using 2D lookup
                val_1 = _EFFECTIVENESS_CHART[atk][d1]
                val_2 = _EFFECTIVENESS_CHART[atk][d2]
                expected = val_1 * val_2

                # Retrieve from optimized 3D tensor
                actual = _EFFECTIVENESS_CHART_3D[atk][d1][d2]

                assert actual == expected
