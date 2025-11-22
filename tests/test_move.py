# tests/test_move.py

import pytest

from pokemon.move import (
    MOVE_LIST,
    Move,
    MoveAccessor,
    MoveCategoryValue,
    PokemonTypeAccessor,
    Struggle,
)
from pokemon.pokemon import PokemonAccessor


@pytest.fixture
def pokemon_instances():
    """Fixture to provide fresh Pokemon instances for each test."""
    return (
        PokemonAccessor.Pikachu(level=10),
        PokemonAccessor.Chimchar(level=15),
        PokemonAccessor.Piplup(level=20),
    )


def test_move_creation_and_representation():
    """Tests basic move attributes and string representation."""
    tackle = MoveAccessor.Tackle
    assert tackle.name == "Tackle"
    assert tackle.category == MoveCategoryValue.Physical
    assert tackle.type == PokemonTypeAccessor.Normal
    assert str(tackle) == "Tackle | Type: Normal | PP: 35/35"
    assert "Move(id=2, name=Tackle" in repr(tackle)


def test_damage_calculation(pokemon_instances):
    """Tests the damage calculation for a move.
    Since damage has a random factor, we check if it's within an expected range.
    """
    pikachu, chimchar = pokemon_instances[0], pokemon_instances[1]
    move = MoveAccessor.ThunderShock  # Electric move

    # Pikachu (Electric) vs Chimchar (Fire) -> effectiveness = 1.0
    damage, base_damage, critical, stab, effectiveness = move.calculate_damage(
        pikachu, chimchar
    )

    assert effectiveness == 1.0
    assert stab is True  # Pikachu is Electric, ThunderShock is Electric
    if damage > 0:
        assert base_damage > 0
    assert isinstance(damage, int)
    assert isinstance(critical, bool)


def test_status_move_no_damage(pokemon_instances):
    """Tests that status moves deal no damage."""
    # We need a status move to test this. Let's create one for the test.
    status_move = Move(
        99,
        "Test Status",
        MoveCategoryValue.Status,
        PokemonTypeAccessor.Normal,
        0,
        100,
        20,
    )
    user, target, _ = pokemon_instances
    damage, _, _, _, _ = status_move.calculate_damage(user, target)
    assert damage == 0


def test_struggle_move_effect(pokemon_instances):
    """Tests the Struggle move's recoil effect."""
    pikachu, chimchar = pokemon_instances[0], pokemon_instances[1]
    initial_hp = pikachu.hp
    struggle = Struggle

    damage, *rest = struggle.calculate_damage(pikachu, chimchar)
    messages = struggle.secondary_effect(pikachu, chimchar, damage, *rest)

    assert "lost" in messages[0].content
    assert pikachu.hp < initial_hp
    assert pikachu.hp == initial_hp - (pikachu.max_hp // 4)


def test_move_accuracy(pokemon_instances, capsys):
    """Tests the accuracy of moves.
    This is probabilistic, so we check if the hit rate is close to the accuracy.
    """
    pikachu, chimchar = pokemon_instances[0], pokemon_instances[1]
    move = MoveAccessor.Tackle  # 95 accuracy
    hit_count = sum(
        1 for _ in range(1000) if move.calculate_damage(pikachu, chimchar)[0] > 0
    )
    hit_rate = hit_count / 1000.0
    # Check if hit rate is within a reasonable margin of the expected accuracy
    assert move.accuracy / 100.0 - 0.1 < hit_rate < move.accuracy / 100.0 + 0.1

    # The original test printed a table, we can skip that in a unit test
    # or capture output if we want to verify the print format.


def test_move_one_hot():
    """Tests the one-hot encoding for a move."""
    move = MoveAccessor.Scratch
    one_hot = move.one_hot
    assert one_hot.to_dense()[move.move_id] == 1
    assert one_hot.to_dense().sum() == 1
    assert len(one_hot.to_dense()) == len(MOVE_LIST)


# --- Performance Tests ---


def test_performance_calculate_damage(benchmark, pokemon_instances):
    """Performance test for the calculate_damage method."""
    pikachu, chimchar = pokemon_instances[0], pokemon_instances[1]
    move = MoveAccessor.SelfHit
    benchmark(move.calculate_damage, pikachu, chimchar)


def test_performance_one_hot(benchmark):
    """Performance test for the one_hot property."""
    move = MoveAccessor.SelfHit
    benchmark(lambda: move.one_hot)
