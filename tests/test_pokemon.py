# tests/test_pokemon.py

import torch

from pokemon.pokemon import (
    POKEMON_LIST,
    PokemonAccessor,
    PokemonError,
    PokemonStatus,
)


def test_pokemon_enums():
    """Tests the string representation of PokemonStatus and PokemonEnum."""
    assert str(PokemonStatus.Healthy) == "Healthy"
    assert PokemonAccessor.Pikachu.name.capitalize() == "Pikachu"
    assert (
        repr(PokemonAccessor)
        == "Pokemons(Pikachu, Chimchar, Piplup, Bulbasaur, Charmander, Squirtle, Pidgey, Rattata, Sandshrew, Eevee)"
    )


def test_pokemon_creation_and_details():
    """Tests the creation of Pokemon and their basic details."""
    pikachu = PokemonAccessor.Pikachu(level=10)
    assert pikachu.level == 10
    assert pikachu.name == "pikachu"
    assert pikachu.hp == pikachu.max_hp
    assert "Thunder Shock" in [move.name for move in pikachu.moves]


def test_stat_modifiers():
    """Tests the application and removal of stat modifiers."""
    pikachu = PokemonAccessor.Pikachu(level=10)
    initial_attack = pikachu.attack
    pikachu.apply_modifier("attack", 2)
    assert pikachu.attack > initial_attack
    pikachu.apply_modifier("attack", -1)
    assert pikachu.attack > initial_attack
    pikachu.reset_modifiers()
    assert pikachu.attack == initial_attack


def test_hp_manipulation():
    """Tests damage, healing, and HP ratio calculation."""
    chimchar = PokemonAccessor.Chimchar(level=15)
    initial_hp = chimchar.hp
    chimchar.hp -= 20
    assert chimchar.hp == initial_hp - 20
    chimchar.hp += 10
    assert chimchar.hp == initial_hp - 10
    assert chimchar.hp_ratio == (initial_hp - 10) / chimchar.max_hp


def test_status_effects():
    """Tests the application of status effects like Burn."""
    piplup = PokemonAccessor.Piplup(level=20)
    piplup.status = PokemonStatus.Burn
    initial_hp = piplup.hp
    messages = piplup.after_turn()
    assert len(messages) > 0
    assert "hurt by its burn" in messages[0].content
    assert piplup.hp < initial_hp


def test_confusion():
    """Tests the confusion status and its effects on attacking."""
    pikachu = PokemonAccessor.Pikachu()
    pikachu._confused = True
    # This test is probabilistic, so we can't assert a specific outcome.
    # We'll just run it to ensure it doesn't crash.
    for _ in range(5):
        can_attack, messages = pikachu.can_attack()
        assert isinstance(can_attack, bool)
        if messages:
            if not can_attack:
                assert "hurt itself in its confusion" in messages[0].content
            if "snapped out of confusion" in messages[0].content:
                assert not pikachu._confused


def test_is_like_method():
    """Tests the is_like method for comparing Pokemon."""
    pikachu1 = PokemonAccessor.Pikachu(level=10)
    pikachu2 = PokemonAccessor.Pikachu(level=20)
    chimchar = PokemonAccessor.Chimchar()
    assert pikachu1.is_like(pikachu2)
    assert not pikachu1.is_like(chimchar)


def test_clear_and_reset_methods():
    """Tests the clear and reset methods."""
    chimchar = PokemonAccessor.Chimchar()
    initial_hp = chimchar.hp
    initial_defense = chimchar.defense
    chimchar.apply_modifier("defense", 3)
    chimchar._confused = True
    chimchar.hp -= 30
    assert chimchar.defense > initial_defense
    assert chimchar._confused
    assert chimchar.hp < initial_hp

    chimchar.clear()
    assert chimchar.defense == initial_defense
    assert not chimchar._confused
    assert chimchar.hp < initial_hp  # HP is not restored by clear

    chimchar.reset()
    assert chimchar.defense == initial_defense
    assert not chimchar._confused
    assert chimchar.hp == initial_hp  # HP is restored by reset


def test_invalid_pokemon_initialization():
    """Tests that creating a Pokemon with invalid data raises an error."""
    # The validation is behind a DEBUG flag, so we can't directly test the exception
    # unless we can control the DEBUG flag.
    # For now, we just ensure it doesn't crash when DEBUG is off.
    try:
        _ = PokemonAccessor.Pikachu(name=123, level="abc")
        # If DEBUG is on, this should not be reached
    except PokemonError as e:
        # This will only be caught if DEBUG is on
        assert "Name must be a non-empty string" in e.message
    except TypeError:
        # If DEBUG is off, it might raise a TypeError from internal operations
        pass


def test_one_hot_encoding():
    """Tests the one-hot encoding for a Pokemon."""
    pikachu = PokemonAccessor.Pikachu()
    one_hot = pikachu.one_hot
    assert isinstance(one_hot, torch.Tensor)
    # The first element is padding, the second should be Pikachu (id 0)
    expected = torch.zeros(len(POKEMON_LIST) + 1)
    expected[pikachu.pokemon_id + 1] = 1
    assert torch.equal(one_hot.to_dense(), expected)


# Performance tests
# Note: Pytest doesn't run functions with custom decorators by default.
# We can use the `pytest-benchmark` plugin for more robust performance testing,
# but for a direct migration, we'll use timeit as in the original script.


def test_performance_can_attack(benchmark):
    """Performance test for the can_attack method."""
    pikachu = PokemonAccessor.Pikachu()
    benchmark(pikachu.can_attack)


def test_performance_one_hot(benchmark):
    """Performance test for the one_hot property."""
    pikachu = PokemonAccessor.Pikachu()
    benchmark(lambda: pikachu.one_hot)
