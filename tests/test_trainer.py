# tests/test_trainer.py

import pytest
import torch

from envs.pokemon.item import ItemAccessor
from envs.pokemon.pokemon import PokemonAccessor
from envs.pokemon.trainer import MAX_POKEMON_PER_TRAINER, Trainer, TrainerError


@pytest.fixture
def sample_trainer():
    """Fixture to create a sample trainer for testing."""
    pikachu = PokemonAccessor.Pikachu(level=10)
    chimchar = PokemonAccessor.Chimchar(level=12)
    potion = ItemAccessor.Potion(default_quantity=2)
    return Trainer(
        name="Ash",
        pokemon_team=[pikachu, chimchar],
        inventory={ItemAccessor.Potion: potion},
    )


def test_trainer_initialization(sample_trainer):
    """Tests the basic initialization of a Trainer."""
    assert sample_trainer.name == "Ash"
    assert len(sample_trainer.pokemon_team) == 2
    assert sample_trainer.pokemon_team[0].name == "pikachu"
    assert ItemAccessor.Potion in sample_trainer.inventory
    assert sample_trainer.inventory[ItemAccessor.Potion].quantity == 2


def test_switch_pokemon(sample_trainer):
    """Tests the pokemon switching logic."""
    initial_active_pokemon = sample_trainer.pokemon_team[0]
    chimchar = sample_trainer.pokemon_team[1]

    sample_trainer.switch_pokemon(chimchar)

    assert sample_trainer.pokemon_team[0] == chimchar
    assert sample_trainer.pokemon_team[1] == initial_active_pokemon


def test_invalid_switch_pokemon(sample_trainer):
    """Tests invalid pokemon switching scenarios."""
    piplup = PokemonAccessor.Piplup()  # Not in team

    # Try switching to a pokemon not in the team
    with pytest.raises(TrainerError):
        sample_trainer.switch_pokemon(piplup)

    # Try switching to the same pokemon
    with pytest.raises(TrainerError):
        sample_trainer.switch_pokemon(sample_trainer.pokemon_team[0])

    # Try switching to a fainted pokemon
    sample_trainer.pokemon_team[1].hp = 0
    with pytest.raises(TrainerError):
        sample_trainer.switch_pokemon(sample_trainer.pokemon_team[1])


def test_is_defeated(sample_trainer):
    """Tests the is_defeated property."""
    assert not sample_trainer.is_defeated
    for pokemon in sample_trainer.pokemon_team:
        pokemon.hp = 0
    assert sample_trainer.is_defeated


def test_team_reset_and_clear(sample_trainer):
    """Tests the reset_team and clear_team methods."""
    pokemon = sample_trainer.pokemon_team[0]
    initial_hp = pokemon.hp

    pokemon.hp -= 10
    pokemon._confused = True
    assert pokemon.hp < initial_hp
    assert pokemon._confused

    sample_trainer.clear_team()
    assert not pokemon._confused  # clear should reset status

    sample_trainer.reset_team()
    assert pokemon.hp == initial_hp  # reset should restore hp


def test_invalid_trainer_initialization():
    """Tests that creating a Trainer with invalid data raises an error.
    This assumes DEBUG is on for validation to occur.
    """
    # Test with too many pokemon
    with pytest.raises(TrainerError):
        too_many_pokemon = [
            PokemonAccessor.Pikachu() for _ in range(MAX_POKEMON_PER_TRAINER + 1)
        ]
        Trainer(name="Gary", pokemon_team=too_many_pokemon)

    # We can't test for name, team, or inventory type errors without controlling
    # the DEBUG flag, but we can test for constraints that are always checked.


def test_trainer_tensor(sample_trainer):
    """Tests the creation of the trainer's tensor representation."""
    tensor = sample_trainer.tensor
    description = sample_trainer.tensor_description

    assert isinstance(tensor, torch.Tensor)
    assert len(tensor) == len(description)

    # Check a few values
    # First pokemon is pikachu, so its one-hot should be at the start
    assert tensor[PokemonAccessor.Pikachu.pokemon_id + 1] == 1
    # HP ratio of the first pokemon
    hp_ratio_index = description.index("Pokemon 0 HP Ratio")
    assert tensor[hp_ratio_index] == sample_trainer.pokemon_team[0].hp_ratio


# --- Performance Test ---


def test_performance_tensor_creation(benchmark, sample_trainer):
    """Performance test for the tensor creation."""
    benchmark(lambda: sample_trainer.tensor)
