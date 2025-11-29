# tests/test_trainer.py

import pytest

from pokemon.item import Items, Potion
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.trainer import Trainer, TrainerError


@pytest.fixture
def sample_trainer() -> Trainer:
    """Fixture to create a sample trainer for testing."""
    return Trainer(
        name="Ash",
        pokemon_team=[
            Pokemon(species=Pokedex.Pikachu, level=10),
            Pokemon(species=Pokedex.Chimchar, level=12),
        ],
        inventory={Items.Potion: 2, Items.Revive: 1},
    )


def test_trainer_initialization(sample_trainer: Trainer):
    """Tests the basic initialization of a Trainer."""
    assert sample_trainer.name == "Ash"
    assert len(sample_trainer.pokemon_team) == 2
    assert sample_trainer.active_pokemon.species.name == "Pikachu"
    assert sample_trainer.get_item_quantity(Items.Potion) == 2
    assert sample_trainer.get_item_quantity(Items.Revive) == 1
    assert sample_trainer.get_item_quantity(Items.SuperPotion) == 0


def test_decrease_item_quantity(sample_trainer: Trainer):
    """Tests the item usage and quantity decrement."""
    initial_quantity = sample_trainer.get_item_quantity(Items.Potion)
    assert initial_quantity == 2

    # Use one potion
    sample_trainer.decrease_item_quantity(Items.Potion)
    assert sample_trainer.get_item_quantity(Items.Potion) == initial_quantity - 1

    # Use the second potion
    sample_trainer.decrease_item_quantity(Items.Potion)
    assert sample_trainer.get_item_quantity(Items.Potion) == 0

    # Test using an item that is out of stock
    with pytest.raises(TrainerError, match="No Potion left to use."):
        sample_trainer.decrease_item_quantity(Items.Potion)


def test_switch_pokemon(sample_trainer: Trainer):
    """Tests the pokemon switching logic using indices."""
    initial_active_pokemon = sample_trainer.active_pokemon
    chimchar = sample_trainer.pokemon_team[1]

    # Switch to the second Pokémon (index 1)
    sample_trainer.switch_pokemon(1)

    assert sample_trainer.active_pokemon == chimchar
    assert sample_trainer.pokemon_team[1] == initial_active_pokemon


def test_invalid_switch_pokemon(sample_trainer: Trainer):
    """Tests invalid pokemon switching scenarios."""
    # Try switching to an out-of-bounds index
    with pytest.raises(TrainerError, match="Invalid Pokémon index for switch"):
        sample_trainer.switch_pokemon(99)

    # Try switching to the active Pokémon (index 0)
    with pytest.raises(TrainerError, match="Invalid Pokémon index for switch"):
        sample_trainer.switch_pokemon(0)

    # Try switching to a fainted pokemon
    sample_trainer.pokemon_team[1].hp = 0
    with pytest.raises(TrainerError, match="Cannot switch to a fainted Pokémon."):
        sample_trainer.switch_pokemon(1)


def test_is_defeated(sample_trainer: Trainer):
    """Tests the is_defeated property."""
    assert not sample_trainer.is_defeated
    for p in sample_trainer.pokemon_team:
        p.hp = 0
    assert sample_trainer.is_defeated


def test_reset_methods(sample_trainer: Trainer):
    """Tests the reset, reset_team, and reset_inventory methods."""
    # Modify state
    sample_trainer.pokemon_team[0].hp -= 10
    sample_trainer.decrease_item_quantity(Items.Potion)

    # Check that state is modified
    assert sample_trainer.pokemon_team[0].hp < sample_trainer.pokemon_team[0].max_hp
    assert sample_trainer.get_item_quantity(Items.Potion) == 1

    # Test reset
    sample_trainer.reset()

    # Check that state is restored
    assert sample_trainer.pokemon_team[0].hp == sample_trainer.pokemon_team[0].max_hp
    # Inventory should be reset to initial state (2 Potions)
    assert sample_trainer.get_item_quantity(Items.Potion) == 2


def test_copy_is_deep(sample_trainer: Trainer):
    """Tests that the copy method performs a deep copy of mutable state."""
    p1 = sample_trainer
    p2 = p1.copy()

    # Assert they are different instances
    assert p1 is not p2
    assert p1.pokemon_team is not p2.pokemon_team
    assert p1._inventory_quantities is not p2._inventory_quantities

    # Modify the copy and check if the original is unaffected
    p2.pokemon_team[0].hp -= 10
    p2.decrease_item_quantity(Items.Potion)

    assert p1.pokemon_team[0].hp == p1.pokemon_team[0].max_hp
    assert p1.get_item_quantity(Items.Potion) == 2
