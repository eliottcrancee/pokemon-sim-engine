# tests/test_action.py

import pytest
import torch

from pokemon.action import Action, ActionError, ActionType
from pokemon.item import ItemAccessor
from pokemon.move import MoveAccessor
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer


@pytest.fixture
def trainers():
    """Fixture to provide a trainer and an opponent for tests."""
    trainer = Trainer(
        name="Ash",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=10),
            PokemonAccessor.Chimchar(level=12),
        ],
        inventory={ItemAccessor.Potion(): ItemAccessor.Potion(default_quantity=2)},
    )
    opponent = Trainer(
        name="Gary",
        pokemon_team=[PokemonAccessor.Piplup(level=8)],
        inventory={},
    )
    return trainer, opponent


def test_attack_action(trainers):
    """Tests the creation and execution of an ATTACK action."""
    trainer, opponent = trainers
    move = trainer.pokemon_team[0].moves[0]
    action = Action(
        action_type=ActionType.ATTACK,
        trainer=trainer,
        opponent=opponent,
        move=move,
    )

    assert str(action) == f"Attack using {move.name}"

    initial_opponent_hp = opponent.pokemon_team[0].hp
    messages = action.execute()

    assert len(messages) > 0
    # Opponent should have taken damage (unless the move missed)
    assert opponent.pokemon_team[0].hp <= initial_opponent_hp


def test_switch_action(trainers):
    """Tests the creation and execution of a SWITCH action."""
    trainer, _ = trainers
    action = Action(action_type=ActionType.SWITCH, trainer=trainer, target_index=1)

    initial_active_pokemon = trainer.pokemon_team[0]
    assert str(action) == "Switch to Chimchar"

    messages = action.execute()

    assert trainer.pokemon_team[0].name == "chimchar"
    assert trainer.pokemon_team[1] == initial_active_pokemon
    assert len(messages) == 1  # Switch action itself is one message


def test_use_item_action(trainers):
    """Tests the creation and execution of an USE_ITEM action."""
    trainer, _ = trainers
    trainer.pokemon_team[1].hp -= 10  # Damage the pokemon to use a potion
    item = trainer.inventory[ItemAccessor.Potion()]

    action = Action(
        action_type=ActionType.USE_ITEM,
        trainer=trainer,
        target_index=1,
        item=item,
    )

    assert str(action) == "Use Potion on Chimchar"

    initial_target_hp = trainer.pokemon_team[1].hp
    initial_item_quantity = item.quantity

    messages = action.execute()

    assert len(messages) > 1  # Use item + effect message
    assert trainer.pokemon_team[1].hp > initial_target_hp
    assert item.quantity < initial_item_quantity


def test_invalid_action_creation(trainers):
    """Tests that creating invalid actions raises ActionError."""
    trainer, _ = trainers

    # Attack without opponent
    with pytest.raises(ActionError):
        Action(
            action_type=ActionType.ATTACK, trainer=trainer, move=MoveAccessor.Scratch
        )

    # Switch with invalid index
    with pytest.raises(ActionError):
        Action(action_type=ActionType.SWITCH, trainer=trainer, target_index=99)


def test_action_one_hot_encoding(trainers):
    """Tests the one-hot encoding for different action types."""
    trainer, opponent = trainers

    # Attack
    attack_action = Action(
        action_type=ActionType.ATTACK,
        trainer=trainer,
        opponent=opponent,
        move=trainer.pokemon_team[0].moves[0],
    )
    attack_one_hot = attack_action.one_hot
    assert isinstance(attack_one_hot, torch.Tensor)
    assert attack_one_hot[0] == 1  # Attack type
    assert attack_one_hot[3 + MoveAccessor.Scratch.move_id] == 1  # Move

    # Switch
    switch_action = Action(
        action_type=ActionType.SWITCH, trainer=trainer, target_index=1
    )
    switch_one_hot = switch_action.one_hot
    assert switch_one_hot[1] == 1  # Switch type

    # Use Item
    item_action = Action(
        action_type=ActionType.USE_ITEM,
        trainer=trainer,
        target_index=1,
        item=trainer.inventory[ItemAccessor.Potion()],
    )
    item_one_hot = item_action.one_hot
    assert item_one_hot[2] == 1  # Use Item type


# --- Performance Test ---


def test_performance_one_hot(benchmark, trainers):
    """Performance test for the one_hot property of actions."""
    trainer, opponent = trainers
    attack = Action(
        action_type=ActionType.ATTACK,
        trainer=trainer,
        opponent=opponent,
        move=trainer.pokemon_team[0].moves[0],
    )
    switch = Action(action_type=ActionType.SWITCH, trainer=trainer, target_index=1)
    use_item = Action(
        action_type=ActionType.USE_ITEM,
        trainer=trainer,
        target_index=1,
        item=trainer.inventory[ItemAccessor.Potion()],
    )

    def get_all_one_hots():
        _ = attack.one_hot
        _ = switch.one_hot
        _ = use_item.one_hot

    benchmark(get_all_one_hots)
