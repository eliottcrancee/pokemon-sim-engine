# tests/test_battle.py

import pytest
import torch

from envs.pokemon.action import ActionType
from envs.pokemon.battle import Battle
from envs.pokemon.pokemon import PokemonAccessor
from envs.pokemon.trainer import Trainer


@pytest.fixture
def battle_instance():
    """Fixture to create a standard battle instance for testing."""
    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Chimchar(level=10),
        ],
    )
    trainer2 = Trainer(name="Gary", pokemon_team=[PokemonAccessor.Piplup(level=11)])
    return Battle(trainer1, trainer2)


def test_battle_initialization(battle_instance):
    """Tests that the battle is initialized correctly."""
    assert battle_instance.round == 0
    assert not battle_instance.done
    assert battle_instance.winner is None
    assert battle_instance.trainer_0.name == "Ash"
    assert battle_instance.trainer_1.name == "Gary"


def test_get_possible_actions(battle_instance):
    """Tests the generation of possible actions for a trainer."""
    actions_0 = battle_instance.get_possible_actions(0)

    # Trainer 0 (Ash) has Pikachu (2 moves) and Chimchar (1 move).
    # Expected actions: 2 attacks for Pikachu + 1 switch to Chimchar.
    action_types = [action.action_type for action in actions_0]

    assert ActionType.ATTACK in action_types
    assert ActionType.SWITCH in action_types

    # Count attack actions
    attack_actions = [a for a in actions_0 if a.action_type == ActionType.ATTACK]
    assert len(attack_actions) == len(battle_instance.trainer_0.pokemon_team[0].moves)

    # Count switch actions
    switch_actions = [a for a in actions_0 if a.action_type == ActionType.SWITCH]
    assert len(switch_actions) == 1  # Only Chimchar is available to switch


def test_battle_turn(battle_instance):
    """Tests a single turn of a battle."""
    actions_0 = battle_instance.get_possible_actions(0)
    actions_1 = battle_instance.get_possible_actions(1)

    # Let's take the first attack action for both
    attack_action_0 = next(a for a in actions_0 if a.action_type == ActionType.ATTACK)
    attack_action_1 = next(a for a in actions_1 if a.action_type == ActionType.ATTACK)

    initial_hp_0 = battle_instance.trainer_0.pokemon_team[0].hp
    initial_hp_1 = battle_instance.trainer_1.pokemon_team[0].hp

    messages = battle_instance.turn(attack_action_0, attack_action_1)

    assert battle_instance.round == 1
    # Check if at least one pokemon took damage (unless both missed)
    pokemon_0_hp = battle_instance.trainer_0.pokemon_team[0].hp
    pokemon_1_hp = battle_instance.trainer_1.pokemon_team[0].hp
    assert pokemon_0_hp < initial_hp_0 or pokemon_1_hp < initial_hp_1
    assert len(messages) > 0


def test_battle_end_condition(battle_instance):
    """Tests the end condition of a battle."""
    # Manually set a pokemon's HP to 0 to simulate a faint
    battle_instance.trainer_1.pokemon_team[0].hp = 0
    assert battle_instance.trainer_1.is_defeated

    # The end() method should now return True and set a winner
    assert battle_instance.end() is True
    assert battle_instance.winner == 0  # Trainer 0 should be the winner


def test_battle_tie_condition(battle_instance):
    """Tests the tie condition of a battle."""
    battle_instance.round = battle_instance.max_rounds
    assert battle_instance.end() is True
    assert battle_instance.tie is True
    assert battle_instance.winner is None


def test_battle_tensor_creation(battle_instance):
    """Tests the creation of the battle state tensor."""
    tensor_0 = battle_instance.tensor(0)
    tensor_1 = battle_instance.tensor(1)

    assert isinstance(tensor_0, torch.Tensor)
    assert isinstance(tensor_1, torch.Tensor)
    assert tensor_0.shape == tensor_1.shape

    # The tensors should be different as they represent different perspectives
    assert not torch.equal(tensor_0, tensor_1)

    # Check description length
    description = battle_instance.tensor_description
    assert len(tensor_0) == len(description)


# --- Performance Test ---


def test_performance_tensor_creation(benchmark, battle_instance):
    """Performance test for the tensor creation."""
    benchmark(battle_instance.tensor, trainer_id=0)
