# tests/test_battle.py

import pytest

from pokemon.action import Action, ActionType
from pokemon.battle import Battle
from pokemon.message import Message
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.trainer import Trainer


@pytest.fixture
def battle_instance():
    """Fixture to create a standard battle instance for testing."""
    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            Pokemon(species=Pokedex.Pikachu, level=12),
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Gary", pokemon_team=[Pokemon(species=Pokedex.Piplup, level=11)]
    )
    return Battle(trainers=(trainer1, trainer2), headless=False)


@pytest.fixture
def faint_scenario_battle():
    """Fixture to create a battle where a pokemon can faint easily."""
    pikachu = Pokemon(species=Pokedex.Pikachu, level=10)
    pikachu.hp = 1  # Manually set HP after creation

    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            pikachu,
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Gary",
        pokemon_team=[
            Pokemon(species=Pokedex.Piplup, level=10),
        ],
    )
    battle = Battle(trainers=(trainer1, trainer2), headless=False)
    battle.trainers[0].active_pokemon.hp = 1  # Set Pikachu's HP to 1 after reset
    # Ensure Piplup has an attack that can one-shot Pikachu
    piplup = battle.trainers[1].active_pokemon
    piplup.move_slots[0].move.power = 100  # Make the first move very powerful
    return battle


@pytest.fixture
def struggle_scenario_battle():
    """Fixture to create a battle where a pokemon is forced to Struggle."""
    # Create a Pikachu with all PP reduced to 0
    pikachu_struggle = Pokemon(species=Pokedex.Pikachu, level=10)
    for slot in pikachu_struggle.move_slots:
        slot.current_pp = 0

    trainer1 = Trainer(
        name="Ash",
        pokemon_team=[
            pikachu_struggle,
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Gary",
        pokemon_team=[
            Pokemon(species=Pokedex.Piplup, level=10),
        ],
    )
    battle = Battle(trainers=(trainer1, trainer2), headless=False)
    for slot in battle.trainers[0].active_pokemon.move_slots: # Set PP to 0 after battle reset
        slot.current_pp = 0
    return battle


def test_battle_initialization(battle_instance):
    """Tests that the battle is initialized correctly."""
    assert battle_instance.round == 0
    assert not battle_instance.done
    assert battle_instance.winner is None
    assert battle_instance.trainers[0].name == "Ash"
    assert battle_instance.trainers[1].name == "Gary"


def test_get_possible_actions(battle_instance):
    """Tests the generation of possible actions for a trainer."""
    actions_0 = battle_instance.get_possible_actions(0)

    action_types = [action.action_type for action in actions_0]

    assert ActionType.ATTACK in action_types
    assert ActionType.SWITCH in action_types

    attack_actions = [a for a in actions_0 if a.action_type == ActionType.ATTACK]
    assert len(attack_actions) == len(
        battle_instance.trainers[0].pokemon_team[0].move_slots
    )

    switch_actions = [a for a in actions_0 if a.action_type == ActionType.SWITCH]
    assert len(switch_actions) == 1


def test_battle_turn(battle_instance):
    """Tests a single turn of a battle."""
    actions_0 = battle_instance.get_possible_actions(0)
    actions_1 = battle_instance.get_possible_actions(1)

    attack_action_0 = next(a for a in actions_0 if a.action_type == ActionType.ATTACK)
    attack_action_1 = next(a for a in actions_1 if a.action_type == ActionType.ATTACK)

    initial_hp_0 = battle_instance.trainers[0].pokemon_team[0].hp
    initial_hp_1 = battle_instance.trainers[1].pokemon_team[0].hp

    messages = battle_instance.turn(attack_action_0, attack_action_1)

    assert battle_instance.round == 1
    pokemon_0_hp = battle_instance.trainers[0].pokemon_team[0].hp
    pokemon_1_hp = battle_instance.trainers[1].pokemon_team[0].hp
    assert pokemon_0_hp < initial_hp_0 or pokemon_1_hp < initial_hp_1
    assert len(messages) > 0


def test_faint_and_switch_logic(faint_scenario_battle):
    """Tests the new faint and switch logic."""
    battle = faint_scenario_battle

    # Trainer 1 (Gary's Piplup) attacks Trainer 0's (Ash's Pikachu)
    # Pikachu has 1 HP, so it will faint.
    actions_t1 = battle.get_possible_actions(1)
    attack_t1 = next(a for a in actions_t1 if a.action_type == ActionType.ATTACK)

    # For Trainer 0, let's just make Pikachu try to attack (it will faint)
    actions_t0_initial = battle.get_possible_actions(0)
    attack_t0 = next(a for a in actions_t0_initial if a.action_type == ActionType.ATTACK)

    # Execute the turn where Pikachu faints
    messages = battle.turn(attack_t0, attack_t1)

    assert any("fainted" in m.text for m in messages)
    assert not battle.trainers[0].active_pokemon.is_alive
    assert battle.round == 1
    assert not battle.done  # Battle is not done, Ash still has Chimchar

    # Now, check possible actions for the next state
    # Trainer 0 (Ash) should only be able to switch
    actions_t0_after_faint = battle.get_possible_actions(0)
    assert len(actions_t0_after_faint) == 1  # Only one switch action for Chimchar
    assert actions_t0_after_faint[0].action_type == ActionType.SWITCH
    assert actions_t0_after_faint[0].pokemon_index == 1  # Switch to Chimchar

    # Trainer 1 (Gary) should only be able to pass
    actions_t1_after_faint = battle.get_possible_actions(1)
    assert len(actions_t1_after_faint) == 1
    assert actions_t1_after_faint[0].action_type == ActionType.PASS

    # Simulate the switch turn
    switch_action_t0 = actions_t0_after_faint[0]
    pass_action_t1 = actions_t1_after_faint[0]

    messages_switch_turn = battle.turn(switch_action_t0, pass_action_t1)

    assert battle.round == 2
    assert (
        battle.trainers[0].active_pokemon.surname == "Chimchar"
    )  # Ash switched to Chimchar
    assert any("Ash brings in Chimchar!" in m.text for m in messages_switch_turn)
    assert any("waits." in m.text for m in messages_switch_turn)
    assert not battle.done


def test_struggle_mechanics(struggle_scenario_battle):
    """Tests that a Pokemon forced to Struggle behaves correctly."""
    battle = struggle_scenario_battle

    # Trainer 0 (Ash's Pikachu) should be forced to Struggle
    actions_t0 = battle.get_possible_actions(0)
    assert Action.create_attack(move_slot_index=-1) in actions_t0
    assert Action.create_switch(pokemon_index=1) in actions_t0 # Pikachu's teammate
    assert len(actions_t0) == 2 # Struggle and one Switch

    struggle_action_t0 = actions_t0[0]

    # Trainer 1 (Gary's Piplup) can use a normal attack
    actions_t1 = battle.get_possible_actions(1)
    attack_t1 = next(a for a in actions_t1 if a.action_type == ActionType.ATTACK)

    initial_pikachu_hp = battle.trainers[0].active_pokemon.hp
    initial_piplup_hp = battle.trainers[1].active_pokemon.hp

    # Execute the turn
    messages = battle.turn(struggle_action_t0, attack_t1)

    # Assert recoil damage to user (Pikachu)
    assert battle.trainers[0].active_pokemon.hp < initial_pikachu_hp
    # Assert damage to target (Piplup)
    assert battle.trainers[1].active_pokemon.hp < initial_piplup_hp

    # Check messages for Struggle and recoil
    assert any("Struggle" in m.text for m in messages)
    assert any("recoil" in m.text for m in messages)

    assert battle.round == 1
    assert not battle.done  # Battle should still be ongoing


def test_battle_end_condition(battle_instance):
    """Tests the end condition of a battle."""
    battle_instance.trainers[1].pokemon_team[0].hp = 0
    assert battle_instance.trainers[1].is_defeated

    assert battle_instance.end() is True
    assert battle_instance.winner == 0


def test_battle_tie_condition(battle_instance):
    """Tests the tie condition of a battle."""
    battle_instance.round = battle_instance.max_rounds
    assert battle_instance.end() is True
    assert battle_instance.tie is True
    assert battle_instance.winner is None

def test_battle_custom_copy_performance(benchmark, battle_instance):
    """Benchmarks the custom copy method."""
    custom_copied_battle = benchmark(battle_instance.copy)

    assert custom_copied_battle is not battle_instance
    assert custom_copied_battle.trainers[0] is not battle_instance.trainers[0]
    assert custom_copied_battle.trainers[1] is not battle_instance.trainers[1]
    assert (
        custom_copied_battle.trainers[0].pokemon_team[0]
        is not battle_instance.trainers[0].pokemon_team[0]
    )
    assert (
        custom_copied_battle.trainers[1].pokemon_team[0]
        is not battle_instance.trainers[1].pokemon_team[0]
    )

    assert custom_copied_battle.round == battle_instance.round
    assert custom_copied_battle.winner == battle_instance.winner
    assert custom_copied_battle.tie == battle_instance.tie

    assert custom_copied_battle.trainers[0].name == battle_instance.trainers[0].name
    assert len(custom_copied_battle.trainers[0].pokemon_team) == len(
        battle_instance.trainers[0].pokemon_team
    )
    assert (
        custom_copied_battle.trainers[0].pokemon_team[0].hp
        == battle_instance.trainers[0].pokemon_team[0].hp
    )