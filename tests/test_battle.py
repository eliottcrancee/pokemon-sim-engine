from unittest.mock import patch

import pytest

from pokemon.action import Action, ActionType
from pokemon.battle import Battle
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.pokemon_status import PokemonStatus
from pokemon.trainer import Trainer


@pytest.fixture
def battle_instance():
    """Fixture to create a standard battle instance for testing."""
    trainer1 = Trainer(
        name="Dawn",
        pokemon_team=[
            Pokemon(species=Pokedex.Shinx, level=12),
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Lucas", pokemon_team=[Pokemon(species=Pokedex.Piplup, level=11)]
    )
    return Battle(trainers=(trainer1, trainer2), headless=False)


@pytest.fixture
def faint_scenario_battle():
    """Fixture to create a battle where a pokemon can faint easily."""
    shinx = Pokemon(species=Pokedex.Shinx, level=10)
    shinx.hp = 1  # Manually set HP after creation

    trainer1 = Trainer(
        name="Dawn",
        pokemon_team=[
            shinx,
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Lucas",
        pokemon_team=[
            Pokemon(species=Pokedex.Piplup, level=10),
        ],
    )
    battle = Battle(trainers=(trainer1, trainer2), headless=False)
    battle.trainers[0].active_pokemon.hp = 1  # Set Shinx's HP to 1 after reset
    # Ensure Piplup has an attack that can one-shot Shinx
    piplup = battle.trainers[1].active_pokemon
    piplup.move_slots[0].move.power = 100  # Make the first move very powerful
    return battle


@pytest.fixture
def struggle_scenario_battle():
    """Fixture to create a battle where a pokemon is forced to Struggle."""
    # Create a Shinx with all PP reduced to 0
    shinx_struggle = Pokemon(species=Pokedex.Shinx, level=10)
    for slot in shinx_struggle.move_slots:
        slot.current_pp = 0

    trainer1 = Trainer(
        name="Dawn",
        pokemon_team=[
            shinx_struggle,
            Pokemon(species=Pokedex.Chimchar, level=10),
        ],
    )
    trainer2 = Trainer(
        name="Lucas",
        pokemon_team=[
            Pokemon(species=Pokedex.Piplup, level=10),
        ],
    )
    battle = Battle(trainers=(trainer1, trainer2), headless=False)
    for slot in battle.trainers[
        0
    ].active_pokemon.move_slots:  # Set PP to 0 after battle reset
        slot.current_pp = 0
    return battle


def test_battle_initialization(battle_instance):
    """Tests that the battle is initialized correctly."""
    assert battle_instance.round == 0
    assert not battle_instance.done
    assert battle_instance.winner is None
    assert battle_instance.trainers[0].name == "Dawn"
    assert battle_instance.trainers[1].name == "Lucas"


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
    attack_t0 = next(
        a for a in actions_t0_initial if a.action_type == ActionType.ATTACK
    )

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

    # Trainer 1 (Lucas) should only be able to pass
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
    )  # Dawn switched to Chimchar
    assert any("Dawn brings in Chimchar!" in m.text for m in messages_switch_turn)
    assert any("waits." in m.text for m in messages_switch_turn)
    assert not battle.done


def test_struggle_mechanics(struggle_scenario_battle):
    """Tests that a Pokemon forced to Struggle behaves correctly."""
    battle = struggle_scenario_battle

    # Trainer 0 (Ash's Pikachu) should be forced to Struggle
    actions_t0 = battle.get_possible_actions(0)
    assert Action.create_attack(move_slot_index=-1) in actions_t0
    assert Action.create_switch(pokemon_index=1) in actions_t0  # Pikachu's teammate
    assert len(actions_t0) == 2  # Struggle and one Switch

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


def test_after_turn_poison_can_end_in_tie():
    """If both active Pokemon are at 1 HP and poisoned, end-of-turn should tie."""
    dawn = Trainer(name="Dawn", pokemon_team=[Pokemon(species=Pokedex.Shinx, level=10)])
    lucas = Trainer(
        name="Lucas", pokemon_team=[Pokemon(species=Pokedex.Starly, level=10)]
    )
    battle = Battle((dawn, lucas), headless=False)

    p0 = battle.trainers[0].active_pokemon
    p1 = battle.trainers[1].active_pokemon
    p0.hp = 1
    p1.hp = 1
    p0.status = PokemonStatus.POISON
    p1.status = PokemonStatus.POISON

    messages = battle.turn(Action.create_pass(), Action.create_pass())

    assert battle.done is True
    assert battle.tie is True
    assert battle.winner is None
    assert p0.is_alive is False
    assert p1.is_alive is False
    assert any("hurt by poison" in m.text for m in messages)
    assert any("fainted" in m.text for m in messages)


def test_after_turn_damage_happens_after_actions():
    """A poisoned attacker at 1 HP still executes its move before after-turn KO."""
    dawn = Trainer(name="Dawn", pokemon_team=[Pokemon(species=Pokedex.Shinx, level=20)])
    lucas = Trainer(
        name="Lucas", pokemon_team=[Pokemon(species=Pokedex.Starly, level=20)]
    )
    battle = Battle((dawn, lucas), headless=True)

    attacker = battle.trainers[0].active_pokemon
    defender = battle.trainers[1].active_pokemon
    attacker.status = PokemonStatus.POISON
    attacker.hp = 1
    initial_defender_hp = defender.hp

    damaging_actions = [
        action
        for action in battle.get_possible_actions(0)
        if action.action_type == ActionType.ATTACK
        and action.attack_move_slot_index >= 0
        and attacker.move_slots[action.attack_move_slot_index].move.power > 0
    ]
    assert damaging_actions, "Expected at least one damaging action"

    attack_action = damaging_actions[0]
    attacker.move_slots[attack_action.attack_move_slot_index].move.accuracy = 100

    battle.turn(attack_action, Action.create_pass())

    assert defender.hp < initial_defender_hp
    assert attacker.is_alive is False
    assert battle.done is True
    assert battle.winner == 1


@patch("random.random", return_value=0.5)
def test_turn_skips_second_action_when_first_action_ends_battle(_mock_random):
    """If the first action KOs the last opponent Pokemon, second action is skipped."""
    fast_attacker = Pokemon(species=Pokedex.Shinx, level=50)
    lone_target = Pokemon(species=Pokedex.Starly, level=5)
    lone_target.hp = 1

    trainer_fast = Trainer(name="Fast", pokemon_team=[fast_attacker])
    trainer_slow = Trainer(name="Slow", pokemon_team=[lone_target])
    battle = Battle((trainer_fast, trainer_slow), headless=True)

    battle.trainers[0].active_pokemon.move_slots[0].move.power = 150
    battle.trainers[1].active_pokemon.move_slots[0].move.power = 150

    initial_fast_hp = battle.trainers[0].active_pokemon.hp
    action_fast = Action.create_attack(0)
    action_slow = Action.create_attack(0)

    battle.turn(action_fast, action_slow)

    assert battle.done is True
    assert battle.winner == 0
    assert battle.trainers[0].active_pokemon.hp == initial_fast_hp


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
