import pytest

from pokemon.battle import Battle
from pokemon.battle_registry import BattleRegistry
from pokemon.item import ItemError, Items
from pokemon.move import Moves
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.pokemon_status import PokemonStatus
from pokemon.trainer import Trainer


def create_test_battle():
    """Helper function to create a standard, reproducible battle state."""
    p1 = Pokemon(
        Pokedex.Shinx,
        level=5,
        moves=[Moves.ThunderShock, Moves.Growl],
    )
    t1 = Trainer("Dawn", [p1], inventory={Items.Potion: 1})

    p2 = Pokemon(
        Pokedex.Starly,
        level=5,
        moves=[Moves.Tackle, Moves.TailWhip],
    )
    t2 = Trainer("Lucas", [p2], inventory={Items.Potion: 1})

    battle = Battle((t1, t2))
    return battle


def test_battle_equality():
    """Tests that two identical battle states are considered equal."""
    b1 = create_test_battle()
    b2 = create_test_battle()

    assert b1 == b2
    assert hash(b1) == hash(b2)


def test_battle_inequality_on_round_change():
    """Tests that battles are not equal if the round number differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.round += 1

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_inequality_on_hp_change():
    """Tests that battles are not equal if a Pokemon's HP differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.trainers[0].active_pokemon.hp -= 1

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_inequality_on_pp_change():
    """Tests that battles are not equal if a move's PP differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.trainers[0].active_pokemon.move_slots[0].current_pp -= 1

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_inequality_on_inventory_change():
    """Tests that battles are not equal if a trainer's inventory differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.trainers[1].decrease_item_quantity(Items.Potion)

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_inequality_on_status_change():
    """Tests that battles are not equal if a Pokemon's status differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.trainers[0].active_pokemon.status = PokemonStatus.POISON

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_inequality_on_stat_stage_change():
    """Tests that battles are not equal if a Pokemon's stat stage differs."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b2.trainers[0].active_pokemon.apply_modifier(0, 1)  # +1 Attack stage

    assert b1 != b2
    assert hash(b1) != hash(b2)


def test_battle_hashing_in_set():
    """Tests if hashing works correctly for set membership."""
    b1 = create_test_battle()
    b2 = create_test_battle()
    b3 = create_test_battle()
    b3.trainers[0].active_pokemon.hp -= 5

    state_set = {b1}

    assert b2 in state_set
    assert b3 not in state_set

    state_set.add(b2)
    assert len(state_set) == 1  # Adding an equal object should not increase set size

    state_set.add(b3)
    assert len(state_set) == 2


_VALID_TRAINER_DEF = {
    "name": "Tester",
    "team": [{"species": "Shinx", "level": 10, "moves": ["Tackle"]}],
    "inventory": {"Potion": 1},
}


def test_register_duplicate_battle_name_raises() -> None:
    name = "pytest_registry_duplicate_case"
    BattleRegistry.register(name, _VALID_TRAINER_DEF, _VALID_TRAINER_DEF)

    with pytest.raises(ValueError):
        BattleRegistry.register(name, _VALID_TRAINER_DEF, _VALID_TRAINER_DEF)


def test_register_invalid_species_raises() -> None:
    bad_trainer = {
        "name": "BadSpecies",
        "team": [{"species": "DefinitelyNotAPokemon", "moves": ["Tackle"]}],
        "inventory": {},
    }

    with pytest.raises(ValueError, match="Species"):
        BattleRegistry.register(
            "pytest_registry_bad_species", bad_trainer, _VALID_TRAINER_DEF
        )


def test_register_invalid_move_raises() -> None:
    bad_trainer = {
        "name": "BadMove",
        "team": [{"species": "Shinx", "moves": ["DefinitelyNotAMove"]}],
        "inventory": {},
    }

    with pytest.raises(ValueError, match="Move"):
        BattleRegistry.register(
            "pytest_registry_bad_move", bad_trainer, _VALID_TRAINER_DEF
        )


def test_register_invalid_item_raises() -> None:
    bad_trainer = {
        "name": "BadItem",
        "team": [{"species": "Shinx", "moves": ["Tackle"]}],
        "inventory": {"DefinitelyNotAnItem": 1},
    }

    with pytest.raises(ItemError, match="Item"):
        BattleRegistry.register(
            "pytest_registry_bad_item", bad_trainer, _VALID_TRAINER_DEF
        )


def test_predefined_battle_names_are_registered() -> None:
    registered = set(BattleRegistry.list_battles())
    expected = {
        "sinnoh_opening_sparks",
        "sinnoh_starter_triangle",
        "midgame_pivot_pressure",
        "brain_vs_brawn_gen4",
        "sinnoh_champion_clash_6v6",
        "creation_trio_showdown",
        "lucario_mirror_duel",
    }
    assert expected.issubset(registered)


def test_predefined_battles_can_play_one_turn() -> None:
    for battle_name in BattleRegistry.list_battles():
        battle = BattleRegistry.get(battle_name)
        assert battle is not None

        actions_0 = battle.get_possible_actions(0)
        actions_1 = battle.get_possible_actions(1)

        assert len(actions_0) > 0
        assert len(actions_1) > 0

        battle.turn(actions_0[0], actions_1[0])
        assert battle.round == 1
