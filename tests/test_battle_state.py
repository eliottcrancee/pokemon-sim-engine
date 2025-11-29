# tests/test_battle_state.py
from pokemon.battle import Battle
from pokemon.item import Items
from pokemon.move import Moves
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.pokemon_status import PokemonStatus
from pokemon.trainer import Trainer


def create_test_battle():
    """Helper function to create a standard, reproducible battle state."""
    p1 = Pokemon(
        Pokedex.Pikachu,
        level=5,
        moves=[Moves.ThunderShock, Moves.Growl],
    )
    t1 = Trainer("Ash", [p1], inventory={Items.Potion: 1})

    p2 = Pokemon(
        Pokedex.Eevee,
        level=5,
        moves=[Moves.Tackle, Moves.TailWhip],
    )
    t2 = Trainer("Gary", [p2], inventory={Items.Potion: 1})

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
