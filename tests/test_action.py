import pytest

from pokemon.action import Action, ActionType


def test_action_factory_methods_are_valid() -> None:
    attack = Action.create_attack(0)
    switch = Action.create_switch(1)
    use_item = Action.create_use_item(0, 0)
    pas = Action.create_pass()

    assert attack.attack_move_slot_index == 0
    assert switch.switch_pokemon_index == 1
    assert use_item.use_item_id == 0
    assert use_item.use_item_target_index == 0
    assert pas.action_type == ActionType.PASS


def test_action_invalid_payloads_raise_value_error() -> None:
    with pytest.raises(ValueError):
        Action(ActionType.ATTACK)

    with pytest.raises(ValueError):
        Action(ActionType.SWITCH)

    with pytest.raises(ValueError):
        Action(ActionType.USE_ITEM, item_id=1)

    with pytest.raises(ValueError):
        Action(ActionType.PASS, move_slot_index=0)


def test_action_accessor_wrong_type_raises() -> None:
    attack = Action.create_attack(0)
    with pytest.raises(ValueError):
        _ = attack.switch_pokemon_index


def test_action_hashing_deduplicates_equal_actions() -> None:
    a1 = Action.create_attack(1)
    a2 = Action.create_attack(1)
    a3 = Action.create_switch(1)

    actions = {a1, a2, a3}

    assert len(actions) == 2
    assert a2 in actions


def test_action_is_valid_dict_key() -> None:
    action = Action.create_use_item(0, 0)
    cache = {action: 123.0}

    assert cache[Action.create_use_item(0, 0)] == 123.0
