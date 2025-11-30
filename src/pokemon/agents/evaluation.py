from pokemon.battle import Battle


def simple_evaluation(battle: Battle, trainer_id: int) -> float:
    """
    Simple evaluation based only on HP ratio difference.
    """
    if battle.winner == trainer_id:
        return 100.0
    if battle.winner is not None:
        return -100.0

    my_trainer = battle.get_trainer_by_id(trainer_id)
    op_trainer = battle.get_trainer_by_id(1 - trainer_id)

    my_hp = sum(p.hp / p.max_hp for p in my_trainer.pokemon_team if p.is_alive)
    op_hp = sum(p.hp / p.max_hp for p in op_trainer.pokemon_team if p.is_alive)

    return my_hp - op_hp


def standard_evaluation(battle: Battle, trainer_id: int) -> float:
    """
    Standard evaluation based on HP, alive pokemon count, and items.
    """
    if battle.winner == trainer_id:
        return 100.0
    if battle.winner is not None:
        return -100.0

    my_trainer = battle.get_trainer_by_id(trainer_id)
    op_trainer = battle.get_trainer_by_id(1 - trainer_id)

    # My Score
    my_hp_score = sum(p.hp / p.max_hp for p in my_trainer.pokemon_team if p.is_alive)
    my_alive_bonus = sum(0.5 for p in my_trainer.pokemon_team if p.is_alive)
    my_item_bonus = sum(
        0.2 * quantity for _, quantity in my_trainer.get_possessed_items()
    )
    my_score = my_hp_score + my_alive_bonus + my_item_bonus

    # Opponent Score
    op_hp_score = sum(p.hp / p.max_hp for p in op_trainer.pokemon_team if p.is_alive)
    op_alive_bonus = sum(0.5 for p in op_trainer.pokemon_team if p.is_alive)
    op_item_bonus = sum(
        0.2 * quantity for _, quantity in op_trainer.get_possessed_items()
    )
    op_score = op_hp_score + op_alive_bonus + op_item_bonus

    return my_score - op_score


def aggressive_evaluation(battle: Battle, trainer_id: int) -> float:
    """
    Aggressive evaluation that heavily penalizes opponent living pokemon.
    """
    if battle.winner == trainer_id:
        return 100.0
    if battle.winner is not None:
        return -100.0

    my_trainer = battle.get_trainer_by_id(trainer_id)
    op_trainer = battle.get_trainer_by_id(1 - trainer_id)

    # My Score (Focus on staying alive)
    my_hp_score = sum(p.hp / p.max_hp for p in my_trainer.pokemon_team if p.is_alive)

    # Opponent Score (Focus on killing them)
    # Higher penalty for opponent being alive
    op_alive_penalty = sum(1.0 for p in op_trainer.pokemon_team if p.is_alive)
    op_hp_score = sum(p.hp / p.max_hp for p in op_trainer.pokemon_team if p.is_alive)

    return my_hp_score - (op_hp_score + op_alive_penalty)
