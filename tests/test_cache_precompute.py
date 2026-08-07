from pokemon.agents import AlphaBetaAgent, MinimaxAgent, precompute_shared_action_cache
from pokemon.battle_registry import BattleRegistry


def _get_reference_battle():
    battle = BattleRegistry.get("sinnoh_opening_sparks")
    if battle is None:
        raise AssertionError(
            "BattleRegistry scenario 'sinnoh_opening_sparks' not found"
        )
    return battle


def test_precompute_populates_minimax_shared_cache() -> None:
    MinimaxAgent.clear_shared_cache()
    battle = _get_reference_battle()

    report = precompute_shared_action_cache(
        battle_template=battle,
        cache_agent_factory=lambda: MinimaxAgent(depth=2, use_cache=True),
        opponent_factory=lambda: AlphaBetaAgent(depth=2, use_cache=True),
        total_battles=8,
        workers=2,
        merge_every=4,
        clear_before_merge=True,
        warm_both_sides=False,
    )

    assert report.final_cache_size > 0

    stats_before = MinimaxAgent.get_shared_cache_stats()
    agent = MinimaxAgent(depth=2, use_cache=True)
    agent.get_action(battle.copy(), trainer_id=0)
    stats_after = MinimaxAgent.get_shared_cache_stats()

    assert stats_after["hits"] > stats_before["hits"]


def test_precompute_warms_both_sides_for_selfplay() -> None:
    AlphaBetaAgent.clear_shared_cache()
    battle = _get_reference_battle()

    report = precompute_shared_action_cache(
        battle_template=battle,
        cache_agent_factory=lambda: AlphaBetaAgent(depth=2, use_cache=True),
        opponent_factory=lambda: AlphaBetaAgent(depth=2, use_cache=True),
        total_battles=8,
        workers=2,
        merge_every=4,
        clear_before_merge=True,
        warm_both_sides=True,
    )

    assert report.final_cache_size > 0

    stats_before = AlphaBetaAgent.get_shared_cache_stats()
    agent = AlphaBetaAgent(depth=2, use_cache=True)
    agent.get_action(battle.copy(), trainer_id=1)
    stats_after = AlphaBetaAgent.get_shared_cache_stats()

    assert stats_after["hits"] > stats_before["hits"]
