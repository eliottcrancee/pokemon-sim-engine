from __future__ import annotations

from pokemon.agents import MinimaxAgent, precompute_shared_action_cache
from pokemon.battle_registry import BattleRegistry


def main() -> None:
    battle = BattleRegistry.get("sinnoh_opening_sparks")
    if battle is None:
        raise RuntimeError("Battle scenario 'sinnoh_opening_sparks' not found")

    MinimaxAgent.clear_shared_cache()

    report = precompute_shared_action_cache(
        battle_template=battle,
        cache_agent_factory=lambda: MinimaxAgent(depth=2, use_cache=True),
        opponent_factory=lambda: MinimaxAgent(depth=2, use_cache=True),
        total_battles=24,
        workers=4,
        merge_every=6,
        clear_before_merge=True,
        warm_both_sides=True,
        progress_callback=lambda i, n, done, size: print(
            f"batch {i}/{n} -> battles={done}, cache_size={size}"
        ),
    )

    print("\nprecompute report")
    print(f"- battles: {report.total_battles}")
    print(f"- rounds: {report.total_rounds}")
    print(f"- workers used: {report.workers_used}")
    print(f"- elapsed: {report.elapsed_seconds:.3f}s")
    print(f"- merged entries: {report.merged_entries}")
    print(f"- final cache size: {report.final_cache_size}")

    if report.final_cache_size <= 0:
        raise RuntimeError("Precompute failed: shared cache remained empty")

    stats_before = MinimaxAgent.get_shared_cache_stats()
    agent = MinimaxAgent(depth=2, use_cache=True)
    agent.get_action(battle.copy(), trainer_id=0)
    agent.get_action(battle.copy(), trainer_id=1)
    stats_after = MinimaxAgent.get_shared_cache_stats()

    print("\ncache hit check")
    print(f"- hits before checks: {stats_before['hits']}")
    print(f"- hits after checks: {stats_after['hits']}")

    if stats_after["hits"] <= stats_before["hits"]:
        raise RuntimeError("Precompute failed: no cache hit observed on replay")

    print("\nValidation OK: precompute populates and reuses shared cache.")


if __name__ == "__main__":
    main()
