import hashlib
import random
import pickle

import pytest

from pokemon.agents.search import (
    AlphaBetaAgent,
    MinimaxAgent,
    _evaluation_func_cache_signature,
)
from pokemon.battle_registry import BattleRegistry


def _get_reference_battle():
    battle = BattleRegistry.get("sinnoh_opening_sparks")
    if battle is None:
        raise AssertionError(
            "BattleRegistry scenario 'sinnoh_opening_sparks' not found"
        )
    return battle


def test_minimax_cache_matches_no_cache_action() -> None:
    battle = _get_reference_battle()

    no_cache_agent = MinimaxAgent(depth=2, use_cache=False, parallelize=False)
    cache_agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    action_no_cache = no_cache_agent.get_action(battle.copy(), trainer_id=0)

    random.seed(2026)
    action_cache = cache_agent.get_action(battle.copy(), trainer_id=0)

    assert action_cache == action_no_cache


def test_alphabeta_cache_matches_no_cache_action() -> None:
    battle = _get_reference_battle()

    no_cache_agent = AlphaBetaAgent(depth=2, use_cache=False, parallelize=False)
    cache_agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    action_no_cache = no_cache_agent.get_action(battle.copy(), trainer_id=0)

    random.seed(2026)
    action_cache = cache_agent.get_action(battle.copy(), trainer_id=0)

    assert action_cache == action_no_cache


def test_minimax_root_cache_reuses_action_without_search() -> None:
    battle = _get_reference_battle()
    agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    first_action = agent.get_action(battle.copy(), trainer_id=0)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Minimax search should not run on root cache hit")

    agent._evaluate_minimax_action = _should_not_run  # type: ignore[method-assign]

    second_action = agent.get_action(battle.copy(), trainer_id=0)

    assert second_action == first_action


def test_alphabeta_root_cache_reuses_action_without_search() -> None:
    battle = _get_reference_battle()
    agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    first_action = agent.get_action(battle.copy(), trainer_id=0)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("AlphaBeta search should not run on root cache hit")

    agent._get_min_value = _should_not_run  # type: ignore[method-assign]

    second_action = agent.get_action(battle.copy(), trainer_id=0)

    assert second_action == first_action


def test_minimax_root_cache_is_shared_across_instances() -> None:
    battle = _get_reference_battle()
    MinimaxAgent.clear_shared_cache()

    source_agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)
    target_agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    first_action = source_agent.get_action(battle.copy(), trainer_id=0)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Minimax search should not run on shared cache hit")

    target_agent._evaluate_minimax_action = _should_not_run  # type: ignore[method-assign]

    second_action = target_agent.get_action(battle.copy(), trainer_id=0)

    assert second_action == first_action


def test_alphabeta_root_cache_is_shared_across_instances() -> None:
    battle = _get_reference_battle()
    AlphaBetaAgent.clear_shared_cache()

    source_agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)
    target_agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)

    random.seed(2026)
    first_action = source_agent.get_action(battle.copy(), trainer_id=0)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("AlphaBeta search should not run on shared cache hit")

    target_agent._get_min_value = _should_not_run  # type: ignore[method-assign]

    second_action = target_agent.get_action(battle.copy(), trainer_id=0)

    assert second_action == first_action


def test_minimax_shared_cache_ignores_seed_when_state_is_same() -> None:
    battle = _get_reference_battle()
    MinimaxAgent.clear_shared_cache()

    source_agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)
    target_agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)

    first_state = battle.copy()
    first_state.reseed(111)
    first_action = source_agent.get_action(first_state, trainer_id=0)

    second_state = battle.copy()
    second_state.reseed(222)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("Minimax search should not run on shared cache hit")

    target_agent._evaluate_minimax_action = _should_not_run  # type: ignore[method-assign]

    second_action = target_agent.get_action(second_state, trainer_id=0)

    assert second_action == first_action


def test_alphabeta_shared_cache_ignores_seed_when_state_is_same() -> None:
    battle = _get_reference_battle()
    AlphaBetaAgent.clear_shared_cache()

    source_agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)
    target_agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)

    first_state = battle.copy()
    first_state.reseed(333)
    first_action = source_agent.get_action(first_state, trainer_id=0)

    second_state = battle.copy()
    second_state.reseed(444)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("AlphaBeta search should not run on shared cache hit")

    target_agent._get_min_value = _should_not_run  # type: ignore[method-assign]

    second_action = target_agent.get_action(second_state, trainer_id=0)

    assert second_action == first_action


def test_minimax_shared_cache_persists_to_disk(tmp_path) -> None:
    battle = _get_reference_battle()
    cache_file = tmp_path / "minimax_cache.pkl"

    MinimaxAgent.clear_shared_cache()
    agent = MinimaxAgent(depth=2, use_cache=True, parallelize=False)
    agent.get_action(battle.copy(), trainer_id=0)

    MinimaxAgent.save_shared_cache(cache_file)
    MinimaxAgent.clear_shared_cache()

    MinimaxAgent.load_shared_cache(cache_file)
    stats = MinimaxAgent.get_shared_cache_stats()
    assert stats["size"] > 0


def test_alphabeta_shared_cache_persists_to_disk(tmp_path) -> None:
    battle = _get_reference_battle()
    cache_file = tmp_path / "alphabeta_cache.pkl"

    AlphaBetaAgent.clear_shared_cache()
    agent = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)
    agent.get_action(battle.copy(), trainer_id=0)

    AlphaBetaAgent.save_shared_cache(cache_file)
    AlphaBetaAgent.clear_shared_cache()

    AlphaBetaAgent.load_shared_cache(cache_file)
    stats = AlphaBetaAgent.get_shared_cache_stats()
    assert stats["size"] > 0


def test_minimax_shared_cache_export_and_merge_roundtrip() -> None:
    battle = _get_reference_battle()

    MinimaxAgent.clear_shared_cache()
    source = MinimaxAgent(depth=2, use_cache=True, parallelize=False)
    source.get_action(battle.copy(), trainer_id=0)

    exported = MinimaxAgent.export_shared_cache()
    assert exported

    MinimaxAgent.clear_shared_cache()
    inserted = MinimaxAgent.merge_shared_cache(exported, hits=3, misses=4)
    stats = MinimaxAgent.get_shared_cache_stats()

    assert inserted == len(exported)
    assert stats["size"] == len(exported)
    assert stats["hits"] == 3
    assert stats["misses"] == 4


def test_alphabeta_shared_cache_export_and_merge_roundtrip() -> None:
    battle = _get_reference_battle()

    AlphaBetaAgent.clear_shared_cache()
    source = AlphaBetaAgent(depth=2, use_cache=True, parallelize=False)
    source.get_action(battle.copy(), trainer_id=0)

    exported = AlphaBetaAgent.export_shared_cache()
    assert exported

    AlphaBetaAgent.clear_shared_cache()
    inserted = AlphaBetaAgent.merge_shared_cache(exported, hits=5, misses=6)
    stats = AlphaBetaAgent.get_shared_cache_stats()

    assert inserted == len(exported)
    assert stats["size"] == len(exported)
    assert stats["hits"] == 5
    assert stats["misses"] == 6


def test_evaluation_signature_is_deterministic_value() -> None:
    def _dummy_eval(*args, **kwargs):
        return 0.0

    module = _dummy_eval.__module__
    qualname = _dummy_eval.__qualname__
    expected = int.from_bytes(
        hashlib.blake2b(f"{module}:{qualname}".encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )

    assert _evaluation_func_cache_signature(_dummy_eval) == expected


def test_load_shared_cache_ignores_unknown_payload_version(tmp_path) -> None:
    cache_file = tmp_path / "minimax_invalid_version.pkl"
    payload = {
        "version": 999,
        "cache": {(1, 2, 3, 4, 5, 6): object()},
        "hits": 100,
        "misses": 50,
    }
    with cache_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    MinimaxAgent.clear_shared_cache()
    MinimaxAgent.load_shared_cache(cache_file)
    stats = MinimaxAgent.get_shared_cache_stats()

    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


@pytest.mark.parametrize(
    ("agent_cls", "depth"),
    [
        (MinimaxAgent, 2),
        (AlphaBetaAgent, 2),
    ],
)
@pytest.mark.parametrize("trainer_id", [0, 1])
@pytest.mark.parametrize(
    "battle_name",
    ["sinnoh_opening_sparks", "midgame_pivot_pressure"],
)
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_cache_and_no_cache_match_across_states(
    agent_cls, depth: int, trainer_id: int, battle_name: str, seed: int
) -> None:
    battle = BattleRegistry.get(battle_name)
    if battle is None:
        raise AssertionError(f"BattleRegistry scenario '{battle_name}' not found")

    clear_shared_cache = getattr(agent_cls, "clear_shared_cache", None)
    if callable(clear_shared_cache):
        clear_shared_cache()

    battle.reseed(seed)

    no_cache_agent = agent_cls(depth=depth, use_cache=False, parallelize=False)
    cache_agent = agent_cls(depth=depth, use_cache=True, parallelize=False)

    random.seed(2026)
    action_no_cache = no_cache_agent.get_action(battle.copy(), trainer_id=trainer_id)

    random.seed(2026)
    action_cache = cache_agent.get_action(battle.copy(), trainer_id=trainer_id)

    assert action_cache == action_no_cache
