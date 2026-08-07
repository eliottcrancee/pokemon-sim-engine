from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

from joblib import Parallel, delayed

from pokemon.battle import Battle

from .base_agent import BaseAgent


class _CacheAgentClassProtocol(Protocol):
    @classmethod
    def clear_shared_cache(cls) -> None: ...

    @classmethod
    def get_shared_cache_stats(cls) -> dict[str, int]: ...

    @classmethod
    def export_shared_cache(cls) -> dict[Any, Any]: ...

    @classmethod
    def merge_shared_cache(
        cls,
        cache_entries: dict[Any, Any],
        *,
        hits: int = 0,
        misses: int = 0,
        overwrite: bool = False,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class CachePrecomputeReport:
    total_battles: int
    total_rounds: int
    workers_used: int
    elapsed_seconds: float
    merged_entries: int
    worker_hits: int
    worker_misses: int
    final_cache_size: int
    batches: int


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    cache_entries: dict[Any, Any]
    hits: int
    misses: int
    rounds: int


def _split_even_work(total_items: int, chunks: int) -> list[int]:
    base = total_items // chunks
    remainder = total_items % chunks
    distribution: list[int] = []
    for index in range(chunks):
        distribution.append(base + (1 if index < remainder else 0))
    return [count for count in distribution if count > 0]


def _is_cache_precomputable(agent: BaseAgent) -> bool:
    if not bool(getattr(agent, "use_cache", False)):
        return False

    agent_cls = cast(_CacheAgentClassProtocol, agent.__class__)
    return all(
        callable(getattr(agent_cls, method_name, None))
        for method_name in (
            "clear_shared_cache",
            "get_shared_cache_stats",
            "export_shared_cache",
            "merge_shared_cache",
        )
    )


def _run_worker(
    cache_agent_factory: Callable[[], BaseAgent],
    opponent_factory: Callable[[], BaseAgent],
    battle_template: Battle,
    n_battles: int,
    warm_both_sides: bool,
    base_cache: dict[Any, Any] | None,
) -> _WorkerResult:
    cache_agent = cache_agent_factory()
    opponent = opponent_factory()

    cache_cls = cast(_CacheAgentClassProtocol, cache_agent.__class__)
    cache_cls.clear_shared_cache()

    if base_cache:
        cache_cls.merge_shared_cache(base_cache)

    if hasattr(cache_agent, "parallelize"):
        setattr(cache_agent, "parallelize", False)
    if hasattr(opponent, "parallelize"):
        setattr(opponent, "parallelize", False)

    total_rounds = 0
    for _ in range(n_battles):
        battle = battle_template.copy()
        battle.reset(reseed=True)

        while not battle.done:
            action_0 = cache_agent.get_action(battle, trainer_id=0, verbose=False)
            if warm_both_sides:
                action_1 = cache_agent.get_action(battle, trainer_id=1, verbose=False)
            else:
                action_1 = opponent.get_action(battle, trainer_id=1, verbose=False)
            battle.turn(action_0, action_1)

        total_rounds += battle.round

    stats = cache_cls.get_shared_cache_stats()
    return _WorkerResult(
        cache_entries=cache_cls.export_shared_cache(),
        hits=int(stats.get("hits", 0)),
        misses=int(stats.get("misses", 0)),
        rounds=total_rounds,
    )


def precompute_shared_action_cache(
    *,
    battle_template: Battle,
    cache_agent_factory: Callable[[], BaseAgent],
    opponent_factory: Callable[[], BaseAgent],
    total_battles: int,
    workers: int | None = None,
    merge_every: int | None = None,
    clear_before_merge: bool = True,
    warm_both_sides: bool = False,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> CachePrecomputeReport:
    if total_battles < 1:
        raise ValueError("total_battles must be >= 1")

    effective_workers = workers or (os.cpu_count() or 1)
    effective_workers = max(1, effective_workers)

    probe_agent = cache_agent_factory()
    if not _is_cache_precomputable(probe_agent):
        raise ValueError(
            "cache_agent_factory must create a cache-enabled shared-cache agent"
        )

    cache_cls = cast(_CacheAgentClassProtocol, probe_agent.__class__)

    batches: list[int] = []
    if merge_every is None:
        batches = [total_battles]
    else:
        if merge_every < 1:
            raise ValueError("merge_every must be >= 1 when provided")
        remaining = total_battles
        while remaining > 0:
            batch_size = min(merge_every, remaining)
            batches.append(batch_size)
            remaining -= batch_size

    if clear_before_merge:
        cache_cls.clear_shared_cache()

    merged_entries = 0
    worker_hits = 0
    worker_misses = 0
    total_rounds = 0
    completed_battles = 0
    max_workers_used = 0

    started_at = time.perf_counter()
    with Parallel(n_jobs=effective_workers, prefer="processes") as parallel:
        for batch_index, batch_size in enumerate(batches, 1):
            chunks = _split_even_work(batch_size, effective_workers)
            if not chunks:
                continue

            max_workers_used = max(max_workers_used, len(chunks))
            base_cache = cache_cls.export_shared_cache()

            worker_results = cast(
                list[_WorkerResult],
                parallel(
                    delayed(_run_worker)(
                        cache_agent_factory,
                        opponent_factory,
                        battle_template,
                        battles_for_worker,
                        warm_both_sides,
                        base_cache,
                    )
                    for battles_for_worker in chunks
                ),
            )

            for result in worker_results:
                merged_entries += cache_cls.merge_shared_cache(
                    result.cache_entries,
                    hits=result.hits,
                    misses=result.misses,
                    overwrite=False,
                )
                worker_hits += result.hits
                worker_misses += result.misses
                total_rounds += result.rounds

            completed_battles += batch_size
            current_size = cache_cls.get_shared_cache_stats()["size"]
            if progress_callback is not None:
                progress_callback(
                    batch_index, len(batches), completed_battles, current_size
                )

    elapsed = time.perf_counter() - started_at
    final_stats = cache_cls.get_shared_cache_stats()

    return CachePrecomputeReport(
        total_battles=total_battles,
        total_rounds=total_rounds,
        workers_used=max_workers_used,
        elapsed_seconds=elapsed,
        merged_entries=merged_entries,
        worker_hits=worker_hits,
        worker_misses=worker_misses,
        final_cache_size=final_stats["size"],
        batches=len(batches),
    )
