# play.py

import random

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

from pokemon.agents import BaseAgent, InputAgent
from pokemon.battle import Battle
from pokemon.loguru_logger import logger


def play(
    battle: Battle, agent_0: BaseAgent, agent_1: BaseAgent, verbose=False
) -> float:
    """Main game loop."""

    if isinstance(agent_0, InputAgent) and isinstance(agent_1, InputAgent):
        raise ValueError("UI mode is not supported in this play function.")

    if verbose:
        logger.info(f"Starting a new battle between {agent_0.name} and {agent_1.name}")

    battle.reset()
    while not battle.done:
        action_0 = agent_0.get_action(battle, 0, verbose=verbose)
        action_1 = agent_1.get_action(battle, 1, verbose=verbose)
        messages = battle.turn(action_0, action_1)
        if verbose:
            for message in messages:
                logger.info(message.message)
    return battle.winner


def play_multiple(
    battle: Battle,
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    n_battles=1,
    n_jobs=-1,
    verbose=False,
) -> tuple[int, int, int]:
    """Play multiple battles in parallel using joblib."""

    if isinstance(agent_0, InputAgent) and isinstance(agent_1, InputAgent):
        raise ValueError("UI mode is not supported in this play function.")

    if n_jobs == -1:
        n_jobs = cpu_count()

    def _run():
        return play(battle.copy(), agent_0, agent_1)

    parallel_gen = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
        delayed(_run)() for _ in range(n_battles)
    )

    results = []

    with tqdm(total=n_battles, disable=not verbose) as pbar:
        for result in parallel_gen:
            results.append(result)
            pbar.update(1)

    agent_0_wins = sum(1 for r in results if r == 0)
    agent_1_wins = sum(1 for r in results if r == 1)
    draws = sum(1 for r in results if r == 0.5)

    return agent_0_wins, draws, agent_1_wins


def play_tournament(
    agent_pool: list[BaseAgent],
    battle: Battle,
    n_matches: int,
    n_battles_per_match: int,
) -> None:
    for _ in tqdm(range(n_matches), desc="Glicko-2 Tournament"):
        agent0, agent1 = random.sample(agent_pool, 2)

        agent0_wins, draws, agent1_wins = play_multiple(
            battle,
            agent0,
            agent1,
            n_battles=n_battles_per_match,
            n_jobs=-1,
            verbose=False,
        )

        # Normalized scores (Glicko-2 expects a score per game, but we batch them)
        total = agent0_wins + agent1_wins + draws
        score0 = agent0_wins / total + 0.5 * draws / total
        score1 = agent1_wins / total + 0.5 * draws / total

        # Update Glicko-2 pour le match
        agent0.update_rating([(agent1.rating, agent1.rating_deviation, score0)])
        agent1.update_rating([(agent0.rating, agent0.rating_deviation, score1)])

    agent_pool.sort(key=lambda a: a.rating, reverse=True)

    return agent_pool
