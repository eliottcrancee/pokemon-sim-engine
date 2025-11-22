import math
import os
import sys

from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from pokemon.agent import BaseAgent, InputAgent
from pokemon.battle import Battle


def play(
    battle: Battle, agent_0: BaseAgent, agent_1: BaseAgent, verbose=False
) -> float:
    """Main game loop."""
    if isinstance(agent_0, InputAgent) and isinstance(agent_1, InputAgent):
        raise ValueError("UI mode is not supported in this play function.")

    battle.reset()
    while not battle.done:
        action_0 = agent_0.get_action(battle, 0, verbose=verbose)
        action_1 = agent_1.get_action(battle, 1, verbose=verbose)
        battle.turn(action_0, action_1)

    return battle.winner


def play_multiple(
    battle: Battle,
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    n_battles=1,
    n_jobs=1,
    verbose=True,
) -> tuple[int, int, int]:
    """Play multiple battles in parallel using joblib."""

    def _run():
        return play(battle.copy(), agent_0, agent_1, verbose=False)

    # Use return_as="generator" to update tqdm as batches complete
    parallel_gen = Parallel(n_jobs=n_jobs, prefer="processes", return_as="generator")(
        delayed(_run)() for _ in range(n_battles)
    )

    results = []
    with tqdm(total=n_battles, disable=not verbose) as pbar:
        for result in parallel_gen:
            results.append(result)
            pbar.update(1)

    # Aggregate
    agent_0_wins = sum(1 for r in results if r == 0)
    agent_1_wins = sum(1 for r in results if r == 1)
    draws = sum(1 for r in results if r == 0.5)

    return agent_0_wins, draws, agent_1_wins
