import concurrent.futures
import os
import sys
import traceback

from dotenv import load_dotenv
from tqdm import tqdm

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from envs.pokemon.agent import BaseAgent, InputAgent
from envs.pokemon.battle import Battle

load_dotenv()
DEBUG = os.getenv("DEBUG") in ["True", "true", "1", "t", "y", "yes"]


def play(battle: Battle, agent_0: BaseAgent, agent_1: BaseAgent):
    """Main game loop."""
    if isinstance(agent_0, InputAgent) and isinstance(agent_1, InputAgent):
        raise ValueError("UI mode is not supported in this play function.")

    # Silent simulation loop
    battle.reset()
    while not battle.done:
        # Get actions from agents
        action_0 = agent_0.get_action(battle, 0)
        action_1 = agent_1.get_action(battle, 1)

        # Execute turn (ignoring messages)
        battle.turn(action_0, action_1)

    return battle.winner


def play_multiple(
    battle: Battle,
    agent_0: BaseAgent,
    agent_1: BaseAgent,
    n_battles=1,
    verbose=False,
    ui=False,
) -> tuple[int, int, int]:
    agent_0_wins, agent_1_wins, draws = 0, 0, 0
    for _ in tqdm(range(n_battles), desc="Playing", disable=not verbose):
        winner = play(battle, agent_0, agent_1, verbose=verbose, ui=ui)
        if winner == 0.5:
            draws += 1
        elif winner == 0:
            agent_0_wins += 1
        else:
            agent_1_wins += 1
    return agent_0_wins, draws, agent_1_wins


def evaluate(
    battle_generator, agent_0, agent_1, n_battles=1, verbose=False, ui=False
) -> tuple[int, int]:
    try:
        battle = battle_generator()
        return play_multiple(battle, agent_0, agent_1, n_battles, verbose, ui=ui)
    except Exception as e:
        print(f"An error occurred in evaluate: {repr(e)}")
        traceback.print_exc()
        return 0, 0


def parallel_evaluate(
    battle_generator,
    agent_0,
    agent_1,
    n_battles=1024,
    batch_size=64,
    job_size=32,
    verbose=False,
) -> tuple[int, int, int]:
    total_agent_0_wins = 0
    total_draws = 0
    total_agent_1_wins = 0

    battles = [battle_generator() for _ in range(job_size)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        with tqdm(total=n_battles, desc="Evaluating", disable=not verbose) as pbar:
            for _ in range(0, n_battles, batch_size * job_size):
                for i in range(job_size):
                    futures.append(
                        executor.submit(
                            play_multiple, battles[i], agent_0, agent_1, batch_size
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    try:
                        agent_0_wins, draws, agent_1_wins = future.result()
                        total_agent_0_wins += agent_0_wins
                        total_draws += draws
                        total_agent_1_wins += agent_1_wins
                    except Exception as e:
                        print(f"An error occurred while processing a future: {repr(e)}")
                    pbar.update(batch_size)
                futures.clear()

    return total_agent_0_wins, total_draws, total_agent_1_wins
