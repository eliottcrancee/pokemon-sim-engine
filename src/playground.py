import math
import os
import sys
import time

from tqdm import tqdm

# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from pokemon.agents import (
    AlphaBetaAgent,
    BaseAgent,
    BestAttackAgent,
    BestAttackAndPotionAgent,
    EvaluatingAgent,
    FirstAgent,
    InputAgent,
    MinimaxAgent,
    OneStepAlphaBetaAgent,
    OneStepMinimaxAgent,
    OneStepUniformExpectimaxAgent,
    RandomAgent,
    RandomAttackAgent,
    RandomAttackAndPotionAgent,
    ThreeStepAlphaBetaAgent,
    TwoStepAlphaBetaAgent,
    TwoStepMinimaxAgent,
)
from pokemon.battle import Battle
from pokemon.item import ItemAccessor
from pokemon.play import play_multiple, play_tournament
from pokemon.pokemon import PokemonAccessor
from pokemon.trainer import Trainer
from pokemon.ui import play_ui


def create_trainer(name):
    return Trainer(
        name=name,
        pokemon_team=[
            PokemonAccessor.Pikachu(level=12),
            PokemonAccessor.Charmander(level=12),
            PokemonAccessor.Squirtle(level=12),
        ],
        inventory={
            ItemAccessor.Potion.name: ItemAccessor.Potion(default_quantity=1),
            ItemAccessor.SuperPotion.name: ItemAccessor.SuperPotion(default_quantity=1),
        },
    )


def run_ui():
    ash = create_trainer("Ash")
    gary = create_trainer("Gary")
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    input_agent = InputAgent()
    opponent_agent = AlphaBetaAgent(depth=5, parallelize=True)
    play_ui(battle, input_agent, opponent_agent)


def run_versus():
    n_battles = 32

    ash = create_trainer("Ash")
    gary = create_trainer("Gary")
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    agent_0 = AlphaBetaAgent(depth=2)
    agent_1 = AlphaBetaAgent(depth=4)

    agent_0_wins, draws, agent_1_wins = play_multiple(
        battle,
        agent_0,
        agent_1,
        n_battles=n_battles,
        n_jobs=-1,
        verbose=True,
    )

    print("\n--- Versus Results ---")
    max_name_len = max(len(agent_0.name), len(agent_1.name))
    print(
        f"{agent_0.name:<{max_name_len}} : Wins = {agent_0_wins}, WinRate = {agent_0_wins / n_battles:.4f}"
    )
    print(
        f"{agent_1.name:<{max_name_len}} : Wins = {agent_1_wins}, WinRate = {agent_1_wins / n_battles:.4f}"
    )
    print(f"Draws : {draws}")


def run_test_performance():
    ash = create_trainer("Ash")
    gary = create_trainer("Gary")
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    agent = AlphaBetaAgent(depth=4, parallelize=False)

    start = time.perf_counter()

    n_calls = 10
    for _ in tqdm(range(n_calls), desc="Testing performance of get_action"):
        agent.get_action(battle, 1)
    end = time.perf_counter()
    print(f"Average time per get_action call: {(end - start) / n_calls:.6f} seconds")


def run_tournament():
    print("\n--- Starting Glicko-2 Tournament ---")

    ash = create_trainer("Ash")
    gary = create_trainer("Gary")
    battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

    n_matches = 256
    n_battles_per_match = 16

    agent_pool: list[BaseAgent] = [
        FirstAgent(),
        RandomAgent(),
        RandomAttackAgent(),
        BestAttackAgent(),
        RandomAttackAndPotionAgent(heal_threshold=0.2),
        BestAttackAndPotionAgent(heal_threshold=0.2),
        RandomAttackAndPotionAgent(heal_threshold=0.33),
        BestAttackAndPotionAgent(heal_threshold=0.33),
        OneStepUniformExpectimaxAgent(),
        MinimaxAgent(depth=1),
        AlphaBetaAgent(depth=1),
        AlphaBetaAgent(depth=2),
        AlphaBetaAgent(depth=3),
        AlphaBetaAgent(depth=4),
        AlphaBetaAgent(depth=5),
    ]

    play_tournament(
        agent_pool=agent_pool,
        battle=battle,
        n_matches=n_matches,
        n_battles_per_match=n_battles_per_match,
    )

    agent_pool.sort(key=lambda a: a.rating, reverse=True)
    best_agent = agent_pool[0]

    print("\n--- Agent Glicko-2 Ratings ---")
    max_name_len = max(len(agent.name) for agent in agent_pool)
    for agent in agent_pool:
        agent: BaseAgent
        mu = (agent.rating - 1500) / 173.7178
        mu_j = (best_agent.rating - 1500) / 173.7178
        phi_j = best_agent.rating_deviation / 173.7178
        g_phi_j = 1 / math.sqrt(1 + 3 * phi_j**2 / math.pi**2)
        win_prob = 1 / (1 + math.exp(-g_phi_j * (mu - mu_j)))

        print(
            f"{agent.name:<{max_name_len}} : Rating = {agent.rating:7.2f}, RD = {agent.rating_deviation:6.2f}, WinProb vs Best = {win_prob:.4f}"
        )


if __name__ == "__main__":
    run_ui()


# --- Agent Glicko-2 Ratings ---
# AlphaBeta(depth=3)                         : Rating = 2014.63, RD =  84.40, WinProb vs Best = 0.5000
# AlphaBeta(depth=2)                         : Rating = 1932.15, RD =  88.42, WinProb vs Best = 0.3873
# AlphaBeta(depth=1)                         : Rating = 1881.63, RD =  84.32, WinProb vs Best = 0.3231
# Minimax(depth=1)                           : Rating = 1812.27, RD =  82.03, WinProb vs Best = 0.2450
# OneStepUniformExpectimax                   : Rating = 1716.56, RD =  77.43, WinProb vs Best = 0.1601
# BestAttackAndPotion(heal_threshold=0.2)    : Rating = 1602.72, RD =  67.93, WinProb vs Best = 0.0919
# BestAttackAndPotion(heal_threshold=0.33)   : Rating = 1489.39, RD =  72.10, WinProb vs Best = 0.0511
# RandomAttackAndPotion(heal_threshold=0.2)  : Rating = 1430.54, RD =  80.08, WinProb vs Best = 0.0374
# RandomAttackAndPotion(heal_threshold=0.33) : Rating = 1346.12, RD =  82.86, WinProb vs Best = 0.0237
# BestAttack                                 : Rating = 1284.17, RD =  82.39, WinProb vs Best = 0.0169
# First                                      : Rating = 1234.17, RD =  85.02, WinProb vs Best = 0.0129
# RandomAttack                               : Rating = 1011.08, RD =  90.79, WinProb vs Best = 0.0038
# Random                                     : Rating =  935.95, RD = 104.38, WinProb vs Best = 0.0025
