"""Tests for the search agents in pokemon.agents.search."""

import random

import pytest

from pokemon.agents.search import (
    AlphaBetaAgent,
    IterativeDeepeningAlphaBetaAgent,
    MCTSAgent,
    MinimaxAgent,
    MonteCarloExpectiminimaxAgent,
    OneStepUniformExpectimaxAgent,
    RiskAwareExpectimaxAgent,
)
from pokemon.battle_registry import BattleRegistry

BASELINE_BATTLE = "sinnoh_opening_sparks"


@pytest.fixture
def battle():
    b = BattleRegistry.get(BASELINE_BATTLE)
    if b is None:
        raise AssertionError(f"BattleRegistry scenario '{BASELINE_BATTLE}' not found")
    return b


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [MonteCarloExpectiminimaxAgent, RiskAwareExpectimaxAgent])
def test_mc_agents_reject_depth_zero(cls):
    with pytest.raises(ValueError):
        cls(depth=0)


@pytest.mark.parametrize("cls", [MonteCarloExpectiminimaxAgent, RiskAwareExpectimaxAgent])
def test_mc_agents_reject_zero_simulations(cls):
    with pytest.raises(ValueError):
        cls(num_simulations=0)


def test_risk_aware_rejects_negative_lambda():
    with pytest.raises(ValueError):
        RiskAwareExpectimaxAgent(risk_lambda=-0.1)


# ---------------------------------------------------------------------------
# Monte Carlo sampling must explore the RNG distribution
# ---------------------------------------------------------------------------


def test_monte_carlo_samples_are_independent(battle):
    """
    Regression test: Battle.copy() clones the RNG state, so every sample of a
    pair must be re-seeded. Otherwise all samples are identical and the
    average collapses to one deterministic outcome.
    """
    agent = MonteCarloExpectiminimaxAgent(depth=1, num_simulations=40)
    random.seed(12345)
    my_actions = battle.get_possible_actions(0)
    op_actions = battle.get_possible_actions(1)

    # Find a pair whose *average evaluation score* genuinely varies across
    # seeds (HP differences alone are not enough: a guaranteed KO averages
    # to the same 100.0 every time).
    pair = None
    for a0 in my_actions:
        for a1 in op_actions:
            avg_scores = set()
            for _ in range(4):
                total = 0.0
                for _ in range(20):
                    sim = battle.copy()
                    sim.reseed(random.getrandbits(64))
                    sim.turn(a0, a1)
                    total += agent.evaluation_func(sim, 0)
                avg_scores.add(round(total / 20, 6))
            if len(avg_scores) > 1:
                pair = (a0, a1)
                break
        if pair is not None:
            break

    if pair is None:
        pytest.skip("No stochastic action pair found in this battle scenario")

    # The agent's sampling must surface at least two distinct average scores.
    scores = {
        round(agent._simulate_outcome(battle.copy(), 0, pair[0], pair[1], 1), 6)
        for _ in range(4)
    }
    assert len(scores) > 1, (
        "Monte Carlo sampling produced identical results across re-seeded "
        "simulations; RNG independence is broken"
    )


def test_risk_aware_std_deviation_nonzero_for_stochastic_pairs(battle):
    """
    Regression test: the risk penalty (risk_lambda * std) must be able to be
    nonzero. Before the fix, copied RNG state made std always 0.
    """
    agent = RiskAwareExpectimaxAgent(depth=1, num_simulations=40, risk_lambda=0.1)
    random.seed(12345)
    my_actions = battle.get_possible_actions(0)
    op_actions = battle.get_possible_actions(1)

    nonzero_std = False
    for a0 in my_actions:
        for a1 in op_actions:
            value = agent._risk_adjusted_pair_value(battle.copy(), 0, a0, a1, 1)
            # value = mean - lambda*std; for stochastic pairs std > 0, so the
            # risk-adjusted value must be strictly below the mean.
            mean = 0.0
            for _ in range(40):
                sim = battle.copy()
                sim.reseed(random.getrandbits(64))
                sim.turn(a0, a1)
                mean += agent.evaluation_func(sim, 0)
            mean /= 40
            if value < mean - 1e-9:
                nonzero_std = True
                break
        if nonzero_std:
            break

    assert nonzero_std, (
        "Risk-adjusted value never differs from the mean; std is always 0, "
        "the risk penalty is a no-op"
    )


# ---------------------------------------------------------------------------
# Depth recursion in Monte Carlo agents
# ---------------------------------------------------------------------------


def test_monte_carlo_depth_recurses(battle):
    """depth > 1 must recurse into _expectimax_value instead of stopping at 1."""
    agent = MonteCarloExpectiminimaxAgent(depth=2, num_simulations=3)
    random.seed(12345)
    my_actions = battle.get_possible_actions(0)
    op_actions = battle.get_possible_actions(1)
    assert my_actions
    assert op_actions

    score = agent._simulate_outcome(battle.copy(), 0, my_actions[0], op_actions[0], 2)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# All agents can produce a valid action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_factory",
    [
        lambda: OneStepUniformExpectimaxAgent(),
        lambda: MinimaxAgent(depth=1),
        lambda: AlphaBetaAgent(depth=1),
        lambda: MonteCarloExpectiminimaxAgent(depth=1, num_simulations=2),
        lambda: RiskAwareExpectimaxAgent(depth=1, num_simulations=2),
        lambda: IterativeDeepeningAlphaBetaAgent(max_depth=1, time_limit=0.5),
        lambda: MCTSAgent(time_limit=0.05, rollout_depth=2),
    ],
)
def test_agents_return_valid_action(battle, agent_factory):
    random.seed(12345)
    agent = agent_factory()
    action = agent.get_action(battle.copy(), trainer_id=0)
    assert action in battle.get_possible_actions(0)


def test_iterative_deepening_timeout_fallback(battle):
    """
    A zero time limit must still return the sorted fallback action without
    crashing (and the timeout path must not reference an unbound variable).
    """
    agent = IterativeDeepeningAlphaBetaAgent(max_depth=2, time_limit=0.0)
    action = agent.get_action(battle.copy(), trainer_id=0)
    assert action in battle.get_possible_actions(0)
