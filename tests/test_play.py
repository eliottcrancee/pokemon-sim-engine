# tests/test_play.py

import pytest

from envs.pokemon.agent import BaseAgent, RandomAgent
from envs.pokemon.battle import Battle
from envs.pokemon.play import battle_generator, play, play_multiple


class MockAgent(BaseAgent):
    """A predictable agent for testing purposes."""

    def __init__(self, name="MockAgent"):
        super().__init__(name)

    def get_action(self, battle: Battle, trainer_id: int, verbose: bool = False):
        # Always choose the first available attack
        actions = battle.get_possible_actions(trainer_id)
        return actions[0]


@pytest.fixture
def agents():
    """Fixture to provide agents for tests."""
    return RandomAgent(), RandomAgent()


@pytest.fixture
def mock_agents():
    """Fixture to provide predictable agents."""
    return MockAgent(), MockAgent()


def test_battle_generator():
    """Tests the battle_generator function."""
    battle = battle_generator()
    assert isinstance(battle, Battle)
    assert battle.trainer_0.name == "Ash"
    assert len(battle.trainer_0.pokemon_team) == 2

    random_battle = battle_generator(randomize=True)
    assert isinstance(random_battle, Battle)
    # With randomization, it's hard to assert specific values,
    # but we can check if the objects are created.


def test_play_single_game(agents):
    """Tests running a single battle between two agents."""
    agent_0, agent_1 = agents
    battle = battle_generator()

    winner = play(battle, agent_0, agent_1, verbose=False)

    assert battle.done
    assert winner in [0, 1, 0.5]  # Winner is 0, 1, or 0.5 for a draw


def test_play_single_game_with_mock_agents(mock_agents):
    """Tests a single game with predictable agents to ensure it completes."""
    agent_0, agent_1 = mock_agents
    battle = battle_generator()

    # This test ensures that with predictable agents, the game runs to completion
    # without getting stuck in a loop.
    winner = play(battle, agent_0, agent_1, verbose=False)

    assert battle.done
    assert winner in [0, 1, 0.5]


def test_play_multiple_games(agents):
    """Tests running multiple battles."""
    agent_0, agent_1 = agents
    battle = battle_generator()
    n_battles = 5

    agent_0_wins, draws, agent_1_wins = play_multiple(
        battle, agent_0, agent_1, n_battles=n_battles, verbose=False
    )

    assert agent_0_wins + draws + agent_1_wins == n_battles
    assert agent_0_wins >= 0
    assert agent_1_wins >= 0
    assert draws >= 0
