# tests/test_play.py

import pytest

from pokemon.agents import BaseAgent, RandomAgent
from pokemon.battle import Battle
from pokemon.item import Items
from pokemon.play import play, play_multiple
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.trainer import Trainer


def battle_generator():
    pikachu_ash = Pokemon(species=Pokedex.Pikachu, level=10)
    charmander_ash = Pokemon(species=Pokedex.Chimchar, level=10)
    potion_ash = Items.Potion

    ash = Trainer(
        name="Ash",
        pokemon_team=[pikachu_ash, charmander_ash],
        inventory={potion_ash: 1},
    )

    squirtle_gary = Pokemon(species=Pokedex.Piplup, level=10) # Changed from Pikachu to Piplup to match name and type
    bulbasaur_gary = Pokemon(species=Pokedex.Bulbasaur, level=10) # Changed from Chimchar to Bulbasaur to match name and type
    potion_gary = Items.Potion

    gary = Trainer(
        name="Gary",
        pokemon_team=[squirtle_gary, bulbasaur_gary],
        inventory={potion_gary: 1},
    )

    return Battle(trainers=(ash, gary), max_rounds=100)


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
    assert battle.trainers[0].name == "Ash"
    assert len(battle.trainers[0].pokemon_team) == 2


def test_play_single_game(agents):
    """Tests running a single battle between two agents."""
    agent_0, agent_1 = agents
    battle = battle_generator()

    winner = play(battle, agent_0, agent_1)

    assert battle.done
    assert winner in [0, 1, 0.5]  # Winner is 0, 1, or 0.5 for a draw


def test_play_single_game_with_mock_agents(mock_agents):
    """Tests a single game with predictable agents to ensure it completes."""
    agent_0, agent_1 = mock_agents
    battle = battle_generator()

    # This test ensures that with predictable agents, the game runs to completion
    # without getting stuck in a loop.
    winner = play(battle, agent_0, agent_1)

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
