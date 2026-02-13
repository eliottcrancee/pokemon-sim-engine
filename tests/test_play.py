import pytest

from pokemon.agents import BaseAgent, RandomAgent
from pokemon.battle import Battle
from pokemon.item import Items
from pokemon.move import Moves
from pokemon.play import play, play_multiple
from pokemon.pokemon import Pokedex, Pokemon
from pokemon.trainer import Trainer


def battle_generator():
    shinx_dawn = Pokemon(species=Pokedex.Shinx, level=10)
    chimchar_dawn = Pokemon(species=Pokedex.Chimchar, level=10)
    potion_dawn = Items.Potion

    dawn = Trainer(
        name="Dawn",
        pokemon_team=[shinx_dawn, chimchar_dawn],
        inventory={potion_dawn: 1},
    )

    piplup_lucas = Pokemon(species=Pokedex.Piplup, level=10)
    turtwig_lucas = Pokemon(species=Pokedex.Turtwig, level=10)
    potion_lucas = Items.Potion

    lucas = Trainer(
        name="Lucas",
        pokemon_team=[piplup_lucas, turtwig_lucas],
        inventory={potion_lucas: 1},
    )

    return Battle(trainers=(dawn, lucas), max_rounds=100)


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
    assert battle.trainers[0].name == "Dawn"
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


def test_play_returns_draw_score_on_tie(mock_agents):
    """Tests that play returns 0.5 when battle ends in a tie."""
    agent_0, agent_1 = mock_agents

    dawn = Trainer(
        name="Dawn",
        pokemon_team=[Pokemon(species=Pokedex.Shinx, level=5, moves=[Moves.Growl])],
    )
    lucas = Trainer(
        name="Lucas",
        pokemon_team=[Pokemon(species=Pokedex.Starly, level=5, moves=[Moves.Growl])],
    )
    battle = Battle(trainers=(dawn, lucas), max_rounds=1)

    result = play(battle, agent_0, agent_1)

    assert battle.tie is True
    assert result == 0.5
