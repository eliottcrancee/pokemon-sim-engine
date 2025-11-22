# Pokemon Battle Simulation Engine

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A high-performance, extensible Pokemon battle simulation engine written in Python. Designed with Reinforcement Learning (RL) and AI research in mind, it provides a robust environment for training agents, simulating battles, and testing strategies.

## 🚀 Features

*   **Complete Battle Mechanics**: Implements turn-based combat, type effectiveness, move categories (Physical, Special, Status), stat modifiers, and status conditions (Burn, etc.).
*   **AI-Ready**: Built-in support for tensor representations of game state, making it easy to integrate with PyTorch for Deep Reinforcement Learning.
*   **Customizable**: Easily create custom Pokemon, Moves, Items, and Trainers.
*   **Interactive UI**: Includes a command-line interface (CLI) for human-vs-human or human-vs-AI battles.
*   **Performance**: Optimized for speed to facilitate rapid training of AI agents.

## 📦 Installation

You can install the package directly from the source:

```bash
git clone https://github.com/eliottcrancee/pokemon-sim-engine.git
cd pokemon-sim-engine
pip install .
```

Or using `uv` (recommended):

```bash
uv sync
```

## 🎮 Quick Start

### Running a UI Battle

To jump straight into a battle between Ash and Gary:

```bash
uv run python run_ui_battle.py
```

### Programmatic Usage

Here's how to set up a simple battle in your own script:

```python
from pokemon.battle import Battle
from pokemon.pokemon import PokemonAccessor
from pokemon.item import ItemAccessor
from pokemon.trainer import Trainer
# from pokemon.agent import RandomAgent # Assuming you have a RandomAgent or similar

# 1. Create Pokemon
pikachu = PokemonAccessor.Pikachu(level=50)
charmander = PokemonAccessor.Chimchar(level=50) # Using Chimchar as in your example

# 2. Create Trainers
ash = Trainer(
    name="Ash",
    pokemon_team=[pikachu],
    inventory={ItemAccessor.Potion: 2}
)

gary = Trainer(
    name="Gary",
    pokemon_team=[charmander],
    inventory={ItemAccessor.Potion: 2}
)

# 3. Initialize Battle
battle = Battle(trainer_0=ash, trainer_1=gary, max_rounds=100)

# 4. Run Battle Loop (Simplified)
while not battle.done:
    # Get actions from agents (implementation depends on your Agent class)
    # action_0 = agent_0.act(battle.get_state(0))
    # action_1 = agent_1.act(battle.get_state(1))
    
    # Execute turn
    # battle.step(action_0, action_1)
    pass

print(f"Winner: {battle.winner}")
```

## 🏗️ Project Structure

*   `src/pokemon/`: Core package source code.
    *   `battle.py`: Main battle logic and state management.
    *   `pokemon.py`: Pokemon data structure, stats, and mechanics.
    *   `move.py`: Move definitions and damage calculations.
    *   `trainer.py`: Trainer state (team, inventory).
    *   `agent.py`: Base classes for AI and human agents.
    *   `ui.py`: CLI interface for interactive battles.
*   `tests/`: Comprehensive test suite using `pytest`.
*   `run_ui_battle.py`: Entry point for the demo battle.

## 🧪 Development

To run the tests and ensure everything is working correctly:

```bash
uv run pytest
```

To run benchmarks:

```bash
uv run pytest --benchmark-only
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
