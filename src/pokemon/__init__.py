# src/pokemon/__init__.py

"""Pokemon Battle Simulation Engine."""

# Re-export key components for easier access
from . import agents
from .action import Action, ActionError, ActionType
from .battle import Battle
from .loguru_logger import logger
from .move import Move
from .pokemon import Pokemon
from .trainer import Trainer

# Define what gets imported with `from pokemon import *`
__all__ = [
    "Action",
    "ActionError",
    "ActionType",
    "Battle",
    "logger",
    "Move",
    "Pokemon",
    "Trainer",
    "agents",
]
