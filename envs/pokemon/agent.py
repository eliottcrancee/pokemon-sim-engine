# agent.py

import os
import random
import sys

import torch
from colorama import Fore

# Ensure current working directory is in path
sys.path.append(os.getcwd())

from envs.pokemon.action import Action
from envs.pokemon.battle import Battle

# Device global
DEVICE = torch.device("cpu")


class BaseAgent:
    def __init__(self, name):
        self.name = name

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        action = random.choice(actions)
        return action


class InputAgent(BaseAgent):
    def __init__(self, name="InputAgent"):
        super().__init__(name)

    def get_action(
        self, battle: Battle, trainer_id: int, verbose: bool = False
    ) -> list[Action]:
        actions = battle.get_possible_actions(trainer_id)
        return actions


def get_color(value):
    if value < 0 or value > 1:
        return Fore.WHITE  # Default to white if out of bounds
    red = int((1 - value) * 255)
    green = int(value * 255)
    color = f"\033[38;2;{red};{green};0m"
    return color
