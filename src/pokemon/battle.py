# battle.py

import random
from dataclasses import dataclass

import torch

from pokemon.action import Action, ActionType
from pokemon.config import DEBUG
from pokemon.message import Message
from pokemon.move import MoveAccessor, MoveCategory
from pokemon.trainer import Trainer


@dataclass
class Battle:
    trainer_0: Trainer
    trainer_1: Trainer
    max_rounds: int = 100
    record_messages: bool = False

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()
        self.reset()

    def copy(self):
        cls = self.__class__
        new_battle = cls.__new__(cls)
        new_battle.trainer_0 = self.trainer_0.copy()
        new_battle.trainer_1 = self.trainer_1.copy()
        new_battle.max_rounds = self.max_rounds
        new_battle.record_messages = self.record_messages
        new_battle.round = self.round
        new_battle.winner = self.winner
        new_battle.tie = self.tie
        new_battle.history = [h for h in self.history]
        return new_battle

    def get_trainer_by_id(self, trainer_id: int) -> Trainer:
        if trainer_id == 0:
            return self.trainer_0
        elif trainer_id == 1:
            return self.trainer_1
        else:
            raise ValueError(f"Invalid trainer ID: {trainer_id}")

    def validate_inputs(self):
        if not isinstance(self.trainer_0, Trainer):
            raise TypeError(
                f"Trainer0 must be a Trainer, not {type(self.trainer_0).__name__}"
            )
        if not isinstance(self.trainer_1, Trainer):
            raise TypeError(
                f"Trainer1 must be a Trainer, not {type(self.trainer_1).__name__}"
            )
        if not isinstance(self.max_rounds, int):
            raise TypeError(
                f"Max rounds must be an int, not {type(self.max_rounds).__name__}"
            )
        if self.max_rounds < 1:
            raise ValueError("Max rounds must be greater than 0")
        if self.trainer_0 == self.trainer_1:
            raise ValueError("Trainers must be unique")

    def __str__(self) -> str:
        return f"Battle {self.trainer_0} VS {self.trainer_1}"

    def reset(self):
        self.round = 0
        self.winner = None
        self.tie = False
        self.history = []
        self.trainer_0.reset()
        self.trainer_1.reset()

    def end(self) -> bool:
        if self.trainer_0.is_defeated and self.trainer_1.is_defeated:
            self.tie = True
            return True
        elif self.trainer_0.is_defeated:
            self.winner = 1
            return True
        elif self.trainer_1.is_defeated:
            self.winner = 0
            return True
        elif self.round >= self.max_rounds:
            self.tie = True
            return True
        return False

    @property
    def done(self) -> bool:
        return self.tie or self.winner is not None

    def check_active_alive(self) -> list[Message]:
        fainted = False
        messages = []
        if not self.trainer_0.pokemon_team[0].is_alive:
            messages.append(
                Message(f"{self.trainer_0.pokemon_team[0].surname} fainted!")
            )
            self.trainer_0.switch_to_lowest_alive_pokemon()
            messages.append(
                Message(
                    f"{self.trainer_0.name} switched to {self.trainer_0.pokemon_team[0].surname}"
                )
            )
            fainted = True
        if not self.trainer_1.pokemon_team[0].is_alive:
            messages.append(
                Message(f"{self.trainer_1.pokemon_team[0].surname} fainted!")
            )
            self.trainer_1.switch_to_lowest_alive_pokemon()
            messages.append(
                Message(
                    f"{self.trainer_1.name} switched to {self.trainer_1.pokemon_team[0].surname}"
                )
            )
            fainted = True
        return fainted, messages

    def turn(self, action_0: Action, action_1: Action):
        if DEBUG:
            if self.done:
                raise ValueError("Cannot take turn after battle is done")

        self.round += 1

        first_action = None

        if (
            action_0.action_type == ActionType.ATTACK
            and action_1.action_type == ActionType.ATTACK
        ):
            if (
                action_0.move == MoveAccessor.QuickAttack
                and action_1.move != MoveAccessor.QuickAttack
            ):
                first_action = 0
            elif (
                action_1.move == MoveAccessor.QuickAttack
                and action_0.move != MoveAccessor.QuickAttack
            ):
                first_action = 1
            else:
                if (
                    self.trainer_0.pokemon_team[0].speed
                    > self.trainer_1.pokemon_team[0].speed
                ):
                    first_action = 0
                elif (
                    self.trainer_0.pokemon_team[0].speed
                    == self.trainer_1.pokemon_team[0].speed
                ):
                    first_action = random.choice([0, 1])
                else:
                    first_action = 1
        elif action_0.action_type == ActionType.ATTACK:
            first_action = 1
        elif action_1.action_type == ActionType.ATTACK:
            first_action = 0
        else:
            first_action = random.choice([0, 1])

        if first_action == 0:
            messages = action_0.execute()
        else:
            messages = action_1.execute()

        if self.end():
            return messages

        fainted, new_messages = self.check_active_alive()
        messages += new_messages
        if fainted:
            return messages

        if first_action == 0:
            messages += action_1.execute()
        else:
            messages += action_0.execute()

        if self.end():
            return messages

        fainted, new_messages = self.check_active_alive()
        messages += new_messages
        if fainted:
            return messages

        for pokemon in [self.trainer_0.pokemon_team[0], self.trainer_1.pokemon_team[0]]:
            messages += pokemon.after_turn()

        if self.end():
            return messages

        _, new_messages = self.check_active_alive()
        messages += new_messages

        return messages

    def get_possible_actions(self, trainer_id: int) -> list[Action]:
        trainer = self.get_trainer_by_id(trainer_id)
        opponent = self.get_trainer_by_id(1 - trainer_id)

        actions = []

        pokemon = trainer.pokemon_team[0]

        for move, pp in zip(pokemon.moves, pokemon._pp):
            if pp > 0:
                if move.category == MoveCategory.Status and pokemon._taunted:
                    pass
                else:
                    actions.append(
                        Action(
                            action_type=ActionType.ATTACK,
                            trainer=trainer,
                            opponent=opponent,
                            move=move,
                        )
                    )

        if not actions:
            actions.append(
                Action(
                    action_type=ActionType.ATTACK,
                    trainer=trainer,
                    opponent=opponent,
                    move=MoveAccessor.Struggle,
                )
            )

        for index, pokemon_in_team in enumerate(trainer.pokemon_team[1:]):
            if pokemon_in_team.is_alive:
                actions.append(
                    Action(
                        action_type=ActionType.SWITCH,
                        trainer=trainer,
                        target_index=index + 1,
                    )
                )

        for item in trainer.inventory.values():
            if item.quantity > 0:
                for index, pokemon in enumerate(trainer.pokemon_team):
                    if item.validate(pokemon):
                        actions.append(
                            Action(
                                action_type=ActionType.USE_ITEM,
                                trainer=trainer,
                                target_index=index,
                                item=item,
                            )
                        )

        return actions

    def tensor(self, trainer_id) -> torch.Tensor:
        return (
            torch.cat(
                [
                    self.trainer_0.tensor,
                    self.trainer_1.tensor,
                ]
            )
            if trainer_id == 0
            else torch.cat(
                [
                    self.trainer_1.tensor,
                    self.trainer_0.tensor,
                ]
            )
        )

    @property
    def tensor_description(self):
        return self.trainer_0.tensor_description * 2
