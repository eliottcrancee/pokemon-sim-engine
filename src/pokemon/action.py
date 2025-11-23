# action.py

from dataclasses import dataclass

import torch
from pympler import asizeof

from pokemon.config import DEBUG, MAX_POKEMON_PER_TRAINER
from pokemon.item import ITEM_LIST, ITEM_ONE_HOT_DESCRIPTION, Item
from pokemon.message import Message
from pokemon.move import MOVE_ONE_HOT_DESCRIPTION, Move, MoveCategory
from pokemon.tensor_cache import ONEHOTCACHE
from pokemon.trainer import ZEROS_TENSOR_CACHE, Trainer


class ActionError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ActionTypeValue:
    """Class representing a value in the ActionType enum."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.value == other.value

    @property
    def one_hot(self) -> torch.Tensor:
        return ONEHOTCACHE.get_one_hot(3, self.value)


class ActionType:
    """Enum for Action types."""

    ATTACK = ActionTypeValue(0, "Attack")
    SWITCH = ActionTypeValue(1, "Switch")
    USE_ITEM = ActionTypeValue(2, "Use Item")


@dataclass
class Action:
    action_type: ActionTypeValue
    trainer: Trainer
    opponent: Trainer | None = None
    move: Move | None = None
    target_index: int | None = None
    item: Item | None = None

    def __post_init__(self):
        if DEBUG:
            self.validate_inputs()
        self.target = (
            self.trainer.pokemon_team[self.target_index]
            if self.target_index is not None
            else None
        )

    def validate_inputs(self):
        if not isinstance(self.trainer, Trainer):
            raise ActionError("Trainer must be an instance of Trainer")
        if not isinstance(self.action_type, ActionTypeValue):
            raise ActionError("Action type must be a valid ActionTypeValue")
        if self.action_type == ActionType.ATTACK:
            if not isinstance(self.move, Move):
                raise ActionError("Move must be an instance of Move")
            if not isinstance(self.opponent, Trainer):
                raise ActionError("Opponent must be an instance of Trainer")
        elif self.action_type == ActionType.SWITCH:
            if not isinstance(self.target_index, int):
                raise ActionError("Target index must be an integer")
            if self.target_index >= len(self.trainer.pokemon_team):
                raise ActionError(
                    "Target index must be less than the number of Pokemon in the team"
                )
        elif self.action_type == ActionType.USE_ITEM:
            if not isinstance(self.item, Item):
                raise ActionError("Item must be an instance of Item")
            if self.target_index >= len(self.trainer.pokemon_team):
                raise ActionError(
                    "Target index must be less than the number of Pokemon in the team"
                )

    def __repr__(self) -> str:
        return f"Action(action_type={self.action_type}, trainer={self.trainer}, opponent={self.opponent}, move={self.move}, target_index={self.target_index}, item={self.item})"

    def __str__(self) -> str:
        if self.action_type == ActionType.ATTACK:
            return f"Attack using {self.move.name}"
        elif self.action_type == ActionType.SWITCH:
            return f"Switch to {self.trainer.pokemon_team[self.target_index].surname}"
        elif self.action_type == ActionType.USE_ITEM:
            return f"Use {self.item.name} on {self.trainer.pokemon_team[self.target_index].surname}"

    def execute(self) -> list[Message]:
        s = str(self)
        if s:
            s = self.trainer.name + " chooses to " + s[0].lower() + s[1:]
        messages = [Message(s)]
        if self.action_type == ActionType.ATTACK:
            messages += self.execute_move()
        elif self.action_type == ActionType.SWITCH:
            self.trainer.switch_pokemon(self.target)
        elif self.action_type == ActionType.USE_ITEM:
            messages += self.item.use(self.target)
        return messages

    def execute_move(self) -> list[Message]:
        user = self.trainer.pokemon_team[0]
        target = self.opponent.pokemon_team[0]
        user.decrease_pp(self.move.name)
        can_attack, messages = user.can_attack()
        if can_attack:
            damage, base_damage, critical, stab, effectiveness = (
                self.move.calculate_damage(user, target)
            )
            target.hp -= damage
            if effectiveness == 0:
                messages.append(Message("It doesn't affect the target..."))
            elif damage > 0:
                if effectiveness > 1:
                    messages.append(Message("It's super effective!"))
                elif 0 < effectiveness < 1:
                    messages.append(Message("It's not very effective..."))

                if critical:
                    messages.append(Message("Critical hit!"))
                messages.append(Message(f"{target.surname} lost {damage} HP."))
            elif self.move.category != MoveCategory.Status:
                messages.append(Message(f"{user.surname}'s attack missed!"))
            messages += self.move.secondary_effect(
                user, target, damage, base_damage, critical, stab, effectiveness
            )
        return messages

    @property
    def description(self) -> str:
        if self.action_type == ActionType.ATTACK:
            return f"{self.trainer.name}'s {self.trainer.pokemon_team[0].surname} uses {self.move.name} (PP: {self.trainer.pokemon_team[0].get_pp(self.move.name)}/{self.move.pp})."
        elif self.action_type == ActionType.SWITCH:
            return f"{self.trainer.name} switches from {self.trainer.pokemon_team[0].surname} (PV: {self.trainer.pokemon_team[0].hp}/{self.trainer.pokemon_team[0].max_hp}) to {self.target.surname} (PV: {self.target.hp}/{self.target.max_hp})."
        elif self.action_type == ActionType.USE_ITEM:
            return f"{self.trainer.name} uses {self.item.name} (QT: {self.item.quantity}) on {self.target.surname}."

    @property
    def memory_size(self) -> int:
        return asizeof.asizeof(self)

    @property
    def one_hot(self) -> torch.Tensor:
        if self.action_type == ActionType.ATTACK:
            return torch.cat(
                [
                    self.action_type.one_hot,
                    self.move.one_hot,
                    ZEROS_TENSOR_CACHE[MAX_POKEMON_PER_TRAINER],
                    ZEROS_TENSOR_CACHE[len(ITEM_LIST)],
                ]
            )
        elif self.action_type == ActionType.SWITCH:
            return torch.cat(
                [
                    self.action_type.one_hot,
                    ZEROS_TENSOR_CACHE[len(MOVE_ONE_HOT_DESCRIPTION)],
                    ONEHOTCACHE.get_one_hot(MAX_POKEMON_PER_TRAINER, self.target_index),
                    ZEROS_TENSOR_CACHE[len(ITEM_LIST)],
                ]
            )
        elif self.action_type == ActionType.USE_ITEM:
            return torch.cat(
                [
                    self.action_type.one_hot,
                    ZEROS_TENSOR_CACHE[len(MOVE_ONE_HOT_DESCRIPTION)],
                    ZEROS_TENSOR_CACHE[MAX_POKEMON_PER_TRAINER],
                    self.item.one_hot,
                ]
            )

    @property
    def one_hot_description(self) -> torch.Tensor:
        return ACTION_ONE_HOT_DESCRIPTION


ACTION_ONE_HOT_DESCRIPTION = (
    ["Attack", "Switch", "Use Item"]
    + MOVE_ONE_HOT_DESCRIPTION
    + [f"Target Pokemon {i}" for i in range(MAX_POKEMON_PER_TRAINER)]
    + ITEM_ONE_HOT_DESCRIPTION
)
