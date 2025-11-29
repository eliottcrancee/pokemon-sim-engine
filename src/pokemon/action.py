from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto, unique


@unique
class ActionType(IntEnum):
    """Enum for the different types of actions."""

    ATTACK = auto()
    SWITCH = auto()
    USE_ITEM = auto()
    PASS = auto()


@dataclass(frozen=True, slots=True)
class Action:
    """Class for all actions."""

    action_type: ActionType
    move_slot_index: int | None = None
    pokemon_index: int | None = None
    item_id: int | None = None
    target_index: int | None = None

    def __str__(self) -> str:
        """Return a string representation of the action."""
        if self.action_type == ActionType.ATTACK:
            return f"AttackAction(move_slot_index={self.move_slot_index})"
        if self.action_type == ActionType.SWITCH:
            return f"SwitchAction(pokemon_index={self.pokemon_index})"
        if self.action_type == ActionType.USE_ITEM:
            return f"UseItemAction(item_id={self.item_id}, target_index={self.target_index})"
        if self.action_type == ActionType.PASS:
            return "PassAction()"
        return "UnknownAction()"

    @classmethod
    def create_attack(cls, move_slot_index: int) -> Action:
        """Create an attack action."""
        return cls(action_type=ActionType.ATTACK, move_slot_index=move_slot_index)

    @classmethod
    def create_switch(cls, pokemon_index: int) -> Action:
        """Create a switch action."""
        return cls(action_type=ActionType.SWITCH, pokemon_index=pokemon_index)

    @classmethod
    def create_use_item(cls, item_id: int, target_index: int) -> Action:
        """Create a use item action."""
        return cls(
            action_type=ActionType.USE_ITEM, item_id=item_id, target_index=target_index
        )

    @classmethod
    def create_pass(cls) -> Action:
        """Create a pass action."""
        return cls(action_type=ActionType.PASS)
