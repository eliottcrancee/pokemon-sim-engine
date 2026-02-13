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

    def __post_init__(self) -> None:
        """Validate the action payload against its type.

        This prevents constructing invalid actions such as ATTACK with
        move_slot_index=None, which would otherwise become runtime errors
        when used as list indices.
        """

        match self.action_type:
            case ActionType.ATTACK:
                if self.move_slot_index is None:
                    raise ValueError("ATTACK action requires move_slot_index")
                if (
                    self.pokemon_index is not None
                    or self.item_id is not None
                    or self.target_index is not None
                ):
                    raise ValueError(
                        "ATTACK action cannot have pokemon_index/item_id/target_index"
                    )
            case ActionType.SWITCH:
                if self.pokemon_index is None:
                    raise ValueError("SWITCH action requires pokemon_index")
                if (
                    self.move_slot_index is not None
                    or self.item_id is not None
                    or self.target_index is not None
                ):
                    raise ValueError(
                        "SWITCH action cannot have move_slot_index/item_id/target_index"
                    )
            case ActionType.USE_ITEM:
                if self.item_id is None or self.target_index is None:
                    raise ValueError(
                        "USE_ITEM action requires item_id and target_index"
                    )
                if self.move_slot_index is not None or self.pokemon_index is not None:
                    raise ValueError(
                        "USE_ITEM action cannot have move_slot_index/pokemon_index"
                    )
            case ActionType.PASS:
                if (
                    self.move_slot_index is not None
                    or self.pokemon_index is not None
                    or self.item_id is not None
                    or self.target_index is not None
                ):
                    raise ValueError(
                        "PASS action cannot have move_slot_index/pokemon_index/item_id/target_index"
                    )
            case _:
                raise ValueError(f"Unknown action type: {self.action_type}")

    @property
    def attack_move_slot_index(self) -> int:
        if self.action_type != ActionType.ATTACK or self.move_slot_index is None:
            raise ValueError("Not an ATTACK action")
        return self.move_slot_index

    @property
    def switch_pokemon_index(self) -> int:
        if self.action_type != ActionType.SWITCH or self.pokemon_index is None:
            raise ValueError("Not a SWITCH action")
        return self.pokemon_index

    @property
    def use_item_id(self) -> int:
        if self.action_type != ActionType.USE_ITEM or self.item_id is None:
            raise ValueError("Not a USE_ITEM action")
        return self.item_id

    @property
    def use_item_target_index(self) -> int:
        if self.action_type != ActionType.USE_ITEM or self.target_index is None:
            raise ValueError("Not a USE_ITEM action")
        return self.target_index

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
