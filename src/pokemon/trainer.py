# trainer.py

from __future__ import annotations

from pokemon.item import Item, ItemRegistry
from pokemon.pokemon import Pokemon


class TrainerError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Trainer:
    """
    Represents a Pokémon trainer, optimized for performance in RL environments.
    - Uses __slots__ to reduce memory footprint.
    - Inventory is stored as a fixed-size list of integers for O(1) lookups,
      where the index corresponds to the item's ID in the ItemRegistry.
    """

    __slots__ = (
        "name",
        "pokemon_team",
        "_inventory_quantities",
        "_initial_inventory_quantities",  # Added
    )

    def __init__(
        self,
        name: str,
        pokemon_team: list[Pokemon],
        inventory: dict[Item, int] | None = None,
    ):
        """
        Initializes a Trainer.
        Args:
            name: The name of the trainer.
            pokemon_team: A list of up to 6 Pokemon objects.
            inventory: A dictionary mapping Item objects to their quantities.
                       This is converted to a more efficient internal representation.


        """
        self.name = name
        self.pokemon_team = pokemon_team
        max_item_id = len(ItemRegistry.all())
        self._inventory_quantities = [0] * max_item_id
        if inventory:
            for item, quantity in inventory.items():
                if 0 <= item.id < max_item_id:
                    self._inventory_quantities[item.id] = quantity
        self._initial_inventory_quantities = self._inventory_quantities.copy()

    def get_item_quantity(self, item: Item) -> int:
        """Efficiently gets the quantity of a given item."""
        try:
            return self._inventory_quantities[item.id]
        except IndexError:
            return 0

    def get_possessed_items(self) -> list[tuple[Item, int]]:
        """
        Returns a list of items the trainer actually possesses.
        Useful for generating valid action spaces in RL.
        """
        return [
            (ItemRegistry.get(i), qty)
            for i, qty in enumerate(self._inventory_quantities)
            if qty > 0
        ]

    def decrease_item_quantity(self, item: Item):
        """Decrements the quantity of a given item."""
        if self._inventory_quantities[item.id] <= 0:
            raise TrainerError(f"No {item.name} left to use.")
        self._inventory_quantities[item.id] -= 1

    def copy(self) -> Trainer:
        """
        Creates a deep, efficient copy of the Trainer instance.
        Bypasses __init__ for performance.
        """

        cls = self.__class__
        new_trainer = cls.__new__(cls)
        new_trainer.name = self.name
        new_trainer.pokemon_team = [p.copy() for p in self.pokemon_team]
        new_trainer._inventory_quantities = self._inventory_quantities.copy()
        new_trainer._initial_inventory_quantities = (
            self._initial_inventory_quantities.copy()
        )

        return new_trainer

    def switch_pokemon(self, new_pokemon_index: int):
        """
        Switches the active Pokémon (at index 0) with another Pokémon in the team.
        Args:
            new_pokemon_index: The index of the Pokémon to switch to.
        """
        if not (0 < new_pokemon_index < len(self.pokemon_team)):
            raise TrainerError(f"Invalid Pokémon index for switch: {new_pokemon_index}")
        if not self.pokemon_team[new_pokemon_index].is_alive:
            raise TrainerError("Cannot switch to a fainted Pokémon.")

        # Restore status of the currently active pokemon before switching
        self.pokemon_team[0].restore()

        # Perform the swap
        (
            self.pokemon_team[0],
            self.pokemon_team[new_pokemon_index],
        ) = (
            self.pokemon_team[new_pokemon_index],
            self.pokemon_team[0],
        )

    @property
    def active_pokemon(self) -> Pokemon:
        """Returns the currently active Pokémon."""
        return self.pokemon_team[0]

    @property
    def is_defeated(self) -> bool:
        """Returns True if all Pokémon on the team are fainted."""
        return not any(p.is_alive for p in self.pokemon_team)

    @property
    def lowest_alive_pokemon(self) -> Pokemon | None:
        """Returns the first alive pokemon in the team, or None if all are fainted."""
        for pokemon in self.pokemon_team:
            if pokemon.is_alive:
                return pokemon
        return None

    def reset(self):
        """Resets the trainer's state (Pokemon health and inventory) to initial values."""
        for p in self.pokemon_team:
            p.reset()
        self._inventory_quantities = self._initial_inventory_quantities.copy()

    def __repr__(self) -> str:
        return f"Trainer(name={self.name}, pokemons={self.pokemon_team})"

    def __str__(self) -> str:
        return self.name

    def describe(self) -> str:
        """Returns a detailed multi-line string description of the trainer."""
        lines = [f"{self.name}'s Team:"]
        for i, p in enumerate(self.pokemon_team):
            status = "Alive" if p.is_alive else "Fainted"
            lines.append(f"  [{i}] {p.name} ({status})")

        lines.append(f"{self.name}'s Inventory:")
        items = self.get_possessed_items()
        if items:
            for item, qty in items:
                lines.append(f"  - {item.name}: {qty}")
        else:
            lines.append("  (Empty)")

        return "\n".join(lines)

    def __eq__(self, other):
        if not isinstance(other, Trainer):
            return NotImplemented
        return (
            self.pokemon_team == other.pokemon_team
            and self._inventory_quantities == other._inventory_quantities
        )

    def __hash__(self):
        return hash((tuple(self.pokemon_team), tuple(self._inventory_quantities)))
