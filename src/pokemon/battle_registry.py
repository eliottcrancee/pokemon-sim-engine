# src/pokemon/battle_registry.py

from __future__ import annotations

from pokemon.battle import Battle
from pokemon.item import ItemRegistry
from pokemon.move import MoveRegistry
from pokemon.pokemon import Pokemon, SpeciesRegistry
from pokemon.trainer import Trainer


class BattleRegistry:
    """
    A registry to store and retrieve pre-defined Battle scenarios.

    Battles are defined using dictionaries and can be retrieved by name.
    This is useful for creating standardized tests for agents.
    """

    _battles: dict[str, Battle] = {}

    @classmethod
    def _create_trainer_from_def(cls, trainer_def: dict) -> Trainer:
        """Parses a dictionary definition to create a Trainer object."""
        pokemons = []
        team_def = trainer_def.get("team", [])

        for p_def in team_def:
            species = SpeciesRegistry.get(p_def["species"])
            if not species:
                raise ValueError(
                    f"Species '{p_def['species']}' not found in SpeciesRegistry."
                )

            moves = []
            for move_name in p_def.get("moves", []):
                move = MoveRegistry.get(move_name)
                if not move:
                    raise ValueError(f"Move '{move_name}' not found in MoveRegistry.")
                moves.append(move)

            pokemon = Pokemon(
                species=species, level=p_def.get("level", 50), moves=moves
            )
            pokemons.append(pokemon)

        inventory = {}
        inventory_def = trainer_def.get("inventory", {})
        for item_name, quantity in inventory_def.items():
            item = ItemRegistry.get(item_name)
            if not item:
                raise ValueError(f"Item '{item_name}' not found in ItemRegistry.")
            inventory[item] = quantity

        return Trainer(
            name=trainer_def["name"], pokemon_team=pokemons, inventory=inventory
        )

    @classmethod
    def register(cls, name: str, trainer1_def: dict, trainer2_def: dict) -> Battle:
        """
        Registers a new battle scenario from trainer definitions.

        Args:
            name: The unique name for the battle scenario.
            trainer1_def: A dictionary defining the first trainer.
            trainer2_def: A dictionary defining the second trainer.

        Returns:
            The created Battle object template.
        """
        if name in cls._battles:
            raise ValueError(f"Battle with name '{name}' already registered.")

        trainer1 = cls._create_trainer_from_def(trainer1_def)
        trainer2 = cls._create_trainer_from_def(trainer2_def)

        battle = Battle((trainer1, trainer2))
        cls._battles[name] = battle
        return battle

    @classmethod
    def get(cls, name: str) -> Battle | None:
        """
        Gets a fresh copy of a battle scenario by name.

        Args:
            name: The name of the battle to retrieve.

        Returns:
            A copy of the Battle object, or None if not found.
        """
        battle = cls._battles.get(name)
        return battle.copy() if battle else None

    @classmethod
    def list_battles(cls) -> list[str]:
        """Returns a list of all registered battle names."""
        return list(cls._battles.keys())


# --- Pre-defined Battle Scenarios ---

# Kanto Classic: A simple 1v1 between two iconic Pokemon
BattleRegistry.register(
    "kanto_classic",
    trainer1_def={
        "name": "Ash",
        "team": [
            {
                "species": "Pikachu",
                "level": 50,
                "moves": ["Thunderbolt", "Quick Attack", "Tail Whip", "Thunder Wave"],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
    trainer2_def={
        "name": "Gary",
        "team": [
            {
                "species": "Eevee",
                "level": 50,
                "moves": ["Tackle", "Bite", "Sand Attack"],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
)

# Mirror Match 6v6: A perfectly symmetrical battle with full teams.
BattleRegistry.register(
    "mirror_match_6v6",
    trainer1_def={
        "name": "Player 1",
        "team": [
            {
                "species": "Arcanine",
                "level": 75,
                "moves": ["Flamethrower", "Extreme Speed", "Bite"],
            },
            {
                "species": "Gyarados",
                "level": 75,
                "moves": ["Hydro Pump", "Bite", "Taunt"],
            },
            {
                "species": "Exeggutor",
                "level": 75,
                "moves": ["Psychic", "Solar Beam", "Sleep Powder"],
            },
            {
                "species": "Rhydon",
                "level": 75,
                "moves": ["Earthquake", "Rock Slide", "Body Slam"],
            },
            {
                "species": "Alakazam",
                "level": 75,
                "moves": ["Psychic", "Thunder Wave", "Tackle"],
            },
            {
                "species": "Snorlax",
                "level": 75,
                "moves": ["Body Slam", "Hyper Beam", "Pound"],
            },
        ],
        "inventory": {"Hyper Potion": 2, "Full Heal": 2},
    },
    trainer2_def={
        "name": "Player 2",
        "team": [
            {
                "species": "Arcanine",
                "level": 75,
                "moves": ["Flamethrower", "Extreme Speed", "Bite"],
            },
            {
                "species": "Gyarados",
                "level": 75,
                "moves": ["Hydro Pump", "Bite", "Taunt"],
            },
            {
                "species": "Exeggutor",
                "level": 75,
                "moves": ["Psychic", "Solar Beam", "Sleep Powder"],
            },
            {
                "species": "Rhydon",
                "level": 75,
                "moves": ["Earthquake", "Rock Slide", "Body Slam"],
            },
            {
                "species": "Alakazam",
                "level": 75,
                "moves": ["Psychic", "Thunder Wave", "Tackle"],
            },
            {
                "species": "Snorlax",
                "level": 75,
                "moves": ["Body Slam", "Hyper Beam", "Pound"],
            },
        ],
        "inventory": {"Hyper Potion": 2, "Full Heal": 2},
    },
)

# Type Triangle Tussle: A 3v3 battle to test strategic switching.
BattleRegistry.register(
    "type_triangle_tussle",
    trainer1_def={
        "name": "Green",
        "team": [
            {
                "species": "Venosaur",
                "level": 50,
                "moves": ["Solar Beam", "Sleep Powder", "Razor Leaf"],
            },
            {
                "species": "Arcanine",
                "level": 50,
                "moves": ["Flamethrower", "Extreme Speed", "Bite"],
            },
            {
                "species": "Lapras",
                "level": 50,
                "moves": ["Blizzard", "Hydro Pump", "Body Slam"],
            },
        ],
        "inventory": {"Super Potion": 2, "Full Heal": 1},
    },
    trainer2_def={
        "name": "Red",
        "team": [
            {
                "species": "Charizard",
                "level": 50,
                "moves": ["Flamethrower", "Wing Attack", "Slash"],
            },
            {
                "species": "Blastoise",
                "level": 50,
                "moves": ["Hydro Pump", "Skull Bash", "Bite"],
            },
            {
                "species": "Exeggutor",
                "level": 50,
                "moves": ["Psychic", "Solar Beam", "Tackle"],
            },
        ],
        "inventory": {"Super Potion": 2, "Full Heal": 1},
    },
)

# Brain vs. Brawn: A 3v3 battle of Physical vs. Special attackers.
BattleRegistry.register(
    "brain_vs_brawn",
    trainer1_def={
        "name": "Brawn",
        "team": [
            {
                "species": "Rhydon",
                "level": 52,
                "moves": ["Earthquake", "Rock Slide", "Body Slam"],
            },
            {
                "species": "Snorlax",
                "level": 52,
                "moves": ["Body Slam", "Hyper Beam", "Skull Bash"],
            },
            {
                "species": "Gyarados",
                "level": 52,
                "moves": ["Taunt", "Bite", "Hydro Pump"],
            },
        ],
        "inventory": {"Hyper Potion": 2},
    },
    trainer2_def={
        "name": "Brain",
        "team": [
            {
                "species": "Alakazam",
                "level": 52,
                "moves": ["Psychic", "Thunder Wave", "Tackle"],
            },
            {
                "species": "Pikachu",
                "level": 52,
                "moves": ["Thunderbolt", "Thunder Shock", "Quick Attack"],
            },
            {
                "species": "Exeggutor",
                "level": 52,
                "moves": ["Psychic", "Solar Beam", "Tackle"],
            },
        ],
        "inventory": {"Hyper Potion": 2},
    },
)

# Full Team Battle: A 6v6 battle with more complex teams
BattleRegistry.register(
    "full_team_6v6",
    trainer1_def={
        "name": "Red",
        "team": [
            {
                "species": "Pikachu",
                "level": 88,
                "moves": ["Thunderbolt", "Quick Attack", "Iron Tail"],
            },
            {
                "species": "Charizard",
                "level": 84,
                "moves": ["Flamethrower", "Wing Attack", "Slash"],
            },
            {
                "species": "Blastoise",
                "level": 84,
                "moves": ["Hydro Pump", "Skull Bash", "Bite"],
            },
            {
                "species": "Venosaur",
                "level": 84,
                "moves": ["Solar Beam", "Razor Leaf", "Sleep Powder"],
            },
            {
                "species": "Snorlax",
                "level": 82,
                "moves": ["Body Slam", "Hyper Beam", "Pound"],
            },
            {
                "species": "Lapras",
                "level": 80,
                "moves": ["Blizzard", "Body Slam", "Hydro Pump"],
            },
        ],
        "inventory": {"Hyper Potion": 5, "Full Heal": 5},
    },
    trainer2_def={
        "name": "Blue",
        "team": [
            {
                "species": "Pidgeot",
                "level": 85,
                "moves": ["Wing Attack", "Sky Attack", "Tackle"],
            },
            {
                "species": "Alakazam",
                "level": 82,
                "moves": ["Psychic", "Tackle", "Pound"],
            },
            {
                "species": "Rhydon",
                "level": 82,
                "moves": ["Earthquake", "Rock Slide", "Pound"],
            },
            {
                "species": "Gyarados",
                "level": 84,
                "moves": ["Hydro Pump", "Hyper Beam", "Dragon Rage"],
            },
            {
                "species": "Arcanine",
                "level": 84,
                "moves": ["Flamethrower", "Extreme Speed", "Bite"],
            },
            {
                "species": "Exeggutor",
                "level": 86,
                "moves": ["Psychic", "Solar Beam", "Sleep Powder"],
            },
        ],
        "inventory": {"Hyper Potion": 5, "Full Heal": 5},
    },
)

# Mirror Match: A perfectly symmetrical battle for agent evaluation.
BattleRegistry.register(
    "mirror_match",
    trainer1_def={
        "name": "Player 1",
        "team": [
            {
                "species": "Pikachu",
                "level": 50,
                "moves": ["Thunderbolt", "Quick Attack", "Iron Tail", "Thunder Wave"],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
    trainer2_def={
        "name": "Player 2",
        "team": [
            {
                "species": "Pikachu",
                "level": 50,
                "moves": ["Thunderbolt", "Quick Attack", "Iron Tail", "Thunder Wave"],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
)
