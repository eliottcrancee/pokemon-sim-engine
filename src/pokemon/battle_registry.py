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
        """Registers a new battle scenario from trainer definitions."""
        if name in cls._battles:
            raise ValueError(f"Battle with name '{name}' already registered.")

        trainer1 = cls._create_trainer_from_def(trainer1_def)
        trainer2 = cls._create_trainer_from_def(trainer2_def)

        battle = Battle((trainer1, trainer2))
        cls._battles[name] = battle
        return battle

    @classmethod
    def get(cls, name: str) -> Battle | None:
        """Gets a fresh copy of a battle scenario by name."""
        battle = cls._battles.get(name)
        return battle.copy() if battle else None

    @classmethod
    def list_battles(cls) -> list[str]:
        """Returns a list of all registered battle names."""
        return list(cls._battles.keys())


# --- Pre-defined Battle Scenarios (Gen 4 focused) ---

# Early-route style duel with status pressure and priority finisher.
BattleRegistry.register(
    "sinnoh_opening_sparks",
    trainer1_def={
        "name": "Dawn",
        "team": [
            {
                "species": "Shinx",
                "level": 18,
                "moves": ["Spark", "Bite", "Thunder Wave", "Quick Attack"],
            }
        ],
        "inventory": {"Potion": 1},
    },
    trainer2_def={
        "name": "Lucas",
        "team": [
            {
                "species": "Starly",
                "level": 18,
                "moves": ["Wing Attack", "Quick Attack", "Tackle", "Growl"],
            }
        ],
        "inventory": {"Potion": 1},
    },
)

# Starter final forms in a triangle matchup; switching and coverage are key.
BattleRegistry.register(
    "sinnoh_starter_triangle",
    trainer1_def={
        "name": "Barry",
        "team": [
            {
                "species": "Infernape",
                "level": 58,
                "moves": ["Flare Blitz", "Mach Punch", "U-turn", "Low Kick"],
            },
            {
                "species": "Empoleon",
                "level": 58,
                "moves": ["Hydro Pump", "Flash Cannon", "Ice Beam", "Aqua Jet"],
            },
            {
                "species": "Torterra",
                "level": 58,
                "moves": ["Wood Hammer", "Earthquake", "Crunch", "Solar Beam"],
            },
        ],
        "inventory": {"Super Potion": 2, "Full Heal": 1},
    },
    trainer2_def={
        "name": "Rival",
        "team": [
            {
                "species": "Torterra",
                "level": 58,
                "moves": ["Earthquake", "Wood Hammer", "Crunch", "Razor Leaf"],
            },
            {
                "species": "Infernape",
                "level": 58,
                "moves": ["Flamethrower", "Mach Punch", "U-turn", "Extreme Speed"],
            },
            {
                "species": "Empoleon",
                "level": 58,
                "moves": ["Hydro Pump", "Ice Beam", "Flash Cannon", "Bubble"],
            },
        ],
        "inventory": {"Super Potion": 2, "Full Heal": 1},
    },
)

# Early-game tactical battle with simple pivoting and status pressure.
BattleRegistry.register(
    "midgame_pivot_pressure",
    trainer1_def={
        "name": "Ace Trainer Maylene",
        "team": [
            {
                "species": "Shinx",
                "level": 14,
                "moves": ["Tackle", "Bite", "Thunder Wave", "Quick Attack"],
            },
            {
                "species": "Starly",
                "level": 13,
                "moves": ["Tackle", "Quick Attack", "Wing Attack", "Growl"],
            },
            {
                "species": "Piplup",
                "level": 13,
                "moves": ["Pound", "Growl", "Bubble", "Water Gun"],
            },
        ],
        "inventory": {"Potion": 2},
    },
    trainer2_def={
        "name": "Ace Trainer Volkner",
        "team": [
            {
                "species": "Turtwig",
                "level": 14,
                "moves": ["Tackle", "Withdraw", "Absorb", "Razor Leaf"],
            },
            {
                "species": "Starly",
                "level": 13,
                "moves": ["Tackle", "Quick Attack", "Wing Attack", "Growl"],
            },
            {
                "species": "Shinx",
                "level": 13,
                "moves": ["Tackle", "Bite", "Thunder Wave", "Quick Attack"],
            },
        ],
        "inventory": {"Potion": 2},
    },
)

# Physical offense versus special offense style clash.
BattleRegistry.register(
    "brain_vs_brawn_gen4",
    trainer1_def={
        "name": "Brawn",
        "team": [
            {
                "species": "Rhyperior",
                "level": 52,
                "moves": ["Earthquake", "Rock Slide", "Stone Edge", "Megahorn"],
            },
            {
                "species": "Garchomp",
                "level": 52,
                "moves": ["Dragon Rush", "Earthquake", "Crunch", "Take Down"],
            },
            {
                "species": "Staraptor",
                "level": 52,
                "moves": ["Brave Bird", "Quick Attack", "U-turn", "Take Down"],
            },
        ],
        "inventory": {"Hyper Potion": 2},
    },
    trainer2_def={
        "name": "Brain",
        "team": [
            {
                "species": "Lucario",
                "level": 52,
                "moves": [
                    "Aura Sphere",
                    "Dragon Pulse",
                    "Flash Cannon",
                    "Extreme Speed",
                ],
            },
            {
                "species": "Magnezone",
                "level": 52,
                "moves": ["Thunderbolt", "Flash Cannon", "Tri Attack", "Thunder"],
            },
            {
                "species": "Togekiss",
                "level": 52,
                "moves": ["Air Slash", "Aura Sphere", "Nasty Plot", "Thunder Wave"],
            },
        ],
        "inventory": {"Hyper Potion": 2},
    },
)

# Champion-level full teams with mixed offensive/defensive options.
BattleRegistry.register(
    "sinnoh_champion_clash_6v6",
    trainer1_def={
        "name": "Cynthia",
        "team": [
            {
                "species": "Garchomp",
                "level": 88,
                "moves": ["Dragon Rush", "Earthquake", "Crunch", "Sand Attack"],
            },
            {
                "species": "Infernape",
                "level": 84,
                "moves": ["Flare Blitz", "Mach Punch", "U-turn", "Low Kick"],
            },
            {
                "species": "Empoleon",
                "level": 84,
                "moves": ["Hydro Pump", "Flash Cannon", "Ice Beam", "Aqua Jet"],
            },
            {
                "species": "Torterra",
                "level": 84,
                "moves": ["Wood Hammer", "Earthquake", "Crunch", "Solar Beam"],
            },
            {
                "species": "Lucario",
                "level": 82,
                "moves": ["Aura Sphere", "Dragon Pulse", "Extreme Speed", "Mach Punch"],
            },
            {
                "species": "Togekiss",
                "level": 80,
                "moves": ["Air Slash", "Aura Sphere", "Nasty Plot", "Thunder Wave"],
            },
        ],
        "inventory": {"Hyper Potion": 5, "Full Heal": 4},
    },
    trainer2_def={
        "name": "Palmer",
        "team": [
            {
                "species": "Staraptor",
                "level": 85,
                "moves": ["Brave Bird", "Quick Attack", "U-turn", "Take Down"],
            },
            {
                "species": "Magnezone",
                "level": 82,
                "moves": ["Thunderbolt", "Flash Cannon", "Tri Attack", "Thunder"],
            },
            {
                "species": "Rhyperior",
                "level": 82,
                "moves": ["Earthquake", "Rock Slide", "Stone Edge", "Megahorn"],
            },
            {
                "species": "Weavile",
                "level": 84,
                "moves": ["Ice Shard", "Slash", "Bite", "Low Kick"],
            },
            {
                "species": "Luxray",
                "level": 84,
                "moves": ["Wild Charge", "Bite", "Thunderbolt", "Body Slam"],
            },
            {
                "species": "Dialga",
                "level": 86,
                "moves": ["Draco Meteor", "Flash Cannon", "Fire Blast", "Rock Slide"],
            },
        ],
        "inventory": {"Hyper Potion": 5, "Full Heal": 4},
    },
)

# Box-legend flavored battle with high-power special pressure.
BattleRegistry.register(
    "creation_trio_showdown",
    trainer1_def={
        "name": "Spear Pillar",
        "team": [
            {
                "species": "Dialga",
                "level": 72,
                "moves": ["Draco Meteor", "Flash Cannon", "Fire Blast", "Thunder"],
            },
            {
                "species": "Palkia",
                "level": 72,
                "moves": ["Spacial Rend", "Hydro Pump", "Fire Blast", "Thunder"],
            },
            {
                "species": "Giratina",
                "level": 72,
                "moves": ["Shadow Claw", "Draco Meteor", "Aura Sphere", "Will-O-Wisp"],
            },
        ],
        "inventory": {"Hyper Potion": 2, "Full Heal": 2},
    },
    trainer2_def={
        "name": "Hall of Origin",
        "team": [
            {
                "species": "Arceus",
                "level": 72,
                "moves": ["Judgment", "Extreme Speed", "Swords Dance", "Shadow Claw"],
            },
            {
                "species": "Lucario",
                "level": 70,
                "moves": ["Aura Sphere", "Dragon Pulse", "Extreme Speed", "Mach Punch"],
            },
            {
                "species": "Garchomp",
                "level": 70,
                "moves": ["Dragon Rush", "Earthquake", "Crunch", "Sand Attack"],
            },
        ],
        "inventory": {"Hyper Potion": 2, "Full Heal": 2},
    },
)

# Symmetric duel for deterministic agent-vs-agent comparison.
BattleRegistry.register(
    "lucario_mirror_duel",
    trainer1_def={
        "name": "Player 1",
        "team": [
            {
                "species": "Lucario",
                "level": 50,
                "moves": [
                    "Aura Sphere",
                    "Extreme Speed",
                    "Dragon Pulse",
                    "Thunder Wave",
                ],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
    trainer2_def={
        "name": "Player 2",
        "team": [
            {
                "species": "Lucario",
                "level": 50,
                "moves": [
                    "Aura Sphere",
                    "Extreme Speed",
                    "Dragon Pulse",
                    "Thunder Wave",
                ],
            }
        ],
        "inventory": {"Super Potion": 1},
    },
)
