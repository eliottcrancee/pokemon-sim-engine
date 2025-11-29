from __future__ import annotations

from enum import IntEnum, unique
from typing import Final, NamedTuple


class TypeConfig(NamedTuple):
    label: str
    weaknesses: set[str] = set()
    resistances: set[str] = set()
    immunities: set[str] = set()


@unique
class PokemonType(IntEnum):
    """
    IntEnum allows these members to be used directly as array indices.
    Value starts at 0 for direct mapping to the cache matrix.
    """

    NONETYPE = 0
    NORMAL = 1
    FIRE = 2
    WATER = 3
    GRASS = 4
    ELECTRIC = 5
    ICE = 6
    FIGHTING = 7
    POISON = 8
    GROUND = 9
    FLYING = 10
    PSYCHIC = 11
    BUG = 12
    ROCK = 13
    GHOST = 14
    DRAGON = 15
    DARK = 16
    STEEL = 17
    FAIRY = 18

    def effectiveness_against(self, defenders: tuple[PokemonType, ...]) -> float:
        """
        Calculates the effectiveness multiplier of this type attacking
        the given defender types.

        Complexity: O(1)
        """
        if len(defenders) == 1:
            return _EFFECTIVENESS_CHART[self][defenders[0]]
        return _EFFECTIVENESS_CHART_3D[self][defenders[0]][defenders[1]]

    @property
    def label(self) -> str:
        return _TYPE_CONFIGS[self].label

    def __repr__(self) -> str:
        return f"<PokemonType.{self.name}>"

    def __str__(self) -> str:
        return self.label.capitalize()


_RAW_CONFIG: dict[PokemonType, TypeConfig] = {
    PokemonType.NONETYPE: TypeConfig("Nonetype"),
    PokemonType.NORMAL: TypeConfig(
        "Normal", weaknesses={"Fighting"}, immunities={"Ghost"}
    ),
    PokemonType.FIRE: TypeConfig(
        "Fire",
        weaknesses={"Water", "Ground", "Rock"},
        resistances={"Grass", "Fire", "Steel", "Ice", "Bug", "Fairy"},
    ),
    PokemonType.WATER: TypeConfig(
        "Water",
        weaknesses={"Grass", "Electric"},
        resistances={"Fire", "Water", "Ice", "Steel"},
    ),
    PokemonType.GRASS: TypeConfig(
        "Grass",
        weaknesses={"Fire", "Ice", "Poison", "Flying", "Bug"},
        resistances={"Water", "Grass", "Electric", "Ground"},
    ),
    PokemonType.ELECTRIC: TypeConfig(
        "Electric",
        weaknesses={"Ground"},
        resistances={"Electric", "Flying", "Steel"},
    ),
    PokemonType.ICE: TypeConfig(
        "Ice",
        weaknesses={"Fire", "Fighting", "Rock", "Steel"},
        resistances={"Ice"},
    ),
    PokemonType.FIGHTING: TypeConfig(
        "Fighting",
        weaknesses={"Flying", "Psychic", "Fairy"},
        resistances={"Bug", "Rock", "Dark"},
    ),
    PokemonType.POISON: TypeConfig(
        "Poison",
        weaknesses={"Ground", "Psychic"},
        resistances={"Grass", "Fighting", "Poison", "Bug", "Fairy"},
    ),
    PokemonType.GROUND: TypeConfig(
        "Ground",
        weaknesses={"Water", "Grass", "Ice"},
        resistances={"Poison", "Rock"},
        immunities={"Electric"},
    ),
    PokemonType.FLYING: TypeConfig(
        "Flying",
        weaknesses={"Electric", "Ice", "Rock"},
        resistances={"Grass", "Fighting", "Bug"},
        immunities={"Ground"},
    ),
    PokemonType.PSYCHIC: TypeConfig(
        "Psychic",
        weaknesses={"Bug", "Ghost", "Dark"},
        resistances={"Fighting", "Psychic"},
    ),
    PokemonType.BUG: TypeConfig(
        "Bug",
        weaknesses={"Fire", "Flying", "Rock"},
        resistances={"Grass", "Fighting", "Ground"},
        immunities={"Psychic"},
    ),
    PokemonType.ROCK: TypeConfig(
        "Rock",
        weaknesses={"Water", "Grass", "Fighting", "Ground", "Steel"},
        resistances={"Normal", "Fire", "Poison", "Flying"},
    ),
    PokemonType.GHOST: TypeConfig(
        "Ghost",
        weaknesses={"Ghost", "Dark"},
        resistances={"Bug", "Poison"},
        immunities={"Normal", "Fighting"},
    ),
    PokemonType.DRAGON: TypeConfig(
        "Dragon",
        weaknesses={"Ice", "Dragon", "Fairy"},
        resistances={"Fire", "Water", "Electric", "Grass"},
    ),
    PokemonType.DARK: TypeConfig(
        "Dark",
        weaknesses={"Fighting", "Bug", "Fairy"},
        resistances={"Ghost", "Dark"},
    ),
    PokemonType.STEEL: TypeConfig(
        "Steel",
        weaknesses={"Fire", "Fighting", "Ground"},
        resistances={
            "Normal",
            "Grass",
            "Ice",
            "Flying",
            "Psychic",
            "Bug",
            "Rock",
            "Ghost",
            "Dragon",
            "Dark",
            "Steel",
            "Fairy",
        },
        immunities={"Poison"},
    ),
    PokemonType.FAIRY: TypeConfig(
        "Fairy",
        weaknesses={"Poison", "Steel"},
        resistances={"Fighting", "Bug", "Dark"},
        immunities={"Dragon"},
    ),
}


def _build_effectiveness_chart() -> tuple[tuple[float, ...], ...]:
    """
    Builds a 2D immutable matrix of float multipliers.
    Dimensions: Attacker x Defender.
    """
    size = len(PokemonType)
    matrix = [[1.0] * size for _ in range(size)]

    for attacker in PokemonType:
        atk_cfg = _RAW_CONFIG[attacker]

        for defender in PokemonType:
            def_cfg = _RAW_CONFIG[defender]
            atk_name = atk_cfg.label

            if atk_name in def_cfg.immunities:
                matrix[attacker][defender] = 0.0
            elif atk_name in def_cfg.weaknesses:
                matrix[attacker][defender] = 2.0
            elif atk_name in def_cfg.resistances:
                matrix[attacker][defender] = 0.5

    return tuple(tuple(row) for row in matrix)


_TYPE_CONFIGS: Final = _RAW_CONFIG
_EFFECTIVENESS_CHART: Final[tuple[tuple[float, ...], ...]] = (
    _build_effectiveness_chart()
)


def _build_effectiveness_chart_3d() -> tuple[tuple[float, ...], ...]:
    """
    Builds a 3D immutable matrix of float multipliers.
    Dimensions: Attacker x Defender1 x Defender2.
    """
    size = len(PokemonType)
    matrix = [[[1.0] * size for _ in range(size)] for _ in range(size)]
    for attacker in PokemonType:
        for defender1 in PokemonType:
            for defender2 in PokemonType:
                multiplier = (
                    _EFFECTIVENESS_CHART[attacker][defender1]
                    * _EFFECTIVENESS_CHART[attacker][defender2]
                )
                matrix[attacker][defender1][defender2] = multiplier
    return tuple(tuple(tuple(row) for row in plane) for plane in matrix)


_EFFECTIVENESS_CHART_3D: Final[tuple[tuple[float, ...], ...]] = (
    _build_effectiveness_chart_3d()
)


if __name__ == "__main__":
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # 1. Prepare Data
    # Convert the tuple-of-tuples to a numpy array for plotting
    data = np.array(_EFFECTIVENESS_CHART)

    # Get labels from the Enum
    labels = [t.label for t in PokemonType]

    # 2. Define a Custom Discrete Colormap
    # 0.0 -> Grey, 0.5 -> Red, 1.0 -> White, 2.0 -> Green
    cmap = colors.ListedColormap(["#7f8c8d", "#e74c3c", "#ecf0f1", "#2ecc71"])
    bounds = [0, 0.1, 0.9, 1.1, 2.1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 3. Setup Plot
    plt.figure(figsize=(14, 12))

    ax = sns.heatmap(
        data,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        norm=norm,
        annot=True,  # Show numbers in cells
        fmt=".1g",  # Format as 0, 0.5, 1, 2 (no trailing zeros)
        cbar=False,  # Hide color bar (we use a legend instead)
        linewidths=0.5,
        linecolor="lightgrey",
        square=True,  # Force cells to be square
    )

    # 4. Aesthetics
    ax.set_title("Pokémon Type Effectiveness Chart", fontsize=18, pad=20)
    ax.set_xlabel("Defender", fontsize=14, labelpad=10)
    ax.set_ylabel("Attacker", fontsize=14, labelpad=10)

    # Move X-axis labels to top for easier reading
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="left")
    plt.yticks(rotation=0)

    # 5. Custom Legend (Since we hid the colorbar)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#7f8c8d", label="0x (Immune)"),
        plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="0.5x (Resist)"),
        plt.Rectangle((0, 0), 1, 1, color="#ecf0f1", label="1x (Neutral)"),
        plt.Rectangle((0, 0), 1, 1, color="#2ecc71", label="2x (Weakness)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.show()
