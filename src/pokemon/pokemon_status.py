from enum import IntEnum, unique


@unique
class PokemonStatus(IntEnum):
    HEALTHY = 0
    BURN = 1
    FREEZE = 2
    PARALYSIS = 3
    POISON = 4
    SLEEP = 5
    FAINTED = 99
