from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Message:
    """
    Immutable container for game log messages.
    Using slots=True for memory efficiency in large battle logs.
    """

    text: str

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"'{self.text}'"
