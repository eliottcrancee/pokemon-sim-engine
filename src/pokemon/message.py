# message.py

from dataclasses import dataclass


@dataclass
class Message:
    message: str

    def __str__(self):
        return self.message

    @property
    def content(self):
        return self.message
