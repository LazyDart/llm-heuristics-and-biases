from typing import Protocol


class LLM(Protocol):

    def generate(self, text: str) -> str:
        ...