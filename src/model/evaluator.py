from typing import Protocol

from pydantic import BaseModel

class Evaluator(Protocol):

    def set_prompt_template(self, template_string: str) -> None:
        ...

    def set_output_format(self, pydantic_model: BaseModel) -> None:
        ...

    def evaluate(self, input_variables: dict[str, str]) -> dict:
        ...
    