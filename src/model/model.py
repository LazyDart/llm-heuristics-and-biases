from typing import Protocol


class LLM(Protocol):

    def set_system_prompt(self, prompt: str) -> None:
        """
        Sets system prompt used in this model.
        """
        ...

    def add_messages(self, messages: list[tuple[str, str]]) -> None:
        """
        Adds messages to context.
        """
        ...

    def generate(self, text: str, add_to_history: bool = False) -> str:
        """
        Generates based on:
        1. self.messages including currently set system prompt.
        2. ('user', text) pair, where text is input to this function. 
        
        Arguments:
        text: str - user message based on which llm answer will be added.
        add_to_history: bool - whether to add users and gpts messages to self.messages.
        """
        ...

    def set_experimental_template(self, messages: list[tuple[str, str]]) -> None:
        """
        Template should include system prompt.
        """
        ...
    
    def generate_with_variables(self, variables: dict[str, str]) -> str:
        """
        Uses self.prompt_template filled out with values from 'variables' arg.
        Returns contents generated based on provided context as strings.
        """
        ...