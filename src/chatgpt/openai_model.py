from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts.string import get_template_variables

from src.model.model import LLM

class OpenAIModel(LLM):

    def __init__(
            self, 
            model: str = "gpt-4o-mini", 
            temperature: int = 0, 
            **kwargs
            ):
        
        self.model = model
        self.llm = ChatOpenAI(
            model=model, 
            temperature=temperature,
            **kwargs
            )

        self.roles = ["system", "assistant", "user"]
        
        self.system_prompt = ""

        self.messages = []

        self.prompt_template = ChatPromptTemplate(self.messages)


    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        system_message = ("system", prompt)
        if self.messages:
            self.messages[0] = system_message
        else:
            self.messages = [system_message]

    def add_messages(self, messages: list[tuple[str, str]]) -> None:
        self.messages += messages

    def generate(self, text: str, add_to_history: bool = False) -> str:

        user_message = [("user", text)]

        llm_answer = self.llm.invoke(self.messages + user_message).content

        if add_to_history:
            self.add_messages(
                user_message 
                + [("assistant", llm_answer)]
            )

        return llm_answer
    
    def set_experimental_template(self, messages: list[tuple[str, str]]) -> None:
        self.prompt_template = ChatPromptTemplate.from_messages(messages)

    
    def generate_with_variables(self, variables: dict[str, str]) -> str:
        
        required_variables = sorted(self.prompt_template.input_variables)

        assert required_variables == sorted(list(variables.keys())), \
            f"Atleast one of {required_variables} required variables is missing."

        chain = self.prompt_template | self.llm

        return chain.invoke(variables).content