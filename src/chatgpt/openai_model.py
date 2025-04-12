from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

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


    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
        system_message = ("system", prompt)
        if self.messages:
            self.messages[0] = system_message
        else:
            self.messages = [system_message]

    def add_messages(self, messages: list[tuple[str, str]]):
        self.messages += messages

    def generate(self, text):
        # TODO: Prompt Template is utilised wrongly here.
        prompt = ChatPromptTemplate.from_messages(self.messages + [("user", text)])

        chain = prompt | self.llm

        return chain.invoke({}).content