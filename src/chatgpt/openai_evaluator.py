from logging import warning

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_openai import ChatOpenAI

from src.model.evaluator import Evaluator


class OpenAIEvaluator(Evaluator):

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
        
        self.output_parser: JsonOutputParser = None
        self.output_template: PromptTemplate = None
    

    def set_prompt_template(self, template_string: str) -> None:
        template = PromptTemplate(template=template_string)

        if self.output_parser is not None:
            if "format_instructions" not in template.input_variables:
                warning("Evaluator requires {format_instructions} as variable in prompt." \
                        " Adding them automatically at the end of a prompt.")

                template = PromptTemplate(template = template_string + "\n{format_instructions}")
        
        self.output_template = template
                

    def set_output_format(self, pydantic_model: BaseModel) -> None:
        self.output_parser = JsonOutputParser(pydantic_object=pydantic_model)


    def evaluate(self, input_variables: dict[str, str]) -> dict:
        template = self.output_template.partial(format_instructions=self.output_parser.get_format_instructions())
        chain = template | self.llm | self.output_parser

        # input_dict = {}
        # input_dict.update(input_variables)
        # input_dict.update({"format_instructions": self.output_parser.get_format_instructions()})

        return chain.invoke(input_variables)