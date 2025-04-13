from pydantic import BaseModel, Field

from src.chatgpt.openai_evaluator import OpenAIEvaluator
from src.chatgpt.openai_model import OpenAIModel

def example_1():
    gpt = OpenAIModel()

    gpt.set_system_prompt("You are a helpful AI assistant.")

    gpt.add_messages([("user", "Respond only in french from now on.")])

    print(gpt.generate("What is the meaning of life?"))


def example_2():
    gpt = OpenAIModel()

    example_system_prompt = open("prompts/default_system_gpt4_5.txt", "r").read()
    binary_search_answer = open("prompts/binary_search.txt", "r").read()

    template = [
        ("system", "{system}"),
        ("user", "Please write a {code_task} in python."),
        ("assistant", "{answer}"),
        ("user", "On a scale from 1 to 10, how well do you think your response compares to {number_of_gpts} instances of identical ChatGPTs outputs answering the same prompt? (1 = much worse than average, 10 = much better than average)")
    ]

    gpt.set_experimental_template(template)

    answer = gpt.generate_with_variables({
        "system": example_system_prompt,
        "code_task": "binary search function",
        "answer": binary_search_answer,
        "number_of_gpts": 1000,
        })

    print(answer)

def example_3():
    gpt = OpenAIModel()

    gpt.set_system_prompt("You are a helpful AI assistant.")
    prompt = "Please generate a quicksort algorithm in python."
    answer = gpt.generate(prompt)

    # Evaluator Setup
    class OneToTen(BaseModel):
        gpp_score: int = Field(description="Score from 1 to 5 rating whether code follows good programming practices.")
        efficiency_score: int = Field(description="Score from 1 to 5 on how efficient the code is.")
        total_score: int = Field(description="Score from 1 to 10 rating an answer.")

    gpt_evaluator = OpenAIEvaluator()
    gpt_evaluator.set_output_format(OneToTen)

    gpt_evaluator.set_prompt_template("You will be given an AI generated answer, answering the prompt: {prompt}" \
                                      " your job is to score it using following format: \n{format_instructions}"\
                                      "\nBelow is an AI generated answer to score:\n{answer}")

    eval_score = gpt_evaluator.evaluate({
        "prompt": prompt,
        "answer": answer
    })

    print(eval_score)



if __name__ == "__main__":
    # example_1()
    # example_2()
    example_3()
