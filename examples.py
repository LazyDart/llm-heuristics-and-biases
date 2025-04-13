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

if __name__ == "__main__":
    example_1()
    # example_2()
