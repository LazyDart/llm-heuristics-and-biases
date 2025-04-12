from src.chatgpt.openai_model import OpenAIModel

if __name__ == "__main__":
    gpt = OpenAIModel()

    gpt.set_system_prompt("You are a helpful AI assistant.")

    gpt.add_messages([("user", "Respond only in french from now on.")])

    print(gpt.generate("What is the meaning of life?"))