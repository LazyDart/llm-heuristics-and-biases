import os
import json
from datetime import datetime
from collections import defaultdict

from src.chatgpt.openai_model import OpenAIModel
from src.chatgpt.openai_evaluator import OpenAIEvaluator


def load_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, output_dir="./results", prefix="results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%Y %H%M%S")
    output_path = os.path.join(output_dir, f"{prefix} {timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_responses(prompts, model_name="gpt-4o", temperature=0, n=10):
    llm = OpenAIModel(model=model_name, temperature=temperature)
    answers = defaultdict(dict)

    for prompt in prompts:
        qid = prompt["id"]
        context = prompt.get("context", "")
        scale = prompt.get("scale", False)
    
        if scale:
            formatted_scale = scale["text"].format(min=scale["min"], max=scale["max"], description=scale["description"])
        else:
            formatted_scale = ""

        if prompt.get("multi", False):
            for idx, subq in enumerate(prompt["subquestions"]):
                question_text = f"{context} {subq['question']} {formatted_scale}"
                key = f"Q{qid}_sub{idx + 1}"
                answers[key] = [llm.generate(question_text) for _ in range(n)]
        else:
            question_text = f"{context} {prompt['question']} {formatted_scale}"
            key = f"Q{qid}"
            answers[key] = [llm.generate(question_text) for _ in range(n)]

    answers["model"] = model_name
    answers["temperature"] = temperature
    return answers


def run_experiment(input_file, models=["gpt-4o"], temperatures=[0], n=10):
    prompts = load_questions(input_file)
    results = {"results": []}

    for model_name in models:
        for temp in temperatures:
            responses = generate_responses(prompts, model_name, temp, n)
            results["results"].append(responses)

    return results


if __name__ == "__main__":
    input_path = "./prompts/survey_questions.json"
    results = run_experiment(input_path, n=1)
    save_results(results, prefix="survey")
