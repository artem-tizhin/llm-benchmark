import os

from dotenv import load_dotenv
from groq import Groq

from models.benchmark_runner import load_prompt, load_questions, run_benchmark


load_dotenv()

api_key = os.getenv("GROQ_API_KEY3")
client = Groq(api_key=api_key)

LLAMA_MODEL_NAME = "llama-3.1-8b-instant"
JUDGE_MODEL_NAME = "openai/gpt-oss-120b"


def main():
    questions = load_questions("dataset/questions.jsonl")
    answer_prompt = load_prompt("prompts/answer_prompt.txt")
    judge_prompt = load_prompt("prompts/judge_prompt.txt")

    llama_results = run_benchmark(
        client=client,
        model_name=LLAMA_MODEL_NAME,
        judge_model_name=JUDGE_MODEL_NAME,
        questions=questions,
        answer_prompt=answer_prompt,
        judge_prompt=judge_prompt,
        output_path="results/llama_8B_results.json",
    )

    return llama_results


if __name__ == "__main__":
    main()