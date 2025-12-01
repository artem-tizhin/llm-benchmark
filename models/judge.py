import json
import re
from pathlib import Path

from groq import Groq


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_answers(text: str) -> list[str]:
    return [x.strip() for x in text.split(";") if x.strip()]


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return valid JSON:\n{text}")

    return json.loads(match.group(0))


def judge_answer(
    client: Groq,
    judge_model_name: str,
    judge_prompt: str,
    sample: dict,
    model_answer: str,
) -> dict:
    payload = {
        "Question": sample["Question"],
        "Best Answer": sample["Best Answer"],
        "Correct Answers": split_answers(sample["Correct Answers"]),
        "Incorrect Answers": split_answers(sample["Incorrect Answers"]),
        "Model Answer": model_answer,
    }

    completion = client.chat.completions.create(
        model=judge_model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
    )

    raw_judge_output = completion.choices[0].message.content
    parsed = extract_json(raw_judge_output)

    return parsed