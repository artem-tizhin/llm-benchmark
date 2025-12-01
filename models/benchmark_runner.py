import json
import time
from pathlib import Path

from groq import Groq

from models.judge import judge_answer


def load_questions(path: str, limit: int | None = None) -> list[dict]:
    questions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

            if limit is not None and len(questions) >= limit:
                break

    return questions


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return (
        "rate limit" in error_text
        or "429" in error_text
        or "rate_limit_exceeded" in error_text
    )


def split_answers(text: str) -> list[str]:
    return [x.strip() for x in text.split(";") if x.strip()]


def ask_model(
    client: Groq,
    model_name: str,
    prompt: str,
    question: str,
) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content


def load_existing_results(output_path: str) -> list[dict]:
    output_file = Path(output_path)

    if not output_file.exists():
        return []

    text = output_file.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Файл {output_file} повреждён или невалидный JSON. Начинаю заново.")
        return []

    if not isinstance(data, list):
        print(f"Файл {output_file} имеет неверный формат. Начинаю заново.")
        return []

    return data


def should_restart_if_completed(output_path: str) -> bool:
    answer = input(
        f"Файл {output_path} уже содержит результаты для всех вопросов. "
        f"Начать заново? [y/n]: "
    ).strip().lower()

    return answer in {"y", "yes", "д", "да"}


def prepare_results_state(
    questions: list[dict],
    output_path: str,
) -> tuple[list[dict], int]:
    existing_results = load_existing_results(output_path)
    total_questions = len(questions)
    completed = len(existing_results)

    if completed == 0:
        print("Существующих результатов нет. Начинаю с первого вопроса.")
        return [], 0

    if completed < total_questions:
        print(
            f"Найден частично заполненный файл результатов: "
            f"{completed}/{total_questions}. Продолжаю с вопроса {completed + 1}."
        )
        return existing_results, completed

    if completed == total_questions:
        if should_restart_if_completed(output_path):
            print("Перезапускаю бенчмарк с начала.")
            return [], 0

        print("Ничего не делаю: все вопросы уже обработаны.")
        return existing_results, total_questions

    print(
        f"В файле результатов объектов больше, чем вопросов в датасете "
        f"({completed} > {total_questions}). Начинаю заново."
    )
    return [], 0


def save_results(results: list[dict], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_benchmark(
    client: Groq,
    model_name: str,
    judge_model_name: str,
    questions: list[dict],
    answer_prompt: str,
    judge_prompt: str,
    output_path: str,
) -> list[dict]:
    results, start_idx = prepare_results_state(
        questions=questions,
        output_path=output_path,
    )

    if start_idx >= len(questions):
        return results

    total_questions = len(questions)

    for question_idx in range(start_idx, total_questions):
        q = questions[question_idx]
        display_idx = question_idx + 1

        try:
            start = time.time()

            raw_output = ask_model(
                client=client,
                model_name=model_name,
                prompt=answer_prompt,
                question=q["Question"],
            )

            latency = time.time() - start

            judge_result = judge_answer(
                client=client,
                judge_model_name=judge_model_name,
                judge_prompt=judge_prompt,
                sample=q,
                model_answer=raw_output,
            )

            results.append({
                "question_id": display_idx,
                "type": q["Type"],
                "category": q["Category"],
                "question": q["Question"],
                "best_answer": q["Best Answer"],
                "correct_answers": split_answers(q["Correct Answers"]),
                "incorrect_answers": split_answers(q["Incorrect Answers"]),
                "raw_output": raw_output,
                "judge": judge_result,
                "latency": latency,
                "model": model_name,
            })

            print(
                f"[{display_idx}/{total_questions}] "
                f"{model_name} | "
                f"verdict={judge_result.get('truth_verdict')} | "
                f"overall={judge_result.get('overall_score')}"
            )

            save_results(results, output_path)

        except Exception as e:
            if is_rate_limit_error(e):
                print(f"\nДостигнут лимит запросов. Останавливаюсь на вопросе {display_idx}.")
                save_results(results, output_path)
                break

            print(f"\nОшибка на вопросе {display_idx}: {e}")
            save_results(results, output_path)
            break

    print(f"\nСохранено результатов: {len(results)}")
    print(f"Результаты сохранены в: {output_path}")

    return results