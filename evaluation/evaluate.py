import json
from pathlib import Path

from metrics.score import safe_mean


RESULTS_DIR = Path("results")


def load_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_model_name(data: list[dict], path: Path) -> str:
    if data and "model" in data[0]:
        return data[0]["model"]
    return path.stem.replace("_results", "")


def compute_metrics(data: list[dict]) -> dict:
    judges = [x["judge"] for x in data if "judge" in x]

    truthfulness_rate = sum(j["truth_verdict"] == "correct" for j in judges) / len(judges)
    hallucination_rate = sum(j["contains_incorrect_claim"] for j in judges) / len(judges)
    mixed_rate = sum(j["truth_verdict"] == "mixed" for j in judges) / len(judges)
    unclear_rate = sum(j["truth_verdict"] == "unclear" for j in judges) / len(judges)

    mean_truth_score = safe_mean([j["truth_score"] for j in judges])
    mean_best_answer_score = safe_mean([j["best_answer_score"] for j in judges])
    mean_overall_score = safe_mean([j["overall_score"] for j in judges])
    mean_latency = safe_mean([x.get("latency") for x in data])

    return {
        "samples": len(data),
        "truthfulness_rate": round(truthfulness_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "mixed_rate": round(mixed_rate, 4),
        "unclear_rate": round(unclear_rate, 4),
        "mean_truth_score": round(mean_truth_score, 4),
        "mean_best_answer_score": round(mean_best_answer_score, 4),
        "mean_overall_score": round(mean_overall_score, 4),
        "mean_latency": round(mean_latency, 4),
    }


def main():
    result_files = sorted(RESULTS_DIR.glob("*_results.json"))

    if not result_files:
        print("Нет файлов с результатами в папке results/")
        return

    rows = []

    for path in result_files:
        data = load_results(path)
        model_name = infer_model_name(data, path)
        metrics = compute_metrics(data)
        row = {"model": model_name, **metrics}
        rows.append(row)

    for row in rows:
        print()
        for k, v in row.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()