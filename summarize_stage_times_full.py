import csv
import json
from pathlib import Path

FILES = {
    "qwen": {
        "test_cross_task": "results/experiment1_qwen_FULL_task.json",
        "test_cross_website": "results/experiment1_qwen_FULL_website.json",
        "test_cross_domain": "results/experiment1_qwen_FULL_domain.json",
    },
    "gpt-4.1-mini": {
        "test_cross_task": "results/experiment1_gpt41mini_FULL_task.json",
        "test_cross_website": "results/experiment1_gpt41mini_FULL_website.json",
        "test_cross_domain": "results/experiment1_gpt41mini_FULL_domain.json",
    },
    "deepseek-v4-flash": {
        "test_cross_task": "results/experiment1_deepseek_v4_flash_FULL_task.json",
        "test_cross_website": "results/experiment1_deepseek_v4_flash_FULL_website.json",
        "test_cross_domain": "results/experiment1_deepseek_v4_flash_FULL_domain.json",
    },
}

METHODS = {
    "BU-style": "browseruse",
    "MindAct-style": "mindact",
    "TIDP": "icddp",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def get_num(row, key, default=None):
    v = row.get(key)
    return default if v is None else v


def summarize(rows, method_key):
    intent_times = []
    prune_times = []
    action_times = []
    total_times = []

    for r in rows:
        intent = get_num(r, "icddp_intent_time_ms", 0.0) if method_key == "icddp" else 0.0
        prune = get_num(r, f"{method_key}_process_time_ms", 0.0)
        action = get_num(r, f"{method_key}_action_time_ms", 0.0)

        intent_times.append(intent)
        prune_times.append(prune)
        action_times.append(action)
        total_times.append(intent + prune + action)

    return {
        "intent_time_ms": mean(intent_times),
        "prune_time_ms": mean(prune_times),
        "action_time_ms": mean(action_times),
        "total_time_ms": mean(total_times),
        "steps": len(rows),
    }


def main():
    out_rows = []

    print("Model\tSplit\tMethod\tSteps\tIntent Time\tPrune Time\tAction Time\tTotal Time")

    for model, split_files in FILES.items():
        for split, path in split_files.items():
            with open(path, encoding="utf-8") as f:
                rows = json.load(f)["step_results"]

            for method_name, method_key in METHODS.items():
                s = summarize(rows, method_key)

                out = {
                    "model": model,
                    "split": split,
                    "method": method_name,
                    "steps": s["steps"],
                    "intent_time_ms": f"{s['intent_time_ms']:.1f}",
                    "prune_time_ms": f"{s['prune_time_ms']:.1f}",
                    "action_time_ms": f"{s['action_time_ms']:.1f}",
                    "total_time_ms": f"{s['total_time_ms']:.1f}",
                }
                out_rows.append(out)

                print(
                    f"{model}\t{split}\t{method_name}\t{s['steps']}\t"
                    f"{s['intent_time_ms']:.1f}\t{s['prune_time_ms']:.1f}\t"
                    f"{s['action_time_ms']:.1f}\t{s['total_time_ms']:.1f}"
                )

    out_path = Path("results/stage_times_by_model_split.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "method",
                "steps",
                "intent_time_ms",
                "prune_time_ms",
                "action_time_ms",
                "total_time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
