import csv
import json
from pathlib import Path

FILES = {
    "test_cross_task": "results/experiment1_qwen_FULL_task.json",
    "test_cross_website": "results/experiment1_qwen_FULL_website.json",
    "test_cross_domain": "results/experiment1_qwen_FULL_domain.json",
}

METHODS = {
    "BU-style": "browseruse",
    "MindAct-style": "mindact",
    "TIDP": "icddp",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows, method):
    tokens = []
    pruning_rates = []

    for r in rows:
        raw_tokens = r.get("original_tokens")
        kept_tokens = r.get(f"{method}_tokens")

        if raw_tokens and kept_tokens is not None:
            tokens.append(kept_tokens)
            pruning_rates.append(1 - kept_tokens / raw_tokens)

    return mean(tokens), mean(pruning_rates), len(pruning_rates)


def main():
    out_rows = []

    print("Split\tMethod\tSteps\tTokens\tPruning Rate (%)")

    for split, path in FILES.items():
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)["step_results"]

        for method_name, method_key in METHODS.items():
            avg_tokens, avg_pruning_rate, steps = summarize(rows, method_key)

            out_rows.append({
                "split": split,
                "method": method_name,
                "steps": steps,
                "tokens": f"{avg_tokens:.0f}",
                "pruning_rate_%": f"{avg_pruning_rate * 100:.2f}",
            })

            print(
                f"{split}\t{method_name}\t{steps}\t"
                f"{avg_tokens:.0f}\t{avg_pruning_rate * 100:.2f}"
            )

    out = Path("results/tokens_pruning_rate_by_split.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "method", "steps", "tokens", "pruning_rate_%"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
