import json
import glob
import csv
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results")

MODELS = {
    "qwen": {
        "task": "experiment1_qwen_FULL_task*.json",
        "website": "experiment1_qwen_FULL_website*.json",
        "domain": "experiment1_qwen_FULL_domain*.json",
    },
    "gpt-4.1-mini": {
        "task": "experiment1_gpt41mini_FULL_task*.json",
        "website": "experiment1_gpt41mini_FULL_website*.json",
        "domain": "experiment1_gpt41mini_FULL_domain*.json",
    },
    "deepseek-v4-flash": {
        "task": "experiment1_deepseek_v4_flash_FULL_task*.json",
        "website": "experiment1_deepseek_v4_flash_FULL_website*.json",
        "domain": "experiment1_deepseek_v4_flash_FULL_domain*.json",
    },
}

METHODS = ["browseruse", "mindact", "icddp"]
INTENTS = ["content_browsing", "form_selection", "information_input", "button_interaction"]


def latest_file(pattern):
    files = list(RESULT_DIR.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def macro_op_f1(rows, method):
    labels = ["CLICK", "TYPE", "SELECT"]
    pairs = [
        (r.get("operation_type"), r.get(f"{method}_pred_op"))
        for r in rows
        if r.get("operation_type") and r.get(f"{method}_pred_op")
    ]
    if not pairs:
        return None

    scores = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def summarize_rows(rows, method):
    return {
        "steps": len(rows),
        "tokens": mean([r.get(f"{method}_tokens") for r in rows]),
        "prune_rate": mean([r.get(f"{method}_prune_rate") for r in rows]),
        "key_recall": mean([r.get(f"{method}_recall") for r in rows]),
        "prune_time_ms": mean([r.get(f"{method}_process_time_ms") for r in rows]),
        "action_time_ms": mean([r.get(f"{method}_action_time_ms") for r in rows]),
        "total_time_ms": mean([r.get(f"{method}_total_time_ms") for r in rows]),
        "ele_acc": mean([r.get(f"{method}_element_acc") for r in rows]),
        "op_f1": macro_op_f1(rows, method),
        "step_sr": mean([r.get(f"{method}_step_sr") for r in rows]),
        "errors": sum(str(r.get(f"{method}_action", "")).startswith("Error:") for r in rows),
    }


def pct(x):
    return "" if x is None else f"{x * 100:.2f}"


def num(x, digits=1):
    return "" if x is None else f"{x:.{digits}f}"


def add_summary(out, model, split, intent, method, rows):
    s = summarize_rows(rows, method)
    out.append({
        "model": model,
        "split": split,
        "intent": intent,
        "method": method,
        "steps": s["steps"],
        "tokens": num(s["tokens"], 0),
        "prune_rate_%": pct(s["prune_rate"]),
        "key_recall_%": pct(s["key_recall"]),
        "prune_time_ms": num(s["prune_time_ms"], 1),
        "action_time_ms": num(s["action_time_ms"], 1),
        "total_time_ms": num(s["total_time_ms"], 1),
        "ele_acc_%": pct(s["ele_acc"]),
        "op_f1_%": pct(s["op_f1"]),
        "step_sr_%": pct(s["step_sr"]),
        "action_errors": s["errors"],
    })


def print_table(title, rows):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    fields = ["model", "split", "intent", "method", "steps", "tokens", "prune_rate_%", "key_recall_%",
              "ele_acc_%", "op_f1_%", "step_sr_%", "action_errors"]
    print("\t".join(fields))
    for r in rows:
        print("\t".join(str(r.get(f, "")) for f in fields))


def main():
    loaded = {}
    for model, split_patterns in MODELS.items():
        for split, pattern in split_patterns.items():
            path = latest_file(pattern)
            if path is None:
                print(f"[WARN] missing: {model} {split} pattern={pattern}")
                continue
            data = json.load(open(path, encoding="utf-8"))
            rows = data.get("step_results", [])
            loaded[(model, split)] = rows
            print(f"[LOAD] {model:18s} {split:8s} {len(rows):4d} steps <- {path}")

    split_summary = []
    intent_summary = []
    overall_summary = []

    all_by_model = defaultdict(list)

    for (model, split), rows in loaded.items():
        all_by_model[model].extend(rows)

        for method in METHODS:
            add_summary(split_summary, model, split, "ALL", method, rows)

        for intent in INTENTS:
            intent_rows = [r for r in rows if r.get("intent") == intent]
            for method in METHODS:
                add_summary(intent_summary, model, split, intent, method, intent_rows)

    for model, rows in all_by_model.items():
        for method in METHODS:
            add_summary(overall_summary, model, "ALL", "ALL", method, rows)

    print_table("OVERALL: three splits merged", overall_summary)
    print_table("BY SPLIT", split_summary)
    print_table("BY SPLIT + INTENT", intent_summary)

    out_path = RESULT_DIR / "balanced50_summary_all_models.csv"
    rows = overall_summary + split_summary + intent_summary
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV: {out_path}")


if __name__ == "__main__":
    main()
