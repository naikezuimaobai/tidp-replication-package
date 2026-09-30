import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "" if x is None else f"{x * 100:.2f}"


def num(x, digits=1):
    return "" if x is None else f"{x:.{digits}f}"


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def token_f1(pred, gold):
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    gold_counts = defaultdict(int)
    for t in gold_toks:
        gold_counts[t] += 1

    common = 0
    for t in pred_toks:
        if gold_counts[t] > 0:
            common += 1
            gold_counts[t] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def parse_action(action):
    raw = (action or "").strip()
    if not raw or raw == "SKIPPED" or raw.startswith("Error:"):
        return "", ""

    op_match = re.search(r"\b(CLICK|TYPE|SELECT)\b", raw, re.I)
    op = op_match.group(1).upper() if op_match else ""

    value = ""
    if op in {"TYPE", "SELECT"}:
        after = raw[op_match.end():] if op_match else raw
        after = re.sub(r"\[\d+\]", "", after)
        value = after.strip(" :-\"'")

    return op, value


def gold_operation_string(row):
    op = (row.get("operation_type") or "").upper()
    value = row.get("operation_value") or ""

    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return op


def pred_operation_string(row, variant):
    op, value = parse_action(row.get(f"{variant}_action"))
    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return ""


def recompute_variant(rows, variant):
    by_task = defaultdict(list)

    tokens = []
    pruning_rates = []
    key_recalls = []
    prune_times = []
    action_times = []
    total_times = []
    action_errors = 0

    for row in rows:
        raw_tokens = row.get("original_tokens")
        kept_tokens = row.get(f"{variant}_tokens")

        if kept_tokens is not None:
            tokens.append(float(kept_tokens))
        if raw_tokens and kept_tokens is not None:
            pruning_rates.append(1 - float(kept_tokens) / float(raw_tokens))

        key_recalls.append(row.get(f"{variant}_recall"))
        prune_times.append(row.get(f"{variant}_process_time_ms"))
        action_times.append(row.get(f"{variant}_action_time_ms"))
        total_times.append(row.get(f"{variant}_total_time_ms"))

        action = str(row.get(f"{variant}_action") or "")
        if action.startswith("Error:"):
            action_errors += 1

        task_id = row.get("task_id") or "UNKNOWN_TASK"

        ele = row.get(f"{variant}_element_acc")
        ele = 1.0 if ele else 0.0

        pred_op = pred_operation_string(row, variant)
        gold_op = gold_operation_string(row)
        op_f1 = token_f1(pred_op, gold_op)

        step_sr = 1.0 if ele == 1.0 and op_f1 >= 0.999999 else 0.0

        by_task[task_id].append({
            "ele_acc": ele,
            "op_f1": op_f1,
            "step_sr": step_sr,
        })

    task_metrics = []
    for task_id, steps in by_task.items():
        task_metrics.append({
            "task_id": task_id,
            "steps": len(steps),
            "ele_acc": mean([s["ele_acc"] for s in steps]),
            "op_f1": mean([s["op_f1"] for s in steps]),
            "step_sr": mean([s["step_sr"] for s in steps]),
            "task_sr": 1.0 if steps and all(s["step_sr"] == 1.0 for s in steps) else 0.0,
        })

    return {
        "tasks": len(task_metrics),
        "steps": len(rows),
        "tokens": mean(tokens),
        "pruning_rate": mean(pruning_rates),
        "key_recall": mean(key_recalls),
        "ele_acc": mean([t["ele_acc"] for t in task_metrics]),
        "op_f1": mean([t["op_f1"] for t in task_metrics]),
        "step_sr": mean([t["step_sr"] for t in task_metrics]),
        "task_sr": mean([t["task_sr"] for t in task_metrics]),
        "prune_time_ms": mean(prune_times),
        "action_time_ms": mean(action_times),
        "total_time_ms": mean(total_times),
        "action_errors": action_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/ablation_icddp_cross_task_qwen.json")
    parser.add_argument("--output", default="results/ablation_icddp_cross_task_qwen_recomputed.csv")
    args = parser.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    rows = data.get("step_results", [])

    variants = data.get("variants")
    if not variants:
        variants = sorted({r.get("variant") for r in rows if r.get("variant")})

    out_rows = []

    for variant in variants:
        variant_rows = [r for r in rows if r.get("variant") == variant]
        if not variant_rows:
            continue

        label = variant_rows[0].get("variant_label") or variant
        s = recompute_variant(variant_rows, variant)

        out_rows.append({
            "variant": variant,
            "variant_label": label,
            "tasks": s["tasks"],
            "steps": s["steps"],
            "tokens": num(s["tokens"], 0),
            "pruning_rate_%": pct(s["pruning_rate"]),
            "key_recall_%": pct(s["key_recall"]),
            "Ele.Acc_%": pct(s["ele_acc"]),
            "Op.F1_%": pct(s["op_f1"]),
            "Step_SR_%": pct(s["step_sr"]),
            "Task_SR_%": pct(s["task_sr"]),
            "prune_time_ms": num(s["prune_time_ms"], 1),
            "action_time_ms": num(s["action_time_ms"], 1),
            "total_time_ms": num(s["total_time_ms"], 1),
            "action_errors": s["action_errors"],
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "variant",
        "variant_label",
        "tasks",
        "steps",
        "tokens",
        "pruning_rate_%",
        "key_recall_%",
        "Ele.Acc_%",
        "Op.F1_%",
        "Step_SR_%",
        "Task_SR_%",
        "prune_time_ms",
        "action_time_ms",
        "total_time_ms",
        "action_errors",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nAblation recomputed results")
    print("=" * 120)
    print("\t".join(fields))
    for r in out_rows:
        print("\t".join(str(r[f]) for f in fields))

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
