import argparse
import csv
import glob
import json
import re
from pathlib import Path
from collections import defaultdict

METHODS = ["browseruse", "mindact", "icddp"]

DEFAULT_FILES = {
    "qwen": {
        "task": "results/experiment1_qwen_FULL_task*.json",
        "website": "results/experiment1_qwen_FULL_website*.json",
        "domain": "results/experiment1_qwen_FULL_domain*.json",
    },
    "gpt-4.1-mini": {
        "task": "results/experiment1_gpt41mini_FULL_task*.json",
        "website": "results/experiment1_gpt41mini_FULL_website*.json",
        "domain": "results/experiment1_gpt41mini_FULL_domain*.json",
    },
    "deepseek-v4-flash": {
        "task": "results/experiment1_deepseek_v4_flash_FULL_task*.json",
        "website": "results/experiment1_deepseek_v4_flash_FULL_website*.json",
        "domain": "results/experiment1_deepseek_v4_flash_FULL_domain*.json",
    },
}


def latest(pattern):
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def norm_op(op):
    op = (op or "").upper()
    if op in {"CLICK", "CLICK_ELEMENT"}:
        return "CLICK"
    if op in {"TYPE", "INPUT", "TEXT_INPUT"}:
        return "TYPE"
    if op in {"SELECT", "SELECT_OPTION", "SELECT_OPTION_VALUE"}:
        return "SELECT"
    return op


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


def extract_json_object(raw):
    raw = (raw or "").strip()
    if not raw or raw.startswith("Error:") or raw == "SKIPPED":
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.I | re.S)
    if fenced:
        raw = fenced.group(1).strip()

    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def parse_action(raw):
    raw = (raw or "").strip()
    data = extract_json_object(raw)

    if isinstance(data, dict):
        nested = data.get("action")
        if isinstance(nested, str) and nested.strip() != raw:
            nested_parsed = parse_action(nested)
            if nested_parsed["op"] or nested_parsed["value"]:
                return nested_parsed

        op = norm_op(
            data.get("operation")
            or data.get("op")
            or data.get("action_type")
            or data.get("type")
        )

        value = (
            data.get("value")
            or data.get("text")
            or data.get("option")
            or data.get("input")
            or ""
        )
        return {"op": op, "value": str(value).strip()}

    op_match = re.search(r"\b(CLICK|TYPE|SELECT)\b", raw, re.I)
    op = norm_op(op_match.group(1)) if op_match else ""

    value = ""
    value_match = re.search(
        r"(?:value|text|option|input)\s*[:=]\s*['\"]?(.+?)['\"]?\s*$",
        raw,
        re.I,
    )
    if value_match:
        value = value_match.group(1).strip()

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", raw)
    if not value and quoted and op in {"TYPE", "SELECT"}:
        value = quoted[-1].strip()

    return {"op": op, "value": value}


def gold_operation_string(row):
    op = norm_op(row.get("operation_type"))
    value = (
        row.get("operation_value")
        or row.get("target_value")
        or row.get("value")
        or ""
    )

    # Recover operation_value from step_description when older JSON files omit it.
    desc = row.get("step_description") or ""
    if not value:
        m = re.search(r"value\s*:\s*(.+)$", desc, re.I)
        if m:
            value = m.group(1).strip()

    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return op


def pred_operation_string(row, method):
    parsed = parse_action(row.get(f"{method}_action"))
    op = parsed["op"] or norm_op(row.get(f"{method}_pred_op"))
    value = parsed["value"]

    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return op


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "" if x is None else f"{x * 100:.2f}"


def recompute(rows, method):
    by_task = defaultdict(list)

    for row in rows:
        task_id = row.get("task_id") or row.get("annotation_id") or "UNKNOWN_TASK"
        ele = row.get(f"{method}_element_acc")
        # if ele is None:
        #     continue

        ele = 1.0 if ele else 0.0
        pred_op = pred_operation_string(row, method)
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
            "ele_acc": mean(s["ele_acc"] for s in steps),
            "op_f1": mean(s["op_f1"] for s in steps),
            "step_sr": mean(s["step_sr"] for s in steps),
            "task_sr": 1.0 if all(s["step_sr"] == 1.0 for s in steps) else 0.0,
        })

    return {
        "tasks": len(task_metrics),
        "steps": sum(t["steps"] for t in task_metrics),
        "ele_acc": mean(t["ele_acc"] for t in task_metrics),
        "op_f1": mean(t["op_f1"] for t in task_metrics),
        "step_sr": mean(t["step_sr"] for t in task_metrics),
        "task_sr": mean(t["task_sr"] for t in task_metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mind2web_full_recomputed_metrics.csv")
    args = parser.parse_args()

    output_rows = []

    for model, split_patterns in DEFAULT_FILES.items():
        for split, pattern in split_patterns.items():
            path = latest(pattern)
            if path is None:
                print(f"[WARN] missing {model} {split}: {pattern}")
                continue

            data = json.load(open(path, encoding="utf-8"))
            rows = data.get("step_results", [])
            print(f"[LOAD] {model:18s} {split:8s} {len(rows):4d} steps <- {path}")

            for method in METHODS:
                m = recompute(rows, method)
                output_rows.append({
                    "model": model,
                    "split": split,
                    "method": method,
                    "tasks": m["tasks"],
                    "steps": m["steps"],
                    "Ele.Acc_%": pct(m["ele_acc"]),
                    "Op.F1_%": pct(m["op_f1"]),
                    "Step_SR_%": pct(m["step_sr"]),
                    "Task_SR_%": pct(m["task_sr"]),
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = ["model", "split", "method", "tasks", "steps", "Ele.Acc_%", "Op.F1_%", "Step_SR_%", "Task_SR_%"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + "=" * 90)
    print("Mind2Web-style recomputed metrics")
    print("=" * 90)
    print("\t".join(["model", "split", "method", "tasks", "steps", "Ele.Acc", "Op.F1", "Step SR", "Task SR"]))
    for r in output_rows:
        print("\t".join(str(r[k]) for k in [
            "model", "split", "method", "tasks", "steps", "Ele.Acc_%", "Op.F1_%", "Step_SR_%", "Task_SR_%"
        ]))

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
