"""TIDP ablation runner for Experiment 1.

This script evaluates TIDP variants on a fixed Mind2Web split. It isolates
the pruning modules while keeping the dataset, token budget, intent labels,
action prompt, and decision model fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from data_loader import Mind2WebLoader
from dynamic_pruner import DynamicPruner
from evaluator import Evaluator
from experiment1_small import (
    DEFAULT_INTENT_MODEL_CONFIG,
    LLMModelConfig,
    MODEL_FAMILY_DEFAULTS,
    SimpleActionDecider,
    action_elements,
    count_tokens,
    evaluate_kept,
    parse_action_prediction,
    selected_element_hits_positive,
)
from relevance_scorer import RelevanceScorer


VARIANTS: Dict[str, Dict[str, bool]] = {
    "full": {
        "enable_basic_filter": True,
        "enable_intent_filter": True,
        "enable_relevance_filter": True,
        "enable_inheritance_recovery": True,
    },
    "wo_basic": {
        "enable_basic_filter": False,
        "enable_intent_filter": True,
        "enable_relevance_filter": True,
        "enable_inheritance_recovery": True,
    },
    "wo_intent": {
        "enable_basic_filter": True,
        "enable_intent_filter": False,
        "enable_relevance_filter": True,
        "enable_inheritance_recovery": True,
    },
    "wo_relevance": {
        "enable_basic_filter": True,
        "enable_intent_filter": True,
        "enable_relevance_filter": False,
        "enable_inheritance_recovery": True,
    },
    "wo_recovery": {
        "enable_basic_filter": True,
        "enable_intent_filter": True,
        "enable_relevance_filter": True,
        "enable_inheritance_recovery": False,
    },
}

VARIANT_LABELS = {
    "full": "Full TIDP",
    "wo_basic": "w/o Basic Filtering",
    "wo_intent": "w/o Intent-Aware Filtering",
    "wo_relevance": "w/o Relevance Filtering",
    "wo_recovery": "w/o Inheritance Recovery",
}


def load_intent_cache(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("intents", data)


def get_cached_intent(intent_cache: Dict[str, Dict[str, Any]], step: Any) -> str:
    cached = intent_cache.get(step.uid)
    if cached and cached.get("intent"):
        return cached["intent"]
    raise KeyError(f"Missing cached intent for step uid={step.uid}")


def load_tasks(data_dir: str, split: str, max_files: int | None, sample_tasks: int | None, seed: int) -> List[Any]:
    loader = Mind2WebLoader(data_dir)
    tasks = loader.load_split(split, max_files=max_files)
    if sample_tasks is not None and sample_tasks > 0 and sample_tasks < len(tasks):
        rng = random.Random(seed)
        tasks = rng.sample(tasks, sample_tasks)
    return tasks


def avg(values: List[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def tokenize(text: str | None) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def token_f1(pred: str | None, gold: str | None) -> float:
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)

    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    gold_counts: Dict[str, int] = defaultdict(int)
    for token in gold_toks:
        gold_counts[token] += 1

    common = 0
    for token in pred_toks:
        if gold_counts[token] > 0:
            common += 1
            gold_counts[token] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def parse_action_for_eval(action: str | None) -> tuple[str, str]:
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


def gold_operation_string(row: Dict[str, Any]) -> str:
    op = (row.get("operation_type") or "").upper()
    value = row.get("operation_value") or ""

    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return op


def pred_operation_string(row: Dict[str, Any], prefix: str) -> str:
    op, value = parse_action_for_eval(row.get(f"{prefix}_action"))
    if op == "CLICK":
        return "CLICK"
    if op in {"TYPE", "SELECT"}:
        return f"{op} {value}".strip()
    return ""


def recompute_variant(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    """Recompute Mind2Web-style metrics with macro averaging across tasks."""
    by_task: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    tokens = []
    pruning_rates = []
    key_recalls = []
    prune_times = []
    action_times = []
    total_times = []
    action_errors = 0

    for row in rows:
        raw_tokens = row.get("original_tokens")
        kept_tokens = row.get(f"{prefix}_tokens")

        if kept_tokens is not None:
            tokens.append(float(kept_tokens))
        if raw_tokens and kept_tokens is not None:
            pruning_rates.append(1 - float(kept_tokens) / float(raw_tokens))

        key_recalls.append(row.get(f"{prefix}_recall"))
        prune_times.append(row.get(f"{prefix}_process_time_ms"))
        action_times.append(row.get(f"{prefix}_action_time_ms"))
        total_times.append(row.get(f"{prefix}_total_time_ms"))

        action = str(row.get(f"{prefix}_action") or "")
        if action.startswith("Error:"):
            action_errors += 1

        ele = 1.0 if row.get(f"{prefix}_element_acc") else 0.0
        op_f1 = token_f1(pred_operation_string(row, prefix), gold_operation_string(row))
        step_sr = 1.0 if ele == 1.0 and op_f1 >= 0.999999 else 0.0

        task_id = row.get("task_id") or "UNKNOWN_TASK"
        by_task[task_id].append(
            {
                "ele_acc": ele,
                "op_f1": op_f1,
                "step_sr": step_sr,
            }
        )

    task_metrics = []
    for task_id, steps in by_task.items():
        task_metrics.append(
            {
                "task_id": task_id,
                "steps": len(steps),
                "ele_acc": avg([s["ele_acc"] for s in steps]),
                "op_f1": avg([s["op_f1"] for s in steps]),
                "step_sr": avg([s["step_sr"] for s in steps]),
                "task_sr": 1.0 if steps and all(s["step_sr"] == 1.0 for s in steps) else 0.0,
            }
        )

    return {
        "tasks": len(task_metrics),
        "steps": len(rows),
        "tokens": avg(tokens),
        "pruning_rate": avg(pruning_rates),
        "key_recall": avg(key_recalls),
        "ele_acc": avg([t["ele_acc"] for t in task_metrics]),
        "op_f1": avg([t["op_f1"] for t in task_metrics]),
        "step_sr": avg([t["step_sr"] for t in task_metrics]),
        "task_sr": avg([t["task_sr"] for t in task_metrics]),
        "prune_time_ms": avg(prune_times),
        "action_time_ms": avg(action_times),
        "total_time_ms": avg(total_times),
        "action_errors": action_errors,
    }


def summarize(rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    return recompute_variant(rows, prefix)


def pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def num(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def build_qwen_config(args: argparse.Namespace) -> LLMModelConfig:
    base = MODEL_FAMILY_DEFAULTS["qwen"]
    return LLMModelConfig(
        family="qwen",
        model=args.qwen_model or base.model,
        api_base=args.qwen_api_base or base.api_base,
        api_key_env=args.qwen_api_key_env
        if args.qwen_api_key_env is not None
        else base.api_key_env,
        api_key=args.qwen_api_key,
    )


def run_variant(
    variant_name: str,
    tasks: List[Any],
    split: str,
    scorer: RelevanceScorer,
    evaluator: Evaluator,
    action_decider: SimpleActionDecider | None,
    intent_cache: Dict[str, Dict[str, Any]],
    context_limit: int,
    max_steps: int | None,
) -> List[Dict[str, Any]]:
    pruner = DynamicPruner(scorer=scorer, **VARIANTS[variant_name])
    rows: List[Dict[str, Any]] = []
    prefix = variant_name

    for task in tqdm(tasks, desc=f"Ablation {variant_name}"):
        for step in task.steps:
            if max_steps is not None and len(rows) >= max_steps:
                break
            if not step.raw_html or len(step.raw_html) < 100:
                continue

            intent = get_cached_intent(intent_cache, step)
            decision_intent = intent if VARIANTS[variant_name]["enable_intent_filter"] else "generic"
            subtask_desc = step.action_repr
            if step.operation_value:
                subtask_desc = f"{step.action_repr} value: {step.operation_value}"

            start = time.time()
            kept, stats = pruner.prune(
                html_content=step.raw_html,
                intent=intent,
                subtask_description=subtask_desc,
                context_limit=context_limit,
            )
            prune_ms = (time.time() - start) * 1000

            fields = evaluate_kept(
                evaluator=evaluator,
                name=prefix,
                kept_elements=kept,
                stats=stats,
                pos_candidates=step.pos_candidates,
                neg_candidates=step.neg_candidates,
                process_time_ms=prune_ms,
                action_time_ms=0.0,
                action="SKIPPED",
                expected_operation=step.operation_type,
            )

            action = "SKIPPED"
            action_ms = 0.0
            action_kept = action_elements(kept)
            if action_decider is not None:
                action, action_ms = action_decider.decide_action(
                    kept_elements=action_kept,
                    intent=decision_intent,
                    task_description=task.confirmed_task,
                    step_description=step.action_repr,
                    expected_operation=step.operation_type,
                    operation_value=step.operation_value,
                )

            prediction = parse_action_prediction(action)
            element_acc = selected_element_hits_positive(evaluator, action_kept, step.pos_candidates, prediction)
            op_match = 1 if prediction.operation == step.operation_type else 0 if prediction.operation else None
            step_sr = (
                1
                if element_acc and op_match
                else 0
                if element_acc is not None and op_match is not None
                else None
            )

            fields.update(
                {
                    f"{prefix}_action": action,
                    f"{prefix}_action_time_ms": action_ms,
                    f"{prefix}_total_time_ms": prune_ms + action_ms,
                    f"{prefix}_pred_op": prediction.operation,
                    f"{prefix}_pred_index": prediction.element_index,
                    f"{prefix}_element_acc": element_acc,
                    f"{prefix}_op_match": op_match,
                    f"{prefix}_step_sr": step_sr,
                    f"{prefix}_action_candidate_count": len(action_kept),
                }
            )

            row = {
                "variant": variant_name,
                "variant_label": VARIANT_LABELS[variant_name],
                "split": split,
                "task_id": task.task_id,
                "step_uid": step.uid,
                "website": task.website,
                "intent": intent,
                "decision_intent": decision_intent,
                "operation_type": step.operation_type,
                "operation_value": step.operation_value,
                "step_description": step.action_repr[:160],
                "original_tokens": count_tokens(step.raw_html),
            }
            row.update(fields)
            rows.append(row)

        if max_steps is not None and len(rows) >= max_steps:
            break

    return rows


def write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test_cross_task")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--sample_tasks", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--context_limit", type=int, default=8000)
    parser.add_argument("--scorer_model_path", required=True)
    parser.add_argument("--scorer_device", default="cuda:0")
    parser.add_argument("--intent_cache_path", required=True)
    parser.add_argument("--run_action_llm", action="store_true")
    parser.add_argument("--operation_hint_mode", choices=["none", "value", "full"], default="none")
    parser.add_argument("--variants", default="all", help="Comma-separated variant names or 'all'.")
    parser.add_argument("--qwen_model", default=None)
    parser.add_argument("--qwen_api_base", default=None)
    parser.add_argument("--qwen_api_key_env", default=None)
    parser.add_argument("--qwen_api_key", default=None)
    parser.add_argument("--output_json", default="results/ablation_icddp_cross_task.json")
    parser.add_argument("--output_csv", default="results/ablation_icddp_cross_task_summary.csv")
    args = parser.parse_args()

    selected_variants = list(VARIANTS) if args.variants == "all" else [v.strip() for v in args.variants.split(",")]
    unknown = [v for v in selected_variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid: {list(VARIANTS)}")

    tasks = load_tasks(args.data_dir, args.split, args.max_files, args.sample_tasks, args.sample_seed)
    intent_cache = load_intent_cache(args.intent_cache_path)
    if not intent_cache:
        raise ValueError(f"Intent cache is required for controlled ablation: {args.intent_cache_path}")

    scorer = RelevanceScorer(model_path=args.scorer_model_path, device=args.scorer_device)
    evaluator = Evaluator()
    action_decider = (
        SimpleActionDecider(build_qwen_config(args), operation_hint_mode=args.operation_hint_mode)
        if args.run_action_llm
        else None
    )

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for variant in selected_variants:
        rows = run_variant(
            variant_name=variant,
            tasks=tasks,
            split=args.split,
            scorer=scorer,
            evaluator=evaluator,
            action_decider=action_decider,
            intent_cache=intent_cache,
            context_limit=args.context_limit,
            max_steps=args.max_steps,
        )
        all_rows.extend(rows)
        s = summarize(rows, variant)
        summaries.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
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
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "icddp_ablation",
                "split": args.split,
                "context_limit": args.context_limit,
                "decision_model": build_qwen_config(args).public_dict(),
                "intent_model": DEFAULT_INTENT_MODEL_CONFIG.public_dict(),
                "variants": selected_variants,
                "summary": summaries,
                "step_results": all_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_summary_csv(Path(args.output_csv), summaries)

    print("\nTIDP Ablation Summary")
    print("=" * 120)
    for row in summaries:
        print(
            f"{row['variant_label']:<32} "
            f"Tokens={row['tokens']:<6} "
            f"Prune={row['pruning_rate_%']:<7} "
            f"Recall={row['key_recall_%']:<7} "
            f"Ele.Acc={row['Ele.Acc_%']:<7} "
            f"Op.F1={row['Op.F1_%']:<7} "
            f"Step SR={row['Step_SR_%']:<7} "
            f"Task SR={row['Task_SR_%']:<7} "
            f"Total={row['total_time_ms']} ms"
        )
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

