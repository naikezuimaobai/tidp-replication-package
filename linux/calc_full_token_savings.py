import csv
import json
from pathlib import Path

FILES = {
    "Cross-Task": "results/experiment1_qwen_FULL_task.json",
    "Cross-Website": "results/experiment1_qwen_FULL_website.json",
    "Cross-Domain": "results/experiment1_qwen_FULL_domain.json",
}

METHODS = {
    "MindAct-style": "mindact",
    "BU-style": "browseruse",
    "TIDP": "icddp",
}

BASELINES = ["MindAct-style", "BU-style"]


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("step_results", [])


def fmt_int(x):
    return f"{int(round(x)):,}"


def fmt_float(x, digits=2):
    return f"{x:.{digits}f}"


def main():
    summary_rows = []
    saving_rows = []

    all_totals = {method: 0.0 for method in METHODS}
    all_steps = 0

    for split, path in FILES.items():
        rows = load_rows(path)
        steps = len(rows)
        all_steps += steps

        totals = {}
        avgs = {}

        for method_label, method_key in METHODS.items():
            tokens = [float(r.get(f"{method_key}_tokens") or 0) for r in rows]
            total = sum(tokens)
            avg = total / steps if steps else 0

            totals[method_label] = total
            avgs[method_label] = avg
            all_totals[method_label] += total

            summary_rows.append({
                "split": split,
                "method": method_label,
                "steps": steps,
                "avg_tokens": f"{avg:.0f}",
                "total_tokens": f"{total:.0f}",
                "total_tokens_fmt": fmt_int(total),
            })

        icddp_total = totals["TIDP"]

        for baseline in BASELINES:
            baseline_total = totals[baseline]
            saved = baseline_total - icddp_total
            ratio = saved / baseline_total if baseline_total else 0

            saving_rows.append({
                "split": split,
                "baseline": baseline,
                "baseline_total_tokens": f"{baseline_total:.0f}",
                "icddp_total_tokens": f"{icddp_total:.0f}",
                "saved_tokens": f"{saved:.0f}",
                "saving_ratio_%": f"{ratio * 100:.2f}",
                "baseline_total_tokens_fmt": fmt_int(baseline_total),
                "icddp_total_tokens_fmt": fmt_int(icddp_total),
                "saved_tokens_fmt": fmt_int(saved),
            })

    for method_label in METHODS:
        total = all_totals[method_label]
        avg = total / all_steps if all_steps else 0
        summary_rows.append({
            "split": "ALL",
            "method": method_label,
            "steps": all_steps,
            "avg_tokens": f"{avg:.0f}",
            "total_tokens": f"{total:.0f}",
            "total_tokens_fmt": fmt_int(total),
        })

    icddp_all = all_totals["TIDP"]
    for baseline in BASELINES:
        baseline_total = all_totals[baseline]
        saved = baseline_total - icddp_all
        ratio = saved / baseline_total if baseline_total else 0
        saving_rows.append({
            "split": "ALL",
            "baseline": baseline,
            "baseline_total_tokens": f"{baseline_total:.0f}",
            "icddp_total_tokens": f"{icddp_all:.0f}",
            "saved_tokens": f"{saved:.0f}",
            "saving_ratio_%": f"{ratio * 100:.2f}",
            "baseline_total_tokens_fmt": fmt_int(baseline_total),
            "icddp_total_tokens_fmt": fmt_int(icddp_all),
            "saved_tokens_fmt": fmt_int(saved),
        })

    out_dir = Path("results/full_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "full_token_totals_by_split.csv"
    saving_path = out_dir / "full_icddp_token_savings.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = ["split", "method", "steps", "avg_tokens", "total_tokens", "total_tokens_fmt"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(saving_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "split",
            "baseline",
            "baseline_total_tokens",
            "icddp_total_tokens",
            "saved_tokens",
            "saving_ratio_%",
            "baseline_total_tokens_fmt",
            "icddp_total_tokens_fmt",
            "saved_tokens_fmt",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(saving_rows)

    print("\nToken totals")
    print("=" * 100)
    for r in summary_rows:
        print(
            f"{r['split']:<14} {r['method']:<14} "
            f"steps={r['steps']:<5} avg={r['avg_tokens']:<6} total={r['total_tokens_fmt']}"
        )

    print("\nTIDP token savings")
    print("=" * 100)
    for r in saving_rows:
        print(
            f"{r['split']:<14} vs {r['baseline']:<14} "
            f"baseline={r['baseline_total_tokens_fmt']:<14} "
            f"tidp={r['icddp_total_tokens_fmt']:<14} "
            f"saved={r['saved_tokens_fmt']:<14} "
            f"ratio={r['saving_ratio_%']}%"
        )

    print("\nSaved:")
    print(summary_path)
    print(saving_path)


if __name__ == "__main__":
    main()
