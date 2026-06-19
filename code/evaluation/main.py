"""Offline evaluation: scores already-generated prediction CSVs against sample_claims.csv labels.

This does NOT call Gemini or re-run any strategy. Generate predictions first with
`python main.py --strategy <name> --input ../dataset/sample_claims.csv --output predictions/<name>.csv`,
then run this script to compare strategies and produce evaluation_report.md.

Usage:
    python main.py --predictions single_call=predictions/single_call.csv --predictions two_stage=predictions/two_stage.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data_io import REPO_ROOT
from metrics import score_predictions

# Illustrative pricing assumptions (USD per 1M tokens). Local LM Studio inference has no
# per-token billing (cost is local GPU compute/electricity, not estimated here), so it's $0.
# Treated as explicit assumptions for the operational-cost estimate, not a guarantee.
PRICING_PER_MODEL = {
    "gemini:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini:gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}
DEFAULT_PRICING = {"input": 0.0, "output": 0.0}  # local backends (e.g. lmstudio:*)

DEFAULT_LABELS_PATH = REPO_ROOT / "dataset" / "sample_claims.csv"
FULL_CLAIMS_PATH = REPO_ROOT / "dataset" / "claims.csv"


def parse_predictions_arg(pairs: list[str]) -> dict[str, Path]:
    result = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        if not name or not path:
            raise ValueError(f"--predictions must be name=path.csv, got: {pair}")
        result[name] = Path(path)
    return result


def load_usage(predictions_path: Path) -> dict | None:
    usage_path = predictions_path.with_suffix(".usage.json")
    if not usage_path.exists():
        return None
    return json.loads(usage_path.read_text())


def estimate_cost(usage: dict) -> float:
    pricing = PRICING_PER_MODEL.get(usage.get("model", ""), DEFAULT_PRICING)
    input_cost = (usage["prompt_tokens"] / 1_000_000) * pricing["input"]
    output_cost = (usage["output_tokens"] / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 4)


def render_report(strategy_results: dict[str, dict], full_claims_row_count: int) -> str:
    lines = ["# Evaluation Report", ""]
    lines.append("## Strategy comparison on `dataset/sample_claims.csv`")
    lines.append("")
    field_names = sorted(next(iter(strategy_results.values()))["metrics"]["field_scores"].keys())
    header = "| Strategy | " + " | ".join(field_names) + " |"
    separator = "|---" * (len(field_names) + 1) + "|"
    lines += [header, separator]
    for name, result in strategy_results.items():
        scores = result["metrics"]["field_scores"]
        row = f"| {name} | " + " | ".join(f"{scores[f]:.2%}" for f in field_names) + " |"
        lines.append(row)
    lines.append("")

    lines.append("## claim_status confusion (predicted vs true)")
    for name, result in strategy_results.items():
        lines.append(f"\n**{name}**")
        lines.append("")
        lines.append("| true | predicted | count |")
        lines.append("|---|---|---|")
        for entry in result["metrics"]["claim_status_confusion"]:
            lines.append(f"| {entry['claim_status_true']} | {entry['claim_status_pred']} | {entry['count']} |")
    lines.append("")

    lines.append("## Operational analysis (sample set, then projected to full claims.csv)")
    lines.append("")
    lines.append(
        "| Strategy | Model calls | Prompt tokens | Output tokens (of which reasoning) | Images processed | "
        "Runtime (s) | Est. cost (sample) | Est. cost (full claims.csv) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, result in strategy_results.items():
        usage = result["usage"]
        if usage is None:
            lines.append(f"| {name} | usage sidecar not found | | | | | | |")
            continue
        sample_rows = result["metrics"]["num_rows"]
        scale = full_claims_row_count / sample_rows if sample_rows else 0
        sample_cost = estimate_cost(usage)
        projected_cost = round(sample_cost * scale, 4)
        reasoning_tokens = usage.get("reasoning_tokens", 0)
        lines.append(
            f"| {name} | {usage['model_calls']} | {usage['prompt_tokens']} | "
            f"{usage['output_tokens']} ({reasoning_tokens}) | "
            f"{usage['images_processed']} | {usage['runtime_seconds']} | ${sample_cost} | ${projected_cost} |"
        )
    lines.append("")
    pricing_notes = ", ".join(
        f"{model}: ${p['input']}/1M input, ${p['output']}/1M output" for model, p in PRICING_PER_MODEL.items()
    )
    lines.append(
        f"Cost assumptions (illustrative paid-tier pricing per model, verify against current published "
        f"pricing before relying on these figures): {pricing_notes}."
    )
    lines.append(
        "TPM/RPM notes: `common/gemini_client.py` enforces a configurable requests-per-minute cap "
        "(`--rpm`, default 10) with exponential backoff retries on 429/RESOURCE_EXHAUSTED responses, "
        "and `--workers` controls concurrency. Two-stage issues one extra Gemini call per image "
        "(Stage A) plus one aggregation call (Stage B), so its call count scales with image count "
        "rather than staying flat at one call per claim."
    )
    lines.append(
        "Reasoning-token note: the local model (Gemma) emits a separate chain-of-thought in "
        "`message.reasoning_content`, which the client never parses for the final JSON answer "
        "(only `message.content` is parsed), so accuracy is unaffected. However the API's "
        "`completion_tokens` figure bundles reasoning tokens together with answer tokens; the "
        "'Output tokens (of which reasoning)' column above breaks out how much of each call's "
        "output was thinking overhead versus the actual structured answer."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline strategy comparison from prediction CSVs.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="name=path.csv",
        help="Repeatable: name=path to a prediction CSV produced by ../main.py",
    )
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "evaluation_report.md"))
    args = parser.parse_args()

    predictions_paths = parse_predictions_arg(args.predictions)
    labels = pd.read_csv(args.labels, dtype=str).fillna("")

    full_claims_row_count = sum(1 for _ in open(FULL_CLAIMS_PATH, encoding="utf-8")) - 1

    strategy_results = {}
    for name, path in predictions_paths.items():
        predictions = pd.read_csv(path, dtype=str).fillna("")
        metrics = score_predictions(predictions, labels)
        usage = load_usage(path)
        strategy_results[name] = {"metrics": metrics, "usage": usage}
        print(f"{name}: {metrics['num_rows']} rows scored, field_scores={metrics['field_scores']}")

    report = render_report(strategy_results, full_claims_row_count)
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
