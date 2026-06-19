"""Entry point: run a claim-verification strategy over a claims CSV and write predictions.

Usage:
    python main.py --strategy single_call --input ../dataset/sample_claims.csv --output out.csv --limit 5
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from common.data_io import REPO_ROOT, ClaimRow, load_claims, load_evidence_requirements, load_user_history, write_output_csv
from common.gemini_client import GeminiClient
from common.lmstudio_client import LMStudioClient
from common.usage_tracker import UsageTracker
from strategies import single_call, two_stage

STRATEGIES = {
    "single_call": single_call.run,
    "two_stage": two_stage.run,
}


def process_row(
    row: ClaimRow,
    strategy_fn,
    client,
    history: dict,
    requirements: list[dict],
    tracker: UsageTracker,
) -> dict:
    decision = strategy_fn(row, client, history, requirements, tracker)
    decision = decision.enforce_injection_policy()
    return decision.to_output_row(row.user_id, row.image_paths, row.user_claim, row.claim_object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a claim-verification strategy.")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="single_call")
    parser.add_argument("--input", default=str(REPO_ROOT / "dataset" / "claims.csv"))
    parser.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["gemini", "lmstudio"], default="gemini")
    parser.add_argument("--rpm", type=int, default=10, help="Max Gemini requests per minute (gemini backend only)")
    parser.add_argument("--model", default=None, help="Model name; defaults per backend")
    parser.add_argument(
        "--lmstudio-url", default="http://127.0.0.1:1234/v1", help="LM Studio OpenAI-compatible base URL"
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    rows = load_claims(Path(args.input), limit=args.limit)
    history = load_user_history()
    requirements = load_evidence_requirements()

    if args.backend == "gemini":
        model = args.model or "gemini-2.5-flash"
        client = GeminiClient(model=model, rpm=args.rpm)
    else:
        model = args.model or "google/gemma-4-e4b:2"
        client = LMStudioClient(model=model, base_url=args.lmstudio_url)
    tracker = UsageTracker(strategy=args.strategy, model=f"{args.backend}:{model}")
    strategy_fn = STRATEGIES[args.strategy]

    output_rows: list[dict | None] = [None] * len(rows)
    if args.workers <= 1:
        for row in rows:
            output_rows[row.row_index] = process_row(row, strategy_fn, client, history, requirements, tracker)
            print(f"[{args.strategy}] processed row {row.row_index + 1}/{len(rows)} (user_id={row.user_id})")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_row, row, strategy_fn, client, history, requirements, tracker): row
                for row in rows
            }
            done = 0
            for future in futures:
                row = futures[future]
                output_rows[row.row_index] = future.result()
                done += 1
                print(f"[{args.strategy}] processed row {done}/{len(rows)} (user_id={row.user_id})")

    tracker.finish()
    output_path = Path(args.output)
    write_output_csv(output_rows, output_path)
    usage_path = output_path.with_suffix(".usage.json")
    tracker.write(usage_path)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"Wrote usage stats to {usage_path}")


if __name__ == "__main__":
    main()
