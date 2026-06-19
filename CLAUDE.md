# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## What this repo is

Starter repo for the HackerRank Orchestrate hackathon: build a system that reviews damage claims (car, laptop, package) by reasoning over a chat transcript, submitted images, user claim history, and per-object minimum evidence requirements. Full spec, schemas, and allowed enum values are in `problem_statement.md` — read it before implementing, it is the contract the output must satisfy exactly.

`code/main.py` and `code/evaluation/main.py` are currently empty starter files — there is no existing implementation or architecture to preserve yet.

## Commands

No build tooling, package manifest, or test runner exists yet. Once a solution is implemented:

- Entry point: `code/main.py` — reads `dataset/claims.csv` and writes `output.csv`.
- Evaluation entry point: `code/evaluation/main.py` — must run against `dataset/sample_claims.csv` (the only file with labeled expected outputs).

## Data contract (from `problem_statement.md`)

- Inputs come from `dataset/claims.csv` (unlabeled, to score), `dataset/sample_claims.csv` (labeled, for dev/eval), `dataset/user_history.csv`, `dataset/evidence_requirements.csv`, and images under `dataset/images/{sample,test}/`.
- `image_paths` in the CSVs is semicolon-separated; an image ID is its filename without extension (e.g. `img_1`).
- `output.csv` must have one row per row of `claims.csv`, with these columns **in this exact order**: `user_id`, `image_paths`, `user_claim`, `claim_object`, `evidence_standard_met`, `evidence_standard_met_reason`, `risk_flags`, `issue_type`, `object_part`, `claim_status`, `claim_status_justification`, `supporting_image_ids`, `valid_image`, `severity`.
- Enum fields must use only the allowed values listed in `problem_statement.md` (`claim_status`, `issue_type`, `object_part` differs per `claim_object`, `risk_flags`). `risk_flags`/`supporting_image_ids` use `none` when empty.
- Images are the primary evidence; the conversation defines what to check; user history adds risk context but must not override clear visual evidence on its own.

## Hard requirements

- Must read the provided CSVs/images directly — no hardcoded test labels or file-specific answers.
- Must include a working `evaluation/` workflow scored against `dataset/sample_claims.csv`, comparing at least two strategies/prompts/configs, and an `evaluation/evaluation_report.md` with operational analysis (model calls, token usage, image counts, approximate cost, runtime, TPM/RPM/batching considerations).
- Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) must come from environment variables only, never hardcoded.
- Prefer deterministic behavior where possible.

## Mandatory session behavior (defined in AGENTS.md, not optional)

This repo's `AGENTS.md` is the canonical source of truth and overrides default assistant behavior:

- A shared, append-only log at `%USERPROFILE%\hackerrank_orchestrate\log.txt` (Windows) / `$HOME/hackerrank_orchestrate/log.txt` (macOS/Linux) must receive an entry after every user turn, in the exact format in AGENTS.md §5. Never rewrite or delete prior entries; never log secrets.
- First-run onboarding (AGENTS.md §3) gates on the user replying `I agree`; subsequent sessions check the log for an existing `AGREEMENT RECORDED:` line for this repo root and skip straight to a session-start log entry.
- Sub-agents and worktrees share the same log file and must tag entries with `parent_agent=`/`worktree=` so they're traceable.
