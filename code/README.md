# Multi-Modal Claim Verification — solution README

Two strategies for verifying car/laptop/package damage claims with a multimodal LLM, plus an
offline evaluation step. See repo-root `problem_statement.md` for the full I/O schema and allowed
values.

The pipeline supports two interchangeable backends behind the same `--backend` flag:

- **`gemini`** (default) — cloud, Google Gemini API (`gemini-2.5-flash` by default).
- **`lmstudio`** — local inference via LM Studio's OpenAI-compatible server (no API key, no
  per-token cost, no daily quota). This is what the final `output.csv` was generated with — see
  [Final strategy used for `output.csv`](#final-strategy-used-for-outputcsv).

## Setup

```bash
python -m venv .venv          # from the repo root
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r code/requirements.txt
cp .env.example .env          # then fill in GEMINI_API_KEY (only needed for --backend gemini)
```

`GEMINI_API_KEY` is read from the environment (or `.env` via `python-dotenv`). It is never
hardcoded anywhere in the code. The `lmstudio` backend needs no key, just a running LM Studio
server with a vision-capable model loaded (default expected at `http://127.0.0.1:1234/v1`).

## Strategies

- **`single_call`** — one multimodal call per claim: transcript, user history, evidence
  requirements, and all submitted images go into a single prompt; the model returns the full
  structured decision directly. Cheapest, fewest calls.
- **`two_stage`** — Stage A makes one vision-only call *per submitted image*, extracting
  structured observations (issue type, object part, image quality, whether the claimed object is
  shown) with no transcript/history involved. Stage B makes one text-only call that combines all
  Stage-A observations with the transcript, history, and evidence requirements to produce the
  final decision. More calls (proportional to image count), decouples visual grounding from
  reasoning.

Both live under `code/strategies/` and share `code/common/`: schema, CSV/image loading, a
backend-agnostic content representation (`common/llm_client.py`), the Gemini client (retry/backoff
+ RPM cap), the LM Studio client, and a usage tracker. Strategies are written once against `str`
and `ImagePart` content parts and work unchanged against either backend.

## Running a strategy

`code/main.py` is the only place that calls the model. It writes a prediction CSV with the
required 14 columns in order, plus a `*.usage.json` sidecar (same stem) recording model-call
count, prompt/output tokens, images processed, and runtime — used later by the evaluation step.

```bash
cd code
# cheap smoke test on a couple of rows before spending the full token/time budget
python main.py --strategy single_call --backend lmstudio --input ../dataset/sample_claims.csv \
    --output evaluation/predictions/single_call.csv --limit 3

python main.py --strategy two_stage --backend lmstudio --input ../dataset/sample_claims.csv \
    --output evaluation/predictions/two_stage.csv --limit 3

# full sample set for real evaluation
python main.py --strategy single_call --backend lmstudio --input ../dataset/sample_claims.csv \
    --output evaluation/predictions/single_call.csv
python main.py --strategy two_stage --backend lmstudio --input ../dataset/sample_claims.csv \
    --output evaluation/predictions/two_stage.csv

# final predictions for submission
python main.py --strategy single_call --backend lmstudio --input ../dataset/claims.csv --output ../output.csv
```

To use the cloud Gemini backend instead, drop `--backend lmstudio` (it's the default) and make
sure `GEMINI_API_KEY` is set. Note the Gemini free tier is capped at **20 `generate_content`
requests/day per model** — easy to exhaust across a few dev iterations; `--model
gemini-2.5-flash-lite` has its own separate 20/day quota, it does not add to the flash quota.

Flags: `--limit N` (process only the first N rows — use this while testing), `--workers N` (light
concurrency), `--rpm N` (requests-per-minute cap, gemini backend only, default 10), `--model`
(override the default model per backend), `--lmstudio-url` (default `http://127.0.0.1:1234/v1`).

## Evaluation

`code/evaluation/main.py` does **not** call any model and does **not** re-run any strategy. It is
a pure offline comparison: it reads prediction CSVs already produced by `main.py` (plus their
`*.usage.json` sidecars), joins them back to the labeled rows in `dataset/sample_claims.csv`, and
writes `evaluation/evaluation_report.md` with per-field accuracy, a `claim_status` confusion
table, and the operational analysis (model calls, tokens, images, estimated cost, runtime,
TPM/RPM notes), projected from the sample set to the full 44-row `dataset/claims.csv`.

```bash
cd code/evaluation
python main.py --predictions single_call=predictions/single_call.csv \
                --predictions two_stage=predictions/two_stage.csv
```

## Final strategy used for `output.csv`

**`single_call` via the `lmstudio` backend** (model `google/gemma-4-e4b:2`). On the full 20-row
sample set, `single_call` beat `two_stage` on 7 of 8 scored fields, including `claim_status`
accuracy (70% vs 60%), while using fewer than half the model calls and roughly half the runtime —
see `evaluation/evaluation_report.md` for the full comparison and operational analysis.

The Gemini cloud backend was the original design (see `common/gemini_client.py`, still fully
implemented and usable), but its free tier's 20-requests/day/model quota was exhausted during
development across both `gemini-2.5-flash` and `gemini-2.5-flash-lite`, before a full sample-set
run could complete. The `lmstudio` backend was added to unblock development without changing the
strategy logic at all (same `single_call`/`two_stage` code, same schema, same evaluation
pipeline) — only the client swaps.

## Notes / gotchas found during development

- Image files under `dataset/images/` are inconsistently encoded despite the `.jpg` extension —
  some are real JPEG, some WebP, some PNG, and a few are AVIF. `common/images.py` sniffs real
  magic bytes rather than trusting the extension.
- A few test-set images are very large (one is 7908×5931, ~5.6MB); LM Studio's vision endpoint
  rejected them outright. `common/lmstudio_client.py` re-encodes every image to PNG and downscales
  anything above 1536px on its longest side before sending.
- Local models prompted for strict JSON occasionally wrap a scalar enum field in a single-element
  list, or emit a value outside the allowed enum. `lmstudio_client.py` unwraps singleton lists and
  falls back to `"unknown"` for invalid enum values rather than failing the row (the Gemini backend
  doesn't need this — it enforces the schema natively via `response_schema`).
- No test labels or file-specific answers are hardcoded anywhere — every decision comes from a
  model call grounded in the provided CSVs/images.
