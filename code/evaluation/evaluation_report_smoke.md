# Evaluation Report

## Strategy comparison on `dataset/sample_claims.csv`

| Strategy | claim_status | evidence_standard_met | issue_type | object_part | risk_flags | severity | supporting_image_ids | valid_image |
|---|---|---|---|---|---|---|---|---|
| single_call | 66.67% | 66.67% | 0.00% | 100.00% | 33.33% | 0.00% | 100.00% | 66.67% |
| two_stage | 66.67% | 66.67% | 0.00% | 100.00% | 33.33% | 0.00% | 66.67% | 100.00% |

## claim_status confusion (predicted vs true)

**single_call**

| true | predicted | count |
|---|---|---|
| supported | not_enough_information | 1 |
| supported | supported | 2 |

**two_stage**

| true | predicted | count |
|---|---|---|
| supported | not_enough_information | 1 |
| supported | supported | 2 |

## Operational analysis (sample set, then projected to full claims.csv)

| Strategy | Model calls | Prompt tokens | Output tokens | Images processed | Runtime (s) | Est. cost (sample) | Est. cost (full claims.csv) |
|---|---|---|---|---|---|---|---|
| single_call | 3 | 2828 | 677 | 5 | 30.7 | $0.0025 | $0.0367 |
| two_stage | 8 | 3478 | 985 | 5 | 81.61 | $0.0035 | $0.0513 |

Cost assumptions: $0.3/1M input tokens, $2.5/1M output tokens (illustrative Gemini 2.5 Flash paid-tier pricing; verify against current published pricing before relying on these figures).
TPM/RPM notes: `common/gemini_client.py` enforces a configurable requests-per-minute cap (`--rpm`, default 10) with exponential backoff retries on 429/RESOURCE_EXHAUSTED responses, and `--workers` controls concurrency. Two-stage issues one extra Gemini call per image (Stage A) plus one aggregation call (Stage B), so its call count scales with image count rather than staying flat at one call per claim.