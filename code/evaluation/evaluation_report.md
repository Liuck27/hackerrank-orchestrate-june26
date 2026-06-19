# Evaluation Report

## Strategy comparison on `dataset/sample_claims.csv`

| Strategy | claim_status | evidence_standard_met | issue_type | object_part | risk_flags | severity | supporting_image_ids | valid_image |
|---|---|---|---|---|---|---|---|---|
| single_call | 70.00% | 95.00% | 60.00% | 80.00% | 62.92% | 60.00% | 75.00% | 90.00% |
| two_stage | 60.00% | 70.00% | 45.00% | 65.00% | 57.92% | 55.00% | 80.00% | 80.00% |

## claim_status confusion (predicted vs true)

**single_call**

| true | predicted | count |
|---|---|---|
| contradicted | supported | 5 |
| not_enough_information | not_enough_information | 2 |
| supported | contradicted | 1 |
| supported | supported | 12 |

**two_stage**

| true | predicted | count |
|---|---|---|
| contradicted | contradicted | 1 |
| contradicted | not_enough_information | 2 |
| contradicted | supported | 2 |
| not_enough_information | not_enough_information | 2 |
| supported | contradicted | 2 |
| supported | not_enough_information | 2 |
| supported | supported | 9 |

## Operational analysis (sample set, then projected to full claims.csv)

| Strategy | Model calls | Prompt tokens | Output tokens | Images processed | Runtime (s) | Est. cost (sample) | Est. cost (full claims.csv) |
|---|---|---|---|---|---|---|---|
| single_call | 20 | 27482 | 3101 | 29 | 301.22 | $0.0 | $0.0 |
| two_stage | 49 | 43726 | 6229 | 29 | 626.47 | $0.0 | $0.0 |

Cost assumptions (illustrative paid-tier pricing per model, verify against current published pricing before relying on these figures): gemini:gemini-2.5-flash: $0.3/1M input, $2.5/1M output, gemini:gemini-2.5-flash-lite: $0.1/1M input, $0.4/1M output.
TPM/RPM notes: `common/gemini_client.py` enforces a configurable requests-per-minute cap (`--rpm`, default 10) with exponential backoff retries on 429/RESOURCE_EXHAUSTED responses, and `--workers` controls concurrency. Two-stage issues one extra Gemini call per image (Stage A) plus one aggregation call (Stage B), so its call count scales with image count rather than staying flat at one call per claim.