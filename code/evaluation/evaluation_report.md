# Evaluation Report

## Strategy comparison on `dataset/sample_claims.csv`

| Strategy | claim_status | evidence_standard_met | issue_type | object_part | risk_flags | severity | supporting_image_ids | valid_image |
|---|---|---|---|---|---|---|---|---|
| single_call | 90.00% | 90.00% | 60.00% | 80.00% | 68.50% | 55.00% | 82.50% | 90.00% |
| two_stage | 80.00% | 100.00% | 60.00% | 75.00% | 62.50% | 60.00% | 75.00% | 90.00% |
| rule_based | 70.00% | 90.00% | 65.00% | 70.00% | 65.83% | 55.00% | 75.00% | 90.00% |

## claim_status confusion (predicted vs true)

**single_call**

| true | predicted | count |
|---|---|---|
| contradicted | contradicted | 3 |
| contradicted | not_enough_information | 1 |
| contradicted | supported | 1 |
| not_enough_information | not_enough_information | 2 |
| supported | supported | 13 |

**two_stage**

| true | predicted | count |
|---|---|---|
| contradicted | contradicted | 1 |
| contradicted | supported | 4 |
| not_enough_information | not_enough_information | 2 |
| supported | supported | 13 |

**rule_based**

| true | predicted | count |
|---|---|---|
| contradicted | contradicted | 1 |
| contradicted | supported | 4 |
| not_enough_information | contradicted | 1 |
| not_enough_information | supported | 1 |
| supported | supported | 13 |

## Operational analysis (sample set, then projected to full claims.csv)

| Strategy | Model calls | Prompt tokens | Output tokens (of which reasoning) | Images processed | Runtime (s) | Est. cost (sample) | Est. cost (full claims.csv) |
|---|---|---|---|---|---|---|---|
| single_call | 20 | 42642 | 4331 (1009) | 29 | 423.34 | $0.0 | $0.0 |
| two_stage | 49 | 67296 | 6048 (0) | 29 | 543.22 | $0.0 | $0.0 |
| rule_based | 49 | 43909 | 2968 (0) | 29 | 320.51 | $0.0 | $0.0 |

Cost assumptions (illustrative paid-tier pricing per model, verify against current published pricing before relying on these figures): gemini:gemini-2.5-flash: $0.3/1M input, $2.5/1M output, gemini:gemini-2.5-flash-lite: $0.1/1M input, $0.4/1M output.
TPM/RPM notes: `common/gemini_client.py` enforces a configurable requests-per-minute cap (`--rpm`, default 10) with exponential backoff retries on 429/RESOURCE_EXHAUSTED responses, and `--workers` controls concurrency. Two-stage issues one extra Gemini call per image (Stage A) plus one aggregation call (Stage B), so its call count scales with image count rather than staying flat at one call per claim.
Reasoning-token note: the local model (Gemma) emits a separate chain-of-thought in `message.reasoning_content`, which the client never parses for the final JSON answer (only `message.content` is parsed), so accuracy is unaffected. However the API's `completion_tokens` figure bundles reasoning tokens together with answer tokens; the 'Output tokens (of which reasoning)' column above breaks out how much of each call's output was thinking overhead versus the actual structured answer.