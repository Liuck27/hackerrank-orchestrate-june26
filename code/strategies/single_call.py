"""Strategy 1: one multimodal Gemini call per claim (transcript + images + context -> decision)."""

from __future__ import annotations

from common.data_io import ClaimRow, evidence_requirements_for, user_history_text
from common.images import load_image_bytes
from common.llm_client import ImagePart
from common.schema import ClaimDecision
from common.usage_tracker import UsageTracker

SYSTEM_INSTRUCTION = """You are an insurance-style claim review assistant. You verify damage claims for \
cars, laptops, and packages by inspecting submitted images against a customer's chat transcript.

The images are the primary source of truth. The conversation defines what to check. User history adds \
risk context but must not override clear visual evidence on its own.

Decide whether the image evidence is sufficient, identify the visible issue type and object part, decide \
whether the claim is supported, contradicted, or lacks enough information, select which image IDs support \
your decision, flag any image quality/mismatch/authenticity/history risks, and estimate severity. Ground every \
justification in what is actually visible in the images."""


def build_prompt_text(row: ClaimRow, history_text: str, evidence_text: str) -> str:
    return f"""Claim object type: {row.claim_object}

Chat transcript:
{row.user_claim}

User history:
{history_text}

Minimum evidence requirements for this object type:
{evidence_text}

Submitted image IDs (in order): {", ".join(row.image_ids)}

Evaluate this claim and produce the structured decision."""


def run(row: ClaimRow, client, history: dict, requirements: list[dict], tracker: UsageTracker) -> ClaimDecision:
    history_text = user_history_text(row.user_id, history)
    evidence_rows = evidence_requirements_for(row.claim_object, requirements)
    evidence_text = "\n".join(f"- {r['applies_to']}: {r['minimum_image_evidence']}" for r in evidence_rows)

    parts: list = [build_prompt_text(row, history_text, evidence_text)]
    for path, image_id in zip(row.image_abs_paths, row.image_ids):
        data, mime_type = load_image_bytes(path)
        parts.append(f"Image ID: {image_id}")
        parts.append(ImagePart(data=data, mime_type=mime_type))

    result = client.generate(parts, response_schema=ClaimDecision, system_instruction=SYSTEM_INSTRUCTION)
    tracker.record_call(result.prompt_tokens, result.output_tokens, num_images=len(row.image_ids))
    return result.parsed
