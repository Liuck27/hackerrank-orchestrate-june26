"""Strategy 2: per-image vision extraction (Stage A) then a text-only reasoning call (Stage B)."""

from __future__ import annotations

from common.data_io import ClaimRow, evidence_requirements_for, user_history_text
from common.images import load_image_bytes
from common.llm_client import ImagePart
from common.schema import ClaimDecision, ImageObservation
from common.usage_tracker import UsageTracker

STAGE_A_SYSTEM_INSTRUCTION = """You inspect a single submitted image for a damage claim review system. \
Describe only what is visible: the issue type, the object part, whether the image quality is good enough to \
evaluate, and whether the claimed object type is actually shown. Do not use any outside context about the \
claim — judge only from the image itself."""

STAGE_B_SYSTEM_INSTRUCTION = """You are an insurance-style claim review assistant. You receive structured \
visual observations already extracted from each submitted image (not the raw images), plus the customer's \
chat transcript, user history, and minimum evidence requirements.

The image observations are the primary source of truth. The conversation defines what to check. User history \
adds risk context but must not override clear visual evidence on its own.

Combine the per-image observations to decide whether evidence is sufficient, the overall visible issue type \
and object part, the claim status (supported, contradicted, not_enough_information), which image IDs support \
the decision, any risk flags, and severity. Ground every justification in the image observations provided."""


def _run_stage_a(row: ClaimRow, client, tracker: UsageTracker) -> list[ImageObservation]:
    observations: list[ImageObservation] = []
    for path, image_id in zip(row.image_abs_paths, row.image_ids):
        data, mime_type = load_image_bytes(path)
        parts = [
            f"Claim object type: {row.claim_object}. Image ID: {image_id}.",
            ImagePart(data=data, mime_type=mime_type),
        ]
        result = client.generate(
            parts, response_schema=ImageObservation, system_instruction=STAGE_A_SYSTEM_INSTRUCTION
        )
        tracker.record_call(result.prompt_tokens, result.output_tokens, num_images=1)
        observation = result.parsed
        observation.image_id = image_id
        observations.append(observation)
    return observations


def _build_stage_b_prompt(row: ClaimRow, observations: list[ImageObservation], history_text: str, evidence_text: str) -> str:
    obs_text = "\n".join(
        f"- {o.image_id}: issue_type={o.issue_type}, object_part={o.object_part}, "
        f"image_quality_ok={o.image_quality_ok}, shows_claimed_object={o.shows_claimed_object}, "
        f"notes: {o.observation_notes}"
        for o in observations
    )
    return f"""Claim object type: {row.claim_object}

Chat transcript:
{row.user_claim}

User history:
{history_text}

Minimum evidence requirements for this object type:
{evidence_text}

Per-image visual observations (already extracted from the images):
{obs_text}

Evaluate this claim and produce the structured decision."""


def run(row: ClaimRow, client, history: dict, requirements: list[dict], tracker: UsageTracker) -> ClaimDecision:
    observations = _run_stage_a(row, client, tracker)

    history_text = user_history_text(row.user_id, history)
    evidence_rows = evidence_requirements_for(row.claim_object, requirements)
    evidence_text = "\n".join(f"- {r['applies_to']}: {r['minimum_image_evidence']}" for r in evidence_rows)

    prompt = _build_stage_b_prompt(row, observations, history_text, evidence_text)
    result = client.generate(
        [prompt], response_schema=ClaimDecision, system_instruction=STAGE_B_SYSTEM_INSTRUCTION
    )
    tracker.record_call(result.prompt_tokens, result.output_tokens, num_images=0)
    return result.parsed
