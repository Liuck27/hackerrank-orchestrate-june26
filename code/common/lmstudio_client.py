"""Local-inference backend: LM Studio's OpenAI-compatible API (no token cost, no daily quota).

Same generate() signature as GeminiClient, so strategies are backend-agnostic.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ValidationError

from common.llm_client import GenerateResult, ImagePart, Part

DEFAULT_MODEL = "google/gemma-4-e4b:2"
DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Fields that must be scalars per the schema, but this local model sometimes wraps in a
# single-element list (e.g. issue_type: ["dent"] instead of "dent").
_SCALAR_FIELDS = {
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "valid_image",
    "severity",
    "image_id",
    "image_quality_ok",
    "shows_claimed_object",
    "shows_claimed_part",
    "embedded_text_or_instructions",
    "visible_severity",
    "observation_notes",
}


def _unwrap_singleton_lists(data: dict) -> dict:
    return {
        key: value[0] if key in _SCALAR_FIELDS and isinstance(value, list) and len(value) == 1 else value
        for key, value in data.items()
    }


_MAX_DIMENSION = 1536


def _to_png_base64(data: bytes) -> str:
    """LM Studio's vision endpoint rejects WebP and very large images; re-encode to PNG
    and downscale anything above _MAX_DIMENSION on its longest side."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if max(image.size) > _MAX_DIMENSION:
        image.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _to_openai_content_part(part: Part) -> dict:
    if isinstance(part, ImagePart):
        b64 = _to_png_base64(part.data)
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    return {"type": "text", "text": part}


def _validate_with_unknown_fallback(response_schema: type[BaseModel], data: dict) -> BaseModel:
    """Coerce invalid enum (Literal) values to 'unknown' instead of failing outright.

    Local models occasionally emit a value outside the allowed enum (e.g. confusing the claim
    object type with object_part). Since generation runs at temperature=0, a blind retry of the
    same prompt reproduces the same invalid value, so this repairs the parsed JSON in place.
    """
    for _ in range(len(data) + 1):
        try:
            return response_schema.model_validate(data)
        except ValidationError as exc:
            literal_errors = [e for e in exc.errors() if e["type"] == "literal_error" and len(e["loc"]) == 1]
            if not literal_errors:
                raise
            for error in literal_errors:
                data[error["loc"][0]] = "unknown"
    return response_schema.model_validate(data)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


class LMStudioClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        rpm: int = 0,
        max_retries: int = 3,
    ):
        self._client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.max_retries = max_retries

    def generate(
        self,
        parts: list[Part],
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
    ) -> GenerateResult:
        user_content = [_to_openai_content_part(p) for p in parts]
        schema_text = json.dumps(response_schema.model_json_schema())
        json_instruction = (
            "Respond with ONLY a single valid JSON object matching this JSON schema, "
            f"no markdown fences, no commentary: {schema_text}"
        )
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": f"{system_instruction}\n\n{json_instruction}"})
        else:
            messages.append({"role": "system", "content": json_instruction})
        messages.append({"role": "user", "content": user_content})

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
                raw_text = response.choices[0].message.content
                parsed_json = _unwrap_singleton_lists(_extract_json(raw_text))
                parsed = _validate_with_unknown_fallback(response_schema, parsed_json)
                usage = response.usage
                completion_details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
                return GenerateResult(
                    parsed=parsed,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    reasoning_tokens=reasoning_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - retry on parse/validation/connection errors
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"LM Studio call failed after {self.max_retries} attempts: {last_error}") from last_error
