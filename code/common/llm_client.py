"""Backend-agnostic content parts and result type shared by GeminiClient and LMStudioClient.

Strategies build prompts out of `str` and `ImagePart` only, so they work unchanged
against either backend (cloud Gemini or local LM Studio).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class ImagePart:
    data: bytes
    mime_type: str


Part = str | ImagePart


@dataclass
class GenerateResult:
    parsed: BaseModel
    prompt_tokens: int
    output_tokens: int
