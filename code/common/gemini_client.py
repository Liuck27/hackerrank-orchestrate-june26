"""Thin wrapper over the google-genai SDK with retry/backoff and a simple rate limiter."""

from __future__ import annotations

import os
import threading
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from common.llm_client import GenerateResult, ImagePart, Part

DEFAULT_MODEL = "gemini-2.5-flash"


class RateLimiter:
    """Caps requests to roughly `rpm` per rolling 60s window."""

    def __init__(self, rpm: int):
        self.rpm = max(rpm, 1)
        self._lock = threading.Lock()
        self._call_times: list[float] = []

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._call_times = [t for t in self._call_times if now - t < 60]
            if len(self._call_times) >= self.rpm:
                sleep_for = 60 - (now - self._call_times[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                self._call_times = [t for t in self._call_times if now - t < 60]
            self._call_times.append(time.monotonic())


def _to_gemini_content(part: Part):
    if isinstance(part, ImagePart):
        return types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
    return part


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL, rpm: int = 10, max_retries: int = 5):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self._rate_limiter = RateLimiter(rpm)

    def generate(
        self,
        parts: list[Part],
        response_schema: type[BaseModel],
        system_instruction: str | None = None,
    ) -> GenerateResult:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        )
        contents = [_to_gemini_content(p) for p in parts]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._rate_limiter.wait()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                usage = response.usage_metadata
                return GenerateResult(
                    parsed=response.parsed,
                    prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                )
            except Exception as exc:  # noqa: BLE001 - retry on any transient API error
                last_error = exc
                is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                backoff = (2**attempt) if is_rate_limit else 1
                time.sleep(backoff)
        raise RuntimeError(f"Gemini call failed after {self.max_retries} attempts: {last_error}") from last_error
