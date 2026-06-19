"""Accumulates model-call/token/runtime stats so evaluation can report on them offline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UsageTracker:
    strategy: str
    model: str = "gemini-2.5-flash"
    call_count: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    images_processed: int = 0
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    def record_call(self, prompt_tokens: int, output_tokens: int, num_images: int = 0) -> None:
        self.call_count += 1
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens
        self.images_processed += num_images

    def finish(self) -> None:
        self.end_time = time.monotonic()

    def to_dict(self) -> dict:
        runtime_seconds = (self.end_time or time.monotonic()) - self.start_time
        return {
            "strategy": self.strategy,
            "model": self.model,
            "model_calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "images_processed": self.images_processed,
            "runtime_seconds": round(runtime_seconds, 2),
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
