"""Image loading with real-bytes mime sniffing.

Note: files under dataset/images/**/*.jpg are actually WebP-encoded
despite the .jpg extension, so the extension cannot be trusted when
calling the Gemini API.
"""

from __future__ import annotations

from pathlib import Path

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # WEBP within a RIFF container; verified below
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def sniff_mime_type(data: bytes) -> str:
    for magic, mime in _MAGIC_BYTES:
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    if data[4:8] == b"ftyp" and data[8:12] in (b"avif", b"avis"):
        return "image/avif"
    raise ValueError("Unrecognized image format (not jpeg/png/webp/gif/avif)")


def load_image_bytes(path: Path) -> tuple[bytes, str]:
    data = path.read_bytes()
    mime_type = sniff_mime_type(data)
    return data, mime_type
