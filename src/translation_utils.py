"""Shared translation workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .services.translation import TranslationService
from .text_utils import chunk_text_by_paragraph, chunk_text_by_sentence

ChunkStrategy = Literal["paragraph", "sentence"]


@dataclass(frozen=True)
class TranslationTextResult:
    """Translated text plus metadata useful to scripts."""

    text: str
    chunk_count: int


def translate_german_text_to_english(
    text: str,
    translation_service: TranslationService,
    *,
    max_chars: int = 4000,
    strategy: ChunkStrategy = "paragraph",
    joiner: str = "\n\n",
) -> TranslationTextResult:
    """Translate German text to English using shared chunking behavior."""
    if strategy == "paragraph":
        chunks = chunk_text_by_paragraph(text, max_chars=max_chars)
    else:
        chunks = chunk_text_by_sentence(text, max_chars=max_chars)

    translated_chunks = translation_service.translate_german_to_english_batch(chunks)
    return TranslationTextResult(text=joiner.join(translated_chunks), chunk_count=len(chunks))
