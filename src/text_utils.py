"""Shared text processing helpers."""

from __future__ import annotations


def chunk_text_by_paragraph(text: str, max_chars: int = 4000) -> list[str]:
    """Split text into chunks, keeping paragraphs together when possible."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for paragraph in text.split("\n\n"):
        paragraph_length = len(paragraph)
        if current_chunk and current_length + paragraph_length > max_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length
        else:
            current_chunk.append(paragraph)
            current_length += paragraph_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def chunk_text_by_sentence(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into chunks, trying to keep sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current_chunk = ""
    sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
