"""Script to translate German documents to English."""

import json
import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.services import TranslationService


def translate_document(doc_path: Path, translation_service: TranslationService) -> dict:
    """Translate a single processed document from German to English."""
    with open(doc_path) as f:
        doc = json.load(f)

    filename = doc.get("filename", "")
    text = doc.get("text", "")

    if not text.strip():
        logger.warning(f"No text to translate in {filename}")
        return doc

    logger.info(f"Translating {filename} ({len(text)} chars)")

    # Split into chunks if too long (MarianMT has token limits)
    max_chunk = 4000  # Conservative limit
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in text.split("\n\n"):
        para_len = len(paragraph)
        if current_length + para_len > max_chunk and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_length = para_len
        else:
            current_chunk.append(paragraph)
            current_length += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Translate chunks
    logger.debug(f"Translating {len(chunks)} chunks")
    translated_chunks = translation_service.translate_german_to_english_batch(chunks)
    translated_text = "\n\n".join(translated_chunks)

    # Update document
    doc["translated_text"] = translated_text
    doc["source_language"] = "de"
    doc["target_language"] = "en"

    logger.info(f"✓ Translated {filename} ({len(translated_text)} chars)")

    return doc


def main():
    """Translate all processed German documents to English."""
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Translate German documents to English")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory with processed documents",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for translated documents",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf.json",
        help="File pattern to match",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to translate",
    )

    args = parser.parse_args()

    # Default directories
    settings = get_settings()
    input_dir = args.input_dir or (settings.cache_dir / "processed")
    output_dir = args.output_dir or (settings.cache_dir / "translated")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("German to English Translation")
    logger.info("=" * 70)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pattern: {args.pattern}")
    if args.limit:
        logger.info(f"Limit: {args.limit} files")
    logger.info("=" * 70)

    # Find documents
    json_files = sorted(input_dir.glob(args.pattern))
    json_files = [f for f in json_files if f.name != "processing_summary.json"]

    if args.limit:
        json_files = json_files[: args.limit]

    logger.info(f"Found {len(json_files)} documents to translate")

    # Initialize translation service
    translation_service = TranslationService()

    # Translate documents
    translated_count = 0
    for i, doc_path in enumerate(json_files, 1):
        logger.info(f"\n[{i}/{len(json_files)}]")
        try:
            translated_doc = translate_document(doc_path, translation_service)

            # Save translated document
            output_path = output_dir / doc_path.name
            with open(output_path, "w") as f:
                json.dump(translated_doc, f, indent=2, ensure_ascii=False)

            translated_count += 1
        except Exception as e:
            logger.error(f"Failed to translate {doc_path.name}: {e}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Translation Complete")
    logger.info("=" * 70)
    logger.info(f"Total documents: {len(json_files)}")
    logger.info(f"Successfully translated: {translated_count}")
    logger.info(f"Failed: {len(json_files) - translated_count}")
    logger.info(f"\nTranslated documents saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
