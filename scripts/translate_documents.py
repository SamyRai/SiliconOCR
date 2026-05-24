"""Script to translate German documents to English."""

import json
import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.services import TranslationService
from src.storage import ProcessedDocumentStore
from src.translation_utils import translate_german_text_to_english


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

    translation = translate_german_text_to_english(text, translation_service)
    logger.debug(f"Translated {translation.chunk_count} chunks")

    # Update document
    doc["translated_text"] = translation.text
    doc["source_language"] = "de"
    doc["target_language"] = "en"

    logger.info(f"✓ Translated {filename} ({len(translation.text)} chars)")

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
    input_store = ProcessedDocumentStore(input_dir)
    json_files = list(input_store.iter_document_files(args.pattern))

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
