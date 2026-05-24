"""Main script to process inbox documents."""

import sys
from pathlib import Path

from loguru import logger

from src.models import ProcessingOptions
from src.processor import DocumentProcessor


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def main():
    """Process documents from inbox."""
    import argparse

    parser = argparse.ArgumentParser(description="Process documents from inbox")
    parser.add_argument(
        "--inbox",
        type=Path,
        default=Path(__file__).parent.parent / "inbox",
        help="Path to inbox directory (default: ../inbox)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern to match (default: *.pdf)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR text extraction (default: True if no actions specified)",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Enable embedding generation (default: True if no actions specified)",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Enable document classification (default: False)",
    )
    parser.add_argument(
        "--write-text-layer",
        action="store_true",
        help="Write OCR text back to PDF as invisible text layer",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # If no actions specified, default to OCR + embed
    if not any([args.ocr, args.embed, args.classify]):
        args.ocr = True
        args.embed = True

    logger.info("=" * 60)
    logger.info("Document Processing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Inbox: {args.inbox}")
    logger.info(f"Pattern: {args.pattern}")
    if args.limit:
        logger.info(f"Limit: {args.limit} files")

    # Log enabled actions
    actions = []
    if args.ocr:
        actions.append("OCR")
    if args.embed:
        actions.append("Embeddings")
    if args.classify:
        actions.append("Classification")
    logger.info(f"Actions: {', '.join(actions)}")
    logger.info("=" * 60)

    # Initialize processor
    processor = DocumentProcessor()
    options = ProcessingOptions(
        enable_ocr=args.ocr,
        enable_embeddings=args.embed,
        enable_classification=args.classify,
        write_text_layer=args.write_text_layer,
    )

    # Process inbox
    results = processor.process_inbox(
        inbox_dir=args.inbox,
        pattern=args.pattern,
        limit=args.limit,
        options=options,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)

    successful = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    # Document type breakdown
    doc_types = {}
    for doc in results:
        doc_types[doc.document_type] = doc_types.get(doc.document_type, 0) + 1

    logger.info("\nDocument Types:")
    for doc_type, count in sorted(doc_types.items()):
        logger.info(f"  {doc_type}: {count}")

    logger.info("=" * 60)

    # Exit with error if any failed
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
