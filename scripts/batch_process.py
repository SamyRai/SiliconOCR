"""Batch processing script for large inbox."""

import sys
import time
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.processor import DocumentProcessor


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Also log to file
    log_file = get_settings().cache_dir / "batch_processing.log"
    logger.add(log_file, level="DEBUG", rotation="10 MB")


def main():
    """Batch process all inbox documents."""
    setup_logging()

    # Get inbox path
    inbox_path = Path(__file__).parent.parent / "inbox"

    logger.info("=" * 70)
    logger.info("Batch Document Processing")
    logger.info("=" * 70)
    logger.info(f"Inbox: {inbox_path}")

    start_time = time.time()

    # Initialize processor
    processor = DocumentProcessor()

    # Get all PDFs
    pdf_files = sorted(inbox_path.glob("*.pdf"))
    total_files = len(pdf_files)

    logger.info(f"Found {total_files} PDF files")
    logger.info("=" * 70)

    # Process all files
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n[{i}/{total_files}] Processing {pdf_path.name}")
        doc = processor.process_pdf(pdf_path)
        results.append(doc)
        processor.save_result(doc)

    # Save summary
    processor.save_summary(results)

    # Final summary
    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Batch Processing Complete")
    logger.info("=" * 70)
    logger.info(f"Total time: {elapsed:.2f}s ({elapsed / 60:.1f} minutes)")
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Average: {elapsed / total_files:.2f}s per file")

    # Document statistics
    total_pages = sum(r.page_count for r in results)
    total_chars = sum(len(r.text) for r in results)

    logger.info(f"\nTotal pages: {total_pages}")
    logger.info(f"Total characters extracted: {total_chars:,}")

    # Document type breakdown
    doc_types = {}
    for doc in results:
        doc_types[doc.document_type] = doc_types.get(doc.document_type, 0) + 1

    logger.info("\nDocument Types:")
    for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_files) * 100
        logger.info(f"  {doc_type}: {count} ({percentage:.1f}%)")

    logger.info("=" * 70)

    output_dir = get_settings().cache_dir / "processed"
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - Individual results: {output_dir}/*.json")
    logger.info(f"  - Summary: {output_dir}/processing_summary.json")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
