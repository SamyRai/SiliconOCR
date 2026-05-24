"""Export processed documents to various formats."""

import csv
import json
import sys
from pathlib import Path

from loguru import logger

from src.config import get_settings
from src.storage import ProcessedDocumentStore


def export_to_csv(output_file: Path):
    """Export processed documents to CSV."""
    store = ProcessedDocumentStore(get_settings().cache_dir / "processed")
    json_files = list(store.iter_document_files())

    logger.info(f"Exporting {len(json_files)} documents to CSV")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "filename",
                "document_type",
                "page_count",
                "character_count",
                "ocr_engine",
                "extracted_via_ocr",
                "confidence",
                "status",
                "text_preview",
            ]
        )

        # Data
        for json_file in sorted(json_files):
            with open(json_file) as jf:
                doc = json.load(jf)

            text = doc.get("text", "")
            preview = text[:100].replace("\n", " ") if text else ""

            writer.writerow(
                [
                    doc.get("filename", ""),
                    doc.get("document_type", ""),
                    doc.get("page_count", 0),
                    len(text),
                    doc.get("ocr_engine", ""),
                    doc.get("extracted_via_ocr", False),
                    doc.get("confidence", 0.0),
                    doc.get("status", ""),
                    preview,
                ]
            )

    logger.info(f"✓ Exported to {output_file}")


def export_full_text(output_file: Path):
    """Export all text to single text file."""
    store = ProcessedDocumentStore(get_settings().cache_dir / "processed")
    json_files = list(store.iter_document_files())

    logger.info(f"Exporting text from {len(json_files)} documents")

    with open(output_file, "w", encoding="utf-8") as f:
        for json_file in sorted(json_files):
            with open(json_file) as jf:
                doc = json.load(jf)

            f.write("=" * 80 + "\n")
            f.write(f"FILE: {doc.get('filename', '')}\n")
            f.write(f"TYPE: {doc.get('document_type', '')}\n")
            f.write(f"PAGES: {doc.get('page_count', 0)}\n")
            f.write("=" * 80 + "\n")
            f.write(doc.get("text", ""))
            f.write("\n\n")

    logger.info(f"✓ Exported to {output_file}")


def main():
    """Export processed documents."""
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Export processed documents")
    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "text"],
        default="csv",
        help="Export format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path",
    )

    args = parser.parse_args()

    # Default output paths
    if args.output is None:
        if args.format == "csv":
            args.output = Path("documents_export.csv")
        else:
            args.output = Path("documents_export.txt")

    logger.info(f"Export format: {args.format}")
    logger.info(f"Output file: {args.output}")

    if args.format == "csv":
        export_to_csv(args.output)
    else:
        export_full_text(args.output)


if __name__ == "__main__":
    main()
