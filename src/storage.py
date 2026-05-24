"""Storage for processed document JSON results."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from .models import ProcessedDocument, ProcessingStatus


class ProcessedDocumentStore:
    """Read and write processed document result files."""

    SUMMARY_FILENAME = "processing_summary.json"

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def document_path(self, doc: ProcessedDocument) -> Path:
        """Return the output JSON path for a processed document."""
        return self.output_dir / f"{doc.filename}.json"

    def save_document(self, doc: ProcessedDocument) -> Path:
        """Save an individual processed document."""
        output_file = self.document_path(doc)
        output_file.write_text(
            json.dumps(doc.model_dump(), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(f"Saved result to {output_file}")
        return output_file

    def save_summary(self, results: list[ProcessedDocument]) -> Path:
        """Save a processing summary."""
        summary_file = self.output_dir / self.SUMMARY_FILENAME
        summary = self.build_summary(results)
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"✓ Saved summary to {summary_file}")
        logger.info(f"Summary: {summary['successful']}/{summary['total_processed']} successful")
        return summary_file

    def build_summary(self, results: list[ProcessedDocument]) -> dict[str, Any]:
        """Build a serializable processing summary."""
        document_types: dict[str, int] = {}
        for doc in results:
            doc_type = str(doc.document_type)
            document_types[doc_type] = document_types.get(doc_type, 0) + 1

        return {
            "total_processed": len(results),
            "successful": sum(1 for d in results if d.status == ProcessingStatus.COMPLETED),
            "failed": sum(1 for d in results if d.status == ProcessingStatus.FAILED),
            "document_types": document_types,
            "total_pages": sum(d.page_count for d in results),
            "total_characters": sum(len(d.text) for d in results),
        }

    def iter_document_files(self, pattern: str = "*.json") -> Iterator[Path]:
        """Iterate processed document JSON files, excluding the summary."""
        for json_file in sorted(self.output_dir.glob(pattern)):
            if json_file.name == self.SUMMARY_FILENAME:
                continue
            yield json_file

    def iter_documents(self, pattern: str = "*.json") -> Iterator[dict[str, Any]]:
        """Iterate processed document JSON payloads."""
        for json_file in self.iter_document_files(pattern):
            yield json.loads(json_file.read_text(encoding="utf-8"))
