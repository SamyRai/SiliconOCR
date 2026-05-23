"""Document processor for inbox files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from .config import get_settings
from .models import DocumentType, ProcessedDocument, ProcessingStatus
from .pdf_utils import PDFProcessor
from .services import EmbeddingService, OCRService, TranslationService


class DocumentProcessor:
    """Process documents from inbox with OCR, embeddings, and classification."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.ocr_service = OCRService()
        self.embedding_service = EmbeddingService()
        self.translation_service = TranslationService()
        self.pdf_processor = PDFProcessor(
            dpi=self.settings.pdf_dpi,
            max_pixels=self.settings.max_image_pixels,
        )

        # Output directories
        self.output_dir = self.settings.cache_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classify_document(self, filename: str, text: str) -> tuple[DocumentType, float]:
        """Simple rule-based classification.

        Can be enhanced with ML classification later.
        """
        filename_lower = filename.lower()
        text_lower = text.lower()

        # Keywords for classification
        keywords = {
            DocumentType.INVOICE: ["invoice", "rechnung", "faktura", "payment"],
            DocumentType.UTILITIES: [
                "utilities",
                "nebenkosten",
                "betriebskosten",
                "heating",
            ],
            DocumentType.PAYMENT_REMINDER: ["mahnung", "reminder", "payment reminder"],
            DocumentType.CONTRACT: ["vertrag", "contract", "vereinbarung"],
            DocumentType.FORM: ["formular", "form", "antrag", "application"],
            DocumentType.LETTER: ["brief", "letter", "mitteilung"],
        }

        scores = {}
        for doc_type, terms in keywords.items():
            score = 0.0
            for term in terms:
                if term in filename_lower:
                    score += 2.0  # Filename match is stronger
                if term in text_lower:
                    score += 1.0
            scores[doc_type] = score

        # Get best match
        if scores:
            best_type = max(scores, key=lambda k: scores[k])
            max_score = scores[best_type]
            if max_score > 0:
                confidence = min(1.0, max_score / 5.0)  # Normalize
                return best_type, confidence

        return DocumentType.OTHER, 0.0

    # PDF text-layer writing moved to PDFProcessor (see src/pdf_utils.py)

    def process_pdf(
        self,
        pdf_path: Path,
        enable_ocr: bool = True,
        enable_embeddings: bool = True,
        enable_classification: bool = False,
        write_text_layer: bool = False,
        enable_translation: bool = False,
        target_language: str = "en",
    ) -> ProcessedDocument:
        """Process a single PDF file.

        Steps:
        1. Try native text extraction
        2. Fall back to OCR if needed (if enable_ocr)
        3. Generate embeddings (if enable_embeddings)
        4. Classify document (if enable_classification)
        5. Write text layer to PDF (if write_text_layer and OCR was used)
        """
        start_time = time.time()

        doc = ProcessedDocument(
            filename=pdf_path.name,
            filepath=str(pdf_path),
            file_size=pdf_path.stat().st_size,
            status=ProcessingStatus.PROCESSING,
        )

        try:
            # Count pages
            page_count = self.pdf_processor.count_pages(pdf_path)
            doc.page_count = page_count
            logger.info(f"Processing {pdf_path.name} ({page_count} pages)")

            # Try native extraction first
            text, success = self.pdf_processor.extract_text_native(pdf_path)

            if success and text.strip():
                doc.text = text
                doc.extracted_via_ocr = False
                doc.ocr_engine = "native"
                logger.debug(f"Native extraction successful: {len(text)} chars")
            elif enable_ocr:
                # Fall back to OCR
                logger.info(f"Native extraction failed, using OCR for {pdf_path.name}")
                images = self.pdf_processor.convert_to_images(pdf_path)

                text_parts = []
                for i, image in enumerate(images):
                    logger.debug(f"OCR page {i + 1}/{len(images)}")
                    page_text = self.ocr_service.ocr_image_from_pil(image)
                    text_parts.append(page_text)

                doc.text = "\n\n".join(text_parts)
                doc.extracted_via_ocr = True
                doc.ocr_engine = self.settings.ocr_engine
                logger.debug(f"OCR extraction successful: {len(doc.text)} chars")

                # Write text layer back to PDF if requested
                if write_text_layer:
                    logger.warning(
                        "write_text_layer requested for {} but PDF text-layer writing "
                        "is not implemented yet; skipping",
                        pdf_path.name,
                    )
            else:
                logger.warning(f"No text extracted and OCR disabled for {pdf_path.name}")

            # Generate embeddings
            if enable_embeddings and doc.text.strip():
                logger.debug("Generating embeddings...")
                embedding = self.embedding_service.embed_text(doc.text[:5000])  # Limit
                doc.text_embedding = embedding
                doc.embedding_model = self.settings.text_embedding_model

            # Translate document
            if enable_translation and doc.text.strip():
                logger.debug(f"Translating text to {target_language}...")
                try:
                    # Using Marian for de->en as default, but can be configured
                    # We limit text to 5000 characters to prevent huge processing time
                    translated_text = self.translation_service.translate(
                        doc.text[:5000], src_lang="de", tgt_lang=target_language, use_marian=True
                    )
                    doc.translated_text = translated_text
                    doc.target_language = target_language
                    doc.source_language = "de"
                except Exception as e:
                    logger.warning(f"Translation failed: {e}")

            # Classify document
            if enable_classification:
                doc_type, confidence = self.classify_document(pdf_path.name, doc.text)
                doc.document_type = doc_type
                doc.confidence = confidence

            doc.status = ProcessingStatus.COMPLETED
            processing_time = time.time() - start_time
            doc.metadata["processing_time_seconds"] = round(processing_time, 2)

            # Log message
            log_msg = f"✓ Processed {pdf_path.name} in {processing_time:.2f}s "
            if enable_classification:
                log_msg += f"(type: {doc.document_type}, {len(doc.text)} chars)"
            else:
                log_msg += f"({len(doc.text)} chars)"
            logger.info(log_msg)

        except Exception as e:
            doc.status = ProcessingStatus.FAILED
            doc.error_message = str(e)
            logger.error(f"✗ Failed to process {pdf_path.name}: {e}")

        return doc

    def process_inbox(
        self,
        inbox_dir: Path,
        file_pattern: str = "*.pdf",
        limit: int | None = None,
        enable_ocr: bool = True,
        enable_embeddings: bool = True,
        enable_classification: bool = False,
        write_text_layer: bool = False,
        enable_translation: bool = False,
        target_language: str = "en",
    ) -> list[ProcessedDocument]:
        """Process all files in inbox directory.

        Args:
            inbox_dir: Path to inbox directory
            file_pattern: File pattern to match (e.g., "*.pdf")
            limit: Maximum number of files to process
            enable_ocr: Whether to perform OCR on documents without text
            enable_embeddings: Whether to generate text embeddings
            enable_classification: Whether to classify documents
            write_text_layer: Whether to write OCR text back to PDF files

        Returns:
            List of processed documents
        """
        inbox_dir = Path(inbox_dir)

        if not inbox_dir.exists():
            logger.error(f"Inbox directory not found: {inbox_dir}")
            return []

        files = sorted(inbox_dir.glob(file_pattern))
        if limit:
            files = files[:limit]

        logger.info(f"Found {len(files)} files to process in {inbox_dir}")

        results = []
        for pdf_path in files:
            doc = self.process_pdf(
                pdf_path,
                enable_ocr=enable_ocr,
                enable_embeddings=enable_embeddings,
                enable_classification=enable_classification,
                write_text_layer=write_text_layer,
                enable_translation=enable_translation,
                target_language=target_language,
            )
            results.append(doc)

            # Save individual result
            self.save_result(doc)

        # Save summary
        self.save_summary(results)

        return results

    def save_result(self, doc: ProcessedDocument) -> None:
        """Save individual document result to JSON."""
        output_file = self.output_dir / f"{doc.filename}.json"

        with open(output_file, "w") as f:
            json.dump(doc.model_dump(), f, indent=2, default=str)

        logger.debug(f"Saved result to {output_file}")

    def save_summary(self, results: list[ProcessedDocument]) -> None:
        """Save processing summary."""
        summary_file = self.output_dir / "processing_summary.json"

        summary: dict[str, Any] = {
            "total_processed": len(results),
            "successful": sum(1 for d in results if d.status == ProcessingStatus.COMPLETED),
            "failed": sum(1 for d in results if d.status == ProcessingStatus.FAILED),
            "document_types": {},
            "total_pages": sum(d.page_count for d in results),
            "total_characters": sum(len(d.text) for d in results),
        }

        # Count document types
        for doc in results:
            doc_type = doc.document_type
            if doc_type not in summary["document_types"]:
                summary["document_types"][doc_type] = 0
            summary["document_types"][doc_type] += 1

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Saved summary to {summary_file}")
        logger.info(f"Summary: {summary['successful']}/{summary['total_processed']} successful")
