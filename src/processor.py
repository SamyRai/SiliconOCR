"""Document processing orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from .classification import DocumentClassifier
from .config import Settings, get_settings
from .models import ProcessedDocument, ProcessingOptions, ProcessingStatus
from .pdf_utils import PDFProcessor
from .services import EmbeddingService, OCRService, TranslationService
from .storage import ProcessedDocumentStore


class DocumentProcessor:
    """Coordinate PDF extraction, OCR, embeddings, translation, and classification."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pdf_processor: PDFProcessor | None = None,
        ocr_service: OCRService | None = None,
        embedding_service: EmbeddingService | None = None,
        translation_service: TranslationService | None = None,
        classifier: DocumentClassifier | None = None,
        store: ProcessedDocumentStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.pdf_processor = pdf_processor or PDFProcessor(
            dpi=self.settings.pdf_dpi,
            max_pixels=self.settings.max_image_pixels,
        )
        self.ocr_service = ocr_service or OCRService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.translation_service = translation_service or TranslationService()
        self.classifier = classifier or DocumentClassifier()
        self.store = store or ProcessedDocumentStore(self.settings.cache_dir / "processed")

    def process_pdf(
        self,
        pdf_path: Path,
        options: ProcessingOptions | None = None,
    ) -> ProcessedDocument:
        """Process a single PDF file."""
        options = options or ProcessingOptions()
        start_time = time.time()
        pdf_path = Path(pdf_path)

        doc = ProcessedDocument(
            filename=pdf_path.name,
            filepath=str(pdf_path),
            file_size=pdf_path.stat().st_size,
            status=ProcessingStatus.PROCESSING,
        )

        try:
            page_count = self.pdf_processor.count_pages(pdf_path)
            doc.page_count = page_count
            logger.info(f"Processing {pdf_path.name} ({page_count} pages)")

            self._extract_text(pdf_path, doc, options)
            self._embed_text(doc, options)
            self._translate_text(doc, options)
            self._classify_document(doc, options)

            doc.status = ProcessingStatus.COMPLETED
            processing_time = time.time() - start_time
            doc.metadata["processing_time_seconds"] = round(processing_time, 2)

            log_msg = f"✓ Processed {pdf_path.name} in {processing_time:.2f}s "
            if options.enable_classification:
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
        pattern: str = "*.pdf",
        limit: int | None = None,
        options: ProcessingOptions | None = None,
    ) -> list[ProcessedDocument]:
        """Process all matching PDFs in an inbox directory."""
        inbox_dir = Path(inbox_dir)
        options = options or ProcessingOptions()

        if not inbox_dir.exists():
            logger.error(f"Inbox directory not found: {inbox_dir}")
            return []

        files = sorted(inbox_dir.glob(pattern))
        if limit:
            files = files[:limit]

        logger.info(f"Found {len(files)} files to process in {inbox_dir}")

        results = []
        for pdf_path in files:
            doc = self.process_pdf(pdf_path, options=options)
            results.append(doc)
            self.store.save_document(doc)

        self.store.save_summary(results)
        return results

    def _extract_text(
        self,
        pdf_path: Path,
        doc: ProcessedDocument,
        options: ProcessingOptions,
    ) -> None:
        text, success = self.pdf_processor.extract_text_native(pdf_path)

        if success and text.strip():
            doc.text = text
            doc.extracted_via_ocr = False
            doc.ocr_engine = "native"
            logger.debug(f"Native extraction successful: {len(text)} chars")
            return

        if not options.enable_ocr:
            logger.warning(f"No text extracted and OCR disabled for {pdf_path.name}")
            return

        logger.info(f"Native extraction failed, using OCR for {pdf_path.name}")
        images = self.pdf_processor.convert_to_images(pdf_path)

        page_texts = []
        for i, image in enumerate(images):
            logger.debug(f"OCR page {i + 1}/{len(images)}")
            page_texts.append(self.ocr_service.ocr_image_from_pil(image))

        doc.text = "\n\n".join(page_texts)
        doc.extracted_via_ocr = True
        doc.ocr_engine = self.settings.ocr_engine
        logger.debug(f"OCR extraction successful: {len(doc.text)} chars")

        if options.write_text_layer:
            self.pdf_processor.write_text_layer(pdf_path, page_texts)
            logger.info(f"✓ Written text layer to {pdf_path.name}")

    def _embed_text(self, doc: ProcessedDocument, options: ProcessingOptions) -> None:
        if not options.enable_embeddings or not doc.text.strip():
            return

        logger.debug("Generating embeddings...")
        doc.text_embedding = self.embedding_service.embed_text(doc.text[:5000])
        doc.embedding_model = self.settings.text_embedding_model

    def _translate_text(self, doc: ProcessedDocument, options: ProcessingOptions) -> None:
        if not options.enable_translation or not doc.text.strip():
            return

        logger.debug(f"Translating text to {options.target_language}...")
        try:
            doc.translated_text = self.translation_service.translate(
                doc.text[:5000],
                src_lang="de",
                tgt_lang=options.target_language,
                use_marian=True,
            )
            doc.target_language = options.target_language
            doc.source_language = "de"
        except Exception as e:
            logger.warning(f"Translation failed: {e}")

    def _classify_document(self, doc: ProcessedDocument, options: ProcessingOptions) -> None:
        if not options.enable_classification:
            return

        result = self.classifier.classify(doc.filename, doc.text)
        doc.document_type = result.document_type
        doc.confidence = result.confidence
