from pathlib import Path

import pytest

from src.classification import DocumentClassifier
from src.config import Settings
from src.models import DocumentType, ProcessedDocument, ProcessingOptions, ProcessingStatus
from src.processor import DocumentProcessor
from src.storage import ProcessedDocumentStore


class FakePDFProcessor:
    def __init__(self, *, native_text: str = "Sample extracted text", native_success: bool = True):
        self.native_text = native_text
        self.native_success = native_success
        self.text_layer_calls: list[tuple[Path, list[str]]] = []

    def count_pages(self, pdf_path: Path) -> int:
        return 2

    def extract_text_native(self, pdf_path: Path) -> tuple[str, bool]:
        return self.native_text, self.native_success

    def convert_to_images(self, pdf_path: Path) -> list[object]:
        return [object(), object()]

    def write_text_layer(self, pdf_path: Path, page_texts: list[str]) -> None:
        self.text_layer_calls.append((pdf_path, page_texts))


class FakeOCRService:
    def ocr_image_from_pil(self, image: object) -> str:
        return "OCR text"


class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeTranslationService:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def translate(self, text: str, src_lang: str, tgt_lang: str, use_marian: bool = False) -> str:
        if self.fail:
            raise RuntimeError("translation unavailable")
        return f"translated to {tgt_lang}"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        cache_dir=tmp_path / "cache",
        model_cache_dir=tmp_path / "models",
        text_embedding_model="test-embedding-model",
        ocr_engine="fakeocr",
    )


def make_processor(
    settings: Settings,
    *,
    pdf_processor: FakePDFProcessor | None = None,
    translation_service: FakeTranslationService | None = None,
) -> DocumentProcessor:
    return DocumentProcessor(
        settings=settings,
        pdf_processor=pdf_processor or FakePDFProcessor(),
        ocr_service=FakeOCRService(),
        embedding_service=FakeEmbeddingService(),
        translation_service=translation_service or FakeTranslationService(),
        classifier=DocumentClassifier(),
        store=ProcessedDocumentStore(settings.cache_dir / "processed"),
    )


def test_classifier_invoice():
    result = DocumentClassifier().classify("invoice_october.pdf", "payment due in 30 days")
    assert result.document_type == DocumentType.INVOICE
    assert result.confidence > 0.0


def test_classifier_contract():
    result = DocumentClassifier().classify("rental_agreement.pdf", "This vertrag is binding.")
    assert result.document_type == DocumentType.CONTRACT
    assert result.confidence > 0.0


def test_classifier_other():
    result = DocumentClassifier().classify("random_file.txt", "just some words")
    assert result.document_type == DocumentType.OTHER
    assert result.confidence == 0.0


def test_process_pdf_native_success(settings: Settings, mock_pdf_path: Path):
    processor = make_processor(settings)
    options = ProcessingOptions(enable_classification=True)

    doc = processor.process_pdf(mock_pdf_path, options=options)

    assert doc.status == ProcessingStatus.COMPLETED
    assert doc.page_count == 2
    assert doc.text == "Sample extracted text"
    assert doc.extracted_via_ocr is False
    assert doc.ocr_engine == "native"
    assert doc.text_embedding == [0.1, 0.2, 0.3]
    assert doc.embedding_model == "test-embedding-model"


def test_process_pdf_ocr_fallback_writes_text_layer(settings: Settings, mock_pdf_path: Path):
    pdf_processor = FakePDFProcessor(native_text="", native_success=False)
    processor = make_processor(settings, pdf_processor=pdf_processor)
    options = ProcessingOptions(write_text_layer=True)

    doc = processor.process_pdf(mock_pdf_path, options=options)

    assert doc.status == ProcessingStatus.COMPLETED
    assert doc.text == "OCR text\n\nOCR text"
    assert doc.extracted_via_ocr is True
    assert doc.ocr_engine == "fakeocr"
    assert pdf_processor.text_layer_calls == [(mock_pdf_path, ["OCR text", "OCR text"])]


def test_process_pdf_translation_failure_keeps_document_successful(
    settings: Settings, mock_pdf_path: Path
):
    processor = make_processor(settings, translation_service=FakeTranslationService(fail=True))
    options = ProcessingOptions(enable_translation=True)

    doc = processor.process_pdf(mock_pdf_path, options=options)

    assert doc.status == ProcessingStatus.COMPLETED
    assert doc.translated_text is None
    assert doc.error_message is None


def test_process_pdf_classification_disabled(settings: Settings, mock_pdf_path: Path):
    processor = make_processor(settings)
    options = ProcessingOptions(enable_classification=False)

    doc = processor.process_pdf(mock_pdf_path, options=options)

    assert doc.status == ProcessingStatus.COMPLETED
    assert doc.document_type == DocumentType.OTHER
    assert doc.confidence == 0.0


def test_process_inbox_missing_dir_returns_empty(settings: Settings, tmp_path: Path):
    processor = make_processor(settings)

    assert processor.process_inbox(tmp_path / "missing") == []


def test_store_saves_documents_summary_and_excludes_summary(tmp_path: Path):
    store = ProcessedDocumentStore(tmp_path / "processed")
    doc = ProcessedDocument(
        filename="document.pdf",
        filepath="/tmp/document.pdf",
        file_size=100,
        status=ProcessingStatus.COMPLETED,
        text="hello",
        page_count=1,
        document_type=DocumentType.LETTER,
    )

    document_path = store.save_document(doc)
    summary_path = store.save_summary([doc])
    document_files = list(store.iter_document_files())

    assert document_path.exists()
    assert summary_path.exists()
    assert document_files == [document_path]
    assert list(store.iter_documents())[0]["filename"] == "document.pdf"
