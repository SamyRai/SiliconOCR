from unittest.mock import patch

import pytest

from src.models import DocumentType, ProcessingStatus
from src.processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_classify_document_invoice(processor):
    doc_type, conf = processor.classify_document("invoice_october.pdf", "payment due in 30 days")
    assert doc_type == DocumentType.INVOICE
    assert conf > 0.0


def test_classify_document_contract(processor):
    doc_type, conf = processor.classify_document("rental_agreement.pdf", "This vertrag is binding.")
    assert doc_type == DocumentType.CONTRACT
    assert conf > 0.0


def test_classify_document_other(processor):
    doc_type, conf = processor.classify_document("random_file.txt", "just some words")
    assert doc_type == DocumentType.OTHER
    assert conf == 0.0


@patch("src.processor.PDFProcessor.count_pages")
@patch("src.processor.PDFProcessor.extract_text_native")
@patch("src.processor.EmbeddingService.embed_text")
def test_process_pdf_native_success(mock_embed, mock_extract, mock_count, processor, mock_pdf_path):
    # Setup mocks
    mock_count.return_value = 2
    mock_extract.return_value = ("Sample extracted text", True)
    mock_embed.return_value = [0.1, 0.2, 0.3]

    doc = processor.process_pdf(
        mock_pdf_path, enable_ocr=True, enable_embeddings=True, enable_classification=True
    )

    assert doc.status == ProcessingStatus.COMPLETED
    assert doc.page_count == 2
    assert doc.text == "Sample extracted text"
    assert doc.extracted_via_ocr is False
    assert doc.text_embedding == [0.1, 0.2, 0.3]

    mock_extract.assert_called_once_with(mock_pdf_path)
    mock_embed.assert_called_once_with("Sample extracted text")
