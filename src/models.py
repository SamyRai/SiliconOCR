"""Models for document processing."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(StrEnum):
    """Document type classification."""

    INVOICE = "invoice"
    CONTRACT = "contract"
    LETTER = "letter"
    FORM = "form"
    UTILITIES = "utilities"
    PAYMENT_REMINDER = "payment_reminder"
    OTHER = "other"


class ProcessingStatus(StrEnum):
    """Document processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingOptions(BaseModel):
    """Options controlling a document processing run."""

    enable_ocr: bool = True
    enable_embeddings: bool = True
    enable_classification: bool = False
    write_text_layer: bool = False
    enable_translation: bool = False
    target_language: str = "en"


class ClassificationResult(BaseModel):
    """Result of classifying a document."""

    document_type: DocumentType
    confidence: float


class ProcessedDocument(BaseModel):
    """Processed document metadata."""

    filename: str
    filepath: str
    file_size: int
    processed_at: datetime = Field(default_factory=datetime.now)
    status: ProcessingStatus

    # Content
    text: str = ""
    page_count: int = 0

    # Embeddings
    text_embedding: list[float] | None = None
    embedding_model: str = ""

    # Translation (optional)
    translated_text: str | None = None
    source_language: str | None = None
    target_language: str | None = None

    # Classification
    document_type: DocumentType = DocumentType.OTHER
    confidence: float = 0.0

    # OCR metadata
    ocr_engine: str = ""
    extracted_via_ocr: bool = False

    # Error handling
    error_message: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class ProcessingResult(BaseModel):
    """Result of processing operation."""

    success: bool
    documents_processed: int
    documents_failed: int
    processing_time: float
    results: list[ProcessedDocument]
    errors: list[str] = Field(default_factory=list)
