"""FastAPI service for OCR and document processing."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.processor import DocumentProcessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.processor = DocumentProcessor()
    yield
    # Cleanup if necessary


app = FastAPI(
    title="SiliconOCR",
    description="Modern OCR, Embeddings, and Translation Service optimized for Apple Silicon",
    version="0.1.0",
    lifespan=lifespan,
)


def get_processor(request: Request) -> DocumentProcessor:
    return cast(DocumentProcessor, request.app.state.processor)


class ProcessResponse(BaseModel):
    message: str
    document_id: str | None = None
    status: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "SiliconOCR"}


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_ocr: bool = True,
    enable_embeddings: bool = True,
    enable_classification: bool = False,
    enable_translation: bool = False,
    target_language: str = "en",
    processor: DocumentProcessor = Depends(get_processor),
):
    """Process a single document asynchronously."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to a temporary location
    temp_dir = Path(tempfile.gettempdir()) / "SiliconOCR-uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    def process_task(file_path: Path) -> None:
        try:
            processor.process_pdf(
                file_path,
                enable_ocr=enable_ocr,
                enable_embeddings=enable_embeddings,
                enable_classification=enable_classification,
                enable_translation=enable_translation,
                target_language=target_language,
            )
        except Exception as e:
            logging.getLogger("uvicorn.error").error(
                f"Background task failed for {file_path.name}: {e}", exc_info=True
            )
        finally:
            # Clean up the temporary file
            if file_path.exists():
                os.remove(file_path)

    # Add processing to background tasks
    background_tasks.add_task(process_task, temp_path)

    return ProcessResponse(
        message="Document processing started", document_id=file.filename, status="processing"
    )
