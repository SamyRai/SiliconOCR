"""Upload file handling for the API layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile


def validate_pdf_filename(filename: str | None) -> str:
    """Validate an uploaded PDF filename."""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    return Path(filename).name


async def save_upload_to_temp(file: UploadFile) -> Path:
    """Save an uploaded PDF to the API temp directory."""
    filename = validate_pdf_filename(file.filename)
    temp_dir = Path(tempfile.gettempdir()) / "SiliconOCR-uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename
    temp_path.write_bytes(await file.read())
    return temp_path


def cleanup_temp_file(file_path: Path) -> None:
    """Remove a temporary upload file if it still exists."""
    file_path.unlink(missing_ok=True)
