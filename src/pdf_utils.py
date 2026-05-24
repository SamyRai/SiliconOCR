"""PDF processing utilities."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None  # type: ignore

try:
    import pymupdf  # type: ignore
except ImportError:
    pymupdf = None  # type: ignore


class PDFProcessor:
    """Process PDFs with multiple extraction strategies."""

    def __init__(self, dpi: int = 200, max_pixels: int = 178956970) -> None:
        self.dpi = dpi
        self.max_pixels = max_pixels

    def count_pages(self, pdf_path: str | Path) -> int:
        """Count pages in PDF."""
        pdf_path = Path(pdf_path)

        if pymupdf:
            try:
                doc = pymupdf.open(pdf_path)
                count = len(doc)
                doc.close()
                return count
            except Exception as e:
                logger.debug(f"PyMuPDF page count failed: {e}")

        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    return len(pdf.pages)
            except Exception as e:
                logger.debug(f"pdfplumber page count failed: {e}")

        return 0

    def extract_text_native(self, pdf_path: str | Path) -> tuple[str, bool]:
        """Extract text natively from PDF.

        Returns:
            (text, success) tuple
        """
        pdf_path = Path(pdf_path)

        # Try PyMuPDF first (fastest)
        if pymupdf:
            try:
                doc = pymupdf.open(pdf_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                text = "\n\n".join(text_parts)
                if text.strip():
                    logger.debug(f"Extracted text with PyMuPDF: {len(text)} chars")
                    return text, True
            except Exception as e:
                logger.debug(f"PyMuPDF extraction failed: {e}")

        # Try pdfplumber
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_parts.append(page_text)
                    text = "\n\n".join(text_parts)
                    if text.strip():
                        logger.debug(f"Extracted text with pdfplumber: {len(text)} chars")
                        return text, True
            except Exception as e:
                logger.debug(f"pdfplumber extraction failed: {e}")

        return "", False

    def convert_to_images(self, pdf_path: str | Path) -> list[Image.Image]:
        """Convert PDF pages to images for OCR.

        Returns:
            List of PIL Images
        """
        if convert_from_path is None:
            raise RuntimeError("pdf2image not installed")

        pdf_path = Path(pdf_path)

        # Set PIL limit
        Image.MAX_IMAGE_PIXELS = self.max_pixels

        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt="jpeg",
            )
            logger.debug(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"PDF to images conversion failed: {e}")
            raise

    def write_text_layer(self, pdf_path: str | Path, page_texts: list[str]) -> None:
        """Write OCR text to each PDF page as an invisible text layer."""
        if pymupdf is None:
            raise RuntimeError("PyMuPDF is required to write a PDF text layer")

        pdf_path = Path(pdf_path)
        doc = pymupdf.open(pdf_path)
        try:
            for page_index, page_text in enumerate(page_texts):
                if page_index >= len(doc) or not page_text.strip():
                    continue

                page = doc[page_index]
                rect = page.rect
                page.insert_textbox(
                    rect,
                    page_text,
                    fontsize=8,
                    render_mode=3,
                    overlay=True,
                )

            doc.saveIncr()
        finally:
            doc.close()
