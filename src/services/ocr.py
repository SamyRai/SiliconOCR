"""OCR service with multiple engines optimized for Apple Silicon."""

from __future__ import annotations

import math
import threading
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from ..config import get_settings

# Optional dependencies
try:
    import pytesseract  # type: ignore
except ImportError:
    pytesseract = None  # type: ignore

try:
    import easyocr  # type: ignore
except ImportError:
    easyocr = None  # type: ignore

try:
    from doctr.io import DocumentFile  # type: ignore
    from doctr.models import ocr_predictor as doctr_predictor  # type: ignore
except ImportError:
    DocumentFile = None  # type: ignore
    doctr_predictor = None  # type: ignore


class OCRService:
    """Unified OCR service with multiple engine support.

    Supports:
    - EasyOCR (best Apple Silicon support, multilingual)
    - docTR (deep learning OCR)
    - Tesseract (classic OCR)

    Optimized for Mac with MPS support.
    """

    def __init__(self):
        self.settings = get_settings()
        self._easyocr_lock = threading.Lock()
        self._doctr_lock = threading.Lock()
        self._easyocr_reader: easyocr.Reader | None = None
        self._doctr_model = None
        self._device = self._get_device()

        logger.info(f"OCRService initialized with device: {self._device}")

    def _get_device(self) -> str:
        """Determine best device for Mac."""
        settings = self.settings

        # EasyOCR uses string device names differently
        if settings.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _maybe_downscale(self, img: Image.Image) -> Image.Image:
        """Downscale large images to improve OCR speed."""
        max_pixels = self.settings.max_image_pixels
        w, h = img.size

        if max_pixels and w * h > max_pixels:
            scale = math.sqrt(max_pixels / float(w * h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            # Use high-quality resampling
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return img

    def _get_easyocr_reader(self) -> easyocr.Reader:
        """Lazy load EasyOCR reader."""
        if self._easyocr_reader is not None:
            return self._easyocr_reader

        if easyocr is None:
            raise RuntimeError("EasyOCR is not installed")

        with self._easyocr_lock:
            if self._easyocr_reader is not None:
                return self._easyocr_reader

            logger.info(f"Loading EasyOCR for languages: {self.settings.ocr_languages}")

            # EasyOCR uses gpu=True/False
            use_gpu = self._device in ("mps", "cuda")

            reader = easyocr.Reader(
                self.settings.ocr_languages,
                gpu=use_gpu,
                model_storage_directory=str(self.settings.cache_dir / "easyocr"),
            )

            self._easyocr_reader = reader
            logger.info(f"EasyOCR initialized (GPU: {use_gpu})")
            return reader

    def _get_doctr_model(self):
        """Lazy load docTR model."""
        if self._doctr_model is not None:
            return self._doctr_model

        if doctr_predictor is None:
            raise RuntimeError("docTR is not installed")

        with self._doctr_lock:
            if self._doctr_model is not None:
                return self._doctr_model

            logger.info("Loading docTR OCR model")

            # Use lightweight models for better Mac performance
            model = doctr_predictor(
                det_arch="db_mobilenet_v3_large",
                reco_arch="crnn_mobilenet_v3_small",
                pretrained=True,
            )

            if self._device != "cpu":
                model.det_predictor.model.to(self._device)
                model.reco_predictor.model.to(self._device)

                if self.settings.use_fp16:
                    try:
                        model.det_predictor.model.half()
                        model.reco_predictor.model.half()
                    except Exception as e:
                        logger.debug(f"docTR fp16 not supported: {e}")

            self._doctr_model = model
            logger.info("docTR OCR initialized")
            return model

    def ocr_easyocr(self, image: Image.Image) -> str:
        """Run EasyOCR on PIL image."""
        reader = self._get_easyocr_reader()

        # Downscale if needed
        image = self._maybe_downscale(image)

        # EasyOCR expects numpy array
        import numpy as np

        img_array = np.array(image)

        # Run OCR thread-safely
        with self._easyocr_lock:
            results = reader.readtext(img_array, detail=0, paragraph=True)
        return " ".join(results)

    def ocr_doctr(self, image: Image.Image) -> str:
        """Run docTR on PIL image."""
        model = self._get_doctr_model()

        # Downscale if needed
        image = self._maybe_downscale(image)

        # docTR accepts PIL images
        doc = DocumentFile.from_images(image)

        # Run OCR thread-safely
        with self._doctr_lock:
            result = model(doc)

        # Extract text
        words = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        words.append(word.value)

        return " ".join(words)

    def ocr_tesseract(self, image: Image.Image) -> str:
        """Run Tesseract on PIL image."""
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed")

        # Downscale and convert to grayscale
        image = self._maybe_downscale(image)
        gray = image.convert("L")

        return str(pytesseract.image_to_string(gray))

    def ocr_image(
        self,
        image_path: str | Path,
        engine: str | None = None,
    ) -> str:
        """Run OCR on an image file.

        Args:
            image_path: Path to image file
            engine: OCR engine ('easyocr', 'doctr', 'tesseract')
                   If None, uses config default

        Returns:
            Extracted text
        """
        engine = (engine or self.settings.ocr_engine).lower()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Route to appropriate engine
        if engine == "easyocr":
            return self.ocr_easyocr(image)
        elif engine == "doctr":
            return self.ocr_doctr(image)
        elif engine == "tesseract":
            return self.ocr_tesseract(image)
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

    def ocr_image_from_pil(
        self,
        image: Image.Image,
        engine: str | None = None,
    ) -> str:
        """Run OCR on a PIL image.

        Args:
            image: PIL Image object
            engine: OCR engine to use

        Returns:
            Extracted text
        """
        engine = (engine or self.settings.ocr_engine).lower()

        if engine == "easyocr":
            return self.ocr_easyocr(image)
        elif engine == "doctr":
            return self.ocr_doctr(image)
        elif engine == "tesseract":
            return self.ocr_tesseract(image)
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

    def ocr_batch(
        self,
        image_paths: list[str | Path],
        engine: str | None = None,
    ) -> list[str]:
        """Run OCR on multiple images.

        Args:
            image_paths: List of image paths
            engine: OCR engine to use

        Returns:
            List of extracted texts
        """
        return [self.ocr_image(path, engine) for path in image_paths]
