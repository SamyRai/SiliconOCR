# SiliconOCR

[![CI](https://github.com/SamyRai/SiliconOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/SamyRai/SiliconOCR/actions/workflows/ci.yml)

An OCR and text embedding pipeline optimized for Apple Silicon (MPS). It provides a FastAPI backend for processing documents asynchronously.

## Features
- **OCR Engine Support**: EasyOCR, docTR, and Tesseract.
- **Embeddings**: SentenceTransformers for text and CLIP for images.
- **Translation**: Local execution of MarianMT and NLLB.
- **Hardware Acceleration**: Uses PyTorch MPS backend, FP16 precision, and `torch.compile`.
- **Concurrency**: Thread-safe ML inference and background task execution via FastAPI.

## Prerequisites
- Python >= 3.13
- [`uv`](https://github.com/astral-sh/uv)
- macOS (Apple Silicon recommended)

## Quick Start

1. **Install dependencies**:
   ```bash
   make install
   ```

2. **Run the server**:
   ```bash
   make dev
   ```
   API will be at `http://localhost:8000`.

3. **Process a document**:
   ```bash
   curl -X POST "http://localhost:8000/process" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your-document.pdf"
   ```

## Development
Use the `Makefile` for standard checks:

- `make check`: Runs formatting, linting, type-checking, and tests.
- `make format`: Formats code using Ruff.
- `make lint`: Runs Ruff and MyPy.
- `make test`: Runs Pytest.

## Architecture
- `api.py`: FastAPI routes and dependency injection.
- `processor.py`: Orchestrator for extraction, embeddings, and translation routing.
- `services/`: Singleton implementations with thread-locks for ML models (`ocr.py`, `embeddings.py`, `translation.py`).

## License
MIT License. See [LICENSE](LICENSE) for details.
