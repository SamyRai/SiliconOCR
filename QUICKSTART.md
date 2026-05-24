# SiliconOCR Quickstart

OCR, embeddings, and translation pipeline optimized for Apple Silicon (MPS).

## Features

- **OCR**: Multi-engine support (EasyOCR, docTR, Tesseract)
- **Embeddings**: Text embeddings using sentence-transformers
- **Translation**: Multi-language support (NLLB-200, MarianMT)
- **Apple Silicon Optimized**: MPS support, FP16 precision, torch.compile
- **Batch Processing**: Efficient processing of large document collections

## Quick Start

```bash
# Run setup script
./setup.sh

# Process first 5 documents (test run)
python main.py --limit 5 --verbose

# Process all inbox documents
python scripts/batch_process.py
```

## Setup

### Prerequisites

- Python 3.13+
- uv (Python package manager)
- Tesseract (optional, for Tesseract OCR engine)

```bash
# Install Tesseract on Mac
brew install tesseract tesseract-lang
```

### Installation

```bash
# Navigate to the project directory
cd SiliconOCR

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Usage

### Process Inbox Documents

Process all PDFs in the inbox directory:

```bash
# Process all documents with default OCR + embeddings
uv run python main.py

# Process with specific inbox path
uv run python main.py --inbox /path/to/inbox

# Limit number of files
uv run python main.py --limit 10

# Verbose logging
uv run python main.py --verbose
```

### Batch Processing

Process large collections with detailed progress:

```bash
python scripts/batch_process.py
```

### Search Documents

Search processed documents by keyword or semantic similarity:

```bash
# Keyword search
python scripts/search_documents.py "invoice"

# Semantic search (uses embeddings)
python scripts/search_documents.py "housing utilities payment" --semantic
```

### Export Results

Export processed documents to CSV or text:

```bash
# Export to CSV
python scripts/export_results.py --format csv --output results.csv

# Export full text
python scripts/export_results.py --format text --output documents.txt
```

## Output

Processed documents are saved to `~/.cache/SiliconOCR/processed/`:

- Individual document results (JSON)
- Processing summary
- Text embeddings (1024-dimensional vectors)
- Document classification

The processing pipeline is configured through `ProcessingOptions` and coordinated by
`DocumentProcessor`. Classification rules live in `DocumentClassifier`, and result JSON
persistence lives in `ProcessedDocumentStore`.

## Models

- **OCR**: EasyOCR (best Mac support), docTR, Tesseract
- **Embeddings**: nomic-embed-text-v2-moe
- **Translation**: NLLB-200 (200+ languages)

## Performance

On Apple Silicon (M1/M2/M3):

- Native PDF extraction: ~0.1s per page
- OCR extraction: ~2-5s per page
- Embeddings: ~0.01s per document

See [README.md](README.md) for full documentation.
