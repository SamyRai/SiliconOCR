#!/bin/bash
# Quick start script for SiliconOCR

set -e

# Print header
echo "=========================================="
echo "SiliconOCR - Setup"
echo "==================================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the SiliconOCR directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Step 1: Installing dependencies..."
uv sync

echo ""
echo "Step 2: Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Step 3: Checking PyTorch and MPS support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

echo ""
echo "==================================================================="
echo "Setup complete!"
echo "==================================================================="
echo ""
echo "Quick commands:"
echo ""
echo "  # Process inbox documents"
echo "  python main.py"
echo ""
echo "  # Process first 5 documents (test run)"
echo "  python main.py --limit 5 --verbose"
echo ""
echo "  # Batch process all documents"
echo "  python scripts/batch_process.py"
echo ""
echo "  # Search documents"
echo "  python scripts/search_documents.py 'invoice' --semantic"
echo ""
echo "==================================================================="
echo ""
echo "Ready to process documents!"
echo ""
