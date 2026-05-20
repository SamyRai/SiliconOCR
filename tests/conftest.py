from pathlib import Path

import pytest


@pytest.fixture
def mock_pdf_path(tmp_path: Path) -> Path:
    """Fixture providing a mock PDF file path."""
    pdf_path = tmp_path / "dummy_test_file.pdf"
    pdf_path.write_text("dummy content")
    return pdf_path
