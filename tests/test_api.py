import pytest
from fastapi.testclient import TestClient

from src.api import app, get_processor


@pytest.fixture
def client():
    class FakeProcessor:
        def process_pdf(self, file_path, options=None):
            self.file_path = file_path
            self.options = options

    app.dependency_overrides[get_processor] = lambda: FakeProcessor()
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "SiliconOCR"}


def test_process_missing_file(client):
    response = client.post("/process")
    assert response.status_code == 422  # Unprocessable Entity (Missing file)


def test_process_invalid_file_extension(client):
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/process", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_process_pdf_starts_background_task(client):
    files = {"file": ("test.pdf", b"%PDF-1.4", "application/pdf")}
    response = client.post("/process", files=files)
    assert response.status_code == 200
    assert response.json() == {
        "message": "Document processing started",
        "document_id": "test.pdf",
        "status": "processing",
    }
