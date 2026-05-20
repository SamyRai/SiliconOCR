import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    # Because we use lifespan which depends on heavy models, we can mock the processor
    # or just let it load if we want an integration test.
    # Given the heavy nature of models, let's mock the lifespan for unit tests if possible,
    # or just use TestClient which triggers lifespan. For this test, TestClient will trigger lifespan.
    # On a CI server this might be slow, but it verifies actual instantiation.
    with TestClient(app) as test_client:
        yield test_client


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
