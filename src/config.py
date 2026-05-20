"""Configuration for OCR service optimized for Apple Silicon (MPS)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings optimized for Mac."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Device configuration - Apple Silicon optimized
    device: str = Field(
        default="mps",  # Metal Performance Shaders for Apple Silicon
        description="Device to use: 'mps' (Apple Silicon), 'cuda', or 'cpu'",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for faster inference on Apple Silicon",
    )

    # Embedding models
    text_embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v2-moe",  # v2, multilingual
        description="Text embedding model (optimized for Mac)",
    )
    multimodal_model: str = Field(
        default="openai/clip-vit-base-patch32",
        description="CLIP model for image/text embeddings",
    )

    # OCR configuration
    ocr_engine: str = Field(
        default="easyocr",  # Better Apple Silicon support than doctr
        description="OCR engine: 'easyocr', 'doctr', or 'tesseract'",
    )
    ocr_languages: list[str] = Field(
        default=["en", "de"],
        description="Languages for OCR (English and German)",
    )
    max_image_pixels: int = Field(
        default=178956970,  # ~13000x13000
        description="Max pixels for image processing",
    )
    pdf_dpi: int = Field(
        default=200,
        description="DPI for PDF rendering",
    )

    # Translation configuration
    translation_model: str = Field(
        default="Helsinki-NLP/opus-mt-de-en",  # Optimized DE->EN translation
        description="Translation model",
    )
    translation_batch_size: int = Field(
        default=32,
        description="Batch size for translation",
    )

    # Performance tuning
    batch_size: int = Field(
        default=64,  # Apple Silicon can handle larger batches
        description="Default batch size for embeddings",
    )
    max_workers: int = Field(
        default=4,
        description="Max parallel workers",
    )
    enable_torch_compile: bool = Field(
        default=True,
        description="Enable torch.compile for 2-3x speedup (PyTorch 2.0+)",
    )

    # Storage
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "SiliconOCR",
        description="Cache directory for models",
    )
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "huggingface",
        description="HuggingFace model cache",
    )

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables for HuggingFace
        os.environ["HF_HOME"] = str(self.model_cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.model_cache_dir)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
