"""Embedding service optimized for Apple Silicon."""

from __future__ import annotations

import threading
from contextlib import suppress

import numpy as np
import torch
from loguru import logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from ..config import get_settings
from .device import resolve_torch_device


class EmbeddingService:
    """Unified embedding service for text and images.

    Optimized for Apple Silicon with MPS (Metal Performance Shaders) support.
    Features:
    - Text embeddings using nomic-embed
    - Image/text embeddings using CLIP
    - Batch processing for efficiency
    - FP16 support for faster inference
    - torch.compile for 2-3x speedup
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._sbert_lock = threading.Lock()
        self._clip_lock = threading.Lock()
        self._sbert_model: SentenceTransformer | None = None
        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None
        self._device = self._get_device()

        logger.info(f"EmbeddingService initialized with device: {self._device}")

    def _get_device(self) -> str:
        """Determine best device for Mac."""
        return resolve_torch_device(self.settings)

    def _get_sbert_model(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model."""
        if self._sbert_model is not None:
            return self._sbert_model

        with self._sbert_lock:
            if self._sbert_model is not None:
                return self._sbert_model

            logger.info(f"Loading text embedding model: {self.settings.text_embedding_model}")
            model = SentenceTransformer(
                self.settings.text_embedding_model,
                device=self._device,
                trust_remote_code=True,
            )

            # Optimize for Apple Silicon
            if self._device == "mps" and self.settings.use_fp16:
                try:
                    model = model.half()
                    logger.info("Enabled FP16 for text embeddings")
                except Exception as e:
                    logger.debug(f"FP16 conversion failed: {e}")

            # Apply torch.compile for speedup (PyTorch 2.0+)
            if self.settings.enable_torch_compile and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model, mode="reduce-overhead")  # type: ignore
                    logger.info("Text embedding model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"torch.compile failed: {e}")

            self._sbert_model = model
            return model

    def _get_clip_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        """Lazy load CLIP model."""
        if self._clip_model is not None and self._clip_processor is not None:
            return self._clip_model, self._clip_processor

        with self._clip_lock:
            if self._clip_model is not None and self._clip_processor is not None:
                return self._clip_model, self._clip_processor

            logger.info(f"Loading CLIP model: {self.settings.multimodal_model}")
            model = CLIPModel.from_pretrained(self.settings.multimodal_model)
            processor = CLIPProcessor.from_pretrained(self.settings.multimodal_model)

            if self._device != "cpu":
                model = model.to(self._device)

                # Apply torch.compile
                if self.settings.enable_torch_compile and hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        logger.info("CLIP model compiled with torch.compile")
                    except Exception as e:
                        logger.debug(f"torch.compile for CLIP failed: {e}")

            self._clip_model = model
            self._clip_processor = processor
            return model, processor

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed_text_batch([text])[0]

    def embed_text_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts to embed
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._get_sbert_model()
        batch_size = self.settings.batch_size

        # Use inference_mode for efficiency, and lock for thread safety
        with torch.inference_mode(), self._sbert_lock:
            if self._device == "mps" and self.settings.use_fp16:
                # MPS-optimized path
                try:
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize,
                        convert_to_tensor=True,
                        device=self._device,
                        show_progress_bar=show_progress,
                    )
                except Exception as e:
                    logger.warning(f"MPS encoding failed, falling back to CPU: {e}")
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=normalize,
                        convert_to_tensor=True,
                        device="cpu",
                        show_progress_bar=show_progress,
                    )
            else:
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    convert_to_tensor=True,
                    device=self._device,
                    show_progress_bar=show_progress,
                )

        # Convert to list
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu()
        if hasattr(embeddings, "numpy"):
            arr = embeddings.numpy()
            with suppress(Exception):
                arr = arr.astype("float32")
            return arr.tolist()  # type: ignore

        return [[float(x) for x in row] for row in embeddings]

    def embed_image(self, image_path: str) -> list[float]:
        """Generate embedding for an image using CLIP.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector
        """
        model, processor = self._get_clip_model()

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad(), self._clip_lock:
            features = model.get_image_features(**inputs)

        vector = features.cpu().numpy().flatten().tolist()
        return vector  # type: ignore

    def embed_image_batch(self, image_paths: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of embedding vectors
        """
        model, processor = self._get_clip_model()

        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad(), self._clip_lock:
            features = model.get_image_features(**inputs)

        vectors = features.cpu().numpy()
        return vectors.tolist()  # type: ignore

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (-1 to 1)
        """
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
