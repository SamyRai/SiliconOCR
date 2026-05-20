"""Translation service using modern transformer models."""

from __future__ import annotations

import threading

import torch
from loguru import logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
)

from ..config import get_settings


class TranslationService:
    """Translation service optimized for Apple Silicon.

    Supports:
    - NLLB-200 (200 languages, Meta AI)
    - MarianMT (fast, specialized language pairs)

    Optimized for Mac with MPS support.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._nllb_lock = threading.Lock()
        self._marian_locks: dict[str, threading.Lock] = {}
        self._nllb_model: AutoModelForSeq2SeqLM | None = None
        self._nllb_tokenizer: AutoTokenizer | None = None
        self._marian_models: dict[str, tuple[MarianMTModel, MarianTokenizer]] = {}
        self._device = self._get_device()

        logger.info(f"TranslationService initialized with device: {self._device}")

    def _get_device(self) -> str:
        """Determine best device for Mac."""
        settings = self.settings

        if settings.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("MPS/CUDA not available, using CPU")
            return "cpu"

    def _get_nllb_model(self) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """Lazy load NLLB model."""
        if self._nllb_model is not None and self._nllb_tokenizer is not None:
            return self._nllb_model, self._nllb_tokenizer

        with self._nllb_lock:
            if self._nllb_model is not None and self._nllb_tokenizer is not None:
                return self._nllb_model, self._nllb_tokenizer

            logger.info(f"Loading translation model: {self.settings.translation_model}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.translation_model,
                cache_dir=str(self.settings.model_cache_dir),
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.settings.translation_model,
                cache_dir=str(self.settings.model_cache_dir),
            )

            if self._device != "cpu":
                model = model.to(self._device)

                if self.settings.use_fp16:
                    try:
                        model = model.half()
                        logger.info("Translation model using FP16")
                    except Exception as e:
                        logger.debug(f"FP16 conversion failed: {e}")

                # Apply torch.compile
                if self.settings.enable_torch_compile and hasattr(torch, "compile"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        logger.info("Translation model compiled")
                    except Exception as e:
                        logger.debug(f"torch.compile failed: {e}")

            self._nllb_model = model
            self._nllb_tokenizer = tokenizer
            return model, tokenizer

    def _get_marian_model(
        self, src_lang: str, tgt_lang: str
    ) -> tuple[MarianMTModel, MarianTokenizer]:
        """Lazy load MarianMT model for specific language pair."""
        pair_key = f"{src_lang}-{tgt_lang}"

        if pair_key in self._marian_models:
            return self._marian_models[pair_key]

        if pair_key not in self._marian_locks:
            self._marian_locks[pair_key] = threading.Lock()

        with self._marian_locks[pair_key]:
            if pair_key in self._marian_models:
                return self._marian_models[pair_key]

            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
            logger.info(f"Loading MarianMT model: {model_name}")

            try:
                tokenizer = MarianTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.model_cache_dir),
                )
                model = MarianMTModel.from_pretrained(
                    model_name,
                    cache_dir=str(self.settings.model_cache_dir),
                )

                if self._device != "cpu":
                    model = model.to(self._device)

                self._marian_models[pair_key] = (model, tokenizer)
                return model, tokenizer
            except Exception as e:
                logger.error(f"Failed to load MarianMT model {model_name}: {e}")
                raise

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        use_marian: bool = False,
    ) -> str:
        """Translate text from source to target language.

        Args:
            text: Text to translate
            src_lang: Source language code (e.g., 'eng_Latn', 'en')
            tgt_lang: Target language code (e.g., 'spa_Latn', 'es')
            use_marian: Use MarianMT instead of NLLB (faster for common pairs)

        Returns:
            Translated text
        """
        if use_marian:
            return self._translate_marian(text, src_lang, tgt_lang)
        else:
            return self._translate_nllb(text, src_lang, tgt_lang)

    def _translate_nllb(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using NLLB model."""
        model, tokenizer = self._get_nllb_model()

        # NLLB uses language codes like 'eng_Latn', 'spa_Latn'
        tokenizer.src_lang = src_lang

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad(), self._nllb_lock:
            translated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=512,
            )

        result = str(tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
        return result

    def _translate_marian(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using MarianMT model."""
        model, tokenizer = self._get_marian_model(src_lang, tgt_lang)

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        pair_key = f"{src_lang}-{tgt_lang}"
        with torch.no_grad(), self._marian_locks[pair_key]:
            translated = model.generate(**inputs, max_length=512)

        result = str(tokenizer.batch_decode(translated, skip_special_tokens=True)[0])
        return result

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        use_marian: bool = False,
    ) -> list[str]:
        """Translate multiple texts (batch processing).

        Args:
            texts: List of texts to translate
            src_lang: Source language code
            tgt_lang: Target language code
            use_marian: Use MarianMT instead of NLLB

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        if use_marian:
            model, tokenizer = self._get_marian_model(src_lang, tgt_lang)
        else:
            model, tokenizer = self._get_nllb_model()
            tokenizer.src_lang = src_lang

        batch_size = self.settings.translation_batch_size
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            lock = self._marian_locks[f"{src_lang}-{tgt_lang}"] if use_marian else self._nllb_lock
            with torch.no_grad(), lock:
                if use_marian:
                    translated = model.generate(**inputs, max_length=512)
                else:
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        max_length=512,
                    )

            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            results.extend(decoded)

        return results

    def translate_german_to_english(self, text: str) -> str:
        """Convenience method to translate German to English.

        Uses optimized Helsinki-NLP opus-mt-de-en model.

        Args:
            text: German text to translate

        Returns:
            English translation
        """
        return self.translate(text, src_lang="de", tgt_lang="en", use_marian=True)

    def translate_german_to_english_batch(self, texts: list[str]) -> list[str]:
        """Convenience method to translate multiple German texts to English.

        Args:
            texts: List of German texts to translate

        Returns:
            List of English translations
        """
        return self.translate_batch(texts, src_lang="de", tgt_lang="en", use_marian=True)

    @staticmethod
    def get_supported_languages() -> dict[str, list[str]]:
        """Get supported languages for different models.

        Returns:
            Dictionary with model names and their supported languages
        """
        return {
            "nllb": [
                "eng_Latn",
                "spa_Latn",
                "fra_Latn",
                "deu_Latn",
                "ita_Latn",
                "por_Latn",
                "rus_Cyrl",
                "zho_Hans",
                "jpn_Jpan",
                "kor_Hang",
                "ara_Arab",
                "hin_Deva",
                "ben_Beng",
                "tur_Latn",
                "vie_Latn",
                # ... 200+ languages supported
            ],
            "marian": [
                # Common language pairs available
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "ru",
                "zh",
                "ja",
                "ko",
            ],
        }
