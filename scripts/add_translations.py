"""Add language detection and English translation to JSON files."""

import json
import os
import warnings
from pathlib import Path

from langdetect import DetectorFactory, detect

from src.services.translation import TranslationService

# Make language detection deterministic
DetectorFactory.seed = 0


def chunk_text(text, max_chars=1500):
    """Split text into chunks, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by sentences (simple approach)
    sentences = text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def main():
    default_inbox = Path(__file__).resolve().parents[2] / "inbox"

    def _get_inbox_path():
        for var in ("UNIFIED_STORAGE_INBOX", "STORAGE_INBOX", "HETZNER_STORAGE_INBOX"):
            val = os.environ.get(var)
            if val:
                if var == "HETZNER_STORAGE_INBOX":
                    warnings.warn(
                        "HETZNER_STORAGE_INBOX is deprecated. Use UNIFIED_STORAGE_INBOX or STORAGE_INBOX.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                return Path(val)
        return default_inbox

    inbox = _get_inbox_path()
    json_files = sorted(inbox.glob("*.json"))

    print(f"📋 Found {len(json_files)} JSON files")
    print("🔧 Initializing translation service...\n")

    translation_service = TranslationService()

    processed = 0
    updated = 0
    skipped = 0
    errors = []

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "").strip()
            if not text:
                skipped += 1
                continue

            # Detect language
            detected_lang = detect(text)
            data["source_language"] = detected_lang

            # Translate to English if not already English
            if detected_lang == "de":
                print(f"🇩🇪→🇬🇧 {json_file.name}", end="")

                # Chunk long texts
                chunks = chunk_text(text)
                if len(chunks) > 1:
                    print(f" ({len(chunks)} chunks)", end="")

                translations = []
                for chunk in chunks:
                    translation = translation_service.translate_german_to_english(chunk)
                    translations.append(translation)

                data["translation_en"] = " ".join(translations)
                print(" ✓")
                updated += 1
            elif detected_lang != "en":
                print(f"🌍 {json_file.name}: {detected_lang} (keeping original)")
                data["translation_en"] = text  # Keep original for non-German/English
            else:
                print(f"🇬🇧 {json_file.name}: already English")
                data["translation_en"] = text

            # Save updated JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            processed += 1

        except Exception as e:
            error_msg = f"{json_file.name}: {e!s}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")

    print(f"\n✅ Processed: {processed}/{len(json_files)}")
    print(f"🔄 Translated: {updated}")
    print(f"⏭️  Skipped: {skipped}")
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"   - {err}")


if __name__ == "__main__":
    main()
