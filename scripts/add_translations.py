"""Add language detection and English translation to JSON files."""

import json
import os
import warnings
from pathlib import Path

from langdetect import DetectorFactory, detect

from src.services.translation import TranslationService
from src.translation_utils import translate_german_text_to_english

# Make language detection deterministic
DetectorFactory.seed = 0


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

                translation = translate_german_text_to_english(
                    text,
                    translation_service,
                    max_chars=1500,
                    strategy="sentence",
                    joiner=" ",
                )
                if translation.chunk_count > 1:
                    print(f" ({translation.chunk_count} chunks)", end="")

                data["translation_en"] = translation.text
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
