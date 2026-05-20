"""Merge PDF groups and their JSON metadata into single files.

Usage (run from repo root with `uv run python`):
  cd SiliconOCR && uv run python scripts/merge_groups.py

This script merges two groups currently in the inbox:
 - HOWOGE heating year summary parts (prefix: 2025-12-15-howoge-heating-year-summary)
 - ARD/ZDF "inbox" rundfunk documents (prefix: 2026-01-09-inbox 1/2/3)

It writes *_full.pdf and *_full.json files to the inbox directory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

try:
    import fitz  # pymupdf
except Exception as e:
    raise SystemExit(
        "PyMuPDF (pymupdf) is required. Install it in the environment: pip install pymupdf"
    ) from e

INBOX = Path(__file__).resolve().parents[1] / "inbox"


def find_by_prefixes(prefixes: Iterable[str], ext: str = ".pdf") -> list[Path]:
    prefixes = [p.lower() for p in prefixes]
    result = []
    for p in sorted(INBOX.iterdir()):
        if p.suffix.lower() != ext.lower():
            continue
        name = p.name.lower()
        for pref in prefixes:
            if name.startswith(pref):
                result.append(p)
                break
    return result


def merge_pdfs(paths: list[Path], out: Path) -> int:
    if not paths:
        return 0
    out_doc = fitz.open()
    for src in paths:
        src_doc = fitz.open(src)
        out_doc.insert_pdf(src_doc)
        src_doc.close()
    out_doc.save(out)
    out_doc.close()
    return len(paths)


def merge_jsons(json_paths: list[Path], out: Path, out_pdf_name: str):
    merged = {
        "filename": out_pdf_name,
        "filepath": str(out.resolve()),
        "file_size": out.stat().st_size if out.exists() else None,
        "processed_at": None,
        "status": "completed",
        "text": "",
        "page_count": 0,
        "metadata": {"merged_from": [p.name for p in json_paths]},
    }

    texts = []
    page_total = 0
    for jp in sorted(json_paths):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("text"):
            texts.append(data.get("text"))
        page_total += data.get("page_count", 0) or 0
        if data.get("metadata"):
            merged.setdefault("sources_metadata", []).append({jp.name: data.get("metadata")})

    merged["text"] = "\n\n---- MERGED DOCUMENT ----\n\n".join(texts)
    merged["page_count"] = page_total or None
    merged["processed_at"] = __import__("datetime").datetime.utcnow().isoformat()

    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    groups = [
        {
            "prefixes": ["2025-12-15-howoge-heating-year-summary"],
            "out_pdf": "2025-12-15-howoge-heating-year-summary-full.pdf",
            "out_json": "2025-12-15-howoge-heating-year-summary-full.json",
        },
        {
            "prefixes": [
                "2026-01-09-inbox 1",
                "2026-01-09-inbox 2",
                "2026-01-09-inbox 3",
            ],
            "out_pdf": "2025-07-02-ard-zdf-rundfunkbeitrag-full.pdf",
            "out_json": "2025-07-02-ard-zdf-rundfunkbeitrag-full.json",
        },
    ]

    for g in groups:
        pdfs = find_by_prefixes(g["prefixes"], ext=".pdf")
        jsons = find_by_prefixes(g["prefixes"], ext=".json")
        print(f"Merging {len(pdfs)} PDFs and {len(jsons)} JSONs for prefixes: {g['prefixes']}")
        if not pdfs:
            print("  -> no PDFs found, skipping")
            continue
        out_pdf = INBOX / g["out_pdf"]
        out_json = INBOX / g["out_json"]
        merged_count = merge_pdfs(pdfs, out_pdf)
        merge_jsons(jsons, out_json, g["out_pdf"])
        print(f"  -> wrote {out_pdf.name} ({merged_count} PDFs) and {out_json.name}")

    print("Done.")
