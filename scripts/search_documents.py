"""Search processed documents by text or embeddings."""

import json
import sys

from loguru import logger

from src.config import get_settings
from src.services import EmbeddingService
from src.storage import ProcessedDocumentStore


def search_by_text(query: str, top_k: int = 5):
    """Search documents by semantic similarity."""
    logger.info(f"Searching for: {query}")

    # Load processed documents
    store = ProcessedDocumentStore(get_settings().cache_dir / "processed")
    json_files = list(store.iter_document_files())

    logger.info(f"Found {len(json_files)} processed documents")

    # Generate query embedding
    embedding_service = EmbeddingService()
    query_embedding = embedding_service.embed_text(query)

    # Compute similarities
    results = []
    for json_file in json_files:
        with open(json_file) as f:
            doc = json.load(f)

        if doc.get("text_embedding"):
            similarity = embedding_service.similarity(query_embedding, doc["text_embedding"])
            results.append((similarity, doc))

    # Sort by similarity
    results.sort(reverse=True, key=lambda x: x[0])

    # Display top results
    logger.info(f"\nTop {top_k} results:")
    logger.info("=" * 70)

    for i, (similarity, doc) in enumerate(results[:top_k], 1):
        logger.info(f"\n{i}. {doc['filename']} (similarity: {similarity:.3f})")
        logger.info(f"   Type: {doc['document_type']}")
        logger.info(f"   Pages: {doc['page_count']}")

        # Show snippet
        text = doc.get("text", "")[:200]
        if text:
            logger.info(f"   Preview: {text}...")


def search_by_keyword(keyword: str, case_sensitive: bool = False):
    """Search documents by keyword."""
    logger.info(f"Searching for keyword: {keyword}")

    store = ProcessedDocumentStore(get_settings().cache_dir / "processed")
    json_files = list(store.iter_document_files())

    results = []
    for json_file in json_files:
        with open(json_file) as f:
            doc = json.load(f)

        text = doc.get("text", "")
        if not case_sensitive:
            if keyword.lower() in text.lower():
                results.append(doc)
        else:
            if keyword in text:
                results.append(doc)

    logger.info(f"\nFound {len(results)} documents containing '{keyword}':")
    logger.info("=" * 70)

    for doc in results:
        logger.info(f"\n- {doc['filename']}")
        logger.info(f"  Type: {doc['document_type']}")
        logger.info(f"  Pages: {doc['page_count']}")


def main():
    """Search processed documents."""
    import argparse

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Search processed documents")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--semantic",
        "-s",
        action="store_true",
        help="Use semantic search (default: keyword)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to show (semantic search)",
    )
    parser.add_argument(
        "--case-sensitive",
        "-c",
        action="store_true",
        help="Case-sensitive keyword search",
    )

    args = parser.parse_args()

    if args.semantic:
        search_by_text(args.query, args.top_k)
    else:
        search_by_keyword(args.query, args.case_sensitive)


if __name__ == "__main__":
    main()
