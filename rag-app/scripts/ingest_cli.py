#!/usr/bin/env python3
"""
CLI entry point for ingestion, as an alternative to the /ingest HTTP endpoint.

Usage:
    python scripts/ingest_cli.py data/corpus
    python scripts/ingest_cli.py data/corpus/oil_refinery.md --chunk-size 800 --chunk-overlap 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest_directory, ingest_file


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    if path.is_dir():
        results = ingest_directory(path, args.chunk_size, args.chunk_overlap)
    else:
        results = [ingest_file(path, args.chunk_size, args.chunk_overlap)]

    for r in results:
        print(
            f"{r.source}: {r.total_chunks} chunks total "
            f"({r.new_or_changed_chunks} new/changed, {r.skipped_unchanged_chunks} skipped, "
            f"{r.deleted_stale_chunks} stale removed) in {r.elapsed_seconds}s"
        )


if __name__ == "__main__":
    main()
