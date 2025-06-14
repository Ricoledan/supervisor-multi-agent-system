#!/usr/bin/env python3
import sys
import os

sys.path.append('.')

# Import and run the ingestion
from src.utils.ingestion_pipeline import ingest_pdfs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sources")
    args = parser.parse_args()

    print(f"🔄 Processing PDFs from {args.source}")
    ingest_pdfs(args.source)