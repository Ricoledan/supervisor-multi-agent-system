#!/usr/bin/env python
import os
import uuid
import logging
from pathlib import Path
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.databases.vector.config import ChromaDBConfig

SOURCE_DIR = os.getenv("SOURCE_DIR", "sources")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "academic_papers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc]), {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "source_file": str(path),
        "file_type": "pdf"
    }

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read(), {
            "source_file": str(path),
            "file_type": path.suffix.lstrip('.')
        }

def main():
    logger.info(f"Initializing ingestion from {SOURCE_DIR}")
    source_path = Path(SOURCE_DIR)
    if not source_path.exists():
        logger.error(f"Source dir {SOURCE_DIR} not found.")
        return 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = ChromaDBConfig().get_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for file_path in source_path.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                text, metadata = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text, metadata = extract_text_from_txt(file_path)
            else:
                logger.info(f"Skipping unsupported file: {file_path}")
                continue

            chunks = splitter.split_text(text)
            if not chunks:
                continue

            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{**metadata, "chunk_id": i, "chunk_total": len(chunks)} for i in range(len(chunks))]

            collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            logger.info(f"✅ Added {len(chunks)} chunks from {file_path.name}")

        except Exception as e:
            logger.warning(f"❌ Error processing {file_path.name}: {e}")

    logger.info("✅ Ingestion complete.")
    return 0

if __name__ == "__main__":
    exit(main())