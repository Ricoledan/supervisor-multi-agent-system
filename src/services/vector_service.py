import logging
from typing import List, Dict, Any
from src.databases.vector.config import ChromaDBConfig

logger = logging.getLogger("vector_service")


def query_vectordb(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Query ChromaDB for semantically similar documents."""
    logger.info(f"Querying vector DB for: {query[:30]}...")
    try:
        chroma_config = ChromaDBConfig()
        client = chroma_config.get_client()

        try:
            # Use consistent collection name
            collection = client.get_collection(name="academic_papers")
        except Exception:
            logger.info("Collection 'academic_papers' not found. Creating new collection.")
            collection = client.create_collection(name="academic_papers")

        results = collection.query(
            query_texts=[query],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(documents)
            distances = results["distances"][0] if "distances" in results else [1.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                formatted_results.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": 1.0 - float(dist)  # Convert distance to similarity score
                })

            logger.info(f"Retrieved {len(formatted_results)} documents from vector DB")

        return formatted_results
    except Exception as e:
        logger.error(f"Error querying vector DB: {str(e)}", exc_info=True)
        return []