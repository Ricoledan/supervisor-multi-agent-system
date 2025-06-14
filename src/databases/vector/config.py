import os
import logging
from dotenv import load_dotenv
import chromadb

load_dotenv()

logger = logging.getLogger("chroma")


class ChromaDBConfig:
    def __init__(self):
        self.host = os.environ.get('CHROMA_HOST', 'localhost')
        self.port = os.environ.get('CHROMA_PORT', '8001')  # Note: using 8001 for external access
        logger.info(f"ChromaDBConfig initialized with host={self.host}, port={self.port}")

    def get_client(self):
        try:
            port_int = int(self.port)

            client = chromadb.HttpClient(
                host=self.host,
                port=port_int,
                settings=chromadb.Settings(
                    anonymized_telemetry=False  # Disable telemetry for cleaner logs
                )
            )
            client.heartbeat()
            logger.info("ChromaDB client created successfully.")
            return client
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {e}")
            raise

    def get_collection(self, collection_name="academic_papers"):
        """Get or create a collection with proper v0.6.0 syntax"""
        client = self.get_client()
        try:
            # Try to get existing collection
            collection = client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
            return collection
        except Exception:
            # Collection doesn't exist, create it
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
            return collection

    def list_collections(self):
        """List collections with v0.6.0 compatibility"""
        client = self.get_client()
        try:
            # In v0.6.0, list_collections() returns collection names only
            collection_names = client.list_collections()
            return collection_names
        except Exception as e:
            logger.warning(f"Could not list collections: {e}")
            return []
