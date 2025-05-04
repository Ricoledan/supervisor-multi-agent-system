
import os
import logging
from dotenv import load_dotenv
import chromadb

load_dotenv()

logger = logging.getLogger("chroma")
logging.basicConfig(level=logging.INFO)

class ChromaDBConfig:
    def __init__(self):
        self.host = os.getenv("CHROMA_SERVER_HOST")
        self.port = int(os.getenv("CHROMA_SERVER_PORT"))

        logger.info(f"ChromaDBConfig initialized with host={self.host}, port={self.port}")

    def get_client(self) -> chromadb.HttpClient:
        try:
            client = chromadb.HttpClient(host=self.host, port=self.port)
            logger.info("ChromaDB client created successfully.")
            return client
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise