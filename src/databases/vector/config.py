import os
import logging
from dotenv import load_dotenv
import chromadb

load_dotenv()

logger = logging.getLogger("chroma")
logging.basicConfig(level=logging.INFO)


class ChromaDBConfig:
    def __init__(self):
        self.host = os.environ.get('CHROMA_SERVER_HOST')
        self.port = os.environ.get('CHROMA_SERVER_PORT')
        logger.info(f"ChromaDBConfig initialized with host={self.host}, port={self.port}")

    def get_client(self):
        try:
            client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=chromadb.Settings(
                    anonymized_telemetry=True
                )
            )
            client.heartbeat()
            logger.info("ChromaDB client created successfully.")
            return client
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {e}")
            raise