import os
import logging
from dotenv import load_dotenv
import chromadb

load_dotenv()

logger = logging.getLogger("chroma")
logging.basicConfig(level=logging.INFO)


class ChromaDBConfig:
    def __init__(self):
        self.host = os.environ.get('CHROMA_HOST')
        self.port = os.environ.get('CHROMA_PORT')
        logger.info(f"ChromaDBConfig initialized with host={self.host}, port={self.port}")

    def get_client(self):
        try:
            # Explicitly convert port to integer
            port_int = int(self.port)

            client = chromadb.HttpClient(
                host=self.host,
                port=port_int,
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