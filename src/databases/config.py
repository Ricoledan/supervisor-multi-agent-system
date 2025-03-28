import os
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv()

class ChromaDBConfig:
    def __init__(self):
        self.host = os.getenv("CHROMADB_HOST")
        self.port = int(os.getenv("CHROMADB_PORT"))
        self.settings = Settings(allow_reset=True, anonymized_telemetry=False)

    def get_client(self):
        return chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=self.settings
        )

chroma_db_config = ChromaDBConfig()
client = chroma_db_config.get_client()

# allow_reset=True: This setting permits the ChromaDB client to reset the database index, effectively deleting all data.
# By default, this option is disabled (False) to prevent accidental data loss. Enabling it (True) allows operations that
# can clear the database.
