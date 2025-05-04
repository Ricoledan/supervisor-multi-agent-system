import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBConfig:
    def __init__(self):
        self.host = os.getenv("MONGODB_HOST")
        port_str = os.getenv("MONGODB_PORT")
        self.port = int(port_str)
        self.user = os.getenv("MONGODB_USER")
        self.password = os.getenv("MONGODB_PASSWORD")
        self.database = os.getenv("MONGODB_DB")
        self._client = None
        logger.info(f"Initializing MongoDB config with host={self.host}, port={self.port}, db={self.database}")

    def get_client(self):
        if not self._client:
            try:
                logger.info(f"Creating MongoDB client at {self.host}:{self.port}")
                self._client = MongoClient(
                    host=self.host,
                    port=self.port,
                    username=self.user,
                    password=self.password,
                    serverSelectionTimeoutMS=5000
                )
                logger.debug("MongoDB client created successfully")
            except Exception as e:
                logger.error(f"Failed to create MongoDB client: {str(e)}")
                raise
        return self._client

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")


mongo_db_config = MongoDBConfig()
client = None


def get_mongodb_client():
    global client
    if client is None:
        client = mongo_db_config.get_client()
    return client


def get_database():
    return get_mongodb_client()[mongo_db_config.database]
