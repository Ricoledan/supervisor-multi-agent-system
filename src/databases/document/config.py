import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBConfig:
    def __init__(self):
        self.host = os.getenv("MONGODB_HOST", "localhost")
        port_str = os.getenv("MONGODB_PORT", "27017")
        self.port = int(port_str)
        self.user = os.getenv("MONGODB_USER", "user")
        self.password = os.getenv("MONGODB_PASSWORD", "password")
        self.database = os.getenv("MONGODB_DB", "research_db")
        self._client = None
        logger.info(f"Initializing MongoDB config with host={self.host}, port={self.port}, db={self.database}")

    def get_client(self):
        if not self._client:
            try:
                logger.info(f"Creating MongoDB client at {self.host}:{self.port}")
                
                # Build connection string for Docker MongoDB
                if self.user and self.password:
                    connection_string = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?authSource=admin"
                else:
                    connection_string = f"mongodb://{self.host}:{self.port}/{self.database}"
                
                logger.info(f"Using connection string: {connection_string.replace(self.password, '***')}")
                
                self._client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000
                )
                
                # Test the connection
                self._client.server_info()
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
