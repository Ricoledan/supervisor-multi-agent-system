import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

class MongoDBConfig:
    def __init__(self):
        self.host = os.getenv("MONGODB_HOST")
        self.port = int(os.getenv("MONGODB_PORT"))
        self.user = os.getenv("MONGODB_USER")
        self.password = os.getenv("MONGODB_PASSWORD")
        self.database = os.getenv("MONGODB_DB")

    def get_client(self):
        return MongoClient(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password
        )

mongo_db_config = MongoDBConfig()
client = mongo_db_config.get_client()
db = client[mongo_db_config.database]