import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

class Neo4jConfig:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DB")

    def get_driver(self):
        return GraphDatabase.driver(self.uri, auth=(self.user, self.password))

neo4j_config = Neo4jConfig()
driver = neo4j_config.get_driver()