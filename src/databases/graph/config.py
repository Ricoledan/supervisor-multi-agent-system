import os
import time
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

load_dotenv()

logger = logging.getLogger(__name__)

class Neo4jConfig:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DB")
        self._driver = None
        logger.info(f"Initializing Neo4j config with URI={self.uri}, DB={self.database}")

    def get_driver(self, max_retries=5, retry_delay=3):
        if self._driver is None:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"Attempting to connect to Neo4j at {self.uri} (attempt {retry_count+1}/{max_retries})")
                    self._driver = GraphDatabase.driver(
                        self.uri,
                        auth=(self.user, self.password)
                    )
                    # Test the connection
                    self._driver.verify_connectivity()
                    logger.info("Successfully connected to Neo4j")
                    break
                except ServiceUnavailable as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Neo4j connection failed, retrying in {retry_delay} seconds: {str(e)}")
                    time.sleep(retry_delay)
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

neo4j_config = Neo4jConfig()
driver = None

def get_neo4j_driver():
    global driver
    if driver is None:
        driver = neo4j_config.get_driver()
    return driver