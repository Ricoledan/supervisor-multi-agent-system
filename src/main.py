from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.v1.endpoints import status, agent
from src.databases.graph.config import get_neo4j_driver, neo4j_config
from src.databases.vector.config import ChromaDBConfig
from src.databases.document.config import mongo_db_config, get_mongodb_client
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chroma_db_config = ChromaDBConfig()
chroma_client = chroma_db_config.get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")

    max_retries = 5
    retry_delay = 3
    for attempt in range(1, max_retries + 1):
        try:
            driver = get_neo4j_driver()
            driver.verify_connectivity()
            logger.info("✅ graph database connection established")
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"⚠️ graph database connection attempt {attempt} failed: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.critical(f"❌ Failed to connect to graph database after {max_retries} attempts: {str(e)}")
                raise

    try:
        chroma_client.list_collections()
        logger.info(f"✅ vector database connected at {chroma_db_config.host}:{chroma_db_config.port}")
    except Exception as e:
        logger.critical(f"❌ Failed to connect to vector database: {str(e)}")
        raise

    for attempt in range(1, max_retries + 1):
        try:
            mongo_client = get_mongodb_client()
            logger.info(f"✅ document database connected at {mongo_db_config.host}:{mongo_db_config.port}")
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"⚠️ document database connection attempt {attempt} failed: {str(e)}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.critical(f"❌ Failed to connect to document database after {max_retries} attempts: {str(e)}")
                raise

    yield

    neo4j_config.close()
    logger.info("🧹graph database connection closed")

    mongo_db_config.close()
    logger.info("🧹connection closed")

app = FastAPI(lifespan=lifespan)

app.include_router(status.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")