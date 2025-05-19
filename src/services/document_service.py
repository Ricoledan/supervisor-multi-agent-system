import logging
from typing import Dict, Any, List
import re
import nltk
from src.databases.document.config import MongoDBConfig

logger = logging.getLogger("document_service")

try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


def query_mongodb(query: str) -> Dict[str, Any]:
    """Query MongoDB for topic analysis data related to the input query."""
    logger.info(f"Querying MongoDB for: {query[:30]}...")
    try:
        mongo_config = MongoDBConfig()
        client = mongo_config.get_client()
        db = client[mongo_config.database]

        search_terms = extract_search_terms(query)
        if not search_terms:
            logger.warning("No meaningful search terms extracted from query")
            return {"topics": {}, "papers": []}

        patterns = [re.compile(f".*{term}.*", re.IGNORECASE) for term in search_terms]

        papers_collection = db.papers
        paper_query = {"$or": [{"title": {"$in": patterns}},
                               {"abstract": {"$in": patterns}},
                               {"keywords": {"$in": search_terms}}]}

        papers = list(papers_collection.find(
            paper_query,
            {"_id": 0, "title": 1, "authors": 1, "year": 1, "keywords": 1}
        ).limit(10))

        paper_ids = [p.get("paper_id") for p in papers if "paper_id" in p]

        topics_collection = db.topics
        topics_query = {"$or": [
            {"terms.term": {"$in": patterns}},
            {"paper_id": {"$in": paper_ids}}
        ]}

        topics_results = list(topics_collection.find(
            topics_query,
            {"_id": 0, "category": 1, "terms": 1}
        ).limit(5))

        organized_topics = {}
        for topic in topics_results:
            category = topic.get("category", "General")
            if category not in organized_topics:
                organized_topics[category] = []

            terms = topic.get("terms", [])
            organized_topics[category].extend(terms)

        result = {
            "topics": organized_topics,
            "papers": papers
        }

        logger.info(f"Retrieved {len(papers)} papers and {len(organized_topics)} topic categories")
        return result

    except Exception as e:
        logger.error(f"Error querying MongoDB: {str(e)}", exc_info=True)
        return {"topics": {}, "papers": []}


def extract_search_terms(query: str) -> List[str]:
    """Extract meaningful search terms from the query."""
    from nltk.corpus import stopwords

    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {"the", "and", "or", "in", "on", "at", "to", "a", "an", "is", "are",
                      "was", "were", "be", "been", "being", "have", "has", "had", "do",
                      "does", "did", "but", "if", "because", "as", "until", "while"}

    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    search_terms = [word for word in words if word not in stop_words]

    return search_terms[:5]
