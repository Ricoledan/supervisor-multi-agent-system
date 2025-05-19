import logging
from typing import Dict, Any, List
from src.databases.graph.config import Neo4jConfig

logger = logging.getLogger("graph_service")


def query_graphdb(query: str) -> Dict[str, Any]:
    """Query Neo4j graph database for relevant relationships."""
    logger.info(f"Querying graph DB for: {query[:30]}...")
    try:
        neo4j_config = Neo4jConfig()
        driver = neo4j_config.get_driver()

        key_terms = extract_key_terms(query)
        if not key_terms:
            logger.warning("No key terms extracted from query")
            return {"concepts": [], "relationships": [], "papers": []}

        result = {
            "concepts": [],
            "relationships": [],
            "papers": []
        }

        with driver.session() as session:
            # Find relevant concepts based on query terms
            concepts_query = """
            MATCH (c:Concept)
            WHERE ANY(term IN $terms WHERE toLower(c.name) CONTAINS toLower(term) OR toLower(term) CONTAINS toLower(c.name))
            RETURN c.name as name, c.category as category, c.description as description
            LIMIT 15
            """
            concepts_result = session.run(concepts_query, terms=key_terms)
            concepts = [{
                "name": record["name"],
                "category": record.get("category", ""),
                "description": record.get("description", "")
            } for record in concepts_result]

            result["concepts"] = concepts

            if concepts:
                concept_names = [c["name"] for c in concepts]
                rels_query = """
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE a.name IN $names AND b.name IN $names
                RETURN a.name as from, b.name as to, type(r) as type, r.description as description
                LIMIT 30
                """
                rels_result = session.run(rels_query, names=concept_names)
                relationships = [{
                    "from": record["from"],
                    "to": record["to"],
                    "type": record.get("type", "related_to"),
                    "description": record.get("description", "")
                } for record in rels_result]

                result["relationships"] = relationships

                papers_query = """
                MATCH (p:Paper)-[:CONTAINS|MENTIONS|CITES]->(c:Concept)
                WHERE c.name IN $names
                RETURN p.title as title, p.year as year, p.authors as authors,
                       collect(distinct c.name) as concepts
                LIMIT 10
                """
                papers_result = session.run(papers_query, names=concept_names)
                papers = [{
                    "title": record["title"],
                    "year": record.get("year", "Unknown"),
                    "authors": record.get("authors", []),
                    "related_concepts": record["concepts"]
                } for record in papers_result]

                result["papers"] = papers

        logger.info(f"Retrieved {len(result['concepts'])} concepts and {len(result['relationships'])} relationships from Neo4j")
        return result
    except Exception as e:
        logger.error(f"Error querying graph DB: {str(e)}", exc_info=True)
        return {"concepts": [], "relationships": [], "papers": []}


def extract_key_terms(query: str, max_terms: int = 5) -> List[str]:
    """Extract key terms from the query for database searching."""
    import re

    common_words = {"the", "and", "or", "in", "on", "at", "to", "a", "an", "is", "are",
                   "was", "were", "be", "been", "being", "have", "has", "had", "do",
                   "does", "did", "but", "if", "because", "as", "until", "while", "of",
                   "with", "about", "for", "by", "how", "what", "when", "where", "who", "why"}

    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    key_terms = [word for word in words if word not in common_words]

    return key_terms[:max_terms]