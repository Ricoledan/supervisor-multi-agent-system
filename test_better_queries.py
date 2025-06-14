#!/usr/bin/env python3
"""
Test queries that should match your actual paper content
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_database_content():
    """Check what terms actually exist in your databases"""
    print("🔍 Checking what's actually in your databases...")

    # Check Neo4j concepts
    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()

        with driver.session() as session:
            # Get sample concept names
            result = session.run("MATCH (c:Concept) RETURN c.name as name LIMIT 10")
            concepts = [record["name"] for record in result]

            print(f"📊 Neo4j sample concepts: {concepts[:5]}")

    except Exception as e:
        print(f"❌ Neo4j error: {e}")

    # Check MongoDB topics
    try:
        from src.databases.document.config import get_mongodb_client, mongo_db_config
        client = get_mongodb_client()
        db = client[mongo_db_config.database]

        # Get sample topic categories
        topics = list(db.topics.find({}).limit(5))
        categories = [topic.get('category', 'Unknown') for topic in topics]

        print(f"📊 MongoDB sample topics: {categories}")

        # Get sample paper titles
        papers = list(db.papers.find({}, {"metadata.title": 1}).limit(3))
        titles = [paper.get('metadata', {}).get('title', 'Unknown')[:50] + "..." for paper in papers]

        print(f"📄 Sample paper titles: {titles}")

    except Exception as e:
        print(f"❌ MongoDB error: {e}")


def suggest_good_queries():
    """Suggest queries that should work with your data"""

    good_queries = [
        "machine learning",
        "reinforcement learning",
        "neural networks",
        "deep learning",
        "language models",
        "transformers",
        "attention mechanisms",
        "natural language processing",
        "large language models",
        "training methods"
    ]

    print(f"\n💡 Try these queries that should match your papers:")
    for i, query in enumerate(good_queries, 1):
        print(f"   {i}. '{query}'")

    print(f"\n🚀 Test commands:")
    print(f"   python cli.py test -q 'machine learning' --simple")
    print(f"   python cli.py test -q 'reinforcement learning' --simple")
    print(f"   python cli.py test -q 'neural networks' --simple")


if __name__ == "__main__":
    test_database_content()
    suggest_good_queries()