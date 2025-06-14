#!/usr/bin/env python3
"""
Quick database check to verify data exists
Save as: quick_db_check.py
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_neo4j():
    """Check Neo4j database status and content"""
    print("\n🔍 Neo4j Status Check")
    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Count total nodes
            result = session.run("MATCH (n) RETURN count(n) as total")
            total_nodes = result.single()["total"]
            print(f"  📊 Total nodes: {total_nodes}")

            if total_nodes > 0:
                # Count by type
                result = session.run("""
                    MATCH (n) 
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY count DESC
                """)
                for record in result:
                    label = record["label"] or "Unknown"
                    count = record["count"]
                    print(f"    📄 {label}: {count}")

                # Test a simple query
                result = session.run("""
                    MATCH (c:Concept) 
                    RETURN c.name as name 
                    LIMIT 3
                """)
                concepts = [record["name"] for record in result]
                if concepts:
                    print(f"    🔍 Sample concepts: {concepts}")
            else:
                print("    ❌ No data found in Neo4j")

            return total_nodes > 0
    except Exception as e:
        print(f"  ❌ Neo4j Error: {e}")
        return False


def check_mongodb():
    """Check MongoDB status and content"""
    print("\n🔍 MongoDB Status Check")
    try:
        from src.databases.document.config import get_database
        db = get_database()

        collections = db.list_collection_names()
        print(f"  📂 Collections: {collections}")

        has_data = False
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"    📊 {collection_name}: {count} documents")
            if count > 0:
                has_data = True

                # Show a sample document
                sample = collection.find_one()
                if sample:
                    print(f"      🔍 Sample keys: {list(sample.keys())[:5]}")

        if not has_data:
            print("    ❌ No data found in MongoDB")

        return has_data
    except Exception as e:
        print(f"  ❌ MongoDB Error: {e}")
        return False


def check_chromadb():
    """Check ChromaDB status and content"""
    print("\n🔍 ChromaDB Status Check")
    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()

        collections = client.list_collections()
        print(f"  📂 Collections: {[c.name if hasattr(c, 'name') else str(c) for c in collections]}")

        has_data = False
        # Try common collection names
        for collection_name in ["academic_papers", "research_papers"]:
            try:
                collection = client.get_collection(collection_name)
                count = collection.count()
                print(f"    📊 {collection_name}: {count} vectors")
                if count > 0:
                    has_data = True
                    # Show sample metadata
                    sample = collection.peek(limit=1)
                    if sample['metadatas'] and len(sample['metadatas']) > 0:
                        keys = list(sample['metadatas'][0].keys())
                        print(f"      🔍 Metadata keys: {keys}")
            except Exception as e:
                print(f"    ⚠️ Collection {collection_name}: {e}")

        if not has_data:
            print("    ❌ No data found in ChromaDB")

        return has_data
    except Exception as e:
        print(f"  ❌ ChromaDB Error: {e}")
        return False


def test_database_queries():
    """Test the actual database query functions"""
    print("\n🧪 Testing Database Query Functions")

    # Test graph service
    try:
        from src.services.graph_service import query_graphdb
        result = query_graphdb("language model")
        concepts = result.get("concepts", [])
        papers = result.get("papers", [])
        print(f"  🔗 Graph query returned: {len(concepts)} concepts, {len(papers)} papers")
        if concepts:
            print(f"    Sample concept: {concepts[0].get('name', 'Unknown')}")
    except Exception as e:
        print(f"  ❌ Graph service error: {e}")

    # Test document service
    try:
        from src.services.document_service import query_mongodb
        result = query_mongodb("language model")
        topics = result.get("topics", {})
        papers = result.get("papers", [])
        print(f"  📊 Document query returned: {len(topics)} topics, {len(papers)} papers")
        if papers:
            print(f"    Sample paper: {papers[0].get('title', 'Unknown')[:50]}...")
    except Exception as e:
        print(f"  ❌ Document service error: {e}")


def main():
    """Run comprehensive database check"""
    print("🏥 Quick Database Health Check")
    print("=" * 40)

    neo4j_ok = check_neo4j()
    mongo_ok = check_mongodb()
    chroma_ok = check_chromadb()

    test_database_queries()

    print("\n📋 Summary")
    print(f"  Neo4j (Graph): {'✅ Has Data' if neo4j_ok else '❌ Empty/Issues'}")
    print(f"  MongoDB (Documents): {'✅ Has Data' if mongo_ok else '❌ Empty/Issues'}")
    print(f"  ChromaDB (Vectors): {'✅ Has Data' if chroma_ok else '❌ Empty/Issues'}")

    if not any([neo4j_ok, mongo_ok, chroma_ok]):
        print("\n🚨 CRITICAL: All databases appear empty!")
        print("   Next steps:")
        print("   1. Run: python src/utils/ingestion_pipeline.py --test")
        print("   2. Check if your 'sources' folder has PDF files")
        print("   3. Verify database connections in docker-compose.yml")
    elif not all([neo4j_ok, mongo_ok, chroma_ok]):
        print("\n⚠️ WARNING: Some databases are empty")
        print("   Run ingestion pipeline to populate missing databases")
    else:
        print("\n🎉 SUCCESS: All databases have data!")
        print("   Your agents should now work with real data")


if __name__ == "__main__":
    main()