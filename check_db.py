#!/usr/bin/env python3
"""
Fixed database status check for the multi-agent system
"""

import logging
from src.databases.graph.config import get_neo4j_driver
from src.databases.vector.config import ChromaDBConfig
from src.databases.document.config import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_neo4j():
    """Check Neo4j database status and content"""
    print("\n🔍 Neo4j Status Check")
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            # Count nodes by type
            result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)

            total_nodes = 0
            for record in result:
                label = record["label"] or "Unknown"
                count = record["count"]
                total_nodes += count
                print(f"  📊 {label}: {count} nodes")

            print(f"  📈 Total nodes: {total_nodes}")

            # Check relationships
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)

            total_rels = 0
            for record in rel_result:
                rel_type = record["rel_type"]
                count = record["count"]
                total_rels += count
                print(f"  🔗 {rel_type}: {count} relationships")

            print(f"  📈 Total relationships: {total_rels}")

            return total_nodes > 0

    except Exception as e:
        print(f"  ❌ Neo4j Error: {e}")
        return False


def check_mongodb():
    """Check MongoDB status and content - FIXED"""
    print("\n🔍 MongoDB Status Check")
    try:
        db = get_database()  # Use the fixed get_database function

        collections = db.list_collection_names()
        print(f"  📂 Collections: {collections}")

        has_data = False
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})
            print(f"  📊 {collection_name}: {count} documents")
            if count > 0:
                has_data = True

                # Show sample document structure
                sample = collection.find_one()
                if sample:
                    keys = list(sample.keys())[:5]  # First 5 keys
                    print(f"    🔑 Sample keys: {keys}")

        return has_data

    except Exception as e:
        print(f"  ❌ MongoDB Error: {e}")
        return False


def check_chromadb():
    """Check ChromaDB status and content - FIXED for v0.6.0"""
    print("\n🔍 ChromaDB Status Check")
    try:
        config = ChromaDBConfig()

        # Use the new v0.6.0 compatible method
        collection_names = config.list_collections()
        print(f"  📂 Collections: {collection_names}")

        has_data = False

        # Check known collections
        known_collections = ["academic_papers", "research_papers"]

        for collection_name in known_collections:
            try:
                collection = config.get_collection(collection_name)
                count = collection.count()
                print(f"  📊 {collection_name}: {count} vectors")
                if count > 0:
                    has_data = True

                    # Show sample metadata
                    sample = collection.peek(limit=1)
                    if sample['metadatas'] and len(sample['metadatas']) > 0:
                        keys = list(sample['metadatas'][0].keys())
                        print(f"    🔑 Metadata keys: {keys}")
            except Exception as e:
                print(f"  ⚠️ Collection {collection_name} not accessible: {e}")

        return has_data

    except Exception as e:
        print(f"  ❌ ChromaDB Error: {e}")
        return False


def main():
    """Run comprehensive database check - FIXED VERSION"""
    print("🏥 Multi-Agent System Database Health Check (FIXED)")
    print("=" * 55)

    neo4j_ok = check_neo4j()
    mongo_ok = check_mongodb()
    chroma_ok = check_chromadb()

    print("\n📋 Summary")
    print(f"  Neo4j (Graph): {'✅ Has Data' if neo4j_ok else '⚠️ Empty/Issues'}")
    print(f"  MongoDB (Documents): {'✅ Has Data' if mongo_ok else '⚠️ Empty/Issues'}")
    print(f"  ChromaDB (Vectors): {'✅ Has Data' if chroma_ok else '⚠️ Empty/Issues'}")

    if not any([neo4j_ok, mongo_ok, chroma_ok]):
        print("\n🚨 CRITICAL: All databases appear empty!")
        print("   Run ingestion pipeline: python src/utils/ingestion_pipeline.py --test")
    elif not all([neo4j_ok, mongo_ok, chroma_ok]):
        print("\n⚠️ WARNING: Some databases are empty or have issues")
        print("   Your Neo4j has good data! Let's populate the others.")
        print("   Run: python src/utils/ingestion_pipeline.py --source sources")
    else:
        print("\n🎉 SUCCESS: All databases have data!")
        print("   Your multi-agent system should work well.")


if __name__ == "__main__":
    main()