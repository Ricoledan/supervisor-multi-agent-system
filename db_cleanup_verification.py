#!/usr/bin/env python3
"""
Database Cleanup and Verification Script
Clears old data and verifies the enhanced ingestion pipeline setup
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_neo4j():
    """Clean Neo4j database"""
    print("🔗 Cleaning Neo4j database...")
    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()

        with driver.session() as session:
            # Get counts before cleaning
            nodes_before = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels_before = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            print(f"   Before: {nodes_before} nodes, {rels_before} relationships")

            # Clean all data
            session.run("MATCH (n) DETACH DELETE n")

            # Verify cleaning
            nodes_after = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels_after = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

            print(f"   After: {nodes_after} nodes, {rels_after} relationships")
            print("   ✅ Neo4j cleaned successfully")
            return True

    except Exception as e:
        print(f"   ❌ Error cleaning Neo4j: {e}")
        return False


def clean_mongodb():
    """Clean MongoDB collections"""
    print("📊 Cleaning MongoDB collections...")
    try:
        from src.databases.document.config import get_mongodb_client, mongo_db_config
        client = get_mongodb_client()
        db = client[mongo_db_config.database]

        # Get collections before cleaning
        collections_before = db.list_collection_names()
        print(f"   Collections before: {collections_before}")

        # Clean papers collection
        papers_count_before = 0
        if "papers" in collections_before:
            papers_count_before = db.papers.count_documents({})
            db.papers.delete_many({})

        # Clean topics collection
        topics_count_before = 0
        if "topics" in collections_before:
            topics_count_before = db.topics.count_documents({})
            db.topics.delete_many({})

        print(f"   Removed {papers_count_before} papers and {topics_count_before} topics")
        print("   ✅ MongoDB cleaned successfully")
        return True

    except Exception as e:
        print(f"   ❌ Error cleaning MongoDB: {e}")
        return False


def clean_chromadb():
    """Clean ChromaDB collections"""
    print("🔍 Cleaning ChromaDB collections...")
    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()

        # List existing collections
        collections = client.list_collections()
        print(f"   Collections before: {[str(c) for c in collections]}")

        vectors_removed = 0

        # Clean academic_papers collection
        try:
            collection = client.get_collection("academic_papers")
            vectors_before = collection.count()

            # Delete the collection entirely and recreate
            client.delete_collection("academic_papers")
            client.create_collection("academic_papers")

            vectors_removed = vectors_before
            print(f"   Removed {vectors_removed} vectors from academic_papers")

        except Exception as e:
            print(f"   No academic_papers collection to clean: {e}")

        # Clean any other collections that might exist
        for collection_name in ["research_papers"]:  # Old naming
            try:
                collection = client.get_collection(collection_name)
                old_vectors = collection.count()
                client.delete_collection(collection_name)
                vectors_removed += old_vectors
                print(f"   Removed old collection: {collection_name} ({old_vectors} vectors)")
            except:
                pass  # Collection doesn't exist

        print(f"   ✅ ChromaDB cleaned successfully (removed {vectors_removed} total vectors)")
        return True

    except Exception as e:
        print(f"   ❌ Error cleaning ChromaDB: {e}")
        return False


def verify_database_connections():
    """Verify all databases are accessible"""
    print("🔌 Verifying database connections...")

    results = {"neo4j": False, "mongodb": False, "chromadb": False}

    # Test Neo4j
    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()
        driver.verify_connectivity()
        with driver.session() as session:
            session.run("RETURN 1")
        results["neo4j"] = True
        print("   ✅ Neo4j connection verified")
    except Exception as e:
        print(f"   ❌ Neo4j connection failed: {e}")

    # Test MongoDB
    try:
        from src.databases.document.config import get_mongodb_client
        client = get_mongodb_client()
        client.server_info()
        results["mongodb"] = True
        print("   ✅ MongoDB connection verified")
    except Exception as e:
        print(f"   ❌ MongoDB connection failed: {e}")

    # Test ChromaDB
    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()
        client.heartbeat()
        results["chromadb"] = True
        print("   ✅ ChromaDB connection verified")
    except Exception as e:
        print(f"   ❌ ChromaDB connection failed: {e}")

    return results


def verify_collection_structure():
    """Verify the expected collection structure exists"""
    print("📋 Verifying collection structure...")

    # Verify ChromaDB has academic_papers collection
    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()

        # Create academic_papers collection if it doesn't exist
        try:
            collection = client.get_collection("academic_papers")
            print("   ✅ academic_papers collection exists in ChromaDB")
        except:
            collection = client.create_collection("academic_papers")
            print("   ✅ Created academic_papers collection in ChromaDB")

    except Exception as e:
        print(f"   ❌ Error verifying ChromaDB structure: {e}")

    # Verify MongoDB collections will be created as needed
    try:
        from src.databases.document.config import get_mongodb_client, mongo_db_config
        client = get_mongodb_client()
        db = client[mongo_db_config.database]

        collections = db.list_collection_names()
        print(f"   📊 MongoDB collections: {collections}")
        print("   ✅ MongoDB structure ready (collections will be created during ingestion)")

    except Exception as e:
        print(f"   ❌ Error verifying MongoDB structure: {e}")


def check_sources_directory():
    """Check sources directory and PDF files"""
    print("📁 Checking sources directory...")

    sources_path = Path("sources")

    if not sources_path.exists():
        print("   ⚠️ Sources directory not found")
        print("   Creating sources directory...")
        sources_path.mkdir(exist_ok=True)
        print("   📝 Add PDF files to sources/ directory for ingestion")
        return False

    pdf_files = list(sources_path.glob("*.pdf"))

    if not pdf_files:
        print("   ⚠️ No PDF files found in sources/")
        print("   📝 Add academic PDF files to sources/ directory")
        return False

    print(f"   ✅ Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files[:5], 1):
        size_kb = pdf.stat().st_size // 1024
        print(f"      {i}. {pdf.name} ({size_kb}KB)")

    if len(pdf_files) > 5:
        print(f"      ... and {len(pdf_files) - 5} more files")

    return True


def main():
    """Main cleanup and verification process"""
    print("🧹 DATABASE CLEANUP AND VERIFICATION")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Verify connections
    connections = verify_database_connections()
    connected_dbs = sum(connections.values())

    if connected_dbs == 0:
        print("\n❌ No database connections available")
        print("   Make sure Docker services are running:")
        print("   docker-compose up -d")
        return False

    print(f"\n✅ {connected_dbs}/3 databases connected")

    # Step 2: Clean existing data
    print("\n🧹 CLEANING EXISTING DATA")
    print("-" * 30)

    cleaned_dbs = 0
    if connections["neo4j"]:
        if clean_neo4j():
            cleaned_dbs += 1

    if connections["mongodb"]:
        if clean_mongodb():
            cleaned_dbs += 1

    if connections["chromadb"]:
        if clean_chromadb():
            cleaned_dbs += 1

    print(f"\n✅ Cleaned {cleaned_dbs}/{connected_dbs} available databases")

    # Step 3: Verify structure
    print("\n📋 VERIFYING STRUCTURE")
    print("-" * 25)
    verify_collection_structure()

    # Step 4: Check sources
    print("\n📁 CHECKING SOURCE FILES")
    print("-" * 25)
    has_sources = check_sources_directory()

    # Final summary
    print("\n" + "=" * 50)
    print("📊 CLEANUP AND VERIFICATION SUMMARY")
    print("=" * 50)

    print(f"✅ Database connections: {connected_dbs}/3")
    print(f"✅ Databases cleaned: {cleaned_dbs}/{connected_dbs}")
    print(f"✅ Sources available: {'Yes' if has_sources else 'No'}")

    if connected_dbs == 3 and cleaned_dbs == 3:
        print("\n🎉 READY FOR ENHANCED INGESTION!")
        print("\nNext steps:")
        print("1. Run enhanced ingestion test:")
        print("   python src/utils/ingestion_pipeline.py --test")
        print("\n2. If test succeeds, run full ingestion:")
        print("   python src/utils/ingestion_pipeline.py")
        print("\n3. Test agent responses:")
        print("   python cli.py test -q 'machine learning neural networks'")

        return True
    else:
        print("\n⚠️ SOME ISSUES DETECTED")
        print("Fix database connections before running ingestion")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Cleanup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)