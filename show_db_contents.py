#!/usr/bin/env python3
"""
Database Content Viewer
Shows exactly what's in your Neo4j, MongoDB, and ChromaDB databases
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def show_neo4j_contents():
    """Display all Neo4j database contents"""
    print("🔗 NEO4J DATABASE CONTENTS")
    print("-" * 40)

    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()

        with driver.session() as session:
            # Get all labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]
            print(f"📋 Node Types: {labels}")

            # Get all relationship types
            rels_result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in rels_result]
            print(f"🔗 Relationship Types: {rel_types}")

            # Count everything
            nodes_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"📊 Total: {nodes_count} nodes, {rels_count} relationships")

            if nodes_count == 0:
                print("📭 Database is empty")
                return

            # Show each node type
            for label in labels:
                print(f"\n📄 {label} nodes:")
                if label == "Paper":
                    papers = session.run(f"MATCH (n:{label}) RETURN n LIMIT 5").data()
                    for i, paper in enumerate(papers, 1):
                        node = paper['n']
                        title = node.get('title', 'No title')[:50]
                        year = node.get('year', 'No year')
                        print(f"   {i}. {title}... ({year})")

                elif label == "Concept":
                    concepts = session.run(f"MATCH (n:{label}) RETURN n LIMIT 10").data()
                    for i, concept in enumerate(concepts, 1):
                        node = concept['n']
                        name = node.get('name', 'No name')
                        category = node.get('category', 'No category')
                        print(f"   {i}. {name} ({category})")

                elif label == "Author":
                    authors = session.run(f"MATCH (n:{label}) RETURN n LIMIT 10").data()
                    for i, author in enumerate(authors, 1):
                        node = author['n']
                        name = node.get('name', 'No name')
                        print(f"   {i}. {name}")

                else:
                    # Generic node display
                    nodes = session.run(f"MATCH (n:{label}) RETURN n LIMIT 5").data()
                    for i, node_data in enumerate(nodes, 1):
                        node = node_data['n']
                        # Try to find a meaningful property to display
                        display_prop = node.get('name') or node.get('title') or node.get('id') or str(node)[:30]
                        print(f"   {i}. {display_prop}")

            # Show some relationships
            if rels_count > 0:
                print(f"\n🔗 Sample Relationships:")
                rels = session.run("""
                    MATCH (a)-[r]->(b) 
                    RETURN labels(a)[0] as from_type, 
                           coalesce(a.name, a.title, a.id) as from_name,
                           type(r) as rel_type,
                           labels(b)[0] as to_type, 
                           coalesce(b.name, b.title, b.id) as to_name
                    LIMIT 10
                """).data()

                for i, rel in enumerate(rels, 1):
                    print(
                        f"   {i}. {rel['from_type']}({rel['from_name']}) -[{rel['rel_type']}]-> {rel['to_type']}({rel['to_name']})")

    except Exception as e:
        print(f"❌ Error accessing Neo4j: {e}")


def show_mongodb_contents():
    """Display all MongoDB database contents"""
    print("\n📊 MONGODB DATABASE CONTENTS")
    print("-" * 40)

    try:
        from src.databases.document.config import get_mongodb_client, mongo_db_config
        client = get_mongodb_client()
        db = client[mongo_db_config.database]

        collections = db.list_collection_names()
        print(f"📁 Collections: {collections}")

        if not collections:
            print("📭 No collections found")
            return

        # Show papers collection
        if "papers" in collections:
            papers_collection = db.papers
            papers_count = papers_collection.count_documents({})
            print(f"\n📄 Papers Collection: {papers_count} documents")

            if papers_count > 0:
                # Show sample papers
                sample_papers = papers_collection.find({}).limit(3)
                for i, paper in enumerate(sample_papers, 1):
                    metadata = paper.get('metadata', {})
                    title = metadata.get('title', 'No title')[:50]
                    authors = metadata.get('authors', [])
                    year = metadata.get('year', 'Unknown')
                    source = paper.get('source', 'Unknown source')

                    print(f"   {i}. {title}...")
                    print(f"      Authors: {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}")
                    print(f"      Year: {year}, Source: {source}")

                    # Show content structure
                    content = paper.get('content', [])
                    entities = paper.get('entities', {})
                    concepts = entities.get('concepts', [])
                    relationships = entities.get('relationships', [])

                    print(f"      Content: {len(content)} pages")
                    print(f"      Entities: {len(concepts)} concepts, {len(relationships)} relationships")

        # Show topics collection
        if "topics" in collections:
            topics_collection = db.topics
            topics_count = topics_collection.count_documents({})
            print(f"\n🏷️ Topics Collection: {topics_count} documents")

            if topics_count > 0:
                # Show sample topics
                sample_topics = topics_collection.find({}).limit(5)
                for i, topic in enumerate(sample_topics, 1):
                    category = topic.get('category', 'Unknown category')
                    terms = topic.get('terms', [])

                    print(f"   {i}. {category}")
                    if isinstance(terms, list) and terms:
                        # Handle both string terms and dict terms
                        term_strs = []
                        for term in terms[:5]:
                            if isinstance(term, dict):
                                term_strs.append(term.get('term', str(term)))
                            else:
                                term_strs.append(str(term))
                        print(f"      Terms: {', '.join(term_strs)}{'...' if len(terms) > 5 else ''}")

    except Exception as e:
        print(f"❌ Error accessing MongoDB: {e}")


def show_chromadb_contents():
    """Display all ChromaDB database contents"""
    print("\n🔍 CHROMADB DATABASE CONTENTS")
    print("-" * 40)

    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()

        collections = client.list_collections()
        print(f"📚 Collections: {[str(c) for c in collections] if collections else 'None'}")

        if not collections:
            print("📭 No collections found")
            return

        # Show academic_papers collection specifically
        try:
            collection = client.get_collection("academic_papers")
            count = collection.count()
            print(f"\n📄 Academic Papers Collection: {count} vectors")

            if count > 0:
                # Get sample data
                sample = collection.peek(limit=5)

                if sample.get("documents"):
                    print("\n   Sample Documents:")
                    documents = sample["documents"]
                    metadatas = sample.get("metadatas", [{}] * len(documents))
                    ids = sample.get("ids", [f"doc_{i}" for i in range(len(documents))])

                    for i, (doc, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
                        title = metadata.get("title", "Unknown title")
                        source = metadata.get("source", "Unknown source")
                        chunk_id = metadata.get("chunk_id", "?")

                        print(f"   {i + 1}. ID: {doc_id}")
                        print(f"      Title: {title}")
                        print(f"      Source: {source} (chunk {chunk_id})")
                        print(f"      Content: {doc[:100]}...")
                        print()

                # Show metadata structure
                if sample.get("metadatas"):
                    print("   Metadata Keys Available:")
                    all_keys = set()
                    for metadata in sample["metadatas"]:
                        all_keys.update(metadata.keys())
                    print(f"      {', '.join(sorted(all_keys))}")

        except Exception as e:
            print(f"❌ Error accessing academic_papers collection: {e}")

        # Try to list all collections and their counts
        print("\n📊 All Collection Statistics:")
        for collection_name in [str(c) for c in collections]:
            try:
                coll = client.get_collection(collection_name)
                count = coll.count()
                print(f"   {collection_name}: {count} vectors")
            except Exception as e:
                print(f"   {collection_name}: Error - {e}")

    except Exception as e:
        print(f"❌ Error accessing ChromaDB: {e}")


def show_service_query_results():
    """Show what the service functions actually return"""
    print("\n🛠️ SERVICE FUNCTION OUTPUTS")
    print("-" * 40)

    test_query = "deep learning"
    print(f"Testing with query: '{test_query}'")

    # Test graph service
    print("\n🔗 Graph Service Result:")
    try:
        from src.services.graph_service import query_graphdb
        result = query_graphdb(test_query)

        print(f"   Type: {type(result)}")
        print(f"   Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if isinstance(result, dict):
            concepts = result.get("concepts", [])
            relationships = result.get("relationships", [])
            papers = result.get("papers", [])

            print(f"   Concepts: {len(concepts)}")
            if concepts:
                for i, concept in enumerate(concepts[:3], 1):
                    name = concept.get("name", "No name")
                    category = concept.get("category", "No category")
                    print(f"      {i}. {name} ({category})")

            print(f"   Relationships: {len(relationships)}")
            if relationships:
                for i, rel in enumerate(relationships[:3], 1):
                    from_name = rel.get("from", "?")
                    to_name = rel.get("to", "?")
                    rel_type = rel.get("type", "?")
                    print(f"      {i}. {from_name} -[{rel_type}]-> {to_name}")

            print(f"   Papers: {len(papers)}")
            if papers:
                for i, paper in enumerate(papers[:3], 1):
                    title = paper.get("title", "No title")[:50]
                    year = paper.get("year", "No year")
                    print(f"      {i}. {title}... ({year})")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test document service
    print("\n📊 Document Service Result:")
    try:
        from src.services.document_service import query_mongodb
        result = query_mongodb(test_query)

        print(f"   Type: {type(result)}")
        print(f"   Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        if isinstance(result, dict):
            topics = result.get("topics", {})
            papers = result.get("papers", [])

            print(f"   Topic Categories: {len(topics)}")
            if topics:
                for category, terms in list(topics.items())[:3]:
                    print(f"      {category}: {len(terms)} terms")
                    if terms:
                        # Handle different term formats
                        term_names = []
                        for term in terms[:3]:
                            if isinstance(term, dict):
                                term_names.append(term.get('term', str(term)))
                            else:
                                term_names.append(str(term))
                        print(f"         {', '.join(term_names)}...")

            print(f"   Papers: {len(papers)}")
            if papers:
                for i, paper in enumerate(papers[:3], 1):
                    title = paper.get("title", "No title")[:50]
                    authors = paper.get("authors", [])
                    print(f"      {i}. {title}... by {', '.join(authors[:2])}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test vector service
    print("\n🔍 Vector Service Result:")
    try:
        from src.services.vector_service import query_vectordb
        result = query_vectordb(test_query, limit=3)

        print(f"   Type: {type(result)}")
        print(f"   Count: {len(result) if isinstance(result, list) else 'Not a list'}")

        if isinstance(result, list) and result:
            print("   Sample Results:")
            for i, doc in enumerate(result[:3], 1):
                if isinstance(doc, dict):
                    content = doc.get("content", "No content")[:100]
                    metadata = doc.get("metadata", {})
                    relevance = doc.get("relevance_score", 0)
                    title = metadata.get("title", "No title")

                    print(f"      {i}. {title} (relevance: {relevance:.3f})")
                    print(f"         Content: {content}...")
                else:
                    print(f"      {i}. {doc}")

    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    """Show all database contents"""
    print("🔍 DATABASE CONTENT INSPECTION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Show raw database contents
    show_neo4j_contents()
    show_mongodb_contents()
    show_chromadb_contents()

    # Show what service functions return
    show_service_query_results()

    print("\n" + "=" * 60)
    print("✅ Content inspection complete")
    print("\n💡 If databases are empty, run:")
    print("   python populate_test_data.py")
    print("   or")
    print("   python src/utils/ingestion_pipeline.py --test")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Inspection interrupted")
    except Exception as e:
        print(f"\n❌ Error during inspection: {e}")
        import traceback

        traceback.print_exc()