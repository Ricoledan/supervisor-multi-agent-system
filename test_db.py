#!/usr/bin/env python3
"""
Test the fixed database connections and queries
Save as: test_databases.py
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_neo4j():
    """Test Neo4j with fixed queries"""
    print("🔍 Testing Neo4j (Fixed)...")
    try:
        from src.services.graph_service import query_graphdb
        result = query_graphdb("language model reasoning")

        concepts = result.get("concepts", [])
        papers = result.get("papers", [])
        relationships = result.get("relationships", [])

        print(f"  ✅ Found {len(concepts)} concepts")
        print(f"  ✅ Found {len(papers)} papers")
        print(f"  ✅ Found {len(relationships)} relationships")

        if concepts:
            print(f"  📝 Sample concept: {concepts[0].get('name', 'Unknown')}")
        if papers:
            print(f"  📄 Sample paper: {papers[0].get('title', 'Unknown')[:50]}...")

        return True
    except Exception as e:
        print(f"  ❌ Neo4j error: {e}")
        return False


def test_mongodb():
    """Test MongoDB with fixed queries"""
    print("\n🔍 Testing MongoDB (Fixed)...")
    try:
        from src.services.document_service import query_mongodb
        result = query_mongodb("language model reasoning")

        topics = result.get("topics", {})
        papers = result.get("papers", [])

        print(f"  ✅ Found {len(topics)} topic categories")
        print(f"  ✅ Found {len(papers)} papers")

        if topics:
            print(f"  🏷️ Topic categories: {list(topics.keys())}")
        if papers:
            print(f"  📄 Sample paper: {papers[0].get('title', 'Unknown')[:50]}...")

        return True
    except Exception as e:
        print(f"  ❌ MongoDB error: {e}")
        return False


def test_chromadb():
    """Test ChromaDB with fixed queries"""
    print("\n🔍 Testing ChromaDB (Fixed)...")
    try:
        from src.services.vector_service import query_vectordb
        results = query_vectordb("language model reasoning", limit=3)

        print(f"  ✅ Found {len(results)} vector results")

        if results:
            for i, result in enumerate(results[:2]):
                relevance = result.get('relevance_score', 0)
                content = result.get('content', '')[:100]
                print(f"  📝 Result {i + 1} (relevance: {relevance:.3f}): {content}...")

        return True
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")
        return False


def test_agent_tools():
    """Test the actual agent tools"""
    print("\n🤖 Testing Agent Tools...")

    try:
        # Test graph agent tool
        from src.domain.agents.graph_writer import enhanced_graph_tool
        graph_result = enhanced_graph_tool("language model reasoning")
        print(f"  🔗 Graph agent tool: {len(graph_result)} chars output")

        # Test topic agent tool
        from src.domain.agents.topic_model import topic_tool
        topic_result = topic_tool("language model reasoning")
        print(f"  📊 Topic agent tool: {len(topic_result)} chars output")

        return True
    except Exception as e:
        print(f"  ❌ Agent tools error: {e}")
        return False


def main():
    print("🧪 Testing Fixed Database Connections")
    print("=" * 40)

    # Test each database
    neo4j_ok = test_neo4j()
    mongo_ok = test_mongodb()
    chroma_ok = test_chromadb()
    agents_ok = test_agent_tools()

    print("\n📋 Summary")
    print(f"  Neo4j: {'✅' if neo4j_ok else '❌'}")
    print(f"  MongoDB: {'✅' if mongo_ok else '❌'}")
    print(f"  ChromaDB: {'✅' if chroma_ok else '❌'}")
    print(f"  Agent Tools: {'✅' if agents_ok else '❌'}")

    if all([neo4j_ok, mongo_ok, chroma_ok, agents_ok]):
        print("\n🎉 All systems working! Your agents should now use real data.")
        print("\nNext steps:")
        print("1. Start API: python -m uvicorn src.main:app --reload")
        print("2. Test query: curl -X POST http://localhost:8000/api/v1/agent \\")
        print("   -H 'Content-Type: application/json' \\")
        print("   -d '{\"query\": \"What are the main LLM reasoning approaches?\"}'")
    else:
        print("\n⚠️ Some issues found. Apply the database fixes above.")


if __name__ == "__main__":
    main()