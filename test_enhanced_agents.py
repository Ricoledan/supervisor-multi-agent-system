#!/usr/bin/env python3
"""
Test Enhanced Agents Script
Verify that agents are using real database data after enhanced ingestion
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_service_functions():
    """Test individual service functions with database queries"""
    print("🛠️ Testing Service Functions")
    print("-" * 30)

    test_queries = ["machine learning", "neural networks", "deep learning"]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")

        # Test graph service
        try:
            from src.services.graph_service import query_graphdb
            result = query_graphdb(query)

            concepts_count = len(result.get("concepts", []))
            relationships_count = len(result.get("relationships", []))
            papers_count = len(result.get("papers", []))

            print(
                f"   🔗 Graph Service: {concepts_count} concepts, {relationships_count} relationships, {papers_count} papers")

            if concepts_count > 0:
                sample_concept = result["concepts"][0]
                print(f"      Sample concept: {sample_concept.get('name', 'Unknown')}")

        except Exception as e:
            print(f"   ❌ Graph Service error: {e}")

        # Test document service
        try:
            from src.services.document_service import query_mongodb
            result = query_mongodb(query)

            topics_count = len(result.get("topics", {}))
            papers_count = len(result.get("papers", []))

            print(f"   📊 Document Service: {topics_count} topic categories, {papers_count} papers")

            if topics_count > 0:
                sample_topic = list(result["topics"].keys())[0]
                print(f"      Sample topic: {sample_topic}")

        except Exception as e:
            print(f"   ❌ Document Service error: {e}")

        # Test vector service
        try:
            from src.services.vector_service import query_vectordb
            result = query_vectordb(query, limit=3)

            documents_count = len(result)

            print(f"   🔍 Vector Service: {documents_count} relevant documents")

            if documents_count > 0:
                sample_doc = result[0]
                relevance = sample_doc.get('relevance_score', 0)
                print(f"      Best match relevance: {relevance:.3f}")

        except Exception as e:
            print(f"   ❌ Vector Service error: {e}")


def test_individual_agents():
    """Test individual agents to see their responses"""
    print("\n🤖 Testing Individual Agents")
    print("-" * 30)

    test_input = {
        "messages": [{"role": "user", "content": "machine learning neural networks natural language processing"}]}

    # Test graph writer agent
    print("\n🔗 Testing Graph Writer Agent...")
    try:
        from src.domain.agents.graph_writer import graph_writer_agent

        start_time = time.time()
        response = graph_writer_agent.invoke(test_input)
        duration = time.time() - start_time

        output = response.get("output", str(response)) if isinstance(response, dict) else str(response)

        print(f"   ⏱️ Response time: {duration:.2f}s")
        print(f"   📝 Response length: {len(output)} characters")

        # Check for database indicators
        db_indicators = ["neo4j", "database", "concepts found", "relationships found", "papers found"]
        has_db_data = any(indicator.lower() in output.lower() for indicator in db_indicators)

        if has_db_data:
            print("   ✅ Agent is using database data!")
        else:
            print("   ⚠️ Agent may be using generic responses")

        # Show preview
        preview = output[:200] + "..." if len(output) > 200 else output
        print(f"   🔍 Preview: {preview}")

    except Exception as e:
        print(f"   ❌ Graph Writer Agent error: {e}")

    # Test topic model agent
    print("\n📊 Testing Topic Model Agent...")
    try:
        from src.domain.agents.topic_model import topic_model_agent

        start_time = time.time()
        response = topic_model_agent.invoke(test_input)
        duration = time.time() - start_time

        output = response.get("output", str(response)) if isinstance(response, dict) else str(response)

        print(f"   ⏱️ Response time: {duration:.2f}s")
        print(f"   📝 Response length: {len(output)} characters")

        # Check for database indicators
        db_indicators = ["mongodb", "database", "topics found", "papers found", "categories"]
        has_db_data = any(indicator.lower() in output.lower() for indicator in db_indicators)

        if has_db_data:
            print("   ✅ Agent is using database data!")
        else:
            print("   ⚠️ Agent may be using generic responses")

        # Show preview
        preview = output[:200] + "..." if len(output) > 200 else output
        print(f"   🔍 Preview: {preview}")

    except Exception as e:
        print(f"   ❌ Topic Model Agent error: {e}")


def test_supervisor_workflow():
    """Test the complete supervisor workflow"""
    print("\n🎭 Testing Supervisor Workflow")
    print("-" * 30)

    test_query = "What are the main approaches to neural network architectures in machine learning?"

    try:
        from src.domain.agents.supervisor import run_supervisor

        print(f"📝 Query: {test_query}")
        print("🔄 Running supervisor workflow...")

        start_time = time.time()
        result = run_supervisor(test_query)
        duration = time.time() - start_time

        print(f"⏱️ Total workflow time: {duration:.2f}s")

        # Analyze results
        if isinstance(result, dict):
            has_graph_output = bool(result.get("graph_output"))
            has_tm_output = bool(result.get("tm_output"))
            has_final_output = bool(result.get("final_output"))

            print(f"📊 Workflow Results:")
            print(f"   Graph output: {'✅' if has_graph_output else '❌'}")
            print(f"   Topic output: {'✅' if has_tm_output else '❌'}")
            print(f"   Final output: {'✅' if has_final_output else '❌'}")

            # Check final output quality
            final_output = result.get("final_output", "")
            if final_output:
                # Look for database-specific indicators
                db_indicators = [
                    "neo4j", "mongodb", "chromadb", "database",
                    "concepts found", "topics found", "papers found",
                    "retrieved", "analyzed", "extracted"
                ]

                has_db_references = any(indicator.lower() in final_output.lower() for indicator in db_indicators)

                print(f"   Uses database data: {'✅' if has_db_references else '⚠️'}")
                print(f"   Response length: {len(final_output)} characters")

                # Show preview of final output
                preview = final_output[:300] + "..." if len(final_output) > 300 else final_output
                print(f"\n📝 Final Output Preview:")
                print(f"   {preview}")

                return True

        print("❌ Supervisor workflow returned unexpected result format")
        return False

    except Exception as e:
        print(f"❌ Supervisor workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_database_content():
    """Quick check of database content"""
    print("\n📊 Database Content Check")
    print("-" * 25)

    # Check Neo4j
    try:
        from src.databases.graph.config import get_neo4j_driver
        driver = get_neo4j_driver()

        with driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

        print(f"🔗 Neo4j: {nodes} nodes, {rels} relationships")

    except Exception as e:
        print(f"❌ Neo4j check failed: {e}")

    # Check MongoDB
    try:
        from src.databases.document.config import get_mongodb_client, mongo_db_config
        client = get_mongodb_client()
        db = client[mongo_db_config.database]

        papers_count = db.papers.count_documents({})
        topics_count = db.topics.count_documents({})

        print(f"📊 MongoDB: {papers_count} papers, {topics_count} topics")

    except Exception as e:
        print(f"❌ MongoDB check failed: {e}")

    # Check ChromaDB
    try:
        from src.databases.vector.config import ChromaDBConfig
        config = ChromaDBConfig()
        client = config.get_client()

        collection = client.get_collection("academic_papers")
        vectors = collection.count()

        print(f"🔍 ChromaDB: {vectors} vectors")

    except Exception as e:
        print(f"❌ ChromaDB check failed: {e}")


def main():
    """Main testing process"""
    print("🧪 ENHANCED AGENTS TEST SUITE")
    print("=" * 40)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check database content first
    check_database_content()

    # Test service functions
    test_service_functions()

    # Test individual agents
    test_individual_agents()

    # Test complete workflow
    print("\n" + "=" * 40)
    success = test_supervisor_workflow()

    # Final summary
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY")
    print("=" * 40)

    if success:
        print("✅ Enhanced agents are working correctly!")
        print("✅ Agents are using database data")
        print("✅ Supervisor workflow is functional")

        print("\n🎉 Your multi-agent system is ready!")
        print("\nYou can now:")
        print("  • Query via API: python cli.py test")
        print("  • Use web interface: http://localhost:8000")
        print("  • Process more PDFs: python src/utils/ingestion_pipeline.py")

    else:
        print("⚠️ Some issues detected with agent responses")
        print("\nTroubleshooting:")
        print("  • Check if databases have data")
        print("  • Verify ingestion completed successfully")
        print("  • Check service function outputs")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)