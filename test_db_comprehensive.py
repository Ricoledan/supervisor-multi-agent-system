#!/usr/bin/env python3
"""
Database Diagnostic Script for Multi-Agent Research System
Deep analysis of Neo4j, MongoDB, ChromaDB and agent behavior
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging to capture all details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseDiagnostic:
    """Deep diagnostic analysis of database and agent behavior"""

    def __init__(self):
        self.findings = {
            "environment": {},
            "database_connectivity": {},
            "database_content": {},
            "agent_behavior": {},
            "query_tracing": {},
            "configuration_analysis": {}
        }

        print("🔬 Deep Database & Agent Diagnostic Analysis")
        print("=" * 60)

    def analyze_environment(self):
        """Analyze environment configuration in detail"""
        print("\n🔧 Environment Configuration Analysis...")

        # Check Python path and imports
        python_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_path": sys.path[:5],
            "working_directory": str(Path.cwd()),
            "script_location": str(Path(__file__).parent)
        }

        # Environment variables
        env_vars = {
            "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "MISSING",
            "NEO4J_URI": os.getenv("NEO4J_URI", "NOT_SET"),
            "NEO4J_USER": os.getenv("NEO4J_USER", "NOT_SET"),
            "NEO4J_PASSWORD": "SET" if os.getenv("NEO4J_PASSWORD") else "MISSING",
            "MONGODB_HOST": os.getenv("MONGODB_HOST", "NOT_SET"),
            "MONGODB_PORT": os.getenv("MONGODB_PORT", "NOT_SET"),
            "MONGODB_USER": os.getenv("MONGODB_USER", "NOT_SET"),
            "MONGODB_PASSWORD": "SET" if os.getenv("MONGODB_PASSWORD") else "MISSING",
            "MONGODB_DB": os.getenv("MONGODB_DB", "NOT_SET"),
            "CHROMA_HOST": os.getenv("CHROMA_HOST", "NOT_SET"),
            "CHROMA_PORT": os.getenv("CHROMA_PORT", "NOT_SET")
        }

        # File structure analysis
        important_files = {}
        for file_path in [
            ".env",
            "docker-compose.yml",
            "src/databases/graph/config.py",
            "src/databases/document/config.py",
            "src/databases/vector/config.py",
            "src/services/graph_service.py",
            "src/services/document_service.py",
            "src/services/vector_service.py"
        ]:
            full_path = project_root / file_path
            important_files[file_path] = {
                "exists": full_path.exists(),
                "size": full_path.stat().st_size if full_path.exists() else 0,
                "modified": datetime.fromtimestamp(
                    full_path.stat().st_mtime).isoformat() if full_path.exists() else None
            }

        self.findings["environment"] = {
            "python_info": python_info,
            "env_vars": env_vars,
            "file_structure": important_files
        }

        print(f"   Python: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   Working dir: {Path.cwd()}")
        print(f"   Missing env vars: {[k for k, v in env_vars.items() if v in ['MISSING', 'NOT_SET']]}")
        print(f"   Missing files: {[k for k, v in important_files.items() if not v['exists']]}")

    def trace_database_connections(self):
        """Trace database connection attempts with detailed logging"""
        print("\n🔌 Database Connection Tracing...")

        # Test Neo4j with detailed tracing
        print("   🔗 Neo4j Connection Trace:")
        neo4j_trace = self._trace_neo4j_connection()

        # Test MongoDB with detailed tracing
        print("   📊 MongoDB Connection Trace:")
        mongodb_trace = self._trace_mongodb_connection()

        # Test ChromaDB with detailed tracing
        print("   🔍 ChromaDB Connection Trace:")
        chromadb_trace = self._trace_chromadb_connection()

        self.findings["database_connectivity"] = {
            "neo4j": neo4j_trace,
            "mongodb": mongodb_trace,
            "chromadb": chromadb_trace
        }

    def _trace_neo4j_connection(self):
        """Detailed Neo4j connection tracing"""
        trace = {"steps": [], "success": False, "final_error": None}

        try:
            trace["steps"].append("Importing Neo4j config...")
            from src.databases.graph.config import get_neo4j_driver, neo4j_config
            print("      ✅ Config import successful")

            trace["steps"].append("Reading configuration...")
            config_info = {
                "uri": neo4j_config.uri,
                "user": neo4j_config.user,
                "database": neo4j_config.database,
                "password_set": bool(neo4j_config.password)
            }
            trace["config"] = config_info
            print(f"      📋 Config: {neo4j_config.uri} as {neo4j_config.user}")

            trace["steps"].append("Creating driver...")
            driver = get_neo4j_driver()
            print("      ✅ Driver created")

            trace["steps"].append("Verifying connectivity...")
            driver.verify_connectivity()
            print("      ✅ Connectivity verified")

            trace["steps"].append("Testing session...")
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                trace["test_query_result"] = test_value
            print(f"      ✅ Session test successful: {test_value}")

            trace["steps"].append("Gathering database info...")
            with driver.session() as session:
                # Get node count
                nodes_result = session.run("MATCH (n) RETURN count(n) as count")
                nodes_count = nodes_result.single()["count"]

                # Get labels
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]

                # Get relationship types
                rels_result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in rels_result]

                database_info = {
                    "nodes_count": nodes_count,
                    "labels": labels,
                    "relationship_types": rel_types
                }
                trace["database_info"] = database_info
                print(f"      📊 Database: {nodes_count} nodes, {len(labels)} labels")

            trace["success"] = True

        except Exception as e:
            trace["final_error"] = str(e)
            trace["error_traceback"] = traceback.format_exc()
            print(f"      ❌ Failed: {str(e)}")

        return trace

    def _trace_mongodb_connection(self):
        """Detailed MongoDB connection tracing"""
        trace = {"steps": [], "success": False, "final_error": None}

        try:
            trace["steps"].append("Importing MongoDB config...")
            from src.databases.document.config import get_mongodb_client, mongo_db_config
            print("      ✅ Config import successful")

            trace["steps"].append("Reading configuration...")
            config_info = {
                "host": mongo_db_config.host,
                "port": mongo_db_config.port,
                "user": mongo_db_config.user,
                "database": mongo_db_config.database,
                "password_set": bool(mongo_db_config.password)
            }
            trace["config"] = config_info
            print(f"      📋 Config: {mongo_db_config.host}:{mongo_db_config.port}/{mongo_db_config.database}")

            trace["steps"].append("Creating client...")
            client = get_mongodb_client()
            print("      ✅ Client created")

            trace["steps"].append("Testing server info...")
            server_info = client.server_info()
            trace["server_info"] = {
                "version": server_info.get("version"),
                "ok": server_info.get("ok")
            }
            print(f"      ✅ Server info: MongoDB {server_info.get('version')}")

            trace["steps"].append("Accessing database...")
            db = client[mongo_db_config.database]

            trace["steps"].append("Listing collections...")
            collections = db.list_collection_names()
            trace["collections"] = collections
            print(f"      📁 Collections: {collections}")

            # Check collection contents
            collection_stats = {}
            for collection_name in collections:
                collection = db[collection_name]
                count = collection.count_documents({})
                collection_stats[collection_name] = count

            trace["collection_stats"] = collection_stats
            print(f"      📊 Collection stats: {collection_stats}")

            trace["success"] = True

        except Exception as e:
            trace["final_error"] = str(e)
            trace["error_traceback"] = traceback.format_exc()
            print(f"      ❌ Failed: {str(e)}")

        return trace

    def _trace_chromadb_connection(self):
        """Detailed ChromaDB connection tracing"""
        trace = {"steps": [], "success": False, "final_error": None}

        try:
            trace["steps"].append("Importing ChromaDB config...")
            from src.databases.vector.config import ChromaDBConfig
            print("      ✅ Config import successful")

            trace["steps"].append("Reading configuration...")
            config = ChromaDBConfig()
            config_info = {
                "host": config.host,
                "port": config.port
            }
            trace["config"] = config_info
            print(f"      📋 Config: {config.host}:{config.port}")

            trace["steps"].append("Creating client...")
            client = config.get_client()
            print("      ✅ Client created")

            trace["steps"].append("Testing heartbeat...")
            heartbeat = client.heartbeat()
            trace["heartbeat"] = heartbeat
            print(f"      💓 Heartbeat: {heartbeat}")

            trace["steps"].append("Listing collections...")
            collections = client.list_collections()
            trace["collections"] = [str(c) for c in collections] if collections else []
            print(f"      📚 Collections: {len(collections) if collections else 0}")

            # Check for academic_papers collection specifically
            trace["steps"].append("Checking academic_papers collection...")
            try:
                collection = client.get_collection("academic_papers")
                count = collection.count()
                trace["academic_papers_count"] = count
                print(f"      📄 Academic papers: {count} vectors")

                if count > 0:
                    # Get a sample
                    sample = collection.peek(limit=1)
                    trace["sample_data"] = {
                        "has_documents": bool(sample.get("documents")),
                        "has_metadatas": bool(sample.get("metadatas")),
                        "has_ids": bool(sample.get("ids"))
                    }

            except Exception as collection_error:
                trace["academic_papers_error"] = str(collection_error)
                print(f"      ❌ Academic papers collection: {str(collection_error)}")

            trace["success"] = True

        except Exception as e:
            trace["final_error"] = str(e)
            trace["error_traceback"] = traceback.format_exc()
            print(f"      ❌ Failed: {str(e)}")

        return trace

    def analyze_service_functions(self):
        """Analyze the service function behavior in detail"""
        print("\n🛠️ Service Function Analysis...")

        test_queries = ["machine learning", "deep learning", "test"]
        service_analysis = {}

        for query in test_queries:
            print(f"\n   Testing with query: '{query}'")
            query_results = {}

            # Test graph service
            print("      🔗 Testing graph_service...")
            try:
                from src.services.graph_service import query_graphdb
                graph_start = datetime.now()
                graph_result = query_graphdb(query)
                graph_duration = (datetime.now() - graph_start).total_seconds()

                query_results["graph_service"] = {
                    "success": True,
                    "duration_seconds": graph_duration,
                    "result_structure": {
                        "concepts_count": len(graph_result.get("concepts", [])),
                        "relationships_count": len(graph_result.get("relationships", [])),
                        "papers_count": len(graph_result.get("papers", [])),
                        "keys": list(graph_result.keys())
                    },
                    "sample_concepts": graph_result.get("concepts", [])[:2],
                    "sample_relationships": graph_result.get("relationships", [])[:2]
                }
                print(f"         ✅ Success ({graph_duration:.2f}s): {len(graph_result.get('concepts', []))} concepts")

            except Exception as e:
                query_results["graph_service"] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"         ❌ Error: {str(e)}")

            # Test document service
            print("      📊 Testing document_service...")
            try:
                from src.services.document_service import query_mongodb
                doc_start = datetime.now()
                doc_result = query_mongodb(query)
                doc_duration = (datetime.now() - doc_start).total_seconds()

                query_results["document_service"] = {
                    "success": True,
                    "duration_seconds": doc_duration,
                    "result_structure": {
                        "topics_count": len(doc_result.get("topics", {})),
                        "papers_count": len(doc_result.get("papers", [])),
                        "keys": list(doc_result.keys())
                    },
                    "sample_topics": dict(list(doc_result.get("topics", {}).items())[:2]),
                    "sample_papers": doc_result.get("papers", [])[:2]
                }
                print(f"         ✅ Success ({doc_duration:.2f}s): {len(doc_result.get('topics', {}))} topic categories")

            except Exception as e:
                query_results["document_service"] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"         ❌ Error: {str(e)}")

            # Test vector service
            print("      🔍 Testing vector_service...")
            try:
                from src.services.vector_service import query_vectordb
                vector_start = datetime.now()
                vector_result = query_vectordb(query, limit=3)
                vector_duration = (datetime.now() - vector_start).total_seconds()

                query_results["vector_service"] = {
                    "success": True,
                    "duration_seconds": vector_duration,
                    "result_structure": {
                        "documents_count": len(vector_result),
                        "sample_structure": vector_result[0] if vector_result else None
                    },
                    "sample_results": vector_result[:2]
                }
                print(f"         ✅ Success ({vector_duration:.2f}s): {len(vector_result)} documents")

            except Exception as e:
                query_results["vector_service"] = {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"         ❌ Error: {str(e)}")

            service_analysis[query] = query_results

        self.findings["query_tracing"] = service_analysis

    def analyze_agent_behavior(self):
        """Deep analysis of agent wrapper behavior"""
        print("\n🤖 Agent Behavior Analysis...")

        test_input = {"messages": [{"role": "user", "content": "deep learning natural language processing"}]}

        # Test graph writer agent
        print("   🔗 Analyzing graph_writer_agent...")
        try:
            from src.domain.agents.graph_writer import graph_writer_agent

            graph_start = datetime.now()
            graph_response = graph_writer_agent.invoke(test_input)
            graph_duration = (datetime.now() - graph_start).total_seconds()

            graph_analysis = {
                "success": True,
                "duration_seconds": graph_duration,
                "response_type": type(graph_response).__name__,
                "response_keys": list(graph_response.keys()) if isinstance(graph_response, dict) else None,
                "response_structure": {
                    "has_output": "output" in graph_response if isinstance(graph_response, dict) else False,
                    "output_type": type(graph_response.get("output")).__name__ if isinstance(graph_response,
                                                                                             dict) and "output" in graph_response else None,
                    "output_length": len(str(graph_response.get("output", ""))) if isinstance(graph_response,
                                                                                              dict) else len(
                        str(graph_response))
                },
                "response_preview": str(graph_response)[:300] + "..." if len(str(graph_response)) > 300 else str(
                    graph_response)
            }

            print(f"      ✅ Response in {graph_duration:.2f}s, type: {type(graph_response).__name__}")
            print(f"      📝 Preview: {str(graph_response)[:100]}...")

        except Exception as e:
            graph_analysis = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"      ❌ Error: {str(e)}")

        # Test topic model agent
        print("   📊 Analyzing topic_model_agent...")
        try:
            from src.domain.agents.topic_model import topic_model_agent

            topic_start = datetime.now()
            topic_response = topic_model_agent.invoke(test_input)
            topic_duration = (datetime.now() - topic_start).total_seconds()

            topic_analysis = {
                "success": True,
                "duration_seconds": topic_duration,
                "response_type": type(topic_response).__name__,
                "response_keys": list(topic_response.keys()) if isinstance(topic_response, dict) else None,
                "response_structure": {
                    "has_output": "output" in topic_response if isinstance(topic_response, dict) else False,
                    "output_type": type(topic_response.get("output")).__name__ if isinstance(topic_response,
                                                                                             dict) and "output" in topic_response else None,
                    "output_length": len(str(topic_response.get("output", ""))) if isinstance(topic_response,
                                                                                              dict) else len(
                        str(topic_response))
                },
                "response_preview": str(topic_response)[:300] + "..." if len(str(topic_response)) > 300 else str(
                    topic_response)
            }

            print(f"      ✅ Response in {topic_duration:.2f}s, type: {type(topic_response).__name__}")
            print(f"      📝 Preview: {str(topic_response)[:100]}...")

        except Exception as e:
            topic_analysis = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"      ❌ Error: {str(e)}")

        # Test supervisor
        print("   🎭 Analyzing supervisor workflow...")
        try:
            from src.domain.agents.supervisor import run_supervisor

            supervisor_start = datetime.now()
            supervisor_response = run_supervisor("deep learning natural language processing")
            supervisor_duration = (datetime.now() - supervisor_start).total_seconds()

            supervisor_analysis = {
                "success": True,
                "duration_seconds": supervisor_duration,
                "response_type": type(supervisor_response).__name__,
                "response_keys": list(supervisor_response.keys()) if isinstance(supervisor_response, dict) else None,
                "has_graph_output": bool(supervisor_response.get("graph_output")) if isinstance(supervisor_response,
                                                                                                dict) else False,
                "has_tm_output": bool(supervisor_response.get("tm_output")) if isinstance(supervisor_response,
                                                                                          dict) else False,
                "has_final_output": bool(supervisor_response.get("final_output")) if isinstance(supervisor_response,
                                                                                                dict) else False,
                "graph_output_preview": str(supervisor_response.get("graph_output", ""))[:200] + "..." if isinstance(
                    supervisor_response, dict) else "",
                "tm_output_preview": str(supervisor_response.get("tm_output", ""))[:200] + "..." if isinstance(
                    supervisor_response, dict) else "",
                "final_output_preview": str(supervisor_response.get("final_output", ""))[:200] + "..." if isinstance(
                    supervisor_response, dict) else ""
            }

            print(f"      ✅ Supervisor completed in {supervisor_duration:.2f}s")
            print(
                f"      📊 Outputs: Graph={bool(supervisor_response.get('graph_output'))}, Topic={bool(supervisor_response.get('tm_output'))}, Final={bool(supervisor_response.get('final_output'))}")

        except Exception as e:
            supervisor_analysis = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(f"      ❌ Error: {str(e)}")

        self.findings["agent_behavior"] = {
            "graph_writer_agent": graph_analysis,
            "topic_model_agent": topic_analysis,
            "supervisor": supervisor_analysis
        }

    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "=" * 60)
        print("🔬 COMPREHENSIVE DIAGNOSTIC REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Environment summary
        env = self.findings["environment"]
        missing_env = [k for k, v in env["env_vars"].items() if v in ["MISSING", "NOT_SET"]]
        missing_files = [k for k, v in env["file_structure"].items() if not v["exists"]]

        print(f"\n🔧 ENVIRONMENT STATUS")
        print(f"   Python: {env['python_info']['python_version'].split()[0]}")
        print(f"   Working Directory: {env['python_info']['working_directory']}")
        print(
            f"   Missing Env Vars: {len(missing_env)} ({', '.join(missing_env[:3])}{'...' if len(missing_env) > 3 else ''})")
        print(
            f"   Missing Files: {len(missing_files)} ({', '.join(missing_files[:3])}{'...' if len(missing_files) > 3 else ''})")

        # Database connectivity summary
        db_connectivity = self.findings["database_connectivity"]
        print(f"\n🔌 DATABASE CONNECTIVITY")
        for db_name, trace in db_connectivity.items():
            status = "✅ SUCCESS" if trace["success"] else "❌ FAILED"
            error_info = f" ({trace['final_error']})" if not trace["success"] and trace.get("final_error") else ""
            print(f"   {db_name.upper()}: {status}{error_info}")

            if trace["success"] and "database_info" in trace:
                db_info = trace["database_info"]
                if db_name == "neo4j":
                    print(f"      Data: {db_info.get('nodes_count', 0)} nodes, {len(db_info.get('labels', []))} types")
                elif db_name == "mongodb":
                    collections = trace.get("collections", [])
                    stats = trace.get("collection_stats", {})
                    print(f"      Data: {len(collections)} collections, {sum(stats.values())} documents")
                elif db_name == "chromadb":
                    vectors = trace.get("academic_papers_count", 0)
                    print(f"      Data: {vectors} vectors")

        # Service function analysis
        query_tracing = self.findings.get("query_tracing", {})
        if query_tracing:
            print(f"\n🛠️ SERVICE FUNCTION ANALYSIS")

            # Analyze across all test queries
            all_graph_success = all(
                results.get("graph_service", {}).get("success", False) for results in query_tracing.values())
            all_doc_success = all(
                results.get("document_service", {}).get("success", False) for results in query_tracing.values())
            all_vector_success = all(
                results.get("vector_service", {}).get("success", False) for results in query_tracing.values())

            print(f"   Graph Service: {'✅ Working' if all_graph_success else '❌ Issues'}")
            print(f"   Document Service: {'✅ Working' if all_doc_success else '❌ Issues'}")
            print(f"   Vector Service: {'✅ Working' if all_vector_success else '❌ Issues'}")

            # Show data availability
            for query, results in query_tracing.items():
                if results.get("graph_service", {}).get("success"):
                    graph_data = results["graph_service"]["result_structure"]
                    has_graph_data = graph_data["concepts_count"] > 0 or graph_data["relationships_count"] > 0

                if results.get("document_service", {}).get("success"):
                    doc_data = results["document_service"]["result_structure"]
                    has_doc_data = doc_data["topics_count"] > 0 or doc_data["papers_count"] > 0

                if results.get("vector_service", {}).get("success"):
                    vector_data = results["vector_service"]["result_structure"]
                    has_vector_data = vector_data["documents_count"] > 0

                break  # Just check first query

            print(
                f"   Data Available: Graph={'✅' if has_graph_data else '📭'}, Document={'✅' if has_doc_data else '📭'}, Vector={'✅' if has_vector_data else '📭'}")

        # Agent behavior analysis
        agent_behavior = self.findings.get("agent_behavior", {})
        if agent_behavior:
            print(f"\n🤖 AGENT BEHAVIOR ANALYSIS")

            for agent_name, analysis in agent_behavior.items():
                status = "✅ Working" if analysis.get("success") else "❌ Failed"
                duration = f" ({analysis.get('duration_seconds', 0):.1f}s)" if analysis.get("success") else ""
                print(f"   {agent_name}: {status}{duration}")

                if analysis.get("success"):
                    # Check if agent is using actual database data vs generic responses
                    response_preview = analysis.get("response_preview", "").lower()
                    uses_real_data = any(indicator in response_preview for indicator in [
                        "database", "neo4j", "mongodb", "chromadb", "found", "retrieved", "query"
                    ])
                    data_status = "🔍 Using DB" if uses_real_data else "🤖 Generic AI"
                    print(f"      Response Type: {data_status}")

        # Key insights and next steps
        print(f"\n🎯 KEY INSIGHTS")

        # Determine primary issues
        db_connected = sum(1 for trace in db_connectivity.values() if trace["success"])
        services_working = all_graph_success and all_doc_success and all_vector_success if query_tracing else False
        agents_working = all(
            analysis.get("success", False) for analysis in agent_behavior.values()) if agent_behavior else False

        if db_connected == 0:
            print("   🚨 PRIMARY ISSUE: No database connections")
            print("      • Check if Docker services are running")
            print("      • Verify environment variables")
            print("      • Check network connectivity")

        elif db_connected < 3:
            failed_dbs = [name for name, trace in db_connectivity.items() if not trace["success"]]
            print(f"   ⚠️ PARTIAL CONNECTIVITY: {failed_dbs} not connected")

        elif not services_working:
            print("   🛠️ DATABASE CONNECTED BUT SERVICE FUNCTIONS FAILING")
            print("      • Check service function implementations")
            print("      • Verify database schemas/collections")

        elif not agents_working:
            print("   🤖 SERVICES WORK BUT AGENTS FAILING")
            print("      • Check agent wrapper implementations")
            print("      • Verify LangGraph integration")

        elif not has_graph_data and not has_doc_data and not has_vector_data:
            print("   📭 EVERYTHING CONNECTED BUT NO DATA")
            print("      • Run ingestion pipeline to populate databases")
            print("      • Add sample data for testing")

        else:
            print("   ✅ SYSTEM APPEARS FUNCTIONAL")
            print("      • All components working")
            print("      • Data available in databases")
            print("      • Agents responding appropriately")

        # Save detailed report
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(self.findings, f, indent=2, default=str)
            print(f"\n💾 Detailed findings saved to: {report_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save detailed report: {e}")

        print("=" * 60)

        return self.findings


def main():
    """Run comprehensive diagnostic analysis"""
    diagnostic = DatabaseDiagnostic()

    # Run all diagnostic steps
    diagnostic.analyze_environment()
    diagnostic.trace_database_connections()
    diagnostic.analyze_service_functions()
    diagnostic.analyze_agent_behavior()

    # Generate final report
    findings = diagnostic.generate_diagnostic_report()

    # Determine overall system health
    db_connectivity = findings["database_connectivity"]
    connected_dbs = sum(1 for trace in db_connectivity.values() if trace["success"])

    if connected_dbs == 3:
        return 0  # All good
    elif connected_dbs > 0:
        return 1  # Partial issues
    else:
        return 2  # Major problems


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Diagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during diagnostic: {e}")
        traceback.print_exc()
        sys.exit(1)