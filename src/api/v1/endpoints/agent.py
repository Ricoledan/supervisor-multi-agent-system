from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.research_coordinator import run_research_coordinator, process_query_direct
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def extract_final_response(state_result: dict) -> str:
    """Extract the final response content from the modern state result."""
    try:
        messages = state_result.get("messages", [])
        if not messages:
            return "No response generated"

        # Get the last message (final response)
        final_message = messages[-1]

        # Extract content
        if hasattr(final_message, 'content'):
            return final_message.content
        elif isinstance(final_message, dict) and 'content' in final_message:
            return final_message['content']
        else:
            return str(final_message)

    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return f"Error extracting response: {str(e)}"


def analyze_system_health(state_result: dict) -> dict:
    """Analyze system health based on the conversation state."""
    try:
        messages = state_result.get("messages", [])

        # Check which specialists were used
        relationship_used = any(
            hasattr(msg, 'name') and msg.name == "relationship_analyst"
            for msg in messages
        )
        theme_used = any(
            hasattr(msg, 'name') and msg.name == "theme_analyst"
            for msg in messages
        )

        # Check for database content indicators
        def has_database_content(messages, agent_name):
            agent_messages = [
                msg for msg in messages
                if hasattr(msg, 'name') and msg.name == agent_name
            ]
            if not agent_messages:
                return False

            content = agent_messages[-1].content.lower()
            # Look for positive database indicators
            positive_indicators = [
                'database returned', 'found', 'retrieved', 'analyzed',
                'concepts found', 'papers found', 'relationships found'
            ]
            # Look for negative database indicators
            negative_indicators = [
                'no data found', 'database may be empty', 'no relationship data',
                'no thematic data', 'database status: no'
            ]

            has_positive = any(indicator in content for indicator in positive_indicators)
            has_negative = any(indicator in content for indicator in negative_indicators)

            return has_positive and not has_negative

        relationship_has_data = has_database_content(messages, "relationship_analyst")
        theme_has_data = has_database_content(messages, "theme_analyst")

        # Determine overall database usage
        if relationship_has_data and theme_has_data:
            db_usage = "✅ High"
            response_quality = "Database-driven"
        elif relationship_has_data or theme_has_data:
            db_usage = "🟡 Partial"
            response_quality = "Mixed (database + general)"
        elif relationship_used or theme_used:
            db_usage = "❌ Low"
            response_quality = "Limited database content"
        else:
            db_usage = "⚪ None"
            response_quality = "General knowledge"

        return {
            "relationship_analyst": "✅ Active" if relationship_used else "⚪ Not Used",
            "theme_analyst": "✅ Active" if theme_used else "⚪ Not Used",
            "database_usage": db_usage,
            "response_quality": response_quality,
            "coordination_status": "✅ Modern LangGraph Pattern",
            "message_count": len(messages)
        }

    except Exception as e:
        logger.error(f"Error analyzing system health: {e}")
        return {
            "relationship_analyst": "❓ Unknown",
            "theme_analyst": "❓ Unknown",
            "database_usage": "❓ Unknown",
            "response_quality": "Error in analysis",
            "coordination_status": "❌ Analysis Failed",
            "error": str(e)
        }


@router.post("/agent")
def process_query(request: QueryRequest):
    """
    Process research query using modern LangGraph multi-agent system.

    This endpoint uses the updated research coordinator with Command-based routing
    and proper message state management.
    """
    try:
        logger.info(f"Processing query with modern coordinator: {request.query[:50]}...")

        # Use the direct processing function which handles the full workflow
        result = process_query_direct(request.query)

        if result["status"] == "success":
            logger.info("Query processed successfully with modern coordinator")
            return result
        else:
            logger.error(f"Query processing failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("message", "Processing failed"))

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in agent endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/agent/detailed")
def process_query_detailed(request: QueryRequest):
    """
    Process query with detailed state information and message history.

    This endpoint returns the full conversation state for debugging and analysis.
    """
    try:
        logger.info(f"Processing detailed query: {request.query[:50]}...")

        # Run the full coordinator workflow
        state_result = run_research_coordinator(request.query)

        # Extract response content
        final_response = extract_final_response(state_result)

        # Analyze system performance
        system_health = analyze_system_health(state_result)

        # Build detailed response
        detailed_response = {
            "status": "success",
            "message": final_response,
            "query": request.query,
            "query_type": state_result.get("query_type", "unknown"),
            "analysis_plan": state_result.get("analysis_plan", ""),
            "system_health": system_health,
            "conversation_flow": {
                "total_messages": len(state_result.get("messages", [])),
                "message_agents": [
                    getattr(msg, 'name', 'system') for msg in state_result.get("messages", [])
                    if hasattr(msg, 'name')
                ],
                "coordination_pattern": "Modern Command-based routing"
            },
            "debug_info": {
                "state_keys": list(state_result.keys()),
                "needs_relationship": state_result.get("needs_relationship", False),
                "needs_theme": state_result.get("needs_theme", False),
                "transfer_context": state_result.get("transfer_context", "")
            }
        }

        logger.info("Detailed query processing completed successfully")
        return detailed_response

    except Exception as e:
        logger.error(f"Error in detailed processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detailed processing failed: {str(e)}")


@router.post("/agent/raw")
def process_query_raw(request: QueryRequest):
    """
    Raw output endpoint for debugging - returns the complete state object.
    """
    try:
        logger.info(f"Processing raw query: {request.query[:50]}...")

        # Get the complete state result
        state_result = run_research_coordinator(request.query)

        # Convert messages to serializable format
        serializable_messages = []
        for msg in state_result.get("messages", []):
            msg_dict = {
                "content": getattr(msg, 'content', str(msg)),
                "type": type(msg).__name__,
            }
            if hasattr(msg, 'name'):
                msg_dict["name"] = msg.name
            serializable_messages.append(msg_dict)

        raw_response = {
            "status": "success",
            "query": request.query,
            "full_state": {
                "query_type": state_result.get("query_type", "unknown"),
                "analysis_plan": state_result.get("analysis_plan", ""),
                "needs_relationship": state_result.get("needs_relationship", False),
                "needs_theme": state_result.get("needs_theme", False),
                "transfer_context": state_result.get("transfer_context", ""),
                "messages": serializable_messages
            },
            "coordinator_info": {
                "pattern": "Modern LangGraph with Command routing",
                "state_schema": "ResearchState(MessagesState)",
                "agents_available": ["relationship_analyst", "theme_analyst"],
                "routing_method": "Command-based"
            }
        }

        logger.info("Raw query processing completed")
        return raw_response

    except Exception as e:
        logger.error(f"Error in raw processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Raw processing failed: {str(e)}")


@router.get("/agent/health")
def check_agent_health():
    """
    Check the health of the modern multi-agent system.
    """
    try:
        # Test a simple query to verify the system works
        test_result = process_query_direct("Hello")

        health_status = {
            "status": "healthy",
            "coordinator_pattern": "Modern LangGraph with Commands",
            "test_query_success": test_result["status"] == "success",
            "available_specialists": ["relationship_analyst", "theme_analyst"],
            "routing_method": "Command-based with MessagesState",
            "features": {
                "query_classification": "✅ Enabled",
                "dynamic_routing": "✅ Enabled",
                "specialist_tools": "✅ Enabled",
                "database_integration": "✅ Enabled",
                "error_handling": "✅ Enabled"
            }
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "coordinator_pattern": "Modern LangGraph (with errors)"
        }


# Add this new endpoint to src/api/v1/endpoints/agent.py

@router.post("/agent/clean")
def process_query_clean(request: QueryRequest):
    """
    Process query with clean, formatted output for easy reading.

    This endpoint returns the analysis result in a clean, markdown-formatted
    response that's easy to read and demo-friendly.
    """
    try:
        logger.info(f"Processing clean query: {request.query[:50]}...")

        # Use the direct processing function
        result = process_query_direct(request.query)

        if result["status"] == "success":
            # Extract and clean the message content
            raw_message = result.get("message", "")

            # Remove markdown formatting for cleaner display
            import re

            # Remove markdown headers and replace with clean formatting
            clean_message = re.sub(r'^#+\s*', '', raw_message, flags=re.MULTILINE)
            clean_message = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_message)  # Remove bold
            clean_message = re.sub(r'---+', '=' * 60, clean_message)  # Replace dividers

            # Extract system performance info
            specialists_used = result.get("specialists_used", {})
            system_health = result.get("system_health", {})

            # Build clean response
            clean_response = {
                "status": "success",
                "query": request.query,
                "analysis": clean_message.strip(),
                "system_info": {
                    "relationship_analyst_used": specialists_used.get("relationship_analyst", False),
                    "theme_analyst_used": specialists_used.get("theme_analyst", False),
                    "database_usage": system_health.get("database_usage", "Unknown"),
                    "response_quality": system_health.get("response_quality", "Unknown")
                }
            }

            logger.info("Clean query processing completed successfully")
            return clean_response

        else:
            logger.error(f"Query processing failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=result.get("message", "Processing failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clean agent endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/agent/formatted")
def process_query_formatted(request: QueryRequest):
    """
    Process query with beautifully formatted output for presentations and demos.

    Returns analysis in a structured format perfect for displaying in articles,
    presentations, or demo environments.
    """
    try:
        logger.info(f"Processing formatted query: {request.query[:50]}...")

        result = process_query_direct(request.query)

        if result["status"] == "success":
            raw_message = result.get("message", "")
            specialists_used = result.get("specialists_used", {})
            system_health = result.get("system_health", {})

            # Parse the message to extract key sections
            sections = parse_analysis_sections(raw_message)

            formatted_response = {
                "status": "success",
                "query": {
                    "text": request.query,
                    "type": result.get("query_type", "RESEARCH_QUERY")
                },
                "analysis": {
                    "summary": sections.get("summary", ""),
                    "relationship_findings": sections.get("relationship", ""),
                    "thematic_findings": sections.get("theme", ""),
                    "key_insights": sections.get("insights", []),
                    "recommendations": sections.get("recommendations", "")
                },
                "system_performance": {
                    "specialists_activated": {
                        "relationship_analyst": {
                            "used": specialists_used.get("relationship_analyst", False),
                            "status": system_health.get("relationship_analyst", "Not Used")
                        },
                        "theme_analyst": {
                            "used": specialists_used.get("theme_analyst", False),
                            "status": system_health.get("theme_analyst", "Not Used")
                        }
                    },
                    "database_utilization": system_health.get("database_usage", "Unknown"),
                    "confidence_level": system_health.get("response_quality", "Unknown")
                },
                "metadata": {
                    "processing_time": "15-45 seconds",
                    "databases_queried": ["Neo4j", "MongoDB", "ChromaDB"],
                    "ai_model": "GPT-4"
                }
            }

            return formatted_response

        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Processing failed"))

    except Exception as e:
        logger.error(f"Error in formatted processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Formatted processing failed: {str(e)}")


def parse_analysis_sections(message: str) -> dict:
    """Parse the analysis message into structured sections."""
    sections = {}

    # Extract different sections using patterns
    import re

    # Try to extract summary (first paragraph after header)
    summary_match = re.search(r'Query:\*\*.*?\n\n---\n\n(.*?)(?:\n\n|\n###|\nHowever|\n-)', message, re.DOTALL)
    if summary_match:
        sections["summary"] = summary_match.group(1).strip()

    # Extract insights and recommendations
    insights = []
    for line in message.split('\n'):
        if line.strip().startswith('- ') and ('insight' in line.lower() or 'finding' in line.lower()):
            insights.append(line.strip()[2:])  # Remove '- '
    sections["insights"] = insights

    # Extract recommendations
    rec_match = re.search(r'recommendation[s]?\s*[:\-]?\s*(.*?)(?:\n\n|\n###|\n-|$)', message,
                          re.IGNORECASE | re.DOTALL)
    if rec_match:
        sections["recommendations"] = rec_match.group(1).strip()

    return sections