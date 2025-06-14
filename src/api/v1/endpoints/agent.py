# Update your src/api/v1/endpoints/agent.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.supervisor import run_supervisor
import re

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def extract_clean_output(raw_output: str) -> str:
    """Extract clean, readable content from raw agent output"""
    if not raw_output:
        return "No output available"

    # Convert to string if it's not already
    output_str = str(raw_output)

    # If it contains LangGraph messages, extract the actual content
    if "AIMessage(content=" in output_str:
        # Find the last AIMessage content
        pattern = r"AIMessage\(content='([^']*(?:\\.[^']*)*)', additional_kwargs"
        matches = re.findall(pattern, output_str, re.DOTALL)
        if matches:
            content = matches[-1]
            # Clean up escaped characters
            content = content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            return content

    # If it contains 'messages': [HumanMessage... extract differently
    if "'messages':" in output_str and "HumanMessage" in output_str:
        # Try to find content after analysis sections
        if "## 🔗" in output_str:
            start_idx = output_str.find("## 🔗")
            if start_idx != -1:
                # Extract from graph analysis onwards
                content = output_str[start_idx:start_idx + 1000]  # Get reasonable chunk
                # Clean up any remaining message artifacts
                content = re.sub(r"'messages':\s*\[.*?\]", "", content)
                content = re.sub(r"HumanMessage\([^)]+\)", "", content)
                content = re.sub(r"additional_kwargs=\{[^}]*\}", "", content)
                content = re.sub(r"response_metadata=\{[^}]*\}", "", content)
                content = re.sub(r"id=[a-f0-9-]+", "", content)
                return content.strip()

    # If it's already clean content, return as is
    if any(output_str.strip().startswith(marker) for marker in ["##", "**", "- ", "🔗", "📊", "✨"]):
        return output_str.strip()

    # Last resort: try to extract meaningful content
    lines = output_str.split('\n')
    clean_lines = []
    for line in lines:
        # Skip lines with message artifacts
        if not any(artifact in line for artifact in
                   ['HumanMessage', 'additional_kwargs', 'response_metadata', "'messages':"]):
            clean_lines.append(line)

    result = '\n'.join(clean_lines).strip()
    return result if result else "Unable to extract clean content from agent response"


@router.post("/agent")
def process_query(request: QueryRequest):
    """
    API endpoint to process a user query using the supervisor agent.
    Returns clean, formatted output with better response extraction.
    """
    try:
        # Run the supervisor workflow
        response = run_supervisor(request.query)

        # Extract clean outputs
        graph_output = response.get("graph_output", "")
        tm_output = response.get("tm_output", "")
        final_output = response.get("final_output", "")

        # Clean the outputs using improved extraction
        clean_graph = extract_clean_output(str(graph_output))
        clean_topic = extract_clean_output(str(tm_output))
        clean_final = extract_clean_output(str(final_output))

        # Check if we got actual database content vs generic responses
        def has_database_content(content):
            db_indicators = [
                'database', 'neo4j', 'mongodb', 'chromadb',
                'concepts found', 'relationships found', 'topics found',
                'papers found', 'retrieved', 'vectors', 'nodes'
            ]
            return any(indicator.lower() in content.lower() for indicator in db_indicators)

        # Create formatted response with better structure
        formatted_message = f"""# 🎯 Multi-Agent Research Analysis

**Query:** {request.query}

---

## 🔗 Knowledge Graph Analysis
{clean_graph}

---

## 📊 Topic Modeling Analysis  
{clean_topic}

---

## ✨ Synthesized Insights
{clean_final}

---

**Analysis Quality:** {'✅ Database-Driven' if any(has_database_content(content) for content in [clean_graph, clean_topic, clean_final]) else '⚠️ Generic Response'}"""

        return {
            "status": "success",
            "message": formatted_message,
            "query": request.query,
            "agents_used": {
                "graph_writer": bool(clean_graph and clean_graph != "No output available" and len(clean_graph) > 50),
                "topic_model": bool(clean_topic and clean_topic != "No output available" and len(clean_topic) > 50),
                "supervisor": True
            },
            "database_indicators": {
                "graph_has_db_content": has_database_content(clean_graph),
                "topic_has_db_content": has_database_content(clean_topic),
                "final_has_db_content": has_database_content(clean_final)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/raw")
def process_query_raw(request: QueryRequest):
    """
    Raw output endpoint for debugging
    """
    try:
        response = run_supervisor(request.query)
        return {
            "status": "success",
            "message": str(response.get("final_output", "")),
            "debug": {
                "graph_output": str(response.get("graph_output", "")),
                "tm_output": str(response.get("tm_output", ""))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))