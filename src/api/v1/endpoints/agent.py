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

    # If it contains LangGraph messages, extract the actual content
    if "AIMessage(content=" in raw_output:
        # Find the last AIMessage content
        pattern = r"AIMessage\(content='([^']*(?:##[^']*)*)', additional_kwargs"
        matches = re.findall(pattern, raw_output, re.DOTALL)
        if matches:
            content = matches[-1]
            # Clean up escaped characters
            content = content.replace('\\n', '\n').replace('\\"', '"')
            return content

    # If it's already clean content, return as is
    if any(raw_output.startswith(marker) for marker in ["##", "**", "- ", "🔗", "🎯"]):
        return raw_output

    return raw_output


@router.post("/agent")
def process_query(request: QueryRequest):
    """
    API endpoint to process a user query using the supervisor agent.
    Returns clean, formatted output.
    """
    try:
        # Run the supervisor workflow
        response = run_supervisor(request.query)

        # Extract clean outputs
        graph_output = response.get("graph_output", "")
        tm_output = response.get("tm_output", "")
        final_output = response.get("final_output", "")

        # Clean the outputs
        clean_graph = extract_clean_output(str(graph_output))
        clean_topic = extract_clean_output(str(tm_output))
        clean_final = extract_clean_output(str(final_output))

        # Create formatted response
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

**System Performance:** ✅ Multi-Agent Coordination Active"""

        return {
            "status": "success",
            "message": formatted_message,
            "query": request.query,
            "agents_used": {
                "graph_writer": bool(clean_graph and clean_graph != "No output available"),
                "topic_model": bool(clean_topic and clean_topic != "No output available"),
                "supervisor": True
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/raw")
def process_query_raw(request: QueryRequest):
    """
    Raw output endpoint for debugging (keep your original functionality)
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