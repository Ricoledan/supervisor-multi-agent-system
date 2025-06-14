# src/api/v1/endpoints/agent.py - CLEANED UP VERSION

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.supervisor import run_supervisor
import re

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


def extract_clean_content(raw_output: str) -> str:
    """Extract clean, readable content from agent output - IMPROVED"""
    if not raw_output:
        return "No response available"

    content = str(raw_output)

    # Look for tool call results (the actual database responses)
    if "tool_call_id" in content and "## 🔗" in content:
        # Extract graph analysis
        start = content.find("## 🔗")
        end = content.find("', name='enhanced_graph_tool")
        if start != -1 and end != -1:
            extracted = content[start:end]
            # Clean up escape characters
            extracted = extracted.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            return extracted.strip()

    if "tool_call_id" in content and "## 📊" in content:
        # Extract topic analysis
        start = content.find("## 📊")
        end = content.find("', name='topic_tool")
        if start != -1 and end != -1:
            extracted = content[start:end]
            # Clean up escape characters
            extracted = extracted.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            return extracted.strip()

    # Look for AIMessage content
    if "AIMessage(content=" in content:
        # Find the content within AIMessage
        pattern = r"AIMessage\(content=['\"]([^'\"]*(?:\\.[^'\"]*)*)['\"]"
        import re
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            clean_content = matches[-1]  # Get last match
            clean_content = clean_content.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
            if len(clean_content) > 50:
                return clean_content.strip()

    # Look for markdown headers
    if "## " in content:
        lines = content.split('\n')
        meaningful_lines = []
        in_meaningful_section = False

        for line in lines:
            if line.strip().startswith("## "):
                in_meaningful_section = True
                meaningful_lines.append(line.strip())
            elif in_meaningful_section and line.strip():
                if not any(skip in line for skip in ['tool_call_id', 'AIMessage', 'HumanMessage']):
                    meaningful_lines.append(line.strip())
            elif in_meaningful_section and not line.strip():
                meaningful_lines.append("")  # Keep paragraph breaks

        if meaningful_lines:
            return '\n'.join(meaningful_lines).strip()

    # Fallback: Look for any meaningful content
    sentences = content.split('.')
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and
                not any(skip in sentence.lower() for skip in [
                    'tool_call_id', 'aimessage', 'humanmessage', 'additional_kwargs'
                ])):
            clean_sentences.append(sentence)
        if len(clean_sentences) >= 2:
            break

    if clean_sentences:
        return '. '.join(clean_sentences) + '.'

    return "Unable to extract meaningful content"


def clean_duplicate_content(text: str) -> str:
    """Remove duplicate sections from text"""
    lines = text.split('\n')
    seen_lines = set()
    clean_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines and duplicates
        if line and line not in seen_lines:
            seen_lines.add(line)
            clean_lines.append(line)

    return '\n'.join(clean_lines)


@router.post("/agent")
def process_query(request: QueryRequest):
    """
    Process query and return clean, readable output
    """
    try:
        # Run the supervisor workflow
        response = run_supervisor(request.query)

        # Extract outputs
        graph_output = response.get("graph_output", "")
        tm_output = response.get("tm_output", "")
        final_output = response.get("final_output", "")

        # Clean each output
        clean_graph = extract_clean_content(graph_output)
        clean_topic = extract_clean_content(tm_output)
        clean_final = extract_clean_content(final_output)

        # Check for database usage
        def has_real_data(content):
            indicators = [
                'database', 'retrieved', 'found', 'neo4j', 'mongodb',
                'concepts found', 'papers found', 'topics found'
            ]
            return any(indicator.lower() in content.lower() for indicator in indicators)

        graph_has_data = has_real_data(clean_graph)
        topic_has_data = has_real_data(clean_topic)

        # Create clean, formatted response
        if graph_has_data and topic_has_data:
            # Both agents working with data
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**📊 Knowledge Graph Findings:**
{clean_graph}

**🏷️ Topic Analysis:**
{clean_topic}

**💡 Key Insights:**
{clean_final}

*System Status: ✅ All agents active with database content*"""

        elif graph_has_data:
            # Only graph agent working
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**📊 Knowledge Graph Analysis:**
{clean_graph}

**⚠️ Topic Analysis:** Limited data available

**💡 Summary:**
{clean_final}

*System Status: 🟡 Graph data available, topics need attention*"""

        elif topic_has_data:
            # Only topic agent working
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**🏷️ Topic Analysis:**
{clean_topic}

**⚠️ Graph Analysis:** Limited data available

**💡 Summary:**
{clean_final}

*System Status: 🟡 Topic data available, graph needs attention*"""

        else:
            # Generic response mode
            formatted_message = f"""**🎯 Research Response**

*Query:* {request.query}

**Response:**
{clean_final}

*System Status: ⚠️ Using general knowledge (check database ingestion)*"""

        # Remove any duplicate content
        formatted_message = clean_duplicate_content(formatted_message)

        return {
            "status": "success",
            "message": formatted_message,
            "query": request.query,
            "system_health": {
                "graph_agent": "✅ Active" if clean_graph != "No response available" else "❌ Inactive",
                "topic_agent": "✅ Active" if clean_topic != "No response available" else "❌ Inactive",
                "database_usage": "✅ High" if (graph_has_data and topic_has_data) else "🟡 Partial" if (
                            graph_has_data or topic_has_data) else "❌ Low",
                "response_quality": "Database-driven" if (graph_has_data or topic_has_data) else "General knowledge"
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