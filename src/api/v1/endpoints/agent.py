from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.research_coordinator import run_research_coordinator
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
        # Run the supervisor workflow - FIXED: Use correct function name
        response = run_research_coordinator(request.query)

        # Extract outputs - FIXED: Use correct field names from coordinator
        relationship_output = response.get("relationship_output", "")
        theme_output = response.get("theme_output", "")
        final_output = response.get("final_output", "")

        # Clean each output
        clean_relationship = extract_clean_content(relationship_output)
        clean_theme = extract_clean_content(theme_output)
        clean_final = extract_clean_content(final_output)

        # Check for database usage
        def has_real_data(content):
            indicators = [
                'database', 'retrieved', 'found', 'neo4j', 'mongodb',
                'concepts found', 'papers found', 'topics found'
            ]
            return any(indicator.lower() in content.lower() for indicator in indicators)

        relationship_has_data = has_real_data(clean_relationship)
        theme_has_data = has_real_data(clean_theme)

        # Create clean, formatted response
        if relationship_has_data and theme_has_data:
            # Both agents working with data
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**🔗 Relationship Analysis:**
{clean_relationship}

**📊 Theme Analysis:**
{clean_theme}

**💡 Synthesis:**
{clean_final}

*System Status: ✅ All agents active with database content*"""

        elif relationship_has_data:
            # Only relationship agent working
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**🔗 Relationship Analysis:**
{clean_relationship}

**⚠️ Theme Analysis:** Limited data available

**💡 Summary:**
{clean_final}

*System Status: 🟡 Relationship data available, themes need attention*"""

        elif theme_has_data:
            # Only theme agent working
            formatted_message = f"""**🎯 Research Analysis Results**

*Query:* {request.query}

**📊 Theme Analysis:**
{clean_theme}

**⚠️ Relationship Analysis:** Limited data available

**💡 Summary:**
{clean_final}

*System Status: 🟡 Theme data available, relationships need attention*"""

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
                "relationship_analyst": "✅ Active" if clean_relationship != "No response available" else "❌ Inactive",
                "theme_analyst": "✅ Active" if clean_theme != "No response available" else "❌ Inactive",
                "database_usage": "✅ High" if (relationship_has_data and theme_has_data) else "🟡 Partial" if (
                        relationship_has_data or theme_has_data) else "❌ Low",
                "response_quality": "Database-driven" if (relationship_has_data or theme_has_data) else "General knowledge"
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
        response = run_research_coordinator(request.query)
        return {
            "status": "success",
            "message": str(response.get("final_output", "")),
            "debug": {
                "relationship_output": str(response.get("relationship_output", "")),
                "theme_output": str(response.get("theme_output", ""))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))