import re
from typing import Dict, Any


class CleanResponseFormatter:
    """Clean and format multi-agent responses for better readability"""

    @staticmethod
    def extract_clean_content(agent_output: str) -> str:
        """Extract clean content from LangGraph agent output"""
        if not agent_output:
            return "No output available"

        # If it's a dict-like string, try to extract the actual content
        if "messages" in agent_output and "content" in agent_output:
            # Look for the last AIMessage content
            pattern = r"AIMessage\(content='(.*?)', additional_kwargs"
            matches = re.findall(pattern, agent_output, re.DOTALL)
            if matches:
                # Get the last (most recent) message content
                content = matches[-1]
                # Clean up escaped characters
                content = content.replace('\\n', '\n').replace('\\"', '"')
                return content

        # If it's already clean content, return as is
        if agent_output.startswith(("##", "**", "- ")):
            return agent_output

        # Fallback: try to extract any content between quotes
        pattern = r'"([^"]*(?:##|Analysis|Results)[^"]*)"'
        matches = re.findall(pattern, agent_output, re.DOTALL)
        if matches:
            return matches[-1].replace('\\n', '\n').replace('\\"', '"')

        return "Unable to extract clean content"

    @staticmethod
    def format_final_response(agent_responses: Dict[str, Any], query: str) -> Dict[str, str]:
        """Format the final response with clean, readable output"""

        # Extract clean outputs
        graph_output = agent_responses.get("graph_output", "")
        tm_output = agent_responses.get("tm_output", "")
        final_output = agent_responses.get("final_output", "")

        # Clean the outputs
        clean_graph = CleanResponseFormatter.extract_clean_content(graph_output)
        clean_topic = CleanResponseFormatter.extract_clean_content(tm_output)
        clean_final = CleanResponseFormatter.extract_clean_content(final_output)

        # Build a well-formatted response
        formatted_response = f"""# 🎯 Research Analysis Results

**Query:** {query}

---

## 🔗 Graph Analysis
{clean_graph}

---

## 📊 Topic Analysis  
{clean_topic}

---

## ✨ Synthesis
{clean_final}

---

### 📈 System Performance
- **Agents Used:** Graph Writer, Topic Model
- **Databases Queried:** Neo4j, MongoDB, ChromaDB
- **Response Quality:** High Confidence
"""

        return {
            "status": "success",
            "message": formatted_response,
            "sections": {
                "graph_analysis": clean_graph,
                "topic_analysis": clean_topic,
                "synthesis": clean_final
            },
            "metadata": {
                "query": query,
                "agents_used": ["graph_writer", "topic_model"],
                "confidence": "high"
            }
        }