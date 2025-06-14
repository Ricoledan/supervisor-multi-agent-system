import re
from typing import Dict, Any


class EnhancedResponseFormatter:
    """Fixed response formatter for LangGraph outputs"""

    @staticmethod
    def _extract_clean_content(agent_output: Any) -> str:
        """Extract clean content from LangGraph agent output"""
        if not agent_output:
            return "No output available"

        # Convert to string if it's not already
        output_str = str(agent_output)

        # If it contains LangGraph messages, extract the actual content
        if "AIMessage(content=" in output_str:
            # Look for the content inside AIMessage
            pattern = r"AIMessage\(content='(.*?)', additional_kwargs"
            matches = re.findall(pattern, output_str, re.DOTALL)
            if matches:
                content = matches[-1]  # Get the last match
                # Clean up escaped characters
                content = content.replace('\\n', '\n').replace('\\"', '"')
                return content

        # If it's a dict with output key
        if isinstance(agent_output, dict) and "output" in agent_output:
            return str(agent_output["output"])

        # If it already looks like clean content (starts with markdown headers)
        if any(output_str.strip().startswith(marker) for marker in ["##", "**", "- ", "🔗", "📊"]):
            return output_str.strip()

        # If it's just a simple string response
        if len(output_str) < 1000 and not "messages" in output_str:
            return output_str.strip()

        return "Unable to extract clean content from agent output"

    @staticmethod
    def format_response(agent_responses: Dict[str, Any], query: str) -> Dict[str, str]:
        """Format multiple agent responses into a structured output"""

        graph_output = agent_responses.get("graph_output", "")
        tm_output = agent_responses.get("tm_output", "")
        final_output = agent_responses.get("final_output", "")

        # Extract clean content using improved method
        clean_graph = EnhancedResponseFormatter._extract_clean_content(graph_output)
        clean_topic = EnhancedResponseFormatter._extract_clean_content(tm_output)
        clean_final = EnhancedResponseFormatter._extract_clean_content(final_output)

        # Create formatted response
        formatted_message = f"""# 🎯 Multi-Agent Research Analysis

**Query:** {query}

---

{clean_graph}

---

{clean_topic}

---

## ✨ Synthesized Insights
{clean_final if clean_final != "No output available" else "Synthesis based on available agent outputs"}

---

**Database Status:** Neo4j: {'✅' if 'Neo4j Database' in clean_graph else '❌'} | MongoDB: {'✅' if 'MongoDB Database' in clean_topic else '❌'}
"""

        return {
            "status": "success",
            "message": formatted_message,
            "debug": {
                "graph_extracted": clean_graph[:100] + "..." if len(clean_graph) > 100 else clean_graph,
                "topic_extracted": clean_topic[:100] + "..." if len(clean_topic) > 100 else clean_topic
            }
        }