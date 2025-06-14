# Update your existing src/domain/formatters/response_formatter.py

import json
import re
from typing import Dict, Any, List, Optional


class EnhancedResponseFormatter:
    """Formats agent responses with improved structure and readability."""

    @staticmethod
    def _extract_clean_content(agent_output: Any) -> str:
        """Extract clean content from agent output"""
        if isinstance(agent_output, str):
            # If it contains LangGraph message objects, extract the actual content
            if "messages" in agent_output and "AIMessage" in agent_output:
                # Find the last AIMessage content
                pattern = r"AIMessage\(content='([^']*(?:##[^']*)*)', additional_kwargs"
                matches = re.findall(pattern, agent_output, re.DOTALL)
                if matches:
                    content = matches[-1]
                    # Clean up escaped characters
                    content = content.replace('\\n', '\n').replace('\\"', '"')
                    return content

            # If it looks like it's already clean, return as is
            if agent_output.startswith(("##", "**", "- ", "🔗", "🎯")):
                return agent_output

        elif isinstance(agent_output, dict):
            # Try to get the output field
            if "output" in agent_output:
                return str(agent_output["output"])
            elif "content" in agent_output:
                return str(agent_output["content"])

        return str(agent_output) if agent_output else "No output available"

    @staticmethod
    def format_response(agent_responses: Dict[str, Any], query: str) -> Dict[str, str]:
        """Format multiple agent responses into a structured output."""

        graph_output = agent_responses.get("graph_output", "")
        tm_output = agent_responses.get("tm_output", "")
        final_output = agent_responses.get("final_output", "")

        # Extract clean content
        clean_graph = EnhancedResponseFormatter._extract_clean_content(graph_output)
        clean_topic = EnhancedResponseFormatter._extract_clean_content(tm_output)
        clean_final = EnhancedResponseFormatter._extract_clean_content(final_output)

        # Create a clean, readable response
        formatted_message = f"""# 🎯 Multi-Agent Research Analysis

**Query:** {query}

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

**System Info:** ✅ Graph Writer + ✅ Topic Model + ✅ Supervisor Synthesis
"""

        return {
            "status": "success",
            "message": formatted_message,
            "debug": {
                "graph_clean": clean_graph[:200] + "..." if len(clean_graph) > 200 else clean_graph,
                "topic_clean": clean_topic[:200] + "..." if len(clean_topic) > 200 else clean_topic
            }
        }

    @staticmethod
    def _extract_content(agent_output: Dict[str, Any]) -> str:
        """Legacy method - kept for compatibility"""
        return EnhancedResponseFormatter._extract_clean_content(agent_output)