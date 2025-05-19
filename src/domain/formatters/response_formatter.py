import json
from typing import Dict, Any, List, Optional


class EnhancedResponseFormatter:
    """Formats agent responses with improved structure and readability."""

    @staticmethod
    def format_response(agent_responses: Dict[str, Any], query: str) -> Dict[str, str]:
        """Format multiple agent responses into a structured output."""

        graph_output = agent_responses.get("graph_output", {})
        tm_output = agent_responses.get("tm_output", {})

        graph_content = EnhancedResponseFormatter._extract_content(graph_output)
        tm_content = EnhancedResponseFormatter._extract_content(tm_output)

        structured_response = {
            "summary": EnhancedResponseFormatter._generate_summary(query, graph_content, tm_content),
            "sections": EnhancedResponseFormatter._generate_sections(graph_content, tm_content),
            "recommendations": EnhancedResponseFormatter._generate_recommendations(query, graph_content, tm_content),
            "visualizations": EnhancedResponseFormatter._generate_visualization_specs(graph_content, tm_content),
            "metadata": {
                "confidence_score": EnhancedResponseFormatter._calculate_confidence(graph_content, tm_content),
                "sources_count": EnhancedResponseFormatter._count_sources(graph_content, tm_content),
                "query_coverage": EnhancedResponseFormatter._estimate_coverage(query, graph_content, tm_content)
            }
        }

        formatted_markdown = EnhancedResponseFormatter._to_markdown(structured_response)

        return {
            "status": "success",
            "message": formatted_markdown,
            "debug": agent_responses,
            "structured_data": structured_response
        }

    @staticmethod
    def _extract_content(agent_output: Dict[str, Any]) -> str:
        """Extract content from agent output."""
        if isinstance(agent_output, str):
            return agent_output

        messages = agent_output.get("messages", [])
        if messages and len(messages) > 1:
            # Get the last AI message
            for message in reversed(messages):
                if message.get("content"):
                    return message.get("content", "")

        # Fallback to looking for 'output' key
        return agent_output.get("output", "")

    @staticmethod
    def _generate_summary(query: str, graph_content: str, tm_content: str) -> str:
        """Generate a concise summary of findings."""
        # Simple approach: take first 1-2 sentences of content up to ~200 chars
        combined = graph_content + " " + tm_content
        sentences = combined.split(". ")
        summary = ". ".join(sentences[:2])

        if len(summary) > 200:
            summary = summary[:197] + "..."

        return summary

    @staticmethod
    def _generate_sections(graph_content: str, tm_content: str) -> List[Dict[str, str]]:
        """Break content into structured sections."""
        sections = []

        for content, source in [(graph_content, "Knowledge Graph"), (tm_content, "Topic Model")]:
            section_parts = content.split("##")

            for part in section_parts:
                if not part.strip():
                    continue

                lines = part.strip().split("\n")
                if lines:
                    title = lines[0].strip("# ")
                    body = "\n".join(lines[1:]).strip()

                    sections.append({
                        "title": title,
                        "content": body,
                        "source": source
                    })

        return sections

    @staticmethod
    def _generate_recommendations(query: str, graph_content: str, tm_content: str) -> List[str]:
        """Generate practical recommendations based on findings."""
        recommendations = [
            "Explore related papers in the knowledge graph",
            "Consider alternative search terms for more comprehensive results",
            "Analyze citation patterns between key authors"
        ]
        return recommendations

    @staticmethod
    def _generate_visualization_specs(graph_content: str, tm_content: str) -> Dict[str, Any]:
        """Generate visualization specifications for frontend rendering."""
        return {
            "network_graph": {
                "type": "graph",
                "description": "Network of relationships between concepts and papers",
                "data_source": "graph_agent"
            },
            "topic_clusters": {
                "type": "bubbles",
                "description": "Clusters of related topics from the document analysis",
                "data_source": "topic_model_agent"
            }
        }

    @staticmethod
    def _calculate_confidence(graph_content: str, tm_content: str) -> float:
        """Estimate confidence in results."""
        confidence = 0.5

        if len(graph_content) > 100:
            confidence += 0.15

        if len(tm_content) > 100:
            confidence += 0.15

        uncertainty_phrases = ["not found", "error", "could not", "no relevant"]
        for phrase in uncertainty_phrases:
            if phrase in graph_content.lower() or phrase in tm_content.lower():
                confidence -= 0.1

        return min(1.0, max(0.1, confidence))

    @staticmethod
    def _count_sources(graph_content: str, tm_content: str) -> int:
        """Count number of sources referenced."""
        import re
        papers_mentioned = len(re.findall(r'paper|title|published',
                                          graph_content.lower() + tm_content.lower()))
        return max(1, min(papers_mentioned, 20))

    @staticmethod
    def _estimate_coverage(query: str, graph_content: str, tm_content: str) -> float:
        """Estimate how well the response covers the query."""
        query_terms = set(query.lower().split())
        response_text = graph_content.lower() + " " + tm_content.lower()

        matched_terms = sum(1 for term in query_terms if term in response_text)
        return min(1.0, matched_terms / max(1, len(query_terms)))

    @staticmethod
    def _to_markdown(structured_response: Dict[str, Any]) -> str:
        """Convert structured response to formatted markdown."""
        md = []

        md.append(structured_response["summary"])
        md.append("\n\n")

        for section in structured_response["sections"]:
            md.append(f"## {section['title']}")
            md.append(section["content"])
            md.append("\n")

        if structured_response["recommendations"]:
            md.append("## Recommendations")
            for rec in structured_response["recommendations"]:
                md.append(f"- {rec}")
            md.append("\n")

        md.append("---")
        confidence = structured_response["metadata"]["confidence_score"]
        confidence_indicator = "🟢 High" if confidence > 0.7 else "🟡 Medium" if confidence > 0.4 else "🔴 Low"
        md.append(
            f"**Confidence**: {confidence_indicator} | **Sources**: {structured_response['metadata']['sources_count']}")

        return "\n".join(md)