from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import GRAPH_WRITER_AGENT_PROMPT
from src.services.graph_service import query_graphdb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


def enhanced_graph_tool(input: str) -> str:
    """Enhanced graph analysis with better insights"""
    try:
        graph_data = query_graphdb(input)

        concepts = graph_data.get("concepts", [])
        relationships = graph_data.get("relationships", [])
        papers = graph_data.get("papers", [])

        if not concepts:
            return "No relevant concepts found in the knowledge graph for this query."

        # Enhanced analysis
        analysis = "## 🔗 Knowledge Graph Analysis\n\n"

        # Concept analysis with categories
        analysis += f"### 📋 Key Concepts ({len(concepts)})\n"

        # Group concepts by category
        concept_groups = {}
        for concept in concepts:
            category = concept.get('category', 'General')
            if category not in concept_groups:
                concept_groups[category] = []
            concept_groups[category].append(concept)

        for category, group_concepts in concept_groups.items():
            analysis += f"\n**{category}:**\n"
            for concept in group_concepts[:3]:  # Top 3 per category
                desc = concept.get('description', '')[:100]
                analysis += f"- **{concept['name']}**: {desc}...\n"

        # Relationship analysis
        if relationships:
            analysis += f"\n### 🔗 Key Relationships ({len(relationships)})\n"

            # Sort by relationship type
            rel_types = {}
            for rel in relationships:
                rel_type = rel.get('type', 'related_to')
                if rel_type not in rel_types:
                    rel_types[rel_type] = []
                rel_types[rel_type].append(rel)

            for rel_type, type_rels in rel_types.items():
                analysis += f"\n**{rel_type.replace('_', ' ').title()}:**\n"
                for rel in type_rels[:3]:  # Top 3 per type
                    analysis += f"- {rel['from']} → {rel['to']}\n"

        # Paper connections
        if papers:
            analysis += f"\n### 📄 Connected Papers ({len(papers)})\n"
            for paper in papers[:3]:
                concepts_list = ", ".join(paper.get('related_concepts', [])[:3])
                analysis += f"- **{paper.get('title', 'Unknown')}**\n"
                analysis += f"  *Key concepts: {concepts_list}*\n"

        # Add insights summary
        analysis += f"\n### 💡 Key Insights\n"
        analysis += f"- Found {len(concepts)} interconnected concepts\n"
        analysis += f"- Identified {len(relationships)} meaningful relationships\n"
        analysis += f"- Connected to {len(papers)} research papers\n"

        return analysis

    except Exception as e:
        logger.error(f"Enhanced graph analysis error: {e}")
        return f"Error performing graph analysis: {str(e)}"


# Create the enhanced tool
tools = [Tool.from_function(
    enhanced_graph_tool,
    name="enhanced_graph_tool",
    description="Analyzes concept relationships in the knowledge graph with detailed insights"
)]

# Create the agent with enhanced tool
_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=GRAPH_WRITER_AGENT_PROMPT,
    name="enhanced_graph_writer_agent"
)


class WrappedAgent:
    def invoke(self, inputs):
        response = _base_agent.invoke(inputs)

        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, dict) and "output" in response:
            return response
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        return {"output": content}


# Export the enhanced agent
graph_writer_agent = WrappedAgent()