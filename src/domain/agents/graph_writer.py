from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import GRAPH_WRITER_AGENT_PROMPT
from src.services.graph_service import query_graphdb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


def graph_tool(input: str) -> str:
    """Analyzes relationships between concepts in the knowledge graph."""
    try:
        graph_data = query_graphdb(input)

        concepts = graph_data.get("concepts", [])
        relationships = graph_data.get("relationships", [])
        papers = graph_data.get("papers", [])

        if not concepts:
            return "No relevant concepts found in the knowledge graph."

        analysis = "## Knowledge Graph Analysis\n\n"

        analysis += f"### Key Concepts ({len(concepts)})\n"
        for concept in concepts[:7]:  # Top concepts
            analysis += f"- {concept['name']}: {concept.get('description', '')[:100]}\n"

        if relationships:
            analysis += f"\n### Key Relationships ({len(relationships)})\n"
            for rel in relationships[:5]:
                analysis += f"- {rel['from']} → {rel['to']} ({rel.get('type', 'related')})\n"

        return analysis
    except Exception as e:
        return f"Error performing graph analysis: {str(e)}"


tools = [Tool.from_function(graph_tool, name="graph_tool", description="Analyzes knowledge graph relationships")]

_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=GRAPH_WRITER_AGENT_PROMPT,
    name="graph_writer_agent"
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


graph_writer_agent = WrappedAgent()
