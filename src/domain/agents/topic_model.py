from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import TOPIC_MODEL_AGENT_PROMPT
from src.services.document_service import query_mongodb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


def topic_tool(input: str) -> str:
    """Extracts topics from documents related to the input query."""
    try:
        results = query_mongodb(input)

        topics = results.get("topics", {})
        papers = results.get("papers", [])

        if not topics:
            return "No relevant topics found in the database."

        topic_analysis = "## Topic Analysis\n\n"
        for category, terms in topics.items():
            topic_analysis += f"### {category}\n"
            for term in terms[:5]:  # Limit to top 5 terms
                topic_analysis += f"- {term['name']}\n"

        paper_summary = f"\n## Related Papers ({len(papers)})\n"
        for paper in papers[:3]:  # Summarize top 3 papers
            paper_summary += f"- {paper.get('title', 'Untitled')}\n"

        return topic_analysis + paper_summary
    except Exception as e:
        return f"Error performing topic analysis: {str(e)}"


tools = [Tool.from_function(topic_tool, name="topic_tool", description="Analyzes topics in research documents")]

_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=TOPIC_MODEL_AGENT_PROMPT,
    name="topic_model_agent"
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


topic_model_agent = WrappedAgent()