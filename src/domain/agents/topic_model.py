from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import TOPIC_MODEL_AGENT_PROMPT
from src.utils.model_init import get_openai_model

model = get_openai_model()


def identify_topics(text: str) -> str:
    """
    Extract major topics and themes from a set of academic papers.

    Args:
        text: Content of papers to analyze

    Returns:
        JSON formatted list of topics with descriptive labels
    """
    return f"Identified topics in the provided text: [Topic analysis would be performed here]"


def cluster_documents(docs: str) -> str:
    """
    Group similar documents together based on content.

    Args:
        docs: JSON string containing document information

    Returns:
        JSON formatted clusters with document IDs and similarity scores
    """
    return f"Document clustering results: [Clustering would be performed here]"


def analyze_research_trends(papers: str) -> str:
    """
    Identify emerging or dominant research trends across papers.

    Args:
        papers: JSON string containing paper details with dates

    Returns:
        Analysis of trending research directions and potential gaps
    """
    return f"Research trend analysis: [Trend analysis would be performed here]"


tools = [
    Tool.from_function(
        name="identify_topics",
        func=identify_topics,
        description="Extract major topics and themes from academic papers"
    ),
    Tool.from_function(
        name="cluster_documents",
        func=cluster_documents,
        description="Group similar documents by content and provide similarity clusters"
    ),
    Tool.from_function(
        name="analyze_research_trends",
        func=analyze_research_trends,
        description="Identify research trends and potential research gaps"
    )
]

tm_agent = create_react_agent(model, tools, prompt=TOPIC_MODEL_AGENT_PROMPT)


def run_topic_model(query: str) -> dict:
    """
    Executes the topic model agent with the given query.

    Args:
        query (str): The input prompt to process.

    Returns:
        dict: The structured response from the topic model agent.
    """
    try:
        print("Running topic model agent...", flush=True)
        response = tm_agent.invoke({"input": query})
        print("Topic model response:", response, flush=True)
        return response
    except Exception as e:
        error_message = f"Error during topic model execution: {e}"
        print(error_message, flush=True)
        return {"error": error_message}
