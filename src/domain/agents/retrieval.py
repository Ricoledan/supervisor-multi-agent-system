from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.services.vector_service import query_vectordb
from src.services.document_service import query_mongodb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


def retrieval_tool(input: str) -> str:
    """Find papers and documents relevant to the query"""
    try:
        # Semantic search
        vector_results = query_vectordb(input, limit=5)

        # Document search
        doc_results = query_mongodb(input)
        papers = doc_results.get("papers", [])

        analysis = "## 📚 Document Retrieval Results\n\n"

        # Semantic similarity results
        if vector_results:
            analysis += f"### 🔍 Semantic Search ({len(vector_results)} chunks found)\n"
            for i, result in enumerate(vector_results[:3]):
                relevance = result.get('relevance_score', 0)
                content_preview = result.get('content', '')[:150]

                relevance_indicator = "🎯" if relevance > 0.8 else "📋" if relevance > 0.6 else "📄"
                analysis += f"{relevance_indicator} **Match {i + 1}** (relevance: {relevance:.2f})\n"
                analysis += f"*{content_preview}...*\n\n"

        # Document-level results
        if papers:
            analysis += f"### 📄 Related Papers ({len(papers)})\n"
            for paper in papers[:3]:
                title = paper.get('title', 'Unknown')
                authors = paper.get('authors', [])
                author_str = ", ".join(authors[:2]) if authors else "Unknown authors"

                analysis += f"- **{title}**\n"
                analysis += f"  *by {author_str}*\n"

        # Retrieval summary
        analysis += f"\n### 📊 Retrieval Summary\n"
        analysis += f"- Found {len(vector_results)} relevant text chunks\n"
        analysis += f"- Identified {len(papers)} related papers\n"

        if vector_results:
            avg_relevance = sum(r.get('relevance_score', 0) for r in vector_results) / len(vector_results)
            analysis += f"- Average relevance score: {avg_relevance:.2f}\n"

        return analysis

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return f"Error during document retrieval: {str(e)}"


# Create the retrieval tool
tools = [Tool.from_function(
    retrieval_tool,
    name="retrieval_tool",
    description="Finds relevant papers and document chunks using semantic search"
)]

# Create retrieval agent prompt
RETRIEVAL_AGENT_PROMPT = """You are a Retrieval Agent that finds relevant academic papers and documents.
Your job is to search through the document collection and vector database to find the most relevant content for user queries.

Your responsibilities:
- Use semantic search to find relevant text chunks
- Query document metadata to find related papers
- Provide relevance scores and content previews
- Summarize retrieval results clearly

Focus on finding the most relevant and useful content for the user's research needs."""

# Create the agent
_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=RETRIEVAL_AGENT_PROMPT,
    name="retrieval_agent"
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


# Export the retrieval agent
retrieval_agent = WrappedAgent()