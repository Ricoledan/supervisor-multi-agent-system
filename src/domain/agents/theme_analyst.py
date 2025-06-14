# src/domain/agents/topic_model.py

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import TOPIC_MODEL_AGENT_PROMPT
from src.services.document_service import query_mongodb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


def topic_tool(input: str) -> str:
    """Enhanced topic analysis with ACTUAL database queries"""
    try:
        logger.info(f"Topic tool called with input: {input[:50]}...")

        # ACTUALLY CALL THE DATABASE
        results = query_mongodb(input)

        topics = results.get("topics", {})
        papers = results.get("papers", [])

        logger.info(f"MongoDB returned: {len(topics)} topic categories, {len(papers)} papers")

        # If no data found, be explicit about it
        if not topics and not papers:
            return f"""## 📊 Topic Analysis

**Database Query Results:** No data found in MongoDB for query: "{input}"

**Possible Issues:**
- MongoDB may be empty (run: python check_db.py)
- No papers ingested yet
- Query terms don't match document content

**Recommendation:** Run ingestion pipeline to populate MongoDB with academic papers."""

        # Build analysis from ACTUAL data
        analysis = "## 📊 Topic Analysis (from MongoDB Database)\n\n"

        # Real topics from database
        if topics:
            analysis += f"### 🏷️ Topic Categories Found ({len(topics)})\n"
            for category, terms in topics.items():
                analysis += f"\n**{category}:**\n"
                for term in terms[:5]:  # Top 5 terms per category
                    if isinstance(term, dict):
                        term_name = term.get('term', 'Unknown')
                        term_weight = term.get('weight', 0)
                        analysis += f"- {term_name} (weight: {term_weight:.3f})\n"
                    else:
                        analysis += f"- {term}\n"

        # Real papers from database
        if papers:
            analysis += f"\n### 📄 Related Papers ({len(papers)})\n"
            for paper in papers[:5]:
                title = paper.get('title', 'Untitled')
                authors = paper.get('authors', [])
                year = paper.get('year', 'Unknown')
                keywords = paper.get('keywords', [])

                analysis += f"- **{title}**\n"
                if authors:
                    author_str = ", ".join(authors[:2])
                    analysis += f"  *Authors: {author_str}*\n"
                if year != 'Unknown':
                    analysis += f"  *Year: {year}*\n"
                if keywords:
                    keyword_str = ", ".join(keywords[:3])
                    analysis += f"  *Keywords: {keyword_str}*\n"

        analysis += f"\n### 📊 Database Summary\n"
        analysis += f"- Found {len(topics)} topic categories in MongoDB\n"
        analysis += f"- Analyzed {len(papers)} research papers\n"

        total_terms = sum(len(terms) for terms in topics.values()) if topics else 0
        analysis += f"- Extracted {total_terms} total topic terms\n"

        return analysis

    except Exception as e:
        logger.error(f"Topic tool error: {e}", exc_info=True)
        return f"Error querying MongoDB: {str(e)}\nCheck database connection and data availability."


tools = [Tool.from_function(topic_tool, name="topic_tool",
                            description="Analyzes topics in research documents from actual MongoDB database")]

_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=TOPIC_MODEL_AGENT_PROMPT,
    name="topic_model_agent"
)


class WrappedAgent:
    def invoke(self, inputs):
        try:
            # Get the response from the base agent
            response = _base_agent.invoke(inputs)

            # Extract clean content from LangGraph response
            if hasattr(response, 'messages') and response.messages:
                # Get the last AI message
                for message in reversed(response.messages):
                    if hasattr(message, 'content') and message.content:
                        # Look for actual analysis content, not just tool calls
                        content = message.content.strip()
                        if content and len(content) > 50 and ('##' in content or 'Topic Analysis' in content):
                            return {"output": content}

                # If no good content found, look at tool responses
                for message in reversed(response.messages):
                    if hasattr(message, 'content') and message.content:
                        content = message.content.strip()
                        if content and len(content) > 20:
                            return {"output": content}

            # Fallback: check if response has direct output
            if hasattr(response, 'content'):
                return {"output": response.content}
            elif isinstance(response, dict) and "output" in response:
                return response
            elif isinstance(response, str):
                return {"output": response}
            else:
                # Last resort: convert to string
                return {"output": str(response)}

        except Exception as e:
            logger.error(f"Error in topic model agent wrapper: {e}")
            return {"output": f"Error in topic model agent: {str(e)}"}


# Export the fixed agent
topic_model_agent = WrappedAgent()