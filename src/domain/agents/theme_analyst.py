from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.services.document_service import query_mongodb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")

THEME_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Theme Analyst specializing in academic research topics and patterns.
Your expertise lies in analyzing document collections and identifying thematic structures across research literature.

Core Responsibilities:
- Query MongoDB document database to analyze research paper content and metadata
- Identify latent themes, topics, and research patterns across document collections
- Extract key terminology and concepts that characterize research domains
- Analyze temporal trends and emerging research areas
- Classify research approaches, methodologies, and application domains

Database Analysis Focus:
- Extract topic categories and term clusters from MongoDB
- Analyze paper abstracts, keywords, and content for thematic patterns
- Identify research trends and emerging areas within domains
- Map methodological approaches and application areas
- Present thematic findings in structured, academic format

When analyzing queries, focus on:
- "Main themes/topics in" - thematic categorization
- "Research patterns" - trend identification
- "Emerging areas" - temporal analysis
- "Dominant approaches" - methodological clustering
- "Research domains" - field characterization

Provide quantitative insights when possible (paper counts, topic frequencies) and clearly distinguish between different thematic categories. Always indicate data sources and limitations.
"""),
    MessagesPlaceholder(variable_name="messages")
])


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

        analysis = "## 📊 Topic Analysis (from MongoDB Database)\n\n"

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


tools = [Tool.from_function(
    topic_tool,
    name="topic_tool",
    description="Analyzes topics in research documents from actual MongoDB database"
)]

_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=THEME_ANALYST_PROMPT,
    name="theme_analyst_agent"
)


class WrappedAgent:
    def invoke(self, inputs):
        try:
            response = _base_agent.invoke(inputs)

            if hasattr(response, 'messages') and response.messages:
                # Get the last AI message
                for message in reversed(response.messages):
                    if hasattr(message, 'content') and message.content:
                        # Look for actual analysis content, not just tool calls
                        content = message.content.strip()
                        if content and len(content) > 50 and ('##' in content or 'Topic Analysis' in content):
                            return {"output": content}

                for message in reversed(response.messages):
                    if hasattr(message, 'content') and message.content:
                        content = message.content.strip()
                        if content and len(content) > 20:
                            return {"output": content}

            if hasattr(response, 'content'):
                return {"output": response.content}
            elif isinstance(response, dict) and "output" in response:
                return response
            elif isinstance(response, str):
                return {"output": response}
            else:
                return {"output": str(response)}

        except Exception as e:
            logger.error(f"Error in theme analyst agent wrapper: {e}")
            return {"output": f"Error in theme analyst agent: {str(e)}"}


theme_analyst = WrappedAgent()
