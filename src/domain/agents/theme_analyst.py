from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import MessagesState

from src.services.document_service import query_mongodb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


# Modern tool with state injection
@tool
def analyze_research_themes(
        query: str,
        state: Annotated[MessagesState, InjectedState]
) -> str:
    """Analyze themes and topics in research literature using MongoDB document database.

    This tool queries the MongoDB database to identify:
    - Dominant themes and topics across research papers
    - Key terminology and concepts within research domains
    - Temporal trends and emerging research areas
    - Methodological approaches and application domains
    - Research patterns and topic clusters

    Args:
        query: The research question focusing on themes, topics, or patterns
        state: Current conversation state (automatically injected)

    Returns:
        Detailed thematic analysis of research literature found in the database
    """
    try:
        logger.info(f"Analyzing themes for query: {query[:50]}...")

        # Query the document database
        results = query_mongodb(query)

        topics = results.get("topics", {})
        papers = results.get("papers", [])

        logger.info(f"MongoDB returned: {len(topics)} topic categories, {len(papers)} papers")

        # Check if we have actual data
        if not topics and not papers:
            return """## 📊 Topic Analysis

**Database Status:** No thematic data found in MongoDB database.

**Possible Issues:**
- MongoDB may be empty (run ingestion pipeline)
- Query terms don't match document content
- Topic modeling not yet completed

**Recommendation:** Ensure academic papers have been processed using:
```bash
python cli.py start  # This will run ingestion and topic modeling
```

For now, I can provide general insights about research themes, but specific database-driven analysis requires populated data."""

        # Build comprehensive thematic analysis
        analysis = "## 📊 Topic Analysis (MongoDB Database Results)\n\n"

        # Topic categories analysis
        if topics:
            analysis += f"### 🏷️ Research Topic Categories ({len(topics)})\n\n"

            total_terms = 0
            for category, terms in topics.items():
                analysis += f"**{category}:**\n"

                # Handle different term formats (dict vs string)
                term_list = []
                for term in terms[:6]:  # Top 6 terms per category
                    if isinstance(term, dict):
                        term_name = term.get('term', 'Unknown')
                        term_weight = term.get('weight', 0)
                        term_list.append(f"{term_name} (relevance: {term_weight:.3f})")
                    else:
                        term_list.append(str(term))
                    total_terms += 1

                for term_str in term_list:
                    analysis += f"- {term_str}\n"

                if len(terms) > 6:
                    analysis += f"- *...and {len(terms) - 6} more terms*\n"

                analysis += "\n"

        # Research papers analysis
        if papers:
            analysis += f"### 📄 Thematically Related Papers ({len(papers)})\n\n"

            # Analyze paper metadata for additional insights
            years = [p.get('year') for p in papers if p.get('year') and p.get('year') != 'Unknown']
            all_keywords = []
            author_count = 0

            for paper in papers[:5]:  # Top 5 papers
                title = paper.get('title', 'Untitled Paper')
                authors = paper.get('authors', [])
                year = paper.get('year', 'Unknown')
                keywords = paper.get('keywords', [])

                analysis += f"**{title}**\n"

                if authors and authors != ['Unknown Author']:
                    author_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        author_str += f" et al."
                    analysis += f"*Authors: {author_str}*\n"
                    author_count += len(authors)

                if year != 'Unknown':
                    analysis += f"*Year: {year}*\n"

                if keywords:
                    keyword_str = ", ".join(keywords[:4])
                    if len(keywords) > 4:
                        keyword_str += f" +{len(keywords) - 4} more"
                    analysis += f"*Keywords: {keyword_str}*\n"
                    all_keywords.extend(keywords)

                analysis += "\n"

            if len(papers) > 5:
                analysis += f"*...and {len(papers) - 5} additional papers*\n\n"

        # Thematic insights and patterns
        analysis += "### 📈 Thematic Insights\n\n"

        if topics:
            # Category analysis
            category_sizes = {cat: len(terms) for cat, terms in topics.items()}
            largest_category = max(category_sizes.items(), key=lambda x: x[1]) if category_sizes else None

            analysis += f"- **Topic Coverage**: {len(topics)} distinct research categories identified\n"
            analysis += f"- **Term Diversity**: {sum(len(terms) for terms in topics.values())} total topic terms extracted\n"

            if largest_category:
                analysis += f"- **Dominant Category**: '{largest_category[0]}' with {largest_category[1]} key terms\n"

        if papers:
            # Paper analysis
            if years:
                year_range = f"{min(years)}-{max(years)}" if len(set(years)) > 1 else str(years[0])
                analysis += f"- **Temporal Coverage**: Papers spanning {year_range}\n"

            analysis += f"- **Literature Base**: {len(papers)} papers contribute to thematic understanding\n"

            # Keyword frequency analysis
            if all_keywords:
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                most_common = keyword_counts.most_common(3)
                if most_common:
                    common_keywords = ", ".join([f"'{kw}' ({count})" for kw, count in most_common])
                    analysis += f"- **Recurring Keywords**: {common_keywords}\n"

        # Research domain insights
        if topics and papers:
            analysis += "\n### 🔍 Domain Insights\n\n"

            # Cross-category analysis
            if len(topics) > 1:
                categories = list(topics.keys())
                analysis += f"- **Interdisciplinary Scope**: Research spans {len(categories)} categories: {', '.join(categories[:3])}"
                if len(categories) > 3:
                    analysis += f" and {len(categories) - 3} more"
                analysis += "\n"

            # Research evolution (if temporal data available)
            if years and len(set(years)) > 1:
                recent_papers = sum(1 for y in years if y >= max(years) - 2)
                analysis += f"- **Research Activity**: {recent_papers} papers from recent years indicate active research area\n"

        analysis += f"\n**Data Source**: MongoDB Document Database\n"
        analysis += f"**Processing**: Successfully analyzed {len(topics)} topic categories and {len(papers)} research papers"

        return analysis

    except Exception as e:
        logger.error(f"Theme analysis error: {e}", exc_info=True)
        return f"""## 📊 Topic Analysis

**Error**: Failed to analyze themes: {str(e)}

**Troubleshooting Steps:**
1. Check MongoDB database connection
2. Verify topic modeling has been completed
3. Review ingestion pipeline logs

**Status**: Theme analysis service encountered technical difficulties."""


# Create the modern agent using create_react_agent
theme_analyst = create_react_agent(
    model=model,
    tools=[analyze_research_themes],
    system_prompt="""You are a Theme Analyst specializing in academic research topics and patterns.

Your expertise lies in analyzing document collections and identifying thematic structures across research literature using MongoDB database.

**Core Responsibilities:**
- Query MongoDB document database to analyze research paper content and metadata
- Identify latent themes, topics, and research patterns across document collections
- Extract key terminology and concepts that characterize research domains
- Analyze temporal trends and emerging research areas
- Classify research approaches, methodologies, and application domains

**Analysis Focus:**
- Thematic categorization and topic clustering
- Research trend identification and evolution
- Methodological approach analysis
- Domain-specific terminology and concepts
- Cross-disciplinary theme mapping

**Tool Usage:**
Always use the analyze_research_themes tool to query the MongoDB database. This provides actual data-driven insights from processed research papers rather than general knowledge.

**Response Format:**
Provide structured analysis with clear sections for topic categories, related papers, and thematic insights. Include quantitative data when available (paper counts, topic frequencies) and clearly distinguish between different thematic categories. Always indicate when analysis is based on actual database content."""
)