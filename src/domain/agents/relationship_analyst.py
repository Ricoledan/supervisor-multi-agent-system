from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import MessagesState

from src.services.graph_service import query_graphdb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")


# Modern tool with state injection
@tool
def analyze_research_relationships(
        query: str,
        state: Annotated[MessagesState, InjectedState]
) -> str:
    """Analyze relationships between research entities using Neo4j graph database.

    This tool queries the Neo4j knowledge graph to find connections between:
    - Research papers and their citations
    - Authors and their collaborations
    - Concepts and their relationships
    - Research lineages and influence patterns

    Args:
        query: The research question focusing on relationships and connections
        state: Current conversation state (automatically injected)

    Returns:
        Detailed analysis of research relationships found in the database
    """
    try:
        logger.info(f"Analyzing relationships for query: {query[:50]}...")

        # Query the graph database
        graph_data = query_graphdb(query)

        concepts = graph_data.get("concepts", [])
        relationships = graph_data.get("relationships", [])
        papers = graph_data.get("papers", [])

        logger.info(
            f"Database returned: {len(concepts)} concepts, {len(relationships)} relationships, {len(papers)} papers"
        )

        # Check if we have actual data
        if not concepts and not relationships and not papers:
            return """## 🔗 Knowledge Graph Analysis

**Database Status:** No relationship data found in Neo4j database.

**Possible Issues:**
- Database may be empty (run ingestion pipeline)
- Query terms don't match existing data
- Database connection issues

**Recommendation:** Ensure academic papers have been ingested into the system using:
```bash
python cli.py start  # This will run ingestion automatically
```

For now, I can provide general insights about research relationships, but specific database-driven analysis requires populated data."""

        # Build comprehensive analysis
        analysis = "## 🔗 Knowledge Graph Analysis (Neo4j Database Results)\n\n"

        # Concepts analysis
        if concepts:
            analysis += f"### 📋 Research Concepts Found ({len(concepts)})\n\n"

            # Group concepts by category for better organization
            concept_groups = {}
            for concept in concepts:
                category = concept.get('category', 'General')
                if category not in concept_groups:
                    concept_groups[category] = []
                concept_groups[category].append(concept)

            for category, group_concepts in concept_groups.items():
                analysis += f"**{category} Concepts:**\n"
                for concept in group_concepts[:4]:  # Top 4 per category
                    name = concept.get('name', 'Unknown')
                    desc = concept.get('description', 'No description available')
                    # Truncate long descriptions
                    desc_short = desc[:120] + "..." if len(desc) > 120 else desc
                    analysis += f"- **{name}**: {desc_short}\n"
                analysis += "\n"

        # Relationships analysis
        if relationships:
            analysis += f"### 🔗 Key Relationships Found ({len(relationships)})\n\n"

            # Group relationships by type
            rel_groups = {}
            for rel in relationships:
                rel_type = rel.get('type', 'related_to')
                if rel_type not in rel_groups:
                    rel_groups[rel_type] = []
                rel_groups[rel_type].append(rel)

            for rel_type, group_rels in rel_groups.items():
                analysis += f"**{rel_type.replace('_', ' ').title()} Relationships:**\n"
                for rel in group_rels[:3]:  # Top 3 per type
                    from_node = rel.get('from', 'Unknown')
                    to_node = rel.get('to', 'Unknown')
                    description = rel.get('description', '')

                    analysis += f"- {from_node} → {to_node}"
                    if description:
                        desc_short = description[:100] + "..." if len(description) > 100 else description
                        analysis += f"\n  *{desc_short}*"
                    analysis += "\n"
                analysis += "\n"

        # Connected papers analysis
        if papers:
            analysis += f"### 📄 Connected Research Papers ({len(papers)})\n\n"
            for paper in papers[:4]:  # Top 4 papers
                title = paper.get('title', 'Unknown Title')
                year = paper.get('year', 'Unknown Year')
                authors = paper.get('authors', [])
                concepts_list = paper.get('related_concepts', [])

                analysis += f"**{title}** ({year})\n"

                if authors and authors != ['Unknown Author']:
                    author_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        author_str += f" +{len(authors) - 3} more"
                    analysis += f"*Authors: {author_str}*\n"

                if concepts_list:
                    concept_str = ", ".join(concepts_list[:4])
                    if len(concepts_list) > 4:
                        concept_str += f" +{len(concepts_list) - 4} more"
                    analysis += f"*Related concepts: {concept_str}*\n"

                analysis += "\n"

        # Research insights
        analysis += "### 💡 Relationship Insights\n\n"

        if relationships:
            # Analyze relationship patterns
            rel_types = [rel.get('type', 'unknown') for rel in relationships]
            most_common_rel = max(set(rel_types), key=rel_types.count) if rel_types else None

            analysis += f"- **Database Coverage**: {len(concepts)} concepts connected by {len(relationships)} relationships\n"
            analysis += f"- **Network Density**: Found {len(papers)} papers linking these concepts\n"

            if most_common_rel:
                rel_count = rel_types.count(most_common_rel)
                analysis += f"- **Dominant Pattern**: '{most_common_rel.replace('_', ' ').title()}' relationships ({rel_count} instances)\n"

            # Identify central concepts (those appearing in most relationships)
            concept_mentions = {}
            for rel in relationships:
                from_concept = rel.get('from', '')
                to_concept = rel.get('to', '')
                concept_mentions[from_concept] = concept_mentions.get(from_concept, 0) + 1
                concept_mentions[to_concept] = concept_mentions.get(to_concept, 0) + 1

            if concept_mentions:
                central_concept = max(concept_mentions.items(), key=lambda x: x[1])
                analysis += f"- **Central Concept**: '{central_concept[0]}' appears in {central_concept[1]} relationships\n"

        analysis += f"\n**Data Source**: Neo4j Knowledge Graph Database\n"
        analysis += f"**Query Processing**: Successfully analyzed {len(concepts) + len(relationships) + len(papers)} entities"

        return analysis

    except Exception as e:
        logger.error(f"Relationship analysis error: {e}", exc_info=True)
        return f"""## 🔗 Knowledge Graph Analysis

**Error**: Failed to analyze relationships: {str(e)}

**Troubleshooting Steps:**
1. Check Neo4j database connection
2. Verify data has been ingested
3. Review database logs for connection issues

**Status**: Relationship analysis service encountered technical difficulties."""


# Create the modern agent using create_react_agent
relationship_analyst = create_react_agent(
    model=model,
    tools=[analyze_research_relationships],
    system_prompt="""You are a Relationship Analyst specializing in academic research connections and networks.

Your expertise lies in analyzing knowledge graphs and mapping relationships between research entities using Neo4j database.

**Core Responsibilities:**
- Query Neo4j graph database to find connections between papers, authors, and concepts
- Identify research lineages, citation networks, and conceptual relationships  
- Analyze how ideas flow between researchers, institutions, and research domains
- Map collaborative networks and cross-disciplinary connections
- Provide insights into research influence patterns and knowledge evolution

**Analysis Focus:**
- Direct relationships between research entities
- Citation networks and research lineages
- Author collaboration patterns
- Cross-disciplinary connections
- Influential concepts and papers

**Tool Usage:**
Always use the analyze_research_relationships tool to query the Neo4j database. This provides actual data-driven insights rather than general knowledge.

**Response Format:**
Provide structured analysis with clear sections for concepts, relationships, and papers found in the database. Always indicate when analysis is based on actual database content vs. general knowledge."""
)