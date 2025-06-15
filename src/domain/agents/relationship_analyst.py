from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.services.graph_service import query_graphdb
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4")

RELATIONSHIP_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Relationship Analyst specializing in academic research connections and networks.
Your expertise lies in analyzing knowledge graphs and mapping relationships between research entities.

Core Responsibilities:
- Query Neo4j graph database to find connections between papers, authors, and concepts
- Identify research lineages, citation networks, and conceptual relationships
- Analyze how ideas flow between researchers, institutions, and research domains
- Map collaborative networks and cross-disciplinary connections
- Provide insights into research influence patterns and knowledge evolution

Database Analysis Focus:
- Extract meaningful concept relationships from Neo4j
- Identify central nodes and influential connections in research networks
- Trace research lineages and conceptual evolution paths
- Analyze collaborative patterns and institutional connections
- Present relationship findings in clear, academic format

When analyzing queries, focus on:
- "How X relates to Y" - direct relationship analysis
- "Connections between" - network mapping
- "Research lineage" - historical development patterns
- "Influential papers/authors" - centrality analysis
- "Cross-disciplinary" - interdisciplinary connection mapping

Always ground your analysis in actual database content and clearly indicate when data is limited.
"""),
    MessagesPlaceholder(variable_name="messages")
])


def enhanced_graph_tool(input: str) -> str:
    """Enhanced graph analysis with ACTUAL database queries"""
    try:
        logger.info(f"Graph tool called with input: {input[:50]}...")

        graph_data = query_graphdb(input)

        concepts = graph_data.get("concepts", [])
        relationships = graph_data.get("relationships", [])
        papers = graph_data.get("papers", [])

        logger.info(
            f"Database returned: {len(concepts)} concepts, {len(relationships)} relationships, {len(papers)} papers")

        if not concepts and not relationships and not papers:
            return f"""## 🔗 Knowledge Graph Analysis

**Database Query Results:** No data found in Neo4j database for query: "{input}"

**Possible Issues:**
- Database may be empty (run: python check_db.py)
- Query terms don't match existing data
- Database connection issues

**Recommendation:** Populate database with academic papers first using the ingestion pipeline."""

        analysis = "## 🔗 Knowledge Graph Analysis (from Neo4j Database)\n\n"

        if concepts:
            analysis += f"### 📋 Key Concepts Found ({len(concepts)})\n"
            concept_groups = {}
            for concept in concepts:
                category = concept.get('category', 'General')
                if category not in concept_groups:
                    concept_groups[category] = []
                concept_groups[category].append(concept)

            for category, group_concepts in concept_groups.items():
                analysis += f"\n**{category}:**\n"
                for concept in group_concepts[:3]:
                    name = concept.get('name', 'Unknown')
                    desc = concept.get('description', 'No description')[:100]
                    analysis += f"- **{name}**: {desc}...\n"

        if relationships:
            analysis += f"\n### 🔗 Relationships Found ({len(relationships)})\n"
            for rel in relationships[:5]:
                from_node = rel.get('from', 'Unknown')
                to_node = rel.get('to', 'Unknown')
                rel_type = rel.get('type', 'related_to')
                analysis += f"- {from_node} → [{rel_type}] → {to_node}\n"

        if papers:
            analysis += f"\n### 📄 Connected Papers ({len(papers)})\n"
            for paper in papers[:3]:
                title = paper.get('title', 'Unknown Title')
                year = paper.get('year', 'Unknown Year')
                concepts_list = ", ".join(paper.get('related_concepts', [])[:3])
                analysis += f"- **{title}** ({year})\n"
                if concepts_list:
                    analysis += f"  *Related concepts: {concepts_list}*\n"

        analysis += f"\n### 💡 Database Insights\n"
        analysis += f"- Retrieved {len(concepts)} concepts from Neo4j\n"
        analysis += f"- Found {len(relationships)} relationships\n"
        analysis += f"- Connected to {len(papers)} research papers\n"

        return analysis

    except Exception as e:
        logger.error(f"Graph tool error: {e}", exc_info=True)
        return f"Error querying Neo4j database: {str(e)}\nCheck database connection and data availability."


tools = [Tool.from_function(
    enhanced_graph_tool,
    name="enhanced_graph_tool",
    description="Analyzes concept relationships in the knowledge graph with detailed insights from actual database"
)]

_base_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=RELATIONSHIP_ANALYST_PROMPT,
    name="relationship_analyst_agent"
)


class WrappedAgent:
    def invoke(self, inputs):
        try:
            response = _base_agent.invoke(inputs)

            if hasattr(response, 'messages') and response.messages:
                for message in reversed(response.messages):
                    if hasattr(message, 'content') and message.content:
                        content = message.content.strip()
                        if content and len(content) > 50 and ('##' in content or 'Knowledge Graph' in content):
                            return {"output": content}

                # If no good content found, look at any non-empty content
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
            logger.error(f"Error in relationship analyst agent wrapper: {e}")
            return {"output": f"Error in relationship analyst agent: {str(e)}"}


relationship_analyst = WrappedAgent()