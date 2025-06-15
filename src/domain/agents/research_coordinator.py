# src/domain/agents/research_coordinator.py
import logging
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.domain.agents.relationship_analyst import relationship_analyst
from src.domain.agents.theme_analyst import theme_analyst

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("research_coordinator")


class ResearchState(MessagesState):
    """Modern state schema inheriting from MessagesState"""
    query_type: str = "unknown"
    analysis_plan: str = ""
    needs_relationship: bool = False
    needs_theme: bool = False
    transfer_context: str = ""


def query_classification_node(state: ResearchState) -> Command[Literal["planning", "direct_response"]]:
    """Classifies queries and determines routing strategy."""
    last_message = state["messages"][-1].content if state["messages"] else ""
    logger.info(f"Classifying query: {last_message[:50]}...")

    try:
        model = ChatOpenAI(model="gpt-4")

        classification_prompt = [
            {"role": "system", "content": """
You are a research query classifier. Analyze queries and respond with JSON:

{
  "classification": "GREETING|SIMPLE_QUESTION|RESEARCH_QUERY", 
  "needs_relationship": true/false,
  "needs_theme": true/false,
  "reasoning": "brief explanation"
}

Classifications:
- GREETING: Simple greetings ("hi", "hello")
- SIMPLE_QUESTION: Basic questions not requiring research analysis
- RESEARCH_QUERY: Complex academic questions needing specialist analysis

For RESEARCH_QUERY, determine what analysis is needed:
- needs_relationship: Query about connections, citations, networks, influences
- needs_theme: Query about topics, patterns, trends, themes
"""},
            {"role": "user", "content": f"Classify: {last_message}"}
        ]

        response = model.invoke(classification_prompt)

        # Parse JSON response
        import json
        try:
            result = json.loads(response.content)
        except:
            # Fallback parsing
            result = {
                "classification": "RESEARCH_QUERY",
                "needs_relationship": True,
                "needs_theme": True,
                "reasoning": "Failed to parse, defaulting to full research"
            }

        classification = result.get("classification", "RESEARCH_QUERY")

        if classification in ["GREETING", "SIMPLE_QUESTION"]:
            # Handle simple queries directly
            return Command(
                goto="direct_response",
                update={
                    "query_type": classification,
                    "needs_relationship": False,
                    "needs_theme": False
                }
            )
        else:
            # Route to planning for research queries
            return Command(
                goto="planning",
                update={
                    "query_type": classification,
                    "needs_relationship": result.get("needs_relationship", True),
                    "needs_theme": result.get("needs_theme", True)
                }
            )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        # Default to full research on error
        return Command(
            goto="planning",
            update={
                "query_type": "RESEARCH_QUERY",
                "needs_relationship": True,
                "needs_theme": True
            }
        )


def planning_node(state: ResearchState) -> Command[Literal["relationship_analyst", "theme_analyst", "synthesis"]]:
    """Creates analysis plan and routes to appropriate specialists."""
    query = state["messages"][-1].content if state["messages"] else ""
    logger.info(f"Planning analysis for: {query[:50]}...")

    try:
        model = ChatOpenAI(model="gpt-4")

        planning_prompt = [
            {"role": "system", "content": """
You are a Research Coordinator planning academic analysis. Create a brief 2-3 sentence plan.

Available specialists:
- Relationship Analyst: Maps connections between papers, authors, concepts using Neo4j
- Theme Analyst: Identifies patterns, topics, trends using MongoDB

Respond with JSON:
{
  "plan": "2-3 sentence analysis plan",
  "start_with": "relationship_analyst|theme_analyst|both"
}
"""},
            {"role": "user", "content": f"Plan analysis for: {query}"}
        ]

        response = model.invoke(planning_prompt)

        try:
            import json
            result = json.loads(response.content)
            plan = result.get("plan", f"Comprehensive analysis of: {query}")
            start_with = result.get("start_with", "both")
        except:
            plan = f"Comprehensive analysis of: {query}"
            start_with = "both"

        # Determine routing based on needs and plan
        if state.get("needs_relationship") and state.get("needs_theme"):
            if start_with == "theme_analyst":
                next_agent = "theme_analyst"
            else:
                next_agent = "relationship_analyst"  # Default start
        elif state.get("needs_relationship"):
            next_agent = "relationship_analyst"
        elif state.get("needs_theme"):
            next_agent = "theme_analyst"
        else:
            next_agent = "synthesis"

        return Command(
            goto=next_agent,
            update={
                "analysis_plan": plan,
                "messages": [AIMessage(
                    content=f"🎯 Analysis Plan: {plan}",
                    name="research_coordinator"
                )]
            }
        )

    except Exception as e:
        logger.error(f"Planning error: {e}")
        return Command(
            goto="relationship_analyst",
            update={"analysis_plan": f"Analysis of: {query}"}
        )


def direct_response_node(state: ResearchState) -> Command[Literal[END]]:
    """Handles simple queries that don't need specialist analysis."""
    query = state["messages"][-1].content if state["messages"] else ""
    query_type = state.get("query_type", "unknown")

    try:
        model = ChatOpenAI(model="gpt-4")

        if query_type == "GREETING":
            response_content = """Hello! I'm your Research Coordinator, leading a team of specialized analysts.

🔗 **Relationship Analyst** - Maps connections between papers, authors, and concepts
📊 **Theme Analyst** - Identifies patterns and themes across research

Examples of research questions I can help with:
• "How do neural networks connect to medical diagnosis research?"
• "What are the main themes in climate change adaptation papers?"
• "Show me the research lineage of transformer architectures"

What research topic would you like me to analyze today?"""

        else:  # SIMPLE_QUESTION
            simple_prompt = [
                {"role": "system", "content": """
You are a helpful research assistant. Provide a brief, direct answer to simple questions.
If the question would benefit from deeper research analysis using academic databases,
suggest that the user ask a more specific research-oriented question.
"""},
                {"role": "user", "content": query}
            ]

            response = model.invoke(simple_prompt)
            response_content = response.content

        return Command(
            goto=END,
            update={
                "messages": [AIMessage(
                    content=response_content,
                    name="research_coordinator"
                )]
            }
        )

    except Exception as e:
        logger.error(f"Direct response error: {e}")
        return Command(
            goto=END,
            update={
                "messages": [AIMessage(
                    content="I apologize, but I encountered an error. Please try rephrasing your question.",
                    name="research_coordinator"
                )]
            }
        )


def relationship_analyst_node(state: ResearchState) -> Command[Literal["theme_analyst", "synthesis"]]:
    """Executes relationship analysis and routes to next step."""
    query = state["messages"][-1].content if state["messages"] else ""
    logger.info("Starting relationship analysis...")

    try:
        # Get analysis from specialist
        result = relationship_analyst.invoke({
            "messages": [HumanMessage(content=query)]
        })

        analysis_content = result.get("output", "No relationship analysis available")

        # Add analysis to message history with agent name
        analysis_message = AIMessage(
            content=analysis_content,
            name="relationship_analyst"
        )

        # Determine the next step
        if state.get("needs_theme", False):
            next_step = "theme_analyst"
        else:
            next_step = "synthesis"

        return Command(
            goto=next_step,
            update={"messages": [analysis_message]}
        )

    except Exception as e:
        logger.error(f"Relationship analysis error: {e}")
        error_message = AIMessage(
            content=f"Relationship analysis encountered an error: {str(e)}",
            name="relationship_analyst"
        )

        next_step = "theme_analyst" if state.get("needs_theme", False) else "synthesis"
        return Command(
            goto=next_step,
            update={"messages": [error_message]}
        )


def theme_analyst_node(state: ResearchState) -> Command[Literal["relationship_analyst", "synthesis"]]:
    """Executes theme analysis and routes to next step."""
    query = state["messages"][-1].content if state["messages"] else ""
    logger.info("Starting theme analysis...")

    try:
        # Get analysis from specialist
        result = theme_analyst.invoke({
            "messages": [HumanMessage(content=query)]
        })

        analysis_content = result.get("output", "No theme analysis available")

        # Add analysis to message history with agent name
        analysis_message = AIMessage(
            content=analysis_content,
            name="theme_analyst"
        )

        # Determine the next step
        if state.get("needs_relationship", False) and not any(
                msg.name == "relationship_analyst" for msg in state["messages"]
                if hasattr(msg, 'name')
        ):
            next_step = "relationship_analyst"
        else:
            next_step = "synthesis"

        return Command(
            goto=next_step,
            update={"messages": [analysis_message]}
        )

    except Exception as e:
        logger.error(f"Theme analysis error: {e}")
        error_message = AIMessage(
            content=f"Theme analysis encountered an error: {str(e)}",
            name="theme_analyst"
        )

        next_step = "synthesis"
        return Command(
            goto=next_step,
            update={"messages": [error_message]}
        )


def synthesis_node(state: ResearchState) -> Command[Literal[END]]:
    """Synthesizes all analysis results into final response."""
    logger.info("Starting synthesis of specialist outputs...")

    try:
        # Extract messages from specialists
        relationship_messages = [
            msg for msg in state["messages"]
            if hasattr(msg, 'name') and msg.name == "relationship_analyst"
        ]
        theme_messages = [
            msg for msg in state["messages"]
            if hasattr(msg, 'name') and msg.name == "theme_analyst"
        ]

        original_query = state["messages"][0].content if state["messages"] else "Unknown query"
        plan = state.get("analysis_plan", "Research analysis")

        model = ChatOpenAI(model="gpt-4")

        synthesis_prompt = [
            {"role": "system", "content": """
You are a Research Coordinator synthesizing insights from specialist analysts.
Create a comprehensive, well-structured response that integrates findings.

Structure your response with:
1. Brief summary of key findings
2. Integration of relationship insights (if available)
3. Integration of thematic insights (if available)  
4. Synthesis highlighting how the analyses complement each other
5. Actionable conclusions or recommendations

Be authoritative but accessible. Focus on insights that directly answer the research question.
"""},
            {"role": "user", "content": f"""
Research Question: {original_query}

Analysis Plan: {plan}

Relationship Analysis:
{relationship_messages[-1].content if relationship_messages else "No relationship analysis performed"}

Theme Analysis:
{theme_messages[-1].content if theme_messages else "No theme analysis performed"}

Please provide a comprehensive synthesis of these insights.
"""}
        ]

        response = model.invoke(synthesis_prompt)

        # Create final formatted response
        formatted_response = f"""# 🎯 Research Analysis Results

**Query:** {original_query}

---

{response.content}

---

### 📈 System Performance
- **Specialists Used:** {', '.join([
            'Relationship Analyst' if relationship_messages else '',
            'Theme Analyst' if theme_messages else ''
        ]).strip(', ')}
- **Databases Queried:** Neo4j, MongoDB, ChromaDB
- **Analysis Quality:** {'High Confidence' if (relationship_messages and theme_messages) else 'Moderate Confidence'}
"""

        return Command(
            goto=END,
            update={
                "messages": [AIMessage(
                    content=formatted_response,
                    name="research_coordinator"
                )]
            }
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}")

        # Fallback synthesis
        fallback_response = f"""# 🎯 Research Analysis Results

**Query:** {original_query}

I encountered an error during synthesis, but here's what I gathered:

{'**Relationship Analysis:**' + chr(10) + relationship_messages[-1].content + chr(10) + chr(10) if relationship_messages else ''}
{'**Theme Analysis:**' + chr(10) + theme_messages[-1].content + chr(10) + chr(10) if theme_messages else ''}

**Error:** {str(e)}
"""

        return Command(
            goto=END,
            update={
                "messages": [AIMessage(
                    content=fallback_response,
                    name="research_coordinator"
                )]
            }
        )


# Build the modern workflow
logger.info("Building modern research coordinator workflow...")

workflow = StateGraph(ResearchState)

workflow.add_node("query_classification", query_classification_node)
workflow.add_node("planning", planning_node)
workflow.add_node("direct_response", direct_response_node)
workflow.add_node("relationship_analyst", relationship_analyst_node)
workflow.add_node("theme_analyst", theme_analyst_node)
workflow.add_node("synthesis", synthesis_node)

# Set entry point
workflow.add_edge(START, "query_classification")

# Compile the graph
logger.info("Compiling modern research coordinator workflow...")
research_coordinator_graph = workflow.compile()


def run_research_coordinator(query: str) -> ResearchState:
    """
    Main function to run the modern research coordinator workflow.

    Args:
        query: The research question or query from the user

    Returns:
        ResearchState containing all messages and analysis results
    """
    logger.info(f"Starting modern research coordination for: {query[:50]}...")

    try:
        result = research_coordinator_graph.invoke({
            "messages": [HumanMessage(content=query)]
        })
        logger.info("Research coordination completed successfully")
        return result

    except Exception as e:
        logger.error(f"Research coordination failed: {str(e)}", exc_info=True)
        return {
            "messages": [
                HumanMessage(content=query),
                AIMessage(
                    content=f"Error in research coordination: {e}",
                    name="research_coordinator"
                )
            ],
            "query_type": "ERROR"
        }


def process_query_direct(query: str) -> dict:
    """
    Direct processing function for API endpoints.

    Args:
        query: The research question from the user

    Returns:
        Formatted response dictionary for API consumption
    """
    try:
        logger.info(f"Direct processing for query: {query[:50]}...")

        result = run_research_coordinator(query)

        # Extract final message
        final_message = result["messages"][-1] if result["messages"] else None
        final_content = final_message.content if final_message else "No output generated"

        # Count specialist usage
        relationship_used = any(
            hasattr(msg, 'name') and msg.name == "relationship_analyst"
            for msg in result["messages"]
        )
        theme_used = any(
            hasattr(msg, 'name') and msg.name == "theme_analyst"
            for msg in result["messages"]
        )

        response = {
            "status": "success",
            "message": final_content,
            "query": query,
            "query_type": result.get("query_type", "unknown"),
            "specialists_used": {
                "relationship_analyst": relationship_used,
                "theme_analyst": theme_used
            },
            "system_health": {
                "relationship_analyst": "✅ Active" if relationship_used else "⚪ Not Used",
                "theme_analyst": "✅ Active" if theme_used else "⚪ Not Used",
                "database_usage": "✅ High" if (relationship_used and theme_used) else "🟡 Partial" if (
                            relationship_used or theme_used) else "❌ Low",
                "response_quality": "Database-driven" if (relationship_used or theme_used) else "General knowledge"
            }
        }

        logger.info("Direct processing completed successfully")
        return response

    except Exception as e:
        logger.error(f"Direct processing failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing query: {e}",
            "query": query
        }