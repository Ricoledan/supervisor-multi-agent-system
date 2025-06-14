import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from src.domain.agents.relationship_analyst import relationship_analyst
from src.domain.agents.theme_analyst import theme_analyst
from src.domain.formatters.response_formatter import EnhancedResponseFormatter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("research_coordinator")


class ResearchState(TypedDict, total=False):
    input: str
    query_type: str
    plan: Optional[str]
    needs_relationship_analysis: Optional[bool]
    needs_theme_analysis: Optional[bool]
    context: Optional[dict]
    relationship_output: Optional[str]
    theme_output: Optional[str]
    final_output: Optional[str]
    structured_data: Optional[dict]


def query_classification_node(state: ResearchState) -> ResearchState:
    """Classifies the query and determines which specialists are needed."""
    query = state["input"]
    logger.info(f"Classifying query: {query[:50]}...")

    try:
        from src.utils.model_init import get_openai_model
        model = get_openai_model()

        classification_prompt = [
            {"role": "system", "content":
                "You are a research query classifier. Analyze the user's query and classify it:\n\n"
                "GREETING: Simple greetings like 'hi', 'hello', 'hey'\n"
                "SIMPLE_QUESTION: Basic questions not requiring deep research analysis\n"
                "RESEARCH_QUERY: Complex academic research questions requiring analysis\n\n"
                "For RESEARCH_QUERY, also determine what analysis is needed:\n"
                "- RELATIONSHIP_ANALYSIS: Query asks about connections, citations, influences, networks\n"
                "- THEME_ANALYSIS: Query asks about topics, themes, patterns, trends\n"
                "- BOTH: Query needs comprehensive analysis from both perspectives\n\n"
                "Respond in this format:\n"
                "CLASSIFICATION: [GREETING|SIMPLE_QUESTION|RESEARCH_QUERY]\n"
                "ANALYSIS_NEEDED: [RELATIONSHIP_ANALYSIS|THEME_ANALYSIS|BOTH|NONE]"
             },
            {"role": "user", "content": f"Classify this query: {query}"}
        ]

        response = model.invoke(classification_prompt).content.strip()
        logger.info(f"Classification response: {response}")

        # Parse classification
        lines = response.split('\n')
        classification = "RESEARCH_QUERY"  # Default
        analysis_needed = "BOTH"  # Default

        for line in lines:
            if line.startswith("CLASSIFICATION:"):
                classification = line.split(":", 1)[1].strip()
            elif line.startswith("ANALYSIS_NEEDED:"):
                analysis_needed = line.split(":", 1)[1].strip()

        state["query_type"] = classification

        # Set analysis requirements
        if classification == "RESEARCH_QUERY":
            state["needs_relationship_analysis"] = analysis_needed in ["RELATIONSHIP_ANALYSIS", "BOTH"]
            state["needs_theme_analysis"] = analysis_needed in ["THEME_ANALYSIS", "BOTH"]
        else:
            state["needs_relationship_analysis"] = False
            state["needs_theme_analysis"] = False

        logger.info(f"Query classified as: {classification}, Analysis needed: {analysis_needed}")

    except Exception as e:
        logger.error(f"Error in classification: {str(e)}", exc_info=True)
        # Default to research query requiring both analyses
        state["query_type"] = "RESEARCH_QUERY"
        state["needs_relationship_analysis"] = True
        state["needs_theme_analysis"] = True

    return state


def planning_node(state: ResearchState) -> ResearchState:
    """Creates research plan and handles simple queries directly."""
    query = state["input"]
    query_type = state.get("query_type", "RESEARCH_QUERY")

    logger.info(f"Planning approach for {query_type}: {query[:50]}...")

    if query_type == "GREETING":
        state["final_output"] = (
            "Hello! I'm your Research Coordinator, leading a team of specialized analysts.\n\n"
            "🔗 **Relationship Analyst** - Maps connections between papers, authors, and concepts\n"
            "📊 **Theme Analyst** - Identifies patterns and themes across research\n\n"
            "Here are examples of research questions I can help with:\n"
            "• \"How do neural networks connect to medical diagnosis research?\"\n"
            "• \"What are the main themes in climate change adaptation papers?\"\n"
            "• \"Show me the research lineage of transformer architectures\"\n\n"
            "What research topic would you like me to analyze today?"
        )
        return state

    elif query_type == "SIMPLE_QUESTION":
        try:
            from src.utils.model_init import get_openai_model
            model = get_openai_model()

            simple_prompt = [
                {"role": "system", "content":
                    "You are a helpful research assistant. Provide a brief, direct answer to the question. "
                    "If the question would benefit from deeper research analysis using academic databases, "
                    "suggest that the user ask a more specific research-oriented question."
                 },
                {"role": "user", "content": query}
            ]

            response = model.invoke(simple_prompt)
            state["final_output"] = response.content

        except Exception as e:
            logger.error(f"Error handling simple question: {str(e)}", exc_info=True)
            state["final_output"] = (
                "I couldn't process your question properly. Could you try rephrasing it as a specific "
                "research question? For example: 'What are the main approaches to [topic]?' or "
                "'How do [concept A] and [concept B] relate in the literature?'"
            )
        return state

    # For RESEARCH_QUERY, create analysis plan
    try:
        from src.utils.model_init import get_openai_model
        model = get_openai_model()

        planning_prompt = [
            {"role": "system", "content":
                "You are a Research Coordinator planning an academic analysis. Based on the research question, "
                "create a brief plan outlining what insights you'll provide by coordinating specialist analysts.\n\n"
                "Available specialists:\n"
                "- Relationship Analyst: Maps connections between papers, authors, concepts\n"
                "- Theme Analyst: Identifies patterns, topics, and trends\n\n"
                "Create a 2-3 sentence plan describing what analysis approach will best answer the question."
             },
            {"role": "user", "content": f"Create analysis plan for: {query}"}
        ]

        planning_response = model.invoke(planning_prompt)
        state["plan"] = planning_response.content
        logger.info(f"Analysis plan created: {state['plan']}")

    except Exception as e:
        logger.error(f"Error in planning: {str(e)}", exc_info=True)
        state["plan"] = f"Comprehensive analysis of: {query}"

    return state


def relationship_analysis_node(state: ResearchState) -> ResearchState:
    """Analyzes relationships and connections using the Relationship Analyst."""
    query = state["input"]
    logger.info("Starting relationship analysis...")

    try:
        logger.info("Invoking relationship analyst...")
        result = relationship_analyst.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        state["relationship_output"] = result.get("output", "No relationship analysis available")
        logger.info(f"Relationship analysis complete: {len(state['relationship_output'])} chars")

    except Exception as e:
        logger.error(f"Error in relationship analysis: {str(e)}", exc_info=True)
        state["relationship_output"] = f"Error in relationship analysis: {e}"

    return state


def theme_analysis_node(state: ResearchState) -> ResearchState:
    """Analyzes themes and patterns using the Theme Analyst."""
    query = state["input"]
    logger.info("Starting theme analysis...")

    try:
        logger.info("Invoking theme analyst...")
        result = theme_analyst.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        state["theme_output"] = result.get("output", "No theme analysis available")
        logger.info(f"Theme analysis complete: {len(state['theme_output'])} chars")

    except Exception as e:
        logger.error(f"Error in theme analysis: {str(e)}", exc_info=True)
        state["theme_output"] = f"Error in theme analysis: {e}"

    return state


def synthesis_node(state: ResearchState) -> ResearchState:
    """Synthesizes specialist outputs into final comprehensive response."""
    logger.info("Starting synthesis of specialist outputs...")

    # Return early if simple query already handled
    if state.get("final_output"):
        logger.info("Using pre-generated response for simple query")
        return state

    original_query = state["input"]
    plan = state.get("plan", "Comprehensive research analysis")
    relationship_output = state.get("relationship_output", "No relationship analysis performed")
    theme_output = state.get("theme_output", "No theme analysis performed")

    try:
        from src.utils.model_init import get_openai_model
        model = get_openai_model()

        synthesis_prompt = [
            {"role": "system", "content":
                "You are a Research Coordinator synthesizing insights from specialist analysts. "
                "Create a comprehensive, well-structured response that integrates findings from both "
                "relationship and thematic analysis.\n\n"
                "Structure your response with:\n"
                "1. Brief summary of key findings\n"
                "2. Integration of relationship insights (connections, networks, influences)\n"
                "3. Integration of thematic insights (patterns, topics, trends)\n"
                "4. Synthesis highlighting how the relationship and thematic views complement each other\n"
                "5. Actionable conclusions or recommendations\n\n"
                "Be authoritative but accessible. Focus on insights that directly answer the research question."
             },
            {"role": "user", "content":
                f"Research Question: {original_query}\n\n"
                f"Analysis Plan: {plan}\n\n"
                f"Relationship Analysis:\n{relationship_output}\n\n"
                f"Theme Analysis:\n{theme_output}\n\n"
                f"Please provide a comprehensive synthesis of these insights."
             }
        ]

        response = model.invoke(synthesis_prompt)
        state["final_output"] = response.content

        # Apply enhanced formatting if we have both outputs
        if (state.get("relationship_output") and
                state.get("theme_output") and
                "Error" not in state.get("relationship_output", "") and
                "Error" not in state.get("theme_output", "")):

            try:
                specialist_responses = {
                    "relationship_output": state.get("relationship_output"),
                    "theme_output": state.get("theme_output")
                }

                formatted_result = EnhancedResponseFormatter.format_response(
                    specialist_responses, original_query
                )

                state["final_output"] = formatted_result["message"]
                state["structured_data"] = formatted_result.get("structured_data")
                logger.info("Successfully applied enhanced formatting")

            except Exception as e:
                logger.error(f"Error applying enhanced formatting: {str(e)}", exc_info=True)
                # Keep the basic synthesis if formatting fails

    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}", exc_info=True)
        state["final_output"] = (
            f"I apologize, but I encountered an error while synthesizing the research analysis. "
            f"Here's what I gathered:\n\n"
            f"Relationship Analysis: {relationship_output}\n\n"
            f"Theme Analysis: {theme_output}\n\n"
            f"Error details: {str(e)}"
        )

    logger.info("Synthesis complete")
    return state


# Build the workflow
logger.info("Initializing research coordinator workflow...")
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("query_classification", query_classification_node)
workflow.add_node("planning", planning_node)
workflow.add_node("relationship_analysis", relationship_analysis_node)
workflow.add_node("theme_analysis", theme_analysis_node)
workflow.add_node("synthesis", synthesis_node)

# Set entry point
workflow.set_entry_point("query_classification")

# Define workflow edges
workflow.add_edge("query_classification", "planning")


# Conditional routing after planning
def route_after_planning(state: ResearchState) -> str:
    """Route to appropriate analysis based on query type and requirements."""
    if state.get("final_output"):
        # Simple queries already handled, go to synthesis (which will pass through)
        return "synthesis"
    elif state.get("needs_relationship_analysis") and state.get("needs_theme_analysis"):
        # Need both analyses - start with relationship (theme runs in parallel via separate edge)
        return "relationship_analysis"
    elif state.get("needs_relationship_analysis"):
        return "relationship_analysis"
    elif state.get("needs_theme_analysis"):
        return "theme_analysis"
    else:
        return "synthesis"


workflow.add_conditional_edges(
    "planning",
    route_after_planning,
    {
        "relationship_analysis": "relationship_analysis",
        "theme_analysis": "theme_analysis",
        "synthesis": "synthesis"
    }
)


# For comprehensive analysis, run both specialists in parallel
def route_after_relationship(state: ResearchState) -> str:
    """After relationship analysis, check if theme analysis is also needed."""
    if state.get("needs_theme_analysis"):
        return "theme_analysis"
    else:
        return "synthesis"


workflow.add_conditional_edges(
    "relationship_analysis",
    route_after_relationship,
    {
        "theme_analysis": "theme_analysis",
        "synthesis": "synthesis"
    }
)

# Theme analysis always goes to synthesis
workflow.add_edge("theme_analysis", "synthesis")
workflow.add_edge("synthesis", END)

# Compile the workflow
logger.info("Compiling research coordinator workflow...")
research_coordinator_graph = workflow.compile()


def run_research_coordinator(query: str) -> ResearchState:
    """
    Main function to run the research coordinator workflow.

    Args:
        query: The research question or query from the user

    Returns:
        ResearchState containing all outputs and analysis results
    """
    logger.info(f"Starting research coordination for: {query[:50]}...")

    try:
        result = research_coordinator_graph.invoke({"input": query})
        logger.info("Research coordination completed successfully")
        return result

    except Exception as e:
        logger.error(f"Research coordination failed: {str(e)}", exc_info=True)
        return {
            "input": query,
            "final_output": f"Error in research coordination: {e}",
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

        # Run the full research coordinator workflow
        result = run_research_coordinator(query)

        # Format for API response
        response = {
            "status": "success",
            "message": result.get("final_output", "No output generated"),
            "query": query,
            "query_type": result.get("query_type", "unknown"),
            "specialists_used": {
                "relationship_analyst": bool(result.get("relationship_output")),
                "theme_analyst": bool(result.get("theme_output"))
            }
        }

        # Add structured data if available
        if result.get("structured_data"):
            response["structured_data"] = result["structured_data"]

        # Add debug info if available
        if result.get("relationship_output") or result.get("theme_output"):
            response["debug"] = {
                "has_relationship_output": bool(result.get("relationship_output")),
                "has_theme_output": bool(result.get("theme_output"))
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
