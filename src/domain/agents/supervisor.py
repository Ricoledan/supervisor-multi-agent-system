import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from src.domain.agents.graph_writer import graph_writer_agent
from src.domain.agents.topic_model import topic_model_agent
from src.domain.formatters.response_formatter import EnhancedResponseFormatter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("supervisor")


class AgentState(TypedDict, total=False):
    input: str
    query_type: str
    plan: Optional[str]
    plan_requires_graph: Optional[bool]
    plan_requires_topics: Optional[bool]
    context: Optional[dict]
    graph_output: Optional[str]
    tm_output: Optional[str]
    final_output: Optional[str]
    structured_data: Optional[dict]


def retrieval_node(state: AgentState) -> AgentState:
    """Prepares the context for agents to retrieve from their databases directly"""
    query = state["input"]
    logger.info(f"Setting up context for query: {query[:50]}...")

    state["context"] = {"query": query}

    if state.get("query_type") != "RESEARCH_QUERY":
        logger.info("Skipping retrieval for non-research query")

    return state


def planning_node(state: AgentState) -> AgentState:
    """Plans the approach based on the query."""
    query = state["input"]
    logger.info(f"Planning approach for query: {query[:50]}...")

    try:
        from src.utils.model_init import get_openai_model

        model = get_openai_model()

        classification_prompt = [
            {"role": "system", "content":
                "You are a query classifier. Classify the user query into one of these categories:\n"
                "1. GREETING: Simple greetings like 'hi', 'hello', 'hey'\n"
                "2. SIMPLE_QUESTION: Basic questions that don't need complex analysis\n"
                "3. RESEARCH_QUERY: Complex questions requiring research and analysis\n"
                "Respond with ONLY the category name."
             },
            {"role": "user", "content": f"Classify this query: {query}"}
        ]

        classification = model.invoke(classification_prompt).content.strip()
        logger.info(f"Query classified as: {classification}")

        state["query_type"] = classification

        if classification in ["GREETING", "SIMPLE_QUESTION"]:
            state["plan"] = f"This is a {classification}. Provide a direct response."
            state["plan_requires_graph"] = False
            state["plan_requires_topics"] = False

            if classification == "GREETING":
                state["final_output"] = (
                    f"Hello! I'm a research assistant that can help analyze complex topics using knowledge graphs "
                    f"and topic modeling. Here are examples of questions I can help with:\n\n"
                    f"- \"What are the key themes and relationships in climate change research papers?\"\n"
                    f"- \"How do different AI technologies connect to neurobiological concepts?\"\n"
                    f"- \"What are the main research clusters in COVID-19 vaccine development?\"\n\n"
                    f"What research topic would you like me to analyze today?"
                )
        else:
            from src.domain.prompts.agent_prompts import SUPERVISOR_PROMPT

            planning_messages = SUPERVISOR_PROMPT.format_messages(messages=[
                {"role": "user", "content": f"Plan how to answer this research query: {query}"}
            ])

            planning_response = model.invoke(planning_messages)
            plan = planning_response.content
            logger.info(f"Received plan from LLM: {plan[:100]}...")

            state["plan"] = plan
            state["plan_requires_graph"] = any(keyword in plan.lower() for keyword in
                                               ["knowledge graph", "connections", "relationships", "entities"])
            state["plan_requires_topics"] = any(keyword in plan.lower() for keyword
                                                in ["topic", "themes", "clusters"])
    except Exception as e:
        logger.error(f"Error in planning node: {str(e)}", exc_info=True)
        state["plan"] = f"Error in planning: {e}"
        state["plan_requires_graph"] = True
        state["plan_requires_topics"] = True

    return state


def graph_node(state: AgentState) -> AgentState:
    """Pass the query to graph_writer_agent which will access the database directly"""
    text = state["input"]
    logger.info("Starting graph node processing")

    try:
        logger.info("Invoking graph_writer_agent")
        out = graph_writer_agent.invoke({
            "messages": [{"role": "user", "content": text}]
        })
        state["graph_output"] = out.get("output", "No graph output")
        logger.info(f"Graph analysis complete: {len(state['graph_output'])} chars output")
        logger.debug(f"Graph output preview: {state['graph_output'][:100]}...")
    except Exception as e:
        logger.error(f"Error in graph node: {str(e)}", exc_info=True)
        state["graph_output"] = f"Error from graph_writer_agent: {e}"

    return state


def topic_model_node(state: AgentState) -> AgentState:
    """Pass the query to topic_model_agent which will access the database directly"""
    text = state["input"]
    logger.info("Starting topic model node processing")

    try:
        logger.info("Invoking topic_model_agent")
        out = topic_model_agent.invoke({
            "messages": [{"role": "user", "content": text}]
        })
        state["tm_output"] = out.get("output", "No topic modeling output")
        logger.info(f"Topic modeling complete: {len(state['tm_output'])} chars output")
        logger.debug(f"Topic output preview: {state['tm_output'][:100]}...")
    except Exception as e:
        logger.error(f"Error in topic model node: {str(e)}", exc_info=True)
        state["tm_output"] = f"Error from topic_model_agent: {e}"

    return state


def synthesize_node(state: AgentState) -> AgentState:
    """Combines the outputs from graph and topic model agents into a final response."""
    logger.info("Starting synthesis node processing")

    state["debug"] = {
        "graph_output": state.get("graph_output", "No graph output"),
        "tm_output": state.get("tm_output", "No topic modeling output")
    }

    if state.get("final_output"):
        logger.info("Using pre-generated response for simple query")
        return state

    original_query = state["input"]
    query_type = state.get("query_type", "RESEARCH_QUERY")
    graph_output = state.get("graph_output", "No graph analysis was performed.")
    tm_output = state.get("tm_output", "No topic modeling was performed.")
    plan = state.get("plan", "No explicit planning was performed.")

    if query_type == "SIMPLE_QUESTION":
        try:
            from src.utils.model_init import get_openai_model

            model = get_openai_model()
            simple_prompt = [
                {"role": "system", "content":
                    "You are a helpful assistant. Provide a brief, direct answer to the question. "
                    "If the question would benefit from deeper research analysis, suggest that "
                    "the user ask a more specific research-oriented question."
                 },
                {"role": "user", "content": original_query}
            ]

            response = model.invoke(simple_prompt)
            state["final_output"] = response.content
            return state

        except Exception as e:
            logger.error(f"Error handling simple question: {str(e)}", exc_info=True)
            state["final_output"] = f"I couldn't process your question properly. Could you try rephrasing it?"
            return state
    else:
        try:
            from src.utils.model_init import get_openai_model

            model = get_openai_model()
            synthesis_prompt = [
                {"role": "system", "content":
                    "You are a research synthesis assistant that creates comprehensive responses based on "
                    "knowledge graph analysis and topic modeling results. Integrate the insights from both "
                    "sources to create a coherent, informative response."
                 },
                {"role": "user", "content":
                    f"Question: {original_query}\n\n"
                    f"Analysis Plan: {plan}\n\n"
                    f"Knowledge Graph Analysis: {graph_output}\n\n"
                    f"Topic Modeling Results: {tm_output}\n\n"
                    f"Please provide a comprehensive research response that synthesizes these insights."
                 }
            ]

            response = model.invoke(synthesis_prompt)
            state["final_output"] = response.content

            # Apply enhanced formatting if we have the necessary outputs
            if state.get("graph_output") and state.get("tm_output"):
                try:
                    from src.domain.formatters.response_formatter import EnhancedResponseFormatter

                    agent_responses = {
                        "graph_output": state.get("graph_output"),
                        "tm_output": state.get("tm_output")
                    }

                    formatted_result = EnhancedResponseFormatter.format_response(agent_responses, original_query)

                    # Safely update state with formatted result
                    if "message" in formatted_result:
                        state["final_output"] = formatted_result["message"]

                    # Only set structured_data if it exists in the result
                    if "structured_data" in formatted_result:
                        state["structured_data"] = formatted_result["structured_data"]

                    logger.info("Successfully applied enhanced formatting to response")

                except Exception as e:
                    logger.error(f"Error applying enhanced formatting: {str(e)}", exc_info=True)
                    # Keep the original response if formatting fails
                    logger.info("Continuing with basic synthesis response")

        except Exception as e:
            logger.error(f"Error synthesizing research response: {str(e)}", exc_info=True)
            state["final_output"] = (
                f"I apologize, but I encountered an error while analyzing your research query. "
                f"Error details: {str(e)}"
            )

    return state


logger.info("Initializing supervisor workflow")
workflow = StateGraph(AgentState)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("planning", planning_node)
workflow.add_node("graph_node", graph_node)
workflow.add_node("topic_model_node", topic_model_node)
workflow.add_node("synthesize_node", synthesize_node)

workflow.set_entry_point("planning")

logger.info("Setting up conditional workflow edges")

workflow.add_conditional_edges(
    "planning",
    lambda state: state.get("final_output") is None and state.get("query_type") == "RESEARCH_QUERY",
    {
        True: "retrieval",
        False: "synthesize_node"
    }
)

workflow.add_conditional_edges(
    "retrieval",
    lambda state: state.get("plan_requires_graph", False) or state.get("plan_requires_topics", False),
    {
        True: "graph_node",
        False: "synthesize_node"
    }
)

workflow.add_conditional_edges(
    "graph_node",
    lambda state: state.get("plan_requires_topics", False),
    {
        True: "topic_model_node",
        False: "synthesize_node"
    }
)

workflow.add_edge("topic_model_node", "synthesize_node")
workflow.add_edge("synthesize_node", END)

logger.info("Compiling workflow graph")
graph_executor = workflow.compile()


def run_supervisor(query: str) -> AgentState:
    """Runs the supervisor agent to process a query with dynamic planning."""
    logger.info(f"Starting supervisor workflow for query: {query[:50]}...")

    try:
        result = graph_executor.invoke({"input": query})
        logger.info("Supervisor workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Supervisor workflow failed: {str(e)}", exc_info=True)
        return {
            "input": query,
            "final_output": f"Error executing supervisor workflow: {e}"
        }


def process_query(query: str) -> dict:
    """Direct processing function that uses both agents and formats the response."""
    try:
        logger.info(f"Direct processing for query: {query[:50]}...")
        graph_agent_response = graph_writer_agent.invoke(query)
        tm_agent_response = topic_model_agent.invoke(query)

        agent_responses = {
            "graph_output": graph_agent_response,
            "tm_output": tm_agent_response
        }

        # Format the combined response
        formatted_response = EnhancedResponseFormatter.format_response(agent_responses, query)
        logger.info("Direct processing completed successfully")
        return formatted_response
    except Exception as e:
        logger.error(f"Direct processing failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing query: {e}"
        }