from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import GRAPH_WRITER_AGENT_PROMPT
from src.utils.model_init import get_openai_model
from src.databases.graph.config import driver, neo4j_config

model = get_openai_model()


def extract_entities(text: str) -> str:
    """
    Extract entities from academic papers like authors, concepts, methods.

    Args:
        text: Content of papers to analyze

    Returns:
        JSON formatted list of entities with their types
    """
    return f"Extracted entities: [Entity extraction would be performed here]"


def identify_relationships(entities: str) -> str:
    """
    Identify relationships between extracted entities.

    Args:
        entities: JSON string containing entity information

    Returns:
        JSON formatted relationships between entities
    """
    return f"Identified relationships: [Relationship identification would be performed here]"


def generate_cypher(data: str) -> str:
    """
    Generate Neo4j Cypher queries from structured entity and relationship data.

    Args:
        data: JSON string with entities and relationships

    Returns:
        Cypher queries for creating nodes and relationships
    """
    return f"Generated Cypher queries: [Cypher query generation would be performed here]"


def execute_cypher(query: str) -> str:
    """
    Execute Cypher queries against the Neo4j database.

    Args:
        query: Cypher query string

    Returns:
        Results of the query execution
    """
    try:
        with driver.session(database=neo4j_config.database) as session:
            result = session.run(query)
            records = [record.data() for record in result]
            return f"Query executed successfully: {records}"
    except Exception as e:
        return f"Error executing query: {str(e)}"


tools = [
    Tool.from_function(
        name="extract_entities",
        func=extract_entities,
        description="Extract entities like authors, papers, concepts from academic text"
    ),
    Tool.from_function(
        name="identify_relationships",
        func=identify_relationships,
        description="Identify relationships between extracted entities"
    ),
    Tool.from_function(
        name="generate_cypher",
        func=generate_cypher,
        description="Generate Neo4j Cypher queries for nodes and relationships"
    ),
    Tool.from_function(
        name="execute_cypher",
        func=execute_cypher,
        description="Execute Cypher queries against the Neo4j database"
    )
]

graph_agent = create_react_agent(model, tools, prompt=GRAPH_WRITER_AGENT_PROMPT)


def run_graph_writer(query: str) -> dict:
    """
    Executes the graph writer agent with the given query.

    Args:
        query (str): The input prompt to process.

    Returns:
        dict: The structured response from the graph writer agent.
    """
    try:
        print("Running graph writer agent...", flush=True)
        response = graph_agent.invoke({"input": query})
        print("Graph writer response:", response, flush=True)
        return response
    except Exception as e:
        error_message = f"Error during graph writer execution: {e}"
        print(error_message, flush=True)
        return {"error": error_message}