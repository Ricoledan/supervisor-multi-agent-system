from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from src.databases.graph.config import driver, neo4j_config

model = ChatOpenAI()


def graph_agent(state: Annotated[dict, InjectedState], query: str):
    """
    This agent processes the state for graph database algorithms and responds with the results of the graph computations.
    """
    with driver.session(database=neo4j_config.database) as session:
        result = session.run(query)
        response = model.invoke(result)
        return response.content
