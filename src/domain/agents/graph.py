from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

# This agent explicitly has access to the neo4j instance

model = ChatOpenAI()


def graph_agent(state: Annotated[dict, InjectedState]):
    """
    This agent processes the state for graph database algorithms and responds with the results of the graph computations.
    """
    response = model.invoke(...)
    return response.content
