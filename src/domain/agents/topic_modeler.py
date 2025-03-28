from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

model = ChatOpenAI()


def tm_agent(state: Annotated[dict, InjectedState]):
    """
    This agent processes the state for topic modeling and responds with the identified topics.
    """
    response = model.invoke(...)
    return response.content
