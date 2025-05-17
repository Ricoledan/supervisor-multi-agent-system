from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import GRAPH_WRITER_AGENT_PROMPT

model = ChatOpenAI(model="gpt-4")

def dummy_graph_tool(input: str) -> str:
    return "Graph tool output"

tools = [Tool.from_function(dummy_graph_tool, name="graph_tool", description="Handles graph writing")]

graph_writer_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=GRAPH_WRITER_AGENT_PROMPT,
    name="graph_writer_agent"
)