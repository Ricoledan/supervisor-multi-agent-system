from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from src.domain.prompts.agent_prompts import TOPIC_MODEL_AGENT_PROMPT

model = ChatOpenAI(model="gpt-4")

def dummy_topic_tool(input: str) -> str:
    return "Topic model output"

tools = [Tool.from_function(dummy_topic_tool, name="topic_tool", description="Handles topic modeling")]

topic_model_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=TOPIC_MODEL_AGENT_PROMPT,
    name="topic_model_agent"
)