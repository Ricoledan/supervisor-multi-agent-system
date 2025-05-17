from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()


def get_openai_model():
    """Returns a configured ChatOpenAI model using environment variables"""
    api_key_str = os.getenv("OPENAI_API_KEY")
    if not api_key_str:
        raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in your environment.")

    openai_api_key = SecretStr(api_key_str)
    return ChatOpenAI(api_key=openai_api_key)
