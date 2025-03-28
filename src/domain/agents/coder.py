from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os
from dotenv import load_dotenv

# Load environment variables to get the API key if needed for a separate instance
load_dotenv()

# Retrieve the OpenAI API key and wrap it securely
api_key_str = os.getenv("OPENAI_API_KEY")
if not api_key_str:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in your environment.")
openai_api_key = SecretStr(api_key_str)

# Optionally, create a dedicated ChatOpenAI model instance for this agent (or reuse the one from the supervisor)
model = ChatOpenAI(api_key=openai_api_key)


def coder_agent(input_data: dict) -> dict:
    """
    Processes a coding-related query and returns generated code.

    Args:
        input_data (dict): A dictionary containing the key "input" with the user query.

    Returns:
        dict: A dictionary containing the result, e.g., generated code.
    """
    query = input_data.get("input", "")
    if not query:
        return {"error": "No input provided to coder_agent."}

    # Construct a prompt tailored for code generation tasks
    prompt = f"Generate Python code for the following request:\n\n{query}\n\n# Python code:"

    try:
        # Invoke the language model with the prompt
        response = model([{"role": "user", "content": prompt}])
        # Extract the generated code from the response (this extraction might vary based on your model's output)
        generated_code = response.choices[0].message["content"]
        return {"result": generated_code}
    except Exception as e:
        return {"error": f"Coder agent failed: {e}"}