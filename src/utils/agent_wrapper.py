import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_agent_response(response: Any, content_indicators: Optional[list] = None) -> Dict[str, str]:
    """
    Extract clean content from LangGraph agent response.

    Args:
        response: The response from a LangGraph agent
        content_indicators: List of strings that indicate meaningful content (e.g., ['##', 'Analysis'])

    Returns:
        Dictionary with 'output' key containing the extracted content
    """
    if content_indicators is None:
        content_indicators = ['##', 'Analysis', 'Results']

    try:
        if hasattr(response, 'messages') and response.messages:
            # First pass: Look for high-quality content with indicators
            for message in reversed(response.messages):
                if hasattr(message, 'content') and message.content:
                    content = message.content.strip()
                    if (content and len(content) > 50 and
                            any(indicator in content for indicator in content_indicators)):
                        return {"output": content}

            for message in reversed(response.messages):
                if hasattr(message, 'content') and message.content:
                    content = message.content.strip()
                    if content and len(content) > 20:
                        return {"output": content}

        if hasattr(response, 'content'):
            return {"output": response.content}

        if isinstance(response, dict) and "output" in response:
            return response

        if isinstance(response, str):
            return {"output": response}

        return {"output": str(response)}

    except Exception as e:
        logger.error(f"Error extracting agent response: {e}")
        return {"output": f"Error extracting agent response: {str(e)}"}


class AgentWrapper:
    """
    Generic wrapper for LangGraph agents that standardizes response extraction.
    """

    def __init__(self, agent, content_indicators: Optional[list] = None, agent_name: str = "agent"):
        """
        Initialize the agent wrapper.

        Args:
            agent: The LangGraph agent to wrap
            content_indicators: List of strings that indicate meaningful content
            agent_name: Name of the agent for logging purposes
        """
        self.agent = agent
        self.content_indicators = content_indicators or ['##', 'Analysis', 'Results']
        self.agent_name = agent_name

    def invoke(self, inputs) -> Dict[str, str]:
        """
        Invoke the wrapped agent and extract clean response.

        Args:
            inputs: Input parameters for the agent

        Returns:
            Dictionary with 'output' key containing the extracted content
        """
        try:
            logger.debug(f"Invoking {self.agent_name} agent")
            response = self.agent.invoke(inputs)

            result = extract_agent_response(response, self.content_indicators)
            logger.debug(f"{self.agent_name} agent response extracted successfully")

            return result

        except Exception as e:
            logger.error(f"Error in {self.agent_name} agent wrapper: {e}")
            return {"output": f"Error in {self.agent_name} agent: {str(e)}"}


def wrap_agent(agent, content_indicators: Optional[list] = None, agent_name: str = "agent") -> AgentWrapper:
    """
    Quick wrapper function for agents.

    Args:
        agent: The LangGraph agent to wrap
        content_indicators: List of strings that indicate meaningful content
        agent_name: Name of the agent for logging

    Returns:
        Wrapped agent instance
    """
    return AgentWrapper(agent, content_indicators, agent_name)