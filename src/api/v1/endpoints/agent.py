from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.supervisor import run_supervisor

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/agent")
def process_query(request: QueryRequest):
    """
    API endpoint to process a user query using the supervisor agent.

    Args:
        request (QueryRequest): JSON payload containing the user query.

    Returns:
        dict: The AI-generated response formatted for a chat interface.
    """
    try:
        # Run the supervisor workflow
        response = run_supervisor(request.query)

        # Extract the final output for the primary response
        final_output = response.get("final_output", "No response generated.")

        # Return a chat-friendly response structure
        return {
            "status": "success",
            "message": final_output,
            "debug": {
                "graph_output": response.get("graph_output", "No graph output"),
                "tm_output": response.get("tm_output", "No topic modeling output")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
