from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.domain.agents.supervisor import supervisor

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
        dict: The AI-generated response content.
    """
    try:
        response = supervisor.invoke({"messages": [{"role": "user", "content": request.query}]})

        ai_response = next(
            (msg.content for msg in response["messages"] if getattr(msg, "type", None) == "ai"),
            "No valid response"
        )

        return {"response": ai_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))