from fastapi import APIRouter

router = APIRouter()

@router.get("/agent")
def read_status():
    return {"this is the agent endpoint"}