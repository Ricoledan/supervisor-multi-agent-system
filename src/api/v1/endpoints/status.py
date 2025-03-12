from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def read_root():
    return {
        "description": "This project implements a Hierarchical Multi-Agent System (MAS) architecture featuring a central supervisor agent that orchestrates specialized subordinate agents, served through a FastAPI interface."
    }


@router.get("/status")
def read_status():
    return {"status": "ok"}
