from fastapi import FastAPI
from api.v1.endpoints import status, agent

app = FastAPI()

app.include_router(status.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")