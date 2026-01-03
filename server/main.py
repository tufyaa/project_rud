"""FastAPI application entrypoint."""
from fastapi import FastAPI

from server.api.routes import router

app = FastAPI(title="NLP Microservice", version="1.0.0")
app.include_router(router)
