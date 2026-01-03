from fastapi import FastAPI

from server.routes import router

app = FastAPI(title="NLP Microservice", version="1.0.0")
app.include_router(router)
