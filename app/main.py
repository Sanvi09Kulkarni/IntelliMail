from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Email Analyzer")

app.include_router(router)