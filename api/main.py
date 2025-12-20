from fastapi import FastAPI
from api.inference.router import router

app = FastAPI(title="MicroLLM API")
app.include_router(router)
