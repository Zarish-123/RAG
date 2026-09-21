from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="RAG Chatbot API"
)


app.include_router(router)