from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.collection import router as collection_router
from app.api.keywords import router as keywords_router
from app.api.trends import router as trends_router
from app.config import get_settings
from app.database import init_db


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    yield


app = FastAPI(title=settings.service_name, lifespan=lifespan)
app.include_router(collection_router)
app.include_router(keywords_router)
app.include_router(trends_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}