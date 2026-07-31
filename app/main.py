from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.collection import router as collection_router
from app.api.entities import router as entities_router
from app.api.keywords import router as keywords_router
from app.api.newsis_rss import router as newsis_rss_router
from app.api.scheduler import router as scheduler_router
from app.api.search_interest import router as search_interest_router
from app.api.trends import router as trends_router
from app.api.youtube import router as youtube_router
from app.config import get_settings
from app.database import init_db
from app.scheduler.scheduler_manager import scheduler_manager


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    await scheduler_manager.start()
    try:
        yield
    finally:
        await scheduler_manager.shutdown()


app = FastAPI(title=settings.service_name, lifespan=lifespan)
app.include_router(collection_router)
app.include_router(entities_router)
app.include_router(keywords_router)
app.include_router(newsis_rss_router)
app.include_router(scheduler_router)
app.include_router(search_interest_router)
app.include_router(trends_router)
app.include_router(youtube_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.service_name}
