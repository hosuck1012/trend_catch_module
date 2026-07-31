from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends

from app.database import get_db
from app.models.source_document import SourceDocument
from app.schemas.collection import DocumentCountResponse, MockCollectionResponse
from app.services.mock_data_service import collect_mock_data


router = APIRouter(prefix="/api", tags=["collection"])


@router.post("/collect/mock", response_model=MockCollectionResponse)
def collect_mock(session: Session = Depends(get_db)) -> dict[str, object]:
    result = collect_mock_data(session)
    return {
        "status": "ok",
        "inserted_documents": result.inserted_documents,
        "skipped_documents": result.skipped_documents,
        "date_range": {
            "start": result.start_date.isoformat(),
            "end": result.end_date.isoformat(),
        },
    }


@router.get("/documents/count", response_model=DocumentCountResponse)
def count_documents(session: Session = Depends(get_db)) -> dict[str, int]:
    total_documents = session.scalar(select(func.count(SourceDocument.id))) or 0
    source_counts = dict(
        session.execute(
            select(SourceDocument.source, func.count(SourceDocument.id)).group_by(SourceDocument.source)
        ).all()
    )
    return {
        "total_documents": total_documents,
        "youtube": source_counts.get("youtube", 0),
        "naver_news": source_counts.get("naver_news", 0),
    }
