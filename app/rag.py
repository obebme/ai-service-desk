from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.gemini import embed


async def search(db: Session, query: str, limit: int = 5) -> list[dict]:
    vector = await embed(query, task_type="RETRIEVAL_QUERY")
    rows = db.execute(text("""
      SELECT c.content, d.title, d.source_name, 1 - (c.embedding <=> CAST(:embedding AS vector)) AS score
      FROM knowledge_chunks c JOIN knowledge_documents d ON d.id = c.document_id
      WHERE d.status = 'indexed' AND c.embedding_model = :model
      ORDER BY c.embedding <=> CAST(:embedding AS vector) LIMIT :limit
    """), {"embedding": str(vector), "limit": limit,
            "model": settings.gemini_embedding_model}).mappings()
    return [dict(row) for row in rows]
