import asyncio

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.gemini import embed
from app.models import KnowledgeChunk


async def main():
    with SessionLocal() as db:
        chunks = list(db.scalars(select(KnowledgeChunk).order_by(
            KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)))
        try:
            for index, chunk in enumerate(chunks, start=1):
                chunk.embedding = await embed(chunk.content, task_type="RETRIEVAL_DOCUMENT")
                chunk.embedding_model = settings.gemini_embedding_model
                if index % 20 == 0:
                    db.flush()
                    print(f"Re-embedded {index}/{len(chunks)} chunks")
            db.commit()
            print(f"Re-embedded {len(chunks)} knowledge chunks with {settings.gemini_embedding_model}")
        except Exception:
            db.rollback()
            raise


if __name__ == "__main__":
    asyncio.run(main())
