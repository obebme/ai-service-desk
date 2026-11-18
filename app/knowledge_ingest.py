"""Shared document extraction and indexing."""

import io

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.config import settings
from app.gemini import embed
from app.models import KnowledgeChunk, KnowledgeDocument


MAX_UPLOAD_FILE_BYTES = 10_000_000


async def ingest_knowledge_document(file: UploadFile, db: Session) -> dict:
    name = (file.filename or "document").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if len(name) > 240 or "\x00" in name:
        raise HTTPException(400, "Invalid document filename")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if suffix not in {"pdf", "txt", "md"}:
        raise HTTPException(415, "Only PDF, TXT and Markdown files are supported")
    raw = await file.read(MAX_UPLOAD_FILE_BYTES + 1)
    if len(raw) > MAX_UPLOAD_FILE_BYTES:
        raise HTTPException(413, "File exceeds 10 MB limit")
    try:
        content = ("\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
                   if suffix == "pdf" else raw.decode("utf-8-sig"))
    except Exception as exc:
        raise HTTPException(400, "Could not extract document text") from exc
    if not content.strip():
        raise HTTPException(400, "Document contains no extractable text")
    if "\x00" in content:
        raise HTTPException(400, "Document text contains null characters")
    doc = KnowledgeDocument(title=name, source_name=name, status="indexing")
    db.add(doc)
    db.flush()
    chunks = [content[i:i + 1200] for i in range(0, len(content), 1000)]
    try:
        for index, chunk in enumerate(chunks):
            db.add(KnowledgeChunk(document_id=doc.id, chunk_index=index, content=chunk,
                                  embedding=await embed(chunk, task_type="RETRIEVAL_DOCUMENT"),
                                  embedding_model=settings.gemini_embedding_model))
        doc.status = "indexed"
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(503, "Document indexing failed; verify Gemini API key, model and quota")
    return {"id": str(doc.id), "title": doc.title, "status": doc.status, "chunks": len(chunks)}
