import asyncio
import secrets
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app import message_processing, panel_auth
from app.ai import analyze
from app.config import settings
from app.db import engine, get_db
from app.knowledge_ingest import ingest_knowledge_document
from app.models import Department, KnowledgeDocument, Message, Ticket, User
from app.routers import panel
from app.schemas import (
    Analysis, MessageCreate, MessageView, TicketCreate, TicketPatch, TicketView,
    WorkflowMessageRequest, describe_responses,
)
from app.telegram import MENU_CALLBACKS, MENU_COMMANDS, TICKET_CALLBACKS, TICKET_NUMBER, replied_ticket_number
from app.telegram_worker import run_telegram_worker
from app.ticket_writes import create_ticket, get_or_create_user, record_ticket_message, record_ticket_update


MAX_UPLOAD_REQUEST_BYTES = 10_250_000
KNOWLEDGE_UPLOAD_PATHS = {"/ui/knowledge/upload", "/api/v1/knowledge/documents"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_session_signing_secret()
    settings.validate_user_api_tokens()
    with Session(engine) as db:
        for code, name in [("network_support", "Network Support"),
                           ("system_administration", "System Administration"),
                           ("hardware_support", "Hardware Support"),
                           ("software_support", "Software Support"),
                           ("security_team", "Security Team")]:
            db.execute(insert(Department).values(code=code, name=name)
                       .on_conflict_do_nothing(index_elements=[Department.code]))
        db.commit()
    worker = asyncio.create_task(run_telegram_worker())
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


app = FastAPI(title="AI Service Desk", version="0.1.0", lifespan=lifespan)
app.mount(
    "/ui/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="panel-static",
)
app.include_router(panel.router)


@app.middleware("http")
async def disable_panel_page_caching(request: Request, call_next):
    if request.method == "POST" and request.url.path in KNOWLEDGE_UPLOAD_PATHS:
        content_lengths = request.headers.getlist("content-length")
        if not content_lengths:
            return PlainTextResponse("Content-Length is required for uploads", status_code=411)
        if len(content_lengths) != 1:
            return PlainTextResponse("Invalid Content-Length", status_code=400)
        try:
            content_length = int(content_lengths[0])
        except ValueError:
            return PlainTextResponse("Invalid Content-Length", status_code=400)
        if content_length < 0:
            return PlainTextResponse("Invalid Content-Length", status_code=400)
        if content_length > MAX_UPLOAD_REQUEST_BYTES:
            return PlainTextResponse("Upload request exceeds the 10 MB limit", status_code=413)

    response = await call_next(request)
    if request.url.path.startswith("/ui/") and not request.url.path.startswith("/ui/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


bearer = HTTPBearer(auto_error=False)
def principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    x_telegram_user_id: str | None = Header(default=None),
) -> str:
    if credentials is None or not credentials.credentials.isascii():
        raise HTTPException(401, "Invalid bearer token")
    for role, token in panel_auth.ROLE_TOKENS.items():
        if token and secrets.compare_digest(credentials.credentials.encode(), token.encode()):
            return role
    if isinstance(x_telegram_user_id, str):
        token = settings.user_api_tokens.get(x_telegram_user_id)
        if token and secrets.compare_digest(credentials.credentials.encode(), token.encode()):
            return "user"
    raise HTTPException(401, "Invalid bearer token")


def require(*roles):
    def check(role: str = Depends(principal)):
        if role not in roles:
            raise HTTPException(403, "Insufficient role")
        return role
    return check


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready", responses={503: {"description": "Database unavailable"}})
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(503, "Database unavailable") from None
    return {"status": "ready"}


@app.post(
    "/api/v1/tickets",
    response_model=TicketView, status_code=201,
    responses=describe_responses(400, 401, 403),
)
def create_ticket_api(data: TicketCreate, role: str = Depends(require("user", "agent", "admin")),
                      x_telegram_user_id: str | None = Header(default=None), db: Session = Depends(get_db)):
    if role == "user" and not x_telegram_user_id:
        raise HTTPException(400, "X-Telegram-User-ID is required for user token")
    user = get_or_create_user(db, x_telegram_user_id if role == "user" else None,
                              "API requester" if role != "user" else "")
    ticket = create_ticket(db, user.id, data)
    db.add(Message(requester_id=user.id, ticket_id=ticket.id, body=data.description, sender_type="user"))
    db.commit()
    db.refresh(ticket)
    return ticket


@app.get("/api/v1/tickets", response_model=list[TicketView], responses=describe_responses(401))
def list_tickets(status: str | None = None, role: str = Depends(require("user", "agent", "admin")),
                 x_telegram_user_id: str | None = Header(default=None), db: Session = Depends(get_db)):
    query = select(Ticket).order_by(Ticket.created_at.desc()).limit(100)
    if role == "user":
        user = db.scalar(select(User).where(User.telegram_user_id == x_telegram_user_id)) if x_telegram_user_id else None
        if not user:
            return []
        query = query.where(Ticket.requester_id == user.id)
    if status:
        query = query.where(Ticket.status == status)
    return list(db.scalars(query))


@app.get("/api/v1/tickets/{ticket_id}", response_model=TicketView, responses=describe_responses(401, 404))
def ticket_detail(ticket_id: uuid.UUID, role: str = Depends(require("user", "agent", "admin")),
                  x_telegram_user_id: str | None = Header(default=None), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    if role == "user":
        user = db.scalar(select(User).where(User.telegram_user_id == x_telegram_user_id)) if x_telegram_user_id else None
        if user is None or ticket.requester_id != user.id:
            raise HTTPException(404, "Ticket not found")
    return ticket


@app.patch(
    "/api/v1/tickets/{ticket_id}",
    response_model=TicketView,
    responses=describe_responses(401, 403, 404),
)
def update_ticket(ticket_id: uuid.UUID, data: TicketPatch, role: str = Depends(require("agent", "admin")),
                  db: Session = Depends(get_db)):
    return record_ticket_update(db, ticket_id, data)


@app.get(
    "/api/v1/tickets/{ticket_id}/messages",
    response_model=list[MessageView],
    responses=describe_responses(401, 403, 404),
)
def list_ticket_messages(ticket_id: uuid.UUID, role: str = Depends(require("user", "agent", "admin")),
                         x_telegram_user_id: str | None = Header(default=None), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    if role == "user":
        user = db.scalar(select(User).where(User.telegram_user_id == x_telegram_user_id)) \
            if x_telegram_user_id else None
        if user is None or ticket.requester_id != user.id:
            raise HTTPException(404, "Ticket not found")
    return list(db.scalars(
        select(Message).where(Message.ticket_id == ticket.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    ))


@app.post(
    "/api/v1/tickets/{ticket_id}/messages",
    status_code=201,
    responses=describe_responses(401, 403, 404),
)
def add_message(ticket_id: uuid.UUID, data: MessageCreate,
                role: str = Depends(require("user", "agent", "admin")),
                x_telegram_user_id: str | None = Header(default=None), db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    if role == "user":
        user = db.scalar(select(User).where(User.telegram_user_id == x_telegram_user_id)) if x_telegram_user_id else None
        if user is None or user.id != ticket.requester_id:
            raise HTTPException(404, "Ticket not found")
        msg = Message(requester_id=user.id, ticket_id=ticket.id, body=data.body,
                      sender_type="user")
        db.add(msg)
        ticket.updated_at = func.now()
        db.commit()
    else:
        msg = record_ticket_message(db, ticket, data)
    return {"id": str(msg.id), "created_at": msg.created_at}


@app.post("/api/v1/ai/analyze", response_model=Analysis, responses=describe_responses(401, 403, 503))
async def ai_analyze(data: MessageCreate, role: str = Depends(require("agent", "admin")),
                     db: Session = Depends(get_db)):
    try:
        return await analyze(data.body)
    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f"AI unavailable: {exc.__class__.__name__}") from exc


@app.post(
    "/api/v1/knowledge/documents",
    status_code=201,
    responses=describe_responses(400, 401, 403, 411, 413, 415, 503),
)
async def upload_document(file: UploadFile = File(...), role: str = Depends(require("admin")),
                          db: Session = Depends(get_db)):
    return await ingest_knowledge_document(file, db)


@app.get("/api/v1/knowledge/documents", responses=describe_responses(401, 403))
def list_documents(role: str = Depends(require("agent", "admin")), db: Session = Depends(get_db)):
    return [{"id": str(d.id), "title": d.title, "source_name": d.source_name, "status": d.status}
            for d in db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()))]


@app.get("/api/v1/analytics/overview", responses=describe_responses(401, 403))
def overview(role: str = Depends(require("agent", "admin")), db: Session = Depends(get_db)):
    return {status: count for status, count in db.execute(
        select(Ticket.status, func.count()).group_by(Ticket.status)).all()}


@app.post("/integrations/telegram/webhook", responses=describe_responses(400, 401))
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None),
                           db: Session = Depends(get_db)):
    if not settings.telegram_webhook_secret or not secrets.compare_digest(
            x_telegram_bot_api_secret_token or "", settings.telegram_webhook_secret):
        raise HTTPException(401, "Invalid webhook secret")

    try:
        update = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Malformed Telegram update") from None
    if not isinstance(update, dict):
        raise HTTPException(400, "Malformed Telegram update")

    callback_id = None
    callback = update.get("callback_query")
    if isinstance(callback, dict):
        origin = callback.get("message")
        chat = origin.get("chat") if isinstance(origin, dict) else None
        data = callback.get("data")
        callback_id = callback.get("id")
        if (not isinstance(chat, dict) or chat.get("type") != "private"
                or not isinstance(data, str) or len(data) > 64
                or not isinstance(callback_id, str) or not 1 <= len(callback_id) <= 80):
            return {"ok": True}
        action, _, answer_id = data.partition(":")
        if action in MENU_CALLBACKS and data == action:
            body = f"/{action}"
        elif action in TICKET_CALLBACKS and TICKET_NUMBER.fullmatch(answer_id):
            body = f"/{action} {answer_id}"
        elif action in {"helped", "specialist"} and len(answer_id) == 32:
            body = f"/{action} {answer_id}"
        else:
            return {"ok": True}
        update = {"message": {"message_id": origin.get("message_id"), "chat": chat,
                              "from": callback.get("from"), "text": body}}

    # Telegram sends many valid non-message updates (callbacks, membership events, etc.).
    # They do not enter the message-processing flow.
    if "message" not in update:
        return {"ok": True}
    message = update["message"]
    if not isinstance(message, dict):
        raise HTTPException(400, "Malformed Telegram update")

    chat = message.get("chat")
    chat_type = chat.get("type") if isinstance(chat, dict) else None
    reply_number = replied_ticket_number(message) if chat_type == "private" else None
    photo_id = None
    if "text" in message:
        body = message["text"]
    else:
        body = message.get("caption", "")
        photos = message.get("photo")
        if (not isinstance(body, str) or (not reply_number and not body.startswith("/add "))
                or not isinstance(photos, list) or not photos):
            return {"ok": True}
        photo = photos[-1]
        photo_id = photo.get("file_id") if isinstance(photo, dict) else None
        if not isinstance(photo_id, str) or not 1 <= len(photo_id) <= 512:
            raise HTTPException(400, "Malformed Telegram photo")
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    if (not isinstance(chat, dict)
            or not isinstance(chat_id, int) or isinstance(chat_id, bool)
            or not -(2**63) < chat_id < 2**63
            or chat_id == 0
            or not isinstance(chat_type, str)
            or chat_type not in {"private", "group", "supergroup", "channel"}):
        raise HTTPException(400, "Malformed Telegram text update")
    sender = message.get("from")
    telegram_user_id = sender.get("id") if isinstance(sender, dict) else None
    telegram_message_id = message.get("message_id")
    if (not isinstance(body, str) or (not body.strip() and not (photo_id and reply_number))
            or len(body) > 20000
            or not isinstance(telegram_user_id, int) or isinstance(telegram_user_id, bool)
            or not 0 < telegram_user_id < 2**63
            or not isinstance(telegram_message_id, int) or isinstance(telegram_message_id, bool)
            or not 0 < telegram_message_id < 2**63):
        raise HTTPException(400, "Malformed Telegram text update")
    if photo_id and chat_type != "private":
        return {"ok": True}
    body = body.strip()
    try:
        body.encode("utf-8")
    except UnicodeEncodeError:
        raise HTTPException(400, "Malformed Telegram text update") from None
    if reply_number and (
        (photo_id and not body.startswith("/add "))
        or (not body.startswith("/") and body.casefold() not in MENU_COMMANDS)
    ):
        body = f"/add {reply_number} {body}".rstrip()

    telegram_id = str(telegram_user_id)
    external_id = f"callback:{callback_id}" if callback_id else f"chat:{chat_id}:{telegram_message_id}"
    legacy_id = f"{telegram_id}:{telegram_message_id}"
    telegram_chat_id = str(chat_id) if chat_type == "private" else None
    name = sender.get("first_name", "")
    name = name.encode("utf-8", "replace").decode("utf-8")[:160] if isinstance(name, str) else ""

    def find_existing():
        existing = db.scalar(select(Message).where(
            Message.channel == "telegram", Message.external_message_id == external_id,
        ))
        if existing is not None or telegram_chat_id is None or callback_id:
            return existing
        legacy_chat = Message.telegram_chat_id == telegram_chat_id
        if telegram_chat_id == telegram_id:
            legacy_chat = legacy_chat | Message.telegram_chat_id.is_(None)
        return db.scalar(select(Message).where(
            Message.channel == "telegram", Message.external_message_id == legacy_id, legacy_chat,
        ))

    def persist_message():
        existing = find_existing()
        if existing is not None:
            db.commit()
            return existing, True
        user = get_or_create_user(db, telegram_id, name)
        msg = Message(requester_id=user.id, channel="telegram", external_message_id=external_id,
                      body=body, telegram_chat_id=telegram_chat_id,
                      analysis=({"photo_id": photo_id} if photo_id else
                                {"callback_id": callback_id} if callback_id else None),
                      processing_due_at=datetime.now(timezone.utc))
        db.add(msg)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = find_existing()
            if existing is None:
                raise
            db.commit()
            return existing, True
        return msg, False

    msg, duplicate = await run_in_threadpool(persist_message)
    if duplicate:
        return {"ok": True, "duplicate": True}
    return {"ok": True, "message_id": str(msg.id), "queued": True}


@app.post("/api/v1/internal/process-message", responses=describe_responses(401, 404))
async def process_message_from_workflow(payload: WorkflowMessageRequest,
                                        x_workflow_token: str | None = Header(default=None),
                                        db: Session = Depends(get_db)):
    if not settings.workflow_token or not secrets.compare_digest(x_workflow_token or "", settings.workflow_token):
        raise HTTPException(401, "Invalid workflow token")
    msg = await run_in_threadpool(db.get, Message, payload.message_id)
    if msg is None:
        raise HTTPException(404, "Message not found")
    return {"message_id": str(msg.id), "reply": await message_processing.process_message(db, msg)}
