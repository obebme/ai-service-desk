import asyncio
import logging
import secrets
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

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

from app import panel_auth
from app.ai import AIResponseError, analyze, grounded_answer
from app.config import settings
from app.db import SessionLocal, engine, get_db
from app.gemini import GeminiError
from app.knowledge_ingest import ingest_knowledge_document
from app.models import AuditLog, Department, KnowledgeDocument, Message, Ticket, User
from app.rag import search
from app.routers import panel
from app.schemas import (
    Analysis, MessageCreate, MessageView, TicketCreate, TicketPatch, TicketView,
    WorkflowMessageRequest, describe_responses,
)
from app.telegram import send_telegram_message
from app.ticket_writes import get_or_create_user, record_ticket_message, record_ticket_update


logger = logging.getLogger("uvicorn.error").getChild(__name__)

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
    worker = asyncio.create_task(_telegram_processing_worker())
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
DEPARTMENTS = {
    "network": "network_support", "vpn": "network_support", "internet": "network_support",
    "account": "system_administration", "authentication": "system_administration",
    "hardware": "hardware_support", "printer": "hardware_support",
    "software": "software_support", "security": "security_team",
}


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


def route_department(db: Session, category: str, subcategory: str, priority: str):
    key = "security" if priority == "critical" and "security" in f"{category} {subcategory}".lower() else category.lower()
    code = DEPARTMENTS.get(key, DEPARTMENTS.get(subcategory.lower(), "software_support"))
    return db.scalar(select(Department).where(Department.code == code))


def create_ticket(db: Session, requester_id: uuid.UUID, data: TicketCreate) -> Ticket:
    dept = route_department(db, data.category, data.subcategory, data.priority)
    ticket = Ticket(ticket_number=f"SD-{uuid.uuid4().hex[:10].upper()}", requester_id=requester_id,
                    department_id=dept.id if dept else None, category=data.category,
                    subcategory=data.subcategory, priority=data.priority,
                    subject=data.subject, description=data.description)
    db.add(ticket)
    db.flush()
    return ticket


@contextmanager
def _measure_stage(message_id: uuid.UUID | None, stage: str):
    started = perf_counter()
    outcome = "error"
    try:
        yield
        outcome = "ok"
    finally:
        logger.info("message_stage message_id=%s stage=%s outcome=%s elapsed_ms=%.1f",
                    message_id, stage, outcome, (perf_counter() - started) * 1000)


def _claim_automatic_reply(db: Session, msg: Message) -> bool:
    refresh = getattr(db, "refresh", None)
    if callable(refresh):
        refresh(msg, with_for_update=True)
    if msg.telegram_delivery_status in {"accepted", "unconfirmed"}:
        db.commit()
        return False

    # Claim delivery before sending. Telegram cannot deduplicate an ambiguous retry.
    msg.telegram_delivery_status = "unconfirmed"
    msg.telegram_delivery_updated_at = datetime.now(timezone.utc)
    db.commit()
    return True


async def _deliver_automatic_reply(db: Session, msg: Message, reply: str) -> None:
    if msg.channel != "telegram" or not msg.telegram_chat_id or not settings.telegram_token:
        return
    if not await run_in_threadpool(_claim_automatic_reply, db, msg):
        return
    try:
        with _measure_stage(msg.id, "telegram_delivery"):
            if (msg.analysis or {}).get("knowledge_answer") and msg.id:
                await send_telegram_message(msg.telegram_chat_id, reply, answer_id=msg.id)
            else:
                await send_telegram_message(msg.telegram_chat_id, reply)
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        logger.warning("telegram_delivery_unconfirmed message_id=%s", msg.id)
        return
    msg.telegram_delivery_status = "accepted"
    msg.telegram_delivery_updated_at = datetime.now(timezone.utc)
    # A concurrent processor can hold the row lock while this confirmation is saved.
    await run_in_threadpool(db.commit)


def _telegram_ticket_command(db: Session, msg: Message) -> str | None:
    if msg.channel != "telegram" or not msg.telegram_chat_id:
        return None

    command, _, argument = msg.body.partition(" ")
    if command == "/tickets" or msg.body.casefold() == "мои обращения":
        tickets = list(db.scalars(select(Ticket).where(Ticket.requester_id == msg.requester_id)
                                  .order_by(Ticket.updated_at.desc(), Ticket.id.desc()).limit(10)))
        if not tickets:
            return "У вас пока нет обращений."
        statuses = {"open": "открыто", "in_progress": "в работе", "waiting_for_user": "ждём вашего ответа",
                    "resolved": "решено", "closed": "закрыто"}
        lines = [f"{ticket.ticket_number} · {statuses.get(ticket.status, ticket.status)} · "
                 f"{ticket.updated_at:%d.%m %H:%M} UTC\n{ticket.subject[:120]}" for ticket in tickets]
        return "Ваши последние обращения\n\n" + "\n".join(lines) + (
            "\n\nЧтобы дополнить обращение, отправьте /add НОМЕР текст.")
    if command == "/add":
        number, _, body = argument.strip().partition(" ")
        if not number or (not body.strip() and not (msg.analysis or {}).get("photo_id")):
            return "Отправьте /add НОМЕР текст. Номер можно найти командой /tickets."
        ticket = db.scalar(select(Ticket).where(
            Ticket.requester_id == msg.requester_id, Ticket.ticket_number == number.upper()
        ).with_for_update())
        if ticket is None:
            return "Обращение не найдено. Проверьте номер командой /tickets."
        if ticket.status == "closed":
            return "Обращение закрыто. Отправьте новый вопрос, чтобы создать новое обращение."
        msg.body = body.strip() or "Screenshot attached"
        msg.ticket_id = ticket.id
        if ticket.status in {"waiting_for_user", "resolved"}:
            ticket.status = "in_progress"
        ticket.updated_at = datetime.now(timezone.utc)
        return f"Добавил сообщение в обращение {ticket.ticket_number}."
    if command in {"/helped", "/specialist"}:
        try:
            answer_id = uuid.UUID(argument.strip())
        except ValueError:
            return "Используйте команду с кодом из ответа бота."
        original = db.scalar(select(Message).where(
            Message.id == answer_id, Message.requester_id == msg.requester_id,
            Message.telegram_chat_id == msg.telegram_chat_id, Message.channel == "telegram",
            Message.sender_type == "user",
        ).with_for_update())
        analysis = original.analysis if original is not None else None
        if not analysis or not analysis.get("knowledge_answer"):
            return "Ответ базы знаний не найден."
        if original.ticket_id:
            ticket = db.get(Ticket, original.ticket_id)
            if ticket is not None:
                return f"Для этого вопроса уже создано обращение {ticket.ticket_number}."
            return "Для этого вопроса уже создавалось обращение."
        if command == "/helped":
            original.analysis = {**analysis, "feedback": "helped"}
            return "Спасибо за обратную связь."
        classification = Analysis.model_validate(analysis["classification"])
        ticket = create_ticket(db, msg.requester_id, TicketCreate(
            subject=classification.summary.strip()[:240] if len(classification.summary.strip()) >= 3 else "Support request",
            description=original.body, category=classification.category,
            subcategory=classification.subcategory, priority=classification.priority,
        ))
        original.ticket_id = ticket.id
        original.analysis = {**analysis, "escalation_reason": "user_requested_specialist"}
        return f"Создал обращение {ticket.ticket_number}. Специалист продолжит разбор."
    help_text = (
        "Мои обращения — /tickets. Дополнить обращение — /add НОМЕР текст. "
        "Новый вопрос можно отправить обычным сообщением."
    )
    if command in {"/help", "/start"}:
        return help_text
    if command.startswith("/"):
        return f"Неизвестная команда. {help_text}"
    return None


async def process_message(db: Session, msg: Message) -> str:
    refresh = getattr(db, "refresh", None)
    locked = callable(refresh)
    if locked:
        # Waiting for another transaction must leave the event loop free to finish it.
        await run_in_threadpool(refresh, msg, with_for_update=True)
    if msg.analysis and msg.analysis.get("reply"):
        reply = msg.analysis["reply"]
        if locked:
            await run_in_threadpool(db.commit)
        await _deliver_automatic_reply(db, msg, reply)
        return reply
    if msg.channel == "telegram" and msg.telegram_chat_id and (
        msg.body.startswith("/") or msg.body.casefold() == "мои обращения"
    ):
        reply = await run_in_threadpool(_telegram_ticket_command, db, msg)
        if reply is not None:
            msg.analysis = {**(msg.analysis or {}), "reply": reply}
            await run_in_threadpool(db.commit)
            if (callback_id := msg.analysis.get("callback_id")) and settings.telegram_token:
                try:
                    async with httpx.AsyncClient(timeout=3) as client:
                        await client.post(
                            f"https://api.telegram.org/bot{settings.telegram_token}/answerCallbackQuery",
                            json={"callback_query_id": callback_id},
                        )
                except httpx.HTTPError:
                    logger.warning("telegram_callback_ack_failed message_id=%s", msg.id)
            await _deliver_automatic_reply(db, msg, reply)
            return reply
    classification = None
    trusted = []
    answer = None
    reason = None
    try:
        with _measure_stage(msg.id, "classification"):
            classification = await analyze(msg.body)
        if classification.priority == "critical":
            reason = "critical"
        elif classification.needs_human:
            reason = "requires_specialist"
        elif classification.confidence < settings.confidence_threshold:
            reason = "low_confidence"
        else:
            with _measure_stage(msg.id, "knowledge_search"):
                passages = await search(db, msg.body, limit=4)
                trusted = [p for p in passages if p["score"] >= 0.35]
            if not trusted:
                reason = "no_relevant_passages"
            else:
                with _measure_stage(msg.id, "answer_generation"):
                    answer = await grounded_answer(msg.body, trusted)
                if answer is None:
                    reason = "no_confirmed_solution"
    except (httpx.HTTPError, GeminiError, AIResponseError) as exc:
        reason = "ai_error"
        logger.warning("message_ai_error message_id=%s error_type=%s", msg.id, type(exc).__name__)

    reply = await run_in_threadpool(_save_processing_result, db, msg, classification, reason, answer)
    logger.info("message_decision message_id=%s reason=%s", msg.id, reason or "knowledge_answer")
    await _deliver_automatic_reply(db, msg, reply)
    return reply


def _save_processing_result(
    db: Session, msg: Message, classification: Analysis | None, reason: str | None,
    answer: tuple[str, list[str]] | None,
) -> str:
    if reason is not None:
        summary = classification.summary if classification is not None else ""
        subject = summary.strip()[:240].strip()
        if len(subject) < 3:
            subject = "Support request"
        ticket_data = TicketCreate(
            subject=subject,
            description=msg.body[:20000],
            category=classification.category if classification is not None else "other",
            subcategory=classification.subcategory if classification is not None else "other",
            priority=classification.priority if classification is not None else "medium",
        )
        ticket = create_ticket(db, msg.requester_id, ticket_data)
        msg.ticket_id = ticket.id
        if reason == "ai_error":
            reply = f"Не удалось завершить автоматическую диагностику. Создал обращение {ticket.ticket_number}."
        else:
            reply = f"Создал обращение {ticket.ticket_number}. Специалист продолжит разбор."
        msg.analysis = {"reply": reply, "escalation_reason": reason}
        if classification is not None:
            msg.analysis["classification"] = classification.model_dump()
    else:
        answer_text, sources = answer
        reply = f"{answer_text}\n\nИсточник: {', '.join(sources)}"
        msg.analysis = {"classification": classification.model_dump(), "reply": reply,
                        "knowledge_answer": answer_text, "sources": sources}
    db.commit()
    return reply


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
        if action not in {"helped", "specialist"} or len(answer_id) != 32:
            return {"ok": True}
        update = {"message": {"message_id": origin.get("message_id"), "chat": chat,
                              "from": callback.get("from"), "text": f"/{action} {answer_id}"}}

    # Telegram sends many valid non-message updates (callbacks, membership events, etc.).
    # They do not enter the message-processing flow.
    if "message" not in update:
        return {"ok": True}
    message = update["message"]
    if not isinstance(message, dict):
        raise HTTPException(400, "Malformed Telegram update")

    photo_id = None
    if "text" in message:
        body = message["text"]
    else:
        body = message.get("caption")
        photos = message.get("photo")
        if not isinstance(body, str) or not body.startswith("/add ") or not isinstance(photos, list) or not photos:
            return {"ok": True}
        photo = photos[-1]
        photo_id = photo.get("file_id") if isinstance(photo, dict) else None
        if not isinstance(photo_id, str) or not 1 <= len(photo_id) <= 512:
            raise HTTPException(400, "Malformed Telegram photo")
    chat = message.get("chat")
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    chat_type = chat.get("type") if isinstance(chat, dict) else None
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
    if (not isinstance(body, str) or not body.strip()
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


def _claim_telegram_message():
    with SessionLocal.begin() as db:
        now = datetime.now(timezone.utc)
        msg = db.scalar(
            select(Message).where(Message.processing_due_at <= now)
            .order_by(Message.processing_due_at, Message.id)
            .limit(1).with_for_update(skip_locked=True)
        )
        if msg is None:
            return None
        # A crashed worker leaves a recoverable claim. Normal failures retry sooner.
        msg.processing_due_at = now + timedelta(minutes=5)
        return msg.id, msg.processing_due_at


def _finish_telegram_attempt(message_id, claim_until, succeeded):
    with SessionLocal.begin() as db:
        msg = db.get(Message, message_id, with_for_update=True)
        if msg is not None and msg.processing_due_at == claim_until:
            msg.processing_due_at = None if succeeded else datetime.now(timezone.utc) + timedelta(seconds=15)


async def _dispatch_telegram_message(message_id):
    workflow_acknowledged = False
    if settings.n8n_webhook_url:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(210, connect=5)) as client:
                response = await client.post(
                    settings.n8n_webhook_url, json={"message_id": str(message_id)},
                    headers={"X-Workflow-Token": settings.n8n_ingress_token},
                )
                response.raise_for_status()
            workflow_acknowledged = True
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            logger.warning("telegram_n8n_fallback message_id=%s error_type=%s",
                           message_id, type(exc).__name__)

    db = SessionLocal()
    try:
        msg = await run_in_threadpool(db.get, Message, message_id)
        if msg is None:
            return
        if workflow_acknowledged:
            if (msg.analysis or {}).get("reply"):
                return
            logger.warning("telegram_n8n_fallback message_id=%s reason=missing_saved_reply", message_id)
        # A workflow may still be processing this row. Use the same lock and saved result.
        await process_message(db, msg)
    finally:
        await run_in_threadpool(db.close)


async def _telegram_processing_worker():
    while True:
        try:
            claim = await run_in_threadpool(_claim_telegram_message)
            if claim is None:
                await asyncio.sleep(0.25)
                continue
            message_id, claim_until = claim
            succeeded = False
            try:
                await _dispatch_telegram_message(message_id)
                succeeded = True
            except Exception as exc:
                logger.warning("telegram_job_retry message_id=%s error_type=%s",
                               message_id, type(exc).__name__)
            await run_in_threadpool(_finish_telegram_attempt, message_id, claim_until, succeeded)
        except SQLAlchemyError as exc:
            logger.warning("telegram_queue_error error_type=%s", type(exc).__name__)
            await asyncio.sleep(2)


@app.post("/api/v1/internal/process-message", responses=describe_responses(401, 404))
async def process_message_from_workflow(payload: WorkflowMessageRequest,
                                        x_workflow_token: str | None = Header(default=None),
                                        db: Session = Depends(get_db)):
    if not settings.workflow_token or not secrets.compare_digest(x_workflow_token or "", settings.workflow_token):
        raise HTTPException(401, "Invalid workflow token")
    msg = await run_in_threadpool(db.get, Message, payload.message_id)
    if msg is None:
        raise HTTPException(404, "Message not found")
    return {"message_id": str(msg.id), "reply": await process_message(db, msg)}
