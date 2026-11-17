import asyncio
import io
import logging
import secrets
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

import httpx
from jinja2 import TemplateNotFound
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool
from starlette.templating import Jinja2Templates
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from sqlalchemy import func, or_, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.ai import AIResponseError, analyze, grounded_answer
from app.config import settings
from app.db import SessionLocal, engine, get_db
from app.models import AuditLog, Department, KnowledgeChunk, KnowledgeDocument, Message, Ticket, User
from app.panel_auth import (
    CHALLENGE_COOKIE,
    CHALLENGE_TTL_SECONDS,
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    authenticate_role_token,
    create_challenge,
    create_session,
    read_session,
    verify_challenge,
    verify_csrf,
)
from app.gemini import GeminiError, embed
from app.rag import search
from app.schemas import (
    Analysis, MessageCreate, MessageView, TicketCreate, TicketPatch, TicketView,
    WorkflowMessageRequest,
)

logger = logging.getLogger("uvicorn.error").getChild(__name__)

MAX_UPLOAD_FILE_BYTES = 10_000_000
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
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount(
    "/ui/static",
    StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
    name="panel-static",
)


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
ROLE_TOKENS = {
    "agent": settings.agent_api_token,
    "admin": settings.admin_api_token,
}
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
    for role, token in ROLE_TOKENS.items():
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


@app.get("/ui/login", response_class=HTMLResponse)
def panel_login_page(request: Request):
    secret = settings.validate_session_signing_secret()
    if read_session(request.cookies.get(SESSION_COOKIE), secret):
        return RedirectResponse("/ui/tickets", status_code=303)
    csrf, challenge = create_challenge(secret)
    try:
        response = templates.TemplateResponse(
            request=request, name="login.html",
            context={"csrf_token": csrf, "error": None},
        )
    except TemplateNotFound:
        # Keep the already-deployed login usable until the frontend template lands.
        response = HTMLResponse(
            '<!doctype html><html><body><form method="post" action="/ui/login">'
            '<input type="hidden" name="csrf_token" value="' + csrf + '">'
            '<label>Role token <input type="password" name="role_token" required></label>'
            '<button type="submit">Sign in</button></form></body></html>'
        )
    response.set_cookie(
        CHALLENGE_COOKIE, challenge, max_age=CHALLENGE_TTL_SECONDS,
        httponly=True, secure=True, samesite="lax", path="/ui",
    )
    return response


@app.post("/ui/login")
async def panel_login(request: Request):
    secret = settings.validate_session_signing_secret()
    form = await request.form()
    if (set(form.keys()) != {"role_token", "csrf_token"}
            or len(form.multi_items()) != 2):
        raise HTTPException(status_code=400, detail="Invalid login form")
    token, csrf = form.get("role_token"), form.get("csrf_token")
    if not isinstance(token, str) or not isinstance(csrf, str) or not verify_challenge(
            request.cookies.get(CHALLENGE_COOKIE), csrf, secret):
        raise HTTPException(status_code=400, detail="Invalid CSRF challenge")
    role = authenticate_role_token(token, ROLE_TOKENS)
    agent = next((name for name, credential in settings.panel_agent_tokens.items()
                  if secrets.compare_digest(token.encode(), credential.encode())), None)
    if agent is not None:
        role = "agent"
    if role not in {"agent", "admin"}:
        csrf, challenge = create_challenge(secret)
        try:
            response = templates.TemplateResponse(
                request=request, name="login.html",
                context={"csrf_token": csrf, "error": "Invalid role token"},
                status_code=401,
            )
        except TemplateNotFound:
            response = HTMLResponse(
                '<!doctype html><html><body><p>Invalid role token</p>'
                '<form method="post" action="/ui/login">'
                '<input type="hidden" name="csrf_token" value="' + csrf + '">'
                '<label>Role token <input type="password" name="role_token" required></label>'
                '<button type="submit">Sign in</button></form></body></html>',
                status_code=401,
            )
        response.set_cookie(
            CHALLENGE_COOKIE, challenge, max_age=CHALLENGE_TTL_SECONDS,
            httponly=True, secure=True, samesite="lax", path="/ui",
        )
        response.headers["Cache-Control"] = "no-store"
        return response
    _session_csrf, signed_session = create_session(role, secret, agent=agent)
    response = RedirectResponse("/ui/tickets", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, signed_session, max_age=SESSION_TTL_SECONDS,
        httponly=True, secure=True, samesite="lax", path="/ui",
    )
    response.delete_cookie(CHALLENGE_COOKIE, path="/ui", secure=True, httponly=True, samesite="lax")
    return response


@app.post("/ui/logout")
async def panel_logout(request: Request):
    secret = settings.validate_session_signing_secret()
    form = await request.form()
    if (set(form.keys()) != {"csrf_token"} or len(form.multi_items()) != 1
            or not isinstance(form.get("csrf_token"), str)):
        raise HTTPException(status_code=400, detail="Invalid logout form")
    payload = read_session(request.cookies.get(SESSION_COOKIE), secret)
    if payload is None:
        raise HTTPException(status_code=401, detail="Panel session required")
    if not verify_csrf(form["csrf_token"], payload["csrf"]):
        raise HTTPException(status_code=400, detail="Invalid CSRF token")
    response = RedirectResponse("/ui/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/ui", secure=True, httponly=True, samesite="lax")
    return response


def _panel_session(request: Request, *, admin_only: bool = False) -> dict | RedirectResponse:
    secret = settings.validate_session_signing_secret()
    payload = read_session(request.cookies.get(SESSION_COOKIE), secret)
    if payload is None or (payload.get("agent") is not None
                           and payload["agent"] not in settings.panel_agent_tokens):
        return RedirectResponse("/ui/login", status_code=303)
    if admin_only and payload["role"] != "admin":
        raise HTTPException(403, "Admin role required")
    return payload


def _panel_query(request: Request) -> tuple[str | None, str, int, str]:
    pairs = request.query_params.multi_items()
    if len(pairs) != len({key for key, _value in pairs}) or any(
        key not in {"status", "q", "page", "attention"} for key, _value in pairs
    ):
        raise HTTPException(422, "Invalid query parameters")
    values = dict(pairs)
    status = values.get("status") or None
    query = values.get("q", "")
    statuses = {"open", "in_progress", "waiting_for_user", "resolved", "closed"}
    if status is not None and (len(status) > 30 or status not in statuses):
        raise HTTPException(422, "Invalid status filter")
    if len(query) > 200:
        raise HTTPException(422, "Search query exceeds 200 characters")
    page = values.get("page", "1")
    if not page.isascii() or not page.isdecimal() or len(page) > 9 or int(page) < 1:
        raise HTTPException(422, "Invalid page")
    attention = values.get("attention", "")
    if attention not in {"", "reply", "delivery", "mine", "unassigned"}:
        raise HTTPException(422, "Invalid attention filter")
    return status, query.strip(), int(page), attention


async def _panel_form(request: Request, required: set[str],
                      allowed: set[str] | None = None) -> dict[str, str | UploadFile]:
    allowed = allowed or required
    # Text inputs are bounded; Starlette streams file parts separately, so the
    # existing 10 MB upload check remains authoritative for the file itself.
    form = await request.form(max_files=1, max_fields=4, max_part_size=80_000)
    pairs = form.multi_items()
    keys = {key for key, _value in pairs}
    if (not required.issubset(keys) or not keys.issubset(allowed)
            or len({key for key, _value in pairs}) != len(pairs)):
        raise HTTPException(400, "Invalid form")
    values: dict[str, str | UploadFile] = {}
    for key, value in pairs:
        if isinstance(value, StarletteUploadFile):
            values[key] = value
        elif isinstance(value, str):
            values[key] = value
        else:
            raise HTTPException(400, "Invalid form")
    return values


def _validate_panel_csrf(form: dict[str, str | UploadFile], payload: dict) -> None:
    submitted = form.get("csrf_token")
    if not isinstance(submitted, str) or len(submitted) > 256 or not verify_csrf(
        submitted, payload["csrf"]
    ):
        raise HTTPException(400, "Invalid CSRF token")


def _record_ticket_update(db: Session, ticket: Ticket, data: TicketPatch) -> Ticket:
    changes = data.model_dump(exclude_unset=True)
    old_status = ticket.status
    for key, value in changes.items():
        setattr(ticket, key, value)
    if changes.get("status") == "closed" and old_status != "closed":
        ticket.closed_at = datetime.now(timezone.utc)
    elif old_status == "closed" and changes.get("status", old_status) != "closed":
        ticket.closed_at = None
    db.add(AuditLog(action="ticket.updated", entity_type="ticket", entity_id=ticket.id,
                    details={"fields": list(changes)}))
    db.commit()
    db.refresh(ticket)
    return ticket


def _record_ticket_message(
    db: Session, ticket: Ticket, data: MessageCreate,
    telegram_delivery_status: str | None = None,
) -> Message:
    user = get_or_create_user(db, None, "Support agent")
    msg = Message(requester_id=user.id, ticket_id=ticket.id, body=data.body, sender_type="agent")
    if telegram_delivery_status is not None:
        msg.telegram_delivery_status = telegram_delivery_status
        msg.telegram_delivery_updated_at = datetime.now(timezone.utc)
    db.add(msg)
    ticket.updated_at = func.now()
    db.commit()
    db.refresh(msg)
    return msg


def get_or_create_user(db: Session, telegram_id: str | None, name: str = "") -> User:
    if telegram_id:
        statement = (
            insert(User).values(telegram_user_id=telegram_id, display_name=name[:160])
            .on_conflict_do_nothing(index_elements=[User.telegram_user_id])
        )
        db.execute(statement)
        return db.scalars(select(User).where(User.telegram_user_id == telegram_id)).one()
    user = User(display_name=name[:160])
    db.add(user)
    db.flush()
    return user


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


async def _deliver_automatic_reply(db: Session, msg: Message, reply: str) -> None:
    if msg.channel != "telegram" or not msg.telegram_chat_id or not settings.telegram_token:
        return
    refresh = getattr(db, "refresh", None)
    if callable(refresh):
        await run_in_threadpool(refresh, msg, with_for_update=True)
    if msg.telegram_delivery_status in {"accepted", "unconfirmed"}:
        db.commit()
        return

    # Claim delivery before sending. Telegram cannot deduplicate an ambiguous retry.
    msg.telegram_delivery_status = "unconfirmed"
    msg.telegram_delivery_updated_at = datetime.now(timezone.utc)
    db.commit()
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
            db.commit()
        await _deliver_automatic_reply(db, msg, reply)
        return reply
    if msg.channel == "telegram" and msg.telegram_chat_id and (
        msg.body.startswith("/") or msg.body.casefold() == "мои обращения"
    ):
        reply = await run_in_threadpool(_telegram_ticket_command, db, msg)
        if reply is not None:
            msg.analysis = {**(msg.analysis or {}), "reply": reply}
            db.commit()
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
    logger.info("message_decision message_id=%s reason=%s", msg.id, reason or "knowledge_answer")
    await _deliver_automatic_reply(db, msg, reply)
    return reply


async def send_telegram_message(chat_id: str, message: str, *, answer_id: uuid.UUID | None = None):
    payload = {"chat_id": chat_id, "text": message[:4000]}
    if answer_id:
        payload["reply_markup"] = {"inline_keyboard": [[
            {"text": "Помогло", "callback_data": f"helped:{answer_id.hex}"},
            {"text": "Нужен специалист", "callback_data": f"specialist:{answer_id.hex}"},
        ]]}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage",
                # Do not set parse_mode: user/model text must be delivered as plain text.
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError:
        # Never expose the Bot API URL (which includes the token) in error details.
        raise HTTPException(503, "Telegram message delivery failed") from None

    try:
        result = response.json()
    except ValueError:
        raise HTTPException(503, "Telegram message delivery failed") from None
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise HTTPException(503, "Telegram message delivery failed")


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


@app.post("/api/v1/tickets", response_model=TicketView, status_code=201)
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


@app.get("/api/v1/tickets", response_model=list[TicketView])
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


@app.get("/api/v1/tickets/{ticket_id}", response_model=TicketView)
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


@app.patch("/api/v1/tickets/{ticket_id}", response_model=TicketView)
def update_ticket(ticket_id: uuid.UUID, data: TicketPatch, role: str = Depends(require("agent", "admin")),
                  db: Session = Depends(get_db)):
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    return _record_ticket_update(db, ticket, data)


@app.get("/api/v1/tickets/{ticket_id}/messages", response_model=list[MessageView])
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


def _queue_age(created_at: datetime | None) -> str:
    if created_at is None:
        return "—"
    seconds = max(0, int((datetime.now(timezone.utc) - created_at).total_seconds()))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _unconfirmed_auto_reply():
    return (
        (Message.channel == "telegram")
        & (Message.sender_type == "user")
        & (Message.telegram_delivery_status == "unconfirmed")
    )


def _awaiting_reply():
    latest_sender = (
        select(Message.sender_type).where(Message.ticket_id == Ticket.id)
        .order_by(Message.created_at.desc(), Message.id.desc()).limit(1).scalar_subquery()
    )
    return Ticket.status.in_(("open", "in_progress")) & (func.coalesce(latest_sender, "user") == "user")


@app.get("/ui/tickets", response_class=HTMLResponse)
async def panel_ticket_queue(request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    status, query, requested_page, attention = _panel_query(request)
    filters = []
    if status:
        filters.append(Ticket.status == status)
    if query:
        filters.append(or_(
            Ticket.ticket_number.icontains(query, autoescape=True),
            Ticket.subject.icontains(query, autoescape=True),
            Ticket.description.icontains(query, autoescape=True),
        ))

    if attention == "reply":
        filters.append(_awaiting_reply())
    elif attention == "delivery":
        filters.append(Ticket.id.in_(select(Message.ticket_id).where(_unconfirmed_auto_reply())))

    elif attention == "mine":
        agent = payload.get("agent")
        if not agent:
            raise HTTPException(403, "Personal agent session required")
        filters.append(Ticket.assigned_agent == agent)
    elif attention == "unassigned":
        filters.append(Ticket.assigned_agent.is_(None))

    page_size = 25
    total_count = db.scalar(select(func.count()).select_from(Ticket).where(*filters))
    page_count = max(1, (total_count + page_size - 1) // page_size)
    page = min(requested_page, page_count)
    statement = (
        select(Ticket).where(*filters).order_by(
            (Ticket.updated_at if attention == "reply" else Ticket.created_at).desc(), Ticket.id.desc()
        )
        .offset((page - 1) * page_size).limit(page_size)
    )
    tickets = list(db.scalars(statement))
    params = {key: value for key, value in (("q", query), ("status", status), ("attention", attention)) if value}
    previous_url = "/ui/tickets?" + urlencode(dict(params, page=page - 1)) if page > 1 else None
    next_url = "/ui/tickets?" + urlencode(dict(params, page=page + 1)) if page < page_count else None
    pending_count, oldest = db.execute(
        select(func.count(), func.min(Message.created_at))
        .where(Message.processing_due_at.is_not(None))
    ).one()
    unconfirmed = _unconfirmed_auto_reply()
    unconfirmed_count = db.scalar(select(func.count()).select_from(Message).where(unconfirmed))
    ticket_ids = [ticket.id for ticket in tickets]
    unconfirmed_ticket_ids = set(db.scalars(
        select(Message.ticket_id).where(unconfirmed, Message.ticket_id.in_(ticket_ids)).distinct()
    )) if ticket_ids and unconfirmed_count else set()
    return templates.TemplateResponse(
        request=request, name="ticket_queue.html",
        context={
            "request": request, "role": payload["role"], "tickets": tickets,
            "status": status, "q": query, "attention": attention, "total_count": total_count, "agent": payload.get("agent"),
            "page": page, "page_count": page_count, "previous_url": previous_url, "next_url": next_url,
            "csrf_token": payload["csrf"],
            "pending_count": pending_count, "oldest_pending_age": _queue_age(oldest),
            "unconfirmed_count": unconfirmed_count,
            "unconfirmed_ticket_ids": unconfirmed_ticket_ids,
        },
    )


@app.get("/ui/tickets/{ticket_id}", response_class=HTMLResponse)
async def panel_ticket_detail(ticket_id: uuid.UUID, request: Request,
                              db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    messages = list(db.scalars(
        select(Message).where(Message.ticket_id == ticket.id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    ))
    feedback_message = None
    requested_message_id = request.query_params.get("message_id")
    if requested_message_id:
        try:
            feedback_id = uuid.UUID(requested_message_id)
        except ValueError:
            feedback_id = None
        feedback_message = next(
            (message for message in messages
             if message.id == feedback_id and message.sender_type == "agent"),
            None,
        )
    return templates.TemplateResponse(
        request=request, name="ticket_detail.html",
        context={
            "request": request, "role": payload["role"], "ticket": ticket,
            "messages": messages, "csrf_token": payload["csrf"], "agent": payload.get("agent"), "error": None,
            "body": "", "feedback_message": feedback_message,
        },
    )


@app.get("/ui/tickets/{ticket_id}/photos/{message_id}")
async def panel_ticket_photo(ticket_id: uuid.UUID, message_id: uuid.UUID, request: Request,
                             db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    msg = await run_in_threadpool(db.get, Message, message_id)
    photo_id = (msg.analysis or {}).get("photo_id") if msg and msg.ticket_id == ticket_id else None
    if not photo_id or not settings.telegram_token:
        raise HTTPException(404, "Photo not found")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            root = "https://api.telegram.org"
            token = settings.telegram_token
            metadata = await client.post(f"{root}/bot{token}/getFile", json={"file_id": photo_id})
            metadata.raise_for_status()
            path = metadata.json()["result"]["file_path"]
            if not isinstance(path, str) or not path.startswith("photos/") or ".." in path or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-" for char in path
            ):
                raise ValueError("Invalid Telegram photo path")
            async with client.stream("GET", f"{root}/file/bot{token}/{path}") as photo:
                photo.raise_for_status()
                data = bytearray()
                async for chunk in photo.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > 10_000_000:
                        raise ValueError("Photo exceeds limit")
                if not data.startswith(bytes.fromhex("ffd8ff")):
                    raise ValueError("Invalid JPEG")
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        raise HTTPException(503, "Unable to retrieve Telegram photo") from None
    return Response(bytes(data), media_type="image/jpeg",
                    headers={"X-Content-Type-Options": "nosniff"})


@app.post("/ui/tickets/{ticket_id}/claim")
async def panel_claim_ticket(ticket_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(request, {"csrf_token"})
    _validate_panel_csrf(form, payload)
    agent = payload.get("agent")
    if not agent or agent not in settings.panel_agent_tokens:
        raise HTTPException(403, "Personal agent session required")
    ticket = await run_in_threadpool(db.get, Ticket, ticket_id, with_for_update=True)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    if ticket.assigned_agent not in {None, agent} or ticket.status in {"closed", "resolved"}:
        raise HTTPException(409, "Ticket is already assigned or resolved")
    if ticket.assigned_agent is None:
        ticket.assigned_agent = agent
        if ticket.status == "open":
            ticket.status = "in_progress"
        db.add(AuditLog(action="ticket.claimed", entity_type="ticket", entity_id=ticket.id,
                        details={"agent": agent}))
    db.commit()
    return RedirectResponse(f"/ui/tickets/{ticket.id}", status_code=303)


@app.post("/ui/tickets/{ticket_id}/messages")
async def panel_add_ticket_message(ticket_id: uuid.UUID, request: Request,
                                   db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(request, {"body", "csrf_token"})
    _validate_panel_csrf(form, payload)
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    try:
        data = MessageCreate.model_validate({"body": form["body"]})
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc

    eligible_messages = list(db.scalars(
        select(Message).where(
            Message.ticket_id == ticket.id,
            Message.channel == "telegram",
            Message.sender_type == "user",
            Message.telegram_chat_id.is_not(None),
        ).order_by(Message.created_at.asc(), Message.id.asc()).limit(1)
    ))
    eligible_messages = [
        message for message in eligible_messages
        if message.ticket_id == ticket.id
        and message.channel == "telegram"
        and message.sender_type == "user"
        and message.telegram_chat_id is not None
    ]
    recipient = min(
        eligible_messages,
        key=lambda message: (message.created_at, str(message.id)),
        default=None,
    )
    if recipient is not None and len(data.body) > 4000:
        messages = list(db.scalars(
            select(Message).where(Message.ticket_id == ticket.id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        ))
        return templates.TemplateResponse(
            request=request, name="ticket_detail.html",
            context={
                "request": request, "role": payload["role"], "ticket": ticket,
                "messages": messages, "csrf_token": payload["csrf"],
                "error": "Telegram replies must be 4,000 characters or fewer.",
                "body": data.body, "feedback_message": None,
            },
            status_code=400,
        )

    if recipient is None:
        delivery_status = "not_applicable"
    elif not settings.telegram_token:
        delivery_status = "not_configured"
    else:
        delivery_status = "pending"
    saved_message = _record_ticket_message(
        db, ticket, data, telegram_delivery_status=delivery_status
    )
    if recipient is not None and settings.telegram_token:
        try:
            await send_telegram_message(recipient.telegram_chat_id, saved_message.body)
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            saved_message.telegram_delivery_status = "unconfirmed"
        else:
            saved_message.telegram_delivery_status = "accepted"
        saved_message.telegram_delivery_updated_at = datetime.now(timezone.utc)
        db.add(saved_message)
        try:
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            logger.warning(
                "Could not persist Telegram delivery outcome (message_id=%s, error_type=%s)",
                saved_message.id, type(exc).__name__,
            )
    return RedirectResponse(
        f"/ui/tickets/{ticket.id}?message_id={saved_message.id}", status_code=303
    )


@app.post("/ui/tickets/{ticket_id}/update")
async def panel_update_ticket(ticket_id: uuid.UUID, request: Request,
                              db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(
        request, {"csrf_token"}, {"csrf_token", "status", "priority", "resolution"}
    )
    _validate_panel_csrf(form, payload)
    ticket = db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
    patch_data = {
        key: (None if key == "resolution" and value == "" else value)
        for key, value in form.items() if key != "csrf_token"
    }
    try:
        data = TicketPatch.model_validate(patch_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    _record_ticket_update(db, ticket, data)
    return RedirectResponse(f"/ui/tickets/{ticket.id}", status_code=303)


@app.get("/ui/knowledge", response_class=HTMLResponse)
async def panel_knowledge(request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request, admin_only=True)
    if isinstance(payload, RedirectResponse):
        return payload
    query_pairs = request.query_params.multi_items()
    if (len(query_pairs) != len({key for key, _value in query_pairs})
            or any(key != "success" for key, _value in query_pairs)
            or any(len(value) > 500 for _key, value in query_pairs)):
        raise HTTPException(422, "Invalid knowledge-page query")
    documents = list(db.scalars(
        select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
    ))
    return templates.TemplateResponse(
        request=request, name="knowledge.html",
        context={
            "request": request, "role": payload["role"], "documents": documents,
            "csrf_token": payload["csrf"], "error": None,
            "success": dict(query_pairs).get("success"),
        },
    )


@app.post("/ui/knowledge/upload")
async def panel_upload_knowledge(request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request, admin_only=True)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(request, {"file", "csrf_token"})
    _validate_panel_csrf(form, payload)
    file = form["file"]
    if not isinstance(file, StarletteUploadFile):
        raise HTTPException(400, "Invalid upload")
    try:
        result = await _ingest_knowledge_document(file, db)
    except HTTPException as exc:
        documents = list(db.scalars(
            select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
        ))
        return templates.TemplateResponse(
            request=request, name="knowledge.html",
            context={
                "request": request, "role": payload["role"], "documents": documents,
                "csrf_token": payload["csrf"], "error": str(exc.detail), "success": None,
            },
            status_code=exc.status_code,
        )
    from urllib.parse import quote
    message = f"Uploaded {result['title']} ({result['chunks']} chunks)"
    return RedirectResponse(f"/ui/knowledge?success={quote(message)}", status_code=303)


@app.post("/api/v1/tickets/{ticket_id}/messages", status_code=201)
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
        msg = _record_ticket_message(db, ticket, data)
    return {"id": str(msg.id), "created_at": msg.created_at}


@app.post("/api/v1/ai/analyze", response_model=Analysis)
async def ai_analyze(data: MessageCreate, role: str = Depends(require("agent", "admin")),
                     db: Session = Depends(get_db)):
    try:
        return await analyze(data.body)
    except (httpx.HTTPError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, f"AI unavailable: {exc.__class__.__name__}") from exc


async def _ingest_knowledge_document(file: UploadFile, db: Session) -> dict:
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


@app.post("/api/v1/knowledge/documents", status_code=201)
async def upload_document(file: UploadFile = File(...), role: str = Depends(require("admin")),
                          db: Session = Depends(get_db)):
    return await _ingest_knowledge_document(file, db)


@app.get("/api/v1/knowledge/documents")
def list_documents(role: str = Depends(require("agent", "admin")), db: Session = Depends(get_db)):
    return [{"id": str(d.id), "title": d.title, "source_name": d.source_name, "status": d.status}
            for d in db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()))]


@app.get("/api/v1/analytics/overview")
def overview(role: str = Depends(require("agent", "admin")), db: Session = Depends(get_db)):
    return {status: count for status, count in db.execute(
        select(Ticket.status, func.count()).group_by(Ticket.status)).all()}


@app.post("/integrations/telegram/webhook")
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
    if settings.n8n_webhook_url:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(210, connect=5)) as client:
                response = await client.post(
                    settings.n8n_webhook_url, json={"message_id": str(message_id)},
                    headers={"X-Workflow-Token": settings.n8n_ingress_token},
                )
                response.raise_for_status()
            return
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            logger.warning("telegram_n8n_fallback message_id=%s error_type=%s",
                           message_id, type(exc).__name__)

    # A timed-out workflow may still be processing this row. Use the same lock and saved result.
    with SessionLocal() as db:
        msg = await run_in_threadpool(db.get, Message, message_id)
        if msg is not None:
            await process_message(db, msg)


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


@app.post("/api/v1/internal/process-message")
async def process_message_from_workflow(payload: WorkflowMessageRequest,
                                        x_workflow_token: str | None = Header(default=None),
                                        db: Session = Depends(get_db)):
    if not settings.workflow_token or not secrets.compare_digest(x_workflow_token or "", settings.workflow_token):
        raise HTTPException(401, "Invalid workflow token")
    msg = await run_in_threadpool(db.get, Message, payload.message_id)
    if msg is None:
        raise HTTPException(404, "Message not found")
    return {"message_id": str(msg.id), "reply": await process_message(db, msg)}
