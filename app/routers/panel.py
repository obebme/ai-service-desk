"""Server-rendered support panel."""

import logging
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from jinja2 import TemplateNotFound
from pydantic import ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.templating import Jinja2Templates

from app import panel_auth
from app.config import settings
from app.db import get_db
from app.knowledge_ingest import ingest_knowledge_document
from app.models import AuditLog, KnowledgeDocument, Message, Ticket
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
from app.schemas import MessageCreate, TicketPatch, describe_responses
from app.telegram import send_telegram_message
from app.ticket_writes import record_ticket_message, record_ticket_update


logger = logging.getLogger("uvicorn.error").getChild(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / "templates"))


@router.get("/ui/login", response_class=HTMLResponse, responses=describe_responses(303))
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



@router.post(
    "/ui/login",
    status_code=303, response_class=RedirectResponse,
    responses=describe_responses(400, 401),
)
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
    role = authenticate_role_token(token, panel_auth.ROLE_TOKENS)
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



@router.post("/ui/logout", status_code=303, response_class=RedirectResponse, responses=describe_responses(400))
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



@router.get("/ui/tickets", response_class=HTMLResponse, responses=describe_responses(303, 403))
def panel_ticket_queue(request: Request, db: Session = Depends(get_db)):
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



@router.get("/ui/tickets/{ticket_id}", response_class=HTMLResponse, responses=describe_responses(303, 404))
def panel_ticket_detail(ticket_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
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



@router.get("/ui/tickets/{ticket_id}/photos/{message_id}", responses=describe_responses(303, 404, 503))
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



@router.post(
    "/ui/tickets/{ticket_id}/claim",
    status_code=303, response_class=RedirectResponse,
    responses=describe_responses(400, 403, 404, 409),
)
async def panel_claim_ticket(ticket_id: uuid.UUID, request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(request, {"csrf_token"})
    _validate_panel_csrf(form, payload)
    agent = payload.get("agent")
    if not agent or agent not in settings.panel_agent_tokens:
        raise HTTPException(403, "Personal agent session required")
    await run_in_threadpool(_claim_ticket, db, ticket_id, agent)
    return RedirectResponse(f"/ui/tickets/{ticket_id}", status_code=303)


def _claim_ticket(db: Session, ticket_id: uuid.UUID, agent: str) -> None:
    ticket = db.get(Ticket, ticket_id, with_for_update=True)
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



@router.post(
    "/ui/tickets/{ticket_id}/messages",
    status_code=303, response_class=RedirectResponse,
    responses=describe_responses(400, 404),
)
async def panel_add_ticket_message(ticket_id: uuid.UUID, request: Request,
                                   db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(request, {"body", "csrf_token"})
    _validate_panel_csrf(form, payload)
    prepared = await run_in_threadpool(_prepare_ticket_reply, db, ticket_id, form, request, payload)
    if isinstance(prepared, Response):
        return prepared
    saved_message, chat_id = prepared
    if chat_id is not None and settings.telegram_token:
        try:
            await send_telegram_message(chat_id, saved_message.body)
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            status = "unconfirmed"
        else:
            status = "accepted"
        await run_in_threadpool(_record_reply_delivery, db, saved_message, status)
    return RedirectResponse(
        f"/ui/tickets/{ticket_id}?message_id={saved_message.id}", status_code=303
    )


def _prepare_ticket_reply(
    db: Session, ticket_id: uuid.UUID, form: dict, request: Request, payload: dict,
) -> Response | tuple[Message, str | None]:
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
    saved_message = record_ticket_message(
        db, ticket, data, telegram_delivery_status=delivery_status
    )
    return saved_message, recipient.telegram_chat_id if recipient is not None else None


def _record_reply_delivery(db: Session, saved_message: Message, status: str) -> None:
    saved_message.telegram_delivery_status = status
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



@router.post(
    "/ui/tickets/{ticket_id}/update",
    status_code=303, response_class=RedirectResponse,
    responses=describe_responses(400, 403, 404),
)
async def panel_update_ticket(ticket_id: uuid.UUID, request: Request,
                              db: Session = Depends(get_db)):
    payload = _panel_session(request)
    if isinstance(payload, RedirectResponse):
        return payload
    form = await _panel_form(
        request, {"csrf_token"}, {"csrf_token", "status", "priority", "resolution"}
    )
    _validate_panel_csrf(form, payload)
    patch_data = {
        key: (None if key == "resolution" and value == "" else value)
        for key, value in form.items() if key != "csrf_token"
    }
    try:
        data = TicketPatch.model_validate(patch_data)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    await run_in_threadpool(record_ticket_update, db, ticket_id, data)
    return RedirectResponse(f"/ui/tickets/{ticket_id}", status_code=303)



@router.get("/ui/knowledge", response_class=HTMLResponse, responses=describe_responses(303, 403))
def panel_knowledge(request: Request, db: Session = Depends(get_db)):
    payload = _panel_session(request, admin_only=True)
    if isinstance(payload, RedirectResponse):
        return payload
    query_pairs = request.query_params.multi_items()
    if (len(query_pairs) != len({key for key, _value in query_pairs})
            or any(key != "success" for key, _value in query_pairs)
            or any(len(value) > 500 for _key, value in query_pairs)):
        raise HTTPException(422, "Invalid knowledge-page query")
    documents = _list_documents(db)
    return templates.TemplateResponse(
        request=request, name="knowledge.html",
        context={
            "request": request, "role": payload["role"], "documents": documents,
            "csrf_token": payload["csrf"], "error": None,
            "success": dict(query_pairs).get("success"),
        },
    )


def _list_documents(db: Session) -> list[KnowledgeDocument]:
    return list(db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())))



@router.post(
    "/ui/knowledge/upload",
    status_code=303, response_class=RedirectResponse,
    responses=describe_responses(400, 403, 411, 413, 415, 503),
)
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
        result = await ingest_knowledge_document(file, db)
    except HTTPException as exc:
        documents = await run_in_threadpool(_list_documents, db)
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
