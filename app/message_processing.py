"""Classify a message, persist its outcome, and deliver its Telegram reply."""

import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter

import httpx
from fastapi import HTTPException
from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.ai import AIResponseError, analyze, grounded_answer
from app.config import settings
from app.gemini import GeminiError
from app.models import Message, Ticket
from app.rag import search
from app.schemas import Analysis, TicketCreate
from app.telegram import (
    HISTORY_ENTRY, MAIN_KEYBOARD, MENU_COMMANDS, followup_prompt, send_telegram_message, send_telegram_photo,
)
from app.telegram_limits import QUESTION_NOTICE, lock_ticket_creation, telegram_ticket_retry_after
from app.ticket_writes import create_ticket


logger = logging.getLogger("uvicorn.error").getChild(__name__)
TICKET_STATUSES = {
    "open": "открыто", "in_progress": "в работе", "waiting_for_user": "ждём вашего ответа",
    "resolved": "решено", "closed": "закрыто",
}


def _ticket_buttons(msg: Message, rows: list[list[tuple[str, str]]]) -> None:
    msg.analysis = {**(msg.analysis or {}), "reply_markup": {"inline_keyboard": [
        [{"text": label, "callback_data": data} for label, data in row] for row in rows
    ]}}


def _ticket_history(db: Session, msg: Message, command: str, argument: str) -> str:
    visible_sender = (Message.sender_type == "agent") | (
        (Message.sender_type == "user") & (Message.requester_id == msg.requester_id)
    )
    part = 0
    if command == "/history":
        ticket = db.scalar(select(Ticket).where(
            Ticket.requester_id == msg.requester_id, Ticket.ticket_number == argument.upper(),
        ))
        if ticket is None:
            return "Обращение не найдено."
        entry = db.scalar(select(Message).where(Message.ticket_id == ticket.id, visible_sender)
                          .order_by(Message.created_at.desc(), Message.id.desc()).limit(1))
    else:
        match = HISTORY_ENTRY.fullmatch(argument)
        if match is None:
            return "Сообщение не найдено."
        entry_id, part_index = match.groups()
        part = int(part_index)
        result = db.execute(select(Message, Ticket).join(Ticket, Message.ticket_id == Ticket.id).where(
            Message.id == uuid.UUID(hex=entry_id), visible_sender,
            Ticket.requester_id == msg.requester_id,
        )).one_or_none()
        if result is None:
            return "Сообщение не найдено."
        entry, ticket = result

    back = [("К карточке", f"ticket:{ticket.ticket_number}")]
    if entry is None:
        _ticket_buttons(msg, [back])
        return f"История {ticket.ticket_number}\nВ обращении пока нет сообщений."

    body = entry.body
    photo_id = (entry.analysis or {}).get("photo_id")
    has_photo = (entry.channel == "telegram" and entry.sender_type == "user"
                 and isinstance(photo_id, str) and 1 <= len(photo_id) <= 512)
    if has_photo:
        body = "Скриншот" if body == "Screenshot attached" else f"Скриншот\n{body}"
    # Leave room for the header even when every character takes two UTF-16 code units.
    part_size = 1800
    parts = max(1, (len(body) + part_size - 1) // part_size)
    if part >= parts:
        _ticket_buttons(msg, [back])
        return "Часть сообщения не найдена."
    if command == "/history_photo":
        _ticket_buttons(msg, [[("К истории", f"history_entry:{entry.id.hex}:{part}")], back])
        if not has_photo:
            return "Скриншот не найден."
        msg.analysis = {**msg.analysis, "reply_photo_id": photo_id}
        return f"Скриншот к обращению {ticket.ticket_number}"
    rows = []
    if has_photo:
        rows.append([("Открыть скриншот", f"history_photo:{entry.id.hex}:{part}")])
    if parts > 1:
        rows.append([
            (label, f"history_entry:{entry.id.hex}:{index}")
            for label, index in (("Предыдущая часть", part - 1), ("Следующая часть", part + 1))
            if 0 <= index < parts
        ])
    entries = select(Message.id).where(Message.ticket_id == ticket.id, visible_sender)
    position = tuple_(Message.created_at, Message.id)
    cursor = tuple_(entry.created_at, entry.id)
    older = db.scalar(entries.where(position < cursor)
                      .order_by(Message.created_at.desc(), Message.id.desc()).limit(1))
    newer = db.scalar(entries.where(position > cursor)
                      .order_by(Message.created_at.asc(), Message.id.asc()).limit(1))
    navigation = [(label, f"history_entry:{item.hex}:0")
                  for label, item in (("Раньше", older), ("Позже", newer)) if item is not None]
    if navigation:
        rows.append(navigation)
    _ticket_buttons(msg, [*rows, back])
    author = "Специалист" if entry.sender_type == "agent" else "Вы"
    number = f" · часть {part + 1}/{parts}" if parts > 1 else ""
    date = entry.created_at.astimezone(timezone.utc)
    excerpt = body[part * part_size:(part + 1) * part_size]
    return f"История {ticket.ticket_number}\n{author} · {date:%d.%m.%Y %H:%M} UTC{number}\n\n{excerpt}"


def _create_telegram_ticket(db: Session, msg: Message, data: TicketCreate) -> tuple[Ticket | None, str | None]:
    lock_ticket_creation(db, msg.requester_id)
    existing = db.scalar(select(Ticket).where(
        Ticket.requester_id == msg.requester_id,
        Ticket.status.in_(("open", "in_progress", "waiting_for_user")),
        func.lower(func.btrim(Ticket.description)) == func.lower(func.btrim(data.description)),
    ).order_by(Ticket.created_at.desc(), Ticket.id.desc()).limit(1))
    if existing is not None:
        _ticket_buttons(msg, [[("Открыть обращение", f"ticket:{existing.ticket_number}")]])
        return None, f"Такое обращение уже открыто: {existing.ticket_number}. Его можно дополнить из карточки."
    retry_after = telegram_ticket_retry_after(db, msg.requester_id)
    if retry_after is not None:
        _ticket_buttons(msg, [[("Мои обращения", "tickets")]])
        return None, (
            "Достигнут лимит: 3 новых обращения за 10 минут. "
            f"Повторите попытку через {retry_after} сек. или дополните существующее обращение."
        )
    return create_ticket(db, msg.requester_id, data), None


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
            analysis = msg.analysis or {}
            markup = analysis.get("reply_markup")
            if photo_id := analysis.get("reply_photo_id"):
                try:
                    await send_telegram_photo(msg.telegram_chat_id, photo_id, reply, reply_markup=markup)
                except HTTPException as exc:
                    if exc.status_code != 404:
                        raise
                    await send_telegram_message(
                        msg.telegram_chat_id, "Скриншот недоступен в Telegram.", reply_markup=markup,
                    )
            elif markup:
                await send_telegram_message(msg.telegram_chat_id, reply, reply_markup=markup)
            elif analysis.get("knowledge_answer") and msg.id:
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
    command = MENU_COMMANDS.get(msg.body.casefold(), command)
    if command == "/rate-limit" and (msg.analysis or {}).get("telegram_update_kind") == "notice":
        return QUESTION_NOTICE
    if command in {"/start", "/help", "/menu"}:
        msg.analysis = {**(msg.analysis or {}), "reply_markup": MAIN_KEYBOARD}
        return "Выберите действие. Также можно сразу написать вопрос."
    if command == "/question":
        _ticket_buttons(msg, [[("Назад", "menu")]])
        return "Опишите проблему одним сообщением. Я поищу решение в базе знаний."
    if command == "/tickets":
        tickets = list(db.scalars(select(Ticket).where(Ticket.requester_id == msg.requester_id)
                                  .order_by(Ticket.updated_at.desc(), Ticket.id.desc()).limit(10)))
        rows = [[(ticket.ticket_number, f"ticket:{ticket.ticket_number}")] for ticket in tickets]
        _ticket_buttons(msg, [*rows, [("Назад", "menu")]])
        if not tickets:
            return "У вас пока нет обращений."
        lines = [f"{ticket.ticket_number} · {TICKET_STATUSES.get(ticket.status, ticket.status)} · "
                 f"{ticket.updated_at:%d.%m %H:%M} UTC\n{ticket.subject[:120]}" for ticket in tickets]
        return "Ваши последние обращения\n\n" + "\n".join(lines)
    if command in {"/ticket", "/ticket_text", "/ticket_photo"}:
        ticket = db.scalar(select(Ticket).where(
            Ticket.requester_id == msg.requester_id, Ticket.ticket_number == argument.strip().upper(),
        ))
        back = [("Назад", "tickets")]
        if ticket is None:
            _ticket_buttons(msg, [back])
            return "Обращение не найдено."
        if command == "/ticket":
            rows = [[("История", f"history:{ticket.ticket_number}")], back]
            if ticket.status != "closed":
                rows = [[("Добавить сообщение", f"ticket_text:{ticket.ticket_number}")],
                        [("Прикрепить скриншот", f"ticket_photo:{ticket.ticket_number}")], *rows]
            _ticket_buttons(msg, rows)
            status = TICKET_STATUSES.get(ticket.status, ticket.status)
            return f"{ticket.ticket_number}\n{ticket.subject}\nСтатус: {status}"
        if ticket.status == "closed":
            _ticket_buttons(msg, [back])
            return "Обращение закрыто. Отправьте новый вопрос, чтобы создать новое обращение."
        msg.analysis = {**(msg.analysis or {}), "reply_markup": {"force_reply": True}}
        return followup_prompt(command[1:], ticket.ticket_number)
    if command in {"/history", "/history_entry", "/history_photo"}:
        return _ticket_history(db, msg, command, argument.strip())
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
        _ticket_buttons(msg, [[("Открыть обращение", f"ticket:{ticket.ticket_number}")]])
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
                _ticket_buttons(msg, [[("Открыть обращение", f"ticket:{ticket.ticket_number}")]])
                return f"Для этого вопроса уже создано обращение {ticket.ticket_number}."
            return "Для этого вопроса уже создавалось обращение."
        if command == "/helped":
            original.analysis = {**analysis, "feedback": "helped"}
            return "Спасибо за обратную связь."
        classification = Analysis.model_validate(analysis["classification"])
        ticket, limited_reply = _create_telegram_ticket(db, msg, TicketCreate(
            subject=classification.summary.strip()[:240] if len(classification.summary.strip()) >= 3 else "Support request",
            description=original.body, category=classification.category,
            subcategory=classification.subcategory, priority=classification.priority,
        ))
        if ticket is None:
            return limited_reply
        original.ticket_id = ticket.id
        original.analysis = {**analysis, "escalation_reason": "user_requested_specialist"}
        _ticket_buttons(msg, [[("Открыть обращение", f"ticket:{ticket.ticket_number}")]])
        return f"Создал обращение {ticket.ticket_number}. Специалист продолжит разбор."
    help_text = (
        "Мои обращения — /tickets. Дополнить обращение — /add НОМЕР текст. "
        "Новый вопрос можно отправить обычным сообщением."
    )
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
        msg.body.startswith("/") or msg.body.casefold() in MENU_COMMANDS
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
        if msg.channel == "telegram":
            ticket, reply = _create_telegram_ticket(db, msg, ticket_data)
        else:
            ticket, reply = create_ticket(db, msg.requester_id, ticket_data), None
        if ticket is not None:
            msg.ticket_id = ticket.id
        if ticket is not None and reason == "ai_error":
            reply = f"Не удалось завершить автоматическую диагностику. Создал обращение {ticket.ticket_number}."
        elif ticket is not None:
            reply = f"Создал обращение {ticket.ticket_number}. Специалист продолжит разбор."
        msg.analysis = {**(msg.analysis or {}), "reply": reply, "escalation_reason": reason}
        if classification is not None:
            msg.analysis["classification"] = classification.model_dump()
    else:
        answer_text, sources = answer
        reply = f"{answer_text}\n\nИсточник: {', '.join(sources)}"
        msg.analysis = {**(msg.analysis or {}), "classification": classification.model_dump(), "reply": reply,
                        "knowledge_answer": answer_text, "sources": sources}
    db.commit()
    return reply
