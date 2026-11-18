"""PostgreSQL admission limits for Telegram updates and escalations."""

from datetime import datetime, timedelta, timezone
from math import ceil
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Message, Ticket

QUESTION_LIMIT = 10
COMMAND_LIMIT = 30
TICKET_LIMIT = 3
UPDATE_WINDOW = timedelta(minutes=1)
TICKET_WINDOW = timedelta(minutes=10)
QUESTION_NOTICE = "Вы отправили 10 вопросов за минуту. Подождите до минуты и повторите вопрос."


def lock_telegram_queue(db: Session) -> None:
    # The check and insert share one transaction across API replicas.
    db.execute(select(func.pg_advisory_xact_lock(17091, 0)))


def telegram_queue_full(db: Session, limit: int) -> bool:
    return db.scalar(select(Message.id).where(
        Message.channel == "telegram", Message.processing_due_at.is_not(None),
    ).offset(limit - 1).limit(1)) is not None


def update_limit_reached(db: Session, requester_id: UUID, kind: str, now: datetime) -> bool:
    return db.scalar(select(Message.id).where(
        Message.requester_id == requester_id, Message.channel == "telegram",
        Message.created_at > now - UPDATE_WINDOW,
        Message.analysis["telegram_update_kind"].as_string() == kind,
    ).offset({"questions": QUESTION_LIMIT, "commands": COMMAND_LIMIT, "notice": 1}[kind] - 1)
      .limit(1)) is not None


def lock_ticket_creation(db: Session, requester_id: UUID) -> None:
    db.execute(select(func.pg_advisory_xact_lock(17092, func.hashtext(str(requester_id)))))


def telegram_ticket_retry_after(db: Session, requester_id: UUID) -> int | None:
    now = datetime.now(timezone.utc)
    boundary = db.scalar(select(Ticket.created_at).where(
        Ticket.requester_id == requester_id, Ticket.created_at > now - TICKET_WINDOW,
    ).order_by(Ticket.created_at.desc()).offset(TICKET_LIMIT - 1).limit(1))
    if boundary is not None:
        return max(1, ceil((boundary + TICKET_WINDOW - now).total_seconds()))
    return None
