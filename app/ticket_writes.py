"""Shared persistence for ticket updates and replies."""

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import AuditLog, Message, Ticket, User
from app.schemas import MessageCreate, TicketPatch


def record_ticket_update(db: Session, ticket_id: uuid.UUID, data: TicketPatch) -> Ticket:
    ticket = db.get(Ticket, ticket_id, with_for_update=True, populate_existing=True)
    if ticket is None:
        raise HTTPException(404, "Ticket not found")
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



def record_ticket_message(
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
