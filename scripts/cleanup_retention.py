"""Remove records past the 365-day retention period in bounded transactions.

Dry-run is the default. Pass --execute to perform deletion.
"""
import argparse
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from app.db import SessionLocal
from app.models import AuditLog, KnowledgeDocument, Message, Ticket

RETENTION_DAYS = 365
DEFAULT_BATCH_SIZE = 500
MAX_BATCH_SIZE = 5000


def _delete_ticket_batch(cutoff, batch_size):
    with SessionLocal.begin() as session:
        ticket_ids = list(session.scalars(
            select(Ticket.id)
            .where(Ticket.status == "closed", Ticket.closed_at < cutoff)
            .order_by(Ticket.closed_at.asc(), Ticket.id.asc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        ))
        if not ticket_ids:
            return 0, 0
        messages = session.execute(
            delete(Message).where(Message.ticket_id.in_(ticket_ids))
        ).rowcount or 0
        tickets = session.execute(
            delete(Ticket).where(
                Ticket.id.in_(ticket_ids),
                Ticket.status == "closed",
                Ticket.closed_at < cutoff,
            )
        ).rowcount or 0
        return tickets, messages


def _delete_batch(model, predicate, ordering, batch_size):
    criteria = predicate if isinstance(predicate, tuple) else (predicate,)
    with SessionLocal.begin() as session:
        row_ids = list(session.scalars(
            select(model.id).where(*criteria).order_by(*ordering).limit(batch_size)
            .with_for_update(skip_locked=True)
        ))
        if not row_ids:
            return 0
        return session.execute(delete(model).where(model.id.in_(row_ids), *criteria)).rowcount or 0


def _dry_run_counts(cutoff):
    with SessionLocal() as session:
        closed_ticket_filter = (Ticket.status == "closed", Ticket.closed_at < cutoff)
        ticket_count = session.scalar(
            select(func.count()).select_from(Ticket).where(*closed_ticket_filter)
        ) or 0
        attached_message_count = session.scalar(
            select(func.count()).select_from(Message).join(Ticket, Message.ticket_id == Ticket.id)
            .where(*closed_ticket_filter)
        ) or 0
        document_count = session.scalar(
            select(func.count()).select_from(KnowledgeDocument)
            .where(KnowledgeDocument.created_at < cutoff)
        ) or 0
        audit_count = session.scalar(
            select(func.count()).select_from(AuditLog).where(AuditLog.created_at < cutoff)
        ) or 0
        unattached_count = session.scalar(
            select(func.count()).select_from(Message)
            .where(Message.ticket_id.is_(None), Message.created_at < cutoff)
        ) or 0
    return {
        "tickets": ticket_count,
        "messages": attached_message_count + unattached_count,
        "documents": document_count,
        "audit_logs": audit_count,
    }


def cleanup(*, dry_run=True, batch_size=DEFAULT_BATCH_SIZE, now=None):
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=RETENTION_DAYS)
    if dry_run:
        return _dry_run_counts(cutoff)

    totals = {"tickets": 0, "messages": 0, "documents": 0, "audit_logs": 0}
    while True:
        tickets, messages = _delete_ticket_batch(cutoff, batch_size)
        totals["tickets"] += tickets
        totals["messages"] += messages
        if tickets == 0:
            break

    for model, predicate, ordering, key in (
        (KnowledgeDocument, KnowledgeDocument.created_at < cutoff,
         (KnowledgeDocument.created_at.asc(), KnowledgeDocument.id.asc()), "documents"),
        (AuditLog, AuditLog.created_at < cutoff,
         (AuditLog.created_at.asc(), AuditLog.id.asc()), "audit_logs"),
        (Message, (Message.ticket_id.is_(None), Message.created_at < cutoff),
         (Message.created_at.asc(), Message.id.asc()), "messages"),
    ):
        while True:
            removed = _delete_batch(model, predicate, ordering, batch_size)
            totals[key] += removed
            if removed == 0:
                break
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="perform deletion (default is dry-run)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"rows per transaction, 1..{MAX_BATCH_SIZE}")
    args = parser.parse_args()
    counts = cleanup(dry_run=not args.execute, batch_size=args.batch_size)
    print(f"{'Would delete' if not args.execute else 'Deleted'}: {counts}")


if __name__ == "__main__":
    main()
