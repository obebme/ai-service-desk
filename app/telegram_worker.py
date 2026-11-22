"""Dispatch Telegram jobs without occupying workers while n8n responds."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app import message_processing
from app.config import settings
from app.db import SessionLocal
from app.models import Message


logger = logging.getLogger("uvicorn.error").getChild(__name__)


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
        local_fallback = bool((msg.analysis or {}).get("telegram_local_fallback"))
        if local_fallback and message_processing.ai_processing_active(msg):
            # n8n already started processing. Check again later without waiting inside a worker.
            msg.processing_due_at = now + timedelta(seconds=1)
            return None
        # A crashed worker leaves a recoverable claim. Normal failures retry sooner.
        msg.processing_due_at = now + timedelta(minutes=5)
        return msg.id, msg.processing_due_at, local_fallback


def _finish_telegram_attempt(message_id, claim_until, succeeded):
    with SessionLocal.begin() as db:
        msg = db.get(Message, message_id, with_for_update=True)
        if msg is not None and msg.processing_due_at == claim_until:
            msg.processing_due_at = None if succeeded else datetime.now(timezone.utc) + timedelta(seconds=15)


def _complete_n8n_attempt(message_id, claim_until, acknowledged):
    with SessionLocal.begin() as db:
        msg = db.get(Message, message_id, with_for_update=True)
        # Gemini 429 or another worker may already have replaced this lease.
        if msg is None or msg.processing_due_at != claim_until:
            return
        analysis = msg.analysis or {}
        if acknowledged and analysis.get("reply"):
            msg.processing_due_at = None
            return
        if acknowledged:
            logger.warning("telegram_n8n_fallback message_id=%s reason=missing_saved_reply", message_id)
        msg.analysis = {**analysis, "telegram_local_fallback": True}
        msg.processing_due_at = datetime.now(timezone.utc)


async def _run_n8n_attempt(message_id, claim_until):
    acknowledged = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(210, connect=5)) as client:
            response = await client.post(
                settings.n8n_webhook_url, json={"message_id": str(message_id)},
                headers={"X-Workflow-Token": settings.n8n_ingress_token},
            )
            response.raise_for_status()
        acknowledged = True
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        logger.warning("telegram_n8n_fallback message_id=%s error_type=%s", message_id, type(exc).__name__)
    try:
        await run_in_threadpool(_complete_n8n_attempt, message_id, claim_until, acknowledged)
    except SQLAlchemyError as exc:
        # The queue lease makes the job available again if we cannot complete or requeue it.
        logger.warning("telegram_queue_error error_type=%s", type(exc).__name__)


async def _dispatch_telegram_message(message_id):
    db = SessionLocal()
    try:
        msg = await run_in_threadpool(db.get, Message, message_id)
        if msg is None:
            return
        if message_processing.gemini_retry_pending(msg):
            return
        await message_processing.process_message(db, msg)
    finally:
        await run_in_threadpool(db.close)


async def run_telegram_workers():
    async with asyncio.TaskGroup() as tasks:
        pending_n8n = set()
        for _ in range(settings.telegram_workers):
            tasks.create_task(_run_telegram_worker(tasks, pending_n8n))


async def _run_telegram_worker(tasks, pending_n8n):
    while True:
        try:
            claim = await run_in_threadpool(_claim_telegram_message)
            if claim is None:
                await asyncio.sleep(0.25)
                continue
            message_id, claim_until, local_fallback = claim
            if settings.n8n_webhook_url and not local_fallback and len(pending_n8n) < settings.telegram_workers:
                task = tasks.create_task(_run_n8n_attempt(message_id, claim_until))
                pending_n8n.add(task)
                task.add_done_callback(pending_n8n.discard)
                continue
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
