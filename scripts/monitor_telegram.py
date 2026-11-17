"""Check Telegram ingress and the persisted processing queue."""

import logging
import os
import time

import httpx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.db import engine

logger = logging.getLogger(__name__)


def check_webhook(client, bot, webhook):
    try:
        info = client.get(f"{bot}/getWebhookInfo")
        info.raise_for_status()
        result = info.json()
        if result["ok"] is not True or result["result"]["url"] != webhook:
            return "Telegram webhook registration mismatch"
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return "Telegram webhook registration could not be checked"
    try:
        # Check the public tunnel, ingress and API without disclosing the webhook secret.
        response = client.post(webhook, json={})
    except httpx.HTTPError:
        return "public webhook unreachable"
    if response.status_code != 401:
        return f"webhook HTTP {response.status_code}"
    return None


def queue_problem(count, age, count_limit, age_limit):
    if count >= count_limit or age >= age_limit:
        return f"queue: {count} pending; oldest {age}s"
    return None


def check_n8n(client, url):
    if not url:
        return None
    try:
        # The registered workflow rejects unauthenticated requests before executing.
        response = client.post(url, json={})
    except (httpx.HTTPError, httpx.InvalidURL):
        return "n8n webhook unreachable; FastAPI fallback available"
    if response.status_code not in {401, 403}:
        return f"n8n webhook HTTP {response.status_code}; FastAPI fallback available"
    return None


def check_queue(count_limit, age_limit):
    try:
        with engine.connect() as connection:
            count, age = connection.execute(text("""
                SELECT count(*), coalesce(extract(epoch FROM now() - min(created_at)), 0)
                FROM messages WHERE processing_due_at IS NOT NULL
            """)).one()
    except SQLAlchemyError:
        return "queue check failed: database unavailable"
    return queue_problem(count, max(0, int(age)), count_limit, age_limit)


def send_alert(client, bot, chat_id, message):
    try:
        response = client.post(f"{bot}/sendMessage", json={"chat_id": chat_id, "text": message})
        response.raise_for_status()
        return response.json()["ok"] is True
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return False


def notify(client, bot, chat_id, previous, problems):
    current = frozenset(problems)
    if current == previous:
        return previous
    changes = [problems[key] for key in sorted(current - previous)]
    if not current:
        changes.append("monitoring checks recovered")
    elif cleared := previous - current:
        changes.append(f"recovered: {', '.join(sorted(cleared))}")
    message = "Service Desk: " + "; ".join(changes)
    logger.warning("%s", message)
    if send_alert(client, bot, chat_id, message):
        return current
    logger.error("Telegram alert delivery failed; will retry on the next check")
    return previous


def main():
    webhook = os.environ["TELEGRAM_PUBLIC_WEBHOOK"]
    chat_id = os.environ["TELEGRAM_ALERT_CHAT_ID"]
    count_limit = int(os.getenv("TELEGRAM_QUEUE_ALERT_COUNT", "10"))
    age_limit = int(os.getenv("TELEGRAM_QUEUE_ALERT_AGE_SECONDS", "120"))
    if min(count_limit, age_limit) < 1:
        raise ValueError("Telegram queue thresholds must be positive")
    bot = f"https://api.telegram.org/bot{settings.telegram_token}"
    previous = frozenset()
    with httpx.Client(timeout=10) as client:
        while True:
            checks = {
                "webhook": check_webhook(client, bot, webhook),
                "queue": check_queue(count_limit, age_limit),
                "n8n": check_n8n(client, settings.n8n_webhook_url),
            }
            problems = {key: problem for key, problem in checks.items() if problem}
            previous = notify(client, bot, chat_id, previous, problems)
            time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
