"""Outbound Telegram Bot API calls."""

import uuid

import httpx
from fastapi import HTTPException

from app.config import settings


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
