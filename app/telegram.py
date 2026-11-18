"""Outbound Telegram Bot API calls."""

import re
import uuid

import httpx
from fastapi import HTTPException

from app.config import settings


MENU_CALLBACKS = frozenset({"menu", "question", "tickets"})
QUESTION_BUTTON = "Задать вопрос"
TICKETS_BUTTON = "Мои обращения"
MENU_COMMANDS = {QUESTION_BUTTON.casefold(): "/question", TICKETS_BUTTON.casefold(): "/tickets"}
MAIN_KEYBOARD = {
    "keyboard": [[{"text": QUESTION_BUTTON}, {"text": TICKETS_BUTTON}]],
    "resize_keyboard": True,
    "is_persistent": True,
}
TICKET_CALLBACKS = frozenset({"ticket", "ticket_text", "ticket_photo", "history"})
TICKET_NUMBER = re.compile(r"[A-Z0-9-]{1,24}")
HISTORY_ENTRY = re.compile(r"([0-9a-f]{32}):([0-9]{1,6})")
FOLLOWUP_PROMPTS = {
    "ticket_text": "Ответьте на это сообщение текстом.",
    "ticket_photo": "Ответьте на это сообщение фотографией или скриншотом.",
}


def followup_prompt(action: str, number: str) -> str:
    return f"Дополнение к обращению {number}\n{FOLLOWUP_PROMPTS[action]}"


def replied_ticket_number(message: dict) -> str | None:
    reply = message.get("reply_to_message")
    sender = reply.get("from") if isinstance(reply, dict) else None
    bot_id = settings.telegram_token.partition(":")[0]
    if (not bot_id.isascii() or not bot_id.isdecimal() or not isinstance(sender, dict)
            or sender.get("is_bot") is not True or str(sender.get("id")) != bot_id):
        return None
    text = reply.get("text")
    if not isinstance(text, str):
        return None
    heading, _, instructions = text.partition("\n")
    prefix = "Дополнение к обращению "
    if heading.startswith(prefix) and instructions in FOLLOWUP_PROMPTS.values():
        number = heading.removeprefix(prefix)
        if TICKET_NUMBER.fullmatch(number):
            return number
    return None


async def send_telegram_message(
    chat_id: str, message: str, *, answer_id: uuid.UUID | None = None, reply_markup: dict | None = None,
):
    payload = {"chat_id": chat_id, "text": message[:4000]}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    elif answer_id:
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
