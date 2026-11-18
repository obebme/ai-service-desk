import asyncio
import uuid
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app import message_processing, telegram
from app.models import Message
from app.schemas import Analysis


def test_no_verified_rag_answer_creates_ticket(monkeypatch):
    requester_id = uuid.uuid4()
    user = SimpleNamespace(id=requester_id, telegram_user_id=None)
    msg = Message(requester_id=requester_id, body="VPN still fails", channel="api")
    created = []

    class FakeDB:
        def get(self, model, key):
            return user

        def commit(self):
            pass

        def rollback(self):
            raise AssertionError("normal escalation should not roll back")

    async def fake_analyze(body):
        return Analysis(category="network", subcategory="vpn", priority="medium",
                        summary="VPN unavailable", sentiment="neutral", needs_human=False,
                        confidence=0.95, suggested_action="check_knowledge")

    async def fake_search(*args, **kwargs):
        return []

    def fake_create_ticket(db, user_id, data):
        created.append((user_id, data))
        return SimpleNamespace(id=uuid.uuid4(), ticket_number="SD-TEST")

    monkeypatch.setattr(message_processing, "analyze", fake_analyze)
    monkeypatch.setattr(message_processing, "search", fake_search)
    monkeypatch.setattr(message_processing, "create_ticket", fake_create_ticket)
    reply = asyncio.run(message_processing.process_message(FakeDB(), msg))

    assert reply == "Создал обращение SD-TEST. Специалист продолжит разбор."
    assert len(created) == 1
    assert msg.ticket_id is not None
    assert msg.analysis["reply"] == reply


def test_processed_message_reuses_saved_reply_without_new_ticket(monkeypatch):
    msg = Message(requester_id=uuid.uuid4(), body="already handled", channel="api",
                  analysis={"reply": "Saved response"})

    class FakeDB:
        def get(self, model, key):
            return SimpleNamespace(id=msg.requester_id, telegram_user_id=None)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("processed messages should not call model or RAG")

    monkeypatch.setattr(message_processing, "analyze", fail_if_called)
    monkeypatch.setattr(message_processing, "search", fail_if_called)
    assert asyncio.run(message_processing.process_message(FakeDB(), msg)) == "Saved response"


def test_ai_provider_failure_escalates_without_leaking_provider_error(monkeypatch):
    requester_id = uuid.uuid4()
    user = SimpleNamespace(id=requester_id, telegram_user_id=None)
    msg = Message(requester_id=requester_id, body="Cannot access VPN", channel="api")
    tickets = []

    class FakeDB:
        def get(self, model, key):
            return user

        def commit(self):
            pass

    async def fail_analysis(_body):
        raise httpx.ConnectError("provider response contains sensitive details")

    def create_ticket(_db, user_id, data):
        tickets.append((user_id, data))
        return SimpleNamespace(id=uuid.uuid4(), ticket_number="SD-AIFAIL")

    monkeypatch.setattr(message_processing, "analyze", fail_analysis)
    monkeypatch.setattr(message_processing, "create_ticket", create_ticket)
    reply = asyncio.run(message_processing.process_message(FakeDB(), msg))

    assert reply == "Не удалось завершить автоматическую диагностику. Создал обращение SD-AIFAIL."
    assert msg.ticket_id is not None
    assert len(tickets) == 1
    assert "provider response contains sensitive details" not in msg.analysis["reply"]


def test_database_commit_failure_propagates_without_success_reply(monkeypatch):
    requester_id = uuid.uuid4()
    user = SimpleNamespace(id=requester_id, telegram_user_id="test-telegram-user")
    msg = Message(requester_id=requester_id, body="VPN unavailable", channel="telegram")
    sent_replies = []
    commit_attempts = []

    class FakeDB:
        def get(self, model, key):
            return user

        def commit(self):
            commit_attempts.append(True)
            raise RuntimeError("database commit failed")

    async def successful_analysis(_body):
        return Analysis(
            category="network", subcategory="vpn", priority="medium",
            summary="VPN unavailable", sentiment="neutral", needs_human=False,
            confidence=0.95, suggested_action="check_knowledge",
        )

    async def no_evidence(*_args, **_kwargs):
        return []

    def create_ticket(_db, _user_id, _data):
        return SimpleNamespace(id=uuid.uuid4(), ticket_number="SD-DBFAIL")

    async def record_send(_chat_id, reply):
        sent_replies.append(reply)

    monkeypatch.setattr(message_processing.settings, "telegram_token", "test-token")
    monkeypatch.setattr(message_processing, "analyze", successful_analysis)
    monkeypatch.setattr(message_processing, "search", no_evidence)
    monkeypatch.setattr(message_processing, "_create_telegram_ticket",
                        lambda db, msg, data: (create_ticket(db, msg.requester_id, data), None))
    monkeypatch.setattr(message_processing, "send_telegram_message", record_send)

    sentinel = object()
    no_reply = sentinel
    with pytest.raises(RuntimeError, match="database commit failed"):
        no_reply = asyncio.run(message_processing.process_message(FakeDB(), msg))

    assert no_reply is sentinel
    assert commit_attempts == [True]
    assert sent_replies == []


@pytest.mark.parametrize("failure", ["httpx", "non_json", "telegram_rejected"])
def test_telegram_delivery_failures_map_to_safe_503(monkeypatch, failure):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            if failure == "non_json":
                raise ValueError("provider body contains a secret")
            return {"ok": False}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            if failure == "httpx":
                raise httpx.ConnectError("network details include sensitive data")
            return Response()

    monkeypatch.setattr(message_processing.settings, "telegram_token", "telegram-test-secret")
    monkeypatch.setattr(telegram, "httpx", SimpleNamespace(
        AsyncClient=FakeClient, HTTPError=httpx.HTTPError,
    ))

    with pytest.raises(HTTPException) as error:
        asyncio.run(telegram.send_telegram_message("123", "test reply"))

    assert error.value.status_code == 503
    assert error.value.detail == "Telegram message delivery failed"
    assert "secret" not in str(error.value.detail)


def test_saved_telegram_reply_is_delivered_only_once(monkeypatch):
    msg = Message(channel="telegram", requester_id=uuid.uuid4(), body="VPN",
                  telegram_chat_id="123", analysis={"reply": "Saved response"})
    deliveries = []

    class FakeDB:
        def get(self, model, key):
            return SimpleNamespace(id=msg.requester_id, telegram_user_id="123")

        def commit(self):
            pass

    async def send(chat_id, reply):
        deliveries.append((chat_id, reply))

    monkeypatch.setattr(message_processing.settings, "telegram_token", "test-token")
    monkeypatch.setattr(message_processing, "send_telegram_message", send)
    db = FakeDB()
    assert asyncio.run(message_processing.process_message(db, msg)) == "Saved response"
    assert asyncio.run(message_processing.process_message(db, msg)) == "Saved response"
    assert deliveries == [("123", "Saved response")]
    assert msg.telegram_delivery_status == "accepted"


@pytest.mark.parametrize("status", [200, 400])
def test_telegram_photo_reuses_file_id(monkeypatch, status):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, *, json):
            calls.append((url, json))
            return httpx.Response(status, json={"ok": status == 200}, request=httpx.Request("POST", url))

    monkeypatch.setattr(telegram.settings, "telegram_token", "test-token")
    monkeypatch.setattr(telegram, "httpx", SimpleNamespace(
        AsyncClient=Client, HTTPError=httpx.HTTPError, HTTPStatusError=httpx.HTTPStatusError,
    ))
    markup = {"inline_keyboard": [[{"text": "К истории", "callback_data": "history:SD-TEST"}]]}
    if status == 400:
        with pytest.raises(HTTPException) as error:
            asyncio.run(telegram.send_telegram_photo("123", "photo-id", "Скриншот", reply_markup=markup))
        assert error.value.status_code == 404 and "test-token" not in error.value.detail
    else:
        asyncio.run(telegram.send_telegram_photo("123", "photo-id", "Скриншот", reply_markup=markup))
    assert calls == [("https://api.telegram.org/bottest-token/sendPhoto", {
        "chat_id": "123", "photo": "photo-id", "caption": "Скриншот", "reply_markup": markup,
    })]

@pytest.mark.parametrize(("needs_human", "priority", "confidence", "reason"), [
    (True, "medium", 0.95, "requires_specialist"),
    (False, "critical", 0.95, "critical"),
    (False, "medium", 0.1, "low_confidence"),
])
def test_direct_escalation_skips_knowledge_search(monkeypatch, caplog, needs_human, priority, confidence, reason):
    msg = Message(id=uuid.uuid4(), requester_id=uuid.uuid4(), body="Test request", channel="api")

    class DB:
        def commit(self):
            pass

    async def classify(_body):
        return Analysis(category="network", summary="Test request", priority=priority,
                        needs_human=needs_human, confidence=confidence, suggested_action="review")

    async def unexpected(*_args, **_kwargs):
        pytest.fail("Direct escalation must skip RAG")

    def ticket(_db, _requester, data):
        assert data.priority == priority
        assert data.category == "network"
        return SimpleNamespace(id=uuid.uuid4(), ticket_number="SD-TEST")

    monkeypatch.setattr(message_processing, "analyze", classify)
    monkeypatch.setattr(message_processing, "search", unexpected)
    monkeypatch.setattr(message_processing, "grounded_answer", unexpected)
    monkeypatch.setattr(message_processing, "create_ticket", ticket)
    with caplog.at_level("INFO", logger=message_processing.logger.name):
        assert "SD-TEST" in asyncio.run(message_processing.process_message(DB(), msg))
    assert "stage=classification" in caplog.text
    assert f"reason={reason}" in caplog.text
    assert msg.body not in caplog.text

def test_unknown_telegram_command_returns_help_without_classification(monkeypatch):
    msg = Message(channel="telegram", telegram_chat_id="123", requester_id=uuid.uuid4(),
                  body="/cmd/tickets")

    class FakeDB:
        def commit(self):
            pass

    async def unexpected(*args, **kwargs):
        pytest.fail("An unknown command must not enter AI processing")

    monkeypatch.setattr(message_processing.settings, "telegram_token", "")
    monkeypatch.setattr(message_processing, "analyze", unexpected)
    reply = asyncio.run(message_processing.process_message(FakeDB(), msg))
    assert reply.startswith("Неизвестная команда.")
    assert "/tickets" in reply and "/add" in reply
    assert msg.ticket_id is None
    assert msg.analysis["reply"] == reply
