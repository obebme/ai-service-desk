import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app import knowledge_ingest, main, message_processing, panel_auth, rag, telegram_worker
from app.db import Base, get_db
from app.models import (
    AuditLog, Department, KnowledgeChunk, KnowledgeDocument, Message, Ticket, User,
)
from app.schemas import Analysis


@pytest.fixture
def api_context(monkeypatch):
    import os

    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL API integration tests")
    schema = f"test_{uuid.uuid4().hex}"
    admin_engine = create_engine(url)
    with admin_engine.begin() as connection:
        connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    Base.metadata.create_all(engine, checkfirst=False)
    with Session(engine) as session:
        for code, name in [("network_support", "Network Support"),
                           ("system_administration", "System Administration"),
                           ("hardware_support", "Hardware Support"),
                           ("software_support", "Software Support"),
                           ("security_team", "Security Team")]:
            session.add(Department(code=code, name=name))
        session.commit()

    session = Session(engine, expire_on_commit=False)
    monkeypatch.setattr(panel_auth, "ROLE_TOKENS", {"agent": "agent-test", "admin": "admin-test"})

    monkeypatch.setattr(main.settings, "user_api_tokens", {
        identity: f"token-{identity}"
        for identity in ("tg-integration-user", "different-user", "status-owner",
                         "messages-owner", "messages-outsider")
    })

    def override_db():
        yield session

    main.app.dependency_overrides[get_db] = override_db
    def request(method, url, **kwargs):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                         base_url="http://test") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())

    yield request, session
    main.app.dependency_overrides.pop(get_db, None)
    session.close()
    engine.dispose()
    with admin_engine.begin() as connection:
        connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
    admin_engine.dispose()


def test_user_can_create_and_read_own_ticket(api_context):
    request, db = api_context
    response = request("POST", "/api/v1/tickets", headers={
        "Authorization": "Bearer token-tg-integration-user", "X-Telegram-User-ID": "tg-integration-user",
    }, json={"subject": "VPN unavailable", "description": "Cannot connect", "category": "network"})
    assert response.status_code == 201
    ticket_id = response.json()["id"]
    assert response.json()["ticket_number"].startswith("SD-")
    own = request("GET", f"/api/v1/tickets/{ticket_id}", headers={
        "Authorization": "Bearer token-tg-integration-user", "X-Telegram-User-ID": "tg-integration-user",
    })
    assert own.status_code == 200
    assert own.json()["category"] == "network"


def test_user_cannot_read_another_users_ticket(api_context):
    request, db = api_context
    owner = User(telegram_user_id="ticket-owner", display_name="owner")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-ACLTEST", requester_id=owner.id,
                    subject="private", description="private", category="other", subcategory="other")
    db.add(ticket)
    db.commit()
    response = request("GET", f"/api/v1/tickets/{ticket.id}", headers={
        "Authorization": "Bearer token-different-user", "X-Telegram-User-ID": "different-user",
    })
    assert response.status_code == 404


def test_ticket_update_reloads_status_changed_by_another_session(api_context):
    request, db = api_context
    owner = User(display_name="Status update test")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-STATUS-RACE", requester_id=owner.id,
                    subject="Status update", description="Test", status="closed",
                    closed_at=datetime.now(timezone.utc) - timedelta(days=1))
    db.add(ticket)
    db.commit()
    ticket_id = ticket.id
    previous_close = ticket.closed_at
    with Session(db.get_bind()) as other:
        reopened = other.get(Ticket, ticket_id)
        reopened.status = "open"
        reopened.closed_at = None
        other.commit()

    response = request("PATCH", f"/api/v1/tickets/{ticket_id}",
                       headers={"Authorization": "Bearer agent-test"}, json={"status": "closed"})
    assert response.status_code == 200
    with Session(db.get_bind()) as other:
        reclosed = other.get(Ticket, ticket_id)
        assert reclosed.status == "closed"
        assert reclosed.closed_at > previous_close


def test_user_cannot_change_ticket_status(api_context):
    request, db = api_context
    owner = User(telegram_user_id="status-owner", display_name="owner")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-STATTEST", requester_id=owner.id,
                    subject="change status", description="request", category="other", subcategory="other")
    db.add(ticket)
    db.commit()
    response = request("PATCH", f"/api/v1/tickets/{ticket.id}", headers={
        "Authorization": "Bearer token-status-owner", "X-Telegram-User-ID": "status-owner",
    }, json={"status": "resolved"})
    assert response.status_code == 403


def test_agent_ticket_update_writes_audit_event(api_context):
    request, db = api_context
    owner = User(telegram_user_id="audit-owner", display_name="owner")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-AUDITTEST", requester_id=owner.id,
                    subject="audit", description="request",
                    category="other", subcategory="other")
    db.add(ticket)
    db.commit()

    response = request("PATCH", f"/api/v1/tickets/{ticket.id}", headers={
        "Authorization": "Bearer agent-test",
    }, json={"status": "closed"})
    assert response.status_code == 200
    audit = db.scalar(select(AuditLog).where(
        AuditLog.entity_id == ticket.id, AuditLog.action == "ticket.updated",
    ))
    assert audit is not None
    assert audit.details == {"fields": ["status"]}


def test_ticket_messages_require_owner_and_are_sorted(api_context):
    request, db = api_context
    owner = User(telegram_user_id="messages-owner", display_name="owner")
    outsider = User(telegram_user_id="messages-outsider", display_name="outsider")
    db.add_all([owner, outsider])
    db.flush()
    ticket = Ticket(ticket_number="SD-MSGSORT", requester_id=owner.id,
                    subject="message access", description="request",
                    category="other", subcategory="other")
    db.add(ticket)
    db.flush()
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expected_ids = [uuid.UUID(int=1), uuid.UUID(int=2)]
    db.add_all([
        Message(id=expected_ids[1], requester_id=owner.id, ticket_id=ticket.id,
                body="second", created_at=created_at),
        Message(id=expected_ids[0], requester_id=owner.id, ticket_id=ticket.id,
                body="first", created_at=created_at),
    ])
    db.commit()

    owned = request("GET", f"/api/v1/tickets/{ticket.id}/messages", headers={
        "Authorization": "Bearer token-messages-owner",
        "X-Telegram-User-ID": "messages-owner",
    })
    assert owned.status_code == 200
    assert [item["id"] for item in owned.json()] == [str(item) for item in expected_ids]

    denied = request("GET", f"/api/v1/tickets/{ticket.id}/messages", headers={
        "Authorization": "Bearer token-messages-outsider",
        "X-Telegram-User-ID": "messages-outsider",
    })
    assert denied.status_code == 404

    agent = request("GET", f"/api/v1/tickets/{ticket.id}/messages", headers={
        "Authorization": "Bearer agent-test",
    })
    assert agent.status_code == 200


def test_ai_provider_failure_returns_service_unavailable(api_context, monkeypatch):
    request, _ = api_context

    async def fail_analysis(_body):
        raise httpx.ConnectError("provider unavailable")

    monkeypatch.setattr(main, "analyze", fail_analysis)
    response = request("POST", "/api/v1/ai/analyze", headers={
        "Authorization": "Bearer agent-test",
    }, json={"body": "VPN is unavailable"})
    assert response.status_code == 503
    assert response.json()["detail"] == "AI unavailable: ConnectError"
    assert "provider unavailable" not in response.text


def test_database_unavailable_makes_readiness_fail(monkeypatch):
    class BrokenDB:
        def execute(self, _statement):
            raise OperationalError("SELECT 1", {}, RuntimeError("database offline"))

    def broken_db():
        yield BrokenDB()

    monkeypatch.setitem(main.app.dependency_overrides, get_db, broken_db)

    async def send():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health/ready"), await client.get("/health/live")

    readiness, liveness = asyncio.run(send())
    assert readiness.status_code == 503
    assert readiness.json() == {"detail": "Database unavailable"}
    assert liveness.status_code == 200


def _telegram_request(request, payload, *, secret="telegram-test-secret", content=None):
    kwargs = {"headers": {"X-Telegram-Bot-Api-Secret-Token": secret}}
    if content is not None:
        kwargs["content"] = content
    else:
        kwargs["json"] = payload
    return request("POST", "/integrations/telegram/webhook", **kwargs)


@pytest.fixture
def mocked_webhook_context(monkeypatch):
    class MockDB:
        def __init__(self):
            self.messages = []
            self.added = []
            self.commits = 0
            self.scalar_calls = 0

        def scalar(self, _statement):
            self.scalar_calls += 1
            return next(iter(self.messages), None)

        def add(self, message):
            self.added.append(message)

        def commit(self):
            self.commits += 1
            self.messages.extend(
                message for message in self.added if message not in self.messages
            )

        def rollback(self):
            raise AssertionError("Unexpected rollback")

    db = MockDB()
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    monkeypatch.setattr(main, "lock_telegram_queue", lambda _db: None)
    monkeypatch.setattr(main, "telegram_queue_full", lambda _db, _limit: False)
    monkeypatch.setattr(main, "update_limit_reached", lambda *_args: False)
    monkeypatch.setattr(
        main, "get_or_create_user",
        lambda *_args: SimpleNamespace(id=uuid.uuid4()),
    )

    async def process_message(_db, _message):
        return "test reply"

    monkeypatch.setattr(message_processing, "process_message", process_message)
    previous = main.app.dependency_overrides.get(get_db)
    main.app.dependency_overrides[get_db] = lambda: db

    def request(method, url, **kwargs):
        async def send():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())

    yield request, db
    if previous is None:
        main.app.dependency_overrides.pop(get_db, None)
    else:
        main.app.dependency_overrides[get_db] = previous


@pytest.mark.parametrize(
    "chat",
    [
        None,
        {},
        {"id": 777},
        {"id": True, "type": "private"},
        {"id": 778, "type": "unknown"},
    ],
)
def test_telegram_text_update_with_missing_or_malformed_chat_is_not_persisted(
        mocked_webhook_context, chat):
    request, db = mocked_webhook_context
    message = {"message_id": 1101, "from": {"id": 1102}, "text": "Need help"}
    if chat is not None:
        message["chat"] = chat

    response = _telegram_request(request, {"message": message})

    assert response.status_code == 400
    assert db.added == []
    assert db.messages == []
    assert db.commits == 0
    assert db.scalar_calls == 0


@pytest.mark.parametrize(
    "chat_type,expected_chat_id",
    [("private", "900000000000000001"), ("group", None), ("supergroup", None)],
)
def test_telegram_text_update_persists_chat_id_only_for_private_chats(
        mocked_webhook_context, chat_type, expected_chat_id):
    request, db = mocked_webhook_context
    response = _telegram_request(request, {
        "message": {
            "message_id": 1103,
            "from": {"id": 1104},
            "chat": {"id": 900000000000000001, "type": chat_type},
            "text": "Need help",
        },
    })

    assert response.status_code == 200
    assert db.commits == 1
    assert len(db.messages) == 1
    assert db.messages[0].telegram_chat_id == expected_chat_id


def test_duplicate_old_telegram_message_is_not_backfilled_with_chat_id(
        mocked_webhook_context):
    request, db = mocked_webhook_context
    old_message = Message(
        id=uuid.uuid4(), requester_id=uuid.uuid4(), channel="telegram",
        external_message_id="1106:1105", body="Need help", telegram_chat_id=None,
        analysis={"reply": "Already processed"},
    )
    db.messages.append(old_message)

    response = _telegram_request(request, {
        "message": {
            "message_id": 1105,
            "from": {"id": 1106},
            "chat": {"id": 1106, "type": "private"},
            "text": "Need help",
        },
    })

    assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert old_message.telegram_chat_id is None
    assert db.added == []
    assert db.commits == 1


@pytest.mark.parametrize("content,payload", [
    (b"{broken json", None),
    (None, []),
    (None, {"message": {"text": "   ", "from": {"id": 71}, "message_id": 1}}),
    (None, {"message": {"text": "hello", "from": {"id": True}, "message_id": 2}}),
])
def test_telegram_malformed_updates_return_400(api_context, monkeypatch, content, payload):
    request, _ = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    response = _telegram_request(request, payload, content=content)
    assert response.status_code == 400


def test_telegram_non_text_update_is_acknowledged_without_message(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    response = _telegram_request(request, {
        "message": {"message_id": 3, "from": {"id": 73}, "photo": [{"file_id": "test"}]},
    })
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert db.scalar(select(func.count()).select_from(Message)) == 0


def test_telegram_duplicate_keeps_one_pending_message(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    payload = {"message": {"message_id": 505, "from": {"id": 705}, "chat": {"id": 705, "type": "private"}, "text": "Help"}}
    first = _telegram_request(request, payload)
    second = _telegram_request(request, payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert db.scalars(select(Message)).one().processing_due_at is not None


def test_concurrent_process_message_runs_ai_once_and_reuses_saved_result(
        api_context, monkeypatch):
    _, db = api_context
    owner = User(telegram_user_id="process-lock-user", display_name="test")
    db.add(owner)
    db.flush()
    message = Message(
        requester_id=owner.id, channel="telegram",
        external_message_id="810:910", body="VPN unavailable",
    )
    db.add(message)
    db.commit()
    message_id = message.id
    owner_id = owner.id

    ai_started = threading.Event()
    release_ai = threading.Event()
    ai_calls = []

    async def blocked_analysis(_body):
        ai_calls.append(True)
        ai_started.set()
        assert await asyncio.to_thread(release_ai.wait, 5)
        return Analysis(
            category="network", subcategory="vpn", priority="medium",
            summary="VPN unavailable", sentiment="neutral", needs_human=True,
            confidence=0.9, suggested_action="escalate",
        )

    async def no_evidence(*_args, **_kwargs):
        return []

    monkeypatch.setattr(main.settings, "telegram_token", "")
    monkeypatch.setattr(message_processing, "analyze", blocked_analysis)
    monkeypatch.setattr(message_processing, "search", no_evidence)

    sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)

    async def process_on_one_loop():
        async def process():
            with sessions() as session:
                session.execute(text("SET LOCAL lock_timeout = '3s'"))
                row = session.get(Message, message_id)
                return await message_processing.process_message(session, row)

        first = asyncio.create_task(process())
        assert await asyncio.to_thread(ai_started.wait, 5), "first AI call did not start"
        second = asyncio.create_task(process())
        try:
            await asyncio.sleep(0.1)
            assert len(ai_calls) == 1
        finally:
            release_ai.set()
        return await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

    first_reply, second_reply = asyncio.run(process_on_one_loop())

    assert len(ai_calls) == 1
    assert first_reply == second_reply
    assert db.scalar(select(func.count()).select_from(Ticket).where(
        Ticket.requester_id == owner_id,
    )) == 1


@pytest.mark.parametrize("outcome", [
    "connect", "http", "timeout_after_processing", "timeout_during_processing",
    "ack_without_result", "ack_with_result",
])
def test_n8n_dispatch_requires_saved_result_without_duplicates(api_context, monkeypatch, outcome):
    request, db = api_context
    sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(telegram_worker, "SessionLocal", sessions)
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "telegram_token", "telegram-test-token")
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "http://n8n.test/webhook/incoming")

    def request_session():
        with sessions() as session:
            yield session

    monkeypatch.setitem(main.app.dependency_overrides, get_db, request_session)
    payload = {"message": {"message_id": 1234, "from": {"id": 706},
                           "chat": {"id": 706, "type": "private"}, "text": "VPN needs manual review"}}
    queued = _telegram_request(request, payload)
    assert queued.status_code == 200 and queued.json()["queued"]
    message_id, claim_until = telegram_worker._claim_telegram_message()
    classifications = []
    deliveries = []
    processing_calls = []
    real_client = httpx.AsyncClient

    async def scenario():
        ai_started = asyncio.Event()
        release_ai = asyncio.Event()
        workflow_task = None
        process_message = message_processing.process_message
        headers = {"X-Workflow-Token": "workflow-test-secret"}
        async with real_client(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as api:
            async def workflow():
                response = await api.post("/api/v1/internal/process-message", headers=headers,
                                          json={"message_id": str(message_id)})
                response.raise_for_status()
                return response.json()["reply"]

            async def analyze(body):
                classifications.append(body)
                ai_started.set()
                if outcome == "timeout_during_processing":
                    await asyncio.wait_for(release_ai.wait(), 3)
                return Analysis(summary=body, needs_human=True, confidence=0.95, suggested_action="review")

            async def send(chat_id, reply):
                deliveries.append((chat_id, reply))

            async def process(session, message):
                processing_calls.append(message.id)
                if ai_started.is_set():
                    release_ai.set()
                return await process_message(session, message)

            async def n8n(request):
                nonlocal workflow_task
                if outcome == "connect":
                    raise httpx.ConnectError("unavailable", request=request)
                if outcome == "http":
                    return httpx.Response(502)
                if outcome == "ack_without_result":
                    return httpx.Response(200, json={"message": "Workflow started"})
                if outcome == "ack_with_result":
                    return httpx.Response(200, json={"reply": await workflow()})
                if outcome == "timeout_after_processing":
                    await workflow()
                else:
                    workflow_task = asyncio.create_task(workflow())
                    await asyncio.wait_for(ai_started.wait(), 3)
                raise httpx.ReadTimeout("no response", request=request)

            monkeypatch.setattr(message_processing, "analyze", analyze)
            monkeypatch.setattr(message_processing, "send_telegram_message", send)
            monkeypatch.setattr(message_processing, "process_message", process)
            monkeypatch.setattr(telegram_worker.httpx, "AsyncClient", lambda **kwargs: real_client(
                transport=httpx.MockTransport(n8n), **kwargs))
            try:
                await asyncio.wait_for(telegram_worker._dispatch_telegram_message(message_id), 5)
            finally:
                release_ai.set()
                if workflow_task is not None:
                    await asyncio.wait_for(workflow_task, 5)
            assert len(deliveries) == 1
            if outcome == "ack_with_result":
                assert processing_calls == [message_id]
            assert await workflow() == deliveries[0][1]

    asyncio.run(scenario())
    telegram_worker._finish_telegram_attempt(message_id, claim_until, True)
    ticket = db.scalars(select(Ticket)).one()
    message = db.scalars(select(Message)).one()
    assert len(classifications) == len(deliveries) == 1
    assert deliveries[0][0] == "706" and ticket.ticket_number in deliveries[0][1]
    assert message.ticket_id == ticket.id
    assert message.telegram_delivery_status == "accepted"
    assert message.processing_due_at is None


def test_delivery_confirmation_does_not_block_concurrent_reader(api_context, monkeypatch):
    _, db = api_context
    user = User(display_name="Delivery test")
    db.add(user)
    db.flush()
    message = Message(requester_id=user.id, body="VPN unavailable", channel="telegram",
                      telegram_chat_id="706", analysis={"reply": "Ticket created"})
    db.add(message)
    db.commit()
    message_id = message.id
    sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(main.settings, "telegram_token", "telegram-test-token")

    async def scenario():
        delivery_started = asyncio.Event()
        reader_locked = asyncio.Event()

        async def send(_chat_id, _reply):
            delivery_started.set()
            await asyncio.wait_for(reader_locked.wait(), 3)

        async def read_while_delivering():
            await asyncio.wait_for(delivery_started.wait(), 3)
            with sessions() as reader:
                locked = await asyncio.to_thread(reader.get, Message, message_id, with_for_update=True)
                assert locked.telegram_delivery_status == "unconfirmed"
                reader_locked.set()
                await asyncio.sleep(0.1)
                reader.commit()

        monkeypatch.setattr(message_processing, "send_telegram_message", send)
        with db.get_bind().connect() as connection, sessions(bind=connection) as delivery:
            delivery.execute(text("SET lock_timeout = '1500ms'"))
            row = delivery.get(Message, message_id)
            await asyncio.gather(message_processing.process_message(delivery, row), read_while_delivering())

    asyncio.run(scenario())
    db.refresh(message)
    assert message.telegram_delivery_status == "accepted"


def test_failed_background_attempt_keeps_message_for_retry(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(telegram_worker, "SessionLocal", sessionmaker(bind=db.get_bind()))
    response = _telegram_request(request, {
        "message": {"message_id": 606, "from": {"id": 706},
                    "chat": {"id": 706, "type": "private"}, "text": "Need help"},
    })
    assert response.status_code == 200
    message_id, first_claim = telegram_worker._claim_telegram_message()
    assert telegram_worker._claim_telegram_message() is None
    telegram_worker._finish_telegram_attempt(message_id, first_claim, False)
    msg = db.get(Message, message_id)
    assert datetime.now(timezone.utc) < msg.processing_due_at < first_claim
    telegram_worker._finish_telegram_attempt(message_id, first_claim, True)
    db.refresh(msg)
    assert msg.processing_due_at is not None


def test_n8n_ack_without_result_is_requeued_when_direct_processing_fails(api_context, monkeypatch):
    request, db = api_context
    sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(telegram_worker, "SessionLocal", sessions)
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "http://n8n.test/webhook/incoming")
    response = _telegram_request(request, {
        "message": {"message_id": 607, "from": {"id": 707},
                    "chat": {"id": 707, "type": "private"}, "text": "Need help"},
    })
    assert response.status_code == 200 and response.json()["queued"]
    real_client = httpx.AsyncClient
    monkeypatch.setattr(telegram_worker.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(204)), **kwargs))

    async def processing_failed(_db, _message):
        raise RuntimeError("Direct processing failed")

    async def scenario():
        finished = asyncio.Event()
        loop = asyncio.get_running_loop()
        finish = telegram_worker._finish_telegram_attempt

        def mark_finished(*args):
            finish(*args)
            loop.call_soon_threadsafe(finished.set)

        monkeypatch.setattr(message_processing, "process_message", processing_failed)
        monkeypatch.setattr(telegram_worker, "_finish_telegram_attempt", mark_finished)
        worker = asyncio.create_task(telegram_worker.run_telegram_worker())
        try:
            await asyncio.wait_for(finished.wait(), 5)
        finally:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker

    asyncio.run(scenario())
    msg = db.scalars(select(Message)).one()
    assert msg.processing_due_at is not None
    assert msg.processing_due_at > datetime.now(timezone.utc)
    assert "reply" not in msg.analysis and msg.ticket_id is None


def _seed_retention_rows(db, now):
    cutoff = now - timedelta(days=365)
    user = User(telegram_user_id="retention-user", display_name="test")
    db.add(user)
    db.flush()

    expired = Ticket(
        ticket_number="SD-RETAIN-OLD", requester_id=user.id, subject="old closed",
        description="old", category="other", subcategory="other", status="closed",
        closed_at=cutoff - timedelta(seconds=1),
    )
    boundary = Ticket(
        ticket_number="SD-RETAIN-BOUND", requester_id=user.id, subject="boundary closed",
        description="boundary", category="other", subcategory="other", status="closed",
        closed_at=cutoff,
    )
    opened = Ticket(
        ticket_number="SD-RETAIN-OPEN", requester_id=user.id, subject="open old",
        description="open", category="other", subcategory="other", status="open",
        created_at=cutoff - timedelta(days=500),
    )
    db.add_all([expired, boundary, opened])
    db.flush()
    old_attached = Message(
        requester_id=user.id, ticket_id=expired.id, body="expired history",
        created_at=now,
    )
    boundary_attached = Message(
        requester_id=user.id, ticket_id=boundary.id, body="boundary history",
        created_at=cutoff - timedelta(days=2),
    )
    open_attached = Message(
        requester_id=user.id, ticket_id=opened.id, body="open history",
        created_at=cutoff - timedelta(days=2),
    )
    unattached = Message(
        requester_id=user.id, ticket_id=None, body="old unattached",
        created_at=cutoff - timedelta(seconds=1),
    )
    old_document = KnowledgeDocument(title="old", source_name="old.txt", created_at=cutoff - timedelta(days=1))
    current_document = KnowledgeDocument(title="current", source_name="current.txt", created_at=now)
    db.add_all([old_attached, boundary_attached, open_attached, unattached,
                old_document, current_document])
    db.flush()
    db.add(KnowledgeChunk(
        document_id=old_document.id, chunk_index=0, content="old chunk",
        embedding=[0.0] * main.settings.embedding_dimensions,
    ))
    db.add(AuditLog(action="old", entity_type="test", entity_id=expired.id,
                    created_at=cutoff - timedelta(days=1)))
    db.commit()
    return {
        "expired_ticket": expired.id,
        "boundary_ticket": boundary.id,
        "open_ticket": opened.id,
        "old_document": old_document.id,
        "current_document": current_document.id,
    }


def _use_test_schema_for_retention(monkeypatch, db):
    from scripts import cleanup_retention

    monkeypatch.setattr(cleanup_retention, "SessionLocal", sessionmaker(bind=db.get_bind()))
    return cleanup_retention


def test_retention_dry_run_counts_cutoff_without_deleting(api_context, monkeypatch):
    _, db = api_context
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    ids = _seed_retention_rows(db, now)
    cleanup_retention = _use_test_schema_for_retention(monkeypatch, db)

    counts = cleanup_retention.cleanup(dry_run=True, now=now, batch_size=1)
    assert counts == {"tickets": 1, "messages": 2, "documents": 1, "audit_logs": 1}
    assert db.get(Ticket, ids["expired_ticket"]) is not None
    assert db.get(KnowledgeDocument, ids["old_document"]) is not None


def test_retention_execute_removes_expired_data_but_preserves_open_and_boundary(
        api_context, monkeypatch):
    _, db = api_context
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    ids = _seed_retention_rows(db, now)
    cleanup_retention = _use_test_schema_for_retention(monkeypatch, db)

    counts = cleanup_retention.cleanup(dry_run=False, now=now, batch_size=1)
    assert counts == {"tickets": 1, "messages": 2, "documents": 1, "audit_logs": 1}
    db.expire_all()
    assert db.get(Ticket, ids["expired_ticket"]) is None
    assert db.get(Ticket, ids["boundary_ticket"]) is not None
    assert db.get(Ticket, ids["open_ticket"]) is not None
    assert db.get(KnowledgeDocument, ids["old_document"]) is None
    assert db.get(KnowledgeDocument, ids["current_document"]) is not None
    assert db.scalar(select(func.count()).select_from(Message)) == 2


def test_telegram_message_ids_are_scoped_to_chat(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    for chat_id in (-1001, -1002, -1001):
        response = _telegram_request(request, {"message": {
            "message_id": 1, "from": {"id": 999}, "text": "Help",
            "chat": {"id": chat_id, "type": "group"},
        }})
        assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert set(db.scalars(select(Message.external_message_id))) == {
        "chat:-1001:1", "chat:-1002:1",
    }


def test_concurrent_first_messages_share_one_user(api_context, monkeypatch):
    _, db = api_context
    sessions = sessionmaker(bind=db.get_bind())
    barrier = threading.Barrier(2)
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "n8n_webhook_url", "")
    monkeypatch.setattr(main.settings, "telegram_token", "")

    def override_db():
        with sessions() as session:
            yield session

    original = main.lock_telegram_queue

    def concurrent_admission(*args):
        barrier.wait(timeout=5)
        return original(*args)

    async def process(_session, _message):
        return "Saved"

    monkeypatch.setitem(main.app.dependency_overrides, get_db, override_db)
    monkeypatch.setattr(main, "lock_telegram_queue", concurrent_admission)
    monkeypatch.setattr(message_processing, "process_message", process)

    async def test_requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://test") as client:
            return await asyncio.gather(*[
                client.post("/integrations/telegram/webhook",
                            headers={"X-Telegram-Bot-Api-Secret-Token": "telegram-test-secret"},
                            json={"message": {"message_id": number, "from": {"id": 4321},
                                              "chat": {"id": 4321, "type": "private"},
                                              "text": "Need help"}})
                for number in (1, 2)
            ])

    assert [r.status_code for r in asyncio.run(test_requests())] == [200, 200]
    users = db.scalars(select(User)).all()
    assert len(users) == 1
    assert db.scalars(select(Message.requester_id)).all() == [users[0].id, users[0].id]

def test_webhook_acknowledges_before_slow_n8n_and_deduplicates_update(api_context, monkeypatch):
    _, db = api_context
    sessions = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(telegram_worker, "SessionLocal", sessions)
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test-secret")
    monkeypatch.setattr(main.settings, "telegram_token", "telegram-test-token")
    payload = {"message": {"message_id": 100, "from": {"id": 999},
                           "chat": {"id": 999, "type": "private"}, "text": "Slow request"}}
    calls = []
    deliveries = []

    def request_session():
        with sessions() as session:
            yield session

    async def classify(_body):
        return Analysis(category="network", summary="Slow request", needs_human=True,
                        confidence=0.95, suggested_action="review")

    async def send(chat_id, reply):
        deliveries.append((chat_id, reply))

    monkeypatch.setitem(main.app.dependency_overrides, get_db, request_session)
    monkeypatch.setattr(message_processing, "analyze", classify)
    monkeypatch.setattr(message_processing, "send_telegram_message", send)

    async def scenario():
        started = asyncio.Event()
        headers = {"X-Telegram-Bot-Api-Secret-Token": "telegram-test-secret"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://test") as client:
            async def slow_n8n(message_id):
                calls.append(message_id)
                started.set()
                await asyncio.sleep(11)
                r = await client.post("/api/v1/internal/process-message",
                                      headers={"X-Workflow-Token": "workflow-test-secret"},
                                      json={"message_id": str(message_id)})
                r.raise_for_status()

            monkeypatch.setattr(telegram_worker, "_dispatch_telegram_message", slow_n8n)
            first = await asyncio.wait_for(client.post("/integrations/telegram/webhook",
                                                       json=payload, headers=headers), 2)
            assert first.status_code == 200 and first.json()["queued"]
            worker = asyncio.create_task(telegram_worker.run_telegram_worker())
            try:
                await asyncio.wait_for(started.wait(), 2)
                second = await asyncio.wait_for(client.post("/integrations/telegram/webhook",
                                                            json=payload, headers=headers), 2)
                assert second.status_code == 200 and second.json()["duplicate"]
                for _ in range(135):
                    with sessions() as session:
                        if session.scalars(select(Message)).one().processing_due_at is None:
                            break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("Background processing did not complete")
                third = await client.post("/integrations/telegram/webhook", json=payload, headers=headers)
                assert third.status_code == 200 and third.json()["duplicate"]
            finally:
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker

    asyncio.run(scenario())
    assert len(calls) == len(deliveries) == 1
    msg = db.scalars(select(Message)).one()
    ticket = db.scalars(select(Ticket)).one()
    assert msg.ticket_id == ticket.id
    assert msg.telegram_delivery_status == "accepted"
    assert ticket.ticket_number in deliveries[0][1]

def test_telegram_accepts_reply_but_confirmation_is_lost(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test")
    monkeypatch.setattr(main.settings, "telegram_token", "telegram-test")
    owner = User(telegram_user_id="123")
    db.add(owner)
    db.flush()
    message = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="123",
                      body="Please create a ticket")
    db.add(message)
    db.commit()
    attempts = []

    async def classify(_body):
        return Analysis(summary="Support request", confidence=0.99, needs_human=True,
                        suggested_action="review")

    async def accepted_then_timeout(chat_id, reply):
        with Session(db.get_bind()) as observer:
            assert observer.get(Message, message.id).telegram_delivery_status == "unconfirmed"
        attempts.append((chat_id, reply))
        raise main.HTTPException(503, "Telegram message delivery failed")

    monkeypatch.setattr(message_processing, "analyze", classify)
    monkeypatch.setattr(message_processing, "send_telegram_message", accepted_then_timeout)
    kwargs = {"json": {"message_id": str(message.id)},
              "headers": {"X-Workflow-Token": "workflow-test"}}
    first = request("POST", "/api/v1/internal/process-message", **kwargs)
    second = request("POST", "/api/v1/internal/process-message", **kwargs)

    assert len(attempts) == 1
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    db.refresh(message)
    assert message.telegram_delivery_status == "unconfirmed"
    ticket = db.scalars(select(Ticket)).one()
    assert message.ticket_id == ticket.id
    assert ticket.ticket_number in attempts[0][1]

def test_panel_shows_telegram_queue_and_unconfirmed_reply(api_context, monkeypatch):
    request, db = api_context
    from app.panel_auth import SESSION_COOKIE, create_session

    owner = User(display_name="Queue test")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-QUEUE", requester_id=owner.id,
                    subject="Queue test", description="Queue test")
    db.add(ticket)
    db.flush()
    secret = "local-integration-session-secret-for-tests"
    monkeypatch.setattr(main.settings, "session_signing_secret", secret)
    _, cookie = create_session("agent", secret)
    headers = {"Cookie": f"{SESSION_COOKIE}={cookie}"}
    empty = request("GET", "/ui/tickets", headers=headers)
    assert empty.status_code == 200
    assert "Awaiting completion</dt><dd>0" in empty.text
    assert "Oldest pending request</dt><dd>—" in empty.text

    now = datetime.now(timezone.utc)
    for age in (timedelta(hours=2, minutes=5), timedelta(minutes=5)):
        db.add(Message(channel="telegram", requester_id=owner.id, body="Pending",
                       created_at=now-age, processing_due_at=now))
    db.add(Message(channel="telegram", requester_id=owner.id, body="Auto reply attempted",
                   ticket_id=ticket.id, telegram_delivery_status="unconfirmed"))
    db.add(Message(channel="api", sender_type="agent", requester_id=owner.id,
                   body="Agent reply", ticket_id=ticket.id, telegram_delivery_status="unconfirmed"))
    db.commit()

    queue = request("GET", "/ui/tickets?q=Queue", headers=headers)
    assert queue.status_code == 200
    assert "Awaiting completion</dt><dd>2" in queue.text
    assert "Oldest pending request</dt><dd>2h 5m" in queue.text
    assert "Unconfirmed automatic replies</dt><dd>1" in queue.text
    assert queue.text.count("Automatic reply unconfirmed") == 1
    detail = request("GET", f"/ui/tickets/{ticket.id}", headers=headers)
    assert detail.status_code == 200
    assert detail.text.count("Automatic reply unconfirmed") == 1


@pytest.mark.parametrize("operation", ["panel_queue", "rag_search", "telegram_ticket"])
def test_postgres_lock_wait_does_not_block_another_request(api_context, monkeypatch, operation):
    _, db = api_context
    engine = db.get_bind()
    sessions = sessionmaker(bind=engine, expire_on_commit=False)

    def request_session():
        with sessions() as session:
            yield session

    monkeypatch.setitem(main.app.dependency_overrides, get_db, request_session)
    if operation == "panel_queue":
        secret = "local-integration-session-secret-for-tests"
        monkeypatch.setattr(main.settings, "session_signing_secret", secret)
        _, cookie = panel_auth.create_session("agent", secret)
        method, url = "GET", "/ui/tickets"
        arguments = {"headers": {"Cookie": f"{panel_auth.SESSION_COOKIE}={cookie}"}}
        lock = "LOCK TABLE tickets IN ACCESS EXCLUSIVE MODE"
    else:
        owner = User(display_name="RAG concurrency test")
        db.add(owner)
        db.flush()
        message = Message(requester_id=owner.id, channel="api", body="VPN help")
        db.add(message)
        db.commit()

        async def classify(_):
            return Analysis(summary="VPN help", confidence=0.95, needs_human=operation == "telegram_ticket",
                            suggested_action="Search the knowledge base")

        async def embed(_, *, task_type):
            assert task_type == "RETRIEVAL_QUERY"
            return [0.0] * main.settings.embedding_dimensions

        monkeypatch.setattr(message_processing, "analyze", classify)
        monkeypatch.setattr(rag, "embed", embed)
        monkeypatch.setattr(main.settings, "workflow_token", "lock-test-token")
        method, url = "POST", "/api/v1/internal/process-message"
        arguments = {
            "headers": {"X-Workflow-Token": "lock-test-token"},
            "json": {"message_id": str(message.id)},
        }
        lock = ("LOCK TABLE departments IN ACCESS EXCLUSIVE MODE" if operation == "telegram_ticket"
                else "LOCK TABLE knowledge_chunks IN ACCESS EXCLUSIVE MODE")

    lock_held = threading.Event()
    query_waiting = threading.Event()
    release_lock = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        with engine.begin() as blocker:
            blocker.exec_driver_sql(lock)
            blocker_pid = blocker.exec_driver_sql("SELECT pg_backend_pid()").scalar_one()
            lock_held.set()
            with engine.connect() as observer:
                observer = observer.execution_options(isolation_level="AUTOCOMMIT")
                waiting = text("SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                               "WHERE :pid = ANY(pg_blocking_pids(pid)))")
                for _ in range(200):
                    if release_lock.wait(0.02):
                        break
                    if observer.execute(waiting, {"pid": blocker_pid}).scalar_one():
                        query_waiting.set()
                        # Release even if a synchronous query stalls the event loop.
                        release_lock.wait(4)
                        break
            lock_released.set()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://test") as client:
            blocker = asyncio.create_task(asyncio.to_thread(hold_lock))
            first = None
            try:
                assert await asyncio.to_thread(lock_held.wait, 6)
                first = asyncio.create_task(client.request(method, url, **arguments))
                assert await asyncio.to_thread(query_waiting.wait, 6)
                second = await asyncio.wait_for(client.get("/health/ready"), 3)
                assert second.status_code == 200
                assert not lock_released.is_set(), "The second request waited for the PostgreSQL lock"
                assert not first.done()
            finally:
                release_lock.set()
                await blocker
                if first is not None:
                    result = await asyncio.wait_for(first, 6)
                    assert result.status_code == 200

    asyncio.run(exercise())

@pytest.mark.parametrize("fail_embedding", [False, True])
def test_knowledge_upload_commits_document_and_chunks_together(api_context, monkeypatch, fail_embedding):
    request, db = api_context
    embedding_calls = 0

    async def embed(_, *, task_type):
        nonlocal embedding_calls
        assert task_type == "RETRIEVAL_DOCUMENT"
        embedding_calls += 1
        if embedding_calls == 2:
            with Session(db.get_bind()) as reader:
                assert reader.scalar(select(func.count()).select_from(KnowledgeDocument)) == 0
                assert reader.scalar(select(func.count()).select_from(KnowledgeChunk)) == 0
            if fail_embedding:
                raise RuntimeError("Simulated embedding failure")
        return [0.0] * main.settings.embedding_dimensions

    monkeypatch.setattr(knowledge_ingest, "embed", embed)
    response = request(
        "POST", "/api/v1/knowledge/documents",
        headers={"Authorization": "Bearer admin-test"},
        files={"file": ("guide.txt", "VPN reset. " * 130, "text/plain")},
    )
    assert embedding_calls == 2
    with Session(db.get_bind()) as reader:
        documents = reader.scalars(select(KnowledgeDocument)).all()
        chunks = reader.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.chunk_index)).all()
        if fail_embedding:
            assert response.status_code == 503
            assert not documents and not chunks
        else:
            assert response.status_code == 201
            assert len(documents) == 1 and documents[0].status == "indexed"
            assert [(chunk.document_id, chunk.chunk_index) for chunk in chunks] == [
                (documents[0].id, 0), (documents[0].id, 1),
            ]


def test_panel_searches_old_tickets_and_preserves_filters(api_context, monkeypatch):
    from app.panel_auth import SESSION_COOKIE, create_session

    request, db = api_context
    secret = "local-integration-session-secret-for-tests"
    monkeypatch.setattr(main.settings, "session_signing_secret", secret)
    _, cookie = create_session("agent", secret)
    headers = {"Cookie": f"{SESSION_COOKIE}={cookie}"}
    owner = User(display_name="Pagination test")
    db.add(owner)
    db.flush()
    now = datetime.now(timezone.utc)
    db.add_all(Ticket(ticket_number=f"QA-{i:04}", requester_id=owner.id,
                      subject="Pagination & QA", description="Synthetic data",
                      created_at=now-timedelta(seconds=i), status="open" if i % 2 else "closed")
               for i in range(120))
    db.commit()

    found = request("GET", "/ui/tickets", headers=headers, params={"q": "qa-0115"})
    assert found.status_code == 200
    assert "QA-0115" in found.text and "QA-0000" not in found.text

    page = request("GET", "/ui/tickets", headers=headers,
                   params={"q": "Pagination & QA", "status": "open", "page": 2})
    assert page.status_code == 200
    assert "Page 2 of 3" in page.text
    assert page.text.count('class="ticket-card"') == 25
    assert "QA-0051" in page.text and "QA-0049" not in page.text
    for destination in (1, 3):
        assert f'q=Pagination+%26+QA&amp;status=open&amp;page={destination}' in page.text

def test_telegram_followup_and_personal_assignment(api_context, monkeypatch):
    from app.panel_auth import SESSION_COOKIE, create_session

    request, db = api_context
    secret = "local-integration-session-secret-for-tests"
    monkeypatch.setattr(main.settings, "session_signing_secret", secret)
    monkeypatch.setattr(main.settings, "panel_agent_tokens", {"Alex": "personal-test", "Robin": "other-test"})
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    owner = User(telegram_user_id="4321")
    outsider = User(telegram_user_id="1234")
    db.add_all([owner, outsider])
    db.flush()
    ticket = Ticket(ticket_number="SD-FOLLOWUP", requester_id=owner.id,
                    subject="VPN issue", description="Issue", status="waiting_for_user")
    db.add(ticket)
    db.flush()

    def run_command(body, sender=owner):
        message = Message(requester_id=sender.id, channel="telegram",
                          telegram_chat_id=sender.telegram_user_id, body=body)
        db.add(message)
        db.commit()
        result = request("POST", "/api/v1/internal/process-message",
                         headers={"X-Workflow-Token": "workflow-test"},
                         json={"message_id": str(message.id)})
        assert result.status_code == 200
        return message, result.json()["reply"]

    assert ticket.ticket_number in run_command("/tickets")[1]
    assert ticket.ticket_number not in run_command("/tickets", outsider)[1]
    assert run_command("/add SD-FOLLOWUP wrong user", outsider)[0].ticket_id is None
    message, _ = run_command("/add SD-FOLLOWUP The error is still present")
    db.refresh(ticket)
    assert ticket.status == "in_progress"
    assert message.ticket_id == ticket.id and message.body == "The error is still present"
    csrf, cookie = create_session("agent", secret, agent="Alex")
    headers = {"Cookie": f"{SESSION_COOKIE}={cookie}"}
    page = request("GET", "/ui/tickets?attention=reply", headers=headers)
    assert page.status_code == 200 and ticket.ticket_number in page.text
    path = f"/ui/tickets/{ticket.id}/claim"
    assert request("POST", path, data={"csrf_token": csrf}, headers=headers).status_code == 303
    other_csrf, other = create_session("agent", secret, agent="Robin")
    assert request("POST", path, data={"csrf_token": other_csrf},
                   headers={"Cookie": f"{SESSION_COOKIE}={other}"}).status_code == 409
    page = request("GET", "/ui/tickets?attention=mine", headers=headers)
    assert page.status_code == 200 and ticket.ticket_number in page.text


def test_telegram_feedback_reuses_ticket_and_exposes_knowledge_answer(api_context, monkeypatch):
    from app.panel_auth import SESSION_COOKIE, create_session
    request, db = api_context
    secret = "local-integration-session-secret-for-tests"
    monkeypatch.setattr(main.settings, "session_signing_secret", secret)
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    owner = User(telegram_user_id="4321")
    db.add(owner)
    db.flush()
    original = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                       body="VPN issue", analysis={
                           "reply": "Restart the VPN client.",
                           "knowledge_answer": "Restart the VPN client.",
                           "sources": ["01-VPN.md"],
                           "classification": Analysis(summary="VPN issue", confidence=0.9,
                                                       suggested_action="Restart").model_dump(),
                       })
    db.add(original)
    db.flush()
    for action in ("helped", "specialist", "specialist"):
        command = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                          body=f"/{action} {original.id.hex}")
        db.add(command)
        db.commit()
        reply = request("POST", "/api/v1/internal/process-message",
                        headers={"X-Workflow-Token": "workflow-test"},
                        json={"message_id": str(command.id)})
        assert reply.status_code == 200
    ticket = db.scalars(select(Ticket)).one()
    assert original.ticket_id == ticket.id
    assert original.analysis["feedback"] == "helped"
    _, cookie = create_session("agent", secret)
    detail = request("GET", f"/ui/tickets/{ticket.id}", headers={"Cookie": f"{SESSION_COOKIE}={cookie}"})
    assert detail.status_code == 200
    assert all(value in detail.text for value in ("Previous Gemini answer", "01-VPN.md", "User requested a specialist"))

def test_telegram_callback_and_screenshot_are_queued_once(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test")
    monkeypatch.setattr(main.settings, "telegram_token", "")
    chat = {"id": 4321, "type": "private"}
    sender = {"id": 4321}
    assert _telegram_request(request, {"message": {"message_id": 1, "from": sender,
                                                    "chat": chat, "text": "VPN"}}).status_code == 200
    original = db.scalars(select(Message)).one()
    original.analysis = {"knowledge_answer": "Restart the client", "sources": ["01-VPN.md"],
                         "reply": "Restart the client",
                         "classification": Analysis(summary="VPN issue", confidence=0.9,
                                                     suggested_action="Restart").model_dump()}
    db.commit()
    callback = {"callback_query": {"id": "click-one", "from": sender,
                                   "message": {"message_id": 2, "chat": chat},
                                   "data": f"specialist:{original.id.hex}"}}
    response = _telegram_request(request, callback)
    assert response.status_code == 200 and response.json()["queued"]
    assert _telegram_request(request, callback).json()["duplicate"]
    headers = {"X-Workflow-Token": "workflow-test"}
    route = "/api/v1/internal/process-message"
    assert request("POST", route, headers=headers,
                   json={"message_id": response.json()["message_id"]}).status_code == 200
    ticket = db.scalars(select(Ticket)).one()
    escalation = db.get(Message, uuid.UUID(response.json()["message_id"]))
    assert escalation.analysis["reply_markup"]["inline_keyboard"] == [[
        {"text": "Открыть обращение", "callback_data": f"ticket:{ticket.ticket_number}"},
    ]]
    photo = _telegram_request(request, {"message": {"message_id": 3, "from": sender,
                              "chat": chat, "caption": f"/add {ticket.ticket_number}",
                              "photo": [{"file_id": "telegram-photo-id"}]}})
    assert photo.status_code == 200 and photo.json()["queued"]
    assert request("POST", route, headers=headers,
                   json={"message_id": photo.json()["message_id"]}).status_code == 200
    attachment = db.get(Message, uuid.UUID(photo.json()["message_id"]))
    assert attachment.ticket_id == ticket.id
    assert attachment.analysis["photo_id"] == "telegram-photo-id"
    assert attachment.body == "Screenshot attached"


def test_telegram_ticket_menu_and_replies(api_context, monkeypatch):
    from app import message_processing

    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "workflow_token", "workflow-test")
    monkeypatch.setattr(main.settings, "telegram_token", "7654:fake")
    deliveries = []

    async def send(chat_id, message, **kwargs):
        deliveries.append(kwargs.get("reply_markup"))

    monkeypatch.setattr(message_processing, "send_telegram_message", send)

    class CallbackClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            pass

    monkeypatch.setattr(message_processing, "httpx", type("HTTP", (), {"AsyncClient": CallbackClient}))
    owner = User(telegram_user_id="4321")
    db.add(owner)
    db.flush()
    ticket = Ticket(ticket_number="SD-MENU", requester_id=owner.id, subject="VPN",
                    description="VPN issue", status="waiting_for_user")
    db.add(ticket)
    db.commit()

    def submit(update):
        response = _telegram_request(request, update)
        assert response.status_code == 200 and response.json()["queued"]
        message_id = response.json()["message_id"]
        processed = request("POST", "/api/v1/internal/process-message",
                            headers={"X-Workflow-Token": "workflow-test"},
                            json={"message_id": message_id})
        assert processed.status_code == 200
        return db.get(Message, uuid.UUID(message_id)), processed.json()["reply"]

    def message(message_id, user=4321, **fields):
        return {"message": {"message_id": message_id, "chat": {"id": user, "type": "private"},
                            "from": {"id": user}, **fields}}

    def click(click_id, data, user=4321):
        chat = {"id": user, "type": "private"}
        return submit({"callback_query": {"id": click_id, "from": {"id": user},
                                          "message": {"message_id": 1, "chat": chat}, "data": data}})

    def buttons():
        return {b["text"]: b["callback_data"] for row in deliveries[-1]["inline_keyboard"] for b in row}

    submit(message(1, text="/start"))
    assert deliveries[-1] == {
        "keyboard": [[{"text": "Задать вопрос"}, {"text": "Мои обращения"}]],
        "resize_keyboard": True,
        "is_persistent": True,
    }
    _, question = submit(message(10, text="Задать вопрос"))
    assert "Опишите проблему" in question
    _, listing = submit(message(11, text="Мои обращения"))
    assert "ждём вашего ответа" in listing
    card = next(data for label, data in buttons().items() if ticket.ticket_number in label)
    _, forbidden = click("foreign", card, user=1234)
    assert ticket.ticket_number not in forbidden and "VPN" not in forbidden
    _, detail = click("card", card)
    assert ticket.ticket_number in detail and "VPN" in detail
    actions = buttons()
    for index, label, fields in [
        (2, "Добавить сообщение", {"text": "VPN всё ещё недоступен"}),
        (3, "Прикрепить скриншот", {"photo": [{"file_id": "test-photo"}]}),
    ]:
        _, prompt = click(f"prompt-{index}", actions[label])
        assert deliveries[-1]["force_reply"] is True
        update = message(index, reply_to_message={"from": {"id": 7654, "is_bot": True}, "text": prompt},
                         **fields)
        attachment, _ = submit(update)
        assert attachment.ticket_id == ticket.id
        assert _telegram_request(request, update).json()["duplicate"]
        if "photo" in fields:
            assert attachment.analysis["photo_id"] == "test-photo"
        else:
            assert attachment.body == fields["text"]

    db.refresh(ticket)
    assert ticket.status == "in_progress"
    menu, listing = submit(message(12, text="Мои обращения", reply_to_message={
        "from": {"id": 7654, "is_bot": True}, "text": prompt,
    }))
    assert menu.ticket_id is None and ticket.ticket_number in listing
    ticket.status = "closed"
    db.commit()
    click("closed", card)
    assert "Добавить сообщение" not in buttons()
    stale, reply = submit(message(4, text="Ещё один ответ", reply_to_message={
        "from": {"id": 7654, "is_bot": True}, "text": prompt,
    }))
    assert stale.ticket_id is None and "закрыто" in reply


def test_telegram_question_rate_and_global_queue_cap(api_context, monkeypatch):
    request, db = api_context
    monkeypatch.setattr(main.settings, "telegram_webhook_secret", "telegram-test-secret")
    monkeypatch.setattr(main.settings, "telegram_queue_limit", 13)
    monkeypatch.setattr(main.settings, "telegram_token", "")
    engine = db.get_bind()

    def independent_session():
        with Session(engine, expire_on_commit=False) as session:
            yield session

    monkeypatch.setitem(main.app.dependency_overrides, get_db, independent_session)

    def update(index, sender=4321, body=None):
        return {"message": {"message_id": index, "chat": {"id": sender, "type": "private"},
                            "from": {"id": sender}, "text": body or f"Вопрос {index}"}}

    def send(payload):
        response = _telegram_request(request, payload)
        assert response.status_code == 200
        return response.json()

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(send, [update(i) for i in range(1, 14)]))
    assert sum(item.get("queued", False) for item in outcomes) == 10
    assert sum(item.get("limited") == "questions" for item in outcomes) == 3
    original = next(i for i, result in enumerate(outcomes, start=1) if result.get("queued"))
    assert send(update(original))["duplicate"]
    assert send(update(14, body="/tickets"))["queued"]
    assert db.scalar(select(func.count()).select_from(Message)) == 12  # includes one rate notice
    notice = db.scalar(select(Message).where(Message.body == "/rate-limit"))
    assert "reply" not in notice.analysis
    reply = asyncio.run(message_processing.process_message(db, notice))
    assert "10 вопросов" in reply

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(send, [update(1, sender=2345), update(1, sender=3456)]))
    assert sum(item.get("queued", False) for item in outcomes) == 1
    assert sum(item.get("limited") == "queue" for item in outcomes) == 1
    queued = select(Message).where(Message.processing_due_at.is_not(None))
    assert len(db.scalars(queued).all()) == 13
    for saved in db.scalars(queued):
        saved.created_at -= timedelta(minutes=2)
        saved.processing_due_at = None
    db.commit()
    assert send(update(15))["queued"]


def test_telegram_ticket_limit_applies_to_both_escalation_paths(api_context, monkeypatch):
    _, db = api_context
    monkeypatch.setattr(main.settings, "telegram_token", "")

    async def classify(_body):
        return Analysis(summary="Test issue", priority="critical", confidence=0.99,
                        suggested_action="escalate")

    monkeypatch.setattr(message_processing, "analyze", classify)
    owner = User(telegram_user_id="4321")
    db.add(owner)
    db.flush()
    questions = [Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                         body=f"Test issue {index}") for index in range(5)]
    db.add_all(questions)
    db.commit()
    engine = db.get_bind()

    def process(message_id):
        with Session(engine) as session:
            msg = session.get(Message, message_id)
            return asyncio.run(message_processing.process_message(session, msg))

    with ThreadPoolExecutor(max_workers=5) as pool:
        outcomes = list(pool.map(process, [msg.id for msg in questions]))
    assert sum(reply.startswith("Создал обращение") for reply in outcomes) == 3
    assert sum("лимит" in reply.lower() for reply in outcomes) == 2
    tickets = db.scalars(select(Ticket)).all()
    assert len(tickets) == 3

    repeated = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                       body=tickets[0].description.upper())
    original = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                       body="Another issue", analysis={"knowledge_answer": "Try again"})
    db.add_all([repeated, original])
    db.flush()
    original.analysis = {**original.analysis, "classification": asyncio.run(classify("")).model_dump()}
    clicked = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                      body=f"/specialist {original.id.hex}")
    db.add(clicked)
    db.commit()
    assert tickets[0].ticket_number in process(repeated.id)
    assert "лимит" in process(clicked.id).lower()
    assert db.scalar(select(func.count()).select_from(Ticket)) == 3
    for ticket in tickets:
        ticket.created_at -= timedelta(minutes=11)
    db.commit()
    retry = Message(requester_id=owner.id, channel="telegram", telegram_chat_id="4321",
                    body=f"/specialist {original.id.hex}")
    db.add(retry)
    db.commit()
    assert process(retry.id).startswith("Создал обращение")
