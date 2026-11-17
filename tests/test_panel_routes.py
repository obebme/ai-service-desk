import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.responses import HTMLResponse
from sqlalchemy.exc import SQLAlchemyError

from app import main
from app.config import Settings
from app.db import get_db as database_dependency
from app.models import AuditLog, KnowledgeDocument, Message, Ticket
from app.panel_auth import SESSION_COOKIE, create_session

TEST_SECRET = "panel-route-test-secret"


class _Scalars:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class _DB:
    def __init__(self, tickets=(), messages=(), documents=()):
        self.tickets = list(tickets)
        self.messages = list(messages)
        self.documents = list(documents)
        self.added = []
        self.commits = 0
        self.limit_seen = None

    def execute(self, _statement):
        pending = [message.created_at for message in self.messages
                   if message.processing_due_at is not None]
        return type("Result", (), {"one": lambda self: (len(pending), min(pending, default=None))})()

    def scalar(self, statement):
        if "FROM tickets" in str(statement):
            return len(self.tickets)
        return sum(message.channel == "telegram" and message.sender_type == "user"
                   and message.telegram_delivery_status == "unconfirmed" for message in self.messages)

    def scalars(self, statement):
        sql = str(statement)
        if "FROM tickets" in sql:
            self.limit_seen = statement._limit_clause.value
            rows = sorted(self.tickets, key=lambda row: row.created_at, reverse=True)
            for predicate in statement._where_criteria:
                # The queue's optional filter is a status equality expression.
                value = getattr(predicate.right, "value", None)
                if value is not None:
                    rows = [row for row in rows if row.status == value]
            return _Scalars(rows[:self.limit_seen])
        if "SELECT DISTINCT messages.ticket_id" in sql:
            return _Scalars({message.ticket_id for message in self.messages
                             if message.channel == "telegram" and message.sender_type == "user"
                             and message.telegram_delivery_status == "unconfirmed"})
        if "FROM messages" in sql:
            return _Scalars(sorted(self.messages, key=lambda row: (row.created_at, str(row.id))))
        if "FROM knowledge_documents" in sql:
            return _Scalars(self.documents)
        return _Scalars([])

    def get(self, model, key, **_options):
        if model is Ticket:
            return next((row for row in self.tickets if row.id == key), None)
        return None

    def add(self, value):
        self.added.append(value)
        if isinstance(value, Message) and value.id is None:
            value.id = uuid.uuid4()
        if isinstance(value, Message) and value not in self.messages:
            self.messages.append(value)

    def commit(self):
        self.commits += 1

    def refresh(self, _value):
        return None


def _ticket(index, *, status="open", created=None):
    return Ticket(
        id=uuid.uuid4(), ticket_number=f"SD-{index:010d}", requester_id=uuid.uuid4(),
        category="other", subcategory="other", priority="medium", status=status,
        subject=f"Subject {index}", description=f"Description {index}",
        resolution=None, created_at=created or datetime.now(timezone.utc) - timedelta(seconds=index),
        updated_at=datetime.now(timezone.utc), closed_at=None,
    )


def _telegram_message(ticket, *, chat_id, created=None, channel="telegram", sender="user"):
    return Message(
        id=uuid.uuid4(), ticket_id=ticket.id, requester_id=ticket.requester_id,
        body="Incoming message", sender_type=sender, channel=channel,
        telegram_chat_id=chat_id,
        created_at=created or datetime.now(timezone.utc),
    )


def _stub_agent_user(monkeypatch):
    monkeypatch.setattr(
        main, "get_or_create_user",
        lambda _db, _telegram_id, _name="": type(
            "UserRef", (), {"id": uuid.uuid4()}
        )(),
    )


def _client(monkeypatch, db, role="agent", renders=None):
    monkeypatch.setattr(Settings, "validate_session_signing_secret", lambda self: TEST_SECRET)
    main.app.dependency_overrides[database_dependency] = lambda: db
    if renders is not None:
        monkeypatch.setattr(
            main.templates, "TemplateResponse",
            lambda **kwargs: (
                renders.append(kwargs)
                or HTMLResponse("rendered", status_code=kwargs.get("status_code", 200))
            ),
        )
    csrf, cookie = create_session(role, TEST_SECRET)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="https://test"
    )
    client.headers["cookie"] = f"{SESSION_COOKIE}={cookie}"
    return client, csrf


@pytest.fixture(autouse=True)
def clear_overrides():
    main.app.dependency_overrides.clear()
    yield
    main.app.dependency_overrides.clear()


def test_queue_rejects_duplicate_and_oversized_query_parameters(monkeypatch):
    db = _DB()

    async def exercise():
        client, _csrf = _client(monkeypatch, db)
        async with client:
            duplicate = await client.get("/ui/tickets?q=a&q=b")
            invalid_page = await client.get("/ui/tickets?page=0")
            oversized = await client.get("/ui/tickets?q=" + "x" * 201)
        assert duplicate.status_code == 422
        assert oversized.status_code == invalid_page.status_code == 422

    asyncio.run(exercise())


def test_panel_templates_and_local_assets_are_served(monkeypatch):
    db = _DB()

    async def exercise():
        client, _csrf = _client(monkeypatch, db)
        async with client:
            page = await client.get("/ui/tickets")
            stylesheet = await client.get("/ui/static/panel.css")
            htmx = await client.get("/ui/static/htmx.min.js")
        assert page.status_code == stylesheet.status_code == htmx.status_code == 200
        assert "Ticket queue" in page.text
        assert "/ui/static/panel.css" in page.text
        assert "localStorage" not in page.text
        assert page.headers["cache-control"] == "no-store"
        assert len(htmx.content) > 10_000

    asyncio.run(exercise())


def test_ticket_content_is_html_escaped(monkeypatch):
    ticket = _ticket(7)
    ticket.subject = "<script>alert(1)</script>"
    message = Message(
        id=uuid.uuid4(), ticket_id=ticket.id, requester_id=ticket.requester_id,
        body="<img src=x onerror=alert(1)>", sender_type="user",
        created_at=datetime.now(timezone.utc),
    )

    async def exercise():
        client, _csrf = _client(monkeypatch, _DB([ticket], messages=[message]))
        async with client:
            response = await client.get(f"/ui/tickets/{ticket.id}")
        assert response.status_code == 200
        assert "<script>alert(1)</script>" not in response.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
        assert "<img src=x onerror=alert(1)>" not in response.text

    asyncio.run(exercise())


def test_missing_session_redirects_and_agent_is_denied_knowledge(monkeypatch):
    db = _DB()

    async def exercise():
        client, _csrf = _client(monkeypatch, db, role="agent")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="https://test"
        ) as anonymous:
            response = await anonymous.get("/ui/tickets", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/ui/login"

        async with client:
            response = await client.get("/ui/knowledge")
            upload = await client.post("/ui/knowledge/upload", data={"csrf_token": "x"})
        assert response.status_code == 403
        assert upload.status_code == 403
        assert db.commits == 0

    asyncio.run(exercise())


def test_ticket_detail_renders_ticket_and_chronological_messages(monkeypatch):
    ticket = _ticket(4)
    later = Message(
        id=uuid.uuid4(), ticket_id=ticket.id, requester_id=ticket.requester_id,
        body="later", sender_type="agent",
        created_at=datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    earlier = Message(
        id=uuid.uuid4(), ticket_id=ticket.id, requester_id=ticket.requester_id,
        body="earlier", sender_type="user",
        created_at=datetime.now(timezone.utc),
    )
    db, rendered = _DB([ticket], messages=[later, earlier]), []

    async def exercise():
        client, _csrf = _client(monkeypatch, db, renders=rendered)
        async with client:
            response = await client.get(f"/ui/tickets/{ticket.id}")
        assert response.status_code == 200

    asyncio.run(exercise())
    context = rendered[0]["context"]
    assert rendered[0]["name"] == "ticket_detail.html"
    assert context["ticket"] is ticket
    assert [message.body for message in context["messages"]] == ["earlier", "later"]


def test_ticket_detail_separates_saved_message_from_unconfirmed_delivery(monkeypatch):
    ticket = _ticket(58)
    saved = Message(
        id=uuid.uuid4(), ticket_id=ticket.id, requester_id=ticket.requester_id,
        body="Saved reply", sender_type="agent", telegram_delivery_status="unconfirmed",
        telegram_delivery_updated_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )

    async def exercise():
        client, _csrf = _client(monkeypatch, _DB([ticket], messages=[saved]))
        async with client:
            history = await client.get(f"/ui/tickets/{ticket.id}")
            response = await client.get(
                f"/ui/tickets/{ticket.id}?message_id={saved.id}"
            )
        assert history.status_code == 200
        assert "Agent reply delivery unconfirmed" in history.text
        assert response.status_code == 200
        assert "Message saved." in response.text
        assert "Unconfirmed — Telegram delivery was not confirmed." in response.text

    asyncio.run(exercise())


def test_csrf_rejection_does_not_mutate_ticket(monkeypatch):
    ticket = _ticket(1)
    db = _DB([ticket])

    async def exercise():
        client, _csrf = _client(monkeypatch, db)
        async with client:
            message = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "response", "csrf_token": "wrong"},
            )
            update = await client.post(
                f"/ui/tickets/{ticket.id}/update",
                data={"status": "closed", "csrf_token": "wrong"},
            )
        assert message.status_code == update.status_code == 400
        assert db.commits == 0
        assert db.added == []
        assert ticket.status == "open"

    asyncio.run(exercise())


def test_message_form_rejects_duplicate_keys_and_oversized_body(monkeypatch):
    ticket = _ticket(2)
    db = _DB([ticket])

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            duplicate = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                content=f"body=one&body=two&csrf_token={csrf}",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )
            oversized = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "x" * 20_001, "csrf_token": csrf},
            )
        assert duplicate.status_code == 400
        assert oversized.status_code == 422
        assert db.commits == 0
        assert db.added == []

    asyncio.run(exercise())


def test_empty_update_uses_ticketpatch_validation_without_mutation(monkeypatch):
    ticket = _ticket(3)
    db = _DB([ticket])

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/update", data={"csrf_token": csrf}
            )
        assert response.status_code == 422
        assert "detail" in response.json()
        assert db.commits == 0
        assert ticket.status == "open"

    asyncio.run(exercise())


def test_message_and_ticket_update_preserve_sender_audit_and_close_timestamps(monkeypatch):
    ticket = _ticket(1)
    db = _DB([ticket])
    monkeypatch.setattr(main, "get_or_create_user",
                        lambda _db, _telegram_id, _name="": type("UserRef", (), {"id": uuid.uuid4()})())

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            added = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "Agent reply", "csrf_token": csrf},
                follow_redirects=False,
            )
            closed = await client.post(
                f"/ui/tickets/{ticket.id}/update",
                data={"status": "closed", "priority": "high", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert added.status_code == closed.status_code == 303
        assert added.headers["location"].startswith(f"/ui/tickets/{ticket.id}?message_id=")
        assert closed.headers["location"] == f"/ui/tickets/{ticket.id}"

    asyncio.run(exercise())
    message = next(value for value in db.added if isinstance(value, Message))
    audit = next(value for value in db.added if isinstance(value, AuditLog))
    assert message.body == "Agent reply"
    assert message.sender_type == "agent"
    assert ticket.status == "closed"
    assert ticket.closed_at is not None
    assert audit.action == "ticket.updated"
    assert set(audit.details["fields"]) == {"status", "priority"}

    async def reopen():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/update",
                data={"status": "in_progress", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(reopen())
    assert ticket.status == "in_progress"
    assert ticket.closed_at is None

    ticket.resolution = "previous resolution"

    async def clear_resolution():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/update",
                data={"resolution": "", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(clear_resolution())
    assert ticket.resolution is None


def test_eligible_panel_reply_at_4000_characters_is_saved_pending_then_sent_once(monkeypatch):
    ticket = _ticket(51)
    recipient = _telegram_message(ticket, chat_id="private-chat-51")
    db = _DB([ticket], messages=[recipient])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []

    async def send(chat_id, body):
        message = next(value for value in db.added if isinstance(value, Message))
        calls.append((chat_id, body, message.telegram_delivery_status, db.commits))

    monkeypatch.setattr(main, "send_telegram_message", send)
    body = "x" * 4000

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": body, "csrf_token": csrf}, follow_redirects=False,
            )
        assert response.status_code == 303
        assert response.headers["location"].startswith(f"/ui/tickets/{ticket.id}?message_id=")

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert calls == [("private-chat-51", body, "pending", 1)]
    assert saved.body == body
    assert saved.telegram_delivery_status == "accepted"
    assert saved.telegram_delivery_updated_at is not None
    assert db.commits == 2


def test_eligible_panel_reply_at_4001_is_rejected_before_save_or_send(monkeypatch):
    ticket = _ticket(52)
    db = _DB([ticket], messages=[_telegram_message(ticket, chat_id="private-chat-52")])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []
    monkeypatch.setattr(main, "send_telegram_message", lambda *_args: calls.append(True))
    rendered = []

    async def exercise():
        client, csrf = _client(monkeypatch, db, renders=rendered)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "x" * 4001, "csrf_token": csrf},
            )
        assert response.status_code == 400

    asyncio.run(exercise())
    assert "4,000 characters or fewer" in rendered[0]["context"]["error"]
    assert rendered[0]["context"]["body"] == "x" * 4001
    assert db.added == []
    assert db.commits == 0
    assert calls == []


def test_only_private_telegram_user_message_is_eligible_recipient(monkeypatch):
    ticket = _ticket(53)
    oldest_non_telegram = _telegram_message(
        ticket, chat_id="wrong-channel", channel="api",
        created=datetime.now(timezone.utc) - timedelta(seconds=3),
    )
    telegram_group = _telegram_message(
        ticket, chat_id=None,
        created=datetime.now(timezone.utc) - timedelta(seconds=2),
    )
    private = _telegram_message(
        ticket, chat_id="private-chat-53",
        created=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    db = _DB([ticket], messages=[private, telegram_group, oldest_non_telegram])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []

    async def send(chat_id, body):
        calls.append((chat_id, body))

    monkeypatch.setattr(main, "send_telegram_message", send)

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "private reply", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert calls == [("private-chat-53", "private reply")]
    assert saved.telegram_delivery_status == "accepted"


@pytest.mark.parametrize(
    "source",
    [
        {"channel": "telegram", "sender": "user", "chat_id": None},
        {"channel": "api", "sender": "user", "chat_id": "not-a-telegram-chat"},
        {"channel": "telegram", "sender": "agent", "chat_id": "agent-chat"},
    ],
)
def test_noneligible_panel_reply_is_saved_not_applicable_without_send(monkeypatch, source):
    ticket = _ticket(54)
    db = _DB([ticket], messages=[_telegram_message(ticket, **source)])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []
    monkeypatch.setattr(main, "send_telegram_message", lambda *_args: calls.append(True))

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "ordinary reply", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert saved.telegram_delivery_status == "not_applicable"
    assert saved.telegram_delivery_updated_at is not None
    assert calls == []


def test_eligible_panel_reply_without_token_is_not_configured_and_not_sent(monkeypatch):
    ticket = _ticket(55)
    db = _DB([ticket], messages=[_telegram_message(ticket, chat_id="private-chat-55")])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "")
    calls = []
    monkeypatch.setattr(main, "send_telegram_message", lambda *_args: calls.append(True))

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "reply", "csrf_token": csrf}, follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert saved.telegram_delivery_status == "not_configured"
    assert saved.telegram_delivery_updated_at is not None
    assert calls == []


def test_telegram_failure_preserves_message_and_redirects_unconfirmed(monkeypatch):
    ticket = _ticket(56)
    db = _DB([ticket], messages=[_telegram_message(ticket, chat_id="private-chat-56")])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []

    async def fail(chat_id, body):
        calls.append((chat_id, body))
        raise main.HTTPException(503, "Telegram message delivery failed")

    monkeypatch.setattr(main, "send_telegram_message", fail)

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "keep this reply", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert saved.body == "keep this reply"
    assert saved.telegram_delivery_status == "unconfirmed"
    assert saved.telegram_delivery_updated_at is not None
    assert calls == [("private-chat-56", "keep this reply")]
    assert db.commits == 2


def test_delivery_outcome_commit_failure_redirects_and_leaves_message_pending(monkeypatch):
    class OutcomeCommitFailureDB(_DB):
        def __init__(self, tickets, messages):
            super().__init__(tickets, messages)
            self.rollback_count = 0

        def commit(self):
            self.commits += 1
            if self.commits == 2:
                raise SQLAlchemyError("delivery outcome commit failed")

        def rollback(self):
            self.rollback_count += 1
            saved = next(value for value in self.added if isinstance(value, Message))
            saved.telegram_delivery_status = "pending"

    ticket = _ticket(59)
    db = OutcomeCommitFailureDB(
        [ticket], [_telegram_message(ticket, chat_id="private-chat-59")]
    )
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []

    async def send(chat_id, body):
        calls.append((chat_id, body))

    monkeypatch.setattr(main, "send_telegram_message", send)

    async def exercise():
        client, csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/ui/tickets/{ticket.id}/messages",
                data={"body": "persist before sending", "csrf_token": csrf},
                follow_redirects=False,
            )
        assert response.status_code == 303
        assert response.headers["location"].startswith(
            f"/ui/tickets/{ticket.id}?message_id="
        )

    asyncio.run(exercise())
    saved = next(value for value in db.added if isinstance(value, Message))
    assert saved.body == "persist before sending"
    assert saved.telegram_delivery_status == "pending"
    assert calls == [("private-chat-59", "persist before sending")]
    assert db.commits == 2
    assert db.rollback_count == 1


def test_api_agent_message_does_not_send_telegram(monkeypatch):
    ticket = _ticket(57)
    db = _DB([ticket], messages=[_telegram_message(ticket, chat_id="private-chat-57")])
    _stub_agent_user(monkeypatch)
    monkeypatch.setattr(main, "ROLE_TOKENS", {"user": "", "agent": "agent-test", "admin": ""})
    monkeypatch.setattr(main.settings, "telegram_token", "configured")
    calls = []
    monkeypatch.setattr(main, "send_telegram_message", lambda *_args: calls.append(True))

    async def exercise():
        client, _csrf = _client(monkeypatch, db)
        async with client:
            response = await client.post(
                f"/api/v1/tickets/{ticket.id}/messages",
                headers={"Authorization": "Bearer agent-test"},
                json={"body": "API reply"},
            )
        assert response.status_code == 201

    asyncio.run(exercise())
    assert calls == []
    assert db.commits == 1


def test_admin_knowledge_list_and_upload(monkeypatch):
    document = KnowledgeDocument(
        id=uuid.uuid4(), title="Runbook", source_name="runbook.txt", status="indexed",
        created_at=datetime.now(timezone.utc),
    )
    db, rendered = _DB(documents=[document]), []
    calls = []

    async def ingest(_file, _db):
        calls.append(True)
        return {"title": "runbook.txt", "chunks": 2}

    monkeypatch.setattr(main, "_ingest_knowledge_document", ingest)

    async def exercise():
        client, csrf = _client(monkeypatch, db, role="admin", renders=rendered)
        async with client:
            page = await client.get("/ui/knowledge")
            rejected = await client.post(
                "/ui/knowledge/upload", data={"csrf_token": "wrong"},
                files={"file": ("runbook.txt", b"content", "text/plain")},
            )
            upload = await client.post(
                "/ui/knowledge/upload",
                data={"csrf_token": csrf},
                files={"file": ("runbook.txt", b"content", "text/plain")},
                follow_redirects=False,
            )
        assert page.status_code == 200
        assert rendered[0]["name"] == "knowledge.html"
        assert rendered[0]["context"]["documents"] == [document]
        assert rejected.status_code == 400
        assert upload.status_code == 303
        assert upload.headers["location"].startswith("/ui/knowledge?success=")

    asyncio.run(exercise())
    assert calls == [True]


def test_upload_failure_renders_error_without_success_redirect(monkeypatch):
    db, rendered = _DB(), []

    async def fail_ingest(_file, _db):
        raise main.HTTPException(status_code=415, detail="Only TXT files are supported")

    monkeypatch.setattr(main, "_ingest_knowledge_document", fail_ingest)

    async def exercise():
        client, csrf = _client(monkeypatch, db, role="admin", renders=rendered)
        async with client:
            response = await client.post(
                "/ui/knowledge/upload",
                data={"csrf_token": csrf},
                files={"file": ("unsupported.exe", b"content", "application/octet-stream")},
                follow_redirects=False,
            )
        assert response.status_code == 415
        assert rendered[0]["name"] == "knowledge.html"
        assert rendered[0]["context"]["error"] == "Only TXT files are supported"
        assert rendered[0]["context"]["success"] is None

    asyncio.run(exercise())


def test_upload_type_and_size_limits_are_shared_with_api():
    from io import BytesIO
    from starlette.datastructures import UploadFile

    async def exercise():
        unsupported = UploadFile(filename="runbook.exe", file=BytesIO(b"data"))
        with pytest.raises(main.HTTPException) as type_error:
            await main._ingest_knowledge_document(unsupported, _DB())
        assert type_error.value.status_code == 415

        oversized = UploadFile(filename="large.txt", file=BytesIO(b"x" * 10_000_001))
        with pytest.raises(main.HTTPException) as size_error:
            await main._ingest_knowledge_document(oversized, _DB())
        assert size_error.value.status_code == 413

    asyncio.run(exercise())


def test_oversized_upload_body_is_rejected_before_form_parsing(monkeypatch):
    async def exercise():
        client, _csrf = _client(monkeypatch, _DB(), role="admin")
        oversized = b"x" * (main.MAX_UPLOAD_REQUEST_BYTES + 1)
        async with client:
            response = await client.post(
                "/ui/knowledge/upload",
                files={"file": ("large.txt", oversized, "text/plain")},
                follow_redirects=False,
            )
        assert response.status_code == 413

    asyncio.run(exercise())

@pytest.mark.parametrize("filename, content", [
    ("x" * 237 + ".txt", b"text"),
    ("runbook.txt", b"before\x00after"),
])
def test_upload_rejects_text_that_postgresql_cannot_store(filename, content):
    from io import BytesIO
    from starlette.datastructures import UploadFile

    file = UploadFile(filename=filename, file=BytesIO(content))
    with pytest.raises(main.HTTPException) as error:
        asyncio.run(main._ingest_knowledge_document(file, _DB()))
    assert error.value.status_code == 400


def test_waiting_for_reply_includes_tickets_created_without_messages():
    from sqlalchemy import create_engine, select

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE tickets (id INTEGER, status TEXT)")
        connection.exec_driver_sql(
            "CREATE TABLE messages (id INTEGER, ticket_id INTEGER, sender_type TEXT, created_at TEXT)"
        )
        connection.exec_driver_sql(
            "INSERT INTO tickets VALUES (1, 'open'), (2, 'open'), (3, 'in_progress'), (4, 'closed')"
        )
        connection.exec_driver_sql(
            "INSERT INTO messages VALUES (1, 2, 'agent', '2026-01-01'), "
            "(2, 3, 'agent', '2026-01-01'), (3, 3, 'user', '2026-01-02')"
        )
        waiting = connection.scalars(select(Ticket.status).where(main._awaiting_reply())
                                     .order_by(Ticket.id)).all()
    assert waiting == ["open", "in_progress"]
