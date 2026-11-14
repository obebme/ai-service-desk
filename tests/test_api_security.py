import uuid

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app import main
from app.models import Ticket, User


def test_bearer_token_maps_to_configured_role(monkeypatch):
    monkeypatch.setattr(main.settings, "user_api_tokens", {"123": "user-secret"})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="user-secret")
    assert main.principal(credentials, "123") == "user"
    for wrong_identity in ("456", None):
        with pytest.raises(HTTPException) as error:
            main.principal(credentials, wrong_identity)
        assert error.value.status_code == 401


def test_bearer_token_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr(main, "ROLE_TOKENS", {"user": "user-secret", "agent": "", "admin": ""})
    with pytest.raises(HTTPException) as error:
        main.principal(HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong"))
    assert error.value.status_code == 401


def test_user_role_cannot_call_agent_route():
    guard = main.require("agent", "admin")
    with pytest.raises(HTTPException) as error:
        guard(role="user")
    assert error.value.status_code == 403


def test_user_cannot_read_another_users_ticket():
    owner_id = uuid.uuid4()
    ticket = Ticket(id=uuid.uuid4(), requester_id=owner_id)

    class FakeDB:
        def get(self, model, key):
            return ticket

        def scalar(self, query):
            return User(id=uuid.uuid4(), telegram_user_id="attacker")

    with pytest.raises(HTTPException) as error:
        main.ticket_detail(ticket.id, role="user", x_telegram_user_id="attacker", db=FakeDB())
    assert error.value.status_code == 404
