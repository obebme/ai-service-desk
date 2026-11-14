import asyncio
import re
import time

import httpx
import pytest
from fastapi import HTTPException

from app import main
from app.config import Settings
from app.panel_auth import (
    CHALLENGE_COOKIE,
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    authenticate_role_token,
    create_challenge,
    create_session,
    read_session,
    require_session_role,
    verify_challenge,
)

TEST_SIGNING_SECRET = "unit-test-session-signing-secret"


def _set_auth_config(monkeypatch):
    monkeypatch.setattr(main.settings, "session_signing_secret", TEST_SIGNING_SECRET)
    monkeypatch.setattr(main, "ROLE_TOKENS", {
        "user": "user-role-token",
        "agent": "agent-role-token",
        "admin": "admin-role-token",
    })


def test_login_role_mapping_and_user_rejection():
    tokens = {"user": "user-token", "agent": "agent-token", "admin": "admin-token"}
    assert authenticate_role_token("agent-token", tokens) == "agent"
    assert authenticate_role_token("admin-token", tokens) == "admin"
    assert authenticate_role_token("user-token", tokens) is None
    assert authenticate_role_token("unknown-token", tokens) is None
    assert authenticate_role_token("same", {"user": "same", "agent": "same", "admin": ""}) is None


def test_login_csrf_cookie_flags_and_signed_eight_hour_session(monkeypatch):
    _set_auth_config(monkeypatch)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="https://test"
        ) as client:
            page = await client.get("/ui/login")
            challenge_cookie = next(
                value for value in page.headers.get_list("set-cookie")
                if value.startswith(CHALLENGE_COOKIE + "=")
            )
            assert "httponly" in challenge_cookie.lower()
            assert "secure" in challenge_cookie.lower()
            assert "samesite=lax" in challenge_cookie.lower()
            csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)

            response = await client.post("/ui/login", data={
                "csrf_token": csrf, "role_token": "admin-role-token",
            }, follow_redirects=False)
            assert response.status_code == 303
            assert response.headers["location"] == "/ui/tickets"
            assert "admin-role-token" not in response.text
            session_cookie = next(
                value for value in response.headers.get_list("set-cookie")
                if value.startswith(SESSION_COOKIE + "=")
            )
            assert "httponly" in session_cookie.lower()
            assert "secure" in session_cookie.lower()
            assert "samesite=lax" in session_cookie.lower()
            assert f"max-age={SESSION_TTL_SECONDS}" in session_cookie.lower()
            payload = read_session(client.cookies.get(SESSION_COOKIE), TEST_SIGNING_SECRET)
            assert payload is not None
            assert set(payload) == {"role", "csrf", "exp"}
            assert payload["role"] == "admin"
            assert payload["exp"] - int(time.time()) == pytest.approx(SESSION_TTL_SECONDS, abs=2)
            assert "admin-role-token" not in client.cookies.get(SESSION_COOKIE)
    asyncio.run(exercise())


def test_login_rejects_user_token_after_valid_csrf(monkeypatch):
    _set_auth_config(monkeypatch)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="https://test"
        ) as client:
            page = await client.get("/ui/login")
            csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
            response = await client.post("/ui/login", data={
                "csrf_token": csrf, "role_token": "user-role-token",
            })
            assert response.status_code == 401
            assert "user-role-token" not in response.text
            assert SESSION_COOKIE not in client.cookies
    asyncio.run(exercise())


def test_login_and_logout_reject_invalid_csrf(monkeypatch):
    _set_auth_config(monkeypatch)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="https://test"
        ) as client:
            page = await client.get("/ui/login")
            csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
            bad_login = await client.post("/ui/login", data={
                "csrf_token": "invalid", "role_token": "agent-role-token",
            })
            assert bad_login.status_code == 400
            assert SESSION_COOKIE not in client.cookies

            valid_login = await client.post("/ui/login", data={
                "csrf_token": csrf, "role_token": "agent-role-token",
            }, follow_redirects=False)
            assert valid_login.status_code == 303
            bad_logout = await client.post("/ui/logout", data={"csrf_token": "invalid"})
            assert bad_logout.status_code == 400
            assert SESSION_COOKIE in client.cookies
    asyncio.run(exercise())


def test_logout_clears_session_cookie(monkeypatch):
    _set_auth_config(monkeypatch)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="https://test"
        ) as client:
            page = await client.get("/ui/login")
            csrf = re.search(r'name="csrf_token" value="([^"]+)"', page.text).group(1)
            await client.post("/ui/login", data={
                "csrf_token": csrf, "role_token": "agent-role-token",
            }, follow_redirects=False)
            payload = read_session(client.cookies.get(SESSION_COOKIE), TEST_SIGNING_SECRET)
            response = await client.post("/ui/logout", data={"csrf_token": payload["csrf"]},
                                         follow_redirects=False)
            assert response.status_code == 303
            assert response.headers["location"] == "/ui/login"
            expired = next(
                value for value in response.headers.get_list("set-cookie")
                if value.startswith(SESSION_COOKIE + "=")
            )
            assert "max-age=0" in expired.lower()
            assert "secure" in expired.lower()
            assert "httponly" in expired.lower()
            assert "samesite=lax" in expired.lower()
            assert SESSION_COOKIE not in client.cookies
    asyncio.run(exercise())


def test_session_route_authorization_helper():
    from starlette.requests import Request
    from app.panel_auth import create_session

    _csrf, cookie = create_session("agent", TEST_SIGNING_SECRET)
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/ui/tickets",
        "headers": [],
        "query_string": b"",
        "server": ("test", 443),
        "client": ("test", 1234),
        "scheme": "https",
    })
    request._cookies = {SESSION_COOKIE: cookie}
    assert require_session_role(request, TEST_SIGNING_SECRET) == "agent"
    with pytest.raises(HTTPException) as error:
        require_session_role(request, TEST_SIGNING_SECRET, allowed_roles=("admin",))
    assert error.value.status_code == 403


def test_pre_auth_challenge_and_session_expire():
    csrf, challenge = create_challenge(TEST_SIGNING_SECRET, now=1000)
    assert verify_challenge(challenge, csrf, TEST_SIGNING_SECRET, now=1001)
    assert not verify_challenge(challenge, csrf, TEST_SIGNING_SECRET,
                                now=1000 + 10 * 60)

    _session_csrf, session = create_session("agent", TEST_SIGNING_SECRET, now=1000)
    assert read_session(session, TEST_SIGNING_SECRET, now=1000 + SESSION_TTL_SECONDS - 1)
    assert read_session(session, TEST_SIGNING_SECRET, now=1000 + SESSION_TTL_SECONDS) is None


def test_missing_session_signing_secret_fails_app_startup(monkeypatch):
    missing = Settings(_env_file=None, session_signing_secret=None)
    with pytest.raises(RuntimeError, match="SESSION_SIGNING_SECRET is required"):
        missing.validate_session_signing_secret()

    weak = Settings(_env_file=None, session_signing_secret="too-short")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        weak.validate_session_signing_secret()

    monkeypatch.setattr(main, "settings", missing)

    async def startup():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="SESSION_SIGNING_SECRET is required"):
        asyncio.run(startup())
