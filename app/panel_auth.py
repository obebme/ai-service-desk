"""Stateless authentication primitives for the server-rendered panel."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping

from fastapi import HTTPException, Request

from app.config import settings

ROLE_TOKENS = {
    "agent": settings.agent_api_token,
    "admin": settings.admin_api_token,
}

SESSION_COOKIE = "servicedesk_session"
CHALLENGE_COOKIE = "servicedesk_login_challenge"
SESSION_TTL_SECONDS = 8 * 60 * 60
CHALLENGE_TTL_SECONDS = 10 * 60
PANEL_ROLES = frozenset({"agent", "admin"})


def _require_secret(secret: str) -> None:
    if not secret or not secret.strip():
        raise RuntimeError("SESSION_SIGNING_SECRET is required to use panel authentication")


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign(payload: Mapping[str, object], secret: str) -> str:
    _require_secret(secret)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = _b64encode(serialized)
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_b64encode(signature)}"


def _unsign(value: str | None, secret: str) -> dict | None:
    _require_secret(secret)
    if not value or len(value) > 4096 or value.count(".") != 1:
        return None
    encoded, signature_text = value.split(".", 1)
    try:
        encoded_bytes = encoded.encode("ascii")
        signature = _b64decode(signature_text)
    except (ValueError, UnicodeEncodeError):
        return None
    expected = hmac.new(secret.encode("utf-8"), encoded_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_b64decode(encoded))
    except (ValueError, UnicodeDecodeError, UnicodeEncodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def create_challenge(secret: str, *, now: int | None = None) -> tuple[str, str]:
    csrf = secrets.token_urlsafe(32)
    expiry = (int(time.time()) if now is None else now) + CHALLENGE_TTL_SECONDS
    return csrf, _sign({"csrf": csrf, "exp": expiry}, secret)


def verify_challenge(cookie: str | None, submitted_csrf: str, secret: str,
                     *, now: int | None = None) -> bool:
    payload = _unsign(cookie, secret)
    if not payload or set(payload) != {"csrf", "exp"}:
        return False
    current_time = int(time.time()) if now is None else now
    expected = payload["csrf"]
    expiry = payload["exp"]
    return (
        isinstance(expected, str)
        and type(expiry) is int
        and expiry > current_time
        and hmac.compare_digest(expected.encode("utf-8"), submitted_csrf.encode("utf-8"))
    )


def create_session(role: str, secret: str, *, now: int | None = None, agent: str | None = None) -> tuple[str, str]:
    if role not in PANEL_ROLES:
        raise ValueError("Panel sessions are only available to Agent and Admin roles")
    csrf = secrets.token_urlsafe(32)
    expiry = (int(time.time()) if now is None else now) + SESSION_TTL_SECONDS
    payload = {"role": role, "csrf": csrf, "exp": expiry}
    if agent is not None:
        payload["agent"] = agent
    return csrf, _sign(payload, secret)


def read_session(cookie: str | None, secret: str, *, now: int | None = None) -> dict | None:
    payload = _unsign(cookie, secret)
    if not payload or set(payload) not in ({"role", "csrf", "exp"}, {"role", "csrf", "exp", "agent"}):
        return None
    role, csrf, expiry = payload["role"], payload["csrf"], payload["exp"]
    current_time = int(time.time()) if now is None else now
    if (not isinstance(role, str) or role not in PANEL_ROLES
            or not isinstance(csrf, str) or type(expiry) is not int
            or expiry <= current_time):
        return None
    return payload


def authenticate_role_token(token: str, role_tokens: Mapping[str, str]) -> str | None:
    """Return one unambiguous configured Agent/Admin role, never User."""
    candidate = token.encode("utf-8")
    matches = [
        role for role in ("user", "agent", "admin")
        if role_tokens.get(role)
        and hmac.compare_digest(candidate, role_tokens[role].encode("utf-8"))
    ]
    return matches[0] if len(matches) == 1 and matches[0] in PANEL_ROLES else None


def session_role(request: Request, secret: str) -> str | None:
    payload = read_session(request.cookies.get(SESSION_COOKIE), secret)
    return payload["role"] if payload else None


def require_session_role(request: Request, secret: str,
                         allowed_roles: tuple[str, ...] = ("agent", "admin")) -> str:
    role = session_role(request, secret)
    if role is None:
        raise HTTPException(status_code=401, detail="Panel session required")
    if role not in allowed_roles:
        raise HTTPException(status_code=403, detail="Insufficient panel role")
    return role


def verify_csrf(submitted: str, expected: str) -> bool:
    return hmac.compare_digest(submitted.encode("utf-8"), expected.encode("utf-8"))
