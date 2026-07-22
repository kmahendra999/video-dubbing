"""Authentication.

The app binds 0.0.0.0 so it's reachable from the LAN, and it holds licensed
voices, signed consent records and pre-release footage. Unauthenticated is not
an option once anything real is in it.

Deliberately simple: one shared password, an HMAC-signed session cookie, no user
table. This is a single-operator studio tool, not a SaaS — per-user accounts and
roles are a real feature, not a default, and pretending otherwise would add a
login table nobody maintains. When you need real accounts, this is the seam.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time

from fastapi import HTTPException, Request

from .config import settings

log = logging.getLogger(__name__)

COOKIE = "dub_session"
TTL = 60 * 60 * 24 * 14  # 14 days

# Paths reachable without a session. Everything else requires one.
OPEN_PATHS = {"/api/login", "/api/login-hint", "/api/health", "/login", "/favicon.ico"}
OPEN_PREFIXES = ("/static/",)


def enabled() -> bool:
    return bool(settings.app_password)


def _secret() -> bytes:
    """Derive the signing key from the password, so changing the password
    invalidates every existing session — which is what you want."""
    base = (settings.secret_key or "") + "|" + settings.app_password
    return hashlib.sha256(base.encode()).digest()


def issue() -> str:
    exp = int(time.time()) + TTL
    payload = str(exp)
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    payload, _, sig = token.rpartition(".")
    expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        return int(payload) > time.time()
    except ValueError:
        return False


def check_password(candidate: str) -> bool:
    # compare_digest: constant-time, so a wrong password can't be found by timing.
    return bool(settings.app_password) and secrets.compare_digest(
        candidate or "", settings.app_password
    )


def is_open(path: str) -> bool:
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


def bound_locally() -> bool:
    """True only when the UI is reachable from this machine and nowhere else."""
    return os.environ.get("UI_BIND", "0.0.0.0") in ("127.0.0.1", "localhost", "::1")


def hint() -> str | None:
    """The password, for the login page — or None.

    Deliberately gated on TWO things, not one. Showing a password to anyone who
    can load the page removes the authentication entirely, so this is only
    honoured when the app is also bound to loopback. Setting the flag while
    bound to 0.0.0.0 publishes the password to your whole LAN, so we refuse and
    say why rather than trusting the flag alone.
    """
    if not settings.dev_show_password:
        return None
    if not bound_locally():
        log.warning(
            "DEV_SHOW_PASSWORD is set but UI_BIND=%s — refusing to print the password on a "
            "page your LAN can load. Set UI_BIND=127.0.0.1 to make it genuinely local first.",
            os.environ.get("UI_BIND", "0.0.0.0"),
        )
        return None
    return settings.app_password


async def middleware(request: Request, call_next):
    if not enabled() or is_open(request.url.path):
        return await call_next(request)

    if valid(request.cookies.get(COOKIE)):
        return await call_next(request)

    # An API caller wants 401; a browser wants the login page.
    from fastapi.responses import FileResponse, JSONResponse
    from pathlib import Path

    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "authentication required"}, status_code=401)
    return FileResponse(Path(__file__).parent / "static" / "login.html", status_code=401)


def warn_if_open() -> None:
    if enabled():
        return
    bind = os.environ.get("UI_BIND", "0.0.0.0")
    log.warning(
        "⚠️  NO PASSWORD SET (APP_PASSWORD is empty) — this instance is unauthenticated%s. "
        "Anyone who can reach it can read, delete and dub. Set APP_PASSWORD in .env.",
        " and bound to " + bind + " (reachable from your LAN)" if bind == "0.0.0.0" else "",
    )
