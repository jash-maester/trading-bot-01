"""Kite daily login from the dashboard: login link, callback capture, token store.

Kite access tokens expire around 06:00 IST every day and can only be minted by
a browser login that redirects back with a single-use ``request_token``. This
module turns that into one click on the dashboard:

1. "Log in to Kite" opens ``kite.zerodha.com/connect/login?v=3&api_key=...``.
2. Kite redirects to the app's registered redirect URL. When that URL is the
   dashboard (``http://127.0.0.1:8501/``), the page reads ``request_token``
   from its own query string and exchanges it here. Otherwise the user pastes
   the redirect URL (or the bare token) into the page.
3. The access token is written to ``secrets/kite/access_token.json`` (mode
   0600, gitignored). ``scripts/live_paper.py`` reads it before ``.env``.

Only authentication endpoints are called: ``generate_session`` (the exchange)
and ``profile`` (a validity check). No order-placing endpoint is imported or
called (CLAUDE.md). The token value is never displayed or logged.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

IST = timezone(timedelta(hours=5, minutes=30))
TOKEN_REL = Path("secrets/kite/access_token.json")
LOGIN = "https://kite.zerodha.com/connect/login?v=3&api_key={key}"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{16,64}$")


def token_path(root: Path) -> Path:
    return root / TOKEN_REL


def credentials(root: Path) -> tuple[str, str]:
    """(api_key, api_secret) from the environment, else the repo's .env.

    The dashboard sees the repo read-only at /app, so .env is readable there;
    only these two keys are taken from it.
    """
    key = os.environ.get("KITE_API_KEY", "")
    secret = os.environ.get("KITE_API_SECRET", "")
    env = root / ".env"
    if (not key or not secret) and env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"^\s*(?:export\s+)?(KITE_API_KEY|KITE_API_SECRET)\s*=\s*(.*)$", line)
            if m:
                v = m.group(2).strip().strip("'\"")
                if m.group(1) == "KITE_API_KEY" and not key:
                    key = v
                elif m.group(1) == "KITE_API_SECRET" and not secret:
                    secret = v
    return key, secret


def login_url(api_key: str) -> str:
    return LOGIN.format(key=api_key)


def parse_request_token(text: str) -> str | None:
    """A request token from a pasted redirect URL or a bare token; None if neither.

    A redirect that says ``status`` other than ``success`` yields None.
    """
    text = (text or "").strip()
    if not text:
        return None
    if "request_token" in text:
        q = parse_qs(urlparse(text).query) if "?" in text else parse_qs(text)
        if q.get("status", ["success"])[0] != "success":
            return None
        tok = (q.get("request_token") or [""])[0]
        return tok if _TOKEN_RE.match(tok) else None
    return text if _TOKEN_RE.match(text) else None


def expires_at(minted: datetime) -> datetime:
    """Kite tokens lapse at ~06:00 IST the morning after they were minted
    (or the same morning, if minted before 06:00)."""
    m = minted.astimezone(IST)
    six = datetime.combine(m.date(), time(6, 0), IST)
    return six if m < six else six + timedelta(days=1)


def read_token(root: Path) -> dict | None:
    p = token_path(root)
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if d.get("access_token") else None


def write_token(root: Path, access_token: str, meta: dict) -> dict:
    """Atomically write the token file, 0600. Returns the stored metadata (no token)."""
    p = token_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    minted = datetime.now(IST)
    rec = {"access_token": access_token, "minted_at": minted.isoformat(timespec="seconds"),
           "expires_at": expires_at(minted).isoformat(timespec="seconds"), **meta}
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, p)
    return {k: v for k, v in rec.items() if k != "access_token"}


def exchange(root: Path, request_token: str) -> dict:
    """Swap a request token for an access token, verify it, store it.

    Raises RuntimeError with a readable message on any failure; the token is
    stored only after ``profile`` accepts it.
    """
    import kiteconnect  # noqa: PLC0415

    key, secret = credentials(root)
    if not key or not secret:
        raise RuntimeError("KITE_API_KEY / KITE_API_SECRET not found in the environment or .env")
    k = kiteconnect.KiteConnect(api_key=key)
    try:
        sess = k.generate_session(request_token, api_secret=secret)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Kite refused the request token ({type(e).__name__}: {e}). Request tokens are "
            "single-use and expire within minutes -- log in again."
        ) from None
    tok = sess.get("access_token", "")
    k.set_access_token(tok)
    try:
        prof = k.profile()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"new token failed the profile check ({type(e).__name__})") from None
    return write_token(root, tok, {"user_type": prof.get("user_type", ""),
                                   "broker": prof.get("broker", "")})


def check(root: Path) -> dict:
    """Token status: state in {missing, valid, expired, error}, plus metadata. No token value."""
    d = read_token(root)
    if d is None:
        return {"state": "missing"}
    meta = {k: v for k, v in d.items() if k != "access_token"}
    key, _ = credentials(root)
    try:
        import kiteconnect  # noqa: PLC0415

        k = kiteconnect.KiteConnect(api_key=key)
        k.set_access_token(d["access_token"])
        k.profile()
        return {"state": "valid", **meta}
    except ImportError:
        return {"state": "error", "error": "kiteconnect not installed", **meta}
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        state = "expired" if name in ("TokenException", "PermissionException") else "error"
        return {"state": state, "error": name, **meta}
