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

Only two Kite endpoints are called, over plain HTTPS with the standard library
(no kiteconnect import, which pulls in twisted/autobahn): ``POST
/session/token`` (the exchange) and ``GET /user/profile`` (a validity check).
No order-placing endpoint exists in this module (CLAUDE.md). The token value is
never displayed or logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
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
    rec = {
        "access_token": access_token,
        "minted_at": minted.isoformat(timespec="seconds"),
        "expires_at": expires_at(minted).isoformat(timespec="seconds"),
        **meta,
    }
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, p)
    return {k: v for k, v in rec.items() if k != "access_token"}


API = "https://api.kite.trade"


class KiteError(Exception):
    """A Kite API error: ``kind`` is Kite's error_type (e.g. TokenException)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _call(
    method: str,
    path: str,
    *,
    api_key: str,
    token: str = "",
    form: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    headers = {"X-Kite-Version": "3"}
    if token:
        headers["Authorization"] = f"token {api_key}:{token}"
    data = urllib.parse.urlencode(form).encode() if form else None
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except ValueError:
            body = {}
        raise KiteError(
            body.get("error_type") or f"HTTP{e.code}", body.get("message") or str(e)
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise KiteError("NetworkError", str(e)) from None
    if body.get("status") != "success":
        raise KiteError(body.get("error_type") or "KiteError", body.get("message") or "failed")
    return body.get("data") or {}


def exchange(root: Path, request_token: str) -> dict:
    """Swap a request token for an access token, verify it, store it.

    Raises RuntimeError with a readable message on any failure; the token is
    stored only after the profile check accepts it.
    """
    key, secret = credentials(root)
    if not key or not secret:
        raise RuntimeError("KITE_API_KEY / KITE_API_SECRET not found in the environment or .env")
    checksum = hashlib.sha256((key + request_token + secret).encode()).hexdigest()
    try:
        sess = _call(
            "POST",
            "/session/token",
            api_key=key,
            form={"api_key": key, "request_token": request_token, "checksum": checksum},
        )
    except KiteError as e:
        raise RuntimeError(
            f"Kite refused the request token ({e.kind}: {e}). Request tokens are single-use "
            "and expire within minutes -- log in again."
        ) from None
    tok = sess.get("access_token", "")
    try:
        prof = _call("GET", "/user/profile", api_key=key, token=tok)
    except KiteError as e:
        raise RuntimeError(f"new token failed the profile check ({e.kind})") from None
    return write_token(
        root, tok, {"user_type": prof.get("user_type", ""), "broker": prof.get("broker", "")}
    )


def check(root: Path) -> dict:
    """Token status: state in {missing, valid, expired, error}, plus metadata. No token value."""
    d = read_token(root)
    if d is None:
        return {"state": "missing"}
    meta = {k: v for k, v in d.items() if k != "access_token"}
    key, _ = credentials(root)
    try:
        _call("GET", "/user/profile", api_key=key, token=d["access_token"])
        return {"state": "valid", **meta}
    except KiteError as e:
        state = "expired" if e.kind in ("TokenException", "PermissionException") else "error"
        return {"state": state, "error": e.kind, **meta}
