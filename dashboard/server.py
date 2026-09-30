"""Paper Book web dashboard: a standard-library HTTP server. No framework.

    python dashboard/server.py            # http://127.0.0.1:8501

Serves dashboard/web/ (HTML, CSS, JS, fonts) and a read-only JSON API built by
dashboard/api.py. Two writes exist, both small and both guarded:

* the Kite login: Kite redirects the browser to ``/?request_token=...``
  (the redirect URL registered at developers.kite.trade); the server exchanges
  it, stores the token (dashboard/kite_auth.py) and redirects to ``/#kite``.
  ``POST /api/kite/token`` is the paste fallback.
* ``POST /api/health/rerun`` runs scripts/paper_healthcheck.py, rate-limited.

POSTs must carry ``X-Paper-Book: 1``. A browser cannot add a custom header to a
cross-site request without a CORS preflight, which this server never answers,
so another website cannot trigger them through the user's browser.

No order-placing endpoint exists anywhere in the dashboard.
"""

from __future__ import annotations

import gzip
import json
import mimetypes
import os
import sys
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402
import kite_auth as KA  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("text/javascript", ".js")

GET_ROUTES = {
    "/api/status": api.status,
    "/api/portfolio": api.portfolio,
    "/api/experiment": api.experiment,
    "/api/health": api.health,
    "/api/research": api.research,
    "/api/warmup": api.warmup,
}


def _default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return str(o)


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperBook/1"
    protocol_version = "HTTP/1.1"

    def log_request(self, code="-", size="-") -> None:
        """Quiet access log: errors only (a 30 s poll would otherwise flood it)."""
        if str(getattr(code, "value", code))[:1] in ("4", "5"):
            super().log_request(code, size)

    # ── responses ────────────────────────────────────────────────────────────
    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store",
              extra: dict | None = None) -> None:
        if len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
            body = gzip.compress(body, compresslevel=5)
            extra = {**(extra or {}), "Content-Encoding": "gzip"}
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=_default, separators=(",", ":")).encode(),
                   "application/json; charset=utf-8")

    def _redirect(self, to: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _static(self, rel: str) -> None:
        p = (WEB / rel).resolve()
        if WEB not in p.parents or not p.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        # fonts never change; html/css/js revalidate so a deploy shows at once
        cache = "public, max-age=31536000, immutable" if "/fonts/" in "/" + rel else "no-cache"
        self._send(200, p.read_bytes(), ctype, cache)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/kite/callback") and ("request_token" in q or "status" in q):
                return self._kite_callback(u.query)
            if u.path == "/healthz":
                return self._send(200, b"ok", "text/plain")
            if u.path in GET_ROUTES:
                return self._json(GET_ROUTES[u.path]())
            if u.path == "/api/doc":
                text = api.doc((q.get("name") or [""])[0])
                if text is None:
                    return self._json({"error": "not found"}, 404)
                return self._send(200, text.encode(), "text/markdown; charset=utf-8")
            if u.path == "/api/kite/login-url":
                key, _ = KA.credentials(api.ROOT)
                return self._json({"url": KA.login_url(key) if key else None})
            if u.path == "/" or u.path == "/index.html":
                return self._static("index.html")
            return self._static(u.path.lstrip("/"))
        except Exception as e:  # noqa: BLE001 -- never a stack trace to the page
            sys.stderr.write(f"GET {u.path}: {type(e).__name__}: {e}\n")
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _kite_callback(self, query: str) -> None:
        rt = KA.parse_request_token(query)
        if rt is None:
            return self._redirect("/#kite?err=" + quote("Kite login was not successful."))
        try:
            meta = KA.exchange(api.ROOT, rt)
            api.kite_status(force=True)
            return self._redirect("/#kite?ok=" + quote(meta["expires_at"]))
        except RuntimeError as e:
            return self._redirect("/#kite?err=" + quote(str(e)[:300]))

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self) -> None:
        if self.headers.get("X-Paper-Book") != "1":
            return self._json({"error": "forbidden"}, 403)
        n = int(self.headers.get("Content-Length") or 0)
        if n > 4096:
            return self._json({"error": "too large"}, 413)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad json"}, 400)
        u = urlparse(self.path)
        if u.path == "/api/kite/token":
            rt = KA.parse_request_token(str(body.get("text", "")))
            if rt is None:
                return self._json({"ok": False, "error": "No request_token found, or the "
                                   "login status was not success."})
            try:
                meta = KA.exchange(api.ROOT, rt)
                api.kite_status(force=True)
                return self._json({"ok": True, "expires_at": meta["expires_at"]})
            except RuntimeError as e:
                return self._json({"ok": False, "error": str(e)})
        if u.path == "/api/health/rerun":
            return self._json(api.rerun_health())
        return self._json({"error": "not found"}, 404)


def main() -> None:
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8501"))
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"Paper Book on http://{host}:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
