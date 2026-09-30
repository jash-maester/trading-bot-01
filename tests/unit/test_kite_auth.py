"""dashboard/kite_auth.py: token parsing, expiry, storage. No network."""
from __future__ import annotations

import json
import stat
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))
import kite_auth as KA  # noqa: E402

TOK = "tBmFw8A2rXx5ZzQHdb9UjpM72bgmYbvX"


def test_parse_redirect_url_and_bare_token():
    url = f"http://127.0.0.1:5000/kite/callback?action=login&type=login&status=success&request_token={TOK}"
    assert KA.parse_request_token(url) == TOK
    assert KA.parse_request_token(f"  {TOK}\n") == TOK
    assert KA.parse_request_token(f"action=login&status=success&request_token={TOK}") == TOK


def test_parse_rejects_failed_login_and_junk():
    assert KA.parse_request_token(f"http://x/?status=cancelled&request_token={TOK}") is None
    assert KA.parse_request_token("not a token!") is None
    assert KA.parse_request_token("") is None
    assert KA.parse_request_token("http://x/?request_token=abc;rm -rf") is None


def test_expiry_is_next_six_am_ist():
    ist = KA.IST
    assert KA.expires_at(datetime(2026, 9, 30, 7, 40, tzinfo=ist)) == datetime(2026, 10, 1, 6, 0, tzinfo=ist)
    assert KA.expires_at(datetime(2026, 9, 30, 5, 0, tzinfo=ist)) == datetime(2026, 9, 30, 6, 0, tzinfo=ist)


def test_write_token_is_owner_only_and_meta_hides_token(tmp_path):
    meta = KA.write_token(tmp_path, "secret-token", {"user_type": "individual"})
    p = KA.token_path(tmp_path)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert "access_token" not in meta and meta["user_type"] == "individual"
    assert json.loads(p.read_text())["access_token"] == "secret-token"
    assert KA.read_token(tmp_path)["access_token"] == "secret-token"


def test_credentials_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_API_SECRET", raising=False)
    (tmp_path / ".env").write_text("FOO=1\nKITE_API_KEY=abc\nexport KITE_API_SECRET='xyz'\n")
    assert KA.credentials(tmp_path) == ("abc", "xyz")


def test_check_missing(tmp_path):
    assert KA.check(tmp_path)["state"] == "missing"
