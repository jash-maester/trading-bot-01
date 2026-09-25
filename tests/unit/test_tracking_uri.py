from trader.tracking import tracking_uri


def test_host_default_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("TRADER_MLFLOW_URI", raising=False)
    assert tracking_uri(5555) == "http://localhost:5555"


def test_container_override(monkeypatch) -> None:
    monkeypatch.setenv("TRADER_MLFLOW_URI", "http://mlflow:5000")
    assert tracking_uri(5555) == "http://mlflow:5000"
