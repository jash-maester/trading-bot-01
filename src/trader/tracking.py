"""Where MLflow lives.

Every entry point used to hard-code ``http://localhost:<mlflow_port>``. That is
right on the host and wrong inside a container, where MLflow is the compose
service ``mlflow``. ``TRADER_MLFLOW_URI`` overrides it; it is set only by the
compose ``paper`` service and deliberately NOT in ``.env``, because the paper
loop sources ``.env`` after the container environment and anything there would
win.
"""
from __future__ import annotations

import os


def tracking_uri(port: int | str | None = 5555) -> str:
    override = os.environ.get("TRADER_MLFLOW_URI")
    if override:
        return override
    return f"http://localhost:{port if port is not None else 5555}"
