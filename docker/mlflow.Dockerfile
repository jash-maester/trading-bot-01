FROM python:3.12-slim

# Pinned, not ">=2.14": the floating range resolved to 3.16.0, whose new
# localhost-only security middleware + heavier deps crash-looped the
# workers under the container default memory. 3.11.1 also matches the
# client in .venv, avoiding server/client skew.
# starlette/anyio are pinned alongside mlflow because pinning mlflow alone is
# not enough: pip resolved starlette 1.6.0 + anyio 4.15.0 here, and
# starlette's WSGI middleware then calls `anyio.from_thread.run`, which that
# anyio exposes only through a lazy import — the server started, then every
# request died with "module 'anyio' has no attribute 'from_thread'" and the
# healthcheck returned 500. These are the versions the working .venv resolves.
RUN pip install --no-cache-dir \
        "mlflow==3.11.1" \
        "starlette==1.0.0" \
        "anyio==4.13.0" && \
    mkdir -p /mlflow/artifacts

EXPOSE 5000

# --allowed-hosts: MLflow 3.x rejects any Host header outside localhost and
# private IPs with a 403 ("possible DNS rebinding attack"). Inside the compose
# network the paper container reaches it as mlflow:5000, which that default
# does not cover. Listed explicitly rather than widened to "*".
# --workers 1: the default 4 held 1.8 GB for a server that logs one daily job,
# on a Mac whose Docker VM is capped at 9 GB.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5555,127.0.0.1,127.0.0.1:5555", \
     "--workers", "1"]
