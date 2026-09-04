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

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
