FROM python:3.12-slim

# Pinned, not ">=2.14": the floating range resolved to 3.16.0, whose new
# localhost-only security middleware + heavier deps crash-looped the
# workers under the container default memory. 3.11.1 also matches the
# client in .venv, avoiding server/client skew.
RUN pip install --no-cache-dir "mlflow==3.11.1" && \
    mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
