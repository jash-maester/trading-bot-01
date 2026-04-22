FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow>=2.14" && \
    mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
