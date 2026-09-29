# Read-only monitor for the audit/P2 forward paper test (dashboard/app.py).
#
# Like the paper image, the repo is BIND-MOUNTED at /app rather than copied, so
# the page always shows the working tree's record and logs. Only the Python
# environment lives in the image. The build context is the repo root, but
# .dockerignore whitelists just dashboard/requirements.txt (+ the paper files),
# so it stays a few KB.
FROM python:3.12-slim

ENV TZ=Asia/Kolkata \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    DASHBOARD_ROOT=/app \
    PYTHONPATH=/app/src

COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Non-root: the page can shell out to scripts/paper_healthcheck.py, so it should
# not do so as uid 0. Fixed uid so the one writable bind mount (logs/dashboard)
# has a stable owner inside the container.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dash
USER 10001
ENV HOME=/home/dash

WORKDIR /app
EXPOSE 8501
# curl is not in slim; python does the healthcheck (see docker-compose.yml).
CMD ["streamlit", "run", "dashboard/app.py"]
