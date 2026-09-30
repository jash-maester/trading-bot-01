# Paper Book web dashboard (dashboard/server.py): a standard-library HTTP server
# plus static HTML/CSS/JS. No framework.
#
# The repo is BIND-MOUNTED at /app (read-only) rather than copied, so the page
# always shows the working tree's record and logs. Only the Python environment
# lives in the image. The build context is the repo root, but .dockerignore
# whitelists just dashboard/requirements.txt, so it stays a few KB.
FROM python:3.12-slim

ENV TZ=Asia/Kolkata \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DASHBOARD_ROOT=/app \
    DASHBOARD_HOST=0.0.0.0 \
    DASHBOARD_PORT=8501 \
    PYTHONPATH=/app/src

COPY dashboard/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Non-root: the page can run scripts/paper_healthcheck.py, so not as uid 0.
# Fixed uid so the writable bind mounts (logs/dashboard, secrets/kite) have a
# stable owner inside the container.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin dash
USER 10001
ENV HOME=/home/dash

WORKDIR /app
EXPOSE 8501
CMD ["python", "dashboard/server.py"]
