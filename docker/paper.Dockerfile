# The forward paper loop (audit/P2) as a container, so the schedule does not
# depend on macOS cron -- which never ran it: ~/storage is a USB volume, TCC
# denies cron access to removable volumes, and all 28 scheduled runs from
# 2026-09-10 to 09-25 failed with "Operation not permitted". Docker Desktop
# holds its own access to the volume.
#
# The repo is BIND-MOUNTED at /app, not copied: the loop's outputs (record,
# snapshots, data store) must land on the host, and the code the container runs
# must be the committed code in the working tree. Only the Python environment
# lives in the image, at /opt/venv -- the host .venv is macOS-arm64 and
# unusable here, and pointing uv at /opt/venv leaves it untouched.
FROM python:3.12-slim

ENV TZ=Asia/Kolkata \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_NO_SYNC=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# procps: `free` for memwatch.sh. tzdata: cron fires in IST. git: the loop
# never commits, but provenance reads (git rev-parse) are cheap to allow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps tzdata ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /usr/local/bin/

# supercronic: a cron built for containers -- runs in the foreground, logs to
# stdout, honours TZ, and does not need a system cron daemon.
ARG SUPERCRONIC_VERSION=v0.2.33
RUN arch=$(dpkg --print-architecture) \
    && curl -fsSL -o /usr/local/bin/supercronic \
       "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${arch}" \
    && chmod +x /usr/local/bin/supercronic

# Resolve the environment from the lockfile alone, so the image is rebuilt only
# when dependencies change.
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# The project itself is installed from the mount at run time via PYTHONPATH.
ENV PYTHONPATH=/app/src

WORKDIR /app
COPY docker/paper.crontab /etc/paper.crontab
CMD ["supercronic", "-passthrough-logs", "/etc/paper.crontab"]
