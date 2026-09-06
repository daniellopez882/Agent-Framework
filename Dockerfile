FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY agent_framework ./agent_framework
COPY examples ./examples
RUN pip install --no-deps .

# Non-root; state goes to a volume, not the image.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data
USER app
ENV AGENT_DB_PATH=/data/agent_memory.db \
    CONTEXT_PERSIST_DIR=/data/context_db
VOLUME ["/data"]

# Proves the install and prints the effective configuration; run an example
# with: docker run --rm --env-file .env <image> python examples/06_tools.py
CMD ["python", "-m", "agent_framework", "check"]
