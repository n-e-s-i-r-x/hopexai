FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOPEXAI_CONFIG=/data/hopexai.config.yaml \
    HOPEXAI_STATE_DIR=/data/state

RUN useradd --system --uid 10001 --create-home --home-dir /home/hopexai hopexai
WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY config ./config

RUN mkdir -p /data/state && chown -R 10001:10001 /app /data
USER 10001:10001

EXPOSE 8000

CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers --forwarded-allow-ips='172.30.0.0/16'"]
