FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /run/ai-support \
    && chown 10001:10001 /run/ai-support \
    && chmod 0700 /run/ai-support

WORKDIR /app

COPY requirements.txt requirements-lock.txt ./
RUN pip install -r requirements.txt -c requirements-lock.txt

FROM base AS dev

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt -c requirements-lock.txt
COPY . .
RUN chown -R 10001:10001 /app
CMD ["pytest"]

FROM base AS runtime

COPY src ./src
COPY config ./config
COPY stubs ./stubs
COPY data/kb ./data/kb
RUN mkdir -p /app/data/chroma && chown 10001:10001 /app/data/chroma
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-max-size", "65536"]
