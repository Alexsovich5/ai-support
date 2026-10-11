FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

COPY requirements.txt requirements-lock.txt ./
RUN pip install -r requirements.txt -c requirements-lock.txt

FROM base AS dev

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt -c requirements-lock.txt
COPY . .
CMD ["pytest"]

FROM base AS runtime

COPY src ./src
COPY config ./config
COPY stubs ./stubs
COPY data/kb ./data/kb
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
