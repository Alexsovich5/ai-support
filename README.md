# ai-support

A small FastAPI service for an IT help desk. It classifies support tickets into category, priority and severity and routes them to a team with SLA targets. It answers support questions in chat (REST and WebSocket) using retrieval-augmented generation over a Chroma knowledge base of how-to articles, and it lets operators ingest and search those articles. Because the intended setting is a healthcare IT desk, it strips obvious patient identifiers (record numbers, SSN-style IDs, dates of birth, phone numbers, email addresses) from chat messages and ticket text before they reach a language model.

Personal project built on the 2024-era stack (Python 3.11, FastAPI, LangChain 0.1, Chroma 0.5).

## Status

**Implemented**

- Ticket classification: `POST /api/v1/classify` asks a chat model for JSON and validates it against the category, priority and severity enums, then routes to a team with SLA hours. If the model call or the parse fails, a whole-word keyword fallback takes over. In the keyword fallback, security tickets always get high priority or above; `security`/`ehr_systems` tickets and every fallback result are flagged for escalation. Code: `src/ticket_classifier.py`. Tests: `tests/test_classifier.py`, `tests/test_api.py`.
- Batch classification: `POST /api/v1/classify/batch` takes up to 50 tickets and returns 400 above that. Code: `src/api/routes.py`. Tests: `tests/test_api.py`.
- Knowledge-base ingest and search over a persistent Chroma collection (cosine similarity, category filter, relevance cut-off, per-category stats). Re-ingesting a source replaces its chunks. Code: `src/knowledge_base.py`. Tests: `tests/test_knowledge_base.py`, `tests/test_api.py`.
- Ingest CLI (`python -m src.ingest <dir>`) in a local mode and an `--api-url` mode that posts through the running API, plus seeding from `KB_SEED_DIR` at startup. Six seed articles live in `data/kb/`. The directory walk never follows symlinks, reads only regular files inside the directory, caps file size, file count and total bytes, and reports relative paths only. Code: `src/ingest.py`, `src/main.py`. Tests: `tests/test_ingest.py`, `tests/test_ingest_safety.py`, `tests/test_api.py`.
- RAG chat: `POST /api/v1/chat` retrieves the top chunks, picks a system prompt by security content or user role from `config/prompts.yml`, runs an LCEL chain and returns the answer, sources, a confidence score and an escalation flag. History is kept per session and capped. Code: `src/ai_engine.py`. Tests: `tests/test_ai_engine.py`, `tests/test_api.py`.
- WebSocket chat at `/api/v1/chat/ws/{user_id}` with session start, typing, answer, escalation, error and session-end messages, a message-size limit, a concurrent-session limit and an idle timeout, served by one shared handler. Code: `src/chat_handler.py`. Tests: `tests/test_websocket.py`.
- Regex PHI redaction of chat messages, ticket text, search queries, escalation reasons and knowledge-base articles (before they are embedded or stored), with a `phi_warning` on chat replies. Input is normalised first (encodings, full-width digits, zero-width characters, dashes), patterns run in linear time, and the filter fails closed: oversized input or an internal error returns a fixed "withheld" text. Code: `src/phi_filter.py`. Tests: `tests/test_phi_filter.py`, `tests/test_security.py`.
- Request guards: the ingest endpoint needs a Bearer ingest key (constant-time check, per-address failure throttle); state-changing requests must be same-origin JSON; bodies are capped before parsing; one concurrency limiter with a per-client quota covers every model and embedding path; chat state is capped per client and overall; validation and server errors never echo input or exception text. Code: `src/security.py`, `src/api/routes.py`, `src/main.py`. Tests: `tests/test_security.py`.
- Pluggable model provider: `LLM_PROVIDER=openai` (langchain-openai against the OpenAI API or any compatible `OPENAI_BASE_URL`) or `LLM_PROVIDER=fake` (fully offline), with two retrieval profiles whose thresholds match the embedder. Code: `src/llm_factory.py`, `src/fakes.py`, `src/config.py`, `config/settings.yml`. Tests: `tests/test_llm_factory.py`, `tests/test_fakes.py`, `tests/test_config.py`.
- Health and stats endpoints (`/health/live`, `/health/ready`, `/api/v1/stats`) from an app factory with a lifespan. Code: `src/main.py`, `src/api/models.py`. Tests: `tests/test_api.py`.
- An OpenAI-compatible stub (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) and a Compose stack that runs the API against it. Every service runs unprivileged (uid 10001, no capabilities, `no-new-privileges`, read-only root filesystem), the API port is bound to 127.0.0.1, the stub sits on an internal network, and keys are files created on first start by `secrets-init`. Code: `stubs/openai_stub.py`, `docker-compose.yml`, `src/keygen.py`. Tests: `tests/test_openai_stub.py`, `tests/test_compose.py`, `tests/test_keygen.py`, `tests/integration/test_stack.py`.

**Not implemented / known limitations**

- OpenAI is replaced by a local stub/offline fakes in all tests; the real API is never called by the test suite. Quality with GPT-4 and the `openai` retrieval-profile thresholds are unverified.
- Offline mode and the stub give extractive answers and lexical (hashing) retrieval, not generated or semantic ones. Offline classification always uses the keyword fallback at confidence 0.5.
- Classification accuracy is not measured; there is no labelled ticket dataset.
- Sessions and conversation history are in memory, lost on restart and limited to one worker. Chroma runs in local mode, so only the API process may write to its directory.
- Only ingest is authenticated, with one shared operator key. Chat, classification and search are open to anyone who can reach the loopback port; there are no user accounts and no general rate limiting beyond the concurrency caps.
- Per-client quotas key on the client address, and every request through the published port arrives from the Docker gateway address, so host-side clients share one quota.
- PHI redaction is regex-based. It finds names only after a label such as "patient" or "name:" and street addresses only in "42 Elm Street" form, and it deliberately leaves IP addresses alone. It is not a compliance control.
- Escalation only flags and informs; no ticket is created in any external system, and there are no ServiceNow, Jira, Slack or PagerDuty integrations.
- Not built: Redis session storage, Prometheus/Grafana monitoring, token streaming, persisted user feedback.
- The base image's OS layer is a September 2024 rebuild of the `python:3.11.9-slim-bookworm` tag.

## Built with

- **Python 3.11.9** — fastapi 0.111.0, uvicorn 0.30.1, websockets 12.0, pydantic 2.7.4, PyYAML 6.0.1, langchain 0.1.20, langchain-core 0.1.52, langchain-community 0.0.38, langchain-openai 0.1.7, langchain-text-splitters 0.0.2, openai 1.35.3, chromadb 0.5.3, numpy 1.26.4, httpx 0.27.0
- **Tests** — pytest 8.2.2, pytest-asyncio 0.23.7, pytest-cov 5.0.0
- The full resolved set is in `requirements-lock.txt` and is installed as pip constraints; `tests/test_smoke.py` checks the installed versions.
- **Base image** — `python:3.11.9-slim-bookworm`, pinned by digest (`sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`) in the `Dockerfile`. That digest is Docker's September 2024 rebuild of the 3.11.9 tag: the interpreter is 3.11.9, but the Debian OS layer is newer than the Python packages.

## Running it

Everything runs in Docker images from the project's era, so nothing needs installing locally beyond Docker.

```bash
docker compose up        # start the API and the OpenAI-compatible stub
curl -s localhost:22700/api/v1/knowledge/search -H 'content-type: application/json' -d '{"query":"vpn keeps disconnecting","min_relevance":0.15}'
```

The API publishes container port 8000 on host port 22700, on 127.0.0.1 only. It seeds the knowledge base from `data/kb/` at startup and reports ready on `/health/ready` once seeding has finished without errors. `make seed` re-seeds a running stack through the HTTP ingest endpoint. No OpenAI key is needed: the stack talks to `openai-stub`.

On first start `secrets-init` writes a random ingest key and a placeholder OpenAI key into the `secrets` volume. To ingest an article from the host:

```bash
KEY=$(docker compose exec -T api cat /run/ai-support/ingest_api_key)
curl -s localhost:22700/api/v1/knowledge/ingest -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"content":"Re-enrol the badge at the security desk.","source":"badge-reader.md","category":"hardware"}'
```

Using the real OpenAI API means writing the key into `/run/ai-support/openai_api_key` in that volume and pointing `OPENAI_BASE_URL` at OpenAI; no test covers that.

## Tests

```bash
make test                # runs the suite inside the period image, with networking disabled
make integration         # starts the Compose stack, runs tests/integration against it, tears it down
```

The unit and API tests cover the classifier, config, PHI regexes, hashing embeddings, the fake chat model, the knowledge base against a real Chroma directory, the engine, every HTTP endpoint, the WebSocket protocol, the request guards and the `openai` provider path against the in-process stub; `make test` also checks the rendered Compose config for loopback-only ports and least privilege. They do not cover the real OpenAI API or answer quality.

## Layout

```
.dockerignore
.gitignore
config/
  prompts.yml
  settings.yml
data/
  kb/
    account-lockout.md
    outlook-mobile-auth.md
    printer-offline.md
    report-phishing.md
    vpn-disconnects.md
    wifi-drops.md
docker-compose.yml
Dockerfile
docs/
  IMPLEMENTATION_PLAN.md
  SPEC.md
Makefile
pytest.ini
README.md
requirements-dev.txt
requirements-lock.txt
requirements.txt
src/
  __init__.py
  ai_engine.py
  api/
    __init__.py
    models.py
    routes.py
  chat_handler.py
  config.py
  fakes.py
  ingest.py
  keygen.py
  knowledge_base.py
  llm_factory.py
  main.py
  phi_filter.py
  security.py
  ticket_classifier.py
stubs/
  __init__.py
  openai_stub.py
tests/
  conftest.py
  helpers.py
  integration/
    __init__.py
    test_stack.py
  test_ai_engine.py
  test_api.py
  test_classifier.py
  test_compose.py
  test_config.py
  test_fakes.py
  test_ingest.py
  test_ingest_safety.py
  test_keygen.py
  test_knowledge_base.py
  test_llm_factory.py
  test_openai_stub.py
  test_phi_filter.py
  test_security.py
  test_smoke.py
  test_websocket.py
```
