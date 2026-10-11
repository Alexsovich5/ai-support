# ai-support: Specification

Target period: April to June 2024. Every dependency in this spec was released on or before 2024-06-30.

## Problem

An IT help desk gets a steady stream of repetitive tickets: VPN drops, account lockouts, printer faults, mail sign-in failures. Staff have to triage each one by hand (what category, how urgent, which team) and answer the same questions over and over, even though the answers already sit in internal how-to articles.

This project is a small FastAPI service that:

1. classifies a ticket into category, priority and severity, then routes it to a team with SLA targets,
2. answers support questions in chat, grounded in a searchable knowledge base (retrieval-augmented generation with LangChain and Chroma), and
3. lets operators add articles to that knowledge base and search it.

The deployment context is a healthcare IT desk, so the system also strips obvious patient identifiers out of chat messages and ticket text before they are sent to a language model.

## Scope criterion

A feature is **in scope** when it serves one of the README's features (ticket classification, automated responses from an AI chatbot, knowledge-base search, a FastAPI service) or its healthcare-desk concept, **and** the existing code already implements it (so the work is repair, not invention). Everything else in the README or the existing code is out of scope, with the reason given in the table below.

## In scope

Each item names the README feature or concept it serves.

1. **Ticket classification (LLM + fallback).** Serves "ticket classification". `POST /api/v1/classify` sends the ticket to a chat model and parses its JSON reply into category, priority and severity. If the model call or the parse fails, a deterministic keyword classifier takes over. The result also carries the routed team, SLA response/resolution hours, tags, reasoning and an escalation flag. In the fallback, a ticket that resolves to `security` always gets priority `high` and severity `sev2` (or `critical`/`sev1` when a critical keyword also matches), matching the prompt rule "security incidents are always HIGH priority or above". Keywords match as whole words with an optional inflection suffix (`\b<kw>(?:s|es|ing|ed)?\b`), so "printing" matches `print` and "phones" matches `phone`, while "download" does not match `down`.
2. **Batch classification.** Serves "ticket classification". The existing routes already implement it as a thin loop over item 1. `POST /api/v1/classify/batch` accepts up to 50 tickets and returns HTTP 400 above that.
3. **Knowledge-base ingestion.** Serves "knowledge-base search". `POST /api/v1/knowledge/ingest` splits a document into overlapping chunks (LangChain `RecursiveCharacterTextSplitter`), embeds them and upserts them into a persistent Chroma collection. `doc_id` is derived from `source` alone. Before upserting, the service deletes the existing chunks `where={"source": source}`, so re-ingesting a changed or shortened article replaces it with no duplicates or stale chunks. Only the process that serves the API writes to its Chroma directory. Seeding goes through that process in one of two ways: at startup, when `KB_SEED_DIR` is set, the lifespan ingests every file in the directory before `/health/ready` turns 200; while the API is running, the CLI's `--api-url` mode POSTs each file to `/knowledge/ingest`. The CLI's local mode (`python -m src.ingest <dir>`, which opens Chroma directly) is only for a directory that no running API is using.
4. **Knowledge-base search.** Serves "knowledge-base search". `POST /api/v1/knowledge/search` runs a cosine-similarity search with an optional category filter and a minimum-relevance cut-off. The cut-off defaults to the active retrieval profile's `min_relevance_score` (see item 10). `GET /api/v1/knowledge/stats` returns chunk counts per category.
5. **RAG chat answers (automated responses).** Serves "automated responses". `POST /api/v1/chat` retrieves the top-k chunks and builds a prompt from `config/prompts.yml`. It calls the chat model through an LCEL chain and returns the answer, the sources it used, a confidence score and an escalation flag. The system prompt is chosen as follows:
   - If the message matches the security keyword set (`phishing`, `malware`, `ransomware`, `breach`, `suspicious`, `virus`, `hacked`, with the same inflection rule as item 1), the service uses `security_incident`. This check comes first and ignores the role.
   - Otherwise it maps `user_role`, case-insensitively: `physician`, `nurse`, `clinician`, `clinical` → `clinical_staff`; `it`, `engineer`, `admin`, `technical` → `technical_staff`; any other role or no role → `default`.
6. **Multi-turn context.** Serves "automated responses" (the chatbot follows up on earlier turns). The service keeps per-session conversation history in memory, capped at `chat.max_context_messages`.
7. **Real-time chat over WebSocket.** Serves the README's "24/7 AI chatbot", which the existing `chat_handler.py` implements as a WebSocket protocol. `WS /api/v1/chat/ws/{user_id}?department=&role=` carries these message types: session start, typing start/stop, AI response, escalation, error, and system notices. The `role` query parameter feeds the same role mapping as item 5. The endpoint enforces three limits:
   - a message-size limit (`chat.max_message_length`), where an oversize message gets an `error` reply,
   - a concurrent-session limit (`chat.max_concurrent_sessions`), where the connection gets an `error` and close code 1013, and
   - an idle timeout (`chat.session_timeout_seconds`), where the server sends `session_end` and closes.

   The concurrent-session limit is kept because the existing handler implements it. It only means something with one shared handler, which T11 provides. The lifespan starts the handler's stale-session cleanup task and cancels it on shutdown.
8. **Escalation recommendation.** Chat recommends escalation when retrieval confidence falls below the active profile's `confidence_threshold`. The classifier flags escalation for `auto_escalate_categories` (security, EHR systems) and for every fallback classification. A user can also escalate manually over the WebSocket. The WebSocket `escalation` message text is `prompts.yml` `escalation.threshold_warning`, reworded so that it tells the user to contact the service desk and no longer offers to create a ticket. No ticket is ever created, and the reply has no `ticket_created` field.
9. **PHI redaction.** Serves the healthcare-desk concept. It is in the existing code as `phi_detection`. A regex-based filter redacts medical-record numbers, SSN-style IDs, dates of birth, phone numbers and email addresses from **chat messages (REST and WebSocket) and from ticket title/description/context before classification**. Chat responses then carry a `phi_warning`. Knowledge-base articles are **not** redacted. They are operator-authored IT how-tos and must not contain patient data, which is listed as a limitation.
10. **Pluggable model provider and retrieval profiles.** `LLM_PROVIDER=openai` uses `langchain-openai` (`ChatOpenAI` and `OpenAIEmbeddings`) against the OpenAI API, or against any compatible `OPENAI_BASE_URL`. `OpenAIEmbeddings` is always built with `check_embedding_ctx_length=False`. That makes it send raw strings, and it never loads tiktoken, which would otherwise download `cl100k_base` from the internet on first use. `LLM_PROVIDER=fake` runs fully offline with a deterministic extractive chat model and hashing embeddings (see Simulated/mocked integrations). The relevance thresholds depend on the embedder, so the settings carry two retrieval profiles:
    - `openai` is for real OpenAI embeddings: min relevance 0.3, escalation threshold 0.4.
    - `hashing` is for `HashingEmbeddings`, whether used in-process or through the stub: min relevance 0.15, escalation threshold 0.2, `chunk_size` 400, `chunk_overlap` 80.

    `RETRIEVAL_PROFILE` selects the profile. It defaults to `hashing` when `LLM_PROVIDER=fake` and to `openai` otherwise. The compose stack sets `RETRIEVAL_PROFILE=hashing` because its embeddings come from the stub.
11. **Health and stats endpoints.** Serves "FastAPI service". `GET /health/live` and `GET /health/ready` are the health endpoints. Ready means the knowledge base is initialised and any `KB_SEED_DIR` seed has finished. `GET /api/v1/stats` returns engine and classifier counters.
12. **Containerised stack.** `docker compose up` starts the API and an OpenAI-compatible stub, nothing else. The API seeds `data/kb/` at startup through `KB_SEED_DIR`, and `make seed` re-seeds a running stack through the HTTP ingest endpoint (`python -m src.ingest data/kb --api-url http://localhost:8000`).

## Out of scope

| Item (from original README or existing code) | Reason |
|---|---|
| "70% faster first response", "85% classification accuracy", "24/7 availability" | Business metrics with no measurement behind them. No labelled ticket dataset exists, so accuracy stays unmeasured. |
| Calling the real OpenAI GPT-4 API in tests/CI | Needs a paid key and gives non-deterministic output. The `openai` provider code path is tested against a local stub instead. Using a real key is supported at runtime, but no test covers it. |
| Redis session store (`settings.yml: redis`) | Exists only as config, with no working code. A single-process in-memory store is enough for a solo demo. |
| Prometheus / Grafana monitoring (compose services) | Not a README feature, and there is no working code for it. The referenced `monitoring/prometheus.yml` never existed. |
| JWT auth, rate limiting, CORS/allowed-host lists | Not a README feature, and it exists only as config. The allowed-host lists named an employer domain. Listed as a known limitation. |
| ServiceNow / Jira / Slack / PagerDuty integrations | Not README features, they exist only as config, and they are proprietary external systems. Escalation stays a flag and a message. |
| Automatic ticket creation or human-agent handoff on escalation | No ticketing backend is in scope. The escalation message states only what happens. |
| Token streaming (`streaming_enabled`) and multi-worker deployment | Only a config flag exists. In-memory sessions also assume one worker. |
| Query-refinement and ticket-summary prompts (`prompts.yml`) | Unused by any code, and each would add an extra LLM round-trip. They are removed from config. |
| Persisting user feedback | The WebSocket feedback message is acknowledged and logged only. |
| HIPAA compliance | Regex redaction is best effort, not a compliance control. The docs, prompts and product strings make no compliance claim. |
| Employer, role, "developed during", "Timeline" headers, "Status: Complete" claims | Fabricated. They are removed from every tracked file. |

## Architecture

```
                 +--------------------------- api container (python:3.11.9-slim-bookworm) ---------------------------+
 HTTP / WS       |                                                                                                   |
 client  ------> |  src/main.py  create_app() + lifespan (KB init, KB_SEED_DIR seed, cleanup task start/stop)       |
                 |       |                                                                                           |
                 |  src/api/routes.py ---------------------------------------------------------------+               |
                 |     | /chat, /chat/ws        | /classify(/batch)          | /knowledge/*          |               |
                 |     v                        v                            v                       |               |
                 |  src/chat_handler.py    src/ticket_classifier.py    src/knowledge_base.py         |               |
                 |     |                        |  phi_filter -> LLM JSON   |  splitter -> embed ->   |               |
                 |     v                        |  -> parse, else fallback  |  Chroma PersistentClient|               |
                 |  src/ai_engine.py -----------+---------------------------+  (./data/chroma volume) |               |
                 |   phi_filter -> retrieve -> prompt(role) -> chat model -> answer + confidence       |               |
                 |     |                                                                                           |
                 |  src/llm_factory.py  provider = fake | openai                                                   |
                 |     | fake: src/fakes.py (ExtractiveFakeChatModel, HashingEmbeddings)                           |
                 +-----|---------------------------------------------------------------------------------------------+
                       | openai: langchain-openai -> OPENAI_BASE_URL
                       v
          +------------------------------+        (or the real api.openai.com with a user-supplied key)
          | openai-stub container        |
          | stubs/openai_stub.py         |
          | /v1/chat/completions         |
          | /v1/embeddings  /v1/models   |
          +------------------------------+
```

Chat data flow: the request is validated (pydantic) and PHI is redacted. The engine embeds the query and searches Chroma (top k, filtered by the active profile's min relevance). Confidence is the highest relevance score among the results, or 0 when nothing is retrieved. The prompt combines the system prompt chosen by the item 5 rules, the retrieved context and the last N turns. The chat model produces the answer, the session history is updated, and the service returns `EngineResponse`.

Classification data flow: the title, description and additional context are PHI-redacted. They are rendered into `prompts.yml` `classification.user_template` and sent, with `classification.system_prompt`, to the chat model, which is expected to return JSON. The JSON is validated against the enums (category, priority and severity) and mapped to a team and SLAs through `ROUTING_MAP` and the SLA tables. On any error, the keyword fallback runs with confidence 0.5 and always sets escalation.

## Data model & interfaces

### Python types (kept from existing code)

- `TicketCategory`: network, email, vpn, ehr_systems, active_directory, printing, hardware, software, security, cloud_services, mobile_devices, telephony
- `TicketPriority`: critical, high, medium, low. `TicketSeverity`: sev1 to sev4
- `SLA_RESPONSE_HOURS` = {critical 0.25, high 1, medium 4, low 8}. `SLA_RESOLUTION_HOURS` = {4, 8, 24, 72}
- `ClassificationResult` (dataclass) with `to_dict()`
- `Document(content, source, category, title="", metadata={}, doc_id=sha256(source)[:16])`
- `EngineResponse(message: str, confidence: float, source_documents: list[str], escalation_recommended: bool, response_time_ms: float, phi_detected: bool)` (new)
- `TicketClassifier(llm=None, settings=None, model_name="gpt-4-0125-preview", temperature=0.1)`. `ChatOpenAI` stays imported at module level (so the existing `patch("src.ticket_classifier.ChatOpenAI")` fixture keeps working) and is built lazily, only when `llm` is None.

### Chroma chunk record

`id = f"{doc_id}_{i}"`, `document = chunk text`, `embedding`, `metadata = {source, category, title, chunk_index, total_chunks, doc_id, created_at}`. The collection is created with `{"hnsw:space": "cosine"}`, and relevance is computed as `1 - distance`. Ingest runs `collection.delete(where={"source": source})` and then `upsert`.

### HTTP API (prefix `/api/v1` except health)

| Method | Path | Request model | Response |
|---|---|---|---|
| GET | `/health/live` | none | `{"status":"ok"}` |
| GET | `/health/ready` | none | 200 `{"status":"ready","kb_chunks":N}` / 503 |
| POST | `/chat` | `ChatRequest{message, session_id, department?, user_role?}` | `ChatResponse{message, session_id, confidence, source_documents, escalation_recommended, response_time_ms, phi_warning?}` |
| WS | `/chat/ws/{user_id}?department=&role=` | JSON `{"type":"user_message"\|"feedback"\|"escalate"\|"end_session","content":...}` | JSON `{"type": session_start\|typing_start\|typing_stop\|ai_response\|escalation\|error\|system\|session_end, "timestamp":..., ...}` |
| POST | `/classify` | `ClassifyRequest{title, description, department?, reporter_role?}` | `ClassifyResponse` |
| POST | `/classify/batch` | `list[ClassifyRequest]` (at most 50) | `{"total", "classifications":[...], "timestamp"}` |
| POST | `/knowledge/search` | `KnowledgeSearchRequest{query, max_results=5, category?, min_relevance: float \| None = None}` (None means the profile default) | `KnowledgeSearchResponse{results, total_results, query}` |
| POST | `/knowledge/ingest` | `IngestRequest{content, source, category, title=""}` | `IngestResponse{doc_id, chunks_created, total_documents}` |
| GET | `/knowledge/stats` | none | `{"total_chunks", "categories":{...}, "collection_name", "persist_directory"}` |
| GET | `/stats` | none | `{"engine_stats":{...}, "classifier_stats":{...}, "timestamp"}` |

### CLI

```
python -m src.ingest <directory> [--category <name>] [--reset]          # local mode: opens Chroma directly; only when no API is running on that directory
python -m src.ingest <directory> [--category <name>] --api-url <url>    # POSTs each file to <url>/api/v1/knowledge/ingest; --reset is rejected in this mode
```

The command ingests `*.md` and `*.txt` files. A front-matter line `category: <name>` at the top of a file overrides `--category`. The command prints a JSON summary `{"total_files", "total_chunks", "errors"}` and exits 1 if any file failed. The lifespan's `KB_SEED_DIR` seeding reuses the same directory-ingest function in-process.

### Config: `config/settings.yml` (path overridable with `APP_CONFIG`)

```yaml
app:            {name: "AI IT Support", log_level: INFO}
llm:            {provider: fake, chat_model: gpt-4-0125-preview, classification_model: gpt-4-0125-preview,
                 temperature: 0.3, classification_temperature: 0.1, max_tokens: 1024,
                 request_timeout: 30, embedding_model: text-embedding-3-small}
knowledge_base: {persist_directory: ./data/chroma, collection_name: it_support_kb, max_search_results: 5}
retrieval_profiles:
  openai:  {chunk_size: 1000, chunk_overlap: 200, min_relevance_score: 0.3,  confidence_threshold: 0.4}
  hashing: {chunk_size: 400,  chunk_overlap: 80,  min_relevance_score: 0.15, confidence_threshold: 0.2}
chat:           {max_message_length: 4096, max_context_messages: 10,
                 session_timeout_seconds: 1800, max_concurrent_sessions: 200}
security:       {phi_detection_enabled: true}
escalation:     {auto_escalate_categories: [security, ehr_systems]}
```

Environment variables override the file: `LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CHROMA_PERSIST_DIR`, `RETRIEVAL_PROFILE`, `KB_SEED_DIR` and `LOG_LEVEL`. The loaded settings are a frozen pydantic `Settings` model in `src/config.py`. `settings.retrieval` returns the active profile. Tests change any other value with `settings.model_copy(update={"chat": settings.chat.model_copy(update={...})})` and pass the result to `create_app`.

How the hashing profile was calibrated: I prototyped the `HashingEmbeddings` design below (stop words removed, sublinear tf, blake2b hashing, 400/80 chunks) against a 694-character VPN article, a printer article and an account-lockout article. `"vpn keeps disconnecting"` scored 0.344 against the top VPN chunk. `"My VPN connection keeps dropping"` scored 0.374. Unrelated queries such as `"how do I bake sourdough bread"` scored 0.0. The 0.15/0.2 thresholds sit well between those. T5 re-checks this against the real seed articles.

### Config: `config/prompts.yml`

- **Kept:**
  - `system_prompts.{default, clinical_staff, technical_staff, security_incident}`. The default prompt's "maintaining strict HIPAA compliance" becomes "never repeat patient identifiers".
  - `classification.system_prompt`. Its rules are aligned with the in-code rules: EHR issues affecting patient care are CRITICAL, and security incidents are HIGH or above. The "multiple departments escalate by one level" and "after-hours ... CRITICAL" rules are dropped because nothing can evaluate them.
  - `phi_detection.warning_message`, reworded to "I've filtered what looked like patient identifiers from your message ...", with no compliance claim.
  - `escalation.threshold_warning`, reworded as in item 8.
- **Added:**
  - `chat.user_template`, with the `{context}`, `{history}` and `{question}` placeholders.
  - `classification.user_template`, rendered with `str.format`. It contains `{title}`, `{description}` and `{additional_context}`, the fixed marker line `### TICKET_CLASSIFICATION_JSON ###`, and this exact output contract, with the literal braces doubled for `str.format`:
  ```
  Respond with JSON only:
  {{"category": "<network|email|vpn|ehr_systems|active_directory|printing|hardware|software|security|cloud_services|mobile_devices|telephony>",
    "priority": "<critical|high|medium|low>", "severity": "<sev1|sev2|sev3|sev4>",
    "confidence": <0.0-1.0>, "suggested_tags": ["..."], "reasoning": "<short>", "requires_escalation": <true|false>}}
  ```
  The in-code `CLASSIFICATION_PROMPT` constant is deleted, so `prompts.yml` is the only prompt source.
- **Removed:** `knowledge_search.query_refinement`, `escalation.ticket_summary` and the "Timeline" header.

## Stack & pinned versions

The pins were checked with `tools/check_period.py`: 142 entries checked (14 direct runtime pins + 3 dev pins + 125 lock pins), 0 problems. The lock was resolved with `uv pip compile requirements.txt requirements-dev.txt --exclude-newer 2024-06-30T23:59:59Z --python-version 3.11`. I also installed it into a clean Python 3.11 venv and smoke-tested it: a Chroma persistent client, a cosine query, the fake chat model, and ChatOpenAI construction.

| Component | Version | Released | Why it was the popular choice then |
|---|---|---|---|
| Python | 3.11.9 | 2024-04-02 | 3.11 was the mainstream production Python in 2024, and 3.12 still had gaps in ML wheels. |
| fastapi | 0.111.0 | 2024-05-03 | The default async Python API framework for LLM backends. |
| uvicorn | 0.30.1 | 2024-06-02 | The standard ASGI server for FastAPI. |
| websockets | 12.0 | 2023-10-21 | Uvicorn's WebSocket protocol implementation. |
| pydantic | 2.7.4 | 2024-06-12 | Pydantic v2 was FastAPI's default by 2024. |
| PyYAML | 6.0.1 | 2023-07-17 | The de facto YAML loader. |
| langchain | 0.1.20 | 2024-05-10 | The last 0.1 line (0.2 shipped mid-May 2024). Matches the original stack, and `langchain.text_splitter` shims still exist. |
| langchain-core | 0.1.52 | 2024-05-06 | LCEL runnables, the `BaseChatModel` and `Embeddings` interfaces, and `FakeListChatModel` (`langchain_core.language_models.fake_chat_models`), which the tests use. |
| langchain-community | 0.0.38 | 2024-05-08 | Not imported directly. It is a transitive requirement of langchain 0.1.20, pinned so the direct set is explicit. |
| langchain-openai | 0.1.7 | 2024-05-15 | The partner package for `ChatOpenAI` and `OpenAIEmbeddings` (replacing `langchain.chat_models`). Supports separate `http_client`/`http_async_client` and `check_embedding_ctx_length`. |
| langchain-text-splitters | 0.0.2 | 2024-05-16 | `RecursiveCharacterTextSplitter`. |
| openai | 1.35.3 | 2024-06-20 | The v1 SDK used under langchain-openai. |
| chromadb | 0.5.3 | 2024-06-18 | The most-used local vector store for LangChain RAG demos. Uses the `PersistentClient` API. Local mode is single-process, which is why only the API process writes. |
| numpy | 1.26.4 | 2024-02-05 | Pinned below 2.0 (2024-06-16) because chromadb 0.5.3 breaks on numpy 2. |
| httpx | 0.27.0 | 2024-02-21 | Needed by FastAPI `TestClient` and the openai SDK. `ASGITransport` runs the async stub test in-process. |
| pytest | 8.2.2 | 2024-06-04 | The standard test runner. |
| pytest-asyncio | 0.23.7 | 2024-05-19 | Runs the async tests that the existing suite already uses. |
| pytest-cov | 5.0.0 | 2024-03-24 | Coverage reports in `make test`. |

The files are `requirements.txt` (runtime direct pins), `requirements-dev.txt` (test pins) and `requirements-lock.txt` (the full resolved set of 125 pins, used as pip constraints `-c`).

## Docker images

| Image | Tag / digest | Verified (Docker Hub API) | Use |
|---|---|---|---|
| python (official) | `3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317` | HTTP 200, tag last pushed 2024-09-05 | Base for the `base`, `dev` (tests) and `runtime` stages, and for the openai-stub service. The Dockerfile pins this digest. |
| python (official) | `3.11.9-bookworm@sha256:c95170ff7d59de63e9445d3d503644a635c1b8cb7f4fa99f19a1c76da92a849a` | HTTP 200, tag last pushed 2024-09-06 | Fallback only, if a lock entry ever has to build from sdist and needs a compiler (T1 fails fast if the slim image cannot install the lock) |

The Python interpreter (3.11.9, released 2024-04-02) is the period release. The tag's pullable manifest, however, is Docker's final rebuild of the 3.11.9 tag in September 2024, which refreshed the Debian bookworm OS layer. Earlier in-period manifests of this tag are no longer served under the tag. Every Python package comes from the period lock, so only the OS layer postdates the period.

No other images are used. The original compose file's `redis:7-alpine`, `prom/prometheus:v2.49.0` and `grafana/grafana:10.3.1` are dropped (see Out of scope).

## Simulated/mocked integrations

| Real system | Replacement | Where |
|---|---|---|
| OpenAI Chat Completions (GPT-4) | **`openai-stub`**: a FastAPI app implementing `/v1/chat/completions`, `/v1/embeddings` and `/v1/models`. When the prompt contains the marker `### TICKET_CLASSIFICATION_JSON ###`, chat returns classifier JSON derived from the keyword map. Otherwise it returns an extractive answer from the prompt's context block. Embeddings accept `input` as `str` or `list[str]` and use the same `HashingEmbeddings`. Token-ID arrays get a 400, which catches a missing `check_embedding_ctx_length=False`. | `stubs/openai_stub.py`, compose service `openai-stub` |
| OpenAI chat model (offline mode) | **`ExtractiveFakeChatModel`** (a `langchain_core` `SimpleChatModel`): it answers deterministically from the retrieved context. The classifier gets no JSON back and falls through to the keyword fallback. | `src/fakes.py` |
| OpenAI embeddings (offline mode and stub) | **`HashingEmbeddings`**: lowercase `[a-z0-9]+` tokens with a fixed stop-word list removed, weighted by sublinear tf (`1 + ln tf`). Each token is hashed with `hashlib.blake2b(token, digest_size=8)`: the index is the value mod 512 and the sign is the top bit. The vector is L2-normalised. Python's `hash()` is never used, so vectors are identical across processes whatever `PYTHONHASHSEED` is. It captures lexical overlap, not semantics. | `src/fakes.py` |
| LLM in unit tests | `langchain_core` `FakeListChatModel` with canned JSON. `RecordingFakeChatModel(FakeListChatModel)` in `tests/helpers.py` stores the messages it receives so tests can inspect prompts. The existing classifier tests keep using `AsyncMock`. | `tests/` |

The final README must say that no test calls the real OpenAI API and that offline answers are extractive, not generated.

## Existing code inventory

`git ls-files` returns 14 files: keep 0, refactor 14, delete 0. Every tracked file that carries the fabricated "Timeline: April 2024 - June 2024" header loses it in T1: `config/prompts.yml`, `config/settings.yml`, `docker-compose.yml`, `src/api/models.py`, `src/api/routes.py`, `src/chat_handler.py`, `src/knowledge_base.py`, `src/ticket_classifier.py` and `tests/test_classifier.py`.

| File | Decision | Reason |
|---|---|---|
| `Dockerfile` | refactor | Pin `python:3.11.9-slim-bookworm@sha256:8fb0...c317`, add multi-stage `base`/`dev`/`runtime`, install with lock constraints, run `uvicorn src.main:app`. |
| `README.md` | refactor | Regenerate from `tools/readme_template.md` in the last task, with honest status and the tree taken from `git ls-files`. |
| `config/prompts.yml` | refactor | Keep the role prompts and the classification/PHI/threshold-warning prompts, reworded to remove HIPAA-compliance and ticket-creation claims. Remove the "Timeline" header and the unused refinement/summary prompts. Add `chat.user_template` and `classification.user_template`. |
| `config/settings.yml` | refactor | Remove the employer domains (5 occurrences), the Redis/Prometheus/integrations/JWT blocks and the fake version. Add `llm.provider` and `retrieval_profiles`. Match the schema above. |
| `docker-compose.yml` | refactor | Reduce to `api` plus `openai-stub`. Drop the obsolete `version:`, the nonexistent Prometheus config, and the curl healthcheck (slim has no curl). |
| `requirements.txt` | refactor | Replace the `>=` specs with the exact period pins. Add `langchain-openai`, `langchain-community`, `langchain-text-splitters`, `websockets`, `httpx` and `numpy==1.26.4` (below 2.0, which chromadb 0.5.3 requires). |
| `src/ai_engine.py` | refactor | Replace the deprecated `langchain.chat_models` and `RetrievalQA` imports. Implement `process_message`, `get_stats` and `EngineResponse`, which `routes.py` and `chat_handler.py` already call but which do not exist yet. |
| `src/api/models.py` | refactor | The pydantic v2 request/response models are kept. The "Timeline" docstring line is removed (T1), and `KnowledgeSearchRequest.min_relevance` defaults to None, meaning the profile value (T10). |
| `src/api/routes.py` | refactor | Take dependencies from `app.state` instead of importing `app_state` from `src.main`. Share one classifier and one chat handler. Add classifier stats to `/stats`. Pass the `role` query parameter to the WS handler. |
| `src/chat_handler.py` | refactor | Keep the protocol and lifecycle. Make it a single app-scoped instance so the session limits hold. Read the limits from settings. Replace `del` with `pop`. Fix the hardcoded "ext. 4500" and "5-10 minutes" texts. Remove the "HIPAA compliance" and "HIPAA-compliant logging" wording and the `ticket_created: True` field. Take the role from the connection instead of hardcoding `"staff"`. |
| `src/knowledge_base.py` | refactor | Keep the ingest/search/delete/stats logic. Replace the removed `Settings(chroma_db_impl="duckdb+parquet")` (it raises `ValueError` on chromadb 0.4 and later) with `PersistentClient`. Inject the embeddings. Import from `langchain_text_splitters`. Derive `doc_id` from `source` and delete by source before upserting. |
| `src/main.py` | refactor | Replace the broken `from api.routes` import with a `create_app(settings)` factory and a lifespan that builds the KB, engine, classifier and handler, seeds `KB_SEED_DIR`, and starts and cancels the cleanup task. Add health endpoints. |
| `src/ticket_classifier.py` | refactor | Keep the enums, SLA tables, routing, parser and fallback. Accept an injected `llm` (signature above). Read the prompt from `prompts.yml`. Redact PHI from the input. In the fallback, check the security keywords before email and give security HIGH/sev2 (this fixes the failing test). Validate severity. Use whole-word matching with inflection suffixes. Apply `auto_escalate_categories`. |
| `tests/test_classifier.py` | refactor | 23 real unit tests, 22 of which pass today. The one failure (`test_security_high_priority`, which asserts category SECURITY and priority HIGH) is a genuine bug fixed in T2. The header line is removed, a temporary `xfail` is added in T1 and removed in T2, and new tests are added in T2 and T9. |

## Test strategy

- **Runner.** `make test` builds the `dev` stage of the pinned `python:3.11.9-slim-bookworm` image and runs `pytest` with `docker run --rm --network none -e LLM_PROVIDER=fake ...`, so any network access fails the run. `pytest.ini` sets `pythonpath = .`, `asyncio_mode = strict` and coverage over `src`. No API key is needed.
- **Unit tests.**
  - Classifier: the existing suite plus tests for keyword precedence, security priority, inflection matching, severity validation, auto-escalation, rendered-prompt keys and PHI-free prompts.
  - Config: loading, env overrides, profile selection and the presence of the prompt templates.
  - PHI regexes: positive and negative cases.
  - Hashing embeddings: determinism within one process and across processes with different `PYTHONHASHSEED`, normalisation, and lexical similarity ordering.
  - The fake chat model.
  - Knowledge base, against a real Chroma `PersistentClient` in `tmp_path` with a unique collection name per test (`EphemeralClient` is not used, because its instances share state within a process): chunking, idempotent re-ingest, re-ingest of a shortened source dropping the chunk count, category filter, relevance cut-off, the VPN query at the default `hashing` profile values, delete, stats and directory ingest.
  - AI engine: role and security-content prompt selection, history capping, the confidence/escalation threshold and PHI flagging.
- **API tests.** Each test uses `with TestClient(create_app(settings)) as client` so the lifespan runs, with fake providers and a `tmp_path` Chroma directory. They exercise every endpoint, the 422 validation paths, the batch limit returning 400, startup seeding, and the WebSocket flow: invalid JSON, oversize message, idle timeout, session limit, escalation and end of session.
- **Provider path test.** `ChatOpenAI` and `OpenAIEmbeddings` run against the stub app in-process. Sync calls use `http_client=TestClient(stub_app)` and async calls use `http_async_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_app), base_url="http://stub/v1")`. `llm_factory` accepts both clients. This proves the `openai` provider wiring (chat, embeddings with `check_embedding_ctx_length=False`, and a non-fallback classification) without a key or network.
- **Integration tests.** `make integration` runs `docker compose up --wait` with the API in `LLM_PROVIDER=openai` mode pointed at `openai-stub`. The API seeds `data/kb` itself through `KB_SEED_DIR` before it reports ready. The target then runs `tests/integration/` from a container on the compose network (HTTP plus WebSocket) and always tears the stack down. These tests are skipped when `STACK_URL` is unset, so `make test` stays hermetic.

## Known limitations that will remain

- Classification accuracy is not measured, because there is no labelled dataset.
- Offline mode (`fake`) and the stub give extractive answers and lexical retrieval, not generated or semantic ones. Classification in offline mode always uses the keyword fallback, at confidence 0.5.
- Nothing tests the real OpenAI API. Quality with GPT-4, and the `openai` retrieval-profile thresholds, are unverified.
- Sessions and conversation history live in memory: they are lost on restart and limited to a single worker. Chroma runs in local mode, so only the API process may write to its directory.
- The API has no authentication or rate limiting.
- PHI redaction is regex-based and will miss free-text identifiers such as patient names. Knowledge-base articles are not redacted. It is not a HIPAA control.
- Escalation does not create a ticket anywhere. It only flags and informs.
- The base image's OS layer is a September 2024 rebuild of the 3.11.9 tag (see Docker images).
- The chromadb 0.5.x install pulls in heavy transitive packages (onnxruntime, kubernetes client, opentelemetry), so the image is about 1.5 GB.
