# ai-support: Implementation Plan

There are 14 tasks, one commit each and in dependency order. Every task leaves `make test` green. Acceptance commands run from the repo root on a host with Docker and the Compose v2 plugin.

Conventions:
- `IMAGE ?= ai-support:dev`.
- `make test` means `docker build --target dev -t $(IMAGE) .` followed by `docker run --rm --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False $(IMAGE) pytest`. `--network none` makes any network access (a tiktoken download, a real OpenAI call) fail the run.
- `GUARD` is the repo-wide fabrication check, and `GUARD_STRINGS` checks the product strings (it passes only from T11 on, once `chat_handler.py` is cleaned):
  ```
  GUARD:         ! git grep -niE 'timeline: april|ael-dubai|dubai|acoria|etech|zambaiti|99\.9|uptime|mttr' -- . ':!docs'
  GUARD_STRINGS: ! git grep -niE 'hipaa[ -]complian|ticket_created' -- src config
  ```
- Write the tests first, watch them fail, then implement.

---

## T1: Scaffold pinned Docker image, Makefile and pytest harness, strip fabricated headers

- **Goal.** `make test` runs the existing suite inside the digest-pinned `python:3.11.9-slim-bookworm` with exact period pins. No tracked file still carries the fabricated "Timeline" header.
- **Files.**
  - create: `Makefile` (targets `build`, `test`, `shell`, `clean`), `requirements-dev.txt`, `requirements-lock.txt` (the 125-line lock from `uv pip compile requirements.txt requirements-dev.txt --python-version 3.11 --exclude-newer 2024-06-30T23:59:59Z`), `pytest.ini`, `.dockerignore`, `src/__init__.py`, `src/api/__init__.py`, `tests/test_smoke.py`.
  - modify:
    - `Dockerfile`: `FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`, stages `base`/`dev`/`runtime`, `pip install -r requirements.txt -c requirements-lock.txt`, `ENV ANONYMIZED_TELEMETRY=False`.
    - `requirements.txt`: the exact pins from SPEC.
    - `tests/test_classifier.py`: mark `test_security_high_priority` `@pytest.mark.xfail(strict=True, reason="fallback matches 'email' before security keywords and never sets HIGH for security")`.
    - Remove the "Timeline: April 2024 - June 2024" line from `config/prompts.yml`, `config/settings.yml`, `docker-compose.yml`, `src/api/models.py`, `src/api/routes.py`, `src/chat_handler.py`, `src/knowledge_base.py`, `src/ticket_classifier.py` and `tests/test_classifier.py`.
- **Tests first.** `tests/test_smoke.py` has one test function, `test_pinned_versions`, which asserts `fastapi.__version__ == "0.111.0"`, `chromadb.__version__ == "0.5.3"`, `langchain.__version__ == "0.1.20"` and `numpy.__version__ == "1.26.4"`.
- **Acceptance.** `make test` gives exactly 23 passed (22 existing plus 1 smoke) and 1 xfailed. `! git grep -ni 'timeline: april' -- . ':!docs'` also passes.
- **Commit.**
  ```
  Add pinned Docker test image, Makefile and pytest harness

  Pins every dependency to exact versions with a full constraints lock,
  runs the existing classifier suite inside python:3.11.9-slim-bookworm
  and removes stale timeline headers from source and config files.
  ```

## T2: Fix fallback keyword precedence, security priority and response validation

- **Goal.** The keyword fallback:
  - checks security before email,
  - sets priority HIGH and severity sev2 whenever the category resolves to SECURITY (CRITICAL/sev1 if a critical keyword also matches), and
  - matches keywords as `\b<kw>(?:s|es|ing|ed)?\b`, so "download" no longer counts as "down" while "printing" and "phones" still match.

  `_parse_response` validates severity. Timing never goes negative or explodes when `start_time=0`.
- **Files.** modify: `src/ticket_classifier.py`, `tests/test_classifier.py` (remove the xfail and add tests).
- **Tests first.**
  - `test_phishing_email_is_security`
  - `test_security_fallback_priority_is_high` (the existing security ticket gives `priority == HIGH` and `severity == SEV2`)
  - `test_download_does_not_trigger_critical`
  - `test_printing_matches_print` ("Printing fails on 3rd floor" → printing)
  - `test_phones_match_telephony` ("phones not ringing" → telephony)
  - `test_invalid_severity_raises_error`
  - `test_fallback_time_is_small_when_start_is_zero` (the timer is measured inside the method)
- **Acceptance.** `make test`, with 0 xfailed.
- **Commit.**
  ```
  Fix fallback classifier keyword order and security priority

  Security keywords are checked before email and always yield high
  priority, matching uses word boundaries with inflections, and the
  parser rejects unknown severities.
  ```

## T3: Add settings loader and clean configuration files

- **Goal.** Add `src/config.py`: a frozen pydantic `Settings` model loaded from YAML, with the env overrides and retrieval profiles in the SPEC, plus a prompt loader.
- **Files.**
  - create: `src/config.py`, `tests/test_config.py`.
  - modify `config/settings.yml`: match the SPEC schema, remove the 5 employer-domain entries, and remove the Redis, Prometheus, integrations and JWT blocks.
  - modify `config/prompts.yml`:
    - remove `query_refinement` and `ticket_summary`;
    - reword the default prompt's "maintaining strict HIPAA compliance", `phi_detection.warning_message` and `escalation.threshold_warning` as SPEC describes (no compliance claim, no ticket-creation offer);
    - align the `classification.system_prompt` rules;
    - add `chat.user_template` and `classification.user_template`.
- **Tests first.**
  - The defaults load from the repo config.
  - `LLM_PROVIDER=openai`, `CHROMA_PERSIST_DIR`, `RETRIEVAL_PROFILE` and `KB_SEED_DIR` override the file.
  - `settings.retrieval` is the `hashing` profile when the provider is `fake` and the `openai` profile when it is `openai`, unless `RETRIEVAL_PROFILE` overrides it.
  - `APP_CONFIG` pointing at a missing file raises `FileNotFoundError`. An unknown provider or profile raises `ValidationError`.
  - `load_prompts()` contains all four role prompts, `chat.user_template` with `{context}`, `{history}` and `{question}`, and `classification.user_template` with `{title}`, `{description}`, `{additional_context}` and the marker `### TICKET_CLASSIFICATION_JSON ###`. `classification.user_template.format(title="t", description="d", additional_context="")` succeeds, which proves the braces are escaped.
  - No prompt string contains "HIPAA compliance" or "create a support ticket".
- **Acceptance.** `make test && GUARD && ! git grep -niE 'hipaa[ -]complian|create a support ticket' -- config`
- **Commit.**
  ```
  Add typed settings loader with env overrides

  Settings and prompts load from config/ into a validated model with
  per-embedder retrieval profiles; config files are trimmed to the keys
  the service actually uses and no longer claim compliance.
  ```

## T4: Add offline fake chat model and hashing embeddings

- **Goal.** Add deterministic offline replacements for OpenAI, plus a factory that picks a provider.
- **Files.** create:
  - `src/fakes.py`, with two classes:
    - `HashingEmbeddings(Embeddings)`, specified exactly as in SPEC: stop words, `1 + ln tf`, `blake2b(digest_size=8)` with index mod 512 and the top bit as the sign, and L2 normalisation. It never calls `hash()`.
    - `ExtractiveFakeChatModel(SimpleChatModel)`, which returns the "Suggested steps" answer built from the context block, or "I could not find a relevant article" when no context is given.
  - `src/llm_factory.py`: `get_chat_model(settings, purpose, http_client=None, http_async_client=None)` and `get_embeddings(settings, http_client=None, http_async_client=None)` for `fake`/`openai`. They honour `OPENAI_BASE_URL`, and `OpenAIEmbeddings` is always built with `check_embedding_ctx_length=False`.
  - `tests/test_fakes.py` and `tests/test_llm_factory.py`.
- **Tests first.**
  - Embeddings are deterministic across instances. A subprocess run with `PYTHONHASHSEED=1` and another with `PYTHONHASHSEED=2` produce identical vectors for the same text, and those match the in-process vector.
  - The embedding norm is 1.
  - A query shares more similarity with an overlapping text than with a disjoint one.
  - The fake model's `ainvoke` returns text from the context.
  - The factory returns the fake classes for `fake`. For `openai` it returns a `ChatOpenAI` with the configured model and base URL, and an `OpenAIEmbeddings` with `check_embedding_ctx_length is False`, both constructed with a dummy key and no network.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Add offline fake LLM, hashing embeddings and provider factory

  Lets the service and tests run without an OpenAI key while keeping the
  langchain-openai path selectable via LLM_PROVIDER.
  ```

## T5: Port knowledge base to Chroma PersistentClient

- **Goal.** Make `KnowledgeBaseManager` work on chromadb 0.5.3 with injected embeddings and replace-by-source ingest.
- **Files.**
  - modify `src/knowledge_base.py`:
    - the constructor takes `embeddings`, the chunk settings and an optional `client`;
    - `initialize()` uses `chromadb.PersistentClient(path)` and creates the collection with `embedding_function=None` (so Chroma never loads its default ONNX model);
    - imports come from `langchain_text_splitters`;
    - `doc_id = sha256(source)[:16]`, and ingest runs `delete(where={"source": source})` before `upsert`.
  - create: `tests/test_knowledge_base.py`.
- **Tests first.** Every test uses `HashingEmbeddings` with a `tmp_path` persistent client and a unique collection name (no `EphemeralClient`). The tests check the following:
  - A long document splits into more than one chunk with overlap.
  - Re-ingesting the same document keeps the count constant.
  - Re-ingesting a shortened version of the same source lowers the chunk count, and no chunk from the old version remains.
  - With the default `hashing` profile values from `config/settings.yml` (chunk 400/80, `min_relevance_score` 0.15), the query `"vpn keeps disconnecting"` returns `vpn-disconnects.md` first against a KB holding all six seed-style articles (fixture text) with relevance above `confidence_threshold` 0.2.
  - An unrelated query ("sourdough bread recipe") returns nothing.
  - The category filter excludes other categories.
  - `min_relevance=0.99` returns nothing.
  - Delete removes all of a document's chunks.
  - `get_stats` counts per category.
  - `ingest_directory` reports errors for unreadable files.
  - Data persists across a new manager on the same path.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Port knowledge base to chromadb PersistentClient

  Replaces the removed duckdb+parquet settings, injects the embedding model,
  replaces documents by source on re-ingest and covers ingest, search,
  filters, delete and persistence with tests.
  ```

## T6: Add seed knowledge articles and ingest CLI

- **Goal.** Add `python -m src.ingest <dir> [--category] [--reset | --api-url URL]` and a small set of generic IT articles.
- **Files.**
  - create: `src/ingest.py` (`ingest_directory(kb, path)` for local mode, reused by the lifespan seed; and HTTP mode with `httpx.Client`, POSTing to `/api/v1/knowledge/ingest`).
  - create the seed articles, each starting with a `category:` line: `data/kb/vpn-disconnects.md`, `data/kb/account-lockout.md`, `data/kb/printer-offline.md`, `data/kb/outlook-mobile-auth.md`, `data/kb/wifi-drops.md`, `data/kb/report-phishing.md`.
  - create: `tests/test_ingest.py`.
  - modify: `.dockerignore` (keep `data/kb`, ignore `data/chroma`).
- **Tests first.**
  - `main([...])` on a temp copy of `data/kb` ingests 6 files, with the right categories taken from front matter.
  - `--reset` empties the collection first.
  - The exit code is 1 when a file fails to decode.
  - `--api-url` together with `--reset` exits 2 with a usage error.
  - `--api-url` mode posts one request per file. This is tested with an `httpx.MockTransport` injected through `main(argv, transport=...)`.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Add knowledge-base ingest CLI and seed IT articles

  Front-matter categories drive metadata; the CLI can write locally or
  through a running API so only one process ever owns the Chroma store.
  ```

## T7: Add regex PHI redaction filter

- **Goal.** Add `src/phi_filter.py` with `redact(text) -> RedactionResult(text, found: list[str])`.
- **Files.** create: `src/phi_filter.py`, `tests/test_phi_filter.py`.
- **Tests first.** The filter redacts `MRN 12345678`, `123-45-6789`, `DOB 01/02/1980`, `+1 555-123-4567` and `jane@example.org` to `[REDACTED:<kind>]`. It leaves error codes such as `0x80070005`, IP addresses and ticket numbers untouched. The `found` kinds are reported.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Add regex-based PHI redaction filter

  Redacts record numbers, SSN-style IDs, birth dates, phones and emails
  before text is sent to a language model.
  ```

## T8: Rebuild AI engine as an LCEL RAG pipeline

- **Goal.** Implement `AIEngine.process_message(message, session_id, user_context)` returning `EngineResponse`, plus `get_stats()`.
- **Files.**
  - modify `src/ai_engine.py`:
    - constructor `AIEngine(settings, prompts, knowledge_base, chat_model)`;
    - chain `ChatPromptTemplate | chat_model | StrOutputParser`;
    - per-session `deque(maxlen=max_context_messages)`;
    - prompt selection exactly as in SPEC item 5 (security content first, then the role map);
    - thresholds from `settings.retrieval`.
  - create: `tests/helpers.py` (`RecordingFakeChatModel(FakeListChatModel)`, imported from `langchain_core.language_models.fake_chat_models`, which overrides `_call` to append the received messages to `self.calls` before returning the canned response) and `tests/test_ai_engine.py`.
- **Tests first.** Each test uses a seeded temp KB, `HashingEmbeddings` and `RecordingFakeChatModel`. The tests check the following:
  - The VPN question returns the VPN article as a source, with confidence equal to the top relevance and no escalation.
  - An unrelated question gives confidence 0 and escalation recommended.
  - The role `physician`/`nurse` picks the `clinical_staff` prompt and `it`/`engineer` picks `technical_staff`.
  - A message containing "phishing" picks `security_incident` even with role `nurse`.
  - An unknown role picks `default`.
  - History is capped at N.
  - PHI in the message is absent from the recorded prompt, and `phi_detected` is true.
  - Stats count messages.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Rebuild AI engine as a retrieval-augmented LCEL chain

  Retrieves from Chroma, selects a role or security prompt, keeps capped
  session history and derives confidence and escalation from retrieval
  relevance.
  ```

## T9: Inject model into ticket classifier, redact PHI and apply escalation rules

- **Goal.** The signature becomes `TicketClassifier(llm=None, settings=None, model_name="gpt-4-0125-preview", temperature=0.1)`:
  - `ChatOpenAI` stays imported at module level and is built lazily, only when `llm` is None, so the existing `patch("src.ticket_classifier.ChatOpenAI")` fixture and the `model_name`/`temperature` keyword arguments keep working.
  - The prompt is `classification.system_prompt` plus `classification.user_template` from `prompts.yml`, and the in-code `CLASSIFICATION_PROMPT` is deleted.
  - The title, description and context pass through `phi_filter.redact` before rendering.
  - `auto_escalate_categories` forces escalation.
- **Files.** modify: `src/ticket_classifier.py`, `tests/test_classifier.py` (add tests).
- **Tests first.**
  - The rendered prompt contains every contract key (`category`, `priority`, `severity`, `confidence`, `suggested_tags`, `reasoning`, `requires_escalation`) and the marker `### TICKET_CLASSIFICATION_JSON ###`.
  - A description containing `MRN 12345678` does not reach the `RecordingFakeChatModel` (the recorded prompt has `[REDACTED:mrn]`).
  - A `FakeListChatModel` returning EHR JSON with `requires_escalation: false` still produces `requires_escalation` true.
  - Fenced JSON parses.
  - The `ExtractiveFakeChatModel` path falls back with confidence 0.5.
  - `batch_classify` preserves order.
  - The existing fixture still constructs a classifier.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Inject chat model into ticket classifier and enforce auto-escalation

  The classifier builds its OpenAI client only when none is injected,
  reads its prompt and output contract from config, redacts patient
  identifiers and escalates configured categories.
  ```

## T10: Wire FastAPI app factory, lifespan and REST routes

- **Goal.** `create_app(settings)` builds the KB, engine, classifier and chat handler in a lifespan and stores them on `app.state`. The lifespan does three more things:
  - it seeds `KB_SEED_DIR` through `ingest_directory` when that is set;
  - it starts the chat handler's cleanup task and cancels it on shutdown;
  - it adds the health endpoints, with ready meaning initialised and seeded.

  The routes use `app.state` dependencies. `KnowledgeSearchRequest.min_relevance` defaults to None, which means the profile value.
- **Files.** modify: `src/main.py`, `src/api/routes.py`, `src/api/models.py`, and `src/chat_handler.py` (adds `stop_cleanup_loop()` and a read-only `cleanup_task` property, so the lifespan cancels the task without reaching into private state). create: `tests/conftest.py` and `tests/test_api.py`. The `settings` fixture uses `tmp_path` for `persist_directory`, a unique collection name, and `KB_SEED_DIR` pointing at a temp copy of `data/kb`. The `client` fixture is `with TestClient(create_app(settings)) as client: yield client`, so the lifespan runs.
- **Tests first.**
  - `/health/live` returns 200.
  - `/health/ready` returns 200 with `kb_chunks > 0`, from the startup seed.
  - `/api/v1/chat` about a VPN drop returns `vpn-disconnects.md` in its sources, and returns a `phi_warning` when an MRN is present.
  - An empty message returns 422.
  - `/classify` returns the fallback category for a VPN ticket.
  - `/classify/batch` with 51 items returns 400.
  - Ingest followed by search, with no `min_relevance` given, finds the new doc.
  - `/knowledge/stats` and `/stats` return their counters.
  - After the `with` block exits, the cleanup task is cancelled.
- **Acceptance.** `make test && docker build --target runtime -t ai-support:runtime .`
- **Commit.**
  ```
  Add FastAPI app factory with lifespan-managed services

  Routes resolve the engine, classifier and knowledge base from app.state;
  the lifespan seeds the knowledge base and runs the session cleanup task,
  and liveness and readiness endpoints are exposed.
  ```

## T11: Share WebSocket chat handler across connections

- **Goal.** One `ChatHandler` per app, so the session limits and counts are real. The handler reads its limits from `settings.chat`. Make the lifecycle KeyError-safe.
- **Changes in the chat handler.**
  - Replace the hardcoded phone-extension and wait-time texts.
  - Replace the "for HIPAA compliance" and "HIPAA-compliant logging" wording with "filtered possible patient identifiers".
  - Use `escalation.threshold_warning` as the escalation text, and drop `ticket_created` from the escalation reply.
  - Pass the WS `role` query parameter through instead of the hardcoded `"staff"`.
- **Files.** modify: `src/chat_handler.py`, `src/api/routes.py`, and `src/main.py` (the lifespan passes `settings.chat` and the prompts to the handler). create: `tests/test_websocket.py`.
- **Tests first.** The tests use `client.websocket_connect("/api/v1/chat/ws/u1")` and check the following:
  - The first message is `session_start`.
  - A `user_message` yields `typing_start`, `typing_stop` and then `ai_response` with sources.
  - Non-JSON input yields `error`.
  - A message longer than `max_message_length` yields `error`.
  - `escalate` yields `escalation`, whose payload has no `ticket_created` key and no "HIPAA" text.
  - A PHI message yields a system notice with no "HIPAA" text.
  - `?role=nurse` selects the clinical prompt, checked through a `RecordingFakeChatModel` on `app.state`.
  - `end_session` yields `session_end`, and the active count returns to 0.
  - With `create_app(settings.model_copy(update={"chat": settings.chat.model_copy(update={"max_concurrent_sessions": 1})}))`, a second connection gets `error` and close code 1013.
  - With `session_timeout_seconds` 0.1 set the same way, an idle connection receives `session_end`.
- **Acceptance.** `make test && GUARD && GUARD_STRINGS`
- **Commit.**
  ```
  Share one WebSocket chat handler per app instance

  Session limits and counts now apply across connections, escalation no
  longer claims a ticket was created, and the full message protocol is
  covered with TestClient WebSocket tests.
  ```

## T12: Add OpenAI-compatible stub server

- **Goal.** Add `stubs/openai_stub.py`, a FastAPI app with `/v1/chat/completions`, `/v1/embeddings` and `/v1/models`:
  - It reuses `HashingEmbeddings` and the extractive answer logic.
  - When the prompt contains the marker `### TICKET_CLASSIFICATION_JSON ###`, it returns classifier JSON derived from the keyword map.
  - `/v1/embeddings` accepts `input` as `str` or `list[str]`, and returns 400 for token-ID arrays.
- **Files.** create: `stubs/__init__.py`, `stubs/openai_stub.py`, `tests/test_openai_stub.py`.
- **Tests first.** Build the clients through `llm_factory` with `sync = TestClient(stub_app, base_url="http://stub")` and `async_ = httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_app), base_url="http://stub")`, and with settings `openai`, base URL `http://stub/v1` and key `sk-test`. The tests check the following:
  - `ChatOpenAI.invoke` and `ainvoke` both return content.
  - `OpenAIEmbeddings.embed_documents` and `aembed_documents` return 512-dimensional vectors equal to `HashingEmbeddings` on the same text, which proves raw strings were sent.
  - The stub answers a token-array request with 400.
  - A `TicketClassifier` with that `ChatOpenAI` gets parsed, non-fallback JSON (confidence other than 0.5).
  - The whole module passes under `make test`'s `--network none`, which proves that tiktoken is never loaded.
- **Acceptance.** `make test`
- **Commit.**
  ```
  Add OpenAI-compatible stub for keyless provider testing

  Exercises the real langchain-openai sync and async code paths against
  deterministic chat and embedding endpoints without network access.
  ```

## T13: Compose stack with stub and integration tests

- **Goal.** `docker compose up` runs two services, plus a test-only third:
  - **`openai-stub`** runs from the runtime image with `uvicorn stubs.openai_stub:app --host 0.0.0.0 --port 8001`. Its healthcheck is `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/v1/models')"`.
  - **`api`** runs from the runtime image with these settings:
    - `LLM_PROVIDER=openai`
    - `OPENAI_BASE_URL=http://openai-stub:8001/v1`
    - `OPENAI_API_KEY=${OPENAI_API_KEY:-sk-stub-not-a-real-key}`
    - `RETRIEVAL_PROFILE=hashing`
    - `KB_SEED_DIR=/app/data/kb`
    - the `chroma_data` volume
    - `depends_on: {openai-stub: {condition: service_healthy}}`
    - healthcheck `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"`
  - **`tester`**, under profile `test`, uses the dev image with `STACK_URL=http://api:8000`.
- **Files.** modify: `docker-compose.yml`, `Makefile` (targets `up`, `seed`, `integration`, `down`). `seed` is `docker compose exec -T api python -m src.ingest data/kb --api-url http://localhost:8000`, which goes through the API process. create: `tests/integration/__init__.py`, `tests/integration/test_stack.py` (skipped unless `STACK_URL` is set).
- **Tests first.** Against the live stack, check the following:
  - Ready returns 200 with `kb_chunks > 0`.
  - A chat about VPN cites `vpn-disconnects.md`.
  - `/classify` returns a non-fallback result through the stub.
  - A WebSocket round-trip works.
  - Ingest followed by search works.
- **Acceptance.** `make test && make integration`, where the `integration` recipe is one shell line:
  ```
  docker compose up -d --build --wait && docker compose --profile test run --rm tester pytest tests/integration; rc=$$?; docker compose down -v; exit $$rc
  ```
- **Commit.**
  ```
  Run API and OpenAI stub under Compose with integration tests

  Drops the unused Redis, Prometheus and Grafana services, seeds the
  knowledge base inside the API process and verifies the running stack
  over HTTP and WebSocket.
  ```

## T14: Regenerate README from template

- **Goal.** Write an honest README following `tools/readme_template.md`, with these sections:
  - a concrete description;
  - Status: implemented features mapped to code and tests, and the limitations from the SPEC, including "OpenAI is replaced by a local stub/offline fakes in all tests; the real API is never called by the test suite";
  - Built with: the pinned versions and the base-image digest note;
  - Running it: `docker compose up`, then the smoke command `curl -s localhost:8000/api/v1/knowledge/search -H 'content-type: application/json' -d '{"query":"vpn keeps disconnecting","min_relevance":0.15}'`. The stack seeds itself, and `make seed` re-seeds it through the API;
  - Tests;
  - a Layout tree generated from `git ls-files`.

  Remove the template's rules comment and any badges.
- **Files.** modify: `README.md`.
- **Tests first.** None (docs only). The acceptance command checks the content.
- **Acceptance.**
  ```
  make test && GUARD && GUARD_STRINGS \
    && ! grep -Eiq 'Dubai|Status-Complete|Developed during|70%|85%|99\.9' README.md \
    && ! grep -qw 'AEL' README.md \
    && ! grep -qE 'Rules for this README|img.shields.io' README.md \
    && grep -q 'Personal project built on the 2024-era stack' README.md \
    && git ls-files | while read f; do grep -qF "$(basename "$f")" README.md || echo "missing $f"; done | (! grep .)
  ```
- **Commit.**
  ```
  Rewrite README to match the implemented service

  Documents what is built and tested, names the mocked OpenAI integration
  and lists the layout from the tracked files.
  ```
