IMAGE ?= ai-support:dev
COMPOSE_RENDERED := .compose-rendered.json

.PHONY: build test shell clean up seed integration down

build:
	docker build --target dev -t $(IMAGE) .

# The rendered compose config is checked too (loopback-only ports, least privilege).
test: build
	docker compose --profile test config --format json > $(COMPOSE_RENDERED)
	docker run --rm --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False \
		-e COMPOSE_RENDERED=/tmp/compose-rendered.json \
		-v "$(CURDIR)/$(COMPOSE_RENDERED):/tmp/compose-rendered.json:ro" $(IMAGE) pytest

shell: build
	docker run --rm -it --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False $(IMAGE) bash

up:
	docker compose up -d --build --wait

seed:
	docker compose exec -T api python -m src.ingest data/kb --api-url http://localhost:8000

integration:
	docker compose up -d --build --wait && docker compose --profile test run --rm tester pytest tests/integration; rc=$$?; docker compose down -v; exit $$rc

down:
	docker compose down

clean:
	-docker image rm $(IMAGE)
	rm -rf .pytest_cache .coverage htmlcov $(COMPOSE_RENDERED)
