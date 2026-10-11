IMAGE ?= ai-support:dev

.PHONY: build test shell clean up seed integration down

build:
	docker build --target dev -t $(IMAGE) .

test: build
	docker run --rm --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False $(IMAGE) pytest

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
	rm -rf .pytest_cache .coverage htmlcov
