IMAGE ?= ai-support:dev

.PHONY: build test shell clean

build:
	docker build --target dev -t $(IMAGE) .

test: build
	docker run --rm --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False $(IMAGE) pytest

shell: build
	docker run --rm -it --network none -e LLM_PROVIDER=fake -e ANONYMIZED_TELEMETRY=False $(IMAGE) bash

clean:
	-docker image rm $(IMAGE)
	rm -rf .pytest_cache .coverage htmlcov
