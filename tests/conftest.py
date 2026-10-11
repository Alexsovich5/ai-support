"""Fixtures for tests that run the FastAPI app through its lifespan."""

import shutil
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import load_settings

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"


@pytest.fixture
def settings(tmp_path):
    """Fake-provider settings with a private Chroma directory and seed copy."""
    seed_dir = tmp_path / "kb"
    shutil.copytree(SEED_DIR, seed_dir)
    base = load_settings(env={"LLM_PROVIDER": "fake"})
    return base.model_copy(
        update={
            "knowledge_base": base.knowledge_base.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": f"kb_{uuid.uuid4().hex}",
                }
            ),
            "kb_seed_dir": str(seed_dir),
        }
    )


@pytest.fixture
def app(settings):
    from src.main import create_app

    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client
