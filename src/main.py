"""FastAPI application factory.

``create_app(settings)`` returns an app whose lifespan opens the knowledge
base, seeds it from ``settings.kb_seed_dir`` when set, builds the AI engine,
ticket classifier and chat handler, and stores them on ``app.state``. The
chat handler's session cleanup task runs for the lifetime of the app.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.ai_engine import AIEngine
from src.api.routes import router
from src.chat_handler import ChatHandler
from src.config import Settings, load_prompts, load_settings
from src.ingest import ingest_directory
from src.knowledge_base import KnowledgeBaseManager
from src.llm_factory import get_chat_model, get_embeddings
from src.ticket_classifier import TicketClassifier

logger = logging.getLogger("ai_support.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    prompts = load_prompts()

    kb = KnowledgeBaseManager(
        embeddings=get_embeddings(settings),
        persist_directory=settings.knowledge_base.persist_directory,
        collection_name=settings.knowledge_base.collection_name,
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
    )
    await kb.initialize()

    if settings.kb_seed_dir:
        summary = await ingest_directory(kb, settings.kb_seed_dir)
        logger.info(
            "Seeded knowledge base from %s: files=%d chunks=%d errors=%d",
            settings.kb_seed_dir,
            summary["total_files"],
            summary["total_chunks"],
            len(summary["errors"]),
        )

    engine = AIEngine(settings, prompts, kb, get_chat_model(settings, "chat"))
    classifier = TicketClassifier(
        llm=get_chat_model(settings, "classification"),
        settings=settings,
        prompts=prompts,
    )
    chat_handler = ChatHandler(ai_engine=engine)

    app.state.knowledge_base = kb
    app.state.ai_engine = engine
    app.state.classifier = classifier
    app.state.chat_handler = chat_handler

    await chat_handler.start_cleanup_loop()
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False
        await chat_handler.stop_cleanup_loop()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the application; settings default to ``load_settings()``."""
    settings = settings if settings is not None else load_settings()
    logging.basicConfig(level=settings.app.log_level)

    app = FastAPI(title=settings.app.name, version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.ready = False
    app.include_router(router, prefix="/api/v1")

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready():
        if not getattr(app.state, "ready", False):
            return JSONResponse(status_code=503, content={"status": "starting"})
        return {"status": "ready", "kb_chunks": app.state.knowledge_base.document_count}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
