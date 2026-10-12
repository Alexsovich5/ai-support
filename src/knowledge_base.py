"""
RAG knowledge base manager.

Manages the vector store knowledge base for retrieval-augmented generation:
document ingestion, embedding generation and similarity search over IT
support articles. Article text, titles and search queries pass through the
PHI filter before they are embedded or stored, and failures of the store are
raised as ``KnowledgeBaseError`` instead of being reported as "no results".
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.phi_filter import WITHHELD_KIND, redact

logger = logging.getLogger("ai_support.knowledge_base")


class KnowledgeBaseError(RuntimeError):
    """The vector store could not complete a search or a statistics read."""


@dataclass
class Document:
    content: str
    source: str
    category: str
    title: str = ""
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:16]


@dataclass
class SearchResult:
    content: str
    source: str
    relevance_score: float
    category: str
    metadata: dict = field(default_factory=dict)


class KnowledgeBaseManager:
    """
    Vector store manager for healthcare IT knowledge base.
    Uses a persistent ChromaDB collection for storage and cosine search.
    The embedding model is injected, so Chroma never computes embeddings itself.
    """

    CATEGORIES = [
        "network",
        "email",
        "vpn",
        "ehr_systems",
        "active_directory",
        "printing",
        "hardware",
        "software",
        "security",
        "cloud_services",
        "mobile_devices",
        "telephony",
    ]

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./data/chroma",
        collection_name: str = "it_support_kb",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        client: Optional[ClientAPI] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.document_count = 0
        self.embeddings = embeddings

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._client = client
        self._collection = None

        logger.info(
            "Knowledge base manager created: dir=%s, collection=%s",
            persist_directory,
            collection_name,
        )

    async def initialize(self) -> None:
        """Open the persistent Chroma client and the cosine collection."""
        try:
            if self._client is None:
                Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )

            self.document_count = self._collection.count()
            logger.info(
                "Knowledge base initialized: %d chunks loaded",
                self.document_count,
            )

        except Exception as e:
            logger.error("Failed to initialize knowledge base: %s", type(e).__name__)
            raise

    async def ingest_document(self, document: Document) -> dict:
        """
        Ingest a document into the knowledge base.
        Splits into chunks, generates embeddings, and stores in ChromaDB.
        """
        if not self._collection:
            raise RuntimeError("Knowledge base not initialized")

        content = redact(document.content)
        title = redact(document.title)
        if WITHHELD_KIND in content.found or WITHHELD_KIND in title.found:
            raise ValueError(f"document '{document.source}' could not be checked for patient identifiers")
        phi_found = list(dict.fromkeys(content.found + title.found))
        if phi_found:
            logger.warning(
                "Patient identifiers redacted from article: source=%s kinds=%s",
                document.source,
                ",".join(phi_found),
            )

        chunks = self.text_splitter.split_text(content.text)
        if not chunks:
            raise ValueError(f"document '{document.source}' has no content")
        logger.info(
            "Ingesting document: source=%s, chunks=%d",
            document.source,
            len(chunks),
        )

        # Generate embeddings
        chunk_embeddings = await self._generate_embeddings(chunks)

        # Prepare IDs and metadata for each chunk
        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.doc_id}_{i}"
            ids.append(chunk_id)
            metadatas.append({
                "source": document.source,
                "category": document.category,
                "title": title.text,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_id": document.doc_id,
                "created_at": document.created_at,
                **document.metadata,
            })

        # Replace any earlier version of this source, then write the new chunks
        self._collection.delete(where={"source": document.source})
        self._collection.upsert(
            ids=ids,
            embeddings=chunk_embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        self.document_count = self._collection.count()

        result = {
            "doc_id": document.doc_id,
            "chunks_created": len(chunks),
            "total_documents": self.document_count,
            "source": document.source,
            "phi_redacted": phi_found,
        }

        logger.info("Document ingested: %s", json.dumps(result))
        return result

    async def ingest_directory(self, directory: str, category: str = "general") -> dict:
        """Batch ingest a directory through the bounded, link-refusing walker in ``src.ingest``."""
        from src.ingest import ingest_directory

        results = await ingest_directory(self, directory, category)
        logger.info(
            "Directory ingestion complete: files=%d, chunks=%d, errors=%d",
            results["total_files"],
            results["total_chunks"],
            len(results["errors"]),
        )
        return results

    async def search(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        min_relevance: float = 0.3,
    ) -> list[dict]:
        """
        Similarity search across the knowledge base.
        Returns ranked results with relevance scores. The query is PHI-redacted
        before it is embedded; store errors raise ``KnowledgeBaseError``.
        """
        if not self._collection:
            raise KnowledgeBaseError("knowledge base not initialized")

        query = redact(query).text
        query_embedding = await self._generate_embeddings([query])

        where_filter = None
        if category:
            where_filter = {"category": category}

        available = self._collection.count()
        if available == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, available),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Search failed: %s", type(e).__name__)
            raise KnowledgeBaseError("search failed") from None

        search_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                # Convert cosine distance to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance = 1.0 - distance

                if relevance < min_relevance:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append({
                    "content": doc,
                    "source": metadata.get("source", "unknown"),
                    "relevance_score": round(relevance, 4),
                    "category": metadata.get("category", "general"),
                    "title": metadata.get("title", ""),
                    "metadata": metadata,
                })

        logger.info(
            "Search completed: query_length=%d, results=%d",
            len(query),
            len(search_results),
        )

        return search_results

    async def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks from the knowledge base."""
        if not self._collection:
            return False

        try:
            # Find all chunks belonging to this document
            results = self._collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )

            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                self.document_count = self._collection.count()
                logger.info("Document deleted: doc_id=%s, chunks=%d", doc_id, len(results["ids"]))
                return True

            return False

        except Exception as e:
            logger.error("Failed to delete document %s: %s", doc_id, type(e).__name__)
            return False

    async def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        if not self._collection:
            raise KnowledgeBaseError("knowledge base not initialized")

        categories = {}
        try:
            all_docs = self._collection.get(include=["metadatas"])
            self.document_count = self._collection.count()
        except Exception as e:
            logger.error("Reading knowledge base statistics failed: %s", type(e).__name__)
            raise KnowledgeBaseError("statistics unavailable") from None
        for meta in (all_docs or {}).get("metadatas") or []:
            cat = meta.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_chunks": self.document_count,
            "categories": categories,
            "collection_name": self.collection_name,
        }

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error("Embedding generation failed: %s", type(e).__name__)
            raise
