#!/usr/bin/env python3
"""
RAG Knowledge Base Manager
Project: AI-Powered Healthcare IT Support System

Manages the vector store knowledge base for Retrieval-Augmented Generation.
Handles document ingestion, embedding generation, and semantic search for
healthcare IT support documentation.
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

logger = logging.getLogger("ai_support.knowledge_base")


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
            logger.error("Failed to initialize knowledge base: %s", str(e))
            raise

    async def ingest_document(self, document: Document) -> dict:
        """
        Ingest a document into the knowledge base.
        Splits into chunks, generates embeddings, and stores in ChromaDB.
        """
        if not self._collection:
            raise RuntimeError("Knowledge base not initialized")

        chunks = self.text_splitter.split_text(document.content)
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
                "title": document.title,
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
        }

        logger.info("Document ingested: %s", json.dumps(result))
        return result

    async def ingest_directory(
        self,
        directory: str,
        category: str = "general",
        file_extensions: list[str] = None,
    ) -> dict:
        """Batch ingest all documents from a directory."""
        if file_extensions is None:
            file_extensions = [".md", ".txt", ".rst", ".html"]

        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        results = {"total_files": 0, "total_chunks": 0, "errors": []}

        for ext in file_extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    doc = Document(
                        content=content,
                        source=str(file_path.relative_to(dir_path)),
                        category=category,
                        title=file_path.stem.replace("_", " ").replace("-", " ").title(),
                    )
                    result = await self.ingest_document(doc)
                    results["total_files"] += 1
                    results["total_chunks"] += result["chunks_created"]
                except Exception as e:
                    results["errors"].append({"file": str(file_path), "error": str(e)})
                    logger.error("Error ingesting %s: %s", file_path, str(e))

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
        Semantic search across the knowledge base.
        Returns ranked results with relevance scores.
        """
        if not self._collection:
            return []

        # Generate query embedding
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
            logger.error("Search failed: %s", str(e))
            return []

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
            "Search completed: query='%s...', results=%d",
            query[:50],
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
            logger.error("Failed to delete document %s: %s", doc_id, str(e))
            return False

    async def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        if not self._collection:
            return {"status": "not_initialized"}

        categories = {}
        try:
            all_docs = self._collection.get(include=["metadatas"])
            if all_docs and all_docs["metadatas"]:
                for meta in all_docs["metadatas"]:
                    cat = meta.get("category", "uncategorized")
                    categories[cat] = categories.get(cat, 0) + 1
        except Exception:
            pass

        self.document_count = self._collection.count()
        return {
            "total_chunks": self.document_count,
            "categories": categories,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error("Embedding generation failed: %s", str(e))
            raise
