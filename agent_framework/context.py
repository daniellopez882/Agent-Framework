"""RAG context: PDF -> chunks -> ChromaDB -> retrieved text for the prompt.

Ported from ``step-5-context/context.py`` (its byte-identical copy in step 6
is gone). Changes: the embedding function is injectable, so tests do not
download ChromaDB's default ONNX model; an empty query result is handled; and
``print`` is ``logging``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

import chromadb
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Metadata attached to every chunk of one document."""

    def __init__(
        self,
        source: str,
        doc_type: str = "pdf",
        author: str | None = None,
        created_at: datetime | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.source = source
        self.doc_type = doc_type
        self.author = author
        self.created_at = created_at or datetime.now()
        self.tags = tags or []
        self.additional_metadata = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            **self.additional_metadata,
        }

    def to_chroma_metadata(self) -> dict[str, Any]:
        """ChromaDB metadata values must be scalars; lists and None are not accepted."""
        out: dict[str, Any] = {}
        for k, v in self.to_dict().items():
            if v is None:
                continue
            out[k] = ", ".join(map(str, v)) if isinstance(v, list) else v
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        data = dict(data)
        created = data.pop("created_at", None)
        return cls(
            source=data.pop("source"),
            doc_type=data.pop("doc_type", "pdf"),
            author=data.pop("author", None),
            created_at=datetime.fromisoformat(created) if created else None,
            tags=data.pop("tags", []),
            **data,
        )


class ContextManager:
    """Index PDFs into a ChromaDB collection and retrieve chunks for a query."""

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_function: Any = None,
        client: Any = None,
    ) -> None:
        if persist_dir is None:
            from agent_framework.config import settings

            persist_dir = settings.CONTEXT_PERSIST_DIR
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_results = 3

        if client is None:
            os.makedirs(persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_dir)
        self.client = client
        self._embedding_function = embedding_function
        self.collection = self._open_collection()
        logger.info("Context %s ready in %s", collection_name, persist_dir)

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self._current_query: str | None = None
        self._current_context: str | None = None
        self._indexed_documents: dict[str, DocumentMetadata] = {}

    def _open_collection(self) -> Any:
        kwargs: dict[str, Any] = {
            "name": self.collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return self.client.get_or_create_collection(**kwargs)

    @classmethod
    def initialize(
        cls, collection_name: str | None = None, persist_dir: str | None = None, **kwargs: Any
    ) -> ContextManager:
        if not collection_name:
            collection_name = "collection_" + hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        return cls(collection_name=collection_name, persist_dir=persist_dir, **kwargs)

    @staticmethod
    def _generate_document_id(content: str, metadata: dict[str, Any] | None = None) -> str:
        id_content = content + (json.dumps(metadata, sort_keys=True) if metadata else "")
        return hashlib.sha256(id_content.encode()).hexdigest()[:16]

    @staticmethod
    def _extract_text_from_pdf(pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        with open(pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            if len(reader.pages) == 0:
                raise ValueError(f"PDF file is empty: {pdf_path}")
            text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(f"No text content extracted from PDF: {pdf_path}")
        return text

    def index_text(self, text: str, metadata: DocumentMetadata, key: str | None = None) -> int:
        """Index raw text; returns the number of chunks. Used by ``index_document`` and tests."""
        chunks = self._text_splitter.split_text(text)
        if not chunks:
            raise ValueError("No chunks generated from text")
        meta = metadata.to_chroma_metadata()
        self.collection.add(
            documents=chunks,
            ids=[self._generate_document_id(c, meta) for c in chunks],
            metadatas=[meta for _ in chunks],
        )
        self._indexed_documents[key or metadata.source] = metadata
        return len(chunks)

    def index_document(
        self, pdf_path: str, metadata: dict[str, Any] | DocumentMetadata | None = None
    ) -> bool:
        try:
            if isinstance(metadata, dict):
                metadata = DocumentMetadata.from_dict(metadata)
            elif metadata is None:
                metadata = DocumentMetadata(source=os.path.basename(pdf_path))
            n = self.index_text(self._extract_text_from_pdf(pdf_path), metadata, key=pdf_path)
            logger.info("Indexed %s (%d chunks)", pdf_path, n)
            return True
        except Exception:
            logger.exception("Error indexing document %s", pdf_path)
            return False

    def set_query(
        self, query: str, num_results: int = 3, filter_metadata: dict[str, Any] | None = None
    ) -> str:
        return self.query(query, num_results, filter_metadata)

    def query(
        self, query: str, num_results: int = 3, filter_metadata: dict[str, Any] | None = None
    ) -> str:
        self._current_query = query
        self.num_results = num_results
        try:
            params: dict[str, Any] = {"query_texts": [query], "n_results": num_results}
            if filter_metadata:
                params["where"] = filter_metadata
            results = self.collection.query(**params)
        except Exception:
            logger.exception("Error executing query")
            self._current_context = ""
            return ""

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0] or [{} for _ in docs]
        parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas, strict=False), 1):
            source = (meta or {}).get("source", "Unknown source")
            parts.append(f"Relevant Context {i} (from {source}):\n{doc}\n")
        self._current_context = "\n".join(parts)
        return self._current_context

    def get_document_metadata(self, source: str) -> DocumentMetadata | None:
        return self._indexed_documents.get(source)

    def list_indexed_documents(self) -> list[dict[str, Any]]:
        return [{"source": s, "metadata": m.to_dict()} for s, m in self._indexed_documents.items()]

    def clear_index(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._open_collection()
            self._indexed_documents.clear()
            self._current_context = None
            self._current_query = None
            return True
        except Exception:
            logger.exception("Error clearing index %s", self.collection_name)
            return False

    @property
    def current_query(self) -> str | None:
        return self._current_query

    @property
    def response(self) -> str | None:
        return self._current_context

    def save_state(self) -> dict[str, Any]:
        return {
            "current_query": self._current_query,
            "current_context": self._current_context,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "indexed_documents": {s: m.to_dict() for s, m in self._indexed_documents.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._current_query = state.get("current_query")
        self._current_context = state.get("current_context")
        self.chunk_size = state.get("chunk_size", self.chunk_size)
        self.chunk_overlap = state.get("chunk_overlap", self.chunk_overlap)
        self._indexed_documents = {
            s: DocumentMetadata.from_dict(m) for s, m in state.get("indexed_documents", {}).items()
        }
