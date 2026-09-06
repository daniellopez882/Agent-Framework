"""ContextManager against a real ChromaDB in a temp dir, with a stub embedding
function so nothing is downloaded."""

import hashlib

import pytest

try:
    import chromadb_rust_bindings  # noqa: F401
except ImportError as e:  # pragma: no cover - environment, not code
    pytest.skip(f"chromadb native bindings unavailable here: {e}", allow_module_level=True)

from chromadb.api.types import Documents, EmbeddingFunction

from agent_framework.context import ContextManager, DocumentMetadata


class StubEmbedding(EmbeddingFunction[Documents]):
    """Deterministic 8-dim embeddings; enough for Chroma to index and query."""

    def __call__(self, input):  # noqa: A002 - Chroma's parameter name
        out = []
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            out.append([b / 255.0 for b in h[:8]])
        return out

    @staticmethod
    def name():
        return "stub"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return StubEmbedding()

    def is_legacy(self):
        return False


@pytest.fixture
def ctx(tmp_path):
    return ContextManager(
        "docs", persist_dir=str(tmp_path / "chroma"), embedding_function=StubEmbedding()
    )


def test_metadata_round_trip_and_chroma_shape():
    m = DocumentMetadata(source="a.pdf", author=None, tags=["x", "y"], extra="e")
    d = m.to_dict()
    back = DocumentMetadata.from_dict(dict(d))
    assert (
        back.source == "a.pdf"
        and back.tags == ["x", "y"]
        and back.additional_metadata == {"extra": "e"}
    )
    chroma = m.to_chroma_metadata()
    assert "author" not in chroma and chroma["tags"] == "x, y"


def test_index_and_query(ctx):
    n = ctx.index_text("alpha " * 300 + "omega", DocumentMetadata(source="big.txt"))
    assert n > 1
    out = ctx.set_query("omega", num_results=2)
    assert out.startswith("Relevant Context 1 (from big.txt):")
    assert ctx.response == out and ctx.current_query == "omega" and ctx.num_results == 2
    assert [d["source"] for d in ctx.list_indexed_documents()] == ["big.txt"]
    assert ctx.get_document_metadata("big.txt").source == "big.txt"


def test_repeated_chunks_index_and_reindexing_is_idempotent(ctx):
    # Five identical paragraphs -> identical chunks. The old id (hash of text + metadata)
    # collided and ChromaDB raised DuplicateIDError.
    text = ("same paragraph. " * 40 + "\n\n") * 5
    n = ctx.index_text(text, DocumentMetadata(source="dup.txt"))
    assert n >= 2
    assert ctx.collection.count() == n
    # Indexing the same document again replaces its chunks; it does not fail or duplicate.
    assert ctx.index_text(text, DocumentMetadata(source="dup.txt")) == n
    assert ctx.collection.count() == n


def test_query_on_empty_collection_is_empty(ctx):
    assert ctx.query("anything") == ""
    assert ctx.response == ""


def test_clear_index_and_state(ctx):
    ctx.index_text("hello world", DocumentMetadata(source="s"))
    ctx.query("hello")
    state = ctx.save_state()
    assert state["collection_name"] == "docs" and "s" in state["indexed_documents"]
    assert ctx.clear_index()
    assert ctx.list_indexed_documents() == [] and ctx.response is None
    assert ctx.query("hello") == ""
    ctx.load_state(state)
    assert ctx.current_query == "hello" and ctx.get_document_metadata("s") is not None


def test_missing_pdf_is_reported_not_raised(ctx):
    assert ctx.index_document("/nope/missing.pdf") is False


def test_initialize_generates_a_name(tmp_path):
    c = ContextManager.initialize(
        persist_dir=str(tmp_path / "c"), embedding_function=StubEmbedding()
    )
    assert c.collection_name.startswith("collection_")
