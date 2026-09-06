"""Step 5 — RAG context from a PDF. Needs OPENAI_API_KEY and a PDF path as argv[1].

ChromaDB's default embedding function downloads a small ONNX model on first use.
"""

import sys

from agent_framework import Agent
from agent_framework.context import ContextManager, DocumentMetadata


def main(pdf_path: str) -> None:
    context = ContextManager.initialize(collection_name="simple_docs")
    if not context.index_document(pdf_path, DocumentMetadata(source=pdf_path, tags=["example"])):
        raise SystemExit(f"could not index {pdf_path}")

    agent = Agent("rag_agent")
    agent.persona = (
        "You are a helpful AI assistant that provides accurate information based on the "
        "given context. You analyze documents and explain complex topics clearly."
    )
    agent.instruction = "Focus on key principles; use clear language; give examples."
    agent.strategy = "ReactStrategy"

    context.set_query("What are the main principles discussed in the document?")
    agent.context = context
    print(agent.execute("Identify and explain the key principles in the document"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python examples/05_context.py <file.pdf>")
    main(sys.argv[1])
