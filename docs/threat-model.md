# Threat model

Scope: a Python library that runs an LLM agent with a persona, conversation
memory in SQLite, optional RAG context in ChromaDB, and tools (Wikipedia, and
Tavily web search when a key is present). It is not a service; whoever imports
it decides who can call `execute`.

## What it holds

| Asset | Where | Why it matters |
|---|---|---|
| `OPENAI_API_KEY` | environment / `.env` | Every `execute` spends it; each tool call adds a model call |
| `TAVILY_API_KEY` | environment / `.env` | Billable search |
| Conversation history and tasks | `agent_memory.db` (SQLite, plaintext) | Whatever users typed and the model answered |
| Indexed documents | `context_db/` (ChromaDB) | The PDFs' text, chunked |

`.env` is gitignored and CI fails if one is tracked; gitleaks scans history.
`agent_memory.db` and `context_db/` are gitignored too — the old repo had no
`.gitignore`, so the first run would have committed both.

## Threats

### T1 — Prompt injection through tool output

With a real loop (ADR 0002), tool results are fed back to the model as
observations. Wikipedia text and web search snippets are attacker-writable.
**Controls.** Tools are read-only (lookups); there is no shell, file or HTTP
tool; the loop is bounded by `MAX_TOOL_ITERATIONS`. **Residual.** A crafted
page can still steer the answer. Do not add tools with side effects without
an approval step.

### T2 — Unbounded spend

Each iteration is a model call; a model that keeps calling tools would loop.
**Controls.** `MAX_TOOL_ITERATIONS` (default 5, max 25) ends the run with
`ToolLoopExceeded`; a client timeout is set. **Residual.** No per-caller
quota — that belongs to whatever wraps this library.

### T3 — State on disk

History is stored in plaintext SQLite in the working directory (or
`AGENT_DB_PATH`). Anyone with filesystem access reads every conversation.
**Controls.** Paths are configuration; the container writes to a volume as a
non-root user. **Residual.** No encryption at rest; treat the database like
any other file with user data.

### T4 — Model-chosen tool parameters

The model chooses the tool name and parameter values; they are passed as
keyword arguments to `Tool.execute(**kwargs)`. **Controls.** The parser only
reads `- name: value` lines under `Parameters:`; the built-in tools read one
named parameter and ignore the rest; unknown tools are reported back to the
model, not raised. **Residual.** A custom tool that interpolates parameters
into a command or query must validate them itself.

### T5 — Supply chain

An unpinned `langchain` resolved to a version without the module the code
imported, so the package could not be imported at all. **Controls.** Exact
pins in `requirements.txt`, bounded ranges in `pyproject.toml`, `pip-audit`,
`bandit` and gitleaks in CI; the container runs the test suite against the
pinned set.

## Not addressed

- Authentication and rate limiting: this is a library, not a server.
- Confidentiality of what is sent to the model provider.
- ChromaDB's default embedding function downloads a model on first use; pin
  or vendor it for air-gapped deployments (the tests inject a stub).
