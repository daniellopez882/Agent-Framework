"""SQLite persistence for agent state.

Two defects, both reproduced before the change:

* ``load_agent_state`` called ``agent.initialize_context`` and
  ``agent.set_context_query`` — methods ``Agent`` never had. Any agent saved
  with a context raised ``AttributeError`` inside a bare ``except``, printed
  "Error loading agent state", and returned ``False``: a persisted RAG context
  could never be restored. The context is rebuilt here, through a factory.
* "Most recent state" was ``ORDER BY timestamp DESC``, and the timestamp has
  one-second resolution. Every property setter wrote a state row, so a single
  configuration produced several rows with the same timestamp and the query
  could return any of them. Rows are ordered by their autoincrement id now, and
  setters no longer snapshot (see ``Agent``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_framework.agent import Agent
    from agent_framework.context import ContextManager

logger = logging.getLogger(__name__)

ContextFactory = Callable[[str, str], "ContextManager"]


def _default_context_factory(collection_name: str, persist_dir: str) -> ContextManager:
    # Imported lazily: chromadb is heavy and only needed when a context is restored.
    from agent_framework.context import ContextManager

    return ContextManager(collection_name=collection_name, persist_dir=persist_dir)


class AgentPersistence:
    """Save and load agents in a SQLite database."""

    def __init__(
        self,
        db_path: str | None = None,
        context_factory: ContextFactory | None = None,
    ) -> None:
        if db_path is None:
            from agent_framework.config import settings

            db_path = settings.AGENT_DB_PATH
        self.db_path = db_path
        self._context_factory = context_factory or _default_context_factory
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        conn: sqlite3.Connection = self._local.conn
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    persona TEXT,
                    instruction TEXT,
                    strategy TEXT,
                    context_collection TEXT,
                    context_persist_dir TEXT,
                    context_query TEXT,
                    context_num_results INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS agent_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    task TEXT,
                    history TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (agent_name) REFERENCES agents(name) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_agent_states_agent_name
                ON agent_states(agent_name);
                CREATE TRIGGER IF NOT EXISTS update_agent_timestamp
                AFTER UPDATE ON agents
                BEGIN
                    UPDATE agents SET last_updated = CURRENT_TIMESTAMP WHERE name = NEW.name;
                END;
                """
            )

    def save_agent_state(self, agent: Agent) -> bool:
        try:
            with self._get_conn() as conn:
                ctx = agent._context
                conn.execute(
                    """
                    INSERT INTO agents (
                        name, persona, instruction, strategy,
                        context_collection, context_persist_dir,
                        context_query, context_num_results
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        persona = excluded.persona,
                        instruction = excluded.instruction,
                        strategy = excluded.strategy,
                        context_collection = excluded.context_collection,
                        context_persist_dir = excluded.context_persist_dir,
                        context_query = excluded.context_query,
                        context_num_results = excluded.context_num_results
                    """,
                    (
                        agent.name,
                        agent.persona,
                        agent.instruction,
                        agent.strategy.__class__.__name__ if agent.strategy else None,
                        ctx.collection_name if ctx else None,
                        ctx.persist_dir if ctx else None,
                        ctx.current_query if ctx else None,
                        getattr(ctx, "num_results", 3) if ctx else None,
                    ),
                )
                conn.execute(
                    "INSERT INTO agent_states (agent_name, task, history) VALUES (?, ?, ?)",
                    (agent.name, agent.task, json.dumps(agent.history)),
                )
            return True
        except sqlite3.Error:
            logger.exception("Error saving agent state for %s", agent.name)
            return False

    def load_agent_state(self, agent: Agent, agent_name: str | None = None) -> bool:
        name_to_load = agent_name or agent.name
        try:
            with self._get_conn() as conn:
                agent_data = conn.execute(
                    """
                    SELECT persona, instruction, strategy,
                           context_collection, context_persist_dir,
                           context_query, context_num_results
                    FROM agents WHERE name = ?
                    """,
                    (name_to_load,),
                ).fetchone()
                if not agent_data:
                    return False
                state_data = conn.execute(
                    "SELECT task, history FROM agent_states WHERE agent_name = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (name_to_load,),
                ).fetchone()
        except sqlite3.Error:
            logger.exception("Error loading agent state for %s", name_to_load)
            return False

        agent.persona = agent_data[0] or ""
        agent.instruction = agent_data[1] or ""
        if agent_data[2]:
            agent.strategy = agent_data[2]
        collection, persist_dir, query, num_results = agent_data[3:7]
        if collection and persist_dir:
            self._restore_context(agent, collection, persist_dir, query, num_results or 3)
        if state_data:
            agent.task = state_data[0] or ""
            agent._history = json.loads(state_data[1]) if state_data[1] else []
        return True

    def _restore_context(
        self, agent: Agent, collection: str, persist_dir: str, query: str | None, n: int
    ) -> None:
        ctx = agent._context
        if ctx is None or ctx.collection_name != collection:
            try:
                ctx = self._context_factory(collection, persist_dir)
            except Exception:
                logger.exception(
                    "Could not rebuild context %s; state loaded without it", collection
                )
                return
            agent.context = ctx
        if query:
            ctx.set_query(query, num_results=n)

    def get_agent_history(self, agent_name: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT task, history, timestamp FROM agent_states WHERE agent_name = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (agent_name, limit),
                ).fetchall()
            return [
                {"task": r[0], "history": json.loads(r[1]) if r[1] else [], "timestamp": r[2]}
                for r in rows
            ]
        except sqlite3.Error:
            logger.exception("Error retrieving history for %s", agent_name)
            return []

    def list_saved_agents(self) -> dict[str, datetime]:
        saved: dict[str, datetime] = {}
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT name, last_updated FROM agents ORDER BY last_updated DESC"
                ).fetchall()
            for name, ts in rows:
                saved[name] = datetime.fromisoformat(ts)
        except sqlite3.Error:
            logger.exception("Error listing saved agents")
        return saved

    def delete_agent_state(self, agent_name: str) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM agents WHERE name = ?", (agent_name,))
            return True
        except sqlite3.Error:
            logger.exception("Error deleting agent %s", agent_name)
            return False

    def cleanup_old_states(self, agent_name: str, keep_last: int = 10) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    DELETE FROM agent_states WHERE agent_name = ? AND id NOT IN (
                        SELECT id FROM agent_states WHERE agent_name = ?
                        ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (agent_name, agent_name, keep_last),
                )
            return True
        except sqlite3.Error:
            logger.exception("Error cleaning up states for %s", agent_name)
            return False
