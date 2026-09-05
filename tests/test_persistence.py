import sqlite3
from datetime import datetime

from agent_framework import Agent, AgentPersistence


class StubCtx:
    def __init__(self, collection_name="col", persist_dir="dir"):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.current_query = "what?"
        self.num_results = 2
        self.response = None
        self.queries = []

    def set_query(self, query, num_results=3, filter_metadata=None):
        self.queries.append((query, num_results))
        return ""

    def clear_index(self):
        return True


def test_round_trip(cfg, persistence):
    a = Agent("rt", persistence, autoload=False)
    a.persona, a.instruction, a.task = "P", "I", "pending"
    a.strategy = "ReflectionStrategy"
    a._history = [{"role": "user", "content": "hi"}]
    assert a.save_state()

    b = Agent("rt", persistence, autoload=False)
    assert b.load_state()
    assert (b.persona, b.instruction, b.task) == ("P", "I", "pending")
    assert type(b.strategy).__name__ == "ReflectionStrategy"
    assert b.history == [{"role": "user", "content": "hi"}]


def test_autoload_on_construction(cfg, persistence):
    a = Agent("auto", persistence, autoload=False)
    a.persona = "loaded"
    a.save_state()
    assert Agent("auto", persistence).persona == "loaded"
    assert Agent("nobody", persistence).persona == ""


def test_latest_state_is_by_id_not_by_second_granularity_timestamp(cfg, persistence):
    a = Agent("ties", persistence, autoload=False)
    a.save_state()
    conn = sqlite3.connect(persistence.db_path)
    # Two snapshots in the same second: only the insertion order tells them apart.
    conn.execute(
        "INSERT INTO agent_states (agent_name, task, history, timestamp) VALUES "
        "('ties', 'older', '[]', '2025-01-01 00:00:00')"
    )
    conn.execute(
        "INSERT INTO agent_states (agent_name, task, history, timestamp) VALUES "
        "('ties', 'newer', '[]', '2025-01-01 00:00:00')"
    )
    conn.commit()
    b = Agent("ties", persistence, autoload=False)
    assert b.load_state()
    assert b.task == "newer"
    hist = persistence.get_agent_history("ties", limit=2)
    assert [h["task"] for h in hist] == ["newer", "older"]


def test_context_is_restored_through_the_factory(cfg, tmp_path):
    built = []

    def factory(collection, persist_dir):
        ctx = StubCtx(collection, persist_dir)
        built.append(ctx)
        return ctx

    p = AgentPersistence(db_path=str(tmp_path / "m.db"), context_factory=factory)
    a = Agent("ctx", p, autoload=False)
    a.context = StubCtx("docs", "/data/docs")
    assert a.save_state()

    b = Agent("ctx", p, autoload=False)
    assert b.load_state()
    assert built and built[0].collection_name == "docs" and built[0].persist_dir == "/data/docs"
    assert built[0].queries == [("what?", 2)]
    assert b._context is built[0]


def test_context_factory_failure_does_not_lose_the_rest_of_the_state(cfg, tmp_path):
    def factory(collection, persist_dir):
        raise RuntimeError("chroma down")

    p = AgentPersistence(db_path=str(tmp_path / "m.db"), context_factory=factory)
    a = Agent("ctx", p, autoload=False)
    a.persona = "kept"
    a.context = StubCtx()
    a.save_state()
    b = Agent("ctx", p, autoload=False)
    assert b.load_state()
    assert b.persona == "kept"
    assert b._context is None


def test_existing_matching_context_is_reused(cfg, tmp_path):
    p = AgentPersistence(db_path=str(tmp_path / "m.db"), context_factory=lambda c, d: StubCtx(c, d))
    a = Agent("ctx", p, autoload=False)
    a.context = StubCtx("same", "x")
    a.save_state()
    mine = StubCtx("same", "x")
    b = Agent("ctx", p, context=mine, autoload=False)
    b.load_state()
    assert b._context is mine and mine.queries == [("what?", 2)]


def test_cleanup_keeps_the_newest_by_id_and_delete_cascades(cfg, persistence):
    a = Agent("clean", persistence, autoload=False)
    for i in range(5):
        a.task = f"t{i}"
        a.save_state()
    assert persistence.cleanup_old_states("clean", keep_last=2)
    assert [h["task"] for h in persistence.get_agent_history("clean")] == ["t4", "t3"]
    assert persistence.delete_agent_state("clean")
    conn = sqlite3.connect(persistence.db_path)
    assert (
        conn.execute("SELECT COUNT(*) FROM agent_states WHERE agent_name='clean'").fetchone()[0]
        == 0
    )
    assert not persistence.load_agent_state(Agent("clean", persistence, autoload=False))


def test_list_saved_agents_and_static_helper(cfg, persistence):
    Agent("one", persistence, autoload=False).save_state()
    saved = persistence.list_saved_agents()
    assert set(saved) == {"one"} and isinstance(saved["one"], datetime)
    assert set(Agent.list_saved_agents(persistence)) == {"one"}


def test_clear_history_variants(cfg, persistence, make_client):
    a = Agent("h", persistence, client=make_client(["a1", "a2", "a3"]), autoload=False)
    for t in ("q1", "q2", "q3"):
        a.execute(t)
    assert len(a.history) == 6
    a.clear_history(keep_last=2)  # reload the newest kept snapshot
    assert a.history[-1] == {"role": "assistant", "content": "a3"}
    a.clear_history()
    assert a.history == []
    assert Agent("h", persistence).history == []
