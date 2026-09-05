import sqlite3

import pytest

from agent_framework import Agent, AgentError, ConfigurationError, ToolLoopExceeded

WIKI_CALL = (
    "Thought: I need the capital.\nTool: wikipedia_search\nParameters:\n  - query: Telangana\n"
)
WEB_CALL = (
    "Thought: now the dish.\nTool: web_search\nParameters:\n"
    "  - query: most popular dish Hyderabad\n"
)
FINAL = "Thought: done.\nFinal Answer: Hyderabad; its most popular dish is biryani."


def make_agent(persistence, client, tools=(), **kw):
    a = Agent("t", persistence, client=client, autoload=False, **kw)
    a.tools = list(tools)
    a.strategy = "ReactStrategy"
    return a


def test_two_hop_task_loops_until_final_answer(cfg, persistence, make_client, recording_tool):
    wiki = recording_tool(
        "wikipedia_search", reply="Title: Hyderabad\nSummary: capital of Telangana"
    )
    web = recording_tool("web_search", reply="1. Hyderabadi biryani")
    client = make_client([WIKI_CALL, WEB_CALL, FINAL])
    agent = make_agent(persistence, client, [wiki, web])

    answer = agent.execute("Capital of Telangana, then its most popular dish?")

    assert answer == "Hyderabad; its most popular dish is biryani."
    assert len(client.requests) == 3
    assert wiki.calls == [{"query": "Telangana"}]
    assert web.calls == [{"query": "most popular dish Hyderabad"}]
    # The model saw the first tool's result before deciding the second call.
    second = client.requests[1]["messages"]
    assert second[-1] == {
        "role": "user",
        "content": (
            "Observation (wikipedia_search): Title: Hyderabad\nSummary: capital of Telangana"
        ),
    }
    assert second[-2]["role"] == "assistant" and second[-2]["content"].endswith(
        "- query: Telangana"
    )
    # Nothing from the tool block leaks into the answer.
    assert "- query:" not in answer and "Tool:" not in answer


def test_history_and_state_after_execute(cfg, persistence, make_client):
    client = make_client(["Final Answer: 42"])
    agent = make_agent(persistence, client)
    agent.execute("The answer?")
    assert agent.history == [
        {"role": "user", "content": "The answer?"},
        {"role": "assistant", "content": "42"},
    ]
    assert agent.task == ""
    rows = (
        sqlite3.connect(persistence.db_path)
        .execute("SELECT COUNT(*) FROM agent_states WHERE agent_name='t'")
        .fetchone()[0]
    )
    assert rows == 1


def test_setters_do_not_snapshot_state(cfg, persistence, make_client):
    agent = make_agent(persistence, make_client([]))
    agent.persona = "p"
    agent.instruction = "i"
    agent.task = "t"
    agent.strategy = "ChainOfThoughtStrategy"
    rows = (
        sqlite3.connect(persistence.db_path)
        .execute("SELECT COUNT(*) FROM agent_states")
        .fetchone()[0]
    )
    assert rows == 0


def test_iteration_cap_raises(cfg, persistence, make_client, recording_tool):
    tool = recording_tool("wikipedia_search")
    client = make_client([WIKI_CALL] * 10)
    agent = make_agent(persistence, client, [tool], max_iterations=2)
    with pytest.raises(ToolLoopExceeded):
        agent.execute("loop forever")
    assert len(client.requests) == 2
    assert len(tool.calls) == 2


def test_unknown_tool_becomes_an_observation_and_the_loop_continues(
    cfg, persistence, make_client, recording_tool
):
    client = make_client(["Tool: nope\nParameters:\n  - query: x\n", FINAL])
    agent = make_agent(persistence, client, [recording_tool("wikipedia_search")])
    agent.execute("go")
    obs = client.requests[1]["messages"][-1]["content"]
    assert obs.startswith("Observation (nope) failed: Tool not found: nope")


def test_tool_exception_is_contained(cfg, persistence, make_client, recording_tool):
    bad = recording_tool("wikipedia_search", raise_=True)
    client = make_client([WIKI_CALL, FINAL])
    agent = make_agent(persistence, client, [bad])
    agent.execute("go")
    obs = client.requests[1]["messages"][-1]["content"]
    assert "failed: Tool execution failed: boom" in obs


def test_tool_failure_result_is_reported(cfg, persistence, make_client, recording_tool):
    client = make_client([WIKI_CALL, FINAL])
    agent = make_agent(persistence, client, [recording_tool("wikipedia_search", fail=True)])
    agent.execute("go")
    assert (
        client.requests[1]["messages"][-1]["content"]
        == "Observation (wikipedia_search) failed: nope"
    )


def test_without_tools_a_reply_mentioning_tool_is_just_text(cfg, persistence, make_client):
    client = make_client(["Tool: hammer\nParameters:\n  - size: big"])
    agent = make_agent(persistence, client)
    assert agent.execute("what tool?") == "Tool: hammer\nParameters:\n  - size: big"
    assert len(client.requests) == 1


def test_reply_without_final_marker_is_the_answer(cfg, persistence, make_client):
    agent = make_agent(persistence, make_client(["  plain reply  "]))
    assert agent.execute("hi") == "plain reply"


def test_errors_are_exceptions_not_strings(cfg, persistence, make_client):
    agent = make_agent(persistence, make_client([]))
    with pytest.raises(AgentError, match="No task"):
        agent.execute()

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("rate limited")

    agent = make_agent(persistence, Boom())
    with pytest.raises(AgentError, match="rate limited"):
        agent.execute("x")


def test_missing_api_key_is_a_configuration_error(cfg, persistence, monkeypatch):
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "")
    agent = Agent("t", persistence, autoload=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        agent.execute("x")


def test_model_is_configurable_and_sent(cfg, persistence, make_client):
    client = make_client(["ok"])
    agent = make_agent(persistence, client, model="my-model")
    agent.execute("x")
    assert client.requests[0]["model"] == "my-model"
    assert agent.model == "my-model"


def test_messages_include_persona_instruction_tools_context_history(
    cfg, persistence, make_client, recording_tool
):
    class Ctx:
        response = "doc chunk"
        collection_name = "c"
        persist_dir = "d"
        current_query = "q"

    client = make_client(["first", "second"])
    agent = make_agent(persistence, client, [recording_tool("wikipedia_search")])
    agent.persona = "P"
    agent.instruction = "I"
    agent.context = Ctx()
    agent.execute("one")
    agent.execute("two")
    msgs = client.requests[1]["messages"]
    roles = [m["role"] for m in msgs]
    assert msgs[0] == {"role": "system", "content": "P"}
    assert msgs[1]["content"] == "Global Instruction: I"
    assert "Available Tools" in msgs[2]["content"]
    assert msgs[3]["content"] == "Relevant Context:\ndoc chunk"
    assert {"role": "user", "content": "one"} in msgs and {
        "role": "assistant",
        "content": "first",
    } in msgs
    assert roles[-1] == "user" and "two" in msgs[-1]["content"]
