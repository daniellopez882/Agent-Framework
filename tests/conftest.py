import os

os.environ["AGENT_FRAMEWORK_NO_ENV_FILE"] = "1"

import copy  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from agent_framework.config import settings  # noqa: E402
from agent_framework.persistence import AgentPersistence  # noqa: E402
from agent_framework.tools import Tool, ToolResult  # noqa: E402


class StubCompletions:
    """Stands in for ``client.chat.completions``; replies in order, records requests."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        # Snapshot: the agent keeps appending to the same `messages` list.
        self.requests.append({**kwargs, "messages": copy.deepcopy(kwargs.get("messages", []))})
        if not self.replies:
            raise AssertionError("stub client has no more replies")
        content = self.replies.pop(0)

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content  # type: ignore[attr-defined]
        choice = _Msg()
        choice.message = msg  # type: ignore[attr-defined]
        resp = _Msg()
        resp.choices = [choice]  # type: ignore[attr-defined]
        return resp


class StubClient:
    def __init__(self, replies: list[str]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = StubCompletions(replies)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.chat.completions.requests


class RecordingTool(Tool):
    def __init__(self, name: str, reply: str = "ok", fail: bool = False, raise_: bool = False):
        self._name = name
        self._reply = reply
        self._fail = fail
        self._raise = raise_
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"query": "what to look up"}

    def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if self._raise:
            raise RuntimeError("boom")
        if self._fail:
            return ToolResult(success=False, data="", error="nope")
        return ToolResult(success=True, data=self._reply)


@pytest.fixture
def cfg(monkeypatch):
    """The shared settings object with a key set; tests override attributes as needed."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")
    monkeypatch.setattr(settings, "MAX_TOOL_ITERATIONS", 5)
    return settings


@pytest.fixture
def persistence(tmp_path):
    return AgentPersistence(db_path=str(tmp_path / "mem.db"))


@pytest.fixture
def make_client():
    return StubClient


@pytest.fixture
def recording_tool():
    return RecordingTool
