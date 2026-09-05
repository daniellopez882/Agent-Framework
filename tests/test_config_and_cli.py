import pytest
from pydantic import ValidationError

from agent_framework.__main__ import main
from agent_framework.config import DEFAULT_MODEL, Settings


def test_defaults(monkeypatch):
    for k in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "TAVILY_API_KEY",
        "MAX_TOOL_ITERATIONS",
        "AGENT_DB_PATH",
        "CONTEXT_PERSIST_DIR",
    ):
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.OPENAI_MODEL == DEFAULT_MODEL
    assert s.MAX_TOOL_ITERATIONS == 5
    assert s.AGENT_DB_PATH == "agent_memory.db"
    assert not s.openai_configured and not s.web_search_configured


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "9")
    s = Settings(_env_file=None)
    assert s.OPENAI_MODEL == "gpt-x" and s.MAX_TOOL_ITERATIONS == 9
    assert s.openai_configured and s.web_search_configured


@pytest.mark.parametrize("value", ["0", "26"])
def test_iteration_bounds(monkeypatch, value):
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_check_command(cfg, capsys):
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "model: gpt-4o-mini" in out and "openai_api_key: set" in out
    assert "web_search: disabled" in out and "ReactStrategy" in out


def test_check_strict_fails_without_key(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "")
    assert main(["check", "--strict"]) == 1
    assert "OPENAI_API_KEY is required" in capsys.readouterr().err
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "k")
    assert main(["check", "--strict"]) == 0
