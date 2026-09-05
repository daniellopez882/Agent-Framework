from typing import Any

import pytest

from agent_framework.tools import Tool, ToolRegistry, ToolResult, parse_tool_usage


class DictParamsTool(Tool):
    """The shape the shipped tools used: nested dicts, not name -> description."""

    @property
    def name(self) -> str:
        return "dicty"

    @property
    def description(self) -> str:
        return "renders dict-shaped params"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"query": {"type": "string", "description": "The search query"}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="")


class NoParamsTool(DictParamsTool):
    @property
    def name(self) -> str:
        return "noparams"

    @property
    def parameters(self) -> dict[str, Any]:
        return {}


# --- parsing -----------------------------------------------------------------


def test_parses_name_and_parameters():
    call = parse_tool_usage("Tool: wikipedia_search\nParameters:\n  - query: Telangana\n")
    assert call is not None
    assert call.name == "wikipedia_search"
    assert call.parameters == {"query": "Telangana"}


def test_prose_lines_with_colon_and_dash_are_not_parameters():
    # The old parser took every ':'+'-' line as a parameter: 'Thought' and 'URL' leaked in.
    text = (
        "Thought: first - then second\n"
        "Earlier result URL: https://en.wikipedia.org/wiki/Hyderabad-Deccan\n"
        "Tool: wikipedia_search\n"
        "Parameters:\n"
        "  - query: Telangana\n"
        "Then I will answer - later: maybe\n"
    )
    call = parse_tool_usage(text)
    assert call is not None
    assert call.parameters == {"query": "Telangana"}


def test_span_covers_the_whole_block():
    # The old agent cut the block at the end of the 'Parameters:' line, leaving '- query:' behind.
    text = "Thought: need it\nTool: t\nParameters:\n  - query: x\n  - lang: en\nFinal Answer: no"
    call = parse_tool_usage(text)
    assert call is not None
    block = text[call.start : call.end]
    assert block == "Tool: t\nParameters:\n  - query: x\n  - lang: en"
    assert text[: call.start].endswith("need it\n")


def test_brackets_from_the_template_are_stripped():
    call = parse_tool_usage("Tool: [wikipedia_search]\nParameters:\n  - [query]: [Telangana]")
    assert call is not None
    assert call.name == "wikipedia_search"
    assert call.parameters == {"query": "Telangana"}


def test_tool_line_without_parameters_block():
    call = parse_tool_usage("Tool: ping\nsome prose")
    assert call is not None
    assert call.name == "ping"
    assert call.parameters == {}
    assert call.end == len("Tool: ping")


@pytest.mark.parametrize("text", ["", "no tools here", "Tool Result (x): done", "Tool:   \n"])
def test_no_call(text):
    assert parse_tool_usage(text) is None


def test_first_call_only():
    call = parse_tool_usage("Tool: a\nParameters:\n  - q: 1\n\nTool: b\nParameters:\n  - q: 2")
    assert call is not None
    assert call.name == "a"
    assert call.parameters == {"q": "1"}


# --- rendering -----------------------------------------------------------------


def test_dict_shaped_parameters_render_their_description_not_a_repr():
    rendered = DictParamsTool().to_prompt_format()
    assert "  - query: The search query" in rendered
    assert "{'type'" not in rendered


def test_string_parameters_and_none():
    tool = DictParamsTool()
    assert "Tool: dicty" in tool.to_prompt_format()
    assert "(none)" in NoParamsTool().to_prompt_format()


# --- registry -----------------------------------------------------------------


def test_registry_rejects_non_tools():
    with pytest.raises(TypeError):
        ToolRegistry().register("not a tool")  # type: ignore[arg-type]


def test_registry_lists_gets_and_prompts():
    reg = ToolRegistry()
    assert reg.get_tools_prompt() == "No tools available."
    t = DictParamsTool()
    reg.register(t)
    assert reg.list_tools() == ["dicty"]
    assert reg.get_tool("dicty") is t
    assert reg.get_tool("missing") is None
    assert reg.tools == [t]
    prompt = reg.get_tools_prompt()
    assert "Tool: <tool_name>" in prompt and "Observation" in prompt
