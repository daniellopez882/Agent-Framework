"""Tools: the base class, the registry, and the parser for the model's tool calls.

Two defects lived here.

``parse_tool_usage`` took its parameters from *every* line of the reply that
contained both ``:`` and ``-`` — so ``Thought: first - then second`` became a
parameter named ``Thought``, and a URL from an earlier result became another.
It also returned no position, so the agent re-found the block with
``str.find("Tool:")`` and cut it at the end of the ``Parameters:`` line, leaving
the ``- query: ...`` lines dangling in the answer. The parser now returns a
``ToolCall`` with the exact span of the block, and only the ``- name: value``
lines under ``Parameters:`` are parameters.

``Tool.parameters`` was declared ``Dict[str, str]`` while both shipped tools
returned ``{"query": {"type": ..., "description": ...}}``, so the prompt showed
the model a Python dict repr. The renderer accepts both shapes.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_TOOL_LINE = re.compile(r"^\s*Tool:\s*(.+?)\s*$")
_PARAMS_LINE = re.compile(r"^\s*Parameters:\s*$")
_PARAM_LINE = re.compile(r"^\s*-\s*([^:]+?)\s*:\s*(.*?)\s*$")


@dataclass
class ToolResult:
    """The outcome of one tool execution."""

    success: bool
    data: Any
    error: str | None = None


@dataclass
class ToolCall:
    """A tool invocation the model wrote, with its span in the reply."""

    name: str
    parameters: dict[str, str] = field(default_factory=dict)
    start: int = 0
    end: int = 0


class Tool(ABC):
    """Base class for tools the agent can call."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The tool's name, as the model must write it."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Parameter name -> description (a dict with a ``description`` key is also accepted)."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool."""

    def to_prompt_format(self) -> str:
        """Render the tool for the system prompt."""
        lines = []
        for pname, desc in self.parameters.items():
            text = desc.get("description", str(desc)) if isinstance(desc, dict) else str(desc)
            lines.append(f"  - {pname}: {text}")
        params = "\n".join(lines) if lines else "  (none)"
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters:\n{params}"


class ToolRegistry:
    """The tools available to one agent, by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Tool must be an instance of Tool")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tools_prompt(self) -> str:
        if not self._tools:
            return "No tools available."
        rendered = "\n\n".join(t.to_prompt_format() for t in self._tools.values())
        return (
            "Available Tools:\n\n"
            f"{rendered}\n\n"
            "To use a tool, write exactly this block and then stop; the result will be "
            "given to you as an Observation:\n"
            "Tool: <tool_name>\n"
            "Parameters:\n"
            "  - <param>: <value>\n"
        )


def parse_tool_usage(response: str) -> ToolCall | None:
    """Return the first tool call in ``response``, or ``None``.

    The block is a ``Tool: <name>`` line, optionally followed by a
    ``Parameters:`` line and then ``- name: value`` lines. Parsing stops at the
    first line that is not a parameter line, so prose before or after the block
    is never mistaken for a parameter. Brackets around the name (``Tool:
    [wikipedia_search]``, which models copy from the template) are stripped.
    """
    if "Tool:" not in response:
        return None
    lines = response.split("\n")
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1

    for i, line in enumerate(lines):
        m = _TOOL_LINE.match(line)
        if not m:
            continue
        name = m.group(1).strip().strip("[]`*").strip()
        if not name:
            continue
        start = offsets[i]
        end = offsets[i] + len(line)
        params: dict[str, str] = {}
        j = i + 1
        if j < len(lines) and _PARAMS_LINE.match(lines[j]):
            end = offsets[j] + len(lines[j])
            j += 1
            while j < len(lines):
                pm = _PARAM_LINE.match(lines[j])
                if not pm:
                    break
                params[pm.group(1).strip("[]`*").strip()] = pm.group(2).strip("[]`*").strip()
                end = offsets[j] + len(lines[j])
                j += 1
        return ToolCall(name=name, parameters=params, start=start, end=end)
    return None
