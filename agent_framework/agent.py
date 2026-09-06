"""The agent.

The headline defect was in ``execute``. It called the model **once**, then
``_process_tool_usage`` ran whatever tool the reply named and pasted the
result *into the reply text*, which was returned as the answer. The model never
saw a tool result, so a two-hop task ("the capital, then its most popular
dish") ended after the first lookup, and "ReactStrategy" was a prompt, not a
loop. ``execute`` is a loop now: reply -> tool -> observation -> reply, until
the model answers or ``MAX_TOOL_ITERATIONS`` is reached.

Also changed: the model and client are injectable (the model name was a
constant and a new ``OpenAI()`` was built per call); failures raise instead
of being returned as strings; and setting a property no longer writes a
snapshot row to SQLite — state is saved when a task completes, on ``pause``,
and on ``save_state``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from agent_framework.config import settings
from agent_framework.context import ContextManager
from agent_framework.errors import AgentError, ConfigurationError, ToolLoopExceeded
from agent_framework.persistence import AgentPersistence
from agent_framework.strategy import ExecutionStrategy, StrategyFactory, extract_final_answer
from agent_framework.tools import Tool, ToolRegistry, ToolResult, parse_tool_usage

logger = logging.getLogger(__name__)


class Agent:
    """A persona-driven agent with memory, a prompting strategy, RAG context and tools."""

    def __init__(
        self,
        name: str,
        persistence: AgentPersistence | None = None,
        context: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        *,
        model: str | None = None,
        client: Any = None,
        max_iterations: int | None = None,
        autoload: bool = True,
    ) -> None:
        self._name = name
        self._persona = ""
        self._instruction = ""
        self._task = ""
        self._model = model or settings.OPENAI_MODEL
        self._client = client
        self._max_iterations = max_iterations or settings.MAX_TOOL_ITERATIONS
        self._history: list[dict[str, str]] = []
        self._strategy: ExecutionStrategy | None = None
        self._persistence = persistence or AgentPersistence()
        self._context = context
        self._tool_registry = tool_registry or ToolRegistry()
        if autoload:
            self._persistence.load_agent_state(self)
        logger.info("Initialized agent %s", name)

    # -- configuration -------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def persona(self) -> str:
        return self._persona

    @persona.setter
    def persona(self, value: str) -> None:
        self._persona = value

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str) -> None:
        self._instruction = value

    @property
    def task(self) -> str:
        return self._task

    @task.setter
    def task(self, value: str) -> None:
        self._task = value

    @property
    def model(self) -> str:
        return self._model

    @property
    def tools(self) -> list[Tool]:
        return self._tool_registry.tools

    @tools.setter
    def tools(self, tools: Sequence[Tool]) -> None:
        self._tool_registry = ToolRegistry()
        for tool in tools or []:
            self._tool_registry.register(tool)
            logger.info("Registered tool: %s", tool.name)

    @property
    def strategy(self) -> ExecutionStrategy | None:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_name: str) -> None:
        self._strategy = StrategyFactory.create_strategy(strategy_name)

    @property
    def history(self) -> list[dict[str, str]]:
        return self._history

    @property
    def context(self) -> str | None:
        return self._context.response if self._context else None

    @context.setter
    def context(self, context_manager: ContextManager | None) -> None:
        self._context = context_manager

    def get_available_tools(self) -> list[str]:
        return self._tool_registry.list_tools()

    def get_tools_prompt(self) -> str:
        return self._tool_registry.get_tools_prompt()

    def available_strategies(self) -> list[str]:
        return StrategyFactory.available_strategies()

    # -- execution -----------------------------------------------------------

    def execute_tool(self, tool_name: str, **parameters: Any) -> ToolResult:
        tool = self._tool_registry.get_tool(tool_name)
        if not tool:
            logger.error("Tool not found: %s", tool_name)
            return ToolResult(success=False, data=None, error=f"Tool not found: {tool_name}")
        try:
            result = tool.execute(**parameters)
        except Exception as e:  # a tool must not take the agent down
            logger.exception("Tool %s failed", tool_name)
            return ToolResult(success=False, data=None, error=f"Tool execution failed: {e}")
        logger.info("Tool %s -> success=%s", tool_name, result.success)
        return result

    def _build_messages(self, task: str | None = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.persona}]
        if self.instruction:
            messages.append({"role": "user", "content": f"Global Instruction: {self.instruction}"})
        if self._tool_registry.list_tools():
            messages.append({"role": "system", "content": self.get_tools_prompt()})
        if self.context:
            messages.append({"role": "user", "content": f"Relevant Context:\n{self.context}"})
        messages.extend(self._history)
        current = task if task is not None else self._task
        if self._strategy and current:
            current = self._strategy.build_prompt(current, self.instruction)
        if current:
            messages.append({"role": "user", "content": current})
        return messages

    def _get_client(self) -> Any:
        if self._client is None:
            if not settings.openai_configured:
                raise ConfigurationError("OPENAI_API_KEY is not set")
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY, timeout=settings.OPENAI_TIMEOUT_SECONDS
            )
        return self._client

    def _complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=self._model, messages=messages
            )
        except AgentError:
            raise
        except Exception as e:
            raise AgentError(f"Model call failed: {e}") from e
        content = response.choices[0].message.content
        return content or ""

    def execute(self, task: str | None = None) -> str:
        """Run the task, calling tools in a loop, and return the final answer."""
        if task is not None:
            self._task = task
        if not self._task:
            raise AgentError("No task specified")

        messages = self._build_messages()
        answer: str | None = None
        for _ in range(self._max_iterations):
            content = self._complete(messages)
            if self._strategy:
                content = self._strategy.process_response(content)
            call = parse_tool_usage(content) if self._tool_registry.list_tools() else None
            if call is None:
                answer = content
                break
            result = self.execute_tool(call.name, **call.parameters)
            observation = (
                f"Observation ({call.name}): {result.data}"
                if result.success
                else f"Observation ({call.name}) failed: {result.error}"
            )
            logger.debug("%s", observation)
            messages.append({"role": "assistant", "content": content[: call.end].rstrip()})
            messages.append({"role": "user", "content": observation})
        if answer is None:
            raise ToolLoopExceeded(
                f"No final answer after {self._max_iterations} tool calls "
                "(raise MAX_TOOL_ITERATIONS if the task needs more)"
            )

        final = extract_final_answer(answer)
        answer = final if final is not None else answer.strip()
        self._history.append({"role": "user", "content": self._task})
        self._history.append({"role": "assistant", "content": answer})
        self._task = ""
        self.save_state()
        return answer

    # -- persistence ---------------------------------------------------------

    def save_state(self) -> bool:
        return self._persistence.save_agent_state(self)

    def load_state(self, agent_name: str | None = None) -> bool:
        return self._persistence.load_agent_state(self, agent_name)

    def clear_history(self, keep_last: int = 0) -> None:
        if keep_last > 0:
            self._persistence.cleanup_old_states(self.name, keep_last)
            self.load_state()
        else:
            self._history = []
            self.save_state()

    def pause(self) -> bool:
        return self.save_state()

    def resume(self, agent_name: str | None = None) -> bool:
        return self.load_state(agent_name)

    def get_history_states(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._persistence.get_agent_history(self.name, limit)

    def delete_agent(self) -> bool:
        if self._context:
            self._context.clear_index()
        return self._persistence.delete_agent_state(self.name)

    @staticmethod
    def list_saved_agents(persistence: AgentPersistence | None = None) -> dict[str, datetime]:
        return (persistence or AgentPersistence()).list_saved_agents()
