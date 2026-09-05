"""Prompting strategies.

Ported from the step directories (three byte-identical copies). One change:
the ReAct prompt no longer asks the model to write its own ``Observation:`` —
with a real tool loop, the observation comes from the tool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

FINAL_ANSWER = "Final Answer:"


def extract_final_answer(text: str) -> str | None:
    """Return the text after the last ``Final Answer:`` marker, or ``None``."""
    idx = text.rfind(FINAL_ANSWER)
    if idx < 0:
        return None
    return text[idx + len(FINAL_ANSWER) :].strip()


class ExecutionStrategy(ABC):
    @abstractmethod
    def build_prompt(self, task: str, instruction: str | None = None) -> str:
        """Build the task prompt."""

    @abstractmethod
    def process_response(self, response: str) -> str:
        """Post-process a model reply."""


class ReactStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: str | None = None) -> str:
        base = (
            "Approach this task by reasoning and acting in turns:\n"
            "Thought: what you know and what you still need\n"
            "Action: if you need a tool, write its Tool/Parameters block and stop; "
            "the Observation will be given to you\n"
            "Repeat until you can answer, then write:\n"
            f"{FINAL_ANSWER} [your answer]\n\n"
            f"Task: {task}"
        )
        if instruction:
            base += f"\nAdditional Instruction: {instruction}"
        return base

    def process_response(self, response: str) -> str:
        return response


class ChainOfThoughtStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: str | None = None) -> str:
        base = (
            "Let's solve this step by step:\n\n"
            f"Task: {task}\n\n"
            "Please break down your thinking into clear steps:\n"
            "1) First, ...\n"
            "2) Then, ...\n"
            "(continue with your step-by-step reasoning)\n\n"
            f"{FINAL_ANSWER} [Your conclusion based on the above reasoning]"
        )
        if instruction:
            base += f"\nAdditional Instruction: {instruction}"
        return base

    def process_response(self, response: str) -> str:
        return response


class ReflectionStrategy(ExecutionStrategy):
    def build_prompt(self, task: str, instruction: str | None = None) -> str:
        base = (
            "Complete this task using reflection:\n\n"
            f"Task: {task}\n\n"
            "1) Initial Approach:\n"
            "   - What is your first impression of how to solve this?\n"
            "   - What assumptions are you making?\n\n"
            "2) Analysis:\n"
            "   - What could go wrong with your initial approach?\n"
            "   - What alternative approaches could you consider?\n\n"
            "3) Refined Solution:\n"
            "   - Based on your reflection, what is the best approach?\n"
            "   - Why is this approach better than the alternatives?\n\n"
            f"4) {FINAL_ANSWER}\n"
            "   - Provide your solution\n"
            "   - Briefly explain why this is the optimal approach"
        )
        if instruction:
            base += f"\nAdditional Instruction: {instruction}"
        return base

    def process_response(self, response: str) -> str:
        return response


class StrategyFactory:
    _strategies: dict[str, type[ExecutionStrategy]] = {
        "ReactStrategy": ReactStrategy,
        "ChainOfThoughtStrategy": ChainOfThoughtStrategy,
        "ReflectionStrategy": ReflectionStrategy,
    }

    @classmethod
    def create_strategy(cls, strategy_name: str) -> ExecutionStrategy:
        strategy_class = cls._strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_class()

    @classmethod
    def available_strategies(cls) -> list[str]:
        return list(cls._strategies)
