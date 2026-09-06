import pytest

from agent_framework.strategy import (
    FINAL_ANSWER,
    ChainOfThoughtStrategy,
    ReactStrategy,
    ReflectionStrategy,
    StrategyFactory,
    extract_final_answer,
)


def test_factory():
    assert StrategyFactory.available_strategies() == [
        "ReactStrategy",
        "ChainOfThoughtStrategy",
        "ReflectionStrategy",
    ]
    assert isinstance(StrategyFactory.create_strategy("ReactStrategy"), ReactStrategy)
    with pytest.raises(ValueError, match="Unknown strategy"):
        StrategyFactory.create_strategy("Nope")


@pytest.mark.parametrize("cls", [ReactStrategy, ChainOfThoughtStrategy, ReflectionStrategy])
def test_prompts_carry_task_instruction_and_marker(cls):
    prompt = cls().build_prompt("do the thing", "be brief")
    assert "do the thing" in prompt
    assert "Additional Instruction: be brief" in prompt
    assert FINAL_ANSWER in prompt
    assert cls().process_response("x") == "x"
    assert "Additional Instruction" not in cls().build_prompt("t")


def test_react_prompt_does_not_ask_the_model_to_invent_observations():
    prompt = ReactStrategy().build_prompt("t")
    assert "Observation will be given" in prompt
    assert "Observation: [What you observe" not in prompt


def test_extract_final_answer():
    assert extract_final_answer("no marker") is None
    assert extract_final_answer("Thought: x\nFinal Answer: 42") == "42"
    assert extract_final_answer("Final Answer: a\n...\nFinal Answer:  b ") == "b"
    assert extract_final_answer("Final Answer:") == ""
