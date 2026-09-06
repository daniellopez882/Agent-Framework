"""Step 3 — the three prompting strategies on one task. Needs OPENAI_API_KEY."""

from agent_framework import Agent

TASK = """A city is planning to build a new park. They have the following constraints:
- Budget: $2 million
- Space: 5 acres
- Must include: playground, walking trails, and parking
- Environmental concerns: preserve existing trees
- Community request: include area for community events

How should they approach this project?"""


def main() -> None:
    agent = Agent("Problem Solver")
    agent.persona = (
        "You are an analytical problem-solving assistant. You excel at breaking down "
        "complex problems and explaining your thought process."
    )
    agent.instruction = "Ensure your responses are clear, detailed, and well-structured."
    print("Available strategies:", agent.available_strategies())
    for strategy in agent.available_strategies():
        agent.clear_history()
        agent.strategy = strategy
        print(f"\n=== {strategy} ===\n")
        print(agent.execute(TASK))


if __name__ == "__main__":
    main()
