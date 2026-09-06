"""Step 2 — conversation memory across tasks. Needs OPENAI_API_KEY."""

from agent_framework import Agent


def main() -> None:
    agent = Agent("Research Assistant")
    agent.persona = (
        "You are a knowledgeable research assistant with expertise in scientific topics. "
        "Your communication style is clear, precise, and academic. When you're not sure "
        "about something, you acknowledge the limitations of your knowledge."
    )
    agent.instruction = (
        "Always structure your responses: a brief overview, a detailed explanation, "
        "relevant examples, and a concise summary."
    )
    for task in (
        "What is machine learning?",
        "What is artificial intelligence?",
        "How are these two related?",  # answered from history
    ):
        print(f"\n>>> {task}\n")
        print(agent.execute(task))

    print("\nHistory:")
    for m in agent.history:
        print(f"  {m['role'].upper()}: {m['content'][:80]}...")
    agent.clear_history()
    print(f"History length after clearing: {len(agent.history)}")


if __name__ == "__main__":
    main()
