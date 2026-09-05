"""Step 1 — a persona. Needs OPENAI_API_KEY."""

from agent_framework import Agent


def main() -> None:
    advisor = Agent("FinancialAdvisorBot")
    advisor.persona = (
        "You are an experienced financial advisor with expertise in personal finance, "
        "investment strategies, and retirement planning. Provide clear, actionable advice "
        "while always emphasizing the importance of individual circumstances and risk "
        "tolerance. Never recommend specific stocks or make promises about returns. Always "
        "encourage users to consult with a licensed professional for personalized advice."
    )
    for task in (
        "What are some key considerations for planning retirement in your 30s?",
        "Explain the pros and cons of index fund investing for a beginner",
    ):
        print(f"\nTask: {task}\n")
        print(advisor.execute(task))


if __name__ == "__main__":
    main()
