"""Step 4 — pause an agent, resume it in a new instance. Needs OPENAI_API_KEY.

State is written when a task completes, on ``pause()`` and on ``save_state()``
— not on every property assignment — so the pending task is saved explicitly.
"""

from agent_framework import Agent, AgentPersistence


def main() -> None:
    persistence = AgentPersistence()

    agent = Agent("research_assistant", persistence)
    agent.persona = "You are a helpful research assistant with expertise in scientific papers."
    agent.instruction = "Always provide concise, evidence-based responses."
    agent.strategy = "ChainOfThoughtStrategy"
    print(agent.execute("Summarize the key benefits of quantum computing."))

    agent.task = "What are the current challenges in quantum computing?"
    agent.pause()  # persists persona, instruction, strategy, history and the pending task

    resumed = Agent("research_assistant", persistence)  # loads the saved state
    print("\nResumed with pending task:", resumed.task)
    print(resumed.execute())

    print("\nSaved agents:")
    for name, ts in persistence.list_saved_agents().items():
        print(f"  {name}: last saved {ts}")


if __name__ == "__main__":
    main()
