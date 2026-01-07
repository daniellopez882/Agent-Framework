from agent import Agent

def demonstrate_strategy(agent: Agent, strategy: str):
    """Demonstrate a specific strategy using the agent's current task."""
    print(f"\n=== Testing {strategy} ===")
    agent.strategy = strategy  
    print(f"\nTask: {agent.task}")
    print("\nResponse:")
    response = agent.execute()
    print(response)
    print("\n" + "="*50)

def main():
    # Create a new agent
    agent = Agent("Problem Solver")
    
    # Set the persona
    agent.persona = """You are an analytical problem-solving assistant.
You excel at breaking down complex problems and explaining your thought process.
You are thorough, logical, and clear in your explanations."""

    # Set a global instruction
    agent.instruction = "Ensure your responses are clear, detailed, and well-structured."

    # Define the task that will be used for all demonstrations
    park_planning_task = """A city is planning to build a new park. They have the following constraints:
- Budget: $2 million
- Space: 5 acres
- Must include: playground, walking trails, and parking
- Environmental concerns: preserve existing trees
- Community request: include area for community events

How should they approach this project?"""

    # Print available strategies
    print("Available strategies:", agent.available_strategies())

    # Test each strategy with the same task
    agent.task = park_planning_task
    demonstrate_strategy(agent, "ReactStrategy")
    
    # Clear history and reset task before next strategy
    agent.clear_history()
    agent.task = park_planning_task
    demonstrate_strategy(agent, "ChainOfThoughtStrategy")
    
    # Clear history and reset task before next strategy
    agent.clear_history()
    agent.task = park_planning_task
    demonstrate_strategy(agent, "ReflectionStrategy")

if __name__ == "__main__":
    main()