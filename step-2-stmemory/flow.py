# sample_client.py
from agent import Agent

def main():
    # Create a new agent
    agent = Agent("Research Assistant")
    
    # Set the persona
    agent.persona = """You are a knowledgeable research assistant with expertise in scientific topics.
Your communication style is clear, precise, and academic.
You excel at breaking down complex topics into understandable explanations.
When you're not sure about something, you acknowledge the limitations of your knowledge.
You always provide well-structured, logical responses."""

    # Set a global instruction that will apply to all tasks
    agent.instruction = """Always structure your responses in the following way:
1. Start with a brief overview
2. Provide detailed explanation
3. Include relevant examples when applicable
4. End with a concise summary"""

    # Example interaction chain
    tasks = [
        "What is machine learning?",
        "what is artificial intelligence?",
        "How are these two related?"
    ]

    print(f"Starting interaction with {agent.name}...\n")
    
    for task in tasks:
        print(f"\n>>> Task: {task}")
        print("\nAgent response:")
        response = agent.execute(task)
        print(response)
        print("\n" + "="*50)

    # Demonstrate history
    print("\nConversation History:")
    for message in agent.history:
        print(f"\n{message['role'].upper()}: {message['content'][:100]}...")

    # Clear history demonstration
    print("\nClearing conversation history...")
    agent.clear_history()
    print(f"History length after clearing: {len(agent.history)}")

if __name__ == "__main__":
    main()