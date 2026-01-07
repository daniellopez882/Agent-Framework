import time
from agent import Agent
from persistence import AgentPersistence

def main():

    persistence = AgentPersistence()    
    agent = Agent("research_assistant", persistence)
    agent.persona = "You a helpful research assistant with expertise in analyzing scientific papers."
    agent.instruction = "Always provide concise, evidence-based responses."
    agent.strategy = "ChainOfThoughtStrategy"
    print("Executing initial task...")
    result = agent.execute("Summarize the key benefits of quantum computing.")
    print(f"Initial task result: {result}")
    
    time.sleep(20)
    
    if agent.pause():
        print("Agent paused successfully")
    
    agent.task = "What are the current challenges in quantum computing?"
    new_agent = Agent("research_assistant", persistence)
    print("\nResuming agent state...")
    new_agent.resume()
    
    # Execute the pending task
    print("\nExecuting pending task...")
    result = new_agent.execute()
    print(f"Task result after resume: {result}")

    print("\nListing all saved agents:")
    saved_agents = persistence.list_saved_agents()
    for name, timestamp in saved_agents.items():
        print(f"Agent: {name}, Last saved: {timestamp}")
    
if __name__ == "__main__":
    main()
