# flow.py
from agent import Agent
from context import ContextManager, DocumentMetadata

def main():
    # 1. Initialize context and document
    context = ContextManager.initialize(
        collection_name="simple_docs",
        persist_dir="context_db"
    )
    
    # 2. Create and configure agent
    agent = Agent("rag_agent")
    
    # Set agent configuration
    agent.persona = """You are a helpful AI assistant that provides accurate information 
    based on the given context. You analyze documents and explain complex topics clearly."""
    
    agent.instruction = """When explaining concepts:
    1. Focus on key principles and fundamentals
    2. Use clear and precise language
    3. Provide relevant examples where applicable"""
    
    agent.strategy = "ReactStrategy"
    
    # 3. Query context directly through context manager
    context.set_query("What are the main principles of quantum computing?")
    
    # 4. Set task, context, and execute
    agent.task = "Identify and explain the key principles of quantum computing"
    agent.context=context
    response = agent.execute()
    
    print("\nAgent Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main()