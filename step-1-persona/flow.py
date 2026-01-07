from agent import Agent

if __name__ == "__main__":
    # Create a financial advisor agent
    financial_advisor = Agent("FinancialAdvisorBot")
    
    financial_advisor.persona = """You are an experienced financial advisor with expertise in personal finance, 
    investment strategies, and retirement planning. Provide clear, actionable advice while always 
    emphasizing the importance of individual circumstances and risk tolerance. Never recommend 
    specific stocks or make promises about returns. Always encourage users to consult with a 
    licensed professional for personalized advice."""
    
    # Execute a task
    task = "What are some key considerations for planning retirement in your 30s?"
    response = financial_advisor.execute(task)
    
    print(f"Agent Name: {financial_advisor.name}")
    print(f"Agent Persona: {financial_advisor.persona}")
    print(f"\nTask: {task}")
    print(f"Agent Response:\n{response}")
 
    # Execute another task with the same persona
    task = "Explain the pros and cons of index fund investing for a beginner"
    response = financial_advisor.execute(task)
    
    print(f"\nNew Task: {task}")
    print(f"Agent Response:\n{response}")