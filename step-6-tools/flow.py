from agent import Agent
from wikipedia_tool import WikipediaTool
from websearch_tool import WebSearchTool
from persistence import AgentPersistence
import time

persistence = AgentPersistence()

agent = Agent("test_agent", persistence)
wiki_tool = WikipediaTool()
web_search_tool = WebSearchTool()

agent.persona = "You are a helpful assistant that can explain concepts clearly."
agent.instruction = "When explaining topics, break them down into key points and provide relevant examples. You have access to tools which you can use to find information. Ensure you are using all the available tools before arriving at the final answer."
agent.task = "Find out what is the capital of Telangana and then the most popular dish in that city."
agent.strategy = "ReactStrategy"
agent.tools = [wiki_tool, web_search_tool]

response = agent.execute()
print("\nResponse:")
print(response)
