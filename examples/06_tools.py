"""Step 6 — tools in a real reason/act loop. Needs OPENAI_API_KEY; TAVILY_API_KEY optional.

A two-hop task: the second lookup depends on the first result, which the model
now actually sees (see README, "What changed").
"""

import logging

from agent_framework import Agent
from agent_framework.websearch_tool import WebSearchTool
from agent_framework.wikipedia_tool import WikipediaTool


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    agent = Agent("test_agent")
    agent.persona = "You are a helpful assistant that can explain concepts clearly."
    agent.instruction = (
        "Break topics into key points with examples. Use the available tools to find "
        "information before answering."
    )
    agent.strategy = "ReactStrategy"

    web = WebSearchTool()
    agent.tools = [WikipediaTool(), web] if web.available else [WikipediaTool()]
    print("Tools:", agent.get_available_tools())

    print(
        agent.execute("Find the capital of Telangana and then the most popular dish in that city.")
    )


if __name__ == "__main__":
    main()
