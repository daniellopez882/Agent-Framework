# websearch_tool.py
import os
import json
from typing import Dict, Any
from tools import Tool, ToolResult
from tavily import TavilyClient

class WebSearchTool(Tool):
    """Tool for performing web searches using Tavily API"""
    
    def __init__(self):
        """Initialize the web search tool with API key."""
        self.api_key = os.getenv('TAVILY_API_KEY', '')
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information about a topic"
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string", 
                "description": "The search query to look up"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute a web search using Tavily API.
        
        Args:
            query: The search query string
            
        Returns:
            ToolResult containing search results or error
        """
        try:
            query = kwargs.get("query")
            if not query:
                return ToolResult(
                    success=False,
                    data="",
                    error="No query provided"
                )
            
            print(f"Searching web for: {query}")
            client = TavilyClient(api_key=self.api_key)
            search_response = client.search(query=query)
            
            # Ensure we have results and they're in the expected format
            if not isinstance(search_response, dict) or 'results' not in search_response:
                return ToolResult(
                    success=False,
                    data="",
                    error="Invalid response format from search API"
                )
            
            # Take the top 3 results
            results = search_response['results'][:3]
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get('title', 'No title'),
                    "content": result.get('content', 'No content'),
                    "url": result.get('url', 'No URL')
                })
            
            # Format the output
            formatted_output = self._format_search_results(formatted_results)
            print(f"Found {len(formatted_results)} results for query: {query}")
            
            return ToolResult(
                success=True,
                data=formatted_output
            )
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            print(error_msg)
            return ToolResult(
                success=False,
                data="",
                error=error_msg
            )

    def _format_search_results(self, results: list) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string of search results
        """
        if not results:
            return "No results found."
        
        formatted = ["Search Results:"]
        for i, result in enumerate(results, 1):
            formatted.extend([
                f"\n{i}. {result['title']}",
                f"   URL: {result['url']}",
                f"   Summary: {result['content']}",
                ""
            ])
        
        return "\n".join(formatted)