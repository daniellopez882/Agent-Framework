# wikipedia_tool.py
import wikipedia
from typing import Dict, Any
from tools import Tool, ToolResult

class WikipediaTool(Tool):
    """Tool for searching Wikipedia"""
    
    @property
    def name(self) -> str:
        return "wikipedia_search"
    
    @property
    def description(self) -> str:
        return "Search Wikipedia for information about a topic"
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "query": {
                "type": "string", 
                "description": "The Wikipedia search query"
            }
        }
    
    def execute(self, **kwargs) -> ToolResult:
        try:
            query = kwargs.get("query")
            print(f"Searching Wikipedia for: {query}")
            search_results = wikipedia.search(query)
            if not search_results:
                print("No Wikipedia results found")
                return ToolResult(
                    success=True,
                    data="No Wikipedia articles found for the query."
                )
            
            page = wikipedia.page(search_results[0])
            summary = page.summary[:500] + "..."
            print(f"Found Wikipedia article: {page.title}")
            
            return ToolResult(
                success=True,
                data=f"Title: {page.title}\nSummary: {summary}"
            )
        except Exception as e:
            print(f"Wikipedia search failed: {str(e)}")
            return ToolResult(
                success=False,
                data="",
                error=f"Wikipedia search failed: {str(e)}"
            )

def format_wiki_result(result: Dict) -> str:
    """Format Wikipedia results for inclusion in agent messages."""
    if 'suggestions' in result:
        return result['message']
    
    formatted = [
        f"Wikipedia Article: {result['title']}",
        f"URL: {result['url']}",
        "",
        "Summary:" if result['is_summary'] else "Content:",
        result['content']
    ]
    
    return "\n".join(formatted)