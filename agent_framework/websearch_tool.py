"""Web search via Tavily.

The constructor raised ``ValueError`` when ``TAVILY_API_KEY`` was unset, so
``flow.py`` could not start at all without a Tavily account — not even for
the Wikipedia-only path. The tool now constructs without a key and reports
the missing key as a failed ``ToolResult`` when it is actually called; the
client is injectable for tests.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_framework.config import settings
from agent_framework.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    def __init__(
        self, api_key: str | None = None, max_results: int | None = None, client: Any = None
    ) -> None:
        self._api_key = (api_key if api_key is not None else settings.TAVILY_API_KEY).strip()
        self._max_results = max_results or settings.TAVILY_MAX_RESULTS
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self._api_key)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information about a topic"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"query": "The search query to look up"}

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def execute(self, **kwargs: Any) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, data="", error="No query provided")
        if not self.available:
            return ToolResult(success=False, data="", error="TAVILY_API_KEY is not set")
        try:
            response = self._get_client().search(query=query)
        except Exception as e:
            logger.warning("Web search failed for %r: %s", query, e)
            return ToolResult(success=False, data="", error=f"Web search failed: {e}")
        if not isinstance(response, dict) or "results" not in response:
            return ToolResult(
                success=False, data="", error="Invalid response format from search API"
            )
        results = [
            {
                "title": r.get("title", "No title"),
                "content": r.get("content", "No content"),
                "url": r.get("url", "No URL"),
            }
            for r in response["results"][: self._max_results]
        ]
        logger.info("Web search: %d results for %r", len(results), query)
        return ToolResult(success=True, data=self._format(results))

    @staticmethod
    def _format(results: list[dict[str, str]]) -> str:
        if not results:
            return "No results found."
        out = ["Search Results:"]
        for i, r in enumerate(results, 1):
            out.extend(
                [f"\n{i}. {r['title']}", f"   URL: {r['url']}", f"   Summary: {r['content']}", ""]
            )
        return "\n".join(out)
