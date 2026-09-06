"""Wikipedia lookup tool.

Changes from step 6: a disambiguation page is resolved to its first option
instead of surfacing as a generic failure; the summary is only elided when it
was actually cut; ``format_wiki_result`` (which read keys ``execute`` never
produced) is gone; ``print`` is ``logging``.
"""

from __future__ import annotations

import logging
from typing import Any

import wikipedia

from agent_framework.tools import Tool, ToolResult

logger = logging.getLogger(__name__)

SUMMARY_CHARS = 500


class WikipediaTool(Tool):
    @property
    def name(self) -> str:
        return "wikipedia_search"

    @property
    def description(self) -> str:
        return "Search Wikipedia for information about a topic"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"query": "The Wikipedia search query"}

    def execute(self, **kwargs: Any) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, data="", error="No query provided")
        try:
            titles = wikipedia.search(query)
            if not titles:
                return ToolResult(success=True, data="No Wikipedia articles found for the query.")
            try:
                page = wikipedia.page(titles[0], auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                page = wikipedia.page(e.options[0], auto_suggest=False)
        except Exception as e:
            logger.warning("Wikipedia search failed for %r: %s", query, e)
            return ToolResult(success=False, data="", error=f"Wikipedia search failed: {e}")
        summary = page.summary
        if len(summary) > SUMMARY_CHARS:
            summary = summary[:SUMMARY_CHARS] + "..."
        logger.info("Wikipedia: %s", page.title)
        return ToolResult(success=True, data=f"Title: {page.title}\nSummary: {summary}")
