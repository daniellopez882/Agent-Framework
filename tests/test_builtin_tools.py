import types

from agent_framework import websearch_tool, wikipedia_tool
from agent_framework.websearch_tool import WebSearchTool
from agent_framework.wikipedia_tool import SUMMARY_CHARS, WikipediaTool

# --- web search ---------------------------------------------------------------


class StubTavily:
    def __init__(self, response=None, raise_=False):
        self.response = response
        self.raise_ = raise_
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        if self.raise_:
            raise RuntimeError("tavily down")
        return self.response


def test_constructs_without_a_key_and_reports_it_on_use(cfg):
    tool = WebSearchTool()  # used to raise ValueError here
    assert not tool.available
    r = tool.execute(query="x")
    assert not r.success and r.error == "TAVILY_API_KEY is not set"


def test_key_from_settings_makes_it_available(cfg, monkeypatch):
    monkeypatch.setattr(cfg, "TAVILY_API_KEY", "tv-key")
    assert WebSearchTool().available


def test_results_are_formatted_and_capped(cfg):
    client = StubTavily(
        {
            "results": [
                {"title": f"T{i}", "content": f"C{i}", "url": f"http://{i}"} for i in range(5)
            ]
        }
    )
    r = WebSearchTool(client=client, max_results=2).execute(query="hyderabad")
    assert r.success and client.queries == ["hyderabad"]
    assert "1. T0" in r.data and "2. T1" in r.data and "T2" not in r.data
    assert "URL: http://0" in r.data


def test_bad_shape_and_exceptions_and_empty_query(cfg):
    assert not WebSearchTool(client=StubTavily("nope")).execute(query="x").success
    r = WebSearchTool(client=StubTavily(raise_=True)).execute(query="x")
    assert not r.success and "tavily down" in r.error
    assert (
        WebSearchTool(client=StubTavily({"results": []})).execute(query="x").data
        == "No results found."
    )
    assert WebSearchTool(client=StubTavily({})).execute(query="  ").error == "No query provided"


def test_web_search_parameters_are_plain_descriptions(cfg):
    assert "  - query: The search query to look up" in WebSearchTool().to_prompt_format()
    assert websearch_tool.WebSearchTool().name == "web_search"


# --- wikipedia ------------------------------------------------------------------


class _Page:
    def __init__(self, title, summary):
        self.title, self.summary = title, summary


def stub_wikipedia(
    monkeypatch, *, titles=("Hyderabad",), pages=None, disambiguate=None, raise_=None
):
    class DisambiguationError(Exception):
        def __init__(self, options):
            self.options = options

    mod = types.SimpleNamespace()
    mod.exceptions = types.SimpleNamespace(DisambiguationError=DisambiguationError)
    pages = pages or {"Hyderabad": _Page("Hyderabad", "Capital of Telangana.")}
    calls = []

    def search(q):
        calls.append(("search", q))
        if raise_:
            raise raise_
        return list(titles)

    def page(title, auto_suggest=True):
        calls.append(("page", title, auto_suggest))
        if disambiguate and title == disambiguate[0]:
            raise DisambiguationError(disambiguate[1])
        return pages[title]

    mod.search, mod.page = search, page
    monkeypatch.setattr(wikipedia_tool, "wikipedia", mod)
    return calls


def test_wikipedia_summary(monkeypatch):
    calls = stub_wikipedia(monkeypatch)
    r = WikipediaTool().execute(query="Telangana")
    assert r.success and r.data == "Title: Hyderabad\nSummary: Capital of Telangana."
    assert calls == [("search", "Telangana"), ("page", "Hyderabad", False)]


def test_wikipedia_disambiguation_resolves_to_first_option(monkeypatch):
    stub_wikipedia(
        monkeypatch,
        titles=("Mercury",),
        pages={"Mercury (planet)": _Page("Mercury (planet)", "The first planet.")},
        disambiguate=("Mercury", ["Mercury (planet)", "Mercury (element)"]),
    )
    r = WikipediaTool().execute(query="Mercury")
    assert r.success and r.data.startswith("Title: Mercury (planet)")


def test_wikipedia_elides_only_when_cut(monkeypatch):
    long = "x" * (SUMMARY_CHARS + 10)
    stub_wikipedia(monkeypatch, pages={"Hyderabad": _Page("H", long)})
    data = WikipediaTool().execute(query="q").data
    assert data == "Title: H\nSummary: " + "x" * SUMMARY_CHARS + "..."
    stub_wikipedia(monkeypatch, pages={"Hyderabad": _Page("H", "short")})
    assert WikipediaTool().execute(query="q").data.endswith("Summary: short")  # no '...'


def test_wikipedia_no_results_failure_and_empty_query(monkeypatch):
    stub_wikipedia(monkeypatch, titles=())
    r = WikipediaTool().execute(query="zzz")
    assert r.success and "No Wikipedia articles" in r.data
    stub_wikipedia(monkeypatch, raise_=RuntimeError("offline"))
    r = WikipediaTool().execute(query="zzz")
    assert not r.success and "offline" in r.error
    assert WikipediaTool().execute().error == "No query provided"


def test_wikipedia_parameters_are_plain_descriptions():
    assert "  - query: The Wikipedia search query" in WikipediaTool().to_prompt_format()


def test_no_dead_formatter_helper():
    assert not hasattr(wikipedia_tool, "format_wiki_result")
