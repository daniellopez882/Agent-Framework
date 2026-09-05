"""``python -m agent_framework check`` — report the configuration without calling a model.

This is what the container runs by default and what CI probes: it proves the
package imports and says what is configured. ``--strict`` exits non-zero when
``OPENAI_API_KEY`` is missing, for deployments that must not start half-set.
"""

from __future__ import annotations

import argparse
import sys

from agent_framework.config import settings
from agent_framework.strategy import StrategyFactory


def check(strict: bool = False) -> int:
    print(f"model: {settings.OPENAI_MODEL}")
    print(f"openai_api_key: {'set' if settings.openai_configured else 'missing'}")
    web = "enabled" if settings.web_search_configured else "disabled (no TAVILY_API_KEY)"
    print(f"web_search: {web}")
    print(f"max_tool_iterations: {settings.MAX_TOOL_ITERATIONS}")
    print(f"agent_db_path: {settings.AGENT_DB_PATH}")
    print(f"context_persist_dir: {settings.CONTEXT_PERSIST_DIR}")
    print(f"strategies: {', '.join(StrategyFactory.available_strategies())}")
    if strict and not settings.openai_configured:
        print("error: OPENAI_API_KEY is required with --strict", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_framework")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="print the effective configuration")
    p.add_argument("--strict", action="store_true", help="fail if OPENAI_API_KEY is missing")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(strict=args.strict)
    return 2


if __name__ == "__main__":
    sys.exit(main())
