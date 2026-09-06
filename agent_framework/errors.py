"""Exceptions raised by the framework.

The previous code returned error strings from ``Agent.execute`` ("API key not
found...", "An error occurred: ...") as if they were answers, so a caller could
not tell a failure from a reply. Failures are exceptions now.
"""


class AgentError(Exception):
    """Base class for framework errors."""


class ConfigurationError(AgentError):
    """A required setting is missing or invalid."""


class ToolLoopExceeded(AgentError):
    """The agent kept calling tools past ``MAX_TOOL_ITERATIONS``."""
