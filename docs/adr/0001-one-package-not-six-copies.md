# ADR 0001 — One package, not six copies

**Status:** accepted

## Context

The repository was six directories, `step-1-persona` … `step-6-tools`, each a
snapshot of the same agent at a later stage. Each step copied the previous
one and added a feature. `context.py` (417 lines) existed twice, byte-for-byte;
`persistence.py` (307 lines) twice; `strategy.py` (112 lines) three times.
Modules imported each other flat (`from agent import Agent`), so a step only
ran with the working directory set to it, and nothing could be imported from a
test. Steps 1–5 were strict subsets of step 6: the diffs were comments,
docstrings and columns that step 6 also had.

A README calling this a "framework" while 3,800 lines were 6× the same ~700
was the main thing to fix; the defects inside those lines came second.

## Decision

`agent_framework/` is the step-6 code, once, as an importable package:
`agent`, `tools`, `strategy`, `persistence`, `context`, `config`, `errors`, the
two built-in tools and a `check` command. The six `flow.py` scripts survive as
`examples/01_persona.py` … `06_tools.py`, one per capability level, importing
the package. The step directories are deleted, not kept "for reference": the
examples are the reference, and git history holds the rest.

## Consequences

- One place to fix a bug; the tests import the package like any user would.
- The tutorial progression is still visible, as six short examples instead of
  six forks.
- Anyone who linked to `step-6-tools/agent.py` will find it gone. The
  README's "what changed" table explains where things went.
