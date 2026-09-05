# ADR 0002 — A real tool loop, with a parser that knows where the block ends

**Status:** accepted

## Context

`Agent.execute` called the model once. If the reply contained a `Tool:` line,
`_process_tool_usage` ran the tool and *replaced the block in the reply text*
with `Tool Result (name): …`, then returned that text as the answer. The model
never saw a tool result. Reproduced with a stub client on the original code:
one model call, the observation pasted into the draft, and the `- query: …`
parameter line left dangling because the block was cut at the end of the
`Parameters:` line. "ReactStrategy" was a prompt asking the model to write
its own `Observation:` lines — which it did, from imagination.

The parser took its parameters from every line containing `:` and `-`, so a
`Thought: first - then second` line became a parameter named `Thought`.

## Decision

`execute` is a loop bounded by `MAX_TOOL_ITERATIONS`:

1. call the model;
2. if the reply contains a tool block (and the agent has tools), run it, append
   the reply up to the end of the block as the assistant turn and
   `Observation (<tool>): <result>` as the next user turn, and go to 1;
3. otherwise the reply is the answer. `Final Answer:` is stripped when present.

`parse_tool_usage` returns a `ToolCall` with the block's exact span. Only
`- name: value` lines directly under `Parameters:` are parameters. Brackets
copied from the prompt template (`Tool: [wikipedia_search]`) are stripped.
Exhausting the iteration budget raises `ToolLoopExceeded` rather than
returning a half-answer.

The ReAct prompt no longer asks the model to invent observations.

## Consequences

- Multi-hop tasks work: the second lookup can depend on the first result.
- Each tool call costs one more model call; the budget is a setting.
- A tool's output is attacker-controllable text that goes back to the model
  (see the threat model).
- Tool failures and unknown tools become observations, so the model can
  recover; a tool that raises does not take the agent down.
