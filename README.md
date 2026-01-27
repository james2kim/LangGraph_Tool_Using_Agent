# Tool-Using Agent

A single-loop agentic system built with LangGraph that routes user queries to structured tools, validates inputs/outputs with Zod schemas, and returns traced, observable responses.

## Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ classifyTool  │ ◄─────────────────────────┐
│  (LLM node)  │                            │
└──────┬───────┘                            │
       │                                    │
       ├── tool_calls? ──► verifyAndExecute ─┘
       │                    ToolIntent
       ├── response?  ──► END (final answer)
       │
       └── maxStep?   ──► END (bail out)
```

The graph has two nodes connected in a loop:

1. **classifyTool** - Sends the conversation (including any prior tool results) to Claude with tools bound. Claude either calls a tool or produces a final text answer.
2. **verifyAndExecuteToolIntent** - Validates the LLM's tool call arguments against Zod schemas, executes the tool if valid, and appends a `ToolMessage` back into the conversation. Increments the step counter.

A conditional edge (`routeAfterClassification`) decides whether to loop back or terminate.

## State Management

State is defined via `Annotation.Root`. Most fields use simple replacement (last write wins). The `trace` field uses a **reducer** — each node appends trace entries rather than overwriting, so the full execution history accumulates across loop iterations.

```typescript
trace: Annotation<TraceEntry[]>({
  reducer: (prev, next) => [...prev, ...next], // append, don't replace
  default: () => [],
});
```

## Tools

| Tool                     | Purpose                                            | Input Schema                             |
| ------------------------ | -------------------------------------------------- | ---------------------------------------- |
| `calculator`             | Exact arithmetic (add, subtract, multiply, divide) | `{ a, b, operation }`                    |
| `web_fetch_mock`         | HTTP GET with timeout and truncation               | `{ url, timeout_ms, max_chars, method }` |
| `db_query_candidates`    | Query mock candidates table                        | `{ table, columns, where, limit }`       |
| `db_query_opportunities` | Query mock opportunities table                     | `{ table, columns, where, limit }`       |

## Input Validation

Every tool call from the LLM goes through a **verify-then-execute** gate:

1. Tool name is looked up in `TOOL_BY_NAME` — unknown tools are rejected.
2. Arguments are parsed through `parseStringifiedJsonFields` to handle cases where the LLM sends nested objects as stringified JSON.
3. Arguments are validated against the corresponding Zod schema from `TOOL_INPUT_SCHEMAS`.
4. Only on successful validation does the tool execute.

This prevents the LLM from passing malformed inputs to tool implementations.

## Output Validation

Tool return values are also schema-validated (e.g., `CalculatorSuccessSchema`, `WebObservationSuccessSchema`). Both inputs and outputs are constrained, creating a typed contract between the LLM and the tool layer.

The final `AgentResponse` is validated through `AgentResponseSchema`, a discriminated union on `status: 'success' | 'max_attempts_reached' | 'error'`.

## Tracing

Every LLM decision and tool execution produces a `TraceEntry` recorded in state. Each entry captures:

- **LLM entries**: step number, decision type (tool_use / final_answer / max_attempts), which tools were called
- **Tool entries**: step number, tool name, raw input, validation result, observation, success/failure, duration

This gives full observability into multi-step reasoning without external logging infrastructure.

## Key Tradeoffs

| Decision                                                           | Benefit                                                                  | Cost                                                                              |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Zod validation on both inputs and outputs                          | Strong runtime safety; malformed LLM calls fail fast with clear errors   | More schema code to maintain; schemas must stay in sync with tool implementations |
| Single loop with step counter (`maxStep: 5`)                       | Prevents runaway loops; simple to reason about                           | Fixed cap may cut off legitimate multi-step reasoning                             |
| Discriminated unions for schemas (`success: true/false`)           | Type-safe branching; self-documenting response shapes                    | More verbose than a simple try/catch; requires careful schema alignment           |
| Mock data instead of real DB/HTTP                                  | Deterministic testing; no external dependencies                          | Tools don't prove real integration works                                          |
| `parseStringifiedJsonFields` workaround                            | Handles LLMs that stringify nested JSON args                             | Adds a processing step; could mask real input issues                              |
| Reducer-based trace accumulation                                   | Full history preserved across loop iterations without manual bookkeeping | Traces grow unbounded within a run (mitigated by maxStep)                         |
| Separate tool schemas (`TOOL_INPUT_SCHEMAS`) from tool definitions | Enables pre-validation before execution                                  | Schema duplication — the tool already has a schema passed to `tool()`             |

## Running

```bash
npm install
npx tsx src/main.ts
```

Requires `ANTHROPIC_API_KEY` in `.env.local`.
