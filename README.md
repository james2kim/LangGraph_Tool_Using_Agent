# Tool-Using Agent

A single-loop agentic system that uses Claude to intelligently select and execute tools based on natural language queries.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
│                         "What is 5 times 5?"                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENT LOOP                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Initialize State                                                   │  │
│  │    - messages: [{ role: 'user', content: userQuery }]                 │  │
│  │    - attempts: 0                                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 2. Get Tool Intent (LLM Call)                                         │  │
│  │    - Send messages to Claude with tool definitions                    │  │
│  │    - Returns: 'text' (final answer) OR 'tool_use' (tool request)      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│           ┌──────────────┐                ┌──────────────────┐              │
│           │ type: 'text' │                │ type: 'tool_use' │              │
│           │              │                │                  │              │
│           │ Return text  │                │ Continue...      │              │
│           │ to user      │                │                  │              │
│           └──────────────┘                └──────────────────┘              │
│                                                    │                        │
│                                                    ▼                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 3. Validate Tool Input (Zod)                                          │  │
│  │    - Look up schema from TOOL_INPUT_SCHEMAS                           │  │
│  │    - safeParse(block.input)                                           │  │
│  │    - If invalid: push error tool_result, continue loop                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 4. Execute Tool (Dispatch)                                            │  │
│  │    - Route to appropriate executor based on tool name                 │  │
│  │    - Returns structured observation (success or failure)              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 5. Push Results to Messages                                           │  │
│  │    - Assistant message: tool_use block                                │  │
│  │    - User message: tool_result with observation                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│                          Loop back to step 2                                │
│                        (until text or max attempts)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Tool Definitions

Tools are defined with Zod schemas that serve dual purposes:

- **JSON Schema generation** for the LLM (via `z.toJSONSchema()`)
- **Runtime validation** before execution (via `safeParse()`)

```typescript
const TOOL_INPUT_SCHEMAS = {
  calculator: CalculatorInputSchema,
  web_fetch_mock: WebFetchInputSchema,
  db_query_candidates: CandidatesInputQuerySchema,
  db_query_opportunities: OpportunitiesInputQuerySchema,
};
```

### 2. Available Tools

| Tool                     | Description               | Input                              |
| ------------------------ | ------------------------- | ---------------------------------- |
| `calculator`             | Arithmetic operations     | `{ a, b, operation }`              |
| `web_fetch_mock`         | HTTP GET requests         | `{ url, timeout_ms, max_chars }`   |
| `db_query_candidates`    | Query candidates table    | `{ table, columns, where, limit }` |
| `db_query_opportunities` | Query opportunities table | `{ table, columns, where, limit }` |

### 3. Agent State

```typescript
type AgentState = {
  attempts: number; // Current iteration count
  userQuery: string; // Original user input
  messages: Anthropic.MessageParam[]; // Conversation history
  done: boolean; // Loop termination flag
  finalResponse?: string; // Final text response
};
```

### 4. Observation Schemas

Each tool returns a **discriminated union** observation:

```typescript
// Success
{ success: true, result: 25, a: 5, b: 5, operation: 'multiply' }

// Failure
{ success: false, error_type: 'invalid_calculation', error_message: 'Division by zero' }
```

## Message Flow

```
messages = [
  { role: 'user', content: 'What is 5 times 5?' },
  { role: 'assistant', content: [{ type: 'tool_use', id: '...', name: 'calculator', input: {...} }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: '...', content: '{"success":true,"result":25}' }] },
  { role: 'assistant', content: [{ type: 'text', text: '5 times 5 equals 25.' }] }
]
```

## Validation Flow

```
LLM returns tool_use
        │
        ▼
┌───────────────────┐
│ Schema exists?    │──No──▶ Return error observation
└───────────────────┘
        │ Yes
        ▼
┌───────────────────┐
│ Input valid?      │──No──▶ Return validation error
│ (Zod safeParse)   │
└───────────────────┘
        │ Yes
        ▼
   Execute tool
```

## Error Handling

| Error Type      | Handled By     | Response                                                                         |
| --------------- | -------------- | -------------------------------------------------------------------------------- |
| Unknown tool    | Agent loop     | `{ success: false, error_type: 'invalid_input', error_message: 'Unknown tool' }` |
| Invalid input   | Zod validation | `{ success: false, error_type: 'invalid_input', error_message: '...' }`          |
| Execution error | Tool executor  | Tool-specific failure schema                                                     |
| No results      | Tool executor  | `{ success: false, error_type: 'not_found', error_message: '...' }`              |

## Running the Agent

```bash
# Install dependencies
npm install

# Set up environment
cp .env.local.example .env.local
# Add your ANTHROPIC_API_KEY to .env.local

# Run
npm start

# Development (watch mode)
npm run dev
```

## Example Queries

```typescript
// Calculator
'What is 5 times 5?';
'Calculate 100 divided by 4';

// Web Fetch
'Fetch the content of https://example.com';
'What is on https://httpbin.org/json?';

// Database - Candidates
'Find the candidate named Alice Johnson';
'Look up bob@example.com in candidates';

// Database - Opportunities
'What opportunities are in the onsite stage?';
'Find the Mcdonalds opportunity';
```

## Project Structure

```
src/
└── main.ts
    ├── Schema Definitions (lines 10-96)
    │   ├── Tool input schemas (Zod)
    │   └── Observation schemas (success/failure)
    │
    ├── Tool Definitions (lines 98-129)
    │   └── Anthropic.Tool objects with JSON schemas
    │
    ├── Helper Functions (lines 145-236)
    │   ├── craftDbSuccessObservation
    │   ├── craftDbFailureObservation
    │   └── buildRowSchema
    │
    ├── LLM Interface (lines 311-339)
    │   └── getToolIntent() - Calls Claude API
    │
    ├── Tool Executors (lines 372-640)
    │   └── dispatch() - Routes to tool implementations
    │
    └── Agent Loop (lines 642-734)
        └── getFormattedToolOrAnswerToUserInput()
```

## Key Design Decisions

1. **Zod for dual-purpose schemas** - Single source of truth for both LLM tool definitions and runtime validation

2. **Discriminated unions for observations** - Type-safe success/failure handling with `success: true | false`

3. **Input validation before execution** - Prevents malformed inputs from reaching executors

4. **Structured error responses** - Errors are returned as observations, giving the LLM a chance to retry

5. **Max attempts limit** - Prevents infinite loops (default: 3)

## Limitations

- Single tool execution per turn (last tool_use block only)
- No parallel tool execution
- Mock database (in-memory data)
- No conversation persistence
