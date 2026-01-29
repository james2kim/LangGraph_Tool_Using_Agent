import dotenv from 'dotenv';
import {
  StateGraph,
  START,
  END,
  Annotation,
  MessagesAnnotation,
  MessagesValue,
  StateSchema,
  ReducedValue,
} from '@langchain/langgraph';
import { tool } from 'langchain';
dotenv.config({ path: '.env.local' });
import { z } from 'zod/v4';
import { ChatAnthropic } from '@langchain/anthropic';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';

const model = new ChatAnthropic({
  model: 'claude-sonnet-4-5-20250929',
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const CalculatorInputSchema = z.object({
  a: z.number(),
  b: z.number(),
  operation: z.enum(['multiply', 'add', 'subtract', 'divide']),
});
type CalculatorInput = z.infer<typeof CalculatorInputSchema>;
const WebFetchInputSchema = z.object({
  url: z.string().url().describe('Url we are fetching, must include https://'),
  timeout_ms: z.number().int().min(100).max(10000).default(2500),
  max_chars: z.number().int().min(200).max(5000).default(1500),
  method: z.enum(['GET']).default('GET'),
});
type WebFetchInput = z.infer<typeof WebFetchInputSchema>;

const CANDIDATE_COLUMN_ENUM = z.enum(['id', 'name', 'email', 'address']);

const OPPORTUNITY_COLUMN_ENUM = z.enum(['id', 'name', 'position', 'stage']);

const CandidatesInputQuerySchema = z.object({
  table: z.literal('candidates'),
  columns: z.array(CANDIDATE_COLUMN_ENUM).min(1),
  where: z.discriminatedUnion('field', [
    z.object({
      field: z.literal('name'),
      operator: z.literal('='),
      value: z.string().min(1),
    }),
    z.object({
      field: z.literal('email'),
      operator: z.literal('='),
      value: z.string().email(),
    }),
    z.object({
      field: z.literal('id'),
      operator: z.literal('='),
      value: z.string().uuid(),
    }),
    z.object({
      field: z.literal('address'),
      operator: z.literal('='),
      value: z.string().min(5),
    }),
  ]),
  limit: z.number().int().positive().min(1).max(25).default(10),
});

const OpportunitiesInputQuerySchema = z.object({
  table: z.literal('opportunities'),
  columns: z.array(OPPORTUNITY_COLUMN_ENUM).min(1),
  where: z.discriminatedUnion('field', [
    z.object({
      field: z.literal('name'),
      operator: z.literal('='),
      value: z.string().min(1),
    }),
    z.object({
      field: z.literal('position'),
      operator: z.literal('='),
      value: z.string(),
    }),
    z.object({
      field: z.literal('id'),
      operator: z.literal('='),
      value: z.string().uuid(),
    }),
    z.object({
      field: z.literal('stage'),
      operator: z.literal('='),
      value: z.enum(['applied', 'screen', 'onsite']),
    }),
  ]),
  limit: z.number().int().positive().min(1).max(25).default(10),
});

const dbQueryInputSchema = z.discriminatedUnion('table', [
  CandidatesInputQuerySchema,
  OpportunitiesInputQuerySchema,
]);

const TOOL_INPUT_SCHEMAS = {
  calculator: CalculatorInputSchema,
  web_fetch_mock: WebFetchInputSchema,
  db_query_candidates: CandidatesInputQuerySchema,
  db_query_opportunities: OpportunitiesInputQuerySchema,
} as const;

type DBQueryInput = z.infer<typeof dbQueryInputSchema>;

const calculatorTool = tool(
  (calcInput: CalculatorInput) => {
    if (calcInput.a === 9999) throw new Error('boom');
    let result: number;

    try {
      switch (calcInput.operation) {
        case 'multiply':
          result = calcInput.a * calcInput.b;
          break;
        case 'add':
          result = calcInput.a + calcInput.b;
          break;
        case 'subtract':
          result = calcInput.a - calcInput.b;
          break;
        case 'divide':
          result = calcInput.a / calcInput.b;
          break;
        default:
          throw new Error('Unsupported operation');
      }

      if (!Number.isFinite(result)) {
        const { error_type, error_message } = classifyCalculatorError(calcInput, result);

        return CalculatorFailureSchema.parse({
          success: false,
          ...calcInput,
          error_type,
          error_message,
        });
      }

      return CalculatorSuccessSchema.parse({
        success: true,
        result,
        ...calcInput,
      });
    } catch (err) {
      return CalculatorFailureSchema.parse({
        success: false,
        ...calcInput,
        error_type: 'runtime_error',
        error_message: err instanceof Error ? err.message.slice(0, 50) : 'Unknown runtime error',
      });
    }
  },
  {
    name: 'calculator',
    description:
      'Use this tool when an exact numerical calculation is required and precision matters.',
    schema: CalculatorInputSchema,
  }
);

const webFetchTool = tool(
  async (webInput: WebFetchInput) => {
    const startTime = Date.now();

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), webInput.timeout_ms);

      const response = await fetch(webInput.url, {
        method: webInput.method,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const timing_ms = Date.now() - startTime;

      if (!response.ok) {
        return WebObservationFailureSchema.parse({
          success: false,
          status: response.status,
          timing_ms,
          error_type: 'http_error',
          error: `HTTP ${response.status}: ${response.statusText}`,
        });
      }

      const contentType = response.headers.get('content-type') || undefined;
      const fullText = await response.text();
      const truncated = fullText.length > webInput.max_chars;
      const text = truncated ? fullText.slice(0, webInput.max_chars) : fullText;

      return WebObservationSuccessSchema.parse({
        success: true,
        final_url: response.url,
        status: response.status,
        timing_ms,
        content_type: contentType,
        text,
        truncated,
      });
    } catch (err) {
      const timing_ms = Date.now() - startTime;
      const errorMessage =
        err instanceof Error
          ? err.name === 'AbortError'
            ? `Request timed out after ${webInput.timeout_ms}ms`
            : err.message
          : 'Unknown fetch error';

      return WebObservationFailureSchema.parse({
        success: false,
        timing_ms,
        error_message: errorMessage,
        error_type: 'runtime_error',
      });
    }
  },
  {
    name: 'web_fetch_mock',
    description:
      'Use this tool when we need to fetch the external page content or meta data of a website',
    schema: WebFetchInputSchema,
  }
);

const dbQueryCandidatesTool = tool(
  (dbInput: z.infer<typeof CandidatesInputQuerySchema>) => {
    const mockCandidates = [
      {
        id: '550e8400-e29b-41d4-a716-446655440001',
        name: 'Alice Johnson',
        email: 'alice@example.com',
        address: '123 Main St, NYC',
      },
      {
        id: '550e8400-e29b-41d4-a716-446655440002',
        name: 'Bob Smith',
        email: 'bob@example.com',
        address: '456 Oak Ave, LA',
      },
      {
        id: '550e8400-e29b-41d4-a716-446655440003',
        name: 'Carol White',
        email: 'carol@example.com',
        address: '789 Pine Rd, Chicago',
      },
    ];

    try {
      // Filter by where clause
      const filtered = mockCandidates.filter((candidate) => {
        const field = dbInput.where.field as keyof typeof candidate;
        return candidate[field] === dbInput.where.value;
      });

      // No results found
      if (filtered.length === 0) {
        return craftDbFailureObservation({
          table: 'candidates',
          columns: dbInput.columns,
          error: {
            message: `No candidate found with ${dbInput.where.field} = ${dbInput.where.value}`,
            type: 'not_found',
          },
          map: CANDIDATES_COLUMNS_TO_ZOD,
        });
      }

      // Apply limit
      const limited = filtered.slice(0, dbInput.limit);
      const hasMore = filtered.length > dbInput.limit;

      // Select only requested columns
      const rows = limited.map((row) => {
        const selected: Record<string, unknown> = {};
        for (const col of dbInput.columns) {
          selected[col] = row[col as keyof typeof row];
        }
        return selected;
      });

      return craftDbSuccessObservation({
        table: 'candidates',
        columns: dbInput.columns,
        rows,
        hasMore,
        map: CANDIDATES_COLUMNS_TO_ZOD,
      });
    } catch (err) {
      return craftDbFailureObservation({
        table: 'candidates',
        columns: dbInput.columns,
        error: {
          message: err instanceof Error ? err.message.slice(0, 50) : 'Query failed',
          type: 'runtime_error',
        },
        map: CANDIDATES_COLUMNS_TO_ZOD,
      });
    }
  },
  {
    name: 'db_query_candidates',
    description: 'Query the candidates table by name, email, id, or address',
    schema: CandidatesInputQuerySchema,
  }
);

const dbQueryOpportunitiesTool = tool(
  (dbInput: z.infer<typeof OpportunitiesInputQuerySchema>) => {
    const mockOpportunities = [
      {
        id: '550e8400-e29b-41d4-a716-446655440010',
        name: 'Mcdonalds',
        position: 'Manager',
        stage: 'onsite' as const,
      },
      {
        id: '550e8400-e29b-41d4-a716-446655440011',
        name: 'Burger King',
        position: 'Product Manager',
        stage: 'screen' as const,
      },
      {
        id: '550e8400-e29b-41d4-a716-446655440012',
        name: 'Taco Bell',
        position: 'Designer',
        stage: 'applied' as const,
      },
    ];

    try {
      const filtered = mockOpportunities.filter((opportunity) => {
        const field = dbInput.where.field as keyof typeof opportunity;
        return opportunity[field] === dbInput.where.value;
      });

      // No results found
      if (filtered.length === 0) {
        return craftDbFailureObservation({
          table: 'opportunities',
          columns: dbInput.columns,
          error: {
            message: `No opportunity found with ${dbInput.where.field} = ${dbInput.where.value}`,
            type: 'not_found',
          },
          map: OPPORTUNITIES_COLUMNS_TO_ZOD,
        });
      }

      const limited = filtered.slice(0, dbInput.limit);
      const hasMore = filtered.length > dbInput.limit;

      const rows = limited.map((row) => {
        const selected: Record<string, unknown> = {};
        for (const col of dbInput.columns) {
          selected[col] = row[col as keyof typeof row];
        }
        return selected;
      });

      return craftDbSuccessObservation({
        table: 'opportunities',
        columns: dbInput.columns,
        rows,
        hasMore,
        map: OPPORTUNITIES_COLUMNS_TO_ZOD,
      });
    } catch (err) {
      return craftDbFailureObservation({
        table: 'opportunities',
        columns: dbInput.columns,
        error: {
          message: err instanceof Error ? err.message.slice(0, 50) : 'Query failed',
          type: 'runtime_error',
        },
        map: OPPORTUNITIES_COLUMNS_TO_ZOD,
      });
    }
  },
  {
    name: 'db_query_opportunities',
    description: 'Query the opportunities table by name, position, id, or stage',
    schema: OpportunitiesInputQuerySchema,
  }
);

const TOOL_BY_NAME = {
  calculator: calculatorTool,
  web_fetch_mock: webFetchTool,
  db_query_candidates: dbQueryCandidatesTool,
  db_query_opportunities: dbQueryOpportunitiesTool,
} as const;

const CANDIDATES_COLUMNS_TO_ZOD = {
  id: z.string().uuid(),
  name: z.string().min(1),
  email: z.string().email(),
  address: z.string().min(5),
};

const OPPORTUNITIES_COLUMNS_TO_ZOD = {
  id: z.string().uuid(),
  name: z.string().min(1),
  position: z.string().min(1),
  stage: z.enum(['applied', 'screen', 'onsite']),
};

const craftDbObservationSuccessSchema = (
  table: 'candidates' | 'opportunities',
  columns: string[],
  map: Record<string, z.ZodTypeAny>
) => {
  const rowSchema = buildRowSchema(map, columns);
  const c = table === 'candidates' ? CANDIDATE_COLUMN_ENUM : OPPORTUNITY_COLUMN_ENUM;

  return z.object({
    success: z.literal(true),
    table: z.literal(table),
    columns: z.array(c).min(1),
    rows: z.array(rowSchema),
    rows_returned: z.number().int().min(0).max(25),
    has_more: z.boolean(),
  });
};

const craftDbObservationFailureSchema = (
  table: 'candidates' | 'opportunities',
  _map: Record<string, z.ZodTypeAny>
) => {
  const c = table === 'candidates' ? CANDIDATE_COLUMN_ENUM : OPPORTUNITY_COLUMN_ENUM;
  return z.object({
    success: z.literal(false),
    table: z.literal(table),
    columns: z.array(c).min(1),
    error_message: z.string().min(5).max(50),
    error_type: z.enum(['invalid_input', 'invalid_schema', 'runtime_error', 'not_found']),
  });
};

const craftDbSuccessObservation = ({
  table,
  columns,
  rows,
  hasMore,
  map,
}: {
  table: 'candidates' | 'opportunities';
  columns: string[];
  rows: Record<string, unknown>[];
  hasMore: boolean;
  map: Record<string, z.ZodTypeAny>;
}) => {
  const observation = {
    success: true as const,
    table,
    columns,
    rows,
    has_more: hasMore,
    rows_returned: rows.length,
  };
  craftDbObservationSuccessSchema(table, columns, map).parse(observation);
  return observation;
};

const craftDbFailureObservation = ({
  table,
  columns,
  error,
  map,
}: {
  table: 'candidates' | 'opportunities';
  columns: string[];
  error: { message: string; type: string };
  map: Record<string, z.ZodTypeAny>;
}) => {
  const observation = {
    success: false as const,
    table,
    columns,
    error_message: error.message,
    error_type: error.type,
  };
  craftDbObservationFailureSchema(table, map).parse(observation);
  return observation;
};

function buildRowSchema<TCols extends Record<string, z.ZodTypeAny>, K extends keyof TCols>(
  columnSchemas: TCols,
  columns: readonly K[]
) {
  const shape: Record<string, z.ZodTypeAny> = {};

  for (const col of columns) {
    shape[col as string] = columnSchemas[col];
  }

  return z.object(shape).strict();
}

const CalculatorSuccessSchema = z.object({
  success: z.literal(true),
  result: z.number().finite(),
  a: z.number(),
  b: z.number(),
  operation: z.enum(['multiply', 'add', 'subtract', 'divide']),
});

const CalculatorFailureSchema = z.object({
  success: z.literal(false),
  a: z.number().optional(),
  b: z.number().optional(),
  operation: z.enum(['multiply', 'add', 'subtract', 'divide']),
  error_message: z.string().min(5).max(100),
  error_type: z.enum(['invalid_input', 'invalid_calculation', 'runtime_error', 'invalid_schema']),
});

type CalculatorFailure = z.infer<typeof CalculatorFailureSchema>;

const WebObservationSuccessSchema = z.object({
  success: z.literal(true),
  final_url: z.string().url(),
  status: z.number().int().min(200).max(399),
  timing_ms: z.number().int().nonnegative(),
  content_type: z.string().optional(),
  text: z.string(),
  truncated: z.boolean(),
});

const WebObservationFailureSchema = z.object({
  success: z.literal(false),
  status: z.number().int().min(400).max(599).optional(),
  timing_ms: z.number().int().nonnegative(),
  error_message: z.string().min(5).max(100),
  error_type: z.enum(['invalid_input', 'http_error', 'runtime_error', 'invalid_schema']),
});

function classifyCalculatorError(
  input: CalculatorInput,
  result: number
): { error_type: CalculatorFailure['error_type']; error_message: string } {
  if (!Number.isFinite(result)) {
    if (input.operation === 'divide' && input.b === 0) {
      return {
        error_type: 'invalid_calculation',
        error_message: 'Division by zero',
      };
    }

    return {
      error_type: 'invalid_calculation',
      error_message: 'Non-finite calculation result',
    };
  }

  return {
    error_type: 'runtime_error',
    error_message: 'Unknown calculation error',
  };
}

type ToolType = 'calculator' | 'web_fetch_mock' | 'db_query_candidates' | 'db_query_opportunities';

type ToolTraceEntry = {
  step: number;
  toolName: string;
  toolCallId: string;
  input: Record<string, unknown>;
  inputValid: boolean;
  validationError?: string;
  observation: unknown;
  success: boolean;
  timestamp: number;
  durationMs?: number;
};

type LLMTraceEntry = {
  step: number;
  decision: 'tool_use' | 'final_answer' | 'max_attempts';
  toolsCalled?: string[];
  timestamp: number;
};

type TraceEntry = { type: 'llm'; data: LLMTraceEntry } | { type: 'tool'; data: ToolTraceEntry };

const AgentResponseSchema = z.discriminatedUnion('status', [
  z.object({
    status: z.literal('success'),
    content: z.string(),
    metadata: z.object({
      userQuery: z.string(),
      totalSteps: z.number(),
      toolsUsed: z.array(z.string()),
      startedAt: z.number(),
      completedAt: z.number(),
      durationMs: z.number(),
    }),
    trace: z.array(z.any()),
  }),
  z.object({
    status: z.literal('max_attempts_reached'),
    content: z.string(),
    metadata: z.object({
      userQuery: z.string(),
      totalSteps: z.number(),
      maxSteps: z.number(),
      toolsUsed: z.array(z.string()),
      startedAt: z.number(),
      completedAt: z.number(),
      durationMs: z.number(),
    }),
    trace: z.array(z.any()),
  }),
  z.object({
    status: z.literal('error'),
    content: z.string(),
    error: z.object({
      type: z.string(),
      message: z.string(),
    }),
    metadata: z.object({
      userQuery: z.string(),
      totalSteps: z.number(),
      toolsUsed: z.array(z.string()),
      startedAt: z.number(),
      completedAt: z.number(),
      durationMs: z.number(),
    }),
    trace: z.array(z.any()),
  }),
]);

type AgentResponse = z.infer<typeof AgentResponseSchema>;

export const AgentStateSchema = new StateSchema({
  messages: MessagesValue,
  userQuery: z.string(),
  tool_calls: z
    .array(
      z.object({
        id: z.string(),
        name: z.string(),
        args: z.record(z.unknown()),
      })
    )
    .optional(),
  step: z.number(),
  response: z.string().optional(),
  maxStep: z.number(),
  trace: new ReducedValue(
    z.array(z.any()).default(() => []),
    {
      inputSchema: z.array(z.any()),
      reducer: (prev: any, next: any) => [...prev, ...next],
    }
  ),
});

export type AgentState = typeof AgentStateSchema.State;

export const getToolIntent = async (state: AgentState) => {
  if (state.step >= state.maxStep) {
    const traceEntry: TraceEntry = {
      type: 'llm',
      data: {
        step: state.step,
        decision: 'max_attempts',
        timestamp: Date.now(),
      },
    };
    return {
      response: 'Max attempts reached without final answer',
      trace: [traceEntry],
    };
  }

  const modelWithTools = model.bindTools(Object.values(TOOL_BY_NAME));
  const aiMessage = await modelWithTools.invoke(state.messages);

  const toolCalls = aiMessage.tool_calls ?? [];
  if (toolCalls.length > 0) {
    const traceEntry: TraceEntry = {
      type: 'llm',
      data: {
        step: state.step,
        decision: 'tool_use',
        toolsCalled: toolCalls.map((tc: any) => tc.name),
        timestamp: Date.now(),
      },
    };
    return {
      messages: [aiMessage],
      tool_calls: toolCalls,
      trace: [traceEntry],
    };
  }

  const traceEntry: TraceEntry = {
    type: 'llm',
    data: {
      step: state.step,
      decision: 'final_answer',
      timestamp: Date.now(),
    },
  };
  return {
    response:
      typeof aiMessage.content === 'string' ? aiMessage.content : JSON.stringify(aiMessage.content),
    tool_calls: undefined,
    trace: [traceEntry],
  };
};

export const verifyAndExecuteTool = async (state: AgentState) => {
  const toolMessages: ToolMessage[] = [];
  const traceEntries: TraceEntry[] = [];
  for (const toolCall of state.tool_calls ?? []) {
    const startTime = Date.now();
    const tool = TOOL_BY_NAME[toolCall.name as keyof typeof TOOL_BY_NAME];

    if (!tool) {
      const observation = {
        success: false,
        error_type: 'invalid_input',
        error_message: `Unknown tool call: ${toolCall.name}`,
      };
      toolMessages.push(
        new ToolMessage({
          tool_call_id: toolCall.id,
          content: JSON.stringify(observation),
        })
      );
      traceEntries.push({
        type: 'tool',
        data: {
          step: state.step,
          toolName: toolCall.name,
          toolCallId: toolCall.id,
          input: toolCall.args,
          inputValid: false,
          validationError: 'Unknown tool',
          observation,
          success: false,
          timestamp: startTime,
          durationMs: Date.now() - startTime,
        },
      });
      continue;
    }

    const schema = TOOL_INPUT_SCHEMAS[toolCall.name as ToolType];
    if (!schema) {
      const observation = {
        success: false,
        error_type: 'invalid_schema',
        error_message: `No schema for tool: ${toolCall.name}`,
      };
      toolMessages.push(
        new ToolMessage({
          tool_call_id: toolCall.id,
          content: JSON.stringify(observation),
        })
      );
      traceEntries.push({
        type: 'tool',
        data: {
          step: state.step,
          toolName: toolCall.name,
          toolCallId: toolCall.id,
          input: toolCall.args,
          inputValid: false,
          validationError: 'No schema found',
          observation,
          success: false,
          timestamp: startTime,
          durationMs: Date.now() - startTime,
        },
      });
      continue;
    }

    const processedArgs = parseStringifiedJsonFields(toolCall.args);
    const parsed = schema.safeParse(processedArgs);

    if (!parsed.success) {
      const observation = {
        success: false,
        error_type: 'invalid_schema',
        error_message: 'Schema Validation Failed',
      };
      toolMessages.push(
        new ToolMessage({
          tool_call_id: toolCall.id,
          content: JSON.stringify(observation),
        })
      );
      traceEntries.push({
        type: 'tool',
        data: {
          step: state.step,
          toolName: toolCall.name,
          toolCallId: toolCall.id,
          input: processedArgs,
          inputValid: false,
          validationError: JSON.stringify(parsed.error.issues),
          observation,
          success: false,
          timestamp: startTime,
          durationMs: Date.now() - startTime,
        },
      });
      continue;
    }

    try {
      const toolResult = await (tool as {
        invoke: (input: Record<string, unknown>) => Promise<unknown>;
      }).invoke(parsed.data as Record<string, unknown>);

      const observation = typeof toolResult === 'string' ? JSON.parse(toolResult) : toolResult;
      const isSuccess = observation?.success !== false;

      toolMessages.push(
        new ToolMessage({
          tool_call_id: toolCall.id,
          content: typeof toolResult === 'string' ? toolResult : JSON.stringify(toolResult),
        })
      );
      traceEntries.push({
        type: 'tool',
        data: {
          step: state.step,
          toolName: toolCall.name,
          toolCallId: toolCall.id,
          input: parsed.data as Record<string, unknown>,
          inputValid: true,
          observation,
          success: isSuccess,
          timestamp: startTime,
          durationMs: Date.now() - startTime,
        },
      });
    } catch (err) {
      const observation = {
        success: false,
        error_type: 'runtime_error',
        error_message: err instanceof Error ? err.message : 'Tool execution failed',
      };
      toolMessages.push(
        new ToolMessage({
          tool_call_id: toolCall.id,
          content: JSON.stringify(observation),
        })
      );
      traceEntries.push({
        type: 'tool',
        data: {
          step: state.step,
          toolName: toolCall.name,
          toolCallId: toolCall.id,
          input: parsed.data as Record<string, unknown>,
          inputValid: true,
          observation,
          success: false,
          timestamp: startTime,
          durationMs: Date.now() - startTime,
        },
      });
    }
  }

  return {
    messages: toolMessages,
    tool_calls: undefined,
    step: state.step + 1,
    trace: traceEntries,
  };
};
function routeAfterClassification(state: AgentState) {
  if (state.response) return END;
  if (state.step >= state.maxStep) return END;
  if (state.tool_calls?.length) return 'verifyAndExecuteToolIntent';
  return END;
}

type CalculatorErrorType = 'invalid_input' | 'invalid_calculation' | 'runtime_error';

type CalculatorError = {
  error_type: CalculatorErrorType;
  error_message: string;
};

/**
 * Recursively parse any string values that look like JSON objects/arrays.
 * LLMs sometimes send nested objects as stringified JSON.
 */
const parseStringifiedJsonFields = (obj: Record<string, unknown>): Record<string, unknown> => {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string' && (value.startsWith('{') || value.startsWith('['))) {
      try {
        result[key] = JSON.parse(value);
      } catch {
        result[key] = value;
      }
    } else if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      result[key] = parseStringifiedJsonFields(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }

  return result;
};

const getFormattedToolOrAnswerToUserInput = async (userQuery: string): Promise<AgentResponse> => {
  const startedAt = Date.now();
  const maxStep = 3;

  const initialState = {
    userQuery,
    messages: [new HumanMessage(userQuery)],
    step: 1,
    maxStep,
    trace: [] as TraceEntry[],
  };

  const workflow = new StateGraph(AgentStateSchema)
    .addNode('classifyTool', getToolIntent)
    .addNode('verifyAndExecuteToolIntent', verifyAndExecuteTool)
    .addEdge(START, 'classifyTool')
    .addConditionalEdges('classifyTool', routeAfterClassification)
    .addEdge('verifyAndExecuteToolIntent', 'classifyTool');

  const app = workflow.compile();

  try {
    const result = await app.invoke(initialState);
    const completedAt = Date.now();

    // Extract unique tools used from trace
    const toolsUsed = Array.from(
      new Set(
        result.trace
          .filter((entry: TraceEntry) => entry.type === 'tool')
          .map(
            (entry: TraceEntry) => (entry as { type: 'tool'; data: ToolTraceEntry }).data.toolName
          )
      )
    );

    const hitMaxAttempts = result.trace.some(
      (entry: TraceEntry) => entry.type === 'llm' && entry.data.decision === 'max_attempts'
    );

    if (hitMaxAttempts) {
      return AgentResponseSchema.parse({
        status: 'max_attempts_reached',
        content: result.response ?? 'Max attempts reached without final answer',
        metadata: {
          userQuery,
          totalSteps: result.step,
          maxSteps: maxStep,
          toolsUsed,
          startedAt,
          completedAt,
          durationMs: completedAt - startedAt,
        },
        trace: result.trace,
      });
    }

    return AgentResponseSchema.parse({
      status: 'success',
      content: result.response ?? '',
      metadata: {
        userQuery,
        totalSteps: result.step,
        toolsUsed,
        startedAt,
        completedAt,
        durationMs: completedAt - startedAt,
      },
      trace: result.trace,
    });
  } catch (err) {
    const completedAt = Date.now();

    return AgentResponseSchema.parse({
      status: 'error',
      content: 'An error occurred while processing your request',
      error: {
        type: err instanceof Error ? err.name : 'UnknownError',
        message: err instanceof Error ? err.message : 'Unknown error',
      },
      metadata: {
        userQuery,
        totalSteps: 0,
        toolsUsed: [],
        startedAt,
        completedAt,
        durationMs: completedAt - startedAt,
      },
      trace: [],
    });
  }
};

// Helper to format trace for display
const formatTrace = (trace: TraceEntry[]): string => {
  return trace
    .map((entry, i) => {
      if (entry.type === 'llm') {
        const d = entry.data;
        let line = `[${i + 1}] LLM (step ${d.step}): ${d.decision}`;
        if (d.toolsCalled?.length) {
          line += ` → tools: [${d.toolsCalled.join(', ')}]`;
        }
        return line;
      } else {
        const d = entry.data;
        const status = d.success ? '✓' : '✗';
        let line = `[${i + 1}] TOOL (step ${d.step}): ${d.toolName} ${status}`;
        if (d.durationMs) line += ` (${d.durationMs}ms)`;
        if (!d.inputValid) line += ` | validation error: ${d.validationError}`;
        return line;
      }
    })
    .join('\n');
};

// Main execution
async function main() {
  console.log('Running agent loop...\n');

  const testInputs = [
    'Hello, how are you?',
    'What is 5 times 5?',
    'What is 5 times a',
    'what is 5 divided by 0',
    'Find the candidate named Alice Johnson',
    'Look up Sarah Connor in candidates',
  ];

  for (const input of testInputs) {
    console.log(`═══════════════════════════════════════════════════`);
    console.log(`Input: "${input}"`);
    console.log(`───────────────────────────────────────────────────`);

    const response = await getFormattedToolOrAnswerToUserInput(input);

    console.log(`\nStatus: ${response.status}`);
    console.log(`Duration: ${response.metadata.durationMs}ms`);
    console.log(`Steps: ${response.metadata.totalSteps}`);
    if (response.metadata.toolsUsed.length > 0) {
      console.log(`Tools used: [${response.metadata.toolsUsed.join(', ')}]`);
    }

    console.log('\nTrace:');
    console.log(formatTrace(response.trace as TraceEntry[]));

    console.log(`\nContent:\n${response.content}`);

    if (response.status === 'error') {
      console.log(`\nError: ${response.error.type} - ${response.error.message}`);
    }

    console.log('');
  }
}

main().catch(console.error);
