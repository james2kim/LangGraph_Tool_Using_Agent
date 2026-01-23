import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });
import { z } from 'zod/v4';
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
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

const calculatorTool: Anthropic.Tool = {
  name: 'calculator',
  description:
    'Use this tool when an exact numerical calculation is required and precision matters.',
  input_schema: z.toJSONSchema(CalculatorInputSchema) as Anthropic.Tool.InputSchema,
};

const webFetchTool: Anthropic.Tool = {
  name: 'web_fetch_mock',
  description:
    'Use this tool when we need to fetch the external page content or meta data of a website',
  input_schema: z.toJSONSchema(WebFetchInputSchema) as Anthropic.Tool.InputSchema,
};

const dbQueryCandidatesTool: Anthropic.Tool = {
  name: 'db_query_candidates',
  description: 'Query the candidates table by name, email, id, or address',
  input_schema: z.toJSONSchema(CandidatesInputQuerySchema) as Anthropic.Tool.InputSchema,
};

const dbQueryOpportunitiesTool: Anthropic.Tool = {
  name: 'db_query_opportunities',
  description: 'Query the opportunities table by name, position, id, or stage',
  input_schema: z.toJSONSchema(OpportunitiesInputQuerySchema) as Anthropic.Tool.InputSchema,
};

const tools: Anthropic.Tool[] = [
  calculatorTool,
  webFetchTool,
  dbQueryCandidatesTool,
  dbQueryOpportunitiesTool,
];

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
  columns: string[],
  _map: Record<string, z.ZodTypeAny>
) => {
  const c = table === 'candidates' ? CANDIDATE_COLUMN_ENUM : OPPORTUNITY_COLUMN_ENUM;
  return z.object({
    success: z.literal(false),
    table: z.literal(table),
    columns: z.array(c).min(1),
    error_message: z.string().min(5).max(50),
    error_type: z.string().min(1),
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
  craftDbObservationFailureSchema(table, columns, map).parse(observation);
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
  a: z.number(),
  b: z.number(),
  operation: z.enum(['multiply', 'add', 'subtract', 'divide']),
  error_message: z.string().min(5).max(50),
  error_type: z.enum(['invalid_input', 'invalid_calculation', 'runtime_error']),
});

type CalculatorFailure = z.infer<typeof CalculatorFailureSchema>;

const CalculatorObservationSchema = z.discriminatedUnion('success', [
  CalculatorSuccessSchema,
  CalculatorFailureSchema,
]);

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
  error: z.string(),
});

const WebObservationSchema = z.discriminatedUnion('success', [
  WebObservationSuccessSchema,
  WebObservationFailureSchema,
]);

type AgentState = {
  attempts: number;
  userQuery: string;
  messages: Anthropic.MessageParam[];
  done: boolean;
  finalResponse?: string;
};

type ToolCallTrace = {
  input: CalculatorInput | WebFetchInput | DBQueryInput;
  attempt: number;
  toolName: string;
  observation: unknown;
  inputValid: boolean;
  validationError?: string;
};

type GetToolResult =
  | {
      type: 'tool_use';
      blocks: Anthropic.ToolUseBlock[];
    }
  | {
      type: 'text';
      text: string;
    };

async function getToolIntent(state: AgentState): Promise<GetToolResult> {
  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-5',
    max_tokens: 1024,
    tools,
    messages: state.messages,
  });

  const toolBlocks = response.content.filter(
    (block: Anthropic.ContentBlock): block is Anthropic.ToolUseBlock => block.type === 'tool_use'
  );

  if (toolBlocks.length > 0) {
    return {
      type: 'tool_use',
      blocks: toolBlocks,
    };
  }

  const textContent = response.content
    .filter((block: Anthropic.ContentBlock): block is Anthropic.TextBlock => block.type === 'text')
    .map((block: Anthropic.TextBlock) => block.text)
    .join('\n');

  return {
    type: 'text',
    text: textContent,
  };
}

type CalculatorErrorType = 'invalid_input' | 'invalid_calculation' | 'runtime_error';

type CalculatorError = {
  error_type: CalculatorErrorType;
  error_message: string;
};

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
const dispatch = async (
  toolType: ToolType,
  input: CalculatorInput | WebFetchInput | DBQueryInput
) => {
  const execute: Record<ToolType, () => unknown | Promise<unknown>> = {
    calculator: () => {
      const calcInput = input as CalculatorInput;
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

    web_fetch_mock: async () => {
      const webInput = input as WebFetchInput;
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
          error: errorMessage,
        });
      }
    },

    db_query_candidates: () => {
      const dbInput = input as z.infer<typeof CandidatesInputQuerySchema>;

      // Mock database for candidates
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
            type: 'query_error',
          },
          map: CANDIDATES_COLUMNS_TO_ZOD,
        });
      }
    },

    db_query_opportunities: () => {
      const dbInput = input as z.infer<typeof OpportunitiesInputQuerySchema>;

      // Mock database for opportunities
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
        // Filter by where clause
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
            type: 'query_error',
          },
          map: OPPORTUNITIES_COLUMNS_TO_ZOD,
        });
      }
    },
  };

  return execute[toolType]();
};

const getFormattedToolOrAnswerToUserInput = async (userQuery: AgentState['userQuery']) => {
  const state: AgentState = {
    userQuery,
    attempts: 0,
    messages: [{ role: 'user', content: userQuery }],
    done: false,
  };
  const MAX = 3;

  while (state.attempts < MAX) {
    const response = await getToolIntent(state);
    if (response.type === 'text') {
      return response.text;
    }
    state.messages.push({
      role: 'assistant',
      content: response.blocks,
    });

    const block = [...response.blocks].reverse().find((b) => b.type === 'tool_use');
    if (!block) return 'No tool_use block found';
    const toolName = block.name as ToolType;
    const schema = TOOL_INPUT_SCHEMAS[toolName];
    if (!schema) {
      const observation = {
        success: false,
        tool: toolName,
        error_type: 'invalid_input',
        error_message: 'Unknown tool',
      };

      state.messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: block.id,
            content: JSON.stringify(observation),
          },
        ],
      });

      state.attempts++;
      continue;
    }

    const parsed = schema.safeParse(block.input);

    if (!parsed.success) {
      const observation = {
        success: false,
        tool: toolName,
        error_type: 'invalid_input',
        error_message: parsed.error.issues[0]?.message.slice(0, 50) ?? 'Invalid input',
      };

      state.messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: block.id,
            content: JSON.stringify(observation),
          },
        ],
      });

      state.attempts++;
      continue;
    }
    const observation = await dispatch(
      block.name as ToolType,
      block.input as CalculatorInput | WebFetchInput | DBQueryInput
    );
    console.log('observation', observation);

    // Push tool_result back to messages
    state.messages.push({
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: block.id,
          content: JSON.stringify(observation),
        },
      ],
    });

    state.attempts++;
  }

  return 'Max attempts reached';
};

// Main execution
async function main() {
  console.log('Running agent loop...\n');

  const testInputs = [
    'Hello, how are you?',
    'What is 5 times 5?',
    'what is 5 times a',
    'what is 5 divided by 0',
    'Fetch the content of https://example.com',
    'Fetch https://google.com',
    'Find the candidate named Alice Johnson',
    'Look up Sarah Connor in candidates',
  ];

  for (const input of testInputs) {
    console.log(`Input: "${input}"`);
    const result = await getFormattedToolOrAnswerToUserInput(input);
    console.log('Result:', JSON.stringify(result, null, 2));
    console.log('---\n');
  }
}

main().catch(console.error);
