import { describe, it, expect } from 'vitest';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import { verifyAndExecuteTool } from './main';
import type { AgentState } from './main';

describe('main', () => {
  const bootstrapState = ({
    messages,
    userQuery,
    tool_calls,
    step,
    response,
    maxStep,
    trace,
  }: AgentState | undefined) => {
    return {
      messages: messages ?? [new HumanMessage('What is 5 times 5?')],
      userQuery: userQuery ?? 'What is 5 times 5?',
      tool_calls: tool_calls ?? [],
      step: step ?? 1,
      response,
      maxStep: maxStep ?? 3,
      trace: trace ?? [],
    };
  };

  describe('verifyAndExecuteTool', () => {
    it('Failure: should have unknown tool call error if unknown tool name found in tool definition', async () => {
      const mockState = bootstrapState({
        tool_calls: [{ id: '1', name: 'does_not_exist', args: {} }],
      });
      const result = await verifyAndExecuteTool(mockState);
      const messageContent = JSON.parse(result.messages[0].content);
      expect(messageContent.success).toBe(false);
      expect(messageContent.error_type).toBe('invalid_input');
      expect(messageContent.error_message).toBe('Unknown tool call: does_not_exist');
    });

    it('Failure: should have invalid_schema error if schema arguments are not valid', async () => {
      const mockState = bootstrapState({
        tool_calls: [
          { id: '1', name: 'calculator', args: { a: '5', b: 5, operation: 'multiply' } },
        ],
      });
      const result = await verifyAndExecuteTool(mockState);
      const messageContent = JSON.parse(result.messages[0].content);
      expect(messageContent.success).toBe(false);
      expect(messageContent.error_type).toBe('invalid_schema');
    });

    it('Failure: should have runtime_error tool fails', async () => {
      const mockState = bootstrapState({
        tool_calls: [
          { id: '1', name: 'calculator', args: { a: 9999, b: 5, operation: 'multiply' } },
        ],
      });
      const result = await verifyAndExecuteTool(mockState);
      const messageContent = JSON.parse(result.messages[0].content);
      expect(messageContent.success).toBe(false);
      expect(messageContent.error_type).toBe('runtime_error');
    });
  });
});
