import { describe, expect, it } from 'vitest';
import { mergeToolCallDeltas, type StreamToolCall } from './chatToolCalls';

describe('streamed capability tool calls', () => {
  it('assembles interleaved fragments into one executable call per index', () => {
    const calls: StreamToolCall[] = [];
    mergeToolCallDeltas(calls, [{ index: 0, id: 'first', type: 'function', function: { name: 'web_search', arguments: '{"que' } }, { index: 1, id: 'second', function: { name: 'read', arguments: '{' } }]);
    mergeToolCallDeltas(calls, [{ index: 1, function: { arguments: '}' } }, { index: 0, function: { arguments: 'ry":"fox"}' } }]);
    expect(calls).toHaveLength(2);
    expect(calls[0].function).toEqual({ name: 'web_search', arguments: '{"query":"fox"}' });
    expect(calls[1].function?.arguments).toBe('{}');
  });
  it('preserves native complete calls without indices', () => {
    const calls: StreamToolCall[] = [];
    mergeToolCallDeltas(calls, [{ id: 'native', function: { name: 'web_search', arguments: '{}' } }]);
    expect(calls[0].id).toBe('native');
  });
  it('refuses changed identities and unbounded indices before execution', () => {
    const calls: StreamToolCall[] = [];
    mergeToolCallDeltas(calls, [{ index: 0, id: 'first' }]);
    expect(() => mergeToolCallDeltas(calls, [{ index: 0, id: 'other' }])).toThrow('identity changed');
    expect(() => mergeToolCallDeltas(calls, [{ index: 32, id: 'outside' }])).toThrow('index');
  });
});
