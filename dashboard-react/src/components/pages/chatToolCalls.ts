/** A complete function call or one streamed Chat Completions fragment. */
export interface StreamToolCall {
  id?: string;
  index?: number;
  type?: string;
  function?: { name?: string; arguments?: string };
}

/** Merge bounded function-call fragments before any client-side tool executes. */
export function mergeToolCallDeltas(calls: StreamToolCall[], deltas: StreamToolCall[]): void {
  for (const delta of deltas) {
    const index = delta.index;
    if (index !== undefined && (!Number.isInteger(index) || index < 0 || index >= 32)) throw new Error('Invalid function-call index');
    let existing = calls.find(call => index !== undefined ? call.index === index : delta.id !== undefined && call.id === delta.id);
    if (!existing) {
      if (calls.length >= 32) throw new Error('Too many function calls');
      existing = { index, function: { name: '', arguments: '' } };
      calls.push(existing);
    }
    if (delta.id) {
      if (existing.id && existing.id !== delta.id) throw new Error('Function-call identity changed');
      existing.id = delta.id;
    }
    if (delta.type) existing.type = delta.type;
    existing.function ??= { name: '', arguments: '' };
    existing.function.name = (existing.function.name ?? '') + (delta.function?.name ?? '');
    existing.function.arguments = (existing.function.arguments ?? '') + (delta.function?.arguments ?? '');
    if (existing.function.name.length + existing.function.arguments.length > 131072) throw new Error('Function call exceeded its input bound');
  }
}
