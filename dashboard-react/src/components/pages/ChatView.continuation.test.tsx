import { flushChatStorage, readCommittedChatStorage } from '../../store/chatCoordination';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { store } from '../../store';
import { chatActions, chatSliceReducer } from '../../store/slices/chatSlice';
import { canonicalContinuationJSON, publicHistoryFingerprint } from '../../store/chatContinuation';
import { darkTheme } from '../../theme/theme';
import { ChatView } from './ChatView';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => {
  const t = (_key: string, fallback: string, parameters?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (match, key: string) => parameters?.[key] === undefined ? match : String(parameters[key]));
  return { tolgee: { getLanguage: () => 'en' }, useSkulkTranslation: () => ({ t }) };
});
vi.mock('../../store/endpoints/steward', () => ({ useGetStewardStatusQuery: () => ({ data: { enabled: false } }) }));

const target = { target_node: 'fixture.owner', callable_id: 'independent.continuation', descriptor_revision: 'a'.repeat(16), model: 'code' };
const model = 'capability/independent.chat/code';
let root: Root | null = null, container: HTMLDivElement | null = null;

async function settle(check: () => boolean): Promise<void> {
  for (let index = 0; index < 500; index++) {
    if (check()) return;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
  }
  throw new Error('Continuation interaction did not settle.');
}

async function click(text: string): Promise<void> {
  const button = [...container!.querySelectorAll('button')].find(item => item.textContent === text);
  if (!button) throw new Error(`Missing control: ${text}`);
  await settle(() => !button.disabled);
  await act(async () => { await userEvent.click(button); });
}

async function renderAndEnroll(features: Array<'image_input' | 'function_tools' | 'reasoning_toggle' | 'text_file_input' | 'pdf_input'> = []): Promise<string> {
  store.dispatch(chatActions.selectModel(model));
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
  await act(async () => { root!.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ChatView readyInstances={[]} catalog={{ loaded: true, models: { [model]: { id: model, name: 'Independent code', inference_capability: {
    contract: 'chat.completions@0.2.0', features, processing_domain: 'external', usage: 'unknown', usage_label: 'Synthetic fixture', usage_url: null, expires_at: Date.now() / 1000 + 300,
    continuation: { contract: 'chat.continuation@0.1.0', authorization: 'caller.ed25519@0.1.0', callable_version: '0.1.0', proof_header: 'Skulk-Continuation-Proof', receipt_path: '/v1/chat/continuation/receipt', target },
  } } } }} /></ThemeProvider></Provider>); });
  await act(async () => { await userEvent.click(container!.querySelector('summary')!); });
  await click('Get browser public key');
  await settle(() => !!container!.querySelector<HTMLInputElement>('[aria-label="Browser public key"]')?.value);
  const publicKey = container!.querySelector<HTMLInputElement>('[aria-label="Browser public key"]')!.value;
  const enrollment = { client: { public_key: publicKey, caller_id: 'b'.repeat(32), models: ['code'], expires_at: Date.now() / 1000 + 3600 }, target: { ...target, model: 'unselected' } };
  await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Public enrollment JSON"]')!, JSON.stringify(enrollment)); });
  await click('Enable for this conversation');
  await settle(() => !!store.getState().chat.conversations[store.getState().chat.activeConversationId!]?.continuation);
  return publicKey;
}

async function send(text: string): Promise<void> {
  await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!, text); });
  await settle(() => !container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled);
  await act(async () => { await userEvent.click(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!); });
}

afterEach(async () => {
  await act(async () => { root?.unmount(); }); root = null;
  container?.remove(); container = null;
  for (const id of Object.keys(store.getState().chat.conversations)) await navigator.locks.request('skulk-chat-conversation-v2:' + id, () => {});
  for (const id of Object.keys(store.getState().chat.conversations)) store.dispatch(chatActions.deleteConversation(id));
  await act(async () => { await flushChatStorage().catch(() => {}); });
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Chat client continuation', () => {
  it('owner-enrolls a browser key and preserves exact whitespace/summary through a follow-up', async () => {
    let publicKey = '';
    const requests: Array<{ messages: Record<string, unknown>[]; proof: { request: Record<string, unknown> } }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      const header = new Headers(init?.headers).get('Skulk-Continuation-Proof');
      if (!header) throw new Error('Client proof missing');
      const proof = JSON.parse(atob(header.replaceAll('-', '+').replaceAll('_', '/'))) as { caller_id: string; request: Record<string, unknown>; signature: string };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      const clientKey = await crypto.subtle.importKey('raw', Uint8Array.from(publicKey.match(/../g)!, pair => parseInt(pair, 16)), 'Ed25519', false, ['verify']);
      const signature = Uint8Array.from(proof.signature.match(/../g)!, pair => parseInt(pair, 16));
      expect(await crypto.subtle.verify('Ed25519', clientKey, signature, new TextEncoder().encode(canonicalContinuationJSON({ caller_id: proof.caller_id, request: proof.request }, true)))).toBe(true);
      const persisted = JSON.parse(localStorage.getItem('skulk-chat-v2')!) as { state: { conversations: Record<string, { continuation: { phase: string; request_nonce: string } }> } };
      const saved = persisted.state.conversations[store.getState().chat.activeConversationId!].continuation;
      expect(saved.phase).toBe('pending'); expect(saved.request_nonce).toBe(proof.request.nonce);
      requests.push({ ...body, proof });
      const assistant = { role: 'assistant', content: '  hello 🦊\n```python\nprint(1)\n```\n', reasoning_content: 'brief summary' };
      const receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + String(requests.length).repeat(64), history: await publicHistoryFingerprint([...body.messages, assistant]), expires_at: Date.now() / 1000 + 300 };
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: assistant.content, reasoning_content: assistant.reasoning_content } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], skulk_continuation: receipt })}\n\ndata: [DONE]\n\n`, { headers: { 'Content-Type': 'text/event-stream' } });
    }));
    publicKey = await renderAndEnroll();
    await send('First');
    await settle(() => store.getState().chat.conversations[store.getState().chat.activeConversationId!].messages.some(message => message.role === 'assistant'));
    expect(store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.phase).toBe('ready');
    await send('Next');
    await settle(() => requests.length === 2 && store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.receipt?.handle === 'ct1_' + '2'.repeat(64));
    expect(requests[0].proof.request.mode).toBe('start');
    expect(requests[1].proof.request.mode).toBe('continue');
    expect(requests[1].messages[1]).toEqual({ role: 'assistant', content: '  hello 🦊\n```python\nprint(1)\n```\n', reasoning_content: 'brief summary' });
    expect(container!.querySelector('code')?.textContent).toContain('print(1)');
  });

  it.each(['finish', 'done', 'read error'])('recovers received public output after a lost %s without replay', async loss => {
    let calls = 0, receipt: Record<string, unknown> | undefined;
    const assistant = { role: 'assistant', content: '  complete answer 🦊\n', reasoning_content: 'exact summary' };
    const requests: Record<string, unknown>[][] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/chat/continuation/receipt') return new Response(JSON.stringify({ receipt, error: null }));
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const header = new Headers(init?.headers).get('Skulk-Continuation-Proof')!;
      const proof = JSON.parse(atob(header.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      requests.push(body.messages);
      receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + String(calls).repeat(64), history: await publicHistoryFingerprint([...body.messages, assistant]), expires_at: Date.now() / 1000 + 300 };
      const delta = `data: ${JSON.stringify({ choices: [{ delta: { content: assistant.content, reasoning_content: assistant.reasoning_content } }] })}\n\n`;
      const finish = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], skulk_continuation: receipt })}\n\n`;
      if (calls > 1) return new Response(delta + finish + 'data: [DONE]\n\n');
      if (loss === 'read error') return new Response(new ReadableStream({
        start(controller) { controller.enqueue(new TextEncoder().encode(delta)); },
        pull(controller) { controller.error(new TypeError('Synthetic connection interrupted')); },
      }));
      return new Response(delta + (loss === 'done' ? finish : ''));
    }));
    await renderAndEnroll(); await send('First');
    await settle(() => {
      const conversation = store.getState().chat.conversations[store.getState().chat.activeConversationId!];
      return conversation.continuation?.phase === 'uncertain' && conversation.messages.length === 2;
    });
    const conversation = store.getState().chat.conversations[store.getState().chat.activeConversationId!];
    expect(conversation.messages[1].continuationMessages).toEqual([assistant]);
    expect(conversation.messages[1].content).toBe('complete answer 🦊');
    await click('Recover receipt');
    await settle(() => store.getState().chat.conversations[conversation.id].continuation?.phase === 'ready');
    expect(calls).toBe(1);
    await send('Next');
    await settle(() => store.getState().chat.conversations[conversation.id].continuation?.receipt?.handle === 'ct1_' + '2'.repeat(64));
    expect(calls).toBe(2); expect(requests[1][1]).toEqual(assistant);
  });

  it('retains a received tool request for recovery without executing it', async () => {
    let calls = 0, receipt: Record<string, unknown> | undefined;
    const call = { id: 'call-1', type: 'function', function: { name: 'web_search', arguments: '{"query":"synthetic"}' } };
    const assistant = { role: 'assistant', content: null, tool_calls: [call] };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/chat/continuation/receipt') return new Response(JSON.stringify({ receipt, error: null }));
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const header = new Headers(init?.headers).get('Skulk-Continuation-Proof')!;
      const proof = JSON.parse(atob(header.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + 'f'.repeat(64), history: await publicHistoryFingerprint([...body.messages, assistant]), expires_at: Date.now() / 1000 + 300 };
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { tool_calls: [{ index: 0, ...call }] } }] })}\n\n`);
    }));
    await renderAndEnroll(); await send('First');
    await settle(() => {
      const conversation = store.getState().chat.conversations[store.getState().chat.activeConversationId!];
      return conversation.continuation?.phase === 'uncertain' && conversation.messages.length === 2;
    });
    const conversation = store.getState().chat.conversations[store.getState().chat.activeConversationId!];
    expect(conversation.messages[1].continuationMessages).toEqual([assistant]);
    await click('Recover receipt');
    await settle(() => store.getState().chat.conversations[conversation.id].continuation?.phase === 'ready');
    expect(calls).toBe(1);
  });

  it('keeps a newer pending nonce when an older receipt lookup finishes', async () => {
    let calls = 0, firstReceipt: Record<string, unknown> | undefined, secondResponse: Response | undefined;
    let resolveLookup!: (response: Response) => void, resolveTurn!: (response: Response) => void;
    const lookup = new Promise<Response>(resolve => { resolveLookup = resolve; });
    const turn = new Promise<Response>(resolve => { resolveTurn = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/chat/continuation/receipt') return lookup;
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const proof = JSON.parse(atob(new Headers(init?.headers).get('Skulk-Continuation-Proof')!.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      const assistant = { role: 'assistant', content: 'answer' };
      const receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + String(calls).repeat(64), history: await publicHistoryFingerprint([...body.messages, assistant]), expires_at: Date.now() / 1000 + 300 };
      const response = new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: assistant.content } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], skulk_continuation: receipt })}\n\ndata: [DONE]\n\n`);
      if (calls === 1) { firstReceipt = receipt; return response; }
      secondResponse = response; return turn;
    }));
    await renderAndEnroll(); await send('First');
    await settle(() => store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.phase === 'ready');
    await click('Recover receipt'); await send('Next');
    await settle(() => !!secondResponse);
    const id = store.getState().chat.activeConversationId!, pending = store.getState().chat.conversations[id].continuation;
    expect(pending?.phase).toBe('pending');
    resolveLookup(new Response(JSON.stringify({ receipt: firstReceipt, error: null })));
    await settle(() => container!.textContent!.includes('changed during receipt lookup'));
    expect(store.getState().chat.conversations[id].continuation).toEqual(pending);
    resolveTurn(secondResponse!);
    await settle(() => store.getState().chat.conversations[id].continuation?.receipt?.handle === 'ct1_' + '2'.repeat(64));
    expect(calls).toBe(2);
  });

  it.each([false, true])('persists verified tool rounds; stop during tool = %s', async stopDuringTool => {
    let calls = 0, toolStarted = false, secondStarted = false;
    let resolveTool!: (response: Response) => void, resolveTurn!: (response: Response) => void;
    const toolResult = new Promise<Response>(resolve => { resolveTool = resolve; });
    const nextTurn = new Promise<Response>(resolve => { resolveTurn = resolve; });
    let secondResponse: Response | undefined;
    const call = { id: 'call-1', type: 'function', function: { name: 'web_search', arguments: '{"query":"synthetic"}' } };
    const assistant = { role: 'assistant', content: null, tool_calls: [call] };
    const durable = () => {
      const saved = JSON.parse(localStorage.getItem('skulk-chat-v2')!) as { state: { conversations: Record<string, { continuation: { phase: string; receipt: unknown }; messages: Array<{ role: string; continuationMessages?: Record<string, unknown>[] }> }> } };
      return saved.state.conversations[store.getState().chat.activeConversationId!];
    };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/tools/web_search') {
        expect(durable().continuation.phase).toBe('ready');
        expect(durable().messages[1].continuationMessages).toEqual([assistant]);
        toolStarted = true; return toolResult;
      }
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const proof = JSON.parse(atob(new Headers(init?.headers).get('Skulk-Continuation-Proof')!.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      const message = calls === 1 ? assistant : { role: 'assistant', content: 'finished' };
      const receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + String(calls).repeat(64), history: await publicHistoryFingerprint([...body.messages, message]), expires_at: Date.now() / 1000 + 300 };
      const delta = calls === 1 ? { tool_calls: [{ index: 0, ...call }] } : { content: 'finished' };
      const response = new Response(`data: ${JSON.stringify({ choices: [{ delta }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: calls === 1 ? 'tool_calls' : 'stop' }], skulk_continuation: receipt })}\n\ndata: [DONE]\n\n`);
      if (calls === 1) return response;
      expect(durable().continuation.phase).toBe('pending');
      expect(durable().messages[1].continuationMessages).toEqual(body.messages.slice(1));
      expect(body.messages[2].role).toBe('tool');
      secondResponse = response; secondStarted = true; return nextTurn;
    }));
    await renderAndEnroll(['function_tools']); await send('First');
    await settle(() => toolStarted);
    expect(durable().continuation.receipt).toBeTruthy();
    const submitted = store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.request_nonce;
    if (stopDuringTool) await act(async () => { await userEvent.click(container!.querySelector<HTMLButtonElement>('[aria-label="Cancel generation"]')!); });
    resolveTool(new Response(JSON.stringify({ query: 'synthetic', results: [] })));
    if (stopDuringTool) {
      await settle(() => !!container!.querySelector('[aria-label="Send message"]'));
      expect(calls).toBe(1);
      expect(store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.request_nonce).toBe(submitted);
      expect(durable().continuation.phase).toBe('ready');
      return;
    }
    await settle(() => secondStarted);
    resolveTurn(secondResponse!);
    await settle(() => store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.receipt?.handle === 'ct1_' + '2'.repeat(64));
    expect(durable().messages).toHaveLength(2);
    expect(durable().messages[1].continuationMessages).toHaveLength(3);
    expect(calls).toBe(2);
  });

  it('keeps receipt and history uncertain when their atomic storage write fails', async () => {
    let calls = 0, receipt: Record<string, unknown> | undefined;
    const assistant = { role: 'assistant', content: 'received answer' };
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/chat/continuation/receipt') return new Response(JSON.stringify({ receipt, error: null }));
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const proof = JSON.parse(atob(new Headers(init?.headers).get('Skulk-Continuation-Proof')!.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + 'e'.repeat(64), history: await publicHistoryFingerprint([...body.messages, assistant]), expires_at: Date.now() / 1000 + 300 };
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: assistant.content } }] })}\n\ndata: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }], skulk_continuation: receipt })}\n\ndata: [DONE]\n\n`);
    }));
    await renderAndEnroll();
    const original = IDBObjectStore.prototype.put;
    const write = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
      if (this.name === 'snapshot') {
        const parsed = value as { state: { conversations: Record<string, { continuation?: { phase: string }; messages: Array<{ role: string }> }> } };
        if (Object.values(parsed.state.conversations).some(conversation => conversation.continuation?.phase === 'ready' && conversation.messages.some(message => message.role === 'assistant'))) throw new DOMException('Synthetic full storage', 'QuotaExceededError');
      }
      return original.call(this, value, key!);
    });
    await send('First');
    await settle(() => {
      const conversation = store.getState().chat.conversations[store.getState().chat.activeConversationId!];
      return conversation.continuation?.phase === 'uncertain' && conversation.messages.length === 2;
    });
    const id = store.getState().chat.activeConversationId!;
    expect(store.getState().chat.conversations[id].messages[1].continuationMessages).toEqual([assistant]);
    write.mockRestore(); await click('Recover receipt');
    await settle(() => store.getState().chat.conversations[id].continuation?.phase === 'ready');
    expect(calls).toBe(1);
  });

  it('restores the preceding attempt when Stop occurs during pending storage before HTTP dispatch', async () => {
    let calls = 0, canceled = false;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      if (String(input) === '/v1/chat/completions') calls++;
      return new Response(JSON.stringify({ enabled: false }));
    }));
    await renderAndEnroll();
    const original = IDBObjectStore.prototype.put;
    const storage = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
      const request = original.call(this, value, key!);
      const snapshot = value as { state?: { conversations?: Record<string, { continuation?: { phase: string } }> } };
      if (!canceled && this.name === 'snapshot' && Object.values(snapshot.state?.conversations ?? {}).some(conversation => conversation.continuation?.phase === 'pending')) queueMicrotask(() => {
        const stop = container!.querySelector<HTMLButtonElement>('[aria-label="Cancel generation"]');
        if (stop) { canceled = true; stop.click(); }
      });
      return request;
    });
    await send('Stop before submission');
    await settle(() => canceled && !!container!.querySelector('[aria-label="Send message"]'));
    storage.mockRestore();
    const id = store.getState().chat.activeConversationId!;
    expect(calls).toBe(0);
    expect(store.getState().chat.conversations[id].continuation?.phase).toBe('ready');
    expect(store.getState().chat.conversations[id].continuation?.request_nonce).toBeUndefined();
    const canonical = (await readCommittedChatStorage()).conversations as Record<string, { continuation: { phase: string; request_nonce?: string } }>;
    expect(canonical[id].continuation.phase).toBe('ready');
    expect(canonical[id].continuation.request_nonce).toBeUndefined();
  });

  it.each(['reassigned', 'removed'])('requires an explicit selection after the active conversation is %s elsewhere', async change => {
    await renderAndEnroll();
    const chat = store.getState().chat, id = chat.activeConversationId!;
    const conversations = change === 'removed' ? {} : { [id]: { ...chat.conversations[id], modelId: 'independent/other-model' } };
    const adopted = chatSliceReducer(chat, chatActions.refreshDurable({ conversations }));
    expect(adopted.activeConversationId).toBeNull();
    expect(adopted.selectedModelId).toBe(model);
    expect(adopted.selectionNeedsReview).toBe(true);
    const chosen = chatSliceReducer(adopted, chatActions.newConversation(model));
    expect(chosen.selectionNeedsReview).toBe(false);
    expect(chosen.conversations[chosen.activeConversationId!].modelId).toBe(model);
  });

  it('retains a draft after foreign deletion and explicitly starts a new context with its selected model', async () => {
    await renderAndEnroll();
    const id = store.getState().chat.activeConversationId!;
    await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!, 'Keep this private draft.'); });
    await act(async () => { store.dispatch(chatActions.refreshDurable({ conversations: {} })); });
    await settle(() => container!.textContent!.includes('removed or changed to another model'));
    expect(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!.value).toBe('Keep this private draft.');
    expect(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled).toBe(true);
    await click('Start a new conversation');
    await settle(() => !!store.getState().chat.activeConversationId);
    expect(store.getState().chat.activeConversationId).not.toBe(id);
    expect(store.getState().chat.selectedModelId).toBe(model);
    expect(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!.value).toBe('Keep this private draft.');
  });

  it('creates a separate conversation instead of reassigning an enrolled empty context', async () => {
    await renderAndEnroll();
    const chat = store.getState().chat, id = chat.activeConversationId!;
    const next = chatSliceReducer(chat, chatActions.selectModel('independent/other-model'));
    expect(next.activeConversationId).not.toBe(id);
    expect(next.conversations[id].modelId).toBe(model);
    expect(next.conversations[id].continuation).toEqual(chat.conversations[id].continuation);
    expect(next.conversations[next.activeConversationId!].continuation).toBeUndefined();
  });

  it('keeps reset controls when the destination withdraws continuation', async () => {
    await renderAndEnroll();
    const id = store.getState().chat.activeConversationId!;
    await act(async () => { root!.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ChatView readyInstances={[]} catalog={{ loaded: true, models: {} }} /></ThemeProvider></Provider>); });
    const reset = [...container!.querySelectorAll('button')].find(button => button.textContent === 'Use full history instead');
    expect(reset).toBeTruthy();
    await act(async () => { await userEvent.click(reset!); });
    await settle(() => !store.getState().chat.conversations[id].continuation);
    expect(store.getState().chat.conversations[id].continuation).toBeUndefined();
  });

  it('retains partial output but refuses mismatching receipt recovery and replay', async () => {
    let calls = 0, receipt: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input) === '/v1/chat/continuation/receipt') return new Response(JSON.stringify({ receipt, error: null }));
      if (String(input) !== '/v1/chat/completions') return new Response(JSON.stringify({ enabled: false }));
      calls++;
      const header = new Headers(init?.headers).get('Skulk-Continuation-Proof')!;
      const proof = JSON.parse(atob(header.replaceAll('-', '+').replaceAll('_', '/'))) as { request: Record<string, unknown> };
      const body = JSON.parse(String(init?.body)) as { messages: Record<string, unknown>[] };
      receipt = { contract: 'chat.continuation@0.1.0', conversation_id: proof.request.conversation_id, request_nonce: proof.request.nonce, handle: 'ct1_' + 'e'.repeat(64), history: await publicHistoryFingerprint([...body.messages, { role: 'assistant', content: 'lost response' }]), expires_at: Date.now() / 1000 + 300 };
      return new Response('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
    }));
    await renderAndEnroll(); await send('First');
    await settle(() => store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.phase === 'uncertain');
    await settle(() => store.getState().chat.conversations[store.getState().chat.activeConversationId!].messages.length === 2);
    expect(store.getState().chat.conversations[store.getState().chat.activeConversationId!].messages[1].continuationMessages).toEqual([{ role: 'assistant', content: 'partial' }]);
    await send('Do not replay');
    await settle(() => container!.textContent!.includes('Recover this conversation’s receipt'));
    expect(calls).toBe(1);
    await click('Recover receipt');
    await settle(() => container!.textContent!.includes('committed public history is missing or changed'));
    expect(store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.phase).toBe('uncertain');
    expect(calls).toBe(1);
  });
});
