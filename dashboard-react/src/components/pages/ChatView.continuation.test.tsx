import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
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
  await act(async () => { await userEvent.click(button); });
}

async function renderAndEnroll(): Promise<string> {
  store.dispatch(chatActions.selectModel(model));
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
  await act(async () => { root!.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ChatView readyInstances={[]} catalog={{ loaded: true, models: { [model]: { id: model, name: 'Independent code', inference_capability: {
    contract: 'chat.completions@0.2.0', processing_domain: 'external', usage: 'unknown', usage_label: 'Synthetic fixture', usage_url: null, expires_at: Date.now() / 1000 + 300,
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
  for (const id of Object.keys(store.getState().chat.conversations)) store.dispatch(chatActions.deleteConversation(id));
  vi.unstubAllGlobals();
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
      const persisted = JSON.parse(localStorage.getItem('skulk-chat')!) as { state: { conversations: Record<string, { continuation: { phase: string; request_nonce: string } }> } };
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

  it('fences a missing terminal, prevents replay, and refuses recovery with missing assistant text', async () => {
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
    await send('Do not replay');
    await settle(() => container!.textContent!.includes('previous continuation attempt is uncertain'));
    expect(calls).toBe(1);
    await click('Recover receipt');
    await settle(() => container!.textContent!.includes('committed public history is missing or changed'));
    expect(store.getState().chat.conversations[store.getState().chat.activeConversationId!].continuation?.phase).toBe('uncertain');
    expect(calls).toBe(1);
  });
});
