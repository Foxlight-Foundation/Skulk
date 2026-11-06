import { flushChatStorage } from '../../store/chatCoordination';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
import { chatDocumentIds, removeChatDocuments } from '../../store/chatDocuments';
import { darkTheme } from '../../theme/theme';
import { ChatView } from './ChatView';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => {
  const t = (_key: string, fallback: string) => fallback;
  return { tolgee: { getLanguage: () => 'en' }, useSkulkTranslation: () => ({ t }) };
});
vi.mock('../../store/endpoints/steward', () => ({ useGetStewardStatusQuery: () => ({ data: { enabled: false } }) }));

const model = 'capability/provider/chat/document-model';
const textOnlyModel = 'capability/provider/chat/text-only-model';
let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderChat(): Promise<void> {
  store.dispatch(chatActions.selectModel(model));
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ChatView readyInstances={[]} catalog={{ loaded: true, models: {
      [model]: { id: model, name: 'Document model', inference_capability: {
        contract: 'chat.completions@0.2.0', processing_domain: 'external', usage: 'unknown', usage_label: 'Fixture only', usage_url: null,
        expires_at: Date.now() / 1000 + 300, features: ['text_file_input', 'pdf_input'],
      } },
      [textOnlyModel]: { id: textOnlyModel, name: 'Text-only model', inference_capability: {
        contract: 'chat.completions@0.2.0', processing_domain: 'external', usage: 'unknown', usage_label: 'Fixture only', usage_url: null,
        expires_at: Date.now() / 1000 + 300, features: [],
      } },
    } }} /></ThemeProvider></Provider>);
  });
}

async function waitFor(check: () => boolean): Promise<void> {
  for (let i = 0; i < 200; i++) {
    if (check()) return;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
  }
  throw new Error('Document chat interaction did not settle.');
}

async function select(files: File[], prompt: string): Promise<void> {
  const input = container!.querySelector<HTMLInputElement>('[aria-label="Chat attachment file"]')!;
  expect(input.accept).toContain('.txt');
  expect(input.accept).toContain('.pdf');
  await act(async () => { await userEvent.upload(input, files); });
  await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!, prompt); });
  await act(async () => { await userEvent.click(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!); });
}

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  root = null;
  container?.remove(); container = null;
  for (const id of Object.keys(store.getState().chat.conversations)) await navigator.locks.request('skulk-chat-conversation-v2:' + id, () => {});
  const ids = chatDocumentIds(store.getState().chat.conversations);
  Object.keys(store.getState().chat.conversations).forEach(id => store.dispatch(chatActions.deleteConversation(id)));
  await removeChatDocuments(ids);
  await act(async () => { await flushChatStorage().catch(() => {}); });
  vi.restoreAllMocks(); vi.unstubAllGlobals();
});

describe('Chat document submission', () => {
  it('sends large text and PDF files, retains small history metadata and resends exact files on the next turn', async () => {
    const requests: Array<{ messages: Array<{ content: unknown }> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response('data: {"choices":[{"delta":{"content":"MARIGOLD JUNIPER"}}]}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
    }));
    await renderChat();
    const text = 'MARIGOLD\n' + 'a'.repeat(1_100_000);
    await select([new File([text], 'notes.txt', { type: 'text/plain' }), new File(['%PDF-1.4\nJUNIPER'], 'notes.pdf', { type: 'application/pdf' })], 'Read both.');
    await waitFor(() => requests.length === 1 && container!.textContent!.includes('MARIGOLD JUNIPER'));
    const parts = requests[0].messages[0].content as Array<{ type: string; file?: { filename: string; file_data: string } }>;
    expect(parts.map(part => part.type)).toEqual(['file', 'file', 'text']);
    expect(atob(parts[0].file!.file_data.split(',')[1])).toBe(text);
    expect(parts[1].file!.file_data).toBe('data:application/pdf;base64,' + btoa('%PDF-1.4\nJUNIPER'));
    const messages = Object.values(store.getState().chat.conversations).flatMap(conversation => conversation.messages);
    expect(JSON.stringify(messages).length).toBeLessThan(4096);
    expect(messages[0].attachments?.[0].documentId).toBeTypeOf('string');
    await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!, 'And again?'); });
    await waitFor(() => !container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled);
    await act(async () => { await userEvent.click(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!); });
    await waitFor(() => requests.length === 2);
    expect(requests[1].messages[0].content).toEqual(requests[0].messages[0].content);
  });

  it('retains the draft and selected document when browser storage fails, without sending inference', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await renderChat();
    vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(() => { throw new DOMException('Full', 'QuotaExceededError'); });
    await select([new File(['MARIGOLD'], 'notes.txt', { type: 'text/plain' })], 'Keep this draft.');
    await waitFor(() => container!.textContent!.includes('Could not save documents'));
    expect(fetch).not.toHaveBeenCalled();
    expect(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!.value).toBe('Keep this draft.');
    expect(container!.querySelector('[title="notes.txt"]')).not.toBeNull();
  });

  it('retains the draft when durable handle persistence fails and can retry after recovery', async () => {
    const fetch = vi.fn(async () => new Response('data: {"choices":[{"delta":{"content":"Saved"}}]}\n\ndata: [DONE]\n\n', { headers: { 'Content-Type': 'text/event-stream' } }));
    vi.stubGlobal('fetch', fetch);
    await renderChat();
    const original = IDBObjectStore.prototype.put;
    const failure = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value: unknown, key?: IDBValidKey) {
      if (this.name === 'snapshot') throw new DOMException('Full', 'QuotaExceededError');
      return original.call(this, value, key!);
    });
    await select([new File(['JUNIPER'], 'notes.txt', { type: 'text/plain' })], 'Keep this document.');
    await waitFor(() => container!.textContent!.includes('Could not save conversation history'));
    await waitFor(() => !container!.textContent!.includes('Preparing attachments…'));
    expect(fetch).not.toHaveBeenCalled();
    expect(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!.value).toBe('Keep this document.');
    expect(container!.querySelector('[title="notes.txt"]')).not.toBeNull();
    expect(Object.values(store.getState().chat.conversations).flatMap(conversation => conversation.messages)).toHaveLength(0);
    failure.mockRestore();
    await waitFor(() => !container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled);
    await act(async () => { await userEvent.click(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!); });
    await waitFor(() => fetch.mock.calls.length === 1);
    expect(window.localStorage.getItem('skulk-chat-v2')).toContain('documentId');
  });

  it('keeps selected files visible when switching to an unsupported model and back', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await renderChat();
    await act(async () => { await userEvent.upload(container!.querySelector<HTMLInputElement>('[aria-label="Chat attachment file"]')!, [new File(['JUNIPER'], 'notes.txt', { type: 'text/plain' })]); });
    await act(async () => { await userEvent.fill(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!, 'Keep this draft.'); });
    await act(async () => { store.dispatch(chatActions.selectModel(textOnlyModel)); });
    await waitFor(() => container!.textContent!.includes('does not support these attachments'));
    expect(container!.querySelector('[title="notes.txt"]')).not.toBeNull();
    expect(container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled).toBe(true);
    expect(container!.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')!.value).toBe('Keep this draft.');
    await act(async () => { store.dispatch(chatActions.selectModel(model)); });
    await waitFor(() => !container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!.disabled);
    expect(container!.querySelector('[title="notes.txt"]')).not.toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });
});
