import { configureStore, type Middleware } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import type { Conversation } from '../../types/chat';
import { chatActions, chatSliceReducer, subscribeChatPersistence, synchronizeChatPersistence, type ChatRootState } from '../../store/slices/chatSlice';
import { chatStorageChanges, flushChatStorage, hasUnsavedChatChanges, readCommittedChatStorage, writeChatStorage } from '../../store/chatCoordination';
import { readRetainedChatEdits, retainChatEdits } from '../../store/chatRetainedEdits';
import { isolateChatStorage } from '../../store/chatStorageTestScope';
import { ChatRetainedEdits } from './ChatRetainedEdits';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }));

let restoreScope: () => void, unsubscribe: () => void;
let root: Root | undefined, container: HTMLDivElement;
let observeAction: ((action: unknown) => void) | undefined;
const observeDispatch: Middleware<{}, ChatRootState> = () => next => action => {
  observeAction?.(action);
  return next(action);
};
function conversation(content: string): Conversation {
  return { id: 'a', name: 'A conversation', modelId: 'fixture', createdAt: 1, updatedAt: 1,
    messages: [{ id: 'm', role: 'user', content, timestamp: 1 }] };
}
beforeEach(() => { restoreScope = isolateChatStorage('recovery-ui:' + crypto.randomUUID()); });
afterEach(async () => {
  await act(async () => { await flushChatStorage().catch(() => {}); await synchronizeChatPersistence(); root?.unmount(); });
  unsubscribe?.(); container?.remove(); vi.restoreAllMocks(); restoreScope(); root = undefined;
});

async function renderRecovery() {
  const before = await readCommittedChatStorage();
  const canonical = conversation('Saved in another tab');
  const state = { ...before, conversations: { a: canonical } };
  await writeChatStorage(chatStorageChanges(before, state));
  await retainChatEdits(new Map([['a', { ...conversation('My local edit') }]]));
  const store = configureStore({ reducer: { chat: chatSliceReducer }, middleware: defaults => defaults().concat(observeDispatch) });
  store.dispatch(chatActions.refreshDurable({ conversations: { a: canonical } }));
  store.dispatch(chatActions.selectConversation('a'));
  unsubscribe = subscribeChatPersistence(store);
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
  await act(async () => {
    await synchronizeChatPersistence();
    root!.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ChatRetainedEdits disabled={false} /></ThemeProvider></Provider>);
  });
  return store;
}

async function click(label: string): Promise<void> {
  const button = [...container.querySelectorAll('button')].find(button => button.textContent === label);
  if (!button) throw new Error('Missing recovery control: ' + label);
  await act(async () => { await userEvent.click(button); });
}

async function settled(): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (!(await readRetainedChatEdits()).length && !container.textContent?.includes('Retained local edits')) return;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
  }
  throw new Error('Recovery did not settle');
}

it('reviews both histories and saves a new conversation through the real storage boundary', async () => {
  const store = await renderRecovery();
  expect(container.textContent).toContain('Retained local edits');
  await click('Review versions');
  expect(container.textContent).toContain('My local edit');
  expect(container.textContent).toContain('Saved in another tab');
  await click('Save as separate conversation');
  await settled();
  const conversations = Object.values(store.getState().chat.conversations);
  expect(conversations).toHaveLength(2);
  const fork = conversations.find(conversation => conversation.id !== 'a')!;
  expect(fork.messages[0].content).toBe('My local edit');
  expect(fork.continuation).toBeUndefined();
  expect(store.getState().chat.activeConversationId).toBe(fork.id);
  expect(store.getState().chat.conversations.a.messages[0].content).toBe('Saved in another tab');
});

it('uses the saved version only after the explicit adoption action', async () => {
  const store = await renderRecovery();
  await click('Use saved version');
  await settled();
  expect(Object.keys(store.getState().chat.conversations)).toEqual(['a']);
  expect(store.getState().chat.conversations.a.messages[0].content).toBe('Saved in another tab');
});

it('preserves another conversation’s failed-save warning while adopting retained history', async () => {
  const store = await renderRecovery();
  const unsafeClears: unknown[] = [];
  observeAction = action => {
    if (chatActions.setPersistenceError.match(action) && action.payload === undefined && hasUnsavedChatChanges()) unsafeClears.push(action);
  };
  const put = IDBObjectStore.prototype.put;
  const quota = vi.spyOn(IDBObjectStore.prototype, 'put').mockImplementation(function (this: IDBObjectStore, value, key) {
    if (this.name === 'snapshot' && key === 'current') throw new DOMException('Synthetic unrelated save failure', 'QuotaExceededError');
    return put.call(this, value, key);
  });
  try {
    await act(async () => {
      store.dispatch(chatActions.newConversation('another-model'));
      await flushChatStorage().catch(() => {});
    });
    const pending = store.getState().chat.activeConversationId!;
    expect(hasUnsavedChatChanges()).toBe(true);
    expect(store.getState().chat.persistenceError).toBeTruthy();
    await click('Use saved version');
    for (let attempt = 0; attempt < 100; attempt++) {
      const button = [...container.querySelectorAll('button')].find(button => button.textContent === 'Use saved version');
      if (!button || !button.disabled) break;
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
    }
    expect(await readRetainedChatEdits()).toEqual([]);
    await act(async () => { await synchronizeChatPersistence(); });
    expect(hasUnsavedChatChanges()).toBe(true);
    expect(unsafeClears).toEqual([]);
    expect(container.textContent).not.toContain('Retained local edits');
    expect(store.getState().chat.persistenceError).toBeTruthy();
    expect((await readCommittedChatStorage()).conversations).not.toHaveProperty(pending);
  } finally {
    quota.mockRestore();
    await act(async () => { await synchronizeChatPersistence(); });
    observeAction = undefined;
  }
  expect(store.getState().chat.persistenceError).toBeUndefined();
  expect(hasUnsavedChatChanges()).toBe(false);
});
